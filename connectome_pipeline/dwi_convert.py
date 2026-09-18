#!/usr/bin/env python3
"""
ADNI DWI / derived-map conversion helpers for Step 1.

This module keeps notebook logic thin while preserving clear notebook-facing
configuration. It supports:
- series discovery
- raw DWI conversion with gradient sidecars
- derived scalar conversion
- parallel batch execution
- compact example logging / TSV export
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


DEFAULT_AXIAL_DIR_NAME = "Axial_DTI"
DEFAULT_DERIVED_DIR_NAMES = {
    "EPI_corrected_image",
    "EPI_current_corrected_image",
    "Eddy_current_corrected_image",
    "corrected_AD_image",
    "corrected_FA_image",
    "corrected_MD_image",
    "corrected_RD_image",
}

DEFAULT_MIN_DW_DIRECTIONS = 6
B0_BVALUE_THRESHOLD = 50.0
DW_GRADIENT_NORM_THRESHOLD = 0.05

SOURCE_MANIFEST_HEADER = [
    "series_path",
    "subject",
    "series",
    "category",
    "modality",
    "source_status",
    "source_reason",
    "has_dicoms",
    "has_nifti",
    "is_axial_dti",
    "is_derived_modality",
]


def tool_path(cmd: str) -> str | None:
    found = shutil.which(cmd)
    if found:
        return found
    local_mrtrix = Path.home() / "mrtrix3" / "bin" / cmd
    if local_mrtrix.exists():
        return str(local_mrtrix)
    return None


def tool_ok(cmd: str) -> bool:
    return tool_path(cmd) is not None


def tool_cmd(cmd: str) -> str:
    return tool_path(cmd) or cmd


def subj_id(path: Path) -> str | None:
    for part in path.parts:
        if re.fullmatch(r"\d{3}_S_\d{4,5}", part):
            return part
    return None


def series_id(path: Path) -> str | None:
    for part in path.parts:
        if re.fullmatch(r"I\d+", part):
            return part
    return None


def raw_output_paths(out_dwi: Path, sid: str, ser: str) -> Tuple[Path, Path, Path]:
    stem = out_dwi / f"{sid}_{ser}"
    return stem.with_suffix(".mif"), stem.with_suffix(".bvec"), stem.with_suffix(".bval")


def already_converted_dwi(out_dwi: Path, sid: str, ser: str) -> bool:
    out_mif, out_bvec, out_bval = raw_output_paths(out_dwi, sid, ser)
    return out_mif.exists() and out_bvec.exists() and out_bval.exists()


def already_converted_derived(out_derived: Path, sid: str, ser: str) -> bool:
    return (out_derived / f"{sid}_{ser}.mif").exists()


def _base_for_nii(nii: Path) -> Path:
    name = nii.name
    if name.endswith(".nii.gz"):
        return nii.with_name(name[:-7])
    if name.endswith(".nii"):
        return nii.with_suffix("")
    return nii.with_suffix("")


def mrinfo_size_nvol(nii: Path) -> int:
    try:
        out = subprocess.check_output([tool_cmd("mrinfo"), str(nii), "-size", "-quiet"], text=True)
        parts = out.split()
        return int(parts[3]) if len(parts) >= 4 else 0
    except Exception:
        return 0


def _fsl_gradient_rows(bvec: Path, bval: Path) -> List[Tuple[float, float, float, float]]:
    if not (bvec.exists() and bval.exists()):
        return []
    try:
        bvals = [float(token) for token in bval.read_text(encoding="utf-8", errors="ignore").split()]
        bvec_rows = [
            [float(token) for token in line.split()]
            for line in bvec.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
    except Exception:
        return []
    if len(bvec_rows) != 3 or any(len(row) != len(bvals) for row in bvec_rows):
        return []
    return [(bvec_rows[0][idx], bvec_rows[1][idx], bvec_rows[2][idx], bvals[idx]) for idx in range(len(bvals))]


def _gradient_summary(rows: Sequence[Tuple[float, float, float, float]]) -> Dict[str, object]:
    shells = sorted({round(float(row[3]) / 100.0) * 100.0 for row in rows})
    b0_count = 0
    dw_count = 0
    usable_dw_count = 0
    for x, y, z, bvalue in rows:
        norm = (float(x) * float(x) + float(y) * float(y) + float(z) * float(z)) ** 0.5
        if abs(float(bvalue)) < B0_BVALUE_THRESHOLD:
            b0_count += 1
        else:
            dw_count += 1
            if norm > DW_GRADIENT_NORM_THRESHOLD:
                usable_dw_count += 1
    return {
        "nvol": len(rows),
        "b0_count": b0_count,
        "dw_count": dw_count,
        "usable_dw_count": usable_dw_count,
        "shells": shells,
    }


def _format_gradient_summary(summary: Dict[str, object]) -> str:
    shells = summary.get("shells", [])
    shell_label = ",".join(str(int(shell)) if float(shell).is_integer() else f"{shell:g}" for shell in shells) if shells else "none"
    return (
        f"nvol={summary.get('nvol', 0)}, "
        f"b0={summary.get('b0_count', 0)}, "
        f"dw={summary.get('dw_count', 0)}, "
        f"usable_dw={summary.get('usable_dw_count', 0)}, "
        f"shells={shell_label}"
    )


def dwi_quality_problem(
    mif: Path,
    bvec: Path,
    bval: Path,
    *,
    min_dw_directions: int = DEFAULT_MIN_DW_DIRECTIONS,
) -> str:
    if not (mif.exists() and bvec.exists() and bval.exists()):
        return "missing MIF/BVEC/BVAL output"
    rows = _fsl_gradient_rows(bvec, bval)
    if not rows:
        return "missing or malformed FSL gradient sidecars"
    nvol = mrinfo_size_nvol(mif)
    summary = _gradient_summary(rows)
    if nvol <= 0:
        return f"cannot determine DWI volume count ({_format_gradient_summary(summary)})"
    if nvol != len(rows):
        return f"DWI volume count does not match gradients ({nvol} vs {len(rows)}; {_format_gradient_summary(summary)})"
    if nvol < 2:
        return f"too few volumes for diffusion processing ({_format_gradient_summary(summary)})"
    if int(summary["b0_count"]) < 1:
        return f"missing b=0 volume ({_format_gradient_summary(summary)})"
    if int(summary["dw_count"]) < 1:
        return f"missing diffusion-weighted volume ({_format_gradient_summary(summary)})"
    min_dw = max(1, int(min_dw_directions))
    if int(summary["usable_dw_count"]) < min_dw:
        return (
            f"insufficient non-zero diffusion directions "
            f"(minimum={min_dw}; {_format_gradient_summary(summary)})"
        )
    return ""


def run_cap(cmd: Sequence[str], dry_run: bool = False) -> Tuple[int, str, str]:
    if dry_run:
        print("DRY:", " ".join(cmd))
        return 0, "", ""
    proc = subprocess.run(list(cmd), text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def classify_err(stderr: str) -> str:
    s = (stderr or "").lower()
    if "missing image frames" in s:
        return "bad_dicom_missing_frames"
    if "no gradient information found" in s or "no diffusion gradient" in s:
        return "no_gradients"
    if "already exists" in s:
        return "skip_exists"
    if "unable to interpret image" in s or "error opening image" in s:
        return "bad_dicom"
    return "unknown_error"


def dcm2niix_convert(src_dir: Path, tmp_root: Path, dry_run: bool = False) -> Tuple[Path | None, str]:
    sid = subj_id(src_dir) or "UNK"
    ser = series_id(src_dir) or src_dir.name
    tmp_dir = tmp_root / sid / ser
    if not dry_run:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [tool_cmd("dcm2niix"), "-z", "y", "-b", "y", "-o", str(tmp_dir), str(src_dir)]
    rc, _out, err = run_cap(cmd, dry_run=dry_run)
    if rc != 0:
        return None, err
    return tmp_dir, ""


def pick_best_nifti_with_grads(tmp_dir: Path) -> Tuple[Path | None, Path | None, Path | None]:
    best: Tuple[int, Path, Path, Path] | None = None
    for nii in sorted(tmp_dir.glob("*.nii*")):
        base = _base_for_nii(nii)
        bvec, bval = base.with_suffix(".bvec"), base.with_suffix(".bval")
        if not (bvec.exists() and bval.exists()):
            continue
        nvol = mrinfo_size_nvol(nii)
        if best is None or nvol > best[0]:
            best = (nvol, nii, bvec, bval)
    if best is None:
        return None, None, None
    return best[1:]


def first_nifti_in_dir(path: Path) -> Path | None:
    for pattern in ("*.nii.gz", "*.nii"):
        files = sorted(path.glob(pattern))
        if files:
            return files[0]
    return None


def has_dicoms(path: Path) -> bool:
    if first_nifti_in_dir(path):
        return False
    try:
        return len(list(path.iterdir())) >= 5
    except Exception:
        return False


def _path_contains_part(path: Path, candidates: Iterable[str]) -> bool:
    candidate_set = set(candidates)
    return any(part in candidate_set for part in path.parts)


def source_contract_for_series(
    ser_dir: Path,
    *,
    category: str = "",
    modality: str = "",
    axial_dir_name: str = DEFAULT_AXIAL_DIR_NAME,
    derived_dir_names: Iterable[str] = DEFAULT_DERIVED_DIR_NAMES,
) -> Dict[str, object]:
    """Classify one discovered series against the strict raw-DWI source contract.

    The goal is not to reject valid derived scalar maps; it is to make raw DWI
    selection auditable so derived AD/FA/MD/RD folders cannot silently enter the
    raw DWI conversion lane.
    """

    ser_dir = Path(ser_dir)
    derived_dir_names = set(derived_dir_names)
    sid = subj_id(ser_dir) or ""
    ser = series_id(ser_dir) or ""
    has_nii = first_nifti_in_dir(ser_dir) is not None
    dicom_like = has_dicoms(ser_dir)
    is_axial = bool(modality == axial_dir_name or _path_contains_part(ser_dir, {axial_dir_name}))
    is_derived = bool(modality in derived_dir_names or _path_contains_part(ser_dir, derived_dir_names))

    if not sid or not ser:
        status = "reject_bad_path"
        reason = "missing ADNI subject id or image series id"
    elif category == "raw_dwi" and is_derived:
        status = "reject_derived_as_raw"
        reason = "raw DWI candidate is inside a derived scalar-map modality"
    elif category == "raw_dwi" and not is_axial:
        status = "reject_not_axial_dti"
        reason = f"raw DWI candidate is not under {axial_dir_name}"
    elif category == "raw_dwi" and not (has_nii or dicom_like):
        status = "reject_no_image_inputs"
        reason = "raw DWI candidate has no NIfTI and is not DICOM-like"
    elif category == "raw_dwi":
        status = "strict_raw_ok"
        reason = "raw DWI source satisfies axial-DTI contract"
    elif category == "derived":
        status = "derived_ok" if (has_nii or dicom_like) else "reject_no_image_inputs"
        reason = "derived scalar source" if status == "derived_ok" else "derived folder has no readable image inputs"
    else:
        status = "unknown_category"
        reason = "unexpected discovery category"

    return {
        "series_path": str(ser_dir),
        "subject": sid,
        "series": ser,
        "category": category,
        "modality": modality,
        "source_status": status,
        "source_reason": reason,
        "has_dicoms": int(bool(dicom_like)),
        "has_nifti": int(bool(has_nii)),
        "is_axial_dti": int(bool(is_axial)),
        "is_derived_modality": int(bool(is_derived)),
    }


def filter_strict_raw_dwi_series(
    all_series: Sequence[Tuple[Path, str, str]],
    *,
    strict_raw_dwi_sources: bool = True,
    axial_dir_name: str = DEFAULT_AXIAL_DIR_NAME,
    derived_dir_names: Iterable[str] = DEFAULT_DERIVED_DIR_NAMES,
) -> Tuple[List[Tuple[Path, str, str]], Counter[str], List[Dict[str, object]]]:
    """Filter raw-DWI items that violate the source contract.

    Derived scalar maps are retained so the existing derived-output workflow can
    continue. Rejected raw candidates are returned as manifest rows for QC.
    """

    status_counts: Counter[str] = Counter()
    rejected: List[Dict[str, object]] = []
    kept: List[Tuple[Path, str, str]] = []
    for ser_dir, category, modality in all_series:
        row = source_contract_for_series(
            ser_dir,
            category=category,
            modality=modality,
            axial_dir_name=axial_dir_name,
            derived_dir_names=derived_dir_names,
        )
        status = str(row["source_status"])
        status_counts[status] += 1
        if strict_raw_dwi_sources and category == "raw_dwi" and status != "strict_raw_ok":
            rejected.append(row)
            continue
        kept.append((ser_dir, category, modality))
    return kept, status_counts, rejected


def write_source_manifest(
    all_series: Sequence[Tuple[Path, str, str]],
    manifest_csv: Path,
    *,
    axial_dir_name: str = DEFAULT_AXIAL_DIR_NAME,
    derived_dir_names: Iterable[str] = DEFAULT_DERIVED_DIR_NAMES,
) -> Dict[str, object]:
    rows = [
        source_contract_for_series(
            ser_dir,
            category=category,
            modality=modality,
            axial_dir_name=axial_dir_name,
            derived_dir_names=derived_dir_names,
        )
        for ser_dir, category, modality in all_series
    ]
    counts = Counter(str(row["source_status"]) for row in rows)
    manifest_csv = Path(manifest_csv)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=SOURCE_MANIFEST_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SOURCE_MANIFEST_HEADER})
    return {
        "manifest_csv": str(manifest_csv),
        "rows": rows,
        "status_counts": counts,
        "n_series": len(rows),
    }


def find_all_series(
    root: Path,
    axial_dir_name: str = DEFAULT_AXIAL_DIR_NAME,
    derived_dir_names: Iterable[str] = DEFAULT_DERIVED_DIR_NAMES,
) -> Tuple[List[Tuple[Path, str, str]], Dict[str, int]]:
    derived_dir_names = set(derived_dir_names)
    series: List[Tuple[Path, str, str]] = []
    counts = {"raw_dwi": 0, "derived": 0, "total": 0}

    for subj in sorted(root.glob("*_S_*")):
        for mod_dir in sorted(subj.iterdir()):
            if not mod_dir.is_dir():
                continue
            modality = mod_dir.name
            if modality == axial_dir_name:
                category = "raw_dwi"
            elif modality in derived_dir_names:
                category = "derived"
            else:
                continue

            for date_dir in sorted(mod_dir.iterdir()):
                if not date_dir.is_dir():
                    continue
                for ser in sorted(date_dir.iterdir()):
                    if ser.is_dir() and re.fullmatch(r"I\d+", ser.name):
                        series.append((ser, category, modality))
                        counts[category] += 1
                        counts["total"] += 1

    return series, counts


def _cleanup_partial(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def convert_raw_dwi(
    ser_dir: Path,
    out_dwi: Path,
    *,
    force: bool = False,
    quiet: bool = True,
    dry_run: bool = False,
    min_dw_directions: int = DEFAULT_MIN_DW_DIRECTIONS,
) -> Tuple[str, str]:
    sid, ser = subj_id(ser_dir), series_id(ser_dir)
    if not sid or not ser:
        return "skip_bad_path", "missing SID/SER"

    out_mif, out_bvec, out_bval = raw_output_paths(out_dwi, sid, ser)
    if already_converted_dwi(out_dwi, sid, ser) and not force:
        problem = dwi_quality_problem(out_mif, out_bvec, out_bval, min_dw_directions=min_dw_directions)
        if problem:
            return "needs_input_repair", problem
        return "skip_exists", ""

    if not tool_ok("mrconvert") or not tool_ok("mrinfo"):
        return "env_mrtrix_missing", "mrconvert/mrinfo not found"

    cmd = [
        tool_cmd("mrconvert"),
        str(ser_dir),
        str(out_mif),
        "-export_grad_fsl",
        str(out_bvec),
        str(out_bval),
    ]
    if force:
        cmd.append("-force")
    if quiet:
        cmd.append("-quiet")

    rc, _out, err = run_cap(cmd, dry_run=dry_run)
    if rc == 0:
        if dry_run:
            return "ok_direct", ""
        if already_converted_dwi(out_dwi, sid, ser):
            problem = dwi_quality_problem(out_mif, out_bvec, out_bval, min_dw_directions=min_dw_directions)
            if problem:
                return "needs_input_repair", problem
            return "ok_direct", ""
        _cleanup_partial((out_mif, out_bvec, out_bval))
        return "fail_incomplete_direct", "missing output sidecars after mrconvert"

    if classify_err(err) == "skip_exists" and not force:
        return "skip_exists", ""

    if not tool_ok("dcm2niix"):
        return "fallback_unavailable", "no dcm2niix"

    tmp_root = Path(tempfile.gettempdir()) / "dcm2niix_cache"
    tmp_dir, nx_err = dcm2niix_convert(ser_dir, tmp_root, dry_run=dry_run)
    if tmp_dir is None:
        return "dcm2niix_fail", nx_err

    nii, bvec, bval = pick_best_nifti_with_grads(tmp_dir)
    if nii is None or bvec is None or bval is None:
        return "no_bvec_bval", "no NIfTI with gradients"

    cmd = [
        tool_cmd("mrconvert"),
        str(nii),
        str(out_mif),
        "-fslgrad",
        str(bvec),
        str(bval),
        "-export_grad_fsl",
        str(out_bvec),
        str(out_bval),
    ]
    if force:
        cmd.append("-force")
    if quiet:
        cmd.append("-quiet")

    rc, _out, err = run_cap(cmd, dry_run=dry_run)
    if rc == 0:
        if dry_run:
            return "ok_nii", ""
        if already_converted_dwi(out_dwi, sid, ser):
            problem = dwi_quality_problem(out_mif, out_bvec, out_bval, min_dw_directions=min_dw_directions)
            if problem:
                return "needs_input_repair", problem
            return "ok_nii", ""
        _cleanup_partial((out_mif, out_bvec, out_bval))
        return "fail_incomplete_nii", "missing output sidecars after fallback mrconvert"

    _cleanup_partial((out_mif, out_bvec, out_bval))
    return "fail_mrconvert", err


def convert_derived_scalar(
    ser_dir: Path,
    out_derived: Path,
    *,
    force: bool = False,
    quiet: bool = True,
    dry_run: bool = False,
) -> Tuple[str, str]:
    sid, ser = subj_id(ser_dir), series_id(ser_dir)
    if not sid or not ser:
        return "skip_bad_path", "missing SID/SER"
    if already_converted_derived(out_derived, sid, ser) and not force:
        return "skip_exists", ""

    out_mif = out_derived / f"{sid}_{ser}.mif"

    if not tool_ok("mrconvert"):
        return "env_mrtrix_missing", "mrconvert not found"

    nii = first_nifti_in_dir(ser_dir)
    if nii is not None:
        cmd = [tool_cmd("mrconvert"), str(nii), str(out_mif)]
        success_status = "ok_nifti"
    else:
        if not has_dicoms(ser_dir):
            return "no_inputs_found", "no NIfTI and not a DICOM-like folder"
        cmd = [tool_cmd("mrconvert"), str(ser_dir), str(out_mif)]
        success_status = "ok_dicoms"

    if force:
        cmd.append("-force")
    if quiet:
        cmd.append("-quiet")

    rc, _out, err = run_cap(cmd, dry_run=dry_run)
    if rc == 0:
        if dry_run or out_mif.exists():
            return success_status, ""
        return "fail_incomplete", "missing output mif after mrconvert"

    return "fail_mrconvert", err


def _convert_one(
    item: Tuple[Path, str, str],
    *,
    out_dwi: Path,
    out_derived: Path,
    force: bool,
    quiet: bool,
    dry_run: bool,
    min_dw_directions: int,
) -> Tuple[Path, str, str, str, str]:
    ser_dir, category, modality = item
    try:
        if category == "raw_dwi":
            status, note = convert_raw_dwi(
                ser_dir,
                out_dwi,
                force=force,
                quiet=quiet,
                dry_run=dry_run,
                min_dw_directions=min_dw_directions,
            )
        elif category == "derived":
            status, note = convert_derived_scalar(
                ser_dir,
                out_derived,
                force=force,
                quiet=quiet,
                dry_run=dry_run,
            )
        else:
            status, note = "unknown_category", "unexpected category"
    except Exception as exc:
        status, note = "fail_exception", repr(exc)
    return ser_dir, category, modality, status, note


def run_conversion_batch(
    all_series: Sequence[Tuple[Path, str, str]],
    *,
    out_dwi: Path,
    out_derived: Path,
    force: bool = False,
    quiet: bool = True,
    dry_run: bool = False,
    jobs: int = 1,
    min_dw_directions: int = DEFAULT_MIN_DW_DIRECTIONS,
    strict_raw_dwi_sources: bool = False,
    source_manifest_csv: Path | None = None,
) -> Dict[str, object]:
    status_counts: Counter[str] = Counter()
    examples: DefaultDict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    out_dwi.mkdir(parents=True, exist_ok=True)
    out_derived.mkdir(parents=True, exist_ok=True)

    source_manifest: Dict[str, object] | None = None
    rejected_sources: List[Dict[str, object]] = []
    source_status_counts: Counter[str] = Counter()
    if source_manifest_csv is not None:
        source_manifest = write_source_manifest(all_series, Path(source_manifest_csv))
        source_status_counts = Counter(source_manifest["status_counts"])  # type: ignore[arg-type]

    if strict_raw_dwi_sources:
        all_series, source_status_counts, rejected_sources = filter_strict_raw_dwi_series(
            all_series,
            strict_raw_dwi_sources=True,
        )
        for row in rejected_sources:
            key = f"raw_dwi:{row['source_status']}"
            status_counts[key] += 1
            if len(examples[key]) < 5:
                examples[key].append(
                    (
                        str(row["series_path"]),
                        str(row["modality"]),
                        str(row["source_reason"]),
                    )
                )
        print(
            "Strict raw-DWI source gate: "
            f"kept={len(all_series)} | rejected_raw={len(rejected_sources)}"
        )

    if jobs <= 1:
        iterator = (
            _convert_one(
                item,
                out_dwi=out_dwi,
                out_derived=out_derived,
                force=force,
                quiet=quiet,
                dry_run=dry_run,
                min_dw_directions=min_dw_directions,
            )
            for item in all_series
        )
        for ser_dir, category, modality, status, note in tqdm(
            iterator,
            total=len(all_series),
            desc="convert_all",
            unit="series",
        ):
            key = f"{category}:{status}"
            status_counts[key] += 1
            if (note or not status.startswith("ok")) and len(examples[key]) < 5:
                examples[key].append((str(ser_dir), modality, note))
    else:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            futures = {
                ex.submit(
                    _convert_one,
                    item,
                    out_dwi=out_dwi,
                    out_derived=out_derived,
                    force=force,
                    quiet=quiet,
                    dry_run=dry_run,
                    min_dw_directions=min_dw_directions,
                ): item
                for item in all_series
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="convert_all", unit="series"):
                ser_dir, category, modality, status, note = fut.result()
                key = f"{category}:{status}"
                status_counts[key] += 1
                if (note or not status.startswith("ok")) and len(examples[key]) < 5:
                    examples[key].append((str(ser_dir), modality, note))

    print("\nSummary (by category/status):")
    for key, value in sorted(status_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{key:>32}: {value}")

    return {
        "status_counts": status_counts,
        "examples": dict(examples),
        "source_status_counts": source_status_counts,
        "source_manifest": source_manifest,
        "rejected_sources": rejected_sources,
    }


def write_examples_tsv(
    examples: Dict[str, List[Tuple[str, str, str]]],
    tsv_path: Path,
) -> None:
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w") as fp:
        fp.write("category_status\tseries_path\tmodality\tnote\n")
        for key, rows in examples.items():
            for path, mod, note in rows:
                note = (note or "").replace("\t", " ").replace("\n", " ")
                fp.write(f"{key}\t{path}\t{mod}\t{note}\n")


def collect_conversion_output_status(out_dwi: Path, out_derived: Path) -> Dict[str, int]:
    raw_mif_files = sorted(out_dwi.glob("*.mif"))
    raw_complete = 0
    raw_incomplete = 0

    for mif_path in raw_mif_files:
        stem = mif_path.with_suffix("")
        has_bvec = stem.with_suffix(".bvec").exists()
        has_bval = stem.with_suffix(".bval").exists()
        if has_bvec and has_bval:
            raw_complete += 1
        else:
            raw_incomplete += 1

    derived_mif_files = sum(1 for _ in out_derived.glob("*.mif"))
    return {
        "raw_mif_files": len(raw_mif_files),
        "raw_complete": raw_complete,
        "raw_incomplete": raw_incomplete,
        "derived_mif_files": derived_mif_files,
    }


def print_conversion_output_status(out_dwi: Path, out_derived: Path) -> None:
    counts = collect_conversion_output_status(out_dwi, out_derived)
    print("\nOutput status:")
    print(f"{'raw mif files':>20}: {counts['raw_mif_files']}")
    print(f"{'raw complete':>20}: {counts['raw_complete']}")
    print(f"{'raw incomplete':>20}: {counts['raw_incomplete']}")
    print(f"{'derived mif files':>20}: {counts['derived_mif_files']}")


def print_conversion_hints(status_counts: Dict[str, int]) -> None:
    if any(key.startswith("raw_dwi:env") for key in status_counts):
        print("MRtrix tools missing for raw DWI. Check `which mrconvert` and `which mrinfo`.")
    if any(key.startswith("raw_dwi:fallback_unavailable") for key in status_counts):
        print("Install dcm2niix for fallback path.")
