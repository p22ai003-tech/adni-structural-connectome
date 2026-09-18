#!/usr/bin/env python3
"""
T1 preprocessing pipeline for ADNI (Jupyter/terminal friendly).

Input:
  RAW_T1_ROOT = ~/exp/data/Images/mri

ADNI-style tree (example):
  Images/mri/002_S_0413/MT1__GradWarp__N3m/2010-05-06_12_37_46.0/I291872/  (DICOMs and/or NIfTI)

Outputs:
  derivatives/t1_anat:
    - t1_from_dicom_<SID>_<SER>.nii.gz
    - corrected_T1_<SID>_<SER>.nii.gz
    - T1_ss_<SID>_<SER>.nii.gz
    - brainmask_<SID>_<SER>.nii.gz

  derivatives/t1_fast:
    - fast_<SID>_<SER>_pve_0/1/2.nii.gz
    - fast_<SID>_<SER>_seg.nii.gz
    - fast_<SID>_<SER>_restore.nii.gz
"""

import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from connectome_pipeline.pipeline_paths import resolve_pipeline_paths

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    class _TqdmFallback:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
            self.total = total

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n=1):
            return None

        def write(self, msg):
            print(msg)

    def tqdm(iterable=None, *args, **kwargs):
        return _TqdmFallback(iterable=iterable, total=kwargs.get("total"))

try:
    import SimpleITK as sitk
except ModuleNotFoundError:
    sitk = None

# ---------------------- CONFIG ----------------------
DEFAULT_PATHS = resolve_pipeline_paths(create_layout=True)
ROOT = DEFAULT_PATHS.data_root
RAW_T1 = DEFAULT_PATHS.raw_t1_root
DERIV = DEFAULT_PATHS.deriv_root

T1_ANAT = DERIV / "t1_anat"
T1_FAST = DERIV / "t1_fast"

T1_ANAT.mkdir(parents=True, exist_ok=True)
T1_FAST.mkdir(parents=True, exist_ok=True)

DEFAULT_T1_MODALITY_NAMES = {
    "Accelerated_Sag_IR-FSPGR",
    "Accelerated_Sagittal_IR-FSPGR",
    "Accelerated_Sagittal_MPRAGE",
    "Accelerated_Sagittal_MPRAGE__MSV21_",
    "Accelerated_Sagittal_MPRAGE__MSV22_",
    "Accelerated_Satittal_MPRAGE",
    "MPR__GradWarp__N3",
    "MT1__GradWarp__N3m",
    "MT1__N3m",
    "ORIG_Accelerated_Sag_IR-FSPGR",
    "Sag_Accel_IR-FSPGR",
    "Sagittal_3D_Accelerated_MPRAGE",
}

FORCE = False
# Keep the default subject fan-out low; users can still raise it explicitly.
MAX_WORKERS = min(2, max(1, (os.cpu_count() or 4) // 2))

# restrict subjects if you want, else leave empty
SELECT_SUBJECTS: List[str] = []
# ----------------------------------------------------


# ---------- tool helpers ----------
def _which(tool: str, required=True) -> Optional[str]:
    p = shutil.which(tool)
    if required and p is None:
        raise RuntimeError(f"Required tool '{tool}' not found in PATH")
    return p

def _resolve_tool(tool: str, *extra_candidates: Path) -> Optional[str]:
    found = _which(tool, required=False)
    if found is not None:
        return found
    for candidate in extra_candidates:
        if candidate.exists():
            return str(candidate)
    return None


Dcm2niix = _resolve_tool("dcm2niix", Path.home() / "bin" / "dcm2niix")
MRCONVERT = _resolve_tool("mrconvert", Path.home() / "mrtrix3" / "bin" / "mrconvert") or "mrconvert"
N4 = _resolve_tool(
    "N4BiasFieldCorrection",
    Path(os.environ.get("ANTSPATH", "")) / "N4BiasFieldCorrection" if os.environ.get("ANTSPATH") else Path("/__missing__"),
    Path.home() / "bin" / "N4BiasFieldCorrection",
    Path.home() / ".local" / "bin" / "N4BiasFieldCorrection",
)
BET = _resolve_tool(
    "bet",
    Path.home() / "fsl" / "bin" / "bet",
    Path.home() / "fsl" / "share" / "fsl" / "bin" / "bet",
) or "bet"
FAST = _resolve_tool(
    "fast",
    Path.home() / "fsl" / "bin" / "fast",
    Path.home() / "fsl" / "share" / "fsl" / "bin" / "fast",
) or "fast"

def _run(cmd, env=None):
    env = env or os.environ.copy()
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env
    )
    if proc.returncode != 0:
        msg = (
            f"Command failed (rc={proc.returncode}): {' '.join(map(str, cmd))}\n"
            f"STDOUT: {proc.stdout[:400]}\n"
            f"STDERR: {proc.stderr[:400]}"
        )
        raise RuntimeError(msg)
    return proc


def _run_n4_bias_correction(input_nii: Path, output_nii: Path, bias_nii: Path) -> None:
    if N4:
        _run([
            N4, "-d", "3",
            "-i", str(input_nii),
            "-o", f"[{output_nii},{bias_nii}]",
            "-s", "2",
            "-b", "[200]",
            "-c", "[200x200x100x50,1e-6]",
            "-v",
        ])
        return

    # This EC2 environment may not have ANTs installed. In that case, keep the
    # raw T1 as the registration input so Step 5 can resume incomplete subjects.
    shutil.copy2(input_nii, output_nii)
    if sitk is not None:
        image = sitk.ReadImage(str(input_nii), sitk.sitkFloat32)
        bias_field = sitk.Image(image.GetSize(), sitk.sitkFloat32) + 1.0
        bias_field.CopyInformation(image)
        sitk.WriteImage(bias_field, str(bias_nii))
    else:
        shutil.copy2(input_nii, bias_nii)


# ---------- ID helpers ----------
def subj_id(path: Path) -> Optional[str]:
    for part in path.parts:
        if re.fullmatch(r"\d{3}_S_\d{4}", part):
            return part
    return None

def series_id(path: Path) -> Optional[str]:
    for part in path.parts:
        if re.fullmatch(r"I\d+", part):
            return part
    return None


def _is_visible_path(path: Path) -> bool:
    return not path.name.startswith("._") and not path.name.startswith(".")


def _iter_visible_dirs(path: Path):
    for child in sorted(path.iterdir()):
        if child.is_dir() and _is_visible_path(child):
            yield child


def _visible_glob(path: Path, pattern: str) -> List[Path]:
    return [p for p in sorted(path.glob(pattern)) if p.is_file() and _is_visible_path(p)]


def find_eddy_subjects(eddy_dir: Path) -> List[str]:
    subjects = set()
    for p in _visible_glob(eddy_dir, "*.mif"):
        name = p.name
        if name.endswith("_preproc.mif"):
            series = name.replace("_preproc.mif", "")
        elif name.endswith("_eddy.mif"):
            series = name.replace("_eddy.mif", "")
        else:
            continue
        sid = series.split("_I", 1)[0] if "_I" in series else None
        if sid:
            subjects.add(sid)
    return sorted(subjects)


# ---------- scan for T1 series ----------
def find_t1_series(
    root: Path,
    *,
    modality_names=None,
    select_subjects: Optional[List[str]] = None,
):
    modality_names = set(modality_names or DEFAULT_T1_MODALITY_NAMES)
    select_subjects = set(select_subjects or [])

    for subj in sorted(root.glob("*_S_*")):
        sid = subj_id(subj)
        if not sid:
            continue
        if select_subjects and sid not in select_subjects:
            continue

        for mod_dir in _iter_visible_dirs(subj):
            modality = mod_dir.name

            if modality_names:
                if modality not in modality_names:
                    continue
            else:
                m_upper = modality.upper()
                if "T1" not in m_upper and "MPR" not in m_upper:
                    continue

            for date_dir in _iter_visible_dirs(mod_dir):
                for ser_dir in _iter_visible_dirs(date_dir):
                    ser = series_id(ser_dir)
                    if not ser:
                        continue
                    yield (sid, ser, ser_dir)


# ---------- NIfTI discovery / DICOM conversion ----------
def first_nifti(d: Path) -> Optional[Path]:
    for ext in ("*.nii.gz", "*.nii"):
        files = _visible_glob(d, ext)
        if files:
            return files[0]
    return None

def has_dicoms(d: Path) -> bool:
    if first_nifti(d):
        return False
    try:
        files = [p for p in d.iterdir() if p.is_file() and _is_visible_path(p)]
    except Exception:
        return False
    if not files:
        return False

    # ADNI T1 series can arrive either as many slices or as a single
    # encapsulated DICOM. Accept obvious DICOM-like files instead of requiring
    # an arbitrary minimum file count.
    dicom_suffixes = {".dcm", ".ima", ".dicom", ""}
    for p in files:
        if p.suffix.lower() in dicom_suffixes:
            return True
        try:
            with p.open("rb") as fh:
                header = fh.read(132)
            if len(header) >= 132 and header[128:132] == b"DICM":
                return True
        except Exception:
            continue
    return False

def ensure_t1_nifti(sid: str, ser: str, ser_dir: Path, tmp_root: Path) -> Path:
    nii = first_nifti(ser_dir)
    if nii is not None:
        return nii

    if not has_dicoms(ser_dir):
        raise RuntimeError(f"No NIfTI and doesn't look like DICOMs: {ser_dir}")

    if Dcm2niix is None:
        raise RuntimeError("Need dcm2niix to convert T1 DICOMs to NIfTI, but it's not in PATH.")

    out_dir = tmp_root / sid / ser
    if out_dir.exists():
        for stale in out_dir.iterdir():
            if stale.is_file():
                stale.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [Dcm2niix, "-z", "y", "-b", "y", "-o", str(out_dir), str(ser_dir)]
    _run(cmd)
    nii = first_nifti(out_dir)
    if nii is None:
        raise RuntimeError(f"dcm2niix ran but produced no NIfTI in {out_dir}")
    return nii


# ---------- BET mask helper (this was the buggy bit) ----------
def _bet_mask_path(out_nii: Path) -> Path:
    """
    Given BET output image path (e.g. t1_brain.nii.gz),
    return the corresponding mask path that BET writes with '-m':
        t1_brain_mask.nii.gz
    """
    name = out_nii.name
    if name.endswith(".nii.gz"):
        base = name[:-7]          # strip '.nii.gz'
        return out_nii.with_name(base + "_mask.nii.gz")
    elif name.endswith(".nii"):
        base = name[:-4]          # strip '.nii'
        return out_nii.with_name(base + "_mask.nii.gz")
    else:
        return out_nii.with_name(name + "_mask")


# ---------- worker ----------
def _t1_outputs_complete(base: str, t1_anat: Path, t1_fast: Path) -> bool:
    required = [
        t1_anat / f"t1_from_dicom_{base}.nii.gz",
        t1_anat / f"corrected_T1_{base}.nii.gz",
        t1_anat / f"T1_ss_{base}.nii.gz",
        t1_anat / f"brainmask_{base}.nii.gz",
        t1_fast / f"fast_{base}_pve_2.nii.gz",
        t1_fast / f"fast_{base}_seg.nii.gz",
    ]
    return all(path.exists() for path in required)


def t1_worker(
    sid: str,
    ser: str,
    ser_dir: Path,
    *,
    t1_anat: Path,
    t1_fast: Path,
    force: bool,
) -> Dict[str, str]:
    base = f"{sid}_{ser}"
    out_raw  = t1_anat / f"t1_from_dicom_{base}.nii.gz"
    out_corr = t1_anat / f"corrected_T1_{base}.nii.gz"
    out_ss   = t1_anat / f"T1_ss_{base}.nii.gz"
    out_mask = t1_anat / f"brainmask_{base}.nii.gz"

    fast_base = t1_fast / f"fast_{base}"

    if not force and _t1_outputs_complete(base, t1_anat, t1_fast):
        return {"sid": sid, "ser": ser, "status": "skip_exists", "details": ""}

    try:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)

            # 1) raw NIfTI
            if not force and out_raw.exists():
                in_nii = out_raw
            else:
                in_nii = ensure_t1_nifti(sid, ser, ser_dir, td)
                _run([MRCONVERT, str(in_nii), str(out_raw), "-quiet", "-force"])

            # 2) N4 bias correction
            bias_nii = td / "bias.nii.gz"
            _run_n4_bias_correction(out_raw, out_corr, bias_nii)

            # 3) Skull-stripping with BET
            corr_tmp = td / "corr_T1.nii.gz"
            _run([MRCONVERT, str(out_corr), str(corr_tmp), "-quiet", "-force"])

            tmp_brain = td / "t1_brain.nii.gz"
            _run([
                BET,
                str(corr_tmp), str(tmp_brain),
                "-R", "-f", "0.2", "-g", "0", "-m"
            ])

            bet_mask = _bet_mask_path(tmp_brain)

            # copy / rename to our standardized names in t1_anat
            _run([MRCONVERT, str(tmp_brain), str(out_ss),   "-quiet", "-force"])
            _run([MRCONVERT, str(bet_mask),  str(out_mask), "-quiet", "-force"])

            # 4) FAST segmentation (on bias‑corrected T1)
            fast_base_path = fast_base
            fast_base_path.parent.mkdir(parents=True, exist_ok=True)
            fast_input = out_corr if N4 else out_ss
            _run([
                FAST,
                "-t", "1",
                "-n", "3",
                "-H", "0.1",
                "-I", "4",
                "-l", "20.0",
                "-o", str(fast_base_path),
                str(fast_input)
            ])

        return {"sid": sid, "ser": ser, "status": "ok", "details": ""}

    except Exception as e:
        msg = str(e)
        print(f"\n[T1 pipeline] {sid}_{ser} FAILED\n{msg}\n")
        return {
            "sid": sid,
            "ser": ser,
            "status": f"fail:{type(e).__name__}",
            "details": msg[:500],
        }


# ---------- main driver ----------
def run_t1_preprocessing(
    *,
    raw_t1_root: Path = RAW_T1,
    t1_anat_dir: Path = T1_ANAT,
    t1_fast_dir: Path = T1_FAST,
    modality_names=None,
    select_subjects: Optional[List[str]] = None,
    force: bool = FORCE,
    max_workers: int = MAX_WORKERS,
) -> Dict[str, object]:
    t1_anat_dir.mkdir(parents=True, exist_ok=True)
    t1_fast_dir.mkdir(parents=True, exist_ok=True)

    series = list(
        find_t1_series(
            raw_t1_root,
            modality_names=modality_names,
            select_subjects=select_subjects,
        )
    )
    if not series:
        print(f"No T1 series found under {raw_t1_root}")
        return {"results": [], "status_counts": Counter(), "series_count": 0}

    print(
        f"T1 pipeline: {len(series)} series found | FORCE={force} | max_workers={max_workers}"
    )
    results: List[Dict[str, str]] = []
    status_counts: Counter[str] = Counter()

    with ThreadPoolExecutor(max_workers=max_workers) as ex, \
         tqdm(total=len(series), desc="T1 pipeline", unit="series", dynamic_ncols=True) as pbar:

        futs = {
            ex.submit(
                t1_worker,
                sid,
                ser,
                sdir,
                t1_anat=t1_anat_dir,
                t1_fast=t1_fast_dir,
                force=force,
            ): (sid, ser)
            for sid, ser, sdir in series
        }
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            status_counts[res["status"]] += 1
            pbar.update(1)
            sid, ser = res["sid"], res["ser"]
            detail = f" ({res['details']})" if res["details"] else ""
            pbar.write(f"{sid}_{ser}: {res['status']}{detail}")

    ok   = sum(1 for r in results if r["status"] == "ok")
    skip = sum(1 for r in results if r["status"] == "skip_exists")
    fail = len(results) - ok - skip
    print(f"\nT1 summary: ok={ok}, skip={skip}, fail={fail}")
    return {
        "results": results,
        "status_counts": status_counts,
        "series_count": len(series),
        "selected_subject_count": len(set(select_subjects or [])),
    }


def run_t1_pipeline(max_workers: int = MAX_WORKERS) -> List[Dict[str, str]]:
    return run_t1_preprocessing(
        raw_t1_root=RAW_T1,
        t1_anat_dir=T1_ANAT,
        t1_fast_dir=T1_FAST,
        modality_names=DEFAULT_T1_MODALITY_NAMES,
        select_subjects=SELECT_SUBJECTS,
        force=FORCE,
        max_workers=max_workers,
    )["results"]
