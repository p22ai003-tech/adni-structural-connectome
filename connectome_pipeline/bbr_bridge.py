#!/usr/bin/env python3
"""T1 <-> DWI BBR bridge helper for the ADNI structural-connectome pipeline."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from connectome_pipeline.pipeline_paths import resolve_pipeline_paths

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover
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

        def set_postfix(self, *args, **kwargs):
            return None

    def tqdm(iterable=None, *args, **kwargs):
        return _TqdmFallback(iterable=iterable, total=kwargs.get("total"))


DEFAULT_DERIV = resolve_pipeline_paths(create_layout=True).deriv_root
# Keep the default subject fan-out low; users can still raise it explicitly.
DEFAULT_MAX_WORKERS = min(2, max(1, (os.cpu_count() or 4) // 2))
DEFAULT_STAGE_ROOT = Path.home() / "staging" / "bbr_bridge_stage"


def _which(tool: str, required: bool = True) -> Optional[str]:
    from shutil import which

    path = which(tool)
    if required and path is None:
        raise RuntimeError(f"Required tool '{tool}' not found in PATH")
    return path


def _visible_glob(path: Path, pattern: str) -> List[Path]:
    return [
        p
        for p in sorted(path.glob(pattern))
        if p.is_file() and not p.name.startswith("._") and not p.name.startswith(".")
    ]


def _run(cmd: List[str], *, env: dict, log_file: Optional[Path] = None) -> None:
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a") as lf:
            lf.write("\n$ " + " ".join(map(str, cmd)) + "\n")
            proc = subprocess.run(cmd, stdout=lf, stderr=lf, text=True, env=env, check=False)
    else:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(map(str, cmd))}")


def _subject_from_sid(sid: str) -> str:
    if "_I" in sid:
        return sid.split("_I", 1)[0]
    parts = sid.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else sid


def _get_env() -> dict:
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    return env


def _flirt_paths() -> tuple[str, Optional[Path]]:
    flirt = _which("flirt", required=True)
    fsldir_env = os.environ.get("FSLDIR")
    schedule = None
    if fsldir_env:
        cand = Path(fsldir_env) / "etc" / "flirtsch" / "bbr.sch"
        if cand.exists():
            schedule = cand
    return flirt, schedule


def _find_t1_for_registration(t1_anat: Path, sid: str) -> Path:
    subj = _subject_from_sid(sid)
    ordered_patterns = [
        f"corrected_T1_{sid}_*.nii.gz",
        f"corrected_T1_{subj}_*.nii.gz",
        f"T1_ss_{sid}_*.nii.gz",
        f"T1_ss_{subj}_*.nii.gz",
        f"t1_from_dicom_{sid}_*.nii.gz",
        f"t1_from_dicom_{subj}_*.nii.gz",
    ]
    for pattern in ordered_patterns:
        matches = _visible_glob(t1_anat, pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No usable T1 input found for {sid} under {t1_anat}")


def _find_t1_brain(t1_anat: Path, sid: str) -> Optional[Path]:
    subj = _subject_from_sid(sid)
    for pattern in (f"T1_ss_{sid}_*.nii.gz", f"T1_ss_{subj}_*.nii.gz"):
        matches = _visible_glob(t1_anat, pattern)
        if matches:
            return matches[0]
    return None


def _find_wmseg(t1_fast: Path, sid: str) -> Optional[Path]:
    subj = _subject_from_sid(sid)
    for pattern in (f"fast_{sid}_*_pve_2.nii.gz", f"fast_{subj}_*_pve_2.nii.gz"):
        matches = _visible_glob(t1_fast, pattern)
        if matches:
            return matches[0]
    return None


def _discover_bias_sids(bias_dir: Path) -> List[str]:
    return [p.name.replace("_unbiased.mif", "") for p in _visible_glob(bias_dir, "*_unbiased.mif")]


def _normalize_requested_sids(
    *,
    sids: Optional[List[str]] = None,
    select_sids: Optional[List[str]] = None,
) -> Optional[List[str]]:
    requested = select_sids if select_sids is not None else sids
    if requested is None:
        return None
    normalized = [sid for sid in requested if sid]
    return normalized or None


def _output_paths(bbr_dir: Path, sid: str) -> Dict[str, Path]:
    return {
        "b0mean": bbr_dir / f"{sid}_b0mean_ras.nii.gz",
        "t1_ras": bbr_dir / f"{sid}_t1_ras.nii.gz",
        "t1brain_ras": bbr_dir / f"{sid}_t1brain_ras.nii.gz",
        "mat": bbr_dir / f"{sid}_t12b0_bbr.mat",
        "mask_b0": bbr_dir / f"{sid}_mask_on_b0_bbr.nii.gz",
    }


def _key_outputs_exist(bbr_dir: Path, sid: str) -> bool:
    out = _output_paths(bbr_dir, sid)
    return out["b0mean"].exists() and out["t1_ras"].exists() and out["mat"].exists()


def _append_summary(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    header = [
        "sid",
        "status",
        "cost_mode",
        "t1_input",
        "t1brain_input",
        "wmseg",
        "elapsed_sec",
        "log",
        "details",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=header)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})


def bbr_worker(
    sid: str,
    *,
    deriv: Path = DEFAULT_DERIV,
    force: bool = False,
    stage_root: Path = DEFAULT_STAGE_ROOT,
    overwrite_logs: bool = True,
) -> Dict[str, str]:
    deriv = Path(deriv)
    bias = deriv / "biascorr_1"
    t1_anat = deriv / "t1_anat"
    t1_fast = deriv / "t1_fast"
    bbr_dir = deriv / "dwi_t1_bbr"
    qc_dir = deriv / "qc"
    log_dir = qc_dir / "bbr_logs"
    bbr_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{sid}.log"
    if overwrite_logs and log_file.exists():
        log_file.unlink()

    out = _output_paths(bbr_dir, sid)
    row = {
        "sid": sid,
        "status": "",
        "cost_mode": "",
        "t1_input": "",
        "t1brain_input": "",
        "wmseg": "",
        "elapsed_sec": "",
        "log": str(log_file),
        "details": "",
    }
    t0 = time.time()

    try:
        if not force and _key_outputs_exist(bbr_dir, sid):
            row["status"] = "skip_exists"
            row["details"] = "bridge outputs already exist"
            return row

        env = _get_env()
        flirt, bbr_schedule = _flirt_paths()
        bet = _which("bet", required=True)
        convert_xfm = _which("convert_xfm", required=True)
        epi_reg = _which("epi_reg", required=False)
        mrconvert = _which("mrconvert", required=True)

        ub_b0 = bias / f"{sid}_b0.nii.gz"
        ub_mif = bias / f"{sid}_unbiased.mif"
        if not ub_b0.exists() and not ub_mif.exists():
            raise FileNotFoundError(f"Missing both {ub_b0.name} and {ub_mif.name}")

        t1_input = _find_t1_for_registration(t1_anat, sid)
        t1_brain_input = _find_t1_brain(t1_anat, sid)
        wmseg = _find_wmseg(t1_fast, sid)
        row["t1_input"] = t1_input.name
        row["t1brain_input"] = t1_brain_input.name if t1_brain_input else ""
        row["wmseg"] = wmseg.name if wmseg else ""

        with tempfile.TemporaryDirectory(dir=stage_root) as td_str:
            td = Path(td_str)
            staged_out = _output_paths(td, sid)

            # B0 mean / reference
            if ub_b0.exists():
                _run([mrconvert, str(ub_b0), str(staged_out["b0mean"]), "-quiet", "-force"], env=env, log_file=log_file)
            else:
                tmp_b0 = td / f"{sid}_b0_tmp.nii.gz"
                _run(
                    [mrconvert, str(ub_mif), str(tmp_b0), "-coord", "3", "0", "-quiet", "-force"],
                    env=env,
                    log_file=log_file,
                )
                _run([mrconvert, str(tmp_b0), str(staged_out["b0mean"]), "-quiet", "-force"], env=env, log_file=log_file)

            # T1 in native orientation / space
            t1_native = td / f"{sid}_t1_native.nii.gz"
            _run([mrconvert, str(t1_input), str(t1_native), "-quiet", "-force"], env=env, log_file=log_file)

            # T1 brain in native T1 space is required by epi_reg / BBR.
            t1_brain_native = td / f"{sid}_t1brain_native.nii.gz"
            if t1_brain_input is not None:
                _run([mrconvert, str(t1_brain_input), str(t1_brain_native), "-quiet", "-force"], env=env, log_file=log_file)
            else:
                t1_brain_tmp = td / f"{sid}_t1brain_tmp.nii.gz"
                _run([bet, str(t1_native), str(t1_brain_tmp), "-R", "-f", "0.2", "-g", "0", "-m"], env=env, log_file=log_file)
                _run([mrconvert, str(t1_brain_tmp), str(t1_brain_native), "-quiet", "-force"], env=env, log_file=log_file)

            # Standard BBR direction is EPI/B0 -> T1, then invert to T1 -> B0
            # for the downstream MRtrix / FSL transform chain.
            b02t1_mat = td / f"{sid}_b02t1.mat"
            cost_mode = "normmi"
            if epi_reg is not None:
                epi_prefix = td / f"{sid}_b02t1_bbr"
                epi_cmd = [
                    epi_reg,
                    f"--epi={staged_out['b0mean']}",
                    f"--t1={t1_native}",
                    f"--t1brain={t1_brain_native}",
                    f"--out={epi_prefix}",
                ]
                if wmseg is not None:
                    epi_cmd.append(f"--wmseg={wmseg}")
                try:
                    _run(epi_cmd, env=env, log_file=log_file)
                    epi_mat = Path(f"{epi_prefix}.mat")
                    if epi_mat.exists():
                        shutil.copy2(epi_mat, b02t1_mat)
                        cost_mode = "epi_reg_bbr"
                except Exception as exc:
                    with log_file.open("a") as lf:
                        lf.write(f"\n# epi_reg failed; falling back to flirt BBR/normmi: {exc}\n")

            if not b02t1_mat.exists():
                flirt_cmd = [
                    flirt,
                    "-in",
                    str(staged_out["b0mean"]),
                    "-ref",
                    str(t1_native),
                    "-omat",
                    str(b02t1_mat),
                    "-out",
                    str(td / f"{sid}_b0_in_t1.nii.gz"),
                    "-dof",
                    "6",
                ]
                if wmseg is not None and bbr_schedule is not None:
                    flirt_cmd += ["-cost", "bbr", "-wmseg", str(wmseg), "-schedule", str(bbr_schedule)]
                    cost_mode = "flirt_bbr_b02t1"
                else:
                    flirt_cmd += ["-cost", "normmi"]
                    cost_mode = "flirt_normmi_b02t1"
                _run(flirt_cmd, env=env, log_file=log_file)

            _run(
                [convert_xfm, "-inverse", "-omat", str(staged_out["mat"]), str(b02t1_mat)],
                env=env,
                log_file=log_file,
            )
            row["cost_mode"] = cost_mode

            for src, dest in (
                (t1_native, staged_out["t1_ras"]),
                (t1_brain_native, staged_out["t1brain_ras"]),
            ):
                _run(
                    [
                        flirt,
                        "-in",
                        str(src),
                        "-ref",
                        str(staged_out["b0mean"]),
                        "-applyxfm",
                        "-init",
                        str(staged_out["mat"]),
                        "-interp",
                        "trilinear",
                        "-out",
                        str(dest),
                    ],
                    env=env,
                    log_file=log_file,
                )

            # Brain mask on B0
            b0_brain = td / f"{sid}_b0_brain.nii.gz"
            _run([bet, str(staged_out["b0mean"]), str(b0_brain), "-m", "-f", "0.2"], env=env, log_file=log_file)
            b0_brain_mask = td / f"{sid}_b0_brain_mask.nii.gz"
            if not b0_brain_mask.exists():
                raise RuntimeError(f"BET did not produce {b0_brain_mask.name}")
            _run([mrconvert, str(b0_brain_mask), str(staged_out["mask_b0"]), "-quiet", "-force"], env=env, log_file=log_file)

            for key in ("b0mean", "t1_ras", "mat", "mask_b0", "t1brain_ras"):
                if not staged_out[key].exists():
                    raise RuntimeError(f"Missing staged output: {staged_out[key].name}")

            bbr_dir.mkdir(parents=True, exist_ok=True)
            for key, dest in out.items():
                if staged_out[key].exists():
                    shutil.move(str(staged_out[key]), str(dest))

        row["status"] = "ok"
        row["details"] = "bridge ok"
        return row
    except Exception as exc:
        row["status"] = f"fail:{type(exc).__name__}"
        row["details"] = str(exc)[:500]
        return row
    finally:
        row["elapsed_sec"] = f"{time.time() - t0:.1f}"


def run_bbr_for_missing(
    *,
    deriv: Path = DEFAULT_DERIV,
    force: bool = False,
    sids: Optional[List[str]] = None,
    select_sids: Optional[List[str]] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    stage_root: Path = DEFAULT_STAGE_ROOT,
    overwrite_logs: bool = True,
) -> Dict[str, object]:
    deriv = Path(deriv)
    bias = deriv / "biascorr_1"
    bbr_dir = deriv / "dwi_t1_bbr"
    qc_dir = deriv / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    bbr_dir.mkdir(parents=True, exist_ok=True)

    all_bias_sids = _discover_bias_sids(bias)
    requested_sids = _normalize_requested_sids(sids=sids, select_sids=select_sids)
    candidate_sids = all_bias_sids if requested_sids is None else list(requested_sids)
    todo = [sid for sid in candidate_sids if force or not _key_outputs_exist(bbr_dir, sid)]

    if not todo:
        print("BBR: nothing to do.")
        return {
            "rows": [],
            "status_counts": Counter({"skip_exists": len(candidate_sids)}),
            "scheduled_now": 0,
            "already_done": len(candidate_sids),
            "summary_csv": str(qc_dir / "bbr_summary.csv"),
        }

    already_done = len(candidate_sids) - len(todo)
    print(
        f"BBR plan: bias-ready={len(all_bias_sids)} | selected={len(candidate_sids)} | "
        f"completed={already_done} | pending={len(todo)} | force={force} | "
        f"max_workers={max_workers} | stage_root={stage_root}"
    )

    rows: List[Dict[str, str]] = []
    status_counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex, tqdm(
        total=len(todo),
        desc="BBR persisted",
        unit="sid",
        dynamic_ncols=True,
    ) as pbar:
        futs = {
            ex.submit(
                bbr_worker,
                sid,
                deriv=deriv,
                force=force,
                stage_root=stage_root,
                overwrite_logs=overwrite_logs,
            ): sid
            for sid in todo
        }
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            status_counts[row["status"]] += 1
            pbar.update(1)
            pbar.set_postfix(
                ok=status_counts.get("ok", 0),
                skip=status_counts.get("skip_exists", 0),
                fail=sum(v for k, v in status_counts.items() if k.startswith("fail")),
            )
            pbar.write(f"{row['sid']}: {row['status']} ({row['details']})")

    summary_csv = qc_dir / "bbr_summary.csv"
    _append_summary(summary_csv, rows)
    ok = status_counts.get("ok", 0)
    skip = status_counts.get("skip_exists", 0)
    fail = sum(v for k, v in status_counts.items() if k.startswith("fail"))
    print(f"\nBBR summary: ok={ok}, skip={skip}, fail={fail}")
    return {
        "rows": rows,
        "status_counts": status_counts,
        "scheduled_now": len(todo),
        "already_done": already_done,
        "summary_csv": str(summary_csv),
    }


if __name__ == "__main__":  # pragma: no cover
    run_bbr_for_missing()
