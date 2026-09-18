#!/usr/bin/env python3
"""
Rerunnable DWI eddy / motion-correction stage for ADNI derivatives.

Default behaviour matches the BATMAN-style single-phase-encoding setup used in
this project:
  dwifslpreproc <input> <output> -rpe_none -pe_dir <header-or-j-> \
    -eddy_options="--slm=linear --data_is_shelled"

Inputs:
  derivatives/mif_unringed/<series>_den_unr.mif

Outputs:
  derivatives/eddy/<series>_preproc.mif
  derivatives/qc/eddy_logs/<series>.log
  derivatives/qc/eddy_qc/<series>/
  derivatives/qc/eddy_summary.csv
  derivatives/qc/eddy_quarantine.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from queue import Empty, Queue
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional

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

        def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
            return None

        def set_postfix_str(self, s="", refresh=True):
            return None

        def set_description_str(self, desc):
            return None

        def refresh(self):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, *args, **kwargs):
        return _TqdmFallback(iterable=iterable, total=kwargs.get("total"))


DEFAULT_PATHS = resolve_pipeline_paths(create_layout=True)
DEFAULT_DERIV_ROOT = DEFAULT_PATHS.deriv_root
# Conservative defaults reduce concurrent exec/code-signing pressure on macOS
# while still allowing users to opt into higher parallelism explicitly.
DEFAULT_JOBS = min(2, max(1, (os.cpu_count() or 4) // 3))
DEFAULT_THREADS_PER_JOB = 1
DEFAULT_FORCE = False
DEFAULT_STAGE_ROOT = Path("/scratch/eddy_stage") if Path("/scratch").exists() else Path("/tmp/eddy_stage")
DEFAULT_PE_DIR = "auto"
DEFAULT_FALLBACK_PE_DIR = "j-"
DEFAULT_EDDY_OPTIONS = "--slm=linear --data_is_shelled"
DEFAULT_B0_THRESHOLD = 50.0
DEFAULT_SMALL_MASK_RETRY = True
DEFAULT_RETRY_MASK_DILATE_NPASS = 3
DEFAULT_RETRY_NVOXHP_MIN = 100
DEFAULT_RETRY_NVOXHP_CAP = 500
DEFAULT_HYPERPARAM_RETRY = True
DEFAULT_HYPERPARAM_OPTIONS = "--repol --ol_nstd=4 --nvoxhp=100 --niter=8 --fwhm=10,6,4,2,0,0,0,0"
DEFAULT_FINAL_RESCUE_RETRY = True
DEFAULT_FINAL_RESCUE_OPTIONS = (
    "--flm=linear --slm=none --repol --ol_nstd=4 "
    "--nvoxhp=100 --niter=8 --fwhm=10,6,4,2,0,0,0,0 --ff=10 --initrand=1"
)
DEFAULT_FINAL_RESCUE_THREADS = 1
DEFAULT_STRICT_HQ_RESCUE_RETRY = True
DEFAULT_STRICT_HQ_RESCUE_OPTIONS = (
    "--slm=linear --data_is_shelled --repol --ol_nstd=4 "
    "--nvoxhp=100 --niter=8 --fwhm=10,0,0,0,0,0,0,0 --ff=10 --dont_peas"
)
DEFAULT_STRICT_HQ_RESCUE_THREADS = 1
DEFAULT_QUARANTINE_ENABLE = True
DEFAULT_OVERWRITE_LOGS = True
DEFAULT_PERSIST_EDDY_QC_SUPPORT = True
DEFAULT_ALLOW_CPU_FALLBACK = False
DEFAULT_QC_ONLY_STAGE_ROOT = Path("/tmp/eddy_quad_stage")
VALID_PE_DIRS = {"i", "i-", "j", "j-", "k", "k-"}
VALID_GROUP_FILTERS = {"all", "ad", "mci", "normal", "cn"}
SUMMARY_HEADER = [
    "series",
    "input",
    "output",
    "status",
    "lane_reached",
    "vols_in",
    "vols_out",
    "input_stage",
    "preflight_status",
    "b0_reference_index",
    "usable_b0_indices",
    "sanitize_b0_indices",
    "b0_signal_summary",
    "dwi_signal_summary",
    "pe_dir_requested",
    "pe_dir_effective",
    "pe_dir_source",
    "header_pe_dir",
    "eddy_options",
    "threads_per_job",
    "tsnr_in",
    "tsnr_out",
    "tsnr_gain",
    "duration_seconds",
    "duration_elapsed",
    "notes",
    "log",
]
B0_PROVENANCE_HEADER = [
    "series",
    "input",
    "output",
    "status",
    "lane_reached",
    "input_stage",
    "vols_in",
    "b0_reference_index",
    "usable_b0_indices",
    "sanitize_b0_indices",
    "b0_signal_summary",
    "dwi_signal_summary",
    "pe_dir_requested",
    "pe_dir_effective",
    "pe_dir_source",
    "header_pe_dir",
    "eddy_options",
    "threads_per_job",
    "notes",
    "log",
]
QUARANTINE_HEADER = [
    "series",
    "failure_class",
    "lane_reached",
    "notes",
    "log_path",
]


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "?"
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _short_label(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return f"{text[: width - 3]}..."


def _estimate_remaining_seconds(
    *,
    active_states: List[Dict[str, object]],
    queued: int,
    jobs: int,
    avg_total_seconds: Optional[float],
) -> Optional[float]:
    if not active_states and queued <= 0:
        return 0.0

    slot_remaining: List[float] = []
    for state in active_states:
        eta = state.get("eta_seconds")
        if eta is None:
            if avg_total_seconds is None:
                continue
            eta = max(0.0, avg_total_seconds - float(state.get("elapsed_seconds", 0.0)))
        slot_remaining.append(max(0.0, float(eta)))

    worker_slots = max(1, jobs)
    if not slot_remaining:
        if avg_total_seconds is None:
            return None
        slot_remaining = [0.0] * worker_slots
    elif len(slot_remaining) < worker_slots:
        slot_remaining.extend([0.0] * (worker_slots - len(slot_remaining)))

    if avg_total_seconds is None:
        return max(slot_remaining) if queued <= 0 else None

    slot_remaining.sort()
    for _ in range(max(0, queued)):
        slot_remaining[0] += avg_total_seconds
        slot_remaining.sort()
    return max(slot_remaining) if slot_remaining else 0.0


class _LiveSeriesPanel:
    def __init__(self, max_slots: int, start_pos: int = 1):
        self.max_slots = max_slots
        self.start_pos = start_pos
        self.free_positions = list(range(start_pos, start_pos + max_slots))
        self.rows: Dict[str, tuple[object, int]] = {}
        self.states: Dict[str, Dict[str, object]] = {}
        self.completed_durations: List[float] = []

    def _ensure_row(self, series: str):
        if series in self.rows:
            return self.rows[series][0]
        pos = self.free_positions.pop(0) if self.free_positions else self.start_pos
        bar = tqdm(
            total=100,
            position=pos,
            leave=False,
            dynamic_ncols=True,
            bar_format="{desc:<54} {bar} {n:>3.0f}% | {postfix}",
        )
        self.rows[series] = (bar, pos)
        return bar

    def _estimate_total_seconds(
        self,
        state: Dict[str, object],
        now: float,
        *,
        fallback_total_seconds: Optional[float] = None,
    ) -> Optional[float]:
        started_at = float(state.get("started_at", now))
        elapsed = max(0.0, now - started_at)
        percent = max(0, min(100, int(state.get("percent", 0))))
        if percent >= 20:
            return elapsed / max(percent / 100.0, 0.2)
        return fallback_total_seconds

    def average_total_seconds(self, now: Optional[float] = None) -> Optional[float]:
        now = time.time() if now is None else now
        totals: List[float] = [value for value in self.completed_durations if value > 0]
        for state in self.states.values():
            total = self._estimate_total_seconds(state, now)
            if total is not None and total > 0:
                totals.append(total)
        if not totals:
            return None
        return sum(totals) / len(totals)

    def snapshot(self, now: Optional[float] = None) -> List[Dict[str, object]]:
        now = time.time() if now is None else now
        avg_total_seconds = self.average_total_seconds(now)
        states: List[Dict[str, object]] = []
        for series, state in sorted(self.states.items()):
            elapsed = max(0.0, now - float(state.get("started_at", now)))
            total = self._estimate_total_seconds(
                state,
                now,
                fallback_total_seconds=avg_total_seconds,
            )
            eta = None if total is None else max(0.0, total - elapsed)
            states.append(
                {
                    "series": series,
                    "status": str(state.get("status", "")),
                    "percent": int(state.get("percent", 0)),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                }
            )
        return states

    def update(self, series: str, percent: int, status: str) -> None:
        now = time.time()
        bar = self._ensure_row(series)
        try:
            target = max(0, min(100, int(percent)))
        except Exception:
            target = 0
        state = self.states.get(series, {"started_at": now})
        state["percent"] = target
        state["status"] = status
        state["last_update"] = now
        self.states[series] = state
        elapsed = max(0.0, now - float(state["started_at"]))
        avg_total_seconds = self.average_total_seconds(now)
        total = self._estimate_total_seconds(
            state,
            now,
            fallback_total_seconds=avg_total_seconds,
        )
        eta = None if total is None else max(0.0, total - elapsed)
        delta = target - getattr(bar, "n", 0)
        if delta > 0:
            bar.update(delta)
        elif delta < 0:
            bar.n = target
        bar.set_description_str(f"{_short_label(series, 24)} | {_short_label(status, 26)}")
        bar.set_postfix_str(
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
            refresh=False,
        )
        bar.refresh()

    def close(self, series: str) -> None:
        now = time.time()
        rec = self.rows.pop(series, None)
        state = self.states.pop(series, None)
        if state is not None:
            started_at = float(state.get("started_at", now))
            duration = max(0.0, now - started_at)
            if duration > 0:
                self.completed_durations.append(duration)
        if rec is None:
            return
        bar, pos = rec
        try:
            if getattr(bar, "n", 0) < 100:
                bar.update(100 - getattr(bar, "n", 0))
        except Exception:
            pass
        bar.close()
        if pos not in self.free_positions:
            self.free_positions.append(pos)
            self.free_positions.sort()


def _emit_progress(progress_queue: Optional[Queue], series: str, percent: int, status: str) -> None:
    if progress_queue is None:
        return
    progress_queue.put((series, max(0, min(100, int(percent))), status))


def _build_run_postfix(
    *,
    progress: Dict[str, int],
    live_panel: _LiveSeriesPanel,
    total_inputs: int,
    existing_outputs: int,
    jobs: int,
) -> str:
    active_states = live_panel.snapshot()
    avg_total_seconds = live_panel.average_total_seconds()
    eta_seconds = _estimate_remaining_seconds(
        active_states=active_states,
        queued=progress["queued"],
        jobs=jobs,
        avg_total_seconds=avg_total_seconds,
    )
    persisted_outputs = existing_outputs + progress["succeeded_this_run"]
    running_count = max(int(progress.get("running", 0)), len(active_states))
    pieces = [
        f"persisted={persisted_outputs}/{total_inputs}",
        f"running={running_count}",
        f"queued={progress['queued']}",
        f"ok={progress['succeeded_this_run']}",
        f"fail={progress['failed_this_run']}",
        f"eta={_format_duration(eta_seconds)}",
    ]
    if active_states:
        active_labels = [
            f"{_short_label(str(state['series']), 12)}:{_short_label(str(state['status']), 14)}"
            for state in active_states[:2]
        ]
        if len(active_states) > 2:
            active_labels.append(f"+{len(active_states) - 2}")
        pieces.append(f"active={'; '.join(active_labels)}")
    return " | ".join(pieces)


def _read_existing_summary(summary_csv: Path) -> Dict[str, Dict[str, object]]:
    if not summary_csv.exists():
        return {}
    rows: Dict[str, Dict[str, object]] = {}
    with summary_csv.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            series = row.get("series")
            if series:
                rows[series] = row
    return rows


def _visible_glob(folder: Path, pattern: str) -> Iterable[Path]:
    for path in sorted(folder.glob(pattern)):
        if path.name.startswith("._") or path.name.startswith("."):
            continue
        if not path.exists():
            continue
        yield path


INPUT_STAGE_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("unringed", "_den_unr.mif"),
    ("denoised", "_den.mif"),
    ("raw", ".mif"),
)
INPUT_STAGE_PRIORITY = {stage: rank for rank, (stage, _) in enumerate(INPUT_STAGE_SUFFIXES)}


def _series_from_input_path(path: Path) -> str:
    name = path.name
    for _, suffix in INPUT_STAGE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _series_from_unringed(path: Path) -> str:
    return _series_from_input_path(path)


def _input_stage_from_path(path: Path) -> str:
    name = path.name
    for stage, suffix in INPUT_STAGE_SUFFIXES:
        if name.endswith(suffix):
            return stage
    return "unknown"


def _fallback_input_candidates(path: Path) -> List[Path]:
    stage = _input_stage_from_path(path)
    current_rank = INPUT_STAGE_PRIORITY.get(stage)
    if current_rank is None:
        return []
    deriv_root = path.parent.parent
    series = _series_from_input_path(path)
    candidates = [
        deriv_root / "mif_unringed" / f"{series}_den_unr.mif",
        deriv_root / "mif_denoised" / f"{series}_den.mif",
        deriv_root / "mif_dwi" / f"{series}.mif",
    ]
    fallbacks: List[Path] = []
    for candidate in candidates:
        if candidate == path or not candidate.exists():
            continue
        candidate_stage = _input_stage_from_path(candidate)
        candidate_rank = INPUT_STAGE_PRIORITY.get(candidate_stage)
        if candidate_rank is None or candidate_rank <= current_rank:
            continue
        fallbacks.append(candidate)
    return fallbacks


def _subject_from_series(series: str) -> str:
    return series.split("_I", 1)[0] if "_I" in series else series


def _normalize_group_filter(group: Optional[str]) -> Optional[str]:
    if group is None:
        return None
    value = str(group).strip().lower()
    if not value or value == "all":
        return None
    if value == "ad":
        return "AD"
    if value == "mci":
        return "MCI"
    if value in {"normal", "cn"}:
        return "CN"
    raise ValueError(
        f"Invalid group filter: {group}. Expected one of: "
        f"{', '.join(sorted(VALID_GROUP_FILTERS))}"
    )


def _recode_group(value: str) -> str:
    value = str(value).strip()
    if value in {"EMCI", "LMCI", "SMC"}:
        return "MCI"
    return value


def _resolve_cohort_dti_csv(cohort_dti_csv: Optional[Path]) -> Path:
    candidates: List[Path] = []
    if cohort_dti_csv is not None:
        candidates.append(Path(cohort_dti_csv))
    candidates.extend(
        [
            Path.cwd() / "cohort" / "dti.csv",
            Path(__file__).resolve().parent / "cohort" / "dti.csv",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find cohort DTI CSV. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_group_map(cohort_dti_csv: Path) -> Dict[str, str]:
    group_map: Dict[str, str] = {}
    with cohort_dti_csv.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = (row.get("Subject ID") or row.get("subject_id") or "").strip()
            if not subject_id or subject_id in group_map:
                continue
            group = _recode_group(row.get("Research Group") or row.get("group") or "")
            if group:
                group_map[subject_id] = group
    return group_map


def _filter_series_paths_by_group(
    paths: List[Path],
    *,
    group: Optional[str],
    cohort_dti_csv: Optional[Path],
    series_name_fn,
) -> tuple[List[Path], Optional[str], Optional[Path]]:
    normalized_group = _normalize_group_filter(group)
    if normalized_group is None:
        return paths, None, None
    cohort_path = _resolve_cohort_dti_csv(cohort_dti_csv)
    group_map = _load_group_map(cohort_path)
    filtered = [
        path
        for path in paths
        if group_map.get(_subject_from_series(series_name_fn(path))) == normalized_group
    ]
    return filtered, normalized_group, cohort_path


def _ensure_tools() -> None:
    needed = [
        "mrconvert",
        "mrcat",
        "dwiextract",
        "mrinfo",
        "mrstats",
        "mrmath",
        "mrcalc",
        "dwi2mask",
        "maskfilter",
        "dwifslpreproc",
        "bet",
    ]
    missing = [tool for tool in needed if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"Missing required tools: {', '.join(missing)}")


def _run(
    cmd: List[str],
    *,
    env: Optional[dict] = None,
    cwd: Optional[Path] = None,
    log_file: Optional[Path] = None,
    debug: bool = False,
) -> subprocess.CompletedProcess:
    cmd_str = " ".join(map(str, cmd))
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a") as fp:
            fp.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S')}] > {cmd_str}\n")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
    )
    if log_file is not None:
        with log_file.open("a") as fp:
            if proc.stdout:
                fp.write(proc.stdout)
            if proc.stderr:
                fp.write(proc.stderr)
    if debug and (proc.stdout or proc.stderr):
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
        )
    return proc


def _run_live(
    cmd: List[str],
    *,
    env: Optional[dict] = None,
    cwd: Optional[Path] = None,
    log_file: Optional[Path] = None,
    debug: bool = False,
    line_callback=None,
) -> subprocess.CompletedProcess:
    cmd_str = " ".join(map(str, cmd))
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a") as fp:
            fp.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S')}] > {cmd_str}\n")

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_parts: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        output_parts.append(line)
        if log_file is not None:
            with log_file.open("a") as fp:
                fp.write(line)
        if line_callback is not None:
            line_callback(line)
        if debug:
            print(line, end="")

    proc.wait()
    output = "".join(output_parts)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=output, stderr=output
        )
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout=output, stderr=None)


def _job_env(
    tmpdir: Path,
    threads_per_job: int,
    *,
    allow_cpu_fallback: bool = DEFAULT_ALLOW_CPU_FALLBACK,
) -> dict:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(max(1, int(threads_per_job)))
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TMPDIR"] = str(tmpdir)
    env["MRTRIX_TMPFILE_DIR"] = str(tmpdir)
    if allow_cpu_fallback:
        env.pop("DWIFSLPREPROC_NO_CPU_FALLBACK", None)
    else:
        # Make CUDA failures visible instead of silently dropping to CPU eddy.
        env["DWIFSLPREPROC_NO_CPU_FALLBACK"] = "1"
    return env


def _nvol(path: Path, env: Optional[dict] = None) -> int:
    out = subprocess.check_output(
        ["mrinfo", "-size", str(path)],
        text=True,
        env=env,
    ).split()
    return int(out[3]) if len(out) >= 4 else 0


def _sizes(path: Path, env: Optional[dict] = None) -> List[int]:
    out = subprocess.check_output(
        ["mrinfo", "-size", str(path)],
        text=True,
        env=env,
    ).split()
    return [int(x) for x in out]


def _header_pe_dir(path: Path, env: Optional[dict] = None) -> Optional[str]:
    out = subprocess.check_output(
        ["mrinfo", str(path)],
        text=True,
        env=env,
        stderr=subprocess.PIPE,
    )
    for line in out.splitlines():
        if "PhaseEncodingDirection:" not in line:
            continue
        value = line.split(":", 1)[1].strip()
        return value if value in VALID_PE_DIRS else None
    return None


def _resolve_pe_dir(path: Path, configured_pe_dir: str, env: Optional[dict] = None) -> tuple[str, str, Optional[str]]:
    header_pe = _header_pe_dir(path, env=env)
    requested = str(configured_pe_dir or "").strip()
    if requested.lower() == "auto":
        if header_pe is not None:
            return header_pe, "header", header_pe
        return DEFAULT_FALLBACK_PE_DIR, "default", header_pe
    return requested or DEFAULT_FALLBACK_PE_DIR, "configured", header_pe


def _std(path: Path, env: Optional[dict] = None) -> float:
    out = subprocess.check_output(
        ["mrstats", "-output", "std", str(path)],
        text=True,
        env=env,
    ).split()
    return float(out[0]) if out else 0.0


def _count_nonzero(path: Path, env: Optional[dict] = None) -> int:
    out = subprocess.check_output(
        ["mrstats", "-output", "count", "-ignorezero", str(path)],
        text=True,
        env=env,
        stderr=subprocess.PIPE,
    ).split()
    return int(float(out[0])) if out else 0


def _export_bvals(path: Path, workdir: Path, env: dict) -> List[float]:
    bvec = workdir / "tmp_bvecs"
    bval = workdir / "tmp_bvals"
    for target in (bvec, bval):
        try:
            target.unlink()
        except FileNotFoundError:
            pass
    subprocess.run(
        ["mrinfo", str(path), "-export_grad_fsl", str(bvec), str(bval)],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    return [float(x) for x in bval.read_text().split()]


def _extract_volume(path: Path, index: int, out_path: Path, env: dict) -> None:
    subprocess.run(
        ["mrconvert", str(path), "-coord", "3", str(index), str(out_path), "-quiet", "-force"],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )


def _volume_signal_stats(path: Path, index: int, workdir: Path, env: dict) -> tuple[int, float]:
    vol = workdir / f"vol_{index}.mif"
    _extract_volume(path, index, vol, env)
    nonzero = _count_nonzero(vol, env=env)
    mean_val = 0.0
    if nonzero > 0:
        out = subprocess.check_output(
            ["mrstats", "-output", "mean", str(vol)],
            text=True,
            env=env,
            stderr=subprocess.PIPE,
        ).split()
        mean_val = float(out[0]) if out else 0.0
    return nonzero, mean_val


def _volume_nonzero_count(path: Path, index: int, workdir: Path, env: dict) -> int:
    vol = workdir / f"vol_{index}.mif"
    _extract_volume(path, index, vol, env)
    return _count_nonzero(vol, env=env)


def _volume_mean(path: Path, index: int, workdir: Path, env: dict) -> float:
    vol = workdir / f"vol_{index}.mif"
    _extract_volume(path, index, vol, env)
    out = subprocess.check_output(
        ["mrstats", "-output", "mean", str(vol)],
        text=True,
        env=env,
        stderr=subprocess.PIPE,
    ).split()
    return float(out[0]) if out else 0.0


def _inspect_b0_signal(
    path: Path,
    *,
    workdir: Path,
    env: dict,
    b0_threshold: float,
) -> Dict[str, object]:
    bvals = _export_bvals(path, workdir, env)
    b0_indices = [i for i, value in enumerate(bvals) if abs(value) < b0_threshold]
    stats: List[Dict[str, object]] = []
    for idx in b0_indices:
        nonzero, mean_val = _volume_signal_stats(path, idx, workdir, env)
        stats.append(
            {
                "index": idx,
                "nonzero": nonzero,
                "mean": mean_val,
            }
        )

    usable = [entry for entry in stats if int(entry["nonzero"]) > 0 and float(entry["mean"]) > 0.0]
    reference = max(usable, key=lambda item: (int(item["nonzero"]), float(item["mean"]))) if usable else None
    usable_b0_indices: List[int] = []
    sanitize_b0_indices: List[int] = []
    if reference is not None:
        ref_mean = max(float(reference["mean"]), 1e-12)
        min_mean = max(1e-9, ref_mean * 0.01)
        for entry in stats:
            idx = int(entry["index"])
            nonzero = int(entry["nonzero"])
            mean_val = float(entry["mean"])
            if nonzero > 0 and mean_val >= min_mean:
                usable_b0_indices.append(idx)
            elif idx != int(reference["index"]):
                sanitize_b0_indices.append(idx)

    return {
        "bvals": bvals,
        "b0_indices": b0_indices,
        "stats": stats,
        "reference_b0": None if reference is None else int(reference["index"]),
        "usable_b0_indices": usable_b0_indices,
        "sanitize_b0_indices": sanitize_b0_indices,
        "all_b0_zero": bool(b0_indices) and reference is None,
    }


def _sample_dwi_signal(
    path: Path,
    *,
    workdir: Path,
    env: dict,
    b0_threshold: float,
    bvals: Optional[List[float]] = None,
    max_samples: int = 4,
) -> Dict[str, object]:
    if bvals is None:
        bvals = _export_bvals(path, workdir, env)
    dwi_indices = [i for i, value in enumerate(bvals) if abs(value) >= b0_threshold][: max(1, int(max_samples))]
    stats: List[Dict[str, object]] = []
    for idx in dwi_indices:
        nonzero, mean_val = _volume_signal_stats(path, idx, workdir, env)
        stats.append(
            {
                "index": idx,
                "nonzero": nonzero,
                "mean": mean_val,
            }
        )
    nonzero_indices = [int(entry["index"]) for entry in stats if int(entry["nonzero"]) > 0 and float(entry["mean"]) > 0.0]
    return {
        "indices": dwi_indices,
        "stats": stats,
        "nonzero_indices": nonzero_indices,
    }


def _assess_preflight_candidate(
    input_path: Path,
    *,
    workdir: Path,
    env: dict,
    b0_threshold: float,
) -> Dict[str, object]:
    sizes = _sizes(input_path, env=env)
    result: Dict[str, object] = {
        "input_path": input_path,
        "series": _series_from_input_path(input_path),
        "stage": _input_stage_from_path(input_path),
        "sizes": sizes,
        "vols_in": sizes[3] if len(sizes) >= 4 else "",
        "status": "unknown",
        "note": "",
        "b0_reference_index": None,
        "sanitize_b0_indices": [],
        "usable_b0_indices": [],
        "b0_signal_summary": [],
        "dwi_signal_summary": [],
    }

    if len(sizes) < 4 or any(dim <= 1 for dim in sizes[:3]):
        result["status"] = "bad_shape"
        result["note"] = f"shape={sizes}"
        return result

    if int(sizes[3]) <= 1:
        result["status"] = "bad_nvol"
        result["note"] = f"nvol={sizes[3]}"
        return result

    b0_inspection = _inspect_b0_signal(
        input_path,
        workdir=workdir,
        env=env,
        b0_threshold=b0_threshold,
    )
    dwi_signal = _sample_dwi_signal(
        input_path,
        workdir=workdir,
        env=env,
        b0_threshold=b0_threshold,
        bvals=list(b0_inspection["bvals"]),
    )

    result["b0_reference_index"] = b0_inspection["reference_b0"]
    result["sanitize_b0_indices"] = list(b0_inspection["sanitize_b0_indices"])
    result["usable_b0_indices"] = list(b0_inspection["usable_b0_indices"])
    result["b0_signal_summary"] = list(b0_inspection["stats"])
    result["dwi_signal_summary"] = list(dwi_signal["stats"])

    if b0_inspection["all_b0_zero"]:
        if dwi_signal["nonzero_indices"]:
            sampled = ",".join(str(idx) for idx in dwi_signal["nonzero_indices"])
            result["note"] = f"all_b0_blank; sampled_nonzero_dwi={sampled}"
        else:
            result["note"] = "no valid nonzero b0 found; sampled diffusion volumes blank"
        result["status"] = "b0_unrecoverable"
        return result

    if dwi_signal["indices"] and not dwi_signal["nonzero_indices"]:
        sampled = ",".join(str(idx) for idx in dwi_signal["indices"])
        result["note"] = f"sampled diffusion volumes blank: {sampled}"
        result["status"] = "dwi_signal_missing"
        return result

    result["status"] = "ready"
    return result


def _find_viable_preflight_fallback(
    input_path: Path,
    *,
    workdir: Path,
    env: dict,
    b0_threshold: float,
) -> tuple[Optional[Dict[str, object]], List[Dict[str, object]]]:
    attempts: List[Dict[str, object]] = []
    for candidate in _fallback_input_candidates(input_path):
        assessment = _assess_preflight_candidate(
            candidate,
            workdir=workdir,
            env=env,
            b0_threshold=b0_threshold,
        )
        attempts.append(assessment)
        if assessment["status"] == "ready":
            return assessment, attempts
    return None, attempts


def _build_volume_stack(path: Path, indices: List[int], out_path: Path, *, workdir: Path, env: dict) -> Path:
    if not indices:
        raise ValueError("No volume indices provided for stack construction")
    segments: List[Path] = []
    for order, idx in enumerate(indices):
        segment = workdir / f"{out_path.stem}_seg_{order:03d}_v{idx}.mif"
        _extract_volume(path, idx, segment, env)
        segments.append(segment)
    if len(segments) == 1:
        shutil.copy2(segments[0], out_path)
        return out_path
    subprocess.run(
        ["mrcat", *map(str, segments), "-axis", "3", str(out_path), "-quiet", "-force"],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    return out_path


def _replace_volumes_with_reference(
    path: Path,
    *,
    reference_index: int,
    replace_indices: List[int],
    workdir: Path,
    env: dict,
) -> Path:
    replace_set = {int(idx) for idx in replace_indices if int(idx) != int(reference_index)}
    if not replace_set:
        return path

    reference_volume = workdir / f"replace_ref_v{reference_index}.mif"
    _extract_volume(path, int(reference_index), reference_volume, env)
    segments: List[Path] = []
    for idx in range(_nvol(path, env=env)):
        if idx in replace_set:
            segment = workdir / f"replace_v{idx}.mif"
            shutil.copy2(reference_volume, segment)
        else:
            segment = workdir / f"keep_v{idx}.mif"
            _extract_volume(path, idx, segment, env)
        segments.append(segment)

    out_path = workdir / "sanitized_b0s.mif"
    subprocess.run(
        ["mrcat", *map(str, segments), "-axis", "3", str(out_path), "-quiet", "-force"],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    return out_path


def _find_replacement_b0_index(
    path: Path,
    *,
    workdir: Path,
    env: dict,
    b0_threshold: float,
) -> Optional[int]:
    bvals = _export_bvals(path, workdir, env)
    b0_indices = [i for i, value in enumerate(bvals) if abs(value) < b0_threshold]
    if not b0_indices:
        return None

    first_b0 = b0_indices[0]
    if _volume_nonzero_count(path, first_b0, workdir, env) > 0:
        return None

    best_idx: Optional[int] = None
    best_key: Optional[tuple[int, float]] = None
    for idx in b0_indices[1:]:
        nonzero = _volume_nonzero_count(path, idx, workdir, env)
        if nonzero <= 0:
            continue
        mean_val = _volume_mean(path, idx, workdir, env)
        score = (nonzero, mean_val)
        if best_key is None or score > best_key:
            best_key = score
            best_idx = idx
    return best_idx if best_idx is not None else -1


def _swap_volume_to_front(path: Path, index: int, workdir: Path, env: dict) -> Path:
    out_path = workdir / "reordered.mif"
    v0 = workdir / "swap_v0.mif"
    vx = workdir / f"swap_v{index}.mif"
    segments = [vx]

    _extract_volume(path, 0, v0, env)
    _extract_volume(path, index, vx, env)

    if index > 1:
        mid = workdir / "swap_mid.mif"
        subprocess.run(
            ["mrconvert", str(path), "-coord", "3", f"1:{index-1}", str(mid), "-quiet", "-force"],
            check=True,
            env=env,
            text=True,
            capture_output=True,
        )
        segments.append(mid)

    segments.append(v0)

    nvol = _nvol(path, env=env)
    if index < nvol - 1:
        tail = workdir / "swap_tail.mif"
        subprocess.run(
            ["mrconvert", str(path), "-coord", "3", f"{index+1}:end", str(tail), "-quiet", "-force"],
            check=True,
            env=env,
            text=True,
            capture_output=True,
        )
        segments.append(tail)

    subprocess.run(
        ["mrcat", *map(str, segments), "-axis", "3", str(out_path), "-quiet", "-force"],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    return out_path


def _median_tsnr(path: Path, workdir: Path, env: dict) -> Optional[float]:
    with tempfile.TemporaryDirectory(dir=workdir) as td_str:
        td = Path(td_str)
        mean_mif = td / "mean.mif"
        std_mif = td / "std.mif"
        nz_std = td / "nz_std.mif"
        tsnr = td / "tsnr.mif"
        mask = td / "mask.mif"

        _run(["mrmath", str(path), "mean", str(mean_mif), "-axis", "3", "-quiet", "-force"], env=env, cwd=workdir)
        _run(["mrmath", str(path), "std", str(std_mif), "-axis", "3", "-quiet", "-force"], env=env, cwd=workdir)
        _run(["mrcalc", str(std_mif), "0", "-gt", str(std_mif), "1", "-if", str(nz_std), "-quiet", "-force"], env=env, cwd=workdir)
        _run(["mrcalc", str(mean_mif), str(nz_std), "-div", str(tsnr), "-quiet", "-force"], env=env, cwd=workdir)
        _run(["mrcalc", str(mean_mif), "0", "-gt", str(mask), "-quiet", "-force"], env=env, cwd=workdir)
        if _count_nonzero(mask, env=env) == 0:
            return None
        out = subprocess.check_output(
            ["mrstats", str(tsnr), "-mask", str(mask), "-output", "median"],
            text=True,
            env=env,
            cwd=str(workdir),
            stderr=subprocess.PIPE,
        ).split()
        return float(out[0]) if out else None


def _build_bet_mask_from_b0(
    dwi_path: Path,
    *,
    workdir: Path,
    env: dict,
    log_file: Path,
    debug: bool,
    b0_threshold: float = DEFAULT_B0_THRESHOLD,
) -> tuple[Path, int]:
    b0_stack = workdir / "eddy_b0_stack.mif"
    b0_mean = workdir / "eddy_b0_mean.nii.gz"
    brain_prefix = workdir / "eddy_b0_brain"
    mask_path = workdir / "eddy_b0_brain_mask.nii.gz"
    for path in (b0_stack, b0_mean, mask_path, Path(str(brain_prefix) + ".nii.gz")):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    b0_inspection = _inspect_b0_signal(
        dwi_path,
        workdir=workdir,
        env=env,
        b0_threshold=b0_threshold,
    )
    usable_b0_indices = list(b0_inspection["usable_b0_indices"]) or list(b0_inspection["b0_indices"])
    if not usable_b0_indices:
        raise RuntimeError("No usable b0 volumes available for BET mask construction")
    if len(usable_b0_indices) == len(list(b0_inspection["b0_indices"])):
        _run(["dwiextract", str(dwi_path), str(b0_stack), "-bzero", "-quiet", "-force"], env=env, cwd=workdir, log_file=log_file, debug=debug)
    else:
        _append_log_note(log_file, f"Building BET mask from usable b0 volumes only: {usable_b0_indices}")
        _build_volume_stack(dwi_path, usable_b0_indices, b0_stack, workdir=workdir, env=env)
    _run(["mrmath", str(b0_stack), "mean", str(b0_mean), "-axis", "3", "-quiet", "-force"], env=env, cwd=workdir, log_file=log_file, debug=debug)
    _run(["bet", str(b0_mean), str(brain_prefix), "-m", "-f", "0.2"], env=env, cwd=workdir, log_file=log_file, debug=debug)
    return mask_path, _count_nonzero(mask_path, env=env)


def _merge_eddy_options(base: str, extras: List[str], *, replace: bool = False) -> str:
    ordered_keys: List[str] = []
    token_map: Dict[str, str] = {}

    def add_token(token: str, *, force: bool) -> None:
        if not token:
            return
        key = token.split("=", 1)[0]
        if key not in token_map:
            ordered_keys.append(key)
            token_map[key] = token
            return
        if force:
            token_map[key] = token

    for token in base.split():
        add_token(token, force=False)
    for token in extras:
        add_token(token, force=replace)

    return " ".join(token_map[key] for key in ordered_keys if token_map.get(key))


def _parse_dwifslpreproc_progress(line: str) -> Optional[tuple[int, str]]:
    text = line.strip()
    if not text:
        return None
    if "Command:" in text and "dwi2mask" in text:
        return 8, "building mask"
    if "Command:" in text and "-export_grad_fsl" in text:
        return 12, "preparing eddy inputs"
    if "Command:" in text and "eddy_quad" in text:
        return 88, "running eddy_quad"
    if "Command:" in text and "eddy_" in text:
        return 15, "running eddy"
    iter_match = re.search(r"Iter:\s*(\d+),", text)
    if iter_match:
        iter_idx = min(4, max(0, int(iter_match.group(1))))
        return 20 + int(((iter_idx + 1) / 5.0) * 55), f"eddy iter {iter_idx + 1}/5"
    if "mrconvert dwi_post_eddy.nii.gz" in text:
        return 82, "converting output"
    if "Command:" in text and "mrconvert result.mif" in text:
        return 86, "packing result"
    if "Changing back to original directory" in text or "Deleting scratch directory" in text:
        return 95, "finalizing"
    return None


def _persist_eddy_qc_support_files(scratch: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    must_copy = [
        "eddy_config.txt",
        "eddy_indices.txt",
        "eddy_mask.nii",
        "bvals",
    ]
    for name in must_copy:
        src = scratch / name
        if src.exists():
            shutil.copy2(src, target_dir / name)

    for src in scratch.glob("dwi_post_eddy*"):
        if not src.is_file():
            continue
        # Avoid duplicating the corrected 4D image; it can be recreated later
        # from the persisted .mif output for eddy_quad.
        if src.name.endswith(".nii.gz"):
            continue
        shutil.copy2(src, target_dir / src.name)


def _persist_passthrough_output(
    source_path: Path,
    out_path: Path,
    row: Dict[str, object],
    *,
    lane: str,
    note: str,
    env: dict,
) -> Dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, out_path)
    row["output"] = out_path.name
    row["vols_out"] = _nvol(out_path, env=env)
    row["status"] = "ok"
    row["lane_reached"] = lane
    _append_note(row, note)
    return row


def _estimate_retry_nvoxhp(mask_voxels: int, floor: int, cap: int) -> Optional[int]:
    if mask_voxels <= floor:
        return None
    target = max(floor, mask_voxels // 10)
    target = min(cap, target)
    target = min(target, mask_voxels - 1)
    if target < floor:
        return None
    return target


def _nvoxhp_retry_candidates(
    mask_voxels: int,
    *,
    primary: Optional[int] = None,
    minimum: int = DEFAULT_RETRY_NVOXHP_MIN,
) -> List[int]:
    if mask_voxels <= 1:
        return []
    max_allowed = max(1, int(mask_voxels) - 1)
    ordered: List[int] = []
    raw_values: List[int] = []
    if primary is not None:
        raw_values.append(max(1, int(primary)))
    raw_values.extend([500, 250, 100])
    for value in raw_values:
        bounded = min(max_allowed, max(1, int(value)))
        if bounded < max(1, int(minimum)):
            continue
        if bounded not in ordered:
            ordered.append(bounded)
    return ordered


def _adaptive_gpu_nvoxhp_candidates(mask_voxels: int) -> List[int]:
    return _nvoxhp_retry_candidates(
        mask_voxels,
        primary=DEFAULT_RETRY_NVOXHP_MIN,
        minimum=DEFAULT_RETRY_NVOXHP_MIN,
    )


def _build_eddy_mask(
    dwi_path: Path,
    *,
    workdir: Path,
    env: dict,
    log_file: Path,
    debug: bool,
    b0_threshold: float,
    dilate_npass: int,
    min_mask_voxels: int,
) -> tuple[Path, int, str]:
    mask0 = workdir / "eddy_mask0.mif"
    mask1 = workdir / "eddy_mask1.mif"
    mask_final = workdir / "eddy_mask.mif"
    _run(["dwi2mask", str(dwi_path), str(mask0), "-quiet", "-force"], env=env, cwd=workdir, log_file=log_file, debug=debug)
    _run(
        ["maskfilter", str(mask0), "dilate", str(mask1), "-npass", str(max(1, int(dilate_npass))), "-force"],
        env=env,
        cwd=workdir,
        log_file=log_file,
        debug=debug,
    )
    _run(["maskfilter", str(mask1), "clean", str(mask_final), "-force"], env=env, cwd=workdir, log_file=log_file, debug=debug)
    mask_voxels = _count_nonzero(mask_final, env=env)
    bet_mask, bet_voxels = _build_bet_mask_from_b0(
        dwi_path,
        workdir=workdir,
        env=env,
        log_file=log_file,
        debug=debug,
        b0_threshold=b0_threshold,
    )
    if bet_voxels > mask_voxels:
        return bet_mask, bet_voxels, "bet_b0"
    return mask_final, mask_voxels, "dwi2mask"


def _failure_text(exc: Optional[subprocess.CalledProcessError], log_path: Path) -> str:
    parts = []
    if exc is not None:
        if exc.stderr:
            parts.append(exc.stderr)
        if exc.output:
            parts.append(exc.output)
    if log_path.exists():
        parts.append(log_path.read_text(errors="ignore"))
    return "\n".join(parts)


def _should_retry_small_mask(exc: subprocess.CalledProcessError, log_path: Path) -> bool:
    text = _failure_text(exc, log_path)
    patterns = [
        r"rnvox greater than number of non-zero voxels in mask",
        r"empty mask image",
        r"mean_of_first_b0: Zero mean",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _should_retry_hyperparams(exc: Optional[subprocess.CalledProcessError], log_path: Path) -> bool:
    text = _failure_text(exc, log_path)
    patterns = [
        r"Unable to find valid hyperparameters",
        r"solve\(\): solution not found",
        r"system is singular",
        r"rnvox greater than number of non-zero voxels in mask",
        r"DataSelector::common_constructor",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _classify_final_rescue_failure(exc: Optional[subprocess.CalledProcessError], log_path: Path) -> Optional[str]:
    text = _failure_text(exc, log_path)
    if re.search(r"Unable to find valid hyperparameters", text, re.IGNORECASE) or re.search(
        r"rnvox greater than number of non-zero voxels in mask",
        text,
        re.IGNORECASE,
    ) or re.search(r"DataSelector::common_constructor", text, re.IGNORECASE):
        return "hyperparams"
    if re.search(r"solve\(\): solution not found", text, re.IGNORECASE) or re.search(
        r"system is singular",
        text,
        re.IGNORECASE,
    ):
        return "singular"
    if re.search(r"cudaErrorIllegalAddress", text, re.IGNORECASE) or re.search(
        r"illegal memory access was encountered",
        text,
        re.IGNORECASE,
    ) or re.search(r"thrust::system::system_error", text, re.IGNORECASE):
        return "gpu_runtime"
    return None


def _append_log_note(log_file: Path, message: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a") as fp:
        fp.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}\n")


def _resolved_executable(name: str, env: Optional[dict] = None) -> str:
    search_path = None if env is None else env.get("PATH")
    resolved = shutil.which(name, path=search_path)
    if not resolved:
        return "NOT FOUND"
    try:
        return str(Path(resolved).resolve())
    except OSError:
        return resolved


def _append_note(row: Dict[str, object], message: str) -> None:
    note = str(row.get("notes", "")).strip()
    row["notes"] = message if not note else f"{note}; {message}"


def _json_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, dict)):
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:
            return str(value)
    return str(value)


def _series_runtime_line(row: Dict[str, object]) -> str:
    series = str(row.get("series", "unknown"))
    elapsed = str(row.get("duration_elapsed", "")).strip() or "?"
    status = str(row.get("status", "")).strip() or "unknown"
    lane = str(row.get("lane_reached", "")).strip() or "n/a"
    return f"{series} = {elapsed} | status={status} | lane={lane}"


def _append_runtime_note(runtime_log: Path, line: str) -> None:
    runtime_log.parent.mkdir(parents=True, exist_ok=True)
    with runtime_log.open("a") as fp:
        fp.write(f"{line}\n")


def _stamp_row_duration(row: Dict[str, object], started_at: float, ended_at: Optional[float] = None) -> Dict[str, object]:
    elapsed_seconds = max(0.0, (time.time() if ended_at is None else ended_at) - started_at)
    row["duration_seconds"] = f"{elapsed_seconds:.3f}"
    row["duration_elapsed"] = _format_duration(elapsed_seconds)
    return row


def _base_row(series: str, input_path: Path, log_path: Path) -> Dict[str, object]:
    return {
        "series": series,
        "input": input_path.name,
        "output": "",
        "status": "",
        "lane_reached": "",
        "vols_in": "",
        "vols_out": "",
        "input_stage": "",
        "preflight_status": "",
        "b0_reference_index": "",
        "usable_b0_indices": "",
        "sanitize_b0_indices": "",
        "b0_signal_summary": "",
        "dwi_signal_summary": "",
        "pe_dir_requested": "",
        "pe_dir_effective": "",
        "pe_dir_source": "",
        "header_pe_dir": "",
        "eddy_options": "",
        "threads_per_job": "",
        "tsnr_in": "",
        "tsnr_out": "",
        "tsnr_gain": "",
        "duration_seconds": "",
        "duration_elapsed": "",
        "notes": "",
        "log": str(log_path),
    }


def _existing_row(input_path: Path, out_path: Path, log_path: Path) -> Dict[str, object]:
    series = _series_from_unringed(input_path)
    row = _base_row(series, input_path, log_path)
    row["status"] = "skip_exists"
    row["lane_reached"] = "existing"
    row["output"] = out_path.name
    try:
        row["vols_out"] = _nvol(out_path)
    except Exception:
        pass
    return row


def _quarantine_row(
    row: Dict[str, object],
    status: str,
    *,
    note: Optional[str] = None,
    lane: Optional[str] = None,
) -> Dict[str, object]:
    row["status"] = status
    row["lane_reached"] = lane or str(row.get("lane_reached", "")).strip() or "lane_d_quarantine"
    if note:
        _append_note(row, note)
    return row


def _preserve_failure_stage(
    stage_dir: Path,
    *,
    debug_root: Path,
    series: str,
    log_path: Path,
) -> Optional[Path]:
    if not stage_dir.exists():
        return None
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    dest = debug_root / series / timestamp
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(stage_dir, dest)
    _append_log_note(log_path, f"Preserved failure stage directory: {dest}")
    return dest


def _is_valid_output(path: Path, expected_vols: int, env: dict) -> bool:
    if not path.exists():
        return False
    try:
        if _nvol(path, env=env) != expected_vols:
            return False
        if _std(path, env=env) == 0.0:
            return False
    except Exception:
        return False
    return True


def _build_dwifslpreproc_cmd(
    *,
    stage_dwi: Path,
    stage_out: Path,
    pe_dir: str,
    eddy_options: str,
    threads_per_job: int,
    scratch: Path,
    stage_qc: Path,
    run_eddy_qc: bool,
    eddy_mask: Optional[Path] = None,
) -> List[str]:
    # dwifslpreproc does not automatically mirror its thread setting to the
    # underlying eddy executable, so inject FSL's --nthr unless the caller has
    # already set it explicitly in eddy_options.
    effective_eddy_options = _merge_eddy_options(
        eddy_options,
        [f"--nthr={max(1, int(threads_per_job))}"],
        replace=False,
    )
    cmd = [
        "dwifslpreproc",
        str(stage_dwi),
        str(stage_out),
        "-rpe_none",
        "-pe_dir",
        pe_dir,
        f"-eddy_options={effective_eddy_options}",
        "-nthreads",
        str(max(1, int(threads_per_job))),
        "-scratch",
        str(scratch),
        "-force",
    ]
    if eddy_mask is not None:
        cmd.extend(["-eddy_mask", str(eddy_mask)])
    if run_eddy_qc:
        cmd.extend(["-eddyqc_all", str(stage_qc)])
    return cmd


def _preflight_input(
    input_path: Path,
    *,
    log_path: Path,
    stage_root: Path,
    threads_per_job: int,
    b0_threshold: float,
    allow_input_fallback: bool,
    overwrite_logs: bool,
    debug: bool,
) -> Dict[str, object]:
    series = _series_from_input_path(input_path)
    row = _base_row(series, input_path, log_path)
    plan: Dict[str, object] = {
        "series": series,
        "input_path": input_path,
        "row": row,
        "replacement_b0": None,
        "b0_reference_index": None,
        "sanitize_b0_indices": [],
        "runnable": False,
        "preflight_status": "unknown",
    }

    if overwrite_logs and log_path.exists():
        log_path.unlink()

    with tempfile.TemporaryDirectory(dir=stage_root) as td_str:
        td = Path(td_str)
        env = _job_env(td, threads_per_job)
        try:
            assessment = _assess_preflight_candidate(
                input_path,
                workdir=td,
                env=env,
                b0_threshold=b0_threshold,
            )

            if allow_input_fallback and assessment["status"] in {"b0_unrecoverable", "dwi_signal_missing"}:
                fallback_assessment, fallback_attempts = _find_viable_preflight_fallback(
                    input_path,
                    workdir=td,
                    env=env,
                    b0_threshold=b0_threshold,
                )
                if fallback_attempts:
                    attempts_msg = " | ".join(
                        f"{item['stage']}:{item['status']}:{Path(item['input_path']).name}"
                        for item in fallback_attempts
                    )
                    _append_log_note(log_path, f"Preflight fallback attempts: {attempts_msg}")
                if fallback_assessment is not None:
                    original_stage = str(assessment["stage"])
                    fallback_stage = str(fallback_assessment["stage"])
                    fallback_path = Path(fallback_assessment["input_path"])
                    plan["input_path"] = fallback_path
                    row["input"] = fallback_path.name
                    _append_note(row, f"input_fallback={original_stage}->{fallback_stage}")
                    _append_note(row, f"input_origin={input_path.name}")
                    _append_log_note(
                        log_path,
                        "Preflight selected fallback input "
                        f"{fallback_path.name} ({fallback_stage}) after "
                        f"{input_path.name} ({original_stage}) failed with {assessment['status']}",
                    )
                    assessment = fallback_assessment

            row["vols_in"] = assessment.get("vols_in", "")
            row["input_stage"] = assessment.get("stage", "")
            row["preflight_status"] = assessment.get("status", "")
            row["b0_reference_index"] = assessment.get("b0_reference_index", "")
            row["usable_b0_indices"] = _json_cell(assessment.get("usable_b0_indices", []))
            row["sanitize_b0_indices"] = _json_cell(assessment.get("sanitize_b0_indices", []))
            row["b0_signal_summary"] = _json_cell(assessment.get("b0_signal_summary", []))
            row["dwi_signal_summary"] = _json_cell(assessment.get("dwi_signal_summary", []))
            row["threads_per_job"] = threads_per_job
            plan["b0_reference_index"] = assessment.get("b0_reference_index")
            plan["sanitize_b0_indices"] = list(assessment.get("sanitize_b0_indices", []))

            status = str(assessment["status"])
            note = str(assessment.get("note", "")).strip()
            if status == "bad_shape":
                _append_log_note(log_path, f"Preflight rejected: invalid spatial shape {assessment['sizes']}")
                plan["preflight_status"] = "bad_shape"
                plan["row"] = _quarantine_row(row, "fail_preflight_shape", note=note)
                return plan

            if status == "bad_nvol":
                _append_log_note(log_path, f"Preflight rejected: invalid volume count {assessment['vols_in']}")
                plan["preflight_status"] = "bad_nvol"
                plan["row"] = _quarantine_row(row, "fail_preflight_nvol", note=note)
                return plan

            if status == "b0_unrecoverable":
                _append_log_note(log_path, f"Preflight rejected: {note}")
                plan["preflight_status"] = "b0_unrecoverable"
                plan["row"] = _quarantine_row(row, "fail_preflight_b0_zero", note=note)
                return plan

            if status == "dwi_signal_missing":
                _append_log_note(log_path, f"Preflight rejected: {note}")
                plan["preflight_status"] = "dwi_signal_missing"
                plan["row"] = _quarantine_row(row, "fail_preflight_dwi_zero", note=note)
                return plan

            if plan["sanitize_b0_indices"] and plan["b0_reference_index"] is not None:
                sanitize_list = ",".join(str(idx) for idx in plan["sanitize_b0_indices"])
                _append_note(row, f"b0_sanitize_ref={plan['b0_reference_index']}")
                _append_log_note(
                    log_path,
                    "Preflight prepared b0 sanitization using reference volume "
                    f"{plan['b0_reference_index']} -> [{sanitize_list}]",
                )

            plan["preflight_status"] = "ready"
            plan["runnable"] = True
            return plan

        except Exception as exc:
            _append_log_note(log_path, f"Preflight exception: {type(exc).__name__}: {exc}")
            plan["preflight_status"] = "bad_preflight"
            plan["row"] = _quarantine_row(row, "fail_preflight_exception", note=f"{type(exc).__name__}: {exc}")
            return plan


def _write_summary(rows: List[Dict[str, object]], summary_csv: Path) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=SUMMARY_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_HEADER})


def _write_b0_provenance(rows: List[Dict[str, object]], provenance_csv: Path) -> None:
    provenance_csv.parent.mkdir(parents=True, exist_ok=True)
    with provenance_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=B0_PROVENANCE_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in B0_PROVENANCE_HEADER})


def _write_quarantine(rows: List[Dict[str, object]], quarantine_csv: Path) -> None:
    quarantine_csv.parent.mkdir(parents=True, exist_ok=True)
    with quarantine_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=QUARANTINE_HEADER)
        writer.writeheader()
        for row in rows:
            status = str(row.get("status", ""))
            if status in {"ok", "skip_exists", "ready_for_eddy"}:
                continue
            writer.writerow(
                {
                    "series": row.get("series", ""),
                    "failure_class": status,
                    "lane_reached": row.get("lane_reached", ""),
                    "notes": row.get("notes", ""),
                    "log_path": row.get("log", ""),
                }
            )


def _flush_live_reports(
    rows: List[Dict[str, object]],
    *,
    summary_csv: Path,
    quarantine_csv: Path,
    quarantine_enable: bool,
) -> None:
    out_rows = sorted(rows, key=lambda item: str(item["series"]))
    _write_summary(out_rows, summary_csv)
    if quarantine_enable:
        _write_quarantine(out_rows, quarantine_csv)


def _worker(
    plan: Dict[str, object],
    *,
    out_dir: Path,
    qc_root: Path,
    qc_support_root: Path,
    failure_debug_root: Path,
    log_dir: Path,
    stage_root: Path,
    pe_dir: str,
    eddy_options: str,
    threads_per_job: int,
    force: bool,
    run_eddy_qc: bool,
    debug: bool,
    small_mask_retry: bool,
    b0_threshold: float,
    retry_mask_dilate_npass: int,
    retry_nvoxhp_min: int,
    retry_nvoxhp_cap: int,
    hyperparam_retry: bool,
    hyperparam_options: str,
    final_rescue_retry: bool,
    final_rescue_options: str,
    final_rescue_threads: int,
    strict_hq_rescue_retry: bool,
    strict_hq_rescue_options: str,
    strict_hq_rescue_threads: int,
    overwrite_logs: bool,
    persist_eddy_qc_support: bool,
    allow_cpu_fallback: bool,
    progress_queue: Optional[Queue] = None,
) -> Dict[str, object]:
    input_path = Path(plan["input_path"])
    series = str(plan["series"])
    replacement_b0 = plan.get("replacement_b0")
    b0_reference_index = plan.get("b0_reference_index")
    sanitize_b0_indices = [int(idx) for idx in plan.get("sanitize_b0_indices", [])]
    out_path = out_dir / f"{series}_preproc.mif"
    qc_dir = qc_root / series
    log_path = log_dir / f"{series}.log"
    row: Dict[str, object] = dict(plan["row"])
    row["pe_dir_requested"] = pe_dir
    row["eddy_options"] = eddy_options
    row["threads_per_job"] = threads_per_job
    started_at = time.time()

    if out_path.exists() and not force:
        return _stamp_row_duration(_existing_row(input_path, out_path, log_path), started_at)

    if force:
        if out_path.exists():
            out_path.unlink()
        if qc_dir.exists():
            shutil.rmtree(qc_dir, ignore_errors=True)

    if overwrite_logs and log_path.exists():
        log_path.unlink()
    if qc_dir.exists():
        shutil.rmtree(qc_dir, ignore_errors=True)

    with tempfile.TemporaryDirectory(dir=stage_root) as td_str:
        td = Path(td_str)
        env = _job_env(td, threads_per_job, allow_cpu_fallback=allow_cpu_fallback)
        stage_in = td / "in.mif"
        stage_dwi = stage_in
        stage_out = td / "out.mif"
        stage_qc = td / "qc"
        scratch = td / "scratch"
        scratch.mkdir(exist_ok=True)

        def _finalize_failure(failed_row: Dict[str, object]) -> Dict[str, object]:
            try:
                preserved = _preserve_failure_stage(
                    td,
                    debug_root=failure_debug_root,
                    series=series,
                    log_path=log_path,
                )
            except Exception as exc:
                _append_log_note(log_path, f"Failed to preserve failure stage directory: {exc}")
            else:
                if preserved is not None:
                    _append_note(failed_row, f"debug_stage={preserved}")
            return _stamp_row_duration(failed_row, started_at)

        try:
            _emit_progress(progress_queue, series, 2, "staging input")
            _run(
                ["mrconvert", str(input_path), str(stage_in), "-quiet", "-force"],
                env=env,
                cwd=td,
                log_file=log_path,
                debug=debug,
            )
            if int(row.get("vols_in") or 0) <= 2:
                _append_log_note(
                    log_path,
                    "Input has <=2 volumes; bypassing dwifslpreproc and persisting staged input unchanged",
                )
                _emit_progress(progress_queue, series, 97, "persisting passthrough")
                return _stamp_row_duration(
                    _persist_passthrough_output(
                        stage_in,
                        out_path,
                        row,
                        lane="lane_f_passthrough_low_volume",
                        note="passthrough_low_volume<=2",
                        env=env,
                    ),
                    started_at,
                )
            if sanitize_b0_indices and b0_reference_index is not None:
                _emit_progress(progress_queue, series, 5, "sanitizing b0s")
                _append_log_note(
                    log_path,
                    "Replacing blank / degraded b0 volumes "
                    f"{sanitize_b0_indices} using reference b0 volume {b0_reference_index}",
                )
                stage_dwi = _replace_volumes_with_reference(
                    stage_in,
                    reference_index=int(b0_reference_index),
                    replace_indices=sanitize_b0_indices,
                    workdir=td,
                    env=env,
                )
                _append_note(row, f"b0_sanitized={len(sanitize_b0_indices)}")
            elif replacement_b0 is not None:
                _emit_progress(progress_queue, series, 5, f"b0 swap vol {replacement_b0}")
                _append_log_note(
                    log_path,
                    f"Swapping valid b0 volume {replacement_b0} to the front before dwifslpreproc",
                )
                stage_dwi = _swap_volume_to_front(stage_in, replacement_b0, td, env)

            effective_pe_dir, pe_dir_source, header_pe_dir = _resolve_pe_dir(stage_dwi, pe_dir, env=env)
            row["pe_dir_effective"] = effective_pe_dir
            row["pe_dir_source"] = pe_dir_source
            row["header_pe_dir"] = header_pe_dir or ""
            if pe_dir_source == "header":
                _append_note(row, f"pe_dir={effective_pe_dir} (header)")
            elif pe_dir_source == "default":
                _append_note(row, f"pe_dir={effective_pe_dir} (default)")
                _append_log_note(log_path, f"PhaseEncodingDirection missing in header; falling back to {effective_pe_dir}")
            elif header_pe_dir is not None and header_pe_dir != effective_pe_dir:
                _append_note(row, f"pe_dir_header={header_pe_dir}")
                _append_log_note(
                    log_path,
                    f"Configured pe_dir {effective_pe_dir} differs from image header {header_pe_dir}",
                )

            _emit_progress(progress_queue, series, 6, "estimating baseline tSNR")
            tsnr_in = _median_tsnr(stage_dwi, td, env)
            if tsnr_in is not None:
                row["tsnr_in"] = f"{tsnr_in:.4f}"

            active_options = eddy_options
            active_mask: Optional[Path] = None
            succeeded_lane = "lane_a_standard"
            row["lane_reached"] = "lane_a_standard"

            def _dwifsl_line_callback(line: str) -> None:
                parsed = _parse_dwifslpreproc_progress(line)
                if parsed is not None:
                    percent, status = parsed
                    _emit_progress(progress_queue, series, percent, status)

            try:
                _emit_progress(progress_queue, series, 10, "dwifslpreproc setup")
                _append_log_note(log_path, f"Resolved eddy_cuda: {_resolved_executable('eddy_cuda', env)}")
                _append_log_note(log_path, f"Resolved eddy_cpu: {_resolved_executable('eddy_cpu', env)}")
                _append_log_note(log_path, f"Resolved dwifslpreproc: {_resolved_executable('dwifslpreproc', env)}")
                dwifsl_cmd = _build_dwifslpreproc_cmd(
                    stage_dwi=stage_dwi,
                    stage_out=stage_out,
                    pe_dir=effective_pe_dir,
                    eddy_options=active_options,
                    threads_per_job=threads_per_job,
                    scratch=scratch,
                    stage_qc=stage_qc,
                    run_eddy_qc=run_eddy_qc,
                )
                _append_log_note(log_path, f"dwifslpreproc command: {shlex.join(map(str, dwifsl_cmd))}")
                _run_live(
                    dwifsl_cmd,
                    env=env,
                    cwd=td,
                    log_file=log_path,
                    debug=debug,
                    line_callback=_dwifsl_line_callback,
                )
            except subprocess.CalledProcessError as exc:
                last_exc = exc
                if small_mask_retry and _should_retry_small_mask(exc, log_path):
                    _emit_progress(progress_queue, series, 10, "retrying with mask")
                    retry_mask, mask_voxels, mask_source = _build_eddy_mask(
                        stage_dwi,
                        workdir=td,
                        env=env,
                        log_file=log_path,
                        debug=debug,
                        b0_threshold=b0_threshold,
                        dilate_npass=retry_mask_dilate_npass,
                        min_mask_voxels=retry_nvoxhp_min,
                    )
                    if mask_voxels <= 1:
                        return _finalize_failure(
                            _quarantine_row(
                                row,
                                "fail_small_mask_empty",
                                note=f"mask salvage produced an empty mask ({mask_source})",
                            )
                        )

                    retry_nvoxhp = _estimate_retry_nvoxhp(mask_voxels, retry_nvoxhp_min, retry_nvoxhp_cap)
                    if retry_nvoxhp is None:
                        return _finalize_failure(
                            _quarantine_row(
                                row,
                                "fail_small_mask_too_small",
                                note=f"mask_voxels={mask_voxels}; mask_source={mask_source}",
                            )
                        )

                    active_mask = retry_mask
                    _append_note(row, f"mask_retry={mask_source}:{mask_voxels}vox")
                    row["lane_reached"] = "lane_b_mask_b0_salvage"
                    nvoxhp_candidates = _nvoxhp_retry_candidates(
                        mask_voxels,
                        primary=retry_nvoxhp,
                    )
                    for attempt_index, attempt_nvoxhp in enumerate(nvoxhp_candidates, start=1):
                        active_options = _merge_eddy_options(
                            eddy_options,
                            [f"--nvoxhp={attempt_nvoxhp}"],
                        )
                        _append_log_note(
                            log_path,
                            "Retrying dwifslpreproc with "
                            f"-eddy_mask {retry_mask.name} ({mask_source}, voxels={mask_voxels}) | "
                            f"nvoxhp attempt {attempt_index}/{len(nvoxhp_candidates)} | "
                            f"{active_options}",
                        )
                        try:
                            _run_live(
                                _build_dwifslpreproc_cmd(
                                    stage_dwi=stage_dwi,
                                    stage_out=stage_out,
                                    pe_dir=effective_pe_dir,
                                    eddy_options=active_options,
                                    threads_per_job=threads_per_job,
                                    scratch=scratch,
                                    stage_qc=stage_qc,
                                    run_eddy_qc=run_eddy_qc,
                                    eddy_mask=active_mask,
                                ),
                                env=env,
                                cwd=td,
                                log_file=log_path,
                                debug=debug,
                                line_callback=_dwifsl_line_callback,
                            )
                            succeeded_lane = "lane_b_mask_b0_salvage"
                            last_exc = None
                            break
                        except subprocess.CalledProcessError as exc_mask:
                            last_exc = exc_mask
                            if _should_retry_small_mask(exc_mask, log_path) or _should_retry_hyperparams(exc_mask, log_path):
                                continue
                            break

                if last_exc is not None and hyperparam_retry and _should_retry_hyperparams(last_exc, log_path):
                    _emit_progress(progress_queue, series, 10, "retrying hyperparams")
                    active_options = _merge_eddy_options(
                        active_options,
                        hyperparam_options.split(),
                        replace=True,
                    )
                    row["lane_reached"] = "lane_c_hyperparam_salvage"
                    _append_log_note(
                        log_path,
                        f"Retrying dwifslpreproc with hyperparameter salvage options: {active_options}",
                    )
                    try:
                        _run_live(
                            _build_dwifslpreproc_cmd(
                                stage_dwi=stage_dwi,
                                stage_out=stage_out,
                                pe_dir=effective_pe_dir,
                                eddy_options=active_options,
                                threads_per_job=threads_per_job,
                                scratch=scratch,
                                stage_qc=stage_qc,
                                run_eddy_qc=run_eddy_qc,
                                eddy_mask=active_mask,
                            ),
                            env=env,
                            cwd=td,
                            log_file=log_path,
                            debug=debug,
                            line_callback=_dwifsl_line_callback,
                        )
                        succeeded_lane = "lane_c_hyperparam_salvage"
                    except subprocess.CalledProcessError as exc_hyper:
                        last_exc = exc_hyper
                        if _should_retry_hyperparams(exc_hyper, log_path):
                            pass
                        else:
                            return _finalize_failure(
                                _quarantine_row(
                                    row,
                                    f"fail_dwifsl:{exc_hyper.returncode}",
                                    note="dwifslpreproc failed after hyperparameter salvage",
                                )
                            )
                    else:
                        last_exc = None

                rescue_class = (
                    _classify_final_rescue_failure(last_exc, log_path)
                    if last_exc is not None and final_rescue_retry
                    else None
                )
                direct_strict_rescue = rescue_class == "gpu_runtime"
                if last_exc is not None and rescue_class is not None and not direct_strict_rescue:
                    rescue_threads = max(1, int(final_rescue_threads))
                    rescue_env = _job_env(
                        td,
                        rescue_threads,
                        allow_cpu_fallback=allow_cpu_fallback,
                    )
                    _emit_progress(progress_queue, series, 10, "retrying conservative rescue")
                    active_options = _merge_eddy_options(
                        active_options,
                        final_rescue_options.split(),
                        replace=True,
                    )
                    row["lane_reached"] = "lane_d_conservative_rescue"
                    _append_log_note(
                        log_path,
                        "Retrying dwifslpreproc with conservative rescue options: "
                        f"{active_options} | threads={rescue_threads}",
                    )
                    try:
                        _run_live(
                            _build_dwifslpreproc_cmd(
                                stage_dwi=stage_dwi,
                                stage_out=stage_out,
                                pe_dir=effective_pe_dir,
                                eddy_options=active_options,
                                threads_per_job=rescue_threads,
                                scratch=scratch,
                                stage_qc=stage_qc,
                                run_eddy_qc=run_eddy_qc,
                                eddy_mask=active_mask,
                            ),
                            env=rescue_env,
                            cwd=td,
                            log_file=log_path,
                            debug=debug,
                            line_callback=_dwifsl_line_callback,
                        )
                        succeeded_lane = "lane_d_conservative_rescue"
                        last_exc = None
                    except subprocess.CalledProcessError as exc_final:
                        last_exc = exc_final

                    strict_rescue_class = (
                        _classify_final_rescue_failure(last_exc, log_path)
                        if last_exc is not None and strict_hq_rescue_retry
                        else None
                    )
                    if last_exc is not None and strict_rescue_class is not None:
                        strict_threads = max(1, int(strict_hq_rescue_threads))
                        strict_env = _job_env(
                            td,
                            strict_threads,
                            allow_cpu_fallback=allow_cpu_fallback,
                        )
                        _emit_progress(progress_queue, series, 10, "retrying strict hq rescue")
                        strict_mask = active_mask
                        strict_mask_voxels = 0
                        strict_mask_source = "existing"
                        try:
                            bet_mask, bet_voxels = _build_bet_mask_from_b0(
                                stage_dwi,
                                workdir=td,
                                env=strict_env,
                                log_file=log_path,
                                debug=debug,
                                b0_threshold=b0_threshold,
                            )
                        except subprocess.CalledProcessError as exc_mask:
                            last_exc = exc_mask
                            return _finalize_failure(
                                _quarantine_row(
                                    row,
                                    "fail_strict_hq_mask_build",
                                    note="failed to build BET mask for strict high-quality rescue",
                                    lane="lane_e_strict_hq_rescue",
                                )
                            )

                        if bet_voxels > 1:
                            strict_mask = bet_mask
                            strict_mask_voxels = bet_voxels
                            strict_mask_source = "bet_b0"
                        elif strict_mask is not None:
                            strict_mask_voxels = _count_nonzero(strict_mask, env=strict_env)
                            strict_mask_source = "existing_mask"

                        if strict_mask is None or strict_mask_voxels <= 1:
                            return _finalize_failure(
                                _quarantine_row(
                                    row,
                                    "fail_strict_hq_mask_empty",
                                    note="strict high-quality rescue has no usable mask",
                                    lane="lane_e_strict_hq_rescue",
                                )
                            )

                        strict_options = _merge_eddy_options(
                            eddy_options,
                            strict_hq_rescue_options.split(),
                            replace=True,
                        )
                        if strict_rescue_class == "singular":
                            strict_options = _merge_eddy_options(
                                strict_options,
                                ["--dont_sep_offs_move"],
                                replace=False,
                            )
                        row["lane_reached"] = "lane_e_strict_hq_rescue"
                        _append_note(
                            row,
                            f"strict_hq_mask={strict_mask_source}:{strict_mask_voxels}vox",
                        )
                        _append_log_note(
                            log_path,
                            "Retrying dwifslpreproc with strict high-quality rescue options: "
                            f"{strict_options} | threads={strict_threads} | "
                            f"mask={strict_mask.name} ({strict_mask_source}, voxels={strict_mask_voxels})",
                        )
                        try:
                            _run_live(
                                _build_dwifslpreproc_cmd(
                                    stage_dwi=stage_dwi,
                                    stage_out=stage_out,
                                    pe_dir=effective_pe_dir,
                                    eddy_options=strict_options,
                                    threads_per_job=strict_threads,
                                    scratch=scratch,
                                    stage_qc=stage_qc,
                                    run_eddy_qc=run_eddy_qc,
                                    eddy_mask=strict_mask,
                                ),
                                env=strict_env,
                                cwd=td,
                                log_file=log_path,
                                debug=debug,
                                line_callback=_dwifsl_line_callback,
                            )
                            active_mask = strict_mask
                            succeeded_lane = "lane_e_strict_hq_rescue"
                            last_exc = None
                        except subprocess.CalledProcessError as exc_strict:
                            last_exc = exc_strict
                            adaptive_options = _merge_eddy_options(
                                final_rescue_options,
                                ["--dont_peas"],
                                replace=True,
                            )
                            if strict_rescue_class == "singular":
                                adaptive_options = _merge_eddy_options(
                                    adaptive_options,
                                    ["--dont_sep_offs_move"],
                                    replace=False,
                                )
                            adaptive_candidates = _adaptive_gpu_nvoxhp_candidates(
                                strict_mask_voxels,
                            )
                            if strict_rescue_class in {"hyperparams", "singular"} and adaptive_candidates:
                                row["lane_reached"] = "lane_f_adaptive_gpu_rescue"
                                _append_note(
                                    row,
                                    f"adaptive_gpu_mask={strict_mask_source}:{strict_mask_voxels}vox",
                                )
                                for attempt_index, attempt_nvoxhp in enumerate(adaptive_candidates, start=1):
                                    adaptive_trial_options = _merge_eddy_options(
                                        adaptive_options,
                                        [f"--nvoxhp={attempt_nvoxhp}"],
                                        replace=True,
                                    )
                                    _append_log_note(
                                        log_path,
                                        "Retrying dwifslpreproc with adaptive GPU rescue: "
                                        f"attempt {attempt_index}/{len(adaptive_candidates)} | "
                                        f"{adaptive_trial_options} | "
                                        f"mask={strict_mask.name} ({strict_mask_source}, voxels={strict_mask_voxels})",
                                    )
                                    try:
                                        _run_live(
                                            _build_dwifslpreproc_cmd(
                                                stage_dwi=stage_dwi,
                                                stage_out=stage_out,
                                                pe_dir=effective_pe_dir,
                                                eddy_options=adaptive_trial_options,
                                                threads_per_job=strict_threads,
                                                scratch=scratch,
                                                stage_qc=stage_qc,
                                                run_eddy_qc=run_eddy_qc,
                                                eddy_mask=strict_mask,
                                            ),
                                            env=strict_env,
                                            cwd=td,
                                            log_file=log_path,
                                            debug=debug,
                                            line_callback=_dwifsl_line_callback,
                                        )
                                        active_mask = strict_mask
                                        active_options = adaptive_trial_options
                                        succeeded_lane = "lane_f_adaptive_gpu_rescue"
                                        last_exc = None
                                        break
                                    except subprocess.CalledProcessError as exc_adaptive:
                                        last_exc = exc_adaptive
                                        if _should_retry_hyperparams(exc_adaptive, log_path) or _should_retry_small_mask(exc_adaptive, log_path):
                                            continue
                                        break
                            if last_exc is not None:
                                final_status = {
                                    "hyperparams": "fail_hyperparams_after_strict_hq_rescue",
                                    "singular": "fail_singular_after_strict_hq_rescue",
                                    "gpu_runtime": "fail_gpu_runtime_after_strict_hq_rescue",
                                }.get(strict_rescue_class, "fail_dwifsl_after_strict_hq_rescue")
                                return _finalize_failure(
                                    _quarantine_row(
                                        row,
                                        final_status,
                                        note="strict high-quality rescue exhausted",
                                        lane=str(row.get("lane_reached", "")).strip() or "lane_e_strict_hq_rescue",
                                    )
                                )
                    elif last_exc is not None:
                        final_status = {
                            "hyperparams": "fail_hyperparams_after_final_rescue",
                            "singular": "fail_singular_after_final_rescue",
                            "gpu_runtime": "fail_gpu_runtime_after_final_rescue",
                        }.get(rescue_class, "fail_dwifsl_after_final_rescue")
                        return _finalize_failure(
                            _quarantine_row(
                                row,
                                final_status,
                                note="conservative rescue exhausted",
                                lane="lane_d_conservative_rescue",
                            )
                        )
                elif last_exc is not None and direct_strict_rescue and strict_hq_rescue_retry:
                    strict_rescue_class = rescue_class
                    strict_threads = max(1, int(strict_hq_rescue_threads))
                    strict_env = _job_env(
                        td,
                        strict_threads,
                        allow_cpu_fallback=allow_cpu_fallback,
                    )
                    _emit_progress(progress_queue, series, 10, "retrying strict hq rescue")
                    strict_mask = active_mask
                    strict_mask_voxels = 0
                    strict_mask_source = "existing"
                    try:
                        bet_mask, bet_voxels = _build_bet_mask_from_b0(
                            stage_dwi,
                            workdir=td,
                            env=strict_env,
                            log_file=log_path,
                            debug=debug,
                            b0_threshold=b0_threshold,
                        )
                    except subprocess.CalledProcessError as exc_mask:
                        last_exc = exc_mask
                        return _finalize_failure(
                            _quarantine_row(
                                row,
                                "fail_strict_hq_mask_build",
                                note="failed to build BET mask for strict high-quality rescue",
                                lane="lane_e_strict_hq_rescue",
                            )
                        )

                    if bet_voxels > 1:
                        strict_mask = bet_mask
                        strict_mask_voxels = bet_voxels
                        strict_mask_source = "bet_b0"
                    elif strict_mask is not None:
                        strict_mask_voxels = _count_nonzero(strict_mask, env=strict_env)
                        strict_mask_source = "existing_mask"

                    if strict_mask is None or strict_mask_voxels <= 1:
                        return _finalize_failure(
                            _quarantine_row(
                                row,
                                "fail_strict_hq_mask_empty",
                                note="strict high-quality rescue has no usable mask",
                                lane="lane_e_strict_hq_rescue",
                            )
                        )

                    strict_options = _merge_eddy_options(
                        eddy_options,
                        strict_hq_rescue_options.split(),
                        replace=True,
                    )
                    row["lane_reached"] = "lane_e_strict_hq_rescue"
                    _append_note(
                        row,
                        f"strict_hq_mask={strict_mask_source}:{strict_mask_voxels}vox",
                    )
                    _append_log_note(
                        log_path,
                        "Retrying dwifslpreproc with strict high-quality rescue options: "
                        f"{strict_options} | threads={strict_threads} | "
                        f"mask={strict_mask.name} ({strict_mask_source}, voxels={strict_mask_voxels})",
                    )
                    try:
                        _run_live(
                            _build_dwifslpreproc_cmd(
                                stage_dwi=stage_dwi,
                                stage_out=stage_out,
                                pe_dir=effective_pe_dir,
                                eddy_options=strict_options,
                                threads_per_job=strict_threads,
                                scratch=scratch,
                                stage_qc=stage_qc,
                                run_eddy_qc=run_eddy_qc,
                                eddy_mask=strict_mask,
                            ),
                            env=strict_env,
                            cwd=td,
                            log_file=log_path,
                            debug=debug,
                            line_callback=_dwifsl_line_callback,
                        )
                        active_mask = strict_mask
                        succeeded_lane = "lane_e_strict_hq_rescue"
                        last_exc = None
                    except subprocess.CalledProcessError as exc_strict:
                        last_exc = exc_strict
                        return _finalize_failure(
                            _quarantine_row(
                                row,
                                "fail_gpu_runtime_after_strict_hq_rescue",
                                note="strict high-quality rescue exhausted",
                                lane="lane_e_strict_hq_rescue",
                            )
                        )
                elif last_exc is not None:
                    return _finalize_failure(
                        _quarantine_row(
                            row,
                            f"fail_dwifsl:{last_exc.returncode}",
                            note="dwifslpreproc failed outside salvageable error classes",
                        )
                    )

            if persist_eddy_qc_support:
                _emit_progress(progress_queue, series, 91, "saving eddy support")
                _persist_eddy_qc_support_files(scratch, qc_support_root / series)

            _emit_progress(progress_queue, series, 92, "validating output")
            if not _is_valid_output(stage_out, int(row["vols_in"] or 0), env):
                return _finalize_failure(
                    _quarantine_row(row, "fail_validate", note="staged eddy output failed validation")
                )

            out_dir.mkdir(parents=True, exist_ok=True)
            _emit_progress(progress_queue, series, 97, "persisting output")
            shutil.move(str(stage_out), str(out_path))
            row["output"] = out_path.name
            row["vols_out"] = _nvol(out_path, env=env)

            _emit_progress(progress_queue, series, 98, "estimating output tSNR")
            tsnr_out = _median_tsnr(out_path, td, env)
            if tsnr_out is not None:
                row["tsnr_out"] = f"{tsnr_out:.4f}"
            if row["tsnr_in"] and row["tsnr_out"]:
                gain = float(row["tsnr_out"]) / float(row["tsnr_in"]) if float(row["tsnr_in"]) else 0.0
                row["tsnr_gain"] = f"{gain:.4f}"

            if run_eddy_qc and stage_qc.exists():
                if qc_dir.exists():
                    shutil.rmtree(qc_dir, ignore_errors=True)
                qc_root.mkdir(parents=True, exist_ok=True)
                _emit_progress(progress_queue, series, 99, "persisting qc")
                shutil.copytree(stage_qc, qc_dir)

            row["status"] = "ok"
            row["lane_reached"] = succeeded_lane
            _emit_progress(progress_queue, series, 100, "done")
            return _stamp_row_duration(row, started_at)

        except subprocess.CalledProcessError as exc:
            _emit_progress(progress_queue, series, 100, "failed")
            return _finalize_failure(
                _quarantine_row(row, f"fail_dwifsl:{exc.returncode}", note="uncaught dwifslpreproc failure")
            )
        except Exception as exc:
            _emit_progress(progress_queue, series, 100, "failed")
            return _finalize_failure(
                _quarantine_row(row, f"fail_exception:{type(exc).__name__}", note=str(exc))
            )


def run_eddy_preproc(
    *,
    deriv_root: Path = DEFAULT_DERIV_ROOT,
    jobs: int = DEFAULT_JOBS,
    threads_per_job: int = DEFAULT_THREADS_PER_JOB,
    force: bool = DEFAULT_FORCE,
    stage_root: Path = DEFAULT_STAGE_ROOT,
    pe_dir: str = DEFAULT_PE_DIR,
    eddy_options: str = DEFAULT_EDDY_OPTIONS,
    select_series: Optional[List[str]] = None,
    select_group: str = "all",
    cohort_dti_csv: Optional[Path] = None,
    run_eddy_qc: bool = True,
    debug: bool = False,
    status_only: bool = False,
    return_rows: bool = False,
    small_mask_retry: bool = DEFAULT_SMALL_MASK_RETRY,
    b0_threshold: float = DEFAULT_B0_THRESHOLD,
    retry_mask_dilate_npass: int = DEFAULT_RETRY_MASK_DILATE_NPASS,
    retry_nvoxhp_min: int = DEFAULT_RETRY_NVOXHP_MIN,
    retry_nvoxhp_cap: int = DEFAULT_RETRY_NVOXHP_CAP,
    hyperparam_retry: bool = DEFAULT_HYPERPARAM_RETRY,
    hyperparam_options: str = DEFAULT_HYPERPARAM_OPTIONS,
    final_rescue_retry: bool = DEFAULT_FINAL_RESCUE_RETRY,
    final_rescue_options: str = DEFAULT_FINAL_RESCUE_OPTIONS,
    final_rescue_threads: int = DEFAULT_FINAL_RESCUE_THREADS,
    strict_hq_rescue_retry: bool = DEFAULT_STRICT_HQ_RESCUE_RETRY,
    strict_hq_rescue_options: str = DEFAULT_STRICT_HQ_RESCUE_OPTIONS,
    strict_hq_rescue_threads: int = DEFAULT_STRICT_HQ_RESCUE_THREADS,
    quarantine_enable: bool = DEFAULT_QUARANTINE_ENABLE,
    overwrite_logs: bool = DEFAULT_OVERWRITE_LOGS,
    preflight_only: bool = False,
    persist_eddy_qc_support: bool = DEFAULT_PERSIST_EDDY_QC_SUPPORT,
    allow_cpu_fallback: bool = DEFAULT_ALLOW_CPU_FALLBACK,
    allow_input_fallback: bool = False,
) -> Dict[str, object]:
    _ensure_tools()

    deriv_root = Path(deriv_root)
    in_dir = deriv_root / "mif_unringed"
    out_dir = deriv_root / "eddy"
    qc_dir = deriv_root / "qc" / "eddy_qc"
    qc_support_dir = deriv_root / "qc" / "eddy_quad_support"
    failure_debug_dir = deriv_root / "qc" / "eddy_failed_stage"
    log_dir = deriv_root / "qc" / "eddy_logs"
    runtime_log = deriv_root / "qc" / "eddy_runtime_notes.txt"
    summary_csv = deriv_root / "qc" / "eddy_summary.csv"
    b0_provenance_csv = deriv_root / "qc" / "eddy_b0_provenance.csv"
    quarantine_csv = deriv_root / "qc" / "eddy_quarantine.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_support_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)

    all_inputs = list(_visible_glob(in_dir, "*_den_unr.mif"))
    if select_series:
        wanted = set(select_series)
        all_inputs = [p for p in all_inputs if _series_from_unringed(p) in wanted]
    all_inputs, selected_group, resolved_cohort_dti_csv = _filter_series_paths_by_group(
        all_inputs,
        group=select_group,
        cohort_dti_csv=cohort_dti_csv,
        series_name_fn=_series_from_unringed,
    )

    existing_outputs = 0
    pending_inputs: List[Path] = []
    for path in all_inputs:
        series = _series_from_unringed(path)
        out_path = out_dir / f"{series}_preproc.mif"
        if out_path.exists() and not force:
            existing_outputs += 1
        else:
            pending_inputs.append(path)

    print(
        f"Eddy plan: inputs={len(all_inputs)} | completed={existing_outputs} | "
        f"pending={len(pending_inputs)} | jobs={jobs} | "
        f"threads/job={threads_per_job} | force={force}"
    )
    if selected_group is not None:
        print(
            f"group filter: {selected_group} | "
            f"cohort csv: {resolved_cohort_dti_csv}"
        )
    print(f"PE dir: {pe_dir} | eddy options: {eddy_options}")
    print(f"allow_cpu_fallback={allow_cpu_fallback}")
    print(f"allow_input_fallback={allow_input_fallback}")
    print(
        f"small-mask retry: {small_mask_retry} | "
        f"b0_thresh={b0_threshold:g} | "
        f"dilate={retry_mask_dilate_npass} | "
        f"nvoxhp_min={retry_nvoxhp_min} | "
        f"nvoxhp_cap={retry_nvoxhp_cap} | "
        f"hyperparam_retry={hyperparam_retry} | "
        f"final_rescue_retry={final_rescue_retry} | "
        f"strict_hq_rescue_retry={strict_hq_rescue_retry} | "
        f"quarantine_enable={quarantine_enable} | "
        f"overwrite_logs={overwrite_logs}"
    )
    if hyperparam_retry:
        print(f"hyperparam salvage options: {hyperparam_options}")
    if final_rescue_retry:
        print(
            "final rescue options: "
            f"{final_rescue_options} | final_rescue_threads={max(1, int(final_rescue_threads))}"
        )
    if strict_hq_rescue_retry:
        print(
            "strict HQ rescue options: "
            f"{strict_hq_rescue_options} | "
            f"strict_hq_rescue_threads={max(1, int(strict_hq_rescue_threads))}"
        )

    runtime_lines_this_run: List[str] = []

    rows: List[Dict[str, object]] = []
    preflight_counts: Counter[str] = Counter()

    if existing_outputs and not force and not status_only:
        for path in all_inputs:
            series = _series_from_unringed(path)
            out_path = out_dir / f"{series}_preproc.mif"
            if out_path.exists():
                rows.append(_existing_row(path, out_path, log_dir / f"{series}.log"))

    if status_only:
        for path in tqdm(all_inputs, total=len(all_inputs), desc="eddy-scan", unit="series"):
            series = _series_from_unringed(path)
            out_path = out_dir / f"{series}_preproc.mif"
            row = _base_row(series, path, log_dir / f"{series}.log")
            row["output"] = out_path.name if out_path.exists() else ""
            row["status"] = "skip_exists" if out_path.exists() else "missing_output"
            row["lane_reached"] = "existing" if out_path.exists() else "lane_d_quarantine"
            rows.append(row)
    else:
        runnable_plans: List[Dict[str, object]] = []
        for path in tqdm(pending_inputs, total=len(pending_inputs), desc="eddy-preflight", unit="series"):
            series = _series_from_unringed(path)
            plan = _preflight_input(
                path,
                log_path=log_dir / f"{series}.log",
                stage_root=stage_root,
                threads_per_job=threads_per_job,
                b0_threshold=b0_threshold,
                allow_input_fallback=allow_input_fallback,
                overwrite_logs=overwrite_logs,
                debug=debug,
            )
            plan_row = dict(plan["row"])
            plan_row["pe_dir_requested"] = pe_dir
            if str(pe_dir).strip().lower() != "auto":
                plan_row["pe_dir_effective"] = pe_dir
                plan_row["pe_dir_source"] = "requested_preflight"
            plan_row["eddy_options"] = eddy_options
            plan_row["threads_per_job"] = threads_per_job
            plan["row"] = plan_row
            preflight_status = str(plan["preflight_status"])
            preflight_counts[preflight_status] += 1
            if plan.get("replacement_b0") is not None:
                preflight_counts["b0_swap"] += 1
            if preflight_status == "ready":
                runnable_plans.append(plan)
            else:
                rows.append(dict(plan["row"]))

        preflight_counts["ready"] = len(runnable_plans)
        _flush_live_reports(
            rows,
            summary_csv=summary_csv,
            quarantine_csv=quarantine_csv,
            quarantine_enable=quarantine_enable,
        )

        print("\nEddy preflight")
        print(
            f"pending={len(pending_inputs)} | ready={preflight_counts.get('ready', 0)} | "
            f"bad_shape={preflight_counts.get('bad_shape', 0)} | "
            f"bad_nvol={preflight_counts.get('bad_nvol', 0)} | "
            f"dwi_signal_missing={preflight_counts.get('dwi_signal_missing', 0)} | "
            f"b0_swap={preflight_counts.get('b0_swap', 0)} | "
            f"b0_unrecoverable={preflight_counts.get('b0_unrecoverable', 0)} | "
            f"bad_preflight={preflight_counts.get('bad_preflight', 0)}"
        )

        if preflight_only:
            for plan in runnable_plans:
                row = dict(plan["row"])
                row["status"] = "ready_for_eddy"
                row["lane_reached"] = "lane_a_standard"
                rows.append(row)
        else:
            if not pending_inputs:
                print("All selected eddy outputs already exist; nothing to run.")
            elif not runnable_plans:
                print("No runnable pending eddy inputs after preflight.")
            submitted = len(runnable_plans)
            finished = 0
            progress = {
                "queued": max(submitted - min(jobs, submitted), 0),
                "running": min(jobs, submitted),
                "succeeded_this_run": 0,
                "failed_this_run": 0,
                "quarantined": len([r for r in rows if str(r.get('status', '')).startswith('fail_')]),
            }
            print(f"Persisted outputs before run: {existing_outputs}/{len(all_inputs)}")
            _append_runtime_note(
                runtime_log,
                (
                    f"\n=== Eddy run {time.strftime('%Y-%m-%dT%H:%M:%S')} | "
                    f"inputs={len(all_inputs)} | completed={existing_outputs} | "
                    f"pending={len(pending_inputs)} | jobs={jobs} | threads/job={threads_per_job} ==="
                ),
            )
            progress_queue: Queue = Queue()
            live_panel = _LiveSeriesPanel(max_slots=max(1, min(jobs, submitted)), start_pos=1)
            completed_series: set[str] = set()
            with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
                futures = {
                    ex.submit(
                        _worker,
                        plan,
                        out_dir=out_dir,
                        qc_root=qc_dir,
                        qc_support_root=qc_support_dir,
                        failure_debug_root=failure_debug_dir,
                        log_dir=log_dir,
                        stage_root=stage_root,
                        pe_dir=pe_dir,
                        eddy_options=eddy_options,
                        threads_per_job=threads_per_job,
                        force=force,
                        run_eddy_qc=run_eddy_qc,
                        debug=debug,
                        small_mask_retry=small_mask_retry,
                        b0_threshold=b0_threshold,
                        retry_mask_dilate_npass=retry_mask_dilate_npass,
                        retry_nvoxhp_min=retry_nvoxhp_min,
                        retry_nvoxhp_cap=retry_nvoxhp_cap,
                        hyperparam_retry=hyperparam_retry,
                        hyperparam_options=hyperparam_options,
                        final_rescue_retry=final_rescue_retry,
                        final_rescue_options=final_rescue_options,
                        final_rescue_threads=max(1, int(final_rescue_threads)),
                        strict_hq_rescue_retry=strict_hq_rescue_retry,
                        strict_hq_rescue_options=strict_hq_rescue_options,
                        strict_hq_rescue_threads=max(1, int(strict_hq_rescue_threads)),
                        overwrite_logs=overwrite_logs,
                        persist_eddy_qc_support=persist_eddy_qc_support,
                        allow_cpu_fallback=allow_cpu_fallback,
                        progress_queue=progress_queue,
                    ): plan
                    for plan in runnable_plans
                }
                with tqdm(
                    total=submitted,
                    initial=0,
                    desc="eddy-series",
                    unit="series",
                    position=0,
                    dynamic_ncols=True,
                ) as pbar:
                    pending = dict(futures)
                    while pending:
                        drained_updates = False
                        while True:
                            try:
                                series_name, percent, status = progress_queue.get_nowait()
                            except Empty:
                                break
                            if series_name in completed_series:
                                continue
                            live_panel.update(series_name, percent, status)
                            drained_updates = True

                        progress["running"] = max(0, min(jobs, len(pending)))
                        pbar.set_postfix_str(
                            _build_run_postfix(
                                progress=progress,
                                live_panel=live_panel,
                                total_inputs=len(all_inputs),
                                existing_outputs=existing_outputs,
                                jobs=jobs,
                            ),
                            refresh=False,
                        )
                        if drained_updates:
                            pbar.refresh()

                        done, _ = wait(list(pending.keys()), timeout=0.2, return_when=FIRST_COMPLETED)
                        for fut in done:
                            plan = pending.pop(fut)
                            series_name = str(plan["series"])
                            row = fut.result()
                            rows.append(row)
                            finished += 1
                            remaining = submitted - finished
                            progress["running"] = min(jobs, remaining)
                            progress["queued"] = max(remaining - progress["running"], 0)
                            if str(row["status"]) == "ok":
                                progress["succeeded_this_run"] += 1
                            else:
                                progress["failed_this_run"] += 1
                                progress["quarantined"] += 1
                            pbar.update(1)
                            completed_series.add(series_name)
                            live_panel.close(series_name)
                            runtime_line = _series_runtime_line(row)
                            runtime_lines_this_run.append(runtime_line)
                            _append_runtime_note(runtime_log, runtime_line)
                            _flush_live_reports(
                                rows,
                                summary_csv=summary_csv,
                                quarantine_csv=quarantine_csv,
                                quarantine_enable=quarantine_enable,
                            )
                            tqdm.write(runtime_line)
                            pbar.set_postfix_str(
                                _build_run_postfix(
                                    progress=progress,
                                    live_panel=live_panel,
                                    total_inputs=len(all_inputs),
                                    existing_outputs=existing_outputs,
                                    jobs=jobs,
                                ),
                                refresh=False,
                            )

    out_rows = sorted(rows, key=lambda item: str(item["series"]))
    _write_summary(out_rows, summary_csv)
    _write_b0_provenance(out_rows, b0_provenance_csv)
    if quarantine_enable:
        _write_quarantine(out_rows, quarantine_csv)

    status_counts: Counter[str] = Counter(str(row["status"]) for row in out_rows)
    examples: DefaultDict[str, List[str]] = defaultdict(list)
    for row in out_rows:
        status = str(row["status"])
        if status not in {"ok", "skip_exists", "ready_for_eddy"} and len(examples[status]) < 10:
            examples[status].append(str(row["series"]))

    print("\nEddy summary")
    for key, value in sorted(status_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{key:>24}: {value}")
    if runtime_lines_this_run:
        print("\nEddy runtimes")
        for line in runtime_lines_this_run:
            print(line)

    return {
        "input_count": len(all_inputs),
        "completed_count": existing_outputs,
        "pending_count": len(pending_inputs),
        "preflight_counts": preflight_counts,
        "status_counts": status_counts,
        "examples": dict(examples),
        "summary_csv": str(summary_csv),
        "b0_provenance_csv": str(b0_provenance_csv),
        "quarantine_csv": str(quarantine_csv),
        "output_dir": str(out_dir),
        "qc_dir": str(qc_dir),
        "qc_support_dir": str(qc_support_dir),
        "log_dir": str(log_dir),
        "runtime_log": str(runtime_log),
        "runtime_lines": runtime_lines_this_run,
        "group_filter": selected_group or "all",
        "cohort_dti_csv": str(resolved_cohort_dti_csv) if resolved_cohort_dti_csv else None,
        **({"rows": out_rows} if return_rows else {}),
    }


def run_eddy_quad_qc(
    *,
    deriv_root: Path = DEFAULT_DERIV_ROOT,
    select_series: Optional[List[str]] = None,
    select_group: str = "all",
    cohort_dti_csv: Optional[Path] = None,
    stage_root: Path = DEFAULT_QC_ONLY_STAGE_ROOT,
    force: bool = False,
) -> Dict[str, object]:
    deriv_root = Path(deriv_root)
    eddy_dir = deriv_root / "eddy"
    qc_root = deriv_root / "qc" / "eddy_qc"
    support_root = deriv_root / "qc" / "eddy_quad_support"
    log_dir = deriv_root / "qc" / "eddy_qc_logs"

    qc_root.mkdir(parents=True, exist_ok=True)
    support_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)

    if shutil.which("eddy_quad") is None:
        raise RuntimeError("Required tool 'eddy_quad' not found in PATH")

    support_dirs = [p for p in sorted(support_root.iterdir()) if p.is_dir()]
    if select_series:
        wanted = set(select_series)
        support_dirs = [p for p in support_dirs if p.name in wanted]
    support_dirs, selected_group, resolved_cohort_dti_csv = _filter_series_paths_by_group(
        support_dirs,
        group=select_group,
        cohort_dti_csv=cohort_dti_csv,
        series_name_fn=lambda path: str(path.name),
    )

    print(f"Eddy QC plan: subjects={len(support_dirs)} | force={force}")
    if selected_group is not None:
        print(
            f"group filter: {selected_group} | "
            f"cohort csv: {resolved_cohort_dti_csv}"
        )
    status_counts: Counter[str] = Counter()
    examples: DefaultDict[str, List[str]] = defaultdict(list)

    with tqdm(total=len(support_dirs), desc="eddy_quad", unit="series", dynamic_ncols=True) as pbar:
        for support_dir in support_dirs:
            series = support_dir.name
            qc_dir = qc_root / series
            log_path = log_dir / f"{series}.log"
            preproc_mif = eddy_dir / f"{series}_preproc.mif"

            if qc_dir.exists() and not force:
                status = "skip_exists"
                status_counts[status] += 1
                pbar.update(1)
                continue
            if not preproc_mif.exists():
                status = "missing_preproc"
                status_counts[status] += 1
                if len(examples[status]) < 10:
                    examples[status].append(series)
                pbar.update(1)
                continue

            with tempfile.TemporaryDirectory(dir=stage_root) as td_str:
                td = Path(td_str)
                basename = td / "dwi_post_eddy"
                temp_qc_dir = td / "qc"
                _run(
                    ["mrconvert", str(preproc_mif), str(basename) + ".nii.gz", "-quiet", "-force"],
                    log_file=log_path,
                )
                for src in support_dir.iterdir():
                    if src.is_file():
                        shutil.copy2(src, td / src.name)

                cmd = [
                    "eddy_quad",
                    str(basename),
                    "-idx",
                    str(td / "eddy_indices.txt"),
                    "-par",
                    str(td / "eddy_config.txt"),
                    "-m",
                    str(td / "eddy_mask.nii"),
                    "-b",
                    str(td / "bvals"),
                    "-o",
                    str(temp_qc_dir),
                ]
                rotated_bvecs = td / "dwi_post_eddy.eddy_rotated_bvecs"
                residuals = td / "dwi_post_eddy.eddy_residuals"
                if rotated_bvecs.exists() and residuals.exists():
                    cmd.extend(["-g", str(rotated_bvecs)])

                try:
                    _run(cmd, log_file=log_path)
                    if qc_dir.exists():
                        shutil.rmtree(qc_dir, ignore_errors=True)
                    shutil.copytree(temp_qc_dir, qc_dir)
                    status = "ok"
                except subprocess.CalledProcessError as exc:
                    status = f"fail_qc:{exc.returncode}"
                    if len(examples[status]) < 10:
                        examples[status].append(series)

                status_counts[status] += 1
                pbar.update(1)

    print("\nEddy QC summary")
    for key, value in sorted(status_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{key:>24}: {value}")

    return {
        "input_count": len(support_dirs),
        "status_counts": status_counts,
        "examples": dict(examples),
        "qc_dir": str(qc_root),
        "log_dir": str(log_dir),
        "support_dir": str(support_root),
        "group_filter": selected_group or "all",
        "cohort_dti_csv": str(resolved_cohort_dti_csv) if resolved_cohort_dti_csv else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MRtrix dwifslpreproc on *_den_unr.mif inputs.")
    parser.add_argument("--deriv-root", type=Path, default=DEFAULT_DERIV_ROOT)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--threads-per-job", type=int, default=DEFAULT_THREADS_PER_JOB)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--pe-dir", default=DEFAULT_PE_DIR)
    parser.add_argument("--eddy-options", default=DEFAULT_EDDY_OPTIONS)
    parser.add_argument("--series", nargs="*", default=None)
    parser.add_argument(
        "--group",
        default="all",
        choices=sorted(VALID_GROUP_FILTERS),
        help="Restrict processing to a cohort group from cohort/dti.csv.",
    )
    parser.add_argument(
        "--cohort-dti-csv",
        type=Path,
        default=None,
        help="Optional override for cohort/dti.csv used by --group.",
    )
    parser.add_argument("--no-qc", action="store_true", help="Skip eddy_quad / qc directory generation.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--status-only", action="store_true", help="Only scan disk state and rebuild the summary CSV.")
    parser.add_argument("--no-small-mask-retry", action="store_true")
    parser.add_argument("--b0-threshold", type=float, default=DEFAULT_B0_THRESHOLD)
    parser.add_argument("--retry-mask-dilate-npass", type=int, default=DEFAULT_RETRY_MASK_DILATE_NPASS)
    parser.add_argument("--retry-nvoxhp-min", type=int, default=DEFAULT_RETRY_NVOXHP_MIN)
    parser.add_argument("--retry-nvoxhp-cap", type=int, default=DEFAULT_RETRY_NVOXHP_CAP)
    parser.add_argument("--no-hyperparam-retry", action="store_true")
    parser.add_argument("--hyperparam-options", default=DEFAULT_HYPERPARAM_OPTIONS)
    parser.add_argument("--no-final-rescue-retry", action="store_true")
    parser.add_argument("--final-rescue-options", default=DEFAULT_FINAL_RESCUE_OPTIONS)
    parser.add_argument("--final-rescue-threads", type=int, default=DEFAULT_FINAL_RESCUE_THREADS)
    parser.add_argument("--no-strict-hq-rescue-retry", action="store_true")
    parser.add_argument("--strict-hq-rescue-options", default=DEFAULT_STRICT_HQ_RESCUE_OPTIONS)
    parser.add_argument("--strict-hq-rescue-threads", type=int, default=DEFAULT_STRICT_HQ_RESCUE_THREADS)
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="If CUDA eddy fails, allow dwifslpreproc to retry with CPU eddy.",
    )
    parser.add_argument(
        "--allow-input-fallback",
        action="store_true",
        help="Allow Eddy preflight to fall back from unringed inputs to denoised/raw inputs. Disabled by default for supervisor-compatible runs.",
    )
    parser.add_argument("--no-quarantine-csv", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Run the preflight classification pass and stop before eddy.")
    parser.add_argument("--append-logs", action="store_true", help="Append to existing per-series logs instead of overwriting them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_eddy_preproc(
        deriv_root=args.deriv_root,
        jobs=max(1, args.jobs),
        threads_per_job=max(1, args.threads_per_job),
        force=bool(args.force),
        stage_root=args.stage_root,
        pe_dir=args.pe_dir,
        eddy_options=args.eddy_options,
        select_series=args.series,
        select_group=args.group,
        cohort_dti_csv=args.cohort_dti_csv,
        run_eddy_qc=not args.no_qc,
        debug=bool(args.debug),
        status_only=bool(args.status_only),
        small_mask_retry=not args.no_small_mask_retry,
        b0_threshold=float(args.b0_threshold),
        retry_mask_dilate_npass=max(1, args.retry_mask_dilate_npass),
        retry_nvoxhp_min=max(1, args.retry_nvoxhp_min),
        retry_nvoxhp_cap=max(1, args.retry_nvoxhp_cap),
        hyperparam_retry=not args.no_hyperparam_retry,
        hyperparam_options=args.hyperparam_options,
        final_rescue_retry=not args.no_final_rescue_retry,
        final_rescue_options=args.final_rescue_options,
        final_rescue_threads=max(1, args.final_rescue_threads),
        strict_hq_rescue_retry=not args.no_strict_hq_rescue_retry,
        strict_hq_rescue_options=args.strict_hq_rescue_options,
        strict_hq_rescue_threads=max(1, args.strict_hq_rescue_threads),
        quarantine_enable=not args.no_quarantine_csv,
        overwrite_logs=not args.append_logs,
        preflight_only=bool(args.preflight_only),
        allow_cpu_fallback=bool(args.allow_cpu_fallback),
        allow_input_fallback=bool(args.allow_input_fallback),
    )


if __name__ == "__main__":
    main()
