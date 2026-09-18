#!/usr/bin/env python3
"""
Helper-backed Step 7 structural connectome pipeline.

Default method:
  dhollander response -> ss3t_csd_beta1 -> mtnormalise -> ACT tractography
  -> SIFT2 -> AAL parcellation -> DTI metrics -> connectomes

Fallback:
  dhollander WM response -> single-tissue CSD -> ACT -> SIFT2 -> connectomes

This keeps the existing project layout stable while replacing the old
single-subject notebook scratch cells with a resumable batch runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import contextlib
import csv
import fcntl
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess as sp
import tempfile
import threading
import time
import traceback
from queue import Empty, Queue

from connectome_pipeline.pipeline_paths import resolve_pipeline_paths

try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM_AVAILABLE = True

    def tqdm(*args, **kwargs):
        if os.environ.get("STEP7_CLEAN_LOG", "").strip().lower() in {"1", "true", "yes"}:
            kwargs["disable"] = True
        return _tqdm(*args, **kwargs)

    tqdm.write = _tqdm.write
except Exception:  # pragma: no cover - terminal fallback when tqdm is unavailable
    _TQDM_AVAILABLE = False

    class _DummyBar:
        def __init__(self, total=0, initial=0, desc=None, unit=None, position=0, leave=True, **kwargs):
            self.total = max(0, int(total or 0))
            self.n = max(0, int(initial or 0))
            self.desc = desc or ""
            self.unit = unit or ""
            self.position = int(position or 0)
            self.leave = leave
            self.postfix = ""
            self._closed = False
            self._last_signature: tuple[str, int, int, str] | None = None
            self._render(force=True)

        def _render(self, force: bool = False):
            if self._closed:
                return
            percent = int(round((100.0 * self.n / self.total))) if self.total else 0
            signature = (self.desc, percent, self.total)
            if not force and signature == self._last_signature:
                return
            total_label = str(self.total) if self.total else "?"
            parts = [f"[progress:{self.position}]"]
            if self.desc:
                parts.append(self.desc)
            parts.append(f"{self.n}/{total_label} ({percent}%)")
            if self.postfix:
                parts.append(self.postfix)
            print(" | ".join(parts), flush=True)
            self._last_signature = signature

        def update(self, n=1):
            self.n += n
            self._render()

        def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
            if ordered_dict:
                parts = [f"{key}={value}" for key, value in ordered_dict.items()]
                self.postfix = ", ".join(parts)
            if refresh:
                self._render()

        def set_postfix_str(self, s="", refresh=True):
            self.postfix = s
            if refresh:
                self._render()

        def set_description_str(self, desc):
            self.desc = desc

        def refresh(self):
            self._render(force=True)

        def close(self):
            if self._closed:
                return None
            self._render(force=True)
            self._closed = True
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def tqdm(iterable=None, **kwargs):
        if iterable is None:
            return _DummyBar(
                total=kwargs.get("total", 0),
                initial=kwargs.get("initial", 0),
                desc=kwargs.get("desc"),
                unit=kwargs.get("unit"),
            )
        return iterable

    tqdm.write = print


_UNICODE_MINUS = re.compile(r"[\u2212\u2012\u2013\u2014\u2015]")


class FodInputQualityError(RuntimeError):
    """Raised when a DWI series is not suitable for FOD/CSD estimation."""


class Step7GeometryError(RuntimeError):
    """Raised when ACT geometry artifacts are present but unusable."""


_TCKGEN_LIMIT_LOCK = threading.Lock()
_TCKGEN_LIMIT_SEMAPHORE: threading.BoundedSemaphore | None = None
_TCKGEN_LIMIT_CAPACITY: int = 0


@dataclass
class Step7Config:
    root: Path = field(default_factory=lambda: DEFAULT_PATHS.data_root)
    deriv_root: Path = field(default_factory=lambda: DEFAULT_PATHS.deriv_root)
    bias_root: Path = field(default_factory=lambda: DEFAULT_PATHS.deriv_root / "biascorr_1")
    bbr_root: Path = field(default_factory=lambda: DEFAULT_PATHS.deriv_root / "dwi_t1_bbr")
    mrtrix_bin: Path = field(default_factory=lambda: Path.home() / "mrtrix3" / "bin")
    fsl_dir: Path = field(default_factory=lambda: Path(os.environ.get("FSLDIR", str(Path.home() / "fsl"))))
    aal_mni: Path = field(default_factory=lambda: DEFAULT_PATHS.aal_mni)
    deconv_mode: str = "auto"  # auto | ss3t | single_shell
    use_mtnormalise: bool = True
    batch_size: int = 2
    scan_workers: int = 0
    min_mem_available_gib: float = 4.0
    min_mem_available_no_swap_gib: float = 6.0
    memory_guard_poll_sec: float = 15.0
    tck_threads: int = 1
    omp_5tt_threads: int = 1
    prep_5tt_timeout_sec: int | None = 900
    prep_5tt_fallback: str = "fast"  # fast | none
    prep_5tt_mask_retry: bool = True
    fod_repair_gradients: bool = True
    fod_stage_ss3t_locally: bool = True
    fod_dhollander_erode: int = 3
    fod_dhollander_retry_erode: int | None = 0
    fod_ss3t_timeout_sec: int | None = 1800
    fod_response_fallback: str = "auto_single_shell"  # auto_single_shell | none
    fod_response_fallback_order: tuple[str, ...] = ("tournier", "fa")
    fod_min_dw_directions: int = 6
    fod_min_l0_abs_max: float = 1e-6
    fod_min_non_dc_abs_max: float = 1e-5
    fod_validation_timeout_sec: int | None = 120
    fod_bbr_mask_fallback: bool = True
    fod_mask_source: str = "auto"  # auto | bbr_full | bbr_5tt_intersection
    fod_bbr_mask_min_5tt_overlap_voxels: int = 1000
    fod_bbr_mask_min_5tt_overlap_frac: float = 0.01
    select_streamlines: int = 3_000_000
    chunk_streamlines: int = 200_000
    track_cutoffs: tuple[str, ...] = ("0.06", "0.05", "0.04", "0.03")
    track_seed_mode: str = "gmwmi"  # gmwmi | dynamic
    track_probe_streamlines: int = 2_000
    track_probe_max_seeds: int = 5_000_000
    track_probe_timeout_sec: int | None = 7_200
    track_chunk_timeout_sec: int | None = None
    track_chunk_jobs_per_subject: int = 1
    track_max_parallel_tckgen: int = 0
    track_resume_chunks: bool = True
    track_minlength: int = 10
    track_maxlength: int = 250
    force: bool = False
    sift_fd_scale_gm: bool | None = None
    assignment_radial_search: int | None = 4
    assignment_variant: str = "radial4"  # radial4 | radial8 | forward40 | forward80 | none | legacy
    t1_b0_route: str = "current_bbr"  # current_bbr | reference_flirt_dof6
    five_tt_route: str = "current_transform"  # current_transform | reference_b0_t1
    aal_transform_route: str = "current_direct"  # current_direct | supervisor_two_step
    backup_before_replace: bool = True
    validation_gate: bool = True
    analysis_gate_csv: Path | None = None
    write_assignments: bool = True
    write_invnodevol_count: bool = True
    summary_csv: Path | None = None
    qc_log_root: Path | None = None
    runtime_log: Path | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.deriv_root = Path(self.deriv_root)
        self.bias_root = Path(self.bias_root)
        self.bbr_root = Path(self.bbr_root)
        self.mrtrix_bin = Path(self.mrtrix_bin)
        self.fsl_dir = Path(self.fsl_dir)
        self.aal_mni = Path(self.aal_mni)
        if self.summary_csv is None:
            self.summary_csv = self.deriv_root / "qc" / "step7_summary.csv"
        else:
            self.summary_csv = Path(self.summary_csv)
        if self.qc_log_root is None:
            self.qc_log_root = self.deriv_root / "qc" / "step7_logs"
        else:
            self.qc_log_root = Path(self.qc_log_root)
        if self.runtime_log is None:
            self.runtime_log = self.deriv_root / "qc" / "step7_runtime_notes.txt"
        else:
            self.runtime_log = Path(self.runtime_log)
        if self.analysis_gate_csv is None:
            self.analysis_gate_csv = self.deriv_root / "qc" / "sc_matrix_qc" / "analysis_gate_after_sc_qc.csv"
        else:
            self.analysis_gate_csv = Path(self.analysis_gate_csv)


@dataclass
class FodModeBackfillReport:
    total_subjects: int = 0
    dry_run: bool = True
    allow_infer_without_summary: bool = False
    already_valid: list[str] = field(default_factory=list)
    eligible: list[str] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    missing_wmfod: list[str] = field(default_factory=list)
    empty_wmfod: list[str] = field(default_factory=list)
    missing_mode: list[str] = field(default_factory=list)
    missing_summary_evidence: list[str] = field(default_factory=list)
    invalid_existing_marker: list[str] = field(default_factory=list)
    mode_by_sid: dict[str, str] = field(default_factory=dict)
    source_by_sid: dict[str, str] = field(default_factory=dict)


def _format_duration(seconds: float | None) -> str:
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


def _resolve_scan_workers(cfg: Step7Config, total_subjects: int) -> int:
    if total_subjects <= 1:
        return 1
    requested = int(getattr(cfg, "scan_workers", 0) or 0)
    if requested <= 0:
        requested = max(int(cfg.batch_size), min(32, os.cpu_count() or 1))
    return max(1, min(total_subjects, requested))


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _effective_stage_workers(cfg: Step7Config, stage: str, todo_count: int) -> int:
    requested = max(1, int(cfg.batch_size))
    if todo_count <= 0:
        return 1
    if stage != "tracks":
        return max(1, min(todo_count, requested))
    cpu_slots = max(1, (os.cpu_count() or 1) // max(1, int(cfg.tck_threads)))
    return max(1, min(todo_count, requested, cpu_slots))


def _max_parallel_tckgen(cfg: Step7Config) -> int:
    requested = int(getattr(cfg, "track_max_parallel_tckgen", 0) or 0)
    if requested > 0:
        return max(1, requested)
    return max(1, (os.cpu_count() or 1) // max(1, int(cfg.tck_threads)))


@contextlib.contextmanager
def _tckgen_slot(cfg: Step7Config):
    global _TCKGEN_LIMIT_CAPACITY, _TCKGEN_LIMIT_SEMAPHORE
    capacity = _max_parallel_tckgen(cfg)
    with _TCKGEN_LIMIT_LOCK:
        if _TCKGEN_LIMIT_SEMAPHORE is None or _TCKGEN_LIMIT_CAPACITY != capacity:
            _TCKGEN_LIMIT_SEMAPHORE = threading.BoundedSemaphore(capacity)
            _TCKGEN_LIMIT_CAPACITY = capacity
        semaphore = _TCKGEN_LIMIT_SEMAPHORE
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


def _append_runtime_note(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", errors="ignore") as handle:
        handle.write(f"{line}\n")


@contextlib.contextmanager
def _subject_stage_lock(cfg: Step7Config, sid: str, stage: str, progress_queue: Queue | None = None):
    lock_dir = cfg.deriv_root / "qc" / "step7_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{sid}.lock"
    with open(lock_path, "a+", encoding="ascii", errors="ignore") as handle:
        _emit_stage_progress(progress_queue, sid, 0, f"{stage}: waiting for subject lock")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        handle.truncate()
        handle.write(f"stage={stage}\npid={os.getpid()}\nlocked_at={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        handle.flush()
        try:
            yield
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


def _read_meminfo_kib() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="ascii", errors="ignore") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    try:
                        values[parts[0][:-1]] = int(parts[1])
                    except ValueError:
                        pass
    except OSError:
        pass
    return values


def _format_kib(kib: int) -> str:
    return f"{kib / 1024 / 1024:.1f}GiB"


def _memory_guard_threshold_kib(cfg: Step7Config, meminfo: dict[str, int]) -> int:
    if int(meminfo.get("SwapTotal", 0)) > 0:
        threshold_gib = float(getattr(cfg, "min_mem_available_gib", 0.0) or 0.0)
    else:
        threshold_gib = float(getattr(cfg, "min_mem_available_no_swap_gib", 0.0) or 0.0)
    return max(0, int(threshold_gib * 1024 * 1024))


def _wait_for_memory_headroom(
    cfg: Step7Config,
    *,
    stage: str,
    sid: str,
    progress_queue: Queue | None = None,
) -> None:
    meminfo = _read_meminfo_kib()
    threshold = _memory_guard_threshold_kib(cfg, meminfo)
    if threshold <= 0:
        return
    poll_sec = max(1.0, float(getattr(cfg, "memory_guard_poll_sec", 15.0) or 15.0))
    waiting = False
    while True:
        meminfo = _read_meminfo_kib()
        available = int(meminfo.get("MemAvailable", meminfo.get("MemFree", 0)))
        if available >= threshold:
            if waiting:
                _append_runtime_note(
                    cfg.runtime_log,
                    (
                        f"Step7 memory guard resume {time.strftime('%Y-%m-%dT%H:%M:%S')} | "
                        f"stage={stage} | sid={sid} | MemAvailable={_format_kib(available)} | "
                        f"threshold={_format_kib(threshold)}"
                    ),
                )
            return
        if not waiting:
            _emit_stage_progress(progress_queue, sid, 0, f"{stage}: waiting for memory")
            _append_runtime_note(
                cfg.runtime_log,
                (
                    f"Step7 memory guard wait {time.strftime('%Y-%m-%dT%H:%M:%S')} | "
                    f"stage={stage} | sid={sid} | MemAvailable={_format_kib(available)} | "
                    f"threshold={_format_kib(threshold)} | SwapTotal={_format_kib(int(meminfo.get('SwapTotal', 0)))}"
                ),
            )
            waiting = True
        time.sleep(poll_sec)


def _emit_stage_progress(progress_queue: Queue | None, sid: str, percent: int, status: str) -> None:
    if progress_queue is None:
        return
    progress_queue.put((sid, max(0, min(100, int(percent))), status))


def _stamp_stage_result(result: dict[str, str], started_at: float) -> dict[str, str]:
    elapsed_seconds = max(0.0, time.time() - started_at)
    result["duration_seconds"] = f"{elapsed_seconds:.3f}"
    result["duration_elapsed"] = _format_duration(elapsed_seconds)
    return result


def _stage_success_runtime_line(result: dict[str, str]) -> str:
    sid = result.get("sid", "unknown")
    elapsed = result.get("duration_elapsed", "?")
    stage = result.get("stage", "unknown")
    return f"Sample: {sid} | stage={stage} | time taken: {elapsed}"


def _parse_summary_note(note: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_part in (note or "").split(";"):
        part = raw_part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def _fod_outputs_present(paths: dict[str, Path]) -> bool:
    return paths["WMFOD_FINAL"].exists() and paths["DECONV_MODE_TXT"].exists()


def _infer_existing_fod_mode(paths: dict[str, Path]) -> str:
    if paths["DECONV_MODE_TXT"].exists():
        return paths["DECONV_MODE_TXT"].read_text(encoding="ascii", errors="ignore").strip()
    if paths["WM_TXT_CSD"].exists():
        return "single_shell"
    if paths["WM_TXT"].exists() and (paths["GM_TXT"].exists() or paths["CSF_TXT"].exists()):
        return "ss3t"
    return ""


def _looks_like_populated_mif(cfg: Step7Config, img: Path) -> bool:
    if not img.exists():
        return False
    try:
        if img.stat().st_size <= 4096:
            return False
    except OSError:
        return False
    size = _mrinfo_size(cfg, img)
    if len(size) < 3 or any(value <= 0 for value in size[:3]):
        return False
    if len(size) >= 4 and size[3] <= 0:
        return False
    return True


def _wmfod_sh_quality_details(
    cfg: Step7Config,
    img: Path,
    *,
    context: str = "",
) -> tuple[bool, str]:
    if not img.exists():
        return False, "missing_wmfod"
    size = _mrinfo_size(cfg, img)
    if len(size) < 4 or size[3] < 2:
        return False, f"not_4d_sh_fod: size={size}"

    mask = img.parent / "mask_fod.mif"
    base_cmd = [_tool(cfg, "mrstats"), str(img)]
    if mask.exists():
        base_cmd += ["-mask", str(mask)]
    base_cmd += ["-output", "min", "-output", "max"]

    env = _base_env(cfg)
    try:
        proc = sp.run(
            base_cmd,
            env=env,
            text=True,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            timeout=cfg.fod_validation_timeout_sec,
        )
    except sp.TimeoutExpired:
        return False, f"wmfod_stats_timeout: timeout_sec={cfg.fod_validation_timeout_sec}"
    if proc.returncode != 0 and mask.exists():
        fallback_cmd = [_tool(cfg, "mrstats"), str(img), "-output", "min", "-output", "max"]
        try:
            proc = sp.run(
                fallback_cmd,
                env=env,
                text=True,
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                timeout=cfg.fod_validation_timeout_sec,
            )
        except sp.TimeoutExpired:
            return False, f"wmfod_stats_timeout: timeout_sec={cfg.fod_validation_timeout_sec}; fallback=unmasked"
    if proc.returncode != 0:
        msg = (proc.stdout or "").strip().splitlines()[-1:] or ["mrstats failed"]
        return False, f"wmfod_stats_failed: {msg[0]}"

    rows: list[tuple[float, float]] = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if len(rows) < 2:
        return False, f"wmfod_stats_empty: rows={len(rows)}"

    l0_abs_max = max(abs(rows[0][0]), abs(rows[0][1]))
    non_dc_abs_max = 0.0
    non_dc_sig = 0
    for mn, mx in rows[1:]:
        coeff_abs = max(abs(mn), abs(mx))
        non_dc_abs_max = max(non_dc_abs_max, coeff_abs)
        if coeff_abs >= cfg.fod_min_non_dc_abs_max:
            non_dc_sig += 1

    detail = (
        f"l0_abs_max={l0_abs_max:.6g}; non_dc_abs_max={non_dc_abs_max:.6g}; "
        f"non_dc_coeffs_ge_{cfg.fod_min_non_dc_abs_max:g}={non_dc_sig}; "
        f"ncoeff={len(rows)}"
    )
    if l0_abs_max < cfg.fod_min_l0_abs_max:
        return False, f"invalid_wmfod_sh: low l0 amplitude; {detail}"
    if non_dc_abs_max < cfg.fod_min_non_dc_abs_max or non_dc_sig <= 0:
        return False, f"invalid_wmfod_sh: no meaningful non-DC SH coefficients; {detail}"
    return True, detail


def _wmfod_has_valid_content(
    cfg: Step7Config,
    img: Path,
    *,
    context: str = "",
    prefer_fast: bool = False,
) -> bool:
    if not _looks_like_populated_mif(cfg, img):
        return False
    ok, detail = _wmfod_sh_quality_details(cfg, img, context=context)
    if ok:
        return True
    if context:
        _append_runtime_note(
            cfg.runtime_log,
            (
                f"Step7 FOD SH validation failed {time.strftime('%Y-%m-%dT%H:%M:%S')} | "
                f"{context} | img={img} | {detail}"
            ),
        )
    if prefer_fast:
        return False
    count = _safe_mr_nonzero_count(cfg, img, context=context)
    if count is not None:
        return count > 0 and ok
    return False


def _fod_stage_ready(cfg: Step7Config, sid: str, *, quick: bool = False) -> bool:
    paths = _subject_paths(cfg, sid, create_dirs=False)
    if not _fod_outputs_present(paths):
        return False
    if quick:
        return True
    if not _wmfod_has_valid_content(cfg, paths["WMFOD_FINAL"], context=f"fod_ready sid={sid}"):
        return False
    try:
        _validate_fod_act_geometry(cfg, sid, paths, fod_mask_mode="status_check")
    except Step7GeometryError:
        return False
    return True


def _classify_legacy_fod_completion(
    cfg: Step7Config,
    sid: str,
    *,
    latest_fod_ok_row: dict[str, str] | None = None,
    allow_infer_without_summary: bool = False,
) -> tuple[str, str, str]:
    paths = _subject_paths(cfg, sid, create_dirs=False)
    row = latest_fod_ok_row or {}
    if _fod_outputs_present(paths):
        if _wmfod_has_valid_content(
            cfg,
            paths["WMFOD_FINAL"],
            context=f"legacy_fod sid={sid} outputs_present=1",
            prefer_fast=True,
        ):
            return "already_valid", "", ""
        return "invalid_existing_marker", "", ""
    if paths["DECONV_MODE_TXT"].exists():
        return "invalid_existing_marker", "", ""
    if not paths["WMFOD_FINAL"].exists():
        return "missing_wmfod", "", ""
    if not _wmfod_has_valid_content(
        cfg,
        paths["WMFOD_FINAL"],
        context=f"legacy_fod sid={sid} outputs_present=0",
        prefer_fast=True,
    ):
        return "empty_wmfod", "", ""
    mode = (row.get("mode") or "").strip() or _infer_existing_fod_mode(paths)
    if mode not in {"ss3t", "single_shell"}:
        return "missing_mode", "", ""
    if row:
        return "eligible", mode, "summary"
    if allow_infer_without_summary:
        return "eligible", mode, "inferred"
    return "missing_summary_evidence", "", ""


def _write_legacy_fod_completion_marker(
    cfg: Step7Config,
    sid: str,
    mode: str,
    source: str,
    *,
    write_stage: str = "fod_backfill",
    write_status: str = "ok",
) -> None:
    paths = _subject_paths(cfg, sid, create_dirs=False)
    paths["DECONV_MODE_TXT"].write_text(mode + "\n", encoding="ascii")
    _write_summary_row(
        cfg,
        sid,
        write_stage,
        write_status,
        note=f"restored deconv_mode from {source} legacy outputs",
        mode=mode,
    )
    _append_runtime_note(
        cfg.runtime_log,
        (
            f"Step7 FOD marker restore {time.strftime('%Y-%m-%dT%H:%M:%S')} | "
            f"sid={sid} | mode={mode} | source={source} | stage={write_stage}"
        ),
    )


def _latest_summary_rows(
    cfg: Step7Config,
    *,
    stage: str,
    wanted_sids: set[str],
    statuses: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    if not wanted_sids or not cfg.summary_csv.exists():
        return latest
    try:
        with open(cfg.summary_csv, "r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sid = (row.get("sid") or "").strip()
                if sid not in wanted_sids:
                    continue
                if (row.get("stage") or "").strip() != stage:
                    continue
                status = (row.get("status") or "").strip()
                if statuses is not None and status not in statuses:
                    continue
                latest[sid] = row
    except Exception:
        return {}
    return latest


def _empty_fod_success_counts() -> dict[str, int]:
    return {
        "ss3t": 0,
        "single_shell": 0,
        "wm_fallback": 0,
        "wm_fallback_tournier": 0,
        "wm_fallback_fa": 0,
        "bbr_mask": 0,
        "unknown": 0,
    }


def _accumulate_fod_success_counts(counts: dict[str, int], meta: dict[str, str]) -> None:
    mode = meta.get("mode", "").strip()
    if mode == "ss3t":
        counts["ss3t"] += 1
    elif mode == "single_shell":
        counts["single_shell"] += 1
    else:
        counts["unknown"] += 1
    response_fallback = meta.get("response_fallback", "").strip()
    if response_fallback:
        counts["wm_fallback"] += 1
        if response_fallback == "tournier":
            counts["wm_fallback_tournier"] += 1
        elif response_fallback == "fa":
            counts["wm_fallback_fa"] += 1
    if meta.get("fod_mask_mode", "").startswith("bbr_mask"):
        counts["bbr_mask"] += 1


def _fod_success_meta_from_result(result: dict[str, str]) -> dict[str, str]:
    return {
        "mode": str(result.get("mode", "")),
        "response_fallback": str(result.get("response_fallback", "")),
        "fod_mask_mode": str(result.get("fod_mask_mode", "")),
    }


def _collect_existing_fod_success_counts(cfg: Step7Config, sids: list[str]) -> dict[str, int]:
    counts = _empty_fod_success_counts()

    if not sids or not cfg.summary_csv.exists():
        for sid in sids:
            paths = _subject_paths(cfg, sid, create_dirs=False)
            mode = _infer_existing_fod_mode(paths)
            _accumulate_fod_success_counts(counts, {"mode": mode, "response_fallback": "", "fod_mask_mode": ""})
        return counts
    wanted = set(sids)
    latest: dict[str, dict[str, str]] = {}
    try:
        with open(cfg.summary_csv, "r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sid = (row.get("sid") or "").strip()
                if sid not in wanted:
                    continue
                if (row.get("stage") or "").strip() != "fod":
                    continue
                if (row.get("status") or "").strip() != "ok":
                    continue
                latest[sid] = row
    except Exception:
        return counts
    for sid in sids:
        row = latest.get(sid, {})
        note_fields = _parse_summary_note(row.get("note", ""))
        mode = (row.get("mode") or "").strip()
        if not mode:
            paths = _subject_paths(cfg, sid, create_dirs=False)
            mode = _infer_existing_fod_mode(paths)
        _accumulate_fod_success_counts(
            counts,
            {
                "mode": mode,
                "response_fallback": note_fields.get("response_fallback", ""),
                "fod_mask_mode": note_fields.get("fod_mask_mode", ""),
            },
        )
    return counts


def _format_fod_success_breakdown(counts: dict[str, int], *, compact: bool = False) -> str:
    if compact:
        return (
            f"modes=ss3t:{counts['ss3t']} "
            f"single:{counts['single_shell']} "
            f"wmfb:{counts['wm_fallback']} "
            f"bbr:{counts['bbr_mask']} "
            f"unk:{counts['unknown']}"
        )
    return (
        f"ss3t={counts['ss3t']} | "
        f"single_shell={counts['single_shell']} | "
        f"wm_fallback={counts['wm_fallback']} "
        f"(tournier={counts['wm_fallback_tournier']}, fa={counts['wm_fallback_fa']}) | "
        f"bbr_mask={counts['bbr_mask']} | "
        f"unknown={counts['unknown']}"
    )


def backfill_legacy_fod_completion_markers(
    cfg: Step7Config,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    *,
    dry_run: bool = True,
    allow_infer_without_summary: bool = False,
    show_progress: bool = True,
    jobs: int | None = None,
) -> FodModeBackfillReport:
    sids = _discover_subjects(cfg, include=include, exclude=exclude)
    report = FodModeBackfillReport(
        total_subjects=len(sids),
        dry_run=dry_run,
        allow_infer_without_summary=allow_infer_without_summary,
    )
    latest_fod_ok = _latest_summary_rows(
        cfg,
        stage="fod",
        wanted_sids=set(sids),
        statuses={"ok", "skip_exists"},
    )

    worker_count = max(1, min(len(sids) or 1, int(cfg.batch_size if jobs is None else jobs)))

    def _record_result(sid: str, category: str, mode: str = "", source: str = "") -> None:
        if category == "already_valid":
            report.already_valid.append(sid)
        elif category == "eligible":
            report.eligible.append(sid)
            report.mode_by_sid[sid] = mode
            report.source_by_sid[sid] = source
        elif category == "missing_wmfod":
            report.missing_wmfod.append(sid)
        elif category == "empty_wmfod":
            report.empty_wmfod.append(sid)
        elif category == "missing_mode":
            report.missing_mode.append(sid)
        elif category == "missing_summary_evidence":
            report.missing_summary_evidence.append(sid)
        elif category == "invalid_existing_marker":
            report.invalid_existing_marker.append(sid)
        else:
            raise ValueError(f"Unknown FOD backfill category: {category}")

    def _scan_sid(sid: str) -> tuple[str, str, str, str]:
        category, mode, source = _classify_legacy_fod_completion(
            cfg,
            sid,
            latest_fod_ok_row=latest_fod_ok.get(sid, {}),
            allow_infer_without_summary=allow_infer_without_summary,
        )
        return sid, category, mode, source

    bar = None
    if show_progress:
        bar = tqdm(total=len(sids), desc=f"fod-backfill[{worker_count}]", unit="sid", dynamic_ncols=True)
    try:
        if worker_count == 1:
            for sid in sids:
                scan_sid, category, mode, source = _scan_sid(sid)
                _record_result(scan_sid, category, mode, source)
                if bar is not None:
                    bar.update(1)
                    bar.set_postfix_str(
                        (
                            f"valid={len(report.already_valid)} | "
                            f"eligible={len(report.eligible)} | "
                            f"empty={len(report.empty_wmfod)} | "
                            f"missing_summary={len(report.missing_summary_evidence)}"
                        ),
                        refresh=False,
                    )
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                pending = {executor.submit(_scan_sid, sid): sid for sid in sids}
                while pending:
                    done_futures, _ = wait(list(pending.keys()), timeout=0.2, return_when=FIRST_COMPLETED)
                    if not done_futures:
                        if bar is not None:
                            queued = max(0, len(pending) - worker_count)
                            bar.set_postfix_str(
                                (
                                    f"valid={len(report.already_valid)} | "
                                    f"eligible={len(report.eligible)} | "
                                    f"empty={len(report.empty_wmfod)} | "
                                    f"running={min(worker_count, len(pending))} | "
                                    f"queued={queued}"
                                ),
                                refresh=False,
                            )
                        continue
                    for future in done_futures:
                        pending.pop(future)
                        scan_sid, category, mode, source = future.result()
                        _record_result(scan_sid, category, mode, source)
                        if bar is not None:
                            bar.update(1)
                            queued = max(0, len(pending) - worker_count)
                            bar.set_postfix_str(
                                (
                                    f"valid={len(report.already_valid)} | "
                                    f"eligible={len(report.eligible)} | "
                                    f"empty={len(report.empty_wmfod)} | "
                                    f"running={min(worker_count, len(pending))} | "
                                    f"queued={queued}"
                                ),
                                refresh=False,
                            )
    finally:
        if bar is not None:
            bar.close()

    for values in (
        report.already_valid,
        report.eligible,
        report.written,
        report.missing_wmfod,
        report.empty_wmfod,
        report.missing_mode,
        report.missing_summary_evidence,
        report.invalid_existing_marker,
    ):
        values.sort()

    if dry_run:
        return report

    source_counts = {"summary": 0, "inferred": 0}
    for sid in report.eligible:
        mode = report.mode_by_sid[sid]
        source = report.source_by_sid[sid]
        _write_legacy_fod_completion_marker(cfg, sid, mode, source)
        source_counts[source] = source_counts.get(source, 0) + 1
        report.written.append(sid)
    if report.written:
        _append_runtime_note(
            cfg.runtime_log,
            (
                f"Step7 FOD marker backfill {time.strftime('%Y-%m-%dT%H:%M:%S')} | "
                f"written={len(report.written)} | "
                f"summary={source_counts.get('summary', 0)} | "
                f"inferred={source_counts.get('inferred', 0)}"
            ),
        )
    return report


def print_fod_mode_backfill_report(
    report: FodModeBackfillReport,
    *,
    max_examples: int = 10,
) -> FodModeBackfillReport:
    from_summary = sum(1 for sid in report.eligible if report.source_by_sid.get(sid) == "summary")
    inferred = sum(1 for sid in report.eligible if report.source_by_sid.get(sid) == "inferred")
    print("FOD marker backfill")
    print("-------------------")
    print(f"total subjects checked      : {report.total_subjects}")
    print(f"already valid               : {len(report.already_valid)}")
    print(f"eligible to backfill        : {len(report.eligible)}")
    print(f"eligible via summary        : {from_summary}")
    print(f"eligible via inference      : {inferred}")
    print(f"missing wmfod_final.mif     : {len(report.missing_wmfod)}")
    print(f"empty wmfod_final.mif       : {len(report.empty_wmfod)}")
    print(f"missing mode evidence       : {len(report.missing_mode)}")
    print(f"missing summary evidence    : {len(report.missing_summary_evidence)}")
    print(f"invalid existing marker     : {len(report.invalid_existing_marker)}")
    if report.dry_run:
        print("writes performed            : dry-run only")
    else:
        print(f"writes performed            : {len(report.written)}")

    def _print_examples(label: str, values: list[str]) -> None:
        if not values:
            return
        print(f"{label}: " + ", ".join(values[:max_examples]))

    _print_examples("example eligible", report.eligible)
    _print_examples("example missing summary", report.missing_summary_evidence)
    _print_examples("example empty wmfod", report.empty_wmfod)
    _print_examples("example missing mode", report.missing_mode)
    _print_examples("example invalid marker", report.invalid_existing_marker)
    return report


def _estimate_remaining_seconds(
    *,
    active_states: list[dict[str, object]],
    queued: int,
    jobs: int,
    avg_total_seconds: float | None,
) -> float | None:
    if not active_states and queued <= 0:
        return 0.0
    slot_remaining: list[float] = []
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


class _LiveSidPanel:
    def __init__(self, max_slots: int, start_pos: int = 1):
        self.max_slots = max_slots
        self.start_pos = start_pos
        self.free_positions = list(range(start_pos, start_pos + max_slots))
        self.rows: dict[str, tuple[object, int]] = {}
        self.states: dict[str, dict[str, object]] = {}
        self.completed_durations: list[float] = []

    def _ensure_row(self, sid: str):
        if sid in self.rows:
            return self.rows[sid][0]
        pos = self.free_positions.pop(0) if self.free_positions else self.start_pos
        bar = tqdm(
            total=100,
            position=pos,
            leave=False,
            dynamic_ncols=True,
            bar_format="{desc:<58} {bar} {n:>3.0f}% | {postfix}",
        )
        self.rows[sid] = (bar, pos)
        return bar

    def _estimate_total_seconds(
        self,
        state: dict[str, object],
        now: float,
        *,
        fallback_total_seconds: float | None = None,
    ) -> float | None:
        started_at = float(state.get("started_at", now))
        elapsed = max(0.0, now - started_at)
        percent = max(0, min(100, int(state.get("percent", 0))))
        if percent >= 20:
            return elapsed / max(percent / 100.0, 0.2)
        return fallback_total_seconds

    def average_total_seconds(self, now: float | None = None) -> float | None:
        now = time.time() if now is None else now
        totals = [value for value in self.completed_durations if value > 0]
        for state in self.states.values():
            total = self._estimate_total_seconds(state, now)
            if total is not None and total > 0:
                totals.append(total)
        if not totals:
            return None
        return sum(totals) / len(totals)

    def snapshot(self, now: float | None = None) -> list[dict[str, object]]:
        now = time.time() if now is None else now
        avg_total_seconds = self.average_total_seconds(now)
        states: list[dict[str, object]] = []
        for sid, state in sorted(self.states.items()):
            elapsed = max(0.0, now - float(state.get("started_at", now)))
            total = self._estimate_total_seconds(state, now, fallback_total_seconds=avg_total_seconds)
            eta = None if total is None else max(0.0, total - elapsed)
            states.append(
                {
                    "sid": sid,
                    "status": str(state.get("status", "")),
                    "percent": int(state.get("percent", 0)),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                }
            )
        return states

    def update(self, sid: str, percent: int, status: str) -> None:
        now = time.time()
        bar = self._ensure_row(sid)
        target = max(0, min(100, int(percent)))
        state = self.states.get(sid, {"started_at": now})
        state["percent"] = target
        state["status"] = status
        state["last_update"] = now
        self.states[sid] = state
        elapsed = max(0.0, now - float(state["started_at"]))
        total = self._estimate_total_seconds(
            state,
            now,
            fallback_total_seconds=self.average_total_seconds(now),
        )
        eta = None if total is None else max(0.0, total - elapsed)
        delta = target - getattr(bar, "n", 0)
        if delta > 0:
            bar.update(delta)
        elif delta < 0:
            bar.n = target
        bar.set_description_str(f"{_short_label(sid, 20)} | {_short_label(status, 34)}")
        bar.set_postfix_str(
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
            refresh=False,
        )
        bar.refresh()

    def close(self, sid: str) -> None:
        now = time.time()
        rec = self.rows.pop(sid, None)
        state = self.states.pop(sid, None)
        if state is not None:
            duration = max(0.0, now - float(state.get("started_at", now)))
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


def _build_stage_postfix(
    *,
    stage: str,
    finished: int,
    ok: int,
    total: int,
    fail_count: int,
    queued: int,
    live_panel: _LiveSidPanel,
    jobs: int,
    repair_count: int = 0,
    success_fragment: str = "",
) -> str:
    active_states = live_panel.snapshot()
    eta_seconds = _estimate_remaining_seconds(
        active_states=active_states,
        queued=queued,
        jobs=jobs,
        avg_total_seconds=live_panel.average_total_seconds(),
    )
    pieces = [
        f"finished={finished}/{total}",
        f"ok={ok}",
        f"running={max(len(active_states), min(jobs, max(0, total - finished)))}",
        f"queued={queued}",
        f"fail={fail_count}",
        f"eta={_format_duration(eta_seconds)}",
    ]
    if repair_count:
        pieces.insert(4, f"needs_input={repair_count}")
    if active_states:
        active_labels = [
            f"{_short_label(str(state['sid']), 12)}:{_short_label(str(state['status']), 16)}"
            for state in active_states[:2]
        ]
        if len(active_states) > 2:
            active_labels.append(f"+{len(active_states) - 2}")
        pieces.append(f"active={'; '.join(active_labels)}")
    if success_fragment:
        pieces.append(success_fragment)
    return " | ".join(pieces)


def _runtime_repair_count(status_counts: dict[str, int]) -> int:
    return int(status_counts.get("needs_input_repair", 0))


def _runtime_finished_count(*, done_count: int, status_counts: dict[str, int]) -> int:
    return (
        done_count
        + int(status_counts.get("ok", 0))
        + int(status_counts.get("fail", 0))
        + _runtime_repair_count(status_counts)
    )


def _base_env(cfg: Step7Config) -> dict[str, str]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    env["PATH"] = (
        str(cfg.mrtrix_bin)
        + os.pathsep
        + str(cfg.fsl_dir / "bin")
        + os.pathsep
        + env.get("PATH", "")
    )
    env["FSLDIR"] = str(cfg.fsl_dir)
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    return env


def _env_with_threads(cfg: Step7Config, nthreads: int) -> dict[str, str]:
    env = _base_env(cfg)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = str(max(1, int(nthreads)))
    return env


def _tool(cfg: Step7Config, name: str) -> str:
    explicit = cfg.mrtrix_bin / name
    if explicit.exists():
        return str(explicit)
    found = shutil.which(name, path=_base_env(cfg)["PATH"])
    if not found:
        raise FileNotFoundError(f"Required tool not found: {name}")
    return found


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _visible_glob(path: Path, pattern: str):
    if not path.exists():
        return []
    return [p for p in path.glob(pattern) if not p.name.startswith("._")]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str], env: dict[str, str]) -> tuple[bool, str]:
    cp = sp.run(
        cmd,
        env=env,
        text=True,
        stdout=sp.PIPE,
        stderr=sp.STDOUT,
    )
    return cp.returncode == 0, cp.stdout


def _run_logged(
    cfg: Step7Config,
    sid: str,
    step_name: str,
    cmd: list[str],
    env: dict[str, str] | None = None,
    timeout_sec: int | None = None,
    cwd: Path | None = None,
) -> tuple[bool, Path, float]:
    env = (_base_env(cfg) if env is None else dict(env)).copy()
    if cwd is not None:
        cwd = Path(cwd)
        cwd.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(cwd)
    log_dir = cfg.qc_log_root / sid
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{time.time_ns()}_{step_name}.log"
    t0 = time.time()

    def _terminate_process_group(proc: sp.Popen[str], handle, reason: str) -> None:
        handle.write(f"\n[{reason}] Terminating process group for pid={proc.pid}\n")
        handle.flush()
        pgid: int | None
        try:
            pgid = os.getpgid(proc.pid)
        except Exception:
            pgid = None

        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pgid = None
            except Exception:
                pgid = None
        if pgid is None:
            try:
                proc.terminate()
            except Exception:
                pass

        try:
            proc.wait(timeout=5)
            return
        except sp.TimeoutExpired:
            handle.write(f"[{reason}] Escalating to SIGKILL for pid={proc.pid}\n")
            handle.flush()
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                pass
        else:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

    with open(log_path, "w", encoding="utf-8", errors="ignore") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        if cwd is not None:
            handle.write(f"[cwd] {cwd}\n\n")
        proc = sp.Popen(
            cmd,
            env=env,
            text=True,
            stdout=handle,
            stderr=sp.STDOUT,
            cwd=str(cwd) if cwd is not None else None,
            start_new_session=True,
        )
        try:
            ok = proc.wait(timeout=timeout_sec) == 0
        except sp.TimeoutExpired:
            handle.write(f"\n[TIMEOUT] Command exceeded {timeout_sec} seconds\n")
            _terminate_process_group(proc, handle, "TIMEOUT")
            ok = False
        except BaseException:
            _terminate_process_group(proc, handle, "ABORT")
            raise
    return ok, log_path, time.time() - t0


def _mr_nonzero_count(cfg: Step7Config, img: Path) -> int:
    if not img.exists():
        return 0
    env = _base_env(cfg)
    with tempfile.TemporaryDirectory() as td:
        binm = Path(td) / "bin.mif"
        stat_img = Path(td) / "stat.mif"
        sp.run(
            [_tool(cfg, "mrcalc"), str(img), "0", "-gt", str(binm)],
            env=env,
            check=True,
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
        )
        size = _mrinfo_size(cfg, img)
        if len(size) >= 4 and size[3] > 1:
            sp.run(
                [_tool(cfg, "mrmath"), str(binm), "sum", str(stat_img), "-axis", "3", "-force"],
                env=env,
                check=True,
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
            )
        else:
            stat_img = binm
        mean = sp.run(
            [_tool(cfg, "mrstats"), str(stat_img), "-output", "mean"],
            env=env,
            text=True,
            stdout=sp.PIPE,
            stderr=sp.DEVNULL,
        ).stdout.strip() or "0"
        count = sp.run(
            [_tool(cfg, "mrstats"), str(stat_img), "-output", "count"],
            env=env,
            text=True,
            stdout=sp.PIPE,
            stderr=sp.DEVNULL,
        ).stdout.strip() or "0"
        try:
            return int(round(float(mean) * float(count)))
        except Exception:
            return 0


def _safe_mr_nonzero_count(cfg: Step7Config, img: Path, *, context: str = "") -> int | None:
    try:
        return _mr_nonzero_count(cfg, img)
    except Exception as exc:
        context_note = f"{context} | " if context else ""
        _append_runtime_note(
            cfg.runtime_log,
            (
                f"Step7 probe warning {time.strftime('%Y-%m-%dT%H:%M:%S')} | "
                f"{context_note}img={img} | {type(exc).__name__}: {exc}"
            ),
        )
        return None


def _mr_image_has_nonzero_content(cfg: Step7Config, img: Path, *, context: str = "") -> bool:
    count = _safe_mr_nonzero_count(cfg, img, context=context)
    if count is not None:
        return count > 0
    return _looks_like_populated_mif(cfg, img)


def _mr_binary_overlap_count(
    cfg: Step7Config,
    img_a: Path,
    img_b: Path,
    *,
    context: str = "",
) -> int | None:
    if not img_a.exists() or not img_b.exists():
        return 0
    env = _base_env(cfg)
    try:
        with tempfile.TemporaryDirectory() as td:
            overlap = Path(td) / "overlap.mif"
            proc = sp.run(
                [
                    _tool(cfg, "mrcalc"),
                    str(img_a),
                    "0",
                    "-gt",
                    str(img_b),
                    "0",
                    "-gt",
                    "-mult",
                    str(overlap),
                    "-force",
                    "-quiet",
                ],
                env=env,
                text=True,
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stdout.strip() or "mrcalc overlap failed")
            return _safe_mr_nonzero_count(cfg, overlap, context=context)
    except Exception as exc:
        context_note = f"{context} | " if context else ""
        _append_runtime_note(
            cfg.runtime_log,
            (
                f"Step7 geometry warning {time.strftime('%Y-%m-%dT%H:%M:%S')} | "
                f"{context_note}a={img_a} | b={img_b} | {type(exc).__name__}: {exc}"
            ),
        )
        return None


def _count_for_audit(cfg: Step7Config, img: Path, *, context: str) -> int | None:
    if not img.exists():
        return 0
    return _safe_mr_nonzero_count(cfg, img, context=context)


def _tracks_geometry_audit(
    cfg: Step7Config,
    sid: str,
    paths: dict[str, Path],
    *,
    include_fod_mask: bool = True,
) -> dict[str, int | str | None]:
    audit: dict[str, int | str | None] = {
        "sid": sid,
        "five_fix_nz": _count_for_audit(cfg, paths["FIVE_FIX"], context=f"tracks_audit sid={sid} five_fix"),
        "mask_5tt_nz": _count_for_audit(cfg, paths["MASK_5TT_BRAIN"], context=f"tracks_audit sid={sid} mask_5tt"),
        "gmwmi_nz": _count_for_audit(cfg, paths["GMWMI"], context=f"tracks_audit sid={sid} gmwmi"),
        "fod_mask_nz": None,
        "gmwmi_fod_overlap": None,
        "fod_5tt_overlap": None,
        "gmwmi_5tt_overlap": _mr_binary_overlap_count(
            cfg,
            paths["GMWMI"],
            paths["MASK_5TT_BRAIN"],
            context=f"tracks_audit sid={sid} gmwmi_5tt",
        ),
    }
    if include_fod_mask:
        audit["fod_mask_nz"] = _count_for_audit(
            cfg,
            paths["FOD_MASK"],
            context=f"tracks_audit sid={sid} fod_mask",
        )
        audit["gmwmi_fod_overlap"] = _mr_binary_overlap_count(
            cfg,
            paths["GMWMI"],
            paths["FOD_MASK"],
            context=f"tracks_audit sid={sid} gmwmi_fod",
        )
        audit["fod_5tt_overlap"] = _mr_binary_overlap_count(
            cfg,
            paths["FOD_MASK"],
            paths["MASK_5TT_BRAIN"],
            context=f"tracks_audit sid={sid} fod_5tt",
        )
    return audit


def _audit_int(audit: dict[str, int | str | None], key: str) -> int | None:
    value = audit.get(key)
    return value if isinstance(value, int) else None


def _format_tracks_geometry_audit(audit: dict[str, int | str | None]) -> str:
    keys = (
        "five_fix_nz",
        "mask_5tt_nz",
        "gmwmi_nz",
        "fod_mask_nz",
        "gmwmi_fod_overlap",
        "fod_5tt_overlap",
        "gmwmi_5tt_overlap",
    )
    return "; ".join(f"{key}={audit.get(key)}" for key in keys if audit.get(key) is not None)


def _validate_prep_geometry(cfg: Step7Config, sid: str, paths: dict[str, Path]) -> dict[str, int | str | None]:
    audit = _tracks_geometry_audit(cfg, sid, paths, include_fod_mask=False)
    if (_audit_int(audit, "five_fix_nz") or 0) <= 0:
        raise Step7GeometryError(f"invalid_prep_geometry: 5TT image is empty; {_format_tracks_geometry_audit(audit)}")
    if (_audit_int(audit, "mask_5tt_nz") or 0) <= 0:
        raise Step7GeometryError(f"invalid_prep_geometry: 5TT brain mask is empty; {_format_tracks_geometry_audit(audit)}")
    if (_audit_int(audit, "gmwmi_nz") or 0) <= 0:
        raise Step7GeometryError(f"empty_gmwmi: GMWMI seed image is empty; {_format_tracks_geometry_audit(audit)}")
    return audit


def _validate_fod_act_geometry(
    cfg: Step7Config,
    sid: str,
    paths: dict[str, Path],
    *,
    fod_mask_mode: str = "",
) -> dict[str, int | str | None]:
    audit = _tracks_geometry_audit(cfg, sid, paths, include_fod_mask=True)
    if (_audit_int(audit, "fod_mask_nz") or 0) <= 0:
        raise Step7GeometryError(f"empty_fod_mask: FOD mask is empty; {_format_tracks_geometry_audit(audit)}")
    if (_audit_int(audit, "gmwmi_nz") or 0) <= 0:
        raise Step7GeometryError(f"empty_gmwmi: GMWMI seed image is empty; {_format_tracks_geometry_audit(audit)}")
    if (_audit_int(audit, "gmwmi_fod_overlap") or 0) <= 0:
        mode_note = f"fod_mask_mode={fod_mask_mode}; " if fod_mask_mode else ""
        raise Step7GeometryError(
            "invalid_act_fod_geometry: GMWMI seed image has zero overlap with the FOD mask; "
            f"{mode_note}{_format_tracks_geometry_audit(audit)}"
        )
    if (_audit_int(audit, "fod_5tt_overlap") or 0) <= 0:
        mode_note = f"fod_mask_mode={fod_mask_mode}; " if fod_mask_mode else ""
        raise Step7GeometryError(
            "invalid_act_fod_geometry: FOD mask has zero overlap with the 5TT brain mask; "
            f"{mode_note}{_format_tracks_geometry_audit(audit)}"
        )
    return audit


def _tck_header_count(tck: Path) -> int | None:
    try:
        with open(tck, "rb") as handle:
            header = handle.read(65536).decode("utf-8", errors="replace")
    except OSError:
        return None
    if "\nEND\n" not in header and not header.rstrip().endswith("\nEND"):
        return None
    for line in header.splitlines():
        line = line.strip()
        if not line.startswith("count:"):
            continue
        try:
            return int(line.split(":", 1)[1].strip().split()[0])
        except Exception:
            return None
    return None


def _tck_count(cfg: Step7Config, tck: Path) -> int:
    if not tck.exists():
        return 0
    header_count = _tck_header_count(tck)
    if header_count is not None:
        return header_count
    out = sp.run(
        [_tool(cfg, "tckinfo"), "-count", str(tck)],
        env=_base_env(cfg),
        text=True,
        stdout=sp.PIPE,
        stderr=sp.DEVNULL,
    ).stdout
    for line in out.splitlines():
        if line.strip().startswith("count:"):
            try:
                return int(line.split()[-1])
            except Exception:
                return 0
    return 0


def _parse_tckgen_outcome(log_path: Path) -> dict[str, int | str]:
    result: dict[str, int | str] = {
        "seeds": 0,
        "streamlines": 0,
        "selected": 0,
        "reason": "unknown_zero_tracks",
    }
    if not log_path.exists():
        return result
    last_match: tuple[int, int, int] | None = None
    pattern = re.compile(r"tckgen:\s+\[\s*\d+%\]\s+(\d+)\s+seeds,\s+(\d+)\s+streamlines,\s+(\d+)\s+selected")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            last_match = tuple(int(match.group(i)) for i in range(1, 4))
    if last_match is None:
        return result
    seeds, streamlines, selected = last_match
    result["seeds"] = seeds
    result["streamlines"] = streamlines
    result["selected"] = selected
    if selected > 0:
        result["reason"] = "selected_nonzero_but_empty_file"
    elif streamlines > 0:
        result["reason"] = "act_rejected_all_streamlines"
    elif seeds > 0:
        result["reason"] = "no_streamlines_generated"
    return result


def classify_tracks_failure_note(note: str, retry_notes: list[str] | tuple[str, ...] = ()) -> str:
    text = "\n".join([note or "", *(retry_notes or ())]).lower()
    if "seed_mode=dynamic" in text and "all tractography cutoffs failed" in text:
        return "dynamic_act_failed"
    if "empty_gmwmi" in text or "gmwmi seed image is empty" in text:
        return "empty_gmwmi"
    if "invalid_tracks_geometry" in text and "zero overlap" in text:
        return "zero_gmwmi_fod_overlap"
    if "invalid_act_fod_geometry" in text and "zero overlap" in text:
        return "zero_act_fod_overlap"
    if "all tractography cutoffs failed" in text:
        return "all_cutoffs_failed"
    if "act_rejected_all_streamlines" in text:
        return "act_rejected_all_streamlines"
    if "no_streamlines_generated" in text:
        return "no_streamlines_generated"
    if "chunk failed" in text:
        return "chunk_failed"
    return "other"


def collect_tracks_repair_targets(
    summary_csv: Path,
    *,
    after_row: int = 0,
) -> dict[str, object]:
    latest: dict[str, dict[str, str]] = {}
    retries: dict[str, list[dict[str, str]]] = {}
    seen_rows = 0
    if not summary_csv.exists():
        return {
            "summary_csv": str(summary_csv),
            "after_row": int(after_row),
            "current_data_rows": 0,
            "failures": {},
            "geometry": [],
            "dynamic": [],
            "other": [],
        }
    with summary_csv.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row_no, row in enumerate(csv.DictReader(handle), start=1):
            seen_rows = row_no
            if row_no <= after_row:
                continue
            sid = (row.get("sid") or "").strip()
            stage = (row.get("stage") or "").strip()
            if not sid or stage not in {"tracks", "tracks_retry"}:
                continue
            if stage == "tracks_retry":
                retries.setdefault(sid, []).append(row)
            else:
                latest[sid] = row

    failures: dict[str, str] = {}
    for sid, row in latest.items():
        if (row.get("status") or "").strip() != "error":
            continue
        retry_notes = [(retry.get("note") or "") for retry in retries.get(sid, [])]
        mechanism = classify_tracks_failure_note(row.get("note") or "", retry_notes)
        text = "\n".join([(row.get("note") or ""), *retry_notes]).lower()
        if mechanism in {"empty_gmwmi", "zero_gmwmi_fod_overlap", "zero_act_fod_overlap"}:
            failures[sid] = mechanism
        elif mechanism == "all_cutoffs_failed" and "tckgen probe produced 0 tracks" in text:
            failures[sid] = mechanism
        elif mechanism in {"act_rejected_all_streamlines", "no_streamlines_generated"} and "seed_mode=dynamic" not in text:
            failures[sid] = "all_cutoffs_failed"
        elif mechanism not in {"other"}:
            failures[sid] = mechanism

    geometry = sorted(
        sid
        for sid, mechanism in failures.items()
        if mechanism in {"empty_gmwmi", "zero_gmwmi_fod_overlap", "zero_act_fod_overlap"}
    )
    dynamic = sorted(sid for sid, mechanism in failures.items() if mechanism == "all_cutoffs_failed")
    other = sorted(sid for sid in failures if sid not in set(geometry) | set(dynamic))
    return {
        "summary_csv": str(summary_csv),
        "after_row": int(after_row),
        "current_data_rows": seen_rows,
        "failures": failures,
        "geometry": geometry,
        "dynamic": dynamic,
        "other": other,
    }


def read_step7_subject_file(path: Path | str) -> list[str]:
    path = Path(path)
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="ascii", errors="ignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def unique_subjects(items: list[str] | tuple[str, ...] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items or []:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _latest_stage_rows(cfg: Step7Config, subjects: list[str] | tuple[str, ...]) -> dict[tuple[str, str], dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    targets = set(subjects or [])
    if not targets or not cfg.summary_csv.exists():
        return latest
    try:
        with cfg.summary_csv.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                sid = (row.get("sid") or "").strip()
                stage = (row.get("stage") or "").strip()
                if sid in targets and stage in {"prep", "fod", "tracks", "post"}:
                    latest[(sid, stage)] = row
    except Exception:
        return {}
    return latest


def _summary_status(latest: dict[tuple[str, str], dict[str, str]], sid: str, stage: str) -> str:
    return (latest.get((sid, stage), {}).get("status") or "").strip()


def collect_step7_repair_ui_targets(
    cfg: Step7Config,
    repair_target_dir: Path | str | None = None,
) -> dict[str, object]:
    repair_dir = Path(repair_target_dir) if repair_target_dir is not None else Path(__file__).resolve().parent / "step7_tracks_tmux_latest"
    geometry = read_step7_subject_file(repair_dir / "tracks_repair_geometry_force_prep_fod.txt")
    dynamic = read_step7_subject_file(repair_dir / "tracks_repair_dynamic_seed.txt")
    other = read_step7_subject_file(repair_dir / "tracks_repair_other_review.txt")
    latest = _latest_stage_rows(cfg, geometry + dynamic + other)

    geometry_fod_pending: list[str] = []
    geometry_fod_blocked: dict[str, str] = {}
    for sid in geometry:
        prep_status = _summary_status(latest, sid, "prep")
        fod_status = _summary_status(latest, sid, "fod")
        if prep_status and prep_status != "ok":
            geometry_fod_blocked[sid] = f"prep_{prep_status}"
            continue
        if fod_status != "ok":
            geometry_fod_pending.append(sid)

    geometry_tracks_ready = [sid for sid in geometry if _summary_status(latest, sid, "fod") == "ok"]
    dynamic_tracks_ready = list(dynamic)
    bad_fod_dynamic = [sid for sid in dynamic if _summary_status(latest, sid, "fod") not in {"", "ok"}]
    bad_fod_details = {sid: _summary_status(latest, sid, "fod") for sid in bad_fod_dynamic}
    return {
        "repair_target_dir": str(repair_dir),
        "geometry": geometry,
        "dynamic": dynamic,
        "other": other,
        "geometry_fod_pending": geometry_fod_pending,
        "geometry_fod_blocked": geometry_fod_blocked,
        "geometry_tracks_ready": geometry_tracks_ready,
        "dynamic_tracks_ready": dynamic_tracks_ready,
        "bad_fod_dynamic": bad_fod_dynamic,
        "bad_fod_details": bad_fod_details,
        "all": unique_subjects(geometry + dynamic + other),
    }


def resolve_step7_stage_lanes(
    cfg: Step7Config,
    stage: str,
    include: list[str] | tuple[str, ...] | None = None,
    *,
    target_mode: str = "repair",
    repair_target_dir: Path | str | None = None,
    force: bool | None = None,
    track_seed_mode: str | None = None,
) -> list[dict[str, object]]:
    stage = str(stage).strip().lower()
    mode = str(target_mode or "repair").strip().lower()
    manual_include = unique_subjects(list(include or []))
    force_value = cfg.force if force is None else bool(force)
    seed_mode = track_seed_mode or cfg.track_seed_mode
    if manual_include or mode == "full":
        return [
            {
                "label": "manual" if manual_include else "full",
                "stage": stage,
                "include": manual_include or None,
                "force": force_value,
                "track_seed_mode": seed_mode,
            }
        ]

    targets = collect_step7_repair_ui_targets(cfg, repair_target_dir)
    lanes: list[dict[str, object]] = []
    if stage == "prep":
        if targets["geometry"]:
            lanes.append(
                {
                    "label": "repair_geometry_prep",
                    "stage": "prep",
                    "include": targets["geometry"],
                    "force": True if force is None else bool(force),
                    "track_seed_mode": "gmwmi",
                }
            )
    elif stage == "fod":
        if targets["geometry_fod_pending"]:
            lanes.append(
                {
                    "label": "repair_geometry_fod_pending",
                    "stage": "fod",
                    "include": targets["geometry_fod_pending"],
                    "force": True if force is None else bool(force),
                    "track_seed_mode": "gmwmi",
                }
            )
        if targets["bad_fod_dynamic"]:
            lanes.append(
                {
                    "label": "repair_dynamic_bad_fod",
                    "stage": "fod",
                    "include": targets["bad_fod_dynamic"],
                    "force": True if force is None else bool(force),
                    "track_seed_mode": "dynamic",
                }
            )
    elif stage == "tracks":
        if targets["geometry_tracks_ready"]:
            lanes.append(
                {
                    "label": "repair_geometry_tracks_ready",
                    "stage": "tracks",
                    "include": targets["geometry_tracks_ready"],
                    "force": force_value,
                    "track_seed_mode": "gmwmi",
                }
            )
        if targets["dynamic_tracks_ready"]:
            lanes.append(
                {
                    "label": "repair_dynamic_tracks_ready",
                    "stage": "tracks",
                    "include": targets["dynamic_tracks_ready"],
                    "force": force_value,
                    "track_seed_mode": "dynamic",
                }
            )
    elif stage == "post":
        if targets["all"]:
            lanes.append(
                {
                    "label": "repair_post",
                    "stage": "post",
                    "include": targets["all"],
                    "force": force_value,
                    "track_seed_mode": seed_mode,
                }
            )
    else:
        lanes.append(
            {
                "label": "repair_auto",
                "stage": stage,
                "include": targets["all"] or None,
                "force": force_value,
                "track_seed_mode": seed_mode,
            }
        )
    return lanes


def _parse_weights_txt(weights_path: Path) -> list[float]:
    values: list[float] = []
    if not weights_path.exists():
        return values
    with open(weights_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for token in line.split():
                try:
                    values.append(float(token))
                except Exception:
                    pass
    return values


def _scalar_txt_stats(path: Path) -> dict[str, float | int | bool]:
    count = 0
    min_value = float("inf")
    max_value = -float("inf")
    finite = True
    parse_ok = True
    if not path.exists():
        return {"count": 0, "min": float("nan"), "max": float("nan"), "finite": False, "parse_ok": False}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for token in line.split():
                try:
                    value = float(token)
                except Exception:
                    parse_ok = False
                    continue
                count += 1
                if not math.isfinite(value):
                    finite = False
                    continue
                min_value = min(min_value, value)
                max_value = max(max_value, value)
    if count == 0:
        min_value = float("nan")
        max_value = float("nan")
    return {"count": count, "min": min_value, "max": max_value, "finite": finite, "parse_ok": parse_ok}


def _weights_ready_for_tracks(weights_path: Path, track_count: int) -> bool:
    stats = _scalar_txt_stats(weights_path)
    return (
        bool(stats["parse_ok"])
        and bool(stats["finite"])
        and int(stats["count"]) >= track_count
        and float(stats["min"]) >= -1e-12
    )


def _validate_track_scalar_file(
    path: Path,
    *,
    expected_count: int,
    label: str,
    min_allowed: float,
    max_allowed: float,
) -> None:
    stats = _scalar_txt_stats(path)
    if int(stats["count"]) < expected_count:
        raise RuntimeError(f"{label} scalar file is short: {stats['count']} < {expected_count}")
    if not bool(stats["parse_ok"]) or not bool(stats["finite"]):
        raise RuntimeError(f"{label} scalar file contains non-numeric or non-finite values")
    min_value = float(stats["min"])
    max_value = float(stats["max"])
    if min_value < min_allowed or max_value > max_allowed:
        raise RuntimeError(
            f"{label} scalar values outside expected range: "
            f"min={min_value:g}, max={max_value:g}, allowed=[{min_allowed:g}, {max_allowed:g}]"
        )


def _read_single_float(path: Path, *, label: str) -> float:
    try:
        value = float(path.read_text(encoding="utf-8", errors="replace").strip().split()[0])
    except Exception as exc:
        raise RuntimeError(f"{label} is missing or not numeric: {path}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"{label} is non-finite: {value}")
    return value


def _validate_sift2_outputs(paths: dict[str, Path], track_count: int) -> dict[str, float | int | bool]:
    stats = _scalar_txt_stats(paths["WEIGHTS"])
    if int(stats["count"]) < track_count:
        raise RuntimeError(f"SIFT2 weights are short: {stats['count']} < {track_count}")
    if not bool(stats["parse_ok"]) or not bool(stats["finite"]):
        raise RuntimeError("SIFT2 weights contain non-numeric or non-finite values")
    if float(stats["min"]) < -1e-12:
        raise RuntimeError(f"SIFT2 weights contain negative values: min={float(stats['min']):g}")
    mu = _read_single_float(paths["MU_TXT"], label="SIFT2 mu")
    if mu <= 0:
        raise RuntimeError(f"SIFT2 mu is non-positive: {mu:g}")
    stats["mu"] = mu
    return stats


def _sift2_ready_for_tracks(paths: dict[str, Path], track_count: int) -> bool:
    if not (paths["WEIGHTS"].exists() and paths["MU_TXT"].exists()):
        return False
    try:
        _validate_sift2_outputs(paths, track_count)
    except Exception:
        return False
    return True


def _read_numeric_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = _UNICODE_MINUS.sub("-", line)
            tokens: list[float] = []
            ok = True
            for token in line.replace("\t", " ").split():
                try:
                    tokens.append(float(token))
                except Exception:
                    ok = False
                    break
            if ok and tokens:
                rows.append(tokens)
    return rows


def _sanitize_response_txt_inplace(path: Path) -> bool:
    if not path.exists():
        return False
    rows = _read_numeric_rows(path)
    if not rows:
        return False
    with open(path, "w", encoding="ascii", errors="ignore") as handle:
        for row in rows:
            handle.write(" ".join(f"{x:.16g}" for x in row) + "\n")
    return True


def _write_single_shell_response_txt(src: Path, dst: Path) -> bool:
    rows = _read_numeric_rows(src)
    if not rows:
        return False
    row = rows[-1]
    with open(dst, "w", encoding="ascii", errors="ignore") as handle:
        handle.write(" ".join(f"{x:.16g}" for x in row) + "\n")
    return True


def _has_numeric_rows(path: Path) -> bool:
    return bool(_read_numeric_rows(path))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _is_dhollander_sparse_mask_failure(log_path: Path) -> bool:
    text = _read_text(log_path)
    return any(
        needle in text
        for needle in (
            "Error trying to calculate statistics from image 'eroded_mask.mif'",
            "Error trying to calculate statistics from image 'safe_mask.mif'",
            "Error trying to calculate statistics from image 'crude_gm.mif'",
            "Error trying to calculate statistics from image 'crude_csf.mif'",
        )
    )


def _symlink_or_copy(src: Path, dst: Path) -> None:
    _unlink(dst)
    try:
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def _write_summary_row(
    cfg: Step7Config,
    sid: str,
    stage: str,
    status: str,
    note: str = "",
    mode: str = "",
) -> None:
    _ensure_parent(cfg.summary_csv)
    with open(cfg.summary_csv, "a+", newline="", encoding="utf-8", errors="replace") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        new_file = handle.tell() == 0
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["sid", "stage", "status", "mode", "note"])
        writer.writerow([sid, stage, status, mode, note])
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_input_repair_row(
    cfg: Step7Config,
    sid: str,
    stage: str,
    issue: str,
    *,
    mode: str = "",
) -> None:
    repair_tsv = cfg.deriv_root / "qc" / "step7_input_repair_queue.tsv"
    _ensure_parent(repair_tsv)
    new_file = not repair_tsv.exists()
    if not new_file:
        try:
            with open(repair_tsv, "r", newline="", encoding="utf-8", errors="replace") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    if (
                        (row.get("sid") or "") == sid
                        and (row.get("stage") or "") == stage
                        and (row.get("mode") or "") == mode
                        and (row.get("issue") or "") == issue
                    ):
                        return
        except Exception:
            pass
    with open(repair_tsv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if new_file:
            writer.writerow(["timestamp", "sid", "stage", "mode", "issue"])
        writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), sid, stage, mode, issue])


def _fod_input_qc_failed(paths: dict[str, Path]) -> bool:
    return paths["FOD_INPUT_QC_FAILED"].exists()


def _read_fod_input_qc_issue(paths: dict[str, Path]) -> str:
    text = _read_text(paths["FOD_INPUT_QC_FAILED"]).strip()
    for line in text.splitlines():
        if line.startswith("issue="):
            return line.split("=", 1)[1].strip()
    return text.splitlines()[-1].strip() if text else ""


def _write_fod_input_qc_failure(
    cfg: Step7Config,
    sid: str,
    stage: str,
    issue: str,
    *,
    mode: str = "",
) -> None:
    paths = _subject_paths(cfg, sid, create_dirs=True)
    marker = paths["FOD_INPUT_QC_FAILED"]
    _ensure_parent(marker)
    marker.write_text(
        "\n".join(
            [
                f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%S')}",
                f"sid={sid}",
                f"stage={stage}",
                f"mode={mode}",
                f"issue={issue}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _read_square_csv(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split(",")])
    return rows


_ATLAS_NODE_COUNT_CACHE: dict[str, int] = {}


def _atlas_node_count(cfg: Step7Config) -> int | None:
    key = str(cfg.aal_mni)
    if key in _ATLAS_NODE_COUNT_CACHE:
        return _ATLAS_NODE_COUNT_CACHE[key]
    if not cfg.aal_mni.exists():
        return None
    out = sp.run(
        [_tool(cfg, "mrstats"), str(cfg.aal_mni), "-ignorezero", "-output", "max"],
        env=_base_env(cfg),
        text=True,
        stdout=sp.PIPE,
        stderr=sp.DEVNULL,
        check=False,
    ).stdout.strip()
    try:
        value = int(round(float(out.split()[0])))
    except Exception:
        return None
    if value > 0:
        _ATLAS_NODE_COUNT_CACHE[key] = value
        return value
    return None


def _labelstats_masses(cfg: Step7Config, img: Path) -> list[float]:
    proc = sp.run(
        [_tool(cfg, "labelstats"), str(img)],
        env=_base_env(cfg),
        text=True,
        stdout=sp.PIPE,
        stderr=sp.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return []
    masses: list[float] = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            int(float(parts[0]))
            mass = float(parts[1])
        except Exception:
            continue
        if mass > 0:
            masses.append(mass)
    return masses


def _xyz_size(cfg: Step7Config, img: Path) -> tuple[int, int, int] | None:
    size = _mrinfo_size(cfg, img)
    if len(size) < 3:
        return None
    return tuple(size[:3])


def _mrstats_outputs(
    cfg: Step7Config,
    img: Path,
    outputs: list[str],
    *,
    mask: Path | None = None,
    ignorezero: bool = False,
    label: str,
) -> list[float]:
    cmd = [_tool(cfg, "mrstats"), str(img)]
    if ignorezero:
        cmd.append("-ignorezero")
    if mask is not None and mask.exists():
        cmd += ["-mask", str(mask)]
    for output in outputs:
        cmd += ["-output", output]
    proc = sp.run(
        cmd,
        env=_base_env(cfg),
        text=True,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [f"rc={proc.returncode}"]
        raise RuntimeError(f"{label} mrstats failed: {tail[0]}")
    values: list[float] = []
    for token in proc.stdout.split():
        try:
            values.append(float(token))
        except Exception:
            pass
    if len(values) < len(outputs) or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{label} mrstats returned invalid values: {proc.stdout.strip()!r}")
    return values


def _validate_aal_b0(cfg: Step7Config, paths: dict[str, Path], node_count: int | None) -> dict[str, float | int]:
    if not paths["AAL_B0"].exists():
        raise RuntimeError("AAL_B0 is missing")
    aal_xyz = _xyz_size(cfg, paths["AAL_B0"])
    b0_xyz = _xyz_size(cfg, paths["B0MEAN"])
    if aal_xyz is None or b0_xyz is None or aal_xyz != b0_xyz:
        raise RuntimeError(f"AAL_B0 geometry mismatch: aal={aal_xyz}, b0={b0_xyz}")
    count, max_label = _mrstats_outputs(
        cfg,
        paths["AAL_B0"],
        ["count", "max"],
        ignorezero=True,
        label="AAL_B0",
    )
    if count < 1000:
        raise RuntimeError(f"AAL_B0 has too few labelled voxels: count={count:g}")
    masses = _labelstats_masses(cfg, paths["AAL_B0"])
    label_count = len(masses)
    robust_label_count = sum(1 for mass in masses if mass >= 50)
    min_labels = min(30, node_count or 30)
    if robust_label_count < min_labels:
        raise RuntimeError(
            f"AAL_B0 label coverage is implausibly low: labels>50vox={robust_label_count}, "
            f"required>={min_labels}, max_label={max_label:g}"
        )
    return {
        "count": int(round(count)),
        "max_label": int(round(max_label)),
        "label_count": int(label_count),
        "labels_ge_50vox": int(robust_label_count),
    }


def _post_tensor_mask(paths: dict[str, Path]) -> Path:
    return next(
        (
            paths[key]
            for key in ("MASK_5TT_BRAIN", "MASK_B0", "FOD_MASK", "DWI_MASK")
            if paths[key].exists()
        ),
        paths["FOD_MASK"],
    )


def _ensure_post_dwi_mask(cfg: Step7Config, sid: str, paths: dict[str, Path], tensor_dwi: Path) -> None:
    if paths["DWI_MASK"].exists() and _mr_nonzero_count(cfg, paths["DWI_MASK"]) >= 1000:
        return
    _unlink(paths["DWI_MASK"])
    ok, logp, _ = _run_logged(
        cfg,
        sid,
        "dwi2mask_post_dti",
        [_tool(cfg, "dwi2mask"), str(tensor_dwi), str(paths["DWI_MASK"]), "-force"],
    )
    if not ok:
        raise RuntimeError(f"dwi2mask failed for post DTI repair; see {logp}")
    if _mr_nonzero_count(cfg, paths["DWI_MASK"]) < 1000:
        _append_runtime_note(
            cfg.runtime_log,
            f"Step7 post DTI mask fallback {time.strftime('%Y-%m-%dT%H:%M:%S')} | sid={sid} | "
            "dwi2mask produced too few nonzero voxels; using alternate existing mask",
        )
        _unlink(paths["DWI_MASK"])


def _validate_dti_maps(cfg: Step7Config, paths: dict[str, Path]) -> dict[str, dict[str, float]]:
    mask = _post_tensor_mask(paths)
    stats_by_metric: dict[str, dict[str, float]] = {}
    for metric, key in (("fa", "FA_MIF"), ("md", "MD_MIF"), ("ad", "AD_MIF"), ("rd", "RD_MIF")):
        count, mean, min_value, max_value = _mrstats_outputs(
            cfg,
            paths[key],
            ["count", "mean", "min", "max"],
            mask=mask,
            label=f"{metric.upper()} map",
        )
        if count < 1000:
            raise RuntimeError(f"{metric.upper()} map has too few in-mask voxels: count={count:g}")
        if metric == "fa":
            if min_value < -1e-6 or not (0.0 <= mean <= 1.05) or max_value > 1.3:
                raise RuntimeError(
                    f"FA map outside expected range: mean={mean:g}, min={min_value:g}, max={max_value:g}"
                )
        else:
            if not (-0.02 <= mean <= 0.05) or min_value < -0.1 or max_value > 1.0:
                raise RuntimeError(
                    f"{metric.upper()} map outside expected range: "
                    f"mean={mean:g}, min={min_value:g}, max={max_value:g}"
                )
        stats_by_metric[metric] = {"count": count, "mean": mean, "min": min_value, "max": max_value}
    return stats_by_metric


def _sanitize_dti_scalar_maps(cfg: Step7Config, sid: str, paths: dict[str, Path]) -> None:
    for key in ("FA_MIF", "MD_MIF", "AD_MIF", "RD_MIF"):
        src = paths[key]
        tmp = src.with_name(f"{src.stem}_finite_tmp{src.suffix}")
        _unlink(tmp)
        ok, logp, _ = _run_logged(
            cfg,
            sid,
            f"sanitize_{src.stem}",
            [
                _tool(cfg, "mrcalc"),
                str(src),
                "-finite",
                str(src),
                "0",
                "-if",
                str(tmp),
                "-force",
            ],
        )
        if not ok:
            raise RuntimeError(f"DTI scalar finite sanitation failed for {src.name}; see {logp}")
        tmp.replace(src)


def _pad_square_matrix(matrix: list[list[float]], node_count: int, *, label: str) -> list[list[float]]:
    n = len(matrix)
    if n <= 0 or any(len(row) != n for row in matrix):
        raise RuntimeError(f"{label} connectome matrix is not square")
    if n > node_count:
        raise RuntimeError(f"{label} connectome matrix has {n} nodes, expected <= {node_count}")
    if n == node_count:
        return matrix
    out = [[0.0 for _ in range(node_count)] for _ in range(node_count)]
    for i, row in enumerate(matrix):
        out[i][:n] = row
    return out


def _write_square_csv(path: Path, matrix: list[list[float]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix)


def _sanitize_connectome_matrix(
    matrix: list[list[float]],
    count_matrix: list[list[float]],
    *,
    metric: str,
    node_count: int,
) -> list[list[float]]:
    matrix = _pad_square_matrix(matrix, node_count, label=metric)
    out: list[list[float]] = []
    for i, row in enumerate(matrix):
        out_row: list[float] = []
        for j, value in enumerate(row):
            count_value = count_matrix[i][j]
            if not math.isfinite(value):
                if count_value <= 0:
                    value = 0.0
                else:
                    raise RuntimeError(f"{metric} has non-finite value on connected edge ({i}, {j})")
            if metric in {"count", "fd_sum", "len_mean", "invlen_mean", "fa_mean", "md_mean", "ad_mean", "rd_mean"} and value < -1e-10:
                if metric in {"md_mean", "ad_mean", "rd_mean"} and value >= -0.02:
                    pass
                else:
                    raise RuntimeError(f"{metric} has negative value {value:g} on edge ({i}, {j})")
            if metric == "len_mean" and count_value > 0 and value > 500:
                raise RuntimeError(f"len_mean is implausibly high on edge ({i}, {j}): {value:g}")
            if metric == "invlen_mean" and count_value > 0 and value > 1:
                raise RuntimeError(f"invlen_mean is implausibly high on edge ({i}, {j}): {value:g}")
            if metric == "fa_mean" and count_value > 0 and not (0 <= value <= 1.3):
                raise RuntimeError(f"fa_mean is outside expected range on edge ({i}, {j}): {value:g}")
            if metric in {"md_mean", "ad_mean", "rd_mean"} and count_value > 0 and not (-0.02 <= value <= 0.05):
                raise RuntimeError(f"{metric} is outside expected range on edge ({i}, {j}): {value:g}")
            out_row.append(0.0 if abs(value) < 1e-15 else value)
        out.append(out_row)
    return out


def _write_long_csv(path: Path, matrices: dict[str, list[list[float]]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["i", "j", "metric", "value"])
        size = len(next(iter(matrices.values())))
        for i in range(size):
            for j in range(size):
                for metric_name, matrix in matrices.items():
                    writer.writerow([i, j, metric_name, matrix[i][j]])


def _connectome_primary_summary(matrices: dict[str, list[list[float]]]) -> dict[str, float | int | str]:
    primary_name = "fd_sum" if "fd_sum" in matrices else "count"
    matrix = matrices.get(primary_name)
    if not matrix:
        return {"primary_metric": primary_name, "density": math.nan, "zero_rows": math.nan, "nonzero_upper_edges": math.nan}
    n = len(matrix)
    nonzero_upper = 0
    possible_upper = max(1, n * (n - 1) // 2)
    zero_rows = 0
    for i, row in enumerate(matrix):
        row_sum = 0.0
        for j, value in enumerate(row):
            try:
                val = float(value)
            except Exception:
                val = 0.0
            if math.isfinite(val):
                row_sum += abs(val)
            if i < j and math.isfinite(val) and abs(val) > 0:
                nonzero_upper += 1
        if row_sum == 0:
            zero_rows += 1
    return {
        "primary_metric": primary_name,
        "density": nonzero_upper / possible_upper,
        "zero_rows": zero_rows,
        "nonzero_upper_edges": nonzero_upper,
    }


def _write_sc_generation_provenance(
    cfg: Step7Config,
    sid: str,
    paths: dict[str, Path],
    matrices: dict[str, list[list[float]]],
    *,
    track_count: int,
    node_count: int | None,
    weight_stats: dict[str, float | int],
    track_method: str,
    mode: str,
) -> None:
    matrix_keys = [
        "SC_COUNT",
        "SC_FD_SUM",
        "SC_LEN_MEAN",
        "SC_INVLEN_MEAN",
        "SC_FA_MEAN",
        "SC_MD_MEAN",
        "SC_AD_MEAN",
        "SC_RD_MEAN",
        "SC_ALL",
    ]
    payload = {
        "sid": sid,
        "mode": mode,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "node_count": node_count,
        "track_count": track_count,
        "sift_weight_count": int(weight_stats.get("count", 0) or 0),
        "assignment_radial_search": cfg.assignment_radial_search,
        "assignment_variant": cfg.assignment_variant,
        "assignment_args": _assignment_args(cfg),
        "t1_b0_route": cfg.t1_b0_route,
        "five_tt_route": cfg.five_tt_route,
        "aal_transform_route": cfg.aal_transform_route,
        "backup_before_replace": cfg.backup_before_replace,
        "validation_gate": cfg.validation_gate,
        "write_assignments": cfg.write_assignments,
        "track_method": track_method,
        "required_matrices": ["count", "fd_sum", "len_mean", "fa_mean", "md_mean", "rd_mean", "ad_mean"],
        "matrix_paths": {key: str(paths[key]) for key in matrix_keys if key in paths},
        "assignment_path": str(paths["ASSIGNMENTS"]),
        "aal_b0_path": str(paths["AAL_B0"]),
        "tracks_path": str(paths["TRACKS_FINAL"]),
        "weights_path": str(paths["WEIGHTS"]),
        "sc_qc_gate_required": bool(cfg.validation_gate),
        "sc_qc_gate_path": str(cfg.analysis_gate_csv),
        "final_rectification_decision": (
            "Downstream SC matrix repair from current artifacts is not batch-approved; "
            "FAIL subjects require upstream source-contract replay/full preprocessing validation before promotion."
        ),
        "primary_summary": _connectome_primary_summary(matrices),
    }
    tmp = paths["SC_PROVENANCE"].with_suffix(paths["SC_PROVENANCE"].suffix + ".tmp")
    paths["SC_PROVENANCE"].parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(paths["SC_PROVENANCE"])


def _write_spatial_contract_status(
    cfg: Step7Config,
    sid: str,
    paths: dict[str, Path],
    matrices: dict[str, list[list[float]]],
    *,
    track_count: int,
    node_count: int | None,
    mode: str,
) -> None:
    """Append a compact route/QC status row for Step 7 post outputs."""
    required_keys = [
        "SC_COUNT",
        "SC_FD_SUM",
        "SC_LEN_MEAN",
        "SC_FA_MEAN",
        "SC_MD_MEAN",
        "SC_RD_MEAN",
        "SC_AD_MEAN",
    ]
    summary = _connectome_primary_summary(matrices)
    row = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sid": sid,
        "stage": "post",
        "mode": mode,
        "t1_b0_route": cfg.t1_b0_route,
        "five_tt_route": cfg.five_tt_route,
        "aal_transform_route": cfg.aal_transform_route,
        "assignment_variant": cfg.assignment_variant,
        "assignment_args": " ".join(_assignment_args(cfg)),
        "node_count": node_count if node_count is not None else "",
        "track_count": track_count,
        "primary_metric": summary.get("primary_metric", ""),
        "density": summary.get("density", ""),
        "zero_rows": summary.get("zero_rows", ""),
        "nonzero_upper_edges": summary.get("nonzero_upper_edges", ""),
        "required_matrices_present": int(all((paths.get(key) is not None and paths[key].exists()) for key in required_keys)),
        "aal_b0_exists": int(paths.get("AAL_B0") is not None and paths["AAL_B0"].exists()),
        "assignments_exists": int(paths.get("ASSIGNMENTS") is not None and paths["ASSIGNMENTS"].exists()),
        "validation_gate": int(bool(cfg.validation_gate)),
        "analysis_gate_csv": str(cfg.analysis_gate_csv),
        "provenance_json": str(paths.get("SC_PROVENANCE", "")),
    }
    out_path = cfg.deriv_root / "qc" / "step7_spatial_contract_status.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row)
    with out_path.open("a", newline="", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            write_header = handle.tell() == 0
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _assignment_args(cfg: Step7Config) -> list[str]:
    """Resolve the tck2connectome assignment method from the shared contract.

    ``legacy`` preserves the older ``assignment_radial_search`` field exactly.
    The named variants are intentionally explicit so notebook and tmux runs can
    use the same assignment contract during canaries.
    """
    variant = str(getattr(cfg, "assignment_variant", "") or "").strip().lower()
    if variant in {"", "legacy"}:
        if cfg.assignment_radial_search is None:
            return []
        return ["-assignment_radial_search", str(cfg.assignment_radial_search)]
    if variant in {"none", "default", "end_voxel", "default_end_voxel"}:
        return []
    if variant.startswith("radial"):
        radius = variant.replace("radial", "", 1).strip("_-")
        if radius:
            return ["-assignment_radial_search", radius]
    if variant.startswith("forward"):
        distance = variant.replace("forward", "", 1).strip("_-")
        if distance:
            return ["-assignment_forward_search", distance]
    raise ValueError(f"Unsupported assignment_variant={cfg.assignment_variant!r}")


def _subject_paths(cfg: Step7Config, sid: str, create_dirs: bool = False) -> dict[str, Path]:
    fod_dir = cfg.deriv_root / "fod" / sid
    tracks_dir = cfg.deriv_root / "tracks" / sid
    dti_dir = cfg.deriv_root / "dti" / sid
    parc_dir = cfg.deriv_root / "parc" / sid
    conn_dir = cfg.deriv_root / "connectomes"
    if create_dirs:
        for directory in (fod_dir, tracks_dir, dti_dir, parc_dir, conn_dir):
            directory.mkdir(parents=True, exist_ok=True)

    return {
        "sid": Path(sid),
        "FODDIR": fod_dir,
        "TRKDIR": tracks_dir,
        "DTIDIR": dti_dir,
        "PARCDIR": parc_dir,
        "CONNDIR": conn_dir,
        "FOD_INPUT_QC_FAILED": fod_dir / "input_qc_failed.txt",
        "DWI": cfg.bias_root / f"{sid}_unbiased.mif",
        "DENOISED_DWI": cfg.deriv_root / "mif_denoised" / f"{sid}_den.mif",
        "GIBBS_DWI": cfg.deriv_root / "mif_unringed" / f"{sid}_den_unr.mif",
        "EDDY_DWI": cfg.deriv_root / "eddy" / f"{sid}_preproc.mif",
        "EDDY_DWI_ALT": cfg.deriv_root / "eddy" / f"{sid}_eddy.mif",
        "RAW_DWI_MIF": cfg.deriv_root / "mif_dwi" / f"{sid}.mif",
        "RAW_BVAL": cfg.deriv_root / "mif_dwi" / f"{sid}.bval",
        "RAW_BVEC": cfg.deriv_root / "mif_dwi" / f"{sid}.bvec",
        "DWI_READY": fod_dir / "dwi_step7_ready.mif",
        "B0MEAN": cfg.bbr_root / f"{sid}_b0mean_ras.nii.gz",
        "T1_RAS": cfg.bbr_root / f"{sid}_t1_ras.nii.gz",
        "T1B": cfg.bbr_root / f"{sid}_t1brain_ras.nii.gz",
        "T12B0": cfg.bbr_root / f"{sid}_t12b0_bbr.mat",
        "MASK_B0": cfg.bbr_root / f"{sid}_mask_on_b0_bbr.nii.gz",
        "FIVE_T_T1": fod_dir / "5tt_t1.mif",
        "MT_XFM": fod_dir / "t12b0_bbr_mrtrix.txt",
        "FIVE_B0": fod_dir / "5tt_b0.mif",
        "FIVE_FIX": fod_dir / "5tt_b0_fixed.mif",
        "MASK_5TT_BRAIN": fod_dir / "mask_5tt_brain.mif",
        "DWI_MASK": fod_dir / "mask_dwi.mif",
        "FOD_MASK": fod_dir / "mask_fod.mif",
        "GMWMI": fod_dir / "gmwmi.mif",
        "WM_TXT": fod_dir / "wm.txt",
        "WM_TXT_CSD": fod_dir / "wm_csd.txt",
        "GM_TXT": fod_dir / "gm.txt",
        "CSF_TXT": fod_dir / "csf.txt",
        "VOXELS": fod_dir / "response_voxels.mif",
        "WMFOD": fod_dir / "wmfod.mif",
        "GMFOD": fod_dir / "gmfod.mif",
        "CSFFOD": fod_dir / "csffod.mif",
        "WMFOD_NORM": fod_dir / "wmfod_norm.mif",
        "GMFOD_NORM": fod_dir / "gmfod_norm.mif",
        "CSFFOD_NORM": fod_dir / "csffod_norm.mif",
        "WMFOD_FINAL": fod_dir / "wmfod_final.mif",
        "DECONV_MODE_TXT": fod_dir / "deconv_mode.txt",
        "TRACKS_FINAL": tracks_dir / f"tracks_final_{cfg.select_streamlines // 1000}k.tck",
        "TRACK_CHUNK_DIR": tracks_dir / "_chunks",
        "TRACK_METHOD_TXT": tracks_dir / "track_method.txt",
        "WEIGHTS": tracks_dir / "sift_weights.txt",
        "MU_TXT": tracks_dir / "mu.txt",
        "ASSIGNMENTS": tracks_dir / "assignments_aal.csv",
        "DT_TEN": dti_dir / "dt.mif",
        "DTI_METHOD_TXT": dti_dir / "fit_method.txt",
        "FA_MIF": dti_dir / "fa.mif",
        "MD_MIF": dti_dir / "md.mif",
        "AD_MIF": dti_dir / "ad.mif",
        "RD_MIF": dti_dir / "rd.mif",
        "FA_TSF": tracks_dir / "fa_mean.tsf",
        "MD_TSF": tracks_dir / "md_mean.tsf",
        "AD_TSF": tracks_dir / "ad_mean.tsf",
        "RD_TSF": tracks_dir / "rd_mean.tsf",
        "AAL_T1": parc_dir / "AAL_t1.nii.gz",
        "MNI2T1": parc_dir / "mni2t1.mat",
        "MNI2B0": parc_dir / "mni2b0.mat",
        "AAL_B0": parc_dir / "AAL_b0.nii.gz",
        "SC_COUNT": conn_dir / f"SC_AAL_{sid}_count.csv",
        "SC_FD_SUM": conn_dir / f"SC_AAL_{sid}_fd_sum.csv",
        "SC_LEN_MEAN": conn_dir / f"SC_AAL_{sid}_len_mean.csv",
        "SC_INVLEN_MEAN": conn_dir / f"SC_AAL_{sid}_invlen_mean.csv",
        "SC_FA_MEAN": conn_dir / f"SC_AAL_{sid}_fa_mean.csv",
        "SC_MD_MEAN": conn_dir / f"SC_AAL_{sid}_md_mean.csv",
        "SC_AD_MEAN": conn_dir / f"SC_AAL_{sid}_ad_mean.csv",
        "SC_RD_MEAN": conn_dir / f"SC_AAL_{sid}_rd_mean.csv",
        "SC_COUNT_INVNODEVOL": conn_dir / f"SC_AAL_{sid}_count_invnodevol.csv",
        "SC_ALL": conn_dir / f"SC_AAL_{sid}_ALL.csv",
        "SC_PROVENANCE": conn_dir / f"SC_AAL_{sid}_generation_provenance.json",
    }


def _bbr_ready(paths: dict[str, Path]) -> bool:
    # Align with Step 6d "complete key sets": the BBR bridge is considered ready
    # with the core transform/image set even if mask_on_b0_bbr is missing.
    need = ["DWI", "B0MEAN", "T1_RAS", "T1B", "T12B0"]
    return all(paths[name].exists() for name in need)


def _discover_subjects(
    cfg: Step7Config,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    if include:
        sids = list(include)
    else:
        sids = sorted(
            p.name.replace("_unbiased.mif", "")
            for p in _visible_glob(cfg.bias_root, "*_unbiased.mif")
        )
    if exclude:
        excluded = set(exclude)
        sids = [sid for sid in sids if sid not in excluded]
    return _dedupe_preserve_order(sids)


def _discover_group_status_subjects(
    cfg: Step7Config,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    scope: str = "step7",
) -> list[str]:
    if include:
        sids = list(include)
    elif scope == "step7":
        sids = _discover_subjects(cfg, include=None, exclude=None)
    elif scope in {"full", "raw"}:
        sids = sorted(
            p.stem
            for p in _visible_glob(cfg.deriv_root / "mif_dwi", "*.mif")
        )
    else:
        raise ValueError(f"Unknown group status scope: {scope}")
    if exclude:
        excluded = set(exclude)
        sids = [sid for sid in sids if sid not in excluded]
    return _dedupe_preserve_order(sids)


def _select_deconv_mode(cfg: Step7Config) -> str:
    if cfg.deconv_mode in {"ss3t", "single_shell"}:
        return cfg.deconv_mode
    ss3t = shutil.which("ss3t_csd_beta1", path=_base_env(cfg)["PATH"])
    return "ss3t" if ss3t else "single_shell"


def _gradient_shell_bvalues(cfg: Step7Config, img: Path) -> list[float]:
    if not img.exists():
        return []
    proc = sp.run(
        [_tool(cfg, "mrinfo"), "-shell_bvalues", str(img)],
        env=_base_env(cfg),
        text=True,
        stdout=sp.PIPE,
        stderr=sp.DEVNULL,
    )
    if proc.returncode != 0:
        return []
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return []
    values: list[float] = []
    for token in lines[-1].replace(",", " ").split():
        try:
            values.append(float(token))
        except Exception:
            pass
    return values


def _gradient_summary_from_rows(rows: list[tuple[float, float, float, float]]) -> dict[str, object]:
    shells = sorted({round(float(row[3]) / 50.0) * 50 for row in rows if abs(float(row[3])) >= 50})
    b0_count = 0
    dw_count = 0
    usable_dw_count = 0
    for x, y, z, bvalue in rows:
        bvalue = float(bvalue)
        norm = math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z))
        if abs(bvalue) < 50:
            b0_count += 1
        else:
            dw_count += 1
            if norm > 0.05:
                usable_dw_count += 1
    return {
        "nvol": len(rows),
        "b0_count": b0_count,
        "dw_count": dw_count,
        "usable_dw_count": usable_dw_count,
        "shells": shells,
    }


def _mrinfo_gradient_rows(cfg: Step7Config, img: Path) -> list[tuple[float, float, float, float]]:
    if not img.exists():
        return []
    proc = sp.run(
        [_tool(cfg, "mrinfo"), str(img), "-dwgrad"],
        env=_base_env(cfg),
        text=True,
        stdout=sp.PIPE,
        stderr=sp.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return []
    rows: list[tuple[float, float, float, float]] = []
    for line in proc.stdout.splitlines():
        values: list[float] = []
        for token in line.replace(",", " ").split():
            try:
                values.append(float(token))
            except Exception:
                pass
        if len(values) >= 4:
            rows.append((values[0], values[1], values[2], values[3]))
    return rows


def _fsl_gradient_rows(bval: Path, bvec: Path) -> list[tuple[float, float, float, float]]:
    if not (bval.exists() and bvec.exists()):
        return []
    try:
        bvals = [float(token) for token in bval.read_text(encoding="utf-8", errors="ignore").split()]
        bvec_lines = [
            [float(token) for token in line.split()]
            for line in bvec.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
    except Exception:
        return []
    if len(bvec_lines) != 3 or any(len(line) != len(bvals) for line in bvec_lines):
        return []
    return [
        (bvec_lines[0][idx], bvec_lines[1][idx], bvec_lines[2][idx], bvals[idx])
        for idx in range(len(bvals))
    ]


def _format_gradient_summary(summary: dict[str, object]) -> str:
    shells = summary.get("shells", [])
    shell_label = _format_shell_bvalues([float(value) for value in shells]) if isinstance(shells, list) else "none"
    return (
        f"nvol={summary.get('nvol', 0)}, "
        f"b0={summary.get('b0_count', 0)}, "
        f"dw={summary.get('dw_count', 0)}, "
        f"usable_dw={summary.get('usable_dw_count', 0)}, "
        f"shells={shell_label}"
    )


def _fod_gradient_problem(
    cfg: Step7Config,
    *,
    summary: dict[str, object],
    assigned_shells: list[float] | None = None,
) -> str:
    nvol = int(summary.get("nvol", 0) or 0)
    b0_count = int(summary.get("b0_count", 0) or 0)
    dw_count = int(summary.get("dw_count", 0) or 0)
    usable_dw_count = int(summary.get("usable_dw_count", 0) or 0)
    min_dw = max(1, int(cfg.fod_min_dw_directions))
    if nvol < 2:
        return f"too few volumes for FOD/CSD ({_format_gradient_summary(summary)})"
    if b0_count < 1:
        return f"missing b=0 volume required for response estimation ({_format_gradient_summary(summary)})"
    if dw_count < 1:
        return f"missing diffusion-weighted volume ({_format_gradient_summary(summary)})"
    if usable_dw_count < min_dw:
        return (
            f"insufficient non-zero diffusion directions for FOD/CSD "
            f"(minimum={min_dw}; {_format_gradient_summary(summary)})"
        )
    if assigned_shells is not None:
        has_b0_shell = any(abs(value) < 50 for value in assigned_shells)
        has_dw_shell = any(abs(value) >= 50 for value in assigned_shells)
        if not has_b0_shell or not has_dw_shell:
            return (
                "MRtrix could not assign both b=0 and diffusion shells "
                f"(assigned={_format_shell_bvalues(assigned_shells)}; {_format_gradient_summary(summary)})"
            )
    return ""


def _fod_input_problem(cfg: Step7Config, img: Path) -> str:
    summary = _gradient_summary_from_rows(_mrinfo_gradient_rows(cfg, img))
    return _fod_gradient_problem(cfg, summary=summary, assigned_shells=_gradient_shell_bvalues(cfg, img))


def _format_shell_bvalues(shells: list[float]) -> str:
    if not shells:
        return "none"
    labels: list[str] = []
    for value in shells:
        if float(value).is_integer():
            labels.append(str(int(value)))
        else:
            labels.append(f"{value:g}")
    return ",".join(labels)


def _has_valid_gradients(cfg: Step7Config, img: Path) -> bool:
    shells = _gradient_shell_bvalues(cfg, img)
    has_b0 = any(abs(value) < 50 for value in shells)
    has_dw = any(abs(value) >= 50 for value in shells)
    return has_b0 and has_dw


def _mrinfo_size(cfg: Step7Config, img: Path) -> list[int]:
    out = sp.run(
        [_tool(cfg, "mrinfo"), str(img), "-size"],
        env=_base_env(cfg),
        text=True,
        stdout=sp.PIPE,
        stderr=sp.DEVNULL,
        check=False,
    ).stdout.strip()
    values: list[int] = []
    for token in out.split():
        try:
            values.append(int(token))
        except Exception:
            pass
    return values


def _grad_count_from_sidecars(bval: Path, bvec: Path) -> int:
    if not (bval.exists() and bvec.exists()):
        return -1
    try:
        bval_n = len(bval.read_text(encoding="utf-8", errors="ignore").split())
        bvec_lines = bvec.read_text(encoding="utf-8", errors="ignore").splitlines()
        bvec_n = len(bvec_lines[0].split()) if bvec_lines else -1
        return bval_n if bval_n == bvec_n else -1
    except Exception:
        return -1


def _prepare_dwi_for_fod(cfg: Step7Config, sid: str, paths: dict[str, Path]) -> tuple[Path, str]:
    dwi = paths["DWI"]
    native_problem = _fod_input_problem(cfg, dwi)
    if not native_problem:
        return dwi, "native"

    if not cfg.fod_repair_gradients:
        raise FodInputQualityError(f"FOD input not suitable and gradient repair is disabled: {native_problem}")

    raw_bval = paths["RAW_BVAL"]
    raw_bvec = paths["RAW_BVEC"]
    raw_count = _grad_count_from_sidecars(raw_bval, raw_bvec)
    dwi_size = _mrinfo_size(cfg, dwi)
    nvol = dwi_size[3] if len(dwi_size) >= 4 else -1
    if raw_count <= 0 or raw_count != nvol:
        raise FodInputQualityError(
            f"FOD input not suitable ({native_problem}); raw sidecars do not match "
            f"DWI volume count ({raw_count} vs {nvol})"
        )
    raw_summary = _gradient_summary_from_rows(_fsl_gradient_rows(raw_bval, raw_bvec))
    raw_problem = _fod_gradient_problem(cfg, summary=raw_summary)
    if raw_problem:
        raise FodInputQualityError(
            f"FOD input not suitable ({native_problem}); raw sidecars are also insufficient: {raw_problem}"
        )

    repaired = paths["DWI_READY"]
    if not repaired.exists() or cfg.force or not _has_valid_gradients(cfg, repaired):
        ok, logp, _ = _run_logged(
            cfg,
            sid,
            "mrconvert_repair_grad",
            [
                _tool(cfg, "mrconvert"),
                str(dwi),
                str(repaired),
                "-fslgrad",
                str(raw_bvec),
                str(raw_bval),
                "-force",
                "-quiet",
            ],
            env=_base_env(cfg),
        )
        if not ok:
            raise RuntimeError(f"Gradient repair command failed; see {logp}")
        repaired_problem = _fod_input_problem(cfg, repaired)
        if repaired_problem:
            raise FodInputQualityError(
                f"FOD input not suitable after gradient repair; see {logp}: {repaired_problem}"
            )
    return repaired, "repaired_from_raw_bvec_bval"


def _build_fod_mask(cfg: Step7Config, sid: str, paths: dict[str, Path], dwi_img: Path) -> str:
    env = _base_env(cfg)

    def _refresh_binary_fod_mask(source_mask: Path, *, tag: str) -> None:
        ok, logp, _ = _run_logged(
            cfg,
            sid,
            tag,
            [
                _tool(cfg, "mrcalc"),
                str(source_mask),
                "0",
                "-gt",
                str(paths["FOD_MASK"]),
                "-force",
            ],
            env=env,
        )
        if not ok:
            raise RuntimeError(f"mask binarisation failed; see {logp}")

    def _refresh_fod_mask(source_mask: Path, *, tag: str) -> None:
        ok, logp, _ = _run_logged(
            cfg,
            sid,
            tag,
            [
                _tool(cfg, "mrcalc"),
                str(source_mask),
                "0",
                "-gt",
                str(paths["MASK_5TT_BRAIN"]),
                "0",
                "-gt",
                "-mult",
                str(paths["FOD_MASK"]),
                "-force",
            ],
            env=env,
        )
        if not ok:
            raise RuntimeError(f"mask intersection failed; see {logp}")
        ok, logp, _ = _run_logged(
            cfg,
            sid,
            "maskfilter_dilate",
            [
                _tool(cfg, "maskfilter"),
                str(paths["FOD_MASK"]),
                "dilate",
                str(paths["FOD_MASK"]),
                "-npass",
                "1",
                "-force",
            ],
            env=env,
        )
        if not ok:
            raise RuntimeError(f"maskfilter failed; see {logp}")

    def _fod_mask_counts() -> tuple[int, int, int]:
        fod_mask_nz = _mr_nonzero_count(cfg, paths["FOD_MASK"]) if paths["FOD_MASK"].exists() else 0
        fod_5tt_overlap = (
            _mr_binary_overlap_count(
                cfg,
                paths["FOD_MASK"],
                paths["MASK_5TT_BRAIN"],
                context=f"fod_mask_build sid={sid} fod_5tt",
            )
            or 0
        )
        gmwmi_fod_overlap = (
            _mr_binary_overlap_count(
                cfg,
                paths["GMWMI"],
                paths["FOD_MASK"],
                context=f"fod_mask_build sid={sid} gmwmi_fod",
            )
            or 0
        )
        return fod_mask_nz, fod_5tt_overlap, gmwmi_fod_overlap

    def _mask_has_act_seed_overlap() -> bool:
        fod_mask_nz, fod_5tt_overlap, gmwmi_fod_overlap = _fod_mask_counts()
        return fod_mask_nz > 0 and fod_5tt_overlap > 0 and gmwmi_fod_overlap > 0

    def _try_bbr_mask_fallback(reason: str) -> str | None:
        if not cfg.fod_bbr_mask_fallback or not paths["MASK_B0"].exists():
            return None
        bbr_nz = _mr_nonzero_count(cfg, paths["MASK_B0"])
        if bbr_nz <= 0:
            return None
        mask_5tt_nz = _mr_nonzero_count(cfg, paths["MASK_5TT_BRAIN"])
        min_by_frac = int(math.ceil(mask_5tt_nz * cfg.fod_bbr_mask_min_5tt_overlap_frac))
        required_overlap = max(int(cfg.fod_bbr_mask_min_5tt_overlap_voxels), min_by_frac)
        bbr_5tt_overlap = (
            _mr_binary_overlap_count(
                cfg,
                paths["MASK_B0"],
                paths["MASK_5TT_BRAIN"],
                context=f"fod_mask_build sid={sid} bbr_5tt",
            )
            or 0
        )
        bbr_gmwmi_overlap = (
            _mr_binary_overlap_count(
                cfg,
                paths["MASK_B0"],
                paths["GMWMI"],
                context=f"fod_mask_build sid={sid} bbr_gmwmi",
            )
            or 0
        )
        if bbr_5tt_overlap < required_overlap or bbr_gmwmi_overlap <= 0:
            raise Step7GeometryError(
                "invalid_fod_mask_geometry: DWI mask is ACT-incompatible and BBR mask fallback "
                f"failed quality gate; reason={reason}; bbr_5tt_overlap={bbr_5tt_overlap}; "
                f"bbr_gmwmi_overlap={bbr_gmwmi_overlap}; required_bbr_5tt_overlap={required_overlap}; "
                f"mask_5tt_nz={mask_5tt_nz}; bbr_mask_nz={bbr_nz}"
            )
        _refresh_fod_mask(paths["MASK_B0"], tag="bbr_mask_5tt_intersection")
        if not _mask_has_act_seed_overlap():
            fod_mask_nz, fod_5tt_overlap, gmwmi_fod_overlap = _fod_mask_counts()
            raise Step7GeometryError(
                "invalid_fod_mask_geometry: BBR mask fallback wrote a mask but it still has no "
                f"ACT seed overlap; reason={reason}; fod_mask_nz={fod_mask_nz}; "
                f"fod_5tt_overlap={fod_5tt_overlap}; gmwmi_fod_overlap={gmwmi_fod_overlap}"
            )
        return f"bbr_mask_intersection_after_{reason}"

    requested_mask_source = str(getattr(cfg, "fod_mask_source", "auto") or "auto").strip().lower()
    if requested_mask_source in {"bbr", "bbr_full", "mask_b0", "bbr_mask_full"}:
        if not paths["MASK_B0"].exists() or _mr_nonzero_count(cfg, paths["MASK_B0"]) <= 0:
            raise Step7GeometryError("invalid_fod_mask_geometry: requested bbr_full FOD mask but MASK_B0 is missing or empty")
        _unlink(paths["DWI_MASK"])
        _unlink(paths["FOD_MASK"])
        _refresh_binary_fod_mask(paths["MASK_B0"], tag="bbr_mask_full")
        if not _mask_has_act_seed_overlap():
            fod_mask_nz, fod_5tt_overlap, gmwmi_fod_overlap = _fod_mask_counts()
            raise Step7GeometryError(
                "invalid_fod_mask_geometry: requested bbr_full FOD mask but it has no ACT seed overlap; "
                f"fod_mask_nz={fod_mask_nz}; fod_5tt_overlap={fod_5tt_overlap}; "
                f"gmwmi_fod_overlap={gmwmi_fod_overlap}"
            )
        return "bbr_mask_full"
    if requested_mask_source in {"bbr_5tt", "bbr_5tt_intersection", "bbr_intersection"}:
        fallback_mode = _try_bbr_mask_fallback("requested_bbr_5tt_intersection")
        if fallback_mode:
            return fallback_mode
        raise Step7GeometryError("invalid_fod_mask_geometry: requested bbr_5tt_intersection but fallback could not be built")

    if paths["FOD_MASK"].exists() and not cfg.force:
        if _mask_has_act_seed_overlap():
            return "existing"
        _unlink(paths["FOD_MASK"])
        _unlink(paths["DWI_MASK"])
    _unlink(paths["DWI_MASK"])
    _unlink(paths["FOD_MASK"])
    ok, logp, _ = _run_logged(
        cfg,
        sid,
        "dwi2mask",
        [_tool(cfg, "dwi2mask"), str(dwi_img), str(paths["DWI_MASK"])],
        env=env,
    )
    if not ok:
        raise RuntimeError(f"dwi2mask failed; see {logp}")
    dwi_mask_mode = "native"
    if _mr_nonzero_count(cfg, paths["DWI_MASK"]) == 0:
        if paths["MASK_B0"].exists() and _mr_nonzero_count(cfg, paths["MASK_B0"]) > 0:
            _unlink(paths["DWI_MASK"])
            ok, logp, _ = _run_logged(
                cfg,
                sid,
                "bbr_mask_fallback",
                [
                    _tool(cfg, "mrcalc"),
                    str(paths["MASK_B0"]),
                    "0",
                    "-gt",
                    str(paths["DWI_MASK"]),
                    "-force",
                ],
                env=env,
            )
            if not ok:
                raise RuntimeError(f"BBR mask fallback failed; see {logp}")
            dwi_mask_mode = "bbr_mask"
        else:
            dwi_mask_mode = "empty"

    _refresh_fod_mask(paths["DWI_MASK"], tag="mask_intersection")
    if not _mask_has_act_seed_overlap():
        reason = "empty_dwi_mask" if dwi_mask_mode in {"empty", "bbr_mask"} else "dwi_mask_act_mismatch"
        fallback_mode = _try_bbr_mask_fallback(reason)
        if fallback_mode:
            return fallback_mode
        fod_mask_nz, fod_5tt_overlap, gmwmi_fod_overlap = _fod_mask_counts()
        dwi_mask_nz = _mr_nonzero_count(cfg, paths["DWI_MASK"]) if paths["DWI_MASK"].exists() else 0
        raise Step7GeometryError(
            "invalid_fod_mask_geometry: DWI mask does not overlap ACT anatomy/seed geometry; "
            f"dwi_mask_mode={dwi_mask_mode}; dwi_mask_nz={dwi_mask_nz}; "
            f"fod_mask_nz={fod_mask_nz}; fod_5tt_overlap={fod_5tt_overlap}; "
            f"gmwmi_fod_overlap={gmwmi_fod_overlap}"
        )
    if dwi_mask_mode == "bbr_mask":
        return "bbr_mask_intersection"
    return "intersection"


def _make_binary_mask_from_brain(cfg: Step7Config, brain_img: Path, out_mask: Path, env: dict[str, str]) -> None:
    sp.run(
        [_tool(cfg, "mrcalc"), str(brain_img), "0", "-gt", str(out_mask), "-force"],
        env=env,
        check=True,
        stdout=sp.DEVNULL,
        stderr=sp.DEVNULL,
    )
    if _mr_nonzero_count(cfg, out_mask) == 0:
        raise RuntimeError("Derived T1 brain mask is empty")


def _run_5ttgen_fsl(
    cfg: Step7Config,
    sid: str,
    *,
    input_img: Path,
    output_img: Path,
    env: dict[str, str],
    premasked: bool = False,
    mask_img: Path | None = None,
    log_tag: str = "5ttgen",
) -> tuple[bool, Path]:
    cmd = [
        _tool(cfg, "5ttgen"),
        "fsl",
        str(input_img),
        str(output_img),
        "-nocrop",
        "-quiet",
    ]
    if premasked:
        cmd.append("-premasked")
    if mask_img is not None:
        cmd += ["-mask", str(mask_img)]
    with tempfile.TemporaryDirectory(prefix=f"step7_{log_tag}_{sid}_", dir="/tmp") as td:
        ok, logp, _ = _run_logged(
            cfg,
            sid,
            log_tag,
            cmd,
            env=env,
            timeout_sec=cfg.prep_5tt_timeout_sec,
            cwd=Path(td),
        )
    return ok, logp


def _subject_root_for_native_t1(sid: str) -> str:
    if "_I" in sid:
        return sid.split("_I", 1)[0]
    parts = sid.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else sid


def _visible_native_t1_matches(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(
            path
            for path in sorted(root.glob(pattern))
            if path.is_file() and not path.name.startswith(".") and not path.name.startswith("._")
        )
        if matches:
            return matches
    return matches


def _find_native_t1_inputs(cfg: Step7Config, sid: str) -> tuple[Path, Path | None, str]:
    t1_anat = cfg.deriv_root / "t1_anat"
    subj = _subject_root_for_native_t1(sid)
    native_matches = _visible_native_t1_matches(
        t1_anat,
        (
            f"corrected_T1_{sid}_*.nii.gz",
            f"corrected_T1_{subj}_*.nii.gz",
            f"T1_ss_{sid}_*.nii.gz",
            f"T1_ss_{subj}_*.nii.gz",
            f"t1_from_dicom_{sid}_*.nii.gz",
            f"t1_from_dicom_{subj}_*.nii.gz",
        ),
    )
    if not native_matches:
        raise FileNotFoundError(f"No native T1 found for {sid} under {t1_anat}")
    brain_matches = _visible_native_t1_matches(
        t1_anat,
        (
            f"T1_ss_{sid}_*.nii.gz",
            f"T1_ss_{subj}_*.nii.gz",
        ),
    )
    native_t1 = native_matches[0]
    native_brain = brain_matches[0] if brain_matches else None
    source = native_t1.name
    if native_brain is not None:
        source += f"; native_t1_brain={native_brain.name}"
    return native_t1, native_brain, source


def _build_fast_fallback_5tt(
    cfg: Step7Config,
    sid: str,
    paths: dict[str, Path],
    *,
    native_t1: Path | None = None,
    native_t1_brain: Path | None = None,
) -> None:
    env = _base_env(cfg)
    with tempfile.TemporaryDirectory(prefix=f"step7_fast5tt_{sid}_") as td:
        td_path = Path(td)
        fast_prefix = Path(td) / "fast"
        t1_mask = td_path / "t1_mask.mif"
        masked_t1_mif = td_path / "t1_masked.mif"
        masked_t1_nii = td_path / "t1_masked.nii.gz"
        base_t1 = native_t1 or paths["T1_RAS"]
        brain_t1 = native_t1_brain or paths["T1B"]
        _make_binary_mask_from_brain(cfg, brain_t1, t1_mask, env)
        sp.run(
            [_tool(cfg, "mrcalc"), str(base_t1), str(t1_mask), "-mult", str(masked_t1_mif), "-force"],
            env=env,
            check=True,
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
        )
        sp.run(
            [_tool(cfg, "mrconvert"), str(masked_t1_mif), str(masked_t1_nii), "-force", "-quiet"],
            env=env,
            check=True,
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
        )
        ok, logp, _ = _run_logged(
            cfg,
            sid,
            "fast_fallback",
            [
                _tool(cfg, "fast"),
                "-n",
                "3",
                "-g",
                "-o",
                str(fast_prefix),
                str(masked_t1_nii),
            ],
            env=env,
        )
        if not ok:
            raise RuntimeError(f"FAST fallback failed; see {logp}")

        csf_nii = Path(f"{fast_prefix}_pve_0.nii.gz")
        gm_nii = Path(f"{fast_prefix}_pve_1.nii.gz")
        wm_nii = Path(f"{fast_prefix}_pve_2.nii.gz")
        if not (csf_nii.exists() and gm_nii.exists() and wm_nii.exists()):
            raise RuntimeError("FAST fallback did not produce expected PVE outputs")

        cgm = Path(td) / "cgm.mif"
        sgm = Path(td) / "sgm.mif"
        wm = Path(td) / "wm.mif"
        csf = Path(td) / "csf.mif"
        path = Path(td) / "path.mif"
        for src, dst in ((gm_nii, cgm), (wm_nii, wm), (csf_nii, csf)):
            sp.run(
                [_tool(cfg, "mrconvert"), str(src), str(dst), "-force", "-quiet"],
                env=env,
                check=True,
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
            )
            sp.run(
                [_tool(cfg, "mrcalc"), str(dst), str(t1_mask), "-mult", str(dst), "-force"],
                env=env,
                check=True,
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
            )
        for dst in (sgm, path):
            sp.run(
                [_tool(cfg, "mrcalc"), str(cgm), "0", "-mul", str(dst), "-force"],
                env=env,
                check=True,
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
            )
        if sum(_mr_nonzero_count(cfg, img) for img in (cgm, wm, csf)) == 0:
            raise RuntimeError("FAST fallback produced empty tissue probability maps")
        sp.run(
            [_tool(cfg, "mrcat"), str(cgm), str(sgm), str(wm), str(csf), str(path), "-axis", "3", str(paths["FIVE_T_T1"]), "-force"],
            env=env,
            check=True,
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
        )


def _run_ss3t_with_staging(
    cfg: Step7Config,
    sid: str,
    dwi_img: Path,
    paths: dict[str, Path],
) -> tuple[bool, Path]:
    env = _base_env(cfg)
    if not cfg.fod_stage_ss3t_locally:
        with tempfile.TemporaryDirectory(prefix=f"step7_ss3t_{sid}_", dir="/tmp") as td:
            return _run_logged(
                cfg,
                sid,
                "ss3t_csd_beta1",
                [
                    shutil.which("ss3t_csd_beta1", path=env["PATH"]) or "ss3t_csd_beta1",
                    str(dwi_img),
                    str(paths["WM_TXT"]),
                    str(paths["WMFOD"]),
                    str(paths["GM_TXT"]),
                    str(paths["GMFOD"]),
                    str(paths["CSF_TXT"]),
                    str(paths["CSFFOD"]),
                    "-mask",
                    str(paths["FOD_MASK"]),
                    "-nthreads",
                    "1",
                ],
                env=env,
                timeout_sec=cfg.fod_ss3t_timeout_sec,
                cwd=Path(td),
            )[:2]

    with tempfile.TemporaryDirectory(prefix=f"step7_ss3t_{sid}_", dir="/tmp") as td:
        stage = Path(td)
        staged_dwi = stage / "dwi.mif"
        staged_mask = stage / "mask.mif"
        staged_wm = stage / "wm.txt"
        staged_gm = stage / "gm.txt"
        staged_csf = stage / "csf.txt"
        staged_wmfod = stage / "wmfod.mif"
        staged_gmfod = stage / "gmfod.mif"
        staged_csffod = stage / "csffod.mif"
        for src, dst in (
            (dwi_img, staged_dwi),
            (paths["FOD_MASK"], staged_mask),
            (paths["WM_TXT"], staged_wm),
            (paths["GM_TXT"], staged_gm),
            (paths["CSF_TXT"], staged_csf),
        ):
            shutil.copy2(src, dst)
        ok, logp, _ = _run_logged(
            cfg,
            sid,
            "ss3t_csd_beta1",
            [
                shutil.which("ss3t_csd_beta1", path=env["PATH"]) or "ss3t_csd_beta1",
                str(staged_dwi),
                str(staged_wm),
                str(staged_wmfod),
                str(staged_gm),
                str(staged_gmfod),
                str(staged_csf),
                str(staged_csffod),
                "-mask",
                str(staged_mask),
                "-nthreads",
                "1",
            ],
            env=env,
            timeout_sec=cfg.fod_ss3t_timeout_sec,
            cwd=stage,
        )
        if ok:
            for src, dst in (
                (staged_wmfod, paths["WMFOD"]),
                (staged_gmfod, paths["GMFOD"]),
                (staged_csffod, paths["CSFFOD"]),
            ):
                shutil.copy2(src, dst)
        return ok, logp


def _run_dhollander_response(
    cfg: Step7Config,
    sid: str,
    dwi_img: Path,
    paths: dict[str, Path],
    *,
    erode: int,
    step_name: str = "dwi2response_dhollander",
) -> tuple[bool, Path, float]:
    cmd = [
        _tool(cfg, "dwi2response"),
        "dhollander",
        str(dwi_img),
        str(paths["WM_TXT"]),
        str(paths["GM_TXT"]),
        str(paths["CSF_TXT"]),
        "-mask",
        str(paths["FOD_MASK"]),
        "-voxels",
        str(paths["VOXELS"]),
        "-erode",
        str(erode),
        "-nthreads",
        "1",
    ]
    with tempfile.TemporaryDirectory(prefix=f"step7_{step_name}_{sid}_", dir="/tmp") as td:
        return _run_logged(cfg, sid, step_name, cmd, cwd=Path(td))


def _run_single_shell_response_fallback(
    cfg: Step7Config,
    sid: str,
    dwi_img: Path,
    paths: dict[str, Path],
) -> tuple[str | None, Path | None]:
    if cfg.fod_response_fallback != "auto_single_shell":
        return None, None

    algorithms = _single_shell_response_fallback_algorithms(cfg)
    if not algorithms:
        return None, None

    last_logp: Path | None = None
    for algo in algorithms:
        _unlink(paths["WM_TXT_CSD"])
        _unlink(paths["VOXELS"])
        with tempfile.TemporaryDirectory(prefix=f"step7_dwi2response_{algo}_{sid}_", dir="/tmp") as td:
            ok, logp, _ = _run_logged(
                cfg,
                sid,
                f"dwi2response_{algo}",
                [
                    _tool(cfg, "dwi2response"),
                    algo,
                    str(dwi_img),
                    str(paths["WM_TXT_CSD"]),
                    "-mask",
                    str(paths["FOD_MASK"]),
                    "-voxels",
                    str(paths["VOXELS"]),
                    "-nthreads",
                    "1",
                ],
                cwd=Path(td),
            )
        last_logp = logp
        if ok and _sanitize_response_txt_inplace(paths["WM_TXT_CSD"]):
            return algo, logp
    return None, last_logp


def _single_shell_response_fallback_algorithms(cfg: Step7Config) -> list[str]:
    algorithms: list[str] = []
    for token in cfg.fod_response_fallback_order:
        algo = str(token).strip().lower()
        if algo in {"tournier", "fa"} and algo not in algorithms:
            algorithms.append(algo)
    return algorithms


def _run_single_shell_csd(
    cfg: Step7Config,
    sid: str,
    dwi_img: Path,
    paths: dict[str, Path],
    *,
    step_name: str = "dwi2fod_csd",
) -> tuple[bool, Path]:
    ok, logp, _ = _run_logged(
        cfg,
        sid,
        step_name,
        [
            _tool(cfg, "dwi2fod"),
            "csd",
            str(dwi_img),
            str(paths["WM_TXT_CSD"]),
            str(paths["WMFOD"]),
            "-mask",
            str(paths["FOD_MASK"]),
            "-nthreads",
            "1",
        ],
    )
    return ok, logp


def _stage_prep(cfg: Step7Config, sid: str, progress_queue: Queue | None = None) -> dict[str, str]:
    started_at = time.time()
    paths = _subject_paths(cfg, sid, create_dirs=True)
    try:
        _emit_stage_progress(progress_queue, sid, 2, "prep: checking inputs")
        if not _bbr_ready(paths):
            raise RuntimeError("BBR-ready inputs missing")

        prep_mode = "existing"
        if not (
            paths["FIVE_FIX"].exists()
            and paths["MASK_5TT_BRAIN"].exists()
            and paths["GMWMI"].exists()
            and not cfg.force
        ):
            _emit_stage_progress(progress_queue, sid, 12, "prep: building 5tt/gmwmi")
            env_5tt = _env_with_threads(cfg, cfg.omp_5tt_threads)
            prep_mode = "fsl"
            native_t1, native_t1_brain, native_t1_source = _find_native_t1_inputs(cfg, sid)
            five_tt_input = native_t1_brain or native_t1
            five_tt_premasked = native_t1_brain is not None
            for key in ("FIVE_T_T1", "MT_XFM", "FIVE_B0", "FIVE_FIX", "MASK_5TT_BRAIN", "GMWMI"):
                _unlink(paths[key])

            ok, logp = _run_5ttgen_fsl(
                cfg,
                sid,
                input_img=five_tt_input,
                output_img=paths["FIVE_T_T1"],
                env=env_5tt,
                premasked=five_tt_premasked,
                log_tag="5ttgen_native_t1",
            )
            if not ok:
                if cfg.prep_5tt_mask_retry and native_t1_brain is not None:
                    with tempfile.TemporaryDirectory(prefix=f"step7_t1mask_{sid}_") as td:
                        t1_mask = Path(td) / "t1_mask.nii.gz"
                        _make_binary_mask_from_brain(cfg, native_t1_brain, t1_mask, env_5tt)
                        ok_mask, logp_mask = _run_5ttgen_fsl(
                            cfg,
                            sid,
                            input_img=native_t1,
                            output_img=paths["FIVE_T_T1"],
                            env=env_5tt,
                            premasked=False,
                            mask_img=t1_mask,
                            log_tag="5ttgen_native_t1_masked",
                        )
                    if ok_mask:
                        prep_mode = "fsl_native_t1_mask"
                    else:
                        logp = logp_mask
                if not ok and prep_mode == "fsl" and cfg.prep_5tt_fallback == "fast":
                    _write_summary_row(
                        cfg,
                        sid,
                        "prep_retry",
                        "fallback",
                        note=f"5ttgen failed; using FAST fallback (source log: {logp.name})",
                    )
                    _build_fast_fallback_5tt(
                        cfg,
                        sid,
                        paths,
                        native_t1=native_t1,
                        native_t1_brain=native_t1_brain,
                    )
                    prep_mode = "fast_fallback_native_t1"
                elif not ok and prep_mode == "fsl":
                    raise RuntimeError(f"5ttgen failed; see {logp}")

            _emit_stage_progress(progress_queue, sid, 72, "prep: transforming 5tt")
            ok, logp, _ = _run_logged(
                cfg,
                sid,
                "transformconvert",
                [
                    _tool(cfg, "transformconvert"),
                    str(paths["T12B0"]),
                    str(native_t1),
                    str(paths["B0MEAN"]),
                    "flirt_import",
                    str(paths["MT_XFM"]),
                    "-force",
                ],
            )
            if not ok:
                raise RuntimeError(f"transformconvert failed; see {logp}")

            ok, logp, _ = _run_logged(
                cfg,
                sid,
                "mrtransform_5tt",
                [
                    _tool(cfg, "mrtransform"),
                    str(paths["FIVE_T_T1"]),
                    str(paths["FIVE_B0"]),
                    "-linear",
                    str(paths["MT_XFM"]),
                    "-template",
                    str(paths["B0MEAN"]),
                    "-force",
                    "-quiet",
                ],
            )
            if not ok:
                raise RuntimeError(f"mrtransform failed; see {logp}")

            with tempfile.TemporaryDirectory() as td:
                env = _base_env(cfg)
                vc: list[Path] = []
                vn: list[Path] = []
                for i in range(5):
                    vol = Path(td) / f"v{i}.mif"
                    clamp = Path(td) / f"vc{i}.mif"
                    norm = Path(td) / f"vn{i}.mif"
                    sp.run(
                        [_tool(cfg, "mrconvert"), str(paths["FIVE_B0"]), str(vol), "-coord", "3", str(i)],
                        env=env,
                        check=True,
                        stdout=sp.DEVNULL,
                        stderr=sp.DEVNULL,
                    )
                    sp.run(
                        [_tool(cfg, "mrcalc"), str(vol), "0", "-max", "1", "-min", str(clamp)],
                        env=env,
                        check=True,
                        stdout=sp.DEVNULL,
                        stderr=sp.DEVNULL,
                    )
                    vc.append(clamp)
                    vn.append(norm)

                sum_img = Path(td) / "sum.mif"
                sp.run(
                    [
                        _tool(cfg, "mrcalc"),
                        str(vc[0]),
                        str(vc[1]),
                        "-add",
                        str(vc[2]),
                        "-add",
                        str(vc[3]),
                        "-add",
                        str(vc[4]),
                        "-add",
                        str(sum_img),
                    ],
                    env=env,
                    check=True,
                    stdout=sp.DEVNULL,
                    stderr=sp.DEVNULL,
                )
                for i in range(5):
                    sp.run(
                        [
                            _tool(cfg, "mrcalc"),
                            str(sum_img),
                            "0",
                            "-gt",
                            str(vc[i]),
                            str(sum_img),
                            "-divide",
                            "0",
                            "-if",
                            str(vn[i]),
                        ],
                        env=env,
                        check=True,
                        stdout=sp.DEVNULL,
                        stderr=sp.DEVNULL,
                    )
                sp.run(
                    [_tool(cfg, "mrcat")] + [str(x) for x in vn] + ["-axis", "3", str(paths["FIVE_FIX"])],
                    env=env,
                    check=True,
                    stdout=sp.DEVNULL,
                    stderr=sp.DEVNULL,
                )
                sum_fix = Path(td) / "sum_fix.mif"
                sp.run(
                    [_tool(cfg, "mrmath"), str(paths["FIVE_FIX"]), "sum", "-axis", "3", str(sum_fix)],
                    env=env,
                    check=True,
                    stdout=sp.DEVNULL,
                    stderr=sp.DEVNULL,
                )
                sp.run(
                    [_tool(cfg, "mrcalc"), str(sum_fix), "0.05", "-gt", str(paths["MASK_5TT_BRAIN"]), "-force"],
                    env=env,
                    check=True,
                    stdout=sp.DEVNULL,
                    stderr=sp.DEVNULL,
                )
                sp.run(
                    [
                        _tool(cfg, "maskfilter"),
                        str(paths["MASK_5TT_BRAIN"]),
                        "dilate",
                        str(paths["MASK_5TT_BRAIN"]),
                        "-npass",
                        "1",
                        "-force",
                    ],
                    env=env,
                    check=True,
                    stdout=sp.DEVNULL,
                    stderr=sp.DEVNULL,
                )

            ok, logp, _ = _run_logged(
                cfg,
                sid,
                "5tt2gmwmi",
                [_tool(cfg, "5tt2gmwmi"), str(paths["FIVE_FIX"]), str(paths["GMWMI"]), "-force"],
            )
            if not ok:
                raise RuntimeError(f"5tt2gmwmi failed; see {logp}")

        audit = _validate_prep_geometry(cfg, sid, paths)
        native_note = f"; native_t1={native_t1_source}" if "native_t1_source" in locals() else ""
        note = f"5tt_mode={prep_mode}{native_note}; {_format_tracks_geometry_audit(audit)}"
        _write_summary_row(cfg, sid, "prep", "ok", note=note)
        _emit_stage_progress(progress_queue, sid, 100, "prep: done")
        return _stamp_stage_result({"sid": sid, "ok": True, "stage": "prep", "error": "", "note": note}, started_at)
    except Exception as exc:
        msg = f"{exc}\n{traceback.format_exc()}"
        _write_summary_row(cfg, sid, "prep", "error", msg)
        _emit_stage_progress(progress_queue, sid, 100, "prep: failed")
        return _stamp_stage_result({"sid": sid, "ok": False, "stage": "prep", "error": msg}, started_at)


def _stage_fod(cfg: Step7Config, sid: str, progress_queue: Queue | None = None) -> dict[str, str]:
    started_at = time.time()
    paths = _subject_paths(cfg, sid, create_dirs=True)
    mode = paths["DECONV_MODE_TXT"].read_text().strip() if paths["DECONV_MODE_TXT"].exists() else _select_deconv_mode(cfg)
    dwi_mode = "native"
    fod_mask_mode = ""
    response_fallback_algo = ""
    try:
        _emit_stage_progress(progress_queue, sid, 2, "fod 1/7: checking inputs")
        need = ("DWI", "FIVE_FIX", "MASK_5TT_BRAIN", "GMWMI")
        missing = [name for name in need if not paths[name].exists()]
        if missing:
            raise RuntimeError(f"Stage-prep outputs missing: {', '.join(missing)}")

        if _fod_input_qc_failed(paths) and not cfg.force:
            msg = _read_fod_input_qc_issue(paths)
            _write_summary_row(cfg, sid, "fod", "skip_input_qc_failed", msg, mode=mode)
            _emit_stage_progress(progress_queue, sid, 100, "fod: input qc failed")
            return _stamp_stage_result(
                {
                    "sid": sid,
                    "ok": False,
                    "status": "needs_input_repair",
                    "stage": "fod",
                    "error": msg,
                    "mode": mode,
                },
                started_at,
            )

        if _fod_stage_ready(cfg, sid) and not cfg.force:
            _validate_fod_act_geometry(cfg, sid, paths, fod_mask_mode="existing")
            _write_summary_row(cfg, sid, "fod", "skip_exists", mode=mode)
            _emit_stage_progress(progress_queue, sid, 100, "fod: already exists")
            return _stamp_stage_result({"sid": sid, "ok": True, "stage": "fod", "error": "", "mode": mode}, started_at)
        if not cfg.force:
            latest_fod_ok = _latest_summary_rows(
                cfg,
                stage="fod",
                wanted_sids={sid},
                statuses={"ok", "skip_exists"},
            ).get(sid, {})
            legacy_state, legacy_mode, legacy_source = _classify_legacy_fod_completion(
                cfg,
                sid,
                latest_fod_ok_row=latest_fod_ok,
                allow_infer_without_summary=True,
            )
            if legacy_state == "already_valid":
                mode = legacy_mode or mode
            elif legacy_state == "eligible":
                mode = legacy_mode
                _write_legacy_fod_completion_marker(cfg, sid, mode, legacy_source)
                _write_summary_row(
                    cfg,
                    sid,
                    "fod",
                    "skip_exists",
                    note=f"legacy_restore={legacy_source}",
                    mode=mode,
                )
                _emit_stage_progress(progress_queue, sid, 100, f"fod: restored legacy output ({legacy_source})")
                return _stamp_stage_result(
                    {
                        "sid": sid,
                        "ok": True,
                        "stage": "fod",
                        "error": "",
                        "mode": mode,
                        "legacy_restore_source": legacy_source,
                    },
                    started_at,
                )
        if _fod_outputs_present(paths) and not cfg.force:
            _append_runtime_note(cfg.runtime_log, f"Step7 FOD rerun: stale or empty output detected for {sid}; rebuilding FOD")
        if cfg.force:
            _unlink(paths["DECONV_MODE_TXT"])

        _emit_stage_progress(progress_queue, sid, 10, "fod 2/7: preparing dwi")
        dwi_img, dwi_mode = _prepare_dwi_for_fod(cfg, sid, paths)
        _emit_stage_progress(progress_queue, sid, 22, f"fod 2/7: dwi ready ({dwi_mode})")
        _emit_stage_progress(progress_queue, sid, 26, "fod 3/7: building fod mask")
        fod_mask_mode = _build_fod_mask(cfg, sid, paths, dwi_img)
        _emit_stage_progress(progress_queue, sid, 36, f"fod 3/7: mask ready ({fod_mask_mode})")
        if fod_mask_mode.startswith("bbr_mask"):
            _write_summary_row(
                cfg,
                sid,
                "fod_retry",
                "fallback",
                note=f"dwi2mask empty; using BBR mask fallback ({fod_mask_mode})",
                mode=mode,
            )
        response_retry_note = ""
        response_fallback_note = ""

        for key in ("WM_TXT", "GM_TXT", "CSF_TXT", "WM_TXT_CSD", "VOXELS"):
            _unlink(paths[key])
        _emit_stage_progress(progress_queue, sid, 40, "fod 4/7: dhollander response")
        ok, logp, _ = _run_dhollander_response(
            cfg,
            sid,
            dwi_img,
            paths,
            erode=cfg.fod_dhollander_erode,
        )
        if not ok:
            if cfg.fod_dhollander_retry_erode is not None and _is_dhollander_sparse_mask_failure(logp):
                if (
                    paths["DWI_MASK"].exists()
                    and _mr_nonzero_count(cfg, paths["DWI_MASK"]) > 0
                    and fod_mask_mode != "dwi_mask_only"
                ):
                    shutil.copy2(paths["DWI_MASK"], paths["FOD_MASK"])
                    fod_mask_mode = "dwi_mask_only_retry"
                for key in ("WM_TXT", "GM_TXT", "CSF_TXT", "VOXELS"):
                    _unlink(paths[key])
                _emit_stage_progress(progress_queue, sid, 44, "fod 4/7: dhollander retry")
                ok, retry_logp, _ = _run_dhollander_response(
                    cfg,
                    sid,
                    dwi_img,
                    paths,
                    erode=cfg.fod_dhollander_retry_erode,
                    step_name="dwi2response_dhollander_retry",
                )
                if ok:
                    response_retry_note = f"dhollander_retry_erode={cfg.fod_dhollander_retry_erode}"
                    _write_summary_row(
                        cfg,
                        sid,
                        "fod_retry",
                        "fallback",
                        note=(
                            "dhollander sparse-mask retry succeeded "
                            f"(erode={cfg.fod_dhollander_retry_erode}; mask={fod_mask_mode}; "
                            f"source log: {retry_logp.name})"
                        ),
                        mode=mode,
                    )
                    logp = retry_logp
                else:
                    logp = retry_logp

            if not ok:
                if _mr_nonzero_count(cfg, paths["FOD_MASK"]) == 0:
                    raise RuntimeError("empty_dwi_mask: FOD mask became empty before response fallback")
                _emit_stage_progress(progress_queue, sid, 48, "fod 4/7: wm-response fallback")
                response_fallback_algo, fallback_logp = _run_single_shell_response_fallback(cfg, sid, dwi_img, paths)
                if response_fallback_algo:
                    mode = "single_shell"
                    response_fallback_note = f"response_fallback={response_fallback_algo}"
                    _write_summary_row(
                        cfg,
                        sid,
                        "fod_retry",
                        "fallback",
                        note=(
                            f"dhollander failed; using WM-only fallback ({response_fallback_algo}) "
                            f"(source log: {fallback_logp.name}; prior log: {logp.name})"
                        ),
                        mode=mode,
                    )
                elif cfg.fod_dhollander_retry_erode is not None and _is_dhollander_sparse_mask_failure(logp):
                    failed_log = fallback_logp or logp
                    raise RuntimeError(
                        f"dwi2response dhollander failed after sparse-mask retry and WM-only fallback; see {failed_log}"
                    )
                else:
                    failed_log = fallback_logp or logp
                    raise RuntimeError(f"dwi2response dhollander failed and WM-only fallback exhausted; see {failed_log}")

        if response_fallback_algo:
            _emit_stage_progress(progress_queue, sid, 52, f"fod 5/7: {response_fallback_algo} response ready")
        else:
            _emit_stage_progress(progress_queue, sid, 52, "fod 5/7: sanitizing response")
            for key in ("WM_TXT", "GM_TXT", "CSF_TXT"):
                if not _sanitize_response_txt_inplace(paths[key]):
                    raise RuntimeError(f"Failed to sanitize {key}")

        if mode == "ss3t":
            for key in ("WMFOD", "GMFOD", "CSFFOD", "WMFOD_NORM", "GMFOD_NORM", "CSFFOD_NORM", "WMFOD_FINAL"):
                _unlink(paths[key])
            _emit_stage_progress(progress_queue, sid, 62, "fod 6/7: ss3t deconvolution")
            ok, logp = _run_ss3t_with_staging(cfg, sid, dwi_img, paths)
            if not ok:
                _write_summary_row(
                    cfg,
                    sid,
                    "fod_retry",
                    "fallback",
                    note=f"ss3t_csd_beta1 failed; falling back to single_shell (source log: {logp.name})",
                    mode=mode,
                )
                _emit_stage_progress(progress_queue, sid, 68, "fod 6/7: ss3t fallback -> single_shell")
                mode = "single_shell"
            else:
                final_wmfod = paths["WMFOD"]
                if cfg.use_mtnormalise:
                    _emit_stage_progress(progress_queue, sid, 78, "fod 6/7: mtnormalise")
                    ok, logp, _ = _run_logged(
                        cfg,
                        sid,
                        "mtnormalise",
                        [
                            _tool(cfg, "mtnormalise"),
                            str(paths["WMFOD"]),
                            str(paths["WMFOD_NORM"]),
                            str(paths["GMFOD"]),
                            str(paths["GMFOD_NORM"]),
                            str(paths["CSFFOD"]),
                            str(paths["CSFFOD_NORM"]),
                            "-mask",
                            str(paths["FOD_MASK"]),
                            "-nthreads",
                            "1",
                        ],
                    )
                    if not ok:
                        raise RuntimeError(f"mtnormalise failed; see {logp}")
                    final_wmfod = paths["WMFOD_NORM"]
                _symlink_or_copy(final_wmfod, paths["WMFOD_FINAL"])

        if mode == "single_shell":
            for key in ("WMFOD", "WMFOD_FINAL"):
                _unlink(paths[key])
            if response_fallback_algo:
                _emit_stage_progress(progress_queue, sid, 72, f"fod 6/7: {response_fallback_algo} wm response")
                if not _has_numeric_rows(paths["WM_TXT_CSD"]):
                    raise RuntimeError(
                        f"WM-only response fallback did not produce a valid WM response: {paths['WM_TXT_CSD']}"
                    )
            else:
                _unlink(paths["WM_TXT_CSD"])
                _emit_stage_progress(progress_queue, sid, 72, "fod 6/7: single-shell response")
                if not _write_single_shell_response_txt(paths["WM_TXT"], paths["WM_TXT_CSD"]):
                    raise RuntimeError("Failed to derive single-shell WM response vector from wm.txt")
            _emit_stage_progress(progress_queue, sid, 80, "fod 6/7: dwi2fod csd")
            ok, logp = _run_single_shell_csd(cfg, sid, dwi_img, paths)
            if not ok:
                raise RuntimeError(f"dwi2fod csd failed; see {logp}")
            _symlink_or_copy(paths["WMFOD"], paths["WMFOD_FINAL"])

        _emit_stage_progress(progress_queue, sid, 90, "fod 7/7: validating output")
        fod_ok, fod_detail = _wmfod_sh_quality_details(
            cfg,
            paths["WMFOD_FINAL"],
            context=f"stage_fod sid={sid} final_validation",
        )
        if not fod_ok and mode == "single_shell" and response_fallback_algo:
            algorithms = _single_shell_response_fallback_algorithms(cfg)
            try_next = False
            for algo in algorithms:
                if algo == response_fallback_algo:
                    try_next = True
                    continue
                if not try_next:
                    continue
                _write_summary_row(
                    cfg,
                    sid,
                    "fod_retry",
                    "fallback",
                    note=(
                        f"WM-only fallback ({response_fallback_algo}) produced invalid WM FOD "
                        f"({fod_detail}); trying {algo}"
                    ),
                    mode=mode,
                )
                _emit_stage_progress(progress_queue, sid, 92, f"fod 7/7: retry response {algo}")
                _unlink(paths["WM_TXT_CSD"])
                _unlink(paths["VOXELS"])
                with tempfile.TemporaryDirectory(prefix=f"step7_dwi2response_{algo}_badfod_{sid}_", dir="/tmp") as td:
                    resp_ok, resp_logp, _ = _run_logged(
                        cfg,
                        sid,
                        f"dwi2response_{algo}_bad_fod_retry",
                        [
                            _tool(cfg, "dwi2response"),
                            algo,
                            str(dwi_img),
                            str(paths["WM_TXT_CSD"]),
                            "-mask",
                            str(paths["FOD_MASK"]),
                            "-voxels",
                            str(paths["VOXELS"]),
                            "-nthreads",
                            "1",
                        ],
                        cwd=Path(td),
                    )
                if not resp_ok or not _sanitize_response_txt_inplace(paths["WM_TXT_CSD"]):
                    continue
                response_fallback_algo = algo
                response_fallback_note = f"response_fallback={algo}; response_fallback_retry=bad_fod"
                for key in ("WMFOD", "WMFOD_FINAL"):
                    _unlink(paths[key])
                ok, logp = _run_single_shell_csd(
                    cfg,
                    sid,
                    dwi_img,
                    paths,
                    step_name=f"dwi2fod_csd_{algo}_bad_fod_retry",
                )
                if not ok:
                    _write_summary_row(
                        cfg,
                        sid,
                        "fod_retry",
                        "fallback",
                        note=f"WM-only fallback retry ({algo}) dwi2fod failed; see {logp.name}",
                        mode=mode,
                    )
                    continue
                _symlink_or_copy(paths["WMFOD"], paths["WMFOD_FINAL"])
                fod_ok, fod_detail = _wmfod_sh_quality_details(
                    cfg,
                    paths["WMFOD_FINAL"],
                    context=f"stage_fod sid={sid} fallback={algo} final_validation",
                )
                if fod_ok:
                    _write_summary_row(
                        cfg,
                        sid,
                        "fod_retry",
                        "fallback",
                        note=f"WM-only fallback retry succeeded with {algo}; {fod_detail}",
                        mode=mode,
                    )
                    break
        if not fod_ok:
            raise RuntimeError(f"WM FOD is empty or has no usable SH structure after deconvolution; {fod_detail}")
        geometry_audit = _validate_fod_act_geometry(cfg, sid, paths, fod_mask_mode=fod_mask_mode)
        _unlink(paths["FOD_INPUT_QC_FAILED"])
        paths["DECONV_MODE_TXT"].write_text(mode + "\n", encoding="ascii")
        note_parts = []
        if dwi_mode != "native":
            note_parts.append(f"dwi_mode={dwi_mode}")
        if fod_mask_mode != "intersection":
            note_parts.append(f"fod_mask_mode={fod_mask_mode}")
        if response_retry_note:
            note_parts.append(response_retry_note)
        if response_fallback_note:
            note_parts.append(response_fallback_note)
        if mode == "single_shell":
            if response_fallback_algo:
                note_parts.append(f"single_shell_wm_response={response_fallback_algo}")
            else:
                note_parts.append("single_shell_wm_response=last_row")
        note_parts.append(_format_tracks_geometry_audit(geometry_audit))
        note = "; ".join(note_parts)
        _write_summary_row(cfg, sid, "fod", "ok", note=note, mode=mode)
        _emit_stage_progress(progress_queue, sid, 100, "fod: done")
        return _stamp_stage_result(
            {
                "sid": sid,
                "ok": True,
                "stage": "fod",
                "error": "",
                "mode": mode,
                "dwi_mode": dwi_mode,
                "fod_mask_mode": fod_mask_mode,
                "response_fallback": response_fallback_algo,
            },
            started_at,
        )
    except FodInputQualityError as exc:
        msg = str(exc)
        _unlink(paths["DECONV_MODE_TXT"])
        _write_summary_row(cfg, sid, "fod", "needs_input_repair", msg, mode=mode)
        _write_input_repair_row(cfg, sid, "fod", msg, mode=mode)
        _write_fod_input_qc_failure(cfg, sid, "fod", msg, mode=mode)
        _append_runtime_note(
            cfg.runtime_log,
            f"Sample: {sid} | stage=fod | needs input repair | reason={msg}",
        )
        _emit_stage_progress(progress_queue, sid, 100, "fod: needs input repair")
        return _stamp_stage_result(
            {
                "sid": sid,
                "ok": False,
                "status": "needs_input_repair",
                "stage": "fod",
                "error": msg,
                "mode": mode,
                "dwi_mode": dwi_mode,
                "fod_mask_mode": fod_mask_mode,
                "response_fallback": response_fallback_algo,
            },
            started_at,
        )
    except Exception as exc:
        msg = f"{exc}\n{traceback.format_exc()}"
        _unlink(paths["DECONV_MODE_TXT"])
        _write_summary_row(cfg, sid, "fod", "error", msg, mode=mode)
        _emit_stage_progress(progress_queue, sid, 100, "fod: failed")
        return _stamp_stage_result(
            {
                "sid": sid,
                "ok": False,
                "stage": "fod",
                "error": msg,
                "mode": mode,
                "dwi_mode": dwi_mode,
                "fod_mask_mode": fod_mask_mode,
                "response_fallback": response_fallback_algo,
            },
            started_at,
        )


def _cutoff_label(cutoff: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "p", str(cutoff)).strip("p") or "cutoff"


def _tracks_wmfod_input(paths: dict[str, Path]) -> Path:
    """Return the FOD image whose header contract matches generated tracks."""
    if paths["WMFOD"].exists():
        return paths["WMFOD"]
    return paths["WMFOD_FINAL"]


def _tracks_act_input(paths: dict[str, Path]) -> Path:
    """Return the corrected ACT 5TT image used by Step 7 geometry QC."""
    if paths["FIVE_FIX"].exists():
        return paths["FIVE_FIX"]
    return paths["FIVE_B0"]


def _build_tckgen_cmd(
    cfg: Step7Config,
    paths: dict[str, Path],
    cutoff: str,
    out_tck: Path,
    *,
    select_count: int,
    max_seeds: int = 0,
) -> list[str]:
    wmfod = _tracks_wmfod_input(paths)
    act = _tracks_act_input(paths)
    cmd = [
        _tool(cfg, "tckgen"),
        str(wmfod),
        str(out_tck),
        "-act",
        str(act),
        "-backtrack",
        "-cutoff",
        str(cutoff),
        "-maxlength",
        str(cfg.track_maxlength),
        "-minlength",
        str(cfg.track_minlength),
        "-select",
        str(select_count),
        "-nthreads",
        str(cfg.tck_threads),
    ]
    if max_seeds > 0:
        cmd += ["-seeds", str(max_seeds)]
    if cfg.track_seed_mode == "dynamic":
        cmd += ["-seed_dynamic", str(wmfod)]
    else:
        cmd += ["-seed_gmwmi", str(paths["GMWMI"]), "-crop_at_gmwmi"]
    cmd += ["-force"]
    return cmd


def _track_method_label(cfg: Step7Config) -> str:
    if cfg.track_seed_mode == "dynamic":
        return "act_dynamic_seed"
    return "act_gmwmi"


def _zero_tracks_message(prefix: str, cutoff: str, logp: Path) -> str:
    outcome = _parse_tckgen_outcome(logp)
    return (
        f"{prefix} at cutoff {cutoff} ({outcome['reason']}; "
        f"seeds={outcome['seeds']}; streamlines={outcome['streamlines']}; "
        f"selected={outcome['selected']})"
    )


def _short_tracks_message(prefix: str, cutoff: str, requested: int, count: int, logp: Path) -> str:
    outcome = _parse_tckgen_outcome(logp)
    return (
        f"{prefix} at cutoff {cutoff}; requested={requested}; count={count} "
        f"({outcome['reason']}; seeds={outcome['seeds']}; "
        f"streamlines={outcome['streamlines']}; selected={outcome['selected']})"
    )


def _tracks_geometry_preflight(cfg: Step7Config, sid: str, paths: dict[str, Path]) -> None:
    wmfod = _tracks_wmfod_input(paths)
    fod_ok, fod_detail = _wmfod_sh_quality_details(
        cfg,
        wmfod,
        context=f"tracks_preflight sid={sid}",
    )
    if not fod_ok:
        raise RuntimeError(
            "invalid_tracks_fod: WM FOD has no usable spherical-harmonic structure; "
            "rerun stage='fod' with force=True for this subject; "
            f"{fod_detail}"
        )
    if cfg.track_seed_mode == "dynamic":
        audit = _tracks_geometry_audit(cfg, sid, paths, include_fod_mask=False)
        if (_audit_int(audit, "five_fix_nz") or 0) <= 0:
            raise Step7GeometryError(f"invalid_prep_geometry: 5TT image is empty; {_format_tracks_geometry_audit(audit)}")
        if (_audit_int(audit, "mask_5tt_nz") or 0) <= 0:
            raise Step7GeometryError(f"invalid_prep_geometry: 5TT brain mask is empty; {_format_tracks_geometry_audit(audit)}")
        return
    _validate_prep_geometry(cfg, sid, paths)
    try:
        _validate_fod_act_geometry(cfg, sid, paths, fod_mask_mode="tracks_preflight")
    except Step7GeometryError as exc:
        text = str(exc)
        if "zero overlap" in text:
            raise RuntimeError(
                "invalid_tracks_geometry: GMWMI seed image has zero overlap with the FOD mask; "
                "rerun stage='prep' and stage='fod' with force=True for this subject; "
                f"{text}"
            )
        raise


def _probe_tckgen_cutoff(
    cfg: Step7Config,
    sid: str,
    paths: dict[str, Path],
    cutoff: str,
    chunk_dir: Path,
    env: dict[str, str],
) -> None:
    probe_select = max(0, int(getattr(cfg, "track_probe_streamlines", 0) or 0))
    if probe_select <= 0:
        return
    probe_out = chunk_dir / f"probe_cutoff_{_cutoff_label(cutoff)}.tck"
    _unlink(probe_out)
    cmd = _build_tckgen_cmd(
        cfg,
        paths,
        cutoff,
        probe_out,
        select_count=probe_select,
        max_seeds=max(0, int(getattr(cfg, "track_probe_max_seeds", 0) or 0)),
    )
    ok, logp, _ = _run_logged(
        cfg,
        sid,
        f"tckgen_probe_{_cutoff_label(cutoff)}",
        cmd,
        env=env,
        timeout_sec=getattr(cfg, "track_probe_timeout_sec", None),
    )
    count = _tck_count(cfg, probe_out)
    _unlink(probe_out)
    if not ok and count >= probe_select:
        _append_runtime_note(
            cfg.runtime_log,
            (
                f"Step7 tracks tckgen warning {time.strftime('%Y-%m-%dT%H:%M:%S')} | "
                f"sid={sid} | cutoff={cutoff} | probe nonzero exit but "
                f"valid count={count}; accepted | log={logp}"
            ),
        )
    elif not ok:
        raise RuntimeError(
            _short_tracks_message("tckgen probe failed or was short", cutoff, probe_select, count, logp)
            + f"; see {logp}"
        )
    if count <= 0:
        raise RuntimeError(_zero_tracks_message("tckgen probe produced 0 tracks", cutoff, logp))
    if count < probe_select:
        raise RuntimeError(
            _short_tracks_message("tckgen probe was short", cutoff, probe_select, count, logp)
            + f"; see {logp}"
        )


def _chunk_manifest_text(cfg: Step7Config, cutoff: str, mode: str) -> str:
    return "\n".join(
        [
            f"cutoff={cutoff}",
            f"mode={mode}",
            f"seed_mode={cfg.track_seed_mode}",
            f"select_streamlines={int(cfg.select_streamlines)}",
            f"chunk_streamlines={int(cfg.chunk_streamlines)}",
            f"track_minlength={int(cfg.track_minlength)}",
            f"track_maxlength={int(cfg.track_maxlength)}",
            f"tck_threads={int(cfg.tck_threads)}",
            "",
        ]
    )


def _clear_track_chunk_dir(chunk_dir: Path) -> None:
    for stale in chunk_dir.glob("chunk_*.tck"):
        _unlink(stale)
    for stale in chunk_dir.glob("probe_*.tck"):
        _unlink(stale)
    for stale in chunk_dir.glob("*.tmp.*"):
        _unlink(stale)


def _track_chunk_specs(cfg: Step7Config, chunk_dir: Path) -> list[tuple[int, int, Path]]:
    specs: list[tuple[int, int, Path]] = []
    generated = 0
    chunk_index = 0
    while generated < cfg.select_streamlines:
        current = min(cfg.chunk_streamlines, cfg.select_streamlines - generated)
        specs.append((chunk_index, current, chunk_dir / f"chunk_{chunk_index:03d}_{current // 1000}k.tck"))
        generated += current
        chunk_index += 1
    return specs


def _prepare_track_chunk_dir(
    cfg: Step7Config,
    sid: str,
    chunk_dir: Path,
    cutoff: str,
    mode: str,
) -> None:
    expected = _chunk_manifest_text(cfg, cutoff, mode)
    manifest = chunk_dir / "chunk_manifest.txt"
    existing_chunks = list(chunk_dir.glob("chunk_*.tck"))
    resume = bool(getattr(cfg, "track_resume_chunks", True)) and not bool(cfg.force)
    if not resume:
        _clear_track_chunk_dir(chunk_dir)
        manifest.write_text(expected, encoding="ascii")
        return
    if manifest.exists():
        current = manifest.read_text(encoding="ascii", errors="ignore")
        if current != expected:
            _append_runtime_note(
                cfg.runtime_log,
                (
                    f"Step7 tracks chunk resume reset {time.strftime('%Y-%m-%dT%H:%M:%S')} | "
                    f"sid={sid} | cutoff={cutoff} | reason=manifest_mismatch"
                ),
            )
            _clear_track_chunk_dir(chunk_dir)
    elif existing_chunks and cutoff == str(cfg.track_cutoffs[0]):
        _append_runtime_note(
            cfg.runtime_log,
            (
                f"Step7 tracks chunk resume adopt {time.strftime('%Y-%m-%dT%H:%M:%S')} | "
                f"sid={sid} | cutoff={cutoff} | chunks={len(existing_chunks)}"
            ),
        )
    elif existing_chunks:
        _clear_track_chunk_dir(chunk_dir)
    manifest.write_text(expected, encoding="ascii")
    for stale in chunk_dir.glob("probe_*.tck"):
        _unlink(stale)


def _run_tckgen_chunk(
    cfg: Step7Config,
    sid: str,
    paths: dict[str, Path],
    cutoff: str,
    env: dict[str, str],
    chunk_index: int,
    current: int,
    out_chunk: Path,
) -> Path:
    _unlink(out_chunk)
    cmd = _build_tckgen_cmd(cfg, paths, cutoff, out_chunk, select_count=current)
    with _tckgen_slot(cfg):
        ok, logp, _ = _run_logged(
            cfg,
            sid,
            f"tckgen_{chunk_index:03d}",
            cmd,
            env=env,
            timeout_sec=getattr(cfg, "track_chunk_timeout_sec", None),
        )
    chunk_count = _tck_count(cfg, out_chunk)
    if not ok and chunk_count >= current:
        _append_runtime_note(
            cfg.runtime_log,
            (
                f"Step7 tracks tckgen warning {time.strftime('%Y-%m-%dT%H:%M:%S')} | "
                f"sid={sid} | cutoff={cutoff} | chunk={chunk_index} | "
                f"nonzero exit but valid chunk_count={chunk_count}; accepted | log={logp}"
            ),
        )
    elif not ok:
        raise RuntimeError(f"tckgen chunk failed at cutoff {cutoff}; see {logp}")
    if chunk_count <= 0:
        raise RuntimeError(_zero_tracks_message("tckgen produced 0 tracks", cutoff, logp))
    if chunk_count < current:
        raise RuntimeError(
            f"tckgen chunk was short at cutoff {cutoff}; "
            f"requested={current}; count={chunk_count}; see {logp}"
        )
    return out_chunk


def _chunked_tckgen(
    cfg: Step7Config,
    sid: str,
    paths: dict[str, Path],
    cutoff: str,
    mode: str,
    progress_queue: Queue | None = None,
) -> None:
    env = _base_env(cfg)
    chunk_dir = paths["TRACK_CHUNK_DIR"]
    chunk_dir.mkdir(parents=True, exist_ok=True)
    _prepare_track_chunk_dir(cfg, sid, chunk_dir, cutoff, mode)

    _probe_tckgen_cutoff(cfg, sid, paths, cutoff, chunk_dir, env)

    specs = _track_chunk_specs(cfg, chunk_dir)
    total_chunks = max(1, len(specs))
    complete: set[int] = set()
    pending_specs: list[tuple[int, int, Path]] = []
    for chunk_index, current, out_chunk in specs:
        chunk_count = _tck_count(cfg, out_chunk)
        if chunk_count >= current:
            complete.add(chunk_index)
            continue
        _unlink(out_chunk)
        pending_specs.append((chunk_index, current, out_chunk))

    if complete:
        generated = sum(current for chunk_index, current, _ in specs if chunk_index in complete)
        percent = min(92, 15 + int(round(75.0 * generated / max(1, cfg.select_streamlines))))
        _emit_stage_progress(
            progress_queue,
            sid,
            percent,
            f"tracks: cutoff {cutoff} resumed {len(complete)}/{total_chunks} chunks",
        )

    chunk_jobs = max(1, int(getattr(cfg, "track_chunk_jobs_per_subject", 1) or 1))
    chunk_jobs = min(chunk_jobs, len(pending_specs) or 1)
    if pending_specs and chunk_jobs > 1:
        _append_runtime_note(
            cfg.runtime_log,
            (
                f"Step7 tracks chunk parallel {time.strftime('%Y-%m-%dT%H:%M:%S')} | "
                f"sid={sid} | cutoff={cutoff} | pending_chunks={len(pending_specs)} | "
                f"chunk_jobs={chunk_jobs} | max_parallel_tckgen={_max_parallel_tckgen(cfg)}"
            ),
        )
        executor = ThreadPoolExecutor(max_workers=chunk_jobs)
        pending: dict[object, tuple[int, int, Path]] = {}
        spec_iter = iter(pending_specs)

        def _submit_next_chunk() -> bool:
            try:
                chunk_index, current, out_chunk = next(spec_iter)
            except StopIteration:
                return False
            pending[
                executor.submit(
                    _run_tckgen_chunk,
                    cfg,
                    sid,
                    paths,
                    cutoff,
                    env,
                    chunk_index,
                    current,
                    out_chunk,
                )
            ] = (chunk_index, current, out_chunk)
            return True

        try:
            for _ in range(chunk_jobs):
                if not _submit_next_chunk():
                    break
            while pending:
                done_futures, _ = wait(list(pending.keys()), timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done_futures:
                    chunk_index, _, _ = pending.pop(future)
                    try:
                        future.result()
                    except Exception:
                        for other in pending:
                            other.cancel()
                        raise
                    complete.add(chunk_index)
                    generated = sum(current for idx, current, _ in specs if idx in complete)
                    percent = min(92, 15 + int(round(75.0 * generated / max(1, cfg.select_streamlines))))
                    _emit_stage_progress(
                        progress_queue,
                        sid,
                        percent,
                        f"tracks: cutoff {cutoff} chunks {len(complete)}/{total_chunks}",
                    )
                    _submit_next_chunk()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    else:
        for chunk_index, current, out_chunk in pending_specs:
            _run_tckgen_chunk(cfg, sid, paths, cutoff, env, chunk_index, current, out_chunk)
            complete.add(chunk_index)
            generated = sum(current for idx, current, _ in specs if idx in complete)
            percent = min(92, 15 + int(round(75.0 * generated / max(1, cfg.select_streamlines))))
            _emit_stage_progress(
                progress_queue,
                sid,
                percent,
                f"tracks: cutoff {cutoff} chunk {len(complete)}/{total_chunks}",
            )

    chunks = [str(out_chunk) for _, _, out_chunk in specs]
    _unlink(paths["TRACKS_FINAL"])
    _emit_stage_progress(progress_queue, sid, 95, f"tracks: cutoff {cutoff} concatenating")
    ok, logp, _ = _run_logged(
        cfg,
        sid,
        "tckedit_concat",
        [_tool(cfg, "tckedit")] + chunks + [str(paths["TRACKS_FINAL"]), "-force"],
        env=env,
    )
    if not ok:
        raise RuntimeError(f"tckedit concat failed; see {logp}")
    if _tck_count(cfg, paths["TRACKS_FINAL"]) < cfg.select_streamlines:
        raise RuntimeError("Concatenated tractogram is smaller than requested")


def _stage_tracks(cfg: Step7Config, sid: str, progress_queue: Queue | None = None) -> dict[str, str]:
    started_at = time.time()
    paths = _subject_paths(cfg, sid, create_dirs=True)
    mode = paths["DECONV_MODE_TXT"].read_text().strip() if paths["DECONV_MODE_TXT"].exists() else _select_deconv_mode(cfg)
    track_method = _track_method_label(cfg)
    try:
        _emit_stage_progress(progress_queue, sid, 2, "tracks: checking inputs")
        wmfod_input = _tracks_wmfod_input(paths)
        act_input = _tracks_act_input(paths)
        need = [("WMFOD_TRACK_INPUT", wmfod_input), ("ACT_TRACK_INPUT", act_input)]
        if cfg.track_seed_mode != "dynamic":
            need.append(("GMWMI", paths["GMWMI"]))
        missing = [name for name, path in need if not path.exists()]
        if missing:
            raise RuntimeError(f"FOD/ACT inputs missing: {', '.join(missing)}")
        if not _fod_stage_ready(cfg, sid):
            latest_fod_ok = _latest_summary_rows(
                cfg,
                stage="fod",
                wanted_sids={sid},
                statuses={"ok", "skip_exists"},
            ).get(sid, {})
            legacy_state, legacy_mode, legacy_source = _classify_legacy_fod_completion(
                cfg,
                sid,
                latest_fod_ok_row=latest_fod_ok,
                allow_infer_without_summary=True,
            )
            if legacy_state == "already_valid":
                mode = legacy_mode or mode
            elif legacy_state == "eligible":
                mode = legacy_mode
                _write_legacy_fod_completion_marker(cfg, sid, mode, legacy_source)
            elif legacy_state == "empty_wmfod":
                raise RuntimeError("empty_wmfod: WM FOD is empty; rerun stage='fod' with force=True for this subject")
            else:
                raise RuntimeError("incomplete_fod: rerun stage='fod' with force=True for this subject")
        if not _wmfod_has_valid_content(
            cfg,
            wmfod_input,
            context=f"stage_tracks sid={sid} fod_prereq",
            prefer_fast=True,
        ):
            raise RuntimeError("empty_wmfod: WM FOD is empty or unreadable; rerun stage='fod' with force=True for this subject")

        if _tck_count(cfg, paths["TRACKS_FINAL"]) >= cfg.select_streamlines and not cfg.force:
            _write_summary_row(cfg, sid, "tracks", "skip_exists", mode=mode)
            _emit_stage_progress(progress_queue, sid, 100, "tracks: already exists")
            return _stamp_stage_result({"sid": sid, "ok": True, "stage": "tracks", "error": "", "mode": mode}, started_at)

        _emit_stage_progress(progress_queue, sid, 8, "tracks: checking seed geometry")
        _tracks_geometry_preflight(cfg, sid, paths)

        _unlink(paths["TRACKS_FINAL"])
        cutoffs = list(cfg.track_cutoffs)
        for cutoff in cutoffs:
            try:
                _emit_stage_progress(progress_queue, sid, 15, f"tracks: tckgen cutoff {cutoff}")
                _chunked_tckgen(cfg, sid, paths, cutoff, mode, progress_queue=progress_queue)
                paths["TRACK_METHOD_TXT"].write_text(f"{track_method}\nseed_mode={cfg.track_seed_mode}\n", encoding="ascii")
                _write_summary_row(
                    cfg,
                    sid,
                    "tracks",
                    "ok",
                    note=(
                        f"cutoff={cutoff}; seed_mode={cfg.track_seed_mode}; method={track_method}; "
                        f"fod_input={_tracks_wmfod_input(paths).name}; act_input={_tracks_act_input(paths).name}"
                    ),
                    mode=mode,
                )
                _emit_stage_progress(progress_queue, sid, 100, "tracks: done")
                return _stamp_stage_result(
                    {
                        "sid": sid,
                        "ok": True,
                        "stage": "tracks",
                        "error": "",
                        "mode": mode,
                        "track_method": track_method,
                    },
                    started_at,
                )
            except Exception as exc:
                _write_summary_row(
                    cfg,
                    sid,
                    "tracks_retry",
                    "retry",
                    note=f"cutoff={cutoff}; seed_mode={cfg.track_seed_mode}; method={track_method}; err={exc}",
                    mode=mode,
                )
                _unlink(paths["TRACKS_FINAL"])
        raise RuntimeError(f"All tractography cutoffs failed; seed_mode={cfg.track_seed_mode}; method={track_method}")
    except Exception as exc:
        msg = f"{exc}\n{traceback.format_exc()}"
        _write_summary_row(cfg, sid, "tracks", "error", msg, mode=mode)
        _emit_stage_progress(progress_queue, sid, 100, "tracks: failed")
        return _stamp_stage_result(
            {
                "sid": sid,
                "ok": False,
                "stage": "tracks",
                "error": msg,
                "mode": mode,
                "track_method": track_method,
            },
            started_at,
        )


def _tck2connectome(
    cfg: Step7Config,
    sid: str,
    paths: dict[str, Path],
    out_csv: Path,
    *,
    tck_weights: Path | None = None,
    scale: str | None = None,
    scale_file: Path | None = None,
    stat_edge: str = "mean",
    out_assignments: Path | None = None,
) -> None:
    _unlink(out_csv)
    cmd = [
        _tool(cfg, "tck2connectome"),
        str(paths["TRACKS_FINAL"]),
        str(paths["AAL_B0"]),
        str(out_csv),
        "-symmetric",
        "-zero_diagonal",
    ]
    cmd += _assignment_args(cfg)
    if tck_weights is not None:
        cmd += ["-tck_weights_in", str(tck_weights)]
    if scale == "length":
        cmd += ["-scale_length"]
    elif scale == "invlength":
        cmd += ["-scale_invlength"]
    elif scale == "invnodevol":
        cmd += ["-scale_invnodevol"]
    if scale_file is not None:
        cmd += ["-scale_file", str(scale_file)]
    if out_assignments is not None:
        _unlink(out_assignments)
        cmd += ["-out_assignments", str(out_assignments)]
    cmd += ["-stat_edge", stat_edge]
    ok, logp, _ = _run_logged(cfg, sid, f"tck2connectome_{out_csv.stem}", cmd)
    if not ok:
        raise RuntimeError(f"tck2connectome failed for {out_csv.name}; see {logp}")


def _run_mrtrix_dti_metrics(
    cfg: Step7Config,
    sid: str,
    paths: dict[str, Path],
    tensor_dwi: Path,
    mask_for_tensor: Path,
) -> None:
    ok, logp, _ = _run_logged(
        cfg,
        sid,
        "dwi2tensor",
        [
            _tool(cfg, "dwi2tensor"),
            str(tensor_dwi),
            str(paths["DT_TEN"]),
            "-mask",
            str(mask_for_tensor),
        ],
    )
    if not ok:
        raise RuntimeError(f"dwi2tensor failed; see {logp}")
    ok, logp, _ = _run_logged(
        cfg,
        sid,
        "tensor2metric",
        [
            _tool(cfg, "tensor2metric"),
            str(paths["DT_TEN"]),
            "-fa",
            str(paths["FA_MIF"]),
            "-adc",
            str(paths["MD_MIF"]),
            "-ad",
            str(paths["AD_MIF"]),
            "-rd",
            str(paths["RD_MIF"]),
        ],
    )
    if not ok:
        raise RuntimeError(f"tensor2metric failed; see {logp}")
    paths["DTI_METHOD_TXT"].write_text("mrtrix_dwi2tensor_wls_iwls\n", encoding="ascii")


def _run_fsl_dtifit_metrics(cfg: Step7Config, sid: str, paths: dict[str, Path], tensor_dwi: Path) -> None:
    dtifit = shutil.which("dtifit", path=_base_env(cfg)["PATH"])
    fslmaths = shutil.which("fslmaths", path=_base_env(cfg)["PATH"])
    if not dtifit or not fslmaths:
        raise RuntimeError("FSL dtifit/fslmaths not found for DTI repair fallback")
    with tempfile.TemporaryDirectory(prefix=f"dtifit_{sid}_", dir=str(paths["DT_TEN"].parent)) as td:
        work = Path(td)
        dwi_nii = work / "dwi.nii.gz"
        mask_nii = work / "mask.nii.gz"
        bvec = work / "bvec"
        bval = work / "bval"
        out_prefix = work / "dtifit"
        rd_nii = work / "rd.nii.gz"
        ok, logp, _ = _run_logged(
            cfg,
            sid,
            "dtifit_mrconvert_dwi",
            [
                _tool(cfg, "mrconvert"),
                str(tensor_dwi),
                str(dwi_nii),
                "-export_grad_fsl",
                str(bvec),
                str(bval),
                "-force",
            ],
            cwd=work,
        )
        if not ok:
            raise RuntimeError(f"mrconvert DWI for dtifit failed; see {logp}")
        ok, logp, _ = _run_logged(
            cfg,
            sid,
            "dtifit_mrconvert_mask",
            [_tool(cfg, "mrconvert"), str(_post_tensor_mask(paths)), str(mask_nii), "-force"],
            cwd=work,
        )
        if not ok:
            raise RuntimeError(f"mrconvert mask for dtifit failed; see {logp}")
        ok, logp, _ = _run_logged(
            cfg,
            sid,
            "dtifit_fallback",
            [
                dtifit,
                "-k",
                str(dwi_nii),
                "-o",
                str(out_prefix),
                "-m",
                str(mask_nii),
                "-r",
                str(bvec),
                "-b",
                str(bval),
                "--save_tensor",
            ],
            cwd=work,
        )
        if not ok:
            raise RuntimeError(f"FSL dtifit fallback failed; see {logp}")
        convert_pairs = [
            (work / "dtifit_tensor.nii.gz", paths["DT_TEN"], "dtifit_mrconvert_tensor"),
            (work / "dtifit_FA.nii.gz", paths["FA_MIF"], "dtifit_mrconvert_fa"),
            (work / "dtifit_MD.nii.gz", paths["MD_MIF"], "dtifit_mrconvert_md"),
            (work / "dtifit_L1.nii.gz", paths["AD_MIF"], "dtifit_mrconvert_ad"),
        ]
        for src, dst, tag in convert_pairs:
            ok, logp, _ = _run_logged(
                cfg,
                sid,
                tag,
                [_tool(cfg, "mrconvert"), str(src), str(dst), "-force"],
                cwd=work,
            )
            if not ok:
                raise RuntimeError(f"{tag} failed; see {logp}")
        ok, logp, _ = _run_logged(
            cfg,
            sid,
            "dtifit_rd",
            [fslmaths, str(work / "dtifit_L2.nii.gz"), "-add", str(work / "dtifit_L3.nii.gz"), "-div", "2", str(rd_nii)],
            cwd=work,
        )
        if not ok:
            raise RuntimeError(f"FSL RD calculation failed; see {logp}")
        ok, logp, _ = _run_logged(
            cfg,
            sid,
            "dtifit_mrconvert_rd",
            [_tool(cfg, "mrconvert"), str(rd_nii), str(paths["RD_MIF"]), "-force"],
            cwd=work,
        )
        if not ok:
            raise RuntimeError(f"mrconvert RD for dtifit failed; see {logp}")
    fa_tmp = paths["FA_MIF"].with_name("fa_bounded_tmp.mif")
    _unlink(fa_tmp)
    ok, logp, _ = _run_logged(
        cfg,
        sid,
        "dtifit_bound_fa",
        [_tool(cfg, "mrcalc"), str(paths["FA_MIF"]), "0", "-max", "1", "-min", str(fa_tmp), "-force"],
    )
    if not ok:
        raise RuntimeError(f"FA bounding after dtifit failed; see {logp}")
    fa_tmp.replace(paths["FA_MIF"])
    paths["DTI_METHOD_TXT"].write_text("fsl_dtifit_fallback_fa_bounded\n", encoding="ascii")


def _stage_post(cfg: Step7Config, sid: str, progress_queue: Queue | None = None) -> dict[str, str]:
    started_at = time.time()
    paths = _subject_paths(cfg, sid, create_dirs=True)
    mode = paths["DECONV_MODE_TXT"].read_text().strip() if paths["DECONV_MODE_TXT"].exists() else _select_deconv_mode(cfg)
    try:
        _emit_stage_progress(progress_queue, sid, 2, "post: checking inputs")
        track_count = _tck_count(cfg, paths["TRACKS_FINAL"])
        if track_count <= 0:
            raise RuntimeError("tracks_final missing or empty")
        node_count = _atlas_node_count(cfg)
        if not paths["SC_ALL"].exists():
            for key in ("FA_TSF", "MD_TSF", "AD_TSF", "RD_TSF"):
                _unlink(paths[key])
        parc_done = False
        if paths["AAL_B0"].exists() and bool(_mrinfo_size(cfg, paths["AAL_B0"])):
            try:
                _validate_aal_b0(cfg, paths, node_count)
                parc_done = True
            except Exception as exc:
                _append_runtime_note(
                    cfg.runtime_log,
                    f"Step7 post parc QC rerun {time.strftime('%Y-%m-%dT%H:%M:%S')} | sid={sid} | {exc}",
                )
        if not parc_done and not cfg.aal_mni.exists():
            raise RuntimeError(f"AAL atlas missing: {cfg.aal_mni}")

        if not _sift2_ready_for_tracks(paths, track_count) or cfg.force:
            _unlink(paths["WEIGHTS"])
            _unlink(paths["MU_TXT"])
            sift_wmfod = _tracks_wmfod_input(paths)
            sift_act = _tracks_act_input(paths)
            use_fd_scale_gm = cfg.sift_fd_scale_gm
            if use_fd_scale_gm is None:
                use_fd_scale_gm = mode == "single_shell"
            for attempt, fd_scale_gm in enumerate((use_fd_scale_gm, not use_fd_scale_gm), start=1):
                if attempt > 1:
                    _append_runtime_note(
                        cfg.runtime_log,
                        f"Step7 post SIFT2 alternate retry {time.strftime('%Y-%m-%dT%H:%M:%S')} | "
                        f"sid={sid} | fd_scale_gm={fd_scale_gm}",
                    )
                    _unlink(paths["WEIGHTS"])
                    _unlink(paths["MU_TXT"])
                cmd = [
                    _tool(cfg, "tcksift2"),
                    str(paths["TRACKS_FINAL"]),
                    str(sift_wmfod),
                    str(paths["WEIGHTS"]),
                    "-act",
                    str(sift_act),
                    "-out_mu",
                    str(paths["MU_TXT"]),
                    "-nthreads",
                    "1",
                ]
                if fd_scale_gm:
                    cmd.append("-fd_scale_gm")
                ok, logp, _ = _run_logged(cfg, sid, "tcksift2" if attempt == 1 else "tcksift2_alt", cmd)
                if not ok:
                    if attempt == 2:
                        raise RuntimeError(f"tcksift2 failed; see {logp}")
                    continue
                try:
                    _validate_sift2_outputs(paths, track_count)
                    break
                except Exception:
                    if attempt == 2:
                        raise
        _validate_sift2_outputs(paths, track_count)

        if not parc_done or cfg.force:
            for key in ("AAL_T1", "MNI2T1", "MNI2B0", "AAL_B0"):
                _unlink(paths[key])
            flirt = shutil.which("flirt", path=_base_env(cfg)["PATH"])
            if not flirt:
                raise RuntimeError("FSL flirt not found")
            convert_xfm = shutil.which("convert_xfm", path=_base_env(cfg)["PATH"])
            if not convert_xfm:
                raise RuntimeError("FSL convert_xfm not found")
            native_t1, native_t1_brain, native_t1_source = _find_native_t1_inputs(cfg, sid)
            native_t1_ref = native_t1_brain or native_t1
            aal_route = str(getattr(cfg, "aal_transform_route", "current_direct") or "current_direct").strip().lower()
            if aal_route in {"current_direct", "direct", "composed"}:
                ok, logp, _ = _run_logged(
                    cfg,
                    sid,
                    "flirt_mni2t1",
                    [
                        flirt,
                        "-in",
                        str(cfg.fsl_dir / "data/standard/MNI152_T1_1mm_brain.nii.gz"),
                        "-ref",
                        str(native_t1_ref),
                        "-omat",
                        str(paths["MNI2T1"]),
                        "-dof",
                        "12",
                    ],
                )
                if not ok:
                    raise RuntimeError(f"MNI->T1 flirt failed; see {logp}")
                ok, logp, _ = _run_logged(
                    cfg,
                    sid,
                    "convert_xfm_mni2b0",
                    [
                        convert_xfm,
                        "-omat",
                        str(paths["MNI2B0"]),
                        "-concat",
                        str(paths["T12B0"]),
                        str(paths["MNI2T1"]),
                    ],
                )
                if not ok:
                    raise RuntimeError(f"MNI->B0 transform composition failed; see {logp}")
                ok, logp, _ = _run_logged(
                    cfg,
                    sid,
                    "flirt_aal_to_b0_direct",
                    [
                        flirt,
                        "-in",
                        str(cfg.aal_mni),
                        "-ref",
                        str(paths["B0MEAN"]),
                        "-applyxfm",
                        "-init",
                        str(paths["MNI2B0"]),
                        "-interp",
                        "nearestneighbour",
                        "-datatype",
                        "int",
                        "-out",
                        str(paths["AAL_B0"]),
                    ],
                )
                if not ok:
                    raise RuntimeError(f"Direct AAL->B0 flirt failed; see {logp}")
            elif aal_route in {"supervisor_two_step", "native_t1_two_step", "reference_two_step"}:
                ok, logp, _ = _run_logged(
                    cfg,
                    sid,
                    "flirt_mni2t1_supervisor",
                    [
                        flirt,
                        "-in",
                        str(cfg.fsl_dir / "data/standard/MNI152_T1_1mm_brain.nii.gz"),
                        "-ref",
                        str(native_t1_ref),
                        "-omat",
                        str(paths["MNI2T1"]),
                        "-dof",
                        "12",
                    ],
                )
                if not ok:
                    raise RuntimeError(f"Supervisor-route MNI->T1 flirt failed; see {logp}")
                ok, logp, _ = _run_logged(
                    cfg,
                    sid,
                    "flirt_aal_to_t1_supervisor",
                    [
                        flirt,
                        "-in",
                        str(cfg.aal_mni),
                        "-ref",
                        str(native_t1_ref),
                        "-applyxfm",
                        "-init",
                        str(paths["MNI2T1"]),
                        "-interp",
                        "nearestneighbour",
                        "-datatype",
                        "int",
                        "-out",
                        str(paths["AAL_T1"]),
                    ],
                )
                if not ok:
                    raise RuntimeError(f"Supervisor-route AAL->T1 flirt failed; see {logp}")
                ok, logp, _ = _run_logged(
                    cfg,
                    sid,
                    "flirt_aal_t1_to_b0_supervisor",
                    [
                        flirt,
                        "-in",
                        str(paths["AAL_T1"]),
                        "-ref",
                        str(paths["B0MEAN"]),
                        "-applyxfm",
                        "-init",
                        str(paths["T12B0"]),
                        "-interp",
                        "nearestneighbour",
                        "-datatype",
                        "int",
                        "-out",
                        str(paths["AAL_B0"]),
                    ],
                )
                if not ok:
                    raise RuntimeError(f"Supervisor-route AAL T1->B0 flirt failed; see {logp}")
            else:
                raise RuntimeError(f"Unsupported aal_transform_route={cfg.aal_transform_route!r}")
            _validate_aal_b0(cfg, paths, node_count)

        dti_done = all(
            paths[key].exists() and bool(_mrinfo_size(cfg, paths[key]))
            for key in ("DT_TEN", "FA_MIF", "MD_MIF", "AD_MIF", "RD_MIF")
        )
        if dti_done and not cfg.force:
            try:
                _sanitize_dti_scalar_maps(cfg, sid, paths)
                _validate_dti_maps(cfg, paths)
            except Exception as exc:
                _append_runtime_note(
                    cfg.runtime_log,
                    f"Step7 post DTI QC rerun {time.strftime('%Y-%m-%dT%H:%M:%S')} | sid={sid} | {exc}",
                )
                dti_done = False
        if not dti_done or cfg.force:
            for key in ("DT_TEN", "FA_MIF", "MD_MIF", "AD_MIF", "RD_MIF"):
                _unlink(paths[key])
            tensor_dwi = paths["DWI"]
            if not _has_valid_gradients(cfg, tensor_dwi) and paths["DWI_READY"].exists() and _has_valid_gradients(cfg, paths["DWI_READY"]):
                tensor_dwi = paths["DWI_READY"]
            _ensure_post_dwi_mask(cfg, sid, paths, tensor_dwi)
            mask_for_tensor = _post_tensor_mask(paths)
            _run_mrtrix_dti_metrics(cfg, sid, paths, tensor_dwi, mask_for_tensor)
            _sanitize_dti_scalar_maps(cfg, sid, paths)
            try:
                _validate_dti_maps(cfg, paths)
            except Exception as exc:
                _append_runtime_note(
                    cfg.runtime_log,
                    f"Step7 post DTI dtifit fallback {time.strftime('%Y-%m-%dT%H:%M:%S')} | sid={sid} | {exc}",
                )
                for key in ("DT_TEN", "FA_MIF", "MD_MIF", "AD_MIF", "RD_MIF"):
                    _unlink(paths[key])
                _run_fsl_dtifit_metrics(cfg, sid, paths, tensor_dwi)
                _sanitize_dti_scalar_maps(cfg, sid, paths)
        _validate_dti_maps(cfg, paths)

        for img_key, tsf_key, tag in (
            ("FA_MIF", "FA_TSF", "tcksample_fa"),
            ("MD_MIF", "MD_TSF", "tcksample_md"),
            ("AD_MIF", "AD_TSF", "tcksample_ad"),
            ("RD_MIF", "RD_TSF", "tcksample_rd"),
        ):
            if paths[tsf_key].exists() and not cfg.force:
                try:
                    if tsf_key == "FA_TSF":
                        _validate_track_scalar_file(
                            paths[tsf_key],
                            expected_count=track_count,
                            label="FA tcksample",
                            min_allowed=0.0,
                            max_allowed=1.3,
                        )
                    elif tsf_key in {"MD_TSF", "AD_TSF", "RD_TSF"}:
                        _validate_track_scalar_file(
                            paths[tsf_key],
                            expected_count=track_count,
                            label=f"{tsf_key.removesuffix('_TSF')} tcksample",
                            min_allowed=-0.02,
                            max_allowed=0.05,
                        )
                except Exception as exc:
                    _append_runtime_note(
                        cfg.runtime_log,
                        f"Step7 post tcksample QC rerun {time.strftime('%Y-%m-%dT%H:%M:%S')} | sid={sid} | {tag} | {exc}",
                    )
                    _unlink(paths[tsf_key])
            if not paths[tsf_key].exists() or cfg.force:
                _unlink(paths[tsf_key])
                ok, logp, _ = _run_logged(
                    cfg,
                    sid,
                    tag,
                    [
                        _tool(cfg, "tcksample"),
                        str(paths["TRACKS_FINAL"]),
                        str(paths[img_key]),
                        str(paths[tsf_key]),
                        "-stat_tck",
                        "mean",
                    ],
                )
                if not ok:
                    raise RuntimeError(f"{tag} failed; see {logp}")
                if tsf_key == "FA_TSF":
                    _validate_track_scalar_file(
                        paths[tsf_key],
                        expected_count=track_count,
                        label="FA tcksample",
                        min_allowed=0.0,
                        max_allowed=1.3,
                    )
                elif tsf_key in {"MD_TSF", "AD_TSF", "RD_TSF"}:
                    _validate_track_scalar_file(
                        paths[tsf_key],
                        expected_count=track_count,
                        label=f"{tsf_key.removesuffix('_TSF')} tcksample",
                        min_allowed=-0.02,
                        max_allowed=0.05,
                    )

        _tck2connectome(
            cfg,
            sid,
            paths,
            paths["SC_COUNT"],
            stat_edge="sum",
            out_assignments=paths["ASSIGNMENTS"] if cfg.write_assignments else None,
        )
        _tck2connectome(cfg, sid, paths, paths["SC_FD_SUM"], tck_weights=paths["WEIGHTS"], stat_edge="sum")
        _tck2connectome(cfg, sid, paths, paths["SC_LEN_MEAN"], scale="length", stat_edge="mean")
        _tck2connectome(cfg, sid, paths, paths["SC_INVLEN_MEAN"], scale="invlength", stat_edge="mean")
        _tck2connectome(cfg, sid, paths, paths["SC_FA_MEAN"], scale_file=paths["FA_TSF"], stat_edge="mean")
        _tck2connectome(cfg, sid, paths, paths["SC_MD_MEAN"], scale_file=paths["MD_TSF"], stat_edge="mean")
        _tck2connectome(cfg, sid, paths, paths["SC_AD_MEAN"], scale_file=paths["AD_TSF"], stat_edge="mean")
        _tck2connectome(cfg, sid, paths, paths["SC_RD_MEAN"], scale_file=paths["RD_TSF"], stat_edge="mean")
        if cfg.write_invnodevol_count:
            _tck2connectome(
                cfg,
                sid,
                paths,
                paths["SC_COUNT_INVNODEVOL"],
                scale="invnodevol",
                stat_edge="sum",
            )

        raw_count = _read_square_csv(paths["SC_COUNT"])
        if node_count is None:
            node_count = len(raw_count)
        count_matrix = _sanitize_connectome_matrix(
            raw_count,
            _pad_square_matrix(raw_count, node_count, label="count"),
            metric="count",
            node_count=node_count,
        )
        matrix_paths = {
            "count": paths["SC_COUNT"],
            "fd_sum": paths["SC_FD_SUM"],
            "len_mean": paths["SC_LEN_MEAN"],
            "invlen_mean": paths["SC_INVLEN_MEAN"],
            "fa_mean": paths["SC_FA_MEAN"],
            "md_mean": paths["SC_MD_MEAN"],
            "ad_mean": paths["SC_AD_MEAN"],
            "rd_mean": paths["SC_RD_MEAN"],
        }
        matrices = {"count": count_matrix}
        for metric, matrix_path in matrix_paths.items():
            if metric == "count":
                continue
            matrices[metric] = _sanitize_connectome_matrix(
                _read_square_csv(matrix_path),
                count_matrix,
                metric=metric,
                node_count=node_count,
            )
        for metric, matrix_path in matrix_paths.items():
            _write_square_csv(matrix_path, matrices[metric])
        if cfg.write_invnodevol_count and paths["SC_COUNT_INVNODEVOL"].exists():
            matrices["count_invnodevol"] = _sanitize_connectome_matrix(
                _read_square_csv(paths["SC_COUNT_INVNODEVOL"]),
                count_matrix,
                metric="count_invnodevol",
                node_count=node_count,
            )
            _write_square_csv(paths["SC_COUNT_INVNODEVOL"], matrices["count_invnodevol"])
        _write_long_csv(paths["SC_ALL"], matrices)

        weight_stats = _scalar_txt_stats(paths["WEIGHTS"])
        track_method = (
            paths["TRACK_METHOD_TXT"].read_text(encoding="ascii", errors="ignore").strip().replace("\n", "|")
            if paths["TRACK_METHOD_TXT"].exists()
            else "unknown"
        )
        _write_sc_generation_provenance(
            cfg,
            sid,
            paths,
            matrices,
            track_count=track_count,
            node_count=node_count,
            weight_stats=weight_stats,
            track_method=track_method,
            mode=mode,
        )
        _write_spatial_contract_status(
            cfg,
            sid,
            paths,
            matrices,
            track_count=track_count,
            node_count=node_count,
            mode=mode,
        )
        note = (
            f"tracks={track_count}; "
            f"sift_n={int(weight_stats['count'])}; "
            f"nodes={node_count}; "
            f"mu={paths['MU_TXT'].read_text().strip() if paths['MU_TXT'].exists() else 'nan'}; "
            f"track_method={track_method}"
        )
        _write_summary_row(cfg, sid, "post", "ok", note=note, mode=mode)
        _emit_stage_progress(progress_queue, sid, 100, "post: done")
        return _stamp_stage_result({"sid": sid, "ok": True, "stage": "post", "error": "", "mode": mode}, started_at)
    except Exception as exc:
        msg = f"{exc}\n{traceback.format_exc()}"
        _write_summary_row(cfg, sid, "post", "error", msg, mode=mode)
        _emit_stage_progress(progress_queue, sid, 100, "post: failed")
        return _stamp_stage_result({"sid": sid, "ok": False, "stage": "post", "error": msg, "mode": mode}, started_at)


def _stage_done(cfg: Step7Config, stage: str, sid: str) -> bool:
    paths = _subject_paths(cfg, sid, create_dirs=False)
    if stage == "prep":
        if not (
            paths["FIVE_FIX"].exists()
            and paths["MASK_5TT_BRAIN"].exists()
            and paths["GMWMI"].exists()
        ):
            return False
        try:
            _validate_prep_geometry(cfg, sid, paths)
        except Step7GeometryError:
            return False
        return True
    if stage == "fod":
        return _fod_stage_ready(cfg, sid)
    if stage == "tracks":
        return _tck_count(cfg, paths["TRACKS_FINAL"]) >= cfg.select_streamlines
    if stage == "post":
        return _post_outputs_complete(cfg, paths)
    raise ValueError(f"Unknown stage: {stage}")


def _post_outputs_complete(cfg: Step7Config, paths: dict[str, Path]) -> bool:
    required = [
        "SC_ALL",
        "SC_COUNT",
        "SC_FD_SUM",
        "SC_LEN_MEAN",
        "SC_INVLEN_MEAN",
        "SC_FA_MEAN",
        "SC_MD_MEAN",
        "SC_AD_MEAN",
        "SC_RD_MEAN",
    ]
    if cfg.write_invnodevol_count:
        required.append("SC_COUNT_INVNODEVOL")
    for key in required:
        path = paths[key]
        try:
            if not path.exists() or path.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


def _stage_done_quick(cfg: Step7Config, stage: str, sid: str) -> bool:
    paths = _subject_paths(cfg, sid, create_dirs=False)
    if stage == "prep":
        return (
            paths["FIVE_FIX"].exists()
            and paths["MASK_5TT_BRAIN"].exists()
            and paths["GMWMI"].exists()
        )
    if stage == "fod":
        return _fod_stage_ready(cfg, sid, quick=True)
    if stage == "tracks":
        return paths["TRACKS_FINAL"].exists()
    if stage == "post":
        return _post_outputs_complete(cfg, paths)
    raise ValueError(f"Unknown stage: {stage}")


def _pipeline_stage_done_quick(cfg: Step7Config, stage: str, sid: str) -> bool:
    paths = _subject_paths(cfg, sid, create_dirs=False)
    if stage == "mif":
        return paths["RAW_DWI_MIF"].exists()
    if stage == "denoise":
        return paths["DENOISED_DWI"].exists()
    if stage == "gibbs":
        return paths["GIBBS_DWI"].exists()
    if stage == "eddy":
        return paths["EDDY_DWI"].exists() or paths["EDDY_DWI_ALT"].exists()
    return _stage_done_quick(cfg, stage, sid)


def _stage_prereq_ready(cfg: Step7Config, stage: str, sid: str) -> bool:
    paths = _subject_paths(cfg, sid, create_dirs=False)
    if stage == "prep":
        return _bbr_ready(paths)
    if stage == "fod":
        return _stage_done(cfg, "prep", sid)
    if stage == "tracks":
        return _stage_done(cfg, "fod", sid)
    if stage == "post":
        return _stage_done(cfg, "tracks", sid)
    raise ValueError(f"Unknown stage: {stage}")


def _stage_scan_done(cfg: Step7Config, stage: str, sid: str) -> bool:
    if stage in {"prep", "fod"}:
        return _stage_done(cfg, stage, sid)
    if stage == "post":
        return _stage_done_quick(cfg, stage, sid)
    return _stage_done(cfg, stage, sid)


def _stage_scan_prereq_ready(cfg: Step7Config, stage: str, sid: str) -> bool:
    paths = _subject_paths(cfg, sid, create_dirs=False)
    if stage == "prep":
        return _bbr_ready(paths)
    if stage == "fod":
        return _stage_done(cfg, "prep", sid)
    if stage == "tracks":
        return _stage_done(cfg, "fod", sid)
    if stage == "post":
        return _stage_done(cfg, "tracks", sid)
    raise ValueError(f"Unknown stage: {stage}")


def _scan_stage_subject(cfg: Step7Config, stage: str, sid: str) -> tuple[str, bool, bool]:
    prereq_ready = _stage_scan_prereq_ready(cfg, stage, sid)
    if not prereq_ready:
        return sid, False, False
    if cfg.force:
        return sid, True, False
    return sid, True, _stage_scan_done(cfg, stage, sid)


def _scan_stage_subjects(
    cfg: Step7Config,
    stage: str,
    all_sids: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    worker_count = _resolve_scan_workers(cfg, len(all_sids))
    states: dict[str, tuple[bool, bool]] = {}
    runnable_count = 0
    done_count = 0
    pending_count = 0
    skipped_count = 0
    with tqdm(
        total=len(all_sids),
        desc=f"step7:{stage}:scan",
        unit="sid",
        position=0,
        dynamic_ncols=True,
        leave=False,
    ) as scan_bar:
        if worker_count == 1:
            for sid in all_sids:
                _, prereq_ready, stage_done = _scan_stage_subject(cfg, stage, sid)
                states[sid] = (prereq_ready, stage_done)
                if prereq_ready:
                    runnable_count += 1
                    if stage_done:
                        done_count += 1
                    else:
                        pending_count += 1
                else:
                    skipped_count += 1
                scan_bar.update(1)
                scan_bar.set_postfix_str(
                    f"runnable={runnable_count} | done={done_count} | pending={pending_count} | skipped={skipped_count}",
                    refresh=False,
                )
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                pending = {executor.submit(_scan_stage_subject, cfg, stage, sid): sid for sid in all_sids}
                while pending:
                    done_futures, _ = wait(list(pending.keys()), timeout=0.2, return_when=FIRST_COMPLETED)
                    if not done_futures:
                        continue
                    for future in done_futures:
                        pending.pop(future)
                        sid, prereq_ready, stage_done = future.result()
                        states[sid] = (prereq_ready, stage_done)
                        if prereq_ready:
                            runnable_count += 1
                            if stage_done:
                                done_count += 1
                            else:
                                pending_count += 1
                        else:
                            skipped_count += 1
                        scan_bar.update(1)
                        scan_bar.set_postfix_str(
                            f"runnable={runnable_count} | done={done_count} | pending={pending_count} | skipped={skipped_count}",
                            refresh=False,
                        )

    runnable: list[str] = []
    done: list[str] = []
    todo: list[str] = []
    skipped_prereq: list[str] = []
    for sid in all_sids:
        prereq_ready, stage_done = states.get(sid, (False, False))
        if prereq_ready:
            runnable.append(sid)
            if stage_done:
                done.append(sid)
            else:
                todo.append(sid)
        else:
            skipped_prereq.append(sid)
    return runnable, skipped_prereq, todo, done


def collect_step7_status(
    cfg: Step7Config,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, object]:
    sids = _discover_subjects(cfg, include=include, exclude=exclude)
    counts = {
        "eligible_bias": len(sids),
        "bbr_ready": 0,
        "prep_ready": 0,
        "fod_ready": 0,
        "fod_input_qc_failed": 0,
        "tracks_ready": 0,
        "sift_ready": 0,
        "parc_ready": 0,
        "dti_ready": 0,
        "connectome_ready": 0,
    }
    missing_bbr: list[str] = []
    fod_input_qc_failed: list[str] = []
    missing_connectome: list[str] = []
    for sid in sids:
        paths = _subject_paths(cfg, sid)
        if _bbr_ready(paths):
            counts["bbr_ready"] += 1
        else:
            if len(missing_bbr) < 10:
                missing_bbr.append(sid)
        if _stage_done(cfg, "prep", sid):
            counts["prep_ready"] += 1
        if _stage_done(cfg, "fod", sid):
            counts["fod_ready"] += 1
        elif _fod_input_qc_failed(paths):
            counts["fod_input_qc_failed"] += 1
            if len(fod_input_qc_failed) < 10:
                fod_input_qc_failed.append(sid)
        if _stage_done(cfg, "tracks", sid):
            counts["tracks_ready"] += 1
        if paths["WEIGHTS"].exists() and _parse_weights_txt(paths["WEIGHTS"]):
            counts["sift_ready"] += 1
        if paths["AAL_B0"].exists():
            counts["parc_ready"] += 1
        if all(paths[name].exists() for name in ("FA_MIF", "MD_MIF", "AD_MIF", "RD_MIF")):
            counts["dti_ready"] += 1
        if paths["SC_ALL"].exists():
            counts["connectome_ready"] += 1
        else:
            if len(missing_connectome) < 10:
                missing_connectome.append(sid)
    return {
        "counts": counts,
        "examples_missing_bbr": missing_bbr,
        "examples_fod_input_qc_failed": fod_input_qc_failed,
        "examples_missing_connectome": missing_connectome,
    }


def _subject_root_from_sid(sid: str) -> str:
    parts = str(sid).split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else str(sid)


def _recode_step7_group(value: str) -> str:
    value = str(value).strip()
    if value in {"EMCI", "LMCI", "SMC"}:
        return "MCI"
    return value


def _resolve_step7_cohort_dti_csv(cohort_dti_csv: str | Path | None = None) -> Path:
    candidates: list[Path] = []
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
    raise FileNotFoundError(f"Could not find cohort DTI CSV. Tried: {', '.join(str(p) for p in candidates)}")


def _load_step7_group_map(cohort_dti_csv: Path) -> dict[str, str]:
    group_map: dict[str, str] = {}
    with open(cohort_dti_csv, "r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = (row.get("Subject ID") or row.get("subject_id") or "").strip()
            if not subject_id or subject_id in group_map:
                continue
            group = _recode_step7_group(row.get("Research Group") or row.get("group") or "")
            if group:
                group_map[subject_id] = group
    return group_map


def collect_step7_group_status(
    cfg: Step7Config,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    cohort_dti_csv: str | Path | None = None,
    scope: str = "step7",
) -> dict[str, object]:
    sids = _discover_group_status_subjects(cfg, include=include, exclude=exclude, scope=scope)
    cohort_path = _resolve_step7_cohort_dti_csv(cohort_dti_csv)
    group_map = _load_step7_group_map(cohort_path)
    stages = ("mif", "denoise", "gibbs", "eddy", "prep", "fod", "tracks", "post")
    rows: dict[str, dict[str, int | str]] = {
        group: {"group": group, "total": 0, **{f"{stage}_done": 0 for stage in stages}}
        for group in ("CN", "MCI", "AD")
    }
    unknown_subjects: list[str] = []

    for sid in sids:
        subject_id = _subject_root_from_sid(sid)
        group = group_map.get(subject_id, "")
        if group not in rows:
            if subject_id not in unknown_subjects and len(unknown_subjects) < 20:
                unknown_subjects.append(subject_id)
            continue
        rows[group]["total"] = int(rows[group]["total"]) + 1
        for stage in stages:
            if _pipeline_stage_done_quick(cfg, stage, sid):
                rows[group][f"{stage}_done"] = int(rows[group][f"{stage}_done"]) + 1

    ordered_rows: list[dict[str, int | str]] = []
    for group in ("CN", "MCI", "AD"):
        base = rows[group]
        total = int(base["total"])
        row: dict[str, int | str] = {"group": group, "total": total}
        for stage in stages:
            done = int(base[f"{stage}_done"])
            row[f"{stage}_done"] = done
            row[f"{stage}_remaining"] = max(0, total - done)
        ordered_rows.append(row)

    return {
        "rows": ordered_rows,
        "total_series": len(sids),
        "scope": scope,
        "cohort_dti_csv": str(cohort_path),
        "unknown_subjects": unknown_subjects,
    }


def print_step7_status(
    cfg: Step7Config,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, object]:
    result = collect_step7_status(cfg, include=include, exclude=exclude)
    counts = result["counts"]
    print("Step 7 status")
    print("-------------")
    print(f"eligible bias series        : {counts['eligible_bias']}")
    print(f"bbr key files ready         : {counts['bbr_ready']}")
    print(f"prep ACT geometry ready     : {counts['prep_ready']}")
    print(f"fod ACT-ready               : {counts['fod_ready']}")
    print(f"fod input qc failed         : {counts['fod_input_qc_failed']}")
    print(f"tracks done                 : {counts['tracks_ready']}")
    print(f"sift2 done                  : {counts['sift_ready']}")
    print(f"parcellation done           : {counts['parc_ready']}")
    print(f"dti maps done               : {counts['dti_ready']}")
    print(f"final connectomes done      : {counts['connectome_ready']}")
    if result["examples_missing_bbr"]:
        print("examples missing bbr : " + ", ".join(result["examples_missing_bbr"]))
    if result["examples_fod_input_qc_failed"]:
        print("examples fod input qc: " + ", ".join(result["examples_fod_input_qc_failed"]))
    if result["examples_missing_connectome"]:
        print("examples missing conn: " + ", ".join(result["examples_missing_connectome"]))
    return result


def collect_step7_status_quick(
    cfg: Step7Config,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, object]:
    sids = _discover_subjects(cfg, include=include, exclude=exclude)
    counts = {
        "eligible_bias": len(sids),
        "bbr_ready": 0,
        "prep_ready": 0,
        "fod_ready": 0,
        "fod_input_qc_failed": 0,
        "tracks_ready": 0,
        "sift_ready": 0,
        "parc_ready": 0,
        "dti_ready": 0,
        "connectome_ready": 0,
    }
    missing_bbr: list[str] = []
    fod_input_qc_failed: list[str] = []
    missing_connectome: list[str] = []
    for sid in sids:
        paths = _subject_paths(cfg, sid)
        if _bbr_ready(paths):
            counts["bbr_ready"] += 1
        else:
            if len(missing_bbr) < 10:
                missing_bbr.append(sid)
        if _stage_done_quick(cfg, "prep", sid):
            counts["prep_ready"] += 1
        if _stage_done_quick(cfg, "fod", sid):
            counts["fod_ready"] += 1
        elif _fod_input_qc_failed(paths):
            counts["fod_input_qc_failed"] += 1
            if len(fod_input_qc_failed) < 10:
                fod_input_qc_failed.append(sid)
        if _stage_done_quick(cfg, "tracks", sid):
            counts["tracks_ready"] += 1
        if paths["WEIGHTS"].exists():
            try:
                if paths["WEIGHTS"].stat().st_size > 0:
                    counts["sift_ready"] += 1
            except OSError:
                pass
        if paths["AAL_B0"].exists():
            counts["parc_ready"] += 1
        if all(paths[name].exists() for name in ("FA_MIF", "MD_MIF", "AD_MIF", "RD_MIF")):
            counts["dti_ready"] += 1
        if paths["SC_ALL"].exists():
            counts["connectome_ready"] += 1
        else:
            if len(missing_connectome) < 10:
                missing_connectome.append(sid)
    return {
        "counts": counts,
        "examples_missing_bbr": missing_bbr,
        "examples_fod_input_qc_failed": fod_input_qc_failed,
        "examples_missing_connectome": missing_connectome,
    }


def print_step7_status_quick(
    cfg: Step7Config,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, object]:
    result = collect_step7_status_quick(cfg, include=include, exclude=exclude)
    counts = result["counts"]
    print("Step 7 status (quick)")
    print("---------------------")
    print(f"eligible bias series        : {counts['eligible_bias']}")
    print(f"bbr key files present       : {counts['bbr_ready']}")
    print(f"prep files present          : {counts['prep_ready']}")
    print(f"fod files present (quick)   : {counts['fod_ready']}")
    print(f"fod input qc failed         : {counts['fod_input_qc_failed']}")
    print(f"tracks done                 : {counts['tracks_ready']}")
    print(f"sift2 done                  : {counts['sift_ready']}")
    print(f"parcellation done           : {counts['parc_ready']}")
    print(f"dti maps done               : {counts['dti_ready']}")
    print(f"final connectomes done      : {counts['connectome_ready']}")
    if result["examples_missing_bbr"]:
        print("examples missing bbr : " + ", ".join(result["examples_missing_bbr"]))
    if result["examples_fod_input_qc_failed"]:
        print("examples fod input qc: " + ", ".join(result["examples_fod_input_qc_failed"]))
    if result["examples_missing_connectome"]:
        print("examples missing conn: " + ", ".join(result["examples_missing_connectome"]))
    return result


def _run_stage(
    cfg: Step7Config,
    stage: str,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, object]:
    all_sids = _discover_subjects(cfg, include=include, exclude=exclude)
    runnable, skipped_prereq, todo, done = _scan_stage_subjects(cfg, stage, all_sids)
    input_qc_failed: list[tuple[str, str]] = []
    if stage == "fod" and not cfg.force:
        remaining_todo: list[str] = []
        for sid in todo:
            paths = _subject_paths(cfg, sid, create_dirs=False)
            if _fod_input_qc_failed(paths):
                input_qc_failed.append((sid, _read_fod_input_qc_issue(paths)))
            else:
                remaining_todo.append(sid)
        todo = remaining_todo
    stage_workers = _effective_stage_workers(cfg, stage, len(todo))

    prereq_label = {
        "prep": "bbr_ready",
        "fod": "prep_ready",
        "tracks": "fod_ready",
        "post": "tracks_ready",
    }[stage]
    input_qc_fragment = f" | input_qc_failed={len(input_qc_failed)}" if stage == "fod" else ""
    worker_fragment = f" | workers={stage_workers}" if todo and stage_workers != int(cfg.batch_size) else ""
    print(
        f"Step 7 {stage}: total={len(all_sids)} | {prereq_label}={len(runnable)} | "
        f"stage_done={len(done)} | pending={len(todo)} | skipped_prereq={len(skipped_prereq)} | "
        f"batch_size={cfg.batch_size}{worker_fragment} | force={cfg.force}{input_qc_fragment}"
    )
    fod_success_counts = _empty_fod_success_counts() if stage == "fod" else None
    if stage == "fod":
        fod_success_counts = _collect_existing_fod_success_counts(cfg, done)
        print(f"Step 7 fod successes: {_format_fod_success_breakdown(fod_success_counts)}")

    if skipped_prereq:
        for sid in skipped_prereq:
            _write_summary_row(cfg, sid, stage, "skip_prereq", f"missing {prereq_label} inputs")

    status_counts: dict[str, int] = {}
    examples: dict[str, list[tuple[str, str]]] = {}
    if input_qc_failed:
        status_counts["needs_input_repair"] = len(input_qc_failed)
        examples["needs_input_repair"] = input_qc_failed[:8]

    if not todo:
        if done:
            status_counts["skip_exists"] = len(done)
        if skipped_prereq:
            status_counts["skip_prereq"] = len(skipped_prereq)
        return {
            "stage": stage,
            "status_counts": status_counts,
            "examples": examples,
        }

    worker_map = {
        "prep": _stage_prep,
        "fod": _stage_fod,
        "tracks": _stage_tracks,
        "post": _stage_post,
    }
    worker = worker_map[stage]
    runtime_lines_this_run: list[str] = []

    def _guarded_worker(run_sid: str) -> dict[str, str]:
        _wait_for_memory_headroom(cfg, stage=stage, sid=run_sid, progress_queue=progress_queue)
        with _subject_stage_lock(cfg, run_sid, stage, progress_queue=progress_queue):
            return worker(cfg, run_sid, progress_queue)

    _append_runtime_note(
        cfg.runtime_log,
        (
            f"\n=== Step7 {stage} run {time.strftime('%Y-%m-%dT%H:%M:%S')} | "
            f"total={len(all_sids)} | runnable={len(runnable)} | done={len(done)} | "
            f"pending={len(todo)} | batch_size={cfg.batch_size} | workers={stage_workers} | force={cfg.force} ==="
        ),
    )
    if not _TQDM_AVAILABLE:
        print("Progress bars unavailable: `tqdm` is not installed in this Python environment. Using text progress output instead.")

    progress_queue: Queue = Queue()
    live_panel = _LiveSidPanel(max_slots=max(1, min(stage_workers, max(1, len(todo)))), start_pos=1)
    completed_sids: set[str] = set()
    with tqdm(
        total=len(runnable),
        initial=len(done) + len(input_qc_failed),
        desc=f"step7:{stage}",
        unit="sid",
        position=0,
        dynamic_ncols=True,
    ) as bar:
        with ThreadPoolExecutor(max_workers=stage_workers) as executor:
            future_map = {
                executor.submit(_guarded_worker, sid): sid
                for sid in todo
            }
            pending = dict(future_map)
            while pending:
                drained_updates = False
                while True:
                    try:
                        sid, percent, status = progress_queue.get_nowait()
                    except Empty:
                        break
                    if sid in completed_sids:
                        continue
                    live_panel.update(sid, percent, status)
                    drained_updates = True

                ok_count = len(done) + status_counts.get("ok", 0)
                fail_count = status_counts.get("fail", 0)
                repair_count = _runtime_repair_count(status_counts)
                finished_count = _runtime_finished_count(done_count=len(done), status_counts=status_counts)
                queued = max(0, len(pending) - min(stage_workers, len(pending)))
                bar.set_postfix_str(
                    _build_stage_postfix(
                        stage=stage,
                        finished=finished_count,
                        ok=ok_count,
                        total=len(runnable),
                        fail_count=fail_count,
                        queued=queued,
                        live_panel=live_panel,
                        jobs=stage_workers,
                        repair_count=repair_count,
                        success_fragment=(
                            _format_fod_success_breakdown(fod_success_counts, compact=True)
                            if stage == "fod" and fod_success_counts is not None
                            else ""
                        ),
                    ),
                    refresh=False,
                )
                if drained_updates:
                    bar.refresh()

                done_futures, _ = wait(list(pending.keys()), timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done_futures:
                    sid = pending.pop(future)
                    result = future.result()
                    bar.update(1)
                    if _stage_done(cfg, stage, sid):
                        key = "ok"
                    elif result.get("status") == "needs_input_repair":
                        key = "needs_input_repair"
                    else:
                        key = "fail"
                    status_counts[key] = status_counts.get(key, 0) + 1
                    if key != "ok":
                        examples.setdefault(key, [])
                        if len(examples[key]) < 8:
                            examples[key].append((sid, result["error"].splitlines()[0] if result["error"] else ""))
                    completed_sids.add(sid)
                    live_panel.close(sid)
                    if key == "ok":
                        if stage == "fod" and fod_success_counts is not None:
                            _accumulate_fod_success_counts(
                                fod_success_counts,
                                _fod_success_meta_from_result(result),
                            )
                        runtime_line = _stage_success_runtime_line(result)
                        runtime_lines_this_run.append(runtime_line)
                        _append_runtime_note(cfg.runtime_log, runtime_line)
                        tqdm.write(runtime_line)
                    ok_count = len(done) + status_counts.get("ok", 0)
                    fail_count = status_counts.get("fail", 0)
                    repair_count = _runtime_repair_count(status_counts)
                    finished_count = _runtime_finished_count(done_count=len(done), status_counts=status_counts)
                    queued = max(0, len(pending) - min(stage_workers, len(pending)))
                    bar.set_postfix_str(
                        _build_stage_postfix(
                            stage=stage,
                            finished=finished_count,
                            ok=ok_count,
                            total=len(runnable),
                            fail_count=fail_count,
                            queued=queued,
                            live_panel=live_panel,
                            jobs=stage_workers,
                            repair_count=repair_count,
                            success_fragment=(
                                _format_fod_success_breakdown(fod_success_counts, compact=True)
                                if stage == "fod" and fod_success_counts is not None
                                else ""
                            ),
                        ),
                    refresh=False,
                )

    if done:
        status_counts["skip_exists"] = len(done)
    if skipped_prereq:
        status_counts["skip_prereq"] = len(skipped_prereq)
    if runtime_lines_this_run:
        print(f"\nStep 7 {stage} runtimes")
        for line in runtime_lines_this_run:
            print(line)
    return {
        "stage": stage,
        "status_counts": status_counts,
        "examples": examples,
        "runtime_log": str(cfg.runtime_log),
        "runtime_lines": runtime_lines_this_run,
    }


def run_step7_pipeline(
    cfg: Step7Config,
    stage: str = "all",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, object]:
    valid = ["prep", "fod", "tracks", "post", "all"]
    if stage not in valid:
        raise ValueError(f"stage must be one of {valid}")

    if stage == "all":
        outputs: dict[str, object] = {}
        for substage in ("prep", "fod", "tracks", "post"):
            outputs[substage] = _run_stage(cfg, substage, include=include, exclude=exclude)
        return outputs
    return _run_stage(cfg, stage, include=include, exclude=exclude)
DEFAULT_PATHS = resolve_pipeline_paths(create_layout=True)
