#!/usr/bin/env python3
"""Run route-7 selective AAL3 rescue probes for unresolved AD subjects.

Route 7 separates two failure modes discovered by route 6:

* label_limited_diagnostic: try a nonlinear T1->MNI FNIRT route, invert it, then
  project AAL3 back to native T1 and B0 before using the existing 3M tracks.
* track_coverage_limited: keep the best high-label AAL3-B0 parcellation, but
  generate scratch-only dynamic-seeded ACT tracks plus scratch SIFT2 weights.

This script never overwrites production derivatives. Promotion still requires
the standard numeric scorer, visual overlay QC, and the promotion script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live.run_sc_aal3_assignment_route_probe import (
    assignment_args as assignment_route_args,
    collect_candidate_parcellations,
)
from scripts.scforge.live.run_sc_aal3_bbr_route_probe import bbr_inputs
from scripts.scforge.live.run_sc_aal3_source_contract_probe import (
    AAL3,
    DERIV,
    FLIRT,
    FSLDIR,
    MNI_BRAIN,
    MRTRIX,
    TCK2CONNECTOME,
    aal3_valid_labels,
    discover_inputs,
    label_stats,
    matrix_stats,
    mrtrix_thread_args,
    run,
    write_csv,
    write_json,
)

SCORER = Path("/home/ec2-user/exp/scripts/scforge/live/score_aal3_force_connectome_qc.py")
LEDGER = Path("/home/ec2-user/exp/scripts/scforge/live/summarize_aal3_route_ledger.py")
RUN_PARENT = Path("/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch")
PROBE_PARENT = DERIV / "qc" / "sc_matrix_qc" / "aal3_route7_selective_rescue_probe"
MIN_PROMOTABLE_LABELS = 160

APPLYWARP = FSLDIR / "bin" / "applywarp"
FNIRT = FSLDIR / "bin" / "fnirt"
INVWARP = FSLDIR / "bin" / "invwarp"
TCKGEN = MRTRIX / "tckgen"
TCKSIFT2 = MRTRIX / "tcksift2"
TCKEDIT = MRTRIX / "tckedit"


def assignment_args(name: str) -> list[str]:
    """Route 7 mixes forward-search and deep-assignment variants."""
    if name == "forward40":
        return ["-assignment_forward_search", "40"]
    if name == "forward80":
        return ["-assignment_forward_search", "80"]
    return assignment_route_args(name)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def connectome_timeout_sec(env: dict[str, str]) -> int:
    for key in ("AAL3_CONNECTOME_TIMEOUT_SEC", "ROUTE7_CONNECTOME_TIMEOUT_SEC"):
        value = str(env.get(key, "")).strip()
        if value:
            try:
                return max(60, int(float(value)))
            except ValueError:
                pass
    return 45 * 60


def track_step_timeout_sec(env: dict[str, str], step: str) -> int:
    if step == "tckgen":
        keys = ("AAL3_TCKGEN_TIMEOUT_SEC", "ROUTE7_TCKGEN_TIMEOUT_SEC")
        default = 4 * 60 * 60
    else:
        keys = ("AAL3_TCKSIFT2_TIMEOUT_SEC", "ROUTE7_TCKSIFT2_TIMEOUT_SEC")
        default = 4 * 60 * 60
    for key in keys:
        value = str(env.get(key, "")).strip()
        if value:
            try:
                return max(60, int(float(value)))
            except ValueError:
                pass
    return default


def run_track_cmd(cmd: list[str | Path], log_path: Path, env: dict[str, str], step: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    str_cmd = [str(x) for x in cmd]
    timeout_sec = track_step_timeout_sec(env, step)
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"[{utc()}] $ {' '.join(str_cmd)}\n")
        handle.write(f"[{utc()}] timeout_sec={timeout_sec}\n")
        handle.flush()
        try:
            proc = subprocess.run(
                str_cmd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            handle.write(f"[{utc()}] timeout after {timeout_sec}s\n\n")
            raise TimeoutError(f"{step} timed out after {timeout_sec}s; see {log_path}") from exc
        handle.write(f"[{utc()}] rc={proc.returncode}\n\n")
    if proc.returncode != 0:
        raise RuntimeError(f"{step} failed rc={proc.returncode}; see {log_path}")


def chunked_tckgen_enabled(env: dict[str, str], select_streamlines: int) -> bool:
    flag = str(env.get("AAL3_CHUNKED_TCKGEN", "1")).strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    chunk_size = inum(env.get("AAL3_TCKGEN_CHUNK_STREAMLINES", 1_000_000), 1_000_000)
    return select_streamlines > max(1, chunk_size)


def chunked_tckgen_command(
    *,
    base_cmd: list[str | Path],
    output_tracks: Path,
    select_streamlines: int,
    mode: str,
    logs: Path,
    track_root: Path,
    env: dict[str, str],
    mrtrix_threads: int,
) -> None:
    """Generate a large tractogram in smaller deterministic chunks, then merge.

    This directly addresses the recurring failure mode where one massive tckgen
    process stalls or exits before writing a usable tractogram. The final merged
    file preserves the same downstream contract as the old single-shot command.
    """
    chunk_size = max(1, inum(env.get("AAL3_TCKGEN_CHUNK_STREAMLINES", 1_000_000), 1_000_000))
    max_parallel = max(1, inum(env.get("AAL3_TCKGEN_CHUNK_PARALLEL", 1), 1))
    chunk_dir = track_root / "_chunks" / output_tracks.stem
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[int, int, Path]] = []
    remaining = int(select_streamlines)
    idx = 0
    while remaining > 0:
        idx += 1
        take = min(chunk_size, remaining)
        chunks.append((idx, take, chunk_dir / f"chunk_{idx:03d}_{take // 1000}k.tck"))
        remaining -= take

    manifest = chunk_dir / "chunk_manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                f"generated_utc={utc()}",
                f"mode={mode}",
                f"select_streamlines={select_streamlines}",
                f"chunk_streamlines={chunk_size}",
                f"chunk_count={len(chunks)}",
                f"chunk_parallel={max_parallel}",
                f"mrtrix_threads_per_chunk={mrtrix_threads}",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    def run_one(item: tuple[int, int, Path]) -> Path:
        chunk_idx, chunk_select, chunk_path = item
        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            return chunk_path
        seed = 100_000 + (chunk_idx * 9973)
        cmd = list(base_cmd)
        cmd[2] = chunk_path
        select_pos = cmd.index("-select")
        cmd[select_pos + 1] = str(chunk_select)
        cmd.extend(["-seed", str(seed)])
        run_track_cmd(cmd, logs / f"tckgen_{mode}_chunk{chunk_idx:03d}_{chunk_select // 1000}k.log", env, "tckgen")
        return chunk_path

    if max_parallel == 1:
        chunk_paths = [run_one(chunk) for chunk in chunks]
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        chunk_paths = []
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {pool.submit(run_one, chunk): chunk for chunk in chunks}
            for future in as_completed(futures):
                chunk_paths.append(future.result())
        chunk_paths.sort(key=lambda path: path.name)

    missing = [path for path in chunk_paths if not path.exists() or path.stat().st_size <= 0]
    if missing:
        raise RuntimeError("chunked tckgen failed to produce chunks: " + ", ".join(str(path) for path in missing[:5]))

    run_track_cmd(
        [TCKEDIT, *chunk_paths, output_tracks, "-nthreads", str(max(1, mrtrix_threads)), "-force"],
        logs / f"tckedit_{mode}_{select_streamlines // 1000}k_chunks.log",
        env,
        "tckgen",
    )


def run_connectome_cmd(cmd: list[str | Path], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    str_cmd = [str(x) for x in cmd]
    timeout_sec = connectome_timeout_sec(env)
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"[{utc()}] $ {' '.join(str_cmd)}\n")
        handle.write(f"[{utc()}] timeout_sec={timeout_sec}\n")
        handle.flush()
        try:
            proc = subprocess.run(
                str_cmd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            handle.write(f"[{utc()}] timeout after {timeout_sec}s\n\n")
            raise TimeoutError(f"command timed out after {timeout_sec}s; see {log_path}") from exc
        handle.write(f"[{utc()}] rc={proc.returncode}\n\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}; see {log_path}")


def matrix_meets_early_stop(stats: dict[str, Any], matrix_rows: list[dict[str, Any]]) -> bool:
    prod = next((row for row in matrix_rows if row.get("candidate") == "production_AAL3"), {})
    baseline_density = fnum(prod.get("valid_density"), 0.0)
    baseline_zero = inum(prod.get("valid_zero_rows"), 9999)
    return (
        int(fnum(stats.get("read_ok"), 0)) == 1
        and fnum(stats.get("finite_fraction"), 0.0) >= 1.0
        and fnum(stats.get("symmetry_max_abs"), 1.0) == 0.0
        and fnum(stats.get("diagonal_abs_sum"), 1.0) == 0.0
        and fnum(stats.get("valid_density"), 0.0) >= 0.10
        and fnum(stats.get("valid_density"), 0.0) >= baseline_density
        and inum(stats.get("valid_zero_rows"), 9999) <= 15
        and inum(stats.get("valid_zero_rows"), 9999) <= baseline_zero
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_status(root: Path, **updates: Any) -> None:
    status = read_json(root / "status.json")
    status.update(updates)
    status["updated_utc"] = utc()
    write_json(root / "status.json", status)


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def env_flag(env: dict[str, str], key: str, default: bool) -> bool:
    value = str(env.get(key, "")).strip().lower()
    if not value:
        return default
    return value not in {"0", "false", "no", "off"}


def endpoint_preflight_for_tracks(
    *,
    sid: str,
    probe_root: Path,
    label_stage: str,
    parc_path: Path,
    tracks: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    """Sample track endpoints before SIFT2/connectome and reject clear collapse.

    This is intentionally cheaper than a full candidate matrix. It catches the
    dominant failure mode seen in sparse AAL3 runs: endpoints exist, but they
    land in too few labels or miss the selected parcellation contract.
    """
    if not env_flag(env, "AAL3_ENDPOINT_PREFLIGHT", True):
        return {
            "sid": sid,
            "label_stage": label_stage,
            "tracks_path": str(tracks),
            "endpoint_preflight_decision": "disabled",
            "endpoint_preflight_pass": 1,
        }

    from scripts.scforge.live.diagnose_aal3_endpoint_space_contract import diagnose_subject

    max_streamlines = max(1000, inum(env.get("AAL3_ENDPOINT_PREFLIGHT_MAX_STREAMLINES", 50_000), 50_000))
    min_unique = max(1, inum(env.get("AAL3_ENDPOINT_PREFLIGHT_MIN_UNIQUE_LABELS", 30), 30))
    max_top3 = fnum(env.get("AAL3_ENDPOINT_PREFLIGHT_MAX_TOP3_FRACTION", 0.60), 0.60)
    min_on_label = fnum(env.get("AAL3_ENDPOINT_PREFLIGHT_MIN_ON_LABEL_FRACTION", 0.10), 0.10)

    update_status(
        probe_root,
        phase="endpoint_preflight",
        endpoint_preflight_label_stage=label_stage,
        endpoint_preflight_max_streamlines=max_streamlines,
    )
    row: dict[str, Any] = diagnose_subject(
        sid=sid,
        parc_path=parc_path,
        tracks_path=tracks,
        label=label_stage,
        max_streamlines=max_streamlines,
        compute_distance=False,
    )
    unique = fnum(row.get("unique_endpoint_labels"), 0)
    top3 = fnum(row.get("top3_endpoint_label_fraction"), 1)
    on_label = fnum(row.get("endpoint_on_label_fraction"), 0)
    failure_class = str(row.get("failure_class") or "")
    hard_fail_classes = {
        "no_track_endpoints_sampled",
        "endpoint_world_bbox_outside_parcellation_image",
        "endpoints_inside_image_but_not_on_any_label",
        "endpoint_label_collapse",
        "low_endpoint_label_overlap",
    }
    pass_gate = (
        failure_class not in hard_fail_classes
        and unique >= min_unique
        and top3 <= max_top3
        and on_label >= min_on_label
    )
    row.update(
        {
            "label_stage": label_stage,
            "endpoint_preflight_min_unique_labels": min_unique,
            "endpoint_preflight_max_top3_fraction": max_top3,
            "endpoint_preflight_min_on_label_fraction": min_on_label,
            "endpoint_preflight_pass": int(pass_gate),
            "endpoint_preflight_decision": "pass" if pass_gate else "fail_endpoint_contract",
        }
    )
    preflight_path = probe_root / "endpoint_preflight_summary.csv"
    rows = [r for r in read_csv(preflight_path) if not (r.get("label_stage") == label_stage and r.get("tracks_path") == str(tracks))]
    rows.append(row)
    write_csv(preflight_path, rows)
    update_status(
        probe_root,
        endpoint_preflight_decision=row["endpoint_preflight_decision"],
        endpoint_preflight_failure_class=failure_class,
        endpoint_preflight_unique_labels=unique,
        endpoint_preflight_top3_fraction=top3,
        endpoint_preflight_on_label_fraction=on_label,
    )
    return row


def queue_subjects(route6_root: Path) -> list[dict[str, str]]:
    rows = [
        row
        for row in read_csv(route6_root / "route_qc_decisions.csv")
        if row.get("sid") and row.get("qc_decision") in {"TRY_ANOTHER_ROUTE", "REVIEW_ROUTE"}
    ]
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        sid = row["sid"]
        bucket = row.get("diagnostic_bucket", "")
        if bucket == "track_coverage_limited":
            action = "dynamic_track_rescue"
        elif bucket == "label_limited_diagnostic":
            action = "fnirt_label_rescue"
        else:
            action = "document_unresolved"
        out[sid] = {**row, "route7_action": action}
    return [out[sid] for sid in sorted(out)]


def completed_sids(run_root: Path, retry_failed: bool = False) -> set[str]:
    done: set[str] = set()
    for row in read_csv(run_root / "batch_results.csv"):
        sid = row.get("sid", "")
        if not sid:
            continue
        if row.get("status") == "ok":
            done.add(sid)
        elif row.get("status") == "failed" and not retry_failed:
            done.add(sid)
    return done


def refresh_ledger(route1_root: Path) -> None:
    if not (route1_root / "subjects.csv").exists():
        return
    subprocess.run(
        ["/home/ec2-user/exp/.venv_connectome_app/bin/python", str(LEDGER), "--run-root", str(route1_root)],
        cwd="/home/ec2-user/exp",
        stdout=(route1_root / "route_subject_ledger_stdout.log").open("w"),
        stderr=(route1_root / "route_subject_ledger_stderr.log").open("w"),
        check=False,
    )


def env_for_mrtrix(mrtrix_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["FSLDIR"] = str(FSLDIR)
    env["PATH"] = f"{MRTRIX}:{FSLDIR / 'bin'}:" + env.get("PATH", "")
    env["MRTRIX_NTHREADS"] = str(mrtrix_threads)
    return env


def fod_paths(sid: str) -> dict[str, Path]:
    fod = Path("/data/derivatives/fod") / sid
    wmfod = fod / "wmfod.mif"
    if not wmfod.exists():
        wmfod = fod / "wmfod_final.mif"
    return {
        "fod_root": fod,
        "wmfod": wmfod,
        "act": fod / "5tt_b0_fixed.mif" if (fod / "5tt_b0_fixed.mif").exists() else fod / "5tt_b0.mif",
        "gmwmi": fod / "gmwmi.mif",
        "deconv_mode": fod / "deconv_mode.txt",
    }


def deconv_mode(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii", errors="ignore").strip()
    except Exception:
        return ""


def production_baseline(inputs: dict[str, Any], valid_labels: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    label_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    if inputs.get("production_aal_b0"):
        label_rows.append({"stage": "production_AAL_b0", **label_stats(inputs["production_aal_b0"], valid_labels)})
    baseline_fd = {"candidate": "production_AAL3", "variant": "production", "metric": "fd_sum", **matrix_stats(inputs["production_fd_sum"], valid_labels)}
    matrix_rows.append(baseline_fd)
    return label_rows, matrix_rows, baseline_fd


def choose_best_matrix(matrix_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]], baseline_fd: dict[str, Any]) -> dict[str, Any]:
    labels = {r.get("stage"): fnum(r.get("n_labels"), 0) for r in label_rows}
    base_density = fnum(baseline_fd.get("valid_density"), 0.0)
    base_zero = fnum(baseline_fd.get("valid_zero_rows"), 9999.0)
    candidates = [
        row
        for row in matrix_rows
        if row.get("candidate") != "production_AAL3"
        and row.get("metric") == "fd_sum"
        and int(fnum(row.get("read_ok"), 0)) == 1
    ]

    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        density = fnum(row.get("valid_density"), -1.0)
        zero = fnum(row.get("valid_zero_rows"), 9999.0)
        safe = 1.0 if density >= base_density and zero <= base_zero else 0.0
        return (safe, labels.get(row.get("label_stage"), 0), -zero, density)

    candidates.sort(key=key, reverse=True)
    return candidates[0] if candidates else {}


def row_from_best(sid: str, status: str, probe_root: Path, best: dict[str, Any], baseline_fd: dict[str, Any], label_rows: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    label_stage = str(best.get("label_stage") or "")
    best_label = next((r for r in label_rows if r.get("stage") == label_stage), {})
    base_density = fnum(baseline_fd.get("valid_density"), 0.0)
    best_density = fnum(best.get("valid_density"), 0.0)
    base_zero = inum(baseline_fd.get("valid_zero_rows"), 9999)
    best_zero = inum(best.get("valid_zero_rows"), 9999)
    row: dict[str, Any] = {
        "sid": sid,
        "status": status,
        "route_has_candidate": int(bool(best)),
        "rc": 0 if status == "ok" else 1,
        "probe_root": str(probe_root),
        "tested_route": best.get("candidate", label_stage),
        "best_candidate": best.get("candidate", ""),
        "best_label_stage": label_stage,
        "best_variant": best.get("variant", ""),
        "assignment_file": best.get("assignment_file", ""),
        "baseline_density": base_density,
        "best_density": best_density,
        "density_delta": best_density - base_density,
        "baseline_zero_rows": base_zero,
        "best_zero_rows": best_zero,
        "zero_rows_delta": best_zero - base_zero,
        "source_labels": best_label.get("n_labels", ""),
        "source_label_survival": best_label.get("expected_label_survival_fraction", ""),
        "finished_utc": utc(),
    }
    if extra:
        row.update(extra)
    return row


def run_connectomes(
    *,
    sid: str,
    probe_root: Path,
    label_stage: str,
    parc_path: Path,
    tracks: Path,
    weights: Path,
    variants: list[str],
    label_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
    valid_labels: list[int],
    env: dict[str, str],
) -> list[dict[str, Any]]:
    conn = probe_root / "connectomes"
    logs = probe_root / "logs"
    conn.mkdir(parents=True, exist_ok=True)
    total = len(variants)
    for idx, variant in enumerate(variants, start=1):
        update_status(
            probe_root,
            phase="connectome_candidates",
            active_variant=variant,
            connectome_jobs_done=idx - 1,
            connectome_jobs_total=total,
        )
        out_csv = conn / f"SC_AAL3_{sid}_{label_stage}__{variant}__fd_sum.csv"
        assign_csv = conn / f"assignments_{label_stage}__{variant}__fd_sum.csv"
        stats = matrix_stats(out_csv, valid_labels)
        if int(fnum(stats.get("read_ok"), 0)) != 1 or not assign_csv.exists() or assign_csv.stat().st_size <= 0:
            cmd: list[str | Path] = [
                TCK2CONNECTOME,
                tracks,
                parc_path,
                out_csv,
                "-symmetric",
                "-zero_diagonal",
                *mrtrix_thread_args(env),
                *assignment_args(variant),
                "-out_assignments",
                assign_csv,
                "-stat_edge",
                "sum",
                "-tck_weights_in",
                weights,
            ]
            try:
                run_connectome_cmd(cmd, logs / f"tck2connectome_{label_stage}__{variant}__fd_sum.log", env)
                stats = matrix_stats(out_csv, valid_labels)
            except Exception as exc:
                stats = matrix_stats(out_csv, valid_labels)
                stats["error"] = f"{type(exc).__name__}: {exc}"
        row = {
            "candidate": label_stage,
            "label_stage": label_stage,
            "variant": variant,
            "metric": "fd_sum",
            "assignment_file": str(assign_csv),
            **stats,
        }
        matrix_rows.append(row)
        write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)
        if matrix_meets_early_stop(row, matrix_rows):
            update_status(
                probe_root,
                connectome_jobs_done=idx,
                connectome_jobs_total=total,
                decision=f"early_stop_numeric_candidate_{variant}",
            )
            break
    update_status(probe_root, connectome_jobs_done=total, connectome_jobs_total=total)
    return matrix_rows


def run_fnirt_label_rescue(
    sid: str,
    row: dict[str, str],
    tag: str,
    route1_root: Path,
    route2_root: Path,
    variants: list[str],
    mrtrix_threads: int,
) -> dict[str, Any]:
    env = env_for_mrtrix(mrtrix_threads)
    probe_root = PROBE_PARENT / f"{tag}_{sid}"
    probe_root.mkdir(parents=True, exist_ok=True)
    if (probe_root / "decision.json").exists():
        return read_json(probe_root / "decision.json").get("batch_row", {})
    update_status(probe_root, sid=sid, tag=f"{tag}_{sid}", phase="starting", route7_action=row.get("route7_action"))

    inputs = discover_inputs(sid)
    missing = [key for key in ("tracks", "weights", "production_fd_sum") if not inputs.get(key)]
    bbr = bbr_inputs(sid)
    missing += [f"bbr_{key}" for key, path in bbr.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing route7 FNIRT inputs for {sid}: {missing}")

    valid_labels = aal3_valid_labels()
    label_rows, matrix_rows, baseline_fd = production_baseline(inputs, valid_labels)
    logs = probe_root / "logs"
    parc = probe_root / "parc" / "route7_fnirt_t1brain"
    parc.mkdir(parents=True, exist_ok=True)

    t1_to_mni = parc / "route7_t1brain2mni_affine.mat"
    t1_to_mni_warp = parc / "route7_t1brain2mni_warpcoef.nii.gz"
    mni_to_t1_warp = parc / "route7_mni2t1brain_warpcoef.nii.gz"
    aal_t1 = parc / "route7_fnirt_AAL3_t1brain.nii.gz"
    aal_b0 = parc / "route7_fnirt_AAL3_b0.nii.gz"
    if not aal_b0.exists() or aal_b0.stat().st_size <= 0:
        update_status(probe_root, phase="fnirt_affine")
        run([FLIRT, "-in", bbr["t1brain"], "-ref", MNI_BRAIN, "-omat", t1_to_mni, "-dof", "12"], logs / "t1brain_to_mni_affine.log", env)
        update_status(probe_root, phase="fnirt_nonlinear")
        run([FNIRT, f"--in={bbr['t1brain']}", f"--ref={MNI_BRAIN}", f"--aff={t1_to_mni}", f"--cout={t1_to_mni_warp}"], logs / "t1brain_to_mni_fnirt.log", env)
        update_status(probe_root, phase="invert_warp")
        run([INVWARP, f"--warp={t1_to_mni_warp}", f"--ref={bbr['t1brain']}", f"--out={mni_to_t1_warp}"], logs / "mni_to_t1brain_invwarp.log", env)
        update_status(probe_root, phase="apply_aal3_warp")
        run([APPLYWARP, f"--in={AAL3}", f"--ref={bbr['t1brain']}", f"--warp={mni_to_t1_warp}", f"--out={aal_t1}", "--interp=nn"], logs / "aal3_mni_to_t1brain_fnirt.log", env)
        run(
            [
                FLIRT,
                "-in",
                aal_t1,
                "-ref",
                bbr["b0"],
                "-applyxfm",
                "-init",
                bbr["t12b0"],
                "-interp",
                "nearestneighbour",
                "-datatype",
                "int",
                "-out",
                aal_b0,
            ],
            logs / "aal3_t1brain_to_b0_bbr.log",
            env,
        )

    label_stage = "route7_fnirt_t1brain_parcellation"
    fnirt_label_stats = label_stats(aal_b0, valid_labels)
    label_rows.append({"stage": label_stage, **fnirt_label_stats})
    write_csv(probe_root / "label_survival_summary.csv", label_rows)
    write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)
    n_labels = inum(fnirt_label_stats.get("n_labels"), 0)
    if n_labels < MIN_PROMOTABLE_LABELS:
        batch_row = row_from_best(
            sid,
            "ok",
            probe_root,
            {},
            baseline_fd,
            label_rows,
            {
                "route7_action": row.get("route7_action", ""),
                "route7_reason": f"fnirt label survival {n_labels}/166 below promotable gate {MIN_PROMOTABLE_LABELS}/166",
            },
        )
        write_json(
            probe_root / "decision.json",
            {
                "decision": "AAL3_ROUTE7_FNIRT_LOW_LABEL_SURVIVAL",
                "best": {},
                "batch_row": batch_row,
            },
        )
        update_status(
            probe_root,
            phase="complete",
            completed_utc=utc(),
            decision="AAL3_ROUTE7_FNIRT_LOW_LABEL_SURVIVAL",
        )
        return batch_row
    run_connectomes(
        sid=sid,
        probe_root=probe_root,
        label_stage=label_stage,
        parc_path=aal_b0,
        tracks=inputs["tracks"],
        weights=inputs["weights"],
        variants=variants,
        label_rows=label_rows,
        matrix_rows=matrix_rows,
        valid_labels=valid_labels,
        env=env,
    )
    best = choose_best_matrix(matrix_rows, label_rows, baseline_fd)
    batch_row = row_from_best(sid, "ok", probe_root, best, baseline_fd, label_rows, {"route7_action": row.get("route7_action", "")})
    write_json(probe_root / "decision.json", {"decision": "AAL3_ROUTE7_FNIRT_LABEL_RESCUE", "best": best, "batch_row": batch_row})
    update_status(probe_root, phase="complete", completed_utc=utc(), decision="AAL3_ROUTE7_FNIRT_LABEL_RESCUE")
    return batch_row


def run_dynamic_track_rescue(
    sid: str,
    row: dict[str, str],
    tag: str,
    route1_root: Path,
    route2_root: Path,
    variants: list[str],
    track_modes: list[str],
    select_streamlines: int,
    mrtrix_threads: int,
) -> dict[str, Any]:
    env = env_for_mrtrix(mrtrix_threads)
    probe_root = PROBE_PARENT / f"{tag}_{sid}"
    probe_root.mkdir(parents=True, exist_ok=True)
    if (probe_root / "decision.json").exists():
        return read_json(probe_root / "decision.json").get("batch_row", {})
    update_status(probe_root, sid=sid, tag=f"{tag}_{sid}", phase="starting", route7_action=row.get("route7_action"))

    inputs = discover_inputs(sid)
    missing = [key for key in ("production_fd_sum",) if not inputs.get(key)]
    fpaths = fod_paths(sid)
    missing += [key for key, path in fpaths.items() if key != "deconv_mode" and not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing route7 track inputs for {sid}: {missing}")

    valid_labels = aal3_valid_labels()
    label_rows, matrix_rows, baseline_fd = production_baseline(inputs, valid_labels)
    parc_candidates = collect_candidate_parcellations(sid, route1_root, route2_root, valid_labels)
    parc_candidates = [cand for cand in parc_candidates if int(cand.get("labels") or 0) >= 160]
    if not parc_candidates:
        batch_row = row_from_best(sid, "ok", probe_root, {}, baseline_fd, label_rows, {"route7_action": row.get("route7_action", ""), "route7_reason": "no high-label parcellation for track rescue"})
        write_csv(probe_root / "label_survival_summary.csv", label_rows)
        write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)
        write_json(probe_root / "decision.json", {"decision": "AAL3_ROUTE7_NO_HIGH_LABEL_PARCELLATION", "best": {}, "batch_row": batch_row})
        update_status(probe_root, phase="complete", completed_utc=utc(), decision="AAL3_ROUTE7_NO_HIGH_LABEL_PARCELLATION")
        return batch_row

    parc_candidate = parc_candidates[0]
    base_stage = str(parc_candidate["candidate"])
    label_rows.append({"stage": base_stage, **parc_candidate["label_stats"]})
    write_csv(probe_root / "label_survival_summary.csv", label_rows)
    write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)

    track_root = probe_root / "tracks"
    logs = probe_root / "logs"
    track_root.mkdir(exist_ok=True)
    fd_scale_gm_default = deconv_mode(fpaths["deconv_mode"]) == "single_shell"
    track_mode_errors: list[str] = []
    successful_track_modes: list[str] = []
    for mode in track_modes:
        if mode not in {"dynamic", "gmwmi"}:
            raise ValueError(f"Unsupported route7 track mode: {mode}")
        try:
            track_stage = f"route7_{mode}_tracks_{base_stage}"
            tracks = track_root / f"{sid}_{mode}_{select_streamlines // 1000}k.tck"
            weights = track_root / f"{sid}_{mode}_{select_streamlines // 1000}k_sift2_weights.txt"
            mu = track_root / f"{sid}_{mode}_{select_streamlines // 1000}k_mu.txt"
            if not tracks.exists() or tracks.stat().st_size <= 0:
                update_status(probe_root, phase="tckgen", active_track_mode=mode, select_streamlines=select_streamlines)
                cmd: list[str | Path] = [
                    TCKGEN,
                    fpaths["wmfod"],
                    tracks,
                    "-act",
                    fpaths["act"],
                    "-backtrack",
                    "-cutoff",
                    "0.04",
                    "-maxlength",
                    "250",
                    "-minlength",
                    "10",
                    "-select",
                    str(select_streamlines),
                    "-nthreads",
                    str(mrtrix_threads),
                    "-force",
                ]
                if mode == "dynamic":
                    cmd.extend(["-seed_dynamic", fpaths["wmfod"]])
                else:
                    cmd.extend(["-seed_gmwmi", fpaths["gmwmi"], "-crop_at_gmwmi"])
                if chunked_tckgen_enabled(env, select_streamlines):
                    update_status(
                        probe_root,
                        phase="tckgen_chunked",
                        active_track_mode=mode,
                        select_streamlines=select_streamlines,
                        tckgen_chunk_streamlines=inum(env.get("AAL3_TCKGEN_CHUNK_STREAMLINES", 1_000_000), 1_000_000),
                    )
                    chunked_tckgen_command(
                        base_cmd=cmd,
                        output_tracks=tracks,
                        select_streamlines=select_streamlines,
                        mode=mode,
                        logs=logs,
                        track_root=track_root,
                        env=env,
                        mrtrix_threads=mrtrix_threads,
                    )
                else:
                    run_track_cmd(cmd, logs / f"tckgen_{mode}_{select_streamlines // 1000}k.log", env, "tckgen")
            preflight = endpoint_preflight_for_tracks(
                sid=sid,
                probe_root=probe_root,
                label_stage=track_stage,
                parc_path=Path(parc_candidate["path"]),
                tracks=tracks,
                env=env,
            )
            if int(fnum(preflight.get("endpoint_preflight_pass"), 0)) != 1:
                raise RuntimeError(
                    "endpoint preflight failed before SIFT2/connectome: "
                    f"{preflight.get('failure_class')} | "
                    f"unique_labels={preflight.get('unique_endpoint_labels')} | "
                    f"top3={preflight.get('top3_endpoint_label_fraction')} | "
                    f"on_label={preflight.get('endpoint_on_label_fraction')}"
                )
            if not weights.exists() or weights.stat().st_size <= 0:
                update_status(probe_root, phase="tcksift2", active_track_mode=mode)
                attempts = [fd_scale_gm_default, not fd_scale_gm_default]
                last_error: Exception | None = None
                for idx, fd_scale_gm in enumerate(attempts, start=1):
                    cmd = [
                        TCKSIFT2,
                        tracks,
                        fpaths["wmfod"],
                        weights,
                        "-act",
                        fpaths["act"],
                        "-out_mu",
                        mu,
                        "-nthreads",
                        str(max(1, mrtrix_threads)),
                    ]
                    if fd_scale_gm:
                        cmd.append("-fd_scale_gm")
                    try:
                        run_track_cmd(cmd, logs / f"tcksift2_{mode}_attempt{idx}.log", env, "tcksift2")
                        last_error = None
                        break
                    except Exception as exc:  # retry with opposite fd_scale_gm policy
                        last_error = exc
                        if weights.exists():
                            weights.unlink()
                if last_error is not None:
                    raise last_error
            label_rows.append({"stage": track_stage, **parc_candidate["label_stats"]})
            write_csv(probe_root / "label_survival_summary.csv", label_rows)
            run_connectomes(
                sid=sid,
                probe_root=probe_root,
                label_stage=track_stage,
                parc_path=Path(parc_candidate["path"]),
                tracks=tracks,
                weights=weights,
                variants=variants,
                label_rows=label_rows,
                matrix_rows=matrix_rows,
                valid_labels=valid_labels,
                env=env,
            )
            successful_track_modes.append(mode)
        except Exception as exc:
            message = f"{mode}: {type(exc).__name__}: {exc}"
            track_mode_errors.append(message)
            update_status(probe_root, phase="track_mode_failed", active_track_mode=mode, track_mode_errors=track_mode_errors)
            continue

    if not successful_track_modes:
        raise RuntimeError("all route7 track modes failed: " + "; ".join(track_mode_errors))

    best = choose_best_matrix(matrix_rows, label_rows, baseline_fd)
    batch_row = row_from_best(
        sid,
        "ok",
        probe_root,
        best,
        baseline_fd,
        label_rows,
        {
            "route7_action": row.get("route7_action", ""),
            "select_streamlines": select_streamlines,
            "production_grade_tracks": int(select_streamlines == 3_000_000),
            "successful_track_modes": ",".join(successful_track_modes),
            "track_mode_errors": " | ".join(track_mode_errors),
        },
    )
    write_json(probe_root / "decision.json", {"decision": "AAL3_ROUTE7_DYNAMIC_TRACK_RESCUE", "best": best, "batch_row": batch_row})
    update_status(probe_root, phase="complete", completed_utc=utc(), decision="AAL3_ROUTE7_DYNAMIC_TRACK_RESCUE")
    return batch_row


def run_one(
    subject_row: dict[str, str],
    tag: str,
    route1_root: Path,
    route2_root: Path,
    variants: list[str],
    track_modes: list[str],
    select_streamlines: int,
    mrtrix_threads: int,
) -> dict[str, Any]:
    sid = subject_row["sid"]
    action = subject_row.get("route7_action", "")
    if action == "fnirt_label_rescue":
        return run_fnirt_label_rescue(sid, subject_row, tag, route1_root, route2_root, variants, mrtrix_threads)
    if action == "dynamic_track_rescue":
        return run_dynamic_track_rescue(sid, subject_row, tag, route1_root, route2_root, variants, track_modes, select_streamlines, mrtrix_threads)
    return {
        "sid": sid,
        "status": "ok",
        "route_has_candidate": 0,
        "rc": 0,
        "probe_root": "",
        "tested_route": "",
        "best_candidate": "",
        "best_variant": "",
        "route7_action": action or "document_unresolved",
        "route7_reason": "route6 did not classify this subject into a runnable route7 rescue bucket",
        "finished_utc": utc(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, required=True)
    parser.add_argument("--route2-root", type=Path, required=True)
    parser.add_argument("--route6-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mrtrix-threads", type=int, default=4)
    parser.add_argument("--select-streamlines", type=int, default=3_000_000)
    parser.add_argument("--track-modes", nargs="*", default=["dynamic"])
    parser.add_argument("--variants", nargs="*", default=["forward80", "forward120", "radial8"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    for tool in (FLIRT, FNIRT, INVWARP, APPLYWARP, TCKGEN, TCKSIFT2, TCK2CONNECTOME, AAL3, MNI_BRAIN):
        if not Path(tool).exists():
            raise FileNotFoundError(tool)

    run_root = RUN_PARENT / f"{args.route1_root.name}_route7_selective_track_rescue"
    run_root.mkdir(parents=True, exist_ok=True)
    tag = run_root.name
    subjects = queue_subjects(args.route6_root)
    if args.limit > 0:
        subjects = subjects[: args.limit]
    done = completed_sids(run_root, retry_failed=args.retry_failed)
    pending = [row for row in subjects if row["sid"] not in done]
    results: list[dict[str, Any]] = list(read_csv(run_root / "batch_results.csv"))
    write_json(
        run_root / "run_manifest.json",
        {
            "mode": "route7_selective_registration_or_track_rescue",
            "route1_root": str(args.route1_root),
            "route2_root": str(args.route2_root),
            "route6_root": str(args.route6_root),
            "root": str(run_root),
            "tag": tag,
            "n_subjects": len(subjects),
            "workers": args.workers,
            "mrtrix_threads": args.mrtrix_threads,
            "select_streamlines": args.select_streamlines,
            "track_modes": args.track_modes,
            "variants": args.variants,
            "retry_failed": args.retry_failed,
            "subject_actions": {row["sid"]: row.get("route7_action", "") for row in subjects},
            "started_utc": utc(),
        },
    )
    update_status(
        run_root,
        phase="running",
        total=len(subjects),
        completed=len(done),
        failed=sum(1 for r in results if r.get("status") == "failed"),
        active=[],
    )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                run_one,
                row,
                tag,
                args.route1_root,
                args.route2_root,
                args.variants,
                args.track_modes,
                args.select_streamlines,
                args.mrtrix_threads,
            ): row
            for row in pending
        }
        update_status(run_root, active=[f"{row['sid']}:{row.get('route7_action','')}" for row in pending[: args.workers]])
        for fut in as_completed(futures):
            subject_row = futures[fut]
            sid = subject_row["sid"]
            try:
                row = fut.result()
            except Exception as exc:
                row = {
                    "sid": sid,
                    "status": "failed",
                    "route7_action": subject_row.get("route7_action", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_utc": utc(),
                }
            results = [r for r in results if r.get("sid") != sid] + [row]
            write_csv(run_root / "batch_results.csv", results)
            completed = sum(1 for r in results if r.get("status") == "ok")
            failed = sum(1 for r in results if r.get("status") == "failed")
            remaining = [r for r in pending if r["sid"] not in {x.get("sid") for x in results}]
            update_status(
                run_root,
                completed=completed,
                failed=failed,
                active=[f"{r['sid']}:{r.get('route7_action','')}" for r in remaining[: args.workers]],
                latest=row,
            )
            subprocess.run(
                ["/home/ec2-user/exp/.venv_connectome_app/bin/python", str(SCORER), "--run-root", str(run_root)],
                cwd="/home/ec2-user/exp",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            refresh_ledger(args.route1_root)

    update_status(
        run_root,
        phase="complete",
        completed=sum(1 for r in results if r.get("status") == "ok"),
        failed=sum(1 for r in results if r.get("status") == "failed"),
        active=[],
        completed_utc=utc(),
    )
    subprocess.run(
        ["/home/ec2-user/exp/.venv_connectome_app/bin/python", str(SCORER), "--run-root", str(run_root)],
        cwd="/home/ec2-user/exp",
        stdout=(run_root / "score_stdout.log").open("w"),
        stderr=(run_root / "score_stderr.log").open("w"),
        check=False,
    )
    refresh_ledger(args.route1_root)
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
