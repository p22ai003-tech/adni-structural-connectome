#!/usr/bin/env python3
"""Scratch-only SC-Forge v1 all-subject density improvement batch.

For each completed EC2 fd_sum connectome, this runner replays the empirically
better v1 AAL3 source-contract route in scratch space, generates full candidate
matrices for the best assignment variant, compares them to the current
production matrix, and writes a best-output manifest.

Production connectomes are never overwritten.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
LIVE_ROOT = Path(__file__).resolve().parent
DERIV = ROOT / "data" / "derivatives"
QC_ROOT = DERIV / "qc" / "sc_matrix_qc"
OUT_PARENT = QC_ROOT
PROBE_PARENT = QC_ROOT / "aal3_source_contract_probe"
LATEST_FILE = QC_ROOT / "scforge_v1_density_latest.txt"

sys.path.insert(0, str(ROOT / "scforge"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LIVE_ROOT))

import run_sc_aal3_source_contract_probe as aal3_probe  # noqa: E402
import run_sc_final_spatial_contract_closeout as closeout  # noqa: E402
from scforge.assignment_diversity_qc import classify_assignment  # noqa: E402


VARIANTS = ["default", "radial8", "forward40"]
REQUIRED_METRICS = ["count", "fd_sum", "len_mean", "fa_mean", "md_mean", "rd_mean", "ad_mean"]
IO_LOCK = threading.RLock()

SMOKE_SUBJECTS = [
    "003_S_0908_I1249292",
    "127_S_5028_I401540",
    "014_S_4401_I1556672",
]

CANARY_SUBJECTS = [
    "003_S_0908_I1249292",
    "014_S_4401_I1556672",
    "041_S_5141_I893581",
    "041_S_4427_I1243839",
    "021_S_7092_I1597668",
    "031_S_4021_I1253150",
    "033_S_7114_I11063036",
    "003_S_6257_I974346",
    "127_S_5028_I401540",
    "168_S_6874_I1667523",
]


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        val = float(value)
        if math.isnan(val):
            return default
        return val
    except Exception:
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except Exception:
        return default


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with IO_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with IO_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            if fields:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, row: dict[str, Any]) -> None:
    with IO_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        fields = list(row.keys())
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def append_log(root: Path, text: str) -> None:
    with IO_LOCK:
        with (root / "findings_log.md").open("a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n\n")


def update_status(root: Path, **updates: Any) -> None:
    with IO_LOCK:
        status = read_json(root / "status.json")
        status.update(updates)
        status["last_update_utc"] = utc()
        write_json(root / "status.json", status)


def run_cmd(cmd: list[str | Path], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    str_cmd = [str(x) for x in cmd]
    run_root = log_path.parent.parent if log_path.parent.name == "logs" else log_path.parent
    with IO_LOCK:
        with (run_root / "commands.log").open("a", encoding="utf-8", errors="replace") as command_log:
            command_log.write(f"[{utc()}] $ {' '.join(str_cmd)}\n")
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"[{utc()}] $ {' '.join(str_cmd)}\n")
        handle.flush()
        proc = subprocess.run(str_cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
        handle.write(f"[{utc()}] rc={proc.returncode}\n\n")
    with IO_LOCK:
        with (run_root / "commands.log").open("a", encoding="utf-8", errors="replace") as command_log:
            command_log.write(f"[{utc()}] rc={proc.returncode}\n\n")
    return int(proc.returncode)


def latest_completed_qc() -> Path:
    candidates = sorted(QC_ROOT.glob("completed_connectomes_after_gap_*/sc_matrix_integrity_subjects.csv"))
    if candidates:
        return candidates[-1]
    fallback = QC_ROOT / "sc_matrix_integrity_subjects.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No completed-connectome QC table found under sc_matrix_qc")


def production_matrix_path(sid: str, metric: str) -> Path:
    return DERIV / "connectomes" / f"SC_AAL_{sid}_{metric}.csv"


def load_subject_universe(qc_path: Path) -> list[dict[str, Any]]:
    rows = read_csv(qc_path)
    fd_rows = [r for r in rows if r.get("weight") == "fd_sum" and not str(r.get("read_error", "")).strip()]
    out: list[dict[str, Any]] = []
    for row in fd_rows:
        sid = row.get("sid", "")
        matrix_path = Path(row.get("matrix_path", ""))
        if not sid or not matrix_path.exists():
            continue
        out.append(
            {
                "sid": sid,
                "sample_id": sid,
                "subject_id": row.get("subject_id", ""),
                "group": row.get("group", ""),
                "present_density": fnum(row.get("density"), math.nan),
                "present_valid_zero_rows": inum(row.get("n_valid_zero_rows"), 9999),
                "present_matrix_path": str(matrix_path),
                "present_n_rows": inum(row.get("n_rows"), 0),
                "present_n_cols": inum(row.get("n_cols"), 0),
            }
        )
    return sorted(out, key=lambda r: str(r["sid"]))


def best_probe_candidate(probe_root: Path) -> dict[str, Any]:
    decision = read_json(probe_root / "decision.json")
    best = decision.get("best")
    if isinstance(best, dict) and best:
        return best
    candidates = [
        row
        for row in read_csv(probe_root / "matrix_candidate_summary.csv")
        if row.get("candidate") == "AAL3_source_contract"
        and row.get("metric") == "fd_sum"
        and inum(row.get("read_ok"), 0) == 1
    ]
    candidates.sort(key=lambda r: (-fnum(r.get("valid_density"), -1), inum(r.get("valid_zero_rows"), 9999)))
    return candidates[0] if candidates else {}


def full_rows_by_metric(full_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("metric")): row for row in full_rows}


def assignment_qc_for_fd(full_rows: list[dict[str, Any]]) -> dict[str, Any]:
    fd = full_rows_by_metric(full_rows).get("fd_sum", {})
    assignment_path = Path(str(fd.get("assignments", "")))
    if not assignment_path.exists():
        return {
            "assignment_qc_status": "MISSING",
            "assignment_collapse_class": "MISSING_ASSIGNMENTS",
            "unique_assigned_nodes": 0,
            "top5_endpoint_fraction": 1.0,
        }
    qc = classify_assignment(assignment_path).to_dict()
    return {
        "assignment_qc_status": qc.get("status", ""),
        "assignment_collapse_class": qc.get("collapse_class", ""),
        "unique_assigned_nodes": qc.get("unique_assigned_nodes", 0),
        "top5_endpoint_fraction": qc.get("top5_endpoint_fraction", 1.0),
        "assignment_path": str(assignment_path),
    }


def candidate_decision(
    baseline: dict[str, Any],
    probe_decision: dict[str, Any],
    full_rows: list[dict[str, Any]],
    assignment_qc: dict[str, Any],
) -> dict[str, Any]:
    by_metric = full_rows_by_metric(full_rows)
    fd = by_metric.get("fd_sum", {})
    missing_required = [
        metric
        for metric in REQUIRED_METRICS
        if metric not in by_metric or inum(by_metric[metric].get("read_ok"), 0) != 1
    ]
    candidate_density = fnum(fd.get("valid_density"), -1)
    candidate_zero = inum(fd.get("valid_zero_rows"), 9999)
    baseline_density = fnum(probe_decision.get("baseline_fd_sum", {}).get("valid_density"), math.nan)
    if math.isnan(baseline_density):
        baseline_density = fnum(baseline.get("present_density"), -1)
    baseline_zero = inum(probe_decision.get("baseline_fd_sum", {}).get("valid_zero_rows"), 9999)
    if baseline_zero == 9999:
        baseline_zero = inum(baseline.get("present_valid_zero_rows"), 9999)

    finite_ok = all(fnum(by_metric.get(m, {}).get("finite_fraction"), 0) >= 1.0 for m in REQUIRED_METRICS if m in by_metric)
    symmetry_ok = all(fnum(by_metric.get(m, {}).get("symmetry_max_abs"), 999) <= 1e-8 for m in REQUIRED_METRICS if m in by_metric)
    diagonal_ok = all(fnum(by_metric.get(m, {}).get("diagonal_abs_sum"), 999) <= 1e-8 for m in REQUIRED_METRICS if m in by_metric)
    density_improved = candidate_density > baseline_density
    zero_not_worse = candidate_zero <= baseline_zero
    collapse_class = str(assignment_qc.get("assignment_collapse_class", ""))
    unique_nodes = inum(assignment_qc.get("unique_assigned_nodes"), 0)
    top5 = fnum(assignment_qc.get("top5_endpoint_fraction"), 1.0)
    severe_collapse = collapse_class in {"CATASTROPHIC_COLLAPSE", "SEVERE_COLLAPSE"} or unique_nodes < 20 or top5 > 0.90
    label_failed = str(probe_decision.get("decision", "")).endswith("LABEL_SURVIVAL_FAILED")

    retain = (
        not missing_required
        and finite_ok
        and symmetry_ok
        and diagonal_ok
        and density_improved
        and zero_not_worse
        and not severe_collapse
        and not label_failed
    )
    publication_pass = (
        retain
        and candidate_density >= 0.15
        and candidate_zero == 0
        and unique_nodes >= 140
        and top5 <= 0.35
    )
    candidate_pass = retain and candidate_density >= 0.15 and candidate_zero <= 5 and unique_nodes >= 120

    if publication_pass:
        decision = "RETAIN_CANDIDATE_PUBLICATION_PASS"
    elif retain:
        decision = "RETAIN_CANDIDATE_IMPROVED"
    elif label_failed or severe_collapse:
        decision = "QUARANTINE_UPSTREAM_REPROCESS"
    else:
        decision = "RETAIN_CURRENT"

    return {
        "decision": decision,
        "retain_candidate": int(retain),
        "candidate_pass": int(candidate_pass),
        "publication_pass": int(publication_pass),
        "missing_required_metrics": ";".join(missing_required),
        "baseline_valid_density": baseline_density,
        "candidate_valid_density": candidate_density,
        "density_delta": candidate_density - baseline_density,
        "baseline_valid_zero_rows": baseline_zero,
        "candidate_valid_zero_rows": candidate_zero,
        "zero_rows_delta": candidate_zero - baseline_zero,
        "finite_ok": int(finite_ok),
        "symmetry_ok": int(symmetry_ok),
        "zero_diagonal_ok": int(diagonal_ok),
        "density_improved": int(density_improved),
        "zero_rows_not_worse": int(zero_not_worse),
        "severe_assignment_collapse": int(severe_collapse),
        "label_survival_failed": int(label_failed),
        **assignment_qc,
    }


def write_manifests(root: Path, comparison_rows: list[dict[str, Any]]) -> None:
    best_rows: list[dict[str, Any]] = []
    retained_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []

    for row in comparison_rows:
        sid = str(row.get("sid"))
        variant = str(row.get("best_variant") or "")
        retain = inum(row.get("retain_candidate"), 0) == 1
        decision = str(row.get("decision", ""))
        candidate_dir = root / "full_candidate_matrices" / sid / variant if variant else Path("")
        for metric in REQUIRED_METRICS:
            baseline = production_matrix_path(sid, metric)
            candidate = candidate_dir / f"SC_AAL3_{sid}_{variant}_{metric}.csv" if variant else Path("")
            selected = candidate if retain and candidate.exists() else baseline
            selected_source = "candidate" if retain and candidate.exists() else "current"
            item = {
                "sid": sid,
                "group": row.get("group", ""),
                "metric": metric,
                "decision": decision,
                "selected_source": selected_source,
                "selected_path": str(selected),
                "baseline_path": str(baseline),
                "candidate_path": str(candidate) if variant else "",
            }
            best_rows.append(item)
            if selected_source == "candidate":
                retained_rows.append(item)
            else:
                fallback_rows.append(item)
        if decision == "QUARANTINE_UPSTREAM_REPROCESS":
            quarantine_rows.append(row)

    write_csv(root / "best_connectome_manifest.csv", best_rows)
    write_csv(root / "retained_candidate_manifest.csv", retained_rows)
    write_csv(root / "fallback_current_manifest.csv", fallback_rows)
    write_csv(root / "quarantine_manifest.csv", quarantine_rows)


def phase_subjects(phase: str, universe: list[dict[str, Any]], explicit_subjects: list[str] | None) -> list[dict[str, Any]]:
    by_sid = {str(row["sid"]): row for row in universe}
    if explicit_subjects:
        return [by_sid[sid] for sid in explicit_subjects if sid in by_sid]
    if phase == "smoke":
        return [by_sid[sid] for sid in SMOKE_SUBJECTS if sid in by_sid]
    if phase == "canary":
        return [by_sid[sid] for sid in CANARY_SUBJECTS if sid in by_sid]
    return universe


def completed_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in rows:
        phase = str(row.get("phase") or "")
        sid = str(row.get("sid") or "")
        if phase and sid and row.get("decision"):
            keys.add((phase, sid))
    return keys


def run_phase_subjects(
    run_root: Path,
    env: dict[str, str],
    phase_name: str,
    subjects: list[dict[str, Any]],
    workers: int,
) -> Any:
    if not subjects:
        return
    workers = max(1, int(workers))
    if workers == 1:
        for index, subject_row in enumerate(subjects, start=1):
            try:
                row = run_one_subject(run_root, env, subject_row, index, len(subjects))
            except Exception as exc:
                row = {
                    **subject_row,
                    "decision": "RETAIN_CURRENT",
                    "retain_candidate": 0,
                    "candidate_pass": 0,
                    "publication_pass": 0,
                    "error": f"{type(exc).__name__}:{exc}",
                }
                append_csv(
                    run_root / "subject_stage_log.csv",
                    {"utc": utc(), "sid": subject_row["sid"], "stage": "exception", "error": row["error"]},
                )
            row["phase"] = phase_name
            yield row
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one_subject, run_root, env, subject_row, index, len(subjects)): subject_row
            for index, subject_row in enumerate(subjects, start=1)
        }
        for future in as_completed(futures):
            subject_row = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    **subject_row,
                    "decision": "RETAIN_CURRENT",
                    "retain_candidate": 0,
                    "candidate_pass": 0,
                    "publication_pass": 0,
                    "error": f"{type(exc).__name__}:{exc}",
                }
                append_csv(
                    run_root / "subject_stage_log.csv",
                    {"utc": utc(), "sid": subject_row["sid"], "stage": "exception", "error": row["error"]},
                )
            row["phase"] = phase_name
            yield row


def run_one_subject(root: Path, env: dict[str, str], subject_row: dict[str, Any], index: int, total: int) -> dict[str, Any]:
    sid = str(subject_row["sid"])
    update_status(
        root,
        phase="subject_probe",
        current_sid=sid,
        phase_subject_index=index,
        phase_subject_total=total,
        latest_decision="running",
    )
    stage_log = {
        "utc": utc(),
        "sid": sid,
        "stage": "start",
        "baseline_density": subject_row.get("present_density"),
        "baseline_valid_zero_rows": subject_row.get("present_valid_zero_rows"),
    }
    append_csv(root / "subject_stage_log.csv", stage_log)

    probe_tag = f"{root.name}_{sid}"
    probe_root = PROBE_PARENT / probe_tag
    rc = run_cmd(
        [
            aal3_probe.FSLDIR / "bin" / "python",
            LIVE_ROOT / "run_sc_aal3_source_contract_probe.py",
            "--sid",
            sid,
            "--tag",
            probe_tag,
            "--variants",
            *VARIANTS,
        ],
        root / "logs" / f"probe_{sid}.log",
        env,
    )
    probe_decision = read_json(probe_root / "decision.json")
    best = best_probe_candidate(probe_root)
    if rc != 0 or not probe_decision:
        row = {
            **subject_row,
            "probe_tag": probe_tag,
            "probe_root": str(probe_root),
            "probe_rc": rc,
            "best_variant": "",
            "best_metric": "",
            "decision": "RETAIN_CURRENT",
            "retain_candidate": 0,
            "candidate_pass": 0,
            "publication_pass": 0,
            "error": "probe_failed_or_missing_decision",
        }
        append_csv(root / "subject_stage_log.csv", {"utc": utc(), "sid": sid, "stage": "probe_failed", "rc": rc})
        return row

    if not best:
        dec = candidate_decision(subject_row, probe_decision, [], {})
        row = {
            **subject_row,
            "probe_tag": probe_tag,
            "probe_root": str(probe_root),
            "probe_rc": rc,
            "probe_decision": probe_decision.get("decision", ""),
            "best_variant": "",
            "best_metric": "",
            **dec,
        }
        append_csv(root / "subject_stage_log.csv", {"utc": utc(), "sid": sid, "stage": "no_candidate", "decision": row["decision"]})
        return row

    variant = str(best.get("variant") or "default")
    update_status(root, phase="full_candidate_matrices", current_sid=sid, current_variant=variant)
    full_rows = closeout.generate_full_candidate_matrices(root, sid, variant, probe_root, aal3_probe.aal3_valid_labels(), env)
    append_csv(root / "subject_stage_log.csv", {"utc": utc(), "sid": sid, "stage": "full_candidate_done", "variant": variant})

    assignment_qc = assignment_qc_for_fd(full_rows)
    dec = candidate_decision(subject_row, probe_decision, full_rows, assignment_qc)
    fd = full_rows_by_metric(full_rows).get("fd_sum", {})
    row = {
        **subject_row,
        "probe_tag": probe_tag,
        "probe_root": str(probe_root),
        "probe_rc": rc,
        "probe_decision": probe_decision.get("decision", ""),
        "best_variant": variant,
        "best_metric": best.get("metric", ""),
        "candidate_fd_sum_path": fd.get("candidate_matrix", ""),
        **dec,
    }
    append_csv(
        root / "subject_stage_log.csv",
        {
            "utc": utc(),
            "sid": sid,
            "stage": "decision",
            "decision": row["decision"],
            "baseline_density": row.get("baseline_valid_density"),
            "candidate_density": row.get("candidate_valid_density"),
            "baseline_zero_rows": row.get("baseline_valid_zero_rows"),
            "candidate_zero_rows": row.get("candidate_valid_zero_rows"),
            "unique_assigned_nodes": row.get("unique_assigned_nodes"),
        },
    )
    return row


def canary_gate(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    retained = sum(inum(r.get("retain_candidate"), 0) for r in rows)
    publication = sum(inum(r.get("publication_pass"), 0) for r in rows)
    quarantined = sum(1 for r in rows if r.get("decision") == "QUARANTINE_UPSTREAM_REPROCESS")
    if retained >= 3 and quarantined >= 1:
        return True, f"canary reproduced useful v1 behavior: retained={retained}, publication={publication}, quarantine={quarantined}"
    if retained >= 3:
        return True, f"canary retained at least 3 candidates: retained={retained}, publication={publication}"
    return False, f"canary did not reproduce enough improvement: retained={retained}, publication={publication}, quarantine={quarantined}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=f"scforge_v1_density_batch_{stamp()}")
    parser.add_argument("--phase", choices=["inventory", "smoke", "canary", "full", "all"], default="all")
    parser.add_argument("--subjects", nargs="*", help="Optional explicit subject IDs")
    parser.add_argument("--limit", type=int, help="Limit subject count after selection")
    parser.add_argument("--execute", action="store_true", help="Actually run probes and scratch candidate generation")
    parser.add_argument("--skip-canary-gate", action="store_true", help="For phase=all, continue to full even if canary is weak")
    parser.add_argument("--workers", type=int, default=1, help="Number of subjects to process concurrently")
    parser.add_argument(
        "--mrtrix-threads",
        type=int,
        default=0,
        help="MRtrix -nthreads value per subject. Use 0 for MRtrix default/all threads.",
    )
    args = parser.parse_args()

    qc_path = latest_completed_qc()
    universe = load_subject_universe(qc_path)
    if args.limit:
        universe = universe[: args.limit]

    run_root = OUT_PARENT / args.tag
    run_root.mkdir(parents=True, exist_ok=True)
    for log_name in ("commands.log", "stdout.log", "stderr.log"):
        (run_root / log_name).touch(exist_ok=True)
    LATEST_FILE.write_text(str(run_root), encoding="utf-8")

    env = os.environ.copy()
    env["FSLDIR"] = str(aal3_probe.FSLDIR)
    env["PATH"] = f"{aal3_probe.MRTRIX}:{aal3_probe.FSLDIR / 'bin'}:" + env.get("PATH", "")
    if args.mrtrix_threads and args.mrtrix_threads > 0:
        env["MRTRIX_NTHREADS"] = str(args.mrtrix_threads)

    baseline_rows = [
        {
            "sample_id": row["sample_id"],
            "group": row.get("group", ""),
            "present_density": row["present_density"],
            "present_valid_zero_rows": row["present_valid_zero_rows"],
            "present_matrix_path": row["present_matrix_path"],
        }
        for row in universe
    ]
    write_csv(run_root / "baseline_density.csv", baseline_rows)
    write_json(
        run_root / "run_manifest.json",
        {
            "tag": args.tag,
            "root": str(run_root),
            "started_utc": utc(),
            "execute": bool(args.execute),
            "phase": args.phase,
            "qc_source": str(qc_path),
            "subject_universe_n": len(universe),
            "variants": VARIANTS,
            "required_metrics": REQUIRED_METRICS,
            "production_overwrite": False,
            "workers": int(args.workers),
            "mrtrix_threads": int(args.mrtrix_threads),
        },
    )
    write_json(
        run_root / "status.json",
        {
            "phase": "inventory",
            "root": str(run_root),
            "qc_source": str(qc_path),
            "total_subjects": len(universe),
            "completed_subjects": 0,
            "failed_subjects": 0,
            "retained_candidate_count": 0,
            "retained_current_count": 0,
            "quarantine_count": 0,
            "started_utc": utc(),
            "workers": int(args.workers),
            "mrtrix_threads": int(args.mrtrix_threads),
        },
    )
    append_log(
        run_root,
        "# SC-Forge v1 Density Improvement Batch\n\n"
        f"Started: {utc()}\n\n"
        f"QC source: `{qc_path}`\n\n"
        "Production matrices are not overwritten. Candidate paths are selected only through manifests.",
    )

    if args.phase == "inventory" or not args.execute:
        update_status(root=run_root, phase="inventory_complete", latest_decision="dry_run_inventory_only")
        print(run_root)
        return 0

    phases: list[tuple[str, list[dict[str, Any]]]] = []
    if args.phase == "all":
        smoke_subjects = phase_subjects("smoke", universe, None)
        smoke_seen = {row["sid"] for row in smoke_subjects}
        canary_subjects = [row for row in phase_subjects("canary", universe, None) if row["sid"] not in smoke_seen]
        phases.append(("smoke", smoke_subjects))
        phases.append(("canary", canary_subjects))
        full_subjects = [row for row in universe if row["sid"] not in set(SMOKE_SUBJECTS + CANARY_SUBJECTS)]
        phases.append(("full", full_subjects))
    else:
        phases.append((args.phase, phase_subjects(args.phase, universe, args.subjects)))

    planned_total = sum(len(subjects) for _, subjects in phases)
    comparison_rows: list[dict[str, Any]] = read_csv(run_root / "candidate_density_comparison.csv")
    already_done = completed_keys(comparison_rows)
    canary_rows: list[dict[str, Any]] = [
        row for row in comparison_rows if row.get("phase") in {"smoke", "canary"}
    ]
    if comparison_rows:
        append_log(
            run_root,
            "## Resume Detected\n\n"
            f"Existing completed rows: `{len(comparison_rows)}`. "
            "Subjects with completed phase rows will be skipped.",
        )
        write_manifests(run_root, comparison_rows)
        retained = sum(inum(r.get("retain_candidate"), 0) for r in comparison_rows)
        quarantine = sum(1 for r in comparison_rows if r.get("decision") == "QUARANTINE_UPSTREAM_REPROCESS")
        update_status(
            run_root,
            completed_subjects=len(comparison_rows),
            total_subjects=planned_total,
            retained_candidate_count=retained,
            retained_current_count=len(comparison_rows) - retained - quarantine,
            quarantine_count=quarantine,
            workers=int(args.workers),
            mrtrix_threads=int(args.mrtrix_threads),
        )
    stop_before_full = False
    for phase_name, subjects in phases:
        subjects = [row for row in subjects if (phase_name, str(row["sid"])) not in already_done]
        if stop_before_full and phase_name == "full":
            append_log(run_root, "## Full Batch Blocked\n\nCanary gate failed; full batch was not started.")
            break
        update_status(
            run_root,
            phase=phase_name,
            total_subjects=planned_total,
            phase_remaining_subjects=len(subjects),
            subject_index=0,
            workers=int(args.workers),
            mrtrix_threads=int(args.mrtrix_threads),
        )
        append_log(
            run_root,
            f"## Phase: {phase_name}\n\n"
            f"Remaining subjects: `{len(subjects)}`. "
            f"Workers: `{args.workers}`. MRtrix threads per subject: `{args.mrtrix_threads or 'default'}`.",
        )
        for row in run_phase_subjects(run_root, env, phase_name, subjects, args.workers):
            comparison_rows.append(row)
            already_done.add((phase_name, str(row.get("sid", ""))))
            if phase_name in {"smoke", "canary"}:
                canary_rows.append(row)
            write_csv(run_root / "candidate_density_comparison.csv", comparison_rows)
            write_manifests(run_root, comparison_rows)
            retained = sum(inum(r.get("retain_candidate"), 0) for r in comparison_rows)
            quarantine = sum(1 for r in comparison_rows if r.get("decision") == "QUARANTINE_UPSTREAM_REPROCESS")
            update_status(
                run_root,
                completed_subjects=len(comparison_rows),
                failed_subjects=sum(1 for r in comparison_rows if r.get("error")),
                retained_candidate_count=retained,
                retained_current_count=len(comparison_rows) - retained - quarantine,
                quarantine_count=quarantine,
                latest_sid=row.get("sid"),
                latest_decision=row.get("decision"),
                latest_baseline_density=row.get("baseline_valid_density", row.get("present_density")),
                latest_candidate_density=row.get("candidate_valid_density", ""),
                latest_density_delta=row.get("density_delta", ""),
                latest_baseline_zero_rows=row.get("baseline_valid_zero_rows", row.get("present_valid_zero_rows")),
                latest_candidate_zero_rows=row.get("candidate_valid_zero_rows", ""),
            )
        if phase_name == "canary" and args.phase == "all":
            ok, reason = canary_gate(canary_rows)
            append_log(run_root, f"## Canary Gate\n\n{reason}\n\nProceed to full batch: `{ok or args.skip_canary_gate}`")
            if not ok and not args.skip_canary_gate:
                stop_before_full = True

    write_manifests(run_root, comparison_rows)
    retained = sum(inum(r.get("retain_candidate"), 0) for r in comparison_rows)
    candidate_pass = sum(inum(r.get("candidate_pass"), 0) for r in comparison_rows)
    publication_pass = sum(inum(r.get("publication_pass"), 0) for r in comparison_rows)
    quarantine = sum(1 for r in comparison_rows if r.get("decision") == "QUARANTINE_UPSTREAM_REPROCESS")
    write_csv(
        run_root / "scforge_v1_canary_summary.csv",
        [
            {
                "processed_subjects": len(comparison_rows),
                "retained_candidate": retained,
                "retained_current": len(comparison_rows) - retained - quarantine,
                "candidate_pass": candidate_pass,
                "publication_pass": publication_pass,
                "quarantine_upstream_reprocess": quarantine,
                "full_batch_started": int(any(r.get("phase") == "full" for r in comparison_rows)),
            }
        ],
    )
    append_log(
        run_root,
        "## Final Summary\n\n"
        f"- Processed subjects: `{len(comparison_rows)}`\n"
        f"- Retained candidates: `{retained}`\n"
        f"- Candidate-pass subjects: `{candidate_pass}`\n"
        f"- Publication-pass subjects: `{publication_pass}`\n"
        f"- Quarantine/upstream-reprocess subjects: `{quarantine}`\n",
    )
    update_status(run_root, phase="complete", completed_utc=utc(), latest_decision="complete")
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
