#!/usr/bin/env python3
"""Run Route 9: retry assignment policies on existing Route 8 AAL3 parcellations.

Route 8 can produce a high-label AAL3-to-B0 parcellation but still lose the
subject when a long forward-search tck2connectome call stalls or is interrupted.
Route 9 does not rerun FNIRT or tractography. It reuses Route 8 parcellations
and existing production 3M tracks/SIFT2 weights, then tests cheaper assignment
policies first with per-variant failure isolation.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live.run_aal3_route8_full_t1_fnirt_batch import (  # noqa: E402
    PROBE_PARENT as ROUTE8_PROBE_PARENT,
    RUN_PARENT,
    SCORER,
    TCK2CONNECTOME,
    assignment_args,
    choose_best_matrix,
    empty_failed_matrix_row,
    env_for_mrtrix,
    fnum,
    matrix_stats,
    mrtrix_thread_args,
    production_baseline,
    row_from_best,
    run_with_timeout,
)
from scripts.scforge.live.run_sc_aal3_source_contract_probe import (  # noqa: E402
    DERIV,
    aal3_valid_labels,
    discover_inputs,
    label_stats,
    write_csv,
    write_json,
)

PROBE_PARENT = DERIV / "qc" / "sc_matrix_qc" / "aal3_route9_route8_assignment_retry_probe"
ROUTE9_SCHEMA = "route8_parcellation_assignment_retry_v1"
MIN_LABELS_FOR_RETRY = 160
DEFAULT_TIMEOUT_SEC = 60 * 60
STATUS_LOCK = threading.Lock()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
    with STATUS_LOCK:
        status = read_json(root / "status.json")
        status.update(updates)
        status["updated_utc"] = utc()
        write_json(root / "status.json", status)


def route8_probe_root(route8_root: Path, sid: str) -> Path:
    return ROUTE8_PROBE_PARENT / f"{route8_root.name}_{sid}"


def parcellation_for_stage(probe_root: Path, stage: str) -> Path:
    parc = probe_root / "parc" / "route8_fnirt_full_t1"
    table = {
        "route8_full_t1_inverse_bbr_parcellation": parc / "route8_fnirt_AAL3_b0_bbr_inverse.nii.gz",
        "route8_full_t1_samegrid_parcellation": parc / "route8_fnirt_AAL3_t1full.nii.gz",
        "route8_full_t1_bbr_parcellation": parc / "route8_fnirt_AAL3_b0_bbr.nii.gz",
    }
    return table.get(stage, Path(""))


def route8_parcellation_candidates(probe_root: Path, valid_labels: list[int], min_labels: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    priority = {
        "route8_full_t1_inverse_bbr_parcellation": 3,
        "route8_full_t1_bbr_parcellation": 2,
        "route8_full_t1_samegrid_parcellation": 1,
    }
    for row in read_csv(probe_root / "label_survival_summary.csv"):
        stage = row.get("stage", "")
        if not stage.startswith("route8_"):
            continue
        path = parcellation_for_stage(probe_root, stage)
        if not path.exists() or path.stat().st_size <= 0:
            continue
        stats = label_stats(path, valid_labels)
        labels = int(fnum(stats.get("n_labels"), 0))
        if labels < min_labels:
            continue
        out.append(
            {
                "candidate": f"route9_from_{stage}",
                "label_stage": f"route9_from_{stage}",
                "source_stage": stage,
                "path": path,
                "labels": labels,
                "label_stats": stats,
                "priority": priority.get(stage, 0),
            }
        )
    out.sort(key=lambda row: (int(row["labels"]), int(row["priority"])), reverse=True)
    return out


def queue_subjects(route8_root: Path, include_interim: bool) -> list[str]:
    queued: set[str] = set()
    for row in read_csv(route8_root / "batch_results.csv"):
        sid = row.get("sid", "")
        if sid and row.get("status") == "failed":
            queued.add(sid)
    allowed = {"TRY_ANOTHER_ROUTE", "REVIEW_ROUTE"}
    if include_interim:
        allowed.add("INTERIM_REVIEW_PROMOTION_PENDING_VISUAL")
    for row in read_csv(route8_root / "route_qc_decisions.csv"):
        sid = row.get("sid", "")
        if sid and row.get("qc_decision") in allowed:
            queued.add(sid)
    return sorted(queued)


def finished_sids(run_root: Path, retry_failed: bool) -> set[str]:
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


def run_one(
    *,
    sid: str,
    tag: str,
    route8_root: Path,
    variants: list[str],
    max_parcellations: int,
    min_labels: int,
    mrtrix_threads: int,
    timeout_sec: int,
) -> dict[str, Any]:
    env = env_for_mrtrix(mrtrix_threads)
    probe_root = PROBE_PARENT / f"{tag}_{sid}"
    probe_root.mkdir(parents=True, exist_ok=True)
    cached = read_json(probe_root / "decision.json")
    if cached.get("route9_schema") == ROUTE9_SCHEMA:
        return cached.get("batch_row", {})

    update_status(probe_root, sid=sid, tag=f"{tag}_{sid}", phase="starting")
    inputs = discover_inputs(sid)
    missing = [key for key in ("tracks", "weights", "production_fd_sum") if not inputs.get(key)]
    if missing:
        raise FileNotFoundError(f"Missing Route9 inputs for {sid}: {missing}")

    valid_labels = aal3_valid_labels()
    label_rows, matrix_rows, baseline_fd = production_baseline(inputs, valid_labels)
    r8_probe = route8_probe_root(route8_root, sid)
    parcellations = route8_parcellation_candidates(r8_probe, valid_labels, min_labels)[:max_parcellations]
    for cand in parcellations:
        label_rows.append({"stage": cand["label_stage"], **cand["label_stats"], "route8_source_stage": cand["source_stage"]})
    write_csv(probe_root / "label_survival_summary.csv", label_rows)
    write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)

    if not parcellations:
        batch_row = row_from_best(
            sid,
            "ok",
            probe_root,
            {},
            baseline_fd,
            label_rows,
            {
                "route9_schema": ROUTE9_SCHEMA,
                "route9_reason": f"no Route8 parcellation with >= {min_labels}/166 labels",
            },
        )
        write_json(probe_root / "decision.json", {"route9_schema": ROUTE9_SCHEMA, "decision": "AAL3_ROUTE9_NO_HIGH_LABEL_ROUTE8_PARCELLATION", "best": {}, "batch_row": batch_row})
        update_status(probe_root, phase="complete", completed_utc=utc(), decision="AAL3_ROUTE9_NO_HIGH_LABEL_ROUTE8_PARCELLATION")
        return batch_row

    conn = probe_root / "connectomes"
    logs = probe_root / "logs"
    conn.mkdir(parents=True, exist_ok=True)
    total = len(parcellations) * len(variants)
    done = 0
    for cand in parcellations:
        for variant in variants:
            update_status(
                probe_root,
                phase="connectome_candidates",
                active_parcellation=cand["label_stage"],
                active_variant=variant,
                connectome_jobs_done=done,
                connectome_jobs_total=total,
                connectome_timeout_sec=timeout_sec,
            )
            out_csv = conn / f"SC_AAL3_{sid}_{cand['label_stage']}__{variant}__fd_sum.csv"
            assign_csv = conn / f"assignments_{cand['label_stage']}__{variant}__fd_sum.csv"
            stats = matrix_stats(out_csv, valid_labels)
            if int(fnum(stats.get("read_ok"), 0)) != 1 or not assign_csv.exists() or assign_csv.stat().st_size <= 0:
                cmd: list[str | Path] = [
                    TCK2CONNECTOME,
                    inputs["tracks"],
                    cand["path"],
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
                    inputs["weights"],
                ]
                try:
                    run_with_timeout(cmd, logs / f"tck2connectome_{cand['label_stage']}__{variant}__fd_sum.log", env, timeout_sec)
                    stats = matrix_stats(out_csv, valid_labels)
                except Exception as exc:
                    matrix_rows.append(
                        empty_failed_matrix_row(
                            label_stage=cand["label_stage"],
                            variant=variant,
                            assign_csv=assign_csv,
                            error=exc,
                        )
                    )
                    write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)
                    done += 1
                    continue

            matrix_rows.append(
                {
                    "candidate": cand["label_stage"],
                    "label_stage": cand["label_stage"],
                    "route8_source_stage": cand["source_stage"],
                    "variant": variant,
                    "metric": "fd_sum",
                    "assignment_file": str(assign_csv),
                    **stats,
                }
            )
            write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)
            done += 1

    best = choose_best_matrix(matrix_rows, label_rows, baseline_fd)
    batch_row = row_from_best(
        sid,
        "ok",
        probe_root,
        best,
        baseline_fd,
        label_rows,
        {"route9_schema": ROUTE9_SCHEMA, "route9_action": "route8_parcellation_assignment_retry"},
    )
    decision = "AAL3_ROUTE9_ASSIGNMENT_RETRY_CANDIDATE" if best else "AAL3_ROUTE9_ASSIGNMENT_RETRY_NO_CANDIDATE"
    write_json(probe_root / "decision.json", {"route9_schema": ROUTE9_SCHEMA, "decision": decision, "best": best, "batch_row": batch_row})
    update_status(probe_root, phase="complete", completed_utc=utc(), decision=decision, connectome_jobs_done=done, connectome_jobs_total=total)
    return batch_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, required=True)
    parser.add_argument("--route8-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mrtrix-threads", type=int, default=3)
    parser.add_argument("--variants", nargs="*", default=["endvox", "radial8", "radial16", "forward40"])
    parser.add_argument("--max-parcellations", type=int, default=2)
    parser.add_argument("--min-labels", type=int, default=MIN_LABELS_FOR_RETRY)
    parser.add_argument("--connectome-timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--include-interim", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    route8_root = args.route8_root or RUN_PARENT / f"{args.route1_root.name}_route8_full_t1_fnirt"
    run_root = RUN_PARENT / f"{args.route1_root.name}_route9_route8_assignment_retry"
    run_root.mkdir(parents=True, exist_ok=True)
    tag = run_root.name

    subjects = queue_subjects(route8_root, include_interim=args.include_interim)
    if args.limit > 0:
        subjects = subjects[: args.limit]
    done = finished_sids(run_root, retry_failed=args.retry_failed)
    pending = [sid for sid in subjects if sid not in done]
    results: list[dict[str, Any]] = list(read_csv(run_root / "batch_results.csv"))
    write_json(
        run_root / "run_manifest.json",
        {
            "mode": "route9_route8_parcellation_assignment_retry",
            "route1_root": str(args.route1_root),
            "route8_root": str(route8_root),
            "root": str(run_root),
            "tag": tag,
            "n_subjects": len(subjects),
            "workers": args.workers,
            "mrtrix_threads": args.mrtrix_threads,
            "variants": args.variants,
            "max_parcellations": args.max_parcellations,
            "min_labels": args.min_labels,
            "connectome_timeout_sec": args.connectome_timeout_sec,
            "include_interim": bool(args.include_interim),
            "retry_failed": bool(args.retry_failed),
            "started_utc": utc(),
        },
    )
    update_status(run_root, phase="running", total=len(subjects), completed=len(done), failed=sum(1 for r in results if r.get("status") == "failed"), active=[])

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                run_one,
                sid=sid,
                tag=tag,
                route8_root=route8_root,
                variants=args.variants,
                max_parcellations=args.max_parcellations,
                min_labels=args.min_labels,
                mrtrix_threads=args.mrtrix_threads,
                timeout_sec=args.connectome_timeout_sec,
            ): sid
            for sid in pending
        }
        update_status(run_root, active=pending[: args.workers])
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {"sid": sid, "status": "failed", "route9_schema": ROUTE9_SCHEMA, "error": f"{type(exc).__name__}: {exc}", "finished_utc": utc()}
            results = [r for r in results if r.get("sid") != sid] + [row]
            write_csv(run_root / "batch_results.csv", results)
            completed = sum(1 for r in results if r.get("status") == "ok")
            failed = sum(1 for r in results if r.get("status") == "failed")
            remaining = [s for s in pending if s not in {r.get("sid") for r in results}]
            update_status(run_root, completed=completed, failed=failed, active=remaining[: args.workers], latest=row)
            subprocess.run(["/home/ec2-user/exp/.venv_connectome_app/bin/python", str(SCORER), "--run-root", str(run_root)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    update_status(run_root, phase="complete", completed=sum(1 for r in results if r.get("status") == "ok"), failed=sum(1 for r in results if r.get("status") == "failed"), active=[], completed_utc=utc())
    subprocess.run(["/home/ec2-user/exp/.venv_connectome_app/bin/python", str(SCORER), "--run-root", str(run_root)], cwd=ROOT, stdout=(run_root / "score_stdout.log").open("w"), stderr=(run_root / "score_stderr.log").open("w"), check=False)
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
