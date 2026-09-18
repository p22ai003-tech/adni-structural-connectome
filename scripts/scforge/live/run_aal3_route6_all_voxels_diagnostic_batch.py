#!/usr/bin/env python3
"""Run route-6 all-voxels diagnostic probes for persistent AAL3 failures.

This route is deliberately non-promotable. MRtrix's all-voxels assignment can
answer whether streamlines traverse zero-row ROIs, but it is not the endpoint
connectome definition used for production fd_sum matrices.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROBE = Path("/home/ec2-user/exp/scripts/scforge/live/run_sc_aal3_assignment_route_probe.py")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


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


def update_status(root: Path, **updates: Any) -> None:
    status = read_json(root / "status.json")
    status.update(updates)
    status["updated_utc"] = utc()
    write_json(root / "status.json", status)


def queue_subjects(route5_root: Path) -> list[str]:
    rows = read_csv(route5_root / "route_qc_decisions.csv")
    return sorted(
        {
            row["sid"]
            for row in rows
            if row.get("sid") and row.get("qc_decision") in {"TRY_ANOTHER_ROUTE", "REVIEW_ROUTE"}
        }
    )


def completed_sids(run_root: Path) -> set[str]:
    return {r.get("sid", "") for r in read_csv(run_root / "batch_results.csv") if r.get("status") == "ok"}


def run_one(
    sid: str,
    tag: str,
    route1_root: Path,
    route2_root: Path,
    mrtrix_threads: int,
    min_labels_for_connectome: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/home/ec2-user/exp:" + env.get("PYTHONPATH", "")
    env["MRTRIX_NTHREADS"] = str(mrtrix_threads)
    cmd = [
        "/home/ec2-user/exp/.venv_connectome_app/bin/python",
        str(PROBE),
        "--sid",
        sid,
        "--tag",
        f"{tag}_{sid}",
        "--route1-root",
        str(route1_root),
        "--route2-root",
        str(route2_root),
        "--variants",
        "allvoxels",
        "--min-labels-for-connectome",
        str(min_labels_for_connectome),
        "--max-parcellations",
        "1",
    ]
    proc = subprocess.run(
        cmd,
        cwd="/home/ec2-user/exp",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    probe_root = Path(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else Path("")
    decision = read_json(probe_root / "decision.json") if probe_root else {}
    best = decision.get("best") or {}
    return {
        "sid": sid,
        "status": "ok" if proc.returncode == 0 else "failed",
        "route_has_candidate": int(bool(best)),
        "rc": proc.returncode,
        "probe_root": str(probe_root),
        "tested_route": best.get("candidate", decision.get("source_label_stage", "")),
        "best_candidate": best.get("candidate", ""),
        "best_label_stage": best.get("label_stage", decision.get("source_label_stage", "")),
        "best_variant": best.get("variant", ""),
        "assignment_file": best.get("assignment_file", ""),
        "baseline_density": decision.get("baseline_density", ""),
        "best_density": decision.get("best_density", ""),
        "density_delta": decision.get("density_delta", ""),
        "baseline_zero_rows": decision.get("baseline_zero_rows", ""),
        "best_zero_rows": decision.get("best_zero_rows", ""),
        "zero_rows_delta": decision.get("zero_rows_delta", ""),
        "source_labels": decision.get("source_labels", ""),
        "source_label_survival": decision.get("source_label_survival", ""),
        "log_stdout_tail": proc.stdout[-1000:],
        "log_stderr_tail": proc.stderr[-1000:],
        "finished_utc": utc(),
    }


def diagnostic_row(result: dict[str, str]) -> dict[str, Any]:
    source_labels = inum(result.get("source_labels"), 0)
    baseline_zero = inum(result.get("baseline_zero_rows"), 9999)
    best_zero = inum(result.get("best_zero_rows"), 9999)
    baseline_density = fnum(result.get("baseline_density"), 0.0)
    best_density = fnum(result.get("best_density"), 0.0)
    route_has_candidate = inum(result.get("route_has_candidate"), 0)
    if result.get("status") != "ok":
        bucket = "probe_failed"
        reason = "diagnostic probe failed; route cannot be interpreted"
    elif not route_has_candidate:
        bucket = "label_route_unresolved"
        reason = "no usable parcellation candidate even with relaxed label threshold"
    elif source_labels < 160:
        bucket = "label_limited_diagnostic"
        reason = f"diagnostic all-voxels matrix available, but label survival is only {source_labels}/166"
    elif best_zero <= 15:
        bucket = "endpoint_assignment_limited"
        reason = "all-voxels traversal fills zero-row ROIs; endpoint assignment/termination is likely limiting the endpoint matrix"
    elif best_zero < baseline_zero:
        bucket = "partial_track_coverage"
        reason = "all-voxels traversal improves zero rows but remains above QC; existing tracks partially traverse missing ROIs"
    else:
        bucket = "track_coverage_limited"
        reason = "all-voxels traversal does not reduce zero rows enough; existing tracks likely do not cover these ROIs"
    return {
        "sid": result.get("sid", ""),
        "status": result.get("status", ""),
        "probe_root": result.get("probe_root", ""),
        "tested_route": result.get("tested_route", ""),
        "best_candidate": result.get("best_candidate", ""),
        "best_variant": result.get("best_variant", ""),
        "source_labels": source_labels,
        "source_label_survival": result.get("source_label_survival", ""),
        "baseline_density": baseline_density,
        "diagnostic_density": best_density,
        "density_delta": fnum(result.get("density_delta"), 0.0),
        "baseline_zero_rows": baseline_zero,
        "diagnostic_zero_rows": best_zero,
        "zero_rows_delta": inum(result.get("zero_rows_delta"), 9999),
        "assignment_file": result.get("assignment_file", ""),
        "diagnostic_bucket": bucket,
        "qc_decision": "TRY_ANOTHER_ROUTE",
        "qc_reasons": "diagnostic only; do not promote all-voxels matrices. " + reason,
        "final_qc_status": "not_solved_try_selective_registration_or_track_rescue",
        "finished_utc": result.get("finished_utc", ""),
    }


def score_diagnostics(run_root: Path, results: list[dict[str, Any]]) -> None:
    rows = [diagnostic_row(row) for row in results]
    write_csv(run_root / "route_qc_decisions.csv", rows)
    buckets: dict[str, int] = {}
    for row in rows:
        buckets[row.get("diagnostic_bucket", "")] = buckets.get(row.get("diagnostic_bucket", ""), 0) + 1
    write_json(
        run_root / "route_qc_summary.json",
        {
            "generated_utc": utc(),
            "run_root": str(run_root),
            "rows": len(rows),
            "decisions": {"TRY_ANOTHER_ROUTE": len(rows)},
            "diagnostic_buckets": buckets,
            "non_promotable_reason": "all-voxels assignment is traversal-based, not endpoint structural connectivity",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, required=True)
    parser.add_argument("--route2-root", type=Path, required=True)
    parser.add_argument("--route5-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mrtrix-threads", type=int, default=2)
    parser.add_argument("--min-labels-for-connectome", type=int, default=1)
    args = parser.parse_args()

    run_root = args.route1_root.parent / f"{args.route1_root.name}_route6_all_voxels_diagnostic"
    run_root.mkdir(parents=True, exist_ok=True)
    tag = run_root.name
    subjects = queue_subjects(args.route5_root)
    done = completed_sids(run_root)
    pending = [sid for sid in subjects if sid not in done]
    results: list[dict[str, Any]] = list(read_csv(run_root / "batch_results.csv"))
    write_json(
        run_root / "run_manifest.json",
        {
            "mode": "route6_all_voxels_diagnostic_existing_tracks",
            "route1_root": str(args.route1_root),
            "route2_root": str(args.route2_root),
            "route5_root": str(args.route5_root),
            "root": str(run_root),
            "tag": tag,
            "n_subjects": len(subjects),
            "workers": args.workers,
            "mrtrix_threads": args.mrtrix_threads,
            "assignment_variant": "allvoxels",
            "min_labels_for_connectome": args.min_labels_for_connectome,
            "started_utc": utc(),
            "production_policy": "diagnostic_only_never_promote",
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
    score_diagnostics(run_root, results)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(run_one, sid, tag, args.route1_root, args.route2_root, args.mrtrix_threads, args.min_labels_for_connectome): sid
            for sid in pending
        }
        update_status(run_root, active=pending[: args.workers])
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {"sid": sid, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_utc": utc()}
            results = [r for r in results if r.get("sid") != sid] + [row]
            write_csv(run_root / "batch_results.csv", results)
            score_diagnostics(run_root, results)
            completed = sum(1 for r in results if r.get("status") == "ok")
            failed = sum(1 for r in results if r.get("status") == "failed")
            remaining = [s for s in pending if s not in {r.get("sid") for r in results}]
            update_status(run_root, completed=completed, failed=failed, active=remaining[: args.workers], latest=row)

    update_status(
        run_root,
        phase="complete",
        completed=sum(1 for r in results if r.get("status") == "ok"),
        failed=sum(1 for r in results if r.get("status") == "failed"),
        active=[],
        completed_utc=utc(),
    )
    score_diagnostics(run_root, results)
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
