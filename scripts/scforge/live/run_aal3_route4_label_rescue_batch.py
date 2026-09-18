#!/usr/bin/env python3
"""Run route-4 label-preserving dilation probes for R3 failures."""

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

PROBE = Path("/home/ec2-user/exp/scripts/scforge/live/run_sc_aal3_label_rescue_probe.py")
SCORER = Path("/home/ec2-user/exp/scripts/scforge/live/score_aal3_force_connectome_qc.py")


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


def update_status(root: Path, **updates: Any) -> None:
    status = read_json(root / "status.json")
    status.update(updates)
    status["updated_utc"] = utc()
    write_json(root / "status.json", status)


def queue_subjects(route3_root: Path) -> list[str]:
    rows = read_csv(route3_root / "route_qc_decisions.csv")
    out = []
    for row in rows:
        if row.get("qc_decision") != "TRY_ANOTHER_ROUTE":
            continue
        try:
            labels = int(float(row.get("source_labels") or 0))
        except Exception:
            labels = 0
        if labels >= 160:
            out.append(row["sid"])
    return sorted(dict.fromkeys(out))


def completed_sids(run_root: Path) -> set[str]:
    return {r.get("sid", "") for r in read_csv(run_root / "batch_results.csv") if r.get("status") == "ok"}


def run_one(sid: str, tag: str, route3_root: Path, mrtrix_threads: int) -> dict[str, Any]:
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
        "--route3-root",
        str(route3_root),
    ]
    proc = subprocess.run(cmd, cwd="/home/ec2-user/exp", env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, required=True)
    parser.add_argument("--route3-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mrtrix-threads", type=int, default=4)
    args = parser.parse_args()

    run_root = args.route1_root.parent / f"{args.route1_root.name}_route4_label_rescue"
    run_root.mkdir(parents=True, exist_ok=True)
    tag = run_root.name
    subjects = queue_subjects(args.route3_root)
    done = completed_sids(run_root)
    pending = [sid for sid in subjects if sid not in done]
    results: list[dict[str, Any]] = list(read_csv(run_root / "batch_results.csv"))
    write_json(run_root / "run_manifest.json", {"mode": "route4_label_preserving_dilation", "route3_root": str(args.route3_root), "root": str(run_root), "tag": tag, "n_subjects": len(subjects), "workers": args.workers, "mrtrix_threads": args.mrtrix_threads, "started_utc": utc()})
    update_status(run_root, phase="running", total=len(subjects), completed=len(done), failed=sum(1 for r in results if r.get("status") == "failed"), active=pending[: args.workers])

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, sid, tag, args.route3_root, args.mrtrix_threads): sid for sid in pending}
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {"sid": sid, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_utc": utc()}
            results = [r for r in results if r.get("sid") != sid] + [row]
            write_csv(run_root / "batch_results.csv", results)
            completed = sum(1 for r in results if r.get("status") == "ok")
            failed = sum(1 for r in results if r.get("status") == "failed")
            remaining = [s for s in pending if s not in {r.get("sid") for r in results}]
            update_status(run_root, completed=completed, failed=failed, active=remaining[: args.workers], latest=row)
            subprocess.run(["/home/ec2-user/exp/.venv_connectome_app/bin/python", str(SCORER), "--run-root", str(run_root)], cwd="/home/ec2-user/exp", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    update_status(run_root, phase="complete", completed=sum(1 for r in results if r.get("status") == "ok"), failed=sum(1 for r in results if r.get("status") == "failed"), active=[], completed_utc=utc())
    subprocess.run(["/home/ec2-user/exp/.venv_connectome_app/bin/python", str(SCORER), "--run-root", str(run_root)], cwd="/home/ec2-user/exp", stdout=(run_root / "score_stdout.log").open("w"), stderr=(run_root / "score_stderr.log").open("w"), check=False)
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

