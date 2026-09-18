#!/usr/bin/env python3
"""Run route-2 AAL3 BBR/native-route probes for unresolved AD subjects."""

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

RUN_PARENT = Path("/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch")
PROBE = Path("/home/ec2-user/exp/scripts/scforge/live/run_sc_aal3_bbr_route_probe.py")
SCORER = Path("/home/ec2-user/exp/scripts/scforge/live/score_aal3_force_connectome_qc.py")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def latest_route1_root() -> Path:
    latest = RUN_PARENT / "latest_tag.txt"
    if latest.exists() and latest.read_text(encoding="utf-8").strip():
        return RUN_PARENT / latest.read_text(encoding="utf-8").strip()
    roots = sorted((p for p in RUN_PARENT.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    if not roots:
        raise SystemExit(f"No route-1 roots under {RUN_PARENT}")
    return roots[0]


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


def unresolved_subjects(route1_root: Path, include_interim: bool) -> list[str]:
    rows = read_csv(route1_root / "route_qc_decisions.csv")
    decisions = {"TRY_ANOTHER_ROUTE", "REVIEW_ROUTE"}
    if include_interim:
        decisions.add("INTERIM_REVIEW_PROMOTION_PENDING_VISUAL")
    out: list[str] = []
    for row in rows:
        if row.get("qc_decision") in decisions and row.get("sid"):
            out.append(row["sid"])
    return sorted(dict.fromkeys(out))


def source_subject_groups(route1_root: Path) -> dict[str, str]:
    groups: dict[str, str] = {}
    for row in read_csv(route1_root / "subjects.csv"):
        sid = row.get("sid") or row.get("subject") or row.get("subject_id") or ""
        group = row.get("group") or row.get("diagnosis") or row.get("Group") or row.get("DX") or ""
        if sid:
            groups[sid] = group
    return groups


def completed_sids(run_root: Path) -> set[str]:
    return {row.get("sid", "") for row in read_csv(run_root / "batch_results.csv") if row.get("status") == "ok"}


def run_one(sid: str, tag: str, mrtrix_threads: int, variants: list[str]) -> dict[str, Any]:
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
        "--variants",
        *variants,
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd="/home/ec2-user/exp", check=False)
    probe_root = Path(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else Path("")
    decision = read_json(probe_root / "decision.json") if probe_root else {}
    best = decision.get("best") or {}
    row: dict[str, Any] = {
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
    if proc.returncode != 0:
        row["status"] = "failed"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--mrtrix-threads", type=int, default=4)
    parser.add_argument("--variants", nargs="*", default=["forward40", "forward80"])
    parser.add_argument("--include-interim", action="store_true", help="Also test interim promoted subjects because they are not final QC solved.")
    args = parser.parse_args()

    source_root = args.source_run_root or latest_route1_root()
    run_root = source_root.parent / f"{source_root.name}_route2"
    run_root.mkdir(parents=True, exist_ok=True)
    tag = run_root.name
    subjects = unresolved_subjects(source_root, include_interim=args.include_interim)
    groups = source_subject_groups(source_root)
    write_csv(run_root / "subjects.csv", [{"sid": sid, "group": groups.get(sid, "")} for sid in subjects])
    done_sids = completed_sids(run_root)
    pending = [sid for sid in subjects if sid not in done_sids]
    existing_rows = read_csv(run_root / "batch_results.csv")
    results: list[dict[str, Any]] = list(existing_rows)
    write_json(
        run_root / "run_manifest.json",
        {
            "mode": "route2_bbr_native_aal3_existing_tracks",
            "source_run_root": str(source_root),
            "root": str(run_root),
            "tag": tag,
            "n_subjects": len(subjects),
            "workers": args.workers,
            "mrtrix_threads": args.mrtrix_threads,
            "variants": args.variants,
            "include_interim": args.include_interim,
            "started_utc": utc(),
        },
    )
    update_status(run_root, phase="running", total=len(subjects), completed=len(done_sids), failed=sum(1 for r in results if r.get("status") == "failed"), active=[])

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, sid, tag, args.mrtrix_threads, args.variants): sid for sid in pending}
        active = set(pending[: args.workers])
        update_status(run_root, active=sorted(active))
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {"sid": sid, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_utc": utc()}
            results = [r for r in results if r.get("sid") != sid] + [row]
            write_csv(run_root / "batch_results.csv", results)
            write_csv(run_root / "batch_results_incremental.csv", results)
            completed = sum(1 for r in results if r.get("status") == "ok")
            failed = sum(1 for r in results if r.get("status") == "failed")
            remaining = [s for s in pending if s not in {r.get("sid") for r in results}]
            update_status(run_root, completed=completed, failed=failed, active=remaining[: args.workers], latest=row)

    update_status(run_root, phase="complete", completed=sum(1 for r in results if r.get("status") == "ok"), failed=sum(1 for r in results if r.get("status") == "failed"), active=[], completed_utc=utc())
    subprocess.run(
        [
            "/home/ec2-user/exp/.venv_connectome_app/bin/python",
            str(SCORER),
            "--run-root",
            str(run_root),
        ],
        cwd="/home/ec2-user/exp",
        text=True,
        stdout=(run_root / "score_stdout.log").open("w", encoding="utf-8"),
        stderr=(run_root / "score_stderr.log").open("w", encoding="utf-8"),
        check=False,
    )
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
