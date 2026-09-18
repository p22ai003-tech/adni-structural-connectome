#!/usr/bin/env python3
"""Route 13: rebuild Step 7 prep/FOD inputs for held AAL3 rescue cases.

This route is for subjects where downstream AAL3 assignment routes are no
longer the likely bottleneck. It reuses the canonical Step 7 implementation so
5TT, GMWMI, FOD masks, and WM FODs are regenerated under one auditable runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/ec2-user/exp")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from connectome_pipeline.connectome_step7 import (  # noqa: E402
    Step7Config,
    _format_tracks_geometry_audit,
    _subject_paths,
    _tracks_geometry_audit,
    run_step7_pipeline,
)


RUN_PARENT = Path("/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch")
DEFAULT_R1 = RUN_PARENT / "ad_existing_tracks_20260529T055752Z"


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def default_run_root(route1_root: Path) -> Path:
    return route1_root.parent / f"{route1_root.name}_route13_step7_preflight_rebuild"


def build_targets(route1_root: Path, limit: int = 0) -> list[dict[str, Any]]:
    hold_path = route1_root / "route12_preflight_rebuild_hold.csv"
    rows = read_csv(hold_path)
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        sid = row.get("sid", "").strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        targets.append(
            {
                "sid": sid,
                "group": (row.get("group") or "AD").strip().upper() or "AD",
                "source_hold": str(hold_path),
                "recommended_next_step": row.get("recommended_next_step", ""),
                "recommendation_rationale": row.get("recommendation_rationale", ""),
                "issue_class": row.get("issue_class", ""),
                "latest_route": row.get("latest_route", ""),
                "best_density": row.get("best_density", ""),
                "best_zero_rows": row.get("best_zero_rows", ""),
                "gmwmi_mean_before_audit": row.get("gmwmi_mean", ""),
                "gmwmi_nonzero_fraction_before_audit": row.get("gmwmi_nonzero_fraction", ""),
            }
        )
    if limit and limit > 0:
        targets = targets[:limit]
    return targets


def make_config(args: argparse.Namespace, run_root: Path, stage: str) -> Step7Config:
    batch = {
        "prep": args.prep_batch_size,
        "fod": args.fod_batch_size,
        "tracks": args.tracks_batch_size,
        "post": args.post_batch_size,
    }.get(stage, args.fod_batch_size)
    return Step7Config(
        batch_size=batch,
        force=args.force,
        tck_threads=args.tck_threads,
        select_streamlines=args.select_streamlines,
        track_seed_mode=args.track_seed_mode,
        track_max_parallel_tckgen=args.track_max_parallel_tckgen,
        track_chunk_jobs_per_subject=args.track_chunk_jobs_per_subject,
        prep_5tt_timeout_sec=args.prep_5tt_timeout_sec,
        fod_ss3t_timeout_sec=args.fod_ss3t_timeout_sec,
        backup_before_replace=True,
        validation_gate=True,
        qc_log_root=run_root / "step7_logs",
        runtime_log=run_root / "step7_runtime_notes.txt",
        summary_csv=run_root / "step7_summary.csv",
    )


def write_status(run_root: Path, payload: dict[str, Any]) -> None:
    base = {
        "updated_utc": utc(),
        "phase": "unknown",
    }
    base.update(payload)
    write_json(run_root / "status.json", base)


def geometry_audit(run_root: Path, sids: list[str]) -> list[dict[str, Any]]:
    cfg = Step7Config(
        force=False,
        qc_log_root=run_root / "geometry_audit_logs",
        runtime_log=run_root / "geometry_audit_runtime_notes.txt",
        summary_csv=run_root / "geometry_audit_step7_summary.csv",
    )
    rows: list[dict[str, Any]] = []
    for sid in sids:
        paths = _subject_paths(cfg, sid, create_dirs=False)
        row: dict[str, Any] = {"sid": sid, "audit_utc": utc(), "ok": 0, "error": ""}
        try:
            audit = _tracks_geometry_audit(cfg, sid, paths, include_fod_mask=True)
            row.update(audit)
            row["ok"] = int(
                int(audit.get("five_fix_nz") or 0) > 0
                and int(audit.get("mask_5tt_nz") or 0) > 0
                and int(audit.get("gmwmi_nz") or 0) > 0
                and int(audit.get("fod_mask_nz") or 0) > 0
                and int(audit.get("gmwmi_fod_overlap") or 0) > 0
                and int(audit.get("fod_5tt_overlap") or 0) > 0
            )
            row["note"] = _format_tracks_geometry_audit(audit)
        except Exception as exc:  # diagnostic row, do not abort the route
            row["error"] = str(exc)
        rows.append(row)
    write_csv(run_root / "step7_geometry_audit.csv", rows)
    return rows


def run_stage(args: argparse.Namespace, run_root: Path, stage: str, sids: list[str]) -> dict[str, Any]:
    cfg = make_config(args, run_root, stage)
    active_count = min(int(cfg.batch_size), len(sids))
    write_status(
        run_root,
        {
            "phase": "running",
            "active_stage": stage,
            "active_stage_batch_size": int(cfg.batch_size),
            "active_subject_count_estimate": active_count,
            "total": len(sids),
            "stages": args.stages,
        },
    )
    result = run_step7_pipeline(cfg, stage=stage, include=sids)
    write_json(run_root / f"result_{stage}.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, default=DEFAULT_R1)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stages", default="prep,fod", help="Comma-separated Step 7 stages: prep,fod,tracks,post")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prep-batch-size", type=int, default=2)
    parser.add_argument("--fod-batch-size", type=int, default=1)
    parser.add_argument("--tracks-batch-size", type=int, default=2)
    parser.add_argument("--post-batch-size", type=int, default=2)
    parser.add_argument("--tck-threads", type=int, default=4)
    parser.add_argument("--select-streamlines", type=int, default=3_000_000)
    parser.add_argument("--track-seed-mode", choices=("gmwmi", "dynamic"), default="dynamic")
    parser.add_argument("--track-max-parallel-tckgen", type=int, default=2)
    parser.add_argument("--track-chunk-jobs-per-subject", type=int, default=1)
    parser.add_argument("--prep-5tt-timeout-sec", type=int, default=1200)
    parser.add_argument("--fod-ss3t-timeout-sec", type=int, default=2400)
    parser.add_argument(
        "--skip-geometry-audit",
        action="store_true",
        help="Skip startup/final geometry audits; useful for high-throughput rebuild passes where downstream routes audit geometry.",
    )
    args = parser.parse_args()

    run_root = args.run_root or default_run_root(args.route1_root)
    run_root.mkdir(parents=True, exist_ok=True)
    targets = build_targets(args.route1_root, limit=args.limit)
    sids = [row["sid"] for row in targets]
    write_csv(run_root / "route_queue.csv", targets)
    write_csv(run_root / "subjects.csv", [{"sid": row["sid"], "group": row["group"]} for row in targets])
    write_json(
        run_root / "run_metadata.json",
        {
            "created_utc": utc(),
            "route1_root": str(args.route1_root),
            "run_root": str(run_root),
            "target_count": len(sids),
            "stages": args.stages,
            "force": args.force,
            "execute": args.execute,
            "step7_act_input_policy": "build 5TT from native T1, transform to B0, use 5tt_b0_fixed first",
            "step7_aal_input_policy": "register AAL/MNI to native T1, then apply native T1-to-B0 transform",
        },
    )
    if not args.skip_geometry_audit:
        geometry_audit(run_root, sids)

    print(f"AAL3 Route 13 Step7 preflight rebuild | {utc()}", flush=True)
    print(f"run root: {run_root}", flush=True)
    print(f"targets: {len(sids)} | stages={args.stages} | execute={args.execute}", flush=True)
    if sids:
        print("preview: " + ", ".join(sids[:10]) + (" ..." if len(sids) > 10 else ""), flush=True)
    if not args.execute:
        write_status(run_root, {"phase": "dry_run", "total": len(sids), "stages": args.stages})
        return 0
    if not sids:
        write_status(run_root, {"phase": "complete", "total": 0, "stages": args.stages})
        return 0

    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    valid = {"prep", "fod", "tracks", "post"}
    bad = [stage for stage in stages if stage not in valid]
    if bad:
        raise ValueError(f"Unsupported stages: {bad}")

    completed: list[str] = []
    try:
        for stage in stages:
            result = run_stage(args, run_root, stage, sids)
            completed.append(stage)
            ok_count = sum(1 for row in result.get("results", []) if row.get("ok"))
            write_status(
                run_root,
                {
                    "phase": "running",
                    "active_stage": "",
                    "last_completed_stage": stage,
                    "completed_stages": completed,
                    "last_stage_ok": ok_count,
                    "completed": ok_count if stage == stages[-1] else 0,
                    "active_subject_count_estimate": 0,
                    "total": len(sids),
                    "stages": args.stages,
                },
            )
            if not args.skip_geometry_audit:
                geometry_audit(run_root, sids)
        write_status(
            run_root,
            {
                "phase": "complete",
                "completed_stages": completed,
                "total": len(sids),
                "stages": args.stages,
            },
        )
    except Exception as exc:
        write_status(
            run_root,
            {
                "phase": "failed",
                "completed_stages": completed,
                "total": len(sids),
                "stages": args.stages,
                "error": f"{exc}\n{traceback.format_exc()}",
            },
        )
        raise
    finally:
        if not args.skip_geometry_audit:
            geometry_audit(run_root, sids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
