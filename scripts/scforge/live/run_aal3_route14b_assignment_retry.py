#!/usr/bin/env python3
"""Route 14B: assignment retry on rebuilt Step-7 Route14 tracks.

Route14 showed that rebuilt Step-7 ACT/SIFT2 tracks can substantially improve
AAL3 density, but the first canary still missed full QC. This lane reuses the
existing Route14 scratch tracks and SIFT2 weights, then runs additional
tck2connectome assignment policies before spending CPU on a larger tckgen run.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live import run_aal3_route7_selective_rescue_batch as route7  # noqa: E402
from scripts.scforge.live.run_aal3_route11b_act_sift2_track_coverage_rerun import (  # noqa: E402
    DEFAULT_ROUTE1_ROOT,
    RUN_PARENT,
    collect_route11_parcellations,
)
from scripts.scforge.live.run_sc_aal3_source_contract_probe import (  # noqa: E402
    QC_ROOT,
    aal3_valid_labels,
    discover_inputs,
    label_stats,
    write_csv,
    write_json,
)


SCORER = Path("/home/ec2-user/exp/scripts/scforge/live/score_aal3_force_connectome_qc.py")
LEDGER = Path("/home/ec2-user/exp/scripts/scforge/live/summarize_aal3_route_ledger.py")
SOURCE_SUFFIX = "_route14_step7_rebuilt_track_coverage"
OUT_SUFFIX = "_route14b_step7_assignment_retry"
SOURCE_PROBE_PARENT = QC_ROOT / "aal3_route14_step7_rebuilt_track_coverage_probe"
OUT_PROBE_PARENT = QC_ROOT / "aal3_route14b_step7_assignment_retry_probe"


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
    status = read_json(root / "status.json")
    status.update(updates)
    status["updated_utc"] = utc()
    write_json(root / "status.json", status)


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def source_probe_root(route1_root: Path, sid: str) -> Path:
    return SOURCE_PROBE_PARENT / f"{route1_root.name}{SOURCE_SUFFIX}_{sid}"


def out_probe_root(route1_root: Path, sid: str) -> Path:
    return OUT_PROBE_PARENT / f"{route1_root.name}{OUT_SUFFIX}_{sid}"


def completed_sids(run_root: Path, retry_failed: bool) -> set[str]:
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


def source_subjects(route1_root: Path, requested: list[str]) -> list[str]:
    if requested:
        return sorted(set(requested))
    source_run = route1_root.parent / f"{route1_root.name}{SOURCE_SUFFIX}"
    rows = read_csv(source_run / "batch_results.csv")
    return sorted({row["sid"] for row in rows if row.get("status") == "ok" and row.get("sid")})


def choose_parcellation(sid: str, route1_root: Path, route2_root: Path, valid_labels: list[int]) -> dict[str, Any]:
    candidates = [
        row for row in collect_route11_parcellations(sid, route1_root, route2_root, valid_labels)
        if int(fnum(row.get("labels"), 0)) >= 160
    ]
    if not candidates:
        raise FileNotFoundError(f"No >=160-label AAL3 parcellation found for {sid}")
    return candidates[0]


def track_count_from_source(source_root: Path, sid: str, track_modes: list[str]) -> int:
    counts: list[int] = []
    for mode in track_modes:
        for path in sorted((source_root / "tracks").glob(f"{sid}_{mode}_*k.tck")):
            stem = path.stem
            count_part = stem.rsplit("_", 1)[-1]
            if count_part.endswith("k"):
                try:
                    counts.append(int(count_part[:-1]) * 1000)
                except ValueError:
                    pass
    if not counts:
        raise FileNotFoundError(f"No source Route14 tracks found under {source_root / 'tracks'}")
    return max(counts)


def run_one(
    *,
    sid: str,
    route1_root: Path,
    route2_root: Path,
    run_root: Path,
    variants: list[str],
    track_modes: list[str],
    mrtrix_threads: int,
) -> dict[str, Any]:
    src_root = source_probe_root(route1_root, sid)
    if not src_root.exists():
        raise FileNotFoundError(f"Missing Route14 source probe for {sid}: {src_root}")
    probe_root = out_probe_root(route1_root, sid)
    probe_root.mkdir(parents=True, exist_ok=True)
    update_status(probe_root, sid=sid, phase="starting", source_probe_root=str(src_root))

    inputs = discover_inputs(sid)
    if not inputs.get("production_fd_sum"):
        raise FileNotFoundError(f"Missing production fd_sum for {sid}")

    valid_labels = aal3_valid_labels()
    label_rows, matrix_rows, baseline_fd = route7.production_baseline(inputs, valid_labels)
    parc_candidate = choose_parcellation(sid, route1_root, route2_root, valid_labels)
    base_stage = str(parc_candidate["candidate"])
    select_streamlines = track_count_from_source(src_root, sid, track_modes)
    env = route7.env_for_mrtrix(mrtrix_threads)

    successful_modes: list[str] = []
    errors: list[str] = []
    for mode in track_modes:
        tracks = src_root / "tracks" / f"{sid}_{mode}_{select_streamlines // 1000}k.tck"
        weights = src_root / "tracks" / f"{sid}_{mode}_{select_streamlines // 1000}k_sift2_weights.txt"
        if not tracks.exists() or tracks.stat().st_size <= 0:
            errors.append(f"{mode}: missing tracks")
            continue
        if not weights.exists() or weights.stat().st_size <= 0:
            errors.append(f"{mode}: missing weights")
            continue
        track_stage = f"route14b_{mode}_tracks_{base_stage}"
        label_rows.append({"stage": track_stage, **label_stats(Path(parc_candidate["path"]), valid_labels)})
        write_csv(probe_root / "label_survival_summary.csv", label_rows)
        write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)
        route7.run_connectomes(
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
        successful_modes.append(mode)

    if not successful_modes:
        raise RuntimeError("; ".join(errors) or "no successful track modes")

    best = route7.choose_best_matrix(matrix_rows, label_rows, baseline_fd)
    batch_row = route7.row_from_best(
        sid,
        "ok",
        probe_root,
        best,
        baseline_fd,
        label_rows,
        {
            "route11_schema": "route14b_step7_assignment_retry_v1",
            "route11_lane": "R14B",
            "route11_reason": "assignment-only retry using Route14 rebuilt Step-7 tracks",
            "target_lane": "R14B",
            "route7_action": "route14b_assignment_retry",
            "select_streamlines": select_streamlines,
            "successful_track_modes": ",".join(successful_modes),
            "track_mode_errors": " | ".join(errors),
        },
    )
    write_json(probe_root / "decision.json", {"decision": "AAL3_ROUTE14B_ASSIGNMENT_RETRY", "best": best, "batch_row": batch_row})
    update_status(probe_root, phase="complete", completed_utc=utc(), decision="AAL3_ROUTE14B_ASSIGNMENT_RETRY")
    return batch_row


def refresh_outputs(route1_root: Path, run_root: Path) -> None:
    subprocess.run(
        ["/home/ec2-user/exp/.venv_connectome_app/bin/python", str(SCORER), "--run-root", str(run_root)],
        cwd="/home/ec2-user/exp",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if (route1_root / "subjects.csv").exists():
        subprocess.run(
            ["/home/ec2-user/exp/.venv_connectome_app/bin/python", str(LEDGER), "--run-root", str(route1_root)],
            cwd="/home/ec2-user/exp",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, default=DEFAULT_ROUTE1_ROOT)
    parser.add_argument("--route2-root", type=Path, default=None)
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--variants", nargs="*", default=["forward120", "forward200", "radial24", "allvoxels", "reverse80", "radial8"])
    parser.add_argument("--track-modes", nargs="*", default=["dynamic", "gmwmi"])
    parser.add_argument("--mrtrix-threads", type=int, default=24)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    route1_root = args.route1_root
    route2_root = args.route2_root or route1_root.parent / f"{route1_root.name}_route2"
    run_root = RUN_PARENT / f"{route1_root.name}{OUT_SUFFIX}"
    run_root.mkdir(parents=True, exist_ok=True)
    subjects = source_subjects(route1_root, args.subjects)
    done = completed_sids(run_root, args.retry_failed)
    pending = [sid for sid in subjects if sid not in done]
    results: list[dict[str, Any]] = [row for row in read_csv(run_root / "batch_results.csv") if row.get("sid") in done]

    write_json(
        run_root / "run_manifest.json",
        {
            "mode": "route14b_step7_assignment_retry_v1",
            "route1_root": str(route1_root),
            "route2_root": str(route2_root),
            "source_probe_parent": str(SOURCE_PROBE_PARENT),
            "out_probe_parent": str(OUT_PROBE_PARENT),
            "subjects": subjects,
            "variants": args.variants,
            "track_modes": args.track_modes,
            "mrtrix_threads": args.mrtrix_threads,
            "started_utc": utc(),
        },
    )
    update_status(run_root, phase="running", total=len(subjects), completed=len(done), failed=0, active=pending[:1])

    for sid in pending:
        update_status(run_root, phase="running", active=[sid])
        try:
            row = run_one(
                sid=sid,
                route1_root=route1_root,
                route2_root=route2_root,
                run_root=run_root,
                variants=list(args.variants),
                track_modes=list(args.track_modes),
                mrtrix_threads=max(1, args.mrtrix_threads),
            )
        except Exception as exc:
            row = {
                "sid": sid,
                "status": "failed",
                "route11_schema": "route14b_step7_assignment_retry_v1",
                "route11_lane": "R14B",
                "target_lane": "R14B",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_utc": utc(),
            }
        results = [r for r in results if r.get("sid") != sid] + [row]
        write_csv(run_root / "batch_results.csv", results)
        update_status(
            run_root,
            completed=sum(1 for r in results if r.get("status") == "ok"),
            failed=sum(1 for r in results if r.get("status") == "failed"),
            active=[],
            latest=row,
        )
        refresh_outputs(route1_root, run_root)

    summary = {
        "generated_utc": utc(),
        "route11_schema": "route14b_step7_assignment_retry_v1",
        "out_root": str(run_root),
        "total": len(subjects),
        "completed": sum(1 for r in results if r.get("status") == "ok"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
    }
    write_json(run_root / "route14b_summary.json", summary)
    update_status(run_root, phase="complete", active=[], summary=summary)
    refresh_outputs(route1_root, run_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {run_root / 'batch_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
