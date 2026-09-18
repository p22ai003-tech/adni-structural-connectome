#!/usr/bin/env python3
"""Route 11B: ACT/SIFT2 tract coverage rerun for unresolved AAL3 cases.

This lane is for subjects where a high-label AAL3 parcellation exists, but the
existing 3M streamlines do not terminate/cover the AAL3 nodes well enough. It
generates scratch-only ACT streamlines and scratch SIFT2 weights, then rebuilds
fd_sum connectomes. Production promotion remains handled by the standard scorer
and guarded promotion script.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live import run_aal3_route7_selective_rescue_batch as route7  # noqa: E402
from scripts.scforge.live.run_sc_aal3_source_contract_probe import (  # noqa: E402
    QC_ROOT,
    aal3_valid_labels,
    discover_inputs,
    label_stats,
    write_csv,
    write_json,
)


RUN_PARENT = Path("/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch")
DEFAULT_ROUTE1_ROOT = RUN_PARENT / "ad_existing_tracks_20260529T055752Z"
SCORER = Path("/home/ec2-user/exp/scripts/scforge/live/score_aal3_force_connectome_qc.py")
LEDGER = Path("/home/ec2-user/exp/scripts/scforge/live/summarize_aal3_route_ledger.py")
LANE_CONFIG = {
    "R11B": {
        "suffix": "_route11b_act_sift2_track_coverage_rerun",
        "probe": "aal3_route11b_act_sift2_track_coverage_rerun_probe",
        "schema": "route11b_act_sift2_track_coverage_rerun_v1",
        "default_track_modes": ["dynamic"],
        "default_variants": ["forward80", "forward120", "radial16", "endvox"],
    },
    "R11C": {
        "suffix": "_route11c_zero_row_coverage_rescue",
        "probe": "aal3_route11c_zero_row_coverage_rescue_probe",
        "schema": "route11c_zero_row_coverage_rescue_v1",
        "default_track_modes": ["dynamic", "gmwmi"],
        "default_variants": ["allvoxels", "endvox", "radial24", "forward200"],
    },
    "R12A": {
        "suffix": "_route12a_tckgen_fallback_rescue",
        "probe": "aal3_route12a_tckgen_fallback_rescue_probe",
        "schema": "route12a_tckgen_fallback_rescue_v1",
        "default_track_modes": ["gmwmi", "dynamic"],
        "default_variants": ["endvox", "radial16", "radial24", "forward80"],
    },
    "R12B": {
        "suffix": "_route12b_zero_row_targeted_rerun",
        "probe": "aal3_route12b_zero_row_targeted_rerun_probe",
        "schema": "route12b_zero_row_targeted_rerun_v1",
        "default_track_modes": ["gmwmi", "dynamic"],
        "default_variants": ["allvoxels", "endvox", "radial24", "forward200"],
    },
    "R14": {
        "suffix": "_route14_step7_rebuilt_track_coverage",
        "probe": "aal3_route14_step7_rebuilt_track_coverage_probe",
        "schema": "route14_step7_rebuilt_track_coverage_v1",
        "default_track_modes": ["dynamic", "gmwmi"],
        "default_variants": ["forward80", "radial16", "endvox"],
    },
    "R15": {
        "suffix": "_route15_step7_dense_track_coverage",
        "probe": "aal3_route15_step7_dense_track_coverage_probe",
        "schema": "route15_step7_dense_track_coverage_v1",
        "default_track_modes": ["dynamic"],
        "default_variants": ["forward80"],
    },
    "R17": {
        "suffix": "_route17_step7_dense_unpushed",
        "probe": "aal3_route17_step7_dense_unpushed_probe",
        "schema": "route17_step7_dense_unpushed_v1",
        "default_track_modes": ["dynamic"],
        "default_variants": ["forward80"],
    },
    "R18": {
        "suffix": "_route18_gmwmi_dense_track_coverage",
        "probe": "aal3_route18_gmwmi_dense_track_coverage_probe",
        "schema": "route18_gmwmi_dense_track_coverage_v1",
        "default_track_modes": ["gmwmi", "dynamic"],
        "default_variants": ["forward80", "radial16", "endvox"],
    },
    "R19": {
        "suffix": "_route19_gmwmi_3m_retry",
        "probe": "aal3_route19_gmwmi_3m_retry_probe",
        "schema": "route19_gmwmi_3m_retry_v1",
        "default_track_modes": ["gmwmi", "dynamic"],
        "default_variants": ["forward80", "radial16", "endvox"],
    },
}
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


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def safe_stage(*parts: str) -> str:
    raw = "_".join(part for part in parts if part)
    return re.sub(r"[^A-Za-z0-9_]+", "_", raw)[:140]


def same_grid(first: Path, second: Path) -> bool:
    try:
        img1 = nib.load(str(first))
        img2 = nib.load(str(second))
        return img1.shape[:3] == img2.shape[:3] and bool(np.allclose(img1.affine, img2.affine, atol=1e-3))
    except Exception:
        return False


def candidate_from_path(
    path: Path,
    source: str,
    valid_labels: list[int],
    reference_b0: Path | None,
) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    if source != "production_AAL_b0" and "aal3" not in path.name.lower():
        return None
    if reference_b0 is not None and source != "production_AAL_b0" and not same_grid(path, reference_b0):
        return None
    stats = label_stats(path, valid_labels)
    if int(fnum(stats.get("read_ok"), 0)) != 1:
        return None
    labels = int(fnum(stats.get("n_labels"), 0))
    if labels <= 0 or labels > len(valid_labels):
        return None
    return {
        "candidate": safe_stage("route11", source, path.parent.name, path.name.replace(".nii.gz", "")),
        "path": path,
        "source": source,
        "labels": labels,
        "label_stats": stats,
    }


def candidate_probe_roots(route1_root: Path, route2_root: Path | None, sid: str) -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    named = [
        ("route1_source_contract", QC_ROOT / "aal3_source_contract_probe" / f"{route1_root.name}_{sid}"),
        ("route2_bbr", QC_ROOT / "aal3_route2_bbr_probe" / f"{route2_root.name}_{sid}" if route2_root else Path("")),
    ]
    for source, root in named:
        if str(root) and root.exists():
            roots.append((source, root))
    glob_roots = [
        ("route7_selective", QC_ROOT / "aal3_route7_selective_rescue_probe"),
        ("route8_full_t1", QC_ROOT / "aal3_route8_full_t1_fnirt_probe"),
        ("route9_assignment_retry", QC_ROOT / "aal3_route9_route8_assignment_retry_probe"),
        ("route11a_space_affine", QC_ROOT / "aal3_route11a_space_affine_registration_repair_probe"),
    ]
    for source, parent in glob_roots:
        if parent.exists():
            roots.extend((source, root) for root in sorted(parent.glob(f"*{sid}*")) if root.is_dir())
    return roots


def collect_route11_parcellations(
    sid: str,
    route1_root: Path,
    route2_root: Path | None,
    valid_labels: list[int],
) -> list[dict[str, Any]]:
    inputs = discover_inputs(sid)
    reference_b0 = inputs.get("production_aal_b0")
    candidates: dict[str, dict[str, Any]] = {}
    if reference_b0:
        cand = candidate_from_path(reference_b0, "production_AAL_b0", valid_labels, reference_b0)
        if cand:
            candidates[str(cand["path"])] = cand
    for source, root in candidate_probe_roots(route1_root, route2_root, sid):
        parc = root / "parc"
        if not parc.exists():
            continue
        for path in sorted(parc.rglob("*.nii.gz")):
            if "aal3" not in path.name.lower():
                continue
            cand = candidate_from_path(path, source, valid_labels, reference_b0)
            if cand:
                candidates[str(path)] = cand
    out = list(candidates.values())
    out.sort(
        key=lambda row: (
            int(row.get("labels", 0)) >= 160,
            int(row.get("labels", 0)),
            fnum(row.get("label_stats", {}).get("nonzero_voxels"), 0.0),
            0 if row.get("source") == "production_AAL_b0" else 1,
        ),
        reverse=True,
    )
    return out


def finished_sids(run_root: Path, schema: str, retry_failed: bool) -> set[str]:
    done: set[str] = set()
    for row in read_csv(run_root / "batch_results.csv"):
        sid = row.get("sid", "")
        if not sid:
            continue
        if row.get("status") == "ok" and row.get("route11_schema") == schema:
            done.add(sid)
        elif row.get("status") == "failed" and not retry_failed:
            done.add(sid)
    return done


def select_streamline_count_for_subject(
    *,
    probe_parent: Path,
    tag: str,
    sid: str,
    track_modes: list[str],
    requested: int,
) -> int:
    """Avoid reusing interrupted partial .tck files from killed runs.

    Route 7's helper intentionally reuses existing scratch tracks, but if a
    tckgen process was interrupted before SIFT2 weights were written, the .tck
    can be nonempty and still unusable. Use a nearby filename/count in that
    case so the rerun starts cleanly without deleting scratch evidence.
    """
    count = requested
    while True:
        suffix = f"{count // 1000}k"
        track_root = probe_parent / f"{tag}_{sid}" / "tracks"
        conflict = False
        for mode in track_modes:
            tracks = track_root / f"{sid}_{mode}_{suffix}.tck"
            weights = track_root / f"{sid}_{mode}_{suffix}_sift2_weights.txt"
            if tracks.exists() and tracks.stat().st_size > 0 and (not weights.exists() or weights.stat().st_size <= 0):
                conflict = True
                break
        if not conflict:
            return count
        count += 1000


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


def run_main(default_lane: str = "R11B") -> int:
    cfg = LANE_CONFIG[default_lane]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, default=DEFAULT_ROUTE1_ROOT)
    parser.add_argument("--route2-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mrtrix-threads", type=int, default=4)
    parser.add_argument("--select-streamlines", type=int, default=3_000_000)
    parser.add_argument("--track-modes", nargs="*", default=cfg["default_track_modes"])
    parser.add_argument("--variants", nargs="*", default=cfg["default_variants"])
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    route1_root = args.route1_root
    route2_root = args.route2_root or route1_root.parent / f"{route1_root.name}_route2"
    run_root = route1_root.parent / f"{route1_root.name}{cfg['suffix']}"
    run_root.mkdir(parents=True, exist_ok=True)
    probe_parent = QC_ROOT / str(cfg["probe"])
    probe_parent.mkdir(parents=True, exist_ok=True)

    def current_queue_state() -> tuple[list[dict[str, str]], list[dict[str, str]], set[str], list[dict[str, str]]]:
        all_rows = [row for row in read_csv(run_root / "route_queue.csv") if row.get("sid")]
        if not all_rows and args.subjects:
            all_rows = [
                {
                    "sid": sid,
                    "target_lane": default_lane,
                    "reason": f"{cfg['schema']} direct subject canary",
                }
                for sid in sorted(set(args.subjects))
            ]
        selected_rows = list(all_rows)
        if args.subjects:
            wanted = set(args.subjects)
            selected_rows = [row for row in selected_rows if row.get("sid") in wanted]
        done_sids = finished_sids(run_root, str(cfg["schema"]), args.retry_failed)
        pending_rows = [row for row in selected_rows if row.get("sid") not in done_sids]
        return all_rows, selected_rows, done_sids, pending_rows

    all_queue_rows, queue_rows, done, pending = current_queue_state()
    results: list[dict[str, Any]] = [row for row in read_csv(run_root / "batch_results.csv") if row.get("sid") in done]
    active: set[str] = set()

    route7.PROBE_PARENT = probe_parent
    route7.collect_candidate_parcellations = collect_route11_parcellations

    write_json(
        run_root / "run_manifest.json",
        {
            "mode": cfg["schema"],
            "route1_root": str(route1_root),
            "route2_root": str(route2_root),
            "root": str(run_root),
            "probe_parent": str(probe_parent),
            "n_subjects": len(queue_rows),
            "n_route_subjects": len(all_queue_rows),
            "subject_subset": list(args.subjects),
            "workers": args.workers,
            "mrtrix_threads": args.mrtrix_threads,
            "select_streamlines": args.select_streamlines,
            "track_modes": args.track_modes,
            "variants": args.variants,
            "started_utc": utc(),
        },
    )
    update_status(
        run_root,
        route11_schema=cfg["schema"],
        phase="running",
        total=len(all_queue_rows),
        run_subset_total=len(queue_rows),
        completed=sum(1 for row in results if row.get("status") == "ok"),
        failed=sum(1 for row in results if row.get("status") == "failed"),
        active=[],
    )

    def wrapped(input_row: dict[str, str]) -> dict[str, Any]:
        sid = input_row["sid"]
        active.add(sid)
        update_status(run_root, active=sorted(active))
        try:
            work_row = {
                **input_row,
                "route7_action": "dynamic_track_rescue",
                "route11_schema": cfg["schema"],
                "route11_reason": input_row.get("reason", ""),
            }
            subject_select_streamlines = select_streamline_count_for_subject(
                probe_parent=probe_parent,
                tag=run_root.name,
                sid=sid,
                track_modes=list(args.track_modes),
                requested=args.select_streamlines,
            )
            row = route7.run_dynamic_track_rescue(
                sid,
                work_row,
                run_root.name,
                route1_root,
                route2_root,
                args.variants,
                args.track_modes,
                subject_select_streamlines,
                max(1, args.mrtrix_threads),
            )
            row["route11_schema"] = cfg["schema"]
            row["route11_lane"] = default_lane
            row["route11_reason"] = input_row.get("reason", "")
            row["target_lane"] = input_row.get("target_lane", default_lane)
            return row
        finally:
            active.discard(sid)
            update_status(run_root, active=sorted(active))

    while pending:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(wrapped, row): row for row in pending}
            for future in as_completed(futures):
                input_row = futures[future]
                sid = input_row.get("sid", "")
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "sid": sid,
                        "status": "failed",
                        "route11_schema": cfg["schema"],
                        "route11_lane": default_lane,
                        "route11_reason": input_row.get("reason", ""),
                        "error": f"{type(exc).__name__}: {exc}",
                        "finished_utc": utc(),
                    }
                results = [r for r in results if r.get("sid") != sid] + [row]
                write_csv(run_root / "batch_results.csv", results)
                update_status(
                    run_root,
                    completed=sum(1 for r in results if r.get("status") == "ok"),
                    failed=sum(1 for r in results if r.get("status") == "failed"),
                    active=sorted(active),
                    latest=row,
                )
                refresh_outputs(route1_root, run_root)

        all_queue_rows, queue_rows, done, pending = current_queue_state()
        results = [row for row in read_csv(run_root / "batch_results.csv") if row.get("sid") in done]
        if pending:
            update_status(
                run_root,
                total=len(all_queue_rows),
                run_subset_total=len(queue_rows),
                completed=sum(1 for r in results if r.get("status") == "ok"),
                failed=sum(1 for r in results if r.get("status") == "failed"),
                active=[],
                queue_refreshed_utc=utc(),
                pending_after_queue_refresh=len(pending),
            )

    summary = {
        "generated_utc": utc(),
        "route11_schema": cfg["schema"],
        "route1_root": str(route1_root),
        "out_root": str(run_root),
        "total": len(all_queue_rows),
        "run_subset_total": len(queue_rows),
        "completed": sum(1 for row in results if row.get("status") == "ok"),
        "failed": sum(1 for row in results if row.get("status") == "failed"),
    }
    write_json(run_root / "route11_summary.json", summary)
    update_status(run_root, phase="complete", completed=summary["completed"], failed=summary["failed"], active=[], summary=summary)
    refresh_outputs(route1_root, run_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {run_root / 'batch_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main("R11B"))
