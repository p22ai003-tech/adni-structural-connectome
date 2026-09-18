#!/usr/bin/env python3
"""Route 11A: endpoint-overlap-guided AAL3 space/affine repair.

Route 10 identifies subjects where AAL3 labels survive but the streamline
endpoint cloud misses the selected label image. Route 11A searches existing
scratch parcellations for the same subject, ranks same-grid candidates by
endpoint-cloud overlap plus label survival, then rebuilds fd_sum connectomes
from the best candidates. It is scratch-only; promotion remains a separate
score + promote step.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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

from scripts.scforge.live.run_aal3_route8_full_t1_fnirt_batch import (  # noqa: E402
    run_connectomes_route8,
)
from scripts.scforge.live.run_aal3_route7_selective_rescue_batch import (  # noqa: E402
    choose_best_matrix,
    env_for_mrtrix,
    production_baseline,
    row_from_best,
)
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
DEFAULT_ROUTE10_ROOT = DEFAULT_ROUTE1_ROOT.parent / f"{DEFAULT_ROUTE1_ROOT.name}_route10_endpoint_overlay_triage"
PROBE_PARENT = QC_ROOT / "aal3_route11a_space_affine_registration_repair_probe"
ROUTE11A_SUFFIX = "_route11a_space_affine_registration_repair"
ROUTE11A_SCHEMA = "endpoint_overlap_candidate_repair_v1"
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


def same_grid(a: nib.spatialimages.SpatialImage, b: nib.spatialimages.SpatialImage) -> bool:
    return a.shape[:3] == b.shape[:3] and bool(np.allclose(a.affine, b.affine, atol=1e-3))


def clean_stage(path: Path) -> str:
    parent = path.parent.name
    stem = path.name.replace(".nii.gz", "").replace(".nii", "")
    raw = f"route11a_{parent}_{stem}"
    return re.sub(r"[^A-Za-z0-9_]+", "_", raw)[:140]


def queue_rows(route10_root: Path, subjects: set[str]) -> list[dict[str, str]]:
    rows = [
        row
        for row in read_csv(route10_root / "route10_triage.csv")
        if row.get("next_lane") == "R11A_space_affine_registration_repair"
    ]
    if subjects:
        rows = [row for row in rows if row.get("sid") in subjects]
    return sorted(rows, key=lambda row: row.get("sid", ""))


def finished_sids(run_root: Path, retry_failed: bool) -> set[str]:
    done: set[str] = set()
    for row in read_csv(run_root / "batch_results.csv"):
        sid = row.get("sid", "")
        if not sid:
            continue
        if row.get("status") == "ok" and row.get("route11a_schema") == ROUTE11A_SCHEMA:
            done.add(sid)
        elif row.get("status") == "failed" and not retry_failed:
            done.add(sid)
    return done


def discover_parcellation_candidates(sid: str) -> list[Path]:
    roots = [
        QC_ROOT / "aal3_source_contract_probe",
        QC_ROOT / "aal3_route2_bbr_probe",
        QC_ROOT / "aal3_route7_selective_rescue_probe",
        QC_ROOT / "aal3_route8_full_t1_fnirt_probe",
        QC_ROOT / "aal3_route9_route8_assignment_retry_probe",
    ]
    out: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for subject_root in root.glob(f"*{sid}*"):
            parc_dir = subject_root / "parc"
            if parc_dir.exists():
                out.update(path for path in parc_dir.rglob("*.nii.gz") if "AAL3" in path.name or "AAL3" in str(path))
    return sorted(out)


def endpoint_overlap_candidates(
    *,
    sid: str,
    endpoint_map: Path,
    expected_labels: list[int],
) -> list[dict[str, Any]]:
    endpoint_img = nib.load(str(endpoint_map))
    endpoints = np.nan_to_num(np.asanyarray(endpoint_img.dataobj), nan=0.0, posinf=0.0, neginf=0.0)
    endpoint_total = float(np.abs(endpoints).sum())
    rows: list[dict[str, Any]] = []
    for path in discover_parcellation_candidates(sid):
        row: dict[str, Any] = {"path": str(path), "stage": clean_stage(path)}
        try:
            img = nib.load(str(path))
            row["same_grid_as_endpoint_map"] = int(same_grid(img, endpoint_img))
            stats = label_stats(path, expected_labels)
            row.update({f"label_{k}": v for k, v in stats.items() if k != "path"})
            if not row["same_grid_as_endpoint_map"]:
                row["endpoint_inside_label_fraction"] = ""
                row["candidate_status"] = "geometry_mismatch_skipped"
                rows.append(row)
                continue
            labels = np.rint(np.asanyarray(img.dataobj)).astype(np.int64)
            inside = float(np.abs(endpoints[labels > 0]).sum())
            row["endpoint_inside_label_fraction"] = inside / endpoint_total if endpoint_total > 0 else 0.0
            row["endpoint_weight_inside_labels"] = inside
            row["endpoint_weight_total"] = endpoint_total
            row["candidate_status"] = "ranked"
        except Exception as exc:
            row["candidate_status"] = "failed_read"
            row["candidate_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def select_candidates(rows: list[dict[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    ranked = [
        row
        for row in rows
        if row.get("candidate_status") == "ranked"
        and int(fnum(row.get("label_n_labels"), 0)) >= 160
        and fnum(row.get("endpoint_inside_label_fraction"), -1) >= 0
    ]
    ranked.sort(
        key=lambda row: (
            fnum(row.get("endpoint_inside_label_fraction"), -1),
            fnum(row.get("label_n_labels"), 0),
            fnum(row.get("label_nonzero_voxels"), 0),
        ),
        reverse=True,
    )
    return ranked[:max_candidates]


def run_one(
    *,
    row: dict[str, str],
    tag: str,
    route10_root: Path,
    variants: list[str],
    max_candidates: int,
    mrtrix_threads: int,
    timeout_sec: int,
) -> dict[str, Any]:
    sid = row["sid"]
    env = env_for_mrtrix(mrtrix_threads)
    probe_root = PROBE_PARENT / f"{tag}_{sid}"
    probe_root.mkdir(parents=True, exist_ok=True)
    cached = read_json(probe_root / "decision.json")
    if cached.get("route11a_schema") == ROUTE11A_SCHEMA:
        return cached.get("batch_row", {})

    update_status(probe_root, sid=sid, tag=f"{tag}_{sid}", phase="starting")
    inputs = discover_inputs(sid)
    missing = [key for key in ("tracks", "weights", "production_fd_sum") if not inputs.get(key)]
    endpoint_map = Path(row.get("endpoint_map") or route10_root / "subjects" / sid / f"{sid}_route10_endpoint_density.nii.gz")
    if not endpoint_map.exists():
        missing.append("route10_endpoint_map")
    if missing:
        raise FileNotFoundError(f"Missing Route11A inputs for {sid}: {missing}")

    valid_labels = aal3_valid_labels()
    label_rows, matrix_rows, baseline_fd = production_baseline(inputs, valid_labels)

    update_status(probe_root, phase="rank_endpoint_overlap_candidates")
    candidates = endpoint_overlap_candidates(sid=sid, endpoint_map=endpoint_map, expected_labels=valid_labels)
    write_csv(probe_root / "route11a_candidate_overlap.csv", candidates)
    selected = select_candidates(candidates, max_candidates)
    if not selected:
        batch_row = row_from_best(
            sid,
            "ok",
            probe_root,
            {},
            baseline_fd,
            label_rows,
            {
                "route11a_schema": ROUTE11A_SCHEMA,
                "route11a_reason": "no same-grid AAL3 candidate had >=160 labels",
                "route10_endpoint_inside_label_fraction": row.get("endpoint_inside_label_fraction", ""),
            },
        )
        write_json(probe_root / "decision.json", {"route11a_schema": ROUTE11A_SCHEMA, "decision": "ROUTE11A_NO_USABLE_CANDIDATE", "best": {}, "batch_row": batch_row})
        update_status(probe_root, phase="complete", completed_utc=utc(), decision="ROUTE11A_NO_USABLE_CANDIDATE")
        return batch_row

    for cand in selected:
        parc_path = Path(str(cand["path"]))
        stage = str(cand["stage"])
        stats = label_stats(parc_path, valid_labels)
        label_rows.append(
            {
                "stage": stage,
                **stats,
                "route11a_endpoint_inside_label_fraction": cand.get("endpoint_inside_label_fraction", ""),
                "route11a_source_path": str(parc_path),
            }
        )
        write_csv(probe_root / "label_survival_summary.csv", label_rows)
        run_connectomes_route8(
            sid=sid,
            probe_root=probe_root,
            label_stage=stage,
            parc_path=parc_path,
            tracks=inputs["tracks"],
            weights=inputs["weights"],
            variants=variants,
            label_rows=label_rows,
            matrix_rows=matrix_rows,
            valid_labels=valid_labels,
            env=env,
            timeout_sec=timeout_sec,
        )

    best = choose_best_matrix(matrix_rows, label_rows, baseline_fd)
    best_stage = str(best.get("label_stage") or "")
    best_overlap = next((r.get("route11a_endpoint_inside_label_fraction", "") for r in label_rows if r.get("stage") == best_stage), "")
    batch_row = row_from_best(
        sid,
        "ok",
        probe_root,
        best,
        baseline_fd,
        label_rows,
        {
            "route11a_schema": ROUTE11A_SCHEMA,
            "route10_endpoint_inside_label_fraction": row.get("endpoint_inside_label_fraction", ""),
            "route11a_best_endpoint_inside_label_fraction": best_overlap,
            "route11a_selected_candidates": len(selected),
        },
    )
    write_json(probe_root / "decision.json", {"route11a_schema": ROUTE11A_SCHEMA, "decision": "ROUTE11A_ENDPOINT_OVERLAP_REPAIR", "best": best, "batch_row": batch_row})
    update_status(probe_root, phase="complete", completed_utc=utc(), decision="ROUTE11A_ENDPOINT_OVERLAP_REPAIR")
    return batch_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, default=DEFAULT_ROUTE1_ROOT)
    parser.add_argument("--route10-root", type=Path, default=DEFAULT_ROUTE10_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mrtrix-threads", type=int, default=2)
    parser.add_argument("--variants", nargs="*", default=["forward40", "forward80", "radial16", "endvox"])
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--connectome-timeout-sec", type=int, default=45 * 60)
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    route1_root = args.route1_root
    route10_root = args.route10_root
    out_root = route1_root.parent / f"{route1_root.name}{ROUTE11A_SUFFIX}"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = queue_rows(route10_root, set(args.subjects))
    done = finished_sids(out_root, args.retry_failed)
    rows = [row for row in rows if row.get("sid") not in done]
    tag = out_root.name

    write_csv(out_root / "subjects.csv", [{"sid": row.get("sid", ""), "group": "AD"} for row in rows])
    update_status(
        out_root,
        route11a_schema=ROUTE11A_SCHEMA,
        phase="running",
        route1_root=str(route1_root),
        route10_root=str(route10_root),
        total=len(rows) + len(done),
        completed=len(done),
        failed=0,
        active=[],
    )

    existing = read_csv(out_root / "batch_results.csv")
    results: list[dict[str, Any]] = [row for row in existing if row.get("sid") in done]
    failed: list[dict[str, Any]] = [row for row in existing if row.get("status") == "failed" and row.get("sid") in done]
    active: set[str] = set()

    def wrapped(input_row: dict[str, str]) -> dict[str, Any]:
        sid = input_row["sid"]
        active.add(sid)
        update_status(out_root, active=sorted(active))
        try:
            return run_one(
                row=input_row,
                tag=tag,
                route10_root=route10_root,
                variants=args.variants,
                max_candidates=max(1, args.max_candidates),
                mrtrix_threads=max(1, args.mrtrix_threads),
                timeout_sec=args.connectome_timeout_sec,
            )
        finally:
            active.discard(sid)
            update_status(out_root, active=sorted(active))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {pool.submit(wrapped, row): row for row in rows}
        for future in as_completed(future_map):
            row = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failed_row = {
                    "sid": row.get("sid", ""),
                    "status": "failed",
                    "route11a_schema": ROUTE11A_SCHEMA,
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_utc": utc(),
                }
                results.append(failed_row)
                failed.append(failed_row)
            write_csv(out_root / "batch_results.csv", results)
            update_status(out_root, completed=sum(1 for r in results if r.get("status") == "ok"), failed=sum(1 for r in results if r.get("status") == "failed"), active=sorted(active))

    summary = {
        "generated_utc": utc(),
        "route11a_schema": ROUTE11A_SCHEMA,
        "route1_root": str(route1_root),
        "route10_root": str(route10_root),
        "out_root": str(out_root),
        "total": len(results),
        "completed": sum(1 for row in results if row.get("status") == "ok"),
        "failed": sum(1 for row in results if row.get("status") == "failed"),
    }
    write_json(out_root / "route11a_summary.json", summary)
    update_status(out_root, phase="complete", completed=summary["completed"], failed=summary["failed"], active=[], summary=summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out_root / 'batch_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
