#!/usr/bin/env python3
"""Route 21: AAL3 endpoint-shell assignment-map probe.

This lane tests the dominant current failure mode: AAL3 labels survive, but
streamline endpoints do not land inside many thin / low-volume AAL3 ROIs. It
does not generate new tractography. Instead, it builds a scratch assignment
image by preserving the full AAL3 labels and extending each label into a small
nearest-label shell around the atlas. Existing 3M tracks and SIFT2 weights are
then re-assigned with the standard tck2connectome scorer.

The route is intentionally non-destructive. Production promotion is still
handled by the existing numeric scorer, visual overlay QC, and promotion script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage


ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live import run_aal3_route7_selective_rescue_batch as route7  # noqa: E402
from scripts.scforge.live.run_aal3_route11b_act_sift2_track_coverage_rerun import (  # noqa: E402
    DEFAULT_ROUTE1_ROOT,
    RUN_PARENT,
    collect_route11_parcellations,
    refresh_outputs,
)
from scripts.scforge.live.run_sc_aal3_source_contract_probe import (  # noqa: E402
    QC_ROOT,
    aal3_valid_labels,
    discover_inputs,
    label_stats,
    write_csv,
    write_json,
)


LANE = "R21"
SUFFIX = "_route21_endpoint_shell_assignment_map"
PROBE_NAME = "aal3_route21_endpoint_shell_assignment_map_probe"
SCHEMA = "route21_endpoint_shell_assignment_map_v1"
STATUS_LOCK = threading.Lock()
DEFAULT_AUDIT = Path("/home/ec2-user/exp/reports/aal3_endpoint_compatibility/aal3_endpoint_subject_audit_latest.csv")


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


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def safe_stage(*parts: str) -> str:
    raw = "_".join(part for part in parts if part)
    return re.sub(r"[^A-Za-z0-9_]+", "_", raw)[:140]


def latest_endpoint_audit_subjects(path: Path, limit: int) -> list[str]:
    rows = [
        row
        for row in read_csv(path)
        if row.get("sid")
        and row.get("corrective_class") == "roi_endpoint_coverage_failure"
        and inum(row.get("zero_row_count_recomputed"), 0) > 15
    ]
    rows.sort(
        key=lambda row: (
            inum(row.get("zero_row_count_recomputed"), 0),
            -fnum(row.get("matrix_density_recomputed"), 0.0),
        ),
        reverse=True,
    )
    return [row["sid"] for row in rows[: max(0, limit)]]


def build_endpoint_shell_map(
    *,
    parc_path: Path,
    out_path: Path,
    valid_labels: set[int],
    dilation_iters: int,
) -> dict[str, Any]:
    """Expand valid AAL3 labels into a local nearest-label shell.

    Original label voxels are never changed. Only unlabeled voxels inside the
    requested binary dilation shell get assigned to the nearest original label.
    """
    img = nib.load(str(parc_path))
    data = np.rint(np.asanyarray(img.dataobj)).astype(np.int32)
    valid_mask = np.isin(data, np.array(sorted(valid_labels), dtype=np.int32))
    clean = np.where(valid_mask, data, 0).astype(np.int32)
    label_mask = clean > 0
    if not bool(label_mask.any()):
        raise ValueError(f"No valid AAL3 labels in {parc_path}")

    shell_mask = ndimage.binary_dilation(label_mask, iterations=max(0, int(dilation_iters)))
    fill_mask = shell_mask & ~label_mask
    expanded = clean.copy()
    if bool(fill_mask.any()):
        _, nearest = ndimage.distance_transform_edt(~label_mask, return_indices=True)
        nearest_labels = clean[tuple(axis_idx[fill_mask] for axis_idx in nearest)]
        expanded[fill_mask] = nearest_labels

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(expanded.astype(np.int16), img.affine, img.header)
    out_img.header.set_data_dtype(np.int16)
    nib.save(out_img, str(out_path))

    labels_before = set(int(x) for x in np.unique(clean) if int(x) > 0)
    labels_after = set(int(x) for x in np.unique(expanded) if int(x) > 0)
    return {
        "source_parcellation": str(parc_path),
        "endpoint_shell_parcellation": str(out_path),
        "dilation_iters": int(dilation_iters),
        "source_nonzero_voxels": int(label_mask.sum()),
        "shell_nonzero_voxels": int((expanded > 0).sum()),
        "shell_added_voxels": int(fill_mask.sum()),
        "source_labels": len(labels_before),
        "shell_labels": len(labels_after),
        "labels_gained": len(labels_after - labels_before),
        "labels_lost": len(labels_before - labels_after),
    }


def route21_subject(
    *,
    sid: str,
    route1_root: Path,
    route2_root: Path,
    run_root: Path,
    probe_parent: Path,
    variants: list[str],
    dilation_iters: list[int],
    mrtrix_threads: int,
    min_labels: int,
    force_retry: bool,
) -> dict[str, Any]:
    probe_root = probe_parent / f"{run_root.name}_{sid}"
    marker = run_root / "subjects" / sid / "done.json"
    if marker.exists() and not force_retry:
        return read_json(marker)
    if (probe_root / "decision.json").exists() and not force_retry:
        payload = read_json(probe_root / "decision.json")
        row = payload.get("batch_row", {})
        if row:
            return row

    probe_root.mkdir(parents=True, exist_ok=True)
    update_status(
        probe_root,
        sid=sid,
        tag=f"{run_root.name}_{sid}",
        phase="starting",
        route11_schema=SCHEMA,
        route11_lane=LANE,
        target_lane=LANE,
    )

    inputs = discover_inputs(sid)
    missing = [key for key in ("production_fd_sum", "tracks", "weights") if not inputs.get(key)]
    if missing:
        raise FileNotFoundError(f"Missing route21 inputs for {sid}: {missing}")

    env = route7.env_for_mrtrix(mrtrix_threads)
    valid_labels = aal3_valid_labels()
    valid_set = set(valid_labels)
    label_rows, matrix_rows, baseline_fd = route7.production_baseline(inputs, valid_labels)

    parc_candidates = collect_route11_parcellations(sid, route1_root, route2_root, valid_labels)
    parc_candidates = [cand for cand in parc_candidates if int(cand.get("labels") or 0) >= min_labels]
    if not parc_candidates:
        batch_row = route7.row_from_best(
            sid,
            "ok",
            probe_root,
            {},
            baseline_fd,
            label_rows,
            {
                "route21_action": "endpoint_shell_assignment_map",
                "route21_reason": f"no parcellation with >= {min_labels} surviving labels",
                "route11_schema": SCHEMA,
                "route11_lane": LANE,
                "target_lane": LANE,
            },
        )
        write_csv(probe_root / "label_survival_summary.csv", label_rows)
        write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)
        write_json(probe_root / "decision.json", {"decision": "AAL3_ROUTE21_NO_HIGH_LABEL_PARCELLATION", "best": {}, "batch_row": batch_row})
        update_status(probe_root, phase="complete", completed_utc=utc(), decision="AAL3_ROUTE21_NO_HIGH_LABEL_PARCELLATION", latest=batch_row)
        marker.parent.mkdir(parents=True, exist_ok=True)
        write_json(marker, batch_row)
        return batch_row

    parc_candidate = parc_candidates[0]
    base_stage = str(parc_candidate["candidate"])
    label_rows.append({"stage": base_stage, **parc_candidate["label_stats"]})
    write_csv(probe_root / "label_survival_summary.csv", label_rows)
    write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)

    parc_dir = probe_root / "parc"
    shell_rows: list[dict[str, Any]] = []
    for dilation in sorted(set(int(x) for x in dilation_iters)):
        shell_path = parc_dir / f"{sid}_route21_endpoint_shell_dil{dilation}.nii.gz"
        update_status(probe_root, phase="build_endpoint_shell_map", active_dilation=dilation)
        shell_info = build_endpoint_shell_map(
            parc_path=Path(parc_candidate["path"]),
            out_path=shell_path,
            valid_labels=valid_set,
            dilation_iters=dilation,
        )
        label_stage = safe_stage("route21_shell", f"dil{dilation}", base_stage)
        shell_stats = label_stats(shell_path, valid_labels)
        label_rows.append({"stage": label_stage, **shell_stats})
        shell_row = {
            "sid": sid,
            "label_stage": label_stage,
            "base_label_stage": base_stage,
            "base_parcellation": str(parc_candidate["path"]),
            **shell_info,
        }
        shell_rows.append(shell_row)
        write_csv(probe_root / "endpoint_shell_summary.csv", shell_rows)
        write_csv(probe_root / "label_survival_summary.csv", label_rows)

        route7.run_connectomes(
            sid=sid,
            probe_root=probe_root,
            label_stage=label_stage,
            parc_path=shell_path,
            tracks=Path(inputs["tracks"]),
            weights=Path(inputs["weights"]),
            variants=variants,
            label_rows=label_rows,
            matrix_rows=matrix_rows,
            valid_labels=valid_labels,
            env=env,
        )

    best = route7.choose_best_matrix(matrix_rows, label_rows, baseline_fd)
    best_shell = next((row for row in shell_rows if row.get("label_stage") == best.get("label_stage")), {})
    batch_row = route7.row_from_best(
        sid,
        "ok",
        probe_root,
        best,
        baseline_fd,
        label_rows,
        {
            "route21_action": "endpoint_shell_assignment_map",
            "route11_schema": SCHEMA,
            "route11_lane": LANE,
            "target_lane": LANE,
            "base_label_stage": base_stage,
            "base_parcellation": str(parc_candidate["path"]),
            "shell_dilation_iters": best_shell.get("dilation_iters", ""),
            "shell_added_voxels": best_shell.get("shell_added_voxels", ""),
            "shell_nonzero_voxels": best_shell.get("shell_nonzero_voxels", ""),
            "candidate_parc_path": best_shell.get("endpoint_shell_parcellation", ""),
        },
    )
    write_json(probe_root / "decision.json", {"decision": "AAL3_ROUTE21_ENDPOINT_SHELL_ASSIGNMENT_MAP", "best": best, "batch_row": batch_row})
    update_status(probe_root, phase="complete", completed_utc=utc(), decision="AAL3_ROUTE21_ENDPOINT_SHELL_ASSIGNMENT_MAP", latest=batch_row)
    marker.parent.mkdir(parents=True, exist_ok=True)
    write_json(marker, batch_row)
    return batch_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, default=DEFAULT_ROUTE1_ROOT)
    parser.add_argument("--route2-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--probe-parent", type=Path, default=QC_ROOT / PROBE_NAME)
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--max-subjects", type=int, default=4)
    parser.add_argument(
        "--min-labels",
        type=int,
        default=166,
        help="Route 21 defaults to a full AAL3 label map; lower only for explicit diagnostics.",
    )
    parser.add_argument("--dilation-iters", nargs="*", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--variants", nargs="*", default=["endvox", "radial4", "radial8", "radial16", "forward40"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mrtrix-threads", type=int, default=4)
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args()

    route1_root = args.route1_root
    route2_root = args.route2_root or route1_root.parent / f"{route1_root.name}_route2"
    run_root = args.run_root or (route1_root.parent / f"{route1_root.name}{SUFFIX}")
    run_root.mkdir(parents=True, exist_ok=True)
    probe_parent = args.probe_parent
    probe_parent.mkdir(parents=True, exist_ok=True)
    subjects = sorted(set(args.subjects or latest_endpoint_audit_subjects(args.audit, args.max_subjects)))
    if not subjects:
        raise SystemExit("No Route 21 subjects supplied or found in endpoint audit.")

    existing = read_csv(run_root / "batch_results.csv")
    if args.retry:
        results = [row for row in existing if row.get("sid") not in set(subjects)]
        pending = subjects
    else:
        finished = {row.get("sid") for row in existing if row.get("status") in {"ok", "failed"}}
        results = list(existing)
        pending = [sid for sid in subjects if sid not in finished]

    write_json(
        run_root / "run_manifest.json",
        {
            "mode": SCHEMA,
            "route1_root": str(route1_root),
            "route2_root": str(route2_root),
            "root": str(run_root),
            "probe_parent": str(probe_parent),
            "subjects": subjects,
            "workers": args.workers,
            "mrtrix_threads": args.mrtrix_threads,
            "min_labels": args.min_labels,
            "dilation_iters": args.dilation_iters,
            "variants": args.variants,
            "started_utc": utc(),
            "note": "Scratch-only endpoint-shell assignment map; existing tracks and SIFT2 weights reused.",
        },
    )
    update_status(
        run_root,
        route11_schema=SCHEMA,
        route11_lane=LANE,
        phase="running",
        total=len(subjects),
        completed=sum(1 for row in results if row.get("status") == "ok"),
        failed=sum(1 for row in results if row.get("status") == "failed"),
        active=[],
    )

    active: set[str] = set()

    def wrapped(sid: str) -> dict[str, Any]:
        active.add(sid)
        update_status(run_root, active=sorted(active))
        try:
            return route21_subject(
                sid=sid,
                route1_root=route1_root,
                route2_root=route2_root,
                run_root=run_root,
                probe_parent=probe_parent,
                variants=args.variants,
                dilation_iters=args.dilation_iters,
                mrtrix_threads=max(1, args.mrtrix_threads),
                min_labels=max(1, args.min_labels),
                force_retry=args.retry,
            )
        finally:
            active.discard(sid)
            update_status(run_root, active=sorted(active))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(wrapped, sid): sid for sid in pending}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "sid": sid,
                    "status": "failed",
                    "route21_action": "endpoint_shell_assignment_map",
                    "route11_schema": SCHEMA,
                    "route11_lane": LANE,
                    "target_lane": LANE,
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_utc": utc(),
                }
                marker = run_root / "subjects" / sid / "done.json"
                marker.parent.mkdir(parents=True, exist_ok=True)
                write_json(marker, row)
            results = [old for old in results if old.get("sid") != sid] + [row]
            write_csv(run_root / "batch_results.csv", results)
            update_status(
                run_root,
                completed=sum(1 for item in results if item.get("status") == "ok"),
                failed=sum(1 for item in results if item.get("status") == "failed"),
                latest=row,
                active=sorted(active),
            )
            refresh_outputs(route1_root, run_root)

    update_status(run_root, phase="complete", completed_utc=utc(), active=[])
    refresh_outputs(route1_root, run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
