#!/usr/bin/env python3
"""Audit sampled streamline endpoint proximity to zero-row AAL3 labels.

This is a diagnostic companion to the AAL3-CC node table. Direct assignment
counts tell us whether endpoints were assigned to labels; this audit asks a
different question: are streamline endpoints spatially near the zero-row labels
at 2/4/6/8 mm, or are those labels genuinely outside endpoint coverage?

It never creates connectomes and never promotes production outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage


ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live.audit_aal3_endpoint_compatibility import (  # noqa: E402
    DEFAULT_RUN_ROOT,
    load_route_decisions,
    matrix_from_assignment,
    matrix_zero_labels,
    read_csv,
    selected_decision_for_subject,
)
from scripts.scforge.live.diagnose_aal3_endpoint_space_contract import (  # noqa: E402
    endpoint_samples_for_image,
    stream_tck_endpoints,
)
from scripts.scforge.live.run_sc_aal3_source_contract_probe import (  # noqa: E402
    aal3_valid_labels,
    discover_inputs,
    write_csv,
    write_json,
)


DEFAULT_OUT = Path("/home/ec2-user/exp/reports/aal3_endpoint_radius_coverage")


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return default
        out = float(text)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def assignment_label_stage(path: Path) -> str:
    name = path.name
    match = re.match(r"assignments_(?P<label_stage>.+)__[^_]+__fd_sum\.csv$", name)
    return match.group("label_stage") if match else ""


def selected_parcellation(probe_root: Path, decision: dict[str, str], assignment: Path) -> Path | None:
    label_stage = assignment_label_stage(assignment)
    preferred = [
        label_stage,
        decision.get("tested_route", ""),
        decision.get("best_candidate", ""),
        "production_AAL_b0",
    ]
    rows = read_csv(probe_root / "label_survival_summary.csv")
    by_stage = {row.get("stage", ""): row for row in rows}
    for stage in preferred:
        row = by_stage.get(stage)
        if row and row.get("path"):
            path = Path(row["path"])
            if path.exists():
                return path
    valid_rows = []
    for row in rows:
        path_text = row.get("path", "")
        if not path_text:
            continue
        path = Path(path_text)
        if path.exists() and int(fnum(row.get("read_ok"), 0)) == 1:
            valid_rows.append((int(fnum(row.get("n_labels"), 0)), path))
    if valid_rows:
        valid_rows.sort(reverse=True, key=lambda item: item[0])
        return valid_rows[0][1]
    return None


def selected_tracks(probe_root: Path, assignment: Path, sid: str) -> Path | None:
    label_stage = assignment_label_stage(assignment)
    rows = read_csv(probe_root / "endpoint_preflight_summary.csv")
    for row in rows:
        if row.get("label_stage") == label_stage and row.get("tracks_path"):
            path = Path(row["tracks_path"])
            if path.exists():
                return path
    inputs = discover_inputs(sid)
    tracks = inputs.get("tracks")
    if tracks and Path(tracks).exists():
        return Path(tracks)
    return None


def distance_counts_for_label(
    *,
    label: int,
    parc: np.ndarray,
    endpoints_vox: np.ndarray,
    endpoints_in_bounds: np.ndarray,
    spacing: tuple[float, float, float],
    radii: list[float],
) -> dict[str, Any]:
    mask = parc == int(label)
    voxels = int(mask.sum())
    out: dict[str, Any] = {"label": label, "label_voxels": voxels}
    if voxels <= 0:
        for radius in radii:
            out[f"endpoints_within_{radius:g}mm"] = 0
            out[f"endpoint_fraction_within_{radius:g}mm"] = 0.0
        out["nearest_endpoint_distance_mm"] = ""
        return out
    dist = ndimage.distance_transform_edt(~mask, sampling=spacing)
    if endpoints_vox.size == 0 or not endpoints_in_bounds.any():
        distances = np.asarray([], dtype=float)
    else:
        inside = endpoints_vox[endpoints_in_bounds]
        distances = dist[inside[:, 0], inside[:, 1], inside[:, 2]]
    total = int(distances.size)
    for radius in radii:
        count = int(np.count_nonzero(distances <= radius)) if total else 0
        out[f"endpoints_within_{radius:g}mm"] = count
        out[f"endpoint_fraction_within_{radius:g}mm"] = count / total if total else 0.0
    out["nearest_endpoint_distance_mm"] = float(np.min(distances)) if total else ""
    return out


def audit_subject(
    *,
    sid: str,
    ledger_row: dict[str, str],
    route_rows: dict[tuple[str, str], dict[str, str]],
    valid_labels: list[int],
    max_streamlines: int,
    radii: list[float],
    zero_rows_only: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    route, decision = selected_decision_for_subject(ledger_row, route_rows)
    if decision is None:
        return {"sid": sid, "status": "missing_decision", "selected_route": route}, []
    assignment_text = decision.get("assignment_file", "")
    assignment = Path(assignment_text) if assignment_text else Path()
    probe_root = Path(decision.get("probe_root", ""))
    matrix = matrix_from_assignment(assignment, sid) if assignment_text else None
    zero_labels, _, matrix_ok = matrix_zero_labels(matrix, valid_labels) if matrix else (set(valid_labels), math.nan, False)
    labels = sorted(zero_labels if zero_rows_only else set(valid_labels))
    parc_path = selected_parcellation(probe_root, decision, assignment)
    tracks_path = selected_tracks(probe_root, assignment, sid)
    if parc_path is None or tracks_path is None or not matrix_ok:
        return {
            "sid": sid,
            "status": "missing_parc_tracks_or_matrix",
            "selected_route": route,
            "assignment_file": str(assignment),
            "matrix_file": str(matrix) if matrix else "",
            "parcellation": str(parc_path or ""),
            "tracks": str(tracks_path or ""),
            "zero_row_count": len(zero_labels),
        }, []

    img = nib.load(str(parc_path))
    parc = np.rint(np.asanyarray(img.dataobj)).astype(np.int32)
    endpoints = stream_tck_endpoints(tracks_path, max_streamlines=max_streamlines)
    vox, in_bounds = endpoint_samples_for_image(endpoints, img)
    spacing = tuple(float(x) for x in img.header.get_zooms()[:3])

    label_rows = []
    for label in labels:
        row = distance_counts_for_label(
            label=label,
            parc=parc,
            endpoints_vox=vox,
            endpoints_in_bounds=in_bounds,
            spacing=spacing,
            radii=radii,
        )
        row.update(
            {
                "sid": sid,
                "selected_route": route,
                "parcellation": str(parc_path),
                "tracks": str(tracks_path),
                "sampled_streamlines": min(max_streamlines, int(endpoints.shape[0] // 2)),
                "sampled_endpoints": int(endpoints.shape[0]),
                "endpoints_in_bounds": int(in_bounds.sum()) if endpoints.size else 0,
                "is_zero_row": int(label in zero_labels),
            }
        )
        label_rows.append(row)
    subject = {
        "sid": sid,
        "status": "ok",
        "selected_route": route,
        "assignment_file": str(assignment),
        "matrix_file": str(matrix),
        "parcellation": str(parc_path),
        "tracks": str(tracks_path),
        "zero_row_count": len(zero_labels),
        "labels_audited": len(labels),
        "sampled_streamlines": min(max_streamlines, int(endpoints.shape[0] // 2)),
        "sampled_endpoints": int(endpoints.shape[0]),
        "endpoints_in_bounds_fraction": float(in_bounds.mean()) if endpoints.size else "",
    }
    for radius in radii:
        key = f"labels_with_endpoint_within_{radius:g}mm"
        count_key = f"endpoints_within_{radius:g}mm"
        subject[key] = sum(1 for row in label_rows if int(row.get(count_key) or 0) > 0)
    return subject, label_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-streamlines", type=int, default=50_000)
    parser.add_argument("--radii-mm", nargs="*", type=float, default=[2.0, 4.0, 6.0, 8.0])
    parser.add_argument("--zero-rows-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    valid_labels = aal3_valid_labels()
    ledger = [row for row in read_csv(args.run_root / "route_subject_ledger.csv") if row.get("group") == "AD"]
    if args.subjects:
        wanted = set(args.subjects)
        ledger = [row for row in ledger if row.get("sid") in wanted]
    if args.limit > 0:
        ledger = ledger[: args.limit]
    route_rows = load_route_decisions(args.run_root)

    subject_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    aggregate: dict[int, Counter[str]] = defaultdict(Counter)
    for ledger_row in ledger:
        sid = ledger_row.get("sid", "")
        subject, labels = audit_subject(
            sid=sid,
            ledger_row=ledger_row,
            route_rows=route_rows,
            valid_labels=valid_labels,
            max_streamlines=args.max_streamlines,
            radii=args.radii_mm,
            zero_rows_only=args.zero_rows_only,
        )
        subject_rows.append(subject)
        label_rows.extend(labels)
        for row in labels:
            label = int(row["label"])
            aggregate[label]["subjects_audited"] += 1
            for radius in args.radii_mm:
                if int(row.get(f"endpoints_within_{radius:g}mm") or 0) > 0:
                    aggregate[label][f"subjects_with_endpoint_within_{radius:g}mm"] += 1

    summary_rows = []
    for label, counts in sorted(aggregate.items()):
        row: dict[str, Any] = {"label": label, **counts}
        summary_rows.append(row)

    stamp = utc_stamp()
    out_dir = args.out_dir
    subject_path = out_dir / f"aal3_endpoint_radius_subject_audit_{stamp}.csv"
    label_path = out_dir / f"aal3_endpoint_radius_label_audit_{stamp}.csv"
    summary_path = out_dir / f"aal3_endpoint_radius_label_summary_{stamp}.csv"
    write_csv(subject_path, subject_rows)
    write_csv(label_path, label_rows)
    write_csv(summary_path, summary_rows)
    write_csv(out_dir / "aal3_endpoint_radius_subject_audit_latest.csv", subject_rows)
    write_csv(out_dir / "aal3_endpoint_radius_label_audit_latest.csv", label_rows)
    write_csv(out_dir / "aal3_endpoint_radius_label_summary_latest.csv", summary_rows)
    payload = {
        "generated_utc": stamp,
        "run_root": str(args.run_root),
        "subjects": len(subject_rows),
        "label_rows": len(label_rows),
        "max_streamlines": args.max_streamlines,
        "radii_mm": args.radii_mm,
        "zero_rows_only": args.zero_rows_only,
        "status_counts": dict(Counter(row.get("status", "") for row in subject_rows)),
        "subject_csv": str(subject_path),
        "label_csv": str(label_path),
        "summary_csv": str(summary_path),
    }
    write_json(out_dir / f"aal3_endpoint_radius_summary_{stamp}.json", payload)
    write_json(out_dir / "aal3_endpoint_radius_summary_latest.json", payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
