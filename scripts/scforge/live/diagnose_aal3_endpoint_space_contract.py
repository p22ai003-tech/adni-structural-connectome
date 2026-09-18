#!/usr/bin/env python3
"""Diagnose streamline endpoint space contract against AAL3 parcellations.

This is not a rescue route and it does not write production outputs. It asks a
more basic question for sparse AAL3 matrices: do streamline endpoints occupy the
same world-coordinate space as the AAL3 image, and if so, do they land near many
labels or collapse into a few labels?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage


ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live.run_sc_aal3_source_contract_probe import discover_inputs, write_csv, write_json  # noqa: E402


REPORT_DIR = Path("/home/ec2-user/exp/reports/aal3_endpoint_space_contract")
DEFAULT_CANARY = Path(
    "/home/ec2-user/tmp/aal3_route21_canary/"
    "ad_existing_tracks_20260529T055752Z_route21_endpoint_shell_assignment_map_canary"
)


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def parse_tck_header(path: Path) -> tuple[int, str, dict[str, str]]:
    header: dict[str, str] = {}
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii", errors="replace").strip()
        if first != "mrtrix tracks":
            raise ValueError(f"{path} is not an MRtrix .tck file")
        while True:
            line_b = handle.readline()
            if not line_b:
                raise ValueError(f"{path} header ended before END")
            line = line_b.decode("ascii", errors="replace").strip()
            if line == "END":
                break
            if ":" in line:
                key, value = line.split(":", 1)
                header[key.strip()] = value.strip()

    file_entry = header.get("file", "")
    parts = file_entry.split()
    if len(parts) >= 2:
        offset = int(parts[-1])
    else:
        raise ValueError(f"No data offset found in {path} header file entry: {file_entry!r}")
    datatype = header.get("datatype", "Float32LE")
    return offset, datatype, header


def stream_tck_endpoints(
    path: Path,
    *,
    max_streamlines: int,
    points_per_chunk: int = 1_000_000,
) -> np.ndarray:
    offset, datatype, _ = parse_tck_header(path)
    if datatype not in {"Float32LE", "Float32LE."}:
        raise ValueError(f"Unsupported TCK datatype {datatype!r}; expected Float32LE")

    endpoints: list[np.ndarray] = []
    current_first: np.ndarray | None = None
    current_last: np.ndarray | None = None
    count = 0
    stopped = False
    with path.open("rb") as handle:
        handle.seek(offset)
        while not stopped and count < max_streamlines:
            raw = np.fromfile(handle, dtype="<f4", count=points_per_chunk * 3)
            if raw.size == 0:
                break
            usable = raw.size - (raw.size % 3)
            pts = raw[:usable].reshape((-1, 3))
            for point in pts:
                if np.isinf(point).any():
                    if current_first is not None and current_last is not None and count < max_streamlines:
                        endpoints.extend([current_first.copy(), current_last.copy()])
                        count += 1
                    stopped = True
                    break
                if np.isnan(point).any():
                    if current_first is not None and current_last is not None:
                        endpoints.extend([current_first.copy(), current_last.copy()])
                        count += 1
                        if count >= max_streamlines:
                            stopped = True
                            break
                    current_first = None
                    current_last = None
                    continue
                if current_first is None:
                    current_first = point.copy()
                current_last = point.copy()
    if not endpoints:
        return np.empty((0, 3), dtype=np.float32)
    return np.vstack(endpoints).astype(np.float32)


def bbox(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {
            "min_x": "",
            "min_y": "",
            "min_z": "",
            "max_x": "",
            "max_y": "",
            "max_z": "",
            "center_x": "",
            "center_y": "",
            "center_z": "",
        }
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    center = (mins + maxs) / 2.0
    return {
        "min_x": float(mins[0]),
        "min_y": float(mins[1]),
        "min_z": float(mins[2]),
        "max_x": float(maxs[0]),
        "max_y": float(maxs[1]),
        "max_z": float(maxs[2]),
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
    }


def bbox_center_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    try:
        av = np.array([float(a[f"center_{axis}"]) for axis in ("x", "y", "z")])
        bv = np.array([float(b[f"center_{axis}"]) for axis in ("x", "y", "z")])
        return float(np.linalg.norm(av - bv))
    except Exception:
        return math.nan


def endpoint_samples_for_image(endpoints_world: np.ndarray, img: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray]:
    inv = np.linalg.inv(img.affine)
    hom = np.c_[endpoints_world, np.ones((endpoints_world.shape[0], 1), dtype=np.float32)]
    vox_float = hom @ inv.T
    vox = np.rint(vox_float[:, :3]).astype(np.int64)
    shape = np.array(img.shape[:3], dtype=np.int64)
    in_bounds = np.all((vox >= 0) & (vox < shape), axis=1)
    return vox, in_bounds


def world_bbox_for_mask(mask: np.ndarray, affine: np.ndarray) -> dict[str, Any]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return bbox(np.empty((0, 3)))
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    corners = np.array(
        [
            [mins[0], mins[1], mins[2], 1],
            [mins[0], mins[1], maxs[2], 1],
            [mins[0], maxs[1], mins[2], 1],
            [mins[0], maxs[1], maxs[2], 1],
            [maxs[0], mins[1], mins[2], 1],
            [maxs[0], mins[1], maxs[2], 1],
            [maxs[0], maxs[1], mins[2], 1],
            [maxs[0], maxs[1], maxs[2], 1],
        ],
        dtype=float,
    )
    world = (corners @ affine.T)[:, :3]
    return bbox(world)


def diagnose_subject(
    *,
    sid: str,
    parc_path: Path,
    tracks_path: Path,
    label: str,
    max_streamlines: int,
    compute_distance: bool,
) -> dict[str, Any]:
    img = nib.load(str(parc_path))
    data = np.rint(np.asanyarray(img.dataobj)).astype(np.int32)
    label_mask = data > 0
    labels = sorted(int(v) for v in np.unique(data[label_mask]))
    endpoints = stream_tck_endpoints(tracks_path, max_streamlines=max_streamlines)
    vox, in_bounds = endpoint_samples_for_image(endpoints, img)

    endpoint_labels: Counter[int] = Counter()
    in_label = np.zeros(endpoints.shape[0], dtype=bool)
    if endpoints.size:
        inside_vox = vox[in_bounds]
        sampled = data[inside_vox[:, 0], inside_vox[:, 1], inside_vox[:, 2]] if inside_vox.size else np.array([], dtype=np.int32)
        in_label[in_bounds] = sampled > 0
        endpoint_labels.update(int(v) for v in sampled if int(v) > 0)

    distance_summary: dict[str, Any] = {}
    if compute_distance and endpoints.size and label_mask.any():
        spacing = tuple(float(x) for x in img.header.get_zooms()[:3])
        distances = ndimage.distance_transform_edt(~label_mask, sampling=spacing)
        endpoint_dist = np.full(endpoints.shape[0], math.nan, dtype=float)
        if in_bounds.any():
            inside_vox = vox[in_bounds]
            endpoint_dist[in_bounds] = distances[inside_vox[:, 0], inside_vox[:, 1], inside_vox[:, 2]]
        finite = endpoint_dist[np.isfinite(endpoint_dist)]
        if finite.size:
            distance_summary = {
                "endpoint_to_nearest_label_mm_median": float(np.median(finite)),
                "endpoint_to_nearest_label_mm_p95": float(np.percentile(finite, 95)),
                "endpoint_to_nearest_label_mm_max": float(np.max(finite)),
                "endpoints_within_2mm_label_fraction": float((finite <= 2.0).mean()),
                "endpoints_within_4mm_label_fraction": float((finite <= 4.0).mean()),
                "endpoints_within_8mm_label_fraction": float((finite <= 8.0).mean()),
            }

    endpoint_bbox = bbox(endpoints)
    parc_bbox = world_bbox_for_mask(label_mask, img.affine)
    total_endpoints = int(endpoints.shape[0])
    unique_endpoint_labels = len(endpoint_labels)
    top_labels = endpoint_labels.most_common(12)
    top_fraction = float(sum(v for _, v in endpoint_labels.most_common(3)) / max(1, sum(endpoint_labels.values())))
    if total_endpoints == 0:
        failure_class = "no_track_endpoints_sampled"
    elif float(in_bounds.mean()) < 0.05:
        failure_class = "endpoint_world_bbox_outside_parcellation_image"
    elif float(in_label.mean()) == 0.0:
        failure_class = "endpoints_inside_image_but_not_on_any_label"
    elif unique_endpoint_labels <= 8 or top_fraction >= 0.90:
        failure_class = "endpoint_label_collapse"
    elif float(in_label.mean()) < 0.10:
        failure_class = "low_endpoint_label_overlap"
    else:
        failure_class = "endpoint_space_contract_plausible"

    row = {
        "sid": sid,
        "parcellation_label": label,
        "parcellation_path": str(parc_path),
        "tracks_path": str(tracks_path),
        "sampled_streamlines": int(total_endpoints / 2),
        "sampled_endpoints": total_endpoints,
        "parc_shape": "x".join(str(x) for x in img.shape[:3]),
        "parc_voxel_sizes": "x".join(f"{float(x):.4g}" for x in img.header.get_zooms()[:3]),
        "parc_label_count": len(labels),
        "parc_nonzero_voxels": int(label_mask.sum()),
        "endpoint_in_image_fraction": float(in_bounds.mean()) if total_endpoints else math.nan,
        "endpoint_on_label_fraction": float(in_label.mean()) if total_endpoints else math.nan,
        "unique_endpoint_labels": unique_endpoint_labels,
        "top3_endpoint_label_fraction": top_fraction,
        "top_endpoint_labels": "; ".join(f"{k}:{v}" for k, v in top_labels),
        "bbox_center_distance_mm": bbox_center_distance(endpoint_bbox, parc_bbox),
        "failure_class": failure_class,
    }
    row.update({f"endpoint_bbox_{k}": v for k, v in endpoint_bbox.items()})
    row.update({f"parc_bbox_{k}": v for k, v in parc_bbox.items()})
    row.update(distance_summary)
    return row


def canary_rows(run_root: Path) -> list[dict[str, str]]:
    rows = read_csv(run_root / "batch_results.csv")
    out: list[dict[str, str]] = []
    for row in rows:
        sid = row.get("sid", "")
        parc = row.get("candidate_parc_path", "")
        if sid and parc:
            out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-run-root", type=Path, default=DEFAULT_CANARY)
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--subjects-csv", type=Path, default=None)
    parser.add_argument("--subject-column", default="sid")
    parser.add_argument("--parcellation", type=Path, default=None, help="Single parcellation path if diagnosing one subject.")
    parser.add_argument("--production", action="store_true", help="Diagnose subjects using current production AAL_b0 and tracks.")
    parser.add_argument("--max-streamlines", type=int, default=250_000)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--skip-distance", action="store_true")
    args = parser.parse_args()

    subjects = list(args.subjects)
    if args.subjects_csv:
        for row in read_csv(args.subjects_csv):
            sid = str(row.get(args.subject_column) or "").strip()
            if sid and sid not in subjects:
                subjects.append(sid)

    tasks: list[tuple[str, Path, Path, str]] = []
    if args.production:
        if not subjects:
            raise SystemExit("--production requires one or more --subjects")
        for sid in subjects:
            inputs = discover_inputs(sid)
            tracks = inputs.get("tracks")
            parc = inputs.get("production_aal_b0")
            if not tracks or not parc:
                print(f"Skipping {sid}: missing tracks or production AAL_b0", file=sys.stderr)
                continue
            tasks.append((sid, Path(parc), Path(tracks), "production_AAL_b0"))
    elif args.parcellation:
        if len(subjects) != 1:
            raise SystemExit("--parcellation requires exactly one --subjects value")
        sid = subjects[0]
        tracks = discover_inputs(sid).get("tracks")
        if not tracks:
            raise SystemExit(f"No tracks found for {sid}")
        tasks.append((sid, args.parcellation, Path(tracks), "explicit"))
    else:
        for row in canary_rows(args.canary_run_root):
            sid = row["sid"]
            if args.subjects and sid not in set(args.subjects):
                continue
            if subjects and sid not in set(subjects):
                continue
            tracks = discover_inputs(sid).get("tracks")
            if not tracks:
                continue
            tasks.append((sid, Path(row["candidate_parc_path"]), Path(tracks), "route21_candidate"))

    if not tasks:
        raise SystemExit("No endpoint-space diagnostic tasks found.")

    rows = [
        diagnose_subject(
            sid=sid,
            parc_path=parc,
            tracks_path=tracks,
            label=label,
            max_streamlines=max(1, args.max_streamlines),
            compute_distance=not args.skip_distance,
        )
        for sid, parc, tracks, label in tasks
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    out_csv = args.out_dir / f"aal3_endpoint_space_contract_{stamp}.csv"
    out_json = args.out_dir / f"aal3_endpoint_space_contract_{stamp}.json"
    write_csv(out_csv, rows)
    latest_csv = args.out_dir / "aal3_endpoint_space_contract_latest.csv"
    write_csv(latest_csv, rows)
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": len(rows),
        "max_streamlines": args.max_streamlines,
        "source": str(args.canary_run_root),
        "failure_classes": dict(Counter(row["failure_class"] for row in rows)),
        "csv": str(out_csv),
    }
    write_json(out_json, payload)
    write_json(args.out_dir / "aal3_endpoint_space_contract_latest.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
