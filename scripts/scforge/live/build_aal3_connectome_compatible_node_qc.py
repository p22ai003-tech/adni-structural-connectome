#!/usr/bin/env python3
"""Build an AAL3 node-level QC table for a connectome-compatible node set.

This is a diagnostic artifact, not a production remapping. It summarizes, for
each AAL3 label, whether the label survives in subject parcellations and how
often it becomes a zero row in fd_sum matrices. The output is intended to drive
a manual AAL3-CC merge/exclude review rather than silently changing the atlas.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live.run_sc_aal3_source_contract_probe import (  # noqa: E402
    AAL3_LABELS,
    aal3_valid_labels,
    discover_inputs,
    matrix_array,
)


RUN_PARENT = Path("/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch")
DEFAULT_R1 = RUN_PARENT / "ad_existing_tracks_20260529T055752Z"
DEFAULT_ENDPOINT_LABEL_AUDIT = Path("/home/ec2-user/exp/reports/aal3_endpoint_compatibility/aal3_endpoint_label_audit_latest.csv")
MRCONVERT = Path("/home/ec2-user/mrtrix3/bin/mrconvert")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], delimiter: str = ",") -> None:
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
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)
    tmp.replace(path)


def label_names() -> dict[int, str]:
    names: dict[int, str] = {}
    for row in read_csv(AAL3_LABELS):
        raw = row.get("atlas_value") or row.get("label") or row.get("id")
        if not raw:
            continue
        try:
            label = int(float(raw))
        except Exception:
            continue
        name = (
            row.get("name")
            or row.get("label_name")
            or row.get("atlas_label")
            or row.get("node_name")
            or row.get("region")
            or row.get("roi")
            or f"AAL3_{label:03d}"
        )
        names[label] = str(name)
    return names


def zero_row_labels(matrix_path: Path, valid_labels: list[int]) -> set[int]:
    mat = matrix_array(matrix_path)
    if mat.ndim != 2:
        return set()
    row_signal = np.abs(mat).sum(axis=0) + np.abs(mat).sum(axis=1)
    valid = set(valid_labels)
    return {idx + 1 for idx, value in enumerate(row_signal) if value == 0 and (idx + 1) in valid}


def label_voxel_counts(parc_path: Path, valid_labels: list[int]) -> dict[int, int]:
    try:
        data = np.rint(np.asanyarray(nib.load(str(parc_path)).dataobj)).astype(np.int64)
    except Exception:
        return {}
    labels, counts = np.unique(data, return_counts=True)
    valid = set(valid_labels)
    return {int(label): int(count) for label, count in zip(labels, counts) if int(label) in valid}


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return default
        out = float(text)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def median(values: list[float], default: float = math.nan) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.median(finite)) if finite else default


def gmwmi_source_for_sid(sid: str) -> Path:
    return Path("/data/derivatives/fod") / sid / "gmwmi.mif"


def gmwmi_nifti_for_sid(sid: str, cache_dir: Path, env: dict[str, str]) -> Path | None:
    source = gmwmi_source_for_sid(sid)
    if not source.exists() or source.stat().st_size <= 0:
        return None
    out = cache_dir / f"{sid}_gmwmi.nii.gz"
    if out.exists() and out.stat().st_size > 0 and out.stat().st_mtime >= source.stat().st_mtime:
        return out
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = cache_dir / f"{sid}_mrconvert_gmwmi.log"
    cmd = [str(MRCONVERT), str(source), str(out), "-force"]
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"[{utc()}] $ {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
        handle.write(f"[{utc()}] rc={proc.returncode}\n\n")
    if proc.returncode != 0 or not out.exists() or out.stat().st_size <= 0:
        return None
    return out


def label_gmwmi_metrics(parc_path: Path, gmwmi_path: Path, valid_labels: list[int]) -> dict[int, dict[str, float]]:
    try:
        from scipy import ndimage as ndi
    except Exception:
        ndi = None
    try:
        parc_img = nib.load(str(parc_path))
        gmwmi_img = nib.load(str(gmwmi_path))
        parc = np.rint(np.asanyarray(parc_img.dataobj)).astype(np.int32)
        gmwmi = np.asanyarray(gmwmi_img.dataobj) > 0
    except Exception:
        return {}
    if parc.shape != gmwmi.shape:
        return {}
    distances = None
    if ndi is not None and gmwmi.any():
        spacing = tuple(float(x) for x in parc_img.header.get_zooms()[:3])
        distances = ndi.distance_transform_edt(~gmwmi, sampling=spacing)
    out: dict[int, dict[str, float]] = {}
    for label in valid_labels:
        mask = parc == int(label)
        voxels = int(mask.sum())
        if voxels <= 0:
            continue
        contact = int(np.count_nonzero(mask & gmwmi))
        row: dict[str, float] = {
            "gmwmi_label_voxels": float(voxels),
            "gmwmi_contact_voxels": float(contact),
            "gmwmi_contact_fraction": float(contact / voxels),
        }
        if distances is not None:
            vals = np.asarray(distances[mask], dtype=float)
            row.update(
                {
                    "min_distance_to_gmwmi_mm": float(np.min(vals)),
                    "median_distance_to_gmwmi_mm": float(np.median(vals)),
                    "within_2mm_gmwmi_fraction": float((vals <= 2.0).mean()),
                    "within_4mm_gmwmi_fraction": float((vals <= 4.0).mean()),
                    "within_6mm_gmwmi_fraction": float((vals <= 6.0).mean()),
                    "within_8mm_gmwmi_fraction": float((vals <= 8.0).mean()),
                }
            )
        out[int(label)] = row
    return out


def endpoint_label_audit(path: Path) -> dict[int, dict[str, str]]:
    rows = read_csv(path)
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            label = int(float(row.get("aal3_label", "")))
        except Exception:
            continue
        out[label] = row
    return out


def node_class(name: str) -> str:
    text = name.lower()
    if any(token in text for token in ("thalam", "pallid", "putamen", "caud", "accumb", "substant", "nigra")):
        return "deep_nucleus"
    if any(token in text for token in ("brainstem", "pons", "medulla", "midbrain")):
        return "brainstem"
    if any(token in text for token in ("hipp", "amyg")):
        return "limbic_subcortical"
    return "cortical_or_unknown"


def action_for_row(row: dict[str, Any]) -> tuple[str, str]:
    zero_frequency = float(row["zero_row_frequency"])
    median_voxels = float(row["median_label_voxels"])
    survival_frequency = float(row["label_survival_frequency"])
    gmwmi_eval = int(row.get("subjects_with_gmwmi_qc") or 0)
    gmwmi_contact_freq = fnum(row.get("gmwmi_contact_subject_frequency"), math.nan)
    gmwmi_min_dist = fnum(row.get("median_min_distance_to_gmwmi_mm"), math.nan)
    endpoint_no_support = fnum(row.get("selected_subjects_survives_but_no_endpoint"), 0)
    endpoint_supported = fnum(row.get("selected_subjects_with_endpoint"), 0)
    if survival_frequency < 0.80:
        return ("atlas_survival_review", "label absent in too many subject parcellations")
    if gmwmi_eval > 0 and gmwmi_contact_freq < 0.25 and math.isfinite(gmwmi_min_dist) and gmwmi_min_dist > 4.0:
        return ("merge_or_tractometry_review", "label is repeatedly far from the GMWMI endpoint surface")
    if endpoint_no_support >= 20 and endpoint_supported < endpoint_no_support:
        return ("merge_review", "survives but repeatedly receives no selected-route endpoints")
    if zero_frequency >= 0.80 and median_voxels < 50:
        return ("tractometry_only_review", "tiny recurrent zero-row label")
    if zero_frequency >= 0.50:
        return ("merge_review", "recurrently zero-row label")
    if 0 < median_voxels < 10:
        return ("merge_review", "very small native label")
    return ("keep_connectome_candidate", "passes node-level screening")


def build_node_qc(
    route1_root: Path,
    subjects: list[str],
    *,
    compute_gmwmi: bool,
    gmwmi_cache_dir: Path,
    endpoint_audit_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_labels = aal3_valid_labels()
    names = label_names()
    endpoints = endpoint_label_audit(endpoint_audit_path) if endpoint_audit_path.exists() else {}
    env = os.environ.copy()
    env["PATH"] = "/home/ec2-user/mrtrix3/bin:" + env.get("PATH", "")
    accum: dict[int, dict[str, Any]] = {
        label: {
            "aal3_label": label,
            "aal3_name": names.get(label, f"AAL3_{label:03d}"),
            "node_class": node_class(names.get(label, "")),
            "subjects_with_matrix": 0,
            "subjects_zero_row": 0,
            "subjects_with_label": 0,
            "voxel_counts": [],
            "subjects_with_gmwmi_qc": 0,
            "subjects_touching_gmwmi": 0,
            "gmwmi_contact_fractions": [],
            "min_distance_to_gmwmi_mm": [],
            "median_distance_to_gmwmi_mm": [],
            "within_2mm_gmwmi_fractions": [],
            "within_4mm_gmwmi_fractions": [],
            "within_6mm_gmwmi_fractions": [],
            "within_8mm_gmwmi_fractions": [],
        }
        for label in valid_labels
    }

    for sid in subjects:
        inputs = discover_inputs(sid)
        matrix = inputs.get("production_fd_sum")
        parc = inputs.get("production_aal_b0")
        zero_labels = zero_row_labels(Path(matrix), valid_labels) if matrix else set()
        counts = label_voxel_counts(Path(parc), valid_labels) if parc else {}
        gmwmi_metrics: dict[int, dict[str, float]] = {}
        if compute_gmwmi and parc:
            gmwmi_nifti = gmwmi_nifti_for_sid(sid, gmwmi_cache_dir, env)
            if gmwmi_nifti:
                gmwmi_metrics = label_gmwmi_metrics(Path(parc), gmwmi_nifti, valid_labels)
        for label in valid_labels:
            if matrix:
                accum[label]["subjects_with_matrix"] += 1
                if label in zero_labels:
                    accum[label]["subjects_zero_row"] += 1
            voxels = counts.get(label, 0)
            if voxels > 0:
                accum[label]["subjects_with_label"] += 1
                accum[label]["voxel_counts"].append(voxels)
            gmwmi = gmwmi_metrics.get(label)
            if gmwmi:
                accum[label]["subjects_with_gmwmi_qc"] += 1
                if int(gmwmi.get("gmwmi_contact_voxels") or 0) > 0:
                    accum[label]["subjects_touching_gmwmi"] += 1
                accum[label]["gmwmi_contact_fractions"].append(float(gmwmi.get("gmwmi_contact_fraction") or 0.0))
                for key, accum_key in (
                    ("min_distance_to_gmwmi_mm", "min_distance_to_gmwmi_mm"),
                    ("median_distance_to_gmwmi_mm", "median_distance_to_gmwmi_mm"),
                    ("within_2mm_gmwmi_fraction", "within_2mm_gmwmi_fractions"),
                    ("within_4mm_gmwmi_fraction", "within_4mm_gmwmi_fractions"),
                    ("within_6mm_gmwmi_fraction", "within_6mm_gmwmi_fractions"),
                    ("within_8mm_gmwmi_fraction", "within_8mm_gmwmi_fractions"),
                ):
                    val = fnum(gmwmi.get(key), math.nan)
                    if math.isfinite(val):
                        accum[label][accum_key].append(val)

    rows: list[dict[str, Any]] = []
    merge_template: list[dict[str, Any]] = []
    n_subjects = max(1, len(subjects))
    for label in valid_labels:
        item = accum[label]
        voxels = item.pop("voxel_counts")
        contact_fractions = item.pop("gmwmi_contact_fractions")
        min_distances = item.pop("min_distance_to_gmwmi_mm")
        median_distances = item.pop("median_distance_to_gmwmi_mm")
        within_2 = item.pop("within_2mm_gmwmi_fractions")
        within_4 = item.pop("within_4mm_gmwmi_fractions")
        within_6 = item.pop("within_6mm_gmwmi_fractions")
        within_8 = item.pop("within_8mm_gmwmi_fractions")
        matrix_n = max(1, int(item["subjects_with_matrix"]))
        gmwmi_n = max(1, int(item["subjects_with_gmwmi_qc"]))
        endpoint_row = endpoints.get(label, {})
        row = {
            **item,
            "subjects_total": len(subjects),
            "label_survival_frequency": item["subjects_with_label"] / n_subjects,
            "zero_row_frequency": item["subjects_zero_row"] / matrix_n,
            "median_label_voxels": statistics.median(voxels) if voxels else 0,
            "min_label_voxels": min(voxels) if voxels else 0,
            "max_label_voxels": max(voxels) if voxels else 0,
            "gmwmi_contact_subject_frequency": item["subjects_touching_gmwmi"] / gmwmi_n if item["subjects_with_gmwmi_qc"] else "",
            "median_gmwmi_contact_fraction": median(contact_fractions, default=""),
            "median_min_distance_to_gmwmi_mm": median(min_distances, default=""),
            "median_median_distance_to_gmwmi_mm": median(median_distances, default=""),
            "median_within_2mm_gmwmi_fraction": median(within_2, default=""),
            "median_within_4mm_gmwmi_fraction": median(within_4, default=""),
            "median_within_6mm_gmwmi_fraction": median(within_6, default=""),
            "median_within_8mm_gmwmi_fraction": median(within_8, default=""),
            "selected_subjects_with_endpoint": endpoint_row.get("selected_subjects_with_endpoint", ""),
            "selected_subjects_survives_but_no_endpoint": endpoint_row.get("survives_but_no_endpoint", ""),
            "selected_subjects_endpoint_present_but_no_weighted_edge": endpoint_row.get("endpoint_present_but_no_weighted_edge", ""),
            "selected_total_endpoint_hits": endpoint_row.get("total_endpoint_hits", ""),
            "selected_total_incident_streamlines": endpoint_row.get("total_incident_streamlines", ""),
        }
        action, reason = action_for_row(row)
        row["aal3cc_action_recommendation"] = action
        row["aal3cc_reason"] = reason
        rows.append(row)
        merge_template.append(
            {
                "old_id": label,
                "old_name": row["aal3_name"],
                "new_id": label if action == "keep_connectome_candidate" else "",
                "new_name": row["aal3_name"] if action == "keep_connectome_candidate" else "",
                "action": "keep" if action == "keep_connectome_candidate" else "review",
                "review_recommendation": action,
                "reason": reason,
                "median_label_voxels": row["median_label_voxels"],
                "zero_row_frequency": row["zero_row_frequency"],
                "gmwmi_contact_subject_frequency": row["gmwmi_contact_subject_frequency"],
                "median_min_distance_to_gmwmi_mm": row["median_min_distance_to_gmwmi_mm"],
                "selected_subjects_with_endpoint": row["selected_subjects_with_endpoint"],
            }
        )
    rows.sort(key=lambda r: (-float(r["zero_row_frequency"]), float(r["median_label_voxels"]), int(r["aal3_label"])))
    return rows, merge_template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, default=DEFAULT_R1)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("/home/ec2-user/exp/reports/aal3_node_qc"))
    parser.add_argument("--compute-gmwmi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gmwmi-cache-dir", type=Path, default=Path("/home/ec2-user/exp/reports/aal3_node_qc/gmwmi_cache"))
    parser.add_argument("--endpoint-label-audit", type=Path, default=DEFAULT_ENDPOINT_LABEL_AUDIT)
    args = parser.parse_args()

    if args.subjects:
        subjects = sorted(set(args.subjects))
    else:
        subjects = [row["sid"] for row in read_csv(args.route1_root / "subjects.csv") if row.get("sid")]

    rows, merge_template = build_node_qc(
        args.route1_root,
        subjects,
        compute_gmwmi=args.compute_gmwmi,
        gmwmi_cache_dir=args.gmwmi_cache_dir,
        endpoint_audit_path=args.endpoint_label_audit,
    )
    stamp = utc().replace(":", "").replace("-", "")
    out_dir = args.out_dir
    write_csv(out_dir / f"aal3_node_qc_{stamp}.csv", rows)
    write_csv(out_dir / f"aal3_connectome_compatible_merge_template_{stamp}.tsv", merge_template, delimiter="\t")
    write_csv(out_dir / "aal3_node_qc_latest.csv", rows)
    write_csv(out_dir / "aal3_connectome_compatible_merge_template_latest.tsv", merge_template, delimiter="\t")
    print(f"subjects={len(subjects)}")
    print(f"nodes={len(rows)}")
    print(f"wrote {out_dir / 'aal3_node_qc_latest.csv'}")
    print(f"wrote {out_dir / 'aal3_connectome_compatible_merge_template_latest.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
