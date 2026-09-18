#!/usr/bin/env python3
"""Strict, read-only summary for SC matrix upstream repair canaries.

The lane canary runner records whether a candidate route improves density or
zero rows. That is not sufficient for atlas-labelled connectomes: a candidate
can look denser simply because it produced a smaller matrix or dropped labels.

This script applies stricter gates for the current AAL3 workflow:
  * matrices must remain 170 x 170;
  * AAL label survival must be adequate;
  * AAL_007/AAL_008 should survive the parcellation transform;
  * density and unexpected valid zero rows must improve without hard regressions;
  * very low absolute density remains a failure even if it improved numerically.

It does not modify production outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_QC_ROOT = Path("/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc")
DEFAULT_CANARY = (
    DEFAULT_QC_ROOT
    / "lane_canaries"
    / "sc_lane_canary10_20260513T074853Z"
)
SUPERVISOR_MATRICES = [
    Path("/home/ec2-user/exp/SC_033_S_0908_AAL.csv"),
    Path("/home/ec2-user/exp/SC_033_S_2374_AAL.csv"),
]


def latest_canary_root() -> Path:
    latest = DEFAULT_QC_ROOT / "lane_canaries" / "latest_tag.txt"
    if latest.exists():
        tag = latest.read_text().strip()
        if tag:
            return DEFAULT_QC_ROOT / "lane_canaries" / tag
    return DEFAULT_CANARY


def find_one(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return hits[-1]


def supervisor_density_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in SUPERVISOR_MATRICES:
        rec: dict[str, Any] = {"file": str(path), "exists": path.exists()}
        if path.exists():
            try:
                arr = np.loadtxt(path)
                n = int(arr.shape[0]) if arr.ndim == 2 else 0
                upper = np.triu_indices(n, 1)
                vals = arr[upper] if n else np.array([])
                rec.update(
                    {
                        "shape": f"{arr.shape[0]}x{arr.shape[1]}",
                        "nonzero_upper_edges": int(np.count_nonzero(np.abs(vals) > 0)),
                        "possible_upper_edges": int(len(vals)),
                        "density": float(np.count_nonzero(np.abs(vals) > 0) / len(vals))
                        if len(vals)
                        else np.nan,
                        "zero_rows": int(np.isclose(arr, 0).all(axis=1).sum()),
                        "symmetry_max_abs": float(np.nanmax(np.abs(arr - arr.T))),
                        "diagonal_abs_sum": float(np.nansum(np.abs(np.diag(arr)))),
                    }
                )
            except Exception as exc:  # pragma: no cover - diagnostic path
                rec["read_error"] = str(exc)
        rows.append(rec)
    return pd.DataFrame(rows)


def current_density_summary(integrity_csv: Path) -> pd.DataFrame:
    if not integrity_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(integrity_csv)
    if "weight" in df.columns:
        df = df[df["weight"].astype(str).eq("fd_sum")].copy()
    density = pd.to_numeric(df.get("density"), errors="coerce")
    out = {
        "n_subject_matrices": int(density.notna().sum()),
        "density_mean": float(density.mean()),
        "density_median": float(density.median()),
        "density_q10": float(density.quantile(0.10)),
        "density_q25": float(density.quantile(0.25)),
        "density_q75": float(density.quantile(0.75)),
        "density_q90": float(density.quantile(0.90)),
        "density_q95": float(density.quantile(0.95)),
        "density_max": float(density.max()),
    }
    if "group" in df.columns:
        by_group = (
            df.assign(density=density)
            .groupby("group", dropna=False)["density"]
            .agg(["count", "median", "mean"])
            .reset_index()
        )
        by_group.to_csv(integrity_csv.parent / "current_fd_sum_density_by_group.csv", index=False)
    return pd.DataFrame([out])


def classify_route(row: pd.Series, args: argparse.Namespace) -> tuple[str, str]:
    reasons: list[str] = []
    if not bool(row.get("read_ok", False)):
        reasons.append("matrix_read_failed")
    if int(row.get("n_rows", 0) or 0) != args.expected_n or int(row.get("n_cols", 0) or 0) != args.expected_n:
        reasons.append("matrix_not_170x170")
    if int(row.get("labels_ge_50_vox", 0) or 0) < args.label_warn_min:
        reasons.append("poor_label_survival")
    elif int(row.get("labels_ge_50_vox", 0) or 0) < args.label_strong_min:
        reasons.append("borderline_label_survival")
    if int(row.get("AAL_007_voxels", 0) or 0) < args.frontal_label_min:
        reasons.append("AAL_007_tiny_or_absent")
    if int(row.get("AAL_008_voxels", 0) or 0) < args.frontal_label_min:
        reasons.append("AAL_008_tiny_or_absent")
    m5 = pd.to_numeric(row.get("mask_5tt_overlap_fraction"), errors="coerce")
    if pd.notna(m5) and m5 < args.mask_5tt_min:
        reasons.append("very_low_5tt_overlap")
    density = pd.to_numeric(row.get("density"), errors="coerce")
    if pd.isna(density) or density < args.min_abs_density:
        reasons.append("density_below_repair_gate")
    d_delta = pd.to_numeric(row.get("density_delta_vs_baseline"), errors="coerce")
    if pd.notna(d_delta) and d_delta < -args.density_drop_tol:
        reasons.append("density_worse_than_baseline")
    z_delta = pd.to_numeric(row.get("unexpected_valid_zero_rows_delta_vs_baseline"), errors="coerce")
    if pd.notna(z_delta) and z_delta > args.zero_row_increase_tol:
        reasons.append("zero_rows_worse_than_baseline")
    zero_rows = pd.to_numeric(row.get("unexpected_valid_zero_rows"), errors="coerce")
    if pd.notna(zero_rows) and zero_rows > args.max_unexpected_zero_rows:
        reasons.append("too_many_unexpected_zero_rows")

    hard = {
        "matrix_read_failed",
        "matrix_not_170x170",
        "poor_label_survival",
        "AAL_007_tiny_or_absent",
        "AAL_008_tiny_or_absent",
        "very_low_5tt_overlap",
        "density_below_repair_gate",
        "density_worse_than_baseline",
        "zero_rows_worse_than_baseline",
        "too_many_unexpected_zero_rows",
    }
    status = "CANDIDATE_PASS"
    if any(r in hard for r in reasons):
        status = "CANDIDATE_FAIL"
    elif "borderline_label_survival" in reasons:
        status = "CANDIDATE_WARN"
    return status, ";".join(reasons) if reasons else "passes_strict_route_gate"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canary-root", type=Path, default=latest_canary_root())
    ap.add_argument("--expected-n", type=int, default=170)
    ap.add_argument("--label-strong-min", type=int, default=150)
    ap.add_argument("--label-warn-min", type=int, default=140)
    ap.add_argument("--frontal-label-min", type=int, default=50)
    ap.add_argument("--mask-5tt-min", type=float, default=0.05)
    ap.add_argument("--min-abs-density", type=float, default=0.05)
    ap.add_argument("--max-unexpected-zero-rows", type=int, default=40)
    ap.add_argument("--density-drop-tol", type=float, default=1e-9)
    ap.add_argument("--zero-row-increase-tol", type=float, default=0.0)
    args = ap.parse_args()

    root = args.canary_root
    labels_csv = find_one(root, "reference_parcellation_pilot_labels.csv")
    matrices_csv = find_one(root, "reference_parcellation_pilot_matrices.csv")
    out_dir = root / "strict_upstream_qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(labels_csv)
    matrices = pd.read_csv(matrices_csv)
    primary = matrices[matrices["metric"].astype(str).eq("fd_sum")].copy()
    merged = primary.merge(
        labels.drop(columns=["read_error"], errors="ignore"),
        on=["sid", "route"],
        how="left",
        suffixes=("", "_label"),
    )
    statuses = merged.apply(lambda r: classify_route(r, args), axis=1)
    merged["strict_status"] = [x[0] for x in statuses]
    merged["strict_reason"] = [x[1] for x in statuses]
    merged["matrix_n_ok"] = (
        pd.to_numeric(merged["n_rows"], errors="coerce").eq(args.expected_n)
        & pd.to_numeric(merged["n_cols"], errors="coerce").eq(args.expected_n)
    )
    merged["label_survival_ok"] = pd.to_numeric(
        merged["labels_ge_50_vox"], errors="coerce"
    ).ge(args.label_warn_min)
    merged["density_gate_ok"] = pd.to_numeric(merged["density"], errors="coerce").ge(
        args.min_abs_density
    )
    merged.to_csv(out_dir / "strict_upstream_route_qc.csv", index=False)

    # Per-subject recommendation. Exclude production_current from candidate choice.
    candidates = merged[~merged["route"].astype(str).eq("production_current")].copy()
    rank_map = {"CANDIDATE_PASS": 0, "CANDIDATE_WARN": 1, "CANDIDATE_FAIL": 2}
    candidates["_status_rank"] = candidates["strict_status"].map(rank_map).fillna(9)
    candidates["_density"] = pd.to_numeric(candidates["density"], errors="coerce").fillna(-1)
    candidates["_zero_rows"] = pd.to_numeric(
        candidates["unexpected_valid_zero_rows"], errors="coerce"
    ).fillna(9999)
    best = (
        candidates.sort_values(
            ["sid", "_status_rank", "_density", "_zero_rows"],
            ascending=[True, True, False, True],
        )
        .groupby("sid", as_index=False)
        .head(1)
        .copy()
    )
    base = merged[merged["route"].astype(str).eq("production_current")][
        [
            "sid",
            "density",
            "unexpected_valid_zero_rows",
            "labels_ge_50_vox",
            "AAL_007_voxels",
            "AAL_008_voxels",
            "mask_5tt_overlap_fraction",
        ]
    ].rename(
        columns={
            "density": "baseline_density",
            "unexpected_valid_zero_rows": "baseline_unexpected_valid_zero_rows",
            "labels_ge_50_vox": "baseline_labels_ge_50_vox",
            "AAL_007_voxels": "baseline_AAL_007_voxels",
            "AAL_008_voxels": "baseline_AAL_008_voxels",
            "mask_5tt_overlap_fraction": "baseline_mask_5tt_overlap_fraction",
        }
    )
    subject = best.merge(base, on="sid", how="left")
    subject["recommended_action"] = np.where(
        subject["strict_status"].eq("CANDIDATE_PASS"),
        "eligible_for_canary_repair_validation",
        np.where(
            subject["strict_reason"].str.contains("poor_label_survival|AAL_007|AAL_008", na=False),
            "debug_registration_and_label_transform_before_batch",
            np.where(
                subject["strict_reason"].str.contains("density_below_repair_gate|too_many_unexpected", na=False),
                "debug_assignment_tracks_or_5tt_overlap_after_label_fix",
                "manual_review_before_batch",
            ),
        ),
    )
    subject.to_csv(out_dir / "strict_upstream_subject_decisions.csv", index=False)

    sup = supervisor_density_summary()
    sup.to_csv(out_dir / "supervisor_reference_density_summary.csv", index=False)

    cur = current_density_summary(DEFAULT_QC_ROOT / "sc_matrix_integrity_subjects.csv")
    cur.to_csv(out_dir / "current_fd_sum_density_summary.csv", index=False)

    route_summary = (
        merged.groupby(["route", "strict_status"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["route", "strict_status"])
    )
    route_summary.to_csv(out_dir / "strict_upstream_route_summary.csv", index=False)

    report = []
    report.append("Strict upstream SC matrix canary QC")
    report.append(f"canary_root={root}")
    report.append("")
    report.append("Supervisor reference matrices")
    if not sup.empty:
        for _, r in sup.iterrows():
            if r.get("exists"):
                report.append(
                    f"- {Path(str(r['file'])).name}: shape={r.get('shape')}, "
                    f"density={r.get('density'):.3f}, zero_rows={int(r.get('zero_rows', 0))}"
                )
    report.append("")
    if not cur.empty:
        r = cur.iloc[0]
        report.append(
            "Current AAL3 fd_sum density distribution: "
            f"N={int(r['n_subject_matrices'])}, median={r['density_median']:.3f}, "
            f"q75={r['density_q75']:.3f}, q95={r['density_q95']:.3f}, max={r['density_max']:.3f}"
        )
    report.append("")
    report.append("Strict route status counts")
    for _, r in route_summary.iterrows():
        report.append(f"- {r['route']}: {r['strict_status']}={int(r['n'])}")
    report.append("")
    report.append("Subject decisions")
    for _, r in subject.sort_values("sid").iterrows():
        report.append(
            f"- {r['sid']}: best={r['route']} status={r['strict_status']} "
            f"density={float(r['density']):.4f} zero_rows={int(r['unexpected_valid_zero_rows'])} "
            f"labels_ge50={int(r['labels_ge_50_vox'])} action={r['recommended_action']}"
        )
    report.append("")
    report.append(
        "Interpretation: density is not expected to be 100%. The supervisor AAL116 matrices "
        "are healthy but not complete graphs. For the current AAL3 pipeline, a candidate route "
        "must keep 170x170 shape and materially reduce valid zero rows; very low density remains "
        "a hard failure even if numerically improved."
    )
    (out_dir / "strict_upstream_summary.txt").write_text("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"\nWrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
