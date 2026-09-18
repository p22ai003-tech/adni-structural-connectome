#!/usr/bin/env python3
"""Strict upstream QC summary for SC matrix parcellation canaries.

This is read-only.  It re-evaluates a reference/parcellation pilot with hard
atlas-size and label-survival gates so apparent density gains from shrunken
matrices are not mistaken for valid repairs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RUN_ROOT = Path(
    "/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc/lane_canaries/"
    "sc_lane_canary10_20260513T074853Z/registration_parc_pilot/"
    "sc_lane_canary10_20260513T074853Z_registration_parcellation"
)


def _bool_reason(parts: list[str]) -> str:
    return "; ".join(parts) if parts else "passes strict route gates"


def load_tables(run_root: Path) -> pd.DataFrame:
    labels_path = run_root / "reference_parcellation_pilot_labels.csv"
    matrices_path = run_root / "reference_parcellation_pilot_matrices.csv"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    if not matrices_path.exists():
        raise FileNotFoundError(matrices_path)

    labels = pd.read_csv(labels_path)
    matrices = pd.read_csv(matrices_path)
    matrices = matrices[matrices["metric"].eq("fd_sum")].copy()
    merged = matrices.merge(
        labels,
        on=["sid", "route"],
        how="left",
        suffixes=("_matrix", "_label"),
    )
    return merged


def classify_routes(
    df: pd.DataFrame,
    *,
    expected_n: int,
    min_labels_ge50: int,
    min_valid_labels: int,
    min_density_for_batch: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        reasons: list[str] = []
        n_rows = int(row.get("n_rows", -1)) if pd.notna(row.get("n_rows")) else -1
        n_cols = int(row.get("n_cols", -1)) if pd.notna(row.get("n_cols")) else -1
        labels_ge50 = int(row.get("labels_ge_50_vox", -1)) if pd.notna(row.get("labels_ge_50_vox")) else -1
        valid_labels = int(row.get("valid_labels_present", -1)) if pd.notna(row.get("valid_labels_present")) else -1
        aal7 = int(row.get("AAL_007_voxels", -1)) if pd.notna(row.get("AAL_007_voxels")) else -1
        aal8 = int(row.get("AAL_008_voxels", -1)) if pd.notna(row.get("AAL_008_voxels")) else -1
        density = float(row.get("density", np.nan))
        baseline_density = float(row.get("density_baseline", np.nan))
        zero_rows = float(row.get("unexpected_valid_zero_rows", np.nan))
        baseline_zero_rows = float(row.get("unexpected_valid_zero_rows_baseline", np.nan))
        mask_5tt_overlap = row.get("mask_5tt_overlap_fraction", np.nan)

        if n_rows != expected_n or n_cols != expected_n:
            reasons.append(f"matrix_size_{n_rows}x{n_cols}_not_{expected_n}x{expected_n}")
        if labels_ge50 < min_labels_ge50:
            reasons.append(f"low_label_survival_ge50={labels_ge50}")
        if valid_labels < min_valid_labels:
            reasons.append(f"low_valid_label_count={valid_labels}")
        if aal7 < 50 or aal8 < 50:
            reasons.append(f"frontal_opercular_labels_tiny=AAL7:{aal7},AAL8:{aal8}")
        if np.isfinite(baseline_density) and np.isfinite(density) and density < baseline_density:
            reasons.append(f"density_worse={density:.6f}<{baseline_density:.6f}")
        if np.isfinite(baseline_zero_rows) and np.isfinite(zero_rows) and zero_rows > baseline_zero_rows:
            reasons.append(f"zero_rows_worse={zero_rows}>{baseline_zero_rows}")

        route_gate = "ROUTE_STRICT_OK" if not reasons else "ROUTE_REJECT"
        if route_gate == "ROUTE_STRICT_OK" and density < min_density_for_batch:
            batch_gate = "IMPROVES_BUT_TOO_SPARSE_FOR_BATCH"
        elif route_gate == "ROUTE_STRICT_OK":
            batch_gate = "BATCH_CANDIDATE"
        else:
            batch_gate = "NOT_BATCH_ELIGIBLE"

        rows.append(
            {
                "sid": row["sid"],
                "route": row["route"],
                "n_rows": n_rows,
                "n_cols": n_cols,
                "density": density,
                "density_baseline": baseline_density,
                "density_delta": row.get("density_delta_vs_baseline", np.nan),
                "unexpected_valid_zero_rows": zero_rows,
                "baseline_unexpected_valid_zero_rows": baseline_zero_rows,
                "zero_row_delta": row.get("unexpected_valid_zero_rows_delta_vs_baseline", np.nan),
                "valid_labels_present": valid_labels,
                "labels_ge_50_vox": labels_ge50,
                "AAL_007_voxels": aal7,
                "AAL_008_voxels": aal8,
                "mask_b0_overlap_fraction": row.get("mask_b0_overlap_fraction", np.nan),
                "mask_5tt_overlap_fraction": mask_5tt_overlap,
                "strict_route_gate": route_gate,
                "batch_gate": batch_gate,
                "reject_reason": _bool_reason(reasons),
            }
        )
    return pd.DataFrame(rows)


def subject_decisions(route_qc: pd.DataFrame) -> pd.DataFrame:
    decisions: list[dict[str, object]] = []
    for sid, sub in route_qc.groupby("sid", sort=True):
        ok = sub[sub["strict_route_gate"].eq("ROUTE_STRICT_OK")].copy()
        if ok.empty:
            best = sub.sort_values(
                ["labels_ge_50_vox", "density", "unexpected_valid_zero_rows"],
                ascending=[False, False, True],
            ).iloc[0]
            decision = "UPSTREAM_REGISTRATION_MASK_DEBUG_REQUIRED"
            next_action = (
                "Do not batch. Label survival or matrix size is still invalid; "
                "debug T1/B0 registration, 5TT/DWI mask fit, and AAL transform inputs."
            )
        else:
            best = ok.sort_values(
                ["density", "unexpected_valid_zero_rows", "labels_ge_50_vox"],
                ascending=[False, True, False],
            ).iloc[0]
            if best["batch_gate"] == "BATCH_CANDIDATE":
                decision = "STRICT_CANARY_PASS"
                next_action = "Eligible for small production canary with backup and validation."
            else:
                decision = "STRICT_ROUTE_IMPROVES_BUT_TOO_SPARSE"
                next_action = (
                    "Route is anatomically plausible but matrix remains sparse; "
                    "run assignment/tracks diagnostics before production repair."
                )
        decisions.append(
            {
                "sid": sid,
                "decision": decision,
                "best_strict_route": best["route"],
                "best_density": best["density"],
                "baseline_density": best["density_baseline"],
                "best_unexpected_valid_zero_rows": best["unexpected_valid_zero_rows"],
                "baseline_unexpected_valid_zero_rows": best["baseline_unexpected_valid_zero_rows"],
                "best_labels_ge_50_vox": best["labels_ge_50_vox"],
                "best_valid_labels_present": best["valid_labels_present"],
                "best_AAL_007_voxels": best["AAL_007_voxels"],
                "best_AAL_008_voxels": best["AAL_008_voxels"],
                "best_mask_5tt_overlap_fraction": best["mask_5tt_overlap_fraction"],
                "next_action": next_action,
            }
        )
    return pd.DataFrame(decisions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--expected-n", type=int, default=170)
    parser.add_argument("--min-labels-ge50", type=int, default=140)
    parser.add_argument("--min-valid-labels", type=int, default=150)
    parser.add_argument("--min-density-for-batch", type=float, default=0.05)
    args = parser.parse_args()

    merged = load_tables(args.run_root)
    route_qc = classify_routes(
        merged,
        expected_n=args.expected_n,
        min_labels_ge50=args.min_labels_ge50,
        min_valid_labels=args.min_valid_labels,
        min_density_for_batch=args.min_density_for_batch,
    )
    decisions = subject_decisions(route_qc)

    route_out = args.run_root / "upstream_debug_strict_route_qc.csv"
    decision_out = args.run_root / "upstream_debug_subject_decisions.csv"
    route_qc.to_csv(route_out, index=False)
    decisions.to_csv(decision_out, index=False)

    print(f"Wrote {route_out}")
    print(f"Wrote {decision_out}")
    print("\nDecision counts:")
    print(decisions["decision"].value_counts().to_string())
    print("\nSubject decisions:")
    cols = [
        "sid",
        "decision",
        "best_strict_route",
        "baseline_density",
        "best_density",
        "baseline_unexpected_valid_zero_rows",
        "best_unexpected_valid_zero_rows",
        "best_labels_ge_50_vox",
        "best_mask_5tt_overlap_fraction",
    ]
    print(decisions[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
