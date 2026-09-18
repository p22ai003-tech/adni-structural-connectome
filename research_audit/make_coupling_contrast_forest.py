#!/usr/bin/env python3
"""Create the manuscript forest plot for the coupling falsification.

The source is the immutable participant-level contrast release.  This script
does not refit models or read image data; it visualizes the full-exact
MCI-versus-CN and AD-versus-MCI contrasts that define the rejected inverted-U
candidate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path("/home/ec2-user/exp")
SOURCE = (
    PROJECT
    / "research_audit"
    / "outputs"
    / "coupling_support_falsification_v1"
    / "coupling_support_contrasts.csv"
)
DESTINATION = PROJECT / "research_audit" / "figures"

METRICS = [
    ("raw_nodal_eff_fa", "Nodal efficiency-FA: raw"),
    ("partial_degree_nodal_eff_fa", "Nodal efficiency-FA: degree adjusted"),
    (
        "partial_degree_fa_edges_nodal_eff_fa",
        "Nodal efficiency-FA: degree + FA support adjusted",
    ),
    ("raw_strength_fa", "Strength-FA: raw"),
    ("partial_degree_strength_fa", "Strength-FA: degree adjusted"),
    (
        "partial_degree_fa_edges_strength_fa",
        "Strength-FA: degree + FA support adjusted",
    ),
]
CONTRASTS = [
    ("MCI_vs_CN", "MCI - CN"),
    ("AD_vs_MCI", "AD - MCI"),
]
STRATA = [
    ("full_exact", "Full exact cohort"),
    (
        "timing_le90_siemens54_overlap_sites",
        "≤90 days, Siemens-54, overlap sites",
    ),
]


def main() -> None:
    data = pd.read_csv(SOURCE)
    required = {
        "metric",
        "stratum",
        "contrast",
        "n",
        "rank_normal_beta",
        "ci95_low",
        "ci95_high",
    }
    absent = sorted(required - set(data.columns))
    if absent:
        raise ValueError(f"Missing required columns: {absent}")

    view = data[
        data["stratum"].isin([item[0] for item in STRATA])
        & data["metric"].isin([item[0] for item in METRICS])
        & data["contrast"].isin([item[0] for item in CONTRASTS])
    ].copy()
    expected = len(METRICS) * len(CONTRASTS) * len(STRATA)
    if len(view) != expected or view.duplicated(
        ["metric", "stratum", "contrast"]
    ).any():
        raise ValueError(
            f"Expected {expected} unique metric-stratum-contrast rows; found {len(view)}"
        )
    numeric = view[["rank_normal_beta", "ci95_low", "ci95_high"]].to_numpy(float)
    if not np.isfinite(numeric).all() or not (
        (view["ci95_low"] <= view["rank_normal_beta"])
        & (view["rank_normal_beta"] <= view["ci95_high"])
    ).all():
        raise ValueError("Contrast estimates or confidence intervals are invalid")

    lookup = view.set_index(["metric", "stratum", "contrast"])
    y = np.arange(len(METRICS))[::-1]
    labels = []
    for metric, label in METRICS:
        ns = []
        for stratum, _ in STRATA:
            stratum_ns = sorted(
                {
                    int(lookup.loc[(metric, stratum, contrast), "n"])
                    for contrast, _ in CONTRASTS
                }
            )
            if len(stratum_ns) != 1:
                raise ValueError(
                    f"Contrasts use different n for {metric}/{stratum}: {stratum_ns}"
                )
            ns.append(stratum_ns[0])
        labels.append(f"{label}  (n full/strict={ns[0]}/{ns[1]})")

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.2), sharey=True)
    for row_index, (stratum, stratum_title) in enumerate(STRATA):
        for column_index, (contrast, contrast_title) in enumerate(CONTRASTS):
            axis = axes[row_index, column_index]
            beta = np.array(
                [
                    lookup.loc[(metric, stratum, contrast), "rank_normal_beta"]
                    for metric, _ in METRICS
                ]
            )
            low = np.array(
                [
                    lookup.loc[(metric, stratum, contrast), "ci95_low"]
                    for metric, _ in METRICS
                ]
            )
            high = np.array(
                [
                    lookup.loc[(metric, stratum, contrast), "ci95_high"]
                    for metric, _ in METRICS
                ]
            )
            colors = [
                "#E09F3E" if metric.startswith("raw_") else "#3A86FF"
                for metric, _ in METRICS
            ]
            for position, estimate, lo, hi, color in zip(y, beta, low, high, colors):
                axis.errorbar(
                    estimate,
                    position,
                    xerr=np.array([[estimate - lo], [hi - estimate]]),
                    fmt="o",
                    color=color,
                    ecolor=color,
                    elinewidth=2,
                    capsize=4,
                    markersize=7,
                )
            axis.axvline(0, color="#333333", linewidth=1.2, linestyle="--")
            axis.set_title(
                f"{stratum_title}\n{contrast_title}", fontsize=12.5, weight="bold"
            )
            if row_index == len(STRATA) - 1:
                axis.set_xlabel("Adjusted inverse-normal-scale beta (95% CI)", fontsize=11)
            axis.grid(axis="x", color="#D9E1E8", linewidth=0.8, alpha=0.8)
            axis.set_axisbelow(True)
    axes[0, 0].set_yticks(y, labels, fontsize=9.5)
    axes[1, 0].set_yticks(y, labels, fontsize=9.5)
    axes[0, 1].tick_params(axis="y", labelleft=False)
    axes[1, 1].tick_params(axis="y", labelleft=False)
    fig.suptitle(
        "Exploratory frontal topology-FA coupling falsification",
        fontsize=16,
        weight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.012,
        "Orange: raw coupling. Blue: within-participant support-adjusted coupling. "
        "Historical outputs; site-clustered uncertainty; no corrected-processing confirmation.",
        ha="center",
        fontsize=9.5,
    )
    fig.subplots_adjust(
        left=0.40, right=0.98, top=0.88, bottom=0.09, wspace=0.17, hspace=0.32
    )

    DESTINATION.mkdir(parents=True, exist_ok=True)
    fig.savefig(DESTINATION / "coupling_support_contrast_forest.png", dpi=240)
    fig.savefig(DESTINATION / "coupling_support_contrast_forest.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
