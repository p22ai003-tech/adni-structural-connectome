#!/usr/bin/env python3
"""Build manuscript-only figures from locked aggregate result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "stage_specific_diffusivity_v2" / "sensitivity_contrasts.csv"
OUT = ROOT / "figures" / "author_ready_sensitivity_forest.png"

LABELS = {
    "all_available": "All available",
    "at_least_9_networks": "At least 9 systems",
    "complete_17_networks": "Complete 17 systems",
    "timing_le90": "DTI-T1 interval ≤90 days",
    "timing_le180": "DTI-T1 interval ≤180 days",
    "same_adni_phase": "Same ADNI phase",
    "siemens54": "Siemens 54-direction",
    "original_t1": "Original T1 only",
    "legacy_qc": "Historical QC subset",
}
ORDER = list(LABELS)
COLORS = {"axial_MCI_vs_CN": "#2E6F9E", "radial_AD_vs_MCI": "#B7357B"}
TITLES = {
    "axial_MCI_vs_CN": "AxD: MCI minus CN",
    "radial_AD_vs_MCI": "RD: AD minus MCI",
}


def main() -> None:
    frame = pd.read_csv(SOURCE)
    frame = frame[frame["stratum"].isin(ORDER)].copy()
    frame["stratum"] = pd.Categorical(frame["stratum"], ORDER, ordered=True)
    frame = frame.sort_values("stratum")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), sharey=True)
    y = np.arange(len(ORDER))
    for ax, endpoint in zip(axes, ("axial_MCI_vs_CN", "radial_AD_vs_MCI")):
        part = frame[frame["endpoint"] == endpoint].set_index("stratum").loc[ORDER]
        values = part["beta"].to_numpy(float)
        low = part["ci95_low"].to_numpy(float)
        high = part["ci95_high"].to_numpy(float)
        passes = part["holm_p_two_endpoints"].to_numpy(float) < 0.05
        color = COLORS[endpoint]

        ax.axvline(0, color="#333333", lw=1.1, zorder=0)
        ax.errorbar(
            values,
            y,
            xerr=np.vstack((values - low, high - values)),
            fmt="none",
            ecolor="#8A9298",
            elinewidth=1.8,
            capsize=3.5,
            zorder=1,
        )
        ax.scatter(values[passes], y[passes], s=62, c=color, edgecolors=color, zorder=3)
        ax.scatter(
            values[~passes],
            y[~passes],
            s=62,
            facecolors="white",
            edgecolors=color,
            linewidths=1.8,
            zorder=3,
        )
        for yi, n in zip(y, part["n"].astype(int)):
            ax.text(0.98, yi, f"n={n}", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=8.5, color="#555555")
        ax.set_title(TITLES[endpoint], fontsize=15, pad=10)
        ax.set_xlabel("Adjusted rank-normal beta (95% CI)", fontsize=11)
        ax.grid(axis="x", color="#E4E7E9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_yticks(y, [LABELS[item] for item in ORDER], fontsize=10)
    axes[0].invert_yaxis()
    fig.suptitle("Robustness across completeness and acquisition-restricted analyses",
                 fontsize=17, y=0.98)
    fig.text(
        0.5,
        0.012,
        "Filled points: the endpoint remains significant within the jointly Holm-corrected two-endpoint family; "
        "open points: it does not.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.03, 0.055, 0.99, 0.94), w_pad=2.8)
    fig.savefig(OUT, dpi=320, bbox_inches="tight", facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
