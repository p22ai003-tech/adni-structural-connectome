#!/usr/bin/env python3
"""Build publication figures for the evidence-complete manuscript V2."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/exp/research_audit")
FIG = ROOT / "figures"
EXT = ROOT / "outputs" / "manuscript_multiscale_extension_v2"
EXPANDED = ROOT / "outputs" / "expanded_dimension_screen_v1" / "adjusted_candidate_tests.csv"

BLUE = "#2457A6"
CYAN = "#3A89C9"
GREEN = "#2E7D5B"
AMBER = "#C17A18"
RED = "#A33A3A"
SLATE = "#526274"
LIGHT = "#F3F6F9"
INK = "#17212B"


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def evidence_flow() -> None:
    fig, ax = plt.subplots(figsize=(17, 9.2))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 9.2)
    ax.axis("off")
    ax.text(
        0.2,
        8.75,
        "One positive evidence chain links images, systems, centres and prediction",
        fontsize=22,
        weight="bold",
        color=INK,
    )
    ax.text(
        0.2,
        8.32,
        "Each layer contributes a distinct role to the same jointly controlled adjacent-stage phenotype",
        fontsize=12.5,
        color=SLATE,
    )

    boxes = [
        (0.35, 5.30, 3.60, 2.25, "1  Images to connectomes", "DTI + T1 → 166-node graphs", "MRtrix3/FSL/ANTs processing\nAAL3 systems + SIFT2 weights\nFA, MD, AxD and RD summaries", SLATE, "PIPELINE"),
        (4.48, 5.30, 3.60, 2.25, "2  Tissue context", "Conventional whole-brain MD", "MCI–CN Holm₁₂=.030\nAD–CN Holm₁₂<.001\nexpected diffuse abnormality", GREEN, "CONTEXT"),
        (8.61, 5.30, 3.60, 2.25, "3  Adjacent-stage axes", "One jointly controlled family", "AxD MCI–CN wild-Holm=.0046\nRD AD–MCI wild-Holm=.0164\npair-common scopes also pass", BLUE, "PRIMARY"),
        (12.74, 5.30, 3.60, 2.25, "4  Distributed expression", "17 nonduplicate systems", "AxD: 17/17 FDR-significant\nRD: 13/17 FDR-significant\nwhole-system participation", CYAN, "SYSTEMS"),
        (12.74, 1.43, 3.60, 2.25, "5  Centre robustness", "Paired leave-one-site-out", "51/51 retain both directions\n51/51 retain both nominal p<.05\nno single-centre control", BLUE, "ROBUST"),
        (8.61, 1.43, 3.60, 2.25, "6  Internal transportability", "Nested site-held-out late RD", "systems AUC=.630\nΔ baseline=.193 [.077, .287]\nnetwork-over-scalar uncertain", GREEN, "COHERENCE"),
        (4.48, 1.43, 3.60, 2.25, "7  Integrated V2 phenotype", "Stage × systems × centres", "distributed two-axis candidate\npost-hoc internal discovery\nnot a cellular switch or biomarker", CYAN, "SYNTHESIS"),
        (0.35, 1.43, 3.60, 2.25, "8  V-final confirmation gate", "Uniform corrected processing", "Repeat frozen endpoints after QC\nno outcome-guided retuning\nconfirm, qualify or reject", AMBER, "PENDING"),
    ]

    for x, y, w, h, title, subtitle, body, color, badge in boxes:
        badge_width = min(w - 0.36, max(1.28, 0.105 * len(badge) + 0.52))
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.10",
            linewidth=1.6,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.18, y + h - 0.43),
                badge_width,
                0.28,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=0,
                facecolor=color,
            )
        )
        ax.text(x + 0.18 + badge_width / 2, y + h - 0.29, badge, ha="center", va="center", fontsize=8.2, weight="bold", color="white")
        ax.text(x + 0.20, y + h - 0.78, title, fontsize=12.4, weight="bold", color=INK)
        ax.text(x + 0.20, y + h - 1.08, subtitle, fontsize=10.2, weight="bold", color=color)
        ax.text(x + 0.20, y + h - 1.43, body, fontsize=9.4, color=INK, va="top", linespacing=1.32)

    arrows = [
        ((3.95, 6.43), (4.48, 6.43)),
        ((8.08, 6.43), (8.61, 6.43)),
        ((12.21, 6.43), (12.74, 6.43)),
        ((14.54, 5.30), (14.54, 3.68)),
        ((12.74, 2.55), (12.21, 2.55)),
        ((8.61, 2.55), (8.08, 2.55)),
        ((4.48, 2.55), (3.95, 2.55)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=17,
                linewidth=1.8,
                color="#7B8794",
            )
        )
    ax.text(
        8.5,
        0.50,
        "Alternative graph, edge, selectivity and construct-validity tests remain fully reported in the supplement; they constrain interpretation without becoming the main narrative.",
        ha="center",
        fontsize=11,
        color=SLATE,
        style="italic",
    )
    save(fig, "multiscale_evidence_flow_v2")


def forest_axis(ax, frame: pd.DataFrame, labels: list[str], title: str, p_column: str) -> None:
    y = np.arange(len(frame))[::-1]
    colors = [BLUE if value <= 0.05 else SLATE for value in frame[p_column]]
    for index, (_, row) in enumerate(frame.iterrows()):
        yi = y[index]
        ax.plot([row["ci95_low"], row["ci95_high"]], [yi, yi], color=colors[index], lw=2)
        marker_face = colors[index] if row[p_column] <= 0.05 else "white"
        ax.scatter(row["beta"], yi, s=62, facecolor=marker_face, edgecolor=colors[index], linewidth=1.7, zorder=3)
        ax.text(
            ax.get_xlim()[1] if False else row["ci95_high"] + 0.035,
            yi,
            f"pₐ={row[p_column]:.3g}",
            va="center",
            fontsize=8.4,
            color=colors[index],
        )
    ax.axvline(0, color="#949DA6", lw=1, ls="--")
    ax.set_yticks(y, labels)
    ax.set_title(title, loc="left", fontsize=14, weight="bold", color=INK)
    ax.set_xlabel("Adjusted rank-normal contrast (left group minus right group)")
    ax.grid(axis="x", color="#E5E9ED", lw=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=9)


def microstructure_forest() -> None:
    frame = pd.read_csv(EXT / "global_microstructure_site_tests.csv")
    frame = frame[
        (frame["data_validity"] == "hard_range_valid")
        & (frame["scope"] == "full_site_fe")
    ].copy()
    metric_order = {"FA": 0, "MD": 1, "AxD": 2, "RD": 3}
    contrast_order = {"MCI_vs_CN": 0, "AD_vs_MCI": 1, "AD_vs_CN": 2}
    frame["order"] = frame["metric"].map(metric_order) * 10 + frame["contrast"].map(contrast_order)
    frame = frame.sort_values("order")
    labels = [f"{row.metric}  {row.contrast.replace('_vs_', '–')}" for row in frame.itertuples()]
    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    forest_axis(ax, frame, labels, "Whole-brain tensor context under one 12-test family", "holm_p_12_within_validity_scope")
    ax.set_xlim(-0.95, 1.30)
    ax.text(
        0.01,
        -0.16,
        "Hard-range-valid historical summaries; age, sex, DTI–T1 interval, T1 source and imaging-centre fixed effects; site-clustered CR1 inference. Filled markers pass Holm correction across all 12 rows.",
        transform=ax.transAxes,
        fontsize=9.3,
        color=SLATE,
        wrap=True,
    )
    save(fig, "whole_brain_microstructure_forest_v2")


def graph_forest() -> None:
    global_frame = pd.read_csv(EXPANDED)
    global_frame = global_frame[
        (global_frame["domain"] == "global_topology")
        & global_frame["contrast"].isin(["MCI-CN", "AD-MCI"])
    ].copy()
    global_frame = global_frame.rename(
        columns={
            "ci95_low": "ci95_low",
            "ci95_high": "ci95_high",
            "holm_p_within_domain": "adjusted_p",
        }
    )
    global_frame["contrast_label"] = global_frame["contrast"].str.replace("-", "–", regex=False)
    global_frame = global_frame.sort_values(["metric", "contrast"])
    global_labels = [f"{row.metric.replace('_', ' ')}  {row.contrast_label}" for row in global_frame.itertuples()]

    system = pd.read_csv(EXT / "system_graph_composite_site_tests.csv")
    system = system[
        (system["scope"] == "full_site_fe")
        & system["contrast"].isin(["MCI_vs_CN", "AD_vs_MCI"])
    ].copy()
    system = system.rename(columns={"holm_p_9_within_scope": "adjusted_p"})
    system = system.sort_values(["metric", "contrast"])
    system_labels = [f"{row.metric.replace('_', ' ')}  {row.contrast.replace('_vs_', '–')}" for row in system.itertuples()]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.8), gridspec_kw={"wspace": 0.52})
    forest_axis(axes[0], global_frame, global_labels, "A  Global graph measures", "adjusted_p")
    axes[0].set_xlim(-0.75, 0.80)
    forest_axis(axes[1], system, system_labels, "B  Seventeen-system nodal graph composites", "adjusted_p")
    axes[1].set_xlim(-0.75, 0.80)
    fig.suptitle("Graph topology does not add a corrected adjacent-stage axis", fontsize=19, weight="bold", color=INK, y=1.02)
    fig.text(
        0.5,
        -0.02,
        "Global metrics: 0/8 corrected adjacent contrasts. System-aggregated degree, strength and nodal efficiency: 0/9 corrected tests across all three contrasts. Historical graph measures remain density- and recipe-dependent.",
        ha="center",
        fontsize=10,
        color=SLATE,
    )
    save(fig, "graph_topology_forest_v2")


def derived_validity() -> None:
    rows = [
        ("Brain age", "AD–MCI age-corrected BAG Holm=.0466", "Held-out age R²=.0095", "REJECT CONSTRUCT", RED),
        ("Fixed-velocity delay", "Best raw p=.0326; Holm=.391", "Length divided by fixed 6 mm/ms", "REJECT PHYSIOLOGY", RED),
        ("Profile dispersion", "Best Holm=.666; complete AD n=27", "Post-selection and coverage restricted", "EXPLORATORY ONLY", SLATE),
        ("Spectra / resilience", "No frozen historical claim test", "Density/QC sensitive; prior art occupied", "DEFER TO CORRECTED", AMBER),
        ("Propagation", "No scan-aligned regional target", "Needs atrophy or amyloid/tau seeds", "NOT ESTIMABLE", AMBER),
    ]
    fig, ax = plt.subplots(figsize=(16, 6.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.text(0.2, 6.0, "Candidate third dimensions: validity precedes significance", fontsize=20, weight="bold", color=INK)
    headers = [(0.35, "Construct"), (3.35, "Observed statistical evidence"), (8.05, "Validity test"), (12.6, "Decision")]
    for x, label in headers:
        ax.text(x, 5.35, label, fontsize=10.5, weight="bold", color=SLATE)
    for index, (construct, evidence, validity, decision, color) in enumerate(rows):
        y = 4.62 - index * 0.90
        ax.add_patch(
            FancyBboxPatch(
                (0.25, y - 0.48),
                15.45,
                0.73,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=LIGHT if index % 2 == 0 else "white",
                edgecolor="#D7DEE5",
                linewidth=0.8,
            )
        )
        ax.text(0.42, y - 0.10, construct, fontsize=10.5, weight="bold", color=INK, va="center")
        ax.text(3.42, y - 0.10, evidence, fontsize=9.7, color=INK, va="center")
        ax.text(8.12, y - 0.10, validity, fontsize=9.7, color=INK, va="center")
        ax.add_patch(
            FancyBboxPatch(
                (12.65, y - 0.31),
                2.72,
                0.42,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=color,
                edgecolor=color,
            )
        )
        ax.text(14.01, y - 0.10, decision, fontsize=8.7, weight="bold", color="white", ha="center", va="center")
    ax.text(
        0.3,
        0.18,
        "A nominal or within-domain corrected contrast cannot rescue an invalid measurement model. These tests prevent an unsupported third axis from fragmenting the two-axis paper.",
        fontsize=10.5,
        color=SLATE,
        style="italic",
    )
    save(fig, "derived_construct_validity_v2")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    evidence_flow()
    microstructure_forest()
    graph_forest()
    derived_validity()
    outputs = [
        "multiscale_evidence_flow_v2.png",
        "whole_brain_microstructure_forest_v2.png",
        "graph_topology_forest_v2.png",
        "derived_construct_validity_v2.png",
    ]
    report = {
        "pass": all((FIG / name).is_file() and (FIG / name).stat().st_size > 20_000 for name in outputs),
        "outputs": {name: (FIG / name).stat().st_size for name in outputs},
    }
    (ROOT / "outputs" / "full_evidence_figures_v2_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not report["pass"]:
        raise SystemExit("Figure validation failed")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
