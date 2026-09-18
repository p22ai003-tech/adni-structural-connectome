#!/usr/bin/env python3
"""Build an aggregate-only multivariate centerpiece for manuscript V2.

The figure integrates four already-declared evidence dimensions without
creating a new multiplicity family:

1. nuisance-residualized joint AxD/RD group geometry;
2. paired AxD-early and RD-late effects across 17 nonduplicate systems;
3. anonymous leave-one-site-out joint effect stability; and
4. nested site-held-out incremental discrimination for AD versus MCI.

Participant identifiers, site labels, and participant-level values are never
written. Statistical annotations come from the frozen upstream releases.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import statsmodels.formula.api as smf

from stage_specific_diffusivity_analysis import (
    PRIMARY_ENDPOINTS,
    PRIMARY_NETWORKS,
    _standardize,
    build_composites,
    collect_metrics,
    fit_group_contrast,
    load_exact_cohort,
    normal_scores,
)


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"
OUT = ROOT / "outputs" / "multivariate_stage_vector_v1"
STAGE = ROOT / "outputs" / "stage_specific_diffusivity_v2"
ML = ROOT / "outputs" / "network_resolved_site_validation_v2"

PNG = FIGURE_DIR / "multivariate_stage_vector_v1.png"
SVG = FIGURE_DIR / "multivariate_stage_vector_v1.svg"
VALIDATION = OUT / "validation.json"
SUMMARY = OUT / "evidence_summary.md"

INK = "#17212B"
SLATE = "#526274"
GRID = "#DCE3EA"
CN = "#7B8794"
MCI = "#1769AA"
AD = "#B03060"
ANATOMICAL = "#087F8C"
FUNCTIONAL = "#E07A1F"
GREEN = "#20845B"
AMBER = "#B46A13"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def nuisance_residuals(composites: pd.DataFrame) -> pd.DataFrame:
    """Preserve group shifts while removing declared nuisance terms."""

    use = composites.copy()
    use["age_z"] = _standardize(use["age"])
    use["gap_z"] = _standardize(use["log_gap_days"])
    for source, target in (
        ("axial_burden", "axial_residual"),
        ("radial_burden", "radial_residual"),
    ):
        use["outcome_z"] = normal_scores(use[source])
        terms = ["age_z"]
        if use["sex"].nunique() > 1:
            terms.append("C(sex)")
        if use["log_gap_days"].nunique() > 1:
            terms.append("gap_z")
        if use["t1_source"].nunique() > 1:
            terms.append("C(t1_source)")
        terms.append("C(site)")
        fit = smf.ols("outcome_z ~ " + " + ".join(terms), data=use).fit()
        use[target] = fit.resid
    return use


def cluster_bootstrap_centroids(
    frame: pd.DataFrame, iterations: int = 3000, seed: int = 20260719
) -> dict[str, np.ndarray]:
    """Bootstrap group centroids by resampling imaging centres."""

    sites = np.asarray(sorted(frame["site"].astype(str).unique()))
    index = {site: position for position, site in enumerate(sites)}
    work = frame.copy()
    work["site_index"] = work["site"].astype(str).map(index)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sites), size=(iterations, len(sites)))
    output: dict[str, np.ndarray] = {}
    for group in ("CN", "MCI", "AD"):
        subset = work[work["group"].eq(group)]
        counts = np.zeros(len(sites), dtype=float)
        sum_x = np.zeros(len(sites), dtype=float)
        sum_y = np.zeros(len(sites), dtype=float)
        for row in subset.itertuples():
            position = int(row.site_index)
            counts[position] += 1.0
            sum_x[position] += float(row.axial_residual)
            sum_y[position] += float(row.radial_residual)
        sampled_counts = counts[draws].sum(axis=1)
        valid = sampled_counts > 0
        centroids = np.column_stack(
            (
                sum_x[draws].sum(axis=1)[valid] / sampled_counts[valid],
                sum_y[draws].sum(axis=1)[valid] / sampled_counts[valid],
            )
        )
        output[group] = centroids
    return output


def add_covariance_ellipse(
    ax: plt.Axes,
    samples: np.ndarray,
    centre: tuple[float, float],
    color: str,
    *,
    alpha: float = 0.14,
) -> None:
    covariance = np.cov(samples.T)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values = values[order]
    vectors = vectors[:, order]
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    scale95 = math.sqrt(5.991464547)
    width, height = 2.0 * scale95 * np.sqrt(np.maximum(values, 0.0))
    ax.add_patch(
        Ellipse(
            centre,
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            linewidth=1.4,
            alpha=alpha,
            zorder=4,
        )
    )


def kde_mass_levels(density: np.ndarray, masses: tuple[float, ...]) -> list[float]:
    flat = np.asarray(density, dtype=float).ravel()
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order])
    cumulative /= cumulative[-1]
    thresholds = []
    for mass in masses:
        position = min(int(np.searchsorted(cumulative, mass)), len(order) - 1)
        thresholds.append(float(flat[order[position]]))
    return sorted(set(thresholds))


def panel_joint_geometry(
    ax: plt.Axes,
    frame: pd.DataFrame,
    centroids_boot: dict[str, np.ndarray],
    primary: pd.DataFrame,
) -> None:
    x_all = frame["axial_residual"].to_numpy(float)
    y_all = frame["radial_residual"].to_numpy(float)
    x_lo, x_hi = np.quantile(x_all, [0.01, 0.99])
    y_lo, y_hi = np.quantile(y_all, [0.01, 0.99])
    x_grid = np.linspace(x_lo - 0.20, x_hi + 0.20, 140)
    y_grid = np.linspace(y_lo - 0.20, y_hi + 0.20, 140)
    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack((xx.ravel(), yy.ravel()))

    colours = {"CN": CN, "MCI": MCI, "AD": AD}
    centres: dict[str, tuple[float, float]] = {}
    for group in ("CN", "MCI", "AD"):
        subset = frame[frame["group"].eq(group)]
        values = np.vstack(
            (subset["axial_residual"].to_numpy(float), subset["radial_residual"].to_numpy(float))
        )
        density = gaussian_kde(values)(positions).reshape(xx.shape)
        levels = kde_mass_levels(density, (0.80, 0.50))
        ax.contour(xx, yy, density, levels=levels, colors=[colours[group]], linewidths=[1.15, 2.0], alpha=0.68)
        centre = (
            float(subset["axial_residual"].mean()),
            float(subset["radial_residual"].mean()),
        )
        centres[group] = centre
        add_covariance_ellipse(ax, centroids_boot[group], centre, colours[group])
        ax.scatter(*centre, s=90, color=colours[group], edgecolor="white", linewidth=1.5, zorder=7)
        ax.text(centre[0] + 0.035, centre[1] + 0.035, group, color=colours[group], fontsize=10.5, weight="bold", zorder=8)

    for left, right in (("CN", "MCI"), ("MCI", "AD")):
        ax.add_patch(
            FancyArrowPatch(
                centres[left],
                centres[right],
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=2.1,
                color=colours[right],
                alpha=0.82,
                zorder=6,
                connectionstyle="arc3,rad=0.05",
            )
        )

    early = primary[
        primary["scope"].eq("full_site_fe")
        & primary["endpoint"].eq("axial_MCI_vs_CN")
    ].iloc[0]
    late = primary[
        primary["scope"].eq("full_site_fe")
        & primary["endpoint"].eq("radial_AD_vs_MCI")
    ].iloc[0]
    ax.text(
        0.03,
        0.97,
        f"Primary projections\nAxD MCI–CN: β={early.beta:.3f}, wild-Holm p=.0046\nRD AD–MCI: β={late.beta:.3f}, wild-Holm p=.0164",
        transform=ax.transAxes,
        va="top",
        fontsize=9.2,
        color=INK,
        bbox=dict(boxstyle="round,pad=.45", facecolor="white", edgecolor=GRID, alpha=0.94),
    )
    ax.set_xlabel("AxD burden after nuisance residualization")
    ax.set_ylabel("RD burden after nuisance residualization")
    ax.set_title("A  Joint stage geometry", loc="left", fontsize=14, weight="bold", color=INK)
    ax.text(
        0.03,
        0.03,
        "Contours: 50% and 80% density regions; ellipses: site-bootstrap 95% centroid uncertainty.",
        transform=ax.transAxes,
        fontsize=8.2,
        color=SLATE,
    )
    ax.grid(color=GRID, linewidth=0.7, alpha=0.7)


def _repelled_positions(values: list[float], lower: float, upper: float, gap: float) -> list[float]:
    order = np.argsort(values)
    placed = np.asarray(values, dtype=float)[order]
    placed[0] = max(placed[0], lower)
    for position in range(1, len(placed)):
        placed[position] = max(placed[position], placed[position - 1] + gap)
    overflow = placed[-1] - upper
    if overflow > 0:
        placed -= overflow
        for position in range(len(placed) - 2, -1, -1):
            placed[position] = min(placed[position], placed[position + 1] - gap)
    restored = np.empty_like(placed)
    restored[order] = placed
    return restored.tolist()


def network_effect_frame() -> pd.DataFrame:
    network = pd.read_csv(STAGE / "network_contrasts.csv")
    network = network[network["nonredundant_primary"].astype(bool)].copy()
    axial = network[network["endpoint"].eq("axial_MCI_vs_CN")][
        ["network", "beta", "bh_q_17_nonredundant"]
    ].rename(columns={"beta": "axial_beta", "bh_q_17_nonredundant": "axial_q"})
    radial = network[network["endpoint"].eq("radial_AD_vs_MCI")][
        ["network", "beta", "bh_q_17_nonredundant"]
    ].rename(columns={"beta": "radial_beta", "bh_q_17_nonredundant": "radial_q"})
    coverage = pd.read_csv(STAGE / "network_coverage.csv")
    coverage = coverage.groupby("network", as_index=False)["available_pct"].min().rename(
        columns={"available_pct": "minimum_group_coverage_pct"}
    )
    result = axial.merge(radial, on="network", validate="1:1").merge(
        coverage, on="network", how="left", validate="1:1"
    )
    if set(result["network"]) != set(PRIMARY_NETWORKS):
        raise RuntimeError("The 17-system inventory does not match the frozen primary inventory")
    return result


def panel_system_effects(ax: plt.Axes, frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame["mapping"] = frame["network"].str.split("::").str[0]
    frame["label"] = frame["network"].str.replace("anatomical::", "a ", regex=False).str.replace(
        "functional::", "f ", regex=False
    ).str.replace("Salience_VAN", "Salience/VAN", regex=False).str.replace(
        "DorsalAttention", "Dorsal attention", regex=False
    )
    sizes = 70 + 3.2 * np.clip(frame["minimum_group_coverage_pct"] - 55, 0, 45)
    for mapping, marker, colour in (
        ("anatomical", "o", ANATOMICAL),
        ("functional", "s", FUNCTIONAL),
    ):
        subset = frame[frame["mapping"].eq(mapping)]
        filled = subset["radial_q"].le(0.05)
        ax.scatter(
            subset["axial_beta"],
            subset["radial_beta"],
            s=sizes[subset.index],
            marker=marker,
            facecolors=[colour if flag else "white" for flag in filled],
            edgecolors=colour,
            linewidths=1.8,
            alpha=0.93,
            label=f"{mapping.capitalize()} mapping",
            zorder=5,
        )

    median_x = float(frame["axial_beta"].median())
    for side, subset in (
        ("left", frame[frame["axial_beta"].le(median_x)].copy()),
        ("right", frame[frame["axial_beta"].gt(median_x)].copy()),
    ):
        target_x = 0.145 if side == "left" else 0.435
        positions = _repelled_positions(
            subset["radial_beta"].tolist(), lower=0.225, upper=0.718, gap=0.048
        )
        for (_, row), target_y in zip(subset.iterrows(), positions):
            colour = ANATOMICAL if row["mapping"] == "anatomical" else FUNCTIONAL
            ax.plot(
                [row["axial_beta"], target_x + (0.004 if side == "left" else -0.004)],
                [row["radial_beta"], target_y],
                color=colour,
                linewidth=0.65,
                alpha=0.55,
                zorder=2,
            )
            ax.text(
                target_x,
                target_y,
                row["label"],
                ha="right" if side == "left" else "left",
                va="center",
                fontsize=7.0,
                color=INK,
            )

    ax.axvline(0, color=SLATE, linestyle="--", linewidth=0.9)
    ax.axhline(0, color=SLATE, linestyle="--", linewidth=0.9)
    ax.set_xlim(0.08, 0.49)
    ax.set_ylim(0.19, 0.76)
    ax.set_xlabel("AxD MCI–CN effect (rank-normal β)")
    ax.set_ylabel("RD AD–MCI effect (rank-normal β)")
    ax.set_title("B  Seventeen-system participation map", loc="left", fontsize=14, weight="bold", color=INK)
    ax.text(
        0.03,
        0.965,
        "17/17 AxD effects FDR-significant\n13/17 RD effects FDR-significant",
        transform=ax.transAxes,
        va="top",
        fontsize=9.3,
        color=INK,
        bbox=dict(boxstyle="round,pad=.35", facecolor="white", edgecolor=GRID, alpha=0.94),
    )
    ax.text(
        0.03,
        0.035,
        "Filled = FDR-significant on both axes; hollow = AxD only. Size = minimum group coverage.",
        transform=ax.transAxes,
        fontsize=8.1,
        color=SLATE,
    )
    ax.grid(color=GRID, linewidth=0.7, alpha=0.7)


def leave_one_site_out_pairs(
    composites: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    sites = sorted(composites["site"].dropna().astype(str).unique())
    rows: list[dict[str, float]] = []
    for site in sites:
        frame = composites[~composites["site"].astype(str).eq(site)].copy()
        result: dict[str, float] = {}
        for endpoint in PRIMARY_ENDPOINTS:
            fit = fit_group_contrast(
                frame,
                endpoint.outcome,
                {endpoint.left: 1.0, endpoint.right: -1.0},
                reference="CN",
                scope="full_site_fe",
            )
            prefix = "axial" if endpoint.endpoint.startswith("axial") else "radial"
            result[f"{prefix}_beta"] = float(fit["beta"])
            result[f"{prefix}_p"] = float(fit["p_value"])
        rows.append(result)
    pairs = pd.DataFrame(rows)
    aggregate = {
        "iterations": int(len(pairs)),
        "both_positive_fraction": float(
            ((pairs["axial_beta"] > 0) & (pairs["radial_beta"] > 0)).mean()
        ),
        "both_nominal_p_below_0_05_fraction": float(
            ((pairs["axial_p"] < 0.05) & (pairs["radial_p"] < 0.05)).mean()
        ),
        "axial_beta_minimum": float(pairs["axial_beta"].min()),
        "axial_beta_maximum": float(pairs["axial_beta"].max()),
        "radial_beta_minimum": float(pairs["radial_beta"].min()),
        "radial_beta_maximum": float(pairs["radial_beta"].max()),
    }
    return pairs, aggregate


def panel_site_robustness(
    ax: plt.Axes,
    pairs: pd.DataFrame,
    aggregate: dict[str, float],
    primary: pd.DataFrame,
) -> None:
    ax.scatter(
        pairs["axial_beta"],
        pairs["radial_beta"],
        s=42,
        color=MCI,
        alpha=0.32,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    centre = (float(pairs["axial_beta"].mean()), float(pairs["radial_beta"].mean()))
    add_covariance_ellipse(ax, pairs[["axial_beta", "radial_beta"]].to_numpy(), centre, MCI, alpha=0.18)
    full_axial = float(
        primary[
            primary["scope"].eq("full_site_fe")
            & primary["endpoint"].eq("axial_MCI_vs_CN")
        ]["beta"].iloc[0]
    )
    full_radial = float(
        primary[
            primary["scope"].eq("full_site_fe")
            & primary["endpoint"].eq("radial_AD_vs_MCI")
        ]["beta"].iloc[0]
    )
    ax.scatter(
        full_axial,
        full_radial,
        marker="*",
        s=270,
        facecolor=AD,
        edgecolor="white",
        linewidth=1.2,
        label="Full 51-site estimate",
        zorder=7,
    )
    ax.axvline(0, color=SLATE, linestyle="--", linewidth=0.9)
    ax.axhline(0, color=SLATE, linestyle="--", linewidth=0.9)
    ax.set_xlim(0, max(0.38, aggregate["axial_beta_maximum"] + 0.035))
    ax.set_ylim(0, max(0.72, aggregate["radial_beta_maximum"] + 0.045))
    ax.set_xlabel("AxD MCI–CN β after omitting one site")
    ax.set_ylabel("RD AD–MCI β after omitting the same site")
    ax.set_title("C  Joint leave-one-site-out robustness", loc="left", fontsize=14, weight="bold", color=INK)
    nominal = int(
        round(aggregate["both_nominal_p_below_0_05_fraction"] * aggregate["iterations"])
    )
    ax.text(
        0.04,
        0.96,
        f"{aggregate['iterations']}/{aggregate['iterations']} retain both positive directions\n"
        f"{nominal}/{aggregate['iterations']} have both nominal p<.05",
        transform=ax.transAxes,
        va="top",
        fontsize=9.6,
        color=INK,
        bbox=dict(boxstyle="round,pad=.4", facecolor="white", edgecolor=GRID, alpha=0.95),
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.7)


def classification_evidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(ML / "classification_summary.csv")
    increments = pd.read_csv(ML / "classification_increment_bootstrap.csv")
    ad_summary = summary[summary["contrast"].eq("AD_vs_MCI")].copy()
    ad_increments = increments[
        increments["contrast"].eq("AD_vs_MCI")
        & increments["reference"].eq("baseline")
        & increments["comparator"].isin(["endpoint_composite", "endpoint_networks"])
    ].copy()
    return ad_summary, ad_increments


def panel_transportability(
    ax: plt.Axes, summary: pd.DataFrame, increments: pd.DataFrame
) -> None:
    labels = {"endpoint_composite": "Scalar RD", "endpoint_networks": "17-system RD"}
    order = ["endpoint_composite", "endpoint_networks"]
    frame = increments.set_index("comparator").loc[order].reset_index()
    y = np.array([1, 0])
    colours = [AMBER, GREEN]
    for position, row, colour in zip(y, frame.itertuples(), colours):
        ax.plot(
            [row.delta_auc_ci95_low, row.delta_auc_ci95_high],
            [position, position],
            color=colour,
            linewidth=3.0,
            solid_capstyle="round",
        )
        ax.scatter(row.delta_auc_median, position, s=95, color=colour, edgecolor="white", linewidth=1.2, zorder=5)
        ax.text(
            row.delta_auc_ci95_high + 0.009,
            position,
            f"{row.delta_auc_median:.3f} [{row.delta_auc_ci95_low:.3f}, {row.delta_auc_ci95_high:.3f}]",
            va="center",
            fontsize=9.0,
            color=INK,
        )
    ax.axvline(0, color=SLATE, linestyle="--", linewidth=1.0)
    ax.set_yticks(y, [labels[item] for item in order])
    ax.set_xlim(-0.03, 0.34)
    ax.set_ylim(-0.75, 1.75)
    ax.set_xlabel("Site-bootstrap ΔAUC over demographic/acquisition baseline (95% CI)")
    ax.set_title("D  Held-out-site late-RD information", loc="left", fontsize=14, weight="bold", color=INK)
    baseline_auc = float(summary[summary["model"].eq("baseline")]["roc_auc"].iloc[0])
    scalar_auc = float(summary[summary["model"].eq("endpoint_composite")]["roc_auc"].iloc[0])
    system_auc = float(summary[summary["model"].eq("endpoint_networks")]["roc_auc"].iloc[0])
    system_vs_scalar = pd.read_csv(ML / "classification_increment_bootstrap.csv")
    system_vs_scalar = system_vs_scalar[
        system_vs_scalar["contrast"].eq("AD_vs_MCI")
        & system_vs_scalar["comparator"].eq("endpoint_networks")
        & system_vs_scalar["reference"].eq("endpoint_composite")
    ].iloc[0]
    ax.text(
        0.03,
        0.12,
        f"Held-out AUC: baseline {baseline_auc:.3f} → scalar {scalar_auc:.3f} → systems {system_auc:.3f}\n"
        f"Systems vs scalar: Δ={system_vs_scalar.delta_auc_median:.3f} "
        f"[{system_vs_scalar.delta_auc_ci95_low:.3f}, {system_vs_scalar.delta_auc_ci95_high:.3f}]",
        transform=ax.transAxes,
        fontsize=9.2,
        color=INK,
        bbox=dict(boxstyle="round,pad=.42", facecolor="white", edgecolor=GRID, alpha=0.95),
    )
    ax.text(
        0.03,
        0.03,
        "Nested leave-one-site-out, pair-common 20-site support; post-selection internal validation.",
        transform=ax.transAxes,
        fontsize=8.1,
        color=SLATE,
    )
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def build() -> dict[str, object]:
    cohort = load_exact_cohort()
    metrics = collect_metrics()
    composites, _ = build_composites(cohort, metrics, PRIMARY_NETWORKS)
    residualized = nuisance_residuals(composites)
    centroid_boot = cluster_bootstrap_centroids(residualized)
    primary = pd.read_csv(STAGE / "primary_contrasts.csv")
    systems = network_effect_frame()
    loo_pairs, loo_aggregate = leave_one_site_out_pairs(composites)
    classification, increments = classification_evidence()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".multivariate-stage-vector.", dir=OUT))
    temporary_png = temporary_dir / PNG.name
    temporary_svg = temporary_dir / SVG.name
    try:
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 9.5,
                "axes.labelcolor": INK,
                "xtick.color": SLATE,
                "ytick.color": SLATE,
                "axes.edgecolor": "#AAB4BE",
            }
        )
        fig = plt.figure(figsize=(19.2, 14.2), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], wspace=0.13, hspace=0.13)
        axes = [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1]),
        ]
        panel_joint_geometry(axes[0], residualized, centroid_boot, primary)
        panel_system_effects(axes[1], systems)
        panel_site_robustness(axes[2], loo_pairs, loo_aggregate, primary)
        panel_transportability(axes[3], classification, increments)
        fig.suptitle(
            "A distributed two-axis phenotype across stage, systems, centres and prediction",
            fontsize=21,
            weight="bold",
            color=INK,
            y=1.035,
        )
        fig.text(
            0.5,
            1.008,
            "Four linked views of the same frozen AxD MCI–CN and RD AD–MCI family; no post-hoc third biological axis is asserted",
            ha="center",
            fontsize=10.8,
            color=SLATE,
        )
        fig.savefig(temporary_png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(temporary_svg, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        os.replace(temporary_png, PNG)
        os.replace(temporary_svg, SVG)
    finally:
        for child in temporary_dir.iterdir():
            child.unlink()
        temporary_dir.rmdir()

    radial_significant = int(systems["radial_q"].le(0.05).sum())
    axial_significant = int(systems["axial_q"].le(0.05).sum())
    scalar_increment = increments[increments["comparator"].eq("endpoint_composite")].iloc[0]
    system_increment = increments[increments["comparator"].eq("endpoint_networks")].iloc[0]
    source_paths = [
        STAGE / "primary_contrasts.csv",
        STAGE / "network_contrasts.csv",
        STAGE / "network_coverage.csv",
        ML / "classification_summary.csv",
        ML / "classification_increment_bootstrap.csv",
    ]
    payload: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pass": True,
        "release_status": "POST_HOC_INTERNAL_VALIDATION_NOT_INDEPENDENT_CONFIRMATION",
        "participant_level_values_written": False,
        "site_labels_written": False,
        "n": int(len(composites)),
        "n_sites": int(composites["site"].nunique()),
        "primary_family": {
            "axial_MCI_vs_CN_wild_holm_p": 0.0046,
            "radial_AD_vs_MCI_wild_holm_p": 0.0164,
        },
        "system_participation": {
            "systems": int(len(systems)),
            "axial_fdr_significant": axial_significant,
            "radial_fdr_significant": radial_significant,
        },
        "leave_one_site_out": loo_aggregate,
        "held_out_AD_vs_MCI": {
            "scalar_delta_auc_median": float(scalar_increment["delta_auc_median"]),
            "scalar_delta_auc_ci95": [
                float(scalar_increment["delta_auc_ci95_low"]),
                float(scalar_increment["delta_auc_ci95_high"]),
            ],
            "system_delta_auc_median": float(system_increment["delta_auc_median"]),
            "system_delta_auc_ci95": [
                float(system_increment["delta_auc_ci95_low"]),
                float(system_increment["delta_auc_ci95_high"]),
            ],
        },
        "checks": {
            "cohort_is_513": len(composites) == 513,
            "site_count_is_51": composites["site"].nunique() == 51,
            "network_inventory_is_17": len(systems) == 17,
            "all_axial_systems_pass_fdr": axial_significant == 17,
            "thirteen_radial_systems_pass_fdr": radial_significant == 13,
            "all_deleted_site_pairs_remain_positive": loo_aggregate["both_positive_fraction"] == 1.0,
            "all_deleted_site_pairs_remain_nominal": loo_aggregate[
                "both_nominal_p_below_0_05_fraction"
            ] == 1.0,
            "scalar_delta_ci_excludes_zero": float(scalar_increment["delta_auc_ci95_low"]) > 0,
            "system_delta_ci_excludes_zero": float(system_increment["delta_auc_ci95_low"]) > 0,
            "figure_files_exist": PNG.is_file() and SVG.is_file(),
        },
        "inputs": {
            str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in source_paths
        },
        "figures": {
            PNG.name: {"bytes": PNG.stat().st_size, "sha256": sha256(PNG)},
            SVG.name: {"bytes": SVG.stat().st_size, "sha256": sha256(SVG)},
        },
        "interpretation": (
            "The figure integrates already-declared positive evidence. It does not establish network selectivity, "
            "a cellular AxD-to-RD mechanism, independent replication, or superiority of the network model over scalar RD."
        ),
    }
    payload["pass"] = bool(all(payload["checks"].values()))
    VALIDATION.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = f"""# Multivariate stage-vector centerpiece

**Generated:** {payload['generated_utc']}  
**Validation:** {'PASS' if payload['pass'] else 'FAIL'}

- Joint primary inference: AxD MCI-CN wild-Holm p=0.0046; RD AD-MCI wild-Holm p=0.0164.
- Distributed systems evidence: {axial_significant}/17 AxD and {radial_significant}/17 RD effects pass endpoint-specific FDR.
- Centre robustness: {loo_aggregate['iterations']}/{loo_aggregate['iterations']} anonymous leave-one-site-out fits retain both positive directions and both nominal p<0.05.
- Held-out late-RD evidence: scalar delta AUC={scalar_increment['delta_auc_median']:.3f} [{scalar_increment['delta_auc_ci95_low']:.3f}, {scalar_increment['delta_auc_ci95_high']:.3f}]; 17-system delta AUC={system_increment['delta_auc_median']:.3f} [{system_increment['delta_auc_ci95_low']:.3f}, {system_increment['delta_auc_ci95_high']:.3f}].

This is an aggregate-only integration of frozen evidence, not a new post-hoc significance family. The system model's increment over scalar RD remains uncertain and network selectivity is not claimed.
"""
    SUMMARY.write_text(summary, encoding="utf-8")
    return payload


def main() -> None:
    payload = build()
    print(
        json.dumps(
            {
                "pass": payload["pass"],
                "figure": str(PNG),
                "systems": payload["system_participation"],
                "leave_one_site_out": payload["leave_one_site_out"],
            }
        )
    )
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
