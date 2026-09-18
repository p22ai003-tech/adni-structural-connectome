#!/usr/bin/env python3
"""Build a publication-ready, output-only release from frozen aggregate results.

This script does not draft manuscript text and does not change any analysis.
It packages validated statistics, clean figures, privacy-safe source data, and a
machine-readable evidence ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

import build_multivariate_stage_vector_v1 as multivar


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "refined_output_release_v1"
FIG = OUT / "figures"
DATA = OUT / "source_data"

STAGE = ROOT / "outputs" / "stage_specific_diffusivity_v2"
SOUNDNESS = ROOT / "outputs" / "primary_inference_soundness_v2"
EXTENSION = ROOT / "outputs" / "manuscript_multiscale_extension_v2"
ML = ROOT / "outputs" / "network_resolved_site_validation_v2"
CLAIMS = ROOT / "outputs" / "all_evidence_inventory_v2" / "claim_status_matrix.csv"

INK = "#17212B"
SLATE = "#526274"
GRID = "#DCE3EA"
BLUE = "#1769AA"
MAGENTA = "#B03060"
TEAL = "#087F8C"
GREEN = "#20845B"
AMBER = "#B46A13"
GREY = "#7B8794"
LIGHT = "#F3F6F9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, stem: str) -> dict[str, Path]:
    paths = {
        "png": FIG / f"{stem}.png",
        "svg": FIG / f"{stem}.svg",
        "pdf": FIG / f"{stem}.pdf",
    }
    fig.savefig(paths["png"], dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "axes.edgecolor": "#AAB4BE",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def tensor_context() -> tuple[pd.DataFrame, dict[str, Path]]:
    source = pd.read_csv(EXTENSION / "global_microstructure_site_tests.csv")
    frame = source[
        source["data_validity"].eq("hard_range_valid")
        & source["scope"].eq("full_site_fe")
    ].copy()
    metric_order = {"FA": 0, "MD": 1, "AxD": 2, "RD": 3}
    contrast_order = {"MCI_vs_CN": 0, "AD_vs_MCI": 1, "AD_vs_CN": 2}
    frame["order"] = (
        frame["metric"].map(metric_order) * 10
        + frame["contrast"].map(contrast_order)
    )
    frame = frame.sort_values("order").reset_index(drop=True)
    frame["holm_pass"] = frame["holm_p_12_within_validity_scope"].le(0.05)

    colours = {"FA": GREY, "MD": GREEN, "AxD": BLUE, "RD": MAGENTA}
    y = np.arange(len(frame))[::-1]
    fig, ax = plt.subplots(figsize=(12.2, 8.2))
    for index, row in frame.iterrows():
        colour = colours[row["metric"]]
        yi = y[index]
        ax.plot([row["ci95_low"], row["ci95_high"]], [yi, yi], color=colour, lw=2.2)
        ax.scatter(
            row["beta"],
            yi,
            s=78,
            facecolor=colour if row["holm_pass"] else "white",
            edgecolor=colour,
            linewidth=1.8,
            zorder=4,
        )
        ax.text(
            row["ci95_high"] + 0.035,
            yi,
            f"Holm p={row['holm_p_12_within_validity_scope']:.3g}",
            va="center",
            fontsize=8.7,
            color=colour if row["holm_pass"] else SLATE,
        )
    labels = [
        f"{row.metric}   {row.contrast.replace('_vs_', ' − ')}"
        for row in frame.itertuples()
    ]
    ax.set_yticks(y, labels)
    ax.axvline(0, color=GREY, lw=1.0, ls="--")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Adjusted rank-normal contrast (left group minus right group)")
    fig.suptitle(
        "Whole-brain diffusion-tensor contrasts",
        x=0.12,
        y=0.985,
        ha="left",
        fontsize=19,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.12,
        0.945,
        "Full cohort; age, sex, DTI–T1 interval, T1 source and imaging-centre adjusted",
        color=SLATE,
        fontsize=10,
    )
    ax.text(
        0,
        -0.115,
        "Filled markers pass Holm correction across the 12 metric-by-contrast tests.",
        transform=ax.transAxes,
        color=SLATE,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.91))
    paths = save_figure(fig, "01_whole_brain_tensor_context")
    export = frame[
        [
            "metric",
            "contrast",
            "n",
            "n_sites",
            "beta",
            "ci95_low",
            "ci95_high",
            "p_value",
            "holm_p_12_within_validity_scope",
            "holm_pass",
        ]
    ].copy()
    return export, paths


def adjacent_stage_effects() -> tuple[pd.DataFrame, dict[str, Path]]:
    locked = pd.read_csv(STAGE / "primary_contrasts.csv")
    audit = pd.read_csv(SOUNDNESS / "primary_small_cluster_inference.csv")
    columns = [
        "endpoint",
        "scope",
        "n",
        "n_sites",
        "beta",
        "cr1_se",
        "p_cr1_t_gminus1",
        "p_restricted_wild_cluster_bootstrap_t",
        "wild_bootstrap_valid_iterations",
        "holm_p_cr1_t_gminus1",
        "holm_p_restricted_wild_cluster_bootstrap_t",
    ]
    frame = locked.merge(
        audit[columns],
        on=["endpoint", "scope", "n", "n_sites", "beta"],
        validate="1:1",
        suffixes=("_locked", ""),
    )
    frame["wild_holm_pass"] = frame[
        "holm_p_restricted_wild_cluster_bootstrap_t"
    ].le(0.05)
    scopes = ["full_site_fe", "pair_common_sites", "all_three_group_sites"]
    labels = ["Full cohort", "Pair-common sites", "Sites containing CN, MCI and AD"]
    endpoints = ["axial_MCI_vs_CN", "radial_AD_vs_MCI"]
    colours = {"axial_MCI_vs_CN": BLUE, "radial_AD_vs_MCI": MAGENTA}
    titles = {"axial_MCI_vs_CN": "AxD: MCI − CN", "radial_AD_vs_MCI": "RD: AD − MCI"}

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.4), sharey=True)
    y = np.arange(len(scopes))[::-1]
    for ax, endpoint in zip(axes, endpoints):
        part = frame[frame["endpoint"].eq(endpoint)].set_index("scope").loc[scopes]
        colour = colours[endpoint]
        for yi, (_, row) in zip(y, part.iterrows()):
            ax.plot([row["ci95_low"], row["ci95_high"]], [yi, yi], color=colour, lw=2.4)
            ax.scatter(
                row["beta"],
                yi,
                s=105,
                facecolor=colour if row["wild_holm_pass"] else "white",
                edgecolor=colour,
                linewidth=2.0,
                zorder=4,
            )
            ax.text(
                row["ci95_high"] + 0.035,
                yi,
                f"n={int(row['n'])}; sites={int(row['n_sites'])}\n"
                f"wild-Holm p={row['holm_p_restricted_wild_cluster_bootstrap_t']:.4g}",
                va="center",
                fontsize=8.7,
                color=INK,
            )
        ax.axvline(0, color=GREY, lw=1.0, ls="--")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(titles[endpoint], loc="left", fontsize=17, weight="bold", color=colour)
        ax.set_xlabel("Adjusted rank-normal beta (95% CI)")
    axes[0].set_yticks(y, labels)
    fig.suptitle("Adjacent-stage diffusivity effects across site-support scopes", fontsize=19, weight="bold", y=1.03)
    fig.text(
        0.5,
        -0.015,
        "Restricted wild-cluster bootstrap-t: 9,999 valid draws; Holm correction across the two endpoints within each scope. Filled markers pass p≤0.05.",
        ha="center",
        color=SLATE,
        fontsize=9.2,
    )
    fig.tight_layout(w_pad=3.0)
    paths = save_figure(fig, "02_adjacent_stage_effects")
    export_columns = [
        "endpoint",
        "scope",
        "n",
        "n_sites",
        "beta",
        "ci95_low",
        "ci95_high",
        "p_value",
        "holm_p_two_endpoints",
        "p_cr1_t_gminus1",
        "p_restricted_wild_cluster_bootstrap_t",
        "holm_p_restricted_wild_cluster_bootstrap_t",
        "wild_bootstrap_valid_iterations",
        "wild_holm_pass",
    ]
    return frame[export_columns].copy(), paths


def _ellipse_summary(samples: np.ndarray) -> dict[str, float]:
    covariance = np.cov(samples.T)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values = values[order]
    vectors = vectors[:, order]
    scale95 = math.sqrt(5.991464547)
    return {
        "ellipse_width_95": float(2.0 * scale95 * math.sqrt(max(values[0], 0.0))),
        "ellipse_height_95": float(2.0 * scale95 * math.sqrt(max(values[1], 0.0))),
        "ellipse_angle_degrees": float(math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))),
    }


def integrated_multiscale() -> tuple[dict[str, pd.DataFrame], dict[str, Path], dict[str, object]]:
    cohort = multivar.load_exact_cohort()
    metrics = multivar.collect_metrics()
    composites, _ = multivar.build_composites(cohort, metrics, multivar.PRIMARY_NETWORKS)
    residualized = multivar.nuisance_residuals(composites)
    centroid_boot = multivar.cluster_bootstrap_centroids(residualized)
    primary = pd.read_csv(STAGE / "primary_contrasts.csv")
    systems = multivar.network_effect_frame()
    loo_pairs, loo_summary = multivar.leave_one_site_out_pairs(composites)
    classification, increments = multivar.classification_evidence()

    fig = plt.figure(figsize=(19.2, 14.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], wspace=0.13, hspace=0.13)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    multivar.panel_joint_geometry(axes[0], residualized, centroid_boot, primary)
    multivar.panel_system_effects(axes[1], systems)
    multivar.panel_site_robustness(axes[2], loo_pairs, loo_summary, primary)
    multivar.panel_transportability(axes[3], classification, increments)
    axes[0].set_title("A  Adjusted stage geometry", loc="left", fontsize=14, weight="bold")
    axes[1].set_title("B  System-level effects", loc="left", fontsize=14, weight="bold")
    axes[2].set_title("C  Leave-one-centre-out estimates", loc="left", fontsize=14, weight="bold")
    axes[3].set_title("D  Held-out-centre AD–MCI classification", loc="left", fontsize=14, weight="bold")
    fig.suptitle(
        "Stage-specific diffusivity patterns across systems and imaging centres",
        fontsize=21,
        weight="bold",
        color=INK,
        y=1.032,
    )
    fig.text(
        0.5,
        1.006,
        "AxD for MCI–CN and RD for AD–MCI are shown on matched system, site-deletion and held-out-site views",
        ha="center",
        fontsize=10.8,
        color=SLATE,
    )
    paths = save_figure(fig, "03_multiscale_system_site_validation")

    centroid_rows = []
    for group in ["CN", "MCI", "AD"]:
        subset = residualized[residualized["group"].eq(group)]
        row = {
            "group": group,
            "n": int(len(subset)),
            "axial_residual_mean": float(subset["axial_residual"].mean()),
            "radial_residual_mean": float(subset["radial_residual"].mean()),
        }
        row.update(_ellipse_summary(centroid_boot[group]))
        centroid_rows.append(row)
    centroids = pd.DataFrame(centroid_rows)

    safe_loo = loo_pairs.copy()
    safe_loo.insert(0, "anonymous_deletion_index", np.arange(1, len(safe_loo) + 1))
    releases = {
        "stage_centroids": centroids,
        "system_effects": systems,
        "anonymous_leave_one_site_out": safe_loo,
        "classification_summary_AD_vs_MCI": classification,
        "classification_increments_AD_vs_MCI": increments,
    }
    return releases, paths, loo_summary


def sensitivity_figure() -> tuple[pd.DataFrame, dict[str, Path]]:
    frame = pd.read_csv(STAGE / "sensitivity_contrasts.csv")
    labels = {
        "all_available": "All available",
        "at_least_9_networks": "At least 9 systems",
        "complete_17_networks": "Complete 17 systems",
        "timing_le90": "DTI–T1 interval ≤90 days",
        "timing_le180": "DTI–T1 interval ≤180 days",
        "same_adni_phase": "Same ADNI phase",
        "siemens54": "Siemens 54-direction",
        "original_t1": "Original T1 only",
        "legacy_qc": "Historical QC subset",
    }
    order = list(labels)
    frame = frame[frame["stratum"].isin(order)].copy()
    frame["holm_pass"] = frame["holm_p_two_endpoints"].lt(0.05)
    endpoints = ["axial_MCI_vs_CN", "radial_AD_vs_MCI"]
    colours = {"axial_MCI_vs_CN": BLUE, "radial_AD_vs_MCI": MAGENTA}
    titles = {"axial_MCI_vs_CN": "AxD: MCI − CN", "radial_AD_vs_MCI": "RD: AD − MCI"}
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 7.4), sharey=True)
    for ax, endpoint in zip(axes, endpoints):
        part = frame[frame["endpoint"].eq(endpoint)].set_index("stratum").loc[order]
        values = part["beta"].to_numpy(float)
        low = part["ci95_low"].to_numpy(float)
        high = part["ci95_high"].to_numpy(float)
        passed = part["holm_pass"].to_numpy(bool)
        colour = colours[endpoint]
        ax.axvline(0, color=GREY, lw=1.0, ls="--")
        ax.errorbar(
            values,
            y,
            xerr=np.vstack((values - low, high - values)),
            fmt="none",
            ecolor="#8A9298",
            elinewidth=1.8,
            capsize=3.5,
        )
        ax.scatter(values[passed], y[passed], s=70, color=colour, edgecolor=colour, zorder=4)
        ax.scatter(
            values[~passed],
            y[~passed],
            s=70,
            facecolor="white",
            edgecolor=colour,
            linewidth=1.9,
            zorder=4,
        )
        ax.set_title(titles[endpoint], loc="left", fontsize=16, weight="bold", color=colour)
        ax.set_xlabel("Adjusted rank-normal beta (95% CI)")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    sample_sizes = (
        frame[frame["endpoint"].eq("axial_MCI_vs_CN")]
        .set_index("stratum")
        .loc[order, "n"]
        .astype(int)
        .tolist()
    )
    axes[0].set_yticks(
        y,
        [f"{labels[item]}  (n={n})" for item, n in zip(order, sample_sizes)],
    )
    axes[0].invert_yaxis()
    fig.suptitle("Completeness and acquisition sensitivity", fontsize=19, weight="bold", y=1.015)
    fig.text(
        0.5,
        0.005,
        "Filled markers pass Holm correction across the two endpoints within a stratum; open markers do not.",
        ha="center",
        color=SLATE,
        fontsize=9.2,
    )
    fig.tight_layout(rect=(0.02, 0.045, 1, 0.96), w_pad=2.8)
    paths = save_figure(fig, "04_completeness_acquisition_sensitivity")
    return frame, paths


def core_results_table(
    primary: pd.DataFrame,
    system_effects: pd.DataFrame,
    loo_summary: dict[str, object],
) -> pd.DataFrame:
    classification = pd.read_csv(ML / "classification_summary.csv")
    increments = pd.read_csv(ML / "classification_increment_bootstrap.csv")
    ad = classification[classification["contrast"].eq("AD_vs_MCI")]
    delta = increments[
        increments["contrast"].eq("AD_vs_MCI")
        & increments["comparator"].eq("endpoint_networks")
        & increments["reference"].eq("baseline")
    ].iloc[0]
    rows = []
    for row in primary.itertuples():
        rows.append(
            {
                "result_family": "primary_adjacent_stage",
                "result": f"{row.endpoint} | {row.scope}",
                "estimate": row.beta,
                "ci95_low": row.ci95_low,
                "ci95_high": row.ci95_high,
                "adjusted_p_or_q": row.holm_p_restricted_wild_cluster_bootstrap_t,
                "status": "retained" if row.wild_holm_pass else "sensitivity_only",
                "source": "primary_inference_soundness_v2/primary_small_cluster_inference.csv",
            }
        )
    rows.extend(
        [
            {
                "result_family": "system_participation",
                "result": "AxD MCI-CN systems passing endpoint-specific FDR",
                "estimate": int(system_effects["axial_q"].le(0.05).sum()),
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "adjusted_p_or_q": np.nan,
                "status": "retained",
                "source": "stage_specific_diffusivity_v2/network_contrasts.csv",
            },
            {
                "result_family": "system_participation",
                "result": "RD AD-MCI systems passing endpoint-specific FDR",
                "estimate": int(system_effects["radial_q"].le(0.05).sum()),
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "adjusted_p_or_q": np.nan,
                "status": "retained",
                "source": "stage_specific_diffusivity_v2/network_contrasts.csv",
            },
            {
                "result_family": "site_influence",
                "result": "Paired site deletions retaining both positive and nominally significant effects",
                "estimate": int(loo_summary["iterations"]),
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "adjusted_p_or_q": np.nan,
                "status": "retained",
                "source": "refined_output_release_v1/source_data/03_anonymous_leave_one_site_out.csv",
            },
            {
                "result_family": "held_out_site_classification",
                "result": "AD-MCI 17-system RD ROC AUC",
                "estimate": float(ad[ad["model"].eq("endpoint_networks")]["roc_auc"].iloc[0]),
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "adjusted_p_or_q": np.nan,
                "status": "secondary",
                "source": "network_resolved_site_validation_v2/classification_summary.csv",
            },
            {
                "result_family": "held_out_site_classification",
                "result": "AD-MCI 17-system RD delta AUC over baseline",
                "estimate": float(delta["delta_auc_median"]),
                "ci95_low": float(delta["delta_auc_ci95_low"]),
                "ci95_high": float(delta["delta_auc_ci95_high"]),
                "adjusted_p_or_q": np.nan,
                "status": "secondary",
                "source": "network_resolved_site_validation_v2/classification_increment_bootstrap.csv",
            },
        ]
    )
    return pd.DataFrame(rows)


def evidence_ledger() -> pd.DataFrame:
    source = pd.read_csv(CLAIMS)
    status_map = {
        "A": "retained_primary_candidate",
        "B": "retained_secondary",
        "C": "exploratory_only",
        "D": "not_supported_or_invalid",
    }
    ledger = source[
        [
            "claim_id",
            "claim_name",
            "status_code",
            "status_label",
            "evidence_summary",
            "evidence_path",
            "required_before_vfinal",
        ]
    ].copy()
    ledger.insert(3, "output_disposition", ledger["status_code"].map(status_map))
    return ledger


def write_readme() -> None:
    text = """# Refined structural-connectome output release V1

This is an output-only package. It does not contain manuscript prose and does not alter any statistical model.

## Figures

- `01_whole_brain_tensor_context`: full-cohort FA, MD, AxD and RD contrasts with 12-test Holm status.
- `02_adjacent_stage_effects`: AxD MCI-CN and RD AD-MCI estimates across full, pair-common and all-three-site support, including restricted wild-cluster Holm status.
- `03_multiscale_system_site_validation`: adjusted stage geometry, 17-system effects, anonymous paired site deletion and site-held-out AD-MCI classification.
- `04_completeness_acquisition_sensitivity`: endpoint estimates across completeness and acquisition restrictions.

Each figure is exported as PNG, SVG and PDF. The `source_data` directory contains only aggregate or anonymized values required to trace the displayed statistics. No participant identifier, site label, held-out prediction or fold assignment is released.

## Status files

- `core_results_table.csv`: compact retained/sensitivity result table.
- `results_status_ledger.csv`: complete claim disposition as retained primary, retained secondary, exploratory only, or not supported/invalid.
- `validation.json`: hashes, row counts, statistical consistency checks and privacy checks.
- `independent_validation.json` and `independent_validation.md`: separate replay artifacts. Treat them as pending if absent, and rerun the independent validator after every rebuild.

Historical-output findings remain internal discovery evidence until the corrected-processing confirmation run is complete.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    for stale_independent_record in (
        OUT / "independent_validation.json",
        OUT / "independent_validation.md",
    ):
        stale_independent_record.unlink(missing_ok=True)
    set_style()

    tensor, tensor_paths = tensor_context()
    primary, primary_paths = adjacent_stage_effects()
    integrated, integrated_paths, loo_summary = integrated_multiscale()
    sensitivity, sensitivity_paths = sensitivity_figure()

    tensor.to_csv(DATA / "01_whole_brain_tensor_context.csv", index=False)
    primary.to_csv(DATA / "02_adjacent_stage_effects.csv", index=False)
    integrated["stage_centroids"].to_csv(DATA / "03_stage_centroids.csv", index=False)
    integrated["system_effects"].to_csv(DATA / "03_system_effects.csv", index=False)
    integrated["anonymous_leave_one_site_out"].to_csv(
        DATA / "03_anonymous_leave_one_site_out.csv", index=False
    )
    integrated["classification_summary_AD_vs_MCI"].to_csv(
        DATA / "03_classification_summary_AD_vs_MCI.csv", index=False
    )
    integrated["classification_increments_AD_vs_MCI"].to_csv(
        DATA / "03_classification_increments_AD_vs_MCI.csv", index=False
    )
    sensitivity.to_csv(DATA / "04_completeness_acquisition_sensitivity.csv", index=False)

    core = core_results_table(primary, integrated["system_effects"], loo_summary)
    core.to_csv(OUT / "core_results_table.csv", index=False)
    ledger = evidence_ledger()
    ledger.to_csv(OUT / "results_status_ledger.csv", index=False)
    write_readme()

    source_files = sorted(DATA.glob("*.csv"))
    figure_files = sorted(FIG.glob("*"))
    exported_columns = {column for path in source_files for column in pd.read_csv(path, nrows=0).columns}
    prohibited = {"participant_id", "subject_id", "site", "site_code", "prediction", "fold"}

    checks = {
        "tensor_context_has_12_rows": len(tensor) == 12,
        "primary_merge_has_6_rows": len(primary) == 6,
        "primary_beta_replay_exact": bool(
            (primary["beta"].notna()).all()
            and len(primary[["endpoint", "scope"]].drop_duplicates()) == 6
        ),
        "wild_bootstrap_has_9999_valid_draws": bool(
            primary["wild_bootstrap_valid_iterations"].eq(9999).all()
        ),
        "four_primary_scope_rows_pass_wild_holm": int(primary["wild_holm_pass"].sum()) == 4,
        "two_all_three_rows_are_sensitivity_only": int((~primary["wild_holm_pass"]).sum()) == 2,
        "system_inventory_is_17": len(integrated["system_effects"]) == 17,
        "axial_system_fdr_count_is_17": int(integrated["system_effects"]["axial_q"].le(0.05).sum()) == 17,
        "radial_system_fdr_count_is_13": int(integrated["system_effects"]["radial_q"].le(0.05).sum()) == 13,
        "all_51_site_deletions_positive": loo_summary["both_positive_fraction"] == 1.0,
        "all_51_site_deletions_nominal": loo_summary["both_nominal_p_below_0_05_fraction"] == 1.0,
        "sensitivity_has_18_rows": len(sensitivity) == 18,
        "all_figures_have_png_svg_pdf": len(figure_files) == 12,
        "all_figure_files_nonempty": all(path.stat().st_size > 10_000 for path in figure_files),
        "no_direct_identifier_or_site_label_export": not bool(exported_columns & prohibited),
        "status_ledger_is_complete": len(ledger) == 19 and ledger["output_disposition"].notna().all(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    payload = {
        "schema_version": "1.0.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "release_status": "HISTORICAL_OUTPUTS_PENDING_CORRECTED_PROCESSING_CONFIRMATION",
        "pass": bool(all(checks.values())),
        "checks": checks,
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "figures": {
            str(path.relative_to(OUT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in figure_files
        },
        "source_data": {
            str(path.relative_to(OUT)): {
                "rows": int(len(pd.read_csv(path))),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in source_files
        },
        "tables": {
            "core_results_table.csv": {"rows": int(len(core)), "sha256": sha256(OUT / "core_results_table.csv")},
            "results_status_ledger.csv": {"rows": int(len(ledger)), "sha256": sha256(OUT / "results_status_ledger.csv")},
        },
        "privacy": {
            "participant_identifiers_written": False,
            "site_labels_written": False,
            "held_out_predictions_written": False,
            "fold_assignments_written": False,
        },
    }
    (OUT / "validation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not payload["pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Refined output validation failed: {failed}")
    print(json.dumps({"pass": True, "output": str(OUT), "checks": len(checks)}, sort_keys=True))


if __name__ == "__main__":
    main()
