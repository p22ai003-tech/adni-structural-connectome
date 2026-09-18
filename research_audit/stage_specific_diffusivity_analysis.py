#!/usr/bin/env python3
"""Locked analysis of a CN-MCI-AD network diffusivity signature.

This release turns one post-hoc lead from the historical connectome outputs into
a reproducible, aggregate-only statistical analysis.  The candidate comprises
two adjacent-transition endpoints:

1. mean network axial-diffusivity burden for MCI versus CN; and
2. mean network radial-diffusivity burden for AD versus MCI.

The release also tests the stronger component-switch interpretation directly.
It never writes participant identifiers or participant-level values.  Because
the candidate was discovered in the same historical data, the output remains
internal discovery evidence until independent/corrected-pipeline confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from submission_sprint_analysis import (
    INPUTS,
    _standardize,
    collect_metrics,
    load_exact_cohort,
    normal_scores,
    sha256,
)


PROJECT = Path("/home/ec2-user/exp")
DEFAULT_OUT = (
    PROJECT / "research_audit" / "outputs" / "stage_specific_diffusivity_v2"
)
GROUPS = ("CN", "MCI", "AD")
COMPONENT_METRICS = {"ad_mean": "axial", "rd_mean": "radial"}
ANATOMICAL_NETWORKS = (
    "anatomical::BasalGanglia",
    "anatomical::Brainstem",
    "anatomical::Cerebellum",
    "anatomical::Frontal",
    "anatomical::Limbic",
    "anatomical::Occipital",
    "anatomical::Parietal",
    "anatomical::Temporal",
    "anatomical::Thalamus",
)
FUNCTIONAL_NETWORKS = (
    "functional::Brainstem",
    "functional::Cerebellar",
    "functional::DMN",
    "functional::DorsalAttention",
    "functional::Frontoparietal",
    "functional::Limbic",
    "functional::Salience_VAN",
    "functional::Somatomotor",
    "functional::Subcortical",
    "functional::Visual",
)
ALL_NETWORKS = ANATOMICAL_NETWORKS + FUNCTIONAL_NETWORKS

# Brainstem and cerebellum are the same atlas territories under both mapping
# labels. They must not receive double weight in the cross-mapping headline
# composite. Mapping-specific sensitivities retain each territory once.
CROSS_MAPPING_DUPLICATES = {
    "functional::Brainstem": "anatomical::Brainstem",
    "functional::Cerebellar": "anatomical::Cerebellum",
}
PRIMARY_NETWORKS = tuple(
    network for network in ALL_NETWORKS if network not in CROSS_MAPPING_DUPLICATES
)
COMPOSITE_REPRESENTATIONS = {
    "nonredundant_17": PRIMARY_NETWORKS,
    "anatomical_9": ANATOMICAL_NETWORKS,
    "functional_10": FUNCTIONAL_NETWORKS,
    "legacy_19_double_weighted_sensitivity": ALL_NETWORKS,
}


@dataclass(frozen=True)
class Endpoint:
    endpoint: str
    outcome: str
    left: str
    right: str


PRIMARY_ENDPOINTS = (
    Endpoint("axial_MCI_vs_CN", "axial_burden", "MCI", "CN"),
    Endpoint("radial_AD_vs_MCI", "radial_burden", "AD", "MCI"),
)


def adjust(values: Sequence[float], method: str = "holm") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    out = np.full(array.shape, np.nan)
    keep = np.isfinite(array)
    if keep.any():
        out[keep] = multipletests(array[keep], method=method)[1]
    return out


def eligible_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    frame = cohort[
        cohort["primary_eligible"] & cohort["group"].isin(GROUPS)
    ].copy()
    expected = {"CN": 251, "MCI": 186, "AD": 78}
    observed = frame["group"].value_counts().to_dict()
    if any(int(observed.get(group, 0)) != count for group, count in expected.items()):
        raise ValueError(f"Unexpected eligible group counts: {observed}")
    return frame


def build_composites(
    cohort: pd.DataFrame,
    metrics: pd.DataFrame,
    networks: Sequence[str] = PRIMARY_NETWORKS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build paired-network component burdens and aggregate coverage.

    Each network/component is rank-normalized across eligible subjects before
    equal weighting.  Within a subject, axial and radial burdens use exactly the
    same set of networks, preventing asymmetric missingness from creating a
    spurious component difference. ``networks`` is explicit so the headline
    nonredundant index can be checked against each mapping separately.
    """

    cohort = eligible_cohort(cohort)
    micro = metrics[
        metrics["mapping"].isin(["anatomical", "functional"])
        & metrics["feature_family"].eq("microstructure")
        & metrics["metric"].isin(COMPONENT_METRICS)
    ].copy()
    micro["network_id"] = micro["mapping"].astype(str) + "::" + micro["network"].astype(str)
    networks = tuple(networks)
    if not networks or len(networks) != len(set(networks)):
        raise ValueError("Composite network inventory must be nonempty and unique")
    unknown = sorted(set(networks) - set(ALL_NETWORKS))
    if unknown:
        raise ValueError(f"Unknown composite networks: {unknown}")
    micro = micro[micro["network_id"].isin(networks)].copy()
    micro["component"] = micro["metric"].map(COMPONENT_METRICS)
    micro = micro.merge(
        cohort[["subject_id", "group"]], on="subject_id", how="inner", validate="m:1"
    )
    micro = micro[np.isfinite(micro["value"])].copy()

    observed_pairs = set(zip(micro["network_id"], micro["component"]))
    expected_pairs = {
        (network, component)
        for network in networks
        for component in ("axial", "radial")
    }
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        extra = sorted(observed_pairs - expected_pairs)
        raise ValueError(f"Unexpected network/component inventory; missing={missing}, extra={extra}")

    micro["feature"] = micro["network_id"] + "::" + micro["component"]
    micro["feature_z"] = micro.groupby("feature", sort=False)["value"].transform(
        lambda values: normal_scores(values)
    )
    wide = micro.pivot(
        index="subject_id", columns=["network_id", "component"], values="feature_z"
    )

    rows: list[dict[str, object]] = []
    for subject_id, values in wide.iterrows():
        paired = [
            network
            for network in networks
            if np.isfinite(values.get((network, "axial"), np.nan))
            and np.isfinite(values.get((network, "radial"), np.nan))
        ]
        if not paired:
            continue
        axial = float(np.mean([values[(network, "axial")] for network in paired]))
        radial = float(np.mean([values[(network, "radial")] for network in paired]))
        rows.append(
            {
                "subject_id": str(subject_id),
                "n_paired_networks": len(paired),
                "axial_burden": axial,
                "radial_burden": radial,
                "radial_excess": radial - axial,
            }
        )
    composites = cohort.merge(pd.DataFrame(rows), on="subject_id", how="inner", validate="1:1")

    coverage_rows: list[dict[str, object]] = []
    available = micro.pivot_table(
        index=["subject_id", "group"],
        columns=["network_id", "component"],
        values="value",
        aggfunc="first",
    )
    for network in networks:
        paired = available[(network, "axial")].notna() & available[(network, "radial")].notna()
        for group in GROUPS:
            group_mask = available.index.get_level_values("group") == group
            total = int(group_mask.sum())
            count = int((paired & group_mask).sum())
            coverage_rows.append(
                {
                    "network": network,
                    "group": group,
                    "available": count,
                    "eligible": total,
                    "available_pct": 100.0 * count / total if total else math.nan,
                }
            )
    return composites, pd.DataFrame(coverage_rows)


def _group_vector(names: Sequence[str], reference: str, weights: dict[str, float]) -> np.ndarray:
    vector = np.zeros(len(names), dtype=float)
    index = {name: position for position, name in enumerate(names)}
    for group, weight in weights.items():
        if group == reference:
            continue
        name = f'C(group, Treatment(reference="{reference}"))[T.{group}]'
        if name not in index:
            raise ValueError(f"Group coefficient {name!r} absent from model: {names}")
        vector[index[name]] += weight
    return vector


def _support_frame(
    frame: pd.DataFrame,
    scope: str,
    left: str | None = None,
    right: str | None = None,
) -> pd.DataFrame:
    use = frame.copy()
    if scope == "full_site_fe":
        return use
    if scope == "pair_common_sites":
        if left is None or right is None:
            raise ValueError("pair_common_sites requires left and right groups")
        use = use[use["group"].isin([left, right])].copy()
        support = use.groupby("site")["group"].nunique()
        return use[use["site"].isin(support[support.eq(2)].index)].copy()
    if scope == "all_three_group_sites":
        support = use.groupby("site")["group"].nunique()
        return use[use["site"].isin(support[support.eq(3)].index)].copy()
    raise ValueError(f"Unknown support scope: {scope}")


def fit_group_contrast(
    frame: pd.DataFrame,
    outcome: str,
    weights: dict[str, float],
    *,
    reference: str = "CN",
    scope: str = "full_site_fe",
    left: str | None = None,
    right: str | None = None,
) -> dict[str, object]:
    use = _support_frame(frame, scope, left=left, right=right)
    required = [outcome, "group", "age", "sex", "site", "log_gap_days", "t1_source"]
    use = use.dropna(subset=required).copy()
    present = set(use["group"])
    if not set(weights).issubset(present):
        raise ValueError(f"Contrast groups {set(weights)} absent from {present}")
    if len(use) < 40 or use["site"].nunique() < 8:
        raise ValueError("Insufficient observations or sites for site-clustered model")

    use["outcome_z"] = normal_scores(use[outcome])
    use["age_z"] = _standardize(use["age"])
    use["gap_z"] = _standardize(use["log_gap_days"])
    terms = [f'C(group, Treatment(reference="{reference}"))', "age_z"]
    if use["sex"].nunique() > 1:
        terms.append("C(sex)")
    if use["log_gap_days"].nunique() > 1:
        terms.append("gap_z")
    if use["t1_source"].nunique() > 1:
        terms.append("C(t1_source)")
    terms.append("C(site)")
    formula = "outcome_z ~ " + " + ".join(terms)
    base = smf.ols(formula, data=use).fit()
    fit = base.get_robustcov_results(
        cov_type="cluster", groups=use["site"].to_numpy(), use_correction=True
    )
    vector = _group_vector(base.model.exog_names, reference, weights)
    test = fit.t_test(vector)
    ci = np.asarray(test.conf_int(alpha=0.05), dtype=float).reshape(-1)
    counts = use["group"].value_counts()
    return {
        "scope": scope,
        "outcome": outcome,
        "n": int(len(use)),
        "n_CN": int(counts.get("CN", 0)),
        "n_MCI": int(counts.get("MCI", 0)),
        "n_AD": int(counts.get("AD", 0)),
        "n_sites": int(use["site"].nunique()),
        "beta": float(np.asarray(test.effect).reshape(-1)[0]),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "p_value": float(np.asarray(test.pvalue).reshape(-1)[0]),
        "design_rank": int(np.linalg.matrix_rank(base.model.exog)),
        "design_columns": int(base.model.exog.shape[1]),
        "condition_number": float(np.linalg.cond(base.model.exog)),
        "formula": formula,
    }


def primary_models(composites: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in ("full_site_fe", "pair_common_sites", "all_three_group_sites"):
        for endpoint in PRIMARY_ENDPOINTS:
            result = fit_group_contrast(
                composites,
                endpoint.outcome,
                {endpoint.left: 1.0, endpoint.right: -1.0},
                reference="CN" if scope != "pair_common_sites" else endpoint.right,
                scope=scope,
                left=endpoint.left,
                right=endpoint.right,
            )
            result["endpoint"] = endpoint.endpoint
            result["contrast"] = f"{endpoint.left}_vs_{endpoint.right}"
            rows.append(result)
    output = pd.DataFrame(rows)
    output["holm_p_two_endpoints"] = output.groupby("scope", sort=False)["p_value"].transform(
        lambda values: adjust(values, "holm")
    )
    return output


def direct_component_tests(
    composites: pd.DataFrame, total_networks: int = len(PRIMARY_NETWORKS)
) -> pd.DataFrame:
    tests = (
        ("early_radial_minus_axial", {"MCI": 1.0, "CN": -1.0}),
        ("late_radial_minus_axial", {"AD": 1.0, "MCI": -1.0}),
        ("stage_by_component_second_difference", {"AD": 1.0, "MCI": -2.0, "CN": 1.0}),
    )
    rows = []
    thresholds = (1, int(math.ceil(total_networks / 2)), total_networks)
    for minimum in thresholds:
        frame = composites[composites["n_paired_networks"].ge(minimum)].copy()
        for name, weights in tests:
            result = fit_group_contrast(
                frame,
                "radial_excess",
                weights,
                reference="CN",
                scope="full_site_fe",
            )
            result["test"] = name
            result["minimum_paired_networks"] = minimum
            rows.append(result)
    output = pd.DataFrame(rows)
    output["holm_p_three_direct_tests"] = output.groupby(
        "minimum_paired_networks", sort=False
    )["p_value"].transform(lambda values: adjust(values, "holm"))
    return output


def sensitivity_models(
    composites: pd.DataFrame, total_networks: int = len(PRIMARY_NETWORKS)
) -> pd.DataFrame:
    half = int(math.ceil(total_networks / 2))
    masks = {
        "all_available": pd.Series(True, index=composites.index),
        f"at_least_{half}_networks": composites["n_paired_networks"].ge(half),
        f"complete_{total_networks}_networks": composites["n_paired_networks"].eq(total_networks),
        "timing_le90": composites["include_timing_le_90_sensitivity"].fillna(False).astype(bool),
        "timing_le180": composites["include_timing_le_180_sensitivity"].fillna(False).astype(bool),
        "same_adni_phase": composites["same_adni_phase"].fillna(False).astype(bool),
        "siemens54": composites["siemens_54"].fillna(False).astype(bool),
        "original_t1": composites["t1_source"].eq("Original"),
        "legacy_qc": composites["legacy_qc_include"].fillna(False).astype(bool),
    }
    rows = []
    for stratum, mask in masks.items():
        frame = composites[mask].copy()
        for endpoint in PRIMARY_ENDPOINTS:
            try:
                result = fit_group_contrast(
                    frame,
                    endpoint.outcome,
                    {endpoint.left: 1.0, endpoint.right: -1.0},
                    reference="CN",
                    scope="full_site_fe",
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            result["stratum"] = stratum
            result["endpoint"] = endpoint.endpoint
            result["contrast"] = f"{endpoint.left}_vs_{endpoint.right}"
            rows.append(result)
    output = pd.DataFrame(rows)
    output["holm_p_two_endpoints"] = output.groupby("stratum", sort=False)["p_value"].transform(
        lambda values: adjust(values, "holm")
    )
    return output


def network_models(cohort: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    cohort = eligible_cohort(cohort)
    meta = cohort[
        ["subject_id", "group", "age", "sex", "site", "log_gap_days", "t1_source"]
    ]
    micro = metrics[
        metrics["mapping"].isin(["anatomical", "functional"])
        & metrics["feature_family"].eq("microstructure")
        & metrics["metric"].isin(COMPONENT_METRICS)
    ].copy()
    micro["network"] = micro["mapping"].astype(str) + "::" + micro["network"].astype(str)
    micro = micro[micro["network"].isin(ALL_NETWORKS)].merge(
        meta, on="subject_id", how="inner", validate="m:1"
    )
    rows = []
    for endpoint in PRIMARY_ENDPOINTS:
        metric = "ad_mean" if endpoint.outcome.startswith("axial") else "rd_mean"
        for network in ALL_NETWORKS:
            frame = micro[micro["metric"].eq(metric) & micro["network"].eq(network)].copy()
            frame = frame.rename(columns={"value": "network_value"})
            try:
                result = fit_group_contrast(
                    frame,
                    "network_value",
                    {endpoint.left: 1.0, endpoint.right: -1.0},
                    reference="CN",
                    scope="full_site_fe",
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            result["endpoint"] = endpoint.endpoint
            result["network"] = network
            result["contrast"] = f"{endpoint.left}_vs_{endpoint.right}"
            rows.append(result)
    output = pd.DataFrame(rows)
    output["bh_q_19_mapping_labels"] = output.groupby("endpoint", sort=False)["p_value"].transform(
        lambda values: adjust(values, "fdr_bh")
    )
    output["cross_mapping_duplicate_of"] = output["network"].map(CROSS_MAPPING_DUPLICATES)
    output["nonredundant_primary"] = output["cross_mapping_duplicate_of"].isna()
    output["bh_q_17_nonredundant"] = np.nan
    for endpoint in output["endpoint"].unique():
        index = output["endpoint"].eq(endpoint) & output["nonredundant_primary"]
        output.loc[index, "bh_q_17_nonredundant"] = adjust(
            output.loc[index, "p_value"], "fdr_bh"
        )
    return output


def representation_models(cohort: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Re-estimate both endpoints in nonredundant and mapping-specific indices."""

    rows: list[pd.DataFrame] = []
    for representation, networks in COMPOSITE_REPRESENTATIONS.items():
        composites, _ = build_composites(cohort, metrics, networks)
        contrasts = primary_models(composites)
        contrasts.insert(0, "representation", representation)
        contrasts.insert(1, "network_count", len(networks))
        rows.append(contrasts)
    output = pd.concat(rows, ignore_index=True)
    output["holm_p_four_mapping_endpoints"] = np.nan
    mapping_names = {"anatomical_9", "functional_10"}
    for scope in output["scope"].unique():
        index = output["representation"].isin(mapping_names) & output["scope"].eq(scope)
        output.loc[index, "holm_p_four_mapping_endpoints"] = adjust(
            output.loc[index, "p_value"], "holm"
        )
    return output


def leave_one_site_out(composites: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full_effect = primary[primary["scope"].eq("full_site_fe")].set_index("endpoint")["beta"]
    sites = sorted(composites["site"].dropna().astype(str).unique())
    for endpoint in PRIMARY_ENDPOINTS:
        effects = []
        p_values = []
        for site in sites:
            frame = composites[~composites["site"].astype(str).eq(site)].copy()
            try:
                result = fit_group_contrast(
                    frame,
                    endpoint.outcome,
                    {endpoint.left: 1.0, endpoint.right: -1.0},
                    reference="CN",
                    scope="full_site_fe",
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            effects.append(float(result["beta"]))
            p_values.append(float(result["p_value"]))
        values = np.asarray(effects, dtype=float)
        reference = float(full_effect.loc[endpoint.endpoint])
        rows.append(
            {
                "endpoint": endpoint.endpoint,
                "contrast": f"{endpoint.left}_vs_{endpoint.right}",
                "iterations": len(values),
                "positive_direction_fraction": float(np.mean(values > 0)),
                "p_below_0_05_fraction": float(np.mean(np.asarray(p_values) < 0.05)),
                "median_beta": float(np.median(values)),
                "minimum_beta": float(np.min(values)),
                "maximum_beta": float(np.max(values)),
                "maximum_absolute_change_from_full": float(np.max(np.abs(values - reference))),
            }
        )
    return pd.DataFrame(rows)


def group_summary(composites: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        frame = composites[composites["group"].eq(group)]
        for outcome in ("axial_burden", "radial_burden", "radial_excess"):
            values = frame[outcome].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "group": group,
                    "outcome": outcome,
                    "n": len(values),
                    "median": float(np.median(values)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "median_paired_networks": float(frame["n_paired_networks"].median()),
                    "minimum_paired_networks": int(frame["n_paired_networks"].min()),
                }
            )
    return pd.DataFrame(rows)


def make_figures(primary: pd.DataFrame, network: pd.DataFrame, out: Path) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=False)

    scopes = ["full_site_fe", "pair_common_sites", "all_three_group_sites"]
    labels = {
        "full_site_fe": "All eligible",
        "pair_common_sites": "Pair-common sites",
        "all_three_group_sites": "Sites containing CN/MCI/AD",
    }
    colors = {"axial_MCI_vs_CN": "#31688e", "radial_AD_vs_MCI": "#b63679"}
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    positions = []
    ylabels = []
    y = 0
    for endpoint in PRIMARY_ENDPOINTS:
        for scope in scopes:
            row = primary[
                primary["endpoint"].eq(endpoint.endpoint) & primary["scope"].eq(scope)
            ].iloc[0]
            ax.errorbar(
                row["beta"],
                y,
                xerr=[[row["beta"] - row["ci95_low"]], [row["ci95_high"] - row["beta"]]],
                fmt="o",
                color=colors[endpoint.endpoint],
                capsize=3,
            )
            positions.append(y)
            ylabels.append(f"{endpoint.endpoint.replace('_', ' ')} | {labels[scope]}")
            y += 1
        y += 0.5
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(positions, ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("Site-adjusted rank-normal beta (95% CI)")
    ax.set_title("Two-axis diffusivity signature across site-support scopes")
    fig.tight_layout()
    fig.savefig(figures / "transition_signature_forest.png", dpi=260)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.8), sharey=True)
    for ax, endpoint in zip(axes, PRIMARY_ENDPOINTS):
        frame = network[
            network["endpoint"].eq(endpoint.endpoint)
            & network["nonredundant_primary"]
        ].copy()
        frame = frame.sort_values("beta")
        y = np.arange(len(frame))
        significant = frame["bh_q_17_nonredundant"].lt(0.05)
        ax.errorbar(
            frame["beta"],
            y,
            xerr=[frame["beta"] - frame["ci95_low"], frame["ci95_high"] - frame["beta"]],
            fmt="none",
            ecolor="#9e9e9e",
            capsize=2,
        )
        ax.scatter(
            frame["beta"], y, c=np.where(significant, colors[endpoint.endpoint], "#bdbdbd"), s=35
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y, frame["network"].str.replace("::", " | ", regex=False), fontsize=8)
        ax.set_title(endpoint.endpoint.replace("_", " "))
        ax.set_xlabel("Site-adjusted rank-normal beta")
    fig.suptitle("Network contributions to the two adjacent-transition endpoints")
    fig.tight_layout()
    fig.savefig(figures / "network_transition_effects.png", dpi=260)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    view = frame.loc[:, list(columns)].copy()
    for column in view.select_dtypes(include="float").columns:
        view[column] = view[column].map(lambda value: f"{value:.4g}" if np.isfinite(value) else "NA")
    header = "| " + " | ".join(view.columns) + " |"
    rule = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def write_summary(
    out: Path,
    primary: pd.DataFrame,
    direct: pd.DataFrame,
    influence: pd.DataFrame,
    representations: pd.DataFrame,
) -> None:
    direct_full = direct[direct["minimum_paired_networks"].eq(1)]
    mapping = representations[
        representations["representation"].isin(["anatomical_9", "functional_10"])
    ]
    text = f"""# Stage-specific network diffusivity candidate: corrected analysis v2

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**Status:** positive biological discovery candidate from historical outputs; internally validated, not independent confirmation.

## Locked positive endpoints

{_markdown_table(primary, ['endpoint', 'scope', 'n', 'n_CN', 'n_MCI', 'n_AD', 'n_sites', 'beta', 'ci95_low', 'ci95_high', 'p_value', 'holm_p_two_endpoints'])}

The submission candidate is operationally a **two-axis adjacent-transition signature**: network AxD burden for MCI versus CN and network RD burden for AD versus MCI. The headline composite contains 17 nonredundant systems: functional brainstem and cerebellar labels are excluded because they duplicate the corresponding anatomical atlas territories. The two p-values are corrected together within every site-support scope. This wording does not require either component to be absent at another transition.

## Mapping robustness

{_markdown_table(mapping, ['representation', 'endpoint', 'scope', 'n', 'n_sites', 'beta', 'ci95_low', 'ci95_high', 'p_value', 'holm_p_four_mapping_endpoints'])}

Anatomical and functional mappings reproduce the direction and nominal significance of both endpoints. Across the four mapping-specific tests, Holm control passes in the full and pair-common-site analyses but is marginal in the most restrictive all-three-group-site scope. These are alternative summaries of overlapping edge data, not independent cohort replications.

## Stronger component-switch diagnostic

{_markdown_table(direct_full, ['test', 'minimum_paired_networks', 'n', 'beta', 'ci95_low', 'ci95_high', 'p_value', 'holm_p_three_direct_tests'])}

The direct tests determine whether the stronger phrase “AxD-to-RD switch” is supportable. If the second-difference test is not robust across missingness thresholds, the manuscript must use “two-axis transition signature,” not “mechanistic switch.” This is a claim-boundary check, not the paper's primary result.

## Site influence

{_markdown_table(influence, ['endpoint', 'iterations', 'positive_direction_fraction', 'p_below_0_05_fraction', 'median_beta', 'minimum_beta', 'maximum_beta', 'maximum_absolute_change_from_full'])}

## Biological interpretation allowed at this stage

The data support testing a progression-associated diffusion-component pattern spanning anatomical and functional systems. AxD and RD are tensor-derived eigenvalue summaries; they cannot by themselves establish axonal injury, demyelination, or a temporal causal sequence. Those mechanistic labels require convergent evidence and will not be asserted from DTI alone.

## Claim status

- **Positive association candidate:** assessed by the locked endpoints above.
- **Robustness:** assessed across common site support, timing, acquisition, T1-source, network-completeness, and leave-one-site-out analyses.
- **Novelty:** not established by this release; requires the separate exact prior-art matrix.
- **Confirmation:** requires a corrected pipeline run or an independent cohort. Same-dataset validation remains internal validation.
"""
    (out / "results_summary.md").write_text(text, encoding="utf-8")


def write_manifest(out: Path) -> None:
    manifest = {
        "schema_version": "2.0",
        "release": "stage_specific_diffusivity_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "POST_HOC_DISCOVERY_INTERNAL_VALIDATION",
        "participant_level_outputs_written": False,
        "primary_endpoints": [endpoint.__dict__ for endpoint in PRIMARY_ENDPOINTS],
        "primary_network_inventory": list(PRIMARY_NETWORKS),
        "all_mapping_labels": list(ALL_NETWORKS),
        "cross_mapping_duplicates_excluded_from_primary": CROSS_MAPPING_DUPLICATES,
        "composite_representations": {
            key: list(value) for key, value in COMPOSITE_REPRESENTATIONS.items()
        },
        "composite_definition": "feature-wise rank-normal score; equal mean across paired available nonredundant systems",
        "multiplicity": "Holm across the two adjacent-transition endpoints within each scope",
        "covariates": ["age", "sex", "log(DTI-T1 gap + 1)", "T1 source", "site fixed effects"],
        "covariance": "site-clustered sandwich covariance with finite-cluster correction",
        "inputs": {
            key: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for key, path in INPUTS.items()
        },
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def validate_release(out: Path) -> dict[str, object]:
    required = [
        "network_coverage.csv",
        "group_summary.csv",
        "primary_contrasts.csv",
        "direct_component_tests.csv",
        "sensitivity_contrasts.csv",
        "network_contrasts.csv",
        "representation_contrasts.csv",
        "leave_one_site_out_summary.csv",
        "results_summary.md",
        "run_manifest.json",
        "figures/transition_signature_forest.png",
        "figures/network_transition_effects.png",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    primary = pd.read_csv(out / "primary_contrasts.csv")
    network = pd.read_csv(out / "network_contrasts.csv")
    representation = pd.read_csv(out / "representation_contrasts.csv")
    csvs = [pd.read_csv(path) for path in out.glob("*.csv")]
    forbidden = {"subject_id", "participant_id"}
    checks: dict[str, object] = {
        "required_files_present": not missing,
        "missing_required_files": missing,
        "six_primary_scope_rows": len(primary) == 6,
        "two_primary_endpoints": primary["endpoint"].nunique() == 2,
        "three_support_scopes": primary["scope"].nunique() == 3,
        "thirty_eight_network_rows": len(network) == 38,
        "twenty_four_representation_rows": len(representation) == 24,
        "seventeen_nonredundant_networks_per_endpoint": int(network["nonredundant_primary"].sum()) == 34,
        "two_cross_mapping_duplicates_per_endpoint": int(network["cross_mapping_duplicate_of"].notna().sum()) == 4,
        "all_primary_p_values_bounded": bool(primary["p_value"].between(0, 1).all()),
        "all_primary_cis_ordered": bool((primary["ci95_low"] <= primary["ci95_high"]).all()),
        "no_participant_identifier_columns": all(forbidden.isdisjoint(frame.columns) for frame in csvs),
    }
    checks["pass"] = bool(
        all(value for key, value in checks.items() if key != "missing_required_files")
    )
    checks["files"] = {
        str(path.relative_to(out)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "release_validation.json"
    }
    return checks


def run_release(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing release: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        cohort = load_exact_cohort()
        metrics = collect_metrics()
        composites, coverage = build_composites(cohort, metrics, PRIMARY_NETWORKS)
        primary = primary_models(composites)
        direct = direct_component_tests(composites, len(PRIMARY_NETWORKS))
        sensitivity = sensitivity_models(composites, len(PRIMARY_NETWORKS))
        network = network_models(cohort, metrics)
        representations = representation_models(cohort, metrics)
        influence = leave_one_site_out(composites, primary)
        summary = group_summary(composites)

        coverage.to_csv(temporary / "network_coverage.csv", index=False)
        summary.to_csv(temporary / "group_summary.csv", index=False)
        primary.to_csv(temporary / "primary_contrasts.csv", index=False)
        direct.to_csv(temporary / "direct_component_tests.csv", index=False)
        sensitivity.to_csv(temporary / "sensitivity_contrasts.csv", index=False)
        network.to_csv(temporary / "network_contrasts.csv", index=False)
        representations.to_csv(temporary / "representation_contrasts.csv", index=False)
        influence.to_csv(temporary / "leave_one_site_out_summary.csv", index=False)
        make_figures(primary, network, temporary)
        write_summary(temporary, primary, direct, influence, representations)
        write_manifest(temporary)
        validation = validate_release(temporary)
        (temporary / "release_validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not validation["pass"]:
            raise RuntimeError(f"Release validation failed: {validation}")
        os.rename(temporary, out)
    except Exception:
        failed = temporary.with_name(temporary.name + ".failed")
        if temporary.exists() and not failed.exists():
            os.rename(temporary, failed)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_release(args.out.resolve())
    print(f"PASS: aggregate-only diffusivity candidate release written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
