#!/usr/bin/env python3
"""Aggregate-only higher-order screen for the two-axis discovery phenotype.

The screen deliberately moves beyond scalar DTI effects while retaining a
strict distinction between discovery and confirmation.  It asks whether the
AxD MCI-vs-CN and RD AD-vs-MCI endpoints show:

1. preferential limbic/default-mode involvement relative to other cortex;
2. a network-dependent spatial reconfiguration across the two transitions;
3. dependence on connection architecture (within/between system, memory-
   system incidence, distance, hub status, or hemisphere); and
4. altered topology-microstructure coupling.

Only aggregate results are written.  Historical connectome limitations remain
applicable, and every test in this file is hypothesis-generating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
from scipy.stats import spearmanr
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from stage_specific_diffusivity_analysis import (
    ALL_NETWORKS,
    ANATOMICAL_NETWORKS,
    FUNCTIONAL_NETWORKS,
    PRIMARY_NETWORKS,
    _support_frame,
    eligible_cohort,
    fit_group_contrast,
)
from submission_sprint_analysis import (
    INPUTS,
    _standardize,
    collect_metrics,
    load_exact_cohort,
    normal_scores,
    sha256,
)


PROJECT = Path("/home/ec2-user/exp")
ANALYSIS = Path("/data/derivatives/qc/analysis_cohort")
CONNECTOMES = Path("/data/derivatives/connectomes")
MAPPING_CSV = PROJECT / "atlas" / "AAL" / "AAL3_network_mapping.csv"
NODE_GRAPH = ANALYSIS / "07_node_graph_live" / "live_node_graph_long.csv"
GLOBAL_COUPLING = ANALYSIS / "09_coupling_live" / "live_subject_level_coupling.csv"
NETWORK_CONTRASTS = (
    PROJECT
    / "research_audit"
    / "outputs"
    / "stage_specific_diffusivity_v2"
    / "network_contrasts.csv"
)
DEFAULT_OUT = PROJECT / "research_audit" / "outputs" / "higher_order_hypothesis_screen_v1"

GROUPS = ("CN", "MCI", "AD")
COMPONENTS = {"ad_mean": "axial", "rd_mean": "radial"}
ENDPOINTS = (
    ("axial_MCI_vs_CN", "ad_mean", "axial", "MCI", "CN"),
    ("radial_AD_vs_MCI", "rd_mean", "radial", "AD", "MCI"),
)
MIN_CLASS_EDGES = 20
HUB_FRACTION = 0.15

FUNCTIONAL_CORTICAL = (
    "DMN",
    "DorsalAttention",
    "Frontoparietal",
    "Limbic",
    "Salience_VAN",
    "Somatomotor",
    "Visual",
)
ANATOMICAL_CORTICAL = ("Frontal", "Limbic", "Occipital", "Parietal", "Temporal")
SYSTEM_SPECS = (
    (
        "functional_DMN_vs_other_cortical",
        "functional",
        ("DMN",),
        tuple(network for network in FUNCTIONAL_CORTICAL if network != "DMN"),
    ),
    (
        "functional_Limbic_vs_other_cortical",
        "functional",
        ("Limbic",),
        tuple(network for network in FUNCTIONAL_CORTICAL if network != "Limbic"),
    ),
    (
        "functional_memory_affective_vs_other_cortical",
        "functional",
        ("DMN", "Limbic"),
        tuple(network for network in FUNCTIONAL_CORTICAL if network not in {"DMN", "Limbic"}),
    ),
    (
        "anatomical_Limbic_vs_other_cortical_lobes",
        "anatomical",
        ("Limbic",),
        tuple(network for network in ANATOMICAL_CORTICAL if network != "Limbic"),
    ),
)


def adjust(values: Iterable[float], method: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    output = np.full(array.shape, np.nan)
    keep = np.isfinite(array)
    if keep.any():
        output[keep] = multipletests(array[keep], method=method)[1]
    return output


def meta_frame(cohort: pd.DataFrame) -> pd.DataFrame:
    frame = eligible_cohort(cohort)
    return frame[
        [
            "subject_id",
            "group",
            "age",
            "sex",
            "site",
            "log_gap_days",
            "t1_source",
        ]
    ].copy()


def system_selectivity_tests(cohort: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Test target systems against the remaining cortical systems directly."""

    meta = meta_frame(cohort)
    micro = metrics[
        metrics["feature_family"].eq("microstructure")
        & metrics["mapping"].isin(["functional", "anatomical"])
        & metrics["metric"].isin(COMPONENTS)
    ].copy()
    micro = micro.merge(meta[["subject_id"]], on="subject_id", how="inner", validate="m:1")
    micro = micro[np.isfinite(micro["value"])].copy()
    micro["feature"] = (
        micro["mapping"].astype(str)
        + "::"
        + micro["network"].astype(str)
        + "::"
        + micro["metric"].astype(str)
    )
    micro["feature_z"] = micro.groupby("feature", sort=False)["value"].transform(normal_scores)

    rows: list[dict[str, object]] = []
    for hypothesis, mapping, targets, comparators in SYSTEM_SPECS:
        for endpoint, metric, _component, left, right in ENDPOINTS:
            use = micro[
                micro["mapping"].eq(mapping)
                & micro["metric"].eq(metric)
                & micro["network"].isin(targets + comparators)
            ].copy()
            wide = use.pivot(index="subject_id", columns="network", values="feature_z")
            required_targets = [network for network in targets if network in wide]
            available_comparators = [network for network in comparators if network in wide]
            if len(required_targets) != len(targets) or len(available_comparators) != len(comparators):
                raise ValueError(f"Incomplete network inventory for {hypothesis}")
            target_complete = wide[list(targets)].notna().all(axis=1)
            comparison_count = wide[list(comparators)].notna().sum(axis=1)
            minimum_comparators = max(1, int(math.ceil(len(comparators) / 2)))
            selectivity = wide[list(targets)].mean(axis=1) - wide[list(comparators)].mean(axis=1)
            derived = pd.DataFrame(
                {
                    "subject_id": wide.index.astype(str),
                    "selectivity": selectivity,
                    "target_complete": target_complete,
                    "comparison_count": comparison_count,
                },
                index=wide.index,
            ).reset_index(drop=True)
            derived = derived[
                derived["target_complete"]
                & derived["comparison_count"].ge(minimum_comparators)
                & np.isfinite(derived["selectivity"])
            ].merge(meta, on="subject_id", how="inner", validate="1:1")
            result = fit_group_contrast(
                derived,
                "selectivity",
                {left: 1.0, right: -1.0},
                reference=right,
                scope="pair_common_sites",
                left=left,
                right=right,
            )
            result.update(
                {
                    "hypothesis": hypothesis,
                    "mapping": mapping,
                    "target_systems": "|".join(targets),
                    "comparison_systems": "|".join(comparators),
                    "endpoint": endpoint,
                    "metric": metric,
                    "contrast": f"{left}_vs_{right}",
                    "interpretation_of_positive_beta": "target-minus-comparator selectivity increases",
                    "evidence_status": "post-hoc internal discovery",
                }
            )
            rows.append(result)
    output = pd.DataFrame(rows)
    output["holm_p_eight_tests"] = adjust(output["p_value"], "holm")
    output["bh_q_eight_tests"] = adjust(output["p_value"], "fdr_bh")
    return output


def _reconfiguration_long(cohort: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    meta = meta_frame(cohort)
    micro = metrics[
        metrics["mapping"].isin(["anatomical", "functional"])
        & metrics["feature_family"].eq("microstructure")
        & metrics["metric"].isin(COMPONENTS)
    ].copy()
    micro["network"] = micro["mapping"].astype(str) + "::" + micro["network"].astype(str)
    micro = micro[micro["network"].isin(PRIMARY_NETWORKS)].copy()
    micro["component"] = micro["metric"].map(COMPONENTS)
    micro = micro.merge(meta, on="subject_id", how="inner", validate="m:1")
    micro = micro[np.isfinite(micro["value"])].copy()
    micro["feature"] = micro["network"] + "::" + micro["component"]
    micro["value_z"] = micro.groupby("feature", sort=False)["value"].transform(normal_scores)
    micro["age_z"] = _standardize(micro["age"])
    micro["gap_z"] = _standardize(micro["log_gap_days"])
    return micro


def _cell_vector(base: object, prototype: dict[str, object], network: str) -> np.ndarray:
    rows = []
    for group, component, sign in (
        ("AD", "radial", 1.0),
        ("MCI", "radial", -1.0),
        ("MCI", "axial", -1.0),
        ("CN", "axial", 1.0),
    ):
        row = dict(prototype)
        row.update({"group": group, "component": component, "network": network, "sign": sign})
        rows.append(row)
    new = pd.DataFrame(rows)
    signs = new.pop("sign").to_numpy(dtype=float)
    matrix = patsy.build_design_matrices(
        [base.model.data.design_info], new, return_type="dataframe"
    )[0]
    return signs @ np.asarray(matrix, dtype=float)


def network_reconfiguration_tests(
    cohort: pd.DataFrame, metrics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Test whether the late-RD minus early-AxD contrast varies by system."""

    long = _reconfiguration_long(cohort, metrics)
    joint_rows: list[dict[str, object]] = []
    estimate_rows: list[dict[str, object]] = []
    reference = PRIMARY_NETWORKS[0]
    for scope in ("full_site_fe", "all_three_group_sites"):
        use = _support_frame(long, scope).copy()
        formula = (
            f'value_z ~ C(group, Treatment(reference="CN")) * '
            f'C(component, Treatment(reference="axial")) * '
            f'C(network, Treatment(reference="{reference}")) + '
            "age_z + C(sex) + gap_z + C(t1_source) + C(site)"
        )
        base = smf.ols(formula, data=use).fit()
        fit = base.get_robustcov_results(
            cov_type="cluster", groups=use["site"].to_numpy(), use_correction=True
        )
        prototype = {
            "age_z": 0.0,
            "gap_z": 0.0,
            "sex": str(use["sex"].mode().iloc[0]),
            "t1_source": str(use["t1_source"].mode().iloc[0]),
            "site": str(use["site"].astype(str).sort_values().iloc[0]),
        }
        vectors = {network: _cell_vector(base, prototype, network) for network in PRIMARY_NETWORKS}
        restrictions = np.vstack(
            [vectors[network] - vectors[reference] for network in PRIMARY_NETWORKS[1:]]
        )
        joint = fit.f_test(restrictions)
        joint_rows.append(
            {
                "scope": scope,
                "n_subjects": int(use["subject_id"].nunique()),
                "n_sites": int(use["site"].nunique()),
                "n_observations": int(len(use)),
                "n_networks": len(PRIMARY_NETWORKS),
                "joint_df": len(PRIMARY_NETWORKS) - 1,
                "f_statistic": float(np.asarray(joint.fvalue).reshape(-1)[0]),
                "p_value": float(np.asarray(joint.pvalue).reshape(-1)[0]),
                "contrast_definition": "(AD-MCI radial) - (MCI-CN axial), tested for network variation",
                "covariance": "site-clustered with site fixed effects",
                "evidence_status": "post-hoc internal discovery",
            }
        )
        for network, vector in vectors.items():
            test = fit.t_test(vector)
            ci = np.asarray(test.conf_int(alpha=0.05), dtype=float).reshape(-1)
            estimate_rows.append(
                {
                    "scope": scope,
                    "network": network,
                    "double_transition_estimate": float(np.asarray(test.effect).reshape(-1)[0]),
                    "ci95_low": float(ci[0]),
                    "ci95_high": float(ci[1]),
                    "p_value_descriptive": float(np.asarray(test.pvalue).reshape(-1)[0]),
                    "interpretation": "network-specific descriptive double-transition contrast",
                }
            )

    profiles = pd.read_csv(NETWORK_CONTRASTS)
    profiles = profiles[profiles["nonredundant_primary"].fillna(False).astype(bool)]
    wide = profiles.pivot(index="network", columns="endpoint", values="beta")
    correlation = spearmanr(wide["axial_MCI_vs_CN"], wide["radial_AD_vs_MCI"])
    profile_summary = {
        "n_networks": int(len(wide)),
        "spearman_rho": float(correlation.statistic),
        "p_value_descriptive": float(correlation.pvalue),
    }
    return pd.DataFrame(joint_rows), pd.DataFrame(estimate_rows), profile_summary


def _read_matrix(subject_id: str, suffix: str) -> np.ndarray:
    matches = sorted(
        path
        for path in CONNECTOMES.glob(f"SC_AAL166_{subject_id}_I*_{suffix}.csv")
        if path.is_file() and not path.name.startswith("._")
    )
    if not matches:
        raise FileNotFoundError(
            f"No canonical AAL166 {suffix} matrix for requested participant key"
        )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one canonical AAL166 {suffix} matrix, found {len(matches)}"
        )
    path = matches[0]
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.shape != (166, 166):
        raise ValueError(f"Unexpected {suffix} matrix shape {matrix.shape}")
    return matrix


def _length_template(cohort: pd.DataFrame, row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, float, float]:
    cn_ids = eligible_cohort(cohort).loc[lambda frame: frame["group"].eq("CN"), "subject_id"]
    values: list[np.ndarray] = []
    for subject_id in cn_ids.astype(str):
        matrix = _read_matrix(subject_id, "len_mean")
        edge = matrix[row, col].astype(np.float32)
        edge[~np.isfinite(edge) | (edge <= 0)] = np.nan
        values.append(edge)
    stack = np.vstack(values)
    support = np.sum(np.isfinite(stack), axis=0)
    template = np.full(stack.shape[1], np.nan, dtype=float)
    keep = support >= max(25, int(math.ceil(0.20 * len(stack))))
    template[keep] = np.nanmedian(stack[:, keep], axis=0)
    q1, q2 = np.nanquantile(template, [1 / 3, 2 / 3])
    return template, float(q1), float(q2)


def _hub_nodes(cohort: pd.DataFrame) -> set[int]:
    graph = pd.read_csv(NODE_GRAPH, usecols=["subject_id", "node", "strength"])
    graph["subject_id"] = graph["subject_id"].astype(str)
    exact = eligible_cohort(cohort)[["subject_id", "group"]]
    graph = graph.merge(exact, on="subject_id", how="inner", validate="m:1")
    cn = graph[graph["group"].eq("CN")].copy()
    ranking = cn.groupby("node", sort=True)["strength"].mean().sort_values(ascending=False)
    count = max(1, int(math.ceil(HUB_FRACTION * 166)))
    return set(pd.to_numeric(ranking.head(count).index, errors="raise").astype(int))


def architecture_subject_summaries(
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Derive subject-level summaries in memory, returning only for modeling."""

    mapping = pd.read_csv(MAPPING_CSV).sort_values("matrix_idx")
    if mapping["matrix_idx"].tolist() != list(range(1, 167)):
        raise ValueError("AAL3 network mapping must contain contiguous matrix_idx 1..166")
    functional = mapping["functional_network"].astype(str).to_numpy()
    atlas_label = mapping["atlas_label"].astype(str).to_numpy()
    hubs = _hub_nodes(cohort)
    hub_mask_nodes = np.array([(index + 1) in hubs for index in range(166)], dtype=bool)
    side = np.array(
        ["L" if name.endswith("_L") else "R" if name.endswith("_R") else "M" for name in atlas_label]
    )

    row, col = np.triu_indices(166, k=1)
    length_template, short_cut, long_cut = _length_template(cohort, row, col)
    same_network = functional[row] == functional[col]
    memory_node = np.isin(functional, ["DMN", "Limbic"])
    memory_incident = memory_node[row] | memory_node[col]
    hub_incident = hub_mask_nodes[row] | hub_mask_nodes[col]
    lateral = (side[row] != "M") & (side[col] != "M")
    inter = lateral & (side[row] != side[col])
    intra = lateral & (side[row] == side[col])
    short = np.isfinite(length_template) & (length_template <= short_cut)
    long = np.isfinite(length_template) & (length_template >= long_cut)
    between = ~same_network

    masks = {
        "between_minus_within_functional": (between, same_network),
        "memory_incident_minus_nonmemory": (memory_incident, ~memory_incident),
        "memory_between_minus_other_between": (
            between & memory_incident,
            between & ~memory_incident,
        ),
        "long_minus_short": (long, short),
        "hub_incident_minus_peripheral": (hub_incident, ~hub_incident),
        "inter_minus_intra_hemispheric": (inter, intra),
    }
    meta = meta_frame(cohort)
    records: list[dict[str, object]] = []
    for subject_id in meta["subject_id"].astype(str):
        try:
            axial_matrix = _read_matrix(subject_id, "ad_mean")
            radial_matrix = _read_matrix(subject_id, "rd_mean")
        except FileNotFoundError:
            continue
        axial = axial_matrix[row, col]
        radial = radial_matrix[row, col]
        paired = (
            np.isfinite(axial)
            & np.isfinite(radial)
            & (axial > 0)
            & (radial > 0)
        )
        for dimension, (positive, negative) in masks.items():
            positive_valid = paired & positive
            negative_valid = paired & negative
            n_positive = int(positive_valid.sum())
            n_negative = int(negative_valid.sum())
            if min(n_positive, n_negative) < MIN_CLASS_EDGES:
                continue
            for component, values in (("axial", axial), ("radial", radial)):
                records.append(
                    {
                        "subject_id": subject_id,
                        "dimension": dimension,
                        "component": component,
                        "contrast_value": float(values[positive_valid].mean() - values[negative_valid].mean()),
                        "n_positive_edges": n_positive,
                        "n_negative_edges": n_negative,
                    }
                )
    subject = pd.DataFrame(records).merge(meta, on="subject_id", how="inner", validate="m:1")
    coverage = (
        subject.groupby(["dimension", "component", "group"], as_index=False)
        .agg(
            available=("subject_id", "nunique"),
            median_positive_edges=("n_positive_edges", "median"),
            median_negative_edges=("n_negative_edges", "median"),
        )
    )
    metadata = {
        "hub_node_count": len(hubs),
        "hub_fraction_requested": HUB_FRACTION,
        "length_template_cn_subjects": int(eligible_cohort(cohort)["group"].eq("CN").sum()),
        "short_length_cut": short_cut,
        "long_length_cut": long_cut,
        "minimum_edges_per_class": MIN_CLASS_EDGES,
        "functional_mapping_caveat": "approximate AAL3-to-functional-network crosswalk",
    }
    return subject, coverage, metadata


def architecture_tests(subject: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for endpoint, _metric, component, left, right in ENDPOINTS:
        for dimension in sorted(subject["dimension"].unique()):
            use = subject[
                subject["component"].eq(component) & subject["dimension"].eq(dimension)
            ].copy()
            result = fit_group_contrast(
                use,
                "contrast_value",
                {left: 1.0, right: -1.0},
                reference=right,
                scope="pair_common_sites",
                left=left,
                right=right,
            )
            result.update(
                {
                    "dimension": dimension,
                    "endpoint": endpoint,
                    "component": component,
                    "contrast": f"{left}_vs_{right}",
                    "interpretation_of_positive_beta": "positive-minus-negative edge-class contrast increases",
                    "evidence_status": "post-hoc internal discovery",
                }
            )
            rows.append(result)
    output = pd.DataFrame(rows)
    output["holm_p_twelve_tests"] = adjust(output["p_value"], "holm")
    output["bh_q_twelve_tests"] = adjust(output["p_value"], "fdr_bh")
    return output


def coupling_tests(cohort: pd.DataFrame) -> pd.DataFrame:
    coupling = pd.read_csv(
        GLOBAL_COUPLING,
        usecols=["subject_id", "nodal_metric", "micro_metric", "rho"],
    )
    coupling["subject_id"] = coupling["subject_id"].astype(str)
    coupling["rho"] = pd.to_numeric(coupling["rho"], errors="coerce")
    coupling = coupling[np.isfinite(coupling["rho"])].merge(
        meta_frame(cohort), on="subject_id", how="inner", validate="m:1"
    )
    rows: list[dict[str, object]] = []
    for endpoint, metric, component, left, right in ENDPOINTS:
        for nodal_metric in ("degree", "nodal_eff", "strength"):
            use = coupling[
                coupling["micro_metric"].eq(metric)
                & coupling["nodal_metric"].eq(nodal_metric)
            ].copy()
            result = fit_group_contrast(
                use,
                "rho",
                {left: 1.0, right: -1.0},
                reference=right,
                scope="pair_common_sites",
                left=left,
                right=right,
            )
            result.update(
                {
                    "endpoint": endpoint,
                    "component": component,
                    "nodal_metric": nodal_metric,
                    "coupling": f"{nodal_metric}__{metric}",
                    "contrast": f"{left}_vs_{right}",
                    "interpretation_of_positive_beta": "across-node topology-microstructure Spearman rho increases",
                    "evidence_status": "post-hoc internal discovery",
                }
            )
            rows.append(result)
    output = pd.DataFrame(rows)
    output["holm_p_six_tests"] = adjust(output["p_value"], "holm")
    output["bh_q_six_tests"] = adjust(output["p_value"], "fdr_bh")
    return output


def _short_network(network: str) -> str:
    return network.replace("anatomical::", "A:").replace("functional::", "F:")


def make_figure(
    system: pd.DataFrame,
    architecture: pd.DataFrame,
    reconfiguration: pd.DataFrame,
    out: Path,
) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5))

    profiles = pd.read_csv(NETWORK_CONTRASTS)
    profiles = profiles[profiles["nonredundant_primary"].fillna(False).astype(bool)]
    wide = profiles.pivot(index="network", columns="endpoint", values="beta")
    axes[0].scatter(wide["axial_MCI_vs_CN"], wide["radial_AD_vs_MCI"], color="#31688e")
    for network, row in wide.iterrows():
        axes[0].annotate(_short_network(network), (row["axial_MCI_vs_CN"], row["radial_AD_vs_MCI"]), fontsize=7)
    axes[0].set_xlabel("AxD effect: MCI - CN")
    axes[0].set_ylabel("RD effect: AD - MCI")
    axes[0].set_title("A. Network effect-profile reconfiguration")

    sys = system.sort_values(["endpoint", "beta"]).reset_index(drop=True)
    y = np.arange(len(sys))
    colors = np.where(sys["endpoint"].eq("axial_MCI_vs_CN"), "#31688e", "#b63679")
    axes[1].hlines(y, sys["ci95_low"], sys["ci95_high"], color=colors, linewidth=1.2)
    axes[1].scatter(sys["beta"], y, color=colors, s=28)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(y, [f"{h}\n{e.replace('_vs_', ' vs ')}" for h, e in zip(sys["hypothesis"], sys["contrast"])], fontsize=7)
    axes[1].set_xlabel("Target-minus-comparator stage effect")
    axes[1].set_title("B. Direct system-selectivity tests")

    arch = architecture.sort_values(["endpoint", "beta"]).reset_index(drop=True)
    y = np.arange(len(arch))
    colors = np.where(arch["endpoint"].eq("axial_MCI_vs_CN"), "#31688e", "#b63679")
    axes[2].hlines(y, arch["ci95_low"], arch["ci95_high"], color=colors, linewidth=1.2)
    axes[2].scatter(arch["beta"], y, color=colors, s=28)
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_yticks(y, [f"{d}\n{e.replace('_vs_', ' vs ')}" for d, e in zip(arch["dimension"], arch["contrast"])], fontsize=7)
    axes[2].set_xlabel("Edge-class contrast stage effect")
    axes[2].set_title("C. Connection-architecture tests")

    fig.suptitle("Higher-order screen of the two-axis diffusion-connectome phenotype", fontsize=14)
    fig.tight_layout()
    fig.savefig(figures / "higher_order_hypothesis_screen.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def _format_rows(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame[list(columns)].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(
    out: Path,
    system: pd.DataFrame,
    joint: pd.DataFrame,
    profile: dict[str, float],
    architecture: pd.DataFrame,
    coupling: pd.DataFrame,
) -> None:
    system_pass = int(system["holm_p_eight_tests"].lt(0.05).sum())
    architecture_pass = int(architecture["holm_p_twelve_tests"].lt(0.05).sum())
    coupling_pass = int(coupling["holm_p_six_tests"].lt(0.05).sum())
    text = f"""# Higher-order two-axis hypothesis screen

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**Status:** aggregate-only, post-hoc internal discovery; not corrected-pipeline confirmation.

## Direct system selectivity

{_format_rows(system, ['hypothesis', 'endpoint', 'n', 'n_sites', 'beta', 'p_value', 'holm_p_eight_tests'])}

Holm-significant direct selectivity tests: **{system_pass}/{len(system)}**. A network being individually significant in the stage model is not evidence that it is preferentially affected; these target-versus-comparator contrasts are the relevant test.

## Spatial reconfiguration

{_format_rows(joint, ['scope', 'n_subjects', 'n_sites', 'f_statistic', 'p_value'])}

Across the 17 nonredundant systems, the descriptive Spearman correlation between the early AxD and later RD effect profiles is **rho={profile['spearman_rho']:.3f}** (p={profile['p_value_descriptive']:.4g}). The joint test asks the stronger inferential question: whether `(AD-MCI RD) - (MCI-CN AxD)` varies by system. It does not establish a temporal or cellular mechanism.

## Connection architecture

{_format_rows(architecture, ['dimension', 'endpoint', 'n', 'n_sites', 'beta', 'p_value', 'holm_p_twelve_tests'])}

Holm-significant architecture tests: **{architecture_pass}/{len(architecture)}**. The distance template is frozen from CN median edge lengths; hubs are the top 15% of nodes by CN mean historical SIFT2-weighted strength. These definitions are internal and require corrected-pipeline confirmation.

## Topology-microstructure coupling

{_format_rows(coupling, ['coupling', 'endpoint', 'n', 'n_sites', 'beta', 'p_value', 'holm_p_six_tests'])}

Holm-significant coupling tests: **{coupling_pass}/{len(coupling)}**. Coupling is the within-subject across-node Spearman association, not functional connectivity.

## Claim boundary

This screen can nominate a higher-order hypothesis only when (1) the direct interaction or selectivity test passes its declared multiplicity family, (2) the result is coherent with the primary two-axis endpoints, and (3) an exact primary-literature audit does not show the same conjunction. Generic limbic, DMN, hub, long-range, graph-disruption, or DTI-stage claims are already established and cannot be presented as novel.
"""
    (out / "hypothesis_screen_summary.md").write_text(text, encoding="utf-8")


def write_manifest(out: Path, architecture_meta: dict[str, object], profile: dict[str, float]) -> None:
    inputs = {
        "exact_manifest": INPUTS["exact_manifest"],
        "functional_micro": INPUTS["functional_micro"],
        "anatomical_micro": INPUTS["anatomical_micro"],
        "functional_graph": INPUTS["functional_graph"],
        "anatomical_graph": INPUTS["anatomical_graph"],
        "mapping": MAPPING_CSV,
        "node_graph": NODE_GRAPH,
        "global_coupling": GLOBAL_COUPLING,
        "network_contrasts": NETWORK_CONTRASTS,
    }
    manifest = {
        "release": "higher_order_hypothesis_screen_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "post-hoc internal discovery",
        "writes_participant_data": False,
        "eligible_ceiling": {"CN": 251, "MCI": 186, "AD": 78},
        "matrix_inventory": {
            "ad_mean": len(list(CONNECTOMES.glob("SC_AAL166_*_ad_mean.csv"))),
            "rd_mean": len(list(CONNECTOMES.glob("SC_AAL166_*_rd_mean.csv"))),
            "len_mean": len(list(CONNECTOMES.glob("SC_AAL166_*_len_mean.csv"))),
        },
        "architecture": architecture_meta,
        "effect_profile": profile,
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)} for key, path in inputs.items()
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def validate(out: Path) -> dict[str, object]:
    required = (
        "system_selectivity_tests.csv",
        "network_reconfiguration_joint_tests.csv",
        "network_reconfiguration_estimates.csv",
        "architecture_interaction_tests.csv",
        "architecture_coverage.csv",
        "topology_microstructure_coupling_tests.csv",
        "hypothesis_screen_summary.md",
        "figures/higher_order_hypothesis_screen.png",
        "run_manifest.json",
    )
    missing = [name for name in required if not (out / name).exists()]
    csvs = [pd.read_csv(out / name) for name in required if name.endswith(".csv") and (out / name).exists()]
    prohibited = {
        "subject_id",
        "participant_id",
        "image_id",
        "series_id",
        "rid",
    }
    no_ids = all(not (prohibited & {column.lower() for column in frame.columns}) for frame in csvs)
    probabilities = []
    for frame in csvs:
        for column in frame.columns:
            if column == "p_value" or column.startswith("holm_p") or column.startswith("bh_q"):
                probabilities.extend(pd.to_numeric(frame[column], errors="coerce").dropna().tolist())
    bounded = all(0 <= value <= 1 for value in probabilities)
    report = {
        "pass": not missing and no_ids and bounded,
        "missing_required_files": missing,
        "no_identifier_columns": no_ids,
        "bounded_inferential_probabilities": bounded,
        "system_test_count": int(len(pd.read_csv(out / "system_selectivity_tests.csv"))) if not missing else 0,
        "architecture_test_count": int(len(pd.read_csv(out / "architecture_interaction_tests.csv"))) if not missing else 0,
        "coupling_test_count": int(len(pd.read_csv(out / "topology_microstructure_coupling_tests.csv"))) if not missing else 0,
        "files": {},
    }
    for name in required:
        path = out / name
        if path.exists():
            report["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return report


def run(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing release: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        cohort = load_exact_cohort()
        metrics = collect_metrics()
        system = system_selectivity_tests(cohort, metrics)
        joint, reconfiguration, profile = network_reconfiguration_tests(cohort, metrics)
        architecture_subject, architecture_coverage, architecture_meta = architecture_subject_summaries(cohort)
        architecture = architecture_tests(architecture_subject)
        coupling = coupling_tests(cohort)

        system.to_csv(temporary / "system_selectivity_tests.csv", index=False)
        joint.to_csv(temporary / "network_reconfiguration_joint_tests.csv", index=False)
        reconfiguration.to_csv(temporary / "network_reconfiguration_estimates.csv", index=False)
        architecture.to_csv(temporary / "architecture_interaction_tests.csv", index=False)
        architecture_coverage.to_csv(temporary / "architecture_coverage.csv", index=False)
        coupling.to_csv(temporary / "topology_microstructure_coupling_tests.csv", index=False)
        make_figure(system, architecture, reconfiguration, temporary)
        write_summary(temporary, system, joint, profile, architecture, coupling)
        write_manifest(temporary, architecture_meta, profile)
        report = validate(temporary)
        (temporary / "release_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not report["pass"]:
            raise RuntimeError(f"Release validation failed: {report}")
        os.replace(temporary, out)
    except Exception:
        failed = out.parent / f"{temporary.name}.failed"
        os.replace(temporary, failed)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
