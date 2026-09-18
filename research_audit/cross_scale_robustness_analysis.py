#!/usr/bin/env python3
"""Exploratory cross-scale robustness analysis of frozen connectome outputs.

This analysis reads only already-produced participant-level tables.  It does
not read image voxels, launch MRtrix, or modify production derivatives.  Three
conceptually matched microstructure/topology domains are scored on the exact
CN/MCI/AD cohort, stress-tested across the nine frozen acquisition strata, and
labelled post hoc exploratory/directional throughout.
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
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.formula.api as smf

from submission_sprint_analysis import (
    CONTRASTS,
    GROUPS,
    INPUTS,
    Stratum,
    _standardize,
    adjust_p,
    build_strata,
    collect_metrics,
    load_exact_cohort,
    normal_scores,
    sha256,
    stratum_counts,
)


PROJECT = Path("/home/ec2-user/exp")
DEFAULT_OUT = PROJECT / "research_audit" / "outputs" / "cross_scale_robustness_v1"
RESPONSES = ("micro_score", "topology_score", "difference_score")

# Component signs put all scores on a hypothesized disease-increase axis.
# AxD is stored as ``ad_mean`` in the historical tables.
DOMAINS: dict[str, dict[str, object]] = {
    "WholeBrain": {
        "mapping": "global",
        "network": "WholeBrain",
        "micro_components": {
            "global::microstructure::WholeBrain::fa_mean_edge_mean": -1.0,
            "global::microstructure::WholeBrain::md_mean_edge_mean": 1.0,
            "global::microstructure::WholeBrain::rd_mean_edge_mean": 1.0,
            "global::microstructure::WholeBrain::ad_mean_edge_mean": 1.0,
        },
        "topology_component": (
            "global::graph::WholeBrain::global_efficiency",
            -1.0,
        ),
    },
    "functional_DMN": {
        "mapping": "functional",
        "network": "DMN",
        "micro_components": {
            "functional::microstructure::DMN::fa_mean": -1.0,
        },
        "topology_component": (
            "functional::graph::DMN::nodal_eff",
            -1.0,
        ),
    },
    "functional_Limbic": {
        "mapping": "functional",
        "network": "Limbic",
        "micro_components": {
            "functional::microstructure::Limbic::md_mean": 1.0,
            "functional::microstructure::Limbic::rd_mean": 1.0,
            "functional::microstructure::Limbic::ad_mean": 1.0,
        },
        "topology_component": (
            "functional::graph::Limbic::nodal_eff",
            -1.0,
        ),
    },
}

MODERATORS: dict[str, dict[str, str]] = {
    "log_dti_t1_gap": {
        "label": "one-SD increase in log1p absolute DTI-T1 gap days",
        "column": "log_gap_days",
        "kind": "continuous_z",
    },
    "original_vs_processed_t1": {
        "label": "Original T1 versus Processed T1",
        "column": "t1_source",
        "kind": "original_indicator",
    },
    "siemens54_vs_other": {
        "label": "Siemens 54-direction versus all other protocol buckets",
        "column": "siemens_54",
        "kind": "binary_indicator",
    },
}


def _configuration() -> dict[str, object]:
    """Return the inferential configuration used for a stable run hash."""

    return {
        "schema_version": "1.0",
        "scientific_status": "POST_HOC_EXPLORATORY_DIRECTIONAL_EXISTING_OUTPUT",
        "domains": DOMAINS,
        "responses": list(RESPONSES),
        "difference_definition": "micro_score_minus_topology_score",
        "component_transform": "rank_normal_over_exact_eligible_CN_MCI_AD",
        "score_standardization": "z_over_full_domain_paired_subjects",
        "minimum_per_group": 5,
        "base_adjustment": [
            "age",
            "sex",
            "log_gap_when_variable",
            "T1_source_when_variable",
            "collapsed_protocol_when_variable",
        ],
        "covariance": "site_clustered_if_at_least_8_sites_else_HC3",
        "moderators": MODERATORS,
        "multiplicity": {
            "difference_three_domains_within_stratum_contrast": "Holm",
            "difference_all_strata_contrasts_domains": ["Holm", "BH-FDR"],
            "moderation_difference_three_domains_within_moderator_contrast": "Holm",
            "moderation_difference_global": ["Holm", "BH-FDR"],
        },
    }


def configuration_sha256() -> str:
    payload = json.dumps(
        _configuration(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def required_features() -> list[str]:
    features: list[str] = []
    for definition in DOMAINS.values():
        features.extend(definition["micro_components"].keys())
        features.append(definition["topology_component"][0])
    return sorted(set(features))


def _rank_component(values: pd.Series) -> pd.Series:
    """Rank-normalize finite entries without imputing missing data."""

    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    if finite.any():
        result.loc[finite] = normal_scores(numeric.loc[finite])
    return result


def build_cross_scale_scores(
    cohort: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build paired cross-scale scores and diagnosis-specific missingness.

    Components are rank-normalized across the exact eligible CN/MCI/AD cohort.
    Within each domain, all microstructure components and the topology endpoint
    must be finite for a subject to contribute.  Microstructure and topology
    composites are then standardized over that same full paired set, making
    ``difference_score = micro_score - topology_score`` an exact identity.
    """

    eligible = cohort[
        cohort["primary_eligible"] & cohort["group"].isin(GROUPS)
    ].copy()
    features = required_features()
    available = metrics[metrics["feature_id"].isin(features)][
        ["subject_id", "feature_id", "value"]
    ].copy()
    observed = set(available["feature_id"])
    absent = sorted(set(features) - observed)
    if absent:
        raise ValueError(f"Cross-scale features are absent: {absent}")
    wide = available.pivot(index="subject_id", columns="feature_id", values="value")
    base = eligible.set_index("subject_id").join(wide, how="left")

    score_records: list[pd.DataFrame] = []
    missing_records: list[dict[str, object]] = []
    component_records: list[dict[str, object]] = []

    for domain, definition in DOMAINS.items():
        micro_components: dict[str, float] = definition["micro_components"]
        topology_feature, topology_sign = definition["topology_component"]
        transformed: dict[str, pd.Series] = {}
        for feature_id, sign in micro_components.items():
            transformed[feature_id] = float(sign) * _rank_component(base[feature_id])
            component_records.append(
                {
                    "domain": domain,
                    "scale": "microstructure",
                    "feature_id": feature_id,
                    "orientation_sign": float(sign),
                    "n_finite_eligible": int(base[feature_id].notna().sum()),
                }
            )
        transformed[topology_feature] = float(topology_sign) * _rank_component(
            base[topology_feature]
        )
        component_records.append(
            {
                "domain": domain,
                "scale": "topology",
                "feature_id": topology_feature,
                "orientation_sign": float(topology_sign),
                "n_finite_eligible": int(base[topology_feature].notna().sum()),
            }
        )

        micro_frame = pd.DataFrame(
            {feature: transformed[feature] for feature in micro_components},
            index=base.index,
        )
        micro_complete = micro_frame.notna().all(axis=1)
        topology_complete = transformed[topology_feature].notna()
        paired = micro_complete & topology_complete

        for group in GROUPS:
            group_mask = base["group"].eq(group)
            n_total = int(group_mask.sum())
            n_micro = int((group_mask & micro_complete).sum())
            n_topology = int((group_mask & topology_complete).sum())
            n_paired = int((group_mask & paired).sum())
            missing_records.append(
                {
                    "domain": domain,
                    "group": group,
                    "n_total": n_total,
                    "n_micro_complete": n_micro,
                    "n_topology_available": n_topology,
                    "n_paired_complete": n_paired,
                    "missing_paired_n": n_total - n_paired,
                    "missing_paired_pct": (
                        100.0 * (n_total - n_paired) / n_total
                        if n_total
                        else math.nan
                    ),
                }
            )

        domain_frame = base.loc[paired].copy()
        if domain_frame.empty:
            raise RuntimeError(f"No complete paired subjects for {domain}")
        micro_unscaled = micro_frame.loc[paired].mean(axis=1)
        topology_unscaled = transformed[topology_feature].loc[paired]
        domain_frame["micro_score"] = _standardize(micro_unscaled)
        domain_frame["topology_score"] = _standardize(topology_unscaled)
        domain_frame["difference_score"] = (
            domain_frame["micro_score"] - domain_frame["topology_score"]
        )
        domain_frame["domain"] = domain
        domain_frame["micro_component_count"] = len(micro_components)
        domain_frame["topology_component_count"] = 1
        score_records.append(
            domain_frame.reset_index()[
                [
                    "subject_id",
                    "group",
                    "domain",
                    "micro_component_count",
                    "topology_component_count",
                    "micro_score",
                    "topology_score",
                    "difference_score",
                ]
            ]
        )

    scores = pd.concat(score_records, ignore_index=True)
    missingness = pd.DataFrame(missing_records)
    components = pd.DataFrame(component_records)
    return scores, missingness, components


def _contrast_vectors(names: Sequence[str]) -> dict[str, np.ndarray]:
    mci_name = 'C(group, Treatment(reference="CN"))[T.MCI]'
    ad_name = 'C(group, Treatment(reference="CN"))[T.AD]'
    if mci_name not in names or ad_name not in names:
        return {}
    lookup = {name: position for position, name in enumerate(names)}
    vectors: dict[str, np.ndarray] = {}
    for left, right in CONTRASTS:
        vector = np.zeros(len(names), dtype=float)
        if left == "MCI":
            vector[lookup[mci_name]] += 1.0
        elif left == "AD":
            vector[lookup[ad_name]] += 1.0
        if right == "MCI":
            vector[lookup[mci_name]] -= 1.0
        elif right == "AD":
            vector[lookup[ad_name]] -= 1.0
        vectors[f"{left}_vs_{right}"] = vector
    return vectors


def _robust_fit(formula: str, use: pd.DataFrame):
    base_fit = smf.ols(formula, data=use).fit()
    n_sites = int(use["site"].nunique())
    if n_sites >= 8:
        try:
            fit = base_fit.get_robustcov_results(
                cov_type="cluster",
                groups=use["site"].to_numpy(),
                use_correction=True,
            )
            covariance = "cluster_site"
        except Exception:
            fit = base_fit.get_robustcov_results(cov_type="HC3")
            covariance = "HC3_fallback"
    else:
        fit = base_fit.get_robustcov_results(cov_type="HC3")
        covariance = "HC3"
    return base_fit, fit, covariance


def _base_model_use(
    frame: pd.DataFrame,
    stratum: Stratum,
    minimum_per_group: int,
) -> tuple[pd.DataFrame, list[str]]:
    use = frame[frame["subject_id"].isin(stratum.subjects)].copy()
    use = use[use["group"].isin(GROUPS)].copy()
    required = [*RESPONSES, "group", "age", "sex", "site"]
    if stratum.include_gap:
        required.append("log_gap_days")
    if stratum.include_t1_source:
        required.append("t1_source")
    if stratum.include_protocol:
        required.append("protocol_bucket")
    use = use.dropna(subset=required).copy()
    counts = use["group"].value_counts().reindex(GROUPS, fill_value=0)
    if int(counts.min()) < minimum_per_group:
        return pd.DataFrame(), []

    use["age_z"] = _standardize(use["age"])
    terms = ['C(group, Treatment(reference="CN"))', "age_z"]
    if use["sex"].nunique(dropna=True) > 1:
        terms.append("C(sex)")
    if stratum.include_gap and use["log_gap_days"].nunique(dropna=True) > 1:
        use["log_gap_z"] = _standardize(use["log_gap_days"])
        terms.append("log_gap_z")
    if stratum.include_t1_source and use["t1_source"].nunique(dropna=True) > 1:
        terms.append("C(t1_source)")
    if stratum.include_protocol and use["protocol_bucket"].nunique(dropna=True) > 1:
        terms.append("C(protocol_bucket)")
    return use, terms


def fit_domain_stratum(
    frame: pd.DataFrame,
    stratum: Stratum,
    minimum_per_group: int = 5,
) -> list[dict[str, object]]:
    """Fit all three paired responses with an identical design matrix."""

    use, terms = _base_model_use(frame, stratum, minimum_per_group)
    if use.empty:
        return []
    counts = use["group"].value_counts().reindex(GROUPS, fill_value=0)
    domain = str(use["domain"].iloc[0])
    records: list[dict[str, object]] = []
    for response in RESPONSES:
        formula = f"{response} ~ " + " + ".join(terms)
        base_fit, fit, covariance = _robust_fit(formula, use)
        names = list(base_fit.model.exog_names)
        vectors = _contrast_vectors(names)
        for contrast, vector in vectors.items():
            test = fit.t_test(vector)
            ci = np.asarray(test.conf_int(alpha=0.05), dtype=float).reshape(-1)
            records.append(
                {
                    "domain": domain,
                    "response": response,
                    "stratum": stratum.name,
                    "contrast": contrast,
                    "n": int(len(use)),
                    "n_CN": int(counts["CN"]),
                    "n_MCI": int(counts["MCI"]),
                    "n_AD": int(counts["AD"]),
                    "n_sites": int(use["site"].nunique()),
                    "beta": float(np.asarray(test.effect).reshape(-1)[0]),
                    "ci95_low": float(ci[0]),
                    "ci95_high": float(ci[1]),
                    "p_value": float(np.asarray(test.pvalue).reshape(-1)[0]),
                    "median_CN": float(use.loc[use["group"].eq("CN"), response].median()),
                    "median_MCI": float(use.loc[use["group"].eq("MCI"), response].median()),
                    "median_AD": float(use.loc[use["group"].eq("AD"), response].median()),
                    "covariance": covariance,
                    "formula": formula,
                    "design_rank": int(np.linalg.matrix_rank(base_fit.model.exog)),
                    "design_columns": int(base_fit.model.exog.shape[1]),
                    "design_condition_number": float(np.linalg.cond(base_fit.model.exog)),
                }
            )
    return records


def fit_all_strata(
    scores: pd.DataFrame,
    cohort: pd.DataFrame,
    strata: Sequence[Stratum],
) -> pd.DataFrame:
    metadata = cohort[
        [
            "subject_id",
            "age",
            "sex",
            "site",
            "log_gap_days",
            "t1_source",
            "protocol_bucket",
            "siemens_54",
        ]
    ]
    analysis = scores.merge(metadata, on="subject_id", how="left", validate="m:1")
    records: list[dict[str, object]] = []
    for domain, frame in analysis.groupby("domain", sort=False):
        del domain
        for stratum in strata:
            records.extend(fit_domain_stratum(frame, stratum))
    results = pd.DataFrame(records)
    if results.empty:
        raise RuntimeError("No cross-scale contrasts were estimable")

    difference = results["response"].eq("difference_score")
    results["holm_p_difference_across_domains"] = np.nan
    results.loc[difference, "holm_p_difference_across_domains"] = (
        results.loc[difference]
        .groupby(["stratum", "contrast"], sort=False)["p_value"]
        .transform(lambda values: adjust_p(values, "holm"))
    )
    results["holm_p_difference_global"] = np.nan
    results["bh_q_difference_global"] = np.nan
    results.loc[difference, "holm_p_difference_global"] = adjust_p(
        results.loc[difference, "p_value"], "holm"
    )
    results.loc[difference, "bh_q_difference_global"] = adjust_p(
        results.loc[difference, "p_value"], "fdr_bh"
    )
    results["bh_q_all_cross_scale_tests_global"] = adjust_p(
        results["p_value"], "fdr_bh"
    )
    return results


def algebra_validation(results: pd.DataFrame) -> pd.DataFrame:
    wide = results.pivot_table(
        index=["domain", "stratum", "contrast"],
        columns="response",
        values="beta",
        aggfunc="first",
    ).reset_index()
    wide["micro_minus_topology_beta"] = (
        wide["micro_score"] - wide["topology_score"]
    )
    wide["difference_beta_error"] = (
        wide["difference_score"] - wide["micro_minus_topology_beta"]
    )
    wide["algebra_identity_within_1e_10"] = wide["difference_beta_error"].abs().le(
        1e-10
    )
    return wide


def _moderation_use(frame: pd.DataFrame, moderator: str) -> tuple[pd.DataFrame, list[str]]:
    definition = MODERATORS[moderator]
    required = [
        *RESPONSES,
        "group",
        "age",
        "sex",
        "site",
        "log_gap_days",
        "t1_source",
        "protocol_bucket",
        "siemens_54",
    ]
    use = frame[frame["group"].isin(GROUPS)].dropna(subset=required).copy()
    counts = use["group"].value_counts().reindex(GROUPS, fill_value=0)
    if int(counts.min()) < 5:
        return pd.DataFrame(), []
    use["age_z"] = _standardize(use["age"])
    use["log_gap_z"] = _standardize(use["log_gap_days"])
    kind = definition["kind"]
    if kind == "continuous_z":
        use["moderator_value"] = use["log_gap_z"]
    elif kind == "original_indicator":
        use["moderator_value"] = use["t1_source"].eq("Original").astype(float)
    elif kind == "binary_indicator":
        use["moderator_value"] = use["siemens_54"].astype(float)
    else:
        raise ValueError(f"Unknown moderator kind: {kind}")
    if use["moderator_value"].nunique() < 2:
        return pd.DataFrame(), []

    terms = [
        'C(group, Treatment(reference="CN")) * moderator_value',
        "age_z",
    ]
    if use["sex"].nunique() > 1:
        terms.append("C(sex)")
    if moderator != "log_dti_t1_gap" and use["log_gap_z"].nunique() > 1:
        terms.append("log_gap_z")
    if moderator != "original_vs_processed_t1" and use["t1_source"].nunique() > 1:
        terms.append("C(t1_source)")
    if moderator != "siemens54_vs_other" and use["protocol_bucket"].nunique() > 1:
        terms.append("C(protocol_bucket)")
    return use, terms


def _moderation_vectors(names: Sequence[str]) -> dict[str, np.ndarray]:
    mci = 'C(group, Treatment(reference="CN"))[T.MCI]:moderator_value'
    ad = 'C(group, Treatment(reference="CN"))[T.AD]:moderator_value'
    if mci not in names or ad not in names:
        return {}
    lookup = {name: position for position, name in enumerate(names)}
    output: dict[str, np.ndarray] = {}
    for left, right in CONTRASTS:
        vector = np.zeros(len(names), dtype=float)
        if left == "MCI":
            vector[lookup[mci]] += 1.0
        elif left == "AD":
            vector[lookup[ad]] += 1.0
        if right == "MCI":
            vector[lookup[mci]] -= 1.0
        elif right == "AD":
            vector[lookup[ad]] -= 1.0
        output[f"{left}_vs_{right}"] = vector
    return output


def fit_moderation(
    scores: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    metadata = cohort[
        [
            "subject_id",
            "age",
            "sex",
            "site",
            "log_gap_days",
            "t1_source",
            "protocol_bucket",
            "siemens_54",
        ]
    ]
    analysis = scores.merge(metadata, on="subject_id", how="left", validate="m:1")
    records: list[dict[str, object]] = []
    for domain, frame in analysis.groupby("domain", sort=False):
        for moderator, definition in MODERATORS.items():
            use, terms = _moderation_use(frame, moderator)
            if use.empty:
                continue
            counts = use["group"].value_counts().reindex(GROUPS, fill_value=0)
            for response in RESPONSES:
                formula = f"{response} ~ " + " + ".join(terms)
                base_fit, fit, covariance = _robust_fit(formula, use)
                vectors = _moderation_vectors(list(base_fit.model.exog_names))
                for contrast, vector in vectors.items():
                    test = fit.t_test(vector)
                    ci = np.asarray(test.conf_int(alpha=0.05), dtype=float).reshape(-1)
                    records.append(
                        {
                            "domain": domain,
                            "response": response,
                            "moderator": moderator,
                            "moderator_label": definition["label"],
                            "contrast": contrast,
                            "n": int(len(use)),
                            "n_CN": int(counts["CN"]),
                            "n_MCI": int(counts["MCI"]),
                            "n_AD": int(counts["AD"]),
                            "n_sites": int(use["site"].nunique()),
                            "moderation_beta": float(
                                np.asarray(test.effect).reshape(-1)[0]
                            ),
                            "ci95_low": float(ci[0]),
                            "ci95_high": float(ci[1]),
                            "p_value": float(np.asarray(test.pvalue).reshape(-1)[0]),
                            "covariance": covariance,
                            "formula": formula,
                            "design_rank": int(
                                np.linalg.matrix_rank(base_fit.model.exog)
                            ),
                            "design_columns": int(base_fit.model.exog.shape[1]),
                            "design_condition_number": float(
                                np.linalg.cond(base_fit.model.exog)
                            ),
                        }
                    )
    results = pd.DataFrame(records)
    if results.empty:
        raise RuntimeError("No acquisition-moderation models were estimable")
    difference = results["response"].eq("difference_score")
    results["holm_p_difference_across_domains"] = np.nan
    results.loc[difference, "holm_p_difference_across_domains"] = (
        results.loc[difference]
        .groupby(["moderator", "contrast"], sort=False)["p_value"]
        .transform(lambda values: adjust_p(values, "holm"))
    )
    results["holm_p_difference_global"] = np.nan
    results["bh_q_difference_global"] = np.nan
    results.loc[difference, "holm_p_difference_global"] = adjust_p(
        results.loc[difference, "p_value"], "holm"
    )
    results.loc[difference, "bh_q_difference_global"] = adjust_p(
        results.loc[difference, "p_value"], "fdr_bh"
    )
    return results


def moderation_algebra_validation(results: pd.DataFrame) -> pd.DataFrame:
    wide = results.pivot_table(
        index=["domain", "moderator", "contrast"],
        columns="response",
        values="moderation_beta",
        aggfunc="first",
    ).reset_index()
    wide["micro_minus_topology_beta"] = (
        wide["micro_score"] - wide["topology_score"]
    )
    wide["difference_beta_error"] = (
        wide["difference_score"] - wide["micro_minus_topology_beta"]
    )
    wide["algebra_identity_within_1e_10"] = wide["difference_beta_error"].abs().le(
        1e-10
    )
    return wide


def leave_one_site_out(
    scores: pd.DataFrame,
    cohort: pd.DataFrame,
    full_stratum: Stratum,
    full_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = cohort[
        [
            "subject_id",
            "age",
            "sex",
            "site",
            "log_gap_days",
            "t1_source",
            "protocol_bucket",
            "siemens_54",
        ]
    ]
    analysis = scores.merge(metadata, on="subject_id", how="left", validate="m:1")
    full_lookup = full_results[
        full_results["stratum"].eq("full_exact")
        & full_results["contrast"].eq("AD_vs_CN")
    ].set_index(["domain", "response"])["beta"]
    records: list[dict[str, object]] = []
    for domain, frame in analysis.groupby("domain", sort=False):
        full_domain = frame[frame["subject_id"].isin(full_stratum.subjects)].copy()
        for site in sorted(full_domain["site"].dropna().astype(str).unique()):
            retained_ids = frozenset(
                full_domain.loc[~full_domain["site"].eq(site), "subject_id"]
            )
            omitted = Stratum(
                name=f"full_exact_without_site_{site}",
                subjects=retained_ids,
                include_protocol=full_stratum.include_protocol,
                include_gap=full_stratum.include_gap,
                include_t1_source=full_stratum.include_t1_source,
            )
            result = pd.DataFrame(fit_domain_stratum(full_domain, omitted))
            if result.empty:
                continue
            view = result[result["contrast"].eq("AD_vs_CN")]
            for row in view.itertuples(index=False):
                full_beta = float(full_lookup.loc[(domain, row.response)])
                records.append(
                    {
                        "domain": domain,
                        "response": row.response,
                        "omitted_site": site,
                        "n": row.n,
                        "n_CN": row.n_CN,
                        "n_MCI": row.n_MCI,
                        "n_AD": row.n_AD,
                        "n_sites": row.n_sites,
                        "full_beta": full_beta,
                        "leave_one_site_out_beta": row.beta,
                        "absolute_change": abs(row.beta - full_beta),
                        "signed_change": row.beta - full_beta,
                        "sign_flip": bool(
                            np.sign(row.beta) != np.sign(full_beta)
                            and row.beta != 0
                            and full_beta != 0
                        ),
                        "p_value_descriptive": row.p_value,
                    }
                )
    detail = pd.DataFrame(records)
    summaries: list[dict[str, object]] = []
    for (domain, response), frame in detail.groupby(["domain", "response"]):
        maximum = frame.loc[frame["absolute_change"].idxmax()]
        summaries.append(
            {
                "domain": domain,
                "response": response,
                "full_beta": float(frame["full_beta"].iloc[0]),
                "n_sites_omitted": int(len(frame)),
                "minimum_leave_one_site_out_beta": float(
                    frame["leave_one_site_out_beta"].min()
                ),
                "maximum_leave_one_site_out_beta": float(
                    frame["leave_one_site_out_beta"].max()
                ),
                "maximum_absolute_change": float(maximum["absolute_change"]),
                "most_influential_omitted_site": str(maximum["omitted_site"]),
                "sign_flip_count": int(frame["sign_flip"].sum()),
            }
        )
    return detail, pd.DataFrame(summaries)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], limit: int = 30) -> str:
    view = frame.loc[:, list(columns)].head(limit).copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(
            lambda value: "NA" if not np.isfinite(value) else f"{value:.4g}"
        )

    def cell(value: object) -> str:
        if pd.isna(value):
            return "NA"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(map(cell, view.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows])


def make_figures(
    results: pd.DataFrame,
    loo_detail: pd.DataFrame,
    out: Path,
) -> None:
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=False)
    order = [
        "full_exact",
        "timing_le90",
        "timing_le180",
        "same_phase",
        "siemens54",
        "timing_le90_siemens54",
        "timing_le90_siemens54_overlap_sites",
        "original_t1",
        "legacy_qc_include",
    ]
    labels = [value.replace("_", " ") for value in order]
    domains = list(DOMAINS)
    colors = ["#2678B2", "#F07E26", "#5B9A45"]

    difference = results[
        results["response"].eq("difference_score")
        & results["contrast"].eq("AD_vs_CN")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.7), sharey=True)
    y = np.arange(len(order))
    for ax, domain, color in zip(axes, domains, colors):
        frame = difference[difference["domain"].eq(domain)].set_index("stratum").reindex(order)
        beta = frame["beta"].to_numpy(dtype=float)
        low = frame["ci95_low"].to_numpy(dtype=float)
        high = frame["ci95_high"].to_numpy(dtype=float)
        ax.errorbar(
            beta,
            y,
            xerr=np.vstack([beta - low, high - beta]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2.5,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(domain.replace("_", " "))
        ax.set_xlabel("Microstructure minus topology beta")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Exploratory AD-CN cross-scale difference across acquisition restrictions",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "ad_cn_difference_forest.png", dpi=240)
    plt.close(fig)

    paired = results[results["contrast"].eq("AD_vs_CN")].pivot_table(
        index=["domain", "stratum"], columns="response", values="beta", aggfunc="first"
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7.4, 6.5))
    for domain, color in zip(domains, colors):
        frame = paired[paired["domain"].eq(domain)]
        ax.scatter(
            frame["topology_score"],
            frame["micro_score"],
            s=55,
            label=domain.replace("_", " "),
            color=color,
            alpha=0.85,
        )
    bound = max(
        0.25,
        float(
            np.nanmax(
                np.abs(paired[["topology_score", "micro_score"]].to_numpy(dtype=float))
            )
        )
        * 1.1,
    )
    ax.plot([-bound, bound], [-bound, bound], "--", color="grey", linewidth=0.9)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_xlabel("Oriented topology AD-CN beta")
    ax.set_ylabel("Oriented microstructure AD-CN beta")
    ax.set_title("Paired cross-scale disease effects across nine strata")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "ad_cn_micro_vs_topology.png", dpi=240)
    plt.close(fig)

    loo = loo_detail[loo_detail["response"].eq("difference_score")]
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    positions = np.arange(len(domains))
    for position, domain, color in zip(positions, domains, colors):
        frame = loo[loo["domain"].eq(domain)]
        jitter = np.linspace(-0.16, 0.16, len(frame)) if len(frame) else np.array([])
        ax.scatter(
            np.full(len(frame), position) + jitter,
            frame["leave_one_site_out_beta"],
            s=18,
            alpha=0.65,
            color=color,
        )
        if len(frame):
            ax.hlines(
                frame["full_beta"].iloc[0],
                position - 0.28,
                position + 0.28,
                colors="black",
                linewidth=1.5,
            )
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(positions, [value.replace("_", " ") for value in domains])
    ax.set_ylabel("AD-CN microstructure minus topology beta")
    ax.set_title("Leave-one-site-out influence on the full-cohort difference")
    fig.tight_layout()
    fig.savefig(figure_dir / "ad_cn_difference_leave_one_site_out.png", dpi=240)
    plt.close(fig)


def write_summary(
    out: Path,
    counts: pd.DataFrame,
    missingness: pd.DataFrame,
    results: pd.DataFrame,
    moderation: pd.DataFrame,
    loo_summary: pd.DataFrame,
) -> None:
    ad_cn = results[
        results["response"].eq("difference_score")
        & results["contrast"].eq("AD_vs_CN")
    ].copy()
    ad_cn["stratum_order"] = pd.Categorical(
        ad_cn["stratum"], categories=counts["stratum"].tolist(), ordered=True
    )
    ad_cn = ad_cn.sort_values(["domain", "stratum_order"])
    moderation_difference = moderation[
        moderation["response"].eq("difference_score")
    ].sort_values("p_value")
    text = f"""# Existing-output cross-scale robustness analysis

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**Inferential status:** post hoc exploratory/directional analysis of historical participant-level outputs. This is neither a corrected image-processing result nor confirmatory evidence, and it does not establish novelty.

## Question and score construction

This analysis asks whether disease-group separation is stronger for frozen microstructure endpoints than for a conceptually matched topology endpoint after measured acquisition adjustment. It uses three paired domains: whole brain (-FA, +MD, +RD, +AxD versus -global efficiency), functional DMN (-FA versus -nodal efficiency), and functional limbic (+MD, +RD, +AxD versus -nodal efficiency). Each component was rank-normalized over exact-manifest eligible CN/MCI/AD participants. A participant had to have every component in that domain. Microstructure and topology composites were z-standardized over the same full paired participant set, then `difference = microstructure - topology` was formed exactly.

A positive diagnosis contrast for the difference means greater separation on the oriented microstructure score than on the oriented topology score on this analysis scale. It does **not** mean that microstructure is biologically more important, that topology is invalid, or that acquisition caused the observed difference.

## Exact cohort strata

{_markdown_table(counts, ['stratum', 'n_total', 'n_CN', 'n_MCI', 'n_AD', 'n_sites', 'n_protocol_buckets'])}

## Paired completeness by diagnosis

{_markdown_table(missingness, ['domain', 'group', 'n_total', 'n_micro_complete', 'n_topology_available', 'n_paired_complete', 'missing_paired_pct'])}

## AD-CN paired-scale difference across all nine strata

{_markdown_table(ad_cn, ['domain', 'stratum', 'n', 'n_CN', 'n_MCI', 'n_AD', 'beta', 'ci95_low', 'ci95_high', 'p_value', 'holm_p_difference_across_domains', 'holm_p_difference_global', 'bh_q_difference_global'])}

Holm correction across the three domains is reported for every stratum/contrast. Global Holm and BH-FDR values cover all 81 difference tests (three domains, nine strata, three diagnosis contrasts). The directly fitted difference beta is algebraically checked against the microstructure beta minus topology beta.

## Formal acquisition-moderation sensitivities

{_markdown_table(moderation_difference, ['domain', 'moderator', 'contrast', 'n', 'moderation_beta', 'ci95_low', 'ci95_high', 'p_value', 'holm_p_difference_across_domains', 'holm_p_difference_global', 'bh_q_difference_global'], 30)}

These interaction models are exploratory effect-modification checks, not causal tests. Binary moderation contrasts Original versus Processed T1 and Siemens-54 versus other protocols; the timing coefficient is per one SD of log1p absolute DTI-T1 gap. Sparse joint diagnosis/acquisition support and correlated acquisition factors limit interpretation.

## Leave-one-site-out influence for full AD-CN

{_markdown_table(loo_summary, ['domain', 'response', 'full_beta', 'minimum_leave_one_site_out_beta', 'maximum_leave_one_site_out_beta', 'maximum_absolute_change', 'most_influential_omitted_site', 'sign_flip_count'])}

Leave-one-site-out estimates are influence diagnostics, not independent replications. A stable sign only shows that no single observed site reverses the fitted estimate; it does not remove multi-site bias.

## Non-negotiable limitations

1. Every score derives from historical mixed/repaired outputs. Exact metadata adjustment cannot repair image preprocessing, parcellation, tractography, matrix validity, or spatial-QC defects.
2. Rank-normal composites encode a prespecified direction and equal component weighting. The difference is an analysis-scale contrast, not a physical-unit comparison.
3. Complete pairing improves algebraic comparability but can introduce selection bias. Diagnosis-specific missingness is reported rather than assumed benign.
4. Protocol, site, T1 source, and DTI-T1 interval are partially entangled. Restriction and interaction results may be unstable where common support is small, especially for AD.
5. All tests are post hoc exploratory/directional. A publication claim requires an exact prior-art audit and confirmation in the corrected, locked pipeline run.
"""
    (out / "cross_scale_robustness_summary.md").write_text(text, encoding="utf-8")


def write_run_manifest(
    out: Path,
    input_paths: dict[str, Path],
    counts: pd.DataFrame,
) -> None:
    script_path = Path(__file__).resolve()
    upstream_script = script_path.with_name("submission_sprint_analysis.py")
    config = _configuration()
    manifest = {
        "schema_version": "1.0",
        "release": "cross_scale_robustness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "POST_HOC_EXPLORATORY_DIRECTIONAL_EXISTING_OUTPUT",
        "image_or_voxel_data_read": False,
        "image_processing_run": False,
        "production_outputs_modified": False,
        "novelty_claim_made": False,
        "configuration": config,
        "configuration_sha256": configuration_sha256(),
        "code": {
            "analysis_script": {
                "path": str(script_path),
                "sha256": sha256(script_path),
            },
            "upstream_submission_script": {
                "path": str(upstream_script),
                "sha256": sha256(upstream_script),
            },
        },
        "inputs": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for key, path in input_paths.items()
        },
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "strata": counts.to_dict(orient="records"),
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_release(out: Path) -> dict[str, object]:
    required = [
        "cross_scale_subject_scores.csv",
        "cross_scale_component_dictionary.csv",
        "cross_scale_missingness_by_diagnosis.csv",
        "cross_scale_contrasts.csv",
        "cross_scale_contrast_algebra_validation.csv",
        "acquisition_moderation.csv",
        "acquisition_moderation_algebra_validation.csv",
        "leave_one_site_out_ad_cn.csv",
        "leave_one_site_out_ad_cn_summary.csv",
        "stratum_counts.csv",
        "cross_scale_robustness_summary.md",
        "run_manifest.json",
        "figures/ad_cn_difference_forest.png",
        "figures/ad_cn_micro_vs_topology.png",
        "figures/ad_cn_difference_leave_one_site_out.png",
    ]
    absent = [relative for relative in required if not (out / relative).is_file()]
    scores = pd.read_csv(out / "cross_scale_subject_scores.csv")
    results = pd.read_csv(out / "cross_scale_contrasts.csv")
    algebra = pd.read_csv(out / "cross_scale_contrast_algebra_validation.csv")
    moderation = pd.read_csv(out / "acquisition_moderation.csv")
    moderation_algebra = pd.read_csv(
        out / "acquisition_moderation_algebra_validation.csv"
    )
    missingness = pd.read_csv(out / "cross_scale_missingness_by_diagnosis.csv")
    loo = pd.read_csv(out / "leave_one_site_out_ad_cn.csv")
    paired_identity_error = (
        scores["difference_score"]
        - (scores["micro_score"] - scores["topology_score"])
    ).abs()
    z_summary = scores.groupby("domain")[["micro_score", "topology_score"]].agg(
        ["mean", lambda values: values.std(ddof=0)]
    )
    z_values = z_summary.to_numpy(dtype=float)
    z_means = z_values[:, [0, 2]]
    z_sds = z_values[:, [1, 3]]
    difference = results[results["response"].eq("difference_score")]
    moderation_difference = moderation[moderation["response"].eq("difference_score")]
    checks: dict[str, object] = {
        "required_files_present": not absent,
        "missing_required_files": absent,
        "three_domains_present": set(scores["domain"]) == set(DOMAINS),
        "three_responses_present": set(results["response"]) == set(RESPONSES),
        "nine_strata_present": results["stratum"].nunique() == 9,
        "three_contrasts_present": set(results["contrast"])
        == {"MCI_vs_CN", "AD_vs_CN", "AD_vs_MCI"},
        "complete_contrast_grid_243": len(results) == 3 * 9 * 3 * 3,
        "difference_test_count_81": len(difference) == 3 * 9 * 3,
        "moderation_difference_test_count_27": len(moderation_difference)
        == 3 * 3 * 3,
        "missingness_grid_9": len(missingness) == 3 * 3,
        "paired_subject_identity_within_1e_12": bool(
            paired_identity_error.max() <= 1e-12
        ),
        "score_means_zero_within_1e_12": bool(np.max(np.abs(z_means)) <= 1e-12),
        "score_sds_one_within_1e_12": bool(np.max(np.abs(z_sds - 1.0)) <= 1e-12),
        "contrast_algebra_within_1e_10": bool(
            algebra["algebra_identity_within_1e_10"].all()
        ),
        "contrast_max_algebra_error": float(algebra["difference_beta_error"].abs().max()),
        "moderation_algebra_within_1e_10": bool(
            moderation_algebra["algebra_identity_within_1e_10"].all()
        ),
        "moderation_max_algebra_error": float(
            moderation_algebra["difference_beta_error"].abs().max()
        ),
        "all_p_values_bounded": bool(
            results["p_value"].between(0, 1).all()
            and moderation["p_value"].between(0, 1).all()
        ),
        "all_difference_adjusted_p_bounded": bool(
            difference[
                [
                    "holm_p_difference_across_domains",
                    "holm_p_difference_global",
                    "bh_q_difference_global",
                ]
            ]
            .apply(lambda values: values.between(0, 1).all())
            .all()
        ),
        "leave_one_site_out_all_domains_responses": bool(
            set(loo["domain"]) == set(DOMAINS)
            and set(loo["response"]) == set(RESPONSES)
        ),
        "leave_one_site_out_at_least_20_sites_each": bool(
            loo.groupby(["domain", "response"])["omitted_site"].nunique().min()
            >= 20
        ),
        "manifest_declares_no_image_processing": bool(
            not json.loads((out / "run_manifest.json").read_text())["image_processing_run"]
        ),
    }
    checks["pass"] = bool(
        all(value for key, value in checks.items() if key not in {"missing_required_files", "contrast_max_algebra_error", "moderation_max_algebra_error"})
    )
    checks["files"] = {
        str(path.relative_to(out)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "release_validation.json"
    }
    return checks


def run_release(
    out: Path,
    input_paths: dict[str, Path] = INPUTS,
) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing release: {out}")
    for key, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing input {key}: {path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        cohort = load_exact_cohort(input_paths["exact_manifest"])
        metrics = collect_metrics(input_paths)
        strata = build_strata(cohort)
        counts = stratum_counts(cohort, strata)
        scores, missingness, components = build_cross_scale_scores(cohort, metrics)
        results = fit_all_strata(scores, cohort, strata)
        algebra = algebra_validation(results)
        moderation = fit_moderation(scores, cohort)
        moderation_algebra = moderation_algebra_validation(moderation)
        full_stratum = next(value for value in strata if value.name == "full_exact")
        loo_detail, loo_summary = leave_one_site_out(
            scores, cohort, full_stratum, results
        )

        scores.to_csv(temporary / "cross_scale_subject_scores.csv", index=False)
        components.to_csv(temporary / "cross_scale_component_dictionary.csv", index=False)
        missingness.to_csv(
            temporary / "cross_scale_missingness_by_diagnosis.csv", index=False
        )
        results.to_csv(temporary / "cross_scale_contrasts.csv", index=False)
        algebra.to_csv(
            temporary / "cross_scale_contrast_algebra_validation.csv", index=False
        )
        moderation.to_csv(temporary / "acquisition_moderation.csv", index=False)
        moderation_algebra.to_csv(
            temporary / "acquisition_moderation_algebra_validation.csv", index=False
        )
        loo_detail.to_csv(temporary / "leave_one_site_out_ad_cn.csv", index=False)
        loo_summary.to_csv(
            temporary / "leave_one_site_out_ad_cn_summary.csv", index=False
        )
        counts.to_csv(temporary / "stratum_counts.csv", index=False)
        make_figures(results, loo_detail, temporary)
        write_summary(
            temporary,
            counts,
            missingness,
            results,
            moderation,
            loo_summary,
        )
        write_run_manifest(temporary, input_paths, counts)
        validation = validate_release(temporary)
        (temporary / "release_validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
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
    print(f"PASS: exploratory cross-scale release written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
