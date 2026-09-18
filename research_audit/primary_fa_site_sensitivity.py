#!/usr/bin/env python3
"""Site-confounding addendum for the frozen primary whole-brain FA endpoint.

This bounded analysis reads only the exact cohort manifest and the already
generated participant-level global microstructure table.  It does not read
image voxels or rerun any connectome stage.  Three diagnosis contrasts are
estimated under three explicit site-support scopes:

* all available primary-analysis subjects with site fixed effects;
* sites containing both members of a contrast, fitting only that pair; and
* sites containing CN, MCI, and AD, fitting all three groups.

FA is rank-normalized again inside each exact fitted scope, matching the frozen
endpoint analysis while keeping each coefficient tied to its actual analysis
population.  Site-clustered CR1 covariance with t inference on G-1 degrees of
freedom is the claim gate.  Protocol is intentionally omitted because adding
the collapsed protocol bucket to the site-fixed-effect designs is rank
deficient in the locked data; that aliasing is emitted as a diagnostic.
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
import scipy
from scipy.stats import t as student_t
import statsmodels
import statsmodels.formula.api as smf

from submission_sprint_analysis import (
    CONTRASTS,
    GROUPS,
    adjust_p,
    normal_scores,
    sha256,
)


PROJECT = Path("/home/ec2-user/exp")
SPRINT = PROJECT / "research_audit" / "outputs" / "submission_sprint_v1"
DEFAULT_OUT = (
    PROJECT / "research_audit" / "outputs" / "primary_fa_site_sensitivity_v2"
)
PRIMARY_FEATURE = "global::microstructure::WholeBrain::fa_mean_edge_mean"
STRICT_STRATUM = "timing_le90_siemens54_overlap_sites"
SCOPES = ("full_site_fe", "pair_common_support", "all3_site_support")
DEFAULT_INPUTS = {
    "exact_manifest": SPRINT / "exact_manifest_cohort.csv",
    "global_fa_subject_table": Path(
        "/data/derivatives/qc/analysis_cohort/03_global_microstructure_live/"
        "live_global_microstructure_subject_table.csv"
    ),
    "upstream_targeted_contrasts": SPRINT / "targeted_contrasts.csv",
    "upstream_run_manifest": SPRINT / "run_manifest.json",
    "upstream_release_validation": SPRINT / "release_validation.json",
}


def _configuration() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "scientific_status": (
            "POST_HOC_SITE_CONFOUNDING_ADDENDUM_TO_PRESPECIFIED_PRIMARY_ENDPOINT"
        ),
        "primary_feature": PRIMARY_FEATURE,
        "scopes": list(SCOPES),
        "contrasts": [f"{left}_vs_{right}" for left, right in CONTRASTS],
        "rank_normalization": (
            "Rank-based inverse-normal scores Phi^-1((rank-0.5)/n), recomputed "
            "within each exact fitted scope"
        ),
        "model": (
            "rank_z ~ diagnosis + age_z + sex + log1p_absolute_DTI_T1_gap_z + "
            "T1_source + C(site)"
        ),
        "protocol_handling": (
            "Collapsed protocol omitted; site plus protocol is partially aliased "
            "and rank deficient in every locked fitted scope. Site fixed effects "
            "absorb site-constant acquisition differences but do not identify an "
            "independent protocol effect."
        ),
        "covariance_claim_gate": (
            "site-clustered CR1 small-sample correction; t inference with G-1 df"
        ),
        "pair_support": (
            "Sites containing both named groups; only that diagnosis pair is fitted"
        ),
        "all3_support": (
            "Sites containing CN, MCI, and AD; all three diagnoses are fitted"
        ),
        "multiplicity": {
            "within_each_scope_three_contrasts": "Holm",
            "diagnostic_global_nine_tests": ["Holm", "BH-FDR"],
        },
        "upstream_comparators": ["full_exact", STRICT_STRATUM],
        "image_or_voxel_data_read": False,
    }


def configuration_sha256() -> str:
    payload = json.dumps(
        _configuration(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _contrast_name(left: str, right: str) -> str:
    return f"{left}_vs_{right}"


def _sequence_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    ordered = frame.loc[:, list(columns)].sort_values(list(columns)).copy()
    payload = ordered.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _standardize(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    scale = float(numeric.std(ddof=0))
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - float(numeric.mean())) / scale


def _support_sites(frame: pd.DataFrame, required_groups: Iterable[str]) -> set[str]:
    required = set(required_groups)
    observed = frame.groupby("site", sort=True)["group"].agg(set)
    return set(observed[observed.map(required.issubset)].index.astype(str))


def load_analysis_frame(
    input_paths: dict[str, Path] = DEFAULT_INPUTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(input_paths["exact_manifest"])
    required_manifest = {
        "subject_id",
        "group",
        "primary_eligible",
        "age",
        "sex",
        "log_gap_days",
        "t1_source",
        "protocol_bucket",
    }
    absent = sorted(required_manifest - set(manifest.columns))
    if absent:
        raise ValueError(f"Exact manifest is missing columns: {absent}")
    if len(manifest) != 530 or manifest["subject_id"].nunique() != 530:
        raise ValueError("Exact manifest must contain 530 unique subjects")
    if manifest["subject_id"].duplicated().any():
        raise ValueError("Duplicate subjects in exact manifest")

    eligible = manifest[
        manifest["primary_eligible"].fillna(False).astype(bool)
        & manifest["group"].isin(GROUPS)
    ].copy()
    expected = {"CN": 251, "MCI": 186, "AD": 78}
    observed = eligible["group"].value_counts().to_dict()
    if any(int(observed.get(group, 0)) != count for group, count in expected.items()):
        raise ValueError(f"Unexpected exact eligible diagnosis counts: {observed}")

    fa = pd.read_csv(input_paths["global_fa_subject_table"])
    required_fa = {"subject_id", "group", "fa_mean_edge_mean"}
    absent = sorted(required_fa - set(fa.columns))
    if absent:
        raise ValueError(f"Frozen global FA table is missing columns: {absent}")
    if fa["subject_id"].duplicated().any():
        raise ValueError("Duplicate subjects in frozen global FA table")
    fa = fa[["subject_id", "group", "fa_mean_edge_mean"]].rename(
        columns={"group": "frozen_table_group", "fa_mean_edge_mean": "fa_value"}
    )

    merged = eligible.merge(fa, on="subject_id", how="left", validate="1:1")
    available = pd.to_numeric(merged["fa_value"], errors="coerce").notna()
    mismatch = available & merged["group"].astype(str).ne(
        merged["frozen_table_group"].astype(str)
    )
    if mismatch.any():
        examples = merged.loc[
            mismatch, ["subject_id", "group", "frozen_table_group"]
        ]
        raise ValueError(f"Diagnosis mismatch in frozen FA table:\n{examples}")

    cohort_status = merged[["subject_id", "group"]].copy()
    cohort_status["fa_available"] = available.to_numpy()
    cohort_status["status"] = np.where(
        available, "included_complete_case", "missing_frozen_FA"
    )

    analysis = merged.loc[available].copy()
    analysis["fa_value"] = pd.to_numeric(analysis["fa_value"], errors="raise")
    analysis["age"] = pd.to_numeric(analysis["age"], errors="coerce")
    analysis["log_gap_days"] = pd.to_numeric(
        analysis["log_gap_days"], errors="coerce"
    )
    analysis["site"] = analysis["subject_id"].astype(str).str.slice(0, 3)
    analysis["sex"] = analysis["sex"].astype(str)
    analysis["t1_source"] = analysis["t1_source"].astype(str)
    analysis["protocol_bucket"] = analysis["protocol_bucket"].astype(str)
    complete_columns = [
        "subject_id",
        "group",
        "fa_value",
        "age",
        "sex",
        "log_gap_days",
        "t1_source",
        "site",
        "protocol_bucket",
    ]
    if analysis[complete_columns].isna().any().any():
        missing = analysis[complete_columns].isna().sum()
        raise ValueError(f"Unexpected missing fitted values:\n{missing[missing > 0]}")
    if not np.isfinite(analysis[["fa_value", "age", "log_gap_days"]]).all().all():
        raise ValueError("Non-finite fitted values in primary FA analysis")
    return analysis[complete_columns].copy(), cohort_status


def _scope_frame(
    analysis: pd.DataFrame,
    scope: str,
    left: str,
    right: str,
) -> tuple[pd.DataFrame, set[str], Sequence[str], str]:
    if scope == "full_site_fe":
        sites = set(analysis["site"].astype(str))
        fitted_groups: Sequence[str] = GROUPS
        reference = "CN"
    elif scope == "pair_common_support":
        sites = _support_sites(analysis, (left, right))
        fitted_groups = (right, left)
        reference = right
    elif scope == "all3_site_support":
        sites = _support_sites(analysis, GROUPS)
        fitted_groups = GROUPS
        reference = "CN"
    else:
        raise ValueError(f"Unknown scope: {scope}")

    use = analysis[
        analysis["site"].isin(sites) & analysis["group"].isin(fitted_groups)
    ].copy()
    if use.empty or int(use["site"].nunique()) < 8:
        raise RuntimeError(f"Insufficient sites for {scope} {_contrast_name(left, right)}")
    counts = use["group"].value_counts().reindex(fitted_groups, fill_value=0)
    if int(counts.min()) < 5:
        raise RuntimeError(f"Insufficient group size for {scope} {_contrast_name(left, right)}")
    use["rank_z"] = normal_scores(use["fa_value"])
    use["age_z"] = _standardize(use["age"])
    use["log_gap_z"] = _standardize(use["log_gap_days"])
    return use, sites, fitted_groups, reference


def _test_vector(
    names: Sequence[str], left: str, right: str, reference: str
) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(names)}
    vector = np.zeros(len(names), dtype=float)
    for group, sign in ((left, 1.0), (right, -1.0)):
        if group == reference:
            continue
        name = f'C(group, Treatment(reference="{reference}"))[T.{group}]'
        if name not in lookup:
            raise ValueError(f"Missing diagnosis coefficient: {name}")
        vector[lookup[name]] += sign
    return vector


def _manual_cluster_statistics(
    base_fit,
    use: pd.DataFrame,
    vector: np.ndarray,
) -> dict[str, float]:
    """Independent CR1 sandwich calculation from the fitted design matrix."""

    design = np.asarray(base_fit.model.exog, dtype=float)
    outcome = np.asarray(base_fit.model.endog, dtype=float)
    parameters = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residuals = outcome - design @ parameters
    n, k = design.shape
    groups = use["site"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    g = len(unique_groups)
    bread = np.linalg.inv(design.T @ design)
    meat = np.zeros((k, k), dtype=float)
    for group in unique_groups:
        mask = groups == group
        score = design[mask].T @ residuals[mask]
        meat += np.outer(score, score)
    correction = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    covariance = correction * bread @ meat @ bread
    beta = float(vector @ parameters)
    standard_error = float(np.sqrt(vector @ covariance @ vector))
    statistic = beta / standard_error
    degrees_freedom = float(g - 1)
    p_value = float(2.0 * student_t.sf(abs(statistic), df=degrees_freedom))
    critical = float(student_t.ppf(0.975, df=degrees_freedom))
    return {
        "manual_beta": beta,
        "manual_standard_error": standard_error,
        "manual_t_value": statistic,
        "manual_p_value": p_value,
        "manual_ci95_low": beta - critical * standard_error,
        "manual_ci95_high": beta + critical * standard_error,
        "manual_inference_df": degrees_freedom,
        "manual_cr1_correction": correction,
    }


def fit_primary_fa(
    analysis: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, object]] = []
    subject_rows: list[pd.DataFrame] = []
    support_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    verification_rows: list[dict[str, object]] = []

    for scope in SCOPES:
        for left, right in CONTRASTS:
            contrast = _contrast_name(left, right)
            use, sites, fitted_groups, reference = _scope_frame(
                analysis, scope, left, right
            )
            counts = use["group"].value_counts().reindex(GROUPS, fill_value=0)
            analysis_set_sha = _sequence_sha256(use["subject_id"])
            rank_transform_sha = _frame_sha256(
                use, ["subject_id", "fa_value", "rank_z"]
            )
            formula = (
                f'rank_z ~ C(group, Treatment(reference="{reference}")) + '
                "age_z + C(sex) + log_gap_z + C(t1_source) + C(site)"
            )
            base_fit = smf.ols(formula, data=use).fit()
            design_rank = int(np.linalg.matrix_rank(base_fit.model.exog))
            design_columns = int(base_fit.model.exog.shape[1])
            if design_rank != design_columns:
                raise RuntimeError(f"Rank-deficient site-FE model: {scope} {contrast}")
            vector = _test_vector(
                list(base_fit.model.exog_names), left, right, reference
            )
            robust = base_fit.get_robustcov_results(
                cov_type="cluster",
                groups=use["site"].to_numpy(),
                use_correction=True,
                df_correction=True,
                use_t=True,
            )
            test = robust.t_test(vector)
            ci = np.asarray(test.conf_int(alpha=0.05), dtype=float).reshape(-1)
            beta = float(np.asarray(test.effect).reshape(-1)[0])
            standard_error = float(np.asarray(test.sd).reshape(-1)[0])
            t_value = float(np.asarray(test.tvalue).reshape(-1)[0])
            p_value = float(np.asarray(test.pvalue).reshape(-1)[0])
            inference_df = float(getattr(robust, "df_resid_inference"))

            medians = use.groupby("group")["fa_value"].median().reindex(GROUPS)
            result_rows.append(
                {
                    "endpoint": PRIMARY_FEATURE,
                    "scope": scope,
                    "contrast": contrast,
                    "left_group": left,
                    "right_group": right,
                    "fitted_groups": "+".join(fitted_groups),
                    "n": int(len(use)),
                    "n_CN": int(counts["CN"]),
                    "n_MCI": int(counts["MCI"]),
                    "n_AD": int(counts["AD"]),
                    "n_sites": int(use["site"].nunique()),
                    "median_FA_CN": float(medians["CN"])
                    if np.isfinite(medians["CN"])
                    else math.nan,
                    "median_FA_MCI": float(medians["MCI"])
                    if np.isfinite(medians["MCI"])
                    else math.nan,
                    "median_FA_AD": float(medians["AD"])
                    if np.isfinite(medians["AD"])
                    else math.nan,
                    "rank_normal_beta": beta,
                    "standard_error": standard_error,
                    "ci95_low": float(ci[0]),
                    "ci95_high": float(ci[1]),
                    "t_value": t_value,
                    "inference_df": inference_df,
                    "p_value": p_value,
                    "covariance": "cluster_site_CR1_t",
                    "claim_gate_covariance": True,
                    "formula": formula,
                    "design_rank": design_rank,
                    "design_columns": design_columns,
                    "design_condition_number": float(
                        np.linalg.cond(base_fit.model.exog)
                    ),
                    "analysis_set_sha256": analysis_set_sha,
                    "rank_transform_sha256": rank_transform_sha,
                }
            )

            subjects = use[
                [
                    "subject_id",
                    "group",
                    "site",
                    "fa_value",
                    "rank_z",
                    "age",
                    "age_z",
                    "sex",
                    "log_gap_days",
                    "log_gap_z",
                    "t1_source",
                    "protocol_bucket",
                ]
            ].copy()
            subjects.insert(0, "contrast", contrast)
            subjects.insert(0, "scope", scope)
            subjects["analysis_set_sha256"] = analysis_set_sha
            subjects["rank_transform_sha256"] = rank_transform_sha
            subject_rows.append(subjects)

            support_rows.append(
                {
                    "scope": scope,
                    "contrast": contrast,
                    "fitted_groups": "+".join(fitted_groups),
                    "n": int(len(use)),
                    "n_CN": int(counts["CN"]),
                    "n_MCI": int(counts["MCI"]),
                    "n_AD": int(counts["AD"]),
                    "n_sites": int(use["site"].nunique()),
                    "analysis_set_sha256": analysis_set_sha,
                }
            )
            available_counts = (
                analysis[analysis["site"].isin(sites)]
                .groupby(["site", "group"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=GROUPS, fill_value=0)
            )
            for site, row in available_counts.iterrows():
                membership_rows.append(
                    {
                        "scope": scope,
                        "contrast": contrast,
                        "site": str(site).zfill(3),
                        "fitted_groups": "+".join(fitted_groups),
                        "n_CN_available": int(row["CN"]),
                        "n_MCI_available": int(row["MCI"]),
                        "n_AD_available": int(row["AD"]),
                    }
                )

            manual = _manual_cluster_statistics(base_fit, use, vector)
            verification_rows.append(
                {
                    "scope": scope,
                    "contrast": contrast,
                    **manual,
                    "reported_beta": beta,
                    "reported_standard_error": standard_error,
                    "reported_t_value": t_value,
                    "reported_p_value": p_value,
                    "reported_ci95_low": float(ci[0]),
                    "reported_ci95_high": float(ci[1]),
                    "reported_inference_df": inference_df,
                    "beta_abs_error": abs(manual["manual_beta"] - beta),
                    "standard_error_abs_error": abs(
                        manual["manual_standard_error"] - standard_error
                    ),
                    "p_value_abs_error": abs(manual["manual_p_value"] - p_value),
                    "ci_max_abs_error": max(
                        abs(manual["manual_ci95_low"] - float(ci[0])),
                        abs(manual["manual_ci95_high"] - float(ci[1])),
                    ),
                }
            )

    results = pd.DataFrame(result_rows)
    results["p_holm_within_scope"] = results.groupby("scope", sort=False)[
        "p_value"
    ].transform(lambda values: adjust_p(values, "holm"))
    results["p_holm_global_9"] = adjust_p(results["p_value"], "holm")
    results["q_bh_global_9"] = adjust_p(results["p_value"], "fdr_bh")
    return (
        results,
        pd.concat(subject_rows, ignore_index=True),
        pd.DataFrame(support_rows),
        pd.DataFrame(membership_rows),
        pd.DataFrame(verification_rows),
    )


def protocol_alias_diagnostics(
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in SCOPES:
        for left, right in CONTRASTS:
            contrast = _contrast_name(left, right)
            use, _, _, reference = _scope_frame(analysis, scope, left, right)
            base_formula = (
                f'rank_z ~ C(group, Treatment(reference="{reference}")) + '
                "age_z + C(sex) + log_gap_z + C(t1_source) + C(site)"
            )
            formulas = (
                ("site_fe", base_formula),
                ("site_fe_plus_protocol", base_formula + " + C(protocol_bucket)"),
            )
            site_protocol_counts = use.groupby("site")["protocol_bucket"].nunique()
            protocol_site_counts = use.groupby("protocol_bucket")["site"].nunique()
            for model, formula in formulas:
                fit = smf.ols(formula, data=use).fit()
                rank = int(np.linalg.matrix_rank(fit.model.exog))
                columns = int(fit.model.exog.shape[1])
                rows.append(
                    {
                        "scope": scope,
                        "contrast": contrast,
                        "model": model,
                        "n": int(len(use)),
                        "n_sites": int(use["site"].nunique()),
                        "n_protocol_buckets": int(
                            use["protocol_bucket"].nunique()
                        ),
                        "n_site_protocol_cells": int(
                            use[["site", "protocol_bucket"]]
                            .drop_duplicates()
                            .shape[0]
                        ),
                        "n_sites_with_multiple_protocols": int(
                            site_protocol_counts.gt(1).sum()
                        ),
                        "n_protocols_seen_at_one_site_only": int(
                            protocol_site_counts.eq(1).sum()
                        ),
                        "design_rank": rank,
                        "design_columns": columns,
                        "rank_deficiency": columns - rank,
                        "design_full_rank": bool(rank == columns),
                        "design_condition_number": float(
                            np.linalg.cond(fit.model.exog)
                        ),
                        "formula": formula,
                    }
                )
    return pd.DataFrame(rows)


def upstream_comparison(
    results: pd.DataFrame,
    upstream_path: Path,
) -> pd.DataFrame:
    upstream = pd.read_csv(upstream_path)
    required = {
        "feature_id",
        "stratum",
        "contrast",
        "n",
        "n_CN",
        "n_MCI",
        "n_AD",
        "n_sites",
        "rank_normal_beta",
        "ci95_low",
        "ci95_high",
        "p_value",
        "holm_p_within_family_stratum",
    }
    absent = sorted(required - set(upstream.columns))
    if absent:
        raise ValueError(f"Upstream contrast table is missing columns: {absent}")
    upstream = upstream[
        upstream["feature_id"].eq(PRIMARY_FEATURE)
        & upstream["stratum"].isin(["full_exact", STRICT_STRATUM])
    ].copy()
    if len(upstream) != 6:
        raise ValueError("Expected three pooled and three strict upstream FA rows")
    upstream["analysis"] = np.where(
        upstream["stratum"].eq("full_exact"),
        "upstream_pooled_full_exact",
        "upstream_strict_le90_siemens54_overlap",
    )
    upstream["scope"] = upstream["stratum"]
    upstream["p_adjusted"] = upstream["holm_p_within_family_stratum"]
    upstream["adjustment"] = "upstream_Holm_primary_family_within_stratum"
    upstream["population_note"] = np.where(
        upstream["stratum"].eq("full_exact"),
        "same 514 FA-complete subjects; protocol covariate, no site fixed effects",
        "75-subject <=90-day Siemens54 overlap restriction; no site fixed effects",
    )
    upstream_view = upstream[
        [
            "analysis",
            "scope",
            "contrast",
            "n",
            "n_CN",
            "n_MCI",
            "n_AD",
            "n_sites",
            "rank_normal_beta",
            "ci95_low",
            "ci95_high",
            "p_value",
            "p_adjusted",
            "adjustment",
            "population_note",
        ]
    ]

    current = results.copy()
    current["analysis"] = current["scope"]
    current["p_adjusted"] = current["p_holm_within_scope"]
    current["adjustment"] = "Holm_across_three_contrasts_within_scope"
    current["population_note"] = current["scope"].map(
        {
            "full_site_fe": (
                "same 514 FA-complete subjects; site fixed effects, protocol omitted"
            ),
            "pair_common_support": (
                "contrast-specific sites containing both groups; only pair fitted"
            ),
            "all3_site_support": (
                "sites containing CN, MCI, and AD; all three groups fitted"
            ),
        }
    )
    current_view = current[
        [
            "analysis",
            "scope",
            "contrast",
            "n",
            "n_CN",
            "n_MCI",
            "n_AD",
            "n_sites",
            "rank_normal_beta",
            "ci95_low",
            "ci95_high",
            "p_value",
            "p_adjusted",
            "adjustment",
            "population_note",
        ]
    ]
    return pd.concat([upstream_view, current_view], ignore_index=True)


def _markdown_table(
    frame: pd.DataFrame, columns: Sequence[str], limit: int = 50
) -> str:
    view = frame.loc[:, list(columns)].head(limit).copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(
            lambda value: "NA" if not np.isfinite(value) else f"{value:.5g}"
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


def make_figure(comparison: pd.DataFrame, out: Path) -> None:
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=False)
    analyses = [
        "upstream_pooled_full_exact",
        "full_site_fe",
        "pair_common_support",
        "all3_site_support",
        "upstream_strict_le90_siemens54_overlap",
    ]
    labels = [
        "Upstream pooled full",
        "Full + site FE",
        "Pair common sites + site FE",
        "All-three sites + site FE",
        "Upstream strict acquisition subset",
    ]
    colors = ["#9A9A9A", "#2678B2", "#F07E26", "#5B9A45", "#7A5195"]
    contrasts = [f"{left}_vs_{right}" for left, right in CONTRASTS]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.7), sharey=True)
    y = np.arange(len(analyses))
    for ax, contrast in zip(axes, contrasts):
        frame = comparison[comparison["contrast"].eq(contrast)].set_index(
            "analysis"
        )
        frame = frame.reindex(analyses)
        beta = frame["rank_normal_beta"].to_numpy(dtype=float)
        low = frame["ci95_low"].to_numpy(dtype=float)
        high = frame["ci95_high"].to_numpy(dtype=float)
        for index, color in enumerate(colors):
            ax.errorbar(
                beta[index],
                y[index],
                xerr=np.array(
                    [[beta[index] - low[index]], [high[index] - beta[index]]]
                ),
                fmt="o",
                color=color,
                ecolor=color,
                capsize=2.5,
            )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(contrast.replace("_vs_", " vs "))
        ax.set_xlabel("Adjusted rank-normal FA beta")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Primary whole-brain FA: pooled, site-adjusted, and strict estimates")
    fig.tight_layout()
    fig.savefig(figure_dir / "primary_fa_site_sensitivity_forest.png", dpi=240)
    plt.close(fig)


def write_summary(
    out: Path,
    results: pd.DataFrame,
    support: pd.DataFrame,
    alias: pd.DataFrame,
    comparison: pd.DataFrame,
    cohort_status: pd.DataFrame,
) -> None:
    ordered = results.copy()
    ordered["scope"] = pd.Categorical(ordered["scope"], SCOPES, ordered=True)
    ordered = ordered.sort_values(["scope", "contrast"])
    adcn = ordered[ordered["contrast"].eq("AD_vs_CN")]
    full = adcn[adcn["scope"].eq("full_site_fe")].iloc[0]
    pair = adcn[adcn["scope"].eq("pair_common_support")].iloc[0]
    all3 = adcn[adcn["scope"].eq("all3_site_support")].iloc[0]
    pooled = comparison[
        comparison["analysis"].eq("upstream_pooled_full_exact")
        & comparison["contrast"].eq("AD_vs_CN")
    ].iloc[0]
    strict = comparison[
        comparison["analysis"].eq("upstream_strict_le90_siemens54_overlap")
        & comparison["contrast"].eq("AD_vs_CN")
    ].iloc[0]
    attenuation = 100.0 * (
        1.0 - abs(float(full["rank_normal_beta"])) / abs(float(pooled["rank_normal_beta"]))
    )
    global_survivors = int(results["p_holm_global_9"].lt(0.05).sum())
    bh_survivors = int(results["q_bh_global_9"].lt(0.05).sum())
    missing = cohort_status[~cohort_status["fa_available"]]
    plus_protocol = alias[alias["model"].eq("site_fe_plus_protocol")]

    text = f"""# Primary whole-brain FA site-confounding addendum

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**Status:** post hoc site-confounding sensitivity of the prespecified primary endpoint, using frozen historical participant summaries. No image or voxel data were read, and no pipeline stage was rerun.

## Bottom line

The negative AD-CN FA association remains directionally consistent after explicit site adjustment, but its inferential strength depends on the support definition. In the full 514-subject site-fixed-effect model, AD-CN is supported after Holm correction across the three diagnosis contrasts (beta={full['rank_normal_beta']:.4f}, 95% CI {full['ci95_low']:.4f} to {full['ci95_high']:.4f}, raw p={full['p_value']:.4g}, within-scope Holm p={full['p_holm_within_scope']:.4g}). The coefficient is {attenuation:.1f}% smaller in magnitude than the upstream pooled estimate (beta={pooled['rank_normal_beta']:.4f}).

That support does **not** generalize to the two stricter site-support gates for AD-CN: pair-common-support Holm p={pair['p_holm_within_scope']:.4g}, and all-three-site-support Holm p={all3['p_holm_within_scope']:.4g}. No result survives the diagnostic Holm family across all nine site-adjusted tests (n={global_survivors}); no result passes global BH-FDR at 0.05 (n={bh_survivors}). The correct conclusion is therefore: **full-sample site-adjusted support with stable negative direction, but incomplete robustness to common-site restriction and global multiplicity**. It is not defensible to call the primary FA result universally site-robust.

The upstream strict acquisition subset is a different, much smaller population (n={int(strict['n'])}); its AD-CN estimate is beta={strict['rank_normal_beta']:.4f}, p={strict['p_value']:.4g}. Loss of support there cannot by itself distinguish a smaller effect from low precision or selection.

## Why this analysis was needed

The upstream model clustered standard errors by site but did not include site fixed effects. Clustering changes uncertainty; it does not remove between-site diagnosis/acquisition imbalance from the coefficient. Every model below includes `C(site)` and adjusts age, sex, log1p absolute DTI-T1 gap, and T1 source. FA is rank-normalized within the exact fitted population for each scope.

Site-clustered CR1 covariance with small-sample correction and t inference on G-1 degrees of freedom is the claim gate. Protocol is omitted: adding the collapsed protocol bucket makes all {len(plus_protocol)} fitted designs rank deficient. This is partial site/protocol aliasing, not proof that every protocol is constant within site. Site fixed effects absorb site-constant acquisition differences, but they cannot estimate an independent protocol effect or repair within-site protocol confounding.

## Claim-gate results

{_markdown_table(ordered, ['scope', 'contrast', 'n', 'n_CN', 'n_MCI', 'n_AD', 'n_sites', 'rank_normal_beta', 'ci95_low', 'ci95_high', 'p_value', 'p_holm_within_scope', 'p_holm_global_9', 'q_bh_global_9'])}

## Exact support counts

{_markdown_table(support, ['scope', 'contrast', 'fitted_groups', 'n', 'n_CN', 'n_MCI', 'n_AD', 'n_sites', 'analysis_set_sha256'])}

## Upstream pooled and strict comparison

{_markdown_table(comparison, ['analysis', 'contrast', 'n', 'n_CN', 'n_MCI', 'n_AD', 'n_sites', 'rank_normal_beta', 'ci95_low', 'ci95_high', 'p_value', 'p_adjusted'], 20)}

These rows are not all the same estimand. The full site-FE and upstream pooled rows share the 514 FA-complete subjects but use different acquisition adjustment. Pair and all-three rows impose site common support. The upstream strict row additionally restricts timing and protocol.

## Site/protocol alias audit

{_markdown_table(alias, ['scope', 'contrast', 'model', 'n', 'n_sites', 'n_protocol_buckets', 'n_sites_with_multiple_protocols', 'design_rank', 'design_columns', 'rank_deficiency', 'design_full_rank', 'design_condition_number'], 40)}

## Data boundary

The exact manifest contains 515 eligible CN/MCI/AD records. Frozen whole-brain FA exists for 514: CN=251, MCI=186, AD=77. The one missing eligible record is:

{_markdown_table(missing, ['subject_id', 'group', 'status'])}

This addendum does not correct the historical imaging workflow, validate biological specificity, prove diagnostic discrimination, establish acquisition causation, or prove novelty. It is evidence for manuscript triage and must remain paired with the corrected-pipeline confirmation plan.
"""
    (out / "primary_fa_site_sensitivity_summary.md").write_text(
        text, encoding="utf-8"
    )


def write_run_manifest(
    out: Path,
    input_paths: dict[str, Path],
    results: pd.DataFrame,
    cohort_status: pd.DataFrame,
) -> None:
    script = Path(__file__).resolve()
    test_script = script.parent / "tests" / "test_primary_fa_site_sensitivity.py"
    upstream_script = script.with_name("submission_sprint_analysis.py")
    manifest = {
        "schema_version": "1.0",
        "release": "primary_fa_site_sensitivity_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "POST_HOC_SITE_CONFOUNDING_ADDENDUM_TO_PRESPECIFIED_PRIMARY_ENDPOINT"
        ),
        "image_or_voxel_data_read": False,
        "image_processing_run": False,
        "production_outputs_modified": False,
        "novelty_claim_made": False,
        "configuration": _configuration(),
        "configuration_sha256": configuration_sha256(),
        "code": {
            "analysis_script": {
                "path": str(script),
                "sha256": sha256(script),
            },
            "focused_test": {
                "path": str(test_script),
                "sha256": sha256(test_script),
            },
            "upstream_endpoint_script": {
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
        "eligible_records": int(len(cohort_status)),
        "fa_complete_records": int(cohort_status["fa_available"].sum()),
        "result_rows": int(len(results)),
        "analysis_set_hashes": {
            f"{row.scope}::{row.contrast}": row.analysis_set_sha256
            for row in results.itertuples(index=False)
        },
        "decision_flags": {
            "ad_cn_full_site_fe_within_scope_holm_lt_0_05": bool(
                results[results["scope"].eq("full_site_fe")]
                .set_index("contrast")
                .loc["AD_vs_CN", "p_holm_within_scope"]
                < 0.05
            ),
            "ad_cn_pair_common_support_within_scope_holm_lt_0_05": bool(
                results[results["scope"].eq("pair_common_support")]
                .set_index("contrast")
                .loc["AD_vs_CN", "p_holm_within_scope"]
                < 0.05
            ),
            "ad_cn_all3_support_within_scope_holm_lt_0_05": bool(
                results[results["scope"].eq("all3_site_support")]
                .set_index("contrast")
                .loc["AD_vs_CN", "p_holm_within_scope"]
                < 0.05
            ),
            "any_global_holm_9_lt_0_05": bool(
                results["p_holm_global_9"].lt(0.05).any()
            ),
            "any_global_bh_9_lt_0_05": bool(
                results["q_bh_global_9"].lt(0.05).any()
            ),
        },
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_artifact_manifest(out: Path) -> None:
    excluded = {"artifact_manifest.csv", "release_validation.json"}
    rows = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in excluded:
            rows.append(
                {
                    "relative_path": str(path.relative_to(out)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(rows).to_csv(out / "artifact_manifest.csv", index=False)


def validate_release(out: Path) -> dict[str, object]:
    required = [
        "primary_fa_site_fixed_effect_contrasts.csv",
        "primary_fa_model_subjects.csv",
        "primary_fa_cohort_availability.csv",
        "site_support_counts.csv",
        "site_support_membership.csv",
        "site_protocol_alias_diagnostics.csv",
        "manual_cluster_verification.csv",
        "upstream_primary_fa_comparison.csv",
        "primary_fa_site_sensitivity_summary.md",
        "figures/primary_fa_site_sensitivity_forest.png",
        "run_manifest.json",
        "artifact_manifest.csv",
    ]
    absent = [relative for relative in required if not (out / relative).is_file()]
    results = pd.read_csv(out / "primary_fa_site_fixed_effect_contrasts.csv")
    subjects = pd.read_csv(out / "primary_fa_model_subjects.csv")
    cohort = pd.read_csv(out / "primary_fa_cohort_availability.csv")
    support = pd.read_csv(out / "site_support_counts.csv")
    membership = pd.read_csv(out / "site_support_membership.csv", dtype={"site": str})
    alias = pd.read_csv(out / "site_protocol_alias_diagnostics.csv")
    manual = pd.read_csv(out / "manual_cluster_verification.csv")
    comparison = pd.read_csv(out / "upstream_primary_fa_comparison.csv")
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))

    pair_ok = True
    pair_members = membership[membership["scope"].eq("pair_common_support")]
    for row in pair_members.itertuples(index=False):
        left, right = row.contrast.split("_vs_")
        counts = {
            "CN": int(row.n_CN_available),
            "MCI": int(row.n_MCI_available),
            "AD": int(row.n_AD_available),
        }
        pair_ok = pair_ok and counts[left] > 0 and counts[right] > 0
    all3 = membership[membership["scope"].eq("all3_site_support")]
    all3_ok = bool(
        (
            all3["n_CN_available"].gt(0)
            & all3["n_MCI_available"].gt(0)
            & all3["n_AD_available"].gt(0)
        ).all()
    )

    ranks_exact = True
    set_hashes_exact = True
    for (scope, contrast), frame in subjects.groupby(["scope", "contrast"]):
        expected_rank = normal_scores(frame["fa_value"])
        ranks_exact = ranks_exact and bool(
            np.allclose(frame["rank_z"], expected_rank, atol=1e-12, rtol=0)
        )
        expected_set_hash = _sequence_sha256(frame["subject_id"])
        set_hashes_exact = set_hashes_exact and bool(
            frame["analysis_set_sha256"].eq(expected_set_hash).all()
        )

    expected_within = results.groupby("scope", sort=False)["p_value"].transform(
        lambda values: adjust_p(values, "holm")
    )
    expected_global_holm = adjust_p(results["p_value"], "holm")
    expected_global_bh = adjust_p(results["p_value"], "fdr_bh")
    base_alias = alias[alias["model"].eq("site_fe")]
    protocol_alias = alias[alias["model"].eq("site_fe_plus_protocol")]

    artifact_manifest = pd.read_csv(out / "artifact_manifest.csv")
    artifact_hashes_match = True
    for row in artifact_manifest.itertuples(index=False):
        path = out / row.relative_path
        artifact_hashes_match = (
            artifact_hashes_match
            and path.is_file()
            and int(path.stat().st_size) == int(row.bytes)
            and sha256(path) == row.sha256
        )
    input_hashes_match = all(
        Path(value["path"]).is_file()
        and sha256(Path(value["path"])) == value["sha256"]
        for value in manifest["inputs"].values()
    )
    code_hashes_match = all(
        Path(value["path"]).is_file()
        and sha256(Path(value["path"])) == value["sha256"]
        for value in manifest["code"].values()
    )

    full = results[results["scope"].eq("full_site_fe")]
    checks: dict[str, object] = {
        "required_files_present": not absent,
        "missing_required_files": absent,
        "expected_nine_claim_gate_rows": len(results) == 9,
        "three_scopes_present": set(results["scope"]) == set(SCOPES),
        "three_contrasts_present": set(results["contrast"])
        == {"MCI_vs_CN", "AD_vs_CN", "AD_vs_MCI"},
        "primary_endpoint_only": results["endpoint"].eq(PRIMARY_FEATURE).all(),
        "site_cluster_covariance_only": results["covariance"]
        .eq("cluster_site_CR1_t")
        .all(),
        "all_rows_are_claim_gate": results["claim_gate_covariance"].astype(bool).all(),
        "all_models_include_site_fixed_effects": results["formula"]
        .str.contains("C(site)", regex=False)
        .all(),
        "all_models_include_required_covariates": bool(
            results["formula"].str.contains("age_z", regex=False).all()
            and results["formula"].str.contains("C(sex)", regex=False).all()
            and results["formula"].str.contains("log_gap_z", regex=False).all()
            and results["formula"].str.contains("C(t1_source)", regex=False).all()
        ),
        "no_claim_model_includes_protocol": bool(
            ~results["formula"].str.contains("protocol", case=False).any()
        ),
        "all_claim_models_full_rank": results["design_rank"]
        .eq(results["design_columns"])
        .all(),
        "inference_df_equals_sites_minus_one": bool(
            np.allclose(results["inference_df"], results["n_sites"] - 1)
        ),
        "all_p_values_bounded": results["p_value"].between(0, 1).all(),
        "all_confidence_intervals_ordered": results["ci95_low"]
        .le(results["ci95_high"])
        .all(),
        "within_scope_holm_recomputed_exactly": bool(
            np.allclose(results["p_holm_within_scope"], expected_within)
        ),
        "global_holm_recomputed_exactly": bool(
            np.allclose(results["p_holm_global_9"], expected_global_holm)
        ),
        "global_bh_recomputed_exactly": bool(
            np.allclose(results["q_bh_global_9"], expected_global_bh)
        ),
        "exact_eligible_count_515": len(cohort) == 515,
        "exact_FA_complete_count_514": int(cohort["fa_available"].sum()) == 514,
        "exact_one_missing_FA": int((~cohort["fa_available"].astype(bool)).sum()) == 1,
        "full_scope_exact_counts": bool(
            len(full) == 3
            and full["n"].eq(514).all()
            and full["n_CN"].eq(251).all()
            and full["n_MCI"].eq(186).all()
            and full["n_AD"].eq(77).all()
            and full["n_sites"].eq(51).all()
        ),
        "support_rows_match_results": len(support) == 9,
        "pair_support_contains_both_groups_every_site": bool(pair_ok),
        "all3_support_contains_all_groups_every_site": all3_ok,
        "model_subjects_unique_within_cell": bool(
            ~subjects.duplicated(["scope", "contrast", "subject_id"]).any()
        ),
        "rank_normalization_recomputed_within_each_scope": bool(ranks_exact),
        "analysis_set_hashes_recomputed_exactly": bool(set_hashes_exact),
        "manual_cluster_beta_within_1e_10": bool(
            manual["beta_abs_error"].le(1e-10).all()
        ),
        "manual_cluster_se_within_1e_10": bool(
            manual["standard_error_abs_error"].le(1e-10).all()
        ),
        "manual_cluster_p_within_1e_10": bool(
            manual["p_value_abs_error"].le(1e-10).all()
        ),
        "manual_cluster_ci_within_1e_10": bool(
            manual["ci_max_abs_error"].le(1e-10).all()
        ),
        "site_only_alias_diagnostic_full_rank": bool(
            len(base_alias) == 9 and base_alias["design_full_rank"].astype(bool).all()
        ),
        "site_plus_protocol_rank_deficient_all_nine": bool(
            len(protocol_alias) == 9
            and (~protocol_alias["design_full_rank"].astype(bool)).all()
        ),
        "comparison_has_five_analyses_by_three_contrasts": bool(
            len(comparison) == 15
            and comparison["analysis"].nunique() == 5
            and comparison["contrast"].nunique() == 3
        ),
        "artifact_manifest_hashes_match": bool(artifact_hashes_match),
        "run_manifest_input_hashes_match": bool(input_hashes_match),
        "run_manifest_code_hashes_match": bool(code_hashes_match),
        "manifest_declares_no_image_or_voxel_read": bool(
            not manifest["image_or_voxel_data_read"]
            and not manifest["image_processing_run"]
        ),
    }
    excluded = {"missing_required_files"}
    checks["pass"] = bool(
        all(value for key, value in checks.items() if key not in excluded)
    )
    checks["decision_flags"] = manifest["decision_flags"]
    checks["maximum_manual_errors"] = {
        "beta": float(manual["beta_abs_error"].max()),
        "standard_error": float(manual["standard_error_abs_error"].max()),
        "p_value": float(manual["p_value_abs_error"].max()),
        "confidence_interval": float(manual["ci_max_abs_error"].max()),
    }
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
    input_paths: dict[str, Path] = DEFAULT_INPUTS,
) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing release: {out}")
    for key, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing input {key}: {path}")
    upstream_validation = json.loads(
        input_paths["upstream_release_validation"].read_text(encoding="utf-8")
    )
    if not upstream_validation.get("pass", False):
        raise RuntimeError("Frozen submission-sprint release is not validated")

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        analysis, cohort_status = load_analysis_frame(input_paths)
        results, subjects, support, membership, manual = fit_primary_fa(analysis)
        alias = protocol_alias_diagnostics(analysis)
        comparison = upstream_comparison(
            results, input_paths["upstream_targeted_contrasts"]
        )

        results.to_csv(
            temporary / "primary_fa_site_fixed_effect_contrasts.csv", index=False
        )
        subjects.to_csv(temporary / "primary_fa_model_subjects.csv", index=False)
        cohort_status.to_csv(
            temporary / "primary_fa_cohort_availability.csv", index=False
        )
        support.to_csv(temporary / "site_support_counts.csv", index=False)
        membership.to_csv(temporary / "site_support_membership.csv", index=False)
        alias.to_csv(
            temporary / "site_protocol_alias_diagnostics.csv", index=False
        )
        manual.to_csv(temporary / "manual_cluster_verification.csv", index=False)
        comparison.to_csv(
            temporary / "upstream_primary_fa_comparison.csv", index=False
        )
        make_figure(comparison, temporary)
        write_summary(
            temporary, results, support, alias, comparison, cohort_status
        )
        write_run_manifest(temporary, input_paths, results, cohort_status)
        write_artifact_manifest(temporary)
        validation = validate_release(temporary)
        (temporary / "release_validation.json").write_text(
            json.dumps(
                validation, indent=2, sort_keys=True, default=_json_default
            )
            + "\n",
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
    print(f"PASS: primary FA site-sensitivity release written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
