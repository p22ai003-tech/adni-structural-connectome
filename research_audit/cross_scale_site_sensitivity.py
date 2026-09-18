#!/usr/bin/env python3
"""Site-confounding addendum for the frozen cross-scale robustness release.

This bounded analysis starts from the immutable participant scores in
``cross_scale_robustness_v1``.  It does not read image data or recompute any
connectome.  The addendum estimates within-site diagnosis contrasts using site
fixed effects and reports two explicit common-support restrictions:

* sites containing both members of the diagnosis contrast; and
* sites containing CN, MCI, and AD (a stricter diagnostic).

The original protocol term is intentionally replaced by site fixed effects.
In these data, a design containing both site and the collapsed protocol bucket
is rank deficient.  The estimand is therefore a covariate-adjusted within-site
contrast, not a protocol-specific effect.
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
import statsmodels
import statsmodels.formula.api as smf

from cross_scale_robustness_analysis import DOMAINS, RESPONSES
from submission_sprint_analysis import (
    CONTRASTS,
    GROUPS,
    INPUTS,
    _standardize,
    adjust_p,
    load_exact_cohort,
    sha256,
)


PROJECT = Path("/home/ec2-user/exp")
UPSTREAM = PROJECT / "research_audit" / "outputs" / "cross_scale_robustness_v1"
DEFAULT_OUT = (
    PROJECT / "research_audit" / "outputs" / "cross_scale_site_sensitivity_v1"
)
DEFAULT_INPUTS = {
    "subject_scores": UPSTREAM / "cross_scale_subject_scores.csv",
    "upstream_contrasts": UPSTREAM / "cross_scale_contrasts.csv",
    "upstream_manifest": UPSTREAM / "run_manifest.json",
    "upstream_validation": UPSTREAM / "release_validation.json",
    "exact_manifest": INPUTS["exact_manifest"],
}
SCOPES = ("full_site_fe", "pair_common_support", "all3_site_support")
COVARIANCES = {
    "full_site_fe": ("cluster_site",),
    "pair_common_support": ("cluster_site", "HC3"),
    "all3_site_support": ("cluster_site", "HC3"),
}
CLAIM_GATE_COVARIANCE = {
    "full_site_fe": "cluster_site",
    "pair_common_support": "cluster_site",
    "all3_site_support": "cluster_site",
}


def _configuration() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "scientific_status": "POST_HOC_EXPLORATORY_SITE_CONFOUNDING_ADDENDUM",
        "upstream_release": "cross_scale_robustness_v1",
        "scopes": list(SCOPES),
        "responses": list(RESPONSES),
        "site_adjustment": "C(site) fixed effects",
        "protocol_handling": (
            "collapsed protocol omitted because site plus protocol is rank deficient; "
            "site fixed effects absorb all site-constant acquisition differences"
        ),
        "base_covariates": ["age_z", "sex", "log_gap_z", "T1_source"],
        "full_covariance": "site-clustered small-sample corrected",
        "restricted_covariances": ["site-clustered small-sample corrected", "HC3"],
        "claim_gate_rationale": (
            "site-clustered covariance is used for every claim gate because site "
            "fixed effects remove site means but do not guarantee independent "
            "within-site residuals; HC3 is retained as a diagnostic"
        ),
        "pair_support": (
            "domain-specific sites containing both contrast groups; only the two "
            "contrast groups enter the fitted model"
        ),
        "all3_support": (
            "domain-specific sites containing CN, MCI, and AD; all three groups "
            "enter the fitted model"
        ),
        "minimum_per_fitted_group": 5,
        "minimum_sites": 8,
        "multiplicity": {
            "within_scope_covariance_contrast_across_three_domains": "Holm",
            "claim_gate_27_difference_tests": ["Holm", "BH-FDR"],
            "claim_gate_covariance": CLAIM_GATE_COVARIANCE,
        },
    }


def configuration_sha256() -> str:
    payload = json.dumps(
        _configuration(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contrast_name(left: str, right: str) -> str:
    return f"{left}_vs_{right}"


def _support_sites(frame: pd.DataFrame, required_groups: Iterable[str]) -> set[str]:
    required = set(required_groups)
    observed = frame.groupby("site", sort=True)["group"].agg(set)
    return set(observed[observed.map(required.issubset)].index.astype(str))


def _metadata(path: Path) -> pd.DataFrame:
    cohort = load_exact_cohort(path)
    eligible = cohort[
        cohort["primary_eligible"] & cohort["group"].isin(GROUPS)
    ].copy()
    return eligible[
        [
            "subject_id",
            "group",
            "age",
            "sex",
            "site",
            "log_gap_days",
            "t1_source",
            "protocol_bucket",
        ]
    ]


def load_analysis_frame(input_paths: dict[str, Path]) -> pd.DataFrame:
    scores = pd.read_csv(input_paths["subject_scores"])
    expected = {
        "subject_id",
        "group",
        "domain",
        *RESPONSES,
    }
    absent = sorted(expected - set(scores.columns))
    if absent:
        raise ValueError(f"Subject-score input is missing columns: {absent}")
    if set(scores["domain"]) != set(DOMAINS):
        raise ValueError("Subject scores do not contain the three frozen domains")
    if scores.duplicated(["subject_id", "domain"]).any():
        raise ValueError("Duplicate subject-domain scores detected")
    metadata = _metadata(input_paths["exact_manifest"])
    merged = scores.merge(
        metadata,
        on="subject_id",
        how="left",
        validate="m:1",
        suffixes=("_score", "_manifest"),
    )
    if merged["group_manifest"].isna().any():
        raise ValueError("A scored subject is absent from the exact eligible manifest")
    mismatch = merged["group_score"].astype(str).ne(
        merged["group_manifest"].astype(str)
    )
    if mismatch.any():
        raise ValueError("Diagnosis mismatch between score release and exact manifest")
    merged["group"] = merged["group_manifest"].astype(str)
    merged["site"] = merged["site"].astype(str).str.zfill(3)
    return merged.drop(columns=["group_score", "group_manifest"])


def support_tables(
    analysis: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership: list[dict[str, object]] = []
    counts: list[dict[str, object]] = []

    def add(
        domain: str,
        scope: str,
        contrast: str,
        sites: set[str],
        fitted_groups: Sequence[str],
        frame: pd.DataFrame,
    ) -> None:
        use = frame[
            frame["site"].isin(sites) & frame["group"].isin(fitted_groups)
        ].copy()
        group_counts = use["group"].value_counts().reindex(GROUPS, fill_value=0)
        counts.append(
            {
                "domain": domain,
                "scope": scope,
                "contrast": contrast,
                "fitted_groups": "+".join(fitted_groups),
                "n": int(len(use)),
                "n_CN": int(group_counts["CN"]),
                "n_MCI": int(group_counts["MCI"]),
                "n_AD": int(group_counts["AD"]),
                "n_sites": int(use["site"].nunique()),
            }
        )
        site_counts = (
            frame[frame["site"].isin(sites)]
            .groupby(["site", "group"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=GROUPS, fill_value=0)
        )
        for site, row in site_counts.iterrows():
            membership.append(
                {
                    "domain": domain,
                    "scope": scope,
                    "contrast": contrast,
                    "site": str(site).zfill(3),
                    "fitted_groups": "+".join(fitted_groups),
                    "n_CN_available": int(row["CN"]),
                    "n_MCI_available": int(row["MCI"]),
                    "n_AD_available": int(row["AD"]),
                }
            )

    for domain, frame in analysis.groupby("domain", sort=False):
        all_sites = set(frame["site"].dropna().astype(str))
        for left, right in CONTRASTS:
            contrast = _contrast_name(left, right)
            add(domain, "full_site_fe", contrast, all_sites, GROUPS, frame)
            pair_sites = _support_sites(frame, (left, right))
            add(
                domain,
                "pair_common_support",
                contrast,
                pair_sites,
                (right, left),
                frame,
            )
            all3_sites = _support_sites(frame, GROUPS)
            add(
                domain,
                "all3_site_support",
                contrast,
                all3_sites,
                GROUPS,
                frame,
            )
    return pd.DataFrame(membership), pd.DataFrame(counts)


def _prepare_model_frame(
    frame: pd.DataFrame,
    sites: set[str],
    fitted_groups: Sequence[str],
    minimum_per_group: int = 5,
    minimum_sites: int = 8,
) -> tuple[pd.DataFrame, list[str]]:
    required = [
        *RESPONSES,
        "group",
        "age",
        "sex",
        "site",
        "log_gap_days",
        "t1_source",
    ]
    use = frame[
        frame["site"].isin(sites) & frame["group"].isin(fitted_groups)
    ].dropna(subset=required).copy()
    group_counts = use["group"].value_counts().reindex(fitted_groups, fill_value=0)
    if int(group_counts.min()) < minimum_per_group:
        return pd.DataFrame(), []
    if int(use["site"].nunique()) < minimum_sites:
        return pd.DataFrame(), []
    use["age_z"] = _standardize(use["age"])
    use["log_gap_z"] = _standardize(use["log_gap_days"])
    terms = ["age_z"]
    if use["sex"].nunique(dropna=True) > 1:
        terms.append("C(sex)")
    if use["log_gap_days"].nunique(dropna=True) > 1:
        terms.append("log_gap_z")
    if use["t1_source"].nunique(dropna=True) > 1:
        terms.append("C(t1_source)")
    terms.append("C(site)")
    return use, terms


def _fit_covariance(base_fit, use: pd.DataFrame, covariance: str):
    if covariance == "cluster_site":
        fit = base_fit.get_robustcov_results(
            cov_type="cluster",
            groups=use["site"].to_numpy(),
            use_correction=True,
        )
    elif covariance == "HC3":
        fit = base_fit.get_robustcov_results(cov_type="HC3")
    else:
        raise ValueError(f"Unknown covariance: {covariance}")
    return fit


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


def _model_records(
    domain: str,
    frame: pd.DataFrame,
    scope: str,
    left: str,
    right: str,
    sites: set[str],
) -> list[dict[str, object]]:
    contrast = _contrast_name(left, right)
    if scope == "pair_common_support":
        fitted_groups: Sequence[str] = (right, left)
        reference = right
    else:
        fitted_groups = GROUPS
        reference = "CN"
    use, terms = _prepare_model_frame(frame, sites, fitted_groups)
    if use.empty:
        raise RuntimeError(f"Non-estimable site sensitivity: {domain}, {scope}, {contrast}")
    counts = use["group"].value_counts().reindex(GROUPS, fill_value=0)
    records: list[dict[str, object]] = []
    for response in RESPONSES:
        formula = (
            f'{response} ~ C(group, Treatment(reference="{reference}")) + '
            + " + ".join(terms)
        )
        base_fit = smf.ols(formula, data=use).fit()
        names = list(base_fit.model.exog_names)
        vector = _test_vector(names, left, right, reference)
        rank = int(np.linalg.matrix_rank(base_fit.model.exog))
        columns = int(base_fit.model.exog.shape[1])
        for covariance in COVARIANCES[scope]:
            fit = _fit_covariance(base_fit, use, covariance)
            test = fit.t_test(vector)
            ci = np.asarray(test.conf_int(alpha=0.05), dtype=float).reshape(-1)
            inference_df = getattr(fit, "df_resid_inference", math.nan)
            records.append(
                {
                    "domain": domain,
                    "response": response,
                    "scope": scope,
                    "contrast": contrast,
                    "covariance": covariance,
                    "claim_gate_covariance": bool(
                        covariance == CLAIM_GATE_COVARIANCE[scope]
                    ),
                    "n": int(len(use)),
                    "n_CN": int(counts["CN"]),
                    "n_MCI": int(counts["MCI"]),
                    "n_AD": int(counts["AD"]),
                    "n_sites": int(use["site"].nunique()),
                    "beta": float(np.asarray(test.effect).reshape(-1)[0]),
                    "ci95_low": float(ci[0]),
                    "ci95_high": float(ci[1]),
                    "p_value": float(np.asarray(test.pvalue).reshape(-1)[0]),
                    "formula": formula,
                    "design_rank": rank,
                    "design_columns": columns,
                    "design_condition_number": float(
                        np.linalg.cond(base_fit.model.exog)
                    ),
                    "inference_df": (
                        float(inference_df)
                        if np.isfinite(inference_df)
                        else math.nan
                    ),
                }
            )
    return records


def fit_site_sensitivities(analysis: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for domain, frame in analysis.groupby("domain", sort=False):
        all_sites = set(frame["site"].dropna().astype(str))
        all3_sites = _support_sites(frame, GROUPS)
        for left, right in CONTRASTS:
            records.extend(
                _model_records(
                    domain, frame, "full_site_fe", left, right, all_sites
                )
            )
            pair_sites = _support_sites(frame, (left, right))
            records.extend(
                _model_records(
                    domain,
                    frame,
                    "pair_common_support",
                    left,
                    right,
                    pair_sites,
                )
            )
            records.extend(
                _model_records(
                    domain,
                    frame,
                    "all3_site_support",
                    left,
                    right,
                    all3_sites,
                )
            )
    results = pd.DataFrame(records)
    difference = results["response"].eq("difference_score")
    results["holm_p_difference_across_domains"] = np.nan
    results.loc[difference, "holm_p_difference_across_domains"] = (
        results.loc[difference]
        .groupby(["scope", "covariance", "contrast"], sort=False)["p_value"]
        .transform(lambda values: adjust_p(values, "holm"))
    )
    claim_gate = difference & results["claim_gate_covariance"]
    results["holm_p_difference_claim_gate_global"] = np.nan
    results["bh_q_difference_claim_gate_global"] = np.nan
    results.loc[claim_gate, "holm_p_difference_claim_gate_global"] = adjust_p(
        results.loc[claim_gate, "p_value"], "holm"
    )
    results.loc[claim_gate, "bh_q_difference_claim_gate_global"] = adjust_p(
        results.loc[claim_gate, "p_value"], "fdr_bh"
    )
    return results


def algebra_validation(results: pd.DataFrame) -> pd.DataFrame:
    wide = results.pivot_table(
        index=["domain", "scope", "contrast", "covariance"],
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
    wide["identity_within_1e_10"] = wide["difference_beta_error"].abs().le(
        1e-10
    )
    return wide


def protocol_alias_diagnostics(analysis: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain, frame in analysis.groupby("domain", sort=False):
        sites = set(frame["site"].dropna().astype(str))
        use, terms = _prepare_model_frame(frame, sites, GROUPS)
        site_formula = (
            'difference_score ~ C(group, Treatment(reference="CN")) + '
            + " + ".join(terms)
        )
        site_protocol_formula = site_formula + " + C(protocol_bucket)"
        for model, formula in (
            ("site_fe", site_formula),
            ("site_fe_plus_protocol", site_protocol_formula),
        ):
            fit = smf.ols(formula, data=use).fit()
            rank = int(np.linalg.matrix_rank(fit.model.exog))
            columns = int(fit.model.exog.shape[1])
            rows.append(
                {
                    "domain": domain,
                    "model": model,
                    "n": int(len(use)),
                    "n_sites": int(use["site"].nunique()),
                    "design_rank": rank,
                    "design_columns": columns,
                    "design_full_rank": bool(rank == columns),
                    "design_condition_number": float(np.linalg.cond(fit.model.exog)),
                    "formula": formula,
                }
            )
    return pd.DataFrame(rows)


def base_comparison(
    results: pd.DataFrame, upstream_contrasts_path: Path
) -> pd.DataFrame:
    base = pd.read_csv(upstream_contrasts_path)
    base = base[
        base["response"].eq("difference_score")
        & base["stratum"].eq("full_exact")
        & base["contrast"].eq("AD_vs_CN")
    ][
        [
            "domain",
            "beta",
            "ci95_low",
            "ci95_high",
            "p_value",
            "holm_p_difference_across_domains",
        ]
    ].copy()
    base["analysis"] = "upstream_full_no_site_fe"
    views = [base]
    for scope in SCOPES:
        covariance = CLAIM_GATE_COVARIANCE[scope]
        view = results[
            results["response"].eq("difference_score")
            & results["contrast"].eq("AD_vs_CN")
            & results["scope"].eq(scope)
            & results["covariance"].eq(covariance)
        ][
            [
                "domain",
                "beta",
                "ci95_low",
                "ci95_high",
                "p_value",
                "holm_p_difference_across_domains",
            ]
        ].copy()
        view["analysis"] = f"{scope}_{covariance}"
        views.append(view)
    return pd.concat(views, ignore_index=True)[
        [
            "domain",
            "analysis",
            "beta",
            "ci95_low",
            "ci95_high",
            "p_value",
            "holm_p_difference_across_domains",
        ]
    ]


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], limit: int = 50) -> str:
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


def make_figure(comparison: pd.DataFrame, out: Path) -> None:
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=False)
    domains = list(DOMAINS)
    analyses = [
        "upstream_full_no_site_fe",
        "full_site_fe_cluster_site",
        "pair_common_support_cluster_site",
        "all3_site_support_cluster_site",
    ]
    labels = [
        "Upstream full (no site FE)",
        "Full + site FE",
        "AD-CN common sites + site FE",
        "All-three sites + site FE",
    ]
    colors = ["#9A9A9A", "#2678B2", "#F07E26", "#5B9A45"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.5), sharey=True)
    y = np.arange(len(analyses))
    for ax, domain in zip(axes, domains):
        frame = comparison[comparison["domain"].eq(domain)].set_index("analysis")
        frame = frame.reindex(analyses)
        beta = frame["beta"].to_numpy(dtype=float)
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
        ax.set_title(domain.replace("_", " "))
        ax.set_xlabel("AD-CN oriented microstructure − selected-efficiency β")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Site-confounding sensitivity of the exploratory cross-scale contrast")
    fig.tight_layout()
    fig.savefig(figure_dir / "ad_cn_site_sensitivity_forest.png", dpi=240)
    plt.close(fig)


def write_summary(
    out: Path,
    results: pd.DataFrame,
    support_counts: pd.DataFrame,
    comparison: pd.DataFrame,
    alias: pd.DataFrame,
) -> None:
    adcn = results[
        results["response"].eq("difference_score")
        & results["contrast"].eq("AD_vs_CN")
        & results["claim_gate_covariance"]
    ].copy()
    adcn["scope"] = pd.Categorical(adcn["scope"], SCOPES, ordered=True)
    adcn = adcn.sort_values(["domain", "scope"])
    mci_cn = results[
        results["response"].eq("difference_score")
        & results["contrast"].eq("MCI_vs_CN")
        & results["claim_gate_covariance"]
    ].copy()
    text = f"""# Cross-scale site-confounding sensitivity addendum

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**Status:** post hoc exploratory sensitivity of frozen historical participant-level scores. This is not a corrected imaging result, independent replication, biomarker analysis, or novelty proof.

## Why this addendum was required

The upstream release clustered uncertainty by site but did not include site fixed effects. Clustering changes standard errors; it does not remove between-site diagnosis/acquisition differences from the coefficient. Leave-one-site-out analysis is an influence diagnostic and likewise does not control site confounding. This addendum therefore estimates within-site diagnosis contrasts.

The collapsed protocol term is replaced by `C(site)`. Including both produced a rank-deficient design in every domain, as documented below. Site fixed effects absorb site-constant protocol and other site-level differences, but they do not identify a separate protocol effect and cannot repair unmeasured within-site acquisition variation.

## Estimands

1. `full_site_fe`: all paired subjects in a domain, site fixed effects, site-clustered covariance.
2. `pair_common_support`: only sites containing both groups in the named contrast and only those two groups. Both site-clustered and HC3 covariance are reported.
3. `all3_site_support`: only sites containing CN, MCI, and AD, with all three groups fitted. Both covariance variants are reported.

Site-clustered covariance is the claim-gate analysis for every scope because site fixed effects remove site means but do not guarantee independent residuals within a site. HC3 is retained as a diagnostic and is not assumed to be uniformly more conservative.

All models adjust age, sex, log1p absolute DTI-T1 gap, and T1 source when variable. Every model contains site fixed effects. Scores and their orientation are unchanged from the upstream release.

## AD-CN claim-gate results

{_markdown_table(adcn, ['domain', 'scope', 'covariance', 'n', 'n_CN', 'n_MCI', 'n_AD', 'n_sites', 'beta', 'ci95_low', 'ci95_high', 'p_value', 'holm_p_difference_across_domains', 'holm_p_difference_claim_gate_global', 'bh_q_difference_claim_gate_global'])}

Only a result that survives the exact within-scope three-domain Holm correction should be called statistically supported within that family. The global claim-gate family contains 27 difference tests: three domains, three diagnosis contrasts, and three scopes. It is diagnostic because the scopes are nested and the entire analysis is post hoc.

## MCI-CN claim-gate results

{_markdown_table(mci_cn, ['domain', 'scope', 'covariance', 'n', 'n_CN', 'n_MCI', 'n_AD', 'n_sites', 'beta', 'ci95_low', 'ci95_high', 'p_value', 'holm_p_difference_across_domains'])}

## Common-support counts

{_markdown_table(support_counts, ['domain', 'scope', 'contrast', 'fitted_groups', 'n', 'n_CN', 'n_MCI', 'n_AD', 'n_sites'], 40)}

## Upstream-to-site-adjusted AD-CN comparison

{_markdown_table(comparison, ['domain', 'analysis', 'beta', 'ci95_low', 'ci95_high', 'p_value', 'holm_p_difference_across_domains'])}

## Site/protocol alias diagnostic

{_markdown_table(alias, ['domain', 'model', 'n', 'n_sites', 'design_rank', 'design_columns', 'design_full_rank', 'design_condition_number'])}

## Interpretation boundary

This release can support only a conditional statement about selected standardized endpoints in historical outputs. It cannot support universal microstructure superiority, acquisition causation, biological hierarchy, three-class discrimination, or corrected-pipeline confirmation. A positive fixed-effect estimate is a within-observed-site association under the model; it is not transportable to unobserved sites without further validation.
"""
    (out / "site_sensitivity_summary.md").write_text(text, encoding="utf-8")


def write_run_manifest(
    out: Path,
    input_paths: dict[str, Path],
    results: pd.DataFrame,
) -> None:
    script = Path(__file__).resolve()
    upstream_script = script.with_name("cross_scale_robustness_analysis.py")
    manifest = {
        "schema_version": "1.0",
        "release": "cross_scale_site_sensitivity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "POST_HOC_EXPLORATORY_SITE_CONFOUNDING_ADDENDUM",
        "image_or_voxel_data_read": False,
        "image_processing_run": False,
        "production_outputs_modified": False,
        "novelty_claim_made": False,
        "configuration": _configuration(),
        "configuration_sha256": configuration_sha256(),
        "code": {
            "analysis_script": {"path": str(script), "sha256": sha256(script)},
            "upstream_analysis_script": {
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
        "result_rows": int(len(results)),
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_release(out: Path) -> dict[str, object]:
    required = [
        "site_fixed_effect_contrasts.csv",
        "site_support_counts.csv",
        "site_support_membership.csv",
        "site_contrast_algebra_validation.csv",
        "site_protocol_alias_diagnostics.csv",
        "upstream_vs_site_ad_cn.csv",
        "site_sensitivity_summary.md",
        "run_manifest.json",
        "figures/ad_cn_site_sensitivity_forest.png",
    ]
    absent = [relative for relative in required if not (out / relative).is_file()]
    results = pd.read_csv(out / "site_fixed_effect_contrasts.csv")
    algebra = pd.read_csv(out / "site_contrast_algebra_validation.csv")
    support = pd.read_csv(out / "site_support_counts.csv")
    membership = pd.read_csv(out / "site_support_membership.csv")
    alias = pd.read_csv(out / "site_protocol_alias_diagnostics.csv")
    difference = results[results["response"].eq("difference_score")]
    claim_gate = difference[difference["claim_gate_covariance"]]

    pair_members = membership[membership["scope"].eq("pair_common_support")]
    pair_ok = True
    for row in pair_members.itertuples(index=False):
        left, right = row.contrast.split("_vs_")
        counts = {
            "CN": row.n_CN_available,
            "MCI": row.n_MCI_available,
            "AD": row.n_AD_available,
        }
        pair_ok = pair_ok and counts[left] > 0 and counts[right] > 0
    all3 = membership[membership["scope"].eq("all3_site_support")]
    all3_ok = bool(
        (
            (all3["n_CN_available"] > 0)
            & (all3["n_MCI_available"] > 0)
            & (all3["n_AD_available"] > 0)
        ).all()
    )
    manifest = json.loads((out / "run_manifest.json").read_text())
    manifest_inputs_match = all(
        sha256(Path(value["path"])) == value["sha256"]
        for value in manifest["inputs"].values()
    )
    checks: dict[str, object] = {
        "required_files_present": not absent,
        "missing_required_files": absent,
        "three_domains_present": set(results["domain"]) == set(DOMAINS),
        "three_responses_present": set(results["response"]) == set(RESPONSES),
        "three_scopes_present": set(results["scope"]) == set(SCOPES),
        "three_contrasts_present": set(results["contrast"])
        == {"MCI_vs_CN", "AD_vs_CN", "AD_vs_MCI"},
        "expected_result_rows_135": len(results) == 135,
        "expected_difference_rows_45": len(difference) == 45,
        "expected_claim_gate_difference_rows_27": len(claim_gate) == 27,
        "expected_support_rows_27": len(support) == 27,
        "all_models_full_rank": bool(
            results["design_rank"].eq(results["design_columns"]).all()
        ),
        "all_models_include_site_fixed_effects": bool(
            results["formula"].str.contains("C(site)", regex=False).all()
        ),
        "no_model_includes_protocol": bool(
            ~results["formula"].str.contains("protocol", case=False).any()
        ),
        "all_p_values_bounded": bool(results["p_value"].between(0, 1).all()),
        "all_difference_adjusted_p_bounded": bool(
            difference["holm_p_difference_across_domains"].between(0, 1).all()
            and claim_gate["holm_p_difference_claim_gate_global"]
            .between(0, 1)
            .all()
            and claim_gate["bh_q_difference_claim_gate_global"]
            .between(0, 1)
            .all()
        ),
        "contrast_algebra_within_1e_10": bool(
            algebra["identity_within_1e_10"].all()
        ),
        "contrast_max_algebra_error": float(
            algebra["difference_beta_error"].abs().max()
        ),
        "pair_support_contains_both_groups": bool(pair_ok),
        "all3_support_contains_all_groups": all3_ok,
        "site_fe_design_full_rank": bool(
            alias.loc[alias["model"].eq("site_fe"), "design_full_rank"].all()
        ),
        "site_plus_protocol_rank_deficient_all_domains": bool(
            (~alias.loc[
                alias["model"].eq("site_fe_plus_protocol"), "design_full_rank"
            ]).all()
        ),
        "manifest_input_hashes_match": bool(manifest_inputs_match),
        "manifest_declares_no_image_processing": bool(
            not manifest["image_processing_run"]
        ),
    }
    excluded = {"missing_required_files", "contrast_max_algebra_error"}
    checks["pass"] = bool(
        all(value for key, value in checks.items() if key not in excluded)
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
    input_paths: dict[str, Path] = DEFAULT_INPUTS,
) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing release: {out}")
    for key, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing input {key}: {path}")
    upstream_validation = json.loads(
        input_paths["upstream_validation"].read_text(encoding="utf-8")
    )
    if not upstream_validation.get("pass", False):
        raise RuntimeError("Upstream cross-scale release is not validated")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        analysis = load_analysis_frame(input_paths)
        membership, counts = support_tables(analysis)
        results = fit_site_sensitivities(analysis)
        algebra = algebra_validation(results)
        alias = protocol_alias_diagnostics(analysis)
        comparison = base_comparison(results, input_paths["upstream_contrasts"])

        results.to_csv(temporary / "site_fixed_effect_contrasts.csv", index=False)
        counts.to_csv(temporary / "site_support_counts.csv", index=False)
        membership.to_csv(temporary / "site_support_membership.csv", index=False)
        algebra.to_csv(
            temporary / "site_contrast_algebra_validation.csv", index=False
        )
        alias.to_csv(
            temporary / "site_protocol_alias_diagnostics.csv", index=False
        )
        comparison.to_csv(temporary / "upstream_vs_site_ad_cn.csv", index=False)
        make_figure(comparison, temporary)
        write_summary(temporary, results, counts, comparison, alias)
        write_run_manifest(temporary, input_paths, results)
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
    print(f"PASS: site-sensitivity addendum written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
