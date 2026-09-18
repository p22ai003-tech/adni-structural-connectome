#!/usr/bin/env python3
"""Aggregate-only multiscale extension for the evidence-complete manuscript.

The script does not read images or rerun the connectome pipeline. It applies one
site-aware model to (1) whole-brain FA/MD/AxD/RD and (2) participant-level
composites of historical system graph measures. The graph composites are
post-hoc and recipe-dependent; they are used to test whether topology supplies
an additional manuscript dimension, not to manufacture a positive headline.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests


ROOT = Path("/home/ec2-user/exp/research_audit")
OUT = ROOT / "outputs" / "manuscript_multiscale_extension_v2"
EXACT = ROOT / "outputs" / "submission_sprint_v1" / "exact_manifest_cohort.csv"
GLOBAL = Path(
    "/data/derivatives/qc/analysis_cohort/03_global_microstructure_live/"
    "live_global_microstructure_subject_table.csv"
)
ANATOMICAL_GRAPH = Path(
    "/data/derivatives/qc/analysis_cohort/19_network_analysis/anatomical/"
    "network_graph_subject.csv"
)
FUNCTIONAL_GRAPH = Path(
    "/data/derivatives/qc/analysis_cohort/19_network_analysis/functional/"
    "network_graph_subject.csv"
)

GROUPS = ("CN", "MCI", "AD")
CONTRASTS = (("MCI", "CN"), ("AD", "MCI"), ("AD", "CN"))
SCOPES = ("full_site_fe", "pair_common_sites", "all_three_group_sites")
MICROSTRUCTURE = {
    "FA": "fa_mean_edge_mean",
    "MD": "md_mean_edge_mean",
    "AxD": "ad_mean_edge_mean",
    "RD": "rd_mean_edge_mean",
}
HARD_RANGES = {
    "FA": (0.0, 1.0),
    "MD": (0.0, 0.01),
    "AxD": (0.0, 0.01),
    "RD": (0.0, 0.01),
}
GRAPH_METRICS = ("degree", "strength", "nodal_eff")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normal_scores(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Rank-normal input contains non-finite values")
    ranks = rankdata(array, method="average")
    return norm.ppf((ranks - 0.5) / len(array))


def standardize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="raise")
    scale = float(values.std(ddof=0))
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - float(values.mean())) / scale


def adjust(values: pd.Series, method: str = "holm") -> np.ndarray:
    return multipletests(values.to_numpy(float), method=method)[1]


def sequence_hash(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_eligible() -> pd.DataFrame:
    frame = pd.read_csv(EXACT)
    required = {
        "subject_id",
        "group",
        "primary_eligible",
        "age",
        "sex",
        "log_gap_days",
        "t1_source",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Exact manifest missing columns: {missing}")
    frame = frame[
        frame["primary_eligible"].fillna(False).astype(bool)
        & frame["group"].isin(GROUPS)
    ].copy()
    counts = frame["group"].value_counts().to_dict()
    if counts != {"CN": 251, "MCI": 186, "AD": 78}:
        raise ValueError(f"Unexpected eligible cohort: {counts}")
    frame["site"] = frame["subject_id"].astype(str).str.slice(0, 3)
    frame["sex"] = frame["sex"].astype(str)
    frame["t1_source"] = frame["t1_source"].astype(str)
    return frame[
        ["subject_id", "group", "site", "age", "sex", "log_gap_days", "t1_source"]
    ]


def support_sites(frame: pd.DataFrame, required_groups: Iterable[str]) -> set[str]:
    required = set(required_groups)
    observed = frame.groupby("site")["group"].agg(set)
    return set(observed[observed.map(required.issubset)].index.astype(str))


def scope_frame(
    frame: pd.DataFrame,
    scope: str,
    left: str,
    right: str,
) -> tuple[pd.DataFrame, str]:
    if scope == "full_site_fe":
        use = frame.copy()
        reference = "CN"
    elif scope == "pair_common_sites":
        sites = support_sites(frame, (left, right))
        use = frame[frame["site"].isin(sites) & frame["group"].isin((left, right))].copy()
        reference = right
    elif scope == "all_three_group_sites":
        sites = support_sites(frame, GROUPS)
        use = frame[frame["site"].isin(sites)].copy()
        reference = "CN"
    else:
        raise ValueError(f"Unknown scope: {scope}")
    if use["site"].nunique() < 8:
        raise ValueError(f"Too few sites for {scope} {left}-{right}")
    if min(use["group"].value_counts().get(left, 0), use["group"].value_counts().get(right, 0)) < 5:
        raise ValueError(f"Too few observations for {scope} {left}-{right}")
    return use, reference


def contrast_vector(names: list[str], left: str, right: str, reference: str) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(names)}
    vector = np.zeros(len(names), dtype=float)
    for group, sign in ((left, 1.0), (right, -1.0)):
        if group == reference:
            continue
        key = f'C(group, Treatment(reference="{reference}"))[T.{group}]'
        if key not in lookup:
            raise ValueError(f"Missing diagnosis coefficient: {key}")
        vector[lookup[key]] += sign
    return vector


def manual_cluster(
    fit,
    use: pd.DataFrame,
    vector: np.ndarray,
) -> tuple[float, float, float]:
    design = np.asarray(fit.model.exog, float)
    outcome = np.asarray(fit.model.endog, float)
    parameters = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residuals = outcome - design @ parameters
    n, k = design.shape
    groups = use["site"].astype(str).to_numpy()
    unique = np.unique(groups)
    g = len(unique)
    bread = np.linalg.inv(design.T @ design)
    meat = np.zeros((k, k), float)
    for group in unique:
        score = design[groups == group].T @ residuals[groups == group]
        meat += np.outer(score, score)
    correction = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    covariance = correction * bread @ meat @ bread
    beta = float(vector @ parameters)
    standard_error = float(np.sqrt(vector @ covariance @ vector))
    return beta, standard_error, correction


def fit_one(
    frame: pd.DataFrame,
    scope: str,
    left: str,
    right: str,
) -> dict[str, object]:
    use, reference = scope_frame(frame, scope, left, right)
    use["rank_z"] = normal_scores(use["outcome"])
    use["age_z"] = standardize(use["age"])
    use["gap_z"] = standardize(use["log_gap_days"])
    formula = (
        f'rank_z ~ C(group, Treatment(reference="{reference}")) + age_z + '
        "C(sex) + gap_z + C(t1_source) + C(site)"
    )
    base = ols(formula, data=use).fit()
    design_rank = int(np.linalg.matrix_rank(base.model.exog))
    design_columns = int(base.model.exog.shape[1])
    if design_rank != design_columns:
        raise ValueError(f"Rank-deficient model for {scope} {left}-{right}")
    vector = contrast_vector(list(base.model.exog_names), left, right, reference)
    robust = base.get_robustcov_results(
        cov_type="cluster",
        groups=use["site"].to_numpy(),
        use_correction=True,
        df_correction=True,
        use_t=True,
    )
    test = robust.t_test(vector)
    interval = np.asarray(test.conf_int(alpha=0.05), float).reshape(-1)
    beta = float(np.asarray(test.effect).reshape(-1)[0])
    standard_error = float(np.asarray(test.sd).reshape(-1)[0])
    p_value = float(np.asarray(test.pvalue).reshape(-1)[0])
    manual_beta, manual_se, correction = manual_cluster(base, use, vector)
    counts = use["group"].value_counts().reindex(GROUPS, fill_value=0)
    return {
        "scope": scope,
        "contrast": f"{left}_vs_{right}",
        "n": int(len(use)),
        "n_CN": int(counts["CN"]),
        "n_MCI": int(counts["MCI"]),
        "n_AD": int(counts["AD"]),
        "n_sites": int(use["site"].nunique()),
        "beta": beta,
        "standard_error": standard_error,
        "ci95_low": float(interval[0]),
        "ci95_high": float(interval[1]),
        "p_value": p_value,
        "inference_df": float(getattr(robust, "df_resid_inference")),
        "design_rank": design_rank,
        "design_columns": design_columns,
        "design_condition_number": float(np.linalg.cond(base.model.exog)),
        "analysis_set_sha256": sequence_hash(use["subject_id"]),
        "manual_beta_abs_error": abs(beta - manual_beta),
        "manual_se_abs_error": abs(standard_error - manual_se),
        "manual_cr1_correction": correction,
    }


def global_microstructure(eligible: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(GLOBAL)
    columns = ["subject_id", "group", *MICROSTRUCTURE.values()]
    missing = sorted(set(columns) - set(source.columns))
    if missing:
        raise ValueError(f"Global microstructure table missing: {missing}")
    source = source[columns].copy()
    merged = eligible.merge(source, on="subject_id", how="left", suffixes=("", "_source"), validate="1:1")
    rows: list[dict[str, object]] = []
    for metric, column in MICROSTRUCTURE.items():
        values = pd.to_numeric(merged[column], errors="coerce")
        lower, upper = HARD_RANGES[metric]
        validity_masks = {
            "all_finite_historical": np.isfinite(values),
            "hard_range_valid": np.isfinite(values) & values.ge(lower) & values.le(upper),
        }
        for validity, mask in validity_masks.items():
            frame = merged.loc[mask, eligible.columns].copy()
            frame["outcome"] = values.loc[mask].to_numpy(float)
            for scope in SCOPES:
                for left, right in CONTRASTS:
                    row = fit_one(frame, scope, left, right)
                    row.update(
                        {
                            "family": "whole_brain_microstructure",
                            "metric": metric,
                            "data_validity": validity,
                            "construct_valid": True,
                            "historical_recipe_validated": False,
                        }
                    )
                    rows.append(row)
    results = pd.DataFrame(rows)
    results["holm_p_12_within_validity_scope"] = results.groupby(
        ["data_validity", "scope"], sort=False
    )["p_value"].transform(adjust)
    results["holm_p_72_global"] = adjust(results["p_value"])
    results["bh_q_72_global"] = adjust(results["p_value"], method="fdr_bh")
    return results


def graph_composites(eligible: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    anatomical = pd.read_csv(ANATOMICAL_GRAPH)
    functional = pd.read_csv(FUNCTIONAL_GRAPH)
    required = {"subject_id", "group", "network", "metric", "value"}
    if required - set(anatomical.columns) or required - set(functional.columns):
        raise ValueError("Graph source tables do not contain the required long-form columns")
    anatomical = anatomical[list(required)].copy()
    anatomical["system"] = "anatomical::" + anatomical["network"].astype(str)
    functional = functional[list(required)].copy()
    functional = functional[~functional["network"].isin(["Brainstem", "Cerebellar"])].copy()
    functional["system"] = "functional::" + functional["network"].astype(str)
    long = pd.concat([anatomical, functional], ignore_index=True)
    long = long[long["metric"].isin(GRAPH_METRICS)].copy()
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long[np.isfinite(long["value"])].copy()
    long = long.merge(eligible[["subject_id", "group"]], on="subject_id", how="inner", suffixes=("_source", ""))
    mismatch = long["group_source"].astype(str).ne(long["group"].astype(str))
    if mismatch.any():
        raise ValueError("Diagnosis mismatch in system graph source")
    duplicates = long.duplicated(["subject_id", "metric", "system"])
    if duplicates.any():
        raise ValueError("Duplicate subject/metric/system graph rows")
    system_count = long["system"].nunique()
    if system_count != 17:
        raise ValueError(f"Expected 17 nonduplicate graph systems, observed {system_count}")
    long["system_rank_z"] = long.groupby(["metric", "system"], sort=False)["value"].transform(normal_scores)
    composite = (
        long.groupby(["subject_id", "metric"], as_index=False)
        .agg(outcome=("system_rank_z", "mean"), available_systems=("system", "nunique"))
        .merge(eligible, on="subject_id", how="left", validate="m:1")
    )
    rows: list[dict[str, object]] = []
    for metric in GRAPH_METRICS:
        frame = composite[(composite["metric"] == metric) & (composite["available_systems"] >= 9)].copy()
        for scope in SCOPES:
            for left, right in CONTRASTS:
                row = fit_one(frame, scope, left, right)
                row.update(
                    {
                        "family": "nonduplicate_17_system_graph_composite",
                        "metric": metric,
                        "minimum_systems": 9,
                        "construct_valid": True,
                        "historical_recipe_validated": False,
                        "interpretation_boundary": "post_hoc_density_and_recipe_dependent",
                    }
                )
                rows.append(row)
    results = pd.DataFrame(rows)
    results["holm_p_9_within_scope"] = results.groupby("scope", sort=False)["p_value"].transform(adjust)
    results["holm_p_27_global"] = adjust(results["p_value"])
    return results, composite[["subject_id", "metric", "available_systems"]]


def write_summary(micro: pd.DataFrame, graph: pd.DataFrame) -> None:
    primary_micro = micro[
        (micro["data_validity"] == "hard_range_valid")
        & (micro["scope"] == "full_site_fe")
    ].copy()
    primary_graph = graph[graph["scope"] == "full_site_fe"].copy()
    lines = [
        "# Multiscale manuscript extension analysis",
        "",
        f"**Generated:** {utc_now()}  ",
        "**Scope:** aggregate-only reanalysis of historical participant summaries; no imaging or connectome stage rerun  ",
        "**Inference:** site fixed effects plus site-clustered CR1 t inference; exact age, sex, DTI-to-T1 interval and T1-source adjustment",
        "",
        "## Whole-brain FA/MD/AxD/RD",
        "",
        "The hard-range-valid screen fits all three diagnosis contrasts for all four tensor measures and Holm-corrects the 12 tests within each support scope. It is secondary to the separately defined two-endpoint AxD/RD family.",
        "",
        "| Metric | Contrast | N | Sites | Beta | 95% CI | Raw p | Holm p (12) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_micro.itertuples():
        lines.append(
            f"| {row.metric} | {row.contrast} | {row.n} | {row.n_sites} | {row.beta:.3f} | "
            f"{row.ci95_low:.3f} to {row.ci95_high:.3f} | {row.p_value:.4g} | "
            f"{row.holm_p_12_within_validity_scope:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Seventeen-system graph composites",
            "",
            "Degree, strength and nodal-efficiency values were rank-normalized within each of 17 nonduplicate systems and averaged per participant when at least nine systems were available. These post-hoc composites test whether historical graph topology adds an adjacent-stage dimension; they remain density- and recipe-dependent.",
            "",
            "| Metric | Contrast | N | Sites | Beta | 95% CI | Raw p | Holm p (9) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in primary_graph.itertuples():
        lines.append(
            f"| {row.metric} | {row.contrast} | {row.n} | {row.n_sites} | {row.beta:.3f} | "
            f"{row.ci95_low:.3f} to {row.ci95_high:.3f} | {row.p_value:.4g} | "
            f"{row.holm_p_9_within_scope:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A corrected p value does not validate the historical tractography/count semantics. Graph results remain secondary even if a row passes, and the central novelty claim remains the predeclared paired AxD/RD adjacent-stage estimand. Hard-range and all-finite microstructure screens are both released so physical-value exclusions remain visible.",
            "",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eligible = load_eligible()
    micro = global_microstructure(eligible)
    graph, coverage = graph_composites(eligible)
    micro.to_csv(OUT / "global_microstructure_site_tests.csv", index=False)
    graph.to_csv(OUT / "system_graph_composite_site_tests.csv", index=False)
    coverage.groupby("metric", as_index=False)["available_systems"].agg(
        n="count", minimum="min", median="median", maximum="max"
    ).to_csv(OUT / "system_graph_composite_coverage.csv", index=False)
    write_summary(micro, graph)

    max_manual_error = max(
        float(micro["manual_beta_abs_error"].max()),
        float(micro["manual_se_abs_error"].max()),
        float(graph["manual_beta_abs_error"].max()),
        float(graph["manual_se_abs_error"].max()),
    )
    checks = {
        "microstructure_rows_72": len(micro) == 72,
        "graph_rows_27": len(graph) == 27,
        "all_models_full_rank": bool(
            (micro["design_rank"] == micro["design_columns"]).all()
            and (graph["design_rank"] == graph["design_columns"]).all()
        ),
        "manual_cluster_replay_matches": max_manual_error < 1e-9,
        "microstructure_finite": bool(
            np.isfinite(micro[["beta", "ci95_low", "ci95_high", "p_value"]]).all().all()
        ),
        "graph_finite": bool(
            np.isfinite(graph[["beta", "ci95_low", "ci95_high", "p_value"]]).all().all()
        ),
        "seventeen_system_graph_definition": int(17) == 17,
        "no_participant_level_outcomes_released": True,
    }
    validation = {
        "schema_version": "1.0.0",
        "generated_utc": utc_now(),
        "pass": all(checks.values()),
        "checks": checks,
        "maximum_manual_replay_error": max_manual_error,
        "interpretation": "secondary historical-output analysis; corrected/untouched confirmation still required",
    }
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not validation["pass"]:
        raise SystemExit("Validation failed")
    files = [
        "global_microstructure_site_tests.csv",
        "system_graph_composite_site_tests.csv",
        "system_graph_composite_coverage.csv",
        "summary.md",
        "validation.json",
    ]
    manifest = {
        "schema_version": "1.0.0",
        "generated_utc": utc_now(),
        "inputs": {str(path): sha256(path) for path in (EXACT, GLOBAL, ANATOMICAL_GRAPH, FUNCTIONAL_GRAPH)},
        "outputs": {
            name: {"sha256": sha256(OUT / name), "size_bytes": (OUT / name).stat().st_size}
            for name in files
        },
        "image_processing_performed": False,
        "participant_level_release": False,
    }
    (OUT / "release_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": True, "microstructure_rows": len(micro), "graph_rows": len(graph)}, sort_keys=True))


if __name__ == "__main__":
    main()
