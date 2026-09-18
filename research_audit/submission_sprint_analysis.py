#!/usr/bin/env python3
"""Deadline-safe reanalysis of existing structural-connectome outputs.

This module does not read image voxels, run MRtrix, or modify production
derivatives.  It joins already-produced participant-level metrics to the exact,
checksum-bound 530-pair cohort manifest and emits a versioned exploratory
robustness release.  Historical image-processing limitations remain applicable;
the output is discovery/directional evidence, not a corrected confirmatory run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import mannwhitneyu, norm, rankdata
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


PROJECT = Path("/home/ec2-user/exp")
ANALYSIS = Path("/data/derivatives/qc/analysis_cohort")
DEFAULT_OUT = PROJECT / "research_audit" / "outputs" / "submission_sprint_v1"
GROUPS = ("CN", "MCI", "AD")
CONTRASTS = (("MCI", "CN"), ("AD", "CN"), ("AD", "MCI"))

INPUTS = {
    "exact_manifest": PROJECT
    / "research_audit"
    / "outputs"
    / "available_data_pair_manifest_v2.csv",
    "global_micro": ANALYSIS
    / "03_global_microstructure_live"
    / "live_global_microstructure_subject_table.csv",
    "global_graph": ANALYSIS
    / "06_global_graph_live"
    / "live_global_graph_subject_table.csv",
    "functional_micro": ANALYSIS
    / "19_network_analysis"
    / "functional"
    / "network_microstructure_subject.csv",
    "functional_graph": ANALYSIS
    / "19_network_analysis"
    / "functional"
    / "network_graph_subject.csv",
    "functional_coupling": ANALYSIS
    / "19_network_analysis"
    / "functional"
    / "network_coupling_subject.csv",
    "anatomical_micro": ANALYSIS
    / "19_network_analysis"
    / "anatomical"
    / "network_microstructure_subject.csv",
    "anatomical_graph": ANALYSIS
    / "19_network_analysis"
    / "anatomical"
    / "network_graph_subject.csv",
    "anatomical_coupling": ANALYSIS
    / "19_network_analysis"
    / "anatomical"
    / "network_coupling_subject.csv",
}


# These endpoints were frozen for the deadline sprint before the exact-date
# reanalysis was run.  The broader 214-feature screen is emitted separately and
# is explicitly hypothesis-generating.
TARGET_FAMILIES = {
    "global::microstructure::WholeBrain::fa_mean_edge_mean": "primary_wholebrain_fa",
    "global::microstructure::WholeBrain::md_mean_edge_mean": "supporting_microstructure",
    "global::microstructure::WholeBrain::rd_mean_edge_mean": "supporting_microstructure",
    "global::microstructure::WholeBrain::ad_mean_edge_mean": "supporting_microstructure",
    "functional::microstructure::DMN::fa_mean": "supporting_microstructure",
    "functional::microstructure::Limbic::md_mean": "supporting_microstructure",
    "functional::microstructure::Limbic::rd_mean": "supporting_microstructure",
    "functional::microstructure::Limbic::ad_mean": "supporting_microstructure",
    "anatomical::microstructure::Limbic::fa_mean": "mapping_sensitivity",
    "anatomical::microstructure::Limbic::md_mean": "mapping_sensitivity",
    "anatomical::microstructure::Limbic::rd_mean": "mapping_sensitivity",
    "anatomical::microstructure::Limbic::ad_mean": "mapping_sensitivity",
    "global::graph::WholeBrain::global_efficiency": "topology_secondary",
    "functional::graph::DMN::nodal_eff": "topology_secondary",
    "functional::graph::Limbic::nodal_eff": "topology_secondary",
    "anatomical::topology_microstructure_coupling::Frontal::nodal_eff__fa_mean": "coupling_secondary",
}


@dataclass(frozen=True)
class Stratum:
    name: str
    subjects: frozenset[str]
    include_protocol: bool = True
    include_gap: bool = True
    include_t1_source: bool = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def adjust_p(values: Iterable[float], method: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    result = np.full(array.shape, np.nan)
    keep = np.isfinite(array)
    if keep.any():
        result[keep] = multipletests(array[keep], method=method)[1]
    return result


def protocol_bucket(manufacturer: object, gradients: object) -> str:
    maker = "" if pd.isna(manufacturer) else str(manufacturer).strip().upper()
    try:
        directions = int(round(float(gradients)))
    except (TypeError, ValueError):
        directions = -1
    if "SIEMENS" in maker and directions == 54:
        return "SIEMENS_54"
    if "SIEMENS" in maker and directions == 30:
        return "SIEMENS_30"
    if "SIEMENS" in maker:
        return "SIEMENS_OTHER"
    if maker.startswith("GE") and directions == 41:
        return "GE_41"
    if maker.startswith("GE"):
        return "GE_OTHER"
    if "PHILIPS" in maker:
        return "PHILIPS"
    return "OTHER_OR_MISSING"


def load_exact_cohort(path: Path = INPUTS["exact_manifest"]) -> pd.DataFrame:
    cohort = pd.read_csv(path)
    required = {
        "subject_id",
        "diagnosis_reconciled_harmonized",
        "diagnosis_reconciled_primary_analysis_eligible",
        "dti_age_at_scan",
        "dti_sex",
        "dti_t1_gap_days_abs",
        "include_timing_le_90_sensitivity",
        "include_timing_le_180_sensitivity",
        "same_adni_phase",
        "dti_manufacturer",
        "dti_gradient_directions",
        "dti_protocol_key",
        "current_t1_is_original",
        "legacy_sc_matrix_qc_include",
        "legacy_radial_recipe",
    }
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise ValueError(f"Exact manifest is missing required columns: {missing}")
    if len(cohort) != 530 or cohort["subject_id"].nunique() != 530:
        raise ValueError("Exact manifest must contain 530 unique subjects")

    cohort = cohort.copy()
    cohort["group"] = cohort["diagnosis_reconciled_harmonized"].astype(str)
    cohort["primary_eligible"] = (
        cohort["diagnosis_reconciled_primary_analysis_eligible"]
        .fillna(False)
        .astype(bool)
    )
    eligible = cohort[cohort["primary_eligible"] & cohort["group"].isin(GROUPS)]
    expected = {"CN": 251, "MCI": 186, "AD": 78}
    observed = eligible["group"].value_counts().to_dict()
    if any(int(observed.get(group, 0)) != count for group, count in expected.items()):
        raise ValueError(f"Unexpected exact diagnosis counts: {observed}")
    if int(cohort["group"].eq("SMC").sum()) != 15:
        raise ValueError("Expected 15 SMC subjects retained outside primary contrasts")

    cohort["site"] = cohort["subject_id"].str.slice(0, 3)
    cohort["age"] = pd.to_numeric(cohort["dti_age_at_scan"], errors="coerce")
    cohort["sex"] = cohort["dti_sex"].fillna("Missing").astype(str)
    cohort["gap_days"] = pd.to_numeric(cohort["dti_t1_gap_days_abs"], errors="coerce")
    cohort["log_gap_days"] = np.log1p(cohort["gap_days"])
    cohort["t1_source"] = np.where(
        cohort["current_t1_is_original"].fillna(False).astype(bool),
        "Original",
        "Processed",
    )
    cohort["protocol_bucket"] = [
        protocol_bucket(maker, gradients)
        for maker, gradients in zip(
            cohort["dti_manufacturer"], cohort["dti_gradient_directions"]
        )
    ]
    cohort["siemens_54"] = cohort["protocol_bucket"].eq("SIEMENS_54")
    cohort["legacy_qc_include"] = (
        cohort["legacy_sc_matrix_qc_include"].fillna(False).astype(bool)
    )
    return cohort


def build_strata(cohort: pd.DataFrame) -> list[Stratum]:
    eligible = cohort["primary_eligible"] & cohort["group"].isin(GROUPS)
    le90 = eligible & cohort["include_timing_le_90_sensitivity"].fillna(False).astype(bool)
    le180 = eligible & cohort["include_timing_le_180_sensitivity"].fillna(False).astype(bool)
    same_phase = eligible & cohort["same_adni_phase"].fillna(False).astype(bool)
    siemens = eligible & cohort["siemens_54"]
    core = le90 & cohort["siemens_54"]
    core_frame = cohort.loc[core]
    overlap_sites = set(
        core_frame.groupby("site")["group"].nunique().loc[lambda values: values >= 2].index
    )
    overlap = core & cohort["site"].isin(overlap_sites)
    original_t1 = eligible & cohort["t1_source"].eq("Original")
    legacy_qc = eligible & cohort["legacy_qc_include"]

    def subjects(mask: pd.Series) -> frozenset[str]:
        return frozenset(cohort.loc[mask, "subject_id"].astype(str))

    strata = [
        Stratum("full_exact", subjects(eligible)),
        Stratum("timing_le90", subjects(le90)),
        Stratum("timing_le180", subjects(le180)),
        Stratum("same_phase", subjects(same_phase)),
        Stratum("siemens54", subjects(siemens), include_protocol=False),
        Stratum(
            "timing_le90_siemens54",
            subjects(core),
            include_protocol=False,
        ),
        Stratum(
            "timing_le90_siemens54_overlap_sites",
            subjects(overlap),
            include_protocol=False,
        ),
        Stratum(
            "original_t1",
            subjects(original_t1),
            include_t1_source=False,
        ),
        Stratum("legacy_qc_include", subjects(legacy_qc)),
    ]
    return strata


def collect_metrics(inputs: dict[str, Path] = INPUTS) -> pd.DataFrame:
    records: list[pd.DataFrame] = []

    global_micro = pd.read_csv(inputs["global_micro"])
    for metric in (
        "fa_mean_edge_mean",
        "md_mean_edge_mean",
        "ad_mean_edge_mean",
        "rd_mean_edge_mean",
    ):
        frame = global_micro[["subject_id", metric]].rename(columns={metric: "value"})
        frame["mapping"] = "global"
        frame["feature_family"] = "microstructure"
        frame["network"] = "WholeBrain"
        frame["metric"] = metric
        records.append(frame)

    global_graph = pd.read_csv(inputs["global_graph"])
    for metric in (
        "mean_strength",
        "total_strength",
        "density",
        "global_efficiency",
        "charpath_len",
    ):
        frame = global_graph[["subject_id", metric]].rename(columns={metric: "value"})
        frame["mapping"] = "global"
        frame["feature_family"] = "graph"
        frame["network"] = "WholeBrain"
        frame["metric"] = metric
        records.append(frame)

    for mapping in ("functional", "anatomical"):
        for family, key in (
            ("microstructure", f"{mapping}_micro"),
            ("graph", f"{mapping}_graph"),
            ("topology_microstructure_coupling", f"{mapping}_coupling"),
        ):
            frame = pd.read_csv(inputs[key])[
                ["subject_id", "network", "metric", "value"]
            ].copy()
            frame["mapping"] = mapping
            frame["feature_family"] = family
            records.append(frame)

    metrics = pd.concat(records, ignore_index=True)
    metrics["subject_id"] = metrics["subject_id"].astype(str)
    metrics["value"] = pd.to_numeric(metrics["value"], errors="coerce")
    metrics["feature_id"] = (
        metrics["mapping"].astype(str)
        + "::"
        + metrics["feature_family"].astype(str)
        + "::"
        + metrics["network"].astype(str)
        + "::"
        + metrics["metric"].astype(str)
    )
    duplicates = metrics.duplicated(["subject_id", "feature_id"], keep=False)
    if duplicates.any():
        examples = metrics.loc[duplicates, ["subject_id", "feature_id"]].head(10)
        raise ValueError(f"Duplicate subject-feature rows detected:\n{examples}")
    missing_targets = sorted(set(TARGET_FAMILIES) - set(metrics["feature_id"]))
    if missing_targets:
        raise ValueError(f"Target endpoints are absent: {missing_targets}")
    return metrics


def normal_scores(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("normal_scores requires finite values")
    ranks = rankdata(numeric, method="average")
    return norm.ppf((ranks - 0.5) / len(ranks))


def cliffs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if not len(x) or not len(y):
        return math.nan
    statistic = mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2.0 * float(statistic) / (len(x) * len(y)) - 1.0


def _standardize(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    scale = float(numeric.std(ddof=0))
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - float(numeric.mean())) / scale


def fit_feature_contrasts(
    frame: pd.DataFrame,
    stratum: Stratum,
    minimum_per_group: int = 5,
) -> list[dict[str, object]]:
    """Fit rank-normal OLS and return three diagnosis contrasts.

    Site-clustered covariance is used when at least eight sites are available;
    otherwise HC3 is used.  The returned rank and condition number make weak or
    aliased designs visible rather than silently treating them as reliable.
    """

    use = frame[frame["subject_id"].isin(stratum.subjects)].copy()
    use = use[use["group"].isin(GROUPS) & np.isfinite(use["value"])]
    counts = use["group"].value_counts().reindex(GROUPS, fill_value=0)
    if int(counts.min()) < minimum_per_group:
        return []

    required = ["value", "group", "age", "sex", "site"]
    if stratum.include_gap:
        required.append("log_gap_days")
    if stratum.include_t1_source:
        required.append("t1_source")
    if stratum.include_protocol:
        required.append("protocol_bucket")
    use = use.dropna(subset=required).copy()
    counts = use["group"].value_counts().reindex(GROUPS, fill_value=0)
    if int(counts.min()) < minimum_per_group:
        return []

    use["rank_z"] = normal_scores(use["value"])
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

    formula = "rank_z ~ " + " + ".join(terms)
    base_fit = smf.ols(formula, data=use).fit()
    design_rank = int(np.linalg.matrix_rank(base_fit.model.exog))
    n_columns = int(base_fit.model.exog.shape[1])
    condition = float(np.linalg.cond(base_fit.model.exog))
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

    names = list(base_fit.model.exog_names)
    mci_name = 'C(group, Treatment(reference="CN"))[T.MCI]'
    ad_name = 'C(group, Treatment(reference="CN"))[T.AD]'
    if mci_name not in names or ad_name not in names:
        return []
    index = {name: position for position, name in enumerate(names)}

    first = use.iloc[0]
    output: list[dict[str, object]] = []
    for left, right in CONTRASTS:
        vector = np.zeros(len(names), dtype=float)
        if left == "MCI":
            vector[index[mci_name]] += 1.0
        elif left == "AD":
            vector[index[ad_name]] += 1.0
        if right == "MCI":
            vector[index[mci_name]] -= 1.0
        elif right == "AD":
            vector[index[ad_name]] -= 1.0
        test = fit.t_test(vector)
        ci = np.asarray(test.conf_int(alpha=0.05), dtype=float).reshape(-1)
        left_values = use.loc[use["group"].eq(left), "value"].to_numpy(dtype=float)
        right_values = use.loc[use["group"].eq(right), "value"].to_numpy(dtype=float)
        output.append(
            {
                "stratum": stratum.name,
                "feature_id": first["feature_id"],
                "mapping": first["mapping"],
                "feature_family": first["feature_family"],
                "network": first["network"],
                "metric": first["metric"],
                "contrast": f"{left}_vs_{right}",
                "n": int(len(use)),
                "n_CN": int(counts["CN"]),
                "n_MCI": int(counts["MCI"]),
                "n_AD": int(counts["AD"]),
                "n_sites": n_sites,
                "median_CN": float(use.loc[use["group"].eq("CN"), "value"].median()),
                "median_MCI": float(use.loc[use["group"].eq("MCI"), "value"].median()),
                "median_AD": float(use.loc[use["group"].eq("AD"), "value"].median()),
                "rank_normal_beta": float(np.asarray(test.effect).reshape(-1)[0]),
                "ci95_low": float(ci[0]),
                "ci95_high": float(ci[1]),
                "p_value": float(np.asarray(test.pvalue).reshape(-1)[0]),
                "cliffs_delta": cliffs_delta(left_values, right_values),
                "covariance": covariance,
                "formula": formula,
                "design_rank": design_rank,
                "design_columns": n_columns,
                "design_full_rank": bool(design_rank == n_columns),
                "design_condition_number": condition,
            }
        )
    return output


def stratum_counts(cohort: pd.DataFrame, strata: Sequence[Stratum]) -> pd.DataFrame:
    rows = []
    for stratum in strata:
        frame = cohort[cohort["subject_id"].isin(stratum.subjects)]
        counts = frame["group"].value_counts()
        rows.append(
            {
                "stratum": stratum.name,
                "n_total": len(frame),
                "n_CN": int(counts.get("CN", 0)),
                "n_MCI": int(counts.get("MCI", 0)),
                "n_AD": int(counts.get("AD", 0)),
                "n_sites": int(frame["site"].nunique()),
                "n_protocol_buckets": int(frame["protocol_bucket"].nunique()),
                "n_original_t1": int(frame["t1_source"].eq("Original").sum()),
            }
        )
    return pd.DataFrame(rows)


def run_models(
    cohort: pd.DataFrame,
    metrics: pd.DataFrame,
    strata: Sequence[Stratum],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = cohort[
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
    long = metrics.merge(meta, on="subject_id", how="left", validate="m:1")
    records: list[dict[str, object]] = []
    for stratum in strata:
        for _, feature in long.groupby("feature_id", sort=False):
            records.extend(fit_feature_contrasts(feature, stratum))
    broad = pd.DataFrame(records)
    if broad.empty:
        raise RuntimeError("No estimable feature contrasts were produced")
    broad["bh_q_within_stratum_contrast"] = broad.groupby(
        ["stratum", "contrast"], sort=False
    )["p_value"].transform(lambda values: adjust_p(values, "fdr_bh"))
    broad["bh_q_all_tests_within_stratum"] = broad.groupby("stratum", sort=False)[
        "p_value"
    ].transform(lambda values: adjust_p(values, "fdr_bh"))

    targeted = broad[broad["feature_id"].isin(TARGET_FAMILIES)].copy()
    targeted["target_family"] = targeted["feature_id"].map(TARGET_FAMILIES)
    targeted["holm_p_within_family_stratum"] = targeted.groupby(
        ["stratum", "target_family"], sort=False
    )["p_value"].transform(lambda values: adjust_p(values, "holm"))
    targeted["bh_q_all_targeted_within_stratum"] = targeted.groupby(
        "stratum", sort=False
    )["p_value"].transform(lambda values: adjust_p(values, "fdr_bh"))
    return targeted, broad


def missingness_table(
    cohort: pd.DataFrame,
    metrics: pd.DataFrame,
    strata: Sequence[Stratum],
) -> pd.DataFrame:
    available = metrics[np.isfinite(metrics["value"])][
        ["subject_id", "feature_id"]
    ].copy()
    rows = []
    for stratum in strata:
        base = cohort[cohort["subject_id"].isin(stratum.subjects)]
        for feature_id in sorted(metrics["feature_id"].unique()):
            have = set(available.loc[available["feature_id"].eq(feature_id), "subject_id"])
            record: dict[str, object] = {
                "stratum": stratum.name,
                "feature_id": feature_id,
            }
            for group in GROUPS:
                ids = set(base.loc[base["group"].eq(group), "subject_id"])
                n_total = len(ids)
                n_available = len(ids & have)
                record[f"n_{group}"] = n_total
                record[f"available_{group}"] = n_available
                record[f"missing_pct_{group}"] = (
                    100.0 * (n_total - n_available) / n_total if n_total else math.nan
                )
            rows.append(record)
    return pd.DataFrame(rows)


def robustness_table(results: pd.DataFrame) -> pd.DataFrame:
    core = [
        "full_exact",
        "timing_le90",
        "siemens54",
        "timing_le90_siemens54",
        "timing_le90_siemens54_overlap_sites",
        "original_t1",
    ]
    rows = []
    for (feature_id, contrast), frame in results.groupby(["feature_id", "contrast"]):
        selected = frame[frame["stratum"].isin(core)].copy()
        full = selected.loc[selected["stratum"].eq("full_exact"), "rank_normal_beta"]
        if full.empty or not np.isfinite(full.iloc[0]) or float(full.iloc[0]) == 0:
            reference_sign = math.nan
            agreement = math.nan
        else:
            reference_sign = float(np.sign(full.iloc[0]))
            signs = np.sign(selected["rank_normal_beta"].to_numpy(dtype=float))
            agreement = float(np.mean(signs == reference_sign)) if len(signs) else math.nan
        overlap = selected[
            selected["stratum"].eq("timing_le90_siemens54_overlap_sites")
        ]
        overlap_beta = (
            float(overlap["rank_normal_beta"].iloc[0]) if not overlap.empty else math.nan
        )
        overlap_p = float(overlap["p_value"].iloc[0]) if not overlap.empty else math.nan
        median_abs = float(selected["rank_normal_beta"].abs().median())
        min_abs = float(selected["rank_normal_beta"].abs().min())
        score = (
            agreement * median_abs * math.sqrt(max(int(selected["n"].min()), 1) / 100.0)
            if np.isfinite(agreement)
            else math.nan
        )
        rows.append(
            {
                "feature_id": feature_id,
                "contrast": contrast,
                "n_core_strata_estimable": int(len(selected)),
                "reference_full_sign": reference_sign,
                "sign_agreement_with_full": agreement,
                "median_abs_rank_normal_beta": median_abs,
                "minimum_abs_rank_normal_beta": min_abs,
                "overlap_beta": overlap_beta,
                "overlap_p_uncorrected": overlap_p,
                "minimum_n_across_core_strata": int(selected["n"].min()),
                "directional_priority_score_not_inference": score,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "sign_agreement_with_full",
            "directional_priority_score_not_inference",
            "median_abs_rank_normal_beta",
        ],
        ascending=False,
        na_position="last",
    )


def make_figures(targeted: pd.DataFrame, out: Path) -> None:
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=False)
    desired_strata = [
        "full_exact",
        "timing_le90",
        "siemens54",
        "timing_le90_siemens54",
        "timing_le90_siemens54_overlap_sites",
        "original_t1",
    ]
    view = targeted[targeted["contrast"].eq("AD_vs_CN")].copy()
    matrix = view.pivot_table(
        index="feature_id", columns="stratum", values="rank_normal_beta", aggfunc="first"
    ).reindex(columns=desired_strata)
    matrix = matrix.reindex([feature for feature in TARGET_FAMILIES if feature in matrix.index])
    values = matrix.to_numpy(dtype=float)
    bound = max(float(np.nanmax(np.abs(values))), 0.25)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    ax.set_xticks(range(len(matrix.columns)), [value.replace("_", "\n") for value in matrix.columns])
    ax.set_yticks(
        range(len(matrix.index)),
        [value.replace("::", " | ") for value in matrix.index],
        fontsize=7,
    )
    ax.set_title("AD–CN rank-normal effect across exact-metadata robustness strata")
    ax.set_xlabel("Existing-output sensitivity stratum")
    ax.set_ylabel("Frozen deadline-sprint endpoint")
    fig.colorbar(image, ax=ax, label="Adjusted rank-normal beta (AD minus CN)")
    fig.tight_layout()
    fig.savefig(figure_dir / "targeted_ad_cn_effect_stability.png", dpi=240)
    plt.close(fig)

    full = view[view["stratum"].eq("full_exact")][
        ["feature_id", "target_family", "rank_normal_beta"]
    ].rename(columns={"rank_normal_beta": "full_beta"})
    overlap = view[
        view["stratum"].eq("timing_le90_siemens54_overlap_sites")
    ][["feature_id", "rank_normal_beta"]].rename(columns={"rank_normal_beta": "overlap_beta"})
    plot = full.merge(overlap, on="feature_id", how="inner")
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for family, frame in plot.groupby("target_family"):
        ax.scatter(frame["full_beta"], frame["overlap_beta"], s=50, label=family)
    limit = max(
        0.25,
        float(np.nanmax(np.abs(plot[["full_beta", "overlap_beta"]].to_numpy()))) * 1.1,
    )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.plot([-limit, limit], [-limit, limit], linestyle="--", color="grey", linewidth=0.8)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("Full exact-manifest beta")
    ax.set_ylabel("<=90-day Siemens-54 overlap-site beta")
    ax.set_title("Effect-direction and magnitude stress test (AD–CN)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(figure_dir / "full_vs_overlap_ad_cn_effects.png", dpi=240)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], limit: int = 20) -> str:
    selected = frame.loc[:, list(columns)].head(limit).copy()
    for column in selected.select_dtypes(include=["float"]).columns:
        selected[column] = selected[column].map(
            lambda value: "NA" if not np.isfinite(value) else f"{value:.4g}"
        )
    def cell(value: object) -> str:
        if pd.isna(value):
            return "NA"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(map(cell, selected.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in selected.columns) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows])


def write_summary(
    out: Path,
    counts: pd.DataFrame,
    targeted: pd.DataFrame,
    targeted_robustness: pd.DataFrame,
    broad_robustness: pd.DataFrame,
) -> None:
    primary = targeted[
        targeted["feature_id"].eq(
            "global::microstructure::WholeBrain::fa_mean_edge_mean"
        )
        & targeted["contrast"].eq("AD_vs_CN")
    ].copy()
    top = targeted_robustness[targeted_robustness["contrast"].eq("AD_vs_CN")]
    broad_top = broad_robustness[broad_robustness["contrast"].eq("AD_vs_CN")]
    text = f"""# Submission-sprint exact-metadata reanalysis

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**Inferential status:** exploratory/directional analysis of historical participant-level outputs; not a corrected image-processing result and not confirmatory evidence.

## What this release answers

This release joins existing connectome summaries to the exact 530-pair diagnosis/date/acquisition manifest, excludes the 15 SMC records from CN/MCI/AD contrasts, and stress-tests a frozen 16-endpoint family across exact timing, same-phase, protocol, site-overlap, T1-source, and legacy-QC restrictions. It also retains a transparent all-feature screen for hypothesis generation. Rank-normal regression adjusts for age and sex, plus timing gap, T1 source, and collapsed acquisition protocol when they vary; covariance is clustered by site when feasible. Three diagnosis contrasts are emitted with confidence intervals and declared multiplicity corrections.

## Cohort strata

{markdown_table(counts, ['stratum', 'n_total', 'n_CN', 'n_MCI', 'n_AD', 'n_sites', 'n_protocol_buckets'])}

## Frozen primary retest: whole-brain FA, AD versus CN

{markdown_table(primary, ['stratum', 'n', 'n_CN', 'n_MCI', 'n_AD', 'rank_normal_beta', 'ci95_low', 'ci95_high', 'p_value', 'holm_p_within_family_stratum', 'cliffs_delta', 'design_condition_number'])}

The primary-family Holm correction covers all three diagnosis contrasts within each stratum. Stability across strata matters more than a single p-value because diagnosis and acquisition have limited common support.

## Highest-priority frozen endpoints for exact prior-art review

{markdown_table(top, ['feature_id', 'contrast', 'n_core_strata_estimable', 'sign_agreement_with_full', 'median_abs_rank_normal_beta', 'minimum_abs_rank_normal_beta', 'overlap_beta', 'overlap_p_uncorrected', 'directional_priority_score_not_inference'], 16)}

The directional priority score is only a triage device. It is not a test, corrected p-value, biological ranking, or novelty measure.

## All-feature hypothesis-generating screen

{markdown_table(broad_top, ['feature_id', 'contrast', 'n_core_strata_estimable', 'sign_agreement_with_full', 'median_abs_rank_normal_beta', 'minimum_abs_rank_normal_beta', 'overlap_beta', 'overlap_p_uncorrected', 'directional_priority_score_not_inference'], 20)}

All 214 features and every failure of direction or estimability remain in the CSV outputs; this table is only a queue for falsification and claim-specific literature searches.

## Non-negotiable limitations

1. These metrics came from the historical mixed/repaired processing products. Exact metadata correction does not repair image-processing semantics, spatial QC, matrix construction, or differential missingness.
2. The restricted overlap stratum is small, especially for AD. Wide intervals or loss of significance cannot prove absence, and a large point estimate cannot compensate for imprecision.
3. Protocol/site restrictions address measured overlap only. Residual scanner, selection, diagnosis, and multimodal timing confounding remain possible.
4. The all-feature screen is exploratory and cannot establish novelty. A title/abstract claim requires the targeted statistical audit, near-match/contradiction literature review, and explicit discovery language; corrected canary/full-run confirmation remains separate.
"""
    (out / "submission_sprint_summary.md").write_text(text, encoding="utf-8")


def write_run_manifest(out: Path, input_paths: dict[str, Path], counts: pd.DataFrame) -> None:
    manifest = {
        "schema_version": "1.0",
        "release": "submission_sprint_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "EXPLORATORY_EXISTING_OUTPUT_REANALYSIS",
        "image_or_voxel_data_read": False,
        "image_processing_run": False,
        "production_outputs_modified": False,
        "primary_groups": {"CN": 251, "MCI": 186, "AD": 78},
        "smc_separate": 15,
        "target_endpoints_frozen_before_run": list(TARGET_FAMILIES),
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
        "exact_manifest_cohort.csv",
        "stratum_counts.csv",
        "targeted_contrasts.csv",
        "broad_feature_contrasts.csv",
        "targeted_robustness.csv",
        "broad_feature_robustness.csv",
        "feature_missingness.csv",
        "submission_sprint_summary.md",
        "run_manifest.json",
        "figures/targeted_ad_cn_effect_stability.png",
        "figures/full_vs_overlap_ad_cn_effects.png",
    ]
    absent = [relative for relative in required if not (out / relative).is_file()]
    checks = {
        "required_files_present": not absent,
        "missing_required_files": absent,
    }
    targeted = pd.read_csv(out / "targeted_contrasts.csv")
    broad = pd.read_csv(out / "broad_feature_contrasts.csv")
    cohort = pd.read_csv(out / "exact_manifest_cohort.csv")
    checks.update(
        {
            "cohort_rows_530": len(cohort) == 530,
            "cohort_unique_subjects_530": cohort["subject_id"].nunique() == 530,
            "target_endpoint_count_16": targeted["feature_id"].nunique() == 16,
            "broad_feature_count_at_least_200": broad["feature_id"].nunique() >= 200,
            "three_contrasts_present": set(targeted["contrast"]) == {
                "MCI_vs_CN",
                "AD_vs_CN",
                "AD_vs_MCI",
            },
            "all_p_values_bounded": bool(
                targeted["p_value"].dropna().between(0, 1).all()
                and broad["p_value"].dropna().between(0, 1).all()
            ),
            "all_target_cis_ordered": bool(
                (targeted["ci95_low"] <= targeted["ci95_high"]).all()
            ),
        }
    )
    checks["pass"] = bool(
        all(value for key, value in checks.items() if key not in {"missing_required_files"})
    )
    checks["files"] = {
        str(path.relative_to(out)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "release_validation.json"
    }
    return checks


def run_release(out: Path, input_paths: dict[str, Path] = INPUTS) -> None:
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
        targeted, broad = run_models(cohort, metrics, strata)
        missingness = missingness_table(cohort, metrics, strata)
        targeted_robustness = robustness_table(targeted)
        broad_robustness = robustness_table(broad)

        cohort.to_csv(temporary / "exact_manifest_cohort.csv", index=False)
        counts.to_csv(temporary / "stratum_counts.csv", index=False)
        targeted.to_csv(temporary / "targeted_contrasts.csv", index=False)
        broad.to_csv(temporary / "broad_feature_contrasts.csv", index=False)
        targeted_robustness.to_csv(temporary / "targeted_robustness.csv", index=False)
        broad_robustness.to_csv(temporary / "broad_feature_robustness.csv", index=False)
        missingness.to_csv(temporary / "feature_missingness.csv", index=False)
        make_figures(targeted, temporary)
        write_summary(temporary, counts, targeted, targeted_robustness, broad_robustness)
        write_run_manifest(temporary, input_paths, counts)
        validation = validate_release(temporary)
        (temporary / "release_validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not validation["pass"]:
            raise RuntimeError(f"Release validation failed: {validation}")
        os.rename(temporary, out)
    except Exception:
        # Retain a failed build for diagnosis rather than deleting evidence.
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
    print(f"PASS: exploratory submission-sprint release written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
