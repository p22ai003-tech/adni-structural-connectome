#!/usr/bin/env python3
"""Independent, non-destructive audit of the live structural-connectome results.

This script deliberately writes only to ``research_audit/outputs`` and
``research_audit/figures``.  It does not alter the dashboard, production
matrices, cohort selectors, or pipeline derivatives.

The analysis is an audit/sensitivity analysis, not a replacement confirmatory
analysis.  In particular, the old matrix-QC gate is retained as provenance but
is not treated as current anatomical validation because it predates recovery.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, norm, rankdata
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


PROJECT = Path("/home/ec2-user/exp")
ANALYSIS = Path("/data/derivatives/qc/analysis_cohort")
SC_QC = Path("/data/derivatives/qc/sc_matrix_qc")
OUT = PROJECT / "research_audit" / "outputs"
FIG = PROJECT / "research_audit" / "figures"
RANDOM_SEED = 20260718
GROUP_ORDER = ["CN", "MCI", "AD"]


INPUTS = {
    "master": ANALYSIS / "00_master" / "master_cohort.csv",
    "dti_master": PROJECT / "cohort" / "dti_master.csv",
    "selected_mri": PROJECT / "cohort" / "mri.csv",
    "manifest": SC_QC / "subject_manifest.csv",
    "global_micro": ANALYSIS
    / "03_global_microstructure_live"
    / "live_global_microstructure_subject_table.csv",
    "global_graph": ANALYSIS
    / "06_global_graph_live"
    / "live_global_graph_subject_table.csv",
    "functional_network_micro": ANALYSIS
    / "19_network_analysis"
    / "functional"
    / "network_microstructure_subject.csv",
    "functional_network_graph": ANALYSIS
    / "19_network_analysis"
    / "functional"
    / "network_graph_subject.csv",
    "functional_network_coupling": ANALYSIS
    / "19_network_analysis"
    / "functional"
    / "network_coupling_subject.csv",
    "functional_affectedness": ANALYSIS
    / "19_network_analysis"
    / "functional"
    / "network_affectedness_summary.csv",
    "anatomical_network_micro": ANALYSIS
    / "19_network_analysis"
    / "anatomical"
    / "network_microstructure_subject.csv",
    "anatomical_network_graph": ANALYSIS
    / "19_network_analysis"
    / "anatomical"
    / "network_graph_subject.csv",
    "anatomical_network_coupling": ANALYSIS
    / "19_network_analysis"
    / "anatomical"
    / "network_coupling_subject.csv",
    "anatomical_affectedness": ANALYSIS
    / "19_network_analysis"
    / "anatomical"
    / "network_affectedness_summary.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def bh(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    result = np.full(arr.shape, np.nan)
    keep = np.isfinite(arr)
    if keep.any():
        result[keep] = multipletests(arr[keep], method="fdr_bh")[1]
    return result


def first_nonnull(series: pd.Series):
    values = series.dropna()
    return values.iloc[0] if len(values) else np.nan


def parse_protocol(value: object) -> tuple[str, float, float]:
    text = "" if pd.isna(value) else str(value)
    manufacturer = re.search(r"Manufacturer=([^;]+)", text)
    field = re.search(r"Field Strength=([0-9.]+)", text)
    gradients = re.search(r"Gradient Directions=([0-9.]+)", text)
    return (
        manufacturer.group(1).strip() if manufacturer else "Missing",
        float(field.group(1)) if field else np.nan,
        float(gradients.group(1)) if gradients else np.nan,
    )


def corrected_group(diagnosis: object) -> object:
    if pd.isna(diagnosis):
        return np.nan
    diagnosis = str(diagnosis).strip().upper()
    if diagnosis in {"MCI", "EMCI", "LMCI"}:
        return "MCI"
    if diagnosis in {"CN", "AD"}:
        return diagnosis
    if diagnosis == "SMC":
        return "SMC"
    return diagnosis


def build_cohort_manifest() -> pd.DataFrame:
    master = pd.read_csv(INPUTS["master"])
    dti = pd.read_csv(INPUTS["dti_master"])
    selected_mri = pd.read_csv(INPUTS["selected_mri"])
    manifest = pd.read_csv(INPUTS["manifest"])

    master["subject_id"] = master["subject_id"].astype(str)
    manifest["subject_id"] = manifest["sid"].str.extract(
        r"^(\d{3}_S_\d{4})", expand=False
    )
    manifest["actual_dti_image_id"] = pd.to_numeric(
        manifest["sid"].str.extract(r"_I(\d+)$", expand=False), errors="coerce"
    ).astype("Int64")
    manifest["actual_t1_image_id"] = pd.to_numeric(
        manifest["t1_path"].str.extract(r"_I(\d+)", expand=False), errors="coerce"
    ).astype("Int64")

    # The current manifest has one selected matrix per subject.  Fail loudly if
    # a future run violates this assumption rather than silently choosing one.
    selected = manifest[manifest["subject_id"].isin(master["subject_id"])].copy()
    dup = selected["subject_id"].value_counts()
    if (dup > 1).any():
        raise RuntimeError(
            "Current subject_manifest has multiple selected rows for: "
            + ", ".join(dup[dup > 1].index[:10])
        )

    keep = [
        "subject_id",
        "sid",
        "actual_dti_image_id",
        "actual_t1_image_id",
        "density",
        "band",
        "radial",
        "conn_location",
        "n_weights",
        "stages_present",
        "mask_voxels",
        "mask_quality",
        "reg_ncc",
        "label_survival",
        "best_recipe",
        "t1_path",
        "parc_path",
        "tck_path",
        "sift_path",
    ]
    cohort = master.merge(selected[keep], on="subject_id", how="left", validate="1:1")

    dti["Image ID"] = pd.to_numeric(dti["Image ID"], errors="coerce").astype("Int64")
    dti = dti.drop_duplicates("Image ID").set_index("Image ID")
    dti_columns = {
        "Subject ID": "dti_subject_id",
        "Research Group": "diagnosis_original",
        "Phase": "dti_phase",
        "Age": "dti_age",
        "Sex": "dti_sex",
        "Study Date": "dti_study_date",
        "Visit": "dti_visit",
        "Imaging Protocol": "dti_protocol",
        "Description": "dti_description",
        "Type": "dti_type",
        "APOE A1": "apoe_a1",
        "APOE A2": "apoe_a2",
    }
    for source, target in dti_columns.items():
        cohort[target] = cohort["actual_dti_image_id"].map(dti[source])

    selected_mri = selected_mri.drop_duplicates("Subject ID").set_index("Subject ID")
    for source, target in {
        "Image ID": "selected_t1_image_id_metadata",
        "Age": "t1_age",
        "Phase": "t1_phase",
        "Description": "t1_description",
        "Type": "t1_type",
    }.items():
        cohort[target] = cohort["subject_id"].map(selected_mri[source])

    parsed = cohort["dti_protocol"].map(parse_protocol)
    cohort[["manufacturer", "field_strength_t", "gradient_directions"]] = pd.DataFrame(
        parsed.tolist(), index=cohort.index
    )
    cohort["protocol_key"] = (
        cohort["manufacturer"].astype(str)
        + "|"
        + cohort["field_strength_t"].map(lambda x: "NA" if pd.isna(x) else f"{x:g}T")
        + "|"
        + cohort["gradient_directions"].map(
            lambda x: "NA" if pd.isna(x) else f"{x:g}dir"
        )
    )
    cohort["site"] = cohort["subject_id"].str[:3]
    cohort["group_dashboard"] = cohort["group"]
    cohort["group_corrected"] = cohort["diagnosis_original"].map(corrected_group)
    cohort["smc_misclassified_as_mci"] = (
        cohort["diagnosis_original"].astype(str).str.upper().eq("SMC")
        & cohort["group_dashboard"].eq("MCI")
    )

    cohort["dti_t1_age_gap_years"] = (
        pd.to_numeric(cohort["dti_age"], errors="coerce")
        - pd.to_numeric(cohort["t1_age"], errors="coerce")
    ).abs()
    cohort["dti_t1_phase_match"] = cohort["dti_phase"].eq(cohort["t1_phase"])
    cohort["contemporaneous_t1"] = (
        cohort["dti_t1_age_gap_years"].le(0.5) & cohort["dti_t1_phase_match"]
    )
    cohort["siemens_3t_54dir"] = (
        cohort["manufacturer"].str.upper().eq("SIEMENS")
        & cohort["field_strength_t"].eq(3.0)
        & cohort["gradient_directions"].eq(54.0)
    )
    cohort["dashboard_dti_id_matches_actual"] = pd.to_numeric(
        cohort["Image ID"], errors="coerce"
    ).astype("Int64").eq(cohort["actual_dti_image_id"])
    cohort["dashboard_t1_id_matches_actual"] = pd.to_numeric(
        cohort["Image ID_mri"], errors="coerce"
    ).astype("Int64").eq(cohort["actual_t1_image_id"])
    cohort["apoe_e4_count_corrected"] = (
        pd.to_numeric(cohort["apoe_a1"], errors="coerce").eq(4).astype(int)
        + pd.to_numeric(cohort["apoe_a2"], errors="coerce").eq(4).astype(int)
    )
    cohort["apoe_e4_carrier_corrected"] = cohort["apoe_e4_count_corrected"].gt(0)
    return cohort


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    table = pd.crosstab(x, y)
    if min(table.shape) < 2:
        return np.nan, np.nan, int(table.to_numpy().sum())
    chi2, p, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min(kcorr - 1, rcorr - 1)
    value = math.sqrt(phi2corr / denom) if denom > 0 else np.nan
    return value, p, int(n)


def cohort_tables(cohort: pd.DataFrame) -> None:
    cohort.to_csv(OUT / "cohort_audit_manifest.csv", index=False)

    pairing = []
    for group in [*GROUP_ORDER, "ALL"]:
        part = cohort if group == "ALL" else cohort[cohort["group_corrected"].eq(group)]
        gap = part["dti_t1_age_gap_years"].dropna()
        pairing.append(
            {
                "group": group,
                "n": len(part),
                "n_gap_available": len(gap),
                "mean_gap_years": gap.mean(),
                "median_gap_years": gap.median(),
                "p90_gap_years": gap.quantile(0.9),
                "max_gap_years": gap.max(),
                "n_le_0_5y": gap.le(0.5).sum(),
                "pct_le_0_5y": 100 * gap.le(0.5).mean(),
                "n_le_2y": gap.le(2).sum(),
                "pct_le_2y": 100 * gap.le(2).mean(),
                "n_phase_mismatch": (~part["dti_t1_phase_match"]).sum(),
            }
        )
    pd.DataFrame(pairing).to_csv(OUT / "dti_t1_pairing_summary.csv", index=False)

    confounders = []
    corrected = cohort[cohort["group_corrected"].isin(GROUP_ORDER)]
    for field in [
        "dti_phase",
        "manufacturer",
        "gradient_directions",
        "protocol_key",
        "site",
        "radial",
    ]:
        value, p, n = cramers_v(corrected["group_corrected"], corrected[field])
        confounders.append(
            {"acquisition_factor": field, "n": n, "cramers_v": value, "chi2_p": p}
        )
        pd.crosstab(
            corrected[field].fillna("Missing"), corrected["group_corrected"], margins=True
        ).to_csv(OUT / f"group_by_{field}.csv")
    pd.DataFrame(confounders).to_csv(OUT / "acquisition_group_association.csv", index=False)

    qc = (
        cohort.groupby(["group_corrected", "sc_matrix_qc_status"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    qc.to_csv(OUT / "legacy_qc_status_by_group.csv", index=False)

    mismatched = cohort.loc[
        ~cohort["dashboard_dti_id_matches_actual"],
        [
            "subject_id",
            "group_dashboard",
            "Image ID",
            "actual_dti_image_id",
            "Description",
            "dti_description",
            "Type",
            "dti_type",
        ],
    ]
    mismatched.to_csv(OUT / "dashboard_dti_metadata_mismatches.csv", index=False)


@dataclass(frozen=True)
class Stratum:
    name: str
    mask: pd.Series
    group_column: str
    model_adjustment: str


def make_strata(cohort: pd.DataFrame) -> list[Stratum]:
    corrected = cohort["group_corrected"].isin(GROUP_ORDER)
    contemp = cohort["contemporaneous_t1"] & corrected
    s54 = cohort["siemens_3t_54dir"] & corrected
    core = contemp & s54

    # Overlap sites are determined inside the protocol/contemporaneous subset.
    overlap_sites = (
        cohort.loc[core].groupby("site")["group_corrected"].nunique().loc[lambda x: x >= 2].index
    )
    old_qc = cohort["sc_matrix_qc_include"].fillna(False).astype(bool)
    radial2 = pd.to_numeric(cohort["radial"], errors="coerce").eq(2)
    return [
        Stratum(
            "dashboard_all_including_smc",
            cohort["group_dashboard"].isin(GROUP_ORDER),
            "group_dashboard",
            "age_sex_phase_protocol_site_radial_t1gap",
        ),
        Stratum(
            "diagnosis_corrected_all",
            corrected,
            "group_corrected",
            "age_sex_phase_protocol_site_radial_t1gap",
        ),
        Stratum(
            "contemporaneous_t1",
            contemp,
            "group_corrected",
            "age_sex_protocol_site_radial",
        ),
        Stratum(
            "siemens_3t_54dir",
            s54,
            "group_corrected",
            "age_sex_phase_site_radial_t1gap",
        ),
        Stratum(
            "contemporaneous_siemens_3t_54dir",
            core,
            "group_corrected",
            "age_sex_site_radial",
        ),
        Stratum(
            "contemporary_protocol_overlap_sites",
            core & cohort["site"].isin(overlap_sites),
            "group_corrected",
            "age_sex_site_radial",
        ),
        Stratum(
            "candidate_plus_legacy_qc_radial2",
            core & cohort["site"].isin(overlap_sites) & old_qc & radial2,
            "group_corrected",
            "age_sex_site",
        ),
    ]


def collect_metrics() -> pd.DataFrame:
    records = []

    global_micro = pd.read_csv(INPUTS["global_micro"])
    for metric in [
        "fa_mean_edge_mean",
        "md_mean_edge_mean",
        "ad_mean_edge_mean",
        "rd_mean_edge_mean",
    ]:
        frame = global_micro[["subject_id", metric]].dropna().rename(columns={metric: "value"})
        frame["mapping"] = "global"
        frame["feature_family"] = "microstructure"
        frame["network"] = "WholeBrain"
        frame["metric"] = metric
        records.append(frame)

    global_graph = pd.read_csv(INPUTS["global_graph"])
    for metric in [
        "mean_strength",
        "total_strength",
        "density",
        "global_efficiency",
        "charpath_len",
    ]:
        frame = global_graph[["subject_id", metric]].dropna().rename(columns={metric: "value"})
        frame["mapping"] = "global"
        frame["feature_family"] = "graph"
        frame["network"] = "WholeBrain"
        frame["metric"] = metric
        records.append(frame)

    for mapping in ["functional", "anatomical"]:
        base = ANALYSIS / "19_network_analysis" / mapping
        for family, filename in [
            ("microstructure", "network_microstructure_subject.csv"),
            ("graph", "network_graph_subject.csv"),
            # This is topology--microstructure coupling.  It is intentionally
            # not called structure--function coupling in audit outputs.
            ("topology_microstructure_coupling", "network_coupling_subject.csv"),
        ]:
            frame = pd.read_csv(base / filename)[
                ["subject_id", "network", "metric", "value"]
            ].dropna(subset=["value"])
            frame["mapping"] = mapping
            frame["feature_family"] = family
            records.append(frame)

    long = pd.concat(records, ignore_index=True)
    long["feature_id"] = (
        long["mapping"].astype(str)
        + "::"
        + long["feature_family"].astype(str)
        + "::"
        + long["network"].astype(str)
        + "::"
        + long["metric"].astype(str)
    )
    return long


def adjusted_group_p(frame: pd.DataFrame, adjustment: str) -> tuple[float, int, float]:
    """HC3 rank-ANCOVA omnibus group p, rank and condition number.

    This is a diagnostic robustness model.  The design can remain weakly
    identifiable when diagnosis and acquisition are nearly aliased; the
    returned condition number makes that visible.
    """

    use = frame.copy()
    use = use[np.isfinite(use["value"]) & use["group_model"].isin(GROUP_ORDER)]
    if use["group_model"].value_counts().reindex(GROUP_ORDER).min() < 5:
        return np.nan, 0, np.nan
    n = len(use)
    use["rank_z"] = norm.ppf((rankdata(use["value"], method="average") - 0.5) / n)

    terms = ["C(group_model)", "age_model", "C(sex_model)"]
    if "phase" in adjustment and use["dti_phase"].nunique(dropna=True) > 1:
        terms.append("C(dti_phase)")
    if "protocol" in adjustment and use["protocol_key"].nunique(dropna=True) > 1:
        terms.append("C(protocol_key)")
    if "site" in adjustment and use["site"].nunique(dropna=True) > 1:
        terms.append("C(site)")
    if "radial" in adjustment and use["radial"].nunique(dropna=True) > 1:
        terms.append("C(radial)")
    if "t1gap" in adjustment and use["dti_t1_age_gap_years"].notna().any():
        terms.append("dti_t1_age_gap_years")

    needed = ["rank_z", "group_model", "age_model", "sex_model"]
    if "C(dti_phase)" in terms:
        needed.append("dti_phase")
    if "C(protocol_key)" in terms:
        needed.append("protocol_key")
    if "C(site)" in terms:
        needed.append("site")
    if "C(radial)" in terms:
        needed.append("radial")
    if "dti_t1_age_gap_years" in terms:
        needed.append("dti_t1_age_gap_years")
    use = use.dropna(subset=needed)
    if use["group_model"].value_counts().reindex(GROUP_ORDER, fill_value=0).min() < 5:
        return np.nan, 0, np.nan
    try:
        model = smf.ols("rank_z ~ " + " + ".join(terms), data=use).fit()
        table = sm.stats.anova_lm(model, typ=2, robust="hc3")
        p = float(table.loc["C(group_model)", "PR(>F)"])
        rank = int(np.linalg.matrix_rank(model.model.exog))
        condition = float(np.linalg.cond(model.model.exog))
        return p, rank, condition
    except Exception:
        return np.nan, 0, np.nan


def run_sensitivity(cohort: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_cols = [
        "subject_id",
        "group_dashboard",
        "group_corrected",
        "dti_age",
        "dti_sex",
        "dti_phase",
        "protocol_key",
        "site",
        "radial",
        "dti_t1_age_gap_years",
    ]
    long = metrics.merge(cohort[meta_cols], on="subject_id", how="left", validate="m:1")
    long["age_model"] = pd.to_numeric(long["dti_age"], errors="coerce")
    long["sex_model"] = long["dti_sex"].fillna("Missing").astype(str)

    stratum_counts = []
    results = []
    pairwise = []
    for stratum in make_strata(cohort):
        subjects = cohort.loc[stratum.mask, "subject_id"]
        count_frame = cohort.loc[stratum.mask]
        counts = count_frame[stratum.group_column].value_counts()
        stratum_counts.append(
            {
                "stratum": stratum.name,
                "n_total": len(count_frame),
                **{f"n_{g}": int(counts.get(g, 0)) for g in GROUP_ORDER},
                "n_smc": int(counts.get("SMC", 0)),
                "n_sites": int(count_frame["site"].nunique()),
                "n_protocols": int(count_frame["protocol_key"].nunique()),
            }
        )
        selected = long[long["subject_id"].isin(subjects)].copy()
        selected["group_model"] = selected[stratum.group_column]
        for feature_id, frame in selected.groupby("feature_id", sort=False):
            groups = {
                g: frame.loc[frame["group_model"].eq(g), "value"].dropna().to_numpy()
                for g in GROUP_ORDER
            }
            if min(map(len, groups.values())) < 5:
                continue
            kw_p = float(kruskal(*(groups[g] for g in GROUP_ORDER)).pvalue)
            adj_p, design_rank, condition = adjusted_group_p(frame, stratum.model_adjustment)
            first = frame.iloc[0]
            results.append(
                {
                    "stratum": stratum.name,
                    "feature_id": feature_id,
                    "mapping": first["mapping"],
                    "feature_family": first["feature_family"],
                    "network": first["network"],
                    "metric": first["metric"],
                    "n": sum(len(v) for v in groups.values()),
                    **{f"n_{g}": len(groups[g]) for g in GROUP_ORDER},
                    **{f"median_{g}": np.median(groups[g]) for g in GROUP_ORDER},
                    "kw_p": kw_p,
                    "adjusted_rank_ancova_p": adj_p,
                    "design_rank": design_rank,
                    "design_condition_number": condition,
                    "adjustment": stratum.model_adjustment,
                }
            )
            for left, right in [("MCI", "CN"), ("AD", "CN"), ("AD", "MCI")]:
                x, y = groups[left], groups[right]
                test = mannwhitneyu(x, y, alternative="two-sided")
                delta = 2 * float(test.statistic) / (len(x) * len(y)) - 1
                pairwise.append(
                    {
                        "stratum": stratum.name,
                        "feature_id": feature_id,
                        "contrast": f"{left}_vs_{right}",
                        "n_left": len(x),
                        "n_right": len(y),
                        "cliffs_delta_left_minus_right": delta,
                        "mann_whitney_p": float(test.pvalue),
                    }
                )

    counts_df = pd.DataFrame(stratum_counts)
    counts_df.to_csv(OUT / "sensitivity_stratum_counts.csv", index=False)
    result_df = pd.DataFrame(results)
    pair_df = pd.DataFrame(pairwise)
    if len(result_df):
        result_df["kw_q_across_audit_features"] = result_df.groupby("stratum")["kw_p"].transform(bh)
        result_df["adjusted_q_across_audit_features"] = result_df.groupby("stratum")[
            "adjusted_rank_ancova_p"
        ].transform(bh)
        result_df.to_csv(OUT / "metric_sensitivity_results.csv", index=False)
    if len(pair_df):
        pair_df["pairwise_q_across_audit_features"] = pair_df.groupby(
            ["stratum", "contrast"]
        )["mann_whitney_p"].transform(bh)
        pair_df.to_csv(OUT / "metric_pairwise_sensitivity.csv", index=False)
    return result_df, pair_df


def missingness_audit(cohort: pd.DataFrame, metrics: pd.DataFrame) -> None:
    eligible = cohort[cohort["group_corrected"].isin(GROUP_ORDER)]
    denominator = eligible["group_corrected"].value_counts().to_dict()
    merged = metrics.merge(
        eligible[["subject_id", "group_corrected"]], on="subject_id", how="inner"
    )
    rows = []
    for feature_id, frame in merged.groupby("feature_id"):
        first = frame.iloc[0]
        present = frame.groupby("group_corrected")["subject_id"].nunique()
        row = {
            "feature_id": feature_id,
            "mapping": first["mapping"],
            "feature_family": first["feature_family"],
            "network": first["network"],
            "metric": first["metric"],
        }
        for group in GROUP_ORDER:
            n = int(present.get(group, 0))
            den = int(denominator.get(group, 0))
            row[f"n_present_{group}"] = n
            row[f"pct_missing_{group}"] = 100 * (1 - n / den) if den else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "feature_missingness_by_group.csv", index=False)


def existing_inference_audit() -> None:
    rows = []
    for mapping in ["functional", "anatomical"]:
        path = ANALYSIS / "19_network_analysis" / mapping / "network_affectedness_summary.csv"
        frame = pd.read_csv(path)
        for family, part in frame.groupby("feature_family"):
            rows.append(
                {
                    "mapping": mapping,
                    "feature_family": family,
                    "n_tests": len(part),
                    "n_q_kw_lt_0_05": int(part["q_kw"].lt(0.05).sum()),
                    "n_perm_p_lt_0_05": int(part["perm_p"].lt(0.05).sum()),
                    "n_q_significant_but_perm_not": int(
                        (part["q_kw"].lt(0.05) & part["perm_p"].ge(0.05)).sum()
                    ),
                    "n_perm_significant_but_q_not": int(
                        (part["perm_p"].lt(0.05) & part["q_kw"].ge(0.05)).sum()
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "existing_q_vs_permutation_summary.csv", index=False)


def make_figures(cohort: pd.DataFrame, results: pd.DataFrame) -> None:
    corrected = cohort[cohort["group_corrected"].isin(GROUP_ORDER)].copy()
    colors = {"CN": "#2c7fb8", "MCI": "#fdae61", "AD": "#d7191c"}

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [
        corrected.loc[corrected["group_corrected"].eq(g), "dti_t1_age_gap_years"].dropna()
        for g in GROUP_ORDER
    ]
    bp = ax.boxplot(data, labels=GROUP_ORDER, showfliers=False, patch_artist=True)
    for patch, group in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(colors[group])
        patch.set_alpha(0.75)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="0.5-y audit limit")
    ax.set_ylabel("Absolute DTI–T1 age gap (years)")
    ax.set_title("Anatomical image timing mismatch in the dashboard cohort")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "dti_t1_age_gap_by_group.png", dpi=200)
    plt.close(fig)

    protocol = pd.crosstab(corrected["protocol_key"], corrected["group_corrected"])
    protocol = protocol.loc[protocol.sum(axis=1).sort_values(ascending=False).index]
    protocol.to_csv(OUT / "protocol_counts_by_group.csv")
    top = protocol.head(8)
    other = protocol.iloc[8:].sum().to_frame().T
    if len(other) and other.to_numpy().sum() > 0:
        other.index = ["Other protocols"]
        top = pd.concat([top, other])
    percent = top.div(top.sum(axis=0), axis=1).T * 100
    fig, ax = plt.subplots(figsize=(11, 5.5))
    percent.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("Within-diagnosis percentage")
    ax.set_xlabel("Diagnosis")
    ax.set_title("DTI protocol composition differs sharply by diagnosis")
    ax.legend(title="Protocol", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "protocol_composition_by_group.png", dpi=200)
    plt.close(fig)

    if len(results):
        target_features = [
            "global::microstructure::WholeBrain::fa_mean_edge_mean",
            "global::microstructure::WholeBrain::md_mean_edge_mean",
            "functional::microstructure::Limbic::fa_mean",
            "functional::microstructure::Limbic::md_mean",
            "functional::microstructure::DMN::fa_mean",
            "functional::microstructure::DMN::md_mean",
            "global::graph::WholeBrain::global_efficiency",
            "global::graph::WholeBrain::mean_strength",
        ]
        view = results[results["feature_id"].isin(target_features)].copy()
        pivot = view.pivot(
            index="feature_id", columns="stratum", values="adjusted_rank_ancova_p"
        )
        pivot = pivot.reindex(target_features).dropna(how="all")
        if len(pivot):
            values = -np.log10(pivot.clip(lower=1e-12))
            fig, ax = plt.subplots(figsize=(12, 6))
            image = ax.imshow(values.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=4)
            ax.set_xticks(range(len(values.columns)), values.columns, rotation=35, ha="right")
            ax.set_yticks(
                range(len(values.index)),
                [x.replace("::", " / ") for x in values.index],
                fontsize=8,
            )
            ax.set_title("Sensitivity of adjusted diagnosis effects across audit cohorts")
            cbar = fig.colorbar(image, ax=ax)
            cbar.set_label("−log10(HC3 rank-ANCOVA p)")
            fig.tight_layout()
            fig.savefig(FIG / "key_metric_sensitivity_heatmap.png", dpi=200)
            plt.close(fig)


def write_summary(cohort: pd.DataFrame, results: pd.DataFrame) -> None:
    gaps = cohort["dti_t1_age_gap_years"].dropna()
    corrected = cohort[cohort["group_corrected"].isin(GROUP_ORDER)]
    old_excluded = (~cohort["sc_matrix_qc_include"].fillna(False).astype(bool)).sum()
    mismatch_ids = (~cohort["dashboard_dti_id_matches_actual"]).sum()
    smc = cohort["smc_misclassified_as_mci"].sum()
    s54 = corrected["siemens_3t_54dir"].sum()
    contemp_s54 = (corrected["siemens_3t_54dir"] & corrected["contemporaneous_t1"]).sum()

    lines = [
        "# Independent statistical audit summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Cohort integrity",
        "",
        f"- Dashboard cohort: **{len(cohort)}** subjects.",
        f"- DTI–T1 absolute age gap: mean **{gaps.mean():.2f} y**, median **{gaps.median():.2f} y**, "
        f"90th percentile **{gaps.quantile(0.9):.2f} y**, maximum **{gaps.max():.2f} y**.",
        f"- Contemporaneous (same phase and age gap ≤0.5 y): **{cohort['contemporaneous_t1'].sum()} / {len(cohort)}**.",
        f"- Dashboard DTI metadata row disagrees with the actual manifest DTI image ID for **{mismatch_ids}** subjects.",
        f"- SMC records currently collapsed into MCI: **{smc}**.",
        f"- Current dense cohort still carries a legacy `sc_matrix_qc_include=False` for **{old_excluded}** subjects; "
        "the gate predates recovery and must be regenerated rather than blindly applied.",
        "",
        "## Acquisition and sensitivity",
        "",
        f"- Siemens 3 T / 54-direction subset: **{s54}**; adding contemporaneous T1 leaves **{contemp_s54}**.",
        "- `metric_sensitivity_results.csv` applies BH correction across every audited feature within each cohort definition.",
        "- `adjusted_rank_ancova_p` is a diagnostic HC3 rank-ANCOVA including the covariates named in `adjustment`; "
        "large condition numbers flag weak identification caused by diagnosis–acquisition aliasing.",
        "- These sensitivity analyses do not repair invalid anatomical pairing or replace a rerun with visit-matched T1.",
        "",
        "## Interpretation rule",
        "",
        "A result is not manuscript-ready merely because one row has p<0.05. It must be directionally stable, survive the "
        "prespecified family-wise/FDR scope, remain estimable in contemporaneous and protocol/site-overlap analyses, and be "
        "replicated after the matrices and QC are regenerated from visit-matched inputs.",
        "",
    ]
    (OUT / "audit_statistics_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_run_manifest() -> None:
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "random_seed": RANDOM_SEED,
        "python": sys.version,
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "inputs": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "sha256": sha256(path),
            }
            for name, path in INPUTS.items()
        },
    }
    (OUT / "audit_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    np.random.seed(RANDOM_SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cohort = build_cohort_manifest()
    cohort_tables(cohort)
    metrics = collect_metrics()
    missingness_audit(cohort, metrics)
    existing_inference_audit()
    results, _ = run_sensitivity(cohort, metrics)
    make_figures(cohort, results)
    write_summary(cohort, results)
    write_run_manifest()
    print(f"Audit complete: {len(cohort)} subjects, {metrics['feature_id'].nunique()} features")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
