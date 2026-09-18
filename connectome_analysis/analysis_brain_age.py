from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from connectome_analysis.analysis_config import GROUP_ORDER, AnalysisPaths
from connectome_analysis.analysis_plots import boxplot_with_points, save_dataframe, write_inference_markdown
from connectome_analysis.analysis_stats import group_descriptives, kruskal_summary, pairwise_robust_tests


def _merge_master(df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ["subject_id", "group", "age", "sex", "phase"] if c in master.columns]
    out = df.merge(master[keep].drop_duplicates("subject_id"), on="subject_id", how="left", suffixes=("", "_master"))
    if "group_master" in out.columns:
        out["group"] = out["group"].fillna(out["group_master"])
        out = out.drop(columns=["group_master"])
    return out


def _wide_feature_table(paths: AnalysisPaths, master: pd.DataFrame) -> pd.DataFrame:
    global_dti = pd.read_csv(paths.legacy_analysis_dir / "analysis2_multimetric_threaded" / "global_medians_long_CLEAN.csv")
    node_metrics = pd.read_csv(paths.legacy_analysis_dir / "analysis3_network" / "node_metrics_long_ALL.csv")
    local_dti = pd.read_csv(paths.legacy_analysis_dir / "analysis3_network" / "local_dti_medians_long_CLEAN.csv")

    global_wide = global_dti.pivot_table(index="subject_id", columns="metric", values="value", aggfunc="median")
    global_wide.columns = [f"global_{c}" for c in global_wide.columns]

    node_summary = (
        node_metrics.groupby("subject_id")[["strength", "degree", "nodal_eff"]]
        .mean()
        .rename(columns=lambda c: f"mean_{c}")
    )
    local_summary = (
        local_dti.groupby(["subject_id", "metric"])["value"]
        .median()
        .unstack("metric")
        .rename(columns=lambda c: f"local_median_{c}")
    )
    features = global_wide.join(node_summary, how="outer").join(local_summary, how="outer").reset_index()
    features = _merge_master(features, master)
    return features


def _brain_age_pipeline(n_components: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )


def run_brain_age_analysis(paths: AnalysisPaths, master: pd.DataFrame, n_repeats: int = 10) -> dict:
    out_dir = paths.section_dir("11", "brain_age")
    out_dir.mkdir(parents=True, exist_ok=True)
    feat = _wide_feature_table(paths, master)
    feat = feat.loc[feat["group"].isin(GROUP_ORDER)].copy()
    feat = feat.dropna(subset=["age"])
    feature_cols = [
        c
        for c in feat.columns
        if c.startswith("global_") or c.startswith("mean_") or c.startswith("local_median_")
    ]
    X = feat[feature_cols].to_numpy(dtype=float)
    y = feat["age"].to_numpy(dtype=float)
    n_components = max(2, min(12, X.shape[1], len(feat) - 2))

    rkf = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)
    pred_sum = np.zeros(len(feat), dtype=float)
    pred_n = np.zeros(len(feat), dtype=float)
    for train_idx, test_idx in rkf.split(X):
        pipe = _brain_age_pipeline(n_components=min(n_components, len(train_idx) - 1))
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        pred_sum[test_idx] += pred
        pred_n[test_idx] += 1
    feat["brain_age_pred"] = pred_sum / np.maximum(pred_n, 1)
    feat["BAG"] = feat["brain_age_pred"] - feat["age"]

    # Age-corrected BAG residual.
    coeffs = np.polyfit(feat["age"], feat["BAG"], deg=1)
    feat["BAG_age_corrected"] = feat["BAG"] - np.polyval(coeffs, feat["age"])

    # CN-only normative reference.
    cn = feat.loc[feat["group"] == "CN"].copy()
    if len(cn) >= 20:
        cn_pipe = _brain_age_pipeline(min(n_components, len(cn) - 1))
        cn_pipe.fit(cn[feature_cols].to_numpy(dtype=float), cn["age"].to_numpy(dtype=float))
        feat["brain_age_cn_ref"] = cn_pipe.predict(X)
        cn_resid = cn["age"].to_numpy(dtype=float) - cn_pipe.predict(cn[feature_cols].to_numpy(dtype=float))
        resid_sd = np.nanstd(cn_resid, ddof=1) if len(cn_resid) > 1 else np.nan
        feat["normative_z"] = (feat["age"] - feat["brain_age_cn_ref"]) / resid_sd if np.isfinite(resid_sd) and resid_sd > 0 else np.nan
    else:
        feat["brain_age_cn_ref"] = np.nan
        feat["normative_z"] = np.nan

    save_dataframe(feat, out_dir / "brain_age_predictions.csv")

    summary_rows = []
    for metric in ["BAG", "BAG_age_corrected", "normative_z"]:
        sub = feat[["subject_id", "group", metric]].dropna()
        if sub.empty:
            continue
        desc = group_descriptives(sub, metric)
        pairwise = pairwise_robust_tests(sub, metric)
        omni = kruskal_summary(sub, metric)
        save_dataframe(desc, out_dir / f"{metric}_descriptives.csv")
        save_dataframe(pairwise, out_dir / f"{metric}_pairwise.csv")
        pair_lines = []
        for _, row in pairwise.sort_values("bm_p").head(3).iterrows():
            pair_lines.append(
                f"{row.group_a} vs {row.group_b}: BM q={row.bm_q:.3g}, Cliff's d={row.cliffs_delta:.3g}"
            )
        boxplot_with_points(
            sub,
            "group",
            metric,
            title=f"{metric} by group",
            ylabel=metric,
            out_path=out_dir / f"{metric}_boxplot.png",
            omnibus_text=f"Kruskal p={omni.pvalue:.3g}",
            pairwise_lines=pair_lines,
        )
        summary_rows.append(
            {"metric": metric, "n": omni.n_total, "kw_stat": omni.statistic, "kw_p": omni.pvalue}
        )
    summary = pd.DataFrame(summary_rows)
    save_dataframe(summary, out_dir / "brain_age_summary.csv")
    write_inference_markdown(
        [
            "Brain-age prediction is regularized with PCA plus ridge regression and evaluated with repeated cross-validation.",
            "The main reported phenotype is the age-corrected BAG; the CN-referenced normative z-score is an additional sensitivity analysis.",
        ],
        out_dir / "brain_age_inference.md",
    )
    return {"features": feat, "summary": summary, "out_dir": out_dir}
