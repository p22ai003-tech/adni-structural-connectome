from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from connectome_analysis.analysis_config import GROUP_ORDER, AnalysisPaths
from connectome_analysis.analysis_edges import load_connectome_stack
from connectome_analysis.analysis_plots import boxplot_with_points, heatmap_matrix, manhattan_plot, save_dataframe, scatter_with_group_fit, write_inference_markdown
from connectome_analysis.analysis_stats import bh_fdr, group_descriptives, kruskal_summary, pairwise_robust_tests, permutation_group_effect


def _master_covariates(master: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["subject_id", "group", "age", "sex", "phase"] if c in master.columns]
    return master[cols].drop_duplicates("subject_id")


def _finite_upper(mat: np.ndarray, positive: bool = True) -> np.ndarray:
    iu = np.triu_indices_from(mat, 1)
    vals = mat[iu].astype(float)
    mask = np.isfinite(vals)
    if positive:
        mask &= vals > 0
    return vals[mask]


def _metric_stacks(paths: AnalysisPaths, master: pd.DataFrame, metrics: list[str]) -> dict[str, object]:
    out = {}
    for metric in metrics:
        try:
            out[metric] = load_connectome_stack(paths, master, connectome_type=metric)
        except FileNotFoundError:
            continue
    if not out:
        raise FileNotFoundError(f"No live connectome matrices found for {metrics}")
    return out


def _stat_plot_metric(df: pd.DataFrame, metric: str, out_dir, title: str, ylabel: str) -> dict:
    sub = df[["subject_id", "group", "age", "sex", "phase", metric]].dropna(subset=[metric])
    desc = group_descriptives(sub, metric)
    pairwise = pairwise_robust_tests(sub, metric)
    omni = kruskal_summary(sub, metric)
    perm = permutation_group_effect(sub, metric)
    save_dataframe(desc, out_dir / f"{metric}_descriptives.csv")
    save_dataframe(pairwise, out_dir / f"{metric}_pairwise.csv")
    pair_lines = [
        f"{row.group_a} vs {row.group_b}: BM q={row.bm_q:.3g}, Cliff's d={row.cliffs_delta:.3g}"
        for _, row in pairwise.sort_values("bm_p").head(3).iterrows()
    ]
    boxplot_with_points(
        sub,
        "group",
        metric,
        title=title,
        ylabel=ylabel,
        out_path=out_dir / f"{metric}_boxplot.png",
        omnibus_text=f"Kruskal p={omni.pvalue:.3g}; perm p={perm['p_perm']:.3g}",
        pairwise_lines=pair_lines,
    )
    return {
        "metric": metric,
        "n": omni.n_total,
        "kw_stat": omni.statistic,
        "kw_p": omni.pvalue,
        "perm_f": perm["f_stat"],
        "perm_p": perm["p_perm"],
    }


def run_live_global_microstructure(paths: AnalysisPaths, master: pd.DataFrame) -> dict:
    out_dir = paths.section_dir("03", "global_microstructure_live")
    out_dir.mkdir(parents=True, exist_ok=True)
    stacks = _metric_stacks(paths, master, ["fa_mean", "md_mean", "ad_mean", "rd_mean"])
    rows = {}
    for metric, stack in stacks.items():
        for sid, group, mat in zip(stack.subject_ids, stack.groups, stack.matrices):
            row = rows.setdefault(sid, {"subject_id": sid, "group": group})
            vals = _finite_upper(mat, positive=True)
            row[f"{metric}_edge_mean"] = float(np.nanmean(vals)) if vals.size else np.nan
            row[f"{metric}_edge_median"] = float(np.nanmedian(vals)) if vals.size else np.nan
            row[f"{metric}_edge_iqr"] = float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)) if vals.size else np.nan
            row[f"{metric}_n_edges"] = int(vals.size)
    table = pd.DataFrame(rows.values()).merge(_master_covariates(master), on=["subject_id", "group"], how="left")
    save_dataframe(table, out_dir / "live_global_microstructure_subject_table.csv")
    summary_rows = []
    for metric in [c for c in table.columns if c.endswith("_edge_mean") or c.endswith("_edge_median")]:
        label = metric.replace("_", " ")
        summary_rows.append(_stat_plot_metric(table, metric, out_dir, f"{label} by group", label))
    summary = pd.DataFrame(summary_rows)
    save_dataframe(summary, out_dir / "live_global_microstructure_summary.csv")
    write_inference_markdown(
        [
            "Live fallback microstructure uses current tcksample/tck2connectome edge matrices, not the missing legacy /qc/analysis tables.",
            "Current live edge microstructure includes every validated edge-sampled tensor matrix currently present in /data/derivatives/connectomes.",
            "RD and AD are populated from Step 7 post once SC_AAL166_*_rd_mean.csv and SC_AAL166_*_ad_mean.csv are generated.",
            f"Strongest omnibus result: {summary.sort_values('kw_p').iloc[0].metric if not summary.empty else 'none'}",
        ],
        out_dir / "live_global_microstructure_inference.md",
    )
    return {"table": table, "summary": summary, "out_dir": out_dir}


def run_live_nodewise_microstructure(paths: AnalysisPaths, master: pd.DataFrame, top_n: int = 8) -> dict:
    out_dir = paths.section_dir("04", "node_microstructure_live")
    out_dir.mkdir(parents=True, exist_ok=True)
    stacks = _metric_stacks(paths, master, ["fa_mean", "md_mean", "ad_mean", "rd_mean"])
    rows = []
    for metric, stack in stacks.items():
        for sid, group, mat in zip(stack.subject_ids, stack.groups, stack.matrices):
            valid = np.isfinite(mat) & (mat > 0)
            np.fill_diagonal(valid, False)
            for node in range(mat.shape[0]):
                vals = mat[node, valid[node]]
                rows.append(
                    {
                        "subject_id": sid,
                        "group": group,
                        "node": node + 1,
                        "node_name": f"AAL_{node + 1:03d}",
                        "metric": metric,
                        "value": float(np.nanmean(vals)) if vals.size else np.nan,
                        "n_edges": int(vals.size),
                    }
                )
    table = pd.DataFrame(rows).merge(_master_covariates(master), on=["subject_id", "group"], how="left")
    save_dataframe(table, out_dir / "live_node_microstructure_long.csv")
    summaries = []
    tests_by_metric = {}
    for metric, sub_metric in table.dropna(subset=["value"]).groupby("metric"):
        test_rows = []
        for node, sub in sub_metric.groupby("node"):
            omni = kruskal_summary(sub, "value")
            test_rows.append(
                {
                    "node": node,
                    "node_name": sub["node_name"].iloc[0],
                    "kw_stat": omni.statistic,
                    "kw_p": omni.pvalue,
                    "n": omni.n_total,
                }
            )
        tests = pd.DataFrame(test_rows).sort_values("kw_p", na_position="last")
        tests["kw_q"] = bh_fdr(tests["kw_p"].to_numpy())
        tests_by_metric[metric] = tests
        save_dataframe(tests, out_dir / f"{metric}_node_tests.csv")
        manhattan_plot(
            tests,
            x_col="node",
            y_col="kw_q",
            sig_col="kw_q",
            threshold=0.05,
            title=f"{metric}: node-wise group effect",
            ylabel="BH-FDR q",
            out_path=out_dir / f"{metric}_node_manhattan.png",
        )
        for _, row in tests.nsmallest(top_n, "kw_q").iterrows():
            plot_df = sub_metric.loc[sub_metric["node"] == row["node"], ["group", "value"]]
            boxplot_with_points(
                plot_df,
                "group",
                "value",
                title=f"{metric}: {row.node_name}",
                ylabel=metric,
                out_path=out_dir / f"{metric}_node_{int(row.node):03d}.png",
                omnibus_text=f"Kruskal q={row.kw_q:.3g}",
            )
        summaries.append(
            {
                "metric": metric,
                "n_nodes_tested": len(tests),
                "n_fdr_sig": int((tests["kw_q"] < 0.05).sum()),
                "best_node": tests.iloc[0].node_name if len(tests) else np.nan,
                "best_q": tests.iloc[0].kw_q if len(tests) else np.nan,
            }
        )
    summary = pd.DataFrame(summaries)
    save_dataframe(summary, out_dir / "live_node_microstructure_summary.csv")
    write_inference_markdown(
        [
            "Node-wise live microstructure summarizes each AAL node by the mean sampled edge FA/MD connected to that node.",
            "ROI-level inference uses Kruskal-Wallis with BH-FDR across nodes within each metric family.",
            "These are connectome-edge microstructure summaries, not voxelwise TBSS or tract-specific ROI masks.",
        ],
        out_dir / "live_node_microstructure_inference.md",
    )
    return {"table": table, "tests": tests_by_metric, "summary": summary, "out_dir": out_dir}


def _global_graph_table(paths: AnalysisPaths, master: pd.DataFrame) -> pd.DataFrame:
    stack = load_connectome_stack(paths, master, connectome_type="fd_sum")
    rows = []
    for sid, group, mat in zip(stack.subject_ids, stack.groups, stack.matrices):
        w = np.asarray(mat, dtype=float)
        np.fill_diagonal(w, 0.0)
        valid = np.isfinite(w) & (w > 0)
        tri = np.triu(valid, 1)
        strengths = np.nansum(np.where(valid, w, 0.0), axis=1)
        costs = np.full_like(w, np.inf, dtype=float)
        costs[valid] = 1.0 / w[valid]
        np.fill_diagonal(costs, 0.0)
        dist = shortest_path(np.ascontiguousarray(costs), directed=False, unweighted=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / dist
        np.fill_diagonal(inv, 0.0)
        finite_dist = dist[np.triu_indices_from(dist, 1)]
        finite_dist = finite_dist[np.isfinite(finite_dist)]
        rows.append(
            {
                "subject_id": sid,
                "group": group,
                "mean_strength": float(np.nanmean(strengths)),
                "total_strength": float(np.nansum(strengths) / 2.0),
                "density": float(tri.sum() / (w.shape[0] * (w.shape[0] - 1) / 2.0)),
                "global_efficiency": float(np.nanmean(inv[np.isfinite(inv)])),
                "charpath_len": float(np.nanmean(finite_dist)) if finite_dist.size else np.nan,
            }
        )
    return pd.DataFrame(rows).merge(_master_covariates(master), on=["subject_id", "group"], how="left")


def run_live_global_graph_metrics(paths: AnalysisPaths, master: pd.DataFrame) -> dict:
    out_dir = paths.section_dir("06", "global_graph_live")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = _global_graph_table(paths, master)
    save_dataframe(table, out_dir / "live_global_graph_subject_table.csv")
    summary_rows = []
    for metric in ["mean_strength", "total_strength", "density", "global_efficiency", "charpath_len"]:
        summary_rows.append(_stat_plot_metric(table, metric, out_dir, f"{metric} by group", metric))
    summary = pd.DataFrame(summary_rows)
    save_dataframe(summary, out_dir / "live_global_graph_summary.csv")
    write_inference_markdown(
        [
            "Live graph metrics are computed directly from current SC_AAL fd_sum matrices.",
            "Costs use inverse structural weight for shortest-path metrics; disconnected paths are excluded from characteristic path length.",
            "Primary inference uses Kruskal-Wallis, Brunner-Munzel pairwise tests, Cliff's delta, and permutation ANCOVA sensitivity.",
        ],
        out_dir / "live_global_graph_inference.md",
    )
    return {"table": table, "summary": summary, "out_dir": out_dir}


def run_live_nodewise_graph_metrics(paths: AnalysisPaths, master: pd.DataFrame, top_n: int = 8) -> dict:
    out_dir = paths.section_dir("07", "node_graph_live")
    out_dir.mkdir(parents=True, exist_ok=True)
    stack = load_connectome_stack(paths, master, connectome_type="fd_sum")
    rows = []
    for sid, group, mat in zip(stack.subject_ids, stack.groups, stack.matrices):
        w = np.asarray(mat, dtype=float)
        np.fill_diagonal(w, 0.0)
        valid = np.isfinite(w) & (w > 0)
        costs = np.full_like(w, np.inf, dtype=float)
        costs[valid] = 1.0 / w[valid]
        np.fill_diagonal(costs, 0.0)
        dist = shortest_path(np.ascontiguousarray(costs), directed=False, unweighted=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / dist
        np.fill_diagonal(inv, 0.0)
        for node in range(w.shape[0]):
            finite_inv = inv[node, np.isfinite(inv[node])]
            rows.append(
                {
                    "subject_id": sid,
                    "group": group,
                    "node": node + 1,
                    "node_name": f"AAL_{node + 1:03d}",
                    "strength": float(np.nansum(np.where(valid[node], w[node], 0.0))),
                    "degree": int(valid[node].sum()),
                    "nodal_eff": float(np.nanmean(finite_inv)) if finite_inv.size else np.nan,
                }
            )
    table = pd.DataFrame(rows).merge(_master_covariates(master), on=["subject_id", "group"], how="left")
    save_dataframe(table, out_dir / "live_node_graph_long.csv")
    summaries = []
    tests_by_metric = {}
    for metric in ["strength", "degree", "nodal_eff"]:
        rows = []
        for node, sub in table.groupby("node"):
            omni = kruskal_summary(sub, metric)
            rows.append(
                {
                    "node": node,
                    "node_name": sub["node_name"].iloc[0],
                    "kw_stat": omni.statistic,
                    "kw_p": omni.pvalue,
                    "n": omni.n_total,
                }
            )
        tests = pd.DataFrame(rows).sort_values("kw_p", na_position="last")
        tests["kw_q"] = bh_fdr(tests["kw_p"].to_numpy())
        tests_by_metric[metric] = tests
        save_dataframe(tests, out_dir / f"{metric}_node_tests.csv")
        manhattan_plot(
            tests,
            x_col="node",
            y_col="kw_q",
            sig_col="kw_q",
            threshold=0.05,
            title=f"{metric}: node-wise graph effect",
            ylabel="BH-FDR q",
            out_path=out_dir / f"{metric}_node_manhattan.png",
        )
        for _, row in tests.nsmallest(top_n, "kw_q").iterrows():
            plot_df = table.loc[table["node"] == row.node, ["group", metric]]
            boxplot_with_points(
                plot_df,
                "group",
                metric,
                title=f"{metric}: {row.node_name}",
                ylabel=metric,
                out_path=out_dir / f"{metric}_node_{int(row.node):03d}.png",
                omnibus_text=f"Kruskal q={row.kw_q:.3g}",
            )
        summaries.append(
            {
                "metric": metric,
                "n_nodes_tested": len(tests),
                "n_fdr_sig": int((tests["kw_q"] < 0.05).sum()),
                "best_node": tests.iloc[0].node_name if len(tests) else np.nan,
                "best_q": tests.iloc[0].kw_q if len(tests) else np.nan,
            }
        )
    summary = pd.DataFrame(summaries)
    save_dataframe(summary, out_dir / "live_node_graph_summary.csv")
    write_inference_markdown(
        [
            "Live node graph metrics are computed directly from fd_sum structural matrices.",
            "Node-wise inference is FDR-controlled within each metric family.",
            "AAL numeric labels are used until atlas-side label names are wired into the EC2 analysis environment.",
        ],
        out_dir / "live_node_graph_inference.md",
    )
    return {"node_metrics": table, "tests": tests_by_metric, "summary": summary, "out_dir": out_dir}


def run_live_multimetric_overview(
    paths: AnalysisPaths,
    master: pd.DataFrame,
    global_micro: dict | None = None,
    node_micro: dict | None = None,
) -> dict:
    out_dir = paths.section_dir("05", "multimetric_live")
    out_dir.mkdir(parents=True, exist_ok=True)
    if global_micro is None:
        global_micro = run_live_global_microstructure(paths, master)
    if node_micro is None:
        node_micro = run_live_nodewise_microstructure(paths, master)
    global_summary = global_micro["summary"].copy()
    node_summary = node_micro["summary"].copy()
    save_dataframe(global_summary, out_dir / "live_multimetric_global_summary.csv")
    save_dataframe(node_summary, out_dir / "live_multimetric_node_summary.csv")
    if not global_summary.empty:
        heat = global_summary[["metric", "kw_p", "perm_p"]].set_index("metric")
        heatmap_matrix(
            heat.to_numpy(dtype=float),
            title="Live microstructure p-value overview",
            out_path=out_dir / "live_multimetric_pvalue_heatmap.png",
            cmap="viridis_r",
            vmin=0,
            vmax=max(0.05, np.nanmax(heat.to_numpy(dtype=float))),
        )
    write_inference_markdown(
        [
            "Live multi-metric DTI overview currently summarizes FA and MD edge-sampled connectome microstructure.",
            "Global and node-wise summaries are split so manuscript figures can distinguish whole-connectome effects from ROI-localized effects.",
            "RD and AD are intentionally not fabricated here because matching edge-sampled connectome matrices are not yet present.",
        ],
        out_dir / "live_multimetric_inference.md",
    )
    return {"global": global_summary, "local": node_summary, "summary": global_summary, "out_dir": out_dir}


def run_live_structure_microstructure_coupling(
    paths: AnalysisPaths,
    master: pd.DataFrame,
    node_graph: dict | None = None,
    node_micro: dict | None = None,
) -> dict:
    out_dir = paths.section_dir("09", "coupling_live")
    out_dir.mkdir(parents=True, exist_ok=True)
    if node_graph is None:
        node_graph = run_live_nodewise_graph_metrics(paths, master)
    if node_micro is None:
        node_micro = run_live_nodewise_microstructure(paths, master)
    graph = node_graph["node_metrics"]
    micro = node_micro["table"]
    rows = []
    for nodal_metric, dti_metric in product(["strength", "degree", "nodal_eff"], ["fa_mean", "md_mean", "ad_mean", "rd_mean"]):
        left = graph[["subject_id", "group", "age", "sex", "phase", "node", nodal_metric]].rename(
            columns={nodal_metric: "graph_value"}
        )
        right = micro.loc[micro["metric"] == dti_metric, ["subject_id", "node", "value"]].rename(
            columns={"value": "micro_value"}
        )
        merged = left.merge(right, on=["subject_id", "node"], how="inner")
        for sid, sub in merged.groupby("subject_id"):
            valid = np.isfinite(sub["graph_value"]) & np.isfinite(sub["micro_value"])
            if valid.sum() < 10 or sub.loc[valid, "graph_value"].nunique() < 2 or sub.loc[valid, "micro_value"].nunique() < 2:
                rho = pvalue = np.nan
            else:
                rho, pvalue = stats.spearmanr(sub.loc[valid, "graph_value"], sub.loc[valid, "micro_value"])
            rows.append(
                {
                    "subject_id": sid,
                    "group": sub["group"].iloc[0],
                    "age": sub["age"].iloc[0],
                    "sex": sub["sex"].iloc[0],
                    "phase": sub["phase"].iloc[0],
                    "nodal_metric": nodal_metric,
                    "micro_metric": dti_metric,
                    "rho": rho,
                    "p_subject": pvalue,
                    "n_nodes": int(valid.sum()),
                }
            )
    table = pd.DataFrame(rows)
    save_dataframe(table, out_dir / "live_subject_level_coupling.csv")
    summaries = []
    heat_rows = []
    for (nodal_metric, micro_metric), sub in table.groupby(["nodal_metric", "micro_metric"]):
        metric_name = f"{nodal_metric}_{micro_metric}_rho"
        tmp = sub.rename(columns={"rho": metric_name})
        stat = _stat_plot_metric(tmp, metric_name, out_dir, f"Coupling: {nodal_metric} vs {micro_metric}", "Spearman rho")
        stat.update({"nodal_metric": nodal_metric, "micro_metric": micro_metric})
        summaries.append(stat)
        heat_rows.append({"nodal_metric": nodal_metric, "micro_metric": micro_metric, "kw_p": stat["kw_p"]})
    summary = pd.DataFrame(summaries)
    save_dataframe(summary, out_dir / "live_coupling_summary.csv")
    heat = pd.DataFrame(heat_rows).pivot(index="nodal_metric", columns="micro_metric", values="kw_p")
    if not heat.empty:
        heatmap_matrix(
            heat.to_numpy(dtype=float),
            title="Live coupling omnibus p-values",
            out_path=out_dir / "live_coupling_pvalue_heatmap.png",
            cmap="viridis_r",
            vmin=0,
            vmax=max(0.05, np.nanmax(heat.to_numpy(dtype=float))),
        )
    focus = table.loc[(table["nodal_metric"] == "strength") & (table["micro_metric"] == "fa_mean")]
    if not focus.empty:
        scatter_with_group_fit(
            focus,
            x_col="age",
            y_col="rho",
            group_col="group",
            title="Age vs live strength-FA coupling",
            xlabel="Age",
            ylabel="Spearman rho",
            out_path=out_dir / "live_strength_fa_coupling_vs_age.png",
        )
    write_inference_markdown(
        [
            "Live coupling is subject-level Spearman association across AAL nodes between graph metrics and local edge-sampled FA/MD.",
            "This tests structure-microstructure coupling, not structural-functional decoupling.",
            "Group inference uses the same Kruskal, Brunner-Munzel, Cliff's delta, and permutation sensitivity framework.",
        ],
        out_dir / "live_coupling_inference.md",
    )
    return {"subject_level": table, "summary": summary, "out_dir": out_dir}


def run_live_brain_age(paths: AnalysisPaths, master: pd.DataFrame, n_repeats: int = 5) -> dict:
    out_dir = paths.section_dir("11", "brain_age_live")
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = _global_graph_table(paths, master)
    micro = run_live_global_microstructure(paths, master)["table"]
    feat = graph.merge(micro.drop(columns=[c for c in ["group", "age", "sex", "phase"] if c in micro.columns]), on="subject_id", how="inner")
    feat = feat.loc[feat["group"].isin(GROUP_ORDER)].dropna(subset=["age"]).copy()
    feature_cols = [
        c for c in feat.columns
        if c in {"mean_strength", "total_strength", "density", "global_efficiency", "charpath_len"}
        or c.endswith("_edge_mean")
        or c.endswith("_edge_median")
    ]
    if len(feat) < 20 or len(feature_cols) < 3:
        raise ValueError("Not enough live graph/microstructure features for brain-age modeling")
    x = feat[feature_cols].to_numpy(dtype=float)
    y = feat["age"].to_numpy(dtype=float)
    n_components = max(2, min(8, x.shape[1], len(feat) - 2))
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )
    rkf = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)
    pred_sum = np.zeros(len(feat), dtype=float)
    pred_n = np.zeros(len(feat), dtype=float)
    for train, test in rkf.split(x):
        pipe.fit(x[train], y[train])
        pred_sum[test] += pipe.predict(x[test])
        pred_n[test] += 1
    feat["brain_age_pred"] = pred_sum / np.maximum(pred_n, 1)
    feat["BAG"] = feat["brain_age_pred"] - feat["age"]
    coeffs = np.polyfit(feat["age"], feat["BAG"], deg=1)
    feat["BAG_age_corrected"] = feat["BAG"] - np.polyval(coeffs, feat["age"])
    save_dataframe(feat, out_dir / "live_brain_age_predictions.csv")

    feature_catalog = []
    corr = (
        feat[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .corr(method="spearman")
        .abs()
        if feature_cols
        else pd.DataFrame()
    )
    for col in feature_cols:
        values = pd.to_numeric(feat[col], errors="coerce")
        other = corr[col].drop(labels=[col], errors="ignore") if col in corr.columns else pd.Series(dtype=float)
        high_corr = other[other >= 0.95].sort_values(ascending=False)
        if col in {"mean_strength", "total_strength", "density", "global_efficiency", "charpath_len"}:
            group = "global graph"
            source = "06_global_graph_live/live_global_graph_subject_table.csv"
        else:
            group = "global DTI microstructure"
            source = "03_global_microstructure_live/live_global_microstructure_subject_table.csv"
        feature_catalog.append(
            {
                "feature": col,
                "feature_group": group,
                "source_file": source,
                "used_as_predictor": True,
                "model_role": "candidate predictor transformed by impute -> scale -> PCA -> ridge",
                "missingness": float(values.isna().mean()),
                "variance": float(values.var(skipna=True)) if values.notna().sum() > 1 else np.nan,
                "n_unique": int(values.nunique(dropna=True)),
                "high_corr_with": "; ".join(high_corr.index.astype(str).tolist()[:5]),
                "high_corr_abs_r": float(high_corr.iloc[0]) if not high_corr.empty else np.nan,
                "redundancy_handling": (
                    "highly correlated with another candidate; PCA compresses shared variance before ridge"
                    if not high_corr.empty
                    else "retained as candidate"
                ),
            }
        )
    for col in ["sex", "phase", "group", "subject_id"]:
        if col in feat.columns:
            feature_catalog.append(
                {
                    "feature": col,
                    "feature_group": "metadata / not predictor",
                    "source_file": "master_cohort.csv or merge key",
                    "used_as_predictor": False,
                    "model_role": "not used for brain-age prediction",
                    "missingness": float(feat[col].isna().mean()),
                    "variance": np.nan,
                    "n_unique": int(feat[col].nunique(dropna=True)),
                    "high_corr_with": "",
                    "high_corr_abs_r": np.nan,
                    "redundancy_handling": "excluded by design; age is the target and group is not used as a predictor",
                }
            )
    save_dataframe(pd.DataFrame(feature_catalog), out_dir / "live_brain_age_feature_catalog.csv")

    perf_rows = []
    for label, sub in [("Overall", feat), *[(g, feat.loc[feat["group"] == g]) for g in GROUP_ORDER]]:
        sub = sub.dropna(subset=["age", "brain_age_pred"]).copy()
        if sub.empty:
            continue
        error = sub["brain_age_pred"].to_numpy(dtype=float) - sub["age"].to_numpy(dtype=float)
        row = {
            "group": label,
            "n": int(len(sub)),
            "r2": float(r2_score(sub["age"], sub["brain_age_pred"])) if len(sub) >= 2 else np.nan,
            "mae_years": float(mean_absolute_error(sub["age"], sub["brain_age_pred"])),
            "rmse_years": float(mean_squared_error(sub["age"], sub["brain_age_pred"]) ** 0.5),
            "median_bag_years": float(np.nanmedian(error)),
            "mean_bag_years": float(np.nanmean(error)),
            "pred_min": float(np.nanmin(sub["brain_age_pred"])),
            "pred_max": float(np.nanmax(sub["brain_age_pred"])),
            "age_min": float(np.nanmin(sub["age"])),
            "age_max": float(np.nanmax(sub["age"])),
        }
        if len(sub) >= 3 and sub["age"].nunique() >= 2 and sub["brain_age_pred"].nunique() >= 2:
            pearson_r, pearson_p = stats.pearsonr(sub["age"], sub["brain_age_pred"])
            spearman_r, spearman_p = stats.spearmanr(sub["age"], sub["brain_age_pred"])
            row.update(
                {
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_rho": float(spearman_r),
                    "spearman_p": float(spearman_p),
                }
            )
        perf_rows.append(row)
    save_dataframe(pd.DataFrame(perf_rows), out_dir / "live_brain_age_model_performance.csv")
    save_dataframe(
        pd.DataFrame(
            [
                {
                    "correction": "BAG_age_corrected",
                    "input_metric": "BAG",
                    "covariate": "age",
                    "slope": float(coeffs[0]),
                    "intercept": float(coeffs[1]),
                    "formula": "BAG_age_corrected = BAG - (slope * age + intercept)",
                    "interpretation": "Residual BAG after subtracting the linear BAG component explained by chronological age.",
                }
            ]
        ),
        out_dir / "live_brain_age_age_correction.csv",
    )
    summary_rows = []
    for metric in ["BAG", "BAG_age_corrected"]:
        summary_rows.append(_stat_plot_metric(feat, metric, out_dir, f"Live {metric} by group", metric))
    summary = pd.DataFrame(summary_rows)
    save_dataframe(summary, out_dir / "live_brain_age_summary.csv")
    scatter_with_group_fit(
        feat,
        x_col="age",
        y_col="brain_age_pred",
        group_col="group",
        title="Live structural brain-age prediction",
        xlabel="Chronological age",
        ylabel="Predicted brain age",
        out_path=out_dir / "live_brain_age_pred_vs_age.png",
    )
    write_inference_markdown(
        [
            "Live brain-age uses current global graph plus connectome-sampled FA/MD/RD/AD edge mean and median features.",
            "Predictions are repeated cross-validated PCA plus ridge regression models; BAG is predicted age minus chronological age.",
            "Feature catalog, model performance, and the BAG age-correction equation are exported as live CSV audit tables.",
            "This is a provisional structural brain-age model and should be rerun after final post/connectome QC lock.",
        ],
        out_dir / "live_brain_age_inference.md",
    )
    return {"features": feat, "summary": summary, "out_dir": out_dir}
