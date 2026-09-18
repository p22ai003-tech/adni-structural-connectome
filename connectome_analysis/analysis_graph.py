from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from connectome_analysis.analysis_config import GROUP_ORDER, AnalysisPaths
from connectome_analysis.analysis_plots import boxplot_with_points, manhattan_plot, save_dataframe, write_inference_markdown
from connectome_analysis.analysis_stats import bh_fdr, group_descriptives, kruskal_summary, pairwise_robust_tests, permutation_group_effect


def _legacy_file(paths: AnalysisPaths, *parts: str) -> Path:
    return paths.legacy_analysis_dir.joinpath(*parts)


def _merge_master(df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    add_cols = [c for c in ["subject_id", "group", "age", "sex", "phase", "apoe_e4_count"] if c in master.columns]
    base = master[add_cols].drop_duplicates("subject_id")
    out = df.merge(base, on="subject_id", how="left", suffixes=("", "_master"))
    if "group_master" in out.columns:
        out["group"] = out["group"].fillna(out["group_master"])
        out = out.drop(columns=["group_master"])
    return out


def run_connectome_qc(paths: AnalysisPaths, master: pd.DataFrame) -> dict:
    out_dir = paths.section_dir("05", "connectome_qc")
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = paths.master_dir / "subject_connectome_inventory.csv"
    if inventory_path.exists():
        inventory = pd.read_csv(inventory_path)
    else:
        from connectome_analysis.analysis_cohort import connectome_inventory

        inventory = connectome_inventory(paths)
    data = master[["subject_id", "group"]].drop_duplicates().merge(inventory, on="subject_id", how="left")
    availability_cols = [c for c in data.columns if c.startswith("has_conn_")]
    summary_rows = []
    for col in availability_cols:
        weight = col.replace("has_conn_", "")
        counts = data.groupby("group")[col].sum(min_count=1).reindex(GROUP_ORDER)
        summary_rows.append({"connectome_type": weight, **counts.to_dict()})
    summary = pd.DataFrame(summary_rows).fillna(0)
    save_dataframe(summary, out_dir / "connectome_availability_by_group.csv")
    save_dataframe(data, out_dir / "subject_connectome_availability.csv")
    write_inference_markdown(
        [
            "Connectome availability is summarized before downstream inference so later group effects can be interpreted against effective sample size.",
            "Any connectome type with strong group imbalance should be treated as sensitivity-only.",
        ],
        out_dir / "connectome_qc_inference.md",
    )
    return {"subject_table": data, "summary": summary, "out_dir": out_dir}


def _global_graph_table(paths: AnalysisPaths) -> pd.DataFrame:
    table = pd.read_csv(_legacy_file(paths, "analysis1_rd", "analysis1_table.csv"))
    return table.rename(columns={"rd_median": "global_rd"})


def run_global_graph_metrics(paths: AnalysisPaths, master: pd.DataFrame) -> dict:
    out_dir = paths.section_dir("06", "global_graph")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = _merge_master(_global_graph_table(paths), master)
    metrics = {
        "mean_strength": "Mean strength",
        "global_efficiency": "Global efficiency",
        "charpath_len": "Characteristic path length",
    }
    results = []
    for metric, label in metrics.items():
        sub = table[["subject_id", "group", "age", "sex", "phase", metric]].dropna()
        desc = group_descriptives(sub, metric)
        omni = kruskal_summary(sub, metric)
        pairwise = pairwise_robust_tests(sub, metric)
        perm = permutation_group_effect(sub, metric)
        desc["metric"] = metric
        pairwise["metric"] = metric
        results.append(
            {
                "metric": metric,
                "n": omni.n_total,
                "kw_stat": omni.statistic,
                "kw_p": omni.pvalue,
                "perm_f": perm["f_stat"],
                "perm_p": perm["p_perm"],
            }
        )
        save_dataframe(desc, out_dir / f"{metric}_descriptives.csv")
        save_dataframe(pairwise, out_dir / f"{metric}_pairwise.csv")
        pair_lines = []
        if not pairwise.empty:
            sig = pairwise.sort_values("bm_p").head(3)
            for _, row in sig.iterrows():
                pair_lines.append(
                    f"{row.group_a} vs {row.group_b}: BM q={row.bm_q:.3g}, Cliff's d={row.cliffs_delta:.3g}"
                )
        omnibus_text = f"Kruskal p={omni.pvalue:.3g}; perm p={perm['p_perm']:.3g}"
        boxplot_with_points(
            sub,
            "group",
            metric,
            title=f"{label} by group",
            ylabel=label,
            out_path=out_dir / f"{metric}_boxplot.png",
            omnibus_text=omnibus_text,
            pairwise_lines=pair_lines,
        )
        write_inference_markdown(
            [
                f"{label} is compared across CN, MCI, and AD with Kruskal-Wallis as the primary omnibus test.",
                f"Covariate-adjusted sensitivity uses permutation ANCOVA with age, sex, and phase.",
                "Pairwise results are reported with Brunner-Munzel q-values and Cliff's delta.",
            ],
            out_dir / f"{metric}_inference.md",
        )
    summary = pd.DataFrame(results)
    save_dataframe(summary, out_dir / "global_graph_summary.csv")
    return {"table": table, "summary": summary, "out_dir": out_dir}


def _node_test_table(df: pd.DataFrame, value_col: str, id_col: str = "node") -> pd.DataFrame:
    rows = []
    for node, sub in df.groupby(id_col):
        omni = kruskal_summary(sub, value_col)
        rows.append(
            {
                id_col: node,
                "node_name": sub["node_name"].iloc[0] if "node_name" in sub.columns else f"ROI_{node}",
                "kw_stat": omni.statistic,
                "kw_p": omni.pvalue,
                "n": omni.n_total,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["kw_q"] = bh_fdr(out["kw_p"].to_numpy())
    return out.sort_values("kw_p", na_position="last")


def run_nodewise_metrics(paths: AnalysisPaths, master: pd.DataFrame, top_n: int = 8) -> dict:
    out_dir = paths.section_dir("07", "node_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    node_metrics = pd.read_csv(_legacy_file(paths, "analysis3_network", "node_metrics_long_ALL.csv"))
    node_metrics = _merge_master(node_metrics, master)
    node_metrics["node_name"] = node_metrics["node_name"].fillna(node_metrics["node"].map(lambda n: f"AAL_{int(n)}"))
    test_tables = {}
    for metric in ["strength", "degree", "nodal_eff"]:
        tests = _node_test_table(node_metrics[["subject_id", "group", "node", "node_name", metric]].dropna(), metric)
        save_dataframe(tests, out_dir / f"{metric}_node_tests.csv")
        manhattan_plot(
            tests,
            x_col="node",
            y_col="kw_q",
            sig_col="kw_q",
            threshold=0.05,
            title=f"{metric} node-wise FDR profile",
            ylabel="BH-FDR q",
            out_path=out_dir / f"{metric}_node_manhattan.png",
        )
        top = tests.nsmallest(top_n, "kw_q")
        for _, row in top.iterrows():
            sub = node_metrics.loc[node_metrics["node"] == row["node"], ["group", metric]].dropna()
            boxplot_with_points(
                sub,
                "group",
                metric,
                title=f"{metric}: {row['node_name']}",
                ylabel=metric,
                out_path=out_dir / f"{metric}_node_{int(row['node']):03d}.png",
                omnibus_text=f"Kruskal q={row['kw_q']:.3g}",
            )
        test_tables[metric] = tests
    write_inference_markdown(
        [
            "Node-wise metrics are FDR-controlled across ROIs within each metric family.",
            "Top ROIs are shown with group boxplots; the full node table is exported for manuscript use.",
        ],
        out_dir / "node_metrics_inference.md",
    )
    return {"node_metrics": node_metrics, "tests": test_tables, "out_dir": out_dir}


def run_centrality_review(paths: AnalysisPaths, master: pd.DataFrame, top_n: int = 8) -> dict:
    out_dir = paths.section_dir("08", "centrality")
    out_dir.mkdir(parents=True, exist_ok=True)
    centrality_long = pd.read_csv(
        _legacy_file(paths, "analysis_centrality_length", "centralities_long_fd_sum.csv")
    )
    centrality_global = pd.read_csv(
        _legacy_file(paths, "analysis_centrality_length", "centralities_global_fd_sum.csv")
    )
    centrality_long = _merge_master(centrality_long, master)
    centrality_global = _merge_master(centrality_global, master)
    centrality_long["node_name"] = centrality_long["node"].map(lambda n: f"AAL_{int(n)}")
    summary_rows = []
    for metric, sub_metric in centrality_global.groupby("metric"):
        desc = group_descriptives(sub_metric, "value")
        pairwise = pairwise_robust_tests(sub_metric, "value")
        omni = kruskal_summary(sub_metric, "value")
        save_dataframe(desc, out_dir / f"centrality_global_{metric}_descriptives.csv")
        save_dataframe(pairwise, out_dir / f"centrality_global_{metric}_pairwise.csv")
        pair_lines = []
        for _, row in pairwise.sort_values("bm_p").head(3).iterrows():
            pair_lines.append(
                f"{row.group_a} vs {row.group_b}: BM q={row.bm_q:.3g}, Cliff's d={row.cliffs_delta:.3g}"
            )
        boxplot_with_points(
            sub_metric,
            "group",
            "value",
            title=f"Global centrality: {metric}",
            ylabel=metric,
            out_path=out_dir / f"centrality_global_{metric}.png",
            omnibus_text=f"Kruskal p={omni.pvalue:.3g}",
            pairwise_lines=pair_lines,
        )
        summary_rows.append(
            {
                "metric": f"global_{metric}",
                "n_nodes_tested": int(sub_metric["subject_id"].nunique()),
                "n_fdr_sig": np.nan,
            }
        )
    for metric, sub_metric in centrality_long.groupby("metric"):
        tests = _node_test_table(sub_metric[["subject_id", "group", "node", "node_name", "value"]], "value")
        tests["metric"] = metric
        save_dataframe(tests, out_dir / f"centrality_{metric}_tests.csv")
        manhattan_plot(
            tests,
            x_col="node",
            y_col="kw_q",
            sig_col="kw_q",
            threshold=0.05,
            title=f"{metric} centrality FDR profile",
            ylabel="BH-FDR q",
            out_path=out_dir / f"centrality_{metric}_manhattan.png",
        )
        top = tests.nsmallest(top_n, "kw_q")
        for _, row in top.iterrows():
            plot_df = sub_metric.loc[sub_metric["node"] == row["node"], ["group", "value"]]
            boxplot_with_points(
                plot_df,
                "group",
                "value",
                title=f"{metric}: {row['node_name']}",
                ylabel=metric,
                out_path=out_dir / f"centrality_{metric}_node_{int(row['node']):03d}.png",
                omnibus_text=f"Kruskal q={row['kw_q']:.3g}",
            )
        summary_rows.append(
            {
                "metric": metric,
                "n_nodes_tested": int(len(tests)),
                "n_fdr_sig": int((tests["kw_q"] < 0.05).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    save_dataframe(summary, out_dir / "centrality_summary.csv")
    write_inference_markdown(
        [
            "Centrality complements the default graph metrics by highlighting node-specific hub changes.",
            "ROI-wise inference is FDR-controlled within each centrality family.",
        ],
        out_dir / "centrality_inference.md",
    )
    return {"centrality": centrality_long, "summary": summary, "out_dir": out_dir}
