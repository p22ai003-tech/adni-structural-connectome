from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from connectome_analysis.analysis_config import GROUP_ORDER, AnalysisPaths
from connectome_analysis.analysis_plots import boxplot_with_points, manhattan_plot, save_dataframe, write_inference_markdown
from connectome_analysis.analysis_stats import bh_fdr, group_descriptives, kruskal_summary, pairwise_robust_tests, permutation_group_effect


def _merge_master(df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ["subject_id", "group", "age", "sex", "phase"] if c in master.columns]
    out = df.merge(master[keep].drop_duplicates("subject_id"), on="subject_id", how="left", suffixes=("", "_master"))
    if "group_master" in out.columns:
        out["group"] = out["group"].fillna(out["group_master"])
        out = out.drop(columns=["group_master"])
    return out


def run_global_rd_analysis(paths: AnalysisPaths, master: pd.DataFrame) -> dict:
    out_dir = paths.section_dir("03", "global_rd")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(paths.legacy_analysis_dir / "analysis1_rd" / "analysis1_table.csv")
    table = _merge_master(table, master)
    desc = group_descriptives(table, "rd_median")
    omni = kruskal_summary(table, "rd_median")
    pairwise = pairwise_robust_tests(table, "rd_median")
    perm = permutation_group_effect(table, "rd_median")
    save_dataframe(table, out_dir / "global_rd_subject_table.csv")
    save_dataframe(desc, out_dir / "global_rd_descriptives.csv")
    save_dataframe(pairwise, out_dir / "global_rd_pairwise.csv")
    pair_lines = []
    for _, row in pairwise.sort_values("bm_p").head(3).iterrows():
        pair_lines.append(
            f"{row.group_a} vs {row.group_b}: BM q={row.bm_q:.3g}, Cliff's d={row.cliffs_delta:.3g}"
        )
    boxplot_with_points(
        table,
        "group",
        "rd_median",
        title="Global RD by diagnostic group",
        ylabel="RD median",
        out_path=out_dir / "global_rd_boxplot.png",
        omnibus_text=f"Kruskal p={omni.pvalue:.3g}; perm p={perm['p_perm']:.3g}",
        pairwise_lines=pair_lines,
    )
    write_inference_markdown(
        [
            "Global RD remains the lead white-matter microstructure marker in the notebook because it was part of the original analysis.",
            "Kruskal-Wallis is the primary omnibus test; permutation ANCOVA adjusts for age, sex, and phase.",
            "Pairwise contrasts are reported with Brunner-Munzel q-values and Cliff's delta.",
        ],
        out_dir / "global_rd_inference.md",
    )
    summary = pd.DataFrame(
        [
            {
                "metric": "rd_median",
                "n": omni.n_total,
                "kw_stat": omni.statistic,
                "kw_p": omni.pvalue,
                "perm_f": perm["f_stat"],
                "perm_p": perm["p_perm"],
            }
        ]
    )
    save_dataframe(summary, out_dir / "global_rd_summary.csv")
    return {"table": table, "summary": summary, "out_dir": out_dir}


def _roi_kruskal_table(df: pd.DataFrame, value_col: str, roi_col: str = "roi") -> pd.DataFrame:
    rows = []
    for roi, sub in df.groupby(roi_col):
        omni = kruskal_summary(sub, value_col)
        rows.append(
            {
                roi_col: roi,
                "roi_name": sub["roi_name"].iloc[0] if "roi_name" in sub.columns else f"AAL_{int(roi)}",
                "kw_stat": omni.statistic,
                "kw_p": omni.pvalue,
                "n": omni.n_total,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["kw_q"] = bh_fdr(out["kw_p"].to_numpy())
    return out.sort_values("kw_p", na_position="last")


def run_local_rd_analysis(paths: AnalysisPaths, master: pd.DataFrame, top_n: int = 10) -> dict:
    out_dir = paths.section_dir("04", "local_rd")
    out_dir.mkdir(parents=True, exist_ok=True)
    local = pd.read_csv(
        paths.legacy_analysis_dir / "analysis1b_rd_local" / "rd_parcel_medians_long_with_flags.csv"
    )
    local = local.loc[~local["flag_rd_range"].fillna(False)].copy()
    local = _merge_master(local, master)
    tests = _roi_kruskal_table(local, "rd_median")
    save_dataframe(local, out_dir / "local_rd_subject_table.csv")
    save_dataframe(tests, out_dir / "local_rd_roi_tests.csv")
    manhattan_plot(
        tests,
        x_col="roi",
        y_col="kw_q",
        sig_col="kw_q",
        threshold=0.05,
        title="Local RD ROI-wise FDR profile",
        ylabel="BH-FDR q",
        out_path=out_dir / "local_rd_manhattan.png",
    )
    for _, row in tests.nsmallest(top_n, "kw_q").iterrows():
        sub = local.loc[local["roi"] == row["roi"], ["group", "rd_median"]]
        boxplot_with_points(
            sub,
            "group",
            "rd_median",
            title=f"Local RD: {row['roi_name']}",
            ylabel="RD median",
            out_path=out_dir / f"local_rd_roi_{int(row['roi']):03d}.png",
            omnibus_text=f"Kruskal q={row['kw_q']:.3g}",
        )
    write_inference_markdown(
        [
            "ROI-wise RD is controlled for multiplicity with BH-FDR across parcels.",
            "Only non-flagged local RD values are included in the professor-facing review.",
        ],
        out_dir / "local_rd_inference.md",
    )
    return {"table": local, "tests": tests, "out_dir": out_dir}


def run_multimetric_dti_analysis(paths: AnalysisPaths, master: pd.DataFrame, top_n: int = 8) -> dict:
    out_dir = paths.section_dir("05", "multimetric_dti")
    out_dir.mkdir(parents=True, exist_ok=True)
    global_df = pd.read_csv(
        paths.legacy_analysis_dir / "analysis2_multimetric_threaded" / "global_medians_long_CLEAN.csv"
    )
    local_df = pd.read_csv(
        paths.legacy_analysis_dir / "analysis2_multimetric_threaded" / "local_medians_long_CLEAN.csv"
    )
    global_df = _merge_master(global_df, master)
    local_df = _merge_master(local_df, master)

    global_rows = []
    for metric, sub in global_df.groupby("metric"):
        desc = group_descriptives(sub, "value")
        desc["metric"] = metric
        pairwise = pairwise_robust_tests(sub, "value")
        pairwise["metric"] = metric
        omni = kruskal_summary(sub, "value")
        perm = permutation_group_effect(sub, "value")
        global_rows.append(
            {
                "metric": metric,
                "n": omni.n_total,
                "kw_stat": omni.statistic,
                "kw_p": omni.pvalue,
                "perm_f": perm["f_stat"],
                "perm_p": perm["p_perm"],
            }
        )
        save_dataframe(desc, out_dir / f"global_{metric}_descriptives.csv")
        save_dataframe(pairwise, out_dir / f"global_{metric}_pairwise.csv")
        pair_lines = []
        for _, row in pairwise.sort_values("bm_p").head(3).iterrows():
            pair_lines.append(
                f"{row.group_a} vs {row.group_b}: BM q={row.bm_q:.3g}, Cliff's d={row.cliffs_delta:.3g}"
            )
        boxplot_with_points(
            sub,
            "group",
            "value",
            title=f"Global {metric.upper()} by group",
            ylabel=metric.upper(),
            out_path=out_dir / f"global_{metric}_boxplot.png",
            omnibus_text=f"Kruskal p={omni.pvalue:.3g}; perm p={perm['p_perm']:.3g}",
            pairwise_lines=pair_lines,
        )
    global_summary = pd.DataFrame(global_rows)
    save_dataframe(global_summary, out_dir / "global_multimetric_summary.csv")

    local_tables = []
    for metric, sub_metric in local_df.groupby("metric"):
        tests = _roi_kruskal_table(sub_metric.rename(columns={"value": "metric_value"}), "metric_value")
        tests["metric"] = metric
        save_dataframe(tests, out_dir / f"local_{metric}_roi_tests.csv")
        manhattan_plot(
            tests,
            x_col="roi",
            y_col="kw_q",
            sig_col="kw_q",
            threshold=0.05,
            title=f"{metric.upper()} ROI-wise FDR profile",
            ylabel="BH-FDR q",
            out_path=out_dir / f"local_{metric}_manhattan.png",
        )
        for _, row in tests.nsmallest(top_n, "kw_q").iterrows():
            plot_df = sub_metric.loc[sub_metric["roi"] == row["roi"], ["group", "value"]]
            boxplot_with_points(
                plot_df,
                "group",
                "value",
                title=f"{metric.upper()}: {row['roi_name']}",
                ylabel=metric.upper(),
                out_path=out_dir / f"local_{metric}_roi_{int(row['roi']):03d}.png",
                omnibus_text=f"Kruskal q={row['kw_q']:.3g}",
            )
        local_tables.append(tests)
    local_summary = pd.concat(local_tables, ignore_index=True) if local_tables else pd.DataFrame()
    save_dataframe(local_summary, out_dir / "local_multimetric_summary.csv")
    write_inference_markdown(
        [
            "Multi-metric DTI is summarized from the cleaned threaded outputs already generated in the legacy notebook.",
            "Global plots are paired with robust pairwise tests and permutation-adjusted sensitivity analyses.",
            "Local ROI results are FDR-controlled within each metric family.",
        ],
        out_dir / "multimetric_inference.md",
    )
    return {"global": global_df, "local": local_df, "out_dir": out_dir}
