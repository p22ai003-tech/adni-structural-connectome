from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

from connectome_analysis.analysis_config import GROUP_ORDER, AnalysisPaths
from connectome_analysis.analysis_plots import boxplot_with_points, heatmap_matrix, save_dataframe, scatter_with_group_fit, write_inference_markdown
from connectome_analysis.analysis_stats import group_descriptives, kruskal_summary, pairwise_robust_tests, permutation_group_effect


def _merge_master(df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ["subject_id", "group", "age", "sex", "phase"] if c in master.columns]
    out = df.merge(master[keep].drop_duplicates("subject_id"), on="subject_id", how="left", suffixes=("", "_master"))
    if "group_master" in out.columns:
        out["group"] = out["group"].fillna(out["group_master"])
        out = out.drop(columns=["group_master"])
    return out


def _subject_level_coupling(
    node_df: pd.DataFrame,
    local_df: pd.DataFrame,
    nodal_metric: str,
    dti_metric: str,
) -> pd.DataFrame:
    left = node_df[["subject_id", "group", "age", "sex", "phase", "node", "node_name", nodal_metric]].rename(
        columns={nodal_metric: "graph_value"}
    )
    right = local_df.loc[local_df["metric"] == dti_metric, ["subject_id", "roi", "value"]].rename(
        columns={"roi": "node", "value": "dti_value"}
    )
    merged = left.merge(right, on=["subject_id", "node"], how="inner")
    rows = []
    for sid, sub in merged.groupby("subject_id"):
        x = pd.to_numeric(sub["graph_value"], errors="coerce")
        y = pd.to_numeric(sub["dti_value"], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask].to_numpy()
        y = y[mask].to_numpy()
        if len(x) < 10 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
            rho, p = np.nan, np.nan
        else:
            rho, p = stats.spearmanr(x, y)
        rows.append(
            {
                "subject_id": sid,
                "group": sub["group"].iloc[0],
                "age": sub["age"].iloc[0],
                "sex": sub["sex"].iloc[0],
                "phase": sub["phase"].iloc[0],
                "nodal_metric": nodal_metric,
                "dti_metric": dti_metric,
                "rho": rho,
                "p_subject": p,
                "n_nodes": int(len(x)),
            }
        )
    return pd.DataFrame(rows)


def run_coupling_analysis(paths: AnalysisPaths, master: pd.DataFrame) -> dict:
    out_dir = paths.section_dir("09", "coupling")
    out_dir.mkdir(parents=True, exist_ok=True)
    node_df = pd.read_csv(paths.legacy_analysis_dir / "analysis3_network" / "node_metrics_long_ALL.csv")
    local_df = pd.read_csv(paths.legacy_analysis_dir / "analysis3_network" / "local_dti_medians_long_CLEAN.csv")
    legacy_node = pd.read_csv(paths.legacy_analysis_dir / "analysis3_network" / "node_microstruct_coupling.csv")
    node_df = _merge_master(node_df, master)
    local_df = _merge_master(local_df, master)

    subject_rows = []
    for nodal_metric, dti_metric in product(["strength", "degree", "nodal_eff"], ["rd", "fa", "md", "ad"]):
        subject_rows.append(_subject_level_coupling(node_df, local_df, nodal_metric, dti_metric))
    subject_coupling = pd.concat(subject_rows, ignore_index=True)
    save_dataframe(subject_coupling, out_dir / "subject_level_coupling.csv")
    save_dataframe(legacy_node, out_dir / "legacy_node_coupling.csv")

    summary_rows = []
    heat_rows = []
    for (nodal_metric, dti_metric), sub in subject_coupling.groupby(["nodal_metric", "dti_metric"]):
        desc = group_descriptives(sub, "rho")
        desc["nodal_metric"] = nodal_metric
        desc["dti_metric"] = dti_metric
        pairwise = pairwise_robust_tests(sub, "rho")
        pairwise["nodal_metric"] = nodal_metric
        pairwise["dti_metric"] = dti_metric
        omni = kruskal_summary(sub, "rho")
        perm = permutation_group_effect(sub, "rho")
        save_dataframe(desc, out_dir / f"coupling_{nodal_metric}_{dti_metric}_descriptives.csv")
        save_dataframe(pairwise, out_dir / f"coupling_{nodal_metric}_{dti_metric}_pairwise.csv")
        pair_lines = []
        for _, row in pairwise.sort_values("bm_p").head(3).iterrows():
            pair_lines.append(
                f"{row.group_a} vs {row.group_b}: BM q={row.bm_q:.3g}, Cliff's d={row.cliffs_delta:.3g}"
            )
        boxplot_with_points(
            sub,
            "group",
            "rho",
            title=f"Subject-level coupling: {nodal_metric} vs {dti_metric}",
            ylabel="Spearman rho across nodes",
            out_path=out_dir / f"coupling_{nodal_metric}_{dti_metric}_boxplot.png",
            omnibus_text=f"Kruskal p={omni.pvalue:.3g}; perm p={perm['p_perm']:.3g}",
            pairwise_lines=pair_lines,
        )
        summary_rows.append(
            {
                "nodal_metric": nodal_metric,
                "dti_metric": dti_metric,
                "n_subjects": omni.n_total,
                "kw_p": omni.pvalue,
                "perm_p": perm["p_perm"],
            }
        )
        heat_rows.append({"nodal_metric": nodal_metric, "dti_metric": dti_metric, "kw_p": omni.pvalue})

    summary = pd.DataFrame(summary_rows)
    save_dataframe(summary, out_dir / "coupling_summary.csv")
    heat_df = pd.DataFrame(heat_rows).pivot(index="nodal_metric", columns="dti_metric", values="kw_p")
    heatmap_matrix(
        heat_df.to_numpy(dtype=float),
        title="Coupling omnibus p-values",
        out_path=out_dir / "coupling_kw_heatmap.png",
        cmap="viridis_r",
        vmin=0.0,
        vmax=max(0.05, np.nanmax(heat_df.to_numpy(dtype=float))),
    )

    # Clinical linkage: coupling vs age as a default covariate narrative plot.
    age_focus = subject_coupling.loc[
        (subject_coupling["nodal_metric"] == "strength")
        & (subject_coupling["dti_metric"] == "rd")
    ].copy()
    if not age_focus.empty:
        scatter_with_group_fit(
            age_focus,
            x_col="age",
            y_col="rho",
            group_col="group",
            title="Age vs coupling: strength ~ RD",
            xlabel="Age",
            ylabel="Spearman rho",
            out_path=out_dir / "coupling_strength_rd_vs_age.png",
            annotation_lines=["Exploratory age trend shown by group; formal adjustment uses permutation ANCOVA."],
        )

    write_inference_markdown(
        [
            "Coupling is summarized at the subject level as the across-node Spearman association between nodal graph metrics and local DTI metrics.",
            "This yields a direct between-group test of whether microstructure–network coupling weakens or strengthens across disease stages.",
            "Legacy node-wise coupling outputs are retained and exported alongside the new subject-level summary.",
        ],
        out_dir / "coupling_inference.md",
    )
    return {
        "subject_level": subject_coupling,
        "legacy_node": legacy_node,
        "summary": summary,
        "out_dir": out_dir,
    }
