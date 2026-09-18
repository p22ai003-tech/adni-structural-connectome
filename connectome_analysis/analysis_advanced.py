from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from connectome_analysis.analysis_config import AnalysisPaths
from connectome_analysis.analysis_edges import load_connectome_stack
from connectome_analysis.analysis_plots import boxplot_with_points, manhattan_plot, save_dataframe, write_inference_markdown
from connectome_analysis.analysis_stats import bh_fdr, group_descriptives, kruskal_summary, pairwise_robust_tests, permutation_group_effect


def _aligned_stacks(paths: AnalysisPaths, master: pd.DataFrame, weight_type: str, length_type: str):
    weight = load_connectome_stack(paths, master, connectome_type=weight_type)
    length = load_connectome_stack(paths, master, connectome_type=length_type)
    length_idx = {sid: i for i, sid in enumerate(length.subject_ids)}
    rows = []
    weight_mats = []
    length_mats = []
    groups = []
    for i, sid in enumerate(weight.subject_ids):
        j = length_idx.get(sid)
        if j is None:
            continue
        weight_mats.append(weight.matrices[i])
        length_mats.append(length.matrices[j])
        groups.append(weight.groups[i])
        rows.append(sid)
    if not rows:
        raise FileNotFoundError(f"No aligned {weight_type}/{length_type} connectome pairs found")
    return rows, groups, np.stack(weight_mats, axis=0), np.stack(length_mats, axis=0)


def _cn_edr_model(groups: list[str], weights: np.ndarray, lengths: np.ndarray) -> dict:
    cn_mask = np.asarray(groups) == "CN"
    if cn_mask.sum() < 5:
        cn_mask = np.ones(len(groups), dtype=bool)
    iu = np.triu_indices(weights.shape[1], 1)
    x = np.log(np.clip(lengths[cn_mask][:, iu[0], iu[1]].ravel(), 1e-6, None))
    y = np.log1p(np.clip(weights[cn_mask][:, iu[0], iu[1]].ravel(), 0, None))
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if valid.sum() < 100:
        raise ValueError("Not enough CN edge observations to fit an edge-distance relationship")
    slope, intercept = np.polyfit(x[valid], y[valid], deg=1)
    pred = intercept + slope * np.log(np.clip(lengths[:, iu[0], iu[1]], 1e-6, None))
    obs = np.log1p(np.clip(weights[:, iu[0], iu[1]], 0, None))
    residuals = obs - pred
    cn_lengths = lengths[cn_mask][:, iu[0], iu[1]].ravel()
    cn_lengths = cn_lengths[np.isfinite(cn_lengths) & (cn_lengths > 0)]
    q33, q67 = np.nanpercentile(cn_lengths, [33.333, 66.667])
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "iu": iu,
        "residuals": residuals,
        "short_max_mm": float(q33),
        "medium_max_mm": float(q67),
    }


def _subject_edr_table(
    subject_ids: list[str],
    groups: list[str],
    lengths: np.ndarray,
    residuals: np.ndarray,
    model: dict,
) -> pd.DataFrame:
    iu = model["iu"]
    edge_lengths = lengths[:, iu[0], iu[1]]
    rows = []
    for idx, sid in enumerate(subject_ids):
        resid = residuals[idx]
        length = edge_lengths[idx]
        valid = np.isfinite(resid) & np.isfinite(length) & (length > 0)
        short = valid & (length <= model["short_max_mm"])
        medium = valid & (length > model["short_max_mm"]) & (length <= model["medium_max_mm"])
        long = valid & (length > model["medium_max_mm"])
        short_mean = float(np.nanmean(resid[short])) if short.any() else np.nan
        long_mean = float(np.nanmean(resid[long])) if long.any() else np.nan
        rows.append(
            {
                "subject_id": sid,
                "group": groups[idx],
                "edr_residual_mean": float(np.nanmean(resid[valid])) if valid.any() else np.nan,
                "edr_residual_short": short_mean,
                "edr_residual_medium": float(np.nanmean(resid[medium])) if medium.any() else np.nan,
                "edr_residual_long": long_mean,
                "structural_compensation_index": (
                    short_mean - long_mean if np.isfinite(short_mean) and np.isfinite(long_mean) else np.nan
                ),
                "long_range_vulnerability": -long_mean if np.isfinite(long_mean) else np.nan,
                "short_range_preservation": short_mean,
                "n_edges_valid": int(valid.sum()),
                "n_edges_short": int(short.sum()),
                "n_edges_long": int(long.sum()),
            }
        )
    return pd.DataFrame(rows)


def _edge_gradient_table(groups: list[str], residuals: np.ndarray, model: dict) -> pd.DataFrame:
    group_score = pd.Series(groups).map({"CN": 0.0, "MCI": 1.0, "AD": 2.0}).to_numpy(dtype=float)
    iu = model["iu"]
    rows = []
    for edge_idx in range(residuals.shape[1]):
        vals = residuals[:, edge_idx]
        valid = np.isfinite(vals) & np.isfinite(group_score)
        if (
            valid.sum() < 8
            or len(np.unique(group_score[valid])) < 2
            or pd.Series(vals[valid]).nunique(dropna=True) < 2
        ):
            rho = pvalue = np.nan
        else:
            rho, pvalue = stats.spearmanr(group_score[valid], vals[valid])
        rows.append(
            {
                "i": int(iu[0][edge_idx] + 1),
                "j": int(iu[1][edge_idx] + 1),
                "disease_gradient_rho": float(rho) if np.isfinite(rho) else np.nan,
                "disease_gradient_p": float(pvalue) if np.isfinite(pvalue) else np.nan,
                "n": int(valid.sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["disease_gradient_q"] = bh_fdr(out["disease_gradient_p"].to_numpy())
    return out.sort_values("disease_gradient_p", na_position="last")


def run_structural_innovation_analysis(
    paths: AnalysisPaths,
    master: pd.DataFrame,
    weight_type: str = "fd_sum",
    length_type: str = "len_mean",
    top_n: int = 12,
) -> dict:
    out_dir = paths.section_dir("15", "advanced_structural")
    out_dir.mkdir(parents=True, exist_ok=True)
    subject_ids, groups, weights, lengths = _aligned_stacks(paths, master, weight_type, length_type)
    model = _cn_edr_model(groups, weights, lengths)
    subject_table = _subject_edr_table(subject_ids, groups, lengths, model["residuals"], model)
    subject_table = subject_table.merge(
        master[[c for c in ["subject_id", "age", "sex", "phase"] if c in master.columns]].drop_duplicates("subject_id"),
        on="subject_id",
        how="left",
    )
    edge_table = _edge_gradient_table(groups, model["residuals"], model)
    save_dataframe(subject_table, out_dir / f"edr_subject_level_{weight_type}_{length_type}.csv")
    save_dataframe(edge_table, out_dir / f"edr_edge_disease_gradient_{weight_type}_{length_type}.csv")
    pd.DataFrame(
        [
            {
                "weight_type": weight_type,
                "length_type": length_type,
                "edr_intercept": model["intercept"],
                "edr_slope": model["slope"],
                "short_max_mm": model["short_max_mm"],
                "medium_max_mm": model["medium_max_mm"],
                "n_subjects": len(subject_table),
            }
        ]
    ).to_csv(out_dir / "edr_model_parameters.csv", index=False)

    summary_rows = []
    focus_metrics = [
        "edr_residual_mean",
        "edr_residual_short",
        "edr_residual_long",
        "structural_compensation_index",
        "long_range_vulnerability",
    ]
    for metric in focus_metrics:
        sub = subject_table[["subject_id", "group", "age", "sex", "phase", metric]].dropna(subset=[metric])
        if sub.empty:
            continue
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
            title=f"{metric} by group",
            ylabel=metric,
            out_path=out_dir / f"{metric}_boxplot.png",
            omnibus_text=f"Kruskal p={omni.pvalue:.3g}; perm p={perm['p_perm']:.3g}",
            pairwise_lines=pair_lines,
        )
        summary_rows.append(
            {
                "metric": metric,
                "n": omni.n_total,
                "kw_stat": omni.statistic,
                "kw_p": omni.pvalue,
                "perm_f": perm["f_stat"],
                "perm_p": perm["p_perm"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    save_dataframe(summary, out_dir / "advanced_structural_summary.csv")
    top_edges = edge_table.nsmallest(top_n, "disease_gradient_p")
    save_dataframe(top_edges, out_dir / "top_disease_gradient_edges.csv")
    manhattan_plot(
        edge_table.reset_index(drop=True).assign(edge=np.arange(1, len(edge_table) + 1)),
        x_col="edge",
        y_col="disease_gradient_q",
        sig_col="disease_gradient_q",
        threshold=0.05,
        title="EDR residual disease-gradient FDR profile",
        ylabel="BH-FDR q",
        out_path=out_dir / "edr_disease_gradient_manhattan.png",
    )
    write_inference_markdown(
        [
            "The EDR model fits the CN edge-distance relationship, then tests whether subjects deviate from that reference.",
            "The structural compensation index is short-range residual preservation minus long-range residual preservation.",
            "The edge disease-gradient table ranks edges whose CN-to-MCI-to-AD residuals monotonically decline or increase.",
            "This module is structural only; it does not claim functional decoupling until real FC, EEG, or MEG matrices are supplied.",
        ],
        out_dir / "advanced_structural_inference.md",
    )
    return {
        "subject_level": subject_table,
        "edge_gradient": edge_table,
        "summary": summary,
        "model": model,
        "out_dir": out_dir,
    }
