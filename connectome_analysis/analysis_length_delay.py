from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.csgraph import shortest_path

from connectome_analysis.analysis_config import GROUP_ORDER, AnalysisPaths
from connectome_analysis.analysis_edges import load_connectome_stack
from connectome_analysis.analysis_plots import boxplot_with_points, save_dataframe, write_inference_markdown
from connectome_analysis.analysis_stats import bh_fdr, group_descriptives, kruskal_summary, pairwise_robust_tests, permutation_group_effect


@dataclass
class LengthThresholds:
    short_max_mm: float
    medium_max_mm: float


def _parity_hemisphere(node_index_1based: int) -> str:
    return "L" if node_index_1based % 2 == 1 else "R"


def _weighted_density(weight_matrix: np.ndarray, mask: np.ndarray) -> float:
    vals = weight_matrix[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.nanmean(vals)) if len(vals) else np.nan


def _global_efficiency_from_weights(weight_matrix: np.ndarray, mask: np.ndarray) -> float:
    n = weight_matrix.shape[0]
    costs = np.full((n, n), np.inf, dtype=float)
    valid = mask & np.isfinite(weight_matrix) & (weight_matrix > 0)
    costs[valid] = 1.0 / weight_matrix[valid]
    np.fill_diagonal(costs, 0.0)
    dist = shortest_path(np.ascontiguousarray(costs), directed=False, unweighted=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dist
    np.fill_diagonal(inv, 0.0)
    finite = inv[np.isfinite(inv)]
    return float(np.nanmean(finite)) if finite.size else np.nan


def _mean_path_from_costs(cost_matrix: np.ndarray) -> float:
    dist = shortest_path(np.ascontiguousarray(cost_matrix), directed=False, unweighted=False)
    tri = dist[np.triu_indices_from(dist, 1)]
    tri = tri[np.isfinite(tri)]
    return float(np.nanmean(tri)) if tri.size else np.nan


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return np.nan
    try:
        u_stat = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
        return float((2.0 * u_stat / (a.size * b.size)) - 1.0)
    except Exception:
        gt = sum(float(x > y) for x in a for y in b)
        lt = sum(float(x < y) for x in a for y in b)
        return float((gt - lt) / (a.size * b.size))


def _finite_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _delay_node_rows(
    sid: str,
    group: str,
    weight_mat: np.ndarray,
    delay_mat: np.ndarray,
    delay_threshold: float,
) -> list[dict]:
    rows: list[dict] = []
    n_nodes = min(weight_mat.shape[0], delay_mat.shape[0])
    for idx in range(n_nodes):
        weights = np.asarray(weight_mat[idx, :n_nodes], dtype=float).copy()
        delays = np.asarray(delay_mat[idx, :n_nodes], dtype=float).copy()
        if idx < len(weights):
            weights[idx] = np.nan
        if idx < len(delays):
            delays[idx] = np.nan
        valid = np.isfinite(weights) & (weights > 0) & np.isfinite(delays) & (delays > 0)
        valid_delays = delays[valid]
        valid_weights = weights[valid]
        if not len(valid_delays):
            continue
        weight_sum = float(np.nansum(valid_weights))
        burden = float(np.nansum(valid_weights * valid_delays) / weight_sum) if weight_sum > 0 else np.nan
        node = idx + 1
        rows.append(
            {
                "subject_id": sid,
                "group": group,
                "node": node,
                "node_name": f"AAL_{node:03d}",
                "node_mean_delay_ms": float(np.nanmean(valid_delays)),
                "node_median_delay_ms": float(np.nanmedian(valid_delays)),
                "node_p90_delay_ms": float(np.nanpercentile(valid_delays, 90)),
                "node_delay_burden_ms": burden,
                "node_long_delay_fraction": (
                    float(np.mean(valid_delays >= delay_threshold))
                    if np.isfinite(delay_threshold)
                    else np.nan
                ),
                "node_delay_weighted_strength": float(np.nansum(valid_weights / valid_delays)),
                "node_edge_count": int(len(valid_delays)),
            }
        )
    return rows


def _node_pairwise_rankings(node_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    if node_df.empty:
        return pd.DataFrame()
    for metric in metrics:
        if metric not in node_df.columns:
            continue
        for node_name, node_data in node_df.groupby("node_name", sort=False):
            if "node" in node_data.columns:
                node_value = pd.to_numeric(node_data["node"], errors="coerce").dropna()
                node = int(node_value.iloc[0]) if not node_value.empty else np.nan
            else:
                node = np.nan
            for group_a, group_b in (("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")):
                a = _finite_numeric(node_data.loc[node_data["group"] == group_a, metric])
                b = _finite_numeric(node_data.loc[node_data["group"] == group_b, metric])
                p_value = np.nan
                if a.size >= 2 and b.size >= 2:
                    try:
                        p_value = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
                    except Exception:
                        p_value = np.nan
                median_a = float(np.nanmedian(a)) if a.size else np.nan
                median_b = float(np.nanmedian(b)) if b.size else np.nan
                rows.append(
                    {
                        "metric": metric,
                        "comparison": f"{group_a} vs {group_b}",
                        "node": node,
                        "node_name": node_name,
                        f"n_{group_a}": int(a.size),
                        f"n_{group_b}": int(b.size),
                        f"Median {group_a}": median_a,
                        f"Median {group_b}": median_b,
                        f"Median diff {group_a}-{group_b}": (
                            median_a - median_b if np.isfinite(median_a) and np.isfinite(median_b) else np.nan
                        ),
                        "higher_group": (
                            group_a
                            if np.isfinite(median_a) and np.isfinite(median_b) and median_a > median_b
                            else (group_b if np.isfinite(median_a) and np.isfinite(median_b) and median_b > median_a else "tie")
                        ),
                        "p": p_value,
                        "Cliff's delta": _cliffs_delta(a, b),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["BH q"] = np.nan
        for metric, idx in out.groupby("metric").groups.items():
            out.loc[idx, "BH q"] = bh_fdr(out.loc[idx, "p"].to_numpy(dtype=float))
        out["fdr_q_lt_0_05"] = pd.to_numeric(out["BH q"], errors="coerce") < 0.05
        out = out.sort_values(["metric", "comparison", "p", "BH q"], na_position="last").reset_index(drop=True)
    return out


def _repeated_top_regions(pairwise: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for metric, metric_df in pairwise.groupby("metric", sort=False):
        metric_rows: list[pd.DataFrame] = []
        for comparison, comp_df in metric_df.groupby("comparison", sort=False):
            comp = comp_df.copy()
            comp["p"] = pd.to_numeric(comp["p"], errors="coerce")
            comp = comp.sort_values(["p", "BH q"], na_position="last").head(top_n)
            metric_rows.append(comp)
        if not metric_rows:
            continue
        ranked = pd.concat(metric_rows, ignore_index=True)
        for node_name, node_df in ranked.groupby("node_name", sort=False):
            best = node_df.sort_values(["p", "BH q"], na_position="last").iloc[0]
            rows.append(
                {
                    "metric": metric,
                    "node": best.get("node"),
                    "node_name": node_name,
                    "Top-N hits": int(len(node_df)),
                    "BH q<0.05 hits": int((pd.to_numeric(node_df["BH q"], errors="coerce") < 0.05).sum()),
                    "Best p": pd.to_numeric(pd.Series([node_df["p"].min()]), errors="coerce").iloc[0],
                    "Best BH q": pd.to_numeric(pd.Series([node_df["BH q"].min()]), errors="coerce").iloc[0],
                    "comparisons": "; ".join(dict.fromkeys(node_df["comparison"].astype(str))),
                    "directions": "; ".join(
                        dict.fromkeys(
                            f"{str(row.get('higher_group'))} higher in {row.get('comparison')}"
                            for _, row in node_df.iterrows()
                        )
                    ),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["metric", "BH q<0.05 hits", "Top-N hits", "Best p", "Best BH q"],
            ascending=[True, False, False, True, True],
            na_position="last",
        ).reset_index(drop=True)
        out["Rank"] = out.groupby("metric").cumcount() + 1
        front = ["Rank", "metric", "node", "node_name"]
        out = out[front + [c for c in out.columns if c not in front]]
    return out


def derive_length_thresholds(
    paths: AnalysisPaths,
    master: pd.DataFrame,
    length_type: str = "len_mean",
) -> LengthThresholds:
    stack = load_connectome_stack(paths, master.loc[master["group"] == "CN"], connectome_type=length_type)
    iu = np.triu_indices(stack.nodes, 1)
    lengths = stack.matrices[:, iu[0], iu[1]].ravel()
    lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
    q1, q2 = np.nanpercentile(lengths, [33.333, 66.667])
    thresholds = LengthThresholds(short_max_mm=float(q1), medium_max_mm=float(q2))
    pd.DataFrame(
        [
            {
                "short_max_mm": thresholds.short_max_mm,
                "medium_max_mm": thresholds.medium_max_mm,
                "source_group": "CN",
                "rule": "pooled upper-triangle tertiles from len_mean",
            }
        ]
    ).to_csv(paths.section_dir("12", "length_delay") / "lr_sr_thresholds.csv", index=False)
    return thresholds


def _edge_class_masks(length_matrix: np.ndarray, thresholds: LengthThresholds) -> dict[str, np.ndarray]:
    iu = np.triu_indices_from(length_matrix, 1)
    base = np.zeros_like(length_matrix, dtype=bool)
    valid = np.isfinite(length_matrix) & (length_matrix > 0)
    short = valid & (length_matrix <= thresholds.short_max_mm)
    medium = valid & (length_matrix > thresholds.short_max_mm) & (length_matrix <= thresholds.medium_max_mm)
    long = valid & (length_matrix > thresholds.medium_max_mm)
    masks = {"short": short, "medium": medium, "long": long}
    for key, mask in masks.items():
        mask = np.triu(mask, 1)
        masks[key] = mask | mask.T
    return masks


def _intra_inter_masks(n_nodes: int) -> dict[str, np.ndarray]:
    hemi = np.array([_parity_hemisphere(i + 1) for i in range(n_nodes)])
    same = hemi[:, None] == hemi[None, :]
    np.fill_diagonal(same, False)
    diff = ~same
    np.fill_diagonal(diff, False)
    return {"intra": same, "inter": diff}


def run_lr_sr_analysis(
    paths: AnalysisPaths,
    master: pd.DataFrame,
    weight_type: str = "fd_sum",
    length_type: str = "len_mean",
    velocity_mm_per_ms: float = 6.0,
) -> dict:
    out_dir = paths.section_dir("12", "length_delay")
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = derive_length_thresholds(paths, master, length_type=length_type)
    weight_stack = load_connectome_stack(paths, master, connectome_type=weight_type)
    length_stack = load_connectome_stack(paths, master, connectome_type=length_type)
    intra_inter = _intra_inter_masks(weight_stack.nodes)
    rows = []
    for sid, group, weight_mat, len_mat in zip(
        weight_stack.subject_ids, weight_stack.groups, weight_stack.matrices, length_stack.matrices
    ):
        class_masks = _edge_class_masks(len_mat, thresholds)
        row = {"subject_id": sid, "group": group}
        for cls, mask in class_masks.items():
            tri_mask = np.triu(mask, 1)
            edge_lengths = len_mat[tri_mask]
            edge_weights = weight_mat[tri_mask]
            row[f"{cls}_edge_count"] = int(np.isfinite(edge_lengths).sum())
            row[f"{cls}_len_median"] = float(np.nanmedian(edge_lengths)) if np.isfinite(edge_lengths).any() else np.nan
            row[f"{cls}_len_mean"] = float(np.nanmean(edge_lengths)) if np.isfinite(edge_lengths).any() else np.nan
            row[f"{cls}_w_median"] = float(np.nanmedian(edge_weights)) if np.isfinite(edge_weights).any() else np.nan
            row[f"{cls}_w_mean"] = float(np.nanmean(edge_weights)) if np.isfinite(edge_weights).any() else np.nan
            row[f"{cls}_w_sum"] = float(np.nansum(edge_weights)) if np.isfinite(edge_weights).any() else np.nan
            row[f"{cls}_density"] = _weighted_density(weight_mat, tri_mask)
            row[f"{cls}_global_eff"] = _global_efficiency_from_weights(weight_mat, tri_mask | tri_mask.T)
            delay_matrix = np.where(len_mat > 0, len_mat / velocity_mm_per_ms, np.nan)
            delay_cost = np.full_like(delay_matrix, np.inf, dtype=float)
            delay_valid = (tri_mask | tri_mask.T) & np.isfinite(delay_matrix) & (delay_matrix > 0)
            delay_cost[delay_valid] = delay_matrix[delay_valid]
            np.fill_diagonal(delay_cost, 0.0)
            row[f"{cls}_mean_delay_path_ms"] = _mean_path_from_costs(delay_cost)
        row["sr_lr_w_diff"] = row["short_w_mean"] - row["long_w_mean"]
        row["sr_lr_w_ratio"] = (
            row["short_w_mean"] / row["long_w_mean"]
            if np.isfinite(row["short_w_mean"]) and np.isfinite(row["long_w_mean"]) and row["long_w_mean"] != 0
            else np.nan
        )
        row["sr_lr_w_normdiff"] = (
            (row["short_w_mean"] - row["long_w_mean"]) / (row["short_w_mean"] + row["long_w_mean"])
            if np.isfinite(row["short_w_mean"]) and np.isfinite(row["long_w_mean"]) and (row["short_w_mean"] + row["long_w_mean"]) != 0
            else np.nan
        )
        for hemi_key, hemi_mask in intra_inter.items():
            tri_mask = np.triu(hemi_mask & np.isfinite(len_mat) & (len_mat > 0), 1)
            row[f"{hemi_key}_len_median"] = float(np.nanmedian(len_mat[tri_mask])) if tri_mask.any() else np.nan
            row[f"{hemi_key}_w_mean"] = float(np.nanmean(weight_mat[tri_mask])) if tri_mask.any() else np.nan
        rows.append(row)
    lr_sr = pd.DataFrame(rows)
    save_dataframe(lr_sr, out_dir / f"lr_sr_subject_level_{weight_type}_{length_type}.csv")

    summary_rows = []
    focus_metrics = [
        "short_w_mean",
        "long_w_mean",
        "sr_lr_w_diff",
        "sr_lr_w_ratio",
        "short_global_eff",
        "long_global_eff",
        "intra_w_mean",
        "inter_w_mean",
    ]
    for metric in focus_metrics:
        sub = lr_sr[["subject_id", "group", metric]].dropna()
        if sub.empty:
            continue
        desc = group_descriptives(sub, metric)
        pairwise = pairwise_robust_tests(sub, metric)
        omni = kruskal_summary(sub, metric)
        perm = permutation_group_effect(sub, metric)
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
            omnibus_text=f"Kruskal p={omni.pvalue:.3g}; perm p={perm['p_perm']:.3g}",
            pairwise_lines=pair_lines,
        )
        summary_rows.append(
            {"metric": metric, "n": omni.n_total, "kw_p": omni.pvalue, "perm_p": perm["p_perm"]}
        )
    summary = pd.DataFrame(summary_rows)
    save_dataframe(summary, out_dir / "lr_sr_summary.csv")
    write_inference_markdown(
        [
            "Short / medium / long classes are derived from CN pooled len_mean tertiles and then frozen for all subjects.",
            "This is explicitly a short-range edge proxy analysis, not a superficial white matter or U-fiber extraction.",
            "The module also decomposes intra- versus inter-hemispheric edges using AAL index parity as the hemisphere proxy because label-side metadata are not currently stored.",
        ],
        out_dir / "lr_sr_inference.md",
    )
    return {"subject_level": lr_sr, "summary": summary, "thresholds": thresholds, "out_dir": out_dir}


def run_delay_analysis(
    paths: AnalysisPaths,
    master: pd.DataFrame,
    weight_type: str = "fd_sum",
    length_type: str = "len_mean",
    velocity_mm_per_ms: float = 6.0,
) -> dict:
    out_dir = paths.section_dir("13", "delay")
    out_dir.mkdir(parents=True, exist_ok=True)
    weight_stack = load_connectome_stack(paths, master, connectome_type=weight_type)
    length_stack = load_connectome_stack(paths, master, connectome_type=length_type)
    cn_lengths = length_stack.matrices[np.array(length_stack.groups) == "CN"]
    iu = np.triu_indices(length_stack.nodes, 1)
    cn_delay = (cn_lengths[:, iu[0], iu[1]] / velocity_mm_per_ms).ravel()
    cn_delay = cn_delay[np.isfinite(cn_delay) & (cn_delay > 0)]
    delay_threshold = float(np.nanpercentile(cn_delay, 75)) if len(cn_delay) else np.nan

    rows = []
    node_rows: list[dict] = []
    for sid, group, weight_mat, len_mat in zip(
        weight_stack.subject_ids, weight_stack.groups, weight_stack.matrices, length_stack.matrices
    ):
        delay_mat = np.where(np.isfinite(len_mat) & (len_mat > 0), len_mat / velocity_mm_per_ms, np.nan)
        delay_ut = delay_mat[iu]
        weight_ut = weight_mat[iu]
        valid = np.isfinite(delay_ut) & np.isfinite(weight_ut) & (weight_ut > 0)
        delay_valid = delay_ut[valid]
        weight_valid = weight_ut[valid]
        burden = np.nansum(weight_valid * delay_valid) / np.nansum(weight_valid) if len(weight_valid) else np.nan
        delay_cost = np.full_like(delay_mat, np.inf, dtype=float)
        delay_cost[np.isfinite(delay_mat) & (delay_mat > 0)] = delay_mat[np.isfinite(delay_mat) & (delay_mat > 0)]
        np.fill_diagonal(delay_cost, 0.0)
        rows.append(
            {
                "subject_id": sid,
                "group": group,
                "velocity_mm_per_ms": velocity_mm_per_ms,
                "mean_delay_ms": float(np.nanmean(delay_valid)) if len(delay_valid) else np.nan,
                "median_delay_ms": float(np.nanmedian(delay_valid)) if len(delay_valid) else np.nan,
                "p90_delay_ms": float(np.nanpercentile(delay_valid, 90)) if len(delay_valid) else np.nan,
                "delay_burden_ms": burden,
                "long_delay_fraction": float(np.mean(delay_valid >= delay_threshold)) if len(delay_valid) and np.isfinite(delay_threshold) else np.nan,
                "delay_weighted_strength": float(np.nansum(weight_valid / delay_valid)) if len(delay_valid) else np.nan,
                "delay_path_ms": _mean_path_from_costs(delay_cost),
            }
        )
        node_rows.extend(_delay_node_rows(sid, group, weight_mat, delay_mat, delay_threshold))
    delays = pd.DataFrame(rows)
    save_dataframe(delays, out_dir / f"delay_subject_level_{weight_type}_{length_type}.csv")

    node_level = pd.DataFrame(node_rows)
    delay_node_metrics = [
        "node_mean_delay_ms",
        "node_median_delay_ms",
        "node_p90_delay_ms",
        "node_delay_burden_ms",
        "node_long_delay_fraction",
        "node_delay_weighted_strength",
    ]
    if not node_level.empty:
        save_dataframe(node_level, out_dir / f"delay_node_level_{weight_type}_{length_type}.csv")
        node_rankings = _node_pairwise_rankings(node_level, delay_node_metrics)
        save_dataframe(node_rankings, out_dir / "delay_pairwise_roi_rankings.csv")
        save_dataframe(_repeated_top_regions(node_rankings), out_dir / "delay_repeated_top_regions.csv")
    else:
        save_dataframe(pd.DataFrame(), out_dir / f"delay_node_level_{weight_type}_{length_type}.csv")
        save_dataframe(pd.DataFrame(), out_dir / "delay_pairwise_roi_rankings.csv")
        save_dataframe(pd.DataFrame(), out_dir / "delay_repeated_top_regions.csv")

    summary_rows = []
    for metric in [
        "mean_delay_ms",
        "median_delay_ms",
        "p90_delay_ms",
        "delay_burden_ms",
        "long_delay_fraction",
        "delay_weighted_strength",
        "delay_path_ms",
    ]:
        sub = delays[["subject_id", "group", metric]].dropna()
        if sub.empty:
            continue
        desc = group_descriptives(sub, metric)
        pairwise = pairwise_robust_tests(sub, metric)
        omni = kruskal_summary(sub, metric)
        perm = permutation_group_effect(sub, metric)
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
            omnibus_text=f"Kruskal p={omni.pvalue:.3g}; perm p={perm['p_perm']:.3g}",
            pairwise_lines=pair_lines,
        )
        summary_rows.append(
            {"metric": metric, "n": omni.n_total, "kw_p": omni.pvalue, "perm_p": perm["p_perm"]}
        )
    summary = pd.DataFrame(summary_rows)
    save_dataframe(summary, out_dir / "delay_summary.csv")
    write_inference_markdown(
        [
            "Delay estimates are currently length-only and depend on a configurable fixed conduction velocity assumption.",
            "Because the same velocity is applied to all subjects, group differences are driven by tract-length topology rather than by a myelin-informed speed model.",
            "Microstructure-informed delay remains an optional extension and is intentionally not run unless compatible inputs are supplied.",
        ],
        out_dir / "delay_inference.md",
    )
    return {
        "subject_level": delays,
        "node_level": node_level,
        "summary": summary,
        "velocity_mm_per_ms": velocity_mm_per_ms,
        "delay_threshold_ms": delay_threshold,
        "out_dir": out_dir,
    }


def microstructure_delay_available(paths: AnalysisPaths) -> bool:
    candidates = [
        paths.connectomes_dir.glob("SC_AAL166_*_g_ratio_mean.csv"),
        paths.connectomes_dir.glob("SC_AAL166_*_myelin_mean.csv"),
        paths.connectomes_dir.glob("SC_AAL166_*_velocity.csv"),
    ]
    return any(any(True for _ in group) for group in candidates)
