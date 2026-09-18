from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from connectome_analysis.analysis_config import GROUP_ORDER, AnalysisPaths
from connectome_analysis.analysis_edges import load_connectome_stack
from connectome_analysis.analysis_plots import save_dataframe, write_inference_markdown
from connectome_analysis.analysis_stats import bh_fdr, group_descriptives, kruskal_summary, pairwise_robust_tests, permutation_group_effect


@dataclass(frozen=True)
class EDRExceptionThresholds:
    short_max_mm: float
    long_min_mm: float
    source_group: str
    rule: str


PAIR_OPTIONS = (("CN", "MCI"), ("CN", "AD"), ("MCI", "AD"))
GLOBAL_METRICS = (
    "edr_lambda",
    "edr_slope",
    "sr_exception_pct",
    "mr_exception_pct",
    "lr_exception_pct",
    "sr_exception_strength",
    "lr_exception_strength",
    "sr_lr_exception_pct_diff",
    "sr_lr_exception_strength_ratio",
    "sr_w_mean",
    "mr_w_mean",
    "lr_w_mean",
    "sr_lr_w_diff_q13",
    "sr_lr_w_ratio_q13",
)
NODE_RANK_METRICS = (
    "sr_exception_rate",
    "lr_exception_rate",
    "sr_exception_strength",
    "lr_exception_strength",
    "sr_lr_exception_rate_diff",
)


def _aligned_stacks(paths: AnalysisPaths, master: pd.DataFrame, weight_type: str, length_type: str):
    weight = load_connectome_stack(paths, master, connectome_type=weight_type)
    length = load_connectome_stack(paths, master, connectome_type=length_type)
    length_idx = {sid: i for i, sid in enumerate(length.subject_ids)}
    rows = []
    groups = []
    weight_mats = []
    length_mats = []
    for i, sid in enumerate(weight.subject_ids):
        j = length_idx.get(sid)
        if j is None:
            continue
        rows.append(sid)
        groups.append(weight.groups[i])
        weight_mats.append(weight.matrices[i])
        length_mats.append(length.matrices[j])
    if not rows:
        raise FileNotFoundError(f"No aligned {weight_type}/{length_type} connectomes found")
    return rows, groups, np.stack(weight_mats, axis=0), np.stack(length_mats, axis=0)


def _aal_lookup(paths: AnalysisPaths, n_nodes: int) -> pd.DataFrame:
    label_path = paths.notebook_dir / "atlas" / "AAL" / "AAL3_labels.csv"
    if label_path.exists():
        labels = pd.read_csv(label_path)
        if {"node", "node_name", "atlas_label"}.issubset(labels.columns):
            labels = labels[["node", "node_name", "atlas_label"]].copy()
            labels["node"] = pd.to_numeric(labels["node"], errors="coerce").astype("Int64")
            return labels[labels["node"].between(1, n_nodes)].copy()
    return pd.DataFrame(
        {
            "node": np.arange(1, n_nodes + 1),
            "node_name": [f"AAL_{i:03d}" for i in range(1, n_nodes + 1)],
            "atlas_label": [f"AAL_{i:03d}" for i in range(1, n_nodes + 1)],
        }
    )


def _derive_q13_thresholds(groups: list[str], lengths: np.ndarray) -> EDRExceptionThresholds:
    group_arr = np.asarray(groups, dtype=object)
    source_mask = group_arr == "CN"
    if source_mask.sum() < 5:
        source_mask = np.ones(len(groups), dtype=bool)
        source_group = "all"
    else:
        source_group = "CN"
    iu = np.triu_indices(lengths.shape[1], 1)
    vals = lengths[source_mask][:, iu[0], iu[1]].ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < 100:
        raise ValueError("Not enough positive length observations to define EDR exception Q1/Q3 thresholds")
    q1, q3 = np.nanpercentile(vals, [25, 75])
    return EDRExceptionThresholds(
        short_max_mm=float(q1),
        long_min_mm=float(q3),
        source_group=source_group,
        rule="pooled upper-triangle Q1/Q3 from len_mean; SR <= Q1, MR Q1-Q3, LR > Q3",
    )


def _range_labels(edge_lengths: np.ndarray, thresholds: EDRExceptionThresholds) -> np.ndarray:
    out = np.full(edge_lengths.shape, "invalid", dtype=object)
    valid = np.isfinite(edge_lengths) & (edge_lengths > 0)
    out[valid & (edge_lengths <= thresholds.short_max_mm)] = "SR"
    out[valid & (edge_lengths > thresholds.short_max_mm) & (edge_lengths <= thresholds.long_min_mm)] = "MR"
    out[valid & (edge_lengths > thresholds.long_min_mm)] = "LR"
    return out


def _adaptive_bin_ids(lengths: np.ndarray, min_bin_edges: int = 120, max_bins: int = 30) -> np.ndarray:
    valid = np.isfinite(lengths) & (lengths > 0)
    ids = np.full(lengths.shape, -1, dtype=int)
    n_valid = int(valid.sum())
    if n_valid < max(20, min_bin_edges):
        return ids
    n_bins = int(np.clip(n_valid // min_bin_edges, 3, max_bins))
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.nanquantile(lengths[valid], quantiles))
    if len(edges) < 3:
        return ids
    bin_ids = np.searchsorted(edges[1:-1], lengths[valid], side="right")
    ids[valid] = bin_ids.astype(int)
    return ids


def _fit_subject_edr(edge_lengths: np.ndarray, edge_weights: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(edge_lengths) & (edge_lengths > 0) & np.isfinite(edge_weights) & (edge_weights > 0)
    if valid.sum() < 50 or np.unique(edge_lengths[valid]).size < 3 or np.unique(edge_weights[valid]).size < 3:
        return {"edr_intercept": np.nan, "edr_slope": np.nan, "edr_lambda": np.nan, "edr_r": np.nan, "edr_p": np.nan}
    fit = stats.linregress(edge_lengths[valid], np.log(edge_weights[valid]))
    return {
        "edr_intercept": float(fit.intercept),
        "edr_slope": float(fit.slope),
        "edr_lambda": float(-fit.slope),
        "edr_r": float(fit.rvalue),
        "edr_p": float(fit.pvalue),
    }


def _subject_exception_arrays(
    edge_lengths: np.ndarray,
    edge_weights: np.ndarray,
    min_bin_edges: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(edge_lengths) & (edge_lengths > 0) & np.isfinite(edge_weights) & (edge_weights > 0)
    bin_ids = _adaptive_bin_ids(edge_lengths, min_bin_edges=min_bin_edges)
    exception = np.zeros(edge_weights.shape, dtype=bool)
    bin_mean = np.full(edge_weights.shape, np.nan, dtype=float)
    bin_sd = np.full(edge_weights.shape, np.nan, dtype=float)
    threshold = np.full(edge_weights.shape, np.nan, dtype=float)
    for bin_id in sorted(set(bin_ids[valid].tolist())):
        if bin_id < 0:
            continue
        mask = valid & (bin_ids == bin_id)
        vals = edge_weights[mask]
        if vals.size < 20:
            continue
        mean = float(np.nanmean(vals))
        sd = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else np.nan
        cut = mean + 3.0 * sd if np.isfinite(sd) else np.nan
        bin_mean[mask] = mean
        bin_sd[mask] = sd
        threshold[mask] = cut
        if np.isfinite(cut):
            exception[mask] = edge_weights[mask] > cut
    return exception, bin_mean, bin_sd, threshold


def _range_mask(range_labels: np.ndarray, label: str) -> np.ndarray:
    return np.asarray(range_labels, dtype=object) == label


def _safe_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.nanmean(values)) if values.size else np.nan


def _safe_sum(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.nansum(values)) if values.size else np.nan


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and den != 0 else np.nan


def _subject_summary_row(
    sid: str,
    group: str,
    edge_lengths: np.ndarray,
    edge_weights: np.ndarray,
    range_labels: np.ndarray,
    exception: np.ndarray,
    edr_fit: dict[str, float],
) -> dict:
    valid = np.isfinite(edge_lengths) & (edge_lengths > 0) & np.isfinite(edge_weights) & (edge_weights > 0)
    row = {"subject_id": sid, "group": group, **edr_fit, "n_edges_valid": int(valid.sum()), "n_exceptions": int(exception.sum())}
    row["exception_pct"] = _safe_ratio(float(exception.sum()), float(valid.sum()))
    for prefix, label in (("sr", "SR"), ("mr", "MR"), ("lr", "LR")):
        mask = valid & _range_mask(range_labels, label)
        exc = mask & exception
        row[f"{prefix}_edge_count"] = int(mask.sum())
        row[f"{prefix}_w_mean"] = _safe_mean(edge_weights[mask])
        row[f"{prefix}_w_sum"] = _safe_sum(edge_weights[mask])
        row[f"{prefix}_density"] = _safe_ratio(float(mask.sum()), float(valid.sum()))
        row[f"{prefix}_exception_count"] = int(exc.sum())
        row[f"{prefix}_exception_pct"] = _safe_ratio(float(exc.sum()), float(mask.sum()))
        row[f"{prefix}_exception_strength"] = _safe_sum(edge_weights[exc])
    row["sr_lr_exception_pct_diff"] = row["sr_exception_pct"] - row["lr_exception_pct"]
    row["sr_lr_exception_strength_ratio"] = _safe_ratio(row["sr_exception_strength"], row["lr_exception_strength"])
    row["sr_lr_w_diff_q13"] = row["sr_w_mean"] - row["lr_w_mean"]
    row["sr_lr_w_ratio_q13"] = _safe_ratio(row["sr_w_mean"], row["lr_w_mean"])
    return row


def _node_exception_rows(
    sid: str,
    group: str,
    n_nodes: int,
    iu: tuple[np.ndarray, np.ndarray],
    edge_weights: np.ndarray,
    range_labels: np.ndarray,
    exception: np.ndarray,
    label_map: dict[int, dict[str, str]],
) -> list[dict]:
    rows = []
    for node0 in range(n_nodes):
        incident = (iu[0] == node0) | (iu[1] == node0)
        base = {
            "subject_id": sid,
            "group": group,
            "node": node0 + 1,
            "node_name": label_map.get(node0 + 1, {}).get("node_name", f"AAL_{node0 + 1:03d}"),
            "atlas_label": label_map.get(node0 + 1, {}).get("atlas_label", f"AAL_{node0 + 1:03d}"),
        }
        for prefix, label in (("sr", "SR"), ("mr", "MR"), ("lr", "LR")):
            mask = incident & _range_mask(range_labels, label)
            exc = mask & exception
            base[f"{prefix}_edge_count"] = int(mask.sum())
            base[f"{prefix}_exception_count"] = int(exc.sum())
            base[f"{prefix}_exception_rate"] = _safe_ratio(float(exc.sum()), float(mask.sum()))
            base[f"{prefix}_exception_strength"] = _safe_sum(edge_weights[exc])
        base["sr_lr_exception_rate_diff"] = base["sr_exception_rate"] - base["lr_exception_rate"]
        rows.append(base)
    return rows


def _exception_edge_rows(
    sid: str,
    group: str,
    iu: tuple[np.ndarray, np.ndarray],
    edge_lengths: np.ndarray,
    edge_weights: np.ndarray,
    range_labels: np.ndarray,
    exception: np.ndarray,
    bin_mean: np.ndarray,
    bin_sd: np.ndarray,
    threshold: np.ndarray,
    label_map: dict[int, dict[str, str]],
) -> list[dict]:
    rows = []
    edge_indices = np.where(exception)[0]
    for idx in edge_indices:
        i = int(iu[0][idx] + 1)
        j = int(iu[1][idx] + 1)
        rows.append(
            {
                "subject_id": sid,
                "group": group,
                "i": i,
                "j": j,
                "roi_a": label_map.get(i, {}).get("atlas_label", f"AAL_{i:03d}"),
                "roi_b": label_map.get(j, {}).get("atlas_label", f"AAL_{j:03d}"),
                "length_mm": float(edge_lengths[idx]),
                "weight": float(edge_weights[idx]),
                "edr_class": range_labels[idx],
                "exception_flag": 1,
                "bin_weight_mean": float(bin_mean[idx]) if np.isfinite(bin_mean[idx]) else np.nan,
                "bin_weight_sd": float(bin_sd[idx]) if np.isfinite(bin_sd[idx]) else np.nan,
                "exception_threshold_weight": float(threshold[idx]) if np.isfinite(threshold[idx]) else np.nan,
            }
        )
    return rows


def _stat_outputs(subject_table: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    covariate_cols = [c for c in ("age", "sex", "phase") if c in subject_table.columns]
    for metric in GLOBAL_METRICS:
        if metric not in subject_table.columns:
            continue
        sub = subject_table[["subject_id", "group", *covariate_cols, metric]].dropna(subset=[metric])
        if sub.empty:
            continue
        desc = group_descriptives(sub, metric)
        pairwise = pairwise_robust_tests(sub, metric)
        omni = kruskal_summary(sub, metric)
        perm = permutation_group_effect(sub, metric)
        save_dataframe(desc, out_dir / f"{metric}_descriptives.csv")
        save_dataframe(pairwise, out_dir / f"{metric}_pairwise.csv")
        rows.append({"metric": metric, "n": omni.n_total, "kw_stat": omni.statistic, "kw_p": omni.pvalue, "perm_f": perm["f_stat"], "perm_p": perm["p_perm"]})
    summary = pd.DataFrame(rows)
    save_dataframe(summary, out_dir / "edr_exception_summary.csv")
    return summary


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return np.nan
    try:
        u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
        return float((2.0 * u / (a.size * b.size)) - 1.0)
    except Exception:
        return np.nan


def _pairwise_node_rankings(node_table: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows = []
    for metric in NODE_RANK_METRICS:
        if metric not in node_table.columns:
            continue
        metric_df = node_table.dropna(subset=[metric]).copy()
        for group_a, group_b in PAIR_OPTIONS:
            pair_rows = []
            sub_pair = metric_df[metric_df["group"].isin([group_a, group_b])].copy()
            for (node, node_name, atlas_label), sub in sub_pair.groupby(["node", "node_name", "atlas_label"], dropna=False):
                a = pd.to_numeric(sub.loc[sub["group"] == group_a, metric], errors="coerce").dropna().to_numpy(dtype=float)
                b = pd.to_numeric(sub.loc[sub["group"] == group_b, metric], errors="coerce").dropna().to_numpy(dtype=float)
                if a.size < 5 or b.size < 5:
                    continue
                try:
                    p_value = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
                except Exception:
                    p_value = np.nan
                med_a = float(np.nanmedian(a)) if a.size else np.nan
                med_b = float(np.nanmedian(b)) if b.size else np.nan
                mean_a = float(np.nanmean(a)) if a.size else np.nan
                mean_b = float(np.nanmean(b)) if b.size else np.nan
                if np.isfinite(med_a) and np.isfinite(med_b) and not math.isclose(med_a, med_b, rel_tol=1e-9, abs_tol=1e-12):
                    higher_group = group_a if med_a > med_b else group_b
                elif np.isfinite(mean_a) and np.isfinite(mean_b) and not math.isclose(mean_a, mean_b, rel_tol=1e-9, abs_tol=1e-12):
                    higher_group = group_a if mean_a > mean_b else group_b
                else:
                    higher_group = "equal"
                pair_rows.append(
                    {
                        "metric": metric,
                        "comparison": f"{group_a} vs {group_b}",
                        "node": int(node),
                        "node_name": str(node_name),
                        "atlas_label": str(atlas_label),
                        f"n_{group_a}": int(a.size),
                        f"n_{group_b}": int(b.size),
                        f"median_{group_a}": med_a,
                        f"median_{group_b}": med_b,
                        f"mean_{group_a}": mean_a,
                        f"mean_{group_b}": mean_b,
                        "median_diff_a_minus_b": med_a - med_b if np.isfinite(med_a) and np.isfinite(med_b) else np.nan,
                        "higher_group": higher_group,
                        "p_value": p_value,
                        "cliffs_delta": _cliffs_delta(a, b),
                    }
                )
            pair_df = pd.DataFrame(pair_rows)
            if pair_df.empty:
                continue
            pair_df["q_value"] = bh_fdr(pair_df["p_value"].to_numpy())
            pair_df = pair_df.sort_values(["q_value", "p_value"], na_position="last").reset_index(drop=True)
            pair_df.insert(0, "rank", np.arange(1, len(pair_df) + 1))
            rows.append(pair_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _repeated_top_regions(rankings: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame()
    rows = []
    for metric, metric_df in rankings.groupby("metric"):
        top = metric_df[metric_df["rank"] <= top_n].copy()
        if top.empty:
            continue
        for (node, node_name, atlas_label), sub in top.groupby(["node", "node_name", "atlas_label"], dropna=False):
            directions = []
            for _, row in sub.iterrows():
                comparison = str(row.get("comparison"))
                if " vs " in comparison:
                    a, b = comparison.split(" vs ", 1)
                    higher = str(row.get("higher_group"))
                    if higher == a:
                        direction = f"{a}>{b}"
                    elif higher == b:
                        direction = f"{a}<{b}"
                    else:
                        direction = f"{a}={b}"
                else:
                    direction = comparison
                directions.append(direction)
            q = pd.to_numeric(sub["q_value"], errors="coerce")
            p = pd.to_numeric(sub["p_value"], errors="coerce")
            rows.append(
                {
                    "metric": metric,
                    "node": int(node),
                    "node_name": str(node_name),
                    "atlas_label": str(atlas_label),
                    "top_n_occurrences": int(len(sub)),
                    "fdr_significant_comparisons": int((q < 0.05).sum()),
                    "best_p": float(p.min()) if p.notna().any() else np.nan,
                    "best_q": float(q.min()) if q.notna().any() else np.nan,
                    "comparisons": "; ".join(dict.fromkeys(sub["comparison"].astype(str))),
                    "directions": "; ".join(dict.fromkeys(directions)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        ["fdr_significant_comparisons", "top_n_occurrences", "best_q", "best_p"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def _edge_group_map(
    subject_ids: list[str],
    groups: list[str],
    iu: tuple[np.ndarray, np.ndarray],
    edge_lengths: np.ndarray,
    edge_weights: np.ndarray,
    exception_strength: np.ndarray,
    range_labels_by_subject: np.ndarray,
    label_map: dict[int, dict[str, str]],
) -> pd.DataFrame:
    group_arr = np.asarray(groups, dtype=object)
    rows = []
    positive_lengths = np.where(np.isfinite(edge_lengths) & (edge_lengths > 0), edge_lengths, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_length = np.nanmedian(positive_lengths, axis=0)
    length_valid_n = np.sum(np.isfinite(positive_lengths), axis=0)
    exception_rates = {
        group: np.nanmean(exception_strength[group_arr == group] > 0, axis=0) if np.any(group_arr == group) else np.full(edge_weights.shape[1], np.nan)
        for group in GROUP_ORDER
    }
    class_by_edge = []
    for edge_idx in range(edge_weights.shape[1]):
        labels, counts = np.unique(range_labels_by_subject[:, edge_idx], return_counts=True)
        valid = [(str(label), int(count)) for label, count in zip(labels, counts) if str(label) in {"SR", "MR", "LR"}]
        class_by_edge.append(max(valid, key=lambda x: x[1])[0] if valid else "invalid")
    for group_a, group_b in PAIR_OPTIONS:
        if not (np.any(group_arr == group_a) and np.any(group_arr == group_b)):
            continue
        a_vals = exception_strength[group_arr == group_a]
        b_vals = exception_strength[group_arr == group_b]
        p_values = np.full(edge_weights.shape[1], np.nan, dtype=float)
        u_stats = np.full(edge_weights.shape[1], np.nan, dtype=float)
        med_a = np.nanmedian(a_vals, axis=0)
        med_b = np.nanmedian(b_vals, axis=0)
        mean_a = np.nanmean(a_vals, axis=0)
        mean_b = np.nanmean(b_vals, axis=0)
        for edge_idx in range(edge_weights.shape[1]):
            a = a_vals[:, edge_idx]
            b = b_vals[:, edge_idx]
            if np.count_nonzero(np.isfinite(a)) < 5 or np.count_nonzero(np.isfinite(b)) < 5:
                continue
            if np.nanmax(a) == 0 and np.nanmax(b) == 0:
                continue
            try:
                test = stats.mannwhitneyu(a, b, alternative="two-sided")
                u_stats[edge_idx] = float(test.statistic)
                p_values[edge_idx] = float(test.pvalue)
            except Exception:
                continue
        q_values = bh_fdr(p_values)
        for edge_idx in range(edge_weights.shape[1]):
            if not np.isfinite(p_values[edge_idx]):
                continue
            i = int(iu[0][edge_idx] + 1)
            j = int(iu[1][edge_idx] + 1)
            if med_a[edge_idx] > med_b[edge_idx]:
                direction = f"{group_a}>{group_b}"
            elif med_b[edge_idx] > med_a[edge_idx]:
                direction = f"{group_a}<{group_b}"
            elif mean_a[edge_idx] > mean_b[edge_idx]:
                direction = f"{group_a}>{group_b}"
            elif mean_b[edge_idx] > mean_a[edge_idx]:
                direction = f"{group_a}<{group_b}"
            else:
                direction = f"{group_a}={group_b}"
            rows.append(
                {
                    "comparison": f"{group_a} vs {group_b}",
                    "i": i,
                    "j": j,
                    "roi_a": label_map.get(i, {}).get("atlas_label", f"AAL_{i:03d}"),
                    "roi_b": label_map.get(j, {}).get("atlas_label", f"AAL_{j:03d}"),
                    "length_median_mm": float(median_length[edge_idx]) if np.isfinite(median_length[edge_idx]) else np.nan,
                    "length_valid_subjects": int(length_valid_n[edge_idx]),
                    "edr_class": class_by_edge[edge_idx],
                    f"exception_rate_{group_a}": float(exception_rates[group_a][edge_idx]),
                    f"exception_rate_{group_b}": float(exception_rates[group_b][edge_idx]),
                    f"median_exception_strength_{group_a}": float(med_a[edge_idx]) if np.isfinite(med_a[edge_idx]) else np.nan,
                    f"median_exception_strength_{group_b}": float(med_b[edge_idx]) if np.isfinite(med_b[edge_idx]) else np.nan,
                    "disease_direction": direction,
                    "mwu_u": float(u_stats[edge_idx]) if np.isfinite(u_stats[edge_idx]) else np.nan,
                    "p_value": float(p_values[edge_idx]),
                    "q_value": float(q_values[edge_idx]) if np.isfinite(q_values[edge_idx]) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["q_value", "p_value"], na_position="last").reset_index(drop=True)
    return out


def _build_ml_features(subject_table: pd.DataFrame, node_table: pd.DataFrame, max_nodes_per_metric: int = 30) -> pd.DataFrame:
    feature_cols = [
        col
        for col in subject_table.columns
        if col not in {"subject_id", "group", "age", "sex", "phase"} and pd.api.types.is_numeric_dtype(subject_table[col])
    ]
    features = subject_table[["subject_id", "group", *[c for c in ("age", "sex", "phase") if c in subject_table.columns], *feature_cols]].copy()
    node_features = node_table[["subject_id", "node_name", *[m for m in NODE_RANK_METRICS if m in node_table.columns]]].copy()
    if not node_features.empty:
        pivots = []
        for metric in NODE_RANK_METRICS:
            if metric not in node_features.columns:
                continue
            wide = node_features.pivot_table(index="subject_id", columns="node_name", values=metric, aggfunc="mean")
            variance = wide.var(axis=0, skipna=True).sort_values(ascending=False)
            keep_nodes = variance.head(max_nodes_per_metric).index.tolist()
            wide = wide[keep_nodes]
            wide.columns = [f"node_{metric}_{col}" for col in wide.columns]
            pivots.append(wide)
        if pivots:
            node_wide = pd.concat(pivots, axis=1).reset_index()
            features = features.merge(node_wide, on="subject_id", how="left")
    return features


def _numeric_feature_matrix(df: pd.DataFrame, include_covariates: bool) -> tuple[pd.DataFrame, list[str]]:
    work = df.copy()
    if include_covariates and "sex" in work.columns:
        work["sex_F"] = work["sex"].astype(str).str.upper().str.startswith("F").astype(float)
    candidate_cols = [
        c
        for c in work.columns
        if c not in {"subject_id", "group", "sex", "phase"} and pd.api.types.is_numeric_dtype(work[c])
    ]
    if not include_covariates:
        candidate_cols = [c for c in candidate_cols if c not in {"age", "sex_F"}]
    x = work[candidate_cols].replace([np.inf, -np.inf], np.nan)
    keep = [c for c in x.columns if pd.to_numeric(x[c], errors="coerce").notna().sum() >= 10 and pd.to_numeric(x[c], errors="coerce").nunique(dropna=True) > 1]
    return x[keep], keep


def _model_specs(min_train_class: int):
    return {
        "elastic_net_logistic": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.5,
                        class_weight="balanced",
                        max_iter=1500,
                        tol=1e-2,
                        n_jobs=1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=160,
                        max_features="sqrt",
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def _run_ml(features: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    tasks = [("CN_vs_MCI", ["CN", "MCI"]), ("CN_vs_AD", ["CN", "AD"]), ("MCI_vs_AD", ["MCI", "AD"]), ("multiclass", list(GROUP_ORDER))]
    perf_rows = []
    pred_rows = []
    imp_rows = []
    feature_sets = [("structural_edr", False), ("structural_edr_age_sex", True)]
    for target_name, target_groups in tasks:
        task_df = features[features["group"].isin(target_groups)].copy()
        if task_df["group"].nunique() < 2:
            continue
        y_labels = task_df["group"].astype(str).to_numpy()
        counts = pd.Series(y_labels).value_counts()
        min_class = int(counts.min()) if not counts.empty else 0
        if min_class < 5:
            continue
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_labels)
        n_splits = max(2, min(5, min_class))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for feature_set, include_covariates in feature_sets:
            x, feature_names = _numeric_feature_matrix(task_df, include_covariates=include_covariates)
            if len(feature_names) < 2:
                continue
            for model_name in ("elastic_net_logistic", "random_forest"):
                fold_metrics = []
                importances = []
                for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
                    train_counts = pd.Series(y[train_idx]).value_counts()
                    models = _model_specs(int(train_counts.min()))
                    model = models[model_name]
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        model.fit(x.iloc[train_idx], y[train_idx])
                    y_pred = model.predict(x.iloc[test_idx])
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(x.iloc[test_idx])
                    else:
                        y_prob = None
                    bal_acc = balanced_accuracy_score(y[test_idx], y_pred)
                    auroc = np.nan
                    if y_prob is not None:
                        try:
                            if len(encoder.classes_) == 2:
                                auroc = roc_auc_score(y[test_idx], y_prob[:, 1])
                            else:
                                auroc = roc_auc_score(y[test_idx], y_prob, multi_class="ovr", average="macro")
                        except Exception:
                            auroc = np.nan
                    sensitivity = specificity = np.nan
                    if len(encoder.classes_) == 2:
                        cm = confusion_matrix(y[test_idx], y_pred, labels=[0, 1])
                        if cm.shape == (2, 2):
                            tn, fp, fn, tp = cm.ravel()
                            sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
                            specificity = tn / (tn + fp) if (tn + fp) else np.nan
                    fold_metrics.append({"balanced_accuracy": bal_acc, "auroc": auroc, "sensitivity": sensitivity, "specificity": specificity})
                    for sid, truth, pred, idx in zip(task_df.iloc[test_idx]["subject_id"], y[test_idx], y_pred, test_idx):
                        row = {
                            "target": target_name,
                            "feature_set": feature_set,
                            "model": model_name,
                            "fold": fold,
                            "subject_id": sid,
                            "true_group": encoder.inverse_transform([truth])[0],
                            "predicted_group": encoder.inverse_transform([pred])[0],
                        }
                        if y_prob is not None:
                            for class_idx, class_name in enumerate(encoder.classes_):
                                row[f"prob_{class_name}"] = float(y_prob[list(test_idx).index(idx), class_idx])
                        pred_rows.append(row)
                    fitted_model = model.named_steps["model"]
                    if model_name == "elastic_net_logistic" and hasattr(fitted_model, "coef_"):
                        imp = np.nanmean(np.abs(fitted_model.coef_), axis=0)
                    elif model_name == "random_forest" and hasattr(fitted_model, "feature_importances_"):
                        imp = fitted_model.feature_importances_
                    else:
                        imp = np.zeros(len(feature_names))
                    importances.append(imp)
                metrics_df = pd.DataFrame(fold_metrics)
                if metrics_df.empty:
                    continue
                perf_rows.append(
                    {
                        "target": target_name,
                        "feature_set": feature_set,
                        "model": model_name,
                        "n_subjects": int(len(task_df)),
                        "n_features": int(len(feature_names)),
                        "n_splits": int(n_splits),
                        "balanced_accuracy_mean": float(metrics_df["balanced_accuracy"].mean()),
                        "balanced_accuracy_sd": float(metrics_df["balanced_accuracy"].std(ddof=1)) if len(metrics_df) > 1 else np.nan,
                        "auroc_mean": float(metrics_df["auroc"].mean()),
                        "auroc_sd": float(metrics_df["auroc"].std(ddof=1)) if len(metrics_df) > 1 else np.nan,
                        "sensitivity_mean": float(metrics_df["sensitivity"].mean()),
                        "specificity_mean": float(metrics_df["specificity"].mean()),
                    }
                )
                if importances:
                    mean_imp = np.nanmean(np.vstack(importances), axis=0)
                    order = np.argsort(mean_imp)[::-1][:50]
                    for rank, feature_idx in enumerate(order, start=1):
                        imp_rows.append(
                            {
                                "target": target_name,
                                "feature_set": feature_set,
                                "model": model_name,
                                "rank": rank,
                                "feature": feature_names[int(feature_idx)],
                                "importance": float(mean_imp[int(feature_idx)]),
                            }
                        )
    perf = pd.DataFrame(perf_rows)
    pred = pd.DataFrame(pred_rows)
    imp = pd.DataFrame(imp_rows)
    save_dataframe(features, out_dir / "edr_ml_subject_features.csv")
    save_dataframe(perf, out_dir / "edr_ml_model_performance.csv")
    save_dataframe(pred, out_dir / "edr_ml_cross_validated_predictions.csv")
    save_dataframe(imp, out_dir / "edr_ml_feature_importance.csv")
    return {"features": features, "performance": perf, "predictions": pred, "importance": imp}


def run_edr_exception_analysis(
    paths: AnalysisPaths,
    master: pd.DataFrame,
    weight_type: str = "fd_sum",
    length_type: str = "len_mean",
    min_bin_edges: int = 120,
    top_n: int = 20,
    run_ml: bool = False,
) -> dict:
    out_dir = paths.section_dir("17", "edr_exceptions")
    out_dir.mkdir(parents=True, exist_ok=True)
    subject_ids, groups, weights, lengths = _aligned_stacks(paths, master, weight_type, length_type)
    n_subjects, n_nodes, _ = weights.shape
    iu = np.triu_indices(n_nodes, 1)
    thresholds = _derive_q13_thresholds(groups, lengths)
    labels = _aal_lookup(paths, n_nodes)
    label_map = {
        int(row.node): {"node_name": str(row.node_name), "atlas_label": str(row.atlas_label)}
        for row in labels.itertuples(index=False)
        if pd.notna(row.node)
    }

    subject_rows = []
    node_rows = []
    exception_edge_rows = []
    exception_strength = np.zeros((n_subjects, len(iu[0])), dtype=float)
    edge_weight_ut = np.zeros((n_subjects, len(iu[0])), dtype=float)
    edge_length_ut = np.zeros((n_subjects, len(iu[0])), dtype=float)
    range_label_ut = np.empty((n_subjects, len(iu[0])), dtype=object)

    for idx, (sid, group, weight_mat, len_mat) in enumerate(zip(subject_ids, groups, weights, lengths)):
        edge_lengths = np.asarray(len_mat[iu], dtype=float)
        edge_weights = np.asarray(weight_mat[iu], dtype=float)
        edge_weight_ut[idx] = np.nan_to_num(edge_weights, nan=0.0, posinf=0.0, neginf=0.0)
        edge_length_ut[idx] = edge_lengths
        range_labels = _range_labels(edge_lengths, thresholds)
        range_label_ut[idx] = range_labels
        edr_fit = _fit_subject_edr(edge_lengths, edge_weights)
        exception, bin_mean, bin_sd, exception_threshold = _subject_exception_arrays(
            edge_lengths,
            edge_weights,
            min_bin_edges=min_bin_edges,
        )
        exception_strength[idx] = np.where(exception, edge_weight_ut[idx], 0.0)
        subject_rows.append(_subject_summary_row(sid, group, edge_lengths, edge_weights, range_labels, exception, edr_fit))
        node_rows.extend(_node_exception_rows(sid, group, n_nodes, iu, edge_weights, range_labels, exception, label_map))
        exception_edge_rows.extend(
            _exception_edge_rows(
                sid,
                group,
                iu,
                edge_lengths,
                edge_weights,
                range_labels,
                exception,
                bin_mean,
                bin_sd,
                exception_threshold,
                label_map,
            )
        )

    subject_table = pd.DataFrame(subject_rows).merge(
        master[[c for c in ["subject_id", "age", "sex", "phase"] if c in master.columns]].drop_duplicates("subject_id"),
        on="subject_id",
        how="left",
    )
    node_table = pd.DataFrame(node_rows).merge(
        master[[c for c in ["subject_id", "age", "sex", "phase"] if c in master.columns]].drop_duplicates("subject_id"),
        on="subject_id",
        how="left",
    )
    edge_level = pd.DataFrame(exception_edge_rows)
    edge_group = _edge_group_map(
        subject_ids,
        groups,
        iu,
        edge_length_ut,
        edge_weight_ut,
        exception_strength,
        range_label_ut,
        label_map,
    )
    rankings = _pairwise_node_rankings(node_table, top_n=top_n)
    repeated = _repeated_top_regions(rankings, top_n=top_n)
    summary = _stat_outputs(subject_table, out_dir)

    save_dataframe(
        pd.DataFrame(
            [
                {
                    "short_max_mm": thresholds.short_max_mm,
                    "long_min_mm": thresholds.long_min_mm,
                    "source_group": thresholds.source_group,
                    "rule": thresholds.rule,
                    "weight_type": weight_type,
                    "length_type": length_type,
                    "exception_rule": "within-subject adaptive length bin; weight > bin mean + 3*bin sd",
                    "min_bin_edges": min_bin_edges,
                    "n_subjects": len(subject_table),
                }
            ]
        ),
        out_dir / "edr_exception_thresholds.csv",
    )
    save_dataframe(subject_table, out_dir / f"edr_exception_subject_level_{weight_type}_{length_type}.csv")
    save_dataframe(edge_level, out_dir / f"edr_exception_edge_level_{weight_type}_{length_type}.csv")
    save_dataframe(node_table, out_dir / f"edr_exception_node_level_{weight_type}_{length_type}.csv")
    save_dataframe(edge_group, out_dir / f"edr_exception_edge_group_map_{weight_type}_{length_type}.csv")
    save_dataframe(rankings, out_dir / "edr_exception_pairwise_roi_rankings.csv")
    save_dataframe(repeated, out_dir / "edr_exception_repeated_top_regions.csv")
    write_inference_markdown(
        [
            "EDR exceptions are structural-only edge outliers: each subject is fitted with log(W_ij) = alpha_s - lambda_s L_ij, then adaptive length bins flag edges whose raw weight exceeds the bin mean plus three standard deviations.",
            "SR/MR/LR exception classes use CN-referenced Q1/Q3 len_mean thresholds and are intentionally separate from the existing LR/SR tertile dashboard.",
            "Node-level exception metrics summarize which AAL3 regions participate most in short- or long-range exception edges.",
            "Multiclass ML diagnosis is generated separately in the 18_ml_diagnostics workflow so EDR exception outputs remain a structural analysis block.",
        ],
        out_dir / "edr_exception_inference.md",
    )
    ml = _run_ml(_build_ml_features(subject_table, node_table), out_dir) if run_ml else {}
    return {
        "subject_level": subject_table,
        "node_level": node_table,
        "edge_level": edge_level,
        "edge_group": edge_group,
        "rankings": rankings,
        "repeated": repeated,
        "summary": summary,
        "ml": ml,
        "out_dir": out_dir,
    }
