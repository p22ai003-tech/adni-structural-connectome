from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from connectome_analysis.analysis_config import GROUP_ORDER


@dataclass
class KruskalResult:
    statistic: float
    pvalue: float
    n_total: int
    groups: list[str]


def _ordered_groups(values: Iterable[str]) -> list[str]:
    present = [str(v) for v in values if pd.notna(v)]
    ordered = [g for g in GROUP_ORDER if g in present]
    ordered.extend(sorted(g for g in set(present) if g not in ordered))
    return ordered


def _finite_array(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def bh_fdr(p_values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


def group_descriptives(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "group",
) -> pd.DataFrame:
    rows = []
    for group in _ordered_groups(df[group_col].dropna().unique() if group_col in df else []):
        vals = _finite_array(df.loc[df[group_col] == group, value_col])
        rows.append(
            {
                "group": group,
                "n": int(vals.size),
                "mean": float(np.nanmean(vals)) if vals.size else np.nan,
                "sd": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else np.nan,
                "median": float(np.nanmedian(vals)) if vals.size else np.nan,
                "q1": float(np.nanpercentile(vals, 25)) if vals.size else np.nan,
                "q3": float(np.nanpercentile(vals, 75)) if vals.size else np.nan,
                "iqr": float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)) if vals.size else np.nan,
                "min": float(np.nanmin(vals)) if vals.size else np.nan,
                "max": float(np.nanmax(vals)) if vals.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def kruskal_summary(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "group",
) -> KruskalResult:
    groups = []
    arrays = []
    for group in _ordered_groups(df[group_col].dropna().unique() if group_col in df else []):
        vals = _finite_array(df.loc[df[group_col] == group, value_col])
        if vals.size >= 2:
            groups.append(group)
            arrays.append(vals)
    n_total = int(sum(arr.size for arr in arrays))
    if len(arrays) < 2:
        return KruskalResult(np.nan, np.nan, n_total, groups)
    try:
        stat, pvalue = stats.kruskal(*arrays, nan_policy="omit")
    except Exception:
        stat, pvalue = np.nan, np.nan
    return KruskalResult(float(stat), float(pvalue), n_total, groups)


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


def pairwise_robust_tests(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "group",
) -> pd.DataFrame:
    rows = []
    groups = _ordered_groups(df[group_col].dropna().unique() if group_col in df else [])
    for group_a, group_b in combinations(groups, 2):
        a = _finite_array(df.loc[df[group_col] == group_a, value_col])
        b = _finite_array(df.loc[df[group_col] == group_b, value_col])
        bm_stat = bm_p = mwu_p = np.nan
        if a.size >= 2 and b.size >= 2:
            try:
                bm = stats.brunnermunzel(a, b, alternative="two-sided", nan_policy="omit")
                bm_stat, bm_p = float(bm.statistic), float(bm.pvalue)
            except Exception:
                bm_stat, bm_p = np.nan, np.nan
            try:
                mwu_p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
            except Exception:
                mwu_p = np.nan
        rows.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "n_a": int(a.size),
                "n_b": int(b.size),
                "median_a": float(np.nanmedian(a)) if a.size else np.nan,
                "median_b": float(np.nanmedian(b)) if b.size else np.nan,
                "median_diff_a_minus_b": (
                    float(np.nanmedian(a) - np.nanmedian(b)) if a.size and b.size else np.nan
                ),
                "bm_stat": bm_stat,
                "bm_p": bm_p,
                "mwu_p": mwu_p,
                "cliffs_delta": _cliffs_delta(a, b),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["bm_q"] = bh_fdr(out["bm_p"].to_numpy())
        out["mwu_q"] = bh_fdr(out["mwu_p"].to_numpy())
    return out


def _design_matrix(df: pd.DataFrame, group_col: str, covariates: list[str]) -> tuple[np.ndarray, np.ndarray]:
    groups = pd.Categorical(df[group_col].astype(str), categories=_ordered_groups(df[group_col].unique()))
    group_dummies = pd.get_dummies(groups, drop_first=True, dtype=float)
    parts = [pd.Series(1.0, index=df.index, name="intercept"), group_dummies.set_index(df.index)]
    for cov in covariates:
        if cov not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[cov]):
            vals = pd.to_numeric(df[cov], errors="coerce")
            if vals.notna().sum() >= 2:
                parts.append(vals.fillna(vals.median()).astype(float).rename(cov))
        else:
            dummies = pd.get_dummies(df[cov].astype("category"), prefix=cov, drop_first=True, dtype=float)
            if not dummies.empty:
                parts.append(dummies.set_index(df.index))
    full = pd.concat(parts, axis=1).to_numpy(dtype=float)
    reduced = full[:, :1]
    if len(parts) > 2:
        reduced = np.concatenate([full[:, :1], full[:, 1 + group_dummies.shape[1] :]], axis=1)
    return full, reduced


def _ols_rss(y: np.ndarray, x: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return float(np.sum(resid**2))


def _group_f_stat(y: np.ndarray, full: np.ndarray, reduced: np.ndarray) -> float:
    n = len(y)
    rank_full = np.linalg.matrix_rank(full)
    rank_reduced = np.linalg.matrix_rank(reduced)
    df_num = rank_full - rank_reduced
    df_den = n - rank_full
    if df_num <= 0 or df_den <= 0:
        return np.nan
    rss_full = _ols_rss(y, full)
    rss_reduced = _ols_rss(y, reduced)
    if not np.isfinite(rss_full) or rss_full <= 0:
        return np.nan
    return float(((rss_reduced - rss_full) / df_num) / (rss_full / df_den))


def permutation_group_effect(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "group",
    covariates: list[str] | None = None,
    n_perm: int = 999,
    seed: int = 42,
) -> dict:
    covariates = covariates or [c for c in ["age", "sex", "phase"] if c in df.columns]
    keep_cols = [group_col, value_col] + [c for c in covariates if c in df.columns]
    data = df[keep_cols].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[group_col, value_col])
    if data[group_col].nunique() < 2 or len(data) < 6:
        return {"f_stat": np.nan, "p_perm": np.nan, "n": int(len(data)), "n_perm": 0}
    y = data[value_col].to_numpy(dtype=float)
    full, reduced = _design_matrix(data, group_col, covariates)
    observed = _group_f_stat(y, full, reduced)
    if not np.isfinite(observed):
        return {"f_stat": np.nan, "p_perm": np.nan, "n": int(len(data)), "n_perm": 0}
    rng = np.random.default_rng(seed)
    labels = data[group_col].to_numpy(copy=True)
    exceed = 0
    used = 0
    for _ in range(int(n_perm)):
        perm_data = data.copy()
        perm_data[group_col] = rng.permutation(labels)
        perm_full, perm_reduced = _design_matrix(perm_data, group_col, covariates)
        stat = _group_f_stat(y, perm_full, perm_reduced)
        if not np.isfinite(stat):
            continue
        used += 1
        exceed += int(stat >= observed)
    p_perm = (exceed + 1.0) / (used + 1.0) if used else np.nan
    return {
        "f_stat": float(observed),
        "p_perm": float(p_perm) if np.isfinite(p_perm) else np.nan,
        "n": int(len(data)),
        "n_perm": int(used),
        "covariates": ",".join(covariates),
    }


# --------------------------------------------------------------------------------------------------
# System-wise (network-vs-network) statistics — WITHIN-SUBJECT / paired factor.
# Each subject contributes one value per network, so networks are compared with paired/repeated-measures
# tests (Friedman omnibus, Wilcoxon signed-rank post-hoc) rather than the between-subjects group tests.
# --------------------------------------------------------------------------------------------------
def _rank_biserial_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (paired analog of Cliff's delta). Sign: a>b => positive."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if d.size == 0:
        return float("nan")
    r = stats.rankdata(np.abs(d))
    denom = r.sum()
    return float((r[d > 0].sum() - r[d < 0].sum()) / denom) if denom > 0 else float("nan")


def friedman_systemwise(wide: pd.DataFrame, min_n: int = 8) -> dict:
    """Friedman omnibus across networks (columns) on complete-case subjects (rows). `wide` = subjects x
    networks. Returns chi2, df, p, Kendall's W (concordance of the within-subject network ordering), n."""
    cols = [c for c in wide.columns]
    block = wide[cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    k = block.shape[1]
    n = block.shape[0]
    out = {"chi2": np.nan, "df": (k - 1) if k >= 1 else np.nan, "p": np.nan,
           "kendalls_w": np.nan, "n_complete": int(n), "k": int(k)}
    if k < 3 or n < min_n:
        return out
    try:
        chi2, p = stats.friedmanchisquare(*[block[c].to_numpy() for c in cols])
        out["chi2"] = float(chi2)
        out["p"] = float(p)
        out["kendalls_w"] = float(chi2 / (n * (k - 1))) if n * (k - 1) > 0 else np.nan
    except Exception:
        pass
    return out


def wilcoxon_pairwise(wide: pd.DataFrame, min_n: int = 6) -> pd.DataFrame:
    """Pairwise-complete Wilcoxon signed-rank between every pair of networks (columns), + matched-pairs
    rank-biserial effect size and median paired diff. BH-FDR across all pairs. `wide` = subjects x networks."""
    cols = list(wide.columns)
    rows = []
    for na, nb in combinations(cols, 2):
        pair = wide[[na, nb]].apply(pd.to_numeric, errors="coerce").dropna()
        a = pair[na].to_numpy(); b = pair[nb].to_numpy()
        n = a.size
        p = np.nan
        if n >= min_n and np.any(a - b != 0):
            try:
                p = float(stats.wilcoxon(a, b, zero_method="wilcox", correction=False, method="auto").pvalue)
            except Exception:
                p = np.nan
        rows.append({"net_a": na, "net_b": nb, "n": int(n),
                     "median_diff": float(np.median(a - b)) if n else np.nan,
                     "r_rb": _rank_biserial_paired(a, b), "wilcoxon_p": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["wilcoxon_q"] = bh_fdr(out["wilcoxon_p"].to_numpy())
    return out


def art_interaction(long: pd.DataFrame, value_col: str = "value",
                    group_col: str = "group", network_col: str = "network") -> dict:
    """Aligned-Rank-Transform ANOVA for the group x network interaction ('does the network profile shift by
    group?'). EXPLORATORY: aligns out both main effects, ranks, then OLS factorial on ranks — a
    between-subjects approximation that does not model the within-subject error stratum. Returns F, p."""
    out = {"art_F": np.nan, "art_p": np.nan, "n": 0}
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
        d = long[[value_col, group_col, network_col]].rename(
            columns={value_col: "y", group_col: "g", network_col: "net"}).dropna()
        if d["g"].nunique() < 2 or d["net"].nunique() < 2 or len(d) < 30:
            return out
        gm = d["y"].mean()
        d = d.copy()
        d["gmean"] = d.groupby("g")["y"].transform("mean")
        d["nmean"] = d.groupby("net")["y"].transform("mean")
        d["aligned"] = d["y"] - d["gmean"] - d["nmean"] + gm  # align for the interaction
        d["ry"] = stats.rankdata(d["aligned"])
        model = smf.ols("ry ~ C(g) * C(net)", data=d).fit()
        tbl = anova_lm(model, typ=2)
        row = [ix for ix in tbl.index if ":" in ix]
        if row:
            out["art_F"] = float(tbl.loc[row[0], "F"])
            out["art_p"] = float(tbl.loc[row[0], "PR(>F)"])
        out["n"] = int(len(d))
    except Exception:
        pass
    return out
