"""Formal mediation / compensation chains (X -> M -> Y) on the dense cohort.

X = a structural/microstructure network metric; M = a graph/coupling mediator; Y = an outcome
(diagnosis severity, brain-age gap, or ANOTHER network's metric = the 'compensation' chain).

Method: standardise X, M (and continuous Y) within the analysed sample; covary age + sex.
  path a : M ~ X + age + sex            (a = coef X)
  path b : Y ~ X + M + age + sex        (b = coef M ; c' = coef X = direct effect)
  total  : Y ~ X + age + sex            (c = coef X = total effect)
  indirect = a*b, with percentile bootstrap 95% CI (n_boot resamples); proportion mediated = a*b / c.
Continuous Y -> OLS; binary diagnosis (AD vs CN) -> logistic for path b/total (indirect on logit scale,
flagged as approximate). CROSS-SECTIONAL: associational mediation, NOT proof of causation.

Output: 21_mediation/mediation_results.csv (+ returned summary).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from connectome_analysis.analysis_config import AnalysisPaths
from connectome_analysis.analysis_plots import save_dataframe

N_BOOT = 1000
RNG = np.random.default_rng(17)


def _wide_table(paths: AnalysisPaths, master: pd.DataFrame, scheme: str = "functional") -> pd.DataFrame:
    sdir = paths.section_dir("19", "network_analysis") / scheme
    frames = []
    for fam, fname in [("micro", "network_microstructure_subject.csv"), ("graph", "network_graph_subject.csv")]:
        f = sdir / fname
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d["col"] = fam + "_" + d["network"].astype(str) + "_" + d["metric"].astype(str)
        frames.append(d.pivot_table(index="subject_id", columns="col", values="value"))
    if not frames:
        return pd.DataFrame()
    wide = pd.concat(frames, axis=1).reset_index()
    cov = master[["subject_id", "group", "age", "sex"]].drop_duplicates("subject_id")
    wide = wide.merge(cov, on="subject_id", how="left")
    ba = paths.section_dir("11", "brain_age_live") / "live_brain_age_predictions.csv"
    if ba.exists():
        b = pd.read_csv(ba)
        col = "BAG_age_corrected" if "BAG_age_corrected" in b.columns else "BAG"
        wide = wide.merge(b[["subject_id", col]].rename(columns={col: "BAG"}), on="subject_id", how="left")
    wide["sex_n"] = wide["sex"].map({"M": 1, "F": 0, "Male": 1, "Female": 0})
    wide["dx_ord"] = wide["group"].map({"CN": 0, "MCI": 1, "AD": 2})
    wide["dx_bin"] = wide["group"].map({"CN": 0, "AD": 1})  # MCI -> NaN (excluded for binary)
    return wide


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else s * 0.0


def _rankz(s: pd.Series) -> pd.Series:
    """Inverse-normal (van der Waerden) rank transform — robust to the heavy-tailed diffusivity
    distributions, so mediation captures the same monotone relationships as the rank-based group stats."""
    from scipy import stats as _ss
    x = pd.to_numeric(s, errors="coerce")
    n = int(x.notna().sum())
    if n < 3:
        return x * 0.0
    r = x.rank(method="average")
    return pd.Series(_ss.norm.ppf((r - 0.5) / n), index=s.index)


def _fit_paths(d: pd.DataFrame, binary: bool):
    """Return (a, b, c, cprime) for one resample; None on failure."""
    import statsmodels.api as sm
    Xc = sm.add_constant(d[["X", "age", "sex_n"]], has_constant="add")
    try:
        a = sm.OLS(d["M"], Xc).fit().params["X"]
        XMc = sm.add_constant(d[["X", "M", "age", "sex_n"]], has_constant="add")
        if binary:
            mb = sm.Logit(d["Y"], XMc).fit(disp=0)
            mt = sm.Logit(d["Y"], Xc).fit(disp=0)
        else:
            mb = sm.OLS(d["Y"], XMc).fit()
            mt = sm.OLS(d["Y"], Xc).fit()
        return float(a), float(mb.params["M"]), float(mt.params["X"]), float(mb.params["X"])
    except Exception:
        return None


def _mediate(df: pd.DataFrame, X: str, M: str, Y: str, binary: bool, n_boot: int = N_BOOT) -> dict | None:
    cols = [X, M, Y, "age", "sex_n"]
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna().rename(columns={X: "X", M: "M", Y: "Y"})
    if len(d) < 40 or d["X"].nunique() < 5 or d["M"].nunique() < 5:
        return None
    d = d.copy()
    # rank-based (inverse-normal) transform so linear-model mediation matches the rank-based group stats
    d["X"] = _rankz(d["X"]); d["M"] = _rankz(d["M"])
    if not binary:
        d["Y"] = _rankz(d["Y"])
    d["age"] = _rankz(d["age"])
    base = _fit_paths(d, binary)
    if base is None:
        return None
    a, b, c, cprime = base
    indirect = a * b
    boots = []
    n = len(d)
    for _ in range(n_boot):
        bd = d.iloc[RNG.integers(0, n, n)]
        r = _fit_paths(bd, binary)
        if r is not None:
            boots.append(r[0] * r[1])
    boots = np.array(boots)
    lo, hi = (np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)) if boots.size else (np.nan, np.nan)
    prop = indirect / c if c and np.isfinite(c) and abs(c) > 1e-9 else np.nan
    return {"n": int(n), "a": a, "b": b, "total_c": c, "direct_cprime": cprime,
            "indirect_ab": indirect, "ci_lo": float(lo), "ci_hi": float(hi),
            "sig": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
            "prop_mediated": float(prop) if np.isfinite(prop) else np.nan,
            "outcome_type": "binary(logit)" if binary else "continuous"}


# (name, X, M, Y, kind) — Y in {'dx_bin','BAG', or a wide metric col for compensation}
CHAINS = [
    ("Limbic MD -> Limbic efficiency -> AD diagnosis", "micro_Limbic_md_mean", "graph_Limbic_nodal_eff", "dx_bin", "disease"),
    ("Limbic MD -> Limbic efficiency -> brain-age gap", "micro_Limbic_md_mean", "graph_Limbic_nodal_eff", "BAG", "disease"),
    ("DMN FA -> DMN efficiency -> AD diagnosis", "micro_DMN_fa_mean", "graph_DMN_nodal_eff", "dx_bin", "disease"),
    ("Visual MD -> Visual efficiency -> AD diagnosis", "micro_Visual_md_mean", "graph_Visual_nodal_eff", "dx_bin", "disease"),
    ("Limbic MD -> Limbic efficiency -> Visual efficiency (compensation)", "micro_Limbic_md_mean", "graph_Limbic_nodal_eff", "graph_Visual_nodal_eff", "compensation"),
    ("Limbic MD -> Limbic efficiency -> Frontoparietal efficiency (compensation)", "micro_Limbic_md_mean", "graph_Limbic_nodal_eff", "graph_Frontoparietal_nodal_eff", "compensation"),
    ("DMN MD -> DMN efficiency -> Somatomotor efficiency (compensation)", "micro_DMN_md_mean", "graph_DMN_nodal_eff", "graph_Somatomotor_nodal_eff", "compensation"),
]


def run_mediation(paths: AnalysisPaths, master: pd.DataFrame) -> dict:
    out_dir = paths.section_dir("21", "mediation")
    out_dir.mkdir(parents=True, exist_ok=True)
    wide = _wide_table(paths, master)
    rows = []
    for name, X, M, Y, kind in CHAINS:
        if not {X, M}.issubset(wide.columns) or (Y not in wide.columns):
            continue
        binary = (Y == "dx_bin")
        res = _mediate(wide, X, M, Y, binary)
        if res is None:
            continue
        ynice = {"dx_bin": "AD vs CN diagnosis", "BAG": "brain-age gap"}.get(Y, Y.replace("graph_", "").replace("_", " "))
        verdict = "significant indirect (mediation)" if res["sig"] else "no significant mediation"
        statement = (f"{name}: indirect a·b={res['indirect_ab']:+.3f} (95% CI {res['ci_lo']:+.3f}..{res['ci_hi']:+.3f}), "
                     f"{verdict}; direct={res['direct_cprime']:+.3f}, total={res['total_c']:+.3f}, "
                     f"prop. mediated={res['prop_mediated']:.0%} (n={res['n']}, {res['outcome_type']}).")
        rows.append({"chain": name, "kind": kind, "X": X, "M": M, "Y": ynice, **res, "statement": statement})
    out = pd.DataFrame(rows)
    save_dataframe(out, out_dir / "mediation_results.csv")
    return {"n_chains": int(len(out)), "n_sig": int(out["sig"].sum()) if not out.empty else 0,
            "path": str(out_dir / "mediation_results.csv")}


if __name__ == "__main__":
    from connectome_analysis.analysis_config import get_analysis_paths
    PR = Path("/home/ec2-user/exp")
    p = get_analysis_paths(notebook_dir=PR, deriv_root=PR / "data" / "derivatives", cohort_dti_csv=PR / "cohort" / "dti.csv").ensure()
    m = pd.read_csv(p.master_dir / "master_cohort.csv")
    print(run_mediation(p, m))
