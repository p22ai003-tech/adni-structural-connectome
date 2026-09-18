"""Findings catalog: harvest every FDR-significant network-level result (both schemes) + an age-effect
scan, and template each into a plain-English statement + literature search terms. This is the input to the
literature-retrieval + novelty layer.

Output: ANALYSIS_ROOT/20_findings/findings_catalog.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from connectome_analysis.analysis_config import AnalysisPaths
from connectome_analysis.analysis_plots import save_dataframe
from connectome_analysis.analysis_stats import bh_fdr

Q_THRESH = 0.05

NETWORK_PHRASE = {
    "DMN": "default mode network", "Limbic": "limbic system", "Salience_VAN": "salience network",
    "DorsalAttention": "dorsal attention network", "Frontoparietal": "frontoparietal control network",
    "Somatomotor": "sensorimotor network", "Visual": "visual network", "Subcortical": "subcortical structures",
    "Cerebellar": "cerebellum", "Brainstem": "brainstem nuclei",
    "Frontal": "frontal lobe", "Parietal": "parietal lobe", "Temporal": "temporal lobe",
    "Occipital": "occipital lobe", "BasalGanglia": "basal ganglia", "Thalamus": "thalamus",
    "Cerebellum": "cerebellum",
}
METRIC_PHRASE = {
    "fa_mean": "fractional anisotropy", "md_mean": "mean diffusivity",
    "ad_mean": "axial diffusivity", "rd_mean": "radial diffusivity",
    "strength": "nodal strength", "degree": "nodal degree", "nodal_eff": "nodal efficiency",
    "within": "within-network connectivity", "between": "between-network connectivity",
}
BASE_TERMS = ["Alzheimer's disease", "diffusion MRI", "structural connectome", "tractography"]


def _metric_phrase(metric: str) -> str:
    if metric in METRIC_PHRASE:
        return METRIC_PHRASE[metric]
    if "__" in metric:  # coupling pair, e.g. strength__fa_mean
        a, b = metric.split("__", 1)
        return f"structure-function coupling ({METRIC_PHRASE.get(a, a)} vs {METRIC_PHRASE.get(b, b)})"
    return metric.replace("_", " ")


def _fid(*parts: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", "_".join(str(p) for p in parts)).strip("_")


def _search_terms(network: str, metric: str, family: str, age: bool = False) -> str:
    net = NETWORK_PHRASE.get(network, network.replace("_", " "))
    terms = [net, _metric_phrase(metric)] + BASE_TERMS
    if family == "connectivity":
        terms.append("network disconnection")
    if family == "coupling":
        terms.append("structure-function coupling")
    if age:
        terms += ["brain aging", "age-related"]
    # de-dup, keep order
    seen, out = set(), []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
    return "; ".join(out)


def _direction(delta: float) -> str:
    if not np.isfinite(delta):
        return "differs"
    return "lower in AD" if delta > 0 else "higher in AD"


def _harvest_affectedness(paths: AnalysisPaths) -> pd.DataFrame:
    rows = []
    for scheme in ("functional", "anatomical"):
        f = paths.section_dir("19", "network_analysis") / scheme / "network_affectedness_summary.csv"
        if not f.exists():
            continue
        aff = pd.read_csv(f)
        sig = aff[(aff["q_kw"] < Q_THRESH) & aff["cn_ad_cliffs_delta"].notna()].copy()
        for r in sig.itertuples():
            net = str(getattr(r, "network", "")); metric = str(getattr(r, "metric", ""))
            fam = str(r.feature_family); delta = float(r.cn_ad_cliffs_delta)
            label = f"{NETWORK_PHRASE.get(net, net)} {_metric_phrase(metric)}"
            rows.append({
                "finding_id": _fid(scheme, fam, net, metric),
                "scheme": scheme, "feature_family": fam, "network": net, "metric": metric,
                "n": int(getattr(r, "n", 0)),
                "mean_CN": getattr(r, "mean_CN", np.nan), "mean_MCI": getattr(r, "mean_MCI", np.nan),
                "mean_AD": getattr(r, "mean_AD", np.nan),
                "effect_cliffs_delta": round(delta, 4), "q": float(r.q_kw),
                "direction": _direction(delta),
                "plain_statement": f"{label} is {_direction(delta)} (Cliff's d={delta:+.2f}, q={float(r.q_kw):.2g}).",
                "search_terms": _search_terms(net, metric, fam),
            })
    return pd.DataFrame(rows)


def _interaction_p(sub: pd.DataFrame) -> float:
    """Age x group interaction p from OLS value ~ age * C(group) (joint test of interaction terms)."""
    try:
        import statsmodels.formula.api as smf
        d = sub.rename(columns={"value": "y"})[["y", "age", "group"]].dropna()
        if d["group"].nunique() < 2 or len(d) < 30:
            return float("nan")
        m = smf.ols("y ~ age * C(group)", data=d).fit()
        terms = [t for t in m.params.index if t.startswith("age:C(group)")]
        if not terms:
            return float("nan")
        return float(m.f_test(" , ".join(f"{t} = 0" for t in terms)).pvalue)
    except Exception:
        return float("nan")


def _age_scan(paths: AnalysisPaths, master: pd.DataFrame) -> pd.DataFrame:
    cov = master[["subject_id", "age"]].drop_duplicates("subject_id")
    rows = []
    for scheme in ("functional", "anatomical"):
        sdir = paths.section_dir("19", "network_analysis") / scheme
        for fam, fname in [("microstructure", "network_microstructure_subject.csv"),
                           ("graph", "network_graph_subject.csv"),
                           ("coupling", "network_coupling_subject.csv")]:
            f = sdir / fname
            if not f.exists():
                continue
            df = pd.read_csv(f).merge(cov, on="subject_id", how="left")
            recs = []
            for (net, metric), sub in df.groupby(["network", "metric"]):
                v = sub[["value", "age", "group"]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(v) < 20 or v["value"].nunique() < 3:
                    continue
                rho, p = stats.spearmanr(v["value"], v["age"])
                rec = {"scheme": scheme, "feature_family": "age", "network": net, "metric": metric,
                       "n": int(len(v)), "rho": float(rho), "p": float(p)}
                for g in ("CN", "MCI", "AD"):
                    gv = v[v["group"] == g]
                    rec[f"rho_{g}"] = float(stats.spearmanr(gv["value"], gv["age"]).correlation) if len(gv) >= 10 and gv["value"].nunique() > 2 else np.nan
                rec["interaction_p"] = _interaction_p(v)
                recs.append(rec)
            r = pd.DataFrame(recs)
            if r.empty:
                continue
            r["q"] = bh_fdr(r["p"].to_numpy())
            r["interaction_q"] = bh_fdr(r["interaction_p"].to_numpy())
            for x in r[r["q"] < Q_THRESH].itertuples():
                net = x.network; metric = x.metric
                trend = "increases with age" if x.rho > 0 else "decreases with age"
                label = f"{NETWORK_PHRASE.get(net, net)} {_metric_phrase(metric)}"
                grp_bits = ", ".join(f"{g} {getattr(x, f'rho_{g}'):+.2f}" for g in ("CN", "MCI", "AD")
                                     if pd.notna(getattr(x, f"rho_{g}")))
                inter = (f"; age×group interaction q={x.interaction_q:.2g}"
                         if pd.notna(x.interaction_q) and x.interaction_q < Q_THRESH else "")
                rows.append({
                    "finding_id": _fid(scheme, "age", net, metric),
                    "scheme": scheme, "feature_family": "age", "network": net, "metric": metric,
                    "n": int(x.n), "mean_CN": np.nan, "mean_MCI": np.nan, "mean_AD": np.nan,
                    "effect_cliffs_delta": round(float(x.rho), 4), "q": float(x.q),
                    "rho_CN": getattr(x, "rho_CN"), "rho_MCI": getattr(x, "rho_MCI"), "rho_AD": getattr(x, "rho_AD"),
                    "interaction_p": x.interaction_p, "interaction_q": x.interaction_q,
                    "direction": trend,
                    "plain_statement": f"{label} {trend} (cohort rho={x.rho:+.2f}, q={x.q:.2g}); by group: {grp_bits}{inter}.",
                    "search_terms": _search_terms(net, metric, "age", age=True),
                })
    return pd.DataFrame(rows)


def build_findings_catalog(paths: AnalysisPaths, master: pd.DataFrame) -> dict:
    out_dir = paths.section_dir("20", "findings")
    out_dir.mkdir(parents=True, exist_ok=True)
    cat = pd.concat([_harvest_affectedness(paths), _age_scan(paths, master)], ignore_index=True)
    if not cat.empty:
        cat = cat.sort_values(["feature_family", "q"]).reset_index(drop=True)
    save_dataframe(cat, out_dir / "findings_catalog.csv")
    by_fam = cat["feature_family"].value_counts().to_dict() if not cat.empty else {}
    return {"n_findings": int(len(cat)), "by_family": by_fam, "path": str(out_dir / "findings_catalog.csv")}
