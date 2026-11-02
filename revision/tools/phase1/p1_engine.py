"""Phase 1 step 3 — re-estimate the five core findings under the confound-control conditions.

Since 2026-10-24 every condition except C0 runs on the quality-controlled data written by p1_audit.py: participants
with qc_fail = 1 are excluded (phase1/qc.csv) and physically impossible single values are missing
(features_qc.csv, features_consistent_edges_qc.csv). C0 stays the published analysis, as the reference.

Usage: python p1_engine.py B [C0,C2,...]  ->  effects_long_<conds>.csv etc.; p1_merge.py assembles them.

Findings (as in the manuscript):
  F1  networks (of 10) in which each diffusion measure separates groups, q < 0.05     (Table 2)
  F2  leading whole-brain effects                                                     (Table 4, row 1)
  F3  short- vs long-range effects: mean |delta| over 40 cells, SR > LR cell count,
      and the SR - LR difference with a subject-level bootstrap 95% interval          (Table 3, Fig. 4)
  F4  leading exception-edge effects                                                  (Table 4, row 3)
  F5  coverage of the 51-edge consensus core                                          (Table 6)

Effect size: Cliff's delta, delta(first, second) = P(first > second) - P(first < second), so a negative value
means higher in the later-stage group (manuscript convention). Tests: Brunner-Munzel. BH families as in the
paper: whole-brain = the three contrasts of each network x measure; tract-length and exception = the ten
networks of each measure within a contrast; coverage = its three contrasts.

Covariate adjustment keeps group differences and removes covariate effects: value ~ group + covariates is fitted
on the analysis sample and only the covariate terms are subtracted (the approach used for route control).
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import iit_chain as chain; chain.boot()      # IIT_ROOT and the chain settings; no effect when IIT_ROOT is unset
import os
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

OUT = Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1"))
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rng = np.random.default_rng(20261024)
cov = pd.read_csv(OUT / "covariates.csv").set_index("subject_id")
qcd = pd.read_csv(OUT / "qc.csv", index_col=0).reindex(cov.index)
cov["qc_fail"], cov["route"] = qcd.qc_fail, qcd.route
cov["route_r4"], cov["route_step7"] = (cov.route == "legacy_r4").astype(float), (cov.route == "legacy_step7").astype(float)
cov["log_coverage"] = np.log(qcd.md_coverage + 0.01)       # share of structural edges that carry microstructure
pub = pd.read_csv(OUT / "features_published.csv", index_col=0)
pubq = pd.read_csv(OUT / "features_qc.csv", index_col=0)
conq = pd.read_csv(OUT / "features_consistent_edges_qc.csv", index_col=0)
CONTRASTS = [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]
MEAS = ["FA", "MD", "RD", "AxD"]


def family(feat):
    kind, net, meas = feat.split("|")
    return kind, net, meas


# ------------------------------------------------------------------ statistics
def cliff(a, b):
    """delta(a, b) via ranks (ties count half), O(n log n)."""
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    r = rankdata(np.concatenate([a, b]))
    u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2          # pairs a > b, ties = 0.5
    return 2 * u / (len(a) * len(b)) - 1


def bm(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    with np.errstate(all="ignore"):
        p = stats.brunnermunzel(a, b, alternative="two-sided").pvalue
    return float(p) if np.isfinite(p) else np.nan


def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); ok = np.isfinite(p)
    if ok.sum():
        pv = p[ok]; o = np.argsort(pv); n = len(pv)
        r = pv[o] * n / np.arange(1, n + 1); r = np.minimum.accumulate(r[::-1])[::-1]
        out = np.empty(n); out[o] = np.minimum(r, 1); q[ok] = out
    return q


def adjust(X, groups, Z):
    """Subtract covariate effects estimated jointly with group (per feature, complete cases)."""
    if Z is None:
        return X
    Zc = Z - Z.mean()
    G = pd.get_dummies(groups, drop_first=True).astype(float)
    out = X.copy()
    for c in X.columns:
        ok = X[c].notna() & Zc.notna().all(1)
        if ok.sum() < 20:
            out[c] = np.nan; continue
        D = np.column_stack([np.ones(ok.sum()), G[ok].values, Zc[ok].values])
        beta, *_ = np.linalg.lstsq(D, X.loc[ok, c].values, rcond=None)
        bz = beta[1 + G.shape[1]:]
        out[c] = np.where(ok, X[c].values - (Zc.fillna(0).values @ bz), np.nan)
    return out


def combat(X, cv, batch_col="vendor", cat=("group", "female"), cont=("age", "education")):
    from neuroCombat import neuroCombat
    batches = cv[batch_col]
    usable = [c for c in X.columns
              if all(X.loc[batches == b, c].notna().sum() >= 3 and X.loc[batches == b, c].std() > 0 for b in batches.unique())]
    Xi = X[usable].apply(lambda s: s.fillna(s.median()))
    covars = cv[[batch_col, *cat, *cont]].copy()
    res = neuroCombat(dat=Xi.T.values, covars=covars, batch_col=batch_col,
                      categorical_cols=list(cat), continuous_cols=list(cont))["data"].T
    H = X.copy()
    H[usable] = np.where(X[usable].notna(), res, np.nan)
    return H, len(usable), X.shape[1]


def propensity_match(cv, a, b, ratio):
    """Greedy nearest-neighbour matching of each `b` (smaller, later-stage) to `ratio` of `a`, caliper 0.2 SD logit."""
    d = cv[cv.group.isin([a, b])].dropna(subset=["age", "female", "education"]).copy()
    Xp = pd.concat([d[["age", "female", "education"]], pd.get_dummies(d.vendor, drop_first=True).astype(float)], axis=1)
    y = (d.group == b).astype(int)
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(Xp, y)
    d["logit"] = lr.decision_function(Xp)
    cal = 0.2 * d.logit.std()
    pool = d[d.group == a].copy(); keep_b, keep_a = [], []
    for sid, row in d[d.group == b].sample(frac=1, random_state=1).iterrows():
        dist = (pool.logit - row.logit).abs()
        cand = dist[dist <= cal].nsmallest(ratio)
        if len(cand) == ratio:
            keep_b.append(sid); keep_a += list(cand.index); pool = pool.drop(cand.index)
    m = d.loc[keep_a + keep_b]
    smd = {}
    for c in ["age", "female", "education"]:
        x1, x0 = m.loc[m.group == b, c], m.loc[m.group == a, c]
        smd[c] = float((x1.mean() - x0.mean()) / np.sqrt((x1.var() + x0.var()) / 2))
    return m.index, smd


# ------------------------------------------------------------------ one condition
def effects(X, g, tag):
    rows = []
    for a, b in CONTRASTS:
        A, Bm = g == a, g == b
        for c in X.columns:
            xa, xb = X.loc[A, c].values.astype(float), X.loc[Bm, c].values.astype(float)
            rows.append(dict(condition=tag, feature=c, contrast=f"{a}-{b}", delta=cliff(xa, xb), p=bm(xa, xb),
                             n_first=int(np.isfinite(xa).sum()), n_second=int(np.isfinite(xb).sum())))
    df = pd.DataFrame(rows)
    df[["kind", "network", "measure"]] = df.feature.str.split("|", expand=True)
    df["q"] = np.nan
    wb = df.kind == "WB"
    for f, idx in df[wb].groupby("feature").groups.items():
        df.loc[idx, "q"] = bh(df.loc[idx, "p"])
    for (k, meas, ct), idx in df[df.kind.isin(["TR", "EX"])].groupby(["kind", "measure", "contrast"]).groups.items():
        df.loc[idx, "q"] = bh(df.loc[idx, "p"])
    cv = df.kind == "COV"
    df.loc[cv, "q"] = bh(df.loc[cv, "p"])
    return df


def sr_lr_cells(X, g, a, b, kind="TR"):
    pref = ("SR-", "LR-") if kind == "TR" else ("SRexc-", "LRexc-")
    A, Bm = (g == a).values, (g == b).values
    d_sr, d_lr = [], []
    for net in sorted({c.split("|")[1] for c in X.columns if c.startswith(kind + "|")}):
        for m in MEAS:
            s, l = f"{kind}|{net}|{pref[0]}{m}", f"{kind}|{net}|{pref[1]}{m}"
            if s in X and l in X:
                ds = cliff(X[s].values[A], X[s].values[Bm]); dl = cliff(X[l].values[A], X[l].values[Bm])
                if np.isfinite(ds) and np.isfinite(dl):
                    d_sr.append(abs(ds)); d_lr.append(abs(dl))
    return np.array(d_sr), np.array(d_lr)


def sr_lr_summary(X, g, tag, kind="TR"):
    out = []
    for a, b in CONTRASTS:
        s, l = sr_lr_cells(X, g, a, b, kind)
        idx_a = np.flatnonzero((g == a).values); idx_b = np.flatnonzero((g == b).values)
        boots = []
        sub = X.iloc[np.concatenate([idx_a, idx_b])]; gg = g.iloc[np.concatenate([idx_a, idx_b])]
        ia = np.flatnonzero((gg == a).values); ib = np.flatnonzero((gg == b).values)
        for _ in range(B):
            pick = np.concatenate([rng.choice(ia, len(ia)), rng.choice(ib, len(ib))])
            bs, bl = sr_lr_cells(sub.iloc[pick].reset_index(drop=True), gg.iloc[pick].reset_index(drop=True), a, b, kind)
            boots.append(bs.mean() - bl.mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append(dict(condition=tag, kind=kind, contrast=f"{a}-{b}", n_cells=len(s), sr_mean=s.mean(), lr_mean=l.mean(),
                        diff=s.mean() - l.mean(), ci_lo=lo, ci_hi=hi, sr_gt_lr=int((s > l).sum()),
                        n_first=int((g == a).sum()), n_second=int((g == b).sum())))
    return out


# ------------------------------------------------------------------ conditions
COVS = ["age", "female", "education"]
qc = cov.qc_fail == 0
a34 = (chain.primary(cov) == 1) & qc          # primary sample after quality control (IIT_PRIMARY_SAMPLE; default ADNI-3/4)
PL = chain.sample_label()
conds = []
inc = (qcd.in_cohort == 1) if "in_cohort" in qcd else pd.Series(True, index=cov.index)   # analysis-tree cohort
conds.append(("C0 Pooled 530 (published)" if os.environ.get("IIT_SRC", "published") == "published" else "C0 Pooled, rebuilt connectomes",
              cov.index[inc], pub, None, None))
conds.append(("C0q Pooled after QC", cov.index[qc], pubq, None, None))
conds.append((f"C1 {PL}", cov.index[a34], pubq, None, None))
conds.append((f"C2 {PL} + age, sex, education", cov.index[a34], pubq, COVS, None))
conds.append(("C3 + vendor ComBat", cov.index[a34], pubq, COVS, "combat"))
conds.append(("C4 Propensity-matched", cov.index[a34], pubq, None, "match"))
conds.append(("C5 + MTL volume + WMH", cov.index[a34], pubq, COVS + ["mtl_ratio", "wmh_log"], None))
X6 = pubq.copy(); X6.update(conq)             # consistent-edge WB/TR values; EX and coverage unchanged
conds.append(("C6 Consistent edges", cov.index[a34], X6, COVS, None))
conds.append(("C7 Amyloid-defined groups", cov.index[a34], pubq, COVS, "amyloid"))
conds.append(("C8 Siemens only", cov.index[a34 & (cov.vendor == "Siemens")], pubq, COVS, None))
# processing checks added after the data audit: route and microstructure coverage differ between groups
conds.append(("C9 + processing route", cov.index[a34], pubq, COVS + ["route_r4", "route_step7"], None))
conds.append(("C10 + microstructure coverage", cov.index[a34], pubq, COVS, "coverage"))
conds.append(("C11 Coverage at least 25%", cov.index[a34 & (qcd.md_coverage >= 0.25)], pubq, COVS, None))
SEL = sys.argv[2].split(",") if len(sys.argv) > 2 else [c[0].split(" ", 1)[0] for c in conds]
conds = [c for c in conds if c[0].split(" ", 1)[0] in SEL]
SUFFIX = "" if len(sys.argv) <= 2 else "_" + "_".join(SEL)

all_eff, all_srlr, notes = [], [], {}
for tag, idx, X, covs, special in conds:
    cv = cov.loc[idx].copy(); Xc = X.reindex(idx)
    if special == "amyloid":
        keep = ((cv.group == "CN") & (cv.amyloid_pos == 0)) | ((cv.group.isin(["MCI", "AD"])) & (cv.amyloid_pos == 1))
        cv, Xc = cv[keep], Xc[keep]
    if covs:
        ok = cv[covs].notna().all(1); cv, Xc = cv[ok], Xc[ok]
    g = cv.group
    # published inclusion rule: a feature enters only if every group has >= 5 non-missing values
    elig = [c for c in Xc.columns if min(Xc.loc[g == k, c].notna().sum() for k in ("CN", "MCI", "AD")) >= 5]
    notes[tag + " features"] = f"{len(elig)} of {Xc.shape[1]} eligible (>= 5 per group)"
    Xc = Xc[elig]
    if special == "combat":
        Xc, nh, nt = combat(Xc, cv); notes[tag] = f"ComBat harmonised {nh} of {nt} features (vendor batches {cv.vendor.value_counts().to_dict()})"
    Xa = adjust(Xc, g, cv[covs]) if covs else Xc
    if special == "coverage":                 # microstructure features also adjusted for coverage; core coverage is structural
        micro = [c for c in Xa.columns if not c.startswith("COV|")]
        ok = cv.log_coverage.notna(); cv, g = cv[ok], g[ok]
        Xa = Xa.loc[ok].copy(); Xa[micro] = adjust(Xc.loc[ok, micro], g, cv[covs + ["log_coverage"]])
    if special == "match":
        eff_parts, sr_parts, bal = [], [], {}
        for (a, b), ratio in zip(CONTRASTS, [1, 2, 2]):
            ids, smd = propensity_match(cv, a, b, ratio)
            gm = cv.loc[ids, "group"]; Xm = Xa.loc[ids]
            e = effects(Xm, gm, tag); e = e[e.contrast == f"{a}-{b}"]; eff_parts.append(e)
            s = [r for r in sr_lr_summary(Xm, gm, tag) if r["contrast"] == f"{a}-{b}"]
            s += [r for r in sr_lr_summary(Xm, gm, tag, "EX") if r["contrast"] == f"{a}-{b}"]
            sr_parts += s; bal[f"{a}-{b}"] = dict(n={k: int(v) for k, v in gm.value_counts().items()}, smd={k: round(v, 3) for k, v in smd.items()})
        all_eff.append(pd.concat(eff_parts)); all_srlr += sr_parts; notes[tag] = bal
    else:
        all_eff.append(effects(Xa, g, tag))
        all_srlr += sr_lr_summary(Xa, g, tag) + sr_lr_summary(Xa, g, tag, "EX")
    notes.setdefault(tag + " n", {k: int(v) for k, v in g.value_counts().items()})
    print(f"done: {tag}  n={dict(g.value_counts())}", flush=True)

eff = pd.concat(all_eff, ignore_index=True); srlr = pd.DataFrame(all_srlr)
eff.to_csv(OUT / f"effects_long{SUFFIX}.csv", index=False); srlr.to_csv(OUT / f"sr_lr_summary{SUFFIX}.csv", index=False)
json.dump(notes, open(OUT / f"condition_notes{SUFFIX}.json", "w"), indent=1, default=str)
print("wrote effects_long.csv, sr_lr_summary.csv, condition_notes.json")
