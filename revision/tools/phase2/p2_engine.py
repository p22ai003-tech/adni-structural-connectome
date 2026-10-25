"""Phase 2 — a direct test of the length claim, on edge-level data.

Three analyses, each under the conditions below:
  A. effect size as a continuous function of edge length: participants' mean FA/MD/RD/AxD within ten length bins
     (deciles of CN edge length), Cliff's delta per bin, subject-level bootstrap bands;
  B. length dependence, two ways. (i) PRIMARY, rank-based: the effect e_k in bin k (= -delta for MD, RD, AxD and +delta
     for FA, so e > 0 always means 'abnormal in the later group') is regressed on log2 bin length across the ten bins;
     the slope I_delta (change in effect per doubling of length) is > 0 when long edges are more affected. 95% interval
     from the subject-level bootstrap, p from 2,000 permutations of group labels.
     (ii) the diagnosis x length interaction of a mean-based model: bin values standardised to CN, each participant's
     slope on log2 length (SD per doubling of length); I = s * (mean slope, later group - mean slope, earlier group),
     s = +1 for MD, RD, AxD and -1 for FA, so I > 0 means the disease effect grows with length (long-range edges more
     affected) and I < 0 that it is larger on short edges. p from 2,000 permutations of group labels, 95% interval
     from 2,000 subject-level bootstrap resamples; a linear mixed model (random intercept and slope) is fitted once
     for reference;
  C. the published short- minus long-range statistic (mean |delta| over 40 network x measure cells) under four
     definitions of short and long: quartiles (published), terciles, quintiles and fixed 70 / 140 mm.
Regions: whole brain (all edges) and limbic (edges with at least one limbic node).

Conditions (option 2 of 24 Oct: coverage handled by checks, not by rebuilding):
  P0  pooled 530, published data, unadjusted                       (reference; reproduces the published TR values)
  P2  ADNI-3/4 after quality control, adjusted for age, sex, education   (primary)
  P10 P2 + log microstructure coverage
  P11 P2 restricted to microstructure coverage >= 25%

Usage: python p2_engine.py <source> <condition> [B]     e.g.  p2_engine.py published P2 2000
"""
import os
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
import warnings; warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning; warnings.simplefilter("ignore", ConvergenceWarning)

SRC, COND = sys.argv[1], sys.argv[2]; B = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
O = Path(os.environ.get("IIT_P2", "/home/ec2-user/exp/docs/iit_dipanjan/phase2")); R = O / "results"; R.mkdir(exist_ok=True)
P1 = Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1"))
rng = np.random.default_rng(20261025)
E = np.load(O / f"edges_{SRC}.npz", allow_pickle=True)
cov = pd.read_csv(P1 / "covariates.csv").set_index("subject_id").loc[E["sid"]]
qc = pd.read_csv(P1 / "qc.csv", index_col=0).reindex(cov.index)
MEAS = {"FA": "fa", "MD": "md", "RD": "rd", "AxD": "ad"}; SIGN = {"FA": -1, "MD": 1, "RD": 1, "AxD": 1}
CONTRASTS = [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]
COVS = ["age", "female", "education"]
NETS = ["Visual", "Somatomotor", "DorsalAttention", "Salience_VAN", "Limbic", "Frontoparietal", "DMN", "Subcortical", "Cerebellar", "Brainstem"]
L = E["len"].astype(float); ni, nj = E["net_i"], E["net_j"]; ei, ej = E["i"], E["j"]

# ---------------- condition: participants and covariates
a34 = (cov.adni34 == 1) & (qc.qc_fail == 0)
covs = None
if COND == "P0":   keep = (qc.in_cohort == 1).values if "in_cohort" in qc else np.ones(len(cov), bool)
elif COND == "P2": keep = a34.values; covs = COVS
elif COND == "P10": keep = (a34 & qc.md_coverage.notna()).values; covs = COVS + ["log_coverage"]
elif COND == "P11": keep = (a34 & (qc.md_coverage >= 0.25)).values; covs = COVS
else: raise SystemExit(COND)
cv = cov[keep].copy(); cv["log_coverage"] = np.log(qc.md_coverage[keep] + 0.01)
if covs: cv = cv.dropna(subset=covs)
rows = np.flatnonzero(np.isin(cov.index, cv.index)); g = cv.group.values
VAL = {m: E[k][rows].astype(float) for m, k in MEAS.items()}
if COND != "P0":                       # quality control: physically impossible single edge values set to missing
    for m in VAL: VAL[m][(VAL[m] > 3e-3) if m != "FA" else ((VAL["FA"] > 1) | (VAL["FA"] < 0.05))] = np.nan
LEN = L[rows]


def adjust(X, Z):
    """value ~ group + covariates per column; subtract the covariate part (as in Phase 1)."""
    if Z is None: return X
    Zc = (Z - Z.mean(0)); G = pd.get_dummies(pd.Series(g), drop_first=True).values.astype(float); out = X.copy()
    for c in range(X.shape[1]):
        ok = np.isfinite(X[:, c])
        if ok.sum() < 20: out[:, c] = np.nan; continue
        D = np.column_stack([np.ones(ok.sum()), G[ok], Zc[ok]])
        beta = np.linalg.lstsq(D, X[ok, c], rcond=None)[0]
        out[ok, c] = X[ok, c] - Zc[ok] @ beta[1 + G.shape[1]:]
    return out


Z = cv[covs].values.astype(float) if covs else None


def cliff_cols(A, Bm):
    """Cliff's delta for every column, NaN-aware: delta(A, B) = P(A > B) - P(A < B)."""
    out = np.full(A.shape[1], np.nan)
    for c in range(A.shape[1]):
        a, b = A[:, c], Bm[:, c]; a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 5 or len(b) < 5: continue
        r = rankdata(np.concatenate([a, b])); u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2
        out[c] = 2 * u / (len(a) * len(b)) - 1
    return out


# ---------------- A and B: length bins
cn_all = cov.group.values == "CN"
pool = L[cn_all & (a34.values if COND != "P0" else True)]
edges_q = np.nanpercentile(pool[np.isfinite(pool)], np.linspace(0, 100, 11)); edges_q[0], edges_q[-1] = 0, np.inf
centres = np.array([np.nanmedian(pool[(pool > edges_q[k]) & (pool <= edges_q[k + 1])]) for k in range(10)])
REGIONS = {"Whole brain": np.ones(len(ei), bool), "Limbic": (ni == "Limbic") | (nj == "Limbic")}
binvals = {}                                                  # (region, measure) -> participants x 10
for reg, emask in REGIONS.items():
    for m in MEAS:
        X = np.full((len(rows), 10), np.nan)
        for k in range(10):
            sel = emask & True
            inb = (LEN > edges_q[k]) & (LEN <= edges_q[k + 1]) & sel[None, :]
            v = np.where(inb, VAL[m], np.nan)
            cntk = np.isfinite(v).sum(1); X[:, k] = np.where(cntk >= 3, np.nansum(v, 1) / np.maximum(cntk, 1), np.nan)
        binvals[(reg, m)] = adjust(X, Z)

curves, inter = [], []
lx = np.log2(centres); lx = lx - lx.mean()
for reg in REGIONS:
    for m in MEAS:
        X = binvals[(reg, m)]
        cnm = X[g == "CN"]; mu, sd = np.nanmean(cnm, 0), np.nanstd(cnm, 0, ddof=1)
        Zs = (X - mu) / sd
        # per-participant slope on centred log2 length (>= 5 bins)
        ok = np.isfinite(Zs); nb = ok.sum(1)
        xm = np.where(ok, lx[None, :], np.nan)
        xbar = np.nanmean(xm, 1); ybar = np.nanmean(Zs, 1)
        slope = np.nansum((xm - xbar[:, None]) * (Zs - ybar[:, None]), 1) / np.nansum((xm - xbar[:, None]) ** 2, 1)
        slope[nb < 5] = np.nan
        for a, b in CONTRASTS:
            ia, ib = np.flatnonzero(g == a), np.flatnonzero(g == b)
            d = cliff_cols(X[ia], X[ib]); boots = np.empty((B, 10))
            for t in range(B):
                boots[t] = cliff_cols(X[rng.choice(ia, len(ia))], X[rng.choice(ib, len(ib))])
            lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
            def eslope(dv):
                e = -SIGN[m] * dv; okb = np.isfinite(e)
                return np.polyfit(lx[okb], e[okb], 1)[0] if okb.sum() >= 5 else np.nan
            Id = eslope(d); Id_boot = np.array([eslope(bb) for bb in boots])
            pool_ab = np.concatenate([ia, ib]); Id_perm = np.empty(B)
            for t in range(B):
                pp = rng.permutation(pool_ab); Id_perm[t] = eslope(cliff_cols(X[pp[:len(ia)]], X[pp[len(ia):]]))
            for k in range(10):
                curves.append(dict(source=SRC, condition=COND, region=reg, measure=m, contrast=f"{a}-{b}", bin=k + 1, centre_mm=centres[k],
                                   lower_mm=edges_q[k], upper_mm=edges_q[k + 1], delta=d[k], lo=lo[k], hi=hi[k],
                                   n_first=int(np.isfinite(X[ia, k]).sum()), n_second=int(np.isfinite(X[ib, k]).sum())))
            sa, sb = slope[ia], slope[ib]; sa, sb = sa[np.isfinite(sa)], sb[np.isfinite(sb)]
            I = SIGN[m] * (sb.mean() - sa.mean())
            both = np.concatenate([sa, sb]); na = len(sa)
            perm = np.array([SIGN[m] * (np.mean(p[na:]) - np.mean(p[:na])) for p in (rng.permutation(both) for _ in range(B))])
            boot = np.array([SIGN[m] * (rng.choice(sb, len(sb)).mean() - rng.choice(sa, len(sa)).mean()) for _ in range(B)])
            # reference: linear mixed model on the standardised bin values (random intercept and slope)
            coef = pval = np.nan
            try:
                import statsmodels.formula.api as smf
                sub = np.concatenate([ia, ib]); df = pd.DataFrame({"z": Zs[sub].ravel(), "lx": np.tile(lx, len(sub)),
                                                                   "grp": np.repeat((g[sub] == b).astype(float), 10), "pid": np.repeat(sub, 10)}).dropna()
                fit = smf.mixedlm("z ~ grp * lx", df, groups=df.pid, re_formula="~lx").fit(method="lbfgs", reml=True)
                coef, pval = SIGN[m] * fit.params["grp:lx"], fit.pvalues["grp:lx"]
            except Exception:
                pass
            # effect at the shortest and longest bins, for orientation
            inter.append(dict(source=SRC, condition=COND, region=reg, measure=m, contrast=f"{a}-{b}",
                              I_delta=Id, I_delta_lo=np.nanpercentile(Id_boot, 2.5), I_delta_hi=np.nanpercentile(Id_boot, 97.5),
                              I_delta_p=(np.sum(np.abs(Id_perm[np.isfinite(Id_perm)]) >= abs(Id)) + 1) / (np.isfinite(Id_perm).sum() + 1), I=I,
                              lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
                              p_perm=(np.sum(np.abs(perm) >= abs(I)) + 1) / (B + 1), lmm_coef=coef, lmm_p=pval,
                              delta_short=d[0], delta_long=d[-1], n_first=len(sa), n_second=len(sb)))

# ---------------- C. alternative cut-offs for the published 40-cell statistic
if (P1 / "thresholds.json").is_file():                 # another parcellation / weighting (tools/phase5/p5_features.py)
    th = pd.Series(json.load(open(P1 / "thresholds.json")))
else:
    th = pd.read_csv(Path(os.environ.get("IIT_AROOT", "/home/ec2-user/exp/data/derivatives/qc/analysis_cohort")) / "17_edr_exceptions/edr_exception_thresholds.csv").iloc[0]
pool_c = L[cn_all]; pool_c = pool_c[np.isfinite(pool_c)]
SCHEMES = {"Quartiles (published)": (float(th.short_max_mm), float(th.long_min_mm)),        # the analysis tree's CN Q1 / Q3
           "Terciles": tuple(np.percentile(pool_c, [100 / 3, 200 / 3])), "Quintiles": tuple(np.percentile(pool_c, [20, 80])),
           "Fixed 70 / 140 mm": (70.0, 140.0)}
NN = int(max(ei.max(), ej.max())) + 1                 # 166 AAL3 or 76 DKT regions; network of each region from the edge cache
inc = np.zeros((len(ei), NN)); inc[np.arange(len(ei)), ei] = 1; inc[np.arange(len(ei)), ej] = 1
node_net = dict(zip(ei, ni)) | dict(zip(ej, nj))
netidx = {n: np.array([node_net.get(k) == n for k in range(NN)]) for n in NETS}


def network_values(cls_mask):
    """published recipe: node mean over valid edges, then node-count-weighted network mean = sum S / sum count."""
    out = {}
    for m in MEAS:
        v = VAL[m]; ok = np.isfinite(v) & cls_mask
        S = np.where(ok, v, 0) @ inc; C = ok.astype(float) @ inc
        for n in NETS:
            s_, c_ = S[:, netidx[n]].sum(1), C[:, netidx[n]].sum(1)
            out[(n, m)] = np.where(c_ > 0, s_ / np.maximum(c_, 1), np.nan)
    return out


cut = []
for sch, (smax, lmin) in SCHEMES.items():
    SRv, LRv = network_values(LEN <= smax), network_values(LEN > lmin)
    keys = list(SRv)
    Xs = adjust(np.column_stack([SRv[k] for k in keys]), Z); Xl = adjust(np.column_stack([LRv[k] for k in keys]), Z)
    elig = [c for c in range(len(keys)) if all(np.isfinite(Xs[g == k_, c]).sum() >= 5 and np.isfinite(Xl[g == k_, c]).sum() >= 5 for k_ in ("CN", "MCI", "AD"))]
    Xs, Xl = Xs[:, elig], Xl[:, elig]
    if COND == "P0" and sch.startswith("Quartiles"):      # validation against the analysis tree's TR values
        pubtr = pd.read_csv(P1 / "features_published.csv", index_col=0).reindex(cv.index)
        ks = [keys[c] for c in elig]
        diff = max(np.nanmax(np.abs(Xs[:, q] - pubtr[f"TR|{n}|SR-{m}"].values)) for q, (n, m) in enumerate(ks))
        json.dump({"max_abs_diff_vs_published_TR": float(diff)}, open(R / "validation_P0.json", "w"))
    for a, b in CONTRASTS:
        ia, ib = np.flatnonzero(g == a), np.flatnonzero(g == b)
        ds, dl = np.abs(cliff_cols(Xs[ia], Xs[ib])), np.abs(cliff_cols(Xl[ia], Xl[ib])); okc = np.isfinite(ds) & np.isfinite(dl)
        bs = np.empty(B)
        for t in range(B):
            pa, pb = rng.choice(ia, len(ia)), rng.choice(ib, len(ib))
            s_, l_ = np.abs(cliff_cols(Xs[pa], Xs[pb])), np.abs(cliff_cols(Xl[pa], Xl[pb])); o_ = np.isfinite(s_) & np.isfinite(l_)
            bs[t] = s_[o_].mean() - l_[o_].mean()
        cut.append(dict(source=SRC, condition=COND, scheme=sch, short_max_mm=smax, long_min_mm=lmin, contrast=f"{a}-{b}", n_cells=int(okc.sum()),
                        sr_mean=ds[okc].mean(), lr_mean=dl[okc].mean(), diff=ds[okc].mean() - dl[okc].mean(),
                        lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), sr_gt_lr=int((ds[okc] > dl[okc]).sum())))

pd.DataFrame(curves).to_csv(R / f"curves_{SRC}_{COND}.csv", index=False)
pd.DataFrame(inter).to_csv(R / f"interaction_{SRC}_{COND}.csv", index=False)
pd.DataFrame(cut).to_csv(R / f"cutoffs_{SRC}_{COND}.csv", index=False)
json.dump({"n": {k: int(v) for k, v in pd.Series(g).value_counts().items()}, "bins_mm": edges_q.tolist(), "centres_mm": centres.tolist(),
           "schemes": {k: list(v) for k, v in SCHEMES.items()}}, open(R / f"notes_{SRC}_{COND}.json", "w"), indent=1)
print("done", SRC, COND, dict(pd.Series(g).value_counts()))
