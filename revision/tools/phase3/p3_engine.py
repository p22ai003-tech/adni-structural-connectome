"""Phase 3 — null models for the EDR exceptions and the shared exception core.

The manuscript's core = edges that are exceptions (strength > bin mean + 3 SD within 30 adaptive length bins of the
participant's own connectome) in more than half of ALL participants (51 edges), and 'coverage' = the share of those edges
a participant expresses. Defining the core on everyone and then comparing groups on it is circular, and the core's
stability, threshold dependence and specialness have never been tested. Five analyses:

  A  reproduce the published core and coverage (validation: exceptions recomputed from the matrices);
  B  controls-only core, split-sample: core defined in a random half of CN, coverage compared between the held-out CN
     and MCI / AD (500 splits) — removes the circularity;
  C  split-half stability of the core (500 splits of the whole sample and of CN): Dice overlap of the two half-cores and
     correlation of edge exception frequencies;
  D  threshold sensitivity: exceptions at 2.5, 3.0 and 3.5 SD;
  E  are core edges special? group effects on the strength of core edges against 1,000 sets of non-core edges matched on
     median strength and length (empirical p);
  F  coverage adjusted for total connectivity (edge density and total SIFT2 weight), and for processing route.

Conditions: P0 pooled 530, published data, unadjusted (reference); P2 ADNI-3/4 after Phase 1 quality control, adjusted
for age, sex and education (primary).

Usage: python p3_engine.py <source> [R]          source = published | uniform_v3
"""
import os
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata
import warnings; warnings.filterwarnings("ignore")

SRC = sys.argv[1] if len(sys.argv) > 1 else "published"; R = int(sys.argv[2]) if len(sys.argv) > 2 else 500
D = Path("/home/ec2-user/exp/docs/iit_dipanjan"); O = Path(os.environ.get("IIT_P3", "/home/ec2-user/exp/docs/iit_dipanjan/phase3")); RES = O / "results"; RES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261025)
E = np.load(Path(os.environ.get("IIT_P2", "/home/ec2-user/exp/docs/iit_dipanjan/phase2")) / f"edges_{SRC}.npz", allow_pickle=True)
cov = pd.read_csv(Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1")) / "covariates.csv").set_index("subject_id").loc[E["sid"]]
qc = pd.read_csv(Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1")) / "qc.csv", index_col=0).reindex(cov.index)
WEIGHT = os.environ.get("IIT_P3_WEIGHT", "fd")         # edge weight for the exception rule: fd (SIFT2, published), count or fa
L, W = E["len"].astype(float), E[WEIGHT].astype(float)
NE = L.shape[1]; CT = [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]; COVS = ["age", "female", "education"]


# ---------------------------------------------------------------- exceptions, as published (analysis_edr_exceptions.py)
def exceptions(k_sd, min_bin_edges=120, max_bins=30):
    X = np.zeros(L.shape, bool)
    for p in range(L.shape[0]):
        l, w = L[p], W[p]; v = np.isfinite(l) & (l > 0) & np.isfinite(w) & (w > 0)
        nv = int(v.sum())
        if nv < max(20, min_bin_edges): continue
        nb = int(np.clip(nv // min_bin_edges, 3, max_bins))
        edges = np.unique(np.nanquantile(l[v], np.linspace(0, 1, nb + 1)))
        if len(edges) < 3: continue
        ids = np.full(NE, -1); ids[v] = np.searchsorted(edges[1:-1], l[v], side="right")
        for b in np.unique(ids[v]):
            m = v & (ids == b); vals = w[m]
            if vals.size < 20: continue
            cut = vals.mean() + k_sd * vals.std(ddof=1)
            X[p, m] = w[m] > cut
    return X


t0 = time.time()
EXC = {k: exceptions(k) for k in (2.5, 3.0, 3.5)}
X3 = EXC[3.0]
val = {"exceptions_per_participant_median": float(np.median(X3.sum(1)))}
if SRC == "published":                    # validation against the published edge-level exception table
    pub = pd.read_csv("/home/ec2-user/exp/data/derivatives/qc/analysis_cohort/17_edr_exceptions/edr_exception_edge_level_fd_sum_len_mean.csv",
                      usecols=["subject_id", "i", "j"])
    key = {(int(i), int(j)): e for e, (i, j) in enumerate(zip(E["i"] + 1, E["j"] + 1))}
    P = np.zeros_like(X3); sidx = {s: r for r, s in enumerate(cov.index)}
    for s, i, j in zip(pub.subject_id, pub.i, pub.j):
        if s in sidx: P[sidx[s], key[(min(i, j), max(i, j))]] = True
    val.update(agreement_all_edges=float((P == X3).mean()), published_flags=int(P.sum()), recomputed_flags=int(X3.sum()),
               participants_identical=int((P == X3).all(1).sum()))

# ---------------------------------------------------------------- helpers
def cliff(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3: return np.nan
    r = rankdata(np.concatenate([a, b])); u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return 2 * u / (len(a) * len(b)) - 1


def bm(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return stats.brunnermunzel(a, b).pvalue if min(len(a), len(b)) >= 3 else np.nan


def adjust(y, g, Z):
    """value ~ group + covariates; subtract the covariate part (Phase 1 recipe)."""
    if Z is None: return y
    ok = np.isfinite(y) & np.isfinite(Z).all(1); out = np.full_like(y, np.nan)
    Zc = Z - np.nanmean(Z[ok], 0); G = pd.get_dummies(pd.Series(g), drop_first=True).values.astype(float)
    Dm = np.column_stack([np.ones(ok.sum()), G[ok], Zc[ok]]); beta = np.linalg.lstsq(Dm, y[ok], rcond=None)[0]
    out[ok] = y[ok] - Zc[ok] @ beta[1 + G.shape[1]:]
    return out


def contrasts(y, g):
    return {f"{a}-{b}": (cliff(y[g == a], y[g == b]), bm(y[g == a], y[g == b])) for a, b in CT}


density = (np.isfinite(L) & np.isfinite(W)).mean(1); total_w = np.log(np.nansum(W, 1))
has = np.isfinite(L).any(1) & ((qc.in_cohort == 1).values if "in_cohort" in qc else True)   # in the analysis cohort
cond = {"P0": (has, None), "P2": (((cov.adni34 == 1) & (qc.qc_fail == 0)).values & has, COVS)}
out = {"validation": val}
rows_A, rows_B, rows_C, rows_D, rows_E, rows_F, core_tab = [], [], [], [], [], [], []
DIST = {}                                 # per-split / null values, for the figures

for cname, (keep, covs) in cond.items():
    idx = np.flatnonzero(keep); g = cov.group.values[idx]
    Z = cov.iloc[idx][covs].values.astype(float) if covs else None
    # ---------------- A: published-style core (all participants of the condition, 3 SD, > 50%)
    freq_all = X3[idx].mean(0); core = freq_all > 0.5
    covg = X3[idx][:, core].mean(1); y = adjust(covg, g, Z)
    for k_, (d_, p_) in contrasts(y, g).items():
        rows_A.append(dict(condition=cname, core="all participants", core_size=int(core.sum()), contrast=k_, delta=d_, p=p_))
    if cname == "P0":
        pub_core = core.copy()
        for e in np.flatnonzero(core):
            core_tab.append(dict(i=int(E["i"][e]) + 1, j=int(E["j"][e]) + 1, net_i=str(E["net_i"][e]), net_j=str(E["net_j"][e]),
                                 frequency=float(freq_all[e]), median_length_mm=float(np.nanmedian(L[:, e]))))
    if cname != "P0":                     # the published core (all 530) applied to this sample, for continuity with Phase 1
        y0 = adjust(X3[idx][:, pub_core].mean(1), g, Z)
        for k_, (d_, p_) in contrasts(y0, g).items():
            rows_A.append(dict(condition=cname, core="published (all 530)", core_size=int(pub_core.sum()), contrast=k_, delta=d_, p=p_))
    # ---------------- B: controls-only core, split sample
    cn = np.flatnonzero(g == "CN"); stats_B = {k: [] for k in ("CN-MCI", "CN-AD", "MCI-AD")}; sizes, jac = [], []
    for _ in range(R):
        perm = rng.permutation(cn); disc, held = perm[: len(cn) // 2], perm[len(cn) // 2:]
        c_core = X3[idx[disc]].mean(0) > 0.5; sizes.append(int(c_core.sum()))
        jac.append((c_core & pub_core).sum() / max((c_core | pub_core).sum(), 1))
        test = np.setdiff1d(np.arange(len(idx)), disc)            # held-out CN + all MCI + all AD
        cv_ = X3[idx[test]][:, c_core].mean(1); y_ = adjust(cv_, g[test], Z[test] if Z is not None else None)
        for k_, (d_, p_) in contrasts(y_, g[test]).items(): stats_B[k_].append((d_, p_))
    for k_, v in stats_B.items():
        d_ = np.array([x[0] for x in v]); p_ = np.array([x[1] for x in v]); DIST[f"B|{cname}|{k_}"] = d_
        rows_B.append(dict(condition=cname, contrast=k_, delta_median=np.median(d_), delta_lo=np.percentile(d_, 2.5), delta_hi=np.percentile(d_, 97.5),
                           p_median=np.median(p_), share_p_below_05=float((p_ < 0.05).mean()), core_size_median=float(np.median(sizes)),
                           jaccard_with_published_median=float(np.median(jac)), splits=R))
    # ---------------- C: split-half stability
    for label, pool in [("whole sample", np.arange(len(idx))), ("CN only", cn)]:
        dice, rr = [], []
        for _ in range(R):
            if label == "whole sample":                   # stratified by group
                h1 = np.concatenate([rng.permutation(np.flatnonzero(g == k))[: (g == k).sum() // 2] for k in ("CN", "MCI", "AD")])
            else:
                h1 = rng.permutation(pool)[: len(pool) // 2]
            h2 = np.setdiff1d(pool, h1)
            f1, f2 = X3[idx[h1]].mean(0), X3[idx[h2]].mean(0); c1, c2 = f1 > 0.5, f2 > 0.5
            dice.append(2 * (c1 & c2).sum() / max(c1.sum() + c2.sum(), 1))
            mm = (f1 + f2) > 0; rr.append(np.corrcoef(f1[mm], f2[mm])[0, 1])
        DIST[f"C|{cname}|{label}"] = np.array(dice)
        rows_C.append(dict(condition=cname, sample=label, dice_median=np.median(dice), dice_lo=np.percentile(dice, 2.5), dice_hi=np.percentile(dice, 97.5),
                           freq_corr_median=np.median(rr), splits=R))
    # ---------------- D: threshold sensitivity
    for k_sd, Xk in EXC.items():
        for label, defset in [("all participants", np.arange(len(idx))), ("CN only", cn)]:
            ck = Xk[idx[defset]].mean(0) > 0.5
            ref = X3[idx[defset]].mean(0) > 0.5
            yk = adjust(Xk[idx][:, ck].mean(1) if ck.any() else np.full(len(idx), np.nan), g, Z)
            for c_, (d_, p_) in contrasts(yk, g).items():
                rows_D.append(dict(condition=cname, sd=k_sd, core_defined_in=label, core_size=int(ck.sum()),
                                   jaccard_with_3sd=float((ck & ref).sum() / max((ck | ref).sum(), 1)),
                                   exceptions_per_participant=float(np.median(Xk[idx].sum(1))), contrast=c_, delta=d_, p=p_))
    # ---------------- E: core edges versus strength- and length-matched ordinary edges
    Wz = np.nan_to_num(W[idx], nan=0.0)                          # absent edge = no streamlines = zero strength
    present = np.isfinite(W[idx]).mean(0) >= 0.5
    med_logw = np.log(np.where(np.isfinite(W[idx]), W[idx], np.nan)); med_logw = np.nanmedian(med_logw, 0)
    med_len = np.nanmedian(L[idx], 0)
    cand = np.flatnonzero(present & ~core)
    feat = np.column_stack([med_logw, med_len]); mu, sd = np.nanmean(feat[cand], 0), np.nanstd(feat[cand], 0)
    fz = (feat - mu) / sd
    Wadj = np.column_stack([adjust(Wz[:, e], g, Z) for e in np.flatnonzero(core | present)]) if Z is not None else Wz[:, core | present]
    cols = np.flatnonzero(core | present); colpos = {e: c for c, e in enumerate(cols)}
    deltas = {f"{a}-{b}": np.array([cliff(Wadj[g == a, c], Wadj[g == b, c]) for c in range(len(cols))]) for a, b in CT}
    core_e = np.flatnonzero(core)
    nn = {e: cand[np.argsort(((fz[cand] - fz[e]) ** 2).sum(1))[:20]] for e in core_e}     # 20 nearest ordinary edges
    for k_, dv in deltas.items():
        obs = np.mean([dv[colpos[e]] for e in core_e]); obs_abs = np.mean([abs(dv[colpos[e]]) for e in core_e])
        null = np.empty(1000); null_abs = np.empty(1000)
        for t in range(1000):
            pick = [rng.choice(nn[e]) for e in core_e]
            null[t] = np.mean([dv[colpos[e]] for e in pick]); null_abs[t] = np.mean([abs(dv[colpos[e]]) for e in pick])
        DIST[f"E|{cname}|{k_}"] = null_abs; DIST[f"Eobs|{cname}|{k_}"] = np.array([obs_abs])
        rows_E.append(dict(condition=cname, contrast=k_, core_mean_delta=obs, null_mean=null.mean(), null_lo=np.percentile(null, 2.5),
                           null_hi=np.percentile(null, 97.5), p_two_sided=float((np.abs(null - null.mean()) >= abs(obs - null.mean())).mean()),
                           core_mean_abs_delta=obs_abs, null_abs_mean=null_abs.mean(), p_abs_greater=float((null_abs >= obs_abs).mean()),
                           matched_median_logw_diff=float(np.mean([med_logw[e] for e in core_e]) - np.mean([med_logw[nn[e]].mean() for e in core_e])),
                           matched_median_len_diff=float(np.mean([med_len[e] for e in core_e]) - np.mean([med_len[nn[e]].mean() for e in core_e]))))
    # ---------------- F: coverage adjusted for total connectivity (and route)
    extra = {"+ edge density + total weight": np.column_stack([density[idx], total_w[idx]]),
             "+ processing route": pd.get_dummies(qc.route.values[idx], drop_first=True).values.astype(float)}
    for lab, M in extra.items():
        Zf = np.column_stack([Z, M]) if Z is not None else M
        yf = adjust(covg, g, Zf)
        for k_, (d_, p_) in contrasts(yf, g).items():
            rows_F.append(dict(condition=cname, adjustment=lab, contrast=k_, delta=d_, p=p_))
    out[f"n_{cname}"] = {k: int(v) for k, v in pd.Series(g).value_counts().items()}

for name, rows in [("A_core_coverage", rows_A), ("B_controls_only_split", rows_B), ("C_split_half", rows_C), ("D_thresholds", rows_D),
                   ("E_matched_null", rows_E), ("F_connectivity_adjusted", rows_F), ("core_edges", core_tab)]:
    pd.DataFrame(rows).to_csv(RES / f"{name}_{SRC}.csv", index=False)
# per-edge exception frequency, for the figures
pd.DataFrame({"i": E["i"] + 1, "j": E["j"] + 1, "freq_all": X3.mean(0), "freq_cn": X3[cov.group.values == "CN"].mean(0),
              "freq_ad": X3[cov.group.values == "AD"].mean(0), "freq_mci": X3[cov.group.values == "MCI"].mean(0)}).to_csv(RES / f"edge_frequency_{SRC}.csv", index=False)
np.savez_compressed(RES / f"distributions_{SRC}.npz", **{k.replace("|", "__").replace(" ", "_"): v for k, v in DIST.items()})
out["seconds"] = round(time.time() - t0)
json.dump(out, open(RES / f"summary_{SRC}.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
