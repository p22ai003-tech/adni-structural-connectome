"""Phase 7 step 1 — every number the revised manuscript quotes, from the rebuilt-connectome results of Phases 1-5,
written to phase7_uniform/numbers.json (so the text step never recomputes anything)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

U = Path("/home/ec2-user/exp/docs/iit_dipanjan"); O = U / "phase7_uniform"; O.mkdir(exist_ok=True)
P1, P2, P3, P4, P5 = (U / f"phase{k}_uniform" for k in range(1, 6))
A = Path("/data/derivatives/qc/analysis_cohort_uniform")
N = {}
cov = pd.read_csv(P1 / "covariates.csv").set_index("subject_id"); qc = pd.read_csv(P1 / "qc.csv", index_col=0).reindex(cov.index)
keep = cov[qc.qc_fail == 0]; prim = keep[keep.adni34 == 1]
G = ["CN", "MCI", "AD"]
N["n_sessions"] = len(cov); N["n_excl"] = int((qc.qc_fail == 1).sum())
N["pooled"] = {g: int((keep.group == g).sum()) for g in G}; N["pooled_n"] = len(keep)
N["primary"] = {g: int((prim.group == g).sum()) for g in G}; N["primary_n"] = len(prim)
N["adni2"] = {g: int(((keep.group == g) & (keep.phase == "ADNI-2")).sum()) for g in G}
N["age"] = {g: [round(keep[keep.group == g].age.mean(), 1), round(keep[keep.group == g].age.std(), 1)] for g in G}
N["age_all"] = [round(keep.age.mean(), 1), round(keep.age.std(), 1)]
N["female"] = {g: [int(keep[keep.group == g].female.sum()), round(100 * keep[keep.group == g].female.mean(), 1)] for g in G}
N["female_all"] = [int(keep.female.sum()), round(100 * keep.female.mean(), 1)]
N["age_kw_p"] = float(stats.kruskal(*[keep[keep.group == g].age for g in G]).pvalue)
N["female_chi2_p"] = float(stats.chi2_contingency(pd.crosstab(keep.group, keep.female))[1])
p4 = pd.read_csv(P4 / "p4_data.csv", index_col=0); pp = p4[p4.adni34 == 1]; pk = p4
N["cdrsb_n_primary"] = {g: int(pp[pp.cdrsb.notna() & (pp.group == g)].shape[0]) for g in G}
N["cdrsb_n_pooled"] = {g: int(pk[pk.cdrsb.notna() & (pk.group == g)].shape[0]) for g in G}
N["score_lag_median"] = float(pp.cdrsb_lag.median())
N["score_n_range"] = [int(pp[t].notna().sum()) for t in ("cdrsb", "adas13", "phc_mem", "phc_exf", "mmse", "faq")]

# ---------------- lengths, exceptions, core
E = dict(np.load(P2 / "edges_uniform_v3.npz", allow_pickle=True)); L, W = E["len"].astype(float), E["fd"].astype(float); sid = E["sid"]
cv = cov.loc[sid]; qv = qc.loc[sid]; kp = (qv.qc_fail == 0).values; pr = kp & (cv.adni34 == 1).values; coh = (qv.in_cohort == 1).values
th = pd.read_csv(A / "17_edr_exceptions/edr_exception_thresholds.csv").iloc[0]; SR, LR = float(th.short_max_mm), float(th.long_min_mm)
cnm = pr & (cv.group == "CN").values
vals = L[cnm][np.isfinite(L[cnm])]; N["len_q"] = [round(float(x), 1) for x in np.percentile(vals, [25, 50, 75])]
med = np.nanmedian(L, 1)
N["len_group_median"] = {g: round(float(np.median(med[pr & (cv.group == g).values])), 1) for g in G}


def exceptions(k_sd=3.0):
    X = np.zeros(L.shape, bool)
    for p in range(L.shape[0]):
        l, w = L[p], W[p]; v = np.isfinite(l) & (l > 0) & np.isfinite(w) & (w > 0); nv = int(v.sum())
        if nv < 120: continue
        nb = int(np.clip(nv // 120, 3, 30)); ed = np.unique(np.nanquantile(l[v], np.linspace(0, 1, nb + 1)))
        ids = np.full(L.shape[1], -1); ids[v] = np.searchsorted(ed[1:-1], l[v], side="right")
        for b in np.unique(ids[v]):
            m = v & (ids == b); vv = w[m]
            if vv.size >= 20: X[p, m] = w[m] > vv.mean() + k_sd * vv.std(ddof=1)
    return X


X = exceptions(); freq = X[coh].mean(0); core = freq > 0.5
anyx = X[kp].any(0); N["exc_distinct"] = int(anyx.sum()); N["exc_rare_share"] = round(float((X[kp].mean(0)[anyx] < 0.05).mean()), 3)
N["exc_per_participant_median"] = float(np.median(X[kp].sum(1)))
mp = pd.read_csv(A / "19_network_analysis/network_mapping_used.csv").set_index("matrix_idx")
I, J = E["i"], E["j"]; lab = lambda k: mp.loc[k + 1, "atlas_label"]; net = lambda k: mp.loc[k + 1, "functional_network"]
clen = np.nanmedian(L[kp][:, core], 0); cls = np.where(clen <= SR, "SR", np.where(clen > LR, "LR", "MR"))
ci, cj = I[core], J[core]
N["core_n"] = int(core.sum()); N["core_same_net"] = int(sum(net(i) == net(j) for i, j in zip(ci, cj)))
N["core_homologues"] = int(sum(lab(i).rsplit("_", 1)[0] == lab(j).rsplit("_", 1)[0] and lab(i)[-1] != lab(j)[-1] for i, j in zip(ci, cj)))
N["core_classes"] = {k: int((cls == k).sum()) for k in ("SR", "MR", "LR")}
fr_k = X[kp][:, core].mean(0); order = np.argsort(-fr_k)[:10]
NICE = {"Salience_VAN": "Salience/VAN", "DorsalAttention": "DorsalAttention"}
N["core_top10"] = [[f"{lab(ci[t])} – {lab(cj[t])}", f"{NICE.get(net(ci[t]), net(ci[t]))}–{NICE.get(net(cj[t]), net(cj[t]))}", str(cls[t]),
                    f"{clen[t]:.1f}", f"{100 * fr_k[t]:.1f}"] for t in order]
lrw = {}
for g in G:
    rows = pr & (cv.group == g).values; ex, od = [], []
    for p in np.flatnonzero(rows):
        lr = np.isfinite(L[p]) & (L[p] > LR) & np.isfinite(W[p])
        if (lr & X[p]).sum(): ex.append(np.median(W[p][lr & X[p]]))
        od.append(np.median(W[p][lr & ~X[p]]))
    lrw[g] = [round(float(np.median(ex)), 1), round(float(np.median(od)), 2), round(float(np.median(ex) / np.median(od)), 1)]
N["lr_weights"] = lrw
covg = np.where(kp, X[:, core].sum(1) / core.sum(), np.nan)
N["coverage_mean_primary"] = {g: round(float(np.nanmean(covg[pr & (cv.group == g).values])), 3) for g in G}
st = pd.read_csv(A / "19_network_analysis/functional/network_exception_range_stats.csv")
qs = [c for c in st.columns if c.endswith("_bm_q")]
N["burden_cells"] = [int((st[qs] < 0.05).sum().sum()), int(st[qs].notna().sum().sum()), int((st["cn_mci_bm_q"] < 0.05).sum())]
N["SR"], N["LR"] = round(SR, 1), round(LR, 1)

# ---------------- Phase 1 effects
e = pd.read_csv(P1 / "effects_long.csv"); e["c"] = e.condition.str.split().str[0]
s = pd.read_csv(P1 / "sr_lr_summary.csv"); s["c"] = s.condition.str.split().str[0]
ep = pd.read_csv(U / "phase1/effects_long.csv"); ep = ep[ep.condition.str.startswith("C0 ")]


def eff(c, f, ct):
    r = e[(e.c == c) & (e.feature == f) & (e.contrast == ct)].iloc[0]; return [round(float(r.delta), 3), float(r.q)]


KEYF = {"limbic_axd": "WB|Limbic|AxD", "subc_axd": "WB|Subcortical|AxD", "sal_axd": "WB|Salience_VAN|AxD", "dmn_axd": "WB|DMN|AxD",
        "limbic_md": "WB|Limbic|MD", "limbic_rd": "WB|Limbic|RD", "dmn_fa": "WB|DMN|FA", "limbic_lrmd": "TR|Limbic|LR-MD",
        "limbic_lraxd": "TR|Limbic|LR-AxD", "limbic_sraxd": "TR|Limbic|SR-AxD", "limbic_lrexc_md": "EX|Limbic|LRexc-MD",
        "limbic_lrexc_axd": "EX|Limbic|LRexc-AxD", "dmn_srexc_rd": "EX|DMN|SRexc-RD", "fp_srexc_axd": "EX|Frontoparietal|SRexc-AxD",
        "coverage": "COV|core|coverage"}
N["eff"] = {c: {k: {ct: eff(c, f, ct) for ct in ("CN-MCI", "CN-AD", "MCI-AD")} for k, f in KEYF.items()}
            for c in ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9")}
N["counts"] = {c: {m: [int(((e.c == c) & (e.kind == "WB") & (e.measure == m) & (e.contrast == ct) & (e.q < 0.05)).sum())
                       for ct in ("CN-MCI", "CN-AD", "MCI-AD")] for m in ("AxD", "MD", "RD", "FA")} for c in ("C0", "C2", "C5", "C8")}
N["srlr"] = {c: {kind: {ct: [round(float(x), 3) for x in s[(s.c == c) & (s.kind == kind) & (s.contrast == ct)][["sr_mean", "lr_mean", "diff", "ci_lo", "ci_hi"]].iloc[0]]
                        + [int(s[(s.c == c) & (s.kind == kind) & (s.contrast == ct)].sr_gt_lr.iloc[0]), int(s[(s.c == c) & (s.kind == kind) & (s.contrast == ct)].n_cells.iloc[0])]
                        for ct in ("CN-MCI", "CN-AD", "MCI-AD")} for kind in ("TR", "EX")} for c in ("C0", "C1", "C2", "C3", "C5", "C8")}
leaders = {}
for kind in ("WB", "TR", "EX"):
    for ct in ("CN-MCI", "CN-AD", "MCI-AD"):
        x = e[(e.c == "C2") & (e.kind == kind) & (e.contrast == ct) & (e.q < 0.05)]
        if kind == "TR": x = x[x.feature.str.contains("-")]
        x = x.reindex(x.delta.abs().sort_values(ascending=False).index)
        top = x.iloc[0]; ties = x[(x.delta.abs() >= abs(top.delta) - 0.005)]
        leaders[f"{kind}|{ct}"] = [[f.split("|")[1], f.split("|")[2], round(float(d), 3), float(q)] for f, d, q in zip(ties.feature, ties.delta, ties.q)][:2]
N["leaders"] = leaders
N["verdict"] = pd.read_csv(P1 / "tables/verdict_matrix.csv", index_col=0).to_dict()

# ---------------- Phase 2
it = pd.read_csv(P2 / "tables/interaction_uniform_v3.csv"); it = it[it.condition == "P2"]
N["slopes"] = {f"{r.region}|{r.contrast}|{r.measure}": [round(float(r.I_delta), 3), round(float(r.I_delta_lo), 3), round(float(r.I_delta_hi), 3),
                                                        float(r.I_delta_p), float(r.lmm_p), round(float(r.delta_short), 3), round(float(r.delta_long), 3)]
               for r in it.itertuples()}
# ---------------- Phase 3
R3 = P3 / "results"
B = pd.read_csv(R3 / "B_controls_only_split_uniform_v3.csv"); C = pd.read_csv(R3 / "C_split_half_uniform_v3.csv")
Em = pd.read_csv(R3 / "E_matched_null_uniform_v3.csv"); F = pd.read_csv(R3 / "F_connectivity_adjusted_uniform_v3.csv"); D = pd.read_csv(R3 / "D_thresholds_uniform_v3.csv")
N["ctrl_only"] = {r.contrast: [round(float(r.delta_median), 3), round(float(r.share_p_below_05), 2)] for r in B[B.condition == "P2"].itertuples()}
N["dice"] = {r.sample: round(float(r.dice_median), 2) for r in C[C.condition == "P2"].itertuples()}
N["matched"] = {r.contrast: [round(float(r.core_mean_abs_delta), 3), round(float(r.null_abs_mean), 3), float(r.p_abs_greater)] for r in Em[Em.condition == "P2"].itertuples()}
N["cov_adj_density"] = {r.contrast: [round(float(r.delta), 3), float(r.p)] for r in F[(F.condition == "P2") & F.adjustment.str.startswith("+ edge")].itertuples()}
N["core_sizes_sd"] = {str(r.sd): int(r.core_size) for r in D[(D.condition == "P2") & (D.core_defined_in == "all participants") & (D.contrast == "CN-MCI")].itertuples()}
# ---------------- Phase 4
M4 = pd.read_csv(P4 / "results/models.csv"); C4 = pd.read_csv(P4 / "results/comparisons.csv")
m4 = lambda t, mo, g="all": M4[(M4["sample"] == "primary") & (M4.target == t) & (M4.model == mo) & (M4.group == g)].iloc[0]
c4 = lambda t, a, b, g="all": C4[(C4["sample"] == "primary") & (C4.target == t) & (C4.model == a) & (C4.versus == b) & (C4.group == g)].iloc[0]
TG = ["cdrsb", "adas13", "phc_mem", "phc_exf", "mmse", "faq"]
N["p4"] = {t: {"n": int(m4(t, "C").n), "C": round(float(m4(t, "C")["mean"]), 3), "Dx": round(float(m4(t, "C+Dx")["mean"]), 3),
               "Conn": round(float(m4(t, "C+F1-4")["mean"]), 3), "ConnBest": round(float(max(m4(t, r)["mean"] for r in ["C+F1", "C+F1-2", "C+F1-3", "C+F1-4"])), 3),
               "DxConn": round(float(m4(t, "C+Dx+F1-4")["mean"]), 3),
               "gain": [round(float(c4(t, "C+Dx+F1-4", "C+Dx")["diff"]), 3), round(float(c4(t, "C+Dx+F1-4", "C+Dx").diff_lo), 3), round(float(c4(t, "C+Dx+F1-4", "C+Dx").diff_hi), 3)],
               "mci": [round(float(c4(t, "C+F1-4", "C", "MCI")["diff"]), 3), round(float(c4(t, "C+F1-4", "C", "MCI").diff_lo), 3), round(float(c4(t, "C+F1-4", "C", "MCI").diff_hi), 3)]}
           for t in TG}
A4 = pd.read_csv(P4 / "results/association.csv"); a4 = A4[(A4["sample"] == "primary") & (A4.q < 0.05)]
N["p4_assoc"] = [int(len(a4)), int((A4["sample"] == "primary").sum()), sorted(set(a4.group))]
# ---------------- Phase 5
K5 = pd.read_csv(P5 / "tables/key_effects_variants.csv").set_index("Effect"); N5 = pd.read_csv(P5 / "tables/nulls_variants.csv").set_index("Variant")
N["dkt"] = {"limbic_axd_cnmci": K5.loc["Limbic AxD, CN–MCI", "DKT parcellation"].split(" ")[0], "limbic_axd_cnad": K5.loc["Limbic AxD, CN–AD", "DKT parcellation"].split(" ")[0],
            "core": int(N5.loc["DKT parcellation", "Core edges (pooled)"]), "ctrl_cnad": N5.loc["DKT parcellation", "Controls-only CN–AD: δ (% splits p<0.05)"],
            "ctrl_mciad": N5.loc["DKT parcellation", "Controls-only MCI–AD: δ (% splits p<0.05)"], "ctrl_cnmci": N5.loc["DKT parcellation", "Controls-only CN–MCI: δ (% splits p<0.05)"]}
N["count_core"] = int(N5.loc["Streamline-count weights", "Core edges (pooled)"])
json.dump(N, open(O / "numbers.json", "w"), indent=1, ensure_ascii=False, default=float)
print("numbers.json written:", len(N), "entries")
