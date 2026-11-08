"""Phase 5 step 2 — the Phase 1 feature files for a second parcellation or another edge weighting, so that the Phase 1
engine (p1_engine.py) runs on them unchanged.

Usage: python p5_features.py <edges npz> <node-network csv> <out dir> [weight]
  edges npz         p2_edges.py cache (len, fd, count, fa, md, rd, ad per participant x edge)
  node-network csv  node index (1-based) -> network
  weight            edge weight for the exception rule: fd (SIFT2, published), count (streamline count) or fa

Writes into <out dir>: covariates.csv and qc.csv (from IIT_P1; participants without matrices leave the cohort),
features_published.csv, features_qc.csv and features_consistent_edges_qc.csv with the Phase 1 column names
(WB|net|M, TR|net|SR-M / LR-M, EX|net|SRexc-M / LRexc-M, COV|core|coverage) and thresholds.json.
Recipes as in the analysis pipeline: network value = edge-count-weighted mean of node means over valid edges; SR/LR =
edges at or below the CN first quartile / above the third quartile of edge length; exceptions = weight above the mean +
3 SD of its adaptive equal-count length bin (>= 120 edges, <= 30 bins) in the participant's own connectome; core = edges
that are exceptions in more than half of the cohort; coverage = share of core edges a participant expresses.
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import iit_chain as chain; chain.boot()      # IIT_ROOT and the chain settings; no effect when IIT_ROOT is unset
import os
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

E = dict(np.load(sys.argv[1], allow_pickle=True))            # load every array once (an NpzFile re-reads on each access)
NETMAP = pd.read_csv(sys.argv[2]); O = Path(sys.argv[3]); O.mkdir(parents=True, exist_ok=True)
WEIGHT = sys.argv[4] if len(sys.argv) > 4 else "fd"
P1 = Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1_uniform"))
FREE_WATER = 3.0e-3; M = {"fa": "FA", "md": "MD", "rd": "RD", "ad": "AxD"}
sid = E["sid"]; I, J = E["i"].astype(int), E["j"].astype(int)
L = E["len"].astype(float); Wt = E[WEIGHT].astype(float)
n_nodes = int(max(I.max(), J.max())) + 1
net_of = np.array([None] * n_nodes, dtype=object)
for nd, net in zip(NETMAP.node.astype(int), NETMAP.network): net_of[nd - 1] = net
NETS = [n for n in ["Visual", "Somatomotor", "DorsalAttention", "Salience_VAN", "Limbic", "Frontoparietal", "DMN", "Subcortical",
                    "Cerebellar", "Brainstem"] if n in set(NETMAP.network)]

cov = pd.read_csv(P1 / "covariates.csv").set_index("subject_id")
qc = pd.read_csv(P1 / "qc.csv", index_col=0)
has = np.isfinite(L).any(1)
qc = qc.reindex(cov.index); qc.loc[~qc.index.isin(sid[has]), ["in_cohort", "qc_fail"]] = [0, 1]
qc.loc[~qc.index.isin(sid[has]), "qc_reason"] = "no matrices for this parcellation"
cov.reset_index().to_csv(O / "covariates.csv", index=False); qc.to_csv(O / "qc.csv")
cohort = np.isin(sid, qc.index[qc.in_cohort == 1]) & has

# ---------------- short / long range: CN quartiles of pooled edge length
cn = np.isin(sid, cov.index[cov.group == "CN"]) & cohort
SRMAX, LRMIN = np.nanpercentile(L[cn][np.isfinite(L[cn])], [25, 75])


def exceptions(k_sd=3.0, min_bin_edges=120, max_bins=30):
    X = np.zeros(L.shape, bool)
    for p in range(L.shape[0]):
        l, w = L[p], Wt[p]; v = np.isfinite(l) & (l > 0) & np.isfinite(w) & (w > 0)
        nv = int(v.sum())
        if nv < max(20, min_bin_edges): continue
        nb = int(np.clip(nv // min_bin_edges, 3, max_bins))
        edges = np.unique(np.nanquantile(l[v], np.linspace(0, 1, nb + 1)))
        if len(edges) < 3: continue
        ids = np.full(L.shape[1], -1); ids[v] = np.searchsorted(edges[1:-1], l[v], side="right")
        for b in np.unique(ids[v]):
            m = v & (ids == b); vals = w[m]
            if vals.size < 20: continue
            X[p, m] = w[m] > vals.mean() + k_sd * vals.std(ddof=1)
    return X


X = exceptions()
freq = X[cohort].mean(0); core = freq > 0.5
coverage = np.where(has, X[:, core].sum(1) / max(core.sum(), 1), np.nan); coverage[~cohort] = np.nan


def networks(p, mask):
    """node mean over the masked valid edges -> edge-count-weighted network mean (the pipeline's recipe)."""
    out = {}
    for key, lbl in M.items():
        v = E[key][p].astype(float); ok = mask & np.isfinite(v) & (v > 0)
        s = np.bincount(I[ok], v[ok], n_nodes) + np.bincount(J[ok], v[ok], n_nodes)
        c = np.bincount(I[ok], minlength=n_nodes) + np.bincount(J[ok], minlength=n_nodes)
        node = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        for net in NETS:
            sel = (net_of == net) & np.isfinite(node); w = c[sel]
            out[net, lbl] = (node[sel] * w).sum() / w.sum() if w.sum() else np.nan
    return out


pres_all = np.isfinite(L) & (L > 0) & np.isfinite(E["fd"]) & (E["fd"] > 0)
a34 = np.isin(sid, cov.index[chain.primary(cov) == 1]) & cohort
keep = pres_all[a34].mean(0) >= 0.90
rows, rows_c = {}, {}
for p in np.flatnonzero(has):
    pr = pres_all[p]; sr, lr = pr & (L[p] <= SRMAX), pr & (L[p] > LRMIN)
    r = {}
    for tag, mask in [("WB", pr), ("SR", sr), ("LR", lr), ("SRexc", sr & X[p]), ("LRexc", lr & X[p])]:
        for (net, lbl), val in networks(p, mask).items():
            r[f"WB|{net}|{lbl}" if tag == "WB" else (f"TR|{net}|{tag}-{lbl}" if tag in ("SR", "LR") else f"EX|{net}|{tag}-{lbl}")] = val
    r["COV|core|coverage"] = coverage[p]; rows[sid[p]] = r
    rc = {}
    for tag, mask in [("WB", pr & keep), ("SR", sr & keep), ("LR", lr & keep)]:
        for (net, lbl), val in networks(p, mask).items():
            rc[f"WB|{net}|{lbl}" if tag == "WB" else f"TR|{net}|{tag}-{lbl}"] = val
    rows_c[sid[p]] = rc
F = pd.DataFrame(rows).T.reindex(cov.index); Fc = pd.DataFrame(rows_c).T.reindex(cov.index)
F.to_csv(O / "features_published.csv")


def mask_values(Xf):                                              # the Phase 1 value rule (p1_audit.py)
    Xf = Xf.copy()
    for st in sorted({c[:-2] for c in Xf.columns if c.endswith("FA")}):
        quart = [st + k for k in ("FA", "MD", "AxD", "RD")]
        if not all(q in Xf for q in quart): continue
        bad = (Xf[st + "FA"] > 1) | (Xf[st + "FA"] < 0.05) | (Xf[st + "AxD"] < Xf[st + "RD"]) | \
              (Xf[st + "MD"] > FREE_WATER) | (Xf[st + "AxD"] > FREE_WATER) | (Xf[st + "RD"] > FREE_WATER)
        Xf.loc[bad.fillna(False), quart] = np.nan
    return Xf


mask_values(F).to_csv(O / "features_qc.csv"); mask_values(Fc).to_csv(O / "features_consistent_edges_qc.csv")
pd.DataFrame({"i": I[core] + 1, "j": J[core] + 1, "frequency": freq[core]}).to_csv(O / "core_edges.csv", index=False)
json.dump(dict(weight=WEIGHT, short_max_mm=float(SRMAX), long_min_mm=float(LRMIN), core_edges=int(core.sum()), n_cohort=int(cohort.sum()),
               consistent_edges=int(keep.sum()), edges=int(L.shape[1]), exceptions_median=float(np.median(X[cohort].sum(1))), networks=NETS),
          open(O / "thresholds.json", "w"), indent=1)
print(f"{O.name}: weight {WEIGHT}; {int(has.sum())} participants with matrices, cohort {int(cohort.sum())}; SR <= {SRMAX:.1f} mm, LR > {LRMIN:.1f} mm; "
      f"core {int(core.sum())} edges; consistent edges {int(keep.sum())} of {L.shape[1]}; features {F.shape[1]}")
