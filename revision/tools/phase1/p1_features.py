"""Phase 1 step 2 — per-subject network features.

(a) published network values -> wide table (whole-brain WB, SR/LR tract, SR/LR exception edges, core coverage)
(b) re-derive WB and SR/LR diffusion values from the edge matrices with the published recipe, and check they
    reproduce (a);
(c) the same recipe restricted to 'consistent' edges: present (SIFT2 weight > 0) in >= 90% of ADNI-3/4
    participants (sir's item 1: remove effects of which edges happen to be reconstructed).
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import iit_chain as chain; chain.boot()      # IIT_ROOT and the chain settings; no effect when IIT_ROOT is unset
import os
import glob, re
from pathlib import Path
import numpy as np, pandas as pd

EXP = Path("/home/ec2-user/exp"); A = Path(os.environ.get("IIT_AROOT", "/home/ec2-user/exp/data/derivatives/qc/analysis_cohort"))
OUT = Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1")); CONN = Path(os.environ.get("IIT_CONN", "/data/derivatives/connectomes"))
cov = pd.read_csv(OUT / "covariates.csv")
M = {"fa_mean": "FA", "md_mean": "MD", "rd_mean": "RD", "ad_mean": "AxD"}

# ---------------- (a) published values
wb = pd.read_csv(A / "19_network_analysis/functional/network_microstructure_subject.csv")
wb["feat"] = "WB|" + wb.network + "|" + wb.metric.map(M)
rr = pd.read_csv(A / "19_network_analysis/functional/network_range_restricted_subject.csv")
rr = rr[rr.measure.str.contains("FA|MD|RD|AxD")]; rr["feat"] = "TR|" + rr.network + "|" + rr.measure
ex = pd.read_csv(A / "19_network_analysis/functional/network_exception_measures_subject.csv")
ex = ex[ex.measure.str.contains("FA|MD|RD|AxD")]; ex["feat"] = "EX|" + ex.network + "|" + ex.measure
pub = pd.concat([d[["subject_id", "feat", "value"]] for d in (wb, rr, ex)]).pivot_table(index="subject_id", columns="feat", values="value")
e = pd.read_csv(A / "17_edr_exceptions/edr_exception_edge_level_fd_sum_len_mean.csv", usecols=["subject_id", "i", "j", "exception_flag"])
e = e[e.exception_flag == 1]
cohort = set(pd.read_csv(A / "00_master/master_cohort.csv", usecols=["subject_id"]).subject_id)   # the analysis tree's cohort
n_sub = len(cohort)                                                                                  # 530 for the published data
freq = e.groupby(["i", "j"]).subject_id.nunique() / n_sub
core = set(freq[freq > 0.5].index)
covg = e[e.set_index(["i", "j"]).index.isin(core)].groupby("subject_id").size() / max(len(core), 1)
pub = pub.reindex(cov.subject_id)          # align to all 530 first: coverage is structural (n = 530),
pub["COV|core|coverage"] = pub.index.map(covg).fillna(0.0)   # incl. the structural-only participant
pub.loc[~pub.index.isin(cohort), "COV|core|coverage"] = np.nan   # outside the analysis cohort: no value
pub.to_csv(OUT / "features_published.csv")
print("published features:", pub.shape)

# ---------------- (b)/(c) recompute from matrices
mp = pd.read_csv(A / "19_network_analysis/network_mapping_used.csv")
idx2net = dict(zip(mp.matrix_idx.astype(int), mp.functional_network))
NETS = ["Visual", "Somatomotor", "DorsalAttention", "Salience_VAN", "Limbic", "Frontoparietal", "DMN", "Subcortical", "Cerebellar", "Brainstem"]
th = pd.read_csv(A / "17_edr_exceptions/edr_exception_thresholds.csv").iloc[0]
SR, LR = float(th.short_max_mm), float(th.long_min_mm)
master = pd.read_csv(A / "00_master/master_cohort.csv", low_memory=False)
fsid = {}
for sid, iid in zip(master.subject_id, master["Image ID"].astype(str).str.replace(r"\.0$", "", regex=True)):
    if (CONN / f"SC_AAL166_{sid}_I{iid}_len_mean.csv").is_file():
        fsid[sid] = f"{sid}_I{iid}"            # published choice: master image id when it exists
    else:
        alts = sorted(CONN.glob(f"SC_AAL166_{sid}_I*_len_mean.csv"))
        if alts: fsid[sid] = alts[0].name[len("SC_AAL166_"):-len("_len_mean.csv")]
net_of = np.array([idx2net.get(k + 1, None) for k in range(166)], dtype=object)

def load(sid):
    f = fsid[sid]; rd = lambda k: np.loadtxt(CONN / f"SC_AAL166_{f}_{k}.csv", delimiter=",")
    if not (CONN / f"SC_AAL166_{f}_fa_mean.csv").is_file():
        return rd("len_mean"), rd("fd_sum"), None          # structural-only participant (n = 529 for DTI)
    return rd("len_mean"), rd("fd_sum"), {k: rd(k) for k in M}

def networks(L, W, micro, keep=None):
    """node mean over valid edges -> edge-count-weighted network mean (published recipe)."""
    out = {}
    base_tr = (L > 0) & (W > 0); np.fill_diagonal(base_tr, False)
    sets = {"WB": None, "SR": base_tr & (L <= SR), "LR": base_tr & (L > LR)}
    for tag, rmask in sets.items():
        for key, lbl in M.items():
            arr = micro[key]; v = np.isfinite(arr) & (arr > 0)
            if tag != "WB": v &= rmask
            else: np.fill_diagonal(v, False)
            if keep is not None: v &= keep
            cnt = v.sum(1).astype(float); s = np.where(v, arr, 0.0).sum(1)
            node = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
            for net in NETS:
                sel = (net_of == net) & np.isfinite(node)
                w = cnt[sel]
                val = (node[sel] * w).sum() / w.sum() if w.sum() > 0 else np.nan
                out[(f"WB|{net}|{lbl}" if tag == "WB" else f"TR|{net}|{tag}-{lbl}")] = val
    return out

subs = [s for s in cov.subject_id if s in fsid]
a34 = set(cov[chain.primary(cov) == 1].subject_id)     # the primary sample (IIT_PRIMARY_SAMPLE; default ADNI-3/4)
pres = np.zeros((166, 166)); n34 = 0; cache = {}
for s in subs:
    L, W, mic = load(s); cache[s] = (L, W, mic)
    if s in a34: pres += ((L > 0) & (W > 0)); n34 += 1
keep = (pres / n34) >= 0.90; np.fill_diagonal(keep, False)
print(f"consistent edges (>=90% of {n34} ADNI-3/4): {int(keep.sum()//2)} of {int(((pres>0).sum())//2)} ever-present")
rep, con = {}, {}
for s in subs:
    L, W, mic = cache[s]
    if mic is None: continue
    rep[s] = networks(L, W, mic); con[s] = networks(L, W, mic, keep)
rep = pd.DataFrame(rep).T; con = pd.DataFrame(con).T
common = [c for c in rep.columns if c in pub.columns]
diff = (rep[common] - pub.loc[rep.index, common]).abs()
print(f"replication vs published: {len(common)} features, max |diff| = {np.nanmax(diff.values):.2e}, "
      f"cells off by >1e-9: {int((diff > 1e-9).sum().sum())} of {diff.notna().sum().sum()}")
con.to_csv(OUT / "features_consistent_edges.csv")
print("consistent-edge features:", con.shape)
