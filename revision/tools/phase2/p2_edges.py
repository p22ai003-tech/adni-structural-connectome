"""Phase 2 step 1 — edge-level cache: one array per edge property, participants x upper-triangle edges.

Usage: python p2_edges.py published            -> phase2/edges_published.npz  (the matrices the paper used)
       python p2_edges.py uniform_r2           -> phase2/edges_uniform_r2.npz (the uniform rebuild, when ready)

Arrays (float32, NaN where absent): len (mean streamline length, mm), fd (SIFT2 weight sum), fa, md, rd, ad.
An edge is structurally present when len > 0 and fd > 0; a diffusion value is kept only where it is > 0.
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import iit_chain as chain; chain.boot()      # IIT_ROOT and the chain settings; no effect when IIT_ROOT is unset
import os
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd

EXP = Path("/home/ec2-user/exp"); A = Path(os.environ.get("IIT_AROOT", "/home/ec2-user/exp/data/derivatives/qc/analysis_cohort"))
O = Path(os.environ.get("IIT_P2", "/home/ec2-user/exp/docs/iit_dipanjan/phase2")); O.mkdir(exist_ok=True)
SRC = sys.argv[1] if len(sys.argv) > 1 else "published"
cov = pd.read_csv(Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1")) / "covariates.csv")
NNODE = 76 if SRC.endswith("dkt") else 166
iu = np.triu_indices(NNODE, 1)
KINDS = {"len": "len_mean", "fd": "fd_sum", "count": "count", "fa": "fa_mean", "md": "md_mean", "rd": "rd_mean", "ad": "ad_mean"}

if SRC == "published":
    C = Path("/data/derivatives/connectomes")
    m = pd.read_csv(A / "00_master/master_cohort.csv", low_memory=False)
    files = {}
    for sid, iid in zip(m.subject_id, m["Image ID"].astype(str).str.replace(r"\.0$", "", regex=True)):
        f = C / f"SC_AAL166_{sid}_I{iid}_len_mean.csv"
        stem = f"{sid}_I{iid}" if f.is_file() else sorted(C.glob(f"SC_AAL166_{sid}_I*_len_mean.csv"))[0].name[10:-13]
        files[sid] = lambda k, stem=stem: C / f"SC_AAL166_{stem}_{k}.csv"
elif SRC in ("uniform_v3", "uniform_v3_dkt"):                       # the uniform rebuild of 24-25 Oct (AAL3 or DKT76 matrices)
    if chain.root() is None:
        C = Path("/data/derivatives/connectomes_uniform_v3") / ("matrices_dkt" if SRC.endswith("dkt") else "matrices")
    else:                                                        # a chain reads only its own connectomes
        C = Path(os.environ["IIT_CONN_DKT"] if SRC.endswith("dkt") else os.environ["IIT_CONN"])
    pref = "SC_DKT76" if SRC.endswith("dkt") else "SC_AAL166"
    files = {}
    for sid in cov.subject_id:
        if chain.root() is None:
            unit = Path(sorted(glob.glob(f"/data/derivatives/tracks/{sid}_I*"))[0]).name
        else:                                                    # the unit comes from the connectome itself, not from an older tree
            hits = sorted(C.glob(f"{pref}_{sid}_I*_len_mean.csv"))
            unit = hits[0].name[len(pref) + 1:-len("_len_mean.csv")] if hits else f"{sid}_none"
        files[sid] = lambda k, unit=unit: C / f"{pref}_{unit}_{k}.csv"
elif SRC.startswith("uniform_r"):
    C = Path("/data/derivatives/connectomes_uniform_v1/matrices"); tag = SRC.split("_")[1]
    files = {}
    for sid in cov.subject_id:
        unit = Path(sorted(glob.glob(f"/data/derivatives/tracks/{sid}_I*"))[0]).name
        files[sid] = lambda k, unit=unit: C / f"SC_AAL166_{unit}_{tag}_{k}.csv"
else:
    raise SystemExit(f"unknown source {SRC}")

out = {k: np.full((len(cov), len(iu[0])), np.nan, np.float32) for k in KINDS}
for r, sid in enumerate(cov.subject_id):
    for k, kind in KINDS.items():
        f = files[sid](kind)
        if not f.is_file(): continue
        v = np.loadtxt(f, delimiter=",")[iu]
        out[k][r] = np.where(v > 0, v, np.nan)
pres = np.isfinite(out["len"]) & np.isfinite(out["fd"])
for k in ["fa", "md", "rd", "ad"]: out[k][~pres] = np.nan          # diffusion values only on structurally present edges
mp = pd.read_csv(A / "19_network_analysis/network_mapping_used.csv")
net = dict(zip(mp.matrix_idx.astype(int), mp.functional_network))
if NNODE == 76:                                    # DKT: labels from the overlap with the AAL3 networks (tools/phase5/p5_dkt_map.py)
    dm = chain.dkt_map()
    net = dict(zip(pd.read_csv(dm).node.astype(int), pd.read_csv(dm).network)) if dm.is_file() else {}
np.savez_compressed(O / f"edges_{SRC}.npz", sid=cov.subject_id.values, i=iu[0], j=iu[1],
                    net_i=np.array([net.get(k + 1, "") for k in iu[0]]), net_j=np.array([net.get(k + 1, "") for k in iu[1]]), **out)
print(f"wrote edges_{SRC}.npz: {len(cov)} participants x {len(iu[0])} edges; structural edges present {pres.mean():.3f}, "
      f"with MD {np.isfinite(out['md']).sum() / max(pres.sum(), 1):.3f}")
