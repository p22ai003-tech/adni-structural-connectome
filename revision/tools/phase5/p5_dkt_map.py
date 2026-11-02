"""Phase 5 step 1 — functional-network labels for the 76 DKT regions of the second parcellation.

The AAL3 regions carry the manuscript's ten network labels (network_mapping_used.csv). Both parcellations exist in every
participant's diffusion space (work/<unit>/aal3_nodes166_dwi1mm.nii.gz and dkt76_nodes_dwi1mm.nii.gz, from the rebuild),
so each DKT region gets the network that covers most of its voxels, summed over 60 ADNI-3/4 control participants. The
DKT parcellation has no cerebellum or brainstem, so those two networks are absent.
Writes /data/derivatives/connectomes_uniform_v3/dkt76_network_mapping.csv (node, label, name, network, share).
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import iit_chain as chain; chain.boot()      # IIT_ROOT and the chain settings; no effect when IIT_ROOT is unset
import os
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib

U = Path("/data/derivatives/connectomes_uniform_v3"); W = U / "work"
OUT_MAP = chain.dkt_map(for_writing=True)    # a chain writes its own copy; with no root this is today's file
A = Path(os.environ.get("IIT_AROOT", "/data/derivatives/qc/analysis_cohort_uniform"))
P1 = Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1_uniform"))
CORT = {2: "caudalanteriorcingulate", 3: "caudalmiddlefrontal", 5: "cuneus", 6: "entorhinal", 7: "fusiform", 8: "inferiorparietal",
        9: "inferiortemporal", 10: "isthmuscingulate", 11: "lateraloccipital", 12: "lateralorbitofrontal", 13: "lingual",
        14: "medialorbitofrontal", 15: "middletemporal", 16: "parahippocampal", 17: "paracentral", 18: "parsopercularis",
        19: "parsorbitalis", 20: "parstriangularis", 21: "pericalcarine", 22: "postcentral", 23: "posteriorcingulate", 24: "precentral",
        25: "precuneus", 26: "rostralanteriorcingulate", 27: "rostralmiddlefrontal", 28: "superiorfrontal", 29: "superiorparietal",
        30: "superiortemporal", 31: "supramarginal", 34: "transversetemporal", 35: "insula"}
SUB = {10: "Left-Thalamus", 11: "Left-Caudate", 12: "Left-Putamen", 13: "Left-Pallidum", 17: "Left-Hippocampus", 18: "Left-Amygdala",
       26: "Left-Accumbens", 49: "Right-Thalamus", 50: "Right-Caudate", 51: "Right-Putamen", 52: "Right-Pallidum",
       53: "Right-Hippocampus", 54: "Right-Amygdala", 58: "Right-Accumbens"}
nodes = pd.read_csv(U / "dkt76_nodes.csv")
nodes["name"] = [(("lh-" if l < 2000 else "rh-") + CORT[l % 1000]) if l >= 1000 else SUB[l] for l in nodes.freesurfer_label]
mp = pd.read_csv(A / "19_network_analysis/network_mapping_used.csv")
NETS = sorted(mp.functional_network.unique())
net_of = np.zeros(200, int); net_of[mp.matrix_idx.values] = [NETS.index(n) + 1 for n in mp.functional_network]
cov = pd.read_csv(P1 / "covariates.csv"); qc = pd.read_csv(P1 / "qc.csv", index_col=0)
cn = cov[(cov.group == "CN") & (chain.primary(cov) == 1) & cov.subject_id.map(qc.qc_fail).eq(0)].subject_id
units = [p for s in cn for p in sorted(W.glob(f"{s}_I*")) if (p / "dkt76_nodes_dwi1mm.nii.gz").is_file() and (p / "aal3_nodes166_dwi1mm.nii.gz").is_file()][:60]
C = np.zeros((77, len(NETS) + 1))
for u in units:
    a = np.asarray(nib.load(u / "aal3_nodes166_dwi1mm.nii.gz").dataobj).astype(int)
    d = np.asarray(nib.load(u / "dkt76_nodes_dwi1mm.nii.gz").dataobj).astype(int)
    if a.shape != d.shape: continue
    m = d > 0; np.add.at(C, (d[m], net_of[np.clip(a[m], 0, 199)]), 1)
lab = C[:, 1:].argmax(1); share = C[:, 1:].max(1) / np.maximum(C[:, 1:].sum(1), 1)
nodes["network"] = [NETS[lab[k]] for k in nodes.node]; nodes["share"] = [share[k].round(3) for k in nodes.node]
# subcortical structures present in both atlases take the AAL3 label of the same structure (overlap is degraded by small
# registration offsets, e.g. the right amygdala bleeds into the hippocampus)
SAME = {"Hippocampus": "Hippocampus_L", "Amygdala": "Amygdala_L", "Thalamus": None, "Caudate": "Caudate_L", "Putamen": "Putamen_L",
        "Pallidum": "Pallidum_L"}
lab_of = dict(zip(mp.atlas_label, mp.functional_network))
for i, r in nodes.iterrows():
    key = r["name"].split("-")[-1]
    if key in SAME:
        nodes.loc[i, "network"] = lab_of.get(SAME[key], "Subcortical") if SAME[key] else "Subcortical"
        nodes.loc[i, "rule"] = "same structure in AAL3"
nodes["rule"] = nodes.get("rule", pd.Series(index=nodes.index, dtype=object)).fillna("largest voxel overlap")
nodes["unlabelled_share"] = [(C[k, 0] / max(C[k].sum(), 1)).round(3) for k in nodes.node]
nodes.to_csv(OUT_MAP, index=False)
print(f"{len(units)} control participants; regions per network:", nodes.network.value_counts().to_dict())
print(nodes.sort_values("share").head(8).to_string(index=False))
