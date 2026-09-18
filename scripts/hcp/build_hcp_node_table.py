#!/usr/bin/env python
"""Build the locked HCP-MMP1 (Glasser 360) + FreeSurfer-aseg subcortical (19) = 379-node table.
Node ids are FIXED and shared across all subjects. Emits:
  - hcpmmp1_subcort_nodes.csv   : node_id, vol_label, hemi, structure, source
  - hcpmmp1_subcort_relabel.txt : whitespace 'vol_label new_node_id' remap (for numpy relabel)
The vol_label is the value mri_aparc2aseg(--annot HCPMMP1) writes: cortex = 1000+idx (lh) / 2000+idx (rh);
subcortical = standard aseg ids.  Cortical annot index = row position in the annot colortable (1..180).
"""
import csv
import nibabel.freesurfer.io as fsio

OUT = "/data/derivatives/parc_hcpmmp1"


def parcels(hemi):
    _, _, names = fsio.read_annot(f"/home/ec2-user/hcp_atlas/{hemi}.HCPMMP1.annot")
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    # index 0 = unknown/medial wall; areas are 1..180
    return names


lh, rh = parcels("lh"), parcels("rh")
assert len(lh) == 181 and len(rh) == 181, (len(lh), len(rh))

rows = []          # node_id, vol_label, hemi, structure, source
nid = 0
for idx in range(1, 181):                     # lh cortical -> nodes 1..180
    nid += 1
    rows.append((nid, 1000 + idx, "L", lh[idx], "HCPMMP1"))
for idx in range(1, 181):                     # rh cortical -> nodes 181..360
    nid += 1
    rows.append((nid, 2000 + idx, "R", rh[idx], "HCPMMP1"))

# 19 subcortical (aseg) — fixed order, limbic-relevant nodes included (hipp/amyg/thal)
SUBCORT = [
    (10, "L", "Thalamus"), (11, "L", "Caudate"), (12, "L", "Putamen"), (13, "L", "Pallidum"),
    (17, "L", "Hippocampus"), (18, "L", "Amygdala"), (26, "L", "Accumbens"), (28, "L", "VentralDC"), (8, "L", "Cerebellum"),
    (49, "R", "Thalamus"), (50, "R", "Caudate"), (51, "R", "Putamen"), (52, "R", "Pallidum"),
    (53, "R", "Hippocampus"), (54, "R", "Amygdala"), (58, "R", "Accumbens"), (60, "R", "VentralDC"), (47, "R", "Cerebellum"),
    (16, "M", "BrainStem"),
]
for aseg, hemi, struct in SUBCORT:
    nid += 1
    rows.append((nid, aseg, hemi, struct, "aseg"))

assert nid == 379, nid
with open(f"{OUT}/hcpmmp1_subcort_nodes.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["node_id", "vol_label", "hemi", "structure", "source"])
    w.writerows(rows)
with open(f"{OUT}/hcpmmp1_subcort_relabel.txt", "w") as f:
    for nid_, vol, *_ in rows:
        f.write(f"{vol} {nid_}\n")

print(f"wrote {OUT}/hcpmmp1_subcort_nodes.csv  ({nid} nodes: 360 cortical + 19 subcortical)")
print("cortical sample:", rows[0][3], "|", rows[1][3], "| ... |", rows[179][3])
print("subcortical:", ", ".join(f"{r[3]}-{r[2]}" for r in rows[360:]))
