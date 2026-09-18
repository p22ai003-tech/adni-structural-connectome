#!/usr/bin/env python
"""Relabel a b0-space HCPMMP1+aseg volume to contiguous 1..379 node ids, and emit node volumes (mm3).
Usage: relabel_hcp.py <atlas_b0.nii.gz> <relabel.txt> <out_nodes.nii.gz> <out_volumes.csv>
  relabel.txt: whitespace 'vol_label new_node_id' per line (379 subcortical+cortical entries).
Voxels whose value is not in the map become 0 (background/dropped, e.g. unknown cortex 1000/2000).
"""
import sys
import numpy as np
import nibabel as nib

atlas_path, lut_path, out_nodes, out_vols = sys.argv[1:5]

img = nib.load(atlas_path)
data = np.asanyarray(img.dataobj).astype(np.int32)

# build vol_label -> node_id map
pairs = []
with open(lut_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        vol, nid = line.split()
        pairs.append((int(vol), int(nid)))
n_nodes = max(nid for _, nid in pairs)          # 379

# vectorized remap via lookup array sized to max source label
max_src = max(int(data.max()), max(v for v, _ in pairs))
lut = np.zeros(max_src + 1, dtype=np.int32)
for vol, nid in pairs:
    if vol <= max_src:
        lut[vol] = nid
out = lut[np.clip(data, 0, max_src)]

nii = nib.Nifti1Image(out.astype(np.int32), img.affine, img.header)
nii.set_data_dtype(np.int32)
nib.save(nii, out_nodes)

# per-node volume in mm3
voxvol = abs(np.linalg.det(img.affine[:3, :3]))
counts = np.bincount(out.ravel(), minlength=n_nodes + 1)
with open(out_vols, "w") as f:
    f.write("node_id,volume_mm3,n_voxels\n")
    for nid in range(1, n_nodes + 1):
        nv = int(counts[nid])
        f.write(f"{nid},{nv * voxvol:.4f},{nv}\n")

present = int((counts[1:n_nodes + 1] > 0).sum())
print(f"relabel: {n_nodes} nodes, {present} present in this subject, voxvol={voxvol:.3f}mm3 -> {out_nodes}")
if present < n_nodes:
    missing = [nid for nid in range(1, n_nodes + 1) if counts[nid] == 0]
    print(f"  WARN {n_nodes - present} nodes absent (ids: {missing[:20]}{'...' if len(missing) > 20 else ''})")
