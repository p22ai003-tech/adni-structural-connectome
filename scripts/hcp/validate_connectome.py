#!/usr/bin/env python
"""Sanity-gate a subject's HCP connectome before the batch trusts the pipeline.
Prints VALIDATE_OK or VALIDATE_FAIL with reasons. Catches gross failures
(mis-sized matrix, non-symmetric, empty, atlas fell off the brain, dead limbic hubs).
Usage: validate_connectome.py <fsid>
"""
import sys
import numpy as np

fsid = sys.argv[1]
OUT = f"/data/derivatives/connectomes_hcp_fs/SC_HCPMMP1_{fsid}"
WORK = f"/data/derivatives/parc_hcpmmp1/{fsid}"
N = 379
# node ids (1-based) for limbic hubs from the locked table
HIPP_L, HIPP_R, THAL_L, THAL_R, AMY_L, AMY_R = 365, 374, 361, 370, 366, 375

fail = []
try:
    count = np.loadtxt(f"{OUT}_count.csv", delimiter=",")
except Exception as e:
    print(f"VALIDATE_FAIL: cannot read count matrix: {e}")
    sys.exit(1)

if count.shape != (N, N):
    fail.append(f"shape {count.shape} != ({N},{N})")
if not np.allclose(np.diag(count), 0):
    fail.append("nonzero diagonal")
if not np.allclose(count, count.T, atol=1e-6):
    fail.append("not symmetric")

present = int(((count.sum(0) + count.sum(1)) > 0).sum())
if present < 330:
    fail.append(f"only {present}/379 nodes connected (atlas may be mis-registered / off-brain)")

# atlas L/R symmetry is the real detector of the mis-registration/smearing bug class.
# Read per-node voxel counts; a well-placed atlas has near-symmetric hemispheres.
try:
    import csv as _csv
    vox = {}
    with open(f"{WORK}/node_volumes.csv") as _f:
        for _r in _csv.DictReader(_f):
            vox[int(_r["node_id"])] = int(_r["n_voxels"])
    l_cort = sum(vox.get(n, 0) for n in range(1, 181))       # nodes 1..180 = LH cortex
    r_cort = sum(vox.get(n, 0) for n in range(181, 361))      # 181..360 = RH cortex
    ratio = l_cort / max(r_cort, 1)
    if not (0.6 <= ratio <= 1.67):
        fail.append(f"cortical L/R atlas-voxel asymmetry {ratio:.2f} (mis-registration?)")
    # subcortical hub voxel symmetry (L,R node ids): thal 361/370, hipp 365/374, amyg 366/375, put 363/372
    for nm, (l, r) in {"Thal": (361, 370), "Hipp": (365, 374), "Put": (363, 372)}.items():
        vr = vox.get(l, 0) / max(vox.get(r, 0), 1)
        if not (0.35 <= vr <= 2.85):
            fail.append(f"{nm} L/R atlas-voxel asymmetry {vr:.2f}")
except FileNotFoundError:
    pass  # node_volumes only present after a full pipeline run; skip if validating standalone

density = (count > 0).sum() / (N * (N - 1))
if not (0.05 <= density <= 0.98):
    fail.append(f"implausible edge density {density:.3f}")

total = count.sum()
if total <= 0:
    fail.append("zero total streamlines")

# limbic hubs must be connected (well-connected in a healthy parcellation)
for name, nid in [("L-Hipp", HIPP_L), ("R-Hipp", HIPP_R), ("L-Thal", THAL_L),
                  ("R-Thal", THAL_R), ("L-Amyg", AMY_L), ("R-Amyg", AMY_R)]:
    deg = count[nid - 1].sum() + count[:, nid - 1].sum()
    if deg <= 0:
        fail.append(f"{name} (node {nid}) has zero degree")

# metric matrices exist + finite where edges exist
for m in ["fa", "md", "rd", "ad", "len", "fd_sum", "invlen", "count_invnodevol"]:
    suffix = {"len": "len_mean", "invlen": "invlen_mean", "fd_sum": "fd_sum",
              "count_invnodevol": "count_invnodevol"}.get(m, f"{m}_mean")
    try:
        mm = np.loadtxt(f"{OUT}_{suffix}.csv", delimiter=",")
        if mm.shape != (N, N):
            fail.append(f"{suffix} shape {mm.shape}")
        if not np.isfinite(mm).all():
            fail.append(f"{suffix} has non-finite values")
    except Exception as e:
        fail.append(f"{suffix} missing/unreadable: {e}")

if fail:
    print("VALIDATE_FAIL: " + "; ".join(fail))
    sys.exit(1)
print(f"VALIDATE_OK: {present}/379 nodes, density={density:.3f}, total_streamlines={total:.0f}, "
      f"L-Hipp deg={count[HIPP_L-1].sum()+count[:,HIPP_L-1].sum():.0f}")
