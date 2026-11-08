"""Rebuild scoping: does each participant's existing 3M tractogram cover the brain?
Samples the first 30,000 streamlines and reports the share of b0 brain-mask voxels (2 mm grid) that streamline points
reach, plus the streamline bounding box. Output: one CSV line per participant."""
import sys, glob, itertools, numpy as np, nibabel as nib
sid = sys.argv[1]; T = sorted(glob.glob(f"/data/derivatives/tracks/{sid}_I*"))[0]; unit = T.split("/")[-1]
pts = np.concatenate([s for s in itertools.islice(nib.streamlines.load(f"{T}/tracks_final_3000k.tck", lazy_load=True).streamlines, 30000)])
m = nib.load(f"/data/derivatives/dwi_t1_bbr/{unit}_mask_on_b0_bbr.nii.gz"); M = np.asarray(m.dataobj) > 0
v = np.rint(nib.affines.apply_affine(np.linalg.inv(m.affine), pts)).astype(int)
ok = np.all((v >= 0) & (v < np.array(M.shape)), axis=1); v = v[ok]
hit = np.zeros(M.shape, bool); hit[v[:, 0], v[:, 1], v[:, 2]] = True
# dilate the hit map by one voxel so sparse sampling does not count as missing coverage
from scipy.ndimage import binary_dilation
hit = binary_dilation(hit, iterations=1)
cover = (hit & M).sum() / M.sum(); outside = 1 - ok.mean()
lo, hi = pts.min(0), pts.max(0)
print(f"{sid},{unit},{cover:.4f},{outside:.4f},{lo[0]:.0f},{lo[1]:.0f},{lo[2]:.0f},{hi[0]:.0f},{hi[1]:.0f},{hi[2]:.0f}")
