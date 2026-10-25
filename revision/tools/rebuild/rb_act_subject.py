"""Phase 5 tractography-settings check, one participant: the uniform v3 recipe re-run with anatomically constrained
tractography (ACT) instead of unconstrained tracking, everything else identical.

  FOD      re-derived exactly as in v3 (eddy output -> N4 bias correction -> v3 mask and b=1000 WM response -> CSD lmax 8)
  5TT      from the participant's FastSurfer aparc+aseg already in diffusion space (work/<unit>/aparc_aseg_dwi1mm.nii.gz,
           written by the v3 DKT step): cortical GM (cortex, cerebellar cortex), subcortical GM (thalamus, basal ganglia,
           hippocampus, amygdala, accumbens, ventral DC), WM (cerebral and cerebellar WM, brainstem, corpus callosum, WM
           hypointensities), CSF (ventricles, choroid plexus); the label rule of MRtrix 5ttgen freesurfer
  tracking tckgen iFOD2 -act -backtrack -crop_at_gmwmi, dynamic seeding, 3M streamlines, cutoff 0.06, 10-250 mm, angle 45;
           tcksift2 -act; tensor sampling, atlas (the v3 AAL3 nodes) and assignment (radial 2 mm) unchanged
Output: /data/derivatives/connectomes_uniform_v3_act/{matrices,work}; heavy intermediates deleted.
Usage: rb_act_subject.py <unit> [--threads 8]
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib

ap = argparse.ArgumentParser(); ap.add_argument("unit"); ap.add_argument("--threads", type=int, default=8); a = ap.parse_args()
DER = Path("/data/derivatives"); MR = "/home/ec2-user/mrtrix3/bin"; ANTS = "/home/ec2-user/exp/.envs/ants-2.6.5/bin"
W0 = DER / "connectomes_uniform_v3/work" / a.unit; OUT = DER / "connectomes_uniform_v3_act"
W = OUT / "work" / a.unit; W.mkdir(parents=True, exist_ok=True); M = OUT / "matrices"; M.mkdir(exist_ok=True)
if (M / f"SC_AAL166_{a.unit}_qc.json").is_file(): print("exists", a.unit); sys.exit(0)
env = dict(os.environ, ANTSPATH=ANTS + "/", PATH=f"{ANTS}:{MR}:" + os.environ["PATH"], OMP_NUM_THREADS="1",
           MRTRIX_RNG_SEED=str(int(hashlib.md5(a.unit.encode()).hexdigest()[:7], 16)))
log = open(W / "act.log", "a"); t0 = time.time(); qc = {"unit": a.unit, "recipe": "uniform_v3_act", "select": 3_000_000}


def run(cmd):
    log.write(f"[{time.strftime('%H:%M:%S')}] $ " + " ".join(map(str, cmd)) + "\n"); log.flush()
    subprocess.run(list(map(str, cmd)), check=True, env=env, stdout=log, stderr=subprocess.STDOUT)


# ---------------- FOD (v3 recipe; mask and response from the v3 run)
fod = W / "wmfod.mif"
if not fod.is_file():
    dwi = DER / f"eddy/{a.unit}_preproc.mif"; src = dwi
    if subprocess.run([f"{MR}/mrinfo", dwi, "-dwgrad"], env=env, capture_output=True).returncode != 0:
        run([f"{MR}/mrinfo", DER / f"mif_dwi/{a.unit}.mif", "-export_grad_mrtrix", W / "grad_raw.b", "-force"])
        run([f"{MR}/mrconvert", dwi, W / "dwi_withgrad.mif", "-grad", W / "grad_raw.b", "-force", "-quiet"]); src = W / "dwi_withgrad.mif"
    run([f"{MR}/dwibiascorrect", "ants", src, W / "dwi_unbiased.mif", "-force", "-quiet", "-nthreads", a.threads])
    run([f"{MR}/dwi2fod", "csd", W / "dwi_unbiased.mif", W0 / "wm_csd.txt", fod, "-mask", W0 / "mask.mif", "-lmax", "8", "-force",
         "-quiet", "-nthreads", a.threads])

# ---------------- 5TT from aparc+aseg in diffusion space
tt = W / "5tt.mif"
if not tt.is_file():
    img = nib.load(W0 / "aparc_aseg_dwi1mm.nii.gz"); L = np.rint(np.asarray(img.dataobj)).astype(int)
    cgm = ((L >= 1000) & (L < 3000)) | np.isin(L, [3, 42, 8, 47])
    sgm = np.isin(L, [10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60])
    wm = np.isin(L, [2, 41, 7, 46, 16, 77, 85, 251, 252, 253, 254, 255])
    csf = np.isin(L, [4, 5, 14, 15, 24, 31, 43, 44, 63, 72])
    five = np.stack([cgm, sgm, wm, csf, np.zeros_like(cgm)], -1).astype(np.float32)
    o = nib.Nifti1Image(five, img.affine); nib.save(o, W / "5tt.nii.gz")
    run([f"{MR}/mrconvert", W / "5tt.nii.gz", tt, "-force", "-quiet"])
    qc["tissue_voxels"] = {k: int(v.sum()) for k, v in zip(["cGM", "sGM", "WM", "CSF"], [cgm, sgm, wm, csf])}

tck, wts = W / "tracks.tck", W / "sift2_weights.txt"
if not tck.is_file():
    s = time.time()
    run([f"{MR}/tckgen", fod, W / "tracks_tmp.tck", "-algorithm", "iFOD2", "-act", tt, "-backtrack", "-crop_at_gmwmi",
         "-seed_dynamic", fod, "-select", 3_000_000, "-cutoff", "0.06", "-minlength", "10", "-maxlength", "250", "-angle", "45",
         "-nthreads", a.threads, "-force", "-quiet"])
    os.replace(W / "tracks_tmp.tck", tck); qc["sec_tckgen"] = round(time.time() - s)
if not wts.is_file():
    run([f"{MR}/tcksift2", tck, fod, wts, "-act", tt, "-nthreads", a.threads, "-force", "-quiet"])
for k in ["fa", "md", "ad", "rd"]:
    if not (W / f"{k}_mean.txt").is_file():
        run([f"{MR}/tcksample", tck, W0 / f"{k}.mif", W / f"{k}_mean.txt", "-stat_tck", "mean", "-nthreads", a.threads, "-force", "-quiet"])
run([f"{MR}/tckstats", tck, "-dump", W / "lengths.txt", "-force", "-quiet"])
nodes = W0 / "aal3_nodes166_dwi1mm.nii.gz"
run([f"{MR}/tck2connectome", tck, nodes, W / "count_mrtrix.csv", "-assignment_radial_search", 2.0, "-out_assignments",
     W / "assignments.txt", "-symmetric", "-zero_diagonal", "-nthreads", a.threads, "-force", "-quiet"])


def vec(path):
    with open(path) as fh:
        return np.array(" ".join(l for l in fh if not l.startswith("#")).split(), dtype=float)


nd = np.rint(np.asarray(nib.load(nodes).dataobj)).astype(int); vol = np.bincount(nd.ravel(), minlength=167)[1:167].astype(float)
asg = pd.read_csv(W / "assignments.txt", comment="#", sep=r"\s+", header=None, engine="c").values.astype(int)
Ls, w = vec(W / "lengths.txt"), vec(wts); n = len(asg)
i, j = asg[:, 0], asg[:, 1]; ok = (i > 0) & (j > 0) & (i != j); i, j = i[ok] - 1, j[ok] - 1


def edge_sum(v):
    S = np.zeros((166, 166)); np.add.at(S, (i, j), v); return S + S.T


cnt = edge_sum(np.ones(ok.sum()))
mats = {"count": cnt, "fd_sum": edge_sum(w[ok]), "len_mean": np.divide(edge_sum(Ls[ok]), cnt, out=np.zeros_like(cnt), where=cnt > 0)}
for k in ["fa", "md", "rd", "ad"]:
    v = vec(W / f"{k}_mean.txt")[ok]; good = np.isfinite(v) & (v > 0)
    S = np.zeros((166, 166)); C = np.zeros((166, 166)); np.add.at(S, (i[good], j[good]), v[good]); np.add.at(C, (i[good], j[good]), 1)
    S, C = S + S.T, C + C.T; mats[f"{k}_mean"] = np.divide(S, C, out=np.zeros_like(S), where=C > 0)
iu = np.triu_indices(166, 1); pres = cnt > 0
qc.update(streamlines=int(n), assigned_both=float(ok.mean()), density=float(pres[iu].mean()), zero_nodes=int((pres.sum(1) == 0).sum()),
          seconds=round(time.time() - t0))
for k, mtx in mats.items():
    np.savetxt(M / f"SC_AAL166_{a.unit}_{k}.csv", mtx, delimiter=",", fmt="%.10g")
json.dump(qc, open(M / f"SC_AAL166_{a.unit}_qc.json", "w"), indent=1)
for f in [tck, wts, W / "assignments.txt", W / "lengths.txt", W / "dwi_unbiased.mif", W / "dwi_withgrad.mif", fod,
          *[W / f"{k}_mean.txt" for k in ["fa", "md", "ad", "rd"]]]:
    f.unlink(missing_ok=True)
print(json.dumps(qc))
