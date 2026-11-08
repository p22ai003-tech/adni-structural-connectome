"""Uniform connectome rebuild, one participant (Phase 1 option 1, started 2026-10-24).

One recipe for every participant, from inputs that exist for all 530:
  tracks   /data/derivatives/tracks/<sid>_I<id>/tracks_final_3000k.tck   (3M iFOD2, ACT, GMWMI seeding)
  weights  sift_weights.txt (SIFT2), per-streamline means fa/md/ad/rd_mean.tsf (tcksample -stat_tck mean)
  T1       /data/derivatives/dwi_t1_bbr/<sid>_I<id>_t1brain_ras.nii.gz   (T1 brain already in DWI space, BBR)
Steps: ANTs SyN (quick) MNI152 1 mm brain -> T1-in-DWI-space; AAL3 warped (nearest neighbour) onto a 1 mm grid of
DWI space; labels mapped to the 166 analysis nodes; tck2connectome assigns every streamline (radial search R mm);
all nine matrices are built from the SAME streamlines and assignments: count, fd_sum, count_invnodevol, len_mean,
invlen_mean, fa_mean, md_mean, rd_mean, ad_mean. Nothing published is overwritten.

Usage: rb_subject.py <sid> [--radius 2] [--out /data/derivatives/connectomes_uniform_v1] [--keep]
"""
import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("sid"); ap.add_argument("--radius", type=float, default=2.0)
ap.add_argument("--out", default="/data/derivatives/connectomes_uniform_v1"); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--keep", action="store_true", help="keep the assignment file")
a = ap.parse_args()
DER = Path("/data/derivatives"); MR = "/home/ec2-user/mrtrix3/bin"; ANTS = "/home/ec2-user/exp/.envs/ants-2.6.5/bin"
ATLAS = Path("/home/ec2-user/exp/atlas/AAL/AAL3v1_1mm.nii.gz"); MNI = Path("/home/ec2-user/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz")
MAP = Path("/home/ec2-user/exp/data/derivatives/qc/analysis_cohort/19_network_analysis/network_mapping_used.csv")
tdir = Path(sorted(glob.glob(str(DER / f"tracks/{a.sid}_I*")))[0]); unit = tdir.name
OUT = Path(a.out); W = OUT / "work" / unit; W.mkdir(parents=True, exist_ok=True); M = OUT / "matrices"; M.mkdir(exist_ok=True)
tag = f"r{a.radius:g}"; done = M / f"SC_AAL166_{unit}_{tag}_md_mean.csv"
if done.is_file(): print("exists", unit, tag); sys.exit(0)
env = dict(os.environ, ANTSPATH=ANTS + "/", PATH=f"{ANTS}:{MR}:/home/ec2-user/fsl/bin:" + os.environ["PATH"],
           ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(a.threads), OMP_NUM_THREADS="1", FSLOUTPUTTYPE="NIFTI_GZ")
log = open(W / f"rb_{tag}.log", "a"); t0 = time.time(); qc = {"unit": unit, "radius": a.radius}


def run(cmd):
    log.write("$ " + " ".join(map(str, cmd)) + "\n"); log.flush()
    subprocess.run(list(map(str, cmd)), check=True, env=env, stdout=log, stderr=subprocess.STDOUT)


t1 = DER / f"dwi_t1_bbr/{unit}_t1brain_ras.nii.gz"
nodes = W / "aal3_nodes166_dwi1mm.nii.gz"
if not nodes.is_file():
    if not (W / "mni2t1_1Warp.nii.gz").is_file():
        run([f"{ANTS}/antsRegistrationSyNQuick.sh", "-d", "3", "-f", t1, "-m", MNI, "-o", W / "mni2t1_", "-t", "s", "-n", a.threads])
    run([f"{MR}/mrgrid", t1, "regrid", "-voxel", "1", W / "ref1mm.nii.gz", "-force", "-quiet"])
    run([f"{ANTS}/antsApplyTransforms", "-d", "3", "-i", ATLAS, "-r", W / "ref1mm.nii.gz", "-o", W / "aal3_dwi1mm.nii.gz",
         "-n", "NearestNeighbor", "-t", W / "mni2t1_1Warp.nii.gz", "-t", W / "mni2t1_0GenericAffine.mat"])
    import nibabel as nib
    img = nib.load(W / "aal3_dwi1mm.nii.gz"); lab = np.rint(np.asarray(img.dataobj)).astype(int)
    mp = pd.read_csv(MAP); lut = np.zeros(max(400, lab.max() + 1), int); lut[mp.atlas_value.astype(int)] = mp.matrix_idx.astype(int)
    out_img = nib.Nifti1Image(lut[lab].astype(np.int32), img.affine); out_img.set_data_dtype(np.int32); nib.save(out_img, nodes)
    # registration QC: normalised cross-correlation of the warped template with the target T1, inside the brain
    run([f"{ANTS}/antsApplyTransforms", "-d", "3", "-i", MNI, "-r", t1, "-o", W / "mni_in_t1.nii.gz", "-n", "Linear",
         "-t", W / "mni2t1_1Warp.nii.gz", "-t", W / "mni2t1_0GenericAffine.mat"])
import nibabel as nib
nd = np.rint(np.asarray(nib.load(nodes).dataobj)).astype(int); vol = np.bincount(nd.ravel(), minlength=167)[1:167].astype(float)
x = np.asarray(nib.load(t1).dataobj, float); y = np.asarray(nib.load(W / "mni_in_t1.nii.gz").dataobj, float); m = (x > 0) & (y > 0)
qc.update(reg_ncc=float(np.corrcoef(x[m], y[m])[0, 1]), nodes_present=int((vol > 0).sum()), min_node_vox=int(vol.min()))

asg_f = W / f"assign_{tag}.txt"
run([f"{MR}/tck2connectome", tdir / "tracks_final_3000k.tck", nodes, W / f"count_mrtrix_{tag}.csv", "-assignment_radial_search", a.radius,
     "-out_assignments", asg_f, "-symmetric", "-zero_diagonal", "-nthreads", a.threads, "-force", "-quiet"])
lens = W / "lengths.txt"
if not lens.is_file():
    run([f"{MR}/tckstats", tdir / "tracks_final_3000k.tck", "-dump", lens, "-quiet", "-force"])


def vec(path):
    with open(path) as fh:
        return np.array(" ".join(l for l in fh if not l.startswith("#")).split(), dtype=float)


asg = pd.read_csv(asg_f, comment="#", sep=r"\s+", header=None, engine="c").values.astype(int)
L, w = vec(lens), vec(tdir / "sift_weights.txt"); n = len(asg)
assert len(L) == n and len(w) == n, (len(L), len(w), n)
i, j = asg[:, 0], asg[:, 1]; ok = (i > 0) & (j > 0) & (i != j); i, j = i[ok] - 1, j[ok] - 1
qc.update(streamlines=int(n), assigned_both=float(ok.mean()))


def edge_sum(v):
    S = np.zeros((166, 166)); np.add.at(S, (i, j), v); return S + S.T


cnt = edge_sum(np.ones(ok.sum()))
mats = {"count": cnt, "fd_sum": edge_sum(w[ok]), "count_invnodevol": edge_sum(2.0 / (vol[i] + vol[j])),
        "len_mean": np.divide(edge_sum(L[ok]), cnt, out=np.zeros_like(cnt), where=cnt > 0),
        "invlen_mean": np.divide(edge_sum(1.0 / np.maximum(L[ok], 1e-6)), cnt, out=np.zeros_like(cnt), where=cnt > 0)}
for k in ["fa", "md", "rd", "ad"]:
    v = vec(tdir / f"{k}_mean.tsf"); assert len(v) == n, (k, len(v), n); v = v[ok]; good = np.isfinite(v) & (v > 0)
    S = np.zeros((166, 166)); C = np.zeros((166, 166)); np.add.at(S, (i[good], j[good]), v[good]); np.add.at(C, (i[good], j[good]), 1)
    S, C = S + S.T, C + C.T; mats[f"{k}_mean"] = np.divide(S, C, out=np.zeros_like(S), where=C > 0)
iu = np.triu_indices(166, 1); pres = cnt > 0
qc.update(density=float(pres[iu].mean()), zero_nodes=int((pres.sum(1) == 0).sum()),
          md_coverage=float(((mats["md_mean"] > 0) & pres)[iu].sum() / max(pres[iu].sum(), 1)), seconds=round(time.time() - t0))
for k, mtx in mats.items():
    np.savetxt(M / f"SC_AAL166_{unit}_{tag}_{k}.csv", mtx, delimiter=",", fmt="%.10g")
json.dump(qc, open(M / f"SC_AAL166_{unit}_{tag}_qc.json", "w"), indent=1)
if not a.keep: asg_f.unlink()
print(json.dumps(qc))
