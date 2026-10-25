"""Second parcellation for the uniform rebuild (sir's robustness request: 'a second parcellation such as
Desikan-Killiany'): the participant's own FastSurfer DKT parcellation (aparc+aseg: 62 cortical regions, DKT protocol)
plus 14 subcortical structures = 76 nodes, built from the SAME streamlines, weights and samples as the AAL3 matrices.

T1 -> diffusion: ANTs rigid registration of the FastSurfer brain to the mean b0 (mutual information); labels carried
with nearest-neighbour interpolation onto the 1 mm diffusion grid used for AAL3. QC: Dice of the warped white matter
with FA > 0.25. Failure here never touches the AAL3 outputs.

Usage: rb_dk.py <unit> <workdir> <matrices_dir> [--dwi path] [--threads N]
"""
import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib

ap = argparse.ArgumentParser(); ap.add_argument("unit"); ap.add_argument("work"); ap.add_argument("matrices")
ap.add_argument("--dwi"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--radius", type=float, default=2.0)
a = ap.parse_args()
W, M = Path(a.work), Path(a.matrices); MR = "/home/ec2-user/mrtrix3/bin"; ANTS = "/home/ec2-user/exp/.envs/ants-2.6.5/bin"
env = dict(os.environ, ANTSPATH=ANTS + "/", PATH=f"{ANTS}:{MR}:" + os.environ["PATH"], ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(a.threads))
log = open(W / "rb_dk.log", "a"); t0 = time.time()
sid = a.unit.split("_I")[0]
cands = [Path(p) for p in sorted(glob.glob(f"/data/derivatives/fastsurfer/{sid}_I*/mri")) if (Path(p) / "aparc+aseg.mgz").exists()]
cands.sort(key=lambda p: p.parent.name != a.unit)                      # prefer the session of the diffusion scan
FS = cands[0] if cands else None


def run(cmd):
    log.write(f"[{time.strftime('%H:%M:%S')}] $ " + " ".join(map(str, cmd)) + "\n"); log.flush()
    subprocess.run(list(map(str, cmd)), check=True, env=env, stdout=log, stderr=subprocess.STDOUT)


CORT = [2, 3] + list(range(5, 32)) + [34, 35]                         # DKT protocol: 31 regions per hemisphere
LABELS = [1000 + c for c in CORT] + [2000 + c for c in CORT] + [10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]
pd.DataFrame({"node": range(1, len(LABELS) + 1), "freesurfer_label": LABELS}).to_csv(M.parent / "dkt76_nodes.csv", index=False)
qc = {"unit": a.unit, "atlas": "DKT76 (FastSurfer aparc+aseg)"}
try:
    if FS is None: raise FileNotFoundError("no FastSurfer output")
    dwi = Path(a.dwi) if a.dwi else (W / "dwi_unbiased.mif" if (W / "dwi_unbiased.mif").exists() else Path(f"/data/derivatives/eddy/{a.unit}_preproc.mif"))
    if not (W / "b0mean.nii.gz").exists():
        run(["bash", "-c", f"{MR}/dwiextract {dwi} -bzero - -quiet | {MR}/mrmath - mean {W / 'b0mean.nii.gz'} -axis 3 -force -quiet"])
    br = nib.load(FS / "brain.mgz"); nib.save(nib.Nifti1Image(np.asarray(br.dataobj, np.float32), br.affine), W / "t1brain_fs.nii.gz")
    lab = nib.load(FS / "aparc+aseg.mgz"); nib.save(nib.Nifti1Image(np.asarray(lab.dataobj).astype(np.int32), lab.affine), W / "aparc_aseg.nii.gz")
    run([f"{ANTS}/antsRegistrationSyNQuick.sh", "-d", "3", "-f", W / "b0mean.nii.gz", "-m", W / "t1brain_fs.nii.gz", "-o", W / "t12b0_", "-t", "r", "-n", a.threads])
    run([f"{ANTS}/antsApplyTransforms", "-d", "3", "-i", W / "aparc_aseg.nii.gz", "-r", W / "ref1mm.nii.gz", "-o", W / "aparc_aseg_dwi1mm.nii.gz",
         "-n", "NearestNeighbor", "-t", W / "t12b0_0GenericAffine.mat"])
    img = nib.load(W / "aparc_aseg_dwi1mm.nii.gz"); L = np.rint(np.asarray(img.dataobj)).astype(int)
    lut = np.zeros(max(3000, L.max() + 1), int); lut[LABELS] = np.arange(1, len(LABELS) + 1)
    nodes = lut[L]; o = nib.Nifti1Image(nodes.astype(np.int32), img.affine); o.set_data_dtype(np.int32); nib.save(o, W / "dkt76_nodes_dwi1mm.nii.gz")
    vol = np.bincount(nodes.ravel(), minlength=len(LABELS) + 1)[1:].astype(float)
    # registration QC on the 1 mm grid: warped white matter against FA > 0.25
    run([f"{MR}/mrgrid", W / ("fa.mif" if (W / "fa.mif").exists() else "fa.nii.gz"), "regrid", "-template", W / "ref1mm.nii.gz", W / "fa1mm.nii.gz", "-force", "-quiet"])
    fa = np.asarray(nib.load(W / "fa1mm.nii.gz").dataobj) > 0.25; wm = np.isin(L, [2, 41])
    qc.update(reg_wm_dice=float(2 * (fa & wm).sum() / max(fa.sum() + wm.sum(), 1)), nodes_present=int((vol > 0).sum()), min_node_vox=int(vol.min()))
    run([f"{MR}/tck2connectome", W / "tracks.tck", W / "dkt76_nodes_dwi1mm.nii.gz", W / "count_dkt_mrtrix.csv", "-assignment_radial_search", a.radius,
         "-out_assignments", W / "assign_dkt.txt", "-symmetric", "-zero_diagonal", "-nthreads", a.threads, "-force", "-quiet"])

    def vec(p):
        with open(p) as fh: return np.array(" ".join(l for l in fh if not l.startswith("#")).split(), dtype=float)
    asg = pd.read_csv(W / "assign_dkt.txt", comment="#", sep=r"\s+", header=None, engine="c").values.astype(int)
    Ls, w = vec(W / "lengths.txt"), vec(W / "sift2_weights.txt"); n = len(asg); N = len(LABELS)
    i, j = asg[:, 0], asg[:, 1]; ok = (i > 0) & (j > 0) & (i != j); i, j = i[ok] - 1, j[ok] - 1

    def esum(v):
        S = np.zeros((N, N)); np.add.at(S, (i, j), v); return S + S.T
    cnt = esum(np.ones(ok.sum()))
    mats = {"count": cnt, "fd_sum": esum(w[ok]), "count_invnodevol": esum(2.0 / (vol[i] + vol[j])),
            "len_mean": np.divide(esum(Ls[ok]), cnt, out=np.zeros_like(cnt), where=cnt > 0),
            "invlen_mean": np.divide(esum(1.0 / np.maximum(Ls[ok], 1e-6)), cnt, out=np.zeros_like(cnt), where=cnt > 0)}
    for k in ["fa", "md", "rd", "ad"]:
        v = vec(W / f"{k}_mean.txt")[ok]; g = np.isfinite(v) & (v > 0)
        S = np.zeros((N, N)); C = np.zeros((N, N)); np.add.at(S, (i[g], j[g]), v[g]); np.add.at(C, (i[g], j[g]), 1)
        S, C = S + S.T, C + C.T; mats[f"{k}_mean"] = np.divide(S, C, out=np.zeros_like(S), where=C > 0)
    iu = np.triu_indices(N, 1); pres = cnt > 0
    qc.update(status="ok", assigned_both=float(ok.mean()), density=float(pres[iu].mean()), zero_nodes=int((pres.sum(1) == 0).sum()))
    for k, mtx in mats.items():
        np.savetxt(M / f"SC_DKT76_{a.unit}_{k}.csv", mtx, delimiter=",", fmt="%.10g")
    (W / "assign_dkt.txt").unlink(missing_ok=True)
except Exception as exc:                                                  # never fail the AAL3 rebuild because of DKT
    qc.update(status="failed", error=f"{type(exc).__name__}: {exc}")
qc["seconds"] = round(time.time() - t0)
json.dump(qc, open(M / f"SC_DKT76_{a.unit}_qc.json", "w"), indent=1)
print(json.dumps(qc))
