"""Uniform connectome rebuild v2, one participant: new tractography, one recipe for all 530 (started 2026-10-24).

Why v2: the scoping showed that the stored 3M tractograms (tracks_final_3000k.tck) cover only part of the brain for
about half the cohort (mostly the participants of the 'current' route, whose own tractograms were not kept), and that
the T1-to-diffusion registrations are unreliable. v2 therefore depends on nothing but the diffusion data:

  FOD      /data/derivatives/fod/<unit>/wmfod_final.mif (or wmfod.mif)
  tensor   /data/derivatives/dti/<unit>/{fa,md,ad,rd}.mif
  1. tckgen iFOD2, dynamic seeding, no ACT, 3M streamlines, cutoff 0.06, length 10-250 mm, angle 45
     (the parameters of the current route, which produced most published matrices)
  2. tcksift2 weights;  3. tcksample per-streamline mean FA, MD, AxD, RD
  4. atlas: ANTs SyN (quick) FMRIB58 FA template -> participant FA (no T1 needed); AAL3 warped onto a 1 mm grid of
     diffusion space with nearest-neighbour interpolation; labels mapped to the 166 analysis nodes
  5. tck2connectome, radial search 2 mm, and all nine matrices from the same streamlines and assignments
Heavy intermediates (tractogram, weights, per-streamline samples, assignments) are deleted once the matrices are
written unless --keep. Output: /data/derivatives/connectomes_uniform_v2/{matrices,work}.

Usage: rb2_subject.py <sid> [--threads 8] [--select 3000000] [--keep]
"""
import argparse, glob, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("sid"); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--select", type=int, default=3_000_000); ap.add_argument("--radius", type=float, default=2.0)
ap.add_argument("--out", default="/data/derivatives/connectomes_uniform_v2"); ap.add_argument("--keep", action="store_true")
ap.add_argument("--from-dwi", action="store_true", help="rebuild bias correction, mask, response, FOD and tensor maps from the "
                "eddy-corrected DWI (/data/derivatives/eddy/<unit>_preproc.mif) instead of using the stored FOD and tensor maps")
a = ap.parse_args()
DER = Path("/data/derivatives"); MR = "/home/ec2-user/mrtrix3/bin"; ANTS = "/home/ec2-user/exp/.envs/ants-2.6.5/bin"
ATLAS = Path("/home/ec2-user/exp/atlas/AAL/AAL3v1_1mm.nii.gz"); FA_T = Path("/home/ec2-user/fsl/data/standard/FMRIB58_FA_1mm.nii.gz")
MAP = Path("/home/ec2-user/exp/data/derivatives/qc/analysis_cohort/19_network_analysis/network_mapping_used.csv")
unit = Path(sorted(glob.glob(str(DER / f"tracks/{a.sid}_I*")))[0]).name
OUT = Path(a.out); W = OUT / "work" / unit; W.mkdir(parents=True, exist_ok=True); M = OUT / "matrices"; M.mkdir(exist_ok=True)
if (M / f"SC_AAL166_{unit}_qc.json").is_file(): print("exists", unit); sys.exit(0)
LOCK = W / "running.pid"                  # one worker per participant, even with several drivers
if LOCK.is_file():
    try:
        os.kill(int(LOCK.read_text().strip()), 0); print("busy", unit); sys.exit(0)
    except (ProcessLookupError, ValueError):
        pass                              # stale lock from a stopped machine: take over
LOCK.write_text(str(os.getpid()))
env = dict(os.environ, ANTSPATH=ANTS + "/", PATH=f"{ANTS}:{MR}:" + os.environ["PATH"], ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(a.threads),
           OMP_NUM_THREADS="1", MRTRIX_RNG_SEED=str(int(hashlib.md5(unit.encode()).hexdigest()[:7], 16)))   # reproducible per participant
log = open(W / "rb2.log", "a"); t0 = time.time(); qc = {"unit": unit, "recipe": "uniform_v2", "select": a.select, "radius": a.radius}
fod = next(p for p in [DER / f"fod/{unit}/wmfod_final.mif", DER / f"fod/{unit}/wmfod.mif"] if p.exists())
dti = DER / f"dti/{unit}"
if a.from_dwi:                          # one upstream recipe: N4 bias correction, mask, dhollander WM response, CSD lmax 8, DTI
    qc["recipe"] = "uniform_v3_from_dwi"; dwi = DER / f"eddy/{unit}_preproc.mif"; fod, dti = W / "wmfod.mif", W
    def upstream():
        src = dwi                          # some older eddy outputs lost the gradient table: take it from the raw DWI
        if subprocess.run([f"{MR}/mrinfo", dwi, "-dwgrad"], env=env, capture_output=True).returncode != 0:
            run([f"{MR}/mrinfo", DER / f"mif_dwi/{unit}.mif", "-export_grad_mrtrix", W / "grad_raw.b", "-force"])
            run([f"{MR}/mrconvert", dwi, W / "dwi_withgrad.mif", "-grad", W / "grad_raw.b", "-force", "-quiet"])
            src = W / "dwi_withgrad.mif"; qc["grad_source"] = "raw DWI (not eddy-rotated)"
        run([f"{MR}/dwibiascorrect", "ants", src, W / "dwi_unbiased.mif", "-force", "-quiet", "-nthreads", a.threads])
        run([f"{MR}/dwi2mask", W / "dwi_unbiased.mif", W / "mask.mif", "-force", "-quiet"])
        run([f"{MR}/dwi2response", "dhollander", W / "dwi_unbiased.mif", W / "wm.txt", W / "gm.txt", W / "csf.txt", "-mask", W / "mask.mif",
             "-force", "-quiet", "-nthreads", a.threads])
        rows = [l for l in open(W / "wm.txt") if l.strip() and not l.startswith("#")]   # single-shell CSD: the b=1000 row
        open(W / "wm_csd.txt", "w").write(rows[-1])
        run([f"{MR}/dwi2fod", "csd", W / "dwi_unbiased.mif", W / "wm_csd.txt", fod, "-mask", W / "mask.mif", "-lmax", "8", "-force", "-quiet",
             "-nthreads", a.threads])
        run([f"{MR}/dwi2tensor", W / "dwi_unbiased.mif", W / "dt.mif", "-mask", W / "mask.mif", "-force", "-quiet", "-nthreads", a.threads])
        run([f"{MR}/tensor2metric", W / "dt.mif", "-fa", W / "fa.mif", "-adc", W / "md.mif", "-ad", W / "ad.mif", "-rd", W / "rd.mif",
             "-force", "-quiet", "-nthreads", a.threads])



def run(cmd):
    log.write(f"[{time.strftime('%H:%M:%S')}] $ " + " ".join(map(str, cmd)) + "\n"); log.flush()
    subprocess.run(list(map(str, cmd)), check=True, env=env, stdout=log, stderr=subprocess.STDOUT)


def step(name, fn):
    s = time.time(); fn(); qc[f"sec_{name}"] = round(time.time() - s)


if a.from_dwi and not fod.is_file():
    step("upstream", upstream)


tck, wts = W / "tracks.tck", W / "sift2_weights.txt"
if not tck.is_file():
    step("tckgen", lambda: run([f"{MR}/tckgen", fod, W / "tracks_tmp.tck", "-algorithm", "iFOD2", "-seed_dynamic", fod, "-select", a.select,
                               "-cutoff", "0.06", "-minlength", "10", "-maxlength", "250", "-angle", "45", "-nthreads", a.threads, "-force", "-quiet"]))
    os.replace(W / "tracks_tmp.tck", tck)
if not wts.is_file():
    step("sift2", lambda: run([f"{MR}/tcksift2", tck, fod, wts, "-nthreads", a.threads, "-force", "-quiet"]))
for k in ["fa", "md", "ad", "rd"]:
    if not (W / f"{k}_mean.txt").is_file():
        step(f"sample_{k}", lambda k=k: run([f"{MR}/tcksample", tck, dti / f"{k}.mif", W / f"{k}_mean.txt", "-stat_tck", "mean",
                                            "-nthreads", a.threads, "-force", "-quiet"]))
if not (W / "lengths.txt").is_file():
    run([f"{MR}/tckstats", tck, "-dump", W / "lengths.txt", "-force", "-quiet"])

nodes = W / "aal3_nodes166_dwi1mm.nii.gz"
if not nodes.is_file():
    def atlas():
        run([f"{MR}/mrconvert", dti / "fa.mif", W / "fa.nii.gz", "-force", "-quiet"])
        run([f"{ANTS}/antsRegistrationSyNQuick.sh", "-d", "3", "-f", W / "fa.nii.gz", "-m", FA_T, "-o", W / "fmrib2fa_", "-t", "s", "-n", a.threads])
        run([f"{MR}/mrgrid", W / "fa.nii.gz", "regrid", "-voxel", "1", W / "ref1mm.nii.gz", "-force", "-quiet"])
        run([f"{ANTS}/antsApplyTransforms", "-d", "3", "-i", ATLAS, "-r", W / "ref1mm.nii.gz", "-o", W / "aal3_dwi1mm.nii.gz",
             "-n", "NearestNeighbor", "-t", W / "fmrib2fa_1Warp.nii.gz", "-t", W / "fmrib2fa_0GenericAffine.mat"])
    step("atlas", atlas)
    import nibabel as nib
    img = nib.load(W / "aal3_dwi1mm.nii.gz"); lab = np.rint(np.asarray(img.dataobj)).astype(int)
    mp = pd.read_csv(MAP); lut = np.zeros(max(400, lab.max() + 1), int); lut[mp.atlas_value.astype(int)] = mp.matrix_idx.astype(int)
    o = nib.Nifti1Image(lut[lab].astype(np.int32), img.affine); o.set_data_dtype(np.int32); nib.save(o, nodes)
import nibabel as nib
fa_img = np.asarray(nib.load(W / "fa.nii.gz").dataobj, float); warped = np.asarray(nib.load(W / "fmrib2fa_Warped.nii.gz").dataobj, float)
mm = (fa_img > 0) & (warped > 0); qc["reg_fa_corr"] = float(np.corrcoef(fa_img[mm], warped[mm])[0, 1])
nd = np.rint(np.asarray(nib.load(nodes).dataobj)).astype(int); vol = np.bincount(nd.ravel(), minlength=167)[1:167].astype(float)
qc.update(nodes_present=int((vol > 0).sum()), min_node_vox=int(vol.min()))

asg_f = W / "assignments.txt"
step("assign", lambda: run([f"{MR}/tck2connectome", tck, nodes, W / "count_mrtrix.csv", "-assignment_radial_search", a.radius,
                            "-out_assignments", asg_f, "-symmetric", "-zero_diagonal", "-nthreads", a.threads, "-force", "-quiet"]))


def vec(path):
    with open(path) as fh:
        return np.array(" ".join(l for l in fh if not l.startswith("#")).split(), dtype=float)


asg = pd.read_csv(asg_f, comment="#", sep=r"\s+", header=None, engine="c").values.astype(int)
L, w = vec(W / "lengths.txt"), vec(wts); n = len(asg); assert len(L) == n and len(w) == n, (len(L), len(w), n)
i, j = asg[:, 0], asg[:, 1]; ok = (i > 0) & (j > 0) & (i != j); i, j = i[ok] - 1, j[ok] - 1
qc.update(streamlines=int(n), assigned_both=float(ok.mean()))


def edge_sum(v):
    S = np.zeros((166, 166)); np.add.at(S, (i, j), v); return S + S.T


cnt = edge_sum(np.ones(ok.sum()))
mats = {"count": cnt, "fd_sum": edge_sum(w[ok]), "count_invnodevol": edge_sum(2.0 / (vol[i] + vol[j])),
        "len_mean": np.divide(edge_sum(L[ok]), cnt, out=np.zeros_like(cnt), where=cnt > 0),
        "invlen_mean": np.divide(edge_sum(1.0 / np.maximum(L[ok], 1e-6)), cnt, out=np.zeros_like(cnt), where=cnt > 0)}
for k in ["fa", "md", "rd", "ad"]:
    v = vec(W / f"{k}_mean.txt"); assert len(v) == n, (k, len(v), n); v = v[ok]; good = np.isfinite(v) & (v > 0)
    S = np.zeros((166, 166)); C = np.zeros((166, 166)); np.add.at(S, (i[good], j[good]), v[good]); np.add.at(C, (i[good], j[good]), 1)
    S, C = S + S.T, C + C.T; mats[f"{k}_mean"] = np.divide(S, C, out=np.zeros_like(S), where=C > 0)
iu = np.triu_indices(166, 1); pres = cnt > 0
qc.update(density=float(pres[iu].mean()), zero_nodes=int((pres.sum(1) == 0).sum()),
          md_coverage=float(((mats["md_mean"] > 0) & pres)[iu].sum() / max(pres[iu].sum(), 1)), seconds=round(time.time() - t0))
for k, mtx in mats.items():
    np.savetxt(M / f"SC_AAL166_{unit}_{k}.csv", mtx, delimiter=",", fmt="%.10g")
# second parcellation (DKT, 76 nodes) from the same streamlines; never blocks the AAL3 result
MD_ = OUT / "matrices_dkt"; MD_.mkdir(exist_ok=True)
if not (MD_ / f"SC_DKT76_{unit}_qc.json").is_file():
    subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("rb_dk.py")), unit, str(W), str(MD_), "--threads", str(a.threads)],
                   env=env, stdout=log, stderr=subprocess.STDOUT)
json.dump(qc, open(M / f"SC_AAL166_{unit}_qc.json", "w"), indent=1)
if not a.keep:
    big = [W / "dwi_unbiased.mif", W / "dwi_withgrad.mif", W / "dt.mif", W / "wmfod.mif"] if a.from_dwi else []   # regenerable in ~1 min from eddy output
    for f in [tck, wts, asg_f, W / "lengths.txt", *[W / f"{k}_mean.txt" for k in ["fa", "md", "ad", "rd"]], *big]:
        f.unlink(missing_ok=True)
LOCK.unlink(missing_ok=True)
print(json.dumps(qc))
