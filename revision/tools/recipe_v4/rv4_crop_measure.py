"""What dwidenoise and mrdegibbs do to one bench session, with and without a return to the acquired lines. Tensor only, no eddy. Sessions print as S01..; no diagnosis is read.
Usage: rv4_crop_measure.py S16 [--out /data/derivatives/recipe_v4/crop_test/measure] [--threads 4]"""
import argparse, glob, json, os, shutil, subprocess
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, nibabel as nib
ap = argparse.ArgumentParser(); ap.add_argument("sid"); ap.add_argument("--out", default="/data/derivatives/recipe_v4/crop_test/measure"); ap.add_argument("--threads", default="4"); a = ap.parse_args()
os.nice(19)
MR = "/home/ec2-user/mrtrix3/bin"; DCM = "/home/ec2-user/exp/.envs/dcm2niix-20260724/bin/dcm2niix"; R = f"{a.out}/{a.sid}"; os.makedirs(R, exist_ok=True); J_OUT = f"{a.out}/{a.sid}.json"
env = dict(os.environ, MRTRIX_TMPFILE_DIR=R + "/", MRTRIX_NTHREADS=a.threads, OMP_NUM_THREADS=a.threads)
U = pd.read_csv("/data/derivatives/recipe_v4/lists/bench_units.csv", dtype=str); U["sid"] = [f"S{k + 1:02d}" for k in range(len(U))]; r = U[U.sid == a.sid].iloc[0]


def run(*c): subprocess.run(list(map(str, c)), check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def kspace_crop(d, nx, ny):
    Nx, Ny = d.shape[:2]; ax, ay = Nx // 2 - nx // 2, Ny // 2 - ny // 2; out = np.zeros((nx, ny) + d.shape[2:], np.float32); scale = nx * ny / (Nx * Ny); xy = (slice(None), slice(None))
    for k in np.ndindex(d.shape[2:]):
        F = np.fft.fftshift(np.fft.fft2(d[xy + k])); out[xy + k] = np.abs(np.fft.ifft2(np.fft.ifftshift(F[ax:ax + nx, ay:ay + ny]))) * scale
    return out


def nii(mif, tag):
    p = f"{R}/_{tag}.nii"; run(f"{MR}/mrconvert", mif, p, "-force", "-quiet"); d = np.asanyarray(nib.load(p).dataobj).astype(np.float32); os.unlink(p); return d


def fa(mif, tag):
    run(f"{MR}/dwi2tensor", mif, f"{R}/_{tag}_dt.mif", "-force", "-quiet", "-nthreads", a.threads); run(f"{MR}/tensor2metric", f"{R}/_{tag}_dt.mif", "-fa", f"{R}/_{tag}_fa.nii", "-force", "-quiet", "-nthreads", a.threads)
    v = np.asanyarray(nib.load(f"{R}/_{tag}_fa.nii").dataobj).astype(np.float64); os.unlink(f"{R}/_{tag}_dt.mif"); os.unlink(f"{R}/_{tag}_fa.nii"); return np.nan_to_num(v)


def acf(res, mask):
    o = []
    for ax in (0, 1, 2):
        x_, m_ = np.moveaxis(res, ax, 0), np.moveaxis(mask, ax, 0); both = m_[:-1] & m_[1:]; x, y = x_[:-1][both], x_[1:][both]
        o.append(round(float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())), 3))
    return o


def measure(tag, raw):
    den, deg, noise = f"{R}/{tag}_den.mif", f"{R}/{tag}_deg.mif", f"{R}/{tag}_noise.nii"
    run(f"{MR}/dwidenoise", raw, den, "-noise", noise, "-force", "-quiet", "-nthreads", a.threads); run(f"{MR}/mrdegibbs", den, deg, "-force", "-quiet", "-nthreads", a.threads)
    run(f"{MR}/dwi2mask", deg, f"{R}/{tag}_mask.nii", "-force", "-quiet"); mask = np.asanyarray(nib.load(f"{R}/{tag}_mask.nii").dataobj) > 0
    x, dn, dg = nii(raw, tag + "r"), nii(den, tag + "n"), nii(deg, tag + "g"); sig = np.asanyarray(nib.load(noise).dataobj).astype(float); res = x - dn; m4 = np.broadcast_to(mask[..., None], res.shape); b0 = float(np.median(x[..., 0][mask]))
    rms = float(np.sqrt(np.mean(res[m4] ** 2))); fr, fn, fg = fa(raw, tag + "r"), fa(den, tag + "n"), fa(deg, tag + "g"); wm = mask & (fg > 0.25)
    out = dict(shape=list(map(int, x.shape)), mask_ml=float(mask.sum() * np.prod(nib.load(noise).header.get_zooms()[:3]) / 1000), pct_b0_removed=100 * rms / b0, residual_to_sigma=rms / float(np.median(sig[mask])), acf_residual_xyz=acf(res, m4),
               degibbs_pct_b0=100 * float(np.sqrt(np.mean((dg - dn)[m4] ** 2))) / b0, fa_raw=float(fr[wm].mean()), fa_den=float(fn[wm].mean()), fa_deg=float(fg[wm].mean()), dfa_denoise=float((fn - fr)[wm].mean()), dfa_degibbs=float((fg - fn)[wm].mean()), dfa_both=float((fg - fr)[wm].mean()))
    for f in (den, deg, noise, f"{R}/{tag}_mask.nii"): os.unlink(f)
    return out


ser = glob.glob(f"/data/Images/dti/{r.subj}/*/*/I{r.unit.split('_I')[-1]}")[0]; run(DCM, "-z", "n", "-b", "y", "-f", "raw", "-o", R, ser)
run(f"{MR}/mrconvert", f"{R}/raw.nii", f"{R}/raw.mif", "-fslgrad", f"{R}/raw.bvec", f"{R}/raw.bval", "-json_import", f"{R}/raw.json", "-strides", "-1,2,3,4", "-force", "-quiet")
J = json.load(open(f"{R}/raw.json")); res = dict(sid=a.sid, stratum=f"{r.phase} {r.vendor}", vendor=r.vendor, acq=J.get("AcquisitionMatrixPE"), recon=J.get("ReconMatrixPE"), pct=J.get("PercentPhaseFOV"))
res["asis"] = measure("asis", f"{R}/raw.mif")
if res["recon"] and res["acq"] and res["recon"] > res["acq"] * 1.1:
    im = nib.load(f"{R}/raw.nii"); d = np.asanyarray(im.dataobj).astype(np.float32); n0 = int(res["acq"]); pct = float(res["pct"] or 100); ny = 2 * round(n0 * 100 / pct / 2)
    for tag, (nx_, ny_) in dict(crop=(n0, ny), **({"crop_square": (n0, n0)} if ny != n0 else {})).items():
        af = im.affine.copy(); af[:3, 0] *= d.shape[0] / nx_; af[:3, 1] *= d.shape[1] / ny_; o = nib.Nifti1Image(kspace_crop(d, nx_, ny_), af); o.set_qform(af, code=1); o.set_sform(af, code=1); nib.save(o, f"{R}/{tag}.nii")
        run(f"{MR}/mrconvert", f"{R}/{tag}.nii", f"{R}/{tag}.mif", "-fslgrad", f"{R}/raw.bvec", f"{R}/raw.bval", "-strides", "-1,2,3,4", "-force", "-quiet"); os.unlink(f"{R}/{tag}.nii")
        res[tag] = measure(tag, f"{R}/{tag}.mif"); res[tag]["matrix"] = [nx_, ny_]; os.unlink(f"{R}/{tag}.mif")
json.dump(res, open(J_OUT + ".tmp", "w"), indent=1); os.replace(J_OUT + ".tmp", J_OUT)
for f in os.listdir(R): os.unlink(f"{R}/{f}")
os.rmdir(R); print("ok", a.sid)
