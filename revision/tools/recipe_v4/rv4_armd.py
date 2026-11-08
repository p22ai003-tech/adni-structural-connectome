"""Recipe v4 bench, arm D: Synb0's field (arm C's topup) against the measured field map, for condition 5 of bench/RULE.md.

How the comparison is made was fixed in bench/ARM_D.md before any field was compared. No diagnosis is read: the inputs are lists/bench_units.csv,
lists/fieldmap_bench_ids.txt, the field-map DICOMs under /data/adni_downloads and bench/work/C/<unit>/{topup_field_hz,b0brain}.nii.gz.

Output: bench/work/D/<unit>/, bench/qc/D_<unit>.json, bench/armD.json (median_r, n, per session).
Usage: rv4_armd.py [--units U1,U2] [--threads 4]
"""
import argparse, glob, hashlib, json, os, re, subprocess, time, traceback
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy.stats import spearmanr

ap = argparse.ArgumentParser(); ap.add_argument("--units", default=""); ap.add_argument("--threads", type=int, default=4); a = ap.parse_args()
os.nice(19)
R4 = Path("/data/derivatives/recipe_v4"); B = R4 / "bench"; L = R4 / "lists"; DL = Path("/data/adni_downloads")
FSLDIR = "/home/ec2-user/fsl"; FB = f"{FSLDIR}/bin"; ANTS = "/home/ec2-user/exp/.envs/ants-2.6.5/bin"; DCM = "/home/ec2-user/exp/.envs/dcm2niix-20260724/bin/dcm2niix"
ENV = dict(os.environ, FSLDIR=FSLDIR, FSLOUTPUTTYPE="NIFTI_GZ", ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(a.threads), OMP_NUM_THREADS=str(a.threads))


def run(cmd, W, **kw):
    cmd = [str(c) for c in cmd]
    with open(W / "run.log", "a") as log:
        log.write("$ " + " ".join(cmd) + "\n"); log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, env=dict(ENV, TMPDIR=str(W / "tmp")), **kw)


def canon(p):
    im = nib.as_closest_canonical(nib.load(str(p))); return np.asanyarray(im.dataobj).astype(np.float64), im.affine


def sidecars(d):
    """dcm2niix output in d: one row per image with its type and echo time from the sidecar."""
    rows = []
    for j in sorted(d.glob("*.json")):
        s = json.load(open(j)); it = [x.upper() for x in s.get("ImageType", [])]
        rows.append(dict(nii=j.with_suffix(".nii.gz"), te=float(s["EchoTime"]), phase=("PHASE" in it or "P" in it[2:3]), maker=s.get("Manufacturer", "")))
    return pd.DataFrame(rows)


def rigid_size(mat, W):
    """Translation (mm) and rotation (degrees) of an ITK rigid transform file."""
    txt = W / "fm2b0.txt"; run([f"{ANTS}/ConvertTransformFile", 3, mat, txt], W)
    p = [float(x) for x in re.search(r"\nParameters: (.*)", txt.read_text()).group(1).split()]
    if len(p) != 12: raise RuntimeError(f"expected a 12-parameter affine, got {len(p)}")
    Rm = np.array(p[:9]).reshape(3, 3); t = np.array(p[9:12])
    return float(np.linalg.norm(t)), float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))))


def one(unit, sid, subj, vendor, fm_dirs):
    W = B / "work/D" / unit; (W / "tmp").mkdir(parents=True, exist_ok=True); (W / "run.log").write_text("")
    C = B / "work/C" / unit; info = dict(unit=unit, session=sid, vendor=vendor, series=[p.name for p in fm_dirs])
    dwi = glob.glob(f"/data/Images/*/{subj}/*/*/I{unit.split('_I')[-1]}")
    info["same_visit"] = bool(dwi) and all(p.parent.name[:10] == Path(dwi[0]).parent.name[:10] for p in fm_dirs)
    if not info["same_visit"]: raise RuntimeError(f"field map is not from the diffusion visit: {[p.parent.name for p in fm_dirs]} vs {dwi}")
    if json.load(open(C / "eddy_in.json"))["pads"]: raise RuntimeError("arm C padded this grid; ARM_D.md assumes the topup grid is the b0brain grid")
    X = W / "dcm"; X.mkdir(exist_ok=True)
    for f in X.iterdir(): f.unlink()                     # a re-run must not find the last run's images (dcm2niix adds a suffix instead of overwriting)
    for p in fm_dirs: run([DCM, "-z", "y", "-b", "y", "-f", f"{p.name}_%s_%e", "-o", X, p], W)
    S = sidecars(X); mag = S[~S.phase].sort_values("te"); ph = S[S.phase].sort_values("te")
    te1 = mag.te.iloc[0]; info.update(n_magnitude=len(mag), n_phase=len(ph))
    run([f"{FB}/bet", mag.nii.iloc[0], W / "mag_bet", "-R", "-f", 0.5, "-m"], W)
    run([f"{FB}/fslmaths", W / "mag_bet_mask", "-kernel", "box", 3, "-ero", W / "mag_mask"], W)
    run([f"{FB}/fslmaths", mag.nii.iloc[0], "-mas", W / "mag_mask", W / "mag_brain"], W)
    if vendor == "SIEMENS":
        if len(ph) != 1: raise RuntimeError(f"expected one phase-difference image, found {len(ph)}")
        dte = ph.te.iloc[0] - te1; phase_in = ph.nii.iloc[0]
    else:
        if len(ph) != 2: raise RuntimeError(f"expected two phase images, found {len(ph)}")
        p1, a1 = canon_raw(ph.nii.iloc[0]); p2, _ = canon_raw(ph.nii.iloc[1])
        for p in (p1, p2):
            if p.min() < -3.2 or p.max() > 3.2: raise RuntimeError("Philips phase is not in radians")
        d = np.angle(np.exp(1j * (p2 - p1))); dte = ph.te.iloc[1] - ph.te.iloc[0]
        ref = nib.load(str(mag.nii.iloc[0])); hdr = ref.header.copy(); hdr.set_data_dtype(np.float32); hdr.set_slope_inter(1.0, 0.0)   # magnitude's geometry exactly: fsl_prepare_fieldmap compares voxel sizes as text
        if not np.allclose(a1, ref.affine, atol=1e-4): raise RuntimeError("Philips phase and magnitude are not on one grid")
        phase_in = W / "phasediff_siemens_scale.nii.gz"; nib.save(nib.Nifti1Image(((d / np.pi + 1) * 2048).astype(np.float32), ref.affine, hdr), str(phase_in))
    info["delta_te_ms"] = round(dte * 1000, 4)
    run([f"{FB}/fsl_prepare_fieldmap", "SIEMENS", phase_in, W / "mag_brain", W / "fmap_rads", f"{dte * 1000:.4f}"], W)
    run([f"{FB}/fslmaths", W / "fmap_rads", "-div", 2 * np.pi, W / "fmap_hz"], W)
    run([f"{FB}/fslmaths", W / "mag_brain", "-bin", W / "fmap_mask"], W)
    run([f"{ANTS}/antsRegistration", "-d", 3, "--float", 1, "-o", f"[{W / 'fm2b0_'}]", "-n", "Linear", "-u", 0, "-w", "[0.005,0.995]", "-t", "Rigid[0.1]",
         "-m", f"MI[{C / 'b0brain.nii.gz'},{W / 'mag_brain.nii.gz'},1,32,Regular,0.25]", "-c", "[1000x500x250,1e-6,10]", "-f", "4x2x1", "-s", "2x1x0vox"], W)
    mat = W / "fm2b0_0GenericAffine.mat"; info["rigid_translation_mm"], info["rigid_rotation_deg"] = rigid_size(mat, W)
    ref = C / "topup_field_hz.nii.gz"
    for tag, tr in (("reg", ["-t", mat]), ("world", ["-t", "identity"])):
        run([f"{ANTS}/antsApplyTransforms", "-d", 3, "-i", W / "fmap_hz.nii.gz", "-r", ref, *tr, "-n", "Linear", "-o", W / f"fmap_hz_{tag}.nii.gz"], W)
        run([f"{ANTS}/antsApplyTransforms", "-d", 3, "-i", W / "fmap_mask.nii.gz", "-r", ref, *tr, "-n", "Linear", "-o", W / f"fmap_mask_{tag}.nii.gz"], W)
    tu, af = canon(ref); br, af2 = canon(C / "b0brain.nii.gz")
    if tu.shape != br.shape or not np.allclose(af, af2, atol=1e-3): raise RuntimeError("topup field and b0brain are not on one grid")
    acq = json.load(open(B / "work/_shared" / unit / "acq.json")); ax = int(np.argmax(np.abs(json.load(open(C / "eddy_in.json"))["pe_vector"]))); vox = acq["vox"][ax]
    for tag in ("reg", "world"):
        fm, af3 = canon(W / f"fmap_hz_{tag}.nii.gz"); mk, _ = canon(W / f"fmap_mask_{tag}.nii.gz")
        if not np.allclose(af, af3, atol=1e-3): raise RuntimeError("resampled field map is off the topup grid")
        M = (br > 0) & (mk >= 0.999); x, y = tu[M], fm[M]; r = float(np.corrcoef(x, y)[0, 1])
        info[tag] = dict(r=r, spearman=float(spearmanr(x, y)[0]), voxels=int(M.sum()), ml=float(M.sum() * np.prod(np.abs(np.diag(af)[:3])) / 1000),
                         synb0_hz_p5_p95=[float(np.percentile(x, 5)), float(np.percentile(x, 95))], measured_hz_p5_p95=[float(np.percentile(y, 5)), float(np.percentile(y, 95))],
                         synb0_disp_mm_p95=float(np.percentile(np.abs(x - np.median(x)), 95) * acq["readout"] * vox), measured_disp_mm_p95=float(np.percentile(np.abs(y - np.median(y)), 95) * acq["readout"] * vox))
    info["r_signed"] = info["reg"]["r"]; info["r_rule"] = info["r_signed"] if vendor == "SIEMENS" else abs(info["r_signed"])
    info["r_rule_basis"] = "signed (polarity in the sidecar)" if vendor == "SIEMENS" else "absolute (polarity not stored; arm C used +)"
    return info


def canon_raw(p):
    im = nib.load(str(p)); return np.asanyarray(im.dataobj).astype(np.float64), im.affine


U = pd.read_csv(L / "bench_units.csv"); U["sid"] = [f"S{i + 1:02d}" for i in range(len(U))]
fm_ids = [x for x in open(L / "fieldmap_bench_ids.txt").read().strip().split(",") if x]
disk = {p.name[1:]: p for p in DL.rglob("I*") if p.name[1:] in set(fm_ids) and p.is_dir()}
want = [x for x in a.units.split(",") if x]
res = {}
for _, u in U.iterrows():
    dirs = sorted(p for i, p in disk.items() if f"/{u.subj}/" in str(p))
    if not dirs or (want and u.unit not in want): continue
    t0 = time.time()
    try:
        info = one(u.unit, u.sid, u.subj, u.vendor, dirs); info["status"] = "ok"
    except Exception as e:
        info = dict(unit=u.unit, session=u.sid, vendor=u.vendor, status="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
    info["sec"] = round(time.time() - t0, 1); info["armtag"] = "D"; res[u.sid] = info
    json.dump(info, open(B / "qc" / f"D_{u.unit}.json", "w"), indent=1)
    print(u.sid, u.vendor, info["status"], round(info.get("r_signed", float("nan")), 3), info.get("error", ""), flush=True)

if not want:
    ok = [v for v in res.values() if v["status"] == "ok"]; med = lambda k, vs: float(np.median([v[k] for v in vs])) if vs else None
    out = dict(median_r=med("r_rule", ok), n=len(ok), sessions_with_field_map=len(res), failed=[k for k, v in res.items() if v["status"] != "ok"],
               siemens_median=med("r_rule", [v for v in ok if v["vendor"] == "SIEMENS"]), philips_median=med("r_rule", [v for v in ok if v["vendor"] == "PHILIPS"]),
               median_r_world=float(np.median([v["world"]["r"] if v["vendor"] == "SIEMENS" else abs(v["world"]["r"]) for v in ok])) if ok else None,
               per_session={k: dict(vendor=v["vendor"], r_rule=v.get("r_rule"), r_signed=v.get("r_signed"), status=v["status"]) for k, v in res.items()},
               method="bench/ARM_D.md", method_sha256=hashlib.sha256((B / "ARM_D.md").read_bytes()).hexdigest()[:16], written=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()))
    json.dump(out, open(B / "armD.json", "w"), indent=1); print(json.dumps({k: out[k] for k in ("median_r", "n", "siemens_median", "philips_median", "median_r_world", "failed")}))
