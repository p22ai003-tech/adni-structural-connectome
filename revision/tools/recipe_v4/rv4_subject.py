"""Recipe v4 (one frozen from-raw ACT recipe), one session, one arm. Rule: /data/derivatives/recipe_v4/bench/RULE.md.

No diagnosis, group, CDR or MMSE is read anywhere: the only lists are lists/t1_choice.csv and the DICOM sidecars.

  Stages (each resumable; a .<stage>.done.json marker holds seconds, peak disk and the stage version)
  s1_inputs      raw DWI from DICOM (dcm2niix, so the sidecar belongs to these voxels), gradients, phase-encoding axis and readout
                 time from the scan's own sidecar, cross-check against /data/derivatives/mif_dwi
  s2_import      arm A: stored eddy output (reference)
  s2a_denoise    arms B, C: dwidenoise, mrdegibbs (shared by the arms). With --crop-zf a scan the scanner interpolated goes back on its acquired in-plane lines first
  s2_eddyin      arms B, C: FSL-layout image, gradients, acqp and index from the sidecar (arm C pads odd axes for topup)
  s2b_synb0      arm C: synthetic undistorted b0, native port on CPU (~/synb0disco: prep, 5 folds, inverse)
  s2c_topup      arm C: topup on distorted + synthetic b0 (true PE axis, readout time, then 0 for the synthetic image)
  s2d_mask       arms B, C: eddy mask (dwi2mask + dilate; arm C after applytopup, as dwifslpreproc does)
  s2e_eddy       arms B, C: eddy_cpu --repol --slm=linear --data_is_shelled, arm C with --topup; movement and outlier files kept
  s3_model       b=1000 shell if multi-shell, N4, mask, dhollander WM response, CSD lmax 8, tensor maps, 1 mm grid
  s4a_fastsurfer same-visit T1: stored FastSurfer only if its log shows that T1, else CPU --seg_only (shared by the arms)
  s4b_align      T1 to b0 by ANTs rigid and by BBR (epi_reg); the one with the higher WM-to-FA>0.25 Dice is kept
  s4c_5tt        5ttgen freesurfer -sgm_amyg_hipp (MRtrix's own table); test on the real image that ventral DC is WM
  s5a_syn        MNI to T1 SyN (shared by the arms)
  s5b_atlas      AAL3 through the T1 with the tissue map's transform, 166 labels kept, 12 brainstem nuclei flagged not estimable
  s6a_tckgen     iFOD2 ACT -backtrack -crop_at_gmwmi, dynamic seeding, cutoff 0.06, angle 45, 10-250 mm, step 1 mm
  s6b_sift2      tcksift2 -act, mu saved
  s6c_sample     per-streamline mean FA/MD/AD/RD, lengths, endpoints
  s6d_assign     tck2connectome on the tractogram and on the tractogram rebuilt from the saved endpoints; must match; tractogram deleted
  s7_qc          bench/qc/<arm>_<unit>.json

Output: bench/work/<arm>/<unit>/ and bench/qc/<arm>_<unit>.json; what several arms share in bench/work/_shared/<unit>/.
--tag makes a variant of an existing arm run (10M, second seed): only s6 and s7 run, from the base arm's FOD, tissue map and atlas.

Usage: rv4_subject.py <unit> --arm B [--select 1000000] [--seed 1] [--tag seed2] [--threads 8] [--out DIR] [--crop-zf]
       rv4_subject.py --selftest
"""
import argparse, fcntl, glob, hashlib, json, os, re, resource, subprocess, sys, threading, time, traceback
from pathlib import Path
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"   # numpy in this process only; children get --threads
import numpy as np, pandas as pd, nibabel as nib

ap = argparse.ArgumentParser(); ap.add_argument("unit", nargs="?"); ap.add_argument("--arm", default="B", choices=list("ABCD"))
ap.add_argument("--select", type=int, default=1_000_000); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--tag", default="")
ap.add_argument("--threads", type=int, default=8); ap.add_argument("--out", default="/data/derivatives/recipe_v4/bench")
ap.add_argument("--scratch", default="/data/derivatives/recipe_v4/scratch"); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--crop-zf", action="store_true")
a = ap.parse_args()
os.nice(19)
DER = Path("/data/derivatives"); MR = "/home/ec2-user/mrtrix3/bin"; ANTS = "/home/ec2-user/exp/.envs/ants-2.6.5/bin"
FSLDIR = "/home/ec2-user/fsl"; FB = f"{FSLDIR}/share/fsl/bin"; DCM = "/home/ec2-user/exp/.envs/dcm2niix-20260724/bin/dcm2niix"
PY = "/home/ec2-user/exp/.venv_connectome_app/bin/python"
SYN = "/home/ec2-user/synb0disco"; FS_SH = "/home/ec2-user/FastSurfer/run_fastsurfer.sh"; FS_PY = "/home/ec2-user/miniconda3/envs/fastsurfer/bin/python"
FS_LIC = "/home/ec2-user/freesurfer/license.txt"; FS_LUT = "/home/ec2-user/FastSurfer/FastSurferCNN/config/FreeSurferColorLUT.txt"
ATLAS = "/home/ec2-user/exp/atlas/AAL/AAL3v1_1mm.nii.gz"; MAP = "/home/ec2-user/exp/atlas/AAL/AAL3_network_mapping.csv"
MNI = f"{FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz"; LISTS = Path("/data/derivatives/recipe_v4/lists")
VERSIONS = dict(s1_inputs=1, s2_import=1, s2a_denoise=1, s2_eddyin=1, s2b_synb0=1, s2c_topup=1, s2d_mask=1, s2e_eddy=1, s3_model=1, s4a_fastsurfer=3, s4b_align=3,
                s4c_5tt=1, s5a_syn=2, s5b_atlas=1, s6a_tckgen=1, s6b_sift2=1, s6c_sample=1, s6d_assign=1, s7_qc=1)
ROI = {"orbitofrontal": ["OFCmed", "OFCant", "OFCpost", "OFClat", "Rectus", "Frontal_Med_Orb", "Frontal_Inf_Orb_2"],
       "temporal_pole": ["Temporal_Pole_Sup", "Temporal_Pole_Mid"], "medial_temporal": ["Hippocampus", "ParaHippocampal", "Amygdala"]}
WM_DICE_LABELS = [2, 41]                                 # the label set of reg_wm_dice (rb_dk.py), on which the .65 bar was calibrated
TT_EXPECT = {"cGM": ([3, 42, 8, 47], 0), "sGM": ([10, 49, 11, 50, 12, 51, 13, 52, 26, 58, 17, 53, 18, 54], 1), "WM": ([2, 41, 7, 46, 16, 28, 60], 2),
             "CSF": ([4, 43, 5, 44, 14, 15, 24, 31, 63], 3), "pathology": ([77], 4)}


def dice(x, y):
    return float(2 * (x & y).sum() / max(int(x.sum()) + int(y.sum()), 1))


def canon(path):
    """Voxels and affine in RAS+ voxel order, whatever the on-disk layout, so images on one grid compare voxel by voxel."""
    im = nib.as_closest_canonical(nib.load(str(path))); return np.asanyarray(im.dataobj), im.affine


def tck_read(path):
    """Float32LE .tck to (n_points_total, 3) with NaN separators; the inf terminator removed."""
    with open(path, "rb") as fh:
        hdr = b""
        while not hdr.endswith(b"END\n"):
            line = fh.readline()
            if not line: raise RuntimeError(f"no END in header of {path}")
            hdr += line
    if b"Float32LE" not in hdr: raise RuntimeError("tck datatype is not Float32LE")
    off = int(re.search(rb"file:\s*\.\s+(\d+)", hdr).group(1))
    x = np.fromfile(path, dtype="<f4", offset=off).reshape(-1, 3)
    if len(x) and np.isinf(x[-1]).all(): x = x[:-1]
    return x, hdr


def endpoints_from_tck(path):
    """tckresample -endpoints writes point, point, NaN for each streamline."""
    x, _ = tck_read(path); x = x.reshape(-1, 3, 3)
    if not np.isnan(x[:, 2]).all() or np.isnan(x[:, :2]).any(): raise RuntimeError("endpoint file is not two vertices per streamline")
    return x[:, :2].astype(np.float32).copy()


def endpoints_to_tck(ends, path):
    """Rebuild a two-vertex tractogram from the saved float32 endpoints."""
    n = len(ends); body = np.full((n, 3, 3), np.nan, dtype="<f4"); body[:, :2] = ends
    head = f"mrtrix tracks\ncount: {n}\ndatatype: Float32LE\nfile: . "
    off = len(head) + 16; head += f"{off}\nEND\n"; head = head + " " * (off - len(head))      # END, then padding up to the data offset
    with open(path, "wb") as fh:
        fh.write(head.encode()); body.tofile(fh); np.full(3, np.inf, dtype="<f4").tofile(fh)


def read_vec(path, dtype=np.float32):
    """Numbers from a text file written by MRtrix; comment lines dropped."""
    with open(path) as fh: txt = fh.read()
    if "#" in txt: txt = re.sub(r"(?m)^#.*$", "", txt)
    return np.fromstring(txt, dtype=np.float64, sep=" ").astype(dtype)


def pe_vector(axis, affine_raw, affine_in):
    """Sidecar axis ('j', 'j-', 'i', ...) is defined on the voxels dcm2niix wrote. Returns the FSL acqp vector for the image eddy reads."""
    m = re.fullmatch(r"([ijk])(-?)", axis)
    if not m: raise RuntimeError(f"unexpected phase-encoding axis {axis}")
    v = np.zeros(3); v["ijk".index(m.group(1))] = -1.0 if m.group(2) else 1.0
    A0, A1 = np.asarray(affine_raw)[:3, :3], np.asarray(affine_in)[:3, :3]
    w = A0 @ v; w /= np.linalg.norm(w); out = np.linalg.solve(A1 / np.linalg.norm(A1, axis=0), w)
    if np.linalg.det(A1) > 0: out[0] *= -1                                                     # FSL's x runs against a positive determinant
    r = np.rint(out).astype(int)
    if np.abs(r).sum() != 1 or np.abs(out - r).max() > 0.01: raise RuntimeError(f"phase-encoding axis does not map onto a voxel axis: {out}")
    return r.tolist()


def fsl_vox2mm(im):
    """FSL's scaled-voxel coordinates: x runs the other way if the image has a positive determinant."""
    S = np.diag(list(im.header.get_zooms()[:3]) + [1.0])
    if np.linalg.det(im.affine[:3, :3]) > 0:
        F = np.eye(4); F[0, 0] = -1; F[0, 3] = im.shape[0] - 1; return S @ F
    return S


def flirt_to_itk(mat, src, ref, outp):
    """FLIRT matrix (src to ref) as an ITK file for ANTs. World is RAS in NIfTI, LPS in ITK, and ITK stores the map from fixed (ref) points to moving (src) points."""
    s, r = nib.load(str(src)), nib.load(str(ref))
    T = r.affine @ np.linalg.inv(fsl_vox2mm(r)) @ np.loadtxt(mat) @ fsl_vox2mm(s) @ np.linalg.inv(s.affine)
    L = np.diag([-1.0, -1, 1, 1]); I = L @ np.linalg.inv(T) @ L
    p = " ".join(f"{x:.12g}" for x in I[:3, :3].ravel()) + " " + " ".join(f"{x:.12g}" for x in I[:3, 3])
    open(outp, "w").write(f"#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_double_3_3\nParameters: {p}\nFixedParameters: 0 0 0\n")


def kspace_crop(d, nx, ny):
    """Keep nx by ny lines of the in-plane Fourier transform of each slice of d (x, y, ...). Voxel l of the result sits at voxel l * N / n of the input, and intensities keep their scale."""
    Nx, Ny = d.shape[:2]; ax, ay = Nx // 2 - nx // 2, Ny // 2 - ny // 2; out = np.zeros((nx, ny) + d.shape[2:], np.float32); scale = nx * ny / (Nx * Ny); xy = (slice(None), slice(None))
    for k in np.ndindex(d.shape[2:]):
        F = np.fft.fftshift(np.fft.fft2(d[xy + k])); out[xy + k] = np.abs(np.fft.ifft2(np.fft.ifftshift(F[ax:ax + nx, ay:ay + ny]))) * scale
    return out


def selftest():
    """No participant data. (1) MRtrix's table on a synthetic label image: ventral DC is WM, hippocampus and amygdala sGM, brain stem WM. (2) endpoint round trip. (3) PE vectors. (4) k-space crop keeps position and scale."""
    S = Path(a.scratch) / f"selftest_{os.getpid()}"; S.mkdir(parents=True); env = dict(os.environ, PATH=f"{MR}:" + os.environ["PATH"], MRTRIX_TMPFILE_DIR=str(S) + "/")
    labels = [l for v in TT_EXPECT.values() for l in v[0]] + [1002, 2002]; L = np.zeros((len(labels) + 4, 6, 6), np.int32)
    for k, l in enumerate(labels): L[k + 2, 2:4, 2:4] = l
    nib.save(nib.Nifti1Image(L, np.eye(4)), S / "aseg_like.nii.gz")
    subprocess.run([f"{MR}/5ttgen", "freesurfer", S / "aseg_like.nii.gz", S / "tt.nii.gz", "-lut", FS_LUT, "-sgm_amyg_hipp", "-nocrop", "-scratch", S, "-force", "-quiet", "-nthreads", "1"], check=True, env=env)
    tt, _ = canon(S / "tt.nii.gz"); res = {}; ok = True
    for name, (ls, ch) in TT_EXPECT.items():
        for l in ls:
            got = int(np.argmax(tt[:, 2:4, 2:4, :].reshape(-1, 5)[(L[:, 2:4, 2:4].ravel() == l)].mean(0))); res[str(l)] = ["cGM", "sGM", "WM", "CSF", "pathology"][got]; ok &= got == ch
    for l in (1002, 2002):
        got = int(np.argmax(tt[:, 2:4, 2:4, :].reshape(-1, 5)[(L[:, 2:4, 2:4].ravel() == l)].mean(0))); res[str(l)] = ["cGM", "sGM", "WM", "CSF", "pathology"][got]; ok &= got == 0
    assert res["28"] == "WM" and res["60"] == "WM", "ventral DC is not WM"; assert res["16"] == "WM"
    assert all(res[str(l)] == "sGM" for l in (17, 53, 18, 54)), "hippocampus/amygdala are not sGM"
    e = np.random.default_rng(1).normal(size=(1000, 2, 3)).astype(np.float32); endpoints_to_tck(e, S / "e.tck")
    assert np.array_equal(endpoints_from_tck(S / "e.tck"), e), "endpoint round trip"
    n = subprocess.run([f"{MR}/tckinfo", S / "e.tck", "-count"], check=True, env=env, capture_output=True, text=True).stdout
    assert "1000" in n, n
    las = np.diag([-2.0, 2, 2, 1]); ras = np.diag([2.0, 2, 2, 1])
    assert pe_vector("j", las, las) == [0, 1, 0] and pe_vector("j-", las, las) == [0, -1, 0] and pe_vector("i", las, las) == [1, 0, 0]
    assert pe_vector("i", ras, las) == [-1, 0, 0], "x flip between raw and eddy image"
    gx, gy = np.mgrid[:256, :256]; blob = 1000 * np.exp(-((gx - 101.3) ** 2 + (gy - 140.7) ** 2) / (2 * 14.0 ** 2)); c = kspace_crop(blob[..., None, None].astype(np.float32), 116, 194)[..., 0, 0]; cx, cy = np.mgrid[:116, :194]
    pos = ((c * cx).sum() / c.sum() * 256 / 116, (c * cy).sum() / c.sum() * 256 / 194)
    assert abs(pos[0] - 101.3) < 0.02 and abs(pos[1] - 140.7) < 0.02, f"k-space crop moved the image: {pos}"; assert abs(c.max() / 1000 - 1) < 0.01, f"k-space crop changed the scale: {c.max()}"
    out = dict(ok=bool(ok), label_to_tissue=res, tck_roundtrip=True, pe_vectors=True, kspace_crop=True, when=time.strftime("%FT%TZ", time.gmtime()))
    Path(a.out).mkdir(parents=True, exist_ok=True); json.dump(out, open(Path(a.out) / "selftest.json", "w"), indent=1)
    for f in S.iterdir(): f.unlink()
    S.rmdir(); print(json.dumps(out)); sys.exit(0 if ok else 1)


if a.selftest: selftest()
if a.arm == "D": sys.exit("arm D needs the measured field-map series, which are not on disk; not built")
UNIT = a.unit; BASE = Path(a.out); ARMTAG = a.arm + (f"_{a.tag}" if a.tag else "")
row = pd.read_csv(LISTS / "t1_choice.csv", dtype=str).set_index("unit").loc[UNIT]; SUBJ, T1ID = row["subj"], row["t1_iid"]
W = BASE / "work" / ARMTAG / UNIT; W.mkdir(parents=True, exist_ok=True); SH = BASE / "work/_shared" / UNIT; SH.mkdir(parents=True, exist_ok=True)
W0 = BASE / "work" / a.arm / UNIT                        # where FOD, tissue map and atlas live (same as W unless this is a variant)
QC = BASE / "qc"; QC.mkdir(exist_ok=True); SCR = Path(a.scratch) / f"{ARMTAG}_{UNIT}"; SCR.mkdir(parents=True, exist_ok=True); QCF = QC / f"{ARMTAG}_{UNIT}.json"
T = str(a.threads); SEED = int(hashlib.md5(f"{UNIT}:{a.seed}".encode()).hexdigest()[:7], 16)
env = dict(os.environ, ANTSPATH=ANTS + "/", FSLDIR=FSLDIR, FSLOUTPUTTYPE="NIFTI_GZ", PATH=f"{ANTS}:{MR}:" + os.environ["PATH"], ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=T,
           OMP_NUM_THREADS=T, MKL_NUM_THREADS=T, MRTRIX_NTHREADS=T, MRTRIX_TMPFILE_DIR=str(Path(a.scratch)) + "/", TMPDIR=str(SCR), MRTRIX_RNG_SEED=str(SEED), ANTS_RANDOM_SEED="20261102", PYTHONWARNINGS="ignore")
log = open(W / "run.log", "a"); qc = dict(unit=UNIT, arm=a.arm, tag=a.tag, armtag=ARMTAG, select=a.select, seed=a.seed, rng_seed=SEED, threads=a.threads, phase=row["phase"], vendor=row["vendor"],
                                         t1_iid=T1ID, stages={}, versions=VERSIONS, status="running", crop_zf=a.crop_zf, code_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest())
dirty = False


def mr(x): return f"{MR}/{x}"


def run(cmd, cwd=None, timeout=None, extra=None):
    cmd = list(map(str, cmd)); log.write(f"[{time.strftime('%H:%M:%S')}] $ " + " ".join(cmd) + "\n"); log.flush()
    subprocess.run(cmd, check=True, env=dict(env, **extra) if extra else env, stdout=log, stderr=subprocess.STDOUT, cwd=str(cwd or W), timeout=timeout)


def out(cmd):
    return subprocess.run(list(map(str, cmd)), check=True, env=env, capture_output=True, text=True).stdout.strip()


def rm(*paths):
    for p in paths: Path(p).unlink(missing_ok=True)


def dir_bytes(*roots):
    s = 0
    for r in roots:
        for d, _, fs in os.walk(r):
            for f in fs:
                try: s += os.stat(os.path.join(d, f)).st_size
                except OSError: pass
    return s


class Peak(threading.Thread):
    """Largest size of the stage's folders while the stage runs (polled)."""
    def __init__(s, *roots):
        super().__init__(daemon=True); s.roots, s.peak, s.halt = roots, 0, threading.Event()
    def run(s):
        while not s.halt.is_set():
            s.peak = max(s.peak, dir_bytes(*s.roots)); s.halt.wait(10)
    def stop(s):
        s.halt.set(); s.join(); s.peak = max(s.peak, dir_bytes(*s.roots)); return round(s.peak / 1e9, 3)


class Lock:
    """Blocks while another arm of this session builds the same shared output."""
    def __init__(s, p): s.p = p
    def __enter__(s): s.f = open(s.p, "w"); fcntl.flock(s.f, fcntl.LOCK_EX); return s
    def __exit__(s, *e): fcntl.flock(s.f, fcntl.LOCK_UN); s.f.close()


def marker(name, shared): return (SH if shared else W) / f".{name}.done.json"


def stamps():
    """What this arm's finished stages were built from: another arm may have rebuilt a shared stage (or the base run of a variant) since."""
    fs = [SH / f".{n}.done.json" for n in ("s1_inputs", "s2a_denoise", "s4a_fastsurfer", "s5a_syn")]
    if a.tag: fs += [W0 / f".{n}.done.json" for n in ("s3_model", "s4c_5tt", "s5b_atlas")]
    return {f.name: hashlib.md5(f.read_bytes()).hexdigest()[:10] for f in fs if f.is_file()}


def done(name, shared=False, outputs=()):
    mk = marker(name, shared)
    if not mk.is_file(): return None
    m = json.load(open(mk))
    if m.get("ver") != VERSIONS[name] or not all(Path(o).exists() for o in outputs): return None
    if not shared:
        st = stamps()
        if any(st.get(k) != v for k, v in m.get("deps", {}).items()): return None
    return m


def _stage(name, fn, shared, outputs):
    global dirty
    m = done(name, shared, outputs)
    if m and (shared or not dirty):
        qc["stages"][name] = dict(m, reused=True, shared=shared); return
    dirty = True; s = time.time(); pk = Peak(SH if shared else W, SCR); pk.start(); qc["failed_stage"] = name; save_qc()
    log.write(f"[{time.strftime('%H:%M:%S')}] ==== {name}\n"); log.flush()
    try: info = fn() or {}
    finally: peak = pk.stop()
    m = dict(ver=VERSIONS[name], sec=round(time.time() - s), peak_gb=peak, rss_so_far_gb=round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 2), threads=a.threads,
             finished=time.strftime("%FT%TZ", time.gmtime()), **info)
    if not shared: m["deps"] = stamps()
    tmp = marker(name, shared).with_suffix(".tmp"); json.dump(m, open(tmp, "w"), indent=1); os.replace(tmp, marker(name, shared))
    qc["stages"][name] = dict(m, reused=False, shared=shared); qc.pop("failed_stage", None)


def stage(name, fn, outputs=()): _stage(name, fn, False, outputs)


def shared_stage(name, fn, outputs=()):
    with Lock(SH / f".{name}.lock"): _stage(name, fn, True, outputs)


def save_qc():
    tmp = QCF.with_suffix(".tmp"); json.dump(qc, open(tmp, "w"), indent=1, default=float); os.replace(tmp, QCF)


# ---------------- s1: raw data and its own sidecar
def s1_inputs():
    iid = UNIT.split("_I")[-1]; ser = glob.glob(f"/data/Images/dti/{SUBJ}/*/*/I{iid}")
    if len(ser) != 1: raise RuntimeError(f"raw series found {len(ser)} times")
    run([DCM, "-z", "n", "-b", "y", "-f", "raw", "-o", SH, ser[0]], cwd=SH)
    J = json.load(open(SH / "raw.json")); pe = J.get("PhaseEncodingDirection"); known = pe is not None; pe_src = "sidecar PhaseEncodingDirection"
    if pe is None: pe, pe_src = J.get("PhaseEncodingAxis"), "sidecar PhaseEncodingAxis (polarity not stored; + used)"
    rt = J.get("TotalReadoutTime"); rt_src = "sidecar TotalReadoutTime"
    if rt is None: rt, rt_src = J.get("EstimatedTotalReadoutTime"), "sidecar EstimatedTotalReadoutTime"
    if pe is None or rt is None: raise RuntimeError("sidecar has no phase-encoding axis or no readout time")
    im = nib.load(SH / "raw.nii")
    run([mr("mrconvert"), SH / "raw.nii", SH / "raw.mif", "-fslgrad", SH / "raw.bvec", SH / "raw.bval", "-json_import", SH / "raw.json", "-strides", "-1,2,3,4", "-force", "-quiet"], cwd=SH)
    bv = np.loadtxt(SH / "raw.bval"); old = SH / "grad_mif_dwi.b"; ck = {}
    sdw = DER / f"mif_dwi/{UNIT}.mif"
    if sdw.exists():                                     # same volumes as the stored raw image? (order may differ on Philips)
        run([mr("mrinfo"), sdw, "-export_grad_mrtrix", old, "-force"], cwd=SH); g = np.loadtxt(old, comments="#")
        ck = dict(n_vols_equal=int(len(g)) == len(bv), n_b0_equal=int((g[:, 3] < 100).sum()) == int((bv < 100).sum()), first_vol_b0_raw=bool(bv[0] < 100), first_vol_b0_mif_dwi=bool(g[0, 3] < 100))
    shells = [float(x) for x in out([mr("mrinfo"), SH / "raw.mif", "-shell_bvalues"]).split()]; sizes = [int(x) for x in out([mr("mrinfo"), SH / "raw.mif", "-shell_sizes"]).split()]
    acq = dict(model=J.get("ManufacturersModelName"), series=J.get("SeriesDescription"), pe_axis=pe, pe_polarity_known=known, pe_source=pe_src, readout=float(rt), readout_source=rt_src,
               partial_fourier=J.get("PartialFourier"), acq_matrix_pe=J.get("AcquisitionMatrixPE"), recon_matrix_pe=J.get("ReconMatrixPE"),
               zero_filled=bool(J.get("AcquisitionMatrixPE") and J.get("ReconMatrixPE") and J["ReconMatrixPE"] > J["AcquisitionMatrixPE"] * 1.1),
               dims=list(map(int, im.shape)), vox=[float(x) for x in im.header.get_zooms()[:3]], affine=im.affine.tolist(), shells=shells, shell_sizes=sizes,
               multishell=len([s for s in shells if s > 50]) > 1, raw_check=ck)
    json.dump(acq, open(SH / "acq.json", "w"), indent=1)
    rm(SH / "raw.nii", old, *glob.glob(str(SH / "raw_*")))


def crop_to_acquired():
    """dwidenoise and mrdegibbs assume voxels that were not interpolated. A zero-filled scan goes back on its acquired lines: the acquisition matrix on the frequency axis,
    that matrix over the phase share of the field of view on the phase axis (128 lines over 66% are 194 on this grid, as the b=0 spectrum shows). Voxel 0 stays where it was."""
    acq = json.load(open(SH / "acq.json")); J = json.load(open(SH / "raw.json")); n0 = int(J["AcquisitionMatrixPE"]); pct = float(J.get("PercentPhaseFOV") or 100); pe = "ij".index(acq["pe_axis"][0])
    n = [n0, n0]; n[pe] = 2 * round(n0 * 100 / pct / 2)
    run([mr("mrconvert"), SH / "raw.mif", SH / "crop_in.nii", "-strides", "-1,2,3,4", "-datatype", "float32le", "-export_grad_mrtrix", SH / "crop.b", "-force", "-quiet"], cwd=SH)
    im = nib.load(SH / "crop_in.nii"); A0, A1 = np.asarray(acq["affine"])[:3, :3], im.affine[:3, :3]
    if not np.allclose(A0 / np.linalg.norm(A0, axis=0), A1 / np.linalg.norm(A1, axis=0), atol=1e-3): raise RuntimeError("raw.mif and the sidecar's image differ in axis direction: the phase axis would not apply")
    d = np.asanyarray(im.dataobj).astype(np.float32); z = kspace_crop(d, *n); af = im.affine.copy(); af[:3, 0] *= d.shape[0] / n[0]; af[:3, 1] *= d.shape[1] / n[1]

    def centre(v, aff):
        m = v > v.mean(); return aff[:3, :3] @ np.argwhere(m).mean(0) + aff[:3, 3], float(v[m].mean())
    (c0, m0), (c1, m1) = centre(d[..., 0], im.affine), centre(z[..., 0], af); shift = float(np.linalg.norm(c1 - c0))
    if shift > 1.0: raise RuntimeError(f"the crop moved the brain by {shift:.2f} mm")
    o = nib.Nifti1Image(z, af); o.set_qform(af, code=1); o.set_sform(af, code=1); o.header.set_xyzt_units("mm", "sec"); nib.save(o, SH / "crop_out.nii"); del d, z
    run([mr("mrconvert"), SH / "crop_out.nii", SH / "raw_crop.mif", "-grad", SH / "crop.b", "-json_import", SH / "raw.json", "-strides", "-1,2,3,4", "-datatype", "float32le", "-force", "-quiet"], cwd=SH)
    rm(SH / "crop_in.nii", SH / "crop_out.nii", SH / "crop.b")
    vox = [float(x) for x in o.header.get_zooms()[:3]]; info = dict(matrix_from=list(map(int, im.shape[:2])), matrix_to=n, percent_phase_fov=pct, vox_to=vox, centre_shift_mm=round(shift, 3), brain_mean_ratio=round(m1 / m0, 4))
    acq.update(dims=list(map(int, o.shape)), vox=vox, affine=af.tolist(), cropped=info); json.dump(acq, open(SH / "acq.json", "w"), indent=1)
    return info


def s2a_denoise():
    if not (SH / "raw.mif").exists(): s1_inputs()
    src = SH / "raw.mif"; info = {}
    if a.crop_zf and json.load(open(SH / "acq.json"))["zero_filled"]:
        info["crop"] = crop_to_acquired(); src = SH / "raw_crop.mif"
    run([mr("dwidenoise"), src, SH / "den.mif", "-noise", SH / "noise.mif", "-force", "-quiet", "-nthreads", T], cwd=SH)
    run([mr("mrdegibbs"), SH / "den.mif", SH / "deg.mif", "-force", "-quiet", "-nthreads", T], cwd=SH)
    rm(SH / "den.mif", SH / "raw.mif", SH / "raw_crop.mif")
    return info


# ---------------- s2: preprocessing by arm
def s2_import():
    src = DER / f"eddy/{UNIT}_preproc.mif"; info = dict(grad_source="stored eddy output")
    if subprocess.run([mr("mrinfo"), src, "-dwgrad"], env=env, capture_output=True).returncode != 0:     # as rb2_subject.py: table from the stored raw image, same volume order
        run([mr("mrinfo"), DER / f"mif_dwi/{UNIT}.mif", "-export_grad_mrtrix", W / "grad_raw.b", "-force"])
        run([mr("mrconvert"), src, W / "dwi.mif", "-grad", W / "grad_raw.b", "-force", "-quiet"]); info["grad_source"] = "stored raw image (not eddy-rotated)"
    return info


def even_pad(src, dst, strides):
    """topup needs even sizes: repeat the last plane on an odd axis, as dwifslpreproc does. Returns [(axis, original size)]."""
    size = [int(x) for x in out([mr("mrinfo"), src, "-size"]).split()]; cur = src; pads = []; tmp = []
    for ax in range(3):
        if size[ax] % 2:
            last, nxt = W / f"pad{ax}_last.mif", W / f"pad{ax}_{Path(dst).name}.mif"
            run([mr("mrconvert"), cur, last, "-coord", ax, size[ax] - 1, "-force", "-quiet"]); run([mr("mrcat"), cur, last, nxt, "-axis", ax, "-force", "-quiet"])
            tmp += [last, nxt]; cur = nxt; pads.append((ax, size[ax]))
    run([mr("mrconvert"), cur, dst, "-strides", strides, "-datatype", "float32le", "-force", "-quiet"]); rm(*tmp); return pads


def eddy_inputs():
    """FSL-layout image, gradients, acqp and index. Arm C pads odd axes so that topup, its field and eddy share one grid; arm B does not need to."""
    acq = json.load(open(SH / "acq.json"))
    run([mr("mrconvert"), SH / "deg.mif", W / "dwi_tmp.mif", "-strides", "-1,2,3,4", "-export_grad_fsl", W / "dwi_in.bvec", W / "dwi_in.bval", "-force", "-quiet"])
    pads = even_pad(W / "dwi_tmp.mif", W / "dwi_in.nii", "-1,2,3,4") if a.arm == "C" else []
    if a.arm != "C": run([mr("mrconvert"), W / "dwi_tmp.mif", W / "dwi_in.nii", "-datatype", "float32le", "-force", "-quiet"])
    rm(W / "dwi_tmp.mif")
    v = pe_vector(acq["pe_axis"], acq["affine"], nib.load(W / "dwi_in.nii").affine)
    check = None
    if acq["pe_polarity_known"]:                         # MRtrix's own reading of the same sidecar must agree
        ex = W / "acqp_mrtrix.txt"; run([mr("mrinfo"), SH / "deg.mif", "-export_pe_eddy", ex, W / "index_mrtrix.txt", "-force"]); m = np.loadtxt(ex).reshape(-1, 4)[0]
        check = bool(np.allclose(m[:3], v) and abs(m[3] - acq["readout"]) < 1e-4); rm(ex, W / "index_mrtrix.txt")
    open(W / "acqp.txt", "w").write(f"{v[0]} {v[1]} {v[2]} {acq['readout']:.7f}\n" + (f"{v[0]} {v[1]} {v[2]} 0.0\n" if a.arm == "C" else ""))
    n = nib.load(W / "dwi_in.nii").shape[3]; open(W / "index.txt", "w").write(" ".join(["1"] * n) + "\n")
    json.dump(dict(pads=pads, pe_vector=v, mrtrix_pe_agrees=check), open(W / "eddy_in.json", "w"))


def s2b_synb0():
    """Native port: same steps and order as ~/synb0disco/synb0_full.sh 1-5, but b0s from the denoised raw series and the T1 of t1_choice."""
    S = W / "syn"; S.mkdir(exist_ok=True); t1 = DER / f"t1_anat/corrected_T1_{SUBJ}_I{T1ID}.nii.gz"
    if not t1.exists(): raise RuntimeError("same-visit corrected T1 not on disk")
    run([mr("dwiextract"), SH / "deg.mif", S / "b0s.nii.gz", "-bzero", "-strides", "-1,2,3,4", "-force", "-quiet"]); im = nib.load(S / "b0s.nii.gz")
    if im.ndim == 3 or im.shape[3] == 1: run([mr("mrconvert"), S / "b0s.nii.gz", S / "b0_native.nii.gz", "-axes", "0,1,2", "-force", "-quiet"])
    else:                                                 # motion between b0s would blur the input and move it off eddy's reference: align to the first b0, then mean
        run([f"{FB}/mcflirt", "-in", S / "b0s", "-refvol", 0, "-out", S / "b0s_mc"]); run([f"{FB}/fslmaths", S / "b0s_mc", "-Tmean", S / "b0_native"]); rm(S / "b0s_mc.nii.gz")
    rm(S / "b0s.nii.gz")
    run([PY, f"{SYN}/synb0_prep.py", t1, S / "b0_native.nii.gz", S], cwd=S)
    folds = []
    for i in range(1, 6):
        m = glob.glob(f"{SYN}/src/train_lin/num_fold_{i}_total_folds_5_*.pth"); f = S / f"b0_u_FOLD_{i}.nii.gz"
        run([PY, f"{SYN}/src/inference.py", S / "T1_norm_lin_atlas_2_5.nii.gz", S / "b0_d_lin_atlas_2_5.nii.gz", f, m[0]], cwd=S); folds.append(f)
    run([f"{FB}/fslmerge", "-t", S / "b0_u_merged", *folds]); run([f"{FB}/fslmaths", S / "b0_u_merged", "-Tmean", S / "b0_u_atlas"])
    run([PY, f"{SYN}/synb0_inverse.py", S], cwd=S); rm(S / "b0_u_merged.nii.gz", *folds)
    run([f"{FB}/fslmaths", S / "b0_native", "-s", "1.15", S / "b0_d_smooth"])


def s2c_topup():
    S = W / "syn"
    even_pad(S / "b0_d_smooth.nii.gz", W / "b0_d_pad.nii.gz", "-1,2,3"); even_pad(S / "b0_u.nii.gz", W / "b0_u_pad.nii.gz", "-1,2,3")
    run([f"{FB}/fslmerge", "-t", W / "b0_all", W / "b0_d_pad", W / "b0_u_pad"]); rm(W / "b0_d_pad.nii.gz", W / "b0_u_pad.nii.gz")
    run([f"{FB}/topup", "-v", f"--imain={W / 'b0_all'}", f"--datain={W / 'acqp.txt'}", f"--config={SYN}/src/synb0.cnf", f"--iout={W / 'b0_all_topup'}", f"--out={W / 'topup'}",
         f"--fout={W / 'topup_field_hz'}", f"--nthr={T}"], timeout=8 * 3600)


def s2d_mask():
    src = W / "dwi_in.mif"
    if a.arm == "C":
        run([f"{FB}/applytopup", f"--imain={W / 'dwi_in.nii'}", f"--datain={W / 'acqp.txt'}", "--inindex=1", f"--topup={W / 'topup'}", "--method=jac", "--interp=spline", f"--out={W / 'dwi_tu'}"])
        run([mr("mrconvert"), W / "dwi_tu.nii.gz", src, "-fslgrad", W / "dwi_in.bvec", W / "dwi_in.bval", "-force", "-quiet"]); rm(W / "dwi_tu.nii.gz")
    else:
        run([mr("mrconvert"), W / "dwi_in.nii", src, "-fslgrad", W / "dwi_in.bvec", W / "dwi_in.bval", "-force", "-quiet"])
    run(["bash", "-c", f"set -o pipefail; {MR}/dwi2mask {src} - -quiet | {MR}/maskfilter - dilate - -quiet | {MR}/mrconvert - {W / 'eddy_mask.nii'} -datatype float32le -strides -1,2,3 -force -quiet"])
    rm(src)


def s2e_eddy():
    x = W / "eddy_out"; topup = [f"--topup={W / 'topup'}"] if a.arm == "C" else []
    run([f"{FB}/eddy_cpu", f"--imain={W / 'dwi_in.nii'}", f"--mask={W / 'eddy_mask.nii'}", f"--acqp={W / 'acqp.txt'}", f"--index={W / 'index.txt'}", f"--bvecs={W / 'dwi_in.bvec'}",
         f"--bvals={W / 'dwi_in.bval'}", "--slm=linear", "--data_is_shelled", "--repol", f"--nthr={T}", f"--out={x}", "--verbose", *topup], timeout=16 * 3600, extra=dict(FSLOUTPUTTYPE="NIFTI"))   # a 54-volume 256 x 256 scan needs over 4 h at 8 threads; a busy machine must not count as a failed session
    pads = json.load(open(W / "eddy_in.json"))["pads"]; crop = []
    for ax, n in pads: crop += ["-coord", ax, f"0:{n - 1}"]
    run([mr("mrconvert"), f"{x}.nii", W / "dwi.mif", "-fslgrad", f"{x}.eddy_rotated_bvecs", W / "dwi_in.bval", *crop, "-force", "-quiet"])
    rm(f"{x}.nii", f"{x}.eddy_outlier_free_data.nii", W / "dwi_in.nii", W / "eddy_mask.nii")
    if a.arm in "BC" and (BASE / "work/B" / UNIT / ".s2e_eddy.done.json").exists() and (BASE / "work/C" / UNIT / ".s2e_eddy.done.json").exists():
        rm(SH / "deg.mif")


def preprocess():
    if a.arm == "A":
        stage("s2_import", s2_import); return
    if done("s2e_eddy") and not dirty:                   # the shared denoised image may be gone by now; it is not needed again
        for n in ["s2a_denoise"]:
            m = done(n, True)
            if m: qc["stages"][n] = dict(m, reused=True, shared=True)
        for n in ["s2_eddyin"] + (["s2b_synb0", "s2c_topup"] if a.arm == "C" else []) + ["s2d_mask", "s2e_eddy"]:
            m = done(n)
            if m: qc["stages"][n] = dict(m, reused=True, shared=False)
        return
    shared_stage("s2a_denoise", s2a_denoise, [SH / "deg.mif"])
    stage("s2_eddyin", eddy_inputs, [W / "eddy_in.json", W / "dwi_in.nii"])
    if a.arm == "C":
        stage("s2b_synb0", s2b_synb0, [W / "syn/b0_u.nii.gz"]); stage("s2c_topup", s2c_topup, [W / "topup_fieldcoef.nii.gz"])
    stage("s2d_mask", s2d_mask, [W / "eddy_mask.nii"]); stage("s2e_eddy", s2e_eddy, [W / "dwi.mif"])


# ---------------- s3: model
def s3_model():
    if a.arm != "A" and not (W / "dwi.mif").exists(): raise RuntimeError(f"arm {a.arm}: eddy's output is gone, so s3 cannot re-run; remove .s2e_eddy.done.json to build it again")   # never the stored eddy output in its place
    src = W / "dwi.mif" if (W / "dwi.mif").exists() else DER / f"eddy/{UNIT}_preproc.mif"
    shells = [float(x) for x in out([mr("mrinfo"), src, "-shell_bvalues"]).split()]; dw = [s for s in shells if s > 50]; info = dict(shells_in=shells, csd_shell=None)
    k = min(dw, key=lambda s: abs(s - 1000)); info["csd_shell"] = k
    if abs(k - 1000) > 100: raise RuntimeError(f"no b=1000 shell: {shells}")
    if len(dw) > 1:                                      # eddy saw every shell; CSD and tensor see b=0 and b=1000 only
        lo = [s for s in shells if s <= 50]; spec = ",".join(str(int(round(s))) for s in lo + [k])
        run([mr("dwiextract"), src, W / "dwi_sh.mif", "-shells", spec, "-force", "-quiet"]); src = W / "dwi_sh.mif"
    run([mr("dwibiascorrect"), "ants", src, W / "dwi_unb.mif", "-scratch", SCR, "-force", "-quiet", "-nthreads", T])
    run([mr("dwi2mask"), W / "dwi_unb.mif", W / "mask.mif", "-force", "-quiet"])
    info["mask_ml"] = round(float(out([mr("mrstats"), W / "mask.mif", "-mask", W / "mask.mif", "-output", "count", "-quiet"]).split()[0]) * np.prod([float(x) for x in out([mr("mrinfo"), W / "mask.mif", "-spacing"]).split()[:3]]) / 1000, 1)
    if info["mask_ml"] < 600: raise RuntimeError(f"DWI mask is only {info['mask_ml']} mL: the diffusion image is not a whole brain")
    run([mr("dwi2response"), "dhollander", W / "dwi_unb.mif", W / "wm.txt", W / "gm.txt", W / "csf.txt", "-mask", W / "mask.mif", "-scratch", SCR, "-force", "-quiet", "-nthreads", T])
    rows = [l for l in open(W / "wm.txt") if l.strip() and not l.startswith("#")]; open(W / "wm_csd.txt", "w").write(rows[-1])    # single-shell CSD: the b=1000 row
    run([mr("dwi2fod"), "csd", W / "dwi_unb.mif", W / "wm_csd.txt", W / "wmfod.mif", "-mask", W / "mask.mif", "-lmax", 8, "-force", "-quiet", "-nthreads", T])
    run([mr("dwi2tensor"), W / "dwi_unb.mif", W / "dt.mif", "-mask", W / "mask.mif", "-force", "-quiet", "-nthreads", T])
    run([mr("tensor2metric"), W / "dt.mif", "-fa", W / "fa.mif", "-adc", W / "md.mif", "-ad", W / "ad.mif", "-rd", W / "rd.mif", "-force", "-quiet", "-nthreads", T]); rm(W / "dt.mif")
    run(["bash", "-c", f"set -o pipefail; {MR}/dwiextract {W / 'dwi_unb.mif'} -bzero - -quiet | {MR}/mrmath - mean {W / 'b0mean.nii.gz'} -axis 3 -force -quiet"])
    run([mr("mrcalc"), W / "b0mean.nii.gz", W / "mask.mif", "-mult", W / "b0brain.nii.gz", "-force", "-quiet"])
    run([mr("mrgrid"), W / "fa.mif", "regrid", "-voxel", 1, W / "fa1mm.nii.gz", "-force", "-quiet"])
    run([mr("mrgrid"), W / "mask.mif", "regrid", "-template", W / "fa1mm.nii.gz", "-interp", "nearest", W / "mask1mm.nii.gz", "-datatype", "uint8", "-force", "-quiet"])
    rm(W / "dwi.mif", W / "dwi_sh.mif", W / "dwi_unb.mif", W / "grad_raw.b")
    return info


# ---------------- s4: tissue
def s4a_fastsurfer():
    st = DER / f"fastsurfer/{UNIT}"; lg = st / "scripts/deep-seg.log"; need = ["orig.mgz", "orig_nu.mgz", "mask.mgz", "aparc.DKTatlas+aseg.deep.mgz"]; info = {}
    used = [Path(x).name for x in re.findall(r"--t1[ =]+(\S+)", lg.read_text(errors="ignore")) if "/mri/" not in x] if lg.is_file() else []
    reuse = bool(used) and all(f"_I{T1ID}." in n for n in used) and all((st / "mri" / n).exists() for n in need)
    if reuse: src = st / "mri"; info.update(fastsurfer="stored, same T1")
    else:
        t1 = DER / f"t1_anat/corrected_T1_{SUBJ}_I{T1ID}.nii.gz"
        if not t1.exists(): raise RuntimeError("same-visit corrected T1 not on disk")
        src = SH / "fs" / UNIT / "mri"; dn = SH / "fs_cpu_done.json"
        if not dn.exists() or not all((src / n).exists() for n in need):
            s = time.time(); (SH / "fs").mkdir(exist_ok=True)
            run([FS_SH, "--t1", t1, "--sid", UNIT, "--sd", SH / "fs", "--seg_only", "--no_cereb", "--no_hypothal", "--device", "cpu", "--viewagg_device", "cpu", "--threads", T,
                 "--py", f"{FS_PY} -s", "--fs_license", FS_LIC], cwd=SCR)
            json.dump(dict(fastsurfer_sec=round(time.time() - s), fastsurfer_threads=a.threads), open(dn, "w"))
        d = json.load(open(dn)); info.update(fastsurfer="run here on CPU, --seg_only", fastsurfer_sec=d["fastsurfer_sec"], fastsurfer_threads=d.get("fastsurfer_threads", d.get("threads")))   # "threads" is the stage marker's own key
    run([mr("mrconvert"), src / "orig.mgz", SH / "t1_orig.nii.gz", "-strides", "-1,2,3", "-datatype", "float32le", "-force", "-quiet"], cwd=SH)
    run([mr("mrcalc"), src / "orig_nu.mgz", src / "mask.mgz", "-mult", SH / "t1brain_tmp.mif", "-force", "-quiet"], cwd=SH)
    run([mr("mrconvert"), SH / "t1brain_tmp.mif", SH / "t1brain.nii.gz", "-strides", "-1,2,3", "-datatype", "float32le", "-force", "-quiet"], cwd=SH); rm(SH / "t1brain_tmp.mif")
    run([mr("mrconvert"), src / "aparc.DKTatlas+aseg.deep.mgz", SH / "aparc.nii.gz", "-strides", "-1,2,3", "-datatype", "int32le", "-force", "-quiet"], cwd=SH)
    im = nib.load(SH / "aparc.nii.gz"); L = np.asanyarray(im.dataobj)
    w = nib.Nifti1Image(np.isin(L, [2, 41, 77]).astype(np.uint8), im.affine); w.set_qform(im.affine, code=1); w.set_sform(im.affine, code=1); nib.save(w, SH / "wmseg.nii.gz")
    return info


def aat(moving, ref, outp, *xfm):
    tx = []
    for x in xfm: tx += ["-t", x]
    run([f"{ANTS}/antsApplyTransforms", "-d", 3, "-i", moving, "-r", ref, "-o", outp, "-n", "NearestNeighbor", "-u", "int", *tx])


def s4b_align():
    fa, aff = canon(W / "fa1mm.nii.gz"); fa = fa > 0.25; res = {}
    def wmdice(p):
        L, af = canon(p)
        if not np.allclose(af, aff, atol=1e-3): raise RuntimeError("labels are not on the FA grid")
        return dice(np.isin(L, WM_DICE_LABELS), fa)
    run([f"{ANTS}/antsRegistrationSyNQuick.sh", "-d", 3, "-f", W / "b0mean.nii.gz", "-m", SH / "t1brain.nii.gz", "-o", W / "t12b0_", "-t", "r", "-n", T])
    aat(SH / "aparc.nii.gz", W / "fa1mm.nii.gz", W / "labels_rigid.nii.gz", W / "t12b0_0GenericAffine.mat"); res["dice_rigid"] = wmdice(W / "labels_rigid.nii.gz")
    rm(W / "t12b0_Warped.nii.gz", W / "t12b0_InverseWarped.nii.gz")
    try:
        run([f"{FB}/epi_reg", f"--epi={W / 'b0brain.nii.gz'}", f"--t1={SH / 't1_orig.nii.gz'}", f"--t1brain={SH / 't1brain.nii.gz'}", f"--wmseg={SH / 'wmseg.nii.gz'}", f"--out={W / 'bbr'}"])
    except subprocess.CalledProcessError as e:                      # a session where BBR cannot run keeps rigid and says so; anything after this point is a bug and stops the run
        res["dice_bbr"] = None; res["bbr_error"] = str(e)[:300]; log.write(f"BBR failed: {e}\n")
    else:
        run([f"{FB}/convert_xfm", "-omat", W / "t12b0_bbr.mat", "-inverse", W / "bbr.mat"]); flirt_to_itk(W / "t12b0_bbr.mat", SH / "t1brain.nii.gz", W / "b0brain.nii.gz", W / "t12b0_bbr.txt")
        aat(SH / "aparc.nii.gz", W / "fa1mm.nii.gz", W / "labels_bbr.nii.gz", W / "t12b0_bbr.txt"); res["dice_bbr"] = wmdice(W / "labels_bbr.nii.gz")
        # the conversion must carry the same transform: FLIRT's own resampling against ANTs with the converted one
        run([f"{FB}/flirt", "-in", SH / "wmseg.nii.gz", "-ref", W / "b0brain.nii.gz", "-applyxfm", "-init", W / "t12b0_bbr.mat", "-interp", "nearestneighbour", "-out", W / "chk_flirt"])
        aat(SH / "wmseg.nii.gz", W / "b0brain.nii.gz", W / "chk_ants.nii.gz", W / "t12b0_bbr.txt")
        x, _ = canon(W / "chk_flirt.nii.gz"); y, _ = canon(W / "chk_ants.nii.gz"); res["bbr_flirt_vs_ants_dice"] = dice(x > 0, y > 0); rm(W / "chk_flirt.nii.gz", W / "chk_ants.nii.gz")
        if res["bbr_flirt_vs_ants_dice"] < 0.98: raise RuntimeError(f"FLIRT to ITK conversion disagrees: {res['bbr_flirt_vs_ants_dice']}")
    for f in glob.glob(str(W / "bbr*")):
        if not f.endswith(".mat"): rm(f)
    use_bbr = res["dice_bbr"] is not None and res["dice_bbr"] > res["dice_rigid"]; res["chosen"] = "bbr" if use_bbr else "rigid"; res["wm_dice"] = max(res["dice_rigid"], res["dice_bbr"] or 0)
    os.replace(W / f"labels_{res['chosen']}.nii.gz", W / "aparc_dwi1mm.nii.gz"); rm(W / "labels_rigid.nii.gz", W / "labels_bbr.nii.gz")
    res["xfm"] = str(W / ("t12b0_bbr.txt" if use_bbr else "t12b0_0GenericAffine.mat")); json.dump(res, open(W / "align.json", "w"), indent=1)
    return {k: v for k, v in res.items() if k != "xfm"}


def s4c_5tt():
    run([mr("5ttgen"), "freesurfer", W / "aparc_dwi1mm.nii.gz", W / "5tt.mif", "-lut", FS_LUT, "-sgm_amyg_hipp", "-nocrop", "-scratch", SCR, "-force", "-quiet", "-nthreads", T])
    run([mr("5ttcheck"), W / "5tt.mif"]); run([mr("mrconvert"), W / "5tt.mif", W / "tt_chk.nii", "-force", "-quiet"])
    tt, af = canon(W / "tt_chk.nii"); L, af2 = canon(W / "aparc_dwi1mm.nii.gz")
    if not np.allclose(af, af2, atol=1e-3): raise RuntimeError("5TT and labels are not on one grid")
    chk = {}; vox = {}
    for name, (ls, ch) in TT_EXPECT.items():
        for l in ls:
            m = L == l
            if m.any(): chk[str(l)] = round(float(tt[m][:, ch].mean()), 4)
        vox[name] = int((tt[..., ch] > 0.5).sum())
    ctx = (L >= 1000) & (L < 3000); chk["cortex_1000s"] = round(float(tt[ctx][:, 0].mean()), 4); vdc = np.isin(L, [28, 60])
    if not vdc.any() or tt[vdc][:, 2].min() < 0.999: raise RuntimeError("test failed: ventral DC is not WM in the 5TT")
    if min(v for k, v in chk.items()) < 0.999: raise RuntimeError(f"test failed: a label is off MRtrix's table: {chk}")
    fa, _ = canon(W / "fa1mm.nii.gz"); rm(W / "tt_chk.nii")
    return dict(label_share_in_expected_tissue=chk, ventral_dc_in_wm=float(tt[vdc][:, 2].mean()), tissue_voxels=vox, wm5tt_dice=dice(tt[..., 2] > 0.5, fa > 0.25))


# ---------------- s5: atlas through the T1
def s5a_syn():
    run([f"{ANTS}/antsRegistrationSyNQuick.sh", "-d", 3, "-f", SH / "t1brain.nii.gz", "-m", MNI, "-o", SH / "mni2t1_", "-t", "s", "-n", T], cwd=SH)
    rm(SH / "mni2t1_InverseWarped.nii.gz", SH / "mni2t1_Warped.nii.gz", SH / "mni2t1_1InverseWarp.nii.gz")


def nmap():
    m = pd.read_csv(MAP).sort_values("matrix_idx").reset_index(drop=True); assert list(m.matrix_idx) == list(range(1, 167))
    assert m.atlas_label.iloc[154:].str.match(r"(VTA|SN_pc|SN_pr|Red_N|LC|Raphe)").all() and not m.atlas_label.iloc[:154].str.match(r"(VTA|SN_pc|SN_pr|Red_N|LC|Raphe)").any()
    return m


def s5b_atlas():
    x = json.load(open(W / "align.json"))["xfm"]; m = nmap()
    aat(ATLAS, W / "fa1mm.nii.gz", W / "aal3_dwi1mm.nii.gz", x, SH / "mni2t1_1Warp.nii.gz", SH / "mni2t1_0GenericAffine.mat")
    im = nib.load(W / "aal3_dwi1mm.nii.gz"); lab = np.rint(np.asanyarray(im.dataobj)).astype(int)
    lut = np.zeros(max(400, lab.max() + 1), int); lut[m.atlas_value.astype(int)] = m.matrix_idx.astype(int)
    o = nib.Nifti1Image(lut[lab].astype(np.int32), im.affine); o.set_data_dtype(np.int32); o.set_qform(im.affine, code=1); o.set_sform(im.affine, code=1); nib.save(o, W / "nodes166.nii.gz"); rm(W / "aal3_dwi1mm.nii.gz")
    nd, af = canon(W / "nodes166.nii.gz"); mask, af2 = canon(W / "mask1mm.nii.gz")
    if not np.allclose(af, af2, atol=1e-3): raise RuntimeError("nodes and mask are not on one grid")
    vol = np.bincount(nd.ravel(), minlength=167)[1:167]; inm = np.bincount(nd[mask > 0].ravel(), minlength=167)[1:167]; share = inm / np.maximum(vol, 1)
    return dict(nodes_present_154=int((vol[:154] > 0).sum()), nuclei_present_12=int((vol[154:] > 0).sum()), min_vox_154=int(vol[:154].min()), nodes_under_half_in_dwi_mask_154=int((share[:154] < 0.5).sum()), median_node_share_in_mask_154=float(np.median(share[:154])))


# ---------------- s6: tracking, weights, samples; the tractogram does not survive
def s6a_tckgen():
    cap = max(3, a.select / 1e6 * 1.6) * 3600
    s = time.time()
    run([mr("tckgen"), W0 / "wmfod.mif", W / "tracks.tck", "-algorithm", "iFOD2", "-act", W0 / "5tt.mif", "-backtrack", "-crop_at_gmwmi", "-seed_dynamic", W0 / "wmfod.mif", "-select", a.select,
         "-cutoff", "0.06", "-angle", 45, "-minlength", 10, "-maxlength", 250, "-step", 1, "-nthreads", T, "-force", "-quiet"], timeout=cap)
    kv = dict((l.split(":", 1)[0].strip(), l.split(":", 1)[1].strip()) for l in out([mr("tckinfo"), W / "tracks.tck"]).splitlines() if ":" in l and "command_history" not in l)
    c, tot = int(kv["count"]), int(kv["total_count"])
    if c != a.select: raise RuntimeError(f"tckgen stopped at {c} of {a.select} streamlines")
    return dict(count=c, total_count=tot, acceptance=c / max(tot, 1), tck_gb=round(os.path.getsize(W / "tracks.tck") / 1e9, 3), tckgen_sec=round(time.time() - s))


def s6b_sift2():
    run([mr("tcksift2"), W / "tracks.tck", W0 / "wmfod.mif", W / "weights.txt", "-act", W0 / "5tt.mif", "-out_mu", W / "mu.txt", "-nthreads", T, "-force", "-quiet"])
    np.save(W / "weights_f32.npy", read_vec(W / "weights.txt")); rm(W / "weights.txt")
    return dict(mu=float(open(W / "mu.txt").read().split()[-1]))


def s6c_sample():
    for k in ["fa", "md", "ad", "rd"]:
        run([mr("tcksample"), W / "tracks.tck", W0 / f"{k}.mif", W / f"{k}_mean.txt", "-stat_tck", "mean", "-nthreads", T, "-force", "-quiet"])
        np.save(W / f"{k}_f32.npy", read_vec(W / f"{k}_mean.txt")); rm(W / f"{k}_mean.txt")
    run([mr("tckstats"), W / "tracks.tck", "-dump", W / "lengths.txt", "-force", "-quiet"]); np.save(W / "lengths_f32.npy", read_vec(W / "lengths.txt")); rm(W / "lengths.txt")
    run([mr("tckresample"), W / "tracks.tck", "-endpoints", W / "ends.tck", "-force", "-quiet", "-nthreads", T]); np.save(W / "endpoints_f32.npy", endpoints_from_tck(W / "ends.tck")); rm(W / "ends.tck")


def s6d_assign():
    """Assignments from the saved float32 endpoints must equal tck2connectome on the whole tractogram; only then is the tractogram deleted."""
    nodes = W0 / "nodes166.nii.gz"; common = ["-assignment_radial_search", 2.0, "-symmetric", "-zero_diagonal", "-force", "-quiet", "-nthreads", T]
    run([mr("tck2connectome"), W / "tracks.tck", nodes, W / "count_full.csv", "-out_assignments", W / "asg_full.txt", *common])
    ends = np.load(W / "endpoints_f32.npy"); endpoints_to_tck(ends, W / "ends_rebuilt.tck")
    run([mr("tck2connectome"), W / "ends_rebuilt.tck", nodes, W / "count_ends.csv", "-out_assignments", W / "asg_ends.txt", *common])
    A = read_vec(W / "asg_full.txt", np.int32).reshape(-1, 2); B = read_vec(W / "asg_ends.txt", np.int32).reshape(-1, 2); C1 = np.loadtxt(W / "count_full.csv", delimiter=","); C2 = np.loadtxt(W / "count_ends.csv", delimiter=",")
    n = len(np.load(W / "lengths_f32.npy"))
    if not (len(A) == len(B) == len(ends) == n == a.select): raise RuntimeError(f"streamline counts differ: {len(A)} {len(B)} {len(ends)} {n}")
    proof = dict(n=int(len(A)), assignment_mismatches=int((A != B).any(1).sum()), count_matrix_equal=bool(np.array_equal(C1, C2)))
    if proof["assignment_mismatches"] or not proof["count_matrix_equal"]: raise RuntimeError(f"endpoint assignment differs from tck2connectome: {proof}")
    for k in ["fa", "md", "ad", "rd", "weights", "lengths", "endpoints"]:              # a text parse that stops early must not pass
        if len(np.load(W / f"{k}_f32.npy", mmap_mode="r")) != n: raise RuntimeError(f"{k} array has the wrong length")
    np.save(W / "assignments_u8.npy", A.astype(np.uint8)); rm(W / "tracks.tck", W / "ends_rebuilt.tck", W / "asg_full.txt", W / "asg_ends.txt", W / "count_full.csv", W / "count_ends.csv")
    return dict(assign_proof=proof)


# ---------------- s7: QC json
def net_values(M, m):
    """p5_features.networks on the 154 nodes: per network, mean edge value over edges touching it (an edge inside the network counts twice)."""
    idx = np.arange(154); iu = np.triu_indices(154, 1); nets = m.functional_network.values[:154]; o = {}
    pres = (M["len_mean"][:154, :154][iu] > 0) & (M["fd_sum"][:154, :154][iu] > 0)
    for lab, key in (("FA", "fa_mean"), ("MD", "md_mean"), ("RD", "rd_mean"), ("AxD", "ad_mean")):
        v = M[key][:154, :154][iu]; ok = pres & np.isfinite(v) & (v > 0); o[lab] = {}
        for n in sorted(set(nets)):
            w = (nets[iu[0]] == n).astype(float) + (nets[iu[1]] == n).astype(float)
            o[lab][n] = float((w[ok] * v[ok]).sum() / max(w[ok].sum(), 1))
    glob_ = {lab: float(M[key][:154, :154][iu][(M[key][:154, :154][iu] > 0) & pres].mean()) for lab, key in (("FA", "fa_mean"), ("MD", "md_mean"), ("RD", "rd_mean"), ("AxD", "ad_mean"))}
    return o, glob_


def eddy_metrics():
    x = W0 / "eddy_out"
    if not Path(f"{x}.eddy_movement_rms").exists(): return None
    rms = np.loadtxt(f"{x}.eddy_movement_rms"); rr = np.loadtxt(f"{x}.eddy_restricted_movement_rms"); om = np.loadtxt(f"{x}.eddy_outlier_map", skiprows=1); bv = np.loadtxt(W0 / "dwi_in.bval")
    pads = json.load(open(W0 / "eddy_in.json"))["pads"]
    for ax, n in pads:
        if ax == 2: om = om[:, :n]
    dw = bv > 100
    return dict(abs_rms_mean=float(rms[:, 0].mean()), rel_rms_mean=float(rms[:, 1].mean()), rel_restricted_rms_mean=float(rr[:, 1].mean()), abs_rms_max=float(rms[:, 0].max()),
                outlier_slices=int(om[dw].sum()), outlier_pct=float(100 * om[dw].mean()), vols_over_5pct_outliers=int((om[dw].mean(1) > 0.05).sum()))


def topup_metrics():
    if not (W0 / "topup_field_hz.nii.gz").exists(): return None
    acq = json.load(open(SH / "acq.json")); ein = json.load(open(W0 / "eddy_in.json")); ax = int(np.argmax(np.abs(ein["pe_vector"]))); vox = acq["vox"][ax]
    hz, _ = canon(W0 / "topup_field_hz.nii.gz"); br, _ = canon(W0 / "b0brain.nii.gz"); hz = hz[: br.shape[0], : br.shape[1], : br.shape[2]]        # padding sits at the far end of an axis
    d = np.abs(hz[br > 0]) * acq["readout"] * vox
    return dict(displacement_mm_median=float(np.median(d)), displacement_mm_p95=float(np.percentile(d, 95)), displacement_mm_p99=float(np.percentile(d, 99)), pe_voxel_mm=vox)


def s7_qc():
    m = nmap(); ass = np.load(W / "assignments_u8.npy").astype(int); Ls, w = np.load(W / "lengths_f32.npy"), np.load(W / "weights_f32.npy"); n = len(ass)
    i, j = ass[:, 0], ass[:, 1]; ok = (i > 0) & (j > 0) & (i != j); i, j = i[ok] - 1, j[ok] - 1
    nd, af = canon(W0 / "nodes166.nii.gz"); vol = np.bincount(nd.ravel(), minlength=167)[1:167].astype(float)

    def esum(v):
        S = np.zeros((166, 166)); np.add.at(S, (i, j), v); return S + S.T
    cnt = esum(np.ones(ok.sum()))
    M = dict(count=cnt, fd_sum=esum(w[ok]), count_invnodevol=esum(2.0 / np.maximum(vol[i] + vol[j], 1)), len_mean=np.divide(esum(Ls[ok]), cnt, out=np.zeros_like(cnt), where=cnt > 0),
             invlen_mean=np.divide(esum(1.0 / np.maximum(Ls[ok], 1e-6)), cnt, out=np.zeros_like(cnt), where=cnt > 0))
    samp = {}
    for k in ["fa", "md", "rd", "ad"]:
        v = np.load(W / f"{k}_f32.npy"); samp[k] = v; vv = v[ok]; g = np.isfinite(vv) & (vv > 0)
        S = np.zeros((166, 166)); C = np.zeros((166, 166)); np.add.at(S, (i[g], j[g]), vv[g]); np.add.at(C, (i[g], j[g]), 1); S, C = S + S.T, C + C.T
        M[f"{k}_mean"] = np.divide(S, C, out=np.zeros_like(S), where=C > 0)
    np.savez_compressed(W / "matrices_aal166.npz", **M)
    p154 = cnt[:154, :154] > 0; iu154 = np.triu_indices(154, 1); iu166 = np.triu_indices(166, 1)
    tr = qc["stages"]["s6a_tckgen"]; fa_all = samp["fa"]; g = np.isfinite(fa_all) & (fa_all > 0)
    qc.update(acceptance=tr["acceptance"], count=tr["count"], total_count=tr["total_count"], mu=qc["stages"]["s6b_sift2"]["mu"], assign_proof=qc["stages"]["s6d_assign"]["assign_proof"],
              assigned_both=float(ok.mean()), density_154=float(p154[iu154].mean()), density_166=float((cnt > 0)[iu166].mean()), empty_nodes_154=int((p154.sum(1) == 0).sum()),
              empty_nuclei_of_12=int(((cnt > 0).sum(1)[154:] == 0).sum()), not_estimable_nodes=m.atlas_label.iloc[154:].tolist(),
              fa_along_mean=float(fa_all[g].mean()), fa_along_sift2_mean=float(np.average(fa_all[g], weights=np.maximum(w[g], 0))), len_median=float(np.median(Ls)), len_p95=float(np.percentile(Ls, 95)))
    qc["wb_net_values"], qc["wb_global"] = net_values(M, m)
    # WM against FA>0.25, on this arm's own FA and transform; the regional mask is fixed in T1 space (AAL3 labels) and travels with the transform
    al = json.load(open(W0 / "align.json")); fa, af1 = canon(W0 / "fa1mm.nii.gz"); L, af2 = canon(W0 / "aparc_dwi1mm.nii.gz"); mk, _ = canon(W0 / "mask1mm.nii.gz")
    if not (np.allclose(af, af1, atol=1e-3) and np.allclose(af, af2, atol=1e-3)): raise RuntimeError("QC images are not on one grid")
    fa = fa > 0.25; wm = np.isin(L, WM_DICE_LABELS); roi_all = np.zeros(fa.shape, bool); reg = {}
    for name, stems in ROI.items():
        ids = m.matrix_idx[m.atlas_label.str.replace(r"_[LR]$", "", regex=True).isin(stems)].values; r = np.isin(nd, ids); roi_all |= r
        reg[name] = dict(n_labels=int(len(ids)), dice=dice(wm & r, fa & r), wm_vox=int((wm & r).sum()), roi_vox=int(r.sum()))
    qc.update(wm_dice=al["wm_dice"], wm_dice_regional=dice(wm & roi_all, fa & roi_all), wm_dice_regional_parts=reg, registration=dict(chosen=al["chosen"], dice_rigid=al["dice_rigid"], dice_bbr=al.get("dice_bbr"), bbr_flirt_vs_ants_dice=al.get("bbr_flirt_vs_ants_dice")),
              fs_brain_in_dwi_mask=float((mk[L > 0] > 0).mean()), dwi_mask_ml=float((mk > 0).sum() / 1000))
    if a.tag and (W0 / "matrices_aal166.npz").exists():   # for information; the rule's whole-brain r is in rv4_compare.py
        Z = np.load(W0 / "matrices_aal166.npz"); x, y = Z["fd_sum"][:154, :154][iu154], M["fd_sum"][:154, :154][iu154]; both = (x > 0) & (y > 0)
        qc["vs_base"] = dict(density_base=float((x > 0).mean()), log_fd_r_on_shared_edges=float(np.corrcoef(np.log(x[both]), np.log(y[both]))[0, 1]), shared_edges=int(both.sum()))
    t = qc["stages"]
    for name, key in (("s4c_5tt", "tissue"), ("s5b_atlas", "atlas"), ("s3_model", "model"), ("s4a_fastsurfer", "t1_source")):
        if name in t: qc[key] = {k: v for k, v in t[name].items() if k not in ("ver", "sec", "peak_gb", "rss_so_far_gb", "threads", "reused", "shared", "finished", "deps")}
    qc["acq"] = {k: v for k, v in json.load(open(SH / "acq.json")).items() if k != "affine"}; qc["eddy"] = eddy_metrics(); qc["topup"] = topup_metrics()
    if a.arm in "BC" and (W0 / "eddy_in.json").exists(): qc["eddy_in"] = json.load(open(W0 / "eddy_in.json"))


def seconds_total(): return sum(v["sec"] for v in qc["stages"].values() if not v.get("reused"))


try:
    LOCK = W / "running.pid"                             # one worker per arm and session
    if LOCK.is_file():
        try:
            os.kill(int(LOCK.read_text().strip()), 0); print("busy", ARMTAG); sys.exit(0)
        except (ProcessLookupError, ValueError): pass
    LOCK.write_text(str(os.getpid())); t0 = time.time()
    prev = json.load(open(QCF)) if QCF.is_file() else {}; qc["attempt"] = prev.get("attempt", 0) + 1
    if prev.get("status") == "ok" and prev.get("versions") == VERSIONS:
        print("exists", ARMTAG); LOCK.unlink(missing_ok=True); sys.exit(0)
    save_qc()
    shared_stage("s1_inputs", s1_inputs, [SH / "acq.json"])
    if not a.tag:
        preprocess()
        stage("s3_model", s3_model, [W / "wmfod.mif"])
        shared_stage("s4a_fastsurfer", s4a_fastsurfer, [SH / "aparc.nii.gz", SH / "wmseg.nii.gz"])
        stage("s4b_align", s4b_align, [W / "aparc_dwi1mm.nii.gz"]); stage("s4c_5tt", s4c_5tt, [W / "5tt.mif"])
        shared_stage("s5a_syn", s5a_syn, [SH / "mni2t1_1Warp.nii.gz"]); stage("s5b_atlas", s5b_atlas, [W / "nodes166.nii.gz"])
    else:
        for f in ["wmfod.mif", "5tt.mif", "nodes166.nii.gz", "fa.mif", "md.mif", "ad.mif", "rd.mif", "align.json"]:
            if not (W0 / f).exists(): raise RuntimeError(f"variant run needs {f} from the base run of arm {a.arm}")
    stage("s6a_tckgen", s6a_tckgen, [W / "tracks.tck"] if not (W / "assignments_u8.npy").exists() else [])
    stage("s6b_sift2", s6b_sift2, [W / "weights_f32.npy"])
    stage("s6c_sample", s6c_sample, [W / "endpoints_f32.npy"])
    stage("s6d_assign", s6d_assign, [W / "assignments_u8.npy"])
    rm(marker("s7_qc", False)); _stage("s7_qc", s7_qc, False, [])                   # always recomputed: it fills this run's QC from the saved arrays
    qc.update(status="ok", seconds_total_this_attempt=round(time.time() - t0), work_gb_final=round(dir_bytes(W) / 1e9, 3)); qc.pop("failed_stage", None)
    if a.arm == "A" and not a.tag: rm(W / "wmfod.mif", W / "5tt.mif")      # the reference arm has no variants
    log.write(f"[{time.strftime('%H:%M:%S')}] done\n"); save_qc(); LOCK.unlink(missing_ok=True); print("ok", ARMTAG, round(time.time() - t0), "s")
    try: SCR.rmdir()                                     # the per-run scratch folder, if nothing is left in it
    except OSError: pass
except SystemExit:
    raise
except BaseException as exc:                             # the driver reads status and allows one retry
    qc.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:800], traceback=traceback.format_exc()[-1500:]); save_qc()
    (W / "running.pid").unlink(missing_ok=True); log.write(qc["traceback"] + "\n"); print("FAILED", ARMTAG, qc.get("failed_stage"), str(exc)[:300], file=sys.stderr); sys.exit(1)
