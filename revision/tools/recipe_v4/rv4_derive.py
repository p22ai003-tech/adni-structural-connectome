"""Analysis inputs from one finished recipe v4 session (after the cohort run; not part of the signed recipe, logged in FREEZE.md).
Reads only what the run kept (per-streamline arrays, node images, QC files); writes only under <out>; never touches the run's folders.

  aal_act10m   the ACT run's AAL3 matrices (matrices_aal166.npz) as the CSV files + qc.json the analysis chain reads
  aal_noact3m  the same for the no-ACT run (3M streamlines)
  aal_act3m    AAL3 matrices of the ACT run's 3M subset (first 3,000,000 streamlines, their own SIFT2 weights: recipe step 8),
               built with the recipe's own matrix code (s7_qc); the like-for-like partner of aal_noact3m
  dkt_*        the same three on DKT76 (62 DKT cortical regions + 14 subcortical, as rb_dk.py), from the session's own FastSurfer
               aparc.DKTatlas+aseg.deep already on the diffusion grid (aparc_dwi1mm.nii.gz, the recipe's transform); endpoints
               assigned by tck2connectome with the recipe's settings (radial search 2 mm), as the recipe does for AAL3

Checks, any failure stops this session with nothing written: the matrix code applied to the ACT and no-ACT runs' own arrays
must give back each run's matrices_aal166.npz exactly; every array must have one row per streamline.
Usage: rv4_derive.py <unit> <out root>"""
import hashlib, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np, nibabel as nib

U, OUT = sys.argv[1], Path(sys.argv[2])
B = Path("/data/derivatives/recipe_v4/cohort"); W, WN = B / "work/C" / U, B / "work/C_noact" / U
MR = "/home/ec2-user/mrtrix3/bin"; SUB = 3_000_000
CODE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CORT = [2, 3] + list(range(5, 32)) + [34, 35]
DKT = [1000 + c for c in CORT] + [2000 + c for c in CORT] + [10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]


def canon(path):
    im = nib.as_closest_canonical(nib.load(str(path))); return np.asanyarray(im.dataobj), im.affine


def endpoints_to_tck(ends, path):                       # rv4_subject.endpoints_to_tck
    n = len(ends); body = np.full((n, 3, 3), np.nan, dtype="<f4"); body[:, :2] = ends
    head = f"mrtrix tracks\ncount: {n}\ndatatype: Float32LE\nfile: . "
    off = len(head) + 16; head += f"{off}\nEND\n"; head = head + " " * (off - len(head))
    with open(path, "wb") as fh:
        fh.write(head.encode()); body.tofile(fh); np.full(3, np.inf, dtype="<f4").tofile(fh)


def read_vec(path, dtype=np.float32):                   # rv4_subject.read_vec
    with open(path) as fh: txt = fh.read()
    if "#" in txt: txt = re.sub(r"(?m)^#.*$", "", txt)
    return np.fromstring(txt, dtype=np.float64, sep=" ").astype(dtype)


def matrices(ass, Ls, w, samp, vol, N):
    """rv4_subject.s7_qc's matrix block, with the number of nodes as a parameter."""
    i, j = ass[:, 0], ass[:, 1]; ok = (i > 0) & (j > 0) & (i != j); i, j = i[ok] - 1, j[ok] - 1

    def esum(v):
        S = np.zeros((N, N)); np.add.at(S, (i, j), v); return S + S.T
    cnt = esum(np.ones(ok.sum()))
    M = dict(count=cnt, fd_sum=esum(w[ok]), count_invnodevol=esum(2.0 / np.maximum(vol[i] + vol[j], 1)),
             len_mean=np.divide(esum(Ls[ok]), cnt, out=np.zeros_like(cnt), where=cnt > 0),
             invlen_mean=np.divide(esum(1.0 / np.maximum(Ls[ok], 1e-6)), cnt, out=np.zeros_like(cnt), where=cnt > 0))
    for k in ["fa", "md", "rd", "ad"]:
        vv = samp[k][ok]; g = np.isfinite(vv) & (vv > 0)
        S = np.zeros((N, N)); C = np.zeros((N, N)); np.add.at(S, (i[g], j[g]), vv[g]); np.add.at(C, (i[g], j[g]), 1); S, C = S + S.T, C + C.T
        M[f"{k}_mean"] = np.divide(S, C, out=np.zeros_like(S), where=C > 0)
    return M, float(ok.mean())


def arrays(d, n_expect):
    a = dict(ass=np.load(d / "assignments_u8.npy").astype(int), Ls=np.load(d / "lengths_f32.npy"), w=np.load(d / "weights_f32.npy"),
             ends=np.load(d / "endpoints_f32.npy", mmap_mode="r"), samp={k: np.load(d / f"{k}_f32.npy") for k in ["fa", "md", "ad", "rd"]})
    for k, v in [("ass", a["ass"]), ("Ls", a["Ls"]), ("w", a["w"]), ("ends", a["ends"])] + list(a["samp"].items()):
        if len(v) != n_expect: raise RuntimeError(f"{d.parent.name}: {k} has {len(v)} rows, expected {n_expect}")
    return a


def same(M, npz, what):
    z = np.load(npz)
    if sorted(z.files) != sorted(M): raise RuntimeError(f"{what}: matrix kinds differ")
    for k in z.files:
        if not np.array_equal(z[k], M[k]): raise RuntimeError(f"{what}: {k} differs from the run's own matrices (max {np.abs(z[k] - M[k]).max():.3g})")


def write(dest, prefix, M, qc):
    d = OUT / dest; d.mkdir(parents=True, exist_ok=True)
    for k, m in M.items():
        tmp = d / f".{prefix}_{U}_{k}.tmp{os.getpid()}"; np.savetxt(tmp, m, delimiter=",", fmt="%.10g"); os.replace(tmp, d / f"{prefix}_{U}_{k}.csv")
    tmp = d / f".{prefix}_{U}_qc.tmp{os.getpid()}"; json.dump(qc, open(tmp, "w"), indent=1, default=float); os.replace(tmp, d / f"{prefix}_{U}_qc.json")


def density(M, N):
    iu = np.triu_indices(N, 1); return float((M["count"][iu] > 0).mean())


t0 = time.time()
qa, qn = (json.load(open(B / f"qc/{p}_{U}.json")) for p in ("C", "C_noact"))
if qa.get("status") != "ok" or qn.get("status") != "ok": sys.exit(f"{U}: not both runs ok")
if qa["recipe_sha256"] != qn["recipe_sha256"]: sys.exit(f"{U}: the two runs carry different recipe hashes")
nd, af = canon(W / "nodes166.nii.gz"); vol166 = np.bincount(nd.ravel(), minlength=167)[1:167].astype(float)
A, N3 = arrays(W, int(qa["count"])), arrays(WN, int(qn["count"]))
if len(A["ass"]) != 10_000_000 or len(N3["ass"]) != SUB: raise RuntimeError("unexpected streamline counts")
w3 = np.load(W / "weights_3M_f32.npy")
if len(w3) != SUB: raise RuntimeError("3M subset weights have the wrong length")

# AAL3: the matrix code must give back both runs' own matrices before it builds the 3M subset
Ma, _ = matrices(A["ass"], A["Ls"], A["w"], A["samp"], vol166, 166); same(Ma, W / "matrices_aal166.npz", "ACT 10M")
Mn, _ = matrices(N3["ass"], N3["Ls"], N3["w"], N3["samp"], vol166, 166); same(Mn, WN / "matrices_aal166.npz", "no-ACT 3M")
s = slice(0, SUB)
M3, ab3 = matrices(A["ass"][s], A["Ls"][s], w3, {k: v[s] for k, v in A["samp"].items()}, vol166, 166)

# DKT76 nodes on the diffusion grid, from the session's own FastSurfer labels already carried by the recipe's transform
L, af2 = canon(W / "aparc_dwi1mm.nii.gz")
if not np.allclose(af, af2, atol=1e-3): raise RuntimeError("aparc_dwi1mm and nodes166 are not on one grid")
L = np.rint(L).astype(int); lut = np.zeros(max(3000, L.max() + 1), int); lut[DKT] = np.arange(1, len(DKT) + 1)
nodes = lut[L]; vol76 = np.bincount(nodes.ravel(), minlength=len(DKT) + 1)[1:].astype(float)
tmp = Path(tempfile.mkdtemp(prefix=f"derive_", dir="/data/derivatives/recipe_v4/scratch"))
try:
    im = nib.as_closest_canonical(nib.load(str(W / "aparc_dwi1mm.nii.gz")))
    o = nib.Nifti1Image(nodes.astype(np.int32), im.affine); o.set_data_dtype(np.int32); o.set_qform(im.affine, code=1); o.set_sform(im.affine, code=1)
    nib.save(o, tmp / "dkt76.nii.gz")
    asg = {}
    for name, ar in (("act", A), ("noact", N3)):
        endpoints_to_tck(np.asarray(ar["ends"]), tmp / f"{name}.tck")
        subprocess.run([f"{MR}/tck2connectome", tmp / f"{name}.tck", tmp / "dkt76.nii.gz", tmp / f"{name}.csv", "-out_assignments", tmp / f"{name}.txt",
                        "-assignment_radial_search", "2.0", "-symmetric", "-zero_diagonal", "-force", "-quiet", "-nthreads", "4"], check=True)
        asg[name] = read_vec(tmp / f"{name}.txt", np.int32).reshape(-1, 2)
        if len(asg[name]) != len(ar["ass"]): raise RuntimeError(f"DKT {name}: {len(asg[name])} assignments for {len(ar['ass'])} streamlines")
        (tmp / f"{name}.tck").unlink()
    Da, aba = matrices(asg["act"], A["Ls"], A["w"], A["samp"], vol76, 76)
    Dn, abn = matrices(asg["noact"], N3["Ls"], N3["w"], N3["samp"], vol76, 76)
    D3, abd3 = matrices(asg["act"][s], A["Ls"][s], w3, {k: v[s] for k, v in A["samp"].items()}, vol76, 76)
    if not np.array_equal(np.loadtxt(tmp / "act.csv", delimiter=","), Da["count"]): raise RuntimeError("DKT ACT: count differs from tck2connectome's own matrix")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

drv = lambda what, **k: dict(builder="rv4_derive.py", builder_sha256=CODE, built=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), what=what, **k)
write("aal_act10m", "SC_AAL166", Ma, qa)
write("aal_noact3m", "SC_AAL166", Mn, qn)
write("aal_act3m", "SC_AAL166", M3, {**qa, "count": SUB, "assigned_both": ab3, "density_154": density({"count": M3["count"][:154, :154]}, 154),
                                     "density_166": density(M3, 166), "mu": float(open(W / "mu_3M.txt").read().split()[-1]),
                                     "derived": drv("ACT run, first 3,000,000 streamlines with their own SIFT2 weights (recipe step 8)")})
dq = lambda q, ab, M, what: {"unit": U, "atlas": "DKT76 (FastSurfer aparc.DKTatlas+aseg.deep, recipe transform)", "status": "ok", "recipe_sha256": q["recipe_sha256"],
                             "wm_dice": q["wm_dice"], "reg_wm_dice": q["wm_dice"], "assigned_both": ab, "density": density(M, 76),
                             "zero_nodes": int(((M["count"] > 0).sum(1) == 0).sum()), "nodes_present": int((vol76 > 0).sum()), "min_node_vox": int(vol76.min()),
                             "derived": drv(what)}
write("dkt_act10m", "SC_DKT76", Da, dq(qa, aba, Da, "ACT run, 10M"))
write("dkt_noact3m", "SC_DKT76", Dn, dq(qn, abn, Dn, "no-ACT run, 3M"))
write("dkt_act3m", "SC_DKT76", D3, dq(qa, abd3, D3, "ACT run, first 3M with their own SIFT2 weights"))
print(json.dumps(dict(unit="<session>", seconds=round(time.time() - t0), dkt_nodes_present=int((vol76 > 0).sum()),
                      density=dict(act10m=round(density(Ma, 166), 4), act3m=round(density(M3, 166), 4), noact3m=round(density(Mn, 166), 4), dkt_act10m=round(density(Da, 76), 4)))))
