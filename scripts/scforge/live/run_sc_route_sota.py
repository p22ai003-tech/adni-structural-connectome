#!/usr/bin/env python3
"""SOTA structural-connectome route (the root-cause fix).

What this fixes vs the legacy step7 default route:
  * ANTs SyN *nonlinear* T1<->MNI registration (legacy used 12-DOF affine only).
  * AAL3/Schaefer kept at 1 mm and transformed into DWI *world* space with
    mrtransform (legacy crushed the atlas onto the 2 mm b0 voxel grid, which
    fragments/loses small deep nuclei).
  * AAL3 relabelled to a contiguous 1..166 node set (legacy emitted 169/170-node
    matrices with the structural gap labels 35/36/81/82 inflating zero-rows).
  * 10M ACT/SIFT2 streamlines for the fine 166-node atlas (legacy used 3M).

Reuses existing per-subject artifacts (native T1, BBR T1->b0 transform, FOD,
5TT, GMWMI, and optionally the existing 3M tracks). The expensive ANTs SyN
transform is cached per subject so the cheap `existing_tracks` diagnostic pass
and the full `full_10m` pass share it.

Outputs go to a scratch run root; nothing in production is overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import nibabel as nib

ROOT = Path("/home/ec2-user/exp")
DERIV = Path("/data/derivatives")
MRBIN = Path.home() / "mrtrix3" / "bin"
FSLBIN = Path.home() / "fsl" / "bin"
MNI_BRAIN = Path.home() / "fsl" / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz"
MNI_FULL = Path.home() / "fsl" / "data" / "standard" / "MNI152_T1_1mm.nii.gz"

AAL3_166_MNI = ROOT / "atlas" / "AAL" / "AAL3v1_1mm_166.nii.gz"
SCHAEFER_MNI = ROOT / "atlas" / "Schaefer2018" / "Schaefer2018_200Parcels_17Networks_1mm.nii.gz"

# AAL3 contiguous node set is 1..166 with NO structural gaps after labelconvert.
AAL3_NODES = 166
SCHAEFER_NODES = 200

# Required connectome weights: (suffix, [extra tck2connectome args], stat_edge, scale)
# fd_sum is the primary structural weight (SIFT2 fibre-density sum).
WEIGHT_SPECS = [
    ("count", [], None, None),
    ("fd_sum", [], "sum", None),
    ("count_invnodevol", [], None, "invnodevol"),
    ("len_mean", [], "mean", "length"),
]


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def base_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{MRBIN}{os.pathsep}{FSLBIN}{os.pathsep}" + env.get("PATH", "")
    env["FSLDIR"] = str(Path.home() / "fsl")
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    env["LC_ALL"] = "C"
    # Cap per-command parallelism so N workers can't each grab all cores. With many workers an
    # UNCAPPED mrtrix/ANTs defaults to nproc threads EACH (a 38-worker launch hit load 3000+ on a
    # 192-core box). MRTRIX_NTHREADS caps mrtrix (dwi2fod/dwi2response/dwi2mask/mrgrid/tckgen/...);
    # ITK_*/OMP cap ANTs. Honors the launch override; safe default of 4 otherwise.
    nt = os.environ.get("MRTRIX_NTHREADS", "4")
    env["MRTRIX_NTHREADS"] = nt
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = nt
    env["OMP_NUM_THREADS"] = nt
    return env


def run(cmd, log, env=None, timeout=None):
    env = env or base_env()
    cmd = [str(c) for c in cmd]
    with open(log, "a", encoding="utf-8", errors="replace") as h:
        h.write(f"[{utc()}] $ {' '.join(cmd)}\n")
        h.flush()
        p = subprocess.run(cmd, stdout=h, stderr=subprocess.STDOUT, env=env, timeout=timeout)
        h.write(f"[{utc()}] rc={p.returncode}\n\n")
    if p.returncode != 0:
        raise RuntimeError(f"command failed rc={p.returncode}: {' '.join(cmd[:3])}... (see {log})")


def subj_root(sid: str) -> str:
    return sid.split("_I", 1)[0] if "_I" in sid else sid


def first_glob(d: Path, *patterns):
    for pat in patterns:
        for p in sorted(d.glob(pat)):
            if p.is_file() and not p.name.startswith("."):
                return p
    return None


def resolve_inputs(sid: str) -> dict:
    sr = subj_root(sid)
    t1_anat = DERIV / "t1_anat"
    fod = DERIV / "fod" / sid
    bbr = DERIV / "dwi_t1_bbr"
    tracks = DERIV / "tracks" / sid
    t1_brain = first_glob(t1_anat, f"T1_ss_{sid}_*.nii.gz", f"T1_ss_{sr}_*.nii.gz")
    t1_full = first_glob(t1_anat, f"corrected_T1_{sid}_*.nii.gz", f"corrected_T1_{sr}_*.nii.gz",
                         f"t1_from_dicom_{sid}_*.nii.gz", f"t1_from_dicom_{sr}_*.nii.gz")
    inp = {
        "sid": sid,
        "t1_brain": t1_brain,
        "t1_full": t1_full,
        "wmfod": fod / "wmfod_final.mif",
        "five_tt": fod / "5tt_b0_fixed.mif",
        "gmwmi": fod / "gmwmi.mif",
        "t12b0_mrtrix": fod / "t12b0_bbr_mrtrix.txt",
        "tracks_3m": tracks / "tracks_final_3000k.tck",
        "sift_3m": tracks / "sift_weights.txt",
    }
    return inp


def check_inputs(inp: dict, mode: str) -> list[str]:
    missing = []
    need = ["t1_brain", "wmfod", "t12b0_mrtrix"]
    if mode == "existing_tracks":
        need += ["tracks_3m", "sift_3m"]
    else:
        need += ["five_tt", "gmwmi"]
    for k in need:
        v = inp.get(k)
        if v is None or not Path(v).exists():
            missing.append(k)
    return missing


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Normalized cross-correlation of two volumes over a shared mask."""
    x = a[mask].astype(np.float64); y = b[mask].astype(np.float64)
    if x.size < 100 or x.std() < 1e-6 or y.std() < 1e-6:
        return 0.0
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    return float(np.clip((x * y).mean(), -1.0, 1.0))


def ants_register_and_warp(inp: dict, sdir: Path, log: Path, threads: int,
                           reg_mode: str = "brain") -> dict:
    """ANTs SyN MNI->T1, warp AAL3-166 + Schaefer into T1 (genericLabel). Cached.

    reg_mode: "brain" registers skull-stripped MNI<->T1 brain; "fullhead" registers
    the full MNI152 head <-> full corrected T1 (more robust when brain extraction is
    imperfect, which was the failure mode on ~half of subjects).

    Returns a registration-quality NCC (diagnostic only; the keep-best gate uses the
    candidate connectome itself to detect failed warps).
    """
    import ants  # heavy import, done inside worker
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    aal_t1 = sdir / "aal3_166_t1.nii.gz"
    sch_t1 = sdir / "schaefer200_t1.nii.gz"
    ncc_cache = sdir / "reg_ncc.txt"
    if aal_t1.exists() and sch_t1.exists() and ncc_cache.exists():
        return {"aal_t1": aal_t1, "sch_t1": sch_t1, "cached": True,
                "reg_ncc": float(ncc_cache.read_text().strip() or 0.0)}
    if reg_mode == "fullhead" and inp.get("t1_full") and Path(inp["t1_full"]).exists():
        fixed_path, moving_path = inp["t1_full"], MNI_FULL
    else:
        fixed_path, moving_path = inp["t1_brain"], MNI_BRAIN
    with open(log, "a") as h:
        h.write(f"[{utc()}] ANTs SyNRA register MNI->T1 mode={reg_mode} "
                f"(fixed={Path(fixed_path).name}, moving={Path(moving_path).name})\n")
    fixed = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))
    # SyNRA = Rigid+Affine+SyN; robust default. Affine init keeps the nonlinear
    # stage from wandering, which was the failure mode on some subjects.
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyNRA")
    fwd = reg["fwdtransforms"]  # maps moving(MNI)->fixed(T1)
    saved = []
    for i, t in enumerate(fwd):
        ext = ".nii.gz" if t.endswith(".nii.gz") else Path(t).suffix
        dst = sdir / f"ants_fwd_{i}{ext}"
        shutil.copy(t, dst)
        saved.append(str(dst))
    (sdir / "ants_fwd_transforms.json").write_text(json.dumps(saved))
    # registration QC: warp the MNI brain into T1 and correlate with the T1 brain
    warped_mni = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=fwd,
                                       interpolator="linear")
    fb = fixed.numpy(); wb = warped_mni.numpy()
    reg_ncc = _ncc(fb, wb, fb > 0)
    ncc_cache.write_text(f"{reg_ncc:.4f}\n")
    for atlas_mni, out in ((AAL3_166_MNI, aal_t1), (SCHAEFER_MNI, sch_t1)):
        amov = ants.image_read(str(atlas_mni))
        warped = ants.apply_transforms(fixed=fixed, moving=amov, transformlist=fwd,
                                       interpolator="genericLabel")
        ants.image_write(warped, str(out))
    return {"aal_t1": aal_t1, "sch_t1": sch_t1, "cached": False, "reg_ncc": reg_ncc}


def parc_to_dwi(parc_t1: Path, out_dwi: Path, t12b0_mrtrix: Path, log: Path):
    """Reposition the 1 mm T1-space parc into DWI world space (no crushing to b0 grid)."""
    run([MRBIN / "mrconvert", parc_t1, out_dwi.with_suffix(".step.mif"),
         "-datatype", "uint32", "-force"], log)
    run([MRBIN / "mrtransform", out_dwi.with_suffix(".step.mif"),
         "-linear", t12b0_mrtrix, "-interp", "nearest", "-datatype", "uint32",
         out_dwi, "-force"], log)
    out_dwi.with_suffix(".step.mif").unlink(missing_ok=True)


def rebuild_from_eddy(inp: dict, sdir: Path, log: Path) -> bool:
    """LANE 1 FIX (FOD): rebuild the FOD + post-eddy b0 from the eddy-preprocessed DWI, in one
    consistent space. The cached fod/<sid>/wmfod_final.mif is on some subjects weakly scaled
    (FOD amplitudes ~3x low), which starves tckgen under the 0.06 cutoff -> sparse tracks -> low
    density even when registration is fine (verified 005: cached 0.284 -> rebuilt 0.759). Also
    extracts the post-eddy mean b0 (the cached biascorr b0 is pre-eddy) as the registration target.
    On success points inp['wmfod'] at the rebuilt FOD, sets inp['b0_post'], writes sdir/mask_fod.mif.
    """
    sid = inp["sid"]
    eddy = DERIV / "eddy" / f"{sid}_preproc.mif"
    resp = DERIV / "fod" / sid / "wm_csd.txt"
    if not eddy.exists():
        return False
    fod = sdir / "wmfod_rebuilt.mif"
    mask = sdir / "mask_fod.mif"
    b0 = sdir / "b0_post.nii.gz"
    # Some eddy mifs lost their gradient table -> dwi2mask/dwi2fod fail ("no valid diffusion
    # gradient table"). Restore it from the raw mif_dwi bvec/bval (valid; pre-eddy rotation
    # error is negligible) so the subject is recovered instead of crashing.
    grad = subprocess.run([str(MRBIN / "mrinfo"), str(eddy), "-dwgrad"],
                          capture_output=True, text=True).stdout.strip()
    if not grad:
        bvec = DERIV / "mif_dwi" / f"{sid}.bvec"; bval = DERIV / "mif_dwi" / f"{sid}.bval"
        if not (bvec.exists() and bval.exists()):
            return False
        fixed = sdir / "eddy_gradfix.mif"
        if not fixed.exists():
            run([MRBIN / "mrconvert", eddy, fixed, "-fslgrad", str(bvec), str(bval), "-force"], log)
        eddy = fixed
    if not fod.exists():
        run([MRBIN / "dwi2mask", eddy, mask, "-force"], log)
        # dwi2mask occasionally yields a DEGENERATE mask even at rc=0: either EMPTY (098_S_6601: 0
        # voxels) OR a tiny speck of a few hundred voxels (098_S_6658: 1070 vox, while the healthy
        # original is 730917). Both make the FOD untrackable -> tckgen 0-tracks ("image is empty" ROI
        # for the 0 case; near-zero seeds for the tiny case). A real brain mask is ~4e5-2e6 voxels,
        # so anything <1e5 is a failed mask. Fall back to the original production mask (regridded onto
        # the eddy grid to be grid-safe) whenever it is materially healthier, so the subject is
        # recovered, not lost as "0-track / data-limited".
        def _vox(m):
            s = subprocess.run([str(MRBIN / "mrstats"), str(m), "-output", "count", "-ignorezero"],
                               capture_output=True, text=True).stdout.strip()
            try:
                return int(s)
            except (TypeError, ValueError):
                return 0
        nzc = _vox(mask)
        if nzc < 100000:
            orig_mask = DERIV / "fod" / sid / "mask_fod.mif"
            if orig_mask.exists():
                ovc = _vox(orig_mask)
                if ovc > max(nzc * 2, 100000):
                    run([MRBIN / "mrgrid", orig_mask, "regrid", "-template", eddy,
                         "-interp", "nearest", "-datatype", "bit", mask, "-force"], log)
                    with open(log, "a", encoding="utf-8", errors="replace") as _h:
                        _h.write(f"[{utc()}] [mask-fallback] {sid}: fresh dwi2mask={nzc} vox "
                                 f"-> orig prod mask={ovc} vox (regridded onto eddy)\n\n")
        # DONOR-RESPONSE lane: when per-subject dwi2response collapses (degenerate/near-zero WM
        # response -> weak FOD -> sparse tracks), use a robust GROUP-AVERAGE response instead.
        donor = os.environ.get("DONOR_RESPONSE", "")
        # L7 response-regen: regenerate the WM response if the cached one is missing (0-track FOD),
        # or force it in the recovery lane (FORCE_RESPONSE=1) when the cached response is suspect.
        if donor and Path(donor).exists():
            resp = Path(donor)
            with open(log, "a", encoding="utf-8", errors="replace") as _h:
                _h.write(f"[{utc()}] [donor-response] {sid}: using group response {donor}\n")
        elif (not resp.exists()) or os.environ.get("FORCE_RESPONSE") == "1":
            rresp = sdir / "wm_response.txt"
            if not rresp.exists():
                run([MRBIN / "dwi2response", "tournier", eddy, rresp, "-mask", mask, "-force"], log)
            resp = rresp
        run([MRBIN / "dwi2fod", "csd", eddy, resp, fod, "-mask", mask, "-force"], log)
    if not b0.exists():
        b0s = sdir / "_b0s.mif"
        run([MRBIN / "dwiextract", "-bzero", eddy, b0s, "-force"], log)
        run([MRBIN / "mrmath", b0s, "mean", b0, "-axis", "3", "-force"], log)
        b0s.unlink(missing_ok=True)
    inp["wmfod"] = fod
    inp["b0_post"] = b0
    return True


def fresh_t12b0(inp: dict, sdir: Path, log: Path) -> Path | None:
    """LANE 1 FIX: recompute T1->b0 with a fresh rigid flirt (normmi) instead of the
    cached BBR transform. The cached BBR (fod/<sid>/t12b0_bbr_mrtrix.txt) is mis-registered
    on a large fraction of subjects (off by ~cm), which lands the atlas outside the brain in
    DWI space and yields empty connectomes. A fresh rigid flirt recovers them
    (verified on 005: salvage-with-cached-BBR 0.0017 -> fresh-flirt 0.759; DISCO added nothing).
    Returns the new mrtrix transform path, or None if a clean b0 reference is unavailable
    (the subject then keeps the cached transform and, if still empty, drops to a later lane).
    """
    sid = inp["sid"]
    # prefer the post-eddy b0 + rebuilt mask (consistent with the rebuilt FOD); else cached
    b0_src = inp.get("b0_post") or (DERIV / "biascorr_1" / f"{sid}_b0.nii.gz")
    mask_fod = (sdir / "mask_fod.mif") if (sdir / "mask_fod.mif").exists() \
        else (Path(inp["wmfod"]).parent / "mask_fod.mif")
    t1b = inp.get("t1_brain")
    if not (b0_src.exists() and mask_fod.exists() and t1b and Path(t1b).exists()):
        return None
    try:  # b0 reference grid must match the FOD/wmfod space, else the transform is meaningless
        size = lambda p: subprocess.run([str(MRBIN/"mrinfo"), str(p), "-size"],
                                        capture_output=True, text=True).stdout.split()[:3]
        if size(b0_src) != size(inp["wmfod"]):
            return None
    except Exception:
        return None
    b0_brain = sdir / "b0_brain.nii.gz"
    mask_nii = sdir / "mask_fod.nii.gz"
    mat = sdir / "t12b0_fresh.mat"
    out = sdir / "t12b0_fresh_mrtrix.txt"
    if out.exists():
        return out
    run([MRBIN/"mrconvert", mask_fod, mask_nii, "-force"], log)
    run([FSLBIN/"fslmaths", b0_src, "-mas", mask_nii, b0_brain], log)
    # Constrain the search to +/-40 deg: T1 and b0 are the same subject in roughly the same
    # scanner frame, so a large rotation is never correct. The unconstrained +/-90 default could
    # settle in a flipped/axis-permuted optimum (seen on 003_S_5165), silently emptying the matrix.
    run([FSLBIN/"flirt", "-in", t1b, "-ref", b0_brain, "-omat", mat,
         "-dof", "6", "-cost", "normmi", "-searchcost", "normmi",
         "-searchrx", "-40", "40", "-searchry", "-40", "40", "-searchrz", "-40", "40"], log)
    run([MRBIN/"transformconvert", mat, t1b, b0_brain, "flirt_import", out, "-force"], log)
    return out


# observed: a completed 10M .tck is ~5 GB -> ~500 bytes per selected streamline.
_BYTES_PER_STREAMLINE = 500.0
_TARGET_STREAMLINES = 10_000_000


def gen_tracks_10m(inp: dict, sdir: Path, log: Path, threads: int) -> dict:
    """Probe-and-defer tckgen.

    Run tckgen toward the 10M -select target, but watch its real-time rate (via
    .tck file growth). After a short probe window, extrapolate time-to-10M:
      * projects to finish within budget -> let it run (hard cap as backstop);
      * projects slow (atrophied / low-efficiency brain) -> SIGINT early and DEFER
        the subject (return {"deferred": True}) so its worker is freed immediately
        for a quicker subject. Deferred subjects are handled in a separate slow lane.
    This finishes the quick subjects first instead of stalling workers for hours.
    """
    # TCKGEN_SELECT lets a fast triage pass (e.g. 3M) classify recovery cheaply before
    # committing the full 10M production tracking only to the subjects that recover.
    target_select = int(os.environ.get("TCKGEN_SELECT", str(_TARGET_STREAMLINES)))
    tck = sdir / "tracks_10M.tck"
    sift = sdir / "sift2_10M.txt"
    if tck.exists() and sift.exists():
        return {"tck": tck, "sift": sift, "cached": True}
    probe_s = int(os.environ.get("TCKGEN_PROBE_S", "480"))         # 8 min: estimate the rate
    budget_s = int(os.environ.get("TCKGEN_BUDGET_S", "99999999"))  # default: NO defer (anisotropy
                                                                   # fix means slow subjects are rare)
    hard_cap_s = int(os.environ.get("TCKGEN_CAP_S", "6000"))       # 100 min: absolute backstop
    target_bytes = target_select * _BYTES_PER_STREAMLINE
    # -seeds caps total seed attempts so atrophied/untrackable brains exit CLEANLY with
    # a partial instead of running until MRtrix's internal limit and aborting (rc=-6 SIGABRT).
    # Good subjects reach the 10M -select target well before this cap; pathological ones stop
    # with whatever they got (the keep-best gate then rejects the too-sparse ones).
    seed_cap = os.environ.get("TCKGEN_SEED_CAP", "200000000")
    # Anisotropic acquisitions (thick slices) have unreliable 5TT partial volumes, so ACT
    # rejects ~all streamlines (verified: 0.02% acceptance with ACT vs 96% without on the
    # same FOD). Detect anisotropy and fall back to FOD-based tracking (no ACT, brain-masked,
    # dynamic seeding) for those subjects; keep full ACT+GMWMI+backtrack for clean isotropic data.
    try:
        sp = subprocess.run([str(MRBIN / "mrinfo"), str(inp["wmfod"]), "-spacing"],
                            capture_output=True, text=True, timeout=20).stdout.split()
        sx, sy, sz = float(sp[0]), float(sp[1]), float(sp[2])
        anisotropic = sz > 1.4 * max(sx, sy)
    except Exception:
        anisotropic = False
    cmd = [str(MRBIN / "tckgen"), str(inp["wmfod"]), str(tck),
           "-select", str(target_select), "-seeds", str(seed_cap), "-cutoff", "0.06",
           "-minlength", "10", "-maxlength", "250", "-nthreads", str(threads), "-force"]
    # ACT is broken even on CLEAN ISOTROPIC data: the 5TT/gmwmi seed is misaligned with the atlas
    # in DWI space, so streamlines generate but don't assign to nodes -> near-empty connectome
    # (verified: 116_S_6543 = 0.003 with ACT vs 0.768 with noACT, identical FOD/atlas). Cohort-wide
    # the ACT path medians 0.19 vs 0.62 for noACT. Until the 5TT alignment is fixed, force the robust
    # FOD-based noACT path for ALL subjects (FORCE_NOACT=1, default). Set FORCE_NOACT=0 to restore ACT.
    use_noact = anisotropic or os.environ.get("FORCE_NOACT", "1") == "1"
    if use_noact:
        cmd += ["-seed_dynamic", str(inp["wmfod"])]
        mask = Path(inp["wmfod"]).parent / "mask_fod.mif"
        if mask.exists():
            cmd += ["-mask", str(mask)]
        track_mode = "noACT_dynamic_" + ("anisotropic" if anisotropic else "isotropic")
    else:
        cmd += ["-act", str(inp["five_tt"]), "-backtrack", "-seed_gmwmi", str(inp["gmwmi"])]
        track_mode = "ACT_gmwmi_isotropic"
    h = open(log, "a", encoding="utf-8", errors="replace")
    h.write(f"[{utc()}] track_mode={track_mode} (vox aniso={anisotropic})\n")
    h.write(f"[{utc()}] $ (probe {probe_s}s / budget {budget_s}s / cap {hard_cap_s}s) {' '.join(cmd)}\n")
    h.flush()
    proc = subprocess.Popen(cmd, stdout=h, stderr=subprocess.STDOUT, env=base_env())
    t0 = time.time(); deferred = False; eta_min = None
    while True:
        try:
            proc.wait(timeout=30)
            break  # tckgen finished on its own (reached 10M)
        except subprocess.TimeoutExpired:
            pass
        el = time.time() - t0
        sz = tck.stat().st_size if tck.exists() else 0
        if el >= probe_s and sz > 0:
            eta = el * target_bytes / sz          # extrapolate to full 10M
            eta_min = eta / 60.0
            if eta > budget_s:                    # projected slow -> defer early
                h.write(f"[{utc()}] DEFER {sdir.name}: proj ETA {eta_min:.0f}min "
                        f"(> {budget_s//60}min budget); {sz/1e9:.2f}GB in {el/60:.1f}min\n")
                proc.send_signal(signal.SIGINT); deferred = True
                try: proc.wait(timeout=60)
                except subprocess.TimeoutExpired: proc.kill()
                break
        if el >= hard_cap_s:                      # passed probe but overran -> cap partial
            h.write(f"[{utc()}] HARD CAP {sdir.name} at {el/60:.0f}min; accept partial {sz/1e9:.2f}GB\n")
            proc.send_signal(signal.SIGINT)
            try: proc.wait(timeout=60)
            except subprocess.TimeoutExpired: proc.kill()
            break
    h.write(f"[{utc()}] tckgen rc={proc.returncode} deferred={deferred} eta_est_min={eta_min}\n\n")
    h.close()
    if deferred:
        # leave the partial .tck for diagnostics; do not build a connectome from it
        return {"deferred": True, "eta_est_min": eta_min}
    # Accept: 0 = reached 10M target; 2 = MRtrix's CLEAN SIGINT exit (our hard-cap stops a
    # slow-but-fine subject and MRtrix finalizes a valid partial .tck, verified usable at
    # ~0.71 density); -2/130/124 = other clean-stop signal variants.
    # Reject: -9 SIGKILL / -6 SIGABRT etc. -> CORRUPT/truncated .tck.
    if proc.returncode not in (0, 2, -2, 124, 130):
        raise RuntimeError(f"tckgen exited abnormally rc={proc.returncode} (likely killed) "
                           f"-> corrupt partial discarded: {tck.name} (see {log})")
    if not tck.exists() or tck.stat().st_size < 5_000_000:
        raise RuntimeError(f"tckgen produced no/too-small track file: {tck.name} (see {log})")
    # SIFT2 also needs ACT; only apply it when we actually tracked with ACT
    sift_cmd = [MRBIN / "tcksift2", tck, inp["wmfod"], sift]
    if track_mode.startswith("ACT"):
        sift_cmd += ["-act", str(inp["five_tt"])]
    sift_cmd += ["-nthreads", str(threads), "-force"]
    run(sift_cmd, log)
    return {"tck": tck, "sift": sift, "cached": False, "deferred": False, "track_mode": track_mode}


def build_connectomes(tck: Path, sift: Path, parc_dwi: Path, prefix: str,
                      cdir: Path, log: Path, threads: int, radial: int = 4) -> dict:
    cdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for suffix, extra, stat_edge, scale in WEIGHT_SPECS:
        csv_out = cdir / f"{prefix}_{suffix}.csv"
        cmd = [MRBIN / "tck2connectome", tck, parc_dwi, csv_out,
               "-symmetric", "-zero_diagonal",
               "-assignment_radial_search", str(radial),
               "-tck_weights_in", sift, "-nthreads", str(threads), "-force"]
        if stat_edge:
            cmd += ["-stat_edge", stat_edge]
        if scale == "invnodevol":
            cmd += ["-scale_invnodevol"]
        elif scale == "length":
            cmd += ["-scale_length"]
        run(cmd, log)
        out[suffix] = csv_out
    return out


def qc_matrix(count_csv: Path, n_nodes: int) -> dict:
    M = np.loadtxt(count_csv, delimiter=",")
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return {"ok": False, "reason": f"bad shape {getattr(M,'shape',None)}"}
    np.fill_diagonal(M, 0)
    n = M.shape[0]
    deg = M.sum(axis=1)
    zero_rows = int((deg == 0).sum())
    poss = n_nodes * (n_nodes - 1) / 2
    conn = int((np.triu(M, 1) > 0).sum())
    return {"ok": True, "n": n, "zero_rows": zero_rows, "density": conn / poss}


AAL3_GAPS = {35, 36, 81, 82}
REG_NCC_MIN = 0.30  # below this the ANTs warp is treated as failed


def baseline_qc(sid: str) -> dict:
    """QC of the subject's EXISTING production AAL matrix (170-node, with gaps),
    reported on the 166 valid nodes so it is comparable to the 166-node candidate."""
    p = DERIV / "connectomes" / f"SC_AAL_{sid}_count.csv"
    if not p.exists():
        return {"ok": False, "reason": "no production matrix"}
    try:
        M = np.loadtxt(p, delimiter=",")
        np.fill_diagonal(M, 0)
        n = M.shape[0]
        deg = M.sum(axis=1)
        zero_rows = int(sum(1 for i in range(n) if deg[i] == 0 and (i + 1) not in AAL3_GAPS))
        conn = int((np.triu(M, 1) > 0).sum())
        return {"ok": True, "zero_rows": zero_rows, "density": conn / (166 * 165 / 2)}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def gate_decision(base: dict, cand: dict, reg_ncc: float) -> dict:
    """Keep-best gate: the candidate connectome IS the ground truth of whether the
    registration worked. A misplaced atlas yields MORE zero-rows, so requiring the
    candidate to strictly reduce zero-rows (without losing density) cleanly rejects
    failed registrations. Otherwise the subject keeps its current matrix (no
    regressions, ever). reg_ncc is kept only as a logged diagnostic."""
    if not cand.get("ok"):
        return {"promote": False, "reason": "candidate QC failed"}
    if not base.get("ok"):
        # no usable baseline (e.g. empty production) -> accept a non-empty candidate
        if cand.get("zero_rows", 166) < 120:
            return {"promote": True, "reason": "no usable baseline; candidate non-empty"}
        return {"promote": False, "reason": "no baseline and candidate still empty"}
    better_zero = cand["zero_rows"] < base["zero_rows"]
    not_worse_density = cand["density"] >= base["density"] - 1e-6
    if better_zero and not_worse_density:
        return {"promote": True,
                "reason": (f"win: z {base['zero_rows']}->{cand['zero_rows']}, "
                           f"d {base['density']:.3f}->{cand['density']:.3f}")}
    return {"promote": False,
            "reason": (f"not better (cand z={cand['zero_rows']} d={cand['density']:.3f} vs "
                       f"base z={base['zero_rows']} d={base['density']:.3f})")}


def process_subject(sid: str, run_root: str, mode: str, threads: int, radial: int,
                    reg_mode: str = "brain") -> dict:
    t0 = time.time()
    run_root = Path(run_root)
    sdir = run_root / sid
    sdir.mkdir(parents=True, exist_ok=True)
    log = sdir / "sota.log"
    res = {"sid": sid, "mode": mode, "status": "fail"}
    # Resume guard: skip subjects already completed with a genuinely good connectome,
    # so a re-run only reprocesses failed / corrupt / deferred / never-started subjects.
    # (RESUME_MIN_DENSITY=0 disables; corrupt empties have density 0 and WILL be redone.)
    prev_f = sdir / "result.json"
    if prev_f.exists():
        try:
            prev = json.loads(prev_f.read_text())
            thr = float(os.environ.get("RESUME_MIN_DENSITY", "0.6"))
            if prev.get("status") == "ok" and (prev.get("aal_qc") or {}).get("density", 0) >= thr:
                prev["resumed_skip"] = True
                return prev
        except Exception:
            pass
    try:
        inp = resolve_inputs(sid)
        missing = check_inputs(inp, mode)
        if missing:
            res["reason"] = f"missing inputs: {missing}"
            return res
        # Stage A0 (LANE 1 FOD fix): rebuild FOD + post-eddy b0 in one consistent space.
        # Also rebuild if the cached FOD is EMPTY/all-zero, else tckgen SIGSEGVs at "segmenting FODs".
        _cached_empty = False
        try:
            _fm = subprocess.run([str(MRBIN / "mrstats"), str(inp["wmfod"]), "-output", "max"],
                                 capture_output=True, text=True, timeout=240).stdout.strip()
            _cached_empty = (float(_fm) == 0.0)
        except Exception:
            pass
        if os.environ.get("REBUILD_FOD", "1") == "1" or _cached_empty:
            res["fod"] = "rebuilt" if rebuild_from_eddy(inp, sdir, log) else "cached_fallback"
        # Stage A: ANTs SyN + warp atlases into T1
        warped = ants_register_and_warp(inp, sdir, log, threads, reg_mode)
        # Stage C: tracks FIRST (tracks depend on the FOD/5TT, not on the T1->b0 transform)
        if mode == "existing_tracks":
            tck, sift = inp["tracks_3m"], inp["sift_3m"]
        else:
            tk = gen_tracks_10m(inp, sdir, log, threads)
            if tk.get("deferred"):
                # projected too slow to reach 10M in budget -> hand to the slow lane
                res["status"] = "deferred_slow"
                res["eta_est_min"] = tk.get("eta_est_min")
                res["reason"] = "tckgen projected ETA exceeds budget; deferred"
                res["seconds"] = round(time.time() - t0, 1)
                (sdir / "result.json").write_text(json.dumps(res, indent=2))
                return res
            tck, sift = tk["tck"], tk["sift"]
            res["track_mode"] = tk.get("track_mode")
        # Stage A2/B: KEEP-BEST over T1->b0 registration. Place the AAL3 atlas via the cached BBR
        # and (if enabled) a fresh flirt, score each by AAL3 count-density on the SAME tracks, and
        # keep whichever is better -> re-registration is applied only when it actually helps (a good
        # cached registration is never replaced by a worse fresh one).
        def _score(t12b0, tag):
            ad = sdir / f"aal3_166_dwi_{tag}.mif"
            parc_to_dwi(warped["aal_t1"], ad, t12b0, log)
            cc = sdir / f"_score_{tag}.csv"
            run([MRBIN / "tck2connectome", tck, ad, cc, "-tck_weights_in", sift, "-symmetric",
                 "-zero_diagonal", "-assignment_radial_search", str(radial), "-force"], log)
            m = np.loadtxt(cc, delimiter=","); iu = np.triu_indices(m.shape[0], 1)
            return float(np.mean(m[iu] > 0)), ad
        # Try the CACHED BBR first; only re-register (fresh flirt) if the cached one is poor.
        # This avoids re-registering subjects whose registration is already fine (and the flip risk),
        # while still rescuing the genuinely mis-registered ones.
        d_cached, aal_dwi = _score(inp["t12b0_mrtrix"], "cached")
        best_d, best_tag, best_t = d_cached, "cached", inp["t12b0_mrtrix"]
        res["reg_choice"] = {"cached": round(d_cached, 3)}
        rereg_thresh = float(os.environ.get("REREG_THRESH", "0.5"))
        if d_cached < rereg_thresh and os.environ.get("FRESH_T12B0", "1") == "1":
            fresh = fresh_t12b0(inp, sdir, log)
            if fresh is not None:
                d_fresh, ad_fresh = _score(fresh, "fresh")
                res["reg_choice"]["fresh"] = round(d_fresh, 3)
                if d_fresh > best_d:
                    best_d, best_tag, best_t, aal_dwi = d_fresh, "fresh", fresh, ad_fresh
        inp["t12b0_mrtrix"] = best_t
        res["t12b0"] = best_tag
        # Schaefer atlas via the chosen transform
        sch_dwi = sdir / "schaefer200_dwi.mif"
        parc_to_dwi(warped["sch_t1"], sch_dwi, best_t, log)
        # label survival in DWI space
        aal_img = nib.load(str(_mif_to_nii(aal_dwi, log)))
        u = np.unique(aal_img.get_fdata().astype(int)); u = u[u > 0]
        res["aal_label_survival"] = int(len(u))
        # Stage D/E: connectomes + QC for both atlases
        cdir = sdir / "connectomes"
        aal_cc = build_connectomes(tck, sift, aal_dwi, f"SC_AAL3cc_{sid}", cdir, log, threads, radial)
        sch_cc = build_connectomes(tck, sift, sch_dwi, f"SC_Schaefer200_{sid}", cdir, log, threads, radial)
        res["aal_qc"] = qc_matrix(aal_cc["count"], AAL3_NODES)
        res["schaefer_qc"] = qc_matrix(sch_cc["count"], SCHAEFER_NODES)
        res["reg_ncc"] = round(float(warped.get("reg_ncc", 0.0)), 4)
        # keep-best gate vs existing production matrix
        res["baseline_qc"] = baseline_qc(sid)
        res["decision"] = gate_decision(res["baseline_qc"], res["aal_qc"], res["reg_ncc"])
        res["status"] = "ok"
    except Exception as exc:
        res["reason"] = f"{type(exc).__name__}: {exc}"
        with open(log, "a") as h:
            h.write(traceback.format_exc())
    res["seconds"] = round(time.time() - t0, 1)
    (sdir / "result.json").write_text(json.dumps(res, indent=2))
    return res


def _mif_to_nii(mif: Path, log: Path) -> Path:
    nii = mif.with_suffix(".nii.gz")
    if not nii.exists():
        run([MRBIN / "mrconvert", mif, nii, "-force"], log)
    return nii


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--mode", choices=["existing_tracks", "full_10m"], default="existing_tracks")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--radial", type=int, default=4)
    ap.add_argument("--reg-mode", choices=["brain", "fullhead"], default="brain")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[{utc()}] SOTA route start | mode={args.mode} | n={len(args.subjects)} | "
          f"workers={args.workers} x {args.threads} threads | run_root={run_root}", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_subject, sid, str(run_root), args.mode, args.threads,
                          args.radial, args.reg_mode): sid
                for sid in args.subjects}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            aq = r.get("aal_qc", {}) or {}
            bq = r.get("baseline_qc", {}) or {}
            dec = r.get("decision", {}) or {}
            cz = aq.get("zero_rows", "?"); cd = aq.get("density")
            cd = round(cd, 3) if isinstance(cd, float) else cd
            bz = bq.get("zero_rows", "?"); bd = bq.get("density")
            bd = round(bd, 3) if isinstance(bd, float) else bd
            flag = "PROMOTE" if dec.get("promote") else "keep-base"
            print(f"[{utc()}] {r['sid']:24s} {r['status']:4s} ncc={r.get('reg_ncc','?')} "
                  f"surv={r.get('aal_label_survival','?')}/166 "
                  f"cand[z={cz} d={cd}] base[z={bz} d={bd}] -> {flag} "
                  f"{r.get('reason','') or dec.get('reason','')} ({r.get('seconds','?')}s)", flush=True)

    summ = run_root / f"_summary_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    summ.write_text(json.dumps(results, indent=2))
    ok = [r for r in results if r["status"] == "ok"]
    print(f"[{utc()}] DONE ok={len(ok)}/{len(results)} -> {summ}", flush=True)


if __name__ == "__main__":
    main()
