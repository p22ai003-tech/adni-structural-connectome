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


def ants_register_and_warp(inp: dict, sdir: Path, log: Path, threads: int) -> dict:
    """ANTs SyN MNI->T1, warp AAL3-166 + Schaefer into T1 (genericLabel). Cached."""
    import ants  # heavy import, done inside worker
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    aal_t1 = sdir / "aal3_166_t1.nii.gz"
    sch_t1 = sdir / "schaefer200_t1.nii.gz"
    xfm_cache = sdir / "ants_fwd_transforms.json"
    if aal_t1.exists() and sch_t1.exists():
        return {"aal_t1": aal_t1, "sch_t1": sch_t1, "cached": True}
    with open(log, "a") as h:
        h.write(f"[{utc()}] ANTs SyNRA register MNI->T1 (fixed={inp['t1_brain'].name})\n")
    fixed = ants.image_read(str(inp["t1_brain"]))
    moving = ants.image_read(str(MNI_BRAIN))
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyNRA")
    fwd = reg["fwdtransforms"]  # maps moving(MNI)->fixed(T1)
    # persist transforms into the subject dir for reuse
    saved = []
    for i, t in enumerate(fwd):
        ext = ".nii.gz" if t.endswith(".nii.gz") else Path(t).suffix
        dst = sdir / f"ants_fwd_{i}{ext}"
        shutil.copy(t, dst)
        saved.append(str(dst))
    xfm_cache.write_text(json.dumps(saved))
    for atlas_mni, out in ((AAL3_166_MNI, aal_t1), (SCHAEFER_MNI, sch_t1)):
        amov = ants.image_read(str(atlas_mni))
        warped = ants.apply_transforms(fixed=fixed, moving=amov, transformlist=fwd,
                                       interpolator="genericLabel")
        ants.image_write(warped, str(out))
    return {"aal_t1": aal_t1, "sch_t1": sch_t1, "cached": False}


def parc_to_dwi(parc_t1: Path, out_dwi: Path, t12b0_mrtrix: Path, log: Path):
    """Reposition the 1 mm T1-space parc into DWI world space (no crushing to b0 grid)."""
    run([MRBIN / "mrconvert", parc_t1, out_dwi.with_suffix(".step.mif"),
         "-datatype", "uint32", "-force"], log)
    run([MRBIN / "mrtransform", out_dwi.with_suffix(".step.mif"),
         "-linear", t12b0_mrtrix, "-interp", "nearest", "-datatype", "uint32",
         out_dwi, "-force"], log)
    out_dwi.with_suffix(".step.mif").unlink(missing_ok=True)


def gen_tracks_10m(inp: dict, sdir: Path, log: Path, threads: int) -> dict:
    tck = sdir / "tracks_10M.tck"
    sift = sdir / "sift2_10M.txt"
    if tck.exists() and sift.exists():
        return {"tck": tck, "sift": sift, "cached": True}
    run([MRBIN / "tckgen", inp["wmfod"], tck,
         "-act", inp["five_tt"], "-backtrack", "-seed_gmwmi", inp["gmwmi"],
         "-select", "10000000", "-cutoff", "0.06", "-minlength", "10", "-maxlength", "250",
         "-nthreads", str(threads), "-force"], log)
    run([MRBIN / "tcksift2", tck, inp["wmfod"], sift,
         "-act", inp["five_tt"], "-nthreads", str(threads), "-force"], log)
    return {"tck": tck, "sift": sift, "cached": False}


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


def process_subject(sid: str, run_root: str, mode: str, threads: int, radial: int) -> dict:
    t0 = time.time()
    run_root = Path(run_root)
    sdir = run_root / sid
    sdir.mkdir(parents=True, exist_ok=True)
    log = sdir / "sota.log"
    res = {"sid": sid, "mode": mode, "status": "fail"}
    try:
        inp = resolve_inputs(sid)
        missing = check_inputs(inp, mode)
        if missing:
            res["reason"] = f"missing inputs: {missing}"
            return res
        # Stage A: ANTs SyN + warp atlases into T1
        warped = ants_register_and_warp(inp, sdir, log, threads)
        # Stage B: atlases -> DWI world space (1 mm preserved)
        aal_dwi = sdir / "aal3_166_dwi.mif"
        sch_dwi = sdir / "schaefer200_dwi.mif"
        parc_to_dwi(warped["aal_t1"], aal_dwi, inp["t12b0_mrtrix"], log)
        parc_to_dwi(warped["sch_t1"], sch_dwi, inp["t12b0_mrtrix"], log)
        # label survival in DWI space
        aal_img = nib.load(str(_mif_to_nii(aal_dwi, log)))
        u = np.unique(aal_img.get_fdata().astype(int)); u = u[u > 0]
        res["aal_label_survival"] = int(len(u))
        # Stage C: tracks
        if mode == "existing_tracks":
            tck, sift = inp["tracks_3m"], inp["sift_3m"]
        else:
            tk = gen_tracks_10m(inp, sdir, log, threads)
            tck, sift = tk["tck"], tk["sift"]
        # Stage D/E: connectomes + QC for both atlases
        cdir = sdir / "connectomes"
        aal_cc = build_connectomes(tck, sift, aal_dwi, f"SC_AAL3cc_{sid}", cdir, log, threads, radial)
        sch_cc = build_connectomes(tck, sift, sch_dwi, f"SC_Schaefer200_{sid}", cdir, log, threads, radial)
        res["aal_qc"] = qc_matrix(aal_cc["count"], AAL3_NODES)
        res["schaefer_qc"] = qc_matrix(sch_cc["count"], SCHAEFER_NODES)
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
    args = ap.parse_args()

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[{utc()}] SOTA route start | mode={args.mode} | n={len(args.subjects)} | "
          f"workers={args.workers} x {args.threads} threads | run_root={run_root}", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_subject, sid, str(run_root), args.mode, args.threads, args.radial): sid
                for sid in args.subjects}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            aq = r.get("aal_qc", {}) or {}
            print(f"[{utc()}] {r['sid']:24s} {r['status']:4s} "
                  f"surv={r.get('aal_label_survival','?')}/166 "
                  f"zero_rows={aq.get('zero_rows','?')} dens={aq.get('density','?') if not isinstance(aq.get('density'),float) else round(aq['density'],3)} "
                  f"{r.get('reason','')} ({r.get('seconds','?')}s)", flush=True)

    summ = run_root / f"_summary_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    summ.write_text(json.dumps(results, indent=2))
    ok = [r for r in results if r["status"] == "ok"]
    print(f"[{utc()}] DONE ok={len(ok)}/{len(results)} -> {summ}", flush=True)


if __name__ == "__main__":
    main()

