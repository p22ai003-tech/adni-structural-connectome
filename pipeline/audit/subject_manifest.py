#!/usr/bin/env python
"""Master per-subject manifest — the single source of truth for the whole project.

One row per subject (all 648). Density is computed DIRECTLY from the production SC_AAL166 count.csv
(main dir or _midband/_lowband_archive), so band/count never drift from what's on disk. Records
presence + path at EVERY pipeline stage (raw -> S7), plus EC2/S3 flags. Drives the band symlink views
(build_quality_views.py), the stage funnel (stage_funnel.py), and the dashboard table.

Output: /data/derivatives/qc/sc_matrix_qc/subject_manifest.csv
Run after any connectome change.
"""
import csv, glob, os, re, json, subprocess
import numpy as np
import concurrent.futures as cf

D = "/data/derivatives"
CONN = f"{D}/connectomes"
MR = "/home/ec2-user/mrtrix3/bin"
OUT = f"{D}/qc/sc_matrix_qc/subject_manifest.csv"
GROUPCSV = f"{D}/sc_forge_v1/group_qc/current_connectome_subject_availability_with_group.csv"

G2 = {}
for r in csv.DictReader(open(GROUPCSV)):
    g = r.get("group", "?"); G2[r["sid"]] = "MCI" if g == "SMC" else g

def base(sid):  # site_S_num  (T1 image-id differs from DWI image-id)
    m = re.match(r"(\d{3}_S_\d+)", sid); return m.group(1) if m else sid

def band(d):
    if d is None or d <= 0: return "none"
    return "good" if d >= 0.6 else ("mid" if d >= 0.4 else "low")

def density(p):
    try:
        m = np.loadtxt(p, delimiter=",");
        if m.ndim != 2 or m.shape[0] != m.shape[1]: return None
        n = m.shape[0]; off = m[~np.eye(n, dtype=bool)]; return round(float((off > 0).sum() / off.size), 4)
    except Exception:
        return None

def find_conn(sid):  # count.csv: main (good) or band subfolders
    for sub in ("", "/_midband", "/_lowband_archive"):
        p = f"{CONN}{sub}/SC_AAL166_{sid}_count.csv"
        if os.path.exists(p): return p, ("good_main" if sub == "" else sub.strip("/"))
    return None, None

def first(*paths):
    for p in paths:
        g = glob.glob(p)
        if g: return g[0]
    return ""

# ---- S3 inventory (one listing per prefix) ----
def s3_sets():
    has = {}
    for pref in ("tracks/", "connectomes/"):
        out = subprocess.run(["aws", "s3", "ls", f"s3://sabeesh/exp/{pref}", "--recursive"],
                             capture_output=True, text=True).stdout
        for ln in out.splitlines():
            key = ln.split()[-1] if ln.split() else ""
            m = re.search(r"\d{3}_S_\d+_I\d+", key)
            if not m: continue
            s = has.setdefault(m.group(0), set()); fn = key.split("/")[-1]
            if fn.endswith(".tck"): s.add("tck")
            elif fn.endswith(".tsf"): s.add("tsf")
            elif "sift" in fn: s.add("sift")
            elif fn.endswith(".csv") and "SC_AAL166_" in fn: s.add("csv")
    return has

print("listing S3 ...")
S3 = s3_sets()

# recovery-promoted (radial 2) vs production (radial 4)
MANI = f"{D}/qc/sc_matrix_qc/claude_repaired_manifest.csv"
RECOVERED = {r["sid"] for r in csv.DictReader(open(MANI))} if os.path.exists(MANI) else set()

def mrcount(p):
    try:
        return int(float(subprocess.run([f"{MR}/mrstats", p, "-output", "count", "-mask", p],
                   capture_output=True, text=True, timeout=60).stdout.strip()))
    except Exception:
        return None

def best_result(sid):
    best = None; bd = -1
    for rj in glob.glob(f"{D}/qc/sc_matrix_qc/*/{sid}/result.json"):
        try: r = json.load(open(rj))
        except Exception: continue
        d = (r.get("aal_qc") or {}).get("density") or 0
        if d > bd: bd = d; best = r
    return best

# stage -> (path-or-glob template using {sid}/{base}); "" means absent
STAGES = [
    ("raw_dwi",      lambda s: (f"/data/Images/dti/{base(s)}" if os.path.isdir(f"/data/Images/dti/{base(s)}") else "")),
    ("raw_t1",       lambda s: (f"/data/Images/mri/{base(s)}" if os.path.isdir(f"/data/Images/mri/{base(s)}") else "")),
    ("mif_dwi",      lambda s: first(f"{D}/mif_dwi/{s}.mif")),
    ("mif_denoised", lambda s: first(f"{D}/mif_denoised/{s}_den.mif")),
    ("mif_unringed", lambda s: first(f"{D}/mif_unringed/{s}_den_unr.mif")),
    ("eddy",         lambda s: first(f"{D}/eddy/{s}_preproc.mif", f"{D}/eddy/{s}_eddy.mif")),
    ("biascorr",     lambda s: first(f"{D}/biascorr_1/{s}_unbiased.mif", f"{D}/biascorr_1/{s}_b0.nii.gz")),
    ("t1",           lambda s: first(f"{D}/t1_anat/T1_ss_{base(s)}_*.nii.gz")),
    ("bbr",          lambda s: first(f"{D}/dwi_t1_bbr/{s}_t12b0_bbr_mrtrix.txt", f"{D}/dwi_t1_bbr/{s}_*.mat")),
    ("fod",          lambda s: first(f"{D}/fod/{s}/wmfod*.mif")),
    ("dti",          lambda s: first(f"{D}/dti/{s}/dt.mif")),
    ("parc",         lambda s: first(f"{D}/parc/{s}/AAL_b0.nii.gz")),
    ("tck",          lambda s: first(f"{D}/tracks/{s}/tracks_final_3000k.tck")),
    ("sift",         lambda s: first(f"{D}/tracks/{s}/sift_weights.txt")),
    ("tsf",          lambda s: first(f"{D}/tracks/{s}/fa_mean.tsf")),
]

def row(sid):
    g = G2.get(sid, "?")
    cnt, loc = find_conn(sid)
    d = density(cnt) if cnt else None
    nw = len(glob.glob(f"{CONN}{('/'+loc) if loc and loc!='good_main' else ''}/SC_AAL166_{sid}_*.csv")) if cnt else 0
    # radial: recovery-promoted=2, production-with-prod-track=4, else ?
    radial = 2 if sid in RECOVERED else (4 if os.path.exists(f"{D}/tracks/{sid}/tracks_final_3000k.tck") else "?")
    mfod = f"{D}/fod/{sid}/mask_fod.mif"; m5tt = f"{D}/fod/{sid}/mask_5tt_brain.mif"
    mv = mrcount(mfod) if os.path.exists(mfod) else None
    bv = mrcount(m5tt) if os.path.exists(m5tt) else None
    mq = round(mv / bv, 3) if (mv and bv) else ""
    res = best_result(sid); ncc = ""; surv = ""; recipe = ""
    if res:
        ncc = res.get("reg_ncc", ""); surv = res.get("aal_label_survival", "")
        rc = res.get("reg_choice") or {}; recipe = f"fod={res.get('fod','?')};reg={'fresh' if rc.get('fresh') else 'cached'}"
    r = {"sid": sid, "group": g, "density": d if d is not None else "", "band": band(d),
         "conn_location": loc or "", "n_weights": nw, "radial": radial,
         "mask_voxels": mv if mv else "", "mask_quality": mq,
         "reg_ncc": ncc, "label_survival": surv, "best_recipe": recipe}
    present = 0
    for name, fn in STAGES:
        p = fn(sid); r[name] = 1 if p else 0; r[name + "_path"] = p
        if p and name not in ("raw_dwi", "raw_t1"): present += 1
    r["stages_present"] = present
    s = S3.get(sid, set())
    for t in ("tck", "tsf", "csv", "sift"):
        r["s3_" + t] = int(t in s)
    r["s3_complete"] = int(all((("csv" if t == "csv" else t) in s) for t in ("tck", "tsf", "csv", "sift"))) if d and d >= 0.6 else 0
    return r

print(f"building manifest for {len(G2)} subjects ...")
rows = []
with cf.ThreadPoolExecutor(max_workers=16) as ex:
    for r in ex.map(row, sorted(G2)): rows.append(r)

stage_names = [n for n, _ in STAGES]
cols = (["sid", "group", "density", "band", "radial", "conn_location", "n_weights", "stages_present",
         "mask_voxels", "mask_quality", "reg_ncc", "label_survival", "best_recipe"]
        + stage_names + [n + "_path" for n in stage_names]
        + ["s3_tck", "s3_tsf", "s3_csv", "s3_sift", "s3_complete"])
with open(OUT, "w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=cols); w.writeheader()
    for r in sorted(rows, key=lambda x: (x["group"], -(x["density"] or 0) if isinstance(x["density"], float) else 1)):
        w.writerow(r)

from collections import Counter
bc = Counter((r["group"], r["band"]) for r in rows)
print(f"\nwrote {OUT}  ({len(rows)} subjects)")
for g in ["AD", "MCI", "CN"]:
    print(f"  {g:<4} good={bc[(g,'good')]:<4} mid={bc[(g,'mid')]:<3} low={bc[(g,'low')]:<3} none={bc[(g,'none')]}")
print(f"  TOTAL good={sum(1 for r in rows if r['band']=='good')}")
