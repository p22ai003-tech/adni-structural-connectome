#!/usr/bin/env python
"""Band-navigable symlink views — browse every subject's FULL file set (inputs -> outputs) by quality band.
Pure symlinks (zero data movement, reversible). Reads subject_manifest.csv (uses its *_path columns).

Layout:
  /data/derivatives/_by_quality/{good_ge0.6, mid_0.4_0.6, low_lt0.4, none}/<sid>/
     <stage> -> <real path>   (one symlink per present stage)
     audit_row.txt            (the subject's manifest row, human-readable)

Rebuildable: `rm -rf /data/derivatives/_by_quality` then re-run. Safe full reset.
"""
import csv, os, shutil

MAN = "/data/derivatives/qc/sc_matrix_qc/subject_manifest.csv"
ROOT = "/data/derivatives/_by_quality"
BANDDIR = {"good": "good_ge0.6", "mid": "mid_0.4_0.6", "low": "low_lt0.4", "none": "none"}
STAGES = ["raw_dwi", "raw_t1", "mif_dwi", "mif_denoised", "mif_unringed", "eddy", "biascorr", "t1", "bbr",
          "fod", "dti", "parc", "tck", "sift", "tsf"]

if os.path.isdir(ROOT):
    shutil.rmtree(ROOT)
rows = list(csv.DictReader(open(MAN)))
made = {b: 0 for b in BANDDIR}
links = 0
for r in rows:
    band = r["band"]; sid = r["sid"]
    sdir = os.path.join(ROOT, BANDDIR[band], sid)
    os.makedirs(sdir, exist_ok=True); made[band] += 1
    for st in STAGES:
        p = r.get(st + "_path", "")
        if p and os.path.exists(p):
            link = os.path.join(sdir, st)
            try: os.symlink(p, link); links += 1
            except FileExistsError: pass
    # connectome link (banded location)
    cl = r.get("conn_location", "")
    if cl:
        sub = "" if cl == "good_main" else "/" + cl
        cpath = f"/data/derivatives/connectomes{sub}"
        cc = f"{cpath}/SC_AAL166_{sid}_count.csv"
        if os.path.exists(cc):
            try: os.symlink(cpath, os.path.join(sdir, "connectomes_dir")); links += 1
            except FileExistsError: pass
    with open(os.path.join(sdir, "audit_row.txt"), "w") as h:
        h.write(f"sid={sid} group={r['group']} density={r['density']} band={band} "
                f"radial(see manifest) n_weights={r['n_weights']} stages_present={r['stages_present']}\n"
                f"conn_location={cl}\n")

print(f"built {ROOT}")
for b, d in BANDDIR.items():
    print(f"  {d}: {made[b]} subjects")
print(f"  total symlinks: {links}")
