#!/usr/bin/env python
"""Reconstruct the fsid -> full-head-T1 manifest for the HCP/FastSurfer replication, persistently.
Reads the 530-subject master cohort, joins each to its corrected_T1 (full head) by participant id,
and checks connectome inputs. Writes /data/derivatives/parc_hcpmmp1/cohort_manifest.tsv.
"""
import csv, glob, os, re, collections

MASTER = "/data/derivatives/qc/analysis_cohort/00_master/master_cohort.csv"
T1DIR = "/data/derivatives/t1_anat"
OUT = "/data/derivatives/parc_hcpmmp1/cohort_manifest.tsv"

def participant(fsid):
    # 002_S_0413_I863064 -> 002_S_0413
    m = re.match(r"(\d{3}_S_\d+)_I\d+", fsid)
    return m.group(1) if m else fsid

rows = []
with open(MASTER) as f:
    for r in csv.DictReader(f):
        part = r["subject_id"].strip()                 # participant id, e.g. 002_S_0413
        dwi_id = r["Image ID"].strip()                  # DWI image id used for the connectome
        fsid = f"{part}_I{dwi_id}"                       # full fsid, e.g. 002_S_0413_I863064
        grp = (r.get("group") or r.get("Research Group") or "").strip()
        t1s = sorted(glob.glob(f"{T1DIR}/corrected_T1_{part}_I*.nii.gz"))
        t1 = t1s[0] if t1s else ""
        trk = f"/data/derivatives/tracks/{fsid}"
        inputs_ok = bool(t1) and all(os.path.isfile(p) for p in [
            f"/data/derivatives/dwi_t1_bbr/{fsid}_b0mean_ras.nii.gz",
            f"{trk}/tracks_final_3000k.tck", f"{trk}/sift_weights.txt",
            f"{trk}/fa_mean.tsf", f"{trk}/md_mean.tsf", f"{trk}/rd_mean.tsf", f"{trk}/ad_mean.tsf",
        ])
        rows.append((fsid, grp, t1, "1" if inputs_ok else "0"))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["fsid", "group", "t1_fullhead", "inputs_ok"])
    w.writerows(rows)

by_grp = collections.Counter(r[1] for r in rows)
ok = [r for r in rows if r[3] == "1"]
by_grp_ok = collections.Counter(r[1] for r in ok)
missing_t1 = sum(1 for r in rows if not r[2])
print(f"wrote {OUT}: {len(rows)} subjects")
print("group breakdown (all):     ", dict(by_grp))
print("group breakdown (inputs_ok):", dict(by_grp_ok), f"= {len(ok)} total")
print(f"missing full-head T1: {missing_t1}")
