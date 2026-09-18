#!/usr/bin/env python
"""Stage funnel — how many subjects are present at each pipeline stage, overall and by cohort/band.
Reads subject_manifest.csv. Output: /data/derivatives/qc/sc_matrix_qc/stage_funnel.csv (+ stdout table)."""
import csv
from collections import defaultdict

MAN = "/data/derivatives/qc/sc_matrix_qc/subject_manifest.csv"
OUT = "/data/derivatives/qc/sc_matrix_qc/stage_funnel.csv"
STAGES = ["raw_dwi", "raw_t1", "mif_dwi", "mif_denoised", "mif_unringed", "eddy", "biascorr", "t1", "bbr",
          "fod", "dti", "parc", "tck", "sift", "tsf"]

rows = list(csv.DictReader(open(MAN)))
def count(stage, group=None, bandset=None):
    return sum(1 for r in rows if r[stage] == "1"
               and (group is None or r["group"] == group)
               and (bandset is None or r["band"] in bandset))

with open(OUT, "w", newline="") as h:
    w = csv.writer(h)
    w.writerow(["stage", "total", "AD", "MCI", "CN", "good", "mid", "low"])
    for s in STAGES + ["connectome_good"]:
        if s == "connectome_good":
            tot = sum(1 for r in rows if r["band"] == "good")
            ad = sum(1 for r in rows if r["band"] == "good" and r["group"] == "AD")
            mci = sum(1 for r in rows if r["band"] == "good" and r["group"] == "MCI")
            cn = sum(1 for r in rows if r["band"] == "good" and r["group"] == "CN")
            w.writerow([s, tot, ad, mci, cn, tot, 0, 0])
        else:
            w.writerow([s, count(s), count(s, "AD"), count(s, "MCI"), count(s, "CN"),
                        count(s, bandset={"good"}), count(s, bandset={"mid"}), count(s, bandset={"low"})])

print(f"wrote {OUT}\n")
print(f"{'stage':<16}{'total':>7}{'AD':>6}{'MCI':>6}{'CN':>6}   {'good':>5}{'mid':>5}{'low':>5}")
print("-" * 64)
for r in csv.DictReader(open(OUT)):
    print(f"{r['stage']:<16}{r['total']:>7}{r['AD']:>6}{r['MCI']:>6}{r['CN']:>6}   {r['good']:>5}{r['mid']:>5}{r['low']:>5}")
