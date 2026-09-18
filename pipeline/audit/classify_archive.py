#!/usr/bin/env python
"""Phase 3: quarantine sub-standard connectomes out of the main dataset (reversible, local).
good (>=0.6) stay in connectomes/.  mid (0.4-0.6) -> connectomes/_midband/.  low (<0.4) -> connectomes/_lowband_archive/.
Moves SC_AAL166_* and SC_Schaefer200_* for each subject. Source inputs (eddy/fod/dti/tracks) untouched.
The Phase-4 S3 sync (--delete) mirrors this structure to S3 afterwards.
"""
import csv, glob, os, shutil
CONN = "/data/derivatives/connectomes"
MID = f"{CONN}/_midband"; LOW = f"{CONN}/_lowband_archive"
for d in (MID, LOW): os.makedirs(d, exist_ok=True)
AUDIT = "/data/derivatives/qc/sc_matrix_qc/subject_audit.csv"

rows = list(csv.DictReader(open(AUDIT)))
moved = {"mid": 0, "low": 0}
for r in rows:
    band = r["band"]; sid = r["sid"]
    if band not in ("mid", "low"): continue
    dest = MID if band == "mid" else LOW
    files = glob.glob(f"{CONN}/SC_AAL166_{sid}_*.csv") + glob.glob(f"{CONN}/SC_Schaefer200_{sid}_*.csv")
    for f in files:
        tgt = os.path.join(dest, os.path.basename(f))
        shutil.move(f, tgt)
    if files: moved[band] += 1

# verify main dir now holds only good
def sidset(pat):
    import re
    return {re.search(r"[0-9]{3}_S_[0-9]+_I[0-9]+", os.path.basename(f)).group(0)
            for f in glob.glob(pat) if re.search(r"[0-9]{3}_S_[0-9]+_I[0-9]+", os.path.basename(f))}
good_sids = {r["sid"] for r in rows if r["band"] == "good"}
main_sids = sidset(f"{CONN}/SC_AAL166_*_count.csv")
leak = main_sids - good_sids
print(f"moved: mid={moved['mid']} subjects -> _midband/, low={moved['low']} subjects -> _lowband_archive/")
print(f"main connectomes/ now holds {len(main_sids)} subjects | non-good leaked into main: {len(leak)}")
print(f"  _midband files: {len(glob.glob(MID+'/*.csv'))} | _lowband files: {len(glob.glob(LOW+'/*.csv'))}")
