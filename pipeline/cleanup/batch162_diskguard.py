#!/usr/bin/env python
# Disk guard for batch162: tracks_10M.tck = ~7GB each, 162 of them would overflow 620G.
# Promotion only needs the connectome CSVs (already built when result.json exists), so:
#  - FAILED subjects (<0.6): delete tck + intermediates immediately (never promoted).
#  - PASSED subjects (>=0.6): keep tck (preserve best tractogram), purge only intermediates.
#  - Emergency: if disk free < 120G, delete oldest PASSED tcks too (connectome already safe).
import glob, os, json, time, shutil
ROOTGLOB = "/data/derivatives/qc/sc_matrix_qc/batch162_*"
INTERMED = ["eddy_gradfix.mif", "wmfod_rebuilt.mif", "_b0s.mif",
            "aal3_166_dwi_cached.step.mif", "schaefer200_dwi.step.mif",
            "b0_brain.nii.gz", "mask_fod.nii.gz", "aal3_166_dwi_cached.nii.gz"]
def freeG():
    s=os.statvfs("/data"); return s.f_bavail*s.f_frsize/1e9
def rm(p):
    try: os.remove(p); return os.path.getsize(p) if os.path.exists(p) else 0
    except Exception: return 0
while True:
    roots=glob.glob(ROOTGLOB)
    if not roots:  # batch finished/removed -> exit
        break
    drivers=os.popen("pgrep -fc 'run_sc_route_sota.py.*batch162'").read().strip()
    for root in roots:
        for d in glob.glob(root+"/*/"):
            rj=os.path.join(d,"result.json")
            if not os.path.exists(rj): continue
            try: dens=(json.load(open(rj)).get("aal_qc") or {}).get("density") or 0
            except Exception: continue
            for f in INTERMED: rm(os.path.join(d,f))            # always purge intermediates
            tck=os.path.join(d,"tracks_10M.tck")
            if dens < 0.6 and os.path.exists(tck): rm(tck)      # failed -> drop tck
    if freeG() < 120:                                            # emergency valve
        passed=[]
        for root in roots:
            for d in glob.glob(root+"/*/"):
                rj=os.path.join(d,"result.json"); tck=os.path.join(d,"tracks_10M.tck")
                if os.path.exists(rj) and os.path.exists(tck): passed.append(tck)
        passed.sort(key=os.path.getmtime)
        for tck in passed[:10]:
            if freeG()>=160: break
            rm(tck)
    if drivers=="0":  # batch drivers gone -> final sweep done, exit
        break
    time.sleep(90)
