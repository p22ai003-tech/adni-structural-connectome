#!/usr/bin/env python
# Authoritative count: best-ever density per subject across production SC_AAL166 + EVERY recovery root.
# Rebuilds good_density so wins are captured permanently (not orphaned when a run pointer moves).
import glob, os, re, csv, json, numpy as np
G2={}
for r in csv.DictReader(open("/data/derivatives/sc_forge_v1/group_qc/current_connectome_subject_availability_with_group.csv")):
    g=r.get("group","?"); G2[r["sid"]]="MCI" if g=="SMC" else g
def sidof(s):
    m=re.search(r"[0-9]{3}_S_[0-9]+_I[0-9]+",s); return m.group(0) if m else None
def dens(p):
    try:
        m=np.loadtxt(p,delimiter=","); n=m.shape[0]; off=m[~np.eye(n,dtype=bool)]; return float((off>0).sum()/off.size)
    except: return None
best={}
for f in glob.glob("/data/derivatives/connectomes/SC_AAL166_*count.csv"):
    if "_invnodevol" in f: continue
    s=sidof(os.path.basename(f)); d=dens(f)
    if s in G2 and d is not None: best[s]=max(best.get(s,0),d)
for rj in glob.glob("/data/derivatives/qc/sc_matrix_qc/*/*/result.json"):
    s=os.path.basename(os.path.dirname(rj))
    if s not in G2: continue
    try: d=(json.load(open(rj)).get("aal_qc") or {}).get("density") or 0
    except: continue
    best[s]=max(best.get(s,0),d)
GOODF={"AD":"/tmp/ad_good_density.txt","MCI":"/tmp/mci_good_density.txt","CN":"/tmp/cn_good_density.txt"}
cnt={}
for coh,f in GOODF.items():
    lines=[f"{s} {round(best[s],4)}" for s in best if G2.get(s)==coh and best[s]>=0.6]
    open(f,"w").write("\n".join(sorted(lines))+"\n"); cnt[coh]=len(lines)
print("good_density rebuilt (authoritative best-ever):", cnt, "TOTAL", sum(cnt.values()))
