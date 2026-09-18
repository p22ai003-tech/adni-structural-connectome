#!/bin/bash
# Real MCI/CN recovery: the UNDER-TRIED pending (<3 lanes) finally get the proven sturdy lane
# (cached healthy FOD + 10M + fresh SyN reg + noACT). Closest-to-0.6 first. Shows as L5 in cascade.
PY=/home/ec2-user/exp/.venv_connectome_app/bin/python
cd /home/ec2-user/exp
$PY - <<'PYEOF'
import glob,json,os,re,csv,numpy as np
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
    if s and d is not None: best[s]=max(best.get(s,0),d)
lanes={}
for rj in glob.glob("/data/derivatives/qc/sc_matrix_qc/*/*/result.json"):
    sid=os.path.basename(os.path.dirname(rj))
    if sid not in G2: continue
    try: d=(json.load(open(rj)).get("aal_qc") or {}).get("density") or 0
    except: continue
    best[sid]=max(best.get(sid,0),d)
    fam=re.sub(r'_?\d{4,}T?\d*Z?.*$','',rj.split("/sc_matrix_qc/")[1].split("/")[0])
    lanes.setdefault(sid,set()).add(fam)
cand=[(best.get(s,0),s) for s in G2 if G2[s] in ("MCI","CN") and best.get(s,0)<0.6 and len(lanes.get(s,set()))<3]
cand.sort(reverse=True)   # closest to 0.6 first
open("/tmp/mcicn_undertried.txt","w").write("\n".join(s for _,s in cand)+"\n")
print(f"under-tried MCI/CN to recover: {len(cand)} (closest first: {cand[0][1]} @ {cand[0][0]:.3f})")
PYEOF
RROOT=/data/derivatives/qc/sc_matrix_qc/mcicn_recover_$(date -u +%m%dT%H%M%SZ)
mkdir -p "$RROOT"; echo "$RROOT" > /tmp/lane5_mcicn_run_root.txt
MRTRIX_NTHREADS=6 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=6 OMP_NUM_THREADS=6 \
FORCE_NOACT=1 REBUILD_FOD=0 FRESH_T12B0=1 REREG_THRESH=0.5 \
TCKGEN_SELECT=10000000 RESUME_MIN_DENSITY=0.6 TMPDIR=/data/scratch_tmp \
  nohup setsid nice -n 5 $PY scripts/scforge/live/run_sc_route_sota.py \
    --subjects $(cat /tmp/mcicn_undertried.txt) \
    --run-root "$RROOT" --mode full_10m --workers 28 --threads 6 --radial 2 --reg-mode fullhead \
    > "$RROOT.log" 2>&1 </dev/null &
sleep 3
echo "MCI/CN recovery launched -> $RROOT (workers 28 x 6) | drivers: $(pgrep -f mcicn_recover | wc -l)"
