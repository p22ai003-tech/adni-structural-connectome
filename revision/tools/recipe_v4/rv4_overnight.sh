#!/bin/bash
# Overnight after the cohort run (FREEZE.md, 6 Nov): rehearsal on shuffled diagnoses as soon as the first derive pass is done; then, only if
# every rehearsal stage is ok, and once the run is DONE with every ok session derived and none failed: the real chains act10m, then
# noact3m + act10m_noflag (they need act10m's ruler), then act3m (needs noact3m's edges). Stops and says so on anything unexpected.
# Log: recipe_v4/chains/overnight.log
set -u; T=/home/ec2-user/exp/docs/iit_dipanjan/tools/recipe_v4; R4=/data/derivatives/recipe_v4; B=$R4/cohort; C=$R4/chains; D=$R4/derived
PY=/home/ec2-user/exp/.venv_connectome_app/bin/python
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M IST') $*"; }
bad() { awk -F'\t' 'NR>1 && $2!=0' "$C/$1/logs/chain_status.tsv" 2>/dev/null | wc -l; }
mkdir -p "$C/inputs"
until grep -q 'derived so far' $B/logs/derive.log; do sleep 60; done
say "first derive pass done: $(ls $D/dkt_act3m/*_qc.json | wc -l) sessions; rehearsal starts"
bash $T/rv4_chains.sh rehearsal > $C/rehearsal.log 2>&1; rc=$?
n=$(bad rehearsal); say "rehearsal: exit $rc, $n stages not ok"
if [ "$rc" -ne 0 ] || [ "$n" -ne 0 ]; then say "rehearsal not clean: real chains NOT started; look at it ($C/rehearsal/logs/chain_status.tsv)"; exit 1; fi
until grep -q 'run DONE and every ok session handled' $B/logs/derive.log; do sleep 120; done
nf=$(ls $D/failed_*.txt 2>/dev/null | wc -l); nd=$(ls $D/dkt_act3m/*_qc.json | wc -l)
if [ "$nf" -ne 0 ]; then say "$nf sessions failed to derive: real chains NOT started; look at it"; exit 1; fi
$PY -I - "$R4/lists/cohort_units.csv" "$B" "$C/inputs/under30_dw_volumes.txt" <<'PYEOF'
import json, os, sys, numpy as np
units, B, out = sys.argv[1], sys.argv[2], sys.argv[3]
ok = lambda p: os.path.exists(p) and json.load(open(p)).get("status") == "ok"
low = [u for u in (l.split(",")[0] for l in open(units).read().splitlines()[1:])
       if ok(f"{B}/qc/C_{u}.json") and ok(f"{B}/qc/C_noact_{u}.json") and int((np.loadtxt(f"{B}/work/C/{u}/dwi_in.bval") > 100).sum()) < 30]
open(out, "w").write("".join(f"{u}\n" for u in low)); print(len(low))
PYEOF
say "sessions derived $nd; under 30 diffusion-weighted volumes: $(wc -l < $C/inputs/under30_dw_volumes.txt) (expected 11)"
bash $T/rv4_chains.sh act10m > $C/act10m.log 2>&1 & A=$!
until [ -f $C/act10m/connectomes/ruler_mni_distance.npz ] || ! kill -0 $A 2>/dev/null; do sleep 30; done
[ -f $C/act10m/connectomes/ruler_mni_distance.npz ] || { wait $A; say "act10m ended without a ruler (exit $?): nothing else started"; exit 1; }
bash $T/rv4_chains.sh noact3m > $C/noact3m.log 2>&1 & N=$!
bash $T/rv4_chains.sh act10m_noflag > $C/act10m_noflag.log 2>&1 & F=$!
say "act10m running; noact3m and act10m_noflag started"
until [ -f $C/noact3m/phase2/edges_uniform_v3.npz ] || ! kill -0 $N 2>/dev/null; do sleep 60; done
if [ -f $C/noact3m/phase2/edges_uniform_v3.npz ]; then bash $T/rv4_chains.sh act3m > $C/act3m.log 2>&1 & M=$!; say "act3m started (paired ACT vs no-ACT)"; else M=; say "noact3m ended without edges: act3m NOT started"; fi
for p in $A $N $F $M; do wait $p; done
say "ALL CHAINS ENDED: stages not ok: act10m $(bad act10m), noact3m $(bad noact3m), act10m_noflag $(bad act10m_noflag), act3m $(bad act3m)"
