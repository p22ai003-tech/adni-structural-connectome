#!/bin/bash
# Recipe v4 bench driver. Lowest CPU priority; WORKERS x THREADS never above the 40-thread cap. Safe to start again at any time: a driver lock
# prevents two drivers, finished runs are skipped, per-run locks prevent duplicate work, each failed run gets one retry.
#   rv4_run.sh canary   2 sessions (1 Siemens, 1 GE) through A, B, C at 1M; no DONE
#   rv4_run.sh all      every session and arm, then second seed and 10M on arm B (arm C too if the rule adopts it); compare.md; bench/DONE
#   rv4_run.sh status   counts only, no identifiers
#   rv4_run.sh worker   internal
# Start detached:  ( setsid nohup bash rv4_run.sh all >> /data/derivatives/recipe_v4/bench/logs/driver.log 2>&1 & )
# Stop gently:     touch /data/derivatives/recipe_v4/bench/STOP   (running jobs finish; nothing new starts)
export B=/data/derivatives/recipe_v4/bench SCR=/data/derivatives/recipe_v4/scratch L=/data/derivatives/recipe_v4/lists
export T=/home/ec2-user/exp/docs/iit_dipanjan/tools/recipe_v4 PY=/home/ec2-user/exp/.venv_connectome_app/bin/python
export MRTRIX_TMPFILE_DIR=$SCR/ TMPDIR=$SCR
export MAX_BENCH_GB=140 MIN_FREE_GB=300 MIN_MEM_GB=20
cd "$SCR" || exit 1
mkdir -p "$B/logs" "$B/qc"

say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M IST') $*" >> "$B/logs/driver.log"; }
export -f say

case "${1:-}" in
worker)
  # one job: arm unit select seed tag threads. Waits for memory and disk, runs, retries once.
  shift; arm=$1 unit=$2 sel=$3 seed=$4 tag=$5 thr=$6; [ "$tag" = "-" ] && tag=""
  name=$arm${tag:+_$tag}; crop=""; [ "$arm" != "A" ] && crop="--crop-zf"   # RULE.md, change of 3 Nov: a no-op on scans that were not interpolated
  while :; do
    [ -f "$B/STOP" ] && exit 0
    mem=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    free=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
    used=$(du -sBG "$B" "$SCR" 2>/dev/null | awk '{s+=$1} END{print int(s)}')
    [ "$mem" -ge "$MIN_MEM_GB" ] && [ "$free" -ge "$MIN_FREE_GB" ] && [ "$used" -le "$MAX_BENCH_GB" ] && break
    say "wait: mem ${mem}G free ${free}G bench ${used}G"; sleep 120
  done
  for try in 1 2; do
    if nice -n 19 "$PY" -I "$T/rv4_subject.py" "$unit" --arm "$arm" --select "$sel" --seed "$seed" ${tag:+--tag "$tag"} --threads "$thr" --out "$B" $crop >> "$B/logs/${name}_jobs.log" 2>&1; then break; fi
    say "run failed, try $try, arm $name (see qc json)"
  done
  exit 0 ;;
status)
  for f in "$B"/qc/*.json; do [ -f "$f" ] && "$PY" -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['armtag'], d['status'], d.get('failed_stage',''))" "$f"; done | sort | uniq -c
  echo "running jobs: $(ps -eo cmd | grep -c '[r]v4_subject.py')"; du -sh "$B" "$SCR" 2>/dev/null | sed 's#/data/derivatives/recipe_v4/##'; df -h /data | tail -1 | awk '{print "disk free", $4}'
  [ -f "$B/DONE" ] && cat "$B/DONE"; [ -f "$B/STOP" ] && echo "STOP file present"
  tail -3 "$B/logs/driver.log" 2>/dev/null; exit 0 ;;
esac

exec 9>"$B/driver.lock"; flock -n 9 || { say "driver already running"; exit 0; }
say "driver start: ${1:-all}"

# jobs(): feed lines to a pool of workers
pool() { # file workers threads-are-in-the-file
  [ -s "$1" ] && xargs -a "$1" -P "$2" -L 1 nice -n 19 bash "$T/rv4_run.sh" worker
}

# Fixed sets, seeded, no diagnosis: canaries, 4 sessions for 10M (one per acquisition stratum), 6 for the second seed
"$PY" -I - <<'EOF'
import os, numpy as np, pandas as pd
B = os.environ["B"]; L = os.environ["L"]
b = pd.read_csv(f"{L}/bench_units.csv", dtype=str); b["stratum"] = b.phase + " " + b.vendor
rng = np.random.default_rng(20261102)
rows = []
for st, n10, n2 in [("ADNI 3 SIEMENS", 1, 2), ("ADNI 3 PHILIPS", 1, 1), ("ADNI 3 GE", 1, 2), ("ADNI 2 GE", 1, 1)]:
    q = b[b.stratum == st].sort_values("unit"); u = q.unit.values
    rows += [(x, "10M") for x in rng.choice(u, n10, replace=False)] + [(x, "seed2") for x in rng.choice(u, n2, replace=False)]
pd.DataFrame(rows, columns=["unit", "variant"]).to_csv(f"{B}/variants.csv", index=False)
sm = b[b.vendor == "SIEMENS"].unit.iloc[0]
ge = [u for u in b[(b.vendor == "GE") & (b.phase == "ADNI 3")].unit if not os.path.isdir(f"/data/derivatives/fastsurfer/{u}")][0]   # GE with no stored FastSurfer: measures the CPU route
open(f"{B}/canary_units.txt", "w").write(f"{sm}\n{ge}\n")
EOF

WK=5; TH=8   # 5 x 8 = 40 threads
tmp=$B/logs/jobs_$$.txt

if [ "${1:-all}" = "canary" ]; then
  : > "$tmp"
  for a in B A C; do while read -r u; do echo "$a $u 1000000 1 - $TH"; done < "$B/canary_units.txt"; done >> "$tmp"
  pool "$tmp" "$WK"
  say "canary finished"; rm -f "$tmp"; exit 0
fi

# Phase 1: arms B, C and A of each session together, acquisition strata in turn, so that a fault in any arm or stratum shows in the first hour, not after a whole arm has run.
# Shared steps are locked per session, so arms cannot collide.
"$PY" -I - "$TH" > "$tmp" <<'EOF'
import os, sys, pandas as pd
b = pd.read_csv(os.environ["L"] + "/bench_units.csv", dtype=str); b["turn"] = b.groupby(["phase", "vendor"]).cumcount()
for u in b.sort_values("turn", kind="stable").unit:
    for arm in "BCA": print(arm, u, 1000000, 1, "-", sys.argv[1])
EOF
say "phase 1: arms B, C, A at 1M, $(wc -l < "$tmp") jobs"; pool "$tmp" "$WK"
[ -f "$B/STOP" ] && { say "stopped by STOP file"; rm -f "$tmp"; exit 0; }

# Phase 2: second seed on arm B, which needs the base runs of phase 1
grep ',seed2$' "$B/variants.csv" | cut -d, -f1 | while read -r u; do echo "B $u 1000000 2 seed2 $TH"; done > "$tmp"
say "phase 2: second seed on arm B, $(wc -l < "$tmp") jobs"; pool "$tmp" "$WK"
[ -f "$B/STOP" ] && { say "stopped by STOP file"; rm -f "$tmp"; exit 0; }

# Phase 3: what the rule needs for the arm it picks
"$PY" -I "$T/rv4_compare.py" --out "$B" >> "$B/logs/driver.log" 2>&1
PICK=$("$PY" -I -c "import json; print(json.load(open('$B/compare.json'))['recipe_arm'])")
say "rule picks arm $PICK at 1M"
grep ',10M$' "$B/variants.csv" | cut -d, -f1 | while read -r u; do echo "B $u 10000000 1 10M 20"; done > "$tmp"
if [ "$PICK" = "C" ]; then
  { grep ',seed2$' "$B/variants.csv" | cut -d, -f1 | while read -r u; do echo "C $u 1000000 2 seed2 20"; done
    grep ',10M$' "$B/variants.csv" | cut -d, -f1 | while read -r u; do echo "C $u 10000000 1 10M 20"; done; } >> "$tmp"
fi
say "phase 3: 10M and variants at 2 x 20 threads, $(wc -l < "$tmp") jobs"; pool "$tmp" 2
[ -f "$B/STOP" ] && { say "stopped by STOP file"; rm -f "$tmp"; exit 0; }

"$PY" -I "$T/rv4_compare.py" --out "$B" --final >> "$B/logs/driver.log" 2>&1
rm -f "$tmp"
"$PY" -I -c "
import json; d = json.load(open('$B/compare.json')); print('finished', __import__('time').strftime('%F %H:%M UTC', __import__('time').gmtime()), '| recipe arm', d['recipe_arm'], '| go/no-go', d['verdict'])" > "$B/DONE"
say "all finished: $(cat "$B/DONE")"
