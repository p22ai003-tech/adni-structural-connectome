#!/bin/bash
# Recipe v4 cohort driver: the frozen recipe (FREEZE.md) on lists/cohort_units.csv. Per session, one job: arm C at 10M with ACT and a 3M subset, then the no-ACT
# variant at 3M from the same FOD, atlas and seed (FREEZE.md decision 7: like-for-like with the ACT run's 3M subset). Lowest CPU priority. Safe to start again at any time: a driver lock prevents two drivers, finished runs are skipped,
# per-run locks prevent duplicate work, each failed run gets one retry.
#   rv4_cohort.sh canary   lists/canary_units.txt into recipe_v4/canary (no signature needed)
#   rv4_cohort.sh all      every cohort session into recipe_v4/cohort (rv4_subject.py refuses without SIGNED_RECIPE.json naming its code); then flags.csv and DONE
#   rv4_cohort.sh status   counts only, no identifiers
#   rv4_cohort.sh worker   internal
# Start detached:  ( setsid nohup bash rv4_cohort.sh all >> /data/derivatives/recipe_v4/cohort/logs/driver.log 2>&1 & )
# Stop gently:     touch /data/derivatives/recipe_v4/<cohort|canary>/STOP   (running jobs finish their session; nothing new starts)
# Workers: (vCPUs - 24) / 8 at 8 threads each, at least 1: 5 on c6a.16xlarge (64), 21 on c7a.48xlarge (192). Override with WK=.
# Several machines (FREEZE.md decision 5): SLICE=k/N takes every N-th session from position k of the list (k = 0..N-1); slices are disjoint.
export R4=/data/derivatives/recipe_v4 SCR=/data/derivatives/recipe_v4/scratch L=/data/derivatives/recipe_v4/lists
export T=/home/ec2-user/exp/docs/iit_dipanjan/tools/recipe_v4 PY=/home/ec2-user/exp/.venv_connectome_app/bin/python
export MRTRIX_TMPFILE_DIR=$SCR/ TMPDIR=$SCR MIN_FREE_GB=100 MIN_MEM_GB=20 TH=8
mode=${1:-status}
case "$mode" in canary) export B=$R4/canary ;; worker) : ;; *) export B=$R4/cohort ;; esac
cd "$SCR" || exit 1
mkdir -p "$B/logs" "$B/qc"
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M IST') $*" >> "$B/logs/driver.log"; }
export -f say

case "$mode" in
worker)
  # one session: ACT run, then its no-ACT variant. Waits for memory and disk; one retry per run.
  unit=$2
  for tag in "" noact; do
    if [ -z "$tag" ]; then sel="--select 10000000 --subset 3000000"; else sel="--select 3000000"; fi
    while :; do
      [ -f "$B/STOP" ] && exit 0
      mem=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); free=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
      [ "$mem" -ge "$MIN_MEM_GB" ] && [ "$free" -ge "$MIN_FREE_GB" ] && break
      say "wait: mem ${mem}G free ${free}G"; sleep 120
    done
    ok=0
    for try in 1 2; do
      if nice -n 19 "$PY" -I "$T/rv4_subject.py" "$unit" --arm C $sel --seed 1 ${tag:+--tag "$tag"} --threads "$TH" --crop-zf --out "$B" \
           >> "$B/logs/C${tag:+_$tag}_jobs.log" 2>&1; then ok=1; break; fi
      say "run failed, try $try, arm C${tag:+_$tag} (see qc json)"
    done
    [ $ok = 1 ] || exit 0                                # the no-ACT variant needs the ACT run's FOD and atlas
  done
  exit 0 ;;
status)
  for f in "$B"/qc/*.json; do [ -f "$f" ] && "$PY" -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['armtag'], d['status'], d.get('failed_stage',''))" "$f"; done | sort | uniq -c
  echo "running jobs: $(ps -eo cmd | grep -c '[r]v4_subject.py')"; du -sh "$B" "$SCR" 2>/dev/null | sed 's#/data/derivatives/recipe_v4/##'; df -h /data | tail -1 | awk '{print "disk free", $4}'
  [ -f "$B/DONE" ] && cat "$B/DONE"; [ -f "$B/STOP" ] && echo "STOP file present"
  tail -3 "$B/logs/driver.log" 2>/dev/null; exit 0 ;;
canary|all) : ;;
*) echo "usage: rv4_cohort.sh canary|all|status"; exit 1 ;;
esac

exec 9>"$B/driver.lock"; flock -n 9 || { say "driver already running"; exit 0; }
WK=${WK:-$(( ($(nproc) - 24) / 8 ))}; [ "$WK" -lt 1 ] && WK=1
say "driver start: $mode, $WK workers x $TH threads"
tmp=$B/logs/jobs_$$.txt
if [ "$mode" = canary ]; then cp "$L/canary_units.txt" "$tmp"
else
  # acquisition strata in turn, so that a fault in any stratum shows in the first hours, not after a whole stratum has run
  "$PY" -I -c "
import pandas as pd; c = pd.read_csv('$L/cohort_units.csv', dtype=str); c['turn'] = c.groupby(['phase', 'vendor']).cumcount()
u = list(c.sort_values(['turn', 'phase', 'vendor'], kind='stable').unit); k, n = map(int, '${SLICE:-0/1}'.split('/'))
print('\n'.join(u[k::n]))" > "$tmp"
fi
say "$mode: $(wc -l < "$tmp") sessions${SLICE:+, slice $SLICE}"
"$PY" -I "$T/rv4_t1_ingest.py" --units "$tmp" --jobs "$(( $(nproc) / 2 ))" >> "$B/logs/t1_ingest.log" 2>&1   # T1s still missing for these sessions; finished ones are skipped
say "T1s: $(tail -1 "$B/logs/t1_ingest.log")"
xargs -a "$tmp" -P "$WK" -L 1 nice -n 19 bash "$T/rv4_cohort.sh" worker
rm -f "$tmp"
[ -f "$B/STOP" ] && { say "stopped by STOP file"; exit 0; }
# FREEZE.md decision 3: whole-brain WM Dice under .65 flags a session for a look with diagnosis hidden and for the analyses repeated without it
"$PY" -I - "$B" > "$B/DONE" <<'EOF'
import glob, json, sys, time, pandas as pd
B = sys.argv[1]; rows = []
for f in glob.glob(f"{B}/qc/C_*.json") + glob.glob(f"{B}/qc/C_noact_*.json"):
    d = json.load(open(f)); rows.append(dict(unit=d["unit"], armtag=d["armtag"], status=d["status"], failed_stage=d.get("failed_stage", ""), wm_dice=d.get("wm_dice")))
q = pd.DataFrame(rows).drop_duplicates(["unit", "armtag"])
f = q[q.armtag == "C"].assign(flag_wm_dice_under_065=lambda x: x.wm_dice.astype(float) < 0.65); f.to_csv(f"{B}/flags.csv", index=False)
print("finished", time.strftime("%F %H:%M UTC", time.gmtime()), "|", q.groupby(["armtag", "status"]).size().to_dict(), "| flagged WM Dice < .65:", int(f.flag_wm_dice_under_065.sum()))
EOF
say "all finished: $(cat "$B/DONE")"
