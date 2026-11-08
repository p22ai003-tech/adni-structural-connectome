#!/bin/bash
# After the cohort run (FREEZE.md, 6 Nov): rv4_derive.py for every session whose two runs are ok on this server, 24 at a time at nice 19,
# until the run is DONE and every ok session is derived. A session counts as derived when its last file (dkt_act3m qc.json) exists.
# Log: cohort/logs/derive.log; failures: derived/failed_<unit>.txt (the error), retried once.
set -u; T=/home/ec2-user/exp/docs/iit_dipanjan/tools/recipe_v4; R4=/data/derivatives/recipe_v4; B=$R4/cohort; O=$R4/derived; PY=/home/ec2-user/exp/.venv_connectome_app/bin/python
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M IST') $*"; }
mkdir -p "$O"; cd "$R4/scratch" || exit 1
one() { u=$1; out=$(nice -n 19 $PY -I $T/rv4_derive.py "$u" "$O" 2>&1) && { rm -f "$O/failed_$u.txt"; return 0; }
        echo "$out" | tail -3 > "$O/failed_$u.txt"; return 1; }
export -f one; export T O PY
while :; do
  todo=$(tail -n +2 $R4/lists/cohort_units.csv | cut -d, -f1 | while read -r u; do
    grep -qs '"status": "ok"' "$B/qc/C_noact_$u.json" && grep -qs '"status": "ok"' "$B/qc/C_$u.json" || continue
    [ -f "$O/dkt_act3m/SC_DKT76_${u}_qc.json" ] && continue
    [ -f "$O/failed_$u.txt" ] && [ -f "$O/retried_$u" ] && continue
    [ -f "$O/failed_$u.txt" ] && touch "$O/retried_$u"
    echo "$u"; done)
  n=$(echo -n "$todo" | grep -c .)
  if [ "$n" -gt 0 ]; then
    say "deriving $n sessions"
    echo "$todo" | xargs -P 24 -I{} bash -c 'one "$1"' _ {}
    say "derived so far: $(ls $O/dkt_act3m/*_qc.json 2>/dev/null | wc -l); failed: $(ls $O/failed_*.txt 2>/dev/null | wc -l)"
  fi
  if [ -f "$B/DONE" ] && [ "$n" -eq 0 ]; then say "run DONE and every ok session handled: derived $(ls $O/dkt_act3m/*_qc.json | wc -l), failed $(ls $O/failed_*.txt 2>/dev/null | wc -l)"; exit 0; fi
  sleep 300
done
