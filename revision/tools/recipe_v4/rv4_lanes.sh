#!/bin/bash
# Runs the signed per-session worker (`rv4_cohort.sh worker`, unchanged) over an explicit list of sessions, so that the rest of the cohort can be split
# across machines mid-run (FREEZE.md, 5 Nov). Same worker, settings and results; only which sessions this machine takes changes. Keeps TARGET sessions
# running on this machine, counting any still running from an earlier driver, and adds a slot only with memory to spare.
# Usage: rv4_lanes.sh <units file> [target, default 32]
# Start detached:  ( setsid nohup bash rv4_lanes.sh FILE 32 >> /data/derivatives/recipe_v4/cohort/logs/lanes.log 2>&1 < /dev/null & )
# Afterwards, on the main server only: `rv4_cohort.sh all` once, which skips every finished session and writes flags.csv and DONE.
set -u; LIST=$1; TARGET=${2:-32}
T=/home/ec2-user/exp/docs/iit_dipanjan/tools/recipe_v4; export B=/data/derivatives/recipe_v4/cohort
cd /data/derivatives/recipe_v4/scratch || exit 1
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M IST') $*"; }
running() { ps -eo args | grep -c '^bash [^ ]*rv4_cohort\.sh worker '; }   # session processes only (not xargs, whose arguments also name the worker)
while [ "$(running)" -ge "$TARGET" ]; do sleep 30; done                # machine already full: wait for a free slot before starting
P=$(( TARGET - $(running) )); [ "$P" -lt 1 ] && P=1
xargs -a "$LIST" -P "$P" -L 1 nice -n 19 bash "$T/rv4_cohort.sh" worker &
X=$!
say "lanes start: $(wc -l < "$LIST") sessions from $LIST, xargs $X, $P slots, target $TARGET running"
while kill -0 "$X" 2>/dev/null; do
  sleep 60
  mem=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$(running)" -lt "$TARGET" ] && [ "$mem" -ge 60 ] && [ "$P" -lt "$TARGET" ]; then kill -USR1 "$X"; P=$((P + 1)); say "slots $P"; fi
done
wait "$X"; say "lanes: list finished"
