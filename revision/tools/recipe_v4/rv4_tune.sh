#!/bin/bash
# Keeps the cohort driver at as many parallel sessions as the machine can hold. CPU is the limit; memory and disk are guards. Changes no result: each
# session runs the signed code with its own 8 threads; this only decides how many sessions run at once (GNU xargs: SIGUSR1 one more slot, SIGUSR2 one fewer;
# running sessions are never stopped). Every 10 min: CPU busy under 90% with memory and disk to spare -> one more; short of memory or disk -> one fewer.
# Start detached:  ( setsid nohup bash rv4_tune.sh <xargs pid> <its current -P> >> /data/derivatives/recipe_v4/cohort/logs/tune.log 2>&1 < /dev/null & )
set -u; X=$1; P=$2; MAX=${MAX:-32}; MIN=${MIN:-16}
cpu() { awk 'NR==1{print $2+$3+$4+$5+$6+$7+$8+$9, $5+$6}' /proc/stat; }
read t0 i0 < <(cpu)
echo "$(TZ=Asia/Kolkata date '+%F %H:%M IST') start: xargs $X, sessions $P, max $MAX"
while kill -0 "$X" 2>/dev/null; do
  sleep 600
  read t1 i1 < <(cpu); busy=$(( 100 - 100 * (i1 - i0) / (t1 - t0) )); t0=$t1; i0=$i1
  mem=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); free=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9); act=""
  if [ "$mem" -lt 40 ] || [ "$free" -lt 200 ]; then
    [ "$P" -gt "$MIN" ] && { kill -USR2 "$X"; P=$((P - 1)); act=" -> one fewer"; }
  elif [ "$busy" -lt 90 ] && [ "$mem" -ge 80 ] && [ "$free" -ge 350 ] && [ "$P" -lt "$MAX" ]; then
    kill -USR1 "$X"; P=$((P + 1)); act=" -> one more"
  fi
  echo "$(TZ=Asia/Kolkata date '+%F %H:%M IST') busy ${busy}% mem ${mem}G free ${free}G sessions ${P}${act}"
done
echo "$(TZ=Asia/Kolkata date '+%F %H:%M IST') xargs gone, tuner stops"
