#!/bin/bash
# On the main server: copies finished sessions back from a copy of it (FREEZE.md decisions 5-6), every 20 min, until every session on that copy's list
# is back. A session counts as finished on the copy when both its QC files say ok and its FOD is pruned; data first, QC files last, so a QC file
# on the main server always means the data beside it is complete. Only sessions on that copy's list are copied.
# Usage: rv4_pull.sh <copy's private IP> <copy number> <its units file>
# Start detached:  ( setsid nohup bash rv4_pull.sh IP k FILE >> /data/derivatives/recipe_v4/cohort/logs/pull_k.log 2>&1 < /dev/null & )
set -u; IP=$1; K=$2; LIST=$3; B=/data/derivatives/recipe_v4/cohort; KEY=/data/derivatives/recipe_v4/keys
SSH="ssh -i $KEY/clone_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KEY/known_hosts -o BatchMode=yes -o ConnectTimeout=20"
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M IST') $*"; }
mkdir -p "$B/logs/clone_$K"
while :; do
  todo=$(while read -r u; do [ -f "$B/qc/C_noact_$u.json" ] || [ -f "$B/logs/clone_$K/failed_$u" ] || echo "$u"; done < "$LIST")
  [ -z "$todo" ] && { say "copy $K: all $(wc -l < "$LIST") sessions back"; exit 0; }
  ready=$($SSH "ec2-user@$IP" "cd $B; for u in $(echo $todo); do
      a=qc/C_\$u.json; n=qc/C_noact_\$u.json
      if [ -f \$a ] && grep -q '\"status\": \"failed\"' \$a && ! pgrep -f \"worker \$u\" >/dev/null; then echo \"failed \$u\";
      elif [ -f \$n ] && grep -q '\"status\": \"ok\"' \$a && grep -q '\"status\": \"ok\"' \$n && [ ! -e work/C/\$u/wmfod.mif ]; then echo \"ok \$u\"; fi; done") || { say "copy $K: ssh failed"; sleep 300; continue; }
  n=0
  while read -r s u; do
    [ -z "${u:-}" ] && continue
    for d in work/C work/C_noact work/_shared; do rsync -a -e "$SSH" "ec2-user@$IP:$B/$d/$u/" "$B/$d/$u/" 2>/dev/null; done
    rsync -a -e "$SSH" "ec2-user@$IP:$B/qc/C_$u.json" "$B/qc/C_$u.json" || continue
    if [ "$s" = failed ]; then touch "$B/logs/clone_$K/failed_$u"; else rsync -a -e "$SSH" "ec2-user@$IP:$B/qc/C_noact_$u.json" "$B/qc/C_noact_$u.json" || continue; fi
    n=$((n + 1))
  done <<< "$ready"
  rsync -a -e "$SSH" "ec2-user@$IP:$B/logs/" "$B/logs/clone_$K/" 2>/dev/null
  say "copy $K: $n sessions copied back this round; $(echo "$todo" | wc -w) were outstanding"
  sleep 1200
done
