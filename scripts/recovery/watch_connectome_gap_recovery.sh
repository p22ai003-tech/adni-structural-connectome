#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-}"
INTERVAL="${INTERVAL:-0}"
DERIV_ROOT="${DERIV_ROOT:-/home/ec2-user/exp/data/derivatives}"
LATEST_FILE="$DERIV_ROOT/qc/connectome_gap_recovery_latest.txt"

if [[ -z "$RUN_ROOT" && -f "$LATEST_FILE" ]]; then
  RUN_ROOT="$(cat "$LATEST_FILE")"
fi
if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="$(ls -td "$DERIV_ROOT"/qc/connectome_gap_recovery_* 2>/dev/null | head -n 1 || true)"
fi

bar() {
  local done="$1" total="$2" width="${3:-28}"
  python3 - "$done" "$total" "$width" <<'PY'
import sys
done=int(sys.argv[1]); total=max(int(sys.argv[2]),1); width=int(sys.argv[3])
n=round(width*done/total)
print("[" + "#"*n + "."*(width-n) + f"] {100*done/total:5.1f}% {done}/{total}")
PY
}

render() {
  local now targets total
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  clear || true
  echo "Targeted CN/MCI connectome gap recovery | $now"
  echo
  if [[ -z "$RUN_ROOT" || ! -d "$RUN_ROOT" ]]; then
    echo "No run root found. Set RUN_ROOT=/path/to/qc/connectome_gap_recovery_*"
    return
  fi
  targets="$RUN_ROOT/targets.txt"
  echo "run_root: $RUN_ROOT"
  if [[ ! -f "$targets" ]]; then
    echo "targets.txt not found yet"
    return
  fi
  total="$(grep -cv '^[[:space:]]*$' "$targets" || true)"
  python3 - "$DERIV_ROOT" "$RUN_ROOT" <<'PY'
from __future__ import annotations
import json
from pathlib import Path
import sys

deriv=Path(sys.argv[1])
run_root=Path(sys.argv[2])
targets=[x.strip() for x in (run_root/"targets.txt").read_text().splitlines() if x.strip()]
suffixes=("ALL","count","fd_sum","len_mean","invlen_mean","fa_mean","md_mean","rd_mean","ad_mean","count_invnodevol")
meta={}
if (run_root/"run_metadata.json").exists():
    meta=json.loads((run_root/"run_metadata.json").read_text())
target_group={}
inv=run_root/"target_inventory.csv"
if inv.exists():
    import csv
    with inv.open() as f:
        for row in csv.DictReader(f):
            target_group[row["sid"]]=row.get("group","")
groups=["CN","MCI","AD"]
rows=[]
for group in groups:
    group_targets=[sid for sid in targets if target_group.get(sid)==group]
    if not group_targets:
        continue
    full=0
    all_file=0
    for sid in group_targets:
        if (deriv/"connectomes"/f"SC_AAL_{sid}_ALL.csv").exists():
            all_file += 1
        if all((deriv/"connectomes"/f"SC_AAL_{sid}_{s}.csv").exists() for s in suffixes):
            full += 1
    baseline=int(meta.get("baseline_complete_post_by_group",{}).get(group,0) or 0)
    rows.append((group, len(group_targets), all_file, full, baseline, baseline + full))
overall_full=sum(row[3] for row in rows)
overall_total=sum(row[1] for row in rows)
print(f"overall full post set: {overall_full}/{overall_total}")
for group,total,all_file,full,baseline,expected in rows:
    print(f"{group:>3} targets={total:3d} | ALL={all_file:3d} | full={full:3d} | expected connectomes={expected:3d}")
PY
  echo
  read -r full_done <<<"$(python3 - "$DERIV_ROOT" "$targets" <<'PY'
from pathlib import Path
import sys
deriv=Path(sys.argv[1]); targets=[x.strip() for x in Path(sys.argv[2]).read_text().splitlines() if x.strip()]
suffixes=("ALL","count","fd_sum","len_mean","invlen_mean","fa_mean","md_mean","rd_mean","ad_mean","count_invnodevol")
full=0
for sid in targets:
    if all((deriv/"connectomes"/f"SC_AAL_{sid}_{s}.csv").exists() for s in suffixes):
        full += 1
print(full)
PY
)"
  echo -n "full post set "; bar "$full_done" "$total"
  echo
  echo "active processes"
  pgrep -af 'run_connectome_gap_recovery|connectome_step7|tck2connectome|tcksample' | grep -v 'watch_connectome_gap_recovery' || echo "  none"
  echo
  echo "recent log"
  if [[ -f "$RUN_ROOT/recovery.log" ]]; then
    tail -n 18 "$RUN_ROOT/recovery.log"
  else
    echo "  no recovery.log yet"
  fi
  echo
  echo "attach run:     tmux attach -t connectome_gap_recovery"
  echo "attach monitor: tmux attach -t connectome_gap_recovery_monitor"
  echo "one-shot:       RUN_ROOT=\"$RUN_ROOT\" /home/ec2-user/exp/scripts/recovery/watch_connectome_gap_recovery.sh"
}

if [[ "$INTERVAL" == "0" ]]; then
  render
else
  while true; do
    render
    sleep "$INTERVAL"
  done
fi
