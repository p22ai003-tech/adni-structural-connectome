#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-}"
INTERVAL="${INTERVAL:-0}"
DERIV_ROOT="${DERIV_ROOT:-/home/ec2-user/exp/data/derivatives}"

if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="$(ls -td "$DERIV_ROOT"/qc/ad_connectome_recovery_* 2>/dev/null | head -n 1 || true)"
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
  local now targets total all_done req_done running fail
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  clear || true
  echo "AD targeted connectome recovery | $now"
  echo
  if [[ -z "$RUN_ROOT" || ! -d "$RUN_ROOT" ]]; then
    echo "No run root found. Set RUN_ROOT=/path/to/qc/ad_connectome_recovery_*"
    return
  fi
  targets="$RUN_ROOT/targets.txt"
  echo "run_root: $RUN_ROOT"
  if [[ ! -f "$targets" ]]; then
    echo "targets.txt not found yet"
    return
  fi
  total="$(grep -cv '^[[:space:]]*$' "$targets" || true)"
  read -r all_done req_done fail <<<"$(python3 - "$DERIV_ROOT" "$targets" <<'PY'
from pathlib import Path
import sys
deriv=Path(sys.argv[1]); targets=Path(sys.argv[2]).read_text().splitlines()
targets=[x.strip() for x in targets if x.strip()]
suffixes=("ALL","count","fd_sum","len_mean","invlen_mean","fa_mean","md_mean","rd_mean","ad_mean","count_invnodevol")
all_done=0; req_done=0
for sid in targets:
    if (deriv/"connectomes"/f"SC_AAL_{sid}_ALL.csv").exists():
        all_done += 1
    if all((deriv/"connectomes"/f"SC_AAL_{sid}_{s}.csv").exists() for s in suffixes):
        req_done += 1
print(all_done, req_done, 0)
PY
)"
  echo -n "ALL files     "; bar "$all_done" "$total"
  echo -n "full post set "; bar "$req_done" "$total"
  echo
  echo "expected AD connectome count if full post set reaches target: 77 + $req_done / 23"
  echo
  echo "active processes"
  pgrep -af 'run_ad_connectome_recovery|connectome_step7|tck2connectome|tcksample' | grep -v 'watch_ad_connectome_recovery' || echo "  none"
  echo
  echo "recent log"
  if [[ -f "$RUN_ROOT/recovery.log" ]]; then
    tail -n 18 "$RUN_ROOT/recovery.log"
  else
    echo "  no recovery.log yet"
  fi
  echo
  echo "attach run:     tmux attach -t ad_connectome_recovery"
  echo "attach monitor: tmux attach -t ad_connectome_recovery_monitor"
}

if [[ "$INTERVAL" == "0" ]]; then
  render
else
  while true; do
    render
    sleep "$INTERVAL"
  done
fi
