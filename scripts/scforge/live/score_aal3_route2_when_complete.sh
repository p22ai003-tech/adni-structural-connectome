#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch/ad_existing_tracks_20260529T055752Z_route2}"
PY="/home/ec2-user/exp/.venv_connectome_app/bin/python"
SCORER="/home/ec2-user/exp/scripts/scforge/live/score_aal3_force_connectome_qc.py"
LEDGER="/home/ec2-user/exp/scripts/scforge/live/summarize_aal3_route_ledger.py"
LAST_MTIME=""

refresh_ledger() {
  local r1_root="${RUN_ROOT%_route2}"
  if [[ -s "$r1_root/subjects.csv" ]]; then
    "$PY" "$LEDGER" --run-root "$r1_root" \
      > "$r1_root/route_subject_ledger_stdout.log" \
      2> "$r1_root/route_subject_ledger_stderr.log" || true
  fi
}

while true; do
  results="$RUN_ROOT/batch_results.csv"
  if [[ -s "$results" ]]; then
    mtime="$(stat -c %Y "$results" 2>/dev/null || echo "")"
    if [[ -n "$mtime" && "$mtime" != "$LAST_MTIME" ]]; then
      cd /home/ec2-user/exp
      "$PY" "$SCORER" --run-root "$RUN_ROOT" > "$RUN_ROOT/score_stdout.log" 2> "$RUN_ROOT/score_stderr.log" || true
      refresh_ledger
      LAST_MTIME="$mtime"
    fi
  fi
  phase="$("$PY" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "status.json"
try:
    print(json.loads(path.read_text()).get("phase", ""))
except Exception:
    print("")
PY
)"
  if [[ "$phase" == "complete" ]]; then
    cd /home/ec2-user/exp
    "$PY" "$SCORER" --run-root "$RUN_ROOT" > "$RUN_ROOT/score_stdout.log" 2> "$RUN_ROOT/score_stderr.log" || true
    refresh_ledger
    exit 0
  fi
  sleep 60
done
