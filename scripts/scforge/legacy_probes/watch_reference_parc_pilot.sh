#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${INTERVAL:-30}"
ROOT="/home/ec2-user/exp"
QC_ROOT="$ROOT/data/derivatives/qc/sc_matrix_qc/registration_parc_pilot"

bar() {
  local done="$1" total="$2" width="${3:-28}"
  python3 - "$done" "$total" "$width" <<'PY'
import sys
done=int(float(sys.argv[1] or 0)); total=int(float(sys.argv[2] or 0)); width=int(sys.argv[3])
frac=0 if total <= 0 else max(0, min(1, done / total))
fill=round(frac * width)
print("[" + "#"*fill + "."*(width-fill) + f"] {frac*100:5.1f}% {done}/{total}")
PY
}

latest_run() {
  find "$QC_ROOT" -maxdepth 2 -name reference_parc_pilot_status.json -type f 2>/dev/null \
    -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}'
}

render_once() {
  clear 2>/dev/null || true
  printf 'Reference-route parcellation pilot | %s UTC\n' "$(date -u +%Y-%m-%dT%H:%M:%S)"
  printf 'QC root: %s\n\n' "$QC_ROOT"

  local status_file
  status_file="$(latest_run || true)"
  if [[ -z "${status_file:-}" ]]; then
    echo "No reference parcellation pilot status found yet."
    echo "Expected after launch: $QC_ROOT/<tag>/reference_parc_pilot_status.json"
    return
  fi

  python3 - "$status_file" <<'PY'
import csv, json, os, sys
from pathlib import Path

status_path = Path(sys.argv[1])
status = json.loads(status_path.read_text())
run_root = Path(status.get("run_root", status_path.parent))
print(f"tag     {status.get('tag','')}")
print(f"run     {run_root}")
print(f"stage   {status.get('current_stage','')}")
print(f"subject {status.get('current_sid','') or '-'}")
print(f"done    {status.get('done', False)}")
print()
total = int(status.get("total_subjects", 0) or 0)
completed = int(status.get("completed_subjects", 0) or 0)
failed = int(status.get("failed_subjects", 0) or 0)
print(f"progress completed={completed} failed={failed} total={total}")

summary = run_root / "reference_parcellation_pilot_summary.csv"
if summary.exists():
    rows = list(csv.DictReader(summary.open()))
    if rows:
        print("\nlatest decisions")
        for row in rows[-6:]:
            print(
                f"  {row.get('sid',''):<24} {row.get('decision',''):<20} "
                f"best={row.get('best_route',''):<38} "
                f"density {row.get('baseline_density','?')} -> {row.get('best_density','?')} "
                f"zero_rows {row.get('baseline_unexpected_valid_zero_rows','?')} -> {row.get('best_unexpected_valid_zero_rows','?')}"
            )

failures = run_root / "reference_parcellation_pilot_failures.csv"
if failures.exists():
    rows = [r for r in csv.DictReader(failures.open()) if r.get("sid")]
    if rows:
        print("\nfailures")
        for row in rows[-5:]:
            print(f"  {row.get('sid','')}: {row.get('error','')[:180]}")
PY

  echo
  bar "$(python3 - "$status_file" <<'PY'
import json, sys
s=json.load(open(sys.argv[1])); print(s.get("completed_subjects",0)+s.get("failed_subjects",0))
PY
)" "$(python3 - "$status_file" <<'PY'
import json, sys
s=json.load(open(sys.argv[1])); print(s.get("total_subjects",0))
PY
)"

  echo
  echo "active processes"
  ps -eo pid,pcpu,pmem,etime,comm,args \
    | awk '$5 !~ /^(tmux|bash|sh|tee|awk|grep)$/ && $0 ~ /(reference_parcellation_pilot.py|flirt|convert_xfm|tck2connectome|mrconvert)/ {print}' \
    | head -20 || true
  if ! ps -eo comm,args \
    | awk '$1 !~ /^(tmux|bash|sh|tee|awk|grep)$/ && $0 ~ /(reference_parcellation_pilot.py|flirt|convert_xfm|tck2connectome|mrconvert)/ {found=1} END {exit found?0:1}'; then
    echo "  none"
  fi
  echo
  echo "attach runner : tmux attach -t reference_parc_pilot"
  echo "attach monitor: tmux attach -t reference_parc_pilot_monitor"
}

while true; do
  render_once
  printf '\nrefresh=%ss\n' "$INTERVAL"
  sleep "$INTERVAL"
done
