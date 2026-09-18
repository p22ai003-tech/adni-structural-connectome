#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ec2-user/exp"
QC_ROOT="${ROOT}/data/derivatives/qc/sc_matrix_qc"
LATEST_FILE="${QC_ROOT}/scforge_v1_density_latest.txt"
INTERVAL="${INTERVAL:-30}"

bar() {
  local done="$1" total="$2" width="${3:-28}"
  if [[ "${total}" -le 0 ]]; then
    printf '[%*s] 0/0' "${width}" ''
    return
  fi
  local filled=$(( done * width / total ))
  local empty=$(( width - filled ))
  printf '['
  printf '%*s' "${filled}" '' | tr ' ' '#'
  printf '%*s' "${empty}" '' | tr ' ' '.'
  printf '] %s/%s' "${done}" "${total}"
}

latest_root() {
  if [[ -s "${LATEST_FILE}" ]]; then
    cat "${LATEST_FILE}"
    return
  fi
  find "${QC_ROOT}" -maxdepth 1 -type d -name 'scforge_v1_density_batch_*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

while true; do
  clear
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  root="$(latest_root || true)"
  echo "SC-Forge density batch | ${now}"
  if [[ -z "${root}" || ! -d "${root}" ]]; then
    echo "No SC-Forge v1 density run found yet."
    echo "Start with: /home/ec2-user/exp/launch_scforge_v1_density_batch.sh"
    sleep "${INTERVAL}"
    continue
  fi

  status="${root}/status.json"
  if [[ -f "${status}" ]]; then
    python - "${status}" "${root}" <<'PY'
import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
path = Path(sys.argv[1])
root = Path(sys.argv[2])
try:
    s = json.loads(path.read_text())
except Exception as exc:
    print(f"status read failed: {exc}")
    raise SystemExit

def fnum(value):
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except Exception:
        return math.nan

def fint(value, default=0):
    value = fnum(value)
    return int(value) if math.isfinite(value) else default

def fmt(value, digits=3):
    value = fnum(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else "-"

def fmt_candidate_density(value):
    value = fnum(value)
    if not math.isfinite(value) or value < 0:
        return "NA"
    return f"{value:.3f}"

def fmt_zero(value):
    out = fint(value, default=9999)
    return "NA" if out >= 9999 else str(out)

def parse_utc(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None

phase = s.get("phase", "unknown")
sid = s.get("current_sid") or s.get("latest_sid") or ""
done = fint(s.get("completed_subjects"))
total = fint(s.get("total_subjects"))
remaining = max(total - done, 0)
ret = fint(s.get("retained_candidate_count"))
cur = fint(s.get("retained_current_count"))
qua = fint(s.get("quarantine_count"))
fail = fint(s.get("failed_subjects"))
percent = (100.0 * done / total) if total else 0.0
print(f"Run: {root.name}")
print(f"Phase: {phase.replace('_', ' ')}")
print(f"Progress: {done}/{total} ({percent:.1f}%) | remaining={remaining}")
print(f"Decisions: candidate={ret} | current={cur} | quarantine={qua} | failed={fail}")
workers = s.get("workers", "")
threads = s.get("mrtrix_threads", "")
print(f"Workers: {workers} subjects x {threads} MRtrix threads | active/latest={sid or '-'}")

events = []
log_path = root / "subject_stage_log.csv"
if log_path.exists():
    with log_path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            event_time = parse_utc(row[0])
            if event_time and row[2] in {"decision", "probe_failed"}:
                events.append(event_time)

now = datetime.now(timezone.utc)
recent_1h = [t for t in events if t >= now - timedelta(hours=1)]
recent_30m = [t for t in events if t >= now - timedelta(minutes=30)]
rate_1h = len(recent_1h)
rate_30m = len(recent_30m) * 2
if rate_1h > 0:
    eta_hours = remaining / rate_1h
    print(f"Rate: {rate_1h}/hr last 1h | {rate_30m}/hr last 30m | ETA about {eta_hours:.1f} hr")
else:
    print("Rate: waiting for enough recent completed subjects")

print(
    "latest: "
    f"decision={s.get('latest_decision','')} | "
    f"density {fmt(s.get('latest_baseline_density'))}->{fmt_candidate_density(s.get('latest_candidate_density'))} | "
    f"zero {fmt_zero(s.get('latest_baseline_zero_rows'))}->{fmt_zero(s.get('latest_candidate_zero_rows'))}"
)
print(f"updated: {s.get('last_update_utc','')}")
PY
    done="$(python -c "import json; s=json.load(open('${status}')); print(int(float(s.get('completed_subjects',0) or 0)))" 2>/dev/null || echo 0)"
    total="$(python -c "import json; s=json.load(open('${status}')); print(int(float(s.get('total_subjects',0) or 0)))" 2>/dev/null || echo 0)"
    echo -n "progress "
    bar "${done}" "${total}"
    echo
  else
    echo "status.json missing"
  fi

  echo
  echo "active subjects"
  mapfile -t active_sids < <(
    pgrep -af "run_sc_aal3_source_contract_probe.py|/tck2connectome|/flirt" \
      | sed -nE \
          -e 's/.*--sid ([^ ]+).*/\1/p' \
          -e 's#.*/tracks/([^/ ]+)/.*#\1#p' \
          -e 's#.*scforge_v1_density_batch_[0-9TZ]+_([0-9]{3}_S_[0-9]{4}_I[0-9]+).*#\1#p' \
      | sort -u
  )
  if [[ "${#active_sids[@]}" -eq 0 ]]; then
    echo "  none"
  else
    printf '  %s\n' "${active_sids[@]}" | head -12
  fi
  probes="$(pgrep -fc "run_sc_aal3_source_contract_probe.py" || true)"
  tck="$(pgrep -fc "/tck2connectome" || true)"
  flirt="$(pgrep -fc "/flirt" || true)"
  echo "tools: probes=${probes} | tck2connectome=${tck} | flirt=${flirt}"

  echo
  echo "recent events"
  if [[ -f "${root}/subject_stage_log.csv" ]]; then
    python - "${root}/subject_stage_log.csv" <<'PY'
import csv
import sys
from collections import deque
from pathlib import Path

path = Path(sys.argv[1])
rows = deque(maxlen=8)
with path.open(newline="") as handle:
    for row in csv.reader(handle):
        if len(row) >= 3:
            rows.append(row)

for row in rows:
    when = row[0][11:19] if "T" in row[0] else row[0]
    sid = row[1]
    event = row[2]
    if event == "start":
        base = row[3] if len(row) > 3 else "-"
        zero = row[4] if len(row) > 4 else "-"
        print(f"  {when} {sid} start | density={base[:6]} zero={zero}")
    elif event == "decision":
        decision = row[3] if len(row) > 3 else "-"
        base = row[4] if len(row) > 4 else "-"
        cand = row[5] if len(row) > 5 else "-"
        print(f"  {when} {sid} {decision} | {base[:6]}->{cand[:6]}")
    elif event == "full_candidate_done":
        variant = row[3] if len(row) > 3 else "-"
        print(f"  {when} {sid} candidate done | {variant}")
    elif event == "probe_failed":
        rc = row[3] if len(row) > 3 else "-"
        print(f"  {when} {sid} failed | rc={rc}")
    else:
        print(f"  {when} {sid} {event}")
PY
  else
    echo "  subject_stage_log.csv not written yet"
  fi

  echo
  echo "attach: tmux attach -t scforge_v1_density_monitor | refresh=${INTERVAL}s"
  echo "run root: ${root}"
  sleep "${INTERVAL}"
done
