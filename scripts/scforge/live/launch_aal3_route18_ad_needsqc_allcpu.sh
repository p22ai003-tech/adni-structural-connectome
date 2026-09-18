#!/usr/bin/env bash
set -euo pipefail

R1_ROOT="${1:-/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch/ad_existing_tracks_20260529T055752Z}"
PY="/home/ec2-user/exp/.venv_connectome_app/bin/python"
LIVE="/home/ec2-user/exp/scripts/scforge/live"

ROUTE18_WORKERS="${ROUTE18_WORKERS:-16}"
ROUTE18_THREADS="${ROUTE18_THREADS:-4}"
ROUTE18_SELECT_STREAMLINES="${ROUTE18_SELECT_STREAMLINES:-6000000}"
ROUTE18_LIMIT="${ROUTE18_LIMIT:-0}"

LEDGER="${R1_ROOT}/route_subject_ledger.csv"
if [[ ! -s "${LEDGER}" ]]; then
  echo "missing route ledger: ${LEDGER}" >&2
  exit 1
fi

mapfile -t SUBJECTS < <(
  "${PY}" - "${LEDGER}" "${ROUTE18_LIMIT}" <<'PY'
import csv
import sys

ledger = sys.argv[1]
limit = int(sys.argv[2])
subjects = []
with open(ledger, newline="", encoding="utf-8", errors="replace") as handle:
    for row in csv.DictReader(handle):
        if row.get("group") != "AD":
            continue
        if row.get("final_qc_status") == "solved":
            continue
        routes = {part.strip() for part in (row.get("routes_tested") or "").split(",") if part.strip()}
        if "R18" in routes:
            continue
        subjects.append(row["sid"])
subjects = sorted(set(subjects))
if limit > 0:
    subjects = subjects[:limit]
for subject in subjects:
    print(subject)
PY
)

if [[ "${#SUBJECTS[@]}" -eq 0 ]]; then
  echo "no AD needs-QC subjects without R18 remaining"
  exit 0
fi

cd /home/ec2-user/exp
export PYTHONPATH="/home/ec2-user/exp:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MRTRIX_NTHREADS="${ROUTE18_THREADS}"

echo "Route18 AD needs-QC all-CPU launch | $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "route1 root: ${R1_ROOT}"
echo "subjects: ${#SUBJECTS[@]}"
echo "workers=${ROUTE18_WORKERS} | mrtrix_threads=${ROUTE18_THREADS} | streamlines=${ROUTE18_SELECT_STREAMLINES}"
printf 'first subjects:'
printf ' %s' "${SUBJECTS[@]:0:12}"
printf '\n'

exec "${PY}" "${LIVE}/run_aal3_route18_gmwmi_dense_track_coverage.py" \
  --route1-root "${R1_ROOT}" \
  --route2-root "${R1_ROOT}_route2" \
  --workers "${ROUTE18_WORKERS}" \
  --mrtrix-threads "${ROUTE18_THREADS}" \
  --select-streamlines "${ROUTE18_SELECT_STREAMLINES}" \
  --subjects "${SUBJECTS[@]}" \
  --track-modes gmwmi dynamic \
  --variants forward80 radial16 endvox
