#!/usr/bin/env bash
set -euo pipefail

R1_ROOT="${1:-/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch/ad_existing_tracks_20260529T055752Z}"
PY="/home/ec2-user/exp/.venv_connectome_app/bin/python"
LIVE="/home/ec2-user/exp/scripts/scforge/live"

ROUTE19_WORKERS="${ROUTE19_WORKERS:-4}"
ROUTE19_THREADS="${ROUTE19_THREADS:-4}"
ROUTE19_SELECT_STREAMLINES="${ROUTE19_SELECT_STREAMLINES:-3000000}"
ROUTE19_LIMIT="${ROUTE19_LIMIT:-0}"

R18_RESULTS="${R1_ROOT}_route18_gmwmi_dense_track_coverage/batch_results.csv"
if [[ ! -s "${R18_RESULTS}" ]]; then
  echo "missing Route18 batch results: ${R18_RESULTS}" >&2
  exit 1
fi

mapfile -t SUBJECTS < <(
  "${PY}" - "${R18_RESULTS}" "${ROUTE19_LIMIT}" <<'PY'
import csv
import sys

path = sys.argv[1]
limit = int(sys.argv[2])
subjects = []
with open(path, newline="", encoding="utf-8", errors="replace") as handle:
    for row in csv.DictReader(handle):
        if row.get("status") == "failed":
            subjects.append(row.get("sid", ""))
subjects = sorted({sid for sid in subjects if sid})
if limit > 0:
    subjects = subjects[:limit]
for subject in subjects:
    print(subject)
PY
)

if [[ "${#SUBJECTS[@]}" -eq 0 ]]; then
  echo "no failed Route18 subjects available for Route19"
  exit 0
fi

cd /home/ec2-user/exp
export PYTHONPATH="/home/ec2-user/exp:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MRTRIX_NTHREADS="${ROUTE19_THREADS}"

echo "Route19 GMWMI 3M retry launch | $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "route1 root: ${R1_ROOT}"
echo "source: failed Route18 subjects"
echo "subjects: ${#SUBJECTS[@]}"
echo "workers=${ROUTE19_WORKERS} | mrtrix_threads=${ROUTE19_THREADS} | streamlines=${ROUTE19_SELECT_STREAMLINES}"
printf 'subjects:'
printf ' %s' "${SUBJECTS[@]}"
printf '\n'

exec "${PY}" "${LIVE}/run_aal3_route19_gmwmi_3m_retry.py" \
  --route1-root "${R1_ROOT}" \
  --route2-root "${R1_ROOT}_route2" \
  --workers "${ROUTE19_WORKERS}" \
  --mrtrix-threads "${ROUTE19_THREADS}" \
  --select-streamlines "${ROUTE19_SELECT_STREAMLINES}" \
  --subjects "${SUBJECTS[@]}" \
  --track-modes gmwmi dynamic \
  --variants forward80 radial16 endvox
