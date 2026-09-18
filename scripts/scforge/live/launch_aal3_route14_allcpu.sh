#!/usr/bin/env bash
set -euo pipefail

R1_ROOT="${1:-/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch/ad_existing_tracks_20260529T055752Z}"
PY="/home/ec2-user/exp/.venv_connectome_app/bin/python"
LIVE="/home/ec2-user/exp/scripts/scforge/live"
R13_ROOT="${R1_ROOT}_route13_step7_preflight_rebuild"
R14_ROOT="${R1_ROOT}_route14_step7_rebuilt_track_coverage"

ROUTE14_WORKERS="${ROUTE14_WORKERS:-16}"
ROUTE14_THREADS="${ROUTE14_THREADS:-4}"
ROUTE14_SELECT_STREAMLINES="${ROUTE14_SELECT_STREAMLINES:-3000000}"

cd /home/ec2-user/exp
export PYTHONPATH="/home/ec2-user/exp:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MRTRIX_NTHREADS="${ROUTE14_THREADS}"

if [[ ! -s "${R13_ROOT}/subjects.csv" ]]; then
  echo "missing Route13 subjects.csv: ${R13_ROOT}/subjects.csv" >&2
  exit 1
fi

if [[ -d "${R14_ROOT}" && ! -f "${R14_ROOT}/.full38_queue_initialized" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  mv "${R14_ROOT}" "${R14_ROOT}_old_canary_${stamp}"
fi

mkdir -p "${R14_ROOT}"
{
  printf 'sid,group,reason,target_lane\n'
  awk -F, 'NR > 1 && $1 != "" {printf "%s,%s,%s,%s\n", $1, $2, "route14 full Route13 rebuilt Step7 queue", "R14"}' \
    "${R13_ROOT}/subjects.csv"
} > "${R14_ROOT}/route_queue.csv.tmp"
mv "${R14_ROOT}/route_queue.csv.tmp" "${R14_ROOT}/route_queue.csv"
touch "${R14_ROOT}/.full38_queue_initialized"

echo "Route14 all-CPU launch | $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "route1 root: ${R1_ROOT}"
echo "route13 root: ${R13_ROOT}"
echo "route14 root: ${R14_ROOT}"
echo "queue rows: $(($(wc -l < "${R14_ROOT}/route_queue.csv") - 1))"
echo "workers=${ROUTE14_WORKERS} | mrtrix_threads=${ROUTE14_THREADS} | streamlines=${ROUTE14_SELECT_STREAMLINES}"

exec "${PY}" "${LIVE}/run_aal3_route14_step7_rebuilt_track_coverage.py" \
  --route1-root "${R1_ROOT}" \
  --route2-root "${R1_ROOT}_route2" \
  --workers "${ROUTE14_WORKERS}" \
  --mrtrix-threads "${ROUTE14_THREADS}" \
  --select-streamlines "${ROUTE14_SELECT_STREAMLINES}" \
  --track-modes dynamic gmwmi \
  --variants forward80 radial16 endvox
