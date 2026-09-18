#!/usr/bin/env bash
set -euo pipefail

R1_ROOT="${1:-/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch/ad_existing_tracks_20260529T055752Z}"
PY="/home/ec2-user/exp/.venv_connectome_app/bin/python"
LIVE="/home/ec2-user/exp/scripts/scforge/live"

ROUTE17_WORKERS="${ROUTE17_WORKERS:-2}"
ROUTE17_THREADS="${ROUTE17_THREADS:-16}"
ROUTE17_SELECT_STREAMLINES="${ROUTE17_SELECT_STREAMLINES:-12000000}"

SUBJECTS=(
  027_S_4938_I394341
  027_S_6849_I1287078
  127_S_6433_I1351391
  135_S_6545_I1226546
)

cd /home/ec2-user/exp
export PYTHONPATH="/home/ec2-user/exp:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MRTRIX_NTHREADS="${ROUTE17_THREADS}"

echo "Route17 dense unpushed launch | $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "route1 root: ${R1_ROOT}"
echo "subjects: ${SUBJECTS[*]}"
echo "workers=${ROUTE17_WORKERS} | mrtrix_threads=${ROUTE17_THREADS} | streamlines=${ROUTE17_SELECT_STREAMLINES}"

exec "${PY}" "${LIVE}/run_aal3_route17_step7_dense_unpushed.py" \
  --route1-root "${R1_ROOT}" \
  --route2-root "${R1_ROOT}_route2" \
  --workers "${ROUTE17_WORKERS}" \
  --mrtrix-threads "${ROUTE17_THREADS}" \
  --select-streamlines "${ROUTE17_SELECT_STREAMLINES}" \
  --subjects "${SUBJECTS[@]}" \
  --track-modes dynamic \
  --variants forward80
