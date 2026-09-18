#!/usr/bin/env bash
set -euo pipefail

R1_ROOT="${1:-/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch/ad_existing_tracks_20260529T055752Z}"
PY="/home/ec2-user/exp/.venv_connectome_app/bin/python"
LIVE="/home/ec2-user/exp/scripts/scforge/live"

ROUTE15_WORKERS="${ROUTE15_WORKERS:-2}"
ROUTE15_THREADS="${ROUTE15_THREADS:-24}"
ROUTE15_SELECT_STREAMLINES="${ROUTE15_SELECT_STREAMLINES:-12000000}"

SUBJECTS=(
  <SUBJECT>_I<IMAGEID>
  <SUBJECT>_I<IMAGEID>
  <SUBJECT>_I<IMAGEID>
  <SUBJECT>_I<IMAGEID>
  <SUBJECT>_I<IMAGEID>
  <SUBJECT>_I<IMAGEID>
  <SUBJECT>_I<IMAGEID>
)

cd /home/ec2-user/exp
export PYTHONPATH="/home/ec2-user/exp:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MRTRIX_NTHREADS="${ROUTE15_THREADS}"

echo "Route15 dense interim launch | $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "route1 root: ${R1_ROOT}"
echo "subjects: ${SUBJECTS[*]}"
echo "workers=${ROUTE15_WORKERS} | mrtrix_threads=${ROUTE15_THREADS} | streamlines=${ROUTE15_SELECT_STREAMLINES}"

exec "${PY}" "${LIVE}/run_aal3_route15_step7_dense_track_coverage.py" \
  --route1-root "${R1_ROOT}" \
  --route2-root "${R1_ROOT}_route2" \
  --workers "${ROUTE15_WORKERS}" \
  --mrtrix-threads "${ROUTE15_THREADS}" \
  --select-streamlines "${ROUTE15_SELECT_STREAMLINES}" \
  --subjects "${SUBJECTS[@]}" \
  --track-modes dynamic \
  --variants forward80 radial16 endvox
