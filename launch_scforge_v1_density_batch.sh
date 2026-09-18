#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ec2-user/exp"
PY="/home/ec2-user/fsl/bin/python"
RUNNER="${ROOT}/run_scforge_v1_density_batch.py"
MONITOR="${ROOT}/watch_scforge_v1_density_monitor.sh"
TAG="${SCFORGE_V1_TAG:-scforge_v1_density_batch_$(date -u +%Y%m%dT%H%M%SZ)}"
PHASE="${SCFORGE_V1_PHASE:-all}"
INTERVAL="${INTERVAL:-30}"
WORKERS="${SCFORGE_V1_WORKERS:-1}"
MRTRIX_THREADS="${SCFORGE_V1_MRTRIX_THREADS:-0}"
RUN_ROOT="${ROOT}/data/derivatives/qc/sc_matrix_qc/${TAG}"

mkdir -p "${RUN_ROOT}"
touch "${RUN_ROOT}/stdout.log" "${RUN_ROOT}/stderr.log" "${RUN_ROOT}/commands.log"

tmux kill-session -t scforge_v1_density_batch 2>/dev/null || true
tmux kill-session -t scforge_v1_density_monitor 2>/dev/null || true

tmux new-session -d -s scforge_v1_density_batch \
  "cd '${ROOT}' && PYTHONPATH='${ROOT}/scforge:${ROOT}' '${PY}' '${RUNNER}' --tag '${TAG}' --phase '${PHASE}' --execute --workers '${WORKERS}' --mrtrix-threads '${MRTRIX_THREADS}' >> '${RUN_ROOT}/stdout.log' 2>> '${RUN_ROOT}/stderr.log'"

tmux new-session -d -s scforge_v1_density_monitor \
  "cd '${ROOT}' && INTERVAL='${INTERVAL}' '${MONITOR}'"

echo "SC-Forge v1 density batch started"
echo "tag: ${TAG}"
echo "workers: ${WORKERS}"
echo "mrtrix threads/subject: ${MRTRIX_THREADS}"
echo "runner:  tmux attach -t scforge_v1_density_batch"
echo "monitor: tmux attach -t scforge_v1_density_monitor"
