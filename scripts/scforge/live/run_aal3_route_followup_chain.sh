#!/usr/bin/env bash
set -euo pipefail

R1_ROOT="${1:-/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch/ad_existing_tracks_20260529T055752Z}"
WORKERS="${WORKERS:-8}"
MRTRIX_THREADS="${MRTRIX_THREADS:-4}"
ROUTE7_WORKERS="${ROUTE7_WORKERS:-4}"
ROUTE7_THREADS="${ROUTE7_THREADS:-3}"
SELECT_STREAMLINES="${SELECT_STREAMLINES:-3000000}"

PY="/home/ec2-user/exp/.venv_connectome_app/bin/python"
LIVE="/home/ec2-user/exp/scripts/scforge/live"
R2_ROOT="${R1_ROOT}_route2"
R3_ROOT="${R1_ROOT}_route3_assignment"
R4_ROOT="${R1_ROOT}_route4_label_rescue"
R5_ROOT="${R1_ROOT}_route5_deep_assignment"
R6_ROOT="${R1_ROOT}_route6_all_voxels_diagnostic"
R7_ROOT="${R1_ROOT}_route7_selective_track_rescue"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

wait_for_process_pattern_to_end() {
  local label="$1"
  local pattern="$2"
  while pgrep -f "$pattern" >/dev/null; do
    log "waiting for ${label}"
    sleep 60
  done
}

refresh_ledger() {
  if [[ -s "${R1_ROOT}/subjects.csv" ]]; then
    log "refreshing consolidated route ledger"
    "${PY}" "${LIVE}/summarize_aal3_route_ledger.py" --run-root "${R1_ROOT}" \
      > "${R1_ROOT}/route_subject_ledger_stdout.log" \
      2> "${R1_ROOT}/route_subject_ledger_stderr.log" || true
  fi
}

run_score() {
  local root="$1"
  if [[ -s "${root}/batch_results.csv" ]]; then
    log "scoring ${root}"
    "${PY}" "${LIVE}/score_aal3_force_connectome_qc.py" --run-root "${root}" \
      > "${root}/score_stdout.log" 2> "${root}/score_stderr.log" || true
    refresh_ledger
  fi
}

run_route2_catchup() {
  log "running route2 BBR/reuse catch-up for any newly scored R1 unresolved subjects"
  "${PY}" "${LIVE}/run_aal3_route2_bbr_batch.py" \
    --source-run-root "${R1_ROOT}" \
    --include-interim \
    --workers "${WORKERS}" \
    --mrtrix-threads "${MRTRIX_THREADS}" \
    --variants forward40 forward80 \
    >> "${R2_ROOT}/followup_stdout.log" 2>> "${R2_ROOT}/followup_stderr.log"
  run_score "${R2_ROOT}"
}

mkdir -p "${R2_ROOT}" "${R3_ROOT}" "${R4_ROOT}" "${R5_ROOT}" "${R6_ROOT}" "${R7_ROOT}"
cd /home/ec2-user/exp
export PYTHONPATH="/home/ec2-user/exp:${PYTHONPATH:-}"
refresh_ledger

wait_for_process_pattern_to_end \
  "route2 BBR/reuse batch" \
  "run_aal3_route2_bbr_batch.py --source-run-root ${R1_ROOT}"
run_route2_catchup

log "starting/resuming route3 assignment-policy rescue"
"${PY}" "${LIVE}/run_aal3_route3_assignment_batch.py" \
  --route1-root "${R1_ROOT}" \
  --route2-root "${R2_ROOT}" \
  --workers "${WORKERS}" \
  --mrtrix-threads "${MRTRIX_THREADS}" \
  --variants reverse0 reverse80 radial8 forward120 \
  >> "${R3_ROOT}/followup_stdout.log" 2>> "${R3_ROOT}/followup_stderr.log"
run_score "${R3_ROOT}"

log "starting/resuming route4 label-preserving rescue"
"${PY}" "${LIVE}/run_aal3_route4_label_rescue_batch.py" \
  --route1-root "${R1_ROOT}" \
  --route3-root "${R3_ROOT}" \
  --workers "${WORKERS}" \
  --mrtrix-threads "${MRTRIX_THREADS}" \
  >> "${R4_ROOT}/followup_stdout.log" 2>> "${R4_ROOT}/followup_stderr.log"
run_score "${R4_ROOT}"

log "starting/resuming route5 deep endpoint-assignment rescue"
"${PY}" "${LIVE}/run_aal3_route5_deep_assignment_batch.py" \
  --route1-root "${R1_ROOT}" \
  --route2-root "${R2_ROOT}" \
  --route3-root "${R3_ROOT}" \
  --route4-root "${R4_ROOT}" \
  --workers "${WORKERS}" \
  --mrtrix-threads "${MRTRIX_THREADS}" \
  --variants endvox radial16 radial24 forward200 \
  >> "${R5_ROOT}/followup_stdout.log" 2>> "${R5_ROOT}/followup_stderr.log"
run_score "${R5_ROOT}"

log "starting/resuming route6 all-voxels diagnostic"
"${PY}" "${LIVE}/run_aal3_route6_all_voxels_diagnostic_batch.py" \
  --route1-root "${R1_ROOT}" \
  --route2-root "${R2_ROOT}" \
  --route5-root "${R5_ROOT}" \
  --workers "${WORKERS}" \
  --mrtrix-threads "${MRTRIX_THREADS}" \
  >> "${R6_ROOT}/followup_stdout.log" 2>> "${R6_ROOT}/followup_stderr.log"
# Route 6 is non-promotable diagnostic output. The route6 runner writes its own
# diagnostic decisions and buckets; do not replace them with the generic scorer.

wait_for_process_pattern_to_end \
  "current route7 selective rescue batch" \
  "run_aal3_route7_selective_rescue_batch.py --route1-root ${R1_ROOT}"

log "starting/resuming route7 selective registration/track rescue"
"${PY}" "${LIVE}/run_aal3_route7_selective_rescue_batch.py" \
  --route1-root "${R1_ROOT}" \
  --route2-root "${R2_ROOT}" \
  --route6-root "${R6_ROOT}" \
  --workers "${ROUTE7_WORKERS}" \
  --mrtrix-threads "${ROUTE7_THREADS}" \
  --select-streamlines "${SELECT_STREAMLINES}" \
  --track-modes dynamic \
  --variants forward80 forward120 radial8 \
  >> "${R7_ROOT}/followup_stdout.log" 2>> "${R7_ROOT}/followup_stderr.log"
run_score "${R7_ROOT}"
refresh_ledger

log "route follow-up chain complete"
