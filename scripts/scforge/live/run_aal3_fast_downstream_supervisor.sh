#!/usr/bin/env bash
set -euo pipefail

R1_ROOT="${1:-/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch/ad_existing_tracks_20260529T055752Z}"
PY="/home/ec2-user/exp/.venv_connectome_app/bin/python"
LIVE="/home/ec2-user/exp/scripts/scforge/live"
R2_ROOT="${R1_ROOT}_route2"
R3_ROOT="${R1_ROOT}_route3_assignment"
R4_ROOT="${R1_ROOT}_route4_label_rescue"
R5_ROOT="${R1_ROOT}_route5_deep_assignment"
R6_ROOT="${R1_ROOT}_route6_all_voxels_diagnostic"
R7_ROOT="${R1_ROOT}_route7_selective_track_rescue"
R8_ROOT="${R1_ROOT}_route8_full_t1_fnirt"
R9_ROOT="${R1_ROOT}_route9_route8_assignment_retry"
R10_ROOT="${R1_ROOT}_route10_endpoint_overlay_triage"
R11A_ROOT="${R1_ROOT}_route11a_space_affine_registration_repair"
R11B_ROOT="${R1_ROOT}_route11b_act_sift2_track_coverage_rerun"
R11C_ROOT="${R1_ROOT}_route11c_zero_row_coverage_rescue"
R12A_ROOT="${R1_ROOT}_route12a_tckgen_fallback_rescue"
R12B_ROOT="${R1_ROOT}_route12b_zero_row_targeted_rerun"
R12C_ROOT="${R1_ROOT}_route12c_registration_contract_repair"

ROUTE2_WORKERS="${ROUTE2_WORKERS:-8}"
ROUTE3_WORKERS="${ROUTE3_WORKERS:-8}"
ROUTE4_WORKERS="${ROUTE4_WORKERS:-8}"
ROUTE5_WORKERS="${ROUTE5_WORKERS:-8}"
ROUTE6_WORKERS="${ROUTE6_WORKERS:-4}"
ROUTE7_WORKERS="${ROUTE7_WORKERS:-2}"
ROUTE8_WORKERS="${ROUTE8_WORKERS:-12}"
ROUTE9_WORKERS="${ROUTE9_WORKERS:-8}"
ROUTE10_WORKERS="${ROUTE10_WORKERS:-4}"
ROUTE11A_WORKERS="${ROUTE11A_WORKERS:-4}"
ROUTE11B_WORKERS="${ROUTE11B_WORKERS:-4}"
ROUTE11C_WORKERS="${ROUTE11C_WORKERS:-8}"
ROUTE12A_WORKERS="${ROUTE12A_WORKERS:-3}"
ROUTE12B_WORKERS="${ROUTE12B_WORKERS:-6}"
ROUTE12C_WORKERS="${ROUTE12C_WORKERS:-4}"
MRTRIX_THREADS="${MRTRIX_THREADS:-4}"
ROUTE6_THREADS="${ROUTE6_THREADS:-2}"
ROUTE7_THREADS="${ROUTE7_THREADS:-4}"
ROUTE8_THREADS="${ROUTE8_THREADS:-4}"
ROUTE9_THREADS="${ROUTE9_THREADS:-3}"
ROUTE10_THREADS="${ROUTE10_THREADS:-2}"
ROUTE11A_THREADS="${ROUTE11A_THREADS:-2}"
ROUTE11B_THREADS="${ROUTE11B_THREADS:-4}"
ROUTE11C_THREADS="${ROUTE11C_THREADS:-4}"
ROUTE12A_THREADS="${ROUTE12A_THREADS:-4}"
ROUTE12B_THREADS="${ROUTE12B_THREADS:-3}"
ROUTE12C_THREADS="${ROUTE12C_THREADS:-2}"
ROUTE11A_CONNECTOME_TIMEOUT_SEC="${ROUTE11A_CONNECTOME_TIMEOUT_SEC:-2700}"
ROUTE7_SELECT_STREAMLINES="${ROUTE7_SELECT_STREAMLINES:-3000000}"
ROUTE11B_SELECT_STREAMLINES="${ROUTE11B_SELECT_STREAMLINES:-3000000}"
ROUTE11C_SELECT_STREAMLINES="${ROUTE11C_SELECT_STREAMLINES:-3000000}"
ROUTE12A_SELECT_STREAMLINES="${ROUTE12A_SELECT_STREAMLINES:-5000000}"
ROUTE12B_SELECT_STREAMLINES="${ROUTE12B_SELECT_STREAMLINES:-2000000}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-180}"
MAX_LOOPS="${MAX_LOOPS:-999}"

TAG="$(basename "$R1_ROOT")"
LOCK="/tmp/aal3_fast_downstream_${TAG}_v2.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another downstream supervisor is already running for ${TAG}" >&2
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

json_field() {
  local path="$1"
  local key="$2"
  "${PY}" - "$path" "$key" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
key = sys.argv[2]
try:
    payload = json.loads(path.read_text())
except Exception:
    payload = {}
print(payload.get(key, ""))
PY
}

run_score() {
  local root="$1"
  if [[ -s "${root}/batch_results.csv" ]]; then
    if [[ -s "${root}/route_qc_decisions.csv" ]]; then
      local newest_results="${root}/batch_results.csv"
      if [[ -s "${root}/batch_results_incremental.csv" && "${root}/batch_results_incremental.csv" -nt "${newest_results}" ]]; then
        newest_results="${root}/batch_results_incremental.csv"
      fi
      local decision_mtime result_mtime
      decision_mtime="$(stat -c %Y "${root}/route_qc_decisions.csv" 2>/dev/null || echo 0)"
      result_mtime="$(stat -c %Y "${newest_results}" 2>/dev/null || echo 1)"
      if [[ "${decision_mtime}" -ge "${result_mtime}" ]]; then
        log "score current for ${root}; skipping"
        return
      fi
    fi
    log "scoring ${root}"
    "${PY}" "${LIVE}/score_aal3_force_connectome_qc.py" --run-root "${root}" \
      > "${root}/score_stdout.log" 2> "${root}/score_stderr.log" || true
  fi
}

run_promote() {
  local root="$1"
  if [[ -s "${root}/route_qc_decisions.csv" ]]; then
    if [[ -s "${root}/production_promotion_summary.json" ]]; then
      local promotion_mtime decision_mtime
      promotion_mtime="$(stat -c %Y "${root}/production_promotion_summary.json" 2>/dev/null || echo 0)"
      decision_mtime="$(stat -c %Y "${root}/route_qc_decisions.csv" 2>/dev/null || echo 1)"
      if [[ "${promotion_mtime}" -ge "${decision_mtime}" ]]; then
        log "promotion current for ${root}; skipping"
        return
      fi
    fi
    log "promoting guarded candidates from ${root}"
    "${PY}" "${LIVE}/promote_aal3_qc_passed.py" --run-root "${root}" \
      --allow-interim-without-visual-pass \
      --allow-final-without-visual-pass \
      --execute \
      > "${root}/promote_stdout.log" 2> "${root}/promote_stderr.log" || true
  fi
}

run_score_promote() {
  local root="$1"
  run_score "$root"
  run_promote "$root"
}

refresh_ledger() {
  if [[ -s "${R1_ROOT}/subjects.csv" ]]; then
    log "refreshing consolidated route ledger"
    "${PY}" "${LIVE}/summarize_aal3_route_ledger.py" --run-root "${R1_ROOT}" \
      > "${R1_ROOT}/route_subject_ledger_stdout.log" \
      2> "${R1_ROOT}/route_subject_ledger_stderr.log" || true
  fi
}

proc_running() {
  local pattern="$1"
  pgrep -f "$pattern" >/dev/null
}

route2_running() {
  proc_running "run_aal3_route2_bbr_batch.py --source-run-root ${R1_ROOT}"
}

route3_running() {
  proc_running "run_aal3_route3_assignment_batch.py --route1-root ${R1_ROOT}"
}

route4_running() {
  proc_running "run_aal3_route4_label_rescue_batch.py --route1-root ${R1_ROOT}"
}

route5_running() {
  proc_running "run_aal3_route5_deep_assignment_batch.py --route1-root ${R1_ROOT}"
}

route6_running() {
  proc_running "run_aal3_route6_all_voxels_diagnostic_batch.py --route1-root ${R1_ROOT}"
}

route7_running() {
  proc_running "run_aal3_route7_selective_rescue_batch.py --route1-root ${R1_ROOT}"
}

route8_running() {
  proc_running "run_aal3_route8_full_t1_fnirt_batch.py --route1-root ${R1_ROOT}"
}

route9_running() {
  proc_running "run_aal3_route9_route8_assignment_retry_batch.py --route1-root ${R1_ROOT}"
}

route10_running() {
  proc_running "run_aal3_route10_endpoint_overlay_triage.py --route1-root ${R1_ROOT}"
}

route11a_running() {
  proc_running "run_aal3_route11a_space_affine_registration_repair.py --route1-root ${R1_ROOT}"
}

route11b_running() {
  proc_running "run_aal3_route11b_act_sift2_track_coverage_rerun.py --route1-root ${R1_ROOT}"
}

route11c_running() {
  proc_running "run_aal3_route11c_zero_row_coverage_rescue.py --route1-root ${R1_ROOT}"
}

route12a_running() {
  proc_running "run_aal3_route12a_tckgen_fallback_rescue.py --route1-root ${R1_ROOT}"
}

route12b_running() {
  proc_running "run_aal3_route12b_zero_row_targeted_rerun.py --route1-root ${R1_ROOT}"
}

route12c_running() {
  proc_running "run_aal3_route12c_registration_contract_repair.py --route1-root ${R1_ROOT}"
}

status_phase() {
  json_field "$1/status.json" phase
}

route_done() {
  local root="$1"
  "${PY}" - "$root" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))

subjects = {r.get("sid") or r.get("subject_id") or r.get("subject") for r in rows(root / "subjects.csv")}
subjects.discard(None)
subjects.discard("")
done: set[str] = set()
for name in ("batch_results_incremental.csv", "batch_results.csv", "route_qc_decisions.csv"):
    for row in rows(root / name):
        sid = row.get("sid") or row.get("subject_id") or row.get("subject")
        if sid:
            done.add(sid)
try:
    status = json.loads((root / "status.json").read_text())
except Exception:
    status = {}
total = 0
try:
    total = int(float(status.get("total") or 0))
except Exception:
    total = 0
if not subjects or len(done & subjects) >= len(subjects):
    if subjects and len(done & subjects) >= len(subjects):
        sys.exit(0)
    if not subjects and (status.get("phase") == "complete" or (total > 0 and len(done) >= total)):
        sys.exit(0)
    if not subjects and not total and done:
        sys.exit(0)
if status.get("phase") == "complete" and not subjects:
    sys.exit(0)
sys.exit(1)
PY
}

route_empty_or_done() {
  local root="$1"
  if [[ ! -s "${root}/subjects.csv" ]] || [[ "$(wc -l < "${root}/subjects.csv")" -le 1 ]]; then
    return 0
  fi
  route_done "$root"
}

csv_has_rows() {
  local path="$1"
  [[ -s "$path" ]] && [[ "$(wc -l < "$path")" -gt 1 ]]
}

score_when_idle() {
  local label="$1"
  local root="$2"
  local running_check="$3"
  if "$running_check"; then
    log "${label} is still running; skipping generic score this loop"
    return
  fi
  run_score "$root"
}

launch_route2() {
  if route_done "$R2_ROOT"; then
    log "route2 already complete; skipping"
    return
  fi
  if route2_running; then
    log "route2 already running; leaving existing writer alone"
    return
  fi
  if ! csv_has_rows "${R1_ROOT}/route_qc_decisions.csv"; then
    log "route2 waiting for route1 decisions"
    return
  fi
  log "starting route2 catch-up with workers=${ROUTE2_WORKERS}, mrtrix_threads=${MRTRIX_THREADS}"
  (
    "${PY}" "${LIVE}/run_aal3_route2_bbr_batch.py" \
      --source-run-root "${R1_ROOT}" \
      --include-interim \
      --workers "${ROUTE2_WORKERS}" \
      --mrtrix-threads "${MRTRIX_THREADS}" \
      --variants forward40 forward80
  ) >> "${R2_ROOT}/fast_supervisor_stdout.log" 2>> "${R2_ROOT}/fast_supervisor_stderr.log" 9>&- &
}

launch_route3() {
  if route_done "$R3_ROOT"; then
    log "route3 already complete; skipping"
    return
  fi
  if route3_running; then
    log "route3 already running; leaving existing writer alone"
    return
  fi
  if ! csv_has_rows "${R2_ROOT}/route_qc_decisions.csv"; then
    log "route3 waiting for route2 decisions"
    return
  fi
  log "starting route3 catch-up with workers=${ROUTE3_WORKERS}, mrtrix_threads=${MRTRIX_THREADS}"
  (
    "${PY}" "${LIVE}/run_aal3_route3_assignment_batch.py" \
      --route1-root "${R1_ROOT}" \
      --route2-root "${R2_ROOT}" \
      --workers "${ROUTE3_WORKERS}" \
      --mrtrix-threads "${MRTRIX_THREADS}" \
      --variants reverse0 reverse80 radial8 forward120
  ) >> "${R3_ROOT}/fast_supervisor_stdout.log" 2>> "${R3_ROOT}/fast_supervisor_stderr.log" 9>&- &
}

launch_route4() {
  if route_done "$R4_ROOT"; then
    log "route4 already complete; skipping"
    return
  fi
  if route4_running; then
    log "route4 already running; leaving existing writer alone"
    return
  fi
  log "starting route4 catch-up with workers=${ROUTE4_WORKERS}, mrtrix_threads=${MRTRIX_THREADS}"
  (
    "${PY}" "${LIVE}/run_aal3_route4_label_rescue_batch.py" \
      --route1-root "${R1_ROOT}" \
      --route3-root "${R3_ROOT}" \
      --workers "${ROUTE4_WORKERS}" \
      --mrtrix-threads "${MRTRIX_THREADS}"
  ) >> "${R4_ROOT}/fast_supervisor_stdout.log" 2>> "${R4_ROOT}/fast_supervisor_stderr.log" 9>&- &
}

launch_route5() {
  if route_done "$R5_ROOT"; then
    log "route5 already complete; skipping"
    return
  fi
  if route5_running; then
    log "route5 already running; leaving existing writer alone"
    return
  fi
  if ! csv_has_rows "${R3_ROOT}/route_qc_decisions.csv"; then
    log "route5 waiting for route3 decisions"
    return
  fi
  log "starting route5 catch-up with workers=${ROUTE5_WORKERS}, mrtrix_threads=${MRTRIX_THREADS}"
  (
    "${PY}" "${LIVE}/run_aal3_route5_deep_assignment_batch.py" \
      --route1-root "${R1_ROOT}" \
      --route2-root "${R2_ROOT}" \
      --route3-root "${R3_ROOT}" \
      --route4-root "${R4_ROOT}" \
      --workers "${ROUTE5_WORKERS}" \
      --mrtrix-threads "${MRTRIX_THREADS}" \
      --variants endvox radial16 radial24 forward200
  ) >> "${R5_ROOT}/fast_supervisor_stdout.log" 2>> "${R5_ROOT}/fast_supervisor_stderr.log" 9>&- &
}

launch_route6() {
  if route_done "$R6_ROOT"; then
    log "route6 already complete; skipping"
    return
  fi
  if route6_running; then
    log "route6 already running; leaving existing writer alone"
    return
  fi
  if ! csv_has_rows "${R5_ROOT}/route_qc_decisions.csv"; then
    log "route6 waiting for route5 decisions"
    return
  fi
  log "starting route6 diagnostic catch-up with workers=${ROUTE6_WORKERS}, mrtrix_threads=${ROUTE6_THREADS}"
  (
    "${PY}" "${LIVE}/run_aal3_route6_all_voxels_diagnostic_batch.py" \
      --route1-root "${R1_ROOT}" \
      --route2-root "${R2_ROOT}" \
      --route5-root "${R5_ROOT}" \
      --workers "${ROUTE6_WORKERS}" \
      --mrtrix-threads "${ROUTE6_THREADS}"
  ) >> "${R6_ROOT}/fast_supervisor_stdout.log" 2>> "${R6_ROOT}/fast_supervisor_stderr.log" 9>&- &
}

launch_route7() {
  if route_done "$R7_ROOT"; then
    log "route7 already complete; skipping"
    return
  fi
  if route7_running; then
    log "route7 already running; leaving existing writer alone"
    return
  fi
  if ! csv_has_rows "${R6_ROOT}/route_qc_decisions.csv"; then
    log "route7 waiting for route6 decisions"
    return
  fi
  log "starting route7 selective rescue with workers=${ROUTE7_WORKERS}, mrtrix_threads=${ROUTE7_THREADS}, select_streamlines=${ROUTE7_SELECT_STREAMLINES}"
  (
    "${PY}" "${LIVE}/run_aal3_route7_selective_rescue_batch.py" \
      --route1-root "${R1_ROOT}" \
      --route2-root "${R2_ROOT}" \
      --route6-root "${R6_ROOT}" \
      --workers "${ROUTE7_WORKERS}" \
      --mrtrix-threads "${ROUTE7_THREADS}" \
      --select-streamlines "${ROUTE7_SELECT_STREAMLINES}" \
      --track-modes dynamic \
      --variants forward80 forward120 radial8
  ) >> "${R7_ROOT}/fast_supervisor_stdout.log" 2>> "${R7_ROOT}/fast_supervisor_stderr.log" 9>&- &
}

launch_route8() {
  if route_done "$R8_ROOT"; then
    log "route8 already complete; skipping"
    return
  fi
  if route8_running; then
    log "route8 already running; leaving existing writer alone"
    return
  fi
  if ! csv_has_rows "${R6_ROOT}/route_qc_decisions.csv"; then
    log "route8 waiting for route6 decisions"
    return
  fi
  log "starting route8 full-T1 FNIRT rescue with workers=${ROUTE8_WORKERS}, mrtrix_threads=${ROUTE8_THREADS}"
  (
    "${PY}" "${LIVE}/run_aal3_route8_full_t1_fnirt_batch.py" \
      --route1-root "${R1_ROOT}" \
      --route6-root "${R6_ROOT}" \
      --workers "${ROUTE8_WORKERS}" \
      --mrtrix-threads "${ROUTE8_THREADS}" \
      --variants forward80 forward120 radial8
  ) >> "${R8_ROOT}/fast_supervisor_stdout.log" 2>> "${R8_ROOT}/fast_supervisor_stderr.log" 9>&- &
}

launch_route9() {
  if route_done "$R9_ROOT"; then
    log "route9 already complete; skipping"
    return
  fi
  if route9_running; then
    log "route9 already running; leaving existing writer alone"
    return
  fi
  if ! csv_has_rows "${R8_ROOT}/route_qc_decisions.csv"; then
    log "route9 waiting for route8 decisions"
    return
  fi
  log "starting route9 assignment retry with workers=${ROUTE9_WORKERS}, mrtrix_threads=${ROUTE9_THREADS}"
  (
    "${PY}" "${LIVE}/run_aal3_route9_route8_assignment_retry_batch.py" \
      --route1-root "${R1_ROOT}" \
      --route8-root "${R8_ROOT}" \
      --workers "${ROUTE9_WORKERS}" \
      --mrtrix-threads "${ROUTE9_THREADS}" \
      --variants endvox radial8 radial16 forward40 \
      --include-interim
  ) >> "${R9_ROOT}/fast_supervisor_stdout.log" 2>> "${R9_ROOT}/fast_supervisor_stderr.log" 9>&- &
}

launch_route10() {
  if route_done "$R10_ROOT"; then
    log "route10 already complete; skipping"
    return
  fi
  if route10_running; then
    log "route10 already running; leaving existing writer alone"
    return
  fi
  if ! csv_has_rows "${R9_ROOT}/route_qc_decisions.csv"; then
    log "route10 waiting for route9 decisions"
    return
  fi
  log "starting route10 endpoint-overlay triage with workers=${ROUTE10_WORKERS}, mrtrix_threads=${ROUTE10_THREADS}"
  (
    "${PY}" "${LIVE}/run_aal3_route10_endpoint_overlay_triage.py" \
      --route1-root "${R1_ROOT}" \
      --route9-root "${R9_ROOT}" \
      --workers "${ROUTE10_WORKERS}" \
      --mrtrix-threads "${ROUTE10_THREADS}"
  ) >> "${R10_ROOT}/fast_supervisor_stdout.log" 2>> "${R10_ROOT}/fast_supervisor_stderr.log" 9>&- &
}

launch_route11a() {
  if route_done "$R11A_ROOT"; then
    log "route11a already complete; skipping"
    return
  fi
  if route11a_running; then
    log "route11a already running; leaving existing writer alone"
    return
  fi
  if ! csv_has_rows "${R10_ROOT}/route10_triage.csv"; then
    log "route11a waiting for route10 triage"
    return
  fi
  log "starting route11a space/affine repair with workers=${ROUTE11A_WORKERS}, mrtrix_threads=${ROUTE11A_THREADS}"
  (
    "${PY}" "${LIVE}/run_aal3_route11a_space_affine_registration_repair.py" \
      --route1-root "${R1_ROOT}" \
      --route10-root "${R10_ROOT}" \
      --workers "${ROUTE11A_WORKERS}" \
      --mrtrix-threads "${ROUTE11A_THREADS}" \
      --connectome-timeout-sec "${ROUTE11A_CONNECTOME_TIMEOUT_SEC}" \
      --variants forward40 forward80 radial16 endvox
  ) >> "${R11A_ROOT}/fast_supervisor_stdout.log" 2>> "${R11A_ROOT}/fast_supervisor_stderr.log" 9>&- &
}

materialize_route11_queues() {
  if route10_running || route11a_running; then
    log "route11 queue materialization waiting for route10/route11a to become idle"
    return
  fi
  if ! csv_has_rows "${R10_ROOT}/route10_triage.csv"; then
    log "route11 queue materialization waiting for route10 triage"
    return
  fi
  log "materializing route11b/route11c queues"
  "${PY}" "${LIVE}/materialize_aal3_route11_downstream_queues.py" \
    --route1-root "${R1_ROOT}" \
    --route10-root "${R10_ROOT}" \
    --route11a-root "${R11A_ROOT}" \
    > "${R1_ROOT}/route11_queue_materialize_stdout.log" \
    2> "${R1_ROOT}/route11_queue_materialize_stderr.log" || true
}

launch_route11b() {
  if route_done "$R11B_ROOT"; then
    log "route11b already complete; skipping"
    return
  fi
  if route11b_running; then
    log "route11b already running; leaving existing writer alone"
    return
  fi
  if ! csv_has_rows "${R11B_ROOT}/route_queue.csv"; then
    log "route11b waiting for materialized queue"
    return
  fi
  log "starting route11b ACT/SIFT2 track-coverage rerun with workers=${ROUTE11B_WORKERS}, mrtrix_threads=${ROUTE11B_THREADS}, select_streamlines=${ROUTE11B_SELECT_STREAMLINES}"
  (
    "${PY}" "${LIVE}/run_aal3_route11b_act_sift2_track_coverage_rerun.py" \
      --route1-root "${R1_ROOT}" \
      --route2-root "${R2_ROOT}" \
      --workers "${ROUTE11B_WORKERS}" \
      --mrtrix-threads "${ROUTE11B_THREADS}" \
      --select-streamlines "${ROUTE11B_SELECT_STREAMLINES}" \
      --track-modes dynamic \
      --variants forward80 forward120 radial16 endvox
  ) >> "${R11B_ROOT}/fast_supervisor_stdout.log" 2>> "${R11B_ROOT}/fast_supervisor_stderr.log" 9>&- &
}

launch_route11c() {
  if route_done "$R11C_ROOT"; then
    log "route11c already complete; skipping"
    return
  fi
  if route11c_running; then
    log "route11c already running; leaving existing writer alone"
    return
  fi
  if ! csv_has_rows "${R11C_ROOT}/route_queue.csv"; then
    log "route11c waiting for materialized queue"
    return
  fi
  log "starting route11c zero-row coverage rescue with workers=${ROUTE11C_WORKERS}, mrtrix_threads=${ROUTE11C_THREADS}, select_streamlines=${ROUTE11C_SELECT_STREAMLINES}"
  (
    "${PY}" "${LIVE}/run_aal3_route11c_zero_row_coverage_rescue.py" \
      --route1-root "${R1_ROOT}" \
      --route2-root "${R2_ROOT}" \
      --workers "${ROUTE11C_WORKERS}" \
      --mrtrix-threads "${ROUTE11C_THREADS}" \
      --select-streamlines "${ROUTE11C_SELECT_STREAMLINES}" \
      --track-modes dynamic gmwmi \
      --variants allvoxels endvox radial24 forward200
  ) >> "${R11C_ROOT}/fast_supervisor_stdout.log" 2>> "${R11C_ROOT}/fast_supervisor_stderr.log" 9>&- &
}

materialize_route12_queues() {
  if route12a_running || route12b_running || route12c_running; then
    log "route12 queue materialization waiting for active route12 writers"
    return
  fi
  if ! route_empty_or_done "$R11C_ROOT"; then
    log "route12 queue materialization waiting for route11c to finish"
    return
  fi
  log "refreshing route failure investigation for route12"
  "${PY}" "${LIVE}/investigate_aal3_route_failures.py" --run-root "${R1_ROOT}" \
    > "${R1_ROOT}/route12_investigation_stdout.log" \
    2> "${R1_ROOT}/route12_investigation_stderr.log" || true
  log "materializing route12a/route12b/route12c queues"
  "${PY}" "${LIVE}/materialize_aal3_route12_downstream_queues.py" \
    --route1-root "${R1_ROOT}" \
    > "${R1_ROOT}/route12_queue_materialize_stdout.log" \
    2> "${R1_ROOT}/route12_queue_materialize_stderr.log" || true
}

launch_route12a() {
  if route_done "$R12A_ROOT"; then
    log "route12a already complete; skipping"
    return
  fi
  if route12a_running; then
    log "route12a already running; leaving existing writer alone"
    return
  fi
  if ! csv_has_rows "${R12A_ROOT}/route_queue.csv"; then
    log "route12a waiting for materialized queue"
    return
  fi
  log "starting route12a high-density dynamic-first tckgen coverage rerun with workers=${ROUTE12A_WORKERS}, mrtrix_threads=${ROUTE12A_THREADS}, select_streamlines=${ROUTE12A_SELECT_STREAMLINES}"
  (
    "${PY}" "${LIVE}/run_aal3_route12a_tckgen_fallback_rescue.py" \
      --route1-root "${R1_ROOT}" \
      --route2-root "${R2_ROOT}" \
      --workers "${ROUTE12A_WORKERS}" \
      --mrtrix-threads "${ROUTE12A_THREADS}" \
      --select-streamlines "${ROUTE12A_SELECT_STREAMLINES}" \
      --track-modes dynamic gmwmi \
      --variants endvox radial16 radial24 forward80
  ) >> "${R12A_ROOT}/fast_supervisor_stdout.log" 2>> "${R12A_ROOT}/fast_supervisor_stderr.log" 9>&- &
}

launch_route12b() {
  if route_done "$R12B_ROOT"; then
    log "route12b already complete; skipping"
    return
  fi
  if route12b_running; then
    log "route12b already running; leaving existing writer alone"
    return
  fi
  if ! csv_has_rows "${R12B_ROOT}/route_queue.csv"; then
    log "route12b waiting for materialized queue"
    return
  fi
  log "starting route12b zero-row targeted fallback with workers=${ROUTE12B_WORKERS}, mrtrix_threads=${ROUTE12B_THREADS}, select_streamlines=${ROUTE12B_SELECT_STREAMLINES}"
  (
    "${PY}" "${LIVE}/run_aal3_route12b_zero_row_targeted_rerun.py" \
      --route1-root "${R1_ROOT}" \
      --route2-root "${R2_ROOT}" \
      --workers "${ROUTE12B_WORKERS}" \
      --mrtrix-threads "${ROUTE12B_THREADS}" \
      --select-streamlines "${ROUTE12B_SELECT_STREAMLINES}" \
      --track-modes gmwmi dynamic \
      --variants allvoxels endvox radial24 forward200
  ) >> "${R12B_ROOT}/fast_supervisor_stdout.log" 2>> "${R12B_ROOT}/fast_supervisor_stderr.log" 9>&- &
}

launch_route12c() {
  if route_done "$R12C_ROOT"; then
    log "route12c already complete; skipping"
    return
  fi
  if route12c_running; then
    log "route12c already running; leaving existing writer alone"
    return
  fi
  if ! csv_has_rows "${R12C_ROOT}/route_queue.csv"; then
    log "route12c waiting for materialized queue"
    return
  fi
  log "starting route12c registration contract repair with workers=${ROUTE12C_WORKERS}, mrtrix_threads=${ROUTE12C_THREADS}"
  (
    "${PY}" "${LIVE}/run_aal3_route12c_registration_contract_repair.py" \
      --route1-root "${R1_ROOT}" \
      --route10-root "${R10_ROOT}" \
      --workers "${ROUTE12C_WORKERS}" \
      --mrtrix-threads "${ROUTE12C_THREADS}" \
      --max-candidates 3 \
      --connectome-timeout-sec "${ROUTE11A_CONNECTOME_TIMEOUT_SEC}" \
      --variants forward40 forward80 radial16 endvox
  ) >> "${R12C_ROOT}/fast_supervisor_stdout.log" 2>> "${R12C_ROOT}/fast_supervisor_stderr.log" 9>&- &
}

cd /home/ec2-user/exp
export PYTHONPATH="/home/ec2-user/exp:${PYTHONPATH:-}"
mkdir -p "$R2_ROOT" "$R3_ROOT" "$R4_ROOT" "$R5_ROOT" "$R6_ROOT"
mkdir -p "$R7_ROOT" "$R8_ROOT" "$R9_ROOT" "$R10_ROOT" "$R11A_ROOT" "$R11B_ROOT" "$R11C_ROOT"
mkdir -p "$R12A_ROOT" "$R12B_ROOT" "$R12C_ROOT"

for ((loop = 1; loop <= MAX_LOOPS; loop++)); do
  log "downstream catch-up loop ${loop}/${MAX_LOOPS}"
  launch_route2
  launch_route3
  launch_route4
  launch_route5
  launch_route6
  launch_route7
  launch_route8
  launch_route9
  launch_route10
  launch_route11a
  materialize_route11_queues
  launch_route11b
  launch_route11c
  materialize_route12_queues
  launch_route12a
  launch_route12b
  launch_route12c
  score_when_idle "route2" "${R2_ROOT}" route2_running
  score_when_idle "route3" "${R3_ROOT}" route3_running
  score_when_idle "route4" "${R4_ROOT}" route4_running
  score_when_idle "route5" "${R5_ROOT}" route5_running
  # Route 6 is diagnostic-only. Its runner writes route_qc_decisions.csv with
  # diagnostic buckets, and the generic scorer would incorrectly mark density
  # improvements as promotable candidates.
  score_when_idle "route7" "${R7_ROOT}" route7_running
  score_when_idle "route8" "${R8_ROOT}" route8_running
  score_when_idle "route9" "${R9_ROOT}" route9_running
  score_when_idle "route10" "${R10_ROOT}" route10_running
  score_when_idle "route11a" "${R11A_ROOT}" route11a_running
  score_when_idle "route11b" "${R11B_ROOT}" route11b_running
  score_when_idle "route11c" "${R11C_ROOT}" route11c_running
  score_when_idle "route12a" "${R12A_ROOT}" route12a_running
  score_when_idle "route12b" "${R12B_ROOT}" route12b_running
  score_when_idle "route12c" "${R12C_ROOT}" route12c_running
  if ! route2_running; then run_promote "${R2_ROOT}"; fi
  if ! route3_running; then run_promote "${R3_ROOT}"; fi
  if ! route4_running; then run_promote "${R4_ROOT}"; fi
  if ! route5_running; then run_promote "${R5_ROOT}"; fi
  if ! route7_running; then run_promote "${R7_ROOT}"; fi
  if ! route8_running; then run_promote "${R8_ROOT}"; fi
  if ! route9_running; then run_promote "${R9_ROOT}"; fi
  if ! route11a_running; then run_promote "${R11A_ROOT}"; fi
  if ! route11b_running; then run_promote "${R11B_ROOT}"; fi
  if ! route11c_running; then run_promote "${R11C_ROOT}"; fi
  if ! route12a_running; then run_promote "${R12A_ROOT}"; fi
  if ! route12b_running; then run_promote "${R12B_ROOT}"; fi
  if ! route12c_running; then run_promote "${R12C_ROOT}"; fi
  materialize_route11_queues
  materialize_route12_queues
  refresh_ledger

  r3_phase="$(json_field "${R3_ROOT}/status.json" phase)"
  r3_completed="$(json_field "${R3_ROOT}/status.json" completed)"
  r3_total="$(json_field "${R3_ROOT}/status.json" total)"
  log "route3 status: phase=${r3_phase:-unknown} completed=${r3_completed:-0}/${r3_total:-0}"

  if [[ "${r3_phase}" == "complete" ]] \
    && ! route3_running \
    && route_done "$R4_ROOT" \
    && route_done "$R5_ROOT" \
    && route_done "$R6_ROOT" \
    && route_done "$R7_ROOT" \
    && route_done "$R8_ROOT" \
    && route_empty_or_done "$R9_ROOT" \
    && route_empty_or_done "$R10_ROOT" \
    && route_empty_or_done "$R11A_ROOT" \
    && route_empty_or_done "$R11B_ROOT" \
    && route_empty_or_done "$R11C_ROOT" \
    && route_empty_or_done "$R12A_ROOT" \
    && route_empty_or_done "$R12B_ROOT" \
    && route_empty_or_done "$R12C_ROOT"; then
    log "route3 is complete; downstream catch-up supervisor complete"
    break
  fi
  sleep "$INTERVAL_SECONDS"
done
