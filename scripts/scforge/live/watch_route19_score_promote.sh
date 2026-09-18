#!/usr/bin/env bash
set -euo pipefail

R1_ROOT="${1:-/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch/ad_existing_tracks_20260529T055752Z}"
RUN_ROOT="${R1_ROOT}_route19_gmwmi_3m_retry"
PY="/home/ec2-user/exp/.venv_connectome_app/bin/python"
LIVE="/home/ec2-user/exp/scripts/scforge/live"
INTERVAL_SEC="${INTERVAL_SEC:-180}"

cd /home/ec2-user/exp
export PYTHONPATH="/home/ec2-user/exp:${PYTHONPATH:-}"

last_scored_batch_mtime=0

while true; do
  if [[ -s "${RUN_ROOT}/batch_results.csv" ]]; then
    batch_mtime="$(stat -c %Y "${RUN_ROOT}/batch_results.csv" 2>/dev/null || echo 0)"
    if [[ "${batch_mtime}" -gt "${last_scored_batch_mtime}" || ! -s "${RUN_ROOT}/route_qc_decisions.csv" ]]; then
      "${PY}" "${LIVE}/score_aal3_force_connectome_qc.py" --run-root "${RUN_ROOT}" \
        > "${RUN_ROOT}/route19_watcher_score_stdout.log" \
        2> "${RUN_ROOT}/route19_watcher_score_stderr.log" || true
      last_scored_batch_mtime="${batch_mtime}"
    fi
  fi

  if [[ -s "${RUN_ROOT}/route_qc_decisions.csv" ]]; then
    promote_needed=1
    if [[ -s "${RUN_ROOT}/production_promotion_summary.json" ]]; then
      decision_mtime="$(stat -c %Y "${RUN_ROOT}/route_qc_decisions.csv" 2>/dev/null || echo 0)"
      promotion_mtime="$(stat -c %Y "${RUN_ROOT}/production_promotion_summary.json" 2>/dev/null || echo 0)"
      if [[ "${promotion_mtime}" -ge "${decision_mtime}" ]]; then
        promote_needed=0
      fi
    fi
    if [[ "${promote_needed}" == "1" ]]; then
      "${PY}" "${LIVE}/promote_aal3_qc_passed.py" --run-root "${RUN_ROOT}" \
        --allow-interim-without-visual-pass \
        --allow-final-without-visual-pass \
        --execute \
        > "${RUN_ROOT}/route19_watcher_promote_stdout.log" \
        2> "${RUN_ROOT}/route19_watcher_promote_stderr.log" || true
      "${PY}" "${LIVE}/summarize_aal3_route_ledger.py" --run-root "${R1_ROOT}" \
        > "${R1_ROOT}/route19_watcher_ledger_stdout.log" \
        2> "${R1_ROOT}/route19_watcher_ledger_stderr.log" || true
    fi
  fi

  sleep "${INTERVAL_SEC}"
done
