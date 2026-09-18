#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/ec2-user/exp}
APP_ROOT=${APP_ROOT:-"$PROJECT_ROOT/apps/connectome_dashboard"}
PYTHON=${PYTHON:-"$PROJECT_ROOT/.venv_connectome_app/bin/python"}
MODE=${MODE:-quick}
INTERVAL=${INTERVAL:-300}
LOG_DIR=${LOG_DIR:-/data/derivatives/qc/analysis_cohort/logs}
SNAPSHOT_MODE=${SNAPSHOT_MODE:-provisional}

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

while true; do
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  log="$LOG_DIR/dashboard_refresh_${MODE}_${ts}.log"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] refresh mode=$MODE snapshot=$SNAPSHOT_MODE" | tee -a "$log"
  "$PYTHON" "$APP_ROOT/refresh_connectome_dashboard_data.py" \
    --mode "$MODE" \
    --snapshot-mode "$SNAPSHOT_MODE" \
    2>&1 | tee -a "$log" || true
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] sleeping ${INTERVAL}s" | tee -a "$log"
  sleep "$INTERVAL"
done
