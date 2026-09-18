#!/usr/bin/env bash
set -euo pipefail

ROUTE9_ROOT="${1:-/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch/ad_existing_tracks_20260529T055752Z_route9_route8_assignment_retry}"
ROUTE1_ROOT="${ROUTE9_ROOT%_route9_route8_assignment_retry}"
PY="/home/ec2-user/exp/.venv_connectome_app/bin/python"
LIVE="/home/ec2-user/exp/scripts/scforge/live"

date -u +"AAL3 Route9 promote/watch | %Y-%m-%dT%H:%M:%SZ"
echo "route9 root: $ROUTE9_ROOT"

if [[ ! -d "$ROUTE9_ROOT" ]]; then
  echo "Route9 root not found."
  exit 0
fi

"$PY" "$LIVE/score_aal3_force_connectome_qc.py" --run-root "$ROUTE9_ROOT" || true
"$PY" "$LIVE/promote_aal3_qc_passed.py" --run-root "$ROUTE9_ROOT" --allow-interim-without-visual-pass --allow-final-without-visual-pass --execute || true
"$PY" "$LIVE/summarize_aal3_route_ledger.py" --run-root "$ROUTE1_ROOT"
"$LIVE/watch_aal3_rescue_overview.sh"
