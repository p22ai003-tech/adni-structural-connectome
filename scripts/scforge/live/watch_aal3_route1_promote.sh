#!/usr/bin/env bash
set -euo pipefail

ROUTE1_ROOT="${1:-/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch/ad_existing_tracks_20260529T055752Z}"
PY="/home/ec2-user/exp/.venv_connectome_app/bin/python"
LIVE="/home/ec2-user/exp/scripts/scforge/live"

date -u +"AAL3 Route1 promote/watch | %Y-%m-%dT%H:%M:%SZ"
echo "route1 root: $ROUTE1_ROOT"

if [[ ! -d "$ROUTE1_ROOT" ]]; then
  echo "Route1 root not found."
  exit 0
fi

"$PY" "$LIVE/score_aal3_force_connectome_qc.py" --run-root "$ROUTE1_ROOT" || true
"$PY" "$LIVE/promote_aal3_qc_passed.py" --run-root "$ROUTE1_ROOT" --allow-interim-without-visual-pass --allow-final-without-visual-pass --execute || true
"$PY" "$LIVE/summarize_aal3_route_ledger.py" --run-root "$ROUTE1_ROOT" || true
"$LIVE/watch_aal3_rescue_overview.sh"
