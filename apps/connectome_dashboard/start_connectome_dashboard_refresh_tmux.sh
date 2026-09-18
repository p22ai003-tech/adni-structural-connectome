#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-connectome_dashboard_refresh}
PROJECT_ROOT=${PROJECT_ROOT:-/home/ec2-user/exp}
APP_ROOT=${APP_ROOT:-"$PROJECT_ROOT/apps/connectome_dashboard"}
MODE=${MODE:-quick}
INTERVAL=${INTERVAL:-300}
SNAPSHOT_MODE=${SNAPSHOT_MODE:-provisional}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "attach: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "PROJECT_ROOT='$PROJECT_ROOT' APP_ROOT='$APP_ROOT' MODE='$MODE' INTERVAL='$INTERVAL' SNAPSHOT_MODE='$SNAPSHOT_MODE' '$APP_ROOT/run_connectome_dashboard_refresh_loop.sh'"

echo "started tmux session: $SESSION"
echo "attach: tmux attach -t $SESSION"
