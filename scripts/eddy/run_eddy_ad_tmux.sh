#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-eddy-ad}"
JOBS="${2:-1}"
THREADS="${3:-4}"
shift $(( $# >= 3 ? 3 : $# )) || true
EXTRA_ARGS=("$@")

LOG_DIR="$HOME/exp/logs"
RUN_LOG="$LOG_DIR/${SESSION}.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

printf -v EXTRA_Q '%q ' "${EXTRA_ARGS[@]}"

WATCH_CMD="source '$SCRIPT_DIR/eddy_env.sh' && '$SCRIPT_DIR/watch_eddy_gpu.sh' 1"
RUN_CMD="source '$SCRIPT_DIR/eddy_env.sh' && echo \"Logging to $RUN_LOG\" && eddy_ad_run $JOBS $THREADS ${EXTRA_Q} 2>&1 | tee -a $RUN_LOG"

tmux new-session -d -s "$SESSION" -n eddy "bash -lc '$WATCH_CMD'"
tmux split-window -v -t "$SESSION:0" "bash -lc '$RUN_CMD'"
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
tmux select-layout -t "$SESSION:0" even-vertical >/dev/null
tmux select-pane -t "$SESSION:0.1"

echo "Started tmux session '$SESSION'"
echo "Top pane   : live GPU monitor"
echo "Bottom pane: AD Eddy CLI run"
echo "Attach with: tmux attach -t $SESSION"
