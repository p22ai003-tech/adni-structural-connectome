#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-dwi_denoise_gibbs}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/ec2-user/exp}"
DERIV_ROOT="${DERIV_ROOT:-/home/ec2-user/exp/data/derivatives}"
PYTHON="${PYTHON:-/home/ec2-user/fsl/bin/python}"
JOBS="${JOBS:-4}"
EXECUTE="${EXECUTE:-0}"
FORCE="${FORCE:-0}"
FORCE_INVALID_ONLY="${FORCE_INVALID_ONLY:-1}"

args=(
  env "PYTHONPATH=${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m connectome_pipeline.dwi_denoise_gibbs
  --deriv-root "$DERIV_ROOT"
  --jobs "$JOBS"
)

if [[ "$EXECUTE" == "1" ]]; then
  args+=(--execute)
fi
if [[ "$FORCE" == "1" ]]; then
  args+=(--force)
fi
if [[ "$FORCE_INVALID_ONLY" != "1" ]]; then
  args+=(--no-force-invalid-only)
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "$(printf '%q ' "${args[@]}"); echo; echo 'done'; read -r _"

cat <<EOF
Started tmux session: $SESSION
Attach: tmux attach -t $SESSION

Defaults are conservative:
  EXECUTE=$EXECUTE
  FORCE=$FORCE
  FORCE_INVALID_ONLY=$FORCE_INVALID_ONLY
  JOBS=$JOBS
EOF
