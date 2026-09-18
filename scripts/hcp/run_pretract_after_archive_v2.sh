#!/usr/bin/env bash
set -euo pipefail

while tmux has-session -t hcp379v2_weights_followup 2>/dev/null \
    || tmux has-session -t hcp379v2_weights_ss3t_retry 2>/dev/null \
    || tmux has-session -t hcp379v2_tensor_repair_followup 2>/dev/null; do
    sleep 20
done

WORKFLOW_PYTHON=/home/ec2-user/exp/.venv_connectome_workflow/bin/python
if [[ ! -x "$WORKFLOW_PYTHON" ]]; then
    printf 'ERROR: workflow Python is not executable: %s\n' "$WORKFLOW_PYTHON" >&2
    exit 1
fi

exec "$WORKFLOW_PYTHON" -u \
    /home/ec2-user/exp/scforge/workflow/run_h04a_r1_retry4_pretract_recovery3.py \
    --cores 32 \
    > /data/derivatives/hcp379_v2/logs/pretract_recovery3_launcher.log 2>&1
