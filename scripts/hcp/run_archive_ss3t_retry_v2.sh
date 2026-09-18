#!/usr/bin/env bash
set -euo pipefail

while tmux has-session -t hcp379v2_weights_followup 2>/dev/null; do
    sleep 20
done

exec python3 -u \
    /home/ec2-user/exp/scripts/hcp/hcp379_v2_orchestrator.py \
    archive-weights --workers 1 --nthreads 12 \
    > /data/derivatives/hcp379_v2/logs/archive_weights_ss3t_retry.log 2>&1
