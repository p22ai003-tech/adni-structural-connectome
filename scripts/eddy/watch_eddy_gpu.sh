#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${1:-1}"
LOG_FILE="${2:-}"

export PATH="$HOME/bin:$HOME/mrtrix3/bin:$HOME/fsl/bin:$HOME/fsl/share/fsl/bin:$PATH"
export FSLDIR="${FSLDIR:-$HOME/fsl}"

echo "Resolved tools"
echo "--------------"
echo "dwifslpreproc : $(command -v dwifslpreproc || echo 'not found')"
echo "eddy_cuda     : $(command -v eddy_cuda || echo 'not found')"
if command -v eddy_cuda >/dev/null 2>&1; then
  echo "eddy_cuda -> $(readlink -f "$(command -v eddy_cuda)")"
fi
echo "eddy_cpu      : $(command -v eddy_cpu || echo 'not found')"
if command -v eddy_cpu >/dev/null 2>&1; then
  echo "eddy_cpu  -> $(readlink -f "$(command -v eddy_cpu)")"
fi
echo
echo "Watching GPU every ${INTERVAL}s"
if [[ -n "$LOG_FILE" ]]; then
  echo "Tailing log: $LOG_FILE"
fi
echo "Press Ctrl+C to stop."
echo

while true; do
  clear
  date -u +"%Y-%m-%d %H:%M:%S UTC"
  echo
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  echo
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || true
  if [[ -n "$LOG_FILE" ]]; then
    echo
    echo "Last log lines"
    echo "--------------"
    if [[ -f "$LOG_FILE" ]]; then
      tail -n 20 "$LOG_FILE"
    else
      echo "Log file does not exist yet."
    fi
  fi
  sleep "$INTERVAL"
done
