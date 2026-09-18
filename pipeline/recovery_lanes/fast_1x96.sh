#!/bin/bash
# FASTEST per-subject: 1 subject x 96 threads (whole box) -> 10M tckgen ~6min, count ticks one-by-one.
# Usage: bash /tmp/fast_1x96.sh "<sid1> <sid2> ..."   (env overrides optional)
PY=/home/ec2-user/exp/.venv_connectome_app/bin/python
cd /home/ec2-user/exp
SUBS="$1"; [ -z "$SUBS" ] && { echo "usage: fast_1x96.sh \"sid1 sid2 ...\""; exit 1; }
RROOT=/data/derivatives/qc/sc_matrix_qc/fast1x96_$(date -u +%m%dT%H%M%SZ)
mkdir -p "$RROOT"; echo "$RROOT" > /tmp/lane5_mcicn_run_root.txt
NUMEXPR_MAX_THREADS=96 NUMEXPR_NUM_THREADS=96 \
MRTRIX_NTHREADS=96 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=96 OMP_NUM_THREADS=96 \
FORCE_NOACT=1 REBUILD_FOD=1 FRESH_T12B0=1 REREG_THRESH=0.5 \
TCKGEN_SELECT=10000000 TCKGEN_CAP_S=4200 RESUME_MIN_DENSITY=0.6 TMPDIR=/data/scratch_tmp \
  nohup setsid nice -n 5 $PY scripts/scforge/live/run_sc_route_sota.py \
    --subjects $SUBS \
    --run-root "$RROOT" --mode full_10m --workers 1 --threads 96 --radial 2 --reg-mode fullhead \
    > "$RROOT.log" 2>&1 </dev/null &
sleep 3
echo "1x96 batch launched (1 worker x 96 threads) -> $RROOT"
echo "drivers: $(pgrep -fc 'run_sc_route_sota.py.*fast1x96')"
