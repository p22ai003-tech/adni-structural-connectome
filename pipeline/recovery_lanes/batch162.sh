#!/bin/bash
# FULL corrected-recipe batch over ALL 162 pending (mid-band first).
# Recipe = the winning crasher recipe: REBUILD_FOD=1 (rebuilds FOD + generates b0_post so fresh_t12b0 CAN fire),
# FRESH_T12B0=1 + REREG_THRESH=0.99 (always attempt fresh reg; runner keeps BEST of cached/fresh so it can't hurt),
# 10M tckgen. 16 workers x 12 threads = full box; status_writer renices compute below the PE app (8502/8503).
PY=/home/ec2-user/exp/.venv_connectome_app/bin/python
cd /home/ec2-user/exp
RROOT=/data/derivatives/qc/sc_matrix_qc/batch162_$(date -u +%m%dT%H%M%SZ)
mkdir -p "$RROOT"; echo "$RROOT" > /tmp/lane5_mcicn_run_root.txt
NUMEXPR_MAX_THREADS=64 NUMEXPR_NUM_THREADS=12 \
MRTRIX_NTHREADS=12 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=12 OMP_NUM_THREADS=12 \
FORCE_NOACT=1 REBUILD_FOD=1 FRESH_T12B0=1 REREG_THRESH=0.99 \
TCKGEN_SELECT=10000000 TCKGEN_CAP_S=4200 RESUME_MIN_DENSITY=0.6 TMPDIR=/data/scratch_tmp \
  nohup setsid nice -n 5 $PY scripts/scforge/live/run_sc_route_sota.py \
    --subjects $(cat /tmp/all162.txt) \
    --run-root "$RROOT" --mode full_10m --workers 16 --threads 12 --radial 2 --reg-mode fullhead \
    > "$RROOT.log" 2>&1 </dev/null &
sleep 4
echo "BATCH162 launched (16 workers x 12 threads, corrected recipe) -> $RROOT"
echo "drivers: $(pgrep -fc 'run_sc_route_sota.py.*batch162')"
