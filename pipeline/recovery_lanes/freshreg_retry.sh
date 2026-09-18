#!/bin/bash
# Untried lever for the healthy-FOD mcicn fails: FORCE fresh registration.
# REBUILD_FOD=1 (REQUIRED: produces grid-matched b0_post that fresh_t12b0 needs) | FRESH_T12B0=1 + REREG_THRESH=0.99 (always attempt fresh reg).
# 1x96 (user-approved fast config): fastest per-subject, count ticks one-by-one. Ordered closest-to-0.6 first.
PY=/home/ec2-user/exp/.venv_connectome_app/bin/python
cd /home/ec2-user/exp
SUBS="941_S_5193_I860961 168_S_6815_I1233199 168_S_6591_I1378384 168_S_6873_I1318212 301_S_6508_I1032140 941_S_6017_I852757 305_S_6877_I1478941 301_S_6056_I1224048"
RROOT=/data/derivatives/qc/sc_matrix_qc/freshreg_$(date -u +%m%dT%H%M%SZ)
mkdir -p "$RROOT"; echo "$RROOT" > /tmp/lane5_mcicn_run_root.txt
NUMEXPR_MAX_THREADS=96 NUMEXPR_NUM_THREADS=96 \
MRTRIX_NTHREADS=96 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=96 OMP_NUM_THREADS=96 \
FORCE_NOACT=1 REBUILD_FOD=1 FRESH_T12B0=1 REREG_THRESH=0.99 \
TCKGEN_SELECT=10000000 TCKGEN_CAP_S=4200 RESUME_MIN_DENSITY=0.6 TMPDIR=/data/scratch_tmp \
  nohup setsid nice -n 5 $PY scripts/scforge/live/run_sc_route_sota.py \
    --subjects $SUBS \
    --run-root "$RROOT" --mode full_10m --workers 1 --threads 96 --radial 2 --reg-mode fullhead \
    > "$RROOT.log" 2>&1 </dev/null &
sleep 3
echo "fresh-reg retry launched (1x96, REBUILD_FOD=0, forced fresh reg) -> $RROOT"
echo "drivers: $(pgrep -fc 'run_sc_route_sota.py.*freshreg')"
