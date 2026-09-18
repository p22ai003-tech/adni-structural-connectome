#!/bin/bash
# AD blitz v2 — FIX: REBUILD_FOD=0 (use the healthy cached wmfod_final.mif; the v1 rebuild produced
# degenerate near-zero FODs -> tckgen accepted ~0 streamlines -> 54yr ETA -> deferred). Keep fresh
# SyN registration (atlas placement lever) + 10M + noACT. Skipping the rebuild also makes it far faster.
PY=/home/ec2-user/exp/.venv_connectome_app/bin/python
cd /home/ec2-user/exp
RROOT=/data/derivatives/qc/sc_matrix_qc/ad_blitz2_$(date -u +%m%dT%H%M%SZ)
mkdir -p "$RROOT"
echo "$RROOT" > /tmp/ad_blitz_root.txt
MRTRIX_NTHREADS=6 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=6 OMP_NUM_THREADS=6 \
FORCE_NOACT=1 REBUILD_FOD=0 FRESH_T12B0=1 REREG_THRESH=0.5 \
TCKGEN_SELECT=10000000 RESUME_MIN_DENSITY=0.6 TMPDIR=/data/scratch_tmp \
  nohup setsid nice -n 5 $PY scripts/scforge/live/run_sc_route_sota.py \
    --subjects $(cat /tmp/ad_pending.txt) \
    --run-root "$RROOT" --mode full_10m --workers 24 --threads 6 --radial 2 --reg-mode fullhead \
    > "$RROOT.log" 2>&1 </dev/null &
sleep 2
echo "launched AD blitz v2 -> $RROOT (24 AD, REBUILD_FOD=0, healthy cached FOD, 10M)"
echo "drivers: $(pgrep -f 'ad_blitz2' | wc -l)"
