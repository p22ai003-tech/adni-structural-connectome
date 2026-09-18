#!/bin/bash
# Re-run the empty-FOD crashers FAST: REBUILD_FOD=1 (forces a real FOD, no empty-check dependency),
# 2 subjects x 48 threads = full box, each ~10min tckgen -> count climbs ~2 every ~20min (near-real-time).
PY=/home/ec2-user/exp/.venv_connectome_app/bin/python
cd /home/ec2-user/exp
RROOT=/data/derivatives/qc/sc_matrix_qc/fastcrash_$(date -u +%m%dT%H%M%SZ)
mkdir -p "$RROOT"; echo "$RROOT" > /tmp/lane5_mcicn_run_root.txt  # show in MCI/CN cascade L5
MRTRIX_NTHREADS=48 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=48 OMP_NUM_THREADS=48 \
FORCE_NOACT=1 REBUILD_FOD=1 FRESH_T12B0=1 REREG_THRESH=0.5 \
TCKGEN_SELECT=10000000 TCKGEN_CAP_S=4200 RESUME_MIN_DENSITY=0.6 TMPDIR=/data/scratch_tmp \
  nohup setsid nice -n 5 $PY scripts/scforge/live/run_sc_route_sota.py \
    --subjects $(cat /tmp/fast_crashers.txt) \
    --run-root "$RROOT" --mode full_10m --workers 2 --threads 48 --radial 2 --reg-mode fullhead \
    > "$RROOT.log" 2>&1 </dev/null &
sleep 3
echo "fast crasher batch launched (2 workers x 48 threads, REBUILD_FOD=1) -> $RROOT"
echo "drivers: $(pgrep -f 'run_sc_route_sota.py.*fastcrash' | wc -l)"
