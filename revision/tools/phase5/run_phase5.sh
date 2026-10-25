#!/bin/bash
# Phase 5 robustness on the rebuilt connectomes: second parcellation (DKT76) and other edge weights for the exception
# rule (streamline count, FA), plus the same-recipe recomputation on AAL3 with SIFT2 weights (aal3_fd) as the check that
# p5_features.py reproduces the pipeline's features. Each variant gets the Phase 1 engine (13 conditions) and tables; DKT
# also gets Phase 2 (length test, P0 and P2) and Phase 3 (null models); the weight variants get Phase 3.
#   run_phase5.sh [B=2000] [R=500]
set -uo pipefail
B=${1:-2000}; R=${2:-500}
U=/home/ec2-user/exp/docs/iit_dipanjan; T=$U/tools; PY=$T/.venv/bin/python; P5=$U/phase5_uniform
export IIT_AROOT=/data/derivatives/qc/analysis_cohort_uniform IIT_P1_COMPARE=$U/phase1
step() { echo; echo "== $(date -u +%FT%TZ)  $*"; }
LOG=$P5/run_phase5.log; exec > >(tee -a "$LOG") 2>&1

step "1. features (skipped when present)"
for v in aal3_fd:fd aal3_count:count aal3_fa:fa; do
  name=${v%%:*}; w=${v##*:}
  [ -f $P5/$name/phase1/features_qc.csv ] || IIT_P1=$U/phase1_uniform $PY -I $T/phase5/p5_features.py $U/phase2_uniform/edges_uniform_v3.npz $P5/aal3_network_mapping.csv $P5/$name/phase1 $w
done
[ -f $P5/dkt/phase1/features_qc.csv ] || IIT_P1=$U/phase1_uniform $PY -I $T/phase5/p5_features.py $P5/dkt/phase2/edges_uniform_v3_dkt.npz /data/derivatives/connectomes_uniform_v3/dkt76_network_mapping.csv $P5/dkt/phase1 fd

step "2. Phase 1 engine, 13 conditions per variant"
cd $T/phase1
for name in dkt aal3_fd aal3_count aal3_fa; do
  for c in C0 C0q C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11; do
    IIT_P1=$P5/$name/phase1 IIT_SRC=uniform_v3_$name $PY -I -W ignore p1_engine.py $B $c > $P5/$name/phase1/engine_$c.log 2>&1 &
  done
done
wait $(jobs -p) || echo "WARNING: at least one engine run failed"
for name in dkt aal3_fd aal3_count aal3_fa; do
  (export IIT_P1=$P5/$name/phase1 IIT_SRC=uniform_v3_$name; $PY -I p1_merge.py > /dev/null; $PY -I -W ignore p1_tables_figs.py > /dev/null) || echo "WARNING: tables $name"
done

step "3. Phase 2 on DKT (P0, P2)"
cd $T/phase2
for c in P0 P2; do IIT_P1=$P5/dkt/phase1 IIT_P2=$P5/dkt/phase2 OMP_NUM_THREADS=4 $PY -I p2_engine.py uniform_v3_dkt $c $B > $P5/dkt/phase2/engine_$c.log 2>&1 & done
wait $(jobs -p) || echo "WARNING: Phase 2 engine"

step "4. Phase 3 on DKT and with count / FA weights"
cd $T/phase3
mkdir -p $P5/dkt/phase3 $P5/aal3_count/phase3 $P5/aal3_fa/phase3
IIT_P1=$P5/dkt/phase1 IIT_P2=$P5/dkt/phase2 IIT_P3=$P5/dkt/phase3 $PY -I p3_engine.py uniform_v3_dkt $R > $P5/dkt/phase3/engine.log 2>&1 &
for w in count fa; do
  IIT_P1=$U/phase1_uniform IIT_P2=$U/phase2_uniform IIT_P3=$P5/aal3_$w/phase3 IIT_P3_WEIGHT=$w $PY -I p3_engine.py uniform_v3 $R > $P5/aal3_$w/phase3/engine.log 2>&1 &
done
wait $(jobs -p) || echo "WARNING: Phase 3 engine"
step "done"
