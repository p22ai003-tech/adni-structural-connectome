#!/bin/bash
# Phase 5 robustness on the rebuilt connectomes: second parcellation (DKT76) and other edge weights for the exception
# rule (streamline count, FA), plus the same-recipe recomputation on AAL3 with SIFT2 weights (aal3_fd) as the check that
# p5_features.py reproduces the pipeline's features. Each variant gets the Phase 1 engine (13 conditions) and tables; DKT
# also gets Phase 2 (length test, P0 and P2) and Phase 3 (null models); the weight variants get Phase 3.
#   run_phase5.sh [B=2000] [R=500]
# Refuses to run unless the connectomes carry the signed recipe (tools/iit_chain.py, exit 3).
set -uo pipefail
B=${1:-2000}; R=${2:-500}
U=/home/ec2-user/exp/docs/iit_dipanjan; T=$U/tools; PY=$T/.venv/bin/python; APY=/home/ec2-user/exp/.venv_connectome_app/bin/python
. "$T/chain_lib.sh"
if [ -z "${IIT_ROOT:-}" ]; then
  export IIT_AROOT=/data/derivatives/qc/analysis_cohort_uniform IIT_P1_COMPARE=$U/phase1
fi
chain_env run_phase5
P1=${IIT_P1:-$U/phase1_uniform}; P2=${IIT_P2:-$U/phase2_uniform}; P3=${IIT_P3:-$U/phase3_uniform}; P5=${IIT_P5:-$U/phase5_uniform}
DKTMAP=${IIT_DKT_MAP:-/data/derivatives/connectomes_uniform_v3/dkt76_network_mapping.csv}
if [ "${IIT_PRINT_PATHS:-0}" = 1 ]; then echo "IIT_AROOT=$IIT_AROOT P1=$P1 P2=$P2 P3=$P3 P5=$P5 DKTMAP=$DKTMAP"; exit 0; fi
step() { echo; echo "== $(date -u +%FT%TZ)  $*"; }
mkdir -p "$P5"
LOG=$P5/run_phase5.log; exec > >(tee -a "$LOG") 2>&1
CONDS="C0 C0q C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11"

# features <variant> <edges> <node-network map> <weight>: reused only when the inputs are the ones they were computed from.
features() {
  local name=$1 edges=$2 map=$3 w=$4 dir=$P5/$1/phase1 stamp
  stamp=$(cat "$edges" "$map" "$P1/covariates.csv" "$P1/qc.csv" "$T/phase5/p5_features.py" | sha256sum | cut -d' ' -f1)
  stamp="$stamp weight=$w sample=${IIT_PRIMARY_SAMPLE:-adni34}"
  if [ -f "$dir/features_qc.csv" ] && [ "$(cat "$dir/.features_stamp" 2>/dev/null)" = "$stamp" ]; then
    echo "features of $name: inputs unchanged, reused"; return 0
  fi
  IIT_P1=$P1 $PY -I $T/phase5/p5_features.py "$edges" "$map" "$dir" "$w" && echo "$stamp" > "$dir/.features_stamp"
}

if [ -n "${IIT_ROOT:-}" ]; then
  step "0. inputs of this chain: AAL3 node-network map from its tree, DKT edge cache from its matrices"
  [ -f "$P5/aal3_network_mapping.csv" ] || $APY -I - "$IIT_AROOT" "$P5/aal3_network_mapping.csv" <<'PYEOF'
import sys
import pandas as pd
m = pd.read_csv(f"{sys.argv[1]}/19_network_analysis/network_mapping_used.csv")
pd.DataFrame({"node": m.matrix_idx, "network": m.functional_network}).to_csv(sys.argv[2], index=False)
PYEOF
  [ -e "${IIT_CONN_DKT:-}" ] || { echo "no DKT matrices at ${IIT_CONN_DKT:-}: the DKT variant cannot run"; exit 1; }
  mkdir -p "$P5/dkt/phase2"
  (cd "$T/phase2" && IIT_P1=$P1 IIT_P2=$P5/dkt/phase2 $PY -I p2_edges.py uniform_v3_dkt) || exit 1
fi

step "1. features (recomputed unless the inputs are unchanged)"
for v in aal3_fd:fd aal3_count:count aal3_fa:fa; do
  name=${v%%:*}; w=${v##*:}
  features "$name" "$P2/edges_uniform_v3.npz" "$P5/aal3_network_mapping.csv" "$w"
done
features dkt "$P5/dkt/phase2/edges_uniform_v3_dkt.npz" "$DKTMAP" fd

step "2. Phase 1 engine, 13 conditions per variant"
cd $T/phase1
if [ -n "${IIT_LEDGER:-}" ]; then
  export -f chain_stage
  for name in dkt aal3_fd aal3_count aal3_fa; do
    for c in $CONDS; do echo "$name $c"; done
  done | xargs -P "${IIT_JOBS:-8}" -L1 bash -c 'IIT_P1='"$P5"'/$0/phase1 IIT_SRC=uniform_v3_$0 chain_stage p5_engine_$0_$1 '"$PY"' -I -W ignore p1_engine.py '"$B"' $1'
else
  for name in dkt aal3_fd aal3_count aal3_fa; do
    for c in $CONDS; do
      IIT_P1=$P5/$name/phase1 IIT_SRC=uniform_v3_$name $PY -I -W ignore p1_engine.py $B $c > $P5/$name/phase1/engine_$c.log 2>&1 &
    done
  done
  wait $(jobs -p) || echo "WARNING: at least one engine run failed"
fi
for name in dkt aal3_fd aal3_count aal3_fa; do
  (export IIT_P1=$P5/$name/phase1 IIT_SRC=uniform_v3_$name; chain_stage p5_merge_$name bash -c "$PY -I p1_merge.py > /dev/null; $PY -I -W ignore p1_tables_figs.py > /dev/null") || echo "WARNING: tables $name"
done

step "3. Phase 2 on DKT (P0, P2)"
cd $T/phase2
for c in P0 P2; do (IIT_P1=$P5/dkt/phase1 IIT_P2=$P5/dkt/phase2 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} chain_stage p5_dkt_p2_$c $PY -I p2_engine.py uniform_v3_dkt $c $B > $P5/dkt/phase2/engine_$c.log 2>&1) & done
wait $(jobs -p) || echo "WARNING: Phase 2 engine"

step "4. Phase 3 on DKT and with count / FA weights"
cd $T/phase3
mkdir -p $P5/dkt/phase3 $P5/aal3_count/phase3 $P5/aal3_fa/phase3
(IIT_P1=$P5/dkt/phase1 IIT_P2=$P5/dkt/phase2 IIT_P3=$P5/dkt/phase3 chain_stage p5_dkt_p3 $PY -I p3_engine.py uniform_v3_dkt $R > $P5/dkt/phase3/engine.log 2>&1) &
for w in count fa; do
  (IIT_P1=$P1 IIT_P2=$P2 IIT_P3=$P5/aal3_$w/phase3 IIT_P3_WEIGHT=$w chain_stage p5_${w}_p3 $PY -I p3_engine.py uniform_v3 $R > $P5/aal3_$w/phase3/engine.log 2>&1) &
done
wait $(jobs -p) || echo "WARNING: Phase 3 engine"
step "done"
