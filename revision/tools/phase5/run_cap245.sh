#!/bin/bash
# Phase 5 check: edges whose mean streamline length is at the 250-mm tracking limit (>= 245 mm; 4% of edges, 18% of
# long-range edges) are removed, and the Phase 1 primary analysis and the Phase 2 length test are repeated.
# Refuses to run unless the connectomes carry the signed recipe (tools/iit_chain.py, exit 3).
U=/home/ec2-user/exp/docs/iit_dipanjan; T=$U/tools; PY=$T/.venv/bin/python
. "$T/chain_lib.sh"
chain_env run_cap245
P1=${IIT_P1:-$U/phase1_uniform}; P2=${IIT_P2:-$U/phase2_uniform}; P5=${IIT_P5:-$U/phase5_uniform}; C=$P5/cap245; B=${IIT_B:-2000}
if [ "${IIT_PRINT_PATHS:-0}" = 1 ]; then echo "P1=$P1 P2=$P2 P5=$P5"; exit 0; fi
mkdir -p $C/phase1 $C/phase2
$PY -I - "$P2/edges_uniform_v3.npz" "$C/phase2/edges_uniform_v3.npz" <<'PYEOF'
import sys
import numpy as np
E = dict(np.load(sys.argv[1], allow_pickle=True)); cap = E["len"] >= 245
for k in ("len", "fd", "count", "fa", "md", "rd", "ad"): E[k] = np.where(cap, np.nan, E[k]).astype(np.float32)
np.savez_compressed(sys.argv[2], **E); print("edges removed:", int(cap.sum()))
PYEOF
IIT_P1=$P1 $PY -I $T/phase5/p5_features.py $C/phase2/edges_uniform_v3.npz $P5/aal3_network_mapping.csv $C/phase1 fd
cd $T/phase1
if [ -n "${IIT_LEDGER:-}" ]; then
  export -f chain_stage
  printf '%s\n' C0 C1 C2 C3 C5 C8 | xargs -P "${IIT_JOBS:-8}" -I{} bash -c "IIT_P1=$C/phase1 IIT_SRC=uniform_v3_cap chain_stage p5_cap_engine_{} $PY -I -W ignore p1_engine.py $B {}"
else
  for c in C0 C1 C2 C3 C5 C8; do IIT_P1=$C/phase1 IIT_SRC=uniform_v3_cap $PY -I -W ignore p1_engine.py $B $c > $C/phase1/engine_$c.log 2>&1 & done; wait
fi
(export IIT_P1=$C/phase1 IIT_SRC=uniform_v3_cap IIT_P1_COMPARE=${IIT_P1_COMPARE:-$U/phase1}; $PY -I p1_merge.py > /dev/null; $PY -I -W ignore p1_tables_figs.py > /dev/null)
cd $T/phase2
for c in P0 P2; do (IIT_P1=$C/phase1 IIT_P2=$C/phase2 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} chain_stage p5_cap_p2_$c $PY -I p2_engine.py uniform_v3 $c $B > $C/phase2/engine_$c.log 2>&1) & done; wait
echo CAP_DONE
