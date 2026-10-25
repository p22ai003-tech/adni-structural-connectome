#!/bin/bash
# Phase 5 check: edges whose mean streamline length is at the 250-mm tracking limit (>= 245 mm; 4% of edges, 18% of
# long-range edges) are removed, and the Phase 1 primary analysis and the Phase 2 length test are repeated.
U=/home/ec2-user/exp/docs/iit_dipanjan; T=$U/tools; PY=$T/.venv/bin/python; C=$U/phase5_uniform/cap245
$PY -I - <<'PYEOF'
import numpy as np
U = "/home/ec2-user/exp/docs/iit_dipanjan"
E = dict(np.load(f"{U}/phase2_uniform/edges_uniform_v3.npz", allow_pickle=True)); cap = E["len"] >= 245
for k in ("len", "fd", "count", "fa", "md", "rd", "ad"): E[k] = np.where(cap, np.nan, E[k]).astype(np.float32)
np.savez_compressed(f"{U}/phase5_uniform/cap245/phase2/edges_uniform_v3.npz", **E); print("edges removed:", int(cap.sum()))
PYEOF
IIT_P1=$U/phase1_uniform $PY -I $T/phase5/p5_features.py $C/phase2/edges_uniform_v3.npz $U/phase5_uniform/aal3_network_mapping.csv $C/phase1 fd
cd $T/phase1
for c in C0 C1 C2 C3 C5 C8; do IIT_P1=$C/phase1 IIT_SRC=uniform_v3_cap $PY -I -W ignore p1_engine.py 2000 $c > $C/phase1/engine_$c.log 2>&1 & done; wait
(export IIT_P1=$C/phase1 IIT_SRC=uniform_v3_cap IIT_P1_COMPARE=$U/phase1; $PY -I p1_merge.py > /dev/null; $PY -I -W ignore p1_tables_figs.py > /dev/null)
cd $T/phase2
for c in P0 P2; do IIT_P1=$C/phase1 IIT_P2=$C/phase2 OMP_NUM_THREADS=4 $PY -I p2_engine.py uniform_v3 $c 2000 > $C/phase2/engine_$c.log 2>&1 & done; wait
echo CAP_DONE
