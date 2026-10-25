#!/bin/bash
# Re-run Phases 1-3 on the uniformly rebuilt connectomes (24-25 Oct 2026).
#   refresh_all.sh            full run (B = 2000 bootstrap / permutation resamples, R = 500 splits)
#   refresh_all.sh dry        quick check of the whole chain on whatever is rebuilt so far (B = 20, R = 20), separate folders
# Outputs: analysis tree $IIT_AROOT, phase1_uniform/, phase2_uniform/, phase3_uniform/ (published-data folders are untouched).
set -euo pipefail
MODE=${1:-full}
U=/home/ec2-user/exp/docs/iit_dipanjan; T=$U/tools; PY=$T/.venv/bin/python; APY=/home/ec2-user/exp/.venv_connectome_app/bin/python
SUF=_uniform; B=2000; R=500
if [ "$MODE" = dry ]; then SUF=_uniform_dry; B=20; R=20; fi
export IIT_CONN=/data/derivatives/connectomes_uniform_v3/matrices
export IIT_AROOT=/data/derivatives/qc/analysis_cohort$SUF
export IIT_P1=$U/phase1$SUF IIT_P2=$U/phase2$SUF IIT_P3=$U/phase3$SUF IIT_SUFFIX=$SUF IIT_SRC=uniform_v3
export IIT_P1_COMPARE=$U/phase1                       # 'what changed' compares with the published-data run
mkdir -p "$IIT_P1" "$IIT_P2" "$IIT_P3" "$IIT_AROOT"
LOG=$IIT_P1/refresh.log; exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "== $(date -u +%FT%TZ)  $*"; }

step "1. analysis tree on the rebuilt matrices: all 35 stages, rebuild exclusions (build_tree.sh; SKIP_ANALYSIS=1 skips)"
[ "${SKIP_ANALYSIS:-0}" = 1 ] || $T/rebuild/build_tree.sh "$IIT_AROOT"

step "2. Phase 1: features, quality control, 13 conditions, tables"
cp "$U/phase1/covariates.csv" "$IIT_P1/"               # same participants and sessions: covariates do not change
cd "$T/phase1"
$PY -I p1_features.py
$PY -I -W ignore p1_audit.py | tail -5
for c in C0 C0q C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11; do $PY -I -W ignore p1_engine.py $B $c > "$IIT_P1/engine_$c.log" 2>&1 & done; wait $(jobs -p) || echo "WARNING: at least one engine run failed (see its log)"
$PY -I p1_merge.py; $PY -I -W ignore p1_tables_figs.py | tail -3; $PY -I p1_number_changes.py > /dev/null || echo "WARNING: number-changes table failed"
# p1_qc_tables.py and p1_narrative.py are run by hand after reviewing the rebuilt results (the wording changes)

step "3. Phase 2: edge cache, four conditions, tables and figures"
cd "$T/phase2"
$PY -I p2_edges.py uniform_v3
for c in P0 P2 P10 P11; do OMP_NUM_THREADS=4 $PY -I p2_engine.py uniform_v3 $c $B > "$IIT_P2/engine_$c.log" 2>&1 & done; wait $(jobs -p) || echo "WARNING: at least one engine run failed (see its log)"
$PY -I p2_tables_figs.py uniform_v3 2>&1 | tail -1

step "4. Phase 3: exceptions, null models, tables and figures"
cd "$T/phase3"
$PY -I p3_engine.py uniform_v3 $R > "$IIT_P3/engine.log" 2>&1; tail -3 "$IIT_P3/engine.log"
$PY -I p3_tables_figs.py uniform_v3

step "5. notebooks (executed copies with the suffix $SUF)"
cd "$T/phase1" && $PY -I build_phase1_notebook.py
cd "$T/phase2" && $PY -I build_phase2_notebook.py
cd "$T/phase3" && $PY -I build_phase3_notebook.py
cd "$U/notebooks"
for nb in phase1_confound_control$SUF phase2_length_test$SUF phase3_exception_nulls$SUF; do
  $T/.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 \
      --ExecutePreprocessor.kernel_name=iit-phase1 "$nb.ipynb" > "$IIT_P1/nb_$nb.log" 2>&1 || echo "notebook $nb: see $IIT_P1/nb_$nb.log"
done
step "done"
