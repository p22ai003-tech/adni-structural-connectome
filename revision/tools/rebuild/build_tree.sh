#!/bin/bash
# Build the analysis tree on the uniformly rebuilt connectomes: every stage of connectome_analysis.run_analysis (--all,
# 35 stages, including the ML stages the thesis reports), with the rebuild exclusions (exclusions_uniform.yaml).
#   build_tree.sh [<analysis root>]      default /data/derivatives/qc/analysis_cohort_uniform, or the chain's tree under IIT_ROOT
# The pipeline builds its cohort from every connectome in the matrix folder (density >= the floor, SC_DENSITY_FLOOR), so the
# rebuild exclusions are applied by giving it a folder that holds only the connectomes passing rebuild quality control:
# matrices_qc/, symlinks to matrices/ minus the participants in the exclusions file (step 0). Phase scripts still read
# matrices/ (IIT_CONN) so that the excluded connectomes are audited too.
# Refuses to run unless the connectomes carry the signed recipe (tools/iit_chain.py, exit 3).
set -uo pipefail
RB=/home/ec2-user/exp/docs/iit_dipanjan/tools/rebuild; APY=/home/ec2-user/exp/.venv_connectome_app/bin/python
. "$RB/../chain_lib.sh"
chain_env build_tree
ROOT=${1:-${IIT_AROOT:-/data/derivatives/qc/analysis_cohort_uniform}}
export SC_CONNECTOMES_DIR=${IIT_CONN_QC:-/data/derivatives/connectomes_uniform_v3/matrices_qc} SC_EXCLUSIONS=${IIT_EXCLUSIONS:-$RB/exclusions_uniform.yaml}
if [ "${IIT_PRINT_PATHS:-0}" = 1 ]; then echo "ROOT=$ROOT SC_CONNECTOMES_DIR=$SC_CONNECTOMES_DIR SC_EXCLUSIONS=$SC_EXCLUSIONS"; exit 0; fi
mkdir -p "$ROOT"
sed "1i figs_dir: $ROOT/figs" /home/ec2-user/exp/configs/analysis.yaml > "$ROOT/analysis_config.yaml"
run() { (cd /home/ec2-user/exp && $APY -m connectome_analysis.run_analysis "$@" --connectomes-dir "$SC_CONNECTOMES_DIR" \
        --exclusions "$SC_EXCLUSIONS" --analysis-root "$ROOT" --config "$ROOT/analysis_config.yaml" --resume); }
step() { echo; echo "== $(date -u +%FT%TZ)  $*"; }

step "0. matrices_qc/: the rebuilt connectomes that pass rebuild quality control"
if [ -n "${IIT_ROOT:-}" ]; then
  [ -f "$SC_EXCLUSIONS" ] || $APY -I "$CHAIN_T/iit_chain.py" exclusions || exit 3
  $APY -I "$CHAIN_T/iit_chain.py" views || exit 3
else
python3 - <<'PYEOF'
import os, glob, yaml
D = "/data/derivatives/connectomes_uniform_v3"; Q = f"{D}/matrices_qc"; os.makedirs(Q, exist_ok=True)
excl = set(yaml.safe_load(open(os.environ["SC_EXCLUSIONS"]))["exclude_subjects"])
for f in glob.glob(f"{Q}/*"):                                    # drop links of newly excluded participants
    if os.path.basename(f)[10:20] in excl: os.remove(f)
for f in sorted(glob.glob(f"{D}/matrices/SC_AAL166_*")):
    b = os.path.basename(f)
    if b[10:20] not in excl and not os.path.lexists(f"{Q}/{b}"): os.symlink(f, f"{Q}/{b}")
print(len({os.path.basename(f)[10:20] for f in glob.glob(f"{Q}/*_count.csv")}), "participants in matrices_qc")
PYEOF
fi
if [ "${IIT_LENGTH_MEASURE:-per_person}" = control_median_act ] && [ -z "${IIT_LENGTH_REF:-}" ]; then
  step "0b. master cohort, then the control-median ruler it names, then matrices_qc with that ruler"
  run --stage analysis_snapshot --stage master_cohort || exit 1
  $APY -I "$CHAIN_T/iit_chain.py" views --ruler || exit 3
fi
step "1. all stages"
run --all
RUN_EXIT=$?
cp "$ROOT/exports/stage_status.csv" "$ROOT/exports/stage_status_final.csv"
step "done: $(grep -c ',ok,\|,skipped,' "$ROOT/exports/stage_status.csv") of 35 stages ok or current"
[ -z "${IIT_ROOT:-}" ] || exit $RUN_EXIT
