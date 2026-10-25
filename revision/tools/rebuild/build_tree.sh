#!/bin/bash
# Build the analysis tree on the uniformly rebuilt connectomes: every stage of connectome_analysis.run_analysis (--all,
# 35 stages, including the ML stages the thesis reports), with the rebuild exclusions (exclusions_uniform.yaml).
#   build_tree.sh [<analysis root>]      default /data/derivatives/qc/analysis_cohort_uniform
# The pipeline builds its cohort from every connectome in the matrix folder (density >= 0.6), so the rebuild exclusions are
# applied by giving it a folder that holds only the connectomes passing rebuild quality control: matrices_qc/, symlinks
# to matrices/ minus the participants in exclusions_uniform.yaml (step 0). Phase scripts still read matrices/ (IIT_CONN)
# so that the excluded connectomes are audited too.
set -uo pipefail
RB=/home/ec2-user/exp/docs/iit_dipanjan/tools/rebuild; APY=/home/ec2-user/exp/.venv_connectome_app/bin/python
ROOT=${1:-/data/derivatives/qc/analysis_cohort_uniform}
export SC_CONNECTOMES_DIR=/data/derivatives/connectomes_uniform_v3/matrices_qc SC_EXCLUSIONS=$RB/exclusions_uniform.yaml
mkdir -p "$ROOT"
sed "1i figs_dir: $ROOT/figs" /home/ec2-user/exp/configs/analysis.yaml > "$ROOT/analysis_config.yaml"
run() { (cd /home/ec2-user/exp && $APY -m connectome_analysis.run_analysis "$@" --connectomes-dir "$SC_CONNECTOMES_DIR" \
        --exclusions "$SC_EXCLUSIONS" --analysis-root "$ROOT" --config "$ROOT/analysis_config.yaml" --resume); }
step() { echo; echo "== $(date -u +%FT%TZ)  $*"; }

step "0. matrices_qc/: the rebuilt connectomes that pass rebuild quality control"
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
step "1. all stages"
run --all
cp "$ROOT/exports/stage_status.csv" "$ROOT/exports/stage_status_final.csv"
step "done: $(grep -c ',ok,\|,skipped,' "$ROOT/exports/stage_status.csv") of 35 stages ok or current"
