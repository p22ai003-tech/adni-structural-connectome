#!/bin/bash
# The whole chain under one root, stage by stage: tree, Phases 1-7, document builders. Every stage's result goes into the
# ledger ($IIT_ROOT/logs/chain_status.tsv); a restart skips what is done. nice 19, at most IIT_JOBS processes (default 8).
#
#   IIT_ROOT=<root> SC_DENSITY_FLOOR=.. SC_SMC=.. IIT_PRIMARY_SAMPLE=.. IIT_LENGTH_MEASURE=.. SC_LABELS=.. SC_LABEL_SEED=.. \
#     run_chain.sh --connectomes DIR --dkt DIR --other-arm EDGES.npz --version NAME [--rehearsal | --no-act-base] [--docs-only] [--no-docs] [--no-phase5]
#
# Without --rehearsal it refuses unless the connectomes carry the signed recipe (exit 3). With --rehearsal the bypass is
# written into chain_settings.json, REHEARSAL_UNSIGNED.txt and the first line of every document. A rehearsal first runs the
# document builders strictly (stopping at a contradicted sentence is the expected result on shuffled labels), then again
# with the contradictions recorded so that the rest of each builder is exercised. --no-act-base is for the existing no-ACT
# connectomes only (iit_chain.NO_ACT_BASE): recorded in chain_settings.json and NO_ACT_BASE_UNSIGNED.txt, guard strict, no banner.
set -uo pipefail
renice -n 19 $$ > /dev/null 2>&1 || true
U=/home/ec2-user/exp/docs/iit_dipanjan; T=$U/tools; PY=$T/.venv/bin/python; APY=/home/ec2-user/exp/.venv_connectome_app/bin/python
. "$T/chain_lib.sh"
CONN=""; DKT=""; OTHER=""; VERSION=""; REHEARSE=""; NO_ACT=""; DOCS_ONLY=""; NO_DOCS=""; NO_P5=""
while [ $# -gt 0 ]; do
  case "$1" in
    --connectomes) CONN=$2; shift 2 ;;
    --dkt) DKT=$2; shift 2 ;;
    --other-arm) OTHER=$2; shift 2 ;;
    --version) VERSION=$2; shift 2 ;;
    --rehearsal) REHEARSE=1; shift ;;
    --no-act-base) NO_ACT=1; shift ;;
    --docs-only) DOCS_ONLY=1; shift ;;
    --no-docs) NO_DOCS=1; shift ;;
    --no-phase5) NO_P5=1; shift ;;
    --real-labels) export SC_LABELS=real; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "${IIT_ROOT:-}" ] || { echo "run_chain.sh needs IIT_ROOT" >&2; exit 2; }
[ -n "$CONN" ] && [ -n "$VERSION" ] || { echo "run_chain.sh needs --connectomes and --version" >&2; exit 2; }
$APY -I "$T/iit_chain.py" init --connectomes "$CONN" ${DKT:+--dkt "$DKT"} ${REHEARSE:+--rehearsal} ${NO_ACT:+--no-act-base} || exit $?
chain_env run_chain
export IIT_LEDGER=$IIT_LOGS/chain_status.tsv IIT_DOC_VERSION=$VERSION IIT_JOBS=${IIT_JOBS:-8} IIT_B=${IIT_B:-2000} IIT_R=${IIT_R:-500}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=$IIT_JOBS MPLBACKEND=Agg
export IIT_OTHER_ARM_EDGES=${OTHER:-${IIT_OTHER_ARM_EDGES:-}}
mkdir -p "$IIT_LOGS"
[ -f "$IIT_LEDGER" ] || printf 'stage\texit\tseconds\tlast line\n' > "$IIT_LEDGER"

if [ -z "$DOCS_ONLY" ]; then
  chain_stage exclusions $APY -I "$T/iit_chain.py" exclusions
  chain_stage build_tree "$T/rebuild/build_tree.sh" "$IIT_AROOT"
  $APY -I - "$IIT_AROOT/exports/stage_status_final.csv" "$IIT_LEDGER" <<'PYEOF'
import csv, os, sys
if os.path.exists(sys.argv[1]):            # each stage of the tree, once, in the same ledger
    have = open(sys.argv[2]).read()
    with open(sys.argv[2], "a") as out:
        for r in csv.DictReader(open(sys.argv[1])):
            if f"tree_{r['stage']}\t" not in have:
                out.write(f"tree_{r['stage']}\t{0 if r['status'] in ('ok', 'skipped') else 1}\t{r.get('seconds', '')}\t{r['status']}\n")
PYEOF
  SKIP_ANALYSIS=1 "$T/rebuild/refresh_all.sh" full
  cd "$T/phase4"
  chain_stage p4_data $PY -I p4_data.py
  chain_stage p4_engine $PY -I p4_engine.py "${IIT_P4_PERM:-500}" "${IIT_P4_REPEATS:-20}"
  chain_stage p4_tables $PY -I p4_tables_figs.py
  if [ -z "$NO_P5" ]; then
    "$T/phase5/run_phase5.sh" "$IIT_B" "$IIT_R"
    cd "$T/phase5"
    chain_stage p5_cap245 "$T/phase5/run_cap245.sh"
    [ -z "$IIT_OTHER_ARM_EDGES" ] || chain_stage p5_act $PY -I p5_act.py
    chain_stage p5_tables $PY -I p5_tables_figs.py
    cd "$T/phase6"
    chain_stage p6_figures $APY -I p6_figures.py
  fi
fi
[ -z "$NO_DOCS" ] || { echo "chain finished, no documents: see $IIT_LEDGER"; exit 0; }

# ------------------------------------------------------------------ documents
docs() {   # docs <suffix>: every builder once; the version name differs per pass so that nothing is overwritten
  local tag=$1
  ( export IIT_DOC_VERSION=$VERSION$tag
    "$T/rebuild/docs_rebuilt.sh" "$IIT_DOC_VERSION"
    cd "$T/phase4" && chain_stage "docs_p4_docx_$IIT_DOC_VERSION" $PY -I p4_docx.py
    cd "$T/phase5" && chain_stage "docs_p5_docx_$IIT_DOC_VERSION" $PY -I p5_docx.py
    cd "$T/phase6" && chain_stage "docs_p6_docx_$IIT_DOC_VERSION" $PY -I p6_docx.py
    cd "$T/phase7"
    chain_stage "docs_p7_ms_$IIT_DOC_VERSION" "$T/phase7/build_ms.sh" "$IIT_DOC_VERSION"
    chain_stage "docs_p7_supplement_$IIT_DOC_VERSION" $PY -I p7_supplement.py
    chain_stage "docs_p7_thesis_$IIT_DOC_VERSION" "$T/phase7/build_thesis.sh" "$IIT_DOC_VERSION" )
}
docs ""
if [ -n "$REHEARSE" ]; then
  IIT_NARRATIVE_GUARD=record docs _recorded
fi
echo "chain finished: see $IIT_LEDGER"
