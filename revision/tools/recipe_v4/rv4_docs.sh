#!/bin/bash
# Document builders for one recipe v4 chain root (the analysis is done; --docs-only), with the chain's own settings.
#   rv4_docs.sh <act10m|act10m_noflag|act3m> <new version name> [record]
# 'record' lets every builder finish and writes each contradicted sentence to <root>/docs/.guard/ instead of stopping (drafts).
# Phase 1 is updated in a copy of IIT_P1_SOURCE (default: the user's iteration-3 report). Log: chains/<arm>_docs_<version>.log
set -u; ARM=$1; VER=$2; MODE=${3:-strict}; R4=/data/derivatives/recipe_v4; C=$R4/chains; TOOLS=/home/ec2-user/exp/docs/iit_dipanjan/tools
export SC_DENSITY_FLOOR=none SC_SMC=hold_out IIT_PRIMARY_SAMPLE=adni3 IIT_LENGTH_MEASURE=mni_distance SC_N_NODES=154 \
       IIT_R4_FIELD=wm_dice IIT_R4_MIN=0 IIT_FASTSURFER=${IIT_FASTSURFER:-/data/derivatives/recipe_v4/derived/fastsurfer_same_visit} IIT_JOBS=${IIT_JOBS:-16} SC_LABELS=real
case "$ARM" in
  act10m) ;;
  act10m_noflag) export IIT_R4_MIN=0.65 ;;
  act3m) export IIT_LENGTH_REF=$C/act10m/connectomes/ruler_mni_distance.npz IIT_OTHER_ARM_EDGES=$C/noact3m/phase2/edges_uniform_v3.npz ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
export IIT_ROOT=$C/$ARM IIT_P1_SOURCE=${IIT_P1_SOURCE:-"/home/ec2-user/exp/docs/iit_dipanjan/iteration_3_received/iteration 3/Phase1_Confound_Control.docx"}
[ -d "$IIT_ROOT" ] || { echo "no chain root $IIT_ROOT"; exit 2; }
[ "$MODE" = record ] && export IIT_NARRATIVE_GUARD=record
cd "$TOOLS" && bash "$TOOLS/run_chain.sh" --connectomes "$C/inputs/$ARM/aal" --dkt "$C/inputs/$ARM/dkt" --version "$VER" --docs-only
