#!/bin/bash
# The analysis chains on recipe v4 (FREEZE.md, settings agreed 5 Nov 23:02 IST), each under its own root in recipe_v4/chains/.
#   rv4_chains.sh rehearsal            shuffled diagnoses (seed 20261106) on the ACT 10M matrices present now; no documents
#   rv4_chains.sh act10m | noact3m | act3m | act10m_noflag     real diagnoses; noact3m and act3m take act10m's length ruler;
#                                      act3m carries the paired ACT vs no-ACT comparison (--other-arm noact3m); act10m_noflag = act10m
#                                      without the flagged sessions (WM Dice < .65 by the R4 rule, < 30 diffusion-weighted volumes by list)
# Inputs are folders of links to recipe_v4/derived made at launch, complete sessions only (qc.json is written last). Log: chains/<name>.log
set -u; MODE=$1; R4=/data/derivatives/recipe_v4; D=$R4/derived; C=$R4/chains; TOOLS=/home/ec2-user/exp/docs/iit_dipanjan/tools
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M IST') $*"; }
links() {   # links <derived subfolder> <prefix> <dest> [exclude file]: every complete session, minus the excluded units
  local src=$D/$1 p=$2 dst=$3 ex=${4:-/dev/null}; mkdir -p "$dst"
  for q in "$src"/${p}_*_qc.json; do u=${q#$src/${p}_}; u=${u%_qc.json}; grep -qx "$u" "$ex" && continue; for f in "$src"/${p}_${u}_*; do ln -sf "$f" "$dst/"; done; done
  echo "$(ls "$dst"/*_qc.json | wc -l)"; }
export SC_DENSITY_FLOOR=none SC_SMC=hold_out IIT_PRIMARY_SAMPLE=adni3 IIT_LENGTH_MEASURE=mni_distance SC_N_NODES=154 \
       IIT_R4_FIELD=wm_dice IIT_R4_MIN=0 IIT_FASTSURFER=${IIT_FASTSURFER:-/data/derivatives/fastsurfer} IIT_JOBS=${IIT_JOBS:-16} SC_LABELS=real
EXTRA=(); ARM=act10m; NAME=$MODE${RV4_SUFFIX:-}   # RV4_SUFFIX: a second root of the same mode (6 Nov: same-visit FastSurfer view)
case "$MODE" in
  rehearsal) export SC_LABELS=shuffled SC_LABEL_SEED=20261106; EXTRA=(--rehearsal --no-docs) ;;
  act10m) EXTRA=(--no-docs) ;;
  noact3m|act3m) ARM=$MODE; EXTRA=(--no-docs); export IIT_LENGTH_REF=$C/act10m/connectomes/ruler_mni_distance.npz
         [ -f "$IIT_LENGTH_REF" ] || { echo "needs act10m's ruler first: $IIT_LENGTH_REF"; exit 2; }
         if [ "$MODE" = act3m ]; then o=$C/noact3m/phase2/edges_uniform_v3.npz; [ -f "$o" ] || { echo "needs noact3m's edges first: $o"; exit 2; }; EXTRA+=(--other-arm "$o"); fi ;;
  act10m_noflag) EXTRA=(--no-docs); export IIT_R4_MIN=0.65; EX=$C/inputs/under30_dw_volumes.txt ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
export IIT_ROOT=$C/$NAME
[ -e "$IIT_ROOT" ] && { echo "$IIT_ROOT exists: a chain root is used once"; exit 2; }
IN=$C/inputs/$NAME; [ -e "$IN" ] && { echo "$IN exists"; exit 2; }
na=$(links aal_$ARM SC_AAL166 "$IN/aal" "${EX:-/dev/null}"); nd=$(links dkt_$ARM SC_DKT76 "$IN/dkt" "${EX:-/dev/null}")
say "$NAME: $na AAL3 + $nd DKT sessions; labels $SC_LABELS; R4 wm_dice >= $IIT_R4_MIN; jobs $IIT_JOBS; ${EXTRA[*]}"
cd "$TOOLS" && bash "$TOOLS/run_chain.sh" --connectomes "$IN/aal" --dkt "$IN/dkt" --version "rv4_$NAME" "${EXTRA[@]}"
rc=$?; say "$NAME: run_chain.sh exit $rc; ledger $IIT_ROOT/logs/chain_status.tsv: $(awk -F'\t' 'NR>1 && $2!=0' "$IIT_ROOT/logs/chain_status.tsv" 2>/dev/null | wc -l) stages not ok"
exit $rc
