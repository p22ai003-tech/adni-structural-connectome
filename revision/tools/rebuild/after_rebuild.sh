#!/bin/bash
# Waits for the uniform rebuild to finish, completes the second parcellation for the first participants (processed
# before the DKT step existed: their AAL3 outputs are set aside and the participant is rebuilt in full, so AAL3 and
# DKT come from the same streamlines), then runs the full refresh of Phases 1-3.
D=/data/derivatives/connectomes_uniform_v3; R=/home/ec2-user/exp/docs/iit_dipanjan/tools/rebuild
LOG=$D/after_rebuild.log; exec >> "$LOG" 2>&1
echo "$(date -u +%FT%TZ) waiting for 530 participants"
until [ "$(ls $D/matrices/*_qc.json 2>/dev/null | wc -l)" -ge 530 ] && ! pgrep -f '[r]b2_subject.py' > /dev/null; do sleep 120; done
echo "$(date -u +%FT%TZ) rebuild complete: $(ls $D/matrices/*_qc.json | wc -l) AAL3, $(ls $D/matrices_dkt/*_qc.json 2>/dev/null | wc -l) DKT"
mkdir -p $D/set_aside_no_dkt
missing=()
for q in $D/matrices/SC_AAL166_*_qc.json; do
  unit=$(basename "$q" _qc.json); unit=${unit#SC_AAL166_}
  [ -f "$D/matrices_dkt/SC_DKT76_${unit}_qc.json" ] || missing+=("$unit")
done
echo "participants without DKT: ${#missing[@]}"
for unit in "${missing[@]}"; do mv $D/matrices/SC_AAL166_${unit}_* $D/set_aside_no_dkt/; done
cd $R
printf '%s\n' "${missing[@]}" | sed 's/_I[0-9]*$//' | xargs -P 18 -I{} nice -n 10 /home/ec2-user/exp/.venv_connectome_app/bin/python -I \
    rb2_subject.py {} --threads 10 --from-dwi --out $D >> $D/run.log 2>&1
echo "$(date -u +%FT%TZ) after backfill: $(ls $D/matrices/*_qc.json | wc -l) AAL3, $(ls $D/matrices_dkt/*_qc.json | wc -l) DKT"
[ "$(ls $D/matrices/*_qc.json | wc -l)" -ge 530 ] || { echo "AAL3 incomplete after backfill: stopping before the refresh"; exit 1; }
echo "$(date -u +%FT%TZ) starting the full refresh"
$R/refresh_all.sh full > $D/refresh_full.out 2>&1 && echo "$(date -u +%FT%TZ) refresh finished" || echo "$(date -u +%FT%TZ) refresh FAILED: see $D/refresh_full.out"
