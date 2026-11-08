#!/bin/bash
# Uniform rebuild v3, self-sizing driver: as many 10-thread workers as the machine has cores for (minus a reserve for
# other services), unfinished participants only, lowest CPU priority. Safe to start any number of times: a driver lock
# prevents two drivers, and per-participant locks in rb2_subject.py prevent duplicate work.
D=/data/derivatives/connectomes_uniform_v3
cd /home/ec2-user/exp/docs/iit_dipanjan/tools/rebuild || exit 1
exec 9>"$D/driver.lock"; flock -n 9 || { echo "driver already running"; exit 0; }
CORES=$(nproc); WORKERS=$(( (CORES - 6) / 10 )); [ $WORKERS -lt 1 ] && WORKERS=1
todo=$(mktemp)
while read -r sid; do
  unit=$(basename "$(ls -d /data/derivatives/tracks/${sid}_I* | head -1)")
  [ -f "$D/matrices/SC_AAL166_${unit}_qc.json" ] || echo "$sid"
done < "$D/all_sids.txt" > "$todo"
echo "$(date -u +%FT%TZ) driver start: cores=$CORES workers=$WORKERS to do=$(wc -l < "$todo")" >> "$D/run.log"
[ -s "$todo" ] && xargs -a "$todo" -P "$WORKERS" -I{} nice -n 19 /home/ec2-user/exp/.venv_connectome_app/bin/python -I \
    rb2_subject.py {} --threads 10 --from-dwi --out "$D" >> "$D/run.log" 2>&1
echo "$(date -u +%FT%TZ) driver end: done=$(ls $D/matrices/*_qc.json | wc -l)" >> "$D/run.log"
rm -f "$todo"
