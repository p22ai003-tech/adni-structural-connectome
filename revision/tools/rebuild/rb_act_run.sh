#!/bin/bash
# Phase 5 tractography-settings check: ACT re-run for the 45 participants in units.txt (15 CN, 15 MCI, 15 AD; ADNI-3/4,
# passing quality control; random, seed 2026). 8 workers x 8 threads, low priority.
O=/data/derivatives/connectomes_uniform_v3_act
cat $O/units.txt | xargs -P 8 -I{} nice -n 10 /home/ec2-user/exp/.venv_connectome_app/bin/python -I \
    /home/ec2-user/exp/docs/iit_dipanjan/tools/rebuild/rb_act_subject.py {} --threads 8 >> $O/run.log 2>&1
echo "$(date -u +%FT%TZ) finished: $(ls $O/matrices/*_qc.json 2>/dev/null | wc -l) of $(wc -l < $O/units.txt)" >> $O/run.log
