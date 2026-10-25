#!/bin/bash
# Uniform rebuild v3 for all 530 participants: 5 at a time x 10 threads, lowest CPU priority so other services keep priority.
D=/data/derivatives/connectomes_uniform_v3
cd /home/ec2-user/exp/docs/iit_dipanjan/tools/rebuild
cat $D/all_sids.txt | xargs -P 5 -I{} nice -n 19 /home/ec2-user/exp/.venv_connectome_app/bin/python -I rb2_subject.py {} --threads 10 --from-dwi --out $D >> $D/run.log 2>&1
echo "ALL DONE $(date -u)" >> $D/run.log
