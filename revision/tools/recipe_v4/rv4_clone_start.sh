#!/bin/bash
# Run on a clone of the server (FREEZE.md decision 5), over ssh from the main server: make sure the PE app and its backups are off, read the tools once so
# the snapshot's lazily loaded blocks arrive before 21 jobs need them, then start this machine's slice of the cohort.
# Usage: rv4_clone_start.sh <k> <n>      (slice k of n; the main server runs slice 0)
set -u; k=$1; n=$2
[ -f /home/ec2-user/RECIPE_V4_CLONE ] || { echo "no RECIPE_V4_CLONE marker: this is not a clone, refusing"; exit 1; }
systemctl --user disable --now pe-quant-prod.service pe-review-backup.timer pe-prod-review-backup.timer pe-review-backup-verify.timer pe-prod-review-backup-verify.timer >/dev/null 2>&1
echo "PE units still active: $(systemctl --user list-units --all 'pe-*' --state=active --no-legend | wc -l)"
nice -n 19 find /home/ec2-user/fsl /home/ec2-user/FastSurfer /home/ec2-user/miniconda3/envs/fastsurfer /home/ec2-user/synb0disco /home/ec2-user/mrtrix3 \
  /home/ec2-user/exp/.envs /home/ec2-user/exp/.venv_connectome_app -type f -print0 2>/dev/null | xargs -0 -P 32 cat > /dev/null 2>&1
echo "tools read: $(date -u +%T)"
cd /data/derivatives/recipe_v4/scratch && ( SLICE=$k/$n setsid nohup bash /home/ec2-user/exp/docs/iit_dipanjan/tools/recipe_v4/rv4_cohort.sh all \
  >> /data/derivatives/recipe_v4/cohort/logs/driver_stdout.log 2>&1 < /dev/null & )
sleep 5; tail -2 /data/derivatives/recipe_v4/cohort/logs/driver.log
