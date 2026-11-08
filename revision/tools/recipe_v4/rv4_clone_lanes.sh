#!/bin/bash
# Run on a copy of the server, over ssh from the main server (FREEZE.md decision 5, mid-run split of 5 Nov): make sure the PE app and its backups are
# off, read the tools once so the snapshot's lazily loaded blocks arrive before 32 sessions need them, then run this copy's list with rv4_lanes.sh.
# Usage: rv4_clone_lanes.sh <units file>
set -u; LIST=$1; T=/home/ec2-user/exp/docs/iit_dipanjan/tools/recipe_v4
[ -f /home/ec2-user/RECIPE_V4_CLONE ] || { echo "no RECIPE_V4_CLONE marker: this is not a copy, refusing"; exit 1; }
systemctl --user disable --now pe-quant-prod.service pe-review-backup.timer pe-prod-review-backup.timer pe-review-backup-verify.timer pe-prod-review-backup-verify.timer >/dev/null 2>&1
echo "PE units still active: $(systemctl --user list-units --all 'pe-*' --state=active --no-legend | wc -l)"
nice -n 19 find /home/ec2-user/fsl /home/ec2-user/FastSurfer /home/ec2-user/miniconda3/envs/fastsurfer /home/ec2-user/synb0disco /home/ec2-user/mrtrix3 \
  /home/ec2-user/exp/.envs /home/ec2-user/exp/.venv_connectome_app -type f -print0 2>/dev/null | xargs -0 -P 32 cat > /dev/null 2>&1
echo "tools read: $(TZ=Asia/Kolkata date '+%H:%M IST')"
( setsid nohup bash "$T/rv4_lanes.sh" "$LIST" 32 >> /data/derivatives/recipe_v4/cohort/logs/lanes.log 2>&1 < /dev/null & )
sleep 5; tail -2 /data/derivatives/recipe_v4/cohort/logs/lanes.log
