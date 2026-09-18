#!/bin/bash
cd /data/derivatives/connectomes
aws s3 sync . s3://sabeesh/exp/connectomes/ \
  --exclude "*" --include "SC_AAL166_*.csv" --include "SC_Schaefer200_*.csv" \
  --no-progress > /tmp/sync_connectomes.log 2>&1
echo "SYNC DONE rc=$? at $(date -u +%H:%M:%SZ)" >> /tmp/sync_connectomes.log
# verify a recovery landed
aws s3 ls s3://sabeesh/exp/connectomes/SC_AAL166_<SUBJECT>_I<IMAGEID>_count.csv >> /tmp/sync_connectomes.log 2>&1
