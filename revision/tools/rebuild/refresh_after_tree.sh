#!/bin/bash
# Start the phase refresh (SKIP_ANALYSIS=1) as soon as build_tree.sh has written the range-restricted network tables
# (its step 2); the ML stages of step 3 keep running in parallel.
D=/data/derivatives/connectomes_uniform_v3; U=/home/ec2-user/exp/docs/iit_dipanjan
until grep -q "3. resume" $D/build_tree.out; do
  sleep 20
  pgrep -f "tools/rebuild/build_tree.sh" > /dev/null || { echo "build_tree.sh stopped before step 3" >> $D/refresh_n520.out; exit 1; }
done
cd $U && SKIP_ANALYSIS=1 tools/rebuild/refresh_all.sh full > $D/refresh_n520.out 2>&1
echo "REFRESH_EXIT=$?" >> $D/refresh_n520.out
