#!/bin/bash
# Waits for the ACT re-tracking (rb_act_run.sh) to finish, then compares the settings and refreshes the Phase 5 report.
# Needs IIT_DOC_VERSION: a new version name, nothing is overwritten.
O=/data/derivatives/connectomes_uniform_v3_act; U=/home/ec2-user/exp/docs/iit_dipanjan
[ -n "${IIT_DOC_VERSION:-}" ] || { echo "after_act.sh needs IIT_DOC_VERSION" >&2; exit 2; }
until grep -q "finished:" $O/run.log 2>/dev/null; do sleep 120; done
cd $U && tools/.venv/bin/python -I tools/phase5/p5_act.py > phase5_uniform/act_compare.log 2>&1 && \
  tools/.venv/bin/python -I tools/phase5/p5_docx.py >> phase5_uniform/act_compare.log 2>&1 && echo "ACT_DONE" >> phase5_uniform/act_compare.log \
  || echo "ACT_FAILED" >> phase5_uniform/act_compare.log
