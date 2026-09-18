#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc/assignment_pilot
HYBRID_ROOT=/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc/hybrid_parcellation_pilot/hybrid_033_20260517T1314/033_S_7114_I11063036
SID=033_S_7114_I11063036

run_one() {
  local label="$1"
  local parc="$2"
  local stamp="$3"
  mkdir -p "$ROOT/$stamp"
  echo "$label" > "$ROOT/$stamp/route_label.txt"
  /home/ec2-user/fsl/bin/python /home/ec2-user/exp/run_sc_assignment_pilot.py \
    --stamp "$stamp" \
    --sid "$SID" \
    --parc-path "$parc" \
    --variants forward30 forward40 \
    --weights count fd_sum \
    > "$ROOT/$stamp/stdout.log" \
    2> "$ROOT/$stamp/stderr.log"
}

STAMP_BASE="033_hybrid_basefill_forward_$(date -u +%Y%m%dT%H%M%SZ)"
echo "$STAMP_BASE" > "$ROOT/latest_033_hybrid_basefill_stamp.txt"
run_one "hybrid_basefill_t1brain" "$HYBRID_ROOT/AAL_b0_hybrid_basefill_t1brain.nii.gz" "$STAMP_BASE"

STAMP_OVER="033_hybrid_missing_overwrite_forward_$(date -u +%Y%m%dT%H%M%SZ)"
echo "$STAMP_OVER" > "$ROOT/latest_033_hybrid_missing_overwrite_stamp.txt"
run_one "hybrid_missing_overwrite_t1brain" "$HYBRID_ROOT/AAL_b0_hybrid_missing_overwrite_t1brain.nii.gz" "$STAMP_OVER"
