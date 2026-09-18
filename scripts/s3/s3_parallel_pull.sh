#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  s3_parallel_pull.sh [dry-run|run]

Description:
  Sync the exp dataset from S3 down to the EC2-native layout in parallel.

Behavior:
  - Pulls cohort/, atlas/, data/Images/, and derivatives stage folders
  - Prefers canonical S3 prefixes under s3://.../data/... when present
  - Falls back to legacy flat prefixes such as s3://.../biascorr_1/ when needed
  - Runs multiple aws s3 sync jobs in parallel
  - Writes per-job logs under ~/exp/logs/s3-pull/<timestamp>/

Environment overrides:
  S3_ROOT=s3://bucket/prefix      Source root (default: s3://sabeesh/exp)
  EXP_ROOT=/home/ec2-user/exp     Project root (default: ~/exp)
  PARALLEL_JOBS=4                 Max concurrent sync jobs (default: 4)
  LOG_ROOT=~/exp/logs/s3-pull     Log root (default: ~/exp/logs/s3-pull)
EOF
}

mode="${1:-dry-run}"
case "$mode" in
  dry-run|run) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac

EXP_ROOT="${EXP_ROOT:-$HOME/exp}"
DATA_ROOT="${DATA_ROOT:-$EXP_ROOT/data}"
S3_ROOT="${S3_ROOT:-s3://sabeesh/exp}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
LOG_ROOT="${LOG_ROOT:-$EXP_ROOT/logs/s3-pull}"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$LOG_ROOT/$RUN_TS"

mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$LOG_ROOT/latest"

count_s3_objects() {
  local prefix="$1"
  aws s3 ls "$prefix" --recursive --summarize 2>/dev/null | awk -F': ' '/Total Objects:/ {print $2}' | tail -n 1
}

prefix_has_objects() {
  local prefix="$1"
  local without_scheme bucket key_prefix object_count

  without_scheme="${prefix#s3://}"
  bucket="${without_scheme%%/*}"
  if [[ "$without_scheme" == "$bucket" ]]; then
    key_prefix=""
  else
    key_prefix="${without_scheme#*/}"
  fi

  object_count="$(
    aws s3api list-objects-v2 \
      --bucket "$bucket" \
      --prefix "$key_prefix" \
      --max-keys 1 \
      --query 'length(Contents)' \
      --output text 2>/dev/null
  )"
  [[ "$object_count" =~ ^[1-9][0-9]*$ ]]
}

choose_s3_prefix() {
  local preferred="$1"
  local legacy="${2:-}"
  if prefix_has_objects "$preferred"; then
    printf '%s' "$preferred"
    return
  fi

  if [[ -n "$legacy" ]] && prefix_has_objects "$legacy"; then
    printf '%s' "$legacy"
    return
  fi

  printf '%s' "$preferred"
}

REQUESTED_TARGETS=(
  "cohort|${S3_ROOT}/cohort/|$EXP_ROOT/cohort|"
  "atlas|${S3_ROOT}/atlas/|$EXP_ROOT/atlas|"
  "images_dti|${S3_ROOT}/data/Images/dti/|$DATA_ROOT/Images/dti|${S3_ROOT}/Images/dti/"
  "images_mri|${S3_ROOT}/data/Images/mri/|$DATA_ROOT/Images/mri|${S3_ROOT}/Images/mri/"
  "mif_dwi|${S3_ROOT}/data/derivatives/mif_dwi/|$DATA_ROOT/derivatives/mif_dwi|${S3_ROOT}/mif_dwi/"
  "mif_denoised|${S3_ROOT}/data/derivatives/mif_denoised/|$DATA_ROOT/derivatives/mif_denoised|${S3_ROOT}/mif_denoised/"
  "mif_unringed|${S3_ROOT}/data/derivatives/mif_unringed/|$DATA_ROOT/derivatives/mif_unringed|${S3_ROOT}/mif_unringed/"
  "eddy|${S3_ROOT}/data/derivatives/eddy/|$DATA_ROOT/derivatives/eddy|${S3_ROOT}/eddy/"
  "biascorr_1|${S3_ROOT}/data/derivatives/biascorr_1/|$DATA_ROOT/derivatives/biascorr_1|${S3_ROOT}/biascorr_1/"
  "dwi_t1_bbr|${S3_ROOT}/data/derivatives/dwi_t1_bbr/|$DATA_ROOT/derivatives/dwi_t1_bbr|${S3_ROOT}/dwi_t1_bbr/"
  "qc|${S3_ROOT}/data/derivatives/qc/|$DATA_ROOT/derivatives/qc|${S3_ROOT}/qc/"
)

TARGETS_FILE="$RUN_DIR/_targets.txt"
: > "$TARGETS_FILE"
for target in "${REQUESTED_TARGETS[@]}"; do
  IFS='|' read -r label preferred_src dest legacy_src <<<"$target"
  chosen_src="$(choose_s3_prefix "$preferred_src" "$legacy_src")"
  printf '%s|%s|%s\n' "$label" "$chosen_src" "$dest" >> "$TARGETS_FILE"
done

{
  echo "run_ts=$RUN_TS"
  echo "mode=$mode"
  echo "direction=download"
  echo "exp_root=$EXP_ROOT"
  echo "data_root=$DATA_ROOT"
  echo "s3_root=$S3_ROOT"
  echo "parallel_jobs=$PARALLEL_JOBS"
  echo "log_dir=$RUN_DIR"
  echo "aws_sync_args=s3 sync --no-progress"
} | tee "$RUN_DIR/_summary.txt"

while IFS='|' read -r label src dest; do
  mkdir -p "$dest"
  echo "queued label=$label src=$src dest=$dest" | tee -a "$RUN_DIR/_summary.txt"
done < "$TARGETS_FILE"

export RUN_DIR mode

worker_script='
set -euo pipefail
target="$1"
IFS="|" read -r label src dest <<<"$target"
log_file="$RUN_DIR/${label}.log"

{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] label=$label"
  echo "mode=$mode"
  echo "src=$src"
  echo "dest=$dest"

  if [[ "$label" == "mif_unringed" ]]; then
    broken_symlinks="$(find "$dest" -maxdepth 1 -xtype l | wc -l | tr -d " " || true)"
    broken_symlinks="${broken_symlinks:-0}"
    echo "broken_symlinks_at_dest=$broken_symlinks"
    if (( broken_symlinks > 0 )); then
      if [[ "$mode" == "dry-run" ]]; then
        echo "dryrun_note=would_remove_broken_symlinks_before_sync"
        echo "status=completed"
        exit 0
      fi
      find "$dest" -maxdepth 1 -xtype l -delete
      echo "broken_symlinks_removed=$broken_symlinks"
    fi
  fi

  if [[ "$mode" == "dry-run" ]]; then
    aws s3 sync --no-progress --dryrun "$src" "$dest/"
  else
    aws s3 sync --no-progress "$src" "$dest/"
  fi

  echo "status=completed"
} >"$log_file" 2>&1
'

exit_code=0
if ! xargs -I{} -P "$PARALLEL_JOBS" bash -lc "$worker_script" _ "{}" < "$TARGETS_FILE"; then
  exit_code=1
fi

while IFS='|' read -r label _; do
  if [[ -f "$RUN_DIR/${label}.log" ]]; then
    final_status="$(grep -E '^status=' "$RUN_DIR/${label}.log" | tail -n 1 || true)"
    final_status="${final_status#status=}"
    if [[ -z "$final_status" ]]; then
      final_status="failed_or_interrupted"
      exit_code=1
    fi
    echo "finished label=$label status=$final_status" | tee -a "$RUN_DIR/_summary.txt"
  else
    echo "finished label=$label status=missing_log" | tee -a "$RUN_DIR/_summary.txt"
    exit_code=1
  fi
done < "$TARGETS_FILE"

if (( exit_code == 0 )); then
  echo "all sync jobs completed successfully" | tee -a "$RUN_DIR/_summary.txt"
else
  echo "one or more sync jobs failed; inspect logs in $RUN_DIR" | tee -a "$RUN_DIR/_summary.txt"
fi

exit "$exit_code"
