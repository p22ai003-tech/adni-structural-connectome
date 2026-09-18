#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  s3_parallel_sync.sh [dry-run|run]

Description:
  Sync the EC2-native exp dataset to S3 in parallel.

Behavior:
  - Mirrors the canonical EC2 layout into s3://sabeesh/exp/
  - Uploads cohort/, atlas/, data/Images/, and canonical data/derivatives/ stage folders
  - Skips symlinks intentionally with --no-follow-symlinks
  - Runs multiple aws s3 sync jobs in parallel
  - Writes per-job logs under ~/exp/logs/s3-sync/<timestamp>/

Environment overrides:
  S3_ROOT=s3://bucket/prefix      Destination root (default: s3://sabeesh/exp)
  EXP_ROOT=/home/ec2-user/exp     Project root (default: ~/exp)
  PARALLEL_JOBS=4                 Max concurrent sync jobs (default: 4)
  LOG_ROOT=~/exp/logs/s3-sync     Log root (default: ~/exp/logs/s3-sync)
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
LOG_ROOT="${LOG_ROOT:-$EXP_ROOT/logs/s3-sync}"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$LOG_ROOT/$RUN_TS"

mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$LOG_ROOT/latest"

AWS_SYNC_ARGS=(s3 sync --no-progress --no-follow-symlinks)
if [[ "$mode" == "dry-run" ]]; then
  AWS_SYNC_ARGS+=(--dryrun)
fi

SYNC_TARGETS=(
  "cohort|$EXP_ROOT/cohort|$S3_ROOT/cohort"
  "atlas|$EXP_ROOT/atlas|$S3_ROOT/atlas"
  "images_dti|$DATA_ROOT/Images/dti|$S3_ROOT/data/Images/dti"
  "images_mri|$DATA_ROOT/Images/mri|$S3_ROOT/data/Images/mri"
  "mif_dwi|$DATA_ROOT/derivatives/mif_dwi|$S3_ROOT/data/derivatives/mif_dwi"
  "mif_denoised|$DATA_ROOT/derivatives/mif_denoised|$S3_ROOT/data/derivatives/mif_denoised"
  "mif_unringed|$DATA_ROOT/derivatives/mif_unringed|$S3_ROOT/data/derivatives/mif_unringed"
  "eddy|$DATA_ROOT/derivatives/eddy|$S3_ROOT/data/derivatives/eddy"
  "biascorr_1|$DATA_ROOT/derivatives/biascorr_1|$S3_ROOT/data/derivatives/biascorr_1"
  "dwi_t1_bbr|$DATA_ROOT/derivatives/dwi_t1_bbr|$S3_ROOT/data/derivatives/dwi_t1_bbr"
  "qc|$DATA_ROOT/derivatives/qc|$S3_ROOT/data/derivatives/qc"
)

TARGETS_FILE="$RUN_DIR/_targets.txt"
printf "%s\n" "${SYNC_TARGETS[@]}" > "$TARGETS_FILE"

{
  echo "run_ts=$RUN_TS"
  echo "mode=$mode"
  echo "direction=upload"
  echo "exp_root=$EXP_ROOT"
  echo "data_root=$DATA_ROOT"
  echo "s3_root=$S3_ROOT"
  echo "parallel_jobs=$PARALLEL_JOBS"
  echo "log_dir=$RUN_DIR"
  echo "aws_sync_args=${AWS_SYNC_ARGS[*]}"
} | tee "$RUN_DIR/_summary.txt"

while IFS='|' read -r label src dest; do
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

  if [[ ! -d "$src" ]]; then
    echo "status=missing_directory"
    exit 0
  fi

  if ! find "$src" -mindepth 1 -print -quit | grep -q .; then
    echo "status=empty_directory"
    exit 0
  fi

  du -sh "$src" || true

  if [[ "$label" == "mif_unringed" ]]; then
    echo "symlinks_total=$(find "$src" -maxdepth 1 -type l | wc -l | tr -d " ")"
    echo "symlinks_broken=$(find "$src" -maxdepth 1 -xtype l | wc -l | tr -d " ")"
    echo "sync_policy=skip_all_symlinks"
  fi

  if [[ "$mode" == "dry-run" ]]; then
    aws s3 sync --no-progress --no-follow-symlinks --dryrun "$src/" "$dest/"
  else
    aws s3 sync --no-progress --no-follow-symlinks "$src/" "$dest/"
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
