#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-$HOME/exp/logs/s3-sync}"
WATCH_SECONDS=0
S3_REFRESH_SECONDS="${S3_REFRESH_SECONDS:-30}"
RUN_SPEC="latest"
DASHBOARD_SESSION_NAME="${DASHBOARD_SESSION_NAME:-s3_sync_dashboard}"
RUNNER_SESSION_NAME="${RUNNER_SESSION_NAME:-}"
INVENTORY_HELPER="${INVENTORY_HELPER:-$HOME/exp/s3_inventory_stats.py}"

usage() {
  cat <<'EOF'
Usage:
  s3_sync_dashboard.sh [run-dir|latest] [--watch SECONDS]
  s3_sync_dashboard.sh --watch SECONDS [run-dir|latest]

Examples:
  s3_sync_dashboard.sh
  s3_sync_dashboard.sh latest --watch 5
  s3_sync_dashboard.sh /home/ec2-user/exp/logs/s3-sync/20260420T143309Z
EOF
}

parse_args() {
  local positional=()
  while (($#)); do
    case "$1" in
      --watch)
        WATCH_SECONDS="${2:-5}"
        shift 2
        ;;
      -h|--help|help)
        usage
        exit 0
        ;;
      *)
        positional+=("$1")
        shift
        ;;
    esac
  done

  if ((${#positional[@]} > 0)); then
    RUN_SPEC="${positional[0]}"
  fi
}

resolve_run_dir() {
  if [[ "$RUN_SPEC" == "latest" ]]; then
    if [[ ! -L "$LOG_ROOT/latest" && ! -d "$LOG_ROOT/latest" ]]; then
      echo "No latest run found under $LOG_ROOT" >&2
      exit 1
    fi
    readlink -f "$LOG_ROOT/latest"
  elif [[ -d "$RUN_SPEC" ]]; then
    readlink -f "$RUN_SPEC"
  elif [[ -d "$LOG_ROOT/$RUN_SPEC" ]]; then
    readlink -f "$LOG_ROOT/$RUN_SPEC"
  else
    echo "Run directory not found: $RUN_SPEC" >&2
    exit 1
  fi
}

parse_args "$@"
RUN_DIR="$(resolve_run_dir)"
TARGETS_FILE="$RUN_DIR/_targets.txt"
SUMMARY_FILE="$RUN_DIR/_summary.txt"
REMOTE_CACHE="$RUN_DIR/_remote_totals.tsv"

if [[ ! -f "$TARGETS_FILE" ]]; then
  echo "Targets file missing: $TARGETS_FILE" >&2
  exit 1
fi

run_field() {
  local field="$1"
  awk -F'=' -v key="$field" '$1 == key { print substr($0, index($0, "=") + 1); exit }' "$SUMMARY_FILE"
}

RUN_DIRECTION="$(run_field direction)"
S3_ROOT="$(run_field s3_root)"
RUN_DIRECTION="${RUN_DIRECTION:-}"

is_s3_path() {
  [[ "$1" == s3://* ]]
}

inventory_stats() {
  local mode="$1"
  local target="$2"
  python3 "$INVENTORY_HELPER" "$mode" "$target" 2>/dev/null || printf '0|0|0|--|--'
}

alternate_s3_prefix_for_label() {
  local label="$1"
  local active_prefix="$2"
  local canonical legacy

  case "$label" in
    biascorr_1|dwi_t1_bbr|eddy|mif_denoised|mif_dwi|mif_unringed|qc)
      canonical="${S3_ROOT}/data/derivatives/${label}/"
      legacy="${S3_ROOT}/${label}/"
      ;;
    *)
      printf ''
      return
      ;;
  esac

  if [[ "$active_prefix" == "$canonical" ]]; then
    printf '%s' "$legacy"
  elif [[ "$active_prefix" == "$legacy" ]]; then
    printf '%s' "$canonical"
  else
    printf '%s' "$legacy"
  fi
}

should_refresh_remote_cache() {
  [[ ! -f "$REMOTE_CACHE" ]] && return 0
  local now last
  now="$(date +%s)"
  last="$(stat -c %Y "$REMOTE_CACHE")"
  (( now - last >= S3_REFRESH_SECONDS ))
}

refresh_remote_cache() {
  local tmp="${REMOTE_CACHE}.tmp"
  : > "$tmp"

  while IFS='|' read -r label src dest; do
    local active_prefix alt_prefix active_stats alt_stats
    local active_count active_size active_subjects active_fps active_dist
    local alt_count alt_size alt_subjects alt_fps alt_dist

    if is_s3_path "$src"; then
      active_prefix="$src"
    elif is_s3_path "$dest"; then
      active_prefix="$dest"
    else
      active_prefix=""
    fi

    active_count=0
    active_size=0
    active_subjects=0
    active_fps="--"
    active_dist="--"
    alt_count=0
    alt_size=0
    alt_subjects=0
    alt_fps="--"
    alt_dist="--"
    alt_prefix=""

    if [[ -n "$active_prefix" ]]; then
      active_stats="$(inventory_stats s3 "$active_prefix")"
      IFS='|' read -r active_count active_size active_subjects active_fps active_dist <<<"$active_stats"
      alt_prefix="$(alternate_s3_prefix_for_label "$label" "$active_prefix")"
      if [[ -n "$alt_prefix" && "$alt_prefix" != "$active_prefix" ]]; then
        alt_stats="$(inventory_stats s3 "$alt_prefix")"
        IFS='|' read -r alt_count alt_size alt_subjects alt_fps alt_dist <<<"$alt_stats"
      fi
    fi

    printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
      "$label" "$active_prefix" "$active_count" "$active_size" "$active_subjects" "$active_fps" "$active_dist" \
      "$alt_prefix" "$alt_count" "$alt_size" "$alt_subjects" "$alt_fps" "$alt_dist" >> "$tmp"
  done < "$TARGETS_FILE"

  mv "$tmp" "$REMOTE_CACHE"
}

remote_cache_field() {
  local label="$1"
  local col="$2"
  awk -F'|' -v key="$label" -v idx="$col" '$1 == key { print $idx; exit }' "$REMOTE_CACHE"
}

label_status() {
  local label="$1"
  local src="$2"
  local dest="$3"
  local log_file="$RUN_DIR/${label}.log"
  local active_path

  if [[ ! -f "$log_file" ]]; then
    printf 'queued'
    return
  fi

  if grep -q '^status=completed$' "$log_file"; then
    printf 'completed'
    return
  fi
  if grep -q '^status=empty_directory$' "$log_file"; then
    printf 'empty'
    return
  fi
  if grep -q '^status=missing_directory$' "$log_file"; then
    printf 'missing'
    return
  fi

  if is_s3_path "$src"; then
    active_path="$dest/"
  else
    active_path="$src/"
  fi

  if ps -ef | grep -F '/usr/bin/aws s3 sync --no-progress' | grep -F -- "$active_path" | grep -v grep >/dev/null; then
    printf 'running'
    return
  fi

  printf 'stalled'
}

label_last_update() {
  local label="$1"
  local log_file="$RUN_DIR/${label}.log"
  [[ -f "$log_file" ]] || { printf 'waiting for log'; return; }

  local line
  line="$(grep -E '^(upload:|download:|copy:|status=)' "$log_file" | tail -n 1 || true)"
  [[ -z "$line" ]] && line="$(tail -n 1 "$log_file" 2>/dev/null || true)"
  [[ -z "$line" ]] && line="no activity yet"

  line="${line#upload: }"
  line="${line#download: }"
  line="${line#copy: }"
  line="${line#status=}"
  printf '%s' "$line"
}

label_progress() {
  local status="$1"
  local completed_count="$2"
  local goal="$3"

  case "$status" in
    completed)
      printf '100%%'
      ;;
    empty|missing)
      printf '%s' '--'
      ;;
    *)
      if [[ "$goal" =~ ^[0-9]+$ ]] && (( goal > 0 )); then
        local pct=$(( completed_count * 100 / goal ))
        (( pct > 99 )) && pct=99
        printf '%s%%' "$pct"
      else
        printf '%s' '--'
      fi
      ;;
  esac
}

trim_note() {
  local text="$1"
  if ((${#text} > 74)); then
    printf '%s...' "${text:0:71}"
  else
    printf '%s' "$text"
  fi
}

distribution_note() {
  local subjects="$1"
  local distribution="$2"
  local files_per_subject="$3"

  [[ "$subjects" == "0" || "$subjects" == "" || "$distribution" == "--" ]] && return
  if [[ "$distribution" == "${files_per_subject}x${subjects}" ]]; then
    return
  fi
  printf 'src_dist=%s' "$distribution"
}

render_once() {
  local snapshot_file="$RUN_DIR/_dashboard_snapshot.txt"
  [[ -t 1 ]] && clear
  should_refresh_remote_cache && refresh_remote_cache

  local run_ts mode direction parallel_jobs active_workers running_labels
  local total_goal=0
  local total_completed=0
  local overall_pct="--"
  run_ts="$(run_field run_ts)"
  mode="$(run_field mode)"
  direction="${RUN_DIRECTION:-}"
  if [[ -z "$direction" ]]; then
    if awk -F'|' 'NR==1 { exit($2 ~ /^s3:\/\// ? 0 : 1) }' "$TARGETS_FILE"; then
      direction="download"
    else
      direction="upload"
    fi
  fi
  if [[ -z "$RUNNER_SESSION_NAME" ]]; then
    if [[ "$direction" == "download" ]]; then
      RUNNER_SESSION_NAME="s3_pull_exp"
    else
      RUNNER_SESSION_NAME="s3_sync_exp"
    fi
  fi
  parallel_jobs="$(run_field parallel_jobs)"
  running_labels=0

  {
    printf 'S3 Sync Dashboard\n'
    printf 'Run: %s\n' "$run_ts"
    printf 'Direction: %s | Mode: %s | S3 root: %s\n' "$direction" "$mode" "$S3_ROOT"
    printf 'Max workers: %s | S3 refresh: %ss\n' "$parallel_jobs" "$S3_REFRESH_SECONDS"
    printf 'Logs: %s\n' "$RUN_DIR"
    printf '\n'
    printf '%-14s %-10s %7s %7s %7s %7s %7s %6s  %s\n' 'Label' 'Status' 'Local' 'S3' 'SrcSub' 'DstSub' 'F/Subj' 'Pct' 'Notes'
    printf '%-14s %-10s %7s %7s %7s %7s %7s %6s  %s\n' '-----' '------' '-----' '--' '------' '------' '------' '---' '-----'

    while IFS='|' read -r label src dest; do
      local local_path local_stats local_count local_bytes local_subjects local_fps local_dist
      local remote_count remote_subjects remote_fps remote_dist remote_prefix
      local alt_prefix alt_count alt_subjects alt_fps alt_dist
      local status pct last_update note goal completed_count
      local src_subjects dst_subjects src_fps dist_note

      if is_s3_path "$src"; then
        local_path="$dest"
        remote_prefix="$src"
      else
        local_path="$src"
        remote_prefix="$dest"
      fi

      local_stats="$(inventory_stats local "$local_path")"
      IFS='|' read -r local_count local_bytes local_subjects local_fps local_dist <<<"$local_stats"
      remote_count="$(remote_cache_field "$label" 3)"
      remote_subjects="$(remote_cache_field "$label" 5)"
      remote_fps="$(remote_cache_field "$label" 6)"
      remote_dist="$(remote_cache_field "$label" 7)"
      alt_prefix="$(remote_cache_field "$label" 8)"
      alt_count="$(remote_cache_field "$label" 9)"
      alt_subjects="$(remote_cache_field "$label" 11)"
      alt_fps="$(remote_cache_field "$label" 12)"
      alt_dist="$(remote_cache_field "$label" 13)"
      remote_count="${remote_count:-0}"
      remote_subjects="${remote_subjects:-0}"
      remote_fps="${remote_fps:---}"
      remote_dist="${remote_dist:---}"
      alt_count="${alt_count:-0}"
      alt_subjects="${alt_subjects:-0}"
      alt_fps="${alt_fps:---}"
      alt_dist="${alt_dist:---}"

      if [[ "$direction" == "upload" ]]; then
        goal="$local_count"
        completed_count="$remote_count"
        src_subjects="$local_subjects"
        dst_subjects="$remote_subjects"
        src_fps="$local_fps"
        dist_note="$(distribution_note "$local_subjects" "$local_dist" "$local_fps")"
      else
        goal="$remote_count"
        completed_count="$local_count"
        src_subjects="$remote_subjects"
        dst_subjects="$local_subjects"
        src_fps="$remote_fps"
        dist_note="$(distribution_note "$remote_subjects" "$remote_dist" "$remote_fps")"
      fi

      status="$(label_status "$label" "$src" "$dest")"
      [[ "$status" == "running" ]] && ((running_labels+=1))
      pct="$(label_progress "$status" "$completed_count" "$goal")"
      if [[ "$goal" =~ ^[0-9]+$ ]]; then
        total_goal=$((total_goal + goal))
        if [[ "$completed_count" =~ ^[0-9]+$ ]]; then
          if (( completed_count > goal )); then
            total_completed=$((total_completed + goal))
          else
            total_completed=$((total_completed + completed_count))
          fi
        fi
      fi
      last_update="$(label_last_update "$label")"
      note="$last_update"

      if [[ -n "$alt_prefix" && "$alt_count" != "0" ]]; then
        if [[ "$remote_prefix" == *"/data/derivatives/"* ]]; then
          note="legacy=${alt_count}f/${alt_subjects}s/${alt_fps}fps | $note"
        elif [[ "$remote_prefix" == "${S3_ROOT}/"* && "$remote_prefix" != *"/data/"* ]]; then
          note="canonical=${alt_count}f/${alt_subjects}s/${alt_fps}fps | $note"
        fi
      fi
      if [[ -n "$dist_note" ]]; then
        note="${dist_note} | $note"
      fi

      printf '%-14s %-10s %7s %7s %7s %7s %7s %6s  %s\n' \
        "$label" "$status" "$local_count" "$remote_count" "$src_subjects" "$dst_subjects" "$src_fps" "$pct" "$(trim_note "$note")"
    done < "$TARGETS_FILE"

    if (( total_goal > 0 )); then
      overall_pct="$(( total_completed * 100 / total_goal ))"
    fi

    printf '\n'
    printf 'Overall progress: %s%% (%s/%s files)\n' "$overall_pct" "$total_completed" "$total_goal"
    printf 'Active labels in this run: %s\n' "$running_labels"
    printf 'Attach upload/pull runner:\n'
    printf '  tmux attach -t %s\n' "$RUNNER_SESSION_NAME"
    printf 'Attach this dashboard:\n'
    printf '  tmux attach -t %s\n' "$DASHBOARD_SESSION_NAME"
  } > "$snapshot_file"

  cat "$snapshot_file"
}

refresh_remote_cache

if (( WATCH_SECONDS > 0 )); then
  while true; do
    render_once
    sleep "$WATCH_SECONDS"
  done
else
  render_once
fi
