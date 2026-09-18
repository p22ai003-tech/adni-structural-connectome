#!/usr/bin/env bash
set -euo pipefail

UPLOAD_LOG_ROOT="${UPLOAD_LOG_ROOT:-$HOME/exp/logs/s3-sync}"
PULL_LOG_ROOT="${PULL_LOG_ROOT:-$HOME/exp/logs/s3-pull}"
WATCH_SECONDS="${WATCH_SECONDS:-5}"

usage() {
  cat <<'EOF'
Usage:
  s3_overview_dashboard.sh [--watch SECONDS]

Shows a compact progress-bar overview for:
  - upload dashboard
  - pull dashboard
  - combined average
EOF
}

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
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

read_snapshot_field() {
  local snapshot="$1"
  local prefix="$2"
  [[ -f "$snapshot" ]] || return 1
  grep -F "$prefix" "$snapshot" | tail -n 1 | sed "s/^${prefix}//"
}

extract_percent() {
  local snapshot="$1"
  local line
  line="$(read_snapshot_field "$snapshot" 'Overall progress: ' || true)"
  [[ -z "$line" ]] && { printf '%s' '--'; return; }
  line="${line%%%*}"
  printf '%s' "$line"
}

extract_meta() {
  local snapshot="$1"
  local run direction_line direction mode active
  run="$(read_snapshot_field "$snapshot" 'Run: ' || true)"
  direction_line="$(read_snapshot_field "$snapshot" 'Direction: ' || true)"
  active="$(read_snapshot_field "$snapshot" 'Active labels in this run: ' || true)"

  direction="${direction_line%% | Mode:*}"
  mode="${direction_line#*| Mode: }"
  mode="${mode%% | S3 root:*}"

  printf '%s|%s|%s|%s\n' "${run:---}" "${direction:---}" "${mode:---}" "${active:---}"
}

snapshot_age_seconds() {
  local snapshot="$1"
  [[ -f "$snapshot" ]] || { printf '%s' '--'; return; }
  local now last
  now="$(date +%s)"
  last="$(stat -c %Y "$snapshot")"
  printf '%s' "$(( now - last ))"
}

progress_bar() {
  local pct="$1"
  local width=24
  if [[ ! "$pct" =~ ^[0-9]+$ ]]; then
    printf '[%*s]' "$width" '' | tr ' ' '.'
    return
  fi

  local filled=$(( pct * width / 100 ))
  local empty=$(( width - filled ))
  printf '['
  printf '%*s' "$filled" '' | tr ' ' '#'
  printf '%*s' "$empty" '' | tr ' ' '-'
  printf ']'
}

render_once() {
  local upload_snapshot="$UPLOAD_LOG_ROOT/latest/_dashboard_snapshot.txt"
  local pull_snapshot="$PULL_LOG_ROOT/latest/_dashboard_snapshot.txt"
  local upload_pct pull_pct combined_pct="--"
  local upload_meta pull_meta
  local upload_age pull_age
  local upload_run upload_direction upload_mode upload_active
  local pull_run pull_direction pull_mode pull_active

  [[ -t 1 ]] && clear

  upload_pct="$(extract_percent "$upload_snapshot")"
  pull_pct="$(extract_percent "$pull_snapshot")"
  upload_age="$(snapshot_age_seconds "$upload_snapshot")"
  pull_age="$(snapshot_age_seconds "$pull_snapshot")"
  upload_meta="$(extract_meta "$upload_snapshot" || printf '%s' '--|--|--|--')"
  pull_meta="$(extract_meta "$pull_snapshot" || printf '%s' '--|--|--|--')"
  IFS='|' read -r upload_run upload_direction upload_mode upload_active <<<"$upload_meta"
  IFS='|' read -r pull_run pull_direction pull_mode pull_active <<<"$pull_meta"

  if [[ "$upload_pct" =~ ^[0-9]+$ && "$pull_pct" =~ ^[0-9]+$ ]]; then
    combined_pct="$(( (upload_pct + pull_pct) / 2 ))"
  elif [[ "$upload_pct" =~ ^[0-9]+$ ]]; then
    combined_pct="$upload_pct"
  elif [[ "$pull_pct" =~ ^[0-9]+$ ]]; then
    combined_pct="$pull_pct"
  fi

  printf 'S3 Overview Dashboard\n'
  printf 'Refresh: every %ss\n' "$WATCH_SECONDS"
  printf '\n'
  printf '%-10s %-28s %6s  %s\n' 'Flow' 'Progress' 'Pct' 'Details'
  printf '%-10s %-28s %6s  %s\n' '----' '--------' '---' '-------'
  printf '%-10s %-28s %6s  %s\n' \
    'Upload' "$(progress_bar "$upload_pct")" "${upload_pct}% " \
    "run=${upload_run} mode=${upload_mode} active=${upload_active} age=${upload_age}s"
  printf '%-10s %-28s %6s  %s\n' \
    'Pull' "$(progress_bar "$pull_pct")" "${pull_pct}% " \
    "run=${pull_run} mode=${pull_mode} active=${pull_active} age=${pull_age}s"
  printf '%-10s %-28s %6s  %s\n' \
    'Combined' "$(progress_bar "$combined_pct")" "${combined_pct}% " \
    'average of upload and pull overall progress'
  printf '\n'
  printf 'Attach dashboards:\n'
  printf '  tmux attach -t s3_sync_dashboard\n'
  printf '  tmux attach -t s3_pull_dashboard\n'
  printf '  tmux attach -t s3_overview_dashboard\n'
}

while true; do
  render_once
  sleep "$WATCH_SECONDS"
done
