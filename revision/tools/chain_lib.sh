# Helpers shared by the launchers. Sourced, never run.
CHAIN_T=/home/ec2-user/exp/docs/iit_dipanjan/tools
CHAIN_APY=/home/ec2-user/exp/.venv_connectome_app/bin/python

# chain_env <launcher>: under IIT_ROOT, export the chain's locations and settings; then the recipe check, which exits 3 when
# refused. IIT_PRINT_PATHS=1 stops before the check, for the test that the unset root still means today's paths.
chain_env() {
  if [ -n "${IIT_ROOT:-}" ]; then
    local lines; lines=$("$CHAIN_APY" -I "$CHAIN_T/iit_chain.py" shell-env) || exit 3
    eval "$lines"
  fi
  [ "${IIT_PRINT_PATHS:-0}" = 1 ] && return 0
  "$CHAIN_APY" -I "$CHAIN_T/iit_chain.py" guard "$1" || exit 3
}

# chain_stage <name> <command...>. Without a ledger it just runs the command, which is today's behaviour. With IIT_LEDGER it
# keeps the output in $IIT_LOGS/<name>.log, adds name, exit code, seconds and last line to the ledger, skips a stage already
# done (so a restart resumes), and carries on after a failure so that every stage's result gets listed.
chain_stage() {
  local name=$1; shift
  if [ -z "${IIT_LEDGER:-}" ]; then "$@"; return $?; fi
  local marker=$IIT_LOGS/done/$name log=$IIT_LOGS/$name.log t0=$SECONDS e=0
  mkdir -p "$IIT_LOGS/done"
  if [ -f "$marker" ]; then echo "[skip] $name"; return 0; fi
  echo "[run ] $name  $(date -u +%T)"
  "$@" > "$log" 2>&1 || e=$?
  local last; last=$(grep -v '^[[:space:]]*$' "$log" | tail -1 | cut -c1-200 | sed -E 's/[0-9]{3}_S_[0-9]{4}(_I[0-9]+)?/SID/g' | tr '\t' ' ')
  printf '%s\t%s\t%s\t%s\n' "$name" "$e" "$((SECONDS - t0))" "$last" >> "$IIT_LEDGER"
  [ "$e" = 0 ] && : > "$marker"
  echo "[$([ "$e" = 0 ] && echo ok || echo FAIL)] $name exit=$e"
  return 0
}

# chain_each <prefix> "<items>" "<command with {}>": one stage per item, at most IIT_JOBS at a time.
chain_each() {
  local prefix=$1 items=$2 template=$3
  export -f chain_stage
  printf '%s\n' $items | xargs -P "${IIT_JOBS:-8}" -I{} bash -c "chain_stage ${prefix}_{} $template"
}
