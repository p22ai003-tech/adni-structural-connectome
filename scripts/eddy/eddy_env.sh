#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file instead of executing it:"
  echo "source ~/exp/scripts/eddy/eddy_env.sh"
  exit 1
fi

export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/exp}"
export FSLDIR="${FSLDIR:-$HOME/fsl}"
export EDDY_DERIV_ROOT="${EDDY_DERIV_ROOT:-$PROJECT_ROOT/data/derivatives}"
export EDDY_STAGE_ROOT="${EDDY_STAGE_ROOT:-/scratch/eddy_stage}"
export EDDY_COHORT_DTI_CSV="${EDDY_COHORT_DTI_CSV:-$PROJECT_ROOT/cohort/dti.csv}"
export EDDY_PYTHON="${EDDY_PYTHON:-$FSLDIR/bin/python}"
export EDDY_DEFAULT_JOBS="${EDDY_DEFAULT_JOBS:-1}"
export EDDY_DEFAULT_THREADS="${EDDY_DEFAULT_THREADS:-4}"
export EDDY_PE_DIR="${EDDY_PE_DIR:-j-}"
export EDDY_OPTIONS="${EDDY_OPTIONS:---slm=linear --data_is_shelled}"
export EDDY_ALLOW_INPUT_FALLBACK="${EDDY_ALLOW_INPUT_FALLBACK:-0}"

mkdir -p "$HOME/bin"
if [[ -x "$FSLDIR/bin/eddy_cuda11.0" ]]; then
  ln -sf "$FSLDIR/bin/eddy_cuda11.0" "$HOME/bin/eddy_cuda"
fi

for candidate in \
  "$HOME/bin" \
  "$HOME/mrtrix3/bin" \
  "$FSLDIR/bin" \
  "$FSLDIR/share/fsl/bin"
do
  case ":$PATH:" in
    *":$candidate:"*) ;;
    *) export PATH="$candidate:$PATH" ;;
  esac
done

case ":${PYTHONPATH:-}:" in
  *":$PROJECT_ROOT:"*) ;;
  *) export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

hash -r

eddy_python() {
  "$EDDY_PYTHON" "$@"
}

eddy_cli() {
  eddy_python -m connectome_pipeline.dwi_eddy "$@"
}

eddy_status() {
  local group="${1:-all}"
  shift || true

  eddy_cli \
    --deriv-root "$EDDY_DERIV_ROOT" \
    --cohort-dti-csv "$EDDY_COHORT_DTI_CSV" \
    --group "$group" \
    --status-only \
    "$@"
}

eddy_run() {
  local jobs="${1:-$EDDY_DEFAULT_JOBS}"
  shift || true
  local threads="${1:-$EDDY_DEFAULT_THREADS}"
  shift || true
  local group="${1:-all}"
  shift || true
  local extra_args=()
  if [[ "$EDDY_ALLOW_INPUT_FALLBACK" == "1" ]]; then
    extra_args+=(--allow-input-fallback)
  fi

  eddy_cli \
    --deriv-root "$EDDY_DERIV_ROOT" \
    --stage-root "$EDDY_STAGE_ROOT" \
    --cohort-dti-csv "$EDDY_COHORT_DTI_CSV" \
    --group "$group" \
    --jobs "$jobs" \
    --threads-per-job "$threads" \
    --pe-dir "$EDDY_PE_DIR" \
    --eddy-options "$EDDY_OPTIONS" \
    --no-qc \
    "${extra_args[@]}" \
    "$@"
}

eddy_ad_status() {
  eddy_status ad "$@"
}

eddy_ad_run() {
  local jobs="${1:-$EDDY_DEFAULT_JOBS}"
  shift || true
  local threads="${1:-$EDDY_DEFAULT_THREADS}"
  shift || true

  eddy_run "$jobs" "$threads" ad "$@"
}

echo "Loaded Eddy shell environment"
echo "PROJECT_ROOT         = $PROJECT_ROOT"
echo "EDDY_DERIV_ROOT      = $EDDY_DERIV_ROOT"
echo "EDDY_STAGE_ROOT      = $EDDY_STAGE_ROOT"
echo "EDDY_COHORT_DTI_CSV  = $EDDY_COHORT_DTI_CSV"
echo "EDDY_PYTHON          = $EDDY_PYTHON"
echo "EDDY_PE_DIR          = $EDDY_PE_DIR"
echo "EDDY_OPTIONS         = $EDDY_OPTIONS"
echo "EDDY_ALLOW_INPUT_FALLBACK = $EDDY_ALLOW_INPUT_FALLBACK"
echo "dwifslpreproc        = $(command -v dwifslpreproc || echo 'not found')"
echo "eddy_cuda            = $(command -v eddy_cuda || echo 'not found')"
if command -v eddy_cuda >/dev/null 2>&1; then
  echo "eddy_cuda resolves   = $(readlink -f "$(command -v eddy_cuda)")"
fi
