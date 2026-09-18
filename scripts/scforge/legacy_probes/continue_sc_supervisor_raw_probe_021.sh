#!/usr/bin/env bash
set -u -o pipefail

RUN="/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc/supervisor_raw_end_to_end_probe/raw_supervisor_021_deriv_t1_20260520T211515Z"
WORK="${RUN}/work"
LOGS="${RUN}/logs"
SID="021_S_7092_I1597668"
SELECT_STREAMLINES="1000000"
NTHREADS="8"
CUTOFF="0.06"

export FSLDIR="/home/ec2-user/fsl"
export FSLOUTPUTTYPE="NIFTI_GZ"
export PATH="/home/ec2-user/mrtrix3/bin:/home/ec2-user/fsl/share/fsl/bin:/home/ec2-user/fsl/bin:${PATH}"

status() {
  /home/ec2-user/fsl/bin/python - "$RUN/status.json" "$SID" "$1" <<'PY'
import json, sys, time
from pathlib import Path
path = Path(sys.argv[1])
sid = sys.argv[2]
phase = sys.argv[3]
payload = {}
if path.exists():
    try:
        payload = json.loads(path.read_text())
    except Exception:
        payload = {}
payload.update({
    "sid": sid,
    "phase": phase,
    "last_update_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "run_dir": str(path.parent),
    "continued_from_phase8": True,
})
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
tmp.replace(path)
PY
}

fail_status() {
  /home/ec2-user/fsl/bin/python - "$RUN/status.json" "$SID" "$1" "$2" <<'PY'
import json, sys, time
from pathlib import Path
path = Path(sys.argv[1])
sid = sys.argv[2]
phase = sys.argv[3]
error = sys.argv[4]
payload = {}
if path.exists():
    try:
        payload = json.loads(path.read_text())
    except Exception:
        payload = {}
payload.update({
    "sid": sid,
    "phase": "failed",
    "failed_phase": phase,
    "error": error,
    "last_update_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "run_dir": str(path.parent),
    "continued_from_phase8": True,
})
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
tmp.replace(path)
PY
}

run_cmd() {
  local phase="$1"
  local logfile="$2"
  shift 2
  status "$phase"
  {
    echo
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
  } >> "$logfile"
  "$@" >> "$logfile" 2>&1
  local rc=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] rc=${rc}" >> "$logfile"
  if [ "$rc" -ne 0 ]; then
    fail_status "$phase" "command failed rc=${rc}; see ${logfile}"
    exit "$rc"
  fi
}

mkdir -p "$WORK" "$LOGS"

run_cmd "8_5tt_and_gmwmi" "$LOGS/08_5tt.log" \
  /home/ec2-user/mrtrix3/bin/5ttgen fsl "$WORK/t12b0.nii.gz" "$WORK/5tt1.mif" -debug -force
run_cmd "8_5tt_and_gmwmi" "$LOGS/08_5tt.log" \
  /home/ec2-user/mrtrix3/bin/5tt2gmwmi "$WORK/5tt1.mif" "$WORK/gmwmSeed.mif" -force

run_cmd "9_tckgen" "$LOGS/09_tckgen.log" \
  /home/ec2-user/mrtrix3/bin/tckgen -act "$WORK/5tt1.mif" -backtrack -seed_gmwmi "$WORK/gmwmSeed.mif" \
  -maxlength 250 -cutoff "$CUTOFF" -select "$SELECT_STREAMLINES" -nthreads "$NTHREADS" \
  "$WORK/wmfod_norm.mif" "$WORK/tracks_${SELECT_STREAMLINES}.tck" -force

run_cmd "10_tcksift2" "$LOGS/10_tcksift2.log" \
  /home/ec2-user/mrtrix3/bin/tcksift2 -act "$WORK/5tt1.mif" -out_coeffs "$WORK/sift_coeffs.txt" \
  -nthreads "$NTHREADS" "$WORK/tracks_${SELECT_STREAMLINES}.tck" "$WORK/wmfod_norm.mif" \
  "$WORK/sift_${SELECT_STREAMLINES}.txt" -force

run_cmd "11_aal116_to_b0" "$LOGS/11_aal_route.log" \
  /home/ec2-user/fsl/bin/flirt -in /home/ec2-user/exp/atlas/AAL116_spm12/aal/atlas/AAL.nii \
  -ref /home/ec2-user/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz -dof 6 \
  -omat "$WORK/AAL2MNI.mat" -interp nearestneighbour -datatype int -out "$WORK/AAL2MNI.nii.gz"
run_cmd "11_aal116_to_b0" "$LOGS/11_aal_route.log" \
  /home/ec2-user/fsl/bin/flirt -in "$WORK/AAL2MNI.nii.gz" -ref "$WORK/brainmask.nii.gz" \
  -omat "$WORK/aalmni2t1.mat" -interp nearestneighbour -datatype int -out "$WORK/AALmni2t1.nii.gz" -dof 12
run_cmd "11_aal116_to_b0" "$LOGS/11_aal_route.log" \
  /home/ec2-user/fsl/bin/flirt -in "$WORK/AALmni2t1.nii.gz" -ref "$WORK/vol0000.nii.gz" \
  -applyxfm -init "$WORK/t12b0.mat" -interp nearestneighbour -out "$WORK/AAL_sub.nii.gz"

status "11_aal116_to_b0_remap"
/home/ec2-user/fsl/bin/python - "$RUN" <<'PY'
import importlib.util, sys
from pathlib import Path
run = Path(sys.argv[1])
mod_path = Path("/home/ec2-user/exp/run_sc_supervisor_raw_end_to_end_probe.py")
spec = importlib.util.spec_from_file_location("raw_probe", mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
report = mod.aal116_sparse_to_contiguous(run / "work" / "AAL_sub.nii.gz", run / "work" / "AAL_sub_contig116.nii.gz")
mod.write_csv(run / "aal116_label_remap.csv", [report])
PY
if [ "$?" -ne 0 ]; then
  fail_status "11_aal116_to_b0_remap" "AAL116 contiguous remap failed"
  exit 1
fi

run_cmd "12_connectome" "$LOGS/12_connectome.log" \
  /home/ec2-user/mrtrix3/bin/tck2connectome -symmetric -zero_diagonal -scale_invnodevol \
  -tck_weights_in "$WORK/sift_${SELECT_STREAMLINES}.txt" "$WORK/tracks_${SELECT_STREAMLINES}.tck" \
  "$WORK/AAL_sub_contig116.nii.gz" "$RUN/SC_AAL.csv" -out_assignment "$RUN/assignmentsaal.csv" -force
run_cmd "12_connectome" "$LOGS/12_connectome.log" \
  /home/ec2-user/mrtrix3/bin/tck2connectome -symmetric -zero_diagonal \
  -tck_weights_in "$WORK/sift_${SELECT_STREAMLINES}.txt" "$WORK/tracks_${SELECT_STREAMLINES}.tck" \
  "$WORK/AAL_sub_contig116.nii.gz" "$RUN/TL_AAL.csv" -scale_length -stat_edge mean -force

status "13_matrix_stats"
/home/ec2-user/fsl/bin/python - "$RUN" "$SID" <<'PY'
import importlib.util, json, sys, time
from pathlib import Path
run = Path(sys.argv[1])
sid = sys.argv[2]
mod_path = Path("/home/ec2-user/exp/run_sc_supervisor_raw_end_to_end_probe.py")
spec = importlib.util.spec_from_file_location("raw_probe", mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
stats = [
    mod.matrix_stats(run / "SC_AAL.csv", "probe_SC_AAL"),
    mod.matrix_stats(run / "TL_AAL.csv", "probe_TL_AAL"),
]
subject_id = mod.subject_id_from_sid(sid)
benchmark = mod.SUPERVISOR_BENCHMARKS.get(subject_id)
if benchmark and benchmark.exists():
    stats.append(mod.matrix_stats(benchmark, "supervisor_benchmark_SC_AAL"))
mod.write_csv(run / "matrix_stats.csv", stats)
mod.write_findings(run, [
    "# Supervisor Raw End-to-End Probe",
    "",
    f"SID: `{sid}`",
    f"Run root: `{run}`",
    "Mode: continued near-reference replay from phase 8 after EC2 5ttgen/FIRST stdout workaround.",
    "",
    "## Matrix stats",
    "",
    *(f"- {row['label']}: n={row.get('n')}, density={row.get('density')}, zero_rows={row.get('zero_rows')}, error={row.get('error', '')}" for row in stats),
    "",
    "## Interpretation rule",
    "",
    "If this raw replay gives supervisor-like density and few/no whole zero rows, then the fix belongs upstream in preprocessing/T1-to-B0/AAL routing, not in post-only SC matrix repair.",
])
status_path = run / "status.json"
payload = json.loads(status_path.read_text()) if status_path.exists() else {}
payload.update({
    "sid": sid,
    "phase": "done",
    "last_update_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "run_dir": str(run),
    "matrix_stats_csv": str(run / "matrix_stats.csv"),
    "findings_md": str(run / "findings.md"),
    "continued_from_phase8": True,
})
tmp = status_path.with_suffix(status_path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
tmp.replace(status_path)
PY
if [ "$?" -ne 0 ]; then
  fail_status "13_matrix_stats" "matrix stats/finding generation failed"
  exit 1
fi
