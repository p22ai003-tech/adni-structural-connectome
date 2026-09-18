#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to an existing supervisor raw replay directory}"
SID="${SID:-003_S_0908_I1249292}"
SELECT_STREAMLINES="${SELECT_STREAMLINES:-1000000}"
NTHREADS="${NTHREADS:-8}"
CUTOFF="${CUTOFF:-0.06}"

WORK="${RUN_ROOT}/work"
LOGS="${RUN_ROOT}/logs"
MRTRIX3TISSUE="/home/ec2-user/MRtrix3Tissue/bin"
MRTRIX="/home/ec2-user/mrtrix3/bin"
FSLDIR="${FSLDIR:-/home/ec2-user/fsl}"
export FSLDIR FSLOUTPUTTYPE=NIFTI_GZ LC_ALL=C LANG=C
# Use the standard MRtrix build for normal MRtrix commands. MRtrix3Tissue is
# used explicitly only for ss3t_csd_beta1; putting it first in PATH makes Python
# wrappers such as 5ttgen import the wrong mrtrix3.py under Python 3.12.
export PATH="${MRTRIX}:${FSLDIR}/bin:${FSLDIR}/share/fsl/bin:${PATH}"
export OMP_NUM_THREADS="${NTHREADS}"
export OPENBLAS_NUM_THREADS="${NTHREADS}"
export MKL_NUM_THREADS="${NTHREADS}"
export NUMEXPR_NUM_THREADS="${NTHREADS}"

status() {
  local phase="$1"
  shift || true
  /home/ec2-user/fsl/bin/python - "$RUN_ROOT" "$phase" "$@" <<'PY'
import json, sys, time
from pathlib import Path
run_root = Path(sys.argv[1])
phase = sys.argv[2]
status_path = run_root / "status.json"
try:
    data = json.loads(status_path.read_text())
except Exception:
    data = {}
data["phase"] = phase
data["last_update_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
if phase != "failed":
    data.pop("error", None)
for item in sys.argv[3:]:
    if "=" in item:
        k, v = item.split("=", 1)
        data[k] = v
status_path.write_text(json.dumps(data, indent=2, sort_keys=True))
PY
}

runlog() {
  local log="$1"
  shift
  mkdir -p "${LOGS}"
  {
    printf '[%s] $' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf ' %q' "$@"
    printf '\n'
  } >> "${LOGS}/${log}"
  "$@" >> "${LOGS}/${log}" 2>&1
  local rc=$?
  printf '[%s] rc=%s\n\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${rc}" >> "${LOGS}/${log}"
  return "${rc}"
}

cd "${WORK}"

status "7_fod" "resume=ss3t_python39"
runlog "07_fod.log" /usr/bin/python "${MRTRIX3TISSUE}/ss3t_csd_beta1" \
  "${WORK}/sub_unbiased.mif" "${WORK}/wm.txt" "${WORK}/wmfod.mif" \
  "${WORK}/gm.txt" "${WORK}/gmfod.mif" \
  "${WORK}/csf.txt" "${WORK}/csffod.mif" \
  -mask "${WORK}/mask_on_b0.nii.gz" -force

runlog "07_fod.log" "${MRTRIX}/mtnormalise" \
  "${WORK}/wmfod.mif" "${WORK}/wmfod_norm.mif" \
  "${WORK}/gmfod.mif" "${WORK}/gmfod_norm.mif" \
  "${WORK}/csffod.mif" "${WORK}/csffod_norm.mif" \
  -mask "${WORK}/mask_on_b0.nii.gz" -force

status "8_5tt_and_gmwmi"
runlog "08_5tt.log" "${MRTRIX}/5ttgen" fsl "${WORK}/t12b0.nii.gz" "${WORK}/5tt1.mif" -force
runlog "08_5tt.log" "${MRTRIX}/5tt2gmwmi" "${WORK}/5tt1.mif" "${WORK}/gmwmSeed.mif" -force

status "9_tckgen"
runlog "09_tckgen.log" "${MRTRIX}/tckgen" \
  -act "${WORK}/5tt1.mif" -backtrack -seed_gmwmi "${WORK}/gmwmSeed.mif" \
  -maxlength 250 -cutoff "${CUTOFF}" -select "${SELECT_STREAMLINES}" -nthreads "${NTHREADS}" \
  "${WORK}/wmfod_norm.mif" "${WORK}/tracks_${SELECT_STREAMLINES}.tck" -force

status "10_tcksift2"
runlog "10_tcksift2.log" "${MRTRIX}/tcksift2" \
  -act "${WORK}/5tt1.mif" -out_coeffs "${WORK}/sift_coeffs.txt" -nthreads "${NTHREADS}" \
  "${WORK}/tracks_${SELECT_STREAMLINES}.tck" "${WORK}/wmfod_norm.mif" "${WORK}/sift_${SELECT_STREAMLINES}.txt" -force

status "11_aal116_to_b0"
runlog "11_aal_route.log" "${FSLDIR}/bin/flirt" \
  -in /home/ec2-user/exp/atlas/AAL116_spm12/aal/atlas/AAL.nii \
  -ref "${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz" \
  -dof 6 -omat "${WORK}/AAL2MNI.mat" -interp nearestneighbour -datatype int -out "${WORK}/AAL2MNI.nii.gz"
runlog "11_aal_route.log" "${FSLDIR}/bin/flirt" \
  -in "${WORK}/AAL2MNI.nii.gz" -ref "${WORK}/brainmask.nii.gz" \
  -omat "${WORK}/aalmni2t1.mat" -interp nearestneighbour -datatype int -out "${WORK}/AALmni2t1.nii.gz" -dof 12
runlog "11_aal_route.log" "${FSLDIR}/bin/flirt" \
  -in "${WORK}/AALmni2t1.nii.gz" -ref "${WORK}/vol0000.nii.gz" \
  -applyxfm -init "${WORK}/t12b0.mat" -interp nearestneighbour -out "${WORK}/AAL_sub.nii.gz"

PYTHONPATH=/home/ec2-user/exp /home/ec2-user/fsl/bin/python - "$RUN_ROOT" <<'PY'
from pathlib import Path
import sys
from run_sc_supervisor_raw_end_to_end_probe import aal116_sparse_to_contiguous, write_csv

run_root = Path(sys.argv[1])
work = run_root / "work"
report = aal116_sparse_to_contiguous(work / "AAL_sub.nii.gz", work / "AAL_sub_contig116.nii.gz")
write_csv(run_root / "aal116_label_remap.csv", [report])
PY

status "12_connectome"
runlog "12_connectome.log" "${MRTRIX}/tck2connectome" \
  -symmetric -zero_diagonal -scale_invnodevol -tck_weights_in "${WORK}/sift_${SELECT_STREAMLINES}.txt" \
  "${WORK}/tracks_${SELECT_STREAMLINES}.tck" "${WORK}/AAL_sub_contig116.nii.gz" "${RUN_ROOT}/SC_AAL.csv" \
  -out_assignment "${RUN_ROOT}/assignmentsaal.csv" -force
runlog "12_connectome.log" "${MRTRIX}/tck2connectome" \
  -symmetric -zero_diagonal -tck_weights_in "${WORK}/sift_${SELECT_STREAMLINES}.txt" \
  "${WORK}/tracks_${SELECT_STREAMLINES}.tck" "${WORK}/AAL_sub_contig116.nii.gz" "${RUN_ROOT}/TL_AAL.csv" \
  -scale_length -stat_edge mean -force

status "13_matrix_stats"
PYTHONPATH=/home/ec2-user/exp /home/ec2-user/fsl/bin/python - "$RUN_ROOT" "$SID" <<'PY'
from pathlib import Path
import sys
from run_sc_supervisor_raw_end_to_end_probe import matrix_stats, write_csv, write_findings, SUPERVISOR_BENCHMARKS, subject_id_from_sid, update_status

run_root = Path(sys.argv[1])
sid = sys.argv[2]
stats = [
    matrix_stats(run_root / "SC_AAL.csv", "probe_SC_AAL"),
    matrix_stats(run_root / "TL_AAL.csv", "probe_TL_AAL"),
]
benchmark = SUPERVISOR_BENCHMARKS.get(subject_id_from_sid(sid))
if benchmark and benchmark.exists():
    stats.append(matrix_stats(benchmark, "supervisor_benchmark_SC_AAL"))
write_csv(run_root / "matrix_stats.csv", stats)
write_findings(
    run_root,
    [
        "# Supervisor Raw End-to-End Probe",
        "",
        f"SID: `{sid}`",
        f"Run root: `{run_root}`",
        "Mode: near-reference replay with MRtrix3Tissue SS3T and FSL bias-correction fallback because ANTs N4 is not installed.",
        "",
        "## Matrix stats",
        "",
        *(f"- {row['label']}: n={row.get('n')}, density={row.get('density')}, zero_rows={row.get('zero_rows')}, error={row.get('error', '')}" for row in stats),
        "",
        "## Interpretation rule",
        "",
        "If this raw replay gives supervisor-like density and few/no whole zero rows, then the fix belongs upstream in preprocessing/T1-to-B0/AAL routing, not in post-only SC matrix repair.",
    ],
)
update_status(run_root, phase="done", matrix_stats_csv=str(run_root / "matrix_stats.csv"), findings_md=str(run_root / "findings.md"))
PY

echo "done: ${RUN_ROOT}"
