#!/usr/bin/env bash
set -euo pipefail

# Reference MRtrix/FSL AAL connectome pipeline.
# This script is kept as a readable contract, not as the live SC-Forge batch
# runner. Prefer the scripts under scripts/scforge/live for current production
# rescue work.

SUB="${SUB:-subj}"
DWI_NII="${DWI_NII:-Axial_DTI*.nii.gz}"
BVEC="${BVEC:-Axial_DTI*.bvec}"
BVAL="${BVAL:-Axial_DTI*.bval}"
JSON="${JSON:-Axial_DTI*.json}"
T1W="${T1W:-*T1w.nii.gz}"
AAL_MNI="${AAL_MNI:-/home/pc/Documents/research/utils/BrainNetViewer_20191031/Data/ExampleFiles/AAL90/aal.nii}"
MNI_BRAIN="${MNI_BRAIN:-${FSL_DIR}/data/standard/MNI152_T1_1mm_brain.nii.gz}"
FS_SUBJECT_DIR="${FS_SUBJECT_DIR:-/home/pc/freesurfer/subjects/subj_recon}"
THREADS="${THREADS:-8}"

DWI_MIF="${SUB}_dwi_raw.mif"
DWI_DEN="${SUB}_dwi_den.mif"
DWI_UNR="${SUB}_dwi_den_unr.mif"
DWI_PRE="${SUB}_dwi_preproc.mif"
DWI_UNB="${SUB}_dwi_unbiased.mif"
MEAN_B0_MIF="${SUB}_mean_b0.mif"
MEAN_B0_NII="${SUB}_mean_b0.nii.gz"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] subject=${SUB}"

mrconvert ${DWI_NII} "${DWI_MIF}" \
  -fslgrad ${BVEC} ${BVAL} \
  -json_import ${JSON} \
  -force

dwidenoise "${DWI_MIF}" "${DWI_DEN}" \
  -noise "${SUB}_noise.mif" \
  -force

mrdegibbs "${DWI_DEN}" "${DWI_UNR}" \
  -force

dwifslpreproc "${DWI_UNR}" "${DWI_PRE}" \
  -rpe_none \
  -pe_dir j- \
  -eddy_options " --slm=linear --data_is_shelled" \
  -force

dwibiascorrect ants "${DWI_PRE}" "${DWI_UNB}" \
  -fslgrad ${BVEC} ${BVAL} \
  -force

# Use the mean of all b0 volumes. Do not fslsplit and assume vol0000 is the
# representative b0; that was one source of brittle historical behavior.
dwiextract "${DWI_UNB}" - -bzero | mrmath - mean "${MEAN_B0_MIF}" -axis 3 -force
mrconvert "${MEAN_B0_MIF}" "${MEAN_B0_NII}" -force

# FreeSurfer output conversion. Run recon-all separately if needed:
# recon-all -i "${T1W}" -s subj_recon -all -nthreads "${THREADS}"
mrconvert "${FS_SUBJECT_DIR}/mri/nu.mgz" corrected_T1.nii.gz -force
mrconvert "${FS_SUBJECT_DIR}/mri/brainmask.mgz" brainmask.nii.gz -force
bet2 brainmask.nii.gz mask.nii.gz

flirt -in corrected_T1.nii.gz \
  -ref "${MEAN_B0_NII}" \
  -omat t12b0.mat \
  -out t12b0.nii.gz \
  -dof 6

flirt -in mask.nii.gz \
  -ref "${MEAN_B0_NII}" \
  -applyxfm \
  -init t12b0.mat \
  -interp nearestneighbour \
  -out mask_on_b0.nii.gz

dwi2response dhollander "${DWI_UNB}" wm.txt gm.txt csf.txt \
  -shell 0,1000 \
  -mask mask_on_b0.nii.gz \
  -voxels voxels.mif \
  -fslgrad ${BVEC} ${BVAL} \
  -force

ss3t_csd_beta1 "${DWI_UNB}" wm.txt wmfod.mif gm.txt gmfod.mif csf.txt csffod.mif \
  -mask mask_on_b0.nii.gz \
  -force

mtnormalise wmfod.mif wmfod_norm.mif gmfod.mif gmfod_norm.mif csffod.mif csffod_norm.mif \
  -mask mask_on_b0.nii.gz \
  -force

5ttgen fsl t12b0.nii.gz 5tt1.mif -force
5tt2gmwmi 5tt1.mif gmwmSeed.mif -force

tckgen wmfod_norm.mif tracks_10M.tck \
  -act 5tt1.mif \
  -backtrack \
  -seed_gmwmi gmwmSeed.mif \
  -maxlength 250 \
  -cutoff 0.06 \
  -select 10000000 \
  -nthreads "${THREADS}" \
  -force

tcksift2 tracks_10M.tck wmfod_norm.mif sift_10M.txt \
  -act 5tt1.mif \
  -out_coeffs sift_coeffs.txt \
  -nthreads "${THREADS}" \
  -force

# High-resolution atlas contract:
# 1. AAL/AAL3 in MNI -> native T1 with nearest-neighbour labels.
# 2. Transform native-T1 labels into DWI/world space without forcing the atlas
#    to the low-resolution b0 voxel grid when a 1 mm T1 label image is available.
flirt -in "${MNI_BRAIN}" \
  -ref corrected_T1.nii.gz \
  -omat mni2t1.mat \
  -dof 12

flirt -in "${AAL_MNI}" \
  -ref corrected_T1.nii.gz \
  -applyxfm \
  -init mni2t1.mat \
  -interp nearestneighbour \
  -datatype int \
  -out AAL_T1_1mm.nii.gz

transformconvert t12b0.mat corrected_T1.nii.gz "${MEAN_B0_NII}" flirt_import T1_to_b0_mrtrix.txt
mrconvert AAL_T1_1mm.nii.gz AAL_T1_1mm.mif -datatype uint32 -force
mrtransform AAL_T1_1mm.mif \
  -linear T1_to_b0_mrtrix.txt \
  -interp nearest \
  -datatype uint32 \
  AAL_dwi_1mm.mif \
  -force

tck2connectome tracks_10M.tck AAL_dwi_1mm.mif SC_AAL_fd_sum.csv \
  -symmetric \
  -zero_diagonal \
  -assignment_radial_search 4 \
  -scale_invnodevol \
  -stat_edge sum \
  -tck_weights_in sift_10M.txt \
  -out_assignments assignments_AAL_fd_sum.csv \
  -force

tck2connectome tracks_10M.tck AAL_dwi_1mm.mif TL_AAL_len_mean.csv \
  -symmetric \
  -zero_diagonal \
  -assignment_radial_search 4 \
  -tck_weights_in sift_10M.txt \
  -scale_length \
  -stat_edge mean \
  -force
