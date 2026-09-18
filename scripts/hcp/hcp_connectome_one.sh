#!/bin/bash
# Per-subject HCP-MMP1(+subcortical) parcellation -> connectome, on ALREADY-EXISTING streamlines.
# Requires a completed FreeSurfer/FastSurfer recon in $SUBJECTS_DIR/<fsid> (surfaces + sphere.reg + aseg).
# Mirrors the frozen scforge AAL3 recipe exactly: radial4, -symmetric -zero_diagonal, same 8 weightings.
# Usage: hcp_connectome_one.sh <fsid>
set -o pipefail
FSID="${1:?need fsid}"

export FREESURFER_HOME=/home/ec2-user/freesurfer
export SUBJECTS_DIR=/data/derivatives/fastsurfer
export FS_LICENSE=/home/ec2-user/freesurfer/license.txt
source "$FREESURFER_HOME/SetUpFreeSurfer.sh" >/dev/null 2>&1
export PATH="/home/ec2-user/mrtrix3/bin:$PATH"
PY=/home/ec2-user/fsl/bin/python
HCP=/home/ec2-user/hcp_atlas
LUT=/data/derivatives/parc_hcpmmp1/hcpmmp1_subcort_relabel.txt

TRK=/data/derivatives/tracks/$FSID
B0=/data/derivatives/dwi_t1_bbr/${FSID}_b0mean_ras.nii.gz
WORK=/data/derivatives/parc_hcpmmp1/$FSID
OUT=/data/derivatives/connectomes_hcp_fs
mkdir -p "$WORK" "$OUT"

log(){ echo "$(date -u +%H:%M:%SZ) [$FSID] $*"; }
need(){ [ -e "$1" ] || { log "MISSING $1"; exit 3; }; }

# idempotent + race-safe: one builder per subject; skip if already complete.
# (lets a rolling worker + the master's Phase D coexist without clobbering each other)
exec 9>"$OUT/.lock_${FSID}"
flock -n 9 || { log "locked by another builder, skip"; exit 0; }
[ -f "$OUT/SC_HCPMMP1_${FSID}_count_invnodevol.csv" ] && { log "already built, skip"; exit 0; }

need "$SUBJECTS_DIR/$FSID/surf/lh.sphere.reg"
need "$SUBJECTS_DIR/$FSID/mri/aseg.mgz"
need "$TRK/tracks_final_3000k.tck"; need "$TRK/sift_weights.txt"; need "$B0"
for m in fa md rd ad; do need "$TRK/${m}_mean.tsf"; done

# ensure fsaverage visible to surf2surf
[ -e "$SUBJECTS_DIR/fsaverage" ] || ln -s "$FREESURFER_HOME/subjects/fsaverage" "$SUBJECTS_DIR/fsaverage" 2>/dev/null

# 1) map HCP-MMP1 annot fsaverage -> subject surface (both hemis)
for h in lh rh; do
  [ -f "$SUBJECTS_DIR/$FSID/label/${h}.HCPMMP1.annot" ] || \
  mri_surf2surf --srcsubject fsaverage --trgsubject "$FSID" --hemi "$h" \
    --sval-annot "$HCP/${h}.HCPMMP1.annot" \
    --tval "$SUBJECTS_DIR/$FSID/label/${h}.HCPMMP1.annot" >>"$WORK/step.log" 2>&1 || { log "surf2surf $h FAIL"; exit 4; }
done
log "surf2surf done"

# 2) HCP cortical + aseg subcortical volume (FS conformed space): cortex=1000+idx/2000+idx, subcort=aseg ids
mri_aparc2aseg --s "$FSID" --annot HCPMMP1 --o "$WORK/HCPMMP1+aseg.mgz" >>"$WORK/step.log" 2>&1 \
  || { log "aparc2aseg FAIL"; exit 5; }
log "aparc2aseg done"

# 3) register b0 -> FS subject (boundary-based, T2-like contrast)
bbregister --s "$FSID" --mov "$B0" --t2 --init-fsl --reg "$WORK/b0_to_fs.dat" >>"$WORK/step.log" 2>&1 \
  || { log "bbregister FAIL"; exit 6; }
# resample the FS-space HCP+aseg atlas ONTO the b0 grid. mri_label2vol (seg->template via reg) has
# unambiguous geometry (out = --temp = b0) + NN label handling; mri_vol2vol --inv mis-sized + smeared it.
mri_label2vol --seg "$WORK/HCPMMP1+aseg.mgz" --temp "$B0" --reg "$WORK/b0_to_fs.dat" \
  --o "$WORK/HCPMMP1+aseg_b0.nii.gz" >>"$WORK/step.log" 2>&1 || { log "label2vol FAIL"; exit 7; }
log "atlas in b0 space"

# 4) relabel to contiguous 1..379 + node volumes
"$PY" /home/ec2-user/exp/scripts/hcp/relabel_hcp.py \
  "$WORK/HCPMMP1+aseg_b0.nii.gz" "$LUT" "$WORK/nodes_b0.nii.gz" "$WORK/node_volumes.csv" \
  >>"$WORK/step.log" 2>&1 || { log "relabel FAIL"; exit 8; }
NODES="$WORK/nodes_b0.nii.gz"
log "relabelled to nodes_b0"

# 5) connectomes — mirror scforge exactly (radial4, symmetric, zero_diagonal)
TCK="$TRK/tracks_final_3000k.tck"; SIFT="$TRK/sift_weights.txt"
COMMON=(-symmetric -zero_diagonal -assignment_radial_search 4 -force -quiet -nthreads 2)
P="$OUT/SC_HCPMMP1_${FSID}"
tck2connectome "$TCK" "$NODES" "${P}_count.csv"       "${COMMON[@]}" -stat_edge sum || exit 9
tck2connectome "$TCK" "$NODES" "${P}_fd_sum.csv"      "${COMMON[@]}" -tck_weights_in "$SIFT" -stat_edge sum || exit 9
tck2connectome "$TCK" "$NODES" "${P}_len_mean.csv"    "${COMMON[@]}" -scale_length -stat_edge mean || exit 9
tck2connectome "$TCK" "$NODES" "${P}_invlen_mean.csv" "${COMMON[@]}" -scale_invlength -stat_edge mean || exit 9
for m in fa md rd ad; do
  tck2connectome "$TCK" "$NODES" "${P}_${m}_mean.csv" "${COMMON[@]}" -scale_file "$TRK/${m}_mean.tsf" -stat_edge mean || exit 9
done
log "8 tck2connectome matrices written"

# 6) derive count_invnodevol = 2*count/(Vi+Vj) from count + node volumes (scforge formula)
"$PY" - "$FSID" <<'PYEOF'
import sys, numpy as np, csv
fsid = sys.argv[1]
OUT=f"/data/derivatives/connectomes_hcp_fs/SC_HCPMMP1_{fsid}"
WORK=f"/data/derivatives/parc_hcpmmp1/{fsid}"
count = np.loadtxt(f"{OUT}_count.csv", delimiter=",")
vols = {}
with open(f"{WORK}/node_volumes.csv") as f:
    r = csv.DictReader(f)
    for row in r: vols[int(row["node_id"])] = float(row["volume_mm3"])
n = count.shape[0]
V = np.array([vols.get(i+1, 0.0) for i in range(n)])
denom = V[:,None] + V[None,:]
with np.errstate(divide="ignore", invalid="ignore"):
    inv = np.where(denom > 0, 2.0*count/denom, 0.0)
np.fill_diagonal(inv, 0.0)
np.savetxt(f"{OUT}_count_invnodevol.csv", inv, delimiter=",", fmt="%.10g")
print(f"count_invnodevol {n}x{n} written")
PYEOF
log "DONE — 9 matrices in $OUT"
