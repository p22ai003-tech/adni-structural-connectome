# Connectome Pipeline Package

Reusable structural-connectome pipeline modules.

## Contents

- `pipeline_paths.py`: shared path resolution.
- `pipeline_status.py`: stage status and cohort summaries.
- `dwi_convert.py`: strict raw-DWI source selection and MIF conversion helpers.
- `dwi_denoise_gibbs.py`: MRtrix denoise/Gibbs helper.
- `dwi_eddy.py`: Eddy/B0 provenance runner.
- `t1_pipeline.py`, `bbr_bridge.py`, `bias_correction.py`: anatomical and preprocessing helpers.
- `connectome_step7.py`: Step 7 implementation for T1/BBR, 5TT/GMWMI, FOD, tracks, parcellation, DTI, and connectome outputs.
- `connectome_run_control.py`: shared notebook/wrapper run-control gates.

Keep implementation here; put command launchers under `/home/ec2-user/exp/scripts`.
