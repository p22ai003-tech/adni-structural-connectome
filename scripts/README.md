# Runnable Script Layout

This folder holds runnable helpers that do not need to live at the project root.

## Folders

- `scforge/`: SC matrix QC, SC-Forge canaries, promotion, context generation, and repair review helpers.
- `scforge/legacy_probes/`: old one-off supervisor/debug probes kept for reference.
- `recovery/`: targeted connectome gap recovery runners and monitors.
- `preprocessing/`: shell launchers for legacy AAL preprocessing and denoise/Gibbs.
- `eddy/`: Eddy shell environment, GPU watcher, and AD Eddy tmux launcher.
- `s3/`: S3 pull/sync helper scripts.
- `dashboard/`: reserved for dashboard-specific helper wrappers; the hosted app itself lives in `apps/connectome_dashboard/`.
- `maintenance/`: reserved for future cleanup or inventory utilities.

## Root-Level Exceptions

The following active SC-Forge density files remain at `/home/ec2-user/exp` while the current `scforge_v1_density_batch` tmux run is live:

- `run_scforge_v1_density_batch.py`
- `run_sc_aal3_source_contract_probe.py`
- `run_sc_final_spatial_contract_closeout.py`
- `launch_scforge_v1_density_batch.sh`
- `watch_scforge_v1_density_monitor.sh`

They use absolute paths in the active tmux command and subprocess calls. Move them only after the active batch finishes or after adding tested compatibility wrappers.
