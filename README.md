# Structural Connectome Project

This is the structural-connectome project root. It remains at `/home/ec2-user/exp` because active scripts, notebooks, tmux sessions, dashboard service files, and SC-Forge runners use absolute paths under this directory.

## Main Entry Points

- `research_audit/SUPERLIST.md` / `SUPERLIST.docx`: authoritative execution checklist and human gates (“superlist”).
- `research_audit/objective1_audit_report.md` / `.docx`: decisive scientific/technical audit and corrective plan.
- `structural_connectome_context.md` / `structural_connectome_context.docx`: current technical context and implementation summary.
- `notebooks/structural_connectome_A.ipynb`: early DWI conversion, denoise/Gibbs, Eddy/B0 provenance.
- `notebooks/structural_connectome_B.ipynb`: Step 7 T1/BBR, 5TT/GMWMI, FOD, tracks, parcellation, DTI, connectomes.
- `notebooks/structural_connectome_QC.ipynb`: SC matrix QC and repair decision work.
- `connectome_pipeline/`: reusable pipeline modules used by notebooks and command-line wrappers, including `connectome_step7.py`, DWI conversion/denoise/Eddy helpers, T1/BBR helpers, and shared path/status utilities.
- `connectome_analysis/`: packaged analysis modules used by notebook C, dashboard refreshes, and SC matrix QC reports.
- `scforge/`: contract-first SC-Forge package, configs, workflow rules, and tests.
- `apps/connectome_dashboard/`: hosted Streamlit dashboard entrypoint, refresh loop, credentials note, and old app logs.
- `scripts/`: runnable helpers organized by purpose: `scforge/`, `recovery/`, `preprocessing/`, `eddy/`, `s3/`, and `maintenance/`.
- `run_scforge_v1_density_batch.py`: compatibility entry point for the historical SC-Forge v1 density batch.
- `watch_scforge_v1_density_monitor.sh`: historical/resumable compact density-batch monitor. No active tractography or refresh process was found in the 2026-07-18 audit.

## Important Paths

- `data -> /data`: live structural-connectome data and derivatives symlink.
- `data/derivatives/qc/sc_matrix_qc`: SC QC and SC-Forge run evidence.
- `atlas/`: AAL3/AAL116 atlas resources.
- `cohort/`: ADNI cohort metadata.
- `docs/literature/`: structural-connectome literature and slide material moved out of the project root.
- `docs/s3/` and `docs/recovery/`: operational pointers moved out of the project root.
- `archive/`: recoverable historical notebooks, old monitors, legacy debug scripts, and large import archives.

## Hosted Dashboard

The EC2 Streamlit service now points to:

```bash
/home/ec2-user/exp/apps/connectome_dashboard/connectome_app.py
```

Service definition:

```bash
/home/ec2-user/exp/deploy/connectome-dashboard.service
```

Refresh helper:

```bash
/home/ec2-user/exp/apps/connectome_dashboard/refresh_connectome_dashboard_data.py
```

## Run Safety

Do not move or rename these if `scforge_v1_density_batch` is deliberately resumed:

- `run_scforge_v1_density_batch.py`
- `run_sc_aal3_source_contract_probe.py`
- `run_sc_final_spatial_contract_closeout.py`
- `launch_scforge_v1_density_batch.sh`
- `watch_scforge_v1_density_monitor.sh`

The three top-level `run_sc*.py` files are compatibility wrappers. Their implementations live in:

```bash
/home/ec2-user/exp/scripts/scforge/live/
```

Monitor:

```bash
tmux attach -t scforge_v1_density_monitor
```
