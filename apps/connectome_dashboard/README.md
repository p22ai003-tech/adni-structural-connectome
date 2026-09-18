# Connectome Dashboard App

Hosted Streamlit dashboard for the structural-connectome cohort outputs.

## Entrypoints

- `connectome_app.py`: Streamlit app served by `connectome-dashboard.service`.
- `refresh_connectome_dashboard_data.py`: rebuilds the generated dashboard CSV/PNG/export tree.
- `run_connectome_dashboard_refresh_loop.sh`: repeated refresh loop.
- `start_connectome_dashboard_refresh_tmux.sh`: tmux launcher for the refresh loop.

## Runtime Paths

- Project root: `/home/ec2-user/exp`
- Analysis output root: `/home/ec2-user/exp/data/derivatives/qc/analysis_cohort`
- Live matrices: `/home/ec2-user/exp/data/derivatives/connectomes`
- Service file: `/home/ec2-user/exp/deploy/connectome-dashboard.service`
- Credentials note: kept outside the repository (see `CONNECTOME_CREDENTIALS_FILE`); never commit it.

The app code lives here. Reusable analysis modules live in `/home/ec2-user/exp/connectome_analysis`, and runnable helper scripts live in `/home/ec2-user/exp/scripts`.
