# Connectome C Dashboard Deployment

This deployment serves the read-only Streamlit dashboard from localhost behind nginx.

- Streamlit binds to `127.0.0.1:8501`.
- nginx exposes the dashboard on HTTPS `443`; the default nginx HTTP `80` site is not used for the dashboard.
- nginx basic auth is required before the app is visible.
- The certificate is self-signed unless replaced with a real domain certificate.
- External access is restricted at the EC2 security-group layer to specific `/32` client IPs, not the open internet.
- Direct external access to Streamlit port `8501` is not allowed; Streamlit listens on localhost only.

Current URL:

`https://<dashboard-host>/`

Credentials are stored locally at:

a location outside the repository (set `CONNECTOME_CREDENTIALS_FILE`). The note that previously lived in the repo was moved out before the repository was placed under version control; rotate the basic-auth password before reusing it.

The Streamlit entrypoint is:

`/home/ec2-user/exp/apps/connectome_dashboard/connectome_app.py`

The app reads analysis outputs from:

`/home/ec2-user/exp/data/derivatives/qc/analysis_cohort`

Those outputs are generated from:

- cohort tables: `/home/ec2-user/exp/cohort/dti.csv` and `/home/ec2-user/exp/cohort/mri.csv`
- current pipeline derivatives: `/home/ec2-user/exp/data/derivatives`
- final connectome matrices: `/home/ec2-user/exp/data/derivatives/connectomes/SC_AAL_*_*.csv`

Refresh commands:

```bash
# One lightweight refresh of counts, availability, source tables, and export manifests
/home/ec2-user/exp/.venv_connectome_app/bin/python /home/ec2-user/exp/apps/connectome_dashboard/refresh_connectome_dashboard_data.py --mode quick

# Background quick refresh every 5 minutes
/home/ec2-user/exp/apps/connectome_dashboard/start_connectome_dashboard_refresh_tmux.sh
tmux attach -t connectome_dashboard_refresh

# Manual heavier refresh that regenerates section statistics and PNGs
/home/ec2-user/exp/.venv_connectome_app/bin/python /home/ec2-user/exp/apps/connectome_dashboard/refresh_connectome_dashboard_data.py --mode full
```

Useful commands:

```bash
sudo systemctl status connectome-dashboard --no-pager
sudo systemctl restart connectome-dashboard
sudo nginx -t
sudo systemctl status nginx --no-pager
```
