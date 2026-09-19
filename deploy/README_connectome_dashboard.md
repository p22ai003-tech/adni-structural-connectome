# Connectome C Dashboard Deployment (original server)

This page records how the read-only Streamlit dashboard was served on the
original server. "Connectome C" is only the dashboard's name; it is the same
dashboard that [README.md](README.md) deploys. To deploy it on your own server, follow
[README.md](README.md) in this folder; it explains every value below that has
to change.

## How it was set up

- Streamlit binds to `127.0.0.1:8501` (this machine only).
- nginx exposes the dashboard on HTTPS port `443`. The default nginx HTTP `80`
  site is not used for the dashboard.
- nginx basic authentication (a user name and password) is required before the
  app is visible.
- The certificate is self-signed unless it is replaced with a real domain
  certificate.
- External access is restricted at the EC2 security-group layer (the cloud
  firewall) to specific `/32` client addresses, not the open internet.
- Direct external access to Streamlit port `8501` is not allowed; Streamlit
  listens on localhost only.

URL: `https://<dashboard-host>/`

Credentials are kept outside the repository. The password file is
`/etc/connectome-dashboard/htpasswd` and the certificate and key are in
`/etc/connectome-dashboard/tls/`. The note of the login was also kept outside
the repository, at a location recorded in `CONNECTOME_CREDENTIALS_FILE` (no
program reads that variable). The note that previously lived in the repository
was moved out before the repository was placed under version control. Rotate
(that is, change) the basic-auth password, the one nginx asks for, before
reusing that login.

## What it runs and reads

All paths below are relative to the repository root, or given as the `SC_*`
variables that `sc_config.py` resolves.

- Service file: `deploy/connectome-dashboard.service`
- Streamlit entry point: `apps/connectome_dashboard/connectome_app.py`
- Analysis outputs it displays: `$SC_ANALYSIS_ROOT` (on that server, the
  default `data/derivatives/qc/analysis_cohort`), passed to the app as
  `CONNECTOME_ANALYSIS_ROOT`.

Those outputs are generated from:

- cohort tables: `$SC_COHORT_DIR/dti.csv` and `$SC_COHORT_DIR/mri.csv`
- pipeline derivatives: `$SC_DERIV_ROOT`
- connectome matrices: `$SC_CONNECTOMES_DIR/SC_AAL166_<SUBJECT>_I<IMAGEID>_<type>.csv`

## Refreshing the outputs

On the original server a loop re-ran the quick refresh every 5 minutes while
the imaging was still producing matrices
(`apps/connectome_dashboard/start_connectome_dashboard_refresh_tmux.sh`, which
starts `run_connectome_dashboard_refresh_loop.sh` in a tmux session; tmux is a
program that keeps a terminal session running after you disconnect). Its
defaults (`PROJECT_ROOT`, `PYTHON`, `LOG_DIR`) are that server's folders; set
those variables before using it anywhere else.

The refresh script, `apps/connectome_dashboard/refresh_connectome_dashboard_data.py`,
now only forwards to the analysis runner. The equivalent runner commands, run
from the repository root with the venv on, are:

```bash
# --mode quick: the cohort and QC stages
python -m connectome_analysis.run_analysis --group cohort

# --mode full: every stage, including the statistics and figures
python -m connectome_analysis.run_analysis --all
```

See `connectome_analysis/README.md` for what each stage does.

## Useful commands

```bash
sudo systemctl status connectome-dashboard --no-pager
sudo systemctl restart connectome-dashboard
sudo nginx -t
sudo systemctl status nginx --no-pager
```
