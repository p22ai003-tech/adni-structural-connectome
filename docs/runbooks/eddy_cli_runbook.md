# Eddy CLI Runbook

Run Eddy from the terminal instead of from notebook workflow cells.

## Load the shell environment on EC2

```bash
source ~/exp/scripts/eddy/eddy_env.sh
```

Available helpers after sourcing:

- `eddy_status <group>`
- `eddy_run <jobs> <threads> <group> [extra args]`
- `eddy_ad_status`
- `eddy_ad_run`
- `eddy_cli` (raw passthrough; use only when you really want the low-level script interface)
- sync/report helpers are in `~/exp/scripts/eddy/eddy_sync.py`

## Check Eddy status

All groups:

```bash
source ~/exp/scripts/eddy/eddy_env.sh
eddy_status all
```

AD only:

```bash
source ~/exp/scripts/eddy/eddy_env.sh
eddy_ad_status
```

## Run Eddy jobs

All groups:

```bash
source ~/exp/scripts/eddy/eddy_env.sh
eddy_run 1 1 all
```

AD only:

```bash
source ~/exp/scripts/eddy/eddy_env.sh
eddy_ad_run 1 1
```

One specific AD series:

```bash
source ~/exp/scripts/eddy/eddy_env.sh
eddy_ad_run 1 1 --series 129_S_6763_I1326107
```

Examples for the other group filters:

```bash
eddy_status mci
eddy_run 1 1 mci

eddy_status cn
eddy_run 1 1 cn
```

## Important EC2 output locations

- Eddy outputs: `~/exp/data/derivatives/eddy`
- Logs: `~/exp/data/derivatives/qc/eddy_logs`
- Summary CSV: `~/exp/data/derivatives/qc/eddy_summary.csv`
- Quarantine CSV: `~/exp/data/derivatives/qc/eddy_quarantine.csv`

## About the old local-machine paths

The old exported notebook paths existed only on the local machine that produced those notebooks.
They are not part of the EC2 workflow anymore.

Use the CLI helpers and the EC2-native notebooks against `~/exp/data/...`.
