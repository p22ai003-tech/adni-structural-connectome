# connectome_monitor — imaging progress page

## What is this folder?

A one-page progress screen for the imaging pipeline, the part of the project
that turns raw MRI images into connectome matrices. It is a short Streamlit
script (Streamlit turns a Python script into a web page). It reads one status
file, written by the imaging pipeline while it runs, and shows:

- how many connectome matrices are finished, overall and per group (CN, MCI,
  AD), with progress bars;
- the mean and median matrix density;
- any alerts;
- backup and track-rebuild progress;
- which compute lanes and background services are running;
- the machine's load, free memory and free disk space.

The page reloads itself every 20 seconds. It only reads; it never changes a
file.

## Do I need it?

- **Only** on the computer that is running the imaging pipeline, while it runs.
- **No**, if you work with finished matrices or with the analysis results.
  There is nothing for it to show on such a machine.

How the imaging pipeline works and how it writes the status file: see the
Imaging section of the top-level [README.md](../../README.md).

The Streamlit dashboard has a richer version of the same screen: its
"🔴 Live Pipeline Monitor" view (see
[connectome_dashboard/README.md](../connectome_dashboard/README.md)).

## What's inside

| File | What it does |
|---|---|
| `monitor_app.py` | The whole page. Reads the status file, draws it, waits 20 seconds, and starts over. |

## Before you start

1. The project is installed (Install section of the
   [top-level README.md](../../README.md)) and the venv is on:

   ```bash
   source .venv/bin/activate
   ```

2. The Streamlit packages are installed:

   ```bash
   pip install -r requirements/dashboard.txt
   ```

3. The imaging pipeline has written its status file (see [Inputs](#inputs)).

## How to run it

**Step 1.** From the repository root:

```bash
python -m streamlit run apps/connectome_monitor/monitor_app.py \
  --server.address 127.0.0.1 --server.port 8502 \
  --server.headless true --browser.gatherUsageStats false
```

Port 8502 is used so the monitor does not clash with the dashboard on 8501.

What you should see:

```
  You can now view your Streamlit app in your browser.

  URL: http://127.0.0.1:8502
```

**Step 2.** Open http://127.0.0.1:8502. On the imaging machine you should see
something like:

```
🧠 Connectome Pipeline — Live Monitor
updated 2026-07-23T15:22:16Z UTC · auto-refresh 20s · source: canonical status file
✓ All systems nominal — no alerts
CONNECTOMES COMPLETE:  530 / 648   (82%)
Done (≥0.6)      530 / 648
Mean density     0.745
Median density   0.741
By cohort
AD — 78/100 (78%)
MCI — 201/241 (83%)
CN — 251/307 (82%)
...
```

**Step 3.** Press `Ctrl+C` in the terminal to stop it.

## Inputs

One JSON file: `connectome_status.json` in the `sc_matrix_qc/` folder of the
QC tree (`derivatives/qc/sc_matrix_qc/connectome_status.json` on the imaging
machine's data volume). It is written by `pipeline/monitoring/status_writer.py`
on the imaging machine.

**The location is fixed in the code.** It is the `STATUS = ...` line near the
top of `monitor_app.py`, and it does not follow the `SC_*` environment
variables. If the status file lives somewhere else on your machine, change that
one line to point at it.

The page expects these keys in the file: `ts`, `overall` (`done`, `total`,
`mean_d`, `median_d`), `cohort` (per group: `done`, `total`), `runs`, `host`
(`load_ratio`, `ncpu`, `ram_free`, `ram_total`, `disk_free_g`, `tckgen`), and
optionally `alerts`, `s3_pushed`, `s3_total` and `loops`.

## Outputs

None. The page only reads the status file.

## Settings you can change

| Setting | Where | What it does | Default |
|---|---|---|---|
| Status file | `STATUS = ...` line in `monitor_app.py` | Which file to read. | the imaging machine's `derivatives/qc/sc_matrix_qc/connectome_status.json` |
| Reload interval | `time.sleep(20)` at the end of `monitor_app.py` | Seconds between reloads. | 20 |
| `--server.port` | Streamlit option | Port number. | 8502 in the command above |
| `--server.address` | Streamlit option | Who may connect. Keep `127.0.0.1`. | `127.0.0.1` in the command above. If you leave this option out, Streamlit listens on all network interfaces and anyone on your network can open the page. Always keep `--server.address 127.0.0.1`. |

## Logins and passwords

The page has no login. With `--server.address 127.0.0.1` only your own
computer can open it. Keep it that way. On the project's shared server,
viewers sit behind nginx with a username and password; the credentials file is
kept outside the repository (by convention its location is in
`CONNECTOME_CREDENTIALS_FILE`) and must never be copied into it.

## If something goes wrong

**The page shows `Status file not available yet: [Errno 2] No such file or directory: '.../connectome_status.json'`**
The status file is not where the code expects it. On a machine that is not
running the imaging pipeline this is expected, and the page keeps retrying
every 10 seconds. On the imaging machine, check that the status writer is
running, or change the `STATUS = ...` line to the file's real location.

**`.../python: No module named streamlit`**
Run `source .venv/bin/activate` and `pip install -r requirements/dashboard.txt`.

**`Port 8502 is not available`**
Another program uses port 8502. Stop it, or choose another port with
`--server.port`.
