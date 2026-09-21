# ADNI structural connectome

Diffusion MRI to structural connectomes to the statistical and machine-learning
results behind the thesis. Two pipelines, each with one entry point.

```bash
# describe your study once: where the data is, how it is laid out, where output goes
python sc_study.py --init configs/study.yaml

# imaging: a folder of T1 and DWI scans -> connectome matrices
source env.sh
python run_imaging.py --study configs/study.yaml doctor
python run_imaging.py --study configs/study.yaml discover --out work/pairs.csv
python sc_manifest_build.py --pairs work/pairs.csv --out configs/acquisition_manifest.csv
python run_imaging.py probe && python run_imaging.py validate
python run_imaging.py approve --run-root <run> --all --by "Your Name"
python run_imaging.py run --run-root <run> --cores 16
python run_imaging.py publish --run-root <run>

# analysis: connectome matrices -> every table the thesis reports
python -m connectome_analysis.run_analysis --all --resume
```

The imaging route is documented step by step in **[IMAGING.md](IMAGING.md)**, and
the three input layouts it accepts in
**[docs/INPUT_LAYOUTS.md](docs/INPUT_LAYOUTS.md)**. Your data may be a local
directory, a mounted volume, or an `s3://` prefix that is staged before the run.

| | |
|---|---|
| `IMAGING.md` | imaging runbook: install, layouts, the six steps, the gates |
| `docs/INPUT_LAYOUTS.md` | what your input folder has to look like |
| `configs/study.example.yaml` | the one file describing a study |
| `connectome_analysis/README.md` | analysis runbook: the 37-stage graph, parameters, outputs |
| `sc_config.py` | every path, from at most three `SC_*` roots |
| `sc_doctor.py` | preflight: paths, toolchain, packages, input contract |
| `configs/analysis.yaml` | statistical parameters |
| `configs/connectome_v2.yaml` | imaging workflow parameters |

## Install

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements/analysis.txt
.venv/bin/pip install -r requirements/dev.txt        # to run the 270 tests
cp env.sh.example env.sh                             # then edit the three tool paths
python sc_doctor.py
```

Pins are exact, not minimums: the ML results move with the scikit-learn version.

## Nothing here is tied to this machine

Paths derive from at most three environment variables, resolved in
`sc_config.py`; `python -c "import sc_config; print(sc_config.describe())"`
prints what they resolve to and marks anything absent.

## This repository ships no data

No imaging, no participant tables, no acquisition manifests. Per-subject
derivatives, cohort tables and manifests are excluded by `.gitignore`, and CI
fails a commit that adds one. `LICENSE` covers the source code only.

The results here were produced from ADNI, which is governed by a Data Use
Agreement; obtain that data from https://adni.loni.usc.edu/ under your own
agreement. The pipeline itself is not ADNI-specific — see
[docs/INPUT_LAYOUTS.md](docs/INPUT_LAYOUTS.md).

## Third-party licences

The Python dependencies are permissively licensed, and the atlas shipped in
`atlas/` is AAL3, free for academic use.

Of the four imaging toolkits the pipeline calls, three — MRtrix3, ANTs and
dcm2niix — are permissively licensed. **FSL is free for academic use but not for
commercial use**; read its licence at
https://fsl.fmrib.ox.ac.uk/fsl/docs/#/license before using this pipeline
commercially. FSL is used for eddy-current and motion correction and for BBR
registration.

## Two things a reader of the results should know

**T1–DWI gap.** The median gap between the T1 and the DWI is 753 days, and 321
of 530 pairs exceed 180 days. The AAL3 parcellation is carried from the T1 into
DWI space, so for most subjects the anatomy defining the nodes predates the
diffusion data. The strata are unbalanced across groups.

**SMC.** The v2 manifest records 15 subjects as SMC and flags them
`smc_retained_separately`; `master_cohort.csv` labels all 15 `MCI`. The MCI
group of 201 is 186 MCI plus 15 SMC.

## Historical entry points

These predate the two runners above and are kept because the audit trail
references them.

- `research_audit/SUPERLIST.md`: execution checklist and human gates.
- `research_audit/objective1_audit_report.md`: scientific and technical audit.
- `structural_connectome_context.md`: technical context and implementation summary.
- `notebooks/structural_connectome_A.ipynb`: DWI conversion, denoise/Gibbs, eddy provenance.
- `notebooks/structural_connectome_B.ipynb`: T1/BBR, 5TT/GMWMI, FOD, tracks, connectomes.
- `notebooks/structural_connectome_QC.ipynb`: SC matrix QC and repair decisions.
- `connectome_pipeline/`: pipeline modules used by the notebooks and CLI wrappers.
- `scforge/`: contract-first SC-Forge package, configs, workflow rules and tests.
- `apps/connectome_dashboard/`: Streamlit dashboard and its refresh loop.
- `scripts/`: helpers by purpose (`scforge/`, `recovery/`, `preprocessing/`, `eddy/`, `s3/`, `maintenance/`).

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

## Analysis + ML pipeline

One command regenerates every number the thesis and the dashboard report.
See `connectome_analysis/README.md` for the full runbook.

```bash
python -m connectome_analysis.run_analysis --list          # the 37-stage graph
python -m connectome_analysis.run_analysis --all --resume   # run what is missing or stale
```

Paths come from `sc_config.py` and the `SC_*` environment variables; run
`python -c "import sc_config; print(sc_config.describe())"` to check them first.

The older entry point still works and forwards to the runner:

```bash
apps/connectome_dashboard/refresh_connectome_dashboard_data.py --mode quick|full
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
