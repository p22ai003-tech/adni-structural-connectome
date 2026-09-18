# Analysis + ML pipeline

Everything from the connectome matrices to the tables the thesis and the
dashboard report. Given a derivatives tree containing per-subject connectomes
and the cohort CSVs, one command regenerates every number.

```bash
python -m connectome_analysis.run_analysis --list          # the stage graph
python -m connectome_analysis.run_analysis --all           # run everything
python -m connectome_analysis.run_analysis --all --resume  # only what is missing or stale
```

## Install

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements/analysis.txt      # to run the pipeline
.venv/bin/pip install -r requirements/dev.txt           # to run the tests
.venv/bin/pip install -r requirements/extras-ml.txt     # optional model backends
```

Python 3.12 is required. The pins are exact, not minimums: the ML results move
with the scikit-learn version, because tree-ensemble split selection and the
SHAP interface both changed across minor releases.

## Where the data is

No path is written in the code. Everything derives from at most three roots,
each overridable by one environment variable, resolved in `sc_config.py`:

```bash
export SC_PROJECT_ROOT=/path/to/repo      # default: the repo itself
export SC_DATA_ROOT=/mnt/adni             # default: $SC_PROJECT_ROOT/data
export SC_DERIV_ROOT=/mnt/adni/derivatives
python -c "import sc_config; print(sc_config.describe())"   # check before running
```

`describe()` marks anything that does not exist, which is the fastest way to
find a misconfigured volume before a long run fails in the middle.

## The stage graph

37 stages in five groups, each declaring what it needs and what it produces:

| group | what it does |
|---|---|
| `cohort` | master cohort table, QC, completeness, demographics |
| `measures` | per-subject microstructure, graph metrics, coupling, network tables |
| `inference` | edge-wise tests, brain age, length splits, **EDR exceptions**, mediation |
| `ml` | diagnostic ML sweep, clinical-outcome search |
| `build` | the feature matrix, the final model, SHAP, the consensus core |

Selection is topologically sorted, never taken in the order given:

```bash
--stage build_shap_overall --with-deps   # that stage and all 13 it depends on
--from edr_exceptions                    # that stage and everything downstream
--group build                            # one group
--skip clinical_outcome_search           # drop a stage from the selection
--dry-run                                # print the plan, run nothing
```

Declared inputs are checked before a stage runs. Several analysis functions
guard their reads with `if path.exists()` and degrade silently when an input is
absent; here a missing declared input is an error, reported as `[BLOCK]`.

Resume is judged from the files on disk rather than from a checkpoint that can
disagree with them. A stage is skipped when all of its declared outputs exist
and none is older than an input.

Progress is written to `exports/stage_status.csv` and
`00_master/run_analysis_status.json`.

## Parameters

`configs/analysis.yaml` holds the statistics. Resolution order is
`stages.<stage>.<key>`, then `defaults.<key>`, then the value in the code, so
the file only has to mention what you want to change.

```bash
python -m connectome_analysis.run_analysis --all --config my_analysis.yaml
```

Two things in that file are worth reading before changing anything: the final
model definition, and the note explaining that this project has **two different
length splits** — `lr_sr` uses 33/67 tertiles, `edr_exceptions` uses CN
quartiles (SR ≤ 78.6856 mm, LR > 171.021 mm). They are not interchangeable and
neither reads the other. The thesis uses the CN-quartile definition.

## Ordering that matters

Two orderings are load-bearing and are enforced by the graph rather than by
convention:

- `mediation` runs after `live_brain_age`, whose predictions it reads. Its read
  is guarded by `exists()`, so running it early drops the brain-age mediator
  silently and still reports success.
- `build_shap_overall` runs last. It rewrites four tables that
  `build_ml_explain_v2` and `build_explain_v3` also write, with the
  seed-averaged version computed over all 105 features. Running it earlier
  leaves the single-seed, top-k-truncated tables in place, which overstate the
  reported shares.

## Outputs

Stages write straight into the analysis tree section their consumer reads, so
there is no copy step. The two that used to exist — five `network_*` tables into
`19_network_analysis/functional/` and forty-three files into
`20_exception_specificity/` — are gone.

```
00_master/          master cohort, QC snapshot, run status
01_qc/ .. 15_/      completeness, demographics, microstructure, graph, coupling,
                    edge-wise, brain age, length/delay, clinical, structural
17_edr_exceptions/  the thesis's central stage
19_network_analysis/functional/   network tables (AAL3 -> Yeo)
20_exception_specificity/         feature matrix, final model, SHAP, consensus core
21_mediation/       mediation chains
exports/            stage_status.csv
```

## Atlas labels

Use `atlas/AAL/aal3_node_map_166.csv`, never `AAL3_labels.csv`. The latter is
keyed on the original AAL3 atlas value, and AAL3v1 leaves values 35, 36, 81 and
82 unused; the 166-node matrix compacts the survivors, so matrix row 37 is
atlas value 39. Reading names from the wrong table mislabels 132 of 166 nodes.
Network assignments are unaffected either way, because
`AAL3_network_mapping.csv` is keyed on the matrix index and every consumer joins
on the index rather than the name.

`relabel_edr_rois.py` repairs artifacts written before this was fixed.

## Backwards compatibility

`apps/connectome_dashboard/refresh_connectome_dashboard_data.py` still works and
still takes `--mode quick|full`. It forwards to this runner.

## Tests

```bash
.venv/bin/python -m pytest -q                      # 270 tests
.venv/bin/python -m pytest tests/dashboard_parity  # 23, the dashboard contract
```
