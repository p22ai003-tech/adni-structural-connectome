# connectome_dashboard_core — the results-reading library

## What is this folder?

A small Python library (a package of functions other programs import). It
knows where the analysis results live, which result files belong to which
dashboard section, and how to turn those files into tidy tables and summaries.
It does not start a server and does not draw anything.

Its main user is the data server in `apps/connectome_api/`: every answer that
server gives is produced by a function here. The dashboard tests in
`tests/dashboard_parity/` use it too. (The Streamlit dashboard in
`apps/connectome_dashboard/` does not import it; it has its own built-in
readers, and the tests check that both give the same numbers.)

It only reads. Nothing here writes, moves or deletes a file.

## Do I need it?

- **You use it without noticing** whenever you run the data server or the
  React web app. There is nothing to start.
- **You call it directly** if you want the same numbers as the web app inside
  your own Python script or notebook.
- **You can ignore it** if you only run the analysis or the Streamlit
  dashboard.

## What's inside

| File | What it does |
|---|---|
| `__init__.py` | Makes `Settings` and `get_settings` available as `from connectome_dashboard_core import get_settings`. |
| `settings.py` | Decides which folders to read (see [Where it looks](#where-it-looks)). The result, `Settings`, is passed to every other function. |
| `sections.py` | The list of the 18 dashboard sections: id, label, menu group, description, and which numbered result folders belong to each. |
| `artifacts.py` | Finds every result file (`.csv`, `.json`, `.md`, `.png`, `.svg`, `.pdf`) in the results folder, gives each a stable id, and reads one page of a table or the text of a JSON or Markdown file. Refuses symlinks and paths outside the results folder. |
| `connectomes.py` | Lists which participants have which of the nine matrix types, and reads one participant's 166 x 166 matrix: summary numbers, the full matrix, or a sorted list of its connections. |
| `section_analytics.py` | Per-section statistics: group summaries of one metric, per-region comparisons between two groups, regions that repeat across comparisons, and the demographics summary. The per-region comparison uses the Mann-Whitney U test (compares two groups by ranking all values instead of using the raw numbers) with Benjamini-Hochberg correction across regions (a way to adjust p-values when many tests are run at once, so that chance findings are not over-counted). |
| `advanced_analytics.py` | Readers for the machine-learning results (recorded, not re-trained), the network-level tables, region-level coupling rankings, and the findings catalogue. |
| `lr_sr.py` | LR/SR = long-range / short-range connections. Tract-length distributions (read from the `len_mean` matrices), the long-range/short-range cut-offs, the EDR exception tables (EDR = exponential distance rule: longer connections are usually weaker; an exception is a connection much stronger than its length predicts), the feature families used by the models, and the brain-network mapping. |
| `network_ranked.py` | One table that ranks every brain network within each measure by the size of the difference between two groups: CN vs AD by default; CN vs MCI and MCI vs AD are also available (`contrast` = `cn_ad`, `cn_mci`, `mci_ad`). |
| `exception_ml.py` | Reader for the exception-specificity results and their simple cognition models (all precomputed). |
| `statistics.py` | Shared statistical helpers and the group tests used by `lr_sr.py`. Holm correction is another way to adjust p-values for many tests (stricter than Benjamini-Hochberg). The paired rank-biserial effect size is a number from -1 to +1 that says how consistently one of two paired values (for example a participant's exceptional and ordinary connections) is larger than the other. The group order is always CN, MCI, AD. |
| `pipeline.py` | Reads imaging progress files. Those exist only on the imaging server. |
| `serialization.py` | Turns tables and numbers into plain JSON-friendly values. Numbers that are not finite become `None`. |

## Before you start

1. The project is installed and the venv is on (Install section of the
   [top-level README.md](../README.md)):

   ```bash
   source .venv/bin/activate
   ```

   The library needs only numpy, pandas and scipy, which `requirements.txt`
   already installs.

2. The analysis has been run, so there is something to read (see
   [connectome_analysis/README.md](../connectome_analysis/README.md)):

   ```bash
   python -m connectome_analysis.run_analysis --all --resume
   ```

3. Run your Python from the repository root, so that both
   `connectome_dashboard_core` and `sc_config` can be imported.

## How to use it

**Example 1. How many result files does each section have?**

Save this as a file, or paste it into `python` started from the repository
root:

```python
from connectome_dashboard_core import get_settings
from connectome_dashboard_core.sections import SECTIONS
from connectome_dashboard_core.artifacts import artifact_inventory, section_artifacts

s = get_settings()
print("analysis root:", s.analysis_root)
print("sections:", len(SECTIONS), "| result files:", len(artifact_inventory(s)))
for sec in SECTIONS[:3]:
    print(f"  {sec.id:<16} {len(section_artifacts(s, sec))} files")
```

What you should see (your path and counts will differ):

```
analysis root: <your results folder>
sections: 18 | result files: 843
  demographics     34 files
  novel-findings   476 files
  global-dti       27 files
```

**Example 2. Summary of one participant's matrix.**

```python
from connectome_dashboard_core import get_settings
from connectome_dashboard_core.connectomes import connectome_subjects, matrix_summary

s = get_settings()
subjects = connectome_subjects(s)            # a table: one row per participant
first = subjects["subject_id"].iloc[0]
out = matrix_summary(s, first, "fd_sum")
print(len(subjects), "subjects;", "shape", out["metadata"]["shape"],
      "density", round(out["summary"]["density"], 3))
```

What you should see (numbers depend on your data):

```
530 subjects; shape [166, 166] density 0.722
```

The example deliberately prints no participant identifier. ADNI identifiers
(such as `XXX_S_0001`) are restricted data: do not paste them into shared
notes, issues or documents.

## Where it looks

`get_settings()` builds the `Settings` once per Python process and keeps it.
For each folder, the first of these that is set wins:

| `Settings` field | 1st choice | 2nd choice (from `sc_config.py`) | Default |
|---|---|---|---|
| `analysis_root` | `CONNECTOME_ANALYSIS_ROOT` | `SC_ANALYSIS_ROOT` | `$SC_DERIV_ROOT/qc/analysis_cohort` |
| `connectomes_root` | `CONNECTOME_MATRICES_DIR` | `SC_CONNECTOMES_DIR` | `$SC_DERIV_ROOT/connectomes` |
| `project_root` | `CONNECTOME_PROJECT_ROOT` | `SC_PROJECT_ROOT` | the repository |
| `outputs_root` | `CONNECTOME_ENHANCED_ML_ROOT` | — | `<project_root>/outputs` (not read by any function yet) |
| `aal_labels` | — | — | `<project_root>/atlas/AAL/aal3_labels_166.csv`; region names shown next to matrix rows and in region tables, keyed on the matrix row (1–166). Do not point this at `AAL3_labels.csv`, which is keyed on the atlas value and names rows from 35 onward wrongly. |
| `aal_node_map` | — | `SC_ATLAS_ROOT` | `atlas/AAL/aal3_node_map_166.csv` (not read by any function yet) |
| `pipeline_status`, `pipeline_density`, `pipeline_manifest` | — | `SC_QC_ROOT` | files in `$SC_QC_ROOT/sc_matrix_qc/` (imaging server only) |
| `hcp379_*` | — | — | files under `<project_root>/research_audit/outputs/`, or `None` when absent (imaging server only) |
| `table_default_limit`, `table_max_limit` | — | — | 200 and 5000 rows per page |

`SC_DERIV_ROOT` itself defaults to `$SC_DATA_ROOT/derivatives`, and
`SC_DATA_ROOT` to `<repo>/data`. `SC_QC_ROOT` is the quality-control folder
(default `$SC_DERIV_ROOT/qc`). To see what the `SC_*` names resolve to:

```bash
python -c "import sc_config; print(sc_config.describe())"
```

All paths are turned into full paths (symlinks followed) when the settings are
built. Because the settings are kept for the life of the process, set any
environment variable **before** you start Python, or restart it after a
change.

## Inputs

Everything is read from `analysis_root` unless noted. The folder names are the
numbered section folders the analysis writes.

| Section id | Result folders it reads |
|---|---|
| `demographics` | `00_master`, `01_qc`, `02_demographics` |
| `novel-findings` | `03_global_microstructure_live`, `04_node_microstructure_live`, `06_global_graph_live`, `07_node_graph_live`, `10_edgewise_fd_sum`, `11_brain_age_live`, `12_length_delay`, `13_delay`, `15_advanced_structural`, `17_edr_exceptions`, `18_ml_diagnostics` (plus `20_findings` for the cards) |
| `global-dti` | `03_global_microstructure_live`, `03_global_rd` |
| `local-roi-dti` | `04_node_microstructure_live`, `04_local_rd` |
| `global-graph` | `06_global_graph_live`, `06_global_graph` |
| `sc-matrix-viewer` | `00_master`, `01_qc`, `03_global_microstructure_live`, `06_global_graph_live`, `07_node_graph_live`, `09_coupling_live`, `11_brain_age_live`, `12_length_delay`, `13_delay`, `17_edr_exceptions`, `18_ml_diagnostics` (plus the matrices in `connectomes_root`) |
| `node-metrics` | `07_node_graph_live`, `07_node_metrics`, `08_centrality` |
| `coupling` | `09_coupling_live`, `09_coupling` |
| `coupling-aal` | `09_coupling_live`, `04_node_microstructure_live`, `07_node_graph_live` |
| `brain-age` | `11_brain_age_live`, `11_brain_age` |
| `lr-sr` | `12_length_delay` |
| `lr-sr-analysis` | `12_length_delay`, `18_ml_diagnostics` (plus the `len_mean` matrices, `17_edr_exceptions`, `19_network_analysis`, `20_exception_specificity`) |
| `edr-exceptions` | `17_edr_exceptions` |
| `ml-diagnostics` | `18_ml_diagnostics` |
| `delay` | `13_delay` |
| `advanced` | `15_advanced_structural` |
| `network-analysis` | `19_network_analysis` |
| `functional-pending` | `16_functional_placeholder` |

`00_master/master_cohort.csv` is needed by almost everything. The matrices are
read from `connectomes_root` as
`SC_AAL166_<SUBJECT>_I<IMAGEID>_<type>.csv` (headerless 166 x 166,
comma-separated; `<type>` is one of `count`, `fd_sum`, `len_mean`,
`fa_mean`, `md_mean`, `rd_mean`, `ad_mean`, `count_invnodevol`,
`invlen_mean`). Groups are CN, MCI and AD; by the project's grouping rule
EMCI, LMCI and SMC participants are already counted as MCI in the analysis
tables.

## Outputs

Nothing is written to disk. Functions return Python values: dictionaries,
lists, and pandas tables (DataFrames). Some results are kept in memory so the
second request is fast; most of these memories are keyed on the files'
modification times, so a changed file is read again. The first request that
needs the tract lengths reads every `len_mean` matrix and takes a few seconds.

## Settings you can change

Only the environment variables in [Where it looks](#where-it-looks). There is
no configuration file. The section list lives in `sections.py`; if you add a
section there, the data server and the web app pick it up, but the dashboard
tests expect exactly 18.

## Tests

```bash
python -m pytest tests/dashboard_parity -q
```

**These 23 tests only pass on the project's own server. On any other machine
most of them fail, and that is expected; they are not a check for new
users.** They compare the library's output with frozen reference values kept
in `research_audit/outputs/`, a folder that is not part of the repository, at
a location fixed in `tests/dashboard_parity/conftest.py`. Some also read the
imaging server's progress files. They also check that no file outside the
results folder can be reached. To check your own setup, run Examples 1 and 2
above instead.

On the project's server the result is:

```
.......................                                                  [100%]
23 passed in 15.18s
```

## If something goes wrong

**`FileNotFoundError: [Errno 2] No such file or directory: '.../00_master/master_cohort.csv'`**
(or the same path without the `[Errno 2]` part)
The analysis has not been run, or `analysis_root` points at the wrong folder.
Print `get_settings().analysis_root` to see where it looks, then fix
`SC_ANALYSIS_ROOT` (or `CONNECTOME_ANALYSIS_ROOT`) and restart Python.

**`KeyError: 'unknown section: ...'`**
The section id is misspelt. The valid ids are
`[s.id for s in SECTIONS]` from `connectome_dashboard_core.sections`.

**`ModuleNotFoundError: No module named 'connectome_dashboard_core'`**
Python was started outside the repository root. `cd` to the repository root
and try again.

**`pipeline.py` reports a source as unavailable, or `pipeline_subjects` raises `FileNotFoundError`**
The imaging progress readers were called on a machine without the imaging
server's progress files under `research_audit/outputs/`. `release_status`
and `legacy_pipeline_status` list each missing source under `warnings`, and
`pipeline_subjects` raises
`FileNotFoundError` ("... source is not available on this machine") for
`source="hcp379"` or `"recovery-ledger"`; the data server turns that into a 404.
Expected off that machine; everything else in the library is unaffected.

**You changed an environment variable but the old folder is still used**
`get_settings()` is computed once per process. Restart Python (or the data
server).
