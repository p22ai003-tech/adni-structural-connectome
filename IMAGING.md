# Imaging pipeline

From a folder of T1 and DWI scans to connectome matrices.

This page is the whole route: what to install, how to arrange your data, and
the six commands that run it. Nothing here assumes you have used MRtrix or FSL
before.

---

## 1. What you need

**A Linux machine** (or macOS) with at least 8 cores, 32 GB of RAM, and disk
space. Budget roughly **2 GB per subject** for the run tree, plus the raw data.
The slowest step is distortion and motion correction; on CPU it takes a few
hours per subject, so a large cohort is a multi-day run. This is normal.

**Four imaging toolkits**, none of which are Python packages:

| tool | what it does here | version used |
|---|---|---|
| [MRtrix3](https://www.mrtrix.org/download/) | denoising, FOD, tractography, connectome assembly | 3.0.7 |
| [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index) | eddy-current and motion correction, BBR registration | 6.0.7 |
| [ANTs](https://github.com/ANTsX/ANTs/releases) | T1↔template registration | 2.6.5 |
| [dcm2niix](https://github.com/rordenlab/dcm2niix/releases) | DICOM → NIfTI, and reading acquisition parameters | any recent |

MRtrix3 is usually installed from source or conda; FSL has an official
installer script; ANTs and dcm2niix ship prebuilt binaries. Install them
however your site prefers — the pipeline only needs them on `PATH`, or named
by the environment variables below.

> **FSL licence.** FSL is free for academic use but **not for commercial use**.
> See its licence before using this pipeline commercially. The other three are
> permissively licensed.

**Python 3.12**, in two virtual environments — one for the workflow engine and
one for everything else. They are separate because Snakemake pins versions that
would otherwise fight with the analysis stack:

```bash
python3.12 -m venv .venv_connectome_workflow
.venv_connectome_workflow/bin/pip install -r scforge/workflow/requirements.lock.txt

python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 2. Tell the pipeline where everything is

```bash
cp env.sh.example env.sh      # edit the four toolchain paths
source env.sh
```

`env.sh` is gitignored because those paths belong to one machine. If your tools
are already on `PATH` you can skip it.

Then check the machine before touching any data:

```bash
python run_imaging.py doctor
```

It reports every binary with its version, every path with whether it exists,
and every Python package. **A missing export is the single most common failure
here** — an unset `MRTRIX_BIN` once surfaced as "unable to extract WM-FOD l=0
coefficient", which looks like a data problem and is not one. `doctor` exists
so you find out in two seconds instead.

## 3. Describe your study

```bash
python sc_study.py --init configs/study.yaml
```

Edit the file it writes. It is short:

```yaml
study:
  name: my-study
  layout: simple            # simple | bids | adni

input:
  source: local             # local | s3
  path: /data/mystudy/raw
  participants: participants.csv

output:
  data_root: /data/mystudy/work
```

`layout` describes how your folders are arranged, `input` says where the raw
scans are, and `output.data_root` is the one directory everything is written
under. **[docs/INPUT_LAYOUTS.md](docs/INPUT_LAYOUTS.md) shows exactly what each
layout looks like** — build one of those three and you are done thinking about
paths.

If your data is on S3, set `source: s3`, give the `uri` and a `stage_root`, and
it is synced down before the run. Anything else remote — an internal object
store, a URL, tape — you stage to a directory yourself and use `source: local`.
The pipeline's contract is a readable directory; it never reaches out to the
network mid-run.

## 4. Run it

Six commands. Each one writes a file the next one reads, so you can stop, look
at what it produced, and carry on.

```bash
STUDY=configs/study.yaml

# a. find the scans and pair each DWI with a T1     -> work/pairs.csv
python run_imaging.py --study $STUDY discover --out work/pairs.csv

# b. turn that into the 47-field input contract     -> configs/acquisition_manifest.csv
python sc_manifest_build.py --pairs work/pairs.csv --out configs/acquisition_manifest.csv

# c. read acquisition parameters out of the headers -> same file, filled in
python run_imaging.py probe

# d. check the contract; must print PASS
python run_imaging.py validate

# e. approve the subjects you want to process       -> <run>/contract/
python run_imaging.py approve --run-root /data/mystudy/work/derivatives/scforge_v2/run1 \
    --all --by "Your Name" --mode phase-b

# f. run
python run_imaging.py run --run-root /data/mystudy/work/derivatives/scforge_v2/run1 --cores 16
```

Steps **a–d** take minutes and are all reversible. Step **f** is the long one.

Finally, put the matrices where the analysis reads them:

```bash
python run_imaging.py publish --run-root /data/mystudy/work/derivatives/scforge_v2/run1
```

That writes `SC_AAL166_<subject>_I<image>_<metric>.csv` per subject, plus a
provenance sidecar naming the run that produced each one. From here, see
[connectome_analysis/README.md](connectome_analysis/README.md).

### What each step leaves behind

```
work/pairs.csv                     which DWI goes with which T1, and the gap
configs/acquisition_manifest.csv   the input contract: 47 fields per subject
<run>/contract/                    what was approved, by whom, and the source lock
<run>/subjects/<unit>/             everything computed, stage by stage
<run>/subjects/<unit>/07_connectome/matrices/   the nine matrices
<connectomes_dir>/SC_AAL166_*.csv  published for the analysis
```

## 5. Two gates you cannot skip

**Approval.** `run` will not start without an approval record. `approve` writes
one naming the subjects, who authorised them, the core and wall-clock ceilings,
and a SHA-256 of every raw file involved. This is not ceremony: it is what
makes a run reproducible and what stops a half-configured command from
consuming a week of compute on the wrong cohort.

**The run root.** Its path must contain `scforge_v2`, and it may not overlap a
production tree. A long run cannot scatter output somewhere unintended.

`run --dry-run` plans the whole DAG without either, so you can always see what
would happen first.

## 6. Minimum acquisition

The fibre-orientation step fits spherical harmonics to order 6. That needs 28
coefficients, and therefore **at least 28 unique diffusion directions** in the
shell being fitted, plus at least one b=0 volume. A 16-direction acquisition is
rejected with a clear message rather than quietly fitted at a lower order,
because a connectome built at a lower order is not comparable with one built at
order 6.

## 7. Acquisition parameters are read, never assumed

`probe` runs dcm2niix in sidecar-only mode — no image written, about half a
second per series — and records `phase_encoding_direction` and
`total_readout_time` **together with where each came from**, so a measured
value is always distinguishable from an estimated one. A header that gives the
axis but not the polarity is recorded as exactly that and never completed by
guesswork; those subjects fail their own unit rather than being processed on an
invented value.

This replaces an assumption. An older shell pipeline passed a fixed `-pe_dir
j-` to every subject. Across the 530-subject ADNI cohort this repository was
built on:

| phase encoding | subjects |
|---|---|
| `j`  | 436 |
| `j-` | 76 |
| `i`  | 18 |

Only 76 match the assumption, and 18 are on a different **axis** — all Siemens,
all ADNI-3. Total readout time spans 0.0333–0.0973 s and clusters by vendor, so
one value is wrong for most of a multi-site cohort.

Whether this changed the published connectomes is a separate question, and the
answer appears to be no: comparing the 18 axis-`i` subjects against ADNI-3
Siemens peers, no microstructure metric differs significantly. FA is lower in
that group (Cliff's δ −0.22 overall, −0.30 within CN, p ≈ 0.11), the direction
imperfect distortion correction would produce, but n = 18 cannot rule a small
effect in or out.

## 8. T1–DWI pairing is a policy, not a filter

`discover` pairs each DWI with the nearest T1 in time and **labels** the gap
rather than hiding it: `le_90_days`, `days_91_180`, `gt_180_days`, or `undated`
where the layout records no dates. No pair is dropped for timing; use
`--max-gap-days` if your analysis needs a hard limit.

It matters for interpretation. The atlas is carried from the T1 into DWI space,
so where the gap is large the anatomy defining the nodes predates the diffusion
data. In the ADNI cohort here the median gap is 753 days and 321 of 530 pairs
exceed 180 days — refusing distant pairs would have discarded most of the
cohort, and the strata are unbalanced across groups (42% of CN pairs within 90
days against 32% of MCI).

## 9. The stages

`scforge/workflow/rules/`, in the order the Snakefile includes them:

| stage | what it does |
|---|---|
| `00_manifest` | input contract, source content lock, checksums |
| `00_inputs` | DICOM → NIfTI, gradient tables, acquisition metadata |
| `01_dwi_preproc` | denoise, Gibbs unringing, eddy/motion, bias field |
| `02_anat` | T1 processing |
| `03_spatial_contract` | DWI ↔ T1 registration and its checks |
| `04_atlas` | AAL3 into DWI space |
| `05_5tt_fod` | tissue segmentation, response function, FOD |
| `06_preflight` | the gate before tractography |
| `07_tracks_connectome` | tractography, SIFT2, connectome assembly |
| `08_qc_publish` | QC and publication |

The `*_retry*` and `*_recovery*` files beside them are one-off extensions from
past incidents; the Snakefile does not include them.

Every parameter lives in `configs/connectome_v2.yaml`. Nothing is hard-coded in
a rule.

## 10. Stopping, resuming and failures

The workflow is a DAG, so **interrupting it loses only the jobs that were
running**. Re-issue the same `run` command and it picks up from the last
completed output. Add `--rerun-incomplete` if a job was killed mid-write.

A subject that fails does not stop the run. Each unit carries its own outcome,
and the ones that succeeded still publish.

If two runs are started against the same tree, Snakemake locks it and the
second refuses. That is working as intended; unlock with
`snakemake --unlock` only when you are certain nothing else is running.

## 11. Provenance

`workflow_source_manifest.tsv` pins a SHA-256 for every source file the
contract covers, so a rule cannot change without the contract noticing.
Regenerate it with `python scforge/workflow/lock_source_manifest.py` when you
change a covered file deliberately; `--check` verifies without writing.

Separately, `approve` builds a **source content lock**: one row per raw file
with its size, hash and inode identity. Every file is re-checked against it
before conversion, so a source that changed underneath the run fails closed
instead of quietly producing a different answer.

## 12. Manifests are not committed

Every manifest row names a participant and an absolute path to their imaging.
Manifests are generated on the machine that holds the data and stay there. This
repository ships code, configuration and the atlas — **no imaging data and no
participant tables**.
