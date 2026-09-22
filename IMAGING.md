# Imaging pipeline

From a folder of T1 and DWI scans to connectome matrices.

This page is the whole route: what to install, how to arrange your data, and
the six commands that run it. Nothing here assumes you have used MRtrix or FSL
before.

---

## 1. What you need

**A Linux machine** (or macOS) with at least 8 cores, 32 GB of RAM, and disk
space. Budget roughly **15 GB per subject** for a complete run tree (see
[Running a large cohort](#11-running-a-large-cohort)), plus the raw data.
The slowest step is distortion and motion correction, and on CPU it is slow:
on this project's data it took **8 hours** for a 60-direction Siemens scan and
**13 hours** for a 55-volume GE scan with a larger matrix, one core each. Many
subjects can run at once, so a cohort is a multi-day run rather than a
multi-month one, but a single subject is most of a day. This is normal.

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

Where a study file and `env.sh` both say where data lives, the **study file
wins**, so a machine-wide setting cannot quietly redirect one study's output.
A command-line flag beats both.

Then check the machine before touching any data:

```bash
python run_imaging.py doctor
```

It reports every binary with its version, every path with whether it exists,
and every Python package.

Then record where the tools are on this machine:

```bash
python run_imaging.py lock-env
```

This writes `configs/environment.local.yaml`: the location, version and SHA-256
of every executable the workflow calls. It finds each toolkit from its usual
variable (`MRTRIX_BIN`, `FSLDIR`, `ANTSPATH`, `MRTRIX3TISSUE`, `C3D_BIN`), then
from `PATH`, and it tells you which versions differ from the machine the thesis
cohort was processed on. The recipe never names a path on any particular
machine; it refers to this file. Each approved run freezes a copy, so the run
records exactly which executables produced it. Re-run `lock-env` after
installing or upgrading a tool, and `lock-env --check` tells you whether the
file is still true. **A missing export is the single most common failure
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

Each command writes a file the next one reads, so you can stop, look
at what it produced, and carry on.

```bash
STUDY=configs/study.yaml
RUN=/data/mystudy/work/derivatives/scforge_v2/run1

# a. find the scans and pair each DWI with a T1     -> work/pairs.csv
python run_imaging.py --study $STUDY discover --out work/pairs.csv

# b. turn that into the 47-field input contract     -> configs/acquisition_manifest.csv
python sc_manifest_build.py --pairs work/pairs.csv --out configs/acquisition_manifest.csv

# c. read acquisition parameters out of the headers -> same file, filled in
python run_imaging.py probe

# d. check the contract; must print PASS
python run_imaging.py validate

# e. approve the subjects you want to process       -> <run>/contract/
python run_imaging.py approve --run-root $RUN --all --by "Your Name"

# f. run phase A: preprocessing and one response function per subject
python run_imaging.py run --run-root $RUN --cores 16
```

Steps **a–d** take minutes and are all reversible. Step **f** is most of a day
per subject, almost all of it in eddy-current and motion correction.

Phase A stops there on purpose. What follows alternates between a person and
the machine; `run` always works out which phase comes next, and says what a
person has to do first when it is not its turn:

```bash
# g. pool the per-subject responses into the one the FODs are fitted against
python run_imaging.py freeze --run-root $RUN --by "Your Name"

# h. pre-tractography: FODs, tissue segmentation, atlas registration, and
#    three overlay images per subject for you to look at
python run_imaging.py run --run-root $RUN --cores 16

# i. look at <run>/subjects/*/06_preflight/review_b0_vs_{t1,5tt,aal3}.png
#    and record your verdict (--status fail --units ... for the ones that are off)
python run_imaging.py review --run-root $RUN --reviewer "Your Name"

# j. authorise tractography for the subjects that passed
python run_imaging.py continue --run-root $RUN --by "Your Name"

# k. phase B: tractography, SIFT2, the nine matrices
python run_imaging.py run --run-root $RUN --cores 16
```

Each person-step writes a signed decision under `<run>/contract/`, and every
later phase re-checks it along with the hashes of the recipe, the workflow
source and the tools frozen at approval. A decision is never overwritten: to
change one, start a new run root.

Finally, put the matrices where the analysis reads them:

```bash
python run_imaging.py publish --run-root $RUN
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
one naming the subjects, who authorised them, the core, wall-clock and storage
ceilings (`--max-cores`, `--wall-clock-hours`, `--storage-gb`), how many
subjects must yield a usable response function for the calibration to stand
(`--min-valid-calibration`, default half the batch), and a SHA-256 of every raw
file involved. It also freezes this machine's tool locations into the run, so a
later `lock-env` cannot change a run already under way. This is not ceremony: it is what
makes a run reproducible and what stops a half-configured command from
consuming a week of compute on the wrong cohort.

**The run root.** Its path must contain `scforge_v2`, and it may not overlap a
production tree. A long run cannot scatter output somewhere unintended.

`run --dry-run` prints which phase is next and the exact launcher command,
without starting it.

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

## 11. Running a large cohort

Measured on this project's data, a complete subject costs about **15 GB**: the
10-million-streamline `.tck` alone is 5–12 GB (median 7), preprocessing another
4 GB, and the model and registration stages about 3.5 GB between them. For 500
subjects that is roughly **8 TB**, which is the binding constraint long before
CPU is.

So a full cohort is not one run that you let finish. It is batches, each of
which is swept before the next begins.

Three things make that work:

**Sweep as you go.** Once a subject's matrices are published, its largest
intermediates are no longer needed:

```bash
python run_imaging.py publish --run-root <run>
python run_imaging.py sweep   --run-root <run> --dry-run   # see what it would free
python run_imaging.py sweep   --run-root <run>
```

There are two levels:

| `--level` | removes | leaves a subject at |
|---|---|---|
| `intermediates` (default) | streamlines, denoised and unringed volumes, converted raw | about 7 GB |
| `all-regenerable` | the above, plus the preprocessed DWI, the FODs, the tensor, the 5TT segmentation, the template warps and eddy's diagnostic volumes | under 1 GB |

`intermediates` keeps the preprocessed DWI, so a later stage can be rerun
without paying for eddy again — the right choice while a cohort is still being
worked on. `all-regenerable` keeps only what the analysis and an audit need,
and is what makes a full cohort fit.

It will not touch a unit unless the provenance sidecar in the analysis
directory names *this* run: matrices for the same subject from an earlier run
are not evidence that this one finished. `--keep-tracks` keeps the `.tck` files
if you want to re-derive other metrics from them later.

Everything it removes is regenerable, and since every stochastic step is
seeded, regenerable to the same answer.

**Give the run cores, not one subject cores.** Distortion and motion
correction is the wall clock, and FSL's `eddy_cpu` is single-threaded by
default — it takes `--nthr` and ignores `OMP_NUM_THREADS`. A cohort therefore
goes faster by running many subjects at once than by giving one subject more
cores, and `--cores` is what controls that. Reserve roughly 3.5 GB of RAM per
concurrent subject.

`dwi_preprocessing.eddy.scheduler_threads` is what one job reserves; the
default of 4 lets `--cores 32` work on eight subjects at a time.
`dwi_preprocessing.eddy.threads` is `eddy`'s own `--nthr`, left at its default
of 1 because changing it changes the order of floating-point reductions and so,
in principle, the result.

**Run in batches.** `approve` takes `--units`, `--units-file` or `--first N`, so
a cohort goes through in groups, each with its own run root. Each batch is
independently approvable, resumable and sweepable. Size a batch by dividing
your free space by 15 GB and leaving headroom — on a 1.5 TB volume, batches of
about 50.

**Watch the ceilings.** The approval records a wall-clock and a storage stop,
and the run refuses to start if the free space is already below the storage
ceiling. Set them with `--wall-clock-hours` and `--storage-gb`.

## 12. Provenance

`workflow_source_manifest.tsv` pins a SHA-256 for every source file the
contract covers, so a rule cannot change without the contract noticing.
Regenerate it with `python scforge/workflow/lock_source_manifest.py` when you
change a covered file deliberately; `--check` verifies without writing.

Separately, `approve` builds a **source content lock**: one row per raw file
with its size, hash and inode identity. Every file is re-checked against it
before conversion, so a source that changed underneath the run fails closed
instead of quietly producing a different answer.

## 13. Manifests are not committed

Every manifest row names a participant and an absolute path to their imaging.
Manifests are generated on the machine that holds the data and stay there. This
repository ships code, configuration and the atlas — **no imaging data and no
participant tables**.
