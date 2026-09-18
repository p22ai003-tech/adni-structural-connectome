# Imaging pipeline

From a folder of T1 and DWI images to connectome matrices.

```bash
source env.sh                                    # toolchain (copy from env.sh.example)
python run_imaging.py doctor                     # can this machine run it?
python run_imaging.py discover --raw-root /data/Images
python run_imaging.py probe                      # acquisition parameters from the DICOMs
python run_imaging.py validate                   # check the input contract
python run_imaging.py run --dry-run              # plan
python run_imaging.py run --cores 16             # execute
```

The first four steps are quick. The last is hours per cohort, which is why
everything knowable in advance is checked before it starts.

## Input layout

The standard ADNI download tree:

```
<raw>/dti/<SUBJECT>/<PROTOCOL>/<YYYY-MM-DD_HH_MM_SS.0>/<IMAGE_ID>/*.dcm
<raw>/mri/<SUBJECT>/<PROTOCOL>/<YYYY-MM-DD_HH_MM_SS.0>/<IMAGE_ID>/*.dcm
```

## The manifest is the only input contract

`discover` writes it, everything downstream reads it, and nothing downstream
globs the raw tree. It is validated against
`scforge/workflow/schemas/acquisition_manifest_v2.schema.json` (47 fields).

Manifests are **not** committed: every row names an ADNI participant and the
absolute path to their imaging. They are generated on the machine holding the
data.

### T1–DWI pairing is a policy

`discover` pairs each DWI with the nearest T1 in time and labels the gap rather
than hiding it: `le_90_days`, `days_91_180`, `gt_180_days`. In this cohort the
median gap is **753 days** and 321 of 530 pairs exceed 180 days, so refusing
distant pairs would discard most of the cohort. `--max-gap-days` imposes a hard
limit when an analysis needs one.

This matters for interpretation: the AAL3 parcellation is carried from the T1
into DWI space, so for most subjects the anatomy defining the nodes predates the
diffusion data by years, in a cohort where atrophy is the process being studied.
The strata are also unbalanced across groups — 42% of CN pairs are within 90
days against 32% of MCI.

### Acquisition parameters are read, never assumed

`probe` runs dcm2niix in sidecar-only mode (no image written, about half a
second per series) and records `phase_encoding_direction` and
`total_readout_time` together with where each came from, so a measured value is
always distinguishable from an estimated one. A series whose header gives the
axis but no polarity is recorded as exactly that, never completed by guesswork.

This replaces an assumption. The older shell pipeline passed a fixed
`-pe_dir j-` to every subject. Across all 530:

| phase encoding | subjects |
|---|---|
| `j`  | 436 |
| `j-` | 76 |
| `i`  | 18 |

Only 76 of 530 match the assumption, and 18 are on a different **axis**
entirely — all Siemens, all ADNI-3. Total readout time spans 0.0333–0.0973 s,
clustering by vendor, so a single value is wrong for most of the cohort.

Whether this changed the published connectomes is a separate question, and the
answer appears to be no: comparing the 18 axis-`i` subjects against ADNI-3
Siemens peers, no microstructure metric differs significantly. FA is lower in
the axis-`i` group (Cliff's δ −0.22 overall, −0.30 within CN, p ≈ 0.11), which
is the direction imperfect distortion correction would produce, but n = 18 is
too small to rule a small effect in or out.

## Stages

`scforge/workflow/rules/`, in the order the Snakefile includes them:

| stage | what it does |
|---|---|
| `00_manifest` | input contract, checksums |
| `00_inputs` | DICOM → NIfTI, gradient tables |
| `01_dwi_preproc` | denoise, Gibbs unringing, eddy/motion, bias field |
| `02_anat` | T1 processing |
| `03_spatial_contract` | DWI ↔ T1 registration contract |
| `04_atlas` | AAL3 into DWI space |
| `05_5tt_fod` | tissue segmentation, response, FOD |
| `06_preflight` | the gate before tractography |
| `07_tracks_connectome` | tractography, SIFT2, connectome assembly |
| `08_qc_publish` | QC and publication |

The `*_retry*` and `*_recovery*` variants in that directory are one-off
extensions from past incidents; the Snakefile does not include them.

Parameters live in `configs/connectome_v2.yaml`.

## Toolchain

MRtrix3, FSL, ANTs and dcm2niix. They are usually installed but not exported,
which is the failure this pipeline used to hit: the v2 workflow reported
"unable to extract WM-FOD l=0 coefficient", which was `MRTRIX_BIN` being unset,
not a problem with the FOD. `doctor` reports all eleven binaries with versions
before anything runs.

`workflow_source_manifest.tsv` pins a SHA-256 for every source file the contract
covers. Regenerate it with `python scforge/workflow/lock_source_manifest.py`
when a covered file changes deliberately; `--check` verifies without writing.

## Output

`<run-root>/` holds per-subject derivatives and the published connectome
matrices. The analysis pipeline reads those; see `connectome_analysis/README.md`.
