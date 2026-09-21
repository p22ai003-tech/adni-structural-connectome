# Input layouts

The pipeline needs one thing from you: a readable directory of raw scans, in
one of three arrangements, plus a small table saying who the subjects are.
Everything after that is derived from the files themselves.

Pick the layout that matches the data you already have. If you are starting
from nothing, use **simple**.

---

## simple — the template

```
<raw>/
├── dwi/
│   ├── sub-001/
│   │   └── ses-01/            (optional; omit if there is one session)
│   │       ├── 0001.dcm       a DICOM series, OR
│   │       ├── 0002.dcm
│   │       └── ...
│   └── sub-002/
│       └── ses-01/
│           ├── dwi.nii.gz     a NIfTI bundle: image + all three sidecars
│           ├── dwi.bval
│           ├── dwi.bvec
│           └── dwi.json
├── anat/
│   ├── sub-001/
│   │   └── ses-01/
│   │       └── t1.nii.gz      a single NIfTI, OR a folder of .dcm
│   └── sub-002/
│       └── ses-01/
│           └── t1.nii.gz
└── participants.csv
```

`participants.csv` needs a `subject_id` column and whichever of `diagnosis`,
`age` and `sex` you have. `group`, `dx`, `participant_id` and `gender` are
accepted as aliases.

```csv
subject_id,diagnosis,age,sex
sub-001,CN,71,F
sub-002,AD,78,M
```

## bids

A BIDS raw dataset works as-is:

```
<raw>/
├── sub-001/
│   └── ses-01/
│       ├── dwi/sub-001_ses-01_dwi.nii.gz  (+ .bval, .bvec, .json)
│       └── anat/sub-001_ses-01_T1w.nii.gz
└── participants.tsv   → save a CSV copy, or point input.participants at your own
```

## adni

The layout an ADNI IDA download produces, kept so an ADNI collection can be
used without rearranging it:

```
<raw>/
├── dti/<SUBJECT>/<PROTOCOL>/<YYYY-MM-DD_hh_mm_ss.0>/I<IMAGEID>/*.dcm
└── mri/<SUBJECT>/<PROTOCOL>/<YYYY-MM-DD_hh_mm_ss.0>/I<IMAGEID>/*.dcm
```

---

## What each modality may be

|          | DICOM series                  | NIfTI                                        |
|----------|-------------------------------|----------------------------------------------|
| **DWI**  | a folder of `*.dcm`           | `*.nii[.gz]` **with** `.bval`, `.bvec`, `.json` |
| **T1**   | a folder of `*.dcm`           | a single `*.nii[.gz]`                          |

A DWI given as NIfTI must carry all three sidecars. The gradient table cannot
be recovered from the image, and `PhaseEncodingDirection` and
`TotalReadoutTime` from the JSON are what distortion correction is driven by.
If you converted with `dcm2niix`, you already have them.

## Minimum acquisition

The fibre-orientation step fits spherical harmonics to order 6, which needs 28
coefficients and therefore at least **28 unique diffusion directions** in the
shell being fitted. A 16-direction acquisition is rejected with a clear message
rather than silently fitted at a lower order — the resulting connectome would
not be comparable with one built at order 6.

You also need at least one b=0 volume, and the DWI and T1 must be from the same
subject.

## Dates are optional

Sessions named `ses-01` carry no acquisition date, and that is fine. Where both
scans have a date the pipeline records the gap between them and labels the pair
(`within_90d`, `within_180d`, `beyond_180d`); where they do not, the stratum is
`undated` and the gap is left empty. Nothing is invented, and no pair is
dropped for timing.

## Where the data lives

The directory can be local, or anything mounted onto the machine — NFS, SMB, an
external disk. If it is on S3, give the URI and a staging directory and the
data is synced down before the run:

```yaml
input:
  source: s3
  uri: s3://your-bucket/your-prefix
  stage_root: /data/staging
```

Any other remote store is handled by staging it to a directory yourself and
using `source: local`. The pipeline's contract is a readable directory; it
never reaches out to the network during a run.

## Building the manifest

```bash
python sc_study.py --init configs/study.yaml    # then edit it
python run_imaging.py --study configs/study.yaml doctor
python run_imaging.py --study configs/study.yaml discover --out work/pairs.csv
python sc_manifest_build.py --pairs work/pairs.csv --out configs/acquisition_manifest.csv
python run_imaging.py probe                      # fills PE direction and readout time
python run_imaging.py validate                   # must print PASS before you run
```
