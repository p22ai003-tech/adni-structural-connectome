# Processing-route attribution, 2026-09-19

Which of the pipeline variants produced each of the 530 published SC_AAL166
matrices. Derived forensically on the machine that ran them, from MRtrix
command histories embedded in image headers (`hist530.json`), `tckinfo`
headers of the track files (`tckinfo530.json`), file timestamps and backups.
It cannot be re-derived anywhere else: several run roots no longer exist.

`final.csv` holds one row per subject; column `s4` is the fine-grained route.
`route_attribution_counts_by_cohort.csv` counts routes by group.

Families used for the sensitivity analysis (`connectome_analysis/route_sensitivity.py`):

| family | `s4` prefix | what it was | n |
|---|---|---|---|
| `new` | C, D, E, F | 3M/10M tracks, SyN full-head atlas, mostly radius 2 | 366 |
| `legacy_r4` | B | legacy 3M ACT tracks re-assigned at radius 4 | 111 |
| `legacy_step7` | A | pre-SOTA step 7, affine 170-label atlas, unweighted count | 53 |

The route is unevenly split by diagnosis (new-family share CN 65%, MCI 67%,
AD 87%), which is why it has to be tested as a confound.

The scripts here are the scratch tools that produced these files; they point
at the session scratchpad and are kept as evidence, not as runnable code.
