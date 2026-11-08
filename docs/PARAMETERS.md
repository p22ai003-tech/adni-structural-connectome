# Parameters — `run_sc_route_sota.py` (recovery engine) + connectome build

## Env flags
| Flag | Default | Meaning |
|---|---|---|
| `REBUILD_FOD` | 1 | Rebuild FOD from eddy (dwi2response+dwi2fod). **Set 0 to use the healthy cached FOD** — rebuilding a healthy FOD can collapse it. Required =1 to enable fresh reg (generates `b0_post`). |
| `FRESH_T12B0` | 1 | Compute a fresh rigid FLIRT T1→b0 (normmi, ±40°) when cached BBR is poor; runner keeps `max(cached,fresh)`. Needs `b0_post` (from REBUILD_FOD=1). |
| `REREG_THRESH` | 0.5 | Attempt fresh reg only if cached connectome density < this. Set 0.99 to always attempt. |
| `FORCE_NOACT` | 1 | Route all subjects through FOD-based **noACT** dynamic seeding. ACT is broken on anisotropic ADNI data (median 0.19 ACT vs 0.62 noACT). |
| `TCKGEN_SELECT` | 10000000 | Streamlines to select (10M recovery; production was 3M). |
| `TCKGEN_CAP_S` | 6000 | tckgen wall-clock hard cap (s). |
| `RESUME_MIN_DENSITY` | 0.6 | Skip a subject if it already has status=ok & density ≥ this. |
| `DONOR_RESPONSE` | — | Group-average response fallback (proved counterproductive — over-normalizes good FODs). |

## CLI
`--subjects … --run-root … --mode full_10m --workers N --threads T --radial {2|4} --reg-mode {brain|fullhead}`

## Connectome weights (`WEIGHT_SPECS`, structural 4 the recovery writes)
`count` · `fd_sum` (-stat_edge sum) · `count_invnodevol` (-scale_invnodevol) · `len_mean` (-stat_edge mean -scale_length).
Production also builds diffusivity 4 (`fa/md/ad/rd_mean`, via `tcksample` tsf + `-scale_file`) + `invlen_mean` = 9 total.

## Assignment radius (`-assignment_radial_search`)
- **Production = 4 mm**, **recovery = 2 mm** (mixed dataset — see `DECISIONS.md`).
- Density is **monotonic non-decreasing in R** (larger radius only adds endpoint→node assignments, never
  removes). So re-assigning *up* (2→4) never regresses; *down* (4→2) can drop near-0.6 subjects.

## tckgen (recovery/noACT)
`-select 10000000 -seeds 200000000 -cutoff 0.06 -minlength 10 -maxlength 250 -seed_dynamic <wmfod> -mask <mask_fod>` (no `-act`).

## Thread/scaling notes
96-thread tckgen = 4.63× of 16 (near-linear). Box is **disk-bound (~128 MB/s)** → loading 10M tracks (~5–8 GB)
≈ 10 min each; mass re-assignment is impractical — prefer copy-promote of existing connectomes.
Numexpr cap: set `NUMEXPR_MAX_THREADS` when running at 96 threads.

## Analysis settings (`sc_cohort_settings.py`)
Environment variables read by the analysis pipeline and the revision scripts. The defaults reproduce the published cohort byte for byte.
| Variable | Default | Meaning |
|---|---|---|
| `SC_DENSITY_FLOOR` | 0.6 | Minimum count density of a connectome to enter the master cohort; `none` or 0 switches the floor off. |
| `SC_SMC` | `mci` | `mci` counts subjective memory concern as MCI; `hold_out` keeps those participants out of every group. |
| `SC_N_NODES` | 166 | Nodes analysed: the first n of the AAL3 nodes. |
| `SC_LABELS`, `SC_LABEL_SEED` | unset | `shuffled` permutes diagnosis and clinical scores between participants with a recorded seed, for debugging; `real` is the flag that allows real labels under a chain root or on connectomes with a recorded recipe hash. |
| `IIT_PRIMARY_SAMPLE` | `adni34` | Primary sample of the revision analyses: `adni34`, `adni3` or `pooled`. |
| `IIT_LENGTH_MEASURE` | `per_person` | Edge length of the revision analyses: `per_person`, `mni_distance` or `control_median_act`. |
| `IIT_ROOT` | unset | Chain root: everything is read from and written under it (`revision/tools/iit_chain.py`). |
