# Structural-connectome v2 matrix data dictionary

**Contract:** `connectome-v2.0.1-canary-candidate`  
**Normative configuration:** [`configs/connectome_v2.yaml`](../configs/connectome_v2.yaml)  
**Status:** frozen candidate for a blinded canary; not approved for the full cohort  
**Primary atlas:** AAL3v1, 166 contiguous valid nodes  

## 1. Scope and non-negotiable rules

This document defines the nine subject-level matrices produced from one approved DTI/T1 pair, one validated AAL3 label image, one 10-million-streamline tractogram, and one SIFT2 weight vector. It supersedes all legacy/recovery naming conventions.

Every matrix must be 166 x 166, finite, symmetric, zero-diagonal, nonnegative within the stated numerical tolerance, and accompanied by the same node-table hash and recipe hash. Missing labels or matrices are failures; they are never repaired by zero-padding. Failed tensor samples are never clipped or replaced with zero.

The primary endpoint assignment is a direct 4-mm radial search against the contiguous AAL3 image. No dilated endpoint catchment is used. The same tractogram, atlas, radius, and assignment rule apply to all nine outputs. MRtrix documents radial endpoint assignment and the `tck2connectome` aggregation/scaling options in its [official command reference](https://mrtrix.readthedocs.io/en/latest/reference/commands/tck2connectome.html).

## 2. Common notation

- `i,j`: AAL3 node indices in `1..166`.
- `S_ij`: streamlines assigned to the unordered inter-node pair `(i,j)`.
- `N_ij = |S_ij|`: raw assigned streamline count.
- `w_s`: SIFT2 weighting factor for streamline `s`.
- `L_s`: arc length of streamline `s`, in mm.
- `q_s`: mean of a scalar tensor map sampled along streamline `s`.
- `V_i`: physical volume of node `i` in the actual connectome label image, in mm3.
- Empty edge: `S_ij` is empty. It is encoded as zero, not NA.

SIFT2 optimises a cross-section multiplier for every streamline; the weights are a proportional fibre-cross-section proxy, not a histological axon count. The exact command and `mu` are retained in provenance. See the [official SIFT2 reference](https://mrtrix.readthedocs.io/en/latest/reference/commands/tcksift2.html).

## 3. Required matrix definitions

| Matrix | Mathematical definition for `i != j` | Generation contract | Unit | Scientific interpretation |
|---|---|---|---|---|
| `count` | `N_ij` | `tck2connectome`; no `-tck_weights_in`; sum aggregation | streamlines | Raw algorithmic streamline count. It is not fibre density, axon count, or SIFT2 strength. |
| `fd_sum` | `sum(w_s for s in S_ij)` | `tck2connectome -tck_weights_in <weights>`; sum aggregation | proportional SIFT2 units | SIFT2-informed connection-strength proxy after locked response/intensity normalisation. |
| `count_invnodevol` | `2*N_ij/(V_i+V_j)` | Deterministic post-processing from `count` and measured physical node volumes | streamlines/mm3 | Raw count normalised by the mean physical volume of the two parcels. |
| `len_mean` | `mean(L_s for s in S_ij)` | `-scale_length -stat_edge mean`; no SIFT2 weights | mm | Arithmetic mean streamline arc length. |
| `invlen_mean` | `mean(1/L_s for s in S_ij)` | `-scale_invlength -stat_edge mean`; no SIFT2 weights | mm-1 | Arithmetic mean inverse arc length; it is not `1/len_mean`. |
| `fa_mean` | `mean(q_s for s in S_ij)`, where `q_s=mean(FA along s)` | `tcksample -stat_tck mean`, then `tck2connectome -scale_file ... -stat_edge mean`; no SIFT2 weights | dimensionless | Edge-average fractional anisotropy along reconstructed streamlines. |
| `md_mean` | Same two-level mean using MD | Same two-stage command pattern | mm2/s | Edge-average mean diffusivity. |
| `rd_mean` | Same two-level mean using RD | Same two-stage command pattern | mm2/s | Edge-average radial diffusivity. |
| `ad_mean` | Same two-level mean using axial diffusivity | Same two-stage command pattern | mm2/s | Edge-average axial diffusivity. Manuscript/dashboard label is **AxD**, never “AD”. |

The two-level tensor definition is important: voxel values are first averaged along each streamline and those per-streamline means are then averaged within the edge. MRtrix shows this exact FA pattern in the [`tck2connectome` examples](https://mrtrix.readthedocs.io/en/latest/reference/commands/tck2connectome.html), while `tcksample -stat_tck mean` is defined in the [`tcksample` reference](https://mrtrix.readthedocs.io/en/latest/reference/commands/tcksample.html).

### Node-volume implementation note

For v2, `count_invnodevol` is computed explicitly from physical volumes rather than trusting implicit image-grid assumptions. For a label image with voxel volume `v_voxel`, `V_i = number_of_voxels(label=i) * v_voxel`. The installed MRtrix 3.0.7 source implements `-scale_invnodevol` as `2/(nvox_i+nvox_j)`; that is numerically equivalent only on a 1-mm3 grid. The independent formula is therefore the normative implementation and must reproduce any command-derived candidate within tolerance.

## 4. Empty edges and support

Zero means “no assigned inter-node streamline” for all mean matrices; it is not a biological zero for FA, diffusivity, or length. Any analysis of `len_mean`, `invlen_mean`, FA, MD, RD, or AxD must use `count > 0` as the edge-support mask and explicitly model or report support differences.

For every mean matrix:

1. `count == 0` requires the matrix value to equal zero.
2. `count > 0` requires a finite sampled value.
3. A missing/non-finite connected-edge value fails the subject; it is not filled with zero.

`fd_sum == 0` with `count > 0` is possible if all assigned streamline weights are zero. It is retained and flagged, not silently changed.

## 5. Algebraic and physical invariants

Each matrix set must pass all of the following before spatial/human QC:

1. Exact shape `(166,166)`; no cropping or padding.
2. Finite, symmetric, zero-diagonal, and nonnegative values.
3. `count` is integer-valued within absolute tolerance `1e-6`.
4. `count` is not empirically all-close to `fd_sum` (`rtol=1e-10`, `atol=1e-12`).
5. The SIFT2 vector length equals the tractogram streamline count.
6. The upper-triangle sum of `count` equals the number of assigned, inter-node, non-self streamlines reconstructed from the assignment ledger.
7. `count_invnodevol` exactly reproduces `2*count/(V_i+V_j)` for all edges.
8. The six mean matrices have support consistent with `count` as specified above.
9. FA lies in `[0,1]`; no clipping.
10. MD, RD, and AxD lie in `[0,0.01]` mm2/s. Values above the hard ceiling or below zero fail and trigger image-level review.
11. Because the same tensor samples and linear averaging operators are used, `MD = (AxD + 2*RD)/3` within `1e-8` mm2/s; normally `AxD >= MD >= RD` within tolerance.
12. Every node `1..166` exists, has positive physical volume, and matches the immutable node-table hash.
13. No unexplained all-zero `count` row is permitted in the primary matrix.

MRtrix defines MD/ADC, FA, axial diffusivity, and radial diffusivity in the official [`tensor2metric` reference](https://mrtrix.readthedocs.io/en/latest/reference/commands/tensor2metric.html). With b-values recorded in s/mm2, diffusivity is reported in mm2/s.

## 6. Assignment and spatial contract

All matrices use:

```text
-assignment_radial_search 4 -symmetric -zero_diagonal
```

The count run additionally writes the complete assignment ledger. Assignment is accepted only when the atlas contains all 166 labels, all node volumes are positive, at least 85% of endpoints are assigned, all 166 nodes receive assignments, endpoint concentration passes the frozen threshold, and blinded spatial overlays pass. Density is reported but is not an inclusion or route-selection criterion.

The primary AAL3 image is created by composing the MNI-to-T1 nonlinear transform and T1-to-mean-b0 rigid transform, then resampling the labels once with label-preserving interpolation onto a 1-mm isotropic grid in DWI world coordinates. A low-resolution b0-grid label copy is for QC only.

## 7. Analysis-use boundaries

- `count` and `fd_sum` are distinct feature families and must never be duplicated under different labels.
- `count_invnodevol` is a size-normalised derivative of `count`, not an independent acquisition.
- `fd_sum` is the preferred SIFT2-informed strength proxy; its cross-subject use requires the locked common response/intensity-normalisation contract and protocol-aware inference.
- Raw `count` is retained for topology sensitivity and QC; it is not a direct biological fibre-count estimate.
- Tensor edge means are streamline-conditioned summaries and can be influenced by tractography/edge support. They do not replace voxelwise DTI inference.
- Zeros in mean matrices must not enter arithmetic averages as measured zero tissue values.
- The anatomical AAL3 systems are primary. Any Yeo mapping is a separately validated secondary mapping.

## 8. Required sidecar fields

One immutable JSON sidecar per subject matrix set records:

- subject, DTI and T1 IDs/UIDs/exact dates and pair interval;
- input/config/node-table/output SHA-256 values;
- exact commands, environment/container digest, and tool versions;
- transform files, route, quantitative spatial QC, and overlay paths;
- atlas label counts and physical volumes;
- ACT, FOD, response, tractography, random-seed, SIFT2 and assignment parameters;
- requested and actual streamline counts, assignment ledger counts, SIFT2 weight count and `mu`;
- all invariant results, tensor ranges, exclusion reasons, and human QC decision;
- execution timestamps and immutable recipe ID.

Only a complete nine-matrix set with a final `PASS` decision may enter the locked analysis cohort.

## 9. Canary decisions that remain open

The normative starting recipe is ACT, 10M streamlines, radial4, and BBR. The blinded canary must test ACT versus a uniform noACT diagnostic, 3M/5M/10M convergence, radial2 versus radial4, BBR versus rigid mutual information, and an independent random seed. These comparisons may not be selected using diagnosis separation or connectome density. Promoting any alternative requires SL-H04 approval and a new recipe ID before a full-cohort run.
