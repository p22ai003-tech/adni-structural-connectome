# Closed-world analysis design v1.1 — SL-D01 / pre-SL-H03-C1

**Project:** structural-connectome CN/MCI/AD analysis  
**Decision:** SL-D01, locally available images only  
**Version date:** 2026-07-18 UTC  
**Status:** pre-H03 design recommendation; **not yet approved**. This is not the frozen statistical analysis plan (SAP). SL-H03-C1 can approve the cohort/estimand design only; the outcome/QC/exchangeability implementation remains to be frozen later and no corrected biological group result has been opened here.

## 1. Decision and claim boundary

This design accepts the dataset as a closed world. The project will attempt the corrected pipeline on every one of the 530 locally available DTI–T1 pairs. It will not wait for, request, or substitute external images. Acquisition interval and usable image quality are therefore properties of the study rather than acquisition-screen exclusions.

The immutable input denominator is **530 pairs**. That does not make every derived value a valid biological measurement. Low-quality but mathematically and technically valid outputs remain eligible, with quantitative QC adjustment and sensitivity analyses. A failed registration, invalid gradient/tensor result, non-finite or malformed matrix, physically impossible value, missing required provenance, or other hard validity failure is retained in the attrition ledger and represented as missing for the affected outcome; it is never converted to zero, clipped into range, imputed as a connectome, or promoted as a measurement.

The study-level estimand is an **available-data, cross-sectional association** between diagnosis and corrected structural-connectome measurements in these local subjects, conditional on measured demographic, timing, acquisition, processing, and QC variables. It is not a causal effect of diagnosis, a same-visit multimodal biomarker estimand, or an estimate for all ADNI participants. The full-cohort result is exploratory because this cohort and its provisional results have already been examined.

## 2. Fixed evidence and analysis populations

The following counts are fixed by the checksum-bound 530-row [`available_data_pair_manifest_v2.csv`](outputs/available_data_pair_manifest_v2.csv), diagnosis reconciliation, and the outcome-blind SL-P0-10 [`cohort_v2_population_summary.csv`](outputs/cohort_v2_population_summary.csv). The SL-P0-10 validation status is `PASS` with release state `FEASIBILITY_COMPLETE_NO_OUTCOMES_OPENED`.

| Local pair stratum | CN | MCI | AD | SMC | Total |
|---|---:|---:|---:|---:|---:|
| All locally available pairs | 251 | 186 | 78 | 15 | 530 |
| Exact interval <=90 days | 106 | 60 | 28 | 0 | 194 |
| Exact interval 91–180 days | 0 | 0 | 3 | 0 | 3 |
| Exact interval <=180 days | 106 | 60 | 31 | 0 | 197 |
| Exact interval >180 days | 145 | 126 | 47 | 15 | 333 |

The diagnosis-comparison population is the 515 subjects locally labeled CN, MCI, or AD: CN 251, MCI 186, and AD 78. The 15 SMC subjects remain in processing, QC, and attrition accounting but are not relabeled as MCI. They form a descriptive-only secondary group because n=15 and every SMC pair is more than 180 days apart. No CN/MCI/AD primary contrast uses SMC.

The <=180-day analysis is not a meaningfully independent graded-window check: relative to <=90 days, it adds only three AD subjects and no CN or MCI subjects. Both windows will still be reported because they were requested and fixed before the corrected results, but agreement between them must not be advertised as two independent replications. The continuous timing analysis is essential.

The following acquisition-overlap strata are also fixed as sensitivity populations, subject to regenerated outcome-specific QC:

- Siemens 3 T, 54-direction DTI: n=260 (CN 145, MCI 93, AD 22).
- <=90-day Siemens 3 T, 54-direction subset: n=94 (CN 57, MCI 27, AD 10); all 94 have a zero-day gap.
- <=90-day Siemens 3 T, 54-direction, multi-group-site subset: n=75 (CN 42, MCI 26, AD 7; 11 sites); all 75 have a zero-day gap.
- Original-T1 source class: n=305 (CN 157, MCI 105, AD 43).
- Processed-T1 source class: n=210 (CN 94, MCI 81, AD 35) in the 515-person comparison population. There are 225 processed-T1 inputs in the 530-pair processing denominator after adding 15 SMC.
- Same-ADNI-phase DTI/T1 sensitivity: n=333 (CN 157, MCI 101, AD 75).

The timing/protocol overlap subsets are low-power stress tests, especially for AD. Failure to reach p<.05 in them is not evidence of absence; a sign reversal, gross magnitude change, or lack of estimability is evidence that the full-cohort association is not robustly identified.

## 3. Identification problem to preserve, not hide

Diagnosis is strongly associated with acquisition in the 515-subject CN/MCI/AD set:

- ADNI phase Cramér's V=0.410; p=3.46e-37.
- Manufacturer Cramér's V=0.392; p=1.19e-31.
- Gradient-direction count Cramér's V=0.423; p=2.94e-35.
- Protocol key Cramér's V=0.421; p=1.15e-32.
- Site Cramér's V=0.411; p=4.28e-18.

The corrected diagnosis-by-phase counts are highly imbalanced: AD is 33/44/1, CN 3/242/6, and MCI 0/180/6 across ADNI 2/3/4. Measured common support is correspondingly shallow:

- 20/51 sites contain CN, MCI, and AD; only two sites contain at least five from every group and one contains at least ten.
- 18/61 site-by-protocol cells contain all three groups; none contains at least five from every group.
- In the exact timing-by-site-by-protocol-by-T1-source partition, 13/131 cells contain all three groups, accounting for CN 50, MCI 44, and AD 18; none contains at least five from every group.
- CN-MCI has 41 shared site-by-protocol cells and measured support for 245 CN/184 MCI, including eight cells with at least five per group.
- MCI-AD has 18 shared cells and measured support for 82 MCI/34 AD, with no cell reaching five per group.
- CN-AD has 21 shared cells and measured support for 129 CN/42 AD, with no cell reaching five per group.

A model can adjust measured differences where groups overlap; it cannot manufacture comparators in acquisition cells containing only one diagnosis. The full-cohort aggregate CN-MCI contrast is estimable with measured common support. Aggregate MCI-AD and CN-AD contrasts are estimable only with **limited cell depth** and require explicit extrapolation/support diagnostics. Cell-specific rows marked `NOT_ESTIMABLE_*` in [`cohort_v2_estimability.csv`](outputs/cohort_v2_estimability.csv) remain non-estimable. They are not rescued by deleting nuisance terms until a coefficient or small p-value appears.

Scanner manufacturer, field strength, gradient directions, phase, and `protocol_key` are partially redundant. In fact, the ADNI 2 indicator is **exactly identical** to the `GE MEDICAL SYSTEMS|3T|41dir` protocol indicator in this dataset (36 subjects: CN 3, MCI 0, AD 33). The outcome-blind draft-sized phase-plus-protocol proxy therefore has 27 columns but rank 26; adding T1 class gives 28 columns but rank 27. The saturated site-fixed stress proxy has 78 columns but rank 74. These exact rank failures prohibit the draft phase-plus-protocol/site-fixed specification.

The primary fixed acquisition representation is therefore `protocol_key` plus T1 source class (`Original` versus `Processed`), without a simultaneous phase fixed effect. In the outcome-blind linear feasibility proxy, this representation with diagnosis, age, sex, and log-gap has 16 columns, rank 16, residual df 499, and column-normalized condition number 5.58. The corresponding protocol-only proxy has 15 columns, rank 15, residual df 500, and condition number 3.86. These diagnostics support a parsimonious nuisance representation; they do **not** prove that the future outcome-specific spline/mixed model will be full rank.

Site/scanner partial pooling is retained only if the outcome-specific model converges and its fixed design remains identifiable; the random-effect structure may not be made more complex after seeing diagnosis results. Phase, manufacturer, field strength, gradient directions, same-phase status, and scanner model enter declared support/alternative specifications rather than the same saturated primary fixed design. Processing recipe, assignment radius, and tractography route must be constant in the corrected primary run; any unavoidable variation becomes an explicit fixed technical factor and a stratified sensitivity.

## 4. Outcome and hypothesis tiers

### 4.1 No independent confirmatory tier

No result from these same 530 subjects is an independent confirmation of a candidate selected from their earlier dashboard outputs. Prespecification before the corrected rerun prevents further outcome-driven flexibility, but it does not erase prior exposure. The terms below therefore distinguish reporting priority; they do not convert the study into an external validation.

### 4.2 Prespecified primary analysis tier

The primary outcome family is deliberately small and FA-centred:

1. corrected whole-brain participant-level edge-supported mean FA;
2. the three diagnosis contrasts CN–MCI, MCI–AD, and CN–AD.

For participant `s`, the primary scalar is fixed as:

```text
WB_FA_s = mean(fa_mean_s[i,j] for i < j and count_s[i,j] > 0)
```

Thus, only the unique undirected upper triangle is used, the diagonal is excluded, every supported edge receives equal weight, and unsupported zero-coded edges do not enter the mean. A `count > 0` edge with a missing/non-finite or out-of-range FA value fails that participant's FA outcome rather than being dropped edgewise. The atlas node table, matrix symmetry/diagonal tolerances, `[0,1]` FA range, and count-support invariant remain mandatory.

Prespecified aggregation sensitivities are (i) streamline-count-weighted mean FA across the same supported upper-triangle edges, (ii) SIFT2-`fd_sum`-weighted mean FA when the complete weights pass their own validity contract, and (iii) the unweighted median over the same supported edges. Edge density/support is reported and tested as a technical sensitivity; no density threshold or common-edge mask is selected after diagnosis results. The unweighted supported-edge mean above remains primary regardless of which sensitivity is more significant.

The primary population is the full available CN/MCI/AD analytic set after outcome-specific minimum-validity gating, with 515 as the pre-QC ceiling and 530 as the input/attrition denominator. Strong family-wise error control is across the three contrasts, using a max-T procedure when valid exchangeability is available and Holm correction otherwise. Report adjusted marginal mean differences in native FA units, standardized differences, 95% confidence intervals, adjusted p-values, and the exact analytic n by diagnosis. A p-value alone cannot qualify a result.

### 4.3 Locked internal-retest tier

The following previously viewed candidates are carried forward as one declared internal-retest family, not as new discoveries:

- whole-brain MD, RD, and AxD;
- mapped Limbic FA, MD, RD, and AxD;
- mapped DMN FA;
- anatomical Limbic versions of the same microstructure summaries when the atlas definition is fixed.

All three diagnosis contrasts are estimated. Multiplicity is controlled across every outcome-contrast combination in this tier; the exact family size is frozen after removal of literal duplicate definitions and before corrected group values are read. The approximate AAL3-to-functional-network crosswalk cannot by itself support a network-specific headline. A mapped Limbic or DMN result must be reported as mapping-dependent unless it agrees in direction and material magnitude with the frozen anatomical definition and a declared alternate-map sensitivity.

### 4.4 Secondary/exploratory tier

Graph topology, topology–microstructure coupling, nodewise and edgewise discovery, NBS/TFNBS, EDR/LR-SR summaries, brain age, cognition, clinical classification, indirect-association/path models, and diagnostic machine learning are secondary. They have separately declared correction families and appear in the supplement unless their own validation requirements are met. No functional MRI measure is present, so graph–DTI associations are called **topology–microstructure coupling**, never structure–function coupling. Cross-sectional paths are associations, never mediation, mechanism, or compensation.

Visual/Limbic nodal efficiency and frontal coupling are specifically secondary retests because the existing effects weakened, reversed, or became highly incomplete under acquisition restrictions. Existing NBS, brain-age, clinical-prediction, and ML results are null/weak baselines, not positive priors.

## 5. Primary statistical model

For each continuous outcome, let `G` be CN/MCI/AD diagnosis at the DTI acquisition, `d` the absolute DTI–T1 interval in days, `T` the T1 source class (`Original` or `Processed`), and `Q` the frozen quantitative QC vector. The proposed full-cohort model is:

```text
Y = diagnosis
    + f(age_at_DTI)
    + sex
    + f(log1p(d))
    + diagnosis × f(log1p(d))
    + protocol_key
    + T1_source_class
    + processing_recipe_if_variable
    + Q
    + site/scanner partial-pooling intercept
    + error
```

`f` is a low-complexity natural cubic spline. Its degrees of freedom and knot locations are computed once from the pooled, outcome-blind locked manifest, written to configuration, and never tuned against diagnosis effects. A three-degree-of-freedom basis remains the design target for the full cohort, but its exact spline construction and the complete outcome-specific fixed/mixed design must pass a fresh rank/condition audit before SAP freeze. The SL-P0-10 polynomial proxies are feasibility diagnostics, not a frozen spline basis.

The diagnosis-by-gap interaction is tested jointly. Group contrasts are reported as standardized marginal contrasts over the observed empirical covariate distribution and as contrast curves over ranges where the compared groups have support. A coefficient evaluated at gap=0 is not reported as a general diagnosis effect if one group has little or no support near zero. If the interaction is material, the paper must state that the association depends on pairing interval; it may not average it into an unconditional biomarker claim.

Model fitting uses the original measurement scale, heteroscedasticity-robust or cluster-robust uncertainty, and site-cluster bootstrap intervals as a sensitivity. Residual shape, influence, leverage, site dominance, convergence, design rank, and partial-pooling assumptions are reported. A robust rank or quantile model is a sensitivity, not a replacement selected after seeing which model is significant.

No variable is retained or removed based on its p-value. Acquisition factors may be consolidated only from scanner/protocol semantics and outcome-blind support rules. Diagnosis-related covariates that are consequences of disease, such as contemporaneous cognitive severity, are not routine confounder adjustments.

## 6. Prespecified timing analyses

All timing analyses use exact absolute interval in days from the locked manifest.

1. **Full available cohort:** the primary exploratory estimand, with nonlinear gap and diagnosis-by-gap interaction.
2. **<=180 days:** n ceiling 197 (CN 106, MCI 60, AD 31), using the same outcome and contrasts but **without** a nonlinear gap term or diagnosis-by-gap interaction. This set has only five distinct gap values; 193/197 are zero-day pairs, and the three 91–180-day observations are AD only.
3. **<=90 days:** n ceiling 194 (CN 106, MCI 60, AD 28), using the same reduced timing-window model. It has only two distinct gap values and 193/194 are zero-day pairs, so a gap main effect would be driven by one observation and nonlinear/interaction terms are not estimable.
4. **Categorical descriptive check:** <=90, 91–180, and >180 days. The middle category is not used to estimate a three-level diagnosis interaction because it contains only three AD subjects.
5. **Functional-form checks in the full cohort only:** linear gap in days and linear `log1p(gap_days)` are compared with the prespecified spline without choosing the most favorable result.

The <=90-day Siemens-54 and its 75-person site-overlap subset are uniformly zero-gap, so every gap term is structurally non-estimable and is omitted by rule. Timing-window models also cannot support a saturated phase/site/protocol fixed design. Their role is to compare effect direction and magnitude after restricting the anatomical interval; protocol/site and T1-source sensitivities are reported separately. If a restricted contrast has inadequate group support or an unstable design, it is reported as non-estimable with its n and uncertainty rather than simplified post hoc.

## 7. Minimum validity and quantitative QC

### 7.1 Hard outcome-validity gate

Every local pair is attempted. The corrected pipeline must fail an affected outcome when any required condition is violated, including:

- wrong or unresolved subject/series identity, gradient table, shell/volume, or geometry;
- failed DTI–T1/atlas registration under the frozen spatial-QC rule;
- missing required provenance, atlas labels, matrix, or edge-support relationship;
- wrong dimensions, non-finite values, asymmetry beyond tolerance, nonzero diagonal, or impossible sign/range;
- raw `count` not integer-valued or cohort-level duplication of `count` and SIFT2 `fd_sum`;
- FA outside [0,1], diffusivity outside the frozen physical range, or mean-matrix values inconsistent with `count > 0` support;
- unexplained all-zero required nodes or failed minimum assignment/label-survival criteria.

Exact spatial thresholds for registration, label survival, assignment, motion/Eddy outliers, and other continuous QC quantities must be frozen after diagnosis-blind canary review and before group results. They cannot be chosen because they strengthen diagnosis separation.

### 7.2 Low quality that remains valid

Outputs above the hard validity floor stay in the primary available-data analysis even if their QC is poor. Continuous motion/Eddy, registration, mask/label, endpoint-assignment, streamline, and matrix-support measurements enter `Q` in the declared form. Prespecified higher-QC subsets and influence analyses test whether results are quality-driven. The old `sc_matrix_qc_include` flag is not used: it predates the repaired outputs and is false for 363/530 current rows.

The current tree illustrates why this distinction is necessary but does not define final exclusions: all 530 current count matrices are readable, yet 190 contain at least one zero row; current diffusion matrices exist for 529 subjects; three current FA matrices exceed 1; and negative diffusivity occurs in current MD/RD/AxD files. Corrected-run QC, not density or these stale artifacts, determines the outcome-specific analytic sets.

## 8. Missingness and attrition

The CONSORT-style technical flow starts at 530 and reports, by diagnosis, timing band, phase, site, scanner/protocol, and recipe:

- input found and checksum-valid;
- preprocessing completed;
- registration/atlas/QC completed;
- each of the nine matrices valid;
- each primary/retest outcome computable;
- final outcome-specific analytic n.

Failure reasons are mutually exclusive at the primary failure level and may also carry secondary flags. Invalid outcomes are NA, never numerical zero. Complete-case analysis is outcome-specific; a subject with a valid FA result is not discarded because a secondary coupling outcome is missing.

For each primary/retest outcome, model the observation indicator using diagnosis, age, sex, gap, phase, site/protocol, recipe, and available pre-outcome QC. Publish differential attrition, standardized differences, and whether group/acquisition predicts missingness. Multiple imputation is allowed for sporadically missing covariates when its assumptions and diagnostics are documented; it is not allowed to synthesize failed connectome matrices or primary imaging outcomes. Inverse-probability weighting is a sensitivity only when observation probabilities have adequate overlap and a plausible missing-at-random interpretation. Otherwise, report a bounded/tipping-point sensitivity and calibrate the claim to possible informative missingness.

SMC remains visible in every processing and failure table even though it is outside CN/MCI/AD inference. No denominator may silently change from 530 to 515 or from attempted to technically valid without an explicit label.

## 9. Harmonization policy

The primary association model uses unharmonized corrected features with explicit acquisition adjustment. Harmonization is a sensitivity, not a substitute for common support or a way to erase scanner differences.

- A batch definition (site, scanner, or protocol) is fixed from acquisition metadata, not chosen by biological p-values.
- Batches without multi-diagnosis support are identified before harmonization. ComBat cannot separate diagnosis from batch where they are aliased; such cells remain an identification limitation.
- Association-only ComBat/ComBat-GAM sensitivity may preserve the declared biological covariates, including diagnosis, but must be labeled label-conditioned and cannot be reused as a prediction pipeline.
- For ML, harmonization is fit only on each outer-training fold, uses no diagnosis/target information, and is applied to held-out data without using its outcome distribution.
- No imputation, scaling, residualization, harmonization, PCA, feature filtering, or batch correction is fit once on the full dataset before cross-validation.

Agreement between adjusted-unharmonized, acquisition-restricted, and harmonized estimates is informative. Disagreement is reported; the most favorable version is not selected as the main result.

## 10. Permutation and exchangeability

Global label shuffling is invalid because diagnosis is associated with site, phase, manufacturer, gradient directions, and protocol. The default covariate-aware test is a Freedman–Lane residual permutation under the frozen nuisance model, restricted within prespecified exchangeability blocks based on site/scanner, phase, and protocol where the data permit. At least 10,000 permutations are used for final primary and NBS/TFNBS inference; exact enumeration is used when fewer distinct valid permutations exist.

Before testing, record block sizes, diagnosis diversity per block, number of movable observations, and number of unique permutations. If the scientifically defensible blocks do not permit a valid test for a contrast, that permutation p-value is **not estimable**. Blocks are not relaxed merely to obtain a small p-value. Cluster wild-bootstrap or model-based robust intervals may be reported as complementary uncertainty estimates, but they do not cure absent overlap.

Primary-family max-T correction uses the same permutation draw for every primary contrast/outcome. Edge/component inference uses the same nuisance and block structure with maximum-component control. Permutation of ML labels repeats the entire nested pipeline and respects the same grouping; if class–block aliasing prevents valid permutation, no ML significance p-value is claimed.

## 11. Machine-learning boundary and leakage controls

Diagnostic ML is internal benchmarking only. An external/temporal validation set does not exist in this closed world, so no model is a clinically validated classifier.

- Split by subject and acquisition group; outer folds are site/scanner-grouped and class feasibility is documented.
- Leave-phase and leave-protocol analyses are stress tests. If a held-out fold lacks a class or training support, report that the target generalization is not identifiable rather than replacing it with random folds.
- Imputation, QC-derived filtering, scaling, harmonization, nuisance residualization, dimensionality reduction, feature selection, class balancing, and hyperparameter tuning occur entirely inside the training portion of nested CV.
- No SMOTE/oversampling is applied before splitting; no duplicate subject, derivative, or closely coupled feature leaks across folds.
- Report balanced accuracy, macro-AUROC, macro-F1, classwise sensitivity/specificity, Brier score, calibration intercept/slope, and grouped-bootstrap confidence intervals. Accuracy alone is insufficient.
- Compare against majority, stratified-random, age/sex-only, and acquisition-only baselines. Performance similar to an acquisition-only model is evidence of confounding, not disease signal.

The current random-fold balanced accuracy of approximately 0.505 and macro-AUROC of 0.676 remain historical exploratory baselines. They are not acceptance targets and are not evidence that a corrected model will generalize.

## 12. Claim calibration and decision rules

A result can be described only at the strongest level justified below:

- **Available-data association:** effect estimate and uncertainty from the full corrected analytic set, with multiplicity control and complete diagnostics.
- **Timing-robust within this dataset:** direction and material magnitude do not collapse in both <=180- and <=90-day analyses, while acknowledging that those windows differ by only three AD subjects; the continuous gap interaction is compatible with the stated summary.
- **Acquisition-robust within observed overlap:** direction and material magnitude remain compatible in Siemens-54 and protocol/site-overlap stress tests, or heterogeneity is explicitly modeled and reported.
- **QC-robust:** result is not driven by hard-failure coding, one site, influential observations, differential missingness, or a favorable QC threshold.
- **Internally retested:** applies to previously viewed candidates after corrected processing; never call this replicated or externally validated.
- **Novel:** not assigned by this analysis. It requires the separate claim-to-prior-art review at the exact modality/atlas/metric/network/stage/covariate/validation granularity.

Before unblinding, the team should define a scientifically defensible smallest effect size of interest for the primary FA outcome. If no defensible threshold exists, report continuous effects and confidence intervals and avoid binary “materially robust” language. Statistical significance after correction is necessary for a primary positive label but is not sufficient for biological importance, robustness, or novelty. Sensitivity analyses are judged primarily by estimates and uncertainty, not by repeated p<.05 thresholds in small subsets.

Sign reversal, severe attenuation, a material diagnosis-by-gap interaction, non-estimability under acquisition overlap, or differential attrition downgrades the claim. Null and failed outcomes remain in the results and supplement. No coherent manuscript story may be created by omitting a prespecified contradictory result.

## 13. What statistics cannot fix

The following remain limitations even under perfect implementation:

1. A multi-year DTI–T1 interval can contain genuine atrophy and anatomical change. A gap covariate cannot reconstruct the anatomy present at DTI acquisition.
2. Registration success does not make non-contemporaneous anatomy contemporaneous.
3. Hard-invalid tensor, atlas, tractography, or matrix outputs are not recoverable through covariate adjustment, harmonization, imputation, or larger permutation counts.
4. Diagnosis–phase/site/scanner/protocol cells with no overlap do not identify an adjusted disease contrast; model extrapolation and ComBat do not create missing comparators.
5. The <=90- and <=180-day analyses are almost the same cohort, and their small AD samples cannot provide independent replication.
6. SMC cannot be treated as MCI, and with n=15 all beyond 180 days it cannot support a timing-restricted disease-stage claim.
7. The approximate functional-network crosswalk cannot establish a uniquely Limbic-versus-DMN anatomical effect without mapping sensitivity.
8. Cross-sectional associations cannot establish progression, mediation, compensation, or causal mechanisms.
9. Internal grouped CV cannot replace external/temporal validation or establish clinical utility.
10. Statistical robustness cannot establish novelty; prior art must be audited separately.

The strongest defensible endpoint of this closed-world design is therefore a carefully qualified, internally retested **available-data association**, accompanied by explicit timing, acquisition, QC, missingness, and support limits.

## 14. Freeze and execution gates

### 14.1 Pre-canary implementation blockers (SL-P0-11–14)

Even if the user approves SL-H03-C1, no canary or full imaging run is authorized until all four reopened implementation gates pass:

- **SL-P0-11 — input/processing specification:** issue a new recipe/config version bound to the locked local-source manifest or validated workflow projection; attempt all 530 rows, including 333 >180-day pairs, 225 processed-T1 sources, and 15 SMC; define `processing_approved` separately from `primary_analysis_eligible`; extract authoritative DTI UID/scanner/phase-encoding/readout metadata where available; use nullable `t1_dicom_series_uid` plus a mandatory stable T1 source ID for processed NIfTI sources; stage and hash every workflow input without fabricating missing metadata.
- **SL-P0-12 — executable DAG:** normalize the approved raw-DICOM and single-NIfTI source classes into the frozen processing contract; allow independent subjects to complete or fail without aborting the cohort; emit an explicit terminal state for every attempted row; preserve v2.0.11 only as historical candidate evidence.
- **SL-P0-13 — invariant/integration tests:** prove source adaptation, >180-day/processed-T1/SMC attempt inclusion, hard-invalid-to-NA propagation, complete mixed PASS/FAIL attrition, and tamper/failure behavior while retaining all nine matrix invariants.
- **SL-P0-14 — provenance and run ledgers:** publish expected, attempted, PASS, FAIL, excluded, and outcome-specific NA states; record one primary failure plus secondary flags, exact source/content identities, and partial-run recovery; ledger publication must not require zero failures, while analysis-ready biological outputs remain validity-gated.

The SL-P0-10 workflow-contract audit found 24 blocking rows across missing/derived fields and policy conflicts. In particular, the old candidate points to a nonexistent acquisition manifest, rejects >180-day pairs and SMC, requires Original T1 despite processed inputs, and assumes metadata/staged hashes that are not yet rowwise validated. H03 design approval does not waive any of these blockers.

### 14.2 SAP freeze requirements

Before corrected group results are opened, the implementation must produce and lock:

- the 530-row local-pair manifest with exact dates/gaps, diagnosis at DTI, site/scanner/protocol, recipe, paths, and hashes;
- diagnosis-blind hard-validity and quantitative-QC rules;
- one outcome dictionary, edge-support aggregation, contrast matrix, spline basis, nuisance representation, exchangeability blocks, correction families, and random seeds;
- a blind/synthetic analysis dry run proving attrition, missing-data, multiplicity, permutation, and leakage controls;
- a feasibility report listing common-support cells and which contrasts/strata are estimable.

After execution, every result table must carry the input denominator, outcome-valid n by group, model formula, covariate/support diagnostics, effect/CI, multiplicity scope, timing stratum, acquisition/QC sensitivity status, and claim label. Deviations from this memo are versioned and justified without reference to whether they improve a diagnosis result.

## 15. H03 decision boundary

The recommended option is **SL-H03-C1**: approve this fixed-data cohort and estimand design only. Approval would lock the 530-attempt/515-primary/15-SMC population roles, the participant-level primary FA summary, the parsimonious protocol-plus-T1-source acquisition representation, the support-calibrated contrasts, and the required sensitivity families as the basis for the next implementation work. It would not freeze the final SAP or authorize a canary, full imaging execution, corrected group-result unblinding, manuscript finding, or dashboard inference release.

The separate recommendation record is [`sl_h03_c1_recommendation_20260718.md`](decisions/sl_h03_c1_recommendation_20260718.md). Its current state is `AWAITING_USER_APPROVAL`.
