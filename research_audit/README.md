# Structural-connectome research audit package

**Audit date:** 2026-07-18 UTC  
**Current decision:** stop manuscript claims; SL-H01, the exact 530-pair source lock, fixed-data feasibility, and design-only SL-H03-C1 are complete. SL-P0-11 is active; no imaging execution is authorized, and SL-P0-11–14 must be refactored/refrozen before a canary.

## Primary deliverables

- [Superlist — authoritative master checklist](SUPERLIST.md) ([Word](SUPERLIST.docx))
- [Objective 1 audit and corrective plan](objective1_audit_report.md) ([Word](objective1_audit_report.docx))
- [Manuscript v1](manuscript_v1.md) ([Word](manuscript_v1.docx))
- [Current project context](../structural_connectome_context.md) ([Word](structural_connectome_context_current.docx))
- [108-study evidence map](literature_evidence_map.csv)
- [Rapid structured literature review](literature_review_v1.md) ([Word](literature_review_v1.docx))
- [Approved local-only data-scope decision](decisions/closed_world_data_scope_20260718.md) ([JSON](decisions/closed_world_data_scope_20260718.json))
- [Approved available-data/minimum-validity policy — SL-H01](decisions/sl_h01_available_data_policy_20260718.md) ([JSON](decisions/sl_h01_available_data_policy_20260718.json))
- [Closed-world analysis design v1.1](closed_world_analysis_design_v1.md) ([Word](closed_world_analysis_design_v1.docx))
- [Approved SL-H03-C1 design decision](decisions/sl_h03_c1_approval_20260718.md) ([JSON](decisions/sl_h03_c1_approval_20260718.json); [Word](decisions/sl_h03_c1_approval_20260718.docx); [full recommendation](decisions/sl_h03_c1_recommendation_20260718.md))

## Evidence audits

- [Pipeline and live-results audit](pipeline_and_results_audit.md) ([Word](pipeline_and_results_audit.docx))
- [Requested Codex/Claude chat reconstruction](chat_history_context.md) ([Word](chat_history_context.docx))
- [Exact-date catalog source inventory](catalog_source_inventory.md) ([Word](catalog_source_inventory.docx))
- [Historical provisional v1 snapshot](snapshots/historical_provisional_v1/summary.md) ([Word](snapshots/historical_provisional_v1/summary.docx))
- [Independent statistics summary](outputs/audit_statistics_summary.md)
- [Subject-level cohort audit manifest](outputs/cohort_audit_manifest.csv)
- [All 214-feature sensitivity results](outputs/metric_sensitivity_results.csv)
- [Visit-matched T1 dry-run summary](outputs/visit_matched_t1_manifest_summary.md)
- [Visit-matched T1 dry-run manifest](outputs/visit_matched_t1_manifest_dryrun.csv)
- [Current-pair exact-date evidence](outputs/current_t1_exact_date_evidence_v2.csv)
- [Replacement-candidate catalog coverage](outputs/replacement_t1_catalog_coverage_v2.csv)
- [Full exact-date T1 request roster](outputs/exact_date_t1_candidate_request_v2.csv) ([validation](outputs/exact_date_t1_candidate_request_validation_v2.json); [summary](outputs/exact_date_t1_candidate_request_summary_v2.md))
- [Diagnosis reconciliation](outputs/diagnosis_reconciliation_v2.csv) ([32-row DTI identity crosswalk](outputs/dti_identity_mismatch_resolution_v2.csv); [validation](outputs/diagnosis_reconciliation_validation_v2.json); [summary](outputs/diagnosis_reconciliation_summary_v2.md); [Word summary](outputs/diagnosis_reconciliation_summary_v2.docx))
- [SL-D01 available-data pair manifest](outputs/available_data_pair_manifest_v2.csv) ([strata](outputs/available_data_pair_strata_v2.csv); [flow](outputs/available_data_pair_flow_v2.csv); [validation](outputs/available_data_pair_manifest_validation_v2.json); [summary](outputs/available_data_pair_manifest_summary_v2.md); [Word summary](outputs/available_data_pair_manifest_summary_v2.docx))
- [Immutable local-input content lock](outputs/available_data_content_lock_v2/available_data_content_lock_validation_v2.json) ([locked 530-row manifest](outputs/available_data_content_lock_v2/available_data_pair_manifest_locked_v2.csv); [451,115-file inventory](outputs/available_data_content_lock_v2/available_data_file_content_inventory_v2.csv); [summary](outputs/available_data_content_lock_v2/available_data_content_lock_summary_v2.md); [Word summary](outputs/available_data_content_lock_summary_v2.docx))
- [Fixed-data feasibility/power memo](outputs/fixed_data_feasibility_power_memo_v1.md) ([Word](outputs/fixed_data_feasibility_power_memo_v1.docx); [validation](outputs/cohort_v2_feasibility_validation.json); [common support](outputs/cohort_v2_common_support.csv); [design diagnostics](outputs/cohort_v2_design_diagnostics.csv); [estimability](outputs/cohort_v2_estimability.csv); [workflow-contract audit](outputs/cohort_v2_workflow_contract_feasibility.csv))
- [Historical header-only source preflight](outputs/historical_source_preflight_proxy_summary_v1.md) ([Word](outputs/historical_source_preflight_proxy_summary_v1.docx); [validation](outputs/historical_source_preflight_proxy_validation_v1.json))
- [ADNI/IDA exact-date export handoff — contingency, superseded by SL-D01](adni_exact_date_export_handoff.md)
- [Fail-closed exact-date catalog validator](validate_exact_date_catalog.py)
- [Checksum-attested visit-pair builder](build_visit_pair_manifest.py)

## Figures

- [Pipeline blockers and corrected path](figures/pipeline_audit_flow.png)
- [Cohort repair funnel](figures/cohort_repair_flow.png)
- [DTI–T1 timing gap](figures/dti_t1_age_gap_by_group.png)
- [Protocol composition by diagnosis](figures/protocol_composition_by_group.png)
- [Key-metric sensitivity heatmap](figures/key_metric_sensitivity_heatmap.png)
- [Fixed-data common support](figures/cohort_v2_common_support.png)
- [Detectable-effect/attrition stress](figures/cohort_v2_mde_attrition.png)

SVG sources are supplied for the two Graphviz flowcharts. Word versions of the primary reports are generated from the Markdown sources; Markdown remains authoritative.

## Reproduce the non-destructive audit

```bash
/home/ec2-user/exp/.venv_connectome_app/bin/python \
  /home/ec2-user/exp/research_audit/audit_analysis.py

/home/ec2-user/exp/.venv_connectome_app/bin/python \
  /home/ec2-user/exp/research_audit/build_visit_matched_manifest.py

/home/ec2-user/exp/.venv_connectome_app/bin/python \
  /home/ec2-user/exp/research_audit/audit_catalog_sources.py

/home/ec2-user/exp/.venv_connectome_app/bin/python \
  /home/ec2-user/exp/research_audit/run_fixed_data_feasibility.py

PYTHONPATH=/home/ec2-user/exp \
  /home/ec2-user/exp/.venv_connectome_app/bin/python -m unittest discover \
  -s research_audit/tests -p 'test_*.py' -v
```

The audit scripts write only to `research_audit/outputs` and `research_audit/figures`; the final command currently runs 62 tests. `audit_catalog_sources.py` preserves the separately attested AWS refresh and reports both the 4,015-ID full universe and nested 330-row provisional subset. `run_fixed_data_feasibility.py` opens no biological outcomes; its separately attested upstream proxy reads headers only and no voxel arrays. None alters production matrices or refreshes the dashboard. The 92.262743-GiB content lock is already complete; do not rerun `lock_available_data_content.py` casually, and use its identity-bound checkpoint/release rules if a deliberate revalidation is required.

## Interpretation boundary

The evidence map is a rapid structured screen, not a registered systematic review or proof of absence. The manuscript is a v1 audit/reproducibility draft, not a submission-ready biological discovery paper. Final claims require the P0 rebuild and exact-claim novelty confirmation specified in Objective 1.
