#!/usr/bin/env python3
"""Build a complete evidence and claim-status ledger for manuscript v2.

This is an inventory/audit utility.  It does not rerun imaging, alter dashboard
products, or infer new biological effects.  Its purpose is to make every local
result visible while separating internally validated candidates from secondary,
exploratory, and rejected claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from submission_sprint_analysis import INPUTS, collect_metrics


PROJECT = Path("/home/ec2-user/exp")
DASHBOARD = Path("/data/derivatives/qc/analysis_cohort")
AUDIT_OUTPUTS = PROJECT / "research_audit" / "outputs"
DEFAULT_OUT = AUDIT_OUTPUTS / "all_evidence_inventory_v2"

STATUS = {
    "A": "validated internal candidate",
    "B": "valid secondary/supporting",
    "C": "exploratory/provisional",
    "D": "rejected/invalid/superseded",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def status_fields(code: str, rationale: str, role: str) -> dict[str, str]:
    return {
        "status_code": code,
        "status_label": STATUS[code],
        "status_rationale": rationale,
        "manuscript_role": role,
    }


def dashboard_family(relative: Path) -> tuple[str, str, str, str]:
    top = relative.parts[0] if relative.parts else "root"
    if top in {"00_master", "01_qc", "02_demographics", "05_connectome_qc"}:
        return (
            top,
            "B",
            "Descriptive cohort/QC evidence; not a biological novelty endpoint.",
            "Methods, cohort table, and QC supplement",
        )
    if top in {"03_global_microstructure_live", "04_node_microstructure_live", "05_multimetric_live", "19_network_analysis"}:
        return (
            top,
            "C",
            "Historical discovery source; exact selected AxD/RD claims are promoted only in the claim ledger.",
            "Results context and comprehensive supplement",
        )
    if top in {"06_global_graph_live", "07_node_graph_live", "10_edgewise_fd_sum", "12_length_delay", "14_clinical_sensitivity", "15_advanced_structural", "17_edr_exceptions", "18_ml_diagnostics"}:
        return (
            top,
            "C",
            "Historical or post-hoc result requiring claim-specific site/multiplicity/QC validation.",
            "Secondary or exploratory supplement",
        )
    if top in {"09_coupling_live", "11_brain_age_live", "13_delay", "16_functional_placeholder", "21_mediation"}:
        return (
            top,
            "D",
            "The current biological construct or headline interpretation failed validation or is not estimable.",
            "Falsification/limitations register only",
        )
    if top in {"20_findings", "exports", "figures", "tables"}:
        return (
            top,
            "C",
            "Presentation or prior interpretation artifact; authoritative status comes from versioned audit releases.",
            "Traceability only",
        )
    if top in {"logs", "99_appendix"} or "archive" in relative.parts:
        return (
            top,
            "D",
            "Operational, archived, or superseded artifact; not evidence for a manuscript claim.",
            "Audit trail only",
        )
    return (
        top,
        "C",
        "Unclassified historical dashboard artifact; no claim promotion without explicit validation.",
        "Traceability pending review",
    )


def audit_family(relative: Path) -> tuple[str, str, str, str]:
    top = relative.parts[0] if relative.parts else "root"
    if top in {"stage_specific_diffusivity_v2", "two_axis_novelty_audit_v1"}:
        return (
            top,
            "A",
            "Current internally validated discovery and bounded exact-prior-art release.",
            "Primary candidate evidence",
        )
    if top in {
        "two_axis_extensions_v1",
        "network_resolved_site_validation_v2",
        "higher_order_novelty_audit_v1",
        "primary_fa_site_sensitivity_v2",
        "cross_scale_site_sensitivity_v1",
        "cross_scale_robustness_v1",
        "submission_sprint_v1",
    }:
        return (
            top,
            "B",
            "Current secondary robustness, falsification, or coherence release.",
            "Secondary results or supplement",
        )
    if top in {
        "higher_order_hypothesis_screen_v1",
        "expanded_dimension_screen_v1",
        "coupling_support_falsification_v1",
    }:
        return (
            top,
            "D",
            "Current release records failed headline hypotheses or invalid constructs.",
            "Falsification and claim-boundary supplement",
        )
    if top.startswith(".") or ".failed" in top or "superseded" in top or top.endswith("_v1") and top in {
        "stage_specific_diffusivity_v1",
        "network_resolved_site_validation_v1",
        "primary_fa_site_sensitivity_v1",
    }:
        return (
            top,
            "D",
            "Failed, superseded, or replaced release.",
            "Audit trail only",
        )
    if top.startswith("h04a") or top in {
        "pipeline_530_progress_v1",
        "source_metadata_workflow_preflight_v1",
        "available_data_content_lock_v2",
    }:
        return (
            top,
            "B",
            "Pipeline/QC/provenance evidence, not a biological result.",
            "Methods and confirmation-status supplement",
        )
    if top.startswith("manuscript") or top.startswith("submission_package"):
        return (
            top,
            "B",
            "Derived manuscript or submission artifact; claims inherit the underlying ledger status.",
            "Manuscript packaging",
        )
    return (
        top,
        "C",
        "Versioned audit artifact not independently promoted to a primary claim.",
        "Supporting audit trail",
    )


def inventory_tree(
    root: Path,
    root_name: str,
    exclude_roots: tuple[Path, ...] = (),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        resolved = path.resolve()
        if any(resolved.is_relative_to(excluded.resolve()) for excluded in exclude_roots):
            continue
        relative = path.relative_to(root)
        family, code, rationale, role = (
            dashboard_family(relative) if root_name == "dashboard" else audit_family(relative)
        )
        stat = path.stat()
        rows.append(
            {
                "root": root_name,
                "relative_path": str(relative),
                "absolute_path": str(path),
                "family": family,
                "extension": path.suffix.lower(),
                "bytes": int(stat.st_size),
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                **status_fields(code, rationale, role),
            }
        )
    return rows


def metric_family_inventory() -> pd.DataFrame:
    metrics = collect_metrics()
    finite = metrics[pd.to_numeric(metrics["value"], errors="coerce").notna()].copy()
    rows: list[dict[str, object]] = []
    for (mapping, family), frame in finite.groupby(["mapping", "feature_family"], sort=True):
        code = "C"
        rationale = "Historical feature family; only prespecified exact claims can be promoted."
        role = "Comprehensive supplement and hypothesis context"
        if family == "topology_microstructure_coupling":
            code = "D"
            rationale = "Raw coupling headline failed shared-support falsification."
            role = "Falsification supplement"
        if mapping == "global":
            source_key = "global_micro" if family == "microstructure" else "global_graph"
        else:
            suffix = "coupling" if family == "topology_microstructure_coupling" else (
                "micro" if family == "microstructure" else "graph"
            )
            source_key = f"{mapping}_{suffix}"
        rows.append(
            {
                "domain": f"{mapping}::{family}",
                "source": str(INPUTS[source_key]),
                "unique_features": int(frame["feature_id"].nunique()),
                "subjects_with_any_value": int(frame["subject_id"].nunique()),
                "finite_observations": int(len(frame)),
                **status_fields(code, rationale, role),
            }
        )

    manual = [
        ("nodal_microstructure", DASHBOARD / "04_node_microstructure_live", "C", "Large nodal screen from historical matrices; selected regions require multiplicity/site confirmation.", "Exploratory supplement"),
        ("nodal_graph_topology", DASHBOARD / "07_node_graph_live", "C", "Density/recipe-sensitive historical graph screen; no validated headline.", "Exploratory supplement"),
        ("edgewise_connectome", DASHBOARD / "10_edgewise_fd_sum", "C", "Historical edgewise/NBS-style evidence; corrected matrices and familywise inference are pending.", "Exploratory supplement"),
        ("length_and_edr", DASHBOARD / "12_length_delay", "C", "Tract-length and EDR descriptors are descriptive; recipe dependence remains.", "Secondary supplement"),
        ("fixed_velocity_delay", DASHBOARD / "13_delay", "D", "A fixed 6 mm/ms conversion is a length rescaling, not measured conduction physiology.", "Rejected-construct register"),
        ("brain_age", DASHBOARD / "11_brain_age_live", "D", "Overall held-out age R2 is 0.0095; diagnostic brain-age interpretation is invalid.", "Rejected-construct register"),
        ("clinical_associations", DASHBOARD / "14_clinical_sensitivity", "C", "Scan-aligned clinical coverage and multiplicity require validation.", "Exploratory supplement"),
        ("advanced_structural", DASHBOARD / "15_advanced_structural", "C", "Post-hoc advanced measures require construct- and claim-specific validation.", "Exploratory supplement"),
        ("legacy_ml_diagnostics", DASHBOARD / "18_ml_diagnostics", "C", "Legacy models are not external validation; only nested site-held-out RD coherence is retained separately.", "Secondary/exploratory supplement"),
        ("functional_placeholder", DASHBOARD / "16_functional_placeholder", "D", "No functional MRI measure is present; structure-function claims are not estimable.", "Limitations only"),
        ("cross_sectional_mediation", DASHBOARD / "21_mediation", "D", "Cross-sectional paths cannot establish mediation or mechanism and are not validated.", "Prohibited-claim register"),
    ]
    for domain, source, code, rationale, role in manual:
        files = [path for path in source.rglob("*") if path.is_file()] if source.exists() else []
        rows.append(
            {
                "domain": domain,
                "source": str(source),
                "unique_features": pd.NA,
                "subjects_with_any_value": pd.NA,
                "finite_observations": pd.NA,
                "artifact_count": len(files),
                **status_fields(code, rationale, role),
            }
        )
    return pd.DataFrame(rows)


def claim_ledger() -> pd.DataFrame:
    rows = [
        ("CLM-01", "Joint adjacent-stage two-axis phenotype", "17-system AxD burden is higher in MCI than CN and matched RD burden is higher in AD than MCI.", "A", "Both endpoints survive restricted wild-cluster bootstrap-t inference with Holm correction in the full and pair-common-site scopes (pair-common Holm p=0.0054 and 0.0091); all leave-one-site-out estimates retain positive direction. The stricter 20-site all-three-group sensitivity is borderline after wild-bootstrap Holm correction (both p=0.061).", "primary_inference_soundness_v2/summary.md", "Primary inferential spine", "Confirm both locked endpoints with corrected matrices or an untouched cohort; retain the all-three-site qualification."),
        ("CLM-02", "Distributed spatial heterogeneity", "The two component effects may vary in magnitude across systems without proving a unique epicentre.", "C", "The original CR1 interaction tests were nominally positive, but restricted cluster-score sign tests did not pass for AxD (p=0.0792) or RD (p=0.585); heterogeneity is therefore exploratory rather than a validated biological layer.", "primary_inference_soundness_v2/summary.md", "Exploratory spatial context only", "Re-estimate the frozen interaction tests on corrected matrices or an untouched cohort with adequate acquisition-cluster support before promotion."),
        ("CLM-03", "Late-RD site-held-out coherence", "System RD features carry AD-versus-MCI information across held-out sites.", "B", "Nested site-held-out AUC=0.630; delta over baseline=0.193, site-bootstrap 95% interval 0.077 to 0.287.", "network_resolved_site_validation_v2/results_summary.md", "Secondary generalization layer", "Do not call external validation or a clinical biomarker."),
        ("CLM-04", "Whole-brain FA AD-CN", "Lower historical whole-brain FA in AD than CN is directionally supportive.", "B", "Full/site-adjusted result is significant, but restricted overlap and global multiplicity do not consistently support a headline.", "primary_fa_site_sensitivity_v2/primary_fa_site_sensitivity_summary.md", "Contextual conventional-DTI result", "Report estimates across support strata and keep outside the novelty claim."),
        ("CLM-05", "Preferential limbic/DMN involvement", "Limbic or DMN systems are uniquely more affected than other cortex.", "D", "Zero of eight direct target-versus-other-cortex tests survive Holm correction; adjusted p values are 1.", "higher_order_hypothesis_screen_v1/hypothesis_screen_summary.md", "Explicitly rejected selectivity claim", "May describe participation within a distributed phenotype only."),
        ("CLM-06", "AxD-to-RD mechanistic switch", "MCI reflects axonal injury followed by demyelination in AD.", "D", "Direct stage-by-component second difference p=0.1687 (Holm p=0.3375); tensor eigenvalues are not cell-specific and data are cross-sectional.", "stage_specific_diffusivity_v2/results_summary.md", "Prohibited mechanistic interpretation", "Use 'two-axis adjacent-transition phenotype', not switch or temporal mechanism."),
        ("CLM-07", "Spatial reconfiguration across stages", "The anatomical profile remaps from early AxD to late RD.", "C", "Full 51-site joint test p=0.1865; positive only in restrictive 20-site all-three-group scope.", "higher_order_hypothesis_screen_v1/hypothesis_screen_summary.md", "Exploratory sensitivity", "Requires corrected full-support confirmation before promotion."),
        ("CLM-08", "Connection-architecture specificity", "Within/between-network, hub, long-range, or hemispheric organization differentiates stages.", "D", "Zero of twelve prespecified architecture contrasts survive Holm correction.", "higher_order_hypothesis_screen_v1/hypothesis_screen_summary.md", "Negative boundary result", "Retain full test table in supplement; no positive architecture claim."),
        ("CLM-09", "Frontal topology-FA inverted U", "MCI compensation precedes AD collapse in frontal topology-microstructure coupling.", "D", "The shape fails degree/edge-support residualization and declared strata; biological inverted-U decision rejected.", "coupling_support_falsification_v1/coupling_support_falsification_summary.md", "Falsification result", "Report as shared-support measurement caution only."),
        ("CLM-10", "Conduction-delay phenotype", "Fixed-velocity dashboard delay estimates measure physiological conduction delay.", "D", "A fixed 6 mm/ms conversion is a rescaled tract-length proxy and exact contrasts fail correction.", "expanded_dimension_screen_v1/screen_summary.md", "Rejected construct", "Do not use physiological delay language without measured/validated velocity information."),
        ("CLM-11", "Accelerated connectome brain age", "Diagnostic groups show meaningful accelerated structural-connectome ageing.", "D", "Legacy age model overall R2=0.0095; a group contrast cannot rescue invalid age prediction.", "expanded_dimension_screen_v1/screen_summary.md", "Rejected construct", "Omit from main story; retain validation failure in supplement."),
        ("CLM-12", "Additional global-topology dimension", "Global efficiency, path length, density, or strength adds a robust biological layer.", "D", "No exact site-aware adjacent contrast survives within-domain correction; topology is recipe/density sensitive.", "expanded_dimension_screen_v1/screen_summary.md", "Rejected headline", "Revisit only with corrected semantically valid count/fd_sum matrices."),
        ("CLM-13", "Network resolution beats scalar RD", "A 17-system RD model is demonstrably superior to the scalar RD composite.", "D", "AUC delta=0.070 but site-bootstrap interval -0.010 to 0.134 crosses zero.", "network_resolved_site_validation_v2/results_summary.md", "Prohibited ML increment claim", "Use ML only as coherence for the late RD axis."),
        ("CLM-14", "Diagnostic biomarker", "The current model is clinically validated for individual diagnosis.", "D", "Discovery cohort reused; no untouched external cohort; MCI-CN AUC is near chance.", "network_resolved_site_validation_v2/results_summary.md", "Prohibited clinical claim", "Require external/untouched validation and clinical utility analysis."),
        ("CLM-15", "Complete-system profile dispersion", "Within-person spatial unevenness supplies an independent third dimension.", "C", "Post-selection, coverage-restricted construct; retained only as a falsification screen and not independent confirmation.", "expanded_dimension_screen_v1/screen_summary.md", "Exploratory supplement", "Predeclare and confirm on corrected complete-system data."),
        ("CLM-16", "Individual nodal/edgewise loci", "Specific parcels or edges are independently disease-defining.", "C", "Dashboard screens are high-dimensional historical outputs with recipe/QC and multiplicity concerns.", "submission_sprint_v1/submission_sprint_summary.md", "Comprehensive exploratory atlas", "Apply site-aware familywise inference on corrected matrices before naming loci as discoveries."),
        ("CLM-17", "Clinical-score linkage", "Structural axes explain contemporaneous cognitive or functional severity.", "C", "Scan-aligned clinical coverage has not yet met a frozen missingness/multiplicity gate.", "closed_world_analysis_design_v1.md", "Optional exploratory layer", "Audit visit alignment and missingness; omit if support is inadequate."),
        ("CLM-18", "Cross-sectional mediation", "Topology or microstructure mediates diagnosis-to-clinical relationships.", "D", "Cross-sectional association cannot establish temporal mediation or mechanism; local construct is not validated.", "closed_world_analysis_design_v1.md", "Prohibited causal claim", "Use association language only; longitudinal/pathology data are required."),
        ("CLM-19", "Exact-conjunction novelty", "The jointly corrected, common-support 17-system adjacent-transition AxD/RD estimand is differentiated from identified prior art.", "A", "The bounded 108-study map and exact matrices found close component/stage precedents but not the same conjunction.", "two_axis_novelty_audit_v1/novelty_boundary_and_claim_decision.md", "Calibrated novelty candidate", "Complete backward/forward chaining and a second bibliographic database; never claim proof of priority."),
    ]
    columns = [
        "claim_id", "claim_name", "claim_text", "status_code", "evidence_summary",
        "evidence_path", "manuscript_role", "required_before_vfinal",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame.insert(4, "status_label", frame["status_code"].map(STATUS))
    return frame


def build_summary(artifacts: pd.DataFrame, families: pd.DataFrame, claims: pd.DataFrame) -> str:
    artifact_counts = artifacts.groupby(["root", "status_code"]).size().to_dict()
    claim_counts = claims["status_code"].value_counts().to_dict()
    lines = [
        "# All-evidence inventory v2",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "This ledger preserves every local result while preventing a dashboard plot from becoming a manuscript claim without an explicit statistical, construct-validity, provenance, and prior-art decision.",
        "",
        "## Status key",
        "",
    ]
    for code, label in STATUS.items():
        lines.append(f"- **{code} — {label}**")
    lines.extend([
        "",
        "## Inventory coverage",
        "",
        f"- Dashboard artifacts inventoried: **{int((artifacts['root'] == 'dashboard').sum())}**",
        f"- Versioned research-audit artifacts inventoried: **{int((artifacts['root'] == 'research_audit').sum())}**",
        f"- Metric/construct families registered: **{len(families)}**",
        f"- Exact manuscript claims registered: **{len(claims)}**",
        "",
        "## Claim disposition",
        "",
    ])
    for code, label in STATUS.items():
        lines.append(f"- {code}: **{int(claim_counts.get(code, 0))}** — {label}")
    lines.extend([
        "",
        "## Current coherent hierarchy",
        "",
        "1. Primary candidate: jointly corrected adjacent-stage 17-system AxD MCI-CN and RD AD-MCI phenotype.",
        "2. Systems context: anatomical and approximate functional summaries show broad participation, while limbic/DMN selectivity and small-cluster-robust spatial heterogeneity are not established.",
        "3. Generalization support: site deletion stability and secondary held-out-site coherence of the late RD axis.",
        "4. Conventional context: FA/MD, nodal, graph, edgewise, clinical, delay, brain-age, coupling, and ML outputs remain visible in the comprehensive supplement at their ledger status.",
        "5. Confirmation boundary: corrected canary drift determines the smallest repair lane; corrected matrices or an untouched cohort are required for V-final confirmation.",
        "",
        "## Machine-readable outputs",
        "",
        "- `artifact_inventory.csv`: every current dashboard and research-audit file.",
        "- `metric_family_inventory.csv`: coverage and role of each metric/construct family.",
        "- `claim_status_matrix.csv`: exact allowed, secondary, exploratory, and prohibited claims.",
        "- `inventory_summary.json`: counts and source roots.",
        "",
        "Artifact-level status is deliberately conservative. A raw family can remain C while a prespecified exact claim derived from it is A; the claim ledger is authoritative for manuscript wording.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not DASHBOARD.is_dir() or not AUDIT_OUTPUTS.is_dir():
        raise FileNotFoundError("Dashboard and research-audit output roots must exist")
    artifacts = pd.DataFrame(
        inventory_tree(DASHBOARD, "dashboard")
        + inventory_tree(
            AUDIT_OUTPUTS,
            "research_audit",
            exclude_roots=(args.out,),
        )
    )
    families = metric_family_inventory()
    claims = claim_ledger()
    summary_json = {
        "schema_version": "2.0.0",
        "generated_utc": utc_now(),
        "roots": {"dashboard": str(DASHBOARD), "research_audit": str(AUDIT_OUTPUTS)},
        "artifact_count": int(len(artifacts)),
        "artifact_count_by_root": artifacts["root"].value_counts().to_dict(),
        "artifact_count_by_status": artifacts["status_code"].value_counts().to_dict(),
        "metric_family_count": int(len(families)),
        "claim_count": int(len(claims)),
        "claim_count_by_status": claims["status_code"].value_counts().to_dict(),
        "authoritative_claim_ledger": "claim_status_matrix.csv",
        "validation": {
            "artifact_paths_unique": bool(
                not artifacts.duplicated(["root", "relative_path"]).any()
            ),
            "claim_ids_unique": bool(not claims["claim_id"].duplicated().any()),
            "statuses_known": bool(
                set(artifacts["status_code"])
                | set(families["status_code"])
                | set(claims["status_code"])
                <= set(STATUS)
            ),
            "all_dashboard_top_level_families_seen": bool(
                set(
                    path.name
                    for path in DASHBOARD.iterdir()
                    if path.is_dir() and any(item.is_file() for item in path.rglob("*"))
                )
                <= set(artifacts.loc[artifacts["root"].eq("dashboard"), "family"])
            ),
        },
    }
    if not all(summary_json["validation"].values()):
        raise RuntimeError(f"Evidence inventory validation failed: {summary_json['validation']}")

    args.out.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.out / "artifact_inventory.csv", artifacts)
    atomic_csv(args.out / "metric_family_inventory.csv", families)
    atomic_csv(args.out / "claim_status_matrix.csv", claims)
    atomic_text(args.out / "summary.md", build_summary(artifacts, families, claims))
    atomic_text(args.out / "inventory_summary.json", json.dumps(summary_json, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary_json, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
