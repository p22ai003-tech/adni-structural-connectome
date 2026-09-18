#!/usr/bin/env python3
"""Localize the HCP379 density symptom by working backward through the pipeline.

This audit is diagnosis-blind and non-imaging.  It binds the completed
15-canary balanced-density evidence to the corresponding promoted pretract and
final-HROI atlas records, then reports stage-local evidence without pretending
that density alone identifies a root cause.

The output deliberately distinguishes:

* a confirmed failed gate;
* the earliest stage at which the symptom is visible; and
* the counterfactual experiment needed to separate registration/label effects
  from tractography-support effects.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
DENSITY_ATTEMPTS = (
    HCP_ROOT / "phase_b_hroi_balanced_density_canary_cohort_v1/attempts"
)
UNIT_DENSITY_ATTEMPTS = (
    HCP_ROOT / "phase_b_hroi_balanced_density_canary_v1/attempts"
)
ENDPOINT_RADIUS_ATTEMPTS = (
    HCP_ROOT / "phase_b_hroi_endpoint_radius_localization_v1/attempts"
)
REGISTRATION_COUNTERFACTUAL_ATTEMPTS = (
    HCP_ROOT / "phase_b_hroi_registration_counterfactual_v1/attempts"
)
GMWMI_COUNTERFACTUAL_ATTEMPTS = (
    HCP_ROOT / "phase_b_hroi_gmwmi_seeding_counterfactual_v1/attempts"
)
PROMOTED_PRETRACT = (
    HCP_ROOT
    / "pretract_promoted_recovery4_v6/"
    "pretract_promoted_recovery4_v6_manifest.json"
)
EXISTING_302 = (
    EXP
    / "research_audit/outputs/hcp379_existing_302_density_v2/"
    "existing_302_density_subjects.csv"
)
LIVE_530 = (
    EXP
    / "research_audit/outputs/hcp379_live_530_status_density_v1/"
    "summary.json"
)
SUPPORT_SATURATION = (
    EXP
    / "research_audit/outputs/hcp379_support_saturation_audit_v1/"
    "summary.json"
)
GMWMI_VALIDATION = (
    EXP
    / "research_audit/outputs/hcp379_gmwmi_seeding_counterfactual_v1/"
    "validation.json"
)
GMWMI_DECISION = (
    EXP
    / "research_audit/outputs/hcp379_gmwmi_seeding_counterfactual_v1/"
    "decision.json"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/hcp379_backward_root_cause_audit_v1"
)

EXPECTED_CANARIES = 15
EXPECTED_NODES = 379
DENSITY_TARGET = 0.60
MINIMUM_ASSIGNMENT = 0.85
MINIMUM_ATLAS_INSIDE_MASK = 0.80
MINIMUM_SELECTED_REGISTRATION_DICE = 0.80
MINIMUM_SELECTED_REGISTRATION_ATLAS_INSIDE = 0.75


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty root-cause table")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("root-cause rows have inconsistent columns")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def latest_density_summary() -> Path:
    candidates = sorted(DENSITY_ATTEMPTS.glob("*/summary.json"), reverse=True)
    for path in candidates:
        try:
            value = load_json(path)
        except Exception:
            continue
        if (
            value.get("record_type")
            == "diagnosis_blind_hcp379_hroi_balanced_density_canary_cohort_summary"
            and value.get("status")
            == "PASS_COMPLETE_COUNT_ONLY_DENSITY_EVIDENCE"
            and value.get("processed_unit_n") == EXPECTED_CANARIES
            and value.get("failure_n") == 0
            and len(value.get("units", [])) == EXPECTED_CANARIES
        ):
            return path.resolve()
    raise FileNotFoundError("complete 15-canary density summary is absent")


def latest_completed_summary(
    attempts: Path,
    *,
    record_type: str,
    status: str,
) -> Path:
    for path in sorted(attempts.glob("*/summary.json"), reverse=True):
        try:
            value = load_json(path)
        except Exception:
            continue
        if (
            value.get("record_type") == record_type
            and value.get("status") == status
        ):
            return path.resolve()
    raise FileNotFoundError(
        f"complete summary absent for {record_type} under {attempts}"
    )


def promoted_pretract_units() -> set[str]:
    value = load_json(PROMOTED_PRETRACT)
    rows = value.get("units", [])
    if (
        value.get("record_type")
        != "diagnosis_blind_hcp379_pretract_recovery4_manifest"
        or value.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or value.get("diagnosis_labels_used") is not False
        or value.get("unit_count") != EXPECTED_CANARIES
        or len(rows) != EXPECTED_CANARIES
    ):
        raise ValueError("promoted Recovery4 pretract manifest differs")
    units = {str(row.get("unit", "")) for row in rows}
    if "" in units or len(units) != EXPECTED_CANARIES:
        raise ValueError("promoted pretract unit set differs")
    return units


def existing_302_counts() -> dict[str, int]:
    with EXISTING_302.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    densities = [float(row["density"]) for row in rows]
    return {
        "available": len(rows),
        "density_pass": sum(value >= DENSITY_TARGET for value in densities),
        "density_fail": sum(value < DENSITY_TARGET for value in densities),
    }


def existing_302_density_pass_units() -> set[str]:
    with EXISTING_302.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    units = [str(row["unit"]) for row in rows]
    if len(rows) != 302 or len(set(units)) != 302 or "" in units:
        raise ValueError("existing HCP379 subject identities differ")
    return {
        str(row["unit"])
        for row in rows
        if float(row["density"]) >= DENSITY_TARGET
    }


def selected_registration_metrics(
    registration: Mapping[str, Any],
) -> tuple[float, float]:
    route = str(registration.get("route", ""))
    if route == "seeded_ants_rigid_failover":
        selected = registration.get("seeded_ants_candidate", {})
    elif route == "bbr_initialized_syn_conservative":
        selected = registration.get("promoted_candidate", {})
    else:
        selected = registration.get("recovery3_bbr", {})
    return (
        float(selected.get("five_tt_dwi_mask_dice", -1.0)),
        float(selected.get("atlas_inside_dwi_mask_fraction", -1.0)),
    )


def latest_unit_summary(
    unit: str, cohort_record: Mapping[str, Any]
) -> Path:
    fallback = Path(str(cohort_record["path"])).resolve()
    candidates = sorted(
        UNIT_DENSITY_ATTEMPTS.glob(f"*-{unit}/summary.json"),
        reverse=True,
    )
    for path in candidates:
        try:
            value = load_json(path)
        except Exception:
            continue
        if (
            value.get("record_type")
            == "diagnosis_blind_hcp379_hroi_balanced_density_canary_summary"
            and value.get("status")
            == "PASS_HROI_BALANCED_DENSITY_CANARY"
            and value.get("unit") == unit
            and set(value.get("levels", {}))
            == {"6000000", "10000000", "15000000", "20000000"}
        ):
            return path.resolve()
    return fallback


def localize(row: Mapping[str, Any]) -> tuple[str, str, str]:
    if row["pretract_status"] != "PASS":
        return (
            "PRETRACT_OR_EARLIER",
            "CONFIRMED_GATE_FAILURE",
            "Inspect gradient, Eddy, b0, scalar and FOD records; resume from "
            "the earliest failed pretract stage.",
        )
    if row["source_label_status"] != "PASS":
        return (
            "HCP379_SOURCE_LABEL",
            "CONFIRMED_GATE_FAILURE",
            "Rebuild the HCP379 source parcellation before registration.",
        )
    if row["registration_status"] != "PASS":
        return (
            "T1_TO_B0_REGISTRATION",
            "CONFIRMED_GATE_FAILURE",
            "Test locked BBR and seeded-rigid transforms on the same "
            "tractogram; retain only an independently passing transform.",
        )
    if row["endpoint_assignment_status"] != "PASS":
        return (
            "ENDPOINT_ASSIGNMENT",
            "CONFIRMED_GATE_FAILURE",
            "Hold tractography fixed and audit endpoint-to-label distances; "
            "do not inflate the assignment radius to manufacture density.",
        )
    if row["connected_node_status"] != "PASS":
        return (
            "NODE_CONNECTIVITY",
            "LOCALIZED_NOT_CAUSALLY_RESOLVED",
            "Remap the identical tractogram through alternate locked "
            "registration/label candidates.  If the disconnected nodes "
            "persist, the cause moves downstream to tractography support.",
        )
    if row["density_status"] != "PASS":
        return (
            "TRACTOGRAPHY_EDGE_DIVERSITY",
            "LOCALIZED_NOT_CAUSALLY_RESOLVED",
            "Keep the passing atlas and registration fixed; compare a "
            "bounded GMWMI-seeded ACT variant with the locked dynamic-seed "
            "recipe using two independent seeds and a passing control.",
        )
    return (
        "COUNT_DENSITY_CHAIN_PASS",
        "PASS_PENDING_ALL_NINE",
        "Generate all nine matched matrices and test technical invariants, "
        "weighted-support agreement and seed stability.",
    )


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    density_path = latest_density_summary()
    density = load_json(density_path)
    endpoint_radius_path = latest_completed_summary(
        ENDPOINT_RADIUS_ATTEMPTS,
        record_type="diagnosis_blind_hcp379_endpoint_radius_localization",
        status="PASS_COMPLETE_ENDPOINT_RADIUS_LOCALIZATION",
    )
    endpoint_radius = load_json(endpoint_radius_path)
    registration_counterfactual_path = latest_completed_summary(
        REGISTRATION_COUNTERFACTUAL_ATTEMPTS,
        record_type=(
            "diagnosis_blind_hcp379_hroi_registration_counterfactual"
        ),
        status="PASS_COMPLETE_036_REGISTRATION_COUNTERFACTUAL",
    )
    registration_counterfactual = load_json(
        registration_counterfactual_path
    )
    support_saturation = load_json(SUPPORT_SATURATION)
    if (
        support_saturation.get("record_type")
        != "hcp379_support_saturation_audit"
        or support_saturation.get("status")
        != "PASS_DIAGNOSTIC_SUPPORT_SATURATION_AUDIT"
        or support_saturation.get("observed_density_pass_n") != 10
        or support_saturation.get("observed_density_fail_n") != 5
    ):
        raise ValueError("support-saturation audit differs")
    gmwmi_counterfactual_path = latest_completed_summary(
        GMWMI_COUNTERFACTUAL_ATTEMPTS,
        record_type=(
            "diagnosis_blind_hcp379_gmwmi_seeding_counterfactual"
        ),
        status="PASS_COMPLETE_GMWMI_SEEDING_COUNTERFACTUAL",
    )
    gmwmi_counterfactual = load_json(gmwmi_counterfactual_path)
    gmwmi_validation = load_json(GMWMI_VALIDATION)
    gmwmi_decision = load_json(GMWMI_DECISION)
    if (
        gmwmi_validation.get("record_type")
        != "independent_hcp379_gmwmi_seeding_counterfactual_validation"
        or gmwmi_validation.get("status") != "PASS"
        or Path(str(gmwmi_validation.get("summary", ""))).resolve()
        != gmwmi_counterfactual_path
        or gmwmi_decision.get("record_type")
        != "hcp379_gmwmi_counterfactual_expansion_gate"
        or gmwmi_decision.get("status")
        != "REJECT_GMWMI_AS_UNIFORM_REMEDY_AT_6M_SCREEN"
        or Path(str(gmwmi_decision.get("summary", ""))).resolve()
        != gmwmi_counterfactual_path
        or Path(
            str(gmwmi_decision.get("independent_validation", ""))
        ).resolve()
        != GMWMI_VALIDATION.resolve()
    ):
        raise ValueError("GMWMI counterfactual evidence differs")
    pretract = promoted_pretract_units()
    rows: list[dict[str, Any]] = []
    for density_row in density["units"]:
        unit = str(density_row["unit"])
        unit_summary_path = latest_unit_summary(
            unit, density_row["summary"]
        )
        unit_summary = load_json(unit_summary_path)
        atlas_result_path = Path(
            unit_summary["hroi_atlas_result"]["path"]
        ).resolve()
        atlas_result = load_json(atlas_result_path)
        atlas = atlas_result["atlas_qc"]
        registration = atlas_result["registration_route"]
        selected_dice, selected_atlas = selected_registration_metrics(
            registration
        )
        level = unit_summary["levels"]["20000000"]
        complementarity = level["support_complementarity"]
        assignment = (
            float(level["primary_endpoint_assignment_fraction"])
            + float(level["independent_endpoint_assignment_fraction"])
        ) / 2.0
        labels_found = int(atlas.get("labels_found", -1))
        source_label_pass = bool(
            atlas.get("status") == "PASS"
            and labels_found == EXPECTED_NODES
            and not atlas.get("missing_labels")
            and not atlas.get("unexpected_labels")
            and atlas.get("node_volume_preservation", {}).get(
                "fraction_within_0_5_to_2_0", 0.0
            )
            >= 0.95
        )
        registration_pass = bool(
            float(atlas["atlas_inside_dwi_mask_fraction"])
            >= MINIMUM_ATLAS_INSIDE_MASK
            and selected_dice >= MINIMUM_SELECTED_REGISTRATION_DICE
            and selected_atlas
            >= MINIMUM_SELECTED_REGISTRATION_ATLAS_INSIDE
        )
        base = {
            "unit": unit,
            "pretract_status": "PASS" if unit in pretract else "FAIL",
            "source_label_status": "PASS" if source_label_pass else "FAIL",
            "registration_status": "PASS" if registration_pass else "FAIL",
            "registration_route": registration.get("route"),
            "selected_5tt_dwi_mask_dice": selected_dice,
            "selected_atlas_inside_dwi_mask_fraction": selected_atlas,
            "final_atlas_inside_dwi_mask_fraction": float(
                atlas["atlas_inside_dwi_mask_fraction"]
            ),
            "labels_found": labels_found,
            "minimum_voxels_per_node": int(
                atlas["minimum_voxels_per_node"]
            ),
            "endpoint_assignment_fraction": assignment,
            "endpoint_assignment_status": (
                "PASS" if assignment >= MINIMUM_ASSIGNMENT else "FAIL"
            ),
            "connected_nodes": int(level["connected_nodes"]),
            "connected_node_status": (
                "PASS"
                if int(level["connected_nodes"]) == EXPECTED_NODES
                else "FAIL"
            ),
            "edge_density_6m": float(
                unit_summary["levels"]["6000000"]["edge_density"]
            ),
            "edge_density_10m": float(
                unit_summary["levels"]["10000000"]["edge_density"]
            ),
            "edge_density_15m": float(
                unit_summary["levels"]["15000000"]["edge_density"]
            ),
            "edge_density_20m": float(level["edge_density"]),
            "density_status": (
                "PASS"
                if float(level["edge_density"]) >= DENSITY_TARGET
                else "FAIL"
            ),
            "seed_support_jaccard_20m": float(
                complementarity["support_jaccard"]
            ),
            "union_supported_edges_20m": int(
                complementarity["union_supported_edges"]
            ),
            "all_nine_matrix_qc_status": "NOT_RUN",
            "weighted_support_status": "NOT_RUN",
            "full_seed_stability_status": "NOT_RUN",
            "human_visual_qc_status": "AWAITING_FINAL_HROI_16_OF_16",
        }
        stage, confidence, action = localize(base)
        endpoint_result = endpoint_radius.get("units", {}).get(unit)
        if (
            stage == "NODE_CONNECTIVITY"
            and isinstance(endpoint_result, dict)
            and endpoint_result.get("localization")
            == "NO_TRACTOGRAPHY_ENDPOINT_SUPPORT_WITHIN_8MM"
        ):
            stage = "TRACTOGRAPHY_ENDPOINT_SUPPORT"
            confidence = "CONFIRMED_NO_ENDPOINT_SUPPORT_WITHIN_8MM"
            action = (
                "Keep the selected atlas, registration and 4 mm production "
                "assignment fixed; test a bounded cohort-uniform tractography "
                "seeding counterfactual with independent seeds and a passing "
                "control."
            )
        rows.append(
            {
                **base,
                "earliest_visible_failure_stage": stage,
                "localization_confidence": confidence,
                "next_isolation_experiment": action,
            }
        )

    for row in rows:
        if row["earliest_visible_failure_stage"] in {
            "TRACTOGRAPHY_EDGE_DIVERSITY",
            "TRACTOGRAPHY_ENDPOINT_SUPPORT",
        }:
            row["next_isolation_experiment"] = (
                "GMWMI seeding has been independently rejected as a "
                "uniform remedy at the bounded 6M screen. Keep the fixed "
                "atlas, assignment rule and two-seed 6M total, then "
                "predeclare one different tractography-diversity factor "
                "before any further execution."
            )

    rows.sort(key=lambda value: value["unit"])
    if len(rows) != EXPECTED_CANARIES:
        raise ValueError("root-cause row count differs")
    buckets = Counter(
        str(row["earliest_visible_failure_stage"]) for row in rows
    )
    live = load_json(LIVE_530)
    full = next(
        row
        for row in live["rows"]
        if row["processing_group"] == "Full release total"
    )
    historical = existing_302_counts()
    historical_pass_units = existing_302_density_pass_units()
    corrected_pass_units = {
        str(row["unit"])
        for row in rows
        if row["density_status"] == "PASS"
    }
    pass_overlap_units = historical_pass_units & corrected_pass_units
    stale_009 = next(
        row for row in density["units"] if row["unit"] == "009_S_4324_I1186579"
    )
    current_009 = next(
        row for row in rows if row["unit"] == "009_S_4324_I1186579"
    )
    corrected_009_summary = latest_unit_summary(
        "009_S_4324_I1186579", stale_009["summary"]
    )
    summary = {
        "schema_version": "1.4.0",
        "record_type": "hcp379_backward_root_cause_audit",
        "generated_utc": utc_now(),
        "status": "PASS_LOCALIZATION_AUDIT_READY_FOR_COUNTERFACTUALS",
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "imaging_executed": False,
        "counterfactual_imaging_executed": True,
        "matrices_modified": False,
        "density_target": DENSITY_TARGET,
        "important_interpretation": (
            "Density is an observed matrix property, not an accuracy metric "
            "and not by itself a root-cause diagnosis."
        ),
        "current_530_state": {
            "target_n": 530,
            "pretract_completed_n": int(full["pretract_completed_n"]),
            "pretract_running_n": int(full["running_n"]),
            "corrective_hold_n": int(full["corrective_hold_n"]),
            "waiting_n": int(full["waiting_n"]),
            "existing_hcp379": historical,
            "corrected_hcp379_release_available_n": int(
                full["corrected_hcp379_available_n"]
            ),
        },
        "corrected_15_canary_count_only_state": {
            "density_pass_n": sum(
                row["density_status"] == "PASS" for row in rows
            ),
            "density_fail_n": sum(
                row["density_status"] == "FAIL" for row in rows
            ),
            "all_nine_qc_complete_n": 0,
            "final_release_usable_n": 0,
        },
        "cross_construction_density_reconciliation": {
            "historical_density_pass_n": len(historical_pass_units),
            "corrected_canary_density_pass_n": len(corrected_pass_units),
            "pass_overlap_n": len(pass_overlap_units),
            "pass_overlap_units": sorted(pass_overlap_units),
            "unique_units_with_any_observed_density_pass_n": len(
                historical_pass_units | corrected_pass_units
            ),
            "pooling_authorized": False,
            "interpretation": (
                "The historical and corrected-canary constructions are not "
                "a uniform analysis cohort. The union is an inventory count, "
                "not an analysis-ready denominator."
            ),
        },
        "completed_counterfactual_evidence": {
            "009_route_propagation_correction": {
                "changed_factor": (
                    "propagate the exact promoted T1-to-b0 route into the "
                    "final-HROI atlas builder"
                ),
                "stale_route_density_20m": float(
                    stale_009["levels"]["20000000"]["edge_density"]
                ),
                "corrected_route_density_20m": float(
                    current_009["edge_density_20m"]
                ),
                "density_delta": float(current_009["edge_density_20m"])
                - float(
                    stale_009["levels"]["20000000"]["edge_density"]
                ),
                "corrected_connected_nodes": int(
                    current_009["connected_nodes"]
                ),
                "corrected_summary": file_record(corrected_009_summary),
                "interpretation": (
                    "The route-propagation defect materially reduced density "
                    "but did not restore the two unsupported parcels."
                ),
            },
            "036_registration_counterfactual": {
                "localization": registration_counterfactual["localization"],
                "selected_bbr_density_20m": float(
                    registration_counterfactual["baseline"][
                        "edge_density_20m"
                    ]
                ),
                "alternate_ants_density_20m": float(
                    registration_counterfactual["counterfactual"]["levels"][
                        "20000000"
                    ]["edge_density"]
                ),
                "alternate_minus_selected_density": float(
                    registration_counterfactual["difference"][
                        "edge_density_20m"
                    ]
                ),
                "production_route_selected": False,
                "summary": file_record(registration_counterfactual_path),
            },
            "endpoint_radius_localization": {
                "tested_radius_mm": [2, 4, 6, 8],
                "localization_by_unit": {
                    unit: value["localization"]
                    for unit, value in sorted(
                        endpoint_radius["units"].items()
                    )
                },
                "production_assignment_radius_selected": False,
                "interpretation": (
                    "The disconnected parcels remain unsupported through "
                    "8 mm, so increasing assignment radius cannot resolve "
                    "their node-connectivity failure."
                ),
                "summary": file_record(endpoint_radius_path),
            },
            "support_saturation": {
                "density_fail_n": int(
                    support_saturation["observed_density_fail_n"]
                ),
                "failed_with_diagnostic_ceiling_below_target_n": int(
                    support_saturation[
                        "failed_with_diagnostic_ceiling_below_target_n"
                    ]
                ),
                "interpretation": (
                    "All five observed density failures have a two-seed "
                    "Chapman support estimate below 0.60 and already observe "
                    "approximately 98 percent or more of that diagnostic "
                    "estimate. More streamlines under the unchanged recipe "
                    "are therefore not the preferred next experiment."
                ),
                "production_gate_changed": False,
                "summary": file_record(SUPPORT_SATURATION),
            },
            "gmwmi_seeding_counterfactual": {
                "status": gmwmi_decision["status"],
                "changed_factor": (
                    "seed_dynamic_to_seed_gmwmi_only at fixed two-seed "
                    "balanced 6M total"
                ),
                "tested_units": sorted(gmwmi_decision["evidence"]),
                "evidence": {
                    unit: {
                        "dynamic_density_6m": float(
                            value["dynamic_density_6m"]
                        ),
                        "gmwmi_density_6m": float(
                            value["gmwmi_density_6m"]
                        ),
                        "density_difference": float(
                            value["density_difference"]
                        ),
                        "gmwmi_connected_nodes": int(
                            value["gmwmi_connected_nodes"]
                        ),
                        "gmwmi_endpoint_assignment_fraction": float(
                            value["gmwmi_endpoint_assignment_fraction"]
                        ),
                        "seed_edge_gains": value["seed_edge_gains"],
                    }
                    for unit, value in sorted(
                        gmwmi_decision["evidence"].items()
                    )
                },
                "criteria_results": gmwmi_decision["criteria_results"],
                "production_recipe_selected": False,
                "scale_up_authorized": False,
                "interpretation": (
                    "GMWMI seeding reduced density and supported-edge "
                    "diversity in both the adverse case and passing control. "
                    "It is rejected as the cohort-uniform remedy; no bounded "
                    "expansion or production scale-up is authorized."
                ),
                "summary": file_record(gmwmi_counterfactual_path),
                "independent_validation": file_record(GMWMI_VALIDATION),
                "decision": file_record(GMWMI_DECISION),
            },
        },
        "localization_buckets": dict(sorted(buckets.items())),
        "stage_logic": [
            "pretract/raw-gradient/Eddy/b0/scalar/FOD contract",
            "HCP379 source-label completeness and parcel-volume preservation",
            "T1-to-b0 registration and final atlas-inside-mask geometry",
            "endpoint assignment fraction",
            "379-node connectivity",
            "count-matrix density",
            "all-nine algebraic and weighted-support QC",
            "two-seed stability and final blinded HROI visual review",
        ],
        "next_experiment_design": {
            "node_coverage_arm": [
                row["unit"]
                for row in rows
                if row["connected_node_status"] == "FAIL"
            ],
            "tractography_diversity_arm": [
                row["unit"]
                for row in rows
                if row["connected_node_status"] == "PASS"
                and row["density_status"] == "FAIL"
            ],
            "passing_control": max(
                rows, key=lambda row: row["edge_density_20m"]
            )["unit"],
            "rejected_uniform_counterfactuals": [
                "alternate ANTs registration for 036",
                "endpoint assignment radius increase through 8 mm",
                "more streamlines under the unchanged dynamic-seed recipe",
                "GMWMI seeding at balanced 6M",
            ],
            "current_next_factor_status": (
                "REQUIRES_PREDECLARATION_AFTER_GMWMI_REJECTION"
            ),
            "rules": [
                "Keep diagnosis and outcome labels hidden.",
                "Change one stage at a time and reuse the same tractogram "
                "for atlas/registration counterfactuals.",
                "For tractography counterfactuals, keep atlas, assignment "
                "rule, streamline total and all downstream QC fixed.",
                "Do not select a subject-specific production recipe.",
                "Do not authorize 530 scale-up until one uniform recipe "
                "passes density, all-nine QC and two-seed stability.",
            ],
        },
        "records": {
            "density_summary": file_record(density_path),
            "promoted_pretract": file_record(PROMOTED_PRETRACT),
            "existing_302": file_record(EXISTING_302),
            "live_530": file_record(LIVE_530),
            "endpoint_radius_localization": file_record(
                endpoint_radius_path
            ),
            "registration_counterfactual_036": file_record(
                registration_counterfactual_path
            ),
            "support_saturation": file_record(SUPPORT_SATURATION),
            "gmwmi_counterfactual": file_record(
                gmwmi_counterfactual_path
            ),
            "gmwmi_independent_validation": file_record(
                GMWMI_VALIDATION
            ),
            "gmwmi_expansion_decision": file_record(GMWMI_DECISION),
            "builder": file_record(Path(__file__)),
        },
    }
    return rows, summary


def write_markdown(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    corrected = summary["corrected_15_canary_count_only_state"]
    existing = summary["current_530_state"]["existing_hcp379"]
    reconciled = summary["cross_construction_density_reconciliation"]
    lines = [
        "# HCP379 backward root-cause audit",
        "",
        f"Generated: {summary['generated_utc']}",
        "",
        "Density is not an accuracy metric and is not, by itself, a root-cause "
        "diagnosis.",
        "",
        "## Current evidence",
        "",
        f"- Existing HCP379 matrices: {existing['available']}; "
        f"{existing['density_pass']} exceed density 0.60.",
        f"- Corrected 20M count-only canaries: "
        f"{corrected['density_pass_n']}/15 exceed 0.60.",
        f"- Across both constructions, "
        f"{reconciled['unique_units_with_any_observed_density_pass_n']} "
        "unique subjects have at least one observed count density >=0.60 "
        f"({reconciled['pass_overlap_n']} subject overlaps), but pooling is "
        "not authorized.",
        "- Corrected final release: 0 subjects, because all-nine QC and the "
        "uniform recipe gate have not yet passed.",
        "- Registration and endpoint-radius counterfactuals do not recover "
        "the remaining disconnected parcels; the earliest confirmed failure "
        "for those cases is tractography endpoint support.",
        "- All five density failures have a diagnostic two-seed support "
        "ceiling below 0.60 and are already near that estimate; increasing "
        "streamline count under the unchanged recipe is not supported.",
        "- The independently replayed two-case GMWMI counterfactual reduced "
        "density in both the adverse case and passing control; GMWMI seeding "
        "is rejected as the uniform remedy at the balanced 6M screen.",
        "",
        "## Canary localization",
        "",
        "| Unit | Density | Nodes | Assignment | Registration | Earliest "
        "visible stage |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['unit']} | {row['edge_density_20m']:.3f} | "
            f"{row['connected_nodes']} | "
            f"{row['endpoint_assignment_fraction']:.3f} | "
            f"{row['registration_route']} | "
            f"{row['earliest_visible_failure_stage']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Cases with fewer than 379 connected nodes enter a same-tractogram "
            "registration/label counterfactual first. Cases with 379 connected "
            "nodes, adequate endpoint assignment and low density enter a "
            "fixed-atlas tractography-diversity counterfactual. Passing density "
            "still does not make a case releasable until all nine matched "
            "matrices, weighted support, seed stability, visual QC, provenance "
            "and hashes pass.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows, summary = build()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subject_path = OUTPUT / "subject_stage_localization.csv"
    summary_path = OUTPUT / "summary.json"
    markdown_path = OUTPUT / "STATUS.md"
    atomic_csv(subject_path, rows)
    atomic_json(summary_path, summary)
    write_markdown(markdown_path, rows, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "subject_table": str(subject_path),
                "summary": str(summary_path),
                "status_markdown": str(markdown_path),
                "localization_buckets": summary["localization_buckets"],
                "corrected_15_canary_count_only_state": (
                    summary["corrected_15_canary_count_only_state"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
