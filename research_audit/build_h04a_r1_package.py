#!/usr/bin/env python3
"""Build the exact diagnosis-blind 15-unit SL-H04A-R1 execution package."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path("/home/ec2-user/exp")
AUDIT = ROOT / "research_audit"
PARENT = AUDIT / "outputs/connectome_v2_input_manifest_v2.csv"
ORIGINAL_SUBSET = AUDIT / "outputs/h04a_canary_package_v1/candidate_canary_manifest_v1.csv"
APPROVAL = AUDIT / "decisions/sl_h04a_r1_approval_20260718.json"
RECOMMENDATION = AUDIT / "decisions/sl_h04a_r1_recovery_recommendation_20260718.json"
CONFIG = ROOT / "configs/connectome_v2.yaml"
ENVIRONMENT = ROOT / "scforge/workflow/environment_contract.yaml"
OUTPUT = AUDIT / "outputs/h04a_r1_recovery_package_v1"
RUN_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260718_v1")
CONVERTER = ROOT / "tools/dcm2niix/v1.0.20260416/dcm2niix"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def staged_record(staged: Path, final: Path) -> dict[str, Any]:
    return {
        "path": str(final.resolve()),
        "sha256": sha256_file(staged),
        "size_bytes": staged.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return data


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or ())
    if not rows or not fields:
        raise ValueError(f"empty CSV: {path}")
    return rows, fields


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def unit(row: Mapping[str, str]) -> str:
    return f"{row['subject_id']}_I{row['dti_image_id']}"


def manufacturer_family(value: str) -> str:
    normalized = value.strip().upper()
    if "SIEMENS" in normalized:
        return "SIEMENS"
    if normalized.startswith("GE"):
        return "GE"
    if "PHILIPS" in normalized:
        return "PHILIPS"
    return "OTHER"


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"immutable output package exists: {OUTPUT}")
    approval = load_json(APPROVAL)
    recommendation = load_json(RECOMMENDATION)
    if (
        approval.get("status") != "APPROVED"
        or approval.get("approved") is not True
        or approval.get("decision_id") != "SL-H04A-R1"
        or approval.get("user_response") != "Approve SL-H04A-I1 and SL-H04A-R1"
        or recommendation.get("decision_id") != "SL-H04A-R1"
        or sha256_file(RECOMMENDATION) != approval.get("recommendation", {}).get("sha256")
    ):
        raise ValueError("R1 approval/recommendation binding differs")
    if sha256_file(CONVERTER) != approval.get("converter", {}).get("sha256"):
        raise ValueError("pinned dcm2niix binary differs")

    parent, fields = read_csv(PARENT)
    original, original_fields = read_csv(ORIGINAL_SUBSET)
    if fields != original_fields or len(parent) != 530 or len(original) != 24:
        raise ValueError("parent/original H04A manifest contract differs")
    parent_by_unit = {unit(row): row for row in parent}
    original_by_unit = {unit(row): row for row in original}
    units = approval.get("units")
    if (
        not isinstance(units, list)
        or units != sorted(set(units))
        or len(units) != approval.get("approved_unit_count")
        or len(units) != 15
    ):
        raise ValueError("approved R1 unit set is invalid")
    if not set(units).issubset(original_by_unit):
        raise ValueError("R1 contains a unit outside the diagnosis-blind H04A canary")
    rows = [parent_by_unit[name] for name in units]
    if any(parent_by_unit[name] != original_by_unit[name] for name in units):
        raise ValueError("R1 row differs between parent and approved H04A subset")
    family_counts: dict[str, int] = {}
    t1_counts: dict[str, int] = {}
    for row in rows:
        family = manufacturer_family(row["manufacturer"])
        family_counts[family] = family_counts.get(family, 0) + 1
        t1_kind = row["t1_source_kind"]
        t1_counts[t1_kind] = t1_counts.get(t1_kind, 0) + 1
    if family_counts != {"GE": 8, "SIEMENS": 7}:
        raise ValueError(f"manufacturer strata differ: {family_counts}")
    if t1_counts != {"dicom_series": 8, "nifti_single": 7}:
        raise ValueError(f"T1 source strata differ: {t1_counts}")

    environment = yaml.safe_load(ENVIRONMENT.read_text(encoding="utf-8"))
    recipe = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["contract"]["recipe_id"]
    workflow_manifest = Path(environment["workflow"]["source_manifest"]["path"])
    temporary = Path(tempfile.mkdtemp(prefix=".h04a_r1_package.", dir=OUTPUT.parent))
    try:
        subset_stage = temporary / "recovery_execution_manifest_v1.csv"
        write_csv(subset_stage, rows, fields)
        subset_final = OUTPUT / subset_stage.name
        decision = {
            "schema_version": "2.0.0",
            "decision_type": "connectome_response_recovery_subset_approval",
            "status": "APPROVED",
            "approval_mode": "H04A_R1_BOUNDED_RESPONSE_RECOVERY",
            "approved_by": approval["approved_by"],
            "approved_utc": approval["approved_utc"],
            "user_response": approval["user_response"],
            "execution_scope": "canary",
            "recipe_id": recipe,
            "parent_acquisition_manifest_sha256": sha256_file(PARENT),
            "execution_subset_manifest_sha256": sha256_file(subset_stage),
            "approved_unit_count": len(units),
            "units": units,
            "diagnosis_labels_used": False,
            "selection_locked": True,
            "selection_source": "exact_technical_subset_of_prior_diagnosis_blind_h04a_canary",
            "proposed_run_root": str(RUN_ROOT),
            "authorized_modes": ["response-calibration-phase-a"],
            "authorized_through": "response_calibration_phase_a_freeze_decision",
            "maximum_cores": 32,
            "minimum_valid_response_calibration_units": 12,
            "minimum_valid_manufacturer_families": 2,
            "minimum_valid_t1_source_classes": 2,
            "required_valid_t1_source_classes": ["dicom_series", "nifti_single"],
            "h04a_wall_clock_stop_hours": 24,
            "h04a_storage_stop_gb": 100,
            "tractography_authorized": False,
            "matrix_generation_authorized": False,
            "full_cohort_authorized": False,
            "fod_reconstruction_authorized": False,
            "tensor_maps_authorized": False,
            "statistical_analysis_authorized": False,
            "dashboard_publication_authorized": False,
            "normative_config_sha256": sha256_file(CONFIG),
            "workflow_source_manifest_sha256": sha256_file(workflow_manifest),
            "environment_contract_sha256": sha256_file(ENVIRONMENT),
            "converter": file_record(CONVERTER),
            "source_approval": file_record(APPROVAL),
            "source_recommendation": file_record(RECOMMENDATION),
            "manufacturer_family_counts": family_counts,
            "t1_source_class_counts": t1_counts,
        }
        decision_stage = temporary / "recovery_execution_decision_v1.json"
        decision_stage.write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validation = {
            "schema_version": "1.0.0",
            "record_type": "h04a_r1_recovery_package_validation",
            "status": "PASS",
            "diagnosis_labels_consulted_for_selection": False,
            "parent_manifest": file_record(PARENT),
            "prior_h04a_subset": file_record(ORIGINAL_SUBSET),
            "source_approval": file_record(APPROVAL),
            "source_recommendation": file_record(RECOMMENDATION),
            "execution_manifest": staged_record(subset_stage, subset_final),
            "execution_decision": staged_record(
                decision_stage, OUTPUT / decision_stage.name
            ),
            "unit_count": len(units),
            "units": units,
            "manufacturer_family_counts": family_counts,
            "t1_source_class_counts": t1_counts,
            "converter": file_record(CONVERTER),
            "run_root": str(RUN_ROOT),
            "hard_stop": "response_calibration_phase_a_freeze_decision",
        }
        validation_stage = temporary / "recovery_package_validation_v1.json"
        validation_stage.write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, OUTPUT)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(load_json(OUTPUT / "recovery_package_validation_v1.json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
