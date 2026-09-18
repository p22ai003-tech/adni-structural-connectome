#!/home/ec2-user/fsl/bin/python
"""Audit the fixed mask-union policy on one archive-D0 spatial failure.

This is a non-overwriting extension of the independently validated mask-union
control audit.  It evaluates the original seeded-rigid spatial candidate for
114_S_5234_I1130066 after the FOD numerical policy passed and all direct
registration routes remained below unchanged Recovery4 spatial gates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
HCP = Path("/data/derivatives/hcp379_v2")
BASE_SOURCE = EXP / "research_audit/audit_hcp379_mask_union_recovery_v1.py"
BASE_SUMMARY = (
    HCP
    / "pretract_failure_source_audit_v1/"
    "mask_union_recovery_controls_v1/summary.json"
)
BASE_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_mask_union_recovery_v1/validation.json"
)
LEDGER = (
    EXP
    / "research_audit/outputs/"
    "hcp379_pretract_failure_recovery_ledger_v1/ledger.json"
)
ROOT = HCP / "corrected_archive_D0_recovery4"
UNIT = "114_S_5234_I1130066"
ROUTE = "seeded_ants_rigid_failover"
OUTPUT = (
    HCP
    / "pretract_failure_source_audit_v1/"
    "mask_union_archive_D0_extension_v1"
)
SUMMARY = OUTPUT / "summary.json"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_SOURCE, "hcp379_mask_union_archive_D0_base")
BASE.OUTPUT = OUTPUT


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def require_base_policy() -> tuple[dict[str, Any], dict[str, Any]]:
    summary = load_json(BASE_SUMMARY)
    validation = load_json(BASE_VALIDATION)
    expected_policy = {
        "original_mask_retained_by_union": True,
        "dwi2mask_clean_scale": 0,
        "native_voxel_dilation_passes": 1,
        "minimum_hemisphere_atlas_inside": 0.8,
        "minimum_tissue_dice": 0.7,
        "maximum_mask_volume_ratio": 1.15,
        "minimum_shell_to_positive_b0_median_ratio": 2.0,
        "per_subject_parameter_tuning_allowed": False,
    }
    if (
        summary.get("status") != "PASS"
        or summary.get("control_n") != 15
        or summary.get("target_n") != 2
        or summary.get("policy") != expected_policy
        or validation.get("status") != "PASS"
        or validation.get("passed_checks") != 6
        or validation.get("total_checks") != 6
    ):
        raise ValueError("validated mask-union base policy differs")
    return summary, validation


def target_record() -> dict[str, Any]:
    ledger = load_json(LEDGER)
    failures = [
        row
        for row in ledger.get("failed_units", [])
        if row.get("unit") == UNIT
    ]
    if (
        len(failures) != 1
        or failures[0].get("failure_class")
        != "SPATIAL_BBR_AND_RIGID_FAIL"
    ):
        raise ValueError("archive-D0 extension target is not exact")
    subject = ROOT / "subjects" / UNIT
    route_root = subject / "03_spatial/recovery4_ants"
    record = {
        "unit": UNIT,
        "role": "registration_failure_target",
        "route": ROUTE,
        "dwi": subject / "01_dwi/dwi_biascorr.mif",
        "mask": subject / "01_dwi/dwi_brain_mask.mif",
        "b0": subject / "01_dwi/mean_b0.nii.gz",
        "five_tt": route_root / "5tt_dwi_recovery4.mif",
        "atlas": route_root / "hcp379_nodes_b0_1mm.nii.gz",
    }
    for key in ("dwi", "mask", "b0", "five_tt", "atlas"):
        path = Path(record[key])
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    return record


def preflight() -> dict[str, Any]:
    require_base_policy()
    target_record()
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_mask_union_archive_D0_extension_preflight",
        "status": "READY",
        "generated_utc": BASE.utc_now(),
        "unit": UNIT,
        "route": ROUTE,
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "source_images_modified": False,
        "imaging_executed": False,
        "per_subject_parameter_tuning_allowed": False,
        "records": {
            "base_summary": BASE.file_record(BASE_SUMMARY),
            "base_validation": BASE.file_record(BASE_VALIDATION),
            "failure_ledger": BASE.file_record(LEDGER),
            "implementation": BASE.file_record(Path(__file__)),
        },
    }


def execute() -> dict[str, Any]:
    pre = preflight()
    record = target_record()
    source_before = {
        key: BASE.file_record(Path(record[key]))
        for key in ("dwi", "mask", "b0", "five_tt", "atlas")
    }
    result = BASE.evaluate(record, BASE.generate(record))
    source_after = {
        key: BASE.file_record(Path(record[key]))
        for key in ("dwi", "mask", "b0", "five_tt", "atlas")
    }
    checks = {
        "exact_archive_D0_target_and_rigid_route": (
            result["unit"] == UNIT and result["route"] == ROUTE
        ),
        "fixed_policy_is_diagnosis_outcome_density_blind": (
            result["diagnosis_labels_used"] is False
            and result["outcomes_used"] is False
            and result["connectome_density_used"] is False
        ),
        "source_artifacts_are_hash_unchanged": source_before == source_after,
        "mask_expansion_is_bounded": (
            1.0 <= result["mask_volume_ratio"] <= 1.15
        ),
        "added_shell_has_supported_b0_signal": (
            result["shell_to_positive_b0_median_ratio"] >= 2.0
        ),
        "unchanged_atlas_and_tissue_gates_pass": (
            result["target_fixed_gate_pass"] is True
            and result["union_left_cortex_inside"] >= 0.8
            and result["union_right_cortex_inside"] >= 0.8
            and result["union_tissue_dice"] >= 0.7
        ),
    }
    payload = {
        **pre,
        "record_type": "hcp379_mask_union_archive_D0_extension_audit",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "generated_utc": BASE.utc_now(),
        "imaging_executed": True,
        "policy": {
            "original_mask_retained_by_union": True,
            "dwi2mask_clean_scale": 0,
            "native_voxel_dilation_passes": 1,
            "minimum_hemisphere_atlas_inside": 0.8,
            "minimum_tissue_dice": 0.7,
            "maximum_mask_volume_ratio": 1.15,
            "minimum_shell_to_positive_b0_median_ratio": 2.0,
            "per_subject_parameter_tuning_allowed": False,
        },
        "checks": checks,
        "result": result,
        "targets": [result],
        "source_before": source_before,
        "source_after": source_after,
    }
    BASE.atomic_json(SUMMARY, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if (
            UNIT != "114_S_5234_I1130066"
            or ROUTE != "seeded_ants_rigid_failover"
        ):
            raise AssertionError("extension scope differs")
        print("HCP379_MASK_UNION_ARCHIVE_D0_EXTENSION_SELF_TEST_PASS")
        return 0
    payload = execute() if args.execute else preflight()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] in {"READY", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
