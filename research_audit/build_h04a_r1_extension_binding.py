#!/usr/bin/env python3
"""Freeze every R1-only executable source into one immutable extension binding."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
AUDIT = ROOT / "research_audit"
PACKAGE = AUDIT / "outputs/h04a_r1_recovery_package_v1"
OUTPUT = AUDIT / "outputs/h04a_r1_recovery_extension_binding_v3.json"
RUN_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260718_retry2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"immutable R1 extension binding exists: {OUTPUT}")
    binding = {
        "schema_version": "1.0.0",
        "record_type": "h04a_r1_recovery_extension_binding",
        "status": "LOCKED",
        "run_root": str(RUN_ROOT),
        "source_files": {
            "snakefile": file_record(ROOT / "scforge/workflow/Snakefile_h04a_r1"),
            "manifest_rule": file_record(ROOT / "scforge/workflow/extensions/h04a_r1/00_manifest.smk"),
            "input_rule": file_record(ROOT / "scforge/workflow/extensions/h04a_r1/00_inputs.smk"),
            "launcher": file_record(ROOT / "scforge/workflow/run_h04a_r1_recovery.py"),
            "compatibility_validator": file_record(ROOT / "scforge/scforge/h04a_r1.py"),
        },
        "converter": file_record(ROOT / "tools/dcm2niix/v1.0.20260416/dcm2niix"),
        "source_approval": file_record(AUDIT / "decisions/sl_h04a_r1_approval_20260718.json"),
        "source_recommendation": file_record(AUDIT / "decisions/sl_h04a_r1_recovery_recommendation_20260718.json"),
        "execution_decision": file_record(PACKAGE / "recovery_execution_decision_retry2.json"),
        "execution_manifest": file_record(PACKAGE / "recovery_execution_manifest_v1.csv"),
        "package_validation": file_record(PACKAGE / "recovery_package_validation_v1.json"),
        "allowed_terminal_stage": "response_calibration_phase_a_freeze_decision",
        "diagnosis_labels_used": False,
        "approved_unit_count": 15,
        "units": [
            "003_S_4118_I1124861",
            "007_S_5196_I390043",
            "009_S_4324_I1186579",
            "013_S_4268_I1075344",
            "014_S_6087_I926924",
            "016_S_4353_I295021",
            "016_S_6816_I10309274",
            "032_S_6804_I1230908",
            "033_S_6889_I10988194",
            "036_S_2380_I1023443",
            "094_S_4737_I326938",
            "126_S_6721_I1439616",
            "129_S_6784_I1482244",
            "135_S_6840_I1263427",
            "168_S_6735_I1175371"
        ],
        "maximum_cpu_cores": 32,
        "maximum_wall_clock_hours": 24,
        "maximum_additional_storage_gib": 100,
        "minimum_valid_response_calibration_units": 12,
        "required_manufacturer_families": 2,
        "required_t1_source_classes": 2,
        "forbidden_stages": [
            "fod_reconstruction",
            "tensor_maps",
            "tractography",
            "sift2",
            "connectome_matrices",
            "statistical_analysis",
            "dashboard_publication"
        ]
    }
    payload = (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": "PASS", "binding": file_record(OUTPUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
