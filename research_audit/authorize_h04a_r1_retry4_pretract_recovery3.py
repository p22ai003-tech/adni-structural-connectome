#!/usr/bin/env python3
"""Release the exact retry4 pretract Recovery3 package after user approval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path("/home/ec2-user/exp")
AUDIT = ROOT / "research_audit"
PACKAGE = AUDIT / "outputs/h04a_r1_retry4_pretract_recovery3_package_v1"
VALIDATION = PACKAGE / "validation.json"
DECISION_CANDIDATE = PACKAGE / "recovery3_decision_proposed.json"
DECISION_APPROVED = PACKAGE / "recovery3_decision_approved.json"
EXECUTION_BINDING = PACKAGE / "recovery3_execution_binding.json"
RELEASE_VALIDATION = PACKAGE / "approval_release_validation.json"
REQUIRED_USER_RESPONSE = "Approve SL-H04A-CAL-R3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular evidence file: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-response", required=True)
    parser.add_argument("--approved-by", default="project_user_via_codex_chat")
    parser.add_argument("--approved-utc")
    args = parser.parse_args()
    if args.user_response.strip() != REQUIRED_USER_RESPONSE:
        raise PermissionError(f"exact approval required: {REQUIRED_USER_RESPONSE!r}")
    approved_by = args.approved_by.strip()
    if not approved_by:
        raise ValueError("--approved-by cannot be blank")
    approved_utc = args.approved_utc or datetime.now(timezone.utc).isoformat()
    parsed = datetime.fromisoformat(approved_utc.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--approved-utc must include a timezone")
    outputs = (DECISION_APPROVED, EXECUTION_BINDING, RELEASE_VALIDATION)
    if any(path.exists() for path in outputs):
        raise FileExistsError("Recovery3 approval release already exists or is incomplete")

    validation = load_json(VALIDATION)
    if (
        validation.get("status") != "PASS"
        or validation.get("unit_count") != 15
        or validation.get("imaging_executed") is not False
        or validation.get("scheduled_job_counts", {}).get("total") != 166
        or validation.get("forbidden_scheduled_rules") != []
        or validation.get("next_gate") != "SL-H04A-CAL-R3"
        or validation.get("required_user_response") != REQUIRED_USER_RESPONSE
    ):
        raise ValueError("Recovery3 package is not the exact passing release")
    for record in validation.get("evidence", {}).values():
        if not isinstance(record, dict) or file_record(Path(str(record.get("path", "")))) != record:
            raise ValueError(f"Recovery3 package evidence drifted: {record}")

    candidate = load_json(DECISION_CANDIDATE)
    if (
        candidate.get("status") != "AWAITING_USER_APPROVAL"
        or candidate.get("decision_gate") != "SL-H04A-CAL-R3"
        or candidate.get("required_user_response") != REQUIRED_USER_RESPONSE
        or candidate.get("tractography_authorized") is not False
        or candidate.get("matrix_generation_authorized") is not False
        or candidate.get("full_cohort_authorized") is not False
    ):
        raise ValueError("Recovery3 decision candidate differs")
    for record in candidate.get("source_files", {}).values():
        if file_record(Path(str(record["path"]))) != record:
            raise ValueError(f"Recovery3 source drifted: {record.get('path')}")

    approved = {
        **candidate,
        "status": "APPROVED",
        "approved_by": approved_by,
        "approved_utc": approved_utc,
        "user_response": REQUIRED_USER_RESPONSE,
        "package_validation": file_record(VALIDATION),
        "authorization_builder": file_record(Path(__file__)),
    }
    write_immutable_json(DECISION_APPROVED, approved)
    binding = {
        "schema_version": "1.0.0",
        "record_type": "retry4_pretract_recovery3_execution_binding",
        "status": "LOCKED",
        "run_root": candidate["run_root"],
        "units": candidate["units"],
        "recovery3_decision": file_record(DECISION_APPROVED),
        "prior_recovery2_completion": candidate["prior_recovery2_completion"],
        "prior_resolved_dag_config": candidate["prior_resolved_dag_config"],
        "response_calibration_binding": candidate["response_calibration_binding"],
        "pretract_extension_binding": candidate["pretract_extension_binding"],
        "converter_contract": candidate["converter_contract"],
        "reusable_output_inventory": candidate["reusable_output_inventory"],
        "source_files": candidate["source_files"],
        "maximum_cpu_cores": candidate["maximum_cpu_cores"],
        "maximum_cumulative_wall_clock_hours": candidate["maximum_cumulative_wall_clock_hours"],
        "total_run_root_storage_stop_bytes": candidate["total_run_root_storage_stop_bytes"],
        "imaging_execution_authorized": True,
        "diagnosis_labels_used": False,
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
        "statistical_analysis_authorized": False,
        "dashboard_publication_authorized": False,
        "full_cohort_authorized": False,
        "next_human_gate": "SL-H04B",
    }
    write_immutable_json(EXECUTION_BINDING, binding)
    release = {
        "schema_version": "1.0.0",
        "record_type": "retry4_pretract_recovery3_approval_release_validation",
        "status": "PASS",
        "approved_utc": approved_utc,
        "user_response": REQUIRED_USER_RESPONSE,
        "package_validation": file_record(VALIDATION),
        "recovery3_decision": file_record(DECISION_APPROVED),
        "recovery3_execution_binding": file_record(EXECUTION_BINDING),
        "imaging_executed": False,
    }
    write_immutable_json(RELEASE_VALIDATION, release)
    print(json.dumps(release, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
