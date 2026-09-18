#!/usr/bin/env python3
"""Release the retry4 pre-tractography binding after exact SL-H04A-CAL approval.

This command creates immutable approval/binding records only.  It never invokes
Snakemake or an imaging executable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path("/home/ec2-user/exp")
AUDIT = ROOT / "research_audit"
WORKFLOW = ROOT / "scforge/workflow"
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
PACKAGE = AUDIT / "outputs/h04a_r1_retry4_pretract_package_v3"
VALIDATION = PACKAGE / "validation.json"
RESPONSE_CANDIDATE = PACKAGE / "response_calibration_decision_proposed.json"
PRETRACT_CANDIDATE = PACKAGE / "pretract_extension_decision_proposed.json"
RESPONSE_APPROVED = PACKAGE / "response_calibration_decision_approved.json"
PRETRACT_APPROVED = PACKAGE / "pretract_extension_decision_approved.json"
RESPONSE_BINDING = PACKAGE / "response_calibration_binding.json"
PRETRACT_BINDING = PACKAGE / "pretract_execution_extension_binding.json"
RELEASE_VALIDATION = PACKAGE / "approval_release_validation.json"
PHASE_A_COMPLETION = RUN_ROOT / "publication/response_calibration_phase_a_completion.json"
FROZEN_MANIFEST = (
    RUN_ROOT
    / "frozen_calibration_retry4_v2/frozen_response_calibration_manifest.json"
)
FREEZE_ATTESTATION = (
    RUN_ROOT / "frozen_calibration_retry4_v2/retry4_freeze_attestation.json"
)
REQUIRED_USER_RESPONSE = "Approve SL-H04A-CAL"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular evidence file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


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
    parser.add_argument(
        "--approved-by", default="project_user_via_codex_chat"
    )
    parser.add_argument("--approved-utc")
    args = parser.parse_args()
    if args.user_response.strip() != REQUIRED_USER_RESPONSE:
        raise PermissionError(
            f"exact approval required: {REQUIRED_USER_RESPONSE!r}"
        )
    approved_by = args.approved_by.strip()
    if not approved_by:
        raise ValueError("--approved-by cannot be blank")
    approved_utc = args.approved_utc or datetime.now(timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(approved_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("--approved-utc must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("--approved-utc must include a timezone")

    outputs = (
        RESPONSE_APPROVED,
        PRETRACT_APPROVED,
        RESPONSE_BINDING,
        PRETRACT_BINDING,
        RELEASE_VALIDATION,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("approval release already exists or is incomplete")
    validation = load_json(VALIDATION)
    if (
        validation.get("status") != "PASS"
        or validation.get("unit_count") != 15
        or validation.get("scheduled_job_counts", {}).get("total") != 394
        or validation.get("forbidden_scheduled_rules") != []
        or validation.get("imaging_executed") is not False
        or validation.get("next_gate") != "SL-H04A-CAL"
        or validation.get("required_user_response") != REQUIRED_USER_RESPONSE
    ):
        raise ValueError("pre-tractography package is not an exact passing release")
    for record in validation.get("evidence", {}).values():
        if not isinstance(record, dict):
            raise TypeError("package evidence must contain file records")
        if file_record(Path(str(record["path"]))) != record:
            raise ValueError(f"package evidence drifted: {record.get('path')}")

    response = load_json(RESPONSE_CANDIDATE)
    pretract = load_json(PRETRACT_CANDIDATE)
    if (
        response.get("status") != "AWAITING_USER_APPROVAL"
        or pretract.get("status") != "AWAITING_USER_APPROVAL"
        or response.get("required_user_response") != REQUIRED_USER_RESPONSE
        or pretract.get("required_user_response") != REQUIRED_USER_RESPONSE
    ):
        raise ValueError("candidate decisions are not awaiting this exact gate")
    for source_record in pretract.get("source_files", {}).values():
        if file_record(Path(str(source_record["path"]))) != source_record:
            raise ValueError(f"pre-tractography source drifted: {source_record['path']}")

    approval_fields = {
        "status": "APPROVED",
        "approved_by": approved_by,
        "approved_utc": approved_utc,
        "user_response": REQUIRED_USER_RESPONSE,
        "dry_run_only_until_approved": False,
        "authorization_builder": file_record(Path(__file__)),
        "package_validation": file_record(VALIDATION),
    }
    approved_response = {**response, **approval_fields}
    approved_pretract = {**pretract, **approval_fields}
    write_immutable_json(RESPONSE_APPROVED, approved_response)
    write_immutable_json(PRETRACT_APPROVED, approved_pretract)

    completion = load_json(PHASE_A_COMPLETION)
    execution_binding = completion.get("execution_binding")
    if not isinstance(execution_binding, dict):
        raise ValueError("phase-A completion lacks execution binding")
    sys.path.insert(0, str(ROOT / "scforge"))
    sys.path.insert(0, str(WORKFLOW))
    import run_h04a_r1_retry4 as retry4
    from freeze_response_calibration_retry4 import (
        load_execution_subset_technical_metadata_retry4,
    )
    from scforge.h04a_r1_retry4_pretract import (
        validate_pretract_extension_binding,
    )

    base = retry4.base
    original_loader = base.load_execution_subset_technical_metadata
    base.load_execution_subset_technical_metadata = (
        load_execution_subset_technical_metadata_retry4
    )
    try:
        response_binding = base.prepare_phase_b_binding(
            recipe_id=completion["recipe_id"],
            phase_a_completion_path=PHASE_A_COMPLETION,
            response_calibration_manifest_path=FROZEN_MANIFEST,
            minimum_valid_subjects=12,
            response_calibration_decision_path=RESPONSE_APPROVED,
            execution_binding=execution_binding,
        )
    finally:
        base.load_execution_subset_technical_metadata = original_loader
    write_immutable_json(RESPONSE_BINDING, response_binding)

    extension_binding = {
        "schema_version": "1.0.0",
        "record_type": "retry4_pretract_execution_extension_binding",
        "status": "LOCKED",
        "run_root": str(RUN_ROOT),
        "units": execution_binding["units"],
        "imaging_execution_authorized": True,
        "pretract_decision": file_record(PRETRACT_APPROVED),
        "response_calibration_decision": file_record(RESPONSE_APPROVED),
        "response_calibration_binding": file_record(RESPONSE_BINDING),
        "phase_a_completion": file_record(PHASE_A_COMPLETION),
        "frozen_response_manifest": file_record(FROZEN_MANIFEST),
        "freeze_attestation": file_record(FREEZE_ATTESTATION),
        "package_validation": file_record(VALIDATION),
        "authorization_builder": file_record(Path(__file__)),
        "diagnosis_labels_used": False,
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
        "statistical_analysis_authorized": False,
        "dashboard_publication_authorized": False,
        "full_cohort_authorized": False,
    }
    validate_pretract_extension_binding(
        extension_binding,
        expected_run_root=RUN_ROOT,
        execution_binding=execution_binding,
    )
    write_immutable_json(PRETRACT_BINDING, extension_binding)
    release = {
        "schema_version": "1.0.0",
        "record_type": "retry4_pretract_approval_release_validation",
        "status": "PASS",
        "approved_utc": approved_utc,
        "user_response": REQUIRED_USER_RESPONSE,
        "package_validation": file_record(VALIDATION),
        "response_decision": file_record(RESPONSE_APPROVED),
        "pretract_decision": file_record(PRETRACT_APPROVED),
        "response_calibration_binding": file_record(RESPONSE_BINDING),
        "pretract_extension_binding": file_record(PRETRACT_BINDING),
        "imaging_executed": False,
    }
    write_immutable_json(RELEASE_VALIDATION, release)
    print(json.dumps(release, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
