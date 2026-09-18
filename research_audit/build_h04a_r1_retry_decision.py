#!/usr/bin/env python3
"""Bind the no-imaging launcher retry to the unchanged user-approved R1 scope."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
PACKAGE = ROOT / "research_audit/outputs/h04a_r1_recovery_package_v1"
SOURCE = PACKAGE / "recovery_execution_decision_v1.json"
OUTPUT = PACKAGE / "recovery_execution_decision_retry1.json"
FAILED_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260718_v1")
FAILED_ATTEMPT = (
    FAILED_ROOT
    / "attempts/20260718T155834.637862Z-response-calibration-phase-a-6d38ddb2.end.json"
)
FAILED_COMPLETION = FAILED_ROOT / "publication/response_calibration_phase_a_completion.json"
RETRY_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260718_retry1")


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
        raise FileExistsError(f"immutable retry decision exists: {OUTPUT}")
    decision = json.loads(SOURCE.read_text(encoding="utf-8"))
    failed = json.loads(FAILED_ATTEMPT.read_text(encoding="utf-8"))
    if (
        failed.get("status") != "FAIL"
        or failed.get("returncode") != 1
        or float(failed.get("duration_seconds", -1)) != 0.635717
        or not FAILED_COMPLETION.is_file()
    ):
        raise ValueError("retry evidence differs from the closed no-imaging attempt")
    decision["proposed_run_root"] = str(RETRY_ROOT)
    decision["execution_retry"] = {
        "retry_id": "SL-H04A-R1-RETRY1",
        "authorization_change": False,
        "scientific_parameter_change": False,
        "unit_set_change": False,
        "resource_cap_change": False,
        "reason": "snakemake_cli_thread_override_consumed_target_before_dag_execution",
        "imaging_jobs_started_in_prior_attempt": False,
        "prior_duration_seconds_charged": 0.635717,
        "prior_attempt_end": file_record(FAILED_ATTEMPT),
        "prior_completion": file_record(FAILED_COMPLETION),
        "prior_run_root_preserved": str(FAILED_ROOT),
    }
    payload = (json.dumps(decision, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": "PASS", "decision": file_record(OUTPUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
