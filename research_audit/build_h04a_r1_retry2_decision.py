#!/usr/bin/env python3
"""Bind retry2 to the unchanged R1 scope after two no-imaging failures."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
PACKAGE = ROOT / "research_audit/outputs/h04a_r1_recovery_package_v1"
SOURCE = PACKAGE / "recovery_execution_decision_retry1.json"
OUTPUT = PACKAGE / "recovery_execution_decision_retry2.json"
INITIAL_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260718_v1")
RETRY1_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260718_retry1")
RETRY2_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260718_retry2")
INITIAL_END = INITIAL_ROOT / "attempts/20260718T155834.637862Z-response-calibration-phase-a-6d38ddb2.end.json"
RETRY1_END = RETRY1_ROOT / "attempts/20260718T160216.020945Z-response-calibration-phase-a-194c6282.end.json"
INITIAL_COMPLETION = INITIAL_ROOT / "publication/response_calibration_phase_a_completion.json"
RETRY1_COMPLETION = RETRY1_ROOT / "publication/response_calibration_phase_a_completion.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_failed_attempt(path: Path, *, expected_duration: float) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    duration = float(record.get("duration_seconds", -1))
    if (
        record.get("status") != "FAIL"
        or record.get("returncode") != 1
        or not math.isclose(duration, expected_duration, rel_tol=0, abs_tol=1e-6)
    ):
        raise ValueError(f"closed retry evidence differs: {path}")
    return record


def assert_no_subject_imaging(root: Path) -> None:
    allowed_name = "response_calibration_outcome.json"
    unexpected = [
        path
        for path in (root / "subjects").rglob("*")
        if path.is_file() and path.name != allowed_name
    ]
    if unexpected:
        raise ValueError(
            "retry evidence unexpectedly contains subject imaging outputs: "
            + ", ".join(str(path) for path in unexpected[:10])
        )


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"immutable retry2 decision exists: {OUTPUT}")
    initial = load_failed_attempt(INITIAL_END, expected_duration=0.635717)
    retry1 = load_failed_attempt(RETRY1_END, expected_duration=29.269980)
    if not INITIAL_COMPLETION.is_file() or not RETRY1_COMPLETION.is_file():
        raise ValueError("closed retry completion evidence is incomplete")
    assert_no_subject_imaging(INITIAL_ROOT)
    assert_no_subject_imaging(RETRY1_ROOT)

    decision = json.loads(SOURCE.read_text(encoding="utf-8"))
    previous_retry = decision.get("execution_retry")
    charged = float(initial["duration_seconds"]) + float(retry1["duration_seconds"])
    decision["proposed_run_root"] = str(RETRY2_ROOT)
    decision["execution_retry"] = {
        "retry_id": "SL-H04A-R1-RETRY2",
        "authorization_change": False,
        "scientific_parameter_change": False,
        "unit_set_change": False,
        "resource_cap_change": False,
        "reason": "R1 recovery authorization was compared against the inherited base-canary authorization profile",
        "imaging_jobs_started_in_prior_attempts": False,
        "prior_duration_seconds_charged": charged,
        "prior_attempt_ends": [file_record(INITIAL_END), file_record(RETRY1_END)],
        "prior_completions": [
            file_record(INITIAL_COMPLETION),
            file_record(RETRY1_COMPLETION),
        ],
        "prior_run_roots_preserved": [str(INITIAL_ROOT), str(RETRY1_ROOT)],
        "previous_retry_record": previous_retry,
        "corrective_action": "use an R1-specific preflight rule while preserving every locked base scientific and environment check",
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
