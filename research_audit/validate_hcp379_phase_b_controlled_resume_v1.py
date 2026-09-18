#!/home/ec2-user/fsl/bin/python
"""Validate the narrowly scoped Phase-B recovery resume contract."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "run_hcp379_phase_b_stability_v2.py"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_phase_b_controlled_resume_v1/validation.json"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module(SOURCE, "hcp379_phase_b_resume_validator")
    checks: dict[str, Any] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    compile_probe = subprocess.run(
        [str(PYTHON), "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "source_compiles",
        compile_probe.returncode == 0,
        {
            "returncode": compile_probe.returncode,
            "stderr": compile_probe.stderr[-2000:],
        },
    )
    roots = tuple(
        str(path.resolve())
        for path in module.ALLOWED_CONCURRENT_PRETRACT_ROOTS
    )
    expected_suffixes = {
        "corrected_scaleup_recovery4",
        "corrected_legacy_tensor_recovery4",
        "archive_corrected_calibration_recovery4",
        "corrected_archive_low_recovery4",
        "corrected_archive_D0_recovery4",
        "corrected_freesurfer_recovery4",
    }
    check(
        "allowlist_is_exactly_the_six_recovery4_pretract_roots",
        len(roots) == 6
        and {Path(root).name for root in roots} == expected_suffixes,
        {"roots": roots},
    )
    synthetic = [
        f"{1000 + index} ss3t_csd_beta1 {root}/subjects/probe"
        for index, root in enumerate(roots)
    ]
    allowed = module.resume_process_contract(synthetic)
    check(
        "known_pretract_processes_are_allowed_with_failed_attempt_evidence",
        (
            allowed.get("status") == "PASS_CONTROLLED_PHASE_B_RESUME"
            and len(
                allowed.get(
                    "allowed_concurrent_pretract_processes", []
                )
            )
            == 6
            and allowed.get("unexpected_imaging_processes") == []
            and len(allowed.get("prior_failed_attempts", [])) > 0
        ),
        allowed,
    )
    unexpected_rejected = False
    try:
        module.resume_process_contract(
            synthetic + ["9999 tckgen /tmp/unbound-phase"]
        )
    except RuntimeError:
        unexpected_rejected = True
    check(
        "unbound_or_duplicate_imaging_process_is_rejected",
        unexpected_rejected,
        {"rejected": unexpected_rejected},
    )
    text = SOURCE.read_text(encoding="utf-8")
    required = (
        "--resume-existing-phase-b",
        "resume_process_contract(active)",
        "PASS_CONTROLLED_PHASE_B_RESUME",
        "prior_failed_attempts",
        "--rerun-incomplete",
        "--rerun-triggers",
        "--keep-going",
        "FORBIDDEN_SCHEDULED_RULES",
        "EMERGENCY_FREE_BYTES",
    )
    missing = [fragment for fragment in required if fragment not in text]
    check(
        "resume_retains_dry_run_scope_and_storage_guards",
        not missing,
        {"missing": missing},
    )
    passed = sum(row["status"] == "PASS" for row in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_phase_b_controlled_resume_validation",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "imaging_executed": False,
        "source": module.file_record(SOURCE),
    }
    module.atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
