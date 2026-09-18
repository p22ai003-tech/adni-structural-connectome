#!/home/ec2-user/fsl/bin/python
"""Static and hostile validation of the independent release verifier."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP
    / "research_audit/"
    "verify_hcp379_integration_release_v1.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_integration_release_verifier_v1/validation.json"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_integration_release_verifier_contract", SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module()
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if condition else "FAIL",
            "evidence": evidence,
        }

    compile_result = subprocess.run(
        [str(PYTHON), "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "source_compiles",
        compile_result.returncode == 0,
        {
            "returncode": compile_result.returncode,
            "stderr": compile_result.stderr.strip(),
        },
    )

    self_test = subprocess.run(
        [str(PYTHON), str(SOURCE), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
    )
    self_payload = (
        json.loads(self_test.stdout)
        if self_test.returncode == 0 and self_test.stdout.strip()
        else {}
    )
    check(
        "matrix_contract_self_test",
        self_test.returncode == 0
        and self_payload.get("status") == "PASS"
        and self_payload.get("passed") == 6
        and self_payload.get("total") == 6,
        {
            "returncode": self_test.returncode,
            "payload": self_payload,
            "stderr": self_test.stderr.strip(),
        },
    )

    preflight = subprocess.run(
        [str(PYTHON), str(SOURCE), "--preflight"],
        capture_output=True,
        text=True,
        check=False,
    )
    plan = module.load_json(module.PREFLIGHT)
    check(
        "preflight_is_non_verifying",
        preflight.returncode == 0
        and plan.get("record_type")
        == "hcp379_integration_release_verifier_preflight"
        and plan.get("verification_executed") is False
        and plan.get("status")
        in {"AWAITING_RELEASE", "READY_TO_VERIFY"},
        {
            "returncode": preflight.returncode,
            "status": plan.get("status"),
            "stderr": preflight.stderr.strip(),
        },
    )

    source_text = SOURCE.read_text(encoding="utf-8")
    check(
        "independent_of_assembler_imports",
        "build_hcp379_final_release_v2" not in source_text
        and "importlib" not in source_text
        and "load_module" not in source_text,
        {
            "assembler_name_present": (
                "build_hcp379_final_release_v2" in source_text
            ),
            "dynamic_import_present": (
                "importlib" in source_text or "load_module" in source_text
            ),
        },
    )

    check(
        "exact_530_by_9_contract",
        module.EXPECTED_SUBJECTS == 530
        and module.EXPECTED_NODES == 379
        and len(module.MATRIX_NAMES) == 9
        and module.EXPECTED_FILES == 4770
        and module.MINIMUM_DENSITY == 0.60
        and module.MINIMUM_WEIGHTED_SUPPORT == 0.95
        and module.MINIMUM_ASSIGNMENT_FRACTION == 0.50,
        {
            "subjects": module.EXPECTED_SUBJECTS,
            "nodes": module.EXPECTED_NODES,
            "matrices": len(module.MATRIX_NAMES),
            "files": module.EXPECTED_FILES,
            "density": module.MINIMUM_DENSITY,
            "weighted_support": module.MINIMUM_WEIGHTED_SUPPORT,
            "assignment": module.MINIMUM_ASSIGNMENT_FRACTION,
        },
    )

    check(
        "lane_provenance_contract",
        {
            name: expected
            for name, (_, expected, _) in module.LANE_MANIFESTS.items()
        }
        == {
            "core_227": 227,
            "legacy_tensor_216": 216,
            "archive_low_30": 30,
            "archive_D0_56": 56,
            "freesurfer_1": 1,
            "corrected_archive_86": 86,
        },
        {
            name: expected
            for name, (_, expected, _) in module.LANE_MANIFESTS.items()
        },
    )

    check(
        "route_scenarios_are_exact",
        module.EXPECTED_ROUTE_COUNTS
        == {
            module.ARCHIVE_RETAIN_DECISION: {
                "ARCHIVE_NO_ACT_CALIBRATED": 56,
                "CORRECTED_ACT_227": 227,
                "CORRECTED_ACT_ARCHIVE_LOW_30": 30,
                "CORRECTED_ACT_FREESURFER_1": 1,
                "CORRECTED_ACT_LEGACY_TENSOR_216": 216,
            },
            module.ARCHIVE_RERUN_DECISION: {
                "CORRECTED_ACT_227": 227,
                "CORRECTED_ACT_ARCHIVE_RERUN": 86,
                "CORRECTED_ACT_FREESURFER_1": 1,
                "CORRECTED_ACT_LEGACY_TENSOR_216": 216,
            },
        }
        and all(
            sum(value.values()) == 530
            for value in module.EXPECTED_ROUTE_COUNTS.values()
        ),
        module.EXPECTED_ROUTE_COUNTS,
    )

    expected_fields = module.expected_analysis_fields()
    check(
        "analysis_schema_excludes_diagnosis",
        len(expected_fields) == 24
        and not {
            "diagnosis",
            "group",
            "outcome",
            "research_group",
        }
        & {field.lower() for field in expected_fields},
        sorted(expected_fields),
    )

    check(
        "path_and_hash_guards_present",
        "raw_path.is_symlink()" in source_text
        and 'Path("/data/derivatives") not in path.parents' in source_text
        and "RELEASE_ROOT.resolve() in path.parents" in source_text
        and "file_record(path)" in source_text
        and "path in used_paths" in source_text,
        {
            "symlink_guard": "raw_path.is_symlink()" in source_text,
            "derivative_root_guard": (
                'Path("/data/derivatives") not in path.parents'
                in source_text
            ),
            "release_root_guard": (
                "RELEASE_ROOT.resolve() in path.parents" in source_text
            ),
            "rehash": "file_record(path)" in source_text,
            "duplicate_path_guard": "path in used_paths" in source_text,
        },
    )

    default_before = (
        module.OUTPUT.read_bytes() if module.OUTPUT.is_file() else None
    )
    with tempfile.TemporaryDirectory(
        prefix="hcp379-verifier-missing-release."
    ) as temporary:
        failed_output = Path(temporary) / "verification.json"
        missing = subprocess.run(
            [
                str(PYTHON),
                str(SOURCE),
                "--verify",
                "--output",
                str(failed_output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        failed_payload = module.load_json(failed_output)
    default_after = (
        module.OUTPUT.read_bytes() if module.OUTPUT.is_file() else None
    )
    check(
        "missing_release_cannot_pass",
        missing.returncode != 0
        and failed_payload.get("status") == "FAIL"
        and default_after == default_before,
        {
            "returncode": missing.returncode,
            "payload": failed_payload,
            "default_changed": default_after != default_before,
        },
    )

    with tempfile.TemporaryDirectory(
        prefix="hcp379-verifier-nonoverwrite."
    ) as temporary:
        existing = Path(temporary) / "verification.json"
        sentinel = b"do-not-overwrite\n"
        existing.write_bytes(sentinel)
        denied = subprocess.run(
            [
                str(PYTHON),
                str(SOURCE),
                "--verify",
                "--output",
                str(existing),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        preserved = existing.read_bytes()
    check(
        "existing_verification_is_not_overwritten",
        denied.returncode != 0 and preserved == sentinel,
        {
            "returncode": denied.returncode,
            "preserved": preserved == sentinel,
            "stderr_tail": denied.stderr.strip().splitlines()[-3:],
        },
    )

    check(
        "current_default_output_is_not_falsely_present",
        not module.OUTPUT.is_file(),
        {
            "path": str(module.OUTPUT),
            "present": module.OUTPUT.is_file(),
        },
    )

    passed = sum(
        value["status"] == "PASS" for value in checks.values()
    )
    payload = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_integration_release_verifier_package_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": module.utc_now(),
        "release_verification_executed": False,
        "human_visual_qc_inferred": False,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "source": module.file_record(SOURCE),
        "preflight": module.file_record(module.PREFLIGHT),
    }
    module.atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
