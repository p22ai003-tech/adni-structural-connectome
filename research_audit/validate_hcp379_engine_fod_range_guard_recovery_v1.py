#!/home/ec2-user/fsl/bin/python
"""Validate the exact FOD range-guard recovery runner."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "run_hcp379_engine_fod_range_guard_recovery_v1.py"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_engine_fod_range_guard_recovery_v1/validation.json"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_engine_guard_recovery_validation_target", SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module()
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    compiled = subprocess.run(
        [str(module.PYTHON), "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    self_test = subprocess.run(
        [str(module.PYTHON), str(SOURCE), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "source_compiles_and_self_test_passes",
        compiled.returncode == 0
        and self_test.returncode == 0
        and "HCP379_ENGINE_FOD_RANGE_GUARD_RECOVERY_SELF_TEST_PASS"
        in self_test.stdout,
        {
            "compile_returncode": compiled.returncode,
            "compile_stderr": compiled.stderr,
            "self_test_returncode": self_test.returncode,
            "self_test_stdout": self_test.stdout,
        },
    )
    pre = module.preflight(module.DEFAULT_HUMAN_QC)
    check(
        "targets_are_exact_core_ledger_class",
        pre.get("status")
        in {
            "READY_FOR_EXACT_ENGINE_GUARD_REPLAY",
            "NO_CURRENT_ENGINE_GUARD_TARGETS",
        }
        and pre.get("failure_class") == module.FAILURE_CLASS
        and pre.get("target_n") == len(pre.get("target_units", [])),
        {
            "status": pre.get("status"),
            "target_n": pre.get("target_n"),
            "target_units": pre.get("target_units"),
        },
    )
    guard = pre.get("guard", {})
    check(
        "guard_changes_parser_not_recipe_or_qc",
        guard.get("purpose")
        == "aggregate finite extrema across FOD volumes"
        and guard.get("engine_recipe_changed") is False
        and guard.get("recovery4_threshold_changed") is False
        and guard.get("subject_specific_tuning") is False,
        guard,
    )
    text = SOURCE.read_text(encoding="utf-8")
    check(
        "failed_records_are_preserved_and_standard_guards_installed",
        all(
            fragment in text
            for fragment in (
                "preserve(",
                "install_provisional_engine_guards()",
                "reuse_valid_engine=False",
                "PASS_COMPACTED",
            )
        ),
        {},
    )
    check(
        "preflight_is_blind_nonimaging_and_nonoverwriting",
        pre.get("diagnosis_labels_used") is False
        and pre.get("outcomes_used") is False
        and pre.get("connectome_density_used") is False
        and pre.get("non_overwriting") is True
        and pre.get("historical_outputs_modified") is False
        and pre.get("imaging_executed") is False
        and pre.get("tractography_generated") is False
        and pre.get("matrix_generated") is False
        and pre.get("qc_thresholds_changed") is False,
        {
            key: pre.get(key)
            for key in (
                "diagnosis_labels_used",
                "outcomes_used",
                "connectome_density_used",
                "non_overwriting",
                "historical_outputs_modified",
                "imaging_executed",
                "tractography_generated",
                "matrix_generated",
                "qc_thresholds_changed",
            )
        },
    )
    passed = sum(row["status"] == "PASS" for row in checks.values())
    value = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_engine_fod_range_guard_recovery_v1_validation"
        ),
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "imaging_executed": False,
        "checks": checks,
    }
    module.atomic_json(OUTPUT, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
