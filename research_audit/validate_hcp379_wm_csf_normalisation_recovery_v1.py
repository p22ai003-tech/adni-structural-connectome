#!/usr/bin/env python3
"""Validate the exact degenerate-GM WM+CSF normalisation recovery."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "run_hcp379_wm_csf_normalisation_recovery_v1.py"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_wm_csf_normalisation_recovery_v1/validation.json"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_wm_csf_recovery_validation_target", SOURCE
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

    compile_probe = subprocess.run(
        [str(module.PYTHON), "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    self_test = subprocess.run(
        [str(module.PYTHON), str(SOURCE), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    check(
        "source_compiles_and_self_test_passes",
        compile_probe.returncode == 0
        and self_test.returncode == 0
        and "HCP379_WM_CSF_NORMALISATION_RECOVERY_SELF_TEST_PASS"
        in self_test.stdout,
        {
            "compile_returncode": compile_probe.returncode,
            "compile_stderr": compile_probe.stderr,
            "self_test_returncode": self_test.returncode,
            "self_test_stdout": self_test.stdout,
        },
    )
    preflight = module.preflight(module.DEFAULT_HUMAN_QC)
    check(
        "exact_single_ledger_target_is_degenerate_gm",
        (
            preflight.get("status")
            == "READY_FOR_EXACT_WM_CSF_NORMALISATION_RECOVERY"
            and preflight.get("failure_class") == module.FAILURE_CLASS
            and preflight.get("target_n") == 1
            and max(
                abs(float(preflight["raw_gm_range"]["min"])),
                abs(float(preflight["raw_gm_range"]["max"])),
            )
            <= module.GM_DEGENERATE_ABSOLUTE_MAXIMUM
        ),
        {
            "target_unit": preflight.get("target_unit"),
            "raw_gm_range": preflight.get("raw_gm_range"),
        },
    )
    policy = preflight.get("policy", {})
    check(
        "normalisation_policy_is_fixed_and_does_not_fabricate_gm",
        (
            policy.get("normalised_tissues") == ["wmfod", "csf"]
            and policy.get("excluded_balance_tissue") == "gm"
            and policy.get("gm_spatial_field_application")
            == "gm_raw / shared_norm_field"
            and policy.get("balanced_output_option_used") is False
            and policy.get("per_subject_parameter_tuning_allowed") is False
            and policy.get("scalar_qc_threshold_change_allowed") is False
            and policy.get("spatial_qc_threshold_change_allowed") is False
        ),
        policy,
    )
    source_text = SOURCE.read_text(encoding="utf-8")
    check(
        "execution_is_atomic_preserves_state_and_replays_standard_qc",
        all(
            fragment in source_text
            for fragment in (
                "-check_norm",
                "-check_factors",
                "move_new",
                "preserved_prior_states",
                "validate_reusable_pass_engine_state",
                "reuse_valid_engine=True",
                "PASS_COMPACTED",
            )
        ),
        {"required_fragments_present": True},
    )
    check(
        "preflight_is_non_imaging_and_diagnosis_blind",
        (
            preflight.get("diagnosis_labels_used") is False
            and preflight.get("outcomes_used") is False
            and preflight.get("connectome_density_used") is False
            and preflight.get("non_overwriting") is True
            and preflight.get("imaging_executed") is False
            and preflight.get("tractography_generated") is False
            and preflight.get("matrix_generated") is False
        ),
        {
            key: preflight.get(key)
            for key in (
                "diagnosis_labels_used",
                "outcomes_used",
                "connectome_density_used",
                "non_overwriting",
                "imaging_executed",
                "tractography_generated",
                "matrix_generated",
            )
        },
    )
    passed = sum(row["status"] == "PASS" for row in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_wm_csf_normalisation_recovery_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
