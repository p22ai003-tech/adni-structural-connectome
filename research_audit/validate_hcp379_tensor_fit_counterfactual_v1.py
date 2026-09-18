#!/home/ec2-user/fsl/bin/python
"""Independently validate the bounded HCP379 tensor-fit diagnostic."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE = Path(
    "/home/ec2-user/exp/research_audit/"
    "run_hcp379_tensor_fit_counterfactual_v1.py"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_tensor_fit_counterfactual_v1/validation.json"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_tensor_fit_counterfactual_validation_target",
        SOURCE,
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
        [str(module.EXP.parent / "fsl/bin/python"), "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    self_test = subprocess.run(
        [
            str(module.EXP.parent / "fsl/bin/python"),
            str(SOURCE),
            "--self-test",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    check(
        "source_compiles_and_self_test_passes",
        (
            compile_probe.returncode == 0
            and self_test.returncode == 0
            and "HCP379_TENSOR_FIT_COUNTERFACTUAL_SELF_TEST_PASS"
            in self_test.stdout
        ),
        {
            "compile_returncode": compile_probe.returncode,
            "compile_stderr": compile_probe.stderr,
            "self_test_returncode": self_test.returncode,
            "self_test_stdout": self_test.stdout,
        },
    )

    preflight = module.preflight()
    subjects = preflight.get("subjects", [])
    subject_units = [
        row.get("unit") for row in subjects if isinstance(row, dict)
    ]
    check(
        "exact_target_and_deterministic_acquisition_matched_control",
        (
            preflight.get("status")
            == "READY_BOUNDED_TENSOR_FIT_DIAGNOSTIC"
            and subject_units
            == [module.TARGET_UNIT, module.CONTROL_UNIT]
            and len(subjects) == 2
            and subjects[0].get("acquisition")
            == subjects[1].get("acquisition")
            and subjects[0].get("role")
            == "isolated_current_raw_tensor_failure"
            and subjects[1].get("role")
            == "acquisition_matched_recovery4_pass_control"
        ),
        {
            "status": preflight.get("status"),
            "subject_units": subject_units,
            "acquisition_match": (
                len(subjects) == 2
                and subjects[0].get("acquisition")
                == subjects[1].get("acquisition")
            ),
        },
    )
    methods = preflight.get("methods", [])
    method_names = [
        row.get("name") for row in methods if isinstance(row, dict)
    ]
    check(
        "five_fixed_tensor_fits_include_production_replay",
        (
            method_names
            == [
                "wls_iwls2_replay",
                "wls_initial",
                "wls_iwls4",
                "ols_initial",
                "ols_iwls2",
            ]
            and sum(
                row.get("production_replay") is True
                for row in methods
                if isinstance(row, dict)
            )
            == 1
        ),
        {"method_names": method_names},
    )
    policy = preflight.get("policy", {})
    check(
        "unchanged_gate_and_no_subject_specific_promotion",
        (
            preflight.get("raw_anomaly_fraction_gate") == 0.01
            and policy.get("per_subject_method_tuning_allowed") is False
            and policy.get("qc_threshold_relaxation_allowed") is False
            and policy.get("production_promotion_allowed") is False
            and "both target and control"
            in str(policy.get("candidate_followup_rule", ""))
            and "broader uniform-policy canary"
            in str(policy.get("candidate_followup_rule", ""))
        ),
        {
            "raw_anomaly_fraction_gate": preflight.get(
                "raw_anomaly_fraction_gate"
            ),
            "policy": policy,
        },
    )
    check(
        "preflight_is_diagnosis_blind_nonexecuting_and_nonoverwriting",
        (
            preflight.get("diagnosis_labels_used") is False
            and preflight.get("outcomes_used") is False
            and preflight.get("connectome_density_used") is False
            and preflight.get("non_overwriting") is True
            and preflight.get("imaging_executed") is False
            and preflight.get("production_outputs_modified") is False
        ),
        {
            key: preflight.get(key)
            for key in (
                "diagnosis_labels_used",
                "outcomes_used",
                "connectome_density_used",
                "non_overwriting",
                "imaging_executed",
                "production_outputs_modified",
            )
        },
    )
    records = preflight.get("records", {})
    check(
        "all_inputs_and_tools_are_hash_bound",
        (
            isinstance(records, dict)
            and records.get("runner") == module.file_record(SOURCE)
            and records.get("input_audit")
            == module.file_record(module.INPUT_AUDIT)
            and records.get("failure_ledger")
            == module.file_record(module.FAILURE_LEDGER)
            and records.get("dwi2tensor")
            == module.file_record(module.MRTRIX / "dwi2tensor")
            and records.get("tensor2metric")
            == module.file_record(module.MRTRIX / "tensor2metric")
        ),
        {
            "record_names": sorted(records)
            if isinstance(records, dict)
            else [],
        },
    )

    completed_attempts = sorted(
        (
            path
            for path in module.ATTEMPTS.glob(
                "*-tensor-fit-counterfactual"
            )
            if (path / "attempt.json").is_file()
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if completed_attempts:
        latest_path = completed_attempts[-1]
        latest = module.load_json(latest_path / "attempt.json")
        results = latest.get("results", [])
        keys = {
            (row.get("unit"), row.get("method", {}).get("name"))
            for row in results
            if isinstance(row, dict)
        }
        expected_keys = {
            (unit, method["name"])
            for unit in (module.TARGET_UNIT, module.CONTROL_UNIT)
            for method in module.METHODS
        }
        artifact_errors: list[str] = []
        replayed_qc: dict[tuple[str, str], dict[str, Any]] = {}
        for row in results if isinstance(results, list) else []:
            if not isinstance(row, dict):
                artifact_errors.append("result_type")
                continue
            unit = str(row.get("unit", ""))
            method = str(row.get("method", {}).get("name", ""))
            artifacts = row.get("artifacts", {})
            if not isinstance(artifacts, dict):
                artifact_errors.append(f"{unit}:{method}:artifacts")
                continue
            try:
                for name, record in artifacts.items():
                    if not isinstance(record, dict):
                        raise TypeError(name)
                    if module.file_record(
                        Path(str(record.get("path", "")))
                    ) != record:
                        raise ValueError(name)
                maps = {
                    metric: Path(str(artifacts[metric]["path"]))
                    for metric in ("fa", "md", "rd", "ad")
                }
                mask = Path(str(row["inputs"]["mask"]["path"]))
                replayed_qc[(unit, method)] = module.scalar_qc(
                    maps, mask
                )
            except Exception as exc:
                artifact_errors.append(
                    f"{unit}:{method}:{type(exc).__name__}:{exc}"
                )
        check(
            "latest_attempt_has_all_ten_hash_valid_outputs",
            (
                latest.get("status") == "PASS"
                and latest.get("production_outputs_modified") is False
                and latest.get("completed_method_n") == 10
                and len(results) == 10
                and keys == expected_keys
                and not artifact_errors
            ),
            {
                "attempt": str(latest_path),
                "status": latest.get("status"),
                "completed_method_n": latest.get(
                    "completed_method_n"
                ),
                "observed_key_n": len(keys),
                "errors": artifact_errors,
            },
        )
        qc_errors: list[str] = []
        for row in results if isinstance(results, list) else []:
            if not isinstance(row, dict):
                continue
            key = (
                str(row.get("unit", "")),
                str(row.get("method", {}).get("name", "")),
            )
            if replayed_qc.get(key) != row.get("qc"):
                qc_errors.append(f"{key[0]}:{key[1]}")
        eligible = [
            str(method["name"])
            for method in module.METHODS
            if not method["production_replay"]
            and all(
                replayed_qc.get((unit, str(method["name"])), {}).get(
                    "status"
                )
                == "PASS"
                for unit in (module.TARGET_UNIT, module.CONTROL_UNIT)
            )
        ]
        check(
            "all_eight_anomaly_conditions_and_decision_replay",
            (
                not qc_errors
                and eligible == latest.get("eligible_followup_methods")
                and latest.get("decision")
                in {
                    "UNIFORM_POLICY_CANARY_REQUIRED_BEFORE_PROMOTION",
                    "NO_TENSOR_FIT_VARIANT_RECOVERS_TARGET_AND_CONTROL",
                    "INCONCLUSIVE_PRODUCTION_REPLAY_MISMATCH",
                }
            ),
            {
                "attempt": str(latest_path),
                "qc_mismatches": qc_errors,
                "replayed_eligible_methods": eligible,
                "recorded_eligible_methods": latest.get(
                    "eligible_followup_methods"
                ),
                "decision": latest.get("decision"),
            },
        )
    else:
        check(
            "execution_state_is_explicit",
            True,
            {
                "status": "NOT_STARTED",
                "note": (
                    "Preflight validation only; rerun this validator after "
                    "the bounded execution completes."
                ),
            },
        )

    passed = sum(row["status"] == "PASS" for row in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_tensor_fit_counterfactual_validation",
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
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
