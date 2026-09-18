#!/home/ec2-user/fsl/bin/python
"""Validate the Recovery4-bound four-case keep-route decision package."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_keep_route_concordance_recovery4_v3.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_keep_route_concordance_recovery4_v3/validation.json"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_keep_route_recovery4_validation_target", SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
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

    keep, overlap = module.exact_scope()
    check(
        "exact_214_keep_and_four_calibrators",
        len(keep) == 214 and len(overlap) == 4,
        {"keep_n": len(keep), "calibration_units": overlap},
    )
    plan = module.ENGINE.load_json(module.PLAN)
    check(
        "plan_is_nonexecuting",
        plan.get("status") == "PRE_PHASE_DRY_RUN_PASS"
        and plan.get("execution_authorized") is False
        and plan.get("imaging_executed_by_plan") is False
        and plan.get("selected_streamline_count") is None,
        {
            "status": plan.get("status"),
            "execution_authorized": plan.get("execution_authorized"),
            "selected_streamline_count": plan.get(
                "selected_streamline_count"
            ),
        },
    )
    check(
        "plan_scope_accounting",
        plan.get("keep_subject_count") == 214
        and plan.get("calibration_unit_count") == 4
        and plan.get("remaining_keep_unit_count") == 210
        and plan.get("failure_rerun_count") == 213
        and plan.get("calibration_units") == overlap,
        {
            key: plan.get(key)
            for key in (
                "keep_subject_count",
                "calibration_unit_count",
                "remaining_keep_unit_count",
                "failure_rerun_count",
            )
        },
    )
    check(
        "exact_nine_matrix_families",
        tuple(plan.get("matrix_names", ()))
        == tuple(module.ENGINE.MATRIX_NAMES)
        and len(plan.get("matrix_names", ())) == 9,
        plan.get("matrix_names"),
    )
    matrix_records = [
        record
        for unit_records in plan.get(
            "legacy_matrix_records", {}
        ).values()
        for record in unit_records.values()
    ]
    matrix_bindings_pass = len(matrix_records) == 36
    if matrix_bindings_pass:
        try:
            for record in matrix_records:
                if module.ENGINE.file_record(
                    Path(str(record["path"]))
                ) != record:
                    matrix_bindings_pass = False
                    break
        except Exception:
            matrix_bindings_pass = False
    check(
        "all_36_legacy_matrices_are_hash_bound",
        matrix_bindings_pass,
        {"matrix_record_count": len(matrix_records)},
    )
    phase = plan.get("phase_b_status", {})
    check(
        "phase_b_result_is_not_fabricated",
        all(
            not value.get("present")
            and value.get("status") == "NOT_RUN"
            for value in phase.values()
        ),
        phase,
    )
    contract = module.ENGINE.load_json(module.ENGINE.CONTRACT)
    raw = contract["diagnosis_blind_recipe_stability_gate"]
    expected_thresholds = {
        key: raw[key]
        for key in plan.get("thresholds", {})
    }
    check(
        "thresholds_match_frozen_release_contract",
        plan.get("thresholds") == expected_thresholds,
        plan.get("thresholds"),
    )
    engine_text = module.ENGINE_SOURCE.read_text(encoding="utf-8")
    check(
        "retention_requires_route_and_seed_pass",
        plan.get("retention_requires_route_and_seed_pass") is True
        and '"seed_pass": seed["status"] == "PASS"' in engine_text
        and 'result["status"] == "PASS"' in engine_text,
        {
            "plan_flag": plan.get(
                "retention_requires_route_and_seed_pass"
            )
        },
    )
    try:
        module.execute(overwrite=False, require_pass=False)
    except FileNotFoundError as exc:
        premature_rejected = "Phase-B" in str(exc)
        premature_evidence = str(exc)
    else:
        premature_rejected = False
        premature_evidence = "unexpectedly accepted"
    check(
        "execute_rejects_missing_phase_b",
        premature_rejected,
        premature_evidence,
    )
    check(
        "no_route_decision_was_fabricated",
        not module.OUTPUT.is_file(),
        {"output": str(module.OUTPUT), "present": module.OUTPUT.is_file()},
    )
    source_records = list(plan.get("records", {}).values())
    check(
        "source_contracts_are_hash_bound",
        len(source_records) == 7
        and all(
            len(record.get("sha256", "")) == 64
            for record in source_records
        ),
        {"record_count": len(source_records)},
    )
    check(
        "plan_has_no_diagnosis_or_outcome_fields",
        not {
            "diagnosis",
            "diagnosis_at_dti",
            "group",
            "research_group",
            "outcome",
        }.intersection(plan),
        {"top_level_fields": sorted(plan)},
    )
    self_test = subprocess.run(
        ["/home/ec2-user/fsl/bin/python", str(SOURCE), "--self-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    check(
        "package_self_test",
        self_test.returncode == 0
        and "KEEP_ROUTE_RECOVERY4_SELF_TEST_PASS"
        in self_test.stdout,
        {
            "returncode": self_test.returncode,
            "stdout": self_test.stdout.strip(),
            "stderr": self_test.stderr.strip(),
        },
    )

    failed = sorted(
        name for name, result in checks.items() if result["status"] != "PASS"
    )
    report = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_keep_route_concordance_recovery4_package_validation"
        ),
        "status": "PASS" if not failed else "FAIL",
        "diagnosis_labels_used": False,
        "imaging_executed_by_validation": False,
        "human_visual_qc_inferred": False,
        "total_checks": len(checks),
        "passed_checks": len(checks) - len(failed),
        "failed_checks": failed,
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
