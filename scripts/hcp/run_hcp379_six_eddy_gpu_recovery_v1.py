#!/home/ec2-user/fsl/bin/python
"""Run the exact six-subject Eddy plus Recovery4 replay on one GPU.

The diagnosis-blind, non-overwriting package combines the validated base-four
contract with two independently validated one-subject extensions.  Execution
is sequential because the intended g4dn.4xlarge host exposes one GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import py_compile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
DERIV = Path("/data/derivatives/hcp379_v2")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
SYSTEM_PYTHON = Path("/usr/bin/python3")
HELPER_SOURCE = (
    EXP / "scripts/hcp/run_hcp379_five_eddy_gpu_recovery_v1.py"
)
OVERRIDE_BUILDER_SOURCE = (
    EXP
    / "research_audit/"
    "build_hcp379_pretract_recovery_overrides_v1.py"
)
TOPOLOGY_BUILDER = (
    EXP / "scripts/hcp/build_hcp379_balanced_release_topology_v1.py"
)
TOPOLOGY_VALIDATOR = (
    EXP / "research_audit/validate_hcp379_balanced_release_topology_v1.py"
)
LEDGER_BUILDER = (
    EXP
    / "research_audit/"
    "build_hcp379_pretract_failure_recovery_ledger_v1.py"
)
LEDGER_VALIDATOR = (
    EXP
    / "research_audit/"
    "validate_hcp379_pretract_failure_recovery_ledger_v1.py"
)
LIVE_530_BUILDER = (
    EXP / "research_audit/build_hcp379_live_530_status_density_v1.py"
)
LIVE_530_VALIDATOR = (
    EXP / "research_audit/validate_hcp379_live_530_status_density_v1.py"
)
DASHBOARD_STATUS_WRITER = EXP / "scripts/hcp/hcp379_v2_status.py"
OVERRIDE_OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_pretract_recovery_overrides_v1/overrides.json"
)
TOPOLOGY_OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_balanced_release_topology_v1/topology.json"
)
TOPOLOGY_VALIDATION = TOPOLOGY_OUTPUT.with_name("validation.json")
LEDGER_OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_pretract_failure_recovery_ledger_v1/ledger.json"
)
LEDGER_VALIDATION = LEDGER_OUTPUT.with_name("validation.json")
LIVE_530_SUMMARY = (
    EXP
    / "research_audit/outputs/"
    "hcp379_live_530_status_density_v1/summary.json"
)
LIVE_530_VALIDATION = LIVE_530_SUMMARY.with_name("validation.json")
DASHBOARD_STATUS_OUTPUT = DERIV / "manifests/live_status.json"
BASE_PLAN = (
    EXP
    / "research_audit/outputs/hcp379_eddy_integrity_recovery_plan_v1/"
    "plan.json"
)
BASE_CONTRACT = (
    EXP
    / "research_audit/outputs/hcp379_eddy_acquisition_contract_v1/"
    "contract.json"
)
EXT_0610_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_eddy_extension_005_S_0610_v1"
)
EXT_6084_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_eddy_extension_005_S_6084_v1"
)
BASE_EDDY_RUNNER = (
    EXP / "scripts/hcp/run_hcp379_eddy_integrity_recovery_v1.py"
)
EXT_0610_EDDY_RUNNER = (
    EXP / "scripts/hcp/run_hcp379_eddy_extension_005_S_0610_v1.py"
)
EXT_6084_EDDY_RUNNER = (
    EXP / "scripts/hcp/run_hcp379_eddy_extension_005_S_6084_v1.py"
)
BASE_PRETRACT_RUNNER = (
    EXP / "scripts/hcp/run_hcp379_eddy_pretract_recovery4_v1.py"
)
EXT_0610_PRETRACT_RUNNER = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_eddy_extension_005_pretract_recovery4_v1.py"
)
EXT_6084_PRETRACT_RUNNER = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_eddy_extension_005_S_6084_pretract_recovery4_v1.py"
)
RECOVERY_ROOT = DERIV / "eddy_integrity_recovery_v1"
BASE_PRETRACT_ROOT = DERIV / "eddy_integrity_pretract_recovery4"
EXT_0610_PRETRACT_ROOT = (
    DERIV / "eddy_extension_005_pretract_recovery4"
)
EXT_6084_PRETRACT_ROOT = (
    DERIV / "eddy_extension_005_S_6084_pretract_recovery4"
)
EXPECTED_BASE = {
    "003_S_4373_I378923",
    "003_S_6490_I1043781",
    "003_S_6644_I1083048",
    "006_S_4713_I1483612",
}
EXPECTED_0610 = {"005_S_0610_I906100"}
EXPECTED_6084 = {"005_S_6084_I915209"}
EXPECTED = EXPECTED_BASE | EXPECTED_0610 | EXPECTED_6084
PACKAGE_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_six_eddy_gpu_recovery_v1"
)


def integration_commands() -> list[tuple[str, list[str]]]:
    """Return the exact non-imaging release-status refresh sequence."""
    return [
        (
            "build_recovery_overrides",
            [str(PYTHON), str(OVERRIDE_BUILDER_SOURCE)],
        ),
        (
            "build_release_topology",
            [str(PYTHON), str(TOPOLOGY_BUILDER)],
        ),
        (
            "validate_release_topology",
            [str(PYTHON), str(TOPOLOGY_VALIDATOR)],
        ),
        (
            "build_failure_ledger",
            [str(SYSTEM_PYTHON), str(LEDGER_BUILDER)],
        ),
        (
            "validate_failure_ledger",
            [str(SYSTEM_PYTHON), str(LEDGER_VALIDATOR)],
        ),
        (
            "build_live_530_status",
            [str(PYTHON), str(LIVE_530_BUILDER)],
        ),
        (
            "validate_live_530_status",
            [str(PYTHON), str(LIVE_530_VALIDATOR)],
        ),
        (
            "write_dashboard_live_status",
            [str(PYTHON), str(DASHBOARD_STATUS_WRITER)],
        ),
    ]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPER = load_module(HELPER_SOURCE, "hcp379_six_eddy_helper")
OVERRIDES = load_module(
    OVERRIDE_BUILDER_SOURCE, "hcp379_six_eddy_override_builder"
)


def contract_paths(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / "plan.json",
        root / "contract.json",
        root / "validation.json",
    )


def package_validation() -> dict[str, Any]:
    base_plan = HELPER.load_json(BASE_PLAN)
    base_contract = HELPER.load_json(BASE_CONTRACT)
    base_validation = HELPER.load_json(
        BASE_CONTRACT.with_name("validation.json")
    )
    plan_0610_path, contract_0610_path, validation_0610_path = (
        contract_paths(EXT_0610_ROOT)
    )
    plan_6084_path, contract_6084_path, validation_6084_path = (
        contract_paths(EXT_6084_ROOT)
    )
    plan_0610 = HELPER.load_json(plan_0610_path)
    contract_0610 = HELPER.load_json(contract_0610_path)
    validation_0610 = HELPER.load_json(validation_0610_path)
    plan_6084 = HELPER.load_json(plan_6084_path)
    contract_6084 = HELPER.load_json(contract_6084_path)
    validation_6084 = HELPER.load_json(validation_6084_path)

    plans = (base_plan, plan_0610, plan_6084)
    contracts = (base_contract, contract_0610, contract_6084)
    validations = (
        base_validation,
        validation_0610,
        validation_6084,
    )
    unit_sets = [
        {str(row["unit"]) for row in plan.get("targets", [])}
        for plan in plans
    ]
    scripts = [
        BASE_EDDY_RUNNER,
        EXT_0610_EDDY_RUNNER,
        EXT_6084_EDDY_RUNNER,
        BASE_PRETRACT_RUNNER,
        EXT_0610_PRETRACT_RUNNER,
        EXT_6084_PRETRACT_RUNNER,
        OVERRIDE_BUILDER_SOURCE,
        TOPOLOGY_BUILDER,
        TOPOLOGY_VALIDATOR,
        LEDGER_BUILDER,
        LEDGER_VALIDATOR,
        LIVE_530_BUILDER,
        LIVE_530_VALIDATOR,
        DASHBOARD_STATUS_WRITER,
    ]
    compile_errors: list[str] = []
    for script in scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except Exception as exc:
            compile_errors.append(
                f"{script}:{type(exc).__name__}:{exc}"
            )
    checks = {
        "exact_disjoint_six_subject_contract": (
            unit_sets[0] == EXPECTED_BASE
            and unit_sets[1] == EXPECTED_0610
            and unit_sets[2] == EXPECTED_6084
            and len(set().union(*unit_sets)) == 6
            and set().union(*unit_sets) == EXPECTED
        ),
        "all_recovery_plans_pass": (
            [plan.get("target_n") for plan in plans] == [4, 1, 1]
            and all(plan.get("status") == "PASS" for plan in plans)
        ),
        "all_acquisition_contracts_and_validations_pass": (
            [contract.get("target_n") for contract in contracts]
            == [4, 1, 1]
            and all(
                contract.get("status") == "PASS"
                for contract in contracts
            )
            and all(
                validation.get("status") == "PASS"
                and validation.get("passed_checks")
                == validation.get("total_checks")
                for validation in validations
            )
        ),
        "all_execution_scripts_compile": not compile_errors,
        "all_contracts_are_blind_and_non_overwriting": all(
            plan.get("diagnosis_labels_used") is False
            and plan.get("outcomes_used") is False
            and plan.get("non_overwriting") is True
            and plan.get("historical_eddy_overwrite_allowed") is False
            for plan in plans
        ),
        "all_six_recovery_roots_are_predeclared_for_release_integration": (
            set(OVERRIDES.EDDY_PRETRACT_ROOTS) == EXPECTED
            and set(OVERRIDES.EDDY_ORIGINAL_ROOTS) == EXPECTED
            and OVERRIDES.EDDY_PRETRACT_ROOTS[
                next(iter(EXPECTED_0610))
            ].resolve()
            == EXT_0610_PRETRACT_ROOT.resolve()
            and OVERRIDES.EDDY_PRETRACT_ROOTS[
                next(iter(EXPECTED_6084))
            ].resolve()
            == EXT_6084_PRETRACT_ROOT.resolve()
            and all(
                OVERRIDES.EDDY_PRETRACT_ROOTS[unit].resolve()
                == BASE_PRETRACT_ROOT.resolve()
                for unit in EXPECTED_BASE
            )
        ),
        "successful_execution_has_self_contained_release_integration": (
            OVERRIDE_BUILDER_SOURCE in scripts
            and TOPOLOGY_BUILDER in scripts
            and TOPOLOGY_VALIDATOR in scripts
            and LEDGER_BUILDER in scripts
            and LEDGER_VALIDATOR in scripts
            and LIVE_530_BUILDER in scripts
            and LIVE_530_VALIDATOR in scripts
            and DASHBOARD_STATUS_WRITER in scripts
            and [name for name, _ in integration_commands()]
            == [
                "build_recovery_overrides",
                "build_release_topology",
                "validate_release_topology",
                "build_failure_ledger",
                "validate_failure_ledger",
                "build_live_530_status",
                "validate_live_530_status",
                "write_dashboard_live_status",
            ]
        ),
        "execution_is_sequential_for_one_gpu": True,
    }
    passed = sum(checks.values())
    gpu = HELPER.gpu_record()
    records = {
        "base_plan": HELPER.file_record(BASE_PLAN),
        "base_contract": HELPER.file_record(BASE_CONTRACT),
        "extension_0610_plan": HELPER.file_record(plan_0610_path),
        "extension_0610_contract": HELPER.file_record(
            contract_0610_path
        ),
        "extension_6084_plan": HELPER.file_record(plan_6084_path),
        "extension_6084_contract": HELPER.file_record(
            contract_6084_path
        ),
        "base_contract_validation": HELPER.file_record(
            BASE_CONTRACT.with_name("validation.json")
        ),
        "extension_0610_validation": HELPER.file_record(
            validation_0610_path
        ),
        "extension_6084_validation": HELPER.file_record(
            validation_6084_path
        ),
        "shared_execution_helper": HELPER.file_record(HELPER_SOURCE),
        "release_override_builder": HELPER.file_record(
            OVERRIDE_BUILDER_SOURCE
        ),
        "six_subject_orchestrator": HELPER.file_record(Path(__file__)),
        **{
            script.name: HELPER.file_record(script)
            for script in scripts
        },
    }
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_six_eddy_gpu_recovery_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "execution_readiness": (
            "READY_FOR_GPU_EXECUTION"
            if passed == len(checks) and gpu["ready"]
            else "WAITING_FOR_GPU"
            if passed == len(checks)
            else "INVALID_PACKAGE"
        ),
        "generated_utc": HELPER.utc_now(),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": {
            name: "PASS" if value else "FAIL"
            for name, value in checks.items()
        },
        "compile_errors": compile_errors,
        "gpu": gpu,
        "target_n": len(EXPECTED),
        "units": sorted(EXPECTED),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "imaging_executed": False,
        "records": records,
    }


def pretract_root(unit: str) -> Path:
    if unit in EXPECTED_BASE:
        return BASE_PRETRACT_ROOT
    if unit in EXPECTED_0610:
        return EXT_0610_PRETRACT_ROOT
    if unit in EXPECTED_6084:
        return EXT_6084_PRETRACT_ROOT
    raise KeyError(unit)


def postconditions() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for unit in sorted(EXPECTED):
        root = pretract_root(unit)
        eddy_pointer = RECOVERY_ROOT / "subjects" / unit / "result.json"
        eddy = (
            HELPER.load_json(eddy_pointer)
            if eddy_pointer.is_file()
            else {}
        )
        state_path = root / "qc" / "subjects" / f"{unit}.json"
        compact_path = (
            root / "qc" / "pretract_compaction" / f"{unit}.json"
        )
        pretract = (
            HELPER.load_json(state_path) if state_path.is_file() else {}
        )
        compact = (
            HELPER.load_json(compact_path)
            if compact_path.is_file()
            else {}
        )
        passed = (
            eddy.get("status") == "PASS_EDDY_INTEGRITY_RECOVERY"
            and pretract.get("status") == "PASS_PRETRACT_RECOVERY4"
            and compact.get("status") == "PASS_COMPACTED"
        )
        rows.append(
            {
                "unit": unit,
                "status": "PASS" if passed else "FAIL",
                "eddy_status": eddy.get("status", "MISSING"),
                "pretract_status": pretract.get("status", "MISSING"),
                "compaction_status": compact.get(
                    "status", "MISSING"
                ),
            }
        )
    overrides = HELPER.load_json(OVERRIDE_OUTPUT)
    topology = HELPER.load_json(TOPOLOGY_OUTPUT)
    topology_validation = HELPER.load_json(TOPOLOGY_VALIDATION)
    ledger = HELPER.load_json(LEDGER_OUTPUT)
    ledger_validation = HELPER.load_json(LEDGER_VALIDATION)
    live_summary = HELPER.load_json(LIVE_530_SUMMARY)
    live_validation = HELPER.load_json(LIVE_530_VALIDATION)
    dashboard_status = HELPER.load_json(DASHBOARD_STATUS_OUTPUT)
    completed_eddy_units = {
        str(row.get("unit"))
        for row in overrides.get("completed_eddy_overrides", [])
        if isinstance(row, dict)
    }
    production_rows = {
        str(row.get("unit")): row
        for row in topology.get("production", [])
        if isinstance(row, dict)
    }
    integrated_ready_units = {
        unit
        for unit in EXPECTED
        if unit in production_rows
        and production_rows[unit].get("pretract_recovery_override")
        is True
        and production_rows[unit].get("pretract", {}).get("ready")
        is True
    }
    remaining_failed_units = {
        str(row.get("unit"))
        for row in ledger.get("failed_units", [])
        if isinstance(row, dict)
    } & EXPECTED
    integration_pass = (
        overrides.get("status") == "PASS"
        and overrides.get("completed_eddy_override_n") == len(EXPECTED)
        and completed_eddy_units == EXPECTED
        and integrated_ready_units == EXPECTED
        and not remaining_failed_units
        and topology_validation.get("status") == "PASS"
        and topology_validation.get("passed_checks")
        == topology_validation.get("total_checks")
        and ledger_validation.get("status") == "PASS"
        and ledger_validation.get("passed_checks")
        == ledger_validation.get("total_checks")
        and live_summary.get("total_n") == 530
        and live_validation.get("status") == "PASS"
        and live_validation.get("passed_checks")
        == live_validation.get("total_checks")
        and dashboard_status.get("baseline_freeze", {}).get("target_n")
        == 530
    )
    return {
        "status": (
            "PASS"
            if all(row["status"] == "PASS" for row in rows)
            and integration_pass
            else "FAIL"
        ),
        "passed_n": sum(row["status"] == "PASS" for row in rows),
        "target_n": len(rows),
        "rows": rows,
        "release_integration": {
            "status": "PASS" if integration_pass else "FAIL",
            "completed_eddy_override_n": overrides.get(
                "completed_eddy_override_n"
            ),
            "completed_eddy_units": sorted(completed_eddy_units),
            "integrated_ready_units": sorted(integrated_ready_units),
            "remaining_failed_units": sorted(remaining_failed_units),
            "topology_validation_status": topology_validation.get(
                "status"
            ),
            "ledger_validation_status": ledger_validation.get(
                "status"
            ),
            "live_530_validation_status": live_validation.get(
                "status"
            ),
            "live_530_total_n": live_summary.get("total_n"),
            "dashboard_target_n": dashboard_status.get(
                "baseline_freeze", {}
            ).get("target_n"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--integrate-only", action="store_true")
    parser.add_argument("--eddy-nthreads", type=int, default=8)
    parser.add_argument("--pretract-nthreads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.eddy_nthreads <= 16:
        parser.error("--eddy-nthreads must be in 1..16")
    if not 1 <= args.pretract_nthreads <= 16:
        parser.error("--pretract-nthreads must be in 1..16")

    validation = package_validation()
    HELPER.atomic_json(PACKAGE_ROOT / "validation.json", validation)
    if args.validate_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0 if validation["status"] == "PASS" else 1
    if (
        args.execute
        and validation["execution_readiness"] != "READY_FOR_GPU_EXECUTION"
    ):
        raise SystemExit("exact package passes but a working GPU is required")

    attempt = (
        RECOVERY_ROOT
        / "six_subject_gpu_attempts"
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + (
                "-six-eddy-integration-only"
                if args.integrate_only
                else "-six-eddy-gpu"
            )
        )
    )
    attempt.mkdir(parents=True, exist_ok=False)
    recovery_commands = [
        (
            "base_four_eddy",
            [
                str(PYTHON),
                str(BASE_EDDY_RUNNER),
                "--execute",
                "--backend",
                "gpu",
                "--nthreads",
                str(args.eddy_nthreads),
            ],
        ),
        (
            "extension_0610_eddy",
            [
                str(PYTHON),
                str(EXT_0610_EDDY_RUNNER),
                "--execute",
                "--backend",
                "gpu",
                "--nthreads",
                str(args.eddy_nthreads),
            ],
        ),
        (
            "extension_6084_eddy",
            [
                str(PYTHON),
                str(EXT_6084_EDDY_RUNNER),
                "--execute",
                "--backend",
                "gpu",
                "--nthreads",
                str(args.eddy_nthreads),
            ],
        ),
        (
            "base_four_recovery4",
            [
                str(PYTHON),
                str(BASE_PRETRACT_RUNNER),
                "--execute",
                "--nthreads",
                str(args.pretract_nthreads),
            ],
        ),
        (
            "extension_0610_recovery4",
            [
                str(PYTHON),
                str(EXT_0610_PRETRACT_RUNNER),
                "--execute",
                "--nthreads",
                str(args.pretract_nthreads),
            ],
        ),
        (
            "extension_6084_recovery4",
            [
                str(PYTHON),
                str(EXT_6084_PRETRACT_RUNNER),
                "--execute",
                "--nthreads",
                str(args.pretract_nthreads),
            ],
        ),
    ]
    commands = (
        integration_commands()
        if args.integrate_only
        else recovery_commands + integration_commands()
    )
    steps: list[dict[str, Any]] = []
    for name, command in commands:
        step = HELPER.run_step(name, command, attempt)
        steps.append(step)
        if step["status"] != "PASS":
            break
    post = postconditions()
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_six_eddy_gpu_recovery_invocation",
        "status": (
            "PASS"
            if len(steps) == len(commands)
            and all(row["status"] == "PASS" for row in steps)
            and post["status"] == "PASS"
            else "FAIL"
        ),
        "generated_utc": HELPER.utc_now(),
        "attempt_root": str(attempt.resolve()),
        "validation": validation,
        "steps": steps,
        "postconditions": post,
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "historical_outputs_overwritten": False,
    }
    HELPER.atomic_json(attempt / "invocation.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
