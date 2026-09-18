#!/home/ec2-user/fsl/bin/python
"""Hostile validation for the Phase-B plus parallel-pretract supervisor."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP
    / "scripts/hcp/run_hcp379_parallel_pretract_recovery4_v3.py"
)
PLAN = (
    Path("/data/derivatives/hcp379_v2")
    / "parallel_recovery4/parallel_phase_b_pretract_plan.json"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_parallel_pretract_recovery4_v3/validation.json"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module(SOURCE, "parallel_pretract_validation_target")
    plan = module.load_json(PLAN)
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if condition else "FAIL",
            "evidence": evidence,
        }

    lanes = plan.get("pretract_lanes", [])
    check(
        "exact_six_lane_515_subject_scope",
        len(lanes) == 6
        and sum(int(row.get("target_n", -1)) for row in lanes) == 515
        and [row.get("name") for row in lanes]
        == [
            "corrected_core_216_new",
            "corrected_legacy_tensor_212_new",
            "corrected_freesurfer_1_new",
            "archive_calibration_6",
            "corrected_archive_low_28_new",
            "corrected_archive_D0_52_new",
        ],
        {
            "lane_count": len(lanes),
            "target_n": sum(
                int(row.get("target_n", -1)) for row in lanes
            ),
            "lanes": [row.get("name") for row in lanes],
        },
    )
    resource = plan.get("resource_contract", {})
    check(
        "parallel_resource_contract_matches_storage_forecast",
        resource.get("host_logical_cpus") == 64
        and resource.get("phase_b_cores") == 24
        and resource.get("maximum_active_pretract_subjects") == 4
        and resource.get("pretract_threads_per_subject") == 8
        and resource.get("maximum_declared_parallel_threads") == 56
        and resource.get("streaming_compaction_required") is True
        and resource.get("maximum_declared_parallel_threads")
        <= resource.get("host_logical_cpus"),
        resource,
    )
    launch_order = plan.get("launch_order", {})
    check(
        "phase_b_starts_before_bounded_pretract_queue",
        launch_order.get("phase_b_first") is True
        and launch_order.get(
            "wait_for_phase_b_live_dag_before_pretract"
        )
        is True
        and launch_order.get("maximum_simultaneous_lane_processes") == 4
        and launch_order.get(
            "continue_pretract_if_phase_b_later_fails"
        )
        is True,
        launch_order,
    )
    check(
        "every_lane_uses_one_worker_and_streaming_compaction",
        all(
            "--execute" in row.get("command", [])
            and "--workers" in row.get("command", [])
            and row["command"][row["command"].index("--workers") + 1]
            == "1"
            and "--nthreads" in row.get("command", [])
            and row["command"][row["command"].index("--nthreads") + 1]
            == "8"
            and "--compact-reproducible-after-pass"
            in row.get("command", [])
            for row in lanes
        ),
        {row["name"]: row["command"] for row in lanes},
    )
    hroi_policy = plan.get("hroi_source_policy", {})
    hroi_policy_state_valid = (
        (
            plan.get("status") == "AWAITING_530_HROI_SOURCE_POLICY"
            and hroi_policy.get("pretract_use_authorized") is False
            and hroi_policy.get("receipt") is None
            and hroi_policy.get("status")
            == "WAITING_FOR_530_HROI_POLICY_ADOPTION"
        )
        or (
            plan.get("status")
            == "READY_TO_LAUNCH_AFTER_GENUINE_HUMAN_QC_AND_HROI_POLICY"
            and hroi_policy.get("pretract_use_authorized") is True
            and isinstance(hroi_policy.get("receipt"), dict)
            and hroi_policy.get("status")
            == "PASS_OPERATIONAL_PRETRACT_ADOPTION"
        )
    )
    check(
        "plan_is_non_imaging_and_hroi_policy_state_is_explicit",
        hroi_policy_state_valid
        and plan.get("diagnosis_labels_used") is False
        and plan.get("diagnosis_or_outcomes_used") is False
        and plan.get("non_overwriting") is True
        and plan.get("imaging_executed_by_plan") is False
        and plan.get("selected_count_tractography_authorized") is False,
        {
            "status": plan.get("status"),
            "human_visual_qc": plan.get("human_visual_qc"),
            "hroi_source_policy": plan.get("hroi_source_policy"),
            "selected_count_tractography_authorized": plan.get(
                "selected_count_tractography_authorized"
            ),
        },
    )
    records = plan.get("records", {})
    current_records = {
        "launcher": module.PHASE.file_record(SOURCE),
        "topology_validation": module.PHASE.file_record(
            module.TOPOLOGY_VALIDATION
        ),
        "compactor_validation": module.PHASE.file_record(
            module.COMPACTOR_VALIDATION
        ),
        "storage_forecast": module.PHASE.file_record(
            module.STORAGE_FORECAST
        ),
    }
    check(
        "plan_is_hash_bound_to_current_launch_and_safety_contracts",
        all(records.get(name) == record for name, record in current_records.items())
        and all(
            isinstance(record, dict)
            and len(str(record.get("sha256", ""))) == 64
            and module.PHASE.file_record(
                Path(str(record.get("path", "")))
            )
            == record
            for record in records.values()
        ),
        {
            "current_records": current_records,
            "record_count": len(records),
        },
    )
    attempts_before = (
        {path.name for path in module.ATTEMPTS.iterdir()}
        if module.ATTEMPTS.is_dir()
        else set()
    )
    launch_probe = subprocess.run(
        [
            str(module.PYTHON),
            str(SOURCE),
            "--launch",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    attempts_after = (
        {path.name for path in module.ATTEMPTS.iterdir()}
        if module.ATTEMPTS.is_dir()
        else set()
    )
    launch_rejection_is_expected = (
        "HROI pretract-adoption receipt is required" in launch_probe.stderr
        or bool(module.PHASE.active_imaging_processes())
        or bool(module.other_supervisors())
    )
    check(
        "launch_is_fail_closed_before_attempt_creation",
        launch_probe.returncode != 0
        and launch_rejection_is_expected
        and attempts_after == attempts_before,
        {
            "returncode": launch_probe.returncode,
            "stderr": launch_probe.stderr[-2000:],
            "new_attempts": sorted(attempts_after - attempts_before),
        },
    )
    active_imaging = module.PHASE.active_imaging_processes()
    active_pretract = [
        command
        for command in active_imaging
        if any(
            str(row["source"]) in command and "--execute" in command
            for row in module.LANES
        )
    ]
    check(
        "no_parallel_supervisor_or_pretract_lane_is_active",
        not module.other_supervisors() and not active_pretract,
        {
            "supervisors": module.other_supervisors(),
            "active_pretract": active_pretract,
            "other_active_imaging_allowed": active_imaging,
        },
    )
    compile_probe = subprocess.run(
        [
            str(module.PYTHON),
            "-m",
            "py_compile",
            str(SOURCE),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    self_test_probe = subprocess.run(
        [
            str(module.PYTHON),
            str(SOURCE),
            "--self-test",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    check(
        "source_compiles_and_static_self_test_passes",
        compile_probe.returncode == 0
        and self_test_probe.returncode == 0
        and "HCP379_PARALLEL_PRETRACT_RECOVERY4_SELF_TEST_PASS"
        in self_test_probe.stdout,
        {
            "compile_returncode": compile_probe.returncode,
            "compile_stderr": compile_probe.stderr,
            "self_test_returncode": self_test_probe.returncode,
            "self_test_stdout": self_test_probe.stdout,
            "self_test_stderr": self_test_probe.stderr,
        },
    )
    check(
        "no_selected_count_bulk_tractography_command_is_present",
        all(
            "tractography" not in Path(str(row["command"][1])).name
            for row in lanes
        )
        and plan.get("selected_count_tractography_authorized") is False,
        [row["command"][1] for row in lanes],
    )

    passed = sum(
        value["status"] == "PASS" for value in checks.values()
    )
    payload = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_parallel_pretract_recovery4_v3_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "imaging_executed": False,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "records": {
            "source": module.PHASE.file_record(SOURCE),
            "plan": module.PHASE.file_record(PLAN),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    module.atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
