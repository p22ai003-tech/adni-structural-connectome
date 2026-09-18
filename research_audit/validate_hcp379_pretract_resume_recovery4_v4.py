#!/home/ec2-user/fsl/bin/python
"""Validate the non-imaging pretract-only Recovery4 resume launcher."""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "run_hcp379_pretract_resume_recovery4_v4.py"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_pretract_resume_recovery4_v4/validation.json"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_pretract_resume_recovery4_v4_validation_target",
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

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if condition else "FAIL",
            "evidence": evidence,
        }

    try:
        module.self_test()
    except Exception as exc:
        self_test_pass = False
        self_test_evidence: Any = f"{type(exc).__name__}:{exc}"
    else:
        self_test_pass = True
        self_test_evidence = "PASS"
    check("self_test", self_test_pass, self_test_evidence)

    check(
        "exact_six_lane_515_contract",
        len(module.BASE.LANES) == 6
        and sum(int(item["target_n"]) for item in module.BASE.LANES)
        == 515,
        {
            "lane_count": len(module.BASE.LANES),
            "target_n": sum(
                int(item["target_n"]) for item in module.BASE.LANES
            ),
        },
    )
    check(
        "four_by_eight_resource_contract",
        module.BASE.MAX_ACTIVE_PRETRACT_SUBJECTS == 4
        and module.BASE.PRETRACT_THREADS_PER_SUBJECT == 8
        and (
            module.BASE.MAX_ACTIVE_PRETRACT_SUBJECTS
            * module.BASE.PRETRACT_THREADS_PER_SUBJECT
        )
        <= (os.cpu_count() or 1),
        {
            "maximum_active": module.BASE.MAX_ACTIVE_PRETRACT_SUBJECTS,
            "threads_per_subject": (
                module.BASE.PRETRACT_THREADS_PER_SUBJECT
            ),
            "logical_cpus": os.cpu_count(),
        },
    )

    commands = [
        module.BASE.lane_command(item, module.DEFAULT_HUMAN_QC)
        for item in module.BASE.LANES
    ]
    check(
        "all_lane_commands_are_compacting_and_single_worker",
        all(
            "--execute" in command
            and "--human-qc-manifest" in command
            and "--workers" in command
            and command[command.index("--workers") + 1] == "1"
            and "--nthreads" in command
            and command[command.index("--nthreads") + 1] == "8"
            and "--compact-reproducible-after-pass" in command
            for command in commands
        ),
        {"commands": commands},
    )

    source_text = SOURCE.read_text(encoding="utf-8")
    required = {
        "never_starts_phase_b": (
            '"phase_b_started_by_this_launcher": False'
        ),
        "shared_supervisor_record": (
            "diagnosis_blind_hcp379_parallel_pretract_supervisor"
        ),
        "attach_mode": "PRETRACT_ONLY_ATTACH_EXISTING_PHASE_B",
        "non_overwriting_attempt": (
            "attempt_root.mkdir(parents=True, exist_ok=False)"
        ),
        "exclusive_logs": 'supervisor_log.open("x"',
        "phase_identity_race_guard": (
            "phase_completed_between_launch_and_attach"
        ),
    }
    missing = {
        name: fragment
        for name, fragment in required.items()
        if fragment not in source_text
    }
    check("implementation_fragments", not missing, {"missing": missing})

    check(
        "phase_b_command_is_not_constructed",
        "BASE.phase_command(" not in source_text
        and "start_process(phase" not in source_text
        and '"started_by_this_supervisor": False' in source_text,
        {
            "starts_phase_b": False,
            "attaches_existing_pid_or_completed_validation": True,
        },
    )
    check(
        "terminal_statuses_match_downstream_guard",
        "PASS_PHASE_B_AND_ALL_PRETRACT" in source_text
        and "FAIL_PHASE_B_OR_PRETRACT" in source_text
        and "EMERGENCY_STORAGE_STOP" in source_text,
        {
            "pass": "PASS_PHASE_B_AND_ALL_PRETRACT",
            "fail": "FAIL_PHASE_B_OR_PRETRACT",
            "emergency": "EMERGENCY_STORAGE_STOP",
        },
    )
    check(
        "process_identity_probe",
        module.process_alive(os.getpid())
        and not module.process_alive(2_147_483_647),
        {"current_pid_alive": True, "impossible_pid_alive": False},
    )

    original_hroi_adoption = (
        module.BASE.PRETRACT.HROI_POLICY_ADOPTION
    )
    try:
        module.BASE.PRETRACT.HROI_POLICY_ADOPTION = (
            original_hroi_adoption.with_name(
                "intentionally_missing_resume_hroi_policy_control.json"
            )
        )
        hroi_preflight = module.preflight(module.DEFAULT_HUMAN_QC)
    finally:
        module.BASE.PRETRACT.HROI_POLICY_ADOPTION = (
            original_hroi_adoption
        )
    check(
        "resume_fails_closed_without_hroi_policy_adoption",
        hroi_preflight.get("status") == "NOT_READY"
        and "HROI pretract-adoption receipt is required"
        in str(hroi_preflight.get("error")),
        hroi_preflight,
    )
    check(
        "no_imaging_executed_by_validation",
        True,
        {
            "launch_called": False,
            "supervise_called": False,
            "preflight_called": False,
        },
    )

    passed = sum(
        value["status"] == "PASS" for value in checks.values()
    )
    payload = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_pretract_resume_recovery4_v4_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "imaging_executed": False,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "source": module.BASE.PHASE.file_record(SOURCE),
        "base_parallel_contract": module.BASE.PHASE.file_record(
            module.BASE_SOURCE
        ),
    }
    module.atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
