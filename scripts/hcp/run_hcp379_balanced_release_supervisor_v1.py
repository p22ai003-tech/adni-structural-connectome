#!/home/ec2-user/fsl/bin/python
"""Supervise the balanced HCP379 pilot, scale-up, assembly and verification.

The supervisor is deliberately gate-driven.  It waits for the runtime recipe
selection and signed final-HROI review, runs a six-route pilot with two
16-thread workers while Recovery4 pretract processing may still be active,
stops on any non-pass, then waits for all 515 pretract subjects before the
four-worker full run.  It finally assembles and independently verifies the
530-by-nine release.  No dashboard is activated.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
RUNNER = EXP / "scripts/hcp/run_hcp379_balanced_production_v1.py"
RUNNER_VALIDATOR = (
    EXP
    / "research_audit/"
    "validate_hcp379_balanced_production_v1.py"
)
ASSEMBLER = (
    EXP
    / "scripts/hcp/"
    "assemble_hcp379_balanced_integration_release_v1.py"
)
VERIFIER = (
    EXP
    / "research_audit/"
    "verify_hcp379_balanced_integration_release_v1.py"
)
PRODUCTION_STATE = (
    HCP_ROOT / "balanced_release_v1/production/cohort_state.json"
)
ATTEMPTS = (
    HCP_ROOT / "balanced_release_v1/supervisor_v1/attempts"
)
PILOT_N = 6
PILOT_WORKERS = 2
THREADS = 16
MAX_FULL_WORKERS = 12
FULL_WORKERS = min(
    MAX_FULL_WORKERS,
    max(1, (os.cpu_count() or 1) // THREADS),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run_json(
    command: Sequence[str],
    *,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> tuple[int, dict[str, Any], str]:
    completed = subprocess.run(
        list(command), capture_output=True, text=True, check=False
    )
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text(completed.stderr, encoding="utf-8")
    value: dict[str, Any] = {}
    candidates = [completed.stdout]
    candidates.extend(
        completed.stdout[index + 1 :]
        for index in reversed(
            [
                position
                for position in range(len(completed.stdout))
                if completed.stdout.startswith("\n{", position)
            ]
        )
    )
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            value = parsed
            break
    return completed.returncode, value, completed.stderr[-2000:]


def runner_preflight(*, pilot: bool) -> dict[str, Any]:
    workers = PILOT_WORKERS if pilot else FULL_WORKERS
    command = [
        str(PYTHON),
        str(RUNNER),
        "--workers",
        str(workers),
        "--threads-per-worker",
        str(THREADS),
    ]
    if pilot:
        command.extend(
            ["--limit", str(PILOT_N), "--stratified-pilot"]
        )
    returncode, value, stderr = run_json(command)
    if returncode != 0 or not value:
        raise RuntimeError(
            f"production preflight failed: rc={returncode} {stderr}"
        )
    return value


def compact_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    waiting = list(
        value.get("required_execution_waiting_units") or []
    )
    return {
        "status": value.get("status"),
        "gates": value.get("gates"),
        "selected_balanced_total_streamlines": value.get(
            "selected_balanced_total_streamlines"
        ),
        "selected_per_seed_streamlines": value.get(
            "selected_per_seed_streamlines"
        ),
        "pretract_ready_n": value.get("pretract_ready_n"),
        "pretract_waiting_n": value.get("pretract_waiting_n"),
        "required_execution_unit_n": value.get(
            "required_execution_unit_n"
        ),
        "required_execution_pretract_ready_n": value.get(
            "required_execution_pretract_ready_n"
        ),
        "required_execution_waiting_n": len(waiting),
        "required_execution_waiting_sample": waiting[:10],
        "free_bytes": value.get("free_bytes"),
        "planned_workers": value.get("planned_workers"),
        "threads_per_subject": value.get("threads_per_subject"),
        "minimum_start_free_bytes": value.get(
            "minimum_start_free_bytes"
        ),
    }


def process_lines() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", Path(__file__).name],
        capture_output=True,
        text=True,
        check=False,
    )
    return [
        line
        for line in result.stdout.splitlines()
        if "--supervise" in line
        and "tmux new-session" not in line
        and not line.startswith(f"{os.getpid()} ")
    ]


def update_state(
    path: Path, state: dict[str, Any], **updates: Any
) -> None:
    state.update(updates)
    state["updated_utc"] = utc_now()
    atomic_json(path, state)


def wait_until_ready(
    *,
    pilot: bool,
    poll_seconds: int,
    state_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    stage = "PILOT" if pilot else "FULL_515"
    while True:
        value = runner_preflight(pilot=pilot)
        update_state(
            state_path,
            state,
            status=f"WAITING_FOR_{stage}_GATES",
            active_stage=stage,
            latest_preflight=compact_preflight(value),
        )
        if value.get("status") == (
            "READY_FOR_BALANCED_PRODUCTION_EXECUTION"
        ):
            return value
        time.sleep(poll_seconds)


def execute_runner(
    *,
    pilot: bool,
    attempt: Path,
) -> dict[str, Any]:
    stage = "pilot" if pilot else "full_515"
    workers = PILOT_WORKERS if pilot else FULL_WORKERS
    command = [
        str(PYTHON),
        str(RUNNER),
        "--execute",
        "--workers",
        str(workers),
        "--threads-per-worker",
        str(THREADS),
    ]
    if pilot:
        command.extend(
            ["--limit", str(PILOT_N), "--stratified-pilot"]
        )
    returncode, value, stderr = run_json(
        command,
        stdout_path=attempt / f"{stage}.stdout.json",
        stderr_path=attempt / f"{stage}.stderr.log",
    )
    if returncode != 0 or not str(value.get("status", "")).startswith(
        "PASS_"
    ):
        raise RuntimeError(
            f"{stage} failed: rc={returncode} value={value} {stderr}"
        )
    expected = PILOT_N if pilot else 515
    state = load_json(PRODUCTION_STATE)
    if (
        state.get("status") != "PASS_BALANCED_PRODUCTION_COHORT"
        or state.get("target_n") != expected
        or state.get("processed_n") != expected
        or state.get("pass_n") != expected
        or state.get("unstarted_n") != 0
        or state.get("stopped_early") is not False
    ):
        raise ValueError(f"{stage}: cohort state differs")
    validation_rc, validation, validation_stderr = run_json(
        [str(PYTHON), str(RUNNER_VALIDATOR)],
        stdout_path=attempt / f"{stage}.validation.stdout.json",
        stderr_path=attempt / f"{stage}.validation.stderr.log",
    )
    if (
        validation_rc != 0
        or validation.get("status") != "PASS"
        or validation.get("runtime", {}).get("status") != "PASS"
        or validation.get("runtime", {}).get("processed_n") != expected
    ):
        raise RuntimeError(
            f"{stage}: independent runner validation failed "
            f"{validation_stderr}"
        )
    return {
        "runner": value,
        "cohort_state": state,
        "validation": validation,
    }


def supervise(attempt: Path, poll_seconds: int) -> int:
    attempt.mkdir(parents=True, exist_ok=False)
    state_path = attempt / "supervisor_state.json"
    state: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_balanced_release_supervisor"
        ),
        "status": "STARTING",
        "started_utc": utc_now(),
        "updated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "pilot_subject_n": PILOT_N,
        "pilot_workers": PILOT_WORKERS,
        "full_workers": FULL_WORKERS,
        "threads_per_worker": THREADS,
        "dashboard_activation_authorized": False,
        "attempt_root": str(attempt.resolve()),
    }
    atomic_json(state_path, state)
    try:
        wait_until_ready(
            pilot=True,
            poll_seconds=poll_seconds,
            state_path=state_path,
            state=state,
        )
        update_state(
            state_path,
            state,
            status="RUNNING_SIX_ROUTE_PILOT",
            active_stage="PILOT",
        )
        pilot_result = execute_runner(pilot=True, attempt=attempt)
        update_state(
            state_path,
            state,
            status="PASS_SIX_ROUTE_PILOT_WAITING_FULL_515",
            pilot={
                "status": pilot_result["runner"]["status"],
                "processed_n": pilot_result["runner"]["processed_n"],
                "minimum_density": min(
                    float(row["combined_edge_density"])
                    for row in pilot_result["cohort_state"]["subjects"]
                ),
            },
        )
        wait_until_ready(
            pilot=False,
            poll_seconds=poll_seconds,
            state_path=state_path,
            state=state,
        )
        update_state(
            state_path,
            state,
            status="RUNNING_FULL_515_BALANCED_PRODUCTION",
            active_stage="FULL_515",
        )
        full_result = execute_runner(pilot=False, attempt=attempt)
        update_state(
            state_path,
            state,
            status="PASS_FULL_515_ASSEMBLING_RELEASE",
            full_515={
                "status": full_result["runner"]["status"],
                "processed_n": full_result["runner"]["processed_n"],
                "minimum_density": min(
                    float(row["combined_edge_density"])
                    for row in full_result["cohort_state"]["subjects"]
                ),
            },
            active_stage="ASSEMBLY",
        )
        assembly_rc, assembly, assembly_stderr = run_json(
            [str(PYTHON), str(ASSEMBLER), "--assemble"],
            stdout_path=attempt / "assembly.stdout.json",
            stderr_path=attempt / "assembly.stderr.log",
        )
        if (
            assembly_rc != 0
            or assembly.get("status")
            != "PASS_ASSEMBLED_AWAITING_INDEPENDENT_VERIFICATION"
            or assembly.get("subject_count") != 530
            or assembly.get("matrix_file_count") != 4770
        ):
            raise RuntimeError(
                f"release assembly failed: {assembly_stderr}"
            )
        update_state(
            state_path,
            state,
            status="PASS_ASSEMBLED_RUNNING_INDEPENDENT_VERIFICATION",
            assembly=assembly,
            active_stage="INDEPENDENT_VERIFICATION",
        )
        verify_rc, verification, verify_stderr = run_json(
            [str(PYTHON), str(VERIFIER), "--verify"],
            stdout_path=attempt / "verification.stdout.json",
            stderr_path=attempt / "verification.stderr.log",
        )
        if (
            verify_rc != 0
            or verification.get("status")
            != "PASS_VERIFIED_INTEGRATION_RELEASE"
            or verification.get("subject_count") != 530
            or verification.get("matrix_file_count") != 4770
            or verification.get("final_release_authorized") is not True
        ):
            raise RuntimeError(
                f"independent release verification failed: {verify_stderr}"
            )
        update_state(
            state_path,
            state,
            status="PASS_VERIFIED_INTEGRATION_RELEASE",
            active_stage="COMPLETE",
            verification=verification,
            dashboard_activation_authorized=False,
            completed_utc=utc_now(),
        )
        return 0
    except Exception as exc:
        update_state(
            state_path,
            state,
            status="FAIL_STOPPED",
            active_stage="STOPPED",
            error=f"{type(exc).__name__}:{exc}",
            completed_utc=utc_now(),
            dashboard_activation_authorized=False,
        )
        return 1


def self_test() -> dict[str, Any]:
    checks = {
        "pilot_covers_six_routes": PILOT_N == 6,
        "pilot_capacity_leaves_32_threads_for_pretract": (
            PILOT_WORKERS * THREADS == 32
        ),
        "full_workers_are_host_bounded": (
            1 <= FULL_WORKERS <= MAX_FULL_WORKERS
            and FULL_WORKERS * THREADS <= (os.cpu_count() or 1)
        ),
        "current_host_uses_all_complete_16_thread_slots": (
            FULL_WORKERS
            == min(
                MAX_FULL_WORKERS,
                max(1, (os.cpu_count() or 1) // THREADS),
            )
        ),
        "resize_ceiling_supports_192_logical_cpus": (
            MAX_FULL_WORKERS * THREADS == 192
        ),
        "exact_final_target": 15 + 515 == 530,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 10 <= args.poll_seconds <= 600:
        parser.error("--poll-seconds must be in 10..600")
    if args.self_test:
        value = self_test()
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0 if value["status"] == "PASS" else 1
    if args.supervise:
        if process_lines():
            raise RuntimeError("another balanced release supervisor is active")
        attempt = (
            args.attempt_root.resolve()
            if args.attempt_root is not None
            else ATTEMPTS / f"{stamp()}-balanced-release-supervisor"
        )
        return supervise(attempt, args.poll_seconds)
    pilot = runner_preflight(pilot=True)
    full = runner_preflight(pilot=False)
    value = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_balanced_release_supervisor_preflight"
        ),
        "generated_utc": utc_now(),
        "status": (
            "ACTIVE" if process_lines() else "READY_TO_SUPERVISE"
        ),
        "active_supervisors": process_lines(),
        "pilot": compact_preflight(pilot),
        "full_515": compact_preflight(full),
        "imaging_started": False,
        "dashboard_activation_authorized": False,
    }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
