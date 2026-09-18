#!/home/ec2-user/fsl/bin/python
"""Resume the six Recovery4 pretract lanes beside an existing Phase-B run.

This recovery launcher never starts Phase B.  It attaches read-only status to
the single live Phase-B launcher, or to an already completed valid Phase-B
result, then starts the same six validated and diagnosis-blind pretract lanes
used by the original parallel launcher.  A new non-overwriting attempt binds
the interrupted supervisor state and preserves the four-subject concurrency
and streaming-compaction contracts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
BASE_SOURCE = (
    EXP / "scripts/hcp/run_hcp379_parallel_pretract_recovery4_v3.py"
)
PHASE_SOURCE = EXP / "scripts/hcp/run_hcp379_phase_b_stability_v2.py"
DEFAULT_HUMAN_QC = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
ATTEMPTS = HCP_ROOT / "parallel_recovery4/attempts"
STATUS_SOURCE = EXP / "scripts/hcp/hcp379_v2_status.py"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_SOURCE, "hcp379_parallel_pretract_recovery4_base")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def process_lines(pattern: str) -> list[tuple[int, str]]:
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        head, _, command = line.partition(" ")
        if not head.isdigit() or int(head) == os.getpid():
            continue
        if "pgrep -af" in command:
            continue
        rows.append((int(head), command))
    return rows


def phase_runner_pids() -> list[int]:
    return [
        pid
        for pid, command in process_lines(PHASE_SOURCE.name)
        if str(PHASE_SOURCE) in command
    ]


def active_pretract_processes() -> list[str]:
    names = {Path(item["source"]).name for item in BASE.LANES}
    active: list[str] = []
    for name in sorted(names):
        for pid, command in process_lines(name):
            if "--execute" in command:
                active.append(f"{pid} {command}")
    return active


def active_resume_supervisors() -> list[str]:
    return [
        f"{pid} {command}"
        for pid, command in process_lines(Path(__file__).name)
        if "--supervise-pretract-only" in command
    ]


def process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def phase_validation_state() -> dict[str, Any]:
    path = BASE.PHASE.STABILITY_OUTPUT
    if not path.is_file():
        return {
            "status": "MISSING",
            "path": str(path),
            "returncode": 1,
        }
    try:
        value = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "UNREADABLE",
            "path": str(path),
            "returncode": 1,
            "error": f"{type(exc).__name__}:{exc}",
        }
    selected = value.get(
        "smallest_density_qualified_scale_up_streamline_count"
    )
    passed = bool(
        value.get("record_type")
        == "diagnosis_blind_hcp379_phase_b_recipe_stability_validation"
        and value.get("status") == "PASS"
        and value.get("diagnosis_labels_used") is False
        and value.get("unit_count") == 15
        and selected in {3_000_000, 5_000_000, 10_000_000}
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "path": str(path),
        "returncode": 0 if passed else 1,
        "selected_streamline_count": selected,
        "record": BASE.PHASE.file_record(path),
    }


def latest_supervisor_state() -> dict[str, Any] | None:
    candidates = sorted(
        ATTEMPTS.glob("*/supervisor_state.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            value = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if value.get("record_type") == (
            "diagnosis_blind_hcp379_parallel_pretract_supervisor"
        ):
            return {
                "record": BASE.PHASE.file_record(path),
                "status": value.get("status"),
                "phase_b": value.get("phase_b"),
                "pretract": value.get("pretract"),
            }
    return None


def update_state(
    path: Path, state: dict[str, Any], **updates: Any
) -> None:
    state.update(updates)
    state["updated_utc"] = utc_now()
    atomic_json(path, state)


def terminate_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def validate_resume(human_qc: Path) -> dict[str, Any]:
    static = BASE.validate_static_contract()
    hroi_source_policy = static.get("hroi_source_policy", {})
    if (
        hroi_source_policy.get("status")
        != "PASS_OPERATIONAL_PRETRACT_ADOPTION"
        or hroi_source_policy.get("pretract_use_authorized") is not True
        or hroi_source_policy.get("receipt") is None
    ):
        raise FileNotFoundError(
            "530-subject HROI pretract-adoption receipt is required"
        )
    human = BASE.PHASE.validate_human_qc(
        human_qc, static["canary_units"]
    )
    active_lanes = active_pretract_processes()
    if active_lanes:
        raise RuntimeError(
            "pretract imaging is already active: " + " | ".join(active_lanes)
        )
    active_supervisors = active_resume_supervisors()
    if active_supervisors:
        raise RuntimeError(
            "pretract-only recovery supervisor is active: "
            + " | ".join(active_supervisors)
        )
    phase_pids = phase_runner_pids()
    if len(phase_pids) > 1:
        raise RuntimeError(
            f"multiple Phase-B launchers are active: {phase_pids}"
        )
    phase_validation = phase_validation_state()
    if not phase_pids and phase_validation["status"] != "PASS":
        raise RuntimeError(
            "neither one live Phase-B launcher nor a valid completed "
            "Phase-B result is available"
        )
    free = shutil.disk_usage(HCP_ROOT).free
    if free < BASE.MINIMUM_FREE_TO_LAUNCH_BYTES:
        raise RuntimeError("free storage is below the pretract launch floor")
    return {
        "static": static,
        "human": human,
        "phase_pid": phase_pids[0] if phase_pids else None,
        "phase_validation": phase_validation,
        "free_bytes": free,
        "prior_supervisor": latest_supervisor_state(),
        "hroi_source_policy": hroi_source_policy,
    }


def supervise(
    *,
    attempt_root: Path,
    human_qc: Path,
    phase_pid: int | None,
) -> int:
    validated = validate_resume(human_qc)
    phase_completed_between_launch_and_attach = bool(
        phase_pid is not None
        and validated["phase_pid"] is None
        and not process_alive(phase_pid)
        and validated["phase_validation"]["status"] == "PASS"
    )
    if (
        validated["phase_pid"] != phase_pid
        and not phase_completed_between_launch_and_attach
    ):
        raise RuntimeError(
            "Phase-B process identity changed between launch and supervision"
        )
    state_path = attempt_root / "supervisor_state.json"
    state: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_parallel_pretract_supervisor"
        ),
        "status": "PHASE_B_ATTACHED_AND_PRETRACT_RUNNING",
        "started_utc": utc_now(),
        "updated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "recovery_mode": "PRETRACT_ONLY_ATTACH_EXISTING_PHASE_B",
        "human_visual_qc": validated["human"],
        "hroi_source_policy": validated["hroi_source_policy"],
        "prior_supervisor": validated["prior_supervisor"],
        "phase_b": {
            "pid": phase_pid,
            "attached_existing_process": phase_pid is not None,
            "started_by_this_supervisor": False,
            "returncode": None if phase_pid is not None else 0,
        },
        "pretract": {
            "queue": [str(item["name"]) for item in BASE.LANES],
            "active": {},
            "terminal": {},
        },
    }
    atomic_json(state_path, state)

    queue = list(BASE.LANES)
    active: dict[
        str, tuple[Mapping[str, Any], subprocess.Popen[Any], Any, Path]
    ] = {}
    terminal: dict[str, Any] = {}
    emergency = False
    while queue or active or process_alive(phase_pid):
        free = shutil.disk_usage(HCP_ROOT).free
        if free < BASE.EMERGENCY_FREE_BYTES:
            emergency = True
            for _, process, _, _ in active.values():
                terminate_group(process)
        while (
            queue
            and len(active) < BASE.MAX_ACTIVE_PRETRACT_SUBJECTS
            and free >= BASE.MINIMUM_FREE_TO_LAUNCH_BYTES
            and not emergency
        ):
            item = queue.pop(0)
            name = str(item["name"])
            log_path = attempt_root / f"{name}.log"
            process, handle = BASE.start_process(
                BASE.lane_command(item, human_qc), log_path
            )
            active[name] = (item, process, handle, log_path)
        for name, (item, process, handle, log_path) in list(
            active.items()
        ):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            terminal[name] = {
                "returncode": returncode,
                "log": str(log_path),
                "summary_state": BASE.summary_state(item),
            }
            del active[name]
        if process_alive(phase_pid):
            phase_returncode = None
            phase_validation = None
        else:
            phase_validation = phase_validation_state()
            phase_returncode = phase_validation["returncode"]
        update_state(
            state_path,
            state,
            status=(
                "EMERGENCY_STORAGE_STOP"
                if emergency
                else "PHASE_B_ATTACHED_AND_PRETRACT_RUNNING"
            ),
            free_bytes=free,
            phase_b={
                **state["phase_b"],
                "returncode": phase_returncode,
                "validation": phase_validation,
            },
            pretract={
                "queue": [str(item["name"]) for item in queue],
                "active": {
                    name: {
                        "pid": process.pid,
                        "log": str(log_path),
                    }
                    for name, (_, process, _, log_path) in active.items()
                },
                "terminal": terminal,
            },
        )
        if emergency:
            break
        if not queue and not active and phase_returncode is not None:
            break
        time.sleep(15)

    final_phase = phase_validation_state()
    lane_pass = bool(
        len(terminal) == len(BASE.LANES)
        and all(
            row.get("returncode") == 0
            and row.get("summary_state", {}).get("status") == "PASS"
            for row in terminal.values()
        )
    )
    phase_pass = final_phase["status"] == "PASS"
    final_status = (
        "PASS_PHASE_B_AND_ALL_PRETRACT"
        if lane_pass and phase_pass and not emergency
        else (
            "EMERGENCY_STORAGE_STOP"
            if emergency
            else "FAIL_PHASE_B_OR_PRETRACT"
        )
    )
    update_state(
        state_path,
        state,
        status=final_status,
        completed_utc=utc_now(),
        phase_b={
            **state["phase_b"],
            "returncode": final_phase["returncode"],
            "validation": final_phase,
        },
        pretract={"queue": [], "active": {}, "terminal": terminal},
    )
    subprocess.run(
        [str(PYTHON), str(STATUS_SOURCE)],
        check=False,
        capture_output=True,
        text=True,
    )
    return 0 if final_status == "PASS_PHASE_B_AND_ALL_PRETRACT" else 1


def launch(human_qc: Path) -> dict[str, Any]:
    validated = validate_resume(human_qc)
    attempt_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + f"-pretract-only-recovery4-{os.urandom(4).hex()}"
    )
    attempt_root = ATTEMPTS / attempt_id
    attempt_root.mkdir(parents=True, exist_ok=False)
    command = [
        str(PYTHON),
        str(Path(__file__)),
        "--supervise-pretract-only",
        "--attempt-root",
        str(attempt_root),
        "--human-qc-manifest",
        str(human_qc.resolve()),
    ]
    if validated["phase_pid"] is not None:
        command.extend(("--phase-pid", str(validated["phase_pid"])))
    launch_record = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_pretract_only_recovery_launch"
        ),
        "status": "STARTING_SUPERVISOR",
        "attempt_id": attempt_id,
        "created_utc": utc_now(),
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "phase_b_started_by_this_launcher": False,
        "phase_b_pid": validated["phase_pid"],
        "human_visual_qc": BASE.PHASE.file_record(human_qc),
        "hroi_source_policy": validated["hroi_source_policy"],
        "prior_supervisor": validated["prior_supervisor"],
        "launcher": BASE.PHASE.file_record(Path(__file__)),
        "base_parallel_contract": BASE.PHASE.file_record(BASE_SOURCE),
        "supervisor_command": command,
    }
    launch_path = attempt_root / "launch.json"
    atomic_json(launch_path, launch_record)
    supervisor_log = attempt_root / "supervisor.log"
    handle = supervisor_log.open("x", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=EXP,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    handle.close()
    launch_record.update(
        {
            "status": "SUPERVISOR_STARTED",
            "supervisor_pid": process.pid,
            "supervisor_log": str(supervisor_log),
        }
    )
    atomic_json(launch_path, launch_record)
    return launch_record


def preflight(human_qc: Path) -> dict[str, Any]:
    try:
        validated = validate_resume(human_qc)
    except Exception as exc:
        return {
            "status": "NOT_READY",
            "error": f"{type(exc).__name__}:{exc}",
            "imaging_executed": False,
        }
    return {
        "status": "READY_TO_RESUME_PRETRACT_ONLY",
        "generated_utc": utc_now(),
        "imaging_executed": False,
        "phase_b_pid": validated["phase_pid"],
        "phase_validation": validated["phase_validation"],
        "free_bytes": validated["free_bytes"],
        "prior_supervisor": validated["prior_supervisor"],
        "hroi_source_policy": validated["hroi_source_policy"],
        "lane_count": len(BASE.LANES),
        "target_n": sum(int(item["target_n"]) for item in BASE.LANES),
        "maximum_active_pretract_subjects": (
            BASE.MAX_ACTIVE_PRETRACT_SUBJECTS
        ),
    }


def self_test() -> None:
    if (
        len(BASE.LANES) != 6
        or sum(int(item["target_n"]) for item in BASE.LANES) != 515
        or BASE.MAX_ACTIVE_PRETRACT_SUBJECTS != 4
        or any(
            "--compact-reproducible-after-pass"
            not in BASE.lane_command(item, DEFAULT_HUMAN_QC)
            for item in BASE.LANES
        )
    ):
        raise AssertionError("pretract-only recovery contract differs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--launch-pretract-only", action="store_true")
    mode.add_argument("--supervise-pretract-only", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--human-qc-manifest", type=Path, default=DEFAULT_HUMAN_QC
    )
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--phase-pid", type=int)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("HCP379_PRETRACT_ONLY_RECOVERY4_V4_SELF_TEST_PASS")
        return 0
    if args.supervise_pretract_only:
        if args.attempt_root is None:
            parser.error(
                "--supervise-pretract-only requires --attempt-root"
            )
        return supervise(
            attempt_root=args.attempt_root.resolve(),
            human_qc=args.human_qc_manifest.resolve(),
            phase_pid=args.phase_pid,
        )
    payload = (
        launch(args.human_qc_manifest.resolve())
        if args.launch_pretract_only
        else preflight(args.human_qc_manifest.resolve())
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") != "NOT_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
