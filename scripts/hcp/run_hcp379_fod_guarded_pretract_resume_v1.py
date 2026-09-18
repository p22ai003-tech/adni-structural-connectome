#!/home/ec2-user/fsl/bin/python
"""Resume the two unfinished pretract lanes under the FOD-order guard.

The earlier six-lane supervisor was coupled to a now-absent Phase-B process.
This launcher deliberately does not infer a Phase-B result.  It resumes only
the corrected-core and corrected-legacy/tensor pretract lanes, whose own
execute-mode gates independently validate human QC, HROI policy, inputs,
frozen responses, the current mechanism-specific failure ledger, and the
530-subject FOD-order compatibility audit.
"""

from __future__ import annotations

import argparse
import hashlib
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
CORE_SOURCE = EXP / "scripts/hcp/hcp379_scaleup_pretract_recovery4_v3.py"
LEGACY_SOURCE = (
    EXP / "scripts/hcp/run_hcp379_legacy_density_pretract_recovery4_v3.py"
)
DEFAULT_HUMAN_QC = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
ATTEMPTS = (
    HCP_ROOT / "fod_order_guarded_pretract_resume_v1/attempts"
)
STATUS_SOURCE = EXP / "scripts/hcp/hcp379_v2_status.py"
FOD_VALIDATION = (
    EXP
    / "research_audit/outputs/hcp379_fod_order_compatibility_v1/"
    "validation.json"
)
LEDGER_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_pretract_failure_recovery_ledger_v1/validation.json"
)
MINIMUM_FREE_BYTES = 500 * 1024**3
WORKER_THREADS = 8


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_module(CORE_SOURCE, "hcp379_fod_guarded_resume_core")
LEGACY = load_module(
    LEGACY_SOURCE, "hcp379_fod_guarded_resume_legacy"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


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
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def active_workers() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", "hcp379_.*pretract_recovery4"],
        check=False,
        capture_output=True,
        text=True,
    )
    active: list[str] = []
    for raw in result.stdout.splitlines():
        pid_text, _, command = raw.partition(" ")
        if (
            not pid_text.isdigit()
            or int(pid_text) == os.getpid()
            or "pgrep -af" in command
            or "--execute" not in command
        ):
            continue
        if str(CORE_SOURCE) in command or str(LEGACY_SOURCE) in command:
            active.append(raw)
    return active


def worker_commands(human_qc: Path) -> dict[str, list[str]]:
    common = [
        "--execute",
        "--human-qc-manifest",
        str(human_qc.resolve()),
        "--workers",
        "1",
        "--nthreads",
        str(WORKER_THREADS),
        "--compact-reproducible-after-pass",
    ]
    return {
        "corrected_core_216_guarded": [
            str(PYTHON),
            str(CORE_SOURCE),
            *common,
        ],
        "corrected_legacy_tensor_212_guarded": [
            str(PYTHON),
            str(LEGACY_SOURCE),
            *common,
        ],
    }


def validate(human_qc: Path) -> dict[str, Any]:
    active = active_workers()
    if active:
        raise RuntimeError(
            "pretract worker already active: " + " | ".join(active)
        )
    if not human_qc.is_file():
        raise FileNotFoundError(human_qc)
    fod_validation = load_json(FOD_VALIDATION)
    ledger_validation = load_json(LEDGER_VALIDATION)
    if (
        fod_validation.get("status") != "PASS"
        or fod_validation.get("checks_passed")
        != fod_validation.get("checks_total")
        or ledger_validation.get("status") != "PASS"
        or ledger_validation.get("checks_passed")
        != ledger_validation.get("checks_total")
    ):
        raise ValueError("FOD-order or failure-ledger validation is not PASS")

    core_gate = CORE.validate_recovery4_gate(
        human_qc=human_qc, execute=True
    )
    core_rows, _ = CORE.validate_audit_rows(core_gate)
    legacy_contract = LEGACY.lane_contract()
    LEGACY.specialize(LEGACY.OUTPUT_ROOT)
    legacy_gate = LEGACY.PRETRACT.validate_recovery4_gate(
        human_qc=human_qc, execute=True
    )
    legacy_rows, _ = LEGACY.PRETRACT.validate_audit_rows(
        legacy_gate
    )
    free_bytes = shutil.disk_usage(HCP_ROOT).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"free storage {free_bytes} is below {MINIMUM_FREE_BYTES}"
        )
    if (
        len(core_rows) != 216
        or len(legacy_rows) != 212
        or len(core_gate["fod_order_hold_units"]) != 10
        or len(legacy_gate["fod_order_hold_units"]) != 10
    ):
        raise ValueError("guarded two-lane execution contract differs")
    return {
        "status": "PASS_GUARDED_TWO_LANE_PREFLIGHT",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "non_overwriting": True,
        "imaging_executed": False,
        "free_bytes": free_bytes,
        "core_target_n": len(core_rows),
        "legacy_target_n": len(legacy_rows),
        "legacy_contract": {
            "lane_n": len(legacy_contract["units"]),
            "canary_overlap_n": len(
                legacy_contract["canary_overlap"]
            ),
            "execution_n": len(
                legacy_contract["execution_units"]
            ),
        },
        "fod_order_hold_n": len(core_gate["fod_order_hold_units"]),
        "mechanism_hold_n": len(
            core_gate["mechanism_hold_by_unit"]
        ),
        "records": {
            "human_qc": file_record(human_qc),
            "core_runner": file_record(CORE_SOURCE),
            "legacy_runner": file_record(LEGACY_SOURCE),
            "fod_order_validation": file_record(FOD_VALIDATION),
            "failure_ledger_validation": file_record(
                LEDGER_VALIDATION
            ),
            "launcher": file_record(Path(__file__)),
        },
    }


def terminate_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def supervise(attempt_root: Path, human_qc: Path) -> int:
    preflight = validate(human_qc)
    state_path = attempt_root / "state.json"
    state: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_fod_order_guarded_pretract_resume_supervisor"
        ),
        "status": "STARTING",
        "started_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "non_overwriting": True,
        "phase_b_started_or_inferred": False,
        "preflight": preflight,
        "workers": {},
    }
    atomic_json(state_path, state)
    processes: dict[
        str, tuple[subprocess.Popen[Any], Any, Path, list[str]]
    ] = {}
    try:
        for name, command in worker_commands(human_qc).items():
            log = attempt_root / f"{name}.log"
            handle = log.open("x", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=EXP,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            processes[name] = (process, handle, log, command)
        state["status"] = "RUNNING"
        state["workers"] = {
            name: {
                "pid": process.pid,
                "status": "RUNNING",
                "returncode": None,
                "log": str(log),
                "command": command,
            }
            for name, (process, _handle, log, command) in processes.items()
        }
        atomic_json(state_path, state)
        while any(
            process.poll() is None
            for process, _handle, _log, _command in processes.values()
        ):
            state["updated_utc"] = utc_now()
            for name, (
                process,
                _handle,
                _log,
                _command,
            ) in processes.items():
                state["workers"][name]["returncode"] = process.poll()
                state["workers"][name]["status"] = (
                    "RUNNING"
                    if process.poll() is None
                    else "TERMINAL"
                )
            atomic_json(state_path, state)
            time.sleep(15)
    except BaseException:
        for process, _handle, _log, _command in processes.values():
            terminate_group(process)
        raise
    finally:
        for process, handle, _log, _command in processes.values():
            if process.poll() is None:
                process.wait()
            handle.close()

    returncodes = {
        name: process.returncode
        for name, (process, _handle, _log, _command) in processes.items()
    }
    subprocess.run(
        [str(PYTHON), str(STATUS_SOURCE)],
        cwd=EXP,
        check=False,
        capture_output=True,
        text=True,
    )
    state.update(
        {
            "status": (
                "PASS_WORKERS_TERMINAL"
                if all(code == 0 for code in returncodes.values())
                else "FAIL_WORKER_RETURN_CODE"
            ),
            "completed_utc": utc_now(),
        }
    )
    for name, code in returncodes.items():
        state["workers"][name]["returncode"] = code
        state["workers"][name]["status"] = "TERMINAL"
    atomic_json(state_path, state)
    return 0 if state["status"] == "PASS_WORKERS_TERMINAL" else 1


def launch(human_qc: Path) -> dict[str, Any]:
    preflight = validate(human_qc)
    attempt_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + f"-guarded-two-lane-{os.urandom(4).hex()}"
    )
    attempt_root = ATTEMPTS / attempt_id
    attempt_root.mkdir(parents=True, exist_ok=False)
    command = [
        str(PYTHON),
        str(Path(__file__).resolve()),
        "--supervise",
        "--attempt-root",
        str(attempt_root),
        "--human-qc-manifest",
        str(human_qc.resolve()),
    ]
    log = attempt_root / "supervisor.log"
    handle = log.open("x", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=EXP,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    handle.close()
    record = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_fod_order_guarded_pretract_resume_launch"
        ),
        "status": "SUPERVISOR_STARTED",
        "created_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "non_overwriting": True,
        "phase_b_started_or_inferred": False,
        "attempt_id": attempt_id,
        "attempt_root": str(attempt_root),
        "supervisor_pid": process.pid,
        "supervisor_log": str(log),
        "supervisor_command": command,
        "preflight": preflight,
    }
    atomic_json(attempt_root / "launch.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--supervise", action="store_true")
    parser.add_argument(
        "--human-qc-manifest", type=Path, default=DEFAULT_HUMAN_QC
    )
    parser.add_argument("--attempt-root", type=Path)
    args = parser.parse_args()
    if args.supervise:
        if args.attempt_root is None:
            parser.error("--supervise requires --attempt-root")
        return supervise(
            args.attempt_root.resolve(),
            args.human_qc_manifest.resolve(),
        )
    payload = (
        launch(args.human_qc_manifest.resolve())
        if args.launch
        else validate(args.human_qc_manifest.resolve())
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
