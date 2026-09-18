#!/usr/bin/env python3
"""Finish the 032 primary seed, then prioritize its independent companion.

This bounded recovery is used only because the live Phase-B scheduler chose a
third primary canary before completing a same-unit seed pair. The scheduler is
already SIGSTOP-paused while its running primary tckgen continues. After that
track is safely promoted, this script closes the old scheduler, runs only the
exact independent 10M target through the same Snakefile/config, verifies it,
and starts a fresh official Phase-B launcher to resume the full DAG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
SNAKEMAKE = EXP / ".venv_connectome_workflow/bin/snakemake"
SNAKEFILE = (
    EXP
    / "scforge/workflow/"
    "Snakefile_h04a_r1_retry4_phase_b_hcp379_recovery4"
)
CONFIG = Path(
    "/data/derivatives/hcp379_v2/phase_b_stability_recovery4/"
    "hcp379_phase_b_recovery4_promoted_v6_resolved_config.yaml"
)
OFFICIAL_LAUNCHER = (
    EXP / "scripts/hcp/run_hcp379_phase_b_stability_v2.py"
)
TCK_COUNTER = EXP / "scripts/hcp/tck_header_count_v1.py"
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/"
    "h04a_r1_recovery_20260719_retry4"
)
UNIT = "032_S_6804_I1230908"
PRIMARY = (
    RUN_ROOT
    / "subjects"
    / UNIT
    / "09_hcp379_stability/primary_10m/tracks.tck"
)
INDEPENDENT = (
    RUN_ROOT
    / "subjects"
    / UNIT
    / "09_hcp379_stability/independent_10m/tracks.tck"
)
HUMAN_QC = Path(
    "/data/derivatives/hcp379_v2/"
    "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/phase_b_stability_recovery4/"
    "pair_priority_recovery_v1/attempts"
)
EXPECTED = 10_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular file: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def process_state(pid: int) -> str | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("State:"):
                return line.split()[1]
    except FileNotFoundError:
        return None
    return None


def track_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    result = subprocess.run(
        [str(PYTHON), str(TCK_COUNTER), str(path), "--expect", str(EXPECTED)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    return int(result.stdout.strip())


def active_tckgen() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", "tckgen"],
        check=False,
        capture_output=True,
        text=True,
    )
    return [
        line
        for line in result.stdout.splitlines()
        if "pgrep -af" not in line
    ]


def stop_process(pid: int, timeout: float = 30.0) -> None:
    if not process_alive(pid):
        return
    os.kill(pid, signal.SIGTERM)
    if process_state(pid) == "T":
        os.kill(pid, signal.SIGCONT)
    deadline = time.monotonic() + timeout
    while process_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.5)
    if process_alive(pid):
        raise RuntimeError(f"process did not terminate: {pid}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--old-scheduler-pid", type=int, required=True)
    parser.add_argument("--old-launcher-pid", type=int, required=True)
    parser.add_argument(
        "--continue-after-old-dag-closed",
        action="store_true",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    stopped_scheduler_ready = (
        process_state(args.old_scheduler_pid) == "T"
    )
    closed_dag_ready = (
        args.continue_after_old_dag_closed
        and not process_alive(args.old_scheduler_pid)
        and not process_alive(args.old_launcher_pid)
        and track_count(PRIMARY) == EXPECTED
        and not active_tckgen()
    )
    preflight = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_phase_b_pair_priority_recovery_preflight",
        "status": (
            "READY"
            if stopped_scheduler_ready or closed_dag_ready
            else "NOT_READY"
        ),
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "unit": UNIT,
        "old_scheduler_pid": args.old_scheduler_pid,
        "old_scheduler_state": process_state(args.old_scheduler_pid),
        "old_launcher_pid": args.old_launcher_pid,
        "primary_count": track_count(PRIMARY),
        "independent_count": track_count(INDEPENDENT),
        "target": str(INDEPENDENT),
        "tractography_started": False,
        "continue_after_old_dag_closed": (
            args.continue_after_old_dag_closed
        ),
    }
    if not args.execute:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    if preflight["status"] != "READY":
        raise RuntimeError("old Phase-B scheduler is not safely stopped")
    attempt_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "-phase-b-pair-priority"
    )
    attempt = ATTEMPTS / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    state_path = attempt / "state.json"
    state = {
        **preflight,
        "attempt_id": attempt_id,
        "status": "WAITING_FOR_PRIMARY_10M",
        "started_utc": utc_now(),
        "implementation": file_record(Path(__file__)),
        "snakefile": file_record(SNAKEFILE),
        "config": file_record(CONFIG),
        "official_launcher": file_record(OFFICIAL_LAUNCHER),
        "tck_counter": file_record(TCK_COUNTER),
    }
    atomic_json(state_path, state)
    if not args.continue_after_old_dag_closed:
        while track_count(PRIMARY) != EXPECTED:
            if not process_alive(args.old_scheduler_pid):
                raise RuntimeError(
                    "stopped scheduler disappeared before primary promotion"
                )
            time.sleep(args.poll_seconds)
        state.update(
            {
                "status": "PRIMARY_10M_PROMOTED_STOPPING_OLD_DAG",
                "updated_utc": utc_now(),
                "primary": file_record(PRIMARY),
            }
        )
        atomic_json(state_path, state)
        deadline = time.monotonic() + 120
        while active_tckgen() and time.monotonic() < deadline:
            time.sleep(1)
        if active_tckgen():
            raise RuntimeError("tckgen still active after primary promotion")
        stop_process(args.old_scheduler_pid)
        stop_process(args.old_launcher_pid)
    else:
        state.update(
            {
                "status": "PRIMARY_10M_PROMOTED_OLD_DAG_CLOSED",
                "updated_utc": utc_now(),
                "primary": file_record(PRIMARY),
            }
        )
        atomic_json(state_path, state)
    if track_count(INDEPENDENT) != EXPECTED:
        state.update(
            {
                "status": "RUNNING_TARGETED_INDEPENDENT_10M",
                "updated_utc": utc_now(),
                "tractography_started": True,
            }
        )
        atomic_json(state_path, state)
        target_log = attempt / "targeted_independent_10m.log"
        command = [
            str(SNAKEMAKE),
            "--snakefile",
            str(SNAKEFILE),
            "--configfile",
            str(CONFIG),
            "--cores",
            "16",
            "--rerun-incomplete",
            "--rerun-triggers",
            "input",
            "params",
            "--printshellcmds",
            "--show-failed-logs",
            str(INDEPENDENT),
        ]
        with target_log.open("x", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=EXP,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode or track_count(INDEPENDENT) != EXPECTED:
            state.update(
                {
                    "status": "FAIL_TARGETED_INDEPENDENT_10M",
                    "updated_utc": utc_now(),
                    "returncode": result.returncode,
                    "target_log": file_record(target_log),
                }
            )
            atomic_json(state_path, state)
            return 1
        state["target_log"] = file_record(target_log)
    state.update(
        {
            "status": "PAIR_10M_PASS_RESUMING_OFFICIAL_DAG",
            "updated_utc": utc_now(),
            "independent": file_record(INDEPENDENT),
        }
    )
    atomic_json(state_path, state)
    resumed_log = attempt / "resumed_official_launcher.log"
    handle = resumed_log.open("x", encoding="utf-8")
    process = subprocess.Popen(
        [
            str(PYTHON),
            str(OFFICIAL_LAUNCHER),
            "--human-qc-manifest",
            str(HUMAN_QC),
            "--cores",
            "24",
        ],
        cwd=EXP,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    handle.close()
    state.update(
        {
            "status": "PAIR_10M_PASS_OFFICIAL_DAG_RESUMED",
            "updated_utc": utc_now(),
            "resumed_launcher_pid": process.pid,
            "resumed_launcher_log": str(resumed_log),
        }
    )
    atomic_json(state_path, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
