#!/home/ec2-user/fsl/bin/python
"""Persistently resume Phase B at 32 cores after the old parent exits.

This watcher sends no signal and deletes nothing.  It waits for the identified
24-core Phase-B parent, whose Snakemake child is already draining gracefully,
then resumes the same DAG against existing outputs with 32 cores.
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
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
RUNNER = EXP / "scripts/hcp/run_hcp379_phase_b_stability_v2.py"
HUMAN_QC = Path(
    "/data/derivatives/hcp379_v2/"
    "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
ROOT = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_stability_recovery4/32core_resume_watcher_v1"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
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


def cmdline(pid: int) -> str:
    path = Path(f"/proc/{pid}/cmdline")
    if not path.is_file():
        return ""
    return path.read_bytes().replace(b"\x00", b" ").decode(
        "utf-8", errors="replace"
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    if not 5 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be in 5..60")
    initial = cmdline(args.phase_pid)
    if (
        initial
        and "run_hcp379_phase_b_stability_v2.py" not in initial
    ):
        raise SystemExit("phase PID is not the exact Phase-B runner")
    attempt = (
        ROOT
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "-wait-resume-32cores"
        )
    )
    attempt.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_phase_b_32core_resume_watcher",
        "status": "WAITING_FOR_OLD_PHASE_PARENT",
        "started_utc": utc_now(),
        "old_phase_pid": args.phase_pid,
        "old_phase_cmdline": initial,
        "target_cores": 32,
        "signals_sent": False,
        "completed_outputs_removed": False,
        "diagnosis_labels_used": False,
    }
    state_path = attempt / "state.json"
    atomic_json(state_path, state)
    last_update = time.monotonic()
    while Path(f"/proc/{args.phase_pid}").exists():
        time.sleep(args.poll_seconds)
        if time.monotonic() - last_update >= 300:
            state["last_wait_heartbeat_utc"] = utc_now()
            atomic_json(state_path, state)
            last_update = time.monotonic()
    state["old_phase_parent_exited_utc"] = utc_now()
    state["status"] = "RUNNING_32CORE_RESUME"
    atomic_json(state_path, state)
    command = [
        str(PYTHON),
        str(RUNNER),
        "--resume-existing-phase-b",
        "--human-qc-manifest",
        str(HUMAN_QC),
        "--cores",
        "32",
    ]
    log_path = attempt / "phase_b_32core.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND " + json.dumps(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    state.update(
        {
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "completed_utc": utc_now(),
            "returncode": result.returncode,
            "phase_b_32core_log": str(log_path.resolve()),
            "signals_sent": False,
            "completed_outputs_removed": False,
        }
    )
    atomic_json(state_path, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
