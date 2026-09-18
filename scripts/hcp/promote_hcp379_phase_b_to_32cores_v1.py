#!/home/ec2-user/fsl/bin/python
"""Promote live Phase B from one to two concurrent 16-thread jobs safely.

The current Snakemake scheduler is stopped so it cannot enqueue another job,
its already-running heavy child is allowed to finish, and only then is the
scheduler terminated and resumed against existing outputs with 32 cores.  No
tractogram, matrix, or completed metadata file is removed.
"""

from __future__ import annotations

import argparse
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
RUNNER = EXP / "scripts/hcp/run_hcp379_phase_b_stability_v2.py"
HUMAN_QC = Path(
    "/data/derivatives/hcp379_v2/"
    "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
ROOT = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_stability_recovery4/32core_promotion_v1"
)
HEAVY_MARKERS = ("tckgen", "tcksift2", "tck2connectome")


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


def children(pid: int) -> list[int]:
    task_root = Path(f"/proc/{pid}/task")
    if not task_root.is_dir():
        return []
    observed: set[int] = set()
    for path in task_root.glob("*/children"):
        try:
            observed.update(
                int(value) for value in path.read_text().split()
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
    return sorted(observed)


def descendants(pid: int) -> set[int]:
    observed: set[int] = set()
    pending = list(children(pid))
    while pending:
        child = pending.pop()
        if child in observed:
            continue
        observed.add(child)
        pending.extend(children(child))
    return observed


def heavy_descendants(pid: int) -> list[dict[str, Any]]:
    rows = []
    for child in sorted(descendants(pid)):
        command = cmdline(child)
        if any(marker in command for marker in HEAVY_MARKERS):
            rows.append({"pid": child, "cmdline": command})
    return rows


def validation(phase_pid: int, snakemake_pid: int) -> dict[str, Any]:
    phase_cmd = cmdline(phase_pid)
    snake_cmd = cmdline(snakemake_pid)
    ncpu = os.cpu_count() or 0
    checks = {
        "phase_parent_is_exact_runner": (
            "run_hcp379_phase_b_stability_v2.py" in phase_cmd
        ),
        "scheduler_is_exact_phase_b_snakemake": (
            "snakemake" in snake_cmd
            and "Snakefile_h04a_r1_retry4_phase_b_hcp379_recovery4"
            in snake_cmd
        ),
        "scheduler_is_descendant_of_phase_parent": (
            snakemake_pid in descendants(phase_pid)
        ),
        "host_has_at_least_64_cpus": ncpu >= 64,
        "resume_runner_and_human_qc_exist": (
            RUNNER.is_file() and HUMAN_QC.is_file()
        ),
    }
    passed = sum(checks.values())
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_phase_b_32core_promotion_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": utc_now(),
        "phase_pid": phase_pid,
        "snakemake_pid": snakemake_pid,
        "phase_cmdline": phase_cmd,
        "snakemake_cmdline": snake_cmd,
        "host_cpu_n": ncpu,
        "heavy_descendants": heavy_descendants(snakemake_pid),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": {
            name: "PASS" if value else "FAIL"
            for name, value in checks.items()
        },
        "imaging_executed": False,
        "completed_outputs_removed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--phase-pid", type=int, required=True)
    parser.add_argument("--snakemake-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    if not 5 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be in 5..60")
    check = validation(args.phase_pid, args.snakemake_pid)
    if args.validate_only:
        print(json.dumps(check, indent=2, sort_keys=True))
        return 0 if check["status"] == "PASS" else 1
    if check["status"] != "PASS":
        raise SystemExit("promotion validation failed")

    attempt = (
        ROOT
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "-promote-32cores"
        )
    )
    attempt.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_phase_b_32core_promotion",
        "status": "STOPPING_SCHEDULER_AFTER_ACTIVE_CHILD",
        "started_utc": utc_now(),
        "attempt_root": str(attempt.resolve()),
        "validation": check,
        "completed_outputs_removed": False,
        "diagnosis_labels_used": False,
    }
    atomic_json(attempt / "state.json", state)

    os.kill(args.snakemake_pid, signal.SIGSTOP)
    state["scheduler_stopped_utc"] = utc_now()
    atomic_json(attempt / "state.json", state)
    while heavy_descendants(args.snakemake_pid):
        time.sleep(args.poll_seconds)

    state["active_heavy_child_completed_utc"] = utc_now()
    state["status"] = "TERMINATING_24CORE_SCHEDULER"
    atomic_json(attempt / "state.json", state)
    os.kill(args.snakemake_pid, signal.SIGCONT)
    os.kill(args.snakemake_pid, signal.SIGTERM)

    deadline = time.monotonic() + 180
    while Path(f"/proc/{args.phase_pid}").exists():
        if time.monotonic() > deadline:
            state["status"] = "FAIL"
            state["error"] = "old Phase-B parent did not exit in 180 seconds"
            atomic_json(attempt / "state.json", state)
            return 1
        time.sleep(2)

    state["old_phase_parent_exited_utc"] = utc_now()
    state["status"] = "RUNNING_32CORE_RESUME"
    atomic_json(attempt / "state.json", state)
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
            "completed_outputs_removed": False,
        }
    )
    atomic_json(attempt / "state.json", state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
