#!/home/ec2-user/fsl/bin/python
"""Refresh exact HCP379 topology, recovery ledger, and live status reports.

This diagnosis-blind monitor performs no imaging.  It serially rebuilds and
validates recovery overrides, the 515 production topology/ledger, and the
immutable 530 status-density snapshot, then updates the dashboard-facing live
status from those validated records.  In loop mode it repeats every ten minutes
by default and records each refresh atomically.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
FSL_PYTHON = Path("/home/ec2-user/fsl/bin/python")
SYSTEM_PYTHON = Path("/usr/bin/python3")
ROOT = Path(
    "/data/derivatives/hcp379_v2/audits/live_530_status_refresh_v1"
)
LOCK = ROOT / "refresh.lock"
COMMANDS = [
    (
        "build_recovery_overrides",
        [
            str(FSL_PYTHON),
            str(
                EXP
                / "research_audit/"
                "build_hcp379_pretract_recovery_overrides_v1.py"
            ),
        ],
    ),
    (
        "build_topology",
        [
            str(FSL_PYTHON),
            str(
                EXP
                / "scripts/hcp/"
                "build_hcp379_balanced_release_topology_v1.py"
            ),
        ],
    ),
    (
        "validate_topology",
        [
            str(FSL_PYTHON),
            str(
                EXP
                / "research_audit/"
                "validate_hcp379_balanced_release_topology_v1.py"
            ),
        ],
    ),
    (
        "build_failure_ledger",
        [
            str(SYSTEM_PYTHON),
            str(
                EXP
                / "research_audit/"
                "build_hcp379_pretract_failure_recovery_ledger_v1.py"
            ),
        ],
    ),
    (
        "validate_failure_ledger",
        [
            str(SYSTEM_PYTHON),
            str(
                EXP
                / "research_audit/"
                "validate_hcp379_pretract_failure_recovery_ledger_v1.py"
            ),
        ],
    ),
    (
        "build_live_530_report",
        [
            str(FSL_PYTHON),
            str(
                EXP
                / "research_audit/"
                "build_hcp379_live_530_status_density_v1.py"
            ),
        ],
    ),
    (
        "validate_live_530_report",
        [
            str(FSL_PYTHON),
            str(
                EXP
                / "research_audit/"
                "validate_hcp379_live_530_status_density_v1.py"
            ),
        ],
    ),
    (
        "write_dashboard_live_status",
        [
            str(FSL_PYTHON),
            str(EXP / "scripts/hcp/hcp379_v2_status.py"),
        ],
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


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


def refresh() -> dict[str, Any]:
    attempt = ROOT / "attempts" / stamp()
    attempt.mkdir(parents=True, exist_ok=False)
    steps: list[dict[str, Any]] = []
    for name, command in COMMANDS:
        log_path = attempt / f"{name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND " + json.dumps(command) + "\n")
            log.flush()
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        steps.append(
            {
                "name": name,
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "returncode": result.returncode,
                "log_path": str(log_path.resolve()),
            }
        )
        if result.returncode:
            break
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_live_530_status_refresh",
        "status": (
            "PASS"
            if len(steps) == len(COMMANDS)
            and all(step["status"] == "PASS" for step in steps)
            else "FAIL"
        ),
        "generated_utc": utc_now(),
        "attempt_root": str(attempt.resolve()),
        "steps": steps,
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "imaging_executed": False,
        "matrices_modified": False,
    }
    atomic_json(attempt / "result.json", payload)
    atomic_json(ROOT / "latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 300 <= args.interval_seconds <= 3600:
        parser.error("--interval-seconds must be in 300..3600")
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another live-530 refresh monitor is active")
        while True:
            result = refresh()
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
            if args.once:
                return 0 if result["status"] == "PASS" else 1
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
