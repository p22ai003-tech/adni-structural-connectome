#!/usr/bin/env python3
"""Preserve a live 10M TCK, retire the unsafe DAG, and resume with safe QC."""

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
from typing import Any

from tck_header_count_v1 import read_count


UNIT = "135_S_6840_I1263427"
RUN_ID = "primary_10m"
EXPECTED = 10_000_000
SNAKEMAKE_PID = 1_159_096
LAUNCHER_PID = 1_158_858
SHELL_PID = 1_187_116
TCKGEN_PID = 1_187_124
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/"
    "h04a_r1_recovery_20260719_retry4"
)
RUN_DIR = RUN_ROOT / "subjects" / UNIT / "09_hcp379_stability" / RUN_ID
PARTIAL = RUN_DIR / "tracks.tck.partial.tck"
RESCUE = RUN_DIR / "tracks.tck.rescue-after-tckinfo-segv"
FINAL = RUN_DIR / "tracks.tck"
METADATA = RUN_DIR / "tractography_parameters.json"
STATE_ROOT = Path(
    "/data/derivatives/hcp379_v2/phase_b_stability_recovery4/"
    "tckinfo_segv_recovery_v1"
)
STATE = STATE_ROOT / "state.json"
RESUME_LOG = STATE_ROOT / "resumed_launcher.log"
LAUNCH_COMMAND = [
    "/home/ec2-user/fsl/bin/python",
    "/home/ec2-user/exp/scripts/hcp/run_hcp379_phase_b_stability_v2.py",
    "--human-qc-manifest",
    (
        "/data/derivatives/hcp379_v2/"
        "review_recovery4_corrected_overlay/"
        "human_visual_qc_recovery4_corrected_overlay.csv"
    ),
    "--cores",
    "24",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def process_command(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return None


def alive(pid: int) -> bool:
    command = process_command(pid)
    if command is None:
        return False
    try:
        stat_fields = (
            Path(f"/proc/{pid}/stat")
            .read_text(encoding="utf-8")
            .rsplit(") ", 1)[1]
            .split()
        )
    except (FileNotFoundError, PermissionError, IndexError):
        return False
    # A zombie has completed and only awaits parent reaping. Treating it as
    # live deadlocks recovery when the parent scheduler is intentionally
    # SIGSTOPped.
    return bool(stat_fields) and stat_fields[0] != "Z"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def update(status: str, **extra: Any) -> None:
    current: dict[str, Any] = {}
    if STATE.is_file():
        current = json.loads(STATE.read_text(encoding="utf-8"))
    current.update(extra)
    current.update(
        {
            "schema_version": "1.0.0",
            "record_type": "hcp379_phase_b_tckinfo_segv_recovery",
            "status": status,
            "updated_utc": utc_now(),
            "unit": UNIT,
            "run_id": RUN_ID,
            "expected_streamlines": EXPECTED,
            "diagnosis_labels_used": False,
        }
    )
    atomic_json(STATE, current)


def wait_until_gone(pid: int, timeout_seconds: int) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(1)
    return not alive(pid)


def validate_processes() -> None:
    expected_fragments = {
        SNAKEMAKE_PID: "snakemake",
        LAUNCHER_PID: "run_hcp379_phase_b_stability_v2.py",
        SHELL_PID: "tracks.tck.partial.tck",
        TCKGEN_PID: "tckgen",
    }
    for pid, fragment in expected_fragments.items():
        command = process_command(pid)
        if command is None or fragment not in command:
            raise RuntimeError(f"PID {pid} is absent or differs: {command}")


def ensure_rescue_link() -> None:
    if not PARTIAL.is_file():
        raise FileNotFoundError(PARTIAL)
    if not RESCUE.exists():
        os.link(PARTIAL, RESCUE)
    if PARTIAL.stat().st_ino != RESCUE.stat().st_ino:
        raise RuntimeError("rescue path is not a hard link to the live TCK")


def promote_recovered_track() -> None:
    count = read_count(RESCUE)
    if count != EXPECTED:
        raise RuntimeError(f"rescued TCK count is {count}, expected {EXPECTED}")
    if FINAL.exists() or METADATA.exists():
        raise FileExistsError("refusing to overwrite an existing Phase-B output")
    os.replace(RESCUE, FINAL)
    record = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_stability_tractography_parameters",
        "diagnosis_labels_used": False,
        "unit": UNIT,
        "run_id": RUN_ID,
        "seed_class": "primary",
        "seed_id": (
            "2a1ef03090b3427d0a6b27f3b8e94b2f58e0026df856ef5dadb57c8081483b99"
            "|77afe0d1aa521d98386a0f7639e2be07377a6adcbb3dcc8e6dbc7105a0f9e093"
            "|connectome-v2.1.0-closed-world-canary-candidate"
        ),
        "rng_seed": 244223861,
        "algorithm": "iFOD2",
        "act": True,
        "backtrack": True,
        "crop_at_gmwmi": True,
        "seeding": "seed_dynamic",
        "actual_step_size_mm": 0.45315,
        "requested_streamlines": EXPECTED,
        "actual_streamlines": count,
        "recovery_note": (
            "Promoted from a same-inode hard link after MRtrix tckinfo 3.0.7 "
            "segfaulted during the original post-write count check."
        ),
    }
    atomic_json(METADATA, record)


def retire_old_attempt() -> None:
    if alive(SNAKEMAKE_PID):
        os.kill(SNAKEMAKE_PID, signal.SIGTERM)
        os.kill(SNAKEMAKE_PID, signal.SIGCONT)
        if not wait_until_gone(SNAKEMAKE_PID, 90):
            raise RuntimeError("old Snakemake process did not terminate")
    if alive(LAUNCHER_PID) and not wait_until_gone(LAUNCHER_PID, 90):
        os.kill(LAUNCHER_PID, signal.SIGTERM)
        if not wait_until_gone(LAUNCHER_PID, 30):
            raise RuntimeError("old Phase-B launcher did not terminate")


def launch_patched_attempt() -> int:
    with RESUME_LOG.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            LAUNCH_COMMAND,
            cwd="/home/ec2-user/exp",
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    return process.pid


def execute() -> None:
    validate_processes()
    ensure_rescue_link()
    os.kill(SNAKEMAKE_PID, signal.SIGSTOP)
    update(
        "LIVE_TCK_PROTECTED_SCHEDULER_PAUSED",
        started_utc=utc_now(),
        old_snakemake_pid=SNAKEMAKE_PID,
        old_launcher_pid=LAUNCHER_PID,
        live_tckgen_pid=TCKGEN_PID,
        partial=str(PARTIAL),
        rescue=str(RESCUE),
    )
    while alive(TCKGEN_PID):
        update(
            "WAITING_FOR_LIVE_TCKGEN",
            observed_header_count=read_count(RESCUE),
            observed_size_bytes=RESCUE.stat().st_size,
        )
        time.sleep(30)
    if not wait_until_gone(SHELL_PID, 180):
        raise RuntimeError("original shell did not finish its failed tckinfo check")
    promote_recovered_track()
    update(
        "COMPLETE_TCK_PROMOTED_RETIRING_OLD_DAG",
        recovered_track=str(FINAL),
        recovered_metadata=str(METADATA),
        recovered_count=read_count(FINAL),
    )
    retire_old_attempt()
    resumed_pid = launch_patched_attempt()
    update(
        "PATCHED_PHASE_B_RESUMED",
        resumed_launcher_pid=resumed_pid,
        resumed_launcher_log=str(RESUME_LOG),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    validate_processes()
    ensure_rescue_link()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "unit": UNIT,
                    "current_header_count": read_count(RESCUE),
                    "same_inode": PARTIAL.stat().st_ino == RESCUE.stat().st_ino,
                },
                sort_keys=True,
            )
        )
        return 0
    try:
        execute()
    except Exception as exc:
        update("FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
