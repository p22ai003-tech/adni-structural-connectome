#!/home/ec2-user/fsl/bin/python
"""Finish one protected 10M job and resume Phase B with text counters fixed."""

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

from tck_header_count_v1 import read_count


UNIT = "129_S_6784_I1482244"
RUN_ID = "independent_10m"
EXPECTED = 10_000_000
LAUNCHER_PID = 1_202_169
SNAKEMAKE_PID = 1_202_654
SHELL_PID = 1_202_924
TCKGEN_PID = 1_202_938
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
RUN_DIR = RUN_ROOT / "subjects" / UNIT / "09_hcp379_stability" / RUN_ID
PARTIAL = RUN_DIR / "tracks.tck.partial.tck"
RESCUE = RUN_DIR / "tracks.tck.rescue-before-text-counter-patch"
FINAL = RUN_DIR / "tracks.tck"
METADATA = RUN_DIR / "tractography_parameters.json"
PHASE_RULES = Path(
    "/home/ec2-user/exp/scforge/workflow/extensions/"
    "h04a_r1_retry4_phase_b_hcp379/07_hcp379_stability.smk"
)
VALUE_COUNTER = Path(
    "/home/ec2-user/exp/scripts/hcp/count_mrtrix_numeric_values_v1.py"
)
STATE_ROOT = Path(
    "/data/derivatives/hcp379_v2/phase_b_stability_recovery4/"
    "text_counter_recovery_v1"
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_command(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(
            b"\0", b" "
        ).decode()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return None


def alive(pid: int) -> bool:
    command = process_command(pid)
    if command is None:
        return False
    try:
        fields = (
            Path(f"/proc/{pid}/stat")
            .read_text(encoding="utf-8")
            .rsplit(") ", 1)[1]
            .split()
        )
    except (FileNotFoundError, PermissionError, IndexError):
        return False
    return bool(fields) and fields[0] != "Z"


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


def update(status: str, **extra: Any) -> None:
    current: dict[str, Any] = {}
    if STATE.is_file():
        current = json.loads(STATE.read_text(encoding="utf-8"))
    current.update(extra)
    current.update(
        {
            "schema_version": "1.0.0",
            "record_type": "hcp379_phase_b_text_counter_recovery",
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


def validate_live_state() -> None:
    expected = {
        LAUNCHER_PID: "run_hcp379_phase_b_stability_v2.py",
        SNAKEMAKE_PID: "snakemake",
        SHELL_PID: "tracks.tck.partial.tck",
        TCKGEN_PID: "tckgen",
    }
    for pid, fragment in expected.items():
        command = process_command(pid)
        if command is None or fragment not in command:
            raise RuntimeError(f"PID {pid} is absent or differs: {command}")
    if not PARTIAL.is_file() or not RESCUE.is_file():
        raise FileNotFoundError("protected partial or rescue path is absent")
    if PARTIAL.stat().st_ino != RESCUE.stat().st_ino:
        raise RuntimeError("rescue path is not the live TCK inode")
    if not PHASE_RULES.is_file() or not VALUE_COUNTER.is_file():
        raise FileNotFoundError("patched Phase-B source is absent")
    source = PHASE_RULES.read_text(encoding="utf-8")
    if (
        "wc -l" in source
        or "tckinfo" in source
        or "count_mrtrix_numeric_values_v1.py" not in source
    ):
        raise ValueError("Phase-B rules are not fully text-counter patched")


def ensure_final_track() -> None:
    if FINAL.is_file():
        count = read_count(FINAL)
    elif RESCUE.is_file() and read_count(RESCUE) == EXPECTED:
        os.replace(RESCUE, FINAL)
        count = read_count(FINAL)
    else:
        raise RuntimeError("complete protected TCK is unavailable")
    if count != EXPECTED:
        raise RuntimeError(f"final TCK count is {count}, expected {EXPECTED}")
    if RESCUE.exists():
        if RESCUE.stat().st_ino != FINAL.stat().st_ino:
            raise RuntimeError("rescue and final TCK inodes differ")
        RESCUE.unlink()
    if not METADATA.is_file():
        atomic_json(
            METADATA,
            {
                "schema_version": "1.0.0",
                "record_type": "hcp379_stability_tractography_parameters",
                "diagnosis_labels_used": False,
                "unit": UNIT,
                "run_id": RUN_ID,
                "seed_class": "independent",
                "seed_id": (
                    "82804d867b34299ff6619452eaf37a9376b07bad338d4c9ba"
                    "0133fe9adca5ee3|85756ef5e6eb0ce13f8918981940795e621e"
                    "3607be12bc3c732fb2ee5f2de816|connectome-v2.1.0-"
                    "closed-world-canary-candidate|independent-replicate-1"
                ),
                "rng_seed": 936735008,
                "algorithm": "iFOD2",
                "act": True,
                "backtrack": True,
                "crop_at_gmwmi": True,
                "seeding": "seed_dynamic",
                "actual_step_size_mm": 0.45315,
                "requested_streamlines": EXPECTED,
                "actual_streamlines": count,
                "recovery_note": (
                    "Metadata reconstructed after scheduler pause for the "
                    "MRtrix text-counter compatibility patch."
                ),
            },
        )


def retire_old_attempt() -> None:
    if alive(SNAKEMAKE_PID):
        os.kill(SNAKEMAKE_PID, signal.SIGTERM)
        os.kill(SNAKEMAKE_PID, signal.SIGCONT)
        if not wait_until_gone(SNAKEMAKE_PID, 120):
            raise RuntimeError("old Snakemake process did not terminate")
    if alive(LAUNCHER_PID) and not wait_until_gone(LAUNCHER_PID, 60):
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
    validate_live_state()
    update(
        "WAITING_FOR_PROTECTED_TCKGEN",
        started_utc=utc_now(),
        old_launcher_pid=LAUNCHER_PID,
        old_snakemake_pid=SNAKEMAKE_PID,
        live_tckgen_pid=TCKGEN_PID,
        partial=str(PARTIAL),
        rescue=str(RESCUE),
        patched_phase_rules_sha256=sha256_file(PHASE_RULES),
        numeric_value_counter_sha256=sha256_file(VALUE_COUNTER),
    )
    while alive(TCKGEN_PID):
        update(
            "WAITING_FOR_PROTECTED_TCKGEN",
            observed_header_count=read_count(RESCUE),
            observed_size_bytes=RESCUE.stat().st_size,
        )
        time.sleep(30)
    if not wait_until_gone(SHELL_PID, 120):
        raise RuntimeError("original track shell did not finish")
    ensure_final_track()
    update(
        "COMPLETE_TCK_VERIFIED_RETIRING_OLD_DAG",
        recovered_track=str(FINAL),
        recovered_metadata=str(METADATA),
        recovered_count=read_count(FINAL),
    )
    retire_old_attempt()
    resumed_pid = launch_patched_attempt()
    update(
        "PATCHED_TEXT_COUNTER_PHASE_B_RESUMED",
        resumed_launcher_pid=resumed_pid,
        resumed_launcher_log=str(RESUME_LOG),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    validate_live_state()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "unit": UNIT,
                    "current_header_count": read_count(RESCUE),
                    "same_inode": PARTIAL.stat().st_ino == RESCUE.stat().st_ino,
                    "phase_rules_sha256": sha256_file(PHASE_RULES),
                    "numeric_value_counter_sha256": sha256_file(VALUE_COUNTER),
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
