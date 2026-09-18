#!/usr/bin/env python3
"""Close the approved, abandoned H04A attempt without altering its artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path("/data/derivatives/scforge_v2/h04a_canary_20260718_v1")
ATTEMPT_ID = "20260718T132019.562821Z-response-calibration-phase-a-ba0c318b"
START_PATH = RUN_ROOT / "attempts" / f"{ATTEMPT_ID}.start.json"
LOG_PATH = RUN_ROOT / "attempts" / f"{ATTEMPT_ID}.snakemake.log"
END_PATH = RUN_ROOT / "attempts" / f"{ATTEMPT_ID}.end.json"
LAST_ARTIFACT = (
    RUN_ROOT
    / "subjects"
    / "168_S_6735_I1175371"
    / "01_dwi"
    / "dwi_preproc.mif"
)
RECOMMENDATION = (
    PROJECT_ROOT
    / "research_audit"
    / "decisions"
    / "sl_h04a_i1_interruption_resolution_recommendation_20260718.json"
)
APPROVAL = (
    PROJECT_ROOT
    / "research_audit"
    / "decisions"
    / "sl_h04a_i1_approval_20260718.json"
)

EXPECTED = {
    "recommendation_sha256": "abde9058e5fcb84bd4284f0ed8a505d3e65d778743dc88b38177b333bd852ca7",
    "start_sha256": "1d42f1420b7c9d80a807f126d64a60ccfd80add77d693548cc1b4b0dadb6d706",
    "log_sha256": "1388e9309a6fcf512e7a02d996f0f151ce1b48acae211c59ab508bf48b758277",
    "last_artifact_sha256": "dc7a3ceb92d0f245fd8db1f6c237a6772ea20b8c0c612a75fcdca38c4eed1ade",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_mapping(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return data


def ancestor_pids() -> set[int]:
    pids: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
            pid = int(fields[3])
        except (FileNotFoundError, IndexError, ValueError):
            break
    return pids


def matching_active_processes() -> list[dict[str, Any]]:
    ignored = ancestor_pids()
    needles = (
        "run_connectome_v2.py",
        "snakemake",
        "dwifslpreproc",
        "eddy_cpu",
        "eddy_openmp",
        "dwi2response",
    )
    matches: list[dict[str, Any]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in ignored:
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(RUN_ROOT) in command and any(needle in command for needle in needles):
            matches.append({"pid": int(proc.name), "command": command.strip()})
    return matches


def write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if END_PATH.exists():
        raise FileExistsError(f"attempt already has an immutable end: {END_PATH}")
    approval = load_mapping(APPROVAL)
    recommendation = load_mapping(RECOMMENDATION)
    start = load_mapping(START_PATH)
    if (
        approval.get("status") != "APPROVED"
        or approval.get("approved") is not True
        or approval.get("decision_id") != "SL-H04A-I1"
        or approval.get("user_response") != "Approve SL-H04A-I1 and SL-H04A-R1"
        or approval.get("approved_resolution")
        != "CLOSE_AS_INTERRUPTED_FAIL_PRESERVE_ALL_ARTIFACTS_NO_OLD_ROOT_RESTART"
    ):
        raise ValueError("I1 approval semantics differ from the exact user approval")
    if recommendation.get("recommendation") != approval["approved_resolution"]:
        raise ValueError("recommendation and approval resolutions differ")
    observed = {
        "recommendation_sha256": sha256_file(RECOMMENDATION),
        "start_sha256": sha256_file(START_PATH),
        "log_sha256": sha256_file(LOG_PATH),
        "last_artifact_sha256": sha256_file(LAST_ARTIFACT),
    }
    if observed != EXPECTED:
        raise ValueError(f"bound interruption evidence changed: {observed}")
    active = matching_active_processes()
    if active:
        raise RuntimeError(f"matching imaging processes remain active: {active}")
    if start.get("attempt_id") != ATTEMPT_ID or start.get("status") != "STARTED":
        raise ValueError("attempt start identity or state differs")
    if args.verify_only:
        print(json.dumps({"status": "PASS", "end_path": str(END_PATH)}, indent=2))
        return 0

    end = {
        "schema_version": "1.0.0",
        "status": "FAIL",
        "started_utc": start["started_utc"],
        "ended_utc": "2026-07-18T15:40:18+00:00",
        "duration_seconds": 8398.413256,
        "attempt_id": ATTEMPT_ID,
        "launcher_mode": start["launcher_mode"],
        "run_id": start["run_id"],
        "lineage_id": start["lineage_id"],
        "recipe_id": start["recipe_id"],
        "execution_binding": start["execution_binding"],
        "returncode": None,
        "attempt_start": file_record(START_PATH),
        "combined_execution_log": file_record(LOG_PATH),
        "outputs": {"first_completed_eddy_output": file_record(LAST_ARTIFACT)},
        "interruption_resolution": {
            "resolution_type": "human_approved_launcher_interruption",
            "reason": "launcher_control_process_absent_before_attempt_end",
            "approval": file_record(APPROVAL),
            "recommendation": file_record(RECOMMENDATION),
            "process_recheck_at_resolution": {"status": "PASS", "matches": []},
            "preserve_all_artifacts": True,
            "old_root_restart_authorized": False,
            "launcher_returncode_observed": False,
            "scientifically_sufficient_phase_a": False,
        },
    }
    write_exclusive_json(END_PATH, end)
    print(json.dumps({"status": "PASS", "attempt_end": file_record(END_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
