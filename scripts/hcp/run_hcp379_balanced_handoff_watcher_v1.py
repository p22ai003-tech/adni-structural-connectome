#!/home/ec2-user/fsl/bin/python
"""Wait for the verified balanced release and prepare its analysis handoff.

This watcher performs no imaging and never modifies a dashboard.  Once the
independently verified 530-by-nine release exists, it invokes the immutable
handoff builder, loads the resulting handoff through the intended consumer
with all 4,770 matrix hashes enabled, and records a separate validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
sys.path.insert(0, str(EXP))

from connectome_analysis.hcp379_balanced_verified_handoff import (  # noqa: E402
    load_verified_balanced_handoff,
)


PYTHON = Path("/home/ec2-user/fsl/bin/python")
BUILDER = (
    EXP
    / "scripts/hcp/"
    "prepare_hcp379_balanced_verified_handoff_v1.py"
)
VALIDATOR = (
    EXP
    / "research_audit/"
    "validate_hcp379_balanced_verified_handoff_v1.py"
)
ROOT = (
    Path("/data/derivatives/hcp379_v2")
    / "balanced_release_v1/verified_analysis_handoff_v1"
)
CURRENT = ROOT / "current_handoff.json"
ATTEMPTS = ROOT / "watcher_v1/attempts"
EXPECTED_SUBJECTS = 530
EXPECTED_MATRIX_FILES = 4770


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or not resolved.is_absolute()
    ):
        raise FileNotFoundError(resolved)
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


def process_lines() -> list[str]:
    completed = subprocess.run(
        ["pgrep", "-af", Path(__file__).name],
        capture_output=True,
        text=True,
        check=False,
    )
    return [
        line
        for line in completed.stdout.splitlines()
        if "--watch" in line
        and "tmux new-session" not in line
        and not line.startswith(f"{os.getpid()} ")
    ]


def run_json(command: list[str]) -> tuple[int, dict[str, Any], str, str]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        value = json.loads(completed.stdout)
    except Exception:
        value = {}
    if not isinstance(value, dict):
        value = {}
    return (
        completed.returncode,
        value,
        completed.stdout,
        completed.stderr,
    )


def verify_current_handoff(attempt: Path) -> dict[str, Any]:
    pointer = load_json(CURRENT)
    if (
        pointer.get("record_type")
        != "hcp379_balanced_verified_analysis_handoff_pointer"
        or pointer.get("status") != "PASS"
    ):
        raise ValueError("current handoff pointer contract differs")
    manifest_record = pointer.get("handoff_manifest")
    if not isinstance(manifest_record, dict):
        raise ValueError("handoff manifest record missing")
    manifest_path = Path(str(manifest_record.get("path", ""))).resolve()
    if manifest_record != file_record(manifest_path):
        raise ValueError("handoff manifest file record differs")
    loaded = load_verified_balanced_handoff(
        manifest_path,
        verify_all_matrix_hashes=True,
    )
    total_values = {
        int(value)
        for value in loaded.subjects[
            "selected_balanced_total_streamlines"
        ]
    }
    per_seed_values = {
        int(value)
        for value in loaded.subjects["selected_per_seed_streamlines"]
    }
    if (
        len(loaded.subjects) != EXPECTED_SUBJECTS
        or len(loaded.matrices) != EXPECTED_MATRIX_FILES
        or len(total_values) != 1
        or len(per_seed_values) != 1
        or next(iter(total_values)) != 2 * next(iter(per_seed_values))
    ):
        raise ValueError("loaded handoff cohort/recipe differs")
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_balanced_verified_handoff_consumer_replay"
        ),
        "status": "PASS",
        "generated_utc": utc_now(),
        "handoff_pointer": file_record(CURRENT),
        "handoff_manifest": file_record(manifest_path),
        "subject_count": len(loaded.subjects),
        "matrix_file_count": len(loaded.matrices),
        "all_matrix_hashes_verified": True,
        "selected_balanced_total_streamlines": next(iter(total_values)),
        "selected_per_seed_streamlines": next(iter(per_seed_values)),
        "minimum_edge_density": float(
            loaded.subjects["edge_density"].astype(float).min()
        ),
        "minimum_connected_nodes": int(
            loaded.subjects["connected_nodes"].astype(int).min()
        ),
        "diagnosis_labels_used": False,
        "live_dashboard_modified": False,
        "live_dashboard_activation_authorized": False,
    }
    output = attempt / "consumer_replay.json"
    atomic_json(output, result)
    return {**result, "record": file_record(output)}


def update_state(
    path: Path,
    state: dict[str, Any],
    **updates: Any,
) -> None:
    state.update(updates)
    state["updated_utc"] = utc_now()
    atomic_json(path, state)


def watch(attempt: Path, poll_seconds: int) -> int:
    attempt.mkdir(parents=True, exist_ok=False)
    state_path = attempt / "watcher_state.json"
    state: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_balanced_verified_handoff_watcher",
        "status": "STARTING",
        "started_utc": utc_now(),
        "updated_utc": utc_now(),
        "attempt_root": str(attempt.resolve()),
        "expected_subject_count": EXPECTED_SUBJECTS,
        "expected_matrix_file_count": EXPECTED_MATRIX_FILES,
        "diagnosis_labels_used": False,
        "imaging_executed": False,
        "live_dashboard_modified": False,
        "live_dashboard_activation_authorized": False,
    }
    atomic_json(state_path, state)
    try:
        while not CURRENT.is_file():
            returncode, value, stdout, stderr = run_json(
                [str(PYTHON), str(BUILDER)]
            )
            (attempt / "latest_preflight.stdout.json").write_text(
                stdout, encoding="utf-8"
            )
            (attempt / "latest_preflight.stderr.log").write_text(
                stderr, encoding="utf-8"
            )
            status = value.get("status")
            update_state(
                state_path,
                state,
                status="WAITING_FOR_VERIFIED_BALANCED_RELEASE",
                latest_preflight=value,
            )
            if returncode != 0 or status == "INVALID_VERIFIED_RELEASE":
                raise RuntimeError(
                    f"handoff preflight failed: rc={returncode} {status}"
                )
            if status == "READY_TO_PREPARE_HANDOFF":
                update_state(
                    state_path,
                    state,
                    status="PREPARING_IMMUTABLE_HANDOFF",
                )
                (
                    build_rc,
                    build_value,
                    build_stdout,
                    build_stderr,
                ) = run_json(
                    [str(PYTHON), str(BUILDER), "--prepare"]
                )
                (attempt / "prepare.stdout.json").write_text(
                    build_stdout, encoding="utf-8"
                )
                (attempt / "prepare.stderr.log").write_text(
                    build_stderr, encoding="utf-8"
                )
                if (
                    build_rc != 0
                    or build_value.get("status") != "PASS"
                    or build_value.get("subject_count")
                    != EXPECTED_SUBJECTS
                    or build_value.get("matrix_file_count")
                    != EXPECTED_MATRIX_FILES
                ):
                    raise RuntimeError(
                        "handoff preparation failed: "
                        f"rc={build_rc} {build_stderr[-2000:]}"
                    )
                update_state(
                    state_path,
                    state,
                    status="PASS_HANDOFF_RUNNING_CONSUMER_REPLAY",
                    preparation=build_value,
                )
                break
            time.sleep(poll_seconds)
        replay = verify_current_handoff(attempt)
        (
            validator_rc,
            validator,
            validator_stdout,
            validator_stderr,
        ) = run_json([str(PYTHON), str(VALIDATOR)])
        (attempt / "implementation_validation.stdout.json").write_text(
            validator_stdout, encoding="utf-8"
        )
        (attempt / "implementation_validation.stderr.log").write_text(
            validator_stderr, encoding="utf-8"
        )
        if (
            validator_rc != 0
            or validator.get("status") != "PASS"
            or validator.get("passed_checks") != 7
        ):
            raise RuntimeError(
                "handoff implementation validation failed: "
                f"{validator_stderr[-2000:]}"
            )
        update_state(
            state_path,
            state,
            status="PASS_VERIFIED_ANALYSIS_HANDOFF",
            active_stage="COMPLETE",
            consumer_replay=replay,
            implementation_validation=validator,
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
        )
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if not 10 <= args.poll_seconds <= 600:
        parser.error("--poll-seconds must be in 10..600")
    if args.watch:
        if process_lines():
            raise RuntimeError("another handoff watcher is active")
        attempt = (
            args.attempt_root.resolve()
            if args.attempt_root is not None
            else ATTEMPTS / f"{stamp()}-balanced-handoff-watcher"
        )
        return watch(attempt, args.poll_seconds)
    returncode, preflight, _, stderr = run_json(
        [str(PYTHON), str(BUILDER)]
    )
    value = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_balanced_verified_handoff_watcher_preflight",
        "status": "ACTIVE" if process_lines() else "READY_TO_WATCH",
        "active_watchers": process_lines(),
        "builder_returncode": returncode,
        "builder_preflight": preflight,
        "builder_stderr": stderr[-2000:],
        "imaging_executed": False,
        "live_dashboard_modified": False,
    }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
