#!/home/ec2-user/fsl/bin/python
"""Replay compacted Recovery4 pretract outputs through the tractography consumer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRACT_SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "hcp379_scaleup_tractography_recovery4_v3.py"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_live_pretract_compaction_crosslane_v1/validation.json"
)
LANES = {
    "corrected_core": (
        Path("/data/derivatives/hcp379_v2/corrected_scaleup_recovery4"),
        216,
    ),
    "corrected_legacy_tensor": (
        Path(
            "/data/derivatives/hcp379_v2/"
            "corrected_legacy_tensor_recovery4"
        ),
        212,
    ),
    "corrected_freesurfer": (
        Path(
            "/data/derivatives/hcp379_v2/corrected_freesurfer_recovery4"
        ),
        1,
    ),
    "archive_calibration": (
        Path(
            "/data/derivatives/hcp379_v2/"
            "archive_corrected_calibration_recovery4"
        ),
        6,
    ),
    "corrected_archive_low": (
        Path(
            "/data/derivatives/hcp379_v2/"
            "corrected_archive_low_recovery4"
        ),
        28,
    ),
    "corrected_archive_D0": (
        Path(
            "/data/derivatives/hcp379_v2/"
            "corrected_archive_D0_recovery4"
        ),
        52,
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
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
        "sha256": sha256_file(resolved),
    }


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_tractography_recovery4_crosslane_audit",
        TRACT_SOURCE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(TRACT_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="require all 515 new pretract units and every lane summary PASS",
    )
    args = parser.parse_args()
    module = load_module()
    env = module.PRETRACT.environment(1)
    failures: list[str] = []
    lane_records: dict[str, Any] = {}
    total_valid = 0

    for lane, (root, expected_n) in LANES.items():
        compaction_paths = sorted(
            (root / "qc/pretract_compaction").glob("*.json")
        )
        units: list[dict[str, Any]] = []
        for path in compaction_paths:
            unit = path.stem
            try:
                record = module.validate_recovery4_pretract_compaction(
                    root, unit
                )
                if record is None:
                    raise ValueError("compaction record disappeared")
                paths = module.PRETRACT.spatial_paths(root, unit)
                step = module.ENGINE.tractography_spacing(
                    root=root,
                    unit=unit,
                    paths=paths,
                    env=env,
                )
                state_path = root / "qc/subjects" / f"{unit}.json"
                state = module.PRETRACT.load_json(state_path)
                if (
                    state.get("unit") != unit
                    or state.get("status") != "PASS_PRETRACT_RECOVERY4"
                    or state.get("diagnosis_labels_used") is not False
                    or not math.isfinite(step)
                    or step <= 0
                ):
                    raise ValueError("pretract state or step contract differs")
                units.append(
                    {
                        "unit": unit,
                        "status": "PASS_CONSUMER_REPLAY",
                        "retained_artifact_n": len(
                            record["retained_artifacts"]
                        ),
                        "step_size_mm": step,
                        "compaction": file_record(path),
                        "pretract_state": file_record(state_path),
                    }
                )
            except Exception as exc:
                failure = f"{lane}:{unit}:{type(exc).__name__}:{exc}"
                failures.append(failure)
                units.append(
                    {
                        "unit": unit,
                        "status": "FAIL_CONSUMER_REPLAY",
                        "error": failure,
                    }
                )
        summary_path = root / "manifests/pretract_recovery4_summary.json"
        summary_status = None
        if summary_path.is_file():
            summary_status = module.PRETRACT.load_json(summary_path).get(
                "status"
            )
        observed_n = len(compaction_paths)
        if observed_n > expected_n:
            failures.append(
                f"{lane}:observed_compactions={observed_n}>{expected_n}"
            )
        if args.require_complete and (
            observed_n != expected_n or summary_status != "PASS"
        ):
            failures.append(
                f"{lane}:incomplete:{observed_n}/{expected_n}:"
                f"summary={summary_status}"
            )
        lane_records[lane] = {
            "root": str(root),
            "expected_n": expected_n,
            "observed_n": observed_n,
            "consumer_replay_pass_n": sum(
                row["status"] == "PASS_CONSUMER_REPLAY" for row in units
            ),
            "summary_status": summary_status,
            "units": units,
        }
        total_valid += sum(
            row["status"] == "PASS_CONSUMER_REPLAY" for row in units
        )

    if total_valid == 0:
        failures.append("no_compacted_subject_available")
    complete = bool(
        total_valid == sum(expected for _, expected in LANES.values())
        and all(
            row["observed_n"] == row["expected_n"]
            and row["summary_status"] == "PASS"
            for row in lane_records.values()
        )
    )
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_live_pretract_compaction_crosslane_consumer_validation"
        ),
        "status": (
            "PASS_COMPLETE_515"
            if not failures and complete
            else (
                "PASS_CURRENT_SNAPSHOT"
                if not failures
                else "FAIL"
            )
        ),
        "generated_utc": utc_now(),
        "require_complete": args.require_complete,
        "expected_total_n": 515,
        "consumer_replay_pass_n": total_valid,
        "complete": complete,
        "failures": failures,
        "lanes": lane_records,
        "implementation": file_record(Path(__file__)),
        "consumer": file_record(TRACT_SOURCE),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "expected_total_n": result["expected_total_n"],
                "consumer_replay_pass_n": result[
                    "consumer_replay_pass_n"
                ],
                "complete": result["complete"],
                "failures": result["failures"],
                "lane_counts": {
                    lane: {
                        "observed_n": row["observed_n"],
                        "expected_n": row["expected_n"],
                    }
                    for lane, row in lane_records.items()
                },
                "output": str(OUTPUT),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
