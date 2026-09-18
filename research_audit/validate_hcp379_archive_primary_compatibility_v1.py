#!/home/ec2-user/fsl/bin/python
"""Independently validate the archive primary-compatibility audit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP
    / "research_audit/audit_hcp379_archive_primary_compatibility_v1.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/hcp379_archive_primary_compatibility_v1/"
    "audit.json"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")
VALIDATION = OUTPUT.with_name("validation.json")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_module(SOURCE, "hcp379_archive_primary_compatibility")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {
        "status": "PASS" if passed else "FAIL",
        "evidence": evidence,
    }


def main() -> int:
    value = json.loads(OUTPUT.read_text(encoding="utf-8"))
    replay = AUDIT.build()
    compiled = subprocess.run(
        [str(PYTHON), "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    checks = {
        "source_compiles": check(
            "source_compiles",
            compiled.returncode == 0,
            {"returncode": compiled.returncode, "stderr": compiled.stderr},
        ),
        "output_replays_exactly_except_timestamp": check(
            "output_replays_exactly_except_timestamp",
            {
                key: item
                for key, item in value.items()
                if key != "generated_utc"
            }
            == {
                key: item
                for key, item in replay.items()
                if key != "generated_utc"
            },
            {"output": file_record(OUTPUT)},
        ),
        "exact_archive_identity_and_density_partition": check(
            "exact_archive_identity_and_density_partition",
            value.get("archive_existing_n") == 86
            and value.get("archive_existing_density_qualified_n") == 56,
            {
                "archive_n": value.get("archive_existing_n"),
                "density_qualified_n": value.get(
                    "archive_existing_density_qualified_n"
                ),
            },
        ),
        "historical_route_and_atlas_are_incompatible": check(
            "historical_route_and_atlas_are_incompatible",
            value.get("historical_route") == "NO_ACT_RESTORED_TRACK"
            and value.get("historical_route_n") == 86
            and value.get("historical_atlas_source_n") == 86
            and value.get("final_hroi_changed_voxel_subject_n") == 86,
            {
                "historical_route": value.get("historical_route"),
                "historical_route_n": value.get("historical_route_n"),
                "changed_subject_n": value.get(
                    "final_hroi_changed_voxel_subject_n"
                ),
            },
        ),
        "uniform_balanced_archive_rebuild_is_exact": check(
            "uniform_balanced_archive_rebuild_is_exact",
            value.get("balanced_archive_production_n") == 86
            and value.get("balanced_archive_lane_counts")
            == {
                "archive_calibration": 6,
                "corrected_archive_D0": 52,
                "corrected_archive_low": 28,
            }
            and all(value.get("implementation_markers", {}).values()),
            {
                "production_n": value.get("balanced_archive_production_n"),
                "lane_counts": value.get("balanced_archive_lane_counts"),
                "markers": value.get("implementation_markers"),
            },
        ),
        "audit_is_read_only_blind_and_nonoverwriting": check(
            "audit_is_read_only_blind_and_nonoverwriting",
            value.get("diagnosis_labels_used") is False
            and value.get("outcomes_used") is False
            and value.get("imaging_executed") is False
            and value.get("matrices_modified") is False
            and value.get("non_overwriting") is True,
            {
                key: value.get(key)
                for key in (
                    "diagnosis_labels_used",
                    "outcomes_used",
                    "imaging_executed",
                    "matrices_modified",
                    "non_overwriting",
                )
            },
        ),
    }
    passed = sum(item["status"] == "PASS" for item in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_archive_primary_compatibility_v1_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "source": file_record(SOURCE),
        "audit": file_record(OUTPUT),
    }
    AUDIT.atomic_json(VALIDATION, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
