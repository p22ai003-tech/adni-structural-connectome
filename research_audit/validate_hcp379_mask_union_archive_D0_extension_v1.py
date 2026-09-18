#!/home/ec2-user/fsl/bin/python
"""Independently validate the archive-D0 mask-union extension audit."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
HCP = Path("/data/derivatives/hcp379_v2")
SUMMARY = (
    HCP
    / "pretract_failure_source_audit_v1/"
    "mask_union_archive_D0_extension_v1/summary.json"
)
SOURCE = (
    EXP
    / "research_audit/"
    "audit_hcp379_mask_union_archive_D0_extension_v1.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_mask_union_archive_D0_extension_v1/validation.json"
)
UNIT = "114_S_5234_I1130066"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    value = load_json(SUMMARY)
    result = value.get("result", {})
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    check(
        "scope_is_exact_and_diagnosis_blind",
        value.get("unit") == UNIT
        and value.get("route") == "seeded_ants_rigid_failover"
        and value.get("diagnosis_labels_used") is False
        and value.get("outcomes_used") is False
        and value.get("connectome_density_used") is False,
        {
            "unit": value.get("unit"),
            "route": value.get("route"),
        },
    )
    check(
        "source_hashes_are_unchanged",
        value.get("source_before") == value.get("source_after"),
        {
            "before": value.get("source_before"),
            "after": value.get("source_after"),
        },
    )
    check(
        "mask_expansion_is_bounded",
        1.0 <= float(result.get("mask_volume_ratio", 99.0)) <= 1.15,
        result.get("mask_volume_ratio"),
    )
    check(
        "added_shell_has_supported_signal",
        float(result.get("shell_to_positive_b0_median_ratio", 0.0))
        >= 2.0,
        result.get("shell_to_positive_b0_median_ratio"),
    )
    check(
        "unchanged_atlas_and_tissue_gates_pass",
        result.get("target_fixed_gate_pass") is True
        and float(result.get("union_left_cortex_inside", 0.0)) >= 0.8
        and float(result.get("union_right_cortex_inside", 0.0)) >= 0.8
        and float(result.get("union_tissue_dice", 0.0)) >= 0.7,
        {
            "left": result.get("union_left_cortex_inside"),
            "right": result.get("union_right_cortex_inside"),
            "tissue_dice": result.get("union_tissue_dice"),
        },
    )
    artifacts = result.get("artifacts", {})
    artifact_errors = []
    for name, record in artifacts.items():
        path = Path(str(record.get("path", ""))).resolve()
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record.get("size_bytes")
            or sha256(path) != record.get("sha256")
        ):
            artifact_errors.append(name)
    check(
        "all_audit_artifacts_hash_replay",
        len(artifacts) == 10 and not artifact_errors,
        {"artifact_n": len(artifacts), "errors": artifact_errors},
    )

    passed = sum(row["status"] == "PASS" for row in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_mask_union_archive_D0_extension_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "records": {
            "summary": {
                "path": str(SUMMARY.resolve()),
                "size_bytes": SUMMARY.stat().st_size,
                "sha256": sha256(SUMMARY),
            },
            "audit_source": {
                "path": str(SOURCE.resolve()),
                "size_bytes": SOURCE.stat().st_size,
                "sha256": sha256(SOURCE),
            },
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
