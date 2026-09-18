#!/home/ec2-user/fsl/bin/python
"""Independent replay of the fixed HCP379 mask-union recovery audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np


SUMMARY = Path(
    "/data/derivatives/hcp379_v2/pretract_failure_source_audit_v1/"
    "mask_union_recovery_controls_v1/summary.json"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_mask_union_recovery_v1/validation.json"
)
TARGETS = {"003_S_0908_I1249292", "003_S_4350_I1252856"}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_record(value: Any) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError("file record differs")
    path = Path(str(value.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != value.get("size_bytes")
        or sha256_file(path) != value.get("sha256")
    ):
        raise ValueError(f"file record replay failed: {path}")
    return path


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def image_bool(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj) > 0


def atlas_coverage(atlas: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    left = np.isin(atlas, np.arange(1, 181))
    right = np.isin(atlas, np.arange(181, 361))
    return (
        float(np.logical_and(left, mask).sum() / left.sum()),
        float(np.logical_and(right, mask).sum() / right.sum()),
    )


def main() -> int:
    summary = load_json(SUMMARY)
    rows_path = verify_record(summary["rows"])
    rows = list(csv.DictReader(rows_path.open(newline="", encoding="utf-8")))
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    controls = [
        row for row in rows if row["role"] == "reviewed_canary_control"
    ]
    targets = [
        row for row in rows if row["role"] == "registration_failure_target"
    ]
    add(
        "exact_identity_partition",
        len(rows) == 17
        and len({row["unit"] for row in rows}) == 17
        and len(controls) == 15
        and {row["unit"] for row in targets} == TARGETS,
        {
            "row_n": len(rows),
            "control_n": len(controls),
            "target_units": sorted(row["unit"] for row in targets),
        },
    )
    policy = summary.get("policy", {})
    add(
        "fixed_policy_and_locked_thresholds",
        policy.get("original_mask_retained_by_union") is True
        and policy.get("dwi2mask_clean_scale") == 0
        and policy.get("native_voxel_dilation_passes") == 1
        and policy.get("minimum_hemisphere_atlas_inside") == 0.80
        and policy.get("minimum_tissue_dice") == 0.70
        and policy.get("maximum_mask_volume_ratio") == 1.15
        and policy.get("per_subject_parameter_tuning_allowed") is False,
        policy,
    )
    add(
        "diagnosis_outcome_density_blind",
        summary.get("diagnosis_labels_used") is False
        and summary.get("outcomes_used") is False
        and summary.get("connectome_density_used") is False
        and all(
            row["diagnosis_labels_used"] == "False"
            and row["outcomes_used"] == "False"
            and row["connectome_density_used"] == "False"
            for row in rows
        ),
        {
            "diagnosis_labels_used": summary.get("diagnosis_labels_used"),
            "outcomes_used": summary.get("outcomes_used"),
            "connectome_density_used": summary.get(
                "connectome_density_used"
            ),
        },
    )
    numeric_pass = all(
        1.0 <= float(row["mask_volume_ratio"]) <= 1.15
        and float(row["shell_to_positive_b0_median_ratio"]) >= 2.0
        and float(row["union_tissue_dice"]) >= 0.70
        for row in rows
    )
    add(
        "bounded_signal_supported_control_behavior",
        numeric_pass
        and all(
            float(row["union_all_atlas_inside"]) + 1.0e-12
            >= float(row["original_all_atlas_inside"])
            for row in controls
        ),
        {
            "maximum_mask_volume_ratio": max(
                float(row["mask_volume_ratio"]) for row in rows
            ),
            "minimum_shell_to_positive_b0_median_ratio": min(
                float(row["shell_to_positive_b0_median_ratio"])
                for row in rows
            ),
            "minimum_tissue_dice": min(
                float(row["union_tissue_dice"]) for row in rows
            ),
        },
    )
    replay: dict[str, Any] = {}
    target_values = {
        str(value["unit"]): value for value in summary.get("targets", [])
    }
    target_replay_pass = set(target_values) == TARGETS
    for unit in sorted(TARGETS):
        value = target_values[unit]
        artifacts = value["artifacts"]
        atlas_path = Path(
            "/data/derivatives/hcp379_v2/"
            "corrected_legacy_tensor_recovery4/subjects"
        ) / unit
        route = str(value["route"])
        atlas_path = (
            atlas_path
            / "03_spatial"
            / (
                "recovery4_ants"
                if route == "seeded_ants_rigid_failover"
                else "recovery4_ants_syn"
            )
            / "hcp379_nodes_b0_1mm.nii.gz"
        )
        union_1mm = image_bool(verify_record(artifacts["union_1mm"]))
        atlas = np.rint(
            np.asanyarray(nib.load(str(atlas_path)).dataobj)
        ).astype(np.int32)
        left, right = atlas_coverage(atlas, union_1mm)
        tissue = image_bool(verify_record(artifacts["five_tt_mask"]))
        union_native = image_bool(verify_record(artifacts["union_native"]))
        dice = float(
            2.0 * np.logical_and(tissue, union_native).sum()
            / (tissue.sum() + union_native.sum())
        )
        values_finite = all(math.isfinite(item) for item in (left, right, dice))
        passed = values_finite and left >= 0.80 and right >= 0.80 and dice >= 0.70
        target_replay_pass = target_replay_pass and passed
        replay[unit] = {
            "left_cortex_inside": left,
            "right_cortex_inside": right,
            "tissue_dice": dice,
            "status": "PASS" if passed else "FAIL",
        }
    add(
        "independent_target_image_replay",
        target_replay_pass,
        replay,
    )
    for name, record in summary.get("records", {}).items():
        verify_record(record)
    add(
        "summary_and_all_records_bind",
        summary.get("status") == "PASS"
        and all(value is True for value in summary.get("checks", {}).values()),
        {
            "summary_status": summary.get("status"),
            "summary_checks": summary.get("checks"),
        },
    )
    passed_n = sum(value["status"] == "PASS" for value in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_mask_union_recovery_validation",
        "status": "PASS" if passed_n == len(checks) else "FAIL",
        "generated_utc": (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        ),
        "passed_checks": passed_n,
        "total_checks": len(checks),
        "checks": checks,
        "summary": {
            "path": str(SUMMARY.resolve()),
            "size_bytes": SUMMARY.stat().st_size,
            "sha256": sha256_file(SUMMARY),
        },
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
