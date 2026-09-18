#!/home/ec2-user/fsl/bin/python
"""Build hash-bound production pretract overrides for isolated recoveries."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(
    "/data/derivatives/hcp379_v2/"
    "corrected_legacy_tensor_mask_union_recovery4"
)
ORIGINAL_ROOT = Path(
    "/data/derivatives/hcp379_v2/corrected_legacy_tensor_recovery4"
)
COMPACTION_SUMMARY = (
    ROOT / "manifests/mask_union_compaction_summary.json"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_pretract_recovery_overrides_v1/overrides.json"
)
TARGETS = {"003_S_0908_I1249292", "003_S_4350_I1252856"}
ARCHIVE_D0_ROOT = Path(
    "/data/derivatives/hcp379_v2/"
    "corrected_archive_D0_mask_union_recovery4"
)
ARCHIVE_D0_ORIGINAL_ROOT = Path(
    "/data/derivatives/hcp379_v2/corrected_archive_D0_recovery4"
)
ARCHIVE_D0_COMPACTION_SUMMARY = (
    ARCHIVE_D0_ROOT / "manifests/mask_union_compaction_summary.json"
)
ARCHIVE_D0_TARGETS = {"114_S_5234_I1130066"}
EDDY_RECOVERY_ROOT = Path(
    "/data/derivatives/hcp379_v2/eddy_integrity_recovery_v1"
)
EDDY_PRETRACT_ROOTS = {
    "003_S_4373_I378923": Path(
        "/data/derivatives/hcp379_v2/"
        "eddy_integrity_pretract_recovery4"
    ),
    "003_S_6490_I1043781": Path(
        "/data/derivatives/hcp379_v2/"
        "eddy_integrity_pretract_recovery4"
    ),
    "003_S_6644_I1083048": Path(
        "/data/derivatives/hcp379_v2/"
        "eddy_integrity_pretract_recovery4"
    ),
    "006_S_4713_I1483612": Path(
        "/data/derivatives/hcp379_v2/"
        "eddy_integrity_pretract_recovery4"
    ),
    "005_S_0610_I906100": Path(
        "/data/derivatives/hcp379_v2/"
        "eddy_extension_005_pretract_recovery4"
    ),
    "005_S_6084_I915209": Path(
        "/data/derivatives/hcp379_v2/"
        "eddy_extension_005_S_6084_pretract_recovery4"
    ),
}
EDDY_ORIGINAL_ROOTS = {
    "003_S_4373_I378923": Path(
        "/data/derivatives/hcp379_v2/"
        "corrected_legacy_tensor_recovery4"
    ),
    "003_S_6490_I1043781": Path(
        "/data/derivatives/hcp379_v2/corrected_scaleup_recovery4"
    ),
    "003_S_6644_I1083048": Path(
        "/data/derivatives/hcp379_v2/corrected_scaleup_recovery4"
    ),
    "005_S_0610_I906100": Path(
        "/data/derivatives/hcp379_v2/corrected_scaleup_recovery4"
    ),
    "005_S_6084_I915209": Path(
        "/data/derivatives/hcp379_v2/corrected_scaleup_recovery4"
    ),
    "006_S_4713_I1483612": Path(
        "/data/derivatives/hcp379_v2/corrected_scaleup_recovery4"
    ),
}
REQUIRED_RETAINED = {
    "five_tt",
    "wmfod_norm",
    "hcp_nodes",
    "hcp_volumes",
    "fa",
    "md",
    "rd",
    "ad",
    "atlas_qc",
    "automated_qc",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def verify(value: Any) -> Path:
    if not isinstance(value, Mapping):
        raise TypeError("file record required")
    path = Path(str(value.get("path", ""))).resolve()
    if record(path) != dict(value):
        raise ValueError(f"file record differs: {path}")
    return path


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


def add_completed_eddy_overrides(
    overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for unit in sorted(EDDY_PRETRACT_ROOTS):
        root = EDDY_PRETRACT_ROOTS[unit]
        state_path = root / "qc/subjects" / f"{unit}.json"
        compaction_path = (
            root / "qc/pretract_compaction" / f"{unit}.json"
        )
        pointer_path = (
            EDDY_RECOVERY_ROOT / "subjects" / unit / "result.json"
        )
        present = [
            state_path.is_file(),
            compaction_path.is_file(),
            pointer_path.is_file(),
        ]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f"{unit}: partial Eddy recovery override")
        original_state_path = (
            EDDY_ORIGINAL_ROOTS[unit]
            / "qc/subjects"
            / f"{unit}.json"
        )
        state = load_json(state_path)
        compaction = load_json(compaction_path)
        pointer = load_json(pointer_path)
        original = load_json(original_state_path)
        attempt_state_path = verify(pointer.get("attempt_state"))
        eddy_state = load_json(attempt_state_path)
        retained = compaction.get("retained_artifacts")
        eddy_output = eddy_state.get("eddy_output")
        source_eddy = state.get("source_identity", {}).get("eddy_source")
        if (
            pointer.get("unit") != unit
            or pointer.get("status") != "PASS_EDDY_INTEGRITY_RECOVERY"
            or eddy_state.get("unit") != unit
            or eddy_state.get("status")
            != "PASS_EDDY_INTEGRITY_RECOVERY"
            or eddy_state.get("eddy_volume_integrity", {}).get("gate")
            != "PASS"
            or eddy_state.get("eddy_volume_integrity", {}).get(
                "zero_volume_n"
            )
            != 0
            or eddy_state.get("tensor_audit", {}).get("gate") != "PASS"
            or eddy_state.get("historical_eddy_unchanged") is not True
            or not isinstance(eddy_output, Mapping)
            or not isinstance(source_eddy, Mapping)
            or dict(source_eddy) != dict(eddy_output)
            or state.get("record_type")
            != "diagnosis_blind_hcp379_scaleup_pretract_recovery4"
            or state.get("status") != "PASS_PRETRACT_RECOVERY4"
            or state.get("diagnosis_labels_used") is not False
            or state.get("unit") != unit
            or compaction.get("record_type")
            != "diagnosis_blind_hcp379_pretract_compaction"
            or compaction.get("status") != "PASS_COMPACTED"
            or compaction.get("diagnosis_labels_used") is not False
            or compaction.get("unit") != unit
            or Path(str(compaction.get("root", ""))).resolve()
            != root.resolve()
            or compaction.get("source_pretract_state")
            != record(state_path)
            or not isinstance(retained, Mapping)
            or not REQUIRED_RETAINED.issubset(retained)
            or original.get("status") != "FAIL_PRETRACT_RECOVERY4"
            or original.get("unit") != unit
        ):
            raise ValueError(f"{unit}: Eddy recovery override differs")
        verify(eddy_output)
        for name in REQUIRED_RETAINED:
            verify(retained[name])
        overrides[unit] = {
            "unit": unit,
            "recovery_class": "EDDY_INTEGRITY_RECOVERY_V1",
            "root": str(root.resolve()),
            "state": record(state_path),
            "compaction": record(compaction_path),
            "original_failed_state": record(original_state_path),
            "eddy_recovery_pointer": record(pointer_path),
            "eddy_recovery_attempt_state": record(attempt_state_path),
            "registration_route": state.get("registration_route"),
            "selected_5tt_dwi_mask_dice": state.get(
                "selected_5tt_dwi_mask_dice"
            ),
            "hcp379_atlas_inside_dwi_mask_fraction": state.get(
                "hcp379_atlas_inside_dwi_mask_fraction"
            ),
        }
        records.append(
            {
                "unit": unit,
                "eddy_recovery_pointer": record(pointer_path),
                "eddy_recovery_attempt_state": record(
                    attempt_state_path
                ),
            }
        )
    return records


def build() -> dict[str, Any]:
    summary = load_json(COMPACTION_SUMMARY)
    if (
        summary.get("record_type")
        != "hcp379_mask_union_pretract_recovery_compaction_summary"
        or summary.get("status") != "PASS_COMPACTED"
        or summary.get("non_overwriting") is not True
        or summary.get("target_n") != len(TARGETS)
        or set(summary.get("target_units", [])) != TARGETS
    ):
        raise ValueError("mask-union compaction summary differs")
    overrides: dict[str, Any] = {}
    for unit in sorted(TARGETS):
        state_path = ROOT / "qc/subjects" / f"{unit}.json"
        compaction_path = (
            ROOT / "qc/pretract_compaction" / f"{unit}.json"
        )
        original_state_path = (
            ORIGINAL_ROOT / "qc/subjects" / f"{unit}.json"
        )
        state = load_json(state_path)
        compaction = load_json(compaction_path)
        original = load_json(original_state_path)
        retained = compaction.get("retained_artifacts")
        if (
            state.get("record_type")
            != "diagnosis_blind_hcp379_scaleup_pretract_recovery4"
            or state.get("status") != "PASS_PRETRACT_RECOVERY4"
            or state.get("diagnosis_labels_used") is not False
            or state.get("unit") != unit
            or state.get("mask_recovery_policy")
            != "dwi2mask_uncleaned_dilate1_union_original_v1"
            or compaction.get("record_type")
            != "diagnosis_blind_hcp379_pretract_compaction"
            or compaction.get("status") != "PASS_COMPACTED"
            or compaction.get("diagnosis_labels_used") is not False
            or compaction.get("unit") != unit
            or Path(str(compaction.get("root", ""))).resolve()
            != ROOT.resolve()
            or compaction.get("source_pretract_state")
            != record(state_path)
            or not isinstance(retained, Mapping)
            or not REQUIRED_RETAINED.issubset(retained)
            or original.get("status") != "FAIL_PRETRACT_RECOVERY4"
            or original.get("unit") != unit
        ):
            raise ValueError(f"{unit}: isolated recovery override differs")
        for name in REQUIRED_RETAINED:
            verify(retained[name])
        overrides[unit] = {
            "unit": unit,
            "recovery_class": "MASK_UNION_RECOVERY_V1",
            "root": str(ROOT.resolve()),
            "state": record(state_path),
            "compaction": record(compaction_path),
            "original_failed_state": record(original_state_path),
            "registration_route": state.get("registration_route"),
            "selected_5tt_dwi_mask_dice": state.get(
                "selected_5tt_dwi_mask_dice"
            ),
            "hcp379_atlas_inside_dwi_mask_fraction": state.get(
                "hcp379_atlas_inside_dwi_mask_fraction"
            ),
        }
    archive_summary = load_json(ARCHIVE_D0_COMPACTION_SUMMARY)
    if (
        archive_summary.get("record_type")
        != (
            "hcp379_mask_union_archive_D0_"
            "pretract_recovery_compaction_summary"
        )
        or archive_summary.get("status") != "PASS_COMPACTED"
        or archive_summary.get("non_overwriting") is not True
        or archive_summary.get("target_n") != len(ARCHIVE_D0_TARGETS)
        or set(archive_summary.get("target_units", []))
        != ARCHIVE_D0_TARGETS
    ):
        raise ValueError(
            "archive-D0 mask-union compaction summary differs"
        )
    for unit in sorted(ARCHIVE_D0_TARGETS):
        state_path = ARCHIVE_D0_ROOT / "qc/subjects" / f"{unit}.json"
        compaction_path = (
            ARCHIVE_D0_ROOT
            / "qc/pretract_compaction"
            / f"{unit}.json"
        )
        original_state_path = (
            ARCHIVE_D0_ORIGINAL_ROOT / "qc/subjects" / f"{unit}.json"
        )
        state = load_json(state_path)
        compaction = load_json(compaction_path)
        original = load_json(original_state_path)
        retained = compaction.get("retained_artifacts")
        if (
            state.get("record_type")
            != "diagnosis_blind_hcp379_scaleup_pretract_recovery4"
            or state.get("status") != "PASS_PRETRACT_RECOVERY4"
            or state.get("diagnosis_labels_used") is not False
            or state.get("unit") != unit
            or state.get("mask_recovery_policy")
            != "dwi2mask_uncleaned_dilate1_union_original_v1"
            or compaction.get("record_type")
            != "diagnosis_blind_hcp379_pretract_compaction"
            or compaction.get("status") != "PASS_COMPACTED"
            or compaction.get("diagnosis_labels_used") is not False
            or compaction.get("unit") != unit
            or Path(str(compaction.get("root", ""))).resolve()
            != ARCHIVE_D0_ROOT.resolve()
            or compaction.get("source_pretract_state")
            != record(state_path)
            or not isinstance(retained, Mapping)
            or not REQUIRED_RETAINED.issubset(retained)
            or original.get("status") != "FAIL_PRETRACT_RECOVERY4"
            or original.get("unit") != unit
        ):
            raise ValueError(
                f"{unit}: archive-D0 isolated recovery override differs"
            )
        for name in REQUIRED_RETAINED:
            verify(retained[name])
        overrides[unit] = {
            "unit": unit,
            "recovery_class": "MASK_UNION_RECOVERY_V1",
            "root": str(ARCHIVE_D0_ROOT.resolve()),
            "state": record(state_path),
            "compaction": record(compaction_path),
            "original_failed_state": record(original_state_path),
            "registration_route": state.get("registration_route"),
            "selected_5tt_dwi_mask_dice": state.get(
                "selected_5tt_dwi_mask_dice"
            ),
            "hcp379_atlas_inside_dwi_mask_fraction": state.get(
                "hcp379_atlas_inside_dwi_mask_fraction"
            ),
        }
    completed_eddy_overrides = add_completed_eddy_overrides(overrides)
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_pretract_recovery_overrides",
        "status": "PASS",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "override_n": len(overrides),
        "completed_eddy_override_n": len(completed_eddy_overrides),
        "completed_eddy_overrides": completed_eddy_overrides,
        "overrides": overrides,
        "records": {
            "compaction_summary": record(COMPACTION_SUMMARY),
            "archive_D0_compaction_summary": record(
                ARCHIVE_D0_COMPACTION_SUMMARY
            ),
            "builder": record(Path(__file__)),
        },
    }
    atomic_json(OUTPUT, payload)
    return payload


def main() -> int:
    value = build()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
