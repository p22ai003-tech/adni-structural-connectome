#!/usr/bin/env python3
"""Independently validate the isolated 114 HROI recovery adapter."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP
    / "scripts/hcp/"
    "build_hcp379_hroi_surface_support_114_recovery_v1.py"
)
BASE_SOURCE = (
    EXP / "scripts/hcp/build_hcp379_hroi_surface_support_v1.py"
)
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "hroi_surface_support_114_recovery_v1/attempts"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_hroi_surface_support_114_recovery_v1/validation.json"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")
UNIT = "114_S_6347_I1344943"
EXPECTED_SUBJECT = Path(
    "/data/derivatives/hcp379_v2/fastsurfer_repair/"
    "standard_freesurfer/114_S_6347_I1344943_FS"
)
LUT = Path(
    "/data/derivatives/parc_hcpmmp1/hcpmmp1_subcort_relabel.txt"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def record_matches(record: Any, path: Path) -> bool:
    return isinstance(record, dict) and record == file_record(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def integer_image(
    path: Path,
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    integer = np.asarray(data, dtype=np.int32)
    if data.ndim != 3 or not np.array_equal(data, integer):
        raise ValueError(path)
    return image, integer


def main() -> int:
    checks: dict[str, Any] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    compile_probe = subprocess.run(
        [str(PYTHON), "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "source_compiles",
        compile_probe.returncode == 0,
        {
            "returncode": compile_probe.returncode,
            "stderr": compile_probe.stderr[-2000:],
        },
    )
    dry_probe = subprocess.run(
        [str(PYTHON), str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        dry = json.loads(dry_probe.stdout)
    except Exception:
        dry = {}
    check(
        "dry_run_binds_exact_recovery_route",
        (
            dry_probe.returncode == 0
            and dry.get("status") == "READY_114_HROI_RECOVERY"
            and dry.get("unit") == UNIT
            and dry.get("diagnosis_labels_used") is False
            and dry.get("source_files_modified") is False
            and dry.get("missing_inputs") == []
            and dry.get("cohort_recipe_selected") is False
            and dry.get("selected_count_tractography_authorized")
            is False
            and dry.get("final_release_authorized") is False
        ),
        dry or {"stderr": dry_probe.stderr[-2000:]},
    )
    text = SOURCE.read_text(encoding="utf-8")
    required = (
        "standard_freesurfer/114_S_6347_I1344943_FS",
        "subjects_dir_adapter",
        "os.symlink",
        "BASE.SUBJECTS_DIR",
        "PASS_ALL_379_SOURCE_LABELS",
        "subcortical_voxels_overwritten",
        "other_cortical_labels_overwritten",
        "cohort_recipe_selected",
        "selected_count_tractography_authorized",
        "final_release_authorized",
    )
    missing = [fragment for fragment in required if fragment not in text]
    check(
        "implementation_contains_bounded_adapter_contract",
        not missing,
        {"missing": missing},
    )
    prohibited = (
        "rm -rf",
        "shutil.rmtree",
        ".unlink(",
        "diagnosis.csv",
        "DX_bl",
        "cohort_recipe_selected\": True",
        "selected_count_tractography_authorized\": True",
        "final_release_authorized\": True",
    )
    found = [fragment for fragment in prohibited if fragment in text]
    check(
        "implementation_has_no_destructive_or_outcome_shortcut",
        not found,
        {"found": found},
    )

    summaries = sorted(ATTEMPTS.glob("*/summary.json"))
    if summaries:
        summary_path = summaries[-1]
        try:
            summary = load_json(summary_path)
            manifest_path = Path(
                str(summary.get("candidate_manifest", {}).get("path", ""))
            )
            manifest = load_json(manifest_path)
            candidate_path = Path(
                str(manifest.get("candidate", {}).get("path", ""))
            )
            original_path = Path(
                str(manifest.get("original_source", {}).get("path", ""))
            )
            aseg_path = Path(
                str(manifest.get("aseg", {}).get("path", ""))
            )
            preflight_path = summary_path.parent / "preflight.json"
            records_ok = (
                record_matches(
                    summary.get("candidate_manifest"),
                    manifest_path,
                )
                and record_matches(
                    summary.get("candidate"),
                    candidate_path,
                )
                and record_matches(
                    summary.get("base_builder"),
                    BASE_SOURCE,
                )
                and record_matches(
                    summary.get("implementation"),
                    SOURCE,
                )
                and record_matches(
                    summary.get("preflight"),
                    preflight_path,
                )
                and record_matches(
                    manifest.get("candidate"),
                    candidate_path,
                )
                and record_matches(
                    manifest.get("original_source"),
                    original_path,
                )
                and record_matches(
                    manifest.get("aseg"),
                    aseg_path,
                )
                and record_matches(
                    manifest.get("implementation"),
                    BASE_SOURCE,
                )
            )
            summary_contract = (
                summary.get("status")
                == "PASS_114_HROI_RECOVERY_FOR_COHORT_MERGE"
                and summary.get("unit") == UNIT
                and summary.get("diagnosis_labels_used") is False
                and summary.get("source_files_modified") is False
                and summary.get("failed_standard_candidate_modified")
                is False
                and summary.get("cohort_recipe_selected") is False
                and summary.get(
                    "selected_count_tractography_authorized"
                )
                is False
                and summary.get("final_release_authorized") is False
                and manifest.get("status")
                == "PASS_ALL_379_SOURCE_LABELS"
                and manifest.get("unit") == UNIT
                and manifest.get("diagnosis_labels_used") is False
                and manifest.get("source_files_modified") is False
                and manifest.get("present_source_label_n") == 379
                and manifest.get("missing_source_labels") == []
                and manifest.get("policy", {}).get(
                    "subcortical_voxels_overwritten"
                )
                == 0
                and manifest.get("policy", {}).get(
                    "other_cortical_labels_overwritten"
                )
                == 0
            )
            adapter = Path(
                str(summary.get("subject_adapter", {}).get("path", ""))
            )
            adapter_ok = (
                adapter.is_symlink()
                and adapter.resolve() == EXPECTED_SUBJECT.resolve()
                and summary.get("subject_adapter", {}).get("target")
                == str(EXPECTED_SUBJECT.resolve())
            )
            check(
                "runtime_summary_records_and_adapter_replay",
                summary_contract and records_ok and adapter_ok,
                {
                    "summary": str(summary_path),
                    "summary_contract": summary_contract,
                    "artifact_records_match": records_ok,
                    "adapter_resolves_exact_recovery": adapter_ok,
                },
            )

            original_image, original = integer_image(original_path)
            candidate_image, candidate = integer_image(candidate_path)
            aseg_image, aseg = integer_image(aseg_path)
            geometry_ok = (
                original_image.shape == candidate_image.shape
                == aseg_image.shape
                and np.allclose(
                    original_image.affine,
                    candidate_image.affine,
                    atol=1.0e-5,
                    rtol=0.0,
                )
                and np.allclose(
                    original_image.affine,
                    aseg_image.affine,
                    atol=1.0e-5,
                    rtol=0.0,
                )
            )
            changed = candidate != original
            allowed_left = (
                changed
                & (original == 2)
                & (aseg == 2)
                & (candidate == 1120)
            )
            allowed_right = (
                changed
                & (original == 41)
                & (aseg == 41)
                & (candidate == 2120)
            )
            changes_exact = np.array_equal(
                changed,
                allowed_left | allowed_right,
            )
            expected_labels = {
                int(line.split()[0])
                for line in LUT.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            present = {
                int(value) for value in np.unique(candidate)
            }
            labels_ok = (
                len(expected_labels) == 379
                and expected_labels.issubset(present)
            )
            accounting_ok = (
                int(changed.sum())
                == int(manifest.get("changed_voxel_n", -1))
                == (
                    int(
                        manifest.get("hemispheres", {})
                        .get("lh", {})
                        .get(
                            "white_surface_support_added_voxels",
                            -1,
                        )
                    )
                    + int(
                        manifest.get("hemispheres", {})
                        .get("rh", {})
                        .get(
                            "white_surface_support_added_voxels",
                            -1,
                        )
                    )
                )
            )
            check(
                "runtime_image_policy_independent_replay",
                (
                    geometry_ok
                    and changes_exact
                    and labels_ok
                    and accounting_ok
                    and int(allowed_left.sum()) > 0
                    and int(allowed_right.sum()) > 0
                ),
                {
                    "geometry_exact": geometry_ok,
                    "changes_only_white_matter_to_hroi": (
                        changes_exact
                    ),
                    "all_379_source_labels_present": labels_ok,
                    "changed_voxel_accounting_exact": accounting_ok,
                    "left_changed_voxels": int(allowed_left.sum()),
                    "right_changed_voxels": int(
                        allowed_right.sum()
                    ),
                },
            )
        except Exception as exc:
            check(
                "runtime_execution_replay",
                False,
                {
                    "summary": str(summary_path),
                    "exception": repr(exc),
                },
            )

    passed = sum(row["status"] == "PASS" for row in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_hroi_surface_support_114_recovery_v1_validation"
        ),
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "runtime_summary_detected": bool(summaries),
        "imaging_executed_by_validation": False,
        "source": file_record(SOURCE),
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
