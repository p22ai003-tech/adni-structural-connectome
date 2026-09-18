#!/usr/bin/env python3
"""Validate Recovery3 corrections from the bounded Recovery2 calibration data."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np


ROOT = Path("/home/ec2-user/exp")
RUN_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4")
COMPLETION = RUN_ROOT / "publication/pre_tractography_canary_recovery2_completion.json"
FIVE_TT_SCRATCH = Path("/tmp/scforge-5tt-all15-ZaP3Cd")
REGISTRATION_SCRATCH = Path("/tmp/scforge-mni-5ttmask-test-b78Ctq")
OUTPUT_DIR = ROOT / "research_audit/outputs/h04a_r1_retry4_pretract_recovery3_diagnostics_v1"
OUTPUT = OUTPUT_DIR / "recovery3_diagnostics.json"
FIVE_TT_CHECK = Path("/home/ec2-user/mrtrix3/bin/5ttcheck")
REPRESENTATIVE_UNITS = (
    "003_S_4118_I1124861",
    "009_S_4324_I1186579",
    "014_S_6087_I926924",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular evidence file: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def image_metrics(atlas_path: Path, five_tt_mask_path: Path) -> dict[str, Any]:
    atlas_image = nib.load(str(atlas_path))
    five_tt_image = nib.load(str(five_tt_mask_path))
    atlas = np.rint(atlas_image.get_fdata(dtype=np.float32)).astype(np.int16)
    five_tt = five_tt_image.get_fdata(dtype=np.float32) > 0
    atlas_mask = atlas > 0
    labels = sorted(int(value) for value in np.unique(atlas) if value > 0)
    intersection = int(np.logical_and(atlas_mask, five_tt).sum())
    denominator = int(atlas_mask.sum() + five_tt.sum())
    atlas_world = nib.affines.apply_affine(
        atlas_image.affine, np.argwhere(atlas_mask).mean(axis=0)
    )
    five_tt_world = nib.affines.apply_affine(
        five_tt_image.affine, np.argwhere(five_tt).mean(axis=0)
    )
    return {
        "labels_found": len(labels),
        "labels_are_exact_1_to_166": labels == list(range(1, 167)),
        "atlas_five_tt_mask_dice": 2.0 * intersection / denominator,
        "atlas_inside_five_tt_fraction": intersection / int(atlas_mask.sum()),
        "centroid_delta_mm": float(np.linalg.norm(atlas_world - five_tt_world)),
    }


def main() -> int:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite diagnostics: {OUTPUT_DIR}")
    completion = load_json(COMPLETION)
    if (
        completion.get("record_type")
        != "pre_tractography_canary_recovery2_completion"
        or completion.get("status") != "FAIL"
        or completion.get("tractography_started") is not False
        or completion.get("matrix_generation_started") is not False
    ):
        raise ValueError("Recovery2 terminal state differs")
    units = completion.get("execution_binding", {}).get("units")
    if not isinstance(units, list) or len(units) != 15:
        raise ValueError("Recovery2 unit binding differs")
    end_record = completion.get("terminal_attempt_end")
    if not isinstance(end_record, dict):
        raise TypeError("Recovery2 completion lacks terminal attempt")
    end_path = Path(str(end_record["path"])).resolve()
    if file_record(end_path) != end_record:
        raise ValueError("Recovery2 terminal attempt drifted")
    end = load_json(end_path)
    log_record = end.get("combined_execution_log")
    if not isinstance(log_record, dict):
        raise TypeError("Recovery2 attempt lacks execution log")
    log_path = Path(str(log_record["path"])).resolve()
    if file_record(log_path) != log_record:
        raise ValueError("Recovery2 execution log drifted")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    rule_errors = {
        rule: log_text.count(f"Error in rule {rule}:")
        for rule in ("atlas_contract_qc", "five_tt_dwi")
    }
    if rule_errors != {"atlas_contract_qc": 30, "five_tt_dwi": 30}:
        raise ValueError(f"Recovery2 failure signature differs: {rule_errors}")
    if len(re.findall(r"argument should be a str or an os.PathLike object", log_text)) != 15:
        raise ValueError("Recovery2 atlas Namedlist failure count differs")
    # Snakemake repeats each failed 5ttcheck message once in the final summary.
    if log_text.count("Input image does not conform to 5TT format") != 30:
        raise ValueError("Recovery2 5TT validation failure count differs")

    five_tt_records: dict[str, Any] = {}
    old_atlas_metrics: dict[str, Any] = {}
    for unit in units:
        clamped = FIVE_TT_SCRATCH / unit / "clamped.mif"
        check = subprocess.run(
            [str(FIVE_TT_CHECK), str(clamped)],
            check=False,
            capture_output=True,
            text=True,
        )
        five_tt_records[unit] = {
            "status": "PASS" if check.returncode == 0 else "FAIL",
            "returncode": check.returncode,
            "output": file_record(clamped),
        }
        old_atlas_metrics[unit] = image_metrics(
            RUN_ROOT
            / "subjects"
            / unit
            / "04_atlas/aal3_nodes_166_dwi_1mm.nii.gz",
            FIVE_TT_SCRATCH / unit / "5tt_mask_1mm.nii.gz",
        )

    representative: dict[str, Any] = {}
    for unit in REPRESENTATIVE_UNITS:
        old = old_atlas_metrics[unit]
        new_path = REGISTRATION_SCRATCH / unit / "atlas_dwi.nii.gz"
        new = image_metrics(
            new_path, FIVE_TT_SCRATCH / unit / "5tt_mask_1mm.nii.gz"
        )
        representative[unit] = {
            "old": old,
            "recovery3_candidate": new,
            "candidate_atlas": file_record(new_path),
            "dice_change": new["atlas_five_tt_mask_dice"]
            - old["atlas_five_tt_mask_dice"],
            "centroid_delta_change_mm": new["centroid_delta_mm"]
            - old["centroid_delta_mm"],
        }

    checks = {
        "recovery2_terminal_fail_bound": True,
        "tractography_never_started": True,
        "atlas_namedlist_failure_exact_15": True,
        "five_tt_default_interpolation_failure_exact_15": True,
        "linear_clipped_five_tt_pass_15_of_15": all(
            record["status"] == "PASS" for record in five_tt_records.values()
        ),
        "representative_atlases_retain_166_labels": all(
            record["recovery3_candidate"]["labels_are_exact_1_to_166"]
            for record in representative.values()
        ),
        "representative_centroid_error_at_most_5mm": all(
            record["recovery3_candidate"]["centroid_delta_mm"] <= 5.0
            for record in representative.values()
        ),
        "failed_registration_cases_improve_dice_by_at_least_0_30": all(
            representative[unit]["dice_change"] >= 0.30
            for unit in ("009_S_4324_I1186579", "014_S_6087_I926924")
        ),
        "diagnosis_labels_not_used": True,
    }
    payload = {
        "schema_version": "1.0.0",
        "record_type": "retry4_pretract_recovery3_candidate_diagnostics",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(RUN_ROOT),
        "unit_count": len(units),
        "units": units,
        "checks": checks,
        "recovery2_completion": file_record(COMPLETION),
        "recovery2_attempt_end": file_record(end_path),
        "recovery2_execution_log": file_record(log_path),
        "recovery2_rule_error_counts": rule_errors,
        "five_tt_linear_clip_validation": five_tt_records,
        "recovery2_atlas_metrics": old_atlas_metrics,
        "representative_registration_validation": representative,
        "diagnosis_labels_used": False,
        "tractography_executed": False,
        "matrix_generation_executed": False,
        "full_cohort_executed": False,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    write_immutable_json(OUTPUT, payload)
    print(json.dumps({"status": payload["status"], "output": file_record(OUTPUT)}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
