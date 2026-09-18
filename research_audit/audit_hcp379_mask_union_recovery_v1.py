#!/home/ec2-user/fsl/bin/python
"""Diagnosis-blind control audit for a fixed DWI-mask recovery.

The proposed recovery is deliberately simple and cohort-uniform: construct a
DWI-derived mask with MRtrix peninsula cleaning disabled, dilate it by one
native voxel, and union it with the original mask.  It is evaluated on the two
remaining registration-only failures and all 15 reviewed HCP379 canaries.
This audit does not change a production subject, registration transform, QC
threshold, tractogram, matrix, diagnosis, or outcome.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np


HCP = Path("/data/derivatives/hcp379_v2")
SCFORGE = Path(
    "/data/derivatives/scforge_v2/"
    "h04a_r1_recovery_20260719_retry4"
)
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
ROOT = HCP / "corrected_legacy_tensor_recovery4"
PROMOTED = (
    HCP
    / "pretract_promoted_recovery4_v6/"
    "pretract_promoted_recovery4_v6_manifest.json"
)
LEDGER = (
    Path("/home/ec2-user/exp/research_audit/outputs/")
    / "hcp379_pretract_failure_recovery_ledger_v1/ledger.json"
)
OUTPUT = (
    HCP
    / "pretract_failure_source_audit_v1/"
    "mask_union_recovery_controls_v1"
)
SUMMARY = OUTPUT / "summary.json"
ROWS = OUTPUT / "rows.csv"
TARGET_ROUTES = {
    "003_S_0908_I1249292": "seeded_ants_rigid_failover",
    "003_S_4350_I1252856": "seeded_ants_syn_recovery",
}
MINIMUM_HEMISPHERE_ATLAS_INSIDE = 0.80
MINIMUM_TISSUE_DICE = 0.70
MAXIMUM_MASK_VOLUME_RATIO = 1.15
MINIMUM_SHELL_TO_POSITIVE_B0_MEDIAN_RATIO = 2.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"regular file required: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


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


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run(command: list[str], outputs: tuple[Path, ...]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"command failed rc={result.returncode}: {' '.join(command)}\n"
            f"{result.stderr[-4000:]}"
        )
    absent = [
        str(path)
        for path in outputs
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if absent:
        raise RuntimeError("command outputs absent: " + ",".join(absent))


def require_path(value: Any) -> Path:
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
        raise ValueError("artifact record differs")
    path = Path(str(value["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def canary_records() -> list[dict[str, Any]]:
    promoted = load_json(PROMOTED)
    values = promoted.get("units")
    if not isinstance(values, list) or len(values) != 15:
        raise ValueError("promoted canary manifest differs")
    records: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("promoted canary row differs")
        unit = str(value["unit"])
        manifest = load_json(require_path(value["manifest"]))
        records.append(
            {
                "unit": unit,
                "role": "reviewed_canary_control",
                "route": str(manifest["registration"]["route"]),
                "dwi": require_path(
                    manifest["tractography_inputs"]["dwi_bias_corrected"]
                ),
                "mask": require_path(
                    manifest["tractography_inputs"]["dwi_mask"]
                ),
                "b0": require_path(
                    manifest["tractography_inputs"]["mean_b0"]
                ),
                "five_tt": require_path(
                    manifest["tractography_inputs"]["five_tt"]
                ),
                "atlas": require_path(
                    manifest["hcp379"]["nodes_b0_1mm"]
                ),
            }
        )
    return sorted(records, key=lambda row: str(row["unit"]))


def target_records() -> list[dict[str, Any]]:
    ledger = load_json(LEDGER)
    failed = ledger.get("failed_units")
    if not isinstance(failed, list):
        raise ValueError("failure ledger differs")
    observed = {
        str(row["unit"])
        for row in failed
        if isinstance(row, Mapping)
        and row.get("failure_class") == "SPATIAL_BBR_AND_RIGID_FAIL"
    }
    if observed != set(TARGET_ROUTES):
        raise ValueError(f"exact spatial target set differs: {observed}")
    route_paths = {
        "seeded_ants_rigid_failover": "recovery4_ants",
        "seeded_ants_syn_recovery": "recovery4_ants_syn",
    }
    records: list[dict[str, Any]] = []
    for unit, route in sorted(TARGET_ROUTES.items()):
        subject = ROOT / "subjects" / unit
        route_dir = subject / "03_spatial" / route_paths[route]
        records.append(
            {
                "unit": unit,
                "role": "registration_failure_target",
                "route": route,
                "dwi": subject / "01_dwi/dwi_biascorr.mif",
                "mask": subject / "01_dwi/dwi_brain_mask.mif",
                "b0": subject / "01_dwi/mean_b0.nii.gz",
                "five_tt": route_dir / "5tt_dwi_recovery4.mif",
                "atlas": route_dir / "hcp379_nodes_b0_1mm.nii.gz",
            }
        )
    for record in records:
        for key in ("dwi", "mask", "b0", "five_tt", "atlas"):
            path = Path(record[key])
            if not path.is_file():
                raise FileNotFoundError(path)
    return records


def artifacts_for(record: Mapping[str, Any]) -> dict[str, Path]:
    unit = str(record["unit"])
    output = OUTPUT / "subjects" / unit
    return {
        "root": output,
        "uncleaned": output / "mask_uncleaned.mif",
        "uncleaned_dilate1": output / "mask_uncleaned_dilate1.mif",
        "union": output / "mask_union.mif",
        "original_native": output / "mask_original_native.nii.gz",
        "union_native": output / "mask_union_native.nii.gz",
        "original_1mm": output / "mask_original_1mm.nii.gz",
        "union_1mm": output / "mask_union_1mm.nii.gz",
        "five_tt_sum": output / "five_tt_sum.mif",
        "five_tt_mask_mif": output / "five_tt_mask.mif",
        "five_tt_mask": output / "five_tt_mask_canonical.nii.gz",
    }


def generate(record: Mapping[str, Any]) -> dict[str, Path]:
    paths = artifacts_for(record)
    paths["root"].mkdir(parents=True, exist_ok=True)
    dwi = Path(record["dwi"])
    original = Path(record["mask"])
    atlas = Path(record["atlas"])
    five_tt = Path(record["five_tt"])
    commands = [
        (
            [
                str(MRTRIX / "dwi2mask"),
                str(dwi),
                str(paths["uncleaned"]),
                "-clean_scale",
                "0",
                "-nthreads",
                "8",
            ],
            (paths["uncleaned"],),
        ),
        (
            [
                str(MRTRIX / "maskfilter"),
                str(paths["uncleaned"]),
                "dilate",
                str(paths["uncleaned_dilate1"]),
                "-npass",
                "1",
                "-nthreads",
                "8",
            ],
            (paths["uncleaned_dilate1"],),
        ),
        (
            [
                str(MRTRIX / "mrcalc"),
                str(original),
                str(paths["uncleaned_dilate1"]),
                "-max",
                str(paths["union"]),
            ],
            (paths["union"],),
        ),
        (
            [
                str(MRTRIX / "mrconvert"),
                str(original),
                str(paths["original_native"]),
                "-strides",
                "-1,2,3",
            ],
            (paths["original_native"],),
        ),
        (
            [
                str(MRTRIX / "mrconvert"),
                str(paths["union"]),
                str(paths["union_native"]),
                "-strides",
                "-1,2,3",
            ],
            (paths["union_native"],),
        ),
        (
            [
                str(MRTRIX / "mrtransform"),
                str(original),
                str(paths["original_1mm"]),
                "-template",
                str(atlas),
                "-interp",
                "nearest",
                "-nthreads",
                "8",
            ],
            (paths["original_1mm"],),
        ),
        (
            [
                str(MRTRIX / "mrtransform"),
                str(paths["union"]),
                str(paths["union_1mm"]),
                "-template",
                str(atlas),
                "-interp",
                "nearest",
                "-nthreads",
                "8",
            ],
            (paths["union_1mm"],),
        ),
        (
            [
                str(MRTRIX / "mrmath"),
                str(five_tt),
                "sum",
                str(paths["five_tt_sum"]),
                "-axis",
                "3",
                "-nthreads",
                "8",
            ],
            (paths["five_tt_sum"],),
        ),
        (
            [
                str(MRTRIX / "mrthreshold"),
                str(paths["five_tt_sum"]),
                str(paths["five_tt_mask_mif"]),
                "-abs",
                "0.5",
            ],
            (paths["five_tt_mask_mif"],),
        ),
        (
            [
                str(MRTRIX / "mrconvert"),
                str(paths["five_tt_mask_mif"]),
                str(paths["five_tt_mask"]),
                "-strides",
                "-1,2,3",
            ],
            (paths["five_tt_mask"],),
        ),
    ]
    for command, outputs in commands:
        if all(path.is_file() and path.stat().st_size > 0 for path in outputs):
            continue
        if any(path.exists() for path in outputs):
            raise ValueError(
                "refusing partial diagnostic overwrite: "
                + ",".join(str(path) for path in outputs)
            )
        run(command, outputs)
    return paths


def image_bool(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj) > 0


def coverage(atlas: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    groups = {
        "all": atlas > 0,
        "left_cortex": np.isin(atlas, np.arange(1, 181)),
        "right_cortex": np.isin(atlas, np.arange(181, 361)),
    }
    result: dict[str, float] = {}
    for name, group in groups.items():
        denominator = int(group.sum())
        if not denominator:
            raise ValueError(f"atlas group is empty: {name}")
        result[name] = float(np.logical_and(group, mask).sum() / denominator)
    return result


def evaluate(record: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    original_native = image_bool(paths["original_native"])
    union_native = image_bool(paths["union_native"])
    original_1mm = image_bool(paths["original_1mm"])
    union_1mm = image_bool(paths["union_1mm"])
    tissue = image_bool(paths["five_tt_mask"])
    b0 = np.asanyarray(
        nib.load(str(record["b0"])).dataobj, dtype=np.float64
    )
    atlas = np.rint(
        np.asanyarray(nib.load(str(record["atlas"])).dataobj)
    ).astype(np.int32)
    if (
        original_native.shape != union_native.shape
        or original_native.shape != tissue.shape
        or original_native.shape != b0.shape
        or original_1mm.shape != union_1mm.shape
        or original_1mm.shape != atlas.shape
    ):
        raise ValueError(f"diagnostic geometry differs: {record['unit']}")
    original_signal = b0[original_native]
    shell = np.logical_and(union_native, np.logical_not(original_native))
    shell_signal = b0[shell]
    if not shell_signal.size or not original_signal.size:
        raise ValueError(f"mask shell is empty: {record['unit']}")
    original_p05 = float(np.percentile(original_signal, 5))
    positive_b0 = b0[np.logical_and(np.isfinite(b0), b0 > 0)]
    if not positive_b0.size:
        raise ValueError(f"positive b0 support is empty: {record['unit']}")
    positive_b0_median = float(np.median(positive_b0))
    original_coverage = coverage(atlas, original_1mm)
    union_coverage = coverage(atlas, union_1mm)
    tissue_dice = float(
        2.0 * np.logical_and(tissue, union_native).sum()
        / (tissue.sum() + union_native.sum())
    )
    volume_ratio = float(union_native.sum() / original_native.sum())
    shell_fraction = float((shell_signal >= original_p05).mean())
    shell_to_positive_median_ratio = float(
        np.median(shell_signal) / positive_b0_median
    )
    target_pass = (
        union_coverage["left_cortex"]
        >= MINIMUM_HEMISPHERE_ATLAS_INSIDE
        and union_coverage["right_cortex"]
        >= MINIMUM_HEMISPHERE_ATLAS_INSIDE
        and tissue_dice >= MINIMUM_TISSUE_DICE
        and volume_ratio <= MAXIMUM_MASK_VOLUME_RATIO
        and shell_to_positive_median_ratio
        >= MINIMUM_SHELL_TO_POSITIVE_B0_MEDIAN_RATIO
    )
    return {
        "unit": str(record["unit"]),
        "role": str(record["role"]),
        "route": str(record["route"]),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "original_mask_voxels": int(original_native.sum()),
        "union_mask_voxels": int(union_native.sum()),
        "mask_volume_ratio": volume_ratio,
        "shell_voxels": int(shell.sum()),
        "whole_positive_b0_median": positive_b0_median,
        "original_b0_p05": original_p05,
        "shell_b0_p05": float(np.percentile(shell_signal, 5)),
        "shell_b0_median": float(np.median(shell_signal)),
        "shell_b0_p95": float(np.percentile(shell_signal, 95)),
        "shell_fraction_above_original_p05": shell_fraction,
        "shell_to_positive_b0_median_ratio": (
            shell_to_positive_median_ratio
        ),
        "original_all_atlas_inside": original_coverage["all"],
        "original_left_cortex_inside": original_coverage["left_cortex"],
        "original_right_cortex_inside": original_coverage["right_cortex"],
        "union_all_atlas_inside": union_coverage["all"],
        "union_left_cortex_inside": union_coverage["left_cortex"],
        "union_right_cortex_inside": union_coverage["right_cortex"],
        "union_tissue_dice": tissue_dice,
        "target_fixed_gate_pass": (
            target_pass
            if record["role"] == "registration_failure_target"
            else None
        ),
        "artifacts": {
            name: file_record(path)
            for name, path in paths.items()
            if name != "root"
        },
    }


def execute() -> dict[str, Any]:
    records = canary_records() + target_records()
    rows = [evaluate(record, generate(record)) for record in records]
    controls = [
        row for row in rows if row["role"] == "reviewed_canary_control"
    ]
    targets = [
        row for row in rows if row["role"] == "registration_failure_target"
    ]
    checks = {
        "exact_reviewed_control_and_target_sets": (
            len(controls) == 15
            and {row["unit"] for row in targets} == set(TARGET_ROUTES)
        ),
        "fixed_policy_is_diagnosis_outcome_density_blind": all(
            row["diagnosis_labels_used"] is False
            and row["outcomes_used"] is False
            and row["connectome_density_used"] is False
            for row in rows
        ),
        "mask_union_never_reduces_control_atlas_coverage": all(
            row["union_all_atlas_inside"] + 1.0e-12
            >= row["original_all_atlas_inside"]
            and row["union_left_cortex_inside"] + 1.0e-12
            >= row["original_left_cortex_inside"]
            and row["union_right_cortex_inside"] + 1.0e-12
            >= row["original_right_cortex_inside"]
            for row in controls
        ),
        "all_mask_expansions_are_bounded": all(
            1.0 <= row["mask_volume_ratio"] <= MAXIMUM_MASK_VOLUME_RATIO
            for row in rows
        ),
        "all_added_shells_have_supported_b0_signal": all(
            row["shell_to_positive_b0_median_ratio"]
            >= MINIMUM_SHELL_TO_POSITIVE_B0_MEDIAN_RATIO
            for row in rows
        ),
        "both_targets_pass_unchanged_atlas_and_tissue_gates": all(
            row["target_fixed_gate_pass"] is True for row in targets
        ),
    }
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_mask_union_recovery_control_audit",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "source_images_modified": False,
        "tractography_generated": False,
        "matrix_generated": False,
        "policy": {
            "original_mask_retained_by_union": True,
            "dwi2mask_clean_scale": 0,
            "native_voxel_dilation_passes": 1,
            "minimum_hemisphere_atlas_inside": (
                MINIMUM_HEMISPHERE_ATLAS_INSIDE
            ),
            "minimum_tissue_dice": MINIMUM_TISSUE_DICE,
            "maximum_mask_volume_ratio": MAXIMUM_MASK_VOLUME_RATIO,
            "minimum_shell_to_positive_b0_median_ratio": (
                MINIMUM_SHELL_TO_POSITIVE_B0_MEDIAN_RATIO
            ),
            "per_subject_parameter_tuning_allowed": False,
        },
        "checks": checks,
        "control_n": len(controls),
        "target_n": len(targets),
        "control_summary": {
            "maximum_mask_volume_ratio": max(
                row["mask_volume_ratio"] for row in controls
            ),
            "minimum_shell_to_positive_b0_median_ratio": min(
                row["shell_to_positive_b0_median_ratio"]
                for row in controls
            ),
            "minimum_union_tissue_dice": min(
                row["union_tissue_dice"] for row in controls
            ),
        },
        "targets": targets,
        "rows": {
            "path": str(ROWS.resolve()),
        },
        "records": {
            "promoted_canaries": file_record(PROMOTED),
            "failure_ledger": file_record(LEDGER),
            "implementation": file_record(Path(__file__)),
        },
    }
    atomic_csv(
        ROWS,
        [
            {key: value for key, value in row.items() if key != "artifacts"}
            for row in rows
        ],
    )
    payload["rows"] = file_record(ROWS)
    atomic_json(SUMMARY, payload)
    return payload


def preflight() -> dict[str, Any]:
    records = canary_records() + target_records()
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_mask_union_recovery_control_audit_preflight",
        "status": "READY",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "imaging_executed": False,
        "control_n": sum(
            record["role"] == "reviewed_canary_control"
            for record in records
        ),
        "target_n": sum(
            record["role"] == "registration_failure_target"
            for record in records
        ),
        "target_units": sorted(TARGET_ROUTES),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if (
            len(TARGET_ROUTES) != 2
            or MINIMUM_HEMISPHERE_ATLAS_INSIDE != 0.80
            or MINIMUM_TISSUE_DICE != 0.70
            or MAXIMUM_MASK_VOLUME_RATIO != 1.15
        ):
            raise AssertionError("mask-recovery constants differ")
        print("HCP379_MASK_UNION_RECOVERY_CONTROL_AUDIT_SELF_TEST_PASS")
        return 0
    payload = execute() if args.execute else preflight()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] in {"READY", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
