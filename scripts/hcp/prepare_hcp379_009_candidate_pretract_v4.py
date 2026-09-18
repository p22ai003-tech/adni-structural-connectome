#!/home/ec2-user/fsl/bin/python
"""Finalize the non-promoted 009 conservative-SyN pretract candidate.

The expensive spatial candidate has already been generated under a dedicated,
non-overwriting diagnostic root.  This program validates its complete 5TT,
GMWMI and HCP379 inputs, writes subject-level quantitative QC, and assembles a
15-unit provisional successor manifest with the other 14 Recovery4 canaries
unchanged.  It never writes the canonical human-QC file, never replaces the
Recovery4 master manifest, and never starts tractography.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np


ROOT = Path("/data/derivatives/hcp379_v2")
UNIT = "009_S_4324_I1186579"
ROUTE = "bbr_initialized_syn_conservative"
ORIGINAL_MASTER = (
    ROOT / "pretract_recovery4/pretract_recovery4_manifest.json"
)
TARGETED_ROOT = (
    ROOT
    / "diagnostics/registration_recovery4_targeted"
    / UNIT
)
TARGETED_AUDIT = TARGETED_ROOT / "targeted_registration_audit.json"
CANDIDATE_INPUT_ROOT = TARGETED_ROOT / "ants_syn_bbrinit"
OUTPUT_ROOT = ROOT / "pretract_candidate_recovery4_v4"
UNIT_ROOT = OUTPUT_ROOT / "subjects" / UNIT
UNIT_QC = UNIT_ROOT / "candidate_pretract_qc.json"
ATLAS_QC = UNIT_ROOT / "hcp379_candidate_atlas_qc.json"
NODE_VOLUMES = UNIT_ROOT / "hcp379_candidate_node_volumes.csv"
UNIT_MANIFEST = UNIT_ROOT / "pretract_candidate_recovery4_v4_manifest.json"
MASTER_MANIFEST = (
    OUTPUT_ROOT / "pretract_candidate_recovery4_v4_manifest.json"
)

FIVE_TT_NIFTI = UNIT_ROOT / "5tt_b0_candidate_raw.nii.gz"
FIVE_TT_MIF = UNIT_ROOT / "5tt_b0_candidate_raw.mif"
FIVE_TT_MASK = UNIT_ROOT / "5tt_mask_b0_candidate.nii.gz"
GMWMI = UNIT_ROOT / "gmwmi_b0_candidate.mif"
NODES_1MM = (
    CANDIDATE_INPUT_ROOT / "hcp379_nodes_b0_1mm_syn_bbrinit.nii.gz"
)
NODES_NATIVE = (
    CANDIDATE_INPUT_ROOT / "hcp379_nodes_b0_native_syn_bbrinit.nii.gz"
)
SELECTED_T1 = (
    CANDIDATE_INPUT_ROOT
    / "t1_to_b0_syn_bbrinit_Warped.nii.gz"
)
REVIEW_HCP = (
    CANDIDATE_INPUT_ROOT
    / "review_b0_vs_hcp379_edges_syn_bbrinit.png"
)
REVIEW_T1 = (
    CANDIDATE_INPUT_ROOT / "review_b0_vs_t1_syn_bbrinit.png"
)
AFFINE = (
    CANDIDATE_INPUT_ROOT
    / "t1_to_b0_syn_bbrinit_0GenericAffine.mat"
)
WARP = (
    CANDIDATE_INPUT_ROOT / "t1_to_b0_syn_bbrinit_1Warp.nii.gz"
)
JACOBIAN = CANDIDATE_INPUT_ROOT / "jacobian_syn_bbrinit.nii.gz"
PRIOR_CANDIDATE_MASK = (
    CANDIDATE_INPUT_ROOT / "5tt_mask_b0_syn_bbrinit.nii.gz"
)
DWI_MASK_NIFTI = UNIT_ROOT / "dwi_mask_quantitative.nii.gz"
MRINFO = Path("/home/ec2-user/mrtrix3/bin/mrinfo")
FIVE_TT_CHECK = Path("/home/ec2-user/mrtrix3/bin/5ttcheck")


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
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_file_record(
    record: Mapping[str, Any],
    label: str,
) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if file_record(path) != dict(record):
        raise ValueError(f"{label}: artifact binding differs")
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(
            dict(value),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        newline="",
        encoding="utf-8",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["node_id", "volume_mm3", "n_voxels"],
        )
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def dice(first: np.ndarray, second: np.ndarray) -> float:
    denominator = int(np.count_nonzero(first)) + int(
        np.count_nonzero(second)
    )
    if denominator == 0:
        raise ValueError("empty masks")
    return 2.0 * int(np.count_nonzero(first & second)) / denominator


def verified_image(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"image is missing or symbolic: {path}")
    image = nib.load(path)
    values = image.get_fdata()
    if not np.all(np.isfinite(values)):
        raise ValueError(f"image contains non-finite values: {path}")
    return image, values


def mrinfo_size(path: Path) -> list[int]:
    completed = subprocess.run(
        [str(MRINFO), str(path), "-size"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [int(value) for value in completed.stdout.split()]


def validate_five_tt(
    dwi_mask_path: Path,
) -> dict[str, Any]:
    check = subprocess.run(
        [str(FIVE_TT_CHECK), str(FIVE_TT_MIF)],
        check=False,
        capture_output=True,
        text=True,
    )
    if check.returncode:
        raise ValueError(
            f"5ttcheck failed: {check.stdout}\n{check.stderr}"
        )
    if mrinfo_size(FIVE_TT_MIF) != [256, 256, 80, 5]:
        raise ValueError("candidate 5TT dimensions differ")
    if mrinfo_size(GMWMI) != [256, 256, 80]:
        raise ValueError("candidate GMWMI dimensions differ")
    _, values = verified_image(FIVE_TT_NIFTI)
    if (
        values.shape != (256, 256, 80, 5)
        or values.min() < -1.0e-8
        or values.max() > 1.0 + 1.0e-6
    ):
        raise ValueError("candidate 5TT numeric contract differs")
    sums = values.sum(axis=3)
    candidate_mask = verified_image(FIVE_TT_MASK)[1] > 0.5
    prior_mask = verified_image(PRIOR_CANDIDATE_MASK)[1] > 0.5
    dwi_mask = verified_image(dwi_mask_path)[1] > 0.5
    five_tt_dwi_dice = dice(candidate_mask, dwi_mask)
    candidate_prior_dice = dice(candidate_mask, prior_mask)
    if (
        five_tt_dwi_dice < 0.70
        or candidate_prior_dice < 0.99
        or not np.array_equal(candidate_mask, sums > 0.5)
    ):
        raise ValueError("candidate 5TT support contract differs")
    nonzero_sums = sums[sums > 0]
    return {
        "five_tt_dwi_mask_dice": five_tt_dwi_dice,
        "candidate_mask_vs_prior_diagnostic_mask_dice": (
            candidate_prior_dice
        ),
        "candidate_mask_voxels": int(candidate_mask.sum()),
        "dwi_mask_voxels": int(dwi_mask.sum()),
        "five_tt_minimum": float(values.min()),
        "five_tt_maximum": float(values.max()),
        "five_tt_sum_nonzero_minimum": float(nonzero_sums.min()),
        "five_tt_sum_nonzero_median": float(
            np.median(nonzero_sums)
        ),
        "five_tt_sum_maximum": float(sums.max()),
        "five_ttcheck_pass": True,
        "five_tt_dimensions": [256, 256, 80, 5],
        "gmwmi_dimensions": [256, 256, 80],
    }


def validate_atlas(
    dwi_mask_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image_1mm, values_1mm = verified_image(NODES_1MM)
    image_native, values_native = verified_image(NODES_NATIVE)
    dwi_mask = verified_image(dwi_mask_path)[1] > 0.5
    rounded_1mm = np.rint(values_1mm).astype(np.int32)
    rounded_native = np.rint(values_native).astype(np.int32)
    labels = sorted(
        int(value)
        for value in np.unique(rounded_1mm)
        if value > 0
    )
    native_labels = sorted(
        int(value)
        for value in np.unique(rounded_native)
        if value > 0
    )
    if labels != list(range(1, 380)) or native_labels != labels:
        raise ValueError("candidate HCP379 labels differ")
    voxel_volume = float(np.prod(image_1mm.header.get_zooms()[:3]))
    rows = []
    counts: dict[int, int] = {}
    for label in labels:
        count = int(np.count_nonzero(rounded_1mm == label))
        counts[label] = count
        rows.append(
            {
                "node_id": label,
                "volume_mm3": f"{count * voxel_volume:.4f}",
                "n_voxels": count,
            }
        )
    atlas_mask = rounded_native > 0
    left_mask = (rounded_native >= 1) & (rounded_native <= 180)
    right_mask = (rounded_native >= 181) & (rounded_native <= 360)

    def inside(mask: np.ndarray) -> float:
        denominator = int(mask.sum())
        if not denominator:
            raise ValueError("empty atlas subgroup")
        return int(np.count_nonzero(mask & dwi_mask)) / denominator

    qc = {
        "labels_present": len(labels),
        "missing_labels": [],
        "minimum_node_voxels_1mm": min(counts.values()),
        "maximum_node_voxels_1mm": max(counts.values()),
        "total_atlas_voxels_1mm": sum(counts.values()),
        "one_mm_voxel_volume_mm3": voxel_volume,
        "atlas_inside_dwi_mask_fraction": inside(atlas_mask),
        "left_cortical_inside_dwi_mask_fraction": inside(left_mask),
        "right_cortical_inside_dwi_mask_fraction": inside(right_mask),
        "nodes_1mm_shape": list(values_1mm.shape),
        "nodes_native_shape": list(values_native.shape),
    }
    if (
        qc["minimum_node_voxels_1mm"] < 1
        or qc["atlas_inside_dwi_mask_fraction"] < 0.80
        or qc["left_cortical_inside_dwi_mask_fraction"] < 0.80
        or qc["right_cortical_inside_dwi_mask_fraction"] < 0.80
    ):
        raise ValueError("candidate HCP379 quantitative gate failed")
    return qc, rows


def main() -> int:
    for path in (UNIT_QC, ATLAS_QC, NODE_VOLUMES, UNIT_MANIFEST, MASTER_MANIFEST):
        if path.exists():
            raise FileExistsError(
                f"non-overwriting candidate artifact exists: {path}"
            )
    original_master = load_json(ORIGINAL_MASTER)
    if (
        original_master.get("record_type")
        != "diagnosis_blind_hcp379_pretract_recovery4_manifest"
        or original_master.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or original_master.get("unit_count") != 15
        or original_master.get("diagnosis_labels_used") is not False
        or original_master.get("human_visual_qc_inferred") is not False
    ):
        raise ValueError("original Recovery4 master differs")
    original_row = next(
        row for row in original_master["units"]
        if row["unit"] == UNIT
    )
    original_unit_path = verify_file_record(
        original_row["manifest"], "original 009 unit manifest"
    )
    original_unit = load_json(original_unit_path)
    dwi_mask_path = verify_file_record(
        original_unit["tractography_inputs"]["dwi_mask"],
        "009 DWI mask",
    )

    targeted = load_json(TARGETED_AUDIT)
    if (
        targeted.get("status")
        != "PROVISIONAL_CANDIDATE_AWAITING_HUMAN_VISUAL_QC"
        or targeted.get("unit") != UNIT
        or targeted.get("diagnosis_labels_used") is not False
        or targeted.get("tractography_started") is not False
        or targeted.get("matrix_generation_started") is not False
        or targeted.get("selection", {}).get("recommended_candidate")
        != ROUTE
        or targeted.get("selection", {}).get("promotion_status")
        != "NOT_PROMOTED"
    ):
        raise ValueError("targeted 009 audit differs")

    if not DWI_MASK_NIFTI.is_file() or DWI_MASK_NIFTI.is_symlink():
        raise ValueError("009 quantitative DWI-mask NIfTI differs")
    five_tt_qc = validate_five_tt(DWI_MASK_NIFTI)
    atlas_qc, node_rows = validate_atlas(DWI_MASK_NIFTI)
    targeted_candidate = next(
        row for row in targeted["candidates"]
        if row["candidate"] == ROUTE
    )
    if (
        abs(
            atlas_qc["atlas_inside_dwi_mask_fraction"]
            - float(targeted_candidate["atlas_inside_dwi_overall"])
        )
        > 0.005
        or abs(
            atlas_qc["left_cortical_inside_dwi_mask_fraction"]
            - float(targeted_candidate["left_cortical_inside_dwi"])
        )
        > 0.005
        or abs(
            atlas_qc["right_cortical_inside_dwi_mask_fraction"]
            - float(targeted_candidate["right_cortical_inside_dwi"])
        )
        > 0.005
    ):
        raise ValueError("recomputed atlas metrics differ from targeted audit")
    jacobian = targeted_candidate["jacobian_in_dwi_mask"]
    if jacobian["nonpositive_fraction"] != 0:
        raise ValueError("candidate transform contains folding")

    atomic_csv(NODE_VOLUMES, node_rows)
    atlas_payload = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_009_candidate_atlas_qc"
        ),
        "status": "PASS_PROVISIONAL_AWAITING_HUMAN_VISUAL_QC",
        "generated_utc": utc_now(),
        "unit": UNIT,
        "route": ROUTE,
        "diagnosis_labels_used": False,
        **atlas_qc,
        "jacobian_in_dwi_mask": jacobian,
        "artifacts": {
            "nodes_1mm": file_record(NODES_1MM),
            "nodes_native": file_record(NODES_NATIVE),
            "node_volumes": file_record(NODE_VOLUMES),
            "review_hcp379": file_record(REVIEW_HCP),
        },
    }
    atomic_json(ATLAS_QC, atlas_payload)
    candidate_qc = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_009_candidate_pretract_qc"
        ),
        "status": "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC",
        "generated_utc": utc_now(),
        "unit": UNIT,
        "route": ROUTE,
        "diagnosis_labels_used": False,
        "human_visual_qc_inferred": False,
        "tractography_started": False,
        "matrix_generation_started": False,
        "five_tt_qc": five_tt_qc,
        "atlas_qc": atlas_qc,
        "jacobian_in_dwi_mask": jacobian,
        "thresholds": {
            "minimum_5tt_dwi_mask_dice": 0.70,
            "minimum_atlas_inside_dwi_mask_fraction": 0.80,
            "minimum_each_cortical_hemisphere_inside_dwi_mask_fraction": 0.80,
            "required_hcp379_labels": 379,
            "jacobian_nonpositive_fraction": 0,
        },
        "artifacts": {
            "five_tt": file_record(FIVE_TT_MIF),
            "five_tt_nifti": file_record(FIVE_TT_NIFTI),
            "five_tt_mask": file_record(FIVE_TT_MASK),
            "quantitative_dwi_mask": file_record(DWI_MASK_NIFTI),
            "gmwmi": file_record(GMWMI),
            "nodes_1mm": file_record(NODES_1MM),
            "nodes_native": file_record(NODES_NATIVE),
            "selected_t1": file_record(SELECTED_T1),
            "affine": file_record(AFFINE),
            "warp": file_record(WARP),
            "jacobian": file_record(JACOBIAN),
        },
    }
    atomic_json(UNIT_QC, candidate_qc)

    unit_manifest = {
        **original_unit,
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_pretract_candidate_recovery4_v4_"
            "unit_manifest"
        ),
        "status": "PROVISIONAL_AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC",
        "completed_utc": utc_now(),
        "diagnosis_labels_used": False,
        "human_visual_qc_inferred": False,
        "tractography_started": False,
        "matrix_generation_started": False,
        "registration": {
            "route": ROUTE,
            "selected_5tt_dwi_mask_dice": (
                five_tt_qc["five_tt_dwi_mask_dice"]
            ),
            "hcp379_atlas_inside_dwi_mask_fraction": (
                atlas_qc["atlas_inside_dwi_mask_fraction"]
            ),
            "left_cortical_inside_dwi_mask_fraction": (
                atlas_qc[
                    "left_cortical_inside_dwi_mask_fraction"
                ]
            ),
            "right_cortical_inside_dwi_mask_fraction": (
                atlas_qc[
                    "right_cortical_inside_dwi_mask_fraction"
                ]
            ),
            "selected_t1_in_b0": file_record(SELECTED_T1),
            "transform_chain": {
                "affine": file_record(AFFINE),
                "warp": file_record(WARP),
                "jacobian": file_record(JACOBIAN),
            },
        },
        "tractography_inputs": {
            **original_unit["tractography_inputs"],
            "five_tt": file_record(FIVE_TT_MIF),
            "gmwmi": file_record(GMWMI),
        },
        "hcp379": {
            **original_unit["hcp379"],
            "nodes_b0_1mm": file_record(NODES_1MM),
            "nodes_b0_native_qc": file_record(NODES_NATIVE),
            "node_volumes": file_record(NODE_VOLUMES),
            "atlas_qc": file_record(ATLAS_QC),
            "visual_overlay": file_record(REVIEW_HCP),
        },
        "automated_qc": {
            **original_unit["automated_qc"],
            "candidate_pretract": file_record(UNIT_QC),
            "candidate_atlas": file_record(ATLAS_QC),
            "targeted_registration": file_record(TARGETED_AUDIT),
        },
        "provisional_route_provenance": {
            "original_recovery4_unit_manifest": file_record(
                original_unit_path
            ),
            "candidate_pretract_qc": file_record(UNIT_QC),
            "targeted_registration_audit": file_record(TARGETED_AUDIT),
            "review_t1": file_record(REVIEW_T1),
            "review_hcp379": file_record(REVIEW_HCP),
            "promotion_status": "NOT_PROMOTED",
        },
    }
    atomic_json(UNIT_MANIFEST, unit_manifest)

    master_rows = []
    for row in original_master["units"]:
        if row["unit"] == UNIT:
            master_rows.append(
                {
                    "unit": UNIT,
                    "status": (
                        "PROVISIONAL_AUTOMATED_PASS_AWAITING_"
                        "HUMAN_VISUAL_QC"
                    ),
                    "registration_route": ROUTE,
                    "manifest": file_record(UNIT_MANIFEST),
                }
            )
        else:
            verify_file_record(
                row["manifest"], f"{row['unit']}: original manifest"
            )
            master_rows.append(row)
    master = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_pretract_candidate_recovery4_v4_"
            "manifest"
        ),
        "status": "PROVISIONAL_AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "human_visual_qc_inferred": False,
        "tractography_started": False,
        "matrix_generation_started": False,
        "non_overwriting": True,
        "unit_count": 15,
        "route_substitution": {
            "unit": UNIT,
            "old_route": original_row["registration_route"],
            "candidate_route": ROUTE,
            "promotion_status": "NOT_PROMOTED",
            "human_review_required": True,
        },
        "route_counts": {
            route: sum(
                1
                for row in master_rows
                if row["registration_route"] == route
            )
            for route in sorted(
                {row["registration_route"] for row in master_rows}
            )
        },
        "units": master_rows,
        "sources": {
            "original_recovery4_master": file_record(ORIGINAL_MASTER),
            "targeted_registration_audit": file_record(TARGETED_AUDIT),
            "candidate_unit_manifest": file_record(UNIT_MANIFEST),
        },
        "builder": file_record(Path(__file__)),
    }
    atomic_json(MASTER_MANIFEST, master)
    print(
        json.dumps(
            {
                "status": master["status"],
                "unit_count": master["unit_count"],
                "candidate_unit": UNIT,
                "candidate_route": ROUTE,
                "five_tt_dwi_mask_dice": (
                    five_tt_qc["five_tt_dwi_mask_dice"]
                ),
                "atlas_inside_dwi": (
                    atlas_qc["atlas_inside_dwi_mask_fraction"]
                ),
                "left_inside_dwi": (
                    atlas_qc[
                        "left_cortical_inside_dwi_mask_fraction"
                    ]
                ),
                "right_inside_dwi": (
                    atlas_qc[
                        "right_cortical_inside_dwi_mask_fraction"
                    ]
                ),
                "master_manifest": str(MASTER_MANIFEST),
                "tractography_started": False,
                "promotion_status": "NOT_PROMOTED",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
