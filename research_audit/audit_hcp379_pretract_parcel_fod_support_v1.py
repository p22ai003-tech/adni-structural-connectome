#!/home/ec2-user/fsl/bin/python
"""Audit parcel-level HCP379 support in the pre-tract WM-FOD field.

Recovery4 already requires all 379 labels, global atlas-in-mask coverage,
bilateral cortical atlas-in-mask coverage, and 5TT/DWI-mask agreement.  Those
coarse gates can still miss a small parcel with no direct positive WM-FOD
support.  Such a parcel is at risk of becoming a disconnected node after
tractography, regardless of the total requested streamline count.

This audit resamples each selected HCP379 atlas to its native WM-FOD grid and
reports, for every parcel:

* the fraction of atlas voxels with a positive finite l=0 WM-FOD coefficient;
* the fraction lying within 4 mm of any positive finite l=0 coefficient;
* the mean l=0 coefficient and minimum distance to positive FOD support.

The audit is diagnosis blind, outcome blind, non-overwriting, and
non-executing with respect to tractography.  It does not alter subject states
or authorize an exclusion.  A zero-support result is initially a review flag,
not a final release rule; the summary separately evaluates the rule against
the 15 completed HROI canaries and two lower-direction mechanism references.
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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt


EXP = Path("/home/ec2-user/exp")
HCP = Path("/data/derivatives/hcp379_v2")
OUTPUT_ROOT = (
    EXP
    / "research_audit/outputs/hcp379_pretract_parcel_fod_support_v1"
)
TOPOLOGY = (
    EXP
    / "research_audit/outputs/hcp379_balanced_release_topology_v1/"
    "topology.json"
)
TOPOLOGY_VALIDATION = TOPOLOGY.with_name("validation.json")
LEDGER = (
    EXP
    / "research_audit/outputs/hcp379_pretract_failure_recovery_ledger_v1/"
    "ledger.csv"
)
LEDGER_JSON = LEDGER.with_suffix(".json")
LEDGER_VALIDATION = LEDGER.with_name("validation.json")
FOD_COMPATIBILITY = (
    EXP
    / "research_audit/outputs/hcp379_fod_order_compatibility_v1/"
    "fod_order_compatibility.csv"
)
FOD_COMPATIBILITY_VALIDATION = FOD_COMPATIBILITY.with_name(
    "validation.json"
)
CANARY_COHORT_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_hroi_balanced_density_canary_cohort_v1/validation.json"
)
LOWER_ORDER_FOD_SUMMARY = (
    EXP
    / "research_audit/outputs/"
    "hcp379_acquisition_aware_fod_order_canary_v1/summary.json"
)
LOWER_ORDER_FOD_VALIDATION = LOWER_ORDER_FOD_SUMMARY.with_name(
    "validation.json"
)
LOWER_ORDER_TRACT_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_acquisition_aware_fod_order_tractography_canary_v1/"
    "validation.json"
)
ENDPOINT_RADIUS_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_fod_order_endpoint_radius_localization_v1/validation.json"
)
PRETRACT_CANARY_ROOT = HCP / "pretract_scalar_recovery4/subjects"
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")

EXPECTED_TOTAL = 530
EXPECTED_PRODUCTION = 515
EXPECTED_CANARIES = 15
EXPECTED_NODES = 379
LEFT_CORTEX = range(1, 181)
RIGHT_CORTEX = range(181, 361)
ASSIGNMENT_RADIUS_MM = 4.0
LOW_DIRECT_SUPPORT_FRACTION = 0.01
MINIMUM_ATLAS_MASK_FRACTION = 0.80

SUBJECT_FIELDS = (
    "unit",
    "role",
    "lane",
    "pretract_status",
    "release_input_eligible",
    "support_source",
    "atlas_variant",
    "audit_status",
    "recommended_action",
    "error",
    "recommended_lmax",
    "unique_direction_n",
    "atlas_qc_status",
    "atlas_inside_dwi_mask_fraction",
    "left_cortical_inside_dwi_mask_fraction",
    "right_cortical_inside_dwi_mask_fraction",
    "labels_found_on_fod_grid",
    "zero_direct_fod_support_parcel_n",
    "lt_1pct_direct_fod_support_parcel_n",
    "zero_within_4mm_fod_support_parcel_n",
    "left_zero_direct_n",
    "right_zero_direct_n",
    "subcortical_zero_direct_n",
    "minimum_direct_fod_support_fraction",
    "p05_direct_fod_support_fraction",
    "median_direct_fod_support_fraction",
    "canary_20m_density",
    "canary_20m_connected_nodes",
    "atlas_path",
    "atlas_sha256",
    "fod_l0_path",
    "fod_l0_sha256",
)

PARCEL_FIELDS = (
    "unit",
    "role",
    "lane",
    "pretract_status",
    "release_input_eligible",
    "support_source",
    "parcel_id",
    "parcel_class",
    "atlas_voxels_on_fod_grid",
    "positive_fod_voxels",
    "direct_fod_support_fraction",
    "within_4mm_positive_fod_voxels",
    "within_4mm_fod_support_fraction",
    "minimum_distance_to_positive_fod_mm",
    "mean_fod_l0",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def require_validation(path: Path) -> dict[str, Any]:
    value = load_json(path)
    passed = int(
        value.get("passed_checks", value.get("checks_passed", -1))
    )
    total = int(
        value.get("total_checks", value.get("checks_total", -1))
    )
    if value.get("status") != "PASS" or total <= 0 or passed != total:
        raise ValueError(f"validation is not complete PASS: {path}")
    return value


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
        or resolved.stat().st_size <= 0
    ):
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def verify_record(record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if file_record(path) != {
        "path": str(path),
        "size_bytes": int(record.get("size_bytes", -1)),
        "sha256": str(record.get("sha256", "")),
    }:
        raise ValueError(f"file record drift: {path}")
    return path


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
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


def atomic_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Iterable[str],
) -> None:
    field_list = list(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=field_list,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parcel_class(parcel_id: int) -> str:
    if parcel_id in LEFT_CORTEX:
        return "LEFT_CORTEX"
    if parcel_id in RIGHT_CORTEX:
        return "RIGHT_CORTEX"
    return "SUBCORTICAL"


def finite_float(value: Any) -> float | str:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return ""
    return result if math.isfinite(result) else ""


def canary_inputs() -> dict[str, dict[str, Any]]:
    validation = require_validation(CANARY_COHORT_VALIDATION)
    readiness = validation["checks"][
        "dry_run_is_live_fail_closed_and_nonexecuting"
    ]["evidence"]["readiness"]
    result: dict[str, dict[str, Any]] = {}
    for item in readiness:
        unit = str(item["unit"])
        summary_path = Path(
            str(item["existing_valid_summary"])
        ).resolve()
        summary = load_json(summary_path)
        hroi_nodes = verify_record(summary["hroi_nodes"])
        atlas_qc = hroi_nodes.parent / "atlas_qc.json"
        level = summary.get("levels", {}).get("20000000", {})
        fod = (
            PRETRACT_CANARY_ROOT
            / unit
            / "05_model/wmfod_l0_qc_recovery4.mif"
        )
        result[unit] = {
            "atlas": hroi_nodes,
            "atlas_qc": atlas_qc,
            "fod": fod,
            "density_20m": finite_float(level.get("edge_density")),
            "connected_20m": level.get("connected_nodes", ""),
        }
    if len(result) != EXPECTED_CANARIES:
        raise ValueError("canary input set differs")
    return result


def lower_order_inputs() -> dict[str, dict[str, Any]]:
    require_validation(LOWER_ORDER_FOD_VALIDATION)
    summary = load_json(LOWER_ORDER_FOD_SUMMARY)
    result: dict[str, dict[str, Any]] = {}
    for item in summary.get("units", []):
        if item.get("status") != "PASS":
            continue
        unit = str(item["unit"])
        value = load_json(verify_record(item["result"]))
        fod = verify_record(value["artifacts"]["wmfod_l0"])
        subject = (
            HCP / "corrected_scaleup_recovery4/subjects" / unit
        )
        result[unit] = {
            "atlas": subject
            / "04_hcp/hcp379_nodes_b0_1mm.nii.gz",
            "atlas_qc": subject / "04_hcp/atlas_qc.json",
            "fod": fod,
            "recommended_lmax": int(item["recommended_lmax"]),
            "unique_direction_n": int(item["unique_direction_n"]),
        }
    if set(result) != {
        "031_S_0618_I1229293",
        "031_S_4721_I1093826",
    }:
        raise ValueError("lower-order mechanism set differs")
    return result


def build_specs() -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, str]],
]:
    require_validation(TOPOLOGY_VALIDATION)
    require_validation(LEDGER_VALIDATION)
    require_validation(FOD_COMPATIBILITY_VALIDATION)
    topology = load_json(TOPOLOGY)
    if (
        topology.get("total_unit_n") != EXPECTED_TOTAL
        or topology.get("exact_15_plus_515_equals_530") is not True
        or len(topology.get("production", [])) != EXPECTED_PRODUCTION
        or len(topology.get("canaries", [])) != EXPECTED_CANARIES
    ):
        raise ValueError("release topology differs")

    ledger_rows = read_csv(LEDGER)
    if (
        len(ledger_rows) != EXPECTED_PRODUCTION
        or len({row["unit"] for row in ledger_rows})
        != EXPECTED_PRODUCTION
    ):
        raise ValueError("production ledger differs")
    ledger_by_unit = {row["unit"]: row for row in ledger_rows}
    topology_production = {
        str(row["unit"]): row for row in topology["production"]
    }
    if set(ledger_by_unit) != set(topology_production):
        raise ValueError("topology/ledger production identities differ")

    canaries = {str(row["unit"]): row for row in topology["canaries"]}
    if set(canaries) & set(ledger_by_unit):
        raise ValueError("canary/production identities overlap")
    all_units = set(canaries) | set(ledger_by_unit)
    if len(all_units) != EXPECTED_TOTAL:
        raise ValueError("exact 530 identity partition differs")

    fod_rows = read_csv(FOD_COMPATIBILITY)
    if (
        len(fod_rows) != EXPECTED_TOTAL
        or len({row["unit"] for row in fod_rows}) != EXPECTED_TOTAL
    ):
        raise ValueError("FOD compatibility identity set differs")
    fod_by_unit = {row["unit"]: row for row in fod_rows}
    if set(fod_by_unit) != all_units:
        raise ValueError("FOD/topology identity sets differ")

    canary_paths = canary_inputs()
    lower_paths = lower_order_inputs()
    specs: list[dict[str, Any]] = []
    for unit in sorted(all_units):
        fod_row = fod_by_unit[unit]
        base: dict[str, Any] = {
            "unit": unit,
            "recommended_lmax": int(
                float(fod_row["recommended_lmax"])
            ),
            "unique_direction_n": int(
                float(
                    fod_row[
                        "selected_shell_unique_antipodal_direction_n"
                    ]
                )
            ),
            "density_20m": "",
            "connected_20m": "",
            "atlas": None,
            "atlas_qc": None,
            "fod": None,
        }
        if unit in canaries:
            values = canary_paths[unit]
            base.update(
                {
                    "role": "CANARY_REUSE",
                    "lane": str(canaries[unit]["lane"]),
                    "pretract_status": "PASS_COMPACTED",
                    "release_input_eligible": True,
                    "support_source": "FROZEN_PRETRACT_LMAX6",
                    "atlas_variant": "FINAL_HROI_CANARY",
                    **values,
                }
            )
        else:
            ledger_row = ledger_by_unit[unit]
            topology_row = topology_production[unit]
            status = str(ledger_row["status"])
            base.update(
                {
                    "role": "PRODUCTION",
                    "lane": str(ledger_row["lane"]),
                    "pretract_status": status,
                    "release_input_eligible": (
                        status == "PASS_COMPACTED"
                    ),
                    "support_source": "",
                    "atlas_variant": "",
                }
            )
            if status == "PASS_COMPACTED":
                compaction = load_json(
                    Path(ledger_row["compaction_path"]).resolve()
                )
                retained = compaction.get("retained_artifacts", {})
                base.update(
                    {
                        "support_source": (
                            "FROZEN_PRETRACT_SELECTED_RECIPE"
                        ),
                        "atlas_variant": (
                            "COHORT_UNIFORM_HROI_SELECTED"
                        ),
                        "atlas": verify_record(
                            retained["selected_hcp379_nodes"]
                        ),
                        "atlas_qc": verify_record(
                            retained["selected_hcp379_atlas_qc"]
                        ),
                        "fod": verify_record(
                            retained["wmfod_l0"]
                        ),
                    }
                )
                if (
                    topology_row.get("pretract", {}).get("ready")
                    is not True
                ):
                    raise ValueError(
                        f"ready topology differs: {unit}"
                    )
            elif unit in lower_paths:
                values = lower_paths[unit]
                base.update(
                    {
                        "support_source": (
                            "LOWER_ORDER_COUNTERFACTUAL_REFERENCE"
                        ),
                        "atlas_variant": (
                            "ATLAS_USED_BY_LOWER_ORDER_CANARY"
                        ),
                        **values,
                    }
                )
        specs.append(base)
    return specs, fod_by_unit


def run_command(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): "
            f"{' '.join(command)}\n{completed.stderr}"
        )


def audit_one(
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row: dict[str, Any] = {
        field: "" for field in SUBJECT_FIELDS
    }
    for field in (
        "unit",
        "role",
        "lane",
        "pretract_status",
        "release_input_eligible",
        "support_source",
        "atlas_variant",
        "recommended_lmax",
        "unique_direction_n",
    ):
        row[field] = spec.get(field, "")
    row["canary_20m_density"] = spec.get("density_20m", "")
    row["canary_20m_connected_nodes"] = spec.get(
        "connected_20m", ""
    )
    atlas = spec.get("atlas")
    fod = spec.get("fod")
    atlas_qc_path = spec.get("atlas_qc")
    if not isinstance(atlas, Path) or not isinstance(fod, Path):
        row["audit_status"] = "NOT_READY"
        row["recommended_action"] = (
            "WAIT_FOR_VALIDATED_PRETRACT_INPUTS"
        )
        return row, []

    try:
        atlas_record = file_record(atlas)
        fod_record = file_record(fod)
        row["atlas_path"] = atlas_record["path"]
        row["atlas_sha256"] = atlas_record["sha256"]
        row["fod_l0_path"] = fod_record["path"]
        row["fod_l0_sha256"] = fod_record["sha256"]
        atlas_qc: dict[str, Any] = {}
        if isinstance(atlas_qc_path, Path):
            atlas_qc = load_json(atlas_qc_path)
        row["atlas_qc_status"] = atlas_qc.get("status", "")
        row["atlas_inside_dwi_mask_fraction"] = finite_float(
            atlas_qc.get("atlas_inside_dwi_mask_fraction")
        )
        hemispheres = atlas_qc.get(
            "cortical_hemisphere_inside_dwi_mask_fraction", {}
        )
        if not isinstance(hemispheres, Mapping):
            hemispheres = {}
        row["left_cortical_inside_dwi_mask_fraction"] = (
            finite_float(hemispheres.get("left"))
        )
        row["right_cortical_inside_dwi_mask_fraction"] = (
            finite_float(hemispheres.get("right"))
        )

        with tempfile.TemporaryDirectory(
            prefix=f"hcp379-fod-support-{spec['unit']}-"
        ) as temporary:
            temporary_root = Path(temporary)
            atlas_native = temporary_root / "atlas_on_fod.nii.gz"
            fod_nifti = temporary_root / "fod_l0.nii.gz"
            run_command(
                [
                    str(MRTRIX / "mrtransform"),
                    str(atlas),
                    str(atlas_native),
                    "-template",
                    str(fod),
                    "-interp",
                    "nearest",
                    "-datatype",
                    "uint16",
                    "-force",
                    "-quiet",
                ]
            )
            run_command(
                [
                    str(MRTRIX / "mrconvert"),
                    str(fod),
                    str(fod_nifti),
                    "-coord",
                    "3",
                    "0",
                    "-force",
                    "-quiet",
                ]
            )
            atlas_image = nib.load(str(atlas_native))
            fod_image = nib.load(str(fod_nifti))
            labels = np.rint(
                np.asanyarray(atlas_image.dataobj)
            ).astype(np.int16)
            fod_values = np.squeeze(
                np.asanyarray(fod_image.dataobj).astype(np.float64)
            )
            if labels.shape != fod_values.shape:
                raise ValueError("resampled atlas/FOD geometry differs")
            zooms = tuple(
                float(value)
                for value in fod_image.header.get_zooms()[:3]
            )
            if len(zooms) != 3 or any(
                not math.isfinite(value) or value <= 0
                for value in zooms
            ):
                raise ValueError("invalid FOD voxel spacing")

            label_values = set(
                int(value)
                for value in np.unique(labels)
                if int(value) > 0
            )
            row["labels_found_on_fod_grid"] = len(label_values)
            finite = np.isfinite(fod_values)
            positive = finite & (fod_values > 0)
            if not np.any(positive):
                raise ValueError("FOD l=0 contains no positive voxel")
            distance = distance_transform_edt(
                ~positive,
                sampling=zooms,
            )
            flat_labels = labels.ravel().astype(np.int64)
            valid_labels = (
                (flat_labels >= 1)
                & (flat_labels <= EXPECTED_NODES)
            )
            label_vector = flat_labels[valid_labels]
            flat_fod = fod_values.ravel()[valid_labels]
            flat_positive = positive.ravel()[valid_labels]
            flat_near = (
                distance.ravel()[valid_labels]
                <= ASSIGNMENT_RADIUS_MM
            )
            flat_distance = distance.ravel()[valid_labels]
            counts = np.bincount(
                label_vector,
                minlength=EXPECTED_NODES + 1,
            )
            positive_counts = np.bincount(
                label_vector[flat_positive],
                minlength=EXPECTED_NODES + 1,
            )
            near_counts = np.bincount(
                label_vector[flat_near],
                minlength=EXPECTED_NODES + 1,
            )
            finite_weights = np.where(
                np.isfinite(flat_fod),
                flat_fod,
                0.0,
            )
            sums = np.bincount(
                label_vector,
                weights=finite_weights,
                minlength=EXPECTED_NODES + 1,
            )
            minimum_distance = np.full(
                EXPECTED_NODES + 1,
                np.inf,
                dtype=float,
            )
            np.minimum.at(
                minimum_distance,
                label_vector,
                flat_distance,
            )

            parcel_rows: list[dict[str, Any]] = []
            direct_values: list[float] = []
            zero_parcels: list[int] = []
            low_parcels: list[int] = []
            zero_near_parcels: list[int] = []
            for parcel_id in range(1, EXPECTED_NODES + 1):
                count = int(counts[parcel_id])
                positive_count = int(positive_counts[parcel_id])
                near_count = int(near_counts[parcel_id])
                direct_fraction = (
                    positive_count / count if count else math.nan
                )
                near_fraction = (
                    near_count / count if count else math.nan
                )
                mean_l0 = (
                    float(sums[parcel_id]) / count
                    if count
                    else math.nan
                )
                if math.isfinite(direct_fraction):
                    direct_values.append(direct_fraction)
                    if direct_fraction == 0:
                        zero_parcels.append(parcel_id)
                    if direct_fraction < LOW_DIRECT_SUPPORT_FRACTION:
                        low_parcels.append(parcel_id)
                if math.isfinite(near_fraction) and near_fraction == 0:
                    zero_near_parcels.append(parcel_id)
                parcel_rows.append(
                    {
                        "unit": spec["unit"],
                        "role": spec["role"],
                        "lane": spec["lane"],
                        "pretract_status": spec["pretract_status"],
                        "release_input_eligible": spec[
                            "release_input_eligible"
                        ],
                        "support_source": spec["support_source"],
                        "parcel_id": parcel_id,
                        "parcel_class": parcel_class(parcel_id),
                        "atlas_voxels_on_fod_grid": count,
                        "positive_fod_voxels": positive_count,
                        "direct_fod_support_fraction": (
                            direct_fraction
                            if math.isfinite(direct_fraction)
                            else ""
                        ),
                        "within_4mm_positive_fod_voxels": near_count,
                        "within_4mm_fod_support_fraction": (
                            near_fraction
                            if math.isfinite(near_fraction)
                            else ""
                        ),
                        "minimum_distance_to_positive_fod_mm": (
                            float(minimum_distance[parcel_id])
                            if math.isfinite(
                                minimum_distance[parcel_id]
                            )
                            else ""
                        ),
                        "mean_fod_l0": (
                            mean_l0
                            if math.isfinite(mean_l0)
                            else ""
                        ),
                    }
                )

        zero_set = set(zero_parcels)
        row.update(
            {
                "zero_direct_fod_support_parcel_n": len(
                    zero_parcels
                ),
                "lt_1pct_direct_fod_support_parcel_n": len(
                    low_parcels
                ),
                "zero_within_4mm_fod_support_parcel_n": len(
                    zero_near_parcels
                ),
                "left_zero_direct_n": len(
                    zero_set & set(LEFT_CORTEX)
                ),
                "right_zero_direct_n": len(
                    zero_set & set(RIGHT_CORTEX)
                ),
                "subcortical_zero_direct_n": len(
                    [
                        value
                        for value in zero_set
                        if value > 360
                    ]
                ),
                "minimum_direct_fod_support_fraction": float(
                    np.min(direct_values)
                ),
                "p05_direct_fod_support_fraction": float(
                    np.quantile(direct_values, 0.05)
                ),
                "median_direct_fod_support_fraction": float(
                    np.median(direct_values)
                ),
            }
        )
        atlas_fraction = row[
            "atlas_inside_dwi_mask_fraction"
        ]
        left_fraction = row[
            "left_cortical_inside_dwi_mask_fraction"
        ]
        right_fraction = row[
            "right_cortical_inside_dwi_mask_fraction"
        ]
        atlas_gate_failure = (
            row["atlas_qc_status"] != "PASS"
            or (
                atlas_fraction != ""
                and float(atlas_fraction)
                < MINIMUM_ATLAS_MASK_FRACTION
            )
            or (
                left_fraction != ""
                and float(left_fraction)
                < MINIMUM_ATLAS_MASK_FRACTION
            )
            or (
                right_fraction != ""
                and float(right_fraction)
                < MINIMUM_ATLAS_MASK_FRACTION
            )
            or atlas_qc.get("original_status") == "FAIL"
        )
        if label_values != set(range(1, EXPECTED_NODES + 1)):
            row["audit_status"] = "REVIEW_ATLAS_LABEL_GRID"
            row["recommended_action"] = (
                "REBUILD_OR_REVIEW_ATLAS_ON_NATIVE_FOD_GRID"
            )
        elif atlas_gate_failure:
            row["audit_status"] = "HOLD_ATLAS_MASK_QC"
            row["recommended_action"] = (
                "LOCALIZE_REGISTRATION_OR_DWI_MASK_FAILURE"
            )
        elif zero_parcels:
            row["audit_status"] = "REVIEW_ZERO_DIRECT_FOD_SUPPORT"
            row["recommended_action"] = (
                "LOCALIZE_PARCEL_DWI_FOD_SUPPORT_BEFORE_TRACTOGRAPHY"
            )
        elif low_parcels:
            row["audit_status"] = "REVIEW_LT1PCT_DIRECT_FOD_SUPPORT"
            row["recommended_action"] = (
                "REVIEW_LOW_SUPPORT_PARCELS_BEFORE_TRACTOGRAPHY"
            )
        else:
            row["audit_status"] = "PASS_SUPPORT_SCREEN"
            row["recommended_action"] = (
                "ELIGIBLE_ON_PARCEL_SUPPORT_DIMENSION_ONLY"
            )
        return row, parcel_rows
    except Exception as error:
        row["audit_status"] = "AUDIT_ERROR"
        row["recommended_action"] = "REPAIR_AUDIT_INPUT_OR_GEOMETRY"
        row["error"] = f"{type(error).__name__}:{error}"
        return row, []


def mechanism_evidence(
    subjects: Mapping[str, Mapping[str, Any]],
    parcels: list[Mapping[str, Any]],
) -> dict[str, Any]:
    tract = require_validation(LOWER_ORDER_TRACT_VALIDATION)
    radius = require_validation(ENDPOINT_RADIUS_VALIDATION)
    disconnected = set(
        int(value)
        for value in radius["checks"][
            "same_29_nodes_remain_disconnected_through_8mm"
        ]["evidence"]["unresolved_within_8mm"]
    )
    bad_unit = "031_S_4721_I1093826"
    good_unit = "031_S_0618_I1229293"
    bad_zero = {
        int(row["parcel_id"])
        for row in parcels
        if row["unit"] == bad_unit
        and float(row["direct_fod_support_fraction"]) == 0
    }
    bad_low = {
        int(row["parcel_id"])
        for row in parcels
        if row["unit"] == bad_unit
        and float(row["direct_fod_support_fraction"])
        < LOW_DIRECT_SUPPORT_FRACTION
    }
    bad_zero_near = {
        int(row["parcel_id"])
        for row in parcels
        if row["unit"] == bad_unit
        and float(row["within_4mm_fod_support_fraction"]) == 0
    }
    seed_rows = tract["checks"][
        "all_four_seed_outputs_replay_independently"
    ]["evidence"]["rows"]
    connected = {
        unit: sorted(
            {
                int(row["connected_nodes"])
                for row in seed_rows
                if row["unit"] == unit
            }
        )
        for unit in (good_unit, bad_unit)
    }
    return {
        "good_lower_order_reference": {
            "unit": good_unit,
            "connected_nodes_each_seed": connected[good_unit],
            "zero_direct_fod_support_parcel_n": subjects[good_unit][
                "zero_direct_fod_support_parcel_n"
            ],
            "lt_1pct_direct_fod_support_parcel_n": subjects[
                good_unit
            ]["lt_1pct_direct_fod_support_parcel_n"],
            "zero_within_4mm_fod_support_parcel_n": subjects[
                good_unit
            ]["zero_within_4mm_fod_support_parcel_n"],
        },
        "adverse_lower_order_reference": {
            "unit": bad_unit,
            "connected_nodes_each_seed": connected[bad_unit],
            "disconnected_nodes_reproducible_through_8mm": sorted(
                disconnected
            ),
            "zero_direct_fod_support_parcels": sorted(bad_zero),
            "lt_1pct_direct_fod_support_parcels": sorted(bad_low),
            "zero_within_4mm_fod_support_parcels": sorted(
                bad_zero_near
            ),
            "zero_support_disconnected_overlap": sorted(
                bad_zero & disconnected
            ),
            "low_support_disconnected_overlap": sorted(
                bad_low & disconnected
            ),
            "zero_within_4mm_disconnected_overlap": sorted(
                bad_zero_near & disconnected
            ),
            "zero_support_overlap_fraction_of_disconnected": (
                len(bad_zero & disconnected) / len(disconnected)
            ),
            "low_support_overlap_fraction_of_disconnected": (
                len(bad_low & disconnected) / len(disconnected)
            ),
            "zero_within_4mm_overlap_fraction_of_disconnected": (
                len(bad_zero_near & disconnected)
                / len(disconnected)
            ),
        },
        "interpretation": (
            "Parcel-level support is a pre-tract risk screen, not a "
            "complete predictor of node connectivity. The adverse "
            "reference establishes mechanism relevance. Direct zero "
            "support is sensitive but not specific in the completed "
            "canaries; zero support within the 4 mm endpoint-assignment "
            "neighborhood is more specific in this mechanism reference. "
            "Neither replaces the downstream connected-node gate."
        ),
    }


def write_status(
    path: Path,
    summary: Mapping[str, Any],
) -> None:
    counts = summary["audit_status_counts"]
    ready = summary["release_eligible_support_screen"]
    mechanism = summary["mechanism_evidence"]
    lines = [
        "# HCP379 pre-tract parcel/FOD support audit",
        "",
        f"Generated: `{summary['generated_utc']}`",
        "",
        "This is a diagnosis-blind, non-overwriting pre-tract risk "
        "screen. It does not authorize exclusion or production.",
        "",
        "## Current exact-530 snapshot",
        "",
        f"- Total identities: **{summary['subject_n']}**",
        f"- Parcel support computed: **{summary['computed_subject_n']}**",
        f"- Not ready for this audit: **{counts.get('NOT_READY', 0)}**",
        f"- Audit errors: **{counts.get('AUDIT_ERROR', 0)}**",
        "",
        "## Release-eligible inputs only",
        "",
        f"- Evaluated: **{ready['evaluated_n']}**",
        f"- No zero/low parcel support flag: **{ready['pass_n']}**",
        f"- Zero direct-support review: **{ready['zero_review_n']}**",
        f"- Below-1% direct-support review: "
        f"**{ready['low_review_n']}**",
        f"- Atlas/mask hold: **{ready['atlas_hold_n']}**",
        "",
        "## Mechanism references",
        "",
        "- Good lower-order reference: "
        f"`{mechanism['good_lower_order_reference']['unit']}`; "
        f"connected nodes "
        f"{mechanism['good_lower_order_reference']['connected_nodes_each_seed']}; "
        "zero-support parcels "
        f"{mechanism['good_lower_order_reference']['zero_direct_fod_support_parcel_n']}.",
        "- Adverse lower-order reference: "
        f"`{mechanism['adverse_lower_order_reference']['unit']}`; "
        f"connected nodes "
        f"{mechanism['adverse_lower_order_reference']['connected_nodes_each_seed']}; "
        "zero-support overlap with reproducibly disconnected parcels "
        f"{len(mechanism['adverse_lower_order_reference']['zero_support_disconnected_overlap'])}/29; "
        "zero-within-4-mm overlap "
        f"{len(mechanism['adverse_lower_order_reference']['zero_within_4mm_disconnected_overlap'])}/29.",
        "",
        "The connected-node, density, two-seed stability and all-nine "
        "matrix gates remain mandatory downstream.",
        "",
    ]
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be in [1, 8]")

    specs, _ = build_specs()
    subject_results: dict[str, dict[str, Any]] = {}
    parcel_results: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(audit_one, spec): str(spec["unit"])
            for spec in specs
        }
        for future in as_completed(futures):
            unit = futures[future]
            subject, parcels = future.result()
            subject_results[unit] = subject
            parcel_results[unit] = parcels

    subjects = [
        subject_results[unit] for unit in sorted(subject_results)
    ]
    parcels = [
        row
        for unit in sorted(parcel_results)
        for row in sorted(
            parcel_results[unit],
            key=lambda value: int(value["parcel_id"]),
        )
    ]
    if len(subjects) != EXPECTED_TOTAL:
        raise ValueError("subject result count differs")

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    subject_csv = output / "subject_support_summary.csv"
    parcel_csv = output / "parcel_support.csv"
    atomic_csv(subject_csv, subjects, SUBJECT_FIELDS)
    atomic_csv(parcel_csv, parcels, PARCEL_FIELDS)

    subject_map = {str(row["unit"]): row for row in subjects}
    status_counts = dict(
        sorted(Counter(row["audit_status"] for row in subjects).items())
    )
    eligible = [
        row
        for row in subjects
        if str(row["release_input_eligible"]).lower() == "true"
    ]
    canary_rows = [
        row for row in eligible if row["role"] == "CANARY_REUSE"
    ]
    canary_densities = [
        float(row["canary_20m_density"])
        for row in canary_rows
        if row["canary_20m_density"] != ""
    ]
    canary_not_full = [
        {
            "unit": str(row["unit"]),
            "connected_nodes": int(row["canary_20m_connected_nodes"]),
            "density": float(row["canary_20m_density"]),
            "support_status": str(row["audit_status"]),
            "zero_direct_parcel_n": int(
                row["zero_direct_fod_support_parcel_n"]
            ),
            "zero_within_4mm_parcel_n": int(
                row["zero_within_4mm_fod_support_parcel_n"]
            ),
        }
        for row in canary_rows
        if int(row["canary_20m_connected_nodes"]) != EXPECTED_NODES
    ]
    canary_direct_zero_but_full = [
        str(row["unit"])
        for row in canary_rows
        if int(row["zero_direct_fod_support_parcel_n"]) > 0
        and int(row["canary_20m_connected_nodes"]) == EXPECTED_NODES
    ]
    mechanism = mechanism_evidence(subject_map, parcels)
    release_summary = {
        "evaluated_n": len(eligible),
        "pass_n": sum(
            row["audit_status"] == "PASS_SUPPORT_SCREEN"
            for row in eligible
        ),
        "zero_review_n": sum(
            row["audit_status"]
            == "REVIEW_ZERO_DIRECT_FOD_SUPPORT"
            for row in eligible
        ),
        "low_review_n": sum(
            row["audit_status"]
            == "REVIEW_LT1PCT_DIRECT_FOD_SUPPORT"
            for row in eligible
        ),
        "atlas_hold_n": sum(
            row["audit_status"] == "HOLD_ATLAS_MASK_QC"
            for row in eligible
        ),
        "error_n": sum(
            row["audit_status"] == "AUDIT_ERROR"
            for row in eligible
        ),
    }
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_pretract_parcel_fod_support_audit",
        "status": (
            "PASS_AUDIT_COMPLETE_GATE_NOT_AUTHORIZED"
            if release_summary["error_n"] == 0
            else "FAIL_AUDIT_ERRORS_PRESENT"
        ),
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "tractography_executed": False,
        "source_files_modified": False,
        "non_overwriting": True,
        "subject_exclusion_authorized": False,
        "production_gate_authorized": False,
        "subject_n": len(subjects),
        "production_n": EXPECTED_PRODUCTION,
        "canary_n": EXPECTED_CANARIES,
        "computed_subject_n": sum(
            row["audit_status"] != "NOT_READY" for row in subjects
        ),
        "parcel_row_n": len(parcels),
        "audit_status_counts": status_counts,
        "release_eligible_support_screen": release_summary,
        "canary_calibration": {
            "evaluated_n": len(canary_rows),
            "all_20m_connected_nodes_379": all(
                int(row["canary_20m_connected_nodes"])
                == EXPECTED_NODES
                for row in canary_rows
            ),
            "full_20m_connected_nodes_n": sum(
                int(row["canary_20m_connected_nodes"])
                == EXPECTED_NODES
                for row in canary_rows
            ),
            "not_full_20m_connected_nodes": canary_not_full,
            "direct_zero_but_full_379_units": (
                canary_direct_zero_but_full
            ),
            "support_status_counts": dict(
                sorted(
                    Counter(
                        row["audit_status"] for row in canary_rows
                    ).items()
                )
            ),
            "density_20m_minimum": min(canary_densities),
            "density_20m_median": float(
                np.median(canary_densities)
            ),
            "density_20m_maximum": max(canary_densities),
        },
        "mechanism_evidence": mechanism,
        "screen_contract": {
            "expected_nodes": EXPECTED_NODES,
            "direct_positive_rule": (
                "finite WM-FOD l=0 coefficient > 0"
            ),
            "low_direct_support_fraction": (
                LOW_DIRECT_SUPPORT_FRACTION
            ),
            "neighborhood_radius_mm": ASSIGNMENT_RADIUS_MM,
            "minimum_atlas_mask_fraction": (
                MINIMUM_ATLAS_MASK_FRACTION
            ),
            "zero_support_is_initial_review_not_exclusion": True,
            "downstream_connected_node_gate_still_required": True,
        },
        "records": {
            "topology": file_record(TOPOLOGY),
            "topology_validation": file_record(
                TOPOLOGY_VALIDATION
            ),
            "ledger_csv": file_record(LEDGER),
            "ledger_json": file_record(LEDGER_JSON),
            "ledger_validation": file_record(LEDGER_VALIDATION),
            "fod_compatibility": file_record(FOD_COMPATIBILITY),
            "fod_compatibility_validation": file_record(
                FOD_COMPATIBILITY_VALIDATION
            ),
            "subject_summary": file_record(subject_csv),
            "parcel_support": file_record(parcel_csv),
            "implementation": file_record(Path(__file__)),
        },
    }
    summary_path = output / "summary.json"
    atomic_json(summary_path, summary)
    write_status(output / "STATUS.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
