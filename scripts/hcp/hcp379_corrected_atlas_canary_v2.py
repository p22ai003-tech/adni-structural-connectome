#!/usr/bin/env python3
"""Build diagnosis-blind HCP379 atlases in the corrected Recovery3 DWI space.

This is a bounded, non-overwriting pre-tractography canary.  It reuses each
unit's HCP-MMP1+aseg parcellation only after verifying that FastSurfer used the
same T1 image locked in the corrected 15-unit execution manifest.  Labels are
mapped through the corrected Recovery3 T1-to-b0 affine onto the 1-mm b0 world
grid, relabelled to contiguous 1..379, and subjected to geometry, label-support,
brain-overlap, and left/right-balance checks.

No diagnosis or outcome field is read or written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np


EXP = Path("/home/ec2-user/exp")
RUN_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4")
HCP_V2_ROOT = Path("/data/derivatives/hcp379_v2")
OUTPUT_ROOT = HCP_V2_ROOT / "corrected_atlas_canary"
RECOVERY3_BINDING = (
    EXP
    / "research_audit/outputs/"
    "h04a_r1_retry4_pretract_recovery3_package_v1/"
    "recovery3_execution_binding.json"
)
EXECUTION_MANIFEST = (
    EXP
    / "research_audit/outputs/"
    "h04a_r1_recovery_package_v1/recovery_execution_manifest_v1.csv"
)
HCP_SOURCE_ROOT = Path("/data/derivatives/parc_hcpmmp1")
FASTSURFER_ROOT = Path("/data/derivatives/fastsurfer")
FREESURFER = Path("/home/ec2-user/freesurfer")
FSL = Path("/home/ec2-user/fsl/bin")
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
RELABEL = EXP / "scripts/hcp/relabel_hcp.py"
RELABEL_LUT = HCP_SOURCE_ROOT / "hcpmmp1_subcort_relabel.txt"
FSL_PYTHON = Path("/home/ec2-user/fsl/bin/python")
EXPECTED_NODES = 379
MINIMUM_ATLAS_INSIDE_DWI_MASK = 0.80
MINIMUM_VOXELS_PER_NODE = 1
MINIMUM_NODE_VOLUME_RATIO = 0.10
MAXIMUM_NODE_VOLUME_RATIO = 10.0
CENTRAL_NODE_VOLUME_RATIO_RANGE = (0.50, 2.0)
MINIMUM_CENTRAL_NODE_VOLUME_FRACTION = 0.95
TOTAL_ATLAS_VOLUME_RATIO_RANGE = (0.75, 1.25)
MEDIAN_NODE_VOLUME_RATIO_RANGE = (0.75, 1.25)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run(command: list[str], log: Any, *, env: Mapping[str, str] | None = None) -> None:
    log.write(f"\n[{utc_now()}] COMMAND {json.dumps(command)}\n")
    log.flush()
    completed = subprocess.run(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(env) if env is not None else None,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {command[0]}"
        )


def unit_from_row(row: Mapping[str, str]) -> str:
    return f"{row['subject_id']}_I{row['dti_image_id']}"


def execution_t1_by_unit() -> dict[str, dict[str, str]]:
    with EXECUTION_MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {
        unit_from_row(row): {
            "t1_image_id": str(row["t1_image_id"]),
            "t1_source_id": str(row["t1_source_id"]),
            "t1_source_path": str(row["t1_source_path"]),
        }
        for row in rows
    }


def fastsurfer_t1_path(unit: str) -> Path:
    logs = (
        FASTSURFER_ROOT / unit / "scripts/deep-seg.log",
        FASTSURFER_ROOT / unit / "scripts/exectime.log",
    )
    for path in logs:
        if not path.is_file():
            continue
        match = re.search(
            r"--t1\s+(\S+corrected_T1_\S+?\.nii(?:\.gz)?)",
            path.read_text(encoding="utf-8", errors="replace"),
        )
        if match:
            return Path(match.group(1)).expanduser().resolve()
    raise ValueError(f"unable to recover FastSurfer T1 input for {unit}")


def t1_image_id(path: Path) -> str | None:
    match = re.search(r"_I(\d+)\.nii(?:\.gz)?$", path.name)
    return match.group(1) if match else None


def load_units() -> list[str]:
    binding = json.loads(RECOVERY3_BINDING.read_text(encoding="utf-8"))
    units = binding.get("units")
    if (
        binding.get("record_type")
        != "retry4_pretract_recovery3_execution_binding"
        or binding.get("diagnosis_labels_used") is not False
        or not isinstance(units, list)
        or len(units) != 15
        or len(units) != len(set(units))
    ):
        raise ValueError("Recovery3 canary binding differs")
    return sorted(str(unit) for unit in units)


def atlas_qc(
    *,
    unit: str,
    nodes_path: Path,
    reference_path: Path,
    mask_path: Path,
    node_volumes_path: Path,
    source_node_volumes_path: Path,
) -> dict[str, Any]:
    nodes_image = nib.load(str(nodes_path))
    reference_image = nib.load(str(reference_path))
    mask_image = nib.load(str(mask_path))
    nodes_raw = np.asanyarray(nodes_image.dataobj)
    nodes = np.rint(nodes_raw).astype(np.int32)
    mask = np.asanyarray(mask_image.dataobj) > 0
    failures: list[str] = []

    if not np.allclose(nodes_raw, nodes, atol=1.0e-6):
        failures.append("atlas_contains_noninteger_values")
    if nodes_image.shape != reference_image.shape:
        failures.append(
            f"atlas_reference_shape_mismatch:{nodes_image.shape}:{reference_image.shape}"
        )
    if not np.allclose(nodes_image.affine, reference_image.affine, atol=1.0e-5):
        failures.append("atlas_reference_affine_mismatch")
    if mask_image.shape != reference_image.shape:
        failures.append(
            f"mask_reference_shape_mismatch:{mask_image.shape}:{reference_image.shape}"
        )
    if not np.allclose(mask_image.affine, reference_image.affine, atol=1.0e-5):
        failures.append("mask_reference_affine_mismatch")

    expected = list(range(1, EXPECTED_NODES + 1))
    labels = sorted(int(value) for value in np.unique(nodes) if value > 0)
    if labels != expected:
        missing = sorted(set(expected) - set(labels))
        unexpected = sorted(set(labels) - set(expected))
        failures.append(
            f"label_set_differs:found={len(labels)}:"
            f"missing={missing}:unexpected={unexpected}"
        )
    counts = {
        node: int(np.count_nonzero(nodes == node)) for node in expected
    }
    below_minimum = [
        node for node, count in counts.items() if count < MINIMUM_VOXELS_PER_NODE
    ]
    if below_minimum:
        failures.append(f"nodes_below_voxel_minimum:{below_minimum}")

    atlas_mask = nodes > 0
    atlas_voxels = int(np.count_nonzero(atlas_mask))
    inside = (
        float(np.count_nonzero(atlas_mask & mask)) / atlas_voxels
        if atlas_voxels
        else 0.0
    )
    if inside < MINIMUM_ATLAS_INSIDE_DWI_MASK:
        failures.append(
            f"atlas_inside_dwi_mask={inside:.6f}<"
            f"{MINIMUM_ATLAS_INSIDE_DWI_MASK:.6f}"
        )

    left = sum(counts[node] for node in range(1, 181))
    right = sum(counts[node] for node in range(181, 361))
    cortical_lr = left / max(right, 1)
    if not 0.60 <= cortical_lr <= 1.67:
        failures.append(f"cortical_lr_ratio={cortical_lr:.6f}")
    pair_ratios: dict[str, float] = {}
    for name, left_node, right_node in (
        ("thalamus", 361, 370),
        ("hippocampus", 365, 374),
        ("putamen", 363, 372),
    ):
        ratio = counts[left_node] / max(counts[right_node], 1)
        pair_ratios[name] = ratio
        if not 0.35 <= ratio <= 2.85:
            failures.append(f"{name}_lr_ratio={ratio:.6f}")

    with node_volumes_path.open(newline="", encoding="utf-8") as handle:
        volume_rows = list(csv.DictReader(handle))
    volume_nodes = [int(row["node_id"]) for row in volume_rows]
    if volume_nodes != expected:
        failures.append("node_volume_row_set_differs")
    else:
        for row in volume_rows:
            node = int(row["node_id"])
            if int(row["n_voxels"]) != counts[node]:
                failures.append(f"node_volume_count_differs:{node}")
                break

    with source_node_volumes_path.open(
        newline="", encoding="utf-8"
    ) as handle:
        source_volume_rows = list(csv.DictReader(handle))
    source_volume_nodes = [
        int(row["node_id"]) for row in source_volume_rows
    ]
    node_volume_ratios: np.ndarray
    total_atlas_volume_ratio = 0.0
    median_node_volume_ratio = 0.0
    central_node_volume_fraction = 0.0
    if volume_nodes != expected or source_volume_nodes != expected:
        node_volume_ratios = np.asarray([], dtype=float)
        failures.append("source_or_mapped_node_volume_row_set_differs")
    else:
        mapped_volumes = np.asarray(
            [float(row["volume_mm3"]) for row in volume_rows], dtype=float
        )
        source_volumes = np.asarray(
            [float(row["volume_mm3"]) for row in source_volume_rows],
            dtype=float,
        )
        if (
            not np.isfinite(mapped_volumes).all()
            or not np.isfinite(source_volumes).all()
            or np.any(mapped_volumes <= 0)
            or np.any(source_volumes <= 0)
        ):
            node_volume_ratios = np.asarray([], dtype=float)
            failures.append("source_or_mapped_node_volumes_nonpositive")
        else:
            node_volume_ratios = mapped_volumes / source_volumes
            total_atlas_volume_ratio = float(
                mapped_volumes.sum() / source_volumes.sum()
            )
            median_node_volume_ratio = float(
                np.median(node_volume_ratios)
            )
            central_lower, central_upper = (
                CENTRAL_NODE_VOLUME_RATIO_RANGE
            )
            central_node_volume_fraction = float(
                np.mean(
                    (node_volume_ratios >= central_lower)
                    & (node_volume_ratios <= central_upper)
                )
            )
            extreme_nodes = [
                int(index + 1)
                for index, ratio in enumerate(node_volume_ratios)
                if (
                    ratio < MINIMUM_NODE_VOLUME_RATIO
                    or ratio > MAXIMUM_NODE_VOLUME_RATIO
                )
            ]
            if extreme_nodes:
                failures.append(
                    "node_volume_ratio_extreme:"
                    f"{extreme_nodes}"
                )
            if (
                central_node_volume_fraction
                < MINIMUM_CENTRAL_NODE_VOLUME_FRACTION
            ):
                failures.append(
                    "central_node_volume_fraction="
                    f"{central_node_volume_fraction:.6f}<"
                    f"{MINIMUM_CENTRAL_NODE_VOLUME_FRACTION:.6f}"
                )
            total_lower, total_upper = TOTAL_ATLAS_VOLUME_RATIO_RANGE
            if not total_lower <= total_atlas_volume_ratio <= total_upper:
                failures.append(
                    "total_atlas_volume_ratio="
                    f"{total_atlas_volume_ratio:.6f}"
                )
            median_lower, median_upper = MEDIAN_NODE_VOLUME_RATIO_RANGE
            if not median_lower <= median_node_volume_ratio <= median_upper:
                failures.append(
                    "median_node_volume_ratio="
                    f"{median_node_volume_ratio:.6f}"
                )

    voxel_volume = abs(float(np.linalg.det(nodes_image.affine[:3, :3])))
    return {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_corrected_atlas_qc",
        "status": "PASS" if not failures else "FAIL",
        "unit": unit,
        "diagnosis_labels_used": False,
        "expected_nodes": EXPECTED_NODES,
        "labels_found": len(labels),
        "missing_labels": sorted(set(expected) - set(labels)),
        "unexpected_labels": sorted(set(labels) - set(expected)),
        "minimum_voxels_per_node": min(counts.values()),
        "maximum_voxels_per_node": max(counts.values()),
        "minimum_required_voxels_per_node": MINIMUM_VOXELS_PER_NODE,
        "voxel_volume_mm3": voxel_volume,
        "atlas_voxels": atlas_voxels,
        "atlas_inside_dwi_mask_fraction": inside,
        "minimum_atlas_inside_dwi_mask_fraction": (
            MINIMUM_ATLAS_INSIDE_DWI_MASK
        ),
        "cortical_left_right_voxel_ratio": cortical_lr,
        "subcortical_pair_left_right_ratios": pair_ratios,
        "node_volume_preservation": {
            "source_node_volumes": str(source_node_volumes_path.resolve()),
            "minimum_ratio": (
                float(node_volume_ratios.min())
                if node_volume_ratios.size
                else None
            ),
            "maximum_ratio": (
                float(node_volume_ratios.max())
                if node_volume_ratios.size
                else None
            ),
            "median_ratio": median_node_volume_ratio,
            "total_atlas_volume_ratio": total_atlas_volume_ratio,
            "fraction_within_0_5_to_2_0": central_node_volume_fraction,
            "required_individual_ratio_range": [
                MINIMUM_NODE_VOLUME_RATIO,
                MAXIMUM_NODE_VOLUME_RATIO,
            ],
            "required_central_ratio_range": list(
                CENTRAL_NODE_VOLUME_RATIO_RANGE
            ),
            "minimum_central_fraction": (
                MINIMUM_CENTRAL_NODE_VOLUME_FRACTION
            ),
            "required_total_volume_ratio_range": list(
                TOTAL_ATLAS_VOLUME_RATIO_RANGE
            ),
            "required_median_volume_ratio_range": list(
                MEDIAN_NODE_VOLUME_RATIO_RANGE
            ),
        },
        "geometry": {
            "shape": list(nodes_image.shape),
            "reference_shape": list(reference_image.shape),
            "affine_max_absolute_difference": float(
                np.max(np.abs(nodes_image.affine - reference_image.affine))
            ),
        },
        "failures": failures,
    }


def existing_pass(result_path: Path) -> dict[str, Any] | None:
    if not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "PASS":
            return None
        for record in result.get("artifacts", {}).values():
            path = Path(str(record["path"]))
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != int(record["size_bytes"])
                or sha256_file(path) != record["sha256"]
            ):
                return None
        return result
    except Exception:
        return None


def build_unit(
    unit: str,
    *,
    expected_t1: Mapping[str, str],
    output_root: Path,
) -> dict[str, Any]:
    subject_output = output_root / "subjects" / unit
    result_path = subject_output / "result.json"
    cached = existing_pass(result_path)
    if cached is not None:
        return cached

    subject_output.mkdir(parents=True, exist_ok=True)
    log_path = subject_output / "build.log"
    hcp_mgz = HCP_SOURCE_ROOT / unit / "HCPMMP1+aseg.mgz"
    fastsurfer_t1 = fastsurfer_t1_path(unit)
    corrected_t1 = RUN_ROOT / "subjects" / unit / "02_anat/t1_n4.nii.gz"
    b0 = RUN_ROOT / "subjects" / unit / "03_spatial/mean_b0.nii.gz"
    b0_1mm = (
        RUN_ROOT / "subjects" / unit / "04_atlas/b0_1mm_world_grid.nii.gz"
    )
    t1_to_b0 = (
        RUN_ROOT / "subjects" / unit / "03_spatial/t1_to_b0_bbr.mat"
    )
    dwi_mask_1mm = (
        RUN_ROOT
        / "subjects"
        / unit
        / "06_preflight/dwi_mask_1mm_recovery3.nii.gz"
    )
    hcp_t1 = subject_output / "HCPMMP1+aseg_t1_native.nii.gz"
    nodes_t1 = subject_output / "hcp379_nodes_t1_native.nii.gz"
    node_volumes_t1 = (
        subject_output / "hcp379_node_volumes_t1_native.csv"
    )
    hcp_b0_raw = subject_output / "HCPMMP1+aseg_b0_1mm_raw.nii.gz"
    nodes = subject_output / "hcp379_nodes_b0_1mm.nii.gz"
    node_volumes = subject_output / "hcp379_node_volumes.csv"
    nodes_native = subject_output / "hcp379_nodes_b0_native_qc.nii.gz"
    overlay = subject_output / "review_b0_vs_hcp379.png"
    qc_path = subject_output / "atlas_qc.json"

    required = (
        hcp_mgz,
        fastsurfer_t1,
        corrected_t1,
        b0,
        b0_1mm,
        t1_to_b0,
        dwi_mask_1mm,
        RELABEL,
        RELABEL_LUT,
        FSL_PYTHON,
        FREESURFER / "bin/mri_vol2vol",
        FSL / "flirt",
        FSL / "slices",
        MRTRIX / "mrtransform",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing corrected-atlas inputs: " + ",".join(missing))

    observed_t1_id = t1_image_id(fastsurfer_t1)
    if observed_t1_id != expected_t1["t1_image_id"]:
        raise ValueError(
            f"FastSurfer T1 differs for {unit}: "
            f"{observed_t1_id} != {expected_t1['t1_image_id']}"
        )

    state: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_corrected_atlas_build",
        "status": "RUNNING",
        "unit": unit,
        "diagnosis_labels_used": False,
        "started_utc": utc_now(),
        "non_overwriting": True,
    }
    atomic_json(result_path, state)
    started = time.monotonic()
    env = os.environ.copy()
    env.update(
        {
            "FREESURFER_HOME": str(FREESURFER),
            "FS_LICENSE": str(FREESURFER / "license.txt"),
            "SUBJECTS_DIR": str(FASTSURFER_ROOT),
            "FSLDIR": str(FSL.parent),
            "FSLOUTPUTTYPE": "NIFTI_GZ",
            "PATH": (
                f"{FREESURFER / 'bin'}:{FSL}:{MRTRIX}:"
                f"{env.get('PATH', '')}"
            ),
        }
    )
    try:
        with log_path.open("a", encoding="utf-8") as log:
            run(
                [
                    str(FREESURFER / "bin/mri_vol2vol"),
                    "--mov",
                    str(hcp_mgz),
                    "--targ",
                    str(corrected_t1),
                    "--regheader",
                    "--interp",
                    "nearest",
                    "--keep-precision",
                    "--o",
                    str(hcp_t1),
                ],
                log,
                env=env,
            )
            run(
                [
                    str(FSL_PYTHON),
                    str(RELABEL),
                    str(hcp_t1),
                    str(RELABEL_LUT),
                    str(nodes_t1),
                    str(node_volumes_t1),
                ],
                log,
                env=env,
            )
            run(
                [
                    str(FSL / "flirt"),
                    "-in",
                    str(hcp_t1),
                    "-ref",
                    str(b0_1mm),
                    "-applyxfm",
                    "-init",
                    str(t1_to_b0),
                    "-interp",
                    "nearestneighbour",
                    "-datatype",
                    "int",
                    "-out",
                    str(hcp_b0_raw),
                ],
                log,
                env=env,
            )
            run(
                [
                    str(FSL_PYTHON),
                    str(RELABEL),
                    str(hcp_b0_raw),
                    str(RELABEL_LUT),
                    str(nodes),
                    str(node_volumes),
                ],
                log,
                env=env,
            )

        qc = atlas_qc(
            unit=unit,
            nodes_path=nodes,
            reference_path=b0_1mm,
            mask_path=dwi_mask_1mm,
            node_volumes_path=node_volumes,
            source_node_volumes_path=node_volumes_t1,
        )
        atomic_json(qc_path, qc)
        with log_path.open("a", encoding="utf-8") as log:
            run(
                [
                    str(MRTRIX / "mrtransform"),
                    str(nodes),
                    str(nodes_native),
                    "-template",
                    str(b0),
                    "-interp",
                    "nearest",
                    "-force",
                ],
                log,
                env=env,
            )
            run(
                [
                    str(FSL / "slices"),
                    str(b0),
                    str(nodes_native),
                    "-o",
                    str(overlay),
                ],
                log,
                env=env,
            )

        artifacts = {
            "hcp_source_parcellation": file_record(hcp_mgz),
            "fastsurfer_t1": file_record(fastsurfer_t1),
            "corrected_t1": file_record(corrected_t1),
            "corrected_t1_to_b0_affine": file_record(t1_to_b0),
            "corrected_b0_1mm_reference": file_record(b0_1mm),
            "corrected_dwi_mask_1mm": file_record(dwi_mask_1mm),
            "hcp_t1_native": file_record(hcp_t1),
            "hcp379_nodes_t1_native": file_record(nodes_t1),
            "hcp379_node_volumes_t1_native": file_record(
                node_volumes_t1
            ),
            "hcp_b0_raw": file_record(hcp_b0_raw),
            "hcp379_nodes_b0_1mm": file_record(nodes),
            "hcp379_node_volumes": file_record(node_volumes),
            "hcp379_atlas_qc": file_record(qc_path),
            "hcp379_nodes_b0_native_qc": file_record(nodes_native),
            "hcp379_visual_overlay": file_record(overlay),
        }
        state.update(
            {
                "status": qc["status"],
                "completed_utc": utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "t1_identity": {
                    "execution_manifest_t1_image_id": expected_t1[
                        "t1_image_id"
                    ],
                    "fastsurfer_t1_image_id": observed_t1_id,
                    "match": True,
                },
                "atlas_qc": qc,
                "artifacts": artifacts,
                "failure": None if qc["status"] == "PASS" else "atlas_qc_failed",
            }
        )
    except Exception as exc:
        state.update(
            {
                "status": "FAIL",
                "completed_utc": utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "failure": f"{type(exc).__name__}:{exc}",
            }
        )
    atomic_json(result_path, state)
    return state


def write_summary(output_root: Path, results: list[dict[str, Any]]) -> Path:
    status_counts = {
        status: sum(row.get("status") == status for row in results)
        for status in sorted({str(row.get("status")) for row in results})
    }
    summary = {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_corrected_atlas_canary_summary",
        "status": (
            "PASS"
            if results
            and len(results) == 15
            and all(row.get("status") == "PASS" for row in results)
            else "FAIL"
        ),
        "updated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "unit_count": len(results),
        "passed_unit_count": sum(
            row.get("status") == "PASS" for row in results
        ),
        "status_counts": status_counts,
        "units": [
            {
                "unit": row["unit"],
                "status": row.get("status"),
                "labels_found": row.get("atlas_qc", {}).get("labels_found"),
                "minimum_voxels_per_node": row.get("atlas_qc", {}).get(
                    "minimum_voxels_per_node"
                ),
                "atlas_inside_dwi_mask_fraction": row.get(
                    "atlas_qc", {}
                ).get("atlas_inside_dwi_mask_fraction"),
                "cortical_left_right_voxel_ratio": row.get(
                    "atlas_qc", {}
                ).get("cortical_left_right_voxel_ratio"),
                "failure": row.get("failure"),
                "result": file_record(
                    output_root / "subjects" / row["unit"] / "result.json"
                ),
            }
            for row in sorted(results, key=lambda item: item["unit"])
        ],
    }
    path = output_root / "corrected_atlas_canary_summary.json"
    atomic_json(path, summary)
    csv_path = output_root / "corrected_atlas_canary_summary.csv"
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=csv_path.parent,
        prefix=f".{csv_path.name}.",
        suffix=".tmp",
        delete=False,
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "unit",
                "status",
                "labels_found",
                "minimum_voxels_per_node",
                "atlas_inside_dwi_mask_fraction",
                "cortical_left_right_voxel_ratio",
                "failure",
            ),
        )
        writer.writeheader()
        for row in summary["units"]:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
        temporary = Path(handle.name)
    os.replace(temporary, csv_path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be in 1..8")
    units = load_units()
    t1_by_unit = execution_t1_by_unit()
    if not set(units).issubset(t1_by_unit):
        raise ValueError("execution manifest lacks one or more bound units")
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[{utc_now()}] HCP379_CORRECTED_ATLAS_CANARY_START "
        f"n={len(units)} workers={args.workers}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                build_unit,
                unit,
                expected_t1=t1_by_unit[unit],
                output_root=args.output_root,
            ): unit
            for unit in units
        }
        for index, future in enumerate(as_completed(futures), start=1):
            unit = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "schema_version": "1.0.0",
                    "record_type": (
                        "diagnosis_blind_hcp379_corrected_atlas_build"
                    ),
                    "status": "FAIL",
                    "unit": unit,
                    "diagnosis_labels_used": False,
                    "completed_utc": utc_now(),
                    "failure": f"{type(exc).__name__}:{exc}",
                }
                atomic_json(
                    args.output_root / "subjects" / unit / "result.json",
                    result,
                )
            results.append(result)
            print(
                f"[{utc_now()}] atlas {index}/{len(units)} {unit} "
                f"{result.get('status')} "
                f"labels={result.get('atlas_qc', {}).get('labels_found')} "
                f"inside={result.get('atlas_qc', {}).get('atlas_inside_dwi_mask_fraction')} "
                f"failure={result.get('failure')}",
                flush=True,
            )
    summary_path = write_summary(args.output_root, results)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "passed": summary["passed_unit_count"],
                "total": summary["unit_count"],
                "summary": str(summary_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
