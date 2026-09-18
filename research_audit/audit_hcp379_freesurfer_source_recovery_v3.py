#!/home/ec2-user/fsl/bin/python
"""Attest the recovered FreeSurfer/HCP379 source without promoting old tracks.

The prior HCP build process produced the intended anatomy and atlas artifacts
but its terminal JSON write was interrupted.  This audit is read-only with
respect to imaging artifacts.  It independently verifies the recovered
surface source and exact 379-label parcellation, reports the old-track
connectome only as a failed diagnostic, and emits a separate non-overwriting
source-readiness attestation for corrected Recovery4 processing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
UNIT = "114_S_6347_I1344943"
REPAIR_ROOT = HCP_ROOT / "fastsurfer_repair" / UNIT
STANDARD_RESULT = REPAIR_ROOT / "standard_freesurfer_result.json"
STALE_BUILD_RESULT = REPAIR_ROOT / "hcp_build_result.json"
WORK = REPAIR_ROOT / "hcp"
OUTPUT = REPAIR_ROOT / "hcp_source_recovery_attestation_v3.json"
MATRIX_NAMES = (
    "count",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)
NODES = 379
POSSIBLE_EDGES = NODES * (NODES - 1) // 2
MINIMUM_FINAL_DENSITY = 0.60


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


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
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def matrix_qc(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.loadtxt(path, delimiter=",")
    failures = []
    if matrix.shape != (NODES, NODES):
        failures.append(f"shape={matrix.shape}")
    elif (
        not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, atol=1.0e-6, rtol=1.0e-6)
        or not np.allclose(np.diag(matrix), 0.0, atol=1.0e-8)
        or float(np.min(matrix)) < -1.0e-10
    ):
        failures.append("numerical_contract_failed")
    return matrix, {
        "record": file_record(path),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }


def main() -> int:
    recovery = load_json(STANDARD_RESULT)
    if (
        recovery.get("status") != "PASS_SURFACE_RECOVERY"
        or recovery.get("fsid") != UNIT
        or recovery.get("non_overwriting") is not True
    ):
        raise ValueError("standard FreeSurfer recovery is not PASS")
    subject_dir = Path(str(recovery["subject_dir"])).resolve()
    expected_surface_hashes = recovery.get("required_output_sha256")
    if not isinstance(expected_surface_hashes, Mapping):
        raise ValueError("surface recovery hashes absent")
    surface_paths = {
        "aseg": subject_dir / "mri/aseg.mgz",
        "wmparc": subject_dir / "mri/wmparc.mgz",
        "lh_sphere_reg": subject_dir / "surf/lh.sphere.reg",
        "rh_sphere_reg": subject_dir / "surf/rh.sphere.reg",
        "lh_white": subject_dir / "surf/lh.white",
        "rh_white": subject_dir / "surf/rh.white",
    }
    surface_records = {
        name: file_record(path) for name, path in surface_paths.items()
    }
    surface_hashes_match = all(
        surface_records[name]["sha256"] == expected_surface_hashes[name]
        for name in surface_paths
    )

    anatomy_paths = {
        "lh_hcpmmp1_annotation": (
            subject_dir / "label/lh.HCPMMP1.annot"
        ),
        "rh_hcpmmp1_annotation": (
            subject_dir / "label/rh.HCPMMP1.annot"
        ),
        "hcp379_source_mgz": WORK / "HCPMMP1+aseg.mgz",
        "hcp379_b0_volume": WORK / "HCPMMP1+aseg_b0.nii.gz",
        "hcp379_nodes_b0": WORK / "nodes_b0.nii.gz",
        "hcp379_node_volumes": WORK / "node_volumes.csv",
        "b0_to_freesurfer_registration": WORK / "b0_to_fs.dat",
        "exact_t1": Path(str(recovery["t1_path"])),
        "eddy_dwi": (
            Path("/data/derivatives/eddy")
            / f"{UNIT}_preproc.mif"
        ),
    }
    anatomy_records = {
        name: file_record(path) for name, path in anatomy_paths.items()
    }

    labels = np.unique(
        np.asarray(
            nib.load(str(anatomy_paths["hcp379_nodes_b0"])).dataobj
        )
    )
    rounded = {int(round(float(value))) for value in labels}
    exact_labels = rounded == set(range(NODES + 1))
    with anatomy_paths["hcp379_node_volumes"].open(
        newline="", encoding="utf-8"
    ) as handle:
        volume_rows = list(csv.DictReader(handle))
    volume_ids = {int(row["node_id"]) for row in volume_rows}
    volumes_positive = (
        len(volume_rows) == NODES
        and volume_ids == set(range(1, NODES + 1))
        and all(
            math.isfinite(float(row["volume_mm3"]))
            and float(row["volume_mm3"]) > 0
            and int(row["n_voxels"]) > 0
            for row in volume_rows
        )
    )

    matrix_records: dict[str, Any] = {}
    matrices: dict[str, np.ndarray] = {}
    for name in MATRIX_NAMES:
        path = (
            HCP_ROOT
            / "connectomes"
            / f"SC_HCPMMP1_{UNIT}_{name}.csv"
        )
        matrix, qc = matrix_qc(path)
        matrices[name] = matrix
        matrix_records[name] = qc
    count = matrices["count"]
    supported_edges = int(np.count_nonzero(np.triu(count > 0, 1)))
    density = supported_edges / POSSIBLE_EDGES
    connected_nodes = int(np.count_nonzero(np.sum(count, axis=1) > 0))
    old_track_failures = []
    if density < MINIMUM_FINAL_DENSITY:
        old_track_failures.append(
            f"density={density:.12g}<{MINIMUM_FINAL_DENSITY}"
        )
    if connected_nodes < 360:
        old_track_failures.append(f"connected_nodes={connected_nodes}<360")
    old_track_failures.extend(
        f"{name}:{failure}"
        for name, qc in matrix_records.items()
        for failure in qc["failures"]
    )

    stale = load_json(STALE_BUILD_RESULT)
    source_ready = (
        surface_hashes_match
        and exact_labels
        and volumes_positive
        and len(anatomy_records) == len(anatomy_paths)
    )
    report = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_freesurfer_source_"
            "recovery_attestation"
        ),
        "status": (
            "PASS_HCP_SOURCE_READY_FOR_CORRECTED_ROUTE"
            if source_ready
            else "FAIL_HCP_SOURCE_RECOVERY"
        ),
        "unit": UNIT,
        "diagnosis_or_outcome_fields_used": False,
        "non_overwriting": True,
        "imaging_executed_by_audit": False,
        "surface_recovery_hashes_match": surface_hashes_match,
        "atlas_nodes_expected": NODES,
        "atlas_nodes_present": len(rounded - {0}),
        "exact_label_set_0_through_379": exact_labels,
        "node_volume_rows": len(volume_rows),
        "node_volumes_positive": volumes_positive,
        "old_existing_track_is_final_release_candidate": False,
        "old_existing_track_status": (
            "PASS"
            if not old_track_failures
            else "FAIL_CORRECTED_TRACTOGRAPHY_REQUIRED"
        ),
        "old_existing_track_qc": {
            "supported_edges": supported_edges,
            "possible_edges": POSSIBLE_EDGES,
            "edge_density": density,
            "connected_nodes": connected_nodes,
            "minimum_final_density_inclusive": MINIMUM_FINAL_DENSITY,
            "failures": old_track_failures,
            "matrices": matrix_records,
        },
        "stale_terminal_state_preserved": {
            "status": stale.get("status"),
            "record": file_record(STALE_BUILD_RESULT),
            "interrupted_temporary_files": sorted(
                str(path)
                for path in REPAIR_ROOT.glob(
                    ".hcp_build_result.json.*.tmp"
                )
            ),
        },
        "records": {
            "standard_freesurfer_result": file_record(STANDARD_RESULT),
            "surface_outputs": surface_records,
            "recovered_hcp_source": anatomy_records,
            "audit_source": file_record(Path(__file__)),
        },
        "next_action": (
            "Use the recovered HCP379 source and validated Eddy derivative "
            "in the non-overwriting corrected Recovery4 pretract route; "
            "never promote the sparse old-track matrices."
        ),
    }
    atomic_json(OUTPUT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(not source_ready)


if __name__ == "__main__":
    raise SystemExit(main())
