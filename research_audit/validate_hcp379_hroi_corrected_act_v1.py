#!/home/ec2-user/fsl/bin/python
"""Validate the H-ROI surface-support candidate on corrected ACT tracks.

This is a non-overwriting, diagnosis-blind policy test.  It reuses the exact
completed Phase-B tractogram and its original-atlas count assignment, rebuilds
only the candidate HCP379 atlas through the same selected registration, and
compares endpoint assignment and topology after adding bilateral H-ROI support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np


DEFAULT_UNIT = "135_S_6840_I1263427"
EXPECTED_STREAMLINES = 10_000_000
POSSIBLE_EDGES = 379 * 378 // 2
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
CANDIDATE_ROOT = (
    HCP_ROOT / "source_label_repair_v1/hroi_surface_support_v1"
)
OUTPUT_ROOT = (
    HCP_ROOT
    / "source_label_repair_v1/validation/"
    "hroi_surface_support_corrected_act_v1/attempts"
)
PRETRACT_ROOT = HCP_ROOT / "pretract_recovery4/subjects"
ORIGINAL_ATLAS_ROOT = HCP_ROOT / "corrected_atlas_canary_recovery4/subjects"
FREESURFER_ROOT = Path("/data/derivatives/fastsurfer")
FREESURFER = Path("/home/ec2-user/freesurfer")
FSL = Path("/home/ec2-user/fsl/bin")
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
FSL_PYTHON = FSL / "python"
RELABEL = Path("/home/ec2-user/exp/scripts/hcp/relabel_hcp.py")
RELABEL_LUT = (
    Path("/data/derivatives/parc_hcpmmp1")
    / "hcpmmp1_subcort_relabel.txt"
)
TCK_HEADER_COUNT = (
    Path("/home/ec2-user/exp/scripts/hcp/tck_header_count_v1.py")
)
H_NODES = (120, 300)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, hash_content: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular file: {resolved}")
    record: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if hash_content:
        record["sha256"] = sha256_file(resolved)
    return record


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def verified_path(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(record.get("size_bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise ValueError(f"{label}: bound artifact differs")
    return path


def run_logged(
    command: list[str],
    *,
    log: Path,
    environment: Mapping[str, str],
    outputs: tuple[Path, ...],
) -> None:
    with log.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(environment),
        )
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {command[0]}"
        )
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError("command outputs missing: " + ",".join(missing))


def integer_data(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    integer = np.asarray(data, dtype=np.int32)
    if data.ndim != 3 or not np.array_equal(data, integer):
        raise ValueError(f"not an integer 3D label image: {path}")
    return image, integer


def assignment_summary(path: Path) -> dict[str, Any]:
    row_count = 0
    assigned_count = 0
    endpoint_min = 379
    endpoint_max = 0
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.replace(",", " ").split()
            if len(fields) != 2:
                malformed += 1
                continue
            left, right = (int(value) for value in fields)
            row_count += 1
            endpoint_min = min(endpoint_min, left, right)
            endpoint_max = max(endpoint_max, left, right)
            assigned_count += int(left > 0 and right > 0)
    return {
        "row_count": row_count,
        "assigned_streamline_count": assigned_count,
        "endpoint_assignment_fraction": (
            assigned_count / row_count if row_count else 0.0
        ),
        "minimum_endpoint": endpoint_min if row_count else None,
        "maximum_endpoint": endpoint_max if row_count else None,
        "malformed_row_count": malformed,
    }


def node_names(unit: str) -> dict[int, str]:
    names: dict[int, str] = {}
    for hemi, offset in (("lh", 0), ("rh", 180)):
        annotation = FREESURFER_ROOT / unit / "label" / f"{hemi}.HCPMMP1.annot"
        _, _, raw_names = fsio.read_annot(str(annotation))
        for index in range(1, 181):
            names[offset + index] = raw_names[index].decode("utf-8")
    source_to_node: dict[int, int] = {}
    for line in RELABEL_LUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            source, node = (int(value) for value in line.split())
            source_to_node[source] = node
    lut = FREESURFER / "FreeSurferColorLUT.txt"
    for line in lut.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].isdigit():
            source = int(fields[0])
            node = source_to_node.get(source)
            if node is not None and node >= 361:
                names[node] = fields[1]
    if set(names) != set(range(1, 380)):
        missing = sorted(set(range(1, 380)) - set(names))
        raise ValueError(f"node-name mapping is incomplete: {missing}")
    return names


def matrix_contract(matrix: np.ndarray) -> bool:
    return bool(
        matrix.shape == (379, 379)
        and np.isfinite(matrix).all()
        and (matrix >= 0).all()
        and np.allclose(matrix, matrix.T)
        and np.count_nonzero(np.diag(matrix)) == 0
    )


def top_neighbours(
    matrix: np.ndarray, node: int, names: Mapping[int, str], limit: int = 15
) -> list[dict[str, Any]]:
    row = matrix[node - 1]
    order = np.argsort(row)[::-1]
    return [
        {
            "node": int(index + 1),
            "name": names[int(index + 1)],
            "streamline_count": float(row[index]),
        }
        for index in order
        if index != node - 1 and row[index] > 0
    ][:limit]


def validate(unit: str, threads: int, attempt: Path) -> dict[str, Any]:
    output = attempt / unit
    output.mkdir(parents=True, exist_ok=False)
    candidate_manifest_path = CANDIDATE_ROOT / unit / "candidate_manifest.json"
    pretract_manifest_path = PRETRACT_ROOT / unit / "pretract_recovery4_manifest.json"
    original_result_path = ORIGINAL_ATLAS_ROOT / unit / "result.json"
    candidate_manifest = load_json(candidate_manifest_path)
    pretract_manifest = load_json(pretract_manifest_path)
    original_result = load_json(original_result_path)
    if (
        candidate_manifest.get("status") != "PASS_ALL_379_SOURCE_LABELS"
        or candidate_manifest.get("diagnosis_labels_used") is not False
        or candidate_manifest.get("unit") != unit
    ):
        raise ValueError("candidate manifest differs")
    if (
        pretract_manifest.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or pretract_manifest.get("diagnosis_labels_used") is not False
        or pretract_manifest.get("registration", {}).get("route")
        != "recovery3_fsl_bbr_primary"
    ):
        raise ValueError("pretract manifest or registration route differs")
    if (
        original_result.get("status") != "PASS"
        or original_result.get("diagnosis_labels_used") is not False
    ):
        raise ValueError("original corrected-atlas result differs")

    candidate_source = verified_path(
        candidate_manifest["candidate"], "candidate source"
    )
    corrected_t1 = verified_path(
        original_result["artifacts"]["corrected_t1"],
        "corrected T1 target",
    )
    b0_1mm = verified_path(
        original_result["artifacts"]["corrected_b0_1mm_reference"],
        "corrected b0 reference",
    )
    transform = verified_path(
        pretract_manifest["registration"]["transform"],
        "selected T1-to-b0 transform",
    )
    original_nodes = verified_path(
        pretract_manifest["hcp379"]["nodes_b0_1mm"],
        "original corrected nodes",
    )
    tracks = (
        RUN_ROOT
        / "subjects"
        / unit
        / "09_hcp379_stability/primary_10m/tracks.tck"
    )
    track_metadata = tracks.parent / "tractography_parameters.json"
    original_count = tracks.parent / "matrices/count.csv"
    original_assignments = tracks.parent / "assignments.csv"
    metadata = load_json(track_metadata)
    if (
        metadata.get("act") is not True
        or metadata.get("actual_streamlines") != EXPECTED_STREAMLINES
        or metadata.get("diagnosis_labels_used") is not False
    ):
        raise ValueError("corrected ACT tractography metadata differs")
    count_stdout = subprocess.run(
        [
            str(FSL_PYTHON),
            str(TCK_HEADER_COUNT),
            str(tracks),
            "--expect",
            str(EXPECTED_STREAMLINES),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if int(count_stdout) != EXPECTED_STREAMLINES:
        raise ValueError("tractogram header count differs")
    for path in (
        corrected_t1,
        tracks,
        original_count,
        original_assignments,
        RELABEL,
        RELABEL_LUT,
        FREESURFER / "bin/mri_vol2vol",
        FSL / "flirt",
        MRTRIX / "tck2connectome",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    environment = os.environ.copy()
    environment.update(
        {
            "FREESURFER_HOME": str(FREESURFER),
            "FS_LICENSE": str(FREESURFER / "license.txt"),
            "SUBJECTS_DIR": str(FREESURFER_ROOT),
            "FSLDIR": str(FSL.parent),
            "FSLOUTPUTTYPE": "NIFTI_GZ",
            "PATH": (
                f"{FREESURFER / 'bin'}:{FSL}:{MRTRIX}:"
                f"{environment.get('PATH', '')}"
            ),
        }
    )
    candidate_t1_raw = output / "HCPMMP1+aseg_hroi_t1_native.nii.gz"
    candidate_t1_nodes = output / "hcp379_hroi_nodes_t1_native.nii.gz"
    candidate_t1_volumes = output / "hcp379_hroi_node_volumes_t1_native.csv"
    candidate_b0_raw = output / "HCPMMP1+aseg_hroi_b0_1mm_raw.nii.gz"
    candidate_nodes = output / "hcp379_hroi_nodes_b0_1mm.nii.gz"
    candidate_volumes = output / "hcp379_hroi_node_volumes_b0_1mm.csv"
    candidate_count = output / "count_hroi_surface_support.csv"
    candidate_assignments = output / "assignments_hroi_surface_support.csv"

    run_logged(
        [
            str(FREESURFER / "bin/mri_vol2vol"),
            "--mov",
            str(candidate_source),
            "--targ",
            str(corrected_t1),
            "--regheader",
            "--interp",
            "nearest",
            "--keep-precision",
            "--o",
            str(candidate_t1_raw),
        ],
        log=output / "01_candidate_to_t1.log",
        environment=environment,
        outputs=(candidate_t1_raw,),
    )
    run_logged(
        [
            str(FSL_PYTHON),
            str(RELABEL),
            str(candidate_t1_raw),
            str(RELABEL_LUT),
            str(candidate_t1_nodes),
            str(candidate_t1_volumes),
        ],
        log=output / "02_candidate_relabel_t1.log",
        environment=environment,
        outputs=(candidate_t1_nodes, candidate_t1_volumes),
    )
    run_logged(
        [
            str(FSL / "flirt"),
            "-in",
            str(candidate_t1_raw),
            "-ref",
            str(b0_1mm),
            "-applyxfm",
            "-init",
            str(transform),
            "-interp",
            "nearestneighbour",
            "-datatype",
            "int",
            "-out",
            str(candidate_b0_raw),
        ],
        log=output / "03_candidate_to_b0.log",
        environment=environment,
        outputs=(candidate_b0_raw,),
    )
    run_logged(
        [
            str(FSL_PYTHON),
            str(RELABEL),
            str(candidate_b0_raw),
            str(RELABEL_LUT),
            str(candidate_nodes),
            str(candidate_volumes),
        ],
        log=output / "04_candidate_relabel_b0.log",
        environment=environment,
        outputs=(candidate_nodes, candidate_volumes),
    )
    run_logged(
        [
            str(MRTRIX / "tck2connectome"),
            str(tracks),
            str(candidate_nodes),
            str(candidate_count),
            "-assignment_radial_search",
            "4",
            "-symmetric",
            "-zero_diagonal",
            "-stat_edge",
            "sum",
            "-out_assignments",
            str(candidate_assignments),
            "-nthreads",
            str(threads),
        ],
        log=output / "05_candidate_tck2connectome.log",
        environment=environment,
        outputs=(candidate_count, candidate_assignments),
    )

    original_image, original_labels = integer_data(original_nodes)
    candidate_image, candidate_labels = integer_data(candidate_nodes)
    geometry_pass = bool(
        original_image.shape == candidate_image.shape
        and np.allclose(
            original_image.affine, candidate_image.affine, atol=1e-5, rtol=0
        )
    )
    changed = original_labels != candidate_labels
    allowed_change = (
        (original_labels == 0)
        & np.isin(candidate_labels, np.asarray(H_NODES, dtype=np.int32))
    )
    changed_only_as_intended = bool(
        np.count_nonzero(changed & ~allowed_change) == 0
    )
    labels_present = set(
        int(value) for value in np.unique(candidate_labels)
    )
    all_nodes_present = set(range(1, 380)).issubset(labels_present)

    old = np.loadtxt(original_count, delimiter=",")
    new = np.loadtxt(candidate_count, delimiter=",")
    old_assign = assignment_summary(original_assignments)
    new_assign = assignment_summary(candidate_assignments)
    old_edges = np.triu(old, 1) > 0
    new_edges = np.triu(new, 1) > 0
    delta = new - old
    h_mask = np.zeros((379, 379), dtype=bool)
    for node in H_NODES:
        h_mask[node - 1, :] = True
        h_mask[:, node - 1] = True
    absolute_delta = np.abs(delta)
    total_absolute_delta = float(absolute_delta.sum() / 2)
    h_absolute_delta = float((absolute_delta * h_mask).sum() / 2)
    names = node_names(unit)
    technical_pass = bool(
        geometry_pass
        and changed_only_as_intended
        and all_nodes_present
        and matrix_contract(old)
        and matrix_contract(new)
        and old_assign["row_count"] == EXPECTED_STREAMLINES
        and new_assign["row_count"] == EXPECTED_STREAMLINES
        and old_assign["malformed_row_count"] == 0
        and new_assign["malformed_row_count"] == 0
        and old_assign["minimum_endpoint"] >= 0
        and new_assign["minimum_endpoint"] >= 0
        and old_assign["maximum_endpoint"] <= 379
        and new_assign["maximum_endpoint"] <= 379
        and np.count_nonzero(new[119]) > 0
        and np.count_nonzero(new[299]) > 0
    )
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_hroi_surface_support_corrected_act_validation"
        ),
        "status": (
            "TECHNICAL_PASS_CORRECTED_ACT_SCIENTIFIC_POLICY_REVIEW_PENDING"
            if technical_pass
            else "FAIL"
        ),
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "source_files_modified": False,
        "unit": unit,
        "track_route": "corrected_ACT_iFOD2_primary_10m",
        "streamline_count": EXPECTED_STREAMLINES,
        "registration_route": "recovery3_fsl_bbr_primary",
        "atlas": {
            "geometry_pass": geometry_pass,
            "all_379_candidate_nodes_present": all_nodes_present,
            "changed_only_zero_to_h_nodes": changed_only_as_intended,
            "changed_voxel_n": int(changed.sum()),
            "unexpected_changed_voxel_n": int(
                np.count_nonzero(changed & ~allowed_change)
            ),
            "node120_voxels_original": int((original_labels == 120).sum()),
            "node120_voxels_candidate": int((candidate_labels == 120).sum()),
            "node300_voxels_original": int((original_labels == 300).sum()),
            "node300_voxels_candidate": int((candidate_labels == 300).sum()),
        },
        "assignment": {
            "original": old_assign,
            "candidate": new_assign,
            "fraction_delta": (
                new_assign["endpoint_assignment_fraction"]
                - old_assign["endpoint_assignment_fraction"]
            ),
        },
        "topology": {
            "matrix_contract_original": matrix_contract(old),
            "matrix_contract_candidate": matrix_contract(new),
            "density_original": float(old_edges.sum() / POSSIBLE_EDGES),
            "density_candidate": float(new_edges.sum() / POSSIBLE_EDGES),
            "density_delta": float(
                (new_edges.sum() - old_edges.sum()) / POSSIBLE_EDGES
            ),
            "edge_n_original": int(old_edges.sum()),
            "edge_n_candidate": int(new_edges.sum()),
            "gained_edge_n": int((new_edges & ~old_edges).sum()),
            "lost_edge_n": int((old_edges & ~new_edges).sum()),
            "changed_non_h_edge_n": int(
                np.count_nonzero(
                    np.triu((delta != 0) & ~h_mask, 1)
                )
            ),
            "total_absolute_count_delta": total_absolute_delta,
            "h_incident_absolute_count_delta": h_absolute_delta,
            "h_incident_absolute_delta_fraction": (
                h_absolute_delta / total_absolute_delta
                if total_absolute_delta
                else 1.0
            ),
            "node120_degree_original": int(np.count_nonzero(old[119])),
            "node120_degree_candidate": int(np.count_nonzero(new[119])),
            "node300_degree_original": int(np.count_nonzero(old[299])),
            "node300_degree_candidate": int(np.count_nonzero(new[299])),
            "node120_strength_original": float(old[119].sum()),
            "node120_strength_candidate": float(new[119].sum()),
            "node300_strength_original": float(old[299].sum()),
            "node300_strength_candidate": float(new[299].sum()),
        },
        "anatomical_neighbours": {
            "node120_name": names[120],
            "node120_original_top15": top_neighbours(old, 120, names),
            "node120_candidate_top15": top_neighbours(new, 120, names),
            "node300_name": names[300],
            "node300_original_top15": top_neighbours(old, 300, names),
            "node300_candidate_top15": top_neighbours(new, 300, names),
        },
        "inputs": {
            "candidate_manifest": file_record(candidate_manifest_path),
            "pretract_manifest": file_record(pretract_manifest_path),
            "original_atlas_result": file_record(original_result_path),
            "track_metadata": file_record(track_metadata),
            # The 13.3-GB TCK is bound by size, count and its immutable
            # tractography metadata without adding another full-file hash pass
            # while Phase B is actively consuming it.
            "tractogram": file_record(tracks, hash_content=False),
            "original_nodes": file_record(original_nodes),
            "original_count": file_record(original_count),
            "original_assignments": file_record(
                original_assignments, hash_content=False
            ),
        },
        "outputs": {
            "candidate_nodes": file_record(candidate_nodes),
            "candidate_count": file_record(candidate_count),
            "candidate_assignments": file_record(
                candidate_assignments, hash_content=False
            ),
        },
        "implementation": file_record(Path(__file__)),
    }
    atomic_json(output / "comparison.json", result)
    return result


def reparse_existing(
    unit: str, existing_attempt: Path
) -> dict[str, Any]:
    source_summary = existing_attempt.resolve() / "summary.json"
    existing = load_json(source_summary)
    if (
        existing.get("record_type")
        != "hcp379_hroi_surface_support_corrected_act_validation"
        or existing.get("unit") != unit
        or existing.get("diagnosis_labels_used") is not False
    ):
        raise ValueError("existing corrected-ACT validation differs")
    original_assignments = Path(
        str(existing["inputs"]["original_assignments"]["path"])
    ).resolve()
    candidate_assignments = Path(
        str(existing["outputs"]["candidate_assignments"]["path"])
    ).resolve()
    for path, record in (
        (original_assignments, existing["inputs"]["original_assignments"]),
        (candidate_assignments, existing["outputs"]["candidate_assignments"]),
    ):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(record["size_bytes"])
        ):
            raise ValueError(f"assignment artifact differs: {path}")
    for record, label in (
        (existing["inputs"]["original_nodes"], "original nodes"),
        (existing["inputs"]["original_count"], "original count"),
        (existing["outputs"]["candidate_nodes"], "candidate nodes"),
        (existing["outputs"]["candidate_count"], "candidate count"),
    ):
        verified_path(record, label)
    old_assign = assignment_summary(original_assignments)
    new_assign = assignment_summary(candidate_assignments)
    atlas = existing["atlas"]
    topology = existing["topology"]
    technical_pass = bool(
        atlas["geometry_pass"]
        and atlas["changed_only_zero_to_h_nodes"]
        and atlas["all_379_candidate_nodes_present"]
        and topology["matrix_contract_original"]
        and topology["matrix_contract_candidate"]
        and topology["node120_degree_candidate"] > 0
        and topology["node300_degree_candidate"] > 0
        and old_assign["row_count"] == EXPECTED_STREAMLINES
        and new_assign["row_count"] == EXPECTED_STREAMLINES
        and old_assign["malformed_row_count"] == 0
        and new_assign["malformed_row_count"] == 0
        and old_assign["minimum_endpoint"] >= 0
        and new_assign["minimum_endpoint"] >= 0
        and old_assign["maximum_endpoint"] <= 379
        and new_assign["maximum_endpoint"] <= 379
    )
    result = json.loads(json.dumps(existing))
    result.update(
        {
            "status": (
                "TECHNICAL_PASS_CORRECTED_ACT_SCIENTIFIC_POLICY_REVIEW_PENDING"
                if technical_pass
                else "FAIL"
            ),
            "generated_utc": utc_now(),
            "assignment": {
                "original": old_assign,
                "candidate": new_assign,
                "fraction_delta": (
                    new_assign["endpoint_assignment_fraction"]
                    - old_assign["endpoint_assignment_fraction"]
                ),
            },
            "reparsed_from": file_record(source_summary),
            "implementation": file_record(Path(__file__)),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", default=DEFAULT_UNIT)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reparse-attempt", type=Path)
    args = parser.parse_args()
    unit = args.unit.strip()
    if not unit or "/" in unit:
        parser.error("--unit is invalid")
    if not 1 <= args.threads <= 32:
        parser.error("--threads must be in 1..32")
    required = (
        CANDIDATE_ROOT / unit / "candidate_manifest.json",
        PRETRACT_ROOT / unit / "pretract_recovery4_manifest.json",
        ORIGINAL_ATLAS_ROOT / unit / "result.json",
        RUN_ROOT
        / "subjects"
        / unit
        / "09_hcp379_stability/primary_10m/tracks.tck",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "unit": unit,
                    "required_inputs": [str(path) for path in required],
                    "output_root": str(OUTPUT_ROOT),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt = OUTPUT_ROOT / f"{stamp}-hroi-corrected-act-validation"
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        if args.reparse_attempt is not None:
            result = reparse_existing(unit, args.reparse_attempt)
        else:
            result = validate(unit, args.threads, attempt)
    except Exception as exc:
        result = {
            "schema_version": "1.0.0",
            "record_type": (
                "hcp379_hroi_surface_support_corrected_act_validation"
            ),
            "status": "FAIL",
            "generated_utc": utc_now(),
            "diagnosis_labels_used": False,
            "source_files_modified": False,
            "unit": unit,
            "failure": f"{type(exc).__name__}: {exc}",
            "implementation": file_record(Path(__file__)),
        }
        atomic_json(attempt / "failure.json", result)
    atomic_json(attempt / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("TECHNICAL_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
