#!/home/ec2-user/fsl/bin/python
"""Build a non-overwriting HCP379 source candidate with H-ROI support.

The HCP-MMP1 H parcel (annotation index 120 in each hemisphere) is often
absent or represented by only a handful of voxels after ``mri_aparc2aseg``.
This builder projects the subject's own HCP-MMP1 annotation to the white
surface and adds only those projected voxels that remain cerebral white
matter in both ``aseg.mgz`` and the original hybrid parcellation.

The policy is diagnosis blind, symmetric, cohort-uniform, and never modifies
the locked source parcellation or FreeSurfer subject.
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


FREESURFER_HOME = Path("/home/ec2-user/freesurfer")
SUBJECTS_DIR = Path("/data/derivatives/fastsurfer")
SOURCE_ROOT = Path("/data/derivatives/parc_hcpmmp1")
RECOVERED_SOURCE = (
    Path("/data/derivatives/hcp379_v2")
    / "fastsurfer_repair/114_S_6347_I1344943/hcp/HCPMMP1+aseg.mgz"
)
RELABEL_LUT = SOURCE_ROOT / "hcpmmp1_subcort_relabel.txt"
DEFAULT_OUTPUT_ROOT = (
    Path("/data/derivatives/hcp379_v2/source_label_repair_v1")
    / "hroi_surface_support_v1"
)
LABEL2VOL = FREESURFER_HOME / "bin/mri_label2vol"
EXPECTED_LABEL_N = 379
H_ANNOTATION_INDEX = 120
HEMISPHERES = {
    "lh": {
        "name": "L_H_ROI",
        "aseg_white_matter": 2,
        "source_white_matter": 2,
        "source_h_label": 1120,
        "node": 120,
    },
    "rh": {
        "name": "R_H_ROI",
        "aseg_white_matter": 41,
        "source_white_matter": 41,
        "source_h_label": 2120,
        "node": 300,
    },
}


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


def source_path(unit: str) -> Path:
    standard = SOURCE_ROOT / unit / "HCPMMP1+aseg.mgz"
    if standard.is_file():
        return standard
    if unit == "114_S_6347_I1344943" and RECOVERED_SOURCE.is_file():
        return RECOVERED_SOURCE
    raise FileNotFoundError(f"HCP-MMP1 source is missing for {unit}")


def expected_source_labels() -> set[int]:
    labels: dict[int, int] = {}
    for line in RELABEL_LUT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source, node = (int(value) for value in line.split())
        labels[source] = node
    if (
        len(labels) != EXPECTED_LABEL_N
        or set(labels.values()) != set(range(1, EXPECTED_LABEL_N + 1))
    ):
        raise ValueError("HCP379 relabel LUT differs")
    return set(labels)


def integer_data(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    integer = np.asarray(data, dtype=np.int32)
    if data.ndim != 3 or not np.array_equal(data, integer):
        raise ValueError(f"not an integer 3D label image: {path}")
    return image, integer


def require_same_geometry(
    reference: nib.spatialimages.SpatialImage,
    image: nib.spatialimages.SpatialImage,
    *,
    label: str,
) -> None:
    if reference.shape != image.shape or not np.allclose(
        reference.affine, image.affine, atol=1e-5, rtol=0
    ):
        raise ValueError(f"geometry differs for {label}")


def project_white_surface(unit: str, hemi: str, output: Path, log: Path) -> None:
    subject = SUBJECTS_DIR / unit
    annotation = subject / "label" / f"{hemi}.HCPMMP1.annot"
    orig = subject / "mri/orig.mgz"
    labels, _, names = fsio.read_annot(str(annotation))
    expected_name = HEMISPHERES[hemi]["name"]
    actual_name = names[H_ANNOTATION_INDEX].decode("utf-8")
    if actual_name != expected_name or not np.any(labels == H_ANNOTATION_INDEX):
        raise ValueError(
            f"{unit} {hemi}: annotation index 120 differs ({actual_name})"
        )
    command = [
        str(LABEL2VOL),
        "--annot",
        str(annotation),
        "--subject",
        unit,
        "--hemi",
        hemi,
        "--temp",
        str(orig),
        "--regheader",
        str(orig),
        "--proj",
        "frac",
        "0",
        "0",
        "0.1",
        "--o",
        str(output),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "FREESURFER_HOME": str(FREESURFER_HOME),
            "SUBJECTS_DIR": str(SUBJECTS_DIR),
            "PATH": f"{FREESURFER_HOME / 'bin'}:{environment.get('PATH', '')}",
        }
    )
    with log.open("x", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError(
            f"{unit} {hemi}: mri_label2vol failed with {result.returncode}"
        )


def build(unit: str, output_root: Path) -> dict[str, Any]:
    subject = SUBJECTS_DIR / unit
    original_path = source_path(unit)
    aseg_path = subject / "mri/aseg.mgz"
    output_dir = output_root / unit
    output_dir.mkdir(parents=True, exist_ok=False)
    original_image, original = integer_data(original_path)
    aseg_image, aseg = integer_data(aseg_path)
    require_same_geometry(original_image, aseg_image, label="aseg")
    candidate = original.copy()
    hemispheric_records: dict[str, Any] = {}
    for hemi, policy in HEMISPHERES.items():
        projection_path = output_dir / f"{hemi}.HCPMMP1.white_projection.mgz"
        log_path = output_dir / f"{hemi}.HCPMMP1.white_projection.log"
        project_white_surface(unit, hemi, projection_path, log_path)
        projection_image, projection = integer_data(projection_path)
        require_same_geometry(
            original_image, projection_image, label=f"{hemi} projection"
        )
        source_wm = int(policy["source_white_matter"])
        aseg_wm = int(policy["aseg_white_matter"])
        target = int(policy["source_h_label"])
        support = (
            (projection == H_ANNOTATION_INDEX)
            & (aseg == aseg_wm)
            & (original == source_wm)
        )
        support_n = int(support.sum())
        if support_n <= 0:
            raise ValueError(f"{unit} {hemi}: H-ROI white-surface support is empty")
        before = int((original == target).sum())
        candidate[support] = target
        after = int((candidate == target).sum())
        if after != before + support_n:
            raise ValueError(f"{unit} {hemi}: support accounting differs")
        hemispheric_records[hemi] = {
            "annotation_index": H_ANNOTATION_INDEX,
            "annotation_name": policy["name"],
            "node": policy["node"],
            "source_h_label": target,
            "source_h_voxels_before": before,
            "white_surface_support_added_voxels": support_n,
            "source_h_voxels_after": after,
            "projection": file_record(projection_path),
            "projection_log": file_record(log_path),
        }
    changed = candidate != original
    expected_changed = sum(
        int(record["white_surface_support_added_voxels"])
        for record in hemispheric_records.values()
    )
    if int(changed.sum()) != expected_changed:
        raise ValueError("candidate changed voxels outside the two H-ROI supports")
    expected_labels = expected_source_labels()
    present = set(int(value) for value in np.unique(candidate))
    missing = sorted(expected_labels - present)
    if missing:
        raise ValueError(f"candidate still misses source labels: {missing}")
    candidate_path = output_dir / "HCPMMP1+aseg_hroi_surface_support.mgz"
    output_dtype = original_image.get_data_dtype()
    output_image = nib.MGHImage(
        candidate.astype(output_dtype, copy=False),
        original_image.affine,
        header=original_image.header.copy(),
    )
    nib.save(output_image, str(candidate_path))
    saved_image, saved = integer_data(candidate_path)
    require_same_geometry(original_image, saved_image, label="saved candidate")
    if not np.array_equal(candidate, saved):
        raise ValueError("saved candidate differs from in-memory candidate")
    record = {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_hroi_surface_support_candidate",
        "status": "PASS_ALL_379_SOURCE_LABELS",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "unit": unit,
        "source_files_modified": False,
        "policy": {
            "scope": "bilateral HCP-MMP1 H_ROI only",
            "projection": "subject annotation at white surface, frac 0 to 0",
            "support_tissue": "cerebral white matter in both aseg and source",
            "application": "cohort-uniform candidate; not yet release-approved",
            "subcortical_voxels_overwritten": 0,
            "other_cortical_labels_overwritten": 0,
        },
        "original_source": file_record(original_path),
        "aseg": file_record(aseg_path),
        "hemispheres": hemispheric_records,
        "changed_voxel_n": int(changed.sum()),
        "expected_source_label_n": EXPECTED_LABEL_N,
        "present_source_label_n": EXPECTED_LABEL_N,
        "missing_source_labels": [],
        "candidate": file_record(candidate_path),
        "implementation": file_record(Path(__file__)),
    }
    atomic_json(output_dir / "candidate_manifest.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    unit = args.unit.strip()
    if not unit or "/" in unit:
        parser.error("--unit is invalid")
    for path in (
        source_path(unit),
        SUBJECTS_DIR / unit / "mri/aseg.mgz",
        SUBJECTS_DIR / unit / "mri/orig.mgz",
        SUBJECTS_DIR / unit / "label/lh.HCPMMP1.annot",
        SUBJECTS_DIR / unit / "label/rh.HCPMMP1.annot",
        RELABEL_LUT,
        LABEL2VOL,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "unit": unit,
                    "source": str(source_path(unit)),
                    "output": str(args.output_root / unit),
                },
                sort_keys=True,
            )
        )
        return 0
    record = build(unit, args.output_root)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
