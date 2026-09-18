#!/home/ec2-user/fsl/bin/python
"""Build diagnosis-blind visual-QC pages for the H-ROI support candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
import nibabel as nib
import numpy as np


UNITS = (
    "002_S_0413_I863064",
    "100_S_6273_I1221370",
    "100_S_4469_I1245616",
    "094_S_6440_I1019233",
)
CANDIDATE_ROOT = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "hroi_surface_support_v1"
)
DEFAULT_ASSIGNMENT_ATTEMPT = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/validation/"
    "hroi_surface_support_v1/attempts/"
    "20260728T052429.213645Z-hroi-assignment-validation"
)
OUTPUT_ROOT = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/review/"
    "hroi_surface_support_v1/attempts"
)
NODE_TABLE = Path(
    "/data/derivatives/parc_hcpmmp1/hcpmmp1_subcort_nodes.csv"
)
CMAP = ListedColormap(["#00CFE8", "#FF3D8D"])
NORM = BoundaryNorm([0.5, 1.5, 2.5], CMAP.N)


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


def canonical_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asanyarray(image.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"review image is not 3D or singleton-4D: {path}")
    return data, image.affine


def robust_window(data: np.ndarray) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    positive = finite[finite > 0]
    sample = positive if positive.size else finite
    if not sample.size:
        return 0.0, 1.0
    low, high = np.percentile(sample, [1, 99])
    if high <= low:
        high = low + 1
    return float(low), float(high)


def centroid(mask: np.ndarray) -> tuple[int, int, int]:
    coordinates = np.argwhere(mask > 0)
    if not coordinates.size:
        raise ValueError("review mask is empty")
    center = np.rint(np.median(coordinates, axis=0)).astype(int)
    return tuple(int(value) for value in center)


def plane(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return data[index, :, :].T
    if axis == 1:
        return data[:, index, :].T
    return data[:, :, index].T


def render_row(
    axes: np.ndarray,
    anatomical: np.ndarray,
    overlay: np.ndarray,
    *,
    row_title: str,
) -> None:
    center = centroid(overlay)
    low, high = robust_window(anatomical)
    names = ("sagittal", "coronal", "axial")
    for axis, (name, index) in enumerate(zip(names, center)):
        axes[axis].imshow(
            plane(anatomical, axis, index),
            cmap="gray",
            origin="lower",
            vmin=low,
            vmax=high,
            interpolation="nearest",
        )
        mask = np.ma.masked_where(
            plane(overlay, axis, index) == 0,
            plane(overlay, axis, index),
        )
        axes[axis].imshow(
            mask,
            cmap=CMAP,
            norm=NORM,
            origin="lower",
            alpha=0.85,
            interpolation="nearest",
        )
        axes[axis].set_title(f"{name} · slice {index}", fontsize=9)
        axes[axis].axis("off")
    axes[0].text(
        -0.04,
        0.5,
        row_title,
        transform=axes[0].transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=10,
        fontweight="bold",
    )


def render_unit(
    unit: str,
    assignment_attempt: Path,
    output: Path,
) -> dict[str, Any]:
    original_path = (
        Path("/data/derivatives/parc_hcpmmp1")
        / unit
        / "HCPMMP1+aseg.mgz"
    )
    candidate_path = (
        CANDIDATE_ROOT / unit / "HCPMMP1+aseg_hroi_surface_support.mgz"
    )
    t1_path = Path("/data/derivatives/fastsurfer") / unit / "mri/orig.mgz"
    b0_path = Path(f"/data/derivatives/dwi_t1_bbr/{unit}_b0mean_ras.nii.gz")
    nodes_path = (
        assignment_attempt / unit / "nodes_hroi_surface_support_b0.nii.gz"
    )
    manifest_path = CANDIDATE_ROOT / unit / "candidate_manifest.json"
    for path in (
        original_path,
        candidate_path,
        t1_path,
        b0_path,
        nodes_path,
        manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    original, original_affine = canonical_data(original_path)
    candidate, candidate_affine = canonical_data(candidate_path)
    t1, t1_affine = canonical_data(t1_path)
    if (
        original.shape != candidate.shape
        or original.shape != t1.shape
        or not np.allclose(original_affine, candidate_affine, atol=1e-5)
        or not np.allclose(original_affine, t1_affine, atol=1e-5)
    ):
        raise ValueError(f"{unit}: source review geometry differs")
    added = candidate != original
    source_overlay = np.zeros(candidate.shape, dtype=np.uint8)
    source_overlay[added & (candidate == 1120)] = 1
    source_overlay[added & (candidate == 2120)] = 2
    b0, b0_affine = canonical_data(b0_path)
    nodes, nodes_affine = canonical_data(nodes_path)
    if b0.shape != nodes.shape or not np.allclose(
        b0_affine, nodes_affine, atol=1e-5
    ):
        raise ValueError(f"{unit}: DWI review geometry differs")
    dwi_overlay = np.zeros(nodes.shape, dtype=np.uint8)
    dwi_overlay[nodes == 120] = 1
    dwi_overlay[nodes == 300] = 2
    if not np.any(source_overlay == 1) or not np.any(source_overlay == 2):
        raise ValueError(f"{unit}: source support is incomplete")
    if not np.any(dwi_overlay == 1) or not np.any(dwi_overlay == 2):
        raise ValueError(f"{unit}: DWI H nodes are incomplete")
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    render_row(
        axes[0],
        t1,
        source_overlay,
        row_title="T1 · added white-surface support",
    )
    render_row(
        axes[1],
        b0,
        dwi_overlay,
        row_title="mean b0 · final H nodes",
    )
    left_added = int((source_overlay == 1).sum())
    right_added = int((source_overlay == 2).sum())
    left_dwi = int((dwi_overlay == 1).sum())
    right_dwi = int((dwi_overlay == 2).sum())
    figure.suptitle(
        (
            f"{unit} · HCP-MMP1 H-ROI support review\n"
            f"source added L/R: {left_added}/{right_added} · "
            f"DWI H-node voxels L/R: {left_dwi}/{right_dwi}"
        ),
        fontsize=12,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.01,
        (
            "Diagnosis-blind candidate. Added source voxels are restricted to "
            "subject-surface H_ROI points that remain cerebral white matter."
        ),
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.08,
        top=0.84,
        hspace=0.28,
        wspace=0.10,
    )
    png = output / f"{unit}_hroi_surface_support_review.png"
    figure.savefig(png, dpi=180, facecolor="white")
    record = {
        "unit": unit,
        "left_added_source_voxels": left_added,
        "right_added_source_voxels": right_added,
        "left_dwi_node_voxels": left_dwi,
        "right_dwi_node_voxels": right_dwi,
        "png": file_record(png),
        "candidate_manifest": file_record(manifest_path),
        "nodes": file_record(nodes_path),
    }
    return {"figure": figure, "record": record}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assignment-attempt",
        type=Path,
        default=DEFAULT_ASSIGNMENT_ATTEMPT,
    )
    args = parser.parse_args()
    if not args.assignment_attempt.is_dir():
        raise NotADirectoryError(args.assignment_attempt)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = OUTPUT_ROOT / f"{stamp}-hroi-visual-review"
    output.mkdir(parents=True, exist_ok=False)
    pdf = output / "hcp379_hroi_surface_support_review.pdf"
    records: list[dict[str, Any]] = []
    with PdfPages(pdf) as pages:
        for unit in UNITS:
            rendered = render_unit(unit, args.assignment_attempt, output)
            pages.savefig(rendered["figure"], facecolor="white")
            plt.close(rendered["figure"])
            records.append(rendered["record"])
    manifest = {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_hroi_surface_support_review",
        "status": "READY_FOR_VISUAL_QC",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "unit_n": len(records),
        "units": records,
        "pdf": file_record(pdf),
        "assignment_attempt": str(args.assignment_attempt.resolve()),
        "node_table": file_record(NODE_TABLE),
        "implementation": file_record(Path(__file__)),
    }
    atomic_json(output / "review_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
