#!/usr/bin/env python3
"""Build exact corrected-ACT visual-QC pages for the H-ROI support policy."""

from __future__ import annotations

import hashlib
import importlib.util
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
import nibabel as nib
import numpy as np


ROOT = Path("/home/ec2-user/exp")
BASE_REVIEW_SOURCE = (
    ROOT / "research_audit/build_hcp379_hroi_surface_support_review_v1.py"
)
CANDIDATE_ROOT = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "hroi_surface_support_v1"
)
AFFECTED_ATTEMPT = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/validation/"
    "hroi_affected_corrected_act_v1/attempts/"
    "20260728T060900.544977Z-affected-corrected-act-canary"
)
COMPLETE_SUMMARY = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/validation/"
    "hroi_surface_support_corrected_act_v1/attempts/"
    "20260728T055524.033287Z-hroi-corrected-act-validation/summary.json"
)
OUTPUT_ROOT = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/review/"
    "hroi_corrected_act_v1/attempts"
)


def load_base() -> Any:
    specification = importlib.util.spec_from_file_location(
        "hroi_base_review_v1", BASE_REVIEW_SOURCE
    )
    if specification is None or specification.loader is None:
        raise ImportError(BASE_REVIEW_SOURCE)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


BASE = load_base()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular file: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
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


def cases() -> list[dict[str, Any]]:
    complete = json.loads(COMPLETE_SUMMARY.read_text(encoding="utf-8"))
    complete_nodes = Path(
        complete["outputs"]["candidate_nodes"]["path"]
    ).resolve()
    return [
        {
            "unit": "002_S_0413_I863064",
            "case_class": "affected_source_missing_node_120",
            "nodes": AFFECTED_ATTEMPT / "hcp379_hroi_nodes_b0_1mm.nii.gz",
            "b0": Path(
                "/data/derivatives/hcp379_v2/"
                "corrected_scaleup_recovery4/subjects/"
                "002_S_0413_I863064/04_hcp/b0_1mm_world_grid.nii.gz"
            ),
            "validation": AFFECTED_ATTEMPT / "summary.json",
        },
        {
            "unit": "135_S_6840_I1263427",
            "case_class": "complete_source_exact_10m_reuse",
            "nodes": complete_nodes,
            "b0": Path(
                "/data/derivatives/scforge_v2/"
                "h04a_r1_recovery_20260719_retry4/subjects/"
                "135_S_6840_I1263427/04_atlas/"
                "b0_1mm_world_grid.nii.gz"
            ),
            "validation": COMPLETE_SUMMARY,
        },
    ]


def render_case(case: Mapping[str, Any], output: Path) -> dict[str, Any]:
    unit = str(case["unit"])
    original_path = (
        Path("/data/derivatives/parc_hcpmmp1")
        / unit
        / "HCPMMP1+aseg.mgz"
    )
    candidate_path = (
        CANDIDATE_ROOT / unit / "HCPMMP1+aseg_hroi_surface_support.mgz"
    )
    t1_path = Path("/data/derivatives/fastsurfer") / unit / "mri/orig.mgz"
    nodes_path = Path(case["nodes"])
    b0_path = Path(case["b0"])
    validation_path = Path(case["validation"])
    for path in (
        original_path,
        candidate_path,
        t1_path,
        nodes_path,
        b0_path,
        validation_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    original, original_affine = BASE.canonical_data(original_path)
    candidate, candidate_affine = BASE.canonical_data(candidate_path)
    t1, t1_affine = BASE.canonical_data(t1_path)
    if (
        original.shape != candidate.shape
        or original.shape != t1.shape
        or not np.allclose(original_affine, candidate_affine, atol=1e-5)
        or not np.allclose(original_affine, t1_affine, atol=1e-5)
    ):
        raise ValueError(f"{unit}: source geometry differs")
    added = candidate != original
    source_overlay = np.zeros(candidate.shape, dtype=np.uint8)
    source_overlay[added & (candidate == 1120)] = 1
    source_overlay[added & (candidate == 2120)] = 2

    b0, b0_affine = BASE.canonical_data(b0_path)
    nodes, nodes_affine = BASE.canonical_data(nodes_path)
    if b0.shape != nodes.shape or not np.allclose(
        b0_affine, nodes_affine, atol=1e-5
    ):
        raise ValueError(f"{unit}: corrected-ACT DWI geometry differs")
    dwi_overlay = np.zeros(nodes.shape, dtype=np.uint8)
    dwi_overlay[nodes == 120] = 1
    dwi_overlay[nodes == 300] = 2
    if not all(
        np.any(overlay == value)
        for overlay in (source_overlay, dwi_overlay)
        for value in (1, 2)
    ):
        raise ValueError(f"{unit}: bilateral H support is incomplete")

    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    BASE.render_row(
        axes[0],
        t1,
        source_overlay,
        row_title="T1 · added white-surface support",
    )
    BASE.render_row(
        axes[1],
        b0,
        dwi_overlay,
        row_title="corrected-ACT mean b0 · final H nodes",
    )
    left_added = int(np.count_nonzero(source_overlay == 1))
    right_added = int(np.count_nonzero(source_overlay == 2))
    left_dwi = int(np.count_nonzero(dwi_overlay == 1))
    right_dwi = int(np.count_nonzero(dwi_overlay == 2))
    figure.suptitle(
        (
            f"{unit} · corrected-ACT H-ROI policy review\n"
            f"{case['case_class']} · source added L/R "
            f"{left_added}/{right_added} · DWI nodes L/R "
            f"{left_dwi}/{right_dwi}"
        ),
        fontsize=11,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.01,
        (
            "Cyan: left H; magenta: right H. Diagnosis-blind, "
            "subject-surface support restricted to cerebral white matter."
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
    png = output / f"{unit}_hroi_corrected_act_review.png"
    figure.savefig(png, dpi=180, facecolor="white")
    return {
        "figure": figure,
        "record": {
            "unit": unit,
            "case_class": case["case_class"],
            "left_added_source_voxels": left_added,
            "right_added_source_voxels": right_added,
            "left_dwi_node_voxels": left_dwi,
            "right_dwi_node_voxels": right_dwi,
            "png": file_record(png),
            "nodes": file_record(nodes_path),
            "validation": file_record(validation_path),
        },
    }


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = OUTPUT_ROOT / f"{stamp}-hroi-corrected-act-review"
    output.mkdir(parents=True, exist_ok=False)
    pdf = output / "hcp379_hroi_corrected_act_review.pdf"
    records: list[dict[str, Any]] = []
    with PdfPages(pdf) as pages:
        for case in cases():
            rendered = render_case(case, output)
            pages.savefig(rendered["figure"], facecolor="white")
            plt.close(rendered["figure"])
            records.append(rendered["record"])
    manifest = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_corrected_act_visual_review"
        ),
        "status": "READY_FOR_VISUAL_QC",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "unit_n": len(records),
        "units": records,
        "pdf": file_record(pdf),
        "implementation": file_record(Path(__file__)),
        "base_review_source": file_record(BASE_REVIEW_SOURCE),
    }
    atomic_json(output / "review_manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": manifest["status"],
                "pdf": manifest["pdf"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
