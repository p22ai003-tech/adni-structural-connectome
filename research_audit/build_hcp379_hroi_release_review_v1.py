#!/home/ec2-user/fsl/bin/python
"""Build a diagnosis-blind HROI release-review pack for 15 canaries plus 114."""

from __future__ import annotations

import csv
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

from build_hcp379_hroi_surface_support_review_v1 import (
    atomic_json,
    canonical_data,
    file_record,
    render_row,
    utc_now,
)


POLICY_RECEIPT = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "hroi_surface_support_cohort_v1/policy_adoption_v1.json"
)
CANARY_SUMMARY = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "hroi_canary_atlas_v1/attempts/"
    "20260728T081233.507961Z-hroi-canary-atlas/summary.json"
)
RECOVERED_114_QC = Path(
    "/data/derivatives/hcp379_v2/corrected_freesurfer_recovery4/"
    "qc/subjects/114_S_6347_I1344943.json"
)
RECOVERED_UNIT = "114_S_6347_I1344943"
OUTPUT_ROOT = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/review/"
    "hroi_release_v1/attempts"
)
DEPENDENCY = Path(
    "/home/ec2-user/exp/research_audit/"
    "build_hcp379_hroi_surface_support_review_v1.py"
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def resolve_review_units() -> tuple[list[str], dict[str, Path]]:
    summary = load_json(CANARY_SUMMARY)
    if (
        summary.get("status") != "PASS_HROI_CANARY_ATLAS_15"
        or summary.get("unit_n") != 15
        or summary.get("diagnosis_labels_used") is not False
    ):
        raise ValueError("15-unit HROI canary summary is not an exact PASS")
    units = [str(row["unit"]) for row in summary["units"]]
    if len(units) != 15 or len(set(units)) != 15:
        raise ValueError("HROI canary units are not exactly 15 unique units")
    results = {
        str(row["unit"]): Path(row["result"]["path"]).resolve()
        for row in summary["units"]
    }
    return units + [RECOVERED_UNIT], results


def dwi_overlay_for(
    unit: str,
    canary_results: dict[str, Path],
) -> tuple[Path, dict[str, Any]]:
    if unit in canary_results:
        result = load_json(canary_results[unit])
        if (
            result.get("unit") != unit
            or result.get("status") != "PASS_HROI_CANARY_ATLAS"
            or result.get("diagnosis_labels_used") is not False
        ):
            raise ValueError(f"{unit}: canary result is not an exact PASS")
        overlay = Path(
            result["artifacts"]["hcp379_visual_overlay"]["path"]
        ).resolve()
        registration = result["registration_route"]
        route = (
            str(registration.get("route", "UNKNOWN"))
            if isinstance(registration, dict)
            else str(registration)
        )
        return overlay, {
            "route": route,
            "atlas_inside_dwi_mask_fraction": float(
                result["atlas_qc"]["atlas_inside_dwi_mask_fraction"]
            ),
            "atlas_result": file_record(canary_results[unit]),
        }
    qc = load_json(RECOVERED_114_QC)
    if (
        unit != RECOVERED_UNIT
        or qc.get("unit") != unit
        or qc.get("status") != "PASS_PRETRACT_RECOVERY4"
        or qc.get("diagnosis_labels_used") is not False
    ):
        raise ValueError(f"{unit}: recovered FreeSurfer QC is not an exact PASS")
    overlay = Path(
        qc["artifacts"]["selected_hcp379_overlay"]["path"]
    ).resolve()
    return overlay, {
        "route": str(qc["registration_route"]),
        "atlas_inside_dwi_mask_fraction": float(
            qc["hcp379_atlas_inside_dwi_mask_fraction"]
        ),
        "pretract_qc": file_record(RECOVERED_114_QC),
    }


def render_unit(
    unit: str,
    policy: dict[str, Any],
    canary_results: dict[str, Path],
    output: Path,
) -> dict[str, Any]:
    entry = policy["units"][unit]
    original_path = Path(entry["original_source"]["path"]).resolve()
    candidate_path = Path(entry["candidate"]["path"]).resolve()
    manifest_path = Path(entry["candidate_manifest"]["path"]).resolve()
    candidate_manifest = load_json(manifest_path)
    t1_path = Path(candidate_manifest["aseg"]["path"]).resolve().with_name(
        "orig.mgz"
    )
    dwi_overlay_path, dwi_metrics = dwi_overlay_for(
        unit, canary_results
    )
    for path in (
        original_path,
        candidate_path,
        manifest_path,
        t1_path,
        dwi_overlay_path,
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{unit}: missing or non-regular review input {path}")

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
    left_added = int((source_overlay == 1).sum())
    right_added = int((source_overlay == 2).sum())
    if (
        left_added <= 0
        or right_added <= 0
        or int(added.sum()) != left_added + right_added
        or int(entry["changed_voxel_n"]) != left_added + right_added
    ):
        raise ValueError(f"{unit}: HROI source change is not exact")

    with Image.open(dwi_overlay_path) as image:
        dwi_overlay = np.asarray(image.convert("RGB"))
    figure = plt.figure(figsize=(12.5, 8.5), facecolor="white")
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1, 1, 1, 1.35),
        height_ratios=(1, 1.28),
        left=0.07,
        right=0.985,
        bottom=0.07,
        top=0.86,
        hspace=0.24,
        wspace=0.08,
    )
    source_axes = np.array(
        [figure.add_subplot(grid[0, index]) for index in range(3)]
    )
    render_row(
        source_axes,
        t1,
        source_overlay,
        row_title="T1 source · added H support",
    )
    dwi_axis = figure.add_subplot(grid[1, :3])
    dwi_axis.imshow(dwi_overlay)
    dwi_axis.set_title(
        "Final corrected HCP379 boundary on DWI-space anatomy",
        fontsize=10,
        fontweight="bold",
    )
    dwi_axis.axis("off")
    review_axis = figure.add_subplot(grid[:, 3])
    review_axis.axis("off")
    review_axis.text(
        0,
        1,
        "Human visual-QC prompts",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    prompts = [
        (
            "1. Added cyan/magenta support follows the expected bilateral "
            "cortical white-surface location."
        ),
        (
            "2. No distant, extracranial, subcortical, or contralateral "
            "misprojection is visible."
        ),
        (
            "3. The final red HCP379 boundary follows the DWI-space brain "
            "anatomy without a gross shift or missing hemisphere."
        ),
    ]
    review_axis.text(
        0,
        0.93,
        (
            "\n\n".join(textwrap.fill(item, width=38) for item in prompts)
            + "\n\n"
            "Decision:  PASS   REVIEW   FAIL\n\n"
            + textwrap.fill(
                "Do not infer a decision from automated QC.", width=38
            )
        ),
        va="top",
        fontsize=9.5,
        linespacing=1.35,
    )
    review_axis.text(
        0,
        0.42,
        (
            f"Added source voxels\n"
            f"  left:  {left_added:,}\n"
            f"  right: {right_added:,}\n\n"
            f"DWI-mask coverage\n"
            f"  {dwi_metrics['atlas_inside_dwi_mask_fraction']:.3f}\n\n"
            f"Registration route\n"
            f"  {dwi_metrics['route']}"
        ),
        va="top",
        fontsize=9,
        family="monospace",
    )
    figure.suptitle(
        (
            f"{unit} · HROI source correction and final atlas review\n"
            "Diagnosis-blind · exact adopted source · no tractography decision"
        ),
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )
    figure.text(
        0.07,
        0.02,
        (
            "Cyan = left H_ROI support; magenta = right H_ROI support; "
            "red = final 379-node atlas boundary."
        ),
        fontsize=8.5,
    )
    png = output / f"{unit}_hroi_release_review.png"
    figure.savefig(png, dpi=180, facecolor="white")
    record = {
        "unit": unit,
        "left_added_source_voxels": left_added,
        "right_added_source_voxels": right_added,
        "atlas_inside_dwi_mask_fraction": dwi_metrics[
            "atlas_inside_dwi_mask_fraction"
        ],
        "registration_route": dwi_metrics["route"],
        "original_source": file_record(original_path),
        "candidate": file_record(candidate_path),
        "candidate_manifest": file_record(manifest_path),
        "t1": file_record(t1_path),
        "dwi_overlay": file_record(dwi_overlay_path),
        "dwi_evidence": {
            key: value
            for key, value in dwi_metrics.items()
            if key not in {"route", "atlas_inside_dwi_mask_fraction"}
        },
        "page_png": file_record(png),
    }
    return {"figure": figure, "record": record}


def main() -> int:
    policy = load_json(POLICY_RECEIPT)
    if (
        policy.get("status")
        != "PASS_OPERATIONAL_PRETRACT_ADOPTION_FINAL_HUMAN_QC_PENDING"
        or policy.get("unit_n") != 530
        or policy.get("pretract_use_authorized") is not True
        or policy.get("final_human_visual_qc_pending") is not True
        or policy.get("diagnosis_labels_used") is not False
    ):
        raise ValueError("HROI policy receipt is not the expected pending-QC state")
    units, canary_results = resolve_review_units()
    if len(units) != 16 or len(set(units)) != 16:
        raise ValueError("release review must contain 15 canaries plus 114")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = OUTPUT_ROOT / f"{stamp}-hroi-release-review"
    output.mkdir(parents=True, exist_ok=False)
    pdf = output / "hcp379_hroi_release_review.pdf"
    records: list[dict[str, Any]] = []
    with PdfPages(pdf) as pages:
        for unit in units:
            rendered = render_unit(unit, policy, canary_results, output)
            pages.savefig(rendered["figure"], facecolor="white")
            plt.close(rendered["figure"])
            records.append(rendered["record"])

    review_template = output / "human_visual_qc_hroi_release_template.csv"
    with review_template.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "unit",
                "source_support_status",
                "dwi_atlas_status",
                "overall_status",
                "reviewer",
                "reviewed_utc",
                "note",
            ]
        )
        for unit in units:
            writer.writerow([unit, "", "", "", "", "", ""])

    manifest = {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_hroi_release_visual_review",
        "status": "READY_FOR_HUMAN_VISUAL_QC",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "human_visual_qc_inferred": False,
        "final_release_authorized": False,
        "selected_count_tractography_authorized": False,
        "unit_n": len(records),
        "review_scope": "15 Phase-B canaries plus isolated recovered 114",
        "units": records,
        "pdf": file_record(pdf),
        "review_template": file_record(review_template),
        "policy_receipt": file_record(POLICY_RECEIPT),
        "canary_summary": file_record(CANARY_SUMMARY),
        "implementation": file_record(Path(__file__)),
        "implementation_dependency": file_record(DEPENDENCY),
    }
    manifest_path = output / "review_manifest.json"
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "unit_n": manifest["unit_n"],
                "pdf": str(pdf),
                "review_template": str(review_template),
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
