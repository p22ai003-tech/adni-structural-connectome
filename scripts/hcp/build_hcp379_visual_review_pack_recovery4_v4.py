#!/usr/bin/env python3
"""Build the final Recovery4 candidate review pack without promoting a route.

This pack reuses the corrected, uniform HCP379 parcel-edge panels for 14
canaries.  For 009_S_4324_I1186579 it substitutes the provisional
BBR-initialized conservative-SyN route because the selected BBR route fails the
bilateral atlas-inside-DWI gate.  All three 009 panels must be reviewed again.

The program does not infer human QC and does not alter pre-tractography inputs,
tractography, connectomes, or the canonical human-review CSV.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
V2_SOURCE = (
    EXP / "scripts/hcp/build_hcp379_visual_review_pack_recovery4_v2.py"
)
V3_ROOT = HCP_ROOT / "review_recovery4_corrected_overlay"
V3_MANIFEST = (
    V3_ROOT
    / "hcp379_visual_review_pack_recovery4_corrected_overlay_manifest.json"
)
TARGET_UNIT = "009_S_4324_I1186579"
TARGET_ROOT = (
    HCP_ROOT
    / "diagnostics/registration_recovery4_targeted"
    / TARGET_UNIT
)
TARGET_AUDIT = TARGET_ROOT / "targeted_registration_audit.json"
TARGET_ROUTE = "bbr_initialized_syn_conservative"
OUTPUT_ROOT = HCP_ROOT / "review_recovery4_candidate_overlay_v4"
ROUND1_NUMBERS = (
    Path("/data/derivatives/hcp379_v2")
    / "human_visual_qc_recovery4_template.numbers"
)
ROUND1_TRANSCRIPTION = (
    V3_ROOT / "human_visual_qc_recovery4_round1_transcription_manifest.json"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = load_module(V2_SOURCE, "build_hcp379_visual_review_pack_recovery4_v2")


def copy_bound_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    with Image.open(destination) as image:
        image.verify()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    manifest_path = (
        output_root
        / "hcp379_visual_review_pack_recovery4_candidate_v4_manifest.json"
    )
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"candidate review pack exists; use --overwrite: {manifest_path}"
        )

    v3 = V2.load_json(V3_MANIFEST)
    if (
        v3.get("status") != "READY_FOR_HUMAN_REVIEW"
        or v3.get("diagnosis_labels_used") is not False
        or v3.get("human_visual_qc_inferred") is not False
        or v3.get("unit_count") != 15
    ):
        raise ValueError("corrected Recovery4 review manifest differs")
    v3_units = {str(row["unit"]): row for row in v3["units"]}
    if len(v3_units) != 15 or TARGET_UNIT not in v3_units:
        raise ValueError("corrected Recovery4 unit set differs")

    audit = V2.load_json(TARGET_AUDIT)
    selection = audit.get("selection", {})
    if (
        audit.get("unit") != TARGET_UNIT
        or audit.get("status")
        != "PROVISIONAL_CANDIDATE_AWAITING_HUMAN_VISUAL_QC"
        or audit.get("diagnosis_labels_used") is not False
        or audit.get("tractography_started") is not False
        or audit.get("matrix_generation_started") is not False
        or selection.get("recommended_candidate") != TARGET_ROUTE
        or selection.get("promotion_status") != "NOT_PROMOTED"
    ):
        raise ValueError("targeted 009 audit is not the expected candidate")
    candidate = next(
        row for row in audit["candidates"]
        if row["candidate"] == TARGET_ROUTE
    )
    if (
        candidate["labels_found"] != 379
        or candidate["missing_labels"]
        or candidate["left_cortical_inside_dwi"] < 0.80
        or candidate["right_cortical_inside_dwi"] < 0.80
        or candidate["jacobian_in_dwi_mask"]["nonpositive_fraction"] != 0
    ):
        raise ValueError("targeted 009 candidate no longer clears hard gates")

    output_root.mkdir(parents=True, exist_ok=True)
    page_dir = output_root / "units"
    overlay_dir = output_root / "source_overlays"
    page_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    target_pretract_path = V2.verify_file_record(
        v3_units[TARGET_UNIT]["unit_pretract_manifest"]
    )
    target_pretract = V2.load_json(target_pretract_path)
    b0 = V2.verify_file_record(
        target_pretract["tractography_inputs"]["mean_b0"]
    )
    candidate_root = TARGET_ROOT / "ants_syn_bbrinit"
    t1_overlay = V2.verify_file_record(
        audit["artifacts"]["conservative_t1_overlay"]
    )
    hcp_overlay = V2.verify_file_record(
        audit["artifacts"]["conservative_hcp_overlay"]
    )
    five_tt_mask = candidate_root / "5tt_mask_b0_syn_bbrinit.nii.gz"
    if not five_tt_mask.is_file():
        raise FileNotFoundError(five_tt_mask)
    five_tt_overlay = (
        overlay_dir
        / f"{TARGET_UNIT}_b0_vs_5tt_{TARGET_ROUTE}.png"
    )
    V2.run_slices(b0, five_tt_mask, five_tt_overlay)

    target_sources = [
        ("B0 vs candidate T1", t1_overlay),
        ("B0 vs candidate 5TT support", five_tt_overlay),
        ("B0 vs candidate HCP379: uniform parcel edges", hcp_overlay),
    ]
    target_page = V2.render_unit(
        TARGET_UNIT,
        target_sources,
        route=f"PROVISIONAL {TARGET_ROUTE}",
        dice=float(candidate["five_tt_dwi_dice"]),
        atlas_inside=float(candidate["atlas_inside_dwi_overall"]),
    )

    pages: list[Image.Image] = []
    unit_records: list[dict[str, Any]] = []
    for index, unit in enumerate(sorted(v3_units), start=1):
        destination = page_dir / f"{index:02d}_{unit}.png"
        if unit == TARGET_UNIT:
            target_page.save(destination, format="PNG", optimize=True)
            page = target_page.copy()
            source_panels = {
                label: V2.file_record(path)
                for label, path in target_sources
            }
            route = TARGET_ROUTE
            route_status = "PROVISIONAL_NOT_PROMOTED"
            source_review_page = None
        else:
            source_page = V2.verify_file_record(
                v3_units[unit]["review_page"]
            )
            copy_bound_image(source_page, destination)
            page = Image.open(destination).convert("RGB")
            source_panels = v3_units[unit]["source_panels"]
            route = str(v3_units[unit]["registration_route"])
            route_status = "UNCHANGED_FROM_CORRECTED_RECOVERY4"
            source_review_page = V2.file_record(source_page)
        pages.append(page)
        unit_records.append(
            {
                "unit": unit,
                "diagnosis_labels_used": False,
                "human_visual_qc_inferred": False,
                "registration_route": route,
                "route_status": route_status,
                "source_review_page": source_review_page,
                "source_panels": source_panels,
                "review_page": V2.file_record(destination),
            }
        )
    target_page.close()

    pdf_path = (
        output_root
        / "hcp379_recovery4_v4_provisional_route_blinded_review.pdf"
    )
    pages[0].save(
        pdf_path,
        format="PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=150.0,
    )
    for page in pages:
        page.close()

    template_path = (
        output_root / "human_visual_qc_recovery4_candidate_v4_template.csv"
    )
    with template_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "unit",
                "B0 vsT1",
                "B0 vs 5TT",
                "B0 vs HCP379",
                "reviewer",
                "reviewed_utc",
                "Notes",
            ],
        )
        writer.writeheader()
        for unit in sorted(v3_units):
            target = unit == TARGET_UNIT
            writer.writerow(
                {
                    "unit": unit,
                    "B0 vsT1": "" if target else "PASS",
                    "B0 vs 5TT": "" if target else "PASS",
                    "B0 vs HCP379": "",
                    "reviewer": "",
                    "reviewed_utc": "",
                    "Notes": (
                        "Review all three provisional-route panels."
                        if target else
                        "T1 and 5TT PASS carried from submitted round 1; "
                        "review corrected HCP379 edge panel."
                    ),
                }
            )

    manifest = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_visual_review_pack_"
            "recovery4_candidate_v4"
        ),
        "status": "READY_FOR_HUMAN_CANDIDATE_REVIEW",
        "generated_utc": V2.utc_now(),
        "diagnosis_labels_used": False,
        "human_visual_qc_inferred": False,
        "canonical_human_gate_changed": False,
        "tractography_started": False,
        "matrix_generation_started": False,
        "unit_count": len(unit_records),
        "corrected_review_manifest": V2.file_record(V3_MANIFEST),
        "targeted_009_audit": V2.file_record(TARGET_AUDIT),
        "submitted_round1_numbers": V2.file_record(ROUND1_NUMBERS),
        "round1_transcription_manifest": V2.file_record(
            ROUND1_TRANSCRIPTION
        ),
        "route_substitution": {
            "unit": TARGET_UNIT,
            "old_route": "recovery3_fsl_bbr_primary",
            "candidate_route": TARGET_ROUTE,
            "promotion_status": "NOT_PROMOTED",
            "reason": selection["reason"],
            "human_review_required_for": [
                "B0 vs candidate T1",
                "B0 vs candidate 5TT support",
                "B0 vs candidate HCP379 uniform parcel edges",
            ],
        },
        "carried_human_evidence": {
            "unchanged_units": 14,
            "B0_vs_T1": "PASS",
            "B0_vs_5TT": "PASS",
            "source": "submitted Recovery4 round-1 Numbers workbook",
            "HCP379_status_carried": False,
        },
        "units": unit_records,
        "review_pdf": V2.file_record(pdf_path),
        "human_qc_template": V2.file_record(template_path),
        "instructions": (
            "Review the corrected HCP379 panel for all 15 units. For 009, "
            "also re-review candidate T1 and 5TT panels. Enter only genuine "
            "human decisions; this pack does not promote the 009 route."
        ),
        "builder": V2.file_record(Path(__file__)),
    }
    V2.atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "unit_count": len(unit_records),
                "review_pdf": str(pdf_path),
                "template": str(template_path),
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
