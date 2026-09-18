#!/usr/bin/env python3
"""Build a readable, exact-input-bound Recovery4 V6 human-review pack.

V6 changes only the rendered review layout.  It reuses the exact source-panel
artifacts and pretract manifests bound by V5, shortens headings, and explains
that montage columns are slice positions rather than hemispheres.  No imaging
derivative, registration route, tractography input, or human decision is
changed or inferred.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


EXP = Path("/home/ec2-user/exp")
ROOT = Path("/data/derivatives/hcp379_v2")
TARGET = "009_S_4324_I1186579"
ROUTE = "bbr_initialized_syn_conservative"
V2_SOURCE = (
    EXP / "scripts/hcp/build_hcp379_visual_review_pack_recovery4_v2.py"
)
V5_ROOT = ROOT / "review_recovery4_candidate_overlay_v5"
V5_MANIFEST = (
    V5_ROOT
    / "hcp379_visual_review_pack_recovery4_candidate_v5_manifest.json"
)
ROUND1_TRANSCRIPTION = (
    ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_round1_transcription_manifest.json"
)
OUTPUT_ROOT = ROOT / "review_recovery4_candidate_overlay_v6"

PANEL_WIDTH = 760
PANEL_HEIGHT = 560
HEADER_HEIGHT = 150
PANEL_LABEL_HEIGHT = 42
FOOTER_HEIGHT = 82
FONT_REGULAR = Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf")
FONT_BOLD = Path(
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = load_module(V2_SOURCE, "hcp379_review_v2_for_v6")


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def select_panels(
    source_panels: dict[str, dict[str, Any]],
) -> list[tuple[str, Path, dict[str, Any]]]:
    selected: dict[str, tuple[Path, dict[str, Any]]] = {}
    for source_label, record in source_panels.items():
        lowered = source_label.lower()
        if "hcp379" in lowered:
            key = "hcp"
        elif "5tt" in lowered:
            key = "5tt"
        elif "t1" in lowered:
            key = "t1"
        else:
            raise ValueError(f"unrecognized source panel: {source_label}")
        if key in selected:
            raise ValueError(f"duplicate {key} source panel")
        selected[key] = (
            V2.verify_file_record(record),
            record,
        )
    if set(selected) != {"t1", "5tt", "hcp"}:
        raise ValueError("exact three-panel source set differs")
    return [
        ("B0 vs T1", *selected["t1"]),
        ("B0 vs 5TT support", *selected["5tt"]),
        ("B0 vs HCP379 parcel edges", *selected["hcp"]),
    ]


def render_page(
    *,
    unit: str,
    route: str,
    dice: float,
    atlas_inside: float,
    panels: list[tuple[str, Path, dict[str, Any]]],
) -> Image.Image:
    width = PANEL_WIDTH * 3
    height = (
        HEADER_HEIGHT
        + PANEL_LABEL_HEIGHT
        + PANEL_HEIGHT
        + FOOTER_HEIGHT
    )
    canvas = Image.new("RGB", (width, height), "#101418")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (28, 14),
        f"HCP379-v2 Recovery4 exact spatial review | {unit}",
        fill="#f2f5f7",
        font=font(FONT_BOLD, 31),
    )
    draw.text(
        (28, 55),
        f"Route: {route}  |  5TT/DWI Dice: {dice:.3f}  |  "
        f"HCP379 inside DWI: {atlas_inside:.3f}",
        fill="#c7d0d8",
        font=font(FONT_REGULAR, 20),
    )
    draw.text(
        (28, 88),
        "Montage rows: sagittal, coronal, axial. Columns are different "
        "slice locations, not left/right hemispheres.",
        fill="#f3c969",
        font=font(FONT_BOLD, 18),
    )
    draw.text(
        (28, 117),
        "Judge gross shift, anatomical boundary agreement and true parcel "
        "loss; empty background outside brain is expected.",
        fill="#c7d0d8",
        font=font(FONT_REGULAR, 18),
    )

    opened: list[Image.Image] = []
    for index, (label, path, _) in enumerate(panels):
        image = Image.open(path).convert("RGB")
        opened.append(image)
        fitted = ImageOps.contain(
            image, (PANEL_WIDTH - 18, PANEL_HEIGHT - 18)
        )
        x0 = index * PANEL_WIDTH
        draw.rectangle(
            (
                x0,
                HEADER_HEIGHT,
                x0 + PANEL_WIDTH - 1,
                HEADER_HEIGHT + PANEL_LABEL_HEIGHT + PANEL_HEIGHT,
            ),
            outline="#3a4650",
            width=2,
        )
        draw.text(
            (x0 + 18, HEADER_HEIGHT + 8),
            label,
            fill="#f2f5f7",
            font=font(FONT_BOLD, 22),
        )
        image_x = x0 + (PANEL_WIDTH - fitted.width) // 2
        image_y = (
            HEADER_HEIGHT
            + PANEL_LABEL_HEIGHT
            + (PANEL_HEIGHT - fitted.height) // 2
        )
        canvas.paste(fitted, (image_x, image_y))
    footer_y = HEADER_HEIGHT + PANEL_LABEL_HEIGHT + PANEL_HEIGHT + 10
    draw.text(
        (28, footer_y),
        "For 14 unchanged units, prior T1/5TT PASS is retained; review the "
        "corrected HCP379 panel. For 009, review all three panels.",
        fill="#c7d0d8",
        font=font(FONT_REGULAR, 17),
    )
    draw.text(
        (28, footer_y + 28),
        "Diagnosis/outcome labels are absent. PASS is never inferred.",
        fill="#9ba9b5",
        font=font(FONT_REGULAR, 17),
    )
    for image in opened:
        image.close()
    return canvas


def write_template(path: Path, units: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "unit",
                "B0 vsT1",
                "B0 vs 5TT",
                "B0 vs HCP379",
                "reviewer",
                "reviewed_utc",
                "Notes",
            ),
        )
        writer.writeheader()
        for unit in units:
            target = unit == TARGET
            writer.writerow(
                {
                    "unit": unit,
                    "B0 vsT1": "" if target else "PASS",
                    "B0 vs 5TT": "" if target else "PASS",
                    "B0 vs HCP379": "",
                    "reviewer": "",
                    "reviewed_utc": "",
                    "Notes": (
                        "Review all three exact candidate panels."
                        if target
                        else (
                            "Prior T1/5TT PASS retained; review the "
                            "corrected HCP379 edge panel."
                        )
                    ),
                }
            )


def write_instructions(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# HCP379 Recovery4 exact V6 human review",
                "",
                "This is a diagnosis-blind spatial registration review.",
                "",
                "- Each panel contains three montage rows: sagittal, "
                "coronal, and axial.",
                "- The three montage columns are different slice locations; "
                "they are not left/right hemisphere columns.",
                "- Review all corrected HCP379 panels.",
                "- For 009_S_4324_I1186579, review T1, 5TT, and HCP379 "
                "because its registration route changed.",
                "- Use only PASS, FAIL, or REVIEW.",
                "- REVIEW remains unresolved and cannot authorize execution.",
                "- Save a completed copy, not the immutable template, with "
                "a reviewer and timezone-aware reviewed_utc value.",
                "- No diagnosis or outcome information should be added.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"non-overwriting V6 review root exists: {output_root}"
        )
    v5 = V2.load_json(V5_MANIFEST)
    if (
        v5.get("record_type")
        != (
            "diagnosis_blind_hcp379_visual_review_pack_"
            "recovery4_candidate_v5"
        )
        or v5.get("status")
        != "READY_FOR_EXACT_INPUT_HUMAN_REVIEW"
        or v5.get("diagnosis_labels_used") is not False
        or v5.get("human_visual_qc_inferred") is not False
        or v5.get("unit_count") != 15
        or v5.get("route_substitution", {}).get("unit") != TARGET
        or v5.get("route_substitution", {}).get("candidate_route") != ROUTE
        or v5.get("route_substitution", {}).get(
            "review_panels_bound_to_exact_candidate_inputs"
        )
        is not True
    ):
        raise ValueError("exact-input V5 review manifest differs")
    rows = sorted(v5["units"], key=lambda row: str(row["unit"]))
    if len(rows) != 15 or len({str(row["unit"]) for row in rows}) != 15:
        raise ValueError("V5 review unit set differs")

    page_dir = output_root / "units"
    page_dir.mkdir(parents=True)
    pages: list[Image.Image] = []
    unit_records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        unit = str(row["unit"])
        pretract_path = V2.verify_file_record(row["pretract_manifest"])
        pretract = V2.load_json(pretract_path)
        registration = pretract.get("registration", {})
        route = str(registration.get("route", ""))
        if (
            pretract.get("unit") != unit
            or not route
            or (
                unit == TARGET
                and route != ROUTE
            )
        ):
            raise ValueError(f"{unit}: exact pretract manifest differs")
        panels = select_panels(row["source_panels"])
        page = render_page(
            unit=unit,
            route=route,
            dice=float(
                registration["selected_5tt_dwi_mask_dice"]
            ),
            atlas_inside=float(
                registration[
                    "hcp379_atlas_inside_dwi_mask_fraction"
                ]
            ),
            panels=panels,
        )
        page_path = page_dir / f"{index:02d}_{unit}.png"
        page.save(page_path, format="PNG", optimize=True)
        pages.append(page)
        unit_records.append(
            {
                "unit": unit,
                "diagnosis_labels_used": False,
                "human_visual_qc_inferred": False,
                "registration_route": route,
                "pretract_manifest": V2.file_record(pretract_path),
                "exact_candidate_inputs": row.get(
                    "exact_candidate_inputs"
                ),
                "source_panels": {
                    label: record
                    for label, _, record in panels
                },
                "review_page": V2.file_record(page_path),
            }
        )

    pdf_path = (
        output_root
        / "hcp379_recovery4_v6_exact_readable_blinded_review.pdf"
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
        output_root / "human_visual_qc_recovery4_candidate_v6_template.csv"
    )
    write_template(template_path, [str(row["unit"]) for row in rows])
    instructions_path = output_root / "REVIEW_INSTRUCTIONS.md"
    write_instructions(instructions_path)

    manifest_path = (
        output_root
        / "hcp379_visual_review_pack_recovery4_candidate_v6_manifest.json"
    )
    manifest = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_visual_review_pack_"
            "recovery4_candidate_v6"
        ),
        "status": "READY_FOR_EXACT_INPUT_HUMAN_REVIEW",
        "generated_utc": V2.utc_now(),
        "diagnosis_labels_used": False,
        "human_visual_qc_inferred": False,
        "canonical_human_gate_changed": False,
        "tractography_started": False,
        "matrix_generation_started": False,
        "unit_count": 15,
        "candidate_pretract_master": v5[
            "candidate_pretract_master"
        ],
        "supersedes_review_pack": V2.file_record(V5_MANIFEST),
        "round1_transcription": V2.file_record(
            ROUND1_TRANSCRIPTION
        ),
        "route_substitution": {
            "unit": TARGET,
            "candidate_route": ROUTE,
            "promotion_status": "NOT_PROMOTED",
            "review_panels_bound_to_exact_candidate_inputs": True,
        },
        "carried_human_evidence": v5["carried_human_evidence"],
        "visual_layout_correction": {
            "short_nonoverlapping_panel_titles": True,
            "montage_orientation_and_slice_guidance_added": True,
            "source_panel_artifacts_changed": False,
            "imaging_derivatives_changed": False,
            "prior_hcp379_status_carried": False,
        },
        "units": unit_records,
        "review_pdf": V2.file_record(pdf_path),
        "human_qc_template": V2.file_record(template_path),
        "review_instructions": V2.file_record(instructions_path),
        "instructions": (
            "Review corrected HCP379 for all 15 units. Review all three "
            "exact panels for 009. Enter only genuine human decisions."
        ),
        "builder": V2.file_record(Path(__file__)),
    }
    V2.atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "unit_count": 15,
                "review_pdf": str(pdf_path),
                "template": str(template_path),
                "instructions": str(instructions_path),
                "manifest": str(manifest_path),
                "imaging_derivatives_changed": False,
                "promotion_status": "NOT_PROMOTED",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
