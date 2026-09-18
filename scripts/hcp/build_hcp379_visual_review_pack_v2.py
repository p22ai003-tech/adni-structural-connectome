#!/usr/bin/env python3
"""Build a diagnosis-blind visual review pack for the HCP379 canary.

The pack combines the corrected Recovery3 T1, 5TT and AAL overlays with the
new HCP379 overlay for each of the 15 technically bound units.  It deliberately
emits a blank human-QC CSV template: this program does not impersonate a human
reviewer or infer a visual PASS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps


EXP = Path("/home/ec2-user/exp")
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
OUTPUT_ROOT = HCP_ROOT / "review"
RECOVERY3_COMPLETION = (
    RUN_ROOT
    / "publication/pre_tractography_canary_recovery3_completion.json"
)
RECOVERY3_BINDING = (
    EXP
    / "research_audit/outputs/"
    "h04a_r1_retry4_pretract_recovery3_package_v1/"
    "recovery3_execution_binding.json"
)
ATLAS_SUMMARY = (
    HCP_ROOT
    / "corrected_atlas_canary/corrected_atlas_canary_summary.json"
)
FONT_REGULAR = Path(
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"
)
FONT_BOLD = Path(
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
)
PANEL_WIDTH = 760
PANEL_HEIGHT = 560
HEADER_HEIGHT = 92
FOOTER_HEIGHT = 44
PANEL_LABEL_HEIGHT = 42


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


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


def load_units() -> list[str]:
    binding = load_json(RECOVERY3_BINDING)
    units = binding.get("units")
    if (
        binding.get("record_type")
        != "retry4_pretract_recovery3_execution_binding"
        or binding.get("diagnosis_labels_used") is not False
        or not isinstance(units, list)
        or len(units) != 15
        or len(units) != len(set(units))
    ):
        raise ValueError("Recovery3 binding differs")
    return sorted(str(unit) for unit in units)


def validate_upstream(units: list[str]) -> dict[str, Any]:
    completion = load_json(RECOVERY3_COMPLETION)
    atlas_summary = load_json(ATLAS_SUMMARY)
    if (
        completion.get("status") != "PASS"
        or atlas_summary.get("status") != "PASS"
        or atlas_summary.get("diagnosis_labels_used") is not False
        or atlas_summary.get("unit_count") != 15
        or atlas_summary.get("passed_unit_count") != 15
        or {
            str(row.get("unit"))
            for row in atlas_summary.get("units", [])
        }
        != set(units)
    ):
        raise ValueError(
            "Recovery3 completion or corrected HCP379 atlas summary is not "
            "an exact 15/15 PASS"
        )
    return {
        "recovery3_completion": file_record(RECOVERY3_COMPLETION),
        "corrected_atlas_summary": file_record(ATLAS_SUMMARY),
        "recovery3_binding": file_record(RECOVERY3_BINDING),
    }


def panel_paths(unit: str) -> list[tuple[str, Path]]:
    preflight = RUN_ROOT / "subjects" / unit / "06_preflight"
    atlas = HCP_ROOT / "corrected_atlas_canary/subjects" / unit
    return [
        ("T1 alignment", preflight / "review_b0_vs_t1_recovery3.png"),
        ("5TT / ACT anatomy", preflight / "review_b0_vs_5tt_recovery3.png"),
        ("AAL reference", preflight / "review_b0_vs_aal3_recovery3.png"),
        ("Corrected HCP379", atlas / "review_b0_vs_hcp379.png"),
    ]


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def render_unit(unit: str, sources: list[tuple[str, Path]]) -> Image.Image:
    width = PANEL_WIDTH * len(sources)
    height = HEADER_HEIGHT + PANEL_LABEL_HEIGHT + PANEL_HEIGHT + FOOTER_HEIGHT
    canvas = Image.new("RGB", (width, height), "#101418")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (28, 18),
        f"HCP379 corrected-space blinded review | {unit}",
        fill="#f2f5f7",
        font=font(FONT_BOLD, 34),
    )
    draw.text(
        (28, 57),
        "Inspect boundaries, gross shifts, tissue plausibility and parcel loss.",
        fill="#c7d0d8",
        font=font(FONT_REGULAR, 21),
    )
    evidence: list[Image.Image] = []
    for index, (label, path) in enumerate(sources):
        image = Image.open(path).convert("RGB")
        fitted = ImageOps.contain(
            image, (PANEL_WIDTH - 18, PANEL_HEIGHT - 18)
        )
        evidence.append(fitted)
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
    draw.text(
        (28, height - 33),
        "Diagnosis and outcome labels are intentionally absent. Human decision "
        "must be recorded separately.",
        fill="#9ba9b5",
        font=font(FONT_REGULAR, 18),
    )
    for image in evidence:
        image.close()
    return canvas


def write_review_template(path: Path, units: list[str]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        newline="",
        encoding="utf-8",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("unit", "status", "reviewer", "reviewed_utc", "notes"),
        )
        writer.writeheader()
        for unit in units:
            writer.writerow(
                {
                    "unit": unit,
                    "status": "",
                    "reviewer": "",
                    "reviewed_utc": "",
                    "notes": "",
                }
            )
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    manifest_path = output_root / "hcp379_visual_review_pack_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"review pack exists; use --overwrite: {manifest_path}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    unit_dir = output_root / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    units = load_units()
    upstream = validate_upstream(units)
    pages: list[Image.Image] = []
    unit_records: list[dict[str, Any]] = []
    for index, unit in enumerate(units, start=1):
        sources = panel_paths(unit)
        missing = [str(path) for _, path in sources if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"missing visual inputs for {unit}: {missing}"
            )
        page = render_unit(unit, sources)
        page_path = unit_dir / f"{index:02d}_{unit}.png"
        page.save(page_path, format="PNG", optimize=True)
        pages.append(page)
        unit_records.append(
            {
                "unit": unit,
                "diagnosis_labels_used": False,
                "source_panels": {
                    label: file_record(path) for label, path in sources
                },
                "review_page": file_record(page_path),
            }
        )
    pdf_path = output_root / "hcp379_blinded_visual_review.pdf"
    pages[0].save(
        pdf_path,
        format="PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=150.0,
    )
    for page in pages:
        page.close()
    template_path = output_root / "human_visual_qc_template.csv"
    write_review_template(template_path, units)
    manifest = {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_visual_review_pack",
        "status": "READY_FOR_HUMAN_REVIEW",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "human_visual_qc_inferred": False,
        "unit_count": len(units),
        "upstream": upstream,
        "units": unit_records,
        "review_pdf": file_record(pdf_path),
        "blank_human_qc_template": file_record(template_path),
        "instructions": (
            "A human reviewer must inspect every page and populate status, "
            "reviewer and reviewed_utc. This pack does not assert visual PASS."
        ),
    }
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "unit_count": len(units),
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
