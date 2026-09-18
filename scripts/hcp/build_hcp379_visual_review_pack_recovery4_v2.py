#!/usr/bin/env python3
"""Build the blinded Recovery4 visual-review pack for the 15-unit canary.

Each page binds the exact pre-tractography manifest inputs and displays
mean-b0 versus the selected T1, selected 5TT support, and corrected HCP379
parcellation.  The generated CSV is intentionally blank: visual PASS must be
entered by a real reviewer and is never inferred by this program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps


HCP_ROOT = Path("/data/derivatives/hcp379_v2")
PRETRACT_ROOT = HCP_ROOT / "pretract_recovery4"
PRETRACT_MANIFEST = PRETRACT_ROOT / "pretract_recovery4_manifest.json"
OUTPUT_ROOT = HCP_ROOT / "review_recovery4"
FSL = Path("/home/ec2-user/fsl")
FONT_REGULAR = Path(
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"
)
FONT_BOLD = Path(
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
)
PANEL_WIDTH = 760
PANEL_HEIGHT = 560
HEADER_HEIGHT = 126
FOOTER_HEIGHT = 48
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
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_file_record(record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or int(record.get("size_bytes", -1)) != path.stat().st_size
        or str(record.get("sha256", "")) != sha256_file(path)
    ):
        raise ValueError(f"artifact binding differs: {path}")
    return path


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


def load_units() -> list[tuple[str, Path, dict[str, Any]]]:
    master = load_json(PRETRACT_MANIFEST)
    if (
        master.get("record_type")
        != "diagnosis_blind_hcp379_pretract_recovery4_manifest"
        or master.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or master.get("diagnosis_labels_used") is not False
        or master.get("human_visual_qc_inferred") is not False
        or master.get("unit_count") != 15
    ):
        raise ValueError("Recovery4 pretract manifest is not an exact PASS")
    units: list[tuple[str, Path, dict[str, Any]]] = []
    for row in master["units"]:
        unit = str(row["unit"])
        manifest_path = verify_file_record(row["manifest"])
        record = load_json(manifest_path)
        if (
            record.get("unit") != unit
            or record.get("status")
            != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
            or record.get("diagnosis_labels_used") is not False
            or record.get("human_visual_qc_inferred") is not False
        ):
            raise ValueError(f"unit manifest differs: {unit}")
        units.append((unit, manifest_path, record))
    if len({unit for unit, _, _ in units}) != 15:
        raise ValueError("Recovery4 pretract unit set differs")
    return sorted(units, key=lambda item: item[0])


def run_slices(base: Path, overlay: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "FSLDIR": str(FSL),
            "FSLOUTPUTTYPE": "NIFTI_GZ",
            "PATH": f"{FSL / 'bin'}:{env.get('PATH', '')}",
        }
    )
    completed = subprocess.run(
        [
            str(FSL / "bin/slices"),
            str(base),
            str(overlay),
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if completed.returncode or not output.is_file():
        raise RuntimeError(
            f"slices failed rc={completed.returncode}: "
            f"{completed.stdout}\n{completed.stderr}"
        )


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def render_unit(
    unit: str,
    sources: list[tuple[str, Path]],
    *,
    route: str,
    dice: float,
    atlas_inside: float,
) -> Image.Image:
    width = PANEL_WIDTH * len(sources)
    height = HEADER_HEIGHT + PANEL_LABEL_HEIGHT + PANEL_HEIGHT + FOOTER_HEIGHT
    canvas = Image.new("RGB", (width, height), "#101418")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (28, 16),
        f"HCP379-v2 Recovery4 blinded spatial review | {unit}",
        fill="#f2f5f7",
        font=font(FONT_BOLD, 32),
    )
    draw.text(
        (28, 57),
        f"Route: {route}  |  5TT/DWI Dice: {dice:.3f}  |  "
        f"HCP379 inside DWI: {atlas_inside:.3f}",
        fill="#c7d0d8",
        font=font(FONT_REGULAR, 20),
    )
    draw.text(
        (28, 87),
        "Inspect boundaries, gross shifts, tissue plausibility and parcel loss.",
        fill="#c7d0d8",
        font=font(FONT_REGULAR, 20),
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
        (28, height - 35),
        "Diagnosis/outcome labels are absent. Record the human decision in the "
        "separate CSV only after reviewing all panels.",
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
            fieldnames=(
                "unit",
                "status",
                "reviewer",
                "reviewed_utc",
                "notes",
            ),
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
    output_root = args.output_root.resolve()
    manifest_path = (
        output_root / "hcp379_visual_review_pack_recovery4_manifest.json"
    )
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"review pack exists; use --overwrite: {manifest_path}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    unit_dir = output_root / "units"
    overlay_dir = output_root / "source_overlays"
    unit_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    bound_units = load_units()
    pages: list[Image.Image] = []
    unit_records: list[dict[str, Any]] = []
    for index, (unit, unit_manifest_path, record) in enumerate(
        bound_units, start=1
    ):
        dwi = record["tractography_inputs"]
        b0 = verify_file_record(dwi["mean_b0"])
        t1 = verify_file_record(
            record["registration"]["selected_t1_in_b0"]
        )
        five_tt_mask = verify_file_record(
            load_json(
                Path(record["automated_qc"]["spatial"]["path"])
            )["artifacts"]["selected_five_tt_mask"]
        )
        hcp_overlay = verify_file_record(record["hcp379"]["visual_overlay"])
        t1_overlay = overlay_dir / f"{unit}_b0_vs_t1.png"
        five_tt_overlay = overlay_dir / f"{unit}_b0_vs_5tt.png"
        run_slices(b0, t1, t1_overlay)
        run_slices(b0, five_tt_mask, five_tt_overlay)
        sources = [
            ("B0 vs selected T1", t1_overlay),
            ("B0 vs selected 5TT support", five_tt_overlay),
            ("B0 vs corrected HCP379", hcp_overlay),
        ]
        route = str(record["registration"]["route"])
        dice = float(
            record["registration"]["selected_5tt_dwi_mask_dice"]
        )
        atlas_inside = float(
            record["registration"][
                "hcp379_atlas_inside_dwi_mask_fraction"
            ]
        )
        page = render_unit(
            unit,
            sources,
            route=route,
            dice=dice,
            atlas_inside=atlas_inside,
        )
        page_path = unit_dir / f"{index:02d}_{unit}.png"
        page.save(page_path, format="PNG", optimize=True)
        pages.append(page)
        unit_records.append(
            {
                "unit": unit,
                "diagnosis_labels_used": False,
                "human_visual_qc_inferred": False,
                "registration_route": route,
                "selected_5tt_dwi_mask_dice": dice,
                "hcp379_atlas_inside_dwi_mask_fraction": atlas_inside,
                "unit_pretract_manifest": file_record(unit_manifest_path),
                "source_panels": {
                    label: file_record(path) for label, path in sources
                },
                "review_page": file_record(page_path),
            }
        )

    pdf_path = output_root / "hcp379_recovery4_blinded_visual_review.pdf"
    pages[0].save(
        pdf_path,
        format="PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=150.0,
    )
    for page in pages:
        page.close()
    template_path = output_root / "human_visual_qc_recovery4_template.csv"
    write_review_template(
        template_path,
        [unit for unit, _, _ in bound_units],
    )
    manifest = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_visual_review_pack_recovery4"
        ),
        "status": "READY_FOR_HUMAN_REVIEW",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "human_visual_qc_inferred": False,
        "unit_count": len(bound_units),
        "pretract_manifest": file_record(PRETRACT_MANIFEST),
        "units": unit_records,
        "review_pdf": file_record(pdf_path),
        "blank_human_qc_template": file_record(template_path),
        "instructions": (
            "A human reviewer must inspect every page and populate status, "
            "reviewer and reviewed_utc. This pack does not assert visual PASS."
        ),
        "builder": file_record(Path(__file__)),
    }
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "unit_count": len(bound_units),
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
