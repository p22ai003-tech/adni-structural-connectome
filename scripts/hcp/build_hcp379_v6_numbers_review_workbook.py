#!/usr/bin/env python3
"""Build a fail-closed Apple Numbers workbook for exact HCP379 V6 review.

The workbook is a review convenience only. It embeds the already frozen V6
page images, carries the accepted T1/5TT decisions for 14 unchanged routes,
and leaves every decision that still requires human review blank.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

EXP = Path("/home/ec2-user/exp")
VENDOR = EXP / ".tools/numbers_parser_4_16_3"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

from numbers_parser import Document  # noqa: E402
from numbers_parser.cell import (  # noqa: E402
    Alignment,
    BackgroundImage,
    RGB,
)

ROOT = Path("/data/derivatives/hcp379_v2")
V6_ROOT = ROOT / "review_recovery4_candidate_overlay_v6"
V6_MANIFEST = (
    V6_ROOT
    / "hcp379_visual_review_pack_recovery4_candidate_v6_manifest.json"
)
CSV_TEMPLATE = (
    V6_ROOT / "human_visual_qc_recovery4_candidate_v6_template.csv"
)
OUTPUT = (
    V6_ROOT
    / "human_visual_qc_recovery4_candidate_v6_template.numbers"
)
OUTPUT_MANIFEST = (
    V6_ROOT
    / "human_visual_qc_recovery4_candidate_v6_numbers_manifest.json"
)
TARGET = "009_S_4324_I1186579"
FIELDS = [
    "unit",
    "B0 vsT1",
    "B0 vs 5TT",
    "B0 vs HCP379",
    "reviewer",
    "reviewed_utc",
    "Notes",
]
DECISIONS = ["PASS", "FAIL", "REVIEW"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def load_inputs() -> tuple[dict, list[dict[str, str]]]:
    manifest = json.loads(V6_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "READY_FOR_EXACT_INPUT_HUMAN_REVIEW"
        or manifest.get("unit_count") != 15
        or manifest.get("diagnosis_labels_used") is not False
        or manifest.get("human_visual_qc_inferred") is not False
    ):
        raise RuntimeError("V6 review manifest is not the exact ready contract")

    with CSV_TEMPLATE.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise RuntimeError(
                f"unexpected V6 template fields: {reader.fieldnames}"
            )
        rows = list(reader)
    if len(rows) != 15 or len({row["unit"] for row in rows}) != 15:
        raise RuntimeError("V6 review template must contain 15 unique units")

    by_unit = {item["unit"]: item for item in manifest["units"]}
    if set(by_unit) != {row["unit"] for row in rows}:
        raise RuntimeError("V6 manifest/template unit sets differ")
    for row in rows:
        unit = row["unit"]
        item = by_unit[unit]
        page = Path(item["review_page"]["path"])
        if (
            not page.is_file()
            or sha256(page) != item["review_page"]["sha256"]
            or page.stat().st_size != item["review_page"]["size_bytes"]
        ):
            raise RuntimeError(f"stale exact V6 review page: {unit}")
        if unit == TARGET:
            if any(row[field] for field in FIELDS[1:6]):
                raise RuntimeError("009 must remain fully blank in template")
        elif (
            row["B0 vsT1"] != "PASS"
            or row["B0 vs 5TT"] != "PASS"
            or row["B0 vs HCP379"]
            or row["reviewer"]
            or row["reviewed_utc"]
        ):
            raise RuntimeError(f"unexpected carried review state: {unit}")
    return manifest, rows


def add_styles(doc: Document) -> dict[str, object]:
    return {
        "header": doc.add_style(
            name="V6 Header",
            bg_color=RGB(20, 45, 70),
            font_color=RGB(255, 255, 255),
            bold=True,
            font_size=11.0,
            alignment=Alignment("center", "middle"),
            text_wrap=True,
        ),
        "carried": doc.add_style(
            name="Carried PASS",
            bg_color=RGB(225, 235, 228),
            font_color=RGB(35, 80, 45),
            bold=True,
            font_size=10.0,
            alignment=Alignment("center", "middle"),
        ),
        "editable": doc.add_style(
            name="Human input required",
            bg_color=RGB(255, 241, 176),
            font_color=RGB(65, 50, 0),
            bold=True,
            font_size=10.0,
            alignment=Alignment("center", "middle"),
        ),
        "body": doc.add_style(
            name="V6 Body",
            font_size=10.0,
            alignment=Alignment("left", "middle"),
            text_wrap=True,
        ),
        "title": doc.add_style(
            name="V6 Title",
            bg_color=RGB(12, 25, 36),
            font_color=RGB(255, 255, 255),
            bold=True,
            font_size=14.0,
            alignment=Alignment("left", "middle"),
            text_wrap=True,
        ),
        "warning": doc.add_style(
            name="V6 Warning",
            bg_color=RGB(255, 225, 214),
            font_color=RGB(115, 35, 20),
            bold=True,
            font_size=10.0,
            alignment=Alignment("left", "middle"),
            text_wrap=True,
        ),
        "mono": doc.add_style(
            name="V6 Provenance",
            font_name="Menlo",
            font_size=8.0,
            alignment=Alignment("left", "middle"),
            text_wrap=True,
        ),
    }


def add_review_sheet(
    doc: Document,
    rows: list[dict[str, str]],
    styles: dict[str, object],
) -> None:
    table = doc.default_table
    table.name = "Human review"
    table.num_header_rows = 1
    table.num_header_cols = 0
    widths = [190, 90, 90, 110, 125, 185, 420]
    for col, width in enumerate(widths):
        table.col_width(col, width)
    for col, field in enumerate(FIELDS):
        table.write(0, col, field, style=styles["header"])
    table.row_height(0, 32)

    for row_index, row in enumerate(rows, start=1):
        unit = row["unit"]
        for col, field in enumerate(FIELDS):
            value = row[field]
            style = styles["body"]
            if field in FIELDS[1:4]:
                style = styles["carried"] if value == "PASS" else styles["editable"]
            table.write(row_index, col, value, style=style)
        for col in (1, 2, 3):
            table.set_cell_formatting(
                row_index,
                col,
                "popup",
                popup_values=DECISIONS,
                allow_none=True,
            )
        table.row_height(row_index, 42 if unit == TARGET else 34)


def add_instruction_sheet(
    doc: Document,
    styles: dict[str, object],
    generated_utc: str,
) -> None:
    doc.add_sheet(
        sheet_name="Instructions",
        table_name="Read before review",
        num_rows=13,
        num_cols=2,
    )
    table = doc.sheets[-1].tables[0]
    table.num_header_rows = 0
    table.num_header_cols = 0
    table.col_width(0, 210)
    table.col_width(1, 660)
    content = [
        (
            "HCP379 Recovery4 V6",
            "Diagnosis-blind exact-input spatial review. This workbook does "
            "not authorize or run imaging.",
            "title",
        ),
        (
            "How to read panels",
            "Montage rows are sagittal, coronal and axial. Columns are "
            "different slice locations; they are not left/right hemispheres.",
            "warning",
        ),
        (
            "Fourteen unchanged routes",
            "Prior T1 and 5TT PASS decisions are carried forward. Review "
            "only the corrected HCP379 parcel-edge panel.",
            "body",
        ),
        (
            TARGET,
            "Its route changed to bbr_initialized_syn_conservative. Review "
            "T1, 5TT and HCP379 on its exact candidate page.",
            "warning",
        ),
        (
            "Allowed decisions",
            "Use only PASS, FAIL or REVIEW. REVIEW remains unresolved and "
            "cannot authorize execution.",
            "body",
        ),
        (
            "Reviewer",
            "Fill reviewer and reviewed_utc for every row. Use a timezone-aware "
            "ISO timestamp, for example 2026-07-28T10:30:00+05:30.",
            "body",
        ),
        (
            "Blinding",
            "Do not add diagnosis, outcome or clinical labels.",
            "warning",
        ),
        (
            "Save completed copy as",
            str(
                V6_ROOT
                / "human_visual_qc_recovery4_candidate_v6.numbers"
            ),
            "mono",
        ),
        (
            "Next machine step",
            "Codex transcribes the completed Numbers file losslessly into the "
            "exact CSV. Promotion remains a separate fail-closed operation.",
            "body",
        ),
        (
            "Frozen V6 PDF",
            str(
                V6_ROOT
                / "hcp379_recovery4_v6_exact_readable_blinded_review.pdf"
            ),
            "mono",
        ),
        ("Workbook generated UTC", generated_utc, "mono"),
        ("Diagnosis/outcomes used", "No", "body"),
        ("Imaging executed", "No", "body"),
    ]
    for row_index, (label, text, style_name) in enumerate(content):
        table.write(row_index, 0, label, style=styles[style_name])
        table.write(row_index, 1, text, style=styles[style_name])
        table.row_height(row_index, 54 if row_index < 9 else 34)


def add_panel_sheets(
    doc: Document,
    manifest: dict,
    styles: dict[str, object],
) -> None:
    for index, item in enumerate(manifest["units"], start=1):
        unit = item["unit"]
        page = Path(item["review_page"]["path"])
        doc.add_sheet(
            sheet_name=unit,
            table_name="Exact V6 panel",
            num_rows=4,
            num_cols=1,
        )
        table = doc.sheets[-1].tables[0]
        table.num_header_rows = 0
        table.num_header_cols = 0
        table.col_width(0, 850)
        table.row_height(0, 315)
        panel_style = doc.add_style(
            name=f"Exact panel {index:02d}",
            bg_image=BackgroundImage(page.read_bytes(), page.name),
            bg_color=RGB(10, 15, 20),
            alignment=Alignment("center", "middle"),
        )
        table.write(0, 0, "", style=panel_style)
        table.write(
            1,
            0,
            (
                f"{unit} | route={item['registration_route']} | "
                f"page_sha256={item['review_page']['sha256']}"
            ),
            style=styles["mono"],
        )
        table.write(
            2,
            0,
            (
                "Review all three panels."
                if unit == TARGET
                else "T1/5TT PASS retained; review corrected HCP379 only."
            ),
            style=styles["warning"] if unit == TARGET else styles["body"],
        )
        table.write(
            3,
            0,
            "Enter decisions only in the first Review sheet.",
            style=styles["body"],
        )
        table.row_height(1, 38)
        table.row_height(2, 34)
        table.row_height(3, 30)


def add_provenance_sheet(
    doc: Document,
    manifest: dict,
    styles: dict[str, object],
) -> None:
    doc.add_sheet(
        sheet_name="Provenance",
        table_name="Exact V6 binding",
        num_rows=16,
        num_cols=6,
    )
    table = doc.sheets[-1].tables[0]
    table.num_header_rows = 1
    table.num_header_cols = 0
    headers = [
        "unit",
        "registration_route",
        "review_page_sha256",
        "review_page_bytes",
        "pretract_manifest_sha256",
        "review_page_path",
    ]
    widths = [190, 240, 420, 120, 420, 620]
    for col, (header, width) in enumerate(zip(headers, widths)):
        table.col_width(col, width)
        table.write(0, col, header, style=styles["header"])
    for row_index, item in enumerate(manifest["units"], start=1):
        values = [
            item["unit"],
            item["registration_route"],
            item["review_page"]["sha256"],
            item["review_page"]["size_bytes"],
            item["pretract_manifest"]["sha256"],
            item["review_page"]["path"],
        ]
        for col, value in enumerate(values):
            table.write(row_index, col, value, style=styles["mono"])
        table.row_height(row_index, 38)


def verify_saved_workbook(
    path: Path,
    manifest: dict,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        members = len(archive.namelist())
    if bad_member is not None:
        raise RuntimeError(f"Numbers ZIP member failed CRC: {bad_member}")

    reopened = Document(path)
    expected_sheets = (
        ["Review", "Instructions"]
        + [item["unit"] for item in manifest["units"]]
        + ["Provenance"]
    )
    actual_sheets = [sheet.name for sheet in reopened.sheets]
    if actual_sheets != expected_sheets:
        raise RuntimeError(
            f"saved sheet order differs: {actual_sheets}"
        )

    review = reopened.sheets[0].tables[0]
    actual_header = [
        review.cell(0, col).value for col in range(len(FIELDS))
    ]
    if actual_header != FIELDS:
        raise RuntimeError("saved Review header differs")
    for row_index, expected in enumerate(rows, start=1):
        actual = {
            field: review.cell(row_index, col).value
            for col, field in enumerate(FIELDS)
        }
        if actual != expected:
            raise RuntimeError(
                f"saved Review row differs for {expected['unit']}"
            )
        for col in (1, 2, 3):
            if review.cell(row_index, col)._control_id is None:
                raise RuntimeError(
                    f"decision popup missing: {expected['unit']} col {col}"
                )

    embedded = {}
    for item in manifest["units"]:
        unit = item["unit"]
        cell = reopened.sheets[unit].tables[0].cell(0, 0)
        data = cell.image_data
        if data is None:
            raise RuntimeError(f"embedded panel missing: {unit}")
        embedded_hash = bytes_sha256(data)
        if embedded_hash != item["review_page"]["sha256"]:
            raise RuntimeError(f"embedded panel hash differs: {unit}")
        embedded[unit] = embedded_hash
    return {
        "zip_crc_pass": True,
        "zip_member_count": members,
        "sheet_count": len(actual_sheets),
        "review_row_count": len(rows),
        "popup_cell_count": len(rows) * 3,
        "embedded_panel_count": len(embedded),
        "embedded_panels_exact": True,
    }


def main() -> int:
    if OUTPUT.exists() or OUTPUT_MANIFEST.exists():
        raise FileExistsError(
            "non-overwriting output already exists; version the workbook"
        )
    manifest, rows = load_inputs()
    generated_utc = utc_now()
    doc = Document(
        sheet_name="Review",
        table_name="Human review",
        num_header_rows=1,
        num_header_cols=0,
        num_rows=16,
        num_cols=len(FIELDS),
    )
    styles = add_styles(doc)
    add_review_sheet(doc, rows, styles)
    add_instruction_sheet(doc, styles, generated_utc)
    add_panel_sheets(doc, manifest, styles)
    add_provenance_sheet(doc, manifest, styles)

    temporary = OUTPUT.with_name(
        f".{OUTPUT.stem}.tmp-{os.getpid()}.numbers"
    )
    try:
        doc.save(temporary)
        validation = verify_saved_workbook(temporary, manifest, rows)
        temporary.replace(OUTPUT)
    finally:
        if temporary.exists():
            temporary.unlink()

    record = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_recovery4_v6_numbers_review_workbook_manifest"
        ),
        "status": "READY_FOR_HUMAN_REVIEW",
        "generated_utc": generated_utc,
        "diagnosis_labels_used": False,
        "human_visual_qc_inferred": False,
        "imaging_executed_by_builder": False,
        "tractography_started": False,
        "matrix_generation_started": False,
        "source_v6_manifest": file_record(V6_MANIFEST),
        "source_csv_template": file_record(CSV_TEMPLATE),
        "workbook": file_record(OUTPUT),
        "builder": file_record(Path(__file__)),
        "validation": validation,
        "completion_copy_expected": str(
            V6_ROOT / "human_visual_qc_recovery4_candidate_v6.numbers"
        ),
        "machine_csv_created": False,
    }
    OUTPUT_MANIFEST.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
