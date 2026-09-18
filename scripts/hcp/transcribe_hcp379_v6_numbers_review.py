#!/usr/bin/env python3
"""Validate and losslessly transcribe a completed HCP379 V6 Numbers review.

The script never promotes a route or starts imaging. It verifies the embedded
exact V6 pages, the immutable provenance table, all human fields and timezone
aware timestamps before writing the promotion-input CSV once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

EXP = Path("/home/ec2-user/exp")
VENDOR = EXP / ".tools/numbers_parser_4_16_3"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

from numbers_parser import Document  # noqa: E402

ROOT = Path("/data/derivatives/hcp379_v2")
V6_ROOT = ROOT / "review_recovery4_candidate_overlay_v6"
V6_MANIFEST = (
    V6_ROOT
    / "hcp379_visual_review_pack_recovery4_candidate_v6_manifest.json"
)
NUMBERS_MANIFEST = (
    V6_ROOT
    / "human_visual_qc_recovery4_candidate_v6_numbers_manifest.json"
)
TEMPLATE_NUMBERS = (
    V6_ROOT
    / "human_visual_qc_recovery4_candidate_v6_template.numbers"
)
DEFAULT_COMPLETED = (
    V6_ROOT / "human_visual_qc_recovery4_candidate_v6.numbers"
)
DEFAULT_CSV = (
    V6_ROOT / "human_visual_qc_recovery4_candidate_v6.csv"
)
DEFAULT_RECEIPT = (
    V6_ROOT
    / "human_visual_qc_recovery4_candidate_v6_numbers_transcription.json"
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
DECISION_FIELDS = ["B0 vsT1", "B0 vs 5TT", "B0 vs HCP379"]
DECISIONS = {"PASS", "FAIL", "REVIEW"}
PROVENANCE_FIELDS = [
    "unit",
    "registration_route",
    "review_page_sha256",
    "review_page_bytes",
    "pretract_manifest_sha256",
    "review_page_path",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include an explicit timezone")
    return parsed


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


def text_value(value: object) -> str:
    return "" if value is None else str(value).strip()


def load_static() -> tuple[dict, dict]:
    v6 = json.loads(V6_MANIFEST.read_text(encoding="utf-8"))
    numbers = json.loads(NUMBERS_MANIFEST.read_text(encoding="utf-8"))
    if (
        v6.get("status") != "READY_FOR_EXACT_INPUT_HUMAN_REVIEW"
        or v6.get("unit_count") != 15
        or numbers.get("status") != "READY_FOR_HUMAN_REVIEW"
        or numbers.get("validation", {}).get("embedded_panel_count") != 15
        or numbers.get("validation", {}).get("zip_crc_pass") is not True
    ):
        raise RuntimeError("static V6/Numbers review contract is not ready")
    for key, path in (
        ("source_v6_manifest", V6_MANIFEST),
        ("workbook", TEMPLATE_NUMBERS),
    ):
        record = numbers[key]
        if (
            record["path"] != str(path.resolve())
            or record["sha256"] != sha256(path)
            or record["size_bytes"] != path.stat().st_size
        ):
            raise RuntimeError(f"stale Numbers manifest record: {key}")
    return v6, numbers


def validate_workbook_structure(
    path: Path,
    v6: dict,
) -> tuple[Document, list[dict[str, str]], dict[str, object]]:
    doc = Document(path)
    expected_sheets = (
        ["Review", "Instructions"]
        + [item["unit"] for item in v6["units"]]
        + ["Provenance"]
    )
    actual_sheets = [sheet.name for sheet in doc.sheets]
    if actual_sheets != expected_sheets:
        raise RuntimeError("completed workbook sheet identity/order differs")

    review = doc.sheets["Review"].tables[0]
    headers = [
        text_value(review.cell(0, col).value)
        for col in range(len(FIELDS))
    ]
    if headers != FIELDS:
        raise RuntimeError(f"Review headers differ: {headers}")
    rows = []
    for row_index in range(1, 16):
        row = {
            field: text_value(review.cell(row_index, col).value)
            for col, field in enumerate(FIELDS)
        }
        rows.append(row)
    expected_units = [item["unit"] for item in v6["units"]]
    if [row["unit"] for row in rows] != expected_units:
        raise RuntimeError("completed Review unit identity/order differs")

    embedded_failures = []
    for item in v6["units"]:
        unit = item["unit"]
        cell = doc.sheets[unit].tables[0].cell(0, 0)
        data = (
            cell.style.bg_image.data
            if cell.style is not None
            and cell.style.bg_image is not None
            else None
        )
        if (
            data is None
            or bytes_sha256(data) != item["review_page"]["sha256"]
        ):
            embedded_failures.append(unit)
    if embedded_failures:
        raise RuntimeError(
            f"embedded exact V6 panel mismatch: {embedded_failures}"
        )

    provenance = doc.sheets["Provenance"].tables[0]
    provenance_headers = [
        text_value(provenance.cell(0, col).value)
        for col in range(len(PROVENANCE_FIELDS))
    ]
    if provenance_headers != PROVENANCE_FIELDS:
        raise RuntimeError("Provenance headers differ")
    provenance_failures = []
    for row_index, item in enumerate(v6["units"], start=1):
        expected = [
            item["unit"],
            item["registration_route"],
            item["review_page"]["sha256"],
            text_value(float(item["review_page"]["size_bytes"])),
            item["pretract_manifest"]["sha256"],
            item["review_page"]["path"],
        ]
        actual = [
            text_value(provenance.cell(row_index, col).value)
            for col in range(len(PROVENANCE_FIELDS))
        ]
        if actual != expected:
            provenance_failures.append(item["unit"])
    if provenance_failures:
        raise RuntimeError(
            f"immutable provenance table differs: {provenance_failures}"
        )
    return doc, rows, {
        "sheet_count": len(actual_sheets),
        "review_row_count": len(rows),
        "embedded_panel_count": 15,
        "embedded_panels_exact": True,
        "provenance_rows_exact": True,
    }


def assess_human_rows(
    rows: list[dict[str, str]],
    review_not_before: datetime,
) -> dict[str, object]:
    errors = []
    now = datetime.now(timezone.utc)
    timestamps = []
    for row in rows:
        unit = row["unit"]
        if unit != TARGET and (
            row["B0 vsT1"] != "PASS"
            or row["B0 vs 5TT"] != "PASS"
        ):
            errors.append(
                f"{unit}: carried T1/5TT decisions must remain PASS"
            )
        for field in DECISION_FIELDS:
            if row[field] not in DECISIONS:
                errors.append(
                    f"{unit}: {field} must be PASS, FAIL or REVIEW"
                )
        if not row["reviewer"]:
            errors.append(f"{unit}: reviewer missing")
        if not row["reviewed_utc"]:
            errors.append(f"{unit}: reviewed_utc missing")
        else:
            try:
                parsed = parse_timestamp(row["reviewed_utc"])
                parsed_utc = parsed.astimezone(timezone.utc)
                timestamps.append(parsed_utc)
                if parsed_utc < review_not_before:
                    errors.append(
                        f"{unit}: reviewed_utc predates V6 Numbers pack"
                    )
                if parsed_utc > now + timedelta(minutes=10):
                    errors.append(
                        f"{unit}: reviewed_utc is implausibly in the future"
                    )
            except (ValueError, TypeError) as exc:
                errors.append(f"{unit}: invalid reviewed_utc ({exc})")

    decision_counts = {
        field: dict(Counter(row[field] for row in rows))
        for field in DECISION_FIELDS
    }
    all_pass = all(
        row[field] == "PASS"
        for row in rows
        for field in DECISION_FIELDS
    )
    return {
        "complete": not errors,
        "errors": errors,
        "all_three_panel_families_pass": all_pass,
        "decision_counts": decision_counts,
        "reviewers": sorted({row["reviewer"] for row in rows if row["reviewer"]}),
        "earliest_reviewed_utc": (
            min(timestamps).isoformat() if timestamps else None
        ),
        "latest_reviewed_utc": (
            max(timestamps).isoformat() if timestamps else None
        ),
    }


def write_csv_once(path: Path, rows: list[dict[str, str]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite review CSV: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    with temporary.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def self_test(v6: dict, numbers: dict) -> dict[str, object]:
    _, rows, structure = validate_workbook_structure(
        TEMPLATE_NUMBERS,
        v6,
    )
    carried = sum(
        row["B0 vsT1"] == "PASS"
        and row["B0 vs 5TT"] == "PASS"
        for row in rows
        if row["unit"] != TARGET
    )
    target = next(row for row in rows if row["unit"] == TARGET)
    if carried != 14 or any(target[field] for field in DECISION_FIELDS):
        raise AssertionError("blank Numbers review template contract differs")
    return {
        "status": "PASS",
        "checks": 5,
        "template_hash_current": (
            numbers["workbook"]["sha256"] == sha256(TEMPLATE_NUMBERS)
        ),
        "carried_t1_5tt_pass_n": carried,
        "target_all_three_blank": True,
        "embedded_panel_count": structure["embedded_panel_count"],
        "production_files_written": False,
        "imaging_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--numbers",
        type=Path,
        default=DEFAULT_COMPLETED,
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=DEFAULT_RECEIPT,
    )
    parser.add_argument("--transcribe", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    v6, numbers = load_static()
    if args.self_test:
        print(json.dumps(self_test(v6, numbers), indent=2, sort_keys=True))
        return 0
    source = args.numbers.expanduser().resolve()
    if not source.is_file():
        print(
            json.dumps(
                {
                    "status": "AWAITING_COMPLETED_V6_NUMBERS_REVIEW",
                    "completed_numbers_expected": str(source),
                    "template": str(TEMPLATE_NUMBERS),
                    "transcription_executed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    _, rows, structure = validate_workbook_structure(source, v6)
    review_not_before = parse_timestamp(numbers["generated_utc"])
    assessment = assess_human_rows(rows, review_not_before)
    if not assessment["complete"]:
        print(
            json.dumps(
                {
                    "status": "INCOMPLETE_OR_INVALID_HUMAN_REVIEW",
                    "source_numbers": file_record(source),
                    "structure": structure,
                    "assessment": assessment,
                    "transcription_executed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_recovery4_v6_numbers_review_transcription"
        ),
        "status": (
            "READY_FOR_PROMOTION_PREFLIGHT"
            if assessment["all_three_panel_families_pass"]
            else "HUMAN_REVIEW_REQUIRES_RESOLUTION"
        ),
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "human_visual_qc_inferred": False,
        "imaging_executed_by_transcription": False,
        "tractography_started": False,
        "matrix_generation_started": False,
        "source_numbers": file_record(source),
        "source_v6_manifest": file_record(V6_MANIFEST),
        "numbers_template_manifest": file_record(NUMBERS_MANIFEST),
        "structure": structure,
        "assessment": assessment,
        "transcription_executed": bool(args.transcribe),
    }
    if not args.transcribe:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.receipt.exists():
        raise FileExistsError(
            f"refusing to overwrite transcription receipt: {args.receipt}"
        )
    write_csv_once(args.output_csv, rows)
    result["output_csv"] = file_record(args.output_csv)
    args.receipt.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
