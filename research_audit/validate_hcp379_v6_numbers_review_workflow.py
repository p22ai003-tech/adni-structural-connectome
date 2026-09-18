#!/usr/bin/env python3
"""Independently validate the HCP379 V6 Numbers human-review workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import py_compile
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
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
WORKBOOK = (
    V6_ROOT
    / "human_visual_qc_recovery4_candidate_v6_template.numbers"
)
WORKBOOK_MANIFEST = (
    V6_ROOT
    / "human_visual_qc_recovery4_candidate_v6_numbers_manifest.json"
)
COMPLETED = (
    V6_ROOT / "human_visual_qc_recovery4_candidate_v6.numbers"
)
PRODUCTION_CSV = (
    V6_ROOT / "human_visual_qc_recovery4_candidate_v6.csv"
)
BUILDER = (
    EXP / "scripts/hcp/build_hcp379_v6_numbers_review_workbook.py"
)
TRANSCRIBER = (
    EXP / "scripts/hcp/transcribe_hcp379_v6_numbers_review.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/hcp379_v6_numbers_review_workflow"
    / "validation.json"
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


def text(value: object) -> str:
    return "" if value is None else str(value).strip()


def main() -> int:
    checks: dict[str, dict[str, object]] = {}

    def check(name: str, passed: bool, evidence: dict) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    v6 = json.loads(V6_MANIFEST.read_text(encoding="utf-8"))
    workbook_manifest = json.loads(
        WORKBOOK_MANIFEST.read_text(encoding="utf-8")
    )
    static_records = {
        "workbook": WORKBOOK,
        "source_v6_manifest": V6_MANIFEST,
        "builder": BUILDER,
    }
    stale = []
    for key, path in static_records.items():
        record = workbook_manifest.get(key, {})
        if (
            not path.is_file()
            or record.get("path") != str(path.resolve())
            or record.get("sha256") != sha256(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            stale.append(key)
    check(
        "workbook_manifest_and_sources_are_current",
        not stale
        and workbook_manifest.get("status") == "READY_FOR_HUMAN_REVIEW"
        and workbook_manifest.get("diagnosis_labels_used") is False
        and workbook_manifest.get("human_visual_qc_inferred") is False,
        {"stale_records": stale},
    )

    with zipfile.ZipFile(WORKBOOK) as archive:
        bad_member = archive.testzip()
        member_count = len(archive.namelist())
    check(
        "numbers_container_crc_passes",
        bad_member is None,
        {"bad_member": bad_member, "member_count": member_count},
    )

    doc = Document(WORKBOOK)
    expected_sheets = (
        ["Review", "Instructions"]
        + [item["unit"] for item in v6["units"]]
        + ["Provenance"]
    )
    actual_sheets = [sheet.name for sheet in doc.sheets]
    check(
        "exact_eighteen_sheet_layout",
        actual_sheets == expected_sheets,
        {"sheet_count": len(actual_sheets), "sheets": actual_sheets},
    )

    review = doc.sheets["Review"].tables[0]
    header = [text(review.cell(0, col).value) for col in range(7)]
    rows = [
        {
            field: text(review.cell(row, col).value)
            for col, field in enumerate(FIELDS)
        }
        for row in range(1, 16)
    ]
    carried = sum(
        row["B0 vsT1"] == "PASS"
        and row["B0 vs 5TT"] == "PASS"
        for row in rows
        if row["unit"] != TARGET
    )
    target = next(
        (row for row in rows if row["unit"] == TARGET),
        {},
    )
    popup_missing = [
        [row_index, col]
        for row_index in range(1, 16)
        for col in (1, 2, 3)
        if review.cell(row_index, col)._control_id is None
    ]
    check(
        "fail_closed_review_table_and_popups",
        header == FIELDS
        and len(rows) == 15
        and carried == 14
        and all(not target.get(field) for field in FIELDS[1:6])
        and all(not row["B0 vs HCP379"] for row in rows)
        and not popup_missing,
        {
            "header": header,
            "rows": len(rows),
            "carried_t1_5tt_pass_n": carried,
            "target_all_three_blank": all(
                not target.get(field)
                for field in ("B0 vsT1", "B0 vs 5TT", "B0 vs HCP379")
            ),
            "hcp379_blank_n": sum(
                not row["B0 vs HCP379"] for row in rows
            ),
            "popup_missing": popup_missing,
        },
    )

    embedded_failures = []
    for item in v6["units"]:
        unit = item["unit"]
        cell = doc.sheets[unit].tables[0].cell(0, 0)
        payload = (
            cell.style.bg_image.data
            if cell.style is not None
            and cell.style.bg_image is not None
            else None
        )
        if (
            payload is None
            or bytes_sha256(payload) != item["review_page"]["sha256"]
        ):
            embedded_failures.append(unit)
    check(
        "all_fifteen_exact_pages_are_embedded",
        not embedded_failures,
        {
            "embedded_count": 15 - len(embedded_failures),
            "failures": embedded_failures,
        },
    )

    compile_errors = {}
    for source in (BUILDER, TRANSCRIBER):
        try:
            py_compile.compile(str(source), doraise=True)
        except Exception as exc:
            compile_errors[source.name] = str(exc)
    check(
        "builder_and_transcriber_compile",
        not compile_errors,
        {"errors": compile_errors},
    )

    self_test = subprocess.run(
        [sys.executable, str(TRANSCRIBER), "--self-test"],
        cwd=EXP,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        self_payload = json.loads(self_test.stdout)
    except json.JSONDecodeError:
        self_payload = {}
    check(
        "transcriber_template_self_test",
        self_test.returncode == 0
        and self_payload.get("status") == "PASS"
        and self_payload.get("checks") == 5
        and self_payload.get("production_files_written") is False,
        {
            "returncode": self_test.returncode,
            "payload": self_payload,
            "stderr": self_test.stderr,
        },
    )

    production_before = {
        str(path): path.exists()
        for path in (COMPLETED, PRODUCTION_CSV)
    }
    with tempfile.TemporaryDirectory(
        prefix="hcp379_v6_numbers_validation_"
    ) as temporary:
        temp = Path(temporary)
        completed = temp / "completed.numbers"
        shutil.copy2(WORKBOOK, completed)
        synthetic = Document(completed)
        synthetic_review = synthetic.sheets["Review"].tables[0]
        for row in range(1, 16):
            for col in (1, 2, 3):
                synthetic_review.write(row, col, "PASS")
            synthetic_review.write(row, 4, "Synthetic Reviewer")
            synthetic_review.write(
                row,
                5,
                "2026-07-28T09:45:00+05:30",
            )
        synthetic.save(completed)
        synthetic_csv = temp / "review.csv"
        synthetic_receipt = temp / "receipt.json"
        positive = subprocess.run(
            [
                sys.executable,
                str(TRANSCRIBER),
                "--numbers",
                str(completed),
                "--output-csv",
                str(synthetic_csv),
                "--receipt",
                str(synthetic_receipt),
                "--transcribe",
            ],
            cwd=EXP,
            capture_output=True,
            text=True,
            check=False,
        )
        receipt = (
            json.loads(synthetic_receipt.read_text(encoding="utf-8"))
            if synthetic_receipt.is_file()
            else {}
        )
        with (
            synthetic_csv.open(newline="", encoding="utf-8")
            if synthetic_csv.is_file()
            else open("/dev/null", newline="", encoding="utf-8")
        ) as handle:
            output_rows = list(csv.DictReader(handle))
        check(
            "isolated_all_pass_transcription",
            positive.returncode == 0
            and receipt.get("status") == "READY_FOR_PROMOTION_PREFLIGHT"
            and len(output_rows) == 15
            and receipt.get("structure", {}).get(
                "embedded_panels_exact"
            )
            is True,
            {
                "returncode": positive.returncode,
                "status": receipt.get("status"),
                "output_rows": len(output_rows),
                "stderr": positive.stderr,
            },
        )

        hostile = temp / "hostile.numbers"
        shutil.copy2(WORKBOOK, hostile)
        hostile_doc = Document(hostile)
        hostile_review = hostile_doc.sheets["Review"].tables[0]
        for row in range(1, 16):
            for col in (1, 2, 3):
                hostile_review.write(row, col, "PASS")
            hostile_review.write(row, 4, "Synthetic Reviewer")
            hostile_review.write(
                row,
                5,
                "2026-07-28T09:45:00+05:30",
            )
        hostile_review.write(1, 1, "FAIL")
        hostile_doc.save(hostile)
        hostile_result = subprocess.run(
            [
                sys.executable,
                str(TRANSCRIBER),
                "--numbers",
                str(hostile),
            ],
            cwd=EXP,
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            hostile_payload = json.loads(hostile_result.stdout)
        except json.JSONDecodeError:
            hostile_payload = {}
        check(
            "hostile_carried_decision_mutation_is_rejected",
            hostile_result.returncode == 2
            and hostile_payload.get("status")
            == "INCOMPLETE_OR_INVALID_HUMAN_REVIEW"
            and any(
                "carried T1/5TT" in error
                for error in hostile_payload.get("assessment", {}).get(
                    "errors", []
                )
            ),
            {
                "returncode": hostile_result.returncode,
                "payload": hostile_payload,
                "stderr": hostile_result.stderr,
            },
        )

    production_after = {
        str(path): path.exists()
        for path in (COMPLETED, PRODUCTION_CSV)
    }
    check(
        "validation_writes_no_production_review_or_imaging",
        production_before == production_after
        and not any(production_after.values()),
        {
            "before": production_before,
            "after": production_after,
            "imaging_executed_by_validation": False,
        },
    )

    passed = sum(
        value["status"] == "PASS" for value in checks.values()
    )
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_v6_numbers_review_workflow_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "imaging_executed_by_validation": False,
        "tractography_started": False,
        "matrix_generation_started": False,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "records": {
            "v6_manifest": file_record(V6_MANIFEST),
            "workbook_manifest": file_record(WORKBOOK_MANIFEST),
            "workbook": file_record(WORKBOOK),
            "builder": file_record(BUILDER),
            "transcriber": file_record(TRANSCRIBER),
            "validator": file_record(Path(__file__)),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
