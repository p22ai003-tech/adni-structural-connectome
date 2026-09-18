#!/usr/bin/env python3
"""Materialize the exact V6 chat attestation as a completed Numbers review.

This is a narrow, non-overwriting bridge from a frozen human attestation to
the existing fail-closed Numbers transcription contract. It does not promote
a route, run imaging, or change any embedded review panel or provenance row.
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import transcribe_hcp379_v6_numbers_review as contract  # noqa: E402

ATTESTATION = (
    contract.V6_ROOT
    / "human_visual_qc_recovery4_candidate_v6_chat_attestation.json"
)
OUTPUT = contract.DEFAULT_COMPLETED
REVIEW_NOTE = (
    "Human reviewer confirmed exact V6 PDF: all look good "
    "(Codex chat attestation)."
)


def require_file_record(record: dict[str, object], path: Path) -> None:
    if (
        record.get("path") != str(path.resolve())
        or record.get("sha256") != contract.sha256(path)
    ):
        raise RuntimeError(f"attestation file binding differs: {path}")


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(
            f"refusing to overwrite completed review: {OUTPUT}"
        )

    attestation = json.loads(ATTESTATION.read_text(encoding="utf-8"))
    if (
        attestation.get("status") != "HUMAN_REVIEW_ALL_PASS"
        or attestation.get("reviewer") != "Sabeesh Ethiraj"
        or not attestation.get("reviewed_utc")
        or attestation.get("diagnosis_labels_used") is not False
        or attestation.get("human_visual_qc_inferred") is not False
    ):
        raise RuntimeError("chat attestation is not the accepted V6 contract")

    require_file_record(
        attestation["exact_v6_manifest"],
        contract.V6_MANIFEST,
    )
    require_file_record(
        attestation["numbers_template"],
        contract.TEMPLATE_NUMBERS,
    )
    require_file_record(
        attestation["exact_review_pdf"],
        contract.V6_ROOT
        / "hcp379_recovery4_v6_exact_readable_blinded_review.pdf",
    )

    v6, numbers_manifest = contract.load_static()
    document, rows, structure_before = contract.validate_workbook_structure(
        contract.TEMPLATE_NUMBERS,
        v6,
    )
    if numbers_manifest["workbook"]["sha256"] != contract.sha256(
        contract.TEMPLATE_NUMBERS
    ):
        raise RuntimeError("Numbers template binding is stale")

    table = document.sheets["Review"].tables[0]
    reviewer = str(attestation["reviewer"])
    reviewed_utc = str(attestation["reviewed_utc"])
    for row_index, row in enumerate(rows, start=1):
        for col in (1, 2, 3):
            table.write(row_index, col, "PASS")
        table.write(row_index, 4, reviewer)
        table.write(row_index, 5, reviewed_utc)
        prior_note = str(row["Notes"]).strip()
        table.write(
            row_index,
            6,
            f"{prior_note} {REVIEW_NOTE}".strip()
        )

    temporary = OUTPUT.with_name(
        f".{OUTPUT.stem}.tmp-{os.getpid()}.numbers"
    )
    try:
        document.save(temporary)
        with zipfile.ZipFile(temporary) as archive:
            bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Numbers ZIP member failed CRC: {bad_member}")

        _, completed_rows, structure_after = (
            contract.validate_workbook_structure(temporary, v6)
        )
        assessment = contract.assess_human_rows(
            completed_rows,
            contract.parse_timestamp(numbers_manifest["generated_utc"]),
        )
        if (
            assessment["complete"] is not True
            or assessment["all_three_panel_families_pass"] is not True
            or assessment["reviewers"] != [reviewer]
        ):
            raise RuntimeError(
                f"materialized human review failed contract: {assessment}"
            )
        temporary.replace(OUTPUT)
    finally:
        if temporary.exists():
            temporary.unlink()

    result = {
        "status": "COMPLETED_V6_NUMBERS_REVIEW_MATERIALIZED",
        "source_attestation": contract.file_record(ATTESTATION),
        "source_template": contract.file_record(contract.TEMPLATE_NUMBERS),
        "completed_numbers": contract.file_record(OUTPUT),
        "reviewer": reviewer,
        "reviewed_utc": reviewed_utc,
        "decision": "PASS",
        "unit_count": 15,
        "panel_decision_count": 45,
        "structure_before": structure_before,
        "structure_after": structure_after,
        "assessment": assessment,
        "diagnosis_labels_used": False,
        "imaging_started": False,
        "route_promoted": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
