#!/usr/bin/env python3
"""Validate exact V6 human QC and non-overwritingly promote the 009 route.

The submitted round-one Numbers workbook is valid evidence for the T1 and 5TT
panels, but its HCP379 column was based on the superseded scalar-ID rendering.
This program therefore accepts only a completed copy of the exact-input V6
panelwise CSV.  Every corrected HCP379 panel must be a genuine human PASS, and
all three panels for 009_S_4324_I1186579 must be PASS.

With ``--promote`` the program writes:

* an immutable canonical 15-row human-QC CSV;
* a schema-compatible promoted 009 pretract unit manifest;
* a promoted 15-unit pretract master (the other 14 routes stay byte-bound);
* a promotion receipt resolving only the quantitative 009 hard hold.

It never starts tractography or matrix generation and never overwrites an
existing promotion or canonical human-QC artifact.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path("/data/derivatives/hcp379_v2")
EXP = Path("/home/ec2-user/exp")
TARGET = "009_S_4324_I1186579"
ROUTE = "bbr_initialized_syn_conservative"
ORIGINAL_ROUTE = "recovery3_fsl_bbr_direct_hcp_adjudication"

ORIGINAL_MASTER = (
    ROOT / "pretract_recovery4/pretract_recovery4_manifest.json"
)
CANDIDATE_MASTER = (
    ROOT
    / "pretract_candidate_recovery4_v4/"
    "pretract_candidate_recovery4_v4_manifest.json"
)
V6_ROOT = ROOT / "review_recovery4_candidate_overlay_v6"
V6_MANIFEST = (
    V6_ROOT
    / "hcp379_visual_review_pack_recovery4_candidate_v6_manifest.json"
)
V6_TEMPLATE = (
    V6_ROOT / "human_visual_qc_recovery4_candidate_v6_template.csv"
)
DEFAULT_REVIEW = (
    V6_ROOT / "human_visual_qc_recovery4_candidate_v6.csv"
)
V5_CANDIDATE_MANIFEST = (
    ROOT
    / "review_recovery4_candidate_overlay_v5/"
    "hcp379_visual_review_pack_recovery4_candidate_v5_manifest.json"
)
ROUND1_TRANSCRIPTION = (
    ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_round1_transcription_manifest.json"
)
QUANTITATIVE_AUDIT = (
    ROOT
    / "review_recovery4_corrected_overlay/"
    "hcp379_corrected_overlay_quantitative_audit.json"
)
CANDIDATE_VALIDATION = (
    EXP
    / "research_audit/outputs/hcp379_009_candidate_pretract_v4/"
    "validation.json"
)
V6_REVIEW_VALIDATION = (
    EXP
    / "research_audit/outputs/hcp379_visual_review_recovery4_v6/"
    "validation.json"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT / "pretract_promoted_recovery4_v6"
)
DEFAULT_CANONICAL_HUMAN_QC = (
    ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
PROMOTED_UNIT_RELATIVE = (
    Path("subjects")
    / TARGET
    / "pretract_promoted_recovery4_v6_manifest.json"
)
PROMOTED_MASTER_NAME = "pretract_promoted_recovery4_v6_manifest.json"
PROMOTION_RECEIPT_NAME = "promotion_receipt.json"

PANEL_FIELDS = (
    "B0 vsT1",
    "B0 vs 5TT",
    "B0 vs HCP379",
)
REQUIRED_FIELDS = {
    "unit",
    *PANEL_FIELDS,
    "reviewer",
    "reviewed_utc",
}
FORBIDDEN_FIELDS = {
    "diagnosis",
    "diagnosis_at_dti",
    "group",
    "research_group",
    "outcome",
    "cdr",
    "mmse",
}


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


def future_file_record(source: Path, final_path: Path) -> dict[str, Any]:
    record = file_record(source)
    record["path"] = str(final_path.expanduser().resolve())
    return record


def verify_file_record(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if file_record(path) != dict(record):
        raise ValueError(f"{label}: artifact binding differs")
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            dict(value),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def parse_utc(value: str, *, unit: str) -> tuple[datetime, str]:
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
    except ValueError as exc:
        raise ValueError(
            f"{unit}: reviewed_utc is not ISO-8601: {text!r}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            f"{unit}: reviewed_utc lacks an explicit timezone"
        )
    normalized = parsed.astimezone(timezone.utc)
    if normalized > datetime.now(timezone.utc) + timedelta(minutes=2):
        raise ValueError(f"{unit}: reviewed_utc is in the future")
    return normalized, normalized.isoformat().replace("+00:00", "Z")


def validate_static_inputs() -> dict[str, Any]:
    for path in (
        ORIGINAL_MASTER,
        CANDIDATE_MASTER,
        V6_MANIFEST,
        V6_TEMPLATE,
        V5_CANDIDATE_MANIFEST,
        ROUND1_TRANSCRIPTION,
        QUANTITATIVE_AUDIT,
        CANDIDATE_VALIDATION,
        V6_REVIEW_VALIDATION,
    ):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)

    original = load_json(ORIGINAL_MASTER)
    candidate = load_json(CANDIDATE_MASTER)
    review = load_json(V6_MANIFEST)
    round1 = load_json(ROUND1_TRANSCRIPTION)
    audit = load_json(QUANTITATIVE_AUDIT)
    validation = load_json(CANDIDATE_VALIDATION)
    review_validation = load_json(V6_REVIEW_VALIDATION)

    original_units = {
        str(row.get("unit")): row for row in original.get("units", [])
    }
    candidate_units = {
        str(row.get("unit")): row for row in candidate.get("units", [])
    }
    review_units = {
        str(row.get("unit")): row for row in review.get("units", [])
    }
    if (
        original.get("record_type")
        != "diagnosis_blind_hcp379_pretract_recovery4_manifest"
        or original.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or original.get("diagnosis_labels_used") is not False
        or original.get("human_visual_qc_inferred") is not False
        or original.get("unit_count") != 15
        or original.get("tractography_started") is not False
        or original.get("matrix_generation_started") is not False
        or len(original_units) != 15
    ):
        raise ValueError("original Recovery4 pretract master differs")
    if (
        candidate.get("record_type")
        != (
            "diagnosis_blind_hcp379_pretract_candidate_recovery4_v4_"
            "manifest"
        )
        or candidate.get("status")
        != "PROVISIONAL_AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or candidate.get("diagnosis_labels_used") is not False
        or candidate.get("human_visual_qc_inferred") is not False
        or candidate.get("unit_count") != 15
        or candidate.get("tractography_started") is not False
        or candidate.get("matrix_generation_started") is not False
        or candidate.get("route_substitution", {}).get("unit") != TARGET
        or candidate.get("route_substitution", {}).get(
            "candidate_route"
        )
        != ROUTE
        or candidate.get("route_substitution", {}).get(
            "promotion_status"
        )
        != "NOT_PROMOTED"
        or set(candidate_units) != set(original_units)
    ):
        raise ValueError("candidate Recovery4 pretract master differs")
    if (
        review.get("record_type")
        != (
            "diagnosis_blind_hcp379_visual_review_pack_"
            "recovery4_candidate_v6"
        )
        or review.get("status")
        != "READY_FOR_EXACT_INPUT_HUMAN_REVIEW"
        or review.get("diagnosis_labels_used") is not False
        or review.get("human_visual_qc_inferred") is not False
        or review.get("unit_count") != 15
        or review.get("tractography_started") is not False
        or review.get("matrix_generation_started") is not False
        or review.get("route_substitution", {}).get("unit") != TARGET
        or review.get("route_substitution", {}).get("candidate_route")
        != ROUTE
        or review.get("route_substitution", {}).get("promotion_status")
        != "NOT_PROMOTED"
        or review.get("route_substitution", {}).get(
            "review_panels_bound_to_exact_candidate_inputs"
        )
        is not True
        or set(review_units) != set(original_units)
        or review.get("candidate_pretract_master")
        != file_record(CANDIDATE_MASTER)
        or review.get("human_qc_template")
        != file_record(V6_TEMPLATE)
    ):
        raise ValueError("exact-input V6 review manifest differs")
    if (
        round1.get("record_type")
        != "hcp379_recovery4_human_review_round1_transcription"
        or round1.get("status")
        != "HCP379_COLUMN_SUPERSEDED_BY_CORRECTED_OVERLAY_REVIEW"
        or round1.get("unit_count") != 15
        or round1.get("b0_vs_t1_counts", {}).get("PASS") != 15
        or round1.get("b0_vs_5tt_counts", {}).get("PASS") != 15
        or round1.get("canonical_gate_created") is not False
    ):
        raise ValueError("round-one human transcription differs")
    if (
        audit.get("record_type")
        != (
            "diagnosis_blind_hcp379_recovery4_corrected_overlay_"
            "quantitative_audit"
        )
        or audit.get("diagnosis_labels_used") is not False
        or audit.get("unit_count") != 15
        or audit.get("all_units_have_379_labels") is not True
        or audit.get("hard_hold_units") != [TARGET]
    ):
        raise ValueError("corrected-overlay quantitative audit differs")
    if (
        validation.get("record_type")
        != "hcp379_009_candidate_pretract_recovery4_v4_validation"
        or validation.get("status") != "PASS"
        or validation.get("diagnosis_labels_used") is not False
        or validation.get("tractography_started") is not False
        or validation.get("matrix_generation_started") is not False
        or validation.get("passed_checks") != validation.get(
            "total_checks"
        )
        or validation.get("passed_checks") != 11
        or validation.get("records", {}).get("candidate_master")
        != file_record(CANDIDATE_MASTER)
        or validation.get("records", {}).get("review_manifest")
        != file_record(V5_CANDIDATE_MANIFEST)
    ):
        raise ValueError("independent candidate validation differs")
    if (
        review_validation.get("record_type")
        != "hcp379_visual_review_recovery4_v6_validation"
        or review_validation.get("status") != "PASS"
        or review_validation.get("diagnosis_labels_used") is not False
        or review_validation.get("imaging_executed_by_validation")
        is not False
        or review_validation.get("tractography_started") is not False
        or review_validation.get("matrix_generation_started") is not False
        or review_validation.get("passed_checks") != 9
        or review_validation.get("total_checks") != 9
        or review_validation.get("records", {}).get("v6_manifest")
        != file_record(V6_MANIFEST)
    ):
        raise ValueError("independent V6 review validation differs")

    for unit, row in candidate_units.items():
        manifest_path = verify_file_record(
            row.get("manifest", {}), f"{unit}:candidate unit manifest"
        )
        manifest = load_json(manifest_path)
        expected_status = (
            "PROVISIONAL_AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
            if unit == TARGET
            else "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        )
        if (
            manifest.get("unit") != unit
            or manifest.get("status") != expected_status
            or manifest.get("diagnosis_labels_used") is not False
            or manifest.get("human_visual_qc_inferred") is not False
        ):
            raise ValueError(f"{unit}: candidate unit manifest differs")
        if unit != TARGET and row != original_units[unit]:
            raise ValueError(f"{unit}: unchanged route is not byte-bound")
    target_manifest_path = verify_file_record(
        candidate_units[TARGET]["manifest"],
        f"{TARGET}:candidate unit manifest",
    )
    target_manifest = load_json(target_manifest_path)
    if (
        target_manifest.get("record_type")
        != (
            "diagnosis_blind_hcp379_pretract_candidate_recovery4_v4_"
            "unit_manifest"
        )
        or target_manifest.get("registration", {}).get("route") != ROUTE
        or target_manifest.get("tractography_started") is not False
        or target_manifest.get("matrix_generation_started") is not False
        or target_manifest.get("registration", {}).get(
            "left_cortical_inside_dwi_mask_fraction", 0.0
        )
        < 0.80
        or target_manifest.get("registration", {}).get(
            "right_cortical_inside_dwi_mask_fraction", 0.0
        )
        < 0.80
    ):
        raise ValueError("009 candidate route no longer clears hard gates")
    for row in review_units.values():
        verify_file_record(row.get("review_page", {}), "V6 review page")
        for name, record in row.get("source_panels", {}).items():
            verify_file_record(record, f"V6 source panel: {name}")

    generated_utc, _ = parse_utc(
        str(review.get("generated_utc", "")), unit="V6 review pack"
    )
    return {
        "original": original,
        "candidate": candidate,
        "review": review,
        "candidate_units": candidate_units,
        "review_units": review_units,
        "target_manifest": target_manifest,
        "target_manifest_path": target_manifest_path,
        "review_generated_utc": generated_utc,
    }


def validate_review_csv(
    path: Path,
    *,
    expected_units: set[str],
    review_generated_utc: datetime,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    path = path.expanduser().resolve()
    if path == V6_TEMPLATE.resolve():
        raise ValueError(
            "save a completed copy of the V6 template; do not edit or "
            "submit the immutable blank template itself"
        )
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    if sha256_file(path) == sha256_file(V6_TEMPLATE):
        raise ValueError("submitted review is still the blank V6 template")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(str(field or "").strip() for field in (
            reader.fieldnames or ()
        ))
        normalized_fields = {field.lower() for field in fields}
        if FORBIDDEN_FIELDS.intersection(normalized_fields):
            raise ValueError("human review contains a clinical blinding field")
        if not REQUIRED_FIELDS.issubset(fields):
            missing = sorted(REQUIRED_FIELDS - set(fields))
            raise ValueError(
                "human review lacks required fields: " + ",".join(missing)
            )
        rows = [
            {
                str(key or "").strip(): str(value or "").strip()
                for key, value in row.items()
            }
            for row in reader
            if any(str(value or "").strip() for value in row.values())
        ]
    units = [row.get("unit", "") for row in rows]
    if (
        len(rows) != 15
        or len(set(units)) != 15
        or set(units) != expected_units
    ):
        raise ValueError("human review rows differ from the exact 15 units")

    failures: list[str] = []
    normalized_rows: list[dict[str, str]] = []
    latest_review = review_generated_utc
    for row in rows:
        unit = row["unit"]
        reviewer = row.get("reviewer", "").strip()
        if not reviewer:
            failures.append(f"{unit}: reviewer is blank")
        reviewed_at, reviewed_utc = parse_utc(
            row.get("reviewed_utc", ""), unit=unit
        )
        if reviewed_at < review_generated_utc:
            failures.append(
                f"{unit}: review predates the exact-input V6 pack"
            )
        latest_review = max(latest_review, reviewed_at)
        panel_statuses = {
            field: row.get(field, "").strip().upper()
            for field in PANEL_FIELDS
        }
        for field, status in panel_statuses.items():
            if status != "PASS":
                failures.append(
                    f"{unit}: {field} is {status or 'BLANK'}, not PASS"
                )
        normalized_rows.append(
            {
                "unit": unit,
                **panel_statuses,
                "reviewer": reviewer,
                "reviewed_utc": reviewed_utc,
                "Notes": row.get("Notes", row.get("notes", "")).strip(),
            }
        )
    if failures:
        raise ValueError(
            "exact-input human review is not promotable: "
            + " | ".join(failures)
        )
    normalized_rows.sort(key=lambda row: row["unit"])
    return normalized_rows, {
        "unit_count": len(normalized_rows),
        "all_t1_pass": True,
        "all_5tt_pass": True,
        "all_hcp379_pass": True,
        "target_all_three_pass": True,
        "reviewers": sorted(
            {row["reviewer"] for row in normalized_rows}
        ),
        "latest_reviewed_utc": latest_review.isoformat().replace(
            "+00:00", "Z"
        ),
    }


def write_canonical_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
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
        for row in rows:
            writer.writerow(
                {
                    "unit": row["unit"],
                    "status": "PASS",
                    "reviewer": row["reviewer"],
                    "reviewed_utc": row["reviewed_utc"],
                    "notes": (
                        "Exact-input V6 panels PASS. "
                        + row.get("Notes", "")
                    ).strip(),
                }
            )


def promote(
    *,
    review_csv: Path,
    rows: list[dict[str, str]],
    review_summary: dict[str, Any],
    static: dict[str, Any],
    output_root: Path,
    canonical_human_qc: Path,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    canonical_human_qc = canonical_human_qc.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"promotion root exists: {output_root}")
    if canonical_human_qc.exists():
        raise FileExistsError(
            f"canonical human-QC file exists: {canonical_human_qc}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    canonical_human_qc.parent.mkdir(parents=True, exist_ok=True)
    promotion_utc = utc_now()
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging.",
            dir=output_root.parent,
        )
    )
    canonical_temporary = Path(
        tempfile.mkstemp(
            prefix=f".{canonical_human_qc.name}.",
            suffix=".tmp",
            dir=canonical_human_qc.parent,
        )[1]
    )
    canonical_temporary.unlink()
    canonical_installed = False
    try:
        write_canonical_csv(canonical_temporary, rows)
        canonical_record = future_file_record(
            canonical_temporary, canonical_human_qc
        )
        promoted_unit_path = staging_root / PROMOTED_UNIT_RELATIVE
        final_promoted_unit_path = output_root / PROMOTED_UNIT_RELATIVE
        target_manifest = copy.deepcopy(static["target_manifest"])
        target_manifest.update(
            {
                "schema_version": "1.1.0",
                "record_type": (
                    "diagnosis_blind_hcp379_pretract_recovery4_"
                    "unit_manifest"
                ),
                "status": "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC",
                "completed_utc": promotion_utc,
                "diagnosis_labels_used": False,
                "human_visual_qc_inferred": False,
                "tractography_started": False,
                "matrix_generation_started": False,
                "promotion": {
                    "status": "HUMAN_VISUAL_QC_PASS_ROUTE_PROMOTED",
                    "promotion_utc": promotion_utc,
                    "unit": TARGET,
                    "old_route": ORIGINAL_ROUTE,
                    "promoted_route": ROUTE,
                    "source_candidate_unit_manifest": file_record(
                        static["target_manifest_path"]
                    ),
                    "source_candidate_master": file_record(
                        CANDIDATE_MASTER
                    ),
                    "exact_input_review_manifest": file_record(
                        V6_MANIFEST
                    ),
                    "panelwise_human_review": file_record(review_csv),
                    "canonical_human_qc": canonical_record,
                    "quantitative_hard_hold_resolved": True,
                    "tractography_started_by_promotion": False,
                    "matrix_generation_started_by_promotion": False,
                },
            }
        )
        write_json(promoted_unit_path, target_manifest)
        promoted_unit_record = future_file_record(
            promoted_unit_path, final_promoted_unit_path
        )

        promoted_master = copy.deepcopy(static["candidate"])
        promoted_master.update(
            {
                "schema_version": "1.1.0",
                "record_type": (
                    "diagnosis_blind_hcp379_pretract_recovery4_manifest"
                ),
                "status": "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC",
                "generated_utc": promotion_utc,
                "diagnosis_labels_used": False,
                "human_visual_qc_inferred": False,
                "tractography_started": False,
                "matrix_generation_started": False,
                "source_candidate_master": file_record(CANDIDATE_MASTER),
                "exact_input_review_manifest": file_record(V6_MANIFEST),
                "panelwise_human_review": file_record(review_csv),
                "canonical_human_qc": canonical_record,
                "human_review_summary": review_summary,
            }
        )
        promoted_master["route_substitution"] = {
            "unit": TARGET,
            "old_route": ORIGINAL_ROUTE,
            "candidate_route": ROUTE,
            "promotion_status": "HUMAN_VISUAL_QC_PASS_PROMOTED",
            "human_review_required": True,
            "exact_input_review_complete": True,
        }
        promoted_rows = copy.deepcopy(promoted_master["units"])
        target_rows = [
            row for row in promoted_rows if row.get("unit") == TARGET
        ]
        if len(target_rows) != 1:
            raise ValueError("candidate master does not contain one 009 row")
        target_rows[0].update(
            {
                "manifest": promoted_unit_record,
                "registration_route": ROUTE,
                "status": "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC",
                "promotion_status": "HUMAN_VISUAL_QC_PASS_PROMOTED",
            }
        )
        promoted_master["units"] = promoted_rows
        master_path = staging_root / PROMOTED_MASTER_NAME
        final_master_path = output_root / PROMOTED_MASTER_NAME
        write_json(master_path, promoted_master)
        master_record = future_file_record(master_path, final_master_path)

        receipt = {
            "schema_version": "1.0.0",
            "record_type": (
                "diagnosis_blind_hcp379_recovery4_v6_promotion_receipt"
            ),
            "status": "PASS",
            "generated_utc": promotion_utc,
            "diagnosis_labels_used": False,
            "human_visual_qc_inferred": False,
            "tractography_started": False,
            "matrix_generation_started": False,
            "unit_count": 15,
            "route_promotion": {
                "unit": TARGET,
                "old_route": ORIGINAL_ROUTE,
                "promoted_route": ROUTE,
            },
            "hard_hold_resolution": {
                "source_hard_hold_units": [TARGET],
                "resolved_units": [TARGET],
                "remaining_hard_hold_units": [],
            },
            "review_summary": review_summary,
            "records": {
                "original_pretract_master": file_record(ORIGINAL_MASTER),
                "candidate_pretract_master": file_record(
                    CANDIDATE_MASTER
                ),
                "candidate_validation": file_record(
                    CANDIDATE_VALIDATION
                ),
                "quantitative_audit": file_record(QUANTITATIVE_AUDIT),
                "round1_transcription": file_record(
                    ROUND1_TRANSCRIPTION
                ),
                "exact_input_review_manifest": file_record(V6_MANIFEST),
                "review_pack_validation": file_record(
                    V6_REVIEW_VALIDATION
                ),
                "panelwise_human_review": file_record(review_csv),
                "canonical_human_qc": canonical_record,
                "promoted_unit_manifest": promoted_unit_record,
                "promoted_pretract_master": master_record,
                "promotion_builder": file_record(Path(__file__)),
            },
        }
        receipt_path = staging_root / PROMOTION_RECEIPT_NAME
        write_json(receipt_path, receipt)

        os.link(canonical_temporary, canonical_human_qc)
        canonical_installed = True
        canonical_temporary.unlink()
        staging_root.rename(output_root)

        for record in (
            canonical_record,
            promoted_unit_record,
            master_record,
        ):
            verify_file_record(record, "installed promotion artifact")
        installed_receipt = output_root / PROMOTION_RECEIPT_NAME
        installed = load_json(installed_receipt)
        if installed != receipt:
            raise ValueError("installed promotion receipt differs")
        return {
            "status": "PASS_PROMOTED_NO_IMAGING_STARTED",
            "promotion_root": str(output_root),
            "promoted_pretract_master": str(final_master_path),
            "promotion_receipt": str(installed_receipt),
            "canonical_human_qc": str(canonical_human_qc),
            "promoted_unit": TARGET,
            "promoted_route": ROUTE,
            "other_routes_unchanged": 14,
            "tractography_started": False,
            "matrix_generation_started": False,
        }
    except Exception:
        if canonical_installed and canonical_human_qc.is_file():
            expected_hash = (
                sha256_file(canonical_temporary)
                if canonical_temporary.is_file()
                else None
            )
            if expected_hash is None or (
                sha256_file(canonical_human_qc) == expected_hash
            ):
                canonical_human_qc.unlink()
        raise
    finally:
        if canonical_temporary.exists():
            canonical_temporary.unlink()
        if staging_root.exists():
            shutil.rmtree(staging_root)


def self_test() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    parsed, normalized = parse_utc(
        now.isoformat(), unit="SELF_TEST"
    )
    if parsed.tzinfo is None or not normalized.endswith("Z"):
        raise AssertionError("UTC normalization failed")
    try:
        parse_utc("2026-07-28T07:23:00", unit="SELF_TEST")
    except ValueError:
        pass
    else:
        raise AssertionError("timezone-naive timestamp was accepted")
    static = validate_static_inputs()
    with tempfile.TemporaryDirectory(
        prefix="hcp379_v6_promotion_self_test."
    ) as directory:
        root = Path(directory)
        review_csv = root / "synthetic_review.csv"
        with review_csv.open(
            "x", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "unit",
                    *PANEL_FIELDS,
                    "reviewer",
                    "reviewed_utc",
                    "Notes",
                ),
            )
            writer.writeheader()
            for unit in sorted(static["candidate_units"]):
                writer.writerow(
                    {
                        "unit": unit,
                        **{field: "PASS" for field in PANEL_FIELDS},
                        "reviewer": "SYNTHETIC_SELF_TEST_NOT_HUMAN",
                        "reviewed_utc": utc_now(),
                        "Notes": "Synthetic integration test only.",
                    }
                )
        rows, summary = validate_review_csv(
            review_csv,
            expected_units=set(static["candidate_units"]),
            review_generated_utc=static["review_generated_utc"],
        )
        output_root = root / "promotion"
        canonical = root / "canonical.csv"
        result = promote(
            review_csv=review_csv,
            rows=rows,
            review_summary=summary,
            static=static,
            output_root=output_root,
            canonical_human_qc=canonical,
        )
        if (
            result.get("status") != "PASS_PROMOTED_NO_IMAGING_STARTED"
            or not canonical.is_file()
            or not (
                output_root / PROMOTED_MASTER_NAME
            ).is_file()
            or not (
                output_root / PROMOTION_RECEIPT_NAME
            ).is_file()
        ):
            raise AssertionError("isolated promotion integration test failed")
    return {
        "status": "PASS",
        "checks": 3,
        "imaging_started": False,
        "production_files_written": False,
        "isolated_promotion_tested": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--canonical-human-qc",
        type=Path,
        default=DEFAULT_CANONICAL_HUMAN_QC,
    )
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return 0

    static = validate_static_inputs()
    if not args.review_csv.expanduser().is_file():
        print(
            json.dumps(
                {
                    "status": "AWAITING_COMPLETED_V6_HUMAN_REVIEW",
                    "review_csv_expected": str(
                        args.review_csv.expanduser().resolve()
                    ),
                    "exact_review_pdf": str(
                        V6_ROOT
                        / (
                            "hcp379_recovery4_v6_exact_readable_"
                            "blinded_review.pdf"
                        )
                    ),
                    "template": str(V6_TEMPLATE),
                    "promotion_executed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    rows, review_summary = validate_review_csv(
        args.review_csv,
        expected_units=set(static["candidate_units"]),
        review_generated_utc=static["review_generated_utc"],
    )
    if not args.promote:
        print(
            json.dumps(
                {
                    "status": "READY_TO_PROMOTE_NO_WRITES",
                    "review_csv": file_record(args.review_csv),
                    "review_summary": review_summary,
                    "output_root": str(args.output_root.resolve()),
                    "canonical_human_qc": str(
                        args.canonical_human_qc.resolve()
                    ),
                    "promotion_executed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = promote(
        review_csv=args.review_csv.expanduser().resolve(),
        rows=rows,
        review_summary=review_summary,
        static=static,
        output_root=args.output_root,
        canonical_human_qc=args.canonical_human_qc,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
