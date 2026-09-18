#!/usr/bin/env python3
"""Validate a fresh exact-date ADNI/LONI metadata handoff.

This command is metadata-only and fail-closed. It never downloads imaging,
never authorizes a download, never edits the supplied source file, and refuses
to write into a non-empty output directory.

The validator is intentionally separate from ``build_visit_pair_manifest``:
this stage establishes source provenance, exact-day precision, request-roster
coverage, duplicate consistency, and QC traceability before a catalog can be
used by the deterministic pairing rule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import numbers
import os
import re
import shutil
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from research_audit.build_visit_pair_manifest import (
    ALIASES as PAIR_ALIASES,
    T1_RE,
    file_sha256,
    normalized_name,
)


SUBJECT_RE = re.compile(r"^\d{3}_S_\d{4}$")
ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
US_DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
DICOM_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ALIASES = {
    **PAIR_ALIASES,
    "subject_id": [*PAIR_ALIASES["subject_id"], "ParticipantID", "RID/PTID"],
    "image_id": [*PAIR_ALIASES["image_id"], "LONIImage", "LONI Image ID"],
    "study_date": [*PAIR_ALIASES["study_date"], "MRIQC StudyDate"],
    "visit": [*PAIR_ALIASES["visit"], "VISCODE2"],
    "phase": [*PAIR_ALIASES["phase"], "ADNI Phase", "PHASE"],
    "description": [*PAIR_ALIASES["description"], "SeriesDescription"],
    # Image Type is not reliably the Original/Processed field in IDA exports.
    "type": ["Type", "SeriesType", "SERIES_TYPE"],
    "qc": [*PAIR_ALIASES["qc"], "SeriesQC", "SERIES_QUALITY"],
    "qc_source": [
        "QC Source",
        "Image QC Source",
        "Quality Source",
        "QC Provenance",
    ],
    "series_uid": [*PAIR_ALIASES["series_uid"], "SeriesInstanceUID"],
    "protocol": [*PAIR_ALIASES["protocol"], "SeriesProtocol", "COLPROT"],
    "mri_protocol_phase": ["MRIProtocolPhase", "MRI Protocol Phase"],
    "manufacturer": [*PAIR_ALIASES["manufacturer"], "ScannerManufacturer"],
    "model": [*PAIR_ALIASES["model"], "ScannerModel"],
    "field_strength": [*PAIR_ALIASES["field_strength"], "MagneticFieldStrength"],
    "site": ["Site", "Site ID", "SITEID", "Center", "Center ID"],
    "modality": ["Modality", "Image Modality"],
}

REQUIRED_FIELDS = (
    "subject_id",
    "image_id",
    "study_date",
    "visit",
    "phase",
    "description",
    "type",
    "qc",
)
OPTIONAL_FIELDS = (
    "qc_source",
    "series_uid",
    "protocol",
    "mri_protocol_phase",
    "manufacturer",
    "model",
    "field_strength",
    "site",
    "modality",
)
CONFLICT_FIELDS = (
    "subject_id",
    "study_date",
    "visit",
    "phase",
    "description",
    "type",
    "qc",
    "qc_source",
    "series_uid",
    "protocol",
    "mri_protocol_phase",
    "manufacturer",
    "model",
    "field_strength",
    "site",
    "modality",
)

QC_PROFILES = {
    # Current MRIQC data-dictionary semantics. Numeric values outside this
    # documented set fail rather than being guessed.
    "adni_mriqc_current": {
        "pass": {"1"},
        "fail": {"4"},
        "unknown": {"-1"},
    },
    # Historical Mayo SERIES_QUALITY grades: 1-3 acceptable, 4 unusable;
    # -1 is not assessed and remains ineligible rather than silently passing.
    "adni_mayo_series_quality": {
        "pass": {"1", "2", "3"},
        "fail": {"4"},
        "unknown": {"-1"},
    },
    "generic_text": {
        "pass": {"pass", "passed", "acceptable", "accepted", "yes", "y", "true"},
        "fail": {"fail", "failed", "unacceptable", "rejected", "no", "n", "false"},
        "unknown": {"unknown", "pending", "not assessed", "not available", "unrated", "n/a", "na"},
    },
}

# QC code sets are schema-specific, not interchangeable numeric vocabularies.
# Bind reserved ADNI fields to narrowly enumerated official table/field source
# identifiers, so choosing another profile cannot silently reinterpret values.
QC_PROFILE_BINDINGS = {
    "adni_mriqc_current": {
        "columns": {"seriesqc"},
        "column_label": "SeriesQC",
        "sources": {
            "mriqc",
            "mriqcseriesqc",
            "mayoadirlabmriquality",
            "mayoadirlabmriqualitymriqcseriesqc",
        },
        "source_label": "MRIQC.SeriesQC",
    },
    "adni_mayo_series_quality": {
        "columns": {"seriesquality"},
        "column_label": "SERIES_QUALITY",
        "sources": {
            "mayoadirlmriimageqc",
            "mayoadirlmriimageqcseriesquality",
            "mayoadirlmriadni3",
            "mayoadirlmriadni3seriesquality",
            "mayoadirlmriqualityadni3",
            "mayoadirlmriqualityadni3seriesquality",
            "adnigo2mayoadirlmriquality20191106",
            "adnigo2mayoadirlmriquality20191106seriesquality",
            "mayoadirlmrimch",
            "mayoadirlmrimchseriesquality",
            "jacklabadnigo2mriqc",
            "jacklabadnigo2mriqcseriesquality",
            "jacklabadnigo23mriqc",
            "jacklabadnigo23mriqcseriesquality",
        },
        "source_label": "MAYOADIRL_MRI_ADNI3.SERIES_QUALITY or MAYOADIRL_MRI_IMAGEQC.SERIES_QUALITY",
    },
    # Generic textual QC must not be an escape hatch for either reserved ADNI
    # schema. It is limited to genuinely generic column names.
    "generic_text": {
        "columns": {"qc", "qcstatus", "imageqc", "imageqcstatus", "quality"},
        "column_label": "QC/QC Status/Image QC/Image QC Status/Quality",
        "sources": None,
        "source_label": "a nonblank explicitly named textual QC source",
    },
}

CANONICAL_REQUEST_ROSTER_SHA256 = "4a935b7bbd7d6d5ea5c1c8d80f4cc8fde840abaa88c0d4d2419878591bcc2406"
ADNI_EARLIEST_PLAUSIBLE_DATE = date(2000, 1, 1)

SUSPICIOUS_HEADER_TOKENS = (
    "password",
    "passwd",
    "accesskey",
    "accesstoken",
    "secretkey",
    "sessiontoken",
    "devicecode",
    "cookie",
    "authorizationtoken",
)


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_positive_integer(value: Any) -> int | None:
    if pd.isna(value) or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        number = int(value)
        return number if number > 0 else None
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer() or number <= 0:
            return None
        # Beyond this bound, a binary float cannot safely represent every ID.
        if abs(number) > 2**53:
            return None
        return int(number)
    text = clean_text(value)
    match = re.fullmatch(r"[Ii]?(\d+)", text)
    if not match:
        return None
    number = int(match.group(1))
    return number if number > 0 else None


def parse_exact_calendar_date(value: Any) -> str | None:
    """Return ISO date only when a full, valid calendar day is supplied."""

    text = clean_text(value)
    if not text:
        return None
    # Pandas may represent an otherwise integer DICOM date as a float when the
    # source column also contains blanks. Accept only a zero fractional part.
    if re.fullmatch(r"\d{8}\.0+", text):
        text = text.split(".", 1)[0]
    match = ISO_DATE_RE.fullmatch(text)
    if match:
        parts = tuple(map(int, match.groups()))
    else:
        match = US_DATE_RE.fullmatch(text)
        if match:
            month, day_value, year = map(int, match.groups())
            parts = (year, month, day_value)
        else:
            match = DICOM_DATE_RE.fullmatch(text)
            if not match:
                return None
            parts = tuple(map(int, match.groups()))
    try:
        return date(*parts).isoformat()
    except ValueError:
        return None


def parse_qc_strict(value: Any, qc_profile: str) -> str | None:
    token = clean_text(value).lower()
    # Preserve strict code semantics while tolerating pandas' float rendering
    # of integer-coded columns that contain missing values.
    if re.fullmatch(r"-?\d+\.0+", token):
        token = token.split(".", 1)[0]
    profile = QC_PROFILES[qc_profile]
    for status in ("pass", "fail", "unknown"):
        if token in profile[status]:
            return status
    return None


def qc_source_matches_profile(value: Any, qc_profile: str) -> bool:
    """Return whether a named QC source is compatible with its code profile."""

    text = clean_text(value)
    if not text:
        return False
    accepted = QC_PROFILE_BINDINGS[qc_profile]["sources"]
    normalized = normalized_name(text)
    if accepted is not None:
        return normalized in accepted
    reserved_sources = {
        source
        for binding in QC_PROFILE_BINDINGS.values()
        if binding["sources"] is not None
        for source in binding["sources"]
    }
    return normalized not in reserved_sources


def resolve_column_strict(
    frame: pd.DataFrame, field: str, required: bool = False
) -> str | None:
    accepted = {normalized_name(alias) for alias in ALIASES[field]}
    matches = [column for column in frame.columns if normalized_name(column) in accepted]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous {field!r} columns: {matches}. Retain exactly one accepted alias."
        )
    if not matches:
        if required:
            raise ValueError(
                f"Missing required {field!r} column. Accepted aliases: {ALIASES[field]}"
            )
        return None
    return matches[0]


def validate_source_generated_at(value: str) -> str:
    text = value.strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            parsed = datetime.combine(date.fromisoformat(text), datetime.min.time(), tzinfo=timezone.utc)
        else:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            parsed = parsed.astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError("--source-generated-at must be an ISO date or timestamp") from exc
    if parsed > datetime.now(timezone.utc):
        raise ValueError("--source-generated-at cannot be in the future")
    return parsed.isoformat().replace("+00:00", "Z")


def _issue(code: str, message: str, image_id: int | None = None) -> dict[str, Any]:
    return {"code": code, "image_id": image_id, "message": message}


def _unique_nonblank(values: pd.Series) -> list[str]:
    return sorted({clean_text(value) for value in values if clean_text(value)})


def _extra_source_json(group: pd.DataFrame, mapped_columns: set[str]) -> str:
    extra: dict[str, Any] = {}
    for column in sorted(group.columns, key=lambda value: normalized_name(value)):
        if column in mapped_columns or column.startswith("__"):
            continue
        values = _unique_nonblank(group[column])
        if not values:
            continue
        extra[str(column)] = values[0] if len(values) == 1 else values
    return json.dumps(extra, sort_keys=True, separators=(",", ":"))


def _empty_result(
    errors: list[dict[str, Any]], warnings: list[dict[str, Any]], source: dict[str, Any]
) -> dict[str, Any]:
    return {
        "canonical": pd.DataFrame(),
        "coverage": pd.DataFrame(),
        "rejects": pd.DataFrame(errors),
        "validation": {
            "schema_version": "1.0",
            "status": "FAIL",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "counts": {},
            "checks": {"pairing_ready": False},
            "errors": errors,
            "warnings": warnings,
            "safety": {
                "metadata_only": True,
                "approved_for_download_count": 0,
                "images_downloaded": 0,
            },
            "release_status": "INTAKE_FAILED",
        },
    }


def validate_exact_date_catalog(
    catalog: pd.DataFrame,
    roster: pd.DataFrame,
    *,
    source_name: str,
    source_generated_at: str,
    catalog_sha256: str,
    roster_sha256: str,
    absence_ledger: pd.DataFrame | None = None,
    absence_sha256: str | None = None,
    default_qc_source: str | None = None,
    qc_profile: str = "adni_mriqc_current",
    expected_roster_count: int = 4015,
    expected_subject_count: int = 530,
    expected_roster_sha256: str | None = CANONICAL_REQUEST_ROSTER_SHA256,
    expected_catalog_sha256: str | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    source: dict[str, Any] = {
        "name": source_name.strip(),
        "generated_at": source_generated_at,
        "catalog_sha256": catalog_sha256,
        "roster_sha256": roster_sha256,
        "absence_ledger_sha256": absence_sha256,
        "expected_catalog_sha256": expected_catalog_sha256,
        "expected_roster_sha256": expected_roster_sha256,
        "default_qc_source": default_qc_source,
        "qc_profile": qc_profile,
    }

    if not source["name"]:
        errors.append(_issue("missing_source_name", "Source name is blank"))
    try:
        source["generated_at"] = validate_source_generated_at(source_generated_at)
    except ValueError as exc:
        errors.append(_issue("invalid_source_generated_at", str(exc)))
    if not SHA256_RE.fullmatch(catalog_sha256):
        errors.append(_issue("invalid_catalog_sha256", "Catalog SHA-256 is malformed"))
    if not SHA256_RE.fullmatch(roster_sha256):
        errors.append(_issue("invalid_roster_sha256", "Roster SHA-256 is malformed"))
    if expected_roster_sha256:
        expected_roster_hash = expected_roster_sha256.lower()
        if not SHA256_RE.fullmatch(expected_roster_hash):
            errors.append(_issue("invalid_expected_roster_sha256", "Expected roster SHA-256 is malformed"))
        elif expected_roster_hash != roster_sha256:
            errors.append(_issue("roster_sha256_mismatch", "Roster does not match the frozen request universe"))
    if qc_profile not in QC_PROFILES:
        errors.append(_issue("invalid_qc_profile", f"Unknown QC profile: {qc_profile}"))
    if expected_catalog_sha256:
        expected = expected_catalog_sha256.lower()
        if not SHA256_RE.fullmatch(expected):
            errors.append(_issue("invalid_expected_sha256", "Expected SHA-256 is malformed"))
        elif expected != catalog_sha256:
            errors.append(_issue("catalog_sha256_mismatch", "Catalog does not match --expected-sha256"))

    suspicious = [
        str(column)
        for column in catalog.columns
        if any(token in normalized_name(column) for token in SUSPICIOUS_HEADER_TOKENS)
    ]
    if suspicious:
        errors.append(
            _issue(
                "credential_like_headers",
                f"Credential-like columns are forbidden in metadata handoff: {suspicious}",
            )
        )

    roster_image_column = "t1_image_id" if "t1_image_id" in roster.columns else "candidate_t1_image_id"
    roster_required = {"subject_id", "group", "dti_image_id", roster_image_column}
    missing_roster_columns = sorted(roster_required - set(roster.columns))
    if missing_roster_columns:
        errors.append(
            _issue("missing_roster_columns", f"Roster is missing {missing_roster_columns}")
        )
        return _empty_result(errors, warnings, source)

    roster_std = roster[list(roster.columns)].copy()
    roster_std["__subject_id"] = roster_std["subject_id"].map(clean_text)
    roster_std["__image_id"] = roster_std[roster_image_column].map(normalize_positive_integer)
    roster_std["__dti_image_id"] = roster_std["dti_image_id"].map(normalize_positive_integer)
    if len(roster_std) != expected_roster_count:
        errors.append(
            _issue(
                "roster_count_mismatch",
                f"Expected {expected_roster_count} roster rows, found {len(roster_std)}",
            )
        )
    unique_subject_count = roster_std["__subject_id"].nunique()
    if unique_subject_count != expected_subject_count:
        errors.append(
            _issue(
                "roster_subject_count_mismatch",
                f"Expected {expected_subject_count} subjects, found {unique_subject_count}",
            )
        )
    if roster_std["__image_id"].isna().any() or roster_std["__dti_image_id"].isna().any():
        errors.append(_issue("invalid_roster_ids", "Roster contains non-positive/non-integer IDs"))
    bad_subjects = sorted(
        {value for value in roster_std["__subject_id"] if not SUBJECT_RE.fullmatch(value)}
    )
    if bad_subjects:
        errors.append(_issue("invalid_roster_subjects", f"Invalid ADNI subjects: {bad_subjects[:10]}"))
    duplicates = roster_std.loc[
        roster_std["__image_id"].duplicated(keep=False), "__image_id"
    ].dropna().unique()
    if len(duplicates):
        errors.append(_issue("duplicate_roster_image_id", f"Duplicate values include {list(duplicates[:10])}"))
    allowed_groups = {"CN", "MCI", "AD", "SMC"}
    invalid_groups = sorted(set(roster_std["group"].map(clean_text)) - allowed_groups)
    if invalid_groups:
        errors.append(_issue("invalid_roster_groups", f"Unexpected groups: {invalid_groups}"))
    for subject_id, group in roster_std.groupby("__subject_id", sort=False):
        dti_ids = set(group["__dti_image_id"].dropna().astype(int))
        groups = set(group["group"].map(clean_text))
        if len(dti_ids) != 1 or len(groups) != 1:
            errors.append(
                _issue(
                    "roster_subject_mapping_conflict",
                    f"Subject {subject_id} maps to multiple DTI IDs or groups",
                )
            )
    roster_blocking_codes = {
        "invalid_roster_ids",
        "invalid_roster_subjects",
        "invalid_roster_groups",
        "duplicate_roster_image_id",
        "roster_count_mismatch",
        "roster_subject_count_mismatch",
        "roster_subject_mapping_conflict",
        "roster_sha256_mismatch",
        "invalid_expected_roster_sha256",
        "invalid_qc_profile",
    }
    if any(issue["code"] in roster_blocking_codes for issue in errors):
        return _empty_result(errors, warnings, source)

    column_map: dict[str, str | None] = {}
    for field in (*REQUIRED_FIELDS, *OPTIONAL_FIELDS):
        try:
            column_map[field] = resolve_column_strict(
                catalog, field, required=field in REQUIRED_FIELDS
            )
        except ValueError as exc:
            errors.append(_issue(f"column_{field}", str(exc)))
            column_map[field] = None
    if column_map["qc_source"] is None and not clean_text(default_qc_source):
        errors.append(
            _issue(
                "missing_qc_source",
                "Provide one QC-source column or the explicit --default-qc-source value",
            )
        )
    qc_column_normalized = normalized_name(column_map["qc"]) if column_map.get("qc") else ""
    qc_binding = QC_PROFILE_BINDINGS[qc_profile]
    if qc_column_normalized and qc_column_normalized not in qc_binding["columns"]:
        errors.append(
            _issue(
                "qc_profile_column_mismatch",
                f"QC profile {qc_profile} requires source column {qc_binding['column_label']}; "
                f"found {column_map['qc']}",
            )
        )
    if clean_text(default_qc_source) and not qc_source_matches_profile(
        default_qc_source, qc_profile
    ):
        errors.append(
            _issue(
                "qc_profile_source_mismatch",
                f"QC profile {qc_profile} requires provenance {qc_binding['source_label']}; "
                f"found default {clean_text(default_qc_source)!r}",
            )
        )
    if column_map["qc_source"] is not None:
        source_values = _unique_nonblank(catalog[column_map["qc_source"]])
        incompatible_sources = [
            value
            for value in source_values
            if not qc_source_matches_profile(value, qc_profile)
        ]
        if incompatible_sources:
            errors.append(
                _issue(
                    "qc_profile_source_mismatch",
                    f"QC profile {qc_profile} requires provenance {qc_binding['source_label']}; "
                    f"incompatible source values include {incompatible_sources[:10]}",
                )
            )
    for field in OPTIONAL_FIELDS:
        if column_map[field] is None and field != "qc_source":
            warnings.append(_issue(f"optional_{field}_missing", f"Optional {field} column is absent"))
    if any(issue["code"].startswith("column_") for issue in errors):
        result = _empty_result(errors, warnings, source)
        result["validation"]["column_map"] = column_map
        return result

    catalog_work = catalog.copy()
    catalog_work["__image_id"] = catalog_work[column_map["image_id"]].map(normalize_positive_integer)
    catalog_work["__subject_id"] = catalog_work[column_map["subject_id"]].map(clean_text)
    requested_ids = set(roster_std["__image_id"].astype(int))
    roster_subjects = set(roster_std["__subject_id"])
    candidate_rows = (
        catalog_work[column_map["type"]]
        .map(clean_text)
        .str.lower()
        .eq("original")
        & catalog_work[column_map["description"]]
        .map(lambda value: bool(T1_RE.search(clean_text(value))))
    )
    candidate_universe_drift_mask = (
        catalog_work["__subject_id"].isin(roster_subjects)
        & candidate_rows
        & (
            catalog_work["__image_id"].isna()
            | ~catalog_work["__image_id"].isin(requested_ids)
        )
    )
    candidate_universe_drift = catalog_work[candidate_universe_drift_mask]
    if not candidate_universe_drift.empty:
        examples = []
        for row in candidate_universe_drift.head(10).itertuples(index=False, name=None):
            values = dict(zip(candidate_universe_drift.columns, row))
            image_id = values["__image_id"]
            image_label = "unparseable" if pd.isna(image_id) else str(int(image_id))
            examples.append(f"{values['__subject_id']}:I{image_label}")
        errors.append(
            _issue(
                "candidate_universe_drift",
                f"{len(candidate_universe_drift)} Original T1 candidate rows for roster subjects "
                f"are outside the frozen request Image-ID universe; examples={examples}",
            )
        )

    ignored_invalid_id_mask = (
        catalog_work["__image_id"].isna() & ~candidate_universe_drift_mask
    )
    invalid_id_rows = int(ignored_invalid_id_mask.sum())
    if invalid_id_rows:
        warnings.append(
            _issue(
                "unparseable_catalog_ids_outside_request_unknown",
                f"{invalid_id_rows} catalog rows have unparseable Image IDs and were ignored",
            )
        )
    catalog_requested = catalog_work[catalog_work["__image_id"].isin(requested_ids)].copy()
    ignored_out_of_roster_mask = (
        catalog_work["__image_id"].notna()
        & ~catalog_work["__image_id"].isin(requested_ids)
        & ~candidate_universe_drift_mask
    )
    unexpected_rows = int(ignored_out_of_roster_mask.sum())
    if unexpected_rows:
        warnings.append(
            _issue(
                "out_of_roster_rows_ignored",
                f"{unexpected_rows} non-candidate/non-roster catalog rows are outside the request roster and were not validated",
            )
        )

    absences: dict[int, str] = {}
    if absence_ledger is not None:
        try:
            absence_id_col = resolve_column_strict(absence_ledger, "image_id", required=True)
        except ValueError as exc:
            errors.append(_issue("absence_image_id_column", str(exc)))
            absence_id_col = None
        reason_candidates = [
            column
            for column in absence_ledger.columns
            if normalized_name(column) in {"absencereason", "reason", "sourcesidereason"}
        ]
        if len(reason_candidates) != 1:
            errors.append(
                _issue(
                    "absence_reason_column",
                    "Absence ledger must contain exactly one absence_reason/reason column",
                )
            )
        if absence_id_col and len(reason_candidates) == 1:
            reason_col = reason_candidates[0]
            for row in absence_ledger.itertuples(index=False, name=None):
                values = dict(zip(absence_ledger.columns, row))
                image_id = normalize_positive_integer(values[absence_id_col])
                reason = clean_text(values[reason_col])
                if image_id is None or image_id not in requested_ids:
                    errors.append(_issue("invalid_absence_id", "Absence ledger contains an invalid or unrequested ID", image_id))
                    continue
                if not reason:
                    errors.append(_issue("blank_absence_reason", "Absence reason is blank", image_id))
                    continue
                if image_id in absences and absences[image_id] != reason:
                    errors.append(_issue("conflicting_absence_reason", "Absence reasons conflict", image_id))
                absences[image_id] = reason

    canonical_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    mapped_columns = {column for column in column_map.values() if column}
    roster_by_id = roster_std.set_index("__image_id", drop=False)
    try:
        latest_study_date = datetime.fromisoformat(
            str(source["generated_at"]).replace("Z", "+00:00")
        ).date()
    except ValueError:
        latest_study_date = datetime.now(timezone.utc).date()

    for image_id in sorted(requested_ids):
        request = roster_by_id.loc[image_id]
        group = catalog_requested[catalog_requested["__image_id"].eq(image_id)].copy()
        per_id_errors: list[dict[str, Any]] = []
        per_id_warnings: list[dict[str, Any]] = []
        if group.empty:
            status = "absent_explained" if image_id in absences else "missing"
            if status == "missing":
                issue = _issue("requested_id_missing", "Requested Image ID is absent without a source-side reason", image_id)
                errors.append(issue)
                per_id_errors.append(issue)
            coverage_rows.append(
                {
                    "subject_id": request["__subject_id"],
                    "group": request["group"],
                    "dti_image_id": int(request["__dti_image_id"]),
                    "t1_image_id": image_id,
                    "coverage_status": status,
                    "catalog_row_count": 0,
                    "study_date": "",
                    "qc_status": "",
                    "eligible_for_pairing": False,
                    "absence_reason": absences.get(image_id, ""),
                    "validation_issues": ";".join(issue["code"] for issue in per_id_errors),
                    "approved_for_download": False,
                }
            )
            continue

        if image_id in absences:
            issue = _issue("present_and_absent", "Image ID is both present and listed as absent", image_id)
            errors.append(issue)
            per_id_errors.append(issue)

        canonical: dict[str, Any] = {}
        raw_study_date_values: list[str] = []
        raw_qc_values: list[str] = []
        for field in CONFLICT_FIELDS:
            column = column_map.get(field)
            if field == "qc_source":
                values = _unique_nonblank(group[column]) if column is not None else []
                if not values and clean_text(default_qc_source):
                    values = [clean_text(default_qc_source)]
            elif column is None:
                values = []
            elif field == "study_date":
                raw_values = [clean_text(value) for value in group[column]]
                raw_study_date_values = sorted({value for value in raw_values if value})
                invalid_values = [value for value in raw_values if parse_exact_calendar_date(value) is None]
                if invalid_values:
                    issue = _issue("invalid_or_imprecise_study_date", "Study Date must be a valid full calendar day", image_id)
                    errors.append(issue)
                    per_id_errors.append(issue)
                values = sorted({value for value in (parse_exact_calendar_date(raw) for raw in raw_values) if value})
                for parsed_value in values:
                    parsed_date = date.fromisoformat(parsed_value)
                    if parsed_date < ADNI_EARLIEST_PLAUSIBLE_DATE or parsed_date > latest_study_date:
                        issue = _issue(
                            "implausible_study_date",
                            "Study Date falls outside the prespecified ADNI-era/source-date bounds",
                            image_id,
                        )
                        errors.append(issue)
                        per_id_errors.append(issue)
            elif field == "qc":
                raw_values = [clean_text(value) for value in group[column]]
                raw_qc_values = sorted({value for value in raw_values if value})
                parsed_values = [parse_qc_strict(value, qc_profile) for value in raw_values]
                if any(value is None for value in parsed_values):
                    issue = _issue("blank_or_unrecognized_qc", "QC must be pass, fail, or an explicit recognized unknown code", image_id)
                    errors.append(issue)
                    per_id_errors.append(issue)
                values = sorted({value for value in parsed_values if value})
            else:
                values = _unique_nonblank(group[column])

            if (field in REQUIRED_FIELDS or field == "qc_source") and not values:
                issue = _issue(f"missing_{field}", f"Requested rows have no usable {field}", image_id)
                errors.append(issue)
                per_id_errors.append(issue)
            if len(values) > 1:
                issue = _issue(f"conflicting_{field}", f"Duplicate rows conflict on {field}", image_id)
                errors.append(issue)
                per_id_errors.append(issue)
            canonical[field] = values[0] if len(values) == 1 else ""

        if canonical["subject_id"] and canonical["subject_id"] != request["__subject_id"]:
            issue = _issue("subject_mismatch", "Catalog subject does not match the request roster", image_id)
            errors.append(issue)
            per_id_errors.append(issue)
        if canonical["subject_id"] and not SUBJECT_RE.fullmatch(canonical["subject_id"]):
            issue = _issue("invalid_subject_syntax", "Catalog subject does not match ADNI PTID syntax", image_id)
            errors.append(issue)
            per_id_errors.append(issue)
        if canonical["type"].lower() != "original":
            issue = _issue("not_original", "Requested Image ID is not explicitly Type=Original", image_id)
            errors.append(issue)
            per_id_errors.append(issue)
        if not T1_RE.search(canonical["description"]):
            issue = _issue("not_t1_description", "Requested Image ID does not match the frozen T1 description family", image_id)
            errors.append(issue)
            per_id_errors.append(issue)
        if canonical["modality"] and canonical["modality"].strip().lower() not in {"mri", "mr"}:
            issue = _issue("not_mri_modality", "Requested Image ID is not MRI modality", image_id)
            errors.append(issue)
            per_id_errors.append(issue)
        if canonical["qc"] == "unknown":
            warning = _issue("qc_pending", "QC is explicitly not assessed/pending", image_id)
            warnings.append(warning)
            per_id_warnings.append(warning)

        coverage_status = "present_invalid" if per_id_errors else "present_valid"
        eligible = coverage_status == "present_valid" and canonical["qc"] == "pass"
        coverage_rows.append(
            {
                "subject_id": request["__subject_id"],
                "group": request["group"],
                "dti_image_id": int(request["__dti_image_id"]),
                "t1_image_id": image_id,
                "coverage_status": coverage_status,
                "catalog_row_count": len(group),
                "study_date": canonical["study_date"],
                "qc_status": canonical["qc"],
                "eligible_for_pairing": eligible,
                "absence_reason": "",
                "validation_issues": ";".join(
                    issue["code"] for issue in [*per_id_errors, *per_id_warnings]
                ),
                "approved_for_download": False,
            }
        )
        if not per_id_errors:
            canonical_rows.append(
                {
                    "subject_id": canonical["subject_id"],
                    "image_id": image_id,
                    "study_date": canonical["study_date"],
                    "visit": canonical["visit"],
                    "phase": canonical["phase"],
                    "description": canonical["description"],
                    "type": canonical["type"],
                    "qc_status": canonical["qc"],
                    "qc_raw_values_json": json.dumps(raw_qc_values, separators=(",", ":")),
                    "qc_source": canonical["qc_source"],
                    "study_date_raw_values_json": json.dumps(raw_study_date_values, separators=(",", ":")),
                    "series_uid": canonical["series_uid"],
                    "protocol": canonical["protocol"],
                    "mri_protocol_phase": canonical["mri_protocol_phase"],
                    "manufacturer": canonical["manufacturer"],
                    "model": canonical["model"],
                    "field_strength": canonical["field_strength"],
                    "site": canonical["site"],
                    "modality": canonical["modality"],
                    "catalog_source_row_count": len(group),
                    "source_extra_json": _extra_source_json(group, mapped_columns),
                    "source_name": source["name"],
                    "source_generated_at": source["generated_at"],
                    "source_catalog_sha256": catalog_sha256,
                    "eligible_for_pairing": eligible,
                    "approved_for_download": False,
                }
            )

    canonical_frame = pd.DataFrame(canonical_rows).sort_values("image_id", ignore_index=True) if canonical_rows else pd.DataFrame()
    coverage_frame = pd.DataFrame(coverage_rows).sort_values("t1_image_id", ignore_index=True)
    error_frame = pd.DataFrame(errors)
    all_accounted = bool(
        coverage_frame["coverage_status"].isin(
            ["present_valid", "absent_explained"]
        ).all()
    )
    all_present_valid = bool(coverage_frame["coverage_status"].eq("present_valid").all())
    eligible_subjects = set(
        coverage_frame.loc[coverage_frame["eligible_for_pairing"], "subject_id"]
    )
    all_request_ids_qc_resolved = bool(
        (
            coverage_frame["coverage_status"].eq("present_valid")
            & coverage_frame["qc_status"].isin(["pass", "fail"])
        ).all()
    )
    pairing_ready = bool(
        not errors
        and all_present_valid
        and all_request_ids_qc_resolved
        and eligible_subjects == roster_subjects
    )
    status = "PASS" if not errors and all_accounted else "FAIL"
    validation = {
        "schema_version": "1.0",
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "column_map": column_map,
        "counts": {
            "roster_rows": len(roster_std),
            "roster_subjects": len(roster_subjects),
            "catalog_rows": len(catalog),
            "requested_catalog_rows": len(catalog_requested),
            "candidate_universe_drift_rows": len(candidate_universe_drift),
            "out_of_roster_rows_ignored": unexpected_rows,
            "present_valid": int(coverage_frame["coverage_status"].eq("present_valid").sum()),
            "absent_explained": int(coverage_frame["coverage_status"].eq("absent_explained").sum()),
            "missing": int(coverage_frame["coverage_status"].eq("missing").sum()),
            "present_invalid": int(coverage_frame["coverage_status"].eq("present_invalid").sum()),
            "eligible_for_pairing": int(coverage_frame["eligible_for_pairing"].sum()),
            "subjects_with_eligible_t1": len(eligible_subjects),
            "explicit_qc_fail": int(coverage_frame["qc_status"].eq("fail").sum()),
            "qc_pending": int(coverage_frame["qc_status"].eq("unknown").sum()),
            "errors": len(errors),
            "warnings": len(warnings),
        },
        "checks": {
            "expected_roster_count": expected_roster_count,
            "expected_subject_count": expected_subject_count,
            "all_request_ids_accounted": all_accounted,
            "all_request_ids_present_valid": all_present_valid,
            "all_request_ids_qc_resolved": all_request_ids_qc_resolved,
            "candidate_universe_closed": candidate_universe_drift.empty,
            "qc_profile_column_bound": not any(
                issue["code"] == "qc_profile_column_mismatch" for issue in errors
            ),
            "qc_profile_source_bound": not any(
                issue["code"] == "qc_profile_source_mismatch" for issue in errors
            ),
            "pairing_ready": pairing_ready,
            "exact_calendar_dates_only": not any(issue["code"] == "invalid_or_imprecise_study_date" for issue in errors),
            "duplicate_identity_date_uid_qc_conflicts_absent": not any(issue["code"].startswith("conflicting_") for issue in errors),
            "download_approval_forced_false": not coverage_frame["approved_for_download"].any(),
        },
        "errors": errors,
        "warnings": warnings,
        "safety": {
            "metadata_only": True,
            "approved_for_download_count": int(coverage_frame["approved_for_download"].sum()),
            "images_downloaded": 0,
        },
        "release_status": (
            "PAIRING_READY"
            if status == "PASS" and pairing_ready
            else "INTAKE_PASS_PAIRING_BLOCKED"
            if status == "PASS"
            else "INTAKE_FAILED"
        ),
    }
    return {
        "canonical": canonical_frame,
        "coverage": coverage_frame,
        "rejects": error_frame,
        "validation": validation,
    }


def data_dictionary() -> pd.DataFrame:
    rows = [
        ("subject_id", "string", "ADNI participant ID; must match the request roster"),
        ("image_id", "positive integer", "Exact LONI/IDA Image ID"),
        ("study_date", "YYYY-MM-DD", "Exact DICOM study date; never rounded age or visit date"),
        ("visit", "string", "Source visit label, preferably VISCODE2 plus human-readable visit if present"),
        ("phase", "string", "ADNI phase/project"),
        ("description", "string", "Source series description; must match the frozen T1 family"),
        ("type", "string", "Must be Original for requested candidates"),
        ("qc_status", "pass|fail|unknown", "Normalized source QC status"),
        ("qc_raw_values_json", "JSON array", "All distinct raw QC values for the source Image ID"),
        ("qc_source", "string", "Named source field/table for QC"),
        ("study_date_raw_values_json", "JSON array", "All distinct raw date strings before ISO normalization"),
        ("series_uid", "string", "Series Instance UID or source series identifier where available"),
        ("mri_protocol_phase", "string", "MRI acquisition protocol family; not enrollment phase"),
        ("eligible_for_pairing", "boolean", "True only for structurally valid rows with QC pass"),
        ("approved_for_download", "boolean", "Always false; only a later human gate can approve transfer"),
        ("source_extra_json", "JSON", "Deterministically retained non-canonical source columns"),
        ("source_catalog_sha256", "SHA-256", "Immutable hash of the exact source CSV bytes"),
    ]
    return pd.DataFrame(rows, columns=["field", "type", "definition"])


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output path: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=str(output_dir.parent)
        )
    )
    try:
        filenames = {
            "coverage": "exact_date_catalog_coverage_v2.csv",
            "rejects": "exact_date_catalog_rejects_v2.csv",
            "data_dictionary": "exact_date_catalog_data_dictionary_v2.csv",
        }
        result["coverage"].to_csv(staging / filenames["coverage"], index=False)
        result["rejects"].to_csv(staging / filenames["rejects"], index=False)
        data_dictionary().to_csv(staging / filenames["data_dictionary"], index=False)

        validation = dict(result["validation"])
        outputs: dict[str, Any] = {}
        for name, filename in filenames.items():
            staged_path = staging / filename
            outputs[name] = {
                "path": str((output_dir / filename).resolve()),
                "size_bytes": staged_path.stat().st_size,
                "sha256": file_sha256(staged_path),
            }
        if validation["status"] == "PASS":
            filename = "exact_date_catalog_v2.csv"
            staged_path = staging / filename
            result["canonical"].to_csv(staged_path, index=False)
            outputs["canonical_catalog"] = {
                "path": str((output_dir / filename).resolve()),
                "size_bytes": staged_path.stat().st_size,
                "sha256": file_sha256(staged_path),
            }
        validation["outputs"] = outputs
        _write_json(staging / "exact_date_catalog_validation_v2.json", validation)
        os.rename(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_csv_snapshot(path: Path) -> tuple[pd.DataFrame, str]:
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked metadata input: {path}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"CSV must be UTF-8/UTF-8-BOM: {path}") from exc
    header = next(csv.reader(io.StringIO(decoded)), None)
    if not header:
        raise ValueError(f"CSV has no header: {path}")
    normalized = [normalized_name(column) for column in header]
    duplicate_headers = sorted(
        {header[index] for index, value in enumerate(normalized) if normalized.count(value) > 1}
    )
    if duplicate_headers:
        raise ValueError(f"Duplicate normalized CSV headers: {duplicate_headers}")
    frame = pd.read_csv(io.BytesIO(payload), low_memory=False, encoding="utf-8-sig")
    return frame, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-generated-at", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--absence-ledger", type=Path)
    parser.add_argument("--default-qc-source")
    parser.add_argument("--qc-profile", choices=sorted(QC_PROFILES), required=True)
    parser.add_argument("--expected-roster-count", type=int, default=4015)
    parser.add_argument("--expected-subject-count", type=int, default=530)
    parser.add_argument(
        "--expected-roster-sha256", default=CANONICAL_REQUEST_ROSTER_SHA256
    )
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()

    catalog, catalog_hash = read_csv_snapshot(args.catalog)
    roster, roster_hash = read_csv_snapshot(args.roster)
    if args.absence_ledger:
        absence_ledger, absence_hash = read_csv_snapshot(args.absence_ledger)
    else:
        absence_ledger, absence_hash = None, None
    result = validate_exact_date_catalog(
        catalog,
        roster,
        source_name=args.source_name,
        source_generated_at=args.source_generated_at,
        catalog_sha256=catalog_hash,
        roster_sha256=roster_hash,
        absence_ledger=absence_ledger,
        absence_sha256=absence_hash,
        default_qc_source=args.default_qc_source,
        qc_profile=args.qc_profile,
        expected_roster_count=args.expected_roster_count,
        expected_subject_count=args.expected_subject_count,
        expected_roster_sha256=args.expected_roster_sha256,
        expected_catalog_sha256=args.expected_sha256,
    )
    result["validation"]["source"]["catalog_path"] = str(args.catalog.resolve())
    result["validation"]["source"]["roster_path"] = str(args.roster.resolve())
    result["validation"]["source"]["absence_ledger_path"] = (
        str(args.absence_ledger.resolve()) if args.absence_ledger else None
    )
    write_outputs(result, args.out)
    print(json.dumps(result["validation"]["counts"], indent=2, sort_keys=True))
    print(args.out.resolve())
    if result["validation"]["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
