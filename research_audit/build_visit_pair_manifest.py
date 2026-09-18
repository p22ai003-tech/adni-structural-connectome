#!/usr/bin/env python3
"""Build deterministic exact-date DTI-T1 pairing manifests.

The command is non-destructive. It reads metadata only, never downloads an
image, and marks every selected candidate as awaiting the human pairing and
download gates.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import numbers
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SID_RE = re.compile(r"^(?P<subject>\d{3}_S_\d{4})_I(?P<image_id>\d+)$")
IMAGE_ID_RE = re.compile(r"(?:^|[_/])I(?P<image_id>\d+)(?:[_/.]|$)")
SUBJECT_RE = re.compile(r"^\d{3}_S_\d{4}$")
ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
US_DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
DICOM_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
T1_RE = re.compile(r"MPRAGE|MP[- ]?RAGE|IR[- ]?SPGR|IR[- ]?FSPGR|BRAVO", re.I)
REPEAT_RE = re.compile(r"REPEAT|RPT", re.I)

VALIDATION_SCHEMA_VERSION = "1.0"
FROZEN_REQUEST_ROSTER_SHA256 = (
    "4a935b7bbd7d6d5ea5c1c8d80f4cc8fde840abaa88c0d4d2419878591bcc2406"
)
FROZEN_REQUEST_ROSTER_ROWS = 4015
FROZEN_REQUEST_SUBJECTS = 530

CANONICAL_CATALOG_REQUIRED_COLUMNS = frozenset(
    {
        "subject_id",
        "image_id",
        "study_date",
        "visit",
        "phase",
        "description",
        "type",
        "qc_status",
        "qc_raw_values_json",
        "qc_source",
        "study_date_raw_values_json",
        "series_uid",
        "protocol",
        "mri_protocol_phase",
        "manufacturer",
        "model",
        "field_strength",
        "site",
        "modality",
        "catalog_source_row_count",
        "source_extra_json",
        "source_name",
        "source_generated_at",
        "source_catalog_sha256",
        "eligible_for_pairing",
        "approved_for_download",
    }
)

ALIASES = {
    "subject_id": ["Subject ID", "Subject", "PTID"],
    "image_id": ["Image ID", "ImageID", "Image UID", "IMAGEUID"],
    "study_date": ["Study Date", "StudyDate", "EXAMDATE", "Scan Date", "Acquisition Date"],
    "visit": ["Visit", "Visit Code", "VISCODE", "VISCODE2"],
    "phase": ["Phase", "Project", "Study"],
    "description": ["Description", "Series Description", "Sequence Description"],
    "type": ["Type", "Image Type", "ImageType"],
    "qc": ["QC", "QC Status", "Image QC", "Image QC Status", "Quality"],
    "series_uid": ["Series UID", "SeriesUID", "Series Instance UID"],
    "protocol": ["Imaging Protocol", "Protocol", "Protocol Name"],
    "manufacturer": ["Manufacturer", "Vendor"],
    "model": ["Model", "Scanner Model", "Manufacturer Model Name"],
    "field_strength": ["Field Strength", "Magnetic Field Strength"],
}

PASS_QC = {"pass", "passed", "acceptable", "accepted", "yes", "y", "1", "true", "selected"}
FAIL_QC = {"fail", "failed", "unacceptable", "rejected", "no", "n", "0", "false"}


@dataclass(frozen=True)
class CatalogAttestation:
    """Evidence extracted from a fully verified validator release."""

    validation_path: str
    validation_sha256: str
    catalog_path: str
    catalog_sha256: str
    catalog_size_bytes: int
    canonical_rows: int
    eligible_rows: int
    eligible_subject_ids: frozenset[str]
    roster_sha256: str
    roster_rows: int
    roster_subjects: int
    validation_schema_version: str = VALIDATION_SCHEMA_VERSION
    status: str = "PASS_PAIRING_READY"

    def as_dict(self) -> dict[str, Any]:
        subject_digest = hashlib.sha256(
            "\n".join(sorted(self.eligible_subject_ids)).encode("utf-8")
        ).hexdigest()
        return {
            "validation_path": self.validation_path,
            "validation_sha256": self.validation_sha256,
            "catalog_path": self.catalog_path,
            "catalog_sha256": self.catalog_sha256,
            "catalog_size_bytes": self.catalog_size_bytes,
            "canonical_rows": self.canonical_rows,
            "eligible_rows": self.eligible_rows,
            "eligible_subject_count": len(self.eligible_subject_ids),
            "eligible_subject_ids_sha256": subject_digest,
            "roster_sha256": self.roster_sha256,
            "roster_rows": self.roster_rows,
            "roster_subjects": self.roster_subjects,
            "validation_schema_version": self.validation_schema_version,
            "status": self.status,
        }


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def resolve_column(
    frame: pd.DataFrame, field: str, required: bool = False
) -> str | None:
    lookup = {normalized_name(column): column for column in frame.columns}
    for alias in ALIASES[field]:
        match = lookup.get(normalized_name(alias))
        if match is not None:
            return match
    if required:
        raise ValueError(
            f"Missing required {field!r} column. Accepted aliases: {ALIASES[field]}"
        )
    return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_exact_calendar_date(value: Any) -> pd.Timestamp | None:
    """Parse only explicit calendar dates, never partial dates or timestamps."""

    if pd.isna(value) or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        if value.time() != datetime.min.time():
            return None
        return pd.Timestamp(value.date())
    if isinstance(value, date):
        return pd.Timestamp(value)
    if isinstance(value, numbers.Integral):
        token = str(int(value))
    elif isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            return None
        token = str(int(number))
    else:
        token = str(value).strip()
    match = ISO_DATE_RE.fullmatch(token)
    if match:
        year, month, day = map(int, match.groups())
    else:
        match = US_DATE_RE.fullmatch(token)
        if match:
            month, day, year = map(int, match.groups())
        else:
            match = DICOM_DATE_RE.fullmatch(token)
            if not match:
                return None
            year, month, day = map(int, match.groups())
    try:
        return pd.Timestamp(date(year, month, day))
    except ValueError:
        return None


def _strict_positive_integer(value: Any) -> int | None:
    if pd.isna(value) or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        number = int(value)
        return number if number > 0 else None
    if isinstance(value, numbers.Real):
        number = float(value)
        if (
            not math.isfinite(number)
            or not number.is_integer()
            or number <= 0
            or abs(number) > 2**53
        ):
            return None
        return int(number)
    token = str(value).strip()
    if not re.fullmatch(r"[1-9]\d*", token):
        return None
    return int(token)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"MRI validation {label} must be an object")
    return value


def _require_json_integer(mapping: dict[str, Any], key: str, label: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"MRI validation {label}.{key} must be an integer")
    return value


def _strict_boolean(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    )
    if mapped.isna().any():
        raise ValueError(f"Canonical MRI catalog has invalid {label} values")
    return mapped.astype(bool)


def load_attested_pairing_catalog(
    catalog_path: Path, validation_path: Path
) -> tuple[pd.DataFrame, CatalogAttestation]:
    """Load only a checksum-bound, pairing-ready canonical catalog."""

    if catalog_path.is_symlink() or validation_path.is_symlink():
        raise ValueError("Refusing symlinked MRI catalog or validation input")
    validation_bytes = validation_path.read_bytes()
    try:
        payload = json.loads(validation_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("MRI validation is not valid UTF-8 JSON") from exc
    payload = _require_mapping(payload, "root")
    if payload.get("schema_version") != VALIDATION_SCHEMA_VERSION:
        raise ValueError(
            f"MRI validation schema_version must be {VALIDATION_SCHEMA_VERSION}"
        )
    if payload.get("status") != "PASS":
        raise ValueError("MRI catalog validation status is not PASS")
    if payload.get("release_status") != "PAIRING_READY":
        raise ValueError("MRI catalog validation release_status is not PAIRING_READY")

    errors = payload.get("errors")
    if not isinstance(errors, list) or errors:
        raise ValueError("MRI validation must contain an empty errors array")

    source = _require_mapping(payload.get("source"), "source")
    source_name = source.get("name")
    if not isinstance(source_name, str) or not source_name.strip():
        raise ValueError("MRI validation source.name is missing")
    if (
        not isinstance(source.get("generated_at"), str)
        or not source["generated_at"].strip()
    ):
        raise ValueError("MRI validation source.generated_at is missing")
    if (
        not isinstance(source.get("qc_profile"), str)
        or not source["qc_profile"].strip()
    ):
        raise ValueError("MRI validation source.qc_profile is missing")
    roster_hash = str(source.get("roster_sha256", "")).lower()
    expected_roster_hash = str(source.get("expected_roster_sha256", "")).lower()
    if (
        roster_hash != FROZEN_REQUEST_ROSTER_SHA256
        or expected_roster_hash != FROZEN_REQUEST_ROSTER_SHA256
    ):
        raise ValueError("MRI validation is not bound to the frozen request roster SHA-256")
    source_catalog_hash = str(source.get("catalog_sha256", "")).lower()
    if not SHA256_RE.fullmatch(source_catalog_hash):
        raise ValueError("MRI validation source.catalog_sha256 is malformed")

    column_map = _require_mapping(payload.get("column_map"), "column_map")
    for field in (
        "subject_id",
        "image_id",
        "study_date",
        "visit",
        "phase",
        "description",
        "type",
        "qc",
    ):
        if (
            not isinstance(column_map.get(field), str)
            or not column_map[field].strip()
        ):
            raise ValueError(f"MRI validation column_map.{field} is missing")

    counts = _require_mapping(payload.get("counts"), "counts")
    required_counts = {
        "roster_rows": FROZEN_REQUEST_ROSTER_ROWS,
        "roster_subjects": FROZEN_REQUEST_SUBJECTS,
        "subjects_with_eligible_t1": FROZEN_REQUEST_SUBJECTS,
        "candidate_universe_drift_rows": 0,
        "present_valid": FROZEN_REQUEST_ROSTER_ROWS,
        "absent_explained": 0,
        "missing": 0,
        "present_invalid": 0,
        "qc_pending": 0,
        "errors": 0,
    }
    for key, expected in required_counts.items():
        actual = _require_json_integer(counts, key, "counts")
        if actual != expected:
            raise ValueError(
                f"MRI validation counts.{key} must be {expected}, found {actual}"
            )
    present_valid = counts["present_valid"]
    absent_explained = counts["absent_explained"]
    eligible_count = _require_json_integer(counts, "eligible_for_pairing", "counts")
    explicit_qc_fail = _require_json_integer(counts, "explicit_qc_fail", "counts")
    if present_valid < 0 or absent_explained < 0 or eligible_count <= 0:
        raise ValueError("MRI validation coverage counts are invalid")
    if present_valid + absent_explained != FROZEN_REQUEST_ROSTER_ROWS:
        raise ValueError("MRI validation request-coverage counts do not total 4015")
    if eligible_count + explicit_qc_fail != present_valid:
        raise ValueError("MRI validation pass/fail QC counts do not cover the roster")

    checks = _require_mapping(payload.get("checks"), "checks")
    required_checks: dict[str, Any] = {
        "expected_roster_count": FROZEN_REQUEST_ROSTER_ROWS,
        "expected_subject_count": FROZEN_REQUEST_SUBJECTS,
        "all_request_ids_accounted": True,
        "all_request_ids_present_valid": True,
        "all_request_ids_qc_resolved": True,
        "candidate_universe_closed": True,
        "pairing_ready": True,
        "exact_calendar_dates_only": True,
        "duplicate_identity_date_uid_qc_conflicts_absent": True,
        "download_approval_forced_false": True,
    }
    for key, expected in required_checks.items():
        if checks.get(key) != expected or type(checks.get(key)) is not type(expected):
            raise ValueError(
                f"MRI validation checks.{key} must be {expected!r}"
            )

    safety = _require_mapping(payload.get("safety"), "safety")
    if safety.get("metadata_only") is not True:
        raise ValueError("MRI validation safety.metadata_only must be true")
    if _require_json_integer(safety, "approved_for_download_count", "safety") != 0:
        raise ValueError("MRI validation reports pre-approved downloads")
    if _require_json_integer(safety, "images_downloaded", "safety") != 0:
        raise ValueError("MRI validation is not a metadata-only release")

    outputs = _require_mapping(payload.get("outputs"), "outputs")
    canonical = _require_mapping(
        outputs.get("canonical_catalog"), "outputs.canonical_catalog"
    )
    expected_hash = str(canonical.get("sha256", "")).lower()
    if not SHA256_RE.fullmatch(expected_hash):
        raise ValueError("MRI canonical catalog SHA-256 in validation is malformed")
    expected_size = _require_json_integer(
        canonical, "size_bytes", "outputs.canonical_catalog"
    )
    if not isinstance(canonical.get("path"), str) or not canonical["path"].strip():
        raise ValueError("MRI validation canonical catalog path is missing")

    catalog_bytes = catalog_path.read_bytes()
    actual_hash = hashlib.sha256(catalog_bytes).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError("MRI canonical catalog SHA-256 does not match its validation")
    if expected_size != len(catalog_bytes):
        raise ValueError("MRI canonical catalog size does not match its validation")

    frame = pd.read_csv(io.BytesIO(catalog_bytes), low_memory=False)
    if frame.columns.duplicated().any():
        raise ValueError("Canonical MRI catalog has duplicate column names")
    missing = sorted(CANONICAL_CATALOG_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Canonical MRI catalog schema is incomplete: {missing}")
    if len(frame) != present_valid:
        raise ValueError(
            "Canonical MRI catalog row count does not match validation counts.present_valid"
        )

    image_ids = frame["image_id"].map(_strict_positive_integer)
    if image_ids.isna().any() or image_ids.duplicated().any():
        raise ValueError("Canonical MRI catalog has invalid or duplicate Image IDs")
    subjects = frame["subject_id"].astype(str).str.strip()
    if not subjects.map(lambda value: bool(SUBJECT_RE.fullmatch(value))).all():
        raise ValueError("Canonical MRI catalog has invalid ADNI subject IDs")
    exact_dates = frame["study_date"].map(parse_exact_calendar_date)
    if exact_dates.isna().any():
        raise ValueError("Canonical MRI catalog has non-exact Study Dates")

    eligible = _strict_boolean(frame["eligible_for_pairing"], "eligible_for_pairing")
    approved = _strict_boolean(frame["approved_for_download"], "approved_for_download")
    if approved.any():
        raise ValueError("Canonical MRI catalog must never pre-approve a download")
    if int(eligible.sum()) != eligible_count:
        raise ValueError(
            "Canonical MRI eligible row count does not match validation counts"
        )
    filtered = frame[eligible].copy()
    if filtered.empty:
        raise ValueError("Canonical MRI catalog has no pairing-eligible rows")
    eligible_subjects = frozenset(subjects[eligible])
    if len(eligible_subjects) != FROZEN_REQUEST_SUBJECTS:
        raise ValueError(
            "Canonical MRI catalog does not independently cover all 530 roster subjects"
        )
    if not filtered["qc_status"].astype(str).str.strip().str.lower().eq("pass").all():
        raise ValueError("Pairing-eligible canonical rows must all have QC pass")
    if not filtered["type"].astype(str).str.strip().str.lower().eq("original").all():
        raise ValueError("Pairing-eligible canonical rows must all be Original")
    if not filtered["description"].astype(str).str.contains(T1_RE, na=False).all():
        raise ValueError("Pairing-eligible canonical rows must all be T1-like")
    canonical_source_hashes = set(
        frame["source_catalog_sha256"].astype(str).str.strip().str.lower()
    )
    if canonical_source_hashes != {source_catalog_hash}:
        raise ValueError(
            "Canonical MRI rows are not bound to validation source.catalog_sha256"
        )
    if set(frame["source_name"].astype(str).str.strip()) != {source_name.strip()}:
        raise ValueError("Canonical MRI source_name does not match validation source")

    attestation = CatalogAttestation(
        validation_path=str(validation_path.resolve()),
        validation_sha256=hashlib.sha256(validation_bytes).hexdigest(),
        catalog_path=str(catalog_path.resolve()),
        catalog_sha256=actual_hash,
        catalog_size_bytes=len(catalog_bytes),
        canonical_rows=len(frame),
        eligible_rows=len(filtered),
        eligible_subject_ids=eligible_subjects,
        roster_sha256=roster_hash,
        roster_rows=FROZEN_REQUEST_ROSTER_ROWS,
        roster_subjects=FROZEN_REQUEST_SUBJECTS,
    )
    return filtered, attestation


def parse_qc(value: Any) -> str:
    if pd.isna(value) or not str(value).strip():
        return "unknown"
    token = str(value).strip().lower()
    if token in PASS_QC or token.startswith("pass"):
        return "pass"
    if token in FAIL_QC or token.startswith("fail"):
        return "fail"
    return "unknown"


def parse_current_t1_id(value: Any) -> int | None:
    if pd.isna(value):
        return None
    matches = list(IMAGE_ID_RE.finditer(str(value)))
    return int(matches[-1].group("image_id")) if matches else None


def standardize_series_manifest(
    manifest: pd.DataFrame, band: str | None
) -> pd.DataFrame:
    if "sid" not in manifest.columns:
        raise ValueError("Series manifest must contain a sid column")
    selected = manifest.copy()
    if band is not None:
        if "band" not in selected.columns:
            raise ValueError("--band was provided but the series manifest has no band column")
        selected = selected[selected["band"].astype(str).eq(band)].copy()
    parsed = selected["sid"].astype(str).str.extract(SID_RE)
    if parsed.isna().any(axis=None):
        bad = selected.loc[parsed.isna().any(axis=1), "sid"].head(10).tolist()
        raise ValueError(f"Cannot parse selected DTI Image ID from sid values: {bad}")
    result = pd.DataFrame(
        {
            "series_sid": selected["sid"].astype(str),
            "subject_id": parsed["subject"].astype(str),
            "dti_image_id": pd.to_numeric(parsed["image_id"], errors="raise").astype(int),
            "group_provisional": (
                selected["group"].astype(str) if "group" in selected.columns else ""
            ),
            "band": selected["band"].astype(str) if "band" in selected.columns else "",
            "current_t1_image_id": (
                selected["t1_path"].map(parse_current_t1_id)
                if "t1_path" in selected.columns
                else None
            ),
        }
    )
    if result["series_sid"].duplicated().any():
        raise ValueError("Series manifest contains duplicate sid rows")
    if result["subject_id"].duplicated().any():
        duplicates = result.loc[
            result["subject_id"].duplicated(keep=False), "subject_id"
        ].unique()
        raise ValueError(
            "One DTI per subject is required; duplicates include "
            + ", ".join(map(str, duplicates[:10]))
        )
    return result.reset_index(drop=True)


def standardize_dti_master(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {
        field: resolve_column(frame, field, required=field in {"subject_id", "image_id", "study_date"})
        for field in [
            "subject_id",
            "image_id",
            "study_date",
            "visit",
            "phase",
            "description",
            "protocol",
            "manufacturer",
            "model",
            "field_strength",
        ]
    }
    result = pd.DataFrame(
        {
            "dti_subject_id_source": frame[columns["subject_id"]].astype(str),
            "dti_image_id": pd.to_numeric(frame[columns["image_id"]], errors="coerce").astype("Int64"),
            "dti_study_date": pd.to_datetime(
                frame[columns["study_date"]].map(parse_exact_calendar_date),
                errors="coerce",
            ).dt.normalize(),
        }
    )
    for field in [
        "visit",
        "phase",
        "description",
        "protocol",
        "manufacturer",
        "model",
        "field_strength",
    ]:
        result[f"dti_{field}"] = (
            frame[columns[field]].astype("string") if columns[field] else pd.NA
        )
    result = result[result["dti_image_id"].notna()].copy()
    result["dti_image_id"] = result["dti_image_id"].astype(int)
    return result


def collapse_dti_rows(
    dti: pd.DataFrame, selected_ids: set[int]
) -> pd.DataFrame:
    relevant = dti[dti["dti_image_id"].isin(selected_ids)].copy()
    rows = []
    for image_id, group in relevant.groupby("dti_image_id", sort=True):
        dates = group["dti_study_date"].dropna().unique()
        subjects = group["dti_subject_id_source"].dropna().unique()
        if (
            group["dti_study_date"].isna().any()
            or len(dates) != 1
            or len(subjects) != 1
        ):
            raise ValueError(
                f"Selected DTI I{image_id} has missing, imprecise, or conflicting "
                "exact dates or subjects"
            )
        row = group.sort_index().iloc[0].copy()
        rows.append(row)
    collapsed = pd.DataFrame(rows)
    missing = sorted(selected_ids - set(collapsed["dti_image_id"].astype(int)))
    if missing:
        raise ValueError(f"Selected DTI IDs absent from dti_master: {missing[:10]}")
    return collapsed


def standardize_mri_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {
        field: resolve_column(
            frame,
            field,
            required=field in {"subject_id", "image_id", "study_date", "description"},
        )
        for field in [
            "subject_id",
            "image_id",
            "study_date",
            "visit",
            "phase",
            "description",
            "type",
            "qc",
            "series_uid",
            "protocol",
            "manufacturer",
            "model",
            "field_strength",
        ]
    }
    result = pd.DataFrame(
        {
            "t1_subject_id": frame[columns["subject_id"]].astype(str),
            "t1_image_id": pd.to_numeric(frame[columns["image_id"]], errors="coerce").astype("Int64"),
            "t1_study_date": pd.to_datetime(
                frame[columns["study_date"]].map(parse_exact_calendar_date),
                errors="coerce",
            ).dt.normalize(),
            "t1_description": frame[columns["description"]].astype("string"),
            "source_row": frame.index.astype(int),
        }
    )
    for field in [
        "visit",
        "phase",
        "type",
        "series_uid",
        "protocol",
        "manufacturer",
        "model",
        "field_strength",
    ]:
        result[f"t1_{field}"] = (
            frame[columns[field]].astype("string") if columns[field] else pd.NA
        )
    result["t1_qc_raw"] = (
        frame[columns["qc"]].astype("string") if columns["qc"] else pd.NA
    )
    result["t1_qc_status"] = result["t1_qc_raw"].map(parse_qc)
    result["is_t1_description"] = result["t1_description"].str.contains(
        T1_RE, na=False
    )
    result["is_repeat"] = result["t1_description"].str.contains(REPEAT_RE, na=False)
    result["is_original"] = (
        result["t1_type"].str.strip().str.lower().eq("original")
        if columns["type"]
        else True
    )
    return result


def assert_no_mri_conflicts(mri: pd.DataFrame) -> None:
    relevant = mri[mri["t1_image_id"].notna()].copy()
    for image_id, group in relevant.groupby("t1_image_id", sort=False):
        subjects = group["t1_subject_id"].dropna().unique()
        dates = group["t1_study_date"].dropna().unique()
        if len(subjects) > 1 or len(dates) > 1:
            raise ValueError(
                f"MRI I{int(image_id)} has conflicting subject or Study Date values"
            )


def classify_window(gap_days: int, primary_days: int, sensitivity_days: int) -> str:
    if gap_days <= primary_days:
        return "primary"
    if gap_days <= sensitivity_days:
        return "sensitivity_only"
    return "outside_sensitivity"


def build_manifests(
    dti_master: pd.DataFrame,
    mri_catalog: pd.DataFrame,
    series_manifest: pd.DataFrame,
    primary_days: int = 90,
    sensitivity_days: int = 180,
    band: str | None = None,
    catalog_attestation: CatalogAttestation | None = None,
) -> dict[str, Any]:
    if not isinstance(catalog_attestation, CatalogAttestation):
        raise ValueError(
            "Pair construction requires a CatalogAttestation from the checksum-bound "
            "pairing-ready catalog loader"
        )
    if (
        catalog_attestation.status != "PASS_PAIRING_READY"
        or catalog_attestation.validation_schema_version != VALIDATION_SCHEMA_VERSION
        or catalog_attestation.roster_sha256 != FROZEN_REQUEST_ROSTER_SHA256
        or catalog_attestation.roster_rows != FROZEN_REQUEST_ROSTER_ROWS
        or catalog_attestation.roster_subjects != FROZEN_REQUEST_SUBJECTS
        or catalog_attestation.canonical_rows != FROZEN_REQUEST_ROSTER_ROWS
        or catalog_attestation.eligible_rows < FROZEN_REQUEST_SUBJECTS
        or catalog_attestation.eligible_rows > catalog_attestation.canonical_rows
        or catalog_attestation.catalog_size_bytes <= 0
        or not SHA256_RE.fullmatch(catalog_attestation.catalog_sha256)
        or not SHA256_RE.fullmatch(catalog_attestation.validation_sha256)
        or not isinstance(catalog_attestation.eligible_subject_ids, frozenset)
        or len(catalog_attestation.eligible_subject_ids)
        != FROZEN_REQUEST_SUBJECTS
    ):
        raise ValueError("CatalogAttestation does not satisfy the frozen pairing contract")
    if primary_days < 0 or sensitivity_days < primary_days:
        raise ValueError("Require 0 <= primary_days <= sensitivity_days")

    selected = standardize_series_manifest(series_manifest, band)
    if selected.empty:
        raise ValueError("Series-manifest selection is empty")
    selected_subjects = set(selected["subject_id"])
    missing_attested_subjects = sorted(
        selected_subjects - set(catalog_attestation.eligible_subject_ids)
    )
    if missing_attested_subjects:
        raise ValueError(
            "Selected DTI subjects are absent from attested eligible MRI coverage: "
            + ", ".join(missing_attested_subjects[:10])
        )
    dti = collapse_dti_rows(
        standardize_dti_master(dti_master),
        set(selected["dti_image_id"].astype(int)),
    )
    selected = selected.merge(
        dti, on="dti_image_id", how="left", validate="1:1"
    )
    mismatch = selected[
        selected["subject_id"].ne(selected["dti_subject_id_source"])
    ]
    if not mismatch.empty:
        raise ValueError(
            "Series-manifest subject does not match dti_master for: "
            + ", ".join(mismatch["series_sid"].head(10))
        )

    mri = standardize_mri_catalog(mri_catalog)
    assert_no_mri_conflicts(mri)
    mri = mri[mri["t1_subject_id"].isin(selected_subjects)].copy()

    flow_counts = {
        "input_mri_rows_for_selected_subjects": int(len(mri)),
        "ignored_non_t1_description": int((~mri["is_t1_description"]).sum()),
        "ignored_non_original": int(
            (mri["is_t1_description"] & ~mri["is_original"]).sum()
        ),
    }
    t1_like = mri[mri["is_t1_description"] & mri["is_original"]].copy()
    pairing_eligible = t1_like[
        t1_like["t1_study_date"].notna() & t1_like["t1_qc_status"].eq("pass")
    ]
    actual_eligible_subjects = set(pairing_eligible["t1_subject_id"])
    missing_catalog_subjects = sorted(selected_subjects - actual_eligible_subjects)
    if missing_catalog_subjects:
        raise ValueError(
            "Selected DTI subjects have no pairing-eligible T1 row in the supplied "
            "attested catalog: "
            + ", ".join(missing_catalog_subjects[:10])
        )

    chosen_rows: list[dict[str, Any]] = []
    alternative_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for dti_row in selected.sort_values("series_sid").itertuples(index=False):
        subject_options = t1_like[
            t1_like["t1_subject_id"].eq(dti_row.subject_id)
        ].copy()
        invalid_date = subject_options["t1_study_date"].isna()
        qc_fail = subject_options["t1_qc_status"].eq("fail")
        for row in subject_options[invalid_date | qc_fail].itertuples(index=False):
            rejected_rows.append(
                {
                    "series_sid": dti_row.series_sid,
                    "subject_id": dti_row.subject_id,
                    "dti_image_id": dti_row.dti_image_id,
                    "t1_image_id": row.t1_image_id,
                    "t1_study_date": (
                        row.t1_study_date.strftime("%Y-%m-%d")
                        if pd.notna(row.t1_study_date)
                        else ""
                    ),
                    "t1_description": row.t1_description,
                    "t1_qc_status": row.t1_qc_status,
                    "rejection_reason": (
                        "missing_or_invalid_study_date"
                        if pd.isna(row.t1_study_date)
                        else "explicit_qc_failure"
                    ),
                }
            )
        options = subject_options[~invalid_date & ~qc_fail].copy()
        if not options.empty:
            options["gap_days_signed"] = (
                options["t1_study_date"] - dti_row.dti_study_date
            ).dt.days
            options["gap_days_abs"] = options["gap_days_signed"].abs()
            options["qc_rank"] = options["t1_qc_status"].map(
                {"pass": 0, "unknown": 1}
            )
            options = options.sort_values(
                ["gap_days_abs", "qc_rank", "is_repeat", "t1_image_id"],
                ascending=[True, True, True, True],
                kind="mergesort",
            ).drop_duplicates("t1_image_id", keep="first")
            options["candidate_rank"] = range(1, len(options) + 1)
            options["pairing_window"] = options["gap_days_abs"].map(
                lambda value: classify_window(
                    int(value), primary_days, sensitivity_days
                )
            )
        for row in options.itertuples(index=False):
            alternative_rows.append(
                {
                    "series_sid": dti_row.series_sid,
                    "subject_id": dti_row.subject_id,
                    "group_provisional": dti_row.group_provisional,
                    "dti_image_id": dti_row.dti_image_id,
                    "dti_study_date": dti_row.dti_study_date.strftime("%Y-%m-%d"),
                    "t1_image_id": int(row.t1_image_id),
                    "t1_study_date": row.t1_study_date.strftime("%Y-%m-%d"),
                    "gap_days_signed": int(row.gap_days_signed),
                    "gap_days_abs": int(row.gap_days_abs),
                    "pairing_window": row.pairing_window,
                    "t1_qc_status": row.t1_qc_status,
                    "is_repeat": bool(row.is_repeat),
                    "candidate_rank": int(row.candidate_rank),
                    "selected_by_rule": int(row.candidate_rank) == 1,
                    "t1_description": row.t1_description,
                    "t1_visit": row.t1_visit,
                    "t1_phase": row.t1_phase,
                    "t1_type": row.t1_type,
                    "t1_series_uid": row.t1_series_uid,
                    "source_row": int(row.source_row),
                }
            )
            if row.pairing_window == "outside_sensitivity":
                rejected_rows.append(
                    {
                        "series_sid": dti_row.series_sid,
                        "subject_id": dti_row.subject_id,
                        "dti_image_id": dti_row.dti_image_id,
                        "t1_image_id": int(row.t1_image_id),
                        "t1_study_date": row.t1_study_date.strftime("%Y-%m-%d"),
                        "t1_description": row.t1_description,
                        "t1_qc_status": row.t1_qc_status,
                        "rejection_reason": "outside_sensitivity_window",
                    }
                )

        best = options.iloc[0] if not options.empty else None
        if best is None:
            pair_status = "exclude_no_valid_t1_candidate"
            pairing_window = "none"
        else:
            pairing_window = str(best["pairing_window"])
            if pairing_window == "outside_sensitivity":
                pair_status = "exclude_no_t1_within_sensitivity_window"
            elif best["t1_qc_status"] == "unknown":
                pair_status = f"{pairing_window}_pending_qc"
            else:
                pair_status = f"{pairing_window}_candidate"

        chosen_rows.append(
            {
                "series_sid": dti_row.series_sid,
                "subject_id": dti_row.subject_id,
                "group_provisional": dti_row.group_provisional,
                "diagnosis_status": "provisional_requires_scan_date_reconciliation",
                "band": dti_row.band,
                "dti_image_id": int(dti_row.dti_image_id),
                "dti_study_date": dti_row.dti_study_date.strftime("%Y-%m-%d"),
                "dti_visit": dti_row.dti_visit,
                "dti_phase": dti_row.dti_phase,
                "dti_description": dti_row.dti_description,
                "dti_protocol": dti_row.dti_protocol,
                "current_t1_image_id": dti_row.current_t1_image_id,
                "selected_t1_image_id": (
                    int(best["t1_image_id"]) if best is not None else pd.NA
                ),
                "selected_t1_study_date": (
                    best["t1_study_date"].strftime("%Y-%m-%d")
                    if best is not None
                    else ""
                ),
                "gap_days_signed": (
                    int(best["gap_days_signed"]) if best is not None else pd.NA
                ),
                "gap_days_abs": (
                    int(best["gap_days_abs"]) if best is not None else pd.NA
                ),
                "pairing_window": pairing_window,
                "t1_qc_status": (
                    best["t1_qc_status"] if best is not None else "missing"
                ),
                "t1_description": (
                    best["t1_description"] if best is not None else ""
                ),
                "t1_visit": best["t1_visit"] if best is not None else pd.NA,
                "t1_phase": best["t1_phase"] if best is not None else pd.NA,
                "t1_series_uid": (
                    best["t1_series_uid"] if best is not None else pd.NA
                ),
                "n_valid_t1_candidates": int(len(options)),
                "pair_status": pair_status,
                "current_t1_is_selected": (
                    bool(
                        best is not None
                        and pd.notna(dti_row.current_t1_image_id)
                        and int(best["t1_image_id"])
                        == int(dti_row.current_t1_image_id)
                    )
                ),
                "selection_rule": (
                    "min_abs_gap_days_then_qc_then_nonrepeat_then_image_id"
                ),
                "policy_approval_status": "awaiting_human_gate_SL-H01",
                "approved_for_download": False,
            }
        )

    chosen = pd.DataFrame(chosen_rows)
    alternatives = pd.DataFrame(alternative_rows)
    rejected = pd.DataFrame(rejected_rows)
    status_counts = (
        chosen.groupby(["pair_status", "group_provisional"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    flow = pd.DataFrame(
        [
            {"measure": "selected_dti_rows", "value": len(chosen)},
            *[
                {"measure": key, "value": value}
                for key, value in flow_counts.items()
            ],
            {
                "measure": "primary_candidates",
                "value": int(chosen["pairing_window"].eq("primary").sum()),
            },
            {
                "measure": "sensitivity_only_candidates",
                "value": int(
                    chosen["pairing_window"].eq("sensitivity_only").sum()
                ),
            },
            {
                "measure": "excluded_or_unresolved",
                "value": int(
                    chosen["pairing_window"].isin(
                        ["none", "outside_sensitivity"]
                    ).sum()
                ),
            },
            {
                "measure": "qc_pending_selected",
                "value": int(chosen["t1_qc_status"].eq("unknown").sum()),
            },
            {
                "measure": "approved_for_download",
                "value": int(chosen["approved_for_download"].sum()),
            },
        ]
    )
    validation = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "rules": {
            "primary_max_days": primary_days,
            "sensitivity_max_days": sensitivity_days,
            "ranking": [
                "absolute date gap ascending",
                "QC pass before unknown",
                "non-repeat before repeat",
                "Image ID ascending",
            ],
            "explicit_qc_failures_excluded": True,
            "download_approval_forced_false": True,
        },
        "counts": {
            "chosen_rows": len(chosen),
            "selected_subjects": len(selected_subjects),
            "selected_subjects_with_eligible_t1": len(
                selected_subjects & actual_eligible_subjects
            ),
            "selected_subjects_missing_eligible_t1": len(
                selected_subjects - actual_eligible_subjects
            ),
            "alternative_rows": len(alternatives),
            "rejected_rows": len(rejected),
            "duplicate_series_sid": int(chosen["series_sid"].duplicated().sum()),
            "approved_for_download": int(chosen["approved_for_download"].sum()),
        },
    }
    if len(chosen) != len(selected) or chosen["series_sid"].duplicated().any():
        raise RuntimeError("Pair manifest is not one deterministic row per selected DTI")
    if chosen["approved_for_download"].any():
        raise RuntimeError("Builder must never approve a download")
    return {
        "chosen": chosen,
        "alternatives": alternatives,
        "rejected": rejected,
        "status_counts": status_counts,
        "flow": flow,
        "validation": validation,
    }


def write_outputs(
    result: dict[str, Any],
    output_dir: Path,
    input_paths: dict[str, Path],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing pairing output path: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=str(output_dir.parent)
        )
    )
    try:
        result["chosen"].to_csv(staging / "acquisition_pair_manifest_v2.csv", index=False)
        result["alternatives"].to_csv(staging / "alternative_pairs_v2.csv", index=False)
        result["rejected"].to_csv(staging / "rejected_pairs_v2.csv", index=False)
        result["status_counts"].to_csv(staging / "pair_status_counts_v2.csv", index=False)
        result["flow"].to_csv(staging / "participant_pairing_flow_v2.csv", index=False)
        validation = dict(result["validation"])
        validation["inputs"] = {
            name: {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in input_paths.items()
        }
        (staging / "pairing_validation_v2.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dti", type=Path, required=True)
    parser.add_argument("--mri", type=Path, required=True)
    parser.add_argument("--mri-validation", type=Path, required=True)
    parser.add_argument("--series-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--primary-max-days", type=int, default=90)
    parser.add_argument("--sensitivity-max-days", type=int, default=180)
    parser.add_argument(
        "--band",
        default=None,
        help="Optional exact series-manifest band filter, for example good.",
    )
    args = parser.parse_args()

    dti = pd.read_csv(args.dti, low_memory=False)
    mri, catalog_attestation = load_attested_pairing_catalog(
        args.mri, args.mri_validation
    )
    manifest = pd.read_csv(args.series_manifest, low_memory=False)
    result = build_manifests(
        dti,
        mri,
        manifest,
        primary_days=args.primary_max_days,
        sensitivity_days=args.sensitivity_max_days,
        band=args.band,
        catalog_attestation=catalog_attestation,
    )
    result["validation"]["catalog_attestation"] = catalog_attestation.as_dict()
    write_outputs(
        result,
        args.out,
        {
            "dti": args.dti,
            "mri": args.mri,
            "mri_validation": args.mri_validation,
            "series_manifest": args.series_manifest,
        },
    )
    print(result["flow"].to_string(index=False))
    print(args.out.resolve())


if __name__ == "__main__":
    main()
