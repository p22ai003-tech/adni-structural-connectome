#!/usr/bin/env python3
"""Build the SL-D01 manifest for the 530 locally available DTI-T1 pairs.

This builder is deliberately closed-world and non-destructive.  It does not
select replacement images, download data, run image processing, or modify a
production path.  Instead it binds the current pair identities to their exact
dates, local paths, acquisition metadata, and *legacy* QC evidence.  Timing is
represented as an analysis limitation/sensitivity stratum; availability never
causes a long-gap pair to be described as visit-matched.

The per-directory hashes emitted here are metadata-inventory hashes over
relative file names and byte sizes.  They are not file-content hashes.  Full
content hashing belongs to the later immutable-input lock (SL-P0-09).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT = Path("/home/ec2-user/exp")
OUTPUT = PROJECT / "research_audit" / "outputs"

DEFAULT_EXACT = OUTPUT / "current_t1_exact_date_evidence_v2.csv"
DEFAULT_AUDIT = OUTPUT / "cohort_audit_manifest.csv"
DEFAULT_DRYRUN = OUTPUT / "visit_matched_t1_manifest_dryrun.csv"
DEFAULT_DTI_MASTER = PROJECT / "cohort" / "dti_master.csv"
DEFAULT_SUBJECT_MANIFEST = Path(
    "/data/derivatives/qc/sc_matrix_qc/subject_manifest.csv"
)
DEFAULT_AUDIT_RUN = OUTPUT / "audit_run_manifest.json"

OUTPUT_MANIFEST = "available_data_pair_manifest_v2.csv"
OUTPUT_STRATA = "available_data_pair_strata_v2.csv"
OUTPUT_FLOW = "available_data_pair_flow_v2.csv"
OUTPUT_VALIDATION = "available_data_pair_manifest_validation_v2.json"

EXPECTED_PAIRS = 530
SCHEMA_VERSION = "2.0"
SUBJECT_RE = re.compile(r"^\d{3}_S_\d{4}$")
SID_RE = re.compile(r"^(?P<subject>\d{3}_S_\d{4})_I(?P<image_id>[1-9]\d*)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
US_DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
PATH_DATE_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<hour>\d{2})_"
    r"(?P<minute>\d{2})_(?P<second>\d{2}(?:\.\d+)?))?$"
)


@dataclass(frozen=True)
class SourceTable:
    path: Path
    records: list[dict[str, str]]
    columns: tuple[str, ...]
    sha256: str
    size_bytes: int

    def evidence(self) -> dict[str, Any]:
        stat = self.path.stat()
        return {
            "path": str(self.path.resolve()),
            "size_bytes": self.size_bytes,
            "mtime_utc": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat(),
            "sha256": self.sha256,
            "row_count": len(self.records),
        }


@dataclass(frozen=True)
class DirectoryInventory:
    path: str
    file_count: int
    total_bytes: int
    names_sizes_sha256: str


@dataclass(frozen=True)
class DiagnosisAttachment:
    path: Path
    sha256: str
    rows: Mapping[tuple[str, int], Mapping[str, str]]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_source(path: Path, label: str) -> SourceTable:
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked {label} source: {path}")
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8 CSV") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None:
        raise ValueError(f"{label} has no CSV header")
    columns = [str(name).strip() for name in reader.fieldnames]
    if any(not name for name in columns):
        raise ValueError(f"{label} contains a blank CSV header")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{label} contains duplicate CSV headers")
    records: list[dict[str, str]] = []
    for line_number, raw in enumerate(reader, start=2):
        if None in raw:
            raise ValueError(f"{label} row {line_number} has extra fields")
        records.append(
            {
                str(key).strip(): "" if value is None else str(value).strip()
                for key, value in raw.items()
            }
        )
    return SourceTable(
        path=path,
        records=records,
        columns=tuple(columns),
        sha256=sha256_bytes(data),
        size_bytes=len(data),
    )


def require_columns(source: SourceTable, label: str, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(source.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def strict_positive_int(value: Any, label: str) -> int:
    token = str(value).strip()
    if not re.fullmatch(r"[1-9]\d*", token):
        raise ValueError(f"{label} must be a positive base-10 integer, found {value!r}")
    return int(token)


def strict_nonnegative_int(value: Any, label: str) -> int:
    token = str(value).strip()
    if not re.fullmatch(r"(?:0|[1-9]\d*)", token):
        raise ValueError(f"{label} must be a nonnegative integer, found {value!r}")
    return int(token)


def optional_int(value: Any, label: str) -> int | None:
    token = str(value).strip()
    if token == "":
        return None
    match = re.fullmatch(r"(?P<integer>0|[1-9]\d*)(?:\.0+)?", token)
    if not match:
        raise ValueError(f"{label} must be an integer value, found {value!r}")
    return int(match.group("integer"))


def strict_bool(value: Any, label: str) -> bool:
    token = str(value).strip().lower()
    if token == "true":
        return True
    if token == "false":
        return False
    raise ValueError(f"{label} must be true or false, found {value!r}")


def parse_exact_date(value: Any, label: str) -> date:
    token = str(value).strip()
    match = ISO_DATE_RE.fullmatch(token)
    if match:
        year, month, day = map(int, match.groups())
    else:
        match = US_DATE_RE.fullmatch(token)
        if not match:
            raise ValueError(f"{label} is not an exact calendar date: {value!r}")
        month, day, year = map(int, match.groups())
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid calendar date: {value!r}") from exc


def parse_signed_int(value: Any, label: str) -> int:
    token = str(value).strip()
    if not re.fullmatch(r"-?(?:0|[1-9]\d*)", token):
        raise ValueError(f"{label} must be an integer, found {value!r}")
    return int(token)


def require_subject(value: Any, label: str) -> str:
    subject = str(value).strip()
    if not SUBJECT_RE.fullmatch(subject):
        raise ValueError(f"{label} has invalid ADNI subject ID {value!r}")
    return subject


def unique_index(
    records: Sequence[Mapping[str, str]],
    key_function: Any,
    label: str,
) -> dict[Any, Mapping[str, str]]:
    result: dict[Any, Mapping[str, str]] = {}
    for row_number, record in enumerate(records, start=2):
        key = key_function(record)
        if key in result:
            raise ValueError(f"{label} has duplicate key {key!r} (row {row_number})")
        result[key] = record
    return result


def normalize_path(path_text: Any, label: str, *, directory: bool = False) -> Path:
    token = str(path_text).strip()
    if not token:
        raise ValueError(f"{label} path is empty")
    path = Path(token)
    if not path.is_absolute():
        raise ValueError(f"{label} path is not absolute: {path}")
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked {label} path: {path}")
    if directory and not path.is_dir():
        raise ValueError(f"{label} directory does not exist: {path}")
    if not directory and not path.exists():
        raise ValueError(f"{label} path does not exist: {path}")
    return path.resolve()


def directory_inventory(path: Path, label: str) -> DirectoryInventory:
    root = normalize_path(path, label, directory=True)
    files: list[tuple[str, int]] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise ValueError(f"Refusing symlink inside {label}: {candidate}")
        if candidate.is_file():
            files.append((candidate.relative_to(root).as_posix(), candidate.stat().st_size))
    if not files:
        raise ValueError(f"{label} contains no files: {root}")
    digest = hashlib.sha256()
    for relative_path, size in files:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
    return DirectoryInventory(
        path=str(root),
        file_count=len(files),
        total_bytes=sum(size for _, size in files),
        names_sizes_sha256=digest.hexdigest(),
    )


def resolve_image_directory(subject_root: Path, image_id: int, label: str) -> Path:
    root = normalize_path(subject_root, f"{label} subject root", directory=True)
    matches = sorted(
        (candidate.resolve() for candidate in root.rglob(f"I{image_id}") if candidate.is_dir()),
        key=lambda item: item.as_posix(),
    )
    if len(matches) != 1:
        raise ValueError(
            f"{label} I{image_id} must resolve to exactly one local directory below "
            f"{root}; found {len(matches)}"
        )
    if not matches[0].is_relative_to(root):
        raise ValueError(f"Resolved {label} series escapes its subject root")
    return matches[0]


def path_date(path: Path, label: str) -> tuple[str, str]:
    match = PATH_DATE_RE.fullmatch(path.parent.name)
    if not match:
        raise ValueError(f"{label} parent does not encode an exact date: {path.parent}")
    date_text = match.group("date")
    parse_exact_date(date_text, f"{label} path date")
    if match.group("hour") is None:
        return date_text, ""
    second = float(match.group("second"))
    second_integer = int(second)
    microsecond = int(round((second - second_integer) * 1_000_000))
    value = datetime(
        *map(int, date_text.split("-")),
        int(match.group("hour")),
        int(match.group("minute")),
        second_integer,
        microsecond,
    )
    return date_text, value.isoformat()


def timing_stratum(gap_days_abs: int) -> str:
    if gap_days_abs <= 90:
        return "le_90_days"
    if gap_days_abs <= 180:
        return "days_91_180"
    return "gt_180_days"


def load_diagnosis_attachment(
    path: Path | None,
    expected_sha256: str | None,
    expected_keys: set[tuple[str, int]],
) -> DiagnosisAttachment | None:
    if path is None and expected_sha256 is None:
        return None
    if path is None or expected_sha256 is None:
        raise ValueError(
            "Diagnosis attachment requires both a CSV path and its expected SHA-256"
        )
    expected = expected_sha256.strip().lower()
    if not SHA256_RE.fullmatch(expected):
        raise ValueError("Diagnosis attachment expected SHA-256 is malformed")
    source = read_csv_source(path, "diagnosis reconciliation")
    if source.sha256 != expected:
        raise ValueError("Diagnosis reconciliation SHA-256 does not match")
    require_columns(source, "diagnosis reconciliation", {"subject_id"})
    candidate_key_fields = [
        field
        for field in ("selected_dti_image_id", "dti_image_id")
        if field in source.columns
    ]
    if len(candidate_key_fields) != 1:
        raise ValueError(
            "Diagnosis reconciliation must contain exactly one selected-DTI key "
            "column: selected_dti_image_id or dti_image_id"
        )
    image_key = candidate_key_fields[0]
    rows = unique_index(
        source.records,
        lambda row: (
            require_subject(row["subject_id"], "diagnosis reconciliation"),
            strict_positive_int(
                row[image_key], "diagnosis reconciliation selected DTI image ID"
            ),
        ),
        "diagnosis reconciliation",
    )
    actual_keys = set(rows)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:10]
        extra = sorted(actual_keys - expected_keys)[:10]
        raise ValueError(
            "Diagnosis reconciliation keys do not exactly cover the pair manifest; "
            f"missing={missing}, extra={extra}"
        )
    return DiagnosisAttachment(path=path, sha256=source.sha256, rows=rows)


def _source_bundle_sha256(sources: Mapping[str, SourceTable], audit_run_hash: str) -> str:
    payload = {
        name: source.sha256 for name, source in sorted(sources.items())
    }
    payload["audit_run_manifest"] = audit_run_hash
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _record_sha256(record: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in record.items()
        if key != "pair_record_sha256"
    }
    return sha256_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _optional_path_status(value: str) -> tuple[str, bool]:
    token = str(value).strip()
    return token, bool(token and Path(token).exists())


def build_available_data_manifest(
    *,
    exact_path: Path,
    audit_path: Path,
    dryrun_path: Path,
    dti_master_path: Path,
    subject_manifest_path: Path,
    audit_run_path: Path,
    expected_pairs: int = EXPECTED_PAIRS,
    diagnosis_path: Path | None = None,
    diagnosis_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and assemble the closed-world pair manifest in memory."""

    exact = read_csv_source(exact_path, "current exact-date evidence")
    audit = read_csv_source(audit_path, "cohort audit manifest")
    dryrun = read_csv_source(dryrun_path, "legacy T1 dry-run manifest")
    dti_master = read_csv_source(dti_master_path, "DTI master")
    subject_manifest = read_csv_source(subject_manifest_path, "subject manifest")

    require_columns(
        exact,
        "current exact-date evidence",
        {
            "subject_id",
            "group",
            "dti_image_id",
            "dti_study_date",
            "dti_visit",
            "dti_master_phase",
            "current_t1_image_id",
            "current_t1_local_subject_id",
            "current_t1_path_date",
            "current_t1_path_datetime",
            "current_t1_gap_days_signed",
            "current_t1_gap_days_abs",
            "current_within_90_days",
            "current_within_180_days",
            "current_t1_image_dir",
            "current_t1_file_count",
            "current_t1_total_bytes",
            "current_exact_date_source",
            "recommended_action",
        },
    )
    require_columns(
        audit,
        "cohort audit manifest",
        {
            "subject_id",
            "actual_dti_image_id",
            "actual_t1_image_id",
            "Image ID",
            "Image ID_mri",
            "Research Group",
            "diagnosis_original",
            "group_dashboard",
            "group_corrected",
            "smc_misclassified_as_mci",
            "dti_phase",
            "dti_age",
            "dti_sex",
            "dti_study_date",
            "dti_visit",
            "dti_protocol",
            "dti_description",
            "dti_type",
            "t1_phase",
            "t1_age",
            "t1_description",
            "t1_type",
            "manufacturer",
            "field_strength_t",
            "gradient_directions",
            "protocol_key",
            "site",
            "dti_t1_phase_match",
            "dti_t1_age_gap_years",
            "sc_matrix_qc_status",
            "sc_matrix_qc_gate",
            "sc_matrix_qc_include",
            "n_sc_qc_series",
            "n_sc_qc_failed_series",
            "sc_matrix_qc_statuses",
            "density_y",
            "band",
            "radial",
            "mask_quality",
            "reg_ncc",
            "label_survival",
            "t1_path",
            "parc_path",
            "tck_path",
            "sift_path",
        },
    )
    require_columns(
        dryrun,
        "legacy T1 dry-run manifest",
        {
            "subject_id",
            "group",
            "dti_image_id",
            "dti_phase",
            "current_t1_image_id",
            "current_t1_phase",
            "current_gap_years",
            "current_is_contemporaneous",
            "recommended_action",
        },
    )
    require_columns(
        dti_master,
        "DTI master",
        {
            "Subject ID",
            "Image ID",
            "Study Date",
            "Visit",
            "Phase",
            "Research Group",
            "Description",
            "Type",
            "Imaging Protocol",
            "Age",
            "Sex",
        },
    )
    require_columns(
        subject_manifest,
        "subject manifest",
        {
            "sid",
            "group",
            "band",
            "raw_dwi_path",
            "raw_t1_path",
            "t1_path",
            "parc_path",
            "tck_path",
            "sift_path",
            "tsf_path",
            "density",
            "reg_ncc",
            "label_survival",
            "mask_quality",
        },
    )

    if len(exact.records) != expected_pairs:
        raise ValueError(
            f"Exact-date evidence must contain {expected_pairs} rows, found {len(exact.records)}"
        )

    exact_index = unique_index(
        exact.records,
        lambda row: require_subject(row["subject_id"], "exact-date evidence"),
        "current exact-date evidence",
    )
    audit_index = unique_index(
        audit.records,
        lambda row: require_subject(row["subject_id"], "cohort audit manifest"),
        "cohort audit manifest",
    )
    dryrun_index = unique_index(
        dryrun.records,
        lambda row: require_subject(row["subject_id"], "legacy T1 dry-run manifest"),
        "legacy T1 dry-run manifest",
    )
    if set(audit_index) != set(exact_index) or set(dryrun_index) != set(exact_index):
        raise ValueError("Exact-date, cohort-audit, and dry-run subject sets differ")

    selected_subject_rows: list[Mapping[str, str]] = []
    for row in subject_manifest.records:
        if row["band"] != "good":
            continue
        match = SID_RE.fullmatch(row["sid"])
        if not match:
            raise ValueError(f"Cannot parse good-band subject-manifest sid {row['sid']!r}")
        selected_subject_rows.append(row)
    selected_index = unique_index(
        selected_subject_rows,
        lambda row: SID_RE.fullmatch(row["sid"]).group("subject"),  # type: ignore[union-attr]
        "good-band subject manifest",
    )
    if len(selected_index) != expected_pairs or set(selected_index) != set(exact_index):
        raise ValueError(
            "The good-band subject manifest does not exactly match the current pair cohort"
        )

    selected_dti_ids = {
        strict_positive_int(row["dti_image_id"], "exact dti_image_id")
        for row in exact.records
    }
    relevant_dti_rows = [
        row
        for row in dti_master.records
        if strict_positive_int(row["Image ID"], "DTI master Image ID")
        in selected_dti_ids
    ]
    dti_index = unique_index(
        relevant_dti_rows,
        lambda row: strict_positive_int(row["Image ID"], "DTI master Image ID"),
        "selected DTI master rows",
    )
    if set(dti_index) != selected_dti_ids:
        missing = sorted(selected_dti_ids - set(dti_index))
        raise ValueError(f"Selected DTI IDs are absent from dti_master: {missing[:10]}")

    audit_run_path = normalize_path(audit_run_path, "audit run manifest")
    audit_run_bytes = audit_run_path.read_bytes()
    audit_run_hash = sha256_bytes(audit_run_bytes)
    try:
        audit_run = json.loads(audit_run_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Audit run manifest is not valid UTF-8 JSON") from exc
    recorded_inputs = audit_run.get("inputs", {})
    if recorded_inputs.get("dti_master", {}).get("sha256") != dti_master.sha256:
        raise ValueError("Audit run is not checksum-bound to the current dti_master")
    if recorded_inputs.get("manifest", {}).get("sha256") != subject_manifest.sha256:
        raise ValueError("Audit run is not checksum-bound to the current subject manifest")

    sources = {
        "current_t1_exact_date_evidence": exact,
        "cohort_audit_manifest": audit,
        "legacy_t1_dryrun_manifest": dryrun,
        "dti_master": dti_master,
        "subject_manifest": subject_manifest,
    }
    bundle_hash = _source_bundle_sha256(sources, audit_run_hash)
    expected_diagnosis_keys = {
        (
            subject,
            strict_positive_int(exact_index[subject]["dti_image_id"], "dti_image_id"),
        )
        for subject in exact_index
    }
    diagnosis = load_diagnosis_attachment(
        diagnosis_path, diagnosis_sha256, expected_diagnosis_keys
    )

    records: list[dict[str, Any]] = []
    for subject in sorted(exact_index):
        exact_row = exact_index[subject]
        audit_row = audit_index[subject]
        dry_row = dryrun_index[subject]
        selected_row = selected_index[subject]

        dti_id = strict_positive_int(exact_row["dti_image_id"], "dti_image_id")
        t1_id = strict_positive_int(
            exact_row["current_t1_image_id"], "current_t1_image_id"
        )
        dti_row = dti_index[dti_id]
        _assert_equal(
            require_subject(dti_row["Subject ID"], "DTI master Subject ID"),
            subject,
            f"{subject} DTI master subject",
        )
        _assert_equal(
            strict_positive_int(audit_row["actual_dti_image_id"], "actual DTI ID"),
            dti_id,
            f"{subject} cohort-audit DTI ID",
        )
        _assert_equal(
            strict_positive_int(audit_row["actual_t1_image_id"], "actual T1 ID"),
            t1_id,
            f"{subject} cohort-audit T1 ID",
        )
        _assert_equal(
            strict_positive_int(dry_row["dti_image_id"], "dry-run DTI ID"),
            dti_id,
            f"{subject} dry-run DTI ID",
        )
        _assert_equal(
            strict_positive_int(dry_row["current_t1_image_id"], "dry-run T1 ID"),
            t1_id,
            f"{subject} dry-run T1 ID",
        )
        sid_match = SID_RE.fullmatch(selected_row["sid"])
        assert sid_match is not None
        _assert_equal(
            int(sid_match.group("image_id")),
            dti_id,
            f"{subject} subject-manifest DTI ID",
        )

        dti_date = parse_exact_date(dti_row["Study Date"], f"{subject} DTI date")
        exact_dti_date = parse_exact_date(
            exact_row["dti_study_date"], f"{subject} exact-evidence DTI date"
        )
        audit_dti_date = parse_exact_date(
            audit_row["dti_study_date"], f"{subject} audit DTI date"
        )
        _assert_equal(dti_date, exact_dti_date, f"{subject} DTI exact date")
        _assert_equal(dti_date, audit_dti_date, f"{subject} audit DTI exact date")
        t1_date = parse_exact_date(
            exact_row["current_t1_path_date"], f"{subject} T1 path date"
        )
        gap_signed = (t1_date - dti_date).days
        gap_abs = abs(gap_signed)
        _assert_equal(
            parse_signed_int(
                exact_row["current_t1_gap_days_signed"], "signed exact gap"
            ),
            gap_signed,
            f"{subject} signed exact gap",
        )
        _assert_equal(
            strict_nonnegative_int(
                exact_row["current_t1_gap_days_abs"], "absolute exact gap"
            ),
            gap_abs,
            f"{subject} absolute exact gap",
        )
        _assert_equal(
            strict_bool(exact_row["current_within_90_days"], "within-90 flag"),
            gap_abs <= 90,
            f"{subject} within-90 flag",
        )
        _assert_equal(
            strict_bool(exact_row["current_within_180_days"], "within-180 flag"),
            gap_abs <= 180,
            f"{subject} within-180 flag",
        )

        _assert_equal(
            exact_row["dti_visit"], dti_row["Visit"], f"{subject} DTI visit"
        )
        _assert_equal(
            exact_row["dti_master_phase"],
            dti_row["Phase"],
            f"{subject} DTI phase",
        )
        _assert_equal(
            audit_row["dti_phase"], dti_row["Phase"], f"{subject} audit DTI phase"
        )
        _assert_equal(
            audit_row["dti_visit"], dti_row["Visit"], f"{subject} audit DTI visit"
        )
        _assert_equal(
            audit_row["t1_phase"],
            dry_row["current_t1_phase"],
            f"{subject} current T1 phase",
        )
        phase_match = audit_row["dti_phase"] == audit_row["t1_phase"]
        _assert_equal(
            strict_bool(audit_row["dti_t1_phase_match"], "phase-match flag"),
            phase_match,
            f"{subject} phase-match flag",
        )
        _assert_equal(
            require_subject(
                exact_row["current_t1_local_subject_id"], "local T1 subject"
            ),
            subject,
            f"{subject} local T1 subject",
        )

        _assert_equal(
            exact_row["group"],
            audit_row["group_corrected"],
            f"{subject} provisional diagnosis label",
        )
        _assert_equal(
            exact_row["group"],
            dry_row["group"],
            f"{subject} dry-run diagnosis label",
        )
        _assert_equal(
            audit_row["diagnosis_original"],
            dti_row["Research Group"],
            f"{subject} DTI-row diagnosis label",
        )

        dti_subject_root = normalize_path(
            selected_row["raw_dwi_path"], f"{subject} local DTI root", directory=True
        )
        t1_subject_root = normalize_path(
            selected_row["raw_t1_path"], f"{subject} local T1 root", directory=True
        )
        dti_series = resolve_image_directory(dti_subject_root, dti_id, "DTI")
        t1_series = normalize_path(
            exact_row["current_t1_image_dir"],
            f"{subject} local T1 series",
            directory=True,
        )
        if t1_series.name != f"I{t1_id}" or not t1_series.is_relative_to(t1_subject_root):
            raise ValueError(
                f"{subject} T1 series does not match I{t1_id} inside its subject root"
            )
        dti_path_date, dti_path_datetime = path_date(dti_series, f"{subject} DTI")
        t1_path_date, t1_path_datetime = path_date(t1_series, f"{subject} T1")
        _assert_equal(dti_path_date, dti_date.isoformat(), f"{subject} DTI path date")
        _assert_equal(t1_path_date, t1_date.isoformat(), f"{subject} T1 path date")
        if exact_row["current_t1_path_datetime"]:
            _assert_equal(
                exact_row["current_t1_path_datetime"],
                t1_path_datetime,
                f"{subject} T1 path datetime",
            )

        dti_inventory = directory_inventory(dti_series, f"{subject} DTI series")
        t1_inventory = directory_inventory(t1_series, f"{subject} T1 series")
        _assert_equal(
            t1_inventory.file_count,
            strict_nonnegative_int(
                exact_row["current_t1_file_count"], "T1 exact-evidence file count"
            ),
            f"{subject} T1 file count",
        )
        _assert_equal(
            t1_inventory.total_bytes,
            strict_nonnegative_int(
                exact_row["current_t1_total_bytes"], "T1 exact-evidence bytes"
            ),
            f"{subject} T1 total bytes",
        )

        legacy_qc_include = strict_bool(
            audit_row["sc_matrix_qc_include"], "legacy QC include flag"
        )
        stratum = timing_stratum(gap_abs)
        if stratum == "gt_180_days":
            visit_status = "not_visit_matched_long_gap"
        else:
            visit_status = "not_asserted_t1_visit_metadata_unavailable"

        attachment_payload: dict[str, str] = {}
        if diagnosis is not None:
            attached = diagnosis.rows[(subject, dti_id)]
            attachment_payload = {
                key: value
                for key, value in attached.items()
                if key not in {"subject_id", "dti_image_id", "selected_dti_image_id"}
            }

        def attached_bool(field: str) -> bool | str:
            value = attachment_payload.get(field, "")
            return "" if value == "" else strict_bool(value, f"diagnosis {field}")

        processed_t1_path, processed_t1_exists = _optional_path_status(
            audit_row["t1_path"]
        )
        parc_path, parc_exists = _optional_path_status(audit_row["parc_path"])
        tck_path, tck_exists = _optional_path_status(audit_row["tck_path"])
        sift_path, sift_exists = _optional_path_status(audit_row["sift_path"])

        record: dict[str, Any] = {
            "manifest_schema_version": SCHEMA_VERSION,
            "data_scope_decision": "SL-D01_local_only",
            "analysis_role": "exploratory_available_data",
            "subject_id": subject,
            "pair_id": f"{subject}_I{dti_id}_I{t1_id}",
            "diagnosis_join_key": f"{subject}|I{dti_id}",
            "dti_image_id": dti_id,
            "current_t1_image_id": t1_id,
            "dti_study_date": dti_date.isoformat(),
            "current_t1_study_date": t1_date.isoformat(),
            "current_t1_date_source": exact_row["current_exact_date_source"],
            "dti_t1_gap_days_signed": gap_signed,
            "dti_t1_gap_days_abs": gap_abs,
            "timing_stratum": stratum,
            "include_timing_le_90_sensitivity": gap_abs <= 90,
            "include_timing_le_180_sensitivity": gap_abs <= 180,
            "include_full_available_exploratory": True,
            "visit_matched_claim": False,
            "visit_match_status": visit_status,
            "available_data_action": "retain_current_local_pair_as_is",
            "legacy_replacement_branch_status": "retired_by_SL-D01",
            "legacy_pairing_recommended_action": exact_row["recommended_action"],
            "legacy_dryrun_recommended_action": dry_row["recommended_action"],
            "legacy_rounded_age_contemporaneous_flag": strict_bool(
                dry_row["current_is_contemporaneous"],
                "legacy rounded-age contemporaneous flag",
            ),
            "legacy_rounded_age_gap_years": dry_row["current_gap_years"],
            "dti_visit": dti_row["Visit"],
            "dti_phase": dti_row["Phase"],
            "current_t1_phase": audit_row["t1_phase"],
            "same_adni_phase": phase_match,
            "cross_adni_phase": not phase_match,
            "phase_relation": "same_phase" if phase_match else "cross_phase",
            "dti_master_research_group_raw": dti_row["Research Group"],
            "cohort_audit_diagnosis_original": audit_row["diagnosis_original"],
            "cohort_audit_group_corrected_provisional": audit_row[
                "group_corrected"
            ],
            "dashboard_group_raw": audit_row["group_dashboard"],
            "dashboard_research_group_raw": audit_row["Research Group"],
            "smc_misclassified_as_mci_in_dashboard": strict_bool(
                audit_row["smc_misclassified_as_mci"], "SMC dashboard flag"
            ),
            "diagnosis_reconciliation_status": (
                "attached_checksum_verified"
                if diagnosis is not None
                else "not_attached_pending_SL-P0-05"
            ),
            "diagnosis_reconciliation_source_path": (
                str(diagnosis.path.resolve()) if diagnosis is not None else ""
            ),
            "diagnosis_reconciliation_source_sha256": (
                diagnosis.sha256 if diagnosis is not None else ""
            ),
            "diagnosis_reconciliation_payload_json": json.dumps(
                attachment_payload,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "diagnosis_reconciled_raw_at_dti": attachment_payload.get(
                "diagnosis_raw_at_dti", ""
            ),
            "diagnosis_reconciled_harmonized": attachment_payload.get(
                "diagnosis_harmonized", ""
            ),
            "diagnosis_reconciled_harmonization_rule": attachment_payload.get(
                "diagnosis_harmonization_rule", ""
            ),
            "diagnosis_reconciled_resolution_status": attachment_payload.get(
                "diagnosis_resolution_status", ""
            ),
            "diagnosis_reconciled_resolution_method": attachment_payload.get(
                "diagnosis_resolution_method", ""
            ),
            "diagnosis_reconciled_source_catalog": attachment_payload.get(
                "diagnosis_source_catalog", ""
            ),
            "diagnosis_reconciled_source_catalog_sha256": attachment_payload.get(
                "diagnosis_source_catalog_sha256", ""
            ),
            "diagnosis_reconciled_day_offset": attachment_payload.get(
                "diagnosis_day_offset", ""
            ),
            "diagnosis_reconciled_is_exact_dti_date": attached_bool(
                "diagnosis_is_exact_dti_date"
            ),
            "diagnosis_reconciled_primary_analysis_group": attachment_payload.get(
                "primary_analysis_group", ""
            ),
            "diagnosis_reconciled_primary_analysis_eligible": attached_bool(
                "primary_analysis_eligible"
            ),
            "diagnosis_reconciled_smc_retained_separately": attached_bool(
                "smc_retained_separately"
            ),
            "diagnosis_reconciled_dti_identity_mismatch": attached_bool(
                "dti_identity_mismatch"
            ),
            "diagnosis_reconciled_dti_identity_resolution_status": attachment_payload.get(
                "dti_identity_resolution_status", ""
            ),
            "dti_age_at_scan": dti_row["Age"],
            "dti_sex": dti_row["Sex"],
            "dti_description": dti_row["Description"],
            "dti_type": dti_row["Type"],
            "dti_imaging_protocol_raw": dti_row["Imaging Protocol"],
            "dti_manufacturer": audit_row["manufacturer"],
            "dti_field_strength_t": audit_row["field_strength_t"],
            "dti_gradient_directions": audit_row["gradient_directions"],
            "dti_protocol_key": audit_row["protocol_key"],
            "site": audit_row["site"],
            "current_t1_age_metadata": audit_row["t1_age"],
            "current_t1_description": audit_row["t1_description"],
            "current_t1_type": audit_row["t1_type"],
            "current_t1_is_original": audit_row["t1_type"].lower() == "original",
            "current_t1_image_qc_status": "not_available_in_current_sources",
            "legacy_sc_matrix_qc_status": audit_row["sc_matrix_qc_status"],
            "legacy_sc_matrix_qc_gate": audit_row["sc_matrix_qc_gate"],
            "legacy_sc_matrix_qc_include": legacy_qc_include,
            "legacy_sc_matrix_qc_series_count": optional_int(
                audit_row["n_sc_qc_series"], "legacy QC series count"
            ),
            "legacy_sc_matrix_qc_failed_series_count": optional_int(
                audit_row["n_sc_qc_failed_series"], "legacy QC failed-series count"
            ),
            "legacy_sc_matrix_qc_statuses": audit_row["sc_matrix_qc_statuses"],
            "legacy_connectome_density": audit_row["density_y"],
            "legacy_density_band": audit_row["band"],
            "legacy_radial_recipe": audit_row["radial"],
            "legacy_mask_quality": audit_row["mask_quality"],
            "legacy_registration_ncc": audit_row["reg_ncc"],
            "legacy_label_survival": audit_row["label_survival"],
            "corrected_rerun_qc_status": "not_run",
            "corrected_rerun_analysis_eligible": False,
            "available_for_corrected_rerun": True,
            "local_dti_subject_root": str(dti_subject_root),
            "local_dti_series_dir": str(dti_series),
            "local_dti_path_date": dti_path_date,
            "local_dti_path_datetime": dti_path_datetime,
            "local_dti_file_count": dti_inventory.file_count,
            "local_dti_total_bytes": dti_inventory.total_bytes,
            "local_dti_names_sizes_sha256": dti_inventory.names_sizes_sha256,
            "local_t1_subject_root": str(t1_subject_root),
            "local_t1_series_dir": str(t1_series),
            "local_t1_path_date": t1_path_date,
            "local_t1_path_datetime": t1_path_datetime,
            "local_t1_file_count": t1_inventory.file_count,
            "local_t1_total_bytes": t1_inventory.total_bytes,
            "local_t1_names_sizes_sha256": t1_inventory.names_sizes_sha256,
            "local_series_hash_kind": "sha256_sorted_relative_names_and_sizes_not_content",
            "local_content_sha256_status": "deferred_to_SL-P0-09",
            "legacy_processed_t1_path": processed_t1_path,
            "legacy_processed_t1_path_exists": processed_t1_exists,
            "legacy_parcellation_path": parc_path,
            "legacy_parcellation_path_exists": parc_exists,
            "legacy_tractogram_path": tck_path,
            "legacy_tractogram_path_exists": tck_exists,
            "legacy_sift_weights_path": sift_path,
            "legacy_sift_weights_path_exists": sift_exists,
            "provenance_source_bundle_sha256": bundle_hash,
        }
        record["pair_record_sha256"] = _record_sha256(record)
        records.append(record)

    if len(records) != expected_pairs:
        raise ValueError(f"Built {len(records)} rows, expected {expected_pairs}")
    long_gap_bad = [
        row
        for row in records
        if row["timing_stratum"] == "gt_180_days"
        and (
            row["visit_matched_claim"] is not False
            or "not_visit_matched_long_gap" != row["visit_match_status"]
        )
    ]
    if long_gap_bad:
        raise ValueError("A long-gap pair was incorrectly described as visit-matched")

    return {
        "records": records,
        "sources": sources,
        "audit_run": {
            "path": str(audit_run_path),
            "sha256": audit_run_hash,
            "size_bytes": len(audit_run_bytes),
            "generated_utc": audit_run.get("generated_utc", ""),
            "script": audit_run.get("script", ""),
        },
        "source_bundle_sha256": bundle_hash,
        "diagnosis_attachment": diagnosis,
    }


def build_strata(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dimensions = [
        ("timing_stratum", "timing_stratum"),
        ("phase_relation", "phase_relation"),
        ("dti_master_diagnosis_raw", "dti_master_research_group_raw"),
        ("diagnosis_reconciled_harmonized", "diagnosis_reconciled_harmonized"),
        ("dti_protocol_key", "dti_protocol_key"),
        ("site", "site"),
        ("current_t1_type", "current_t1_type"),
        ("legacy_sc_matrix_qc_status", "legacy_sc_matrix_qc_status"),
        ("legacy_sc_matrix_qc_gate", "legacy_sc_matrix_qc_gate"),
    ]
    rows: list[dict[str, Any]] = []
    for dimension, field in dimensions:
        levels = sorted({str(record[field]) for record in records})
        for level in levels:
            selected = [record for record in records if str(record[field]) == level]
            rows.append(
                {
                    "stratum_dimension": dimension,
                    "stratum_level": level,
                    "pair_count": len(selected),
                    "timing_le_90_count": sum(
                        record["timing_stratum"] == "le_90_days"
                        for record in selected
                    ),
                    "timing_91_180_count": sum(
                        record["timing_stratum"] == "days_91_180"
                        for record in selected
                    ),
                    "timing_gt_180_count": sum(
                        record["timing_stratum"] == "gt_180_days"
                        for record in selected
                    ),
                    "legacy_qc_include_count": sum(
                        record["legacy_sc_matrix_qc_include"] is True
                        for record in selected
                    ),
                    "legacy_qc_exclude_count": sum(
                        record["legacy_sc_matrix_qc_include"] is False
                        for record in selected
                    ),
                    "corrected_rerun_qc_not_run_count": sum(
                        record["corrected_rerun_qc_status"] == "not_run"
                        for record in selected
                    ),
                }
            )
    return rows


def build_flow(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    definitions = [
        (
            "fixed_local_pair_cohort",
            "Current selected DTI and T1 IDs retained under SL-D01",
            lambda row: row["include_full_available_exploratory"],
            "Exploratory available-data denominator; not a validity claim.",
        ),
        (
            "exact_dates_and_intervals_complete",
            "Both exact dates present and signed/absolute interval recomputed",
            lambda row: bool(row["dti_study_date"] and row["current_t1_study_date"]),
            "Timing is measured rather than inferred from rounded age.",
        ),
        (
            "local_series_resolved",
            "Selected DTI and T1 local image directories each resolve uniquely",
            lambda row: row["available_for_corrected_rerun"],
            "Availability only; corrected processing and QC have not run.",
        ),
        (
            "timing_le_90_days",
            "Absolute DTI-T1 interval <=90 days",
            lambda row: row["timing_stratum"] == "le_90_days",
            "Prespecified timing sensitivity; not asserted to be the same visit.",
        ),
        (
            "timing_91_180_days",
            "Absolute DTI-T1 interval 91-180 days",
            lambda row: row["timing_stratum"] == "days_91_180",
            "Intermediate timing sensitivity stratum.",
        ),
        (
            "timing_gt_180_days",
            "Absolute DTI-T1 interval >180 days",
            lambda row: row["timing_stratum"] == "gt_180_days",
            "Long-gap available pairs; never labeled visit-matched.",
        ),
        (
            "same_adni_phase",
            "DTI phase equals current T1 phase",
            lambda row: row["same_adni_phase"],
            "Phase equality is not a visit-match claim.",
        ),
        (
            "cross_adni_phase",
            "DTI phase differs from current T1 phase",
            lambda row: row["cross_adni_phase"],
            "Explicit cross-phase acquisition limitation.",
        ),
        (
            "legacy_qc_include",
            "Existing matrix passed the legacy include/warn gate",
            lambda row: row["legacy_sc_matrix_qc_include"],
            "Historical QC only; does not approve corrected inference.",
        ),
        (
            "diagnosis_reconciled_primary_analysis_eligible",
            "Checksum-bound diagnosis reconciliation assigns CN, MCI, or AD",
            lambda row: row["diagnosis_reconciled_primary_analysis_eligible"] is True,
            "SMC and any unresolved diagnosis remain outside the primary CN/MCI/AD cohort.",
        ),
        (
            "corrected_rerun_qc_complete",
            "Corrected recipe has completed current technical QC",
            lambda row: row["corrected_rerun_qc_status"] != "not_run",
            "Expected zero before the corrected rerun.",
        ),
    ]
    return [
        {
            "order": index,
            "flow_or_stratum": name,
            "criterion": criterion,
            "pair_count": sum(bool(predicate(row)) for row in records),
            "interpretation": interpretation,
        }
        for index, (name, criterion, predicate, interpretation) in enumerate(
            definitions, start=1
        )
    ]


def _csv_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    if not records:
        raise ValueError("Cannot write an empty CSV")
    fieldnames = list(records[0])
    for record in records:
        if list(record) != fieldnames:
            raise ValueError("CSV records do not share an identical ordered schema")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for record in records:
        formatted = {
            key: (
                "true"
                if value is True
                else "false"
                if value is False
                else ""
                if value is None
                else value
            )
            for key, value in record.items()
        }
        writer.writerow(formatted)
    return stream.getvalue().encode("utf-8")


def _output_evidence(path: Path, data: bytes, row_count: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": len(data),
        "sha256": sha256_bytes(data),
        "row_count": row_count,
    }


def write_release(
    result: Mapping[str, Any], output_dir: Path, generated_utc: str | None = None
) -> dict[str, Any]:
    """Atomically create a write-once release; existing outputs are never replaced."""

    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "manifest": output_dir / OUTPUT_MANIFEST,
        "strata": output_dir / OUTPUT_STRATA,
        "flow": output_dir / OUTPUT_FLOW,
        "validation": output_dir / OUTPUT_VALIDATION,
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to replace existing available-data release files: "
            + ", ".join(existing)
        )

    records = result["records"]
    strata = build_strata(records)
    flow = build_flow(records)
    manifest_bytes = _csv_bytes(records)
    strata_bytes = _csv_bytes(strata)
    flow_bytes = _csv_bytes(flow)

    counts = {
        "pairs": len(records),
        "subjects": len({record["subject_id"] for record in records}),
        "unique_dti_image_ids": len({record["dti_image_id"] for record in records}),
        "unique_t1_image_ids": len(
            {record["current_t1_image_id"] for record in records}
        ),
        "timing_le_90_days": sum(
            record["timing_stratum"] == "le_90_days" for record in records
        ),
        "timing_91_180_days": sum(
            record["timing_stratum"] == "days_91_180" for record in records
        ),
        "timing_gt_180_days": sum(
            record["timing_stratum"] == "gt_180_days" for record in records
        ),
        "same_phase": sum(record["same_adni_phase"] is True for record in records),
        "cross_phase": sum(record["cross_adni_phase"] is True for record in records),
        "legacy_qc_include": sum(
            record["legacy_sc_matrix_qc_include"] is True for record in records
        ),
        "legacy_qc_exclude": sum(
            record["legacy_sc_matrix_qc_include"] is False for record in records
        ),
        "t1_original": sum(record["current_t1_is_original"] is True for record in records),
        "t1_non_original": sum(
            record["current_t1_is_original"] is False for record in records
        ),
        "corrected_rerun_qc_complete": sum(
            record["corrected_rerun_qc_status"] != "not_run" for record in records
        ),
        "diagnosis_reconciled_rows": sum(
            record["diagnosis_reconciliation_status"]
            == "attached_checksum_verified"
            for record in records
        ),
        "diagnosis_primary_analysis_eligible": sum(
            record["diagnosis_reconciled_primary_analysis_eligible"] is True
            for record in records
        ),
        "diagnosis_smc_retained_separately": sum(
            record["diagnosis_reconciled_smc_retained_separately"] is True
            for record in records
        ),
        "long_gap_visit_matched_claims": sum(
            record["timing_stratum"] == "gt_180_days"
            and record["visit_matched_claim"] is True
            for record in records
        ),
    }
    diagnosis = result["diagnosis_attachment"]
    generated = generated_utc or datetime.now(timezone.utc).isoformat()
    validation = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "release_status": (
            "READY_FOR_HUMAN_POLICY_GATE"
            if diagnosis is not None
            else "PROVISIONAL_DIAGNOSIS_JOIN_PENDING"
        ),
        "generated_utc": generated,
        "scope": {
            "decision": "SL-D01",
            "local_only": True,
            "replacement_selection_performed": False,
            "image_transfer_performed": False,
            "image_processing_performed": False,
            "full_cohort_role": "exploratory_available_data",
        },
        "sources": {
            name: source.evidence()
            for name, source in sorted(result["sources"].items())
        },
        "audit_run_binding": result["audit_run"],
        "source_bundle_sha256": result["source_bundle_sha256"],
        "diagnosis_attachment": {
            "status": "attached_checksum_verified" if diagnosis else "pending_SL-P0-05",
            "path": str(diagnosis.path.resolve()) if diagnosis else "",
            "sha256": diagnosis.sha256 if diagnosis else "",
            "join_keys": ["subject_id", "dti_image_id"],
        },
        "counts": counts,
        "checks": {
            "source_subject_sets_identical": True,
            "selected_image_ids_consistent_across_sources": True,
            "dti_exact_dates_cross_checked": True,
            "t1_path_dates_complete": True,
            "signed_and_absolute_intervals_recomputed": True,
            "timing_strata_exhaustive_and_disjoint": (
                counts["timing_le_90_days"]
                + counts["timing_91_180_days"]
                + counts["timing_gt_180_days"]
                == counts["pairs"]
            ),
            "local_dti_series_uniquely_resolved": True,
            "local_t1_series_uniquely_resolved": True,
            "local_file_inventory_complete": True,
            "long_gap_visit_matched_claims_absent": counts[
                "long_gap_visit_matched_claims"
            ]
            == 0,
            "corrected_qc_not_inferred_from_legacy_qc": counts[
                "corrected_rerun_qc_complete"
            ]
            == 0,
            "diagnosis_source_columns_preserved": True,
            "diagnosis_attachment_complete": (
                counts["diagnosis_reconciled_rows"] == counts["pairs"]
                if diagnosis is not None
                else False
            ),
        },
        "hash_definitions": {
            "pair_record_sha256": (
                "SHA-256 of canonical JSON for every emitted row except its own hash"
            ),
            "local_series_names_sizes_sha256": (
                "SHA-256 of sorted relative-path, NUL, byte-size, newline records; "
                "this is a metadata-inventory hash, not a file-content hash"
            ),
            "content_hash_status": "deferred_to_SL-P0-09",
        },
        "known_gaps": [
            (
                "Diagnosis reconciliation is not final until a 530-key, checksum-bound "
                "SL-P0-05 attachment is supplied."
                if diagnosis is None
                else "Diagnosis reconciliation attachment is checksum-bound but remains subject to the human policy gate."
            ),
            "Current T1 dates are path-derived; the source evidence reports the available cross-check coverage.",
            "Current T1 image QC is unavailable in the supplied sources and is not invented.",
            "Legacy matrix QC is retained as historical evidence only; corrected recipe QC has not run.",
            "Local series hashes cover names and sizes, not file contents; content hashes are deferred to SL-P0-09.",
            "Timing covariates/sensitivities cannot reconstruct contemporaneous anatomy for long-gap pairs.",
        ],
        "safety": {
            "inputs_read_only": True,
            "outputs_confined_to_requested_directory": True,
            "production_paths_modified": 0,
            "images_downloaded": 0,
            "images_deleted": 0,
            "images_processed": 0,
        },
        "outputs": {
            "pair_manifest": _output_evidence(
                targets["manifest"], manifest_bytes, len(records)
            ),
            "strata": _output_evidence(targets["strata"], strata_bytes, len(strata)),
            "flow": _output_evidence(targets["flow"], flow_bytes, len(flow)),
        },
    }
    validation_bytes = (
        json.dumps(validation, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    stage_root = Path(tempfile.mkdtemp(prefix="available-data-v2-", dir=output_dir))
    try:
        staged = {
            "manifest": stage_root / OUTPUT_MANIFEST,
            "strata": stage_root / OUTPUT_STRATA,
            "flow": stage_root / OUTPUT_FLOW,
            "validation": stage_root / OUTPUT_VALIDATION,
        }
        staged["manifest"].write_bytes(manifest_bytes)
        staged["strata"].write_bytes(strata_bytes)
        staged["flow"].write_bytes(flow_bytes)
        staged["validation"].write_bytes(validation_bytes)
        for name in ("manifest", "strata", "flow", "validation"):
            os.rename(staged[name], targets[name])
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--cohort-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--legacy-dryrun", type=Path, default=DEFAULT_DRYRUN)
    parser.add_argument("--dti-master", type=Path, default=DEFAULT_DTI_MASTER)
    parser.add_argument(
        "--subject-manifest", type=Path, default=DEFAULT_SUBJECT_MANIFEST
    )
    parser.add_argument("--audit-run-manifest", type=Path, default=DEFAULT_AUDIT_RUN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--expected-pairs", type=int, default=EXPECTED_PAIRS)
    parser.add_argument("--diagnosis-reconciliation", type=Path)
    parser.add_argument("--diagnosis-reconciliation-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_available_data_manifest(
        exact_path=args.exact,
        audit_path=args.cohort_audit,
        dryrun_path=args.legacy_dryrun,
        dti_master_path=args.dti_master,
        subject_manifest_path=args.subject_manifest,
        audit_run_path=args.audit_run_manifest,
        expected_pairs=args.expected_pairs,
        diagnosis_path=args.diagnosis_reconciliation,
        diagnosis_sha256=args.diagnosis_reconciliation_sha256,
    )
    validation = write_release(result, args.output_dir)
    counts = validation["counts"]
    print(
        f"PASS: {counts['pairs']} fixed local pairs; "
        f"<=90d={counts['timing_le_90_days']}, "
        f"91-180d={counts['timing_91_180_days']}, "
        f">180d={counts['timing_gt_180_days']}; "
        f"release={validation['release_status']}"
    )


if __name__ == "__main__":
    main()
