#!/usr/bin/env python3
"""Build the locked 530-pair source-metadata/workflow preflight.

This audit is intentionally non-imaging.  It reads the checksum-locked file
inventory, selected DICOM headers with ``stop_before_pixels=True``, and the 225
processed-T1 NIfTI headers.  It never loads a pixel/voxel array, never converts
an image, and never guesses BIDS phase-encoding direction or TotalReadoutTime.

The output is a workflow *projection*, not an executable staging manifest.
Fields that require dcm2niix/mrconvert conversion or conversion-time gradient
validation are explicitly ``PENDING_CONVERSION``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import pydicom
from pydicom.multival import MultiValue


PROJECT = Path("/home/ec2-user/exp")
AUDIT = PROJECT / "research_audit"
OUTPUT_ROOT = AUDIT / "outputs"
LOCK_DIR = OUTPUT_ROOT / "available_data_content_lock_v2"
DEFAULT_LOCK_VALIDATION = LOCK_DIR / "available_data_content_lock_validation_v2.json"
DEFAULT_LOCKED_MANIFEST = LOCK_DIR / "available_data_pair_manifest_locked_v2.csv"
DEFAULT_INVENTORY = LOCK_DIR / "available_data_file_content_inventory_v2.csv"
DEFAULT_FINAL_DIR = OUTPUT_ROOT / "source_metadata_workflow_preflight_v1"

OUTPUT_TABLE = "source_metadata_workflow_projection_v1.csv"
OUTPUT_VALIDATION = "source_metadata_workflow_projection_validation_v1.json"
OUTPUT_MEMO = "source_metadata_workflow_projection_memo_v1.md"
SCHEMA_VERSION = "1.0"
EXPECTED_PAIRS = 530
EXPECTED_FILES = 451_115
EXPECTED_DTI_FILES = 396_082
EXPECTED_T1_FILES = 55_033
EXPECTED_ORIGINAL_T1 = 305
EXPECTED_PROCESSED_T1 = 225
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UID_CHARS_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")

# Header fields only.  PixelData is never requested and dcmread stops before it.
DICOM_TAGS = (
    "SeriesInstanceUID",
    "StudyInstanceUID",
    "SOPInstanceUID",
    "PatientID",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "ProtocolName",
    "SeriesDescription",
    "ImageType",
    "InPlanePhaseEncodingDirection",
    "NumberOfPhaseEncodingSteps",
    "EchoTrainLength",
    "PixelBandwidth",
    "AcquisitionMatrix",
    "DiffusionBValue",
    "DiffusionGradientOrientation",
    "Rows",
    "Columns",
    "InstanceNumber",
    "AcquisitionNumber",
)

CONVERSION_REQUIREMENTS = (
    "convert locked raw DTI DICOM with a frozen dcm2niix/mrconvert version",
    "derive BIDS phase_encoding_direction from converted orientation and source metadata",
    "derive TotalReadoutTime from authoritative conversion metadata; never use a default",
    "export bvec/bval from the same conversion",
    "verify gradient count equals DWI volume count",
    "verify b-vector norms, b0 vectors, shells, and conversion-time rotation",
    "hash exact staged DWI, bvec, bval, JSON, and T1 inputs",
)


class SourceMutationError(RuntimeError):
    """Raised when a source no longer matches the locked file identity."""


@dataclass(frozen=True)
class InventoryFile:
    pair_id: str
    subject_id: str
    modality: str
    image_id: int
    series_dir: str
    relative_path: str
    absolute_path: str
    size_bytes: int
    sha256: str
    st_dev: int
    st_ino: int
    st_mtime_ns: int
    st_ctime_ns: int

    @property
    def identity(self) -> tuple[int, int, int, int, int]:
        return (
            self.st_dev,
            self.st_ino,
            self.size_bytes,
            self.st_mtime_ns,
            self.st_ctime_ns,
        )


@dataclass(frozen=True)
class Inputs:
    rows: list[dict[str, str]]
    groups: dict[tuple[str, str], list[InventoryFile]]
    evidence: dict[str, Any]
    content_validation: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_strict(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked {label}: {path}")
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None:
        raise ValueError(f"{label} has no header")
    columns = [str(value).strip() for value in reader.fieldnames]
    if any(not value for value in columns) or len(columns) != len(set(columns)):
        raise ValueError(f"{label} has blank or duplicate headers")
    rows: list[dict[str, str]] = []
    for line, raw in enumerate(reader, start=2):
        if None in raw:
            raise ValueError(f"{label} row {line} has extra fields")
        rows.append({key: "" if value is None else value.strip() for key, value in raw.items()})
    return columns, rows


def require_columns(columns: Iterable[str], required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def strict_int(value: Any, label: str, *, positive: bool = False) -> int:
    token = str(value).strip()
    pattern = r"[1-9]\d*" if positive else r"(?:0|[1-9]\d*)"
    if not re.fullmatch(pattern, token):
        raise ValueError(f"{label} is not a valid integer: {value!r}")
    return int(token)


def strict_bool(value: Any, label: str) -> bool:
    token = str(value).strip().lower()
    if token == "true":
        return True
    if token == "false":
        return False
    raise ValueError(f"{label} must be true/false, found {value!r}")


def _series_bundle(files: Sequence[InventoryFile]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value.relative_path):
        digest.update(item.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.size_bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_inputs(
    lock_validation_path: Path,
    locked_manifest_path: Path,
    inventory_path: Path,
) -> Inputs:
    for path, label in (
        (lock_validation_path, "content-lock validation"),
        (locked_manifest_path, "locked pair manifest"),
        (inventory_path, "locked file inventory"),
    ):
        if path.is_symlink():
            raise ValueError(f"Refusing symlinked {label}: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    validation = json.loads(lock_validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or validation.get("release_status") != "CONTENT_HASH_LOCKED":
        raise ValueError("Content-lock validation is not PASS/CONTENT_HASH_LOCKED")
    locked_evidence = validation.get("outputs", {}).get("locked_pair_manifest", {})
    inventory_evidence = validation.get("outputs", {}).get("file_inventory", {})
    if (
        sha256_file(locked_manifest_path) != locked_evidence.get("sha256")
        or locked_manifest_path.stat().st_size != locked_evidence.get("size_bytes")
        or locked_evidence.get("row_count") != EXPECTED_PAIRS
    ):
        raise ValueError("Locked pair manifest is not checksum/row-count bound")
    if (
        sha256_file(inventory_path) != inventory_evidence.get("sha256")
        or inventory_path.stat().st_size != inventory_evidence.get("size_bytes")
        or inventory_evidence.get("row_count") != EXPECTED_FILES
    ):
        raise ValueError("Locked inventory is not checksum/row-count bound")
    source_manifest_sha = validation.get("source", {}).get("canonical_pair_manifest", {}).get("sha256")
    universe_sha = validation.get("file_universe_sha256")
    if not SHA256_RE.fullmatch(str(source_manifest_sha)) or not SHA256_RE.fullmatch(str(universe_sha)):
        raise ValueError("Content-lock source/universe hashes are invalid")

    columns, rows = read_csv_strict(locked_manifest_path, "locked pair manifest")
    required = {
        "pair_id",
        "subject_id",
        "diagnosis_reconciled_harmonized",
        "diagnosis_reconciled_primary_analysis_eligible",
        "diagnosis_reconciled_smc_retained_separately",
        "dti_image_id",
        "current_t1_image_id",
        "dti_study_date",
        "current_t1_study_date",
        "dti_t1_gap_days_abs",
        "dti_phase",
        "site",
        "dti_manufacturer",
        "dti_field_strength_t",
        "dti_protocol_key",
        "current_t1_type",
        "local_dti_series_dir",
        "local_t1_series_dir",
        "local_dti_content_file_count",
        "local_dti_content_total_bytes",
        "local_dti_content_bundle_sha256",
        "local_t1_content_file_count",
        "local_t1_content_total_bytes",
        "local_t1_content_bundle_sha256",
        "pair_content_bundle_sha256",
        "locked_pair_record_sha256",
        "content_lock_source_manifest_sha256",
        "content_lock_file_universe_sha256",
        "locked_local_content_sha256_status",
    }
    require_columns(columns, required, "locked pair manifest")
    if len(rows) != EXPECTED_PAIRS:
        raise ValueError(f"Locked manifest has {len(rows)} rows, expected {EXPECTED_PAIRS}")
    pair_map = {row["pair_id"]: row for row in rows}
    if len(pair_map) != len(rows) or len({row["subject_id"] for row in rows}) != len(rows):
        raise ValueError("Locked manifest pair/subject keys are not unique")
    t1_types = Counter(row["current_t1_type"] for row in rows)
    if t1_types != {"Original": EXPECTED_ORIGINAL_T1, "Processed": EXPECTED_PROCESSED_T1}:
        raise ValueError(f"Unexpected T1 source classes: {dict(t1_types)}")
    for row in rows:
        if (
            row["content_lock_source_manifest_sha256"] != source_manifest_sha
            or row["content_lock_file_universe_sha256"] != universe_sha
            or row["locked_local_content_sha256_status"] != "COMPLETE_SHA256"
        ):
            raise ValueError(f"Locked row binding is invalid: {row['pair_id']}")

    groups: dict[tuple[str, str], list[InventoryFile]] = {}
    seen_paths: set[str] = set()
    inventory_columns = [
        "lock_schema_version",
        "source_pair_manifest_sha256",
        "pair_id",
        "subject_id",
        "modality",
        "image_id",
        "series_dir",
        "relative_path",
        "absolute_path",
        "size_bytes",
        "sha256",
        "st_dev",
        "st_ino",
        "st_mtime_ns",
        "st_ctime_ns",
    ]
    count = 0
    modality_counts: Counter[str] = Counter()
    with inventory_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != inventory_columns:
            raise ValueError("Locked inventory schema changed")
        for line, raw in enumerate(reader, start=2):
            count += 1
            pair_id = raw["pair_id"]
            source = pair_map.get(pair_id)
            if source is None:
                raise ValueError(f"Inventory row has unknown pair at line {line}")
            modality = raw["modality"]
            if modality not in {"DTI", "T1"}:
                raise ValueError(f"Inventory row has invalid modality at line {line}")
            expected_image = source["dti_image_id"] if modality == "DTI" else source["current_t1_image_id"]
            expected_dir = source["local_dti_series_dir"] if modality == "DTI" else source["local_t1_series_dir"]
            relative = raw["relative_path"]
            pure = PurePosixPath(relative)
            absolute = raw["absolute_path"]
            if pure.is_absolute() or ".." in pure.parts or str(Path(expected_dir) / relative) != absolute:
                raise ValueError(f"Inventory path escapes selected series at line {line}")
            if absolute in seen_paths:
                raise ValueError(f"Duplicate inventory path: {absolute}")
            seen_paths.add(absolute)
            if (
                raw["lock_schema_version"] != "2.0"
                or raw["source_pair_manifest_sha256"] != source_manifest_sha
                or raw["subject_id"] != source["subject_id"]
                or strict_int(raw["image_id"], f"inventory image ID line {line}", positive=True)
                != strict_int(expected_image, f"manifest image ID {pair_id}", positive=True)
                or raw["series_dir"] != expected_dir
                or not SHA256_RE.fullmatch(raw["sha256"])
            ):
                raise ValueError(f"Inventory binding mismatch at line {line}")
            item = InventoryFile(
                pair_id=pair_id,
                subject_id=raw["subject_id"],
                modality=modality,
                image_id=int(raw["image_id"]),
                series_dir=raw["series_dir"],
                relative_path=relative,
                absolute_path=absolute,
                size_bytes=strict_int(raw["size_bytes"], f"inventory size line {line}"),
                sha256=raw["sha256"],
                st_dev=strict_int(raw["st_dev"], f"inventory device line {line}"),
                st_ino=strict_int(raw["st_ino"], f"inventory inode line {line}"),
                st_mtime_ns=strict_int(raw["st_mtime_ns"], f"inventory mtime line {line}"),
                st_ctime_ns=strict_int(raw["st_ctime_ns"], f"inventory ctime line {line}"),
            )
            groups.setdefault((pair_id, modality), []).append(item)
            modality_counts[modality] += 1
    if count != EXPECTED_FILES or len(seen_paths) != EXPECTED_FILES:
        raise ValueError(f"Inventory coverage changed: {count}/{len(seen_paths)}")
    if modality_counts != {"DTI": EXPECTED_DTI_FILES, "T1": EXPECTED_T1_FILES}:
        raise ValueError(f"Inventory modality counts changed: {dict(modality_counts)}")
    if len(groups) != 2 * EXPECTED_PAIRS:
        raise ValueError(f"Inventory series count changed: {len(groups)}")
    for row in rows:
        for modality, count_field, bytes_field, bundle_field in (
            (
                "DTI",
                "local_dti_content_file_count",
                "local_dti_content_total_bytes",
                "local_dti_content_bundle_sha256",
            ),
            (
                "T1",
                "local_t1_content_file_count",
                "local_t1_content_total_bytes",
                "local_t1_content_bundle_sha256",
            ),
        ):
            files = groups.get((row["pair_id"], modality), [])
            expected_count = strict_int(row[count_field], f"{row['pair_id']} {modality} count")
            expected_bytes = strict_int(row[bytes_field], f"{row['pair_id']} {modality} bytes")
            if (
                len(files) != expected_count
                or sum(item.size_bytes for item in files) != expected_bytes
                or _series_bundle(files) != row[bundle_field]
            ):
                raise ValueError(f"Locked series bundle mismatch: {row['pair_id']} {modality}")
            files.sort(key=lambda value: value.relative_path)

    evidence = {
        "content_lock_validation": {
            "path": str(lock_validation_path.resolve()),
            "sha256": sha256_file(lock_validation_path),
            "size_bytes": lock_validation_path.stat().st_size,
        },
        "locked_pair_manifest": {
            "path": str(locked_manifest_path.resolve()),
            "sha256": locked_evidence["sha256"],
            "size_bytes": locked_evidence["size_bytes"],
            "rows": len(rows),
        },
        "locked_file_inventory": {
            "path": str(inventory_path.resolve()),
            "sha256": inventory_evidence["sha256"],
            "size_bytes": inventory_evidence["size_bytes"],
            "rows": count,
        },
        "canonical_pair_manifest_sha256": source_manifest_sha,
        "file_universe_sha256": universe_sha,
    }
    return Inputs(rows=rows, groups=groups, evidence=evidence, content_validation=validation)


def _identity_matches(metadata: os.stat_result, item: InventoryFile) -> bool:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    ) == item.identity


def _open_locked(item: InventoryFile) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(item.absolute_path, flags)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or not _identity_matches(metadata, item):
        os.close(descriptor)
        raise SourceMutationError(f"Locked source identity changed: {item.absolute_path}")
    return descriptor, metadata


def _postcheck(item: InventoryFile) -> None:
    metadata = os.lstat(item.absolute_path)
    if not stat.S_ISREG(metadata.st_mode) or not _identity_matches(metadata, item):
        raise SourceMutationError(f"Locked source changed after header read: {item.absolute_path}")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, MultiValue)):
        return json.dumps([_text(item) for item in value], separators=(",", ":"))
    return " ".join(str(value).replace("\x00", "").split())


def _number_text(value: Any) -> str:
    token = _text(value)
    if not token:
        return ""
    try:
        number = float(token)
    except ValueError:
        return token
    if not math.isfinite(number):
        return token
    return f"{number:.12g}"


def dicom_uid_syntax(value: str) -> str:
    """Validate DICOM UI syntax without normalizing the authoritative value."""
    if not value:
        return "MISSING"
    issues: list[str] = []
    if len(value) > 64:
        issues.append("LENGTH_GT_64")
    if not UID_CHARS_RE.fullmatch(value):
        issues.append("NON_NUMERIC_OR_EMPTY_COMPONENT")
    else:
        if any(len(component) > 1 and component.startswith("0") for component in value.split(".")):
            issues.append("LEADING_ZERO_COMPONENT")
    return "VALID" if not issues else "NONCONFORMANT_" + "+".join(issues)


def stable_source_id(kind: str, image_id: str | int, series_bundle_sha256: str) -> str:
    if not SHA256_RE.fullmatch(series_bundle_sha256):
        raise ValueError("Stable source ID requires a valid series bundle SHA-256")
    image = strict_int(image_id, f"{kind} image ID", positive=True)
    payload = (
        f"{kind}\0ADNI_IMAGE_ID\0{image}\0"
        f"LOCKED_SERIES_CONTENT_BUNDLE_SHA256\0{series_bundle_sha256}\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _read_dicom(item: InventoryFile) -> pydicom.dataset.Dataset:
    descriptor, _ = _open_locked(item)
    try:
        stream = os.fdopen(descriptor, "rb", closefd=True)
    except Exception:
        os.close(descriptor)
        raise
    # The FileIO object now owns the descriptor.  Do not call os.close again:
    # in a threaded scan the descriptor number could be reused by another
    # worker between FileIO.close and a redundant close, corrupting that
    # worker's stat/read identity check.
    with stream:
        dataset = pydicom.dcmread(
            stream,
            stop_before_pixels=True,
            specific_tags=DICOM_TAGS,
            force=False,
        )
        after = os.fstat(stream.fileno())
        if not _identity_matches(after, item):
            raise SourceMutationError(f"Source changed during DICOM header read: {item.absolute_path}")
    _postcheck(item)
    if "PixelData" in dataset:
        raise RuntimeError(f"PixelData was unexpectedly loaded: {item.absolute_path}")
    return dataset


def summarize_dicom_series(
    files: Sequence[InventoryFile], expected_subject: str
) -> dict[str, Any]:
    if not files:
        raise ValueError("Cannot summarize an empty DICOM series")
    values: dict[str, Counter[str]] = {
        "series_uid": Counter(),
        "study_uid": Counter(),
        "sop_uid": Counter(),
        "patient_id": Counter(),
        "manufacturer": Counter(),
        "model": Counter(),
        "field_strength": Counter(),
        "in_plane_pe": Counter(),
        "phase_steps": Counter(),
        "echo_train": Counter(),
        "pixel_bandwidth": Counter(),
        "acquisition_matrix": Counter(),
        "public_b_value": Counter(),
        "public_gradient": Counter(),
    }
    errors: list[str] = []
    headers_read = 0
    for item in files:
        if not item.relative_path.lower().endswith(".dcm"):
            errors.append(f"NOT_DICOM:{item.relative_path}")
            continue
        try:
            dataset = _read_dicom(item)
        except SourceMutationError:
            raise
        except Exception as exc:  # row-level authoritative parsing failure
            errors.append(f"DICOM_READ:{item.relative_path}:{type(exc).__name__}")
            continue
        headers_read += 1
        mapping = {
            "series_uid": _text(getattr(dataset, "SeriesInstanceUID", None)),
            "study_uid": _text(getattr(dataset, "StudyInstanceUID", None)),
            "sop_uid": _text(getattr(dataset, "SOPInstanceUID", None)),
            "patient_id": _text(getattr(dataset, "PatientID", None)),
            "manufacturer": _text(getattr(dataset, "Manufacturer", None)),
            "model": _text(getattr(dataset, "ManufacturerModelName", None)),
            "field_strength": _number_text(getattr(dataset, "MagneticFieldStrength", None)),
            "in_plane_pe": _text(getattr(dataset, "InPlanePhaseEncodingDirection", None)),
            "phase_steps": _number_text(getattr(dataset, "NumberOfPhaseEncodingSteps", None)),
            "echo_train": _number_text(getattr(dataset, "EchoTrainLength", None)),
            "pixel_bandwidth": _number_text(getattr(dataset, "PixelBandwidth", None)),
            "acquisition_matrix": _text(getattr(dataset, "AcquisitionMatrix", None)),
            "public_b_value": _number_text(getattr(dataset, "DiffusionBValue", None)),
            "public_gradient": _text(getattr(dataset, "DiffusionGradientOrientation", None)),
        }
        for key, value in mapping.items():
            values[key][value] += 1

    total = len(files)

    def present(counter: Counter[str]) -> dict[str, int]:
        return {key: count for key, count in counter.items() if key}

    def one(counter: Counter[str]) -> str:
        active = sorted(present(counter))
        return active[0] if len(active) == 1 else ""

    def consistency(counter: Counter[str], label: str) -> str:
        active = present(counter)
        missing = counter.get("", 0)
        if missing:
            return f"FAIL_{label}_MISSING"
        if len(active) != 1:
            return f"FAIL_{label}_CONFLICT"
        return "PASS"

    uid = one(values["series_uid"])
    uid_consistency = consistency(values["series_uid"], "SERIES_UID")
    model_consistency = consistency(values["model"], "SCANNER_MODEL")
    manufacturer_consistency = consistency(values["manufacturer"], "MANUFACTURER")
    field_consistency = consistency(values["field_strength"], "FIELD_STRENGTH")
    patient_active = present(values["patient_id"])
    patient_status = (
        "PASS"
        if values["patient_id"].get("", 0) == 0
        and set(patient_active) == {expected_subject}
        else "FAIL_PATIENT_ID_MISMATCH_OR_MISSING"
    )
    study_status = consistency(values["study_uid"], "STUDY_UID")
    sop_present = present(values["sop_uid"])
    sop_missing = values["sop_uid"].get("", 0)
    sop_duplicate = sum(count - 1 for count in sop_present.values() if count > 1)
    sop_status = "PASS" if not sop_missing and not sop_duplicate else "FAIL_SOP_UID_MISSING_OR_DUPLICATE"
    identity_failures = [
        status
        for status in (
            uid_consistency,
            model_consistency,
            manufacturer_consistency,
            field_consistency,
            patient_status,
            study_status,
            sop_status,
        )
        if status != "PASS"
    ]
    if errors:
        identity_failures.append("FAIL_DICOM_HEADER_READ")
    syntax = dicom_uid_syntax(uid) if uid else "NOT_EVALUABLE"
    warnings = [] if syntax == "VALID" else [syntax]
    identity_ready = not identity_failures and headers_read == total

    def joined(counter: Counter[str]) -> str:
        return "|".join(sorted(present(counter)))

    return {
        "inventory_file_count": total,
        "headers_read": headers_read,
        "read_error_count": len(errors),
        "read_errors_json": json.dumps(errors[:10], separators=(",", ":")),
        "series_uid": uid,
        "series_uid_distinct": len(present(values["series_uid"])),
        "series_uid_missing": values["series_uid"].get("", 0),
        "series_uid_consistency_status": uid_consistency,
        "series_uid_syntax_status": syntax,
        "study_uid": one(values["study_uid"]),
        "study_uid_distinct": len(present(values["study_uid"])),
        "study_uid_missing": values["study_uid"].get("", 0),
        "study_uid_status": study_status,
        "sop_uid_missing": sop_missing,
        "sop_uid_duplicate": sop_duplicate,
        "sop_uid_status": sop_status,
        "patient_id_values": joined(values["patient_id"]),
        "patient_id_status": patient_status,
        "manufacturer": one(values["manufacturer"]),
        "manufacturer_values": joined(values["manufacturer"]),
        "manufacturer_status": manufacturer_consistency,
        "scanner_model": one(values["model"]),
        "scanner_model_values": joined(values["model"]),
        "scanner_model_status": model_consistency,
        "field_strength_t": one(values["field_strength"]),
        "field_strength_values": joined(values["field_strength"]),
        "field_strength_status": field_consistency,
        "in_plane_pe_values": joined(values["in_plane_pe"]),
        "in_plane_pe_headers_present": sum(present(values["in_plane_pe"]).values()),
        "in_plane_pe_status": consistency(values["in_plane_pe"], "IN_PLANE_PE"),
        "phase_steps_values": joined(values["phase_steps"]),
        "echo_train_values": joined(values["echo_train"]),
        "pixel_bandwidth_values": joined(values["pixel_bandwidth"]),
        "acquisition_matrix_values": joined(values["acquisition_matrix"]),
        "public_b_value_headers_present": sum(present(values["public_b_value"]).values()),
        "public_b_values": joined(values["public_b_value"]),
        "public_gradient_headers_present": sum(present(values["public_gradient"]).values()),
        "identity_ready": identity_ready,
        "identity_failures": identity_failures,
        "identity_warnings": warnings,
    }


def inspect_nifti_header(files: Sequence[InventoryFile]) -> dict[str, Any]:
    if len(files) != 1 or not files[0].relative_path.lower().endswith((".nii", ".nii.gz")):
        return {
            "status": "FAIL_EXPECTED_ONE_NIFTI",
            "path": "",
            "sha256": "",
            "shape_json": "",
            "zooms_json": "",
            "dtype": "",
            "qform_code": "",
            "sform_code": "",
            "error": f"files={len(files)}",
        }
    item = files[0]
    descriptor, _ = _open_locked(item)
    try:
        stream = os.fdopen(descriptor, "rb", closefd=True)
    except Exception:
        os.close(descriptor)
        raise
    with stream:
        first = stream.read(4)
        stream.seek(0)
        little = int.from_bytes(first, "little", signed=True)
        big = int.from_bytes(first, "big", signed=True)
        if 348 in {little, big}:
            header = nib.Nifti1Header.from_fileobj(stream, check=True)
        elif 540 in {little, big}:
            header = nib.Nifti2Header.from_fileobj(stream, check=True)
        else:
            raise ValueError(f"unsupported NIfTI sizeof_hdr: {little}/{big}")
        after = os.fstat(stream.fileno())
        if not _identity_matches(after, item):
            raise SourceMutationError(f"Source changed during NIfTI header read: {item.absolute_path}")
    _postcheck(item)
    return {
        "status": "PASS_HEADER_ONLY",
        "path": item.absolute_path,
        "sha256": item.sha256,
        "shape_json": json.dumps(list(header.get_data_shape()), separators=(",", ":")),
        "zooms_json": json.dumps([float(value) for value in header.get_zooms()], separators=(",", ":")),
        "dtype": str(header.get_data_dtype()),
        "qform_code": str(int(header["qform_code"])),
        "sform_code": str(int(header["sform_code"])),
        "error": "",
    }


def _norm_text(value: str) -> str:
    return " ".join(value.upper().split())


def _float_match(left: str, right: str, tolerance: float = 0.01) -> bool:
    try:
        return math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0)
    except (TypeError, ValueError):
        return False


OUTPUT_COLUMNS = [
    "projection_schema_version",
    "content_lock_validation_sha256",
    "locked_pair_manifest_sha256",
    "locked_inventory_sha256",
    "content_lock_source_manifest_sha256",
    "content_lock_file_universe_sha256",
    "pair_id",
    "subject_id",
    "diagnosis_at_dti",
    "primary_analysis_eligible",
    "smc_retained_separately",
    "pair_processing_approved",
    "dti_image_id",
    "t1_image_id",
    "dti_study_date",
    "t1_study_date",
    "abs_pair_gap_days",
    "phase",
    "site",
    "protocol",
    "dti_source_class",
    "t1_source_class",
    "dti_source_bundle_sha256",
    "t1_source_bundle_sha256",
    "pair_source_bundle_sha256",
    "source_id_scheme",
    "dti_source_id",
    "t1_source_id",
    "dti_series_dir",
    "dti_inventory_file_count",
    "dti_inventory_total_bytes",
    "dti_dicom_headers_read",
    "dti_dicom_read_error_count",
    "dti_series_instance_uid",
    "dti_series_uid_distinct_count",
    "dti_series_uid_missing_headers",
    "dti_series_uid_consistency_status",
    "dti_series_uid_syntax_status",
    "dti_study_instance_uid",
    "dti_study_uid_status",
    "dti_sop_uid_missing_headers",
    "dti_sop_uid_duplicate_count",
    "dti_patient_id_status",
    "dti_manufacturer_header",
    "dti_manufacturer_status",
    "dti_manufacturer_manifest",
    "dti_manufacturer_manifest_match",
    "scanner_model",
    "scanner_model_values",
    "scanner_model_consistency_status",
    "dti_field_strength_header_t",
    "dti_field_strength_status",
    "dti_field_strength_manifest_t",
    "dti_field_strength_manifest_match",
    "dti_dicom_in_plane_phase_encoding_values",
    "dti_dicom_in_plane_phase_encoding_headers_present",
    "dti_dicom_in_plane_phase_encoding_status",
    "dti_dicom_phase_encoding_steps_values",
    "dti_dicom_echo_train_length_values",
    "dti_dicom_pixel_bandwidth_values",
    "dti_dicom_acquisition_matrix_values",
    "dti_public_b_value_headers_present",
    "dti_public_b_values",
    "dti_public_gradient_headers_present",
    "dti_source_identity_status",
    "t1_series_dir",
    "t1_inventory_file_count",
    "t1_inventory_total_bytes",
    "t1_dicom_headers_read",
    "t1_dicom_read_error_count",
    "t1_dicom_series_uid",
    "t1_dicom_series_uid_distinct_count",
    "t1_dicom_series_uid_missing_headers",
    "t1_dicom_series_uid_consistency_status",
    "t1_dicom_series_uid_syntax_status",
    "t1_source_id_status",
    "t1_nifti_source_path",
    "t1_nifti_source_sha256",
    "t1_nifti_header_status",
    "t1_nifti_shape_json",
    "t1_nifti_zooms_json",
    "t1_nifti_dtype",
    "t1_nifti_qform_code",
    "t1_nifti_sform_code",
    "t1_source_identity_status",
    "dwi_nifti",
    "dwi_nifti_status",
    "dwi_bvec",
    "dwi_bvec_status",
    "dwi_bval",
    "dwi_bval_status",
    "dwi_json",
    "dwi_json_status",
    "t1_staged_nifti",
    "t1_staged_nifti_status",
    "phase_encoding_direction",
    "phase_encoding_direction_status",
    "total_readout_time",
    "total_readout_time_status",
    "gradient_table_status",
    "gradient_count_vs_volume_status",
    "gradient_norm_and_rotation_status",
    "conversion_requirements_json",
    "workflow_projection_status",
    "canary_ready",
    "blocking_reasons_json",
    "warnings_json",
]


def build_projection_row(
    source: Mapping[str, str],
    dti: Mapping[str, Any],
    t1_dicom: Mapping[str, Any] | None,
    t1_nifti: Mapping[str, Any] | None,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    pair_id = source["pair_id"]
    t1_type = source["current_t1_type"]
    dti_bundle = source["local_dti_content_bundle_sha256"]
    t1_bundle = source["local_t1_content_bundle_sha256"]
    dti_id = stable_source_id("DTI", source["dti_image_id"], dti_bundle)
    t1_id = stable_source_id("T1", source["current_t1_image_id"], t1_bundle)
    manufacturer_match = (
        dti["manufacturer_status"] == "PASS"
        and _norm_text(dti["manufacturer"]) == _norm_text(source["dti_manufacturer"])
    )
    field_match = (
        dti["field_strength_status"] == "PASS"
        and _float_match(dti["field_strength_t"], source["dti_field_strength_t"])
    )
    dti_ready = bool(dti["identity_ready"] and manufacturer_match and field_match)
    warnings = list(dti["identity_warnings"])
    if not manufacturer_match:
        warnings.append("DTI_MANUFACTURER_MANIFEST_MISMATCH")
    if not field_match:
        warnings.append("DTI_FIELD_STRENGTH_MANIFEST_MISMATCH")

    if t1_type == "Original":
        assert t1_dicom is not None and t1_nifti is None
        t1_ready = bool(t1_dicom["identity_ready"])
        t1_uid = t1_dicom["series_uid"]
        t1_uid_distinct = t1_dicom["series_uid_distinct"]
        t1_uid_missing = t1_dicom["series_uid_missing"]
        t1_uid_consistency = t1_dicom["series_uid_consistency_status"]
        t1_uid_syntax = t1_dicom["series_uid_syntax_status"]
        t1_headers = t1_dicom["headers_read"]
        t1_read_errors = t1_dicom["read_error_count"]
        warnings.extend(t1_dicom["identity_warnings"])
        nifti_values = {
            "path": "",
            "sha256": "",
            "status": "NOT_APPLICABLE_DICOM_SOURCE",
            "shape_json": "",
            "zooms_json": "",
            "dtype": "",
            "qform_code": "",
            "sform_code": "",
        }
        t1_staged_status = "PENDING_CONVERSION"
        t1_source_class = "ORIGINAL_DICOM"
    else:
        assert t1_dicom is None and t1_nifti is not None
        t1_ready = t1_nifti["status"] == "PASS_HEADER_ONLY"
        t1_uid = ""
        t1_uid_distinct = 0
        t1_uid_missing = 0
        t1_uid_consistency = "NOT_APPLICABLE_PROCESSED_NIFTI"
        t1_uid_syntax = "NOT_APPLICABLE_PROCESSED_NIFTI"
        t1_headers = 0
        t1_read_errors = 0
        nifti_values = dict(t1_nifti)
        t1_staged_status = "LOCKED_SOURCE_NIFTI_AVAILABLE_PENDING_STAGING_PROJECTION"
        t1_source_class = "PROCESSED_NIFTI_DERIVED"

    identity_failures: list[str] = []
    if not dti_ready:
        identity_failures.extend(["DTI_SOURCE_IDENTITY"] + list(dti["identity_failures"]))
    if not t1_ready:
        identity_failures.append("T1_SOURCE_IDENTITY")
        if t1_dicom is not None:
            identity_failures.extend(t1_dicom["identity_failures"])
        elif t1_nifti is not None:
            identity_failures.append(t1_nifti.get("status", "T1_NIFTI_UNKNOWN"))
    pending = [
        "PENDING_DWI_CONVERSION",
        "PENDING_BIDS_PHASE_ENCODING_DIRECTION",
        "PENDING_TOTAL_READOUT_TIME",
        "PENDING_GRADIENT_EXPORT_AND_VALIDATION",
        "PENDING_WRITE_ONCE_STAGING_HASHES",
    ]
    if t1_type == "Original":
        pending.append("PENDING_T1_CONVERSION")
    blockers = identity_failures + pending
    workflow_status = (
        "SOURCE_IDENTITY_READY_CONVERSION_PENDING"
        if not identity_failures
        else "BLOCKED_SOURCE_IDENTITY"
    )
    dti_identity_status = (
        "PASS_WITH_UID_SYNTAX_WARNING"
        if dti_ready and dti["series_uid_syntax_status"] != "VALID"
        else ("PASS" if dti_ready else "FAIL")
    )
    if t1_ready and t1_type == "Original" and t1_uid_syntax != "VALID":
        t1_identity_status = "PASS_WITH_UID_SYNTAX_WARNING"
    else:
        t1_identity_status = "PASS" if t1_ready else "FAIL"

    row: dict[str, Any] = {
        "projection_schema_version": SCHEMA_VERSION,
        "content_lock_validation_sha256": evidence["content_lock_validation"]["sha256"],
        "locked_pair_manifest_sha256": evidence["locked_pair_manifest"]["sha256"],
        "locked_inventory_sha256": evidence["locked_file_inventory"]["sha256"],
        "content_lock_source_manifest_sha256": evidence["canonical_pair_manifest_sha256"],
        "content_lock_file_universe_sha256": evidence["file_universe_sha256"],
        "pair_id": pair_id,
        "subject_id": source["subject_id"],
        "diagnosis_at_dti": source["diagnosis_reconciled_harmonized"],
        "primary_analysis_eligible": strict_bool(
            source["diagnosis_reconciled_primary_analysis_eligible"],
            f"{pair_id} primary eligibility",
        ),
        "smc_retained_separately": strict_bool(
            source["diagnosis_reconciled_smc_retained_separately"],
            f"{pair_id} SMC flag",
        ),
        "pair_processing_approved": True,
        "dti_image_id": int(source["dti_image_id"]),
        "t1_image_id": int(source["current_t1_image_id"]),
        "dti_study_date": source["dti_study_date"],
        "t1_study_date": source["current_t1_study_date"],
        "abs_pair_gap_days": int(source["dti_t1_gap_days_abs"]),
        "phase": source["dti_phase"],
        "site": source["site"],
        "protocol": source["dti_protocol_key"],
        "dti_source_class": "ORIGINAL_DICOM",
        "t1_source_class": t1_source_class,
        "dti_source_bundle_sha256": dti_bundle,
        "t1_source_bundle_sha256": t1_bundle,
        "pair_source_bundle_sha256": source["pair_content_bundle_sha256"],
        "source_id_scheme": "SHA256_KIND_ADNI_IMAGE_ID_LOCKED_SERIES_BUNDLE_V1",
        "dti_source_id": dti_id,
        "t1_source_id": t1_id,
        "dti_series_dir": source["local_dti_series_dir"],
        "dti_inventory_file_count": int(source["local_dti_content_file_count"]),
        "dti_inventory_total_bytes": int(source["local_dti_content_total_bytes"]),
        "dti_dicom_headers_read": dti["headers_read"],
        "dti_dicom_read_error_count": dti["read_error_count"],
        "dti_series_instance_uid": dti["series_uid"],
        "dti_series_uid_distinct_count": dti["series_uid_distinct"],
        "dti_series_uid_missing_headers": dti["series_uid_missing"],
        "dti_series_uid_consistency_status": dti["series_uid_consistency_status"],
        "dti_series_uid_syntax_status": dti["series_uid_syntax_status"],
        "dti_study_instance_uid": dti["study_uid"],
        "dti_study_uid_status": dti["study_uid_status"],
        "dti_sop_uid_missing_headers": dti["sop_uid_missing"],
        "dti_sop_uid_duplicate_count": dti["sop_uid_duplicate"],
        "dti_patient_id_status": dti["patient_id_status"],
        "dti_manufacturer_header": dti["manufacturer"],
        "dti_manufacturer_status": dti["manufacturer_status"],
        "dti_manufacturer_manifest": source["dti_manufacturer"],
        "dti_manufacturer_manifest_match": manufacturer_match,
        "scanner_model": dti["scanner_model"],
        "scanner_model_values": dti["scanner_model_values"],
        "scanner_model_consistency_status": dti["scanner_model_status"],
        "dti_field_strength_header_t": dti["field_strength_t"],
        "dti_field_strength_status": dti["field_strength_status"],
        "dti_field_strength_manifest_t": source["dti_field_strength_t"],
        "dti_field_strength_manifest_match": field_match,
        "dti_dicom_in_plane_phase_encoding_values": dti["in_plane_pe_values"],
        "dti_dicom_in_plane_phase_encoding_headers_present": dti["in_plane_pe_headers_present"],
        "dti_dicom_in_plane_phase_encoding_status": dti["in_plane_pe_status"],
        "dti_dicom_phase_encoding_steps_values": dti["phase_steps_values"],
        "dti_dicom_echo_train_length_values": dti["echo_train_values"],
        "dti_dicom_pixel_bandwidth_values": dti["pixel_bandwidth_values"],
        "dti_dicom_acquisition_matrix_values": dti["acquisition_matrix_values"],
        "dti_public_b_value_headers_present": dti["public_b_value_headers_present"],
        "dti_public_b_values": dti["public_b_values"],
        "dti_public_gradient_headers_present": dti["public_gradient_headers_present"],
        "dti_source_identity_status": dti_identity_status,
        "t1_series_dir": source["local_t1_series_dir"],
        "t1_inventory_file_count": int(source["local_t1_content_file_count"]),
        "t1_inventory_total_bytes": int(source["local_t1_content_total_bytes"]),
        "t1_dicom_headers_read": t1_headers,
        "t1_dicom_read_error_count": t1_read_errors,
        "t1_dicom_series_uid": t1_uid,
        "t1_dicom_series_uid_distinct_count": t1_uid_distinct,
        "t1_dicom_series_uid_missing_headers": t1_uid_missing,
        "t1_dicom_series_uid_consistency_status": t1_uid_consistency,
        "t1_dicom_series_uid_syntax_status": t1_uid_syntax,
        "t1_source_id_status": "PASS_STABLE_FROM_IMAGE_ID_AND_LOCKED_BUNDLE",
        "t1_nifti_source_path": nifti_values["path"],
        "t1_nifti_source_sha256": nifti_values["sha256"],
        "t1_nifti_header_status": nifti_values["status"],
        "t1_nifti_shape_json": nifti_values["shape_json"],
        "t1_nifti_zooms_json": nifti_values["zooms_json"],
        "t1_nifti_dtype": nifti_values["dtype"],
        "t1_nifti_qform_code": nifti_values["qform_code"],
        "t1_nifti_sform_code": nifti_values["sform_code"],
        "t1_source_identity_status": t1_identity_status,
        "dwi_nifti": "",
        "dwi_nifti_status": "PENDING_CONVERSION",
        "dwi_bvec": "",
        "dwi_bvec_status": "PENDING_CONVERSION",
        "dwi_bval": "",
        "dwi_bval_status": "PENDING_CONVERSION",
        "dwi_json": "",
        "dwi_json_status": "PENDING_CONVERSION",
        "t1_staged_nifti": "",
        "t1_staged_nifti_status": t1_staged_status,
        "phase_encoding_direction": "",
        "phase_encoding_direction_status": "PENDING_CONVERSION",
        "total_readout_time": "",
        "total_readout_time_status": "PENDING_CONVERSION",
        "gradient_table_status": "PENDING_CONVERSION",
        "gradient_count_vs_volume_status": "PENDING_CONVERSION",
        "gradient_norm_and_rotation_status": "PENDING_CONVERSION",
        "conversion_requirements_json": json.dumps(CONVERSION_REQUIREMENTS, separators=(",", ":")),
        "workflow_projection_status": workflow_status,
        "canary_ready": False,
        "blocking_reasons_json": json.dumps(sorted(set(blockers)), separators=(",", ":")),
        "warnings_json": json.dumps(sorted(set(warnings)), separators=(",", ":")),
    }
    if set(row) != set(OUTPUT_COLUMNS):
        raise RuntimeError(f"Projection row schema drift for {pair_id}")
    return row


def _csv_value(value: Any) -> Any:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return ""
    return value


def build_memo(generated: str, rows: Sequence[Mapping[str, Any]], counts: Mapping[str, Any], evidence: Mapping[str, Any]) -> str:
    models = Counter(str(row["scanner_model"]) for row in rows)
    statuses = Counter(str(row["workflow_projection_status"]) for row in rows)
    lines = [
        "# Locked source-metadata and workflow-projection preflight v1",
        "",
        f"**Generated:** {generated}",
        "**Scope:** exact locked source headers only; no pixel/voxel arrays, connectome values, conversions, preprocessing, or biological outcomes were read.",
        "**Release role:** source identity and conversion-readiness preflight. It is not an executable staging manifest and does not authorize a canary.",
        "",
        "## Result",
        "",
        f"- Pair rows: {counts['pairs']} (CN/MCI/AD primary ceiling {counts['primary_eligible']}; SMC separate {counts['smc']}).",
        f"- DTI DICOM headers read: {counts['dti_dicom_headers_read']:,}; Original-T1 DICOM headers read: {counts['t1_dicom_headers_read']:,}; processed-T1 NIfTI headers read: {counts['processed_t1_nifti_headers_read']}.",
        f"- DTI source-identity ready: {counts['dti_identity_ready']}/{counts['pairs']}; Original-T1 DICOM identity ready: {counts['original_t1_identity_ready']}/{counts['original_t1']}; processed-T1 locked NIfTI header ready: {counts['processed_t1_identity_ready']}/{counts['processed_t1']}.",
        f"- Standards warning: {counts['dti_uid_syntax_nonconformant']} DTI and {counts['t1_uid_syntax_nonconformant']} Original-T1 SeriesInstanceUID values are internally consistent but fail strict DICOM UI syntax (principally leading-zero components). The exact authoritative header strings are retained; they are not rewritten or fabricated.",
        f"- Processed T1 handling: {counts['processed_t1']} rows have a nullable DICOM UID and a mandatory stable `t1_source_id` derived from ADNI Image ID plus the locked T1 series-content bundle.",
        f"- Workflow projection: `{json.dumps(dict(statuses), sort_keys=True)}`; canary-ready rows: {counts['canary_ready']}.",
        "",
        "## Scanner and acquisition metadata",
        "",
        f"All-file scanner-model consistency passed in {counts['scanner_model_consistent']}/{counts['pairs']} DTI series. Model distribution: `{json.dumps(dict(sorted(models.items())), sort_keys=True)}`.",
        f"Manifest/header manufacturer agreement passed in {counts['manufacturer_manifest_match']}/{counts['pairs']}; field-strength agreement passed in {counts['field_strength_manifest_match']}/{counts['pairs']}.",
        f"The public DICOM `InPlanePhaseEncodingDirection` field is complete and within-series consistent in {counts['in_plane_pe_complete_consistent']}/{counts['pairs']} DTI series. This ROW/COL field is retained as evidence but is not a BIDS axis/sign and is never copied into `phase_encoding_direction`.",
        f"Public diffusion b-value tags occur in at least one header for {counts['series_with_public_b_value']} DTI series; public gradient-orientation tags occur for {counts['series_with_public_gradient']} series. Partial public tags do not replace conversion-time gradient export and validation.",
        "",
        "## Mandatory conversion-time work",
        "",
        "Every row remains `PENDING_CONVERSION` for DWI NIfTI/JSON, BIDS phase-encoding direction, TotalReadoutTime, bvec/bval export, gradient count/norm/rotation checks, and exact staged-file hashes. TotalReadoutTime and PE direction are blank by design: no value was guessed from protocol labels, historical derivatives, or incomplete public tags.",
        "",
        "Original T1 DICOM requires locked conversion. The 225 processed-T1 sources already have a checksum-locked NIfTI input, but still require an explicit staging projection and exact staged-path binding. The workflow contract must accept nullable `t1_dicom_series_uid` and use the stable `t1_source_id` for all T1 inputs.",
        "",
        "## Provenance and safety",
        "",
        f"- Content-lock validation SHA-256: `{evidence['content_lock_validation']['sha256']}`.",
        f"- Locked pair manifest SHA-256: `{evidence['locked_pair_manifest']['sha256']}`.",
        f"- Locked file inventory SHA-256: `{evidence['locked_file_inventory']['sha256']}`.",
        f"- Locked file universe SHA-256: `{evidence['file_universe_sha256']}`.",
        "- DICOM access used pydicom `stop_before_pixels=True` with an explicit tag allow-list. NIfTI access instantiated header objects only. Pixel/voxel arrays loaded: 0.",
        "- Source files modified, transferred, converted, normalized, or preprocessed: 0.",
        "",
    ]
    return "\n".join(lines)


def _fsync(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def run_preflight(
    lock_validation_path: Path = DEFAULT_LOCK_VALIDATION,
    locked_manifest_path: Path = DEFAULT_LOCKED_MANIFEST,
    inventory_path: Path = DEFAULT_INVENTORY,
    final_dir: Path = DEFAULT_FINAL_DIR,
    *,
    workers: int = 4,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite source-metadata preflight: {final_dir}")
    generated = generated_utc or utc_now()
    inputs = load_inputs(lock_validation_path, locked_manifest_path, inventory_path)

    # Suppress pydicom's warning emission for raw nonconformant UID strings.  We
    # validate and report the exact values ourselves without normalizing them.
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE

    dti_results: dict[str, dict[str, Any]] = {}
    t1_dicom_results: dict[str, dict[str, Any]] = {}
    t1_nifti_results: dict[str, dict[str, Any]] = {}
    tasks: dict[Any, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for source in inputs.rows:
            pair = source["pair_id"]
            tasks[
                executor.submit(
                    summarize_dicom_series,
                    inputs.groups[(pair, "DTI")],
                    source["subject_id"],
                )
            ] = (pair, "DTI")
            if source["current_t1_type"] == "Original":
                tasks[
                    executor.submit(
                        summarize_dicom_series,
                        inputs.groups[(pair, "T1")],
                        source["subject_id"],
                    )
                ] = (pair, "T1_DICOM")
            else:
                tasks[
                    executor.submit(inspect_nifti_header, inputs.groups[(pair, "T1")])
                ] = (pair, "T1_NIFTI")
        completed = 0
        for future in as_completed(tasks):
            pair, kind = tasks[future]
            result = future.result()
            if kind == "DTI":
                dti_results[pair] = result
            elif kind == "T1_DICOM":
                t1_dicom_results[pair] = result
            else:
                t1_nifti_results[pair] = result
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                print(
                    json.dumps(
                        {
                            "phase": "header_scan",
                            "series_complete": completed,
                            "series_total": len(tasks),
                            "utc": utc_now(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    projection: list[dict[str, Any]] = []
    for source in inputs.rows:
        pair = source["pair_id"]
        projection.append(
            build_projection_row(
                source,
                dti_results[pair],
                t1_dicom_results.get(pair),
                t1_nifti_results.get(pair),
                inputs.evidence,
            )
        )
    if len(projection) != EXPECTED_PAIRS or len({row["pair_id"] for row in projection}) != EXPECTED_PAIRS:
        raise RuntimeError("Projection did not produce one row per locked pair")

    original_rows = [row for row in projection if row["t1_source_class"] == "ORIGINAL_DICOM"]
    processed_rows = [row for row in projection if row["t1_source_class"] == "PROCESSED_NIFTI_DERIVED"]
    counts = {
        "pairs": len(projection),
        "primary_eligible": sum(bool(row["primary_analysis_eligible"]) for row in projection),
        "smc": sum(bool(row["smc_retained_separately"]) for row in projection),
        "original_t1": len(original_rows),
        "processed_t1": len(processed_rows),
        "dti_dicom_headers_read": sum(int(row["dti_dicom_headers_read"]) for row in projection),
        "t1_dicom_headers_read": sum(int(row["t1_dicom_headers_read"]) for row in projection),
        "processed_t1_nifti_headers_read": sum(row["t1_nifti_header_status"] == "PASS_HEADER_ONLY" for row in processed_rows),
        "dti_identity_ready": sum(row["dti_source_identity_status"] in {"PASS", "PASS_WITH_UID_SYNTAX_WARNING"} for row in projection),
        "original_t1_identity_ready": sum(row["t1_source_identity_status"] in {"PASS", "PASS_WITH_UID_SYNTAX_WARNING"} for row in original_rows),
        "processed_t1_identity_ready": sum(row["t1_source_identity_status"] == "PASS" for row in processed_rows),
        "dti_uid_syntax_nonconformant": sum(str(row["dti_series_uid_syntax_status"]).startswith("NONCONFORMANT") for row in projection),
        "t1_uid_syntax_nonconformant": sum(str(row["t1_dicom_series_uid_syntax_status"]).startswith("NONCONFORMANT") for row in original_rows),
        "scanner_model_consistent": sum(row["scanner_model_consistency_status"] == "PASS" for row in projection),
        "manufacturer_manifest_match": sum(bool(row["dti_manufacturer_manifest_match"]) for row in projection),
        "field_strength_manifest_match": sum(bool(row["dti_field_strength_manifest_match"]) for row in projection),
        "in_plane_pe_complete_consistent": sum(row["dti_dicom_in_plane_phase_encoding_status"] == "PASS" for row in projection),
        "series_with_public_b_value": sum(int(row["dti_public_b_value_headers_present"]) > 0 for row in projection),
        "series_with_public_gradient": sum(int(row["dti_public_gradient_headers_present"]) > 0 for row in projection),
        "source_identity_blocked": sum(row["workflow_projection_status"] == "BLOCKED_SOURCE_IDENTITY" for row in projection),
        "conversion_pending": sum(row["workflow_projection_status"] == "SOURCE_IDENTITY_READY_CONVERSION_PENDING" for row in projection),
        "canary_ready": sum(bool(row["canary_ready"]) for row in projection),
    }

    stage = Path(tempfile.mkdtemp(prefix=".source-metadata-preflight-v1-", dir=final_dir.parent))
    try:
        table_path = stage / OUTPUT_TABLE
        with table_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in projection:
                writer.writerow({key: _csv_value(row[key]) for key in OUTPUT_COLUMNS})
        memo_path = stage / OUTPUT_MEMO
        memo_path.write_text(
            build_memo(generated, projection, counts, inputs.evidence), encoding="utf-8"
        )
        outputs = {
            "workflow_projection": {
                "path": str(final_dir / OUTPUT_TABLE),
                "sha256": sha256_file(table_path),
                "size_bytes": table_path.stat().st_size,
                "row_count": len(projection),
            },
            "memo": {
                "path": str(final_dir / OUTPUT_MEMO),
                "sha256": sha256_file(memo_path),
                "size_bytes": memo_path.stat().st_size,
            },
        }
        release_status = (
            "SOURCE_METADATA_PREFLIGHT_COMPLETE_CONVERSION_PENDING"
            if counts["source_identity_blocked"] == 0
            else "SOURCE_METADATA_PREFLIGHT_COMPLETE_SOURCE_BLOCKERS_PRESENT"
        )
        validation = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "release_status": release_status,
            "generated_utc": generated,
            "canary_authorized": False,
            "inputs": inputs.evidence,
            "methods": {
                "header_scan_workers": workers,
                "dicom_reader": f"pydicom {pydicom.__version__}",
                "dicom_access": "all locked DICOM files; explicit tag allow-list; stop_before_pixels=True",
                "nifti_reader": f"nibabel {nib.__version__}",
                "nifti_access": "header object only; no image data proxy access",
                "source_id": "SHA-256(kind NUL ADNI_IMAGE_ID NUL id NUL LOCKED_SERIES_CONTENT_BUNDLE_SHA256 NUL bundle newline)",
                "phase_readout_policy": "blank PENDING_CONVERSION; never inferred or defaulted",
            },
            "counts": counts,
            "checks": {
                "content_lock_validation_checksum_bound": True,
                "locked_manifest_checksum_bound": True,
                "locked_inventory_checksum_bound": True,
                "all_530_pairs_projected": len(projection) == EXPECTED_PAIRS,
                "all_locked_files_accounted": (
                    counts["dti_dicom_headers_read"]
                    + counts["t1_dicom_headers_read"]
                    + counts["processed_t1_nifti_headers_read"]
                    == EXPECTED_FILES
                ),
                "processed_t1_dicom_uid_nullable": all(not row["t1_dicom_series_uid"] for row in processed_rows),
                "stable_t1_source_id_complete": all(SHA256_RE.fullmatch(str(row["t1_source_id"])) for row in projection),
                "phase_encoding_direction_not_guessed": all(not row["phase_encoding_direction"] for row in projection),
                "total_readout_time_not_guessed": all(not row["total_readout_time"] for row in projection),
                "conversion_metadata_pending": all(row["phase_encoding_direction_status"] == "PENDING_CONVERSION" and row["total_readout_time_status"] == "PENDING_CONVERSION" and row["gradient_table_status"] == "PENDING_CONVERSION" for row in projection),
                "no_canary_ready_rows": counts["canary_ready"] == 0,
            },
            "safety": {
                "dicom_headers_read": counts["dti_dicom_headers_read"] + counts["t1_dicom_headers_read"],
                "nifti_headers_read": counts["processed_t1_nifti_headers_read"],
                "pixel_arrays_read": 0,
                "voxel_arrays_read": 0,
                "connectome_or_biological_outcomes_read": 0,
                "images_converted_or_processed": 0,
                "source_files_modified_or_deleted": 0,
                "production_paths_modified": 0,
            },
            "outputs": outputs,
        }
        validation_path = stage / OUTPUT_VALIDATION
        validation_path.write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for path in (table_path, memo_path, validation_path):
            _fsync(path)
        descriptor = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.rename(stage, final_dir)
        descriptor = os.open(final_dir.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return validation
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-validation", type=Path, default=DEFAULT_LOCK_VALIDATION)
    parser.add_argument("--locked-manifest", type=Path, default=DEFAULT_LOCKED_MANIFEST)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_preflight(
        args.lock_validation,
        args.locked_manifest,
        args.inventory,
        args.final_dir,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "release_status": result["release_status"],
                "validation": str((args.final_dir / OUTPUT_VALIDATION).resolve()),
                "projection": str((args.final_dir / OUTPUT_TABLE).resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
