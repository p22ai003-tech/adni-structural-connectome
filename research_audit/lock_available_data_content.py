#!/usr/bin/env python3
"""Create the SL-P0-09 file-content lock for the canonical local pair manifest.

Only files below the 530 selected ``local_dti_series_dir`` and
``local_t1_series_dir`` roots are read.  Images are not decoded or processed.
Every regular file is SHA-256 hashed through a no-follow file descriptor with
pre/post ``fstat`` and post-read ``lstat`` identity checks.  A checksum- and
universe-bound SQLite checkpoint supports safe resume.
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
import sqlite3
import stat
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT = Path("/home/ec2-user/exp")
OUTPUT_ROOT = PROJECT / "research_audit" / "outputs"
CANONICAL_MANIFEST = OUTPUT_ROOT / "available_data_pair_manifest_v2.csv"
CANONICAL_VALIDATION = OUTPUT_ROOT / "available_data_pair_manifest_validation_v2.json"
FINAL_RELEASE = OUTPUT_ROOT / "available_data_content_lock_v2"
WORK_DIR = OUTPUT_ROOT / ".available_data_content_lock_v2_work"

PINNED_CANONICAL_SHA256 = (
    "a51503e30f0b8e60a2216dc4b70cb69546f302eda56bf5e75469c7bafca8fdd1"
)
EXPECTED_PAIRS = 530
LOCK_SCHEMA_VERSION = "2.0"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

INVENTORY_NAME = "available_data_file_content_inventory_v2.csv"
LOCKED_MANIFEST_NAME = "available_data_pair_manifest_locked_v2.csv"
DUPLICATES_NAME = "available_data_content_duplicate_files_v2.csv"
VALIDATION_NAME = "available_data_content_lock_validation_v2.json"
SUMMARY_NAME = "available_data_content_lock_summary_v2.md"
PROGRESS_NAME = "content_hash_progress.jsonl"
PROGRESS_FINAL_NAME = "content_hash_progress_final.json"
CHECKPOINT_NAME = "checkpoint.sqlite3"
CHECKPOINT_RELEASE_NAME = "content_hash_checkpoint.sqlite3"


@dataclass(frozen=True)
class FileDescriptor:
    pair_id: str
    subject_id: str
    modality: str
    image_id: int
    series_dir: str
    relative_path: str
    absolute_path: str
    size_bytes: int
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
class CanonicalInput:
    records: list[dict[str, str]]
    columns: list[str]
    manifest_sha256: str
    manifest_size_bytes: int
    validation_sha256: str
    validation_size_bytes: int
    validation: dict[str, Any]


@dataclass(frozen=True)
class Enumeration:
    files: list[FileDescriptor]
    universe_sha256: str
    total_bytes: int
    series_count: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def strict_positive_int(value: Any, label: str) -> int:
    token = str(value).strip()
    if not re.fullmatch(r"[1-9]\d*", token):
        raise ValueError(f"{label} must be a positive integer, found {value!r}")
    return int(token)


def strict_nonnegative_int(value: Any, label: str) -> int:
    token = str(value).strip()
    if not re.fullmatch(r"(?:0|[1-9]\d*)", token):
        raise ValueError(f"{label} must be a nonnegative integer, found {value!r}")
    return int(token)


def read_csv_bytes(data: bytes, label: str) -> tuple[list[str], list[dict[str, str]]]:
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
        rows.append(
            {key.strip(): "" if value is None else value.strip() for key, value in raw.items()}
        )
    return columns, rows


def load_canonical(
    manifest_path: Path,
    validation_path: Path,
    *,
    expected_manifest_sha256: str = PINNED_CANONICAL_SHA256,
    expected_pairs: int = EXPECTED_PAIRS,
) -> CanonicalInput:
    if manifest_path.is_symlink() or validation_path.is_symlink():
        raise ValueError("Canonical manifest/validation must not be symlinks")
    manifest_bytes = manifest_path.read_bytes()
    validation_bytes = validation_path.read_bytes()
    manifest_hash = sha256_bytes(manifest_bytes)
    if manifest_hash != expected_manifest_sha256:
        raise ValueError(
            f"Canonical manifest SHA-256 drift: {manifest_hash} != {expected_manifest_sha256}"
        )
    try:
        validation = json.loads(validation_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Canonical validation is not valid UTF-8 JSON") from exc
    if validation.get("status") != "PASS" or validation.get("release_status") != "READY_FOR_HUMAN_POLICY_GATE":
        raise ValueError("Canonical validation is not PASS/READY_FOR_HUMAN_POLICY_GATE")
    recorded = validation.get("outputs", {}).get("pair_manifest", {})
    if (
        recorded.get("sha256") != manifest_hash
        or recorded.get("size_bytes") != len(manifest_bytes)
        or recorded.get("row_count") != expected_pairs
    ):
        raise ValueError("Canonical validation is not bound to the manifest bytes/count")
    columns, records = read_csv_bytes(manifest_bytes, "canonical pair manifest")
    required = {
        "pair_id",
        "subject_id",
        "dti_image_id",
        "current_t1_image_id",
        "local_dti_series_dir",
        "local_t1_series_dir",
        "local_dti_file_count",
        "local_dti_total_bytes",
        "local_dti_names_sizes_sha256",
        "local_t1_file_count",
        "local_t1_total_bytes",
        "local_t1_names_sizes_sha256",
        "pair_record_sha256",
        "provenance_source_bundle_sha256",
    }
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"Canonical manifest schema is incomplete: {missing}")
    if len(records) != expected_pairs:
        raise ValueError(f"Canonical manifest has {len(records)} rows, expected {expected_pairs}")
    pair_ids = [row["pair_id"] for row in records]
    subjects = [row["subject_id"] for row in records]
    if len(set(pair_ids)) != len(records) or len(set(subjects)) != len(records):
        raise ValueError("Canonical manifest pair_id/subject_id keys are not unique")
    return CanonicalInput(
        records=records,
        columns=columns,
        manifest_sha256=manifest_hash,
        manifest_size_bytes=len(manifest_bytes),
        validation_sha256=sha256_bytes(validation_bytes),
        validation_size_bytes=len(validation_bytes),
        validation=validation,
    )


def _absolute_real_directory(path_text: str, label: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        raise ValueError(f"{label} is not absolute: {path}")
    normalized = Path(os.path.abspath(path))
    try:
        resolved = normalized.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} cannot be resolved: {path}") from exc
    if normalized != resolved:
        raise ValueError(f"{label} contains a symlink or noncanonical component: {path}")
    mode = os.lstat(resolved).st_mode
    if not stat.S_ISDIR(mode):
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def _walk_regular_files(root: Path) -> list[tuple[str, os.stat_result]]:
    found: list[tuple[str, os.stat_result]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            entry_path = Path(entry.path)
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"Symlink inside selected series: {entry_path}")
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(entry_path)
            elif stat.S_ISREG(metadata.st_mode):
                found.append((entry_path.relative_to(root).as_posix(), metadata))
            else:
                raise ValueError(f"Nonregular object inside selected series: {entry_path}")
    found.sort(key=lambda item: item[0])
    if not found:
        raise ValueError(f"Selected series contains no regular files: {root}")
    return found


def _names_sizes_sha256(files: Sequence[tuple[str, os.stat_result]]) -> str:
    digest = hashlib.sha256()
    for relative, metadata in files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(metadata.st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def enumerate_selected_files(canonical: CanonicalInput) -> Enumeration:
    descriptors: list[FileDescriptor] = []
    roots: dict[str, tuple[str, str]] = {}
    absolute_paths: set[str] = set()
    objects: set[tuple[int, int]] = set()
    for row in sorted(canonical.records, key=lambda value: value["pair_id"]):
        pair_id = row["pair_id"]
        subject = row["subject_id"]
        for modality, path_column, id_column, count_column, bytes_column, names_hash_column in (
            (
                "DTI",
                "local_dti_series_dir",
                "dti_image_id",
                "local_dti_file_count",
                "local_dti_total_bytes",
                "local_dti_names_sizes_sha256",
            ),
            (
                "T1",
                "local_t1_series_dir",
                "current_t1_image_id",
                "local_t1_file_count",
                "local_t1_total_bytes",
                "local_t1_names_sizes_sha256",
            ),
        ):
            root = _absolute_real_directory(row[path_column], f"{pair_id} {modality} root")
            root_text = str(root)
            if root_text in roots:
                raise ValueError(
                    f"Duplicate selected series root {root_text}: {roots[root_text]} and {(pair_id, modality)}"
                )
            roots[root_text] = (pair_id, modality)
            files = _walk_regular_files(root)
            count = strict_nonnegative_int(row[count_column], f"{pair_id} {modality} file count")
            total = strict_nonnegative_int(row[bytes_column], f"{pair_id} {modality} bytes")
            if len(files) != count or sum(item.st_size for _, item in files) != total:
                raise ValueError(f"{pair_id} {modality} count/byte inventory drift")
            if _names_sizes_sha256(files) != row[names_hash_column]:
                raise ValueError(f"{pair_id} {modality} names/sizes SHA-256 drift")
            image_id = strict_positive_int(row[id_column], f"{pair_id} {modality} image ID")
            for relative, metadata in files:
                absolute = str(root / relative)
                if absolute in absolute_paths:
                    raise ValueError(f"Duplicate selected file path: {absolute}")
                absolute_paths.add(absolute)
                object_key = (metadata.st_dev, metadata.st_ino)
                if object_key in objects:
                    raise ValueError(f"Duplicate/hardlinked selected file object: {absolute}")
                objects.add(object_key)
                descriptors.append(
                    FileDescriptor(
                        pair_id=pair_id,
                        subject_id=subject,
                        modality=modality,
                        image_id=image_id,
                        series_dir=root_text,
                        relative_path=relative,
                        absolute_path=absolute,
                        size_bytes=metadata.st_size,
                        st_dev=metadata.st_dev,
                        st_ino=metadata.st_ino,
                        st_mtime_ns=metadata.st_mtime_ns,
                        st_ctime_ns=metadata.st_ctime_ns,
                    )
                )
    root_paths = sorted(Path(value) for value in roots)
    for index, root in enumerate(root_paths):
        for other in root_paths[index + 1 :]:
            if other.is_relative_to(root):
                raise ValueError(f"Selected series roots overlap/nest: {root} and {other}")
    descriptors.sort(key=lambda item: (item.pair_id, item.modality, item.relative_path))
    digest = hashlib.sha256()
    for item in descriptors:
        values = (
            item.pair_id,
            item.subject_id,
            item.modality,
            str(item.image_id),
            item.series_dir,
            item.relative_path,
            item.absolute_path,
            str(item.size_bytes),
            str(item.st_dev),
            str(item.st_ino),
            str(item.st_mtime_ns),
            str(item.st_ctime_ns),
        )
        digest.update("\0".join(values).encode("utf-8"))
        digest.update(b"\n")
    return Enumeration(
        files=descriptors,
        universe_sha256=digest.hexdigest(),
        total_bytes=sum(item.size_bytes for item in descriptors),
        series_count=len(roots),
    )


def _same_identity(metadata: os.stat_result, descriptor: FileDescriptor) -> bool:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    ) == descriptor.identity


def hash_file_stable(
    descriptor: FileDescriptor,
    *,
    block_size: int = 4 * 1024 * 1024,
    after_first_block: Callable[[], None] | None = None,
) -> str:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(descriptor.absolute_path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not _same_identity(before, descriptor):
            raise RuntimeError(f"File identity changed before hashing: {descriptor.absolute_path}")
        digest = hashlib.sha256()
        first = True
        while True:
            block = os.read(fd, block_size)
            if not block:
                break
            digest.update(block)
            if first and after_first_block is not None:
                after_first_block()
            first = False
        after = os.fstat(fd)
        if not _same_identity(after, descriptor):
            raise RuntimeError(f"File mutated during hashing: {descriptor.absolute_path}")
    finally:
        os.close(fd)
    final = os.lstat(descriptor.absolute_path)
    if not stat.S_ISREG(final.st_mode) or not _same_identity(final, descriptor):
        raise RuntimeError(f"File changed/replaced after hashing: {descriptor.absolute_path}")
    return digest.hexdigest()


class Checkpoint:
    def __init__(self, path: Path, binding: Mapping[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError("Checkpoint path must not be a symlink")
        existed = path.exists()
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS file_hashes (
                absolute_path TEXT PRIMARY KEY,
                pair_id TEXT NOT NULL,
                modality TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                st_dev INTEGER NOT NULL,
                st_ino INTEGER NOT NULL,
                st_mtime_ns INTEGER NOT NULL,
                st_ctime_ns INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                hashed_utc TEXT NOT NULL
            )"""
        )
        existing = dict(self.connection.execute("SELECT key, value FROM meta"))
        expected = {"checkpoint_schema_version": LOCK_SCHEMA_VERSION, **binding}
        if existed and existing and existing != expected:
            raise ValueError("Checkpoint binding does not match the canonical file universe")
        if not existing:
            self.connection.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)", sorted(expected.items())
            )
            self.connection.commit()

    def cached_hash(self, item: FileDescriptor) -> str | None:
        row = self.connection.execute(
            """SELECT pair_id, modality, relative_path, size_bytes, st_dev, st_ino,
                      st_mtime_ns, st_ctime_ns, sha256
               FROM file_hashes WHERE absolute_path = ?""",
            (item.absolute_path,),
        ).fetchone()
        if row is None:
            return None
        expected = (
            item.pair_id,
            item.modality,
            item.relative_path,
            item.size_bytes,
            item.st_dev,
            item.st_ino,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if row[:8] != expected or not SHA256_RE.fullmatch(row[8]):
            raise ValueError(f"Unsafe/stale checkpoint row for {item.absolute_path}")
        current = os.lstat(item.absolute_path)
        if not stat.S_ISREG(current.st_mode) or not _same_identity(current, item):
            raise RuntimeError(f"Checkpointed file changed: {item.absolute_path}")
        return row[8]

    def add(self, item: FileDescriptor, digest: str) -> None:
        self.connection.execute(
            """INSERT INTO file_hashes(
                   absolute_path, pair_id, modality, relative_path, size_bytes,
                   st_dev, st_ino, st_mtime_ns, st_ctime_ns, sha256, hashed_utc
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.absolute_path,
                item.pair_id,
                item.modality,
                item.relative_path,
                item.size_bytes,
                item.st_dev,
                item.st_ino,
                item.st_mtime_ns,
                item.st_ctime_ns,
                digest,
                utc_now(),
            ),
        )

    def commit(self) -> None:
        self.connection.commit()

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM file_hashes").fetchone()[0])

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _progress_payload(
    *,
    phase: str,
    started: float,
    total_files: int,
    total_bytes: int,
    complete_files: int,
    complete_bytes: int,
    new_files: int,
    reused_files: int,
) -> dict[str, Any]:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = complete_bytes / elapsed
    remaining = max(total_bytes - complete_bytes, 0)
    return {
        "utc": utc_now(),
        "phase": phase,
        "files_total": total_files,
        "bytes_total": total_bytes,
        "files_complete": complete_files,
        "bytes_complete": complete_bytes,
        "files_hashed_new": new_files,
        "files_reused_checkpoint": reused_files,
        "elapsed_seconds": round(elapsed, 3),
        "bytes_per_second": round(rate, 3),
        "eta_seconds": round(remaining / rate, 3) if rate > 0 else None,
    }


def _emit_progress(work_dir: Path, payload: Mapping[str, Any]) -> None:
    _atomic_json(work_dir / "progress.json", payload)
    with (work_dir / PROGRESS_NAME).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(payload, sort_keys=True), flush=True)


def hash_with_checkpoint(
    enumeration: Enumeration,
    checkpoint: Checkpoint,
    work_dir: Path,
    *,
    commit_every: int = 1000,
    progress_every: int = 5000,
) -> tuple[dict[str, str], dict[str, Any]]:
    if commit_every <= 0 or progress_every <= 0:
        raise ValueError("commit_every and progress_every must be positive")
    started = time.monotonic()
    hashes: dict[str, str] = {}
    new_files = 0
    reused_files = 0
    complete_bytes = 0
    _emit_progress(
        work_dir,
        _progress_payload(
            phase="hashing",
            started=started,
            total_files=len(enumeration.files),
            total_bytes=enumeration.total_bytes,
            complete_files=0,
            complete_bytes=0,
            new_files=0,
            reused_files=0,
        ),
    )
    for index, item in enumerate(enumeration.files, start=1):
        digest = checkpoint.cached_hash(item)
        if digest is None:
            digest = hash_file_stable(item)
            checkpoint.add(item, digest)
            new_files += 1
        else:
            reused_files += 1
        hashes[item.absolute_path] = digest
        complete_bytes += item.size_bytes
        if index % commit_every == 0:
            checkpoint.commit()
        if index % progress_every == 0 or index == len(enumeration.files):
            checkpoint.commit()
            _emit_progress(
                work_dir,
                _progress_payload(
                    phase="hashing",
                    started=started,
                    total_files=len(enumeration.files),
                    total_bytes=enumeration.total_bytes,
                    complete_files=index,
                    complete_bytes=complete_bytes,
                    new_files=new_files,
                    reused_files=reused_files,
                ),
            )
    checkpoint.commit()
    if checkpoint.count() != len(enumeration.files):
        raise RuntimeError("Checkpoint row count does not equal the frozen file universe")
    final = _progress_payload(
        phase="hashing_complete",
        started=started,
        total_files=len(enumeration.files),
        total_bytes=enumeration.total_bytes,
        complete_files=len(enumeration.files),
        complete_bytes=enumeration.total_bytes,
        new_files=new_files,
        reused_files=reused_files,
    )
    _emit_progress(work_dir, final)
    return hashes, final


def _csv_value(value: Any) -> Any:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return ""
    return value


def _series_metrics(
    files: Sequence[FileDescriptor], hashes: Mapping[str, str]
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[FileDescriptor]] = {}
    for item in files:
        grouped.setdefault((item.pair_id, item.modality), []).append(item)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, items in grouped.items():
        digest = hashlib.sha256()
        for item in sorted(items, key=lambda value: value.relative_path):
            digest.update(item.relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(item.size_bytes).encode("ascii"))
            digest.update(b"\0")
            digest.update(hashes[item.absolute_path].encode("ascii"))
            digest.update(b"\n")
        result[key] = {
            "file_count": len(items),
            "total_bytes": sum(item.size_bytes for item in items),
            "content_bundle_sha256": digest.hexdigest(),
        }
    return result


def write_release(
    canonical: CanonicalInput,
    enumeration: Enumeration,
    hashes: Mapping[str, str],
    final_progress: Mapping[str, Any],
    work_dir: Path,
    final_dir: Path,
    checkpoint_path: Path,
    run_started_utc: str,
) -> dict[str, Any]:
    if final_dir.exists():
        raise FileExistsError(f"Refusing to replace existing release: {final_dir}")
    stage = Path(tempfile.mkdtemp(prefix=".content-lock-v2-stage-", dir=final_dir.parent))
    started = time.monotonic()
    metrics = _series_metrics(enumeration.files, hashes)
    content_counts = Counter(hashes.values())
    duplicate_hashes = {digest: count for digest, count in content_counts.items() if count > 1}
    try:
        inventory_path = stage / INVENTORY_NAME
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
        with inventory_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=inventory_columns, lineterminator="\n")
            writer.writeheader()
            for item in enumeration.files:
                writer.writerow(
                    {
                        "lock_schema_version": LOCK_SCHEMA_VERSION,
                        "source_pair_manifest_sha256": canonical.manifest_sha256,
                        "pair_id": item.pair_id,
                        "subject_id": item.subject_id,
                        "modality": item.modality,
                        "image_id": item.image_id,
                        "series_dir": item.series_dir,
                        "relative_path": item.relative_path,
                        "absolute_path": item.absolute_path,
                        "size_bytes": item.size_bytes,
                        "sha256": hashes[item.absolute_path],
                        "st_dev": item.st_dev,
                        "st_ino": item.st_ino,
                        "st_mtime_ns": item.st_mtime_ns,
                        "st_ctime_ns": item.st_ctime_ns,
                    }
                )

        duplicate_path = stage / DUPLICATES_NAME
        duplicate_columns = [
            "sha256",
            "duplicate_group_file_count",
            "size_bytes",
            "pair_id",
            "subject_id",
            "modality",
            "absolute_path",
        ]
        duplicate_file_rows = 0
        with duplicate_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=duplicate_columns, lineterminator="\n")
            writer.writeheader()
            for item in enumeration.files:
                digest = hashes[item.absolute_path]
                if digest not in duplicate_hashes:
                    continue
                duplicate_file_rows += 1
                writer.writerow(
                    {
                        "sha256": digest,
                        "duplicate_group_file_count": duplicate_hashes[digest],
                        "size_bytes": item.size_bytes,
                        "pair_id": item.pair_id,
                        "subject_id": item.subject_id,
                        "modality": item.modality,
                        "absolute_path": item.absolute_path,
                    }
                )

        appended = [
            "content_lock_schema_version",
            "content_lock_source_manifest_sha256",
            "content_lock_file_universe_sha256",
            "local_dti_content_file_count",
            "local_dti_content_total_bytes",
            "local_dti_content_bundle_sha256",
            "local_t1_content_file_count",
            "local_t1_content_total_bytes",
            "local_t1_content_bundle_sha256",
            "pair_content_bundle_sha256",
            "locked_pair_record_sha256",
            "locked_local_content_sha256_status",
        ]
        if set(appended) & set(canonical.columns):
            raise ValueError("Locked-manifest columns collide with canonical schema")
        locked_path = stage / LOCKED_MANIFEST_NAME
        with locked_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=canonical.columns + appended, lineterminator="\n"
            )
            writer.writeheader()
            for source in canonical.records:
                dti = metrics[(source["pair_id"], "DTI")]
                t1 = metrics[(source["pair_id"], "T1")]
                pair_digest = hashlib.sha256(
                    (
                        "DTI\0"
                        + dti["content_bundle_sha256"]
                        + "\nT1\0"
                        + t1["content_bundle_sha256"]
                        + "\n"
                    ).encode("ascii")
                ).hexdigest()
                locked_row_digest = hashlib.sha256(
                    (
                        source["pair_record_sha256"]
                        + "\0"
                        + dti["content_bundle_sha256"]
                        + "\0"
                        + t1["content_bundle_sha256"]
                        + "\n"
                    ).encode("ascii")
                ).hexdigest()
                row = dict(source)
                row.update(
                    {
                        "content_lock_schema_version": LOCK_SCHEMA_VERSION,
                        "content_lock_source_manifest_sha256": canonical.manifest_sha256,
                        "content_lock_file_universe_sha256": enumeration.universe_sha256,
                        "local_dti_content_file_count": dti["file_count"],
                        "local_dti_content_total_bytes": dti["total_bytes"],
                        "local_dti_content_bundle_sha256": dti["content_bundle_sha256"],
                        "local_t1_content_file_count": t1["file_count"],
                        "local_t1_content_total_bytes": t1["total_bytes"],
                        "local_t1_content_bundle_sha256": t1["content_bundle_sha256"],
                        "pair_content_bundle_sha256": pair_digest,
                        "locked_pair_record_sha256": locked_row_digest,
                        "locked_local_content_sha256_status": "COMPLETE_SHA256",
                    }
                )
                writer.writerow({key: _csv_value(value) for key, value in row.items()})

        locked_columns, locked_rows = read_csv_bytes(
            locked_path.read_bytes(), "locked pair manifest"
        )
        if locked_columns != canonical.columns + appended or len(locked_rows) != len(
            canonical.records
        ):
            raise RuntimeError("Locked pair manifest schema/count drift")
        canonical_preserved = all(
            all(locked[key] == source[key] for key in canonical.columns)
            for source, locked in zip(canonical.records, locked_rows)
        )
        if not canonical_preserved:
            raise RuntimeError("Locked pair manifest altered canonical column values")

        shutil.copy2(work_dir / PROGRESS_NAME, stage / PROGRESS_NAME)
        _atomic_json(stage / PROGRESS_FINAL_NAME, final_progress)
        checkpoint_release_path = stage / CHECKPOINT_RELEASE_NAME
        shutil.copy2(checkpoint_path, checkpoint_release_path)
        checkpoint_hash = file_sha256(checkpoint_release_path)

        completed_utc = utc_now()
        elapsed_hash = float(final_progress["elapsed_seconds"])
        count_values = {
            "files": len(enumeration.files),
            "bytes": enumeration.total_bytes,
            "gibibytes": round(enumeration.total_bytes / 1024**3, 6),
            "dti_files": sum(item.modality == "DTI" for item in enumeration.files),
            "dti_bytes": sum(
                item.size_bytes for item in enumeration.files if item.modality == "DTI"
            ),
            "t1_files": sum(item.modality == "T1" for item in enumeration.files),
            "t1_bytes": sum(
                item.size_bytes for item in enumeration.files if item.modality == "T1"
            ),
            "unique_content_sha256": len(content_counts),
            "duplicate_content_hash_groups": len(duplicate_hashes),
            "files_in_duplicate_content_groups": duplicate_file_rows,
            "checkpoint_reused_files": final_progress["files_reused_checkpoint"],
            "newly_hashed_files": final_progress["files_hashed_new"],
        }
        summary = f"""# SL-P0-09 local content-hash lock summary

**Status:** PASS — CONTENT_HASH_LOCKED  
**Completed:** {completed_utc}  
**Canonical manifest SHA-256:** `{canonical.manifest_sha256}`

- Selected pairs: **{len(canonical.records)}**
- Selected series roots: **{enumeration.series_count}**
- Files hashed: **{len(enumeration.files):,}**
- Bytes hashed: **{enumeration.total_bytes:,}** ({enumeration.total_bytes / 1024**3:.2f} GiB)
- DTI: **{count_values['dti_files']:,} files / {count_values['dti_bytes']:,} bytes**
- T1: **{count_values['t1_files']:,} files / {count_values['t1_bytes']:,} bytes**
- Hash runtime: **{elapsed_hash:.1f} seconds** ({float(final_progress['bytes_per_second']) / 1024**2:.2f} MiB/s)
- Checkpoint reused/new: **{final_progress['files_reused_checkpoint']:,} / {final_progress['files_hashed_new']:,} files**
- Duplicate content hash groups: **{len(duplicate_hashes):,}** involving **{duplicate_file_rows:,} files**

Every inventory row is a SHA-256 of exact file bytes. The source canonical manifest was not modified. The duplicate-content report is descriptive; content equality does not by itself imply an identity or acquisition error.
"""
        summary_path = stage / SUMMARY_NAME
        summary_path.write_text(summary, encoding="utf-8")
        output_paths = {
            "file_inventory": inventory_path,
            "locked_pair_manifest": locked_path,
            "duplicate_content_files": duplicate_path,
            "progress_log": stage / PROGRESS_NAME,
            "progress_final": stage / PROGRESS_FINAL_NAME,
            "closed_checkpoint": checkpoint_release_path,
            "summary": summary_path,
        }
        output_evidence = {
            name: {
                "path": str(final_dir / path.name),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in output_paths.items()
        }
        output_evidence["file_inventory"]["row_count"] = len(enumeration.files)
        output_evidence["locked_pair_manifest"]["row_count"] = len(canonical.records)
        output_evidence["duplicate_content_files"]["row_count"] = duplicate_file_rows

        validation = {
            "schema_version": LOCK_SCHEMA_VERSION,
            "status": "PASS",
            "release_status": "CONTENT_HASH_LOCKED",
            "run_started_utc": run_started_utc,
            "completed_utc": completed_utc,
            "source": {
                "canonical_pair_manifest": {
                    "path": str(CANONICAL_MANIFEST.resolve()),
                    "size_bytes": canonical.manifest_size_bytes,
                    "sha256": canonical.manifest_sha256,
                    "hard_pinned_sha256": PINNED_CANONICAL_SHA256,
                    "row_count": len(canonical.records),
                },
                "canonical_validation": {
                    "path": str(CANONICAL_VALIDATION.resolve()),
                    "size_bytes": canonical.validation_size_bytes,
                    "sha256": canonical.validation_sha256,
                },
            },
            "scope": {
                "selected_pair_count": len(canonical.records),
                "selected_series_count": enumeration.series_count,
                "only_manifest_series_columns_hashed": [
                    "local_dti_series_dir",
                    "local_t1_series_dir",
                ],
                "images_processed": 0,
                "images_transferred": 0,
                "production_paths_modified": 0,
            },
            "counts": count_values,
            "runtime": {
                "hashing_seconds": elapsed_hash,
                "hashing_bytes_per_second": final_progress["bytes_per_second"],
                "release_write_seconds": round(time.monotonic() - started, 3),
            },
            "checks": {
                "canonical_manifest_hash_pinned": canonical.manifest_sha256 == PINNED_CANONICAL_SHA256,
                "all_selected_series_unique_nonoverlapping": True,
                "all_selected_files_regular_and_unique": True,
                "pre_inventory_matches_canonical_names_sizes": True,
                "fd_pre_post_and_path_post_stat_checks_passed": True,
                "post_hash_whole_inventory_matches_pre_inventory": True,
                "all_files_sha256_complete": len(hashes) == len(enumeration.files),
                "locked_manifest_preserves_canonical_rows": canonical_preserved,
                "locked_row_hash_binds_source_row_and_both_series": True,
                "duplicate_content_report_emitted": True,
                "no_image_processing_or_transfer": True,
            },
            "file_universe_sha256": enumeration.universe_sha256,
            "checkpoint": {
                "work_path": str(checkpoint_path.resolve()),
                "released_copy_path": str(final_dir / CHECKPOINT_RELEASE_NAME),
                "sha256": checkpoint_hash,
                "identity_bound_resume": True,
                "closed_before_copy": True,
            },
            "hash_contract": {
                "file_hash": "SHA-256 over exact file bytes",
                "series_bundle_hash": "SHA-256 over relative_path NUL size NUL file_sha256 newline, sorted by relative_path",
                "pair_bundle_hash": "SHA-256 over DTI and T1 series bundle hashes with modality labels",
                "locked_row_hash": "SHA-256 over source pair_record_sha256 NUL DTI series bundle NUL T1 series bundle newline",
                "mutation_detection": "O_NOFOLLOW open, fstat before/after, lstat after, and pre/post whole-universe comparison",
            },
            "outputs": output_evidence,
        }
        validation_path = stage / VALIDATION_NAME
        _atomic_json(validation_path, validation)
        for artifact in list(output_paths.values()) + [validation_path]:
            _fsync_file(artifact)
        _fsync_directory(stage)
        os.rename(stage, final_dir)
        _fsync_directory(final_dir.parent)
        return validation
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def run_lock(
    *,
    manifest_path: Path = CANONICAL_MANIFEST,
    validation_path: Path = CANONICAL_VALIDATION,
    work_dir: Path = WORK_DIR,
    final_dir: Path = FINAL_RELEASE,
    expected_manifest_sha256: str = PINNED_CANONICAL_SHA256,
    expected_pairs: int = EXPECTED_PAIRS,
    progress_every: int = 5000,
) -> dict[str, Any]:
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if final_dir.exists():
        raise FileExistsError(f"Refusing to replace existing release: {final_dir}")
    run_started_utc = utc_now()
    canonical = load_canonical(
        manifest_path,
        validation_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_pairs=expected_pairs,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    enumeration = enumerate_selected_files(canonical)
    binding = {
        "canonical_manifest_sha256": canonical.manifest_sha256,
        "canonical_validation_sha256": canonical.validation_sha256,
        "file_universe_sha256": enumeration.universe_sha256,
        "file_count": str(len(enumeration.files)),
        "total_bytes": str(enumeration.total_bytes),
    }
    checkpoint_path = work_dir / CHECKPOINT_NAME
    checkpoint = Checkpoint(checkpoint_path, binding)
    try:
        hashes, final_progress = hash_with_checkpoint(
            enumeration,
            checkpoint,
            work_dir,
            progress_every=progress_every,
        )
        post = enumerate_selected_files(canonical)
        if (
            post.universe_sha256 != enumeration.universe_sha256
            or len(post.files) != len(enumeration.files)
            or post.total_bytes != enumeration.total_bytes
        ):
            raise RuntimeError("Selected file universe changed between pre- and post-hash scans")
    finally:
        checkpoint.close()
    return write_release(
        canonical,
        enumeration,
        hashes,
        final_progress,
        work_dir,
        final_dir,
        checkpoint_path,
        run_started_utc,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CANONICAL_MANIFEST)
    parser.add_argument("--validation", type=Path, default=CANONICAL_VALIDATION)
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR)
    parser.add_argument("--final-dir", type=Path, default=FINAL_RELEASE)
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = run_lock(
        manifest_path=args.manifest,
        validation_path=args.validation,
        work_dir=args.work_dir,
        final_dir=args.final_dir,
        progress_every=args.progress_every,
    )
    counts = validation["counts"]
    print(
        f"PASS: {counts['files']:,} files / {counts['bytes']:,} bytes content-locked at "
        f"{validation['outputs']['locked_pair_manifest']['path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
