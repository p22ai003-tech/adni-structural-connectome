#!/usr/bin/env python3
"""Build the source content lock and the metadata projection from a manifest.

The workflow contract requires two artifacts that describe the raw source
universe before anything is converted:

``<out>/source_content_inventory.csv``
    one row per raw file, with its size, SHA-256 and stat identity. The
    workflow slices this per unit and re-checks every file against it before
    conversion, so a source that changed underneath the run fails closed.

``<out>/source_metadata_projection.csv`` (+ ``..._validation.json``)
    the per-pair projection of source identity that the run manifest records
    for provenance.

Both used to live under ``research_audit/outputs`` as hand-built cohort
artifacts. They are derived from the manifest and the files it points at, so
the pipeline builds them itself and any cohort can produce its own.

    python sc_source_lock.py --manifest configs/acquisition_manifest.csv \
        --out <run_root>/contract/source_lock
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

LOCK_SCHEMA_VERSION = "2.0"
PROJECTION_SCHEMA_VERSION = "1.0"

PROJECTION_COLUMNS = (
    "projection_schema_version",
    "locked_pair_manifest_sha256",
    "locked_inventory_sha256",
    "pair_id",
    "subject_id",
    "diagnosis_at_dti",
    "analysis_role",
    "processing_authorized",
    "dti_image_id",
    "t1_image_id",
    "dti_study_date",
    "t1_study_date",
    "abs_pair_gap_days",
    "timing_stratum",
    "phase",
    "site",
    "protocol",
    "manufacturer",
    "field_strength_t",
    "dti_source_kind",
    "t1_source_kind",
    "dti_source_id",
    "t1_source_id",
    "dti_dicom_series_uid",
    "t1_dicom_series_uid",
    "dti_raw_bundle_sha256",
    "t1_raw_bundle_sha256",
    "pair_content_bundle_sha256",
    "dti_inventory_file_count",
    "dti_inventory_total_bytes",
    "t1_inventory_file_count",
    "t1_inventory_total_bytes",
    "phase_encoding_direction",
    "phase_encoding_source",
    "total_readout_time",
    "total_readout_time_source",
    "normalization_readiness",
    "source_lock_status",
)

INVENTORY_COLUMNS = (
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
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_id(row: dict) -> str:
    return f"{row['subject_id']}_I{row['dti_image_id']}_I{row['t1_image_id']}"


def _member_files(row: dict, modality: str) -> tuple[Path, list[Path]]:
    """Return (series_dir, files) for one modality of one pair.

    ``series_dir`` is the directory the inventory paths are relative to: the
    series folder for a DICOM series, the containing folder for loose NIfTI.
    """
    if modality == "DTI":
        kind = row["dti_source_kind"]
        if kind == "dicom_series":
            root = Path(row["dti_source_path"])
            return root, sorted(p for p in root.rglob("*") if p.is_file())
        members = [
            Path(row[key])
            for key in ("dwi_nifti_path", "dwi_bvec_path", "dwi_bval_path", "dwi_json_path")
            if row.get(key)
        ]
        return (members[0].parent if members else Path()), members
    kind = row["t1_source_kind"]
    root = Path(row["t1_source_path"])
    if kind == "dicom_series":
        return root, sorted(p for p in root.rglob("*") if p.is_file())
    return root.parent, [root]


def build(manifest: Path, out_dir: Path, *, rehash: bool = True) -> dict:
    manifest = manifest.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha = sha256_file(manifest)
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))

    inventory_path = out_dir / "source_content_inventory.csv"
    projection_path = out_dir / "source_metadata_projection.csv"
    validation_path = out_dir / "source_metadata_projection_validation.json"

    counts: dict[str, dict[str, int]] = {}
    inventory_rows = 0
    missing: list[str] = []

    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS)
        writer.writeheader()
        for row in rows:
            pid = pair_id(row)
            counts[pid] = {}
            for modality, image_key in (("DTI", "dti_image_id"), ("T1", "t1_image_id")):
                series_dir, members = _member_files(row, modality)
                total_bytes = 0
                for member in members:
                    try:
                        stat = member.stat()
                    except FileNotFoundError:
                        missing.append(f"{pid} {modality} {member}")
                        continue
                    try:
                        relative = member.relative_to(series_dir).as_posix()
                    except ValueError:
                        relative = member.name
                    writer.writerow(
                        {
                            "lock_schema_version": LOCK_SCHEMA_VERSION,
                            "source_pair_manifest_sha256": manifest_sha,
                            "pair_id": pid,
                            "subject_id": row["subject_id"],
                            "modality": modality,
                            "image_id": row[image_key],
                            "series_dir": str(series_dir),
                            "relative_path": relative,
                            "absolute_path": str(member),
                            "size_bytes": stat.st_size,
                            "sha256": sha256_file(member) if rehash else "",
                            "st_dev": stat.st_dev,
                            "st_ino": stat.st_ino,
                            "st_mtime_ns": stat.st_mtime_ns,
                            "st_ctime_ns": stat.st_ctime_ns,
                        }
                    )
                    inventory_rows += 1
                    total_bytes += stat.st_size
                counts[pid][f"{modality}_count"] = len(members)
                counts[pid][f"{modality}_bytes"] = total_bytes

    inventory_sha = sha256_file(inventory_path)

    with projection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROJECTION_COLUMNS)
        writer.writeheader()
        for row in rows:
            pid = pair_id(row)
            record = {
                "projection_schema_version": PROJECTION_SCHEMA_VERSION,
                "locked_pair_manifest_sha256": manifest_sha,
                "locked_inventory_sha256": inventory_sha,
                "pair_id": pid,
                "dti_inventory_file_count": counts[pid]["DTI_count"],
                "dti_inventory_total_bytes": counts[pid]["DTI_bytes"],
                "t1_inventory_file_count": counts[pid]["T1_count"],
                "t1_inventory_total_bytes": counts[pid]["T1_bytes"],
            }
            for column in PROJECTION_COLUMNS:
                if column not in record:
                    record[column] = row.get(column, "")
            writer.writerow(record)

    projection_sha = sha256_file(projection_path)
    readiness = {}
    for row in rows:
        key = row.get("normalization_readiness", "")
        readiness[key] = readiness.get(key, 0) + 1

    validation = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source_pair_manifest": {"path": str(manifest), "sha256": manifest_sha},
        "locked_inventory": {
            "path": str(inventory_path),
            "sha256": inventory_sha,
            "row_count": inventory_rows,
        },
        "source_metadata_projection": {
            "path": str(projection_path),
            "sha256": projection_sha,
            "row_count": len(rows),
        },
        "checks": {
            "all_pairs_projected": True,
            "all_manifest_files_present": not missing,
            "file_contents_hashed": rehash,
            "phase_encoding_direction_not_guessed": all(
                row.get("phase_encoding_source", "") != "assumed" for row in rows
            ),
            "total_readout_time_not_guessed": all(
                row.get("total_readout_time_source", "") != "assumed" for row in rows
            ),
        },
        "counts": {
            "pairs": len(rows),
            "inventory_files": inventory_rows,
            "missing_files": len(missing),
            "normalization_readiness": readiness,
        },
        "missing_files": missing[:50],
    }
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

    return {
        "inventory_path": inventory_path,
        "inventory_sha256": inventory_sha,
        "inventory_row_count": inventory_rows,
        "projection_path": projection_path,
        "projection_sha256": projection_sha,
        "projection_row_count": len(rows),
        "validation_path": validation_path,
        "validation_sha256": sha256_file(validation_path),
        "missing_files": missing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="record size and stat identity but skip content hashing (fast, weaker lock)",
    )
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    result = build(args.manifest, args.out, rehash=not args.no_hash)
    print(f"inventory  : {result['inventory_row_count']} files -> {result['inventory_path']}")
    print(f"projection : {result['projection_row_count']} pairs -> {result['projection_path']}")
    if result["missing_files"]:
        print(f"MISSING    : {len(result['missing_files'])} source files", file=sys.stderr)
        for entry in result["missing_files"][:10]:
            print(f"  {entry}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
