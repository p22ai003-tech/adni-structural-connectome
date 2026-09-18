#!/usr/bin/env python3
"""Verify a fresh restored copy against the immutable local 530-release plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def write_immutable(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def verify_file(root: Path, record: Mapping[str, Any]) -> str | None:
    relative = str(record.get("relative_path", ""))
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        return f"invalid_relative_path:{relative}"
    path = root / relative
    if path.is_symlink() or not path.is_file():
        return f"missing_or_nonregular:{relative}"
    if path.stat().st_size != int(record.get("size_bytes", -1)):
        return f"size_mismatch:{relative}"
    if sha256_file(path) != record.get("sha256"):
        return f"sha256_mismatch:{relative}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-plan", type=Path, required=True)
    parser.add_argument("--restored-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("sample", "full"), default="sample")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    plan_path = args.copy_plan.expanduser().resolve()
    restored_root = args.restored_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite restore validation: {out_dir}")
    if not restored_root.is_dir():
        raise FileNotFoundError(restored_root)
    plan = load_json(plan_path)
    if plan.get("record_type") != "thesis_grade_530_s3_copy_plan":
        raise ValueError("unexpected copy plan")
    files = plan.get("local_files")
    sample_units = plan.get("restore_sample_units")
    if not isinstance(files, list) or not isinstance(sample_units, list) or len(sample_units) != 15:
        raise ValueError("copy plan file/sample evidence is malformed")
    root_names = {
        "dataset_description.json",
        "README.md",
        "release_manifest.json",
        "files_sha256.tsv",
        "case_status.csv",
        "analysis_ready_manifest.csv",
        "all_outcome_validity.csv",
    }
    if args.scope == "full":
        selected = files
    else:
        prefixes = tuple(f"cases/{unit}/" for unit in sample_units)
        selected = [
            item
            for item in files
            if str(item.get("relative_path", "")) in root_names
            or str(item.get("relative_path", "")).startswith(prefixes)
        ]
    errors = []
    verified_count = 0
    for item in selected:
        if not isinstance(item, Mapping):
            errors.append("copy_plan_file_record_not_mapping")
            continue
        error = verify_file(restored_root, item)
        if error:
            errors.append(error)
        else:
            verified_count += 1
    actual_files = {
        path.relative_to(restored_root).as_posix()
        for path in restored_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    expected_files = {str(item["relative_path"]) for item in selected if isinstance(item, Mapping)}
    if args.scope == "full":
        unexpected = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
    else:
        sample_prefixes = tuple(f"cases/{unit}/" for unit in sample_units)
        relevant_actual = {
            item for item in actual_files if item in root_names or item.startswith(sample_prefixes)
        }
        unexpected = sorted(relevant_actual - expected_files)
        missing = sorted(expected_files - relevant_actual)
    errors.extend(f"unexpected:{item}" for item in unexpected)
    errors.extend(f"missing:{item}" for item in missing)
    passed = not errors and bool(selected)
    report = {
        "schema_version": "1.0.0",
        "record_type": "thesis_grade_530_restored_copy_validation",
        "status": "PASS" if passed else "FAIL",
        "generated_utc": utc_now(),
        "scope": args.scope,
        "copy_plan_path": str(plan_path),
        "copy_plan_sha256": sha256_file(plan_path),
        "destination": plan.get("destination"),
        "restored_root": str(restored_root),
        "expected_file_count": len(selected),
        "verified_file_count": verified_count,
        "restore_sample_units": sample_units,
        "errors": errors,
        "copy_green": passed and args.scope == "full",
        "sample_restore_green": passed and args.scope == "sample",
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    write_immutable(
        out_dir / "restore_validation.json",
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
    )
    summary = "\n".join(
        [
            "# Thesis-grade release restore validation",
            "",
            f"**Status:** {report['status']}",
            f"**Scope:** {args.scope}",
            f"**Files verified:** {report['verified_file_count']}/{report['expected_file_count']}",
            "",
            *[f"- {error}" for error in errors],
        ]
    ) + "\n"
    write_immutable(out_dir / "restore_validation.md", summary.encode())
    print(json.dumps({"status": report["status"], "scope": args.scope, "files": len(selected)}, indent=2))
    if args.require_pass and not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
