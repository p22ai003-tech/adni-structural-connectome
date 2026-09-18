#!/usr/bin/env python3
"""Prepare a checksum-bound S3 copy plan for an already GREEN 530-case release.

This command never contacts S3.  It fails closed unless the independent release
evaluator reports full-hash TECHNICAL_GREEN_530.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path("/home/ec2-user/exp")
DEFAULT_CONTRACT = (
    ROOT / "research_audit/outputs/thesis_grade_530_release_contract_v1/contract.json"
)
DEFAULT_INPUT_MANIFEST = ROOT / "research_audit/outputs/connectome_v2_input_manifest_v2.csv"
EVALUATOR_SOURCE = ROOT / "research_audit/evaluate_thesis_grade_530_release.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def hash_payload(root: Path, *, excluded: set[str]) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if path.is_symlink():
            raise ValueError(f"release payload contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"release payload contains a nonregular entry: {relative}")
        record = file_record(path)
        record["relative_path"] = relative
        records.append(record)
    return records


def choose_restore_units(rows: list[dict[str, str]], count: int = 15) -> list[str]:
    """Deterministic diagnosis-blind sample spanning manufacturer/T1 source cells."""
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{row['manufacturer']}|{row['t1_source_kind']}|{row['subject_id']}|{row['dti_image_id']}".encode()
        ).hexdigest(),
    )
    selected: list[dict[str, str]] = []
    seen_cells: set[tuple[str, str]] = set()
    for row in ordered:
        cell = (row["manufacturer"], row["t1_source_kind"])
        if cell not in seen_cells:
            selected.append(row)
            seen_cells.add(cell)
    for row in ordered:
        if row not in selected and len(selected) < count:
            selected.append(row)
    if len(selected) != count:
        raise ValueError(f"cannot select {count} restore units")
    return [f"{row['subject_id']}_I{row['dti_image_id']}" for row in selected]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument(
        "--s3-uri",
        default="s3://sabeesh/exp/releases/structural_connectome_530_v1/",
    )
    args = parser.parse_args()
    release_root = args.release_root.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    manifest_path = args.input_manifest.expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite copy-plan output: {out_dir}")
    evaluation = load_json(evaluation_path)
    if (
        evaluation.get("status") != "PASS"
        or evaluation.get("technical_green_530") is not True
        or evaluation.get("verification_mode") != "full"
        or evaluation.get("green_case_count") != 530
        or evaluation.get("terminal_record_count") != 530
    ):
        raise PermissionError("full-hash TECHNICAL_GREEN_530 evaluation is required")
    if not release_root.is_dir():
        raise FileNotFoundError(release_root)
    contract = load_json(contract_path)
    if evaluation.get("contract") != file_record(contract_path):
        raise PermissionError("technical evaluation is not bound to this contract")
    if evaluation.get("manifest") != file_record(manifest_path):
        raise PermissionError("technical evaluation is not bound to this input manifest")
    if evaluation.get("evaluator_source") != file_record(EVALUATOR_SOURCE):
        raise PermissionError("technical evaluation source differs")
    evaluation_checks = evaluation.get("checks")
    case_results = evaluation.get("case_results")
    if (
        not isinstance(evaluation_checks, Mapping)
        or not evaluation_checks
        or not all(value is True for value in evaluation_checks.values())
        or not isinstance(case_results, list)
        or len(case_results) != 530
        or not all(item.get("green") is True for item in case_results if isinstance(item, Mapping))
        or any(not isinstance(item, Mapping) for item in case_results)
    ):
        raise PermissionError("technical evaluation evidence is incomplete")
    expected_s3 = contract["copy_contract"]["independent_copy"]
    if args.s3_uri != expected_s3:
        raise PermissionError(f"copy target differs from contract: {expected_s3}")
    required = set(contract["copy_contract"]["required_local_artifacts"])
    generated = {"files_sha256.tsv", "release_manifest.json"}
    existing_required = required - generated
    missing = sorted(name for name in existing_required if not (release_root / name).is_file())
    if missing:
        raise FileNotFoundError(f"release root lacks required artifacts: {missing}")
    if any((release_root / name).exists() for name in generated):
        raise FileExistsError("release checksum manifests already exist")
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected_units = {f"{row['subject_id']}_I{row['dti_image_id']}" for row in rows}
    case_root = release_root / "cases"
    if not case_root.is_dir():
        raise FileNotFoundError(case_root)
    actual_units = {path.name for path in case_root.iterdir() if path.is_dir() and not path.is_symlink()}
    if actual_units != expected_units:
        raise ValueError(
            f"release case set differs: missing={len(expected_units-actual_units)}, unexpected={len(actual_units-expected_units)}"
        )
    payload = hash_payload(release_root, excluded=generated)
    if not payload:
        raise ValueError("release payload is empty")
    lines = ["relative_path\tsize_bytes\tsha256"]
    lines.extend(
        f"{item['relative_path']}\t{item['size_bytes']}\t{item['sha256']}"
        for item in payload
    )
    checksum_path = release_root / "files_sha256.tsv"
    write_immutable(checksum_path, ("\n".join(lines) + "\n").encode("utf-8"))
    restore_units = choose_restore_units(rows)
    release_manifest = {
        "schema_version": "1.0.0",
        "record_type": "thesis_grade_structural_connectome_530_local_release",
        "status": "LOCAL_GREEN_COPY_PENDING",
        "generated_utc": utc_now(),
        "release_root": str(release_root),
        "case_count": 530,
        "case_units_sha256": hashlib.sha256(
            ("\n".join(sorted(expected_units)) + "\n").encode()
        ).hexdigest(),
        "payload_file_count": len(payload),
        "payload_total_bytes": sum(int(item["size_bytes"]) for item in payload),
        "payload_checksums": file_record(checksum_path),
        "technical_evaluation": file_record(evaluation_path),
        "contract": file_record(contract_path),
        "input_manifest": file_record(manifest_path),
        "restore_sample_units": restore_units,
        "copy_target": args.s3_uri,
        "copy_executed": False,
    }
    release_manifest_path = release_root / "release_manifest.json"
    write_immutable(
        release_manifest_path,
        (json.dumps(release_manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    all_files = hash_payload(release_root, excluded=set())
    out_dir.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema_version": "1.0.0",
        "record_type": "thesis_grade_530_s3_copy_plan",
        "status": "READY_NOT_EXECUTED",
        "generated_utc": utc_now(),
        "source_root": str(release_root),
        "destination": args.s3_uri,
        "file_count": len(all_files),
        "total_bytes": sum(int(item["size_bytes"]) for item in all_files),
        "local_files": all_files,
        "release_manifest": file_record(release_manifest_path),
        "restore_sample_units": restore_units,
        "upload_command": [
            "aws",
            "s3",
            "sync",
            "--no-progress",
            "--no-follow-symlinks",
            "--checksum-algorithm",
            "SHA256",
            "--checksum-mode",
            "ENABLED",
            str(release_root) + "/",
            args.s3_uri,
        ],
        "copy_executed": False,
        "required_postcopy_checks": [
            "remote object count and total bytes equal this plan",
            "stored remote checksums or S3 checksum report cover every object",
            "remote release_manifest.json SHA256 equals this plan",
            "fresh restore of root manifests and all files for the 15 sampled units reproduces SHA256",
        ],
    }
    plan_path = out_dir / "copy_plan.json"
    write_immutable(
        plan_path,
        (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode(),
    )
    summary = "\n".join(
        [
            "# Thesis-grade 530 independent-copy plan",
            "",
            "**Status:** READY_NOT_EXECUTED",
            f"**Files:** {plan['file_count']}",
            f"**Bytes:** {plan['total_bytes']}",
            f"**Destination:** `{args.s3_uri}`",
            "",
            "No network or S3 mutation was performed. Execute only after revalidating this plan and AWS identity.",
        ]
    ) + "\n"
    write_immutable(out_dir / "copy_plan.md", summary.encode())
    print(json.dumps({"status": plan["status"], "file_count": plan["file_count"], "out_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
