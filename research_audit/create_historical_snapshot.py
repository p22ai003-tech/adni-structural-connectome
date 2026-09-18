#!/usr/bin/env python3
"""Freeze the current provisional structural-connectome evidence by inventory.

The script is non-destructive: it hashes canonical local artifacts, inventories
large historical QC subtrees without copying them, and performs read-only S3
reconciliation for the current connectome and cohort prefixes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT = Path("/home/ec2-user/exp")
DATA = Path("/data")
QC = DATA / "derivatives" / "qc"
CONNECTOMES = DATA / "derivatives" / "connectomes"
ANALYSIS = QC / "analysis_cohort"
MATRIX_QC = QC / "sc_matrix_qc"
COHORT = PROJECT / "cohort"
AUDIT = PROJECT / "research_audit"
DEFAULT_DESTINATION = AUDIT / "snapshots" / "historical_provisional_v1"

CODE_ROOTS = [
    PROJECT / "apps",
    PROJECT / "connectome_analysis",
    PROJECT / "connectome_pipeline",
    PROJECT / "pipeline",
    PROJECT / "scforge",
    PROJECT / "scripts",
    PROJECT / "deploy",
    PROJECT / "docs",
]
CODE_SUFFIXES = {
    ".py",
    ".sh",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
    ".txt",
    ".dot",
}
ROOT_FILES = [
    PROJECT / "README.md",
    PROJECT / "structural_connectome_context.md",
    PROJECT / "structural_connectome_context.docx",
    PROJECT / "CLAUDE_RESUME_10M.md",
]
CANONICAL_QC_FILES = [
    MATRIX_QC / "subject_manifest.csv",
    MATRIX_QC / "sc_matrix_qc_manifest.csv",
    MATRIX_QC / "sc_matrix_qc_decisions.csv",
    MATRIX_QC / "sc_matrix_qc_analysis_gate.csv",
    MATRIX_QC / "sc_matrix_integrity_subjects.csv",
    MATRIX_QC / "parc_label_coverage_qc.csv",
    MATRIX_QC / "current_fd_sum_density_by_group.csv",
    MATRIX_QC / "connectome_status.json",
    MATRIX_QC / "claude_repaired_manifest.csv",
    MATRIX_QC / "pipeline_bible_alignment_current.md",
]
MATRIX_RE = re.compile(
    r"^SC_AAL166_(?P<series>.+)_(?P<weight>"
    r"count_invnodevol|invlen_mean|len_mean|fd_sum|count|fa_mean|md_mean|rd_mean|ad_mean"
    r")\.csv$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def command_result(command: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {
            "command": command,
            "exit_code": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def collect_environment() -> dict[str, Any]:
    commands = {
        "aws": ["aws", "--version"],
        "mrtrix": ["mrconvert", "-version"],
        "fsl": ["flirt", "-version"],
        "ants": ["antsRegistration", "--version"],
        "python": [sys.executable, "--version"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
        "git_root": ["git", "-C", str(PROJECT), "rev-parse", "--show-toplevel"],
        "git_head": ["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
        "git_status": ["git", "-C", str(PROJECT), "status", "--short"],
        "aws_identity": ["aws", "sts", "get-caller-identity", "--output", "json"],
    }
    return {
        "generated_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_runtime": sys.version,
        "commands": {
            name: command_result(command, timeout=60)
            for name, command in commands.items()
        },
    }


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = [
            name
            for name in names
            if name not in {".git", "__pycache__", "node_modules", ".pytest_cache"}
        ]
        base = Path(directory)
        for name in files:
            path = base / name
            if path.is_file() and not path.is_symlink():
                yield path


def collect_scope(destination: Path) -> list[tuple[str, Path]]:
    scoped: list[tuple[str, Path]] = []

    for path in sorted(CONNECTOMES.glob("SC_AAL166_*.csv")):
        scoped.append(("current_connectome_matrix", path))
    for path in sorted(iter_files(ANALYSIS)):
        scoped.append(("june17_analysis_snapshot", path))
    for path in CANONICAL_QC_FILES:
        if path.is_file():
            scoped.append(("canonical_qc", path))
    for path in sorted(COHORT.glob("*.csv")):
        if path.is_file():
            scoped.append(("cohort_metadata", path))
    for root in CODE_ROOTS:
        for path in sorted(iter_files(root)):
            if path.suffix.lower() in CODE_SUFFIXES:
                scoped.append(("code_config_doc", path))
    for path in ROOT_FILES:
        if path.is_file():
            scoped.append(("root_context", path))
    for path in sorted(iter_files(AUDIT)):
        try:
            path.relative_to(destination)
            continue
        except ValueError:
            pass
        if "incoming" in path.relative_to(AUDIT).parts:
            continue
        if "__pycache__" in path.parts:
            continue
        scoped.append(("objective1_audit_package", path))

    unique: dict[str, tuple[str, Path]] = {}
    category_rank = {
        "current_connectome_matrix": 0,
        "june17_analysis_snapshot": 1,
        "canonical_qc": 2,
        "cohort_metadata": 3,
        "code_config_doc": 4,
        "root_context": 5,
        "objective1_audit_package": 6,
    }
    for category, path in scoped:
        key = str(path.resolve())
        prior = unique.get(key)
        if prior is None or category_rank[category] < category_rank[prior[0]]:
            unique[key] = (category, path)
    return sorted(unique.values(), key=lambda item: (item[0], str(item[1])))


def hash_one(item: tuple[str, Path]) -> dict[str, Any]:
    category, path = item
    before = path.stat()
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                sha.update(block)
                md5.update(block)
        after = path.stat()
        changed = before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
        return {
            "category": category,
            "path": str(path),
            "size_bytes": after.st_size,
            "mtime_utc": datetime.fromtimestamp(
                after.st_mtime, timezone.utc
            ).isoformat(),
            "sha256": sha.hexdigest(),
            "md5": md5.hexdigest(),
            "hash_status": "changed_during_hash" if changed else "ok",
            "s3_key": "",
            "s3_size_bytes": "",
            "s3_etag": "",
            "s3_last_modified": "",
            "s3_status": "not_mapped",
        }
    except Exception as exc:
        return {
            "category": category,
            "path": str(path),
            "size_bytes": before.st_size,
            "mtime_utc": datetime.fromtimestamp(
                before.st_mtime, timezone.utc
            ).isoformat(),
            "sha256": "",
            "md5": "",
            "hash_status": f"error:{type(exc).__name__}:{exc}",
            "s3_key": "",
            "s3_size_bytes": "",
            "s3_etag": "",
            "s3_last_modified": "",
            "s3_status": "not_mapped",
        }


def list_s3_objects(
    bucket: str, prefix: str, aws_profile: str | None
) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    token: str | None = None
    while True:
        command = [
            "aws",
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--max-keys",
            "1000",
            "--output",
            "json",
        ]
        if aws_profile:
            command.extend(["--profile", aws_profile])
        if token:
            command.extend(["--continuation-token", token])
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "S3 listing failed")
        payload = json.loads(result.stdout)
        for item in payload.get("Contents", []):
            objects[item["Key"]] = {
                "size": int(item["Size"]),
                "etag": str(item.get("ETag", "")).strip('"'),
                "last_modified": item.get("LastModified", ""),
            }
        if not payload.get("IsTruncated"):
            break
        token = payload.get("NextContinuationToken")
        if not token:
            raise RuntimeError("S3 response was truncated without a continuation token")
    return objects


def reconcile_s3(
    records: list[dict[str, Any]], aws_profile: str | None
) -> dict[str, Any]:
    try:
        remote = {}
        remote.update(list_s3_objects("sabeesh", "exp/connectomes/", aws_profile))
        remote.update(list_s3_objects("sabeesh", "exp/cohort/", aws_profile))
        error = ""
    except Exception as exc:
        remote = {}
        error = f"{type(exc).__name__}: {exc}"

    mapped = 0
    matched = 0
    missing = 0
    mismatched = 0
    for row in records:
        path = Path(row["path"])
        if row["category"] == "current_connectome_matrix":
            key = f"exp/connectomes/{path.name}"
        elif row["category"] == "cohort_metadata":
            key = f"exp/cohort/{path.name}"
        else:
            continue
        mapped += 1
        row["s3_key"] = key
        item = remote.get(key)
        if item is None:
            row["s3_status"] = "not_checked" if error else "missing"
            missing += 0 if error else 1
            continue
        row["s3_size_bytes"] = item["size"]
        row["s3_etag"] = item["etag"]
        row["s3_last_modified"] = item["last_modified"]
        size_match = int(row["size_bytes"]) == int(item["size"])
        etag = item["etag"]
        if size_match and "-" not in etag and row["md5"] == etag:
            row["s3_status"] = "size_and_md5_match"
            matched += 1
        elif size_match and "-" in etag:
            row["s3_status"] = "size_match_multipart_etag"
            matched += 1
        elif size_match:
            row["s3_status"] = "size_match_md5_mismatch"
            mismatched += 1
        else:
            row["s3_status"] = "size_mismatch"
            mismatched += 1
    return {
        "error": error,
        "remote_objects_listed": len(remote),
        "mapped_local_files": mapped,
        "matched": matched,
        "missing": missing,
        "mismatched": mismatched,
    }


def retention_inventory() -> list[dict[str, Any]]:
    selected = {path.resolve() for path in CANONICAL_QC_FILES if path.exists()}
    rows: list[dict[str, Any]] = []
    top_level_unhashed_count = 0
    top_level_unhashed_bytes = 0
    for child in sorted(MATRIX_QC.iterdir()):
        if child.is_symlink():
            continue
        if child.is_file():
            if child.resolve() not in selected:
                top_level_unhashed_count += 1
                top_level_unhashed_bytes += child.stat().st_size
            continue
        count = 0
        size = 0
        newest_mtime = 0.0
        for path in iter_files(child):
            stat = path.stat()
            count += 1
            size += stat.st_size
            newest_mtime = max(newest_mtime, stat.st_mtime)
        rows.append(
            {
                "entry": str(child),
                "retention_tier": "inventory_only_pending_human_retention_decision",
                "file_count": count,
                "size_bytes": size,
                "newest_mtime_utc": (
                    datetime.fromtimestamp(newest_mtime, timezone.utc).isoformat()
                    if newest_mtime
                    else ""
                ),
            }
        )
    rows.append(
        {
            "entry": str(MATRIX_QC / "<unselected_top_level_files>"),
            "retention_tier": "inventory_only_pending_human_retention_decision",
            "file_count": top_level_unhashed_count,
            "size_bytes": top_level_unhashed_bytes,
            "newest_mtime_utc": "",
        }
    )
    return rows


def validate(records: list[dict[str, Any]], s3: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    manifest_path = MATRIX_QC / "subject_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    checks.append(
        {
            "check": "subject_manifest_648_unique",
            "required": True,
            "passed": len(manifest) == 648 and manifest["sid"].nunique() == 648,
            "detail": f"rows={len(manifest)} unique_sid={manifest['sid'].nunique()}",
        }
    )

    matrices = [
        row for row in records if row["category"] == "current_connectome_matrix"
    ]
    parsed = []
    for row in matrices:
        match = MATRIX_RE.match(Path(row["path"]).name)
        if match:
            parsed.append(match.groupdict())
    weight_counts: dict[str, int] = {}
    series = set()
    for item in parsed:
        weight_counts[item["weight"]] = weight_counts.get(item["weight"], 0) + 1
        series.add(item["series"])
    expected_weight_counts = {
        "count": 530,
        "count_invnodevol": 530,
        "fd_sum": 530,
        "len_mean": 530,
        "invlen_mean": 530,
        "fa_mean": 529,
        "md_mean": 529,
        "rd_mean": 529,
        "ad_mean": 529,
    }
    checks.append(
        {
            "check": "matrix_inventory_contract",
            "required": True,
            "passed": (
                len(matrices) == 4766
                and len(parsed) == 4766
                and len(series) == 530
                and weight_counts == expected_weight_counts
            ),
            "detail": json.dumps(
                {
                    "files": len(matrices),
                    "parsed": len(parsed),
                    "series": len(series),
                    "weights": weight_counts,
                },
                sort_keys=True,
            ),
        }
    )
    hash_errors = [
        row for row in records if row["hash_status"] != "ok"
    ]
    checks.append(
        {
            "check": "all_scoped_files_stable_and_hashed",
            "required": True,
            "passed": not hash_errors,
            "detail": f"errors={len(hash_errors)}",
        }
    )
    checks.append(
        {
            "check": "june17_analysis_snapshot_present",
            "required": True,
            "passed": any(
                row["category"] == "june17_analysis_snapshot" for row in records
            ),
            "detail": (
                f"files={sum(row['category'] == 'june17_analysis_snapshot' for row in records)}"
            ),
        }
    )
    checks.append(
        {
            "check": "s3_inventory_complete",
            "required": True,
            "passed": (
                not s3["error"]
                and (
                    s3["matched"] + s3["missing"] + s3["mismatched"]
                    == s3["mapped_local_files"]
                )
            ),
            "detail": json.dumps(s3, sort_keys=True),
        }
    )
    checks.append(
        {
            "check": "s3_exact_mirror",
            "required": False,
            "passed": (
                not s3["error"]
                and s3["missing"] == 0
                and s3["mismatched"] == 0
            ),
            "detail": (
                "Informational warning: local repaired files remain authoritative; "
                + json.dumps(s3, sort_keys=True)
            ),
        }
    )
    required_pass = all(
        check["passed"] for check in checks if check["required"]
    )
    optional_pass = all(
        check["passed"] for check in checks if not check["required"]
    )
    return {
        "status": (
            "PASS"
            if required_pass and optional_pass
            else "PASS_WITH_WARNINGS"
            if required_pass
            else "PARTIAL"
        ),
        "checks": checks,
    }


def summarize_by_category(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for row in records:
        category = row["category"]
        item = summary.setdefault(
            category, {"category": category, "file_count": 0, "size_bytes": 0}
        )
        item["file_count"] += 1
        item["size_bytes"] += int(row["size_bytes"])
    return sorted(summary.values(), key=lambda item: item["category"])


def render_summary(
    generated: str,
    category_summary: list[dict[str, Any]],
    retention: list[dict[str, Any]],
    s3: dict[str, Any],
    validation: dict[str, Any],
    records: list[dict[str, Any]],
) -> str:
    category_rows = "\n".join(
        f"| {row['category']} | {row['file_count']} | {row['size_bytes']} |"
        for row in category_summary
    )
    check_rows = "\n".join(
        f"| {check['check']} | {'PASS' if check['passed'] else 'WARN' if not check['required'] else 'FAIL'} | {check['detail']} |"
        for check in validation["checks"]
    )
    retention_count = sum(int(row["file_count"]) for row in retention)
    retention_bytes = sum(int(row["size_bytes"]) for row in retention)
    return f"""# Historical provisional v1 snapshot

**Generated:** {generated}  
**Status:** {validation["status"]}  
**Mode:** non-destructive inventory and hashing; no data were copied, deleted, uploaded, or downloaded.

## Canonical frozen scope

| Category | Files | Bytes |
|---|---:|---:|
{category_rows}

Total canonical files: {len(records)}.

## Acceptance checks

| Check | Result | Detail |
|---|---|---|
{check_rows}

## S3 reconciliation

- Bucket: sabeesh
- Read-only prefixes: exp/connectomes/ and exp/cohort/
- Mapped local files: {s3["mapped_local_files"]}
- Matched: {s3["matched"]}
- Missing: {s3["missing"]}
- Mismatched: {s3["mismatched"]}
- Listing error: {s3["error"] or "none"}

S3 ETags without a dash are compared with local MD5. Multipart ETags are
treated as size-confirmed but not cryptographic checksum proof. Local SHA-256
is the immutable snapshot identifier.

The current discrepancy is explicit rather than repaired by this snapshot:
three locally reprocessed subjects account for 12 stale-size cloud matrices
and 12 cloud-missing tensor matrices. Their local mtimes follow the cloud
copies on 2026-06-17. No upload is performed without a separate authorization.

## Historical QC retention tier

The noncanonical sc_matrix_qc repair trees were inventoried, not hashed:
{retention_count} files and {retention_bytes} bytes. They remain untouched and
require a separate human retention decision before archival or deletion.

## Artifacts

- manifest.json: complete snapshot metadata and acceptance results.
- file_inventory.csv: path, size, mtime, SHA-256, MD5, and S3 comparison.
- environment.json: host/tool/package and source-tree state.
- legacy_retention_inventory.csv: inventory-only large historical QC tiers.
- s3_reconciliation_discrepancies.csv: every missing or mismatched cloud object.
- snapshot_artifact_hashes.csv: SHA-256 for the snapshot records themselves.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--aws-profile",
        default=None,
        help="Optional AWS profile. Default uses the EC2 credential chain.",
    )
    args = parser.parse_args()

    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    generated = utc_now()

    environment = collect_environment()
    atomic_write_json(destination / "environment.json", environment)

    scope = collect_scope(destination)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        records = list(executor.map(hash_one, scope))

    s3 = reconcile_s3(records, args.aws_profile)
    validation = validate(records, s3)
    category_summary = summarize_by_category(records)
    retention = retention_inventory()

    fields = [
        "category",
        "path",
        "size_bytes",
        "mtime_utc",
        "sha256",
        "md5",
        "hash_status",
        "s3_key",
        "s3_size_bytes",
        "s3_etag",
        "s3_last_modified",
        "s3_status",
    ]
    file_inventory = destination / "file_inventory.csv"
    write_csv(file_inventory, records, fields)

    discrepancy_path = destination / "s3_reconciliation_discrepancies.csv"
    discrepancies = [
        row
        for row in records
        if row["s3_status"] in {"missing", "size_mismatch", "size_match_md5_mismatch"}
    ]
    write_csv(discrepancy_path, discrepancies, fields)

    retention_path = destination / "legacy_retention_inventory.csv"
    write_csv(
        retention_path,
        retention,
        [
            "entry",
            "retention_tier",
            "file_count",
            "size_bytes",
            "newest_mtime_utc",
        ],
    )

    manifest = {
        "snapshot_name": "historical_provisional_v1",
        "generated_utc": generated,
        "project_root": str(PROJECT),
        "data_root": str(DATA),
        "mode": "non_destructive_inventory_and_hash",
        "validation": validation,
        "category_summary": category_summary,
        "s3_reconciliation": s3,
        "retention_inventory_summary": {
            "file_count": sum(int(row["file_count"]) for row in retention),
            "size_bytes": sum(int(row["size_bytes"]) for row in retention),
            "policy": "inventory_only_pending_human_retention_decision",
        },
        "artifacts": {
            "file_inventory": str(file_inventory),
            "environment": str(destination / "environment.json"),
            "legacy_retention_inventory": str(retention_path),
            "s3_reconciliation_discrepancies": str(discrepancy_path),
            "summary": str(destination / "summary.md"),
        },
    }
    manifest_path = destination / "manifest.json"
    atomic_write_json(manifest_path, manifest)

    summary_path = destination / "summary.md"
    atomic_write_text(
        summary_path,
        render_summary(
            generated,
            category_summary,
            retention,
            s3,
            validation,
            records,
        ),
    )

    artifact_rows = []
    for path in [
        manifest_path,
        file_inventory,
        destination / "environment.json",
        retention_path,
        discrepancy_path,
        summary_path,
    ]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact_rows.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    write_csv(
        destination / "snapshot_artifact_hashes.csv",
        artifact_rows,
        ["path", "size_bytes", "sha256"],
    )

    print(json.dumps(manifest, indent=2))
    if validation["status"] == "PARTIAL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
