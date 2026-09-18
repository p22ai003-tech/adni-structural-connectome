#!/usr/bin/env python3
"""Promote retained SC-Forge v1 candidate connectomes into the live folder.

This script is intentionally narrow:
  - Reads retained_candidate_manifest.csv from a SC-Forge v1 density batch.
  - Backs up the current live matrix file before replacement.
  - Copies the retained scratch candidate into the original live filename.
  - Writes an audit CSV and JSON summary.

It never promotes fallback/quarantine rows and it never deletes files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path("/home/ec2-user/exp")
DERIV_ROOT = PROJECT_ROOT / "data" / "derivatives"
QC_ROOT = DERIV_ROOT / "qc" / "sc_matrix_qc"
LIVE_CONNECTOME_ROOT = DERIV_ROOT / "connectomes"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_v1_run() -> Path:
    runs = sorted(QC_ROOT.glob("scforge_v1_density_batch_*"))
    if not runs:
        raise FileNotFoundError(f"No scforge_v1_density_batch_* run found under {QC_ROOT}")
    return runs[-1]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def promote(run_root: Path, execute: bool) -> Path:
    manifest_path = run_root / "retained_candidate_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing retained candidate manifest: {manifest_path}")

    rows = read_manifest(manifest_path)
    if not rows:
        raise RuntimeError(f"No retained candidate rows in {manifest_path}")

    stamp = utc_stamp()
    promotion_root = QC_ROOT / f"scforge_v1_live_promotion_{stamp}"
    backup_root = DERIV_ROOT / f"connectomes_backup_scforge_v1_{stamp}"
    promotion_root.mkdir(parents=True, exist_ok=True)
    if execute:
        backup_root.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict[str, str]] = []
    errors: list[str] = []
    promoted_files = 0
    promoted_subjects: set[str] = set()

    for row in rows:
        sid = row.get("sid", "")
        metric = row.get("metric", "")
        src = Path(row.get("candidate_path") or row.get("selected_path") or "")
        dst = Path(row.get("baseline_path") or "")

        if not sid or not metric:
            errors.append(f"Missing sid/metric in row: {row}")
            continue
        if not src.exists():
            errors.append(f"Missing candidate for {sid} {metric}: {src}")
            continue
        if not dst.exists():
            errors.append(f"Missing live baseline for {sid} {metric}: {dst}")
            continue
        if LIVE_CONNECTOME_ROOT not in dst.parents:
            errors.append(f"Destination is not in live connectome folder for {sid} {metric}: {dst}")
            continue

        backup = backup_root / dst.name
        src_hash = sha256_file(src)
        old_hash = sha256_file(dst)

        if execute:
            shutil.copy2(dst, backup)
            shutil.copy2(src, dst)
            new_hash = sha256_file(dst)
        else:
            new_hash = ""

        promoted_files += 1
        promoted_subjects.add(sid)
        audit_rows.append(
            {
                "sid": sid,
                "metric": metric,
                "decision": row.get("decision", ""),
                "source_candidate": str(src),
                "live_destination": str(dst),
                "backup_path": str(backup),
                "candidate_sha256": src_hash,
                "old_live_sha256": old_hash,
                "new_live_sha256": new_hash,
                "executed": str(bool(execute)),
            }
        )

    audit_path = promotion_root / "promotion_audit.csv"
    with audit_path.open("w", newline="") as f:
        fieldnames = [
            "sid",
            "metric",
            "decision",
            "source_candidate",
            "live_destination",
            "backup_path",
            "candidate_sha256",
            "old_live_sha256",
            "new_live_sha256",
            "executed",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    summary = {
        "run_root": str(run_root),
        "execute": execute,
        "promotion_root": str(promotion_root),
        "backup_root": str(backup_root),
        "retained_manifest": str(manifest_path),
        "promoted_subjects": len(promoted_subjects),
        "promoted_files": promoted_files,
        "errors": errors,
    }
    (promotion_root / "promotion_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if errors:
        (promotion_root / "promotion_errors.txt").write_text("\n".join(errors) + "\n")
        raise RuntimeError(
            f"Promotion completed with {len(errors)} errors; see {promotion_root / 'promotion_errors.txt'}"
        )

    return promotion_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=None, help="SC-Forge v1 density batch root.")
    parser.add_argument("--execute", action="store_true", help="Actually copy files into live connectomes.")
    args = parser.parse_args()

    run_root = args.run_root or latest_v1_run()
    promotion_root = promote(run_root=run_root, execute=args.execute)
    print(f"promotion_root={promotion_root}")
    print(f"execute={args.execute}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
