#!/usr/bin/env python3
"""Restore SC matrix repair backups without losing failed repair outputs.

Default mode is dry-run.  With --execute, current repaired artifacts are moved
to qc/sc_matrix_qc/failed_repair_outputs/<stamp>/ before the pre-repair backup
files are copied back into data/derivatives.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path
from typing import Iterable


ROOT = Path("/home/ec2-user/exp")
DERIV = ROOT / "data" / "derivatives"
QC_ROOT = DERIV / "qc" / "sc_matrix_qc"
BACKUP_ROOT = QC_ROOT / "repair_backups" / "20260512T103726Z"
RESTORE_KEYS = (
    ("parc", "AAL_t1.nii.gz"),
    ("parc", "mni2t1.mat"),
    ("parc", "mni2b0.mat"),
    ("parc", "AAL_b0.nii.gz"),
    ("tracks", "assignments_aal.csv"),
)


def read_failed_sids(validation_csv: Path) -> list[str]:
    if not validation_csv.exists():
        raise FileNotFoundError(validation_csv)
    sids: list[str] = []
    with validation_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("validation_status") == "FAIL_VALIDATION":
                sid = (row.get("sid") or "").strip()
                if sid:
                    sids.append(sid)
    return sorted(dict.fromkeys(sids))


def _backup_current(path: Path, failed_root: Path) -> None:
    if not path.exists():
        return
    rel = path.relative_to(DERIV)
    dest = failed_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    shutil.move(str(path), str(dest))


def _copy_backup(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _artifact_pairs(sid: str, backup_root: Path) -> Iterable[tuple[Path, Path]]:
    for family, name in RESTORE_KEYS:
        src = backup_root / sid / family / sid / name
        dst = DERIV / family / sid / name
        yield src, dst
    for src in sorted((backup_root / sid / "connectomes").glob(f"SC_AAL_{sid}_*.csv")):
        yield src, DERIV / "connectomes" / src.name


def restore_subject(sid: str, backup_root: Path, failed_root: Path, execute: bool) -> dict[str, object]:
    pairs = list(_artifact_pairs(sid, backup_root))
    available = [src for src, _ in pairs if src.exists()]
    current = [dst for _, dst in pairs if dst.exists()]
    row = {
        "sid": sid,
        "backup_files_available": len(available),
        "current_files_to_preserve": len(current),
        "restore_status": "DRY_RUN",
        "failed_output_backup_root": str(failed_root),
    }
    if not available:
        row["restore_status"] = "NO_BACKUP_FOUND"
        return row
    if not execute:
        return row
    # Preserve current failed repair outputs only when the matching restore
    # source exists. This prevents a partial backup from removing an artifact
    # without replacing it.
    for src, dst in pairs:
        if src.exists():
            _backup_current(dst, failed_root)
    for src, dst in pairs:
        _copy_backup(src, dst)
    row["restore_status"] = "RESTORED"
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-csv", type=Path, default=QC_ROOT / "post_parcellation_repair_completed_now_validation_subjects.csv")
    parser.add_argument("--backup-root", type=Path, default=BACKUP_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    sids = read_failed_sids(args.validation_csv)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    failed_root = QC_ROOT / "failed_repair_outputs" / stamp
    rows = [restore_subject(sid, args.backup_root, failed_root, args.execute) for sid in sids]
    report = QC_ROOT / f"restore_failed_post_parc_repair_{stamp}.csv"
    write_csv(report, rows)
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["restore_status"])
        counts[status] = counts.get(status, 0) + 1
    print(f"subjects selected: {len(sids)}")
    print("status:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"report: {report}")
    if not args.execute:
        print("dry-run only; add --execute to move failed repaired outputs aside and copy backups back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
