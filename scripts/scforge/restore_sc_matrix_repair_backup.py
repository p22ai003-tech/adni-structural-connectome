#!/usr/bin/env python3
"""Restore backed-up SC repair outputs after a failed validation.

The repair runner moves production outputs into
qc/sc_matrix_qc/repair_backups/<stamp>/<sid>/... before regenerating them.
This script preserves the failed regenerated files separately, then moves the
backup files back into their original derivative locations.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/ec2-user/exp")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from connectome_pipeline.pipeline_paths import resolve_pipeline_paths


DEFAULT_PATHS = resolve_pipeline_paths(create_layout=True)
DEFAULT_QC_ROOT = DEFAULT_PATHS.deriv_root / "qc" / "sc_matrix_qc"


def _read_sids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            out = [
                (row.get("sid") or row.get("subject") or row.get("subject_id") or "").strip()
                for row in reader
            ]
    else:
        out = [line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(dict.fromkeys(sid for sid in out if sid and not sid.startswith("#")))


def _move_current_as_failed(dest: Path, failed_dest: Path) -> None:
    if not dest.exists():
        return
    failed_dest.parent.mkdir(parents=True, exist_ok=True)
    if failed_dest.exists():
        if failed_dest.is_dir():
            shutil.rmtree(failed_dest)
        else:
            failed_dest.unlink()
    shutil.move(str(dest), str(failed_dest))


def _restore_subject(sid: str, backup_root: Path, deriv_root: Path, failed_root: Path) -> list[dict[str, object]]:
    sid_backup = backup_root / sid
    rows: list[dict[str, object]] = []
    if not sid_backup.exists():
        return [{"sid": sid, "status": "no_backup_dir", "backup_dir": str(sid_backup)}]
    files = [path for path in sid_backup.rglob("*") if path.is_file()]
    if not files:
        return [{"sid": sid, "status": "no_backup_files", "backup_dir": str(sid_backup)}]
    for backup_file in files:
        rel = backup_file.relative_to(sid_backup)
        dest = deriv_root / rel
        failed_dest = failed_root / sid / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        _move_current_as_failed(dest, failed_dest)
        shutil.move(str(backup_file), str(dest))
        rows.append(
            {
                "sid": sid,
                "status": "restored",
                "restored_to": str(dest),
                "failed_candidate_saved_to": str(failed_dest) if failed_dest.exists() else "",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--include", type=Path, required=True)
    parser.add_argument("--deriv-root", type=Path, default=DEFAULT_PATHS.deriv_root)
    parser.add_argument("--qc-root", type=Path, default=DEFAULT_QC_ROOT)
    parser.add_argument("--tag", default="")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    stamp = args.tag.strip() or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    failed_root = args.qc_root / "failed_repair_outputs" / stamp
    rows: list[dict[str, object]] = []
    for sid in _read_sids(args.include):
        rows.extend(_restore_subject(sid, args.backup_root, args.deriv_root, failed_root))

    report = args.report or args.qc_root / f"restore_repair_backup_{stamp}.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
        with report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        report.write_text("", encoding="utf-8")
    restored = sum(1 for row in rows if row.get("status") == "restored")
    print(f"restore rows={len(rows)} restored_files={restored} failed_candidates={failed_root}")
    print(f"report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
