#!/usr/bin/env python3
"""
Helpers to audit and optionally delete redundant derivative artifacts.

Default behavior is dry-run only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass
class CleanupReport:
    safe_now: list[Path]
    legacy_optional: list[Path]
    keep: list[Path]


SAFE_NOW_NAMES = {
    "._dwi_t1_bbr",
    "._mif_convert_summary_all.tsv",
    "._pipeline_status.csv",
    "._pipeline_status_backup_20251122_203846.csv",
    "._t1_anat",
    "._t1_fast",
    "bvals",
    "bvecs",
    "check_edy_count.txt",
    "derivatives.code-workspace",
    "dwi_post_eddy_dbg.eddy_command_txt",
    "dwi_post_eddy_dbg.eddy_shell_indicies.json",
    "dwi_post_eddy_dbg.eddy_values_of_all_input_parameters",
    "edy.py",
    "edy_new.py",
    "image.png",
}

LEGACY_OPTIONAL_NAMES = {
    "_logs",
    "freesurfer_subjects",
    "mif_derived",
    "mif_convert_summary_all.tsv",
}

KEEP_NAMES = {
    "biascorr_1",
    "connectomes",
    "dti",
    "dwi_t1_bbr",
    "eddy",
    "fod",
    "metrics_excel.xlsx",
    "mif_denoised",
    "mif_dwi",
    "mif_unringed",
    "parc",
    "pipeline_status.csv",
    "pipeline_status_backup_20251122_203846.csv",
    "qc",
    "t1_anat",
    "t1_fast",
    "tracks",
}


def collect_cleanup_candidates(deriv_root: Path) -> CleanupReport:
    deriv_root = Path(deriv_root)
    safe_now: list[Path] = []
    legacy_optional: list[Path] = []
    keep: list[Path] = []

    for entry in sorted(deriv_root.iterdir(), key=lambda p: p.name.lower()):
        name = entry.name
        if name.startswith("._") or name in SAFE_NOW_NAMES:
            safe_now.append(entry)
        elif name in LEGACY_OPTIONAL_NAMES:
            legacy_optional.append(entry)
        else:
            keep.append(entry)
    return CleanupReport(safe_now=safe_now, legacy_optional=legacy_optional, keep=keep)


def print_cleanup_report(deriv_root: Path) -> CleanupReport:
    report = collect_cleanup_candidates(deriv_root)
    print("Derivative cleanup audit")
    print("------------------------")
    print(f"root: {Path(deriv_root)}")
    print()
    print("Safe to delete now:")
    for path in report.safe_now:
        print(f"  - {path.name}")
    print()
    print("Legacy / optional:")
    for path in report.legacy_optional:
        print(f"  - {path.name}")
    print()
    print("Keep:")
    for path in report.keep:
        print(f"  - {path.name}")
    return report


def delete_paths(paths: list[Path], dry_run: bool = True) -> list[Path]:
    deleted: list[Path] = []
    for path in paths:
        if dry_run:
            print(f"DRY-RUN delete: {path}")
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        deleted.append(path)
        print(f"Deleted: {path}")
    return deleted
