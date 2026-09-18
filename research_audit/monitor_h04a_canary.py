#!/usr/bin/env python3
"""Aggregate-only status monitor for the bounded H04A phase-A canary.

The monitor deliberately reports no unit directories, participant identifiers,
source paths, or log contents.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOT = Path("/data/derivatives/scforge_v2/h04a_canary_20260718_v1")
EXPECTED = 24
STAGES = (
    ("t1_normalized", "t1_normalization.json"),
    ("dwi_normalized", "dwi_normalization.json"),
    ("dwi_normalization_failed", "dwi_normalization_failure.json"),
    ("input_contract_passed", "input_contract.json"),
    ("gradient_contract_passed", "gradient_contract.json"),
    ("denoised", "dwi_denoised.mif"),
    ("degibbs_complete", "dwi_denoised_degibbs.mif"),
    ("eddy_motion_complete", "dwi_preproc.mif"),
    ("bias_corrected", "dwi_preproc_biascorr.mif"),
    ("brain_mask_complete", "dwi_brain_mask.mif"),
    ("fod_shell_selected", "fod_shell_selection.json"),
    ("response_outcome_complete", "response_calibration_outcome.json"),
    ("phase_a_completion", "response_calibration_phase_a_completion.json"),
)
PROCESS_NAMES = {
    "snakemake",
    "eddy_cpu",
    "dwidenoise",
    "mrdegibbs",
    "dwibiascorrect",
    "dwi2mask",
    "dwiextract",
    "mrmath",
    "dwiextract",
    "dwi2response",
}


def count(root: Path, basename: str) -> int:
    return sum(1 for _ in root.rglob(basename))


def active_processes() -> dict[str, int]:
    result = subprocess.run(
        ["ps", "-eo", "comm="], text=True, capture_output=True, check=True
    )
    counts: dict[str, int] = {}
    for name in (line.strip() for line in result.stdout.splitlines()):
        if name in PROCESS_NAMES:
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def snapshot(root: Path) -> dict[str, object]:
    if not root.is_dir():
        raise FileNotFoundError(f"Canary run root does not exist: {root}")
    stages = {label: count(root, basename) for label, basename in STAGES}
    response_status: dict[str, int] = {}
    for path in root.rglob("response_calibration_outcome.json"):
        try:
            status = str(json.loads(path.read_text())["status"])
        except Exception:
            status = "UNREADABLE"
        response_status[status] = response_status.get(status, 0) + 1
    files = [path for path in root.rglob("*") if path.is_file()]
    latest = max((path.stat().st_mtime for path in files), default=None)
    return {
        "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "expected_units": EXPECTED,
        "stages": stages,
        "response_status": dict(sorted(response_status.items())),
        "active_processes": active_processes(),
        "latest_file_activity_utc": (
            datetime.fromtimestamp(latest, timezone.utc).isoformat(timespec="seconds")
            if latest is not None
            else None
        ),
        "run_root_gib": round(
            sum(path.stat().st_size for path in files) / (1024**3), 3
        ),
    }


def render_text(record: dict[str, object]) -> str:
    lines = [
        f"checked_utc: {record['checked_utc']}",
        f"latest_file_activity_utc: {record['latest_file_activity_utc']}",
        f"run_root_gib: {record['run_root_gib']}",
        "",
        "aggregate_stage_counts:",
    ]
    expected = int(record["expected_units"])
    for label, value in record["stages"].items():
        lines.append(f"  {label}: {value}/{expected}")
    lines.extend(
        [
            "",
            "response_status: " + json.dumps(record["response_status"], sort_keys=True),
            "active_processes: " + json.dumps(record["active_processes"], sort_keys=True),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    record = snapshot(args.root.resolve())
    print(json.dumps(record, indent=2, sort_keys=True) if args.json else render_text(record))


if __name__ == "__main__":
    main()
