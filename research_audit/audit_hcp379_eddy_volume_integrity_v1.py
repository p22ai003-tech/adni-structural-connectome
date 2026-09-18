#!/home/ec2-user/fsl/bin/python
"""Screen all 515 production Eddy derivatives for zero-valued volumes.

The audit is diagnosis-, outcome-, and connectome-density-blind.  It reads the
exact production ledger, computes each volume's global maximum using MRtrix,
and fail-closes any derivative containing a volume whose maximum is <= 1e-6.
For flagged derivatives only, it repeats the same test on the immutable
pre-Eddy MIF to distinguish source-image loss from a broken Eddy derivative.
Results are resumable and written atomically after every subject.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
DERIV = Path("/data/derivatives")
LEDGER = (
    EXP
    / "research_audit/outputs/"
    "hcp379_pretract_failure_recovery_ledger_v1/ledger.csv"
)
INPUT_AUDITS = (
    EXP
    / "research_audit/outputs/hcp379_scaleup_input_audit_v2/"
    "scaleup_input_audit.csv",
    EXP
    / "research_audit/outputs/"
    "hcp379_legacy_density_recovery_input_audit_v3/"
    "legacy_density_recovery_input_audit.csv",
    EXP
    / "research_audit/outputs/"
    "hcp379_archive_corrected_input_audit_v2/"
    "archive_corrected_input_audit.csv",
    EXP
    / "research_audit/outputs/hcp379_freesurfer_corrected_input_v3/"
    "freesurfer_corrected_input.csv",
)
OUTPUT_ROOT = (
    DERIV / "hcp379_v2/audits/eddy_volume_integrity_v1"
)
ROWS = OUTPUT_ROOT / "rows.csv"
SUMMARY = OUTPUT_ROOT / "summary.json"
MRSTATS = Path("/home/ec2-user/mrtrix3/bin/mrstats")
EXPECTED_N = 515
ZERO_MAXIMUM_THRESHOLD = 1.0e-6
FIELDS = (
    "unit",
    "lane",
    "status",
    "eddy_path",
    "eddy_size_bytes",
    "eddy_volume_n",
    "eddy_zero_volume_n",
    "eddy_zero_volume_indices",
    "eddy_minimum_volume_maximum",
    "raw_path",
    "raw_checked",
    "raw_volume_n",
    "raw_zero_volume_n",
    "raw_zero_volume_indices",
    "raw_minimum_volume_maximum",
    "diagnosis_labels_used",
    "outcomes_used",
    "connectome_density_used",
    "error",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {str(k): str(v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def maxima(path: Path) -> list[float]:
    result = subprocess.run(
        [str(MRSTATS), str(path), "-output", "max", "-quiet"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [float(value) for value in result.stdout.split()]
    if not values:
        raise ValueError(f"no volume maxima: {path}")
    return values


def metrics(path: Path) -> dict[str, Any]:
    values = maxima(path)
    zero = [
        index
        for index, value in enumerate(values)
        if value <= ZERO_MAXIMUM_THRESHOLD
    ]
    return {
        "volume_n": len(values),
        "zero_volume_n": len(zero),
        "zero_volume_indices": ";".join(str(value) for value in zero),
        "minimum_volume_maximum": min(values),
    }


def inputs() -> tuple[list[dict[str, str]], dict[str, str]]:
    ledger = read_csv(LEDGER)
    if len(ledger) != EXPECTED_N or len({r["unit"] for r in ledger}) != EXPECTED_N:
        raise ValueError("production ledger is not exact 515")
    eddy: dict[str, str] = {}
    for audit in INPUT_AUDITS:
        for row in read_csv(audit):
            unit = row["unit"]
            path = row.get("eddy_path", "")
            if not path:
                continue
            previous = eddy.get(unit)
            if previous is not None and Path(previous).resolve() != Path(path).resolve():
                raise ValueError(f"conflicting Eddy derivative: {unit}")
            eddy[unit] = path
    units = {row["unit"] for row in ledger}
    if set(eddy).intersection(units) != units:
        raise ValueError(
            "missing production Eddy mappings: "
            + ",".join(sorted(units - set(eddy)))
        )
    return sorted(ledger, key=lambda row: row["unit"]), eddy


def write_summary(
    ledger: list[dict[str, str]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    flagged = [
        row for row in rows if row["status"] == "FAIL_ZERO_EDDY_VOLUMES"
    ]
    errors = [row for row in rows if row["status"] == "ERROR"]
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_eddy_volume_integrity_audit"
        ),
        "status": (
            "ERROR"
            if errors
            else (
                "RECOVERY_REQUIRED"
                if len(rows) == EXPECTED_N and flagged
                else (
                    "PASS"
                    if len(rows) == EXPECTED_N
                    else "IN_PROGRESS"
                )
            )
        ),
        "updated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "zero_volume_maximum_threshold": ZERO_MAXIMUM_THRESHOLD,
        "target_n": len(ledger),
        "audited_n": len(rows),
        "pass_n": sum(row["status"] == "PASS" for row in rows),
        "flagged_n": len(flagged),
        "error_n": len(errors),
        "flagged_units": [row["unit"] for row in flagged],
        "raw_intact_flagged_units": [
            row["unit"]
            for row in flagged
            if int(row["raw_zero_volume_n"]) == 0
        ],
        "rows": str(ROWS.resolve()),
        "implementation": str(Path(__file__).resolve()),
    }
    atomic_json(SUMMARY, summary)
    return summary


def scan_one(
    source: Mapping[str, str], eddy_source: str
) -> dict[str, Any]:
    unit = str(source["unit"])
    eddy_path = Path(eddy_source).resolve()
    raw_path = (DERIV / "mif_dwi" / f"{unit}.mif").resolve()
    row: dict[str, Any] = {
        "unit": unit,
        "lane": source["lane"],
        "status": "RUNNING",
        "eddy_path": str(eddy_path),
        "eddy_size_bytes": (
            eddy_path.stat().st_size if eddy_path.is_file() else ""
        ),
        "raw_path": str(raw_path),
        "raw_checked": False,
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "error": "",
    }
    try:
        if not eddy_path.is_file() or not raw_path.is_file():
            raise FileNotFoundError(f"{eddy_path} or {raw_path}")
        eddy_metrics = metrics(eddy_path)
        row.update(
            {
                f"eddy_{key}": value
                for key, value in eddy_metrics.items()
            }
        )
        if eddy_metrics["zero_volume_n"]:
            raw_metrics = metrics(raw_path)
            row.update(
                {
                    f"raw_{key}": value
                    for key, value in raw_metrics.items()
                }
            )
            row["raw_checked"] = True
            row["status"] = "FAIL_ZERO_EDDY_VOLUMES"
        else:
            row["status"] = "PASS"
    except Exception as exc:
        row["status"] = "ERROR"
        row["error"] = f"{type(exc).__name__}:{exc}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be in 1..4")
    ledger, eddy = inputs()
    completed = {
        row["unit"]: row for row in read_csv(ROWS)
    } if ROWS.is_file() else {}
    pending = [
        source
        for source in ledger
        if not (
            source["unit"] in completed
            and completed[source["unit"]]["status"]
            in {"PASS", "FAIL_ZERO_EDDY_VOLUMES"}
        )
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(scan_one, source, eddy[source["unit"]]): source
            for source in pending
        }
        for future in as_completed(futures):
            source = futures[future]
            unit = source["unit"]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "unit": unit,
                    "lane": source["lane"],
                    "status": "ERROR",
                    "diagnosis_labels_used": False,
                    "outcomes_used": False,
                    "connectome_density_used": False,
                    "error": f"{type(exc).__name__}:{exc}",
                }
            completed[unit] = row
            ordered = [completed[key] for key in sorted(completed)]
            atomic_csv(ROWS, ordered)
            summary = write_summary(ledger, ordered)
            print(
                f"[{utc_now()}] eddy_volume_integrity "
                f"{len(completed)}/{EXPECTED_N} {unit} {row['status']} "
                f"zero={row.get('eddy_zero_volume_n')} "
                f"flagged_total={summary['flagged_n']}",
                flush=True,
            )
    summary = write_summary(
        ledger, [completed[key] for key in sorted(completed)]
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["status"] == "ERROR" else 0


if __name__ == "__main__":
    raise SystemExit(main())
