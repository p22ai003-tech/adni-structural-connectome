#!/usr/bin/env python3
"""Audit whether every HCP379-v2 DWI can support the frozen lmax=6 FOD fit.

This is a diagnosis-blind, header-only audit.  It prefers the corrected Eddy
image used by Recovery4 and falls back to the historical MIF only when the
Eddy header has no usable diffusion gradient table.  No image data, disease
labels, outcomes, tractography, or connectome density enter the decision.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXP = Path("/home/ec2-user/exp")
MRINFO = Path("/home/ec2-user/mrtrix3/bin/mrinfo")
COHORT = (
    EXP
    / "research_audit/outputs/historical_source_preflight_proxy_v1.csv"
)
LEDGER = (
    EXP
    / "research_audit/outputs/"
    "hcp379_pretract_failure_recovery_ledger_v1/ledger.csv"
)
EDDY_ROOT = Path("/data/derivatives/eddy")
OUTPUT_ROOT = (
    EXP
    / "research_audit/outputs/hcp379_fod_order_compatibility_v1"
)
OUTPUT_CSV = OUTPUT_ROOT / "fod_order_compatibility.csv"
OUTPUT_JSON = OUTPUT_ROOT / "audit.json"

EXPECTED_N = 530
TARGET_SHELL = 1000.0
SHELL_TOLERANCE = 100.0
BZERO_THRESHOLD = 50.0
CURRENT_LMAX = 6
ROUND_DIGITS = 4
MAX_WORKERS = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_mrinfo(path: Path, option: str) -> str:
    result = subprocess.run(
        [str(MRINFO), str(path), option],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(
            f"mrinfo {option} failed for {path}: {result.stderr.strip()}"
        )
    return result.stdout


def parse_dwgrad(text: str) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    for raw in text.splitlines():
        fields = raw.strip().split()
        if len(fields) != 4:
            continue
        try:
            values = tuple(float(value) for value in fields)
        except ValueError:
            continue
        rows.append(values)  # type: ignore[arg-type]
    if not rows:
        raise ValueError("diffusion gradient table is empty")
    return rows


def canonical_direction(
    x: float, y: float, z: float
) -> tuple[float, float, float]:
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 0.05:
        raise ValueError("nonzero-shell gradient has near-zero norm")
    vector = [x / norm, y / norm, z / norm]
    for value in vector:
        if abs(value) > 1e-8:
            if value < 0:
                vector = [-component for component in vector]
            break
    return tuple(round(component, ROUND_DIGITS) for component in vector)


def selected_shell_rows(
    gradients: Iterable[tuple[float, float, float, float]]
) -> tuple[float, list[tuple[float, float, float, float]]]:
    rows = list(gradients)
    nonzero_bvalues = sorted(
        {float(row[3]) for row in rows if float(row[3]) > BZERO_THRESHOLD}
    )
    if not nonzero_bvalues:
        raise ValueError("no nonzero diffusion shell")
    selected = min(
        nonzero_bvalues,
        key=lambda value: (abs(value - TARGET_SHELL), value),
    )
    if abs(selected - TARGET_SHELL) > SHELL_TOLERANCE:
        raise ValueError(
            f"nearest shell {selected:g} is outside target tolerance"
        )
    # MRtrix shell clustering can leave small floating variation in b-values.
    shell_rows = [
        row for row in rows if abs(float(row[3]) - selected) <= 50.0
    ]
    if not shell_rows:
        raise ValueError("selected shell has no gradient rows")
    return selected, shell_rows


def required_coefficients(lmax: int) -> int:
    if lmax < 0 or lmax % 2:
        raise ValueError(f"lmax must be nonnegative and even: {lmax}")
    return (lmax + 1) * (lmax + 2) // 2


def supported_lmax(unique_direction_n: int) -> int:
    if unique_direction_n >= required_coefficients(6):
        return 6
    if unique_direction_n >= required_coefficients(4):
        return 4
    if unique_direction_n >= required_coefficients(2):
        return 2
    return 0


def inspect_source(path: Path) -> dict[str, Any]:
    gradients = parse_dwgrad(run_mrinfo(path, "-dwgrad"))
    selected_b, shell_rows = selected_shell_rows(gradients)
    directions = {
        canonical_direction(row[0], row[1], row[2])
        for row in shell_rows
    }
    return {
        "selected_shell_bvalue": selected_b,
        "selected_shell_volume_n": len(shell_rows),
        "selected_shell_unique_antipodal_direction_n": len(directions),
    }


def inspect_unit(source_row: dict[str, str]) -> dict[str, Any]:
    unit = (
        f"{source_row['subject_id']}_I{source_row['dti_image_id']}"
    )
    eddy = EDDY_ROOT / f"{unit}_preproc.mif"
    historical = Path(source_row["historical_mif_path"])
    errors: list[str] = []
    selected_path: Path | None = None
    source_route = ""
    result: dict[str, Any] | None = None
    if eddy.exists():
        try:
            result = inspect_source(eddy)
            selected_path = eddy
            source_route = "EDDY_HEADER"
        except Exception as exc:
            errors.append(f"eddy:{type(exc).__name__}:{exc}")
    if result is None:
        try:
            result = inspect_source(historical)
            selected_path = historical
            source_route = "HISTORICAL_MIF_GRADIENT_FALLBACK"
        except Exception as exc:
            errors.append(f"historical:{type(exc).__name__}:{exc}")
    if result is None or selected_path is None:
        return {
            "unit": unit,
            "status": "FAIL_NO_USABLE_GRADIENT_TABLE",
            "source_route": "NONE",
            "source_path": "",
            "source_error": " | ".join(errors),
            "selected_shell_bvalue": "",
            "selected_shell_volume_n": "",
            "selected_shell_unique_antipodal_direction_n": "",
            "current_lmax": CURRENT_LMAX,
            "required_even_sh_coefficient_n": required_coefficients(
                CURRENT_LMAX
            ),
            "maximum_supported_lmax": "",
            "recommended_lmax": "",
            "fixed_lmax6_compatible": False,
            "recovery_action": "FAIL_CLOSED_SOURCE_GRADIENT_AUDIT",
            "diagnosis_labels_used": False,
            "outcomes_used": False,
        }
    unique_n = int(
        result["selected_shell_unique_antipodal_direction_n"]
    )
    max_lmax = supported_lmax(unique_n)
    compatible = max_lmax >= CURRENT_LMAX
    return {
        "unit": unit,
        "status": (
            "PASS_FIXED_LMAX6_COMPATIBLE"
            if compatible
            else "HOLD_ACQUISITION_AWARE_LMAX_REQUIRED"
        ),
        "source_route": source_route,
        "source_path": str(selected_path.resolve()),
        "source_error": " | ".join(errors),
        **result,
        "current_lmax": CURRENT_LMAX,
        "required_even_sh_coefficient_n": required_coefficients(
            CURRENT_LMAX
        ),
        "maximum_supported_lmax": max_lmax,
        "recommended_lmax": max_lmax,
        "fixed_lmax6_compatible": compatible,
        "recovery_action": (
            "CONTINUE_FROZEN_LMAX6"
            if compatible
            else f"VALIDATE_AND_RERUN_WITH_LMAX{max_lmax}"
        ),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "unit",
        "lane",
        "live_pretract_status",
        "status",
        "source_route",
        "source_path",
        "source_error",
        "selected_shell_bvalue",
        "selected_shell_volume_n",
        "selected_shell_unique_antipodal_direction_n",
        "current_lmax",
        "required_even_sh_coefficient_n",
        "maximum_supported_lmax",
        "recommended_lmax",
        "fixed_lmax6_compatible",
        "recovery_action",
        "diagnosis_labels_used",
        "outcomes_used",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if not MRINFO.exists() or not COHORT.exists() or not LEDGER.exists():
        raise FileNotFoundError("required HCP379 audit input is missing")
    source_rows = read_csv(COHORT)
    ledger_rows = read_csv(LEDGER)
    if len(source_rows) != EXPECTED_N:
        raise ValueError(f"expected {EXPECTED_N} source rows")
    ledger_by_unit = {row["unit"]: row for row in ledger_rows}
    if len(ledger_by_unit) != 515:
        raise ValueError("live production ledger must contain 515 units")

    audited: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(inspect_unit, row): row
            for row in source_rows
        }
        for future in as_completed(futures):
            audited.append(future.result())
    audited.sort(key=lambda row: str(row["unit"]))

    if len({str(row["unit"]) for row in audited}) != EXPECTED_N:
        raise ValueError("cohort unit identity is not exactly 530 unique units")
    for row in audited:
        live = ledger_by_unit.get(str(row["unit"]), {})
        row["lane"] = live.get("lane", "phase_b_canary")
        row["live_pretract_status"] = live.get(
            "status", "PASS_COMPACTED"
        )

    write_csv(OUTPUT_CSV, audited)
    errors = [
        row
        for row in audited
        if row["status"] == "FAIL_NO_USABLE_GRADIENT_TABLE"
    ]
    incompatible = [
        row
        for row in audited
        if row["status"] == "HOLD_ACQUISITION_AWARE_LMAX_REQUIRED"
    ]
    unique_direction_counts = Counter(
        int(row["selected_shell_unique_antipodal_direction_n"])
        for row in audited
        if row["selected_shell_unique_antipodal_direction_n"] != ""
    )
    recommended_counts = Counter(
        int(row["recommended_lmax"])
        for row in audited
        if row["recommended_lmax"] != ""
    )
    source_counts = Counter(str(row["source_route"]) for row in audited)
    live_counts = Counter(
        str(row["live_pretract_status"]) for row in audited
    )
    audit = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_fod_order_compatibility_audit",
        "status": (
            "PASS_AUDIT_COMPLETE_WITH_ACQUISITION_AWARE_HOLDS"
            if not errors and incompatible
            else (
                "PASS_ALL_FIXED_LMAX6_COMPATIBLE"
                if not errors
                else "FAIL_INCOMPLETE_SOURCE_EVIDENCE"
            )
        ),
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "image_data_loaded": False,
        "tractography_generated": False,
        "connectome_density_used": False,
        "subject_outputs_modified": False,
        "expected_unit_n": EXPECTED_N,
        "observed_unit_n": len(audited),
        "current_frozen_lmax": CURRENT_LMAX,
        "required_even_sh_coefficient_n": required_coefficients(
            CURRENT_LMAX
        ),
        "target_shell_s_per_mm2": TARGET_SHELL,
        "shell_tolerance_s_per_mm2": SHELL_TOLERANCE,
        "direction_policy": (
            "Unique unit directions are normalized, antipodally "
            f"canonicalized, and rounded to {ROUND_DIGITS} decimals. "
            "The maximum even lmax is the largest of 2, 4, or 6 whose "
            "even-SH coefficient count does not exceed that number."
        ),
        "source_route_counts": dict(sorted(source_counts.items())),
        "unique_direction_count_distribution": {
            str(key): value
            for key, value in sorted(unique_direction_counts.items())
        },
        "recommended_lmax_counts": {
            str(key): value
            for key, value in sorted(recommended_counts.items())
        },
        "live_pretract_status_counts": dict(
            sorted(live_counts.items())
        ),
        "fixed_lmax6_compatible_n": (
            len(audited) - len(incompatible) - len(errors)
        ),
        "acquisition_aware_hold_n": len(incompatible),
        "source_failure_n": len(errors),
        "acquisition_aware_hold_units": [
            {
                "unit": row["unit"],
                "unique_direction_n": row[
                    "selected_shell_unique_antipodal_direction_n"
                ],
                "recommended_lmax": row["recommended_lmax"],
                "live_pretract_status": row["live_pretract_status"],
                "lane": row["lane"],
            }
            for row in incompatible
        ],
        "production_decision": (
            "Do not promote or compact the listed low-direction units "
            "under fixed lmax=6. Validate a non-overwriting, "
            "acquisition-aware lmax counterfactual first. All other units "
            "remain on the frozen lmax=6 path."
        ),
        "records": {
            "implementation": file_record(Path(__file__)),
            "cohort_source": file_record(COHORT),
            "live_ledger": file_record(LEDGER),
            "subject_table": file_record(OUTPUT_CSV),
            "mrinfo": file_record(MRINFO),
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
