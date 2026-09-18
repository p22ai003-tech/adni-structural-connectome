#!/usr/bin/env python3
"""Independently validate the HCP379-v2 FOD-order compatibility audit."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
MRINFO = Path("/home/ec2-user/mrtrix3/bin/mrinfo")
COHORT = (
    EXP
    / "research_audit/outputs/historical_source_preflight_proxy_v1.csv"
)
OUTPUT_ROOT = (
    EXP
    / "research_audit/outputs/hcp379_fod_order_compatibility_v1"
)
TABLE = OUTPUT_ROOT / "fod_order_compatibility.csv"
AUDIT = OUTPUT_ROOT / "audit.json"
VALIDATION = OUTPUT_ROOT / "validation.json"
EDDY_ROOT = Path("/data/derivatives/eddy")
EXPECTED_N = 530


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def lmax_for(n: int) -> int:
    if n >= 28:
        return 6
    if n >= 15:
        return 4
    if n >= 6:
        return 2
    return 0


def independent_direction_count(path: Path) -> int:
    result = subprocess.run(
        [str(MRINFO), str(path), "-dwgrad"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip())
    gradients: list[tuple[float, float, float, float]] = []
    for raw in result.stdout.splitlines():
        fields = raw.split()
        if len(fields) != 4:
            continue
        try:
            gradients.append(tuple(float(value) for value in fields))
        except ValueError:
            continue
    if not gradients:
        raise ValueError("empty gradient table")
    nonzero = sorted({row[3] for row in gradients if row[3] > 50.0})
    if not nonzero:
        raise ValueError("no nonzero shell")
    selected = min(nonzero, key=lambda value: (abs(value - 1000.0), value))
    if abs(selected - 1000.0) > 100.0:
        raise ValueError("nearest shell outside tolerance")
    canonical: set[tuple[float, float, float]] = set()
    for x, y, z, bvalue in gradients:
        if abs(bvalue - selected) > 50.0:
            continue
        norm = math.sqrt(x * x + y * y + z * z)
        if norm <= 0.05:
            raise ValueError("near-zero nonzero-shell vector")
        vector = [x / norm, y / norm, z / norm]
        for value in vector:
            if abs(value) > 1e-8:
                if value < 0:
                    vector = [-component for component in vector]
                break
        canonical.add(tuple(round(value, 4) for value in vector))
    if not canonical:
        raise ValueError("no selected-shell directions")
    return len(canonical)


def inspect_source(source: dict[str, str]) -> tuple[str, int, str]:
    unit = f"{source['subject_id']}_I{source['dti_image_id']}"
    eddy = EDDY_ROOT / f"{unit}_preproc.mif"
    if eddy.exists():
        try:
            return unit, independent_direction_count(eddy), "EDDY_HEADER"
        except Exception:
            pass
    historical = Path(source["historical_mif_path"])
    return (
        unit,
        independent_direction_count(historical),
        "HISTORICAL_MIF_GRADIENT_FALLBACK",
    )


def check(
    rows: list[dict[str, Any]],
    name: str,
    passed: bool,
    evidence: str,
) -> None:
    rows.append(
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }
    )


def main() -> int:
    source_rows = load_csv(COHORT)
    table_rows = load_csv(TABLE)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    expected_units = {
        f"{row['subject_id']}_I{row['dti_image_id']}"
        for row in source_rows
    }
    table_by_unit = {row["unit"]: row for row in table_rows}
    check(
        checks,
        "exact_530_identity",
        len(source_rows) == EXPECTED_N
        and len(table_rows) == EXPECTED_N
        and len(table_by_unit) == EXPECTED_N
        and set(table_by_unit) == expected_units,
        f"source={len(source_rows)} table={len(table_rows)} "
        f"unique={len(table_by_unit)}",
    )

    independently: dict[str, tuple[int, str]] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(inspect_source, row): row
            for row in source_rows
        }
        for future in as_completed(futures):
            source = futures[future]
            unit = (
                f"{source['subject_id']}_I{source['dti_image_id']}"
            )
            try:
                result_unit, count, route = future.result()
                independently[result_unit] = (count, route)
            except Exception as exc:
                errors[unit] = f"{type(exc).__name__}:{exc}"
    check(
        checks,
        "independent_header_replay_complete",
        len(independently) == EXPECTED_N and not errors,
        f"replayed={len(independently)} errors={len(errors)}",
    )

    mismatch: list[str] = []
    for unit, (count, route) in independently.items():
        row = table_by_unit[unit]
        if (
            int(row[
                "selected_shell_unique_antipodal_direction_n"
            ])
            != count
            or row["source_route"] != route
            or int(row["recommended_lmax"]) != lmax_for(count)
            or (
                row["fixed_lmax6_compatible"].strip().lower()
                == "true"
            )
            != (count >= 28)
        ):
            mismatch.append(unit)
    check(
        checks,
        "rowwise_direction_and_order_replay",
        not mismatch,
        f"mismatch_n={len(mismatch)} "
        f"first={','.join(mismatch[:5])}",
    )

    direction_counts = Counter(
        count for count, _route in independently.values()
    )
    route_counts = Counter(
        route for _count, route in independently.values()
    )
    lmax_counts = Counter(lmax_for(count) for count in direction_counts.elements())
    hold_units = sorted(
        unit
        for unit, (count, _route) in independently.items()
        if count < 28
    )
    table_hold_units = sorted(
        row["unit"]
        for row in table_rows
        if row["status"]
        == "HOLD_ACQUISITION_AWARE_LMAX_REQUIRED"
    )
    check(
        checks,
        "cohort_distribution",
        direction_counts
        == Counter(
            {6: 2, 15: 6, 16: 1, 19: 1, 30: 102, 32: 96, 35: 2, 41: 35, 48: 285}
        )
        and route_counts
        == Counter(
            {
                "EDDY_HEADER": 475,
                "HISTORICAL_MIF_GRADIENT_FALLBACK": 55,
            }
        )
        and lmax_counts == Counter({2: 2, 4: 8, 6: 520}),
        f"directions={dict(sorted(direction_counts.items()))} "
        f"routes={dict(sorted(route_counts.items()))} "
        f"lmax={dict(sorted(lmax_counts.items()))}",
    )
    check(
        checks,
        "exact_low_direction_holds",
        len(hold_units) == 10
        and hold_units == table_hold_units
        and hold_units
        == sorted(
            row["unit"]
            for row in audit["acquisition_aware_hold_units"]
        ),
        f"hold_n={len(hold_units)} units={','.join(hold_units)}",
    )
    check(
        checks,
        "no_clinical_or_connectome_selection",
        audit.get("diagnosis_labels_used") is False
        and audit.get("outcomes_used") is False
        and audit.get("connectome_density_used") is False
        and audit.get("image_data_loaded") is False
        and all(
            row["diagnosis_labels_used"].lower() == "false"
            and row["outcomes_used"].lower() == "false"
            for row in table_rows
        ),
        "selection uses only DWI header gradients and fixed degrees-of-freedom rules",
    )
    check(
        checks,
        "fail_closed_production_decision",
        audit.get("status")
        == "PASS_AUDIT_COMPLETE_WITH_ACQUISITION_AWARE_HOLDS"
        and audit.get("fixed_lmax6_compatible_n") == 520
        and audit.get("acquisition_aware_hold_n") == 10
        and "Do not promote or compact" in str(
            audit.get("production_decision")
        ),
        str(audit.get("production_decision")),
    )

    passed = sum(row["status"] == "PASS" for row in checks)
    validation = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_fod_order_compatibility_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "independent_replay_error_by_unit": errors,
    }
    VALIDATION.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
