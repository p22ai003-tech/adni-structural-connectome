#!/usr/bin/env python3
"""Audit corrected-route inputs for the 216 legacy/tensor HCP379 units.

The target set is defined solely by frozen technical inventory lanes:
214 legacy-keep units plus two tensor-repair units.  The audit includes the
two currently density-qualified legacy units so they can be regenerated
without delay if route concordance fails.  It is read-only and does not use
diagnosis or outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
AUDIT_SOURCE = EXP / "scripts/hcp/audit_hcp379_scaleup_inputs_v2.py"
RECOVERY_PLAN = (
    EXP
    / "research_audit/outputs/hcp379_density_recovery_plan_v3/"
    "hcp379_density_recovery_subjects.csv"
)
OUTPUT_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_legacy_density_recovery_input_audit_v3"
)
TARGET_LANES = {"legacy_keep_214", "tensor_repair_2"}
EXPECTED_N = 216
EXPECTED_LOW_N = 214
READY_ROUTES = {
    "READY_FOR_CORRECTED_PRETRACT_FROM_EDDY",
    "READY_AFTER_GRADIENT_HEADER_REPAIR",
}


def load_audit_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "audit_hcp379_scaleup_inputs_v2_for_legacy_density_recovery",
        AUDIT_SOURCE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(AUDIT_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_audit_module()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        newline="",
        encoding="utf-8",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("--workers must be in 1..32")

    plan_rows = read_csv(RECOVERY_PLAN)
    if len(plan_rows) != 530:
        raise ValueError(f"density recovery plan has {len(plan_rows)} rows")
    targets = {
        row["unit"]: row
        for row in plan_rows
        if row["inventory_lane"] in TARGET_LANES
    }
    if len(targets) != EXPECTED_N:
        raise ValueError(f"legacy/tensor target set is {len(targets)}, not 216")
    low_n = sum(
        row["density_group"] != "D0_THRESHOLD_QUALIFIED"
        for row in targets.values()
    )
    if low_n != EXPECTED_LOW_N:
        raise ValueError(f"legacy/tensor low-density set is {low_n}, not 214")

    acquisition = AUDIT.acquisition_by_unit()
    missing = sorted(set(targets) - set(acquisition))
    if missing:
        raise ValueError(f"acquisition manifest lacks units: {missing}")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                AUDIT.audit_unit,
                unit,
                recommended_action="LEGACY_DENSITY_RECOVERY_OR_D0_FALLBACK",
                acquisition=acquisition[unit],
            ): unit
            for unit in sorted(targets)
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            unit = str(result["unit"])
            plan = targets[unit]
            result.update(
                {
                    "inventory_lane": plan["inventory_lane"],
                    "density_group": plan["density_group"],
                    "current_density": plan["current_density"],
                    "current_density_threshold_pass": plan[
                        "current_density_threshold_pass"
                    ],
                    "selected_by_diagnosis_or_outcome": False,
                }
            )
            results.append(result)
            print(
                f"[{utc_now()}] {index}/{EXPECTED_N} {unit} "
                f"{result['density_group']} {result['route']}",
                flush=True,
            )
    results.sort(key=lambda row: str(row["unit"]))
    route_counts = Counter(str(row["route"]) for row in results)
    density_counts = Counter(str(row["density_group"]) for row in results)
    route_density: dict[str, Counter[str]] = defaultdict(Counter)
    for row in results:
        route_density[str(row["route"])][str(row["density_group"])] += 1
    ready_n = sum(str(row["route"]) in READY_ROUTES for row in results)
    low_ready_n = sum(
        str(row["route"]) in READY_ROUTES
        and str(row["density_group"]) != "D0_THRESHOLD_QUALIFIED"
        for row in results
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_root / "legacy_density_recovery_input_audit.csv"
    atomic_csv(output_csv, results)
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_legacy_density_recovery_input_audit"
        ),
        "status": (
            "PASS"
            if len(results) == EXPECTED_N and ready_n == EXPECTED_N
            else "FAIL"
        ),
        "generated_utc": utc_now(),
        "diagnosis_or_outcomes_used": False,
        "target_n": EXPECTED_N,
        "audited_n": len(results),
        "below_density_threshold_n": EXPECTED_LOW_N,
        "density_qualified_fallback_n": EXPECTED_N - EXPECTED_LOW_N,
        "ready_n": ready_n,
        "below_threshold_ready_n": low_ready_n,
        "route_counts": dict(sorted(route_counts.items())),
        "density_group_counts": dict(sorted(density_counts.items())),
        "counts_by_route_and_density_group": {
            route: dict(sorted(values.items()))
            for route, values in sorted(route_density.items())
        },
        "routing_policy": {
            "ready_from_eddy": (
                "reuse the locked Eddy derivative; rebuild corrected "
                "T1-to-DWI anatomy, HCP379, FOD/tensors and tractography"
            ),
            "gradient_header_only": (
                "repair gradients from locked bvec/bval sidecars before the "
                "same corrected route"
            ),
            "not_ready": (
                "repair DWI preprocessing or exact-T1 anatomy first; do not "
                "promote the historical low-density matrix"
            ),
            "D0_fallback": (
                "the two D0 legacy units use these inputs only if route "
                "concordance or semantic QC rejects retention"
            ),
        },
        "records": {
            "input_audit": file_record(output_csv),
            "density_recovery_plan": file_record(RECOVERY_PLAN),
            "base_audit_source": file_record(AUDIT_SOURCE),
            "builder": file_record(Path(__file__)),
        },
    }
    atomic_json(
        args.output_root
        / "legacy_density_recovery_input_audit_summary.json",
        summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
