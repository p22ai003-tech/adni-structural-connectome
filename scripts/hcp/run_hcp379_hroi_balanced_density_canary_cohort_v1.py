#!/usr/bin/env python3
"""Aggregate corrected-HROI balanced-density evidence across 15 canaries.

The runner reuses valid per-unit corrected-HROI balanced-density attempts and
executes only missing per-unit endpoint reassignment jobs.  It never generates
tractography.  The smallest density-qualified total is reported only when all
15 primary/independent exact-10M pairs are present and every canary reaches
the fixed 0.60 density threshold.

Even a complete passing count-only result does not select a production recipe
or authorize scale-up; the chosen balanced construction must still generate
and pass all nine matrices jointly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
SINGLE_SOURCE = (
    EXP
    / "scripts/hcp/"
    "build_hcp379_hroi_balanced_seed_density_canary_v1.py"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_hroi_balanced_density_canary_cohort_v1/attempts"
)
EXPECTED_N = 15
EXPECTED_TOTALS = (6_000_000, 10_000_000, 15_000_000, 20_000_000)
DENSITY_TARGET = 0.60


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SINGLE = load_module(SINGLE_SOURCE, "hcp379_hroi_balanced_cohort_single")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def attempt_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def units() -> list[str]:
    values = sorted(SINGLE.ATLAS.BASE.BASE.load_units())
    if len(values) != EXPECTED_N or len(set(values)) != EXPECTED_N:
        raise ValueError("Phase-B canary unit set differs")
    return values


def current_single_source_record() -> dict[str, Any]:
    return SINGLE.BALANCED.file_record(SINGLE_SOURCE)


def valid_unit_summary(unit: str) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(SINGLE.ATTEMPTS.glob("*/summary.json")):
        try:
            value = SINGLE.load_json(path)
        except Exception:
            continue
        if value.get("unit") == unit:
            candidates.append((path.resolve(), value))
    if not candidates:
        return None
    path, value = candidates[-1]
    if (
        value.get("record_type")
        != "diagnosis_blind_hcp379_hroi_balanced_density_canary_summary"
        or value.get("status") != "PASS_HROI_BALANCED_DENSITY_CANARY"
        or value.get("diagnosis_labels_used") is not False
        or value.get("non_overwriting") is not True
        or value.get("tractography_reused_not_regenerated") is not True
        or value.get("sift2_or_scalar_matrices_generated") is not False
        or value.get("scale_up_authorized") is not False
        or value.get("cohort_uniform_recipe_selected") is not False
        or value.get("implementation") != current_single_source_record()
        or set(value.get("levels", {}))
        != {str(total) for total in EXPECTED_TOTALS}
    ):
        return None
    for seed in ("primary", "independent"):
        row = value.get("seeds", {}).get(seed, {})
        for name in (
            "tracks",
            "metadata",
            "hroi_count_10m",
            "hroi_assignments_10m",
        ):
            record = row.get(name)
            if not isinstance(record, Mapping):
                return None
            artifact = Path(str(record.get("path", ""))).resolve()
            # Size/path are cheap live checks here. The independent validator
            # rehashes every referenced artifact before accepting the cohort.
            if (
                not artifact.is_file()
                or artifact.is_symlink()
                or record.get("path") != str(artifact)
                or record.get("size_bytes") != artifact.stat().st_size
                or len(str(record.get("sha256", ""))) != 64
            ):
                return None
    return path, value


def readiness(unit_list: list[str]) -> list[dict[str, Any]]:
    rows = []
    for unit in unit_list:
        pair = SINGLE.validate_phase_pair(unit)
        existing = valid_unit_summary(unit)
        rows.append(
            {
                "unit": unit,
                "pair_ready": pair["ready"],
                "pair_missing": pair["missing"],
                "existing_valid_summary": (
                    str(existing[0]) if existing is not None else None
                ),
            }
        )
    return rows


def run_unit(unit: str, threads: int, log_root: Path) -> dict[str, Any]:
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{unit}.stdout.json"
    stderr_path = log_root / f"{unit}.stderr.log"
    result = subprocess.run(
        [
            str(PYTHON),
            str(SINGLE_SOURCE),
            "--unit",
            unit,
            "--threads",
            str(threads),
            "--execute",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"{unit}: corrected-HROI density runner exited "
            f"{result.returncode}; see {stderr_path}"
        )
    output = json.loads(result.stdout)
    if (
        output.get("status") != "PASS_HROI_BALANCED_DENSITY_CANARY"
        or output.get("unit") != unit
    ):
        raise ValueError(f"{unit}: single-canary terminal output differs")
    validated = valid_unit_summary(unit)
    if validated is None:
        raise ValueError(f"{unit}: valid terminal summary is absent")
    return {
        "unit": unit,
        "summary": str(validated[0]),
        "runner_stdout": str(stdout_path),
        "runner_stderr": str(stderr_path),
    }


def build_unit_row(
    unit: str, path: Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    levels = {
        str(total): {
            "edge_density": float(
                value["levels"][str(total)]["edge_density"]
            ),
            "density_target_0_60_met": bool(
                value["levels"][str(total)][
                    "density_target_0_60_met"
                ]
            ),
            "connected_nodes": int(
                value["levels"][str(total)]["connected_nodes"]
            ),
            "primary_endpoint_assignment_fraction": float(
                value["levels"][str(total)][
                    "primary_endpoint_assignment_fraction"
                ]
            ),
            "independent_endpoint_assignment_fraction": float(
                value["levels"][str(total)][
                    "independent_endpoint_assignment_fraction"
                ]
            ),
        }
        for total in EXPECTED_TOTALS
    }
    return {
        "unit": unit,
        "status": value["status"],
        "summary": SINGLE.BALANCED.file_record(path),
        "levels": levels,
    }


def self_test() -> dict[str, Any]:
    checks = {
        "exact_canary_n": len(units()) == EXPECTED_N,
        "balanced_totals": EXPECTED_TOTALS
        == (6_000_000, 10_000_000, 15_000_000, 20_000_000),
        "density_target": DENSITY_TARGET == 0.60,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads-per-unit", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be in 1..4")
    if not 1 <= args.threads_per_unit <= 16:
        parser.error("--threads-per-unit must be in 1..16")
    if args.workers * args.threads_per_unit > 32:
        parser.error("worker/thread product must not exceed 32")
    if args.self_test:
        value = self_test()
        print(json.dumps(value, sort_keys=True))
        return 0 if value["status"] == "PASS" else 1

    unit_list = units()
    live = readiness(unit_list)
    ready = [row["unit"] for row in live if row["pair_ready"]]
    waiting = [row["unit"] for row in live if not row["pair_ready"]]
    reusable = [
        row["unit"]
        for row in live
        if row["pair_ready"] and row["existing_valid_summary"] is not None
    ]
    runnable = sorted(set(ready) - set(reusable))
    preflight = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_balanced_density_"
            "canary_cohort_preflight"
        ),
        "generated_utc": utc_now(),
        "status": (
            "READY_ALL_15_EXACT_PAIRS"
            if not waiting
            else "WAITING_FOR_EXACT_PHASE_B_PAIRS"
        ),
        "diagnosis_labels_used": False,
        "expected_unit_n": EXPECTED_N,
        "pair_ready_n": len(ready),
        "waiting_n": len(waiting),
        "reusable_valid_n": len(reusable),
        "runnable_n": len(runnable),
        "waiting_units": waiting,
        "readiness": live,
        "tractography_generated": False,
        "cohort_recipe_selected": False,
        "scale_up_authorized": False,
    }
    if not args.execute:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    if waiting and not args.allow_partial:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 2
    if not ready:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 2

    attempt = (
        ATTEMPTS
        / f"{attempt_stamp()}-hroi-balanced-density-canary-cohort"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_json(attempt / "preflight.json", preflight)
    executed: dict[str, Any] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_by_unit = {
            pool.submit(
                run_unit,
                unit,
                args.threads_per_unit,
                attempt / "unit_logs",
            ): unit
            for unit in runnable
        }
        for future in as_completed(future_by_unit):
            unit = future_by_unit[future]
            try:
                executed[unit] = future.result()
            except Exception as exc:
                failures[unit] = f"{type(exc).__name__}:{exc}"

    rows = []
    for unit in ready:
        validated = valid_unit_summary(unit)
        if validated is None:
            failures.setdefault(unit, "valid terminal summary is absent")
            continue
        rows.append(build_unit_row(unit, *validated))
    rows.sort(key=lambda row: row["unit"])
    complete = len(rows) == EXPECTED_N and not failures
    qualified = [
        total
        for total in EXPECTED_TOTALS
        if complete
        and all(
            row["levels"][str(total)]["density_target_0_60_met"]
            for row in rows
        )
    ]
    density_ranges = {
        str(total): {
            "minimum": min(
                row["levels"][str(total)]["edge_density"] for row in rows
            )
            if rows
            else None,
            "maximum": max(
                row["levels"][str(total)]["edge_density"] for row in rows
            )
            if rows
            else None,
        }
        for total in EXPECTED_TOTALS
    }
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_balanced_density_"
            "canary_cohort_summary"
        ),
        "generated_utc": utc_now(),
        "status": (
            "PASS_COMPLETE_COUNT_ONLY_DENSITY_EVIDENCE"
            if complete
            else "PARTIAL_COUNT_ONLY_DENSITY_EVIDENCE"
            if not failures
            else "FAIL_COUNT_ONLY_DENSITY_EVIDENCE"
        ),
        "diagnosis_labels_used": False,
        "expected_unit_n": EXPECTED_N,
        "processed_unit_n": len(rows),
        "failure_n": len(failures),
        "failures": failures,
        "reused_unit_n": len(reusable),
        "executed_unit_n": len(executed),
        "density_target": DENSITY_TARGET,
        "density_ranges": density_ranges,
        "density_qualified_balanced_total_counts": qualified,
        "smallest_density_qualified_balanced_total_count": (
            min(qualified) if qualified else None
        ),
        "count_only_projection": True,
        "tractography_reused_not_regenerated": True,
        "sift2_or_scalar_matrices_generated": False,
        "cohort_uniform_recipe_selected": False,
        "scale_up_authorized": False,
        "requires_joint_all_nine_matrix_stability": True,
        "requires_uniform_530_subject_construction": True,
        "units": rows,
        "preflight": SINGLE.BALANCED.file_record(
            attempt / "preflight.json"
        ),
        "single_runner": current_single_source_record(),
        "cohort_runner": SINGLE.BALANCED.file_record(Path(__file__)),
    }
    atomic_json(attempt / "summary.json", summary)
    print(
        json.dumps(
            {
                "attempt_root": str(attempt),
                "status": summary["status"],
                "processed_unit_n": len(rows),
                "failure_n": len(failures),
                "smallest_density_qualified_balanced_total_count": (
                    summary[
                        "smallest_density_qualified_balanced_total_count"
                    ]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
