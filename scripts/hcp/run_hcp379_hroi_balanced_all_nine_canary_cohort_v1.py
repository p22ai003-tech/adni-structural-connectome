#!/usr/bin/env python3
"""Run and aggregate selected-HROI all-nine evidence for all 15 canaries.

Execution is fail-closed until the count-only HROI density cohort has selected
one smallest uniform balanced total and both exact 10M source seeds exist for
every canary.  Existing terminal per-unit attempts are reused only when their
implementation and selected density-summary bindings still match byte-for-byte.

A complete pass confirms a candidate construction for human/release review. It
does not itself authorize 530-subject tractography or a scientific release.
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
    "build_hcp379_hroi_balanced_all_nine_canary_v1.py"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_hroi_balanced_all_nine_canary_cohort_v1/attempts"
)
EXPECTED_N = 15


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SINGLE = load_module(SINGLE_SOURCE, "hcp379_hroi_all_nine_cohort_single")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def attempt_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


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
    _, by_unit = SINGLE.atlas_contract()
    values = sorted(by_unit)
    if len(values) != EXPECTED_N or len(set(values)) != EXPECTED_N:
        raise ValueError("fixed HROI canary unit set differs")
    return values


def density_summary() -> tuple[Path, dict[str, Any]] | None:
    path = SINGLE.latest_density_summary()
    if path is None:
        return None
    return path, SINGLE.valid_density_summary(path)


def valid_unit_summary(
    unit: str,
    *,
    density_path: Path,
    selected_total: int,
) -> tuple[Path, dict[str, Any]] | None:
    current_source = SINGLE.file_record(SINGLE_SOURCE)
    current_density = SINGLE.file_record(density_path)
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(SINGLE.ATTEMPTS.glob("*/summary.json")):
        try:
            value = load_json(path)
        except Exception:
            continue
        if value.get("unit") == unit:
            candidates.append((path.resolve(), value))
    for path, value in reversed(candidates):
        if (
            value.get("record_type")
            != "diagnosis_blind_hcp379_hroi_balanced_all_nine_canary_summary"
            or value.get("status")
            not in {
                "PASS_HROI_BALANCED_ALL_NINE_CANARY",
                "FAIL_HROI_BALANCED_ALL_NINE_CANARY",
            }
            or value.get("diagnosis_labels_used") is not False
            or value.get("non_overwriting") is not True
            or value.get("source_files_modified") is not False
            or value.get("probabilistic_tractography_generated") is not False
            or value.get("source_10m_tractograms_reused") is not True
            or value.get("deterministic_prefix_or_concatenation_only")
            is not True
            or value.get("cohort_recipe_selected") is not False
            or value.get("scale_up_authorized") is not False
            or value.get("implementation") != current_source
            or value.get("density_summary") != current_density
            or value.get("selected_balanced_total_streamlines")
            != selected_total
            or set(value.get("matrix_names", []))
            != set(SINGLE.MATRIX_NAMES)
            or set(value.get("views", {})) != set(SINGLE.VIEW_NAMES)
        ):
            continue
        artifact_ok = True
        for view in SINGLE.VIEW_NAMES:
            row = value["views"][view]
            for name in SINGLE.MATRIX_NAMES:
                record = row.get("matrices", {}).get(name, {})
                artifact = Path(str(record.get("path", ""))).resolve()
                if (
                    not artifact.is_file()
                    or artifact.is_symlink()
                    or record.get("path") != str(artifact)
                    or record.get("size_bytes") != artifact.stat().st_size
                    or len(str(record.get("sha256", ""))) != 64
                ):
                    artifact_ok = False
                    break
            if not artifact_ok:
                break
        if artifact_ok:
            return path, value
    return None


def readiness(
    unit_list: list[str],
    density_path: Path | None,
    selected_total: int | None,
) -> list[dict[str, Any]]:
    rows = []
    for unit in unit_list:
        pair = SINGLE.phase_readiness(unit)
        existing = (
            valid_unit_summary(
                unit,
                density_path=density_path,
                selected_total=selected_total,
            )
            if density_path is not None and selected_total is not None
            else None
        )
        rows.append(
            {
                "unit": unit,
                "phase_pair_ready": pair["ready"],
                "phase_pair_missing": pair["missing"],
                "existing_terminal_summary": (
                    str(existing[0]) if existing is not None else None
                ),
                "existing_terminal_status": (
                    existing[1]["status"] if existing is not None else None
                ),
            }
        )
    return rows


def run_unit(
    unit: str,
    *,
    density_path: Path,
    selected_total: int,
    threads: int,
    log_root: Path,
) -> dict[str, Any]:
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{unit}.stdout.json"
    stderr_path = log_root / f"{unit}.stderr.log"
    completed = subprocess.run(
        [
            str(PYTHON),
            str(SINGLE_SOURCE),
            "--unit",
            unit,
            "--density-summary",
            str(density_path),
            "--balanced-total",
            str(selected_total),
            "--threads",
            str(threads),
            "--execute",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            f"{unit}: runner exited {completed.returncode}; "
            f"see {stderr_path}"
        )
    try:
        terminal = json.loads(completed.stdout)
    except Exception as exc:
        raise ValueError(f"{unit}: runner stdout is not JSON") from exc
    if terminal.get("unit") != unit or terminal.get("status") not in {
        "PASS_HROI_BALANCED_ALL_NINE_CANARY",
        "FAIL_HROI_BALANCED_ALL_NINE_CANARY",
    }:
        raise ValueError(f"{unit}: runner terminal contract differs")
    validated = valid_unit_summary(
        unit,
        density_path=density_path,
        selected_total=selected_total,
    )
    if validated is None:
        raise ValueError(f"{unit}: terminal summary cannot be rebound")
    return {
        "unit": unit,
        "status": validated[1]["status"],
        "summary": str(validated[0]),
        "runner_stdout": str(stdout_path),
        "runner_stderr": str(stderr_path),
    }


def unit_row(
    unit: str,
    path: Path,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    balanced = value["views"]["balanced"]
    return {
        "unit": unit,
        "status": value["status"],
        "selected_balanced_total_streamlines": value[
            "selected_balanced_total_streamlines"
        ],
        "selected_per_seed_prefix_streamlines": value[
            "selected_per_seed_prefix_streamlines"
        ],
        "combined_edge_density": value["combined_edge_density"],
        "combined_density_target_met": value[
            "combined_density_target_met"
        ],
        "all_views_technical_qc_pass": value[
            "all_views_technical_qc_pass"
        ],
        "seed_reliability_status": value["seed_reliability"]["status"],
        "balanced_connected_nodes": balanced["technical_qc"][
            "connected_nodes"
        ],
        "summary": SINGLE.file_record(path),
    }


def self_test() -> dict[str, Any]:
    values = units()
    checks = {
        "exact_canary_n": len(values) == EXPECTED_N,
        "exact_matrix_n": len(SINGLE.MATRIX_NAMES) == 9,
        "balanced_view_present": "balanced" in SINGLE.VIEW_NAMES,
        "density_target_fixed": SINGLE.DENSITY_TARGET == 0.60,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--balanced-total",
        type=int,
        choices=tuple(SINGLE.BALANCED_TOTALS),
        help=(
            "Uniform density-qualified candidate to test. Defaults to the "
            "smallest density-qualified candidate."
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-unit", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error("--workers must be in 1..2")
    if not 1 <= args.threads_per_unit <= 16:
        parser.error("--threads-per-unit must be in 1..16")
    if args.workers * args.threads_per_unit > 16:
        parser.error("worker/thread product must not exceed 16")
    if args.self_test:
        value = self_test()
        print(json.dumps(value, sort_keys=True))
        return 0 if value["status"] == "PASS" else 1

    unit_list = units()
    density = density_summary()
    density_path = density[0] if density is not None else None
    qualified = (
        list(
            density[1]["density_qualified_balanced_total_counts"]
        )
        if density is not None
        else []
    )
    selected = (
        args.balanced_total
        if args.balanced_total is not None
        else (
            density[1][
                "smallest_density_qualified_balanced_total_count"
            ]
            if density is not None
            else None
        )
    )
    selection_error = (
        "requested total is not density-qualified across all 15"
        if args.balanced_total is not None
        and args.balanced_total not in qualified
        else None
    )
    live = readiness(unit_list, density_path, selected)
    ready = [
        row["unit"] for row in live if row["phase_pair_ready"]
    ]
    waiting = [
        row["unit"] for row in live if not row["phase_pair_ready"]
    ]
    reusable = [
        row["unit"]
        for row in live
        if row["existing_terminal_summary"] is not None
    ]
    runnable = sorted(set(ready) - set(reusable))
    fully_ready = (
        density is not None
        and selected in qualified
        and selection_error is None
        and not waiting
    )
    preflight = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_balanced_all_nine_"
            "canary_cohort_preflight"
        ),
        "generated_utc": utc_now(),
        "status": (
            "READY_COMPLETE_DENSITY_AND_ALL_15_EXACT_PAIRS"
            if fully_ready
            else "WAITING_FOR_COMPLETE_DENSITY_OR_ALL_15_EXACT_PAIRS"
        ),
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "expected_unit_n": EXPECTED_N,
        "density_summary": (
            str(density_path) if density_path is not None else None
        ),
        "selected_balanced_total_streamlines": selected,
        "density_qualified_balanced_total_counts": qualified,
        "selected_candidate_rank": (
            qualified.index(selected) + 1 if selected in qualified else None
        ),
        "selection_mode": (
            "EXPLICIT_UNIFORM_DENSITY_QUALIFIED_CANDIDATE"
            if args.balanced_total is not None
            else "SMALLEST_DENSITY_QUALIFIED_CANDIDATE"
        ),
        "selection_error": selection_error,
        "pair_ready_n": len(ready),
        "waiting_n": len(waiting),
        "waiting_units": waiting,
        "reusable_terminal_n": len(reusable),
        "runnable_n": len(runnable),
        "readiness": live,
        "probabilistic_tractography_generated": False,
        "all_nine_matrix_generation_started": False,
        "uniform_candidate_recipe_confirmed": False,
        "cohort_recipe_selected": False,
        "scale_up_authorized": False,
    }
    if not args.execute:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    if not fully_ready or density_path is None:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 2

    attempt = (
        ATTEMPTS
        / f"{attempt_stamp()}-hroi-balanced-all-nine-canary-cohort"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    bound_preflight = {
        **preflight,
        "status": "EXECUTION_INPUTS_BOUND",
        "all_nine_matrix_generation_started": True,
        "density_summary_record": SINGLE.file_record(density_path),
        "single_runner": SINGLE.file_record(SINGLE_SOURCE),
        "cohort_runner": SINGLE.file_record(Path(__file__)),
    }
    atomic_json(attempt / "preflight.json", bound_preflight)
    executed: dict[str, Any] = {}
    execution_failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_unit,
                unit,
                density_path=density_path,
                selected_total=selected,
                threads=args.threads_per_unit,
                log_root=attempt / "unit_logs",
            ): unit
            for unit in runnable
        }
        for future in as_completed(futures):
            unit = futures[future]
            try:
                executed[unit] = future.result()
            except Exception as exc:
                execution_failures[unit] = (
                    f"{type(exc).__name__}:{exc}"
                )

    rows = []
    missing_terminal = []
    for unit in unit_list:
        terminal = valid_unit_summary(
            unit,
            density_path=density_path,
            selected_total=selected,
        )
        if terminal is None:
            missing_terminal.append(unit)
            continue
        rows.append(unit_row(unit, *terminal))
    rows.sort(key=lambda row: row["unit"])
    complete = (
        len(rows) == EXPECTED_N
        and not missing_terminal
        and not execution_failures
    )
    passing = complete and all(
        row["status"] == "PASS_HROI_BALANCED_ALL_NINE_CANARY"
        and row["selected_balanced_total_streamlines"] == selected
        and row["combined_density_target_met"] is True
        and row["all_views_technical_qc_pass"] is True
        and row["seed_reliability_status"] == "PASS"
        for row in rows
    )
    densities = [float(row["combined_edge_density"]) for row in rows]
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_balanced_all_nine_"
            "canary_cohort_summary"
        ),
        "generated_utc": utc_now(),
        "status": (
            "PASS_COMPLETE_HROI_ALL_NINE_CANARY"
            if passing
            else "FAIL_COMPLETE_HROI_ALL_NINE_CANARY"
            if complete
            else "INCOMPLETE_HROI_ALL_NINE_CANARY"
        ),
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "expected_unit_n": EXPECTED_N,
        "processed_unit_n": len(rows),
        "missing_terminal_units": missing_terminal,
        "execution_failures": execution_failures,
        "selected_balanced_total_streamlines": selected,
        "density_qualified_balanced_total_counts": qualified,
        "selected_candidate_rank": qualified.index(selected) + 1,
        "selection_mode": preflight["selection_mode"],
        "selected_per_seed_prefix_streamlines": (
            SINGLE.BALANCED_TOTALS.get(selected)
        ),
        "matrix_names": list(SINGLE.MATRIX_NAMES),
        "minimum_combined_edge_density": (
            min(densities) if densities else None
        ),
        "maximum_combined_edge_density": (
            max(densities) if densities else None
        ),
        "all_units_density_target_met": bool(rows)
        and all(row["combined_density_target_met"] for row in rows),
        "all_units_all_nine_technical_qc_pass": bool(rows)
        and all(row["all_views_technical_qc_pass"] for row in rows),
        "all_units_seed_reliability_pass": bool(rows)
        and all(row["seed_reliability_status"] == "PASS" for row in rows),
        "uniform_candidate_recipe_confirmed": passing,
        "probabilistic_tractography_generated_by_this_stage": False,
        "source_10m_tractograms_reused": True,
        "sift2_refit_on_each_exact_construction": True,
        "cohort_recipe_selected": False,
        "scale_up_authorized": False,
        "requires_hroi_human_release_gate": True,
        "requires_separate_non_overwriting_530_execution": True,
        "units": rows,
        "preflight": SINGLE.file_record(attempt / "preflight.json"),
        "density_summary": SINGLE.file_record(density_path),
        "single_runner": SINGLE.file_record(SINGLE_SOURCE),
        "cohort_runner": SINGLE.file_record(Path(__file__)),
    }
    atomic_json(attempt / "summary.json", summary)
    print(
        json.dumps(
            {
                "attempt_root": str(attempt),
                "status": summary["status"],
                "processed_unit_n": len(rows),
                "selected_balanced_total_streamlines": selected,
                "minimum_combined_edge_density": summary[
                    "minimum_combined_edge_density"
                ],
                "uniform_candidate_recipe_confirmed": passing,
                "scale_up_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
