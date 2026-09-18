#!/usr/bin/env python3
"""Validate the complete 15-canary selected-HROI all-nine cohort."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_hroi_balanced_all_nine_canary_cohort_v1.py"
)
SINGLE_VALIDATOR = (
    EXP
    / "research_audit/"
    "validate_hcp379_hroi_balanced_all_nine_canary_v1.py"
)
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_hroi_balanced_all_nine_canary_cohort_v1/attempts"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_hroi_balanced_all_nine_canary_cohort_v1/validation.json"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COHORT = load_module(SOURCE, "hcp379_hroi_all_nine_cohort_validator")
SINGLE = COHORT.SINGLE


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def latest_summary() -> Path | None:
    paths = sorted(ATTEMPTS.glob("*/summary.json"))
    return paths[-1].resolve() if paths else None


def runtime_replay(path: Path) -> dict[str, Any]:
    value = load_json(path)
    failures: list[str] = []
    rows = value.get("units", [])
    if (
        value.get("record_type")
        != "diagnosis_blind_hcp379_hroi_balanced_all_nine_canary_cohort_summary"
        or value.get("status")
        not in {
            "PASS_COMPLETE_HROI_ALL_NINE_CANARY",
            "FAIL_COMPLETE_HROI_ALL_NINE_CANARY",
            "INCOMPLETE_HROI_ALL_NINE_CANARY",
        }
        or value.get("diagnosis_labels_used") is not False
        or value.get("non_overwriting") is not True
        or value.get("cohort_recipe_selected") is not False
        or value.get("scale_up_authorized") is not False
        or value.get("requires_hroi_human_release_gate") is not True
        or value.get("requires_separate_non_overwriting_530_execution")
        is not True
        or set(value.get("matrix_names", [])) != set(SINGLE.MATRIX_NAMES)
    ):
        failures.append("cohort_summary_contract_differs")
    try:
        if value.get("cohort_runner") != SINGLE.file_record(SOURCE):
            failures.append("cohort_runner_binding_differs")
        if value.get("single_runner") != SINGLE.file_record(
            COHORT.SINGLE_SOURCE
        ):
            failures.append("single_runner_binding_differs")
        density_path = SINGLE.verify_record(
            value["density_summary"], label="density summary"
        )
        density = SINGLE.valid_density_summary(density_path)
        selected = int(value["selected_balanced_total_streamlines"])
        qualified = density[
            "density_qualified_balanced_total_counts"
        ]
        if selected not in qualified:
            failures.append("selected_count_is_not_density_qualified")
        if value.get("selected_candidate_rank") != (
            qualified.index(selected) + 1
        ):
            failures.append("selected_candidate_rank_differs")
    except Exception as exc:
        selected = -1
        failures.append(f"input_binding:{type(exc).__name__}:{exc}")

    unit_replays: dict[str, Any] = {}
    row_units = [
        str(row.get("unit", "")) for row in rows if isinstance(row, dict)
    ]
    if (
        len(rows) != COHORT.EXPECTED_N
        or len(row_units) != COHORT.EXPECTED_N
        or len(set(row_units)) != COHORT.EXPECTED_N
        or set(row_units) != set(COHORT.units())
    ):
        failures.append("unit_set_differs")
    for row in rows:
        if not isinstance(row, dict):
            continue
        unit = str(row.get("unit", ""))
        try:
            summary_path = SINGLE.verify_record(
                row["summary"], label=f"{unit}/summary"
            )
            terminal = load_json(summary_path)
            if (
                terminal.get("unit") != unit
                or terminal.get("status") != row.get("status")
                or terminal.get("selected_balanced_total_streamlines")
                != selected
                or abs(
                    float(terminal.get("combined_edge_density", -1.0))
                    - float(row.get("combined_edge_density", -2.0))
                )
                > 1.0e-15
            ):
                raise ValueError("unit row differs from terminal summary")
            replay = subprocess.run(
                [
                    str(PYTHON),
                    str(SINGLE_VALIDATOR),
                    "--summary",
                    str(summary_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            replay_value = json.loads(replay.stdout)
            unit_replays[unit] = {
                "returncode": replay.returncode,
                "status": replay_value.get("status"),
                "passed_checks": replay_value.get("passed_checks"),
                "total_checks": replay_value.get("total_checks"),
            }
            if (
                replay.returncode != 0
                or replay_value.get("status") != "PASS"
                or replay_value.get("runtime_summary_detected") is not True
            ):
                failures.append(f"{unit}:independent_replay_failed")
        except Exception as exc:
            failures.append(
                f"{unit}:runtime_replay:{type(exc).__name__}:{exc}"
            )

    complete = len(rows) == COHORT.EXPECTED_N and not any(
        "unit_set_differs" in failure for failure in failures
    )
    all_pass = complete and all(
        row.get("status") == "PASS_HROI_BALANCED_ALL_NINE_CANARY"
        and row.get("selected_balanced_total_streamlines") == selected
        and row.get("combined_density_target_met") is True
        and row.get("all_views_technical_qc_pass") is True
        and row.get("seed_reliability_status") == "PASS"
        for row in rows
    )
    claimed_pass = (
        value.get("status") == "PASS_COMPLETE_HROI_ALL_NINE_CANARY"
        and value.get("uniform_candidate_recipe_confirmed") is True
    )
    if all_pass != claimed_pass:
        failures.append("cohort_terminal_status_differs_from_rows")
    densities = [
        float(row["combined_edge_density"])
        for row in rows
        if isinstance(row, dict) and "combined_edge_density" in row
    ]
    if densities:
        if abs(
            min(densities)
            - float(value.get("minimum_combined_edge_density", -1.0))
        ) > 1.0e-15:
            failures.append("minimum_density_differs")
        if abs(
            max(densities)
            - float(value.get("maximum_combined_edge_density", -1.0))
        ) > 1.0e-15:
            failures.append("maximum_density_differs")
    return {
        "summary": SINGLE.file_record(path),
        "selected_balanced_total_streamlines": selected,
        "unit_replays": unit_replays,
        "runtime_status": "PASS" if not failures else "FAIL",
        "runtime_failures": sorted(set(failures)),
    }


def main() -> int:
    checks: dict[str, Any] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    compile_probe = subprocess.run(
        [str(PYTHON), "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "source_compiles",
        compile_probe.returncode == 0,
        {
            "returncode": compile_probe.returncode,
            "stderr": compile_probe.stderr[-2000:],
        },
    )
    self_probe = subprocess.run(
        [str(PYTHON), str(SOURCE), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        self_value = json.loads(self_probe.stdout)
    except Exception:
        self_value = {}
    check(
        "self_test_passes",
        self_probe.returncode == 0 and self_value.get("status") == "PASS",
        self_value or {"stderr": self_probe.stderr[-2000:]},
    )
    dry_probe = subprocess.run(
        [str(PYTHON), str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        dry = json.loads(dry_probe.stdout)
    except Exception:
        dry = {}
    check(
        "dry_run_is_complete_cohort_fail_closed",
        (
            dry_probe.returncode == 0
            and dry.get("status")
            in {
                "READY_COMPLETE_DENSITY_AND_ALL_15_EXACT_PAIRS",
                "WAITING_FOR_COMPLETE_DENSITY_OR_ALL_15_EXACT_PAIRS",
            }
            and dry.get("expected_unit_n") == COHORT.EXPECTED_N
            and dry.get("pair_ready_n", 0) + dry.get("waiting_n", 0)
            == COHORT.EXPECTED_N
            and dry.get("all_nine_matrix_generation_started") is False
            and dry.get("uniform_candidate_recipe_confirmed") is False
            and dry.get("cohort_recipe_selected") is False
            and dry.get("scale_up_authorized") is False
        ),
        dry or {"stderr": dry_probe.stderr[-2000:]},
    )
    text = SOURCE.read_text(encoding="utf-8")
    required = (
        "PASS_COMPLETE_HROI_ALL_NINE_CANARY",
        "FAIL_COMPLETE_HROI_ALL_NINE_CANARY",
        "uniform_candidate_recipe_confirmed",
        "all_units_density_target_met",
        "all_units_all_nine_technical_qc_pass",
        "all_units_seed_reliability_pass",
        "requires_hroi_human_release_gate",
        "requires_separate_non_overwriting_530_execution",
    )
    missing = [fragment for fragment in required if fragment not in text]
    check(
        "implementation_contains_complete_15_canary_contract",
        not missing,
        {"missing": missing},
    )
    prohibited = (
        "tckgen",
        "rm -rf",
        "shutil.rmtree",
        ".unlink(",
        "diagnosis.csv",
        "DX_bl",
        '"cohort_recipe_selected": True',
        '"scale_up_authorized": True',
    )
    found = [fragment for fragment in prohibited if fragment in text]
    check(
        "implementation_has_no_tractography_destructive_or_outcome_shortcut",
        not found,
        {"found": found},
    )

    runtime_path = latest_summary()
    runtime = runtime_replay(runtime_path) if runtime_path else None
    if runtime is not None:
        check(
            "runtime_cohort_replays",
            runtime["runtime_status"] == "PASS",
            runtime,
        )
    passed = sum(row["status"] == "PASS" for row in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_hroi_balanced_all_nine_canary_cohort_v1_validation"
        ),
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "runtime_summary_detected": runtime is not None,
        "runtime": runtime,
        "imaging_executed_by_validation": False,
        "source": SINGLE.file_record(SOURCE),
        "single_validator": SINGLE.file_record(SINGLE_VALIDATOR),
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
