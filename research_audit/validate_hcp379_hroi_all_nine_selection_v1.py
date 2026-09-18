#!/usr/bin/env python3
"""Validate the ordered final-HROI all-nine recipe selector."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP / "scripts/hcp/run_hcp379_hroi_all_nine_selection_v1.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_hroi_all_nine_selection_v1/validation.json"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SELECTOR = load_module(SOURCE, "hcp379_hroi_selector_validator")
SINGLE = SELECTOR.SINGLE


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
        "dry_run_is_ordered_and_fail_closed",
        (
            dry_probe.returncode == 0
            and dry.get("status")
            in {
                "READY_TO_TEST_ORDERED_JOINT_CANDIDATES",
                "WAITING_FOR_COMPLETE_DENSITY_EVIDENCE",
                "WAITING_FOR_ALL_15_EXACT_PAIRS",
                "NO_DENSITY_QUALIFIED_CANDIDATE_UP_TO_20M",
            }
            and dry.get("pair_ready_n", 0)
            <= dry.get("expected_pair_n", 0)
            and dry.get("probabilistic_tractography_generated") is False
            and dry.get("all_nine_execution_started") is False
            and dry.get("selected_balanced_total_streamlines") is None
            and dry.get("selected_count_tractography_authorized") is False
            and dry.get("final_release_authorized") is False
            and (
                dry.get("status")
                != "NO_DENSITY_QUALIFIED_CANDIDATE_UP_TO_20M"
                or (
                    dry.get("density_calibration_terminal") is True
                    and dry.get("ordered_density_qualified_candidates")
                    == []
                    and dry.get("density_calibration_conclusion")
                    == "NO_UNIFORM_15_CANARY_CANDIDATE_AT_6M_10M_15M_OR_20M"
                )
            )
        ),
        dry or {"stderr": dry_probe.stderr[-2000:]},
    )
    text = SOURCE.read_text(encoding="utf-8")
    required = (
        "PASS_SMALLEST_JOINTLY_QUALIFIED_BALANCED_RECIPE",
        "FAIL_NO_JOINTLY_QUALIFIED_BALANCED_RECIPE",
        "complete_scientific_fail",
        "escalation prohibited",
        "density_qualified_balanced_total_counts",
        "selection_is_smallest_joint_pass",
        "per_subject_streamline_tuning_allowed",
        "requires_final_hroi_human_review",
    )
    missing = [fragment for fragment in required if fragment not in text]
    check(
        "implementation_contains_ordered_scientific_escalation_contract",
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
        '"selected_count_tractography_authorized": True',
        '"final_release_authorized": True',
    )
    found = [fragment for fragment in prohibited if fragment in text]
    check(
        "selector_has_no_new_tractography_destructive_or_outcome_shortcut",
        not found,
        {"found": found},
    )
    runtime: dict[str, Any] = {
        "detected": SELECTOR.SELECTION.is_file(),
        "status": "NOT_YET_AVAILABLE",
    }
    if SELECTOR.SELECTION.is_file():
        runtime_errors: list[str] = []
        try:
            selection = SELECTOR.load_json(SELECTOR.SELECTION)
            candidates = selection.get(
                "ordered_density_qualified_candidates", []
            )
            evaluated = selection.get("evaluated_candidates", [])
            selected = selection.get(
                "selected_balanced_total_streamlines"
            )
            expected_evaluated = (
                candidates[: candidates.index(selected) + 1]
                if selected in candidates
                else []
            )
            observed_evaluated = [
                row.get("balanced_total_streamlines")
                for row in evaluated
                if isinstance(row, Mapping)
            ]
            if (
                selection.get("record_type")
                != "diagnosis_blind_hcp379_hroi_all_nine_recipe_selection"
                or selection.get("status")
                != "PASS_SMALLEST_JOINTLY_QUALIFIED_BALANCED_RECIPE"
                or selection.get("diagnosis_labels_used") is not False
                or selection.get("non_overwriting") is not True
                or candidates != sorted(set(candidates))
                or any(
                    total not in SINGLE.BALANCED_TOTALS
                    for total in candidates
                )
                or selected not in candidates
                or observed_evaluated != expected_evaluated
                or not evaluated
                or evaluated[-1].get("status") != "PASS_JOINT_ALL_NINE"
                or any(
                    row.get("status") != "FAIL_JOINT_ALL_NINE"
                    for row in evaluated[:-1]
                )
                or selection.get("selection_is_smallest_joint_pass")
                is not True
                or selection.get("selected_per_seed_streamlines")
                != SINGLE.BALANCED_TOTALS[selected]
                or selection.get(
                    "per_subject_streamline_tuning_allowed"
                )
                is not False
                or selection.get(
                    "subject_exclusion_for_low_density_allowed"
                )
                is not False
                or selection.get(
                    "selected_count_tractography_authorized"
                )
                is not False
                or selection.get("final_release_authorized") is not False
                or selection.get("requires_final_hroi_human_review")
                is not True
            ):
                runtime_errors.append("selection contract differs")
            density_path = SINGLE.verify_record(
                selection["density_summary"],
                label="runtime/density summary",
            )
            SINGLE.valid_density_summary(density_path)
            for index, row in enumerate(evaluated):
                if not isinstance(row, Mapping):
                    runtime_errors.append(
                        f"candidate[{index}] is not an object"
                    )
                    continue
                total = row.get("balanced_total_streamlines")
                summary_path = SINGLE.verify_record(
                    row["summary"], label=f"candidate[{index}]/summary"
                )
                validation_path = SINGLE.verify_record(
                    row["validation"],
                    label=f"candidate[{index}]/validation",
                )
                summary = SELECTOR.load_json(summary_path)
                validation = SELECTOR.load_json(validation_path)
                expected_summary_status = (
                    "PASS_COMPLETE_HROI_ALL_NINE_CANARY"
                    if index == len(evaluated) - 1
                    else "FAIL_COMPLETE_HROI_ALL_NINE_CANARY"
                )
                if (
                    summary.get("status") != expected_summary_status
                    or summary.get(
                        "selected_balanced_total_streamlines"
                    )
                    != total
                    or summary.get("processed_unit_n")
                    != SELECTOR.COHORT.EXPECTED_N
                    or summary.get("missing_terminal_units")
                    or summary.get("execution_failures")
                ):
                    runtime_errors.append(
                        f"candidate[{index}] summary differs"
                    )
                if (
                    validation.get("status") != "PASS"
                    or validation.get("runtime_summary_detected") is not True
                    or validation.get("runtime", {}).get(
                        "runtime_status"
                    )
                    != "PASS"
                    or validation.get("runtime", {}).get(
                        "selected_balanced_total_streamlines"
                    )
                    != total
                ):
                    runtime_errors.append(
                        f"candidate[{index}] validation differs"
                    )
            runtime = {
                "detected": True,
                "status": "PASS" if not runtime_errors else "FAIL",
                "selected_balanced_total_streamlines": selected,
                "selected_per_seed_streamlines": selection.get(
                    "selected_per_seed_streamlines"
                ),
                "evaluated_candidate_n": len(evaluated),
                "errors": runtime_errors,
                "selection": SINGLE.file_record(SELECTOR.SELECTION),
            }
        except Exception as exc:
            runtime = {
                "detected": True,
                "status": "FAIL",
                "errors": [f"{type(exc).__name__}:{exc}"],
            }
        check(
            "runtime_selection_replays_as_smallest_joint_pass",
            runtime.get("status") == "PASS",
            runtime,
        )
    passed = sum(row["status"] == "PASS" for row in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_hroi_all_nine_selection_v1_validation",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "runtime_selection_detected": runtime["detected"],
        "runtime": runtime,
        "imaging_executed_by_validation": False,
        "source": SINGLE.file_record(SOURCE),
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
