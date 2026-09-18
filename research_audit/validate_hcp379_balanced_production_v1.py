#!/home/ec2-user/fsl/bin/python
"""Independently validate the balanced HCP379 production implementation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP / "scripts/hcp/run_hcp379_balanced_production_v1.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_balanced_production_v1/validation.json"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_module(SOURCE, "hcp379_balanced_production_validator")


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
        "dry_run_is_non_imaging_and_fail_closed",
        (
            dry_probe.returncode == 0
            and dry.get("status")
            in {
                "WAITING_FOR_BALANCED_PRODUCTION_GATES",
                "READY_FOR_BALANCED_PRODUCTION_EXECUTION",
            }
            and dry.get("diagnosis_labels_used") is False
            and dry.get("non_overwriting") is True
            and dry.get("imaging_executed") is False
            and dry.get("probabilistic_tractography_generated") is False
            and dry.get("production_n") == 515
            and dry.get("pretract_ready_n", 0)
            + dry.get("pretract_waiting_n", 0)
            == 515
            and dry.get("per_subject_streamline_tuning_allowed")
            is False
            and dry.get("subject_exclusion_for_low_density_allowed")
            is False
            and dry.get("final_release_authorized") is False
        ),
        dry or {"stderr": dry_probe.stderr[-2000:]},
    )
    text = SOURCE.read_text(encoding="utf-8")
    required = (
        "PASS_SMALLEST_JOINTLY_QUALIFIED_BALANCED_RECIPE",
        "PASS_COMPLETED_HUMAN_HROI_REVIEW",
        "PASS_COMPACTED",
        "primary",
        "independent",
        "independent-replicate-1",
        "MRTRIX_RNG_SEED",
        "tckgen",
        "concatenate",
        "build_view",
        "sift2_fit_on_exact_balanced_tractogram",
        "combined_density_target_met",
        "per_subject_streamline_tuning_allowed",
        "subject_exclusion_for_low_density_allowed",
        "fail_fast_after_first_nonpass",
        "stratified_pilot",
        "final_release_authorized",
    )
    missing = [fragment for fragment in required if fragment not in text]
    check(
        "implementation_contains_balanced_all_nine_contract",
        not missing,
        {"missing": missing},
    )
    prohibited = (
        "rm -rf",
        "shutil.rmtree",
        "DX_bl",
        "diagnosis.csv",
        '"final_release_authorized": True',
        '"per_subject_streamline_tuning_allowed": True',
        '"subject_exclusion_for_low_density_allowed": True',
    )
    found = [fragment for fragment in prohibited if fragment in text]
    check(
        "implementation_has_no_broad_delete_or_outcome_shortcut",
        not found,
        {"found": found},
    )
    runtime: dict[str, Any] = {
        "detected": RUNNER.ROOT.joinpath("cohort_state.json").is_file(),
        "status": "NOT_YET_AVAILABLE",
    }
    state_path = RUNNER.ROOT / "cohort_state.json"
    if state_path.is_file() and not state_path.is_symlink():
        errors: list[str] = []
        try:
            state = RUNNER.load_json(state_path)
            subjects = state.get("subjects", [])
            if (
                state.get("record_type")
                != "diagnosis_blind_hcp379_balanced_production_cohort"
                or state.get("diagnosis_labels_used") is not False
                or state.get("non_overwriting") is not True
                or state.get("processed_n") != len(subjects)
                or state.get("target_n") != len(subjects)
                or state.get("final_release_authorized") is not False
            ):
                errors.append("cohort state contract differs")
            for row in subjects:
                if not isinstance(row, Mapping):
                    errors.append("subject state is not an object")
                    continue
                unit = str(row.get("unit", ""))
                if (
                    row.get("status") != "PASS_BALANCED_PRODUCTION"
                    or row.get("combined_density_target_met") is not True
                    or float(row.get("combined_edge_density", 0))
                    < RUNNER.DENSITY_TARGET
                ):
                    errors.append(f"{unit}: scientific status differs")
                summary_path = RUNNER.verify_record(
                    row["summary"], label=f"{unit}/summary"
                )
                compaction_path = RUNNER.verify_record(
                    row["compaction"], label=f"{unit}/compaction"
                )
                summary = RUNNER.load_json(summary_path)
                compaction = RUNNER.load_json(compaction_path)
                view = summary.get("balanced_view", {})
                matrix_records = view.get("matrices", {})
                if (
                    summary.get("matrix_names")
                    != list(RUNNER.ALL_NINE.MATRIX_NAMES)
                    or view.get("technical_status") != "PASS"
                    or set(matrix_records)
                    != set(RUNNER.ALL_NINE.MATRIX_NAMES)
                    or compaction.get("status")
                    != "PASS_COMPACTED_GENERATED_TRACKS"
                    or compaction.get("retained_scientific_summary")
                    != RUNNER.file_record(summary_path)
                    or any(
                        Path(record["path"]).exists()
                        for record in compaction.get(
                            "generated_tractograms_deleted", {}
                        ).values()
                    )
                ):
                    errors.append(f"{unit}: retained evidence differs")
                    continue
                for name, record in matrix_records.items():
                    matrix_path = RUNNER.verify_record(
                        record, label=f"{unit}/matrix/{name}"
                    )
                    RUNNER.ALL_NINE.matrix(matrix_path)
                RUNNER.verify_record(
                    view["assignments"],
                    label=f"{unit}/assignments",
                )
                for name in ("weights", "mu", "iterations"):
                    RUNNER.verify_record(
                        view["sift2"][name],
                        label=f"{unit}/SIFT2/{name}",
                    )
            runtime = {
                "detected": True,
                "status": "PASS" if not errors else "FAIL",
                "processed_n": len(subjects),
                "errors": errors,
                "cohort_state": RUNNER.file_record(state_path),
            }
        except Exception as exc:
            runtime = {
                "detected": True,
                "status": "FAIL",
                "errors": [f"{type(exc).__name__}:{exc}"],
            }
        check(
            "runtime_subject_evidence_replays",
            runtime.get("status") == "PASS",
            runtime,
        )

    passed = sum(row["status"] == "PASS" for row in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_balanced_production_v1_validation",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "runtime": runtime,
        "imaging_executed_by_validation": False,
        "source": RUNNER.file_record(SOURCE),
        "checks": checks,
    }
    RUNNER.atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
