#!/home/ec2-user/fsl/bin/python
"""Hostile validation for the corrected archive-86 Recovery4 merger."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP
    / "scripts/hcp/"
    "assemble_hcp379_corrected_archive_86_recovery4_v3.py"
)
FINAL_SOURCE = (
    EXP / "scripts/hcp/build_hcp379_final_release_v2.py"
)
PLAN = (
    Path("/data/derivatives/hcp379_v2")
    / "corrected_archive_recovery4/manifests/"
    "corrected_archive_86_merge_plan.json"
)
OUTPUT = (
    Path("/data/derivatives/hcp379_v2")
    / "corrected_archive_recovery4/manifests/"
    "corrected_archive_86_release_manifest.json"
)
FINAL_READINESS = (
    Path("/data/derivatives/hcp379_v2")
    / "release_candidate_v2/hcp379_release_readiness.json"
)
VALIDATION_OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_corrected_archive_86_recovery4_v3/validation.json"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_component(
    *,
    record_type: str,
    unit_count: int,
    selected_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "record_type": record_type,
        "status": "PASS",
        "diagnosis_labels_used": False,
        "unit_count": unit_count,
        "selected_streamline_count": selected_count,
        "density_is_subject_inclusion_gate": False,
        "density_is_whole_cohort_technical_release_gate": True,
        "minimum_edge_density_inclusive": 0.60,
        "subjects_at_or_above_minimum_density": unit_count,
        "minimum_observed_edge_density": 0.60,
        "units": [{} for _ in range(unit_count)],
    }


def main() -> int:
    module = load_module(SOURCE, "corrected_archive_86_validation_target")
    final = load_module(FINAL_SOURCE, "archive_86_final_validation_target")
    plan = final.load_json(PLAN)
    readiness = final.load_json(FINAL_READINESS)
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if condition else "FAIL",
            "evidence": evidence,
        }

    low, D0, archive = module.exact_expected_units()
    check(
        "exact_disjoint_archive_30_plus_56_equals_86",
        len(low) == 30
        and len(D0) == 56
        and not low & D0
        and low | D0 == archive
        and len(archive) == 86,
        {
            "low_n": len(low),
            "D0_n": len(D0),
            "overlap_n": len(low & D0),
            "union_n": len(archive),
        },
    )
    check(
        "preflight_waits_for_real_components_without_imaging",
        plan.get("status") == "WAITING_FOR_CORRECTED_ARCHIVE_COMPONENTS"
        and plan.get("diagnosis_labels_used") is False
        and plan.get("diagnosis_or_outcomes_used") is False
        and plan.get("non_overwriting") is True
        and plan.get("imaging_executed_by_plan") is False
        and plan.get("archive_route_selected_by_plan") is False
        and plan.get("archive_concordance_still_required") is True
        and plan.get("target_n") == 86
        and all(
            state.get("present") is False
            and state.get("status") == "NOT_RUN"
            for state in plan.get("sources", {}).values()
        ),
        {
            "status": plan.get("status"),
            "sources": plan.get("sources"),
            "route_selected": plan.get(
                "archive_route_selected_by_plan"
            ),
        },
    )
    check(
        "hard_density_nine_matrix_and_uniform_count_contract",
        plan.get("minimum_density_inclusive") == 0.60
        and plan.get("required_density_pass_n") == 86
        and tuple(plan.get("required_matrix_names", ()))
        == tuple(final.MATRIX_NAMES)
        and len(plan.get("required_matrix_names", ())) == 9
        and tuple(plan.get("allowed_streamline_counts", ()))
        == final.ALLOWED_STREAMLINE_COUNTS,
        {
            "minimum_density": plan.get("minimum_density_inclusive"),
            "required_density_n": plan.get("required_density_pass_n"),
            "matrix_names": plan.get("required_matrix_names"),
            "streamline_counts": plan.get("allowed_streamline_counts"),
        },
    )
    check(
        "plan_is_hash_bound_to_current_sources_and_topology",
        plan["records"]["builder"] == final.file_record(SOURCE)
        and plan["records"]["final_release_contract"]
        == final.file_record(FINAL_SOURCE)
        and plan["records"]["density_execution_topology"]
        == final.file_record(final.DENSITY_EXECUTION_TOPOLOGY)
        and plan["records"]["density_execution_subjects"]
        == final.file_record(final.DENSITY_EXECUTION_SUBJECTS),
        plan.get("records"),
    )
    try:
        module.assemble()
    except FileNotFoundError as exc:
        missing_sources_rejected = True
        missing_sources_evidence = str(exc)
    else:
        missing_sources_rejected = False
        missing_sources_evidence = "unexpectedly assembled"
    check(
        "assemble_rejects_missing_components",
        missing_sources_rejected and not OUTPUT.exists(),
        {
            "error": missing_sources_evidence,
            "output_present": OUTPUT.exists(),
        },
    )
    original_output = module.OUTPUT
    try:
        with tempfile.TemporaryDirectory(
            prefix="hcp379-archive86-nonoverwrite-"
        ) as temporary:
            occupied = Path(temporary) / "occupied.json"
            occupied.write_text("{}\n", encoding="utf-8")
            module.OUTPUT = occupied
            try:
                module.assemble()
            except FileExistsError as exc:
                overwrite_rejected = True
                overwrite_evidence = str(exc)
            else:
                overwrite_rejected = False
                overwrite_evidence = "unexpectedly overwrote output"
    finally:
        module.OUTPUT = original_output
    check(
        "existing_output_is_never_overwritten",
        overwrite_rejected,
        overwrite_evidence,
    )
    original_low = module.LOW_30
    original_D0 = module.D0_56
    try:
        with tempfile.TemporaryDirectory(
            prefix="hcp379-archive86-count-"
        ) as temporary:
            root = Path(temporary)
            low_path = root / "low.json"
            D0_path = root / "D0.json"
            low_path.write_text(
                json.dumps(
                    synthetic_component(
                        record_type=module.LOW_RECORD_TYPE,
                        unit_count=30,
                        selected_count=3_000_000,
                    )
                ),
                encoding="utf-8",
            )
            D0_path.write_text(
                json.dumps(
                    synthetic_component(
                        record_type=module.D0_RECORD_TYPE,
                        unit_count=56,
                        selected_count=5_000_000,
                    )
                ),
                encoding="utf-8",
            )
            module.LOW_30 = low_path
            module.D0_56 = D0_path
            mismatch = module.preflight_payload()
            D0_path.write_text(
                json.dumps(
                    synthetic_component(
                        record_type=module.D0_RECORD_TYPE,
                        unit_count=56,
                        selected_count=3_000_000,
                    )
                ),
                encoding="utf-8",
            )
            matched = module.preflight_payload()
            try:
                module.validate_component(
                    low_path,
                    expected_n=30,
                    expected_record_type=module.LOW_RECORD_TYPE,
                    expected_units=low,
                )
            except (OSError, TypeError, ValueError) as exc:
                incomplete_rows_rejected = True
                incomplete_rows_evidence = str(exc)
            else:
                incomplete_rows_rejected = False
                incomplete_rows_evidence = (
                    "unexpectedly accepted incomplete rows"
                )
    finally:
        module.LOW_30 = original_low
        module.D0_56 = original_D0
    check(
        "different_phase_b_counts_cannot_merge",
        mismatch.get("status")
        == "WAITING_FOR_CORRECTED_ARCHIVE_COMPONENTS"
        and matched.get("status")
        == "READY_TO_ASSEMBLE_CORRECTED_ARCHIVE_86",
        {
            "mismatch_status": mismatch.get("status"),
            "matched_status": matched.get("status"),
        },
    )
    check(
        "header_only_or_missing_matrix_rows_cannot_assemble",
        incomplete_rows_rejected,
        incomplete_rows_evidence,
    )
    check(
        "final_release_contract_uses_recovery4_archive_86_path",
        final.CORRECTED_ARCHIVE_86 == OUTPUT
        and "corrected_archive_recovery4" in str(
            final.CORRECTED_ARCHIVE_86
        ),
        str(final.CORRECTED_ARCHIVE_86),
    )
    check(
        "archive_route_decision_is_not_fabricated",
        readiness.get("status") == "NOT_READY"
        and readiness.get("archive_route_decision") is None
        and "archive_route_concordance_decision_missing"
        in readiness.get("blockers", [])
        and not final.ARCHIVE_CONCORDANCE.exists(),
        {
            "status": readiness.get("status"),
            "decision": readiness.get("archive_route_decision"),
            "blockers": readiness.get("blockers"),
        },
    )
    compile_probe = subprocess.run(
        [
            "/home/ec2-user/fsl/bin/python",
            "-m",
            "py_compile",
            str(SOURCE),
            str(FINAL_SOURCE),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    self_test_probe = subprocess.run(
        [
            "/home/ec2-user/fsl/bin/python",
            str(SOURCE),
            "--self-test",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    check(
        "sources_compile_and_merger_self_test_passes",
        compile_probe.returncode == 0
        and self_test_probe.returncode == 0
        and "SELF_TEST_PASS" in self_test_probe.stdout,
        {
            "compile_returncode": compile_probe.returncode,
            "compile_stderr": compile_probe.stderr,
            "self_test_returncode": self_test_probe.returncode,
            "self_test_stdout": self_test_probe.stdout,
            "self_test_stderr": self_test_probe.stderr,
        },
    )
    check(
        "no_corrected_archive_86_output_was_created",
        not OUTPUT.exists(),
        {"output": str(OUTPUT), "present": OUTPUT.exists()},
    )

    passed = sum(
        value["status"] == "PASS" for value in checks.values()
    )
    payload = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_corrected_archive_86_recovery4_v3_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "imaging_executed": False,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "records": {
            "source": final.file_record(SOURCE),
            "merge_plan": final.file_record(PLAN),
            "final_release_contract": final.file_record(FINAL_SOURCE),
            "final_readiness": final.file_record(FINAL_READINESS),
        },
    }
    VALIDATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final.atomic_json(VALIDATION_OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
