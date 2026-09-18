#!/home/ec2-user/fsl/bin/python
"""Hostile-gate validation for the Recovery4 archive calibration package."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
PRETRACT_SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_archive_calibration_pretract_recovery4_v3.py"
)
TRACT_SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_archive_calibration_tractography_recovery4_v3.py"
)
REPLICATE_SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_archive_calibration_replicate_recovery4_v3.py"
)
ROUTE_VALIDATOR_SOURCE = (
    EXP
    / "research_audit/"
    "validate_hcp379_archive_route_concordance_recovery4_v3.py"
)
NUMERIC_ROUTE_VALIDATOR_SOURCE = (
    EXP
    / "research_audit/"
    "validate_hcp379_archive_route_concordance_v2.py"
)
KEEP_VALIDATOR_SOURCE = (
    EXP
    / "research_audit/"
    "validate_hcp379_keep_route_concordance_v2.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_archive_calibration_recovery4_v3/validation.json"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    pretract = load_module(PRETRACT_SOURCE, "archive_pretract_validation")
    tract = load_module(TRACT_SOURCE, "archive_tract_validation")
    replicate = load_module(REPLICATE_SOURCE, "archive_replicate_validation")
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if condition else "FAIL",
            "evidence": evidence,
        }

    units = pretract.selection_units()
    check(
        "exact_six_subject_scope",
        len(units) == 6,
        {"unit_count": len(units), "units": sorted(units)},
    )
    selection = pretract.PRETRACT.load_json(pretract.SELECTION)
    check(
        "selection_is_diagnosis_and_outcome_blind",
        selection.get("status") == "PASS"
        and selection.get("diagnosis_or_outcome_fields_used") is False
        and selection.get("target_n") == 6,
        {
            "status": selection.get("status"),
            "diagnosis_or_outcome_fields_used": selection.get(
                "diagnosis_or_outcome_fields_used"
            ),
        },
    )

    pretract.specialize(pretract.OUTPUT_ROOT)
    gate = pretract.PRETRACT.validate_recovery4_gate(
        human_qc=pretract.DEFAULT_HUMAN_QC,
        execute=False,
    )
    rows, overlap = pretract.PRETRACT.validate_audit_rows(gate)
    route_counts: dict[str, int] = {}
    for row in rows:
        route_counts[row["route"]] = route_counts.get(row["route"], 0) + 1
    check(
        "archive_pretract_specialization",
        len(rows) == 6
        and not overlap
        and {row["unit"] for row in rows} == units,
        {"row_count": len(rows), "overlap": overlap},
    )
    check(
        "archive_input_routes_are_exact",
        route_counts
        == {
            "READY_FOR_CORRECTED_PRETRACT_FROM_EDDY": 4,
            "READY_AFTER_GRADIENT_HEADER_REPAIR": 2,
        },
        route_counts,
    )
    try:
        pretract.PRETRACT.validate_recovery4_gate(
            human_qc=pretract.DEFAULT_HUMAN_QC,
            execute=True,
        )
    except FileNotFoundError as exc:
        missing_human_rejected = (
            "genuine human-QC CSV" in str(exc)
            or "HROI pretract-adoption receipt is required" in str(exc)
        )
        missing_human_evidence = str(exc)
    else:
        missing_human_rejected = False
        missing_human_evidence = "unexpectedly accepted"
    check(
        "pretract_execute_rejects_missing_human_review",
        missing_human_rejected,
        missing_human_evidence,
    )

    pretract_plan = pretract.PRETRACT.load_json(
        pretract.OUTPUT_ROOT
        / "manifests/archive_pretract_recovery4_plan.json"
    )
    check(
        "pretract_plan_is_nonexecuting_and_exact",
        pretract_plan.get("status")
        in {
            "PRE_HUMAN_DRY_RUN_PASS",
            "WAITING_FOR_530_HROI_POLICY_ADOPTION",
        }
        and pretract_plan.get("execution_requested") is False
        and pretract_plan.get("imaging_executed_by_plan") is False
        and pretract_plan.get("target_n") == 6
        and set(pretract_plan.get("units", [])) == units,
        {
            "status": pretract_plan.get("status"),
            "execution_requested": pretract_plan.get(
                "execution_requested"
            ),
            "target_n": pretract_plan.get("target_n"),
        },
    )

    tract.specialize(tract.ROOT)
    tract.TRACT.install_recovery4_adapter()
    tract_plan = tract.TRACT.PRETRACT.load_json(
        tract.ROOT
        / "manifests/archive_tractography_recovery4_plan.json"
    )
    check(
        "primary_tractography_plan_is_prephase_only",
        tract_plan.get("status") == "PRE_PHASE_DRY_RUN_PASS"
        and tract_plan.get("execution_authorized") is False
        and tract_plan.get("imaging_executed_by_plan") is False
        and tract_plan.get("target_n") == 6
        and set(tract_plan.get("units", [])) == units,
        {
            "status": tract_plan.get("status"),
            "execution_authorized": tract_plan.get(
                "execution_authorized"
            ),
            "target_n": tract_plan.get("target_n"),
        },
    )
    shared_tract_plan = tract.TRACT.PRETRACT.load_json(
        tract.ROOT
        / "manifests/"
        "tractography_recovery4_selected_recipe_plan.json"
    )
    check(
        "phase_b_recipe_is_not_fabricated",
        shared_tract_plan.get("selected_streamline_count") is None
        and all(
            not value.get("present")
            and value.get("status") == "NOT_RUN"
            for value in shared_tract_plan.get(
                "phase_b_status", {}
            ).values()
        ),
        {
            "selected_streamline_count": shared_tract_plan.get(
                "selected_streamline_count"
            ),
            "phase_b_status": shared_tract_plan.get("phase_b_status"),
        },
    )
    check(
        "exact_nine_matrix_contract",
        tuple(shared_tract_plan.get("matrix_names", ()))
        == tuple(tract.TRACT.ENGINE.MATRIX_NAMES)
        and len(shared_tract_plan.get("matrix_names", ())) == 9,
        shared_tract_plan.get("matrix_names"),
    )
    check(
        "density_is_not_a_subject_exclusion",
        shared_tract_plan.get("density_is_subject_inclusion_gate")
        is False,
        {
            "density_is_subject_inclusion_gate": shared_tract_plan.get(
                "density_is_subject_inclusion_gate"
            )
        },
    )
    adapter = shared_tract_plan.get("pretract_adapter_contract", {})
    check(
        "tractography_consumes_only_promoted_recovery4_inputs",
        adapter.get("required_status") == "PASS_PRETRACT_RECOVERY4"
        and adapter.get("selected_five_tt") is True
        and adapter.get("selected_hcp379") is True
        and adapter.get("bounded_fa_md_rd_ad") is True
        and adapter.get("raw_tensor_maps_are_not_sampled") is True,
        adapter,
    )
    try:
        tract.TRACT.validate_full_gate(tract.ROOT)
    except (FileNotFoundError, ValueError) as exc:
        premature_primary_rejected = "pretract" in str(exc).lower()
        premature_primary_evidence = str(exc)
    else:
        premature_primary_rejected = False
        premature_primary_evidence = "unexpectedly accepted"
    check(
        "primary_execute_rejects_incomplete_pretract",
        premature_primary_rejected,
        premature_primary_evidence,
    )

    replicate.PRIMARY.specialize(replicate.ROOT)
    replicate.TRACT.install_recovery4_adapter()
    replicate_plan = replicate.TRACT.PRETRACT.load_json(
        replicate.ROOT
        / "manifests/archive_replicate_recovery4_plan.json"
    )
    check(
        "independent_seed_plan_is_nonexecuting",
        replicate_plan.get("status") == "PRE_PHASE_DRY_RUN_PASS"
        and replicate_plan.get("execution_authorized") is False
        and replicate_plan.get("imaging_executed_by_plan") is False
        and replicate_plan.get("seed_distinctness_self_test") is True
        and set(replicate_plan.get("units", [])) == units,
        {
            "status": replicate_plan.get("status"),
            "seed_distinctness_self_test": replicate_plan.get(
                "seed_distinctness_self_test"
            ),
        },
    )
    sample = {
        "dti_source_id": "a" * 64,
        "dti_raw_bundle_sha256": "b" * 64,
    }
    primary_token, primary_seed = replicate.TRACT.ENGINE.subject_seed(
        sample, "recipe"
    )
    independent_token, independent_seed = replicate.replicate_seed(
        sample, "recipe"
    )
    check(
        "independent_seed_is_deterministic_and_distinct",
        primary_token != independent_token
        and primary_seed != independent_seed
        and replicate.replicate_seed(sample, "recipe")
        == (independent_token, independent_seed),
        {
            "primary_seed": primary_seed,
            "independent_seed": independent_seed,
        },
    )

    expected_missing = (
        not tract.PRIMARY_MANIFEST.is_file()
        and not replicate.REPLICATE_MANIFEST.is_file()
    )
    subject_root = pretract.OUTPUT_ROOT / "subjects"
    subject_outputs = (
        list(subject_root.iterdir()) if subject_root.is_dir() else []
    )
    check(
        "dry_runs_created_no_passing_subject_outputs",
        expected_missing
        and not any(
            '"status": "PASS_PRETRACT_RECOVERY4"'
            in path.read_text(encoding="utf-8")
            for path in (
                pretract.OUTPUT_ROOT / "qc/subjects"
            ).glob("*.json")
        ),
        {
            "primary_manifest_present": tract.PRIMARY_MANIFEST.is_file(),
            "replicate_manifest_present": (
                replicate.REPLICATE_MANIFEST.is_file()
            ),
            "preexisting_subject_output_count": len(subject_outputs),
        },
    )

    source_records = []
    for plan in (pretract_plan, tract_plan, replicate_plan):
        source_records.extend(plan.get("records", {}).values())
    check(
        "plans_are_hash_bound",
        bool(source_records)
        and all(
            isinstance(record, dict)
            and len(record.get("sha256", "")) == 64
            for record in source_records
        ),
        {"record_count": len(source_records)},
    )
    check(
        "route_validator_requires_recovery4_manifests",
        "archive_corrected_calibration_recovery4_manifest.json"
        in ROUTE_VALIDATOR_SOURCE.read_text(encoding="utf-8")
        and "archive_corrected_replicate_recovery4_manifest.json"
        in ROUTE_VALIDATOR_SOURCE.read_text(encoding="utf-8"),
        {"validator": str(ROUTE_VALIDATOR_SOURCE)},
    )
    numeric_text = NUMERIC_ROUTE_VALIDATOR_SOURCE.read_text(
        encoding="utf-8"
    )
    check(
        "archive_route_decision_is_selective_and_density_locked",
        "EXPECTED_ARCHIVE_D0_N = 56" in numeric_text
        and "EXPECTED_ARCHIVE_LOW_N = 30" in numeric_text
        and "low_archive_route_is_never_retained" in numeric_text
        and (
            "RETAIN_ARCHIVE_D0_56_AND_USE_CORRECTED_LOW_30_"
            in numeric_text
        ),
        {"numeric_validator": str(NUMERIC_ROUTE_VALIDATOR_SOURCE)},
    )
    keep_text = KEEP_VALIDATOR_SOURCE.read_text(encoding="utf-8")
    check(
        "keep_route_requires_independent_seed_pass",
        '"seed_pass": seed["status"] == "PASS"' in keep_text
        and 'result["status"] == "PASS"' in keep_text,
        {"validator": str(KEEP_VALIDATOR_SOURCE)},
    )

    compile_run = subprocess.run(
        [
            "python3",
            "-m",
            "py_compile",
            str(PRETRACT_SOURCE),
            str(TRACT_SOURCE),
            str(REPLICATE_SOURCE),
            str(ROUTE_VALIDATOR_SOURCE),
            str(NUMERIC_ROUTE_VALIDATOR_SOURCE),
            str(KEEP_VALIDATOR_SOURCE),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    check(
        "all_recovery4_route_sources_compile",
        compile_run.returncode == 0,
        {
            "returncode": compile_run.returncode,
            "stderr": compile_run.stderr.strip(),
        },
    )

    failed = sorted(
        name for name, result in checks.items() if result["status"] != "PASS"
    )
    report = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_archive_calibration_recovery4_package_validation"
        ),
        "status": "PASS" if not failed else "FAIL",
        "diagnosis_labels_used": False,
        "imaging_executed_by_validation": False,
        "human_visual_qc_inferred": False,
        "total_checks": len(checks),
        "passed_checks": len(checks) - len(failed),
        "failed_checks": failed,
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
