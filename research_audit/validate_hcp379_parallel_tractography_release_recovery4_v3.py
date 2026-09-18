#!/home/ec2-user/fsl/bin/python
"""Hostile/static validation for the Recovery4 downstream supervisor."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_parallel_tractography_release_recovery4_v3.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_parallel_tractography_release_recovery4_v3/"
    "validation.json"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_parallel_tractography_release_contract", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module(SOURCE)
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if condition else "FAIL",
            "evidence": evidence,
        }

    compile_result = subprocess.run(
        [str(PYTHON), "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "source_compiles",
        compile_result.returncode == 0,
        {
            "returncode": compile_result.returncode,
            "stderr": compile_result.stderr.strip(),
        },
    )

    self_test = subprocess.run(
        [str(PYTHON), str(SOURCE), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
    )
    self_payload = (
        json.loads(self_test.stdout)
        if self_test.returncode == 0 and self_test.stdout.strip()
        else {}
    )
    check(
        "graph_self_test_passes",
        self_test.returncode == 0
        and self_payload.get("status") == "PASS"
        and self_payload.get("passed") == 15
        and self_payload.get("total") == 15,
        {
            "returncode": self_test.returncode,
            "payload": self_payload,
            "stderr": self_test.stderr.strip(),
        },
    )

    preflight = subprocess.run(
        [str(PYTHON), str(SOURCE), "--preflight"],
        capture_output=True,
        text=True,
        check=False,
    )
    plan = module.load_json(module.PLAN)
    check(
        "preflight_is_non_imaging",
        preflight.returncode == 0
        and plan.get("record_type")
        == "diagnosis_blind_hcp379_parallel_tractography_release_plan"
        and plan.get("execution_authorized_by_plan") is False
        and plan.get("imaging_executed_by_plan") is False
        and plan.get("diagnosis_or_outcomes_used") is False
        and plan.get("non_overwriting") is True,
        {
            "returncode": preflight.returncode,
            "status": plan.get("status"),
            "stderr": preflight.stderr.strip(),
        },
    )

    graph = module.tasks()
    check(
        "exact_release_graph",
        len(graph) == 12
        and 227 + 216 + 1 + 30 + 56 == 530
        and set(graph["final_release_530"].dependencies)
        == {
            "core_227",
            "legacy_tensor_216",
            "freesurfer_1",
            "archive_low_30",
            "archive_D0_56",
            "archive_concordance",
            "corrected_archive_86",
        }
        and graph["verify_integration_530"].dependencies
        == ("final_release_530",)
        and graph["prepare_analysis_handoff_530"].dependencies
        == ("verify_integration_530",),
        {
            "tasks": sorted(graph),
            "final_dependencies": list(
                graph["final_release_530"].dependencies
            ),
            "verification_dependencies": list(
                graph["verify_integration_530"].dependencies
            ),
            "handoff_dependencies": list(
                graph["prepare_analysis_handoff_530"].dependencies
            ),
        },
    )
    check(
        "archive_calibration_and_route_order",
        graph["archive_replicate_6"].dependencies
        == ("archive_primary_6",)
        and set(graph["archive_concordance"].dependencies)
        == {"archive_primary_6", "archive_replicate_6"}
        and set(graph["corrected_archive_86"].dependencies)
        == {"archive_low_30", "archive_D0_56"},
        {
            name: list(graph[name].dependencies)
            for name in (
                "archive_replicate_6",
                "archive_concordance",
                "corrected_archive_86",
            )
        },
    )
    check(
        "resource_and_compaction_contract",
        plan.get("resources", {}).get("maximum_active_compute_tasks")
        == 3
        and plan.get("resources", {}).get(
            "subject_workers_per_compute_task"
        )
        == 1
        and plan.get("resources", {}).get(
            "declared_peak_tractography_threads"
        )
        == 48
        and plan.get("resources", {}).get(
            "mandatory_compaction_after_pass"
        )
        is True
        and "zero tractography workers"
        in plan.get("resources", {}).get(
            "cross_supervisor_cpu_guard", ""
        )
        and all(
            (
                task.name == "archive_replicate_6"
                or "--compact-tractograms-after-pass" in task.command
            )
            for task in graph.values()
            if task.compute_task
        ),
        plan.get("resources"),
    )
    check(
        "no_overwrite_or_subject_tuning",
        all("--overwrite" not in task.command for task in graph.values())
        and all(
            (
                not task.compute_task
                or task.command[
                    task.command.index("--workers") + 1
                ]
                == "1"
            )
            for task in graph.values()
        )
        and plan.get("release_contract", {}).get(
            "per_subject_streamline_tuning_allowed"
        )
        is False
        and plan.get("release_contract", {}).get(
            "edge_imputation_allowed"
        )
        is False
        and plan.get("release_contract", {}).get(
            "subject_exclusion_allowed"
        )
        is False
        and plan.get("release_contract", {}).get(
            "independent_post_assembly_verification_required"
        )
        is True
        and plan.get("release_contract", {}).get(
            "immutable_verified_analysis_handoff_required"
        )
        is True
        and plan.get("release_contract", {}).get(
            "legacy_dashboard_switch_authorized"
        )
        is False,
        {
            "commands": {
                name: list(task.command)
                for name, task in graph.items()
            },
            "release_contract": plan.get("release_contract"),
        },
    )
    check(
        "source_hashes_current",
        all(
            record == module.file_record(Path(record["path"]))
            for record in plan.get("sources", {}).values()
        )
        and plan.get("sources", {}).get(SOURCE.name)
        == module.file_record(SOURCE),
        plan.get("sources"),
    )

    attempts_before = (
        {path.name for path in module.ATTEMPT_ROOT.iterdir()}
        if module.ATTEMPT_ROOT.is_dir()
        else set()
    )
    state_before = (
        module.STATE.read_bytes() if module.STATE.is_file() else None
    )
    with tempfile.TemporaryDirectory(
        prefix="hcp379-missing-human-qc."
    ) as temporary:
        missing = Path(temporary) / "does-not-exist.csv"
        denied = subprocess.run(
            [
                str(PYTHON),
                str(SOURCE),
                "--execute",
                "--human-qc",
                str(missing),
                "--upstream-timeout-hours",
                "0.01",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    attempts_after = (
        {path.name for path in module.ATTEMPT_ROOT.iterdir()}
        if module.ATTEMPT_ROOT.is_dir()
        else set()
    )
    state_after = (
        module.STATE.read_bytes() if module.STATE.is_file() else None
    )
    check(
        "missing_human_qc_fails_before_attempt",
        denied.returncode != 0
        and attempts_after == attempts_before
        and state_after == state_before,
        {
            "returncode": denied.returncode,
            "new_attempts": sorted(attempts_after - attempts_before),
            "state_changed": state_after != state_before,
            "stderr_tail": denied.stderr.strip().splitlines()[-3:],
        },
    )

    original_pretract_attempt_root = module.PRETRACT_ATTEMPT_ROOT
    with tempfile.TemporaryDirectory(
        prefix="hcp379-cross-supervisor."
    ) as temporary:
        module.PRETRACT_ATTEMPT_ROOT = Path(temporary)
        state_path = (
            module.PRETRACT_ATTEMPT_ROOT
            / "attempt"
            / "supervisor_state.json"
        )
        common = {
            "record_type": (
                "diagnosis_blind_hcp379_parallel_pretract_supervisor"
            ),
            "pretract": {"active": {}},
        }
        module.atomic_json(
            state_path,
            {
                **common,
                "status": "PHASE_B_AND_PRETRACT_RUNNING",
                "phase_b": {"returncode": None},
            },
        )
        phase_live = module.cross_supervisor_compute_limit(3)
        module.atomic_json(
            state_path,
            {
                **common,
                "status": "PHASE_B_AND_PRETRACT_RUNNING",
                "phase_b": {"returncode": 0},
            },
        )
        pretract_live = module.cross_supervisor_compute_limit(3)
        module.atomic_json(
            state_path,
            {
                **common,
                "status": "PASS_PHASE_B_AND_ALL_PRETRACT",
                "phase_b": {"returncode": 0},
            },
        )
        pretract_done = module.cross_supervisor_compute_limit(3)
    module.PRETRACT_ATTEMPT_ROOT = original_pretract_attempt_root
    check(
        "cross_supervisor_cpu_guard",
        phase_live["limit"] == 0
        and pretract_live["limit"] == 2
        and pretract_done["limit"] == 3,
        {
            "phase_live": phase_live,
            "pretract_live": pretract_live,
            "pretract_done": pretract_done,
        },
    )

    selected = 5_000_000
    artifact_stub = {
        key: {
            "path": f"/synthetic/{key}.csv",
            "sha256": "0" * 64,
            "size_bytes": 1,
        }
        for key in module.MATRIX_ARTIFACT_KEYS
    }
    with tempfile.TemporaryDirectory(
        prefix="hcp379-supervisor-hostile."
    ) as temporary:
        root = Path(temporary)
        core_path = root / "core.json"
        core_task = replace(graph["core_227"], output=core_path)
        core_manifest = {
            "record_type": core_task.record_type,
            "status": "PASS",
            "diagnosis_labels_used": False,
            "unit_count": 227,
            "selected_streamline_count": selected,
            "density_is_whole_cohort_technical_release_gate": True,
            "minimum_edge_density_inclusive": 0.60,
            "subjects_at_or_above_minimum_density": 227,
            "minimum_observed_edge_density": 0.61,
            "units": [
                {
                    "unit": f"synthetic-{index:03d}",
                    "edge_density": 0.61,
                    "artifacts": artifact_stub,
                }
                for index in range(227)
            ],
        }
        core_path.write_text(
            json.dumps(core_manifest), encoding="utf-8"
        )
        positive = module.output_status(core_task, selected)
        sparse_manifest = copy.deepcopy(core_manifest)
        sparse_manifest["units"][11]["edge_density"] = 0.59
        core_path.write_text(
            json.dumps(sparse_manifest), encoding="utf-8"
        )
        sparse = module.output_status(core_task, selected)
        missing_matrix_manifest = copy.deepcopy(core_manifest)
        missing_matrix_manifest["units"][13]["artifacts"].pop(
            "matrix_ad_mean"
        )
        core_path.write_text(
            json.dumps(missing_matrix_manifest), encoding="utf-8"
        )
        missing_matrix = module.output_status(core_task, selected)
        wrong_count_manifest = copy.deepcopy(core_manifest)
        wrong_count_manifest["selected_streamline_count"] = 3_000_000
        core_path.write_text(
            json.dumps(wrong_count_manifest), encoding="utf-8"
        )
        wrong_count = module.output_status(core_task, selected)
    check(
        "terminal_manifest_hostile_checks",
        positive["valid"]
        and not sparse["valid"]
        and not missing_matrix["valid"]
        and not wrong_count["valid"],
        {
            "positive": positive,
            "sparse": sparse,
            "missing_matrix": missing_matrix,
            "wrong_selected_count": wrong_count,
        },
    )

    with tempfile.TemporaryDirectory(
        prefix="hcp379-final-hostile."
    ) as temporary:
        final_path = Path(temporary) / "release.json"
        final_task = replace(
            graph["final_release_530"], output=final_path
        )
        final_manifest = {
            "record_type": final_task.record_type,
            "status": "PASS",
            "subject_count": 530,
            "matrix_family_count": 9,
            "matrix_file_count": 4770,
            "matrix_shape": [379, 379],
            "diagnosis_or_outcome_fields_present": False,
            "density_is_whole_cohort_technical_release_gate": True,
            "subjects_at_or_above_minimum_density": 530,
            "minimum_observed_edge_density": 0.60,
        }
        final_path.write_text(
            json.dumps(final_manifest), encoding="utf-8"
        )
        final_positive = module.output_status(final_task, selected)
        final_manifest["subject_count"] = 529
        final_path.write_text(
            json.dumps(final_manifest), encoding="utf-8"
        )
        final_short = module.output_status(final_task, selected)
    check(
        "final_release_contract_is_exact",
        final_positive["valid"] and not final_short["valid"],
        {
            "positive": final_positive,
            "short": final_short,
        },
    )

    with tempfile.TemporaryDirectory(
        prefix="hcp379-integration-hostile."
    ) as temporary:
        integration_path = Path(temporary) / "verification.json"
        integration_task = replace(
            graph["verify_integration_530"], output=integration_path
        )
        integration_manifest = {
            "record_type": integration_task.record_type,
            "status": "PASS",
            "verification_is_independent_of_assembler_imports": True,
            "diagnosis_or_outcome_fields_present": False,
            "subject_count": 530,
            "matrix_family_count": 9,
            "matrix_file_count": 4770,
            "matrix_shape": [379, 379],
            "selected_streamline_count": selected,
            "subjects_at_or_above_minimum_density": 530,
            "minimum_observed_edge_density": 0.60,
            "subjects_at_or_above_minimum_weighted_support": 530,
            "minimum_observed_weighted_support": 0.95,
            "subjects_at_or_above_minimum_assignment_fraction": 530,
            "minimum_observed_assignment_fraction": 0.50,
        }
        integration_path.write_text(
            json.dumps(integration_manifest), encoding="utf-8"
        )
        integration_positive = module.output_status(
            integration_task, selected
        )
        integration_manifest[
            "subjects_at_or_above_minimum_weighted_support"
        ] = 529
        integration_path.write_text(
            json.dumps(integration_manifest), encoding="utf-8"
        )
        integration_short = module.output_status(
            integration_task, selected
        )
    check(
        "integration_verification_contract_is_exact",
        integration_positive["valid"]
        and not integration_short["valid"],
        {
            "positive": integration_positive,
            "short": integration_short,
        },
    )

    original_hcp_root = module.HCP_ROOT
    with tempfile.TemporaryDirectory(
        prefix="hcp379-handoff-hostile."
    ) as temporary:
        root = Path(temporary)
        module.HCP_ROOT = root
        verification_path = (
            root
            / "release_validation_v1/"
            "integration_verification.json"
        )
        module.atomic_json(
            verification_path,
            {
                "record_type": (
                    "diagnosis_blind_hcp379_integration_"
                    "release_verification"
                ),
                "status": "PASS",
            },
        )
        handoff_root = root / "analysis_handoff_v1/content"
        contract_path = handoff_root / "analysis_contract.json"
        handoff_path = handoff_root / "handoff_manifest.json"
        receipt_path = (
            root
            / "analysis_handoff_v1/"
            "verified_handoff_receipt.json"
        )
        contract = {
            "record_type": "hcp379_v2_verified_analysis_contract",
            "status": "PASS",
            "primary_key": "unit",
            "node_count": 379,
            "selected_streamline_count": selected,
            "source_route_field_required": True,
            "route_sensitivity_required": True,
            "legacy_AAL_dashboard_activation_authorized": False,
        }
        module.atomic_json(contract_path, contract)
        handoff = {
            "record_type": "hcp379_v2_verified_analysis_handoff",
            "status": "PASS",
            "all_matrix_files_rehashed_during_handoff": True,
            "subject_count": 530,
            "matrix_file_count": 4770,
            "live_dashboard_modified": False,
            "live_dashboard_activation_authorized": False,
            "records": {
                "analysis_contract": module.file_record(contract_path)
            },
        }
        module.atomic_json(handoff_path, handoff)
        receipt = {
            "record_type": (
                "hcp379_v2_verified_analysis_handoff_receipt"
            ),
            "status": "PASS",
            "subject_count": 530,
            "matrix_family_count": 9,
            "matrix_file_count": 4770,
            "matrix_shape": [379, 379],
            "diagnosis_or_outcome_fields_present": False,
            "live_dashboard_modified": False,
            "records": {
                "independent_verification": module.file_record(
                    verification_path
                ),
                "handoff_manifest": module.file_record(handoff_path),
            },
        }
        module.atomic_json(receipt_path, receipt)
        handoff_task = replace(
            graph["prepare_analysis_handoff_530"],
            output=receipt_path,
        )
        handoff_positive = module.output_status(
            handoff_task, selected
        )
        contract["selected_streamline_count"] = 3_000_000
        module.atomic_json(contract_path, contract)
        handoff["records"]["analysis_contract"] = module.file_record(
            contract_path
        )
        module.atomic_json(handoff_path, handoff)
        receipt["records"]["handoff_manifest"] = module.file_record(
            handoff_path
        )
        module.atomic_json(receipt_path, receipt)
        handoff_wrong_count = module.output_status(
            handoff_task, selected
        )
        contract["selected_streamline_count"] = selected
        module.atomic_json(contract_path, contract)
        handoff["records"]["analysis_contract"] = module.file_record(
            contract_path
        )
        handoff["live_dashboard_activation_authorized"] = True
        module.atomic_json(handoff_path, handoff)
        receipt["records"]["handoff_manifest"] = module.file_record(
            handoff_path
        )
        module.atomic_json(receipt_path, receipt)
        handoff_activation = module.output_status(
            handoff_task, selected
        )
    module.HCP_ROOT = original_hcp_root
    check(
        "verified_handoff_contract_is_terminal_and_fail_closed",
        handoff_positive["valid"]
        and not handoff_wrong_count["valid"]
        and not handoff_activation["valid"],
        {
            "positive": handoff_positive,
            "wrong_selected_count": handoff_wrong_count,
            "activation_authorized": handoff_activation,
        },
    )

    check(
        "current_plan_stays_gated_without_real_qc",
        (
            plan.get("human_qc", {}).get("ready") is True
            or plan.get("status") == "AWAITING_GENUINE_HUMAN_QC"
        )
        and (
            plan.get("phase_b", {}).get("ready") is True
            or plan.get("selected_streamline_count") is None
        ),
        {
            "status": plan.get("status"),
            "human_qc": plan.get("human_qc"),
            "phase_b": plan.get("phase_b"),
        },
    )

    passed = sum(
        value["status"] == "PASS" for value in checks.values()
    )
    payload = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_parallel_tractography_release_recovery4_"
            "package_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": module.utc_now(),
        "imaging_executed": False,
        "human_visual_qc_inferred": False,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "source": module.file_record(SOURCE),
        "plan": module.file_record(module.PLAN),
    }
    module.atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
