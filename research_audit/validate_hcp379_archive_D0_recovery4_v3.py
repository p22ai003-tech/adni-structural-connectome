#!/home/ec2-user/fsl/bin/python
"""Validate the non-imaging archive-D0 Recovery4 contingency package."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
SUBSET_SUMMARY = (
    EXP
    / "research_audit/outputs/"
    "hcp379_archive_D0_contingency_subset_v3/"
    "archive_D0_contingency_subset_summary.json"
)
PRETRACT_SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_archive_D0_pretract_recovery4_v3.py"
)
TRACT_SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_archive_D0_tractography_recovery4_v3.py"
)
FINAL_SOURCE = (
    EXP / "scripts/hcp/build_hcp379_final_release_v2.py"
)
ROOT = HCP_ROOT / "corrected_archive_D0_recovery4"
PRETRACT_PLAN = (
    ROOT / "manifests/archive_D0_pretract_recovery4_plan.json"
)
TRACT_PLAN = (
    ROOT / "manifests/archive_D0_tractography_recovery4_plan.json"
)
SHARED_TRACT_PLAN = (
    ROOT / "manifests/tractography_recovery4_selected_recipe_plan.json"
)
ARCHIVE_DECISION = (
    HCP_ROOT / "calibration/archive_route_concordance.json"
)
FINAL_READINESS = (
    HCP_ROOT / "release_candidate_v2/hcp379_release_readiness.json"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_archive_D0_recovery4_v3/validation.json"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    pretract = load_module(
        PRETRACT_SOURCE, "archive_D0_pretract_validation_target"
    )
    tract = load_module(
        TRACT_SOURCE, "archive_D0_tract_validation_target"
    )
    final = load_module(
        FINAL_SOURCE, "archive_D0_final_validation_target"
    )
    summary = pretract.PRETRACT.load_json(SUBSET_SUMMARY)
    pre_plan = pretract.PRETRACT.load_json(PRETRACT_PLAN)
    tract_plan = tract.TRACT.PRETRACT.load_json(TRACT_PLAN)
    shared_tract_plan = tract.TRACT.PRETRACT.load_json(
        SHARED_TRACT_PLAN
    )
    contract = pretract.lane_contract()
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if condition else "FAIL",
            "evidence": evidence,
        }

    check(
        "exact_56_D0_lane_4_reuse_52_execution",
        summary.get("status") == "PASS"
        and summary.get("lane_n") == 56
        and summary.get("density_group_counts")
        == {"D0_THRESHOLD_QUALIFIED": 56}
        and summary.get("archive_calibration_primary_reuse_n") == 4
        and summary.get("new_execution_n") == 52
        and summary.get("ready_n") == 52
        and len(contract["reuse_units"]) == 4
        and len(contract["execution_units"]) == 52
        and not contract["reuse_units"] & contract["execution_units"],
        {
            "status": summary.get("status"),
            "density_groups": summary.get("density_group_counts"),
            "reuse_n": len(contract["reuse_units"]),
            "execution_n": len(contract["execution_units"]),
        },
    )
    check(
        "input_selection_is_diagnosis_and_outcome_blind",
        summary.get("diagnosis_labels_used") is False
        and summary.get("diagnosis_or_outcomes_used") is False
        and summary.get("historical_nodes_b0_reuse_allowed") is False
        and summary.get(
            "pretract_may_run_proactively_before_route_decision"
        )
        is True,
        {
            "diagnosis_labels_used": summary.get(
                "diagnosis_labels_used"
            ),
            "diagnosis_or_outcomes_used": summary.get(
                "diagnosis_or_outcomes_used"
            ),
            "proactive_pretract": summary.get(
                "pretract_may_run_proactively_before_route_decision"
            ),
        },
    )
    check(
        "pretract_plan_is_exact_proactive_and_non_imaging",
        pre_plan.get("status")
        in {
            "PRE_HUMAN_DRY_RUN_PASS",
            "WAITING_FOR_530_HROI_POLICY_ADOPTION",
        }
        and pre_plan.get("execution_requested") is False
        and pre_plan.get("imaging_executed_by_plan") is False
        and pre_plan.get("diagnosis_labels_used") is False
        and pre_plan.get("proactive_before_route_decision") is True
        and pre_plan.get("lane_n") == 56
        and pre_plan.get("archive_calibration_primary_reuse_n") == 4
        and pre_plan.get("new_pretract_execution_n") == 52
        and set(pre_plan.get("reuse_units", []))
        == contract["reuse_units"]
        and set(pre_plan.get("execution_units", []))
        == contract["execution_units"],
        {
            "status": pre_plan.get("status"),
            "proactive": pre_plan.get(
                "proactive_before_route_decision"
            ),
            "reuse_n": len(pre_plan.get("reuse_units", [])),
            "execution_n": len(pre_plan.get("execution_units", [])),
        },
    )
    execute_probe = subprocess.run(
        [
            "/home/ec2-user/fsl/bin/python",
            str(PRETRACT_SOURCE),
            "--execute",
            "--workers",
            "1",
            "--nthreads",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    check(
        "missing_genuine_human_qc_is_rejected_before_imaging",
        execute_probe.returncode != 0
        and (
            "genuine human-QC CSV" in execute_probe.stderr
            or "human_visual_qc_recovery4.csv"
            in execute_probe.stderr
            or "HROI pretract-adoption receipt is required"
            in execute_probe.stderr
        ),
        {
            "returncode": execute_probe.returncode,
            "stderr": execute_probe.stderr[-2000:],
        },
    )
    check(
        "tractography_plan_is_exact_prephase_and_non_imaging",
        tract_plan.get("status") == "PRE_PHASE_DRY_RUN_PASS"
        and tract_plan.get("execution_authorized") is False
        and tract_plan.get("imaging_executed_by_plan") is False
        and tract_plan.get("diagnosis_labels_used") is False
        and tract_plan.get("lane_n") == 56
        and tract_plan.get("archive_calibration_primary_reuse_n") == 4
        and tract_plan.get("new_tractography_execution_n") == 52
        and set(tract_plan.get("reuse_units", []))
        == contract["reuse_units"]
        and set(tract_plan.get("execution_units", []))
        == contract["execution_units"],
        {
            "status": tract_plan.get("status"),
            "reuse_n": len(tract_plan.get("reuse_units", [])),
            "execution_n": len(tract_plan.get("execution_units", [])),
        },
    )
    phase_status = shared_tract_plan.get("phase_b_status", {})
    check(
        "phase_b_recipe_is_not_fabricated",
        shared_tract_plan.get("selected_streamline_count") is None
        and bool(phase_status)
        and all(
            value.get("present") is False
            and value.get("status") == "NOT_RUN"
            for value in phase_status.values()
        ),
        {
            "selected_streamline_count": shared_tract_plan.get(
                "selected_streamline_count"
            ),
            "phase_b_status": phase_status,
        },
    )
    check(
        "hard_density_and_nine_matrix_contract",
        tract_plan.get("minimum_density_inclusive") == 0.60
        and tract_plan.get("required_lane_density_pass_n") == 56
        and tract_plan.get(
            "density_is_whole_cohort_technical_release_gate"
        )
        is True
        and tract_plan.get("density_is_subject_exclusion_gate") is False
        and tract.TRACT.ENGINE.MINIMUM_EDGE_DENSITY == 0.60
        and tuple(shared_tract_plan.get("matrix_names", ()))
        == tuple(tract.TRACT.ENGINE.MATRIX_NAMES)
        and len(shared_tract_plan.get("matrix_names", ())) == 9,
        {
            "minimum_density": tract_plan.get(
                "minimum_density_inclusive"
            ),
            "required_lane_n": tract_plan.get(
                "required_lane_density_pass_n"
            ),
            "matrix_names": shared_tract_plan.get("matrix_names"),
        },
    )
    tract.specialize(ROOT)
    tract.TRACT.install_recovery4_adapter()
    try:
        tract.TRACT.validate_full_gate(ROOT)
    except (FileNotFoundError, ValueError) as exc:
        premature_rejected = "pretract" in str(exc).lower()
        premature_evidence = str(exc)
    else:
        premature_rejected = False
        premature_evidence = "unexpectedly accepted"
    check(
        "tractography_rejects_incomplete_pretract",
        premature_rejected,
        premature_evidence,
    )
    check(
        "plans_are_hash_bound_to_current_inputs_and_wrappers",
        pre_plan["records"]["wrapper"]
        == pretract.PRETRACT.file_record(PRETRACT_SOURCE)
        and tract_plan["records"]["wrapper"]
        == tract.TRACT.PRETRACT.file_record(TRACT_SOURCE)
        and pre_plan["records"]["input_subset_summary"]
        == pretract.PRETRACT.file_record(SUBSET_SUMMARY)
        and tract_plan["records"]["input_subset_summary"]
        == tract.TRACT.PRETRACT.file_record(SUBSET_SUMMARY),
        {
            "pretract_wrapper": pre_plan["records"]["wrapper"],
            "tractography_wrapper": tract_plan["records"]["wrapper"],
        },
    )
    compile_probe = subprocess.run(
        [
            "/home/ec2-user/fsl/bin/python",
            "-m",
            "py_compile",
            str(PRETRACT_SOURCE),
            str(TRACT_SOURCE),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    check(
        "wrapper_sources_compile",
        compile_probe.returncode == 0,
        {
            "returncode": compile_probe.returncode,
            "stderr": compile_probe.stderr,
        },
    )
    subject_root = ROOT / "subjects"
    subject_outputs = (
        list(subject_root.iterdir()) if subject_root.is_dir() else []
    )
    check(
        "dry_runs_created_no_subject_or_release_outputs",
        not subject_outputs
        and not tract.EXECUTION_RELEASE_FILENAME
        in {
            path.name
            for path in (ROOT / "manifests").glob("*.json")
        }
        and not tract.COMPOSITE_RELEASE.is_file(),
        {
            "subject_output_n": len(subject_outputs),
            "execution_release_present": (
                ROOT
                / "manifests"
                / tract.EXECUTION_RELEASE_FILENAME
            ).is_file(),
            "composite_release_present": tract.COMPOSITE_RELEASE.is_file(),
        },
    )
    readiness, _ = final.current_readiness()
    check(
        "archive_route_decision_remains_evidence_gated",
        not ARCHIVE_DECISION.is_file()
        and readiness.get("status") == "NOT_READY"
        and "archive_route_concordance_decision_missing"
        in readiness.get("blockers", []),
        {
            "archive_decision_present": ARCHIVE_DECISION.is_file(),
            "readiness_status": readiness.get("status"),
            "blockers": readiness.get("blockers"),
        },
    )

    passed = sum(
        value["status"] == "PASS" for value in checks.values()
    )
    payload = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_archive_D0_recovery4_v3_validation"
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
            "subset_summary": tract.TRACT.PRETRACT.file_record(
                SUBSET_SUMMARY
            ),
            "pretract_plan": tract.TRACT.PRETRACT.file_record(
                PRETRACT_PLAN
            ),
            "tractography_plan": tract.TRACT.PRETRACT.file_record(
                TRACT_PLAN
            ),
            "shared_tractography_plan": (
                tract.TRACT.PRETRACT.file_record(SHARED_TRACT_PLAN)
            ),
            "pretract_wrapper": tract.TRACT.PRETRACT.file_record(
                PRETRACT_SOURCE
            ),
            "tractography_wrapper": (
                tract.TRACT.PRETRACT.file_record(TRACT_SOURCE)
            ),
            "final_readiness": (
                tract.TRACT.PRETRACT.file_record(FINAL_READINESS)
                if FINAL_READINESS.is_file()
                else None
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tract.TRACT.PRETRACT.atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
