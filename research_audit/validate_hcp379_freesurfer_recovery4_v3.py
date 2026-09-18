#!/home/ec2-user/fsl/bin/python
"""Hostile-gate validation of the corrected one-subject FS package."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
UNIT = "114_S_6347_I1344943"
ATTESTATION = (
    HCP_ROOT
    / "fastsurfer_repair"
    / UNIT
    / "hcp_source_recovery_attestation_v3.json"
)
INPUT_SUMMARY = (
    EXP
    / "research_audit/outputs/"
    "hcp379_freesurfer_corrected_input_v3/"
    "freesurfer_corrected_input_summary.json"
)
PRETRACT_SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_freesurfer_pretract_recovery4_v3.py"
)
TRACT_SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_freesurfer_tractography_recovery4_v3.py"
)
FINAL_SOURCE = (
    EXP / "scripts/hcp/build_hcp379_final_release_v2.py"
)
OUTPUT_ROOT = HCP_ROOT / "corrected_freesurfer_recovery4"
PRETRACT_PLAN = (
    OUTPUT_ROOT / "manifests/freesurfer_pretract_recovery4_plan.json"
)
TRACT_PLAN = (
    OUTPUT_ROOT
    / "manifests/freesurfer_tractography_recovery4_plan.json"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_freesurfer_recovery4_v3/validation.json"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    pretract = load_module(PRETRACT_SOURCE, "fs_pretract_validation")
    tract = load_module(TRACT_SOURCE, "fs_tract_validation")
    final = load_module(FINAL_SOURCE, "fs_final_validation")
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if condition else "FAIL",
            "evidence": evidence,
        }

    attestation = load_json(ATTESTATION)
    old = attestation.get("old_existing_track_qc", {})
    check(
        "source_is_379_ready_but_old_track_is_rejected",
        attestation.get("status")
        == "PASS_HCP_SOURCE_READY_FOR_CORRECTED_ROUTE"
        and attestation.get("unit") == UNIT
        and attestation.get("atlas_nodes_present") == 379
        and attestation.get("exact_label_set_0_through_379") is True
        and attestation.get("old_existing_track_is_final_release_candidate")
        is False
        and attestation.get("old_existing_track_status")
        == "FAIL_CORRECTED_TRACTOGRAPHY_REQUIRED"
        and float(old.get("edge_density", 1.0)) < 0.60
        and int(old.get("connected_nodes", 379)) < 360,
        {
            "status": attestation.get("status"),
            "atlas_nodes": attestation.get("atlas_nodes_present"),
            "old_density": old.get("edge_density"),
            "old_connected_nodes": old.get("connected_nodes"),
        },
    )
    input_summary = load_json(INPUT_SUMMARY)
    check(
        "corrected_input_is_exact_one_and_diagnosis_blind",
        input_summary.get("status") == "PASS"
        and input_summary.get("target_n") == 1
        and input_summary.get("audited_n") == 1
        and input_summary.get("ready_n") == 1
        and input_summary.get("diagnosis_labels_used") is False
        and input_summary.get("diagnosis_or_outcomes_used") is False
        and input_summary.get("old_existing_track_reuse_allowed") is False
        and input_summary.get("route_counts")
        == {"READY_FOR_CORRECTED_PRETRACT_FROM_EDDY": 1},
        input_summary,
    )

    units = pretract.lane_contract()
    pretract.specialize(OUTPUT_ROOT)
    check(
        "wrapper_scope_is_exact_one",
        units == {UNIT},
        sorted(units),
    )
    pre_plan = load_json(PRETRACT_PLAN)
    check(
        "pretract_plan_is_nonexecuting_and_gated",
        pre_plan.get("status")
        in {
            "PRE_HUMAN_DRY_RUN_PASS",
            "WAITING_FOR_530_HROI_POLICY_ADOPTION",
        }
        and pre_plan.get("execution_requested") is False
        and pre_plan.get("imaging_executed_by_plan") is False
        and pre_plan.get("lane_n") == 1
        and pre_plan.get("new_pretract_execution_n") == 1
        and pre_plan.get("old_existing_track_reuse_allowed") is False,
        {
            "status": pre_plan.get("status"),
            "execution_requested": pre_plan.get("execution_requested"),
        },
    )
    try:
        pretract.PRETRACT.validate_recovery4_gate(
            human_qc=pretract.DEFAULT_HUMAN_QC,
            execute=True,
        )
    except FileNotFoundError as exc:
        human_rejected = (
            "genuine human-QC CSV" in str(exc)
            or "HROI pretract-adoption receipt is required" in str(exc)
        )
        human_evidence = str(exc)
    else:
        human_rejected = False
        human_evidence = "unexpectedly accepted"
    check(
        "execute_rejects_missing_genuine_human_review",
        human_rejected,
        human_evidence,
    )

    tract.specialize(OUTPUT_ROOT)
    tract.TRACT.install_recovery4_adapter()
    tract_plan = load_json(TRACT_PLAN)
    check(
        "tractography_plan_is_prephase_and_density_locked",
        tract_plan.get("status") == "PRE_PHASE_DRY_RUN_PASS"
        and tract_plan.get("execution_authorized") is False
        and tract_plan.get("imaging_executed_by_plan") is False
        and tract_plan.get("lane_n") == 1
        and tract_plan.get("new_tractography_execution_n") == 1
        and tract_plan.get("minimum_density_inclusive") == 0.60
        and tract_plan.get("required_lane_density_pass_n") == 1
        and tract_plan.get("density_is_subject_exclusion_gate") is False
        and tract_plan.get("old_existing_track_reuse_allowed") is False,
        {
            "status": tract_plan.get("status"),
            "minimum_density": tract_plan.get(
                "minimum_density_inclusive"
            ),
        },
    )
    try:
        tract.TRACT.validate_full_gate(OUTPUT_ROOT)
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
    subject_root = OUTPUT_ROOT / "subjects"
    subject_outputs = (
        list(subject_root.iterdir()) if subject_root.is_dir() else []
    )
    check(
        "dry_runs_created_no_passing_subject_imaging",
        not any(
            '"status": "PASS_PRETRACT_RECOVERY4"'
            in path.read_text(encoding="utf-8")
            for path in (OUTPUT_ROOT / "qc/subjects").glob("*.json")
        ),
        {
            "preexisting_subject_output_count": len(subject_outputs),
            "passing_pretract_state_n": sum(
                '"status": "PASS_PRETRACT_RECOVERY4"'
                in path.read_text(encoding="utf-8")
                for path in (OUTPUT_ROOT / "qc/subjects").glob("*.json")
            ),
        },
    )
    source_records = list(pre_plan.get("records", {}).values()) + list(
        tract_plan.get("records", {}).values()
    )
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
    readiness, _ = final.current_readiness()
    check(
        "final_gate_accepts_source_but_still_requires_corrected_output",
        "freesurfer_hcp_source_not_ready"
        not in readiness.get("blockers", [])
        and "corrected_freesurfer_1_release_not_pass"
        in readiness.get("blockers", [])
        and readiness["records"]["freesurfer_hcp_source_attestation"]
        == final.file_record(final.FREESURFER_SOURCE_ATTESTATION),
        readiness.get("blockers"),
    )
    compile_run = subprocess.run(
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
    )
    check(
        "wrapper_sources_compile",
        compile_run.returncode == 0,
        {
            "returncode": compile_run.returncode,
            "stderr": compile_run.stderr.strip(),
        },
    )

    failed = sorted(
        name
        for name, result in checks.items()
        if result["status"] != "PASS"
    )
    report = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_freesurfer_corrected_recovery4_package_validation"
        ),
        "status": "PASS" if not failed else "FAIL",
        "diagnosis_labels_used": False,
        "imaging_executed_by_validation": False,
        "passed_checks": len(checks) - len(failed),
        "total_checks": len(checks),
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
