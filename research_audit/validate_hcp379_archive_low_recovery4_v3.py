#!/home/ec2-user/fsl/bin/python
"""Validate the non-imaging 30-unit archive-low Recovery4 package."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUBSET_SUMMARY = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_archive_low_density_subset_v3/"
    "archive_low_density_subset_summary.json"
)
PRETRACT_SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "run_hcp379_archive_low_pretract_recovery4_v3.py"
)
TRACT_SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "run_hcp379_archive_low_tractography_recovery4_v3.py"
)
ROOT = Path(
    "/data/derivatives/hcp379_v2/corrected_archive_low_recovery4"
)
PRETRACT_PLAN = (
    ROOT / "manifests/archive_low_density_pretract_recovery4_plan.json"
)
TRACT_PLAN = (
    ROOT
    / "manifests/archive_low_density_tractography_recovery4_plan.json"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_archive_low_recovery4_v3/validation.json"
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
        PRETRACT_SOURCE, "archive_low_pretract_validator_target"
    )
    tract = load_module(
        TRACT_SOURCE, "archive_low_tract_validator_target"
    )
    summary = pretract.PRETRACT.load_json(SUBSET_SUMMARY)
    pre_plan = pretract.PRETRACT.load_json(PRETRACT_PLAN)
    tract_plan = tract.TRACT.PRETRACT.load_json(TRACT_PLAN)
    contract = pretract.lane_contract()
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if condition else "FAIL",
            "evidence": evidence,
        }

    check(
        "exact_30_lane_2_reuse_28_execution",
        summary.get("lane_n") == 30
        and summary.get("archive_calibration_primary_reuse_n") == 2
        and summary.get("new_execution_n") == 28
        and len(contract["reuse_units"]) == 2
        and len(contract["execution_units"]) == 28
        and not contract["reuse_units"] & contract["execution_units"],
        {
            "reuse": sorted(contract["reuse_units"]),
            "execution_n": len(contract["execution_units"]),
        },
    )
    check(
        "exact_density_groups",
        summary.get("density_group_counts")
        == {"D1_NEAR_THRESHOLD": 28, "D2_MODERATE_LOW": 2},
        summary.get("density_group_counts"),
    )
    check(
        "input_routes_ready",
        summary.get("status") == "PASS"
        and summary.get("ready_n") == 28
        and summary.get("route_counts")
        == {
            "READY_AFTER_GRADIENT_HEADER_REPAIR": 25,
            "READY_FOR_CORRECTED_PRETRACT_FROM_EDDY": 3,
        },
        {
            "status": summary.get("status"),
            "ready": summary.get("ready_n"),
            "routes": summary.get("route_counts"),
        },
    )
    check(
        "pretract_plan_exact_and_non_imaging",
        pre_plan.get("status")
        in {
            "PRE_HUMAN_DRY_RUN_PASS",
            "WAITING_FOR_530_HROI_POLICY_ADOPTION",
        }
        and pre_plan.get("execution_requested") is False
        and pre_plan.get("imaging_executed_by_plan") is False
        and pre_plan.get("diagnosis_labels_used") is False
        and pre_plan.get("lane_n") == 30
        and pre_plan.get("archive_calibration_primary_reuse_n") == 2
        and pre_plan.get("new_pretract_execution_n") == 28
        and set(pre_plan.get("reuse_units", []))
        == contract["reuse_units"]
        and set(pre_plan.get("execution_units", []))
        == contract["execution_units"],
        pre_plan,
    )
    check(
        "tractography_plan_exact_and_non_imaging",
        tract_plan.get("status") == "PRE_PHASE_DRY_RUN_PASS"
        and tract_plan.get("execution_authorized") is False
        and tract_plan.get("imaging_executed_by_plan") is False
        and tract_plan.get("diagnosis_labels_used") is False
        and tract_plan.get("lane_n") == 30
        and tract_plan.get("archive_calibration_primary_reuse_n") == 2
        and tract_plan.get("new_tractography_execution_n") == 28
        and set(tract_plan.get("reuse_units", []))
        == contract["reuse_units"]
        and set(tract_plan.get("execution_units", []))
        == contract["execution_units"],
        tract_plan,
    )
    check(
        "hard_density_contract",
        tract_plan.get("minimum_density_inclusive") == 0.60
        and tract_plan.get("required_lane_density_pass_n") == 30
        and tract_plan.get(
            "density_is_whole_cohort_technical_release_gate"
        )
        is True
        and tract_plan.get("density_is_subject_exclusion_gate") is False
        and tract.TRACT.ENGINE.MINIMUM_EDGE_DENSITY == 0.60,
        {
            "minimum": tract_plan.get("minimum_density_inclusive"),
            "required": tract_plan.get(
                "required_lane_density_pass_n"
            ),
            "engine": tract.TRACT.ENGINE.MINIMUM_EDGE_DENSITY,
        },
    )
    check(
        "hash_bindings_current",
        pre_plan["records"]["wrapper"]
        == pretract.PRETRACT.file_record(PRETRACT_SOURCE)
        and tract_plan["records"]["wrapper"]
        == tract.TRACT.PRETRACT.file_record(TRACT_SOURCE)
        and tract_plan["records"]["input_subset_summary"]
        == tract.TRACT.PRETRACT.file_record(SUBSET_SUMMARY),
        {
            "pretract_wrapper": pre_plan["records"]["wrapper"],
            "tract_wrapper": tract_plan["records"]["wrapper"],
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
        "missing_human_qc_rejected_before_imaging",
        execute_probe.returncode != 0
        and (
            "human_visual_qc_recovery4.csv" in execute_probe.stderr
            or "No such file" in execute_probe.stderr
            or "HROI pretract-adoption receipt is required"
            in execute_probe.stderr
        ),
        {
            "returncode": execute_probe.returncode,
            "stderr": execute_probe.stderr[-2000:],
        },
    )
    check(
        "no_subject_outputs_created",
        not (ROOT / "subjects").exists()
        and not (ROOT / "qc/subjects").exists()
        and not (ROOT / "qc/tractography").exists()
        and not tract.COMPOSITE_RELEASE.exists(),
        {
            "subjects": (ROOT / "subjects").exists(),
            "pretract_states": (ROOT / "qc/subjects").exists(),
            "tract_states": (ROOT / "qc/tractography").exists(),
            "composite_release": tract.COMPOSITE_RELEASE.exists(),
        },
    )

    passed = sum(value["status"] == "PASS" for value in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_archive_low_recovery4_v3_validation",
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
            "pretract_plan": tract.TRACT.PRETRACT.file_record(PRETRACT_PLAN),
            "tractography_plan": tract.TRACT.PRETRACT.file_record(
                TRACT_PLAN
            ),
            "pretract_wrapper": tract.TRACT.PRETRACT.file_record(
                PRETRACT_SOURCE
            ),
            "tractography_wrapper": tract.TRACT.PRETRACT.file_record(
                TRACT_SOURCE
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tract.TRACT.PRETRACT.atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
