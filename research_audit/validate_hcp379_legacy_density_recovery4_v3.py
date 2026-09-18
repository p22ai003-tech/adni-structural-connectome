#!/home/ec2-user/fsl/bin/python
"""Validate the non-imaging legacy/tensor Recovery4 density package."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PRETRACT_SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "run_hcp379_legacy_density_pretract_recovery4_v3.py"
)
TRACT_SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "run_hcp379_legacy_density_tractography_recovery4_v3.py"
)
ROOT = Path(
    "/data/derivatives/hcp379_v2/corrected_legacy_tensor_recovery4"
)
PRETRACT_PLAN = (
    ROOT
    / "manifests/legacy_tensor_density_pretract_recovery4_plan.json"
)
TRACT_PLAN = (
    ROOT
    / "manifests/legacy_tensor_density_tractography_recovery4_plan.json"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_legacy_density_recovery4_v3/validation.json"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    pretract = load_module(PRETRACT_SOURCE, "legacy_pretract_validator_target")
    tract = load_module(TRACT_SOURCE, "legacy_tract_validator_target")
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
        "exact_216_lane_and_212_execution",
        len(contract["units"]) == 216
        and len(contract["canary_overlap"]) == 4
        and len(contract["execution_units"]) == 212
        and not (
            contract["canary_overlap"] & contract["execution_units"]
        )
        and contract["canary_overlap"] | contract["execution_units"]
        == contract["units"],
        {
            "lane": len(contract["units"]),
            "canary_reuse": len(contract["canary_overlap"]),
            "execution": len(contract["execution_units"]),
        },
    )
    check(
        "exact_density_groups",
        contract["density_group_counts"]
        == {
            "D0_THRESHOLD_QUALIFIED": 2,
            "D1_NEAR_THRESHOLD": 69,
            "D2_MODERATE_LOW": 59,
            "D3_SEVERE_LOW": 68,
            "D4_CRITICAL_LOW": 18,
        },
        contract["density_group_counts"],
    )
    check(
        "pretract_plan_is_non_imaging_and_exact",
        pre_plan.get("status")
        in {
            "PRE_HUMAN_DRY_RUN_PASS",
            "WAITING_FOR_530_HROI_POLICY_ADOPTION",
        }
        and pre_plan.get("execution_requested") is False
        and pre_plan.get("imaging_executed_by_plan") is False
        and pre_plan.get("diagnosis_labels_used") is False
        and pre_plan.get("non_overwriting") is True
        and pre_plan.get("lane_n") == 216
        and pre_plan.get("phase_b_primary_reuse_n") == 4
        and pre_plan.get("new_pretract_execution_n") == 212
        and set(pre_plan.get("execution_units", []))
        == contract["execution_units"],
        pre_plan,
    )
    check(
        "tractography_plan_is_non_imaging_and_exact",
        tract_plan.get("status") == "PRE_PHASE_DRY_RUN_PASS"
        and tract_plan.get("execution_authorized") is False
        and tract_plan.get("imaging_executed_by_plan") is False
        and tract_plan.get("diagnosis_labels_used") is False
        and tract_plan.get("non_overwriting") is True
        and tract_plan.get("lane_n") == 216
        and tract_plan.get("phase_b_primary_reuse_n") == 4
        and tract_plan.get("new_tractography_execution_n") == 212
        and set(tract_plan.get("execution_units", []))
        == contract["execution_units"],
        tract_plan,
    )
    check(
        "hard_density_contract",
        tract_plan.get("minimum_density_inclusive") == 0.60
        and tract_plan.get("required_lane_density_pass_n") == 216
        and tract_plan.get(
            "density_is_whole_cohort_technical_release_gate"
        )
        is True
        and tract_plan.get("density_is_subject_exclusion_gate") is False
        and tract.TRACT.ENGINE.MINIMUM_EDGE_DENSITY == 0.60,
        {
            "plan_minimum": tract_plan.get("minimum_density_inclusive"),
            "required_n": tract_plan.get(
                "required_lane_density_pass_n"
            ),
            "engine_minimum": tract.TRACT.ENGINE.MINIMUM_EDGE_DENSITY,
        },
    )
    check(
        "plan_hash_bindings_current",
        pre_plan["records"]["wrapper"]
        == pretract.PRETRACT.file_record(PRETRACT_SOURCE)
        and tract_plan["records"]["wrapper"]
        == tract.TRACT.PRETRACT.file_record(TRACT_SOURCE)
        and tract_plan["records"]["input_audit"]
        == tract.TRACT.PRETRACT.file_record(tract.AUDIT_CSV),
        {
            "pretract_wrapper": pre_plan["records"]["wrapper"],
            "tract_wrapper": tract_plan["records"]["wrapper"],
        },
    )

    engine = tract.TRACT.ENGINE
    sparse = np.zeros((engine.EXPECTED_NODES, engine.EXPECTED_NODES))
    for index in range(engine.EXPECTED_NODES - 1):
        sparse[index, index + 1] = sparse[index + 1, index] = 1.0
    volumes = np.full(engine.EXPECTED_NODES, 1000.0)
    inverse = 2.0 * sparse / (volumes[:, None] + volumes[None, :])
    with tempfile.TemporaryDirectory(prefix="hcp379-density-negative.") as tmp:
        paths = {}
        for name in engine.MATRIX_NAMES:
            path = Path(tmp) / f"{name}.csv"
            np.savetxt(
                path,
                inverse if name == "count_invnodevol" else sparse,
                delimiter=",",
            )
            paths[name] = path
        _, failures = engine.matrix_qc(
            paths,
            volumes=volumes,
            assigned_streamlines=engine.EXPECTED_NODES - 1,
        )
    check(
        "engine_rejects_below_0_60",
        any(
            str(failure).startswith("edge_density=")
            for failure in failures
        ),
        failures,
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
            "stdout": execute_probe.stdout[-1000:],
            "stderr": execute_probe.stderr[-2000:],
        },
    )
    check(
        "no_pass_or_tractography_output_created_by_validation",
        not (ROOT / "qc/tractography").exists()
        and not any(
            '"status": "PASS_PRETRACT_RECOVERY4"'
            in path.read_text(encoding="utf-8")
            for path in (ROOT / "qc/subjects").glob("*.json")
        ),
        {
            "subjects": (ROOT / "subjects").exists(),
            "pretract_states": (ROOT / "qc/subjects").exists(),
            "tract_states": (ROOT / "qc/tractography").exists(),
        },
    )

    passed = sum(value["status"] == "PASS" for value in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_legacy_density_recovery4_v3_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "imaging_executed": False,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "records": {
            "pretract_plan": tract.TRACT.PRETRACT.file_record(PRETRACT_PLAN),
            "tractography_plan": tract.TRACT.PRETRACT.file_record(TRACT_PLAN),
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
