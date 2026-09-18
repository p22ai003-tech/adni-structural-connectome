#!/home/ec2-user/fsl/bin/python
"""Validate the isolated HCP379 two-subject mask-union recovery runner."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "run_hcp379_mask_union_pretract_recovery_v1.py"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_mask_union_pretract_recovery_v1/validation.json"
)
EXPECTED_TARGETS = {
    "003_S_0908_I1249292",
    "003_S_4350_I1252856",
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module(SOURCE, "mask_union_recovery_validation_target")
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    compile_result = subprocess.run(
        ["/home/ec2-user/fsl/bin/python", "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    self_test = subprocess.run(
        ["/home/ec2-user/fsl/bin/python", str(SOURCE), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
    )
    add(
        "source_compiles_and_self_test_passes",
        compile_result.returncode == 0
        and self_test.returncode == 0
        and self_test.stdout.strip()
        == "HCP379_MASK_UNION_PRETRACT_RECOVERY_V1_SELF_TEST_PASS",
        {
            "compile_returncode": compile_result.returncode,
            "compile_stderr": compile_result.stderr,
            "self_test_returncode": self_test.returncode,
            "self_test_stdout": self_test.stdout,
            "self_test_stderr": self_test.stderr,
        },
    )

    evidence = module.verify_evidence()
    add(
        "audit_and_independent_replay_are_hash_bound",
        set(evidence)
        == {
            "audit",
            "independent_validation",
            "recovery4_engine",
            "provisional_v2_engine",
        },
        evidence,
    )
    add(
        "shared_engines_are_locked_to_prior_hashes",
        evidence["recovery4_engine"]["sha256"]
        == module.EXPECTED_ENGINE_SHA256
        and evidence["provisional_v2_engine"]["sha256"]
        == module.EXPECTED_V2_SHA256,
        {
            "recovery4": evidence["recovery4_engine"]["sha256"],
            "v2": evidence["provisional_v2_engine"]["sha256"],
        },
    )

    gate, rows = module.selected_rows(module.DEFAULT_HUMAN_QC)
    units = {row["unit"] for row in rows}
    add(
        "exact_two_subject_scope",
        units == EXPECTED_TARGETS and len(rows) == 2,
        {"units": sorted(units), "row_n": len(rows)},
    )
    add(
        "source_route_and_hroi_gate_are_inherited",
        all(
            row["route"] == "READY_FOR_CORRECTED_PRETRACT_FROM_EDDY"
            and row.get("hcp_source_parcellation_policy")
            == "HROI_SURFACE_SUPPORT_V1_COHORT_UNIFORM"
            for row in rows
        )
        and gate.get("hroi_source_policy_status")
        == "PASS_OPERATIONAL_PRETRACT_ADOPTION",
        {
            "routes": sorted({row["route"] for row in rows}),
            "source_policies": sorted(
                {row.get("hcp_source_parcellation_policy") for row in rows}
            ),
            "hroi_status": gate.get("hroi_source_policy_status"),
        },
    )

    with tempfile.TemporaryDirectory() as temporary:
        dry = subprocess.run(
            ["/home/ec2-user/fsl/bin/python", str(SOURCE)],
            capture_output=True,
            text=True,
            check=False,
        )
        plan = json.loads(dry.stdout) if dry.returncode == 0 else {}
    add(
        "default_is_non_imaging_dry_run",
        dry.returncode == 0
        and plan.get("status") == "DRY_RUN_PASS"
        and plan.get("execution_requested") is False
        and plan.get("imaging_executed_by_plan") is False,
        {
            "returncode": dry.returncode,
            "status": plan.get("status"),
            "execution_requested": plan.get("execution_requested"),
            "imaging_executed_by_plan": plan.get(
                "imaging_executed_by_plan"
            ),
            "stderr": dry.stderr,
        },
    )
    policy = plan.get("mask_policy", {})
    add(
        "fixed_uniform_mask_policy",
        policy
        == {
            "name": module.POLICY_NAME,
            "dwi2mask_clean_scale": 0,
            "native_voxel_dilation_passes": 1,
            "original_mask_retained_by_union": True,
            "maximum_mask_volume_ratio": 1.15,
            "per_subject_parameter_tuning_allowed": False,
        },
        policy,
    )
    source_text = SOURCE.read_text(encoding="utf-8")
    add(
        "diagnosis_outcome_density_blind_and_new_namespace",
        plan.get("diagnosis_labels_used") is False
        and plan.get("outcomes_used") is False
        and plan.get("connectome_density_used") is False
        and plan.get("non_overwriting") is True
        and Path(str(plan.get("output_root"))).resolve()
        == module.OUTPUT_ROOT.resolve()
        and "corrected_legacy_tensor_recovery4/subjects" not in source_text,
        {
            "diagnosis_labels_used": plan.get("diagnosis_labels_used"),
            "outcomes_used": plan.get("outcomes_used"),
            "connectome_density_used": plan.get(
                "connectome_density_used"
            ),
            "non_overwriting": plan.get("non_overwriting"),
            "output_root": plan.get("output_root"),
        },
    )

    passed = sum(value["status"] == "PASS" for value in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_mask_union_pretract_recovery_v1_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "imaging_executed": False,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "records": {
            "source": module.PRETRACT.file_record(SOURCE),
        },
    }
    module.PRETRACT.atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
