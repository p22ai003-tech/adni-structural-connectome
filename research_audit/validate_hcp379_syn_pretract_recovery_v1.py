#!/usr/bin/env python3
"""Validate deterministic SyN recovery after three failed spatial routes."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "run_hcp379_syn_pretract_recovery_v1.py"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_syn_pretract_recovery_v1/validation.json"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_syn_recovery_validation_target", SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module()
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    compile_probe = subprocess.run(
        [str(module.PYTHON), "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    self_test = subprocess.run(
        [str(module.PYTHON), str(SOURCE), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    check(
        "source_compiles_and_self_test_passes",
        compile_probe.returncode == 0
        and self_test.returncode == 0
        and "HCP379_SYN_PRETRACT_RECOVERY_SELF_TEST_PASS"
        in self_test.stdout,
        {
            "compile_returncode": compile_probe.returncode,
            "compile_stderr": compile_probe.stderr,
            "self_test_returncode": self_test.returncode,
            "self_test_stdout": self_test.stdout,
        },
    )
    preflight = module.preflight(module.DEFAULT_HUMAN_QC)
    targets = preflight.get("target_units", [])
    check(
        "targets_are_exact_post_affine_failures",
        (
            preflight.get("status")
            in {
                "READY_FOR_EXACT_SYN_RECOVERY",
                "NO_CURRENT_SYN_RECOVERY_TARGETS",
            }
            and preflight.get("failure_class") == module.FAILURE_CLASS
            and preflight.get("target_n") == len(targets)
            and len(set(targets)) == len(targets)
        ),
        {
            "status": preflight.get("status"),
            "target_n": preflight.get("target_n"),
            "target_units": targets,
        },
    )
    policy = preflight.get("policy", {})
    check(
        "syn_is_fourth_route_with_positive_jacobian_gate",
        (
            policy.get("selection")
            == "only after BBR, seeded rigid and seeded affine fail"
            and policy.get("transform") == "seeded_ants_syn"
            and policy.get("ants_transform_type") == "s"
            and policy.get("random_seed") == 1234
            and policy.get(
                "all_jacobian_determinants_must_be_positive"
            )
            is True
            and policy.get("per_subject_tuning_allowed") is False
            and policy.get("engine_recomputation_allowed") is False
            and policy.get("tissue_qc_threshold_change_allowed") is False
            and policy.get("atlas_qc_threshold_change_allowed") is False
            and policy.get("visual_qc_required") is True
        ),
        policy,
    )
    engine_text = module.ENGINE_SOURCE.read_text(encoding="utf-8")
    check(
        "engine_has_disjoint_syn_namespace_and_deformation_qc",
        all(
            fragment in engine_text
            for fragment in (
                "def ants_syn_variant_paths",
                "recovery4_ants_syn",
                'transform_type="s"',
                'route_name="seeded_ants_syn_recovery"',
                "CreateJacobianDeterminantImage",
                "nonpositive_syn_jacobian",
            )
        ),
        {"required_fragments_present": True},
    )
    check(
        "preflight_is_non_imaging_and_diagnosis_blind",
        (
            preflight.get("diagnosis_labels_used") is False
            and preflight.get("outcomes_used") is False
            and preflight.get("connectome_density_used") is False
            and preflight.get("non_overwriting") is True
            and preflight.get("imaging_executed") is False
            and preflight.get("tractography_generated") is False
            and preflight.get("matrix_generated") is False
        ),
        {
            key: preflight.get(key)
            for key in (
                "diagnosis_labels_used",
                "outcomes_used",
                "connectome_density_used",
                "non_overwriting",
                "imaging_executed",
                "tractography_generated",
                "matrix_generated",
            )
        },
    )
    passed = sum(row["status"] == "PASS" for row in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_syn_pretract_recovery_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
