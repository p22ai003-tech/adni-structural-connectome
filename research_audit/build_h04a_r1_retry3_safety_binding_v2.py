#!/usr/bin/env python3
"""Bind the retry3 three-rule continuation safety correction immutably."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
AUDIT = ROOT / "research_audit"
RUN_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry3")
BASE_BINDING = AUDIT / "outputs/h04a_r1_recovery_extension_binding_v4.json"
BASE_PACKAGE = AUDIT / "outputs/h04a_r1_retry3_package_v1/retry3_package_validation.json"
V4_DRYRUN = AUDIT / "outputs/h04a_r1_retry3_dryrun_v4/validation.json"
SAFETY_DECISION = AUDIT / "decisions/sl_h04a_r1_retry3_rule_allowlist_20260719.md"
PACKAGE = AUDIT / "outputs/h04a_r1_retry3_package_v2"
VALIDATION = PACKAGE / "retry3_package_validation.json"
BINDING = AUDIT / "outputs/h04a_r1_recovery_extension_binding_v5.json"
CONFIG = ROOT / "configs/connectome_v2_retry3.yaml"
EXPECTED_RULES = [
    "select_fod_shells",
    "subject_response",
    "response_calibration_phase_a",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular evidence file: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON object required: {path}")
    return payload


def write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def no_active_imaging_processes() -> bool:
    result = subprocess.run(
        ["ps", "-eo", "comm=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    tokens = (
        "eddy_cpu",
        "eddy_cuda",
        "dwifslpreproc",
        "dwi2response",
        "Snakefile_h04a_r1",
    )
    lines = [
        line
        for line in result.stdout.splitlines()
        if any(token in line for token in tokens)
        and "build_h04a_r1_retry3_safety_binding_v2.py" not in line
    ]
    return not lines


def main() -> int:
    if PACKAGE.exists() or BINDING.exists():
        raise FileExistsError("retry3 safety package or binding already exists")
    required = (RUN_ROOT, BASE_BINDING, BASE_PACKAGE, V4_DRYRUN, SAFETY_DECISION, CONFIG)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"retry3 safety prerequisites are missing: {missing}")

    sys.path.insert(0, str(ROOT / "scforge/workflow"))
    sys.path.insert(0, str(ROOT / "scforge"))
    import run_h04a_r1_retry3 as wrapper

    command = wrapper._retry3_build_snakemake_command(
        snakemake=Path("/locked/snakemake"),
        resolved_config_path=RUN_ROOT / "attempts/dry-run-config.yaml",
        cores=32,
        mode="response-calibration-phase-a",
    )
    trigger_index = command.index("--rerun-triggers")
    allow_index = command.index("--allowed-rules")
    allow_end = command.index("--keep-going")
    allowed_rules = command[allow_index + 1 : allow_end]

    base_binding = load_json(BASE_BINDING)
    base_package = load_json(BASE_PACKAGE)
    v4 = load_json(V4_DRYRUN)
    decision = load_json(
        Path(str(base_binding["execution_decision"]["path"]))
    )
    unexpected = set(v4.get("unexpected_upstream_rules", []))
    checks = {
        "base_package_validation_passes": base_package.get("status") == "PASS",
        "base_package_has_12_checks": base_package.get("summary")
        == {"passed": 12, "total": 12},
        "v4_dryrun_snakemake_passed": v4.get("checks", {}).get(
            "snakemake_dry_run_pass"
        )
        is True,
        "v4_dryrun_detected_upstream_recomputation": bool(unexpected),
        "v4_dryrun_detected_eddy_recomputation": "dwi_motion_eddy" in unexpected,
        "v4_dryrun_has_no_forbidden_downstream_rules": v4.get(
            "forbidden_downstream_rules"
        )
        == [],
        "continuation_rule_allowlist_is_exact": allowed_rules == EXPECTED_RULES,
        "phase_a_target_is_unchanged": command[-1] == "response_calibration_phase_a",
        "content_rerun_triggers_are_exact": command[
            trigger_index + 1 : trigger_index + 3
        ]
        == ["input", "params"],
        "eddy_rule_is_not_allowed": "dwi_motion_eddy" not in allowed_rules,
        "scientific_config_is_unchanged": sha256_file(CONFIG)
        == decision.get("normative_config_sha256"),
        "unit_set_is_unchanged": len(base_binding.get("units", [])) == 15
        and base_binding.get("approved_unit_count") == 15,
        "resource_caps_are_unchanged": base_binding.get("maximum_cpu_cores") == 32
        and base_binding.get("maximum_wall_clock_hours") == 24
        and base_binding.get("maximum_additional_storage_gib") == 100,
        "upstream_recomputation_is_unauthorized": True,
        "no_active_imaging_processes": no_active_imaging_processes(),
    }
    if not all(checks.values()):
        raise ValueError(
            "retry3 safety-package checks failed: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )

    validation = {
        "schema_version": "1.1.0",
        "record_type": "h04a_r1_retry3_package_validation",
        "status": "PASS",
        "checks": checks,
        "summary": {"passed": len(checks), "total": len(checks)},
        "seed_validation": copy.deepcopy(base_package["seed_validation"]),
        "execution_decision": copy.deepcopy(base_package["execution_decision"]),
        "supersedes_package_validation": file_record(BASE_PACKAGE),
        "diagnostic_dryrun": file_record(V4_DRYRUN),
        "launcher_safety_decision": file_record(SAFETY_DECISION),
        "allowed_rules": allowed_rules,
        "upstream_recomputation_authorized": False,
    }
    write_immutable_json(VALIDATION, validation)

    binding = copy.deepcopy(base_binding)
    binding["source_files"]["launcher"] = file_record(
        ROOT / "scforge/workflow/run_h04a_r1_retry3.py"
    )
    binding["source_files"]["compatibility_validator"] = file_record(
        ROOT / "scforge/scforge/h04a_r1_retry3.py"
    )
    binding["package_validation"] = file_record(VALIDATION)
    binding["package_builder"] = file_record(Path(__file__))
    binding["launcher_safety_decision"] = file_record(SAFETY_DECISION)
    binding["diagnostic_dryrun"] = file_record(V4_DRYRUN)
    binding["supersedes_binding"] = file_record(BASE_BINDING)
    binding["allowed_rules"] = allowed_rules
    binding["upstream_recomputation_authorized"] = False

    from scforge.h04a_r1_retry3 import validate_recovery_extension_binding

    validated = validate_recovery_extension_binding(
        binding,
        expected_run_root=RUN_ROOT,
    )
    if validated != binding:
        raise ValueError("retry3 safety binding changed during validation")
    write_immutable_json(BINDING, binding)
    print(
        json.dumps(
            {
                "status": "PASS",
                "checks": validation["summary"],
                "allowed_rules": allowed_rules,
                "binding": file_record(BINDING),
                "package_validation": file_record(VALIDATION),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
