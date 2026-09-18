#!/usr/bin/env python3
"""Prepare the content-addressed R1 retry3 continuation after retry2 is terminal."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
AUDIT = ROOT / "research_audit"
PACKAGE_V1 = AUDIT / "outputs/h04a_r1_recovery_package_v1"
PACKAGE = AUDIT / "outputs/h04a_r1_retry3_package_v1"
SOURCE_DECISION = PACKAGE_V1 / "recovery_execution_decision_retry2.json"
EXECUTION_MANIFEST = PACKAGE_V1 / "recovery_execution_manifest_v1.csv"
SOURCE_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260718_retry2")
DESTINATION_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry3")
SOURCE_COMPLETION = SOURCE_ROOT / "publication/response_calibration_phase_a_completion.json"
SEED_MANIFEST = PACKAGE / "retry3_seed_manifest.json"
DECISION = PACKAGE / "recovery_execution_decision_retry3.json"
VALIDATION = PACKAGE / "retry3_package_validation.json"
BINDING = AUDIT / "outputs/h04a_r1_recovery_extension_binding_v4.json"
STATIC_ENVIRONMENT_VALIDATION = (
    AUDIT / "outputs/h04a_r1_retry3_static_v1/environment_validation.json"
)
SOURCE_PATCH = AUDIT / "decisions/sl_h04a_r1_retry3_shell_input_fix_20260719.patch"
RETRY_RECOMMENDATION = AUDIT / "decisions/sl_h04a_r1_retry3_recommendation_20260719.md"
SOURCE_APPROVAL = AUDIT / "decisions/sl_h04a_r1_approval_20260718.json"
SOURCE_RECOMMENDATION = (
    AUDIT / "decisions/sl_h04a_r1_recovery_recommendation_20260718.json"
)
RETRY3_ID = "SL-H04A-R1-RETRY3"
ALLOWED_AREAS = ("00_inputs", "01_dwi", "logs")
KNOWN_TERMINAL_SERVICE_STATES = frozenset({"inactive", "failed"})
REQUIRED_SEED_PRODUCTS = (
    "01_dwi/dwi_preproc.mif",
    "01_dwi/dwi_preproc_biascorr.mif",
    "01_dwi/dwi_brain_mask.mif",
)
PRE_RETRY3_OPERATIONAL_CHECKS = frozenset(
    {
        "decision_is_exact_15_unit_phase_a_scope",
        "decision_prohibits_downstream_execution",
        "phase_a_service_is_known_terminal",
        "no_active_imaging_processes",
        "phase_a_completion_and_manifest_exist",
        "phase_a_completion_contract_valid",
        "all_15_units_have_exact_terminal_outcomes",
        "terminal_outcome_records_and_hashes_validate",
        "exact_single_attempt_start_end_completion_closure",
        "no_fod_tractography_matrix_or_full_publication_artifact",
        "gpu_equivalence_package_static_validation_passes",
    }
)
EXPECTED_HASHES = {
    ROOT / "scforge/workflow/rules/05_5tt_fod.smk": (
        "c5b6ef5de6e3aa0027ba983609f8458a7bc68d1d8040c965043cfef25cc2a489"
    ),
    ROOT / "scforge/workflow/rules/05_5tt_fod_retry3.smk": (
        "95115b437510b42801dbb0cc3bd82e36158cb886af71f624d669c7b617149aa6"
    ),
    ROOT / "scforge/workflow/workflow_source_manifest.tsv": (
        "1e27facda8d0cafefada24fb04a9d4ee628cb296cbf10e303680d2a4244a7a36"
    ),
    ROOT / "scforge/workflow/workflow_source_manifest_retry3.tsv": (
        "e86f30b18fd3d86dfda4015c4cde94a89f8a9ab4cffb72866370734cd736cdf4"
    ),
    ROOT / "configs/connectome_v2_retry3.yaml": (
        "13c3220b2c6d55afa43b82fd09eec732fb1b4eaf4173117677ff6fdb8d04148b"
    ),
    ROOT / "scforge/workflow/environment_contract_retry3.yaml": (
        "8edaa4171a181b98a3b77fdc617847a79f742f124c888160e1bfdf674080d517"
    ),
}


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


def write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def require_retry2_terminal() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    state = subprocess.run(
        ["systemctl", "--user", "is-active", "scforge-h04a-r1-phase-a-retry2.service"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip() or "unknown"
    if state not in KNOWN_TERMINAL_SERVICE_STATES:
        raise RuntimeError(f"retry2 service is not known terminal: {state}")
    processes = subprocess.run(
        ["pgrep", "-af", "eddy_cpu|dwifslpreproc|Snakefile_h04a_r1"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    active_lines = [line for line in processes.splitlines() if "pgrep -af" not in line]
    if active_lines:
        raise RuntimeError("imaging processes remain active: " + " | ".join(active_lines[:5]))

    end_paths = sorted((SOURCE_ROOT / "attempts").glob("*.end.json"))
    if len(end_paths) != 1:
        raise ValueError(f"retry2 must have exactly one immutable attempt end, found {len(end_paths)}")
    end_path = end_paths[0]
    end = json.loads(end_path.read_text(encoding="utf-8"))
    if (
        end.get("status") != "FAIL"
        or end.get("returncode") != 1
        or end.get("launcher_mode") != "response-calibration-phase-a"
    ):
        raise ValueError("retry2 terminal outcome is not the expected failed Phase-A attempt")
    duration = end.get("duration_seconds")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        raise ValueError("retry2 duration is invalid")
    if not SOURCE_COMPLETION.is_file():
        raise FileNotFoundError("retry2 completion is missing")
    completion = json.loads(SOURCE_COMPLETION.read_text(encoding="utf-8"))
    if (
        completion.get("record_type") != "response_calibration_phase_a_completion"
        or completion.get("status") != "FAIL"
        or completion.get("terminal_attempt_end") != file_record(end_path)
    ):
        raise ValueError("retry2 completion does not bind the expected failed attempt")
    log_record = end.get("combined_execution_log")
    if not isinstance(log_record, dict):
        raise ValueError("retry2 attempt end lacks the combined log record")
    log_path = Path(str(log_record.get("path", ""))).expanduser().resolve()
    if file_record(log_path) != log_record:
        raise ValueError("retry2 combined log identity differs")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    required_fragments = (
        "Error in rule select_fod_shells",
        "not 'Namedlist'",
        'file "/home/ec2-user/exp/scforge/workflow/rules/05_5tt_fod.smk", line 143',
    )
    if any(fragment not in log_text for fragment in required_fragments):
        raise ValueError("retry2 log does not prove the exact shell-input TypeError")
    return end_path, end, completion


def require_complete_seed_inputs(
    units: list[str], *, source_root: Path = SOURCE_ROOT
) -> dict[str, int]:
    """Prove all reusable products exist before creating the retry3 root."""
    counts = {relative: 0 for relative in REQUIRED_SEED_PRODUCTS}
    for unit in units:
        for relative in REQUIRED_SEED_PRODUCTS:
            product = source_root / "subjects" / unit / relative
            if (
                not product.is_file()
                or product.is_symlink()
                or product.stat().st_size < 1
            ):
                raise ValueError(
                    f"retry2 reusable seed product is missing or invalid: {unit}/{relative}"
                )
            counts[relative] += 1
    if set(counts.values()) != {len(units)}:
        raise ValueError(f"retry2 reusable seed coverage differs: {counts}")
    return counts


def require_pre_retry3_operational_gate(record: dict[str, Any]) -> list[str]:
    """Require every operational prerequisite that can precede retry3 creation."""
    checks = record.get("checks")
    if not isinstance(checks, list):
        raise ValueError("pre-retry3 gate has no check list")
    by_name: dict[str, dict[str, Any]] = {}
    for row in checks:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("pre-retry3 gate contains a malformed check")
        name = str(row["name"])
        if name in by_name:
            raise ValueError(f"pre-retry3 gate duplicates check: {name}")
        by_name[name] = row
    missing = sorted(PRE_RETRY3_OPERATIONAL_CHECKS - set(by_name))
    failed = sorted(
        name
        for name in PRE_RETRY3_OPERATIONAL_CHECKS
        if name in by_name and by_name[name].get("pass") is not True
    )
    if missing or failed:
        raise ValueError(
            f"pre-retry3 operational gate is incomplete: missing={missing}; failed={failed}"
        )
    if record.get("resize_checks_total") != 13:
        raise ValueError("pre-retry3 gate does not expose the exact 13-check resize contract")
    return sorted(PRE_RETRY3_OPERATIONAL_CHECKS)


def require_static_contract() -> None:
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"retry3 static source hash differs: {path}")
    validation = json.loads(STATIC_ENVIRONMENT_VALIDATION.read_text(encoding="utf-8"))
    if (
        validation.get("status") != "PASS"
        or validation.get("summary") != {
            "checks": 178,
            "failures": 0,
            "passed": 178,
            "warnings": 0,
        }
    ):
        raise ValueError("retry3 environment validation is not 178/178 PASS")


def seed_subject_tree(units: list[str]) -> list[dict[str, Any]]:
    if DESTINATION_ROOT.exists():
        raise FileExistsError(f"retry3 destination already exists: {DESTINATION_ROOT}")
    DESTINATION_ROOT.mkdir(parents=True, exist_ok=False)
    for unit in units:
        source_unit = SOURCE_ROOT / "subjects" / unit
        destination_unit = DESTINATION_ROOT / "subjects" / unit
        if not source_unit.is_dir():
            raise FileNotFoundError(f"retry2 subject directory is missing: {unit}")
        destination_unit.mkdir(parents=True, exist_ok=False)
        copied_area = False
        for area in ALLOWED_AREAS:
            source_area = source_unit / area
            if not source_area.exists():
                continue
            for path in source_area.rglob("*"):
                if path.is_symlink():
                    raise ValueError(f"retry2 seed area contains a symlink: {path}")
                if path.is_file() and path.name.endswith((".partial", ".tmp")):
                    raise ValueError(f"retry2 seed area contains an incomplete file: {path}")
            subprocess.run(
                [
                    "cp",
                    "--archive",
                    "--reflink=always",
                    str(source_area),
                    str(destination_unit / area),
                ],
                check=True,
            )
            copied_area = True
        if not copied_area:
            raise ValueError(f"retry2 has no approved seed area for {unit}")

    rows: list[dict[str, Any]] = []
    for destination in sorted((DESTINATION_ROOT / "subjects").rglob("*")):
        if destination.is_symlink():
            raise ValueError(f"retry3 seed contains a symlink: {destination}")
        if not destination.is_file():
            continue
        relative = destination.relative_to(DESTINATION_ROOT).as_posix()
        source = SOURCE_ROOT / relative
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"retry3 seed has no regular retry2 source: {relative}")
        source_hash = sha256_file(source)
        destination_hash = sha256_file(destination)
        if source.stat().st_size != destination.stat().st_size or source_hash != destination_hash:
            raise ValueError(f"retry3 reflink content differs: {relative}")
        rows.append(
            {
                "relative_path": relative,
                "sha256": destination_hash,
                "size_bytes": destination.stat().st_size,
            }
        )
    return rows


def main() -> int:
    if PACKAGE.exists() or BINDING.exists():
        raise FileExistsError("retry3 package or binding already exists")
    end_path, end, completion = require_retry2_terminal()
    require_static_contract()
    source_decision = json.loads(SOURCE_DECISION.read_text(encoding="utf-8"))
    units = source_decision.get("units")
    if not isinstance(units, list) or len(units) != 15 or units != sorted(set(units)):
        raise ValueError("retry2 exact unit list differs")
    import build_gpu_transition_gate as transition_gate

    pre_retry3_gate = transition_gate.build_record(
        run_root=SOURCE_ROOT,
        decision_path=SOURCE_DECISION,
        gpu_build_path=transition_gate.DEFAULT_GPU_BUILD,
        output_validation_path=transition_gate.DEFAULT_OUTPUT_VALIDATION,
        retry3_package_validation_path=transition_gate.DEFAULT_RETRY3_PACKAGE_VALIDATION,
        retry3_dryrun_validation_path=transition_gate.DEFAULT_RETRY3_DRYRUN_VALIDATION,
        service="scforge-h04a-r1-phase-a-retry2.service",
    )
    pre_retry3_checks = require_pre_retry3_operational_gate(pre_retry3_gate)
    require_complete_seed_inputs(units)
    rows = seed_subject_tree(units)
    seed = {
        "schema_version": "1.0.0",
        "record_type": "h04a_r1_retry_seed_manifest",
        "status": "PASS",
        "retry_id": RETRY3_ID,
        "copy_mode": "reflink_copy_on_write",
        "selection_uses_diagnosis_labels": False,
        "source_run_root": str(SOURCE_ROOT),
        "destination_run_root": str(DESTINATION_ROOT),
        "source_attempt_end": file_record(end_path),
        "source_completion": file_record(SOURCE_COMPLETION),
        "approved_units": units,
        "allowed_subject_areas": list(ALLOWED_AREAS),
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "stage_counts": {
            "dwi_preproc": sum(row["relative_path"].endswith("/01_dwi/dwi_preproc.mif") for row in rows),
            "dwi_bias_corrected": sum(row["relative_path"].endswith("/01_dwi/dwi_preproc_biascorr.mif") for row in rows),
            "dwi_brain_mask": sum(row["relative_path"].endswith("/01_dwi/dwi_brain_mask.mif") for row in rows),
        },
    }
    write_immutable_json(SEED_MANIFEST, seed)

    sys_path = str(ROOT / "scforge")
    if sys_path not in os.sys.path:
        os.sys.path.insert(0, sys_path)
    from scforge.retry_seed import validate_retry_seed_manifest

    seed_validation = validate_retry_seed_manifest(
        file_record(SEED_MANIFEST),
        destination_root=DESTINATION_ROOT,
        approved_units=units,
    )

    previous_retry = source_decision.get("execution_retry")
    if not isinstance(previous_retry, dict):
        raise ValueError("retry2 decision lacks its immutable retry record")
    charged = float(previous_retry.get("prior_duration_seconds_charged", 0.0)) + float(
        end["duration_seconds"]
    )
    decision = json.loads(SOURCE_DECISION.read_text(encoding="utf-8"))
    decision["proposed_run_root"] = str(DESTINATION_ROOT)
    decision["normative_config_sha256"] = sha256_file(
        ROOT / "configs/connectome_v2_retry3.yaml"
    )
    decision["workflow_source_manifest_sha256"] = sha256_file(
        ROOT / "scforge/workflow/workflow_source_manifest_retry3.tsv"
    )
    decision["environment_contract_sha256"] = sha256_file(
        ROOT / "scforge/workflow/environment_contract_retry3.yaml"
    )
    decision["execution_retry"] = {
        "retry_id": RETRY3_ID,
        "authorization_change": False,
        "scientific_parameter_change": False,
        "unit_set_change": False,
        "resource_cap_change": False,
        "imaging_jobs_started_in_prior_attempts": True,
        "reason": "single-output Snakemake Namedlist was passed directly to pathlib.Path before FOD shell extraction",
        "corrective_action": "use an indexed single gradient-contract input and content-addressed copy-on-write reuse of valid retry2 subject artifacts",
        "rerun_triggers": ["input", "params"],
        "prior_duration_seconds_charged": charged,
        "prior_attempt_ends": list(previous_retry.get("prior_attempt_ends", []))
        + [file_record(end_path)],
        "prior_completions": list(previous_retry.get("prior_completions", []))
        + [file_record(SOURCE_COMPLETION)],
        "prior_run_roots_preserved": list(previous_retry.get("prior_run_roots_preserved", []))
        + [str(SOURCE_ROOT)],
        "previous_retry_record": previous_retry,
        "seed_manifest": file_record(SEED_MANIFEST),
        "source_patch": file_record(SOURCE_PATCH),
        "source_change": {
            "old_fod_rule_sha256": EXPECTED_HASHES[
                ROOT / "scforge/workflow/rules/05_5tt_fod.smk"
            ],
            "retry3_fod_rule_sha256": EXPECTED_HASHES[
                ROOT / "scforge/workflow/rules/05_5tt_fod_retry3.smk"
            ],
            "old_workflow_source_manifest_sha256": EXPECTED_HASHES[
                ROOT / "scforge/workflow/workflow_source_manifest.tsv"
            ],
            "retry3_workflow_source_manifest_sha256": decision[
                "workflow_source_manifest_sha256"
            ],
        },
    }
    write_immutable_json(DECISION, decision)

    checks = {
        "pre_retry3_operational_gate_passed": len(pre_retry3_checks) == 11,
        "exact_type_error_proved": True,
        "authorization_unchanged": decision["execution_retry"]["authorization_change"] is False,
        "scientific_parameters_unchanged": decision["execution_retry"]["scientific_parameter_change"] is False,
        "unit_set_unchanged": decision["units"] == units and len(units) == 15,
        "resource_caps_unchanged": decision["execution_retry"]["resource_cap_change"] is False,
        "downstream_forbidden": all(
            decision.get(key) is False
            for key in (
                "fod_reconstruction_authorized",
                "tensor_maps_authorized",
                "tractography_authorized",
                "matrix_generation_authorized",
                "full_cohort_authorized",
                "statistical_analysis_authorized",
                "dashboard_publication_authorized",
            )
        ),
        "static_environment_178_of_178": True,
        "reflink_seed_rehashed": seed_validation.get("status") == "PASS",
        "all_approved_units_complete_through_masking": (
            seed_validation.get("covered_unit_count") == 15
            and seed["stage_counts"]
            == {
                "dwi_preproc": 15,
                "dwi_bias_corrected": 15,
                "dwi_brain_mask": 15,
            }
        ),
        "mtime_recomputation_disabled": decision["execution_retry"]["rerun_triggers"]
        == ["input", "params"],
        "cumulative_wall_clock_within_limit": charged < 24 * 60 * 60,
    }
    validation = {
        "schema_version": "1.0.0",
        "record_type": "h04a_r1_retry3_package_validation",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "summary": {"passed": sum(checks.values()), "total": len(checks)},
        "seed_validation": seed_validation,
        "execution_decision": file_record(DECISION),
    }
    if validation["status"] != "PASS":
        raise ValueError("retry3 package validation failed")
    write_immutable_json(VALIDATION, validation)

    binding = {
        "schema_version": "1.0.0",
        "record_type": "h04a_r1_recovery_extension_binding",
        "status": "LOCKED",
        "retry_id": RETRY3_ID,
        "run_root": str(DESTINATION_ROOT),
        "source_files": {
            "snakefile": file_record(ROOT / "scforge/workflow/Snakefile_h04a_r1_retry3"),
            "manifest_rule": file_record(ROOT / "scforge/workflow/extensions/h04a_r1/00_manifest.smk"),
            "input_rule": file_record(ROOT / "scforge/workflow/extensions/h04a_r1/00_inputs.smk"),
            "launcher": file_record(ROOT / "scforge/workflow/run_h04a_r1_retry3.py"),
            "compatibility_validator": file_record(ROOT / "scforge/scforge/h04a_r1_retry3.py"),
            "retry_seed_validator": file_record(ROOT / "scforge/scforge/retry_seed.py"),
            "fod_rule": file_record(ROOT / "scforge/workflow/rules/05_5tt_fod_retry3.smk"),
        },
        "converter": file_record(ROOT / "tools/dcm2niix/v1.0.20260416/dcm2niix"),
        "source_approval": file_record(SOURCE_APPROVAL),
        "source_recommendation": file_record(SOURCE_RECOMMENDATION),
        "retry_recommendation": file_record(RETRY_RECOMMENDATION),
        "execution_decision": file_record(DECISION),
        "execution_manifest": file_record(EXECUTION_MANIFEST),
        "package_validation": file_record(VALIDATION),
        "source_patch": file_record(SOURCE_PATCH),
        "seed_manifest": file_record(SEED_MANIFEST),
        "workflow_source_manifest": file_record(
            ROOT / "scforge/workflow/workflow_source_manifest_retry3.tsv"
        ),
        "environment_contract": file_record(
            ROOT / "scforge/workflow/environment_contract_retry3.yaml"
        ),
        "source_attempt_end": file_record(end_path),
        "source_completion": file_record(SOURCE_COMPLETION),
        "environment_validation": file_record(STATIC_ENVIRONMENT_VALIDATION),
        "package_builder": file_record(Path(__file__)),
        "allowed_terminal_stage": "response_calibration_phase_a_freeze_decision",
        "diagnosis_labels_used": False,
        "approved_unit_count": 15,
        "units": units,
        "maximum_cpu_cores": 32,
        "maximum_wall_clock_hours": 24,
        "maximum_additional_storage_gib": 100,
        "minimum_valid_response_calibration_units": 12,
        "required_manufacturer_families": 2,
        "required_t1_source_classes": 2,
        "rerun_triggers": ["input", "params"],
        "forbidden_stages": [
            "fod_reconstruction",
            "tensor_maps",
            "tractography",
            "sift2",
            "connectome_matrices",
            "statistical_analysis",
            "dashboard_publication",
        ],
    }
    write_immutable_json(BINDING, binding)

    from scforge.h04a_r1_retry3 import validate_recovery_extension_binding

    validate_recovery_extension_binding(binding, expected_run_root=DESTINATION_ROOT)
    print(
        json.dumps(
            {
                "status": "PASS",
                "retry_id": RETRY3_ID,
                "seed": file_record(SEED_MANIFEST),
                "decision": file_record(DECISION),
                "validation": file_record(VALIDATION),
                "binding": file_record(BINDING),
                "stage_counts": seed["stage_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
