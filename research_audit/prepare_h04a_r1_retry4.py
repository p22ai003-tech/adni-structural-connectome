#!/usr/bin/env python3
"""Prepare an isolated immutable-upstream retry4 after retry3 fails closed."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
AUDIT = ROOT / "research_audit"
SOURCE_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry3"
)
DESTINATION_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
SOURCE_DECISION = (
    AUDIT
    / "outputs/h04a_r1_retry3_package_v1/recovery_execution_decision_retry3.json"
)
SOURCE_BINDING = AUDIT / "outputs/h04a_r1_recovery_extension_binding_v5.json"
SOURCE_COMPLETION = (
    SOURCE_ROOT / "publication/response_calibration_phase_a_completion.json"
)
EXECUTION_MANIFEST = (
    AUDIT / "outputs/h04a_r1_recovery_package_v1/recovery_execution_manifest_v1.csv"
)
PACKAGE = AUDIT / "outputs/h04a_r1_retry4_package_v1"
SEED_MANIFEST = PACKAGE / "retry4_seed_manifest.json"
DECISION = PACKAGE / "recovery_execution_decision_retry4.json"
VALIDATION = PACKAGE / "retry4_package_validation.json"
BINDING = AUDIT / "outputs/h04a_r1_recovery_extension_binding_v6.json"
SOURCE_PATCH = (
    AUDIT / "decisions/sl_h04a_r1_retry4_seed_boundary_fix_20260719.patch"
)
RETRY_RECOMMENDATION = (
    AUDIT / "decisions/sl_h04a_r1_retry4_recommendation_20260719.md"
)
LAUNCHER_SAFETY_DECISION = (
    AUDIT / "decisions/sl_h04a_r1_retry4_rule_allowlist_20260719.md"
)
DIAGNOSTIC_DRYRUN = (
    AUDIT / "outputs/h04a_r1_retry3_dryrun_v5/validation.json"
)
STATIC_ENVIRONMENT_VALIDATION = (
    AUDIT / "outputs/h04a_r1_retry3_static_v1/environment_validation.json"
)
SOURCE_APPROVAL = AUDIT / "decisions/sl_h04a_r1_approval_20260718.json"
SOURCE_RECOMMENDATION = (
    AUDIT / "decisions/sl_h04a_r1_recovery_recommendation_20260718.json"
)
RETRY4_ID = "SL-H04A-R1-RETRY4"
IMMUTABLE_AREAS = ("00_inputs", "01_dwi")
REQUIRED_SEED_PRODUCTS = (
    "01_dwi/dwi_preproc.mif",
    "01_dwi/dwi_preproc_biascorr.mif",
    "01_dwi/dwi_brain_mask.mif",
)
KNOWN_TERMINAL_SERVICE_STATES = frozenset({"inactive", "failed"})


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


def require_retry3_terminal() -> tuple[Path, dict[str, Any]]:
    state = subprocess.run(
        ["systemctl", "--user", "is-active", "scforge-h04a-r1-phase-a-retry3.service"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip() or "unknown"
    if state not in KNOWN_TERMINAL_SERVICE_STATES:
        raise RuntimeError(f"retry3 service is not known terminal: {state}")

    process_text = subprocess.run(
        [
            "pgrep",
            "-af",
            "snakemake|dwi2response|dwiextract|eddy_cpu|dwifslpreproc",
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    active = [line for line in process_text.splitlines() if "pgrep -af" not in line]
    if active:
        raise RuntimeError("imaging processes remain active: " + " | ".join(active[:5]))

    end_paths = sorted((SOURCE_ROOT / "attempts").glob("*.end.json"))
    if len(end_paths) != 1:
        raise ValueError(f"retry3 must have one immutable attempt end, found {len(end_paths)}")
    end_path = end_paths[0]
    end = json.loads(end_path.read_text(encoding="utf-8"))
    if (
        end.get("status") != "FAIL"
        or end.get("returncode") != 1
        or end.get("launcher_mode") != "response-calibration-phase-a"
    ):
        raise ValueError("retry3 terminal outcome is not the expected Phase-A failure")
    duration = end.get("duration_seconds")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        raise ValueError("retry3 duration is invalid")

    completion = json.loads(SOURCE_COMPLETION.read_text(encoding="utf-8"))
    if (
        completion.get("record_type") != "response_calibration_phase_a_completion"
        or completion.get("status") != "FAIL"
        or completion.get("terminal_attempt_end") != file_record(end_path)
    ):
        raise ValueError("retry3 completion does not bind its failed attempt")

    log_record = end.get("combined_execution_log")
    if not isinstance(log_record, dict):
        raise ValueError("retry3 attempt end lacks a combined log record")
    log_path = Path(str(log_record.get("path", ""))).resolve()
    if file_record(log_path) != log_record:
        raise ValueError("retry3 combined log identity differs")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    expected_error = (
        "retry seed file size differs: "
        "subjects/009_S_4324_I1186579/logs/05_select_fod_shells.log"
    )
    if expected_error not in log_text:
        raise ValueError("retry3 log does not prove the mutable-log seed failure")

    shell_outputs = list(SOURCE_ROOT.glob("subjects/*/05_model/dwi_fod_shells.mif"))
    response_outputs = list(SOURCE_ROOT.glob("subjects/*/05_model/response_wm.txt"))
    forbidden = (
        list(SOURCE_ROOT.glob("subjects/*/05_model/wm_fod.mif"))
        + list(SOURCE_ROOT.rglob("*.tck"))
        + list(SOURCE_ROOT.rglob("*matrix*.csv"))
    )
    if len(shell_outputs) != 8 or response_outputs or forbidden:
        raise ValueError(
            "retry3 attrition differs from 8 shell outputs, 0 responses, 0 downstream"
        )
    return end_path, end


def require_complete_seed_inputs(units: list[str]) -> dict[str, int]:
    counts = {relative: 0 for relative in REQUIRED_SEED_PRODUCTS}
    for unit in units:
        for relative in REQUIRED_SEED_PRODUCTS:
            product = SOURCE_ROOT / "subjects" / unit / relative
            if not product.is_file() or product.is_symlink() or product.stat().st_size < 1:
                raise ValueError(f"retry4 reusable input is missing: {unit}/{relative}")
            counts[relative] += 1
    if set(counts.values()) != {len(units)}:
        raise ValueError(f"retry4 reusable seed coverage differs: {counts}")
    return counts


def seed_immutable_tree(units: list[str]) -> list[dict[str, Any]]:
    if DESTINATION_ROOT.exists():
        raise FileExistsError(f"retry4 destination already exists: {DESTINATION_ROOT}")
    DESTINATION_ROOT.mkdir(parents=True, exist_ok=False)
    for unit in units:
        destination_unit = DESTINATION_ROOT / "subjects" / unit
        destination_unit.mkdir(parents=True, exist_ok=False)
        for area in IMMUTABLE_AREAS:
            source_area = SOURCE_ROOT / "subjects" / unit / area
            if not source_area.is_dir():
                raise FileNotFoundError(f"retry3 immutable seed area is missing: {unit}/{area}")
            for path in source_area.rglob("*"):
                if path.is_symlink():
                    raise ValueError(f"retry3 immutable area contains a symlink: {path}")
                if path.is_file() and path.name.endswith((".partial", ".tmp")):
                    raise ValueError(f"retry3 immutable area contains incomplete file: {path}")
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

    rows: list[dict[str, Any]] = []
    for destination in sorted((DESTINATION_ROOT / "subjects").rglob("*")):
        if destination.is_symlink():
            raise ValueError(f"retry4 seed contains a symlink: {destination}")
        if not destination.is_file():
            continue
        relative = destination.relative_to(DESTINATION_ROOT).as_posix()
        source = SOURCE_ROOT / relative
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"retry4 seed has no regular retry3 source: {relative}")
        source_hash = sha256_file(source)
        destination_hash = sha256_file(destination)
        if source.stat().st_size != destination.stat().st_size or source_hash != destination_hash:
            raise ValueError(f"retry4 reflink content differs: {relative}")
        rows.append(
            {
                "relative_path": relative,
                "sha256": destination_hash,
                "size_bytes": destination.stat().st_size,
            }
        )
    return rows


def main() -> int:
    if PACKAGE.exists() or BINDING.exists() or DESTINATION_ROOT.exists():
        raise FileExistsError("retry4 package, binding, or destination already exists")
    for required in (
        SOURCE_DECISION,
        SOURCE_BINDING,
        EXECUTION_MANIFEST,
        SOURCE_PATCH,
        RETRY_RECOMMENDATION,
        LAUNCHER_SAFETY_DECISION,
        DIAGNOSTIC_DRYRUN,
        STATIC_ENVIRONMENT_VALIDATION,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    end_path, end = require_retry3_terminal()
    source_decision = json.loads(SOURCE_DECISION.read_text(encoding="utf-8"))
    units = source_decision.get("units")
    if not isinstance(units, list) or units != sorted(set(units)) or len(units) != 15:
        raise ValueError("retry3 exact unit list differs")
    require_complete_seed_inputs(units)
    rows = seed_immutable_tree(units)

    seed = {
        "schema_version": "1.0.0",
        "record_type": "h04a_r1_retry_seed_manifest",
        "status": "PASS",
        "retry_id": RETRY4_ID,
        "copy_mode": "reflink_copy_on_write",
        "selection_uses_diagnosis_labels": False,
        "source_run_root": str(SOURCE_ROOT),
        "destination_run_root": str(DESTINATION_ROOT),
        "source_attempt_end": file_record(end_path),
        "source_completion": file_record(SOURCE_COMPLETION),
        "approved_units": units,
        "allowed_subject_areas": list(IMMUTABLE_AREAS),
        "continuation_paths_excluded": True,
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "stage_counts": {
            "dwi_preproc": sum(
                row["relative_path"].endswith("/01_dwi/dwi_preproc.mif") for row in rows
            ),
            "dwi_bias_corrected": sum(
                row["relative_path"].endswith("/01_dwi/dwi_preproc_biascorr.mif")
                for row in rows
            ),
            "dwi_brain_mask": sum(
                row["relative_path"].endswith("/01_dwi/dwi_brain_mask.mif") for row in rows
            ),
        },
    }
    write_immutable_json(SEED_MANIFEST, seed)

    sys.path.insert(0, str(ROOT / "scforge"))
    from scforge.retry_seed_retry4 import validate_retry4_seed_manifest

    seed_validation = validate_retry4_seed_manifest(
        file_record(SEED_MANIFEST),
        destination_root=DESTINATION_ROOT,
        approved_units=units,
    )

    previous_retry = source_decision.get("execution_retry")
    if not isinstance(previous_retry, dict):
        raise ValueError("retry3 decision lacks an execution_retry record")
    charged = float(previous_retry.get("prior_duration_seconds_charged", 0.0)) + float(
        end["duration_seconds"]
    )
    decision = json.loads(SOURCE_DECISION.read_text(encoding="utf-8"))
    decision["proposed_run_root"] = str(DESTINATION_ROOT)
    decision["execution_retry"] = {
        "retry_id": RETRY4_ID,
        "authorization_change": False,
        "scientific_parameter_change": False,
        "unit_set_change": False,
        "resource_cap_change": False,
        "imaging_jobs_started_in_prior_attempts": True,
        "reason": "retry3 seed included continuation-owned shell-selection logs and self-invalidated after an authorized write",
        "corrective_action": "seed only immutable 00_inputs and 01_dwi areas; exclude every continuation-owned path",
        "continuation_paths_excluded_from_seed": True,
        "rerun_triggers": ["input", "params"],
        "prior_duration_seconds_charged": charged,
        "prior_attempt_ends": list(previous_retry.get("prior_attempt_ends", []))
        + [file_record(end_path)],
        "prior_completions": list(previous_retry.get("prior_completions", []))
        + [file_record(SOURCE_COMPLETION)],
        "prior_run_roots_preserved": list(
            previous_retry.get("prior_run_roots_preserved", [])
        )
        + [str(SOURCE_ROOT)],
        "previous_retry_record": previous_retry,
        "seed_manifest": file_record(SEED_MANIFEST),
        "source_patch": file_record(SOURCE_PATCH),
        "source_change": {
            "scientific_workflow_changed": False,
            "seed_scope_before": ["00_inputs", "01_dwi", "logs"],
            "seed_scope_after": list(IMMUTABLE_AREAS),
        },
    }
    write_immutable_json(DECISION, decision)

    checks = {
        "retry3_terminal_failure_proved": True,
        "authorization_unchanged": False is decision["execution_retry"]["authorization_change"],
        "scientific_parameters_unchanged": False
        is decision["execution_retry"]["scientific_parameter_change"],
        "unit_set_unchanged": decision["units"] == units and len(units) == 15,
        "resource_caps_unchanged": False
        is decision["execution_retry"]["resource_cap_change"],
        "only_immutable_subject_areas_seeded": seed["allowed_subject_areas"]
        == list(IMMUTABLE_AREAS),
        "continuation_paths_excluded": seed["continuation_paths_excluded"] is True,
        "seed_rehashed": seed_validation.get("status") == "PASS",
        "all_15_complete_through_masking": seed["stage_counts"]
        == {
            "dwi_preproc": 15,
            "dwi_bias_corrected": 15,
            "dwi_brain_mask": 15,
        },
        "cumulative_wall_clock_within_limit": charged < 24 * 60 * 60,
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
    }
    validation = {
        "schema_version": "1.0.0",
        "record_type": "h04a_r1_retry4_package_validation",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "summary": {"passed": sum(checks.values()), "total": len(checks)},
        "seed_validation": seed_validation,
        "execution_decision": file_record(DECISION),
    }
    if validation["status"] != "PASS":
        raise ValueError("retry4 package validation failed")
    write_immutable_json(VALIDATION, validation)

    source_binding = json.loads(SOURCE_BINDING.read_text(encoding="utf-8"))
    binding = dict(source_binding)
    binding.update(
        {
            "retry_id": RETRY4_ID,
            "run_root": str(DESTINATION_ROOT),
            "source_files": {
                "snakefile": file_record(
                    ROOT / "scforge/workflow/Snakefile_h04a_r1_retry4"
                ),
                "base_snakefile": file_record(
                    ROOT / "scforge/workflow/Snakefile_h04a_r1_retry3"
                ),
                "manifest_rule": file_record(
                    ROOT / "scforge/workflow/extensions/h04a_r1/00_manifest.smk"
                ),
                "input_rule": file_record(
                    ROOT / "scforge/workflow/extensions/h04a_r1/00_inputs.smk"
                ),
                "launcher": file_record(
                    ROOT / "scforge/workflow/run_h04a_r1_retry4.py"
                ),
                "base_launcher": file_record(
                    ROOT / "scforge/workflow/run_h04a_r1_recovery.py"
                ),
                "compatibility_validator": file_record(
                    ROOT / "scforge/scforge/h04a_r1_retry4.py"
                ),
                "base_compatibility_validator": file_record(
                    ROOT / "scforge/scforge/h04a_r1.py"
                ),
                "retry_seed_validator": file_record(
                    ROOT / "scforge/scforge/retry_seed_retry4.py"
                ),
                "base_retry_seed_validator": file_record(
                    ROOT / "scforge/scforge/retry_seed.py"
                ),
                "fod_rule": file_record(
                    ROOT / "scforge/workflow/rules/05_5tt_fod_retry3.smk"
                ),
            },
            "retry_recommendation": file_record(RETRY_RECOMMENDATION),
            "execution_decision": file_record(DECISION),
            "package_validation": file_record(VALIDATION),
            "source_patch": file_record(SOURCE_PATCH),
            "seed_manifest": file_record(SEED_MANIFEST),
            "source_attempt_end": file_record(end_path),
            "source_completion": file_record(SOURCE_COMPLETION),
            "package_builder": file_record(Path(__file__)),
            "launcher_safety_decision": file_record(LAUNCHER_SAFETY_DECISION),
            "diagnostic_dryrun": file_record(DIAGNOSTIC_DRYRUN),
            "supersedes_binding": file_record(SOURCE_BINDING),
            "seed_excludes_mutable_continuation_paths": True,
            "upstream_recomputation_authorized": False,
        }
    )
    write_immutable_json(BINDING, binding)

    from scforge.h04a_r1_retry4 import validate_recovery_extension_binding

    validate_recovery_extension_binding(binding, expected_run_root=DESTINATION_ROOT)
    print(
        json.dumps(
            {
                "status": "PASS",
                "retry_id": RETRY4_ID,
                "seed": file_record(SEED_MANIFEST),
                "decision": file_record(DECISION),
                "validation": file_record(VALIDATION),
                "binding": file_record(BINDING),
                "stage_counts": seed["stage_counts"],
                "seed_areas": seed["allowed_subject_areas"],
                "prior_duration_seconds_charged": charged,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
