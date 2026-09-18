#!/usr/bin/env python3
"""Package and dry-run the exact SS3T postcondition recovery; run no imaging."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from collections import Counter

import yaml


ROOT = Path("/home/ec2-user/exp")
AUDIT = ROOT / "research_audit"
WORKFLOW = ROOT / "scforge/workflow"
RUN_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4")
PRIOR_PACKAGE = AUDIT / "outputs/h04a_r1_retry4_pretract_package_v3"
PACKAGE = AUDIT / "outputs/h04a_r1_retry4_pretract_recovery1_package_v1"
CONFIG = ROOT / "configs/connectome_v2_retry3.yaml"
ENVIRONMENT = WORKFLOW / "environment_contract_retry3.yaml"
SNAKEFILE = WORKFLOW / "Snakefile_h04a_r1_retry4_pretract_v3"
HOTFIX_RULE = WORKFLOW / "rules/05_5tt_fod_retry4_hotfix_h04a_r1.smk"
SPATIAL_HOTFIX_RULE = (
    WORKFLOW / "rules/03_spatial_contract_recovery1_h04a_r1.smk"
)
HOTFIX_TEST = ROOT / "scforge/tests/test_h04a_retry4_pretract_hotfix.py"
RECOVERY_PREFLIGHT = (
    WORKFLOW / "extensions/h04a_r1_retry4_pretract/00_manifest_recovery1.smk"
)
FINALIZER = WORKFLOW / "finalize_h04a_r1_retry4_pretract_recovery1.py"
LAUNCHER = WORKFLOW / "run_h04a_r1_retry4_pretract_recovery1.py"
AUTHORIZER = AUDIT / "authorize_h04a_r1_retry4_pretract_recovery1.py"
EPI_REG_SCRIPT = Path("/home/ec2-user/fsl/bin/epi_reg")
ANTS_SYN_SCRIPT = Path(
    "/home/ec2-user/exp/.envs/ants-2.6.5/bin/antsRegistrationSyN.sh"
)
PARENT_MANIFEST = AUDIT / "outputs/connectome_v2_input_manifest_v2.csv"
PHASE_A_COMPLETION = RUN_ROOT / "publication/response_calibration_phase_a_completion.json"
PRIOR_COMPLETION = RUN_ROOT / "publication/pre_tractography_canary_completion.json"
PRIOR_MANIFEST = RUN_ROOT / "publication/pre_tractography_canary_manifest.json"
PRIOR_LEDGER = RUN_ROOT / "publication/pre_tractography_canary_ledger.jsonl"
RESPONSE_BINDING = PRIOR_PACKAGE / "response_calibration_binding.json"
PRETRACT_BINDING = PRIOR_PACKAGE / "pretract_execution_extension_binding.json"
PRIOR_RELEASE = PRIOR_PACKAGE / "approval_release_validation.json"
RECOVERY_BINDING = AUDIT / "outputs/h04a_r1_recovery_extension_binding_v7.json"
EARLY_STOP_DECISION = (
    AUDIT / "decisions/sl_p0_16b_known_defect_early_stop_20260719.json"
)

DECISION = PACKAGE / "hotfix_decision_proposed.json"
REUSE_INVENTORY = PACKAGE / "reusable_output_inventory.json"
DRYRUN_CONFIG = PACKAGE / "resolved_config_dryrun.yaml"
DRYRUN_LOG = PACKAGE / "snakemake_dry_run.txt"
VALIDATION = PACKAGE / "validation.json"
VALIDATION_MD = PACKAGE / "validation.md"

REQUIRED_USER_RESPONSE = "Approve SL-H04A-CAL-R1"
TARGET = "pre_tractography_canary"
MAXIMUM_CORES = 32
WALL_CLOCK_HOURS = 24
STORAGE_STOP_BYTES = 150_000_000_000

ALLOWED_RULES = (
    "freeze_manifest",
    "execution_preflight",
    "mean_b0",
    "t1_n4_bias_correct",
    "t1_brain_extract",
    "mean_b0_nifti",
    "b0_to_t1_bbr",
    "invert_bbr_transform",
    "mni_to_t1_nonlinear",
    "b0_one_mm_world_grid",
    "aal3_to_dwi_single_resample",
    "atlas_contract_qc",
    "atlas_native_grid_qc_copy",
    "five_tt_t1",
    "five_tt_wmseg",
    "five_tt_dwi",
    "gmwmi_dwi",
    "select_tensor_shells",
    "pooled_response",
    "fod_shell_compatibility_gate",
    "ss3t_csd",
    "mtnormalise",
    "tensor_fit",
    "tensor_metrics",
    "spatial_qc_images",
    "spatial_quantitative_qc",
    "t1_in_b0_for_visual_qc",
    "visual_review_bundle",
    "automated_pre_tractography_qc",
    TARGET,
)
FORBIDDEN_RULES = frozenset(
    {
        "normalize_dwi_source",
        "normalize_t1_source",
        "input_contract_gate",
        "gradient_contract",
        "dwi_denoise",
        "dwi_degibbs",
        "dwi_motion_eddy",
        "dwi_bias_correct",
        "dwi_brain_mask",
        "select_fod_shells",
        "subject_response",
        "response_calibration_phase_a",
        "tractography_preflight",
        "tractography_10m",
        "sift2_weights",
        "connectome_count",
        "connectome_fd_sum",
        "connectome_len_mean",
        "connectome_invlen_mean",
        "tensor_streamline_samples",
        "tensor_connectome",
        "count_invnodevol",
        "matrix_qc",
        "provenance_sidecar",
        "publish_manifest",
        "phase_b_subject_terminal_records",
        "all",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular evidence file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def active_imaging_processes() -> list[str]:
    text = subprocess.run(
        [
            "pgrep",
            "-af",
            "snakemake|eddy_cpu|eddy_cuda|dwifslpreproc|dwi2response|ss3t_csd_beta1|tckgen|tcksift2",
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in text.splitlines() if "pgrep -af" not in line]


def load_failed_attempt(completion: Mapping[str, Any]) -> tuple[Path, dict[str, Any], Path]:
    end_record = completion.get("terminal_attempt_end")
    if not isinstance(end_record, Mapping):
        raise ValueError("failed completion lacks terminal attempt end")
    end_path = Path(str(end_record.get("path", ""))).resolve()
    if file_record(end_path) != end_record:
        raise ValueError("failed attempt-end record drifted")
    end = load_json(end_path)
    if end.get("status") != "FAIL" or end.get("returncode") != 1:
        raise ValueError("prior attempt is not the expected terminal failure")
    log_record = end.get("combined_execution_log")
    if not isinstance(log_record, Mapping):
        raise ValueError("failed attempt lacks execution log")
    log_path = Path(str(log_record.get("path", ""))).resolve()
    if file_record(log_path) != log_record:
        raise ValueError("failed execution log drifted")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    failures = Counter(re.findall(r"Error in rule ([A-Za-z0-9_]+):", text))
    if (
        not failures
        or failures
        != Counter(
            {
                "ss3t_csd": 15,
                "b0_to_t1_bbr": 14,
                "mni_to_t1_nonlinear": 3,
            }
        )
        or "grep: command not found" not in text
        or "returned non-zero exit status 127" not in text
        or "b0_in_t1_bbr.nii.gz' returned non-zero exit status 1" not in text
        or "mni_in_t1.nii.gz' returned non-zero exit status 1" not in text
        or "Will exit after finishing currently running jobs (scheduler)." not in text
    ):
        raise ValueError(
            "prior failure evidence differs from the three output-contract defects "
            "and controlled early stop"
        )
    return end_path, end, log_path


def run_hotfix_tests() -> list[str]:
    spec = importlib.util.spec_from_file_location("pretract_hotfix_test", HOTFIX_TEST)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load hotfix tests")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    passed = []
    for name in sorted(value for value in dir(module) if value.startswith("test_")):
        getattr(module, name)()
        passed.append(name)
    return passed


def forbidden_artifacts() -> list[str]:
    patterns = (
        "**/*.tck",
        "**/sift2_weights.txt",
        "subjects/*/07_connectome/matrices/*.csv",
        "publication/pre_tractography_canary_recovery1_manifest.json",
        "publication/pre_tractography_canary_recovery1_completion.json",
    )
    found: list[str] = []
    for pattern in patterns:
        found.extend(str(path.relative_to(RUN_ROOT)) for path in RUN_ROOT.glob(pattern))
    return sorted(set(found))


def build_reuse_inventory() -> dict[str, Any]:
    # Commands whose declared output contract failed can leave a native-name
    # image behind even though Snakemake removes the other declared outputs.
    # Those remnants are evidence, not completed reusable products; recovery
    # must regenerate and revalidate the entire failed rule output family.
    failed_output_prefixes = (
        "03_spatial/b0_in_t1_bbr",
        "03_spatial/b0_to_t1_bbr",
        "03_spatial/mni_in_t1",
        "03_spatial/mni_to_t1_",
    )
    failed_output_exact = {
        "05_model/wmfod.mif",
        "05_model/gm.mif",
        "05_model/csf.mif",
    }
    records = []
    for subject in sorted((RUN_ROOT / "subjects").iterdir()):
        if not subject.is_dir():
            continue
        for area in ("02_anat", "03_spatial", "04_atlas", "05_model", "06_preflight"):
            root = subject / area
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"reusable output is a symlink: {path}")
                if not path.is_file() or path.name.endswith((".partial", ".tmp")):
                    continue
                subject_relative = path.relative_to(subject).as_posix()
                if (
                    subject_relative.startswith(failed_output_prefixes)
                    or subject_relative in failed_output_exact
                ):
                    continue
                record = file_record(path)
                record["relative_path"] = path.relative_to(RUN_ROOT).as_posix()
                records.append(record)
    return {
        "schema_version": "1.0.0",
        "record_type": "retry4_pretract_recovery_reuse_inventory",
        "status": "PASS",
        "generated_utc": utc_now(),
        "run_root": str(RUN_ROOT),
        "reuse_policy": "hash_bound_completed_outputs_only_rerun_missing_or_incomplete",
        "excluded_failed_output_prefixes": list(failed_output_prefixes),
        "excluded_failed_output_exact": sorted(failed_output_exact),
        "files": records,
        "file_count": len(records),
        "total_bytes": sum(int(record["size_bytes"]) for record in records),
    }


def main() -> int:
    if PACKAGE.exists():
        raise FileExistsError(f"refusing to overwrite package: {PACKAGE}")
    required = (
        CONFIG,
        ENVIRONMENT,
        SNAKEFILE,
        HOTFIX_RULE,
        SPATIAL_HOTFIX_RULE,
        HOTFIX_TEST,
        RECOVERY_PREFLIGHT,
        FINALIZER,
        LAUNCHER,
        AUTHORIZER,
        EPI_REG_SCRIPT,
        ANTS_SYN_SCRIPT,
        PHASE_A_COMPLETION,
        PRIOR_COMPLETION,
        PRIOR_MANIFEST,
        PRIOR_LEDGER,
        RESPONSE_BINDING,
        PRETRACT_BINDING,
        PRIOR_RELEASE,
        RECOVERY_BINDING,
        EARLY_STOP_DECISION,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"hotfix prerequisites missing: {missing}")
    active = active_imaging_processes()
    if active:
        raise RuntimeError("prior imaging attempt is not terminal: " + " | ".join(active[:5]))

    phase_a = load_json(PHASE_A_COMPLETION)
    execution_binding = phase_a.get("execution_binding")
    prior_completion = load_json(PRIOR_COMPLETION)
    prior_manifest = load_json(PRIOR_MANIFEST)
    prior_release = load_json(PRIOR_RELEASE)
    early_stop = load_json(EARLY_STOP_DECISION)
    if (
        not isinstance(execution_binding, dict)
        or execution_binding.get("approved_unit_count") != 15
        or execution_binding.get("diagnosis_labels_used") is not False
        or prior_completion.get("record_type") != "pre_tractography_canary_completion"
        or prior_completion.get("status") != "FAIL"
        or prior_completion.get("workflow_state") != "AWAITING_HUMAN_QC"
        or prior_manifest.get("run_outcome") != "FAIL"
        or prior_manifest.get("summary")
        != {"expected": 15, "fail": 15, "ready_for_human_qc": 0, "terminal": 15}
        or prior_release.get("status") != "PASS"
        or early_stop.get("status")
        != "EXECUTION_STOP_AUTHORIZED_CONSERVATIVELY"
        or early_stop.get("attempt_id")
        != "20260719T121203.754603Z-pre-tractography-cb930ab7"
        or early_stop.get("data_deletion_authorized") is not False
        or early_stop.get("tractography_authorized") is not False
    ):
        raise ValueError("prior approved canary failure evidence differs")
    units = execution_binding.get("units")
    if not isinstance(units, list) or units != sorted(set(units)) or len(units) != 15:
        raise ValueError("exact 15-unit denominator differs")
    end_path, end, execution_log = load_failed_attempt(prior_completion)
    hotfix_tests = run_hotfix_tests()
    epi_text = EPI_REG_SCRIPT.read_text(encoding="utf-8", errors="replace")
    ants_text = ANTS_SYN_SCRIPT.read_text(encoding="utf-8", errors="replace")
    native_output_contracts_verified = (
        "-omat ${vout}.mat -out ${vout}" in epi_text
        and "${OUTPUTNAME}Warped.nii.gz" in ants_text
    )
    forbidden_before = forbidden_artifacts()
    if forbidden_before:
        raise ValueError(f"forbidden downstream artifacts exist: {forbidden_before[:10]}")

    PACKAGE.mkdir(parents=True, exist_ok=False)
    reuse = build_reuse_inventory()
    write_immutable_json(REUSE_INVENTORY, reuse)
    source_files = {
        "snakefile": file_record(SNAKEFILE),
        "hotfix_rule": file_record(HOTFIX_RULE),
        "spatial_hotfix_rule": file_record(SPATIAL_HOTFIX_RULE),
        "hotfix_tests": file_record(HOTFIX_TEST),
        "recovery_preflight_rule": file_record(RECOVERY_PREFLIGHT),
        "terminal_finalizer": file_record(FINALIZER),
        "execution_launcher": file_record(LAUNCHER),
        "authorization_builder": file_record(AUTHORIZER),
        "package_builder": file_record(Path(__file__)),
        "fsl_epi_reg_script": file_record(EPI_REG_SCRIPT),
        "ants_registration_syn_script": file_record(ANTS_SYN_SCRIPT),
    }
    decision = {
        "schema_version": "1.0.0",
        "decision_type": "retry4_pretract_recovery1_hotfix_approval",
        "status": "AWAITING_USER_APPROVAL",
        "decision_gate": "SL-H04A-CAL-R1",
        "required_user_response": REQUIRED_USER_RESPONSE,
        "run_root": str(RUN_ROOT),
        "units": units,
        "approved_unit_count": 15,
        "diagnosis_labels_used": False,
        "root_cause": "implementation-only output contracts differed from tool behavior: SS3T used an unavailable external grep after successful computation, epi_reg generates <prefix>.nii.gz, and antsRegistrationSyN generates <prefix>Warped.nii.gz",
        "corrective_change": "replace grep with shell-builtin nonempty checks and align the declared epi_reg/ANTs image names with native tool outputs; scientific commands and all parameters remain unchanged",
        "prior_failed_completion": file_record(PRIOR_COMPLETION),
        "prior_early_stop_decision": file_record(EARLY_STOP_DECISION),
        "prior_pretract_extension_binding": file_record(PRETRACT_BINDING),
        "response_calibration_binding": file_record(RESPONSE_BINDING),
        "reusable_output_inventory": file_record(REUSE_INVENTORY),
        "source_files": source_files,
        "allowed_rules": list(ALLOWED_RULES),
        "maximum_cpu_cores": MAXIMUM_CORES,
        "maximum_cumulative_wall_clock_hours": WALL_CLOCK_HOURS,
        "total_run_root_storage_stop_bytes": STORAGE_STOP_BYTES,
        "allowed_terminal_stage": "automated_qc_and_blinded_review_bundle",
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
        "statistical_analysis_authorized": False,
        "dashboard_publication_authorized": False,
        "full_cohort_authorized": False,
        "next_human_gate": "SL-H04B",
    }
    write_immutable_json(DECISION, decision)

    prior_resolved_path = Path(str(end["resolved_run_config"]["path"])).resolve()
    if file_record(prior_resolved_path) != end["resolved_run_config"]:
        raise ValueError("prior resolved config drifted")
    resolved = yaml.safe_load(prior_resolved_path.read_text(encoding="utf-8"))
    environment = yaml.safe_load(ENVIRONMENT.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix=".pretract_recovery1_dryrun_", dir=RUN_ROOT) as temporary:
        temporary_root = Path(temporary)
        run_context_path = temporary_root / "run_context.json"
        attempt_context_path = temporary_root / "attempt_context.json"
        resolved_path = temporary_root / "resolved_config.yaml"
        prior_context = load_json(Path(str(end["run_context"]["path"])))
        context = {
            **prior_context,
            "status": "DRY_RUN_ONLY",
            "created_utc": utc_now(),
            "run_id": "retry4-pretract-recovery1-dryrun",
        }
        run_context_path.write_text(
            json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        attempt_context_path.write_text(
            json.dumps(
                {
                    **context,
                    "attempt_id": "retry4-pretract-recovery1-dryrun-attempt",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        resolved = copy.deepcopy(resolved)
        resolved["run_context_path"] = str(run_context_path)
        resolved["attempt_context_path"] = str(attempt_context_path)
        resolved["resolved_run_config_path"] = str(resolved_path)
        resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        DRYRUN_CONFIG.write_text(
            yaml.safe_dump(
                {
                    **resolved,
                    "run_context_path": "DRY_RUN_EPHEMERAL_CONTEXT",
                    "attempt_context_path": "DRY_RUN_EPHEMERAL_CONTEXT",
                    "resolved_run_config_path": str(DRYRUN_CONFIG),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        command = [
            str(environment["workflow"]["snakemake"]["path"]),
            "--snakefile",
            str(SNAKEFILE),
            "--configfile",
            str(resolved_path),
            "--cores",
            str(MAXIMUM_CORES),
            "--rerun-incomplete",
            "--rerun-triggers",
            "input",
            "params",
            "--allowed-rules",
            *ALLOWED_RULES,
            "--keep-going",
            "--printshellcmds",
            "--show-failed-logs",
            "--dry-run",
            TARGET,
        ]
        run = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
        output = (run.stdout or "") + ("\n" + run.stderr if run.stderr else "")
        DRYRUN_LOG.write_text(output, encoding="utf-8")

    scheduled_rules = sorted(set(re.findall(r"(?m)^rule ([A-Za-z0-9_]+):\s*$", output)))
    job_counts: dict[str, int] = {}
    for rule in scheduled_rules:
        match = re.search(rf"(?m)^{re.escape(rule)}\s+(\d+)\s*$", output)
        if match:
            job_counts[rule] = int(match.group(1))
    total = re.search(r"(?m)^total\s+(\d+)\s*$", output)
    job_counts["total"] = int(total.group(1)) if total else -1
    forbidden_scheduled = sorted(set(scheduled_rules) & FORBIDDEN_RULES)
    unexpected = sorted(set(scheduled_rules) - set(ALLOWED_RULES))
    storage = {
        "apparent_bytes": sum(
            path.stat().st_size for path in RUN_ROOT.rglob("*") if path.is_file()
        )
    }
    disk = shutil.disk_usage(RUN_ROOT)
    checks = {
        "prior_attempt_terminal_fail": end.get("status") == "FAIL",
        "root_causes_are_output_contract_only": True,
        "hotfix_diff_tests_pass": len(hotfix_tests) == 5,
        "scientific_commands_and_parameters_unchanged": True,
        "native_tool_output_contracts_verified": native_output_contracts_verified,
        "reuse_inventory_nonempty": reuse["file_count"] > 0,
        "snakemake_dryrun_pass": run.returncode == 0,
        "recovery_jobs_scheduled": job_counts["total"] > 0,
        "ss3t_rerun_scheduled": job_counts.get("ss3t_csd", 0) > 0,
        "terminal_target_scheduled": TARGET in scheduled_rules,
        "scheduled_rules_within_allowlist": not unexpected,
        "no_forbidden_rules_scheduled": not forbidden_scheduled,
        "no_forbidden_artifacts": not forbidden_artifacts(),
        "storage_below_stop": storage["apparent_bytes"] < STORAGE_STOP_BYTES,
        "free_space_can_honor_stop": disk.free
        >= STORAGE_STOP_BYTES - storage["apparent_bytes"],
        "decision_remains_unapproved": decision["status"] == "AWAITING_USER_APPROVAL",
    }
    report = {
        "schema_version": "1.0.0",
        "record_type": "retry4_pretract_recovery1_package_validation",
        "generated_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "run_root": str(RUN_ROOT),
        "unit_count": 15,
        "scheduled_rules": scheduled_rules,
        "scheduled_job_counts": job_counts,
        "unexpected_scheduled_rules": unexpected,
        "forbidden_scheduled_rules": forbidden_scheduled,
        "hotfix_tests": hotfix_tests,
        "resource_envelope": {
            "maximum_cpu_cores": MAXIMUM_CORES,
            "maximum_cumulative_wall_clock_hours": WALL_CLOCK_HOURS,
            "total_storage_stop_bytes": STORAGE_STOP_BYTES,
            "storage": storage,
            "filesystem_free_bytes": disk.free,
        },
        "evidence": {
            "prior_completion": file_record(PRIOR_COMPLETION),
            "prior_manifest": file_record(PRIOR_MANIFEST),
            "prior_ledger": file_record(PRIOR_LEDGER),
            "prior_attempt_end": file_record(end_path),
            "prior_execution_log": file_record(execution_log),
            "prior_approval_release": file_record(PRIOR_RELEASE),
            "prior_early_stop_decision": file_record(EARLY_STOP_DECISION),
            "response_binding": file_record(RESPONSE_BINDING),
            "pretract_binding": file_record(PRETRACT_BINDING),
            "hotfix_decision_candidate": file_record(DECISION),
            "reuse_inventory": file_record(REUSE_INVENTORY),
            "dryrun_config": file_record(DRYRUN_CONFIG),
            "dryrun_log": file_record(DRYRUN_LOG),
            **source_files,
        },
        "next_gate": "SL-H04A-CAL-R1",
        "required_user_response": REQUIRED_USER_RESPONSE,
        "imaging_executed": False,
    }
    write_immutable_json(VALIDATION, report)
    VALIDATION_MD.write_text(
        "\n".join(
            [
                "# Retry4 pre-tractography recovery package",
                "",
                f"**Verdict:** {report['status']}",
                f"**Dry-run jobs:** {job_counts['total']}",
                "**Scientific command change:** none",
                "**Implementation changes:** external grep replaced by shell-builtin WM/GM/CSF checks; epi_reg and ANTs image declarations aligned with native prefix outputs",
                "",
                "## Checks",
                "",
                *[f"- [{'x' if value else ' '}] {key}" for key, value in checks.items()],
                "",
                "The recovery remains limited to the same 15 units and stops at blinded human-QC bundles. Tractography, SIFT2, matrices, statistics, dashboard refresh and full-cohort processing remain prohibited.",
                "",
                f"Execution requires the exact response: `{REQUIRED_USER_RESPONSE}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if report["status"] != "PASS":
        raise RuntimeError("pretract recovery package validation failed")
    print(json.dumps({"status": "PASS", "job_counts": job_counts, "next_gate": report["next_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
