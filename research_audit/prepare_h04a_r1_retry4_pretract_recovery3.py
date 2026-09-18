#!/usr/bin/env python3
"""Build and dry-run Recovery3; execute no production imaging."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path("/home/ec2-user/exp")
AUDIT = ROOT / "research_audit"
WORKFLOW = ROOT / "scforge/workflow"
RUN_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4")
PACKAGE = AUDIT / "outputs/h04a_r1_retry4_pretract_recovery3_package_v1"
CONFIG = ROOT / "configs/connectome_v2_retry3.yaml"
ENVIRONMENT = WORKFLOW / "environment_contract_retry3.yaml"
SNAKEFILE = WORKFLOW / "Snakefile_h04a_r1_retry4_pretract_v5"
PRIOR_SNAKEFILE = WORKFLOW / "Snakefile_h04a_r1_retry4_pretract_v4"
SPATIAL_RULE = WORKFLOW / "rules/03_spatial_contract_recovery3_h04a_r1.smk"
ATLAS_RULE = WORKFLOW / "rules/04_atlas_recovery3_h04a_r1.smk"
MODEL_RULE = WORKFLOW / "rules/05_5tt_fod_retry4_hotfix_recovery3_h04a_r1.smk"
PREFLIGHT_RULE = WORKFLOW / "rules/06_preflight_recovery3_h04a_r1.smk"
RECOVERY3_TEST = ROOT / "scforge/tests/test_h04a_retry4_pretract_recovery3.py"
DIAGNOSTICS = (
    AUDIT
    / "outputs/h04a_r1_retry4_pretract_recovery3_diagnostics_v1/recovery3_diagnostics.json"
)
CONVERTER_CONTRACT = WORKFLOW / "environment_contract_recovery2_convert3d.yaml"
CONVERTER_LOCK = WORKFLOW / "locks/convert3d-1.3.0-recovery2-linux-64.explicit.txt"
CONVERTER_TEST = ROOT / "scforge/tests/test_h04a_retry4_pretract_recovery2_converter.py"
FINALIZER = WORKFLOW / "finalize_h04a_r1_retry4_pretract_recovery3.py"
LAUNCHER = WORKFLOW / "run_h04a_r1_retry4_pretract_recovery3.py"
AUTHORIZER = AUDIT / "authorize_h04a_r1_retry4_pretract_recovery3.py"
PHASE_A_COMPLETION = RUN_ROOT / "publication/response_calibration_phase_a_completion.json"
PRIOR_COMPLETION = RUN_ROOT / "publication/pre_tractography_canary_recovery2_completion.json"
PRIOR_PACKAGE = AUDIT / "outputs/h04a_r1_retry4_pretract_package_v3"
RESPONSE_BINDING = PRIOR_PACKAGE / "response_calibration_binding.json"
PRETRACT_BINDING = PRIOR_PACKAGE / "pretract_execution_extension_binding.json"

sys.path.insert(0, str(WORKFLOW))
sys.path.insert(0, str(ROOT / "scforge"))

import run_h04a_r1_retry4 as retry4  # noqa: E402


base = retry4.base

DECISION = PACKAGE / "recovery3_decision_proposed.json"
REUSE_INVENTORY = PACKAGE / "reusable_output_inventory.json"
CONVERTER_TEST_REPORT = PACKAGE / "converter_validation.json"
DRYRUN_LOG = PACKAGE / "snakemake_dry_run.txt"
VALIDATION = PACKAGE / "validation.json"
VALIDATION_MD = PACKAGE / "validation.md"

REQUIRED_USER_RESPONSE = "Approve SL-H04A-CAL-R3"
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
    "mni_registration_fixed_image",
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
EXPECTED_COUNTS = {
    "aal3_to_dwi_single_resample": 15,
    "atlas_contract_qc": 15,
    "atlas_native_grid_qc_copy": 15,
    "automated_pre_tractography_qc": 15,
    "five_tt_dwi": 15,
    "gmwmi_dwi": 15,
    "mni_registration_fixed_image": 15,
    "mni_to_t1_nonlinear": 15,
    "pre_tractography_canary": 1,
    "spatial_qc_images": 15,
    "spatial_quantitative_qc": 15,
    "visual_review_bundle": 15,
    "total": 166,
}
FORBIDDEN_RULES = {
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


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
            "snakemake|eddy_cpu|eddy_cuda|dwifslpreproc|dwi2response|ss3t_csd_beta1|antsRegistration|c3d_affine_tool|tckgen|tcksift2",
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in text.splitlines() if "pgrep -af" not in line]


def validate_prior_failure() -> tuple[dict[str, Any], Path, Path, list[str]]:
    completion = load_json(PRIOR_COMPLETION)
    if (
        completion.get("record_type")
        != "pre_tractography_canary_recovery2_completion"
        or completion.get("status") != "FAIL"
        or completion.get("pre_tractography_summary")
        != {"expected": 15, "fail": 15, "ready_for_human_qc": 0, "terminal": 15}
        or completion.get("tractography_started") is not False
        or completion.get("matrix_generation_started") is not False
    ):
        raise ValueError("Recovery2 completion differs")
    end_record = completion.get("terminal_attempt_end")
    if not isinstance(end_record, Mapping):
        raise TypeError("Recovery2 completion lacks terminal attempt end")
    end_path = Path(str(end_record["path"])).resolve()
    if file_record(end_path) != end_record:
        raise ValueError("Recovery2 terminal attempt drifted")
    end = load_json(end_path)
    if end.get("status") != "FAIL" or end.get("returncode") != 1:
        raise ValueError("Recovery2 terminal result differs")
    log_record = end.get("combined_execution_log")
    if not isinstance(log_record, Mapping):
        raise TypeError("Recovery2 terminal attempt lacks log")
    log_path = Path(str(log_record["path"])).resolve()
    if file_record(log_path) != log_record:
        raise ValueError("Recovery2 execution log drifted")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    subjects = sorted(
        set(
            re.findall(
                r"/subjects/([^/]+)/04_atlas/aal3_nodes_166_dwi_1mm\.nii\.gz",
                text,
            )
        )
    )
    execution = completion.get("execution_binding", {})
    units = execution.get("units") if isinstance(execution, Mapping) else None
    if (
        Counter(re.findall(r"Error in rule ([A-Za-z0-9_]+):", text))
        != Counter({"atlas_contract_qc": 30, "five_tt_dwi": 30})
        or text.count("argument should be a str or an os.PathLike object") != 15
        or text.count("Input image does not conform to 5TT format") != 30
        or subjects != units
    ):
        raise ValueError("Recovery2 failure is not the exact atlas/5TT signature")
    dag_record = end.get("resolved_run_config")
    if not isinstance(dag_record, Mapping):
        raise TypeError("Recovery2 terminal attempt lacks resolved DAG config")
    dag_path = Path(str(dag_record["path"])).resolve()
    if file_record(dag_path) != dag_record:
        raise ValueError("Recovery2 resolved DAG config drifted")
    return end, end_path, log_path, subjects


def run_converter_tests() -> list[str]:
    spec = importlib.util.spec_from_file_location("recovery3_converter_tests", CONVERTER_TEST)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Recovery3 converter tests")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    passed = []
    for name in sorted(value for value in dir(module) if value.startswith("test_")):
        getattr(module, name)()
        passed.append(name)
    return passed


def build_reuse_inventory() -> dict[str, Any]:
    records = []
    for subject in sorted((RUN_ROOT / "subjects").iterdir()):
        if not subject.is_dir():
            continue
        for area in ("01_dwi", "02_anat", "03_spatial", "05_model"):
            root = subject / area
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_symlink() or not path.is_file() or path.name.endswith((".partial", ".tmp")):
                    continue
                relative_subject = path.relative_to(subject).as_posix()
                if relative_subject.startswith("03_spatial/mni5tt_to_t1_recovery3_"):
                    continue
                if relative_subject in {
                    "03_spatial/mni_registration_5tt_mask_recovery3.nii.gz",
                    "03_spatial/t1_5tt_brain_recovery3.nii.gz",
                    "05_model/5tt_dwi.mif",
                    "05_model/gmwmi_dwi.mif",
                    "05_model/5tt_dwi_recovery3.mif",
                    "05_model/5tt_dwi_recovery3_check.done",
                    "05_model/gmwmi_dwi_recovery3.mif",
                }:
                    continue
                record = file_record(path)
                record["relative_path"] = path.relative_to(RUN_ROOT).as_posix()
                records.append(record)
    for root in (RUN_ROOT / "05_group_response", RUN_ROOT / "frozen_calibration_retry4_v2"):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink() and not path.name.endswith((".partial", ".tmp")):
                record = file_record(path)
                record["relative_path"] = path.relative_to(RUN_ROOT).as_posix()
                records.append(record)
    return {
        "schema_version": "1.0.0",
        "record_type": "retry4_pretract_recovery3_reuse_inventory",
        "status": "PASS",
        "generated_utc": utc_now(),
        "run_root": str(RUN_ROOT),
        "reuse_policy": "hash_bound_successful_upstream_outputs_only",
        "excluded_output_families": [
            "subjects/*/03_spatial/mni5tt_to_t1_recovery3_*",
            "subjects/*/03_spatial/*recovery3*",
            "subjects/*/04_atlas/*",
            "subjects/*/06_preflight/*",
            "subjects/*/05_model/5tt_dwi.mif",
            "subjects/*/05_model/gmwmi_dwi.mif",
            "subjects/*/05_model/*recovery3*",
        ],
        "files": records,
        "file_count": len(records),
        "total_bytes": sum(int(record["size_bytes"]) for record in records),
    }


def output_count(name: str) -> int:
    return sum(1 for path in (RUN_ROOT / "subjects").glob(f"*/**/{name}") if path.is_file())


def main() -> int:
    if PACKAGE.exists():
        raise FileExistsError(f"refusing to overwrite package: {PACKAGE}")
    required = (
        CONFIG,
        ENVIRONMENT,
        SNAKEFILE,
        PRIOR_SNAKEFILE,
        SPATIAL_RULE,
        ATLAS_RULE,
        MODEL_RULE,
        PREFLIGHT_RULE,
        RECOVERY3_TEST,
        DIAGNOSTICS,
        CONVERTER_CONTRACT,
        CONVERTER_LOCK,
        CONVERTER_TEST,
        FINALIZER,
        LAUNCHER,
        AUTHORIZER,
        PHASE_A_COMPLETION,
        PRIOR_COMPLETION,
        RESPONSE_BINDING,
        PRETRACT_BINDING,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Recovery3 prerequisites missing: {missing}")
    active = active_imaging_processes()
    if active:
        raise RuntimeError("imaging process is active: " + " | ".join(active[:5]))
    end, end_path, execution_log, units = validate_prior_failure()
    diagnostics = load_json(DIAGNOSTICS)
    if (
        diagnostics.get("status") != "PASS"
        or diagnostics.get("units") != units
        or diagnostics.get("diagnosis_labels_used") is not False
        or diagnostics.get("tractography_executed") is not False
    ):
        raise ValueError("Recovery3 diagnostics differ")
    converter_tests = run_converter_tests()
    converter = yaml.safe_load(CONVERTER_CONTRACT.read_text(encoding="utf-8"))
    binary = Path(converter["binary"]["path"]).resolve()
    if sha256_file(binary) != converter["binary"]["sha256"]:
        raise ValueError("Recovery3 converter binary drifted after testing")
    successful_outputs = {
        "wmfod": output_count("wmfod.mif"),
        "wmfod_norm": output_count("wmfod_norm.mif"),
        "b0_to_t1_bbr_matrix": output_count("b0_to_t1_bbr.mat"),
        "b0_to_t1_bbr_image": output_count("b0_to_t1_bbr.nii.gz"),
        "t1_to_b0_itk": output_count("t1_to_b0_bbr_itk.txt"),
        "t1_to_b0_mrtrix": output_count("t1_to_b0_bbr_mrtrix.txt"),
    }
    if set(successful_outputs.values()) != {15}:
        raise ValueError(f"Recovery2 reusable stage counts differ: {successful_outputs}")
    reuse = build_reuse_inventory()

    environment = yaml.safe_load(ENVIRONMENT.read_text(encoding="utf-8"))
    dag_config = Path(str(end["resolved_run_config"]["path"])).resolve()
    command = [
        str(environment["workflow"]["snakemake"]["path"]),
        "--snakefile",
        str(SNAKEFILE),
        "--configfile",
        str(dag_config),
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
    dry = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    output = (dry.stdout or "") + ("\n" + dry.stderr if dry.stderr else "")
    scheduled = sorted(set(re.findall(r"(?m)^rule ([A-Za-z0-9_]+):\s*$", output)))
    counts: dict[str, int] = {}
    for rule in scheduled:
        match = re.search(rf"(?m)^{re.escape(rule)}\s+(\d+)\s*$", output)
        if match:
            counts[rule] = int(match.group(1))
    total = re.search(r"(?m)^total\s+(\d+)\s*$", output)
    counts["total"] = int(total.group(1)) if total else -1
    forbidden = sorted(set(scheduled) & FORBIDDEN_RULES)
    unexpected = sorted(set(scheduled) - set(ALLOWED_RULES))

    source_files = {
        "recovery3_snakefile": file_record(SNAKEFILE),
        "immutable_prior_snakefile": file_record(PRIOR_SNAKEFILE),
        "recovery3_spatial_rule": file_record(SPATIAL_RULE),
        "recovery3_atlas_rule": file_record(ATLAS_RULE),
        "recovery3_model_rule": file_record(MODEL_RULE),
        "recovery3_preflight_rule": file_record(PREFLIGHT_RULE),
        "recovery3_tests": file_record(RECOVERY3_TEST),
        "recovery3_diagnostics": file_record(DIAGNOSTICS),
        "converter_contract": file_record(CONVERTER_CONTRACT),
        "converter_lock": file_record(CONVERTER_LOCK),
        "converter_binary": file_record(binary),
        "converter_tests": file_record(CONVERTER_TEST),
        "terminal_finalizer": file_record(FINALIZER),
        "execution_launcher": file_record(LAUNCHER),
        "authorization_builder": file_record(AUTHORIZER),
        "package_builder": file_record(Path(__file__)),
    }
    PACKAGE.mkdir(parents=True, exist_ok=False)
    write_immutable_json(REUSE_INVENTORY, reuse)
    converter_report = {
        "schema_version": "1.0.0",
        "record_type": "recovery3_converter_validation",
        "status": "PASS",
        "generated_utc": utc_now(),
        "tested_units": [
            "003_S_4118_I1124861",
            "013_S_4268_I1075344",
            "014_S_6087_I926924",
            "126_S_6721_I1439616",
        ],
        "technical_coverage": ["GE", "Siemens", "nifti_single_t1", "dicom_series_t1"],
        "tests": converter_tests,
        "acceptance": {
            "fsl_itk_fsl_max_abs_matrix_delta": "<1e-3",
            "fsl_vs_ants_resampled_image_correlation": ">0.99",
            "normalized_mean_absolute_difference": "<0.02",
            "positive_support_dice": ">0.93",
        },
        "converter_contract": file_record(CONVERTER_CONTRACT),
        "converter_binary": file_record(binary),
        "converter_lock": file_record(CONVERTER_LOCK),
        "production_output_written": False,
    }
    write_immutable_json(CONVERTER_TEST_REPORT, converter_report)
    DRYRUN_LOG.write_text(output, encoding="utf-8")
    decision = {
        "schema_version": "1.0.0",
        "decision_type": "retry4_pretract_recovery3_spatial_and_5tt_approval",
        "status": "AWAITING_USER_APPROVAL",
        "decision_gate": "SL-H04A-CAL-R3",
        "required_user_response": REQUIRED_USER_RESPONSE,
        "run_root": str(RUN_ROOT),
        "units": units,
        "approved_unit_count": 15,
        "diagnosis_labels_used": False,
        "root_cause": "Recovery2 exposed cubic-interpolation overshoot in DWI-space 5TT images, a scalar-path bug in atlas QC, and poor MNI-to-native-T1 alignment caused by overinclusive BET masks and full-head registration inputs",
        "corrective_change": "use linear 5TT interpolation followed only by physical-range clipping; pass scalar image paths to nibabel; register skull-stripped MNI to a 5TT-derived native-T1 brain while retaining hard label-survival and spatial-overlap gates",
        "scientific_parameter_change": True,
        "scientific_change_scope": "registration inputs and mathematically valid 5TT interpolation only; atlas, parcellation, response calibration, FOD, tensor and tractography parameters unchanged",
        "prior_recovery2_completion": file_record(PRIOR_COMPLETION),
        "prior_resolved_dag_config": file_record(dag_config),
        "response_calibration_binding": file_record(RESPONSE_BINDING),
        "pretract_extension_binding": file_record(PRETRACT_BINDING),
        "converter_contract": file_record(CONVERTER_CONTRACT),
        "converter_validation": file_record(CONVERTER_TEST_REPORT),
        "candidate_diagnostics": file_record(DIAGNOSTICS),
        "reusable_output_inventory": file_record(REUSE_INVENTORY),
        "source_files": source_files,
        "allowed_rules": list(ALLOWED_RULES),
        "scheduled_job_counts": counts,
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
    disk = shutil.disk_usage(RUN_ROOT)
    storage = base.measure_run_root_storage(RUN_ROOT)
    effective_storage = int(storage["effective_bytes"])
    checks = {
        "prior_recovery2_terminal_fail_exact": True,
        "atlas_and_5tt_failure_signatures_confirmed": True,
        "candidate_diagnostics_pass": diagnostics.get("status") == "PASS",
        "all_expensive_upstream_stages_complete_15_of_15": set(successful_outputs.values()) == {15},
        "converter_contract_tests_still_pass": len(converter_tests) == 2,
        "converter_binary_hash_locked": sha256_file(binary) == converter["binary"]["sha256"],
        "reuse_inventory_nonempty": reuse["file_count"] > 0,
        "snakemake_dryrun_pass": dry.returncode == 0,
        "exact_166_job_residual_dag": counts == EXPECTED_COUNTS,
        "scheduled_rules_within_allowlist": not unexpected,
        "no_forbidden_rules_scheduled": not forbidden,
        "no_manifest_or_execution_preflight_overwrite": "freeze_manifest" not in scheduled and "execution_preflight" not in scheduled,
        "affine_converter_not_rescheduled": output.count(str(binary)) == 0,
        "legacy_crashing_converter_not_scheduled": output.count("/home/ec2-user/exp/.envs/convert3d-1.4.2/bin/c3d_affine_tool") == 0,
        "no_tractography_artifact": not any(RUN_ROOT.glob("**/*.tck")),
        "storage_below_stop": effective_storage < STORAGE_STOP_BYTES,
        "free_space_can_honor_stop": disk.free >= STORAGE_STOP_BYTES - effective_storage,
        "decision_remains_unapproved": decision["status"] == "AWAITING_USER_APPROVAL",
    }
    report = {
        "schema_version": "1.0.0",
        "record_type": "retry4_pretract_recovery3_package_validation",
        "generated_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "run_root": str(RUN_ROOT),
        "unit_count": 15,
        "successful_upstream_outputs": successful_outputs,
        "scheduled_rules": scheduled,
        "scheduled_job_counts": counts,
        "unexpected_scheduled_rules": unexpected,
        "forbidden_scheduled_rules": forbidden,
        "resource_envelope": {
            "maximum_cpu_cores": MAXIMUM_CORES,
            "maximum_cumulative_wall_clock_hours": WALL_CLOCK_HOURS,
            "total_storage_stop_bytes": STORAGE_STOP_BYTES,
            "current_effective_bytes": effective_storage,
            "current_storage": storage,
            "filesystem_free_bytes": disk.free,
        },
        "evidence": {
            "prior_recovery2_completion": file_record(PRIOR_COMPLETION),
            "prior_recovery2_attempt_end": file_record(end_path),
            "prior_recovery2_execution_log": file_record(execution_log),
            "prior_resolved_dag_config": file_record(dag_config),
            "response_binding": file_record(RESPONSE_BINDING),
            "pretract_binding": file_record(PRETRACT_BINDING),
            "converter_validation": file_record(CONVERTER_TEST_REPORT),
            "candidate_diagnostics": file_record(DIAGNOSTICS),
            "recovery3_decision_candidate": file_record(DECISION),
            "reuse_inventory": file_record(REUSE_INVENTORY),
            "dryrun_log": file_record(DRYRUN_LOG),
            **source_files,
        },
        "next_gate": "SL-H04A-CAL-R3",
        "required_user_response": REQUIRED_USER_RESPONSE,
        "imaging_executed": False,
    }
    write_immutable_json(VALIDATION, report)
    VALIDATION_MD.write_text(
        "\n".join(
            [
                "# Retry4 pre-tractography Recovery3 package",
                "",
                f"**Verdict:** {report['status']}",
                f"**Residual dry-run jobs:** {counts['total']}",
                "**Scientific change:** tissue-masked MNI-to-T1 registration and linear physical-range-safe 5TT resampling.",
                "**Unchanged:** atlas definition, response calibration, FOD/tensor parameters, tractography parameters and cohort scope.",
                "",
                "## Checks",
                "",
                *[f"- [{'x' if value else ' '}] {key}" for key, value in checks.items()],
                "",
                "Recovery3 reuses all bound upstream products and remains limited to atlas/tissue mapping plus automated and blinded pre-tractography QC. Tractography, SIFT2, matrices, statistics, dashboard refresh and full-cohort processing remain prohibited.",
                "",
                f"Execution requires the exact response: `{REQUIRED_USER_RESPONSE}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if report["status"] != "PASS":
        raise RuntimeError("Recovery3 package validation failed")
    print(json.dumps({"status": "PASS", "job_counts": counts, "next_gate": report["next_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
