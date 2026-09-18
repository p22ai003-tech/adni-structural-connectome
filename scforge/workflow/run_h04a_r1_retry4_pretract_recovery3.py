#!/usr/bin/env python3
"""Execute the approved 15-unit retry4 pre-tractography Recovery3 only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


WORKFLOW = Path(__file__).resolve().parent
ROOT = WORKFLOW.parents[1]
AUDIT = ROOT / "research_audit"
RUN_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4")
PACKAGE = AUDIT / "outputs/h04a_r1_retry4_pretract_recovery3_package_v1"
PACKAGE_VALIDATION = PACKAGE / "validation.json"
APPROVAL_RELEASE = PACKAGE / "approval_release_validation.json"
RECOVERY3_BINDING_PATH = PACKAGE / "recovery3_execution_binding.json"
CONFIG = ROOT / "configs/connectome_v2_retry3.yaml"
ENVIRONMENT = WORKFLOW / "environment_contract_retry3.yaml"
SNAKEFILE = WORKFLOW / "Snakefile_h04a_r1_retry4_pretract_v5"
CONVERTER_CONTRACT = WORKFLOW / "environment_contract_recovery2_convert3d.yaml"
PARENT_MANIFEST = AUDIT / "outputs/connectome_v2_input_manifest_v2.csv"
SUBSET_MANIFEST = AUDIT / "outputs/h04a_r1_recovery_package_v1/recovery_execution_manifest_v1.csv"
PHASE_A_COMPLETION = RUN_ROOT / "publication/response_calibration_phase_a_completion.json"
PRIOR_PACKAGE = AUDIT / "outputs/h04a_r1_retry4_pretract_package_v3"
RESPONSE_BINDING_PATH = PRIOR_PACKAGE / "response_calibration_binding.json"
PRETRACT_BINDING_PATH = PRIOR_PACKAGE / "pretract_execution_extension_binding.json"
PRIOR_COMPLETION_PATH = RUN_ROOT / "publication/pre_tractography_canary_recovery2_completion.json"
COMPLETION_PATH = RUN_ROOT / "publication/pre_tractography_canary_recovery3_completion.json"
RUN_CONTEXT_PATH = RUN_ROOT / "contract/pre_tractography_canary_recovery3_run_context.json"
TARGET = "pre_tractography_canary"
MAXIMUM_CORES = 32
WALL_CLOCK_STOP_SECONDS = 24 * 60 * 60
STORAGE_STOP_BYTES = 150_000_000_000

sys.path.insert(0, str(ROOT / "scforge"))
sys.path.insert(0, str(WORKFLOW))

import run_h04a_r1_retry4 as retry4  # noqa: E402
from finalize_h04a_r1_retry4_pretract_recovery3 import (  # noqa: E402
    LEDGER_NAME,
    MANIFEST_NAME,
    finalize_recovery3,
)
from freeze_response_calibration_retry4 import (  # noqa: E402
    load_execution_subset_technical_metadata_retry4,
)
from scforge.h04a_r1_retry4_pretract import (  # noqa: E402
    ALLOWED_RULES as BASE_ALLOWED_RULES,
    validate_pretract_extension_binding,
)


base = retry4.base
ALLOWED_RULES = (*BASE_ALLOWED_RULES, "mni_registration_fixed_image")
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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def verify_record(record: Mapping[str, Any], *, label: str) -> Path:
    errors = base.verify_file_record(record)
    if errors:
        raise ValueError(f"{label} file record differs: {'; '.join(errors)}")
    return Path(str(record["path"])).expanduser().resolve()


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


def validate_recovery3_binding(
    binding: Mapping[str, Any], *, execution_binding: Mapping[str, Any]
) -> tuple[Path, Path]:
    if (
        binding.get("schema_version") != "1.0.0"
        or binding.get("record_type") != "retry4_pretract_recovery3_execution_binding"
        or binding.get("status") != "LOCKED"
        or binding.get("run_root") != str(RUN_ROOT)
        or binding.get("units") != execution_binding.get("units")
        or binding.get("imaging_execution_authorized") is not True
        or binding.get("diagnosis_labels_used") is not False
        or binding.get("tractography_authorized") is not False
        or binding.get("matrix_generation_authorized") is not False
        or binding.get("full_cohort_authorized") is not False
        or binding.get("next_human_gate") != "SL-H04B"
    ):
        raise PermissionError("Recovery3 execution binding differs")
    decision_path = verify_record(binding["recovery3_decision"], label="Recovery3 decision")
    decision = load_json(decision_path)
    if (
        decision.get("status") != "APPROVED"
        or decision.get("decision_gate") != "SL-H04A-CAL-R3"
        or decision.get("required_user_response") != "Approve SL-H04A-CAL-R3"
        or decision.get("user_response") != "Approve SL-H04A-CAL-R3"
        or decision.get("run_root") != str(RUN_ROOT)
        or decision.get("units") != execution_binding.get("units")
    ):
        raise PermissionError("Recovery3 exact approval differs")
    if binding.get("prior_recovery2_completion") != base.file_record(PRIOR_COMPLETION_PATH):
        raise ValueError("Recovery3 no longer binds the immutable Recovery2 completion")
    converter_path = verify_record(binding["converter_contract"], label="converter contract")
    if converter_path != CONVERTER_CONTRACT.resolve():
        raise ValueError("Recovery3 converter contract path differs")
    converter = yaml.safe_load(converter_path.read_text(encoding="utf-8"))
    binary = Path(str(converter.get("binary", {}).get("path", ""))).resolve()
    if (
        converter.get("status") != "TESTED_CANDIDATE"
        or converter.get("scientific_parameter_change") is not False
        or not binary.is_file()
        or base.sha256_file(binary) != converter.get("binary", {}).get("sha256")
    ):
        raise ValueError("Recovery3 converter binary differs")
    reuse_path = verify_record(binding["reusable_output_inventory"], label="reuse inventory")
    reuse = load_json(reuse_path)
    files = reuse.get("files")
    if (
        reuse.get("record_type") != "retry4_pretract_recovery3_reuse_inventory"
        or reuse.get("status") != "PASS"
        or not isinstance(files, list)
        or reuse.get("file_count") != len(files)
    ):
        raise ValueError("Recovery3 reuse inventory differs")
    for index, record in enumerate(files):
        if not isinstance(record, Mapping):
            raise TypeError(f"reuse record {index} is not a mapping")
        path = verify_record(record, label=f"reuse output {index}")
        if path.is_symlink() or record.get("relative_path") != path.relative_to(RUN_ROOT).as_posix():
            raise ValueError(f"reuse output contract differs: {path}")
    dag_config = verify_record(binding["prior_resolved_dag_config"], label="prior resolved DAG config")
    for label, record in binding.get("source_files", {}).items():
        verify_record(record, label=f"Recovery3 source {label}")
    return decision_path, dag_config


def resource_preflight(*, decision_sha256: str, requested_cores: int) -> dict[str, Any]:
    if requested_cores < 1 or requested_cores > MAXIMUM_CORES:
        raise PermissionError(f"--cores must be in 1..{MAXIMUM_CORES}")
    prior_seconds = 0.0
    prior_attempts = []
    attempts_dir = RUN_ROOT / "attempts"
    for start_path in sorted(attempts_dir.glob("*.start.json")):
        start = load_json(start_path)
        if start.get("launcher_mode") != "pre-tractography-canary":
            continue
        attempt_id = str(start.get("attempt_id", ""))
        end_path = attempts_dir / f"{attempt_id}.end.json"
        if not end_path.is_file():
            raise RuntimeError(f"unclosed pre-tractography attempt: {start_path}")
        end = load_json(end_path)
        duration = end.get("duration_seconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or float(duration) < 0:
            raise ValueError(f"prior attempt duration differs: {end_path}")
        prior_seconds += float(duration)
        prior_attempts.append(base.file_record(end_path))
    if prior_seconds >= WALL_CLOCK_STOP_SECONDS:
        raise RuntimeError("pre-tractography wall-clock cap is exhausted")
    storage = base.measure_run_root_storage(RUN_ROOT)
    if storage["effective_bytes"] >= STORAGE_STOP_BYTES:
        raise RuntimeError("pre-tractography storage cap is exhausted")
    remaining_storage = STORAGE_STOP_BYTES - int(storage["effective_bytes"])
    disk = shutil.disk_usage(RUN_ROOT)
    if disk.free < remaining_storage:
        raise RuntimeError("free disk cannot honor the signed storage stop")
    return {
        "schema_version": "1.0.0",
        "status": "PASS",
        "checked_utc": base.utc_now(),
        "decision_sha256": decision_sha256,
        "maximum_cpu_cores": MAXIMUM_CORES,
        "requested_cpu_cores": requested_cores,
        "prior_attempts": prior_attempts,
        "prior_duration_seconds": prior_seconds,
        "wall_clock_stop_seconds": WALL_CLOCK_STOP_SECONDS,
        "wall_clock_remaining_seconds": WALL_CLOCK_STOP_SECONDS - prior_seconds,
        "storage_stop_bytes": STORAGE_STOP_BYTES,
        "storage_remaining_bytes": remaining_storage,
        "storage": storage,
        "filesystem_free_bytes": disk.free,
    }


def command_for(config_path: Path, *, cores: int, dry_run: bool) -> list[str]:
    environment = yaml.safe_load(ENVIRONMENT.read_text(encoding="utf-8"))
    command = [
        str(environment["workflow"]["snakemake"]["path"]),
        "--snakefile",
        str(SNAKEFILE),
        "--configfile",
        str(config_path),
        "--cores",
        str(cores),
        "--rerun-incomplete",
        "--rerun-triggers",
        "input",
        "params",
        "--allowed-rules",
        *ALLOWED_RULES,
        "--keep-going",
        "--printshellcmds",
        "--show-failed-logs",
    ]
    if dry_run:
        command.append("--dry-run")
    command.append(TARGET)
    return command


def scheduled_contract(text: str) -> tuple[list[str], dict[str, int]]:
    rules = sorted(set(re.findall(r"(?m)^rule ([A-Za-z0-9_]+):\s*$", text)))
    counts: dict[str, int] = {}
    for rule in rules:
        match = re.search(rf"(?m)^{re.escape(rule)}\s+(\d+)\s*$", text)
        if match:
            counts[rule] = int(match.group(1))
    total = re.search(r"(?m)^total\s+(\d+)\s*$", text)
    counts["total"] = int(total.group(1)) if total else -1
    return rules, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, default=MAXIMUM_CORES)
    args = parser.parse_args()
    if COMPLETION_PATH.exists() or RUN_CONTEXT_PATH.exists():
        raise FileExistsError("Recovery3 completion or run context already exists")
    for path in (
        CONFIG,
        ENVIRONMENT,
        SNAKEFILE,
        CONVERTER_CONTRACT,
        PARENT_MANIFEST,
        SUBSET_MANIFEST,
        PHASE_A_COMPLETION,
        RESPONSE_BINDING_PATH,
        PRETRACT_BINDING_PATH,
        PRIOR_COMPLETION_PATH,
        PACKAGE_VALIDATION,
        APPROVAL_RELEASE,
        RECOVERY3_BINDING_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    active = active_imaging_processes()
    if active:
        raise RuntimeError("another imaging process is active: " + " | ".join(active[:5]))

    package = load_json(PACKAGE_VALIDATION)
    release = load_json(APPROVAL_RELEASE)
    completion = load_json(PHASE_A_COMPLETION)
    execution_binding = completion.get("execution_binding")
    response_binding = load_json(RESPONSE_BINDING_PATH)
    pretract_binding = load_json(PRETRACT_BINDING_PATH)
    prior_completion = load_json(PRIOR_COMPLETION_PATH)
    recovery3_binding = load_json(RECOVERY3_BINDING_PATH)
    if (
        package.get("status") != "PASS"
        or package.get("scheduled_job_counts", {}).get("total") != 166
        or package.get("forbidden_scheduled_rules") != []
        or release.get("status") != "PASS"
        or release.get("recovery3_execution_binding") != base.file_record(RECOVERY3_BINDING_PATH)
        or prior_completion.get("status") != "FAIL"
        or prior_completion.get("pre_tractography_summary") != {"expected": 15, "fail": 15, "ready_for_human_qc": 0, "terminal": 15}
        or not isinstance(execution_binding, dict)
    ):
        raise PermissionError("Recovery3 release binding differs")
    validate_pretract_extension_binding(
        pretract_binding,
        expected_run_root=RUN_ROOT,
        execution_binding=execution_binding,
    )
    decision_path, dag_config = validate_recovery3_binding(
        recovery3_binding, execution_binding=execution_binding
    )
    original_loader = base.load_execution_subset_technical_metadata
    base.load_execution_subset_technical_metadata = load_execution_subset_technical_metadata_retry4
    try:
        base.validate_response_calibration_diversity_binding(
            execution_binding=execution_binding,
            response_calibration_binding=response_binding,
        )
    finally:
        base.load_execution_subset_technical_metadata = original_loader

    decision_sha256 = base.sha256_file(decision_path)
    resource = resource_preflight(decision_sha256=decision_sha256, requested_cores=args.cores)
    identity = {
        "lineage_id": completion["lineage_id"],
        "mode": "pre-tractography-canary-recovery3",
        "prior_recovery2_completion_sha256": base.sha256_file(PRIOR_COMPLETION_PATH),
        "recovery3_decision_sha256": decision_sha256,
        "dag_config_sha256": base.sha256_file(dag_config),
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    run_context = {
        "schema_version": "1.0.0",
        "status": "LOCKED",
        "created_utc": base.utc_now(),
        "run_id": run_id,
        "lineage_id": completion["lineage_id"],
        "launcher_mode": "pre-tractography-canary",
        "recovery_mode": "pre-tractography-canary-recovery3",
        "recipe_id": completion["recipe_id"],
        "run_root": str(RUN_ROOT),
        "acquisition_manifest": base.file_record(PARENT_MANIFEST),
        "normative_config": base.file_record(CONFIG),
        "environment_contract": base.file_record(ENVIRONMENT),
        "prior_resolved_dag_config": base.file_record(dag_config),
        "execution_binding": copy.deepcopy(execution_binding),
        "response_calibration_binding": copy.deepcopy(response_binding),
        "pretract_extension_binding": copy.deepcopy(pretract_binding),
        "recovery3_execution_binding": copy.deepcopy(recovery3_binding),
        "prior_recovery2_completion": base.file_record(PRIOR_COMPLETION_PATH),
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
    }
    base.write_immutable_json(RUN_CONTEXT_PATH, run_context)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt_id = f"{timestamp}-pre-tractography-recovery3-{os.urandom(4).hex()}"
    attempts = RUN_ROOT / "attempts"
    attempt_start_path = attempts / f"{attempt_id}.start.json"
    attempt_end_path = attempts / f"{attempt_id}.end.json"
    dryrun_log_path = attempts / f"{attempt_id}.dryrun.log"
    execution_log_path = attempts / f"{attempt_id}.snakemake.log"
    live_command = command_for(dag_config, cores=args.cores, dry_run=False)
    attempt_start = {
        "schema_version": "1.0.0",
        "status": "STARTED",
        "started_utc": base.utc_now(),
        "attempt_id": attempt_id,
        "run_id": run_id,
        "lineage_id": completion["lineage_id"],
        "launcher_mode": "pre-tractography-canary",
        "recovery_mode": "pre-tractography-canary-recovery3",
        "recipe_id": completion["recipe_id"],
        "run_context": base.file_record(RUN_CONTEXT_PATH),
        "resolved_run_config": base.file_record(dag_config),
        "execution_binding": copy.deepcopy(execution_binding),
        "response_calibration_binding": copy.deepcopy(response_binding),
        "pretract_extension_binding": copy.deepcopy(pretract_binding),
        "recovery3_execution_binding": copy.deepcopy(recovery3_binding),
        "recovery3_decision": base.file_record(decision_path),
        "prior_recovery2_completion": base.file_record(PRIOR_COMPLETION_PATH),
        "pretract_resource_preflight": resource,
        "snakemake_invocation": live_command,
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
    }
    base.write_immutable_json(attempt_start_path, attempt_start)

    dry_command = command_for(dag_config, cores=args.cores, dry_run=True)
    dry = subprocess.run(dry_command, cwd=ROOT, check=False, capture_output=True, text=True)
    dry_output = (dry.stdout or "") + ("\n" + dry.stderr if dry.stderr else "")
    with dryrun_log_path.open("x", encoding="utf-8") as handle:
        handle.write(dry_output)
    scheduled, counts = scheduled_contract(dry_output)
    forbidden = sorted(set(scheduled) & FORBIDDEN_RULES)
    unexpected = sorted(set(scheduled) - set(ALLOWED_RULES))
    if (
        dry.returncode != 0
        or forbidden
        or unexpected
        or counts != package.get("scheduled_job_counts")
        or dry_output.count(
            str(yaml.safe_load(CONVERTER_CONTRACT.read_text())["binary"]["path"])
        )
        != 0
    ):
        raise RuntimeError(
            f"Recovery3 live dry-run differs: returncode={dry.returncode}, counts={counts}, forbidden={forbidden}, unexpected={unexpected}"
        )

    returncode, started, ended, runtime = base.execute_command_with_h04a_monitor(
        command=live_command,
        cwd=ROOT,
        execution_log_path=execution_log_path,
        run_root=RUN_ROOT,
        resource_preflight=resource,
    )
    attempt_end = {
        **attempt_start,
        "status": "PASS" if returncode == 0 else "FAIL",
        "ended_utc": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
        "returncode": returncode,
        "combined_execution_log": base.file_record(execution_log_path),
        "preflight_dryrun": base.file_record(dryrun_log_path),
        "resource_runtime": runtime,
    }
    base.write_immutable_json(attempt_end_path, attempt_end)

    normative = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    schema = Path(normative["inputs"]["acquisition_schema"]["path"])
    execution_rows = list(base.load_acquisition_manifest(SUBSET_MANIFEST, schema, run_root=RUN_ROOT))
    execution_rows.sort(key=lambda row: row["unit"])
    manifest = finalize_recovery3(
        base=base,
        run_root=RUN_ROOT,
        execution_rows=execution_rows,
        execution_binding=execution_binding,
        run_id=run_id,
        lineage_id=completion["lineage_id"],
        recipe_id=completion["recipe_id"],
        run_context_path=RUN_CONTEXT_PATH,
        attempt_start_path=attempt_start_path,
        attempt_end_path=attempt_end_path,
        execution_log_path=execution_log_path,
        workflow_returncode=returncode,
        ended_utc=ended.isoformat(),
        prior_failure_completion=base.file_record(PRIOR_COMPLETION_PATH),
    )
    manifest_path = RUN_ROOT / "publication" / MANIFEST_NAME
    ledger_path = RUN_ROOT / "publication" / LEDGER_NAME
    workflow_state = "AWAITING_HUMAN_QC" if manifest["summary"]["ready_for_human_qc"] > 0 else "RECOVERY_REQUIRED"
    completion_record = {
        "schema_version": "2.0.0",
        "record_type": "pre_tractography_canary_recovery3_completion",
        "status": manifest["run_outcome"],
        "workflow_state": workflow_state,
        "mode": "pre-tractography-canary-recovery3",
        "run_id": run_id,
        "lineage_id": completion["lineage_id"],
        "recipe_id": completion["recipe_id"],
        "execution_binding": copy.deepcopy(execution_binding),
        "response_calibration_binding": copy.deepcopy(response_binding),
        "pretract_extension_binding": copy.deepcopy(pretract_binding),
        "recovery3_execution_binding": copy.deepcopy(recovery3_binding),
        "prior_recovery2_completion": base.file_record(PRIOR_COMPLETION_PATH),
        "pre_tractography_summary": manifest["summary"],
        "pre_tractography_manifest": base.file_record(manifest_path),
        "pre_tractography_ledger": base.file_record(ledger_path),
        "run_context": base.file_record(RUN_CONTEXT_PATH),
        "terminal_attempt_start": base.file_record(attempt_start_path),
        "terminal_attempt_end": base.file_record(attempt_end_path),
        "tractography_started": False,
        "matrix_generation_started": False,
        "human_qc_required": True,
    }
    base.write_immutable_json(COMPLETION_PATH, completion_record)
    print(json.dumps(completion_record, indent=2, sort_keys=True))
    return 0 if returncode == 0 and manifest["run_outcome"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
