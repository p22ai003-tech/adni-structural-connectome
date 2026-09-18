#!/usr/bin/env python3
"""Execute the approved retry4 15-unit pre-tractography canary only.

The launcher requires the immutable SL-H04A-CAL release, revalidates a dry-run,
and stops after automated QC plus blinded review bundles.  No tractography,
SIFT2, matrix, statistics, dashboard, preprocessing/Eddy, or response
re-estimation rule is reachable from its target.
"""

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
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
PACKAGE = AUDIT / "outputs/h04a_r1_retry4_pretract_package_v3"
CONFIG = ROOT / "configs/connectome_v2_retry3.yaml"
ENVIRONMENT = WORKFLOW / "environment_contract_retry3.yaml"
SNAKEFILE = WORKFLOW / "Snakefile_h04a_r1_retry4_pretract_v2"
PARENT_MANIFEST = AUDIT / "outputs/connectome_v2_input_manifest_v2.csv"
SUBSET_MANIFEST = (
    AUDIT
    / "outputs/h04a_r1_recovery_package_v1/recovery_execution_manifest_v1.csv"
)
RECOVERY_BINDING = AUDIT / "outputs/h04a_r1_recovery_extension_binding_v7.json"
PHASE_A_COMPLETION = RUN_ROOT / "publication/response_calibration_phase_a_completion.json"
RESPONSE_BINDING_PATH = PACKAGE / "response_calibration_binding.json"
PRETRACT_BINDING_PATH = PACKAGE / "pretract_execution_extension_binding.json"
APPROVAL_RELEASE = PACKAGE / "approval_release_validation.json"
PACKAGE_VALIDATION = PACKAGE / "validation.json"
COMPLETION_PATH = RUN_ROOT / "publication/pre_tractography_canary_completion.json"
RUN_CONTEXT_PATH = RUN_ROOT / "contract/pre_tractography_canary_run_context.json"
TARGET = "pre_tractography_canary"
MAXIMUM_CORES = 32
WALL_CLOCK_STOP_SECONDS = 24 * 60 * 60
STORAGE_STOP_BYTES = 150_000_000_000

sys.path.insert(0, str(ROOT / "scforge"))
sys.path.insert(0, str(WORKFLOW))

import run_h04a_r1_retry4 as retry4  # noqa: E402
from freeze_response_calibration_retry4 import (  # noqa: E402
    load_execution_subset_technical_metadata_retry4,
)
from scforge.h04a_r1_retry4_pretract import (  # noqa: E402
    ALLOWED_RULES,
    validate_pretract_extension_binding,
)


base = retry4.base
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


def pretract_resource_preflight(
    *, decision_sha256: str, requested_cores: int
) -> dict[str, Any]:
    if requested_cores < 1 or requested_cores > MAXIMUM_CORES:
        raise PermissionError(
            f"--cores must be in 1..{MAXIMUM_CORES}, found {requested_cores}"
        )
    prior_seconds = 0.0
    prior_attempts = []
    attempts_dir = RUN_ROOT / "attempts"
    for start_path in sorted(attempts_dir.glob("*.start.json")):
        start = load_json(start_path)
        if (
            start.get("launcher_mode") != "pre-tractography-canary"
            or start.get("pretract_decision", {}).get("sha256")
            != decision_sha256
        ):
            continue
        attempt_id = str(start.get("attempt_id", ""))
        end_path = attempts_dir / f"{attempt_id}.end.json"
        if not end_path.is_file():
            raise RuntimeError(
                f"unclosed pre-tractography attempt requires resolution: {start_path}"
            )
        end = load_json(end_path)
        duration = end.get("duration_seconds")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < 0
        ):
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, default=MAXIMUM_CORES)
    args = parser.parse_args()
    if COMPLETION_PATH.exists():
        raise FileExistsError(f"pre-tractography completion already exists: {COMPLETION_PATH}")
    for path in (
        CONFIG,
        ENVIRONMENT,
        SNAKEFILE,
        PARENT_MANIFEST,
        SUBSET_MANIFEST,
        RECOVERY_BINDING,
        PHASE_A_COMPLETION,
        RESPONSE_BINDING_PATH,
        PRETRACT_BINDING_PATH,
        APPROVAL_RELEASE,
        PACKAGE_VALIDATION,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    active = subprocess.run(
        [
            "pgrep",
            "-af",
            "snakemake|eddy_cpu|eddy_cuda|dwifslpreproc|dwi2response|ss3t_csd_beta1|tckgen|tcksift2",
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    active_lines = [line for line in active.splitlines() if "pgrep -af" not in line]
    if active_lines:
        raise RuntimeError("another imaging process is active: " + " | ".join(active_lines[:5]))

    package = load_json(PACKAGE_VALIDATION)
    release = load_json(APPROVAL_RELEASE)
    completion = load_json(PHASE_A_COMPLETION)
    execution_binding = completion.get("execution_binding")
    response_binding = load_json(RESPONSE_BINDING_PATH)
    pretract_binding = load_json(PRETRACT_BINDING_PATH)
    recovery_binding = load_json(RECOVERY_BINDING)
    if (
        package.get("status") != "PASS"
        or package.get("scheduled_job_counts", {}).get("total") != 394
        or package.get("forbidden_scheduled_rules") != []
        or release.get("status") != "PASS"
        or release.get("imaging_executed") is not False
        or release.get("pretract_extension_binding")
        != base.file_record(PRETRACT_BINDING_PATH)
        or release.get("response_calibration_binding")
        != base.file_record(RESPONSE_BINDING_PATH)
        or not isinstance(execution_binding, dict)
    ):
        raise PermissionError("SL-H04A-CAL release binding differs")
    validate_pretract_extension_binding(
        pretract_binding,
        expected_run_root=RUN_ROOT,
        execution_binding=execution_binding,
    )
    if pretract_binding.get("status") != "LOCKED":
        raise PermissionError("pre-tractography extension is not live-authorized")
    original_loader = base.load_execution_subset_technical_metadata
    base.load_execution_subset_technical_metadata = (
        load_execution_subset_technical_metadata_retry4
    )
    try:
        base.validate_response_calibration_diversity_binding(
            execution_binding=execution_binding,
            response_calibration_binding=response_binding,
        )
    finally:
        base.load_execution_subset_technical_metadata = original_loader

    pretract_decision_path = verify_record(
        pretract_binding["pretract_decision"], label="pre-tractography decision"
    )
    pretract_decision_sha256 = base.sha256_file(pretract_decision_path)
    resource = pretract_resource_preflight(
        decision_sha256=pretract_decision_sha256,
        requested_cores=args.cores,
    )

    normative = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    resolved = base.inject_phase_b_calibration(normative, response_binding)
    resolved["manifest_path"] = str(PARENT_MANIFEST)
    resolved["run_root"] = str(RUN_ROOT)
    resolved["launcher_mode"] = "pre-tractography-canary"
    resolved["execution_binding"] = copy.deepcopy(execution_binding)
    resolved["execution_manifest_path"] = execution_binding[
        "execution_subset_manifest"
    ]["path"]
    resolved["recovery_extension_binding"] = copy.deepcopy(recovery_binding)
    resolved["pretract_extension_binding"] = copy.deepcopy(pretract_binding)
    resolved["inputs"]["approved_pair_manifest"]["path"] = str(PARENT_MANIFEST)
    resolved["inputs"]["approved_pair_manifest"]["sha256"] = base.sha256_file(
        PARENT_MANIFEST
    )

    lineage_id = str(completion["lineage_id"])
    identity = {
        "lineage_id": lineage_id,
        "mode": "pre-tractography-canary",
        "phase_a_completion_sha256": base.sha256_file(PHASE_A_COMPLETION),
        "response_binding_sha256": base.sha256_file(RESPONSE_BINDING_PATH),
        "pretract_decision_sha256": pretract_decision_sha256,
    }
    run_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    run_context = {
        "schema_version": "1.0.0",
        "status": "LOCKED",
        "created_utc": base.utc_now(),
        "run_id": run_id,
        "lineage_id": lineage_id,
        "launcher_mode": "pre-tractography-canary",
        "recipe_id": completion["recipe_id"],
        "run_root": str(RUN_ROOT),
        "acquisition_manifest": base.file_record(PARENT_MANIFEST),
        "normative_config": base.file_record(CONFIG),
        "environment_contract": base.file_record(ENVIRONMENT),
        "workflow_source_manifest": base.file_record(
            Path(yaml.safe_load(ENVIRONMENT.read_text(encoding="utf-8"))[
                "workflow"
            ]["source_manifest"]["path"])
        ),
        "execution_binding": copy.deepcopy(execution_binding),
        "recovery_extension_binding": copy.deepcopy(recovery_binding),
        "response_calibration_binding": copy.deepcopy(response_binding),
        "pretract_extension_binding": copy.deepcopy(pretract_binding),
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
    }
    base.write_immutable_json(RUN_CONTEXT_PATH, run_context)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt_id = f"{timestamp}-pre-tractography-{os.urandom(4).hex()}"
    attempts = RUN_ROOT / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    resolved_path = attempts / f"{attempt_id}.resolved_config.yaml"
    attempt_start_path = attempts / f"{attempt_id}.start.json"
    attempt_end_path = attempts / f"{attempt_id}.end.json"
    dryrun_log_path = attempts / f"{attempt_id}.dryrun.log"
    execution_log_path = attempts / f"{attempt_id}.snakemake.log"
    resolved["run_context_path"] = str(RUN_CONTEXT_PATH)
    resolved["attempt_context_path"] = str(attempt_start_path)
    resolved["resolved_run_config_path"] = str(resolved_path)
    resolved_path.write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    live_command = command_for(resolved_path, cores=args.cores, dry_run=False)
    attempt_start = {
        "schema_version": "1.0.0",
        "status": "STARTED",
        "started_utc": base.utc_now(),
        "attempt_id": attempt_id,
        "run_id": run_id,
        "lineage_id": lineage_id,
        "launcher_mode": "pre-tractography-canary",
        "recipe_id": completion["recipe_id"],
        "run_context": base.file_record(RUN_CONTEXT_PATH),
        "resolved_run_config": base.file_record(resolved_path),
        "execution_binding": copy.deepcopy(execution_binding),
        "recovery_extension_binding": copy.deepcopy(recovery_binding),
        "response_calibration_binding": copy.deepcopy(response_binding),
        "pretract_extension_binding": copy.deepcopy(pretract_binding),
        "pretract_decision": base.file_record(pretract_decision_path),
        "pretract_resource_preflight": resource,
        "snakemake_invocation": live_command,
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
    }
    base.write_immutable_json(attempt_start_path, attempt_start)

    dry_command = command_for(resolved_path, cores=args.cores, dry_run=True)
    dry = subprocess.run(
        dry_command, cwd=ROOT, check=False, capture_output=True, text=True
    )
    dry_output = (dry.stdout or "") + ("\n" + dry.stderr if dry.stderr else "")
    with dryrun_log_path.open("x", encoding="utf-8") as handle:
        handle.write(dry_output)
    scheduled = set(
        re.findall(r"(?m)^rule ([A-Za-z0-9_]+):\s*$", dry_output)
    )
    forbidden = sorted(scheduled & FORBIDDEN_RULES)
    unexpected = sorted(scheduled - set(ALLOWED_RULES))
    if dry.returncode != 0 or forbidden or unexpected or TARGET not in scheduled:
        raise RuntimeError(
            "live preflight dry-run differs: "
            f"returncode={dry.returncode}, forbidden={forbidden}, unexpected={unexpected}"
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

    schema = Path(normative["inputs"]["acquisition_schema"]["path"])
    execution_rows = list(
        base.load_acquisition_manifest(
            SUBSET_MANIFEST, schema, run_root=RUN_ROOT
        )
    )
    execution_rows.sort(key=lambda row: row["unit"])
    manifest = base.finalize_pre_tractography_canary(
        run_root=RUN_ROOT,
        execution_rows=execution_rows,
        execution_binding=execution_binding,
        run_id=run_id,
        lineage_id=lineage_id,
        recipe_id=completion["recipe_id"],
        run_context_path=RUN_CONTEXT_PATH,
        attempt_start_path=attempt_start_path,
        attempt_end_path=attempt_end_path,
        execution_log_path=execution_log_path,
        workflow_returncode=returncode,
        ended_utc=ended.isoformat(),
    )
    manifest_path = RUN_ROOT / "publication/pre_tractography_canary_manifest.json"
    ledger_path = RUN_ROOT / "publication/pre_tractography_canary_ledger.jsonl"
    completion_record = {
        "schema_version": "2.0.0",
        "record_type": "pre_tractography_canary_completion",
        "status": manifest["run_outcome"],
        "workflow_state": "AWAITING_HUMAN_QC",
        "mode": "pre-tractography-canary",
        "run_id": run_id,
        "lineage_id": lineage_id,
        "recipe_id": completion["recipe_id"],
        "execution_binding": copy.deepcopy(execution_binding),
        "response_calibration_binding": copy.deepcopy(response_binding),
        "pretract_extension_binding": copy.deepcopy(pretract_binding),
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
