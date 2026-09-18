#!/usr/bin/env python3
"""Fail-closed dry-run audit for the content-addressed H04A retry3 continuation.

Run only after retry2 is terminal and ``prepare_h04a_r1_retry3.py`` has created
the copy-on-write retry3 root and immutable package. The audit constructs the
same resolved configuration as the launcher, asks Snakemake for a dry run, and
proves that completed normalization/Eddy/bias/mask work and every downstream
FOD/tensor/tractography/matrix stage are absent from the planned DAG.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path("/home/ec2-user/exp")
AUDIT = ROOT / "research_audit"
WORKFLOW = ROOT / "scforge/workflow"
RUN_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry3")
PARENT_MANIFEST = AUDIT / "outputs/connectome_v2_input_manifest_v2.csv"
SUBSET_MANIFEST = (
    AUDIT / "outputs/h04a_r1_recovery_package_v1/recovery_execution_manifest_v1.csv"
)
BASE_PACKAGE = AUDIT / "outputs/h04a_r1_retry3_package_v1"
SAFETY_PACKAGE = AUDIT / "outputs/h04a_r1_retry3_package_v2"
DECISION = BASE_PACKAGE / "recovery_execution_decision_retry3.json"
PACKAGE_VALIDATION = SAFETY_PACKAGE / "retry3_package_validation.json"
SEED_MANIFEST = BASE_PACKAGE / "retry3_seed_manifest.json"
BINDING = AUDIT / "outputs/h04a_r1_recovery_extension_binding_v5.json"
CONFIG = ROOT / "configs/connectome_v2_retry3.yaml"
ENVIRONMENT = WORKFLOW / "environment_contract_retry3.yaml"
OUT = AUDIT / "outputs/h04a_r1_retry3_dryrun_v5"
SERVICE = "scforge-h04a-r1-phase-a-retry2.service"
KNOWN_TERMINAL_SERVICE_STATES = frozenset({"inactive", "failed"})

UPSTREAM_RECOMPUTATION_RULES = {
    "normalize_dwi_source",
    "normalize_t1_source",
    "input_contract_gate",
    "gradient_contract",
    "dwi_denoise",
    "dwi_degibbs",
    "dwi_motion_eddy",
    "dwi_bias_correct",
    "mean_b0",
    "dwi_brain_mask",
}
DOWNSTREAM_FORBIDDEN_RULES = {
    "five_tt_t1",
    "five_tt_wmseg",
    "five_tt_dwi",
    "gmwmi_dwi",
    "pooled_response",
    "select_tensor_shells",
    "fod_shell_compatibility_gate",
    "ss3t_csd",
    "mtnormalise",
    "tensor_fit",
    "tensor_metrics",
    "b0_one_mm_world_grid",
    "aal3_to_dwi_single_resample",
    "atlas_contract_qc",
    "atlas_native_grid_qc_copy",
    "mean_b0_nifti",
    "b0_to_t1_bbr",
    "invert_bbr_transform",
    "mni_to_t1_nonlinear",
    "spatial_qc_images",
    "spatial_quantitative_qc",
    "t1_in_b0_for_visual_qc",
    "visual_review_bundle",
    "automated_pre_tractography_qc",
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
    "pre_tractography_canary",
    "phase_b_subject_terminal_records",
    "all",
}
EXPECTED_CONTINUATION_RULES = {
    "select_fod_shells",
    "subject_response",
    "response_calibration_phase_a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def require_terminal_host() -> str:
    state = subprocess.run(
        ["systemctl", "--user", "is-active", SERVICE],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip() or "unknown"
    if state not in KNOWN_TERMINAL_SERVICE_STATES:
        raise RuntimeError(f"retry2 service is not known terminal: {state}")
    eddy = subprocess.run(
        ["pgrep", "-x", "eddy_cpu"], check=False, capture_output=True, text=True
    ).stdout.strip()
    if eddy:
        raise RuntimeError(f"eddy_cpu processes remain active: {eddy}")
    return state


def add_supported_dry_run_flags(command: list[str]) -> list[str]:
    """Add only flags supported by the locked Snakemake CLI."""
    dry_run_command = list(command)
    dry_run_command[-1:-1] = ["--dry-run"]
    return dry_run_command


def temporary_dry_run_workspace(run_root: Path = RUN_ROOT):
    """Create ephemeral launcher context inside the contract-bound run root."""
    if not run_root.is_dir():
        raise FileNotFoundError(f"retry3 run root is missing: {run_root}")
    return tempfile.TemporaryDirectory(
        prefix=".h04a_retry3_dryrun_",
        dir=run_root,
    )


def write_dry_run_contexts(
    workspace: Path,
    *,
    recipe_id: str,
    execution_binding: dict[str, object],
    recovery_extension_binding: dict[str, object],
    run_root: Path = RUN_ROOT,
) -> tuple[Path, Path]:
    """Materialize non-executable launcher identities required for DAG parsing."""
    run_id = "retry3-dry-run-validation"
    attempt_id = "retry3-dry-run-validation-attempt"
    created_utc = datetime.now(timezone.utc).isoformat()
    run_context_path = workspace / "run_context.json"
    attempt_context_path = workspace / "attempt_context.json"
    run_context = {
        "schema_version": "1.0.0",
        "status": "DRY_RUN_ONLY",
        "created_utc": created_utc,
        "run_id": run_id,
        "launcher_mode": "response-calibration-phase-a",
        "recipe_id": recipe_id,
        "run_root": str(run_root.resolve()),
        "execution_binding": copy.deepcopy(execution_binding),
        "recovery_extension_binding": copy.deepcopy(recovery_extension_binding),
    }
    attempt_context = {
        "schema_version": "1.0.0",
        "status": "DRY_RUN_ONLY",
        "started_utc": created_utc,
        "attempt_id": attempt_id,
        "launcher_mode": "response-calibration-phase-a",
        "run_id": run_id,
        "recipe_id": recipe_id,
        "execution_binding": copy.deepcopy(execution_binding),
        "recovery_extension_binding": copy.deepcopy(recovery_extension_binding),
    }
    run_context_path.write_text(
        json.dumps(run_context, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attempt_context_path.write_text(
        json.dumps(attempt_context, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_context_path, attempt_context_path


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite prior dry-run release: {OUT}")
    service_state = require_terminal_host()
    required = (
        RUN_ROOT,
        PARENT_MANIFEST,
        SUBSET_MANIFEST,
        DECISION,
        PACKAGE_VALIDATION,
        SEED_MANIFEST,
        BINDING,
        CONFIG,
        ENVIRONMENT,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"retry3 dry-run prerequisites are missing: {missing}")

    package_validation = json.loads(PACKAGE_VALIDATION.read_text(encoding="utf-8"))
    seed = json.loads(SEED_MANIFEST.read_text(encoding="utf-8"))
    if package_validation.get("status") != "PASS":
        raise ValueError("retry3 package validation is not PASS")
    if seed.get("stage_counts") != {
        "dwi_preproc": 15,
        "dwi_bias_corrected": 15,
        "dwi_brain_mask": 15,
    }:
        raise ValueError(f"retry3 seed is not complete through masking: {seed.get('stage_counts')}")

    sys.path.insert(0, str(WORKFLOW))
    sys.path.insert(0, str(ROOT / "scforge"))
    import run_h04a_r1_retry3 as wrapper
    from scforge.h04a_r1_retry3 import validate_recovery_extension_binding

    base = wrapper.base
    config = base.load_yaml(CONFIG)
    base.validate_normative_execution_lock(config)
    environment = base.load_yaml(ENVIRONMENT)
    binding = validate_recovery_extension_binding(
        base._load_json_mapping(BINDING, label="retry3 extension binding"),
        expected_run_root=RUN_ROOT,
    )
    parent_rows = list(
        base.load_acquisition_manifest(
            PARENT_MANIFEST,
            Path(config["inputs"]["acquisition_schema"]["path"]),
            run_root=RUN_ROOT,
            expected_rows=int(config["inputs"]["approved_pair_manifest"]["required_row_count"]),
        )
    )
    parent_rows.sort(key=lambda row: row["unit"])
    execution_binding, _ = base.prepare_execution_binding(
        recipe_id=str(config["contract"]["recipe_id"]),
        parent_manifest_path=PARENT_MANIFEST,
        parent_rows=parent_rows,
        execution_subset_manifest_path=SUBSET_MANIFEST,
        execution_subset_decision_path=DECISION,
        acquisition_schema_path=Path(config["inputs"]["acquisition_schema"]["path"]),
        run_root=RUN_ROOT,
    )
    if binding["execution_manifest"] != execution_binding["execution_subset_manifest"]:
        raise ValueError("retry3 binding and dry-run execution manifest differ")
    if binding["execution_decision"] != execution_binding["execution_subset_decision"]:
        raise ValueError("retry3 binding and dry-run decision differ")

    OUT.mkdir(parents=True, exist_ok=False)
    with temporary_dry_run_workspace() as directory:
        workspace = Path(directory)
        resolved_path = workspace / "resolved_config.yaml"
        run_context_path, attempt_context_path = write_dry_run_contexts(
            workspace,
            recipe_id=str(config["contract"]["recipe_id"]),
            execution_binding=execution_binding,
            recovery_extension_binding=binding,
        )
        resolved = copy.deepcopy(config)
        manifest_hash = sha256(PARENT_MANIFEST)
        resolved["manifest_path"] = str(PARENT_MANIFEST)
        resolved["run_root"] = str(RUN_ROOT)
        resolved["run_context_path"] = str(run_context_path)
        resolved["attempt_context_path"] = str(attempt_context_path)
        resolved["resolved_run_config_path"] = str(resolved_path)
        resolved["launcher_mode"] = "response-calibration-phase-a"
        resolved["execution_binding"] = copy.deepcopy(execution_binding)
        resolved["recovery_extension_binding"] = copy.deepcopy(binding)
        resolved["execution_manifest_path"] = execution_binding[
            "execution_subset_manifest"
        ]["path"]
        resolved["inputs"]["approved_pair_manifest"]["path"] = str(PARENT_MANIFEST)
        resolved["inputs"]["approved_pair_manifest"]["sha256"] = manifest_hash
        resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

        command = wrapper._retry3_build_snakemake_command(
            snakemake=Path(environment["workflow"]["snakemake"]["path"]),
            resolved_config_path=resolved_path,
            cores=32,
            mode="response-calibration-phase-a",
        )
        command = add_supported_dry_run_flags(command)
        run = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        output = (run.stdout or "") + ("\n" + run.stderr if run.stderr else "")
        (OUT / "snakemake_dry_run.txt").write_text(output, encoding="utf-8")
        if run.returncode != 0:
            raise RuntimeError(f"retry3 Snakemake dry run failed with {run.returncode}")

    rules = sorted(set(re.findall(r"(?m)^rule ([A-Za-z0-9_]+):\s*$", output)))
    upstream = sorted(set(rules) & UPSTREAM_RECOMPUTATION_RULES)
    downstream = sorted(set(rules) & DOWNSTREAM_FORBIDDEN_RULES)
    missing_expected = sorted(EXPECTED_CONTINUATION_RULES - set(rules))
    allowed_index = command.index("--allowed-rules")
    allowed_end = command.index("--keep-going")
    allowed_rules = command[allowed_index + 1 : allowed_end]
    checks = {
        "retry2_service_terminal": service_state in KNOWN_TERMINAL_SERVICE_STATES,
        "retry3_package_pass": package_validation.get("status") == "PASS",
        "seed_has_15_preproc": seed["stage_counts"]["dwi_preproc"] == 15,
        "seed_has_15_bias_corrected": seed["stage_counts"]["dwi_bias_corrected"] == 15,
        "seed_has_15_masks": seed["stage_counts"]["dwi_brain_mask"] == 15,
        "snakemake_dry_run_pass": run.returncode == 0,
        "no_upstream_recomputation_rules": not upstream,
        "no_downstream_forbidden_rules": not downstream,
        "all_phase_a_continuation_rules_present": not missing_expected,
        "rerun_triggers_are_content_based": command[command.index("--rerun-triggers") + 1 : command.index("--rerun-triggers") + 3]
        == ["input", "params"],
        "command_rule_allowlist_is_exact": set(allowed_rules)
        == EXPECTED_CONTINUATION_RULES
        and len(allowed_rules) == len(EXPECTED_CONTINUATION_RULES),
    }
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "service_state": service_state,
        "scheduled_rules": rules,
        "unexpected_upstream_rules": upstream,
        "forbidden_downstream_rules": downstream,
        "missing_expected_continuation_rules": missing_expected,
        "snakemake_command": command,
        "allowed_rules": allowed_rules,
        "inputs": {
            path.name: file_record(path)
            for path in (
                PARENT_MANIFEST,
                SUBSET_MANIFEST,
                DECISION,
                PACKAGE_VALIDATION,
                SEED_MANIFEST,
                BINDING,
                CONFIG,
                ENVIRONMENT,
            )
        },
    }
    (OUT / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# H04A retry3 content-addressed dry-run validation",
        "",
        f"**Generated:** {report['generated_utc']}",
        f"**Verdict:** {report['status']}",
        f"**Scheduled rules:** {', '.join(rules)}",
        "",
    ]
    for name, passed in checks.items():
        lines.append(f"- [{'x' if passed else ' '}] {name}")
    (OUT / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "scheduled_rules": rules}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
