#!/usr/bin/env python3
"""Audit the connectome-v2 execution contract without running image processing.

The validator has three deliberately separate jobs:

* verify every declared host artifact against its exact SHA-256;
* verify that ``connectome_v2.yaml`` and ``environment_contract.yaml`` describe
  the same recipe and fail-closed executor policy; and
* emulate the workflow's current PATH resolution to prove that the locked
  executables, rather than similarly named alternatives, would be selected.

It is safe to run on a production host: it reads metadata, hashes files, asks
tools for version strings, and inspects workflow source.  It never opens image
inputs and never starts a Snakemake or neuroimaging job.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENVIRONMENT = Path(__file__).resolve().with_name("environment_contract.yaml")
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "connectome_v2.yaml"
DEFAULT_SNAKEFILE = Path(__file__).resolve().with_name("Snakefile")

SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_hashed_artifact(path: Path, expected: str) -> tuple[bool, str | None]:
    """Return whether an artifact exists and has the exact declared digest."""

    if not path.is_file():
        return False, None
    observed = sha256_file(path)
    return bool(SHA256_RE.fullmatch(str(expected).lower()) and observed == str(expected).lower()), observed


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return data


def _nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _check(
    checks: list[dict[str, Any]],
    code: str,
    passed: bool,
    detail: str,
    *,
    severity: str = "error",
    expected: Any = None,
    observed: Any = None,
) -> None:
    row: dict[str, Any] = {
        "code": code,
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
    }
    if expected is not None:
        row["expected"] = expected
    if observed is not None:
        row["observed"] = observed
    checks.append(row)


def _run(command: list[str], timeout: int = 10) -> tuple[int | None, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - host diagnostic
        return None, f"{type(exc).__name__}:{exc}"
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return completed.returncode, output


def _hashed_artifacts(environment: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any], bool]]:
    """Yield (logical name, record, executable expected) for declared locks."""

    for name in (
        "snakemake",
        "requirements_lock",
        "wheel_artifact_manifest",
        "source_manifest",
    ):
        record = _nested(environment, "workflow", name)
        if isinstance(record, dict):
            yield f"workflow.{name}", record, name == "snakemake"

    for section in ("mrtrix3", "mrtrix3tissue", "fsl", "ants", "convert3d"):
        binaries = _nested(environment, section, "binaries", default={})
        if isinstance(binaries, dict):
            for name, record in sorted(binaries.items()):
                if isinstance(record, dict):
                    yield f"{section}.binaries.{name}", record, True

    for section, key in (
        ("ants", "explicit_conda_lock"),
        ("convert3d", "explicit_conda_lock"),
    ):
        record = _nested(environment, section, key)
        if isinstance(record, dict):
            yield f"{section}.{key}", record, False

    mrtrix_python = _nested(environment, "execution_policy", "mrtrix_script_python")
    if isinstance(mrtrix_python, dict) and mrtrix_python.get("sha256"):
        yield "execution_policy.mrtrix_script_python", mrtrix_python, True


def policy_consistency_checks(
    environment: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate cross-file identity and fail-closed policy invariants."""

    checks: list[dict[str, Any]] = []
    env_recipe = environment.get("recipe_id")
    cfg_recipe = _nested(config, "contract", "recipe_id")
    _check(
        checks,
        "identity.recipe_id",
        env_recipe == cfg_recipe,
        "The environment and scientific recipe IDs must be identical.",
        expected=cfg_recipe,
        observed=env_recipe,
    )
    _check(
        checks,
        "identity.frozen_timestamp",
        environment.get("generated_utc") == _nested(config, "contract", "frozen_utc"),
        "The environment snapshot must be frozen with the same recipe revision.",
        expected=_nested(config, "contract", "frozen_utc"),
        observed=environment.get("generated_utc"),
    )

    env_path = _nested(config, "environment", "locked_paths", "environment_contract")
    _check(
        checks,
        "identity.environment_contract_path",
        Path(str(env_path)).resolve() == DEFAULT_ENVIRONMENT.resolve() if env_path else False,
        "The recipe must name the normative environment contract explicitly.",
        expected=str(DEFAULT_ENVIRONMENT.resolve()),
        observed=env_path,
    )

    env_source_manifest = _nested(environment, "workflow", "source_manifest", "path")
    config_source_manifest = _nested(
        config, "environment", "locked_paths", "workflow_source_manifest"
    )
    _check(
        checks,
        "identity.workflow_source_manifest_path",
        bool(
            env_source_manifest
            and config_source_manifest
            and Path(str(env_source_manifest)).resolve()
            == Path(str(config_source_manifest)).resolve()
        ),
        "Both contracts must name the same immutable workflow-source manifest.",
        expected=env_source_manifest,
        observed=config_source_manifest,
    )

    fail_closed = (
        _nested(config, "contract", "failure_policy") == "fail_closed"
        and _nested(config, "contract", "silent_fallbacks_allowed") is False
        and _nested(environment, "execution_policy", "fail_closed") is True
        and _nested(environment, "execution_policy", "automatic_tool_or_algorithm_fallback") is False
    )
    _check(
        checks,
        "policy.fail_closed",
        fail_closed,
        "Both contracts must forbid silent tool and algorithm fallback.",
    )

    config_eddy = _nested(config, "dwi_preprocessing", "eddy", "executable")
    policy_eddy = _nested(environment, "execution_policy", "eddy_executor")
    locked_eddy = _nested(environment, "fsl", "binaries", "eddy_cpu", "path")
    eddy_flags = (
        _nested(config, "dwi_preprocessing", "eddy", "executor_policy")
        == "explicit_cpu_for_entire_recipe_no_runtime_switch"
        and _nested(config, "dwi_preprocessing", "eddy", "automatic_cuda_to_cpu_fallback_allowed") is False
        and _nested(environment, "execution_policy", "cuda_executor_allowed") is False
    )
    _check(
        checks,
        "policy.eddy_single_cpu_executor",
        bool(config_eddy and config_eddy == policy_eddy == locked_eddy and eddy_flags),
        "One exact CPU eddy executable must be selected for every subject, with CUDA and runtime switching disabled.",
        expected=locked_eddy,
        observed={"config": config_eddy, "environment": policy_eddy},
    )

    ss3t_python = _nested(environment, "execution_policy", "ss3t_invocation", "python")
    ss3t_script = _nested(environment, "execution_policy", "ss3t_invocation", "script")
    _check(
        checks,
        "policy.ss3t_runtime",
        ss3t_python == _nested(config, "environment", "locked_paths", "ss3t_python")
        and ss3t_script == _nested(environment, "mrtrix3tissue", "binaries", "ss3t_csd_beta1", "path"),
        "SS3T must have one explicit Python runtime and one exact script.",
        observed={"python": ss3t_python, "script": ss3t_script},
    )
    mrtrix_script_python = _nested(environment, "execution_policy", "mrtrix_script_python", "path")
    _check(
        checks,
        "policy.mrtrix_script_runtime",
        bool(
            mrtrix_script_python
            and mrtrix_script_python
            == _nested(config, "environment", "locked_paths", "mrtrix_script_python")
            and _nested(environment, "execution_policy", "mrtrix_script_python", "sha256")
        ),
        "MRtrix env-python scripts must use one explicitly hashed runtime shared by the recipe and environment lock.",
        observed=mrtrix_script_python,
    )

    required = _nested(config, "environment", "required_versions", default={})
    current = _nested(config, "environment", "current_host_observation", default={})
    version_pairs = {
        "snakemake": _nested(environment, "workflow", "snakemake", "version"),
        "mrtrix3": _nested(environment, "mrtrix3", "version"),
        "fsl": _nested(environment, "fsl", "version"),
        "ants": _nested(environment, "ants", "version"),
    }
    for name, observed in version_pairs.items():
        declared = required.get(name) if isinstance(required, dict) else None
        exact = declared == observed
        _check(
            checks,
            f"versions.config_exact_{name}",
            exact,
            "Frozen required versions should be exact, not a broad release family.",
            severity="warning" if name == "fsl" else "error",
            expected=observed,
            observed=declared,
        )
        if name in {"mrtrix3", "fsl", "ants"}:
            _check(
                checks,
                f"versions.current_host_{name}",
                current.get(name) == observed if isinstance(current, dict) else False,
                "The recipe's current-host observation must equal the environment snapshot.",
                expected=observed,
                observed=current.get(name) if isinstance(current, dict) else None,
            )
    return checks


def _extract_default(source: str, variable: str, config_key: str) -> str | None:
    pattern = re.compile(
        rf"{re.escape(variable)}\s*=\s*Path\(\s*config\.get\(\s*['\"]{re.escape(config_key)}['\"]\s*,\s*['\"]([^'\"]+)['\"]",
        re.DOTALL,
    )
    match = pattern.search(source)
    return match.group(1) if match else None


def workflow_integration_checks(
    environment: dict[str, Any],
    config: dict[str, Any],
    snakefile: Path = DEFAULT_SNAKEFILE,
) -> list[dict[str, Any]]:
    """Prove the current workflow resolves the executables promised by the lock."""

    checks: list[dict[str, Any]] = []
    rule_paths = sorted((snakefile.parent / "rules").glob("*.smk"))
    source = snakefile.read_text(encoding="utf-8")
    rules_source = "\n".join(path.read_text(encoding="utf-8") for path in rule_paths)
    combined = source + "\n" + rules_source

    source_manifest_record = _nested(environment, "workflow", "source_manifest")
    source_manifest_path = (
        Path(str(source_manifest_record.get("path"))).expanduser()
        if isinstance(source_manifest_record, dict) and source_manifest_record.get("path")
        else None
    )
    required_source_paths = {
        snakefile.resolve(),
        *(path.resolve() for path in rule_paths),
        Path(__file__).resolve(),
        snakefile.with_name("smoke_dryrun.py").resolve(),
        snakefile.with_name("run_connectome_v2.py").resolve(),
        snakefile.parent / "schemas" / "acquisition_manifest_v2.schema.json",
        snakefile.parent / "schemas" / "provenance_v2.schema.json",
        snakefile.parent / "schemas" / "run_ledger_v2.schema.json",
        snakefile.with_name("freeze_response_calibration.py").resolve(),
        PROJECT_ROOT / "configs" / "connectome_v2.yaml",
        PROJECT_ROOT / "research_audit" / "matrix_data_dictionary.md",
        PROJECT_ROOT / "research_audit" / "validate_connectome_v2_contract.py",
        PROJECT_ROOT / "scforge" / "scforge" / "__init__.py",
        PROJECT_ROOT / "scforge" / "scforge" / "qc.py",
        PROJECT_ROOT / "scforge" / "scforge" / "connectome.py",
        PROJECT_ROOT / "scforge" / "scforge" / "config.py",
        PROJECT_ROOT / "scforge" / "scforge" / "input_contract.py",
        PROJECT_ROOT / "scforge" / "scforge" / "provenance.py",
        PROJECT_ROOT / "scforge" / "scforge" / "response_calibration.py",
        PROJECT_ROOT / "scforge" / "scforge" / "tractography.py",
    }
    manifest_rows: list[dict[str, str]] = []
    manifest_error: str | None = None
    if source_manifest_path and source_manifest_path.is_file():
        try:
            with source_manifest_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if reader.fieldnames != ["path", "sha256"]:
                    raise ValueError(
                        "expected exactly the tab-delimited columns path and sha256"
                    )
                manifest_rows = list(reader)
        except Exception as exc:
            manifest_error = f"{type(exc).__name__}:{exc}"
    else:
        manifest_error = "missing"

    observed_source_paths: set[Path] = set()
    bad_source_rows: list[dict[str, str]] = []
    for row in manifest_rows:
        declared_path = str(row.get("path", "")).strip()
        expected_hash = str(row.get("sha256", "")).strip().lower()
        candidate = Path(declared_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        candidate = candidate.resolve()
        observed_source_paths.add(candidate)
        inside_project = candidate.is_relative_to(PROJECT_ROOT.resolve())
        passed, observed_hash = (
            validate_hashed_artifact(candidate, expected_hash)
            if declared_path and inside_project
            else (False, None)
        )
        if not passed:
            bad_source_rows.append(
                {
                    "path": declared_path,
                    "expected": expected_hash,
                    "observed": observed_hash or ("outside_project" if not inside_project else "missing"),
                }
            )
    missing_source_paths = sorted(
        str(path) for path in required_source_paths - observed_source_paths
    )
    duplicate_source_paths = len(observed_source_paths) != len(manifest_rows)
    declared_source_rows = (
        source_manifest_record.get("rows")
        if isinstance(source_manifest_record, dict)
        else None
    )
    _check(
        checks,
        "workflow.source_manifest_complete_and_current",
        bool(
            manifest_rows
            and not manifest_error
            and not bad_source_rows
            and not missing_source_paths
            and not duplicate_source_paths
            and declared_source_rows == len(manifest_rows)
        ),
        "The locked source manifest must cover the Snakefile, every rule, validators, and connectome/QC modules with current SHA-256 values.",
        expected={"minimum_files": len(required_source_paths), "missing": []},
        observed={
            "rows": len(manifest_rows),
            "parse_error": manifest_error,
            "bad_rows": bad_source_rows,
            "missing": missing_source_paths,
            "duplicates": duplicate_source_paths,
            "declared_rows": declared_source_rows,
        },
    )
    _check(
        checks,
        "workflow.source_manifest_runtime_enforced",
        bool(
            "workflow_source" in rules_source
            and "workflow source SHA-256 mismatch" in rules_source
            and "workflow_source_files" in rules_source
        ),
        "Execution preflight must rehash every workflow source row before any imaging rule starts.",
    )
    _check(
        checks,
        "workflow.terminal_records_write_once_and_schema_validated",
        bool(
            "validate_schema(record, input.provenance_schema)" in rules_source
            and "write_immutable_json(output[0], record)" in rules_source
            and "terminal_record.json" in rules_source
            and "run_ledger.jsonl" in source
            and "run_ledger_manifest.json" in source
            and "outcome_validity_manifest.csv" in source
        ),
        "Per-unit terminal outcomes and the mixed-outcome run ledger must be schema-validated, write-once evidence outputs.",
    )
    _check(
        checks,
        "workflow.exact_command_capture_launcher",
        bool(
            snakefile.with_name("run_connectome_v2.py").is_file()
            and "--printshellcmds" in snakefile.with_name("run_connectome_v2.py").read_text(encoding="utf-8")
            and "combined_execution_log" in snakefile.with_name("run_connectome_v2.py").read_text(encoding="utf-8")
        ),
        "The supported launcher must capture expanded Snakemake shell commands and immutably attest the final combined log.",
    )
    launcher_source = snakefile.with_name("run_connectome_v2.py").read_text(encoding="utf-8")
    response_calibration_source = (
        PROJECT_ROOT / "scforge" / "scforge" / "response_calibration.py"
    ).read_text(encoding="utf-8")
    _check(
        checks,
        "workflow.h04a_signed_bounded_execution",
        bool(
            "validate_normative_execution_lock" in launcher_source
            and "require_h04a_mode_authorization" in launcher_source
            and "prepare_h04a_resource_preflight" in launcher_source
            and "execute_command_with_h04a_monitor" in launcher_source
            and "start_new_session=True" in launcher_source
            and "H04A cannot authorize launcher mode" in launcher_source
            and "signed H04A authorization" in rules_source
            and "attempt lacks a valid signed H04A resource preflight"
            in rules_source
        ),
        "The immutable recipe must remain locked while exact signed H04A evidence authorizes only bounded pre-tractography modes with core, time, and storage enforcement.",
    )
    _check(
        checks,
        "workflow.h04a_valid_pool_technical_diversity",
        bool(
            "H04A_MINIMUM_VALID_MANUFACTURER_FAMILIES = 2" in launcher_source
            and "H04A_MINIMUM_VALID_T1_SOURCE_CLASSES = 2" in launcher_source
            and "H04A_REQUIRED_VALID_T1_SOURCE_CLASSES" in launcher_source
            and "build_valid_pool_technical_diversity" in launcher_source
            and "validate_response_calibration_diversity_binding"
            in launcher_source
            and "valid_pool_technical_diversity_sha256"
            in response_calibration_source
            and "insufficient manufacturer-family diversity"
            in response_calibration_source
            and "exact required T1 source" in response_calibration_source
            and "expected_technical_diversity" in rules_source
        ),
        "The exact signed H04A manufacturer/T1 diversity contract must stop pooled-response freeze unless satisfied and must be recomputed at phase-B and continuation.",
    )
    _check(
        checks,
        "workflow.mixed_failure_finalization",
        bool(
            "--keep-going" in launcher_source
            and "--mode" in launcher_source
            and "response-calibration-phase-a" in launcher_source
            and "phase-b" in launcher_source
            and "finalize_terminal_publication" in launcher_source
            and "normalization_failure_evidence" in launcher_source
            and "outcome_validity_manifest.csv" in launcher_source
        ),
        "The launcher must finalize all authorized units after mixed failures, using structured normalization markers before log fallback.",
    )
    _check(
        checks,
        "workflow.rng_seed_uses_locked_source_identity",
        bool(
            "dti_source_id" in rules_source
            and "dti_raw_bundle_sha256" in rules_source
            and "rng_seed_token" in rules_source
            and "MRTRIX_RNG_SEED" in rules_source
        ),
        "The recorded tractography seed must derive from stable DTI source ID, locked raw bundle hash, and recipe ID.",
    )
    pooled_rule_match = re.search(
        r"(?ms)^rule pooled_response:\s*(.*?)(?=^rule\s|\Z)", rules_source
    )
    pooled_rule_source = pooled_rule_match.group(1) if pooled_rule_match else ""
    _check(
        checks,
        "workflow.phase_b_response_noncontagion",
        bool(
            pooled_rule_source
            and "FROZEN_RESPONSE_MANIFEST" in pooled_rule_source
            and "validate_frozen_response_calibration" in pooled_rule_source
            and "UNITS" not in pooled_rule_source
        ),
        "Phase B must consume one frozen diagnosis-blind calibration manifest without live all-unit response dependencies.",
    )
    _check(
        checks,
        "workflow.post_conversion_eddy_metadata_gate",
        bool(
            "source_phase_encoding_token" in rules_source
            and "dwi_source_metadata.json" in rules_source
            and "source_metadata=rules.normalize_dwi_source.output.metadata" in rules_source
            and _nested(config, "dwi_preprocessing", "phase_encoding", "hardcoded_default_allowed") is False
            and _nested(config, "dwi_preprocessing", "phase_encoding", "inferred_default_allowed") is False
        ),
        "PE direction/readout must come from normalized source metadata and must never use guessed polarity/defaults.",
    )

    _check(
        checks,
        "workflow.environment_contract_consumed",
        "environment_contract" in source or "environment_contract" in rules_source,
        "The preflight must consume, hash, and validate environment_contract.yaml; a config-only path is not enforcement.",
    )
    _check(
        checks,
        "workflow.environment_hash_verified",
        bool(re.search(r"environment_contract.*sha256|sha256.*environment_contract", combined, re.DOTALL)),
        "The exact environment-contract bytes must be recorded and checked before execution.",
    )
    _check(
        checks,
        "workflow.mni_template_explicit_config",
        bool(re.search(r"config\[['\"]registration['\"]\]\[['\"]mni_to_t1['\"]\]\[['\"]template['\"]\]", source)),
        "MNI_TEMPLATE must come from registration.mni_to_t1.template, not a parallel top-level default.",
        expected=_nested(config, "registration", "mni_to_t1", "template"),
    )
    _check(
        checks,
        "workflow.mni_template_hash_enforced",
        "template_sha256" in rules_source and "MNI_TEMPLATE" in rules_source,
        "Execution preflight must compare the exact MNI template bytes to registration.mni_to_t1.template_sha256.",
        expected=_nested(config, "registration", "mni_to_t1", "template_sha256"),
    )
    ambient_path_exports = re.findall(r"export\s+PATH=\{TOOL_PATH:q\}:\$PATH", rules_source)
    _check(
        checks,
        "workflow.no_ambient_path_fallback",
        not ambient_path_exports,
        "Locked rules must export TOOL_PATH exactly; appending ambient PATH permits undeclared executable fallback.",
        expected=0,
        observed=len(ambient_path_exports),
    )
    placeholders = [
        line.strip()
        for line in combined.splitlines()
        if re.search(r"\bplaceholder\b|\bnot implemented\b", line, re.IGNORECASE)
    ]
    _check(
        checks,
        "workflow.no_placeholders",
        not placeholders,
        "The executable DAG cannot contain placeholder or not-implemented rules.",
        expected=[],
        observed=placeholders,
    )
    for rule_name in ("matrix_qc", "provenance_sidecar", "publish_manifest"):
        _check(
            checks,
            f"workflow.required_rule_{rule_name}",
            bool(re.search(rf"(?m)^rule\s+{re.escape(rule_name)}\s*:", rules_source)),
            "Every rule named by rule all/localrules must have an implementation.",
        )

    contract_driven = "ENVIRONMENT_LOCK" in source

    def selected_dir(variable: str, config_key: str, contract_section: str) -> Path | None:
        if contract_driven:
            prefix = _nested(environment, contract_section, "prefix")
            return (Path(str(prefix)).expanduser().resolve() / "bin") if prefix else None
        configured = config.get(config_key)
        default = _extract_default(source, variable, config_key)
        raw = configured if configured is not None else default
        return Path(str(raw)).expanduser().resolve() if raw else None

    mrtrix_dir = selected_dir("MRTRIX_BIN", "mrtrix_bin", "mrtrix3")
    tissue_dir = selected_dir("MRTRIX3TISSUE_BIN", "mrtrix3tissue_bin", "mrtrix3tissue")
    fsl_dir = selected_dir("FSL_BIN", "fsl_bin", "fsl")
    ants_dir = selected_dir("ANTS_BIN", "ants_bin", "ants")
    c3d_default = _extract_default(source, "C3D_AFFINE_TOOL", "c3d_affine_tool")
    contract_c3d = _nested(environment, "convert3d", "binaries", "c3d_affine_tool", "path")
    c3d_raw = contract_c3d if contract_driven else config.get("c3d_affine_tool", c3d_default)
    c3d_path = Path(str(c3d_raw)).expanduser().resolve() if c3d_raw else None

    locked_dirs = {
        "mrtrix": Path(str(_nested(environment, "mrtrix3", "prefix", default=""))) / "bin",
        "tissue": Path(str(_nested(environment, "mrtrix3tissue", "prefix", default=""))) / "bin",
        "fsl": Path(str(_nested(environment, "fsl", "prefix", default=""))) / "bin",
        "ants": Path(str(_nested(environment, "ants", "prefix", default=""))) / "bin",
    }
    selected_dirs = {
        "mrtrix": mrtrix_dir,
        "tissue": tissue_dir,
        "fsl": fsl_dir,
        "ants": ants_dir,
    }
    for name in ("mrtrix", "tissue", "fsl", "ants"):
        selected = selected_dirs[name]
        expected = locked_dirs[name].resolve()
        _check(
            checks,
            f"workflow.locked_path_{name}",
            selected == expected,
            "The workflow's executable directory must equal the frozen directory, not another installation or wrapper tree.",
            expected=str(expected),
            observed=str(selected) if selected else None,
        )

    declared_normal_path = _nested(environment, "execution_policy", "normal_tool_path_order")
    if contract_driven and isinstance(declared_normal_path, list):
        path_parts = [str(Path(str(path)).resolve()) for path in declared_normal_path]
    else:
        path_parts = [str(path) for path in (tissue_dir, mrtrix_dir, fsl_dir, ants_dir) if path]
    emulated_path = os.pathsep.join(path_parts + ["/usr/bin", "/bin"])
    _check(
        checks,
        "workflow.normal_path_excludes_tissue",
        not tissue_dir or str(tissue_dir) not in path_parts,
        "The global PATH must not place the legacy MRtrix3Tissue fork ahead of locked MRtrix3; SS3T gets a separate scoped PATH.",
        expected="MRtrix3Tissue absent from normal PATH",
        observed=path_parts,
    )
    promised_main = _nested(environment, "mrtrix3", "binaries", default={})
    for command in ("dwifslpreproc", "tckgen", "tcksift2", "tck2connectome"):
        expected = _nested(promised_main, command, "path")
        resolved = shutil.which(command, path=emulated_path)
        _check(
            checks,
            f"workflow.resolution_{command}",
            bool(expected and resolved and Path(resolved).resolve() == Path(str(expected)).resolve()),
            "PATH resolution must select the exact MRtrix3 executable whose hash is frozen.",
            expected=expected,
            observed=resolved,
        )

    expected_ss3t_python = _nested(environment, "execution_policy", "ss3t_invocation", "python")
    ss3t_path_parts = _nested(environment, "execution_policy", "ss3t_invocation", "path_prefix")
    if not isinstance(ss3t_path_parts, list):
        ss3t_path_parts = path_parts
    ss3t_path = os.pathsep.join([str(Path(str(path)).resolve()) for path in ss3t_path_parts] + ["/usr/bin", "/bin"])
    resolved_python = shutil.which("python", path=ss3t_path)
    explicit_ss3t_runtime = "{SS3T_PYTHON:q}" in rules_source and "{SS3T_SCRIPT:q}" in rules_source
    _check(
        checks,
        "workflow.ss3t_python_resolution",
        bool(
            explicit_ss3t_runtime
            or (
                expected_ss3t_python
                and resolved_python
                and Path(resolved_python).resolve() == Path(str(expected_ss3t_python)).resolve()
            )
        ),
        "The SS3T root process must use the frozen Python runtime; explicit invocation overrides its env-python shebang safely.",
        expected=expected_ss3t_python,
        observed=resolved_python,
    )
    _check(
        checks,
        "workflow.ss3t_python_explicit",
        explicit_ss3t_runtime,
        "The SS3T rule should invoke the frozen Python interpreter explicitly rather than inherit an ambient PATH.",
    )

    eddy_path = _nested(config, "dwi_preprocessing", "eddy", "executable")
    _check(
        checks,
        "workflow.eddy_executor_consumed",
        bool(
            eddy_path
            and "EDDY_CPU" in source
            and "{MRTRIX_BIN:q}/dwifslpreproc" in rules_source
            and "fsl_cpu_path" in rules_source
        ),
        "The configured CPU eddy executable must be consumed by the DWI preprocessing rule.",
        expected=eddy_path,
    )
    cuda_candidates = sorted(fsl_dir.glob("eddy_cuda*")) if fsl_dir and fsl_dir.is_dir() else []
    cpu_shim_enforced = (
        "candidate.name.startswith(\"eddy\") and candidate.name != \"eddy_cpu\"" in rules_source
        and "fsl_cpu_path" in rules_source
        and "any(path.name.startswith(\"eddy_cuda\")" in rules_source
    )
    _check(
        checks,
        "workflow.eddy_cuda_not_discoverable",
        not cuda_candidates or cpu_shim_enforced,
        "dwifslpreproc auto-discovers CUDA binaries; its scoped PATH must remove them and prove the CPU shim target.",
        expected="no CUDA entry in the eddy-scoped PATH",
        observed={"global_cuda_candidates": [str(path) for path in cuda_candidates], "cpu_shim_enforced": cpu_shim_enforced},
    )
    _check(
        checks,
        "workflow.no_eddy_fallback_enforced",
        "DWIFSLPREPROC_NO_CPU_FALLBACK" in rules_source,
        "The patched dwifslpreproc no-fallback guard must be exported when that script is used.",
    )
    locked_mrtrix_python = _nested(environment, "execution_policy", "mrtrix_script_python", "path")
    eddy_explicit_python = bool(
        (locked_mrtrix_python and str(locked_mrtrix_python) in rules_source)
        or "{MRTRIX_SCRIPT_PYTHON:q}" in rules_source
    )
    eddy_shim_excludes_python = bool(
        re.search(r"candidate\.name\.startswith\([^\n]*python", rules_source)
        or re.search(r"candidate\.name\s+in\s+[^\n]*python", rules_source)
    )
    eddy_system_python_first = bool(
        re.search(r"export\s+PATH=[^\n]*(?:/usr/bin|MRTRIX_SCRIPT)[^\n]*\{input\.fsl_cpu_path:q\}", rules_source)
    )
    _check(
        checks,
        "workflow.eddy_mrtrix_python_locked",
        eddy_explicit_python or (eddy_shim_excludes_python and eddy_system_python_first),
        "The eddy-scoped FSL shim must not override the frozen Python used by the dwifslpreproc env-python shebang.",
        expected=locked_mrtrix_python,
        observed={
            "explicit_python": eddy_explicit_python,
            "shim_excludes_python": eddy_shim_excludes_python,
            "system_python_before_shim": eddy_system_python_first,
        },
    )

    _check(
        checks,
        "workflow.convert3d_locked_path",
        bool(
            c3d_path
            and _nested(environment, "convert3d", "binaries", "c3d_affine_tool", "path")
            and c3d_path.resolve()
            == Path(str(_nested(environment, "convert3d", "binaries", "c3d_affine_tool", "path"))).resolve()
        ),
        "c3d_affine_tool must resolve to the executable declared in the environment contract.",
        expected=_nested(environment, "convert3d", "binaries", "c3d_affine_tool", "path"),
        observed=str(c3d_path) if c3d_path else None,
    )
    _check(
        checks,
        "workflow.convert3d_exists",
        bool(c3d_path and c3d_path.is_file() and os.access(c3d_path, os.X_OK)),
        "The selected c3d_affine_tool must exist and be executable.",
        observed=str(c3d_path) if c3d_path else None,
    )

    requires_container = "environment_lock.container_digest is required before execution" in rules_source
    container_digest = _nested(config, "environment_lock", "container_digest", default="")
    valid_digest = bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(container_digest).lower()))
    _check(
        checks,
        "workflow.preflight_environment_model",
        not requires_container or valid_digest,
        "The preflight cannot require an undeclared container digest while the recipe uses a host-hash contract.",
        expected="valid digest or host-contract validation",
        observed=container_digest or "missing",
    )

    required_binary_coverage = {
        "mrtrix3": {
            "mrconvert", "mrinfo", "dwidenoise", "mrdegibbs", "dwifslpreproc",
            "dwibiascorrect", "dwiextract", "mrmath", "dwi2mask", "5ttgen",
            "5ttcheck", "5tt2gmwmi", "mrtransform", "mrgrid", "dwi2response",
            "responsemean", "mtnormalise", "dwi2tensor", "tensor2metric",
            "tckgen", "tcksift2", "tcksample", "tck2connectome",
        },
        "fsl": {"eddy_cpu", "fast", "epi_reg", "flirt", "bet", "convert_xfm", "slices"},
        "ants": {"N4BiasFieldCorrection", "antsRegistration", "antsApplyTransforms", "antsRegistrationSyN_sh"},
        "mrtrix3tissue": {
            "ss3t_csd_beta1", "mrinfo", "dwiextract", "mrconvert", "mrcalc",
            "mrcat", "dwi2fod", "shconv", "sh2amp",
        },
    }
    for section, required_names in required_binary_coverage.items():
        declared_names = set(_nested(environment, section, "binaries", default={}))
        missing = sorted(required_names - declared_names)
        complete_manifest = _nested(environment, section, "complete_install_manifest", "sha256")
        _check(
            checks,
            f"coverage.{section}_scientific_executables",
            not missing or bool(complete_manifest),
            "Every invoked scientific executable needs an exact hash, unless a complete immutable installation manifest is locked.",
            severity="warning",
            expected=sorted(required_names),
            observed={"declared": sorted(declared_names), "missing": missing},
        )
    return checks


def audit_environment_contract(
    environment_path: Path = DEFAULT_ENVIRONMENT,
    config_path: Path = DEFAULT_CONFIG,
    *,
    check_host: bool = True,
    check_workflow: bool = True,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        environment = _load_yaml(environment_path)
        config = _load_yaml(config_path)
    except Exception as exc:
        return {
            "status": "FAIL",
            "checks": [],
            "failures": [{"code": "yaml_parse", "detail": f"{type(exc).__name__}:{exc}"}],
            "warnings": [],
        }

    checks.extend(policy_consistency_checks(environment, config))

    mni_template_raw = _nested(config, "registration", "mni_to_t1", "template")
    mni_template_hash = _nested(config, "registration", "mni_to_t1", "template_sha256")
    mni_template = Path(str(mni_template_raw)).expanduser() if mni_template_raw else Path("")
    mni_passed, mni_observed = (
        validate_hashed_artifact(mni_template, str(mni_template_hash))
        if mni_template_raw
        else (False, None)
    )
    _check(
        checks,
        "artifact.config_mni_template.sha256",
        mni_passed,
        "The MNI registration template must exist and match its exact frozen SHA-256.",
        expected=mni_template_hash,
        observed=mni_observed or "missing",
    )

    for logical_name, record, executable in _hashed_artifacts(environment):
        raw_path = record.get("path")
        expected = str(record.get("sha256", "")).lower()
        path = Path(str(raw_path)).expanduser() if raw_path else Path("")
        passed, observed = validate_hashed_artifact(path, expected) if raw_path else (False, None)
        _check(
            checks,
            f"artifact.{logical_name}.sha256",
            passed,
            "Declared artifact must exist and match its exact SHA-256.",
            expected=expected,
            observed=observed or "missing",
        )
        if executable:
            _check(
                checks,
                f"artifact.{logical_name}.executable",
                path.is_file() and os.access(path, os.X_OK),
                "Declared binary must be executable.",
                observed=str(path),
            )

    if check_host:
        pretty_name = ""
        os_release = Path("/etc/os-release")
        if os_release.is_file():
            for line in os_release.read_text(encoding="utf-8").splitlines():
                if line.startswith("PRETTY_NAME="):
                    pretty_name = line.split("=", 1)[1].strip().strip('"')
                    break
        host = environment.get("host", {})
        host_observed = {
            "os": pretty_name,
            "architecture": platform.machine(),
            "kernel": platform.release(),
        }
        for key, observed in host_observed.items():
            _check(
                checks,
                f"host.{key}",
                isinstance(host, dict) and host.get(key) == observed,
                "Host identity must match the frozen observation.",
                expected=host.get(key) if isinstance(host, dict) else None,
                observed=observed,
            )
        rc, _ = _run(["nvidia-smi"], timeout=5)
        gpu_available = rc == 0
        _check(
            checks,
            "host.gpu_driver_available",
            isinstance(host, dict) and host.get("gpu_driver_available") is gpu_available,
            "GPU availability must match the frozen executor decision.",
            expected=host.get("gpu_driver_available") if isinstance(host, dict) else None,
            observed=gpu_available,
        )

        for section in ("mrtrix3", "mrtrix3tissue"):
            prefix = Path(str(_nested(environment, section, "prefix", default="")))
            expected_commit = _nested(environment, section, "git_commit")
            rc, commit_output = _run(["git", "-C", str(prefix), "rev-parse", "HEAD"])
            observed_commit = commit_output.splitlines()[0] if rc == 0 and commit_output else None
            _check(
                checks,
                f"git.{section}.commit",
                rc == 0 and observed_commit == expected_commit,
                "Git commit must match the frozen source revision.",
                expected=expected_commit,
                observed=observed_commit,
            )
            rc, status_output = _run(["git", "-C", str(prefix), "status", "--porcelain"])
            observed_clean = rc == 0 and not status_output.strip()
            _check(
                checks,
                f"git.{section}.clean_state",
                observed_clean is _nested(environment, section, "worktree_clean"),
                "Git worktree cleanliness must match the declared state.",
                expected=_nested(environment, section, "worktree_clean"),
                observed=observed_clean,
            )
            expected_diff = _nested(environment, section, "git_diff_sha256")
            if expected_diff:
                try:
                    diff_proc = subprocess.run(
                        ["git", "-C", str(prefix), "diff"],
                        check=False,
                        capture_output=True,
                        timeout=20,
                    )
                    observed_diff = hashlib.sha256(diff_proc.stdout).hexdigest() if diff_proc.returncode == 0 else None
                except Exception:  # pragma: no cover - host diagnostic
                    observed_diff = None
                _check(
                    checks,
                    f"git.{section}.diff_sha256",
                    observed_diff == expected_diff,
                    "A declared dirty worktree must retain the exact frozen diff.",
                    expected=expected_diff,
                    observed=observed_diff,
                )

        python_records = (
            ("workflow_python", _nested(environment, "workflow", "python")),
            ("ss3t_python", _nested(environment, "mrtrix3tissue", "python_runtime")),
            ("mrtrix_script_python", _nested(environment, "execution_policy", "mrtrix_script_python")),
        )
        shared_mrtrix_python = _nested(environment, "execution_policy", "mrtrix_script_python", default={})
        for name, record in python_records:
            if isinstance(record, dict) and record.get("path"):
                rc, output = _run([str(record["path"]), "--version"])
                _check(
                    checks,
                    f"versions.{name}",
                    rc == 0 and str(record.get("version")) in output,
                    "Python runtime version must match the declaration.",
                    expected=record.get("version"),
                    observed=output.splitlines()[0] if output else None,
                )
                shared_hash = None
                if (
                    name == "ss3t_python"
                    and isinstance(shared_mrtrix_python, dict)
                    and Path(str(record["path"])).resolve()
                    == Path(str(shared_mrtrix_python.get("path", ""))).resolve()
                ):
                    shared_hash = shared_mrtrix_python.get("sha256")
                _check(
                    checks,
                    f"coverage.{name}_sha256",
                    bool(record.get("sha256") or shared_hash),
                    "A path and version do not immutably identify a Python build; record its executable SHA-256.",
                    severity="warning",
                )

        normal_path = _nested(environment, "execution_policy", "normal_tool_path_order", default=[])
        resolved_mrtrix_python = shutil.which(
            "python",
            path=os.pathsep.join([str(path) for path in normal_path] + ["/usr/bin", "/bin"]),
        ) if isinstance(normal_path, list) else None
        declared_mrtrix_python = _nested(environment, "execution_policy", "mrtrix_script_python", default={})
        declared_mrtrix_python_path = declared_mrtrix_python.get("path") if isinstance(declared_mrtrix_python, dict) else None
        declared_mrtrix_python_hash = declared_mrtrix_python.get("sha256") if isinstance(declared_mrtrix_python, dict) else None
        _check(
            checks,
            "coverage.mrtrix_script_python",
            bool(
                resolved_mrtrix_python
                and declared_mrtrix_python_path
                and Path(resolved_mrtrix_python).resolve() == Path(str(declared_mrtrix_python_path)).resolve()
                and declared_mrtrix_python_hash
            ),
            "MRtrix command scripts use env-python; record the exact Python executable selected by the locked normal PATH.",
            severity="warning",
            expected=resolved_mrtrix_python,
            observed=declared_mrtrix_python or "undeclared",
        )

        requirements = _nested(environment, "workflow", "requirements_lock", "path")
        wheel_manifest = _nested(environment, "workflow", "wheel_artifact_manifest", "path")
        if requirements and Path(str(requirements)).is_file():
            lock_text = Path(str(requirements)).read_text(encoding="utf-8")
            locked_packages = {}
            for line in lock_text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "==" not in stripped:
                    continue
                name, version = stripped.split("==", 1)
                canonical = re.sub(r"[-_.]+", "-", name).lower()
                locked_packages[canonical] = version.strip()
            artifact_packages = {}
            artifact_rows_valid = False
            if wheel_manifest and Path(str(wheel_manifest)).is_file():
                try:
                    with Path(str(wheel_manifest)).open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle, delimiter="\t"))
                    artifact_packages = {
                        re.sub(r"[-_.]+", "-", row["package"]).lower(): row["version"]
                        for row in rows
                    }
                    artifact_rows_valid = bool(rows) and all(
                        str(row.get("wheel_filename", "")).endswith(".whl")
                        and bool(SHA256_RE.fullmatch(str(row.get("sha256", "")).lower()))
                        for row in rows
                    )
                except Exception:
                    artifact_packages = {}
                    artifact_rows_valid = False
            _check(
                checks,
                "coverage.python_artifact_hashes",
                artifact_rows_valid and artifact_packages == locked_packages,
                "Every pinned Python package must have one platform-specific wheel filename and SHA-256 in the locked artifact manifest.",
                severity="warning",
                expected=len(locked_packages),
                observed=len(artifact_packages),
            )

    if check_workflow:
        checks.extend(workflow_integration_checks(environment, config))

    failures = [row for row in checks if not row["passed"] and row["severity"] == "error"]
    warnings = [row for row in checks if not row["passed"] and row["severity"] == "warning"]
    status = "FAIL" if failures else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "status": status,
        "environment_contract": str(environment_path.resolve()),
        "environment_contract_sha256": sha256_file(environment_path),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "summary": {
            "checks": len(checks),
            "passed": sum(bool(row["passed"]) for row in checks),
            "failures": len(failures),
            "warnings": len(warnings),
        },
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-host", action="store_true")
    parser.add_argument("--skip-workflow", action="store_true")
    args = parser.parse_args()

    report = audit_environment_contract(
        args.environment,
        args.config,
        check_host=not args.skip_host,
        check_workflow=not args.skip_workflow,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
