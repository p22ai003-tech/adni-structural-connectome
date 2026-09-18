#!/usr/bin/env python3
"""Resolve all three v2.1 hand-off modes against 530 mixed source fixtures.

The fixtures are tiny schema artifacts, never images. Snakemake is always
invoked with ``--dry-run``. Phase A proves every authorized unit reaches the
response-calibration DAG; phase B proves a synthetic frozen calibration can be
consumed without any live all-unit response dependency.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml


WORKFLOW_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKFLOW_DIR.parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "connectome_v2.yaml"
ENVIRONMENT_PATH = WORKFLOW_DIR / "environment_contract.yaml"
SCHEMA_PATH = WORKFLOW_DIR / "schemas" / "acquisition_manifest_v2.schema.json"
sys.path.insert(0, str(PROJECT_ROOT / "scforge"))

from scforge.input_contract import sha256_file  # noqa: E402
from scforge.response_calibration import (  # noqa: E402
    assess_response_calibration_candidate,
    build_pass_response_calibration_outcome,
    file_record,
    freeze_response_calibration_from_phase_a,
    validate_response_calibration_outcome,
)


def _write_fixture(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _manifest_rows(root: Path, fieldnames: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(1, 531):
        subject = f"DRYRUN_{index:04d}"
        dti_id = str(index)
        t1_id = str(10000 + index)
        dwi_dir = root / "sources" / subject / "dwi_dicom"
        dwi_member = dwi_dir / "0001.dcm"
        _write_fixture(dwi_member, f"DICOM schema fixture {subject}\n".encode())
        if (index <= 12 and index % 2 == 1) or 12 < index <= 305:
            t1_kind = "dicom_series"
            t1_source = root / "sources" / subject / "t1_dicom"
            t1_member = t1_source / "0001.dcm"
            _write_fixture(t1_member, f"T1 DICOM schema fixture {subject}\n".encode())
            t1_uid = f"1.2.840.10008.2.{index}"
        else:
            t1_kind = "nifti_single"
            t1_source = root / "sources" / subject / "t1.nii"
            _write_fixture(t1_source, f"T1 NIfTI schema fixture {subject}\n".encode())
            t1_member = t1_source
            t1_uid = ""
        # One missing-metadata row proves scheduling retains the unit. The
        # real normalization rule would emit a structured FAIL marker.
        missing_metadata = index == 530
        diagnosis = "SMC" if index > 515 else ("AD" if index % 7 == 0 else ("MCI" if index % 3 == 0 else "CN"))
        row = {
            "manifest_schema_version": "2.0.0",
            "data_scope_decision": "SL-D01_local_only",
            "processing_authorized": "true",
            "analysis_role": "smc_retained_separately" if diagnosis == "SMC" else "primary_cn_mci_ad",
            "subject_id": subject,
            "diagnosis_at_dti": diagnosis,
            "dti_image_id": dti_id,
            "dti_study_date": "2020-01-01",
            "dti_source_kind": "dicom_series",
            "dti_source_id": f"ADNI-I{dti_id}",
            "dti_source_path": str(dwi_dir),
            "dti_dicom_series_uid": f"1.2.840.10008.1.{index}",
            "dti_raw_bundle_sha256": hashlib.sha256(dwi_member.read_bytes()).hexdigest(),
            "dti_raw_file_count": "1",
            "dti_raw_total_bytes": str(dwi_member.stat().st_size),
            "dwi_nifti_path": "",
            "dwi_bvec_path": "",
            "dwi_bval_path": "",
            "dwi_json_path": "",
            "dwi_nifti_sha256": "",
            "dwi_bvec_sha256": "",
            "dwi_bval_sha256": "",
            "dwi_json_sha256": "",
            "t1_image_id": t1_id,
            "t1_study_date": "2010-01-01" if index % 2 else "2020-01-01",
            "t1_source_kind": t1_kind,
            "t1_source_id": f"ADNI-I{t1_id}",
            "t1_source_path": str(t1_source),
            "t1_dicom_series_uid": t1_uid,
            "t1_raw_bundle_sha256": hashlib.sha256(t1_member.read_bytes()).hexdigest(),
            "t1_raw_file_count": "1",
            "t1_raw_total_bytes": str(t1_member.stat().st_size),
            "abs_pair_gap_days": "3653" if index % 2 else "0",
            "timing_stratum": "gt_180_days" if index % 2 else "le_90_days",
            "phase": "DRYRUN",
            "site": f"{index % 20:03d}",
            "manufacturer": (
                "SIEMENS" if index % 2 == 1 else "GE MEDICAL SYSTEMS"
            ),
            "scanner_model": "DRYRUN",
            "field_strength_t": "3.0",
            "protocol": "single-shell-b1000",
            "phase_encoding_direction": "" if missing_metadata else "j-",
            "phase_encoding_source": "" if missing_metadata else "synthetic_schema_fixture",
            "total_readout_time": "" if missing_metadata else "0.05",
            "total_readout_time_source": "" if missing_metadata else "synthetic_schema_fixture",
            "source_lock_status": "COMPLETE_SHA256",
            "pair_content_bundle_sha256": hashlib.sha256(f"pair-{index}".encode()).hexdigest(),
            "normalization_readiness": "FAIL_MISSING_PHASE_ENCODING_AND_TOTAL_READOUT_TIME" if missing_metadata else "READY",
        }
        rows.append({field: row[field] for field in fieldnames})
    return rows


def _response_outcome(
    root: Path,
    unit: str,
    value: float,
    *,
    recipe_id: str,
    run_context: Path,
    attempt_context: Path,
) -> dict:
    responses: dict[str, Path] = {}
    for tissue in ("wm", "gm", "csf"):
        path = root / "calibration_source" / f"{unit}_{tissue}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = 4 if tissue == "wm" else 1
        array = np.full((2, columns), value, dtype=float)
        np.savetxt(path, array, fmt="%.12g")
        responses[tissue] = path
    response_voxels = root / "calibration_source" / f"{unit}_voxels.mif"
    _write_fixture(response_voxels, b"synthetic response-voxel schema fixture\n")
    response_qc = assess_response_calibration_candidate(
        responses,
        selected_voxel_counts={"wm": 100, "gm": 100, "csf": 100},
        expected_shell_rows=2,
        expected_coefficient_columns={"wm": 4, "gm": 1, "csf": 1},
    )
    shell_selection = root / "calibration_source" / f"{unit}_fod_shell_selection.json"
    _write_fixture(
        shell_selection,
        json.dumps(
            {
                "schema_version": "2.0.0",
                "status": "PASS",
                "unit": unit,
                "selected_shells_s_per_mm2": [0.0, 1000.0],
                "selected_shell_sizes": [5, 30],
                "bzero_threshold_s_per_mm2": 50.0,
                "selection_uses_diagnosis_labels": False,
            },
            sort_keys=True,
        )
        + "\n",
    )
    return build_pass_response_calibration_outcome(
        unit=unit,
        recipe_id=recipe_id,
        source_identity_hashes={
            "dti_raw_bundle_sha256": hashlib.sha256(f"{unit}-dti".encode()).hexdigest(),
            "t1_raw_bundle_sha256": hashlib.sha256(f"{unit}-t1".encode()).hexdigest(),
            "pair_content_bundle_sha256": hashlib.sha256(f"{unit}-pair".encode()).hexdigest(),
        },
        response_paths=responses,
        generated_utc="2020-01-01T00:00:00Z",
        phase_a_run_id="c" * 24,
        phase_a_attempt_id="synthetic-phase-a-attempt",
        run_context_path=run_context,
        attempt_context_path=attempt_context,
        response_qc=response_qc,
        response_voxels_path=response_voxels,
        fod_shell_selection_path=shell_selection,
    )


def _synthetic_frozen_calibration(
    root: Path, recipe_id: str, execution_binding: Mapping[str, object]
) -> dict:
    """Exercise the same complete phase-A handoff required for a real freeze."""

    publication = root / "synthetic_phase_a" / "publication"
    contract = root / "synthetic_phase_a" / "contract"
    attempts = root / "synthetic_phase_a" / "attempts"
    run_context = contract / "response_calibration_phase_a_run_context.json"
    attempt_context = attempts / "synthetic.start.json"
    _write_fixture(
        run_context,
        json.dumps(
            {
                "schema_version": "2.0.0",
                "status": "LOCKED",
                "run_id": "c" * 24,
                "recipe_id": recipe_id,
                "run_root": str((root / "synthetic_phase_a").resolve()),
                "execution_binding": execution_binding,
            },
            sort_keys=True,
        )
        + "\n",
    )
    _write_fixture(
        attempt_context,
        json.dumps(
            {
                "schema_version": "2.0.0",
                "status": "STARTED",
                "attempt_id": "synthetic-phase-a-attempt",
                "launcher_mode": "response-calibration-phase-a",
                "run_id": "c" * 24,
                "recipe_id": recipe_id,
                "run_context": file_record(run_context),
                "execution_binding": execution_binding,
            },
            sort_keys=True,
        )
        + "\n",
    )
    outcomes = [
        _response_outcome(
            root,
            unit,
            float(index),
            recipe_id=recipe_id,
            run_context=run_context,
            attempt_context=attempt_context,
        )
        for index, unit in enumerate(execution_binding["units"], start=1)
    ]
    outcome_records = {}
    outcome_paths = []
    for outcome in outcomes:
        path = publication / "outcomes" / f"{outcome['unit']}.json"
        _write_fixture(path, json.dumps(outcome, indent=2, sort_keys=True) + "\n")
        outcome_records[outcome["unit"]] = file_record(path)
        outcome_paths.append(str(path.resolve()))
    phase_a_manifest = publication / "response_calibration_phase_a_manifest.json"
    _write_fixture(
        phase_a_manifest,
        json.dumps(
            {
                "schema_version": "2.0.0",
                "record_type": "response_calibration_phase_a_manifest",
                "status": "COMPLETE",
                "run_outcome": "PASS",
                "generated_utc": "2020-01-01T00:00:00Z",
                "mode": "response-calibration-phase-a",
                "run_id": "c" * 24,
                "recipe_id": recipe_id,
                "execution_binding": execution_binding,
                "run_context": file_record(run_context),
                "attempt_start": file_record(attempt_context),
                "expected_units": [row["unit"] for row in outcomes],
                "summary": {
                    "expected": 12,
                    "terminal": 12,
                    "pass": 12,
                    "fail": 0,
                },
                "outcome_records": outcome_records,
                "calibration_freeze_input_outcomes": outcome_paths,
                "freeze_required_before_phase_b": True,
                "full_run_terminal_publication": False,
                "dummy_response_files_used": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    completion = publication / "response_calibration_phase_a_completion.json"
    _write_fixture(
        completion,
        json.dumps(
            {
                "schema_version": "2.0.0",
                "record_type": "response_calibration_phase_a_completion",
                "status": "PASS",
                "mode": "response-calibration-phase-a",
                "run_id": "c" * 24,
                "recipe_id": recipe_id,
                "execution_binding": execution_binding,
                "phase_a_manifest": file_record(phase_a_manifest),
                "freeze_required_before_phase_b": True,
                "full_run_terminal_publication": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return freeze_response_calibration_from_phase_a(
        completion,
        phase_a_manifest,
        root / "frozen_calibration",
        minimum_valid_subjects=12,
        generated_utc="2020-01-01T00:00:00Z",
        responsemean_executable="/home/ec2-user/mrtrix3/bin/responsemean",
        expected_responsemean_sha256=sha256_file(
            "/home/ec2-user/mrtrix3/bin/responsemean"
        ),
    )


def _run_dry(snakemake: Path, config_path: Path, target: str | None) -> dict[str, object]:
    command = [
        str(snakemake),
        "--snakefile",
        str(WORKFLOW_DIR / "Snakefile"),
        "--configfile",
        str(config_path),
        "--cores",
        "1",
        "--dry-run",
        "--rerun-triggers",
        "mtime",
        "--",
        target or "all",
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    output = completed.stdout + completed.stderr
    job_counts: dict[str, int] = {}
    declared_outputs = [
        value.strip()
        for line in output.splitlines()
        if line.strip().startswith("output: ")
        for value in line.strip().removeprefix("output: ").split(",")
        if value.strip() and value.strip() != "<TBD>"
    ]
    in_job_stats = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "Job stats:":
            in_job_stats = True
            continue
        if not in_job_stats:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_]+)\s+(\d+)", line.strip())
        if match:
            job_counts[match.group(1)] = int(match.group(2))
        elif job_counts and not line.strip():
            break
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "target": target or "all_phase_b",
        "returncode": completed.returncode,
        "command": command,
        "job_counts": job_counts,
        "declared_outputs": sorted(set(declared_outputs)),
        "output": output,
    }


def _materialize_dry_run_outputs(result: Mapping[str, object], run_root: Path) -> None:
    """Create schema-only placeholders strictly inside a temporary smoke root."""

    root = run_root.resolve()
    materialized: list[Path] = []
    directory_suffixes = (
        Path("contract/fsl_cpu_path"),
        Path("contract/source_runtime_inventory"),
        Path("01_dwi/eddy_qc"),
    )
    for raw_path in result.get("declared_outputs", []):
        path = Path(str(raw_path)).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"dry-run output escapes temporary smoke root: {path}") from exc
        if any(
            relative == suffix or str(relative).endswith(str(suffix))
            for suffix in directory_suffixes
        ):
            path.mkdir(parents=True, exist_ok=True)
            timestamp = path / ".snakemake_timestamp"
            timestamp.touch()
            materialized.extend((path, timestamp))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"schema-only staged dry-run placeholder\n")
            materialized.append(path)
    common_mtime_ns = time.time_ns()
    for path in materialized:
        os.utime(path, ns=(common_mtime_ns, common_mtime_ns))


def run_smoke(mode: str = "both") -> dict[str, object]:
    normative_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    environment = yaml.safe_load(ENVIRONMENT_PATH.read_text(encoding="utf-8"))
    snakemake = Path(environment["workflow"]["snakemake"]["path"])
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    fieldnames = list(schema["required"])

    with tempfile.TemporaryDirectory(prefix="connectome_v2_dryrun_") as directory:
        root = Path(directory)
        smoke_run_root = root / "scforge_v2_smoke"
        manifest = root / "manifest.csv"
        rows = _manifest_rows(root, fieldnames)
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        execution_manifest = root / "canary_execution_subset.csv"
        execution_rows = rows[:12]
        with execution_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(execution_rows)

        recipe_id = normative_config["contract"]["recipe_id"]
        execution_units = [
            f"DRYRUN_{index:04d}_I{index}" for index in range(1, 13)
        ]
        phase_b_units = execution_units[:9]
        subset_decision = root / "canary_execution_subset_decision.json"
        _write_fixture(
            subset_decision,
            json.dumps(
                {
                    "schema_version": "2.0.0",
                    "decision_type": "connectome_canary_execution_subset_approval",
                    "status": "APPROVED",
                    "approval_mode": "H04A_BOUNDED_CANARY",
                    "approved_by": "synthetic-dry-run",
                    "approved_utc": "2020-01-01T00:00:00Z",
                    "user_response": "synthetic dry-run only; no imaging authority",
                    "execution_scope": "canary",
                    "recipe_id": recipe_id,
                    "parent_acquisition_manifest_sha256": sha256_file(manifest),
                    "execution_subset_manifest_sha256": sha256_file(
                        execution_manifest
                    ),
                    "approved_unit_count": len(execution_units),
                    "approved_units": execution_units,
                    "diagnosis_labels_used": False,
                    "selection_locked": True,
                    "proposed_run_root": str(smoke_run_root.resolve()),
                    "authorized_modes": [
                        "response-calibration-phase-a",
                        "pre-tractography-canary",
                    ],
                    "authorized_through": "pre_tractography_review_bundle_only",
                    "maximum_cores": 4,
                    "minimum_valid_response_calibration_units": 12,
                    "minimum_valid_manufacturer_families": 2,
                    "minimum_valid_t1_source_classes": 2,
                    "required_valid_t1_source_classes": [
                        "dicom_series",
                        "nifti_single",
                    ],
                    "h04a_wall_clock_stop_hours": 72,
                    "h04a_storage_stop_gb": 150,
                    "tractography_authorized": False,
                    "matrix_generation_authorized": False,
                    "full_cohort_authorized": False,
                    "normative_config_sha256": sha256_file(CONFIG_PATH),
                    "workflow_source_manifest_sha256": sha256_file(
                        environment["workflow"]["source_manifest"]["path"]
                    ),
                    "environment_contract_sha256": sha256_file(
                        ENVIRONMENT_PATH
                    ),
                },
                sort_keys=True,
            )
            + "\n",
        )
        execution_binding = {
            "schema_version": "2.0.0",
            "binding_type": "connectome_execution_subset_binding",
            "execution_scope": "canary",
            "recipe_id": recipe_id,
            "parent_acquisition_manifest": file_record(manifest),
            "execution_subset_manifest": file_record(execution_manifest),
            "execution_subset_decision": file_record(subset_decision),
            "approved_unit_count": len(execution_units),
            "units": execution_units,
            "diagnosis_labels_used": False,
            "selection_locked": True,
            "h04a_authorization": {
                "approval_mode": "H04A_BOUNDED_CANARY",
                "approved_by": "synthetic-dry-run",
                "approved_utc": "2020-01-01T00:00:00Z",
                "user_response": "synthetic dry-run only; no imaging authority",
                "proposed_run_root": str(smoke_run_root.resolve()),
                "authorized_modes": [
                    "response-calibration-phase-a",
                    "pre-tractography-canary",
                ],
                "authorized_through": "pre_tractography_review_bundle_only",
                "maximum_cores": 4,
                "minimum_valid_response_calibration_units": 12,
                "minimum_valid_manufacturer_families": 2,
                "minimum_valid_t1_source_classes": 2,
                "required_valid_t1_source_classes": [
                    "dicom_series",
                    "nifti_single",
                ],
                "wall_clock_stop_hours": 72,
                "wall_clock_stop_seconds": 72 * 60 * 60,
                "storage_stop_gb": 150,
                "storage_stop_bytes": 150 * 1_000_000_000,
                "tractography_authorized": False,
                "matrix_generation_authorized": False,
                "full_cohort_authorized": False,
                "normative_config_sha256": sha256_file(CONFIG_PATH),
                "workflow_source_manifest_sha256": sha256_file(
                    environment["workflow"]["source_manifest"]["path"]
                ),
                "environment_contract_sha256": sha256_file(ENVIRONMENT_PATH),
            },
        }

        human_qc = root / "human_visual_qc.csv"
        with human_qc.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["unit", "status", "reviewer", "reviewed_utc"])
            for unit in phase_b_units:
                writer.writerow(
                    [unit, "PASS", "synthetic-dry-run", "2020-01-01T00:00:00Z"]
                )

        frozen = _synthetic_frozen_calibration(
            root, recipe_id, execution_binding
        )
        response_calibration_decision = (
            root / "synthetic_response_calibration_decision.json"
        )
        diversity = frozen["valid_pool_technical_diversity"]
        diversity_digest = frozen["valid_pool_technical_diversity_sha256"]
        _write_fixture(
            response_calibration_decision,
            json.dumps(
                {
                    "schema_version": "2.0.0",
                    "decision_type": "response_calibration_phase_b_approval",
                    "status": "APPROVED",
                    "recipe_id": recipe_id,
                    "phase_a_completion_sha256": frozen["phase_a_lineage"][
                        "completion"
                    ]["sha256"],
                    "response_calibration_manifest_sha256": frozen[
                        "manifest_sha256"
                    ],
                    "minimum_valid_subjects": 12,
                    "minimum_valid_manufacturer_families": 2,
                    "minimum_valid_t1_source_classes": 2,
                    "required_valid_t1_source_classes": [
                        "dicom_series",
                        "nifti_single",
                    ],
                    "valid_unit_count": diversity["valid_unit_count"],
                    "valid_manufacturer_counts": diversity[
                        "manufacturer_counts"
                    ],
                    "valid_manufacturer_family_counts": diversity[
                        "manufacturer_family_counts"
                    ],
                    "valid_t1_source_class_counts": diversity[
                        "t1_source_class_counts"
                    ],
                    "valid_pool_technical_diversity_sha256": diversity_digest,
                },
                sort_keys=True,
            )
            + "\n",
        )
        response_binding = {
            "schema_version": "2.0.0",
            "binding_type": "response_calibration_phase_b_binding",
            "recipe_id": recipe_id,
            "phase_a_completion": frozen["phase_a_lineage"]["completion"],
            "phase_a_manifest": frozen["phase_a_lineage"]["manifest"],
            "minimum_valid_subjects": 12,
            "valid_units": frozen["valid_units"],
            "valid_pool_technical_diversity": diversity,
            "valid_pool_technical_diversity_sha256": diversity_digest,
            "response_calibration_manifest": file_record(frozen["manifest_path"]),
            "response_calibration_decision": file_record(
                response_calibration_decision
            ),
            "pooled_responses": frozen["pooled_responses"],
            "diagnosis_labels_used": False,
            "phase_b_live_all_units_dependency": False,
            "dummy_response_files_used": False,
        }
        pre_manifest = root / "synthetic_pre_tractography_manifest.json"
        pre_completion = root / "synthetic_pre_tractography_completion.json"
        continuation_decision = root / "synthetic_continuation_decision.json"
        review_bundle_records = {}
        automated_qc_records = {}
        for unit in phase_b_units:
            evidence_dir = root / "synthetic_pre_tractography_evidence" / unit
            automated = evidence_dir / "automated_pre_tractography_qc.json"
            _write_fixture(
                automated,
                json.dumps(
                    {
                        "schema_version": "2.0.0",
                        "record_type": "automated_pre_tractography_qc",
                        "status": "PASS",
                        "unit": unit,
                        "diagnosis_labels_used": False,
                        "failures": [],
                    },
                    sort_keys=True,
                )
                + "\n",
            )
            automated_qc_records[unit] = file_record(automated)
            artifacts = {}
            for name, suffix in (
                ("b0_t1", "review_b0_vs_t1.png"),
                ("b0_5tt", "review_b0_vs_5tt.png"),
                ("b0_atlas", "review_b0_vs_aal3.png"),
                ("index", "visual_review_index.csv"),
            ):
                artifact = evidence_dir / suffix
                _write_fixture(artifact, f"synthetic {name} review evidence for {unit}\n")
                artifacts[name] = file_record(artifact)
            review_bundle_records[unit] = artifacts
        pre_summary = {
            "expected": len(execution_units),
            "terminal": len(execution_units),
            "ready_for_human_qc": len(phase_b_units),
            "fail": len(execution_units) - len(phase_b_units),
        }
        _write_fixture(
            pre_manifest,
            json.dumps(
                {
                    "schema_version": "2.0.0",
                    "record_type": "pre_tractography_canary_manifest",
                    "status": "COMPLETE",
                    "workflow_state": "AWAITING_HUMAN_QC",
                    "run_outcome": "PARTIAL",
                    "recipe_id": recipe_id,
                    "execution_binding": execution_binding,
                    "response_calibration_binding": response_binding,
                    "expected_units": execution_units,
                    "ready_units": phase_b_units,
                    "summary": pre_summary,
                    "review_bundle_records": review_bundle_records,
                    "automated_pre_tractography_qc_records": automated_qc_records,
                    "tractography_started": False,
                },
                sort_keys=True,
            )
            + "\n",
        )
        _write_fixture(
            pre_completion,
            json.dumps(
                {
                    "schema_version": "2.0.0",
                    "record_type": "pre_tractography_canary_completion",
                    "status": "AWAITING_HUMAN_QC",
                    "mode": "pre-tractography-canary",
                    "recipe_id": recipe_id,
                    "execution_binding": execution_binding,
                    "response_calibration_binding": response_binding,
                    "run_outcome": "PARTIAL",
                    "pre_tractography_manifest": file_record(pre_manifest),
                    "pre_tractography_summary": pre_summary,
                    "automated_pre_tractography_qc_records": automated_qc_records,
                    "full_run_terminal_publication": False,
                    "tractography_started": False,
                },
                sort_keys=True,
            )
            + "\n",
        )
        _write_fixture(
            continuation_decision,
            json.dumps(
                {
                    "schema_version": "2.0.0",
                    "decision_type": "connectome_canary_tractography_continuation_approval",
                    "status": "APPROVED",
                    "execution_scope": "canary",
                    "recipe_id": recipe_id,
                    "execution_subset_manifest_sha256": execution_binding[
                        "execution_subset_manifest"
                    ]["sha256"],
                    "response_calibration_manifest_sha256": response_binding[
                        "response_calibration_manifest"
                    ]["sha256"],
                    "pre_tractography_completion_sha256": sha256_file(
                        pre_completion
                    ),
                    "human_qc_manifest_sha256": sha256_file(human_qc),
                    "minimum_valid_manufacturer_families": 2,
                    "minimum_valid_t1_source_classes": 2,
                    "required_valid_t1_source_classes": [
                        "dicom_series",
                        "nifti_single",
                    ],
                    "valid_unit_count": diversity["valid_unit_count"],
                    "valid_manufacturer_counts": diversity[
                        "manufacturer_counts"
                    ],
                    "valid_manufacturer_family_counts": diversity[
                        "manufacturer_family_counts"
                    ],
                    "valid_t1_source_class_counts": diversity[
                        "t1_source_class_counts"
                    ],
                    "valid_pool_technical_diversity_sha256": diversity_digest,
                    "approved_unit_count": len(phase_b_units),
                    "approved_units": phase_b_units,
                    "diagnosis_labels_used": False,
                    "all_reviewed_units_pass": True,
                },
                sort_keys=True,
            )
            + "\n",
        )
        continuation_binding = {
            "schema_version": "2.0.0",
            "binding_type": "tractography_continuation_binding",
            "execution_scope": "canary",
            "recipe_id": recipe_id,
            "execution_subset_manifest": copy.deepcopy(
                execution_binding["execution_subset_manifest"]
            ),
            "pre_tractography_completion": file_record(pre_completion),
            "pre_tractography_manifest": file_record(pre_manifest),
            "human_qc_manifest": file_record(human_qc),
            "continuation_decision": file_record(continuation_decision),
            "valid_pool_technical_diversity": copy.deepcopy(diversity),
            "valid_pool_technical_diversity_sha256": diversity_digest,
            "approved_unit_count": len(phase_b_units),
            "approved_units": phase_b_units,
            "diagnosis_labels_used": False,
        }

        def mode_config(launcher_mode: str) -> Path:
            config = copy.deepcopy(normative_config)
            config.update(
                {
                    "manifest_path": str(manifest),
                    "execution_manifest_path": str(execution_manifest),
                    "run_root": str(smoke_run_root),
                    "human_qc_manifest": str(human_qc),
                    "launcher_mode": launcher_mode,
                    "execution_binding": copy.deepcopy(execution_binding),
                }
            )
            config["inputs"]["approved_pair_manifest"] = {
                **config["inputs"]["approved_pair_manifest"],
                "path": str(manifest),
                "sha256": sha256_file(manifest),
            }
            config["inputs"]["acquisition_schema"]["sha256"] = sha256_file(
                SCHEMA_PATH
            )
            if launcher_mode != "response-calibration-phase-a":
                config["response_calibration_binding"] = copy.deepcopy(
                    response_binding
                )
                calibration = config["fod"]["response_estimation"]["calibration"]
                calibration["minimum_valid_subjects"] = 12
                calibration["frozen_manifest"] = copy.deepcopy(
                    response_binding["response_calibration_manifest"]
                )
                calibration["pooled_responses"] = copy.deepcopy(
                    response_binding["pooled_responses"]
                )
            if launcher_mode == "phase-b":
                config["tractography_continuation_binding"] = copy.deepcopy(
                    continuation_binding
                )

            token = launcher_mode.replace("-", "_")
            config_path = smoke_run_root / "contract" / f"{token}.resolved.yaml"
            run_context = smoke_run_root / "contract" / f"{token}.run_context.json"
            attempt_context = smoke_run_root / "attempts" / f"{token}.start.json"
            config["run_context_path"] = str(run_context)
            config["attempt_context_path"] = str(attempt_context)
            config["resolved_run_config_path"] = str(config_path)
            context = {
                "schema_version": "1.0.0",
                "status": "LOCKED",
                "run_id": hashlib.sha256(launcher_mode.encode()).hexdigest()[:24],
                "lineage_id": "synthetic-dry-run-lineage",
                "launcher_mode": launcher_mode,
                "recipe_id": recipe_id,
                "run_root": str(smoke_run_root),
                "acquisition_manifest": file_record(manifest),
                "execution_binding": copy.deepcopy(execution_binding),
                "host": {"mode": "schema-only-dry-run"},
            }
            if launcher_mode != "response-calibration-phase-a":
                context["response_calibration_binding"] = copy.deepcopy(
                    response_binding
                )
            if launcher_mode == "phase-b":
                context["tractography_continuation_binding"] = copy.deepcopy(
                    continuation_binding
                )
                context["human_qc_manifest"] = file_record(human_qc)
            _write_fixture(run_context, json.dumps(context, sort_keys=True) + "\n")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            attempt = {
                "schema_version": "1.0.0",
                "status": "STARTED",
                "started_utc": "2020-01-01T00:00:00+00:00",
                "attempt_id": f"schema-only-{token}",
                "launcher_mode": launcher_mode,
                "run_id": context["run_id"],
                "recipe_id": recipe_id,
                "execution_binding": copy.deepcopy(execution_binding),
                "run_context": file_record(run_context),
                "resolved_run_config": file_record(config_path),
                "snakemake_invocation": [
                    "snakemake",
                    "--printshellcmds",
                    "--snakefile",
                    "Snakefile",
                    "--configfile",
                    str(config_path),
                ],
            }
            if launcher_mode != "response-calibration-phase-a":
                attempt["response_calibration_binding"] = copy.deepcopy(
                    response_binding
                )
            if launcher_mode == "phase-b":
                attempt["tractography_continuation_binding"] = copy.deepcopy(
                    continuation_binding
                )
            _write_fixture(attempt_context, json.dumps(attempt, sort_keys=True) + "\n")
            return config_path

        not_run = lambda target: {
            "status": "NOT_RUN",
            "target": target,
            "returncode": None,
            "command": [],
            "job_counts": {},
            "output": "",
        }
        if mode == "both":
            pre_contract_refresh = not_run("pre_tractography_canary_contract_refresh")
            phase_b_contract_refresh = not_run("phase_b_contract_refresh")
            phase_a_config = mode_config("response-calibration-phase-a")
            phase_a = _run_dry(
                snakemake, phase_a_config, "response_calibration_phase_a"
            )
            if phase_a["status"] == "PASS":
                _materialize_dry_run_outputs(phase_a, smoke_run_root)
            pre_config = mode_config("pre-tractography-canary")
            pre_contract_refresh = _run_dry(
                snakemake, pre_config, "execution_preflight"
            )
            if pre_contract_refresh["status"] == "PASS":
                _materialize_dry_run_outputs(
                    pre_contract_refresh, smoke_run_root
                )
            pre_tractography = _run_dry(
                snakemake, pre_config, "pre_tractography_canary"
            )
            if pre_tractography["status"] == "PASS":
                _materialize_dry_run_outputs(pre_tractography, smoke_run_root)
            phase_b_config = mode_config("phase-b")
            phase_b_contract_refresh = _run_dry(
                snakemake,
                phase_b_config,
                "execution_preflight",
            )
            if phase_b_contract_refresh["status"] == "PASS":
                _materialize_dry_run_outputs(
                    phase_b_contract_refresh, smoke_run_root
                )
            phase_b = _run_dry(
                snakemake,
                phase_b_config,
                "phase_b_subject_terminal_records",
            )
        else:
            pre_contract_refresh = not_run("pre_tractography_canary_contract_refresh")
            phase_b_contract_refresh = not_run("phase_b_contract_refresh")
            phase_a = (
                _run_dry(
                    snakemake,
                    mode_config("response-calibration-phase-a"),
                    "response_calibration_phase_a",
                )
                if mode == "phase-a"
                else not_run("response_calibration_phase_a")
            )
            pre_tractography = (
                _run_dry(
                    snakemake,
                    mode_config("pre-tractography-canary"),
                    "pre_tractography_canary",
                )
                if mode == "pre-tractography"
                else not_run("pre_tractography_canary")
            )
            phase_b = (
                _run_dry(
                    snakemake,
                    mode_config("phase-b"),
                    "phase_b_subject_terminal_records",
                )
                if mode == "phase-b"
                else not_run("phase_b_subject_terminal_records")
            )
        phase_a_dag_valid = phase_a["status"] == "NOT_RUN" or (
            phase_a["job_counts"].get("response_calibration_phase_a") == 1
            and phase_a["job_counts"].get("subject_response") == len(execution_rows)
            and "pooled_response" not in phase_a["job_counts"]
        )
        phase_b_dag_valid = phase_b["status"] == "NOT_RUN" or (
            phase_b["job_counts"].get("pooled_response", 0)
            == (0 if mode == "both" else 1)
            and phase_b["job_counts"].get("tractography_10m")
            == len(phase_b_units)
            and phase_b["job_counts"].get("matrix_qc") == len(phase_b_units)
            and phase_b["job_counts"].get("provenance_sidecar")
            == len(phase_b_units)
            and phase_b["job_counts"].get("phase_b_subject_terminal_records") == 1
            and "response_calibration_phase_a" not in phase_b["job_counts"]
            and "subject_response" not in phase_b["job_counts"]
            and "publish_manifest" not in phase_b["job_counts"]
        )
        pre_tractography_dag_valid = pre_tractography["status"] == "NOT_RUN" or (
            pre_tractography["job_counts"].get("pooled_response") == 1
            and pre_tractography["job_counts"].get("automated_pre_tractography_qc")
            == len(execution_units)
            and pre_tractography["job_counts"].get("visual_review_bundle")
            == len(execution_units)
            and pre_tractography["job_counts"].get("pre_tractography_canary") == 1
            and "subject_response" not in pre_tractography["job_counts"]
            and "tractography_preflight" not in pre_tractography["job_counts"]
            and "tractography_10m" not in pre_tractography["job_counts"]
            and "matrix_qc" not in pre_tractography["job_counts"]
        )
        phase_a_scientific_rules = {
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
        }
        pre_review_rules = {
            name
            for name in pre_tractography["job_counts"]
            if name not in {"total", "pre_tractography_canary"}
        }
        staged_no_reschedule = mode != "both" or (
            pre_contract_refresh["job_counts"]
            == {"freeze_manifest": 1, "execution_preflight": 1, "total": 2}
            and not phase_a_scientific_rules.intersection(
                pre_tractography["job_counts"]
            )
            and phase_b_contract_refresh["job_counts"]
            == {"freeze_manifest": 1, "execution_preflight": 1, "total": 2}
            and not pre_review_rules.intersection(phase_b["job_counts"])
        )
        selected = {
            "both": [phase_a, pre_tractography, phase_b],
            "phase-a": [phase_a],
            "pre-tractography": [pre_tractography],
            "phase-b": [phase_b],
        }[mode]
        status = "PASS" if (
            all(phase["status"] == "PASS" for phase in selected)
            and all(
                refresh["status"] in {"PASS", "NOT_RUN"}
                for refresh in (pre_contract_refresh, phase_b_contract_refresh)
            )
            and phase_a_dag_valid
            and pre_tractography_dag_valid
            and phase_b_dag_valid
            and staged_no_reschedule
        ) else "FAIL"
        return {
            "status": status,
            "recipe_id": recipe_id,
            "dry_run": True,
            "fixture_row_count": len(rows),
            "execution_subset_row_count": len(execution_rows),
            "fixture_t1_dicom_count": sum(row["t1_source_kind"] == "dicom_series" for row in rows),
            "fixture_t1_nifti_count": sum(row["t1_source_kind"] == "nifti_single" for row in rows),
            "fixture_missing_eddy_metadata_count": sum(row["normalization_readiness"] != "READY" for row in rows),
            "synthetic_calibration_only": True,
            "phase_a_dag_isolated": phase_a_dag_valid,
            "pre_tractography_dag_stops_before_human_gate": pre_tractography_dag_valid,
            "phase_b_has_no_live_response_dependency": phase_b_dag_valid,
            "staged_no_completed_scientific_or_review_reschedule": staged_no_reschedule,
            "phase_a": phase_a,
            "pre_contract_refresh": pre_contract_refresh,
            "pre_tractography": pre_tractography,
            "phase_b_contract_refresh": phase_b_contract_refresh,
            "phase_b": phase_b,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON evidence path")
    parser.add_argument(
        "--mode",
        choices=("both", "phase-a", "pre-tractography", "phase-b"),
        default="both",
    )
    args = parser.parse_args()
    result = run_smoke(args.mode)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".partial")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.output.exists():
            if args.output.read_bytes() != temporary.read_bytes():
                raise FileExistsError(f"Refusing to replace non-identical dry-run evidence: {args.output}")
            temporary.unlink()
        else:
            temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "recipe_id": result["recipe_id"],
                "dry_run": result["dry_run"],
                "fixture_row_count": result["fixture_row_count"],
                "phase_a": result["phase_a"]["status"],
                "pre_tractography": result["pre_tractography"]["status"],
                "phase_b": result["phase_b"]["status"],
            },
            indent=2,
        )
    )
    if result["status"] != "PASS":
        for phase in ("phase_a", "pre_tractography", "phase_b"):
            if result[phase]["status"] != "PASS":
                if result[phase]["status"] == "NOT_RUN":
                    continue
                print(f"--- {phase} ---\n{result[phase]['output']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
