#!/home/ec2-user/fsl/bin/python
"""Supervise selected-count HCP379 tractography through final release.

The default mode writes a non-imaging preflight plan.  Execution is accepted
only with a genuine Recovery4 human-review CSV.  It then waits for the
Recovery4 Phase-B selection and lane-specific pretract summaries, runs at
most three one-subject tractography workers concurrently, compacts validated
tractograms, evaluates archive route concordance, assembles the corrected
archive sensitivity candidate and finally invokes the independent 530-case
release assembler, independent integration verifier, and immutable
scan-level analysis-handoff builder.

Existing valid terminal manifests are reused.  Existing invalid terminal
manifests are never overwritten.  Independent tasks continue after a peer
failure, while dependent tasks remain fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
PARALLEL_ROOT = HCP_ROOT / "parallel_recovery4"
PLAN = PARALLEL_ROOT / "parallel_tractography_release_plan.json"
STATE = PARALLEL_ROOT / "parallel_tractography_release_state.json"
ATTEMPT_ROOT = PARALLEL_ROOT / "tractography_release_attempts"
PRETRACT_ATTEMPT_ROOT = PARALLEL_ROOT / "attempts"
HUMAN_QC = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
PHASE_VALIDATION = Path(
    "/data/derivatives/scforge_v2/"
    "h04a_r1_recovery_20260719_retry4/publication/"
    "hcp379_recipe_stability_validation.json"
)
FINAL_SOURCE = EXP / "scripts/hcp/build_hcp379_final_release_v2.py"
ARCHIVE_CONCORDANCE_SOURCE = (
    EXP
    / "research_audit/"
    "validate_hcp379_archive_route_concordance_recovery4_v3.py"
)
ARCHIVE_86_SOURCE = (
    EXP
    / "scripts/hcp/"
    "assemble_hcp379_corrected_archive_86_recovery4_v3.py"
)
INTEGRATION_VERIFY_SOURCE = (
    EXP
    / "research_audit/"
    "verify_hcp379_integration_release_v1.py"
)
ANALYSIS_HANDOFF_SOURCE = (
    EXP
    / "scripts/hcp/"
    "prepare_hcp379_verified_analysis_handoff_v1.py"
)
ALLOWED_COUNTS = (3_000_000, 5_000_000, 10_000_000)
MINIMUM_DENSITY = 0.60
MATRIX_ARTIFACT_KEYS = {
    "matrix_count",
    "matrix_fd_sum",
    "matrix_count_invnodevol",
    "matrix_len_mean",
    "matrix_invlen_mean",
    "matrix_fa_mean",
    "matrix_md_mean",
    "matrix_rd_mean",
    "matrix_ad_mean",
}
MINIMUM_START_FREE_BYTES = 700_000_000_000
EMERGENCY_FREE_BYTES = 500_000_000_000
DEFAULT_MAX_ACTIVE = 3
DEFAULT_THREADS = 16
DEFAULT_UPSTREAM_TIMEOUT_HOURS = 336.0
ARCHIVE_RETAIN_DECISION = (
    "RETAIN_ARCHIVE_D0_56_AND_USE_CORRECTED_LOW_30_"
    "WITH_ROUTE_SENSITIVITY"
)
ARCHIVE_RERUN_DECISION = (
    "RERUN_ARCHIVE_86_WITH_LOCKED_CORRECTED_ACT"
)


@dataclass(frozen=True)
class UpstreamGate:
    name: str
    path: Path
    expected_n: int


@dataclass(frozen=True)
class Task:
    name: str
    command: tuple[str, ...]
    dependencies: tuple[str, ...]
    upstream_gates: tuple[str, ...]
    output: Path
    record_type: str
    expected_n: int
    compute_task: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_human_qc(path: Path) -> dict[str, Any]:
    module = load_module(FINAL_SOURCE, "hcp379_final_human_qc_contract")
    module.HUMAN_CANARY_QC = path
    return module.validate_human_canary_qc()


def phase_status() -> dict[str, Any]:
    if not PHASE_VALIDATION.is_file():
        return {
            "ready": False,
            "reason": "phase_b_validation_missing",
            "path": str(PHASE_VALIDATION),
        }
    try:
        value = load_json(PHASE_VALIDATION)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ready": False,
            "reason": f"phase_b_validation_unreadable:{type(exc).__name__}",
            "path": str(PHASE_VALIDATION),
        }
    selected = value.get(
        "smallest_density_qualified_scale_up_streamline_count"
    )
    ready = bool(
        value.get("record_type")
        == "diagnosis_blind_hcp379_phase_b_recipe_stability_validation"
        and value.get("status") == "PASS"
        and value.get("diagnosis_labels_used") is False
        and value.get("unit_count") == 15
        and selected in ALLOWED_COUNTS
        and isinstance(value.get("manifest"), Mapping)
    )
    return {
        "ready": ready,
        "reason": "PASS" if ready else "phase_b_validation_not_pass",
        "path": str(PHASE_VALIDATION),
        "selected_streamline_count": selected,
        "record": file_record(PHASE_VALIDATION),
    }


def upstream_gates() -> dict[str, UpstreamGate]:
    return {
        "core_pretract_216": UpstreamGate(
            "core_pretract_216",
            HCP_ROOT
            / "corrected_scaleup_recovery4/manifests/"
            "pretract_recovery4_summary.json",
            216,
        ),
        "legacy_pretract_212": UpstreamGate(
            "legacy_pretract_212",
            HCP_ROOT
            / "corrected_legacy_tensor_recovery4/manifests/"
            "pretract_recovery4_summary.json",
            212,
        ),
        "archive_calibration_pretract_6": UpstreamGate(
            "archive_calibration_pretract_6",
            HCP_ROOT
            / "archive_corrected_calibration_recovery4/manifests/"
            "pretract_recovery4_summary.json",
            6,
        ),
        "archive_low_pretract_28": UpstreamGate(
            "archive_low_pretract_28",
            HCP_ROOT
            / "corrected_archive_low_recovery4/manifests/"
            "pretract_recovery4_summary.json",
            28,
        ),
        "archive_D0_pretract_52": UpstreamGate(
            "archive_D0_pretract_52",
            HCP_ROOT
            / "corrected_archive_D0_recovery4/manifests/"
            "pretract_recovery4_summary.json",
            52,
        ),
        "freesurfer_pretract_1": UpstreamGate(
            "freesurfer_pretract_1",
            HCP_ROOT
            / "corrected_freesurfer_recovery4/manifests/"
            "pretract_recovery4_summary.json",
            1,
        ),
    }


def upstream_status(gate: UpstreamGate) -> dict[str, Any]:
    if not gate.path.is_file():
        return {
            "ready": False,
            "reason": "missing",
            "path": str(gate.path),
            "expected_n": gate.expected_n,
        }
    try:
        value = load_json(gate.path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ready": False,
            "reason": f"unreadable:{type(exc).__name__}",
            "path": str(gate.path),
            "expected_n": gate.expected_n,
        }
    ready = bool(
        value.get("record_type")
        == "diagnosis_blind_hcp379_scaleup_pretract_recovery4_summary"
        and value.get("status") == "PASS"
        and value.get("diagnosis_labels_used") is False
        and value.get("non_overwriting") is True
        and value.get("target_n") == gate.expected_n
        and value.get("states_present_n") == gate.expected_n
        and value.get("status_counts")
        == {"PASS_PRETRACT_RECOVERY4": gate.expected_n}
    )
    return {
        "ready": ready,
        "reason": "PASS" if ready else "summary_not_pass",
        "path": str(gate.path),
        "expected_n": gate.expected_n,
        "record": file_record(gate.path),
    }


def command(
    source: Path,
    *arguments: str,
) -> tuple[str, ...]:
    return (str(PYTHON), str(source), *arguments)


def tasks() -> dict[str, Task]:
    core_source = (
        EXP / "scripts/hcp/hcp379_scaleup_tractography_recovery4_v3.py"
    )
    legacy_source = (
        EXP
        / "scripts/hcp/"
        "run_hcp379_legacy_density_tractography_recovery4_v3.py"
    )
    archive_primary_source = (
        EXP
        / "scripts/hcp/"
        "run_hcp379_archive_calibration_tractography_recovery4_v3.py"
    )
    archive_replicate_source = (
        EXP
        / "scripts/hcp/"
        "run_hcp379_archive_calibration_replicate_recovery4_v3.py"
    )
    archive_low_source = (
        EXP
        / "scripts/hcp/"
        "run_hcp379_archive_low_tractography_recovery4_v3.py"
    )
    archive_D0_source = (
        EXP
        / "scripts/hcp/"
        "run_hcp379_archive_D0_tractography_recovery4_v3.py"
    )
    freesurfer_source = (
        EXP
        / "scripts/hcp/"
        "run_hcp379_freesurfer_tractography_recovery4_v3.py"
    )
    shared_compute = (
        "--execute",
        "--workers",
        "1",
        "--nthreads",
        str(DEFAULT_THREADS),
        "--compact-tractograms-after-pass",
    )
    root = HCP_ROOT
    return {
        "core_227": Task(
            "core_227",
            command(core_source, *shared_compute),
            (),
            ("core_pretract_216",),
            root
            / "corrected_scaleup_recovery4/manifests/"
            "corrected_227_release_manifest.json",
            "diagnosis_blind_hcp379_corrected_227_release_manifest",
            227,
            True,
        ),
        "legacy_tensor_216": Task(
            "legacy_tensor_216",
            command(legacy_source, *shared_compute),
            (),
            ("legacy_pretract_212",),
            root
            / "corrected_legacy_tensor_recovery4/manifests/"
            "corrected_legacy_tensor_216_release_manifest.json",
            (
                "diagnosis_blind_hcp379_corrected_legacy_tensor_216_"
                "release_manifest"
            ),
            216,
            True,
        ),
        "archive_primary_6": Task(
            "archive_primary_6",
            command(archive_primary_source, *shared_compute),
            (),
            ("archive_calibration_pretract_6",),
            root
            / "archive_corrected_calibration_recovery4/manifests/"
            "archive_corrected_calibration_recovery4_manifest.json",
            (
                "diagnosis_blind_hcp379_archive_corrected_act_"
                "calibration_recovery4_manifest"
            ),
            6,
            True,
        ),
        "freesurfer_1": Task(
            "freesurfer_1",
            command(freesurfer_source, *shared_compute),
            (),
            ("freesurfer_pretract_1",),
            root
            / "corrected_freesurfer_recovery4/manifests/"
            "corrected_freesurfer_1_release_manifest.json",
            (
                "diagnosis_blind_hcp379_corrected_freesurfer_1_"
                "release_manifest"
            ),
            1,
            True,
        ),
        "archive_replicate_6": Task(
            "archive_replicate_6",
            command(
                archive_replicate_source,
                "--execute",
                "--workers",
                "1",
                "--nthreads",
                str(DEFAULT_THREADS),
            ),
            ("archive_primary_6",),
            (),
            root
            / "archive_corrected_calibration_recovery4/manifests/"
            "archive_corrected_replicate_recovery4_manifest.json",
            (
                "diagnosis_blind_hcp379_archive_corrected_act_"
                "replicate_recovery4_manifest"
            ),
            6,
            True,
        ),
        "archive_low_30": Task(
            "archive_low_30",
            command(archive_low_source, *shared_compute),
            ("archive_primary_6",),
            ("archive_low_pretract_28",),
            root
            / "corrected_archive_low_recovery4/manifests/"
            "corrected_archive_low_30_release_manifest.json",
            (
                "diagnosis_blind_hcp379_corrected_archive_low_30_"
                "release_manifest"
            ),
            30,
            True,
        ),
        "archive_D0_56": Task(
            "archive_D0_56",
            command(archive_D0_source, *shared_compute),
            ("archive_primary_6",),
            ("archive_D0_pretract_52",),
            root
            / "corrected_archive_D0_recovery4/manifests/"
            "corrected_archive_D0_56_release_manifest.json",
            (
                "diagnosis_blind_hcp379_corrected_archive_D0_56_"
                "release_manifest"
            ),
            56,
            True,
        ),
        "archive_concordance": Task(
            "archive_concordance",
            command(ARCHIVE_CONCORDANCE_SOURCE),
            ("archive_primary_6", "archive_replicate_6"),
            (),
            root / "calibration/archive_route_concordance.json",
            (
                "diagnosis_blind_hcp379_archive_route_"
                "concordance_recovery4_validation"
            ),
            6,
            False,
        ),
        "corrected_archive_86": Task(
            "corrected_archive_86",
            command(ARCHIVE_86_SOURCE, "--assemble"),
            ("archive_low_30", "archive_D0_56"),
            (),
            root
            / "corrected_archive_recovery4/manifests/"
            "corrected_archive_86_release_manifest.json",
            (
                "diagnosis_blind_hcp379_corrected_archive_86_"
                "release_manifest"
            ),
            86,
            False,
        ),
        "final_release_530": Task(
            "final_release_530",
            command(FINAL_SOURCE, "--assemble", "--require-ready"),
            (
                "core_227",
                "legacy_tensor_216",
                "freesurfer_1",
                "archive_low_30",
                "archive_D0_56",
                "archive_concordance",
                "corrected_archive_86",
            ),
            (),
            root / "release_candidate_v2/release_manifest.json",
            "diagnosis_blind_hcp379_v2_reference_release",
            530,
            False,
        ),
        "verify_integration_530": Task(
            "verify_integration_530",
            command(INTEGRATION_VERIFY_SOURCE, "--verify"),
            ("final_release_530",),
            (),
            root
            / "release_validation_v1/"
            "integration_verification.json",
            (
                "diagnosis_blind_hcp379_integration_"
                "release_verification"
            ),
            530,
            False,
        ),
        "prepare_analysis_handoff_530": Task(
            "prepare_analysis_handoff_530",
            command(ANALYSIS_HANDOFF_SOURCE, "--prepare"),
            ("verify_integration_530",),
            (),
            root
            / "analysis_handoff_v1/"
            "verified_handoff_receipt.json",
            "hcp379_v2_verified_analysis_handoff_receipt",
            530,
            False,
        ),
    }


def output_status(task: Task, selected_count: int | None) -> dict[str, Any]:
    if not task.output.is_file():
        return {
            "valid": False,
            "present": False,
            "reason": "missing",
            "path": str(task.output),
        }
    try:
        value = load_json(task.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "valid": False,
            "present": True,
            "reason": f"unreadable:{type(exc).__name__}",
            "path": str(task.output),
        }
    valid = bool(
        value.get("record_type") == task.record_type
        and value.get("status") in (
            {"PASS", "FAIL"}
            if task.name == "archive_concordance"
            else {"PASS"}
        )
    )
    reason = "PASS"
    if task.name == "archive_concordance":
        valid = bool(
            valid
            and value.get("diagnosis_labels_used") is False
            and value.get("recovery4_inputs_required") is True
            and value.get("calibration_n") == 6
            and value.get("decision")
            in {ARCHIVE_RETAIN_DECISION, ARCHIVE_RERUN_DECISION}
            and value.get("selected_streamline_count") == selected_count
        )
    elif task.name == "final_release_530":
        valid = bool(
            valid
            and value.get("subject_count") == 530
            and value.get("matrix_family_count") == 9
            and value.get("matrix_file_count") == 4770
            and value.get("matrix_shape") == [379, 379]
            and value.get("diagnosis_or_outcome_fields_present") is False
            and value.get("density_is_whole_cohort_technical_release_gate")
            is True
            and value.get("subjects_at_or_above_minimum_density") == 530
            and float(value.get("minimum_observed_edge_density", -1.0))
            >= MINIMUM_DENSITY
        )
    elif task.name == "verify_integration_530":
        valid = bool(
            valid
            and value.get(
                "verification_is_independent_of_assembler_imports"
            )
            is True
            and value.get("diagnosis_or_outcome_fields_present") is False
            and value.get("subject_count") == 530
            and value.get("matrix_family_count") == 9
            and value.get("matrix_file_count") == 4770
            and value.get("matrix_shape") == [379, 379]
            and value.get("selected_streamline_count") == selected_count
            and value.get("subjects_at_or_above_minimum_density") == 530
            and float(value.get("minimum_observed_edge_density", -1.0))
            >= MINIMUM_DENSITY
            and value.get(
                "subjects_at_or_above_minimum_weighted_support"
            )
            == 530
            and float(
                value.get("minimum_observed_weighted_support", -1.0)
            )
            >= 0.95
            and value.get(
                "subjects_at_or_above_minimum_assignment_fraction"
            )
            == 530
            and float(
                value.get("minimum_observed_assignment_fraction", -1.0)
            )
            >= 0.50
        )
    elif task.name == "prepare_analysis_handoff_530":
        records = value.get("records")
        handoff_record = (
            records.get("handoff_manifest")
            if isinstance(records, Mapping)
            else None
        )
        handoff_path = (
            Path(str(handoff_record.get("path", "")))
            if isinstance(handoff_record, Mapping)
            else Path("")
        )
        try:
            handoff = load_json(handoff_path)
            contract_record = handoff.get("records", {}).get(
                "analysis_contract"
            )
            contract_path = Path(
                str(contract_record.get("path", ""))
            )
            contract = load_json(contract_path)
            bound_records_valid = bool(
                isinstance(records, Mapping)
                and records.get("independent_verification")
                == file_record(
                    HCP_ROOT
                    / "release_validation_v1/"
                    "integration_verification.json"
                )
                and handoff_record == file_record(handoff_path)
                and isinstance(contract_record, Mapping)
                and contract_record == file_record(contract_path)
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            handoff = {}
            contract = {}
            bound_records_valid = False
        valid = bool(
            valid
            and value.get("subject_count") == 530
            and value.get("matrix_family_count") == 9
            and value.get("matrix_file_count") == 4770
            and value.get("matrix_shape") == [379, 379]
            and value.get("diagnosis_or_outcome_fields_present") is False
            and value.get("live_dashboard_modified") is False
            and bound_records_valid
            and handoff.get("record_type")
            == "hcp379_v2_verified_analysis_handoff"
            and handoff.get("status") == "PASS"
            and handoff.get("all_matrix_files_rehashed_during_handoff")
            is True
            and handoff.get("subject_count") == 530
            and handoff.get("matrix_file_count") == 4770
            and handoff.get("live_dashboard_modified") is False
            and handoff.get("live_dashboard_activation_authorized") is False
            and contract.get("record_type")
            == "hcp379_v2_verified_analysis_contract"
            and contract.get("status") == "PASS"
            and contract.get("primary_key") == "unit"
            and contract.get("node_count") == 379
            and contract.get("selected_streamline_count")
            == selected_count
            and contract.get("source_route_field_required") is True
            and contract.get("route_sensitivity_required") is True
            and contract.get(
                "legacy_AAL_dashboard_activation_authorized"
            )
            is False
        )
    else:
        units = value.get("units")
        valid = bool(
            valid
            and value.get("diagnosis_labels_used") is False
            and value.get("unit_count") == task.expected_n
            and isinstance(units, list)
            and len(units) == task.expected_n
            and len(
                {
                    str(row.get("unit"))
                    for row in units
                    if isinstance(row, Mapping)
                }
            )
            == task.expected_n
            and all(
                isinstance(row, Mapping)
                and float(row.get("edge_density", -1.0))
                >= MINIMUM_DENSITY
                and isinstance(row.get("artifacts"), Mapping)
                and MATRIX_ARTIFACT_KEYS
                <= set(row["artifacts"])
                for row in units
            )
        )
        if task.name not in {
            "archive_primary_6",
            "archive_replicate_6",
        }:
            valid = bool(
                valid
                and value.get("selected_streamline_count")
                == selected_count
                and value.get(
                    "density_is_whole_cohort_technical_release_gate"
                )
                is True
                and value.get("minimum_edge_density_inclusive")
                == MINIMUM_DENSITY
                and value.get("subjects_at_or_above_minimum_density")
                == task.expected_n
                and float(
                    value.get("minimum_observed_edge_density", -1.0)
                )
                >= MINIMUM_DENSITY
            )
        else:
            valid = bool(
                valid
                and value.get("selected_streamline_count")
                == selected_count
                and all(
                    isinstance(row, Mapping)
                    and row.get("status") == "PASS_ALL_NINE"
                    and float(row.get("edge_density", -1.0))
                    >= MINIMUM_DENSITY
                    and isinstance(row.get("artifacts"), Mapping)
                    and MATRIX_ARTIFACT_KEYS
                    <= set(row["artifacts"])
                    for row in units
                )
            )
    if not valid:
        reason = "terminal_manifest_contract_mismatch"
    return {
        "valid": valid,
        "present": True,
        "reason": reason,
        "path": str(task.output),
        "record": file_record(task.output),
    }


def plan_payload() -> dict[str, Any]:
    graph = tasks()
    gates = upstream_gates()
    phase = phase_status()
    try:
        human = {
            "ready": True,
            "reason": "PASS",
            "record": validate_human_qc(HUMAN_QC),
        }
    except (OSError, ValueError, TypeError, ImportError) as exc:
        human = {
            "ready": False,
            "reason": f"{type(exc).__name__}:{exc}",
            "path": str(HUMAN_QC),
        }
    gate_states = {
        name: upstream_status(gate) for name, gate in gates.items()
    }
    selected = phase.get("selected_streamline_count")
    task_states = {
        name: output_status(task, selected)
        for name, task in graph.items()
    }
    if not human["ready"]:
        status = "AWAITING_GENUINE_HUMAN_QC"
    elif not phase["ready"]:
        status = "AWAITING_PHASE_B"
    elif not all(value["ready"] for value in gate_states.values()):
        status = "AWAITING_PRETRACT"
    elif task_states["prepare_analysis_handoff_530"]["valid"]:
        status = "VERIFIED_ANALYSIS_HANDOFF_ALREADY_PASS"
    else:
        status = "READY_OR_RESUMABLE"
    source_paths = {
        Path(task.command[1])
        for task in graph.values()
        if len(task.command) >= 2
    } | {Path(__file__)}
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_parallel_tractography_release_plan"
        ),
        "status": status,
        "generated_utc": utc_now(),
        "diagnosis_or_outcomes_used": False,
        "execution_authorized_by_plan": False,
        "imaging_executed_by_plan": False,
        "non_overwriting": True,
        "selected_streamline_count": selected,
        "human_qc": human,
        "phase_b": phase,
        "upstream_gates": gate_states,
        "resources": {
            "maximum_active_compute_tasks": DEFAULT_MAX_ACTIVE,
            "subject_workers_per_compute_task": 1,
            "threads_per_subject": DEFAULT_THREADS,
            "declared_peak_tractography_threads": (
                DEFAULT_MAX_ACTIVE * DEFAULT_THREADS
            ),
            "logical_cpus": os.cpu_count(),
            "minimum_start_free_bytes": MINIMUM_START_FREE_BYTES,
            "emergency_free_bytes": EMERGENCY_FREE_BYTES,
            "mandatory_compaction_after_pass": True,
            "cross_supervisor_cpu_guard": (
                "zero tractography workers until the parallel Phase-B "
                "process is terminal; at most two while the four-by-eight "
                "thread pretract supervisor remains nonterminal; at most "
                "three after pretract is terminal"
            ),
        },
        "tasks": {
            name: {
                "dependencies": list(task.dependencies),
                "upstream_gates": list(task.upstream_gates),
                "compute_task": task.compute_task,
                "command": list(task.command),
                "output": str(task.output),
                "record_type": task.record_type,
                "expected_n": task.expected_n,
                "current_output": task_states[name],
            }
            for name, task in graph.items()
        },
        "release_contract": {
            "subject_count": 530,
            "matrix_family_count": 9,
            "matrix_file_count": 4770,
            "minimum_density_inclusive": MINIMUM_DENSITY,
            "subject_exclusion_allowed": False,
            "edge_imputation_allowed": False,
            "per_subject_streamline_tuning_allowed": False,
            "archive_route_decision_required": True,
            "corrected_archive_86_sensitivity_required_by_supervisor": True,
            "independent_post_assembly_verification_required": True,
            "immutable_verified_analysis_handoff_required": True,
            "legacy_dashboard_switch_authorized": False,
        },
        "sources": {
            path.name: file_record(path)
            for path in sorted(source_paths, key=str)
        },
    }


def validate_graph() -> dict[str, Any]:
    graph = tasks()
    names = set(graph)
    checks = {
        "exact_twelve_tasks": len(graph) == 12,
        "all_dependencies_exist": all(
            set(task.dependencies) <= names for task in graph.values()
        ),
        "all_outputs_unique": len({task.output for task in graph.values()})
        == len(graph),
        "no_overwrite_flag": all(
            "--overwrite" not in task.command for task in graph.values()
        ),
        "all_compute_tasks_one_worker": all(
            (not task.compute_task)
            or (
                "--workers" in task.command
                and task.command[
                    task.command.index("--workers") + 1
                ]
                == "1"
            )
            for task in graph.values()
        ),
        "bulk_compute_compaction": all(
            (
                task.name == "archive_replicate_6"
                or "--compact-tractograms-after-pass" in task.command
            )
            for task in graph.values()
            if task.compute_task
        ),
        "final_depends_on_all_release_lanes": set(
            graph["final_release_530"].dependencies
        )
        == {
            "core_227",
            "legacy_tensor_216",
            "freesurfer_1",
            "archive_low_30",
            "archive_D0_56",
            "archive_concordance",
            "corrected_archive_86",
        },
        "release_partition_530": 227 + 216 + 1 + 30 + 56 == 530,
        "resource_peak_within_host": (
            DEFAULT_MAX_ACTIVE * DEFAULT_THREADS
            <= (os.cpu_count() or 0)
        ),
        "archive_decision_is_evidence_task": (
            graph["archive_concordance"].dependencies
            == ("archive_primary_6", "archive_replicate_6")
        ),
        "corrected_archive_is_30_plus_56": (
            set(graph["corrected_archive_86"].dependencies)
            == {"archive_low_30", "archive_D0_56"}
        ),
        "final_command_requires_readiness": (
            "--require-ready" in graph["final_release_530"].command
            and "--assemble" in graph["final_release_530"].command
        ),
        "independent_verifier_precedes_handoff": (
            graph["verify_integration_530"].dependencies
            == ("final_release_530",)
            and graph["verify_integration_530"].command[-1] == "--verify"
        ),
        "verified_handoff_is_terminal": (
            graph["prepare_analysis_handoff_530"].dependencies
            == ("verify_integration_530",)
            and graph["prepare_analysis_handoff_530"].command[-1]
            == "--prepare"
        ),
    }
    # Topological walk proves acyclicity.
    complete: set[str] = set()
    while len(complete) < len(graph):
        ready = {
            name
            for name, task in graph.items()
            if name not in complete
            and set(task.dependencies) <= complete
        }
        if not ready:
            break
        complete |= ready
    checks["acyclic_graph"] = complete == names
    passed = sum(checks.values())
    return {
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed": passed,
        "total": len(checks),
        "checks": checks,
    }


def free_bytes() -> int:
    return shutil.disk_usage(HCP_ROOT).free


def latest_pretract_supervisor_state() -> dict[str, Any] | None:
    if not PRETRACT_ATTEMPT_ROOT.is_dir():
        return None
    candidates = sorted(
        PRETRACT_ATTEMPT_ROOT.glob("*/supervisor_state.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            value = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if value.get("record_type") == (
            "diagnosis_blind_hcp379_parallel_pretract_supervisor"
        ):
            return {**value, "_path": str(path)}
    return None


def cross_supervisor_compute_limit(requested: int) -> dict[str, Any]:
    """Keep combined pretract plus tractography demand within 64 threads."""
    pretract = latest_pretract_supervisor_state()
    if pretract is None:
        return {
            "limit": requested,
            "reason": "no_parallel_pretract_supervisor_state",
            "pretract_state": None,
        }
    status = str(pretract.get("status", ""))
    terminal = status in {
        "PASS_PHASE_B_AND_ALL_PRETRACT",
        "FAIL_PHASE_B_OR_PRETRACT",
        "EMERGENCY_STORAGE_STOP",
        "INTERRUPTED",
    }
    phase_returncode = (
        pretract.get("phase_b", {}).get("returncode")
        if isinstance(pretract.get("phase_b"), Mapping)
        else None
    )
    if terminal:
        limit = requested
        reason = "parallel_pretract_terminal"
    elif phase_returncode is None:
        limit = 0
        reason = "parallel_phase_b_process_not_terminal"
    else:
        # The pretract supervisor may refill to four eight-thread workers
        # between polls. Two sixteen-thread tractography workers therefore
        # preserve the 64-thread host ceiling under the worst refill.
        limit = min(requested, 2)
        reason = "parallel_pretract_may_use_four_by_eight_threads"
    return {
        "limit": limit,
        "reason": reason,
        "pretract_status": status,
        "phase_b_returncode": phase_returncode,
        "pretract_state": pretract.get("_path"),
    }


def terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def execute(args: argparse.Namespace) -> int:
    # This check deliberately occurs before attempt/state creation.
    human_record = validate_human_qc(args.human_qc)
    graph_validation = validate_graph()
    if graph_validation["status"] != "PASS":
        raise ValueError("tractography-release DAG validation failed")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt = ATTEMPT_ROOT / f"{stamp}-selected-tractography-release"
    attempt.mkdir(parents=True, exist_ok=False)
    graph = tasks()
    gates = upstream_gates()
    task_state: dict[str, dict[str, Any]] = {
        name: {
            "status": "PENDING",
            "dependencies": list(task.dependencies),
            "upstream_gates": list(task.upstream_gates),
            "output": str(task.output),
        }
        for name, task in graph.items()
    }
    state: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_parallel_tractography_release_state"
        ),
        "status": "STARTING",
        "started_utc": utc_now(),
        "updated_utc": utc_now(),
        "diagnosis_or_outcomes_used": False,
        "non_overwriting": True,
        "attempt_root": str(attempt),
        "human_qc": human_record,
        "phase_b": phase_status(),
        "upstream_gates": {
            name: upstream_status(gate)
            for name, gate in gates.items()
        },
        "resources": {
            "maximum_active_compute_tasks": args.max_active,
            "threads_per_subject": args.nthreads,
            "subject_workers_per_compute_task": 1,
            "minimum_start_free_bytes": MINIMUM_START_FREE_BYTES,
            "emergency_free_bytes": EMERGENCY_FREE_BYTES,
        },
        "tasks": task_state,
    }

    def write_state(status: str | None = None) -> None:
        if status is not None:
            state["status"] = status
        state["updated_utc"] = utc_now()
        state["free_bytes"] = free_bytes()
        state["phase_b"] = phase_status()
        state["upstream_gates"] = {
            name: upstream_status(gate)
            for name, gate in gates.items()
        }
        atomic_json(STATE, state)
        atomic_json(attempt / "state.json", state)

    active: dict[
        str, tuple[subprocess.Popen[Any], Any, Path]
    ] = {}
    selected: int | None = None
    deadline = (
        time.monotonic() + args.upstream_timeout_hours * 3600.0
    )
    try:
        write_state("WAITING_FOR_PHASE_B_AND_PRETRACT")
        while True:
            phase = phase_status()
            selected = (
                int(phase["selected_streamline_count"])
                if phase["ready"]
                else None
            )
            gate_states = {
                name: upstream_status(gate)
                for name, gate in gates.items()
            }

            # Reuse only exact valid terminal manifests. Invalid existing
            # outputs are immutable failures, never overwrite targets.
            for name, task in graph.items():
                if task_state[name]["status"] != "PENDING":
                    continue
                if selected is None:
                    continue
                terminal = output_status(task, selected)
                if terminal["present"] and terminal["valid"]:
                    task_state[name].update(
                        {
                            "status": "PASS_REUSED_VALIDATED",
                            "terminal": terminal,
                        }
                    )
                elif terminal["present"] and not terminal["valid"]:
                    task_state[name].update(
                        {
                            "status": "FAIL_INVALID_EXISTING_OUTPUT",
                            "terminal": terminal,
                        }
                    )

            # Harvest completed processes.
            for name in list(active):
                process, handle, log = active[name]
                returncode = process.poll()
                if returncode is None:
                    continue
                handle.close()
                del active[name]
                terminal = output_status(graph[name], selected)
                task_state[name].update(
                    {
                        "completed_utc": utc_now(),
                        "returncode": returncode,
                        "log": str(log),
                        "terminal": terminal,
                        "status": (
                            "PASS"
                            if returncode == 0 and terminal["valid"]
                            else "FAIL"
                        ),
                    }
                )

            failed = {
                name
                for name, value in task_state.items()
                if str(value["status"]).startswith("FAIL")
            }
            passed = {
                name
                for name, value in task_state.items()
                if str(value["status"]).startswith("PASS")
            }
            for name, task in graph.items():
                if task_state[name]["status"] != "PENDING":
                    continue
                failed_dependencies = sorted(
                    set(task.dependencies) & failed
                )
                if failed_dependencies:
                    task_state[name].update(
                        {
                            "status": "FAIL_UPSTREAM_DEPENDENCY",
                            "failed_dependencies": failed_dependencies,
                        }
                    )

            # Start ready work without exceeding the storage/thread contract.
            started = False
            cross_limit = cross_supervisor_compute_limit(
                args.max_active
            )
            state["cross_supervisor_compute_limit"] = cross_limit
            for name, task in graph.items():
                if task_state[name]["status"] != "PENDING":
                    continue
                if not phase["ready"]:
                    continue
                if not set(task.dependencies) <= passed:
                    continue
                if not all(
                    gate_states[gate]["ready"]
                    for gate in task.upstream_gates
                ):
                    continue
                active_compute = sum(
                    graph[key].compute_task for key in active
                )
                if (
                    task.compute_task
                    and active_compute >= cross_limit["limit"]
                ):
                    continue
                if task.compute_task and free_bytes() < MINIMUM_START_FREE_BYTES:
                    continue
                log = attempt / f"{name}.log"
                handle = log.open("w", encoding="utf-8")
                task_command = list(task.command)
                if task.compute_task:
                    task_command[
                        task_command.index("--nthreads") + 1
                    ] = str(args.nthreads)
                process = subprocess.Popen(
                    task_command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                    cwd=EXP,
                )
                active[name] = (process, handle, log)
                task_state[name].update(
                    {
                        "status": "RUNNING",
                        "started_utc": utc_now(),
                        "pid": process.pid,
                        "command": task_command,
                        "log": str(log),
                    }
                )
                started = True

            if free_bytes() < EMERGENCY_FREE_BYTES:
                for process, handle, _ in active.values():
                    terminate_process(process)
                    handle.close()
                for name in active:
                    task_state[name]["status"] = (
                        "FAIL_EMERGENCY_STORAGE_STOP"
                    )
                active.clear()
                write_state("FAIL_EMERGENCY_STORAGE_STOP")
                return 1

            final_status = str(
                task_state["prepare_analysis_handoff_530"]["status"]
            )
            if final_status.startswith("PASS"):
                write_state("PASS_VERIFIED_ANALYSIS_HANDOFF_530")
                return 0
            if not active and all(
                value["status"] != "PENDING"
                for value in task_state.values()
            ):
                write_state("FAIL_TERMINAL_TASKS")
                return 1
            if (
                not active
                and not started
                and time.monotonic() >= deadline
            ):
                for value in task_state.values():
                    if value["status"] == "PENDING":
                        value["status"] = "FAIL_UPSTREAM_TIMEOUT"
                write_state("FAIL_UPSTREAM_TIMEOUT")
                return 1
            write_state(
                "RUNNING"
                if active
                else "WAITING_FOR_PHASE_B_AND_PRETRACT"
            )
            time.sleep(args.poll_seconds)
    except (KeyboardInterrupt, SystemExit):
        for process, handle, _ in active.values():
            terminate_process(process)
            handle.close()
        for name in active:
            task_state[name]["status"] = "INTERRUPTED"
        write_state("INTERRUPTED")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--human-qc", type=Path, default=HUMAN_QC)
    parser.add_argument(
        "--max-active",
        type=int,
        default=DEFAULT_MAX_ACTIVE,
        choices=(1, 2, 3),
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=DEFAULT_THREADS,
        choices=(8, 12, 16),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--upstream-timeout-hours",
        type=float,
        default=DEFAULT_UPSTREAM_TIMEOUT_HOURS,
    )
    args = parser.parse_args()
    if args.poll_seconds < 1.0 or args.upstream_timeout_hours <= 0:
        parser.error("poll interval and timeout must be positive")
    if args.max_active * args.nthreads > (os.cpu_count() or 0):
        parser.error("declared tractography threads exceed host CPUs")
    validation = validate_graph()
    if args.self_test:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0 if validation["status"] == "PASS" else 1
    if args.execute:
        return execute(args)
    payload = plan_payload()
    payload["graph_validation"] = validation
    atomic_json(PLAN, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
