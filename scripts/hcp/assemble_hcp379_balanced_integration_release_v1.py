#!/home/ec2-user/fsl/bin/python
"""Assemble the immutable 15+515 balanced HCP379-v2 integration release.

The default mode is a diagnosis-blind readiness audit.  ``--assemble`` is
fail-closed behind the independently replayed final-HROI recipe selection,
the completed 16-page human atlas review, the exact 530-unit topology, and a
complete 515-subject balanced production state.  It re-hashes and re-opens all
4,770 matrices, recomputes subject-level technical QC, and writes only
manifest/index artifacts.  Source derivatives are never copied or modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
TOPOLOGY_BUILDER = (
    EXP / "scripts/hcp/build_hcp379_balanced_release_topology_v1.py"
)
TOPOLOGY_VALIDATOR = (
    EXP
    / "research_audit/"
    "validate_hcp379_balanced_release_topology_v1.py"
)
TOPOLOGY = (
    EXP
    / "research_audit/outputs/"
    "hcp379_balanced_release_topology_v1/topology.json"
)
TOPOLOGY_VALIDATION = TOPOLOGY.with_name("validation.json")
SELECTION = (
    HCP_ROOT
    / "phase_b_hroi_balanced_all_nine_selection_v1/selection.json"
)
SELECTION_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_hroi_all_nine_selection_v1/validation.json"
)
EQUIVALENCE_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_balanced_canary_production_equivalence_v1/validation.json"
)
ARCHIVE_PRIMARY_COMPATIBILITY = (
    EXP
    / "research_audit/outputs/"
    "hcp379_archive_primary_compatibility_v1/audit.json"
)
ARCHIVE_PRIMARY_COMPATIBILITY_VALIDATION = (
    ARCHIVE_PRIMARY_COMPATIBILITY.with_name("validation.json")
)
PRODUCTION_RUNNER = (
    EXP / "scripts/hcp/run_hcp379_balanced_production_v1.py"
)
ALL_NINE_BUILDER = (
    EXP
    / "scripts/hcp/"
    "build_hcp379_hroi_balanced_all_nine_canary_v1.py"
)
HUMAN_REVIEW_ROOT = (
    HCP_ROOT
    / "source_label_repair_v1/review/hroi_release_v1/attempts/"
    "20260728T083614.440650Z-hroi-release-review"
)
HUMAN_REVIEW_MANIFEST = HUMAN_REVIEW_ROOT / "review_manifest.json"
HUMAN_REVIEW = (
    HUMAN_REVIEW_ROOT / "human_visual_qc_hroi_release_completed.csv"
)
PRODUCTION_ROOT = HCP_ROOT / "balanced_release_v1/production"
PRODUCTION_STATE = PRODUCTION_ROOT / "cohort_state.json"
ROOT = HCP_ROOT / "balanced_release_v1/integration_release_v1"
ATTEMPTS = ROOT / "attempts"
READINESS = ROOT / "readiness.json"
CURRENT = ROOT / "current_release.json"

EXPECTED_SUBJECTS = 530
EXPECTED_CANARIES = 15
EXPECTED_PRODUCTION = 515
EXPECTED_NODES = 379
MATRIX_NAMES = (
    "count",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)
EXPECTED_MATRIX_FILES = EXPECTED_SUBJECTS * len(MATRIX_NAMES)
BALANCED_TOTALS = {
    6_000_000: 3_000_000,
    10_000_000: 5_000_000,
    15_000_000: 7_500_000,
    20_000_000: 10_000_000,
}
DENSITY_TARGET = 0.60
MINIMUM_CONNECTED_NODES = 360
MINIMUM_ENDPOINT_ASSIGNMENT_FRACTION = 0.50
MINIMUM_WEIGHTED_SUPPORT_FRACTION = 0.99
HUB_IDS = (361, 365, 366, 370, 374, 375)
EXPECTED_ROUTE_COUNTS = {
    "corrected_core": 227,
    "corrected_legacy_tensor": 216,
    "archive_calibration": 6,
    "corrected_archive_low": 28,
    "corrected_archive_D0": 52,
    "corrected_freesurfer": 1,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def verify_record(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or dict(record) != file_record(path)
    ):
        raise ValueError(f"{label}: file record differs")
    return path


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    delimiter: str = ",",
) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"inconsistent table columns: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter=delimiter
        )
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def parse_utc(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return False
    return parsed.tzinfo is not None


def human_review_contract() -> dict[str, Any]:
    if not HUMAN_REVIEW_MANIFEST.is_file():
        raise FileNotFoundError(HUMAN_REVIEW_MANIFEST)
    manifest = load_json(HUMAN_REVIEW_MANIFEST)
    units = {
        str(row.get("unit", ""))
        for row in manifest.get("units", [])
        if isinstance(row, Mapping)
    }
    if (
        manifest.get("record_type")
        != "diagnosis_blind_hcp379_hroi_release_visual_review"
        or manifest.get("status") != "READY_FOR_HUMAN_VISUAL_QC"
        or manifest.get("diagnosis_labels_used") is not False
        or manifest.get("human_visual_qc_inferred") is not False
        or manifest.get("unit_n") != 16
        or len(units) != 16
    ):
        raise ValueError("final-HROI visual-review manifest differs")
    if not HUMAN_REVIEW.is_file() or HUMAN_REVIEW.is_symlink():
        raise FileNotFoundError(HUMAN_REVIEW)
    rows = read_csv(HUMAN_REVIEW)
    required = {
        "unit",
        "source_support_status",
        "dwi_atlas_status",
        "overall_status",
        "reviewer",
        "reviewed_utc",
        "note",
    }
    if (
        len(rows) != 16
        or {row["unit"] for row in rows} != units
        or set(rows[0]) != required
        or any(
            (
                row["source_support_status"].upper(),
                row["dwi_atlas_status"].upper(),
                row["overall_status"].upper(),
            )
            != ("PASS", "PASS", "PASS")
            or not row["reviewer"]
            or not parse_utc(row["reviewed_utc"])
            for row in rows
        )
    ):
        raise ValueError("completed final-HROI human review differs")
    return {
        "status": "PASS_COMPLETED_HUMAN_HROI_REVIEW",
        "completed_unit_n": len(rows),
        "review": file_record(HUMAN_REVIEW),
        "manifest": file_record(HUMAN_REVIEW_MANIFEST),
    }


def selection_contract() -> dict[str, Any]:
    selection = load_json(SELECTION)
    validation = load_json(SELECTION_VALIDATION)
    selected = selection.get("selected_balanced_total_streamlines")
    per_seed = selection.get("selected_per_seed_streamlines")
    evaluated = selection.get("evaluated_candidates", [])
    selected_rows = [
        row
        for row in evaluated
        if isinstance(row, Mapping)
        and row.get("balanced_total_streamlines") == selected
        and row.get("status") == "PASS_JOINT_ALL_NINE"
    ]
    earlier = [
        row
        for row in evaluated
        if isinstance(row, Mapping)
        and int(row.get("balanced_total_streamlines", 0))
        < int(selected or 0)
    ]
    runtime = validation.get("runtime", {})
    if (
        selection.get("record_type")
        != "diagnosis_blind_hcp379_hroi_all_nine_recipe_selection"
        or selection.get("status")
        != "PASS_SMALLEST_JOINTLY_QUALIFIED_BALANCED_RECIPE"
        or selection.get("diagnosis_labels_used") is not False
        or selection.get("non_overwriting") is not True
        or selected not in BALANCED_TOTALS
        or per_seed != BALANCED_TOTALS.get(selected)
        or selection.get("selection_is_smallest_joint_pass") is not True
        or selection.get("per_subject_streamline_tuning_allowed") is not False
        or selection.get("subject_exclusion_for_low_density_allowed")
        is not False
        or len(selected_rows) != 1
        or any(row.get("status") != "FAIL_JOINT_ALL_NINE" for row in earlier)
        or validation.get("record_type")
        != "hcp379_hroi_all_nine_selection_v1_validation"
        or validation.get("status") != "PASS"
        or validation.get("runtime_selection_detected") is not True
        or runtime.get("status") != "PASS"
        or runtime.get("selected_balanced_total_streamlines") != selected
        or runtime.get("selected_per_seed_streamlines") != per_seed
        or runtime.get("selection") != file_record(SELECTION)
    ):
        raise ValueError("runtime balanced recipe selection differs")
    cohort_summary_path = verify_record(
        selected_rows[0]["summary"],
        label="selected canary cohort summary",
    )
    cohort = load_json(cohort_summary_path)
    if (
        cohort.get("record_type")
        != (
            "diagnosis_blind_hcp379_hroi_balanced_all_nine_"
            "canary_cohort_summary"
        )
        or cohort.get("status")
        != "PASS_COMPLETE_HROI_ALL_NINE_CANARY"
        or cohort.get("processed_unit_n") != EXPECTED_CANARIES
        or cohort.get("selected_balanced_total_streamlines") != selected
        or cohort.get("selected_per_seed_prefix_streamlines") != per_seed
        or cohort.get("uniform_candidate_recipe_confirmed") is not True
        or cohort.get("all_units_density_target_met") is not True
        or cohort.get("all_units_all_nine_technical_qc_pass") is not True
        or cohort.get("all_units_seed_reliability_pass") is not True
        or len(cohort.get("units", [])) != EXPECTED_CANARIES
    ):
        raise ValueError("selected canary cohort evidence differs")
    return {
        "status": "PASS_RUNTIME_RECIPE_SELECTION",
        "selected_total": int(selected),
        "per_seed": int(per_seed),
        "selection": file_record(SELECTION),
        "validation": file_record(SELECTION_VALIDATION),
        "canary_cohort_summary": file_record(cohort_summary_path),
        "canary_cohort": cohort,
    }


def equivalence_contract() -> dict[str, Any]:
    value = load_json(EQUIVALENCE_VALIDATION)
    records = value.get("records", {})
    if (
        value.get("record_type")
        != "hcp379_balanced_canary_production_equivalence_validation"
        or value.get("status") != "PASS"
        or value.get("complete_seed_metadata") is not True
        or value.get("seed_metadata_expected_n") != 30
        or value.get("seed_metadata_pass_n") != 30
        or value.get("seed_metadata_waiting_n") != 0
        or value.get("seed_metadata_failure_n") != 0
        or value.get("canary_pair_complete_n") != EXPECTED_CANARIES
        or value.get("production_execution_authorized") is not True
        or value.get("diagnosis_labels_used") is not False
        or value.get("imaging_executed") is not False
        or records.get("production_runner")
        != file_record(PRODUCTION_RUNNER)
        or records.get("all_nine_builder")
        != file_record(ALL_NINE_BUILDER)
    ):
        raise ValueError(
            "complete calibration-production equivalence differs"
        )
    return {
        "status": "PASS_COMPLETE_CANARY_PRODUCTION_EQUIVALENCE",
        "validation": file_record(EQUIVALENCE_VALIDATION),
        "seed_metadata_pass_n": 30,
        "canary_pair_complete_n": EXPECTED_CANARIES,
    }


def archive_primary_compatibility_contract() -> dict[str, Any]:
    audit = load_json(ARCHIVE_PRIMARY_COMPATIBILITY)
    validation = load_json(ARCHIVE_PRIMARY_COMPATIBILITY_VALIDATION)
    if (
        audit.get("record_type")
        != "diagnosis_blind_hcp379_archive_primary_compatibility_audit"
        or audit.get("status")
        != "PASS_EXISTING_ARCHIVE_NOT_PRIMARY_RELEASE_COMPATIBLE"
        or audit.get("decision")
        != (
            "REBUILD_ALL_86_ARCHIVE_UNITS_WITH_THE_UNIFORM_BALANCED_"
            "FINAL_HROI_ACT_RECIPE"
        )
        or audit.get("diagnosis_labels_used") is not False
        or audit.get("outcomes_used") is not False
        or audit.get("archive_existing_n") != 86
        or audit.get("archive_existing_density_qualified_n") != 56
        or audit.get("historical_route") != "NO_ACT_RESTORED_TRACK"
        or audit.get("historical_route_n") != 86
        or audit.get("historical_atlas_source_n") != 86
        or audit.get("final_hroi_changed_voxel_subject_n") != 86
        or audit.get("balanced_archive_production_n") != 86
        or audit.get("scientific_interpretation", {}).get(
            "existing_archive_matrices_are_primary_release_inputs"
        )
        is not False
        or validation.get("record_type")
        != "hcp379_archive_primary_compatibility_v1_validation"
        or validation.get("status") != "PASS"
        or validation.get("passed_checks") != 6
        or validation.get("total_checks") != 6
        or validation.get("audit")
        != file_record(ARCHIVE_PRIMARY_COMPATIBILITY)
    ):
        raise ValueError("archive primary-compatibility contract differs")
    return {
        "status": "PASS_UNIFORM_ARCHIVE_REBUILD_REQUIRED",
        "audit": file_record(ARCHIVE_PRIMARY_COMPATIBILITY),
        "validation": file_record(
            ARCHIVE_PRIMARY_COMPATIBILITY_VALIDATION
        ),
    }


def topology_contract() -> dict[str, Any]:
    build = subprocess.run(
        [str(PYTHON), str(TOPOLOGY_BUILDER)],
        capture_output=True,
        text=True,
        check=False,
    )
    validate = subprocess.run(
        [str(PYTHON), str(TOPOLOGY_VALIDATOR)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        validation = json.loads(validate.stdout)
    except Exception as exc:
        raise ValueError("topology validator stdout is not JSON") from exc
    topology = load_json(TOPOLOGY)
    canaries = topology.get("canaries", [])
    production = topology.get("production", [])
    if (
        build.returncode != 0
        or validate.returncode != 0
        or validation.get("status") != "PASS"
        or validation.get("topology") != file_record(TOPOLOGY)
        or topology.get("record_type")
        != "diagnosis_blind_hcp379_balanced_release_topology"
        or topology.get("diagnosis_labels_used") is not False
        or topology.get("non_overwriting") is not True
        or topology.get("cohort_uniform") is not True
        or topology.get("total_unit_n") != EXPECTED_SUBJECTS
        or topology.get("canary_reuse_n") != EXPECTED_CANARIES
        or topology.get("production_execution_n") != EXPECTED_PRODUCTION
        or len(canaries) != EXPECTED_CANARIES
        or len(production) != EXPECTED_PRODUCTION
        or len(
            {str(row.get("unit", "")) for row in canaries + production}
        )
        != EXPECTED_SUBJECTS
        or any(not row.get("source_recovery_lane") for row in canaries)
    ):
        raise ValueError("exact balanced release topology differs")
    return {
        "status": (
            "PASS_ALL_515_PRETRACT_READY"
            if topology.get("pretract_ready_n") == EXPECTED_PRODUCTION
            and all(row.get("pretract", {}).get("ready") for row in production)
            else "WAITING_FOR_ALL_515_PRETRACT"
        ),
        "topology": topology,
        "record": file_record(TOPOLOGY),
        "validation": file_record(TOPOLOGY_VALIDATION),
    }


def production_contract(selected: int, per_seed: int) -> dict[str, Any]:
    state = load_json(PRODUCTION_STATE)
    subjects = state.get("subjects", [])
    units = {
        str(row.get("unit", ""))
        for row in subjects
        if isinstance(row, Mapping)
    }
    if (
        state.get("record_type")
        != "diagnosis_blind_hcp379_balanced_production_cohort"
        or state.get("status") != "PASS_BALANCED_PRODUCTION_COHORT"
        or state.get("diagnosis_labels_used") is not False
        or state.get("non_overwriting") is not True
        or state.get("selected_balanced_total_streamlines") != selected
        or state.get("selected_per_seed_streamlines") != per_seed
        or state.get("target_n") != EXPECTED_PRODUCTION
        or state.get("full_production_n") != EXPECTED_PRODUCTION
        or state.get("processed_n") != EXPECTED_PRODUCTION
        or state.get("pass_n") != EXPECTED_PRODUCTION
        or state.get("unstarted_n") != 0
        or state.get("stopped_early") is not False
        or state.get("all_subjects_density_target_met") is not True
        or state.get("per_subject_streamline_tuning_allowed") is not False
        or state.get("subject_exclusion_for_low_density_allowed") is not False
        or len(subjects) != EXPECTED_PRODUCTION
        or len(units) != EXPECTED_PRODUCTION
        or any(
            row.get("status") != "PASS_BALANCED_PRODUCTION"
            or row.get("combined_density_target_met") is not True
            for row in subjects
        )
    ):
        raise ValueError("complete 515-subject production state differs")
    return {
        "status": "PASS_COMPLETE_515_BALANCED_PRODUCTION",
        "state": state,
        "record": file_record(PRODUCTION_STATE),
    }


def current_readiness() -> tuple[dict[str, Any], dict[str, Any]]:
    blockers: list[str] = []
    evidence: dict[str, Any] = {}
    selected = None
    per_seed = None
    try:
        selection = selection_contract()
        selected = selection["selected_total"]
        per_seed = selection["per_seed"]
        evidence["selection"] = selection
    except Exception as exc:
        blockers.append(f"recipe_selection:{type(exc).__name__}:{exc}")
    try:
        evidence["human_review"] = human_review_contract()
    except Exception as exc:
        blockers.append(f"human_review:{type(exc).__name__}:{exc}")
    try:
        evidence["equivalence"] = equivalence_contract()
    except Exception as exc:
        blockers.append(
            f"canary_production_equivalence:{type(exc).__name__}:{exc}"
        )
    try:
        evidence["archive_primary_compatibility"] = (
            archive_primary_compatibility_contract()
        )
    except Exception as exc:
        blockers.append(
            f"archive_primary_compatibility:{type(exc).__name__}:{exc}"
        )
    try:
        topology = topology_contract()
        evidence["topology"] = topology
        if topology["status"] != "PASS_ALL_515_PRETRACT_READY":
            blockers.append("topology:all_515_pretract_not_ready")
    except Exception as exc:
        blockers.append(f"topology:{type(exc).__name__}:{exc}")
    if selected is not None and per_seed is not None:
        try:
            evidence["production"] = production_contract(
                selected, per_seed
            )
        except Exception as exc:
            blockers.append(f"production:{type(exc).__name__}:{exc}")
    else:
        blockers.append("production:waiting_for_recipe_selection")
    readiness = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_balanced_integration_release_readiness"
        ),
        "generated_utc": utc_now(),
        "status": "READY_TO_ASSEMBLE" if not blockers else "NOT_READY",
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "subject_count": EXPECTED_SUBJECTS,
        "matrix_family_count": len(MATRIX_NAMES),
        "matrix_file_count": EXPECTED_MATRIX_FILES,
        "density_target": DENSITY_TARGET,
        "selected_balanced_total_streamlines": selected,
        "selected_per_seed_streamlines": per_seed,
        "blockers": blockers,
        "assembly_authorized": not blockers,
        "final_release_authorized": False,
        "source_files_modified": False,
    }
    return readiness, evidence


def load_matrix(path: Path) -> np.ndarray:
    value = np.loadtxt(path, delimiter=",")
    if value.shape != (EXPECTED_NODES, EXPECTED_NODES):
        raise ValueError(f"{path}: matrix shape is {value.shape}")
    return value


def load_node_volumes(path: Path) -> np.ndarray:
    rows = read_csv(path)
    if len(rows) != EXPECTED_NODES:
        raise ValueError("node-volume row count differs")
    rows.sort(key=lambda row: int(row["node_id"]))
    if [int(row["node_id"]) for row in rows] != list(
        range(1, EXPECTED_NODES + 1)
    ):
        raise ValueError("node-volume IDs differ")
    values = np.asarray(
        [float(row["volume_mm3"]) for row in rows], dtype=float
    )
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("node volumes are non-finite or non-positive")
    return values


def matrix_qc(
    *,
    unit: str,
    matrix_records: Mapping[str, Any],
    node_volumes_record: Mapping[str, Any],
    endpoint_assignment_fraction: float,
) -> dict[str, Any]:
    if set(matrix_records) != set(MATRIX_NAMES):
        raise ValueError(f"{unit}: matrix family differs")
    matrices = {
        name: load_matrix(
            verify_record(
                matrix_records[name], label=f"{unit}/matrix/{name}"
            )
        )
        for name in MATRIX_NAMES
    }
    failures: list[str] = []
    for name, value in matrices.items():
        if not np.isfinite(value).all():
            failures.append(f"{name}:non_finite")
        if np.any(value < -1.0e-12):
            failures.append(f"{name}:negative")
        if not np.allclose(value, value.T, atol=1.0e-8, rtol=1.0e-7):
            failures.append(f"{name}:asymmetric")
        if not np.allclose(np.diag(value), 0.0, atol=1.0e-10, rtol=0):
            failures.append(f"{name}:nonzero_diagonal")
    count = matrices["count"]
    if not np.allclose(count, np.rint(count), atol=1.0e-6, rtol=0):
        failures.append("count:not_integer")
    support = count > 0
    np.fill_diagonal(support, False)
    triangle = np.triu_indices(EXPECTED_NODES, 1)
    supported = int(np.count_nonzero(support[triangle]))
    possible = EXPECTED_NODES * (EXPECTED_NODES - 1) // 2
    density = supported / possible
    connected_nodes = int(np.count_nonzero(np.sum(count, axis=1) > 0))
    hub_strengths = {
        str(node): float(np.sum(count[node - 1])) for node in HUB_IDS
    }
    if density < DENSITY_TARGET:
        failures.append(f"density:{density:.12f}")
    if connected_nodes < MINIMUM_CONNECTED_NODES:
        failures.append(f"connected_nodes:{connected_nodes}")
    if any(value <= 0 for value in hub_strengths.values()):
        failures.append("locked_hub_disconnected")
    support_agreement: dict[str, Any] = {}
    for name in MATRIX_NAMES:
        if name == "count":
            continue
        weighted_support = matrices[name] > 0
        np.fill_diagonal(weighted_support, False)
        outside = int(
            np.count_nonzero((weighted_support & ~support)[triangle])
        )
        covered = int(
            np.count_nonzero((weighted_support & support)[triangle])
        )
        fraction = covered / supported if supported else 0.0
        support_agreement[name] = {
            "count_edges_covered": covered,
            "count_supported_edges": supported,
            "count_edge_support_fraction": fraction,
            "weighted_edges_outside_count_support": outside,
        }
        if outside:
            failures.append(f"{name}:support_outside_count:{outside}")
        if fraction < MINIMUM_WEIGHTED_SUPPORT_FRACTION:
            failures.append(f"{name}:support_fraction:{fraction:.12f}")
    volumes = load_node_volumes(
        verify_record(node_volumes_record, label=f"{unit}/node_volumes")
    )
    expected_inverse = 2.0 * count / (
        volumes[:, None] + volumes[None, :]
    )
    np.fill_diagonal(expected_inverse, 0.0)
    formula_error = float(
        np.max(
            np.abs(
                matrices["count_invnodevol"] - expected_inverse
            )
        )
    )
    if not np.allclose(
        matrices["count_invnodevol"],
        expected_inverse,
        atol=1.0e-8,
        rtol=1.0e-7,
    ):
        failures.append("count_invnodevol:formula_differs")
    if endpoint_assignment_fraction < MINIMUM_ENDPOINT_ASSIGNMENT_FRACTION:
        failures.append(
            f"endpoint_assignment:{endpoint_assignment_fraction:.12f}"
        )
    return {
        "status": "PASS" if not failures else "FAIL",
        "unit": unit,
        "edge_density": density,
        "supported_edges": supported,
        "possible_edges": possible,
        "connected_nodes": connected_nodes,
        "hub_strengths": hub_strengths,
        "endpoint_assignment_fraction": endpoint_assignment_fraction,
        "support_agreement": support_agreement,
        "count_invnodevol_maximum_formula_error": formula_error,
        "failures": failures,
    }


def canary_records(
    topology: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source_lanes = {
        str(row["unit"]): str(row["source_recovery_lane"])
        for row in topology["canaries"]
    }
    cohort = selection["canary_cohort"]
    records = []
    for row in cohort["units"]:
        unit = str(row["unit"])
        summary_path = verify_record(
            row["summary"], label=f"{unit}/canary summary"
        )
        summary = load_json(summary_path)
        view = summary.get("views", {}).get("balanced", {})
        atlas_result_path = verify_record(
            summary["hroi_atlas_result"],
            label=f"{unit}/HROI atlas result",
        )
        atlas_result = load_json(atlas_result_path)
        node_volumes = atlas_result.get("artifacts", {}).get(
            "hcp379_node_volumes"
        )
        if (
            unit not in source_lanes
            or summary.get("status")
            != "PASS_HROI_BALANCED_ALL_NINE_CANARY"
            or summary.get("unit") != unit
            or summary.get("selected_balanced_total_streamlines")
            != selection["selected_total"]
            or summary.get("selected_per_seed_prefix_streamlines")
            != selection["per_seed"]
            or summary.get("combined_density_target_met") is not True
            or summary.get("all_views_technical_qc_pass") is not True
            or summary.get("seed_reliability", {}).get("status") != "PASS"
            or view.get("technical_status") != "PASS"
            or set(view.get("matrices", {})) != set(MATRIX_NAMES)
            or not isinstance(node_volumes, Mapping)
            or view.get("sift2", {}).get("weight_count")
            != selection["selected_total"]
        ):
            raise ValueError(f"{unit}: canary scientific contract differs")
        records.append(
            {
                "unit": unit,
                "cohort_role": "CALIBRATION_CANARY_REUSE",
                "source_recovery_lane": source_lanes[unit],
                "selected_total": selection["selected_total"],
                "per_seed": selection["per_seed"],
                "matrix_records": view["matrices"],
                "node_volumes": node_volumes,
                "assignments": view["assignments"],
                "sift2_weights": view["sift2"]["weights"],
                "endpoint_assignment_fraction": float(
                    view["endpoint_assignment_fraction"]
                ),
                "source_summary": file_record(summary_path),
            }
        )
    return records


def production_records(
    topology: Mapping[str, Any],
    production: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    lanes = {
        str(row["unit"]): str(row["lane"])
        for row in topology["production"]
    }
    records = []
    for row in production["state"]["subjects"]:
        unit = str(row["unit"])
        summary_path = verify_record(
            row["summary"], label=f"{unit}/production summary"
        )
        compaction_path = verify_record(
            row["compaction"], label=f"{unit}/production compaction"
        )
        summary = load_json(summary_path)
        compaction = load_json(compaction_path)
        view = summary.get("balanced_view", {})
        preflight_path = verify_record(
            summary["preflight"], label=f"{unit}/production preflight"
        )
        preflight = load_json(preflight_path)
        node_volumes = preflight.get("model_inputs", {}).get(
            "node_volumes"
        )
        if (
            unit not in lanes
            or summary.get("status") != "PASS_BALANCED_PRODUCTION"
            or summary.get("unit") != unit
            or summary.get("selected_balanced_total_streamlines")
            != selection["selected_total"]
            or summary.get("selected_per_seed_streamlines")
            != selection["per_seed"]
            or summary.get("combined_density_target_met") is not True
            or summary.get("technical_qc_pass") is not True
            or view.get("technical_status") != "PASS"
            or set(view.get("matrices", {})) != set(MATRIX_NAMES)
            or not isinstance(node_volumes, Mapping)
            or view.get("sift2", {}).get("weight_count")
            != selection["selected_total"]
            or compaction.get("status")
            != "PASS_COMPACTED_GENERATED_TRACKS"
            or compaction.get("retained_scientific_summary")
            != file_record(summary_path)
            or any(
                Path(str(record.get("path", ""))).exists()
                for record in compaction.get(
                    "generated_tractograms_deleted", {}
                ).values()
            )
        ):
            raise ValueError(
                f"{unit}: balanced production scientific contract differs"
            )
        records.append(
            {
                "unit": unit,
                "cohort_role": "BALANCED_PRODUCTION",
                "source_recovery_lane": lanes[unit],
                "selected_total": selection["selected_total"],
                "per_seed": selection["per_seed"],
                "matrix_records": view["matrices"],
                "node_volumes": node_volumes,
                "assignments": view["assignments"],
                "sift2_weights": view["sift2"]["weights"],
                "endpoint_assignment_fraction": float(
                    view["endpoint_assignment_fraction"]
                ),
                "source_summary": file_record(summary_path),
                "compaction": file_record(compaction_path),
            }
        )
    return records


def assemble(
    readiness: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if readiness.get("status") != "READY_TO_ASSEMBLE":
        raise RuntimeError("balanced integration release is not ready")
    if CURRENT.exists():
        raise FileExistsError(
            f"immutable balanced release already exists: {CURRENT}"
        )
    topology = evidence["topology"]["topology"]
    selection = evidence["selection"]
    records = canary_records(topology, selection) + production_records(
        topology, evidence["production"], selection
    )
    records.sort(key=lambda row: row["unit"])
    if (
        len(records) != EXPECTED_SUBJECTS
        or len({row["unit"] for row in records}) != EXPECTED_SUBJECTS
    ):
        raise ValueError("15+515 resolved record set is not exact 530")
    route_counts = Counter(
        row["source_recovery_lane"] for row in records
    )
    if dict(route_counts) != EXPECTED_ROUTE_COUNTS:
        raise ValueError(
            f"source recovery route counts differ: {dict(route_counts)}"
        )
    attempt = ATTEMPTS / f"{stamp()}-balanced-integration-release"
    attempt.mkdir(parents=True, exist_ok=False)
    bound = {
        **dict(readiness),
        "status": "ASSEMBLY_INPUTS_BOUND",
        "implementation": file_record(Path(__file__)),
        "topology": evidence["topology"]["record"],
        "topology_validation": evidence["topology"]["validation"],
        "selection": selection["selection"],
        "selection_validation": selection["validation"],
        "canary_cohort_summary": selection[
            "canary_cohort_summary"
        ],
        "canary_production_equivalence": evidence["equivalence"][
            "validation"
        ],
        "archive_primary_compatibility": evidence[
            "archive_primary_compatibility"
        ]["audit"],
        "archive_primary_compatibility_validation": evidence[
            "archive_primary_compatibility"
        ]["validation"],
        "human_review": evidence["human_review"]["review"],
        "human_review_manifest": evidence["human_review"]["manifest"],
        "production_state": evidence["production"]["record"],
    }
    atomic_json(attempt / "bound_inputs.json", bound)
    subject_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        unit = record["unit"]
        verify_record(
            record["assignments"], label=f"{unit}/assignments"
        )
        verify_record(
            record["sift2_weights"], label=f"{unit}/SIFT2 weights"
        )
        qc = matrix_qc(
            unit=unit,
            matrix_records=record["matrix_records"],
            node_volumes_record=record["node_volumes"],
            endpoint_assignment_fraction=record[
                "endpoint_assignment_fraction"
            ],
        )
        if qc["status"] != "PASS":
            failures.append({"unit": unit, "failures": qc["failures"]})
        subject_rows.append(
            {
                "unit": unit,
                "cohort_role": record["cohort_role"],
                "source_recovery_lane": record[
                    "source_recovery_lane"
                ],
                "selected_balanced_total_streamlines": record[
                    "selected_total"
                ],
                "selected_per_seed_streamlines": record["per_seed"],
                "edge_density": f"{qc['edge_density']:.12g}",
                "connected_nodes": qc["connected_nodes"],
                "endpoint_assignment_fraction": (
                    f"{qc['endpoint_assignment_fraction']:.12g}"
                ),
                "node_volumes_path": record["node_volumes"]["path"],
                "node_volumes_sha256": record["node_volumes"]["sha256"],
                "assignments_path": record["assignments"]["path"],
                "assignments_sha256": record["assignments"]["sha256"],
                "sift2_weights_path": record["sift2_weights"]["path"],
                "sift2_weights_sha256": record[
                    "sift2_weights"
                ]["sha256"],
                "source_summary_path": record["source_summary"]["path"],
                "source_summary_sha256": record[
                    "source_summary"
                ]["sha256"],
                "compaction_path": record.get("compaction", {}).get(
                    "path", ""
                ),
                "compaction_sha256": record.get("compaction", {}).get(
                    "sha256", ""
                ),
                **{
                    f"matrix_{name}_path": record[
                        "matrix_records"
                    ][name]["path"]
                    for name in MATRIX_NAMES
                },
                **{
                    f"matrix_{name}_sha256": record[
                        "matrix_records"
                    ][name]["sha256"]
                    for name in MATRIX_NAMES
                },
            }
        )
        qc_rows.append(
            {
                "unit": unit,
                "status": qc["status"],
                "cohort_role": record["cohort_role"],
                "source_recovery_lane": record[
                    "source_recovery_lane"
                ],
                "edge_density": f"{qc['edge_density']:.12g}",
                "supported_edges": qc["supported_edges"],
                "connected_nodes": qc["connected_nodes"],
                "endpoint_assignment_fraction": (
                    f"{qc['endpoint_assignment_fraction']:.12g}"
                ),
                "minimum_weighted_count_support_fraction": (
                    f"{min(row['count_edge_support_fraction'] for row in qc['support_agreement'].values()):.12g}"
                ),
                "maximum_count_invnodevol_formula_error": (
                    f"{qc['count_invnodevol_maximum_formula_error']:.12g}"
                ),
                "failure_count": len(qc["failures"]),
                "failures": "|".join(qc["failures"]),
            }
        )
        for name in MATRIX_NAMES:
            matrix_rows.append(
                {
                    "unit": unit,
                    "matrix": name,
                    "path": record["matrix_records"][name]["path"],
                    "size_bytes": record["matrix_records"][name][
                        "size_bytes"
                    ],
                    "sha256": record["matrix_records"][name]["sha256"],
                    "cohort_role": record["cohort_role"],
                    "source_recovery_lane": record[
                        "source_recovery_lane"
                    ],
                }
            )
        print(
            f"[{utc_now()}] balanced-release-qc "
            f"{index}/{EXPECTED_SUBJECTS} {unit} {qc['status']}",
            flush=True,
        )
    if (
        failures
        or len(subject_rows) != EXPECTED_SUBJECTS
        or len(matrix_rows) != EXPECTED_MATRIX_FILES
    ):
        atomic_json(
            attempt / "terminal_failure.json",
            {
                "status": "FAIL_MATRIX_QC",
                "generated_utc": utc_now(),
                "subject_n": len(subject_rows),
                "matrix_file_n": len(matrix_rows),
                "failures": failures,
            },
        )
        raise ValueError(
            f"balanced release matrix QC failed for {len(failures)} subjects"
        )
    atomic_csv(attempt / "analysis_ready_manifest.csv", subject_rows)
    atomic_csv(attempt / "subject_qc.tsv", qc_rows, delimiter="\t")
    atomic_csv(attempt / "matrix_files.tsv", matrix_rows, delimiter="\t")
    atomic_json(
        attempt / "dataset_description.json",
        {
            "Name": "HCP379-v2 balanced structural connectome release",
            "BIDSVersion": "1.10.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": (
                        "HCP379 Recovery4 balanced two-seed ACT and "
                        "final-HROI matrix pipeline"
                    ),
                    "Version": "1",
                }
            ],
            "Description": (
                "Diagnosis-blind 379-node structural connectome matrix "
                "manifest with nine matched matrix families."
            ),
        },
    )
    atomic_text(
        attempt / "README.md",
        "# HCP379-v2 balanced integration release\n\n"
        "This immutable diagnosis-blind release indexes 530 scan-level "
        "units: 15 validated calibration canaries and 515 production "
        "subjects. Every unit uses one uniform balanced two-seed recipe and "
        "the same final HROI atlas policy. Nine matched 379 x 379 matrices "
        "are retained: count, SIFT2-weighted fibre-density sum, inverse-node-"
        "volume count, mean length, mean inverse length, and FA/MD/RD/AxD. "
        "All count matrices independently pass density >= 0.60 and the "
        "technical QC in subject_qc.tsv. Source-recovery lane and cohort role "
        "must remain covariates/sensitivity dimensions in downstream work. "
        "No diagnosis or outcome field is present; outcomes are attached only "
        "after this release passes independent verification.\n",
    )
    densities = [float(row["edge_density"]) for row in qc_rows]
    manifest = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_balanced_integration_release"
        ),
        "status": "PASS_ASSEMBLED_AWAITING_INDEPENDENT_VERIFICATION",
        "generated_utc": utc_now(),
        "diagnosis_or_outcome_fields_present": False,
        "non_overwriting": True,
        "source_files_modified": False,
        "subject_count": len(subject_rows),
        "calibration_canary_count": EXPECTED_CANARIES,
        "production_subject_count": EXPECTED_PRODUCTION,
        "matrix_family_count": len(MATRIX_NAMES),
        "matrix_file_count": len(matrix_rows),
        "matrix_shape": [EXPECTED_NODES, EXPECTED_NODES],
        "matrix_names": list(MATRIX_NAMES),
        "selected_balanced_total_streamlines": selection[
            "selected_total"
        ],
        "selected_per_seed_streamlines": selection["per_seed"],
        "cohort_uniform": True,
        "per_subject_streamline_tuning_allowed": False,
        "subject_exclusion_for_low_density_allowed": False,
        "density_is_whole_cohort_technical_gate": True,
        "minimum_required_edge_density": DENSITY_TARGET,
        "minimum_observed_edge_density": min(densities),
        "maximum_observed_edge_density": max(densities),
        "subjects_at_or_above_density_target": sum(
            value >= DENSITY_TARGET for value in densities
        ),
        "source_recovery_route_counts": dict(sorted(route_counts.items())),
        "source_recovery_route_sensitivity_required": True,
        "final_release_authorized": False,
        "requires_independent_verification": True,
        "records": {
            "bound_inputs": file_record(attempt / "bound_inputs.json"),
            "analysis_ready_manifest": file_record(
                attempt / "analysis_ready_manifest.csv"
            ),
            "subject_qc": file_record(attempt / "subject_qc.tsv"),
            "matrix_files": file_record(attempt / "matrix_files.tsv"),
            "dataset_description": file_record(
                attempt / "dataset_description.json"
            ),
            "readme": file_record(attempt / "README.md"),
            "canary_production_equivalence": evidence[
                "equivalence"
            ]["validation"],
            "archive_primary_compatibility": evidence[
                "archive_primary_compatibility"
            ]["audit"],
            "archive_primary_compatibility_validation": evidence[
                "archive_primary_compatibility"
            ]["validation"],
        },
    }
    atomic_json(attempt / "release_manifest.json", manifest)
    pointer = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_balanced_integration_release_pointer"
        ),
        "status": "PASS_ASSEMBLED_AWAITING_INDEPENDENT_VERIFICATION",
        "created_utc": utc_now(),
        "non_overwriting": True,
        "release_manifest": file_record(attempt / "release_manifest.json"),
    }
    if CURRENT.exists():
        raise FileExistsError(CURRENT)
    atomic_json(CURRENT, pointer)
    return {
        "status": pointer["status"],
        "attempt_root": str(attempt.resolve()),
        "subject_count": len(subject_rows),
        "matrix_file_count": len(matrix_rows),
        "minimum_observed_edge_density": min(densities),
        "current_release": str(CURRENT.resolve()),
        "final_release_authorized": False,
    }


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="hcp379-balanced-release-test."
    ) as temporary:
        root = Path(temporary)
        volumes_path = root / "volumes.csv"
        volume_rows = [
            {"node_id": index, "volume_mm3": float(index + 10)}
            for index in range(1, EXPECTED_NODES + 1)
        ]
        atomic_csv(volumes_path, volume_rows)
        count = np.zeros((EXPECTED_NODES, EXPECTED_NODES), dtype=float)
        target = math.ceil(
            DENSITY_TARGET
            * EXPECTED_NODES
            * (EXPECTED_NODES - 1)
            / 2
        )
        upper = np.triu_indices(EXPECTED_NODES, 1)
        count[upper[0][:target], upper[1][:target]] = 1
        count = count + count.T
        volumes = load_node_volumes(volumes_path)
        inverse = 2.0 * count / (
            volumes[:, None] + volumes[None, :]
        )
        np.fill_diagonal(inverse, 0.0)
        records = {}
        for name in MATRIX_NAMES:
            path = root / f"{name}.csv"
            value = inverse if name == "count_invnodevol" else count
            np.savetxt(path, value, delimiter=",", fmt="%.12g")
            records[name] = file_record(path)
        passed = matrix_qc(
            unit="synthetic",
            matrix_records=records,
            node_volumes_record=file_record(volumes_path),
            endpoint_assignment_fraction=0.90,
        )
        sparse = np.zeros_like(count)
        for index in range(EXPECTED_NODES - 1):
            sparse[index, index + 1] = sparse[index + 1, index] = 1
        sparse_path = root / "sparse_count.csv"
        np.savetxt(sparse_path, sparse, delimiter=",", fmt="%.12g")
        sparse_records = dict(records)
        sparse_records["count"] = file_record(sparse_path)
        failed = matrix_qc(
            unit="synthetic_sparse",
            matrix_records=sparse_records,
            node_volumes_record=file_record(volumes_path),
            endpoint_assignment_fraction=0.90,
        )
    checks = {
        "exact_subject_partition": (
            EXPECTED_CANARIES + EXPECTED_PRODUCTION == EXPECTED_SUBJECTS
        ),
        "exact_matrix_file_count": (
            EXPECTED_SUBJECTS * len(MATRIX_NAMES)
            == EXPECTED_MATRIX_FILES
        ),
        "route_counts_sum_to_530": (
            sum(EXPECTED_ROUTE_COUNTS.values()) == EXPECTED_SUBJECTS
        ),
        "synthetic_valid_matrix_family_passes": passed["status"] == "PASS",
        "synthetic_sparse_or_inconsistent_family_fails": (
            failed["status"] == "FAIL"
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        value = self_test()
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0 if value["status"] == "PASS" else 1
    readiness, evidence = current_readiness()
    atomic_json(READINESS, readiness)
    if args.assemble:
        value = assemble(readiness, evidence)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    print(json.dumps(readiness, indent=2, sort_keys=True))
    return int(args.require_ready and readiness["status"] != "READY_TO_ASSEMBLE")


if __name__ == "__main__":
    raise SystemExit(main())
