#!/usr/bin/env python3
"""Audit and assemble the diagnosis-blind HCP379-v2 reference release.

The default mode writes a current readiness record and never fabricates a
release.  ``--assemble`` is fail closed: it requires exact 530-subject route
accounting, all route-calibration decisions, the corrected 227 release, the
isolated repair outcomes, and 4,770 rehashed 379x379 matrices.

The assembled release is a non-overwriting reference manifest over the
validated derivative files.  It does not copy or relabel source matrices and
does not attach diagnosis or research outcomes.  Assembly independently
recomputes count-matrix density and fails closed unless all 530 subjects meet
the prespecified whole-cohort technical density requirement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
ROUTE_ROOT = (
    EXP / "research_audit/outputs/hcp_rerun_triage_v1"
)
FREEZE = HCP_ROOT / "manifests/historical_freeze_manifest.csv"
TRIAGE = ROUTE_ROOT / "hcp_rerun_triage_manifest.csv"
PLAN = (
    EXP
    / "research_audit/outputs/hcp379_usable_cohort_plan_v1.md"
)
DENSITY_RECOVERY_PLAN = (
    EXP
    / "research_audit/outputs/hcp379_density_recovery_plan_v3/"
    "hcp379_density_recovery_plan.json"
)
DENSITY_EXECUTION_TOPOLOGY = (
    EXP
    / "research_audit/outputs/hcp379_density_execution_topology_v3/"
    "hcp379_density_execution_topology.json"
)
DENSITY_EXECUTION_SUBJECTS = (
    EXP
    / "research_audit/outputs/hcp379_density_execution_topology_v3/"
    "hcp379_density_execution_subjects.csv"
)
ARCHIVE_SUMMARY = HCP_ROOT / "manifests/archive_restore_summary.json"
TENSOR_SUMMARY = HCP_ROOT / "manifests/tensor_repair_summary.json"
FREESURFER_SOURCE_ATTESTATION = (
    HCP_ROOT
    / "fastsurfer_repair/114_S_6347_I1344943/"
    "hcp_source_recovery_attestation_v3.json"
)
PHASE_VALIDATION = Path(
    "/data/derivatives/scforge_v2/"
    "h04a_r1_recovery_20260719_retry4/publication/"
    "hcp379_recipe_stability_validation.json"
)
HUMAN_CANARY_QC = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
RECOVERY4_PRETRACT_MASTER = (
    HCP_ROOT / "pretract_recovery4/pretract_recovery4_manifest.json"
)
CORRECTED_227 = (
    HCP_ROOT
    / "corrected_scaleup_recovery4/manifests/"
    "corrected_227_release_manifest.json"
)
CORRECTED_LEGACY_TENSOR_216 = (
    HCP_ROOT
    / "corrected_legacy_tensor_recovery4/manifests/"
    "corrected_legacy_tensor_216_release_manifest.json"
)
CORRECTED_ARCHIVE_LOW_30 = (
    HCP_ROOT
    / "corrected_archive_low_recovery4/manifests/"
    "corrected_archive_low_30_release_manifest.json"
)
CORRECTED_FREESURFER_1 = (
    HCP_ROOT
    / "corrected_freesurfer_recovery4/manifests/"
    "corrected_freesurfer_1_release_manifest.json"
)
KEEP_CONCORDANCE = (
    HCP_ROOT / "calibration/keep_route_concordance.json"
)
ARCHIVE_CONCORDANCE = (
    HCP_ROOT / "calibration/archive_route_concordance.json"
)
CORRECTED_LEGACY_217 = (
    HCP_ROOT
    / "corrected_legacy_act/manifests/"
    "corrected_legacy_act_217_release_manifest.json"
)
CORRECTED_ARCHIVE_86 = (
    HCP_ROOT
    / "corrected_archive_recovery4/manifests/"
    "corrected_archive_86_release_manifest.json"
)
READINESS_OUTPUT = (
    HCP_ROOT / "release_candidate_v2/hcp379_release_readiness.json"
)
RELEASE_ROOT = HCP_ROOT / "release_candidate_v2"
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
EXPECTED_N = 530
EXPECTED_NODES = 379
EXPECTED_MATRICES = EXPECTED_N * len(MATRIX_NAMES)
HUB_IDS = (361, 365, 366, 370, 374, 375)
KEEP_RETAIN_DECISION = "RETAIN_REMAINING_210_HIGH_CONFIDENCE_CASES"
KEEP_RERUN_DECISION = (
    "RERUN_REMAINING_210_KEEP_PLUS_3_AUXILIARY_LEGACY_ACT_CASES"
)
ARCHIVE_RETAIN_DECISION = (
    "RETAIN_ARCHIVE_D0_56_AND_USE_CORRECTED_LOW_30_WITH_ROUTE_SENSITIVITY"
)
ARCHIVE_RERUN_DECISION = "RERUN_ARCHIVE_86_WITH_LOCKED_CORRECTED_ACT"
ALLOWED_STREAMLINE_COUNTS = (3_000_000, 5_000_000, 10_000_000)
MINIMUM_EDGE_DENSITY = 0.60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def load_json_optional(path: Path) -> dict[str, Any] | None:
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular file: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def validate_human_canary_qc() -> dict[str, Any]:
    master = load_json(RECOVERY4_PRETRACT_MASTER)
    units = {str(row["unit"]) for row in master.get("units", [])}
    if (
        master.get("record_type")
        != "diagnosis_blind_hcp379_pretract_recovery4_manifest"
        or master.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or master.get("diagnosis_labels_used") is not False
        or master.get("human_visual_qc_inferred") is not False
        or len(units) != 15
    ):
        raise ValueError("Recovery4 pretract master differs")
    with HUMAN_CANARY_QC.open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        forbidden = {
            "diagnosis",
            "diagnosis_at_dti",
            "group",
            "research_group",
            "outcome",
        }
        if forbidden.intersection(
            str(field).strip().lower() for field in fields
        ):
            raise ValueError("human canary QC is not diagnosis blind")
        required = {"unit", "status", "reviewer", "reviewed_utc"}
        if not required.issubset(fields):
            raise ValueError("human canary QC lacks required fields")
        rows = [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if len(rows) != 15 or {row["unit"] for row in rows} != units:
        raise ValueError("human canary QC unit set differs")
    for row in rows:
        if (
            row["status"].upper() != "PASS"
            or not row["reviewer"]
            or not row["reviewed_utc"]
        ):
            raise ValueError(
                f"human canary QC is incomplete: {row['unit']}"
            )
        text = row["reviewed_utc"]
        reviewed_at = datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
        if reviewed_at.tzinfo is None:
            raise ValueError(
                f"human canary QC timestamp lacks timezone: {row['unit']}"
            )
        if reviewed_at > datetime.now(timezone.utc):
            raise ValueError(
                f"human canary QC timestamp is future-dated: {row['unit']}"
            )
    return file_record(HUMAN_CANARY_QC)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        newline="",
        encoding="utf-8",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t" if path.suffix.lower() == ".tsv" else ",",
        )
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def unit_set(path: Path, expected: int) -> set[str]:
    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(values) != expected:
        raise ValueError(f"{path.name} has {len(values)}, not {expected}")
    return values


def route_identity() -> dict[str, set[str]]:
    routes = {
        "keep": unit_set(
            ROUTE_ROOT / "keep_high_confidence_214.txt", 214
        ),
        "archive": unit_set(
            ROUTE_ROOT / "archive_recovery_86.txt", 86
        ),
        "remap": unit_set(
            ROUTE_ROOT / "hcp_remap_probe_111.txt", 111
        ),
        "direct_corrected": unit_set(
            ROUTE_ROOT / "corrected_track_rebuild_116.txt", 116
        ),
        "tensor": unit_set(
            ROUTE_ROOT / "tensor_weighted_repair_2.txt", 2
        ),
        "freesurfer": unit_set(
            ROUTE_ROOT / "fastsurfer_retry_1.txt", 1
        ),
    }
    values = list(routes.values())
    overlap = set()
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            overlap |= left & right
    union = set().union(*values)
    freeze_units = {row["fsid"] for row in read_csv(FREEZE)}
    if (
        overlap
        or len(union) != EXPECTED_N
        or union != freeze_units
    ):
        raise ValueError(
            f"route identity differs: union={len(union)} "
            f"overlap={len(overlap)} freeze={len(freeze_units)}"
        )
    return routes


def density_execution_lane_units() -> dict[str, set[str]]:
    rows = read_csv(DENSITY_EXECUTION_SUBJECTS)
    if len(rows) != EXPECTED_N:
        raise ValueError("density execution topology is not exact 530")
    lanes: dict[str, set[str]] = {}
    for row in rows:
        if str(row.get("diagnosis_or_outcomes_used", "")).lower() != "false":
            raise ValueError("density execution topology is not blinded")
        lane = str(row.get("execution_lane", ""))
        unit = str(row.get("unit", ""))
        if not lane or not unit:
            raise ValueError("density execution topology identity is blank")
        lanes.setdefault(lane, set()).add(unit)
    expected = {
        "corrected_core_227": 227,
        "corrected_legacy_tensor_216": 216,
        "corrected_archive_low_30": 30,
        "conditional_archive_D0_56": 56,
        "corrected_freesurfer_1": 1,
    }
    if {name: len(units) for name, units in lanes.items()} != expected:
        raise ValueError("density execution lane counts differ")
    values = list(lanes.values())
    if (
        len(set().union(*values)) != EXPECTED_N
        or any(
            left & right
            for index, left in enumerate(values)
            for right in values[index + 1 :]
        )
    ):
        raise ValueError("density execution lanes overlap or omit subjects")
    return lanes


def record_from_path_hash(
    path: Path, expected_sha256: str
) -> dict[str, Any]:
    record = file_record(path)
    if record["sha256"] != expected_sha256:
        raise ValueError(f"hash differs: {path}")
    return record


def artifacts_to_matrices(
    unit: str, artifacts: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    matrices = {}
    for name in MATRIX_NAMES:
        raw = artifacts.get(f"matrix_{name}")
        if not isinstance(raw, Mapping):
            raise ValueError(f"{unit}:matrix artifact absent: {name}")
        record = dict(raw)
        path = Path(str(record.get("path", "")))
        if record != file_record(path):
            raise ValueError(f"{unit}:matrix artifact differs: {name}")
        matrices[name] = record
    return matrices


def corrected_manifest_records(
    path: Path, expected: int, route: str
) -> dict[str, dict[str, Any]]:
    manifest = load_json(path)
    rows = manifest.get("units")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("diagnosis_labels_used") is not False
        or manifest.get("unit_count") != expected
        or manifest.get(
            "density_is_whole_cohort_technical_release_gate"
        )
        is not True
        or manifest.get("minimum_edge_density_inclusive")
        != MINIMUM_EDGE_DENSITY
        or manifest.get("subjects_at_or_above_minimum_density")
        != expected
        or float(manifest.get("minimum_observed_edge_density", -1.0))
        < MINIMUM_EDGE_DENSITY
        or not isinstance(rows, list)
        or len(rows) != expected
    ):
        raise ValueError(f"corrected manifest is not {expected} PASS: {path}")
    output = {}
    source_record = file_record(path)
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(f"corrected unit is not a mapping: {path}")
        unit = str(row.get("unit", ""))
        artifacts = row.get("artifacts")
        density = float(row.get("edge_density", -1.0))
        if (
            not unit
            or unit in output
            or not isinstance(artifacts, Mapping)
            or density < MINIMUM_EDGE_DENSITY
        ):
            raise ValueError(f"corrected unit identity differs: {unit}")
        output[unit] = {
            "unit": unit,
            "source_route": route,
            "matrices": artifacts_to_matrices(unit, artifacts),
            "endpoint_assignment_fraction": row.get(
                "endpoint_assignment_fraction"
            ),
            "edge_density": density,
            "source_record": source_record,
        }
    return output


def freeze_by_unit() -> dict[str, dict[str, str]]:
    rows = read_csv(FREEZE)
    output = {row["fsid"]: row for row in rows}
    if len(rows) != EXPECTED_N or len(output) != EXPECTED_N:
        raise ValueError("freeze identity differs")
    return output


def legacy_records(
    units: set[str], route: str
) -> dict[str, dict[str, Any]]:
    freeze = freeze_by_unit()
    output = {}
    source_record = file_record(FREEZE)
    for unit in sorted(units):
        row = freeze[unit]
        matrices = {}
        for name in MATRIX_NAMES:
            path = Path(row[f"matrix_{name}_path"])
            expected_hash = row[f"matrix_{name}_sha256"]
            matrices[name] = record_from_path_hash(path, expected_hash)
        output[unit] = {
            "unit": unit,
            "source_route": route,
            "matrices": matrices,
            "source_record": source_record,
        }
    return output


def archive_records(
    units: set[str],
) -> dict[str, dict[str, Any]]:
    summary = load_json(ARCHIVE_SUMMARY)
    rows = summary.get("subjects")
    if (
        summary.get("status_counts") != {"PASS_ALL_NINE": 86}
        or not isinstance(rows, list)
        or len(rows) != 86
    ):
        raise ValueError("archive summary is not 86/86 PASS")
    by_unit = {
        str(row.get("fsid", "")): row
        for row in rows
        if isinstance(row, Mapping)
    }
    all_archive_units = unit_set(
        ROUTE_ROOT / "archive_recovery_86.txt", 86
    )
    if set(by_unit) != all_archive_units or not units.issubset(
        all_archive_units
    ):
        raise ValueError("archive summary identity differs")
    output = {}
    source_record = file_record(ARCHIVE_SUMMARY)
    for unit in sorted(units):
        row = by_unit[unit]
        if row.get("status") != "PASS_ALL_NINE":
            raise ValueError(f"{unit}:archive state is not PASS")
        hashes = row.get("matrix_sha256")
        if not isinstance(hashes, Mapping):
            raise ValueError(f"{unit}:archive hashes absent")
        matrices = {}
        for name in MATRIX_NAMES:
            path = (
                HCP_ROOT
                / "connectomes"
                / f"SC_HCPMMP1_{unit}_{name}.csv"
            )
            matrices[name] = record_from_path_hash(
                path, str(hashes[name])
            )
        output[unit] = {
            "unit": unit,
            "source_route": "ARCHIVE_NO_ACT_CALIBRATED",
            "matrices": matrices,
            "endpoint_assignment_fraction": row.get(
                "assignment_fraction"
            ),
            "edge_density": row.get("density"),
            "source_record": source_record,
        }
    return output


def tensor_records(
    units: set[str],
) -> dict[str, dict[str, Any]]:
    summary = load_json(TENSOR_SUMMARY)
    rows = summary.get("subjects")
    if (
        summary.get("pass") != 2
        or summary.get("fail") != 0
        or not isinstance(rows, list)
        or len(rows) != 2
    ):
        raise ValueError("tensor summary is not 2/2 PASS")
    by_unit = {
        str(row.get("fsid", "")): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if set(by_unit) != units:
        raise ValueError("tensor repair identity differs")
    output = {}
    source_record = file_record(TENSOR_SUMMARY)
    for unit in sorted(units):
        row = by_unit[unit]
        if row.get("status") != "PASS_ALL_NINE":
            raise ValueError(f"{unit}:tensor repair is not PASS")
        hashes = row.get("matrix_sha256")
        prefix = str(row.get("output_prefix", ""))
        if not isinstance(hashes, Mapping) or not prefix:
            raise ValueError(f"{unit}:tensor repair artifacts differ")
        matrices = {
            name: record_from_path_hash(
                Path(f"{prefix}_{name}.csv"), str(hashes[name])
            )
            for name in MATRIX_NAMES
        }
        output[unit] = {
            "unit": unit,
            "source_route": "LEGACY_ACT_TENSOR_REPAIR",
            "matrices": matrices,
            "source_record": source_record,
        }
    return output


def matrix_qc(
    unit: str, records: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    matrices = {}
    failures = []
    for name in MATRIX_NAMES:
        record = records[name]
        path = Path(str(record["path"]))
        if dict(record) != file_record(path):
            failures.append(f"{name}:hash")
            continue
        try:
            matrix = np.loadtxt(path, delimiter=",")
        except Exception as exc:
            failures.append(f"{name}:read:{type(exc).__name__}")
            continue
        matrices[name] = matrix
        if matrix.shape != (EXPECTED_NODES, EXPECTED_NODES):
            failures.append(f"{name}:shape")
            continue
        if not np.isfinite(matrix).all():
            failures.append(f"{name}:nonfinite")
        if not np.allclose(matrix, matrix.T, atol=1e-6, rtol=1e-6):
            failures.append(f"{name}:asymmetric")
        if not np.allclose(np.diag(matrix), 0.0, atol=1e-8):
            failures.append(f"{name}:diagonal")
        if float(np.min(matrix)) < -1e-10:
            failures.append(f"{name}:negative")
    if set(matrices) != set(MATRIX_NAMES):
        return {"status": "FAIL", "failures": sorted(set(failures))}
    count = matrices["count"]
    if not np.allclose(count, np.rint(count), atol=1e-6, rtol=0):
        failures.append("count:not_integer")
    support = count > 0
    np.fill_diagonal(support, False)
    strengths = np.sum(count, axis=1)
    connected = int(np.count_nonzero(strengths > 0))
    if connected < 360:
        failures.append(f"connected_nodes={connected}")
    for hub in HUB_IDS:
        if strengths[hub - 1] <= 0:
            failures.append(f"hub_{hub}:disconnected")
    supported_edges = int(np.count_nonzero(np.triu(support, 1)))
    possible = EXPECTED_NODES * (EXPECTED_NODES - 1) // 2
    edge_density = supported_edges / possible
    if edge_density < MINIMUM_EDGE_DENSITY:
        failures.append(
            f"edge_density={edge_density:.12f}"
            f"<{MINIMUM_EDGE_DENSITY:.12f}"
        )
    weighted_support = {}
    for name in (
        "fd_sum",
        "count_invnodevol",
        "len_mean",
        "invlen_mean",
        "fa_mean",
        "md_mean",
        "rd_mean",
        "ad_mean",
    ):
        nonzero = matrices[name] != 0
        value = (
            float(np.count_nonzero(np.triu(nonzero & support, 1)))
            / supported_edges
            if supported_edges
            else 0.0
        )
        weighted_support[name] = value
        if value < 0.95:
            failures.append(f"{name}:support={value:.6f}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "unit": unit,
        "connected_nodes": connected,
        "edge_density": edge_density,
        "supported_edges": supported_edges,
        "weighted_support": weighted_support,
        "failures": sorted(set(failures)),
    }


def current_readiness() -> tuple[dict[str, Any], dict[str, set[str]]]:
    routes = route_identity()
    blockers = []
    archive = load_json_optional(ARCHIVE_SUMMARY)
    freesurfer = load_json_optional(FREESURFER_SOURCE_ATTESTATION)
    phase = load_json_optional(PHASE_VALIDATION)
    corrected = load_json_optional(CORRECTED_227)
    corrected_legacy = load_json_optional(CORRECTED_LEGACY_TENSOR_216)
    corrected_archive_low = load_json_optional(CORRECTED_ARCHIVE_LOW_30)
    corrected_freesurfer = load_json_optional(CORRECTED_FREESURFER_1)
    archive_concordance = load_json_optional(ARCHIVE_CONCORDANCE)
    density_plan = load_json_optional(DENSITY_RECOVERY_PLAN)
    density_topology = load_json_optional(DENSITY_EXECUTION_TOPOLOGY)
    human_record = None
    phase_selected = None
    density_plan_valid = bool(
        density_plan
        and density_plan.get("record_type")
        == "diagnosis_blind_hcp379_density_recovery_plan"
        and density_plan.get("status") == "PLAN_READY_EXECUTION_PENDING"
        and density_plan.get("diagnosis_or_outcomes_used") is False
        and isinstance(
            density_plan.get("goal_acceptance_contract"), Mapping
        )
        and density_plan["goal_acceptance_contract"].get("target_subjects")
        == EXPECTED_N
        and density_plan["goal_acceptance_contract"].get(
            "required_final_subjects_at_or_above_threshold"
        )
        == EXPECTED_N
        and density_plan["goal_acceptance_contract"].get(
            "minimum_density_inclusive"
        )
        == MINIMUM_EDGE_DENSITY
        and density_plan["goal_acceptance_contract"].get(
            "density_is_whole_cohort_technical_release_gate"
        )
        is True
        and density_plan["goal_acceptance_contract"].get(
            "subject_exclusion_to_meet_density_allowed"
        )
        is False
        and density_plan["goal_acceptance_contract"].get(
            "edge_imputation_or_artificial_filling_allowed"
        )
        is False
    )
    if not density_plan_valid:
        blockers.append("density_recovery_contract_not_pass")
    density_topology_valid = bool(
        density_topology
        and density_topology.get("record_type")
        == "diagnosis_blind_hcp379_density_execution_topology"
        and density_topology.get("status")
        == "EXECUTION_TOPOLOGY_READY_GATED"
        and density_topology.get("diagnosis_or_outcomes_used") is False
        and density_topology.get("goal_acceptance", {}).get("target_n")
        == EXPECTED_N
        and density_topology.get("goal_acceptance", {}).get(
            "required_pass_n"
        )
        == EXPECTED_N
        and density_topology.get("goal_acceptance", {}).get(
            "minimum_density_inclusive"
        )
        == MINIMUM_EDGE_DENSITY
        and density_topology.get("release_scenarios", {}).get(
            "archive_concordance_pass", {}
        ).get("total_n")
        == EXPECTED_N
        and density_topology.get("release_scenarios", {}).get(
            "archive_concordance_fail", {}
        ).get("total_n")
        == EXPECTED_N
    )
    if not density_topology_valid:
        blockers.append("density_execution_topology_not_pass")
    if not archive or archive.get("status_counts") != {"PASS_ALL_NINE": 86}:
        blockers.append("archive_86_not_pass")
    if (
        not freesurfer
        or freesurfer.get("record_type")
        != (
            "diagnosis_blind_hcp379_freesurfer_source_"
            "recovery_attestation"
        )
        or freesurfer.get("status")
        != "PASS_HCP_SOURCE_READY_FOR_CORRECTED_ROUTE"
        or freesurfer.get("unit") != "114_S_6347_I1344943"
        or freesurfer.get("diagnosis_or_outcome_fields_used") is not False
        or freesurfer.get("atlas_nodes_present") != EXPECTED_NODES
        or freesurfer.get("exact_label_set_0_through_379") is not True
        or freesurfer.get("old_existing_track_is_final_release_candidate")
        is not False
    ):
        blockers.append("freesurfer_hcp_source_not_ready")
    phase_valid = bool(
        phase
        and phase.get("record_type")
        == "diagnosis_blind_hcp379_phase_b_recipe_stability_validation"
        and phase.get("status") == "PASS"
        and phase.get("diagnosis_labels_used") is False
        and phase.get("unit_count") == 15
        and phase.get(
            "smallest_density_qualified_scale_up_streamline_count"
        )
        in ALLOWED_STREAMLINE_COUNTS
        and isinstance(phase.get("manifest"), Mapping)
    )
    if not phase_valid:
        blockers.append("phase_b_recipe_stability_not_pass")
    else:
        phase_selected = int(
            phase[
                "smallest_density_qualified_scale_up_streamline_count"
            ]
        )
    try:
        human_record = validate_human_canary_qc()
    except (OSError, ValueError, TypeError):
        blockers.append("real_canary_human_visual_qc_missing")
    def corrected_valid(
        manifest: Mapping[str, Any] | None,
        *,
        record_type: str,
        expected_n: int,
    ) -> bool:
        return bool(
            manifest
            and manifest.get("record_type") == record_type
            and manifest.get("status") == "PASS"
            and manifest.get("diagnosis_labels_used") is False
            and manifest.get("unit_count") == expected_n
            and phase_selected is not None
            and manifest.get("selected_streamline_count") == phase_selected
            and manifest.get(
                "density_is_whole_cohort_technical_release_gate"
            )
            is True
            and manifest.get("minimum_edge_density_inclusive")
            == MINIMUM_EDGE_DENSITY
            and manifest.get("subjects_at_or_above_minimum_density")
            == expected_n
            and float(
                manifest.get("minimum_observed_edge_density", -1.0)
            )
            >= MINIMUM_EDGE_DENSITY
            and isinstance(manifest.get("units"), list)
            and len(manifest["units"]) == expected_n
            and len(
                {
                    str(row.get("unit"))
                    for row in manifest["units"]
                    if isinstance(row, Mapping)
                    and float(row.get("edge_density", -1.0))
                    >= MINIMUM_EDGE_DENSITY
                }
            )
            == expected_n
        )

    corrected_core_valid = corrected_valid(
        corrected,
        record_type=(
            "diagnosis_blind_hcp379_corrected_227_release_manifest"
        ),
        expected_n=227,
    )
    if not corrected_core_valid:
        blockers.append("corrected_227_release_not_pass")
    corrected_legacy_valid = corrected_valid(
        corrected_legacy,
        record_type=(
            "diagnosis_blind_hcp379_corrected_legacy_tensor_216_"
            "release_manifest"
        ),
        expected_n=216,
    )
    if not corrected_legacy_valid:
        blockers.append("corrected_legacy_tensor_216_release_not_pass")
    corrected_archive_low_valid = corrected_valid(
        corrected_archive_low,
        record_type=(
            "diagnosis_blind_hcp379_corrected_archive_low_30_"
            "release_manifest"
        ),
        expected_n=30,
    )
    if not corrected_archive_low_valid:
        blockers.append("corrected_archive_low_30_release_not_pass")
    corrected_freesurfer_valid = corrected_valid(
        corrected_freesurfer,
        record_type=(
            "diagnosis_blind_hcp379_corrected_freesurfer_1_"
            "release_manifest"
        ),
        expected_n=1,
    )
    if not corrected_freesurfer_valid:
        blockers.append("corrected_freesurfer_1_release_not_pass")
    archive_record_valid = bool(
        archive_concordance
        and archive_concordance.get("record_type")
        == (
            "diagnosis_blind_hcp379_archive_route_"
            "concordance_recovery4_validation"
        )
        and archive_concordance.get("diagnosis_labels_used") is False
        and archive_concordance.get("recovery4_inputs_required") is True
        and archive_concordance.get("calibration_n") == 6
        and archive_concordance.get("selected_streamline_count")
        in ALLOWED_STREAMLINE_COUNTS
    )
    archive_decision = (
        archive_concordance.get("decision")
        if archive_record_valid
        else None
    )
    if archive_decision == ARCHIVE_RETAIN_DECISION:
        if (
            archive_concordance.get("status") != "PASS"
            or not corrected_archive_low_valid
        ):
            blockers.append("archive_retain_decision_not_pass")
    elif archive_decision == ARCHIVE_RERUN_DECISION:
        rerun = load_json_optional(CORRECTED_ARCHIVE_86)
        if (
            not rerun
            or rerun.get("status") != "PASS"
            or rerun.get("unit_count") != 86
        ):
            blockers.append("corrected_archive_86_not_pass")
    else:
        blockers.append("archive_route_concordance_decision_missing")

    lane_presence = {
        "archive_pass_all_nine": (
            86
            if archive
            and archive.get("status_counts") == {"PASS_ALL_NINE": 86}
            else 0
        ),
        "corrected_227_pass_all_nine": (
            227 if corrected_core_valid else 0
        ),
        "corrected_legacy_tensor_216_pass_all_nine": (
            216 if corrected_legacy_valid else 0
        ),
        "corrected_archive_low_30_pass_all_nine": (
            30 if corrected_archive_low_valid else 0
        ),
        "corrected_freesurfer_1_pass_all_nine": (
            1 if corrected_freesurfer_valid else 0
        ),
    }
    report = {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_final_release_readiness",
        "status": "READY_TO_ASSEMBLE" if not blockers else "NOT_READY",
        "generated_utc": utc_now(),
        "diagnosis_or_outcome_fields_used_for_routing": False,
        "expected_subjects": EXPECTED_N,
        "expected_matrix_files": EXPECTED_MATRICES,
        "minimum_edge_density_inclusive": MINIMUM_EDGE_DENSITY,
        "required_subjects_at_or_above_minimum_density": EXPECTED_N,
        "density_is_whole_cohort_technical_release_gate": True,
        "density_is_subject_exclusion_gate": False,
        "route_identity_counts": {
            name: len(units) for name, units in routes.items()
        },
        "matrix_bundles_present_by_lane": lane_presence,
        "legacy_tensor_route_decision": (
            "CORRECTED_ACT_ALL_216"
        ),
        "archive_route_decision": archive_decision,
        "blockers": sorted(set(blockers)),
        "records": {
            "historical_freeze": (
                file_record(FREEZE) if FREEZE.is_file() else None
            ),
            "triage": file_record(TRIAGE) if TRIAGE.is_file() else None,
            "execution_plan": file_record(PLAN) if PLAN.is_file() else None,
            "density_recovery_plan": (
                file_record(DENSITY_RECOVERY_PLAN)
                if DENSITY_RECOVERY_PLAN.is_file()
                else None
            ),
            "density_execution_topology": (
                file_record(DENSITY_EXECUTION_TOPOLOGY)
                if DENSITY_EXECUTION_TOPOLOGY.is_file()
                else None
            ),
            "density_execution_subjects": (
                file_record(DENSITY_EXECUTION_SUBJECTS)
                if DENSITY_EXECUTION_SUBJECTS.is_file()
                else None
            ),
            "archive_summary": (
                file_record(ARCHIVE_SUMMARY)
                if ARCHIVE_SUMMARY.is_file()
                else None
            ),
            "tensor_summary": (
                file_record(TENSOR_SUMMARY)
                if TENSOR_SUMMARY.is_file()
                else None
            ),
            "freesurfer_hcp_source_attestation": (
                file_record(FREESURFER_SOURCE_ATTESTATION)
                if FREESURFER_SOURCE_ATTESTATION.is_file()
                else None
            ),
            "phase_b_validation": (
                file_record(PHASE_VALIDATION)
                if PHASE_VALIDATION.is_file()
                else None
            ),
            "recovery4_pretract_master": (
                file_record(RECOVERY4_PRETRACT_MASTER)
                if RECOVERY4_PRETRACT_MASTER.is_file()
                else None
            ),
            "human_canary_qc": human_record,
            "corrected_227_release": (
                file_record(CORRECTED_227)
                if CORRECTED_227.is_file()
                else None
            ),
            "corrected_legacy_tensor_216_release": (
                file_record(CORRECTED_LEGACY_TENSOR_216)
                if CORRECTED_LEGACY_TENSOR_216.is_file()
                else None
            ),
            "corrected_archive_low_30_release": (
                file_record(CORRECTED_ARCHIVE_LOW_30)
                if CORRECTED_ARCHIVE_LOW_30.is_file()
                else None
            ),
            "corrected_freesurfer_1_release": (
                file_record(CORRECTED_FREESURFER_1)
                if CORRECTED_FREESURFER_1.is_file()
                else None
            ),
            "archive_concordance": (
                file_record(ARCHIVE_CONCORDANCE)
                if ARCHIVE_CONCORDANCE.is_file()
                else None
            ),
        },
    }
    return report, routes


def resolved_records(
    routes: Mapping[str, set[str]],
) -> dict[str, dict[str, Any]]:
    execution_lanes = density_execution_lane_units()
    corrected = corrected_manifest_records(
        CORRECTED_227, 227, "CORRECTED_ACT_227"
    )
    expected_corrected = routes["remap"] | routes["direct_corrected"]
    if (
        set(corrected) != expected_corrected
        or set(corrected) != execution_lanes["corrected_core_227"]
    ):
        raise ValueError("corrected 227 identity differs")
    output = dict(corrected)
    additions = corrected_manifest_records(
        CORRECTED_LEGACY_TENSOR_216,
        216,
        "CORRECTED_ACT_LEGACY_TENSOR_216",
    )
    expected = routes["keep"] | routes["tensor"]
    if (
        set(additions) != expected
        or set(additions)
        != execution_lanes["corrected_legacy_tensor_216"]
    ):
        raise ValueError("corrected legacy/tensor 216 identity differs")
    if set(output) & set(additions):
        raise ValueError("corrected core and legacy/tensor units overlap")
    output.update(additions)

    additions = corrected_manifest_records(
        CORRECTED_FREESURFER_1,
        1,
        "CORRECTED_ACT_FREESURFER_1",
    )
    if (
        set(additions) != routes["freesurfer"]
        or set(additions)
        != execution_lanes["corrected_freesurfer_1"]
    ):
        raise ValueError("corrected FreeSurfer one-subject identity differs")
    if set(output) & set(additions):
        raise ValueError("corrected FreeSurfer and other units overlap")
    output.update(additions)

    archive_decision = load_json(ARCHIVE_CONCORDANCE)
    if archive_decision["decision"] == ARCHIVE_RETAIN_DECISION:
        corrected_low = corrected_manifest_records(
            CORRECTED_ARCHIVE_LOW_30,
            30,
            "CORRECTED_ACT_ARCHIVE_LOW_30",
        )
        retained_D0 = archive_records(
            execution_lanes["conditional_archive_D0_56"]
        )
        if (
            set(corrected_low)
            != execution_lanes["corrected_archive_low_30"]
            or set(retained_D0)
            != execution_lanes["conditional_archive_D0_56"]
            or set(corrected_low) & set(retained_D0)
            or set(corrected_low) | set(retained_D0)
            != routes["archive"]
        ):
            raise ValueError("selective archive recovery identity differs")
        additions = {**corrected_low, **retained_D0}
    elif archive_decision["decision"] == ARCHIVE_RERUN_DECISION:
        additions = corrected_manifest_records(
            CORRECTED_ARCHIVE_86,
            86,
            "CORRECTED_ACT_ARCHIVE_RERUN",
        )
        if set(additions) != routes["archive"]:
            raise ValueError("corrected archive 86 identity differs")
    else:
        raise ValueError("archive route decision differs")
    if set(output) & set(additions):
        raise ValueError("archive and other route units overlap")
    output.update(additions)
    if len(output) != EXPECTED_N:
        raise ValueError(f"resolved release has {len(output)}, not 530")
    return output


def assemble(
    root: Path,
    readiness: Mapping[str, Any],
    routes: Mapping[str, set[str]],
    overwrite: bool,
    readiness_path: Path,
) -> dict[str, Any]:
    if readiness["status"] != "READY_TO_ASSEMBLE":
        raise ValueError("release readiness is not PASS")
    release_manifest_path = root / "release_manifest.json"
    if release_manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"release exists; use --overwrite: {release_manifest_path}"
        )
    records = resolved_records(routes)
    subject_rows = []
    file_rows = []
    route_counts: Counter[str] = Counter()
    failures = []
    for index, unit in enumerate(sorted(records), start=1):
        row = records[unit]
        qc = matrix_qc(unit, row["matrices"])
        if qc["status"] != "PASS":
            failures.append({"unit": unit, "failures": qc["failures"]})
        route = str(row["source_route"])
        route_counts[route] += 1
        subject_rows.append(
            {
                "unit": unit,
                "status": qc["status"],
                "source_route": route,
                "connected_nodes": qc.get("connected_nodes"),
                "edge_density": qc.get("edge_density"),
                "endpoint_assignment_fraction": row.get(
                    "endpoint_assignment_fraction"
                ),
                **{
                    f"matrix_{name}_path": row["matrices"][name]["path"]
                    for name in MATRIX_NAMES
                },
                **{
                    f"matrix_{name}_sha256": row["matrices"][name][
                        "sha256"
                    ]
                    for name in MATRIX_NAMES
                },
            }
        )
        for name in MATRIX_NAMES:
            record = row["matrices"][name]
            file_rows.append(
                {
                    "unit": unit,
                    "matrix": name,
                    "path": record["path"],
                    "size_bytes": record["size_bytes"],
                    "sha256": record["sha256"],
                    "source_route": route,
                }
            )
        print(
            f"[{utc_now()}] release-qc {index}/{EXPECTED_N} "
            f"{unit} {qc['status']}",
            flush=True,
        )
    if (
        failures
        or len(subject_rows) != EXPECTED_N
        or len(file_rows) != EXPECTED_MATRICES
    ):
        raise ValueError(
            f"final matrix QC failed: subjects={len(subject_rows)} "
            f"files={len(file_rows)} failures={len(failures)}"
        )
    observed_densities = [
        float(row["edge_density"]) for row in subject_rows
    ]
    if (
        len(observed_densities) != EXPECTED_N
        or min(observed_densities) < MINIMUM_EDGE_DENSITY
    ):
        raise ValueError(
            "final whole-cohort density gate failed: "
            f"n={len(observed_densities)} "
            f"minimum={min(observed_densities, default=float('nan'))}"
        )
    root.mkdir(parents=True, exist_ok=True)
    atomic_csv(root / "analysis_ready_manifest.csv", subject_rows)
    atomic_csv(root / "files_sha256.tsv", file_rows)
    atomic_csv(
        root / "case_status.csv",
        [
            {
                "unit": row["unit"],
                "status": row["status"],
                "source_route": row["source_route"],
            }
            for row in subject_rows
        ],
    )
    dataset = {
        "Name": "HCP379-v2 structural connectome reference release",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": (
                    "HCP379-v2 Recovery4 bounded recovery, calibration "
                    "and selected-recipe pipeline"
                ),
                "Version": "3",
            }
        ],
        "Description": (
            "Diagnosis-blind 379-node structural connectome matrix "
            "reference manifest; source-route labels are mandatory."
        ),
    }
    atomic_json(root / "dataset_description.json", dataset)
    atomic_text(
        root / "README.md",
        "# HCP379-v2 reference release\n\n"
        "This diagnosis-blind manifest indexes 530 subjects and nine "
        "validated 379 x 379 matrix families per subject. Matrix files "
        "remain at their non-overwritten derivative paths. `source_route` "
        "must be retained in all analyses and route-exclusion sensitivity "
        "is mandatory. Every count matrix independently clears the "
        "prespecified density >= 0.60 whole-cohort technical gate; density "
        "does not establish anatomical validity and no subject may be "
        "excluded or have edges imputed to satisfy it. Diagnosis and "
        "research outcomes are attached only after this technical release "
        "is frozen.\n",
    )
    manifest = {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_v2_reference_release",
        "status": "PASS",
        "generated_utc": utc_now(),
        "subject_count": len(subject_rows),
        "matrix_family_count": len(MATRIX_NAMES),
        "matrix_file_count": len(file_rows),
        "matrix_shape": [EXPECTED_NODES, EXPECTED_NODES],
        "diagnosis_or_outcome_fields_present": False,
        "density_is_subject_inclusion_gate": False,
        "density_is_whole_cohort_technical_release_gate": True,
        "minimum_edge_density_inclusive": MINIMUM_EDGE_DENSITY,
        "subjects_at_or_above_minimum_density": len(observed_densities),
        "minimum_observed_edge_density": min(observed_densities),
        "route_counts": dict(sorted(route_counts.items())),
        "route_sensitivity_required": True,
        "legacy_tensor_route_decision": readiness[
            "legacy_tensor_route_decision"
        ],
        "archive_route_decision": readiness[
            "archive_route_decision"
        ],
        "records": {
            "readiness": file_record(readiness_path),
            "analysis_ready_manifest": file_record(
                root / "analysis_ready_manifest.csv"
            ),
            "case_status": file_record(root / "case_status.csv"),
            "files_sha256": file_record(root / "files_sha256.tsv"),
            "dataset_description": file_record(
                root / "dataset_description.json"
            ),
            "readme": file_record(root / "README.md"),
        },
    }
    atomic_json(release_manifest_path, manifest)
    return manifest


def self_test() -> None:
    count = np.zeros((EXPECTED_NODES, EXPECTED_NODES), dtype=float)
    target_edges = math.ceil(
        MINIMUM_EDGE_DENSITY
        * EXPECTED_NODES
        * (EXPECTED_NODES - 1)
        / 2
    )
    written = 0
    for left in range(EXPECTED_NODES):
        for right in range(left + 1, EXPECTED_NODES):
            if written >= target_edges:
                break
            count[left, right] = count[right, left] = written + 1
            written += 1
        if written >= target_edges:
            break
    with tempfile.TemporaryDirectory(prefix="hcp379-release-test.") as tmp:
        root = Path(tmp)
        records = {}
        for name in MATRIX_NAMES:
            path = root / f"{name}.csv"
            np.savetxt(path, count, delimiter=",")
            records[name] = file_record(path)
        result = matrix_qc("test", records)
        if result["status"] != "PASS":
            raise AssertionError(result)
        sparse = np.zeros(
            (EXPECTED_NODES, EXPECTED_NODES), dtype=float
        )
        for index in range(EXPECTED_NODES - 1):
            sparse[index, index + 1] = sparse[index + 1, index] = 1
        sparse_records = {}
        for name in MATRIX_NAMES:
            path = root / f"sparse_{name}.csv"
            np.savetxt(path, sparse, delimiter=",")
            sparse_records[name] = file_record(path)
        sparse_result = matrix_qc("sparse-test", sparse_records)
        if (
            sparse_result["status"] != "FAIL"
            or not any(
                str(value).startswith("edge_density=")
                for value in sparse_result["failures"]
            )
        ):
            raise AssertionError(sparse_result)
    print("SELF_TEST_PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-output", type=Path, default=READINESS_OUTPUT)
    parser.add_argument("--release-root", type=Path, default=RELEASE_ROOT)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    readiness, routes = current_readiness()
    atomic_json(args.readiness_output, readiness)
    print(json.dumps(readiness, indent=2, sort_keys=True), flush=True)
    if args.assemble:
        manifest = assemble(
            args.release_root,
            readiness,
            routes,
            args.overwrite,
            args.readiness_output,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return int(
        args.require_ready and readiness["status"] != "READY_TO_ASSEMBLE"
    )


if __name__ == "__main__":
    raise SystemExit(main())
