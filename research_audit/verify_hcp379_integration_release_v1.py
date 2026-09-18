#!/home/ec2-user/fsl/bin/python
"""Independently verify a completed HCP379-v2 integration release.

This verifier does not import the release assembler. It re-opens the frozen
tables and every referenced matrix, re-hashes all 4,770 files, recomputes
subject-level numerical and support QC, checks the exact 530 identities,
replays Phase-B/route/human-review provenance and rejects diagnosis or outcome
fields. The verification output is separate from the scientific derivatives.
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
RELEASE_ROOT = HCP_ROOT / "release_candidate_v2"
RELEASE = RELEASE_ROOT / "release_manifest.json"
ANALYSIS = RELEASE_ROOT / "analysis_ready_manifest.csv"
CASES = RELEASE_ROOT / "case_status.csv"
FILES = RELEASE_ROOT / "files_sha256.tsv"
DATASET = RELEASE_ROOT / "dataset_description.json"
README = RELEASE_ROOT / "README.md"
READINESS = RELEASE_ROOT / "hcp379_release_readiness.json"
FREEZE = HCP_ROOT / "manifests/historical_freeze_manifest.csv"
HUMAN_QC = (
    HCP_ROOT / "review_recovery4/human_visual_qc_recovery4.csv"
)
PHASE_VALIDATION = Path(
    "/data/derivatives/scforge_v2/"
    "h04a_r1_recovery_20260719_retry4/publication/"
    "hcp379_recipe_stability_validation.json"
)
ARCHIVE_CONCORDANCE = (
    HCP_ROOT / "calibration/archive_route_concordance.json"
)
LANE_MANIFESTS = {
    "core_227": (
        HCP_ROOT
        / "corrected_scaleup_recovery4/manifests/"
        "corrected_227_release_manifest.json",
        227,
        "diagnosis_blind_hcp379_corrected_227_release_manifest",
    ),
    "legacy_tensor_216": (
        HCP_ROOT
        / "corrected_legacy_tensor_recovery4/manifests/"
        "corrected_legacy_tensor_216_release_manifest.json",
        216,
        (
            "diagnosis_blind_hcp379_corrected_legacy_tensor_216_"
            "release_manifest"
        ),
    ),
    "archive_low_30": (
        HCP_ROOT
        / "corrected_archive_low_recovery4/manifests/"
        "corrected_archive_low_30_release_manifest.json",
        30,
        (
            "diagnosis_blind_hcp379_corrected_archive_low_30_"
            "release_manifest"
        ),
    ),
    "archive_D0_56": (
        HCP_ROOT
        / "corrected_archive_D0_recovery4/manifests/"
        "corrected_archive_D0_56_release_manifest.json",
        56,
        (
            "diagnosis_blind_hcp379_corrected_archive_D0_56_"
            "release_manifest"
        ),
    ),
    "freesurfer_1": (
        HCP_ROOT
        / "corrected_freesurfer_recovery4/manifests/"
        "corrected_freesurfer_1_release_manifest.json",
        1,
        (
            "diagnosis_blind_hcp379_corrected_freesurfer_1_"
            "release_manifest"
        ),
    ),
    "corrected_archive_86": (
        HCP_ROOT
        / "corrected_archive_recovery4/manifests/"
        "corrected_archive_86_release_manifest.json",
        86,
        (
            "diagnosis_blind_hcp379_corrected_archive_86_"
            "release_manifest"
        ),
    ),
}
OUTPUT_ROOT = HCP_ROOT / "release_validation_v1"
OUTPUT = OUTPUT_ROOT / "integration_verification.json"
SUBJECT_QC = OUTPUT_ROOT / "integration_subject_qc.tsv"
PREFLIGHT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_integration_release_verifier_v1/preflight.json"
)
EXPECTED_SUBJECTS = 530
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
EXPECTED_FILES = EXPECTED_SUBJECTS * len(MATRIX_NAMES)
MINIMUM_DENSITY = 0.60
MINIMUM_WEIGHTED_SUPPORT = 0.95
MINIMUM_CONNECTED_NODES = 360
MINIMUM_ASSIGNMENT_FRACTION = 0.50
HUB_IDS = (361, 365, 366, 370, 374, 375)
ALLOWED_COUNTS = (3_000_000, 5_000_000, 10_000_000)
ARCHIVE_RETAIN_DECISION = (
    "RETAIN_ARCHIVE_D0_56_AND_USE_CORRECTED_LOW_30_"
    "WITH_ROUTE_SENSITIVITY"
)
ARCHIVE_RERUN_DECISION = (
    "RERUN_ARCHIVE_86_WITH_LOCKED_CORRECTED_ACT"
)
EXPECTED_ROUTE_COUNTS = {
    ARCHIVE_RETAIN_DECISION: {
        "ARCHIVE_NO_ACT_CALIBRATED": 56,
        "CORRECTED_ACT_227": 227,
        "CORRECTED_ACT_ARCHIVE_LOW_30": 30,
        "CORRECTED_ACT_FREESURFER_1": 1,
        "CORRECTED_ACT_LEGACY_TENSOR_216": 216,
    },
    ARCHIVE_RERUN_DECISION: {
        "CORRECTED_ACT_227": 227,
        "CORRECTED_ACT_ARCHIVE_RERUN": 86,
        "CORRECTED_ACT_FREESURFER_1": 1,
        "CORRECTED_ACT_LEGACY_TENSOR_216": 216,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"regular non-symlink file required: {path}")
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


def read_table(path: Path, delimiter: str) -> tuple[
    tuple[str, ...], list[dict[str, str]]
]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = tuple(reader.fieldnames or ())
        rows = [
            {
                str(key): str(value or "").strip()
                for key, value in row.items()
            }
            for row in reader
        ]
    return fields, rows


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


def atomic_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("TSV rows cannot be empty")
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def bundle_qc(
    unit: str, matrices: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    failures: list[str] = []
    if set(matrices) != set(MATRIX_NAMES):
        return {
            "status": "FAIL",
            "unit": unit,
            "failures": ["matrix_family_set"],
        }
    for name in MATRIX_NAMES:
        matrix = matrices[name]
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
    count = matrices["count"]
    if not np.allclose(count, np.rint(count), atol=1e-6, rtol=0):
        failures.append("count:not_integer")
    support = count > 0
    np.fill_diagonal(support, False)
    strengths = np.sum(count, axis=1)
    connected_nodes = int(np.count_nonzero(strengths > 0))
    if connected_nodes < MINIMUM_CONNECTED_NODES:
        failures.append(f"connected_nodes={connected_nodes}")
    for hub in HUB_IDS:
        if strengths[hub - 1] <= 0:
            failures.append(f"hub_{hub}:disconnected")
    supported_edges = int(np.count_nonzero(np.triu(support, 1)))
    possible = EXPECTED_NODES * (EXPECTED_NODES - 1) // 2
    density = supported_edges / possible
    if density < MINIMUM_DENSITY:
        failures.append(f"edge_density={density:.12f}")
    weighted_support: dict[str, float] = {}
    for name in MATRIX_NAMES[1:]:
        value = (
            float(
                np.count_nonzero(
                    np.triu((matrices[name] != 0) & support, 1)
                )
            )
            / supported_edges
            if supported_edges
            else 0.0
        )
        weighted_support[name] = value
        if value < MINIMUM_WEIGHTED_SUPPORT:
            failures.append(f"{name}:support={value:.12f}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "unit": unit,
        "connected_nodes": connected_nodes,
        "supported_edges": supported_edges,
        "edge_density": density,
        "weighted_support": weighted_support,
        "minimum_weighted_support": min(
            weighted_support.values(), default=0.0
        ),
        "failures": sorted(set(failures)),
    }


def validate_human_qc() -> dict[str, Any]:
    fields, rows = read_table(HUMAN_QC, ",")
    forbidden = {
        "diagnosis",
        "diagnosis_at_dti",
        "group",
        "research_group",
        "outcome",
    }
    if forbidden & {field.strip().lower() for field in fields}:
        raise ValueError("human QC contains diagnosis/outcome fields")
    if not {"unit", "status", "reviewer", "reviewed_utc"} <= set(fields):
        raise ValueError("human QC fields differ")
    if len(rows) != 15 or len({row["unit"] for row in rows}) != 15:
        raise ValueError("human QC identity differs")
    for row in rows:
        if (
            row["status"].upper() != "PASS"
            or not row["reviewer"]
            or not row["reviewed_utc"]
        ):
            raise ValueError(f"incomplete human QC: {row['unit']}")
        text = row["reviewed_utc"]
        reviewed = datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
        if reviewed.tzinfo is None or reviewed > datetime.now(timezone.utc):
            raise ValueError(f"invalid human QC time: {row['unit']}")
    return file_record(HUMAN_QC)


def phase_and_lane_provenance() -> dict[str, Any]:
    phase = load_json(PHASE_VALIDATION)
    selected = phase.get(
        "smallest_density_qualified_scale_up_streamline_count"
    )
    if (
        phase.get("record_type")
        != "diagnosis_blind_hcp379_phase_b_recipe_stability_validation"
        or phase.get("status") != "PASS"
        or phase.get("diagnosis_labels_used") is not False
        or phase.get("unit_count") != 15
        or selected not in ALLOWED_COUNTS
    ):
        raise ValueError("Phase-B validation differs")
    lane_records: dict[str, Any] = {}
    lane_units: dict[str, set[str]] = {}
    for name, (path, expected_n, record_type) in LANE_MANIFESTS.items():
        value = load_json(path)
        units = value.get("units")
        if (
            value.get("record_type") != record_type
            or value.get("status") != "PASS"
            or value.get("diagnosis_labels_used") is not False
            or value.get("unit_count") != expected_n
            or value.get("selected_streamline_count") != selected
            or value.get("minimum_edge_density_inclusive")
            != MINIMUM_DENSITY
            or value.get("subjects_at_or_above_minimum_density")
            != expected_n
            or float(value.get("minimum_observed_edge_density", -1.0))
            < MINIMUM_DENSITY
            or not isinstance(units, list)
            or len(units) != expected_n
        ):
            raise ValueError(f"lane manifest differs: {name}")
        observed = {
            str(row.get("unit", ""))
            for row in units
            if isinstance(row, Mapping)
            and float(row.get("edge_density", -1.0))
            >= MINIMUM_DENSITY
        }
        if len(observed) != expected_n:
            raise ValueError(f"lane subject QC differs: {name}")
        lane_units[name] = observed
        lane_records[name] = file_record(path)
    if (
        lane_units["core_227"] & lane_units["legacy_tensor_216"]
        or lane_units["archive_low_30"] & lane_units["archive_D0_56"]
        or lane_units["archive_low_30"] | lane_units["archive_D0_56"]
        != lane_units["corrected_archive_86"]
    ):
        raise ValueError("lane overlap/union differs")
    return {
        "selected_streamline_count": selected,
        "phase_b": file_record(PHASE_VALIDATION),
        "lanes": lane_records,
        "lane_units": lane_units,
    }


def expected_analysis_fields() -> set[str]:
    return {
        "unit",
        "status",
        "source_route",
        "connected_nodes",
        "edge_density",
        "endpoint_assignment_fraction",
    } | {
        f"matrix_{name}_{suffix}"
        for name in MATRIX_NAMES
        for suffix in ("path", "sha256")
    }


def preflight_payload() -> dict[str, Any]:
    release_present = RELEASE.is_file()
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_integration_release_verifier_preflight",
        "status": "READY_TO_VERIFY" if release_present else "AWAITING_RELEASE",
        "generated_utc": utc_now(),
        "verification_executed": False,
        "release_manifest": (
            file_record(RELEASE) if release_present else None
        ),
        "contract": {
            "subjects": EXPECTED_SUBJECTS,
            "matrix_families": len(MATRIX_NAMES),
            "matrix_files": EXPECTED_FILES,
            "shape": [EXPECTED_NODES, EXPECTED_NODES],
            "minimum_density_inclusive": MINIMUM_DENSITY,
            "minimum_weighted_support": MINIMUM_WEIGHTED_SUPPORT,
            "minimum_connected_nodes": MINIMUM_CONNECTED_NODES,
            "minimum_endpoint_assignment_fraction": (
                MINIMUM_ASSIGNMENT_FRACTION
            ),
            "diagnosis_or_outcome_fields_allowed": False,
            "matrix_symlinks_allowed": False,
            "matrix_paths_inside_release_root_allowed": False,
        },
        "source": file_record(Path(__file__)),
    }


def verify() -> dict[str, Any]:
    release = load_json(RELEASE)
    if (
        release.get("record_type")
        != "diagnosis_blind_hcp379_v2_reference_release"
        or release.get("status") != "PASS"
        or release.get("subject_count") != EXPECTED_SUBJECTS
        or release.get("matrix_family_count") != len(MATRIX_NAMES)
        or release.get("matrix_file_count") != EXPECTED_FILES
        or release.get("matrix_shape")
        != [EXPECTED_NODES, EXPECTED_NODES]
        or release.get("diagnosis_or_outcome_fields_present") is not False
        or release.get("density_is_whole_cohort_technical_release_gate")
        is not True
        or release.get("subjects_at_or_above_minimum_density")
        != EXPECTED_SUBJECTS
        or float(release.get("minimum_observed_edge_density", -1.0))
        < MINIMUM_DENSITY
        or release.get("archive_route_decision")
        not in EXPECTED_ROUTE_COUNTS
    ):
        raise ValueError("release manifest contract differs")
    release_records = release.get("records")
    expected_release_records = {
        "readiness": READINESS,
        "analysis_ready_manifest": ANALYSIS,
        "case_status": CASES,
        "files_sha256": FILES,
        "dataset_description": DATASET,
        "readme": README,
    }
    if not isinstance(release_records, Mapping):
        raise ValueError("release records absent")
    for name, path in expected_release_records.items():
        if release_records.get(name) != file_record(path):
            raise ValueError(f"release record differs: {name}")

    readiness = load_json(READINESS)
    if (
        readiness.get("record_type")
        != "diagnosis_blind_hcp379_final_release_readiness"
        or readiness.get("status") != "READY_TO_ASSEMBLE"
        or readiness.get("blockers") != []
        or readiness.get("expected_subjects") != EXPECTED_SUBJECTS
        or readiness.get("expected_matrix_files") != EXPECTED_FILES
        or readiness.get("archive_route_decision")
        != release.get("archive_route_decision")
        or readiness.get("legacy_tensor_route_decision")
        != "CORRECTED_ACT_ALL_216"
        or readiness.get("diagnosis_or_outcome_fields_used_for_routing")
        is not False
    ):
        raise ValueError("release readiness differs")

    human_record = validate_human_qc()
    provenance = phase_and_lane_provenance()
    selected = provenance["selected_streamline_count"]
    concordance = load_json(ARCHIVE_CONCORDANCE)
    if (
        concordance.get("record_type")
        != (
            "diagnosis_blind_hcp379_archive_route_"
            "concordance_recovery4_validation"
        )
        or concordance.get("diagnosis_labels_used") is not False
        or concordance.get("recovery4_inputs_required") is not True
        or concordance.get("calibration_n") != 6
        or concordance.get("selected_streamline_count") != selected
        or concordance.get("decision")
        != release.get("archive_route_decision")
    ):
        raise ValueError("archive concordance differs")

    freeze_fields, freeze_rows = read_table(FREEZE, ",")
    if "fsid" not in freeze_fields:
        raise ValueError("freeze identity field absent")
    frozen_units = {row["fsid"] for row in freeze_rows}
    if len(freeze_rows) != EXPECTED_SUBJECTS or len(frozen_units) != 530:
        raise ValueError("freeze identity differs")

    analysis_fields, analysis_rows = read_table(ANALYSIS, ",")
    case_fields, case_rows = read_table(CASES, ",")
    file_fields, file_rows = read_table(FILES, "\t")
    if set(analysis_fields) != expected_analysis_fields():
        raise ValueError("analysis manifest fields differ")
    if set(case_fields) != {"unit", "status", "source_route"}:
        raise ValueError("case-status fields differ")
    if set(file_fields) != {
        "unit",
        "matrix",
        "path",
        "size_bytes",
        "sha256",
        "source_route",
    }:
        raise ValueError("file manifest fields differ")
    if (
        len(analysis_rows) != EXPECTED_SUBJECTS
        or len(case_rows) != EXPECTED_SUBJECTS
        or len(file_rows) != EXPECTED_FILES
    ):
        raise ValueError("release table row count differs")
    analysis_by_unit = {row["unit"]: row for row in analysis_rows}
    case_by_unit = {row["unit"]: row for row in case_rows}
    file_by_key = {
        (row["unit"], row["matrix"]): row for row in file_rows
    }
    if (
        len(analysis_by_unit) != EXPECTED_SUBJECTS
        or len(case_by_unit) != EXPECTED_SUBJECTS
        or len(file_by_key) != EXPECTED_FILES
        or set(analysis_by_unit) != frozen_units
        or set(case_by_unit) != frozen_units
        or {unit for unit, _ in file_by_key} != frozen_units
        or {name for _, name in file_by_key} != set(MATRIX_NAMES)
    ):
        raise ValueError("release identity/product differs")

    expected_routes = EXPECTED_ROUTE_COUNTS[
        str(release["archive_route_decision"])
    ]
    observed_routes = Counter(
        row["source_route"] for row in analysis_rows
    )
    if dict(sorted(observed_routes.items())) != expected_routes:
        raise ValueError("release route counts differ")
    if release.get("route_counts") != expected_routes:
        raise ValueError("manifest route counts differ")

    used_paths: set[Path] = set()
    subject_qc: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, unit in enumerate(sorted(frozen_units), start=1):
        row = analysis_by_unit[unit]
        case = case_by_unit[unit]
        if (
            row["status"] != "PASS"
            or case
            != {
                "unit": unit,
                "status": "PASS",
                "source_route": row["source_route"],
            }
        ):
            raise ValueError(f"case status differs: {unit}")
        matrices: dict[str, np.ndarray] = {}
        for name in MATRIX_NAMES:
            file_row = file_by_key[(unit, name)]
            raw_path = Path(file_row["path"])
            if (
                not raw_path.is_absolute()
                or not raw_path.is_file()
                or raw_path.is_symlink()
            ):
                raise ValueError(f"invalid matrix path: {unit}/{name}")
            path = raw_path.resolve()
            if (
                Path("/data/derivatives") not in path.parents
                or RELEASE_ROOT.resolve() in path.parents
                or path in used_paths
            ):
                raise ValueError(f"matrix path scope differs: {unit}/{name}")
            used_paths.add(path)
            record = file_record(path)
            if (
                file_row["unit"] != unit
                or file_row["matrix"] != name
                or file_row["source_route"] != row["source_route"]
                or int(file_row["size_bytes"]) != record["size_bytes"]
                or file_row["sha256"] != record["sha256"]
                or row[f"matrix_{name}_path"] != record["path"]
                or row[f"matrix_{name}_sha256"] != record["sha256"]
            ):
                raise ValueError(f"matrix record differs: {unit}/{name}")
            try:
                matrices[name] = np.loadtxt(path, delimiter=",")
            except Exception as exc:
                raise ValueError(
                    f"matrix unreadable: {unit}/{name}:"
                    f"{type(exc).__name__}"
                ) from exc
        qc = bundle_qc(unit, matrices)
        try:
            assignment = float(row["endpoint_assignment_fraction"])
        except ValueError:
            assignment = -1.0
        if (
            assignment < MINIMUM_ASSIGNMENT_FRACTION
            or assignment > 1.0
        ):
            qc["failures"].append(
                f"endpoint_assignment_fraction={assignment}"
            )
            qc["status"] = "FAIL"
        reported_connected = int(float(row["connected_nodes"]))
        reported_density = float(row["edge_density"])
        if reported_connected != qc["connected_nodes"]:
            qc["failures"].append("reported_connected_nodes")
            qc["status"] = "FAIL"
        if not math.isclose(
            reported_density,
            float(qc["edge_density"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            qc["failures"].append("reported_edge_density")
            qc["status"] = "FAIL"
        subject_qc.append(
            {
                "unit": unit,
                "status": qc["status"],
                "source_route": row["source_route"],
                "connected_nodes": qc["connected_nodes"],
                "edge_density": f"{qc['edge_density']:.12f}",
                "minimum_weighted_support": (
                    f"{qc['minimum_weighted_support']:.12f}"
                ),
                "endpoint_assignment_fraction": (
                    f"{assignment:.12f}"
                ),
                "matrix_files_rehashed": len(MATRIX_NAMES),
            }
        )
        if qc["status"] != "PASS":
            failures.append(
                {"unit": unit, "failures": sorted(set(qc["failures"]))}
            )
        print(
            f"[{utc_now()}] integration-verify {index}/530 "
            f"{unit} {qc['status']}",
            flush=True,
        )
    if failures or len(used_paths) != EXPECTED_FILES:
        raise ValueError(
            f"integration subject QC failed: failures={len(failures)} "
            f"files={len(used_paths)}"
        )
    densities = [float(row["edge_density"]) for row in subject_qc]
    supports = [
        float(row["minimum_weighted_support"]) for row in subject_qc
    ]
    assignments = [
        float(row["endpoint_assignment_fraction"]) for row in subject_qc
    ]
    atomic_tsv(SUBJECT_QC, subject_qc)
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_integration_release_verification"
        ),
        "status": "PASS",
        "generated_utc": utc_now(),
        "verification_is_independent_of_assembler_imports": True,
        "diagnosis_or_outcome_fields_present": False,
        "subject_count": EXPECTED_SUBJECTS,
        "matrix_family_count": len(MATRIX_NAMES),
        "matrix_file_count": len(used_paths),
        "matrix_shape": [EXPECTED_NODES, EXPECTED_NODES],
        "selected_streamline_count": selected,
        "archive_route_decision": release["archive_route_decision"],
        "route_counts": expected_routes,
        "subjects_at_or_above_minimum_density": sum(
            value >= MINIMUM_DENSITY for value in densities
        ),
        "minimum_observed_edge_density": min(densities),
        "subjects_at_or_above_minimum_weighted_support": sum(
            value >= MINIMUM_WEIGHTED_SUPPORT for value in supports
        ),
        "minimum_observed_weighted_support": min(supports),
        "subjects_at_or_above_minimum_assignment_fraction": sum(
            value >= MINIMUM_ASSIGNMENT_FRACTION for value in assignments
        ),
        "minimum_observed_assignment_fraction": min(assignments),
        "records": {
            "release": file_record(RELEASE),
            "readiness": file_record(READINESS),
            "freeze_identity_only": file_record(FREEZE),
            "human_qc": human_record,
            "phase_b": provenance["phase_b"],
            "archive_concordance": file_record(ARCHIVE_CONCORDANCE),
            "lane_manifests": provenance["lanes"],
            "subject_qc": file_record(SUBJECT_QC),
            "verifier": file_record(Path(__file__)),
        },
    }
    return result


def self_test() -> dict[str, Any]:
    possible = EXPECTED_NODES * (EXPECTED_NODES - 1) // 2
    target = math.ceil(MINIMUM_DENSITY * possible)
    count = np.zeros((EXPECTED_NODES, EXPECTED_NODES), dtype=float)
    written = 0
    for left in range(EXPECTED_NODES):
        for right in range(left + 1, EXPECTED_NODES):
            if written >= target:
                break
            count[left, right] = count[right, left] = written + 1
            written += 1
        if written >= target:
            break
    matrices = {
        name: count.copy() for name in MATRIX_NAMES
    }
    positive = bundle_qc("positive", matrices)
    sparse = {
        name: np.eye(EXPECTED_NODES, k=1)
        + np.eye(EXPECTED_NODES, k=-1)
        for name in MATRIX_NAMES
    }
    sparse_result = bundle_qc("sparse", sparse)
    unsupported = {
        name: matrix.copy() for name, matrix in matrices.items()
    }
    unsupported["ad_mean"][:] = 0.0
    unsupported_result = bundle_qc("unsupported", unsupported)
    asymmetric = {
        name: matrix.copy() for name, matrix in matrices.items()
    }
    asymmetric["fa_mean"][0, 1] += 1.0
    asymmetric_result = bundle_qc("asymmetric", asymmetric)
    checks = {
        "positive_dense_bundle_passes": positive["status"] == "PASS",
        "sparse_bundle_fails": sparse_result["status"] == "FAIL",
        "unsupported_weighted_matrix_fails": (
            unsupported_result["status"] == "FAIL"
        ),
        "asymmetric_matrix_fails": asymmetric_result["status"] == "FAIL",
        "exact_contract": (
            EXPECTED_SUBJECTS == 530
            and len(MATRIX_NAMES) == 9
            and EXPECTED_FILES == 4770
            and EXPECTED_NODES == 379
        ),
        "route_scenarios_cover_530": all(
            sum(counts.values()) == 530
            for counts in EXPECTED_ROUTE_COUNTS.values()
        ),
    }
    passed = sum(checks.values())
    return {
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed": passed,
        "total": len(checks),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    if args.verify:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(
                f"verification exists; use a new path: {args.output}"
            )
        try:
            result = verify()
        except Exception as exc:
            result = {
                "schema_version": "1.0.0",
                "record_type": (
                    "diagnosis_blind_hcp379_integration_"
                    "release_verification"
                ),
                "status": "FAIL",
                "generated_utc": utc_now(),
                "error": f"{type(exc).__name__}:{exc}",
                "release": (
                    file_record(RELEASE) if RELEASE.is_file() else None
                ),
                "verifier": file_record(Path(__file__)),
            }
        atomic_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    result = preflight_payload()
    atomic_json(PREFLIGHT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
