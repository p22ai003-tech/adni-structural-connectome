#!/home/ec2-user/fsl/bin/python
"""Prepare a fail-closed analysis handoff for the verified HCP379-v2 release.

The live dashboard reads a legacy AAL-style directory by filename glob.  This
tool deliberately does not populate or modify that directory.  Instead it
requires both the completed 530-unit reference release and the independent
post-assembly PASS record, re-hashes every one of the 4,770 referenced matrix
files, and writes an immutable, scan-level index for HCP379-aware consumers.

The default invocation only writes a current preflight record.  ``--prepare``
creates a new content-addressed handoff directory and has no overwrite or
activation mode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
DEFAULT_RELEASE = HCP_ROOT / "release_candidate_v2/release_manifest.json"
DEFAULT_VERIFICATION = (
    HCP_ROOT / "release_validation_v1/integration_verification.json"
)
DEFAULT_HANDOFF_ROOT = HCP_ROOT / "analysis_handoff_v1"
RECEIPT_NAME = "verified_handoff_receipt.json"
DEFAULT_PREFLIGHT = (
    EXP
    / "research_audit/outputs/hcp379_verified_analysis_handoff_v1/"
    "preflight.json"
)
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
EXPECTED_SUBJECTS = 530
EXPECTED_NODES = 379
EXPECTED_FILES = EXPECTED_SUBJECTS * len(MATRIX_NAMES)
MINIMUM_DENSITY = 0.60
MINIMUM_WEIGHTED_SUPPORT = 0.95
MINIMUM_ASSIGNMENT = 0.50
ALLOWED_STREAMLINE_COUNTS = (3_000_000, 5_000_000, 10_000_000)
UNIT_RE = re.compile(
    r"^(?P<subject_id>\d{3}_S_\d+)_I(?P<image_id>\d+)$"
)
FORBIDDEN_FIELDS = {
    "diagnosis",
    "diagnosis_at_dti",
    "diagnostic_group",
    "group",
    "research_group",
    "outcome",
    "clinical_outcome",
    "mmse",
    "cdr",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"regular absolute non-symlink file required: {path}")
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def staged_record(staged: Path, final: Path) -> dict[str, Any]:
    record = file_record(staged.resolve())
    record["path"] = str(final.resolve())
    return record


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_table(
    path: Path, delimiter: str
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
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


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def write_table(
    path: Path, rows: list[dict[str, Any]], delimiter: str
) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter=delimiter
        )
        writer.writeheader()
        writer.writerows(rows)


def expected_analysis_fields() -> set[str]:
    fields = {
        "unit",
        "status",
        "source_route",
        "connected_nodes",
        "edge_density",
        "endpoint_assignment_fraction",
    }
    for name in MATRIX_NAMES:
        fields.add(f"matrix_{name}_path")
        fields.add(f"matrix_{name}_sha256")
    return fields


def forbidden_headers(fields: tuple[str, ...] | set[str]) -> set[str]:
    return {
        str(field).strip().lower()
        for field in fields
        if str(field).strip().lower() in FORBIDDEN_FIELDS
    }


def unit_parts(unit: str) -> tuple[str, str]:
    match = UNIT_RE.fullmatch(unit)
    if match is None:
        raise ValueError(f"scan-level HCP379 unit required: {unit}")
    return match.group("subject_id"), match.group("image_id")


def record_path(
    records: Mapping[str, Any], name: str, expected_parent: Path
) -> Path:
    value = records.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing file record: {name}")
    raw = Path(str(value.get("path", "")))
    if (
        not raw.is_absolute()
        or not raw.is_file()
        or raw.is_symlink()
        or raw.resolve().parent != expected_parent.resolve()
        or dict(value) != file_record(raw)
    ):
        raise ValueError(f"file record differs: {name}")
    return raw.resolve()


def dashboard_compatibility() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_v2_legacy_dashboard_compatibility_audit",
        "status": "REQUIRES_HCP379_AWARE_CONSUMER",
        "live_dashboard_modified": False,
        "live_dashboard_activation_authorized": False,
        "legacy_dashboard_compatible": False,
        "legacy_assumptions": {
            "matrix_glob": "SC_AAL166_*",
            "matrix_root": "data/derivatives/connectomes",
            "identity_key": "subject root",
            "matrix_families": [
                "ALL",
                "count",
                "fd_sum",
                "fa_mean",
                "md_mean",
                "len_mean",
            ],
        },
        "hcp379_contract": {
            "atlas": "HCP-MMP1",
            "node_count": EXPECTED_NODES,
            "identity_key": "scan-level unit",
            "subject_count": EXPECTED_SUBJECTS,
            "matrix_families": list(MATRIX_NAMES),
            "manifest_and_independent_verification_required": True,
        },
        "incompatibilities": [
            "legacy filename glob does not establish HCP379 provenance",
            "legacy subject-root collapse can merge distinct scan units",
            "legacy loader does not require the independent release verifier",
            "legacy matrix-family inventory omits three HCP379 families",
            "AAL-labelled outputs cannot be relabelled as HCP-MMP1 outputs",
        ],
        "safe_next_step": (
            "consume subject_index.csv and matrix_index.tsv by scan-level "
            "unit in a separate HCP379-aware analysis root"
        ),
    }


def validate_sources(
    release_path: Path,
    verification_path: Path,
    *,
    rehash_matrices: bool,
    allowed_matrix_root: Path,
    handoff_root: Path,
) -> dict[str, Any]:
    release_path = release_path.resolve()
    verification_path = verification_path.resolve()
    release = load_json(release_path)
    verification = load_json(verification_path)

    if (
        release.get("record_type")
        != "diagnosis_blind_hcp379_v2_reference_release"
        or release.get("status") != "PASS"
        or release.get("subject_count") != EXPECTED_SUBJECTS
        or release.get("matrix_family_count") != len(MATRIX_NAMES)
        or release.get("matrix_file_count") != EXPECTED_FILES
        or release.get("matrix_shape") != [EXPECTED_NODES, EXPECTED_NODES]
        or release.get("diagnosis_or_outcome_fields_present") is not False
        or release.get("route_sensitivity_required") is not True
        or release.get("density_is_whole_cohort_technical_release_gate")
        is not True
        or release.get("subjects_at_or_above_minimum_density")
        != EXPECTED_SUBJECTS
        or float(release.get("minimum_observed_edge_density", -1))
        < MINIMUM_DENSITY
    ):
        raise ValueError("reference release contract differs")
    if (
        verification.get("record_type")
        != "diagnosis_blind_hcp379_integration_release_verification"
        or verification.get("status") != "PASS"
        or verification.get(
            "verification_is_independent_of_assembler_imports"
        )
        is not True
        or verification.get("diagnosis_or_outcome_fields_present") is not False
        or verification.get("subject_count") != EXPECTED_SUBJECTS
        or verification.get("matrix_family_count") != len(MATRIX_NAMES)
        or verification.get("matrix_file_count") != EXPECTED_FILES
        or verification.get("matrix_shape")
        != [EXPECTED_NODES, EXPECTED_NODES]
        or verification.get("selected_streamline_count")
        not in ALLOWED_STREAMLINE_COUNTS
        or verification.get("subjects_at_or_above_minimum_density")
        != EXPECTED_SUBJECTS
        or float(verification.get("minimum_observed_edge_density", -1))
        < MINIMUM_DENSITY
        or verification.get(
            "subjects_at_or_above_minimum_weighted_support"
        )
        != EXPECTED_SUBJECTS
        or float(
            verification.get("minimum_observed_weighted_support", -1)
        )
        < MINIMUM_WEIGHTED_SUPPORT
        or verification.get(
            "subjects_at_or_above_minimum_assignment_fraction"
        )
        != EXPECTED_SUBJECTS
        or float(
            verification.get("minimum_observed_assignment_fraction", -1)
        )
        < MINIMUM_ASSIGNMENT
        or verification.get("archive_route_decision")
        != release.get("archive_route_decision")
        or verification.get("route_counts") != release.get("route_counts")
    ):
        raise ValueError("independent verification contract differs")

    verification_records = verification.get("records")
    if (
        not isinstance(verification_records, Mapping)
        or verification_records.get("release") != file_record(release_path)
    ):
        raise ValueError("verification is not bound to this release")
    release_records = release.get("records")
    if not isinstance(release_records, Mapping):
        raise ValueError("release records absent")

    release_parent = release_path.parent
    analysis_path = record_path(
        release_records, "analysis_ready_manifest", release_parent
    )
    case_path = record_path(release_records, "case_status", release_parent)
    files_path = record_path(
        release_records, "files_sha256", release_parent
    )
    record_path(release_records, "readiness", release_parent)
    record_path(release_records, "dataset_description", release_parent)
    record_path(release_records, "readme", release_parent)

    subject_qc_record = verification_records.get("subject_qc")
    if not isinstance(subject_qc_record, Mapping):
        raise ValueError("verification subject-QC record absent")
    subject_qc_path = Path(str(subject_qc_record.get("path", "")))
    if (
        not subject_qc_path.is_absolute()
        or subject_qc_path.resolve().parent
        != verification_path.parent.resolve()
        or dict(subject_qc_record) != file_record(subject_qc_path)
    ):
        raise ValueError("verification subject-QC record differs")

    analysis_fields, analysis_rows = read_table(analysis_path, ",")
    case_fields, case_rows = read_table(case_path, ",")
    file_fields, file_rows = read_table(files_path, "\t")
    qc_fields, qc_rows = read_table(subject_qc_path, "\t")
    expected_qc_fields = {
        "unit",
        "status",
        "source_route",
        "connected_nodes",
        "edge_density",
        "minimum_weighted_support",
        "endpoint_assignment_fraction",
        "matrix_files_rehashed",
    }
    if set(analysis_fields) != expected_analysis_fields():
        raise ValueError("analysis-ready fields differ")
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
        raise ValueError("matrix file-index fields differ")
    if set(qc_fields) != expected_qc_fields:
        raise ValueError("subject-QC fields differ")
    for fields in (analysis_fields, case_fields, file_fields, qc_fields):
        if forbidden_headers(set(fields)):
            raise ValueError("diagnosis or outcome field present")
    if (
        len(analysis_rows) != EXPECTED_SUBJECTS
        or len(case_rows) != EXPECTED_SUBJECTS
        or len(qc_rows) != EXPECTED_SUBJECTS
        or len(file_rows) != EXPECTED_FILES
    ):
        raise ValueError("handoff source row count differs")

    analysis_by_unit = {row["unit"]: row for row in analysis_rows}
    case_by_unit = {row["unit"]: row for row in case_rows}
    qc_by_unit = {row["unit"]: row for row in qc_rows}
    file_by_key = {
        (row["unit"], row["matrix"]): row for row in file_rows
    }
    units = set(analysis_by_unit)
    if (
        len(units) != EXPECTED_SUBJECTS
        or set(case_by_unit) != units
        or set(qc_by_unit) != units
        or len(file_by_key) != EXPECTED_FILES
        or {key[0] for key in file_by_key} != units
        or {key[1] for key in file_by_key} != set(MATRIX_NAMES)
    ):
        raise ValueError("handoff identity or product set differs")

    matrix_root = allowed_matrix_root.resolve()
    forbidden_roots = {release_parent.resolve(), handoff_root.resolve()}
    used_paths: set[Path] = set()
    subject_index: list[dict[str, Any]] = []
    matrix_index: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()

    for unit in sorted(units):
        subject_id, image_id = unit_parts(unit)
        analysis = analysis_by_unit[unit]
        case = case_by_unit[unit]
        qc = qc_by_unit[unit]
        route = analysis["source_route"]
        if (
            analysis["status"] != "PASS"
            or case
            != {
                "unit": unit,
                "status": "PASS",
                "source_route": route,
            }
            or qc["status"] != "PASS"
            or qc["source_route"] != route
            or int(qc["matrix_files_rehashed"]) != len(MATRIX_NAMES)
            or float(qc["edge_density"]) < MINIMUM_DENSITY
            or float(qc["minimum_weighted_support"])
            < MINIMUM_WEIGHTED_SUPPORT
            or float(qc["endpoint_assignment_fraction"])
            < MINIMUM_ASSIGNMENT
            or int(float(qc["connected_nodes"]))
            != int(float(analysis["connected_nodes"]))
            or abs(
                float(qc["edge_density"])
                - float(analysis["edge_density"])
            )
            > 1e-12
            or abs(
                float(qc["endpoint_assignment_fraction"])
                - float(analysis["endpoint_assignment_fraction"])
            )
            > 1e-12
        ):
            raise ValueError(f"subject release/QC record differs: {unit}")
        route_counts[route] += 1
        subject_index.append(
            {
                "unit": unit,
                "subject_id": subject_id,
                "image_id": image_id,
                "status": "PASS",
                "source_route": route,
                "connected_nodes": int(float(qc["connected_nodes"])),
                "edge_density": f"{float(qc['edge_density']):.12f}",
                "minimum_weighted_support": (
                    f"{float(qc['minimum_weighted_support']):.12f}"
                ),
                "endpoint_assignment_fraction": (
                    f"{float(qc['endpoint_assignment_fraction']):.12f}"
                ),
            }
        )
        for family in MATRIX_NAMES:
            source = file_by_key[(unit, family)]
            raw = Path(source["path"])
            if (
                not raw.is_absolute()
                or not raw.is_file()
                or raw.is_symlink()
            ):
                raise ValueError(f"invalid matrix path: {unit}/{family}")
            resolved = raw.resolve()
            if (
                resolved in used_paths
                or not resolved.is_relative_to(matrix_root)
                or any(resolved.is_relative_to(root) for root in forbidden_roots)
            ):
                raise ValueError(f"matrix path scope differs: {unit}/{family}")
            used_paths.add(resolved)
            if (
                source["unit"] != unit
                or source["matrix"] != family
                or source["source_route"] != route
                or analysis[f"matrix_{family}_path"] != str(resolved)
                or analysis[f"matrix_{family}_sha256"]
                != source["sha256"]
            ):
                raise ValueError(f"matrix index differs: {unit}/{family}")
            if rehash_matrices:
                observed = file_record(resolved)
                if (
                    observed["sha256"] != source["sha256"]
                    or observed["size_bytes"] != int(source["size_bytes"])
                ):
                    raise ValueError(
                        f"matrix changed after verification: {unit}/{family}"
                    )
            elif resolved.stat().st_size != int(source["size_bytes"]):
                raise ValueError(
                    f"matrix size changed after verification: {unit}/{family}"
                )
            matrix_index.append(
                {
                    "unit": unit,
                    "subject_id": subject_id,
                    "image_id": image_id,
                    "atlas": "HCP-MMP1",
                    "node_count": EXPECTED_NODES,
                    "matrix_family": family,
                    "path": str(resolved),
                    "size_bytes": int(source["size_bytes"]),
                    "sha256": source["sha256"],
                    "source_route": route,
                }
            )

    observed_routes = dict(sorted(route_counts.items()))
    if (
        len(used_paths) != EXPECTED_FILES
        or observed_routes != release.get("route_counts")
    ):
        raise ValueError("matrix uniqueness or route counts differ")

    release_record = file_record(release_path)
    verification_record = file_record(verification_path)
    contract = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_v2_verified_analysis_contract",
        "status": "PASS",
        "atlas": "HCP-MMP1",
        "node_count": EXPECTED_NODES,
        "primary_key": "unit",
        "unit_is_scan_level": True,
        "subject_count": EXPECTED_SUBJECTS,
        "matrix_families": list(MATRIX_NAMES),
        "matrix_file_count": EXPECTED_FILES,
        "matrix_paths_are_external_references": True,
        "matrix_files_copied_or_relabelled": False,
        "minimum_density_inclusive": MINIMUM_DENSITY,
        "density_is_whole_cohort_technical_release_gate": True,
        "density_is_subject_exclusion_rule": False,
        "edge_imputation_allowed": False,
        "selected_streamline_count": verification[
            "selected_streamline_count"
        ],
        "source_route_field_required": True,
        "route_sensitivity_required": True,
        "archive_route_decision": release["archive_route_decision"],
        "route_counts": observed_routes,
        "diagnosis_or_outcome_fields_present": False,
        "clinical_join_allowed_only_after_technical_freeze": True,
        "legacy_AAL_dashboard_activation_authorized": False,
    }
    return {
        "release": release,
        "verification": verification,
        "release_record": release_record,
        "verification_record": verification_record,
        "subject_index": subject_index,
        "matrix_index": matrix_index,
        "contract": contract,
    }


def prepare(
    release_path: Path,
    verification_path: Path,
    output_root: Path,
    *,
    allowed_matrix_root: Path = Path("/data/derivatives"),
) -> dict[str, Any]:
    validated = validate_sources(
        release_path,
        verification_path,
        rehash_matrices=True,
        allowed_matrix_root=allowed_matrix_root,
        handoff_root=output_root,
    )
    release_record = validated["release_record"]
    verification_record = validated["verification_record"]
    handoff_id = (
        f"hcp379-v2-r{release_record['sha256'][:16]}"
        f"-v{verification_record['sha256'][:16]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    final_root = output_root / handoff_id
    receipt_path = output_root / RECEIPT_NAME
    if receipt_path.exists():
        raise FileExistsError(
            f"immutable handoff receipt already exists; no overwrite "
            f"allowed: {receipt_path}"
        )
    if final_root.exists():
        existing_manifest_path = final_root / "handoff_manifest.json"
        existing = load_json(existing_manifest_path)
        if (
            existing.get("record_type")
            != "hcp379_v2_verified_analysis_handoff"
            or existing.get("status") != "PASS"
            or existing.get("handoff_id") != handoff_id
            or existing.get("records", {}).get("release")
            != release_record
            or existing.get("records", {}).get(
                "independent_verification"
            )
            != verification_record
        ):
            raise FileExistsError(
                f"non-matching immutable handoff exists: {final_root}"
            )
        receipt = {
            "schema_version": "1.0.0",
            "record_type": "hcp379_v2_verified_analysis_handoff_receipt",
            "status": "PASS",
            "generated_utc": utc_now(),
            "handoff_id": handoff_id,
            "subject_count": EXPECTED_SUBJECTS,
            "matrix_family_count": len(MATRIX_NAMES),
            "matrix_file_count": EXPECTED_FILES,
            "matrix_shape": [EXPECTED_NODES, EXPECTED_NODES],
            "diagnosis_or_outcome_fields_present": False,
            "live_dashboard_modified": False,
            "records": {
                "release": release_record,
                "independent_verification": verification_record,
                "handoff_manifest": file_record(
                    existing_manifest_path.resolve()
                ),
            },
        }
        atomic_json(receipt_path, receipt)
        return {
            "status": "PASS",
            "handoff_id": handoff_id,
            "handoff_root": str(final_root),
            "manifest": str(existing_manifest_path),
            "receipt": str(receipt_path),
            "subject_count": EXPECTED_SUBJECTS,
            "matrix_file_count": EXPECTED_FILES,
            "recovered_missing_receipt": True,
            "live_dashboard_modified": False,
        }
    staged = Path(
        tempfile.mkdtemp(prefix=f".{handoff_id}.", dir=output_root)
    )
    try:
        subject_path = staged / "subject_index.csv"
        matrix_path = staged / "matrix_index.tsv"
        contract_path = staged / "analysis_contract.json"
        compatibility_path = staged / "dashboard_compatibility.json"
        readme_path = staged / "README.md"
        write_table(subject_path, validated["subject_index"], ",")
        write_table(matrix_path, validated["matrix_index"], "\t")
        write_json(contract_path, validated["contract"])
        write_json(compatibility_path, dashboard_compatibility())
        write_text(
            readme_path,
            "# HCP379-v2 verified analysis handoff\n\n"
            "This immutable, diagnosis-blind handoff indexes 530 scan-level "
            "units and 4,770 independently verified HCP-MMP1 matrix files. "
            "The matrix files remain at their validated derivative paths and "
            "are neither copied nor renamed. Use `unit`, not subject-root "
            "identity, as the primary key. Retain `source_route` and perform "
            "the mandatory route-sensitivity analysis. Clinical or diagnostic "
            "metadata may be joined only after this technical index is frozen.\n\n"
            "This handoff is not compatible with the live legacy AAL glob "
            "loader and does not activate or modify that dashboard.\n",
        )
        final_subject = final_root / subject_path.name
        final_matrix = final_root / matrix_path.name
        final_contract = final_root / contract_path.name
        final_compatibility = final_root / compatibility_path.name
        final_readme = final_root / readme_path.name
        manifest = {
            "schema_version": "1.0.0",
            "record_type": "hcp379_v2_verified_analysis_handoff",
            "status": "PASS",
            "generated_utc": utc_now(),
            "handoff_id": handoff_id,
            "subject_count": EXPECTED_SUBJECTS,
            "matrix_family_count": len(MATRIX_NAMES),
            "matrix_file_count": EXPECTED_FILES,
            "matrix_shape": [EXPECTED_NODES, EXPECTED_NODES],
            "all_matrix_files_rehashed_during_handoff": True,
            "diagnosis_or_outcome_fields_present": False,
            "live_dashboard_modified": False,
            "live_dashboard_activation_authorized": False,
            "records": {
                "release": release_record,
                "independent_verification": verification_record,
                "subject_index": staged_record(
                    subject_path, final_subject
                ),
                "matrix_index": staged_record(matrix_path, final_matrix),
                "analysis_contract": staged_record(
                    contract_path, final_contract
                ),
                "dashboard_compatibility": staged_record(
                    compatibility_path, final_compatibility
                ),
                "readme": staged_record(readme_path, final_readme),
                "builder": file_record(Path(__file__).resolve()),
            },
        }
        write_json(staged / "handoff_manifest.json", manifest)
        os.rename(staged, final_root)
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        raise
    receipt = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_v2_verified_analysis_handoff_receipt",
        "status": "PASS",
        "generated_utc": utc_now(),
        "handoff_id": handoff_id,
        "subject_count": EXPECTED_SUBJECTS,
        "matrix_family_count": len(MATRIX_NAMES),
        "matrix_file_count": EXPECTED_FILES,
        "matrix_shape": [EXPECTED_NODES, EXPECTED_NODES],
        "diagnosis_or_outcome_fields_present": False,
        "live_dashboard_modified": False,
        "records": {
            "release": release_record,
            "independent_verification": verification_record,
            "handoff_manifest": file_record(
                (final_root / "handoff_manifest.json").resolve()
            ),
        },
    }
    atomic_json(receipt_path, receipt)
    return {
        "status": "PASS",
        "handoff_id": handoff_id,
        "handoff_root": str(final_root),
        "manifest": str(final_root / "handoff_manifest.json"),
        "receipt": str(receipt_path),
        "subject_count": EXPECTED_SUBJECTS,
        "matrix_file_count": EXPECTED_FILES,
        "live_dashboard_modified": False,
    }


def preflight(
    release_path: Path,
    verification_path: Path,
    output_root: Path,
    preflight_output: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_v2_verified_analysis_handoff_preflight",
        "generated_utc": utc_now(),
        "release_path": str(release_path),
        "release_present": release_path.is_file(),
        "verification_path": str(verification_path),
        "verification_present": verification_path.is_file(),
        "handoff_root": str(output_root),
        "handoff_created": False,
        "matrix_rehash_executed": False,
        "live_dashboard_modified": False,
        "compatibility": dashboard_compatibility(),
    }
    if not release_path.is_file() or not verification_path.is_file():
        result["status"] = "AWAITING_VERIFIED_RELEASE"
    else:
        try:
            validated = validate_sources(
                release_path,
                verification_path,
                rehash_matrices=False,
                allowed_matrix_root=Path("/data/derivatives"),
                handoff_root=output_root,
            )
            result.update(
                {
                    "status": "READY_TO_PREPARE_WITH_FULL_REHASH",
                    "release": validated["release_record"],
                    "verification": validated["verification_record"],
                    "subject_count": len(validated["subject_index"]),
                    "matrix_file_count": len(validated["matrix_index"]),
                }
            )
        except Exception as exc:
            result.update(
                {
                    "status": "FAIL_CONTROL_VALIDATION",
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
    atomic_json(preflight_output, result)
    return result


def self_test() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(
        prefix="hcp379-handoff-selftest-"
    ) as raw_root:
        root = Path(raw_root)
        matrix_root = root / "derivatives"
        release_root = root / "release"
        verify_root = root / "verify"
        matrix_root.mkdir()
        release_root.mkdir()
        verify_root.mkdir()
        subject_rows: list[dict[str, Any]] = []
        case_rows: list[dict[str, Any]] = []
        file_rows: list[dict[str, Any]] = []
        qc_rows: list[dict[str, Any]] = []
        first_matrix: Path | None = None
        for index in range(EXPECTED_SUBJECTS):
            unit = (
                f"{(index % 999) + 1:03d}_S_{index + 1000:04d}"
                f"_I{index + 1000000}"
            )
            route = "CORRECTED_ACT_TEST"
            analysis: dict[str, Any] = {
                "unit": unit,
                "status": "PASS",
                "source_route": route,
                "connected_nodes": 379,
                "edge_density": "0.700000000000",
                "endpoint_assignment_fraction": "0.800000000000",
            }
            unit_root = matrix_root / unit
            unit_root.mkdir()
            for family in MATRIX_NAMES:
                path = unit_root / f"{family}.csv"
                path.write_text(f"{unit},{family}\n", encoding="utf-8")
                if first_matrix is None:
                    first_matrix = path
                record = file_record(path.resolve())
                analysis[f"matrix_{family}_path"] = record["path"]
                analysis[f"matrix_{family}_sha256"] = record["sha256"]
                file_rows.append(
                    {
                        "unit": unit,
                        "matrix": family,
                        "path": record["path"],
                        "size_bytes": record["size_bytes"],
                        "sha256": record["sha256"],
                        "source_route": route,
                    }
                )
            subject_rows.append(analysis)
            case_rows.append(
                {
                    "unit": unit,
                    "status": "PASS",
                    "source_route": route,
                }
            )
            qc_rows.append(
                {
                    "unit": unit,
                    "status": "PASS",
                    "source_route": route,
                    "connected_nodes": 379,
                    "edge_density": "0.700000000000",
                    "minimum_weighted_support": "1.000000000000",
                    "endpoint_assignment_fraction": "0.800000000000",
                    "matrix_files_rehashed": 9,
                }
            )
        analysis_path = release_root / "analysis_ready_manifest.csv"
        case_path = release_root / "case_status.csv"
        files_path = release_root / "files_sha256.tsv"
        readiness_path = release_root / "hcp379_release_readiness.json"
        dataset_path = release_root / "dataset_description.json"
        readme_path = release_root / "README.md"
        write_table(analysis_path, subject_rows, ",")
        write_table(case_path, case_rows, ",")
        write_table(files_path, file_rows, "\t")
        write_json(readiness_path, {"status": "READY_TO_ASSEMBLE"})
        write_json(dataset_path, {"Name": "synthetic"})
        write_text(readme_path, "synthetic\n")
        release_path = release_root / "release_manifest.json"
        release = {
            "record_type": "diagnosis_blind_hcp379_v2_reference_release",
            "status": "PASS",
            "subject_count": EXPECTED_SUBJECTS,
            "matrix_family_count": len(MATRIX_NAMES),
            "matrix_file_count": EXPECTED_FILES,
            "matrix_shape": [EXPECTED_NODES, EXPECTED_NODES],
            "diagnosis_or_outcome_fields_present": False,
            "route_sensitivity_required": True,
            "density_is_whole_cohort_technical_release_gate": True,
            "subjects_at_or_above_minimum_density": EXPECTED_SUBJECTS,
            "minimum_observed_edge_density": 0.7,
            "archive_route_decision": "SYNTHETIC",
            "route_counts": {"CORRECTED_ACT_TEST": EXPECTED_SUBJECTS},
            "records": {
                "readiness": file_record(readiness_path.resolve()),
                "analysis_ready_manifest": file_record(
                    analysis_path.resolve()
                ),
                "case_status": file_record(case_path.resolve()),
                "files_sha256": file_record(files_path.resolve()),
                "dataset_description": file_record(dataset_path.resolve()),
                "readme": file_record(readme_path.resolve()),
            },
        }
        write_json(release_path, release)
        subject_qc_path = verify_root / "integration_subject_qc.tsv"
        write_table(subject_qc_path, qc_rows, "\t")
        verification_path = verify_root / "integration_verification.json"
        verification = {
            "record_type": (
                "diagnosis_blind_hcp379_integration_release_verification"
            ),
            "status": "PASS",
            "verification_is_independent_of_assembler_imports": True,
            "diagnosis_or_outcome_fields_present": False,
            "subject_count": EXPECTED_SUBJECTS,
            "matrix_family_count": len(MATRIX_NAMES),
            "matrix_file_count": EXPECTED_FILES,
            "matrix_shape": [EXPECTED_NODES, EXPECTED_NODES],
            "selected_streamline_count": 3_000_000,
            "archive_route_decision": "SYNTHETIC",
            "route_counts": {"CORRECTED_ACT_TEST": EXPECTED_SUBJECTS},
            "subjects_at_or_above_minimum_density": EXPECTED_SUBJECTS,
            "minimum_observed_edge_density": 0.7,
            "subjects_at_or_above_minimum_weighted_support": (
                EXPECTED_SUBJECTS
            ),
            "minimum_observed_weighted_support": 1.0,
            "subjects_at_or_above_minimum_assignment_fraction": (
                EXPECTED_SUBJECTS
            ),
            "minimum_observed_assignment_fraction": 0.8,
            "records": {
                "release": file_record(release_path.resolve()),
                "subject_qc": file_record(subject_qc_path.resolve()),
            },
        }
        write_json(verification_path, verification)
        output_root = root / "handoff"
        result = prepare(
            release_path,
            verification_path,
            output_root,
            allowed_matrix_root=matrix_root,
        )
        manifest = load_json(Path(result["manifest"]))
        receipt = load_json(Path(result["receipt"]))
        checks["positive_fixture_builds"] = (
            manifest.get("status") == "PASS"
        )
        checks["terminal_receipt_is_release_bound"] = (
            receipt.get("status") == "PASS"
            and receipt.get("handoff_id") == manifest.get("handoff_id")
            and receipt.get("records", {}).get("release")
            == file_record(release_path.resolve())
            and receipt.get("records", {}).get(
                "independent_verification"
            )
            == file_record(verification_path.resolve())
            and receipt.get("records", {}).get("handoff_manifest")
            == file_record(Path(result["manifest"]).resolve())
        )
        checks["exact_scan_and_matrix_counts"] = (
            manifest.get("subject_count") == EXPECTED_SUBJECTS
            and manifest.get("matrix_file_count") == EXPECTED_FILES
        )
        checks["no_live_activation"] = (
            manifest.get("live_dashboard_modified") is False
            and manifest.get("live_dashboard_activation_authorized") is False
        )
        sys.path.insert(0, str(EXP))
        from connectome_analysis.hcp379_verified_handoff import (
            load_verified_handoff,
        )

        loaded = load_verified_handoff(
            result["manifest"], verify_all_matrix_hashes=True
        )
        checks["consumer_loader_replays"] = (
            len(loaded.subjects) == EXPECTED_SUBJECTS
            and len(loaded.matrices) == EXPECTED_FILES
            and loaded.matrix_path(
                loaded.subjects.iloc[0]["unit"], MATRIX_NAMES[0]
            ).is_file()
        )
        try:
            prepare(
                release_path,
                verification_path,
                output_root,
                allowed_matrix_root=matrix_root,
            )
            checks["existing_handoff_refused"] = False
        except FileExistsError:
            checks["existing_handoff_refused"] = True
        assert first_matrix is not None
        first_matrix.write_text("tampered\n", encoding="utf-8")
        try:
            validate_sources(
                release_path,
                verification_path,
                rehash_matrices=True,
                allowed_matrix_root=matrix_root,
                handoff_root=output_root,
            )
            checks["post_verification_matrix_tamper_fails"] = False
        except ValueError:
            checks["post_verification_matrix_tamper_fails"] = True
        first_file_row = file_rows[0]
        first_matrix.write_text(
            f"{first_file_row['unit']},{first_file_row['matrix']}\n",
            encoding="utf-8",
        )
        bad_analysis = [dict(row) for row in subject_rows]
        bad_analysis[0]["diagnosis"] = "CN"
        write_table(analysis_path, bad_analysis, ",")
        release["records"]["analysis_ready_manifest"] = file_record(
            analysis_path.resolve()
        )
        write_json(release_path, release)
        verification["records"]["release"] = file_record(
            release_path.resolve()
        )
        write_json(verification_path, verification)
        try:
            validate_sources(
                release_path,
                verification_path,
                rehash_matrices=False,
                allowed_matrix_root=matrix_root,
                handoff_root=output_root,
            )
            checks["diagnosis_field_fails"] = False
        except ValueError:
            checks["diagnosis_field_fails"] = True
    passed = sum(checks.values())
    return {
        "record_type": "hcp379_v2_verified_analysis_handoff_selftest",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed": passed,
        "total": len(checks),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument(
        "--verification", type=Path, default=DEFAULT_VERIFICATION
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_HANDOFF_ROOT
    )
    parser.add_argument(
        "--preflight-output", type=Path, default=DEFAULT_PREFLIGHT
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            result = self_test()
        elif args.prepare:
            result = prepare(
                args.release, args.verification, args.output_root
            )
        else:
            result = preflight(
                args.release,
                args.verification,
                args.output_root,
                args.preflight_output,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") in {
            "PASS",
            "AWAITING_VERIFIED_RELEASE",
            "READY_TO_PREPARE_WITH_FULL_REHASH",
        } else 1
    except Exception as exc:
        failure = {
            "record_type": "hcp379_v2_verified_analysis_handoff_failure",
            "status": "FAIL",
            "generated_utc": utc_now(),
            "error": f"{type(exc).__name__}:{exc}",
            "handoff_created": False,
            "live_dashboard_modified": False,
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
