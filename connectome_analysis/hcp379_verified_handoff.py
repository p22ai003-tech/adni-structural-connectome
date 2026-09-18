"""Read-only loader for an immutable HCP379-v2 verified analysis handoff.

This module is intentionally not imported by the current AAL dashboard.  A new
HCP379-aware analysis can use it to retain scan-level identity, matrix-family
semantics, source-route provenance, and the upstream verification bindings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


MATRIX_FAMILIES = (
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
EXPECTED_FILES = EXPECTED_SUBJECTS * len(MATRIX_FAMILIES)
SUBJECT_FIELDS = {
    "unit",
    "subject_id",
    "image_id",
    "status",
    "source_route",
    "connected_nodes",
    "edge_density",
    "minimum_weighted_support",
    "endpoint_assignment_fraction",
}
MATRIX_FIELDS = {
    "unit",
    "subject_id",
    "image_id",
    "atlas",
    "node_count",
    "matrix_family",
    "path",
    "size_bytes",
    "sha256",
    "source_route",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"regular absolute non-symlink file required: {path}")
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _bound_path(
    manifest: dict[str, Any], name: str, parent: Path
) -> Path:
    records = manifest.get("records")
    if not isinstance(records, dict) or not isinstance(
        records.get(name), dict
    ):
        raise ValueError(f"handoff record absent: {name}")
    expected = records[name]
    path = Path(str(expected.get("path", "")))
    if (
        not path.is_absolute()
        or path.resolve().parent != parent.resolve()
        or expected != _file_record(path)
    ):
        raise ValueError(f"handoff record differs: {name}")
    return path.resolve()


@dataclass(frozen=True)
class VerifiedHCP379Handoff:
    manifest_path: Path
    manifest: dict[str, Any]
    contract: dict[str, Any]
    subjects: pd.DataFrame
    matrices: pd.DataFrame

    def matrix_record(self, unit: str, family: str) -> pd.Series:
        if family not in MATRIX_FAMILIES:
            raise KeyError(f"unknown HCP379 matrix family: {family}")
        selected = self.matrices.loc[
            self.matrices["unit"].astype(str).eq(str(unit))
            & self.matrices["matrix_family"].astype(str).eq(family)
        ]
        if len(selected) != 1:
            raise KeyError(f"matrix not uniquely indexed: {unit}/{family}")
        return selected.iloc[0]

    def matrix_path(self, unit: str, family: str) -> Path:
        return Path(str(self.matrix_record(unit, family)["path"]))


def load_verified_handoff(
    manifest_path: str | Path,
    *,
    verify_all_matrix_hashes: bool = False,
) -> VerifiedHCP379Handoff:
    """Load and validate the handoff without changing any dashboard state.

    Control files are always re-hashed. Matrix files are checked for path,
    size, uniqueness, and non-symlink status; set ``verify_all_matrix_hashes``
    to re-hash all 4,770 matrix files again at consumer start.
    """

    path = Path(manifest_path).expanduser().resolve()
    manifest = _load_json(path)
    if (
        manifest.get("record_type")
        != "hcp379_v2_verified_analysis_handoff"
        or manifest.get("status") != "PASS"
        or manifest.get("subject_count") != EXPECTED_SUBJECTS
        or manifest.get("matrix_family_count") != len(MATRIX_FAMILIES)
        or manifest.get("matrix_file_count") != EXPECTED_FILES
        or manifest.get("matrix_shape") != [EXPECTED_NODES, EXPECTED_NODES]
        or manifest.get("all_matrix_files_rehashed_during_handoff") is not True
        or manifest.get("diagnosis_or_outcome_fields_present") is not False
        or manifest.get("live_dashboard_modified") is not False
        or manifest.get("live_dashboard_activation_authorized") is not False
    ):
        raise ValueError("HCP379 handoff manifest contract differs")

    parent = path.parent
    subject_path = _bound_path(manifest, "subject_index", parent)
    matrix_path = _bound_path(manifest, "matrix_index", parent)
    contract_path = _bound_path(manifest, "analysis_contract", parent)
    _bound_path(manifest, "dashboard_compatibility", parent)
    _bound_path(manifest, "readme", parent)
    contract = _load_json(contract_path)
    if (
        contract.get("record_type")
        != "hcp379_v2_verified_analysis_contract"
        or contract.get("status") != "PASS"
        or contract.get("atlas") != "HCP-MMP1"
        or contract.get("node_count") != EXPECTED_NODES
        or contract.get("primary_key") != "unit"
        or contract.get("unit_is_scan_level") is not True
        or contract.get("subject_count") != EXPECTED_SUBJECTS
        or contract.get("matrix_families") != list(MATRIX_FAMILIES)
        or contract.get("matrix_file_count") != EXPECTED_FILES
        or contract.get("source_route_field_required") is not True
        or contract.get("route_sensitivity_required") is not True
        or contract.get("diagnosis_or_outcome_fields_present") is not False
        or contract.get("legacy_AAL_dashboard_activation_authorized")
        is not False
    ):
        raise ValueError("HCP379 analysis contract differs")

    subjects = pd.read_csv(
        subject_path,
        dtype={"unit": str, "subject_id": str, "image_id": str},
    )
    matrices = pd.read_csv(
        matrix_path,
        sep="\t",
        dtype={"unit": str, "subject_id": str, "image_id": str},
    )
    if (
        set(subjects.columns) != SUBJECT_FIELDS
        or set(matrices.columns) != MATRIX_FIELDS
        or len(subjects) != EXPECTED_SUBJECTS
        or len(matrices) != EXPECTED_FILES
        or subjects["unit"].nunique() != EXPECTED_SUBJECTS
        or matrices[["unit", "matrix_family"]]
        .drop_duplicates()
        .shape[0]
        != EXPECTED_FILES
        or set(matrices["unit"]) != set(subjects["unit"])
        or set(matrices["matrix_family"]) != set(MATRIX_FAMILIES)
        or set(matrices["atlas"]) != {"HCP-MMP1"}
        or set(pd.to_numeric(matrices["node_count"])) != {EXPECTED_NODES}
        or set(subjects["status"]) != {"PASS"}
    ):
        raise ValueError("HCP379 handoff tables differ")
    subject_routes = subjects.set_index("unit")["source_route"].astype(str)
    matrix_routes = matrices["unit"].map(subject_routes)
    if not matrix_routes.astype(str).eq(
        matrices["source_route"].astype(str)
    ).all():
        raise ValueError("source-route provenance differs")
    if (
        dict(
            sorted(
                subjects["source_route"]
                .astype(str)
                .value_counts()
                .to_dict()
                .items()
            )
        )
        != contract.get("route_counts")
    ):
        raise ValueError("source-route counts differ")

    observed_paths: set[Path] = set()
    for row in matrices.itertuples(index=False):
        matrix_file = Path(str(row.path))
        if (
            not matrix_file.is_absolute()
            or not matrix_file.is_file()
            or matrix_file.is_symlink()
        ):
            raise ValueError(
                f"matrix path invalid: {row.unit}/{row.matrix_family}"
            )
        resolved = matrix_file.resolve()
        if resolved in observed_paths:
            raise ValueError(f"matrix path reused: {resolved}")
        observed_paths.add(resolved)
        if resolved.stat().st_size != int(row.size_bytes):
            raise ValueError(f"matrix size differs: {resolved}")
        if verify_all_matrix_hashes and _sha256(resolved) != str(row.sha256):
            raise ValueError(f"matrix hash differs: {resolved}")
    if len(observed_paths) != EXPECTED_FILES:
        raise ValueError("matrix-path cardinality differs")

    return VerifiedHCP379Handoff(
        manifest_path=path,
        manifest=manifest,
        contract=contract,
        subjects=subjects,
        matrices=matrices,
    )
