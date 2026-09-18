"""Read-only loader for a verified balanced HCP379 analysis handoff.

This module is intentionally separate from the legacy AAL dashboard loader.
It preserves scan-level identity, all nine matrix families, calibration versus
production role, corrected recovery route, and the frozen two-seed recipe.
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
EXPECTED_ROLES = {
    "CALIBRATION_CANARY_REUSE": 15,
    "BALANCED_PRODUCTION": 515,
}
EXPECTED_ROUTES = {
    "corrected_core": 227,
    "corrected_legacy_tensor": 216,
    "archive_calibration": 6,
    "corrected_archive_low": 28,
    "corrected_archive_D0": 52,
    "corrected_freesurfer": 1,
}
SUBJECT_FIELDS = {
    "unit",
    "subject_id",
    "image_id",
    "status",
    "cohort_role",
    "source_recovery_lane",
    "selected_balanced_total_streamlines",
    "selected_per_seed_streamlines",
    "edge_density",
    "connected_nodes",
    "minimum_weighted_count_support_fraction",
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
    "cohort_role",
    "source_recovery_lane",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_absolute() or not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"regular absolute non-symlink file required: {path}")
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


def _bound_internal_path(
    manifest: dict[str, Any],
    name: str,
    parent: Path,
) -> Path:
    records = manifest.get("records")
    if not isinstance(records, dict) or not isinstance(
        records.get(name), dict
    ):
        raise ValueError(f"handoff record absent: {name}")
    expected = records[name]
    path = Path(str(expected.get("path", ""))).resolve()
    if (
        path.parent != parent.resolve()
        or expected != _file_record(path)
    ):
        raise ValueError(f"handoff record differs: {name}")
    return path


def _bound_external_path(
    manifest: dict[str, Any],
    name: str,
) -> Path:
    records = manifest.get("records")
    if not isinstance(records, dict) or not isinstance(
        records.get(name), dict
    ):
        raise ValueError(f"handoff record absent: {name}")
    expected = records[name]
    path = Path(str(expected.get("path", ""))).resolve()
    if expected != _file_record(path):
        raise ValueError(f"handoff record differs: {name}")
    return path


@dataclass(frozen=True)
class VerifiedBalancedHCP379Handoff:
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


def load_verified_balanced_handoff(
    manifest_path: str | Path,
    *,
    verify_all_matrix_hashes: bool = False,
) -> VerifiedBalancedHCP379Handoff:
    """Load an immutable handoff without changing dashboard state."""
    path = Path(manifest_path).expanduser().resolve()
    manifest = _load_json(path)
    if (
        manifest.get("record_type")
        != "hcp379_balanced_verified_analysis_handoff"
        or manifest.get("status") != "PASS"
        or manifest.get("subject_count") != EXPECTED_SUBJECTS
        or manifest.get("matrix_family_count") != len(MATRIX_FAMILIES)
        or manifest.get("matrix_file_count") != EXPECTED_FILES
        or manifest.get("matrix_shape")
        != [EXPECTED_NODES, EXPECTED_NODES]
        or manifest.get("all_matrix_files_rehashed_during_handoff")
        is not True
        or manifest.get(
            "calibration_production_equivalence_verified"
        )
        is not True
        or manifest.get("diagnosis_or_outcome_fields_present") is not False
        or manifest.get("matrix_files_copied") is not False
        or manifest.get("cohort_uniform") is not True
        or manifest.get("live_dashboard_modified") is not False
        or manifest.get("live_dashboard_activation_authorized") is not False
    ):
        raise ValueError("balanced HCP379 handoff manifest differs")
    parent = path.parent
    subject_path = _bound_internal_path(
        manifest, "subject_index", parent
    )
    matrix_path = _bound_internal_path(
        manifest, "matrix_index", parent
    )
    contract_path = _bound_internal_path(
        manifest, "analysis_contract", parent
    )
    _bound_internal_path(
        manifest, "dashboard_compatibility", parent
    )
    _bound_internal_path(manifest, "readme", parent)
    equivalence_path = _bound_external_path(
        manifest, "canary_production_equivalence"
    )
    equivalence = _load_json(equivalence_path)
    if (
        equivalence.get("record_type")
        != "hcp379_balanced_canary_production_equivalence_validation"
        or equivalence.get("status") != "PASS"
        or equivalence.get("complete_seed_metadata") is not True
        or equivalence.get("seed_metadata_expected_n") != 30
        or equivalence.get("seed_metadata_pass_n") != 30
        or equivalence.get("seed_metadata_waiting_n") != 0
        or equivalence.get("seed_metadata_failure_n") != 0
        or equivalence.get("canary_pair_complete_n") != 15
        or equivalence.get("production_execution_authorized") is not True
        or equivalence.get("diagnosis_labels_used") is not False
    ):
        raise ValueError(
            "balanced HCP379 calibration-production equivalence differs"
        )
    contract = _load_json(contract_path)
    if (
        contract.get("record_type")
        != "hcp379_balanced_verified_analysis_contract"
        or contract.get("status") != "PASS"
        or contract.get("atlas") != "HCP-MMP1"
        or contract.get("node_count") != EXPECTED_NODES
        or contract.get("primary_key") != "unit"
        or contract.get("unit_is_scan_level") is not True
        or contract.get("subject_count") != EXPECTED_SUBJECTS
        or contract.get("matrix_families") != list(MATRIX_FAMILIES)
        or contract.get("matrix_file_count") != EXPECTED_FILES
        or contract.get("cohort_role_field_required") is not True
        or contract.get("source_recovery_lane_field_required") is not True
        or contract.get(
            "source_recovery_route_sensitivity_required"
        )
        is not True
        or contract.get(
            "calibration_production_equivalence_required"
        )
        is not True
        or contract.get("cohort_uniform") is not True
        or contract.get("per_subject_streamline_tuning_allowed") is not False
        or contract.get("subject_exclusion_for_low_density_allowed") is not False
        or contract.get("diagnosis_or_outcome_fields_present") is not False
        or contract.get("legacy_AAL_dashboard_activation_authorized")
        is not False
    ):
        raise ValueError("balanced HCP379 analysis contract differs")
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
        or set(pd.to_numeric(matrices["node_count"]))
        != {EXPECTED_NODES}
        or set(subjects["status"]) != {"PASS"}
    ):
        raise ValueError("balanced HCP379 handoff tables differ")
    for field, expected in (
        ("cohort_role", EXPECTED_ROLES),
        ("source_recovery_lane", EXPECTED_ROUTES),
    ):
        observed = (
            subjects[field].astype(str).value_counts().to_dict()
        )
        if observed != expected:
            raise ValueError(f"{field} counts differ")
        mapping = subjects.set_index("unit")[field].astype(str)
        if not matrices["unit"].map(mapping).astype(str).eq(
            matrices[field].astype(str)
        ).all():
            raise ValueError(f"{field} matrix provenance differs")
    selected = set(
        pd.to_numeric(
            subjects["selected_balanced_total_streamlines"]
        )
    )
    per_seed = set(
        pd.to_numeric(subjects["selected_per_seed_streamlines"])
    )
    if (
        selected
        != {int(contract["selected_balanced_total_streamlines"])}
        or per_seed
        != {int(contract["selected_per_seed_streamlines"])}
        or next(iter(selected)) != 2 * next(iter(per_seed))
    ):
        raise ValueError("balanced recipe columns differ")
    observed_paths: set[Path] = set()
    for row in matrices.itertuples(index=False):
        matrix_file = Path(str(row.path)).resolve()
        if (
            not matrix_file.is_absolute()
            or not matrix_file.is_file()
            or matrix_file.is_symlink()
            or matrix_file in observed_paths
            or matrix_file.stat().st_size != int(row.size_bytes)
        ):
            raise ValueError(
                f"matrix path invalid: {row.unit}/{row.matrix_family}"
            )
        observed_paths.add(matrix_file)
        if (
            verify_all_matrix_hashes
            and _sha256(matrix_file) != str(row.sha256)
        ):
            raise ValueError(f"matrix hash differs: {matrix_file}")
    if len(observed_paths) != EXPECTED_FILES:
        raise ValueError("matrix-path cardinality differs")
    return VerifiedBalancedHCP379Handoff(
        manifest_path=path,
        manifest=manifest,
        contract=contract,
        subjects=subjects,
        matrices=matrices,
    )
