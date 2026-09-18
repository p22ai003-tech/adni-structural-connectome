#!/home/ec2-user/fsl/bin/python
"""Prepare an immutable analysis handoff for the verified balanced release.

The default invocation is a read-only preflight.  ``--prepare`` requires the
balanced 530-subject release and its implementation-independent PASS record,
re-hashes every one of the 4,770 matrix files, and writes content-addressed
scan-level indexes for an HCP379-aware consumer.  It neither copies matrices
nor modifies or activates the legacy AAL dashboard.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
RELEASE_ROOT = (
    HCP_ROOT / "balanced_release_v1/integration_release_v1"
)
RELEASE_POINTER = RELEASE_ROOT / "current_release.json"
VERIFICATION_POINTER = (
    RELEASE_ROOT
    / "independent_verification_v1/current_verification.json"
)
ROOT = HCP_ROOT / "balanced_release_v1/verified_analysis_handoff_v1"
ATTEMPTS = ROOT / "attempts"
PREFLIGHT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_balanced_verified_handoff_v1/preflight.json"
)
CURRENT = ROOT / "current_handoff.json"
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


def atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    delimiter: str,
) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
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


def read_table(
    path: Path, *, delimiter: str
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = tuple(reader.fieldnames or ())
        rows = [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    return fields, rows


def unit_parts(unit: str) -> tuple[str, str]:
    matched = UNIT_RE.fullmatch(unit)
    if matched is None:
        raise ValueError(f"scan-level HCP379 unit required: {unit}")
    return matched.group("subject_id"), matched.group("image_id")


def forbidden(fields: Sequence[str]) -> list[str]:
    return sorted(
        field
        for field in fields
        if field.strip().lower() in FORBIDDEN_FIELDS
    )


def verified_release_contract() -> dict[str, Any]:
    release_pointer = load_json(RELEASE_POINTER)
    verification_pointer = load_json(VERIFICATION_POINTER)
    release_path = verify_record(
        release_pointer["release_manifest"], label="release manifest"
    )
    verification_path = verify_record(
        verification_pointer["verification"], label="verification"
    )
    if (
        verification_pointer.get("release_manifest")
        != file_record(release_path)
    ):
        raise ValueError("verification/release binding differs")
    release = load_json(release_path)
    verification = load_json(verification_path)
    release_records = release.get("records", {})
    if not isinstance(release_records, Mapping):
        raise ValueError("release records absent")
    equivalence_path = verify_record(
        release_records.get("canary_production_equivalence", {}),
        label="canary-production equivalence",
    )
    equivalence_record = file_record(equivalence_path)
    if (
        release_pointer.get("record_type")
        != "diagnosis_blind_hcp379_balanced_integration_release_pointer"
        or release_pointer.get("status")
        != "PASS_ASSEMBLED_AWAITING_INDEPENDENT_VERIFICATION"
        or release.get("record_type")
        != "diagnosis_blind_hcp379_balanced_integration_release"
        or release.get("status")
        != "PASS_ASSEMBLED_AWAITING_INDEPENDENT_VERIFICATION"
        or release.get("subject_count") != EXPECTED_SUBJECTS
        or release.get("matrix_file_count") != EXPECTED_FILES
        or release.get("diagnosis_or_outcome_fields_present") is not False
        or release.get("cohort_uniform") is not True
        or release.get("per_subject_streamline_tuning_allowed") is not False
        or release.get("subject_exclusion_for_low_density_allowed") is not False
        or verification_pointer.get("record_type")
        != "hcp379_balanced_integration_release_verification_pointer"
        or verification_pointer.get("status")
        != "PASS_VERIFIED_INTEGRATION_RELEASE"
        or verification.get("record_type")
        != "hcp379_balanced_integration_release_independent_verification"
        or verification.get("status")
        != "PASS_VERIFIED_INTEGRATION_RELEASE"
        or verification.get("release_manifest") != file_record(release_path)
        or verification.get("subject_count") != EXPECTED_SUBJECTS
        or verification.get("matrix_file_count") != EXPECTED_FILES
        or verification.get("all_subjects_density_target_met") is not True
        or verification.get(
            "calibration_production_equivalence_verified"
        )
        is not True
        or verification.get("canary_seed_metadata_replayed_n") != 30
        or verification.get("canary_pair_replayed_n") != 15
        or verification.get("canary_production_equivalence")
        != equivalence_record
        or verification.get("failure_subject_n") != 0
        or verification.get("final_release_authorized") is not True
    ):
        raise ValueError("verified balanced release contract differs")
    return {
        "release_path": release_path,
        "release": release,
        "verification_path": verification_path,
        "verification": verification,
        "equivalence_path": equivalence_path,
        "equivalence_record": equivalence_record,
        "release_pointer": file_record(RELEASE_POINTER),
        "verification_pointer": file_record(VERIFICATION_POINTER),
    }


def preflight() -> dict[str, Any]:
    detected = RELEASE_POINTER.is_file() and VERIFICATION_POINTER.is_file()
    if not detected:
        return {
            "schema_version": "1.0.0",
            "record_type": (
                "hcp379_balanced_verified_analysis_handoff_preflight"
            ),
            "generated_utc": utc_now(),
            "status": "AWAITING_VERIFIED_BALANCED_RELEASE",
            "verified_release_detected": False,
            "handoff_preparation_started": False,
            "matrix_rehash_started": False,
            "live_dashboard_modified": False,
            "live_dashboard_activation_authorized": False,
        }
    try:
        verified = verified_release_contract()
        status = "READY_TO_PREPARE_HANDOFF"
        error = None
        release = file_record(verified["release_path"])
        verification = file_record(verified["verification_path"])
    except Exception as exc:
        status = "INVALID_VERIFIED_RELEASE"
        error = f"{type(exc).__name__}:{exc}"
        release = None
        verification = None
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_balanced_verified_analysis_handoff_preflight"
        ),
        "generated_utc": utc_now(),
        "status": status,
        "verified_release_detected": True,
        "release_manifest": release,
        "independent_verification": verification,
        "error": error,
        "handoff_preparation_started": False,
        "matrix_rehash_started": False,
        "live_dashboard_modified": False,
        "live_dashboard_activation_authorized": False,
    }


def prepare() -> dict[str, Any]:
    ready = preflight()
    if ready["status"] != "READY_TO_PREPARE_HANDOFF":
        raise RuntimeError(ready["status"])
    if CURRENT.exists():
        raise FileExistsError(f"immutable handoff already exists: {CURRENT}")
    verified = verified_release_contract()
    release = verified["release"]
    verification = verified["verification"]
    release_parent = verified["release_path"].parent
    records = release["records"]
    analysis_path = verify_record(
        records["analysis_ready_manifest"], label="analysis manifest"
    )
    matrix_path = verify_record(
        records["matrix_files"], label="matrix index"
    )
    verified_qc_path = verify_record(
        verification["verified_subject_qc"],
        label="independently verified subject QC",
    )
    analysis_fields, subjects = read_table(
        analysis_path, delimiter=","
    )
    matrix_fields, matrices = read_table(
        matrix_path, delimiter="\t"
    )
    qc_fields, verified_qc = read_table(
        verified_qc_path, delimiter="\t"
    )
    if forbidden(
        analysis_fields + matrix_fields + qc_fields
    ):
        raise ValueError("diagnosis or outcome field in technical release")
    subject_units = [row["unit"] for row in subjects]
    qc_by_unit = {row["unit"]: row for row in verified_qc}
    if (
        len(subjects) != EXPECTED_SUBJECTS
        or len(set(subject_units)) != EXPECTED_SUBJECTS
        or len(verified_qc) != EXPECTED_SUBJECTS
        or set(qc_by_unit) != set(subject_units)
        or len(matrices) != EXPECTED_FILES
        or len(
            {(row["unit"], row["matrix"]) for row in matrices}
        )
        != EXPECTED_FILES
        or {row["unit"] for row in matrices} != set(subject_units)
        or {row["matrix"] for row in matrices} != set(MATRIX_NAMES)
        or Counter(row["cohort_role"] for row in subjects)
        != Counter(EXPECTED_ROLES)
        or Counter(row["source_recovery_lane"] for row in subjects)
        != Counter(EXPECTED_ROUTES)
        or any(
            row["status"] != "PASS"
            or row["failure_count"] != "0"
            for row in verified_qc
        )
    ):
        raise ValueError("verified 530-by-nine tables differ")
    selected_totals = {
        int(row["selected_balanced_total_streamlines"])
        for row in subjects
    }
    per_seed_values = {
        int(row["selected_per_seed_streamlines"]) for row in subjects
    }
    if (
        selected_totals
        != {int(release["selected_balanced_total_streamlines"])}
        or per_seed_values
        != {int(release["selected_per_seed_streamlines"])}
        or next(iter(selected_totals))
        != 2 * next(iter(per_seed_values))
    ):
        raise ValueError("uniform balanced recipe differs")
    matrix_rows: list[dict[str, Any]] = []
    for index, row in enumerate(
        sorted(matrices, key=lambda item: (item["unit"], item["matrix"])),
        start=1,
    ):
        unit = row["unit"]
        subject_id, image_id = unit_parts(unit)
        path = Path(row["path"]).resolve()
        if (
            not path.is_file()
            or path.is_symlink()
            or str(path) != row["path"]
            or path.stat().st_size != int(row["size_bytes"])
            or sha256(path) != row["sha256"]
        ):
            raise ValueError(f"{unit}/{row['matrix']}: matrix drift")
        matrix_rows.append(
            {
                "unit": unit,
                "subject_id": subject_id,
                "image_id": image_id,
                "atlas": "HCP-MMP1",
                "node_count": EXPECTED_NODES,
                "matrix_family": row["matrix"],
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": row["sha256"],
                "cohort_role": row["cohort_role"],
                "source_recovery_lane": row[
                    "source_recovery_lane"
                ],
            }
        )
        print(
            f"[{utc_now()}] handoff-matrix-rehash "
            f"{index}/{EXPECTED_FILES} {unit}/{row['matrix']}",
            flush=True,
        )
    subject_rows: list[dict[str, Any]] = []
    for row in sorted(subjects, key=lambda item: item["unit"]):
        unit = row["unit"]
        subject_id, image_id = unit_parts(unit)
        qc = qc_by_unit[unit]
        if (
            qc["cohort_role"] != row["cohort_role"]
            or qc["source_recovery_lane"]
            != row["source_recovery_lane"]
            or abs(float(qc["edge_density"]) - float(row["edge_density"]))
            > 5.0e-10
        ):
            raise ValueError(f"{unit}: verified QC binding differs")
        subject_rows.append(
            {
                "unit": unit,
                "subject_id": subject_id,
                "image_id": image_id,
                "status": "PASS",
                "cohort_role": row["cohort_role"],
                "source_recovery_lane": row[
                    "source_recovery_lane"
                ],
                "selected_balanced_total_streamlines": row[
                    "selected_balanced_total_streamlines"
                ],
                "selected_per_seed_streamlines": row[
                    "selected_per_seed_streamlines"
                ],
                "edge_density": qc["edge_density"],
                "connected_nodes": qc["connected_nodes"],
                "minimum_weighted_count_support_fraction": qc[
                    "minimum_weighted_count_support_fraction"
                ],
                "endpoint_assignment_fraction": qc[
                    "endpoint_assignment_fraction"
                ],
            }
        )
    content_token = (
        file_record(verified["release_path"])["sha256"]
        + file_record(verified["verification_path"])["sha256"]
    )
    content_id = hashlib.sha256(
        content_token.encode("utf-8")
    ).hexdigest()
    attempt = ATTEMPTS / f"handoff-{content_id[:20]}"
    if attempt.exists():
        raise FileExistsError(f"content-addressed handoff exists: {attempt}")
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_csv(
        attempt / "subject_index.csv",
        subject_rows,
        delimiter=",",
    )
    atomic_csv(
        attempt / "matrix_index.tsv",
        matrix_rows,
        delimiter="\t",
    )
    route_counts = dict(
        sorted(Counter(row["source_recovery_lane"] for row in subject_rows).items())
    )
    role_counts = dict(
        sorted(Counter(row["cohort_role"] for row in subject_rows).items())
    )
    contract = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_balanced_verified_analysis_contract",
        "status": "PASS",
        "generated_utc": utc_now(),
        "atlas": "HCP-MMP1",
        "node_count": EXPECTED_NODES,
        "primary_key": "unit",
        "unit_is_scan_level": True,
        "subject_count": EXPECTED_SUBJECTS,
        "matrix_families": list(MATRIX_NAMES),
        "matrix_file_count": EXPECTED_FILES,
        "cohort_role_field_required": True,
        "source_recovery_lane_field_required": True,
        "source_recovery_route_sensitivity_required": True,
        "calibration_production_equivalence_required": True,
        "route_counts": route_counts,
        "role_counts": role_counts,
        "selected_balanced_total_streamlines": next(
            iter(selected_totals)
        ),
        "selected_per_seed_streamlines": next(iter(per_seed_values)),
        "cohort_uniform": True,
        "per_subject_streamline_tuning_allowed": False,
        "subject_exclusion_for_low_density_allowed": False,
        "diagnosis_or_outcome_fields_present": False,
        "outcome_join_allowed_only_after_handoff_freeze": True,
        "legacy_AAL_dashboard_activation_authorized": False,
    }
    atomic_json(attempt / "analysis_contract.json", contract)
    compatibility = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_balanced_legacy_dashboard_compatibility"
        ),
        "status": "REQUIRES_HCP379_AWARE_CONSUMER",
        "live_dashboard_modified": False,
        "live_dashboard_activation_authorized": False,
        "legacy_dashboard_compatible": False,
        "reasons": [
            "legacy loader assumes AAL-labelled filenames",
            "legacy loader may collapse distinct scan-level units",
            "legacy matrix inventory does not enforce all nine families",
            "AAL outputs cannot be relabelled as HCP-MMP1",
        ],
        "safe_consumer": (
            "connectome_analysis.hcp379_balanced_verified_handoff"
        ),
    }
    atomic_json(
        attempt / "dashboard_compatibility.json", compatibility
    )
    atomic_text(
        attempt / "README.md",
        "# Verified balanced HCP379 analysis handoff\n\n"
        "This content-addressed handoff indexes 530 scan-level units and "
        "4,770 externally stored HCP-MMP1 matrices after independent release "
        "verification and a second matrix rehash. It copies no matrix and "
        "contains no diagnosis or outcome. Preserve `cohort_role` and "
        "`source_recovery_lane`; route sensitivity is mandatory. Attach "
        "clinical metadata only after this handoff is frozen. Do not point "
        "the legacy AAL dashboard at these indexes.\n",
    )
    manifest = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_balanced_verified_analysis_handoff"
        ),
        "status": "PASS",
        "generated_utc": utc_now(),
        "content_id": content_id,
        "diagnosis_or_outcome_fields_present": False,
        "non_overwriting": True,
        "matrix_files_copied": False,
        "all_matrix_files_rehashed_during_handoff": True,
        "calibration_production_equivalence_verified": True,
        "subject_count": EXPECTED_SUBJECTS,
        "matrix_family_count": len(MATRIX_NAMES),
        "matrix_file_count": EXPECTED_FILES,
        "matrix_shape": [EXPECTED_NODES, EXPECTED_NODES],
        "selected_balanced_total_streamlines": next(
            iter(selected_totals)
        ),
        "selected_per_seed_streamlines": next(iter(per_seed_values)),
        "cohort_uniform": True,
        "route_counts": route_counts,
        "role_counts": role_counts,
        "live_dashboard_modified": False,
        "live_dashboard_activation_authorized": False,
        "records": {
            "release_manifest": file_record(
                verified["release_path"]
            ),
            "independent_verification": file_record(
                verified["verification_path"]
            ),
            "canary_production_equivalence": verified[
                "equivalence_record"
            ],
            "release_pointer": verified["release_pointer"],
            "verification_pointer": verified[
                "verification_pointer"
            ],
            "subject_index": file_record(
                attempt / "subject_index.csv"
            ),
            "matrix_index": file_record(
                attempt / "matrix_index.tsv"
            ),
            "analysis_contract": file_record(
                attempt / "analysis_contract.json"
            ),
            "dashboard_compatibility": file_record(
                attempt / "dashboard_compatibility.json"
            ),
            "readme": file_record(attempt / "README.md"),
            "implementation": file_record(Path(__file__)),
        },
    }
    atomic_json(attempt / "handoff_manifest.json", manifest)
    pointer = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_balanced_verified_analysis_handoff_pointer"
        ),
        "status": "PASS",
        "created_utc": utc_now(),
        "handoff_manifest": file_record(
            attempt / "handoff_manifest.json"
        ),
    }
    if CURRENT.exists():
        raise FileExistsError(CURRENT)
    atomic_json(CURRENT, pointer)
    return {
        "status": "PASS",
        "attempt_root": str(attempt.resolve()),
        "content_id": content_id,
        "subject_count": EXPECTED_SUBJECTS,
        "matrix_file_count": EXPECTED_FILES,
        "current_handoff": str(CURRENT.resolve()),
        "live_dashboard_modified": False,
        "live_dashboard_activation_authorized": False,
    }


def self_test() -> dict[str, Any]:
    subject, image = unit_parts("003_S_4118_I1124861")
    invalid_rejected = False
    try:
        unit_parts("003_S_4118")
    except ValueError:
        invalid_rejected = True
    checks = {
        "unit_parser": (
            subject == "003_S_4118" and image == "1124861"
        ),
        "invalid_unit_rejected": invalid_rejected,
        "exact_530_by_9": EXPECTED_FILES == 4770,
        "role_counts_sum_to_530": (
            sum(EXPECTED_ROLES.values()) == EXPECTED_SUBJECTS
        ),
        "route_counts_sum_to_530": (
            sum(EXPECTED_ROUTES.values()) == EXPECTED_SUBJECTS
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        value = self_test()
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0 if value["status"] == "PASS" else 1
    if args.prepare:
        value = prepare()
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    value = preflight()
    atomic_json(PREFLIGHT, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
