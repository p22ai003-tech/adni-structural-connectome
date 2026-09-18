#!/home/ec2-user/fsl/bin/python
"""Validate the balanced verified-handoff builder and read-only loader."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
BUILDER_SOURCE = (
    EXP
    / "scripts/hcp/"
    "prepare_hcp379_balanced_verified_handoff_v1.py"
)
LOADER_SOURCE = (
    EXP
    / "connectome_analysis/"
    "hcp379_balanced_verified_handoff.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_balanced_verified_handoff_v1/validation.json"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = load_module(BUILDER_SOURCE, "balanced_handoff_validator_builder")
LOADER = load_module(LOADER_SOURCE, "balanced_handoff_validator_loader")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    BUILDER.atomic_json(path, value)


def synthetic_loader_replay() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="hcp379-balanced-handoff-validation."
    ) as temporary:
        root = Path(temporary)
        route_sequence: list[str] = []
        for route, count in BUILDER.EXPECTED_ROUTES.items():
            route_sequence.extend([route] * count)
        units = [
            f"{index % 1000:03d}_S_{100000 + index}_I{200000 + index}"
            for index in range(BUILDER.EXPECTED_SUBJECTS)
        ]
        subjects = []
        matrices = []
        for index, (unit, route) in enumerate(
            zip(units, route_sequence, strict=True)
        ):
            subject_id, image_id = BUILDER.unit_parts(unit)
            role = (
                "CALIBRATION_CANARY_REUSE"
                if index < 15
                else "BALANCED_PRODUCTION"
            )
            subjects.append(
                {
                    "unit": unit,
                    "subject_id": subject_id,
                    "image_id": image_id,
                    "status": "PASS",
                    "cohort_role": role,
                    "source_recovery_lane": route,
                    "selected_balanced_total_streamlines": 20_000_000,
                    "selected_per_seed_streamlines": 10_000_000,
                    "edge_density": "0.61",
                    "connected_nodes": 379,
                    "minimum_weighted_count_support_fraction": "0.995",
                    "endpoint_assignment_fraction": "0.90",
                }
            )
            for family in BUILDER.MATRIX_NAMES:
                matrix_path = root / "matrices" / unit / f"{family}.csv"
                matrix_path.parent.mkdir(parents=True, exist_ok=True)
                matrix_path.write_text(
                    f"{unit},{family}\n", encoding="utf-8"
                )
                matrices.append(
                    {
                        "unit": unit,
                        "subject_id": subject_id,
                        "image_id": image_id,
                        "atlas": "HCP-MMP1",
                        "node_count": 379,
                        "matrix_family": family,
                        "path": str(matrix_path.resolve()),
                        "size_bytes": matrix_path.stat().st_size,
                        "sha256": sha256(matrix_path),
                        "cohort_role": role,
                        "source_recovery_lane": route,
                    }
                )
        subject_path = root / "subject_index.csv"
        matrix_path = root / "matrix_index.tsv"
        BUILDER.atomic_csv(
            subject_path, subjects, delimiter=","
        )
        BUILDER.atomic_csv(
            matrix_path, matrices, delimiter="\t"
        )
        contract = {
            "schema_version": "1.0.0",
            "record_type": "hcp379_balanced_verified_analysis_contract",
            "status": "PASS",
            "atlas": "HCP-MMP1",
            "node_count": 379,
            "primary_key": "unit",
            "unit_is_scan_level": True,
            "subject_count": 530,
            "matrix_families": list(BUILDER.MATRIX_NAMES),
            "matrix_file_count": 4770,
            "cohort_role_field_required": True,
            "source_recovery_lane_field_required": True,
            "source_recovery_route_sensitivity_required": True,
            "calibration_production_equivalence_required": True,
            "route_counts": dict(BUILDER.EXPECTED_ROUTES),
            "role_counts": dict(BUILDER.EXPECTED_ROLES),
            "selected_balanced_total_streamlines": 20_000_000,
            "selected_per_seed_streamlines": 10_000_000,
            "cohort_uniform": True,
            "per_subject_streamline_tuning_allowed": False,
            "subject_exclusion_for_low_density_allowed": False,
            "diagnosis_or_outcome_fields_present": False,
            "outcome_join_allowed_only_after_handoff_freeze": True,
            "legacy_AAL_dashboard_activation_authorized": False,
        }
        contract_path = root / "analysis_contract.json"
        atomic_json(contract_path, contract)
        compatibility_path = root / "dashboard_compatibility.json"
        atomic_json(
            compatibility_path,
            {
                "record_type": (
                    "hcp379_balanced_legacy_dashboard_compatibility"
                ),
                "status": "REQUIRES_HCP379_AWARE_CONSUMER",
            },
        )
        readme_path = root / "README.md"
        readme_path.write_text("synthetic\n", encoding="utf-8")
        equivalence_path = root / "canary_production_equivalence.json"
        atomic_json(
            equivalence_path,
            {
                "record_type": (
                    "hcp379_balanced_canary_production_equivalence_validation"
                ),
                "status": "PASS",
                "complete_seed_metadata": True,
                "seed_metadata_expected_n": 30,
                "seed_metadata_pass_n": 30,
                "seed_metadata_waiting_n": 0,
                "seed_metadata_failure_n": 0,
                "canary_pair_complete_n": 15,
                "production_execution_authorized": True,
                "diagnosis_labels_used": False,
            },
        )
        manifest = {
            "schema_version": "1.0.0",
            "record_type": "hcp379_balanced_verified_analysis_handoff",
            "status": "PASS",
            "subject_count": 530,
            "matrix_family_count": 9,
            "matrix_file_count": 4770,
            "matrix_shape": [379, 379],
            "all_matrix_files_rehashed_during_handoff": True,
            "calibration_production_equivalence_verified": True,
            "diagnosis_or_outcome_fields_present": False,
            "matrix_files_copied": False,
            "cohort_uniform": True,
            "live_dashboard_modified": False,
            "live_dashboard_activation_authorized": False,
            "records": {
                "subject_index": record(subject_path),
                "matrix_index": record(matrix_path),
                "analysis_contract": record(contract_path),
                "dashboard_compatibility": record(
                    compatibility_path
                ),
                "readme": record(readme_path),
                "canary_production_equivalence": record(
                    equivalence_path
                ),
            },
        }
        manifest_path = root / "handoff_manifest.json"
        atomic_json(manifest_path, manifest)
        loaded = LOADER.load_verified_balanced_handoff(
            manifest_path, verify_all_matrix_hashes=True
        )
        first_matrix = Path(matrices[0]["path"])
        first_matrix.write_text("tampered\n", encoding="utf-8")
        tamper_rejected = False
        try:
            LOADER.load_verified_balanced_handoff(
                manifest_path, verify_all_matrix_hashes=True
            )
        except ValueError:
            tamper_rejected = True
        return {
            "loaded_subject_n": len(loaded.subjects),
            "loaded_matrix_n": len(loaded.matrices),
            "matrix_lookup_matches": (
                loaded.matrix_path(
                    units[0], BUILDER.MATRIX_NAMES[0]
                )
                == first_matrix
            ),
            "tamper_rejected": tamper_rejected,
        }


def main() -> int:
    checks: dict[str, Any] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    compile_probe = subprocess.run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(BUILDER_SOURCE),
            str(LOADER_SOURCE),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "sources_compile",
        compile_probe.returncode == 0,
        {
            "returncode": compile_probe.returncode,
            "stderr": compile_probe.stderr[-2000:],
        },
    )
    self_probe = subprocess.run(
        [str(PYTHON), str(BUILDER_SOURCE), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        self_value = json.loads(self_probe.stdout)
    except Exception:
        self_value = {}
    check(
        "builder_self_test_passes",
        self_probe.returncode == 0
        and self_value.get("status") == "PASS",
        self_value or {"stderr": self_probe.stderr[-2000:]},
    )
    dry_probe = subprocess.run(
        [str(PYTHON), str(BUILDER_SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        dry = json.loads(dry_probe.stdout)
    except Exception:
        dry = {}
    check(
        "dry_run_is_fail_closed_and_non_mutating",
        (
            dry_probe.returncode == 0
            and dry.get("status")
            in {
                "AWAITING_VERIFIED_BALANCED_RELEASE",
                "READY_TO_PREPARE_HANDOFF",
                "INVALID_VERIFIED_RELEASE",
            }
            and dry.get("handoff_preparation_started") is False
            and dry.get("matrix_rehash_started") is False
            and dry.get("live_dashboard_modified") is False
            and dry.get("live_dashboard_activation_authorized") is False
        ),
        dry or {"stderr": dry_probe.stderr[-2000:]},
    )
    replay = synthetic_loader_replay()
    check(
        "full_synthetic_530_by_9_loader_replay_passes",
        (
            replay["loaded_subject_n"] == 530
            and replay["loaded_matrix_n"] == 4770
            and replay["matrix_lookup_matches"] is True
        ),
        replay,
    )
    check(
        "matrix_tamper_is_rejected",
        replay["tamper_rejected"] is True,
        replay,
    )
    builder_text = BUILDER_SOURCE.read_text(encoding="utf-8")
    loader_text = LOADER_SOURCE.read_text(encoding="utf-8")
    required = (
        "CALIBRATION_CANARY_REUSE",
        "BALANCED_PRODUCTION",
        "source_recovery_lane",
        "selected_balanced_total_streamlines",
        "all_matrix_files_rehashed_during_handoff",
        "calibration_production_equivalence_required",
        "calibration_production_equivalence_verified",
        "legacy_AAL_dashboard_activation_authorized",
    )
    missing = [
        f"{path.name}:{fragment}"
        for path, text in (
            (BUILDER_SOURCE, builder_text),
            (LOADER_SOURCE, loader_text),
        )
        for fragment in required
        if fragment not in text
    ]
    check(
        "balanced_role_route_recipe_contract_is_explicit",
        not missing,
        {"missing": missing},
    )
    prohibited = (
        "shutil.copy",
        "shutil.copy2",
        "rm -rf",
        "--overwrite",
        '"live_dashboard_activation_authorized": True',
    )
    found = [
        f"{path.name}:{fragment}"
        for path, text in (
            (BUILDER_SOURCE, builder_text),
            (LOADER_SOURCE, loader_text),
        )
        for fragment in prohibited
        if fragment in text
    ]
    check(
        "no_copy_delete_overwrite_or_activation_shortcut",
        not found,
        {"found": found},
    )
    passed = sum(row["status"] == "PASS" for row in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_balanced_verified_handoff_v1_validation"
        ),
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "imaging_executed": False,
        "builder": record(BUILDER_SOURCE),
        "loader": record(LOADER_SOURCE),
        "checks": checks,
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
