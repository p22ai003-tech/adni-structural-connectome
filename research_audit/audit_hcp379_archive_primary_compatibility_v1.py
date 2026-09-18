#!/home/ec2-user/fsl/bin/python
"""Audit whether the 86 restored archive connectomes can enter the primary release.

The restored archive matrices are valuable historical evidence, and many clear
the numerical density target.  Density alone cannot establish compatibility
with the current balanced final-HROI release.  This read-only audit binds the
historical tractography/atlas provenance to the current 530-unit HROI policy
and exact 15+515 balanced-production topology.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP = Path("/data/derivatives/hcp379_v2")
ARCHIVE_INPUTS = (
    EXP
    / "research_audit/outputs/hcp379_archive_corrected_input_audit_v2/"
    "archive_corrected_input_audit.csv"
)
ARCHIVE_RESTORE = HCP / "manifests/archive_restore_summary.json"
HROI_POLICY = (
    HCP
    / "source_label_repair_v1/hroi_surface_support_cohort_v1/"
    "policy_adoption_v1.json"
)
TOPOLOGY = (
    EXP
    / "research_audit/outputs/hcp379_balanced_release_topology_v1/"
    "topology.json"
)
PRODUCTION_RUNNER = (
    EXP / "scripts/hcp/run_hcp379_balanced_production_v1.py"
)
ASSEMBLER = (
    EXP / "scripts/hcp/assemble_hcp379_balanced_integration_release_v1.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/hcp379_archive_primary_compatibility_v1/"
    "audit.json"
)
EXPECTED_ARCHIVE_N = 86
EXPECTED_DENSITY_QUALIFIED_N = 56
EXPECTED_PRODUCTION_ARCHIVE_N = 86
DENSITY_TARGET = 0.60


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


def as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"not a canonical boolean: {value!r}")


def build() -> dict[str, Any]:
    with ARCHIVE_INPUTS.open(newline="", encoding="utf-8") as handle:
        archive_rows = list(csv.DictReader(handle))
    restore = load_json(ARCHIVE_RESTORE)
    hroi = load_json(HROI_POLICY)
    topology = load_json(TOPOLOGY)
    production_text = PRODUCTION_RUNNER.read_text(encoding="utf-8")
    assembler_text = ASSEMBLER.read_text(encoding="utf-8")

    archive_by_unit = {str(row["unit"]): row for row in archive_rows}
    restore_by_unit = {
        str(row["fsid"]): row for row in restore.get("subjects", [])
    }
    hroi_by_unit = hroi.get("units", {})
    archive_topology = [
        row
        for row in topology.get("production", [])
        if row.get("lane")
        in {
            "archive_calibration",
            "corrected_archive_low",
            "corrected_archive_D0",
        }
    ]
    topology_by_unit = {
        str(row["unit"]): row for row in archive_topology
    }

    units = set(archive_by_unit)
    if (
        len(archive_rows) != EXPECTED_ARCHIVE_N
        or len(units) != EXPECTED_ARCHIVE_N
        or set(restore_by_unit) != units
        or not units.issubset(hroi_by_unit)
        or set(topology_by_unit) != units
    ):
        raise ValueError("archive identity binding differs")

    density_qualified = {
        unit
        for unit, row in archive_by_unit.items()
        if float(row["archive_density"]) >= DENSITY_TARGET
    }
    if len(density_qualified) != EXPECTED_DENSITY_QUALIFIED_N:
        raise ValueError("archive density-qualified count differs")

    route_errors: list[str] = []
    atlas_errors: list[str] = []
    changed_voxel_counts: list[int] = []
    for unit in sorted(units):
        source = archive_by_unit[unit]
        restored = restore_by_unit[unit]
        adopted = hroi_by_unit[unit]
        if (
            source["archive_route"] != "NO_ACT_RESTORED_TRACK"
            or as_bool(source["historical_nodes_b0_reuse_allowed"])
            or restored.get("weighted_recipe", {}).get("sift2")
            != "no-ACT tcksift2 against reconstructed FOD"
        ):
            route_errors.append(unit)
        original = adopted.get("original_source", {})
        candidate = adopted.get("candidate", {})
        changed = int(adopted.get("changed_voxel_n", 0))
        changed_voxel_counts.append(changed)
        if (
            Path(str(source.get("hcp_source_parcellation_path", ""))).resolve()
            != Path(str(original.get("path", ""))).resolve()
            or not str(restored.get("nodes_source", "")).endswith(
                f"/{unit}/nodes_b0.nii.gz"
            )
            or original.get("sha256") == candidate.get("sha256")
            or Path(str(original.get("path", ""))).resolve()
            == Path(str(candidate.get("path", ""))).resolve()
            or changed <= 0
        ):
            atlas_errors.append(unit)

    expected_lanes = {
        "archive_calibration": 6,
        "corrected_archive_low": 28,
        "corrected_archive_D0": 52,
    }
    observed_lanes = {
        lane: sum(row.get("lane") == lane for row in archive_topology)
        for lane in expected_lanes
    }
    implementation_markers = {
        "balanced_runner_declares_final_hroi": (
            "balanced final-HROI recipe" in production_text
            and "all nine matrices are built from" in production_text
            and "the final HROI atlas" in production_text
        ),
        "balanced_runner_declares_two_equal_act_replicates": (
            "two equal deterministic tractography replicates"
            in production_text
            and '"-act"' in production_text
        ),
        "assembler_requires_exact_archive_route_counts": (
            '"archive_calibration": 6' in assembler_text
            and '"corrected_archive_low": 28' in assembler_text
            and '"corrected_archive_D0": 52' in assembler_text
        ),
        "assembler_requires_4770_matrices": (
            "EXPECTED_MATRIX_FILES = EXPECTED_SUBJECTS * len(MATRIX_NAMES)"
            in assembler_text
        ),
    }
    if (
        route_errors
        or atlas_errors
        or observed_lanes != expected_lanes
        or len(archive_topology) != EXPECTED_PRODUCTION_ARCHIVE_N
        or not all(implementation_markers.values())
        or hroi.get("unit_n") != 530
        or hroi.get("cohort_uniform") is not True
        or hroi.get("policy", {}).get("application")
        != "same diagnosis-blind construction for all 530"
        or topology.get("production_execution_n") != 515
        or topology.get("cohort_uniform") is not True
    ):
        raise ValueError("archive primary-compatibility evidence differs")

    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_primary_compatibility_audit"
        ),
        "generated_utc": utc_now(),
        "status": "PASS_EXISTING_ARCHIVE_NOT_PRIMARY_RELEASE_COMPATIBLE",
        "decision": (
            "REBUILD_ALL_86_ARCHIVE_UNITS_WITH_THE_UNIFORM_BALANCED_"
            "FINAL_HROI_ACT_RECIPE"
        ),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "imaging_executed": False,
        "matrices_modified": False,
        "non_overwriting": True,
        "density_target": DENSITY_TARGET,
        "archive_existing_n": len(units),
        "archive_existing_density_qualified_n": len(density_qualified),
        "archive_existing_density_qualified_fraction": (
            len(density_qualified) / len(units)
        ),
        "historical_route": "NO_ACT_RESTORED_TRACK",
        "historical_route_n": len(units),
        "historical_atlas_source": "ORIGINAL_PRE_FINAL_HROI_SOURCE",
        "historical_atlas_source_n": len(units),
        "final_hroi_changed_voxel_subject_n": sum(
            value > 0 for value in changed_voxel_counts
        ),
        "final_hroi_changed_voxel_min": min(changed_voxel_counts),
        "final_hroi_changed_voxel_median": sorted(changed_voxel_counts)[
            len(changed_voxel_counts) // 2
        ],
        "final_hroi_changed_voxel_max": max(changed_voxel_counts),
        "balanced_archive_production_n": len(archive_topology),
        "balanced_archive_lane_counts": observed_lanes,
        "implementation_markers": implementation_markers,
        "scientific_interpretation": {
            "density_is_necessary_not_sufficient": True,
            "existing_archive_matrices_are_retained_as_evidence": True,
            "existing_archive_matrices_are_primary_release_inputs": False,
            "reason": (
                "The historical matrices use a no-ACT tractography route and "
                "the original atlas source, whereas the primary release uses "
                "two deterministic ACT replicates, refitted SIFT2, all nine "
                "matched matrices and the cohort-uniform final HROI atlas."
            ),
        },
        "records": {
            "archive_inputs": file_record(ARCHIVE_INPUTS),
            "archive_restore": file_record(ARCHIVE_RESTORE),
            "hroi_policy": file_record(HROI_POLICY),
            "topology": file_record(TOPOLOGY),
            "production_runner": file_record(PRODUCTION_RUNNER),
            "assembler": file_record(ASSEMBLER),
            "implementation": file_record(Path(__file__)),
        },
    }
    return result


def main() -> int:
    result = build()
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
