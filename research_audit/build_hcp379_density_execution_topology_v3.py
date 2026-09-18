#!/home/ec2-user/fsl/bin/python
"""Build the exact non-overwriting execution topology for 530 HCP379 units."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
ROOT = Path("/data/derivatives/hcp379_v2")
DENSITY_PLAN = (
    EXP
    / "research_audit/outputs/hcp379_density_recovery_plan_v3/"
    "hcp379_density_recovery_subjects.csv"
)
CORE_AUDIT = (
    EXP
    / "research_audit/outputs/hcp379_scaleup_input_audit_v2/"
    "scaleup_input_audit_summary.json"
)
LEGACY_AUDIT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_legacy_density_recovery_input_audit_v3/"
    "legacy_density_recovery_input_audit_summary.json"
)
ARCHIVE_AUDIT = (
    EXP
    / "research_audit/outputs/hcp379_archive_corrected_input_audit_v2/"
    "archive_corrected_input_audit_summary.json"
)
RECOVERY4_MASTER = (
    ROOT / "pretract_recovery4/pretract_recovery4_manifest.json"
)
ARCHIVE_CALIBRATION = (
    EXP
    / "research_audit/outputs/hcp379_archive_corrected_input_audit_v2/"
    "archive_corrected_calibration_selection.json"
)
OUTPUT_ROOT = (
    EXP / "research_audit/outputs/hcp379_density_execution_topology_v3"
)
OUTPUT_CSV = OUTPUT_ROOT / "hcp379_density_execution_subjects.csv"
OUTPUT_JSON = OUTPUT_ROOT / "hcp379_density_execution_topology.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"required regular file missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


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
        json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "unit",
        "inventory_lane",
        "density_group",
        "current_density",
        "planned_final_route",
        "execution_lane",
        "pretract_action",
        "tractography_action",
        "reuse_source",
        "route_decision_dependency",
        "density_release_requirement",
        "diagnosis_or_outcomes_used",
    ]
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def assert_audits() -> None:
    core = load_json(CORE_AUDIT)
    legacy = load_json(LEGACY_AUDIT)
    archive = load_json(ARCHIVE_AUDIT)
    if (
        core.get("status") != "PASS"
        or core.get("audited_n") != 227
        or legacy.get("status") != "PASS"
        or legacy.get("audited_n") != 216
        or legacy.get("ready_n") != 216
        or archive.get("status") != "PASS"
        or archive.get("audited_n") != 86
        or archive.get("ready_n") != 86
    ):
        raise ValueError("one or more corrected-route input audits are not PASS")


def build() -> dict[str, Any]:
    assert_audits()
    base = read_csv(DENSITY_PLAN)
    if len(base) != 530 or len({row["unit"] for row in base}) != 530:
        raise ValueError("density plan is not exact unique 530")
    by_unit = {row["unit"]: row for row in base}

    recovery = load_json(RECOVERY4_MASTER)
    canaries = {
        str(row["unit"]) for row in recovery.get("units", [])
    }
    if len(canaries) != 15:
        raise ValueError("Recovery4 canary identity is not exact 15")
    archive_selection = load_json(ARCHIVE_CALIBRATION)
    archive_calibrators = {
        str(row["unit"]) for row in archive_selection.get("units", [])
    }
    if len(archive_calibrators) != 6:
        raise ValueError("archive calibration identity is not exact six")

    lanes: dict[str, set[str]] = {
        "corrected_core_227": {
            unit
            for unit, row in by_unit.items()
            if row["inventory_lane"]
            in {"corrected_track_rebuild_116", "hcp_remap_probe_111"}
        },
        "corrected_legacy_tensor_216": {
            unit
            for unit, row in by_unit.items()
            if row["inventory_lane"]
            in {"legacy_keep_214", "tensor_repair_2"}
        },
        "corrected_archive_low_30": {
            unit
            for unit, row in by_unit.items()
            if row["inventory_lane"] == "archive_86"
            and row["density_group"] != "D0_THRESHOLD_QUALIFIED"
        },
        "conditional_archive_D0_56": {
            unit
            for unit, row in by_unit.items()
            if row["inventory_lane"] == "archive_86"
            and row["density_group"] == "D0_THRESHOLD_QUALIFIED"
        },
        "corrected_freesurfer_1": {
            unit
            for unit, row in by_unit.items()
            if row["inventory_lane"] == "fastsurfer_hcp_1"
        },
    }
    expected = {
        "corrected_core_227": 227,
        "corrected_legacy_tensor_216": 216,
        "corrected_archive_low_30": 30,
        "conditional_archive_D0_56": 56,
        "corrected_freesurfer_1": 1,
    }
    if {name: len(units) for name, units in lanes.items()} != expected:
        raise ValueError("execution lane counts differ")
    if set().union(*lanes.values()) != set(by_unit):
        raise ValueError("execution lanes do not cover exact 530")
    for name, left in lanes.items():
        for other, right in lanes.items():
            if name < other and left & right:
                raise ValueError(f"execution lanes overlap: {name}/{other}")

    canary_overlap = {
        name: lanes[name] & canaries
        for name in ("corrected_core_227", "corrected_legacy_tensor_216")
    }
    if {name: len(values) for name, values in canary_overlap.items()} != {
        "corrected_core_227": 11,
        "corrected_legacy_tensor_216": 4,
    }:
        raise ValueError("Phase-B reuse overlap differs")
    archive_low_calibration = (
        lanes["corrected_archive_low_30"] & archive_calibrators
    )
    archive_D0_calibration = (
        lanes["conditional_archive_D0_56"] & archive_calibrators
    )
    if len(archive_low_calibration) != 2 or len(archive_D0_calibration) != 4:
        raise ValueError("archive calibration density split differs")

    lane_by_unit = {
        unit: lane for lane, units in lanes.items() for unit in units
    }
    rows: list[dict[str, Any]] = []
    for unit in sorted(by_unit):
        source = by_unit[unit]
        lane = lane_by_unit[unit]
        if lane == "corrected_core_227":
            reuse = "RECOVERY4_PHASE_B_PRIMARY" if unit in canaries else ""
            pretract = (
                "REUSE_CANARY_PRETRACT"
                if unit in canaries
                else "RUN_CORRECTED_RECOVERY4_FROM_EDDY"
            )
            tract = (
                "REUSE_PHASE_B_SELECTED_PRIMARY"
                if unit in canaries
                else "RUN_PHASE_B_SELECTED_UNIFORM_RECIPE"
            )
            final_route = "CORRECTED_ACT"
            dependency = "PHASE_B_AND_HUMAN_QC"
        elif lane == "corrected_legacy_tensor_216":
            reuse = "RECOVERY4_PHASE_B_PRIMARY" if unit in canaries else ""
            pretract = (
                "REUSE_CANARY_PRETRACT"
                if unit in canaries
                else "RUN_CORRECTED_RECOVERY4_FROM_EDDY"
            )
            tract = (
                "REUSE_PHASE_B_SELECTED_PRIMARY"
                if unit in canaries
                else "RUN_PHASE_B_SELECTED_UNIFORM_RECIPE"
            )
            final_route = "CORRECTED_ACT"
            dependency = "PHASE_B_AND_HUMAN_QC"
        elif lane == "corrected_archive_low_30":
            reuse = (
                "ARCHIVE_CORRECTED_CALIBRATION_PRIMARY"
                if unit in archive_low_calibration
                else ""
            )
            pretract = (
                "REUSE_ARCHIVE_CALIBRATION_PRETRACT"
                if unit in archive_low_calibration
                else "RUN_CORRECTED_RECOVERY4_FROM_EDDY"
            )
            tract = (
                "REUSE_ARCHIVE_CALIBRATION_PRIMARY"
                if unit in archive_low_calibration
                else "RUN_PHASE_B_SELECTED_UNIFORM_RECIPE"
            )
            final_route = "CORRECTED_ACT"
            dependency = "PHASE_B_HUMAN_QC_AND_ARCHIVE_CALIBRATION"
        elif lane == "conditional_archive_D0_56":
            reuse = "HISTORICAL_ARCHIVE_NO_ACT"
            pretract = "NONE_IF_ROUTE_CONCORDANCE_PASS"
            tract = "RETAIN_ONLY_IF_ROUTE_CONCORDANCE_PASS"
            final_route = "ARCHIVE_NO_ACT_OR_CORRECTED_ACT"
            dependency = (
                "ARCHIVE_ROUTE_CONCORDANCE; if FAIL run corrected Recovery4 "
                "for all 56"
            )
        else:
            reuse = ""
            pretract = "COMPLETE_HCP_SOURCE_THEN_CORRECTED_RECOVERY4_FROM_EDDY"
            tract = "RUN_PHASE_B_SELECTED_UNIFORM_RECIPE"
            final_route = "CORRECTED_ACT"
            dependency = "FREESURFER_HCP_REPAIR_PHASE_B_AND_HUMAN_QC"
        rows.append(
            {
                "unit": unit,
                "inventory_lane": source["inventory_lane"],
                "density_group": source["density_group"],
                "current_density": source["current_density"],
                "planned_final_route": final_route,
                "execution_lane": lane,
                "pretract_action": pretract,
                "tractography_action": tract,
                "reuse_source": reuse,
                "route_decision_dependency": dependency,
                "density_release_requirement": "RECOMPUTED_DENSITY>=0.60",
                "diagnosis_or_outcomes_used": False,
            }
        )
    atomic_csv(OUTPUT_CSV, rows)

    topology = {
        "record_type": "diagnosis_blind_hcp379_density_execution_topology",
        "schema_version": "3.0.0",
        "generated_utc": utc_now(),
        "status": "EXECUTION_TOPOLOGY_READY_GATED",
        "diagnosis_or_outcomes_used": False,
        "goal_acceptance": {
            "target_n": 530,
            "minimum_density_inclusive": 0.60,
            "required_pass_n": 530,
            "subject_exclusion_allowed": False,
            "edge_imputation_allowed": False,
            "per_subject_streamline_count_tuning_allowed": False,
        },
        "lanes": {
            "corrected_core_227": {
                "n": 227,
                "phase_b_primary_reuse_n": 11,
                "new_pretract_and_tractography_n": 216,
            },
            "corrected_legacy_tensor_216": {
                "n": 216,
                "current_below_0_60_n": 214,
                "D0_fallback_n": 2,
                "phase_b_primary_reuse_n": 4,
                "new_pretract_and_tractography_n": 212,
                "rationale": (
                    "all legacy/tensor units use one corrected ACT route; only "
                    "one additional D0 subject is newly processed because the "
                    "other D0 subject is already a Phase-B canary"
                ),
            },
            "corrected_archive_low_30": {
                "n": 30,
                "archive_calibration_primary_reuse_n": 2,
                "new_pretract_and_tractography_n": 28,
            },
            "conditional_archive_D0_56": {
                "n": 56,
                "retain_if_concordance_pass_n": 56,
                "corrected_rerun_if_concordance_fail_n": 56,
                "mandatory_route_label_and_sensitivity": True,
            },
            "corrected_freesurfer_1": {
                "n": 1,
                "new_pretract_and_tractography_n": 1,
            },
        },
        "release_scenarios": {
            "archive_concordance_pass": {
                "corrected_ACT_n": 474,
                "retained_archive_no_ACT_n": 56,
                "total_n": 530,
            },
            "archive_concordance_fail": {
                "corrected_ACT_n": 530,
                "retained_archive_no_ACT_n": 0,
                "total_n": 530,
            },
        },
        "execution_gates": [
            "genuine blinded visual QC for all 15 Recovery4 canaries",
            (
                "Phase B selects one uniform 3M/5M/10M count with every "
                "primary and independent canary density>=0.60 plus stability"
            ),
            "FreeSurfer/HCP source repair completes for the one missing unit",
            "archive D0 route concordance decision",
            "per-subject nine-matrix and density>=0.60 QC",
        ],
        "current_input_readiness": {
            "corrected_core_227": "PASS_227",
            "corrected_legacy_tensor_216": "PASS_216",
            "archive_all_86": "PASS_86",
            "freesurfer_1": "IN_PROGRESS",
        },
        "records": {
            "subject_topology": file_record(OUTPUT_CSV),
            "density_recovery_plan": file_record(DENSITY_PLAN),
            "core_input_audit": file_record(CORE_AUDIT),
            "legacy_input_audit": file_record(LEGACY_AUDIT),
            "archive_input_audit": file_record(ARCHIVE_AUDIT),
            "recovery4_master": file_record(RECOVERY4_MASTER),
            "archive_calibration_selection": file_record(
                ARCHIVE_CALIBRATION
            ),
            "builder": file_record(Path(__file__)),
        },
        "density_group_counts": dict(
            sorted(Counter(row["density_group"] for row in rows).items())
        ),
    }
    atomic_json(OUTPUT_JSON, topology)
    return topology


if __name__ == "__main__":
    result = build()
    print(
        json.dumps(
            {
                "status": result["status"],
                "lanes": {
                    name: value["n"]
                    for name, value in result["lanes"].items()
                },
                "release_scenarios": result["release_scenarios"],
                "plan": str(OUTPUT_JSON),
            },
            indent=2,
            sort_keys=True,
        )
    )
