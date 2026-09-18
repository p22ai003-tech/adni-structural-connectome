#!/usr/bin/env python3
"""Build the exact 15-canary plus 515-production balanced-release topology.

This is a diagnosis-blind, non-imaging inventory.  It reconciles the six
corrected Recovery4 pretract lanes with the fixed 15 HROI canaries and the
authoritative 530-unit HROI source policy.  It reports live pretract readiness
but cannot authorize tractography, select a streamline count, or release a
matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_balanced_release_topology_v1/topology.json"
)
HROI_POLICY = (
    HCP_ROOT
    / "source_label_repair_v1/hroi_surface_support_cohort_v1/"
    "policy_adoption_v1.json"
)
HROI_ATLAS_SUMMARY = (
    HCP_ROOT
    / "source_label_repair_v1/hroi_canary_atlas_v1/attempts/"
    "20260728T081233.507961Z-hroi-canary-atlas/summary.json"
)
PRETRACT_RECOVERY_OVERRIDES = (
    EXP
    / "research_audit/outputs/"
    "hcp379_pretract_recovery_overrides_v1/overrides.json"
)
EDDY_INTEGRITY_ROWS = (
    HCP_ROOT / "audits/eddy_volume_integrity_v1/rows.csv"
)
EDDY_INTEGRITY_SUMMARY = EDDY_INTEGRITY_ROWS.with_name("summary.json")
EXPECTED_TOTAL_N = 530
EXPECTED_CANARY_N = 15
EXPECTED_PRODUCTION_N = 515

LANES = {
    "corrected_core": {
        "csv": (
            EXP
            / "research_audit/outputs/hcp379_scaleup_input_audit_v2/"
            "scaleup_input_audit.csv"
        ),
        "root": HCP_ROOT / "corrected_scaleup_recovery4",
        "source_n": 227,
        "production_n": 216,
        "exclude_canaries": True,
    },
    "corrected_legacy_tensor": {
        "csv": (
            EXP
            / "research_audit/outputs/"
            "hcp379_legacy_density_recovery_input_audit_v3/"
            "legacy_density_recovery_input_audit.csv"
        ),
        "root": HCP_ROOT / "corrected_legacy_tensor_recovery4",
        "source_n": 216,
        "production_n": 212,
        "exclude_canaries": True,
    },
    "archive_calibration": {
        "csv": (
            EXP
            / "research_audit/outputs/"
            "hcp379_archive_corrected_input_audit_v2/"
            "archive_calibration_input_audit.csv"
        ),
        "root": HCP_ROOT / "archive_corrected_calibration_recovery4",
        "source_n": 6,
        "production_n": 6,
        "exclude_canaries": False,
    },
    "corrected_archive_low": {
        "csv": (
            EXP
            / "research_audit/outputs/hcp379_archive_low_density_subset_v3/"
            "archive_low_density_execution_28.csv"
        ),
        "root": HCP_ROOT / "corrected_archive_low_recovery4",
        "source_n": 28,
        "production_n": 28,
        "exclude_canaries": False,
    },
    "corrected_archive_D0": {
        "csv": (
            EXP
            / "research_audit/outputs/"
            "hcp379_archive_D0_contingency_subset_v3/"
            "archive_D0_contingency_execution_52.csv"
        ),
        "root": HCP_ROOT / "corrected_archive_D0_recovery4",
        "source_n": 52,
        "production_n": 52,
        "exclude_canaries": False,
    },
    "corrected_freesurfer": {
        "csv": (
            EXP
            / "research_audit/outputs/hcp379_freesurfer_corrected_input_v3/"
            "freesurfer_corrected_input.csv"
        ),
        "root": HCP_ROOT / "corrected_freesurfer_recovery4",
        "source_n": 1,
        "production_n": 1,
        "exclude_canaries": False,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
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


def source_contract() -> tuple[dict[str, Any], set[str]]:
    policy = load_json(HROI_POLICY)
    units = set(policy.get("units", {}))
    if (
        policy.get("record_type")
        != "diagnosis_blind_hcp379_hroi_surface_support_policy_adoption"
        or policy.get("status")
        != "PASS_OPERATIONAL_PRETRACT_ADOPTION_FINAL_HUMAN_QC_PENDING"
        or policy.get("diagnosis_labels_used") is not False
        or policy.get("cohort_uniform") is not True
        or policy.get("per_subject_policy_tuning_allowed") is not False
        or policy.get("pretract_use_authorized") is not True
        or policy.get("selected_count_tractography_authorized") is not False
        or policy.get("final_release_authorized") is not False
        or policy.get("unit_n") != EXPECTED_TOTAL_N
        or len(units) != EXPECTED_TOTAL_N
    ):
        raise ValueError("authoritative 530-unit HROI policy differs")
    return policy, units


def canary_contract() -> tuple[dict[str, Any], set[str]]:
    summary = load_json(HROI_ATLAS_SUMMARY)
    rows = summary.get("units", [])
    units = {
        str(row.get("unit"))
        for row in rows
        if isinstance(row, Mapping)
    }
    if (
        summary.get("record_type") != "hcp379_hroi_canary_atlas_summary"
        or summary.get("status") != "PASS_HROI_CANARY_ATLAS_15"
        or summary.get("diagnosis_labels_used") is not False
        or summary.get("non_overwriting") is not True
        or len(rows) != EXPECTED_CANARY_N
        or len(units) != EXPECTED_CANARY_N
        or any(
            row.get("status") != "PASS_HROI_CANARY_ATLAS"
            or row.get("labels_found") != 379
            for row in rows
        )
    ):
        raise ValueError("fixed HROI canary contract differs")
    return summary, units


def hex64(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def recovery_overrides(policy_units: set[str]) -> dict[str, Path]:
    if not PRETRACT_RECOVERY_OVERRIDES.is_file():
        return {}
    value = load_json(PRETRACT_RECOVERY_OVERRIDES)
    raw = value.get("overrides")
    if (
        value.get("record_type") != "hcp379_pretract_recovery_overrides"
        or value.get("status") != "PASS"
        or value.get("diagnosis_labels_used") is not False
        or value.get("outcomes_used") is not False
        or value.get("connectome_density_used") is not False
        or value.get("non_overwriting") is not True
        or not isinstance(raw, Mapping)
        or value.get("override_n") != len(raw)
        or not set(raw).issubset(policy_units)
    ):
        raise ValueError("pretract recovery override contract differs")
    roots: dict[str, Path] = {}
    for unit, record in raw.items():
        if not isinstance(record, Mapping) or record.get("unit") != unit:
            raise ValueError(f"invalid pretract override: {unit}")
        root = Path(str(record.get("root", ""))).resolve()
        state = Path(str(record.get("state", {}).get("path", ""))).resolve()
        compaction = Path(
            str(record.get("compaction", {}).get("path", ""))
        ).resolve()
        if (
            record.get("recovery_class")
            not in {
                "MASK_UNION_RECOVERY_V1",
                "EDDY_INTEGRITY_RECOVERY_V1",
            }
            or file_record(state) != record.get("state")
            or file_record(compaction) != record.get("compaction")
            or state
            != root / "qc/subjects" / f"{unit}.json"
            or compaction
            != root / "qc/pretract_compaction" / f"{unit}.json"
            or not pretract_status(root, unit)["ready"]
        ):
            raise ValueError(f"pretract override replay failed: {unit}")
        roots[str(unit)] = root
    return roots


def eddy_integrity_holds(policy_units: set[str]) -> set[str]:
    summary = load_json(EDDY_INTEGRITY_SUMMARY)
    rows = read_csv(EDDY_INTEGRITY_ROWS)
    flagged = {
        str(row["unit"])
        for row in rows
        if row.get("status") == "FAIL_ZERO_EDDY_VOLUMES"
    }
    if (
        summary.get("record_type")
        != "diagnosis_blind_hcp379_eddy_volume_integrity_audit"
        or summary.get("status") not in {"PASS", "RECOVERY_REQUIRED"}
        or summary.get("diagnosis_labels_used") is not False
        or summary.get("outcomes_used") is not False
        or summary.get("connectome_density_used") is not False
        or summary.get("target_n") != EXPECTED_PRODUCTION_N
        or summary.get("audited_n") != EXPECTED_PRODUCTION_N
        or len(rows) != EXPECTED_PRODUCTION_N
        or len({str(row["unit"]) for row in rows})
        != EXPECTED_PRODUCTION_N
        or summary.get("flagged_n") != len(flagged)
        or flagged != set(summary.get("flagged_units", []))
        or not flagged.issubset(policy_units)
    ):
        raise ValueError("complete Eddy integrity hold contract differs")
    return flagged


def pretract_status(root: Path, unit: str) -> dict[str, Any]:
    state_path = root / "qc/subjects" / f"{unit}.json"
    compaction_path = (
        root / "qc/pretract_compaction" / f"{unit}.json"
    )
    state_status = "MISSING"
    state_valid = False
    if state_path.is_file() and not state_path.is_symlink():
        try:
            state = load_json(state_path)
            state_status = str(state.get("status", "UNKNOWN"))
            state_valid = bool(
                state.get("record_type")
                == "diagnosis_blind_hcp379_scaleup_pretract_recovery4"
                and state.get("status") == "PASS_PRETRACT_RECOVERY4"
                and state.get("diagnosis_labels_used") is False
                and state.get("unit") == unit
                and state.get("raw_tensor_maps_preserved") is True
                and state.get("bounded_tensor_maps_separate") is True
            )
        except Exception as exc:
            state_status = f"UNREADABLE:{type(exc).__name__}"
    compaction_status = "MISSING"
    compaction_valid = False
    if compaction_path.is_file() and not compaction_path.is_symlink():
        try:
            compaction = load_json(compaction_path)
            compaction_status = str(compaction.get("status", "UNKNOWN"))
            compaction_valid = bool(
                compaction.get("record_type")
                == "diagnosis_blind_hcp379_pretract_compaction"
                and compaction.get("status") == "PASS_COMPACTED"
                and compaction.get("diagnosis_labels_used") is False
                and compaction.get("unit") == unit
                and Path(str(compaction.get("root", ""))).resolve()
                == root.resolve()
                and compaction.get("recovery4_inputs") is True
            )
        except Exception as exc:
            compaction_status = f"UNREADABLE:{type(exc).__name__}"
    return {
        "ready": state_valid and compaction_valid,
        "state_status": state_status,
        "compaction_status": compaction_status,
        "state_path": str(state_path.resolve()),
        "compaction_path": str(compaction_path.resolve()),
    }


def production_pretract_status(
    root: Path,
    unit: str,
    *,
    overrides: Mapping[str, Path],
    integrity_holds: set[str],
) -> dict[str, Any]:
    value = pretract_status(root, unit)
    held = unit in integrity_holds and unit not in overrides
    if held:
        value = {
            **value,
            "ready_before_eddy_integrity_hold": value["ready"],
            "ready": False,
            "eddy_integrity_hold": True,
            "eddy_integrity_status": "FAIL_ZERO_EDDY_VOLUMES",
        }
    else:
        value["eddy_integrity_hold"] = False
        value["eddy_integrity_status"] = "PASS_OR_RECOVERED"
    return value


def build() -> dict[str, Any]:
    policy, policy_units = source_contract()
    atlas_summary, canary_units = canary_contract()
    overrides = recovery_overrides(policy_units)
    integrity_holds = eddy_integrity_holds(policy_units)
    production: list[dict[str, Any]] = []
    lane_rows: dict[str, list[dict[str, str]]] = {}
    lane_counts: dict[str, Any] = {}
    canary_source_lanes: dict[str, list[str]] = {
        unit: [] for unit in canary_units
    }
    for lane, contract in LANES.items():
        rows = read_csv(contract["csv"])
        row_units = [row.get("unit", "") for row in rows]
        if (
            len(rows) != contract["source_n"]
            or len(set(row_units)) != contract["source_n"]
            or any(not unit for unit in row_units)
        ):
            raise ValueError(f"{lane}: source row set differs")
        for unit in canary_units & set(row_units):
            canary_source_lanes[unit].append(lane)
        selected = [
            row
            for row in rows
            if not (
                contract["exclude_canaries"]
                and row["unit"] in canary_units
            )
        ]
        if len(selected) != contract["production_n"]:
            raise ValueError(f"{lane}: production row count differs")
        lane_rows[lane] = selected
        for row in selected:
            if (
                row.get("diagnosis_labels_used") != "False"
                or not hex64(row.get("dti_source_id", ""))
                or not hex64(row.get("dti_raw_bundle_sha256", ""))
            ):
                raise ValueError(
                    f"{lane}/{row.get('unit')}: identity binding differs"
                )
            readiness_root = overrides.get(
                row["unit"], contract["root"]
            )
            readiness = production_pretract_status(
                readiness_root,
                row["unit"],
                overrides=overrides,
                integrity_holds=integrity_holds,
            )
            production.append(
                {
                    "unit": row["unit"],
                    "lane": lane,
                    "role": "BALANCED_PRODUCTION_EXECUTION",
                    "dti_source_id": row["dti_source_id"],
                    "dti_raw_bundle_sha256": row[
                        "dti_raw_bundle_sha256"
                    ],
                    "pretract_root": str(readiness_root.resolve()),
                    "pretract_recovery_override": (
                        row["unit"] in overrides
                    ),
                    "pretract": readiness,
                }
            )
        lane_counts[lane] = {
            "source_n": len(rows),
            "canary_overlap_excluded_n": len(rows) - len(selected),
            "production_n": len(selected),
            "pretract_ready_n": sum(
                production_pretract_status(
                    overrides.get(row["unit"], contract["root"]),
                    row["unit"],
                    overrides=overrides,
                    integrity_holds=integrity_holds,
                )["ready"]
                for row in selected
            ),
        }

    production_units = [row["unit"] for row in production]
    if (
        len(production) != EXPECTED_PRODUCTION_N
        or len(set(production_units)) != EXPECTED_PRODUCTION_N
        or set(production_units) & canary_units
        or set(production_units) | canary_units != policy_units
    ):
        raise ValueError("15+515 topology does not equal exact policy 530")
    production.sort(key=lambda row: row["unit"])
    ready_n = sum(row["pretract"]["ready"] for row in production)
    invalid_canary_sources = {
        unit: lanes
        for unit, lanes in canary_source_lanes.items()
        if len(lanes) != 1
    }
    if invalid_canary_sources:
        raise ValueError(
            "canary source-recovery lane is not unique: "
            f"{invalid_canary_sources}"
        )
    canary_rows = [
        {
            "unit": unit,
            "lane": "hroi_all_nine_canary",
            "source_recovery_lane": canary_source_lanes[unit][0],
            "role": "VALIDATED_CANARY_REUSE_AFTER_RUNTIME_PASS",
        }
        for unit in sorted(canary_units)
    ]
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_balanced_release_topology"
        ),
        "generated_utc": utc_now(),
        "status": (
            "READY_FOR_BALANCED_GATE_NOT_EXECUTION"
            if ready_n == EXPECTED_PRODUCTION_N
            else "PRETRACT_IN_PROGRESS"
        ),
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "cohort_uniform": True,
        "total_unit_n": EXPECTED_TOTAL_N,
        "canary_reuse_n": len(canary_rows),
        "production_execution_n": len(production),
        "pretract_ready_n": ready_n,
        "pretract_waiting_n": EXPECTED_PRODUCTION_N - ready_n,
        "exact_15_plus_515_equals_530": True,
        "density_target": 0.60,
        "balanced_total_streamline_count": None,
        "balanced_per_seed_streamline_count": None,
        "selected_count_tractography_authorized": False,
        "final_release_authorized": False,
        "requires_complete_hroi_density_cohort": True,
        "requires_complete_hroi_all_nine_canary": True,
        "requires_final_hroi_human_review": True,
        "per_subject_streamline_tuning_allowed": False,
        "subject_exclusion_for_low_density_allowed": False,
        "lane_counts": lane_counts,
        "canaries": canary_rows,
        "production": production,
        "records": {
            "hroi_policy": file_record(HROI_POLICY),
            "hroi_atlas_summary": file_record(HROI_ATLAS_SUMMARY),
            "pretract_recovery_overrides": (
                file_record(PRETRACT_RECOVERY_OVERRIDES)
                if PRETRACT_RECOVERY_OVERRIDES.is_file()
                else None
            ),
            "eddy_integrity_rows": file_record(EDDY_INTEGRITY_ROWS),
            "eddy_integrity_summary": file_record(
                EDDY_INTEGRITY_SUMMARY
            ),
            "lane_sources": {
                lane: file_record(contract["csv"])
                for lane, contract in LANES.items()
            },
            "implementation": file_record(Path(__file__)),
        },
    }


def self_test() -> dict[str, Any]:
    checks = {
        "expected_partition": EXPECTED_CANARY_N + EXPECTED_PRODUCTION_N
        == EXPECTED_TOTAL_N,
        "six_production_lanes": len(LANES) == 6,
        "lane_total": sum(
            int(contract["production_n"])
            for contract in LANES.values()
        )
        == EXPECTED_PRODUCTION_N,
        "density_target": 0.60 == 0.60,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        value = self_test()
        print(json.dumps(value, sort_keys=True))
        return 0 if value["status"] == "PASS" else 1
    value = build()
    if not args.no_write:
        atomic_json(args.output.resolve(), value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "total_unit_n": value["total_unit_n"],
                "canary_reuse_n": value["canary_reuse_n"],
                "production_execution_n": value[
                    "production_execution_n"
                ],
                "pretract_ready_n": value["pretract_ready_n"],
                "pretract_waiting_n": value["pretract_waiting_n"],
                "lane_counts": value["lane_counts"],
                "output": (
                    str(args.output.resolve()) if not args.no_write else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
