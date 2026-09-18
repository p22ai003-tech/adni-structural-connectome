#!/usr/bin/env python3
"""Independently validate the exact HCP379 balanced-release topology."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP / "scripts/hcp/build_hcp379_balanced_release_topology_v1.py"
)
TOPOLOGY = (
    EXP
    / "research_audit/outputs/"
    "hcp379_balanced_release_topology_v1/topology.json"
)
OUTPUT = TOPOLOGY.with_name("validation.json")
PYTHON = Path("/home/ec2-user/fsl/bin/python")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_module(SOURCE, "hcp379_balanced_topology_validator_builder")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


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


def verified_record(value: Mapping[str, Any], label: str) -> Path:
    path = Path(str(value.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or dict(value) != record(path)
    ):
        raise ValueError(f"{label}: record differs")
    return path


def read_units(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return [str(row.get("unit", "")) for row in rows]


def main() -> int:
    checks: dict[str, Any] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    compile_probe = subprocess.run(
        [str(PYTHON), "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "source_compiles",
        compile_probe.returncode == 0,
        {
            "returncode": compile_probe.returncode,
            "stderr": compile_probe.stderr[-2000:],
        },
    )
    self_probe = subprocess.run(
        [str(PYTHON), str(SOURCE), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        self_value = json.loads(self_probe.stdout)
    except Exception:
        self_value = {}
    check(
        "self_test_passes",
        self_probe.returncode == 0 and self_value.get("status") == "PASS",
        self_value or {"stderr": self_probe.stderr[-2000:]},
    )

    value = load_json(TOPOLOGY)
    canary_rows = [
        row
        for row in value.get("canaries", [])
        if isinstance(row, Mapping)
    ]
    canaries = {
        str(row.get("unit", ""))
        for row in canary_rows
    }
    production_rows = [
        row
        for row in value.get("production", [])
        if isinstance(row, Mapping)
    ]
    production = {str(row.get("unit", "")) for row in production_rows}
    policy = load_json(BUILDER.HROI_POLICY)
    policy_units = set(policy.get("units", {}))
    check(
        "exact_15_plus_515_equals_policy_530",
        (
            value.get("total_unit_n") == BUILDER.EXPECTED_TOTAL_N
            and value.get("canary_reuse_n") == BUILDER.EXPECTED_CANARY_N
            and value.get("production_execution_n")
            == BUILDER.EXPECTED_PRODUCTION_N
            and len(canaries) == BUILDER.EXPECTED_CANARY_N
            and len(production) == BUILDER.EXPECTED_PRODUCTION_N
            and not canaries & production
            and canaries | production == policy_units
            and len(policy_units) == BUILDER.EXPECTED_TOTAL_N
        ),
        {
            "canary_n": len(canaries),
            "production_n": len(production),
            "union_n": len(canaries | production),
            "overlap_n": len(canaries & production),
            "policy_n": len(policy_units),
        },
    )
    expected_lane_sets: dict[str, set[str]] = {}
    expected_canary_sources: dict[str, list[str]] = {
        unit: [] for unit in canaries
    }
    lane_source_records_ok = True
    for lane, contract in BUILDER.LANES.items():
        source_path = contract["csv"].resolve()
        source_units = read_units(source_path)
        for unit in canaries & set(source_units):
            expected_canary_sources[unit].append(lane)
        selected = {
            unit
            for unit in source_units
            if not (
                contract["exclude_canaries"] and unit in canaries
            )
        }
        expected_lane_sets[lane] = selected
        try:
            verified_record(
                value["records"]["lane_sources"][lane],
                f"{lane}/source",
            )
        except Exception:
            lane_source_records_ok = False
    observed_lane_sets = {
        lane: {
            str(row["unit"])
            for row in production_rows
            if row.get("lane") == lane
        }
        for lane in BUILDER.LANES
    }
    check(
        "six_lane_identity_sets_replay",
        (
            expected_lane_sets == observed_lane_sets
            and lane_source_records_ok
            and sum(map(len, expected_lane_sets.values()))
            == BUILDER.EXPECTED_PRODUCTION_N
        ),
        {
            lane: {
                "expected_n": len(expected_lane_sets[lane]),
                "observed_n": len(observed_lane_sets[lane]),
            }
            for lane in BUILDER.LANES
        },
    )
    override_roots = BUILDER.recovery_overrides(policy_units)
    integrity_holds = BUILDER.eddy_integrity_holds(policy_units)
    observed_overrides = {
        str(row["unit"]): Path(str(row["pretract_root"])).resolve()
        for row in production_rows
        if row.get("pretract_recovery_override") is True
    }
    check(
        "isolated_pretract_recovery_overrides_replay",
        observed_overrides == override_roots
        and all(
            row.get("pretract_recovery_override")
            is (str(row["unit"]) in override_roots)
            for row in production_rows
        ),
        {
            "expected": {
                unit: str(root)
                for unit, root in sorted(override_roots.items())
            },
            "observed": {
                unit: str(root)
                for unit, root in sorted(observed_overrides.items())
            },
        },
    )
    observed_holds = {
        str(row["unit"])
        for row in production_rows
        if row.get("pretract", {}).get("eddy_integrity_hold") is True
    }
    check(
        "zero_volume_eddy_inputs_are_fail_closed",
        observed_holds == integrity_holds - set(override_roots)
        and all(
            row.get("pretract", {}).get("ready") is False
            for row in production_rows
            if str(row["unit"]) in observed_holds
        ),
        {
            "expected_holds": sorted(
                integrity_holds - set(override_roots)
            ),
            "observed_holds": sorted(observed_holds),
        },
    )
    observed_canary_sources = {
        str(row.get("unit", "")): str(
            row.get("source_recovery_lane", "")
        )
        for row in canary_rows
    }
    expected_canary_source_single = {
        unit: lanes[0]
        for unit, lanes in expected_canary_sources.items()
        if len(lanes) == 1
    }
    check(
        "canary_source_recovery_lanes_are_unique_and_replayed",
        (
            len(expected_canary_source_single)
            == BUILDER.EXPECTED_CANARY_N
            and observed_canary_sources
            == expected_canary_source_single
            and all(
                row.get("lane") == "hroi_all_nine_canary"
                and row.get("role")
                == "VALIDATED_CANARY_REUSE_AFTER_RUNTIME_PASS"
                for row in canary_rows
            )
        ),
        {
            "expected": expected_canary_source_single,
            "observed": observed_canary_sources,
        },
    )

    live_ready = 0
    recorded_ready = 0
    recorded_ready_still_valid = True
    for row in production_rows:
        root = Path(str(row["pretract_root"])).resolve()
        live = BUILDER.production_pretract_status(
            root,
            str(row["unit"]),
            overrides=override_roots,
            integrity_holds=integrity_holds,
        )
        live_ready += int(live["ready"])
        recorded = bool(row.get("pretract", {}).get("ready"))
        recorded_ready += int(recorded)
        if recorded and not live["ready"]:
            recorded_ready_still_valid = False
    check(
        "snapshot_pretract_readiness_is_monotonic_and_valid",
        (
            recorded_ready_still_valid
            and recorded_ready == value.get("pretract_ready_n")
            and live_ready >= recorded_ready
            and value.get("pretract_ready_n")
            + value.get("pretract_waiting_n")
            == BUILDER.EXPECTED_PRODUCTION_N
        ),
        {
            "recorded_ready_n": recorded_ready,
            "live_ready_n": live_ready,
            "recorded_ready_still_valid": recorded_ready_still_valid,
        },
    )
    records_ok = True
    record_errors = []
    try:
        if value["records"]["implementation"] != record(SOURCE):
            raise ValueError("implementation binding differs")
        verified_record(value["records"]["hroi_policy"], "HROI policy")
        verified_record(
            value["records"]["hroi_atlas_summary"],
            "HROI atlas summary",
        )
        if BUILDER.PRETRACT_RECOVERY_OVERRIDES.is_file():
            verified_record(
                value["records"]["pretract_recovery_overrides"],
                "pretract recovery overrides",
            )
        verified_record(
            value["records"]["eddy_integrity_rows"],
            "Eddy integrity rows",
        )
        verified_record(
            value["records"]["eddy_integrity_summary"],
            "Eddy integrity summary",
        )
    except Exception as exc:
        records_ok = False
        record_errors.append(f"{type(exc).__name__}:{exc}")
    check(
        "source_and_policy_records_are_exact",
        records_ok,
        {"errors": record_errors},
    )
    identities_ok = all(
        row.get("role") == "BALANCED_PRODUCTION_EXECUTION"
        and BUILDER.hex64(str(row.get("dti_source_id", "")))
        and BUILDER.hex64(str(row.get("dti_raw_bundle_sha256", "")))
        for row in production_rows
    )
    check(
        "production_identity_bindings_are_complete_and_blind",
        (
            identities_ok
            and value.get("diagnosis_labels_used") is False
            and value.get("cohort_uniform") is True
            and value.get("per_subject_streamline_tuning_allowed") is False
            and value.get("subject_exclusion_for_low_density_allowed")
            is False
        ),
        {"production_rows_checked": len(production_rows)},
    )
    check(
        "execution_and_release_remain_fail_closed",
        (
            value.get("balanced_total_streamline_count") is None
            and value.get("balanced_per_seed_streamline_count") is None
            and value.get("selected_count_tractography_authorized") is False
            and value.get("final_release_authorized") is False
            and value.get("requires_complete_hroi_density_cohort") is True
            and value.get("requires_complete_hroi_all_nine_canary") is True
            and value.get("requires_final_hroi_human_review") is True
        ),
        {
            "status": value.get("status"),
            "selected_count_tractography_authorized": value.get(
                "selected_count_tractography_authorized"
            ),
            "final_release_authorized": value.get(
                "final_release_authorized"
            ),
        },
    )
    source_text = SOURCE.read_text(encoding="utf-8")
    prohibited = (
        "tckgen",
        "rm -rf",
        "shutil.rmtree",
        ".unlink(",
        "diagnosis.csv",
        "DX_bl",
        '"selected_count_tractography_authorized": True',
        '"final_release_authorized": True',
    )
    found = [fragment for fragment in prohibited if fragment in source_text]
    check(
        "topology_builder_has_no_imaging_destructive_or_outcome_shortcut",
        not found,
        {"found": found},
    )

    passed = sum(row["status"] == "PASS" for row in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_balanced_release_topology_v1_validation",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "imaging_executed_by_validation": False,
        "topology": record(TOPOLOGY),
        "source": record(SOURCE),
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
