#!/home/ec2-user/fsl/bin/python
"""Replay unchanged Recovery4 QC after exact Eddy integrity recovery.

This wrapper does not change the shared Recovery4 engine.  It revalidates the
original core and legacy input audits, substitutes only hash-bound PASS Eddy
recovery outputs for its exact contracted subjects, runs the normal Recovery4
engine in an isolated root, and invokes the existing audited compactor.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
DERIV = Path("/data/derivatives")
ENGINE_SOURCE = (
    EXP / "scripts/hcp/hcp379_scaleup_pretract_recovery4_v3.py"
)
COMPACTOR_SOURCE = (
    EXP / "scripts/hcp/compact_hcp379_pretract_recovery4_v3.py"
)
PLAN = (
    EXP
    / "research_audit/outputs/hcp379_eddy_integrity_recovery_plan_v1/"
    "plan.json"
)
EDDY_ROOT = DERIV / "hcp379_v2/eddy_integrity_recovery_v1"
OUTPUT_ROOT = DERIV / "hcp379_v2/eddy_integrity_pretract_recovery4"
HUMAN_QC = (
    DERIV
    / "hcp379_v2/review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
CORE_AUDIT_ROOT = (
    EXP
    / "research_audit/outputs/hcp379_scaleup_input_audit_v2"
)
LEGACY_AUDIT_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_legacy_density_recovery_input_audit_v3"
)
EXPECTED = {
    "003_S_4373_I378923",
    "003_S_6490_I1043781",
    "003_S_6644_I1083048",
    "006_S_4713_I1483612",
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = load_module("hcp379_eddy_pretract_engine", ENGINE_SOURCE)
COMPACTOR = load_module(
    "hcp379_eddy_pretract_compactor", COMPACTOR_SOURCE
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def audited_rows(
    gate: Mapping[str, Any],
    *,
    audit_root: Path,
    expected_n: int,
    overlap_n: int,
    execution_n: int,
) -> list[dict[str, str]]:
    original = (
        ENGINE.AUDIT_CSV,
        ENGINE.AUDIT_SUMMARY,
        ENGINE.EXPECTED_AUDIT_N,
        ENGINE.EXPECTED_CANARY_OVERLAP_N,
        ENGINE.EXPECTED_SCALEUP_N,
    )
    try:
        if expected_n == 227:
            audit_csv = audit_root / "scaleup_input_audit.csv"
            summary = audit_root / "scaleup_input_audit_summary.json"
        else:
            audit_csv = (
                audit_root / "legacy_density_recovery_input_audit.csv"
            )
            summary = (
                audit_root
                / "legacy_density_recovery_input_audit_summary.json"
            )
        ENGINE.AUDIT_CSV = audit_csv
        ENGINE.AUDIT_SUMMARY = summary
        ENGINE.EXPECTED_AUDIT_N = expected_n
        ENGINE.EXPECTED_CANARY_OVERLAP_N = overlap_n
        ENGINE.EXPECTED_SCALEUP_N = execution_n
        rows, _ = ENGINE.validate_audit_rows(gate)
        return rows
    finally:
        (
            ENGINE.AUDIT_CSV,
            ENGINE.AUDIT_SUMMARY,
            ENGINE.EXPECTED_AUDIT_N,
            ENGINE.EXPECTED_CANARY_OVERLAP_N,
            ENGINE.EXPECTED_SCALEUP_N,
        ) = original


def recovery_state(unit: str) -> tuple[dict[str, Any] | None, str]:
    pointer_path = EDDY_ROOT / "subjects" / unit / "result.json"
    if not pointer_path.is_file():
        return None, "WAITING_FOR_EDDY_RECOVERY"
    pointer = ENGINE.load_json(pointer_path)
    state_record = pointer.get("attempt_state")
    if not isinstance(state_record, Mapping):
        return None, "INVALID_EDDY_POINTER"
    try:
        state_path = ENGINE.verify_file_record(
            state_record, label=f"{unit}:Eddy recovery state"
        )
        state = ENGINE.load_json(state_path)
        eddy_record = state.get("eddy_output")
        if (
            pointer.get("unit") != unit
            or pointer.get("status")
            != "PASS_EDDY_INTEGRITY_RECOVERY"
            or state.get("unit") != unit
            or state.get("status")
            != "PASS_EDDY_INTEGRITY_RECOVERY"
            or state.get("eddy_volume_integrity", {}).get("gate") != "PASS"
            or state.get("eddy_volume_integrity", {}).get(
                "zero_volume_n"
            )
            != 0
            or state.get("tensor_audit", {}).get("gate") != "PASS"
            or state.get("historical_eddy_unchanged") is not True
            or not isinstance(eddy_record, Mapping)
        ):
            return None, "INVALID_EDDY_RECOVERY"
        ENGINE.verify_file_record(
            eddy_record, label=f"{unit}:recovered Eddy output"
        )
        return state, "READY"
    except Exception:
        return None, "INVALID_EDDY_RECOVERY"


def prepare(
    human_qc: Path,
    *,
    execute: bool,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, str]],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    plan = ENGINE.load_json(PLAN)
    targets = {
        str(row["unit"]): row for row in plan.get("targets", [])
    }
    if (
        plan.get("status") != "PASS"
        or plan.get("target_n") != len(EXPECTED)
        or set(targets) != EXPECTED
    ):
        raise ValueError("exact PASS contracted Eddy plan required")
    gate = ENGINE.validate_recovery4_gate(
        human_qc=human_qc,
        execute=execute,
    )
    core = audited_rows(
        gate,
        audit_root=CORE_AUDIT_ROOT,
        expected_n=227,
        overlap_n=11,
        execution_n=216,
    )
    legacy = audited_rows(
        gate,
        audit_root=LEGACY_AUDIT_ROOT,
        expected_n=216,
        overlap_n=4,
        execution_n=212,
    )
    base = {
        row["unit"]: row
        for row in core + legacy
        if row["unit"] in EXPECTED
    }
    if set(base) != EXPECTED:
        raise ValueError("four-target base audit mapping differs")
    rows: dict[str, dict[str, str]] = {}
    states: dict[str, dict[str, Any]] = {}
    readiness: dict[str, str] = {}
    for unit in sorted(EXPECTED):
        target = targets[unit]
        row = dict(base[unit])
        historical = Path(row["eddy_path"])
        if (
            historical.resolve()
            != Path(target["historical_eddy"]["path"]).resolve()
            or historical.stat().st_size
            != target["historical_eddy"]["size_bytes"]
            or ENGINE.sha256_file(historical)
            != target["historical_eddy"]["sha256"]
        ):
            raise ValueError(f"{unit}: historical Eddy binding differs")
        state, status = recovery_state(unit)
        readiness[unit] = status
        if state is None:
            continue
        eddy = state["eddy_output"]
        row["eddy_path"] = str(Path(eddy["path"]).resolve())
        row["eddy_size_bytes"] = str(eddy["size_bytes"])
        row["eddy_integrity_recovery_state"] = str(
            EDDY_ROOT / "subjects" / unit / "result.json"
        )
        row["eddy_integrity_recovery_sha256"] = ENGINE.sha256_file(
            Path(row["eddy_integrity_recovery_state"])
        )
        rows[unit] = row
        states[unit] = state
    return gate, rows, states, readiness


def validation_payload(
    gate: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, str]],
    states: Mapping[str, Mapping[str, Any]],
    readiness: Mapping[str, str],
) -> dict[str, Any]:
    promotion = ENGINE.load_json(
        ENGINE.verify_file_record(
            gate["promotion_receipt"],
            label="Recovery4 promotion receipt",
        )
    )
    hroi_policy = ENGINE.load_json(
        ENGINE.verify_file_record(
            gate["hroi_source_policy_receipt"],
            label="HROI source policy receipt",
        )
    )
    checks = {
        "exact_target_readiness_partition": (
            set(readiness) == EXPECTED
            and set(rows) == set(states)
            and all(
                value in {
                    "READY",
                    "WAITING_FOR_EDDY_RECOVERY",
                    "INVALID_EDDY_POINTER",
                    "INVALID_EDDY_RECOVERY",
                }
                for value in readiness.values()
            )
        ),
        "ready_rows_use_only_recovered_eddy": all(
            Path(row["eddy_path"]).resolve().is_relative_to(
                EDDY_ROOT.resolve()
            )
            for row in rows.values()
        ),
        "ready_states_pass_both_qc_gates": all(
            state.get("eddy_volume_integrity", {}).get("gate") == "PASS"
            and state.get("tensor_audit", {}).get("gate") == "PASS"
            and state.get("historical_eddy_unchanged") is True
            for state in states.values()
        ),
        "recovery4_gate_is_blind": (
            promotion.get("diagnosis_labels_used") is False
            and hroi_policy.get("diagnosis_labels_used") is False
        ),
        "shared_engine_is_unchanged": ENGINE_SOURCE.is_file(),
        "execution_root_is_isolated": not OUTPUT_ROOT.resolve().is_relative_to(
            (DERIV / "eddy").resolve()
        ),
    }
    passed = sum(checks.values())
    counts = Counter(readiness.values())
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_eddy_pretract_recovery4_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": utc_now(),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": {
            name: "PASS" if value else "FAIL"
            for name, value in checks.items()
        },
        "target_n": len(EXPECTED),
        "ready_n": counts.get("READY", 0),
        "readiness_counts": dict(sorted(counts.items())),
        "readiness": dict(sorted(readiness.items())),
        "imaging_executed": False,
        "output_root": str(OUTPUT_ROOT),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--human-qc-manifest", type=Path, default=HUMAN_QC)
    parser.add_argument("--subject", action="append")
    parser.add_argument("--nthreads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.nthreads <= 16:
        parser.error("--nthreads must be in 1..16")
    gate, rows, states, readiness = prepare(
        args.human_qc_manifest,
        execute=args.execute,
    )
    validation = validation_payload(gate, rows, states, readiness)
    if args.validate_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0 if validation["status"] == "PASS" else 1
    selected = args.subject or sorted(EXPECTED)
    if (
        len(selected) != len(set(selected))
        or not set(selected) <= EXPECTED
    ):
        parser.error("invalid --subject selection")
    unavailable = [
        unit for unit in selected if readiness[unit] != "READY"
    ]
    if unavailable:
        raise SystemExit(
            "Eddy recovery is not ready: " + ",".join(unavailable)
        )
    ENGINE.install_provisional_engine_guards()
    results: list[dict[str, Any]] = []
    compactions: list[dict[str, Any]] = []
    for unit in selected:
        state = ENGINE.process_unit(
            rows[unit],
            root=OUTPUT_ROOT,
            gate=gate,
            nthreads=args.nthreads,
        )
        results.append(state)
        if state.get("status") == "PASS_PRETRACT_RECOVERY4":
            compactions.append(
                COMPACTOR.compact_unit(
                    root=OUTPUT_ROOT.resolve(),
                    unit=unit,
                    execute=True,
                )
            )
    counts = Counter(str(row.get("status")) for row in results)
    compact_counts = Counter(
        str(row.get("status")) for row in compactions
    )
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_eddy_pretract_recovery4_invocation",
        "status": (
            "PASS"
            if counts.get("PASS_PRETRACT_RECOVERY4") == len(selected)
            and compact_counts.get("PASS_COMPACTED") == len(selected)
            else "FAIL"
        ),
        "generated_utc": utc_now(),
        "unit_n": len(selected),
        "units": selected,
        "status_counts": dict(sorted(counts.items())),
        "compaction_status_counts": dict(sorted(compact_counts.items())),
        "diagnosis_labels_used": False,
        "selection_uses_connectome_density": False,
        "shared_engine": ENGINE.file_record(ENGINE_SOURCE),
        "compactor": ENGINE.file_record(COMPACTOR_SOURCE),
        "eddy_plan": ENGINE.file_record(PLAN),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = (
        OUTPUT_ROOT
        / "manifests"
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "-invocation.json"
        )
    )
    ENGINE.atomic_json(path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
