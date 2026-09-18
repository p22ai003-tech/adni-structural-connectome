#!/usr/bin/env python3
"""Validate the non-overwriting 15-canary HROI atlas builder."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "build_hcp379_hroi_canary_atlas_v1.py"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_hroi_canary_atlas_v1/validation.json"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_hroi_canary_atlas_validation_target", SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module()
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if condition else "FAIL",
            "evidence": evidence,
        }

    units, candidates, missing = module.required_candidates()
    check(
        "exact_phase_b_canary_scope",
        len(units) == module.EXPECTED_N
        and len(set(units)) == module.EXPECTED_N
        and set(candidates).isdisjoint(missing)
        and set(candidates).union(missing) == set(units),
        {
            "unit_n": len(units),
            "candidate_n": len(candidates),
            "missing_n": len(missing),
        },
    )
    check(
        "available_candidates_are_hash_bound_and_all_379",
        all(
            len(row["candidate"]["sha256"]) == 64
            and len(row["candidate_manifest"]["sha256"]) == 64
            and len(row["original_source"]["sha256"]) == 64
            and row["changed_voxel_n"] >= 0
            for row in candidates.values()
        ),
        {"candidate_n": len(candidates)},
    )

    attempts_before = (
        {path.name for path in module.ATTEMPTS.iterdir()}
        if module.ATTEMPTS.is_dir()
        else set()
    )
    probe = subprocess.run(
        ["/home/ec2-user/fsl/bin/python", str(SOURCE)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    attempts_after = (
        {path.name for path in module.ATTEMPTS.iterdir()}
        if module.ATTEMPTS.is_dir()
        else set()
    )
    preflight = json.loads(probe.stdout)
    expected_status = (
        "READY_HROI_CANARY_ATLAS_15"
        if not missing
        else "WAITING_FOR_HROI_CANARY_SOURCES"
    )
    check(
        "preflight_matches_live_candidate_readiness",
        probe.returncode == 0
        and preflight.get("status") == expected_status
        and preflight.get("expected_unit_n") == module.EXPECTED_N
        and preflight.get("ready_unit_n") == len(candidates)
        and set(preflight.get("missing_units", [])) == set(missing)
        and preflight.get("imaging_executed") is False,
        preflight,
    )
    check(
        "preflight_creates_no_attempt_or_image",
        attempts_after == attempts_before,
        {"new_attempts": sorted(attempts_after - attempts_before)},
    )

    source_text = SOURCE.read_text(encoding="utf-8")
    required = {
        "new_attempt_each_execution": (
            "attempt.mkdir(parents=True, exist_ok=False)"
        ),
        "candidate_source": "candidate_record(unit)",
        "locked_transform": "BASE.registration_route(unit)",
        "direct_atlas_qc": "BASE.BASE.atlas_qc",
        "no_matrix_authorization": '"matrix_generation_started": False',
        "no_source_modification": '"source_files_modified": False',
        "diagnosis_blind": '"diagnosis_labels_used": False',
    }
    missing_fragments = {
        name: text
        for name, text in required.items()
        if text not in source_text
    }
    check(
        "implementation_has_required_safety_fragments",
        not missing_fragments,
        {"missing": missing_fragments},
    )
    check(
        "implementation_cannot_generate_tractograms",
        "tckgen" not in source_text
        and "tcksift2" not in source_text
        and "tck2connectome" not in source_text,
        {
            "tckgen": "tckgen" in source_text,
            "tcksift2": "tcksift2" in source_text,
            "tck2connectome": "tck2connectome" in source_text,
        },
    )
    compile_probe = subprocess.run(
        [
            "/home/ec2-user/fsl/bin/python",
            "-m",
            "py_compile",
            str(SOURCE),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    check(
        "source_compiles",
        compile_probe.returncode == 0,
        {
            "returncode": compile_probe.returncode,
            "stderr": compile_probe.stderr,
        },
    )
    check(
        "base_recovery4_atlas_remains_separate",
        module.ATTEMPTS.resolve()
        != module.BASE.OUTPUT_ROOT.resolve()
        and not str(module.ATTEMPTS.resolve()).startswith(
            str(module.BASE.OUTPUT_ROOT.resolve()) + "/"
        ),
        {
            "new_attempt_root": str(module.ATTEMPTS),
            "original_atlas_root": str(module.BASE.OUTPUT_ROOT),
        },
    )

    executed_summaries = sorted(module.ATTEMPTS.glob("*/summary.json"))
    if executed_summaries:
        summary_path = executed_summaries[-1].resolve()
        summary = module.load_json(summary_path)
        rows = summary.get("units", [])
        result_failures: dict[str, str] = {}
        observed_units: set[str] = set()
        if isinstance(rows, list):
            for row in rows:
                unit = str(row.get("unit", ""))
                observed_units.add(unit)
                try:
                    result_record = row.get("result")
                    if not isinstance(result_record, dict):
                        raise TypeError("result record is absent")
                    result_path = Path(str(result_record.get("path", ""))).resolve()
                    if module.BASE.file_record(result_path) != result_record:
                        raise ValueError("result artifact binding differs")
                    result = module.load_json(result_path)
                    artifacts = result.get("artifacts", {})
                    if (
                        result.get("record_type")
                        != "diagnosis_blind_hcp379_hroi_canary_atlas_build"
                        or result.get("status") != "PASS_HROI_CANARY_ATLAS"
                        or result.get("unit") != unit
                        or result.get("diagnosis_labels_used") is not False
                        or result.get("source_files_modified") is not False
                        or result.get("non_overwriting") is not True
                        or result.get("tractography_started") is not False
                        or result.get("matrix_generation_started") is not False
                        or result.get("atlas_qc", {}).get("status") != "PASS"
                        or result.get("atlas_qc", {}).get("labels_found")
                        != module.BASE.BASE.EXPECTED_NODES
                        or not isinstance(artifacts, dict)
                    ):
                        raise ValueError("executed unit result differs")
                    for name in (
                        "hcp_source_parcellation_hroi",
                        "hcp_source_candidate_manifest",
                        "selected_t1_to_b0_transform",
                        "hcp379_nodes_b0_1mm",
                        "hcp379_node_volumes",
                        "hcp379_atlas_qc",
                        "hcp379_visual_overlay",
                    ):
                        record = artifacts.get(name)
                        if not isinstance(record, dict):
                            raise TypeError(f"{name} record is absent")
                        if module.BASE.file_record(
                            Path(str(record.get("path", ""))).resolve()
                        ) != record:
                            raise ValueError(f"{name} binding differs")
                except Exception as exc:
                    result_failures[unit or "<missing>"] = (
                        f"{type(exc).__name__}:{exc}"
                    )
        else:
            result_failures["<summary>"] = "units is not a list"
        check(
            "latest_executed_atlas_summary_is_exact_15_pass",
            (
                summary.get("record_type")
                == "hcp379_hroi_canary_atlas_summary"
                and summary.get("status") == "PASS_HROI_CANARY_ATLAS_15"
                and summary.get("diagnosis_labels_used") is False
                and summary.get("source_files_modified") is False
                and summary.get("non_overwriting") is True
                and summary.get("unit_n") == module.EXPECTED_N
                and summary.get("passed_unit_n") == module.EXPECTED_N
                and summary.get("failed_unit_n") == 0
                and summary.get("tractography_reused_not_regenerated") is True
                and summary.get("matrix_generation_started") is False
                and observed_units == set(units)
                and not result_failures
            ),
            {
                "summary": str(summary_path),
                "status": summary.get("status"),
                "passed_unit_n": summary.get("passed_unit_n"),
                "failed_unit_n": summary.get("failed_unit_n"),
                "observed_unit_n": len(observed_units),
                "result_failures": result_failures,
            },
        )

    passed = sum(
        value["status"] == "PASS" for value in checks.values()
    )
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_hroi_canary_atlas_v1_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "imaging_executed": False,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "source": module.BASE.file_record(SOURCE),
    }
    module.atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
