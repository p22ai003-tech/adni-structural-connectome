#!/usr/bin/env python3
"""Validate the bounded Phase-B same-unit pair-priority recovery."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "recover_hcp379_phase_b_pair_priority_v1.py"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_phase_b_pair_priority_v1/validation.json"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_phase_b_pair_priority_validation_target", SOURCE
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

    check(
        "target_is_exact_same_unit_independent_companion",
        module.UNIT == "032_S_6804_I1230908"
        and "/032_S_6804_I1230908/" in str(module.PRIMARY)
        and "/primary_10m/tracks.tck" in str(module.PRIMARY)
        and "/032_S_6804_I1230908/" in str(module.INDEPENDENT)
        and "/independent_10m/tracks.tck" in str(module.INDEPENDENT)
        and module.EXPECTED == 10_000_000,
        {
            "unit": module.UNIT,
            "primary": str(module.PRIMARY),
            "independent": str(module.INDEPENDENT),
        },
    )
    check(
        "primary_is_exact_and_hashable",
        module.track_count(module.PRIMARY) == module.EXPECTED
        and module.file_record(module.PRIMARY)["size_bytes"] > 0,
        {
            "primary_count": module.track_count(module.PRIMARY),
            "primary": module.file_record(module.PRIMARY),
        },
    )
    check(
        "same_official_snakefile_config_and_launcher_are_bound",
        module.SNAKEFILE.is_file()
        and module.CONFIG.is_file()
        and module.OFFICIAL_LAUNCHER.is_file()
        and len(module.file_record(module.SNAKEFILE)["sha256"]) == 64
        and len(module.file_record(module.CONFIG)["sha256"]) == 64,
        {
            "snakefile": str(module.SNAKEFILE),
            "config": str(module.CONFIG),
            "launcher": str(module.OFFICIAL_LAUNCHER),
        },
    )
    source_text = SOURCE.read_text(encoding="utf-8")
    required = {
        "requires_stopped_or_closed_old_dag": "closed_dag_ready",
        "checks_no_tckgen_before_takeover": "and not active_tckgen()",
        "exact_target_only": "str(INDEPENDENT)",
        "exact_header_verification": "track_count(INDEPENDENT)",
        "resumes_official_launcher": "PAIR_10M_PASS_OFFICIAL_DAG_RESUMED",
        "new_attempt": "attempt.mkdir(parents=True, exist_ok=False)",
        "diagnosis_blind": '"diagnosis_labels_used": False',
    }
    missing = {
        name: fragment
        for name, fragment in required.items()
        if fragment not in source_text
    }
    check(
        "implementation_has_bounded_recovery_fragments",
        not missing,
        {"missing": missing},
    )
    check(
        "implementation_has_no_source_or_output_deletion",
        "rm -rf" not in source_text
        and "unlink(" not in source_text
        and "rmtree(" not in source_text,
        {"destructive_fragment_found": False},
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
    attempts = sorted(module.ATTEMPTS.glob("*/state.json"))
    current = json.loads(attempts[-1].read_text()) if attempts else {}
    check(
        "live_attempt_is_auditable",
        bool(attempts)
        and current.get("record_type")
        == "hcp379_phase_b_pair_priority_recovery_preflight"
        and current.get("diagnosis_labels_used") is False
        and current.get("unit") == module.UNIT
        and current.get("status")
        in {
            "WAITING_FOR_PRIMARY_10M",
            "PRIMARY_10M_PROMOTED_STOPPING_OLD_DAG",
            "PRIMARY_10M_PROMOTED_OLD_DAG_CLOSED",
            "RUNNING_TARGETED_INDEPENDENT_10M",
            "PAIR_10M_PASS_RESUMING_OFFICIAL_DAG",
            "PAIR_10M_PASS_OFFICIAL_DAG_RESUMED",
            "FAIL_TARGETED_INDEPENDENT_10M",
        },
        {
            "attempt_n": len(attempts),
            "status": current.get("status"),
            "unit": current.get("unit"),
        },
    )

    passed = sum(
        value["status"] == "PASS" for value in checks.values()
    )
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_phase_b_pair_priority_v1_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "imaging_executed_by_validation": False,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "source": module.file_record(SOURCE),
    }
    module.atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
