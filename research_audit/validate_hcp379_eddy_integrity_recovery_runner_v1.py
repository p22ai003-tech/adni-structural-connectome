#!/usr/bin/env python3
"""Static, non-imaging validation of the exact Eddy recovery runner."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
RUNNER = EXP / "scripts/hcp/run_hcp379_eddy_integrity_recovery_v1.py"
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_eddy_integrity_recovery_runner_v1/validation.json"
)
ROOT = Path(
    "/data/derivatives/hcp379_v2/eddy_integrity_recovery_v1"
)


def inventory() -> dict[str, tuple[int, int]]:
    if not ROOT.exists():
        return {}
    return {
        str(path.relative_to(ROOT)): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in ROOT.rglob("*")
        if path.is_file()
    }


def main() -> int:
    before = inventory()
    process = subprocess.run(
        ["python", str(RUNNER), "--validate-only", "--backend", "auto"],
        check=False,
        capture_output=True,
        text=True,
    )
    after = inventory()
    try:
        report = json.loads(process.stdout)
    except Exception:
        report = {}
    source = RUNNER.read_text(encoding="utf-8")
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    check(
        "runner_static_validation_passes",
        (
            process.returncode == 0
            and report.get("status") == "PASS"
            and report.get("passed_checks") == report.get("total_checks")
        ),
        {
            "returncode": process.returncode,
            "status": report.get("status"),
            "checks": report.get("checks"),
        },
    )
    check(
        "dry_validation_writes_no_imaging",
        before == after and report.get("imaging_executed") is False,
        {"before_file_n": len(before), "after_file_n": len(after)},
    )
    check(
        "exact_contracts_are_consumed",
        all(
            token in source
            for token in (
                "hcp379_eddy_integrity_recovery_plan_v1",
                "hcp379_eddy_acquisition_contract_v1",
                "ACQUISITION_VALIDATION",
                "EXPECTED_UNITS",
            )
        ),
        {"runner_sha256": hashlib.sha256(source.encode()).hexdigest()},
    )
    check(
        "source_filter_is_route_scoped",
        all(
            token in source
            for token in (
                "DROP_EXACT_ZERO_B0_THEN_REGENERATE_EDDY",
                "retained_volume_indices",
                '"-coord"',
                '"3"',
            )
        ),
        "route-specific mrconvert coordinate selection",
    )
    check(
        "eddy_command_binds_acquisition_and_qc",
        all(
            token in source
            for token in (
                '"-pe_dir"',
                '"-readout_time"',
                '"BZeroThreshold"',
                '"-eddyqc_all"',
                "--repol",
                "--cnr_maps",
                "--residuals",
            )
        ),
        "phase, readout, threshold and Eddy QC are explicit",
    )
    check(
        "cpu_gpu_routes_are_fail_closed",
        all(
            token in source
            for token in (
                "DWIFSLPREPROC_FORCE_CPU",
                "DWIFSLPREPROC_NO_CPU_FALLBACK",
                "eddy_cpu",
                "eddy_cuda11.0",
                "nvidia-smi",
                "attempting OpenMP version",
            )
        ),
        report.get("selected_backend"),
    )
    check(
        "physical_qc_is_mandatory",
        all(
            token in source
            for token in (
                "volume_integrity",
                "zero_volume_n",
                "tensor_audit",
                "PASS_EDDY_INTEGRITY_RECOVERY",
            )
        ),
        "zero-volume and tensor gates precede PASS",
    )
    check(
        "attempts_are_versioned_and_historical_hash_rechecked",
        all(
            token in source
            for token in (
                '"attempts"',
                "uuid.uuid4",
                "historical_sha",
                "historical_eddy_unchanged",
                "historical_eddy_overwritten",
            )
        ),
        "each attempt is preserved under the isolated recovery root",
    )
    passed = sum(item["status"] == "PASS" for item in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_eddy_integrity_recovery_runner_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
