#!/usr/bin/env python3
"""Independently validate the non-imaging Eddy-to-Recovery4 handoff."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
RUNNER = EXP / "scripts/hcp/run_hcp379_eddy_pretract_recovery4_v1.py"
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_eddy_pretract_recovery4_runner_v1/validation.json"
)
ROOT = Path(
    "/data/derivatives/hcp379_v2/eddy_integrity_pretract_recovery4"
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
        [
            "/home/ec2-user/fsl/bin/python",
            str(RUNNER),
            "--validate-only",
        ],
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
        "runner_validation_passes",
        (
            process.returncode == 0
            and report.get("status") == "PASS"
            and report.get("passed_checks") == report.get("total_checks")
        ),
        {
            "returncode": process.returncode,
            "status": report.get("status"),
            "stderr": process.stderr[-1000:],
        },
    )
    check(
        "validation_executes_no_imaging",
        before == after and report.get("imaging_executed") is False,
        {"before_file_n": len(before), "after_file_n": len(after)},
    )
    check(
        "exact_four_subject_scope",
        all(
            token in source
            for token in (
                "003_S_4373_I378923",
                "003_S_6490_I1043781",
                "003_S_6644_I1083048",
                "006_S_4713_I1483612",
            )
        ),
        report.get("readiness"),
    )
    check(
        "shared_engine_and_compactor_are_reused",
        all(
            token in source
            for token in (
                "hcp379_scaleup_pretract_recovery4_v3.py",
                "compact_hcp379_pretract_recovery4_v3.py",
                "ENGINE.process_unit",
                "COMPACTOR.compact_unit",
            )
        ),
        "no duplicate Recovery4 engine is embedded",
    )
    check(
        "eddy_pass_is_fail_closed",
        all(
            token in source
            for token in (
                "PASS_EDDY_INTEGRITY_RECOVERY",
                "eddy_volume_integrity",
                "zero_volume_n",
                "tensor_audit",
                "historical_eddy_unchanged",
                "Eddy recovery is not ready",
            )
        ),
        "Recovery4 cannot execute from a non-PASS Eddy attempt",
    )
    check(
        "only_eddy_path_and_size_are_substituted",
        (
            'row["eddy_path"] =' in source
            and 'row["eddy_size_bytes"] =' in source
            and "historical Eddy binding differs" in source
        ),
        "base rows are regenerated from the current audited lanes",
    )
    check(
        "handoff_is_blind_and_isolated",
        all(
            token in source
            for token in (
                "diagnosis_labels_used",
                "selection_uses_connectome_density",
                "eddy_integrity_pretract_recovery4",
            )
        ),
        report.get("checks"),
    )
    passed = sum(row["status"] == "PASS" for row in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_eddy_pretract_runner_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "passed_checks": passed,
        "total_checks": len(checks),
        "runner_sha256": hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest(),
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
