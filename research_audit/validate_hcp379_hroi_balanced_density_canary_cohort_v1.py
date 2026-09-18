#!/usr/bin/env python3
"""Validate the 15-canary corrected-HROI density aggregator."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_hroi_balanced_density_canary_cohort_v1.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_hroi_balanced_density_canary_cohort_v1/validation.json"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


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
    dry_probe = subprocess.run(
        [str(PYTHON), str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        dry = json.loads(dry_probe.stdout)
    except Exception:
        dry = {}
    check(
        "dry_run_is_live_fail_closed_and_nonexecuting",
        (
            dry_probe.returncode == 0
            and dry.get("status")
            in {
                "READY_ALL_15_EXACT_PAIRS",
                "WAITING_FOR_EXACT_PHASE_B_PAIRS",
            }
            and dry.get("expected_unit_n") == 15
            and dry.get("tractography_generated") is False
            and dry.get("cohort_recipe_selected") is False
            and dry.get("scale_up_authorized") is False
            and dry.get("pair_ready_n", 0) + dry.get("waiting_n", 0) == 15
        ),
        dry or {"stderr": dry_probe.stderr[-2000:]},
    )
    text = SOURCE.read_text(encoding="utf-8")
    required = (
        "PASS_COMPLETE_COUNT_ONLY_DENSITY_EVIDENCE",
        "PARTIAL_COUNT_ONLY_DENSITY_EVIDENCE",
        "density_qualified_balanced_total_counts",
        "smallest_density_qualified_balanced_total_count",
        "requires_joint_all_nine_matrix_stability",
        "requires_uniform_530_subject_construction",
        "tractography_reused_not_regenerated",
        "cohort_uniform_recipe_selected",
        "scale_up_authorized",
        "6_000_000",
        "10_000_000",
        "15_000_000",
        "20_000_000",
    )
    missing = [fragment for fragment in required if fragment not in text]
    check(
        "implementation_contains_complete_fail_closed_contract",
        not missing,
        {"missing": missing},
    )
    prohibited = (
        "rm -rf",
        "shutil.rmtree",
        ".unlink(",
        "diagnosis.csv",
        "DX_bl",
        "cohort_uniform_recipe_selected\": True",
        "scale_up_authorized\": True",
    )
    found = [fragment for fragment in prohibited if fragment in text]
    check(
        "implementation_has_no_destructive_or_outcome_aware_shortcut",
        not found,
        {"found": found},
    )

    passed = sum(row["status"] == "PASS" for row in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_hroi_balanced_density_canary_cohort_v1_validation"
        ),
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "imaging_executed_by_validation": False,
        "source": record(SOURCE),
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
