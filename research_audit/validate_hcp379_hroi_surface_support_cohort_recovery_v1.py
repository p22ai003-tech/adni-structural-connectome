#!/usr/bin/env python3
"""Validate the exact 530-unit HROI cohort recovery assembly."""

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
    "assemble_hcp379_hroi_surface_support_cohort_recovery_v1.py"
)
ADOPTER = (
    EXP / "scripts/hcp/adopt_hcp379_hroi_surface_support_policy_v1.py"
)
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "hroi_surface_support_cohort_v1/attempts"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_hroi_surface_support_cohort_recovery_v1/validation.json"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")
UNIT = "114_S_6347_I1344943"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
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
        "dry_run_rehashes_exact_530_without_execution",
        (
            dry_probe.returncode == 0
            and dry.get("status") == "READY_TO_ASSEMBLE_EXACT_530"
            and dry.get("valid_unit_n") == 530
            and dry.get("diagnosis_labels_used") is False
            and dry.get("source_files_modified") is False
            and dry.get("cohort_recipe_selected") is False
            and dry.get("selected_count_tractography_authorized")
            is False
            and dry.get("final_release_authorized") is False
        ),
        dry or {"stderr": dry_probe.stderr[-2000:]},
    )
    text = SOURCE.read_text(encoding="utf-8")
    required = (
        "FAIL_INCOMPLETE_HROI_SUPPORT_COHORT",
        "PASS_114_HROI_RECOVERY_FOR_COHORT_MERGE",
        "PASS_COHORT_UNIFORM_HROI_SUPPORT_530",
        "REHASHED_VALID_FROM_FAILED_COHORT",
        "RECOVERED_VALID_ISOLATED_INPUT_ROUTE",
        "inventory_units",
        "validate_manifest",
        "selected_count_tractography_authorized",
        "final_release_authorized",
    )
    missing = [fragment for fragment in required if fragment not in text]
    check(
        "implementation_contains_exact_recovery_contract",
        not missing,
        {"missing": missing},
    )
    prohibited = (
        "rm -rf",
        "shutil.rmtree",
        ".unlink(",
        "diagnosis.csv",
        "DX_bl",
        "selected_count_tractography_authorized\": True",
        "final_release_authorized\": True",
    )
    found = [fragment for fragment in prohibited if fragment in text]
    check(
        "implementation_has_no_destructive_or_outcome_shortcut",
        not found,
        {"found": found},
    )

    summaries = sorted(
        path
        for path in ATTEMPTS.glob(
            "*-hroi-cohort-recovery/summary.json"
        )
    )
    if summaries:
        summary_path = summaries[-1]
        try:
            summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
            rows = summary.get("units", [])
            by_unit = {
                str(row.get("unit", "")): row for row in rows
            }
            contract = (
                summary.get("record_type")
                == (
                    "diagnosis_blind_hcp379_hroi_surface_support_"
                    "cohort_summary"
                )
                and summary.get("status")
                == "PASS_COHORT_UNIFORM_HROI_SUPPORT_530"
                and summary.get("diagnosis_labels_used") is False
                and summary.get("source_files_modified") is False
                and summary.get("release_adopted") is False
                and summary.get("expected_unit_n") == 530
                and summary.get("valid_unit_n") == 530
                and summary.get("reused_unit_n") == 529
                and summary.get("built_unit_n") == 1
                and summary.get("failure_n") == 0
                and summary.get("failures") == {}
                and len(rows) == 530
                and len(by_unit) == 530
                and "" not in by_unit
                and by_unit.get(UNIT, {}).get("status")
                == "RECOVERED_VALID_ISOLATED_INPUT_ROUTE"
                and sum(
                    row.get("status")
                    == "REHASHED_VALID_FROM_FAILED_COHORT"
                    for row in rows
                )
                == 529
            )
            check(
                "runtime_summary_exact_530_contract",
                contract,
                {
                    "summary": str(summary_path),
                    "unit_n": len(by_unit),
                    "recovered_114_status": by_unit.get(
                        UNIT,
                        {},
                    ).get("status"),
                },
            )
            adopter_probe = subprocess.run(
                [
                    str(PYTHON),
                    str(ADOPTER),
                    "--cohort-summary",
                    str(summary_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            try:
                adoption = json.loads(adopter_probe.stdout)
            except Exception:
                adoption = {}
            check(
                "independent_policy_adopter_rehashes_all_530",
                (
                    adopter_probe.returncode == 0
                    and adoption.get("status")
                    == "READY_TO_ADOPT_FOR_PRETRACT_ONLY"
                    and adoption.get("unit_n") == 530
                    and adoption.get("pretract_use_authorized") is False
                    and adoption.get("final_release_authorized") is False
                ),
                adoption
                or {"stderr": adopter_probe.stderr[-4000:]},
            )
        except Exception as exc:
            check(
                "runtime_execution_replay",
                False,
                {
                    "summary": str(summary_path),
                    "exception": repr(exc),
                },
            )

    passed = sum(row["status"] == "PASS" for row in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_hroi_surface_support_cohort_recovery_v1_validation"
        ),
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "runtime_summary_detected": bool(summaries),
        "imaging_executed_by_validation": False,
        "source": file_record(SOURCE),
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
