#!/home/ec2-user/fsl/bin/python
"""Validate the diagnosis-blind 216-unit legacy density input audit."""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "audit_hcp379_legacy_density_recovery_inputs_v3.py"
)
SUMMARY = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_legacy_density_recovery_input_audit_v3/"
    "legacy_density_recovery_input_audit_summary.json"
)
AUDIT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_legacy_density_recovery_input_audit_v3/"
    "legacy_density_recovery_input_audit.csv"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_legacy_density_recovery_input_audit_v3/validation.json"
)
EXPECTED_GROUPS = {
    "D0_THRESHOLD_QUALIFIED": 2,
    "D1_NEAR_THRESHOLD": 69,
    "D2_MODERATE_LOW": 59,
    "D3_SEVERE_LOW": 68,
    "D4_CRITICAL_LOW": 18,
}


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_legacy_density_input_audit_validation_target", SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    with AUDIT.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if condition else "FAIL",
            "evidence": evidence,
        }

    units = {row["unit"] for row in rows}
    groups = Counter(row["density_group"] for row in rows)
    check(
        "exact_unique_216",
        len(rows) == 216 and len(units) == 216,
        {"rows": len(rows), "unique": len(units)},
    )
    check(
        "exact_density_groups",
        dict(groups) == EXPECTED_GROUPS,
        dict(groups),
    )
    check(
        "all_216_ready_from_eddy",
        all(
            row["route"] == "READY_FOR_CORRECTED_PRETRACT_FROM_EDDY"
            and not row["failures"]
            for row in rows
        ),
        Counter(row["route"] for row in rows),
    )
    check(
        "exact_214_below_threshold_ready",
        sum(
            row["density_group"] != "D0_THRESHOLD_QUALIFIED"
            and row["route"] in module.READY_ROUTES
            for row in rows
        )
        == 214,
        {
            "reported": summary.get("below_threshold_ready_n"),
            "computed": sum(
                row["density_group"] != "D0_THRESHOLD_QUALIFIED"
                and row["route"] in module.READY_ROUTES
                for row in rows
            ),
        },
    )
    forbidden = {
        "diagnosis",
        "diagnosis_at_dti",
        "group",
        "research_group",
        "outcome",
    }
    check(
        "diagnosis_outcome_blind",
        not forbidden.intersection(
            str(field).strip().lower() for field in fields
        )
        and summary.get("diagnosis_or_outcomes_used") is False
        and all(
            row.get("selected_by_diagnosis_or_outcome") == "False"
            and row.get("diagnosis_labels_used") == "False"
            for row in rows
        ),
        {"fields": list(fields)},
    )
    check(
        "summary_and_hash_bindings_current",
        summary.get("status") == "PASS"
        and summary.get("target_n") == 216
        and summary.get("audited_n") == 216
        and summary.get("ready_n") == 216
        and summary["records"]["input_audit"] == module.file_record(AUDIT)
        and summary["records"]["builder"] == module.file_record(SOURCE),
        summary,
    )

    passed = sum(value["status"] == "PASS" for value in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_legacy_density_recovery_input_audit_v3_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "records": {
            "summary": module.file_record(SUMMARY),
            "audit": module.file_record(AUDIT),
            "source": module.file_record(SOURCE),
            "validator": module.file_record(Path(__file__)),
        },
    }
    module.atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
