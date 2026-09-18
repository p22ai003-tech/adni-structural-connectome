#!/home/ec2-user/fsl/bin/python
"""Independently validate the live exact-530 status/density crosswalk."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
ROOT = EXP / "research_audit/outputs/hcp379_live_530_status_density_v1"
SUMMARY = ROOT / "summary.json"
GROUPS = ROOT / "group_status_density.csv"
SUBJECTS = ROOT / "subject_status_density.csv"
TOPOLOGY = ROOT / "topology_snapshot.json"
LEDGER = ROOT / "ledger_snapshot.csv"
OUTPUT = ROOT / "validation.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def main() -> int:
    summary = load_json(SUMMARY)
    topology = load_json(TOPOLOGY)
    groups = read_csv(GROUPS)
    subjects = read_csv(SUBJECTS)
    ledger = read_csv(LEDGER)
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    expected = {
        str(row["unit"]) for row in topology["canaries"]
    } | {str(row["unit"]) for row in topology["production"]}
    observed = {row["unit"] for row in subjects}
    check(
        "exact_disjoint_15_plus_515_subjects",
        (
            len(topology["canaries"]) == 15
            and len(topology["production"]) == 515
            and len(expected) == 530
            and len(subjects) == 530
            and observed == expected
        ),
        {
            "canary_n": len(topology["canaries"]),
            "production_n": len(topology["production"]),
            "subject_row_n": len(subjects),
            "unique_n": len(observed),
        },
    )

    status = Counter(row["status"] for row in ledger)
    total = groups[-1] if groups else {}
    check(
        "live_status_counts_replay",
        (
            total.get("processing_group") == "Full release total"
            and int(total["target_n"]) == 530
            and int(total["pretract_completed_n"])
            == 15 + status.get("PASS_COMPACTED", 0)
            and int(total["running_n"]) == status.get("RUNNING", 0)
            and int(total["corrective_hold_n"])
            == status.get("FAILED", 0)
            and int(total["waiting_n"]) == status.get("WAITING", 0)
        ),
        {"ledger_status_counts": dict(status), "total_row": total},
    )

    aal_values = [
        float(row["existing_aal3_166_density"]) for row in subjects
    ]
    hcp_values = [
        float(row["existing_hcp379_density"])
        for row in subjects
        if row["existing_hcp379_density"]
    ]
    check(
        "existing_density_coverage_is_exact",
        (
            len(aal_values) == 530
            and len(hcp_values) == 302
            and int(total["existing_aal3_166_available_n"]) == 530
            and int(total["existing_hcp379_available_n"]) == 302
            and int(total["existing_aal3_166_pass_0_60_n"])
            == sum(value >= 0.60 for value in aal_values)
            and int(total["existing_hcp379_pass_0_60_n"])
            == sum(value >= 0.60 for value in hcp_values)
        ),
        {
            "aal_available_n": len(aal_values),
            "aal_pass_0_60_n": sum(v >= 0.60 for v in aal_values),
            "hcp_available_n": len(hcp_values),
            "hcp_pass_0_60_n": sum(v >= 0.60 for v in hcp_values),
        },
    )

    check(
        "corrected_hcp379_is_fail_closed",
        (
            topology.get("final_release_authorized") is False
            and all(
                row["corrected_hcp379_density"] == ""
                and row["corrected_hcp379_status"]
                == "PENDING_RECIPE_AND_PRODUCTION"
                for row in subjects
            )
            and int(total["corrected_hcp379_available_n"]) == 0
        ),
        {
            "final_release_authorized": topology.get(
                "final_release_authorized"
            ),
            "corrected_available_n": total.get(
                "corrected_hcp379_available_n"
            ),
        },
    )

    records = summary.get("records", {})
    check(
        "source_and_output_hashes_bind",
        (
            records.get("topology_snapshot", {}).get("sha256")
            == sha256_file(TOPOLOGY)
            and records.get("ledger_snapshot", {}).get("sha256")
            == sha256_file(LEDGER)
            and records.get("group_csv", {}).get("sha256")
            == sha256_file(GROUPS)
            and records.get("subject_csv", {}).get("sha256")
            == sha256_file(SUBJECTS)
        ),
        {
            "topology_sha256": sha256_file(TOPOLOGY),
            "ledger_sha256": sha256_file(LEDGER),
            "group_csv_sha256": sha256_file(GROUPS),
            "subject_csv_sha256": sha256_file(SUBJECTS),
        },
    )

    check(
        "audit_is_nonexecuting_and_blind",
        (
            summary.get("diagnosis_labels_used") is False
            and summary.get("outcomes_used") is False
            and summary.get("imaging_executed") is False
            and summary.get("matrices_modified") is False
        ),
        {
            key: summary.get(key)
            for key in (
                "diagnosis_labels_used",
                "outcomes_used",
                "imaging_executed",
                "matrices_modified",
            )
        },
    )

    passed = sum(item["status"] == "PASS" for item in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_live_530_status_density_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
