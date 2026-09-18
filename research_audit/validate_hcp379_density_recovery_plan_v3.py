#!/home/ec2-user/fsl/bin/python
"""Independently validate the exact-530 HCP379 density recovery plan."""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE = Path(
    "/home/ec2-user/exp/research_audit/"
    "build_hcp379_density_recovery_plan_v3.py"
)
PLAN = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_density_recovery_plan_v3/hcp379_density_recovery_plan.json"
)
SUBJECTS = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_density_recovery_plan_v3/hcp379_density_recovery_subjects.csv"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_density_recovery_plan_v3/validation.json"
)
EXPECTED = {
    "D0_THRESHOLD_QUALIFIED": 58,
    "D1_NEAR_THRESHOLD": 97,
    "D2_MODERATE_LOW": 61,
    "D3_SEVERE_LOW": 68,
    "D4_CRITICAL_LOW": 18,
    "U0_UNGENERATED": 228,
}


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_density_recovery_plan_validation_target", SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    with SUBJECTS.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if condition else "FAIL",
            "evidence": evidence,
        }

    units = [row["unit"] for row in rows]
    counts = Counter(row["density_group"] for row in rows)
    check(
        "exact_unique_530",
        len(rows) == 530 and len(set(units)) == 530,
        {"rows": len(rows), "unique_units": len(set(units))},
    )
    check(
        "exact_density_groups",
        dict(counts) == EXPECTED,
        dict(counts),
    )
    forbidden = {
        "diagnosis",
        "diagnosis_at_dti",
        "group",
        "research_group",
        "outcome",
        "label",
    }
    check(
        "diagnosis_outcome_fields_absent",
        not forbidden.intersection(
            str(field).strip().lower() for field in fields
        )
        and all(
            row.get("diagnosis_or_outcomes_used") == "False" for row in rows
        ),
        {"fields": list(fields), "forbidden": sorted(forbidden)},
    )
    numeric_ok = True
    action_ok = True
    edge_count_ok = True
    for row in rows:
        group = row["density_group"]
        text = row["current_density"].strip()
        value = float(text) if text else None
        numeric_ok &= module.density_group(value) == group
        if value is None:
            edge_count_ok &= not row[
                "current_nonzero_undirected_edges"
            ].strip()
        else:
            edges = int(row["current_nonzero_undirected_edges"])
            edge_count_ok &= abs(
                value - edges / module.POSSIBLE_EDGES
            ) <= 1.0 / module.POSSIBLE_EDGES
        if group == "D0_THRESHOLD_QUALIFIED":
            action_ok &= (
                row["action_class"] == "DENSITY_PASS_ROUTE_PENDING"
                and "candidate only" in row["immediate_action"]
            )
        elif group == "U0_UNGENERATED":
            action_ok &= "GENERATION" in row["action_class"]
        else:
            action_ok &= (
                row["action_class"] == "CORRECTED_DENSITY_RECOVERY"
                and "Do not promote" in row["immediate_action"]
            )
    check(
        "density_recomputed_into_correct_bins",
        numeric_ok,
        {"threshold": module.THRESHOLD},
    )
    check(
        "edge_counts_consistent",
        edge_count_ok,
        {"possible_undirected_edges": module.POSSIBLE_EDGES},
    )
    check(
        "actions_fail_closed",
        action_ok,
        {
            "D0": "conditional candidate",
            "D1_D4": "corrected recovery",
            "U0": "generation",
        },
    )
    contract = plan["goal_acceptance_contract"]
    check(
        "hard_530_of_530_density_contract",
        contract["target_subjects"] == 530
        and contract["required_final_subjects_at_or_above_threshold"] == 530
        and contract["minimum_density_inclusive"] == 0.60
        and contract["density_is_whole_cohort_technical_release_gate"] is True,
        contract,
    )
    check(
        "anti_gaming_contract",
        contract["subject_exclusion_to_meet_density_allowed"] is False
        and contract["edge_imputation_or_artificial_filling_allowed"] is False
        and contract["diagnosis_or_outcome_aware_tuning_allowed"] is False
        and contract["per_subject_streamline_count_tuning_allowed"] is False
        and contract["historical_output_overwrite_allowed"] is False,
        contract,
    )
    policy = plan["execution_policy"]
    check(
        "uniform_recipe_and_fail_closed_policy",
        "smallest prespecified uniform streamline count" in policy["phase_b"]
        and policy["no_qualified_phase_b_recipe"]
        == "Return NO_DENSITY_QUALIFIED_RECIPE and do not scale."
        and "fail closed unless all 530 values are >=0.60"
        in policy["final_gate"],
        policy,
    )
    check(
        "source_hashes_current",
        plan["records"]["builder"] == module.file_record(SOURCE)
        and plan["records"]["subject_plan"] == module.file_record(SUBJECTS),
        {
            "builder": plan["records"]["builder"],
            "subjects": plan["records"]["subject_plan"],
        },
    )

    passed = sum(item["status"] == "PASS" for item in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_density_recovery_plan_v3_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "records": {
            "plan": module.file_record(PLAN),
            "subject_plan": module.file_record(SUBJECTS),
            "builder": module.file_record(SOURCE),
            "validator": module.file_record(Path(__file__)),
        },
    }
    module.atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
