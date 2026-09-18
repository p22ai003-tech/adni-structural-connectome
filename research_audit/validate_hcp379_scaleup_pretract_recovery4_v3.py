#!/home/ec2-user/fsl/bin/python
"""Static and hostile-gate validation for the Recovery4 scale-up runner."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "hcp379_scaleup_pretract_recovery4_v3.py"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_scaleup_pretract_recovery4_v3/validation.json"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_scaleup_pretract_recovery4_v3_validation_target",
        SOURCE,
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

    gate = module.validate_recovery4_gate(
        human_qc=module.DEFAULT_HUMAN_QC,
        execute=False,
    )
    hroi_policy_ready = (
        gate["hroi_source_policy_status"]
        == "PASS_OPERATIONAL_PRETRACT_ADOPTION"
    )
    check(
        "recovery4_gate_without_execution",
        len(gate["canary_units"]) == module.EXPECTED_CANARY_N
        and gate["human_visual_qc"] is None
        and gate["active_recovery4_source_mode"]
        == "V6_HUMAN_REVIEWED_PROMOTION"
        and gate["corrected_overlay_raw_hard_hold_units"]
        == ["009_S_4324_I1186579"]
        and gate["corrected_overlay_hard_hold_units"] == []
        and gate["promotion_receipt"] is not None,
        {
            "canary_n": len(gate["canary_units"]),
            "human_visual_qc": gate["human_visual_qc"],
            "active_recovery4_source_mode": gate[
                "active_recovery4_source_mode"
            ],
            "raw_hard_holds": gate[
                "corrected_overlay_raw_hard_hold_units"
            ],
            "effective_hard_holds": gate[
                "corrected_overlay_hard_hold_units"
            ],
            "promotion_receipt": gate["promotion_receipt"],
        },
    )
    check(
        "hroi_source_policy_is_ready_or_fails_closed",
        (
            hroi_policy_ready
            and gate["hroi_source_policy_receipt"] is not None
            and len(gate["hroi_source_policy_units"])
            == module.EXPECTED_HROI_POLICY_N
        )
        or (
            not hroi_policy_ready
            and gate["hroi_source_policy_status"]
            == "WAITING_FOR_530_HROI_POLICY_ADOPTION"
            and gate["hroi_source_policy_receipt"] is None
            and gate["hroi_source_policy_units"] == {}
        ),
        {
            "status": gate["hroi_source_policy_status"],
            "receipt": gate["hroi_source_policy_receipt"],
            "unit_n": len(gate["hroi_source_policy_units"]),
        },
    )

    rows, overlap = module.validate_audit_rows(gate)
    check(
        "exact_noncanary_set",
        len(rows) == module.EXPECTED_SCALEUP_N
        and len({row["unit"] for row in rows})
        == module.EXPECTED_SCALEUP_N
        and len(overlap) == module.EXPECTED_CANARY_OVERLAP_N,
        {
            "scaleup_n": len(rows),
            "unique_n": len({row["unit"] for row in rows}),
            "overlap_n": len(overlap),
        },
    )
    check(
        "audit_rows_use_only_adopted_hroi_candidates",
        (
            hroi_policy_ready
            and all(
                row.get("hcp_source_parcellation_policy")
                == "HROI_SURFACE_SUPPORT_V1_COHORT_UNIFORM"
                and row.get("hcp_source_parcellation_path")
                == gate["hroi_source_policy_units"][row["unit"]][
                    "candidate"
                ]["path"]
                and row.get("hcp_source_parcellation_original_path")
                == gate["hroi_source_policy_units"][row["unit"]][
                    "original_source"
                ]["path"]
                for row in rows
            )
        )
        or (
            not hroi_policy_ready
            and all(
                "hcp_source_parcellation_policy" not in row
                for row in rows
            )
        ),
        {
            "policy_ready": hroi_policy_ready,
            "bound_row_n": sum(
                row.get("hcp_source_parcellation_policy")
                == "HROI_SURFACE_SUPPORT_V1_COHORT_UNIFORM"
                for row in rows
            ),
        },
    )
    check(
        "canary_exclusion_is_disjoint",
        not set(gate["canary_units"]).intersection(
            row["unit"] for row in rows
        ),
        {"overlap_after_exclusion": 0},
    )

    route_counts: dict[str, int] = {}
    for row in rows:
        route_counts[row["route"]] = route_counts.get(row["route"], 0) + 1
    check(
        "audited_input_routes",
        route_counts
        == {
            "READY_AFTER_GRADIENT_HEADER_REPAIR": 27,
            "READY_FOR_CORRECTED_PRETRACT_FROM_EDDY": 189,
        },
        route_counts,
    )

    try:
        module.validate_recovery4_gate(
            human_qc=module.DEFAULT_HUMAN_QC.with_name(
                "intentionally_missing_validator_control.csv"
            ),
            execute=True,
        )
    except FileNotFoundError as exc:
        missing_human_rejected = "genuine human-QC CSV" in str(exc)
        missing_human_evidence = str(exc)
    else:
        missing_human_rejected = False
        missing_human_evidence = "unexpectedly accepted"
    check(
        "execute_rejects_missing_human_qc",
        missing_human_rejected,
        missing_human_evidence,
    )

    original_hroi_adoption = module.HROI_POLICY_ADOPTION
    try:
        module.HROI_POLICY_ADOPTION = original_hroi_adoption.with_name(
            "intentionally_missing_hroi_policy_validator_control.json"
        )
        try:
            module.validate_recovery4_gate(
                human_qc=module.DEFAULT_HUMAN_QC,
                execute=True,
            )
        except FileNotFoundError as exc:
            missing_hroi_rejected = (
                "HROI pretract-adoption receipt is required" in str(exc)
            )
            missing_hroi_evidence = str(exc)
        else:
            missing_hroi_rejected = False
            missing_hroi_evidence = "unexpectedly accepted"
    finally:
        module.HROI_POLICY_ADOPTION = original_hroi_adoption
    check(
        "execute_rejects_missing_hroi_policy_adoption",
        missing_hroi_rejected,
        missing_hroi_evidence,
    )

    units = set(gate["canary_units"])
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with tempfile.TemporaryDirectory(prefix="hcp379-v3-human-validator-") as tmp:
        valid_path = Path(tmp) / "synthetic_validator_control.csv"
        with valid_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "unit",
                    "status",
                    "reviewer",
                    "reviewed_utc",
                    "notes",
                ),
            )
            writer.writeheader()
            for unit in sorted(units):
                writer.writerow(
                    {
                        "unit": unit,
                        "status": "PASS",
                        "reviewer": "synthetic-validator-control",
                        "reviewed_utc": now,
                        "notes": "not a real review; parser control only",
                    }
                )
        valid_record = module.validate_human_qc(valid_path, units)
        check(
            "synthetic_human_parser_positive_control",
            valid_record["size_bytes"] > 0
            and len(valid_record["sha256"]) == 64,
            {
                "purpose": "parser control only; not execution evidence",
                "recorded": True,
            },
        )

        forbidden_path = Path(tmp) / "forbidden.csv"
        with forbidden_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "unit",
                    "status",
                    "reviewer",
                    "reviewed_utc",
                    "diagnosis",
                ),
            )
            writer.writeheader()
            for unit in sorted(units):
                writer.writerow(
                    {
                        "unit": unit,
                        "status": "PASS",
                        "reviewer": "synthetic-validator-control",
                        "reviewed_utc": now,
                        "diagnosis": "FORBIDDEN",
                    }
                )
        try:
            module.validate_human_qc(forbidden_path, units)
        except ValueError as exc:
            forbidden_rejected = "not diagnosis blind" in str(exc)
            forbidden_evidence = str(exc)
        else:
            forbidden_rejected = False
            forbidden_evidence = "unexpectedly accepted"
        check(
            "human_qc_rejects_diagnosis_field",
            forbidden_rejected,
            forbidden_evidence,
        )

    plan = module.load_json(
        module.OUTPUT_ROOT / "manifests/pretract_recovery4_plan.json"
    )
    dryrun_mode = (
        plan.get("status")
        == (
            "PRE_HUMAN_DRY_RUN_PASS"
            if hroi_policy_ready
            else "WAITING_FOR_530_HROI_POLICY_ADOPTION"
        )
        and plan.get("execution_requested") is False
        and plan.get("pretract_execution_authorized") is False
    )
    execution_bound_mode = (
        hroi_policy_ready
        and plan.get("status") == "EXECUTION_GATE_PASS"
        and plan.get("execution_requested") is True
        and plan.get("pretract_execution_authorized") is True
        and isinstance(
            plan.get("gate_records", {}).get("human_visual_qc"),
            dict,
        )
        and len(
            plan["gate_records"]["human_visual_qc"].get("sha256", "")
        )
        == 64
    )
    check(
        "current_plan_exact_scope",
        (dryrun_mode or execution_bound_mode)
        and plan.get("imaging_executed_by_plan") is False
        and plan.get("hroi_source_policy_ready") is hroi_policy_ready
        and plan.get("target_n") == module.EXPECTED_SCALEUP_N
        and len(plan.get("units", [])) == module.EXPECTED_SCALEUP_N,
        {
            "status": plan.get("status"),
            "execution_requested": plan.get("execution_requested"),
            "imaging_executed_by_plan": plan.get(
                "imaging_executed_by_plan"
            ),
            "hroi_source_policy_ready": plan.get(
                "hroi_source_policy_ready"
            ),
            "pretract_execution_authorized": plan.get(
                "pretract_execution_authorized"
            ),
            "accepted_mode": (
                "DRY_RUN"
                if dryrun_mode
                else (
                    "EXECUTION_BOUND"
                    if execution_bound_mode
                    else "INVALID"
                )
            ),
            "target_n": plan.get("target_n"),
            "unit_n": len(plan.get("units", [])),
        },
    )
    check(
        "dryrun_plan_has_no_diagnosis_or_outcome",
        all(
            not {
                "diagnosis",
                "diagnosis_at_dti",
                "group",
                "research_group",
                "outcome",
            }.intersection(row)
            for row in plan.get("units", [])
        ),
        {"unit_fields": sorted(plan["units"][0])},
    )
    check(
        "route_selection_is_not_density_driven",
        plan["registration_policy"]["select_route_by_connectome_density"]
        is False
        and plan["registration_policy"]["silent_fallback"] is False,
        plan["registration_policy"],
    )
    check(
        "raw_and_bounded_tensor_contract",
        plan["tensor_policy"]["raw_maps_preserved"] is True
        and plan["tensor_policy"]["bounded_maps_separate"] is True
        and plan["tensor_policy"][
            "maximum_fraction_per_raw_anomaly_condition"
        ]
        == 0.01,
        plan["tensor_policy"],
    )
    check(
        "tool_contract_is_bound",
        len(plan.get("tool_contract", {})) >= 25
        and all(
            len(record.get("sha256", "")) == 64
            for record in plan.get("tool_contract", {}).values()
        ),
        {"tool_count": len(plan.get("tool_contract", {}))},
    )
    check(
        "source_contracts_are_bound",
        all(
            len(plan[name]["sha256"]) == 64
            for name in (
                "builder",
                "provisional_engine",
                "scalar_qc_implementation",
                "hroi_policy_adoption_implementation",
            )
        ),
        {
            name: plan[name]["sha256"]
            for name in (
                "builder",
                "provisional_engine",
                "scalar_qc_implementation",
                "hroi_policy_adoption_implementation",
            )
        },
    )

    source_text = SOURCE.read_text(encoding="utf-8")
    required_source_fragments = {
        "versioned_root": "corrected_scaleup_recovery4",
        "direct_hcp_qc": "ORIGINAL_ATLAS_QC",
        "bbr_primary": "fsl_bbr_primary",
        "ants_failover": "seeded_ants_rigid_failover",
        "ants_seed": "ANTS_RANDOM_SEED = 1234",
        "five_tt_gate": "MINIMUM_5TT_DWI_DICE = 0.70",
        "bounded_tensor_gate": "SCALAR.process_unit",
        "multivolume_fod_range": "V2.mrstats_range = provisional_mrstats_range",
        "non_overwriting_retry": "archive_prior_subject_state",
        "hash_promotion": "PASS_PRETRACT_RECOVERY4",
        "hroi_policy_gate": "validate_hroi_policy_adoption",
        "hroi_candidate_binding": (
            "HROI_SURFACE_SUPPORT_V1_COHORT_UNIFORM"
        ),
        "cached_source_invalidation": (
            "hcp_source_policy_adoption_receipt"
        ),
    }
    missing_fragments = {
        name: fragment
        for name, fragment in required_source_fragments.items()
        if fragment not in source_text
    }
    check(
        "recovery4_implementation_fragments",
        not missing_fragments,
        {"missing": missing_fragments},
    )

    original_run = module.subprocess.run

    class SyntheticCompleted:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def synthetic_mrstats(command: list[str], **_: Any) -> Any:
        name = Path(command[1]).name
        if name == "synthetic_multivolume.mif":
            return SyntheticCompleted(
                "0.0 1.0\n-0.5 0.5\n-0.25 0.25\n"
            )
        if name == "synthetic_scalar.mif":
            return SyntheticCompleted("0.1 0.9\n")
        raise AssertionError(f"unexpected synthetic path: {name}")

    try:
        module.subprocess.run = synthetic_mrstats
        multivolume = module.all_volume_range(
            Path("/synthetic/synthetic_multivolume.mif")
        )
        scalar = module.all_volume_range(
            Path("/synthetic/synthetic_scalar.mif")
        )
        module.install_provisional_engine_guards()
        guarded = module.V2.mrstats_range(
            Path("/synthetic/synthetic_multivolume.mif")
        )
    finally:
        module.subprocess.run = original_run
    check(
        "scalar_and_multivolume_mrstats_range_guard",
        multivolume
        == {
            "volume_count": 3,
            "min": -0.5,
            "max": 1.0,
            "all_finite": True,
        }
        and scalar
        == {
            "volume_count": 1,
            "min": 0.1,
            "max": 0.9,
            "all_finite": True,
        }
        and guarded == {"min": -0.5, "max": 1.0},
        {
            "multivolume": multivolume,
            "scalar": scalar,
            "guarded_multivolume": guarded,
        },
    )
    scalar_source_text = module.SCALAR_SOURCE.read_text(encoding="utf-8")
    observed_fraction = 3 / 101_113
    widespread_fraction = 101 / 100_000
    numerical_guard_fragments = {
        "hard_magnitude_constant": (
            "FOD_TISSUE_HARD_NEGATIVE_TOLERANCE = 1.0e-4"
        ),
        "fraction_constant": (
            "FOD_TISSUE_BELOW_TOLERANCE_FRACTION_MAXIMUM = 1.0e-3"
        ),
        "hard_count_failure": "below_hard_tolerance",
        "fraction_failure": "below_tolerance_fractions",
        "diagnosis_blind": "No diagnosis or outcome field is read",
    }
    missing_numerical_guard_fragments = {
        name: fragment
        for name, fragment in numerical_guard_fragments.items()
        if fragment not in scalar_source_text
    }
    check(
        "fod_numerical_guard_uses_magnitude_and_fraction",
        (
            module.SCALAR.FOD_TISSUE_NUMERICAL_TOLERANCE == 1.0e-5
            and module.SCALAR.FOD_TISSUE_HARD_NEGATIVE_TOLERANCE
            == 1.0e-4
            and module.SCALAR.FOD_TISSUE_BELOW_TOLERANCE_FRACTION_MAXIMUM
            == 1.0e-3
            and observed_fraction
            <= module.SCALAR.FOD_TISSUE_BELOW_TOLERANCE_FRACTION_MAXIMUM
            and widespread_fraction
            > module.SCALAR.FOD_TISSUE_BELOW_TOLERANCE_FRACTION_MAXIMUM
            and not missing_numerical_guard_fragments
        ),
        {
            "soft_tolerance": (
                module.SCALAR.FOD_TISSUE_NUMERICAL_TOLERANCE
            ),
            "hard_negative_tolerance": (
                module.SCALAR.FOD_TISSUE_HARD_NEGATIVE_TOLERANCE
            ),
            "maximum_below_tolerance_fraction": (
                module.SCALAR
                .FOD_TISSUE_BELOW_TOLERANCE_FRACTION_MAXIMUM
            ),
            "observed_three_of_101113_fraction": observed_fraction,
            "synthetic_widespread_fraction": widespread_fraction,
            "missing_source_fragments": (
                missing_numerical_guard_fragments
            ),
        },
    )

    passed = sum(row["status"] == "PASS" for row in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_scaleup_pretract_recovery4_v3_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "imaging_executed": False,
        "human_visual_qc_inferred": False,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "source": module.file_record(SOURCE),
        "dryrun_plan": module.file_record(
            module.OUTPUT_ROOT / "manifests/pretract_recovery4_plan.json"
        ),
    }
    module.atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
