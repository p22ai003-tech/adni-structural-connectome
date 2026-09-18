#!/home/ec2-user/fsl/bin/python
"""Validate the Recovery4-aware selected-recipe tractography runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any


SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "hcp379_scaleup_tractography_recovery4_v3.py"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_scaleup_tractography_recovery4_v3/validation.json"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_scaleup_tractography_recovery4_v3_validation_target",
        SOURCE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module()
    module.install_recovery4_adapter()
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if condition else "FAIL",
            "evidence": evidence,
        }

    rows, overlap = module.validate_prephase_scope()
    check(
        "exact_noncanary_scope",
        len(rows) == module.EXPECTED_SCALEUP_N
        and len({row["unit"] for row in rows})
        == module.EXPECTED_SCALEUP_N
        and len(overlap) == module.EXPECTED_CANARY_OVERLAP_N,
        {
            "noncanary_n": len(rows),
            "unique_n": len({row["unit"] for row in rows}),
            "canary_overlap_n": len(overlap),
        },
    )

    plan = module.PRETRACT.load_json(module.TRACT_PLAN)
    check(
        "prephase_plan_is_nonexecuting",
        plan.get("status") == "PRE_PHASE_DRY_RUN_PASS"
        and plan.get("execution_authorized") is False
        and plan.get("imaging_executed_by_plan") is False
        and plan.get("selected_streamline_count") is None,
        {
            "status": plan.get("status"),
            "execution_authorized": plan.get("execution_authorized"),
            "imaging_executed_by_plan": plan.get(
                "imaging_executed_by_plan"
            ),
            "selected_streamline_count": plan.get(
                "selected_streamline_count"
            ),
        },
    )
    check(
        "phase_b_is_not_fabricated",
        all(
            not value["present"] and value["status"] == "NOT_RUN"
            for value in plan["phase_b_status"].values()
        ),
        plan["phase_b_status"],
    )
    check(
        "allowed_count_selection_contract",
        plan.get("allowed_streamline_counts")
        == list(module.ALLOWED_COUNTS)
        and "smallest Phase-B density-qualified stable count"
        in plan.get("selection_rule", ""),
        {
            "allowed": plan.get("allowed_streamline_counts"),
            "selection_rule": plan.get("selection_rule"),
        },
    )
    check(
        "exact_nine_matrix_contract",
        tuple(plan.get("matrix_names", []))
        == tuple(module.ENGINE.MATRIX_NAMES)
        and len(plan.get("matrix_names", [])) == 9,
        plan.get("matrix_names"),
    )
    check(
        "density_is_not_subject_exclusion",
        plan.get("density_is_subject_inclusion_gate") is False,
        {
            "density_is_subject_inclusion_gate": plan.get(
                "density_is_subject_inclusion_gate"
            )
        },
    )
    adapter = plan["pretract_adapter_contract"]
    check(
        "recovery4_adapter_uses_promoted_inputs",
        adapter.get("required_status") == "PASS_PRETRACT_RECOVERY4"
        and adapter.get("selected_five_tt") is True
        and adapter.get("selected_hcp379") is True
        and adapter.get("bounded_fa_md_rd_ad") is True
        and adapter.get("raw_tensor_maps_are_not_sampled") is True,
        adapter,
    )
    check(
        "plan_has_no_diagnosis_or_outcome",
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
        "exact_corrected_release_scope",
        plan.get("target_noncanary_n") == 216
        and plan.get("canary_overlap_n") == 11
        and plan.get("corrected_release_n") == 227
        and len(plan.get("units", [])) == 216,
        {
            "target_noncanary_n": plan.get("target_noncanary_n"),
            "canary_overlap_n": plan.get("canary_overlap_n"),
            "corrected_release_n": plan.get("corrected_release_n"),
            "unit_n": len(plan.get("units", [])),
        },
    )
    check(
        "tool_contract_is_hash_bound",
        set(plan.get("tool_contract", {}))
        == {
            "mrinfo",
            "tckgen",
            "tckinfo",
            "tcksift2",
            "tck2connectome",
            "tcksample",
        }
        and all(
            len(record.get("sha256", "")) == 64
            for record in plan.get("tool_contract", {}).values()
        ),
        {"tools": sorted(plan.get("tool_contract", {}))},
    )
    check(
        "source_contracts_are_hash_bound",
        all(
            len(plan[name].get("sha256", "")) == 64
            for name in (
                "builder",
                "matrix_engine",
                "pretract_runner",
                "phase_b_rules",
                "config",
            )
        )
        and set(plan.get("counter_helpers", {}))
        == {"tck_header_count", "numeric_value_count"}
        and all(
            len(record.get("sha256", "")) == 64
            for record in plan.get("counter_helpers", {}).values()
        ),
        {
            "sources": {
                name: plan[name].get("sha256")
                for name in (
                    "builder",
                    "matrix_engine",
                    "pretract_runner",
                    "phase_b_rules",
                    "config",
                )
            },
            "counter_helpers": {
                name: record.get("sha256")
                for name, record in plan.get(
                    "counter_helpers", {}
                ).items()
            },
        },
    )

    phase_text = module.PHASE_RULES_SOURCE.read_text(encoding="utf-8")
    engine_text = module.ENGINE_SOURCE.read_text(encoding="utf-8")
    parity_fragments = (
        "-algorithm",
        "iFOD2",
        "-act",
        "-backtrack",
        "-crop_at_gmwmi",
        "-seed_dynamic",
        "-reg_tikhonov",
        "0.0",
        "-reg_tv",
        "0.1",
        "-assignment_radial_search",
        "4",
        "-scale_invlength",
        "-stat_tck",
        "mean",
        "tck_header_count_v1.py",
        "count_mrtrix_numeric_values_v1.py",
    )
    missing_phase = [
        value for value in parity_fragments if value not in phase_text
    ]
    missing_engine = [
        value for value in parity_fragments if value not in engine_text
    ]
    check(
        "phase_scaleup_command_parity",
        not missing_phase
        and not missing_engine
        and "wc -l" not in phase_text
        and "wc -l" not in engine_text,
        {
            "missing_phase": missing_phase,
            "missing_engine": missing_engine,
            "phase_has_wc_line_count": "wc -l" in phase_text,
            "engine_has_wc_line_count": "wc -l" in engine_text,
        },
    )

    seed_row = {
        "dti_source_id": "a" * 64,
        "dti_raw_bundle_sha256": "b" * 64,
    }
    recipe_id = "synthetic-recipe"
    token, observed_seed = module.ENGINE.subject_seed(seed_row, recipe_id)
    expected_seed = (
        int.from_bytes(
            hashlib.sha256(token.encode("utf-8")).digest()[:4], "big"
        )
        & 0x7FFFFFFF
    )
    check(
        "deterministic_seed_matches_phase_formula",
        observed_seed == expected_seed and token.endswith(recipe_id),
        {
            "observed_seed": observed_seed,
            "expected_seed": expected_seed,
        },
    )

    self_test = subprocess.run(
        [
            "/home/ec2-user/fsl/bin/python",
            str(module.ENGINE_SOURCE),
            "--self-test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    self_payload = (
        json.loads(self_test.stdout)
        if self_test.returncode == 0 and self_test.stdout.strip()
        else {}
    )
    check(
        "matrix_engine_self_test",
        self_test.returncode == 0
        and self_payload.get("status") == "PASS",
        {
            "returncode": self_test.returncode,
            "payload": self_payload,
            "stderr": self_test.stderr.strip(),
        },
    )

    try:
        module.validate_full_gate(module.ROOT)
    except (FileNotFoundError, ValueError) as exc:
        early_execute_rejected = (
            "pretract" in str(exc).lower()
            or "phase-b" in str(exc).lower()
        )
        early_execute_evidence = str(exc)
    else:
        early_execute_rejected = False
        early_execute_evidence = "unexpectedly accepted"
    check(
        "execute_rejects_incomplete_upstream_gates",
        early_execute_rejected,
        early_execute_evidence,
    )

    generated_tracks = list(
        module.ROOT.glob("subjects/*/07_tractography/*.tck")
        if (module.ROOT / "subjects").is_dir()
        else []
    )
    generated_matrices = list(
        module.ROOT.glob("subjects/*/08_connectome/*.csv")
        if (module.ROOT / "subjects").is_dir()
        else []
    )
    check(
        "dryrun_generated_no_tracks_or_matrices",
        not generated_tracks and not generated_matrices,
        {
            "tractogram_n": len(generated_tracks),
            "matrix_n": len(generated_matrices),
        },
    )

    passed = sum(value["status"] == "PASS" for value in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_scaleup_tractography_recovery4_v3_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": module.PRETRACT.utc_now(),
        "imaging_executed": False,
        "human_visual_qc_inferred": False,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "source": module.PRETRACT.file_record(SOURCE),
        "prephase_plan": module.PRETRACT.file_record(module.TRACT_PLAN),
    }
    module.PRETRACT.atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
