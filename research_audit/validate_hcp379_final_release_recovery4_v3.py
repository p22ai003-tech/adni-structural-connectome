#!/home/ec2-user/fsl/bin/python
"""Validate Recovery4 bindings and fail-closed behavior of final assembly."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/build_hcp379_final_release_v2.py"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_final_release_recovery4_v3/validation.json"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_final_release_recovery4_v3_validation_target",
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

    routes = module.route_identity()
    union = set().union(*routes.values())
    overlap = set()
    values = list(routes.values())
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            overlap |= left & right
    check(
        "exact_disjoint_530_route_identity",
        len(union) == module.EXPECTED_N
        and not overlap
        and {name: len(value) for name, value in routes.items()}
        == {
            "keep": 214,
            "archive": 86,
            "remap": 111,
            "direct_corrected": 116,
            "tensor": 2,
            "freesurfer": 1,
        },
        {
            "union_n": len(union),
            "overlap_n": len(overlap),
            "route_counts": {
                name: len(value) for name, value in routes.items()
            },
        },
    )

    readiness, returned_routes = module.current_readiness()
    expected_blockers = {
        "archive_route_concordance_decision_missing",
        "corrected_227_release_not_pass",
        "corrected_archive_low_30_release_not_pass",
        "corrected_freesurfer_1_release_not_pass",
        "corrected_legacy_tensor_216_release_not_pass",
        "phase_b_recipe_stability_not_pass",
        "real_canary_human_visual_qc_missing",
    }
    check(
        "current_readiness_fails_closed",
        readiness.get("status") == "NOT_READY"
        and set(readiness.get("blockers", [])) == expected_blockers
        and returned_routes == routes,
        {
            "status": readiness.get("status"),
            "blockers": readiness.get("blockers"),
        },
    )
    check(
        "recovery4_paths_are_authoritative",
        "review_recovery4" in str(module.HUMAN_CANARY_QC)
        and "corrected_scaleup_recovery4" in str(module.CORRECTED_227)
        and "corrected_archive_recovery4"
        in str(module.CORRECTED_ARCHIVE_86)
        and "pretract_recovery4" in str(
            module.RECOVERY4_PRETRACT_MASTER
        ),
        {
            "human_qc": str(module.HUMAN_CANARY_QC),
            "corrected_227": str(module.CORRECTED_227),
            "corrected_archive_86": str(
                module.CORRECTED_ARCHIVE_86
            ),
            "pretract_master": str(module.RECOVERY4_PRETRACT_MASTER),
        },
    )
    check(
        "readiness_binds_recovery4_master",
        readiness["records"]["recovery4_pretract_master"]
        == module.file_record(module.RECOVERY4_PRETRACT_MASTER)
        and readiness["records"]["human_canary_qc"] is None
        and readiness["records"]["corrected_227_release"] is None,
        {
            "master": readiness["records"][
                "recovery4_pretract_master"
            ],
            "human": readiness["records"]["human_canary_qc"],
            "corrected": readiness["records"][
                "corrected_227_release"
            ],
        },
    )
    check(
        "readiness_binds_recovered_freesurfer_source_only",
        readiness["records"]["freesurfer_hcp_source_attestation"]
        == module.file_record(module.FREESURFER_SOURCE_ATTESTATION)
        and "freesurfer_hcp_source_not_ready"
        not in readiness.get("blockers", []),
        {
            "source": readiness["records"].get(
                "freesurfer_hcp_source_attestation"
            ),
            "blockers": readiness.get("blockers"),
        },
    )
    check(
        "exact_release_shape_contract",
        module.EXPECTED_N == 530
        and module.EXPECTED_NODES == 379
        and module.EXPECTED_MATRICES == 4_770
        and len(module.MATRIX_NAMES) == 9
        and module.MINIMUM_EDGE_DENSITY == 0.60,
        {
            "subjects": module.EXPECTED_N,
            "nodes": module.EXPECTED_NODES,
            "matrices": module.EXPECTED_MATRICES,
            "families": list(module.MATRIX_NAMES),
            "minimum_edge_density": module.MINIMUM_EDGE_DENSITY,
        },
    )
    check(
        "readiness_binds_hard_density_contract",
        readiness.get("minimum_edge_density_inclusive") == 0.60
        and readiness.get(
            "required_subjects_at_or_above_minimum_density"
        )
        == 530
        and readiness.get(
            "density_is_whole_cohort_technical_release_gate"
        )
        is True
        and readiness.get("density_is_subject_exclusion_gate") is False
        and readiness["records"]["density_recovery_plan"]
        == module.file_record(module.DENSITY_RECOVERY_PLAN),
        {
            "minimum": readiness.get("minimum_edge_density_inclusive"),
            "required_n": readiness.get(
                "required_subjects_at_or_above_minimum_density"
            ),
            "whole_cohort_gate": readiness.get(
                "density_is_whole_cohort_technical_release_gate"
            ),
            "subject_exclusion_gate": readiness.get(
                "density_is_subject_exclusion_gate"
            ),
            "plan": readiness["records"].get("density_recovery_plan"),
        },
    )
    execution_lanes = module.density_execution_lane_units()
    check(
        "readiness_binds_exact_density_execution_topology",
        readiness["records"]["density_execution_topology"]
        == module.file_record(module.DENSITY_EXECUTION_TOPOLOGY)
        and readiness["records"]["density_execution_subjects"]
        == module.file_record(module.DENSITY_EXECUTION_SUBJECTS)
        and {name: len(units) for name, units in execution_lanes.items()}
        == {
            "corrected_core_227": 227,
            "corrected_legacy_tensor_216": 216,
            "corrected_archive_low_30": 30,
            "conditional_archive_D0_56": 56,
            "corrected_freesurfer_1": 1,
        }
        and readiness.get("legacy_tensor_route_decision")
        == "CORRECTED_ACT_ALL_216",
        {
            "lanes": {
                name: len(units)
                for name, units in execution_lanes.items()
            },
            "legacy_tensor_route_decision": readiness.get(
                "legacy_tensor_route_decision"
            ),
        },
    )
    check(
        "phase_count_contract",
        module.ALLOWED_STREAMLINE_COUNTS
        == (3_000_000, 5_000_000, 10_000_000),
        list(module.ALLOWED_STREAMLINE_COUNTS),
    )

    master = module.load_json(module.RECOVERY4_PRETRACT_MASTER)
    units = sorted(str(row["unit"]) for row in master["units"])
    original_human_path = module.HUMAN_CANARY_QC
    with tempfile.TemporaryDirectory(
        prefix="hcp379-final-human-validator-"
    ) as tmp:
        valid = Path(tmp) / "synthetic_parser_control.csv"
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with valid.open("w", newline="", encoding="utf-8") as handle:
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
            for unit in units:
                writer.writerow(
                    {
                        "unit": unit,
                        "status": "PASS",
                        "reviewer": "synthetic-parser-control",
                        "reviewed_utc": now,
                        "notes": "not real review evidence",
                    }
                )
        module.HUMAN_CANARY_QC = valid
        record = module.validate_human_canary_qc()
        check(
            "human_qc_parser_positive_control",
            record["size_bytes"] > 0 and len(record["sha256"]) == 64,
            {"purpose": "parser control only", "recorded": True},
        )

        forbidden = Path(tmp) / "forbidden.csv"
        with forbidden.open("w", newline="", encoding="utf-8") as handle:
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
            for unit in units:
                writer.writerow(
                    {
                        "unit": unit,
                        "status": "PASS",
                        "reviewer": "synthetic-parser-control",
                        "reviewed_utc": now,
                        "diagnosis": "FORBIDDEN",
                    }
                )
        module.HUMAN_CANARY_QC = forbidden
        try:
            module.validate_human_canary_qc()
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
    module.HUMAN_CANARY_QC = original_human_path

    try:
        module.assemble(
            module.RELEASE_ROOT,
            readiness,
            routes,
            False,
            module.READINESS_OUTPUT,
        )
    except ValueError as exc:
        assembly_rejected = "readiness is not PASS" in str(exc)
        assembly_evidence = str(exc)
    else:
        assembly_rejected = False
        assembly_evidence = "unexpectedly assembled"
    check(
        "assembly_rejects_not_ready_state",
        assembly_rejected,
        assembly_evidence,
    )
    check(
        "no_false_release_manifest",
        not (module.RELEASE_ROOT / "release_manifest.json").is_file(),
        {
            "path": str(module.RELEASE_ROOT / "release_manifest.json"),
            "present": (
                module.RELEASE_ROOT / "release_manifest.json"
            ).is_file(),
        },
    )

    self_test = subprocess.run(
        [
            "/home/ec2-user/fsl/bin/python",
            str(SOURCE),
            "--self-test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    check(
        "matrix_contract_self_test",
        self_test.returncode == 0
        and "SELF_TEST_PASS" in self_test.stdout,
        {
            "returncode": self_test.returncode,
            "stdout": self_test.stdout.strip(),
            "stderr": self_test.stderr.strip(),
        },
    )

    passed = sum(value["status"] == "PASS" for value in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_final_release_recovery4_v3_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": module.utc_now(),
        "assembly_executed": False,
        "human_visual_qc_inferred": False,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "source": module.file_record(SOURCE),
        "readiness": module.file_record(module.READINESS_OUTPUT),
    }
    module.atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
