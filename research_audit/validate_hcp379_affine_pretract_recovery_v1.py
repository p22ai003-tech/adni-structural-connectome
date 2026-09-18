#!/usr/bin/env python3
"""Validate fail-closed affine recovery for dual rigid-route failures."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "run_hcp379_affine_pretract_recovery_v1.py"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_affine_pretract_recovery_v1/validation.json"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_affine_recovery_validation_target", SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module()
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    compile_probe = subprocess.run(
        [str(module.PYTHON), "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    self_test = subprocess.run(
        [str(module.PYTHON), str(SOURCE), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    check(
        "source_compiles_and_self_test_passes",
        compile_probe.returncode == 0
        and self_test.returncode == 0
        and "HCP379_AFFINE_PRETRACT_RECOVERY_SELF_TEST_PASS"
        in self_test.stdout,
        {
            "compile_returncode": compile_probe.returncode,
            "compile_stderr": compile_probe.stderr,
            "self_test_returncode": self_test.returncode,
            "self_test_stdout": self_test.stdout,
        },
    )
    preflight = module.preflight(module.DEFAULT_HUMAN_QC)
    targets = preflight.get("target_units", [])
    records = preflight.get("records", {})
    check(
        "targets_come_only_from_exact_dual_route_failure_class",
        (
            preflight.get("status")
            in {
                "READY_FOR_EXACT_AFFINE_RECOVERY",
                "NO_CURRENT_AFFINE_RECOVERY_TARGETS",
            }
            and preflight.get("failure_class")
            == module.FAILURE_CLASS
            and preflight.get("target_n") == len(targets)
            and len(set(targets)) == len(targets)
        ),
        {
            "status": preflight.get("status"),
            "target_n": preflight.get("target_n"),
            "target_units": targets,
        },
    )
    check(
        "validated_515_row_ledger_csv_is_hash_bound",
        (
            isinstance(records, dict)
            and records.get("ledger_csv")
            == module.file_record(module.LEDGER_CSV)
        ),
        records.get("ledger_csv")
        if isinstance(records, dict)
        else records,
    )
    policy = preflight.get("policy", {})
    check(
        "affine_is_third_route_with_unchanged_qc",
        (
            policy.get("selection")
            == "only after BBR and seeded rigid direct QC both fail"
            and policy.get("transform") == "seeded_ants_affine"
            and policy.get("ants_transform_type") == "a"
            and policy.get("per_subject_tuning_allowed") is False
            and policy.get("engine_recomputation_allowed") is False
            and policy.get("tissue_qc_threshold_change_allowed") is False
            and policy.get("atlas_qc_threshold_change_allowed") is False
            and policy.get("visual_qc_required") is True
            and policy.get("mechanism_guard_authorization")
            == (
                "only the exact current ledger target whose failure "
                "class equals SPATIAL_BBR_AND_RIGID_FAIL"
            )
        ),
        policy,
    )
    engine_text = module.ENGINE_SOURCE.read_text(encoding="utf-8")
    runner_text = SOURCE.read_text(encoding="utf-8")
    check(
        "mechanism_authorization_binds_held_state_and_prior_archive",
        all(
            fragment in runner_text
            for fragment in (
                "validated_515_row_ledger_plus_hash_bound_held_state",
                '"subject_state": file_record(state_path)',
                '"prior_state_archive": dict(prior_archive)',
                '"ledger_csv": file_record(LEDGER_CSV)',
            )
        ),
        {"required_fragments_present": True},
    )
    check(
        "engine_has_disjoint_affine_namespace_and_fixed_route",
        all(
            fragment in engine_text
            for fragment in (
                "def ants_affine_variant_paths",
                "recovery4_ants_affine",
                'transform_type="a"',
                'route_name="seeded_ants_affine_recovery"',
                "allow_affine_failover",
            )
        ),
        {"required_fragments_present": True},
    )
    check(
        "preflight_is_non_imaging_and_diagnosis_blind",
        (
            preflight.get("diagnosis_labels_used") is False
            and preflight.get("outcomes_used") is False
            and preflight.get("connectome_density_used") is False
            and preflight.get("non_overwriting") is True
            and preflight.get("imaging_executed") is False
            and preflight.get("tractography_generated") is False
            and preflight.get("matrix_generated") is False
        ),
        {
            key: preflight.get(key)
            for key in (
                "diagnosis_labels_used",
                "outcomes_used",
                "connectome_density_used",
                "non_overwriting",
                "imaging_executed",
                "tractography_generated",
                "matrix_generated",
            )
        },
    )
    completed_attempts = sorted(
        (
            path
            for path in module.ATTEMPTS.glob(
                "*-affine-pretract-recovery"
            )
            if (path / "attempt.json").is_file()
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
    latest_attempt = (
        completed_attempts[-1] if completed_attempts else None
    )
    latest = (
        module.load_json(latest_attempt / "attempt.json")
        if latest_attempt is not None
        else {}
    )
    result_rows = latest.get("results", [])
    result = (
        result_rows[0]
        if isinstance(result_rows, list) and len(result_rows) == 1
        else {}
    )
    state = result.get("state", {}) if isinstance(result, dict) else {}
    compaction = (
        result.get("compaction", {})
        if isinstance(result, dict)
        else {}
    )
    check(
        "latest_completed_attempt_passes_exact_affine_route",
        (
            latest.get("status") == "PASS"
            and latest.get("target_n") == 1
            and latest.get("pass_n") == 1
            and latest.get("engine_recomputed") is False
            and state.get("status") == "PASS_PRETRACT_RECOVERY4"
            and state.get("registration_route")
            == "seeded_ants_affine_recovery"
            and state.get("diagnosis_labels_used") is False
            and state.get("non_overwriting") is True
            and latest.get("diagnosis_labels_used") is False
            and latest.get("outcomes_used") is False
            and latest.get("connectome_density_used") is False
            and latest.get("non_overwriting") is True
            and compaction.get("status") == "PASS_COMPACTED"
            and compaction.get("registration_route")
            == "seeded_ants_affine_recovery"
        ),
        {
            "attempt": str(latest_attempt)
            if latest_attempt is not None
            else None,
            "status": latest.get("status"),
            "target_n": latest.get("target_n"),
            "pass_n": latest.get("pass_n"),
            "unit": result.get("unit")
            if isinstance(result, dict)
            else None,
            "state_status": state.get("status"),
            "registration_route": state.get("registration_route"),
            "compaction_status": compaction.get("status"),
        },
    )
    live_state_matches = False
    live_compaction_matches = False
    atlas_qc: dict[str, Any] = {}
    if isinstance(result, dict) and result.get("unit") and result.get("root"):
        unit = str(result["unit"])
        root = Path(str(result["root"]))
        state_path = root / "qc/subjects" / f"{unit}.json"
        compaction_path = (
            root / "qc/pretract_compaction" / f"{unit}.json"
        )
        live_state_matches = (
            state_path.is_file()
            and module.load_json(state_path) == state
        )
        live_compaction_matches = (
            compaction_path.is_file()
            and module.load_json(compaction_path) == compaction
        )
        atlas_qc_path = (
            root
            / "subjects"
            / unit
            / "06_preflight/"
            "ants_affine_hcp379_atlas_qc_recovery4.json"
        )
        if atlas_qc_path.is_file():
            atlas_qc = module.load_json(atlas_qc_path)
    hemisphere = atlas_qc.get(
        "cortical_hemisphere_inside_dwi_mask_fraction", {}
    )
    check(
        "live_state_compaction_and_unchanged_qc_replay",
        (
            live_state_matches
            and live_compaction_matches
            and atlas_qc.get("status") == "PASS"
            and atlas_qc.get("labels_found") == 379
            and atlas_qc.get("missing_labels") == []
            and float(
                atlas_qc.get(
                    "atlas_inside_dwi_mask_fraction", 0.0
                )
            )
            >= 0.8
            and isinstance(hemisphere, dict)
            and float(hemisphere.get("left", 0.0)) >= 0.8
            and float(hemisphere.get("right", 0.0)) >= 0.8
            and float(state.get("selected_5tt_dwi_mask_dice", 0.0))
            >= 0.8
        ),
        {
            "live_state_matches_attempt": live_state_matches,
            "live_compaction_matches_attempt": live_compaction_matches,
            "atlas_qc_status": atlas_qc.get("status"),
            "labels_found": atlas_qc.get("labels_found"),
            "atlas_inside_dwi_mask_fraction": atlas_qc.get(
                "atlas_inside_dwi_mask_fraction"
            ),
            "cortical_hemisphere_fraction": hemisphere,
            "selected_5tt_dwi_mask_dice": state.get(
                "selected_5tt_dwi_mask_dice"
            ),
        },
    )
    prior_qc = (
        result.get("prior_qc", {})
        if isinstance(result, dict)
        else {}
    )
    prior_errors: list[str] = []
    if not isinstance(prior_qc, dict) or not prior_qc:
        prior_errors.append("prior_qc_missing")
    else:
        for name, binding in prior_qc.items():
            if not isinstance(binding, dict):
                prior_errors.append(f"{name}:binding_type")
                continue
            source_record = binding.get("source")
            archive_record = binding.get("archive")
            if not isinstance(source_record, dict) or not isinstance(
                archive_record, dict
            ):
                prior_errors.append(f"{name}:record_type")
                continue
            archive_path = Path(str(archive_record.get("path", "")))
            try:
                replayed_archive = module.file_record(archive_path)
            except Exception as exc:
                prior_errors.append(f"{name}:archive:{exc}")
                continue
            if replayed_archive != archive_record:
                prior_errors.append(f"{name}:archive_record")
            if (
                archive_record.get("sha256")
                != source_record.get("sha256")
                or archive_record.get("size_bytes")
                != source_record.get("size_bytes")
            ):
                prior_errors.append(f"{name}:source_archive_binding")
    check(
        "prior_failure_and_scalar_qc_are_immutably_archived",
        not prior_errors,
        {
            "attempt": str(latest_attempt)
            if latest_attempt is not None
            else None,
            "archive_names": sorted(prior_qc)
            if isinstance(prior_qc, dict)
            else [],
            "errors": prior_errors,
        },
    )
    passed = sum(row["status"] == "PASS" for row in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_affine_pretract_recovery_validation",
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
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
