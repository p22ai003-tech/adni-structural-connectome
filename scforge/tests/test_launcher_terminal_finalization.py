from __future__ import annotations

import csv
import copy
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import yaml

from scforge.provenance import file_record, sha256_file, validate_schema
from scforge.response_calibration import (
    build_valid_pool_technical_diversity,
    load_execution_subset_technical_metadata,
    technical_diversity_sha256,
)
from tests.test_terminal_provenance_v2 import RECIPE_ID, RUN_ID, pass_record
from workflow.run_connectome_v2 import (
    ENVIRONMENT_CONTRACT,
    NORMATIVE_CONFIG,
    PROVENANCE_SCHEMA,
    _normalization_failure_evidence,
    build_snakemake_command,
    derive_lineage_id,
    derive_mode_run_id,
    execute_command_with_h04a_monitor,
    finalize_response_calibration_phase_a,
    finalize_pre_tractography_canary,
    finalize_terminal_publication,
    inject_phase_b_calibration,
    measure_run_root_storage,
    prepare_h04a_resource_preflight,
    prepare_phase_b_binding,
    prepare_execution_binding,
    prepare_tractography_continuation_binding,
    require_h04a_mode_authorization,
    run,
    validate_normative_execution_lock,
    validate_phase_b_calibration,
)


def identity(unit: str, *, dti: str, t1: str) -> dict[str, str]:
    return {
        "unit": unit,
        "subject_id": unit.split("_I")[0],
        "diagnosis_at_dti": "CN",
        "dti_image_id": dti,
        "t1_image_id": t1,
        "abs_pair_gap_days": "0",
        "dti_raw_bundle_sha256": "a" * 64,
        "t1_raw_bundle_sha256": "b" * 64,
        "pair_content_bundle_sha256": "c" * 64,
    }


class LauncherTerminalFinalizationTests(unittest.TestCase):
    def test_normative_recipe_remains_false_and_cannot_be_toggled(self) -> None:
        candidate = {
            "contract": {
                "status": "implementation_candidate_pending_validation",
                "imaging_execution_authorized": False,
                "current_human_gate": "SL-H03-C1",
                "authorization_scope": "design_only_no_imaging",
                "next_execution_gate": "SL-H04A",
                "failure_policy": "fail_closed",
                "silent_fallbacks_allowed": False,
            }
        }
        validate_normative_execution_lock(candidate)
        toggled = copy.deepcopy(candidate)
        toggled["contract"]["imaging_execution_authorized"] = True
        with self.assertRaisesRegex(ValueError, "must remain immutable"):
            validate_normative_execution_lock(toggled)

    def test_pending_h04a_decision_creates_no_run_state_or_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "scforge_v2_pending_auth"
            parent_manifest = root / "parent.csv"
            subset_manifest = root / "subset.csv"
            decision = root / "pending.json"
            parent_manifest.write_text("fixture parent\n", encoding="utf-8")
            subset_manifest.write_text("fixture subset\n", encoding="utf-8")
            decision.write_text(
                json.dumps(
                    {
                        "schema_version": "2.0.0",
                        "decision_type": (
                            "connectome_canary_execution_subset_approval"
                        ),
                        "status": "PENDING_USER_DECISION",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = [
                identity(
                    f"S{index}_I{index}",
                    dti=str(index),
                    t1=str(index + 100),
                )
                for index in range(1, 13)
            ]
            args = Namespace(
                manifest=parent_manifest,
                run_root=run_root,
                human_qc_manifest=None,
                execution_subset_manifest=subset_manifest,
                execution_subset_decision=decision,
                cores=1,
                mode="response-calibration-phase-a",
                response_calibration_manifest=None,
                response_calibration_minimum=None,
                response_calibration_decision=None,
                tractography_continuation_decision=None,
            )
            with (
                mock.patch(
                    "workflow.run_connectome_v2.load_acquisition_manifest",
                    side_effect=(tuple(rows), tuple(rows)),
                ),
                mock.patch(
                    "workflow.run_connectome_v2.subprocess.Popen"
                ) as popen,
            ):
                with self.assertRaisesRegex(ValueError, "differs at status"):
                    run(args)
            self.assertFalse(run_root.exists())
            popen.assert_not_called()

    def _execution_binding(
        self, root: Path, units: list[str]
    ) -> dict[str, object]:
        fixture_dir = root / "execution_fixture"
        fixture_dir.mkdir(parents=True, exist_ok=True)
        parent = fixture_dir / "parent.csv"
        subset = fixture_dir / "subset.csv"
        decision = fixture_dir / "decision.json"
        manifest_text = (
            "unit,manufacturer,t1_source_kind\n"
            + "".join(
                f"{unit},{'SIEMENS' if index % 2 == 0 else 'GE MEDICAL SYSTEMS'},"
                f"{'dicom_series' if index % 2 == 0 else 'nifti_single'}\n"
                for index, unit in enumerate(sorted(units))
            )
        )
        parent.write_text(manifest_text, encoding="utf-8")
        subset.write_bytes(parent.read_bytes())
        decision.write_text("{}\n", encoding="utf-8")
        return {
            "schema_version": "2.0.0",
            "binding_type": "connectome_execution_subset_binding",
            "execution_scope": "canary",
            "recipe_id": RECIPE_ID,
            "parent_acquisition_manifest": file_record(parent),
            "execution_subset_manifest": file_record(subset),
            "execution_subset_decision": file_record(decision),
            "approved_unit_count": len(units),
            "units": sorted(units),
            "diagnosis_labels_used": False,
            "selection_locked": True,
            "h04a_authorization": {
                "minimum_valid_response_calibration_units": 12,
                "minimum_valid_manufacturer_families": 2,
                "minimum_valid_t1_source_classes": 2,
                "required_valid_t1_source_classes": [
                    "dicom_series",
                    "nifti_single",
                ],
            },
        }

    def _phase_a_fixture(
        self, root: Path
    ) -> tuple[
        list[dict[str, str]],
        Path,
        Path,
        Path,
        Path,
        dict[str, dict[str, object]],
        dict[str, object],
    ]:
        rows = sorted(
            (
                identity(
                    f"S{index}_I{index}", dti=str(index), t1=str(index + 10)
                )
                for index in range(1, 14)
            ),
            key=lambda row: row["unit"],
        )
        execution_binding = self._execution_binding(
            root, [row["unit"] for row in rows]
        )
        run_context = root / "contract" / "response_calibration_phase_a_run_context.json"
        run_context.parent.mkdir(parents=True)
        # the contract as the launcher reads it: ${repo} resolved and this
        # machine's tool sections laid over the reference
        from scforge.environment import merged_contract

        environment = merged_contract()
        run_context.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "status": "LOCKED",
                    "run_id": RUN_ID,
                    "lineage_id": "d" * 24,
                    "launcher_mode": "response-calibration-phase-a",
                    "recipe_id": RECIPE_ID,
                    "execution_binding": execution_binding,
                    "normative_config": file_record(NORMATIVE_CONFIG),
                    "environment_contract": file_record(ENVIRONMENT_CONTRACT),
                    "workflow_source_manifest": file_record(
                        environment["workflow"]["source_manifest"]["path"]
                    ),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        attempt_start = root / "attempts" / "attempt.start.json"
        attempt_start.parent.mkdir(parents=True)
        attempt_start.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "status": "STARTED",
                    "attempt_id": "phase-a-attempt-1",
                    "launcher_mode": "response-calibration-phase-a",
                    "run_id": RUN_ID,
                    "lineage_id": "d" * 24,
                    "recipe_id": RECIPE_ID,
                    "execution_binding": execution_binding,
                    "run_context": file_record(run_context),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        attempt_end = root / "attempts" / "attempt.end.json"
        attempt_end.write_text("{}\n", encoding="utf-8")
        execution_log = root / "attempts" / "attempt.snakemake.log"
        execution_log.write_text(
            "Error in rule normalize_dwi_source:\n"
            "    wildcards: unit=S13_I13\n"
            "    shell:\n"
            "        dcm2niix failed\n",
            encoding="utf-8",
        )

        response_shapes = {"wm": [2, 4], "gm": [2, 1], "csf": [2, 1]}
        first_responses: dict[str, dict[str, object]] = {}
        for index in range(1, 13):
            unit = f"S{index}_I{index}"
            unit_responses: dict[str, dict[str, object]] = {}
            for tissue, values in (
                ("wm", "1 2 3 4\n5 6 7 8\n"),
                ("gm", "3\n4\n"),
                ("csf", "5\n6\n"),
            ):
                path = (
                    root
                    / "subjects"
                    / unit
                    / "05_model"
                    / f"response_{tissue}.txt"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(values, encoding="ascii")
                unit_responses[tissue] = file_record(path)
            if index == 1:
                first_responses = unit_responses
            response_voxels = (
                root / "subjects" / unit / "05_model" / "response_voxels.mif"
            )
            response_voxels.write_bytes(b"test response voxel evidence")
            fod_shell_selection = (
                root
                / "subjects"
                / unit
                / "05_model"
                / "fod_shell_selection.json"
            )
            fod_shell_selection.write_text(
                json.dumps(
                    {
                        "schema_version": "2.0.0",
                        "status": "PASS",
                        "unit": unit,
                        "selected_shells_s_per_mm2": [0.0, 1000.0],
                        "selected_shell_sizes": [5, 30],
                        "bzero_threshold_s_per_mm2": 50.0,
                        "selection_uses_diagnosis_labels": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            pass_outcome = {
                "schema_version": "2.0.0",
                "record_type": "response_calibration_outcome",
                "status": "PASS",
                "generated_utc": "2026-07-18T01:00:30+00:00",
                "unit": unit,
                "recipe_id": RECIPE_ID,
                "diagnosis_blinded": True,
                "source_identity_hashes": {
                    "dti_raw_bundle_sha256": "a" * 64,
                    "t1_raw_bundle_sha256": "b" * 64,
                    "pair_content_bundle_sha256": "c" * 64,
                },
                "phase_a_identity": {
                    "run_id": RUN_ID,
                    "attempt_id": "phase-a-attempt-1",
                    "run_context": file_record(run_context),
                    "attempt_context": file_record(attempt_start),
                },
                "responses": unit_responses,
                "response_qc": {
                    "status": "PASS",
                    "diagnosis_labels_used": False,
                    "selected_voxel_counts": {"wm": 100, "gm": 80, "csf": 60},
                    "selected_voxel_count_validity_rule": (
                        "each_dhollander_csf_gm_wm_volume_count_must_be_positive"
                    ),
                    "stronger_absolute_minimum_applied": False,
                    "expected_shell_rows": 2,
                    "expected_coefficient_columns": {"wm": 4, "gm": 1, "csf": 1},
                    "observed_response_shapes": response_shapes,
                    "finite_coefficients": {"wm": True, "gm": True, "csf": True},
                    "positive_isotropic_coefficients": {
                        "wm": True,
                        "gm": True,
                        "csf": True,
                    },
                    "failure_reasons": [],
                },
                "response_voxels": file_record(response_voxels),
                "fod_shell_contract": {
                    "selection_record": file_record(fod_shell_selection),
                    "selected_shells_s_per_mm2": [0.0, 1000.0],
                    "selected_shell_sizes": [5, 30],
                    "bzero_threshold_s_per_mm2": 50.0,
                    "selection_uses_diagnosis_labels": False,
                },
                "invalid_values_replaced": False,
            }
            pass_outcome_path = (
                root
                / "subjects"
                / unit
                / "05_model"
                / "response_calibration_outcome.json"
            )
            pass_outcome_path.write_text(
                json.dumps(pass_outcome, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        marker_path = (
            root
            / "subjects"
            / "S13_I13"
            / "00_inputs"
            / "failures"
            / "phase-a-attempt-1"
            / "dwi_normalization_failure.json"
        )
        marker_path.parent.mkdir(parents=True)
        marker = {
            "schema_version": "2.0.0",
            "record_type": "source_normalization_failure",
            "status": "FAIL",
            "generated_utc": "2026-07-18T01:00:05+00:00",
            "unit": "S13_I13",
            "stage": "source_normalization",
            "normalizer": "dwi",
            "run_id": RUN_ID,
            "attempt_id": "phase-a-attempt-1",
            "attempt_context": file_record(attempt_start),
            "primary_failure_reason": "MissingPhaseEncodingDirection",
            "secondary_flags": ["no_value_invented"],
            "source_identity_hashes": {
                "dti_raw_bundle_sha256": "a" * 64,
                "pair_content_bundle_sha256": "c" * 64,
            },
        }
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        return (
            rows,
            run_context,
            attempt_start,
            attempt_end,
            execution_log,
            first_responses,
            execution_binding,
        )

    def test_stale_normalization_marker_hashes_are_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt_context = root / "attempts" / "current.start.json"
            attempt_context.parent.mkdir(parents=True)
            attempt_context.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "status": "STARTED",
                        "attempt_id": "current-attempt",
                        "run_id": RUN_ID,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            marker_path = (
                root
                / "subjects"
                / "S2_I2"
                / "00_inputs"
                / "failures"
                / "current-attempt"
                / "dwi_normalization_failure.json"
            )
            marker_path.parent.mkdir(parents=True)
            marker_path.write_text(
                json.dumps(
                    {
                        "schema_version": "2.0.0",
                        "record_type": "source_normalization_failure",
                        "status": "FAIL",
                        "generated_utc": "2026-07-18T01:00:05+00:00",
                        "unit": "S2_I2",
                        "stage": "source_normalization",
                        "normalizer": "dwi",
                        "run_id": RUN_ID,
                        "attempt_id": "current-attempt",
                        "attempt_context": file_record(attempt_context),
                        "primary_failure_reason": "stale reason",
                        "secondary_flags": [],
                        "source_identity_hashes": {"dti": "0" * 64},
                    }
                ),
                encoding="utf-8",
            )
            evidence = _normalization_failure_evidence(
                root,
                "S2_I2",
                {"dti": "1" * 64},
                attempt_context_path=attempt_context,
            )
            assert evidence is not None
            stage, reason, flags, records = evidence
            self.assertEqual(stage, "workflow_orchestration")
            self.assertIn("malformed_normalization_failure_marker", reason)
            self.assertIn("structured_failure_stage_untrusted", flags)
            self.assertIn("dwi_normalization_failure", records)

    def test_prior_failed_attempt_marker_is_ignored_after_corrected_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempts = root / "attempts"
            attempts.mkdir()
            prior_context = attempts / "prior.start.json"
            current_context = attempts / "current.start.json"
            for path, attempt_id in (
                (prior_context, "attempt-fail-a"),
                (current_context, "attempt-success-b"),
            ):
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": "1.0.0",
                            "status": "STARTED",
                            "attempt_id": attempt_id,
                            "run_id": RUN_ID,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            prior_marker = (
                root
                / "subjects"
                / "S2_I2"
                / "00_inputs"
                / "failures"
                / "attempt-fail-a"
                / "dwi_normalization_failure.json"
            )
            prior_marker.parent.mkdir(parents=True)
            prior_marker.write_text(
                json.dumps(
                    {
                        "schema_version": "2.0.0",
                        "record_type": "source_normalization_failure",
                        "status": "FAIL",
                        "generated_utc": "2026-07-18T01:00:05+00:00",
                        "unit": "S2_I2",
                        "stage": "source_normalization",
                        "normalizer": "dwi",
                        "run_id": RUN_ID,
                        "attempt_id": "attempt-fail-a",
                        "attempt_context": file_record(prior_context),
                        "primary_failure_reason": "prior failure",
                        "secondary_flags": [],
                        "source_identity_hashes": {
                            "dti_raw_bundle_sha256": "a" * 64,
                            "pair_content_bundle_sha256": "c" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            evidence = _normalization_failure_evidence(
                root,
                "S2_I2",
                {
                    "dti_raw_bundle_sha256": "a" * 64,
                    "t1_raw_bundle_sha256": "b" * 64,
                    "pair_content_bundle_sha256": "c" * 64,
                },
                attempt_context_path=current_context,
            )
            self.assertIsNone(evidence)
            self.assertTrue(prior_marker.is_file())

    def test_retry_uses_current_failure_and_preserves_prior_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempts = root / "attempts"
            attempts.mkdir()
            contexts: dict[str, Path] = {}
            markers: dict[str, Path] = {}
            for attempt_id, reason in (
                ("attempt-fail-a", "first failure"),
                ("attempt-fail-b", "second distinct failure"),
            ):
                context = attempts / f"{attempt_id}.start.json"
                context.write_text(
                    json.dumps(
                        {
                            "schema_version": "1.0.0",
                            "status": "STARTED",
                            "attempt_id": attempt_id,
                            "run_id": RUN_ID,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                marker = (
                    root
                    / "subjects"
                    / "S2_I2"
                    / "00_inputs"
                    / "failures"
                    / attempt_id
                    / "dwi_normalization_failure.json"
                )
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(
                    json.dumps(
                        {
                            "schema_version": "2.0.0",
                            "record_type": "source_normalization_failure",
                            "status": "FAIL",
                            "generated_utc": "2026-07-18T01:00:05+00:00",
                            "unit": "S2_I2",
                            "stage": "source_normalization",
                            "normalizer": "dwi",
                            "run_id": RUN_ID,
                            "attempt_id": attempt_id,
                            "attempt_context": file_record(context),
                            "primary_failure_reason": reason,
                            "secondary_flags": [],
                            "source_identity_hashes": {
                                "dti_raw_bundle_sha256": "a" * 64,
                                "pair_content_bundle_sha256": "c" * 64,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                contexts[attempt_id] = context
                markers[attempt_id] = marker
            evidence = _normalization_failure_evidence(
                root,
                "S2_I2",
                {
                    "dti_raw_bundle_sha256": "a" * 64,
                    "t1_raw_bundle_sha256": "b" * 64,
                    "pair_content_bundle_sha256": "c" * 64,
                },
                attempt_context_path=contexts["attempt-fail-b"],
            )
            assert evidence is not None
            stage, reason, _, records = evidence
            self.assertEqual(stage, "source_normalization")
            self.assertEqual(reason, "second distinct failure")
            self.assertEqual(
                records["dwi_normalization_failure"],
                file_record(markers["attempt-fail-b"]),
            )
            self.assertTrue(markers["attempt-fail-a"].is_file())

    def test_early_normalization_failure_reaches_complete_mixed_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scforge_v2_test_run"
            root.mkdir()
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"valid pass evidence")
            run_context = root / "contract" / "run_context.json"
            run_context.parent.mkdir()
            run_context.write_text("{}\n", encoding="utf-8")
            attempt_start = root / "attempts" / "attempt.start.json"
            attempt_start.parent.mkdir()
            attempt_start.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "status": "STARTED",
                        "started_utc": "2026-07-18T01:00:00+00:00",
                        "attempt_id": "full-attempt-1",
                        "run_id": RUN_ID,
                        "recipe_id": RECIPE_ID,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            execution_log = root / "attempts" / "attempt.snakemake.log"
            execution_log.write_text(
                "Error in rule normalize_dwi_source:\n"
                "    wildcards: unit=S2_I2\n"
                "    shell:\n"
                "        dcm2niix failed\n",
                encoding="utf-8",
            )
            manifest = root / "attempt_manifest.csv"
            rows = [
                identity("S1_I1", dti="1", t1="11"),
                identity("S2_I2", dti="2", t1="22"),
            ]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            pass_path = root / "subjects" / "S1_I1" / "08_qc" / "terminal_record.json"
            pass_path.parent.mkdir(parents=True)
            current_pass = pass_record(artifact, unit="S1_I1")
            current_pass["contract_files"] = {
                "run_context": file_record(run_context),
                "attempt_context": file_record(attempt_start),
            }
            pass_path.write_text(
                json.dumps(current_pass), encoding="utf-8"
            )

            marker_path = (
                root
                / "subjects"
                / "S2_I2"
                / "00_inputs"
                / "failures"
                / "full-attempt-1"
                / "dwi_normalization_failure.json"
            )
            marker_path.parent.mkdir(parents=True)
            marker = {
                "schema_version": "2.0.0",
                "record_type": "source_normalization_failure",
                "status": "FAIL",
                "generated_utc": "2026-07-18T01:00:05+00:00",
                "unit": "S2_I2",
                "stage": "source_normalization",
                "normalizer": "dwi",
                "run_id": RUN_ID,
                "attempt_id": "full-attempt-1",
                "attempt_context": file_record(attempt_start),
                "primary_failure_reason": "MissingPhaseEncodingDirection",
                "secondary_flags": ["no_value_invented"],
                "source_identity_hashes": {
                    "dti_raw_bundle_sha256": "a" * 64,
                    "pair_content_bundle_sha256": "c" * 64,
                },
            }
            marker_path.write_text(json.dumps(marker), encoding="utf-8")

            ledger = finalize_terminal_publication(
                run_root=root,
                manifest_rows=rows,
                config={
                    "manifest_path": str(manifest),
                    "registration": {},
                    "atlas": {},
                    "fod": {},
                    "tensor": {},
                    "tractography": {},
                    "sift2": {},
                    "connectome": {"assignment": {}},
                },
                run_id=RUN_ID,
                recipe_id=RECIPE_ID,
                run_context_path=run_context,
                manifest_path=manifest,
                attempt_start_path=attempt_start,
                execution_log_path=execution_log,
                workflow_returncode=1,
                ended_utc="2026-07-18T01:01:00+00:00",
            )
            self.assertEqual(
                ledger["summary"],
                {
                    "expected": 2,
                    "terminal": 2,
                    "pass": 1,
                    "partial": 0,
                    "fail": 1,
                    "analysis_ready": 1,
                    "outcome_valid_rows": 14,
                },
            )
            failed_path = root / "subjects" / "S2_I2" / "08_qc" / "terminal_record.json"
            failed = json.loads(failed_path.read_text(encoding="utf-8"))
            self.assertEqual(validate_schema(failed, PROVENANCE_SCHEMA), [])
            self.assertEqual(failed["status"], "FAIL")
            self.assertEqual(failed["terminal_stage"], "source_normalization")
            self.assertEqual(
                failed["primary_failure_reason"], "MissingPhaseEncodingDirection"
            )
            self.assertTrue(
                all(
                    outcome["status"] == "NA"
                    for outcome in failed["outcome_validity"].values()
                )
            )
            self.assertEqual(
                failed["execution"]["command_logs"], [file_record(execution_log)]
            )
            self.assertEqual(
                failed["intermediate_files"]["dwi_normalization_failure"],
                file_record(marker_path),
            )

    def test_phase_a_closes_every_outcome_without_full_run_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scforge_v2_phase_a"
            root.mkdir()
            (
                rows,
                run_context,
                attempt_start,
                attempt_end,
                execution_log,
                _,
                execution_binding,
            ) = self._phase_a_fixture(root)
            manifest = finalize_response_calibration_phase_a(
                run_root=root,
                manifest_rows=rows,
                execution_binding=execution_binding,
                run_id=RUN_ID,
                lineage_id="d" * 24,
                recipe_id=RECIPE_ID,
                run_context_path=run_context,
                attempt_start_path=attempt_start,
                attempt_end_path=attempt_end,
                execution_log_path=execution_log,
                workflow_returncode=1,
                ended_utc="2026-07-18T01:01:00+00:00",
            )
            self.assertEqual(
                manifest["summary"],
                {"expected": 13, "terminal": 13, "pass": 12, "fail": 1},
            )
            self.assertEqual(manifest["run_outcome"], "PARTIAL")
            self.assertFalse(manifest["full_run_terminal_publication"])
            self.assertFalse((root / "publication" / "run_ledger.jsonl").exists())
            self.assertFalse(
                (root / "subjects" / "S13_I13" / "08_qc" / "terminal_record.json").exists()
            )
            failed_outcome_path = (
                root
                / "subjects"
                / "S13_I13"
                / "05_model"
                / "response_calibration_outcome.json"
            )
            failed = json.loads(failed_outcome_path.read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "FAIL")
            self.assertEqual(failed["failure_stage"], "source_normalization")
            self.assertEqual(failed["responses"], {})
            self.assertEqual(failed["combined_execution_log"], file_record(execution_log))
            self.assertEqual(
                manifest["outcome_records"]["S13_I13"], file_record(failed_outcome_path)
            )

    def test_phase_a_freeze_phase_b_handoff_is_exact_and_nonmutating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scforge_v2_handoff"
            root.mkdir()
            (
                rows,
                run_context,
                attempt_start,
                attempt_end,
                execution_log,
                responses,
                execution_binding,
            ) = self._phase_a_fixture(root)
            phase_a_manifest = finalize_response_calibration_phase_a(
                run_root=root,
                manifest_rows=rows,
                execution_binding=execution_binding,
                run_id=RUN_ID,
                lineage_id="d" * 24,
                recipe_id=RECIPE_ID,
                run_context_path=run_context,
                attempt_start_path=attempt_start,
                attempt_end_path=attempt_end,
                execution_log_path=execution_log,
                workflow_returncode=1,
                ended_utc="2026-07-18T01:01:00+00:00",
            )
            publication = root / "publication"
            phase_a_manifest_path = publication / "response_calibration_phase_a_manifest.json"
            phase_a_ledger_path = publication / "response_calibration_phase_a_ledger.jsonl"
            completion_path = publication / "response_calibration_phase_a_completion.json"
            completion = {
                "schema_version": "2.0.0",
                "record_type": "response_calibration_phase_a_completion",
                "status": "PARTIAL",
                "mode": "response-calibration-phase-a",
                "run_id": RUN_ID,
                "lineage_id": "d" * 24,
                "recipe_id": RECIPE_ID,
                "execution_binding": execution_binding,
                "run_context": file_record(run_context),
                "terminal_attempt_start": file_record(attempt_start),
                "terminal_attempt_end": file_record(attempt_end),
                "phase_a_summary": phase_a_manifest["summary"],
                "phase_a_manifest": file_record(phase_a_manifest_path),
                "phase_a_ledger": file_record(phase_a_ledger_path),
                "freeze_required_before_phase_b": True,
                "full_run_terminal_publication": False,
            }
            completion_path.write_text(
                json.dumps(completion, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            frozen_dir = root / "frozen_calibration"
            frozen_dir.mkdir()
            pooled: dict[str, dict[str, object]] = {}
            response_shapes = {"wm": [2, 4], "gm": [2, 1], "csf": [2, 1]}
            for tissue in ("wm", "gm", "csf"):
                path = frozen_dir / f"pooled_response_{tissue}.txt"
                source = Path(str(responses[tissue]["path"]))
                path.write_bytes(source.read_bytes())
                pooled[tissue] = {
                    **file_record(path),
                    "shape": response_shapes[tissue],
                }
            frozen_manifest_path = (
                frozen_dir / "frozen_response_calibration_manifest.json"
            )
            valid_units = sorted(
                f"S{index}_I{index}" for index in range(1, 13)
            )
            frozen_inputs = {}
            for unit in valid_units:
                outcome_path = (
                    root
                    / "subjects"
                    / unit
                    / "05_model"
                    / "response_calibration_outcome.json"
                )
                outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
                frozen_inputs[unit] = {
                    tissue: {
                        **outcome["responses"][tissue],
                        "shape": response_shapes[tissue],
                    }
                    for tissue in ("wm", "gm", "csf")
                }
            subset_record, unit_metadata = load_execution_subset_technical_metadata(
                execution_binding
            )
            technical_diversity = build_valid_pool_technical_diversity(
                valid_units=valid_units,
                unit_metadata=unit_metadata,
                execution_subset_manifest=subset_record,
                minimum_valid_manufacturer_families=2,
                minimum_valid_t1_source_classes=2,
                required_valid_t1_source_classes=[
                    "dicom_series",
                    "nifti_single",
                ],
            )
            technical_diversity_digest = technical_diversity_sha256(
                technical_diversity
            )
            frozen = {
                "schema_version": "2.0.0",
                "record_type": "frozen_response_calibration",
                "status": "PASS",
                "generated_utc": "2026-07-18T01:02:00+00:00",
                "method": "mrtrix_responsemean_nonlegacy_scale_compensated",
                "responsemean_executable": file_record(
                    "/home/ec2-user/mrtrix3/bin/responsemean"
                ),
                "responsemean_legacy_option_used": False,
                "diagnosis_labels_used": False,
                "minimum_valid_subjects": 12,
                "total_outcome_count": 13,
                "valid_subject_count": 12,
                "invalid_subject_count": 1,
                "valid_units": valid_units,
                "invalid_units": ["S13_I13"],
                "inputs": frozen_inputs,
                "pooled_responses": pooled,
                "fod_shell_compatibility": {
                    "rule": "exact_MRtrix_clustered_b0_and_nonzero_centroids",
                    "maximum_centroid_difference_s_per_mm2": 0.0,
                    "selected_shells_s_per_mm2": [0.0, 1000.0],
                    "bzero_threshold_s_per_mm2": 50.0,
                    "candidate_shell_sizes": {
                        unit: [5, 30] for unit in valid_units
                    },
                },
                "valid_pool_technical_diversity": technical_diversity,
                "valid_pool_technical_diversity_sha256": technical_diversity_digest,
                "phase_a_lineage": {
                    "run_id": RUN_ID,
                    "recipe_id": RECIPE_ID,
                    "completion": file_record(completion_path),
                    "manifest": file_record(phase_a_manifest_path),
                    "attempt_start": file_record(attempt_start),
                    "expected_outcome_count": 13,
                    "terminal_outcome_count": 13,
                },
                "release_ready_for_phase_b": True,
                "phase_b_live_all_units_dependency": False,
                "dummy_response_files_used": False,
            }
            frozen_manifest_path.write_text(
                json.dumps(frozen, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            decision_path = root / "response_calibration_phase_b_approval.json"
            decision = {
                "schema_version": "2.0.0",
                "decision_type": "response_calibration_phase_b_approval",
                "status": "APPROVED",
                "recipe_id": RECIPE_ID,
                "phase_a_completion_sha256": sha256_file(completion_path),
                "response_calibration_manifest_sha256": sha256_file(
                    frozen_manifest_path
                ),
                "minimum_valid_subjects": 12,
                "minimum_valid_manufacturer_families": 2,
                "minimum_valid_t1_source_classes": 2,
                "required_valid_t1_source_classes": [
                    "dicom_series",
                    "nifti_single",
                ],
                "valid_unit_count": 12,
                "valid_manufacturer_counts": technical_diversity[
                    "manufacturer_counts"
                ],
                "valid_manufacturer_family_counts": technical_diversity[
                    "manufacturer_family_counts"
                ],
                "valid_t1_source_class_counts": technical_diversity[
                    "t1_source_class_counts"
                ],
                "valid_pool_technical_diversity_sha256": technical_diversity_digest,
            }
            decision_path.write_text(
                json.dumps(decision, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            binding = prepare_phase_b_binding(
                recipe_id=RECIPE_ID,
                phase_a_completion_path=completion_path,
                response_calibration_manifest_path=frozen_manifest_path,
                minimum_valid_subjects=12,
                response_calibration_decision_path=decision_path,
                execution_binding=execution_binding,
            )
            normative = yaml.safe_load(NORMATIVE_CONFIG.read_text(encoding="utf-8"))
            untouched = copy.deepcopy(normative)
            normative_bytes = NORMATIVE_CONFIG.read_bytes()
            resolved = inject_phase_b_calibration(normative, binding)
            resolved["execution_binding"] = copy.deepcopy(execution_binding)
            validated = validate_phase_b_calibration(resolved)
            self.assertEqual(normative, untouched)
            self.assertEqual(NORMATIVE_CONFIG.read_bytes(), normative_bytes)
            self.assertEqual(validated["valid_units"], valid_units)
            self.assertEqual(
                resolved["response_calibration_binding"], binding
            )

            manifest_stub = root / "manifest.csv"
            manifest_stub.write_text("unit\nS1_I1\n", encoding="utf-8")
            lineage = derive_lineage_id(
                recipe_id=RECIPE_ID,
                acquisition_manifest_sha256=sha256_file(manifest_stub),
                execution_binding=execution_binding,
                run_root=root,
            )
            phase_a_id = derive_mode_run_id(
                lineage_id=lineage, mode="response-calibration-phase-a"
            )
            continuation = {
                "pre_tractography_completion": file_record(completion_path),
                "human_qc_manifest": file_record(decision_path),
                "continuation_decision": file_record(decision_path),
            }
            phase_b_id = derive_mode_run_id(
                lineage_id=lineage,
                mode="phase-b",
                phase_b_binding=binding,
                continuation_binding=continuation,
            )
            changed = copy.deepcopy(binding)
            changed["response_calibration_decision"]["sha256"] = "f" * 64
            changed_phase_b_id = derive_mode_run_id(
                lineage_id=lineage,
                mode="phase-b",
                phase_b_binding=changed,
                continuation_binding=continuation,
            )
            self.assertNotEqual(phase_a_id, phase_b_id)
            self.assertNotEqual(phase_b_id, changed_phase_b_id)

    def test_launcher_modes_select_only_the_phase_a_target(self) -> None:
        common = {
            "snakemake": Path("/locked/snakemake"),
            "resolved_config_path": Path("/attempt/config.yaml"),
            "cores": 3,
        }
        phase_a = build_snakemake_command(
            **common, mode="response-calibration-phase-a"
        )
        phase_b = build_snakemake_command(**common, mode="phase-b")
        pre_gate = build_snakemake_command(
            **common, mode="pre-tractography-canary"
        )
        self.assertEqual(phase_a[-1], "response_calibration_phase_a")
        self.assertEqual(pre_gate[-1], "pre_tractography_canary")
        self.assertEqual(phase_b[-1], "phase_b_subject_terminal_records")
        self.assertNotIn("publish_manifest", phase_b)
        self.assertIn("--keep-going", phase_a)
        self.assertIn("--keep-going", phase_b)
        for command in (phase_a, pre_gate, phase_b):
            trigger_index = command.index("--rerun-triggers")
            self.assertEqual(command[trigger_index + 1], "mtime")
            self.assertEqual(command.count("--rerun-triggers"), 1)

    def test_execution_subset_requires_exact_approved_12_to_24_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scforge_v2_subset"
            root.mkdir()
            parent_path = root / "parent.csv"
            subset_path = root / "subset.csv"
            decision_path = root / "subset_approval.json"
            parent_rows = [
                {
                    **identity(f"S{index}_I{index}", dti=str(index), t1=str(index + 100)),
                    "stable": f"row-{index}",
                }
                for index in range(1, 14)
            ]
            subset_rows = [dict(row) for row in parent_rows[:12]]
            parent_path.write_text("parent exact bytes\n", encoding="utf-8")
            subset_path.write_text("subset exact bytes\n", encoding="utf-8")
            normative_path = root / "normative.yaml"
            normative_path.write_text("contract: {recipe_id: test}\n", encoding="utf-8")
            source_manifest_path = root / "workflow_source_manifest.tsv"
            source_manifest_path.write_text(
                "path\tsha256\nfixture\t" + "a" * 64 + "\n",
                encoding="utf-8",
            )
            environment_path = root / "environment.yaml"
            environment_path.write_text(
                yaml.safe_dump(
                    {
                        "workflow": {
                            "source_manifest": {
                                "path": str(source_manifest_path),
                                "sha256": sha256_file(source_manifest_path),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            valid_decision = {
                "schema_version": "2.0.0",
                "decision_type": "connectome_canary_execution_subset_approval",
                "status": "APPROVED",
                "approval_mode": "H04A_BOUNDED_CANARY",
                "approved_by": "unit-test-reviewer",
                "approved_utc": "2026-07-18T12:00:00Z",
                "user_response": "Approve SL-H04A for this exact fixture",
                "execution_scope": "canary",
                "recipe_id": RECIPE_ID,
                "parent_acquisition_manifest_sha256": sha256_file(parent_path),
                "execution_subset_manifest_sha256": sha256_file(subset_path),
                "approved_unit_count": 12,
                "diagnosis_labels_used": False,
                "selection_locked": True,
                "proposed_run_root": str(root.resolve()),
                "authorized_modes": [
                    "response-calibration-phase-a",
                    "pre-tractography-canary",
                ],
                "authorized_through": "pre_tractography_review_bundle_only",
                "maximum_cores": 4,
                "minimum_valid_response_calibration_units": 12,
                "minimum_valid_manufacturer_families": 2,
                "minimum_valid_t1_source_classes": 2,
                "required_valid_t1_source_classes": [
                    "dicom_series",
                    "nifti_single",
                ],
                "h04a_wall_clock_stop_hours": 72,
                "h04a_storage_stop_gb": 150,
                "tractography_authorized": False,
                "matrix_generation_authorized": False,
                "full_cohort_authorized": False,
                "normative_config_sha256": sha256_file(normative_path),
                "workflow_source_manifest_sha256": sha256_file(
                    source_manifest_path
                ),
                "environment_contract_sha256": sha256_file(environment_path),
            }
            decision_path.write_text(
                json.dumps(valid_decision) + "\n",
                encoding="utf-8",
            )
            with mock.patch(
                "workflow.run_connectome_v2.load_acquisition_manifest",
                return_value=tuple(subset_rows),
            ):
                binding, observed = prepare_execution_binding(
                    recipe_id=RECIPE_ID,
                    parent_manifest_path=parent_path,
                    parent_rows=parent_rows,
                    execution_subset_manifest_path=subset_path,
                    execution_subset_decision_path=decision_path,
                    acquisition_schema_path=root / "schema.json",
                    run_root=root,
                    normative_config_path=normative_path,
                    environment_contract_path=environment_path,
                )
            self.assertEqual(binding["approved_unit_count"], 12)
            self.assertEqual(binding["units"], sorted(row["unit"] for row in subset_rows))
            self.assertEqual([row["unit"] for row in observed], binding["units"])
            self.assertEqual(binding["h04a_authorization"]["maximum_cores"], 4)
            self.assertEqual(
                binding["h04a_authorization"]["authorized_modes"],
                [
                    "response-calibration-phase-a",
                    "pre-tractography-canary",
                ],
            )
            require_h04a_mode_authorization(
                binding,
                mode="response-calibration-phase-a",
                requested_cores=4,
            )
            require_h04a_mode_authorization(
                binding,
                mode="pre-tractography-canary",
                requested_cores=1,
            )
            with self.assertRaisesRegex(PermissionError, "exceeds"):
                require_h04a_mode_authorization(
                    binding,
                    mode="response-calibration-phase-a",
                    requested_cores=5,
                )
            with self.assertRaisesRegex(PermissionError, "cannot authorize"):
                require_h04a_mode_authorization(
                    binding, mode="phase-b", requested_cores=1
                )

            pending = copy.deepcopy(valid_decision)
            pending["status"] = "PENDING_USER_DECISION"
            decision_path.write_text(json.dumps(pending) + "\n", encoding="utf-8")
            with mock.patch(
                "workflow.run_connectome_v2.load_acquisition_manifest",
                return_value=tuple(subset_rows),
            ):
                with self.assertRaisesRegex(ValueError, "differs at status"):
                    prepare_execution_binding(
                        recipe_id=RECIPE_ID,
                        parent_manifest_path=parent_path,
                        parent_rows=parent_rows,
                        execution_subset_manifest_path=subset_path,
                        execution_subset_decision_path=decision_path,
                        acquisition_schema_path=root / "schema.json",
                        run_root=root,
                        normative_config_path=normative_path,
                        environment_contract_path=environment_path,
                    )

            with mock.patch(
                "workflow.run_connectome_v2.load_acquisition_manifest",
                return_value=tuple(subset_rows[:11]),
            ):
                with self.assertRaisesRegex(ValueError, "12--24"):
                    prepare_execution_binding(
                        recipe_id=RECIPE_ID,
                        parent_manifest_path=parent_path,
                        parent_rows=parent_rows,
                        execution_subset_manifest_path=subset_path,
                        execution_subset_decision_path=decision_path,
                        acquisition_schema_path=root / "schema.json",
                        run_root=root,
                        normative_config_path=normative_path,
                        environment_contract_path=environment_path,
                    )

    def _resource_execution_binding(self) -> dict[str, object]:
        return {
            "execution_subset_decision": {
                "path": "/signed/h04a.json",
                "sha256": "f" * 64,
                "size_bytes": 100,
            },
            "h04a_authorization": {
                "approval_mode": "H04A_BOUNDED_CANARY",
                "approved_by": "unit-test-reviewer",
                "approved_utc": "2026-07-18T12:00:00Z",
                "user_response": "Approve SL-H04A",
                "proposed_run_root": "/test/scforge_v2",
                "authorized_modes": [
                    "response-calibration-phase-a",
                    "pre-tractography-canary",
                ],
                "authorized_through": "pre_tractography_review_bundle_only",
                "maximum_cores": 4,
                "minimum_valid_response_calibration_units": 12,
                "minimum_valid_manufacturer_families": 2,
                "minimum_valid_t1_source_classes": 2,
                "required_valid_t1_source_classes": [
                    "dicom_series",
                    "nifti_single",
                ],
                "wall_clock_stop_hours": 72,
                "wall_clock_stop_seconds": 72 * 60 * 60,
                "storage_stop_gb": 150,
                "storage_stop_bytes": 150 * 1_000_000_000,
                "tractography_authorized": False,
                "matrix_generation_authorized": False,
                "full_cohort_authorized": False,
                "normative_config_sha256": "a" * 64,
                "workflow_source_manifest_sha256": "b" * 64,
                "environment_contract_sha256": "c" * 64,
            },
        }

    def test_h04a_resource_preflight_counts_pass_and_fail_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "scforge_v2_resource"
            attempts = run_root / "attempts"
            attempts.mkdir(parents=True)
            binding = self._resource_execution_binding()
            for index, (mode, status, duration) in enumerate(
                (
                    ("response-calibration-phase-a", "PASS", 100.0),
                    ("pre-tractography-canary", "FAIL", 200.0),
                ),
                start=1,
            ):
                attempt_id = f"attempt-{index}"
                common = {
                    "attempt_id": attempt_id,
                    "launcher_mode": mode,
                    "execution_binding": binding,
                }
                (attempts / f"{attempt_id}.start.json").write_text(
                    json.dumps({**common, "status": "STARTED"}) + "\n",
                    encoding="utf-8",
                )
                (attempts / f"{attempt_id}.end.json").write_text(
                    json.dumps(
                        {
                            **common,
                            "status": status,
                            "duration_seconds": duration,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            roomy_disk = mock.Mock(
                total=1_000_000_000_000,
                used=100_000_000_000,
                free=900_000_000_000,
            )
            with mock.patch(
                "workflow.run_connectome_v2.shutil.disk_usage",
                return_value=roomy_disk,
            ):
                preflight = prepare_h04a_resource_preflight(
                    run_root=run_root, execution_binding=binding
                )
            self.assertEqual(preflight["status"], "PASS")
            self.assertEqual(preflight["prior_duration_seconds"], 300.0)
            self.assertEqual(len(preflight["prior_attempts"]), 2)

    def test_h04a_resource_preflight_rejects_unclosed_and_exhausted_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "scforge_v2_resource"
            attempts = run_root / "attempts"
            attempts.mkdir(parents=True)
            binding = self._resource_execution_binding()
            common = {
                "attempt_id": "unclosed",
                "launcher_mode": "response-calibration-phase-a",
                "execution_binding": binding,
            }
            start = attempts / "unclosed.start.json"
            start.write_text(
                json.dumps({**common, "status": "STARTED"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "unclosed matching H04A"):
                prepare_h04a_resource_preflight(
                    run_root=run_root, execution_binding=binding
                )
            (attempts / "unclosed.end.json").write_text(
                json.dumps(
                    {
                        **common,
                        "status": "FAIL",
                        "duration_seconds": 72 * 60 * 60,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "wall-clock stop"):
                prepare_h04a_resource_preflight(
                    run_root=run_root, execution_binding=binding
                )

    def test_storage_measurement_deduplicates_links_and_does_not_follow_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            run_root = parent / "scforge_v2_storage"
            run_root.mkdir()
            payload = run_root / "payload.bin"
            payload.write_bytes(b"bounded payload")
            os.link(payload, run_root / "payload-hardlink.bin")
            external = parent / "external.bin"
            external.write_bytes(b"x" * 1_000_000)
            (run_root / "external-link.bin").symlink_to(external)
            storage = measure_run_root_storage(run_root)
            self.assertTrue(storage["hardlinks_deduplicated"])
            self.assertFalse(storage["symlinks_followed"])
            self.assertEqual(storage["symlink_count"], 1)
            self.assertLess(storage["logical_bytes"], external.stat().st_size)

    def test_runtime_monitor_terminates_harmless_process_at_wall_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "scforge_v2_monitor"
            run_root.mkdir()
            log_path = run_root / "monitor.log"
            preflight = {
                "wall_clock_stop_seconds": 0.05,
                "prior_duration_seconds": 0.0,
                "storage_stop_bytes": 1_000_000_000,
            }
            returncode, _, _, runtime = execute_command_with_h04a_monitor(
                command=[
                    sys.executable,
                    "-c",
                    "import time; time.sleep(30)",
                ],
                cwd=run_root,
                execution_log_path=log_path,
                run_root=run_root,
                resource_preflight=preflight,
                monitor_interval_seconds=0.01,
                termination_grace_seconds=0.05,
            )
            self.assertEqual(returncode, 124)
            self.assertIsNotNone(runtime)
            self.assertIn(
                runtime["violation"]["type"],
                {"wall_clock_stop", "postflight_wall_clock_stop"},
            )

    def test_pre_tractography_pause_and_human_continuation_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scforge_v2_pre_gate"
            root.mkdir()
            rows = [
                identity("S1_I1", dti="1", t1="11"),
                identity("S2_I2", dti="2", t1="22"),
            ]
            execution_binding = self._execution_binding(
                root, [row["unit"] for row in rows]
            )
            run_context = root / "contract" / "pre_tractography_canary_run_context.json"
            run_context.parent.mkdir(parents=True)
            run_context.write_text("{}\n", encoding="utf-8")
            attempt_start = root / "attempts" / "pre.start.json"
            attempt_start.parent.mkdir()
            attempt_start.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "status": "STARTED",
                        "started_utc": "2026-07-18T02:00:00+00:00",
                        "attempt_id": "pre-attempt-1",
                        "run_id": RUN_ID,
                        "recipe_id": RECIPE_ID,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            attempt_end = root / "attempts" / "pre.end.json"
            attempt_end.write_text("{}\n", encoding="utf-8")
            execution_log = root / "attempts" / "pre.log"
            execution_log.write_text(
                "Error in rule visual_review_bundle:\n"
                "    wildcards: unit=S2_I2\n",
                encoding="utf-8",
            )
            for unit, automated_status in (
                ("S1_I1", "PASS"),
                ("S2_I2", "FAIL"),
            ):
                review_dir = root / "subjects" / unit / "06_preflight"
                review_dir.mkdir(parents=True)
                for name in (
                    "review_b0_vs_t1.png",
                    "review_b0_vs_5tt.png",
                    "review_b0_vs_aal3.png",
                ):
                    (review_dir / name).write_bytes(
                        b"review image " + name.encode()
                    )
                (review_dir / "visual_review_index.csv").write_text(
                    "unit,status,b0_t1,b0_5tt,b0_atlas\n"
                    f"{unit},PENDING,a,b,c\n",
                    encoding="utf-8",
                )
                (review_dir / "automated_pre_tractography_qc.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "2.0.0",
                            "record_type": "automated_pre_tractography_qc",
                            "status": automated_status,
                            "unit": unit,
                            "diagnosis_labels_used": False,
                            "image_ranges": {"wm_fod": [0.0, 1.0]},
                            "wmfod_qc": {"finite": True},
                            "commands": ["mrstats wm_fod_norm.mif"],
                            "failures": (
                                []
                                if automated_status == "PASS"
                                else ["wmfod_nonfinite"]
                            ),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            pre = finalize_pre_tractography_canary(
                run_root=root,
                execution_rows=rows,
                execution_binding=execution_binding,
                run_id=RUN_ID,
                lineage_id="d" * 24,
                recipe_id=RECIPE_ID,
                run_context_path=run_context,
                attempt_start_path=attempt_start,
                attempt_end_path=attempt_end,
                execution_log_path=execution_log,
                workflow_returncode=1,
                ended_utc="2026-07-18T03:00:00+00:00",
            )
            self.assertEqual(
                pre["summary"],
                {"expected": 2, "terminal": 2, "ready_for_human_qc": 1, "fail": 1},
            )
            self.assertEqual(pre["workflow_state"], "AWAITING_HUMAN_QC")
            self.assertFalse((root / "publication" / "run_ledger.jsonl").exists())

            pre_manifest_path = root / "publication" / "pre_tractography_canary_manifest.json"
            pre_ledger_path = root / "publication" / "pre_tractography_canary_ledger.jsonl"
            completion_path = root / "publication" / "pre_tractography_canary_completion.json"
            response_manifest_stub = root / "frozen_response_manifest.json"
            response_manifest_stub.write_text("{}\n", encoding="utf-8")
            technical_diversity = {
                "minimum_valid_manufacturer_families": 2,
                "minimum_valid_t1_source_classes": 2,
                "required_valid_t1_source_classes": [
                    "dicom_series",
                    "nifti_single",
                ],
                "valid_unit_count": 12,
                "manufacturer_counts": {
                    "GE MEDICAL SYSTEMS": 6,
                    "SIEMENS": 6,
                },
                "manufacturer_family_counts": {"GE": 6, "SIEMENS": 6},
                "t1_source_class_counts": {
                    "dicom_series": 6,
                    "nifti_single": 6,
                },
            }
            technical_diversity_digest = technical_diversity_sha256(
                technical_diversity
            )
            response_binding = {
                "binding": "test-calibration",
                "response_calibration_manifest": file_record(
                    response_manifest_stub
                ),
            }
            completion = {
                "schema_version": "2.0.0",
                "record_type": "pre_tractography_canary_completion",
                "status": "AWAITING_HUMAN_QC",
                "run_outcome": "PARTIAL",
                "mode": "pre-tractography-canary",
                "run_id": RUN_ID,
                "recipe_id": RECIPE_ID,
                "execution_binding": execution_binding,
                "response_calibration_binding": response_binding,
                "pre_tractography_summary": pre["summary"],
                "pre_tractography_manifest": file_record(pre_manifest_path),
                "pre_tractography_ledger": file_record(pre_ledger_path),
                "automated_pre_tractography_qc_records": pre[
                    "automated_pre_tractography_qc_records"
                ],
                "full_run_terminal_publication": False,
                "tractography_started": False,
            }
            completion_path.write_text(
                json.dumps(completion, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            human_qc = root / "human_qc.csv"
            human_qc.write_text(
                "unit,status,reviewer,reviewed_utc,reason\n"
                "S1_I1,PASS,reviewer-1,2026-07-18T04:00:00Z,\n",
                encoding="utf-8",
            )
            decision = root / "continuation.json"
            decision.write_text(
                json.dumps(
                    {
                        "schema_version": "2.0.0",
                        "decision_type": "connectome_canary_tractography_continuation_approval",
                        "status": "APPROVED",
                        "execution_scope": "canary",
                        "recipe_id": RECIPE_ID,
                        "execution_subset_manifest_sha256": execution_binding[
                            "execution_subset_manifest"
                        ]["sha256"],
                        "response_calibration_manifest_sha256": response_binding[
                            "response_calibration_manifest"
                        ]["sha256"],
                        "pre_tractography_completion_sha256": sha256_file(completion_path),
                        "human_qc_manifest_sha256": sha256_file(human_qc),
                        "minimum_valid_manufacturer_families": 2,
                        "minimum_valid_t1_source_classes": 2,
                        "required_valid_t1_source_classes": [
                            "dicom_series",
                            "nifti_single",
                        ],
                        "valid_unit_count": 12,
                        "valid_manufacturer_counts": technical_diversity[
                            "manufacturer_counts"
                        ],
                        "valid_manufacturer_family_counts": technical_diversity[
                            "manufacturer_family_counts"
                        ],
                        "valid_t1_source_class_counts": technical_diversity[
                            "t1_source_class_counts"
                        ],
                        "valid_pool_technical_diversity_sha256": (
                            technical_diversity_digest
                        ),
                        "approved_unit_count": 1,
                        "diagnosis_labels_used": False,
                        "all_reviewed_units_pass": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch(
                "workflow.run_connectome_v2.validate_response_calibration_diversity_binding",
                return_value=technical_diversity,
            ):
                continuation = prepare_tractography_continuation_binding(
                    recipe_id=RECIPE_ID,
                    execution_binding=execution_binding,
                    response_calibration_binding=response_binding,
                    pre_tractography_completion_path=completion_path,
                    human_qc_manifest_path=human_qc,
                    continuation_decision_path=decision,
                )
            self.assertEqual(continuation["approved_units"], ["S1_I1"])
            self.assertEqual(continuation["approved_unit_count"], 1)
            self.assertEqual(
                set(continuation["automated_pre_tractography_qc_records"]),
                {"S1_I1"},
            )
            artifact = root / "phase_b_valid_artifact.bin"
            artifact.write_bytes(b"phase-b valid evidence")
            terminal_path = (
                root
                / "subjects"
                / "S1_I1"
                / "08_qc"
                / "terminal_record.json"
            )
            terminal_path.parent.mkdir(parents=True)
            terminal = pass_record(artifact, unit="S1_I1")
            terminal["contract_files"] = {
                "run_context": file_record(run_context),
                "attempt_context": file_record(attempt_start),
            }
            terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
            ledger = finalize_terminal_publication(
                run_root=root,
                manifest_rows=rows,
                config={
                    "registration": {},
                    "atlas": {},
                    "fod": {},
                    "tensor": {},
                    "tractography": {},
                    "sift2": {},
                    "connectome": {"assignment": {}},
                },
                execution_binding=execution_binding,
                continuation_binding=continuation,
                run_id=RUN_ID,
                recipe_id=RECIPE_ID,
                run_context_path=run_context,
                manifest_path=Path(
                    execution_binding["parent_acquisition_manifest"]["path"]
                ),
                attempt_start_path=attempt_start,
                execution_log_path=execution_log,
                workflow_returncode=1,
                ended_utc="2026-07-18T05:00:00+00:00",
            )
            self.assertEqual(ledger["summary"]["fail"], 1)
            failed_terminal = json.loads(
                (
                    root
                    / "subjects"
                    / "S2_I2"
                    / "08_qc"
                    / "terminal_record.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                failed_terminal["primary_failure_reason"],
                "automated_pre_tractography_qc_not_pass:wmfod_nonfinite",
            )
            self.assertIn(
                "tractography_not_authorized_by_h04b_continuation",
                failed_terminal["secondary_flags"],
            )
            (
                root
                / "subjects"
                / "S1_I1"
                / "06_preflight"
                / "automated_pre_tractography_qc.json"
            ).write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                with mock.patch(
                    "workflow.run_connectome_v2.validate_response_calibration_diversity_binding",
                    return_value=technical_diversity,
                ):
                    prepare_tractography_continuation_binding(
                        recipe_id=RECIPE_ID,
                        execution_binding=execution_binding,
                        response_calibration_binding=response_binding,
                        pre_tractography_completion_path=completion_path,
                        human_qc_manifest_path=human_qc,
                        continuation_decision_path=decision,
                    )


if __name__ == "__main__":
    unittest.main()
