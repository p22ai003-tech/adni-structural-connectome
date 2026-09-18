from __future__ import annotations

import csv
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from scforge.input_contract import (
    assess_atlas_label_support,
    assess_pre_tractography_image_ranges,
    bundle_sha256,
    load_acquisition_manifest,
    require_eddy_metadata,
    select_model_shells,
    sha256_file,
    source_phase_encoding_token,
    stable_unit,
    staged_hashes,
    tensor_design_diagnostics,
    validate_resolved_runtime_config,
    verify_runtime_source_inventory,
    write_normalization_failure_marker,
)


SCHEMA = Path(__file__).resolve().parents[1] / "workflow" / "schemas" / "acquisition_manifest_v2.schema.json"


def _base_row(root: Path) -> dict[str, str]:
    dwi_dir = root / "dwi_dicom"
    t1 = root / "t1.nii"
    dwi_dir.mkdir()
    (dwi_dir / "1.dcm").write_bytes(b"dicom fixture")
    t1.write_bytes(b"nifti fixture")
    digest = "a" * 64
    return {
        "manifest_schema_version": "2.0.0",
        "data_scope_decision": "SL-D01_local_only",
        "processing_authorized": "true",
        "analysis_role": "primary_cn_mci_ad",
        "subject_id": "DRYRUN_001",
        "diagnosis_at_dti": "CN",
        "dti_image_id": "1",
        "dti_study_date": "2020-01-01",
        "dti_source_kind": "dicom_series",
        "dti_source_id": "ADNI-I1",
        "dti_source_path": str(dwi_dir),
        "dti_dicom_series_uid": "1.2.3",
        "dti_raw_bundle_sha256": digest,
        "dti_raw_file_count": "1",
        "dti_raw_total_bytes": "13",
        "dwi_nifti_path": "",
        "dwi_bvec_path": "",
        "dwi_bval_path": "",
        "dwi_json_path": "",
        "dwi_nifti_sha256": "",
        "dwi_bvec_sha256": "",
        "dwi_bval_sha256": "",
        "dwi_json_sha256": "",
        "t1_image_id": "2",
        "t1_study_date": "2010-01-01",
        "t1_source_kind": "nifti_single",
        "t1_source_id": "ADNI-I2",
        "t1_source_path": str(t1),
        "t1_dicom_series_uid": "",
        "t1_raw_bundle_sha256": "b" * 64,
        "t1_raw_file_count": "1",
        "t1_raw_total_bytes": "13",
        "abs_pair_gap_days": "3652",
        "timing_stratum": "gt_180_days",
        "phase": "ADNI 3",
        "site": "DRYRUN",
        "manufacturer": "SIEMENS",
        "scanner_model": "Prisma_fit",
        "field_strength_t": "3.0",
        "protocol": "single-shell-b1000",
        "phase_encoding_direction": "j-",
        "phase_encoding_source": "mrtrix_dicom_header",
        "total_readout_time": "0.0333",
        "total_readout_time_source": "mrtrix_dicom_header",
        "source_lock_status": "COMPLETE_SHA256",
        "pair_content_bundle_sha256": "c" * 64,
        "normalization_readiness": "READY",
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class AcquisitionManifestV2Tests(unittest.TestCase):
    def test_long_gap_smc_or_diagnosis_values_do_not_reject_processing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _base_row(root)
            row["diagnosis_at_dti"] = "SMC"
            path = root / "manifest.csv"
            _write_manifest(path, [row])
            observed = load_acquisition_manifest(path, SCHEMA, run_root=root / "scforge_v2_run", expected_rows=1)
        self.assertEqual(observed[0]["abs_pair_gap_days"], "3652")
        self.assertEqual(observed[0]["diagnosis_at_dti"], "SMC")

    def test_nullable_t1_uid_is_required_for_single_nifti(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _base_row(root)
            path = root / "manifest.csv"
            _write_manifest(path, [row])
            observed = load_acquisition_manifest(path, SCHEMA, run_root=root / "scforge_v2_run", expected_rows=1)
        self.assertEqual(observed[0]["t1_dicom_series_uid"], "")

    def test_dicom_t1_requires_uid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _base_row(root)
            t1_dir = root / "t1_dicom"
            t1_dir.mkdir()
            (t1_dir / "1.dcm").write_bytes(b"dicom")
            row["t1_source_kind"] = "dicom_series"
            row["t1_source_path"] = str(t1_dir)
            path = root / "manifest.csv"
            _write_manifest(path, [row])
            with self.assertRaisesRegex(ValueError, "t1_dicom_series_uid"):
                load_acquisition_manifest(path, SCHEMA, run_root=root / "scforge_v2_run")

    def test_missing_eddy_metadata_stays_schedulable_but_fails_runtime_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _base_row(root)
            row["phase_encoding_direction"] = ""
            row["total_readout_time"] = ""
            row["normalization_readiness"] = "FAIL_MISSING_PHASE_ENCODING_AND_TOTAL_READOUT_TIME"
            path = root / "manifest.csv"
            _write_manifest(path, [row])
            observed = load_acquisition_manifest(path, SCHEMA, run_root=root / "scforge_v2_run")
            with self.assertRaisesRegex(ValueError, "no default is allowed"):
                require_eddy_metadata(observed[0])

    def test_phase_encoding_vector_or_anatomical_token_is_never_guessed(self) -> None:
        self.assertEqual(source_phase_encoding_token("j-"), "j-")
        for value in ([0, -1, 0], "AP", "ROW", {"axis": "j", "sign": -1}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "no polarity mapping|image-axis string"):
                source_phase_encoding_token(value)

    def test_false_processing_authorization_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _base_row(root)
            row["processing_authorized"] = "false"
            path = root / "manifest.csv"
            _write_manifest(path, [row])
            with self.assertRaisesRegex(ValueError, "processing_authorized"):
                load_acquisition_manifest(path, SCHEMA, run_root=root / "scforge_v2_run")

    def test_nifti_bundle_requires_four_distinct_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _base_row(root)
            row["dti_source_kind"] = "nifti_bundle"
            row["dti_dicom_series_uid"] = ""
            for name, content in (
                ("dwi.nii", b"nifti"),
                ("dwi.bvec", b"0 1"),
                ("dwi.bval", b"0 1000"),
                ("dwi.json", b"{}"),
            ):
                path = root / name
                path.write_bytes(content)
            for key, name in (
                ("dwi_nifti_path", "dwi.nii"),
                ("dwi_bvec_path", "dwi.bvec"),
                ("dwi_bval_path", "dwi.bval"),
                ("dwi_json_path", "dwi.json"),
            ):
                row[key] = str(root / name)
                row[key.replace("_path", "_sha256")] = sha256_file(root / name)
            row["dti_source_path"] = row["dwi_nifti_path"]
            path = root / "manifest.csv"
            _write_manifest(path, [row])
            observed = load_acquisition_manifest(path, SCHEMA, run_root=root / "scforge_v2_run")
        self.assertEqual(observed[0]["dti_source_kind"], "nifti_bundle")


class SourceHashSeparationTests(unittest.TestCase):
    def test_canonical_aal3_source_is_compatible_with_hard_label_survival_rule(self) -> None:
        atlas_path = Path(__file__).resolve().parents[2] / "atlas" / "AAL" / "AAL3v1_1mm_166.nii.gz"
        data = np.rint(np.asanyarray(nib.load(atlas_path).dataobj)).astype(int)
        counts = {
            label: int(np.count_nonzero(data == label))
            for label in range(1, 167)
        }
        result = assess_atlas_label_support(counts, expected_nodes=166)
        self.assertEqual(result["status"], "PASS", result["failures"])
        self.assertEqual(result["minimum_observed_voxels"], 8)
        with self.assertRaisesRegex(ValueError, "diagnosis-blind H04"):
            assess_atlas_label_support(
                counts,
                expected_nodes=166,
                hard_minimum_voxels=50,
            )

    def test_automated_pre_tractography_qc_rejects_zero_or_nan_wmfod(self) -> None:
        valid = {
            "wmfod_all_sh_coefficients": {"min": -0.1, "max": 1.0},
            "wmfod_l0": {"min": 0.0, "max": 1.0},
            "gm": {"min": 0.0, "max": 1.0},
            "csf": {"min": 0.0, "max": 1.0},
            "fa": {"min": 0.0, "max": 1.0},
            "md": {"min": 0.0, "max": 0.002},
            "rd": {"min": 0.0, "max": 0.002},
            "ad": {"min": 0.0, "max": 0.003},
        }
        self.assertEqual(assess_pre_tractography_image_ranges(valid), [])
        epsilon = copy.deepcopy(valid)
        epsilon["wmfod_l0"] = {"min": -5e-13, "max": 1.0}
        epsilon["fa"] = {"min": -5e-9, "max": 1.0 + 5e-9}
        epsilon["md"] = {"min": -5e-9, "max": 0.01 + 5e-9}
        self.assertEqual(assess_pre_tractography_image_ranges(epsilon), [])
        all_zero = copy.deepcopy(valid)
        all_zero["wmfod_l0"] = {"min": 0.0, "max": 0.0}
        self.assertIn(
            "wmfod_l0:negative_or_all_zero",
            assess_pre_tractography_image_ranges(all_zero),
        )
        nonfinite = copy.deepcopy(valid)
        nonfinite["wmfod_all_sh_coefficients"] = {"min": float("nan"), "max": 1.0}
        self.assertIn(
            "nonfinite_range:wmfod_all_sh_coefficients",
            assess_pre_tractography_image_ranges(nonfinite),
        )

    def test_multishell_fod_and_tensor_selection_are_distinct(self) -> None:
        selected = select_model_shells(
            [0, 0, 700, 1000, 1000, 2000],
            b0_threshold=50,
            fod_target=1000,
            fod_tolerance=100,
            tensor_maximum=1500,
        )
        self.assertEqual(selected["observed_nonzero_shells"], [700, 1000, 2000])
        self.assertEqual(selected["fod_shells"], [0, 1000])
        self.assertEqual(selected["tensor_shells"], [0, 700, 1000])
        self.assertFalse(selected["selection_uses_diagnosis_labels"])

    def test_mrtrix_shell_command_contract_is_threshold_consistent(self) -> None:
        project = Path(__file__).resolve().parents[2]
        config = yaml.safe_load((project / "configs" / "connectome_v2.yaml").read_text())
        model_rules = (project / "scforge" / "workflow" / "rules" / "05_5tt_fod.smk").read_text()
        dwi_rules = (project / "scforge" / "workflow" / "rules" / "01_dwi_preproc.smk").read_text()
        self.assertEqual(
            config["fod"]["response_estimation"]["shell_ordered_lmax"],
            "0,6",
        )
        self.assertNotIn("-lmax 6,0,0", model_rules)
        self.assertIn(
            "-lmax {config[fod][response_estimation][shell_ordered_lmax]:q}",
            model_rules,
        )
        for command in (
            "dwi2response dhollander",
            "dwifslpreproc",
            "dwi2mask",
            "dwi2tensor",
            "dwiextract",
        ):
            with self.subTest(command=command):
                source = model_rules if command in {"dwi2response dhollander", "dwi2tensor", "dwiextract"} else dwi_rules
                command_lines = [
                    line
                    for line in source.splitlines()
                    if command in line
                ]
                self.assertTrue(command_lines, command)
        self.assertGreaterEqual(
            (model_rules + dwi_rules).count("-config BZeroThreshold {BZERO_THRESHOLD}"),
            8,
        )

    def test_tensor_design_rejects_six_unique_but_coplanar_directions(self) -> None:
        coplanar = [
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (1, -1, 0),
            (2, 1, 0),
            (1, 2, 0),
        ]
        self.assertEqual(len(set(coplanar)), 6)
        diagnostics = tensor_design_diagnostics(coplanar)
        self.assertLess(diagnostics["rank"], diagnostics["required_rank"])

    def test_runtime_source_mutation_is_rehashed_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _base_row(root)
            row["unit"] = stable_unit(row)
            member = Path(row["dti_source_path"]) / "1.dcm"
            member_hash = sha256_file(member)
            stat = member.stat()
            locked_bundle = bundle_sha256(
                [(member.name, stat.st_size, member_hash)]
            )
            row["dti_raw_bundle_sha256"] = locked_bundle
            inventory = root / "dwi_inventory.json"
            inventory.write_text(
                json.dumps(
                    {
                        "unit": row["unit"],
                        "modality": "dwi",
                        "locked_bundle_sha256": locked_bundle,
                        "members": [
                            {
                                "relative_path": member.name,
                                "absolute_path": str(member.resolve()),
                                "size_bytes": stat.st_size,
                                "sha256": member_hash,
                                "st_dev": stat.st_dev,
                                "st_ino": stat.st_ino,
                                "st_mtime_ns": stat.st_mtime_ns,
                                "st_ctime_ns": stat.st_ctime_ns,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            unchanged = verify_runtime_source_inventory(row, "dwi", inventory)
            self.assertEqual(unchanged["content_rehash_count"], 0)
            os.utime(member, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            metadata_changed = verify_runtime_source_inventory(row, "dwi", inventory)
            self.assertEqual(metadata_changed["content_rehash_count"], 1)
            member.write_bytes(b"mutated content")
            with self.assertRaisesRegex(ValueError, "content mutation"):
                verify_runtime_source_inventory(row, "dwi", inventory)

    def test_resolved_config_runtime_projection_is_exact_in_all_three_modes(self) -> None:
        normative = {
            "contract": {"recipe_id": "recipe"},
            "inputs": {"approved_pair_manifest": {"path": "LOCKED", "sha256": "a" * 64}},
            "fod": {
                "response_estimation": {
                    "calibration": {
                        "minimum_valid_subjects": "REQUIRED_H04",
                        "frozen_manifest": {"path": "REQUIRED", "sha256": "REQUIRED"},
                        "pooled_responses": {"wm": {}, "gm": {}, "csf": {}},
                        "response_qc": {"stronger_absolute_voxel_minimum": None},
                    }
                }
            },
            "scientific_field": {"must_not_change": 7},
        }
        phase_a = copy.deepcopy(normative)
        phase_a.update(
            {
                "manifest_path": "/attempt/manifest.csv",
                "run_root": "/attempt/run",
                "human_qc_manifest": "/attempt/qc.csv",
                "run_context_path": "/attempt/run.json",
                "attempt_context_path": "/attempt/start.json",
                "resolved_run_config_path": "/attempt/config.yaml",
                "launcher_mode": "response-calibration-phase-a",
                "execution_manifest_path": "/attempt/subset.csv",
            }
        )
        phase_a["inputs"]["approved_pair_manifest"] = {
            "path": "/attempt/manifest.csv",
            "sha256": "b" * 64,
        }
        units = [f"UNIT_{index:02d}" for index in range(12)]
        execution_binding = {
            "schema_version": "2.0.0",
            "binding_type": "connectome_execution_subset_binding",
            "execution_scope": "canary",
            "recipe_id": "recipe",
            "parent_acquisition_manifest": {
                "path": "/attempt/manifest.csv",
                "sha256": "b" * 64,
                "size_bytes": 100,
            },
            "execution_subset_manifest": {
                "path": "/attempt/subset.csv",
                "sha256": "e" * 64,
                "size_bytes": 50,
            },
            "execution_subset_decision": {
                "path": "/attempt/subset-decision.json",
                "sha256": "f" * 64,
                "size_bytes": 50,
            },
            "approved_unit_count": len(units),
            "units": units,
            "diagnosis_labels_used": False,
            "selection_locked": True,
            "h04a_authorization": {
                "approval_mode": "H04A_BOUNDED_CANARY",
                "approved_by": "unit-test-reviewer",
                "approved_utc": "2026-07-18T12:00:00Z",
                "user_response": "Approve SL-H04A",
                "proposed_run_root": "/attempt/run",
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
                "normative_config_sha256": "1" * 64,
                "workflow_source_manifest_sha256": "2" * 64,
                "environment_contract_sha256": "3" * 64,
            },
        }
        phase_a["execution_binding"] = copy.deepcopy(execution_binding)
        validate_resolved_runtime_config(
            normative,
            phase_a,
            {
                "launcher_mode": "response-calibration-phase-a",
                "execution_binding": execution_binding,
            },
        )

        valid_unit_metadata = {
            unit: {
                "manufacturer": (
                    "SIEMENS" if index % 2 == 0 else "GE MEDICAL SYSTEMS"
                ),
                "manufacturer_family": "SIEMENS" if index % 2 == 0 else "GE",
                "t1_source_class": (
                    "dicom_series" if index % 2 == 0 else "nifti_single"
                ),
            }
            for index, unit in enumerate(units)
        }
        diversity = {
            "schema_version": "1.0.0",
            "status": "PASS",
            "rule": "PASS_response_units_joined_to_bound_execution_subset_metadata",
            "diagnosis_labels_used": False,
            "execution_subset_manifest": copy.deepcopy(
                execution_binding["execution_subset_manifest"]
            ),
            "minimum_valid_manufacturer_families": 2,
            "minimum_valid_t1_source_classes": 2,
            "required_valid_t1_source_classes": [
                "dicom_series",
                "nifti_single",
            ],
            "valid_unit_count": 12,
            "manufacturer_count": 2,
            "manufacturer_counts": {
                "GE MEDICAL SYSTEMS": 6,
                "SIEMENS": 6,
            },
            "manufacturer_family_count": 2,
            "manufacturer_family_counts": {"GE": 6, "SIEMENS": 6},
            "t1_source_class_count": 2,
            "t1_source_class_counts": {
                "dicom_series": 6,
                "nifti_single": 6,
            },
            "valid_unit_metadata": valid_unit_metadata,
        }
        diversity_digest = hashlib.sha256(
            json.dumps(diversity, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        binding = {
            "binding_type": "response_calibration_phase_b_binding",
            "recipe_id": "recipe",
            "minimum_valid_subjects": 12,
            "valid_units": units,
            "valid_pool_technical_diversity": diversity,
            "valid_pool_technical_diversity_sha256": diversity_digest,
            "response_calibration_manifest": {
                "path": "/frozen/manifest.json",
                "sha256": "c" * 64,
                "size_bytes": 100,
            },
            "pooled_responses": {
                tissue: {
                    "path": f"/frozen/{tissue}.txt",
                    "sha256": "d" * 64,
                    "size_bytes": 10,
                }
                for tissue in ("wm", "gm", "csf")
            },
            "diagnosis_labels_used": False,
            "phase_b_live_all_units_dependency": False,
            "dummy_response_files_used": False,
        }
        pre_tractography = copy.deepcopy(phase_a)
        pre_tractography["launcher_mode"] = "pre-tractography-canary"
        pre_tractography["response_calibration_binding"] = copy.deepcopy(binding)
        calibration = pre_tractography["fod"]["response_estimation"]["calibration"]
        calibration["minimum_valid_subjects"] = 12
        calibration["frozen_manifest"] = copy.deepcopy(
            binding["response_calibration_manifest"]
        )
        calibration["pooled_responses"] = copy.deepcopy(binding["pooled_responses"])
        validate_resolved_runtime_config(
            normative,
            pre_tractography,
            {
                "launcher_mode": "pre-tractography-canary",
                "execution_binding": execution_binding,
                "response_calibration_binding": binding,
            },
        )

        phase_b_units = units[:9]
        continuation = {
            "schema_version": "2.0.0",
            "binding_type": "tractography_continuation_binding",
            "execution_scope": "canary",
            "recipe_id": "recipe",
            "execution_subset_manifest": copy.deepcopy(
                execution_binding["execution_subset_manifest"]
            ),
            "pre_tractography_completion": {
                "path": "/attempt/pre-completion.json",
                "sha256": "1" * 64,
                "size_bytes": 10,
            },
            "pre_tractography_manifest": {
                "path": "/attempt/pre-manifest.json",
                "sha256": "2" * 64,
                "size_bytes": 10,
            },
            "human_qc_manifest": {
                "path": "/attempt/human.csv",
                "sha256": "3" * 64,
                "size_bytes": 10,
            },
            "continuation_decision": {
                "path": "/attempt/continue.json",
                "sha256": "4" * 64,
                "size_bytes": 10,
            },
            "valid_pool_technical_diversity": copy.deepcopy(diversity),
            "valid_pool_technical_diversity_sha256": diversity_digest,
            "approved_unit_count": len(phase_b_units),
            "approved_units": phase_b_units,
            "diagnosis_labels_used": False,
        }
        phase_b = copy.deepcopy(pre_tractography)
        phase_b["launcher_mode"] = "phase-b"
        phase_b["tractography_continuation_binding"] = copy.deepcopy(continuation)
        phase_b_context = {
            "launcher_mode": "phase-b",
            "execution_binding": execution_binding,
            "response_calibration_binding": binding,
            "tractography_continuation_binding": continuation,
        }
        validate_resolved_runtime_config(normative, phase_b, phase_b_context)
        tampered = copy.deepcopy(phase_b)
        tampered["scientific_field"]["must_not_change"] = 8
        with self.assertRaisesRegex(ValueError, "scientific fields"):
            validate_resolved_runtime_config(
                normative,
                tampered,
                phase_b_context,
            )

    def test_staged_hashes_hash_exact_normalized_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dwi = root / "dwi_raw.mif"
            t1 = root / "t1_native.nii"
            dwi.write_bytes(b"staged dwi")
            t1.write_bytes(b"staged t1")
            expected_dwi_hash = sha256_file(dwi)
            observed = staged_hashes({"dwi": dwi, "t1": t1})
        self.assertEqual(observed["dwi"]["sha256"], expected_dwi_hash)
        self.assertNotEqual(observed["dwi"]["sha256"], "a" * 64)

    def test_bundle_hash_is_framed_and_order_independent(self) -> None:
        rows = [("b", 2, "b" * 64), ("a", 1, "a" * 64)]
        self.assertEqual(bundle_sha256(rows), bundle_sha256(reversed(rows)))
        self.assertNotEqual(bundle_sha256(rows), bundle_sha256([("a", 1, "a" * 64)]))

    def test_unit_does_not_depend_on_diagnosis_or_t1_source(self) -> None:
        row = {"subject_id": "002_S_0413", "dti_image_id": "863064"}
        self.assertEqual(stable_unit(row), "002_S_0413_I863064")

    def test_normalization_failure_marker_is_atomic_and_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt_context = root / "attempt-a.json"
            attempt_context.write_text('{"attempt_id":"attempt-a","run_id":"run-a"}\n')
            path = root / "failures" / "attempt-a" / "dwi_normalization_failure.json"
            common = {
                "unit": "DRYRUN_001_I1",
                "normalizer": "dwi",
                "source_identity_hashes": {"dti_raw_bundle_sha256": "a" * 64},
                "run_id": "run-a",
                "attempt_id": "attempt-a",
                "attempt_context_path": attempt_context,
            }
            first = write_normalization_failure_marker(
                path,
                **common,
                primary_failure_reason="ValueError: missing TotalReadoutTime",
                generated_utc="2026-07-18T00:00:00Z",
            )
            with self.assertRaisesRegex(FileExistsError, "Conflicting immutable"):
                write_normalization_failure_marker(
                    path,
                    **common,
                    primary_failure_reason="RuntimeError: later retry",
                    generated_utc="2026-07-18T01:00:00Z",
                )
            partials = list(path.parent.glob("*.partial"))
        self.assertEqual(first["status"], "FAIL")
        self.assertEqual(first["stage"], "source_normalization")
        self.assertEqual(first["attempt_id"], "attempt-a")
        self.assertEqual(partials, [])


if __name__ == "__main__":
    unittest.main()
