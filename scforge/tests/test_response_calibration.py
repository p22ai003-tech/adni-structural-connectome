from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scforge.response_calibration import (
    assess_response_calibration_candidate,
    build_failed_response_calibration_outcome,
    build_pass_response_calibration_outcome,
    file_record,
    freeze_response_calibration,
    validate_frozen_response_calibration,
)


RECIPE = "test-response-recipe"
RUN_ID = "a" * 24
ATTEMPT_ID = "phase-a-test-attempt"
# subprocess.run is mocked in these tests, so responsemean is only ever hashed,
# never executed. A stub file stands in for it, which keeps the tests free of
# any MRtrix installation.
_STUB_DIR = Path(tempfile.mkdtemp(prefix="responsemean_stub_"))
RESPONSEMEAN = _STUB_DIR / "responsemean"
RESPONSEMEAN.write_text("#!/bin/sh\n# stand-in for MRtrix responsemean; never executed\n")
RESPONSEMEAN.chmod(0o755)


class ResponseCalibrationNonContagionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.responsemean_patch = mock.patch(
            "scforge.response_calibration.subprocess.run",
            side_effect=self._fake_responsemean,
        )
        self.mock_responsemean = self.responsemean_patch.start()
        self.addCleanup(self.responsemean_patch.stop)

    @staticmethod
    def _fake_responsemean(command, **_kwargs):
        arrays = [
            np.loadtxt(path, dtype=float, ndmin=2)
            for path in command[1:-1]
        ]
        np.savetxt(
            command[-1],
            np.mean(np.stack(arrays, axis=0), axis=0),
            fmt="%.12g",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    def _contexts(self, root: Path) -> tuple[Path, Path]:
        run = root / "run_context.json"
        run.write_text(
            json.dumps({"run_id": RUN_ID, "recipe_id": RECIPE}) + "\n",
            encoding="utf-8",
        )
        attempt = root / "attempt_context.json"
        attempt.write_text(
            json.dumps(
                {
                    "attempt_id": ATTEMPT_ID,
                    "run_id": RUN_ID,
                    "recipe_id": RECIPE,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return run, attempt

    @staticmethod
    def _source_hashes(unit: str) -> dict[str, str]:
        return {
            "dti_raw_bundle_sha256": hashlib.sha256(f"{unit}-dti".encode()).hexdigest(),
            "t1_raw_bundle_sha256": hashlib.sha256(f"{unit}-t1".encode()).hexdigest(),
            "pair_content_bundle_sha256": hashlib.sha256(f"{unit}-pair".encode()).hexdigest(),
        }

    def _candidate_files(
        self, root: Path, unit: str, value: float
    ) -> tuple[dict[str, Path], Path]:
        responses: dict[str, Path] = {}
        for tissue in ("wm", "gm", "csf"):
            path = root / f"{unit}_{tissue}.txt"
            columns = 4 if tissue == "wm" else 1
            np.savetxt(path, np.full((2, columns), value), fmt="%.12g")
            responses[tissue] = path
        voxels = root / f"{unit}_response_voxels.mif"
        voxels.write_bytes(b"test response voxel evidence\n")
        return responses, voxels

    def _pass(
        self, root: Path, unit: str, value: float, run: Path, attempt: Path
    ) -> dict:
        responses, voxels = self._candidate_files(root, unit, value)
        qc = assess_response_calibration_candidate(
            responses,
            selected_voxel_counts={"wm": 100, "gm": 80, "csf": 60},
            expected_shell_rows=2,
            expected_coefficient_columns={"wm": 4, "gm": 1, "csf": 1},
        )
        selection = root / f"{unit}_fod_shell_selection.json"
        selection.write_text(
            json.dumps(
                {
                    "schema_version": "2.0.0",
                    "status": "PASS",
                    "unit": unit,
                    "selected_shells_s_per_mm2": [0.0, 1000.0],
                    "selected_shell_sizes": [5, 30],
                    "bzero_threshold_s_per_mm2": 50.0,
                    "selection_uses_diagnosis_labels": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return build_pass_response_calibration_outcome(
            unit=unit,
            recipe_id=RECIPE,
            source_identity_hashes=self._source_hashes(unit),
            response_paths=responses,
            response_qc=qc,
            response_voxels_path=voxels,
            fod_shell_selection_path=selection,
            generated_utc="2026-07-18T00:00:00Z",
            phase_a_run_id=RUN_ID,
            phase_a_attempt_id=ATTEMPT_ID,
            run_context_path=run,
            attempt_context_path=attempt,
        )

    def _fail(self, root: Path, unit: str, run: Path, attempt: Path) -> dict:
        return build_failed_response_calibration_outcome(
            unit=unit,
            recipe_id=RECIPE,
            source_identity_hashes=self._source_hashes(unit),
            generated_utc="2026-07-18T00:00:00Z",
            phase_a_run_id=RUN_ID,
            phase_a_attempt_id=ATTEMPT_ID,
            run_context_path=run,
            attempt_context_path=attempt,
            primary_failure_reason="synthetic upstream failure",
        )

    def _lineage(self, root: Path, total: int) -> dict:
        records = {}
        for name in ("completion", "manifest", "attempt_start"):
            path = root / f"lineage_{name}.json"
            path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
            records[name] = file_record(path)
        return {
            "run_id": RUN_ID,
            "recipe_id": RECIPE,
            **records,
            "expected_outcome_count": total,
            "terminal_outcome_count": total,
        }

    def _technical_diversity_kwargs(
        self,
        root: Path,
        units: list[str],
        *,
        manufacturers: list[str] | None = None,
        t1_source_classes: list[str] | None = None,
        minimum_manufacturer_families: int = 1,
        required_t1_source_classes: list[str] | None = None,
    ) -> dict:
        manufacturers = manufacturers or ["SIEMENS"] * len(units)
        t1_source_classes = t1_source_classes or ["dicom_series"] * len(units)
        required_t1_source_classes = required_t1_source_classes or [
            "dicom_series"
        ]
        subset = root / "execution_subset.csv"
        subset.write_text(
            "unit,manufacturer,t1_source_kind\n"
            + "".join(
                f"{unit},{manufacturer},{t1_source}\n"
                for unit, manufacturer, t1_source in zip(
                    units, manufacturers, t1_source_classes
                )
            ),
            encoding="utf-8",
        )
        metadata = {
            unit: {
                "manufacturer": manufacturer,
                "manufacturer_family": (
                    "SIEMENS" if "SIEMENS" in manufacturer.upper() else "GE"
                ),
                "t1_source_class": t1_source,
            }
            for unit, manufacturer, t1_source in zip(
                units, manufacturers, t1_source_classes
            )
        }
        return {
            "unit_acquisition_metadata": metadata,
            "execution_subset_manifest": file_record(subset),
            "minimum_valid_manufacturer_families": minimum_manufacturer_families,
            "minimum_valid_t1_source_classes": len(required_t1_source_classes),
            "required_valid_t1_source_classes": required_t1_source_classes,
        }

    def test_one_failed_unit_does_not_collapse_valid_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, attempt = self._contexts(root)
            outcomes = [
                self._pass(root, "U1", 1.0, run, attempt),
                self._fail(root, "U_FAIL", run, attempt),
                self._pass(root, "U2", 3.0, run, attempt),
            ]
            frozen = freeze_response_calibration(
                outcomes,
                root / "frozen",
                minimum_valid_subjects=2,
                generated_utc="2026-07-18T00:00:00Z",
                responsemean_executable=RESPONSEMEAN,
                expected_responsemean_sha256=file_record(RESPONSEMEAN)["sha256"],
                phase_a_lineage=self._lineage(root, len(outcomes)),
                **self._technical_diversity_kwargs(
                    root, ["U1", "U2", "U_FAIL"]
                ),
            )
            observed = np.loadtxt(frozen["pooled_responses"]["wm"]["path"], ndmin=2)
            validated = validate_frozen_response_calibration(
                frozen["manifest_path"],
                expected_manifest_sha256=frozen["manifest_sha256"],
                minimum_valid_subjects=2,
                expected_responses=frozen["pooled_responses"],
                expected_technical_diversity=frozen[
                    "valid_pool_technical_diversity"
                ],
            )
        self.assertEqual(observed.shape, (2, 4))
        self.assertTrue(np.isfinite(observed).all())
        self.assertEqual(
            validated["method"],
            "mrtrix_responsemean_nonlegacy_scale_compensated",
        )
        self.assertEqual(validated["valid_subject_count"], 2)
        self.assertEqual(validated["invalid_units"], ["U_FAIL"])
        self.assertFalse(validated["phase_b_live_all_units_dependency"])
        self.assertFalse(validated["dummy_response_files_used"])

    def test_prespecified_minimum_n_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, attempt = self._contexts(root)
            outcomes = [
                self._pass(root, "U1", 1.0, run, attempt),
                self._fail(root, "U_FAIL", run, attempt),
            ]
            with self.assertRaisesRegex(ValueError, "below prespecified minimum"):
                freeze_response_calibration(
                    outcomes,
                    root / "frozen",
                    minimum_valid_subjects=2,
                    generated_utc="2026-07-18T00:00:00Z",
                    responsemean_executable=RESPONSEMEAN,
                    expected_responsemean_sha256=file_record(RESPONSEMEAN)["sha256"],
                )

    def test_diagnosis_fields_are_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, attempt = self._contexts(root)
            outcome = self._pass(root, "U1", 1.0, run, attempt)
            outcome["diagnosis"] = "CN"
            with self.assertRaisesRegex(ValueError, "Diagnosis/group fields"):
                freeze_response_calibration(
                    [outcome],
                    root / "frozen",
                    minimum_valid_subjects=1,
                    generated_utc="2026-07-18T00:00:00Z",
                    responsemean_executable=RESPONSEMEAN,
                    expected_responsemean_sha256=file_record(RESPONSEMEAN)["sha256"],
                )

    def test_response_qc_failure_is_excluded_not_imputed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, attempt = self._contexts(root)
            responses, voxels = self._candidate_files(root, "U_BAD_QC", 2.0)
            qc = assess_response_calibration_candidate(
                responses,
                selected_voxel_counts={"wm": 2, "gm": 0, "csf": 3},
                expected_shell_rows=2,
                expected_coefficient_columns={"wm": 4, "gm": 1, "csf": 1},
            )
            self.assertEqual(qc["status"], "FAIL")
            invalid = build_failed_response_calibration_outcome(
                unit="U_BAD_QC",
                recipe_id=RECIPE,
                source_identity_hashes=self._source_hashes("U_BAD_QC"),
                generated_utc="2026-07-18T00:00:00Z",
                phase_a_run_id=RUN_ID,
                phase_a_attempt_id=ATTEMPT_ID,
                run_context_path=run,
                attempt_context_path=attempt,
                primary_failure_reason="response_qc_failed",
                response_qc=qc,
                candidate_response_paths=responses,
                response_voxels_path=voxels,
            )
            outcomes = [self._pass(root, "U1", 1.0, run, attempt), invalid]
            frozen = freeze_response_calibration(
                outcomes,
                root / "frozen",
                minimum_valid_subjects=1,
                generated_utc="2026-07-18T00:00:00Z",
                responsemean_executable=RESPONSEMEAN,
                expected_responsemean_sha256=file_record(RESPONSEMEAN)["sha256"],
                phase_a_lineage=self._lineage(root, len(outcomes)),
                **self._technical_diversity_kwargs(
                    root, ["U1", "U_BAD_QC"]
                ),
            )
            self.assertEqual(frozen["invalid_units"], ["U_BAD_QC"])
            self.assertNotIn("U_BAD_QC", frozen["inputs"])

    def test_twelve_same_manufacturer_units_fail_before_pooling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, attempt = self._contexts(root)
            units = [f"U{index:02d}" for index in range(12)]
            outcomes = [
                self._pass(root, unit, float(index + 1), run, attempt)
                for index, unit in enumerate(units)
            ]
            technical = self._technical_diversity_kwargs(
                root,
                units,
                manufacturers=["SIEMENS"] * 12,
                t1_source_classes=[
                    "dicom_series" if index % 2 == 0 else "nifti_single"
                    for index in range(12)
                ],
                minimum_manufacturer_families=2,
                required_t1_source_classes=["dicom_series", "nifti_single"],
            )
            with self.assertRaisesRegex(
                ValueError, "insufficient manufacturer-family diversity"
            ):
                freeze_response_calibration(
                    outcomes,
                    root / "frozen",
                    minimum_valid_subjects=12,
                    generated_utc="2026-07-18T00:00:00Z",
                    responsemean_executable=RESPONSEMEAN,
                    expected_responsemean_sha256=file_record(RESPONSEMEAN)[
                        "sha256"
                    ],
                    phase_a_lineage=self._lineage(root, len(outcomes)),
                    **technical,
                )
            self.assertEqual(self.mock_responsemean.call_count, 0)
            self.assertFalse((root / "frozen").exists())

    def test_twelve_single_t1_source_units_fail_before_pooling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, attempt = self._contexts(root)
            units = [f"U{index:02d}" for index in range(12)]
            outcomes = [
                self._pass(root, unit, float(index + 1), run, attempt)
                for index, unit in enumerate(units)
            ]
            technical = self._technical_diversity_kwargs(
                root,
                units,
                manufacturers=[
                    "SIEMENS" if index % 2 == 0 else "GE MEDICAL SYSTEMS"
                    for index in range(12)
                ],
                t1_source_classes=["dicom_series"] * 12,
                minimum_manufacturer_families=2,
                required_t1_source_classes=["dicom_series", "nifti_single"],
            )
            with self.assertRaisesRegex(ValueError, "exact required T1 source"):
                freeze_response_calibration(
                    outcomes,
                    root / "frozen",
                    minimum_valid_subjects=12,
                    generated_utc="2026-07-18T00:00:00Z",
                    responsemean_executable=RESPONSEMEAN,
                    expected_responsemean_sha256=file_record(RESPONSEMEAN)[
                        "sha256"
                    ],
                    phase_a_lineage=self._lineage(root, len(outcomes)),
                    **technical,
                )
            self.assertEqual(self.mock_responsemean.call_count, 0)
            self.assertFalse((root / "frozen").exists())


if __name__ == "__main__":
    unittest.main()
