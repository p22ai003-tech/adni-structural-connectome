from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from research_audit.build_visit_pair_manifest import (
    CatalogAttestation,
    FROZEN_REQUEST_ROSTER_ROWS,
    FROZEN_REQUEST_ROSTER_SHA256,
    FROZEN_REQUEST_SUBJECTS,
    build_manifests,
    file_sha256,
    load_attested_pairing_catalog,
    parse_exact_calendar_date,
)


def manifest(subjects: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sid": [
                f"{subject}_I{1000 + index}"
                for index, subject in enumerate(subjects)
            ],
            "group": ["CN"] * len(subjects),
            "band": ["good"] * len(subjects),
            "t1_path": [""] * len(subjects),
        }
    )


def dti_master(subjects: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Subject ID": subjects,
            "Image ID": [1000 + index for index in range(len(subjects))],
            "Study Date": ["2021-01-01"] * len(subjects),
            "Visit": ["v1"] * len(subjects),
            "Phase": ["ADNI 3"] * len(subjects),
            "Description": ["Axial DTI"] * len(subjects),
        }
    )


def catalog_attestation(subjects: list[str]) -> CatalogAttestation:
    eligible_subjects = set(subjects)
    index = 1
    while len(eligible_subjects) < FROZEN_REQUEST_SUBJECTS:
        eligible_subjects.add(f"999_S_{index:04d}")
        index += 1
    return CatalogAttestation(
        validation_path="/validated/exact_date_catalog_validation_v2.json",
        validation_sha256="b" * 64,
        catalog_path="/validated/exact_date_catalog_v2.csv",
        catalog_sha256="c" * 64,
        catalog_size_bytes=1,
        canonical_rows=FROZEN_REQUEST_ROSTER_ROWS,
        eligible_rows=len(eligible_subjects),
        eligible_subject_ids=frozenset(eligible_subjects),
        roster_sha256=FROZEN_REQUEST_ROSTER_SHA256,
        roster_rows=FROZEN_REQUEST_ROSTER_ROWS,
        roster_subjects=FROZEN_REQUEST_SUBJECTS,
    )


def canonical_catalog() -> pd.DataFrame:
    subjects = [
        f"001_S_{((index - 1) % FROZEN_REQUEST_SUBJECTS) + 1:04d}"
        for index in range(1, FROZEN_REQUEST_ROSTER_ROWS + 1)
    ]
    eligible = [
        index <= FROZEN_REQUEST_SUBJECTS
        for index in range(1, FROZEN_REQUEST_ROSTER_ROWS + 1)
    ]
    frame = pd.DataFrame(
        {
            "subject_id": subjects,
            "image_id": list(range(1, FROZEN_REQUEST_ROSTER_ROWS + 1)),
            "study_date": ["2021-01-01"] * FROZEN_REQUEST_ROSTER_ROWS,
            "visit": ["v1"] * FROZEN_REQUEST_ROSTER_ROWS,
            "phase": ["ADNI 3"] * FROZEN_REQUEST_ROSTER_ROWS,
            "description": ["MPRAGE"] * FROZEN_REQUEST_ROSTER_ROWS,
            "type": ["Original"] * FROZEN_REQUEST_ROSTER_ROWS,
            "qc_status": ["pass" if value else "fail" for value in eligible],
            "qc_raw_values_json": [
                '["1"]' if value else '["4"]' for value in eligible
            ],
            "qc_source": ["MRIQC.SeriesQC"] * FROZEN_REQUEST_ROSTER_ROWS,
            "study_date_raw_values_json": [
                '["20210101"]'
            ]
            * FROZEN_REQUEST_ROSTER_ROWS,
            "series_uid": [
                f"1.2.840.{index}"
                for index in range(1, FROZEN_REQUEST_ROSTER_ROWS + 1)
            ],
            "protocol": ["Accelerated T1"] * FROZEN_REQUEST_ROSTER_ROWS,
            "mri_protocol_phase": [""] * FROZEN_REQUEST_ROSTER_ROWS,
            "manufacturer": ["SIEMENS"] * FROZEN_REQUEST_ROSTER_ROWS,
            "model": [""] * FROZEN_REQUEST_ROSTER_ROWS,
            "field_strength": ["3.0"] * FROZEN_REQUEST_ROSTER_ROWS,
            "site": [""] * FROZEN_REQUEST_ROSTER_ROWS,
            "modality": ["MRI"] * FROZEN_REQUEST_ROSTER_ROWS,
            "catalog_source_row_count": [1] * FROZEN_REQUEST_ROSTER_ROWS,
            "source_extra_json": ["{}"] * FROZEN_REQUEST_ROSTER_ROWS,
            "source_name": [
                "ADNI MRIQC plus IDA Advanced Search"
            ]
            * FROZEN_REQUEST_ROSTER_ROWS,
            "source_generated_at": [
                "2026-07-17T12:00:00+00:00"
            ]
            * FROZEN_REQUEST_ROSTER_ROWS,
            "source_catalog_sha256": ["a" * 64]
            * FROZEN_REQUEST_ROSTER_ROWS,
            "eligible_for_pairing": eligible,
            "approved_for_download": [False] * FROZEN_REQUEST_ROSTER_ROWS,
        }
    )
    return frame


def validation_payload(catalog: Path, frame: pd.DataFrame) -> dict:
    eligible = frame["eligible_for_pairing"].astype(bool)
    present_valid = len(frame)
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "release_status": "PAIRING_READY",
        "source": {
            "name": "ADNI MRIQC plus IDA Advanced Search",
            "generated_at": "2026-07-17T12:00:00+00:00",
            "catalog_sha256": "a" * 64,
            "roster_sha256": FROZEN_REQUEST_ROSTER_SHA256,
            "expected_roster_sha256": FROZEN_REQUEST_ROSTER_SHA256,
            "qc_profile": "adni_mriqc_current",
        },
        "column_map": {
            "subject_id": "ParticipantID",
            "image_id": "LONIImage",
            "study_date": "StudyDate",
            "visit": "VISCODE2",
            "phase": "Phase",
            "description": "SeriesDescription",
            "type": "SeriesType",
            "qc": "SeriesQC",
        },
        "counts": {
            "roster_rows": FROZEN_REQUEST_ROSTER_ROWS,
            "roster_subjects": FROZEN_REQUEST_SUBJECTS,
            "catalog_rows": present_valid,
            "requested_catalog_rows": present_valid,
            "candidate_universe_drift_rows": 0,
            "out_of_roster_rows_ignored": 0,
            "present_valid": present_valid,
            "absent_explained": 0,
            "missing": 0,
            "present_invalid": 0,
            "eligible_for_pairing": int(eligible.sum()),
            "subjects_with_eligible_t1": FROZEN_REQUEST_SUBJECTS,
            "explicit_qc_fail": int((~eligible).sum()),
            "qc_pending": 0,
            "errors": 0,
            "warnings": 0,
        },
        "checks": {
            "expected_roster_count": FROZEN_REQUEST_ROSTER_ROWS,
            "expected_subject_count": FROZEN_REQUEST_SUBJECTS,
            "all_request_ids_accounted": True,
            "all_request_ids_present_valid": True,
            "all_request_ids_qc_resolved": True,
            "candidate_universe_closed": True,
            "pairing_ready": True,
            "exact_calendar_dates_only": True,
            "duplicate_identity_date_uid_qc_conflicts_absent": True,
            "download_approval_forced_false": True,
        },
        "errors": [],
        "warnings": [],
        "safety": {
            "metadata_only": True,
            "approved_for_download_count": 0,
            "images_downloaded": 0,
        },
        "outputs": {
            "canonical_catalog": {
                "path": str(catalog.resolve()),
                "size_bytes": catalog.stat().st_size,
                "sha256": file_sha256(catalog),
            }
        },
    }


class PairBuilderTests(unittest.TestCase):
    def test_pair_builder_rejects_unattested_catalog(self) -> None:
        subjects = ["000_S_0001"]
        mri = pd.DataFrame(
            [["000_S_0001", 1, "2021-01-01", "MPRAGE"]],
            columns=["Subject ID", "Image ID", "Study Date", "Description"],
        )
        with self.assertRaisesRegex(ValueError, "CatalogAttestation"):
            build_manifests(dti_master(subjects), mri, manifest(subjects))

    def test_catalog_attestation_rehashes_filters_and_recomputes_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "exact_date_catalog_v2.csv"
            validation = root / "exact_date_catalog_validation_v2.json"
            source = canonical_catalog()
            source.to_csv(catalog, index=False)
            payload = validation_payload(catalog, source)
            validation.write_text(json.dumps(payload), encoding="utf-8")
            frame, attestation = load_attested_pairing_catalog(catalog, validation)
            self.assertEqual(len(frame), FROZEN_REQUEST_SUBJECTS)
            self.assertNotIn(531, set(frame["image_id"]))
            self.assertEqual(attestation.status, "PASS_PAIRING_READY")
            self.assertEqual(
                len(attestation.eligible_subject_ids), FROZEN_REQUEST_SUBJECTS
            )

            catalog.write_text(catalog.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_attested_pairing_catalog(catalog, validation)

    def test_minimal_or_drifted_validation_attestation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "exact_date_catalog_v2.csv"
            validation = root / "exact_date_catalog_validation_v2.json"
            source = canonical_catalog()
            source.to_csv(catalog, index=False)

            minimal = {
                "status": "PASS",
                "release_status": "PAIRING_READY",
                "checks": {"pairing_ready": True},
                "outputs": {
                    "canonical_catalog": {"sha256": file_sha256(catalog)}
                },
            }
            validation.write_text(json.dumps(minimal), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_attested_pairing_catalog(catalog, validation)

            payload = validation_payload(catalog, source)
            payload["source"]["roster_sha256"] = "d" * 64
            validation.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen request roster"):
                load_attested_pairing_catalog(catalog, validation)

            payload = validation_payload(catalog, source)
            payload["counts"]["roster_subjects"] = 529
            validation.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "roster_subjects"):
                load_attested_pairing_catalog(catalog, validation)

    def test_boundaries_qc_and_tie_break(self) -> None:
        subjects = [
            "001_S_0001",
            "001_S_0002",
            "001_S_0003",
            "001_S_0004",
            "001_S_0005",
            "001_S_0006",
        ]
        rows = [
            # Same date: pass beats unknown; non-repeat beats repeat.
            ["001_S_0001", 11, "2021-01-10", "MPRAGE", "Original", "Unknown"],
            ["001_S_0001", 12, "2021-01-10", "MPRAGE Repeat", "Original", "Pass"],
            ["001_S_0001", 13, "2021-01-10", "MPRAGE", "Original", "Pass"],
            # Exact rule boundaries relative to 2021-01-01.
            ["001_S_0002", 21, "2021-04-01", "MPRAGE", "Original", "Pass"],
            ["001_S_0003", 31, "2021-04-02", "MPRAGE", "Original", "Pass"],
            ["001_S_0004", 41, "2021-06-30", "MPRAGE", "Original", "Pass"],
            ["001_S_0005", 51, "2021-07-01", "MPRAGE", "Original", "Pass"],
            # Explicit QC failure is excluded even when it is closer.
            ["001_S_0006", 61, "2021-01-02", "MPRAGE", "Original", "Fail"],
            ["001_S_0006", 62, "2021-01-20", "MPRAGE", "Original", "Pass"],
        ]
        mri = pd.DataFrame(
            rows,
            columns=[
                "Subject ID",
                "Image ID",
                "Study Date",
                "Description",
                "Type",
                "QC Status",
            ],
        )
        result = build_manifests(
            dti_master(subjects),
            mri,
            manifest(subjects),
            90,
            180,
            "good",
            catalog_attestation=catalog_attestation(subjects),
        )
        chosen = result["chosen"].set_index("subject_id")
        self.assertEqual(chosen.loc["001_S_0001", "selected_t1_image_id"], 13)
        self.assertEqual(chosen.loc["001_S_0002", "gap_days_abs"], 90)
        self.assertEqual(chosen.loc["001_S_0002", "pairing_window"], "primary")
        self.assertEqual(chosen.loc["001_S_0003", "gap_days_abs"], 91)
        self.assertEqual(
            chosen.loc["001_S_0003", "pairing_window"], "sensitivity_only"
        )
        self.assertEqual(chosen.loc["001_S_0004", "gap_days_abs"], 180)
        self.assertEqual(
            chosen.loc["001_S_0004", "pairing_window"], "sensitivity_only"
        )
        self.assertEqual(chosen.loc["001_S_0005", "gap_days_abs"], 181)
        self.assertEqual(
            chosen.loc["001_S_0005", "pairing_window"], "outside_sensitivity"
        )
        self.assertEqual(chosen.loc["001_S_0006", "selected_t1_image_id"], 62)
        self.assertFalse(result["chosen"]["approved_for_download"].any())
        self.assertEqual(
            result["validation"]["counts"]["selected_subjects_missing_eligible_t1"],
            0,
        )
        self.assertIn(
            "explicit_qc_failure",
            set(result["rejected"]["rejection_reason"]),
        )

    def test_deterministic_under_input_shuffle(self) -> None:
        subjects = ["002_S_0001"]
        mri = pd.DataFrame(
            [
                ["002_S_0001", 8, "2021-01-03", "MPRAGE", "Original", "Pass"],
                ["002_S_0001", 7, "2021-01-03", "MPRAGE", "Original", "Pass"],
            ],
            columns=[
                "Subject ID",
                "Image ID",
                "Study Date",
                "Description",
                "Type",
                "QC",
            ],
        )
        first = build_manifests(
            dti_master(subjects),
            mri,
            manifest(subjects),
            catalog_attestation=catalog_attestation(subjects),
        )
        second = build_manifests(
            dti_master(subjects),
            mri.sample(frac=1, random_state=42),
            manifest(subjects),
            catalog_attestation=catalog_attestation(subjects),
        )
        self.assertEqual(
            first["chosen"].iloc[0]["selected_t1_image_id"],
            second["chosen"].iloc[0]["selected_t1_image_id"],
        )
        self.assertEqual(first["chosen"].iloc[0]["selected_t1_image_id"], 7)

    def test_conflicting_mri_date_fails(self) -> None:
        subjects = ["003_S_0001"]
        mri = pd.DataFrame(
            [
                ["003_S_0001", 70, "2021-01-03", "MPRAGE"],
                ["003_S_0001", 70, "2021-01-04", "MPRAGE"],
            ],
            columns=["Subject ID", "Image ID", "Study Date", "Description"],
        )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            build_manifests(
                dti_master(subjects),
                mri,
                manifest(subjects),
                catalog_attestation=catalog_attestation(subjects),
            )

    def test_missing_or_conflicting_dti_date_fails(self) -> None:
        subjects = ["004_S_0001"]
        mri = pd.DataFrame(
            [["004_S_0001", 80, "2021-01-03", "MPRAGE"]],
            columns=["Subject ID", "Image ID", "Study Date", "Description"],
        )
        missing = dti_master(subjects)
        missing.loc[0, "Study Date"] = ""
        with self.assertRaisesRegex(ValueError, "missing, imprecise, or conflicting"):
            build_manifests(
                missing,
                mri,
                manifest(subjects),
                catalog_attestation=catalog_attestation(subjects),
            )

        conflicting = pd.concat(
            [
                dti_master(subjects),
                dti_master(subjects).assign(**{"Study Date": "2021-01-02"}),
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "missing, imprecise, or conflicting"):
            build_manifests(
                conflicting,
                mri,
                manifest(subjects),
                catalog_attestation=catalog_attestation(subjects),
            )

        mixed_validity = pd.concat(
            [
                dti_master(subjects),
                dti_master(subjects).assign(**{"Study Date": "2021-01"}),
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "missing, imprecise, or conflicting"):
            build_manifests(
                mixed_validity,
                mri,
                manifest(subjects),
                catalog_attestation=catalog_attestation(subjects),
            )

    def test_selected_subject_without_actual_eligible_t1_fails_closed(self) -> None:
        subjects = ["005_S_0001", "005_S_0002"]
        mri = pd.DataFrame(
            [
                ["005_S_0001", 91, "2021-01-03", "MPRAGE", "Original", "Pass"],
                ["005_S_0002", 92, "2021-01-03", "MPRAGE", "Original", "Fail"],
            ],
            columns=[
                "Subject ID",
                "Image ID",
                "Study Date",
                "Description",
                "Type",
                "QC",
            ],
        )
        with self.assertRaisesRegex(ValueError, "no pairing-eligible T1"):
            build_manifests(
                dti_master(subjects),
                mri,
                manifest(subjects),
                catalog_attestation=catalog_attestation(subjects),
            )

    def test_dti_dates_are_strict_exact_calendar_dates(self) -> None:
        self.assertEqual(
            parse_exact_calendar_date("1/2/2021"), pd.Timestamp("2021-01-02")
        )
        self.assertEqual(
            parse_exact_calendar_date("20210102"), pd.Timestamp("2021-01-02")
        )
        for invalid in (
            "2021",
            "2021-01",
            "2021-01-01T00:00:00Z",
            "20210230",
            "",
        ):
            with self.subTest(invalid=invalid):
                self.assertIsNone(parse_exact_calendar_date(invalid))

        subjects = ["006_S_0001"]
        mri = pd.DataFrame(
            [["006_S_0001", 93, "2021-01-03", "MPRAGE", "Original", "Pass"]],
            columns=[
                "Subject ID",
                "Image ID",
                "Study Date",
                "Description",
                "Type",
                "QC",
            ],
        )
        for invalid in ("2021", "2021-01", "2021-01-01T00:00:00Z"):
            bad_dti = dti_master(subjects)
            bad_dti.loc[0, "Study Date"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "imprecise"
            ):
                build_manifests(
                    bad_dti,
                    mri,
                    manifest(subjects),
                    catalog_attestation=catalog_attestation(subjects),
                )


if __name__ == "__main__":
    unittest.main()
