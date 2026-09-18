from __future__ import annotations

import unittest

import pandas as pd

from research_audit.build_diagnosis_reconciliation import build_reconciliation


def cohort(rows: list[list[object]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "subject_id",
            "actual_dti_image_id",
            "Image ID",
            "group_dashboard",
            "dti_study_date",
            "Description",
            "Type",
        ],
    )


def dti(rows: list[list[object]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "Subject ID",
            "Phase",
            "Sex",
            "Research Group",
            "Visit",
            "Study Date",
            "Age",
            "Description",
            "Type",
            "Image ID",
        ],
    )


def mri(rows: list[list[object]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        rows or [],
        columns=["Subject ID", "Research Group", "Study Date", "Image ID"],
    )


class DiagnosisReconciliationTests(unittest.TestCase):
    def test_exact_image_diagnosis_and_processed_identity_resolution(self) -> None:
        selected = cohort(
            [
                [
                    "001_S_0001",
                    101,
                    102,
                    "MCI",
                    "2020-01-02",
                    "corrected FA image <- Axial DTI",
                    "Processed",
                ],
                [
                    "001_S_0002",
                    201,
                    201,
                    "MCI",
                    "2020-02-03",
                    "Axial DTI",
                    "Original",
                ],
            ]
        )
        catalog = dti(
            [
                ["001_S_0001", "ADNI 3", "F", "EMCI", "Y1", "1/2/2020", 70.0, "Axial DTI", "Original", 101],
                ["001_S_0001", "ADNI 3", "F", "EMCI", "Y1", "1/2/2020", 70.0, "corrected FA image <- Axial DTI", "Processed", 102],
                ["001_S_0002", "ADNI 3", "M", "SMC", "Y2", "2/3/2020", 71.0, "Axial DTI", "Original", 201],
            ]
        )
        result, validation = build_reconciliation(
            selected, catalog, mri(), expected_subjects=2
        )
        first = result.set_index("subject_id").loc["001_S_0001"]
        second = result.set_index("subject_id").loc["001_S_0002"]
        self.assertEqual(first["diagnosis_raw_at_dti"], "EMCI")
        self.assertEqual(first["diagnosis_harmonized"], "MCI")
        self.assertEqual(first["diagnosis_harmonization_rule"], "explicit_EMCI_to_MCI")
        self.assertEqual(
            first["dti_identity_resolution_status"],
            "RESOLVED_PROCESSED_DERIVATIVE_TO_ORIGINAL_DTI",
        )
        self.assertEqual(second["diagnosis_harmonized"], "SMC")
        self.assertFalse(bool(second["primary_analysis_eligible"]))
        self.assertTrue(bool(second["smc_retained_separately"]))
        self.assertEqual(validation["counts"]["dti_identity_mismatches_resolved"], 1)
        self.assertTrue(validation["checks"]["smc_not_mapped_to_mci"])

    def test_nearest_local_diagnosis_retains_day_offset(self) -> None:
        selected = cohort(
            [["001_S_0001", 101, 101, "CN", "2020-01-02", "Axial DTI", "Original"]]
        )
        catalog = dti(
            [["001_S_0001", "ADNI 3", "F", None, "Y1", "1/2/2020", 70.0, "Axial DTI", "Original", 101]]
        )
        result, _ = build_reconciliation(
            selected,
            catalog,
            mri([["001_S_0001", "CN", "1/5/2020", 901]]),
            expected_subjects=1,
        )
        row = result.iloc[0]
        self.assertEqual(row["diagnosis_resolution_status"], "RESOLVED_NEAREST_LOCAL_DIAGNOSIS")
        self.assertEqual(row["diagnosis_day_offset"], 3)
        self.assertFalse(bool(row["diagnosis_is_exact_dti_date"]))
        self.assertEqual(row["diagnosis_harmonized"], "CN")

    def test_same_date_conflict_is_explicit_unknown(self) -> None:
        selected = cohort(
            [["001_S_0001", 101, 101, "CN", "2020-01-02", "Axial DTI", "Original"]]
        )
        catalog = dti(
            [["001_S_0001", "ADNI 3", "F", None, "Y1", "1/2/2020", 70.0, "Axial DTI", "Original", 101]]
        )
        crosscheck = mri(
            [
                ["001_S_0001", "CN", "1/2/2020", 901],
                ["001_S_0001", "MCI", "1/2/2020", 902],
            ]
        )
        result, validation = build_reconciliation(
            selected, catalog, crosscheck, expected_subjects=1
        )
        row = result.iloc[0]
        self.assertEqual(row["diagnosis_harmonized"], "UNKNOWN")
        self.assertEqual(
            row["diagnosis_resolution_status"],
            "UNKNOWN_SAME_DATE_DIAGNOSIS_CONFLICT",
        )
        self.assertTrue(validation["checks"]["all_unknown_states_explicit"])

    def test_unresolved_identity_mismatch_is_not_silently_accepted(self) -> None:
        selected = cohort(
            [["001_S_0001", 101, 102, "CN", "2020-01-02", "Axial DTI", "Original"]]
        )
        catalog = dti(
            [
                ["001_S_0001", "ADNI 3", "F", "CN", "Y1", "1/2/2020", 70.0, "Axial DTI", "Original", 101],
                ["001_S_0001", "ADNI 3", "F", "CN", "Y2", "1/3/2020", 70.0, "corrected FA image <- Axial DTI", "Processed", 102],
            ]
        )
        result, validation = build_reconciliation(
            selected, catalog, mri(), expected_subjects=1
        )
        self.assertEqual(
            result.iloc[0]["dti_identity_resolution_status"],
            "UNRESOLVED_DTI_IDENTITY_MISMATCH",
        )
        self.assertEqual(
            validation["status"], "PASS_WITH_EXPLICIT_UNRESOLVED_IDENTITY"
        )
        self.assertEqual(validation["counts"]["dti_identity_mismatches_unresolved"], 1)

    def test_duplicate_subject_key_fails(self) -> None:
        selected = cohort(
            [
                ["001_S_0001", 101, 101, "CN", "2020-01-02", "Axial DTI", "Original"],
                ["001_S_0001", 102, 102, "CN", "2020-01-03", "Axial DTI", "Original"],
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate subject_id"):
            build_reconciliation(selected, dti([]), mri(), expected_subjects=2)


if __name__ == "__main__":
    unittest.main()
