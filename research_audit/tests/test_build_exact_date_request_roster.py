from __future__ import annotations

import unittest

import pandas as pd

from research_audit.build_exact_date_request_roster import (
    build_request_roster,
    positive_ids,
)


def dryrun() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["001_S_0001", "001_S_0002"],
            "group": ["CN", "MCI"],
            "dti_image_id": [101, 102],
            "current_t1_image_id": [201, 204],
            "candidate_t1_image_id": [202, 204],
            "recommended_action": [
                "replace_after_exact_date_and_qc",
                "retain_current_contemporaneous_t1",
            ],
        }
    )


def all_mri() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["001_S_0001", "ADNI 3", 70.0, "MRI", "MPRAGE", "Original", 201],
            ["001_S_0001", "ADNI 3", 70.1, "MRI", "MPRAGE Repeat", "Original", 202],
            ["001_S_0001", "ADNI 3", 70.1, "MRI", "MPRAGE", "Processed", 203],
            ["001_S_0002", "ADNI 3", 71.0, "MRI", "IR-FSPGR", "Original", 204],
            ["001_S_0002", "ADNI 3", 71.0, "MRI", "FLAIR", "Original", 205],
        ],
        columns=["Subject ID", "Phase", "Age", "Modality", "Description", "Type", "Image ID"],
    )


class ExactDateRequestRosterTests(unittest.TestCase):
    def test_complete_original_t1_universe_is_retained(self) -> None:
        result, validation = build_request_roster(all_mri(), dryrun(), expected_subjects=2)
        self.assertEqual(set(result["t1_image_id"]), {201, 202, 204})
        self.assertEqual(validation["counts"]["request_rows"], 3)
        self.assertEqual(validation["counts"]["provisional_replacement_ids"], 1)
        self.assertEqual(result["is_rounded_age_screen_candidate"].sum(), 2)
        self.assertEqual(result["is_provisional_replacement_candidate"].sum(), 1)

    def test_missing_screen_candidate_fails(self) -> None:
        source = all_mri()
        source = source[source["Image ID"].ne(202)].copy()
        with self.assertRaisesRegex(ValueError, "missing_screen_ids"):
            build_request_roster(source, dryrun(), expected_subjects=2)

    def test_id_parser_rejects_scientific_notation_and_unsafe_float(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid positive integer"):
            positive_ids(pd.Series(["1e3"]), "test IDs")
        with self.assertRaisesRegex(ValueError, "invalid positive integer"):
            positive_ids(pd.Series([float(2**53 + 2)]), "test IDs")

    def test_non_mri_and_invalid_subject_rows_fail_closed(self) -> None:
        source = all_mri()
        source.loc[source["Image ID"].eq(202), "Modality"] = "PET"
        with self.assertRaisesRegex(ValueError, "missing_screen_ids"):
            build_request_roster(source, dryrun(), expected_subjects=2)

        selected = dryrun()
        selected.loc[0, "subject_id"] = "unsafe"
        with self.assertRaisesRegex(ValueError, "invalid subject IDs"):
            build_request_roster(all_mri(), selected, expected_subjects=2)


if __name__ == "__main__":
    unittest.main()
