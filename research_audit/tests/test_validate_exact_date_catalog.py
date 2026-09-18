from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from research_audit.validate_exact_date_catalog import (
    normalize_positive_integer,
    parse_exact_calendar_date,
    read_csv_snapshot,
    validate_exact_date_catalog,
    write_outputs,
)


CATALOG_SHA = "a" * 64
ROSTER_SHA = "b" * 64


def roster(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [f"001_S_{index:04d}" for index in range(1, n + 1)],
            "group": ["CN"] * n,
            "dti_image_id": [1000 + index for index in range(1, n + 1)],
            "candidate_t1_image_id": [2000 + index for index in range(1, n + 1)],
        }
    )


def catalog_rows(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ParticipantID": [f"001_S_{index:04d}" for index in range(1, n + 1)],
            "LONIImage": [2000 + index for index in range(1, n + 1)],
            "StudyDate": [f"20210{index}02" for index in range(1, n + 1)],
            "VISCODE2": [f"m{index * 12:02d}" for index in range(1, n + 1)],
            "Phase": ["ADNI 3"] * n,
            "SeriesDescription": ["Accelerated Sagittal MPRAGE"] * n,
            "SeriesType": ["Original"] * n,
            "SeriesQC": [1] * n,
            "SeriesInstanceUID": [f"1.2.840.{index}" for index in range(1, n + 1)],
            "SeriesProtocol": ["Accelerated T1"] * n,
            "ScannerManufacturer": ["SIEMENS"] * n,
            "ReleaseForAnalysis": [1] * n,
        }
    )


def validate(
    catalog: pd.DataFrame,
    request: pd.DataFrame | None = None,
    **kwargs,
):
    request = roster() if request is None else request
    return validate_exact_date_catalog(
        catalog,
        request,
        source_name=kwargs.pop("source_name", "ADNI MRIQC plus IDA Advanced Search"),
        source_generated_at=kwargs.pop("source_generated_at", "2026-07-17T12:00:00Z"),
        catalog_sha256=kwargs.pop("catalog_sha256", CATALOG_SHA),
        roster_sha256=kwargs.pop("roster_sha256", ROSTER_SHA),
        default_qc_source=kwargs.pop("default_qc_source", "MRIQC.SeriesQC"),
        qc_profile=kwargs.pop("qc_profile", "adni_mriqc_current"),
        expected_roster_count=kwargs.pop("expected_roster_count", len(request)),
        expected_subject_count=kwargs.pop("expected_subject_count", request["subject_id"].nunique()),
        expected_roster_sha256=kwargs.pop("expected_roster_sha256", ROSTER_SHA),
        **kwargs,
    )


class ExactDateCatalogValidatorTests(unittest.TestCase):
    def test_official_mriqc_aliases_pass_and_preserve_extra_fields(self) -> None:
        source = catalog_rows()
        source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
        result = validate(source)
        self.assertEqual(result["validation"]["status"], "PASS")
        self.assertEqual(len(result["canonical"]), 2)
        first = result["canonical"].set_index("image_id").loc[2001]
        self.assertEqual(first["study_date"], "2021-01-02")
        self.assertEqual(first["catalog_source_row_count"], 2)
        self.assertIn("ReleaseForAnalysis", first["source_extra_json"])
        self.assertEqual(first["qc_raw_values_json"], '["1"]')
        self.assertEqual(first["study_date_raw_values_json"], '["20210102"]')
        self.assertEqual(first["protocol"], "Accelerated T1")
        self.assertTrue(first["eligible_for_pairing"])
        self.assertFalse(result["coverage"]["approved_for_download"].any())

    def test_documented_absence_accounts_for_missing_source_id(self) -> None:
        source = catalog_rows().iloc[[0]].copy()
        absence = pd.DataFrame({"Image ID": [2002], "absence_reason": ["Not returned by source export"]})
        result = validate(source, absence_ledger=absence, absence_sha256="c" * 64)
        self.assertEqual(result["validation"]["status"], "PASS")
        self.assertTrue(result["validation"]["checks"]["all_request_ids_accounted"])
        self.assertFalse(result["validation"]["checks"]["all_request_ids_present_valid"])
        statuses = set(result["coverage"]["coverage_status"])
        self.assertEqual(statuses, {"present_valid", "absent_explained"})
        self.assertFalse(result["validation"]["checks"]["pairing_ready"])
        self.assertEqual(result["validation"]["release_status"], "INTAKE_PASS_PAIRING_BLOCKED")

    def test_unexplained_missing_id_fails(self) -> None:
        result = validate(catalog_rows().iloc[[0]].copy())
        self.assertEqual(result["validation"]["status"], "FAIL")
        self.assertEqual(result["validation"]["counts"]["missing"], 1)
        self.assertIn("requested_id_missing", {item["code"] for item in result["validation"]["errors"]})

    def test_duplicate_conflicts_and_subject_mismatch_fail(self) -> None:
        source = catalog_rows()
        conflicting = source.iloc[[0]].copy()
        conflicting.loc[:, "StudyDate"] = "20210103"
        conflicting.loc[:, "SeriesInstanceUID"] = "9.9.9"
        conflicting.loc[:, "SeriesQC"] = 4
        source = pd.concat([source, conflicting], ignore_index=True)
        source.loc[source["LONIImage"].eq(2002), "ParticipantID"] = "999_S_9999"
        result = validate(source)
        codes = {item["code"] for item in result["validation"]["errors"]}
        self.assertEqual(result["validation"]["status"], "FAIL")
        self.assertTrue({"conflicting_study_date", "conflicting_series_uid", "conflicting_qc", "subject_mismatch"}.issubset(codes))

    def test_imprecise_date_nonoriginal_and_unrecognized_qc_fail(self) -> None:
        source = catalog_rows()
        source["SeriesQC"] = source["SeriesQC"].astype(object)
        source.loc[0, "StudyDate"] = "2021-01"
        source.loc[0, "SeriesType"] = "Processed"
        source.loc[0, "SeriesQC"] = "looks fine"
        result = validate(source)
        codes = {item["code"] for item in result["validation"]["errors"]}
        self.assertTrue({"invalid_or_imprecise_study_date", "not_original", "blank_or_unrecognized_qc"}.issubset(codes))

    def test_explicit_unknown_and_fail_qc_are_retained_but_ineligible(self) -> None:
        source = catalog_rows()
        source.loc[0, "SeriesQC"] = -1.0
        source.loc[1, "SeriesQC"] = 4.0
        result = validate(source)
        self.assertEqual(result["validation"]["status"], "PASS")
        self.assertEqual(set(result["canonical"]["qc_status"]), {"unknown", "fail"})
        self.assertFalse(result["canonical"]["eligible_for_pairing"].any())
        self.assertEqual(result["validation"]["counts"]["explicit_qc_fail"], 1)
        self.assertEqual(result["validation"]["counts"]["qc_pending"], 1)
        self.assertFalse(result["validation"]["checks"]["all_request_ids_qc_resolved"])
        self.assertFalse(result["validation"]["checks"]["pairing_ready"])

    def test_historical_mayo_quality_profile_accepts_grades_two_and_three(self) -> None:
        source = catalog_rows()
        source = source.rename(columns={"SeriesQC": "SERIES_QUALITY"})
        source["SERIES_QUALITY"] = [2.0, 3.0]
        result = validate(
            source,
            qc_profile="adni_mayo_series_quality",
            default_qc_source="MAYOADIRL_MRI_ADNI3.SERIES_QUALITY",
        )
        self.assertEqual(result["validation"]["status"], "PASS")
        self.assertTrue(result["validation"]["checks"]["pairing_ready"])
        self.assertTrue(result["canonical"]["eligible_for_pairing"].all())

        rejected_by_current_profile = validate(source, qc_profile="adni_mriqc_current")
        self.assertEqual(rejected_by_current_profile["validation"]["status"], "FAIL")

    def test_qc_profile_field_and_source_semantics_cannot_be_swapped(self) -> None:
        current_with_legacy_source = validate(
            catalog_rows(),
            default_qc_source="MAYOADIRL_MRI_ADNI3.SERIES_QUALITY",
        )
        self.assertIn(
            "qc_profile_source_mismatch",
            {item["code"] for item in current_with_legacy_source["validation"]["errors"]},
        )

        legacy = catalog_rows().rename(columns={"SeriesQC": "SERIES_QUALITY"})
        legacy["SERIES_QUALITY"] = [2, 3]
        legacy_with_current_source = validate(
            legacy,
            qc_profile="adni_mayo_series_quality",
            default_qc_source="MRIQC.SeriesQC",
        )
        self.assertIn(
            "qc_profile_source_mismatch",
            {item["code"] for item in legacy_with_current_source["validation"]["errors"]},
        )

        current_field_with_legacy_profile = validate(
            catalog_rows(),
            qc_profile="adni_mayo_series_quality",
            default_qc_source="MAYOADIRL_MRI_ADNI3.SERIES_QUALITY",
        )
        self.assertIn(
            "qc_profile_column_mismatch",
            {item["code"] for item in current_field_with_legacy_profile["validation"]["errors"]},
        )

        generic = catalog_rows().rename(columns={"SeriesQC": "QC"})
        generic["QC"] = ["pass", "pass"]
        reserved_source_through_generic = validate(
            generic,
            qc_profile="generic_text",
            default_qc_source="MRIQC.SeriesQC",
        )
        self.assertIn(
            "qc_profile_source_mismatch",
            {
                item["code"]
                for item in reserved_source_through_generic["validation"]["errors"]
            },
        )

    def test_blank_qc_source_fails_unless_explicit_default_is_supplied(self) -> None:
        source = catalog_rows()
        source["QC Source"] = ""
        failed = validate(source, default_qc_source=None)
        self.assertEqual(failed["validation"]["status"], "FAIL")
        self.assertIn("missing_qc_source", {item["code"] for item in failed["validation"]["errors"]})

        passed = validate(source, default_qc_source="MRIQC.SeriesQC")
        self.assertEqual(passed["validation"]["status"], "PASS")
        self.assertEqual(set(passed["canonical"]["qc_source"]), {"MRIQC.SeriesQC"})

    def test_ambiguous_alias_and_credential_header_fail(self) -> None:
        source = catalog_rows()
        source["Image ID"] = source["LONIImage"]
        source["Access Token"] = "never-store-this"
        result = validate(source)
        codes = {item["code"] for item in result["validation"]["errors"]}
        self.assertIn("column_image_id", codes)
        self.assertIn("credential_like_headers", codes)

    def test_roster_drift_and_present_absence_conflict_fail(self) -> None:
        request = roster()
        request.loc[1, "candidate_t1_image_id"] = request.loc[0, "candidate_t1_image_id"]
        result = validate(catalog_rows(), request=request)
        self.assertEqual(result["validation"]["status"], "FAIL")
        self.assertIn("duplicate_roster_image_id", {item["code"] for item in result["validation"]["errors"]})

        absence = pd.DataFrame({"ImageID": [2001], "reason": ["erroneous absence"]})
        result = validate(catalog_rows(), absence_ledger=absence, absence_sha256="c" * 64)
        self.assertIn("present_and_absent", {item["code"] for item in result["validation"]["errors"]})

    def test_multiple_candidates_per_subject_are_allowed_and_one_pass_is_sufficient(self) -> None:
        request = pd.DataFrame(
            {
                "subject_id": ["001_S_0001", "001_S_0001", "001_S_0002"],
                "group": ["CN", "CN", "MCI"],
                "dti_image_id": [1001, 1001, 1002],
                "t1_image_id": [2001, 2003, 2002],
            }
        )
        source = catalog_rows()
        extra = source.iloc[[0]].copy()
        extra.loc[:, "LONIImage"] = 2003
        extra.loc[:, "SeriesQC"] = 4
        source = pd.concat([source, extra], ignore_index=True)
        result = validate(source, request=request)
        self.assertEqual(result["validation"]["status"], "PASS")
        self.assertTrue(result["validation"]["checks"]["pairing_ready"])
        self.assertEqual(result["validation"]["counts"]["roster_subjects"], 2)

    def test_unknown_candidate_blocks_readiness_even_when_each_subject_has_a_pass(self) -> None:
        request = pd.DataFrame(
            {
                "subject_id": ["001_S_0001", "001_S_0001", "001_S_0002"],
                "group": ["CN", "CN", "MCI"],
                "dti_image_id": [1001, 1001, 1002],
                "t1_image_id": [2001, 2003, 2002],
            }
        )
        source = catalog_rows()
        pending = source.iloc[[0]].copy()
        pending.loc[:, "LONIImage"] = 2003
        pending.loc[:, "SeriesQC"] = -1
        source = pd.concat([source, pending], ignore_index=True)
        result = validate(source, request=request)
        self.assertEqual(result["validation"]["status"], "PASS")
        self.assertEqual(result["validation"]["counts"]["subjects_with_eligible_t1"], 2)
        self.assertFalse(result["validation"]["checks"]["all_request_ids_qc_resolved"])
        self.assertFalse(result["validation"]["checks"]["pairing_ready"])
        self.assertEqual(result["validation"]["release_status"], "INTAKE_PASS_PAIRING_BLOCKED")

    def test_explained_absence_blocks_readiness_even_when_subject_has_another_pass(self) -> None:
        request = pd.DataFrame(
            {
                "subject_id": ["001_S_0001", "001_S_0001", "001_S_0002"],
                "group": ["CN", "CN", "MCI"],
                "dti_image_id": [1001, 1001, 1002],
                "t1_image_id": [2001, 2003, 2002],
            }
        )
        absence = pd.DataFrame(
            {"Image ID": [2003], "absence_reason": ["Not returned by source export"]}
        )
        result = validate(
            catalog_rows(),
            request=request,
            absence_ledger=absence,
            absence_sha256="c" * 64,
        )
        self.assertEqual(result["validation"]["status"], "PASS")
        self.assertEqual(result["validation"]["counts"]["subjects_with_eligible_t1"], 2)
        self.assertFalse(result["validation"]["checks"]["all_request_ids_present_valid"])
        self.assertFalse(result["validation"]["checks"]["pairing_ready"])

    def test_out_of_roster_candidate_for_roster_subject_is_universe_drift(self) -> None:
        source = catalog_rows()
        drift = source.iloc[[0]].copy()
        drift.loc[:, "LONIImage"] = 2999
        source = pd.concat([source, drift], ignore_index=True)
        result = validate(source)
        self.assertEqual(result["validation"]["status"], "FAIL")
        self.assertIn(
            "candidate_universe_drift",
            {item["code"] for item in result["validation"]["errors"]},
        )
        self.assertEqual(result["validation"]["counts"]["candidate_universe_drift_rows"], 1)
        self.assertFalse(result["validation"]["checks"]["candidate_universe_closed"])

    def test_noncandidate_out_of_roster_row_does_not_expand_t1_universe(self) -> None:
        source = catalog_rows()
        other_series = source.iloc[[0]].copy()
        other_series.loc[:, "LONIImage"] = 2999
        other_series.loc[:, "SeriesDescription"] = "Axial FLAIR"
        source = pd.concat([source, other_series], ignore_index=True)
        result = validate(source)
        self.assertEqual(result["validation"]["status"], "PASS")
        self.assertTrue(result["validation"]["checks"]["candidate_universe_closed"])
        self.assertEqual(result["validation"]["counts"]["out_of_roster_rows_ignored"], 1)

    def test_roster_hash_and_study_date_bounds_fail_closed(self) -> None:
        source = catalog_rows()
        bad_hash = validate(source, expected_roster_sha256="c" * 64)
        self.assertIn("roster_sha256_mismatch", {item["code"] for item in bad_hash["validation"]["errors"]})

        source.loc[0, "StudyDate"] = "19990101"
        source.loc[1, "StudyDate"] = "20270101"
        bad_dates = validate(source)
        self.assertEqual(bad_dates["validation"]["status"], "FAIL")
        self.assertIn("implausible_study_date", {item["code"] for item in bad_dates["validation"]["errors"]})

    def test_image_type_is_preserved_as_extra_not_treated_as_original_flag(self) -> None:
        source = catalog_rows()
        source["Image Type"] = "image volume"
        result = validate(source)
        self.assertEqual(result["validation"]["status"], "PASS")
        self.assertIn("Image Type", result["canonical"].iloc[0]["source_extra_json"])

    def test_strict_date_and_id_parsers(self) -> None:
        self.assertEqual(parse_exact_calendar_date("6/02/2011"), "2011-06-02")
        self.assertEqual(parse_exact_calendar_date("2011-06-02"), "2011-06-02")
        self.assertEqual(parse_exact_calendar_date("20110602"), "2011-06-02")
        self.assertEqual(parse_exact_calendar_date(20110602.0), "2011-06-02")
        for invalid in ("2011", "2011-06", "2011-06-02T12:00:00Z", "20110230", ""):
            self.assertIsNone(parse_exact_calendar_date(invalid))
        self.assertEqual(normalize_positive_integer("I123"), 123)
        self.assertEqual(normalize_positive_integer(123.0), 123)
        self.assertIsNone(normalize_positive_integer("123.5"))
        self.assertIsNone(normalize_positive_integer("1e3"))
        self.assertIsNone(normalize_positive_integer(0))

    def test_raw_duplicate_headers_are_rejected_before_pandas_mangling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            path.write_text("Image ID,Image_ID\n1,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate normalized CSV headers"):
                read_csv_snapshot(path)

    def test_write_is_pass_only_and_refuses_nonempty_output(self) -> None:
        result = validate(catalog_rows())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "intake"
            write_outputs(result, output)
            self.assertTrue((output / "exact_date_catalog_v2.csv").is_file())
            self.assertTrue((output / "exact_date_catalog_validation_v2.json").is_file())
            with self.assertRaises(FileExistsError):
                write_outputs(result, output)

        failed = validate(catalog_rows().iloc[[0]].copy())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            write_outputs(failed, output)
            self.assertFalse((output / "exact_date_catalog_v2.csv").exists())
            self.assertTrue((output / "exact_date_catalog_validation_v2.json").is_file())


if __name__ == "__main__":
    unittest.main()
