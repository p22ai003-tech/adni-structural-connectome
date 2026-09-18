from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "workflow"
sys.path.insert(0, str(WORKFLOW_DIR))

from freeze_response_calibration_retry4 import (  # noqa: E402
    load_execution_subset_technical_metadata_retry4,
)
from scforge.response_calibration import file_record  # noqa: E402


class Retry4FreezeAdapterTests(unittest.TestCase):
    def test_canonical_acquisition_columns_derive_exact_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            subset = Path(directory) / "subset.csv"
            subset.write_text(
                "subject_id,dti_image_id,manufacturer,t1_source_kind,diagnosis_at_dti\n"
                "003_S_4118,1124861,GE MEDICAL SYSTEMS,nifti_single,CN\n"
                "014_S_6087,926924,SIEMENS,dicom_series,MCI\n",
                encoding="utf-8",
            )
            units = ["003_S_4118_I1124861", "014_S_6087_I926924"]
            record, metadata = load_execution_subset_technical_metadata_retry4(
                {
                    "execution_subset_manifest": file_record(subset),
                    "units": units,
                }
            )
            self.assertEqual(record, file_record(subset))
            self.assertEqual(set(metadata), set(units))
            self.assertEqual(metadata[units[0]]["manufacturer_family"], "GE")
            self.assertEqual(metadata[units[0]]["t1_source_class"], "nifti_single")
            self.assertEqual(metadata[units[1]]["manufacturer_family"], "SIEMENS")
            self.assertEqual(metadata[units[1]]["t1_source_class"], "dicom_series")

    def test_explicit_and_canonical_unit_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            subset = Path(directory) / "subset.csv"
            subset.write_text(
                "unit,subject_id,dti_image_id,manufacturer,t1_source_kind\n"
                "WRONG,003_S_4118,1124861,GE MEDICAL SYSTEMS,nifti_single\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "explicit and canonical unit identities differ"
            ):
                load_execution_subset_technical_metadata_retry4(
                    {
                        "execution_subset_manifest": file_record(subset),
                        "units": ["003_S_4118_I1124861"],
                    }
                )

    def test_missing_identity_columns_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            subset = Path(directory) / "subset.csv"
            subset.write_text(
                "manufacturer,t1_source_kind\nSIEMENS,dicom_series\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "lacks technical diversity fields"):
                load_execution_subset_technical_metadata_retry4(
                    {
                        "execution_subset_manifest": file_record(subset),
                        "units": ["014_S_6087_I926924"],
                    }
                )


if __name__ == "__main__":
    unittest.main()
