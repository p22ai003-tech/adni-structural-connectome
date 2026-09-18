from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

from research_audit.build_source_metadata_preflight import (
    DEFAULT_FINAL_DIR,
    InventoryFile,
    OUTPUT_TABLE,
    OUTPUT_VALIDATION,
    SourceMutationError,
    build_projection_row,
    dicom_uid_syntax,
    inspect_nifti_header,
    stable_source_id,
    summarize_dicom_series,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory_file(
    path: Path,
    *,
    modality: str,
    image_id: int,
    pair_id: str = "PAIR-1",
    subject_id: str = "001_S_0001",
) -> InventoryFile:
    metadata = path.stat()
    return InventoryFile(
        pair_id=pair_id,
        subject_id=subject_id,
        modality=modality,
        image_id=image_id,
        series_dir=str(path.parent),
        relative_path=path.name,
        absolute_path=str(path),
        size_bytes=metadata.st_size,
        sha256=_sha256(path),
        st_dev=metadata.st_dev,
        st_ino=metadata.st_ino,
        st_mtime_ns=metadata.st_mtime_ns,
        st_ctime_ns=metadata.st_ctime_ns,
    )


def _write_dicom(
    path: Path,
    *,
    series_uid: str = "1.2.826.0.1.3680043.8.498.1",
    sop_uid: str = "1.2.826.0.1.3680043.8.498.101",
    instance: int = 1,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.999"
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SeriesInstanceUID = series_uid
    dataset.StudyInstanceUID = "1.2.826.0.1.3680043.8.498.2"
    dataset.SOPInstanceUID = sop_uid
    dataset.SOPClassUID = MRImageStorage
    dataset.PatientID = "001_S_0001"
    dataset.Manufacturer = "SIEMENS"
    dataset.ManufacturerModelName = "Prisma_fit"
    dataset.MagneticFieldStrength = "3"
    dataset.InPlanePhaseEncodingDirection = "COL"
    dataset.AcquisitionMatrix = [0, 2, 2, 0]
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.InstanceNumber = instance
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelData = b"\0\0" * 4
    dataset.save_as(path, enforce_file_format=True)


class SourceMetadataPreflightTests(unittest.TestCase):
    def test_canonical_release_is_checksum_bound_and_conversion_pending(self) -> None:
        validation_path = DEFAULT_FINAL_DIR / OUTPUT_VALIDATION
        table_path = DEFAULT_FINAL_DIR / OUTPUT_TABLE
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        self.assertEqual(validation["status"], "PASS")
        self.assertEqual(validation["counts"]["pairs"], 530)
        self.assertEqual(validation["counts"]["source_identity_blocked"], 0)
        self.assertEqual(validation["counts"]["conversion_pending"], 530)
        self.assertFalse(validation["canary_authorized"])
        self.assertTrue(all(validation["checks"].values()))
        expected = validation["outputs"]["workflow_projection"]
        self.assertEqual(hashlib.sha256(table_path.read_bytes()).hexdigest(), expected["sha256"])
        with table_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 530)
        self.assertEqual(len({row["pair_id"] for row in rows}), 530)
        self.assertTrue(all(not row["phase_encoding_direction"] for row in rows))
        self.assertTrue(all(not row["total_readout_time"] for row in rows))
        self.assertTrue(all(row["canary_ready"] == "false" for row in rows))

    def test_uid_syntax_and_stable_source_id_are_fail_closed_and_deterministic(self) -> None:
        self.assertEqual(dicom_uid_syntax("1.2.840.10008.1.2.1"), "VALID")
        self.assertEqual(
            dicom_uid_syntax("1.02.3"),
            "NONCONFORMANT_LEADING_ZERO_COMPONENT",
        )
        bundle = "a" * 64
        first = stable_source_id("T1", "123", bundle)
        self.assertEqual(first, stable_source_id("T1", 123, bundle))
        self.assertNotEqual(first, stable_source_id("T1", 124, bundle))
        with self.assertRaisesRegex(ValueError, "valid series bundle"):
            stable_source_id("T1", 123, "not-a-hash")

    def test_dicom_summary_reads_headers_only_and_checks_all_file_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "one.dcm"
            second_path = root / "two.dcm"
            _write_dicom(first_path)
            _write_dicom(
                second_path,
                sop_uid="1.2.826.0.1.3680043.8.498.102",
                instance=2,
            )
            files = [
                _inventory_file(first_path, modality="DTI", image_id=101),
                _inventory_file(second_path, modality="DTI", image_id=101),
            ]
            with mock.patch(
                "research_audit.build_source_metadata_preflight.os.close",
                wraps=os.close,
            ) as explicit_close:
                result = summarize_dicom_series(files, "001_S_0001")
            self.assertEqual(
                explicit_close.call_count,
                0,
                "fdopen owns the descriptor; a redundant close is thread-unsafe",
            )
            self.assertTrue(result["identity_ready"])
            self.assertEqual(result["headers_read"], 2)
            self.assertEqual(result["series_uid_consistency_status"], "PASS")
            self.assertEqual(result["scanner_model_status"], "PASS")
            self.assertEqual(result["acquisition_matrix_values"], '["0","2","2","0"]')

            changed_path = root / "changed.dcm"
            _write_dicom(
                changed_path,
                series_uid="1.2.826.0.1.3680043.8.498.3",
                sop_uid="1.2.826.0.1.3680043.8.498.103",
                instance=3,
            )
            changed = _inventory_file(changed_path, modality="DTI", image_id=101)
            conflict = summarize_dicom_series(files + [changed], "001_S_0001")
            self.assertFalse(conflict["identity_ready"])
            self.assertEqual(
                conflict["series_uid_consistency_status"],
                "FAIL_SERIES_UID_CONFLICT",
            )

    def test_locked_file_stat_mutation_is_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.dcm"
            _write_dicom(path)
            item = _inventory_file(path, modality="DTI", image_id=101)
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaises(SourceMutationError):
                summarize_dicom_series([item], "001_S_0001")

    def test_processed_t1_has_nullable_dicom_uid_and_conversion_fields_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dti_path = root / "dti.dcm"
            t1_path = root / "t1.nii"
            _write_dicom(dti_path)
            image = nib.Nifti1Image(
                np.zeros((2, 3, 4), dtype=np.float32),
                np.eye(4),
            )
            nib.save(image, t1_path)
            dti_item = _inventory_file(dti_path, modality="DTI", image_id=101)
            t1_item = _inventory_file(t1_path, modality="T1", image_id=202)
            dti = summarize_dicom_series([dti_item], "001_S_0001")
            t1 = inspect_nifti_header([t1_item])
            source = {
                "pair_id": "PAIR-1",
                "subject_id": "001_S_0001",
                "diagnosis_reconciled_harmonized": "CN",
                "diagnosis_reconciled_primary_analysis_eligible": "true",
                "diagnosis_reconciled_smc_retained_separately": "false",
                "dti_image_id": "101",
                "current_t1_image_id": "202",
                "dti_study_date": "2020-01-01",
                "current_t1_study_date": "2020-01-01",
                "dti_t1_gap_days_abs": "0",
                "dti_phase": "ADNI3",
                "site": "001",
                "dti_protocol_key": "TEST_PROTOCOL",
                "dti_manufacturer": "Siemens",
                "dti_field_strength_t": "3.0",
                "current_t1_type": "Processed",
                "local_dti_content_bundle_sha256": "a" * 64,
                "local_t1_content_bundle_sha256": "b" * 64,
                "pair_content_bundle_sha256": "c" * 64,
                "local_dti_series_dir": str(root),
                "local_t1_series_dir": str(root),
                "local_dti_content_file_count": "1",
                "local_dti_content_total_bytes": str(dti_item.size_bytes),
                "local_t1_content_file_count": "1",
                "local_t1_content_total_bytes": str(t1_item.size_bytes),
            }
            evidence = {
                "content_lock_validation": {"sha256": "d" * 64},
                "locked_pair_manifest": {"sha256": "e" * 64},
                "locked_file_inventory": {"sha256": "f" * 64},
                "canonical_pair_manifest_sha256": "1" * 64,
                "file_universe_sha256": "2" * 64,
            }
            row = build_projection_row(source, dti, None, t1, evidence)
            self.assertEqual(row["t1_dicom_series_uid"], "")
            self.assertRegex(row["t1_source_id"], r"^[0-9a-f]{64}$")
            self.assertEqual(row["t1_nifti_header_status"], "PASS_HEADER_ONLY")
            self.assertEqual(row["phase_encoding_direction"], "")
            self.assertEqual(row["total_readout_time"], "")
            self.assertEqual(row["gradient_table_status"], "PENDING_CONVERSION")
            self.assertFalse(row["canary_ready"])


if __name__ == "__main__":
    unittest.main()
