from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research_audit.build_available_data_manifest import (
    OUTPUT_FLOW,
    OUTPUT_MANIFEST,
    OUTPUT_STRATA,
    OUTPUT_VALIDATION,
    build_available_data_manifest,
    write_release,
)


SUBJECTS = ["001_S_0001", "001_S_0002", "001_S_0003", "001_S_0004"]
DTI_DATE = "2020-01-01"
T1_DATES = ["2020-03-31", "2020-04-01", "2020-06-29", "2020-06-30"]
GAPS = [90, 91, 180, 181]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.exact = root / "exact.csv"
        self.audit = root / "audit.csv"
        self.dryrun = root / "dryrun.csv"
        self.dti_master = root / "dti_master.csv"
        self.subject_manifest = root / "subject_manifest.csv"
        self.audit_run = root / "audit_run.json"

        exact_rows: list[dict[str, object]] = []
        audit_rows: list[dict[str, object]] = []
        dry_rows: list[dict[str, object]] = []
        dti_rows: list[dict[str, object]] = []
        subject_rows: list[dict[str, object]] = []

        for index, (subject, t1_date, gap) in enumerate(
            zip(SUBJECTS, T1_DATES, GAPS), start=1
        ):
            dti_id = 1000 + index
            t1_id = 2000 + index
            dti_root = root / "dti" / subject
            t1_root = root / "mri" / subject
            dti_dir = dti_root / "Axial_DTI" / f"{DTI_DATE}_12_00_00.0" / f"I{dti_id}"
            t1_dir = t1_root / "MPRAGE" / f"{t1_date}_13_00_00.0" / f"I{t1_id}"
            dti_dir.mkdir(parents=True)
            t1_dir.mkdir(parents=True)
            (dti_dir / f"I{dti_id}_1.dcm").write_bytes(b"dti" + bytes([index]))
            (t1_dir / f"I{t1_id}.nii").write_bytes(b"t1" + bytes([index]))
            t1_size = (t1_dir / f"I{t1_id}.nii").stat().st_size
            same_phase = index in {1, 3}
            t1_phase = "ADNI 3" if same_phase else "ADNI 2"
            diagnosis = ["CN", "MCI", "AD", "SMC"][index - 1]

            exact_rows.append(
                {
                    "subject_id": subject,
                    "group": diagnosis,
                    "dti_image_id": dti_id,
                    "dti_study_date": DTI_DATE,
                    "dti_visit": "v1",
                    "dti_master_phase": "ADNI 3",
                    "current_t1_image_id": t1_id,
                    "current_t1_local_subject_id": subject,
                    "current_t1_path_date": t1_date,
                    "current_t1_path_datetime": f"{t1_date}T13:00:00",
                    "current_t1_gap_days_signed": gap,
                    "current_t1_gap_days_abs": gap,
                    "current_within_90_days": str(gap <= 90),
                    "current_within_180_days": str(gap <= 180),
                    "current_t1_image_dir": str(t1_dir),
                    "current_t1_file_count": 1,
                    "current_t1_total_bytes": t1_size,
                    "current_exact_date_source": "local_raw_path_date",
                    "recommended_action": "replace_after_exact_date_and_qc",
                }
            )
            audit_rows.append(
                {
                    "subject_id": subject,
                    "actual_dti_image_id": dti_id,
                    "actual_t1_image_id": t1_id,
                    "Image ID": dti_id,
                    "Image ID_mri": t1_id,
                    "Research Group": diagnosis,
                    "diagnosis_original": diagnosis,
                    "group_dashboard": "MCI" if diagnosis == "SMC" else diagnosis,
                    "group_corrected": diagnosis,
                    "smc_misclassified_as_mci": str(diagnosis == "SMC"),
                    "dti_phase": "ADNI 3",
                    "dti_age": "70.0",
                    "dti_sex": "F",
                    "dti_study_date": "1/1/2020",
                    "dti_visit": "v1",
                    "dti_protocol": "Manufacturer=SIEMENS;Field Strength=3.0;Gradient Directions=54",
                    "dti_description": "Axial DTI",
                    "dti_type": "Original",
                    "t1_phase": t1_phase,
                    "t1_age": "69.0",
                    "t1_description": "MPRAGE",
                    "t1_type": "Original" if index != 4 else "Processed",
                    "manufacturer": "SIEMENS",
                    "field_strength_t": "3.0",
                    "gradient_directions": "54",
                    "protocol_key": "SIEMENS|3T|54dir",
                    "site": "001",
                    "dti_t1_phase_match": str(same_phase),
                    "dti_t1_age_gap_years": str(gap / 365.25),
                    "sc_matrix_qc_status": "PASS" if index == 1 else "FAIL_REGISTRATION",
                    "sc_matrix_qc_gate": "include" if index == 1 else "exclude_pending_repair",
                    "sc_matrix_qc_include": str(index == 1),
                    "n_sc_qc_series": "1.0",
                    "n_sc_qc_failed_series": "0.0" if index == 1 else "1.0",
                    "sc_matrix_qc_statuses": "PASS" if index == 1 else "FAIL_REGISTRATION",
                    "density_y": "0.7",
                    "band": "good",
                    "radial": "4",
                    "mask_quality": "0.9",
                    "reg_ncc": "0.8",
                    "label_survival": "0.95",
                    "t1_path": str(t1_dir / f"I{t1_id}.nii"),
                    "parc_path": str(t1_dir / f"I{t1_id}.nii"),
                    "tck_path": str(dti_dir / f"I{dti_id}_1.dcm"),
                    "sift_path": str(dti_dir / f"I{dti_id}_1.dcm"),
                }
            )
            dry_rows.append(
                {
                    "subject_id": subject,
                    "group": diagnosis,
                    "dti_image_id": dti_id,
                    "dti_phase": "ADNI 3",
                    "current_t1_image_id": t1_id,
                    "current_t1_phase": t1_phase,
                    "current_gap_years": str(gap / 365.25),
                    "current_is_contemporaneous": str(gap <= 180 and same_phase),
                    "recommended_action": "replace_after_exact_date_and_qc",
                }
            )
            dti_rows.append(
                {
                    "Subject ID": subject,
                    "Image ID": dti_id,
                    "Study Date": "1/1/2020",
                    "Visit": "v1",
                    "Phase": "ADNI 3",
                    "Research Group": diagnosis,
                    "Description": "Axial DTI",
                    "Type": "Original",
                    "Imaging Protocol": "Manufacturer=SIEMENS;Field Strength=3.0;Gradient Directions=54",
                    "Age": "70.0",
                    "Sex": "F",
                }
            )
            subject_rows.append(
                {
                    "sid": f"{subject}_I{dti_id}",
                    "group": "MCI" if diagnosis == "SMC" else diagnosis,
                    "band": "good",
                    "raw_dwi_path": str(dti_root),
                    "raw_t1_path": str(t1_root),
                    "t1_path": str(t1_dir / f"I{t1_id}.nii"),
                    "parc_path": str(t1_dir / f"I{t1_id}.nii"),
                    "tck_path": str(dti_dir / f"I{dti_id}_1.dcm"),
                    "sift_path": str(dti_dir / f"I{dti_id}_1.dcm"),
                    "tsf_path": str(dti_dir / f"I{dti_id}_1.dcm"),
                    "density": "0.7",
                    "reg_ncc": "0.8",
                    "label_survival": "0.95",
                    "mask_quality": "0.9",
                }
            )

        write_csv(self.exact, exact_rows)
        write_csv(self.audit, audit_rows)
        write_csv(self.dryrun, dry_rows)
        write_csv(self.dti_master, dti_rows)
        write_csv(self.subject_manifest, subject_rows)
        self.audit_run.write_text(
            json.dumps(
                {
                    "generated_utc": "2026-07-18T00:00:00+00:00",
                    "script": "/audit.py",
                    "inputs": {
                        "dti_master": {"sha256": sha256(self.dti_master)},
                        "manifest": {"sha256": sha256(self.subject_manifest)},
                    },
                }
            ),
            encoding="utf-8",
        )

    def build(self, **kwargs):
        return build_available_data_manifest(
            exact_path=self.exact,
            audit_path=self.audit,
            dryrun_path=self.dryrun,
            dti_master_path=self.dti_master,
            subject_manifest_path=self.subject_manifest,
            audit_run_path=self.audit_run,
            expected_pairs=4,
            **kwargs,
        )


class AvailableDataManifestTests(unittest.TestCase):
    def test_boundaries_long_gap_language_and_write_once_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            result = fixture.build()
            rows = result["records"]
            self.assertEqual(
                [row["timing_stratum"] for row in rows],
                ["le_90_days", "days_91_180", "days_91_180", "gt_180_days"],
            )
            self.assertFalse(rows[-1]["visit_matched_claim"])
            self.assertEqual(
                rows[-1]["visit_match_status"], "not_visit_matched_long_gap"
            )
            self.assertEqual(rows[-1]["dti_master_research_group_raw"], "SMC")
            self.assertEqual(
                rows[-1]["diagnosis_reconciliation_status"],
                "not_attached_pending_SL-P0-05",
            )
            self.assertTrue(all(not row["corrected_rerun_analysis_eligible"] for row in rows))

            output = Path(directory) / "release"
            validation = write_release(
                result, output, generated_utc="2026-07-18T00:00:00+00:00"
            )
            self.assertEqual(
                validation["release_status"], "PROVISIONAL_DIAGNOSIS_JOIN_PENDING"
            )
            self.assertEqual(validation["counts"]["timing_le_90_days"], 1)
            self.assertEqual(validation["counts"]["timing_91_180_days"], 2)
            self.assertEqual(validation["counts"]["timing_gt_180_days"], 1)
            for name in (OUTPUT_MANIFEST, OUTPUT_STRATA, OUTPUT_FLOW, OUTPUT_VALIDATION):
                self.assertTrue((output / name).is_file())
            with self.assertRaises(FileExistsError):
                write_release(result, output)

    def test_tampered_gap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            rows = read_csv(fixture.exact)
            rows[0]["current_t1_gap_days_abs"] = "89"
            write_csv(fixture.exact, rows)
            with self.assertRaisesRegex(ValueError, "absolute exact gap"):
                fixture.build()

    def test_cross_source_image_id_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            rows = read_csv(fixture.audit)
            rows[0]["actual_t1_image_id"] = "9999"
            write_csv(fixture.audit, rows)
            with self.assertRaisesRegex(ValueError, "cohort-audit T1 ID"):
                fixture.build()

    def test_duplicate_subject_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            rows = read_csv(fixture.exact)
            rows[1]["subject_id"] = rows[0]["subject_id"]
            write_csv(fixture.exact, rows)
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                fixture.build()

    def test_path_date_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            rows = read_csv(fixture.exact)
            rows[0]["current_t1_path_date"] = "2020-03-30"
            rows[0]["current_t1_gap_days_signed"] = "89"
            rows[0]["current_t1_gap_days_abs"] = "89"
            rows[0]["current_within_90_days"] = "True"
            write_csv(fixture.exact, rows)
            with self.assertRaisesRegex(ValueError, "T1 path date mismatch"):
                fixture.build()

    def test_diagnosis_attachment_is_checksum_and_key_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            diagnosis = Path(directory) / "diagnosis.csv"
            rows = [
                {
                    "subject_id": subject,
                    "selected_dti_image_id": 1000 + index,
                    "diagnosis_at_dti": ["CN", "MCI", "AD", "SMC"][index - 1],
                    "resolution_status": "resolved",
                }
                for index, subject in enumerate(SUBJECTS, start=1)
            ]
            write_csv(diagnosis, rows)
            with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
                fixture.build(diagnosis_path=diagnosis, diagnosis_sha256="a" * 64)

            result = fixture.build(
                diagnosis_path=diagnosis, diagnosis_sha256=sha256(diagnosis)
            )
            self.assertEqual(
                result["records"][0]["diagnosis_reconciliation_status"],
                "attached_checksum_verified",
            )
            self.assertIn(
                '"diagnosis_at_dti":"CN"',
                result["records"][0]["diagnosis_reconciliation_payload_json"],
            )

            write_csv(diagnosis, rows[:-1])
            with self.assertRaisesRegex(ValueError, "do not exactly cover"):
                fixture.build(
                    diagnosis_path=diagnosis, diagnosis_sha256=sha256(diagnosis)
                )


if __name__ == "__main__":
    unittest.main()
