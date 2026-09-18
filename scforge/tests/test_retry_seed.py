from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scforge.retry_seed import sha256_file, validate_retry_seed_manifest
from scforge.retry_seed_retry4 import validate_retry4_seed_manifest


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


class RetrySeedValidationTests(unittest.TestCase):
    def build_fixture(self, root: Path) -> tuple[Path, dict[str, object]]:
        source = root / "retry2"
        destination = root / "retry3"
        destination_file = destination / "subjects/U1/01_dwi/dwi_preproc.mif"
        destination_file.parent.mkdir(parents=True)
        destination_file.write_bytes(b"validated corrected DWI")
        attempt_end = source / "attempts/retry2.end.json"
        completion = source / "publication/response_calibration_phase_a_completion.json"
        attempt_end.parent.mkdir(parents=True)
        completion.parent.mkdir(parents=True)
        attempt_end.write_text('{"status":"FAIL"}\n', encoding="utf-8")
        completion.write_text('{"status":"FAIL"}\n', encoding="utf-8")
        manifest = {
            "schema_version": "1.0.0",
            "record_type": "h04a_r1_retry_seed_manifest",
            "status": "PASS",
            "retry_id": "SL-H04A-R1-RETRY3",
            "copy_mode": "reflink_copy_on_write",
            "selection_uses_diagnosis_labels": False,
            "source_run_root": str(source),
            "destination_run_root": str(destination),
            "source_attempt_end": file_record(attempt_end),
            "source_completion": file_record(completion),
            "approved_units": ["U1"],
            "files": [
                {
                    "relative_path": "subjects/U1/01_dwi/dwi_preproc.mif",
                    "sha256": sha256_file(destination_file),
                    "size_bytes": destination_file.stat().st_size,
                }
            ],
            "file_count": 1,
            "total_bytes": destination_file.stat().st_size,
        }
        manifest_path = root / "seed_manifest.json"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        return destination, file_record(manifest_path)

    def test_accepts_exact_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination, record = self.build_fixture(Path(directory))
            result = validate_retry_seed_manifest(
                record, destination_root=destination, approved_units=["U1"]
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["file_count"], 1)

    def test_rejects_destination_content_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination, record = self.build_fixture(Path(directory))
            (destination / "subjects/U1/01_dwi/dwi_preproc.mif").write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "size differs|SHA-256 differs"):
                validate_retry_seed_manifest(
                    record, destination_root=destination, approved_units=["U1"]
                )

    def test_rejects_unapproved_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination, record = self.build_fixture(Path(directory))
            manifest_path = Path(str(record["path"]))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"][0]["relative_path"] = (
                "subjects/U1/07_tractography/tracks_10m.tck"
            )
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            record = file_record(manifest_path)
            with self.assertRaisesRegex(ValueError, "outside the approved subject areas"):
                validate_retry_seed_manifest(
                    record, destination_root=destination, approved_units=["U1"]
                )

    def test_retry4_accepts_only_immutable_upstream_areas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination, record = self.build_fixture(Path(directory))
            manifest_path = Path(str(record["path"]))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["retry_id"] = "SL-H04A-R1-RETRY4"
            manifest["allowed_subject_areas"] = ["00_inputs", "01_dwi"]
            manifest["continuation_paths_excluded"] = True
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            record = file_record(manifest_path)

            result = validate_retry4_seed_manifest(
                record, destination_root=destination, approved_units=["U1"]
            )

            self.assertTrue(result["continuation_paths_excluded"])
            self.assertEqual(
                result["immutable_subject_areas"], ["00_inputs", "01_dwi"]
            )
            self.assertTrue(result["content_rehashed"])

            worker_result = validate_retry4_seed_manifest(
                record,
                destination_root=destination,
                approved_units=["U1"],
                rehash_content=False,
            )
            self.assertFalse(worker_result["content_rehashed"])

    def test_retry4_rejects_continuation_log_even_when_base_allows_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination, record = self.build_fixture(root)
            mutable_log = destination / "subjects/U1/logs/05_select_fod_shells.log"
            mutable_log.parent.mkdir(parents=True)
            mutable_log.write_bytes(b"")
            manifest_path = Path(str(record["path"]))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["retry_id"] = "SL-H04A-R1-RETRY4"
            manifest["allowed_subject_areas"] = ["00_inputs", "01_dwi"]
            manifest["continuation_paths_excluded"] = True
            manifest["files"] = [
                {
                    "relative_path": "subjects/U1/logs/05_select_fod_shells.log",
                    "sha256": sha256_file(mutable_log),
                    "size_bytes": 0,
                }
            ]
            manifest["file_count"] = 1
            manifest["total_bytes"] = 1
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            record = file_record(manifest_path)

            with self.assertRaisesRegex(ValueError, "mutable continuation path"):
                validate_retry4_seed_manifest(
                    record, destination_root=destination, approved_units=["U1"]
                )


if __name__ == "__main__":
    unittest.main()
