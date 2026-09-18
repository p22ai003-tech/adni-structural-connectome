from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research_audit.lock_available_data_content import (
    CHECKPOINT_RELEASE_NAME,
    INVENTORY_NAME,
    LOCKED_MANIFEST_NAME,
    SUMMARY_NAME,
    Checkpoint,
    enumerate_selected_files,
    hash_file_stable,
    load_canonical,
    run_lock,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def names_sizes(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.iterdir(), key=lambda value: value.name):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode())
        digest.update(b"\n")
    return digest.hexdigest()


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.dti = root / "dti" / "I1001"
        self.t1 = root / "t1" / "I2001"
        self.dti.mkdir(parents=True)
        self.t1.mkdir(parents=True)
        (self.dti / "a.dcm").write_bytes(b"identical")
        (self.t1 / "b.nii").write_bytes(b"identical")
        self.manifest = root / "manifest.csv"
        self.validation = root / "validation.json"
        self.columns = [
            "pair_id",
            "subject_id",
            "dti_image_id",
            "current_t1_image_id",
            "local_dti_series_dir",
            "local_t1_series_dir",
            "local_dti_file_count",
            "local_dti_total_bytes",
            "local_dti_names_sizes_sha256",
            "local_t1_file_count",
            "local_t1_total_bytes",
            "local_t1_names_sizes_sha256",
            "pair_record_sha256",
            "provenance_source_bundle_sha256",
        ]
        self.row = {
            "pair_id": "001_S_0001_I1001_I2001",
            "subject_id": "001_S_0001",
            "dti_image_id": "1001",
            "current_t1_image_id": "2001",
            "local_dti_series_dir": str(self.dti),
            "local_t1_series_dir": str(self.t1),
            "local_dti_file_count": "1",
            "local_dti_total_bytes": str((self.dti / "a.dcm").stat().st_size),
            "local_dti_names_sizes_sha256": names_sizes(self.dti),
            "local_t1_file_count": "1",
            "local_t1_total_bytes": str((self.t1 / "b.nii").stat().st_size),
            "local_t1_names_sizes_sha256": names_sizes(self.t1),
            "pair_record_sha256": "a" * 64,
            "provenance_source_bundle_sha256": "b" * 64,
        }
        self.write_sources()

    def write_sources(self) -> None:
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.columns, lineterminator="\n")
            writer.writeheader()
            writer.writerow(self.row)
        manifest_hash = sha256(self.manifest)
        self.validation.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "release_status": "READY_FOR_HUMAN_POLICY_GATE",
                    "outputs": {
                        "pair_manifest": {
                            "sha256": manifest_hash,
                            "size_bytes": self.manifest.stat().st_size,
                            "row_count": 1,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    def canonical(self):
        return load_canonical(
            self.manifest,
            self.validation,
            expected_manifest_sha256=sha256(self.manifest),
            expected_pairs=1,
        )


class ContentLockTests(unittest.TestCase):
    def test_enumeration_hashes_files_and_rejects_duplicate_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            canonical = fixture.canonical()
            inventory = enumerate_selected_files(canonical)
            self.assertEqual(len(inventory.files), 2)
            self.assertEqual(inventory.total_bytes, 18)
            self.assertEqual(
                hash_file_stable(inventory.files[0]),
                hashlib.sha256(b"identical").hexdigest(),
            )
            canonical.records[0]["local_t1_series_dir"] = str(fixture.dti)
            with self.assertRaisesRegex(ValueError, "Duplicate selected series root"):
                enumerate_selected_files(canonical)

    def test_mutation_during_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            item = enumerate_selected_files(fixture.canonical()).files[0]

            def mutate() -> None:
                Path(item.absolute_path).write_bytes(b"changed-and-longer")

            with self.assertRaisesRegex(RuntimeError, "mutated during hashing"):
                hash_file_stable(item, block_size=2, after_first_block=mutate)

    def test_checkpoint_is_universe_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            first = Checkpoint(path, {"manifest": "a", "universe": "b"})
            first.close()
            with self.assertRaisesRegex(ValueError, "binding"):
                Checkpoint(path, {"manifest": "a", "universe": "c"})

    def test_full_release_is_atomic_write_once_and_reports_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = Fixture(root)
            work = root / "work"
            release = root / "release"
            result = run_lock(
                manifest_path=fixture.manifest,
                validation_path=fixture.validation,
                work_dir=work,
                final_dir=release,
                expected_manifest_sha256=sha256(fixture.manifest),
                expected_pairs=1,
                progress_every=1,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["counts"]["files"], 2)
            self.assertEqual(result["counts"]["duplicate_content_hash_groups"], 1)
            self.assertTrue((release / INVENTORY_NAME).is_file())
            self.assertTrue((release / LOCKED_MANIFEST_NAME).is_file())
            self.assertTrue((release / SUMMARY_NAME).is_file())
            self.assertTrue((release / CHECKPOINT_RELEASE_NAME).is_file())
            with (release / LOCKED_MANIFEST_NAME).open(encoding="utf-8") as stream:
                locked = next(csv.DictReader(stream))
            self.assertEqual(locked["pair_record_sha256"], fixture.row["pair_record_sha256"])
            self.assertRegex(locked["locked_pair_record_sha256"], r"^[0-9a-f]{64}$")
            with self.assertRaises(FileExistsError):
                run_lock(
                    manifest_path=fixture.manifest,
                    validation_path=fixture.validation,
                    work_dir=work,
                    final_dir=release,
                    expected_manifest_sha256=sha256(fixture.manifest),
                    expected_pairs=1,
                    progress_every=1,
                )

    def test_nonpositive_progress_interval_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "progress_every"):
                run_lock(
                    manifest_path=fixture.manifest,
                    validation_path=fixture.validation,
                    work_dir=Path(directory) / "work",
                    final_dir=Path(directory) / "release",
                    expected_manifest_sha256=sha256(fixture.manifest),
                    expected_pairs=1,
                    progress_every=0,
                )


if __name__ == "__main__":
    unittest.main()
