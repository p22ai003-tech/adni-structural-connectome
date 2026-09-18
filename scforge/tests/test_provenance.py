from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scforge.provenance import (
    build_run_ledger_entries,
    file_record,
    validate_schema,
    verify_file_record,
    write_immutable_json,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "workflow" / "schemas" / "provenance_v1.schema.json"
LEDGER_SCHEMA = ROOT / "workflow" / "schemas" / "run_ledger_v1.schema.json"
MATRICES = (
    "count",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)


def valid_record(artifact: Path, *, unit: str = "S1_I1") -> dict:
    recorded = file_record(artifact)
    return {
        "schema_version": "1.0.0",
        "status": "PASS",
        "generated_utc": "2026-07-18T00:00:00+00:00",
        "recipe_id": "recipe-test",
        "run_id": "a" * 24,
        "unit": unit,
        "source_identity": {
            "subject_id": unit.split("_I")[0],
            "diagnosis_at_dti": "CN",
            "dti_image_id": "1",
            "dti_series_uid": "1.2.3",
            "dti_study_date": "2020-01-01",
            "t1_image_id": "2",
            "t1_series_uid": "1.2.4",
            "t1_study_date": "2020-01-01",
            "abs_pair_gap_days": "0",
            "phase": "TEST",
            "site": "TEST",
            "manufacturer": "TEST",
            "scanner_model": "TEST",
            "field_strength_t": "3.0",
            "protocol": "TEST",
        },
        "source_files": {
            key: recorded
            for key in ("dwi_nifti", "dwi_bvec", "dwi_bval", "dwi_json", "t1_nifti")
        },
        "contract_files": {
            key: recorded
            for key in (
                "manifest",
                "normative_config",
                "resolved_run_config",
                "environment",
                "workflow_source_manifest",
                "execution_preflight",
                "run_context",
                "attempt_context",
                "workflow_requirements_lock",
                "wheel_artifact_manifest",
                "matrix_dictionary",
                "mni_template",
                "provenance_schema",
            )
        },
        "intermediate_files": {
            key: recorded
            for key in (
                "b0_to_t1_fsl",
                "t1_to_b0_fsl",
                "t1_to_b0_itk",
                "mni_to_t1_affine",
                "mni_to_t1_warp",
                "atlas_dwi",
                "five_tt_dwi",
                "wm_fod_norm",
                "tensor",
            )
        },
        "outputs": {
            key: recorded for key in ("tracks", "sift2_weights", "sift2_mu", *MATRICES)
        },
        "execution": {
            "run_started_utc": "2026-07-18T00:00:00+00:00",
            "host": {"node": "test"},
            "tool_versions": {"mrtrix": "3.0.7"},
            "snakemake_invocation": ["snakemake", "--printshellcmds"],
            "command_capture": {
                "mode": "snakemake_printshellcmds_combined_log",
                "printshellcmds": True,
                "final_log_hash_in_run_completion": True,
            },
            "command_logs": [recorded],
            "internal_commands": [],
        },
        "processing_contract": {
            key: {} for key in ("registration", "atlas", "fod", "tensor", "tractography", "sift2", "assignment")
        },
        "qc": {
            "gradient": {"status": "PASS"},
            "atlas": {"status": "PASS"},
            "matrix": {"status": "PASS"},
            "spatial": {"status": "PASS"},
            "preflight": {"status": "PASS", "human_visual_qc": "PASS"},
        },
        "safety": {
            "production_overwrite": False,
            "automatic_fallbacks": False,
            "density_route_selection": False,
        },
    }


class ProvenanceTests(unittest.TestCase):
    def test_complete_record_passes_schema_and_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"evidence")
            sidecar = root / "S1_I1.provenance.json"
            record = valid_record(artifact)
            sidecar.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(validate_schema(record, SCHEMA), [])
            entries = build_run_ledger_entries(
                [sidecar], schema_path=SCHEMA, required_matrices=MATRICES
            )
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["unit"], "S1_I1")

    def test_missing_group_extra_matrix_and_bad_hash_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.bin"
            artifact.write_bytes(b"evidence")
            record = valid_record(artifact)
            missing = copy.deepcopy(record)
            missing.pop("processing_contract")
            self.assertTrue(validate_schema(missing, SCHEMA))
            extra = copy.deepcopy(record)
            extra["outputs"]["unexpected"] = file_record(artifact)
            self.assertTrue(validate_schema(extra, SCHEMA))
            bad_hash = copy.deepcopy(record)
            bad_hash["outputs"]["count"]["sha256"] = "x" * 64
            self.assertTrue(validate_schema(bad_hash, SCHEMA))

    def test_tampered_artifact_fails_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.bin"
            artifact.write_bytes(b"before")
            record = file_record(artifact)
            artifact.write_bytes(b"after")
            self.assertTrue(verify_file_record(record))

    def test_write_once_rejects_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "immutable.json"
            write_immutable_json(destination, {"value": 1})
            with self.assertRaises(FileExistsError):
                write_immutable_json(destination, {"value": 2})

    def test_completed_ledger_manifest_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.bin"
            artifact.write_bytes(b"evidence")
            recorded = file_record(artifact)
            ledger = {
                "schema_version": "1.0.0",
                "ledger_type": "connectome_run_ledger",
                "status": "COMPLETE",
                "generated_utc": "2026-07-18T00:00:00+00:00",
                "run_id": "a" * 24,
                "recipe_id": "recipe-test",
                "run_context": recorded,
                "expected_units": ["S1_I1"],
                "summary": {"expected": 1, "pass": 1, "fail": 0, "excluded": 0},
                "run_ledger": recorded,
                "analysis_ready_manifest": recorded,
                "subject_provenance": {"S1_I1": recorded},
                "publish_only_statuses": ["PASS"],
                "external_digest_anchor_required_for_release": True,
            }
            self.assertEqual(validate_schema(ledger, LEDGER_SCHEMA), [])

    def test_ledger_rejects_duplicate_unit_and_nonpass_qc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"evidence")
            sidecar_a = root / "a.json"
            sidecar_b = root / "b.json"
            record = valid_record(artifact)
            sidecar_a.write_text(json.dumps(record), encoding="utf-8")
            sidecar_b.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_run_ledger_entries(
                    [sidecar_a, sidecar_b], schema_path=SCHEMA, required_matrices=MATRICES
                )
            failed = copy.deepcopy(record)
            failed["qc"]["matrix"]["status"] = "FAIL"
            sidecar_b.write_text(json.dumps(failed), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_run_ledger_entries(
                    [sidecar_b], schema_path=SCHEMA, required_matrices=MATRICES
                )


if __name__ == "__main__":
    unittest.main()
