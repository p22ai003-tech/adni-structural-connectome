from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scforge.provenance import (
    MATRIX_OUTCOMES,
    REQUIRED_OUTCOMES,
    build_run_ledger_entries,
    file_record,
    make_failure_terminal_record,
    matrix_qc_outcome_invalidity,
    publish_terminal_ledger,
    validate_schema,
    validate_terminal_semantics,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "workflow" / "schemas" / "provenance_v2.schema.json"
LEDGER_SCHEMA = ROOT / "workflow" / "schemas" / "run_ledger_v2.schema.json"
RUN_ID = "b" * 24
RECIPE_ID = "connectome-v2.1.0-test"


def source_identity(unit: str) -> dict[str, str]:
    return {
        "subject_id": unit.split("_I")[0],
        "diagnosis_at_dti": "CN",
        "dti_image_id": unit.rsplit("_I", 1)[-1],
        "t1_image_id": "2",
        "abs_pair_gap_days": "0",
    }


def pass_record(artifact: Path, *, unit: str = "S1_I1") -> dict:
    recorded = file_record(artifact)
    outputs = {
        name: recorded
        for name in ("tracks", "sift2_weights", "sift2_mu", *MATRIX_OUTCOMES)
    }
    outcomes = {}
    for name in REQUIRED_OUTCOMES:
        outcome = {"status": "VALID", "reason": None}
        if name in MATRIX_OUTCOMES:
            outcome["artifact"] = recorded
        outcomes[name] = outcome
    return {
        "schema_version": "2.0.0",
        "record_type": "connectome_subject_terminal_state",
        "status": "PASS",
        "generated_utc": "2026-07-18T01:01:00+00:00",
        "started_utc": "2026-07-18T01:00:00+00:00",
        "ended_utc": "2026-07-18T01:01:00+00:00",
        "recipe_id": RECIPE_ID,
        "run_id": RUN_ID,
        "unit": unit,
        "terminal_stage": "complete",
        "primary_failure_reason": None,
        "secondary_flags": [],
        "source_identity": source_identity(unit),
        "source_identity_hashes": {"dti_bundle": "a" * 64, "t1_bundle": "b" * 64},
        "staged_input_hashes": {"dwi_raw": "c" * 64, "t1_native": "d" * 64},
        "source_files": {"dwi_raw": recorded, "t1_native": recorded},
        "contract_files": {"run_context": recorded},
        "intermediate_files": {"tensor": recorded},
        "outputs": outputs,
        "outcome_validity": outcomes,
        "execution": {
            "attempt_started_utc": "2026-07-18T01:00:00+00:00",
            "terminal_utc": "2026-07-18T01:01:00+00:00",
            "command_logs": [recorded],
            "internal_commands": [],
        },
        "processing_contract": {
            key: {}
            for key in (
                "registration",
                "atlas",
                "fod",
                "tensor",
                "tractography",
                "sift2",
                "assignment",
            )
        },
        "qc": {"matrix": {"status": "PASS"}},
        "safety": {
            "production_overwrite": False,
            "automatic_fallbacks": False,
            "density_route_selection": False,
            "invalid_values_replaced": False,
        },
    }


def partial_record(artifact: Path, *, unit: str = "S2_I2") -> dict:
    record = pass_record(artifact, unit=unit)
    recorded = file_record(artifact)
    record.update(
        {
            "status": "PARTIAL",
            "terminal_stage": "matrix_qc",
            "primary_failure_reason": "matrix_bundle_failed_strict_qc",
            "secondary_flags": ["invalid_matrices_retained_as_nonanalytical_evidence"],
            "outputs": {
                "tracks": recorded,
                "sift2_weights": recorded,
                "sift2_mu": recorded,
            },
        }
    )
    record["outcome_validity"] = {
        name: (
            {"status": "VALID", "reason": None}
            if name in {"processing", "tractography"}
            else {"status": "NA", "reason": "matrix_bundle_failed_strict_qc"}
        )
        for name in REQUIRED_OUTCOMES
    }
    record["qc"] = {"matrix": {"status": "FAIL"}}
    return record


def write_record(path: Path, record: dict) -> Path:
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def matrix_qc_record(*failures: str) -> dict:
    status = "FAIL" if failures else "PASS"
    return {
        "status": status,
        "pass_strict": not failures,
        "expected_nodes": 166,
        "matrix_names": list(MATRIX_OUTCOMES),
        "failures": list(failures),
        "matrix_summaries": {
            name: {
                "shape": [166, 166],
                "finite": True,
                "symmetric": True,
                "zero_diagonal": True,
                "nonnegative": True,
            }
            for name in MATRIX_OUTCOMES
        },
    }


class TerminalProvenanceV2Tests(unittest.TestCase):
    def test_per_matrix_invalidity_preserves_independent_artifacts(self) -> None:
        invalid = matrix_qc_outcome_invalidity(
            matrix_qc_record("fa_mean:outside_0_to_1")
        )
        self.assertEqual(
            {name for name, reasons in invalid.items() if reasons},
            {"fa_mean"},
        )

        invalid = matrix_qc_outcome_invalidity(
            matrix_qc_record("len_mean:zero_on_connected_edge")
        )
        self.assertEqual(
            {name for name, reasons in invalid.items() if reasons},
            {
                "count",
                "len_mean",
                "invlen_mean",
                "fa_mean",
                "md_mean",
                "rd_mean",
                "ad_mean",
            },
        )

    def test_cross_matrix_failures_invalidate_every_implicated_metric(self) -> None:
        cases = {
            "count_invnodevol:formula_mismatch": {
                "count",
                "count_invnodevol",
            },
            "count_and_fd_sum:allclose_duplicate": {"count", "fd_sum"},
            "tensor_identity:md_ne_ad_plus_2rd_over_3": {
                "md_mean",
                "rd_mean",
                "ad_mean",
            },
        }
        for failure, expected in cases.items():
            with self.subTest(failure=failure):
                invalid = matrix_qc_outcome_invalidity(
                    matrix_qc_record(failure)
                )
                self.assertEqual(
                    {name for name, reasons in invalid.items() if reasons},
                    expected,
                )

    def test_low_support_is_retained_as_quantitative_qc_not_invalidity(self) -> None:
        invalid = matrix_qc_outcome_invalidity(
            matrix_qc_record(
                "count:unexplained_zero_nodes:166",
                "endpoint_assignment_fraction:0.71",
                "unique_assigned_nodes:143",
                "top5_endpoint_fraction:0.41",
            )
        )
        self.assertTrue(all(not reasons for reasons in invalid.values()))

    def test_unknown_matrix_failure_invalidates_the_complete_bundle(self) -> None:
        invalid = matrix_qc_outcome_invalidity(
            matrix_qc_record("novel_unclassified_qc_failure")
        )
        self.assertEqual(
            {name for name, reasons in invalid.items() if reasons},
            set(MATRIX_OUTCOMES),
        )

    def test_mixed_pass_partial_fail_publication_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"immutable evidence")
            run_context = root / "run_context.json"
            run_context.write_text("{}\n", encoding="utf-8")
            pass_path = write_record(root / "pass.json", pass_record(artifact))
            partial_path = write_record(root / "partial.json", partial_record(artifact))
            failure = make_failure_terminal_record(
                run_id=RUN_ID,
                recipe_id=RECIPE_ID,
                unit="S3_I3",
                terminal_stage="source_normalization",
                primary_failure_reason="missing_required_dwi_metadata",
                started_utc="2026-07-18T01:00:00+00:00",
                ended_utc="2026-07-18T01:00:10+00:00",
                source_identity=source_identity("S3_I3"),
                source_identity_hashes={"dti_bundle": "e" * 64, "t1_bundle": "f" * 64},
                staged_input_hashes={},
                source_files={},
                contract_files={"run_context": file_record(run_context)},
                command_logs=[file_record(artifact)],
            )
            fail_path = write_record(root / "fail.json", failure)
            for record in (pass_record(artifact), partial_record(artifact), failure):
                self.assertEqual(validate_schema(record, SCHEMA), [])
                self.assertEqual(validate_terminal_semantics(record), [])

            publication = root / "publication"
            manifest = publish_terminal_ledger(
                [partial_path, fail_path, pass_path],
                provenance_schema_path=SCHEMA,
                ledger_schema_path=LEDGER_SCHEMA,
                run_context_path=run_context,
                expected_units=("S1_I1", "S2_I2", "S3_I3"),
                required_matrices=MATRIX_OUTCOMES,
                analysis_manifest_path=publication / "analysis_ready_manifest.csv",
                outcome_manifest_path=publication / "outcome_validity_manifest.csv",
                ledger_path=publication / "run_ledger.jsonl",
                ledger_manifest_path=publication / "run_ledger_manifest.json",
                generated_utc="2026-07-18T02:00:00+00:00",
            )
            self.assertEqual(validate_schema(manifest, LEDGER_SCHEMA), [])
            self.assertEqual(manifest["run_outcome"], "PARTIAL")
            self.assertEqual(
                manifest["summary"],
                {
                    "expected": 3,
                    "terminal": 3,
                    "pass": 1,
                    "partial": 1,
                    "fail": 1,
                    "analysis_ready": 1,
                    "outcome_valid_rows": len(REQUIRED_OUTCOMES) + 2,
                },
            )
            with (publication / "analysis_ready_manifest.csv").open(newline="") as handle:
                self.assertEqual([row["unit"] for row in csv.DictReader(handle)], ["S1_I1"])
            with (publication / "outcome_validity_manifest.csv").open(newline="") as handle:
                outcome_rows = list(csv.DictReader(handle))
            self.assertEqual(len(outcome_rows), 3 * len(REQUIRED_OUTCOMES))
            self.assertTrue(
                all(
                    row["validity"] == "NA"
                    for row in outcome_rows
                    if row["unit"] == "S3_I3"
                )
            )

    def test_duplicate_and_partial_attempt_sets_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"evidence")
            first = write_record(root / "first.json", pass_record(artifact))
            duplicate = write_record(root / "duplicate.json", pass_record(artifact))
            with self.assertRaisesRegex(ValueError, "duplicate terminal-record unit"):
                build_run_ledger_entries(
                    [first, duplicate],
                    schema_path=SCHEMA,
                    required_matrices=MATRIX_OUTCOMES,
                    expected_units=("S1_I1",),
                )
            with self.assertRaisesRegex(ValueError, "missing=S2_I2"):
                build_run_ledger_entries(
                    [first],
                    schema_path=SCHEMA,
                    required_matrices=MATRIX_OUTCOMES,
                    expected_units=("S1_I1", "S2_I2"),
                )

    def test_na_outcome_cannot_publish_artifact_or_zero_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.bin"
            artifact.write_bytes(b"evidence")
            record = partial_record(artifact)
            record["outcome_validity"]["fa_mean"] = {
                "status": "NA",
                "reason": "nonfinite",
                "artifact": file_record(artifact),
            }
            record["outputs"]["fa_mean"] = file_record(artifact)
            errors = validate_schema(record, SCHEMA) + validate_terminal_semantics(record)
            self.assertTrue(any("NA outcome" in error or "artifact" in error for error in errors))
            sentinel = copy.deepcopy(record)
            sentinel["outcome_validity"]["fa_mean"] = {
                "status": "NA",
                "reason": 0,
            }
            self.assertTrue(validate_schema(sentinel, SCHEMA))

    def test_tamper_and_partial_existing_publication_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"evidence")
            run_context = root / "run_context.json"
            run_context.write_text("{}\n", encoding="utf-8")
            terminal = write_record(root / "terminal.json", pass_record(artifact))
            publication = root / "publication"
            kwargs = dict(
                terminal_record_paths=[terminal],
                provenance_schema_path=SCHEMA,
                ledger_schema_path=LEDGER_SCHEMA,
                run_context_path=run_context,
                expected_units=("S1_I1",),
                required_matrices=MATRIX_OUTCOMES,
                analysis_manifest_path=publication / "analysis_ready_manifest.csv",
                outcome_manifest_path=publication / "outcome_validity_manifest.csv",
                ledger_path=publication / "run_ledger.jsonl",
                ledger_manifest_path=publication / "run_ledger_manifest.json",
                generated_utc="2026-07-18T02:00:00+00:00",
            )
            publish_terminal_ledger(**kwargs)
            (publication / "analysis_ready_manifest.csv").write_text(
                "tampered\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "publication differs"):
                publish_terminal_ledger(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"evidence")
            run_context = root / "run_context.json"
            run_context.write_text("{}\n", encoding="utf-8")
            terminal = write_record(root / "terminal.json", pass_record(artifact))
            publication = root / "publication"
            publication.mkdir()
            (publication / "analysis_ready_manifest.csv").write_text(
                "orphan\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "partial immutable publication"):
                publish_terminal_ledger(
                    [terminal],
                    provenance_schema_path=SCHEMA,
                    ledger_schema_path=LEDGER_SCHEMA,
                    run_context_path=run_context,
                    expected_units=("S1_I1",),
                    required_matrices=MATRIX_OUTCOMES,
                    analysis_manifest_path=publication / "analysis_ready_manifest.csv",
                    outcome_manifest_path=publication / "outcome_validity_manifest.csv",
                    ledger_path=publication / "run_ledger.jsonl",
                    ledger_manifest_path=publication / "run_ledger_manifest.json",
                )

    def test_underlying_artifact_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"before")
            terminal = write_record(root / "terminal.json", pass_record(artifact))
            artifact.write_bytes(b"after")
            with self.assertRaisesRegex(ValueError, "artifact verification failed"):
                build_run_ledger_entries(
                    [terminal],
                    schema_path=SCHEMA,
                    required_matrices=MATRIX_OUTCOMES,
                    expected_units=("S1_I1",),
                )


if __name__ == "__main__":
    unittest.main()
