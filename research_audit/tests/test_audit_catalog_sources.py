from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from research_audit import audit_catalog_sources as catalog_audit
from research_audit.audit_catalog_sources import (
    AWS_CATALOG_SEARCH,
    AWS_PRINCIPAL_VERIFICATION,
    REQUEST_ROSTER,
    REQUEST_ROSTER_VALIDATION,
    format_authenticated_aws_evidence,
    load_authenticated_aws_evidence,
    load_frozen_request_roster,
    summarize_request_coverage,
)


class CatalogSourceAuditTests(unittest.TestCase):
    def test_frozen_request_roster_is_checksum_and_source_bound(self) -> None:
        roster, validation = load_frozen_request_roster()
        self.assertEqual(len(roster), 4015)
        self.assertEqual(roster["t1_image_id"].nunique(), 4015)
        self.assertEqual(roster["subject_id"].nunique(), 530)
        self.assertEqual(
            int(roster["is_provisional_replacement_candidate"].sum()), 330
        )
        self.assertEqual(validation["status"], "PASS")

    def test_tampered_request_roster_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roster_path = root / REQUEST_ROSTER.name
            validation_path = root / REQUEST_ROSTER_VALIDATION.name
            roster_path.write_bytes(REQUEST_ROSTER.read_bytes() + b"\n")
            validation_path.write_bytes(REQUEST_ROSTER_VALIDATION.read_bytes())
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                load_frozen_request_roster(
                    roster_path=roster_path,
                    validation_path=validation_path,
                )

    def test_authenticated_aws_evidence_is_checksum_bound_and_not_stale(self) -> None:
        evidence = load_authenticated_aws_evidence()
        rendered = format_authenticated_aws_evidence(evidence)
        self.assertIn("recorded PASS", rendered)
        self.assertIn("does not reauthenticate", rendered)
        self.assertNotIn("expired", rendered.lower())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            principal_path = root / AWS_PRINCIPAL_VERIFICATION.name
            search_path = root / AWS_CATALOG_SEARCH.name
            principal_path.write_bytes(AWS_PRINCIPAL_VERIFICATION.read_bytes() + b"\n")
            search_path.write_bytes(AWS_CATALOG_SEARCH.read_bytes())
            with self.assertRaisesRegex(RuntimeError, "not bound"):
                load_authenticated_aws_evidence(principal_path, search_path)

    def test_full_and_nested_request_coverage_are_separate(self) -> None:
        roster = pd.DataFrame(
            {
                "subject_id": ["001_S_0001", "001_S_0001", "001_S_0002"],
                "t1_image_id": [11, 12, 21],
                "is_current_t1_image": [True, False, False],
                "is_rounded_age_screen_candidate": [False, True, True],
                "is_provisional_replacement_candidate": [False, True, False],
            }
        )
        summary = summarize_request_coverage(roster, {11, 21}, {12})
        self.assertEqual(summary["request_roster_rows"], 3)
        self.assertEqual(summary["request_roster_subjects"], 2)
        self.assertEqual(summary["request_roster_provisional_replacement_ids"], 1)
        self.assertEqual(summary["request_roster_ids_in_mri_master"], 2)
        self.assertEqual(summary["request_roster_subjects_in_mri_master"], 2)
        self.assertEqual(summary["request_roster_ids_in_local_raw_mri"], 1)

    def test_main_renders_current_report_without_touching_project_docs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit_root = Path(directory)
            output_root = audit_root / "outputs"
            with (
                mock.patch.object(catalog_audit, "AUDIT", audit_root),
                mock.patch.object(catalog_audit, "OUTPUT", output_root),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                catalog_audit.main()
            report = (audit_root / "catalog_source_inventory.md").read_text(
                encoding="utf-8"
            )
            coverage = pd.read_csv(output_root / "catalog_coverage_v2.csv")
            metrics = dict(zip(coverage["measure"], coverage["value"]))
            self.assertEqual(metrics["request_roster_rows"], 4015)
            self.assertEqual(metrics["request_roster_subjects"], 530)
            self.assertEqual(metrics["request_roster_provisional_replacement_ids"], 330)
            self.assertIn("330 is not the complete export universe", report)
            self.assertIn(
                "resolve every frozen Image ID with an exact date and recognized QC",
                report,
            )
            self.assertIn("not evidence of current IDA completeness", report)
            self.assertNotIn("explicit source-side absence reason", report)
            self.assertIn("recorded PASS", report)
            self.assertNotIn("profile remained expired", report)


if __name__ == "__main__":
    unittest.main()
