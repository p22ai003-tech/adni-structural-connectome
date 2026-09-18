import csv
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path("/home/ec2-user/exp")
SCRIPT = ROOT / "research_audit/build_h04a_canary_package.py"
OUTPUT = ROOT / "research_audit/outputs/h04a_canary_package_v1"
PARENT = ROOT / "research_audit/outputs/connectome_v2_input_manifest_v2.csv"


def load_module():
    spec = importlib.util.spec_from_file_location("build_h04a_canary_package", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class H04ACanaryPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()
        cls.subset = read_csv(OUTPUT / "candidate_canary_manifest_v1.csv")
        cls.parent = read_csv(PARENT)
        cls.validation = json.loads(
            (OUTPUT / "release_validation.json").read_text(encoding="utf-8")
        )
        cls.decision = json.loads(
            (OUTPUT / "h04a_decision_draft_v1.json").read_text(encoding="utf-8")
        )

    def test_selection_field_allowlist_is_diagnosis_and_outcome_blind(self):
        self.assertFalse(
            self.module.SELECTION_SOURCE_FIELDS
            & self.module.FORBIDDEN_SELECTION_FIELDS
        )
        self.assertNotIn("diagnosis_at_dti", self.module.SELECTION_SOURCE_FIELDS)
        self.assertNotIn("legacy_connectome_density", self.module.SELECTION_SOURCE_FIELDS)

    def test_exact_24_parent_rows(self):
        parent_by_unit = {
            self.module.stable_unit(row): row for row in self.parent
        }
        self.assertEqual(len(self.subset), 24)
        self.assertEqual(
            len({self.module.stable_unit(row) for row in self.subset}), 24
        )
        for row in self.subset:
            self.assertEqual(row, parent_by_unit[self.module.stable_unit(row)])

    def test_release_validation_passes_without_authorization(self):
        self.assertEqual(self.validation["status"], "PASS")
        self.assertFalse(self.validation["execution_authorized"])
        self.assertTrue(all(self.validation["checks"].values()))

    def test_decision_draft_cannot_authorize_launcher(self):
        self.assertEqual(self.decision["status"], "PENDING_USER_DECISION")
        self.assertIsNone(self.decision["approved_by"])
        self.assertIsNone(self.decision["approved_utc"])
        self.assertIsNone(self.decision["user_response"])
        self.assertEqual(
            self.decision["authorized_modes"],
            [
                "response-calibration-phase-a",
                "pre-tractography-canary",
            ],
        )
        self.assertNotIn("phase-b", self.decision["authorized_modes"])
        self.assertEqual(
            self.decision["authorized_through"],
            "pre_tractography_review_bundle_only",
        )
        self.assertFalse(self.decision["tractography_authorized"])
        self.assertFalse(self.decision["matrix_generation_authorized"])
        self.assertFalse(self.decision["full_cohort_authorized"])
        self.assertEqual(self.decision["maximum_cores"], 4)
        self.assertEqual(
            self.decision["minimum_valid_response_calibration_units"], 12
        )
        self.assertEqual(
            self.decision["minimum_valid_manufacturer_families"], 2
        )
        self.assertEqual(self.decision["minimum_valid_t1_source_classes"], 2)
        self.assertEqual(
            self.decision["required_valid_t1_source_classes"],
            ["dicom_series", "nifti_single"],
        )
        self.assertEqual(self.decision["h04a_wall_clock_stop_hours"], 72)
        self.assertEqual(self.decision["h04a_storage_stop_gb"], 150)
        self.assertEqual(
            self.decision["proposed_run_root"],
            str(self.module.PROPOSED_RUN_ROOT),
        )
        self.assertEqual(
            self.decision["normative_config_sha256"],
            self.module.sha256_file(self.module.RECIPE_CONFIG),
        )
        self.assertEqual(
            self.decision["workflow_source_manifest_sha256"],
            self.module.sha256_file(self.module.WORKFLOW_SOURCE_MANIFEST),
        )
        self.assertEqual(
            self.decision["environment_contract_sha256"],
            self.module.sha256_file(self.module.ENVIRONMENT_CONTRACT),
        )

    def test_selection_is_deterministic(self):
        candidates, _, _ = self.module.build_candidates()
        first = sorted(item["unit"] for item in self.module.select_canary(candidates))
        second = sorted(item["unit"] for item in self.module.select_canary(candidates))
        observed = sorted(self.module.stable_unit(row) for row in self.subset)
        self.assertEqual(first, second)
        self.assertEqual(first, observed)


if __name__ == "__main__":
    unittest.main()
