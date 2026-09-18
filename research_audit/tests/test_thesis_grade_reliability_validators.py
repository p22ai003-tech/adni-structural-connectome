from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


AUDIT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, AUDIT_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STABILITY = load_module(
    "phase_b_recipe_stability_validator", "validate_phase_b_recipe_stability.py"
)
HUMAN_QC = load_module(
    "human_qc_reliability_validator", "validate_human_qc_reliability.py"
)
RELEASE = load_module(
    "thesis_grade_release_evaluator", "evaluate_thesis_grade_530_release.py"
)
BUILDER = load_module(
    "phase_b_stability_manifest_builder",
    "../scforge/workflow/build_phase_b_stability_manifest.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


class PhaseBStabilityValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.nodes = 12
        self.contract = self.root / "contract.json"
        self.contract.write_text(
            json.dumps(
                {
                    "fixed_thresholds": {"matrix_shape": [self.nodes, self.nodes]},
                    "diagnosis_blind_recipe_stability_gate": {
                        "independent_seed_required": True,
                        "streamline_convergence_levels": [3_000_000, 5_000_000, 10_000_000],
                        "minimum_support_matched_upper_triangle_spearman": 0.95,
                        "minimum_tensor_matrix_upper_triangle_pearson": 0.98,
                        "maximum_global_metric_relative_difference": 0.05,
                        "minimum_node_strength_absolute_agreement_icc": 0.90,
                        "maximum_assignment_fraction_absolute_difference": 0.02,
                        "selection_may_use_diagnosis_or_group_effect": False,
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def build_manifest(self, *, corrupt_independent: bool = False) -> Path:
        rng = np.random.default_rng(42)
        raw = rng.uniform(0.2, 1.0, size=(self.nodes, self.nodes))
        base = np.triu(raw, 1)
        base = base + base.T
        tensor_bases = {
            "fa_mean": 0.25 + 0.4 * base,
            "md_mean": 0.0005 + 0.0004 * base,
            "rd_mean": 0.0003 + 0.0003 * base,
            "ad_mean": 0.0008 + 0.0005 * base,
        }
        for matrix in tensor_bases.values():
            np.fill_diagonal(matrix, 0.0)
        runs = []
        run_record_evidence = {}
        run_specs = (
            ("primary_3m", 3_000_000, "seed-A", "primary", 0.901),
            ("primary_5m", 5_000_000, "seed-A", "primary", 0.902),
            ("primary_10m", 10_000_000, "seed-A", "primary", 0.903),
            ("independent_10m", 10_000_000, "seed-B", "independent", 0.904),
        )
        for run_index, (run_id, streamlines, seed, seed_class, assignment) in enumerate(run_specs):
            perturb = np.ones_like(base)
            if run_id == "independent_10m":
                perturb += np.triu(rng.normal(0, 0.001, size=base.shape), 1)
                perturb = np.triu(perturb, 1)
                perturb = perturb + perturb.T
            if corrupt_independent and run_id == "independent_10m":
                perturb = np.triu(rng.uniform(0.01, 2.0, size=base.shape), 1)
                perturb = perturb + perturb.T
            scaled = base * perturb
            matrices = {
                "count": scaled * (streamlines / 3_000_000),
                "fd_sum": scaled * (streamlines / 3_000_000) * 0.7,
            }
            for name, tensor in tensor_bases.items():
                matrices[name] = tensor * (1.0 + 0.0001 * run_index)
            records = {}
            for name, matrix in matrices.items():
                path = self.root / f"{run_id}_{name}.csv"
                np.savetxt(path, matrix, delimiter=",", fmt="%.12g")
                records[name] = record(path)
            artifacts = {}
            for name, payload in {
                "tractogram": f"synthetic tractogram {run_id}\n",
                "sift2_weights": f"synthetic weights {run_id}\n",
                "assignments": f"synthetic assignments {run_id}\n",
                "tractography_metadata": json.dumps({"run_id": run_id}) + "\n",
            }.items():
                path = self.root / f"{run_id}_{name}.dat"
                path.write_text(payload, encoding="utf-8")
                artifacts[name] = record(path)
            assigned_count = int(round(assignment * streamlines))
            assignment = assigned_count / streamlines
            run = {
                "run_id": run_id,
                "streamline_count": streamlines,
                "tractogram_streamline_count": streamlines,
                "sift2_weight_count": streamlines,
                "assignment_row_count": streamlines,
                "assigned_streamline_count": assigned_count,
                "seed_id": seed,
                "seed_class": seed_class,
                "endpoint_assignment_fraction": assignment,
                "matrices": records,
                "artifacts": artifacts,
            }
            runs.append(run)
            run_record_path = self.root / f"{run_id}_run_record.json"
            run_record_path.write_text(json.dumps(run), encoding="utf-8")
            run_record_evidence[f"TEST_001_I1/{run_id}"] = record(run_record_path)
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "record_type": "diagnosis_blind_phase_b_recipe_stability_manifest",
                    "diagnosis_labels_used": False,
                    "builder_source": record(
                        AUDIT_ROOT.parent / "scforge/workflow/build_phase_b_stability_manifest.py"
                    ),
                    "run_record_evidence": run_record_evidence,
                    "units": [{"unit": "TEST_001_I1", "runs": runs}],
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_passes_complete_blinded_convergence_manifest(self):
        report = STABILITY.evaluate_manifest(self.build_manifest(), self.contract)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["passed_unit_count"], 1)
        self.assertEqual(report["unit_results"][0]["comparison_count"], 4)
        self.assertGreaterEqual(report["aggregate"]["minimum_node_strength_icc_a1"], 0.90)

    def test_fails_unstable_independent_seed(self):
        report = STABILITY.evaluate_manifest(
            self.build_manifest(corrupt_independent=True), self.contract
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(
            any("primary_10m_vs_independent_10m" in item for item in report["failures"])
        )

    def test_rejects_diagnosis_field(self):
        manifest = self.build_manifest()
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["units"][0]["diagnosis"] = "CN"
        manifest.write_text(json.dumps(data), encoding="utf-8")
        report = STABILITY.evaluate_manifest(manifest, self.contract)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("forbidden_blinding_field" in item for item in report["failures"]))

    def test_detects_matrix_hash_tampering(self):
        manifest = self.build_manifest()
        data = json.loads(manifest.read_text(encoding="utf-8"))
        path = Path(data["units"][0]["runs"][0]["matrices"]["count"]["path"])
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        report = STABILITY.evaluate_manifest(manifest, self.contract)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("size differs" in item for item in report["failures"]))


class PhaseBStabilityManifestBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def build_records(self, *, mismatch: bool = False, diagnosis: bool = False):
        paths = []
        counts = {
            "primary_3m": 3_000_000,
            "primary_5m": 5_000_000,
            "primary_10m": 10_000_000,
            "independent_10m": 10_000_000,
        }
        for run_id, count in counts.items():
            matrices = {}
            for name in BUILDER.MATRIX_NAMES:
                path = self.root / f"{run_id}_{name}.csv"
                path.write_text("0,1\n1,0\n", encoding="utf-8")
                matrices[name] = str(path)
            artifacts = {}
            for name in BUILDER.ARTIFACT_NAMES:
                path = self.root / f"{run_id}_{name}.dat"
                path.write_text(f"{run_id} {name}\n", encoding="utf-8")
                artifacts[name] = str(path)
            assigned = count - 10
            raw = {
                "schema_version": "1.0.0",
                "record_type": "diagnosis_blind_phase_b_stability_run",
                "diagnosis_labels_used": False,
                "unit": "TEST_001_I1",
                "run_id": run_id,
                "seed_id": "seed-B" if run_id == "independent_10m" else "seed-A",
                "seed_class": "independent" if run_id == "independent_10m" else "primary",
                "streamline_count": count,
                "tractogram_streamline_count": count,
                "sift2_weight_count": count - 1 if mismatch and run_id == "primary_5m" else count,
                "assignment_row_count": count,
                "assigned_streamline_count": assigned,
                "endpoint_assignment_fraction": assigned / count,
                "matrices": matrices,
                "artifacts": artifacts,
            }
            if diagnosis and run_id == "primary_3m":
                raw["diagnosis"] = "CN"
            path = self.root / f"{run_id}_record.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            paths.append(path)
        return paths

    def test_builds_exact_four_run_hash_bound_manifest(self):
        manifest = BUILDER.build_manifest(self.build_records())
        self.assertEqual(manifest["record_type"], "diagnosis_blind_phase_b_recipe_stability_manifest")
        self.assertFalse(manifest["diagnosis_labels_used"])
        self.assertEqual(len(manifest["units"]), 1)
        self.assertEqual(len(manifest["units"][0]["runs"]), 4)
        self.assertEqual(len(manifest["run_record_evidence"]), 4)
        self.assertEqual(manifest["builder_source"], record(Path(BUILDER.__file__)))

    def test_rejects_observed_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "counts differ"):
            BUILDER.build_manifest(self.build_records(mismatch=True))

    def test_rejects_diagnosis_leakage(self):
        with self.assertRaisesRegex(ValueError, "forbidden fields"):
            BUILDER.build_manifest(self.build_records(diagnosis=True))


class HumanQCReliabilityValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.units = [f"U{index:02d}" for index in range(20)]
        self.contract = self.root / "contract.json"
        self.contract.write_text(
            json.dumps({"human_qc_contract": {"minimum_interrater_kappa": 0.80}}),
            encoding="utf-8",
        )
        self.scope = self.root / "scope.json"
        self.scope.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "record_type": "diagnosis_blind_human_qc_scope",
                    "diagnosis_labels_used": False,
                    "audit_salt": "locked-test-salt",
                    "expected_units": self.units,
                    "automated_warning_units": ["U00", "U01"],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def build_reviews(self, *, disagreement: bool = False, omit_secondary: bool = False) -> Path:
        audit = HUMAN_QC.deterministic_audit_sample(self.units, "locked-test-salt")
        required = sorted(set(audit) | {"U00", "U01"})
        rows = []
        fail_unit = required[0]
        for unit in self.units:
            primary_status = "FAIL" if unit == fail_unit else "PASS"
            rows.append(
                {
                    "unit": unit,
                    "role": "PRIMARY",
                    "status": primary_status,
                    "reviewer": "rater-A",
                    "reviewed_utc": "2026-07-19T18:00:00+00:00",
                }
            )
            if unit in required and not (omit_secondary and unit == required[-1]):
                secondary_status = primary_status
                if disagreement and unit == required[-1]:
                    secondary_status = "FAIL" if primary_status == "PASS" else "PASS"
                rows.append(
                    {
                        "unit": unit,
                        "role": "SECONDARY",
                        "status": secondary_status,
                        "reviewer": "rater-B",
                        "reviewed_utc": "2026-07-19T18:10:00+00:00",
                    }
                )
                if secondary_status != primary_status:
                    rows.append(
                        {
                            "unit": unit,
                            "role": "ADJUDICATOR",
                            "status": primary_status,
                            "reviewer": "rater-C",
                            "reviewed_utc": "2026-07-19T18:20:00+00:00",
                        }
                    )
        path = self.root / "reviews.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("unit", "role", "status", "reviewer", "reviewed_utc"),
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_passes_complete_blinded_review_with_perfect_estimable_kappa(self):
        report = HUMAN_QC.evaluate_reviews(
            self.scope, self.build_reviews(), self.contract
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["cohen_kappa"]["value"], 1.0)
        self.assertEqual(report["primary_review_count"], 20)
        self.assertGreaterEqual(report["double_rated_unit_count"], 4)

    def test_fails_missing_required_second_rating(self):
        report = HUMAN_QC.evaluate_reviews(
            self.scope, self.build_reviews(omit_secondary=True), self.contract
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(
            any("required_second_rating_missing" in item for item in report["failures"])
        )

    def test_fails_low_kappa_even_when_disagreement_is_adjudicated(self):
        report = HUMAN_QC.evaluate_reviews(
            self.scope, self.build_reviews(disagreement=True), self.contract
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["disagreement_count"], 1)
        self.assertTrue(any("kappa" in item for item in report["failures"]))

    def test_rejects_group_column(self):
        reviews = self.build_reviews()
        with reviews.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with reviews.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("unit", "role", "status", "reviewer", "reviewed_utc", "group"),
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "group": "CN"})
        report = HUMAN_QC.evaluate_reviews(self.scope, reviews, self.contract)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("forbidden_blinding_field" in item for item in report["failures"]))


class ReleaseEvaluatorPublicationGateTests(unittest.TestCase):
    def test_status_only_json_cannot_forge_stability_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root / "contract.json"
            contract = {
                "fixed_thresholds": {"matrix_shape": [166, 166]},
                "diagnosis_blind_recipe_stability_gate": {
                    "minimum_support_matched_upper_triangle_spearman": 0.95,
                    "minimum_tensor_matrix_upper_triangle_pearson": 0.98,
                    "maximum_global_metric_relative_difference": 0.05,
                    "minimum_node_strength_absolute_agreement_icc": 0.90,
                    "maximum_assignment_fraction_absolute_difference": 0.02,
                },
                "human_qc_contract": {"minimum_interrater_kappa": 0.80},
            }
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            forged = root / "recipe_stability_validation.json"
            forged.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
            passed, failures = RELEASE.strict_validation_record(
                forged,
                kind="stability",
                contract_path=contract_path,
                contract=contract,
                full_hash=True,
            )
            self.assertFalse(passed)
            self.assertIn("record_type_differs", failures)
            self.assertIn("checks_are_not_all_true", failures)


if __name__ == "__main__":
    unittest.main()
