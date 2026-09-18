from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_audit.run_fixed_data_feasibility import (
    DEFAULT_DECISION,
    DEFAULT_DIAGNOSIS,
    DEFAULT_HISTORICAL_PROXY_VALIDATION,
    DEFAULT_MANIFEST,
    DEFAULT_MANIFEST_VALIDATION,
    DEFAULT_WORKFLOW_CONFIG,
    FIG_POWER,
    FIG_SUPPORT,
    OUTPUT_ESTIMABILITY,
    OUTPUT_FEASIBILITY,
    OUTPUT_MEMO,
    OUTPUT_VALIDATION,
    OUTPUT_WORKFLOW_CONTRACT,
    common_support,
    design_diagnostics,
    naive_mde,
    population_masks,
    population_summary,
    run_analysis,
    validate_inputs,
    load_historical_proxy_validation,
    workflow_contract_feasibility,
)


class FixedDataFeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame, cls.evidence = validate_inputs(
            DEFAULT_MANIFEST,
            DEFAULT_MANIFEST_VALIDATION,
            DEFAULT_DIAGNOSIS,
            DEFAULT_DECISION,
        )
        cls.masks = population_masks(cls.frame)

    def test_canonical_populations_and_exact_common_support(self) -> None:
        populations = population_summary(self.frame, self.masks).set_index("population")
        self.assertEqual(int(populations.loc["all_available", "n_total"]), 530)
        self.assertEqual(int(populations.loc["primary_full", "n_total"]), 515)
        self.assertEqual(int(populations.loc["timing_le_90", "n_total"]), 194)
        self.assertEqual(int(populations.loc["timing_le_180", "n_total"]), 197)
        self.assertEqual(
            int(populations.loc["protocol_siemens_3t_54", "n_total"]), 260
        )
        self.assertEqual(
            int(
                populations.loc[
                    "timing_le90_siemens_3t_54_sites_ge2groups", "n_total"
                ]
            ),
            75,
        )
        self.assertEqual(
            int(populations.loc["timing_le90_siemens_3t_54", "gap_days_unique_values"]),
            1,
        )

        support = common_support(self.frame, self.masks)
        exact = support[
            (support["population"] == "primary_full")
            & (support["cell_family"] == "timing_site_protocol_t1_type")
            & (support["support_scope"] == "ALL3")
        ].iloc[0]
        self.assertEqual(int(exact["levels_total"]), 131)
        self.assertEqual(int(exact["levels_min_1"]), 13)
        self.assertEqual(int(exact["levels_min_5"]), 0)
        self.assertEqual(int(exact["n_cn_min_1"]), 50)
        self.assertEqual(int(exact["n_mci_min_1"]), 44)
        self.assertEqual(int(exact["n_ad_min_1"]), 18)

    def test_parsimonious_design_is_full_rank_and_cubic_design_is_not(self) -> None:
        diagnostics, lookup = design_diagnostics(self.frame, self.masks)
        full = diagnostics[diagnostics["population"] == "primary_full"].set_index(
            "specification"
        )
        self.assertTrue(bool(full.loc["parsimonious_protocol", "full_column_rank"]))
        self.assertEqual(
            int(full.loc["parsimonious_protocol", "rank"]),
            int(full.loc["parsimonious_protocol", "columns"]),
        )
        self.assertFalse(bool(full.loc["cubic_phase_protocol", "full_column_rank"]))
        self.assertEqual(
            int(full.loc["cubic_phase_protocol", "rank_deficiency"]), 1
        )
        aliases = full.loc[
            "cubic_phase_protocol", "exact_duplicate_columns_json"
        ]
        self.assertIn("phase[ADNI 2]", aliases)
        self.assertIn("protocol[GE MEDICAL SYSTEMS|3T|41dir]", aliases)
        for contrast in ("CN_vs_MCI", "MCI_vs_AD", "CN_vs_AD"):
            self.assertTrue(
                bool(
                    lookup[("primary_full", "parsimonious_protocol", contrast)][
                        "algebraically_estimable"
                    ]
                )
            )

    def test_workflow_contract_detects_required_policy_conflicts(self) -> None:
        contract, evidence = workflow_contract_feasibility(
            self.frame,
            DEFAULT_WORKFLOW_CONFIG,
            load_historical_proxy_validation(
                DEFAULT_HISTORICAL_PROXY_VALIDATION,
                self.evidence["manifest"]["sha256"],
            ),
        )
        self.assertFalse(evidence["configured_manifest_exists"])
        policies = contract[contract["record_type"] == "policy"].set_index(
            "required_field_or_policy"
        )
        for policy in (
            "pairing.primary_max_abs_days=90",
            "pairing.sensitivity_max_abs_days=180",
            "pairing.require_original_t1=true",
            "workflow hard-rejects SMC",
        ):
            self.assertEqual(policies.loc[policy, "classification"], "policy_conflicting")
            self.assertTrue(bool(policies.loc[policy, "blocking_before_canary"]))
        fields = contract[contract["record_type"] == "required_field"].set_index(
            "required_field_or_policy"
        )
        self.assertEqual(
            fields.loc["total_readout_time", "classification"],
            "derivable_from_local_headers",
        )
        self.assertEqual(
            fields.loc["source_file_sha256", "classification"],
            "derivable_from_local_files",
        )
        self.assertEqual(fields.loc["t1_series_uid", "classification"], "unavailable")
        self.assertIn(
            "t1_source_id", fields.loc["t1_series_uid", "required_action"]
        )
        self.assertIn(
            "225/225", fields.loc["t1_series_uid", "evidence"]
        )

    def test_mde_decreases_with_sample_size(self) -> None:
        small = float(naive_mde(np.array([20.0]), np.array([20.0]))[0])
        large = float(naive_mde(np.array([100.0]), np.array([100.0]))[0])
        self.assertGreater(small, large)
        self.assertTrue(np.isnan(naive_mde(np.array([1.0]), np.array([20.0]))[0]))

    def test_checksum_gate_and_write_once_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tampered = root / "diagnosis.csv"
            shutil.copy2(DEFAULT_DIAGNOSIS, tampered)
            tampered.write_bytes(tampered.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "diagnosis checksum"):
                validate_inputs(
                    DEFAULT_MANIFEST,
                    DEFAULT_MANIFEST_VALIDATION,
                    tampered,
                    DEFAULT_DECISION,
                )

            output = root / "outputs"
            figures = root / "figures"
            result = run_analysis(
                DEFAULT_MANIFEST,
                DEFAULT_MANIFEST_VALIDATION,
                DEFAULT_DIAGNOSIS,
                DEFAULT_DECISION,
                DEFAULT_WORKFLOW_CONFIG,
                DEFAULT_HISTORICAL_PROXY_VALIDATION,
                output,
                figures,
                generated_utc="2026-07-18T10:00:00+00:00",
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["safety"]["core_run_image_files_read"], 0)
            self.assertEqual(result["safety"]["upstream_historical_mif_headers_read"], 530)
            self.assertEqual(result["safety"]["upstream_processed_t1_headers_read"], 225)
            self.assertEqual(result["safety"]["upstream_voxel_arrays_read"], 0)
            for name in (
                OUTPUT_FEASIBILITY,
                OUTPUT_ESTIMABILITY,
                OUTPUT_WORKFLOW_CONTRACT,
                OUTPUT_MEMO,
                OUTPUT_VALIDATION,
            ):
                self.assertTrue((output / name).is_file())
            for name in (FIG_SUPPORT, FIG_POWER):
                self.assertTrue((figures / name).is_file())
            with self.assertRaises(FileExistsError):
                run_analysis(
                    DEFAULT_MANIFEST,
                    DEFAULT_MANIFEST_VALIDATION,
                    DEFAULT_DIAGNOSIS,
                    DEFAULT_DECISION,
                    DEFAULT_WORKFLOW_CONFIG,
                    DEFAULT_HISTORICAL_PROXY_VALIDATION,
                    output,
                    figures,
                    generated_utc="2026-07-18T10:00:00+00:00",
                )


if __name__ == "__main__":
    unittest.main()
