from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_vfinal_confirmation_executor.py"
SPEC = importlib.util.spec_from_file_location("vfinal_confirmation_executor", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class VFinalConfirmationExecutorTests(unittest.TestCase):
    @classmethod
    def synthetic_inputs(cls):
        rng = np.random.default_rng(20260719)
        cohort_rows = []
        feature_rows = []
        groups = ("CN", "MCI", "AD")
        # Four participants per group at each of 15 sites meets both locked
        # support thresholds while keeping the test compact.
        for site_index in range(15):
            for group_index, group in enumerate(groups):
                for replicate in range(4):
                    subject_id = f"S{site_index:02d}_{group}_{replicate}"
                    cohort_rows.append(
                        {
                            "subject_id": subject_id,
                            "group": group,
                            "age": 69.0 + group_index * 2.0 + rng.normal(0, 2),
                            "sex": "F" if replicate % 2 else "M",
                            "site": f"SITE{site_index:02d}",
                            "gap_days": float(10 + replicate),
                            "t1_source": "Original",
                            "qc_pass": True,
                            "exclusion_reason": "",
                        }
                    )
                    # Synthetic signals are used only to exercise the model:
                    # MCI has higher AxD than CN and AD has higher RD than MCI.
                    axd_shift = 0.00006 * (group_index >= 1)
                    rd_shift = 0.00008 * (group_index >= 2)
                    for system_index, system in enumerate(MODULE.EXPECTED_SYSTEMS):
                        feature_rows.append(
                            {
                                "subject_id": subject_id,
                                "system_id": system,
                                "ad_mean": 0.0010 + axd_shift + system_index * 1e-7 + rng.normal(0, 5e-6),
                                "rd_mean": 0.0007 + rd_shift + system_index * 1e-7 + rng.normal(0, 5e-6),
                            }
                        )
        return pd.DataFrame(cohort_rows), pd.DataFrame(feature_rows)

    def test_frozen_contract_loads(self):
        contract = MODULE.load_contract(MODULE.DEFAULT_CONTRACT)
        self.assertEqual(contract["contract_id"], MODULE.EXPECTED_CONTRACT_ID)

    def test_composites_use_all_seventeen_systems(self):
        cohort, features = self.synthetic_inputs()
        composite, coverage = MODULE.build_composites(cohort, features)
        self.assertEqual(len(composite), len(cohort))
        self.assertTrue(composite["n_paired_systems"].eq(17).all())
        self.assertEqual(len(coverage), 17 * 3)

    def test_locked_endpoint_engine_and_classification(self):
        cohort, features = self.synthetic_inputs()
        composite, _coverage = MODULE.build_composites(cohort, features)
        contract = copy.deepcopy(MODULE.load_contract(MODULE.DEFAULT_CONTRACT))
        contract["primary_support_and_model"]["bootstrap_iterations"] = 39
        rows = [MODULE.fit_endpoint(composite, endpoint, contract) for endpoint in MODULE.ENDPOINTS]
        results = pd.DataFrame(rows)
        results["holm_wild_p"] = results["restricted_wild_cluster_bootstrap_t_p"]
        self.assertTrue(results["positive_direction"].all())
        self.assertTrue(results["n_sites"].eq(15).all())
        self.assertTrue(results["wild_bootstrap_valid_iterations"].eq(39).all())
        self.assertIn(
            MODULE.classify(results),
            {
                "PROCESSING_CONFIRMED_INTERNAL",
                "PARTIAL_REPLICATION_QUALIFY_CLAIM",
                "DIRECTIONAL_ONLY_NOT_CONFIRMED",
            },
        )

    def test_direction_reversal_cannot_be_partial_replication(self):
        results = pd.DataFrame(
            {
                "positive_direction": [True, False],
                "holm_wild_p": [0.001, 0.001],
            }
        )
        self.assertEqual(
            MODULE.classify(results),
            "NOT_REPLICATED_UNDER_CORRECTED_PROCESSING",
        )


if __name__ == "__main__":
    unittest.main()
