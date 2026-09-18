import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research_audit"))

from stage_specific_diffusivity_analysis import (  # noqa: E402
    ALL_NETWORKS,
    CROSS_MAPPING_DUPLICATES,
    PRIMARY_NETWORKS,
    _group_vector,
    _support_frame,
    adjust,
    fit_group_contrast,
)


class StageSpecificDiffusivityAnalysisTests(unittest.TestCase):
    def test_primary_inventory_removes_only_cross_mapping_duplicates(self):
        self.assertEqual(len(ALL_NETWORKS), 19)
        self.assertEqual(len(PRIMARY_NETWORKS), 17)
        self.assertEqual(
            set(ALL_NETWORKS) - set(PRIMARY_NETWORKS),
            set(CROSS_MAPPING_DUPLICATES),
        )
        self.assertTrue(
            set(CROSS_MAPPING_DUPLICATES.values()).issubset(PRIMARY_NETWORKS)
        )

    def test_holm_adjustment_bounds_and_orders(self):
        adjusted = adjust([0.001, 0.02], "holm")
        self.assertTrue(np.all((adjusted >= 0) & (adjusted <= 1)))
        self.assertLessEqual(adjusted[0], adjusted[1])
        self.assertAlmostEqual(float(adjusted[0]), 0.002)

    def test_group_vector_encodes_adjacent_and_second_difference(self):
        names = [
            "Intercept",
            'C(group, Treatment(reference="CN"))[T.MCI]',
            'C(group, Treatment(reference="CN"))[T.AD]',
        ]
        adjacent = _group_vector(names, "CN", {"AD": 1.0, "MCI": -1.0})
        second = _group_vector(
            names, "CN", {"AD": 1.0, "MCI": -2.0, "CN": 1.0}
        )
        np.testing.assert_array_equal(adjacent, [0.0, -1.0, 1.0])
        np.testing.assert_array_equal(second, [0.0, -2.0, 1.0])

    def test_pair_common_support_removes_single_group_sites(self):
        frame = pd.DataFrame(
            {
                "site": ["a", "a", "b", "b", "c"],
                "group": ["CN", "MCI", "CN", "CN", "MCI"],
            }
        )
        selected = _support_frame(
            frame, "pair_common_sites", left="MCI", right="CN"
        )
        self.assertEqual(set(selected["site"]), {"a"})
        self.assertEqual(set(selected["group"]), {"CN", "MCI"})

    def test_site_fixed_model_recovers_positive_adjacent_contrast(self):
        rng = np.random.default_rng(20260718)
        rows = []
        for site_index in range(12):
            site_effect = rng.normal(scale=0.2)
            for group_index, group in enumerate(("CN", "MCI", "AD")):
                for subject_index in range(5):
                    rows.append(
                        {
                            "group": group,
                            "site": f"s{site_index:02d}",
                            "age": 70 + rng.normal(),
                            "sex": "F" if subject_index % 2 else "M",
                            "log_gap_days": np.log1p(30 + subject_index),
                            "t1_source": "Original" if site_index % 2 else "Processed",
                            "outcome": group_index * 0.7
                            + site_effect
                            + rng.normal(scale=0.15),
                        }
                    )
        result = fit_group_contrast(
            pd.DataFrame(rows),
            "outcome",
            {"MCI": 1.0, "CN": -1.0},
            reference="CN",
            scope="full_site_fe",
        )
        self.assertGreater(result["beta"], 0)
        self.assertLess(result["p_value"], 0.05)
        self.assertLessEqual(result["design_rank"], result["design_columns"])


if __name__ == "__main__":
    unittest.main()
