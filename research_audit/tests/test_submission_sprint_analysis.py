import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research_audit"))

from submission_sprint_analysis import (  # noqa: E402
    GROUPS,
    Stratum,
    adjust_p,
    cliffs_delta,
    fit_feature_contrasts,
    markdown_table,
    normal_scores,
    protocol_bucket,
)


class SubmissionSprintAnalysisTests(unittest.TestCase):
    def test_protocol_bucket_is_deterministic_and_collapsed(self):
        self.assertEqual(protocol_bucket("SIEMENS", 54), "SIEMENS_54")
        self.assertEqual(protocol_bucket("Siemens", 30.0), "SIEMENS_30")
        self.assertEqual(protocol_bucket("GE MEDICAL SYSTEMS", 41), "GE_41")
        self.assertEqual(protocol_bucket("Philips Medical Systems", 0), "PHILIPS")
        self.assertEqual(protocol_bucket(np.nan, np.nan), "OTHER_OR_MISSING")

    def test_adjust_p_preserves_nan_and_bounds(self):
        result = adjust_p([0.01, np.nan, 0.04], "holm")
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isfinite(result[[0, 2]]).all())
        self.assertTrue(((result[[0, 2]] >= 0) & (result[[0, 2]] <= 1)).all())

    def test_normal_scores_are_ordered_and_centered(self):
        scores = normal_scores(pd.Series([4.0, 1.0, 3.0, 2.0]))
        self.assertAlmostEqual(float(scores.mean()), 0.0, places=12)
        order = np.argsort([4.0, 1.0, 3.0, 2.0])
        self.assertTrue(np.all(np.diff(scores[order]) > 0))

    def test_cliffs_delta_has_expected_direction(self):
        self.assertAlmostEqual(cliffs_delta([3, 4, 5], [0, 1, 2]), 1.0)
        self.assertAlmostEqual(cliffs_delta([0, 1, 2], [3, 4, 5]), -1.0)

    def test_markdown_table_has_no_optional_dependency(self):
        table = markdown_table(
            pd.DataFrame({"label": ["a|b"], "value": [0.125]}),
            ["label", "value"],
        )
        self.assertIn("| label | value |", table)
        self.assertIn("a\\|b", table)
        self.assertIn("0.125", table)

    def test_model_recovers_ordered_group_direction_and_three_contrasts(self):
        rng = np.random.default_rng(20260718)
        rows = []
        for group_index, group in enumerate(GROUPS):
            for index in range(35):
                rows.append(
                    {
                        "subject_id": f"{group}_{index:03d}",
                        "group": group,
                        "age": 70 + rng.normal(),
                        "sex": "F" if index % 2 else "M",
                        "site": f"{index % 10:03d}",
                        "log_gap_days": np.log1p(20 + index),
                        "t1_source": "Original" if index % 3 else "Processed",
                        "protocol_bucket": "SIEMENS_54" if index % 2 else "SIEMENS_30",
                        "value": group_index + rng.normal(scale=0.2),
                        "feature_id": "synthetic::feature",
                        "mapping": "synthetic",
                        "feature_family": "synthetic",
                        "network": "synthetic",
                        "metric": "synthetic",
                    }
                )
        frame = pd.DataFrame(rows)
        stratum = Stratum("synthetic", frozenset(frame["subject_id"]))
        results = pd.DataFrame(fit_feature_contrasts(frame, stratum))
        self.assertEqual(set(results["contrast"]), {"MCI_vs_CN", "AD_vs_CN", "AD_vs_MCI"})
        self.assertTrue((results["rank_normal_beta"] > 0).all())
        self.assertTrue((results["ci95_low"] < results["ci95_high"]).all())
        self.assertTrue(results["p_value"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
