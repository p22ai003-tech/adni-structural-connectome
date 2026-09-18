import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research_audit"))

from coupling_support_falsification import (  # noqa: E402
    partial_spearman,
    raw_spearman,
)


class CouplingSupportFalsificationTests(unittest.TestCase):
    def test_raw_spearman_uses_pairwise_finite_nodes(self):
        frame = pd.DataFrame(
            {"x": [1, 2, 3, 4, 5, np.nan], "y": [2, 4, 6, 8, 10, 4]}
        )
        self.assertAlmostEqual(raw_spearman(frame, "x", "y"), 1.0)

    def test_partial_spearman_removes_shared_support_trend(self):
        rng = np.random.default_rng(20260718)
        support = np.arange(1, 81, dtype=float)
        frame = pd.DataFrame(
            {
                "support": support,
                "x": support + rng.normal(scale=5.0, size=len(support)),
                "y": support + rng.normal(scale=5.0, size=len(support)),
            }
        )
        raw = spearmanr(frame["x"], frame["y"]).statistic
        partial = partial_spearman(frame, "x", "y", ["support"])
        self.assertGreater(raw, 0.85)
        self.assertLess(abs(partial), 0.35)

    def test_partial_spearman_retains_independent_residual_signal(self):
        rng = np.random.default_rng(7)
        support = np.arange(1, 101, dtype=float)
        residual = rng.normal(size=len(support))
        frame = pd.DataFrame(
            {
                "support": support,
                "x": support + 4 * residual,
                "y": support + 4 * residual + rng.normal(scale=0.3, size=len(support)),
            }
        )
        partial = partial_spearman(frame, "x", "y", ["support"])
        self.assertGreater(partial, 0.8)


if __name__ == "__main__":
    unittest.main()
