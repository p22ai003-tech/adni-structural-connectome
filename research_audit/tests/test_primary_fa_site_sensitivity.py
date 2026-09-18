import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research_audit"))

from primary_fa_site_sensitivity import (  # noqa: E402
    SCOPES,
    _scope_frame,
    _support_sites,
    configuration_sha256,
    fit_primary_fa,
)
from submission_sprint_analysis import normal_scores  # noqa: E402


GROUPS = ("CN", "MCI", "AD")


def synthetic_analysis() -> pd.DataFrame:
    rows = []
    subject = 0
    for site_index in range(12):
        site = f"{site_index:03d}"
        site_effect = 2.0 if site_index < 6 else -2.0
        counts = (
            {"CN": 2, "MCI": 2, "AD": 10}
            if site_index < 6
            else {"CN": 10, "MCI": 2, "AD": 2}
        )
        for group in GROUPS:
            for within in range(counts[group]):
                subject += 1
                rows.append(
                    {
                        "subject_id": f"S{subject:04d}",
                        "group": group,
                        "site": site,
                        "fa_value": site_effect + ((within % 3) - 1) * 0.02,
                        "age": 65.0 + (within % 5),
                        "sex": "F" if within % 2 else "M",
                        "log_gap_days": np.log1p(20 + within),
                        "t1_source": (
                            "Original"
                            if (within + site_index) % 3
                            else "Processed"
                        ),
                        "protocol_bucket": (
                            "P1" if site_index < 6 else "P2"
                        ),
                    }
                )
    return pd.DataFrame(rows)


class PrimaryFASiteSensitivityTests(unittest.TestCase):
    def test_support_sites_are_exact(self):
        frame = pd.DataFrame(
            {
                "site": ["001"] * 3 + ["002"] * 2 + ["003"] * 2 + ["004"] * 2,
                "group": ["CN", "MCI", "AD", "CN", "MCI", "CN", "AD", "MCI", "AD"],
            }
        )
        self.assertEqual(_support_sites(frame, ("CN", "AD")), {"001", "003"})
        self.assertEqual(_support_sites(frame, GROUPS), {"001"})

    def test_rank_normalization_is_recomputed_in_exact_pair_scope(self):
        analysis = synthetic_analysis()
        use, sites, fitted_groups, reference = _scope_frame(
            analysis, "pair_common_support", "AD", "CN"
        )
        self.assertEqual(set(use["group"]), {"AD", "CN"})
        self.assertEqual(len(sites), 12)
        self.assertEqual(tuple(fitted_groups), ("CN", "AD"))
        self.assertEqual(reference, "CN")
        order = np.argsort(use["fa_value"].to_numpy())
        ordered_scores = use["rank_z"].to_numpy()[order]
        self.assertTrue(np.all(np.diff(ordered_scores) >= 0))
        self.assertTrue(
            np.allclose(
                use["rank_z"].to_numpy(),
                normal_scores(use["fa_value"]),
                atol=1e-12,
                rtol=0,
            )
        )

    def test_site_fixed_effect_removes_between_site_group_bias(self):
        analysis = synthetic_analysis()
        naive = smf.ols(
            'fa_value ~ C(group, Treatment(reference="CN"))', data=analysis
        ).fit()
        naive_ad = naive.params[
            'C(group, Treatment(reference="CN"))[T.AD]'
        ]
        self.assertGreater(float(naive_ad), 1.0)
        results, _, _, _, _ = fit_primary_fa(analysis)
        row = results[
            results["scope"].eq("full_site_fe")
            & results["contrast"].eq("AD_vs_CN")
        ].iloc[0]
        self.assertLess(abs(float(row["rank_normal_beta"])), 0.10)
        self.assertIn("C(site)", row["formula"])
        self.assertNotIn("protocol", row["formula"].lower())

    def test_nine_claim_gate_models_and_corrections(self):
        results, subjects, support, membership, manual = fit_primary_fa(
            synthetic_analysis()
        )
        self.assertEqual(len(results), 9)
        self.assertEqual(set(results["scope"]), set(SCOPES))
        self.assertEqual(
            set(results["contrast"]),
            {"MCI_vs_CN", "AD_vs_CN", "AD_vs_MCI"},
        )
        self.assertTrue(results["claim_gate_covariance"].all())
        self.assertTrue(results["p_holm_within_scope"].between(0, 1).all())
        self.assertTrue(results["p_holm_global_9"].between(0, 1).all())
        self.assertTrue(results["q_bh_global_9"].between(0, 1).all())
        self.assertEqual(len(support), 9)
        self.assertFalse(subjects.empty)
        self.assertFalse(membership.empty)
        self.assertEqual(len(manual), 9)

    def test_manual_cluster_calculation_matches_statsmodels(self):
        _, _, _, _, manual = fit_primary_fa(synthetic_analysis())
        self.assertLessEqual(float(manual["beta_abs_error"].max()), 1e-10)
        self.assertLessEqual(
            float(manual["standard_error_abs_error"].max()), 1e-10
        )
        self.assertLessEqual(float(manual["p_value_abs_error"].max()), 1e-10)
        self.assertLessEqual(float(manual["ci_max_abs_error"].max()), 1e-10)

    def test_configuration_hash_is_stable_hex(self):
        first = configuration_sha256()
        self.assertEqual(first, configuration_sha256())
        self.assertEqual(len(first), 64)
        int(first, 16)


if __name__ == "__main__":
    unittest.main()
