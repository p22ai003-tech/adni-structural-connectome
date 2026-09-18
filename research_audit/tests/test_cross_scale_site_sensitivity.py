import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research_audit"))

from cross_scale_site_sensitivity import (  # noqa: E402
    _model_records,
    _support_sites,
    algebra_validation,
    configuration_sha256,
    fit_site_sensitivities,
    support_tables,
)


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
                noise = ((within % 3) - 1) * 0.02
                micro = site_effect + noise
                topology = 0.4 * site_effect - noise
                rows.append(
                    {
                        "subject_id": f"S{subject:04d}",
                        "group": group,
                        "domain": "WholeBrain",
                        "micro_score": micro,
                        "topology_score": topology,
                        "difference_score": micro - topology,
                        "age": 65.0 + (within % 5),
                        "sex": "F" if within % 2 else "M",
                        "site": site,
                        "log_gap_days": np.log1p(20 + within),
                        "t1_source": "Original" if within % 2 else "Processed",
                        "protocol_bucket": "SITE_PROTOCOL",
                    }
                )
    one = pd.DataFrame(rows)
    return pd.concat(
        [
            one,
            one.assign(domain="functional_DMN"),
            one.assign(domain="functional_Limbic"),
        ],
        ignore_index=True,
    )


class CrossScaleSiteSensitivityTests(unittest.TestCase):
    def test_pair_and_all_three_support_are_exact(self):
        frame = pd.DataFrame(
            {
                "site": ["001"] * 3 + ["002"] * 2 + ["003"] * 2 + ["004"] * 2,
                "group": ["CN", "MCI", "AD", "CN", "MCI", "CN", "AD", "MCI", "AD"],
            }
        )
        self.assertEqual(_support_sites(frame, ("CN", "AD")), {"001", "003"})
        self.assertEqual(_support_sites(frame, GROUPS), {"001"})

    def test_site_fixed_effect_removes_between_site_group_bias(self):
        frame = synthetic_analysis().query("domain == 'WholeBrain'").copy()
        naive = smf.ols(
            'difference_score ~ C(group, Treatment(reference="CN"))',
            data=frame,
        ).fit()
        naive_ad = naive.params[
            'C(group, Treatment(reference="CN"))[T.AD]'
        ]
        self.assertGreater(float(naive_ad), 1.0)
        sites = set(frame["site"])
        records = pd.DataFrame(
            _model_records(
                "WholeBrain", frame, "full_site_fe", "AD", "CN", sites
            )
        )
        beta = records.loc[
            records["response"].eq("difference_score"), "beta"
        ].iloc[0]
        self.assertLess(abs(float(beta)), 0.05)
        self.assertTrue(records["formula"].str.contains("C(site)", regex=False).all())
        self.assertFalse(records["formula"].str.contains("protocol", case=False).any())

    def test_difference_contrast_is_exactly_micro_minus_topology(self):
        results = fit_site_sensitivities(synthetic_analysis())
        validation = algebra_validation(results)
        self.assertTrue(validation["identity_within_1e_10"].all())
        self.assertLessEqual(
            float(validation["difference_beta_error"].abs().max()), 1e-10
        )

    def test_support_tables_emit_all_domain_scope_contrast_cells(self):
        membership, counts = support_tables(synthetic_analysis())
        self.assertEqual(len(counts), 3 * 3 * 3)
        self.assertEqual(set(counts["scope"]), {
            "full_site_fe", "pair_common_support", "all3_site_support"
        })
        all3 = membership[membership["scope"].eq("all3_site_support")]
        self.assertTrue((all3[["n_CN_available", "n_MCI_available", "n_AD_available"]] > 0).all().all())

    def test_configuration_hash_is_stable_hex(self):
        first = configuration_sha256()
        self.assertEqual(first, configuration_sha256())
        self.assertEqual(len(first), 64)
        int(first, 16)


if __name__ == "__main__":
    unittest.main()
