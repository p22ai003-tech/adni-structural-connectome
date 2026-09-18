import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research_audit"))

from cross_scale_robustness_analysis import (  # noqa: E402
    DOMAINS,
    RESPONSES,
    _moderation_vectors,
    algebra_validation,
    build_cross_scale_scores,
    configuration_sha256,
    fit_domain_stratum,
    required_features,
)
from submission_sprint_analysis import GROUPS, Stratum  # noqa: E402


def synthetic_cohort(n_per_group=12):
    rows = []
    for group_index, group in enumerate(GROUPS):
        for index in range(n_per_group):
            subject_id = f"{group}_{index:03d}"
            rows.append(
                {
                    "subject_id": subject_id,
                    "group": group,
                    "primary_eligible": True,
                    "age": 68.0 + group_index + index / 10,
                    "sex": "F" if index % 2 else "M",
                    "site": f"{index % 9:03d}",
                    "log_gap_days": np.log1p(10 + index),
                    "t1_source": "Original" if index % 3 else "Processed",
                    "protocol_bucket": "SIEMENS_54" if index % 2 else "SIEMENS_30",
                    "siemens_54": bool(index % 2),
                }
            )
    return pd.DataFrame(rows)


def synthetic_metrics(cohort):
    rows = []
    feature_signs = {}
    for definition in DOMAINS.values():
        feature_signs.update(definition["micro_components"])
        feature, sign = definition["topology_component"]
        feature_signs[feature] = sign
    for row in cohort.itertuples(index=False):
        group_index = GROUPS.index(row.group)
        subject_index = int(row.subject_id.rsplit("_", 1)[-1])
        disease_axis = group_index + subject_index / 100.0
        for feature_id, orientation in feature_signs.items():
            # Raw values oppose the disease axis when the declared orientation
            # is negative and follow it when positive.
            value = disease_axis * orientation
            mapping, family, network, metric = feature_id.split("::")
            rows.append(
                {
                    "subject_id": row.subject_id,
                    "feature_id": feature_id,
                    "value": value,
                    "mapping": mapping,
                    "feature_family": family,
                    "network": network,
                    "metric": metric,
                }
            )
    return pd.DataFrame(rows)


class CrossScaleRobustnessTests(unittest.TestCase):
    def test_required_features_cover_all_domain_components(self):
        expected = sum(
            len(definition["micro_components"]) + 1
            for definition in DOMAINS.values()
        )
        self.assertEqual(len(required_features()), expected)
        self.assertEqual(len(set(required_features())), expected)

    def test_paired_scores_are_oriented_standardized_and_exact(self):
        cohort = synthetic_cohort()
        scores, missingness, components = build_cross_scale_scores(
            cohort, synthetic_metrics(cohort)
        )
        self.assertEqual(set(scores["domain"]), set(DOMAINS))
        self.assertEqual(len(missingness), len(DOMAINS) * len(GROUPS))
        self.assertTrue((missingness["missing_paired_n"] == 0).all())
        self.assertTrue(set(components["orientation_sign"]) == {-1.0, 1.0})
        identity = scores["difference_score"] - (
            scores["micro_score"] - scores["topology_score"]
        )
        self.assertLessEqual(float(identity.abs().max()), 1e-12)
        for _, frame in scores.groupby("domain"):
            self.assertAlmostEqual(float(frame["micro_score"].mean()), 0.0, places=12)
            self.assertAlmostEqual(
                float(frame["micro_score"].std(ddof=0)), 1.0, places=12
            )
            self.assertAlmostEqual(float(frame["topology_score"].mean()), 0.0, places=12)
            self.assertAlmostEqual(
                float(frame["topology_score"].std(ddof=0)), 1.0, places=12
            )

    def test_component_missingness_forces_domain_pairing(self):
        cohort = synthetic_cohort()
        metrics = synthetic_metrics(cohort)
        target_subject = cohort.loc[cohort["group"].eq("AD"), "subject_id"].iloc[0]
        target_feature = next(
            iter(DOMAINS["functional_Limbic"]["micro_components"])
        )
        metrics.loc[
            metrics["subject_id"].eq(target_subject)
            & metrics["feature_id"].eq(target_feature),
            "value",
        ] = np.nan
        scores, missingness, _ = build_cross_scale_scores(cohort, metrics)
        limbic_ad = missingness[
            missingness["domain"].eq("functional_Limbic")
            & missingness["group"].eq("AD")
        ].iloc[0]
        self.assertEqual(int(limbic_ad["missing_paired_n"]), 1)
        self.assertFalse(
            (
                scores["domain"].eq("functional_Limbic")
                & scores["subject_id"].eq(target_subject)
            ).any()
        )

    def test_fitted_difference_beta_is_micro_minus_topology(self):
        cohort = synthetic_cohort(n_per_group=24)
        scores, _, _ = build_cross_scale_scores(cohort, synthetic_metrics(cohort))
        domain = scores[scores["domain"].eq("WholeBrain")].merge(
            cohort.drop(columns=["group", "primary_eligible"]),
            on="subject_id",
            how="left",
            validate="1:1",
        )
        stratum = Stratum("synthetic", frozenset(domain["subject_id"]))
        results = pd.DataFrame(fit_domain_stratum(domain, stratum))
        self.assertEqual(set(results["response"]), set(RESPONSES))
        self.assertEqual(
            set(results["contrast"]), {"MCI_vs_CN", "AD_vs_CN", "AD_vs_MCI"}
        )
        validation = algebra_validation(results)
        self.assertTrue(validation["algebra_identity_within_1e_10"].all())
        self.assertLessEqual(float(validation["difference_beta_error"].abs().max()), 1e-10)

    def test_moderation_vectors_define_all_three_group_contrasts(self):
        names = [
            "Intercept",
            'C(group, Treatment(reference="CN"))[T.MCI]',
            'C(group, Treatment(reference="CN"))[T.AD]',
            "moderator_value",
            'C(group, Treatment(reference="CN"))[T.MCI]:moderator_value',
            'C(group, Treatment(reference="CN"))[T.AD]:moderator_value',
        ]
        vectors = _moderation_vectors(names)
        self.assertEqual(
            set(vectors), {"MCI_vs_CN", "AD_vs_CN", "AD_vs_MCI"}
        )
        self.assertAlmostEqual(float(vectors["AD_vs_MCI"].sum()), 0.0)

    def test_configuration_hash_is_stable_hex(self):
        first = configuration_sha256()
        second = configuration_sha256()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        int(first, 16)


if __name__ == "__main__":
    unittest.main()
