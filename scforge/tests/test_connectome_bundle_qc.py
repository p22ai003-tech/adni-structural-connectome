from __future__ import annotations

import unittest

import numpy as np

from scforge.qc import qc_connectome_bundle


def _base_bundle(n: int = 4):
    off_diagonal = np.ones((n, n), dtype=float) - np.eye(n, dtype=float)
    count = 2.0 * off_diagonal
    volumes = np.arange(1, n + 1, dtype=float) * 1000.0
    rd = 0.0006 * off_diagonal
    ad = 0.0012 * off_diagonal
    md = (ad + 2.0 * rd) / 3.0
    bundle = {
        "count": count,
        "fd_sum": 0.65 * count,
        "count_invnodevol": 2.0 * count / (volumes[:, None] + volumes[None, :]),
        "len_mean": 80.0 * off_diagonal,
        "invlen_mean": (1.0 / 80.0) * off_diagonal,
        "fa_mean": 0.45 * off_diagonal,
        "md_mean": md,
        "rd_mean": rd,
        "ad_mean": ad,
    }
    return bundle, volumes


class ConnectomeBundleQCTests(unittest.TestCase):
    def test_valid_complete_bundle_passes_and_reports_density(self) -> None:
        bundle, volumes = _base_bundle()
        result = qc_connectome_bundle(bundle, volumes, expected_nodes=4)
        self.assertTrue(result.pass_strict, result.failures)
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.density, 1.0)
        self.assertEqual(result.zero_count_nodes, ())

    def test_requires_exactly_the_frozen_nine_matrices(self) -> None:
        bundle, volumes = _base_bundle()
        del bundle["ad_mean"]
        bundle["mystery"] = np.zeros((4, 4))
        result = qc_connectome_bundle(bundle, volumes, expected_nodes=4)
        self.assertIn("missing_matrices:ad_mean", result.failures)
        self.assertIn("unexpected_matrices:mystery", result.failures)

    def test_rejects_shape_finite_symmetry_diagonal_and_sign_defects(self) -> None:
        defect_cases = {
            "bad_shape": ("fa_mean", np.zeros((3, 3)), "fa_mean:shape_"),
            "nonfinite": ("fa_mean", None, "fa_mean:nonfinite"),
            "asymmetric": ("fa_mean", None, "fa_mean:asymmetric"),
            "diagonal": ("fa_mean", None, "fa_mean:nonzero_diagonal"),
            "negative": ("fa_mean", None, "fa_mean:negative"),
        }
        for case, (name, replacement, expected_reason) in defect_cases.items():
            bundle, volumes = _base_bundle()
            if case == "nonfinite":
                replacement = bundle[name].copy()
                replacement[0, 1] = replacement[1, 0] = np.nan
            elif case == "asymmetric":
                replacement = bundle[name].copy()
                replacement[0, 1] += 0.1
            elif case == "diagonal":
                replacement = bundle[name].copy()
                replacement[0, 0] = 0.1
            elif case == "negative":
                replacement = bundle[name].copy()
                replacement[0, 1] = replacement[1, 0] = -0.1
            bundle[name] = replacement
            with self.subTest(case=case):
                result = qc_connectome_bundle(bundle, volumes, expected_nodes=4)
                self.assertTrue(any(reason.startswith(expected_reason) for reason in result.failures), result.failures)

    def test_rejects_fractional_count_and_count_fd_sum_duplication(self) -> None:
        bundle, volumes = _base_bundle()
        bundle["count"][0, 1] = bundle["count"][1, 0] = 2.5
        result = qc_connectome_bundle(bundle, volumes, expected_nodes=4)
        self.assertIn("count:fractional", result.failures)

        bundle, volumes = _base_bundle()
        bundle["fd_sum"] = bundle["count"].copy()
        result = qc_connectome_bundle(bundle, volumes, expected_nodes=4)
        self.assertIn("count_and_fd_sum:allclose_duplicate", result.failures)

    def test_rejects_support_mismatches_without_imputation(self) -> None:
        bundle, volumes = _base_bundle()
        bundle["count"][0, 1] = bundle["count"][1, 0] = 0.0
        bundle["count_invnodevol"] = 2.0 * bundle["count"] / (volumes[:, None] + volumes[None, :])
        result = qc_connectome_bundle(bundle, volumes, expected_nodes=4)
        self.assertIn("fa_mean:nonzero_outside_count_support", result.failures)

        bundle, volumes = _base_bundle()
        bundle["len_mean"][0, 1] = bundle["len_mean"][1, 0] = 0.0
        result = qc_connectome_bundle(bundle, volumes, expected_nodes=4)
        self.assertIn("len_mean:zero_on_connected_edge", result.failures)

    def test_rejects_tensor_ranges_order_and_identity(self) -> None:
        bundle, volumes = _base_bundle()
        bundle["fa_mean"][0, 1] = bundle["fa_mean"][1, 0] = 1.1
        bundle["ad_mean"][0, 1] = bundle["ad_mean"][1, 0] = 0.0005
        bundle["md_mean"][0, 2] = bundle["md_mean"][2, 0] = 0.02
        result = qc_connectome_bundle(bundle, volumes, expected_nodes=4)
        self.assertIn("fa_mean:outside_0_to_1", result.failures)
        self.assertIn("md_mean:outside_diffusivity_range", result.failures)
        self.assertIn("tensor_order:ad_below_md", result.failures)
        self.assertIn("tensor_identity:md_ne_ad_plus_2rd_over_3", result.failures)

    def test_rejects_invalid_volumes_and_formula_but_reports_zero_count_nodes(self) -> None:
        bundle, volumes = _base_bundle()
        bad_volumes = volumes.copy()
        bad_volumes[0] = 0.0
        result = qc_connectome_bundle(bundle, bad_volumes, expected_nodes=4)
        self.assertIn(
            "node_volumes_mm3:must_be_exactly_one_finite_positive_value_per_node",
            result.failures,
        )

        bundle, volumes = _base_bundle()
        bundle["count_invnodevol"][0, 1] *= 2.0
        bundle["count_invnodevol"][1, 0] *= 2.0
        result = qc_connectome_bundle(bundle, volumes, expected_nodes=4)
        self.assertIn("count_invnodevol:formula_mismatch", result.failures)

        bundle, volumes = _base_bundle()
        for matrix in bundle.values():
            matrix[3, :] = 0.0
            matrix[:, 3] = 0.0
        result = qc_connectome_bundle(bundle, volumes, expected_nodes=4)
        self.assertEqual(result.zero_count_nodes, (4,))
        self.assertTrue(result.pass_strict, result.failures)


if __name__ == "__main__":
    unittest.main()
