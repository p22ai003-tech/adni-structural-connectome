from __future__ import annotations

import unittest

from scforge.route_selector import SpatialContractQC, choose_best_spatial_contract, score_spatial_contract


class RouteSelectorTests(unittest.TestCase):
    def test_hard_fail_scores_zero(self) -> None:
        qc = SpatialContractQC(
            route_name="bad",
            label_survival_frac=1,
            label_inside_mask_frac=1,
            t1_b0_boundary_score=1,
            five_tt_mask_dice=1,
            transform_sanity_score=1,
            aal_required_label_score=1,
            hard_fail_reasons=("left_right_flip",),
        )
        self.assertEqual(score_spatial_contract(qc), 0.0)

    def test_choose_best_ignores_hard_fail(self) -> None:
        weak = SpatialContractQC("weak", 0.7, 0.7, 0.7, 0.7, 0.7, 1.0)
        strong_fail = SpatialContractQC("strong_fail", 1, 1, 1, 1, 1, 1, hard_fail_reasons=("bad",))
        best = choose_best_spatial_contract([weak, strong_fail])
        self.assertIsNotNone(best)
        self.assertEqual(best.route_name, "weak")


if __name__ == "__main__":
    unittest.main()
