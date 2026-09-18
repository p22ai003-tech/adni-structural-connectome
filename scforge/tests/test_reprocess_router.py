from __future__ import annotations

import unittest

from scforge.config import GateConfig
from scforge.reprocess_router import route_subject


def _candidate_gate() -> GateConfig:
    return GateConfig(
        density_fd_sum_min=0.15,
        valid_zero_rows_max=5,
        missing_valid_labels_max=0,
        unique_assigned_nodes_min=120,
        top5_endpoint_fraction_max=0.50,
    )


def _publication_gate() -> GateConfig:
    return GateConfig(
        density_fd_sum_min=0.15,
        valid_zero_rows_max=0,
        missing_valid_labels_max=0,
        unique_assigned_nodes_min=140,
        top5_endpoint_fraction_max=0.35,
    )


class ReprocessRouterTests(unittest.TestCase):
    def test_publication_pass_route(self) -> None:
        decision = route_subject(
            {
                "density": 0.4,
                "valid_zero_rows": 0,
                "missing_valid_labels": 0,
                "unique_assigned_nodes": 160,
                "top5_endpoint_fraction": 0.2,
                "matrix_n": 166,
                "collapse_class": "PASS_ASSIGNMENT_DIVERSITY",
            },
            candidate_gate=_candidate_gate(),
            publication_gate=_publication_gate(),
        )
        self.assertEqual(decision.route, "PUBLICATION_PASS")
        self.assertTrue(decision.publication_pass)

    def test_label_failure_routes_to_branch_b(self) -> None:
        decision = route_subject(
            {
                "density": 0.4,
                "valid_zero_rows": 0,
                "missing_valid_labels": 1,
                "unique_assigned_nodes": 160,
                "top5_endpoint_fraction": 0.2,
                "matrix_n": 165,
                "collapse_class": "PASS_ASSIGNMENT_DIVERSITY",
            },
            candidate_gate=_candidate_gate(),
            publication_gate=_publication_gate(),
        )
        self.assertEqual(decision.route, "BRANCH_B_REBUILD_AAL3_LABEL_CONTRACT")

    def test_assignment_collapse_routes_to_branch_c(self) -> None:
        decision = route_subject(
            {
                "density": 0.3,
                "valid_zero_rows": 0,
                "missing_valid_labels": 0,
                "unique_assigned_nodes": 3,
                "top5_endpoint_fraction": 1.0,
                "matrix_n": 166,
                "collapse_class": "CATASTROPHIC_COLLAPSE",
            },
            candidate_gate=_candidate_gate(),
            publication_gate=_publication_gate(),
        )
        self.assertEqual(decision.route, "BRANCH_C_REGENERATE_5TT_FOD_TRACKS")


if __name__ == "__main__":
    unittest.main()
