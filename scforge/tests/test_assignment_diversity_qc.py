from __future__ import annotations

import unittest

import numpy as np

from scforge.assignment_diversity_qc import classify_assignment_pairs


class AssignmentDiversityQCTests(unittest.TestCase):
    def test_assignment_diversity_passes_broad_assignments(self) -> None:
        pairs = np.array([(i % 150 + 1, (i * 7) % 150 + 1) for i in range(1000)], dtype=np.int16)
        result = classify_assignment_pairs(pairs)
        self.assertGreaterEqual(result.unique_assigned_nodes, 140)
        self.assertEqual(result.collapse_class, "PASS_ASSIGNMENT_DIVERSITY")

    def test_assignment_diversity_detects_catastrophic_collapse(self) -> None:
        pairs = np.array([(33, 77), (33, 79), (77, 79)] * 100, dtype=np.int16)
        result = classify_assignment_pairs(pairs)
        self.assertEqual(result.unique_assigned_nodes, 3)
        self.assertEqual(result.collapse_class, "CATASTROPHIC_COLLAPSE")


if __name__ == "__main__":
    unittest.main()
