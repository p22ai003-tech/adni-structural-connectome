from __future__ import annotations

import unittest
from pathlib import Path

from scforge.config import load_config
from scforge.qc import aal3_label_summary, load_aal3_label_table


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "scforge.yaml"


class AAL3LabelMappingTests(unittest.TestCase):
    def test_real_aal3_contract(self) -> None:
        cfg = load_config(CONFIG)
        table = load_aal3_label_table(cfg.atlas)
        self.assertEqual(table.n_valid_labels, 166)
        self.assertEqual(table.expected_gaps, (35, 36, 81, 82))
        self.assertIn(7, table.original_to_node)
        self.assertIn(8, table.original_to_node)
        self.assertEqual(table.original_to_node[1], 1)
        self.assertEqual(table.original_to_node[34], 34)
        self.assertEqual(table.original_to_node[37], 35)

    def test_summary_has_required_labels(self) -> None:
        cfg = load_config(CONFIG)
        summary = aal3_label_summary(cfg.atlas)
        self.assertTrue(summary["required_labels_present"])
        self.assertEqual(summary["missing_unexpected_labels"], "")


if __name__ == "__main__":
    unittest.main()
