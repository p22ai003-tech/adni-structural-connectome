from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scforge.atlas_contract_v2 import build_contiguous_aal3_from_array, contiguous_mapping
from scforge.config import AtlasConfig


def _atlas() -> AtlasConfig:
    return AtlasConfig(
        name="AAL3",
        mni_image="/tmp/aal3.nii.gz",
        labels_csv=Path(__file__).resolve().parents[2] / "atlas" / "AAL" / "AAL3_labels.csv",
    )


class AtlasContractV2Tests(unittest.TestCase):
    def test_contiguous_mapping_excludes_expected_gaps(self) -> None:
        mapping = contiguous_mapping(_atlas())
        self.assertNotIn(35, mapping)
        self.assertNotIn(36, mapping)
        self.assertNotIn(81, mapping)
        self.assertNotIn(82, mapping)
        self.assertEqual(mapping[1], 1)
        self.assertEqual(mapping[7], 7)
        self.assertEqual(mapping[170], 166)

    def test_build_contiguous_aal3_array(self) -> None:
        data = np.array([[1, 7, 35], [82, 83, 170]], dtype=np.int16)
        out, node_table = build_contiguous_aal3_from_array(data, _atlas())
        self.assertEqual(out[0, 0], 1)
        self.assertEqual(out[0, 1], 7)
        self.assertEqual(out[0, 2], 0)
        self.assertEqual(out[1, 0], 0)
        self.assertEqual(out[1, 1], 79)
        self.assertEqual(out[1, 2], 166)
        self.assertEqual(len(node_table), 166)


if __name__ == "__main__":
    unittest.main()
