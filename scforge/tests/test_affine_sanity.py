from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scforge.registration import qc_affine_matrix


class AffineSanityTests(unittest.TestCase):
    def test_identity_transform_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "identity.mat"
            np.savetxt(path, np.eye(4))
            result = qc_affine_matrix(path)
            self.assertEqual(result.status, "PASS")
            self.assertEqual(result.reasons, ())

    def test_singular_transform_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "singular.mat"
            matrix = np.eye(4)
            matrix[2, 2] = 0.0
            np.savetxt(path, matrix)
            result = qc_affine_matrix(path)
            self.assertEqual(result.status, "FAIL")
            self.assertIn("singular_or_near_singular", result.reasons)


if __name__ == "__main__":
    unittest.main()
