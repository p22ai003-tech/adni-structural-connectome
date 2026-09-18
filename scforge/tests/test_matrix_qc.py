from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scforge.qc import qc_matrix


class MatrixQCTests(unittest.TestCase):
    def test_passes_dense_symmetric_zero_diagonal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matrix.csv"
            matrix = np.ones((4, 4), dtype=float)
            np.fill_diagonal(matrix, 0.0)
            np.savetxt(path, matrix, delimiter=",")
            result = qc_matrix(path, expected_nodes=4, density_min=0.5, strict_zero_valid_rows=0)
            self.assertTrue(result.pass_strict)
            self.assertEqual(result.status, "PASS")
            self.assertEqual(result.valid_zero_rows, ())

    def test_fails_zero_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matrix.csv"
            matrix = np.ones((4, 4), dtype=float)
            np.fill_diagonal(matrix, 0.0)
            matrix[2, :] = 0.0
            matrix[:, 2] = 0.0
            np.savetxt(path, matrix, delimiter=",")
            result = qc_matrix(path, expected_nodes=4, density_min=0.1, strict_zero_valid_rows=0)
            self.assertFalse(result.pass_strict)
            self.assertIn(3, result.valid_zero_rows)
            self.assertTrue(any(reason.startswith("valid_zero_rows") for reason in result.reasons))


if __name__ == "__main__":
    unittest.main()
