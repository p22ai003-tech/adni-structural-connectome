from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research_audit.audit_historical_source_preflight import (
    OUTPUT_ROWS,
    OUTPUT_SUMMARY,
    OUTPUT_VALIDATION,
    antipodal_unique_directions,
    gradient_proxy,
)


OUTPUT = Path("/home/ec2-user/exp/research_audit/outputs")


class HistoricalSourcePreflightTests(unittest.TestCase):
    def test_antipodal_direction_and_gradient_proxy_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bval = root / "x.bval"
            bvec = root / "x.bvec"
            bval.write_text("0 1000 1000 1000 2000\n", encoding="utf-8")
            bvec.write_text(
                "0 1 -1 0 0\n"
                "0 0 0 1 0\n"
                "0 0 0 0 1\n",
                encoding="utf-8",
            )
            result = gradient_proxy(bval, bvec)
            self.assertEqual(result["n_b0"], 1)
            self.assertEqual(result["unique_nonzero_directions_antipodal_round4"], 3)
            self.assertEqual(result["shells_round50"], "1000;2000")
            self.assertEqual(result["n_nonzero_shells"], 2)
            self.assertEqual(result["b0_vector_norm_gt_0_05_count"], 0)

        bvals = np.array([1000.0, 1000.0])
        bvecs = np.array([[1.0, -1.0], [0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(antipodal_unique_directions(bvals, bvecs), 1)

    def test_canonical_proxy_release_is_checksum_bound_and_limited(self) -> None:
        validation = json.loads((OUTPUT / OUTPUT_VALIDATION).read_text())
        self.assertEqual(validation["status"], "PASS")
        self.assertEqual(
            validation["release_status"],
            "HISTORICAL_PROXY_ONLY_RAW_SOURCE_VALIDATION_REQUIRED",
        )
        self.assertEqual(validation["counts"]["rows"], 530)
        self.assertEqual(validation["counts"]["pe_and_readout_complete"], 380)
        self.assertEqual(validation["counts"]["unique_directions_lt_30"], 10)
        self.assertEqual(validation["counts"]["multiple_nonzero_shells"], 6)
        self.assertEqual(validation["counts"]["b0_vector_norm_issue"], 51)
        self.assertEqual(validation["counts"]["processed_t1_headers"], 225)
        self.assertEqual(validation["counts"]["processed_t1_uid_text_found"], 0)
        self.assertEqual(validation["safety"]["raw_source_voxel_arrays_read"], 0)
        self.assertEqual(
            validation["safety"]["historical_derivative_voxel_arrays_read"], 0
        )

        for name in (OUTPUT_ROWS, OUTPUT_SUMMARY):
            expected = validation["outputs"][name]["sha256"]
            observed = hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest()
            self.assertEqual(observed, expected)
        rows = pd.read_csv(OUTPUT / OUTPUT_ROWS)
        self.assertEqual(len(rows), 530)
        self.assertEqual(
            set(rows["evidence_role"]),
            {"historical_selected_id_derivative_proxy_not_locked_raw_source"},
        )
        processed = rows[rows["current_t1_type"] == "Processed"]
        self.assertEqual(len(processed), 225)
        self.assertFalse(processed["processed_t1_uid_text_found"].astype(bool).any())


if __name__ == "__main__":
    unittest.main()
