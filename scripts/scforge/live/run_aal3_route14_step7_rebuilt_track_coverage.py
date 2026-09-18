#!/usr/bin/env python3
"""Route 14: ACT/SIFT2 coverage rerun after rebuilt native Step-7 products.

This lane uses the current `/data/derivatives/fod/<sid>` FOD, 5TT, and GMWMI
products, then pairs the new scratch streamlines with the best high-label AAL3
parcellation already available for that subject. It is intended to test whether
the sparse matrix is mainly a Step-7 streamline coverage problem rather than an
atlas-warp problem.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live.run_aal3_route11b_act_sift2_track_coverage_rerun import run_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_main("R14"))
