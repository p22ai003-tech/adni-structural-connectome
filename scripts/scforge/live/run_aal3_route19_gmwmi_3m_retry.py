#!/usr/bin/env python3
"""Route 19: 3M GMWMI-first retry for slow Route18 dense-coverage failures.

Route18 targets 6M streamlines and early-stops subjects whose selected
streamline yield predicts an impractical runtime. Route19 keeps the same
GMWMI-first ACT/SIFT2 strategy but uses a separate root and a 3M default so
slow Route18 cases can still reach connectome scoring.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live.run_aal3_route11b_act_sift2_track_coverage_rerun import run_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_main("R19"))
