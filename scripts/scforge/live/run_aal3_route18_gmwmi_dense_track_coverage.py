#!/usr/bin/env python3
"""Route 18: dense GMWMI-first ACT/SIFT2 coverage rescue.

Route15 tests high-density dynamic seeding. The live Route15 all-CPU run is
mostly failing at tckgen coverage/runtime, so this lane keeps the same
high-label AAL3-to-B0 parcellation contract but tries GMWMI-first seeding before
falling back to dynamic seeding.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live.run_aal3_route11b_act_sift2_track_coverage_rerun import run_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_main("R18"))
