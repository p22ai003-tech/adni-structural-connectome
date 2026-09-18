#!/usr/bin/env python3
"""Route 15: denser rebuilt-Step7 ACT/SIFT2 coverage canary.

Route14 proved that rebuilt Step7 products improve a sparse AAL3 matrix, while
Route14B showed assignment-only retries do not add coverage. This lane keeps
the same high-label AAL3-to-B0 parcellation contract and tests whether a denser
scratch ACT/SIFT2 track set pushes density and zero rows to full QC.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live.run_aal3_route11b_act_sift2_track_coverage_rerun import run_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_main("R15"))
