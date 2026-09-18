#!/usr/bin/env python3
"""Route 12B: targeted zero-row coverage rerun for unresolved AAL3 cases."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live.run_aal3_route11b_act_sift2_track_coverage_rerun import run_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_main("R12B"))
