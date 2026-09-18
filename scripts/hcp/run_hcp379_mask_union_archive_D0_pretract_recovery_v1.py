#!/home/ec2-user/fsl/bin/python
"""Run the validated mask-union recovery for one archive-D0 subject.

The shared Recovery4 and mask-union implementations remain unchanged.  This
wrapper binds the independently validated archive-D0 extension audit, the
52-unit archive-D0 lane contract, and a disjoint one-subject output root.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
HCP = Path("/data/derivatives/hcp379_v2")
BASE_SOURCE = (
    EXP / "scripts/hcp/run_hcp379_mask_union_pretract_recovery_v1.py"
)
LANE_SOURCE = (
    EXP / "scripts/hcp/run_hcp379_archive_D0_pretract_recovery4_v3.py"
)
AUDIT_SUMMARY = (
    HCP
    / "pretract_failure_source_audit_v1/"
    "mask_union_archive_D0_extension_v1/summary.json"
)
INDEPENDENT_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_mask_union_archive_D0_extension_v1/validation.json"
)
OUTPUT_ROOT = HCP / "corrected_archive_D0_mask_union_recovery4"
TARGETS = ("114_S_5234_I1130066",)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_SOURCE, "hcp379_mask_union_archive_D0_base_runner")
LANE = load_module(LANE_SOURCE, "hcp379_mask_union_archive_D0_lane")


def specialize() -> None:
    # The shared runner expects these equivalent generic lane names.
    LANE.EXPECTED_AUDIT_N = LANE.EXPECTED_EXECUTION_N
    LANE.EXPECTED_CANARY_OVERLAP_N = 0
    BASE.LANE = LANE
    BASE.AUDIT_SUMMARY = AUDIT_SUMMARY
    BASE.INDEPENDENT_VALIDATION = INDEPENDENT_VALIDATION
    BASE.OUTPUT_ROOT = OUTPUT_ROOT
    BASE.TARGETS = TARGETS
    BASE.__file__ = str(Path(__file__).resolve())


def main() -> int:
    specialize()
    return int(BASE.main())


if __name__ == "__main__":
    raise SystemExit(main())
