#!/home/ec2-user/fsl/bin/python
"""Replay unchanged Recovery4 QC for the 005_S_6084 Eddy extension."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
BASE_SOURCE = (
    EXP / "scripts/hcp/run_hcp379_eddy_pretract_recovery4_v1.py"
)
CONTRACT_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_eddy_extension_005_S_6084_v1"
)
UNIT = "005_S_6084_I915209"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(
    BASE_SOURCE, "hcp379_eddy_extension_005_S_6084_pretract_base"
)


def specialize() -> None:
    BASE.PLAN = CONTRACT_ROOT / "plan.json"
    BASE.EXPECTED = {UNIT}
    BASE.OUTPUT_ROOT = Path(
        "/data/derivatives/hcp379_v2/"
        "eddy_extension_005_S_6084_pretract_recovery4"
    )


def main() -> int:
    specialize()
    return int(BASE.main())


if __name__ == "__main__":
    raise SystemExit(main())
