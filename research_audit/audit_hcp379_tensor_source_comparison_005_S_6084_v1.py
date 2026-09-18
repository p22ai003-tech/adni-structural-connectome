#!/home/ec2-user/fsl/bin/python
"""Compare locked raw and historical-Eddy tensor physics for 005_S_6084.

This is a subject-specialized, non-overwriting wrapper around the validated
005_S_0610 tensor-source audit implementation.  Only immutable source paths
and the isolated output namespace differ.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
BASE_SOURCE = (
    EXP
    / "research_audit/"
    "audit_hcp379_tensor_source_comparison_005_S_0610_v1.py"
)
UNIT = "005_S_6084_I915209"
OUTPUT = (
    Path("/data/derivatives/hcp379_v2")
    / "pretract_failure_source_audit_v1/"
    "tensor_source_comparison_005_S_6084_v1"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_SOURCE, "hcp379_tensor_source_005_S_6084_base")
BASE.UNIT = UNIT
BASE.RAW = Path("/data/derivatives/mif_dwi") / f"{UNIT}.mif"
BASE.EDDY = Path("/data/derivatives/eddy") / f"{UNIT}_preproc.mif"
BASE.OUTPUT = OUTPUT
BASE.SUMMARY = OUTPUT / "summary.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--nthreads", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.nthreads <= 16:
        parser.error("--nthreads must be in 1..16")
    if args.self_test:
        if BASE.UNIT != UNIT or BASE.OUTPUT != OUTPUT:
            raise AssertionError("specialization differs")
        print("HCP379_TENSOR_SOURCE_COMPARISON_005_S_6084_SELF_TEST_PASS")
        return 0
    payload = (
        BASE.execute(args.nthreads) if args.execute else BASE.preflight()
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] in {"READY", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
