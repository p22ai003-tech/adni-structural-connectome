#!/home/ec2-user/fsl/bin/python
"""Compact the exact two-subject mask-union Recovery4 namespace.

This wrapper reuses the validated Recovery4 compactor without changing its
shared source or the active production workers.  Dry-run is the default.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "compact_hcp379_pretract_recovery4_v3.py"
)
ROOT = Path(
    "/data/derivatives/hcp379_v2/"
    "corrected_legacy_tensor_mask_union_recovery4"
)
TARGETS = {"003_S_0908_I1249292", "003_S_4350_I1252856"}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPACTOR = load_module(SOURCE, "hcp379_mask_union_compactor")
COMPACTOR.ROOT_CONTRACTS[ROOT.resolve()] = len(TARGETS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if (
            COMPACTOR.ROOT_CONTRACTS.get(ROOT.resolve()) != 2
            or len(TARGETS) != 2
        ):
            raise AssertionError("mask-union compaction scope differs")
        print("HCP379_MASK_UNION_COMPACTION_V1_SELF_TEST_PASS")
        return 0
    units = COMPACTOR.summary_units(ROOT, len(TARGETS))
    if set(units) != TARGETS:
        raise ValueError(f"mask-union compaction identities differ: {units}")
    results = [
        COMPACTOR.compact_unit(
            root=ROOT.resolve(), unit=unit, execute=args.execute
        )
        for unit in units
    ]
    payload = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_mask_union_pretract_recovery_compaction_summary"
        ),
        "status": (
            "PASS_COMPACTED"
            if args.execute
            and all(
                result.get("status") == "PASS_COMPACTED"
                for result in results
            )
            else "DRY_RUN_READY"
        ),
        "non_overwriting": True,
        "target_n": len(TARGETS),
        "target_units": sorted(TARGETS),
        "shared_compactor_unchanged": True,
        "results": results,
    }
    COMPACTOR.PRETRACT.atomic_json(
        ROOT / "manifests/mask_union_compaction_summary.json",
        payload,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return (
        0
        if payload["status"]
        in {"DRY_RUN_READY", "PASS_COMPACTED"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
