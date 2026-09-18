#!/home/ec2-user/fsl/bin/python
"""Compare raw and historical-Eddy tensor physics for 005_S_0610.

The audit is diagnosis-blind and non-overwriting.  It determines whether the
new Recovery4 tensor failure is introduced by the historical Eddy derivative
or is already present in the locked raw DWI/gradient source.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
HCP = Path("/data/derivatives/hcp379_v2")
UNIT = "005_S_0610_I906100"
RAW = Path("/data/derivatives/mif_dwi") / f"{UNIT}.mif"
EDDY = Path("/data/derivatives/eddy") / f"{UNIT}_preproc.mif"
HELPER = EXP / "scripts/hcp/hcp_eddy_tensor_repair_v2.py"
OUTPUT = (
    HCP
    / "pretract_failure_source_audit_v1/"
    "tensor_source_comparison_005_S_0610_v1"
)
SUMMARY = OUTPUT / "summary.json"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = load_module(HELPER, "hcp379_tensor_source_comparison_helper")


def record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def preflight() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_tensor_source_comparison_preflight",
        "status": "READY",
        "unit": UNIT,
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "source_images_modified": False,
        "imaging_executed": False,
        "records": {
            "raw": record(RAW),
            "historical_eddy": record(EDDY),
            "helper": record(HELPER),
            "implementation": record(Path(__file__)),
        },
    }


def execute(nthreads: int) -> dict[str, Any]:
    pre = preflight()
    if SUMMARY.is_file():
        prior = json.loads(SUMMARY.read_text(encoding="utf-8"))
        if prior.get("records") == pre["records"]:
            return prior
        raise ValueError("source binding changed; refusing audit overwrite")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT / "tensor_source_comparison.log"
    with log_path.open("x", encoding="utf-8") as log:
        raw = HELPERS.tensor_audit(
            RAW, OUTPUT / "raw_tensor_audit", nthreads, log
        )
        eddy = HELPERS.tensor_audit(
            EDDY, OUTPUT / "eddy_tensor_audit", nthreads, log
        )
    if raw["gate"] == "PASS" and eddy["gate"] == "FAIL":
        decision = "RECOVER_EDDY_FROM_LOCKED_RAW"
    elif raw["gate"] == "FAIL":
        decision = "RAW_SOURCE_OR_GRADIENT_RECONVERSION_REQUIRED"
    elif raw["gate"] == "PASS" and eddy["gate"] == "PASS":
        decision = "RECOVERY4_TENSOR_PATH_INVESTIGATION_REQUIRED"
    else:
        decision = "UNRESOLVED"
    payload = {
        **pre,
        "record_type": "hcp379_tensor_source_comparison_audit",
        "status": "PASS",
        "generated_utc": HELPERS.utc_now(),
        "imaging_executed": True,
        "decision": decision,
        "raw_tensor_audit": raw,
        "historical_eddy_tensor_audit": eddy,
        "source_hashes_unchanged": (
            pre["records"]["raw"] == record(RAW)
            and pre["records"]["historical_eddy"] == record(EDDY)
        ),
        "log": record(log_path),
    }
    HELPERS.atomic_json(SUMMARY, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--nthreads", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.nthreads <= 16:
        parser.error("--nthreads must be in 1..16")
    if args.self_test:
        if UNIT != "005_S_0610_I906100":
            raise AssertionError(UNIT)
        print("HCP379_TENSOR_SOURCE_COMPARISON_SELF_TEST_PASS")
        return 0
    payload = execute(args.nthreads) if args.execute else preflight()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] in {"READY", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
