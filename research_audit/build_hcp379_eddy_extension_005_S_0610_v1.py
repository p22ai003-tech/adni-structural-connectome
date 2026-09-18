#!/home/ec2-user/fsl/bin/python
"""Build and validate the source-aware Eddy contract for 005_S_0610."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP = Path("/data/derivatives/hcp379_v2")
UNIT = "005_S_0610_I906100"
RECONVERSION = (
    HCP
    / "pretract_failure_source_audit_v1/"
    "dicom_reconversion_005_S_0610_v1/summary.json"
)
COMPARISON = (
    HCP
    / "pretract_failure_source_audit_v1/"
    "tensor_source_comparison_005_S_0610_v1/summary.json"
)
HISTORICAL_EDDY = (
    Path("/data/derivatives/eddy") / f"{UNIT}_preproc.mif"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_eddy_extension_005_S_0610_v1"
)
PLAN = OUTPUT / "plan.json"
CONTRACT = OUTPUT / "contract.json"
VALIDATION = OUTPUT / "validation.json"
RECOVERY_ROOT = HCP / "eddy_integrity_recovery_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reconversion = load_json(RECONVERSION)
    comparison = load_json(COMPARISON)
    fresh = Path(reconversion["outputs"]["mif"]["path"]).resolve()
    bval = Path(reconversion["outputs"]["bval"]["path"]).resolve()
    bvalues = [float(value) for value in bval.read_text().split()]
    if (
        reconversion.get("status") != "PASS"
        or reconversion.get("decision") != "READY_FOR_SOURCE_AWARE_EDDY"
        or reconversion.get("tensor_audit", {}).get("gate") != "PASS"
        or comparison.get("status") != "PASS"
        or comparison.get("decision")
        != "RAW_SOURCE_OR_GRADIENT_RECONVERSION_REQUIRED"
        or len(bvalues) != 54
        or sum(value < 50 for value in bvalues) != 6
        or sum(value >= 50 for value in bvalues) != 48
    ):
        raise ValueError("source-reconversion evidence differs")
    target = {
        "unit": UNIT,
        "lane": "corrected_core",
        "route": "REGENERATE_EDDY_FROM_RECONVERTED_DICOM",
        "route_ready": True,
        "historical_eddy": record(HISTORICAL_EDDY),
        "raw_source": record(fresh),
        "bval_source": record(bval),
        "eddy_zero_volume_indices": [],
        "raw_zero_volume_indices": [],
        "retained_volume_indices": list(range(54)),
        "retained_b0_n": 6,
        "retained_diffusion_n": 48,
        "output_root": str(
            (RECOVERY_ROOT / "subjects" / UNIT).resolve()
        ),
    }
    plan = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_eddy_integrity_recovery_plan",
        "status": "PASS",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "historical_eddy_overwrite_allowed": False,
        "per_subject_parameter_tuning_allowed": False,
        "qc_threshold_relaxation_allowed": False,
        "target_n": 1,
        "unresolved_n": 0,
        "unresolved_units": [],
        "minimum_retained_b0": 3,
        "minimum_retained_diffusion": 30,
        "execution_backend": (
            "GPU_EDDY_IF_VALIDATED_GPU_HOST_ELSE_EXPLICIT_CPU"
        ),
        "imaging_executed": False,
        "targets": [target],
        "records": {
            "dicom_reconversion": record(RECONVERSION),
            "tensor_source_comparison": record(COMPARISON),
            "builder": record(Path(__file__)),
        },
    }
    atomic_json(PLAN, plan)
    metadata = reconversion["selected_metadata"]
    acquisition = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_eddy_acquisition_contract",
        "status": "PASS",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "per_subject_parameter_tuning_used": False,
        "historical_outputs_modified": False,
        "target_n": 1,
        "acquisitions": [
            {
                "unit": UNIT,
                "metadata_route": "PINNED_DCM2NIIX_RECONVERSION",
                "pe_dir": metadata["PhaseEncodingDirection"],
                "phase_direction_confidence": "DIRECT_DCM2NIIX",
                "total_readout_time_seconds": metadata[
                    "TotalReadoutTime"
                ],
                "readout_confidence": "DIRECT_DCM2NIIX",
                "selected_dcm2niix_metadata": metadata,
                "raw_source": record(fresh),
            }
        ],
        "records": {
            "recovery_plan": record(PLAN),
            "dicom_reconversion": record(RECONVERSION),
            "builder": record(Path(__file__)),
        },
    }
    atomic_json(CONTRACT, acquisition)
    checks = {
        "exact_one_subject_contract": target["unit"] == UNIT,
        "fresh_source_hash_replays": (
            target["raw_source"] == record(fresh)
            and acquisition["acquisitions"][0]["raw_source"]
            == record(fresh)
        ),
        "fresh_tensor_gate_passes": (
            reconversion["tensor_audit"]["gate"] == "PASS"
        ),
        "historical_and_old_raw_tensor_gates_fail": (
            comparison["raw_tensor_audit"]["gate"] == "FAIL"
            and comparison["historical_eddy_tensor_audit"]["gate"]
            == "FAIL"
        ),
        "volume_and_gradient_counts_are_exact": (
            len(bvalues) == 54
            and target["retained_b0_n"] == 6
            and target["retained_diffusion_n"] == 48
        ),
        "acquisition_values_are_direct_and_valid": (
            acquisition["acquisitions"][0]["pe_dir"] == "j"
            and abs(
                float(
                    acquisition["acquisitions"][0][
                        "total_readout_time_seconds"
                    ]
                )
                - 0.089485
            )
            < 1e-12
        ),
        "plan_and_contract_are_blind": all(
            value.get("diagnosis_labels_used") is False
            and value.get("outcomes_used") is False
            and value.get("connectome_density_used") is False
            for value in (plan, acquisition)
        ),
        "historical_eddy_is_outside_recovery_root": (
            not HISTORICAL_EDDY.resolve().is_relative_to(
                RECOVERY_ROOT.resolve()
            )
        ),
    }
    passed = sum(checks.values())
    validation = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_eddy_extension_005_S_0610_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": utc_now(),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": {
            name: "PASS" if value else "FAIL"
            for name, value in checks.items()
        },
        "records": {
            "plan": record(PLAN),
            "contract": record(CONTRACT),
            "builder": record(Path(__file__)),
        },
    }
    atomic_json(VALIDATION, validation)
    return plan, acquisition, validation


def main() -> int:
    plan, contract, validation = build()
    print(
        json.dumps(
            {
                "plan": plan,
                "contract": contract,
                "validation": validation,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
