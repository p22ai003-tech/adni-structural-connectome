#!/usr/bin/env python3
"""Freeze the non-executable one-unit CPU/GPU Eddy equivalence package.

This builder performs only hashing, metadata capture, and static validation.
It cannot resize EC2, start imaging, or create the future GPU output root.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_ROOT = Path("/home/ec2-user/exp/research_audit")
OUTPUT_ROOT = AUDIT_ROOT / "outputs" / "gpu_eddy_equivalence_package_v1"
RUNNER_SOURCE = AUDIT_ROOT / "run_gpu_eddy_equivalence_canary.py"
UNIT = "168_S_6735_I1175371"
CPU_UNIT_ROOT = (
    Path("/data/derivatives/scforge_v2/h04a_canary_20260718_v1/subjects") / UNIT
)
CPU_INPUT = CPU_UNIT_ROOT / "01_dwi" / "dwi_denoised_degibbs.mif"
CPU_OUTPUT = CPU_UNIT_ROOT / "01_dwi" / "dwi_preproc.mif"
CPU_QC = CPU_UNIT_ROOT / "01_dwi" / "eddy_qc"
SOURCE_METADATA = CPU_UNIT_ROOT / "00_inputs" / "dwi_source_metadata.json"
CUDA_EDDY = Path("/home/ec2-user/fsl/bin/eddy_cuda11.0")
MRTRIX_BIN = Path("/home/ec2-user/mrtrix3/bin")
FSL_DIR = Path("/home/ec2-user/fsl")
PHASE_A_SERVICE = "scforge-h04a-r1-phase-a-retry2.service"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def command(argv: list[str]) -> str:
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {argv!r}\n"
            f"{(result.stdout or '')[-2000:]}\n{(result.stderr or '')[-2000:]}"
        )
    return (result.stdout or "").strip()


def require_sources(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required frozen source files are missing: " + ", ".join(missing))


def reference_rows() -> list[dict[str, Any]]:
    sources = [
        ("input_contract", CPU_UNIT_ROOT / "00_inputs" / "input_contract.json"),
        ("source_metadata", SOURCE_METADATA),
        ("source_bval", CPU_UNIT_ROOT / "00_inputs" / "dwi.bval"),
        ("source_bvec", CPU_UNIT_ROOT / "00_inputs" / "dwi.bvec"),
        ("gpu_canary_input", CPU_INPUT),
        ("cpu_reference_output", CPU_OUTPUT),
        ("gradient_contract", CPU_UNIT_ROOT / "01_dwi" / "gradient_contract.json"),
        ("cpu_dwifslpreproc_log", CPU_UNIT_ROOT / "logs" / "01_dwifslpreproc.log"),
        ("cpu_eddy_mask", CPU_QC / "eddy_mask.nii"),
        ("cpu_eddy_parameters", CPU_QC / "eddy_parameters"),
        ("cpu_eddy_outlier_map", CPU_QC / "eddy_outlier_map"),
        ("cpu_eddy_movement_rms", CPU_QC / "eddy_movement_rms"),
        ("cpu_eddy_restricted_movement_rms", CPU_QC / "eddy_restricted_movement_rms"),
    ]
    require_sources([path for _, path in sources])
    return [
        {
            "role": role,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for role, path in sources
    ]


def write_reference_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", "path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def cpu_reference_metadata() -> dict[str, Any]:
    size = [int(value) for value in command([str(MRTRIX_BIN / "mrinfo"), str(CPU_OUTPUT), "-size"]).split()]
    spacing_text = command([str(MRTRIX_BIN / "mrinfo"), str(CPU_OUTPUT), "-spacing"]).split()
    spacing = [None if value.lower() == "nan" else float(value) for value in spacing_text]
    return {
        "unit": UNIT,
        "dimensions": size,
        "spacing": spacing,
        "cpu_output_size_bytes": CPU_OUTPUT.stat().st_size,
        "cpu_output_sha256": sha256(CPU_OUTPUT),
        "cpu_log_sha256": sha256(CPU_UNIT_ROOT / "logs" / "01_dwifslpreproc.log"),
        "cpu_runtime_note": "Approximately 2 h 00 min 08 s from the frozen CPU run log and file timestamps.",
    }


def execution_contract(runner_sha: str, reference_sha: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "status": "PACKAGE_READY_WAITING_PHASE_A_AND_GPU",
        "purpose": (
            "Test whether one CUDA-Eddy output is sufficiently equivalent to the already completed "
            "CPU-Eddy reference before any wider corrected imaging is considered."
        ),
        "authorization": {
            "static_package_preparation": True,
            "gpu_execution_authorized_now": False,
            "ec2_resize_authorized_by_this_contract": False,
            "wider_imaging_authorized": False,
        },
        "package_bindings": {
            "runner_sha256": runner_sha,
            "reference_manifest_sha256": reference_sha,
        },
        "reference": {
            "unit": UNIT,
            "input": str(CPU_INPUT),
            "source_metadata": str(SOURCE_METADATA),
            "cpu_output": str(CPU_OUTPUT),
            "cpu_qc_dir": str(CPU_QC),
        },
        "tools": {
            "mrtrix_bin": str(MRTRIX_BIN),
            "fsl_dir": str(FSL_DIR),
            "python": "/usr/bin/python3.9",
        },
        "gpu_execution": {
            "eddy_cuda_path": str(CUDA_EDDY),
            "eddy_cuda_sha256": sha256(CUDA_EDDY),
            "cpu_fallback_allowed": False,
            "force_cpu_environment_variable_set": False,
            "no_cpu_fallback_environment_variable": "DWIFSLPREPROC_NO_CPU_FALLBACK=1",
            "required_log_evidence": "eddy_cuda",
            "maximum_units": 1,
            "maximum_concurrent_eddy_processes": 1,
        },
        "gates": {
            "phase_a_service": PHASE_A_SERVICE,
            "phase_a_must_have_terminal_validated_record": True,
            "phase_a_service_must_be_inactive": True,
            "post_resize_attestation_required": True,
            "nvidia_smi_device_required": True,
            "allowed_gpu_instance_types": ["g6.4xlarge", "g5.4xlarge", "g4dn.4xlarge"],
            "output_root_prefix": "/data/derivatives/scforge_v2/gpu_eddy_equivalence_",
            "output_root_must_not_exist": True,
        },
        "scope": {
            "allowed_products": [
                "one_gpu_eddy_output",
                "eddy_qc",
                "cpu_gpu_equivalence_report",
                "tensor_equivalence_metrics",
                "response_equivalence_metrics",
                "runtime_and_peak_gpu_memory",
                "terminal_record",
            ],
            "forbidden_products": [
                "fod",
                "act",
                "tractography",
                "sift2",
                "connectome_matrices",
                "dashboard_refresh",
                "full_cohort",
                "production_overwrite",
            ],
        },
        "equivalence_thresholds": {
            "maximum_spacing_difference_mm": 1e-6,
            "maximum_affine_difference": 1e-5,
            "maximum_bval_difference": 1e-3,
            "maximum_bvec_difference": 0.02,
            "minimum_mask_dice": 0.99,
            "minimum_image_correlation": 0.995,
            "maximum_image_normalized_rmse": 0.10,
            "minimum_mean_signal_ratio": 0.98,
            "maximum_mean_signal_ratio": 1.02,
            "minimum_per_volume_correlation": 0.99,
            "maximum_translation_difference_mm": 0.5,
            "maximum_rotation_difference_radians": 0.01,
            "maximum_outlier_fraction_absolute_difference": 0.02,
            "minimum_tensor_correlation": 0.99,
            "maximum_tensor_normalized_rmse": 0.15,
            "maximum_response_relative_l2_difference": 0.15,
        },
        "threshold_interpretation": (
            "These are prospective engineering equivalence tolerances for a one-unit backend canary, "
            "not biological-effect thresholds and not evidence that CPU and GPU algorithms are identical. "
            "Any failed check closes continuation and requires review."
        ),
        "decision_rule": (
            "Pass only if every geometry, gradient, mask, signal, per-volume, motion, outlier, tensor, "
            "and response check passes and the terminal record proves one CUDA execution with no CPU fallback."
        ),
    }


def schemas() -> dict[str, dict[str, Any]]:
    resize_attestation = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Post-Phase-A GPU resize attestation",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "decision",
            "contract_sha256",
            "allowed_output_root",
            "phase_a_terminal_record",
            "phase_a_terminal_record_sha256",
            "phase_a_terminal_record_validated",
            "observed_instance_type",
        ],
        "properties": {
            "decision": {"const": "APPROVE_ONE_UNIT_GPU_EDDY_EQUIVALENCE"},
            "contract_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "allowed_output_root": {
                "type": "string",
                "pattern": "^/data/derivatives/scforge_v2/gpu_eddy_equivalence_[A-Za-z0-9_.-]+$",
            },
            "phase_a_terminal_record": {"type": "string"},
            "phase_a_terminal_record_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "phase_a_terminal_record_validated": {"const": True},
            "observed_instance_type": {
                "enum": ["g6.4xlarge", "g5.4xlarge", "g4dn.4xlarge"]
            },
        },
    }
    terminal_record = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "One-unit GPU Eddy equivalence terminal record",
        "type": "object",
        "required": [
            "schema_version",
            "started_utc",
            "ended_utc",
            "status",
            "scope",
            "output_root",
            "contract_sha256",
            "duration_seconds",
        ],
        "properties": {
            "status": {"enum": ["PASS", "FAIL"]},
            "scope": {"const": "one_unit_gpu_eddy_equivalence_only"},
            "tractography_or_matrix_products": {"maximum": 0},
        },
    }
    return {
        "resize_attestation.schema.json": resize_attestation,
        "terminal_record.schema.json": terminal_record,
    }


def readme(contract_sha: str) -> str:
    return f"""# One-unit GPU Eddy equivalence package v1

Status: **prepared and statically testable; execution is not authorized**.

This package freezes one completed CPU reference (`{UNIT}`), the exact CUDA Eddy
binary, a GPU-only launcher, prospective equivalence tolerances, and fail-closed
post-Phase-A execution gates. It does not contain a resize action, cohort loop,
tractography, SIFT2, connectome-matrix generation, or dashboard refresh.

## Why this exists

The CPU reference took approximately two hours. A single CUDA canary can measure
runtime and peak VRAM while testing whether geometry, gradients, masks, corrected
DWI signal, Eddy QC, tensor maps, and response functions agree within the frozen
tolerances. Only a complete pass can support a later concurrency decision.

## Static validation (safe on the current CPU host)

```bash
/home/ec2-user/exp/.venv_connectome_workflow/bin/python \\
  {OUTPUT_ROOT}/runner.py \\
  --package {OUTPUT_ROOT} \\
  --static-validate
```

The static check hashes package and reference files, proves that CPU fallback is
prohibited, checks the CUDA binary and its linked libraries, and confirms that
execution is closed on the current host while Phase A is active or no GPU exists.

## Execution gate

Execution remains impossible until all of the following are true:

1. Phase A has terminated and its terminal record has been independently validated.
2. The user has performed the controlled stop/resize/start transition.
3. The live instance type is one of the contract's allowed GPU types.
4. `nvidia-smi` proves that an NVIDIA device is functioning.
5. A new resize attestation, bound to contract SHA-256 `{contract_sha}`, approves
   exactly one unused output root.
6. The frozen CUDA binary still matches its hash.

Do not manufacture or pre-sign the attestation before those facts exist. A failed
CUDA call cannot fall back to CPU. A failed equivalence check closes continuation.

## Decision boundary

Passing this canary would validate the GPU backend only for the frozen one-unit
recipe. It would not authorize a 530-subject rerun, tractography, matrices, or a
biological novelty claim. Those remain separately gated.
"""


def build() -> Path:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen package: {OUTPUT_ROOT}")
    require_sources([RUNNER_SOURCE, CUDA_EDDY, MRTRIX_BIN / "mrinfo"])
    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{OUTPUT_ROOT.name}.building.", dir=OUTPUT_ROOT.parent))
    completed = False
    try:
        runner = temporary / "runner.py"
        shutil.copy2(RUNNER_SOURCE, runner)
        runner.chmod(0o755)

        rows = reference_rows()
        reference_manifest = temporary / "reference_manifest.csv"
        write_reference_manifest(reference_manifest, rows)

        contract = execution_contract(sha256(runner), sha256(reference_manifest))
        contract_path = temporary / "execution_contract.json"
        write_json(contract_path, contract)
        contract_sha = sha256(contract_path)

        write_json(temporary / "cpu_reference_metadata.json", cpu_reference_metadata())
        for name, payload in schemas().items():
            write_json(temporary / name, payload)
        (temporary / "README.md").write_text(readme(contract_sha), encoding="utf-8")

        manifest_names = [
            "README.md",
            "cpu_reference_metadata.json",
            "execution_contract.json",
            "reference_manifest.csv",
            "resize_attestation.schema.json",
            "runner.py",
            "terminal_record.schema.json",
        ]
        manifest = {
            "schema_version": "1.0.0",
            "created_utc": utc_now(),
            "status": "PACKAGE_FROZEN_NOT_AUTHORIZED_FOR_EXECUTION",
            "files": [
                {
                    "relative_path": name,
                    "size_bytes": (temporary / name).stat().st_size,
                    "sha256": sha256(temporary / name),
                }
                for name in manifest_names
            ],
        }
        write_json(temporary / "package_manifest.json", manifest)

        validation = subprocess.run(
            [sys.executable, str(runner), "--package", str(temporary), "--static-validate"],
            text=True,
            capture_output=True,
            check=False,
        )
        if validation.returncode != 0:
            raise RuntimeError(
                "Static validation failed before package freeze:\n"
                + (validation.stdout or "")
                + "\n"
                + (validation.stderr or "")
            )
        report = json.loads(validation.stdout)
        if report.get("status") != "PASS":
            raise RuntimeError(f"Unexpected static validation report: {report}")

        os.replace(temporary, OUTPUT_ROOT)
        completed = True
        write_json(
            OUTPUT_ROOT.parent / "gpu_eddy_equivalence_package_v1_build_record.json",
            {
                "schema_version": "1.0.0",
                "created_utc": utc_now(),
                "status": "PASS",
                "package": str(OUTPUT_ROOT),
                "contract_sha256": contract_sha,
                "package_manifest_sha256": sha256(OUTPUT_ROOT / "package_manifest.json"),
                "runner_static_validation": report,
                "execution_performed": False,
                "resize_performed": False,
            },
        )
        return OUTPUT_ROOT
    finally:
        if not completed and temporary.exists():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    built = build()
    print(built)
