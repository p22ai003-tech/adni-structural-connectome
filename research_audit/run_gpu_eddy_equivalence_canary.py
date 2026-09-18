#!/usr/bin/env python3
"""Fail-closed one-unit CUDA Eddy equivalence canary.

The default action is static validation.  Execution requires a separately
created post-Phase-A resize attestation, a live NVIDIA device, an allowed GPU
instance type, an inactive Phase-A service, and an unused isolated output root.
No tractography, connectome matrix, dashboard, or cohort action is implemented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


ALLOWED_ROOT_PREFIX = Path("/data/derivatives/scforge_v2")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def command(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    if stdout_path is None:
        result = subprocess.run(argv, text=True, capture_output=True, env=env, check=False)
    else:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                argv,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
    if check and result.returncode != 0:
        detail = ""
        if stdout_path is not None and stdout_path.is_file():
            detail = stdout_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        else:
            detail = ((result.stdout or "") + "\n" + (result.stderr or ""))[-4000:]
        raise RuntimeError(f"Command failed ({result.returncode}): {argv!r}\n{detail}")
    return result


def command_with_gpu_monitor(
    argv: list[str],
    *,
    env: dict[str, str],
    stdout_path: Path,
    sample_seconds: float = 5.0,
) -> tuple[float, float, int]:
    """Run one command while sampling total device memory usage."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    nvidia_smi = Path("/usr/bin/nvidia-smi")
    started = time.monotonic()
    peak_memory_mb = 0.0
    samples = 0
    with stdout_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            argv,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        while process.poll() is None:
            if nvidia_smi.is_file():
                sample = command(
                    [
                        str(nvidia_smi),
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                )
                if sample.returncode == 0:
                    for line in (sample.stdout or "").splitlines():
                        try:
                            peak_memory_mb = max(peak_memory_mb, float(line.strip()))
                            samples += 1
                        except ValueError:
                            pass
            time.sleep(sample_seconds)
        returncode = process.wait()
    duration = time.monotonic() - started
    if returncode != 0:
        detail = stdout_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"Command failed ({returncode}): {argv!r}\n{detail}")
    return duration, peak_memory_mb, samples


def validate_package(package: Path) -> dict[str, Any]:
    contract_path = package / "execution_contract.json"
    manifest_path = package / "package_manifest.json"
    reference_path = package / "reference_manifest.csv"
    required = [contract_path, manifest_path, reference_path, package / "runner.py", package / "README.md"]
    checks: dict[str, bool] = {"required_files_present": all(path.is_file() for path in required)}
    failures: list[str] = []
    if not checks["required_files_present"]:
        failures.append("required package files are missing")
        return {"status": "FAIL", "checks": checks, "failures": failures}

    contract = read_json(contract_path)
    manifest = read_json(manifest_path)
    checks["contract_status_ready_not_authorized"] = (
        contract.get("status") == "PACKAGE_READY_WAITING_PHASE_A_AND_GPU"
    )
    checks["manifest_hashes_match"] = True
    for item in manifest.get("files", []):
        path = package / item["relative_path"]
        if not path.is_file() or sha256(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            checks["manifest_hashes_match"] = False
            failures.append(f"package hash/size mismatch: {path}")
    bindings = contract.get("package_bindings", {})
    checks["contract_binds_runner_and_reference_manifest"] = (
        bindings.get("runner_sha256") == sha256(package / "runner.py")
        and bindings.get("reference_manifest_sha256") == sha256(reference_path)
    )

    reference_rows: list[dict[str, str]] = []
    with reference_path.open(newline="", encoding="utf-8") as handle:
        reference_rows = list(csv.DictReader(handle))
    checks["reference_files_hash_match"] = True
    for row in reference_rows:
        path = Path(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"] or path.stat().st_size != int(row["size_bytes"]):
            checks["reference_files_hash_match"] = False
            failures.append(f"reference hash/size mismatch: {path}")

    checks["one_reference_unit_only"] = contract.get("reference", {}).get("unit") == "168_S_6735_I1175371"
    checks["cpu_fallback_prohibited"] = (
        contract.get("gpu_execution", {}).get("cpu_fallback_allowed") is False
        and contract.get("gpu_execution", {}).get("force_cpu_environment_variable_set") is False
        and contract.get("gpu_execution", {}).get("no_cpu_fallback_environment_variable")
        == "DWIFSLPREPROC_NO_CPU_FALLBACK=1"
    )
    forbidden = set(contract.get("scope", {}).get("forbidden_products", []))
    checks["downstream_scope_forbidden"] = {
        "tractography", "sift2", "connectome_matrices", "dashboard_refresh", "full_cohort"
    }.issubset(forbidden)

    service = contract.get("gates", {}).get("phase_a_service", "")
    service_result = command(["systemctl", "--user", "is-active", service], check=False)
    service_state = (service_result.stdout or "").strip() or "unknown"
    nvidia_path = Path("/usr/bin/nvidia-smi")
    nvidia = command([str(nvidia_path), "-L"], check=False) if nvidia_path.is_file() else None
    gpu_visible = bool(nvidia and nvidia.returncode == 0 and "GPU" in (nvidia.stdout or ""))
    checks["execution_gate_closed_on_current_host"] = service_state == "active" or not gpu_visible

    cuda_path = Path(contract.get("gpu_execution", {}).get("eddy_cuda_path", ""))
    checks["cuda_binary_present_and_locked"] = (
        cuda_path.is_file()
        and sha256(cuda_path) == contract.get("gpu_execution", {}).get("eddy_cuda_sha256")
    )
    if cuda_path.is_file():
        ldd = command(["ldd", str(cuda_path)], check=False)
        checks["cuda_binary_dependencies_resolve"] = ldd.returncode == 0 and "not found" not in (ldd.stdout or "")
    else:
        checks["cuda_binary_dependencies_resolve"] = False

    for name, passed in checks.items():
        if not passed and name != "execution_gate_closed_on_current_host":
            failures.append(f"static check failed: {name}")
    return {
        "schema_version": "1.0.0",
        "generated_utc": utc_now(),
        "status": "PASS" if not failures else "FAIL",
        "package": str(package),
        "contract_sha256": sha256(contract_path),
        "service_state": service_state,
        "gpu_visible": gpu_visible,
        "checks": checks,
        "failures": sorted(set(failures)),
    }


def ec2_instance_type() -> str:
    result = command(["/usr/bin/ec2-metadata", "-t"], check=True)
    value = (result.stdout or "").strip()
    if not value.startswith("instance-type:"):
        raise RuntimeError(f"Unexpected EC2 metadata response: {value!r}")
    return value.split(":", 1)[1].strip()


def validate_execution_gates(
    package: Path,
    attestation_path: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    static = validate_package(package)
    if static["status"] != "PASS":
        raise RuntimeError(f"Static package validation failed: {static['failures']}")
    contract_path = package / "execution_contract.json"
    contract = read_json(contract_path)
    attestation = read_json(attestation_path)
    if attestation.get("decision") != "APPROVE_ONE_UNIT_GPU_EDDY_EQUIVALENCE":
        raise RuntimeError("Resize attestation does not approve the one-unit GPU equivalence canary")
    if attestation.get("contract_sha256") != sha256(contract_path):
        raise RuntimeError("Resize attestation is not bound to this execution contract")
    if attestation.get("allowed_output_root") != str(output_root):
        raise RuntimeError("Requested output root differs from resize attestation")

    resolved = output_root.resolve(strict=False)
    expected_prefix = (ALLOWED_ROOT_PREFIX / "gpu_eddy_equivalence_").as_posix()
    if not resolved.as_posix().startswith(expected_prefix):
        raise RuntimeError(f"Output root must begin with {expected_prefix}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")

    terminal_path = Path(attestation.get("phase_a_terminal_record", ""))
    if (
        not terminal_path.is_file()
        or sha256(terminal_path) != attestation.get("phase_a_terminal_record_sha256")
        or attestation.get("phase_a_terminal_record_validated") is not True
    ):
        raise RuntimeError("Phase-A terminal record is missing, changed, or not validated")

    service = contract["gates"]["phase_a_service"]
    state = command(["systemctl", "--user", "is-active", service], check=False)
    if (state.stdout or "").strip() == "active":
        raise RuntimeError("Phase-A service is still active; GPU canary execution is prohibited")

    instance_type = ec2_instance_type()
    if instance_type not in contract["gates"]["allowed_gpu_instance_types"]:
        raise RuntimeError(f"Instance {instance_type} is not an allowed GPU canary type")
    if instance_type != attestation.get("observed_instance_type"):
        raise RuntimeError("Observed instance type differs from resize attestation")

    nvidia_path = Path("/usr/bin/nvidia-smi")
    if not nvidia_path.is_file():
        raise RuntimeError("nvidia-smi is absent after the claimed GPU transition")
    nvidia = command([str(nvidia_path), "-L"], check=False)
    if nvidia.returncode != 0 or "GPU" not in (nvidia.stdout or ""):
        raise RuntimeError("No functioning NVIDIA device is visible")
    cuda_path = Path(contract["gpu_execution"]["eddy_cuda_path"])
    if sha256(cuda_path) != contract["gpu_execution"]["eddy_cuda_sha256"]:
        raise RuntimeError("CUDA Eddy binary changed after package freeze")

    gate_record = {
        "generated_utc": utc_now(),
        "status": "PASS",
        "contract_sha256": sha256(contract_path),
        "attestation_path": str(attestation_path),
        "attestation_sha256": sha256(attestation_path),
        "phase_a_terminal_record": str(terminal_path),
        "phase_a_terminal_record_sha256": sha256(terminal_path),
        "phase_a_service_state": (state.stdout or "").strip() or "inactive",
        "instance_type": instance_type,
        "nvidia_smi_L": (nvidia.stdout or "").strip(),
    }
    return contract, attestation, gate_record


def build_gpu_shim(contract: dict[str, Any], output_root: Path) -> Path:
    shim = output_root / "contract" / "fsl_gpu_path"
    shim.mkdir(parents=True, exist_ok=False)
    cuda = Path(contract["gpu_execution"]["eddy_cuda_path"])
    (shim / cuda.name).symlink_to(cuda)
    entries = list(shim.iterdir())
    if len(entries) != 1 or entries[0].name != cuda.name or not entries[0].is_symlink():
        raise RuntimeError("GPU shim is not exactly the selected CUDA Eddy executable")
    if any(path.name in {"eddy_cpu", "eddy_openmp", "eddy"} for path in entries):
        raise RuntimeError("CPU Eddy executable leaked into GPU-only shim")
    return shim


def mrinfo_json(image: Path, mrtrix_bin: Path, work: Path) -> dict[str, Any]:
    target = work / f"{image.name}.mrinfo.json"
    command([str(mrtrix_bin / "mrinfo"), str(image), "-json_all", str(target)])
    return read_json(target)


def export_gradients(image: Path, mrtrix_bin: Path, prefix: Path) -> tuple[np.ndarray, np.ndarray]:
    bvec = prefix.with_suffix(".bvec")
    bval = prefix.with_suffix(".bval")
    command([str(mrtrix_bin / "mrinfo"), str(image), "-export_grad_fsl", str(bvec), str(bval)])
    return np.loadtxt(bvec), np.loadtxt(bval)


def masked_metrics(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Array shapes differ: {reference.shape} vs {candidate.shape}")
    if reference.ndim == 4:
        use = np.broadcast_to(mask[..., None], reference.shape)
    else:
        use = mask
    ref = np.asarray(reference[use], dtype=np.float64)
    cand = np.asarray(candidate[use], dtype=np.float64)
    finite = np.isfinite(ref) & np.isfinite(cand)
    ref = ref[finite]
    cand = cand[finite]
    if ref.size < 100:
        raise ValueError("Too few finite masked values for equivalence")
    ref_centered = ref - ref.mean()
    cand_centered = cand - cand.mean()
    denominator = math.sqrt(float(np.dot(ref_centered, ref_centered) * np.dot(cand_centered, cand_centered)))
    correlation = float(np.dot(ref_centered, cand_centered) / denominator) if denominator > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((cand - ref) ** 2)))
    scale = float(np.std(ref))
    return {
        "finite_fraction": float(finite.mean()),
        "pearson_correlation": correlation,
        "normalized_rmse": rmse / scale if scale > 0 else float("inf"),
        "mean_signal_ratio": float(cand.mean() / ref.mean()) if ref.mean() != 0 else float("nan"),
        "n_values": int(ref.size),
    }


def response_valid(path: Path) -> bool:
    try:
        values = np.loadtxt(path, ndmin=2)
    except Exception:
        return False
    return bool(values.size > 0 and np.isfinite(values).all() and (values[:, 0] > 0).all())


def compare_outputs(contract: dict[str, Any], output_root: Path) -> dict[str, Any]:
    reference = contract["reference"]
    tools = contract["tools"]
    mrtrix = Path(tools["mrtrix_bin"])
    cpu_dwi = Path(reference["cpu_output"])
    gpu_dwi = output_root / "gpu" / "dwi_preproc_gpu.mif"
    cpu_qc = Path(reference["cpu_qc_dir"])
    gpu_qc = output_root / "gpu" / "eddy_qc"
    compare_dir = output_root / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=False)

    cpu_info = mrinfo_json(cpu_dwi, mrtrix, compare_dir)
    gpu_info = mrinfo_json(gpu_dwi, mrtrix, compare_dir)
    dimensions_equal = cpu_info.get("size") == gpu_info.get("size")
    cpu_spacing = cpu_info.get("spacing")
    gpu_spacing = gpu_info.get("spacing")
    if not isinstance(cpu_spacing, list) or not isinstance(gpu_spacing, list):
        raise ValueError("MRtrix metadata did not include image spacing")
    spacing_diff = float(np.max(np.abs(np.asarray(cpu_spacing[:3], float) - np.asarray(gpu_spacing[:3], float))))
    affine_diff = float(np.max(np.abs(np.asarray(cpu_info.get("transform"), float) - np.asarray(gpu_info.get("transform"), float))))
    cpu_bvec, cpu_bval = export_gradients(cpu_dwi, mrtrix, compare_dir / "cpu")
    gpu_bvec, gpu_bval = export_gradients(gpu_dwi, mrtrix, compare_dir / "gpu")

    cpu_nii = compare_dir / "cpu.nii.gz"
    gpu_nii = compare_dir / "gpu.nii.gz"
    command([str(mrtrix / "mrconvert"), str(cpu_dwi), str(cpu_nii), "-datatype", "float32"])
    command([str(mrtrix / "mrconvert"), str(gpu_dwi), str(gpu_nii), "-datatype", "float32"])
    cpu_data = np.asanyarray(nib.load(cpu_nii).dataobj)
    gpu_data = np.asanyarray(nib.load(gpu_nii).dataobj)
    cpu_mask = np.asanyarray(nib.load(cpu_qc / "eddy_mask.nii").dataobj) > 0
    gpu_mask = np.asanyarray(nib.load(gpu_qc / "eddy_mask.nii").dataobj) > 0
    common_mask = cpu_mask & gpu_mask
    mask_dice = float(2 * common_mask.sum() / (cpu_mask.sum() + gpu_mask.sum()))
    image_metrics = masked_metrics(cpu_data, gpu_data, common_mask)
    volume_correlations = []
    for index in range(cpu_data.shape[3]):
        volume_correlations.append(masked_metrics(cpu_data[..., index], gpu_data[..., index], common_mask)["pearson_correlation"])

    cpu_parameters = np.loadtxt(cpu_qc / "eddy_parameters", ndmin=2)
    gpu_parameters = np.loadtxt(gpu_qc / "eddy_parameters", ndmin=2)
    parameter_shapes_equal = cpu_parameters.shape == gpu_parameters.shape
    if parameter_shapes_equal:
        translation_max = float(np.max(np.abs(cpu_parameters[:, :3] - gpu_parameters[:, :3])))
        rotation_max = float(np.max(np.abs(cpu_parameters[:, 3:6] - gpu_parameters[:, 3:6])))
    else:
        translation_max = rotation_max = float("inf")
    cpu_outlier = np.loadtxt(cpu_qc / "eddy_outlier_map", ndmin=2)
    gpu_outlier = np.loadtxt(gpu_qc / "eddy_outlier_map", ndmin=2)
    outlier_fraction_difference = (
        abs(float(np.mean(cpu_outlier != 0)) - float(np.mean(gpu_outlier != 0)))
        if cpu_outlier.shape == gpu_outlier.shape
        else float("inf")
    )

    tensor_dir = output_root / "downstream_tensor"
    tensor_dir.mkdir(parents=True, exist_ok=False)
    tensor_metrics: dict[str, Any] = {}
    for label, dwi in (("cpu", cpu_dwi), ("gpu", gpu_dwi)):
        tensor = tensor_dir / f"{label}_tensor.mif"
        fa = tensor_dir / f"{label}_fa.mif"
        md = tensor_dir / f"{label}_md.mif"
        command([str(mrtrix / "dwi2tensor"), str(dwi), str(tensor), "-mask", str(cpu_qc / "eddy_mask.nii"), "-nthreads", "4"])
        command([str(mrtrix / "tensor2metric"), str(tensor), "-fa", str(fa), "-adc", str(md), "-nthreads", "4"])
        command([str(mrtrix / "mrconvert"), str(fa), str(tensor_dir / f"{label}_fa.nii.gz"), "-datatype", "float32"])
        command([str(mrtrix / "mrconvert"), str(md), str(tensor_dir / f"{label}_md.nii.gz"), "-datatype", "float32"])
    for metric in ("fa", "md"):
        cpu_metric = np.asanyarray(nib.load(tensor_dir / f"cpu_{metric}.nii.gz").dataobj)
        gpu_metric = np.asanyarray(nib.load(tensor_dir / f"gpu_{metric}.nii.gz").dataobj)
        tensor_metrics[metric] = masked_metrics(cpu_metric, gpu_metric, common_mask)

    response_dir = output_root / "downstream_response"
    response_dir.mkdir(parents=True, exist_ok=False)
    response_results: dict[str, Any] = {}
    response_arrays: dict[str, dict[str, np.ndarray]] = {}
    for label, dwi in (("cpu", cpu_dwi), ("gpu", gpu_dwi)):
        wm = response_dir / f"{label}_wm.txt"
        gm = response_dir / f"{label}_gm.txt"
        csf = response_dir / f"{label}_csf.txt"
        voxels = response_dir / f"{label}_voxels.mif"
        log = response_dir / f"{label}_dwi2response.log"
        command([
            str(mrtrix / "dwi2response"), "dhollander", str(dwi), str(wm), str(gm), str(csf),
            "-mask", str(cpu_qc / "eddy_mask.nii"), "-voxels", str(voxels),
            "-lmax", "0,6", "-config", "BZeroThreshold", "50", "-nthreads", "4",
        ], stdout_path=log)
        response_results[label] = {
            "wm_valid": response_valid(wm),
            "gm_valid": response_valid(gm),
            "csf_valid": response_valid(csf),
            "voxel_selection_present": voxels.is_file() and voxels.stat().st_size > 0,
        }
        response_arrays[label] = {
            "wm": np.loadtxt(wm, ndmin=2),
            "gm": np.loadtxt(gm, ndmin=2),
            "csf": np.loadtxt(csf, ndmin=2),
        }
    response_relative_differences: dict[str, float] = {}
    for tissue in ("wm", "gm", "csf"):
        cpu_array = response_arrays["cpu"][tissue]
        gpu_array = response_arrays["gpu"][tissue]
        if cpu_array.shape != gpu_array.shape:
            response_relative_differences[tissue] = float("inf")
            continue
        denominator = float(np.linalg.norm(cpu_array))
        response_relative_differences[tissue] = (
            float(np.linalg.norm(gpu_array - cpu_array) / denominator)
            if denominator > 0
            else float("inf")
        )

    values = {
        "dimensions_equal": dimensions_equal,
        "maximum_spacing_difference_mm": spacing_diff,
        "maximum_affine_difference": affine_diff,
        "maximum_bval_difference": float(np.max(np.abs(cpu_bval - gpu_bval))),
        "maximum_bvec_difference": float(np.max(np.abs(cpu_bvec - gpu_bvec))),
        "mask_dice": mask_dice,
        "image": image_metrics,
        "minimum_volume_correlation": float(np.min(volume_correlations)),
        "median_volume_correlation": float(np.median(volume_correlations)),
        "eddy_parameter_shapes_equal": parameter_shapes_equal,
        "maximum_translation_difference_mm": translation_max,
        "maximum_rotation_difference_radians": rotation_max,
        "outlier_fraction_absolute_difference": outlier_fraction_difference,
        "tensor": tensor_metrics,
        "response": response_results,
        "response_relative_l2_difference": response_relative_differences,
    }
    limits = contract["equivalence_thresholds"]
    checks = {
        "dimensions_equal": dimensions_equal,
        "spacing_within_tolerance": spacing_diff <= limits["maximum_spacing_difference_mm"],
        "affine_within_tolerance": affine_diff <= limits["maximum_affine_difference"],
        "gradients_within_tolerance": (
            values["maximum_bval_difference"] <= limits["maximum_bval_difference"]
            and values["maximum_bvec_difference"] <= limits["maximum_bvec_difference"]
        ),
        "mask_dice_pass": mask_dice >= limits["minimum_mask_dice"],
        "image_correlation_pass": image_metrics["pearson_correlation"] >= limits["minimum_image_correlation"],
        "image_nrmse_pass": image_metrics["normalized_rmse"] <= limits["maximum_image_normalized_rmse"],
        "image_mean_ratio_pass": limits["minimum_mean_signal_ratio"] <= image_metrics["mean_signal_ratio"] <= limits["maximum_mean_signal_ratio"],
        "volume_correlation_pass": values["minimum_volume_correlation"] >= limits["minimum_per_volume_correlation"],
        "motion_pass": parameter_shapes_equal and translation_max <= limits["maximum_translation_difference_mm"] and rotation_max <= limits["maximum_rotation_difference_radians"],
        "outlier_fraction_pass": outlier_fraction_difference <= limits["maximum_outlier_fraction_absolute_difference"],
        "fa_pass": tensor_metrics["fa"]["pearson_correlation"] >= limits["minimum_tensor_correlation"] and tensor_metrics["fa"]["normalized_rmse"] <= limits["maximum_tensor_normalized_rmse"],
        "md_pass": tensor_metrics["md"]["pearson_correlation"] >= limits["minimum_tensor_correlation"] and tensor_metrics["md"]["normalized_rmse"] <= limits["maximum_tensor_normalized_rmse"],
        "response_acceptability_agrees": (
            all(all(record.values()) for record in response_results.values())
            and all(
                difference <= limits["maximum_response_relative_l2_difference"]
                for difference in response_relative_differences.values()
            )
        ),
    }
    return {
        "schema_version": "1.0.0",
        "generated_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "values": values,
        "thresholds": limits,
    }


def execute(package: Path, attestation: Path, output_root: Path) -> None:
    contract, _, gate_record = validate_execution_gates(package, attestation, output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_root / "execution_gate.json", gate_record)
    started = time.monotonic()
    terminal: dict[str, Any] = {
        "schema_version": "1.0.0",
        "started_utc": utc_now(),
        "status": "FAIL",
        "scope": "one_unit_gpu_eddy_equivalence_only",
        "output_root": str(output_root),
        "contract_sha256": sha256(package / "execution_contract.json"),
    }
    try:
        shim = build_gpu_shim(contract, output_root)
        reference = contract["reference"]
        metadata = read_json(Path(reference["source_metadata"]))
        pe = str(metadata["PhaseEncodingDirection"])
        readout = float(metadata["TotalReadoutTime"])
        gpu_dir = output_root / "gpu"
        gpu_dir.mkdir(parents=True, exist_ok=False)
        log = gpu_dir / "dwifslpreproc_gpu.log"
        output = gpu_dir / "dwi_preproc_gpu.mif"
        qc = gpu_dir / "eddy_qc"
        mrtrix = Path(contract["tools"]["mrtrix_bin"])
        env = os.environ.copy()
        env.pop("DWIFSLPREPROC_FORCE_CPU", None)
        env["DWIFSLPREPROC_NO_CPU_FALLBACK"] = "1"
        env["FSLDIR"] = contract["tools"]["fsl_dir"]
        env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
        env["PATH"] = ":".join([str(mrtrix), str(shim), "/usr/bin", "/bin"])
        argv = [
            "/usr/bin/python3.9", str(mrtrix / "dwifslpreproc"),
            reference["input"], str(output), "-rpe_none", "-pe_dir", pe,
            "-readout_time", f"{readout:.12g}", "-config", "BZeroThreshold", "50",
            "-eddy_options", "--slm=linear --data_is_shelled --repol --cnr_maps --residuals",
            "-eddyqc_all", str(qc), "-nthreads", "4", "-debug",
        ]
        write_json_atomic(output_root / "gpu_command.json", {"argv": argv, "environment_policy": {
            "DWIFSLPREPROC_FORCE_CPU": "UNSET",
            "DWIFSLPREPROC_NO_CPU_FALLBACK": "1",
            "gpu_shim": str(shim),
        }})
        gpu_duration, peak_gpu_memory_mb, gpu_memory_samples = command_with_gpu_monitor(
            argv,
            env=env,
            stdout_path=log,
        )
        log_text = log.read_text(encoding="utf-8", errors="replace")
        if "eddy_cuda" not in log_text:
            raise RuntimeError("GPU execution log does not prove CUDA Eddy selection")
        if "attempting OpenMP version" in log_text or "DWIFSLPREPROC_FORCE_CPU is set" in log_text:
            raise RuntimeError("GPU execution log shows a prohibited CPU route")
        comparison = compare_outputs(contract, output_root)
        write_json_atomic(output_root / "equivalence_report.json", comparison)
        terminal.update({
            "status": "PASS" if comparison["status"] == "PASS" else "FAIL",
            "comparison_status": comparison["status"],
            "gpu_output": {"path": str(output), "sha256": sha256(output), "size_bytes": output.stat().st_size},
            "gpu_log": {"path": str(log), "sha256": sha256(log), "size_bytes": log.stat().st_size},
            "gpu_eddy_duration_seconds": gpu_duration,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "gpu_memory_samples": gpu_memory_samples,
            "tractography_or_matrix_products": 0,
        })
        if comparison["status"] != "PASS":
            raise RuntimeError("CPU-GPU equivalence thresholds did not all pass")
    except Exception as exc:
        terminal["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        terminal["ended_utc"] = utc_now()
        terminal["duration_seconds"] = time.monotonic() - started
        write_json_atomic(output_root / "terminal_record.json", terminal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--static-validate", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resize-attestation", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.execute == args.static_validate:
        parser.error("Select exactly one of --static-validate or --execute")
    if args.static_validate:
        report = validate_package(args.package)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["status"] == "PASS" else 1)
    if args.resize_attestation is None or args.output_root is None:
        parser.error("--execute requires --resize-attestation and --output-root")
    execute(args.package, args.resize_attestation, args.output_root)


if __name__ == "__main__":
    main()
