#!/usr/bin/env python3
"""Run an exact contracted, non-overwriting Eddy integrity recovery.

The runner consumes the independently validated recovery plan and acquisition
contract.  It never modifies the historical Eddy derivative.  Each execution
is preserved in a versioned attempt directory, and a subject becomes eligible
for Recovery4 replay only after zero-volume and raw-tensor physical QC pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
PLAN = (
    EXP
    / "research_audit/outputs/hcp379_eddy_integrity_recovery_plan_v1/"
    "plan.json"
)
ACQUISITION = (
    EXP
    / "research_audit/outputs/hcp379_eddy_acquisition_contract_v1/"
    "contract.json"
)
ACQUISITION_VALIDATION = ACQUISITION.with_name("validation.json")
ROOT = Path(
    "/data/derivatives/hcp379_v2/eddy_integrity_recovery_v1"
)
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
FSL = Path("/home/ec2-user/fsl")
CPU_SHIM = Path(
    "/data/derivatives/scforge_v2/"
    "h04a_r1_recovery_20260719_retry4/contract/fsl_cpu_path"
)
EDDY_CPU = FSL / "bin/eddy_cpu"
EDDY_CUDA = FSL / "bin/eddy_cuda11.0"
EXPECTED_UNITS = {
    "003_S_4373_I378923",
    "003_S_6490_I1043781",
    "003_S_6644_I1083048",
    "006_S_4713_I1483612",
}
ZERO_MAXIMUM_THRESHOLD = 1.0e-6


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def run(
    command: list[str],
    log: Any,
    env: dict[str, str] | None = None,
) -> None:
    log.write(f"\n[{utc_now()}] COMMAND {json.dumps(command)}\n")
    log.flush()
    result = subprocess.run(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed rc={result.returncode}: {command[0]}"
        )


def maxima(path: Path) -> list[float]:
    result = subprocess.run(
        [
            str(MRTRIX / "mrstats"),
            str(path),
            "-output",
            "max",
            "-quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [float(value) for value in result.stdout.split()]
    if not values:
        raise ValueError(f"no per-volume maxima: {path}")
    return values


def volume_integrity(path: Path) -> dict[str, Any]:
    values = maxima(path)
    zero = [
        index
        for index, value in enumerate(values)
        if value <= ZERO_MAXIMUM_THRESHOLD
    ]
    return {
        "volume_n": len(values),
        "zero_volume_n": len(zero),
        "zero_volume_indices": zero,
        "minimum_volume_maximum": min(values),
        "zero_maximum_threshold": ZERO_MAXIMUM_THRESHOLD,
        "gate": "PASS" if not zero else "FAIL",
    }


def load_tensor_module() -> Any:
    path = Path(__file__).with_name("hcp_eddy_tensor_repair_v2.py")
    spec = importlib.util.spec_from_file_location(
        "hcp_eddy_tensor_repair_v2", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import tensor audit: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contracts() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    plan = load_json(PLAN)
    acquisition = load_json(ACQUISITION)
    validation = load_json(ACQUISITION_VALIDATION)
    targets = {
        str(row["unit"]): row for row in plan.get("targets", [])
    }
    metadata = {
        str(row["unit"]): row
        for row in acquisition.get("acquisitions", [])
    }
    if (
        plan.get("status") != "PASS"
        or plan.get("target_n") != len(EXPECTED_UNITS)
        or acquisition.get("status") != "PASS"
        or acquisition.get("target_n") != len(EXPECTED_UNITS)
        or validation.get("status") != "PASS"
        or validation.get("passed_checks")
        != validation.get("total_checks")
        or set(targets) != EXPECTED_UNITS
        or set(metadata) != EXPECTED_UNITS
        or acquisition.get("records", {})
        .get("recovery_plan", {})
        .get("sha256")
        != sha256_file(PLAN)
    ):
        raise ValueError("exact validated recovery contracts required")
    for unit, target in targets.items():
        raw = Path(str(target["raw_source"]["path"]))
        historical = Path(str(target["historical_eddy"]["path"]))
        if (
            not raw.is_file()
            or not historical.is_file()
            or sha256_file(raw) != target["raw_source"]["sha256"]
            or sha256_file(historical)
            != target["historical_eddy"]["sha256"]
            or metadata[unit]["raw_source"]["sha256"]
            != target["raw_source"]["sha256"]
        ):
            raise ValueError(f"{unit}: source binding differs")
    return plan, acquisition, targets, metadata


def cpu_backend() -> dict[str, Any]:
    shim_eddy = CPU_SHIM / "eddy_cpu"
    forbidden = [
        path.name
        for path in CPU_SHIM.iterdir()
        if path.name == "eddy" or path.name.startswith("eddy_cuda")
    ]
    if (
        not EDDY_CPU.is_file()
        or not shim_eddy.is_symlink()
        or shim_eddy.resolve() != EDDY_CPU.resolve()
        or forbidden
    ):
        raise RuntimeError("CPU-only Eddy shim is invalid")
    return {
        "backend": "cpu",
        "eddy_binary": file_record(EDDY_CPU),
        "shim": str(CPU_SHIM.resolve()),
        "gpu_visible": gpu_visible(),
    }


def gpu_visible() -> bool:
    nvidia = Path("/usr/bin/nvidia-smi")
    if not nvidia.is_file():
        return False
    result = subprocess.run(
        [str(nvidia), "-L"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and "GPU" in result.stdout


def gpu_backend() -> dict[str, Any]:
    if not gpu_visible() or not EDDY_CUDA.is_file():
        raise RuntimeError("validated NVIDIA GPU/CUDA Eddy is unavailable")
    ldd = subprocess.run(
        ["ldd", str(EDDY_CUDA)],
        capture_output=True,
        text=True,
    )
    if ldd.returncode or "not found" in ldd.stdout:
        raise RuntimeError("CUDA Eddy dependencies do not resolve")
    nvidia = subprocess.run(
        ["/usr/bin/nvidia-smi", "-L"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "backend": "gpu",
        "eddy_binary": file_record(EDDY_CUDA),
        "nvidia_smi_L": nvidia.stdout.strip(),
        "gpu_visible": True,
    }


def select_backend(requested: str) -> dict[str, Any]:
    if requested == "cpu":
        return cpu_backend()
    if requested == "gpu":
        return gpu_backend()
    return gpu_backend() if gpu_visible() else cpu_backend()


def environment(
    backend: str, attempt: Path
) -> tuple[dict[str, str], Path]:
    env = os.environ.copy()
    env["FSLDIR"] = str(FSL)
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    env["DWIFSLPREPROC_NO_CPU_FALLBACK"] = "1"
    if backend == "cpu":
        env["DWIFSLPREPROC_FORCE_CPU"] = "1"
        shim = CPU_SHIM
    else:
        env.pop("DWIFSLPREPROC_FORCE_CPU", None)
        shim = attempt / "fsl_gpu_path"
        shim.mkdir(parents=True, exist_ok=False)
        (shim / EDDY_CUDA.name).symlink_to(EDDY_CUDA)
    env["PATH"] = ":".join(
        [str(MRTRIX), "/usr/bin", "/bin", str(shim)]
    )
    return env, shim


def prior_pass(
    unit_root: Path,
    plan_sha: str,
    acquisition_sha: str,
) -> dict[str, Any] | None:
    pointer = unit_root / "result.json"
    if not pointer.is_file():
        return None
    result = load_json(pointer)
    state_path = Path(str(result.get("attempt_state", {}).get("path", "")))
    if not state_path.is_file():
        return None
    state = load_json(state_path)
    if (
        state.get("status") == "PASS_EDDY_INTEGRITY_RECOVERY"
        and state.get("plan_sha256") == plan_sha
        and state.get("acquisition_contract_sha256") == acquisition_sha
        and result.get("attempt_state", {}).get("sha256")
        == sha256_file(state_path)
    ):
        return state
    return None


def process(
    target: Mapping[str, Any],
    metadata: Mapping[str, Any],
    backend_record: Mapping[str, Any],
    nthreads: int,
) -> dict[str, Any]:
    unit = str(target["unit"])
    unit_root = ROOT / "subjects" / unit
    plan_sha = sha256_file(PLAN)
    acquisition_sha = sha256_file(ACQUISITION)
    cached = prior_pass(unit_root, plan_sha, acquisition_sha)
    if cached is not None:
        return cached

    attempt_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + f"-{backend_record['backend']}-"
        + uuid.uuid4().hex[:8]
    )
    attempt = unit_root / "attempts" / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    state_path = attempt / "state.json"
    log_path = attempt / "eddy_recovery.log"
    source = Path(str(target["raw_source"]["path"]))
    historical = Path(str(target["historical_eddy"]["path"]))
    historical_sha = sha256_file(historical)
    state: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_eddy_integrity_recovery_attempt",
        "status": "RUNNING",
        "started_utc": utc_now(),
        "unit": unit,
        "attempt_id": attempt_id,
        "attempt_root": str(attempt.resolve()),
        "backend": dict(backend_record),
        "plan_sha256": plan_sha,
        "acquisition_contract_sha256": acquisition_sha,
        "route": target["route"],
        "pe_dir": metadata["pe_dir"],
        "total_readout_time_seconds": metadata[
            "total_readout_time_seconds"
        ],
        "retained_volume_indices": target["retained_volume_indices"],
        "retained_b0_n": target["retained_b0_n"],
        "retained_diffusion_n": target["retained_diffusion_n"],
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "historical_eddy_overwritten": False,
        "historical_eddy": file_record(historical),
        "raw_source": file_record(source),
    }
    atomic_json(state_path, state)
    started = time.monotonic()
    try:
        env, shim = environment(str(backend_record["backend"]), attempt)
        filtered = attempt / "dwi_source_retained.mif"
        denoised = attempt / "dwi_denoised.mif"
        noise = attempt / "noise.mif"
        unringed = attempt / "dwi_denoised_degibbs.mif"
        eddy = attempt / "dwi_preproc.mif"
        eddy_qc = attempt / "eddy_qc"
        scratch = attempt / "scratch"
        scratch.mkdir()
        with log_path.open("a", encoding="utf-8") as log:
            if target["route"] == "DROP_EXACT_ZERO_B0_THEN_REGENERATE_EDDY":
                indices = ",".join(
                    str(value)
                    for value in target["retained_volume_indices"]
                )
                run(
                    [
                        str(MRTRIX / "mrconvert"),
                        str(source),
                        str(filtered),
                        "-coord",
                        "3",
                        indices,
                        "-force",
                    ],
                    log,
                    env,
                )
                working_source = filtered
            else:
                working_source = source
            source_integrity = volume_integrity(working_source)
            expected_volume_n = (
                int(target["retained_b0_n"])
                + int(target["retained_diffusion_n"])
            )
            if (
                source_integrity["gate"] != "PASS"
                or source_integrity["volume_n"] != expected_volume_n
            ):
                raise RuntimeError(
                    "retained source failed exact volume-integrity gate"
                )
            run(
                [
                    str(MRTRIX / "dwidenoise"),
                    str(working_source),
                    str(denoised),
                    "-noise",
                    str(noise),
                    "-nthreads",
                    str(nthreads),
                    "-force",
                ],
                log,
                env,
            )
            run(
                [
                    str(MRTRIX / "mrdegibbs"),
                    str(denoised),
                    str(unringed),
                    "-nthreads",
                    str(nthreads),
                    "-force",
                ],
                log,
                env,
            )
            command = [
                "/usr/bin/python3.9",
                str(MRTRIX / "dwifslpreproc"),
                str(unringed),
                str(eddy),
                "-rpe_none",
                "-pe_dir",
                str(metadata["pe_dir"]),
                "-readout_time",
                f"{float(metadata['total_readout_time_seconds']):.12g}",
                "-config",
                "BZeroThreshold",
                "50",
                "-eddy_options",
                (
                    f"--slm=linear --data_is_shelled --repol "
                    f"--cnr_maps --residuals --nthr={nthreads}"
                ),
                "-eddyqc_all",
                str(eddy_qc),
                "-scratch",
                str(scratch),
                "-nthreads",
                str(nthreads),
                "-debug",
            ]
            run(command, log, env)
            post_integrity = volume_integrity(eddy)
            if (
                post_integrity["gate"] != "PASS"
                or post_integrity["volume_n"] != expected_volume_n
            ):
                raise RuntimeError(
                    "regenerated Eddy failed exact volume-integrity gate"
                )
            tensor = load_tensor_module().tensor_audit(
                eddy,
                attempt / "tensor_audit",
                nthreads,
                log,
            )
        log_text = log_path.read_text(
            encoding="utf-8", errors="replace"
        )
        if backend_record["backend"] == "cpu":
            if "eddy_cpu" not in log_text or "eddy_cuda" in log_text:
                raise RuntimeError("CPU execution proof is absent or mixed")
        else:
            if (
                "eddy_cuda" not in log_text
                or "attempting OpenMP version" in log_text
                or "DWIFSLPREPROC_FORCE_CPU is set" in log_text
            ):
                raise RuntimeError("GPU execution proof is absent or mixed")
        if tensor["gate"] != "PASS":
            raise RuntimeError(
                "regenerated Eddy failed raw-tensor physical QC: "
                + ";".join(tensor["failures"])
            )
        if (
            sha256_file(historical) != historical_sha
            or historical_sha != target["historical_eddy"]["sha256"]
        ):
            raise RuntimeError("historical Eddy derivative changed")
        state.update(
            {
                "status": "PASS_EDDY_INTEGRITY_RECOVERY",
                "completed_utc": utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "backend_shim": str(shim.resolve()),
                "source_after_route": file_record(working_source),
                "source_volume_integrity": source_integrity,
                "eddy_output": file_record(eddy),
                "eddy_qc_path": str(eddy_qc.resolve()),
                "eddy_volume_integrity": post_integrity,
                "tensor_audit": tensor,
                "execution_log": file_record(log_path),
                "historical_eddy_unchanged": True,
            }
        )
    except Exception as exc:
        state.update(
            {
                "status": "FAIL_EDDY_INTEGRITY_RECOVERY",
                "completed_utc": utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error": f"{type(exc).__name__}:{exc}",
                "historical_eddy_unchanged": (
                    historical.is_file()
                    and sha256_file(historical) == historical_sha
                ),
            }
        )
    atomic_json(state_path, state)
    state["records"] = (
        {"execution_log": file_record(log_path)}
        if log_path.is_file()
        else {}
    )
    atomic_json(state_path, state)
    # Bind the final state in the subject-level pointer; a file cannot contain
    # a stable hash of itself.
    pointer = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_eddy_integrity_recovery_pointer",
        "updated_utc": utc_now(),
        "unit": unit,
        "status": state["status"],
        "attempt_state": file_record(state_path),
        "attempt_root": str(attempt.resolve()),
        "historical_eddy_overwritten": False,
    }
    atomic_json(unit_root / "result.json", pointer)
    return state


def static_validation(requested_backend: str) -> dict[str, Any]:
    _, _, targets, metadata = contracts()
    backend = select_backend(requested_backend)
    checks = {
        "exact_target_contracts": (
            set(targets) == EXPECTED_UNITS
            and set(metadata) == EXPECTED_UNITS
        ),
        "backend_is_explicit_and_validated": backend["backend"]
        in {"cpu", "gpu"},
        "all_source_and_historical_hashes_replay": all(
            sha256_file(Path(row["raw_source"]["path"]))
            == row["raw_source"]["sha256"]
            and sha256_file(Path(row["historical_eddy"]["path"]))
            == row["historical_eddy"]["sha256"]
            for row in targets.values()
        ),
        "at_most_one_source_filter_route": (
            sum(
                row["route"]
                == "DROP_EXACT_ZERO_B0_THEN_REGENERATE_EDDY"
                for row in targets.values()
            )
            <= 1
        ),
        "all_acquisition_values_valid": all(
            row["pe_dir"] in {"i", "i-", "j", "j-", "k", "k-"}
            and 0.0
            < float(row["total_readout_time_seconds"])
            < 1.0
            for row in metadata.values()
        ),
        "historical_output_root_is_outside_recovery_root": all(
            not Path(row["historical_eddy"]["path"])
            .resolve()
            .is_relative_to(ROOT.resolve())
            for row in targets.values()
        ),
    }
    passed = sum(checks.values())
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_eddy_integrity_recovery_static_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": utc_now(),
        "requested_backend": requested_backend,
        "selected_backend": backend,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": {
            name: "PASS" if value else "FAIL"
            for name, value in checks.items()
        },
        "imaging_executed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--backend", choices=("auto", "cpu", "gpu"), default="auto"
    )
    parser.add_argument("--subject", action="append")
    parser.add_argument("--nthreads", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.nthreads <= 32:
        raise SystemExit("--nthreads must be in 1..32")
    validation = static_validation(args.backend)
    if args.validate_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0 if validation["status"] == "PASS" else 1
    if validation["status"] != "PASS":
        raise SystemExit("static validation failed")
    _, _, targets, metadata = contracts()
    selected = args.subject or sorted(EXPECTED_UNITS)
    if len(selected) != len(set(selected)) or not set(selected) <= EXPECTED_UNITS:
        raise SystemExit("--subject selection is invalid")
    backend = select_backend(args.backend)
    results = [
        process(targets[unit], metadata[unit], backend, args.nthreads)
        for unit in selected
    ]
    summary = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_eddy_integrity_recovery_invocation",
        "status": (
            "PASS"
            if all(
                row["status"] == "PASS_EDDY_INTEGRITY_RECOVERY"
                for row in results
            )
            else "FAIL"
        ),
        "generated_utc": utc_now(),
        "backend": backend,
        "unit_n": len(results),
        "results": [
            {
                "unit": row["unit"],
                "status": row["status"],
                "attempt_root": row.get("attempt_root"),
                "error": row.get("error", ""),
            }
            for row in results
        ],
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    invocation = (
        ROOT
        / "manifests"
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "-invocation.json"
        )
    )
    atomic_json(invocation, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
