#!/home/ec2-user/fsl/bin/python
"""Run the exact five-subject Eddy plus Recovery4 replay on a GPU host.

The package is diagnosis-blind and non-overwriting.  It executes the validated
four-subject raw-source contract followed by the validated one-subject fresh
DICOM contract, then replays the unchanged Recovery4 and compaction gates.
Execution is sequential because the intended g4dn.4xlarge host has one GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
DERIV = Path("/data/derivatives/hcp379_v2")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
BASE_PLAN = (
    EXP
    / "research_audit/outputs/hcp379_eddy_integrity_recovery_plan_v1/"
    "plan.json"
)
BASE_CONTRACT = (
    EXP
    / "research_audit/outputs/hcp379_eddy_acquisition_contract_v1/"
    "contract.json"
)
BASE_CONTRACT_VALIDATION = BASE_CONTRACT.with_name("validation.json")
EXTENSION_ROOT = (
    EXP
    / "research_audit/outputs/hcp379_eddy_extension_005_S_0610_v1"
)
EXTENSION_PLAN = EXTENSION_ROOT / "plan.json"
EXTENSION_CONTRACT = EXTENSION_ROOT / "contract.json"
EXTENSION_VALIDATION = EXTENSION_ROOT / "validation.json"
BASE_EDDY_RUNNER = (
    EXP / "scripts/hcp/run_hcp379_eddy_integrity_recovery_v1.py"
)
EXTENSION_EDDY_RUNNER = (
    EXP / "scripts/hcp/run_hcp379_eddy_extension_005_S_0610_v1.py"
)
BASE_PRETRACT_RUNNER = (
    EXP / "scripts/hcp/run_hcp379_eddy_pretract_recovery4_v1.py"
)
EXTENSION_PRETRACT_RUNNER = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_eddy_extension_005_pretract_recovery4_v1.py"
)
BASE_PRETRACT_ROOT = DERIV / "eddy_integrity_pretract_recovery4"
EXTENSION_PRETRACT_ROOT = (
    DERIV / "eddy_extension_005_pretract_recovery4"
)
RECOVERY_ROOT = DERIV / "eddy_integrity_recovery_v1"
EXPECTED_BASE = {
    "003_S_4373_I378923",
    "003_S_6490_I1043781",
    "003_S_6644_I1083048",
    "006_S_4713_I1483612",
}
EXPECTED_EXTENSION = {"005_S_0610_I906100"}
EXPECTED = EXPECTED_BASE | EXPECTED_EXTENSION


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


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


def gpu_record() -> dict[str, Any]:
    nvidia = Path("/usr/bin/nvidia-smi")
    if not nvidia.is_file():
        return {"ready": False, "reason": "nvidia-smi is missing"}
    result = subprocess.run(
        [str(nvidia), "-L"],
        capture_output=True,
        text=True,
    )
    return {
        "ready": result.returncode == 0 and "GPU" in result.stdout,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def package_validation() -> dict[str, Any]:
    base_plan = load_json(BASE_PLAN)
    base_contract = load_json(BASE_CONTRACT)
    base_validation = load_json(BASE_CONTRACT_VALIDATION)
    extension_plan = load_json(EXTENSION_PLAN)
    extension_contract = load_json(EXTENSION_CONTRACT)
    extension_validation = load_json(EXTENSION_VALIDATION)
    base_units = {
        str(row["unit"]) for row in base_plan.get("targets", [])
    }
    extension_units = {
        str(row["unit"]) for row in extension_plan.get("targets", [])
    }
    scripts = [
        BASE_EDDY_RUNNER,
        EXTENSION_EDDY_RUNNER,
        BASE_PRETRACT_RUNNER,
        EXTENSION_PRETRACT_RUNNER,
    ]
    compile_errors: list[str] = []
    for script in scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except Exception as exc:
            compile_errors.append(f"{script}:{type(exc).__name__}:{exc}")
    checks = {
        "exact_disjoint_five_subject_contract": (
            base_units == EXPECTED_BASE
            and extension_units == EXPECTED_EXTENSION
            and base_units.isdisjoint(extension_units)
            and base_units | extension_units == EXPECTED
        ),
        "both_recovery_plans_pass": (
            base_plan.get("status") == "PASS"
            and base_plan.get("target_n") == 4
            and extension_plan.get("status") == "PASS"
            and extension_plan.get("target_n") == 1
        ),
        "both_acquisition_contracts_pass": (
            base_contract.get("status") == "PASS"
            and base_contract.get("target_n") == 4
            and extension_contract.get("status") == "PASS"
            and extension_contract.get("target_n") == 1
            and base_validation.get("status") == "PASS"
            and extension_validation.get("status") == "PASS"
        ),
        "all_execution_scripts_compile": not compile_errors,
        "all_contracts_are_blind_and_non_overwriting": all(
            plan.get("diagnosis_labels_used") is False
            and plan.get("outcomes_used") is False
            and plan.get("non_overwriting") is True
            and plan.get("historical_eddy_overwrite_allowed") is False
            for plan in (base_plan, extension_plan)
        ),
        "execution_is_sequential_for_one_gpu": True,
    }
    passed = sum(checks.values())
    gpu = gpu_record()
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_five_eddy_gpu_recovery_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "execution_readiness": (
            "READY_FOR_GPU_EXECUTION"
            if passed == len(checks) and gpu["ready"]
            else "WAITING_FOR_GPU"
            if passed == len(checks)
            else "INVALID_PACKAGE"
        ),
        "generated_utc": utc_now(),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": {
            name: "PASS" if value else "FAIL"
            for name, value in checks.items()
        },
        "compile_errors": compile_errors,
        "gpu": gpu,
        "target_n": len(EXPECTED),
        "units": sorted(EXPECTED),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "imaging_executed": False,
        "records": {
            "base_plan": file_record(BASE_PLAN),
            "base_contract": file_record(BASE_CONTRACT),
            "extension_plan": file_record(EXTENSION_PLAN),
            "extension_contract": file_record(EXTENSION_CONTRACT),
            **{
                script.name: file_record(script)
                for script in scripts
            },
        },
    }


def run_step(
    name: str,
    command: list[str],
    attempt: Path,
) -> dict[str, Any]:
    log_path = attempt / f"{name}.log"
    started = utc_now()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND " + json.dumps(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return {
        "name": name,
        "started_utc": started,
        "completed_utc": utc_now(),
        "returncode": result.returncode,
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "log": file_record(log_path),
    }


def postconditions() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for unit in sorted(EXPECTED):
        if unit in EXPECTED_BASE:
            pretract_root = BASE_PRETRACT_ROOT
        else:
            pretract_root = EXTENSION_PRETRACT_ROOT
        eddy_pointer = RECOVERY_ROOT / "subjects" / unit / "result.json"
        state = (
            load_json(eddy_pointer)
            if eddy_pointer.is_file()
            else {}
        )
        pretract_state = (
            pretract_root / "qc" / "subjects" / f"{unit}.json"
        )
        compaction = (
            pretract_root
            / "qc"
            / "pretract_compaction"
            / f"{unit}.json"
        )
        pretract = (
            load_json(pretract_state)
            if pretract_state.is_file()
            else {}
        )
        compact = (
            load_json(compaction) if compaction.is_file() else {}
        )
        passed = (
            state.get("status") == "PASS_EDDY_INTEGRITY_RECOVERY"
            and pretract.get("status") == "PASS_PRETRACT_RECOVERY4"
            and compact.get("status") == "PASS_COMPACTED"
        )
        rows.append(
            {
                "unit": unit,
                "status": "PASS" if passed else "FAIL",
                "eddy_status": state.get("status", "MISSING"),
                "pretract_status": pretract.get("status", "MISSING"),
                "compaction_status": compact.get("status", "MISSING"),
            }
        )
    return {
        "status": (
            "PASS"
            if all(row["status"] == "PASS" for row in rows)
            else "FAIL"
        ),
        "passed_n": sum(row["status"] == "PASS" for row in rows),
        "target_n": len(rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--eddy-nthreads", type=int, default=8)
    parser.add_argument("--pretract-nthreads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.eddy_nthreads <= 16:
        parser.error("--eddy-nthreads must be in 1..16")
    if not 1 <= args.pretract_nthreads <= 16:
        parser.error("--pretract-nthreads must be in 1..16")

    validation = package_validation()
    if args.validate_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0 if validation["status"] == "PASS" else 1
    if validation["execution_readiness"] != "READY_FOR_GPU_EXECUTION":
        raise SystemExit("exact package passes but a working GPU is required")

    attempt = (
        RECOVERY_ROOT
        / "five_subject_gpu_attempts"
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "-five-eddy-gpu"
        )
    )
    attempt.mkdir(parents=True, exist_ok=False)
    commands = [
        (
            "base_four_eddy",
            [
                str(PYTHON),
                str(BASE_EDDY_RUNNER),
                "--execute",
                "--backend",
                "gpu",
                "--nthreads",
                str(args.eddy_nthreads),
            ],
        ),
        (
            "dicom_extension_eddy",
            [
                str(PYTHON),
                str(EXTENSION_EDDY_RUNNER),
                "--execute",
                "--backend",
                "gpu",
                "--nthreads",
                str(args.eddy_nthreads),
            ],
        ),
        (
            "base_four_recovery4",
            [
                str(PYTHON),
                str(BASE_PRETRACT_RUNNER),
                "--execute",
                "--nthreads",
                str(args.pretract_nthreads),
            ],
        ),
        (
            "dicom_extension_recovery4",
            [
                str(PYTHON),
                str(EXTENSION_PRETRACT_RUNNER),
                "--execute",
                "--nthreads",
                str(args.pretract_nthreads),
            ],
        ),
    ]
    steps: list[dict[str, Any]] = []
    for name, command in commands:
        step = run_step(name, command, attempt)
        steps.append(step)
        if step["status"] != "PASS":
            break
    post = postconditions()
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_five_eddy_gpu_recovery_invocation",
        "status": (
            "PASS"
            if len(steps) == len(commands)
            and all(row["status"] == "PASS" for row in steps)
            and post["status"] == "PASS"
            else "FAIL"
        ),
        "generated_utc": utc_now(),
        "attempt_root": str(attempt.resolve()),
        "validation": validation,
        "steps": steps,
        "postconditions": post,
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "historical_outputs_overwritten": False,
    }
    atomic_json(attempt / "invocation.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
