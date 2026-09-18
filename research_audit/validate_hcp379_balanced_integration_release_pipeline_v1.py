#!/home/ec2-user/fsl/bin/python
"""Validate the balanced release assembler and independent verifier."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
ASSEMBLER = (
    EXP
    / "scripts/hcp/"
    "assemble_hcp379_balanced_integration_release_v1.py"
)
VERIFIER = (
    EXP
    / "research_audit/"
    "verify_hcp379_balanced_integration_release_v1.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_balanced_integration_release_pipeline_v1/validation.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run_json(command: list[str]) -> tuple[int, dict[str, Any], str]:
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    try:
        value = json.loads(completed.stdout)
    except Exception:
        value = {}
    return completed.returncode, value, completed.stderr[-2000:]


def main() -> int:
    checks: dict[str, Any] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    compile_probe = subprocess.run(
        [str(PYTHON), "-m", "py_compile", str(ASSEMBLER), str(VERIFIER)],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "sources_compile",
        compile_probe.returncode == 0,
        {
            "returncode": compile_probe.returncode,
            "stderr": compile_probe.stderr[-2000:],
        },
    )
    assembler_self = run_json(
        [str(PYTHON), str(ASSEMBLER), "--self-test"]
    )
    verifier_self = run_json(
        [str(PYTHON), str(VERIFIER), "--self-test"]
    )
    check(
        "assembler_self_test_passes",
        assembler_self[0] == 0
        and assembler_self[1].get("status") == "PASS",
        assembler_self[1] or {"stderr": assembler_self[2]},
    )
    check(
        "verifier_self_test_passes",
        verifier_self[0] == 0
        and verifier_self[1].get("status") == "PASS",
        verifier_self[1] or {"stderr": verifier_self[2]},
    )
    assembler_dry = run_json([str(PYTHON), str(ASSEMBLER)])
    verifier_dry = run_json([str(PYTHON), str(VERIFIER)])
    check(
        "assembler_dry_run_is_non_imaging_and_fail_closed",
        (
            assembler_dry[0] == 0
            and assembler_dry[1].get("status")
            in {"NOT_READY", "READY_TO_ASSEMBLE"}
            and assembler_dry[1].get("subject_count") == 530
            and assembler_dry[1].get("matrix_file_count") == 4770
            and assembler_dry[1].get("diagnosis_labels_used") is False
            and assembler_dry[1].get("source_files_modified") is False
            and assembler_dry[1].get("final_release_authorized") is False
        ),
        assembler_dry[1] or {"stderr": assembler_dry[2]},
    )
    check(
        "verifier_dry_run_is_non_imaging_and_fail_closed",
        (
            verifier_dry[0] == 0
            and verifier_dry[1].get("status")
            in {
                "AWAITING_ASSEMBLED_RELEASE",
                "READY_TO_VERIFY",
                "INVALID_RELEASE_POINTER",
            }
            and verifier_dry[1].get(
                "independent_verification_started"
            )
            is False
            and verifier_dry[1].get("final_release_authorized") is False
        ),
        verifier_dry[1] or {"stderr": verifier_dry[2]},
    )
    assembler_text = ASSEMBLER.read_text(encoding="utf-8")
    verifier_text = VERIFIER.read_text(encoding="utf-8")
    required = (
        "EXPECTED_MATRIX_FILES",
        "EXPECTED_ROUTE_COUNTS",
        "count_invnodevol",
        "MINIMUM_WEIGHTED_SUPPORT_FRACTION",
        "per_subject_streamline_tuning_allowed",
        "subject_exclusion_for_low_density_allowed",
        "PASS_ASSEMBLED_AWAITING_INDEPENDENT_VERIFICATION",
        "archive_primary_compatibility_contract",
    )
    missing_assembler = [
        value for value in required if value not in assembler_text
    ]
    missing_verifier = [
        value for value in required[:4] if value not in verifier_text
    ]
    check(
        "exact_530_by_9_and_qc_contract_is_present",
        not missing_assembler and not missing_verifier,
        {
            "assembler_missing": missing_assembler,
            "verifier_missing": missing_verifier,
        },
    )
    check(
        "independent_verifier_does_not_import_assembler",
        (
            "assemble_hcp379_balanced_integration_release_v1"
            not in verifier_text
            and "importlib" not in verifier_text
        ),
        {},
    )
    prohibited = (
        "rm -rf",
        "shutil.rmtree",
        "diagnosis.csv",
        "DX_bl",
        "--overwrite",
    )
    found = [
        f"{path.name}:{fragment}"
        for path, text in (
            (ASSEMBLER, assembler_text),
            (VERIFIER, verifier_text),
        )
        for fragment in prohibited
        if fragment in text
    ]
    check(
        "no_broad_delete_outcome_or_overwrite_shortcut",
        not found,
        {"found": found},
    )
    passed = sum(row["status"] == "PASS" for row in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_balanced_integration_release_pipeline_v1_validation"
        ),
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "imaging_executed": False,
        "assembler": record(ASSEMBLER),
        "verifier": record(VERIFIER),
        "checks": checks,
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
