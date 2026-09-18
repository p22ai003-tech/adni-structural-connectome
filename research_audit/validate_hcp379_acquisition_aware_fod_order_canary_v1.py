#!/home/ec2-user/fsl/bin/python
"""Independently validate the acquisition-aware FOD-order canary."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP
    / "research_audit/"
    "run_hcp379_acquisition_aware_fod_order_canary_v1.py"
)
SUMMARY = (
    EXP
    / "research_audit/outputs/"
    "hcp379_acquisition_aware_fod_order_canary_v1/summary.json"
)
OUTPUT = SUMMARY.with_name("validation.json")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
EXPECTED = {
    "031_S_0618_I1229293": (2, 6),
    "031_S_4721_I1093826": (4, 15),
}


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


def record_matches(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    path = Path(str(value.get("path", "")))
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == int(value.get("size_bytes", -1))
        and sha256(path) == value.get("sha256")
    )


def main() -> int:
    summary = load_json(SUMMARY)
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    compile_probe = subprocess.run(
        [str(PYTHON), "-m", "py_compile", str(SOURCE)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    self_test = subprocess.run(
        [str(PYTHON), str(SOURCE), "--self-test"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    check(
        "implementation_compiles_and_self_test_passes",
        (
            compile_probe.returncode == 0
            and self_test.returncode == 0
            and "HCP379_ACQUISITION_AWARE_FOD_ORDER_CANARY_SELF_TEST_PASS"
            in self_test.stdout
        ),
        {
            "compile_returncode": compile_probe.returncode,
            "compile_stderr": compile_probe.stderr,
            "self_test_returncode": self_test.returncode,
            "self_test_stdout": self_test.stdout,
        },
    )
    rows = summary.get("units", [])
    by_unit = {
        str(row["unit"]): row
        for row in rows
        if isinstance(row, dict) and row.get("unit")
    }
    check(
        "exact_two_prespecified_units",
        (
            summary.get("target_n") == 2
            and summary.get("passed_n") == 2
            and set(by_unit) == set(EXPECTED)
        ),
        {
            "target_n": summary.get("target_n"),
            "passed_n": summary.get("passed_n"),
            "units": sorted(by_unit),
        },
    )
    order_errors: list[str] = []
    numerical_errors: list[str] = []
    response_errors: list[str] = []
    artifact_errors: list[str] = []
    for unit, (expected_lmax, expected_coefficients) in EXPECTED.items():
        row = by_unit.get(unit, {})
        if (
            row.get("status") != "PASS"
            or int(row.get("recommended_lmax", -1)) != expected_lmax
            or int(row.get("wmfod_coefficient_n", -1))
            != expected_coefficients
        ):
            order_errors.append(unit)
        if row.get("failures") not in ([], None):
            numerical_errors.append(f"{unit}:failures")
        factors = row.get("balance_factors", [])
        if (
            len(factors) != 3
            or not all(float(value) > 0 for value in factors)
        ):
            numerical_errors.append(f"{unit}:balance")
        for key, value in row.get(
            "soft_negative_fractions", {}
        ).items():
            if float(value) > 1.0e-3:
                numerical_errors.append(f"{unit}:{key}")
        for key, value in row.get("numerical_counts", {}).items():
            if key.endswith("_below_hard_tolerance") and int(value):
                numerical_errors.append(f"{unit}:{key}")
        result_record = row.get("result")
        if not record_matches(result_record):
            artifact_errors.append(f"{unit}:result")
            continue
        result = load_json(Path(str(result_record["path"])))
        if result.get("unit") != unit or result.get("status") != "PASS":
            artifact_errors.append(f"{unit}:result-content")
        derivation = result.get("wm_response_derivation", {})
        expected_retained = expected_lmax // 2 + 1
        if (
            derivation.get("source_coefficient_n") != 4
            or derivation.get("retained_coefficient_n")
            != expected_retained
            or derivation.get("retained_orders")
            != list(range(0, expected_lmax + 1, 2))
            or any(
                replay.get("derived")
                != replay.get("source", [])[:expected_retained]
                for replay in derivation.get("values_replay", [])
            )
            or len(derivation.get("values_replay", [])) != 2
            or not record_matches(derivation.get("source"))
            or not record_matches(derivation.get("derived"))
        ):
            response_errors.append(unit)
        for name, artifact in result.get("artifacts", {}).items():
            if not record_matches(artifact):
                artifact_errors.append(f"{unit}:{name}")
        if not record_matches(result.get("log")):
            artifact_errors.append(f"{unit}:log")
    check(
        "orders_and_output_coefficient_counts_are_exact",
        not order_errors,
        {"errors": order_errors},
    )
    check(
        "unchanged_fod_numerical_policy_passes",
        not numerical_errors,
        {"errors": numerical_errors},
    )
    check(
        "pooled_wm_response_is_only_order_truncated",
        not response_errors,
        {"errors": response_errors},
    )
    check(
        "all_subject_results_and_artifacts_hash_replay",
        not artifact_errors,
        {"errors": artifact_errors},
    )
    check(
        "canary_is_blind_non_overwriting_and_pretract_only",
        (
            summary.get("diagnosis_labels_used") is False
            and summary.get("outcomes_used") is False
            and summary.get("connectome_density_used") is False
            and summary.get("non_overwriting") is True
            and summary.get("historical_outputs_modified") is False
            and summary.get("tractography_generated") is False
            and summary.get("matrices_generated") is False
        ),
        {
            key: summary.get(key)
            for key in (
                "diagnosis_labels_used",
                "outcomes_used",
                "connectome_density_used",
                "non_overwriting",
                "historical_outputs_modified",
                "tractography_generated",
                "matrices_generated",
            )
        },
    )
    passed = sum(row["status"] == "PASS" for row in checks.values())
    validation = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_acquisition_aware_fod_order_canary_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(validation, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
