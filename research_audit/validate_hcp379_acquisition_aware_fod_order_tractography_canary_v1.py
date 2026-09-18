#!/home/ec2-user/fsl/bin/python
"""Independently validate the lower-order HCP379 tractography canary."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
HCP = Path("/data/derivatives/hcp379_v2")
OUTPUT_ROOT = (
    HCP / "acquisition_aware_fod_order_tractography_canary_v1"
)
AUDIT_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_acquisition_aware_fod_order_tractography_canary_v1"
)
SUMMARY = AUDIT_ROOT / "summary.json"
VALIDATION = AUDIT_ROOT / "validation.json"
RUNNER = (
    EXP
    / "research_audit/"
    "run_hcp379_acquisition_aware_fod_order_tractography_canary_v1.py"
)
TCK_COUNTER = EXP / "scripts/hcp/tck_header_count_v1.py"
FSL_PYTHON = Path("/home/ec2-user/fsl/bin/python")
TARGETS = {
    "031_S_0618_I1229293": 2,
    "031_S_4721_I1093826": 4,
}
SEEDS = ("primary", "independent")
RECIPE_ID = "connectome-v2.1.0-closed-world-canary-candidate"
PER_SEED = 3_000_000
BALANCED_TOTAL = 6_000_000
NODES = 379
DENSITY_TARGET = 0.60
MINIMUM_ENDPOINT_ASSIGNMENT_FRACTION = 0.50
PREFERRED_ENDPOINT_ASSIGNMENT_FRACTION = 0.85


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or resolved.stat().st_size <= 0
    ):
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def verify_record(record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if file_record(path) != {
        "path": str(path),
        "size_bytes": int(record.get("size_bytes", -1)),
        "sha256": str(record.get("sha256", "")),
    }:
        raise ValueError(f"record differs: {path}")
    return path


def tck_count(path: Path) -> int:
    completed = subprocess.run(
        [str(FSL_PYTHON), str(TCK_COUNTER), str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip())


def load_assignments(path: Path) -> np.ndarray:
    values = np.loadtxt(path, comments="#", dtype=np.int16)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape != (PER_SEED, 2):
        raise ValueError(f"assignment shape differs: {path}")
    if np.any(values < 0) or np.any(values > NODES):
        raise ValueError(f"assignment endpoints differ: {path}")
    return values


def reconstruct_count(assignments: np.ndarray) -> np.ndarray:
    left = assignments[:, 0].astype(np.int32, copy=False)
    right = assignments[:, 1].astype(np.int32, copy=False)
    valid = (left > 0) & (right > 0) & (left != right)
    left = left[valid] - 1
    right = right[valid] - 1
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    linear = low * NODES + high
    upper = np.bincount(
        linear, minlength=NODES * NODES
    ).reshape(NODES, NODES)
    matrix = upper + upper.T
    np.fill_diagonal(matrix, 0)
    return matrix.astype(np.int64, copy=False)


def load_count(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",")
    if (
        matrix.shape != (NODES, NODES)
        or not np.isfinite(matrix).all()
        or np.any(matrix < 0)
        or not np.array_equal(matrix, matrix.T)
        or not np.all(np.diag(matrix) == 0)
        or not np.allclose(
            matrix, np.rint(matrix), atol=1.0e-6, rtol=0.0
        )
    ):
        raise ValueError(f"matrix invariants differ: {path}")
    return np.rint(matrix).astype(np.int64)


def matrix_qc(matrix: np.ndarray) -> dict[str, Any]:
    triangle = matrix[np.triu_indices(NODES, 1)]
    supported = int(np.count_nonzero(triangle > 0))
    possible = NODES * (NODES - 1) // 2
    connected = int(np.count_nonzero(np.sum(matrix, axis=1) > 0))
    return {
        "supported_edges": supported,
        "possible_edges": possible,
        "edge_density": supported / possible,
        "connected_nodes": connected,
        "minimum_connected_nodes_met": connected >= 360,
        "preferred_connected_nodes_met": connected >= 376,
        "all_379_nodes_connected": connected == NODES,
        "total_assigned_nondiagonal_streamlines": int(
            np.sum(triangle)
        ),
    }


def support_stability(
    primary: np.ndarray, independent: np.ndarray
) -> dict[str, Any]:
    triangle = np.triu_indices(NODES, 1)
    first = primary[triangle] > 0
    second = independent[triangle] > 0
    union = first | second
    overlap = first & second
    union_n = int(np.count_nonzero(union))
    return {
        "primary_supported_edges": int(np.count_nonzero(first)),
        "independent_supported_edges": int(np.count_nonzero(second)),
        "overlap_supported_edges": int(np.count_nonzero(overlap)),
        "union_supported_edges": union_n,
        "support_jaccard": (
            float(np.count_nonzero(overlap)) / union_n
            if union_n
            else 0.0
        ),
        "primary_only_supported_edges": int(
            np.count_nonzero(first & ~second)
        ),
        "independent_only_supported_edges": int(
            np.count_nonzero(second & ~first)
        ),
    }


def close_dict(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    if set(observed) != set(expected):
        return False
    for key, left in observed.items():
        right = expected[key]
        if isinstance(left, float) or isinstance(right, float):
            if not np.isclose(
                float(left), float(right), atol=1.0e-12, rtol=1.0e-12
            ):
                return False
        elif left != right:
            return False
    return True


def expected_seed(
    unit_spec: Mapping[str, Any], seed_class: str
) -> tuple[str, int]:
    base = (
        f"{unit_spec['dti_source_id']}|"
        f"{unit_spec['dti_raw_bundle_sha256']}|{RECIPE_ID}"
    )
    token = (
        base
        if seed_class == "primary"
        else f"{base}|independent-replicate-1"
    )
    value = (
        int.from_bytes(
            hashlib.sha256(token.encode("utf-8")).digest()[:4],
            "big",
        )
        & 0x7FFFFFFF
    ) or 1
    return token, value


def validate() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    compile_result = subprocess.run(
        [str(FSL_PYTHON), "-m", "py_compile", str(RUNNER)],
        capture_output=True,
        text=True,
        check=False,
    )
    self_test_result = subprocess.run(
        [str(FSL_PYTHON), str(RUNNER), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
    )
    self_test = (
        json.loads(self_test_result.stdout)
        if self_test_result.returncode == 0
        and self_test_result.stdout.strip()
        else {}
    )
    source_ok = (
        compile_result.returncode == 0
        and self_test_result.returncode == 0
        and self_test.get("status") == "PASS"
        and all(self_test.get("checks", {}).values())
    )
    checks["implementation_compiles_and_self_tests"] = {
        "status": "PASS" if source_ok else "FAIL",
        "evidence": {
            "compile_returncode": compile_result.returncode,
            "self_test": self_test,
        },
    }

    summary = load_json(SUMMARY)
    attempt = Path(str(summary.get("attempt_root", ""))).resolve()
    attempt_summary = attempt / "summary.json"
    preflight_path = attempt / "preflight.json"
    state_path = attempt / "state.json"
    structure_ok = (
        summary.get("record_type")
        == (
            "hcp379_acquisition_aware_fod_order_tractography_"
            "canary_summary"
        )
        and summary.get("status")
        in {
            "PASS_TECHNICAL_BOTH_ORDERS_DENSITY_SCREEN_MET",
            "PASS_TECHNICAL_BOTH_ORDERS_DENSITY_SCREEN_NOT_MET",
            "FAIL_LOWER_ORDER_TRACTOGRAPHY_TECHNICAL_CANARY",
        }
        and summary.get("target_n") == 2
        and summary.get("per_seed_streamlines") == PER_SEED
        and summary.get("balanced_total_streamlines")
        == BALANCED_TOTAL
        and attempt.parent == (OUTPUT_ROOT / "attempts").resolve()
        and attempt.is_dir()
        and not attempt.is_symlink()
        and load_json(attempt_summary) == summary
    )
    checks["summary_and_attempt_structure_are_exact"] = {
        "status": "PASS" if structure_ok else "FAIL",
        "evidence": {
            "summary_status": summary.get("status"),
            "attempt_root": str(attempt),
            "target_n": summary.get("target_n"),
        },
    }

    preflight = load_json(preflight_path)
    state = load_json(state_path)
    units = {
        str(row.get("unit")): row
        for row in preflight.get("units", [])
        if isinstance(row, Mapping)
    }
    bindings_ok = (
        preflight.get("status") == "EXECUTION_INPUTS_BOUND"
        and preflight.get("production_recipe_selected") is False
        and preflight.get("scale_up_authorized") is False
        and preflight.get("all_nine_matrix_generation_authorized")
        is False
        and set(units) == set(TARGETS)
        and all(
            int(units[unit].get("recommended_lmax", -1)) == lmax
            for unit, lmax in TARGETS.items()
        )
        and state.get("status") == "COMPLETE"
        and not state.get("remaining_jobs")
        and len(state.get("completed_jobs", [])) == 4
        and verify_record(summary["preflight"]) == preflight_path
        and verify_record(summary["state"]) == state_path
        and verify_record(summary["implementation"]) == RUNNER.resolve()
    )
    checks["input_bindings_and_terminal_state_replay"] = {
        "status": "PASS" if bindings_ok else "FAIL",
        "evidence": {
            "units": sorted(units),
            "completed_jobs": state.get("completed_jobs"),
            "remaining_jobs": state.get("remaining_jobs"),
        },
    }

    seed_failures: list[str] = []
    matrices: dict[tuple[str, str], np.ndarray] = {}
    replay_rows: list[dict[str, Any]] = []
    for unit in sorted(TARGETS):
        spec = units[unit]
        for seed_class in SEEDS:
            result_path = (
                attempt
                / "subjects"
                / unit
                / seed_class
                / "result.json"
            )
            result = load_json(result_path)
            try:
                artifacts = result["artifacts"]
                tracks = verify_record(artifacts["tractogram"])
                metadata_path = verify_record(
                    artifacts["tractography_parameters"]
                )
                count_path = verify_record(artifacts["count_matrix"])
                assignments_path = verify_record(
                    artifacts["assignments"]
                )
                verify_record(artifacts["tckgen_log"])
                verify_record(artifacts["tck2connectome_log"])
                metadata = load_json(metadata_path)
                token, seed = expected_seed(spec, seed_class)
                if (
                    metadata.get("unit") != unit
                    or metadata.get("seed_class") != seed_class
                    or metadata.get("seed_id") != token
                    or int(metadata.get("rng_seed", -1)) != seed
                    or metadata.get("recipe_id") != RECIPE_ID
                    or metadata.get("algorithm") != "iFOD2"
                    or metadata.get("act") is not True
                    or metadata.get("backtrack") is not True
                    or metadata.get("crop_at_gmwmi") is not True
                    or metadata.get("seeding") != "seed_dynamic"
                    or float(metadata.get("cutoff", -1)) != 0.06
                    or int(metadata.get("actual_streamlines", -1))
                    != PER_SEED
                    or tck_count(tracks) != PER_SEED
                ):
                    raise ValueError("metadata or tract count differs")
                assignments = load_assignments(assignments_path)
                matrix = load_count(count_path)
                reconstructed = reconstruct_count(assignments)
                if not np.array_equal(matrix, reconstructed):
                    raise ValueError("assignment replay differs")
                fraction = float(
                    np.mean(
                        (assignments[:, 0] > 0)
                        & (assignments[:, 1] > 0)
                    )
                )
                qc = matrix_qc(matrix)
                expected_failures: list[str] = []
                if fraction < MINIMUM_ENDPOINT_ASSIGNMENT_FRACTION:
                    expected_failures.append(
                        "endpoint_assignment_fraction_below_minimum"
                    )
                if not qc["minimum_connected_nodes_met"]:
                    expected_failures.append(
                        "connected_nodes_below_minimum"
                    )
                expected_status = (
                    "PASS_TECHNICAL_SEED"
                    if not expected_failures
                    else "FAIL_TECHNICAL_SEED"
                )
                if (
                    result.get("status") != expected_status
                    or result.get("failures") != expected_failures
                    or not np.isclose(
                        fraction,
                        float(
                            result["endpoint_assignment_fraction"]
                        ),
                        atol=1.0e-12,
                        rtol=1.0e-12,
                    )
                    or bool(
                        result["minimum_endpoint_assignment_met"]
                    )
                    != bool(
                        fraction
                        >= MINIMUM_ENDPOINT_ASSIGNMENT_FRACTION
                    )
                    or bool(
                        result["preferred_endpoint_assignment_met"]
                    )
                    != bool(
                        fraction
                        >= PREFERRED_ENDPOINT_ASSIGNMENT_FRACTION
                    )
                    or not close_dict(qc, result["technical_qc"])
                ):
                    raise ValueError("reported seed QC differs")
                matrices[(unit, seed_class)] = matrix
                replay_rows.append(
                    {
                        "unit": unit,
                        "seed_class": seed_class,
                        "endpoint_assignment_fraction": fraction,
                        **qc,
                    }
                )
            except Exception as exc:
                seed_failures.append(
                    f"{unit}/{seed_class}:{type(exc).__name__}:{exc}"
                )
    seeds_replay_ok = not seed_failures and len(matrices) == 4
    checks["all_four_seed_outputs_replay_independently"] = {
        "status": "PASS" if seeds_replay_ok else "FAIL",
        "evidence": {
            "replayed_n": len(matrices),
            "rows": replay_rows,
            "failures": seed_failures,
        },
    }

    balanced_failures: list[str] = []
    balanced_rows: list[dict[str, Any]] = []
    summary_units = {
        str(row.get("unit")): row
        for row in summary.get("units", [])
        if isinstance(row, Mapping)
    }
    if seeds_replay_ok:
        for unit in sorted(TARGETS):
            try:
                reported = summary_units[unit]["balanced_6m"]
                primary_result = load_json(
                    attempt
                    / "subjects"
                    / unit
                    / "primary"
                    / "result.json"
                )
                independent_result = load_json(
                    attempt
                    / "subjects"
                    / unit
                    / "independent"
                    / "result.json"
                )
                seed_pair_passed = (
                    primary_result.get("status")
                    == "PASS_TECHNICAL_SEED"
                    and independent_result.get("status")
                    == "PASS_TECHNICAL_SEED"
                )
                if not seed_pair_passed:
                    if reported is not None:
                        raise ValueError(
                            "failed seed pair has balanced matrix"
                        )
                    balanced_rows.append(
                        {
                            "unit": unit,
                            "seed_pair_technically_eligible": False,
                            "balanced_matrix_expected": False,
                            "balanced_matrix_present": False,
                        }
                    )
                    continue
                balanced_path = verify_record(
                    reported["count_matrix"]
                )
                recorded = load_count(balanced_path)
                expected = (
                    matrices[(unit, "primary")]
                    + matrices[(unit, "independent")]
                )
                qc = matrix_qc(expected)
                stability = support_stability(
                    matrices[(unit, "primary")],
                    matrices[(unit, "independent")],
                )
                if (
                    not np.array_equal(recorded, expected)
                    or not close_dict(qc, reported["technical_qc"])
                    or not close_dict(
                        stability, reported["support_stability"]
                    )
                    or bool(reported["density_target_0_60_met"])
                    != bool(qc["edge_density"] >= DENSITY_TARGET)
                ):
                    raise ValueError("balanced replay differs")
                balanced_rows.append(
                    {
                        "unit": unit,
                        "seed_pair_technically_eligible": True,
                        "balanced_matrix_expected": True,
                        "balanced_matrix_present": True,
                        "density_target_0_60_met": bool(
                            qc["edge_density"] >= DENSITY_TARGET
                        ),
                        **qc,
                        **stability,
                    }
                )
            except Exception as exc:
                balanced_failures.append(
                    f"{unit}:{type(exc).__name__}:{exc}"
                )
    balanced_ok = (
        seeds_replay_ok
        and not balanced_failures
        and len(balanced_rows) == len(TARGETS)
    )
    checks["balanced_6m_matrices_and_density_replay"] = {
        "status": "PASS" if balanced_ok else "FAIL",
        "evidence": {
            "rows": balanced_rows,
            "failures": balanced_failures,
        },
    }

    reported_technical_pass = sum(
        row.get("status")
        == "PASS_TECHNICAL_LOWER_ORDER_CONNECTOME"
        for row in summary_units.values()
    )
    reported_density_pass = sum(
        bool(
            row.get("balanced_6m")
            and row["balanced_6m"].get(
                "density_target_0_60_met"
            )
        )
        for row in summary_units.values()
    )
    decision_ok = (
        set(summary_units) == set(TARGETS)
        and summary.get("technical_pass_n") == reported_technical_pass
        and summary.get("balanced_density_target_pass_n")
        == reported_density_pass
        and summary.get("production_recipe_selected") is False
        and summary.get("scale_up_authorized") is False
        and summary.get("all_nine_matrices_generated") is False
        and summary.get("decision", {}).get(
            "promote_all_ten_lower_order_units"
        )
        is False
        and summary.get("decision", {}).get(
            "select_production_streamline_count"
        )
        is False
    )
    checks["decision_is_fail_closed_and_non_authorizing"] = {
        "status": "PASS" if decision_ok else "FAIL",
        "evidence": {
            "technical_pass_n": reported_technical_pass,
            "density_pass_n": reported_density_pass,
            "decision": summary.get("decision"),
        },
    }

    blind_ok = (
        summary.get("diagnosis_labels_used") is False
        and summary.get("outcomes_used") is False
        and summary.get("group_effects_used") is False
        and summary.get("non_overwriting") is True
        and summary.get("historical_outputs_modified") is False
        and preflight.get("diagnosis_labels_used") is False
        and preflight.get("outcomes_used") is False
        and preflight.get("group_effects_used") is False
        and preflight.get("non_overwriting") is True
        and preflight.get("historical_outputs_modified") is False
    )
    checks["blind_non_overwriting_scope_is_explicit"] = {
        "status": "PASS" if blind_ok else "FAIL",
        "evidence": {
            "diagnosis_labels_used": summary.get(
                "diagnosis_labels_used"
            ),
            "outcomes_used": summary.get("outcomes_used"),
            "attempt_root": str(attempt),
        },
    }

    passed = sum(row["status"] == "PASS" for row in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_acquisition_aware_fod_order_tractography_"
            "canary_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": utc_now(),
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "summary": file_record(SUMMARY),
        "implementation": file_record(Path(__file__)),
    }
    atomic_json(VALIDATION, result)
    return result


def main() -> int:
    value = validate()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
