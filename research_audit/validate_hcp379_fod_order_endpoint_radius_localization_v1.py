#!/home/ec2-user/fsl/bin/python
"""Independently validate the lmax4 endpoint-radius localization."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP
    / "research_audit/"
    "run_hcp379_fod_order_endpoint_radius_localization_v1.py"
)
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/"
    "acquisition_aware_fod_order_endpoint_radius_localization_v1/"
    "attempts"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_fod_order_endpoint_radius_localization_v1/"
    "validation.json"
)
SEEDS = ("primary", "independent")
RADII = (4, 6, 8)
NODES = 379
STREAMLINES = 3_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def verify_record(module: Any, record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    expected = {
        "path": str(path),
        "size_bytes": int(record.get("size_bytes", -1)),
        "sha256": str(record.get("sha256", "")),
    }
    if module.BASE.BALANCED.file_record(path) != expected:
        raise ValueError(f"artifact record differs: {path}")
    return path


def latest_complete_attempt() -> Path:
    candidates = []
    for path in ATTEMPTS.glob("*-fod-order-endpoint-radius-localization"):
        summary = path / "summary.json"
        if summary.is_file():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("no completed localization attempt")
    return sorted(candidates)[-1]


def main() -> int:
    module = load_module(SOURCE, "hcp379_fod_radius_validation_target")
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    compile_result = subprocess.run(
        ["/home/ec2-user/fsl/bin/python", "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    self_test = subprocess.run(
        ["/home/ec2-user/fsl/bin/python", str(SOURCE), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
    )
    add(
        "source_compiles_and_self_test_passes",
        compile_result.returncode == 0
        and self_test.returncode == 0
        and self_test.stdout.strip()
        == "HCP379_FOD_ORDER_ENDPOINT_RADIUS_SELF_TEST_PASS",
        {
            "compile_returncode": compile_result.returncode,
            "compile_stderr": compile_result.stderr,
            "self_test_returncode": self_test.returncode,
            "self_test_stdout": self_test.stdout.strip(),
            "self_test_stderr": self_test.stderr,
        },
    )

    dry = subprocess.run(
        ["/home/ec2-user/fsl/bin/python", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    dry_value = (
        json.loads(dry.stdout)
        if dry.returncode == 0 and dry.stdout.strip()
        else {}
    )
    add(
        "default_is_nonexecuting_exact_scope_preflight",
        dry.returncode == 0
        and dry_value.get("status")
        == "READY_ENDPOINT_RADIUS_LOCALIZATION"
        and dry_value.get("unit") == module.UNIT
        and dry_value.get("reproducible_disconnected_node_n") == 29
        and dry_value.get("diagnostic_radii_mm") == [6, 8]
        and dry_value.get("new_tractography_generated") is False
        and dry_value.get("scale_up_authorized") is False,
        {
            "returncode": dry.returncode,
            "status": dry_value.get("status"),
            "unit": dry_value.get("unit"),
            "disconnected_node_n": dry_value.get(
                "reproducible_disconnected_node_n"
            ),
            "diagnostic_radii_mm": dry_value.get(
                "diagnostic_radii_mm"
            ),
            "stderr": dry.stderr,
        },
    )

    attempt = latest_complete_attempt()
    summary = load_json(attempt / "summary.json")
    preflight = load_json(attempt / "preflight.json")
    records_ok = False
    record_error = ""
    try:
        records_ok = (
            verify_record(module, summary["preflight"])
            == (attempt / "preflight.json").resolve()
            and verify_record(module, summary["implementation"])
            == SOURCE.resolve()
            and verify_record(module, preflight["implementation"])
            == SOURCE.resolve()
        )
    except Exception as exc:
        record_error = f"{type(exc).__name__}:{exc}"
    add(
        "attempt_and_implementation_are_hash_bound",
        records_ok,
        {
            "attempt": str(attempt),
            "record_error": record_error,
        },
    )

    replay_failures: list[str] = []
    replay_rows: list[dict[str, Any]] = []
    disconnected_sets: dict[tuple[str, int], list[int]] = {}
    for seed in SEEDS:
        for radius in RADII:
            level = summary["levels"][seed][str(radius)]
            try:
                count_path = verify_record(
                    module, level["count_matrix"]
                )
                assignments_path = verify_record(
                    module, level["assignments"]
                )
                matrix = module.BASE.BALANCED.load_count(count_path)
                assignments = module.BASE.BALANCED.load_assignments(
                    assignments_path
                )
                reconstructed = module.BASE.BALANCED.count_matrix(
                    assignments, STREAMLINES
                )
                if not np.array_equal(matrix, reconstructed):
                    raise ValueError("assignment replay differs")
                qc = module.BASE.BALANCED.matrix_qc(matrix)
                missing = module.disconnected(matrix)
                fraction = module.BASE.BALANCED.assignment_fraction(
                    assignments, STREAMLINES
                )
                expected = {
                    "supported_edges": int(qc["supported_edges"]),
                    "possible_edges": int(qc["possible_edges"]),
                    "edge_density": float(qc["edge_density"]),
                    "connected_nodes": int(qc["connected_nodes"]),
                    "total_assigned_nondiagonal_streamlines": int(
                        qc[
                            "total_assigned_nondiagonal_streamlines"
                        ]
                    ),
                }
                for key, value in expected.items():
                    observed = level[key]
                    if isinstance(value, float):
                        if not np.isclose(
                            float(observed),
                            value,
                            atol=1.0e-12,
                            rtol=1.0e-12,
                        ):
                            raise ValueError(f"{key} differs")
                    elif observed != value:
                        raise ValueError(f"{key} differs")
                if (
                    missing != level["disconnected_nodes"]
                    or not np.isclose(
                        fraction,
                        float(level["endpoint_assignment_fraction"]),
                        atol=1.0e-12,
                        rtol=1.0e-12,
                    )
                    or bool(level["reused_existing_4mm"])
                    != bool(radius == 4)
                ):
                    raise ValueError(
                        "disconnected/fraction/reuse replay differs"
                    )
                disconnected_sets[(seed, radius)] = missing
                replay_rows.append(
                    {
                        "seed": seed,
                        "radius_mm": radius,
                        "connected_nodes": qc["connected_nodes"],
                        "edge_density": qc["edge_density"],
                        "endpoint_assignment_fraction": fraction,
                        "disconnected_node_n": len(missing),
                    }
                )
            except Exception as exc:
                replay_failures.append(
                    f"{seed}/{radius}:{type(exc).__name__}:{exc}"
                )
    add(
        "all_six_assignment_outputs_replay_independently",
        not replay_failures and len(replay_rows) == 6,
        {
            "rows": replay_rows,
            "failures": replay_failures,
        },
    )

    expected_missing = summary.get("unresolved_within_8mm", [])
    persistence_ok = (
        len(expected_missing) == 29
        and all(
            disconnected_sets.get((seed, radius))
            == expected_missing
            for seed in SEEDS
            for radius in RADII
        )
        and summary.get("consensus_recovered_at_6mm") == []
        and summary.get("consensus_recovered_at_8mm") == []
        and all(
            row.get("first_consensus_radius_mm") is None
            for row in summary.get("targets", {}).values()
        )
    )
    add(
        "same_29_nodes_remain_disconnected_through_8mm",
        persistence_ok,
        {
            "unresolved_within_8mm": expected_missing,
            "recovered_at_6mm": summary.get(
                "consensus_recovered_at_6mm"
            ),
            "recovered_at_8mm": summary.get(
                "consensus_recovered_at_8mm"
            ),
        },
    )

    fail_closed_ok = (
        summary.get("status")
        == "PASS_COMPLETE_ENDPOINT_RADIUS_LOCALIZATION"
        and summary.get("decision")
        == "ENDPOINT_RADIUS_CANNOT_RECOVER_ALL_DISCONNECTED_NODES"
        and summary.get("diagnosis_labels_used") is False
        and summary.get("outcomes_used") is False
        and summary.get("non_overwriting") is True
        and summary.get("source_files_modified") is False
        and summary.get("new_tractography_generated") is False
        and summary.get("production_assignment_radius_selected")
        is False
        and summary.get("radii_above_4mm_are_diagnostic_only")
        is True
        and summary.get("scale_up_authorized") is False
    )
    add(
        "decision_is_blind_nonoverwriting_and_fail_closed",
        fail_closed_ok,
        {
            key: summary.get(key)
            for key in (
                "status",
                "decision",
                "diagnosis_labels_used",
                "outcomes_used",
                "non_overwriting",
                "source_files_modified",
                "new_tractography_generated",
                "production_assignment_radius_selected",
                "radii_above_4mm_are_diagnostic_only",
                "scale_up_authorized",
                "next_gate",
            )
        },
    )

    passed = sum(
        value["status"] == "PASS" for value in checks.values()
    )
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_fod_order_endpoint_radius_localization_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": utc_now(),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "attempt": str(attempt),
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
