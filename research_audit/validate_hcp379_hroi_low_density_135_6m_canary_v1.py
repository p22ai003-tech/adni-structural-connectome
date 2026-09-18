#!/usr/bin/env python3
"""Validate the bounded low-density 135 balanced-6M canary."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_hroi_low_density_135_6m_canary_v1.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_hroi_low_density_135_6m_canary_v1/validation.json"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_hroi_low_density_135_6m_canary_v1/attempts"
)
TCK_COUNTER = EXP / "scripts/hcp/tck_header_count_v1.py"
EXPECTED_NODES = 379
EXPECTED_PRIMARY_ASSIGNMENTS = 10_000_000
EXPECTED_INDEPENDENT_ASSIGNMENTS = 3_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def load_assignments(path: Path, expected: int) -> np.ndarray:
    values = np.loadtxt(path, comments="#", dtype=np.int16)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if (
        values.shape != (expected, 2)
        or np.any(values < 0)
        or np.any(values > EXPECTED_NODES)
    ):
        raise ValueError(f"assignment contract differs: {path}")
    return values


def count_matrix(assignments: np.ndarray, prefix: int) -> np.ndarray:
    selected = assignments[:prefix]
    left = selected[:, 0].astype(np.int32, copy=False)
    right = selected[:, 1].astype(np.int32, copy=False)
    valid = (left > 0) & (right > 0) & (left != right)
    left = left[valid] - 1
    right = right[valid] - 1
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    linear = low * EXPECTED_NODES + high
    upper = np.bincount(
        linear,
        minlength=EXPECTED_NODES * EXPECTED_NODES,
    ).reshape(EXPECTED_NODES, EXPECTED_NODES)
    result = upper + upper.T
    np.fill_diagonal(result, 0)
    return result.astype(np.int64, copy=False)


def load_count(path: Path) -> np.ndarray:
    value = np.loadtxt(path, delimiter=",")
    if (
        value.shape != (EXPECTED_NODES, EXPECTED_NODES)
        or not np.isfinite(value).all()
        or np.any(value < 0)
        or not np.allclose(value, value.T, atol=0.0, rtol=0.0)
        or not np.allclose(np.diag(value), 0.0, atol=0.0, rtol=0.0)
        or not np.allclose(value, np.rint(value), atol=1.0e-6, rtol=0.0)
    ):
        raise ValueError(f"count matrix contract differs: {path}")
    return np.rint(value).astype(np.int64)


def matrix_qc(matrix: np.ndarray) -> dict[str, Any]:
    triangle = matrix[np.triu_indices(EXPECTED_NODES, 1)]
    supported = int(np.count_nonzero(triangle > 0))
    possible = EXPECTED_NODES * (EXPECTED_NODES - 1) // 2
    return {
        "supported_edges": supported,
        "possible_edges": possible,
        "edge_density": supported / possible,
        "connected_nodes": int(
            np.count_nonzero(np.sum(matrix, axis=1) > 0)
        ),
        "total_assigned_nondiagonal_streamlines": int(
            np.sum(triangle)
        ),
    }


def record_matches(record: Any, path: Path) -> bool:
    return (
        isinstance(record, dict)
        and Path(str(record.get("path", ""))).resolve() == path.resolve()
        and record.get("size_bytes") == path.stat().st_size
        and record.get("sha256") == sha256(path)
    )


def tck_count(path: Path) -> int:
    probe = subprocess.run(
        [str(PYTHON), str(TCK_COUNTER), str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(probe.stdout.strip())


def main() -> int:
    checks: dict[str, Any] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    compile_probe = subprocess.run(
        [str(PYTHON), "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "source_compiles",
        compile_probe.returncode == 0,
        {
            "returncode": compile_probe.returncode,
            "stderr": compile_probe.stderr[-2000:],
        },
    )
    self_probe = subprocess.run(
        [str(PYTHON), str(SOURCE), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        self_value = json.loads(self_probe.stdout)
    except Exception:
        self_value = {}
    check(
        "seed_and_balance_self_test_pass",
        self_probe.returncode == 0 and self_value.get("status") == "PASS",
        self_value or {"stderr": self_probe.stderr[-2000:]},
    )
    dry_probe = subprocess.run(
        [str(PYTHON), str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        dry = json.loads(dry_probe.stdout)
    except Exception:
        dry = {}
    check(
        "dry_run_binds_known_adverse_case_without_imaging",
        (
            dry_probe.returncode == 0
            and dry.get("status")
            == "READY_LOW_DENSITY_135_BALANCED_6M_CANARY"
            and dry.get("unit") == "135_S_6840_I1263427"
            and abs(
                float(dry.get("known_primary_hroi_density")) - 0.5571330848375703
            )
            < 1.0e-15
            and dry.get("primary_prefix_streamlines") == 3_000_000
            and dry.get("independent_streamlines_to_generate") == 3_000_000
            and dry.get("balanced_total_streamlines") == 6_000_000
            and dry.get("imaging_executed") is False
            and dry.get("cohort_recipe_selected") is False
            and dry.get("scale_up_authorized") is False
        ),
        dry or {"stderr": dry_probe.stderr[-2000:]},
    )
    text = SOURCE.read_text(encoding="utf-8")
    required = (
        "independent-replicate-1",
        "2106283499",
        "-select",
        "INDEPENDENT_STREAMLINES",
        "tracks.partial.tck",
        "count_hroi.partial.csv",
        "-assignment_radial_search",
        "primary_hroi_assignments",
        "requires_complete_15_canary_family",
        "requires_joint_all_nine_matrix_stability",
        "cohort_recipe_selected",
        "scale_up_authorized",
    )
    missing = [fragment for fragment in required if fragment not in text]
    check(
        "implementation_contains_exact_bounded_contract",
        not missing,
        {"missing": missing},
    )
    prohibited = (
        "rm -rf",
        "shutil.rmtree",
        ".unlink(",
        "diagnosis.csv",
        "DX_bl",
        "cohort_recipe_selected\": True",
        "scale_up_authorized\": True",
    )
    found = [fragment for fragment in prohibited if fragment in text]
    check(
        "implementation_has_no_destructive_or_outcome_shortcut",
        not found,
        {"found": found},
    )
    runtime_summaries = sorted(ATTEMPTS.glob("*/summary.json"))
    runtime_projection: dict[str, Any] | None = None
    if runtime_summaries:
        runtime_summary_path = runtime_summaries[-1]
        try:
            summary = load_json(runtime_summary_path)
            preflight_path = runtime_summary_path.parent / "preflight.json"
            preflight = load_json(preflight_path)
            runtime_contract_ok = (
                summary.get("record_type")
                == (
                    "diagnosis_blind_hcp379_hroi_low_density_135_"
                    "balanced_6m_summary"
                )
                and summary.get("status")
                == "PASS_LOW_DENSITY_135_6M_CALIBRATION"
                and summary.get("unit") == "135_S_6840_I1263427"
                and summary.get("diagnosis_labels_used") is False
                and summary.get("non_overwriting") is True
                and summary.get("source_files_modified") is False
                and summary.get("primary_tractography_reused") is True
                and summary.get("independent_tractography_generated")
                is True
                and summary.get("generated_independent_streamlines")
                == EXPECTED_INDEPENDENT_ASSIGNMENTS
                and summary.get("balanced_total_streamlines")
                == 6_000_000
                and summary.get("cohort_recipe_selected") is False
                and summary.get("scale_up_authorized") is False
                and summary.get("requires_complete_15_canary_family")
                is True
                and summary.get(
                    "requires_joint_all_nine_matrix_stability"
                )
                is True
            )
            check(
                "runtime_summary_contract",
                runtime_contract_ok,
                {
                    "summary": str(runtime_summary_path),
                    "status": summary.get("status"),
                },
            )

            primary_path = Path(
                str(
                    preflight.get("inputs", {})
                    .get("primary_hroi_assignments", {})
                    .get("path", "")
                )
            )
            independent_path = Path(
                str(
                    summary.get("artifacts", {})
                    .get("independent_hroi_assignments", {})
                    .get("path", "")
                )
            )
            independent_count_path = Path(
                str(
                    summary.get("artifacts", {})
                    .get("independent_hroi_count", {})
                    .get("path", "")
                )
            )
            combined_path = Path(
                str(summary.get("count_matrix", {}).get("path", ""))
            )
            metadata_path = Path(
                str(
                    summary.get("artifacts", {})
                    .get("independent_metadata", {})
                    .get("path", "")
                )
            )
            tracks_path = Path(
                str(
                    summary.get("artifacts", {})
                    .get("independent_tracks", {})
                    .get("path", "")
                )
            )
            paths = (
                primary_path,
                independent_path,
                independent_count_path,
                combined_path,
                metadata_path,
                tracks_path,
                preflight_path,
            )
            paths_ok = all(
                path.is_file() and not path.is_symlink()
                for path in paths
            )
            records_ok = (
                paths_ok
                and record_matches(
                    summary["artifacts"][
                        "independent_hroi_assignments"
                    ],
                    independent_path,
                )
                and record_matches(
                    summary["artifacts"]["independent_hroi_count"],
                    independent_count_path,
                )
                and record_matches(
                    summary["artifacts"]["independent_metadata"],
                    metadata_path,
                )
                and record_matches(
                    summary["artifacts"]["preflight"],
                    preflight_path,
                )
                and record_matches(
                    summary["count_matrix"],
                    combined_path,
                )
                and record_matches(
                    summary["implementation"],
                    SOURCE,
                )
            )
            check(
                "runtime_small_artifact_hashes_replay",
                records_ok,
                {
                    "paths_regular_and_not_symlinks": paths_ok,
                    "hash_records_match": records_ok,
                },
            )

            primary = load_assignments(
                primary_path,
                EXPECTED_PRIMARY_ASSIGNMENTS,
            )
            independent = load_assignments(
                independent_path,
                EXPECTED_INDEPENDENT_ASSIGNMENTS,
            )
            recorded_independent = load_count(
                independent_count_path
            )
            replayed_independent = count_matrix(
                independent,
                EXPECTED_INDEPENDENT_ASSIGNMENTS,
            )
            independent_exact = np.array_equal(
                recorded_independent,
                replayed_independent,
            )
            check(
                "runtime_independent_assignment_replay",
                independent_exact,
                {
                    "exact": independent_exact,
                    "maximum_absolute_difference": int(
                        np.max(
                            np.abs(
                                recorded_independent
                                - replayed_independent
                            )
                        )
                    ),
                },
            )

            levels: dict[str, Any] = {}
            matrices: dict[int, np.ndarray] = {}
            for total, prefix in (
                (3_000_000, 1_500_000),
                (5_000_000, 2_500_000),
                (6_000_000, 3_000_000),
            ):
                matrix = (
                    count_matrix(primary, prefix)
                    + count_matrix(independent, prefix)
                )
                matrices[total] = matrix
                qc = matrix_qc(matrix)
                levels[str(total)] = {
                    "construction": (
                        f"first_{prefix}_primary_plus_"
                        f"first_{prefix}_independent"
                    ),
                    "per_seed_prefix_streamlines": prefix,
                    **qc,
                    "density_target_0_60_met": (
                        float(qc["edge_density"]) >= 0.60
                    ),
                }
            recorded_combined = load_count(combined_path)
            combined_exact = np.array_equal(
                matrices[6_000_000],
                recorded_combined,
            )
            summary_density_exact = abs(
                float(summary.get("edge_density", -1.0))
                - float(levels["6000000"]["edge_density"])
            ) < 1.0e-15
            target_exact = (
                summary.get("density_target_0_60_met")
                is levels["6000000"]["density_target_0_60_met"]
            )
            runtime_projection = levels
            check(
                "runtime_balanced_3m_5m_6m_projection_replays",
                (
                    combined_exact
                    and summary_density_exact
                    and target_exact
                    and all(
                        row["connected_nodes"] == EXPECTED_NODES
                        for row in levels.values()
                    )
                ),
                {
                    "levels": levels,
                    "recorded_6m_matrix_exact": combined_exact,
                    "recorded_6m_density_exact": (
                        summary_density_exact
                    ),
                    "recorded_6m_target_exact": target_exact,
                    "cohort_recipe_selected": False,
                },
            )

            metadata = load_json(metadata_path)
            tracks_count = tck_count(tracks_path)
            track_record = summary.get("artifacts", {}).get(
                "independent_tracks",
                {},
            )
            track_contract = (
                tracks_count == EXPECTED_INDEPENDENT_ASSIGNMENTS
                and isinstance(track_record, dict)
                and track_record.get("path")
                == str(tracks_path.resolve())
                and track_record.get("size_bytes")
                == tracks_path.stat().st_size
                and metadata.get("record_type")
                == "hcp379_stability_tractography_parameters"
                and metadata.get("diagnosis_labels_used") is False
                and metadata.get("unit") == "135_S_6840_I1263427"
                and metadata.get("seed_class") == "independent"
                and metadata.get("actual_streamlines")
                == EXPECTED_INDEPENDENT_ASSIGNMENTS
            )
            check(
                "runtime_tractogram_and_metadata_contract",
                track_contract,
                {
                    "track_header_count": tracks_count,
                    "track_record_size_matches": (
                        track_record.get("size_bytes")
                        == tracks_path.stat().st_size
                    ),
                    "metadata": str(metadata_path),
                },
            )
        except Exception as exc:
            check(
                "runtime_execution_replay",
                False,
                {
                    "summary": str(runtime_summary_path),
                    "exception": repr(exc),
                },
            )
    passed = sum(row["status"] == "PASS" for row in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_hroi_low_density_135_6m_canary_v1_validation"
        ),
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "imaging_executed_by_validation": False,
        "runtime_summary_detected": bool(runtime_summaries),
        "runtime_balanced_density_projection": runtime_projection,
        "source": file_record(SOURCE),
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
