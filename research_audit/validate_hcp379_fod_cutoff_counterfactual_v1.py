#!/usr/bin/env python3
"""Independently replay the bounded HCP379 FOD-cutoff counterfactual."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_hroi_fod_cutoff_counterfactual_v1/attempts"
)
PREDECLARATION_VALIDATION = (
    EXP
    / "research_audit/outputs/hcp379_fod_cutoff_counterfactual_v1/"
    "predeclaration_validation.json"
)
DEFAULT_OUTPUT = (
    EXP
    / "research_audit/outputs/hcp379_fod_cutoff_counterfactual_v1/"
    "validation.json"
)
TCK_COUNTER = EXP / "scripts/hcp/tck_header_count_v1.py"
PYTHON = Path("/home/ec2-user/fsl/bin/python")
EXPECTED_UNITS = {
    "007_S_5196_I390043": "LOWEST_DENSITY_FULL_NODE_ADVERSE",
    "032_S_6804_I1230908": "HIGH_DENSITY_PASSING_CONTROL",
}
EXPECTED_SEEDS = {"primary", "independent"}
EXPECTED_NODES = 379
EXPECTED_PER_SEED = 3_000_000
POSSIBLE_EDGES = EXPECTED_NODES * (EXPECTED_NODES - 1) // 2


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def record_matches(record: Mapping[str, Any]) -> bool:
    path = Path(str(record.get("path", ""))).resolve()
    return bool(
        path.is_file()
        and not path.is_symlink()
        and int(record.get("size_bytes", -1)) == path.stat().st_size
        and str(record.get("sha256", "")) == sha256(path)
    )


def latest_summary() -> Path:
    for path in sorted(ATTEMPTS.glob("*/summary.json"), reverse=True):
        try:
            value = load_json(path)
        except Exception:
            continue
        if (
            value.get("record_type")
            == "diagnosis_blind_hcp379_fod_cutoff_counterfactual"
            and value.get("status")
            == "PASS_COMPLETE_FOD_CUTOFF_COUNTERFACTUAL"
        ):
            return path.resolve()
    raise FileNotFoundError("complete FOD-cutoff summary absent")


def load_count(path: Path) -> np.ndarray:
    value = np.loadtxt(path, delimiter=",")
    if (
        value.shape != (EXPECTED_NODES, EXPECTED_NODES)
        or not np.isfinite(value).all()
        or np.any(value < 0)
        or not np.array_equal(value, value.T)
        or not np.array_equal(np.diag(value), np.zeros(EXPECTED_NODES))
        or not np.allclose(value, np.rint(value), atol=1.0e-6, rtol=0)
    ):
        raise ValueError(f"{path}: invalid count matrix")
    return np.rint(value).astype(np.int64)


def load_assignments(path: Path, expected_n: int) -> np.ndarray:
    value = np.loadtxt(path, comments="#", dtype=np.int16)
    if value.ndim == 1:
        value = value.reshape(1, -1)
    if (
        value.shape != (expected_n, 2)
        or np.any(value < 0)
        or np.any(value > EXPECTED_NODES)
    ):
        raise ValueError(f"{path}: invalid endpoint assignments")
    return value


def replay_count(assignments: np.ndarray, prefix: int) -> np.ndarray:
    selected = assignments[:prefix]
    left = selected[:, 0].astype(np.int32, copy=False)
    right = selected[:, 1].astype(np.int32, copy=False)
    valid = (left > 0) & (right > 0) & (left != right)
    low = np.minimum(left[valid], right[valid]) - 1
    high = np.maximum(left[valid], right[valid]) - 1
    linear = low * EXPECTED_NODES + high
    upper = np.bincount(
        linear, minlength=EXPECTED_NODES * EXPECTED_NODES
    ).reshape(EXPECTED_NODES, EXPECTED_NODES)
    matrix = upper + upper.T
    np.fill_diagonal(matrix, 0)
    return matrix.astype(np.int64, copy=False)


def matrix_qc(matrix: np.ndarray) -> dict[str, Any]:
    upper = matrix[np.triu_indices(EXPECTED_NODES, 1)]
    supported = int(np.count_nonzero(upper > 0))
    return {
        "supported_edges": supported,
        "possible_edges": POSSIBLE_EDGES,
        "edge_density": supported / POSSIBLE_EDGES,
        "connected_nodes": int(
            np.count_nonzero(np.sum(matrix, axis=1) > 0)
        ),
        "total_assigned_nondiagonal_streamlines": int(np.sum(upper)),
    }


def complementarity(
    primary: np.ndarray, independent: np.ndarray
) -> dict[str, Any]:
    triangle = np.triu_indices(EXPECTED_NODES, 1)
    first = primary[triangle] > 0
    second = independent[triangle] > 0
    union = first | second
    overlap = first & second
    union_n = int(np.count_nonzero(union))
    return {
        "primary_supported_edges": int(np.count_nonzero(first)),
        "independent_supported_edges": int(np.count_nonzero(second)),
        "overlap_supported_edges": int(np.count_nonzero(overlap)),
        "primary_only_supported_edges": int(
            np.count_nonzero(first & ~second)
        ),
        "independent_only_supported_edges": int(
            np.count_nonzero(second & ~first)
        ),
        "union_supported_edges": union_n,
        "support_jaccard": (
            int(np.count_nonzero(overlap)) / union_n if union_n else 0.0
        ),
    }


def stable_change(
    baseline: Mapping[str, np.ndarray],
    counterfactual: Mapping[str, np.ndarray],
) -> dict[str, int]:
    triangle = np.triu_indices(EXPECTED_NODES, 1)
    bp = baseline["primary"][triangle] > 0
    bi = baseline["independent"][triangle] > 0
    cp = counterfactual["primary"][triangle] > 0
    ci = counterfactual["independent"][triangle] > 0
    return {
        "stable_new_edges": int(
            np.count_nonzero((cp & ci) & ~(bp | bi))
        ),
        "stable_lost_edges": int(
            np.count_nonzero((bp & bi) & ~(cp | ci))
        ),
    }


def tck_count(path: Path) -> int:
    completed = subprocess.run(
        [str(PYTHON), str(TCK_COUNTER), str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip())


def numeric_mapping_equal(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    if set(observed) != set(expected):
        return False
    for key, value in expected.items():
        other = observed[key]
        if isinstance(value, float):
            if not np.isclose(float(other), value, atol=1.0e-12):
                return False
        elif other != value:
            return False
    return True


def validate(summary_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    predecl_validation = load_json(PREDECLARATION_VALIDATION)
    checks: list[dict[str, Any]] = []

    def check(label: str, passed: bool, detail: Any = None) -> None:
        checks.append(
            {"label": label, "pass": bool(passed), "detail": detail}
        )

    check(
        "summary_contract",
        summary.get("record_type")
        == "diagnosis_blind_hcp379_fod_cutoff_counterfactual"
        and summary.get("status")
        == "PASS_COMPLETE_FOD_CUTOFF_COUNTERFACTUAL"
        and summary.get("schema_version") == "1.0.0",
    )
    check(
        "blinding_and_scope",
        summary.get("diagnosis_labels_used") is False
        and summary.get("outcomes_used") is False
        and summary.get("non_overwriting") is True
        and summary.get("source_files_modified") is False
        and summary.get("production_recipe_selected") is False
        and summary.get("scale_up_authorized") is False
        and summary.get("sift2_or_scalar_matrices_generated") is False,
    )
    check(
        "changed_factor_and_total",
        summary.get("changed_factor_only")
        == {
            "parameter": "normalised_wmfod_termination_cutoff",
            "baseline": 0.06,
            "counterfactual": 0.05,
        }
        and int(summary.get("generated_streamlines", -1))
        == len(EXPECTED_UNITS)
        * len(EXPECTED_SEEDS)
        * EXPECTED_PER_SEED,
    )
    check(
        "top_level_records",
        record_matches(summary["preflight"])
        and record_matches(summary["predeclaration"])
        and record_matches(summary["implementation"]),
    )
    check(
        "predeclaration_was_independently_validated",
        predecl_validation.get("status") == "PASS"
        and int(predecl_validation.get("passed_checks", -1)) == 11
        and int(predecl_validation.get("total_checks", -1)) == 11
        and Path(
            str(predecl_validation.get("predeclaration", ""))
        ).resolve()
        == Path(summary["predeclaration"]["path"]).resolve(),
    )
    units = summary.get("units", {})
    check("exact_unit_set", set(units) == set(EXPECTED_UNITS))

    records_ok = True
    metadata_ok = True
    tck_ok = True
    assignments_ok = True
    combined_ok = True
    qc_ok = True
    baseline_ok = True
    complementarity_ok = True
    stable_ok = True
    delta_ok = True
    replay_details: dict[str, Any] = {}

    for unit, role in EXPECTED_UNITS.items():
        row = units[unit]
        baseline_row = row["baseline_cutoff_0p06"]
        counterfactual_row = row["counterfactual_cutoff_0p05"]
        preflight = load_json(
            Path(summary["preflight"]["path"]).resolve()
        )
        density_summary_path = Path(
            preflight["inputs"][unit]["density_summary"]["path"]
        ).resolve()
        density_summary = load_json(density_summary_path)
        density_attempt = density_summary_path.parent
        baseline_matrices: dict[str, np.ndarray] = {}
        counterfactual_matrices: dict[str, np.ndarray] = {}
        seed_details: dict[str, Any] = {}
        for seed in sorted(EXPECTED_SEEDS):
            source_assignments_path = (
                density_attempt / seed / "hroi_assignments_10m.csv"
            )
            source_assignments = load_assignments(
                source_assignments_path, 10_000_000
            )
            baseline_matrices[seed] = replay_count(
                source_assignments, EXPECTED_PER_SEED
            )
            seed_row = counterfactual_row["seeds"][seed]
            records_ok &= all(
                record_matches(seed_row[name])
                for name in (
                    "tracks",
                    "metadata",
                    "count_matrix",
                    "assignments",
                )
            )
            metadata = load_json(Path(seed_row["metadata"]["path"]))
            metadata_ok &= bool(
                metadata.get("record_type")
                == "hcp379_fod_cutoff_counterfactual_parameters"
                and metadata.get("unit") == unit
                and metadata.get("role") == role
                and metadata.get("seed_class") == seed
                and metadata.get("algorithm") == "iFOD2"
                and metadata.get("act") is True
                and metadata.get("backtrack") is True
                and metadata.get("crop_at_gmwmi") is True
                and metadata.get("seeding") == "seed_dynamic"
                and metadata.get("baseline_cutoff") == 0.06
                and metadata.get("counterfactual_cutoff") == 0.05
                and metadata.get("paired_common_random_seed") is True
                and metadata.get("counterfactual_rng_seed")
                == metadata.get("source_rng_seed")
                and metadata.get("maximum_angle_degrees") == 45.0
                and metadata.get("minimum_length_mm") == 10.0
                and metadata.get("maximum_length_mm") == 250.0
                and metadata.get("nthreads") == 16
                and metadata.get("actual_streamlines")
                == EXPECTED_PER_SEED
                and metadata.get("diagnosis_labels_used") is False
            )
            track_path = Path(seed_row["tracks"]["path"]).resolve()
            tck_ok &= tck_count(track_path) == EXPECTED_PER_SEED
            matrix = load_count(
                Path(seed_row["count_matrix"]["path"]).resolve()
            )
            assignments = load_assignments(
                Path(seed_row["assignments"]["path"]).resolve(),
                EXPECTED_PER_SEED,
            )
            assignments_ok &= np.array_equal(
                matrix, replay_count(assignments, EXPECTED_PER_SEED)
            )
            counterfactual_matrices[seed] = matrix
            seed_qc = matrix_qc(matrix)
            qc_ok &= numeric_mapping_equal(
                seed_row["technical_qc"], seed_qc
            )
            assignment_fraction = float(
                np.mean(
                    (assignments[:, 0] > 0)
                    & (assignments[:, 1] > 0)
                )
            )
            qc_ok &= np.isclose(
                assignment_fraction,
                float(seed_row["endpoint_assignment_fraction"]),
                atol=1.0e-12,
            )
            seed_details[seed] = {
                "supported_edges": seed_qc["supported_edges"],
                "endpoint_assignment_fraction": assignment_fraction,
            }

        expected_baseline = (
            baseline_matrices["primary"]
            + baseline_matrices["independent"]
        )
        recorded_baseline = load_count(
            Path(baseline_row["count_matrix"]["path"]).resolve()
        )
        records_ok &= record_matches(baseline_row["count_matrix"])
        baseline_ok &= np.array_equal(
            expected_baseline, recorded_baseline
        )
        expected_counterfactual = (
            counterfactual_matrices["primary"]
            + counterfactual_matrices["independent"]
        )
        recorded_counterfactual = load_count(
            Path(counterfactual_row["count_matrix"]["path"]).resolve()
        )
        records_ok &= record_matches(counterfactual_row["count_matrix"])
        combined_ok &= np.array_equal(
            expected_counterfactual, recorded_counterfactual
        )
        baseline_qc = matrix_qc(recorded_baseline)
        counterfactual_qc = matrix_qc(recorded_counterfactual)
        qc_ok &= all(
            (
                np.isclose(
                    float(counterfactual_row[key]),
                    float(value),
                    atol=1.0e-12,
                )
                if isinstance(value, float)
                else counterfactual_row[key] == value
            )
            for key, value in counterfactual_qc.items()
        )
        baseline_complementarity = complementarity(
            baseline_matrices["primary"],
            baseline_matrices["independent"],
        )
        counterfactual_complementarity = complementarity(
            counterfactual_matrices["primary"],
            counterfactual_matrices["independent"],
        )
        complementarity_ok &= (
            numeric_mapping_equal(
                baseline_row["support_complementarity"],
                baseline_complementarity,
            )
            and numeric_mapping_equal(
                counterfactual_row["support_complementarity"],
                counterfactual_complementarity,
            )
        )
        expected_stable = stable_change(
            baseline_matrices, counterfactual_matrices
        )
        stable_ok &= (
            counterfactual_row["stable_support_change"]
            == expected_stable
        )
        difference = row["difference"]
        delta_ok &= bool(
            np.isclose(
                float(difference["edge_density"]),
                counterfactual_qc["edge_density"]
                - baseline_qc["edge_density"],
                atol=1.0e-12,
            )
            and difference["connected_nodes"]
            == counterfactual_qc["connected_nodes"]
            - baseline_qc["connected_nodes"]
            and all(
                difference["seed_supported_edges"][seed]
                == counterfactual_complementarity[
                    f"{seed}_supported_edges"
                ]
                - baseline_complementarity[
                    f"{seed}_supported_edges"
                ]
                for seed in EXPECTED_SEEDS
            )
            and np.isclose(
                float(difference["support_jaccard"]),
                counterfactual_complementarity["support_jaccard"]
                - baseline_complementarity["support_jaccard"],
                atol=1.0e-12,
            )
            and np.isclose(
                float(baseline_row["edge_density"]),
                baseline_qc["edge_density"],
                atol=1.0e-12,
            )
        )
        replay_details[unit] = {
            "role": role,
            "baseline_density": baseline_qc["edge_density"],
            "counterfactual_density": counterfactual_qc[
                "edge_density"
            ],
            "density_difference": counterfactual_qc["edge_density"]
            - baseline_qc["edge_density"],
            "counterfactual_connected_nodes": counterfactual_qc[
                "connected_nodes"
            ],
            "stable_support_change": expected_stable,
            "seeds": seed_details,
        }

    check("all_output_records_rehash", records_ok)
    check("all_metadata_contracts", metadata_ok)
    check("all_tractogram_counts", tck_ok)
    check("assignment_to_matrix_replay", assignments_ok)
    check("baseline_prefix_replay", baseline_ok)
    check("balanced_6m_matrix_algebra", combined_ok)
    check("matrix_and_assignment_qc_replay", qc_ok)
    check("two_seed_complementarity_replay", complementarity_ok)
    check("stable_support_change_replay", stable_ok)
    check("counterfactual_delta_replay", delta_ok)
    passed = sum(int(row["pass"]) for row in checks)
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "independent_hcp379_fod_cutoff_counterfactual_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "summary": str(summary_path.resolve()),
        "replay": replay_details,
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "scale_up_authorized": False,
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = (
        args.summary.resolve() if args.summary else latest_summary()
    )
    result = validate(summary)
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
