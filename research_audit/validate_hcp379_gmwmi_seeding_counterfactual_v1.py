#!/usr/bin/env python3
"""Independently replay the bounded HCP379 GMWMI-seeding counterfactual."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_hroi_gmwmi_seeding_counterfactual_v1/attempts"
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
EXPECTED_TOTAL = 6_000_000
POSSIBLE_EDGES = EXPECTED_NODES * (EXPECTED_NODES - 1) // 2


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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
            == "diagnosis_blind_hcp379_gmwmi_seeding_counterfactual"
            and value.get("status")
            == "PASS_COMPLETE_GMWMI_SEEDING_COUNTERFACTUAL"
        ):
            return path.resolve()
    raise FileNotFoundError("complete GMWMI counterfactual summary absent")


def load_count(path: Path) -> np.ndarray:
    value = np.loadtxt(path, delimiter=",")
    if (
        value.shape != (EXPECTED_NODES, EXPECTED_NODES)
        or not np.isfinite(value).all()
        or np.any(value < 0)
        or not np.array_equal(value, value.T)
        or not np.array_equal(np.diag(value), np.zeros(EXPECTED_NODES))
        or not np.allclose(value, np.rint(value), atol=1.0e-6, rtol=0.0)
    ):
        raise ValueError(f"{path}: invalid count matrix")
    return np.rint(value).astype(np.int64)


def load_assignments(path: Path) -> np.ndarray:
    value = np.loadtxt(path, comments="#", dtype=np.int16)
    if value.ndim == 1:
        value = value.reshape(1, -1)
    if (
        value.shape != (EXPECTED_PER_SEED, 2)
        or np.any(value < 0)
        or np.any(value > EXPECTED_NODES)
    ):
        raise ValueError(f"{path}: invalid endpoint assignments")
    return value


def replay_count(assignments: np.ndarray) -> np.ndarray:
    left = assignments[:, 0].astype(np.int32, copy=False)
    right = assignments[:, 1].astype(np.int32, copy=False)
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


def tck_count(path: Path) -> int:
    result = subprocess.run(
        [str(PYTHON), str(TCK_COUNTER), str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def validate(path: Path) -> dict[str, Any]:
    summary = load_json(path)
    checks: list[dict[str, Any]] = []

    def check(label: str, passed: bool, detail: Any = None) -> None:
        checks.append(
            {"label": label, "pass": bool(passed), "detail": detail}
        )

    check(
        "summary_contract",
        summary.get("record_type")
        == "diagnosis_blind_hcp379_gmwmi_seeding_counterfactual"
        and summary.get("status")
        == "PASS_COMPLETE_GMWMI_SEEDING_COUNTERFACTUAL"
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
        summary.get("changed_factor") == "seed_dynamic_to_seed_gmwmi_only"
        and int(summary.get("generated_streamlines", -1))
        == len(EXPECTED_UNITS)
        * len(EXPECTED_SEEDS)
        * EXPECTED_PER_SEED,
    )
    check(
        "top_level_records",
        record_matches(summary["preflight"])
        and record_matches(summary["implementation"]),
    )
    units = summary.get("units", {})
    check("exact_unit_set", set(units) == set(EXPECTED_UNITS))

    all_records = True
    all_metadata = True
    all_tck_counts = True
    all_assignment_replays = True
    all_combined_algebra = True
    all_qc = True
    all_delta_replays = True
    all_seed_independence = True
    replay_details: dict[str, Any] = {}

    for unit, role in EXPECTED_UNITS.items():
        row = units[unit]
        baseline = row["dynamic_seed_baseline"]
        gmwmi = row["gmwmi_counterfactual"]
        seed_rows = gmwmi.get("seeds", {})
        all_seed_independence &= set(seed_rows) == EXPECTED_SEEDS
        seed_matrices: dict[str, np.ndarray] = {}
        rng_values: list[int] = []
        seed_details: dict[str, Any] = {}
        for seed in sorted(EXPECTED_SEEDS):
            seed_row = seed_rows[seed]
            records = [
                seed_row["tracks"],
                seed_row["metadata"],
                seed_row["count_matrix"],
                seed_row["assignments"],
            ]
            all_records &= all(record_matches(item) for item in records)
            metadata = load_json(
                Path(seed_row["metadata"]["path"]).resolve()
            )
            all_metadata &= bool(
                metadata.get("record_type")
                == "hcp379_gmwmi_counterfactual_parameters"
                and metadata.get("unit") == unit
                and metadata.get("role") == role
                and metadata.get("seed_class") == seed
                and metadata.get("seeding") == "seed_gmwmi"
                and metadata.get("algorithm") == "iFOD2"
                and metadata.get("act") is True
                and metadata.get("backtrack") is True
                and metadata.get("crop_at_gmwmi") is True
                and metadata.get("diagnosis_labels_used") is False
                and int(metadata.get("actual_streamlines", -1))
                == EXPECTED_PER_SEED
            )
            rng_values.append(int(metadata["rng_seed"]))
            track_path = Path(seed_row["tracks"]["path"]).resolve()
            all_tck_counts &= tck_count(track_path) == EXPECTED_PER_SEED
            matrix = load_count(
                Path(seed_row["count_matrix"]["path"]).resolve()
            )
            assignments = load_assignments(
                Path(seed_row["assignments"]["path"]).resolve()
            )
            replay = replay_count(assignments)
            all_assignment_replays &= np.array_equal(matrix, replay)
            qc = matrix_qc(matrix)
            recorded_qc = seed_row["technical_qc"]
            all_qc &= all(
                (
                    np.isclose(qc[key], recorded_qc[key], atol=1.0e-12)
                    if isinstance(qc[key], float)
                    else qc[key] == recorded_qc[key]
                )
                for key in qc
            )
            expected_assignment_fraction = float(
                np.mean(
                    (assignments[:, 0] > 0)
                    & (assignments[:, 1] > 0)
                )
            )
            all_qc &= np.isclose(
                expected_assignment_fraction,
                float(seed_row["endpoint_assignment_fraction"]),
                atol=1.0e-12,
            )
            seed_matrices[seed] = matrix
            seed_details[seed] = {
                "edge_density": qc["edge_density"],
                "connected_nodes": qc["connected_nodes"],
                "endpoint_assignment_fraction": (
                    expected_assignment_fraction
                ),
            }

        all_seed_independence &= len(set(rng_values)) == 2
        combined = load_count(
            Path(gmwmi["count_matrix"]["path"]).resolve()
        )
        all_records &= record_matches(gmwmi["count_matrix"])
        expected_combined = (
            seed_matrices["primary"] + seed_matrices["independent"]
        )
        all_combined_algebra &= np.array_equal(
            combined, expected_combined
        )
        combined_qc = matrix_qc(combined)
        all_qc &= all(
            (
                np.isclose(combined_qc[key], gmwmi[key], atol=1.0e-12)
                if isinstance(combined_qc[key], float)
                else combined_qc[key] == gmwmi[key]
            )
            for key in combined_qc
        )
        difference = row["difference"]
        all_delta_replays &= bool(
            np.isclose(
                float(difference["edge_density"]),
                combined_qc["edge_density"]
                - float(baseline["edge_density"]),
                atol=1.0e-12,
            )
            and int(difference["connected_nodes"])
            == combined_qc["connected_nodes"]
            - int(baseline["connected_nodes"])
            and int(baseline["connected_nodes"]) == EXPECTED_NODES
            and int(baseline["possible_edges"]) == POSSIBLE_EDGES
            and int(combined_qc["possible_edges"]) == POSSIBLE_EDGES
        )
        replay_details[unit] = {
            "role": role,
            "dynamic_density": float(baseline["edge_density"]),
            "gmwmi_density": combined_qc["edge_density"],
            "density_difference": combined_qc["edge_density"]
            - float(baseline["edge_density"]),
            "gmwmi_connected_nodes": combined_qc["connected_nodes"],
            "seeds": seed_details,
        }

    check("all_output_records_rehash", all_records)
    check("all_metadata_contracts", all_metadata)
    check("all_tractogram_counts", all_tck_counts)
    check("independent_seed_identities", all_seed_independence)
    check("assignment_to_matrix_replay", all_assignment_replays)
    check("balanced_6m_matrix_algebra", all_combined_algebra)
    check("matrix_and_assignment_qc_replay", all_qc)
    check("counterfactual_delta_replay", all_delta_replays)

    passed = sum(item["pass"] for item in checks)
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "independent_hcp379_gmwmi_seeding_counterfactual_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "summary": str(path.resolve()),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "replay": replay_details,
        "interpretation_boundary": (
            "This validation establishes technical integrity only. It does "
            "not select a tractography recipe or authorize scale-up."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = args.summary.resolve() if args.summary else latest_summary()
    result = validate(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
