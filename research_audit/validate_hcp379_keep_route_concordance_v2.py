#!/usr/bin/env python3
"""Decide whether the remaining legacy-ACT HCP379 cases may be retained.

Four diagnosis-blind high-confidence keep cases are also members of the
15-subject corrected Phase-B panel.  This validator compares their legacy
3M ACT connectomes with the selected corrected-ACT primary run and places the
observed difference beside the selected-count independent-seed difference.

Retention is allowed only when every case meets the fixed recipe-stability
thresholds.  Otherwise the result reroutes the remaining 210 keep cases plus
the two tensor-repair and one FreeSurfer-repair legacy-ACT cases.  The four
calibration keep cases already have corrected Phase-B outputs.  No group or
clinical outcome is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
sys.path.insert(0, str(EXP / "research_audit"))

from validate_phase_b_recipe_stability import (  # noqa: E402
    global_metrics as contract_global_metrics,
    icc_a1 as contract_icc_a1,
    normalized_matrix as contract_normalized_matrix,
    relative_difference as contract_relative_difference,
    safe_correlation as contract_safe_correlation,
)

RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
PHASE_MANIFEST = (
    RUN_ROOT / "publication/hcp379_recipe_stability_manifest.json"
)
PHASE_VALIDATION = (
    RUN_ROOT / "publication/hcp379_recipe_stability_validation.json"
)
CONTRACT = (
    EXP
    / "research_audit/outputs/thesis_grade_530_release_contract_v1/"
    "contract.json"
)
KEEP_UNITS = (
    EXP
    / "research_audit/outputs/hcp_rerun_triage_v1/"
    "keep_high_confidence_214.txt"
)
LEGACY_ROOT = Path("/data/derivatives/connectomes_hcp_fs")
OUTPUT = (
    Path("/data/derivatives/hcp379_v2/calibration")
    / "keep_route_concordance.json"
)
MATRIX_NAMES = (
    "count",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)
SUPPORT_NAMES = ("count", "fd_sum", "count_invnodevol")
MEAN_NAMES = (
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)
EXPECTED_NODES = 379


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular file: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def verified_matrix(record: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(record.get("path", ""))).resolve()
    if file_record(path) != dict(record):
        raise ValueError(f"matrix binding differs: {path}")
    matrix = np.loadtxt(path, delimiter=",")
    if (
        matrix.shape != (EXPECTED_NODES, EXPECTED_NODES)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, atol=1.0e-6, rtol=1.0e-6)
        or not np.allclose(np.diag(matrix), 0.0, atol=1.0e-8)
        or float(np.min(matrix)) < -1.0e-10
    ):
        raise ValueError(f"matrix technical contract differs: {path}")
    return matrix


def legacy_matrices(unit: str) -> tuple[
    dict[str, np.ndarray], dict[str, dict[str, Any]]
]:
    arrays = {}
    records = {}
    for name in MATRIX_NAMES:
        path = LEGACY_ROOT / f"SC_HCPMMP1_{unit}_{name}.csv"
        record = file_record(path)
        arrays[name] = verified_matrix(record)
        records[name] = record
    return arrays, records


def run_matrices(run: Mapping[str, Any]) -> dict[str, np.ndarray]:
    raw = run.get("matrices")
    if not isinstance(raw, Mapping) or set(raw) != set(MATRIX_NAMES):
        raise ValueError("Phase-B matrix set differs")
    return {name: verified_matrix(raw[name]) for name in MATRIX_NAMES}


def correlation(
    left: np.ndarray, right: np.ndarray, *, rank: bool
) -> float:
    value, _, _ = contract_safe_correlation(
        left,
        right,
        method="spearman" if rank else "pearson",
        minimum_edges=EXPECTED_NODES,
    )
    return float(value) if value is not None else float("nan")


def icc_a1(values: np.ndarray) -> float:
    value, _ = contract_icc_a1(values)
    return float(value) if value is not None else float("nan")


def normalized(matrix: np.ndarray) -> np.ndarray:
    return contract_normalized_matrix(matrix)


def relative_difference(left: float, right: float) -> float:
    return contract_relative_difference(left, right)


def global_metrics(matrix: np.ndarray) -> dict[str, float]:
    return contract_global_metrics(matrix)


def compare(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    triangle = np.triu_indices(EXPECTED_NODES, 1)
    common = (
        (left["count"][triangle] > 0)
        & (right["count"][triangle] > 0)
    )
    union = (
        (left["count"][triangle] > 0)
        | (right["count"][triangle] > 0)
    )
    support = {}
    mean = {}
    failures = []
    for name in SUPPORT_NAMES:
        value = correlation(
            left[name][triangle][common],
            right[name][triangle][common],
            rank=True,
        )
        support[name] = value
        if (
            not math.isfinite(value)
            or value
            < thresholds[
                "minimum_support_matched_upper_triangle_spearman"
            ]
        ):
            failures.append(f"{name}:spearman")
    for name in MEAN_NAMES:
        value = correlation(
            left[name][triangle][common],
            right[name][triangle][common],
            rank=False,
        )
        mean[name] = value
        if (
            not math.isfinite(value)
            or value
            < thresholds["minimum_tensor_matrix_upper_triangle_pearson"]
        ):
            failures.append(f"{name}:pearson")
    strength_icc = {}
    global_differences = {}
    for name in ("count", "fd_sum"):
        left_norm = normalized(left[name])
        right_norm = normalized(right[name])
        value = icc_a1(
            np.column_stack(
                (
                    np.sum(left_norm, axis=1),
                    np.sum(right_norm, axis=1),
                )
            )
        )
        strength_icc[name] = value
        if (
            not math.isfinite(value)
            or value
            < thresholds["minimum_node_strength_absolute_agreement_icc"]
        ):
            failures.append(f"{name}:strength_icc")
        left_global = global_metrics(left_norm)
        right_global = global_metrics(right_norm)
        differences = {
            metric: relative_difference(
                left_global[metric], right_global[metric]
            )
            for metric in left_global
        }
        global_differences[name] = differences
        if any(
            difference
            > thresholds["maximum_global_metric_relative_difference"]
            for difference in differences.values()
        ):
            failures.append(f"{name}:global_metric")
    return {
        "status": "PASS" if not failures else "FAIL",
        "common_supported_edges": int(np.count_nonzero(common)),
        "support_union_edges": int(np.count_nonzero(union)),
        "support_jaccard": (
            float(np.count_nonzero(common) / np.count_nonzero(union))
            if np.count_nonzero(union)
            else 0.0
        ),
        "support_matched_spearman": support,
        "support_matched_pearson": mean,
        "node_strength_icc_a1": strength_icc,
        "global_metric_relative_differences": global_differences,
        "failures": sorted(set(failures)),
    }


def evaluate() -> dict[str, Any]:
    manifest = load_json(PHASE_MANIFEST)
    validation = load_json(PHASE_VALIDATION)
    contract = load_json(CONTRACT)
    if (
        manifest.get("record_type")
        != "diagnosis_blind_hcp379_phase_b_recipe_stability_manifest"
        or manifest.get("diagnosis_labels_used") is not False
        or not isinstance(manifest.get("units"), list)
        or len(manifest["units"]) != 15
        or validation.get("record_type")
        != "diagnosis_blind_hcp379_phase_b_recipe_stability_validation"
        or validation.get("status") != "PASS"
        or validation.get("diagnosis_labels_used") is not False
        or validation.get("unit_count") != 15
        or validation.get("manifest") != file_record(PHASE_MANIFEST)
    ):
        raise ValueError("Phase-B validation/manifest binding differs")
    selected = int(
        validation["smallest_density_qualified_scale_up_streamline_count"]
    )
    if selected not in (3_000_000, 5_000_000, 10_000_000):
        raise ValueError("Phase-B selected streamline count differs")
    primary_id = f"primary_{selected // 1_000_000}m"
    independent_id = f"independent_{selected // 1_000_000}m"
    keep = {
        line.strip()
        for line in KEEP_UNITS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(keep) != 214:
        raise ValueError("high-confidence keep set differs")
    selected_units = [
        record for record in manifest["units"] if record["unit"] in keep
    ]
    if len(selected_units) != 4:
        raise ValueError(
            f"Phase-B/keep calibration overlap is {len(selected_units)}, not 4"
        )
    raw_thresholds = contract["diagnosis_blind_recipe_stability_gate"]
    thresholds = {
        key: float(raw_thresholds[key])
        for key in (
            "minimum_support_matched_upper_triangle_spearman",
            "minimum_tensor_matrix_upper_triangle_pearson",
            "minimum_node_strength_absolute_agreement_icc",
            "maximum_global_metric_relative_difference",
        )
    }
    results = []
    for record in sorted(selected_units, key=lambda item: item["unit"]):
        unit = str(record["unit"])
        runs = {run["run_id"]: run for run in record["runs"]}
        if primary_id not in runs or independent_id not in runs:
            raise ValueError(f"{unit}:selected Phase-B runs differ")
        legacy, legacy_records = legacy_matrices(unit)
        primary = run_matrices(runs[primary_id])
        independent = run_matrices(runs[independent_id])
        route = compare(legacy, primary, thresholds)
        seed = compare(primary, independent, thresholds)
        results.append(
            {
                "unit": unit,
                "legacy_matrix_records": legacy_records,
                "legacy_vs_corrected_primary": route,
                "corrected_primary_vs_independent": seed,
                "route_pass": route["status"] == "PASS",
                "seed_pass": seed["status"] == "PASS",
                "status": (
                    "PASS"
                    if route["status"] == "PASS"
                    and seed["status"] == "PASS"
                    else "FAIL"
                ),
            }
        )
    all_pass = all(result["status"] == "PASS" for result in results)
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_keep_route_concordance_validation"
        ),
        "status": "PASS" if all_pass else "FAIL",
        "decision": (
            "RETAIN_REMAINING_210_HIGH_CONFIDENCE_CASES"
            if all_pass
            else (
                "RERUN_REMAINING_210_KEEP_PLUS_"
                "3_AUXILIARY_LEGACY_ACT_CASES"
            )
        ),
        "diagnosis_labels_used": False,
        "selected_streamline_count": selected,
        "calibration_unit_count": len(results),
        "remaining_keep_unit_count": 210,
        "auxiliary_legacy_act_unit_count": 3,
        "already_corrected_phase_b_keep_unit_count": 4,
        "failure_rerun_count": 213,
        "failure_route_accounting": {
            "remaining_keep": 210,
            "tensor_repairs": 2,
            "freesurfer_repair": 1,
            "phase_b_keep_already_corrected": 4,
        },
        "thresholds": thresholds,
        "phase_manifest": file_record(PHASE_MANIFEST),
        "phase_validation": file_record(PHASE_VALIDATION),
        "contract": file_record(CONTRACT),
        "keep_identity_set": file_record(KEEP_UNITS),
        "units": results,
        "failure_count": sum(not result["route_pass"] for result in results),
        "seed_failure_count": sum(
            not result["seed_pass"] for result in results
        ),
    }


def write_report(path: Path, report: Mapping[str, Any], overwrite: bool) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def self_test() -> None:
    count = np.zeros((EXPECTED_NODES, EXPECTED_NODES), dtype=float)
    for index in range(EXPECTED_NODES - 1):
        count[index, index + 1] = count[index + 1, index] = index + 1
    count[0, 2] = count[2, 0] = EXPECTED_NODES
    matrices = {name: count.copy() for name in MATRIX_NAMES}
    thresholds = {
        "minimum_support_matched_upper_triangle_spearman": 0.95,
        "minimum_tensor_matrix_upper_triangle_pearson": 0.98,
        "minimum_node_strength_absolute_agreement_icc": 0.9,
        "maximum_global_metric_relative_difference": 0.05,
    }
    result = compare(matrices, matrices, thresholds)
    if result["status"] != "PASS":
        raise AssertionError(result)
    print(json.dumps({"status": "PASS"}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    report = evaluate()
    write_report(args.output, report, args.overwrite)
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return int(args.require_pass and report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
