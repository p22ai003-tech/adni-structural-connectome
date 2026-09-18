#!/usr/bin/env python3
"""Validate diagnosis-blind tractography convergence and seed reliability.

The input is a frozen JSON manifest.  Each unit must contain primary 3M, 5M,
and 10M runs plus an independent-seed 10M run.  Matrix artifacts are bound by
size and SHA-256 so that the report can be replayed during release evaluation.
No diagnosis, group, or outcome field is accepted anywhere in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr, spearmanr


ROOT = Path("/home/ec2-user/exp")
DEFAULT_CONTRACT = (
    ROOT / "research_audit/outputs/thesis_grade_530_release_contract_v1/contract.json"
)
STABILITY_MANIFEST_BUILDER = (
    ROOT / "scforge/workflow/build_phase_b_stability_manifest.py"
)
RUN_SPECS = {
    "primary_3m": (3_000_000, "primary"),
    "primary_5m": (5_000_000, "primary"),
    "primary_10m": (10_000_000, "primary"),
    "independent_10m": (10_000_000, "independent"),
}
COMPARISONS = (
    ("primary_3m", "primary_5m"),
    ("primary_5m", "primary_10m"),
    ("primary_3m", "primary_10m"),
    ("primary_10m", "independent_10m"),
)
SUPPORT_MATRICES = ("count", "fd_sum")
TENSOR_MATRICES = ("fa_mean", "md_mean", "rd_mean", "ad_mean")
REQUIRED_MATRICES = SUPPORT_MATRICES + TENSOR_MATRICES
REQUIRED_ARTIFACTS = (
    "tractogram",
    "sift2_weights",
    "assignments",
    "tractography_metadata",
)
FORBIDDEN_KEYS = {
    "diagnosis",
    "diagnosis_at_dti",
    "group",
    "research_group",
    "outcome",
    "clinical_outcome",
    "disease_status",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def forbidden_paths(value: Any, prefix: str = "$") -> list[str]:
    failures: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            child_path = f"{prefix}.{key}"
            if normalized in FORBIDDEN_KEYS:
                failures.append(child_path)
            failures.extend(forbidden_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(forbidden_paths(child, f"{prefix}[{index}]"))
    return failures


def verified_file_record(record: Mapping[str, Any], *, label: str) -> tuple[Path, dict[str, Any]]:
    try:
        path = Path(str(record["path"])).expanduser().resolve()
        expected_sha = str(record["sha256"])
        expected_size = int(record["size_bytes"])
    except Exception as exc:
        raise ValueError(f"{label}: malformed file record") from exc
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label}: missing/nonregular file: {path}")
    if expected_size != path.stat().st_size:
        raise ValueError(f"{label}: size differs")
    if len(expected_sha) != 64 or sha256_file(path) != expected_sha:
        raise ValueError(f"{label}: SHA256 differs or is malformed")
    return path, file_record(path)


def verified_matrix(
    record: Mapping[str, Any], *, label: str, expected_nodes: int
) -> tuple[np.ndarray, dict[str, Any]]:
    path, evidence = verified_file_record(record, label=label)
    try:
        matrix = np.loadtxt(path, delimiter=",")
    except Exception as exc:
        raise ValueError(f"{label}: cannot parse CSV matrix: {exc}") from exc
    if matrix.shape != (expected_nodes, expected_nodes):
        raise ValueError(f"{label}: shape is {matrix.shape}, expected {(expected_nodes, expected_nodes)}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label}: contains nonfinite values")
    if np.min(matrix) < -1.0e-12:
        raise ValueError(f"{label}: contains negative values")
    if not np.allclose(matrix, matrix.T, atol=1.0e-8, rtol=0.0):
        raise ValueError(f"{label}: is not symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1.0e-8, rtol=0.0):
        raise ValueError(f"{label}: diagonal is not zero")
    return matrix, evidence


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    index = np.triu_indices(matrix.shape[0], 1)
    return matrix[index]


def safe_correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    method: str,
    minimum_edges: int,
) -> tuple[float | None, int, str | None]:
    mask = np.isfinite(left) & np.isfinite(right)
    x = left[mask]
    y = right[mask]
    if x.size < minimum_edges:
        return None, int(x.size), f"only_{x.size}_eligible_edges"
    if np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return None, int(x.size), "constant_edge_vector"
    result = spearmanr(x, y) if method == "spearman" else pearsonr(x, y)
    value = float(result.statistic)
    if not math.isfinite(value):
        return None, int(x.size), "nonfinite_correlation"
    return value, int(x.size), None


def icc_a1(values: np.ndarray) -> tuple[float | None, str | None]:
    """Two-way random-effects, absolute-agreement, single-measure ICC(A,1)."""

    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[0] < 3 or data.shape[1] < 2:
        return None, "insufficient_targets_or_raters"
    if not np.isfinite(data).all():
        return None, "nonfinite_values"
    n_targets, n_raters = data.shape
    grand = float(data.mean())
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    ss_rows = n_raters * float(np.sum((row_means - grand) ** 2))
    ss_cols = n_targets * float(np.sum((col_means - grand) ** 2))
    residual = data - row_means[:, None] - col_means[None, :] + grand
    ss_error = float(np.sum(residual**2))
    ms_rows = ss_rows / (n_targets - 1)
    ms_cols = ss_cols / (n_raters - 1)
    ms_error = ss_error / ((n_targets - 1) * (n_raters - 1))
    denominator = (
        ms_rows
        + (n_raters - 1) * ms_error
        + n_raters * (ms_cols - ms_error) / n_targets
    )
    if denominator <= 0 or not math.isfinite(denominator):
        return None, "nonpositive_or_nonfinite_denominator"
    value = (ms_rows - ms_error) / denominator
    if not math.isfinite(value):
        return None, "nonfinite_icc"
    return float(value), None


def normalized_matrix(matrix: np.ndarray) -> np.ndarray:
    total = float(np.sum(upper_triangle(matrix)))
    if total <= 0:
        raise ValueError("matrix has no positive upper-triangle weight")
    return matrix / total


def global_metrics(matrix: np.ndarray) -> dict[str, float]:
    """Scale-free global descriptors used for the fixed 5% convergence gate."""

    n_nodes = matrix.shape[0]
    tri = upper_triangle(matrix)
    positive = tri > 0
    edge_count = int(np.count_nonzero(positive))
    possible = n_nodes * (n_nodes - 1) // 2
    if edge_count == 0:
        raise ValueError("matrix has no supported edges")
    norm = normalized_matrix(matrix)
    support = matrix > 0
    degree = np.sum(support, axis=1)
    strength = np.sum(norm, axis=1)
    positive_weights = upper_triangle(norm)[positive]
    probabilities = positive_weights / positive_weights.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    entropy_norm = entropy / math.log(edge_count) if edge_count > 1 else 0.0

    cost = np.full(matrix.shape, np.inf, dtype=float)
    np.fill_diagonal(cost, 0.0)
    cost[support] = 1.0 / norm[support]
    distance = shortest_path(cost, directed=False, unweighted=False)
    offdiag = ~np.eye(n_nodes, dtype=bool)
    reachable = offdiag & np.isfinite(distance) & (distance > 0)
    efficiency = float(np.mean(1.0 / distance[reachable])) if np.any(reachable) else 0.0
    mean_strength = float(np.mean(strength))
    return {
        "edge_density": edge_count / possible,
        "mean_degree_fraction": float(np.mean(degree)) / (n_nodes - 1),
        "normalized_strength_cv": (
            float(np.std(strength, ddof=1)) / mean_strength if mean_strength > 0 else 0.0
        ),
        "normalized_edge_weight_entropy": entropy_norm,
        "weighted_global_efficiency": efficiency,
    }


def relative_difference(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right), 1.0e-15)
    return abs(left - right) / denominator


def evaluate_manifest(
    manifest_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    contract_path = contract_path.expanduser().resolve()
    manifest = load_json(manifest_path)
    contract = load_json(contract_path)
    gate = contract["diagnosis_blind_recipe_stability_gate"]
    expected_nodes = int(contract["fixed_thresholds"]["matrix_shape"][0])
    failures: list[str] = []
    forbidden = forbidden_paths(manifest)
    if forbidden:
        failures.extend(f"forbidden_blinding_field:{path}" for path in forbidden)
    if manifest.get("record_type") != "diagnosis_blind_phase_b_recipe_stability_manifest":
        failures.append("unexpected_manifest_record_type")
    if manifest.get("diagnosis_labels_used") is not False:
        failures.append("diagnosis_labels_used_must_be_false")
    if manifest.get("builder_source") != file_record(STABILITY_MANIFEST_BUILDER):
        failures.append("stability_manifest_builder_source_binding_differs")
    units = manifest.get("units")
    if not isinstance(units, list) or not units:
        failures.append("units_must_be_a_nonempty_list")
        units = []
    unit_names = [str(item.get("unit", "")) for item in units if isinstance(item, Mapping)]
    if not all(unit_names) or len(unit_names) != len(set(unit_names)) or len(unit_names) != len(units):
        failures.append("unit_names_must_be_nonblank_and_unique")

    matrix_evidence: dict[str, Any] = {}
    artifact_evidence: dict[str, Any] = {}
    run_record_evidence = manifest.get("run_record_evidence")
    if not isinstance(run_record_evidence, Mapping):
        failures.append("run_record_evidence_missing")
        run_record_evidence = {}
    else:
        for label, record in run_record_evidence.items():
            if not isinstance(record, Mapping):
                failures.append(f"run_record_evidence:{label}:malformed")
                continue
            try:
                verified_file_record(record, label=f"run record {label}")
            except Exception as exc:
                failures.append(f"run_record_evidence:{label}:{exc}")
    unit_results: list[dict[str, Any]] = []
    all_spearman: list[float] = []
    all_tensor_pearson: list[float] = []
    all_icc: list[float] = []
    all_global_differences: list[float] = []
    all_assignment_differences: list[float] = []

    for item in units:
        if not isinstance(item, Mapping):
            continue
        unit = str(item.get("unit", ""))
        unit_failures: list[str] = []
        runs_raw = item.get("runs")
        if not isinstance(runs_raw, list):
            unit_failures.append("runs_must_be_a_list")
            runs_raw = []
        runs = {
            str(run.get("run_id", "")): run
            for run in runs_raw
            if isinstance(run, Mapping)
        }
        if len(runs) != len(runs_raw) or set(runs) != set(RUN_SPECS):
            unit_failures.append("run_set_must_equal_primary_3m_primary_5m_primary_10m_independent_10m")
        loaded: dict[str, dict[str, np.ndarray]] = {}
        assignment: dict[str, float] = {}
        seed_values: dict[str, str] = {}
        for run_id, (expected_count, seed_class) in RUN_SPECS.items():
            run = runs.get(run_id)
            if not isinstance(run, Mapping):
                continue
            try:
                streamline_count = int(run.get("streamline_count"))
            except Exception:
                streamline_count = -1
            if streamline_count != expected_count:
                unit_failures.append(f"{run_id}:streamline_count_differs")
            try:
                tractogram_count = int(run.get("tractogram_streamline_count"))
                weight_count = int(run.get("sift2_weight_count"))
                assignment_count = int(run.get("assignment_row_count"))
                assigned_count = int(run.get("assigned_streamline_count"))
                observed_fraction = float(run.get("endpoint_assignment_fraction"))
                if not (
                    tractogram_count == weight_count == assignment_count == expected_count
                    and 0 <= assigned_count <= assignment_count
                    and abs(observed_fraction - assigned_count / assignment_count) <= 1.0e-12
                ):
                    raise ValueError
            except Exception:
                unit_failures.append(f"{run_id}:tractogram_weight_assignment_counts_differ")
            seed_id = str(run.get("seed_id", "")).strip()
            seed_values[run_id] = seed_id
            if not seed_id:
                unit_failures.append(f"{run_id}:seed_id_blank")
            if str(run.get("seed_class", "")) != seed_class:
                unit_failures.append(f"{run_id}:seed_class_differs")
            try:
                assignment_value = float(run.get("endpoint_assignment_fraction"))
                if not 0.0 <= assignment_value <= 1.0:
                    raise ValueError
                assignment[run_id] = assignment_value
            except Exception:
                unit_failures.append(f"{run_id}:invalid_endpoint_assignment_fraction")
            records = run.get("matrices")
            if not isinstance(records, Mapping) or set(records) != set(REQUIRED_MATRICES):
                unit_failures.append(f"{run_id}:matrix_set_differs")
                continue
            loaded[run_id] = {}
            for name in REQUIRED_MATRICES:
                try:
                    matrix, evidence = verified_matrix(
                        records[name],
                        label=f"{unit}/{run_id}/{name}",
                        expected_nodes=expected_nodes,
                    )
                    loaded[run_id][name] = matrix
                    matrix_evidence[f"{unit}/{run_id}/{name}"] = evidence
                except Exception as exc:
                    unit_failures.append(f"{run_id}:{name}:{exc}")
            artifacts = run.get("artifacts")
            if not isinstance(artifacts, Mapping) or set(artifacts) != set(REQUIRED_ARTIFACTS):
                unit_failures.append(f"{run_id}:artifact_set_differs")
            else:
                for name in REQUIRED_ARTIFACTS:
                    try:
                        _, evidence = verified_file_record(
                            artifacts[name], label=f"{unit}/{run_id}/{name}"
                        )
                        artifact_evidence[f"{unit}/{run_id}/{name}"] = evidence
                    except Exception as exc:
                        unit_failures.append(f"{run_id}:{name}:{exc}")
        primary_seed_ids = {seed_values.get(name) for name in ("primary_3m", "primary_5m", "primary_10m")}
        if None in primary_seed_ids or "" in primary_seed_ids or len(primary_seed_ids) != 1:
            unit_failures.append("primary_runs_do_not_share_one_seed_id")
        if seed_values.get("independent_10m") in primary_seed_ids:
            unit_failures.append("independent_seed_is_not_distinct")

        comparisons: list[dict[str, Any]] = []
        loaded_complete = (
            set(loaded) == set(RUN_SPECS)
            and all(set(matrices) == set(REQUIRED_MATRICES) for matrices in loaded.values())
        )
        if loaded_complete and set(assignment) == set(RUN_SPECS):
            triangle = np.triu_indices(expected_nodes, 1)
            minimum_edges = max(expected_nodes, 10)
            for left_id, right_id in COMPARISONS:
                left = loaded[left_id]
                right = loaded[right_id]
                comparison_failures: list[str] = []
                common_support = (
                    (left["count"][triangle] > 0)
                    & (right["count"][triangle] > 0)
                )
                support_metrics: dict[str, Any] = {}
                for name in SUPPORT_MATRICES:
                    x = left[name][triangle][common_support]
                    y = right[name][triangle][common_support]
                    value, n_edges, reason = safe_correlation(
                        x, y, method="spearman", minimum_edges=minimum_edges
                    )
                    support_metrics[name] = {
                        "spearman": value,
                        "common_supported_edges": n_edges,
                        "failure": reason,
                    }
                    if value is None or value < float(gate["minimum_support_matched_upper_triangle_spearman"]):
                        comparison_failures.append(f"{name}:support_matched_spearman_below_gate")
                    else:
                        all_spearman.append(value)

                tensor_metrics: dict[str, Any] = {}
                for name in TENSOR_MATRICES:
                    x = left[name][triangle][common_support]
                    y = right[name][triangle][common_support]
                    value, n_edges, reason = safe_correlation(
                        x, y, method="pearson", minimum_edges=minimum_edges
                    )
                    tensor_metrics[name] = {
                        "pearson": value,
                        "common_supported_edges": n_edges,
                        "failure": reason,
                    }
                    if value is None or value < float(gate["minimum_tensor_matrix_upper_triangle_pearson"]):
                        comparison_failures.append(f"{name}:tensor_pearson_below_gate")
                    else:
                        all_tensor_pearson.append(value)

                icc_metrics: dict[str, Any] = {}
                global_differences: dict[str, Any] = {}
                for name in SUPPORT_MATRICES:
                    left_norm = normalized_matrix(left[name])
                    right_norm = normalized_matrix(right[name])
                    strengths = np.column_stack(
                        [np.sum(left_norm, axis=1), np.sum(right_norm, axis=1)]
                    )
                    icc, reason = icc_a1(strengths)
                    icc_metrics[name] = {"icc_a1": icc, "failure": reason}
                    if icc is None or icc < float(gate["minimum_node_strength_absolute_agreement_icc"]):
                        comparison_failures.append(f"{name}:node_strength_icc_below_gate")
                    else:
                        all_icc.append(icc)
                    left_global = global_metrics(left[name])
                    right_global = global_metrics(right[name])
                    differences = {
                        metric: relative_difference(left_global[metric], right_global[metric])
                        for metric in left_global
                    }
                    global_differences[name] = {
                        "left": left_global,
                        "right": right_global,
                        "relative_differences": differences,
                    }
                    for metric, value in differences.items():
                        all_global_differences.append(value)
                        if value > float(gate["maximum_global_metric_relative_difference"]):
                            comparison_failures.append(f"{name}:{metric}:relative_difference_above_gate")
                assignment_difference = abs(assignment[left_id] - assignment[right_id])
                all_assignment_differences.append(assignment_difference)
                if assignment_difference > float(gate["maximum_assignment_fraction_absolute_difference"]):
                    comparison_failures.append("endpoint_assignment_fraction_difference_above_gate")
                comparisons.append(
                    {
                        "comparison": f"{left_id}_vs_{right_id}",
                        "support_matched_edge_correlations": support_metrics,
                        "tensor_edge_correlations": tensor_metrics,
                        "node_strength_absolute_agreement": icc_metrics,
                        "global_metric_differences": global_differences,
                        "endpoint_assignment_fraction_absolute_difference": assignment_difference,
                        "status": "PASS" if not comparison_failures else "FAIL",
                        "failures": sorted(set(comparison_failures)),
                    }
                )
                unit_failures.extend(
                    f"{left_id}_vs_{right_id}:{failure}"
                    for failure in comparison_failures
                )
        unit_results.append(
            {
                "unit": unit,
                "status": "PASS" if not unit_failures else "FAIL",
                "comparison_count": len(comparisons),
                "comparisons": comparisons,
                "failures": sorted(set(unit_failures)),
            }
        )
        failures.extend(f"{unit}:{failure}" for failure in unit_failures)

    passed_units = sum(item["status"] == "PASS" for item in unit_results)
    expected_run_record_labels = {
        f"{unit}/{run_id}" for unit in unit_names for run_id in RUN_SPECS
    }
    if set(run_record_evidence) != expected_run_record_labels:
        failures.append("run_record_evidence_set_differs")
    checks = {
        "diagnosis_and_outcome_blind": not forbidden and manifest.get("diagnosis_labels_used") is False,
        "manifest_builder_source_is_current": manifest.get("builder_source") == file_record(STABILITY_MANIFEST_BUILDER),
        "run_record_evidence_is_exact": set(run_record_evidence) == expected_run_record_labels,
        "nonempty_unique_unit_set": bool(units) and len(unit_names) == len(set(unit_names)) == len(units),
        "all_units_pass": bool(unit_results) and passed_units == len(unit_results),
        "all_four_required_runs_present": all(item["comparison_count"] == len(COMPARISONS) for item in unit_results),
        "independent_seed_evaluated": all(
            any(c["comparison"] == "primary_10m_vs_independent_10m" for c in item["comparisons"])
            for item in unit_results
        ),
    }
    status = "PASS" if all(checks.values()) and not failures else "FAIL"
    return {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_phase_b_recipe_stability_validation",
        "generated_utc": utc_now(),
        "status": status,
        "diagnosis_labels_used": False,
        "manifest": file_record(manifest_path),
        "contract": file_record(contract_path),
        "validator_source": file_record(Path(__file__)),
        "thresholds": dict(gate),
        "global_metric_definitions": [
            "edge_density",
            "mean_degree_fraction",
            "normalized_strength_cv",
            "normalized_edge_weight_entropy",
            "weighted_global_efficiency",
        ],
        "unit_count": len(unit_results),
        "passed_unit_count": passed_units,
        "checks": checks,
        "aggregate": {
            "minimum_support_matched_spearman": min(all_spearman) if all_spearman else None,
            "minimum_tensor_pearson": min(all_tensor_pearson) if all_tensor_pearson else None,
            "minimum_node_strength_icc_a1": min(all_icc) if all_icc else None,
            "maximum_global_metric_relative_difference": max(all_global_differences) if all_global_differences else None,
            "maximum_assignment_fraction_absolute_difference": max(all_assignment_differences) if all_assignment_differences else None,
        },
        "matrix_evidence": dict(sorted(matrix_evidence.items())),
        "artifact_evidence": dict(sorted(artifact_evidence.items())),
        "unit_results": unit_results,
        "failure_count": len(set(failures)),
        "failures": sorted(set(failures)),
    }


def write_report(path: Path, report: Mapping[str, Any], *, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {path}")
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o444)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    report = evaluate_manifest(args.manifest, args.contract)
    write_report(args.output, report, overwrite=args.overwrite)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, sort_keys=True))
    return 1 if args.require_pass and report["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
