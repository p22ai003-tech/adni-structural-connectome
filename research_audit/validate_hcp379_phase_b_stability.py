#!/usr/bin/env python3
"""Validate HCP379 technical integrity and streamline-count/seed stability.

Lower streamline levels are candidates, not mandatory passes.  The validation
passes when every artifact is technically valid and at least one streamline
level is stable across independent seeds and against the 10M reference.  It
then reports the smallest stable level.  A 0.60 raw-density target is a
recipe-scale-up gate across the diagnosis-blind canary, but is never used to
include or exclude an individual from biological analysis.  The 0.70 target
remains descriptive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path("/home/ec2-user/exp")
sys.path.insert(0, str(ROOT / "research_audit"))

from validate_phase_b_recipe_stability import (  # noqa: E402
    file_record,
    forbidden_paths,
    global_metrics,
    icc_a1,
    normalized_matrix,
    relative_difference,
    safe_correlation,
    utc_now,
    verified_file_record,
    verified_matrix,
)


BUILDER = (
    ROOT / "scforge/workflow/build_hcp379_phase_b_stability_manifest.py"
)
CONTRACT = (
    ROOT
    / "research_audit/outputs/thesis_grade_530_release_contract_v1/"
    "contract.json"
)
EXPECTED_NODES = 379
RUN_SPECS = {
    "primary_3m": (3_000_000, "primary"),
    "primary_5m": (5_000_000, "primary"),
    "primary_10m": (10_000_000, "primary"),
    "independent_3m": (3_000_000, "independent"),
    "independent_5m": (5_000_000, "independent"),
    "independent_10m": (10_000_000, "independent"),
}
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
SUPPORT_MATRICES = ("count", "fd_sum", "count_invnodevol")
MEAN_MATRICES = (
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)
ARTIFACT_NAMES = (
    "tractogram",
    "sift2_weights",
    "assignments",
    "tractography_metadata",
)
ATLAS_ARTIFACT_NAMES = (
    "source_parcellation",
    "corrected_nodes",
    "node_volumes",
    "source_node_volumes",
    "atlas_qc",
    "visual_overlay",
    "atlas_result",
    "corrected_atlas_gate",
    "selected_t1_to_b0_transform_artifact",
    "corrected_b0_reference",
)
COMPARISONS = (
    ("primary_3m", "primary_5m"),
    ("primary_5m", "primary_10m"),
    ("primary_3m", "primary_10m"),
    ("primary_3m", "independent_3m"),
    ("primary_5m", "independent_5m"),
    ("primary_10m", "independent_10m"),
)
HUB_IDS = (361, 365, 366, 370, 374, 375)
MINIMUM_CONNECTED_NODES = 360
PREFERRED_CONNECTED_NODES = 376
MINIMUM_ENDPOINT_ASSIGNMENT_FRACTION = 0.50
PREFERRED_ENDPOINT_ASSIGNMENT_FRACTION = 0.85
MINIMUM_WEIGHTED_COUNT_SUPPORT_FRACTION = 0.99
PROVISIONAL_DENSITY_TARGETS = (0.60, 0.70)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], 1)]


def matrix_technical_qc(
    matrices: Mapping[str, np.ndarray],
    *,
    node_volumes: np.ndarray,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    count = matrices["count"]
    if not np.allclose(count, np.rint(count), atol=1.0e-6, rtol=0.0):
        failures.append("count_matrix_is_not_integer")
    count_support = count > 0
    np.fill_diagonal(count_support, False)
    supported_edges = int(np.count_nonzero(np.triu(count_support, 1)))
    possible = EXPECTED_NODES * (EXPECTED_NODES - 1) // 2
    connected_nodes = int(np.count_nonzero(np.sum(count, axis=1) > 0))
    hub_strengths = {
        str(node): float(np.sum(count[node - 1])) for node in HUB_IDS
    }
    if connected_nodes < MINIMUM_CONNECTED_NODES:
        failures.append(
            f"connected_nodes={connected_nodes}<{MINIMUM_CONNECTED_NODES}"
        )
    if any(value <= 0 for value in hub_strengths.values()):
        failures.append("one_or_more_locked_hubs_are_disconnected")

    support_agreement: dict[str, Any] = {}
    for name in MATRIX_NAMES:
        if name == "count":
            continue
        support = matrices[name] > 0
        np.fill_diagonal(support, False)
        outside = int(np.count_nonzero(np.triu(support & ~count_support, 1)))
        covered = int(np.count_nonzero(np.triu(support & count_support, 1)))
        fraction = covered / supported_edges if supported_edges else 0.0
        support_agreement[name] = {
            "count_edges_covered": covered,
            "count_supported_edges": supported_edges,
            "count_edge_support_fraction": fraction,
            "weighted_edges_outside_count_support": outside,
        }
        if outside:
            failures.append(f"{name}:support_outside_count={outside}")
        if fraction < MINIMUM_WEIGHTED_COUNT_SUPPORT_FRACTION:
            failures.append(
                f"{name}:count_support_fraction={fraction:.6f}<"
                f"{MINIMUM_WEIGHTED_COUNT_SUPPORT_FRACTION:.6f}"
            )

    denominator = node_volumes[:, None] + node_volumes[None, :]
    expected_inverse_volume = 2.0 * count / denominator
    np.fill_diagonal(expected_inverse_volume, 0.0)
    maximum_formula_error = float(
        np.max(np.abs(matrices["count_invnodevol"] - expected_inverse_volume))
    )
    if not np.allclose(
        matrices["count_invnodevol"],
        expected_inverse_volume,
        atol=1.0e-8,
        rtol=1.0e-7,
    ):
        failures.append("count_invnodevol_formula_differs")
    return (
        {
            "edge_density": supported_edges / possible,
            "supported_edges": supported_edges,
            "possible_edges": possible,
            "connected_nodes": connected_nodes,
            "preferred_connected_nodes_met": (
                connected_nodes >= PREFERRED_CONNECTED_NODES
            ),
            "hub_strengths": hub_strengths,
            "support_agreement": support_agreement,
            "count_invnodevol_maximum_formula_error": maximum_formula_error,
        },
        failures,
    )


def load_node_volumes(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_NODES:
        raise ValueError(f"node-volume row count is {len(rows)}")
    rows.sort(key=lambda row: int(row["node_id"]))
    if [int(row["node_id"]) for row in rows] != list(
        range(1, EXPECTED_NODES + 1)
    ):
        raise ValueError("node-volume ids differ from 1..379")
    values = np.asarray([float(row["volume_mm3"]) for row in rows], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("node volumes are nonfinite or nonpositive")
    return values


def evaluate_comparison(
    *,
    left_id: str,
    right_id: str,
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    left_assignment: float,
    right_assignment: float,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    triangle = np.triu_indices(EXPECTED_NODES, 1)
    common_support = (
        (left["count"][triangle] > 0) & (right["count"][triangle] > 0)
    )
    minimum_edges = EXPECTED_NODES
    failures: list[str] = []
    support_metrics: dict[str, Any] = {}
    for name in SUPPORT_MATRICES:
        value, n_edges, reason = safe_correlation(
            left[name][triangle][common_support],
            right[name][triangle][common_support],
            method="spearman",
            minimum_edges=minimum_edges,
        )
        support_metrics[name] = {
            "spearman": value,
            "common_supported_edges": n_edges,
            "failure": reason,
        }
        if (
            value is None
            or value
            < float(
                gate["minimum_support_matched_upper_triangle_spearman"]
            )
        ):
            failures.append(f"{name}:support_matched_spearman_below_gate")

    mean_metrics: dict[str, Any] = {}
    for name in MEAN_MATRICES:
        value, n_edges, reason = safe_correlation(
            left[name][triangle][common_support],
            right[name][triangle][common_support],
            method="pearson",
            minimum_edges=minimum_edges,
        )
        mean_metrics[name] = {
            "pearson": value,
            "common_supported_edges": n_edges,
            "failure": reason,
        }
        if (
            value is None
            or value < float(gate["minimum_tensor_matrix_upper_triangle_pearson"])
        ):
            failures.append(f"{name}:support_matched_pearson_below_gate")

    icc_metrics: dict[str, Any] = {}
    global_differences: dict[str, Any] = {}
    for name in ("count", "fd_sum"):
        left_normalized = normalized_matrix(left[name])
        right_normalized = normalized_matrix(right[name])
        strengths = np.column_stack(
            [
                np.sum(left_normalized, axis=1),
                np.sum(right_normalized, axis=1),
            ]
        )
        icc, reason = icc_a1(strengths)
        icc_metrics[name] = {"icc_a1": icc, "failure": reason}
        if (
            icc is None
            or icc
            < float(gate["minimum_node_strength_absolute_agreement_icc"])
        ):
            failures.append(f"{name}:node_strength_icc_below_gate")
        left_global = global_metrics(left[name])
        right_global = global_metrics(right[name])
        differences = {
            metric: relative_difference(
                left_global[metric], right_global[metric]
            )
            for metric in left_global
        }
        global_differences[name] = {
            "left": left_global,
            "right": right_global,
            "relative_differences": differences,
        }
        for metric, value in differences.items():
            if value > float(gate["maximum_global_metric_relative_difference"]):
                failures.append(
                    f"{name}:{metric}:relative_difference_above_gate"
                )

    assignment_difference = abs(left_assignment - right_assignment)
    if assignment_difference > float(
        gate["maximum_assignment_fraction_absolute_difference"]
    ):
        failures.append("endpoint_assignment_fraction_difference_above_gate")
    return {
        "comparison": f"{left_id}_vs_{right_id}",
        "left_run": left_id,
        "right_run": right_id,
        "status": "PASS" if not failures else "FAIL",
        "support_matched_edge_correlations": support_metrics,
        "mean_matrix_edge_correlations": mean_metrics,
        "node_strength_absolute_agreement": icc_metrics,
        "global_metric_differences": global_differences,
        "endpoint_assignment_fraction_absolute_difference": (
            assignment_difference
        ),
        "failures": sorted(set(failures)),
    }


def evaluate_manifest(
    manifest_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    contract_path = contract_path.expanduser().resolve()
    manifest = load_json(manifest_path)
    contract = load_json(contract_path)
    gate = contract["diagnosis_blind_recipe_stability_gate"]
    failures: list[str] = []
    forbidden = forbidden_paths(manifest)
    failures.extend(f"forbidden_blinding_field:{path}" for path in forbidden)
    if (
        manifest.get("record_type")
        != "diagnosis_blind_hcp379_phase_b_recipe_stability_manifest"
        or manifest.get("diagnosis_labels_used") is not False
    ):
        failures.append("manifest_identity_or_blinding_differs")
    if manifest.get("builder_source") != file_record(BUILDER):
        failures.append("manifest_builder_source_binding_differs")

    units = manifest.get("units")
    if not isinstance(units, list) or not units:
        failures.append("units_must_be_nonempty")
        units = []
    unit_names = [
        str(item.get("unit", "")) for item in units if isinstance(item, Mapping)
    ]
    if (
        len(unit_names) != len(units)
        or not all(unit_names)
        or len(unit_names) != len(set(unit_names))
    ):
        failures.append("unit_names_must_be_nonblank_and_unique")

    run_record_evidence = manifest.get("run_record_evidence")
    if not isinstance(run_record_evidence, Mapping):
        failures.append("run_record_evidence_missing")
        run_record_evidence = {}
    else:
        for label, record in run_record_evidence.items():
            try:
                verified_file_record(record, label=f"run record {label}")
            except Exception as exc:
                failures.append(f"run_record_evidence:{label}:{exc}")

    unit_results: list[dict[str, Any]] = []
    cohort_density: dict[str, list[float]] = {
        run_id: [] for run_id in RUN_SPECS
    }
    cohort_assignment: dict[str, list[float]] = {
        run_id: [] for run_id in RUN_SPECS
    }
    for item in units:
        if not isinstance(item, Mapping):
            continue
        unit = str(item.get("unit", ""))
        technical_failures: list[str] = []
        atlas = item.get("atlas")
        node_volumes: np.ndarray | None = None
        if not isinstance(atlas, Mapping) or set(atlas) != set(
            ATLAS_ARTIFACT_NAMES
        ):
            technical_failures.append("atlas_artifact_set_differs")
        else:
            verified_atlas_paths: dict[str, Path] = {}
            for name in ATLAS_ARTIFACT_NAMES:
                try:
                    path, _ = verified_file_record(
                        atlas[name], label=f"{unit}/atlas/{name}"
                    )
                    verified_atlas_paths[name] = path
                except Exception as exc:
                    technical_failures.append(f"atlas:{name}:{exc}")
            if set(verified_atlas_paths) == set(ATLAS_ARTIFACT_NAMES):
                try:
                    atlas_qc = load_json(verified_atlas_paths["atlas_qc"])
                    if (
                        atlas_qc.get("status") != "PASS"
                        or atlas_qc.get("unit") != unit
                        or atlas_qc.get("diagnosis_labels_used") is not False
                        or atlas_qc.get("expected_nodes") != EXPECTED_NODES
                        or atlas_qc.get("labels_found") != EXPECTED_NODES
                        or atlas_qc.get("missing_labels") != []
                        or atlas_qc.get("failures") != []
                    ):
                        technical_failures.append("corrected_atlas_qc_not_pass")
                    atlas_result = load_json(
                        verified_atlas_paths["atlas_result"]
                    )
                    if (
                        atlas_result.get("status") != "PASS"
                        or atlas_result.get("unit") != unit
                        or atlas_result.get("diagnosis_labels_used") is not False
                        or atlas_result.get("atlas_qc", {}).get("status")
                        != "PASS"
                    ):
                        technical_failures.append(
                            "corrected_atlas_result_not_pass"
                        )
                    atlas_gate = load_json(
                        verified_atlas_paths["corrected_atlas_gate"]
                    )
                    if (
                        atlas_gate.get("status") != "PASS"
                        or atlas_gate.get("diagnosis_labels_used") is not False
                        or unit
                        not in atlas_gate.get("approved_units", [])
                        or atlas_gate.get("failures") != []
                    ):
                        technical_failures.append(
                            "corrected_atlas_gate_not_pass"
                        )
                    node_volumes = load_node_volumes(
                        verified_atlas_paths["node_volumes"]
                    )
                    load_node_volumes(
                        verified_atlas_paths["source_node_volumes"]
                    )
                except Exception as exc:
                    technical_failures.append(f"atlas_qc_or_volumes:{exc}")

        runs_raw = item.get("runs")
        runs = {
            str(run.get("run_id", "")): run
            for run in runs_raw
            if isinstance(run, Mapping)
        } if isinstance(runs_raw, list) else {}
        if set(runs) != set(RUN_SPECS) or len(runs) != len(runs_raw or []):
            technical_failures.append("run_set_differs")
        loaded: dict[str, dict[str, np.ndarray]] = {}
        assignments: dict[str, float] = {}
        per_run: dict[str, Any] = {}
        seed_ids: dict[str, str] = {}
        for run_id, (expected_count, expected_seed_class) in RUN_SPECS.items():
            run = runs.get(run_id)
            if not isinstance(run, Mapping):
                continue
            try:
                observed_counts = (
                    int(run["streamline_count"]),
                    int(run["tractogram_streamline_count"]),
                    int(run["sift2_weight_count"]),
                    int(run["assignment_row_count"]),
                )
                assigned = int(run["assigned_streamline_count"])
                assignment = float(run["endpoint_assignment_fraction"])
                if (
                    any(value != expected_count for value in observed_counts)
                    or not 0 <= assigned <= expected_count
                    or abs(assignment - assigned / expected_count) > 1.0e-12
                ):
                    raise ValueError
                assignments[run_id] = assignment
                cohort_assignment[run_id].append(assignment)
                if assignment < MINIMUM_ENDPOINT_ASSIGNMENT_FRACTION:
                    technical_failures.append(
                        f"{run_id}:assignment_fraction={assignment:.6f}<"
                        f"{MINIMUM_ENDPOINT_ASSIGNMENT_FRACTION:.6f}"
                    )
            except Exception:
                technical_failures.append(f"{run_id}:observed_counts_differ")
            seed_id = str(run.get("seed_id", "")).strip()
            seed_ids[run_id] = seed_id
            if (
                not seed_id
                or str(run.get("seed_class", "")) != expected_seed_class
            ):
                technical_failures.append(f"{run_id}:seed_identity_differs")
            matrices_raw = run.get("matrices")
            artifacts_raw = run.get("artifacts")
            if (
                not isinstance(matrices_raw, Mapping)
                or set(matrices_raw) != set(MATRIX_NAMES)
            ):
                technical_failures.append(f"{run_id}:matrix_set_differs")
                continue
            if (
                not isinstance(artifacts_raw, Mapping)
                or set(artifacts_raw) != set(ARTIFACT_NAMES)
            ):
                technical_failures.append(f"{run_id}:artifact_set_differs")
            else:
                for name in ARTIFACT_NAMES:
                    try:
                        verified_file_record(
                            artifacts_raw[name],
                            label=f"{unit}/{run_id}/{name}",
                        )
                    except Exception as exc:
                        technical_failures.append(
                            f"{run_id}:{name}:{exc}"
                        )
            loaded[run_id] = {}
            for name in MATRIX_NAMES:
                try:
                    matrix, _ = verified_matrix(
                        matrices_raw[name],
                        label=f"{unit}/{run_id}/{name}",
                        expected_nodes=EXPECTED_NODES,
                    )
                    loaded[run_id][name] = matrix
                except Exception as exc:
                    technical_failures.append(f"{run_id}:{name}:{exc}")
            if (
                set(loaded[run_id]) == set(MATRIX_NAMES)
                and node_volumes is not None
            ):
                qc, qc_failures = matrix_technical_qc(
                    loaded[run_id], node_volumes=node_volumes
                )
                per_run[run_id] = {
                    **qc,
                    "endpoint_assignment_fraction": assignments.get(run_id),
                    "preferred_assignment_fraction_met": (
                        assignments.get(run_id, 0.0)
                        >= PREFERRED_ENDPOINT_ASSIGNMENT_FRACTION
                    ),
                    "density_target_0_60_met": (
                        qc["edge_density"] >= 0.60
                    ),
                    "density_target_0_70_met": (
                        qc["edge_density"] >= 0.70
                    ),
                    "technical_status": (
                        "PASS" if not qc_failures else "FAIL"
                    ),
                    "technical_failures": qc_failures,
                }
                cohort_density[run_id].append(qc["edge_density"])
                technical_failures.extend(
                    f"{run_id}:{failure}" for failure in qc_failures
                )

        for prefix in ("primary", "independent"):
            ids = {
                seed_ids.get(f"{prefix}_3m"),
                seed_ids.get(f"{prefix}_5m"),
                seed_ids.get(f"{prefix}_10m"),
            }
            if None in ids or "" in ids or len(ids) != 1:
                technical_failures.append(
                    f"{prefix}_nested_runs_do_not_share_seed_id"
                )
        if seed_ids.get("primary_10m") == seed_ids.get("independent_10m"):
            technical_failures.append("independent_seed_is_not_distinct")

        comparisons: list[dict[str, Any]] = []
        comparison_by_name: dict[str, dict[str, Any]] = {}
        if (
            set(loaded) == set(RUN_SPECS)
            and all(
                set(matrices) == set(MATRIX_NAMES)
                for matrices in loaded.values()
            )
            and set(assignments) == set(RUN_SPECS)
        ):
            for left_id, right_id in COMPARISONS:
                result = evaluate_comparison(
                    left_id=left_id,
                    right_id=right_id,
                    left=loaded[left_id],
                    right=loaded[right_id],
                    left_assignment=assignments[left_id],
                    right_assignment=assignments[right_id],
                    gate=gate,
                )
                comparisons.append(result)
                comparison_by_name[result["comparison"]] = result

        stable_counts: list[int] = []
        requirements = {
            3_000_000: (
                "primary_3m_vs_primary_10m",
                "primary_3m_vs_independent_3m",
            ),
            5_000_000: (
                "primary_5m_vs_primary_10m",
                "primary_5m_vs_independent_5m",
            ),
            10_000_000: ("primary_10m_vs_independent_10m",),
        }
        for count, names in requirements.items():
            if all(
                comparison_by_name.get(name, {}).get("status") == "PASS"
                for name in names
            ):
                stable_counts.append(count)
        unit_result = {
            "unit": unit,
            "technical_status": (
                "PASS" if not technical_failures else "FAIL"
            ),
            "technical_failures": sorted(set(technical_failures)),
            "runs": per_run,
            "comparisons": comparisons,
            "stable_streamline_counts": stable_counts,
            "smallest_stable_streamline_count": (
                min(stable_counts) if stable_counts else None
            ),
            "ten_million_independent_seed_pass": (
                comparison_by_name.get(
                    "primary_10m_vs_independent_10m", {}
                ).get("status")
                == "PASS"
            ),
        }
        unit_results.append(unit_result)
        failures.extend(
            f"{unit}:{failure}"
            for failure in unit_result["technical_failures"]
        )
        if not stable_counts:
            failures.append(f"{unit}:no_stable_streamline_count")
        if not unit_result["ten_million_independent_seed_pass"]:
            failures.append(f"{unit}:10m_independent_seed_gate_failed")

    expected_run_labels = {
        f"{unit}/{run_id}" for unit in unit_names for run_id in RUN_SPECS
    }
    if set(run_record_evidence) != expected_run_labels:
        failures.append("run_record_evidence_set_differs")

    cohort_stable_counts = [
        count
        for count in (3_000_000, 5_000_000, 10_000_000)
        if unit_results
        and all(
            count in result["stable_streamline_counts"]
            for result in unit_results
        )
    ]
    smallest_cohort_stable = (
        min(cohort_stable_counts) if cohort_stable_counts else None
    )
    if smallest_cohort_stable is None:
        failures.append("no_streamline_count_is_stable_across_all_units")

    count_run_ids = {
        3_000_000: ("primary_3m", "independent_3m"),
        5_000_000: ("primary_5m", "independent_5m"),
        10_000_000: ("primary_10m", "independent_10m"),
    }
    density_scale_up_counts = [
        count
        for count in cohort_stable_counts
        if all(
            result.get("runs", {}).get(run_id, {}).get(
                "edge_density", -1.0
            )
            >= PROVISIONAL_DENSITY_TARGETS[0]
            for result in unit_results
            for run_id in count_run_ids[count]
        )
    ]
    smallest_density_scale_up_count = (
        min(density_scale_up_counts) if density_scale_up_counts else None
    )
    if smallest_density_scale_up_count is None:
        failures.append(
            "no_stable_streamline_count_meets_0_60_recipe_scale_up_target"
        )

    def descriptive(values: list[float]) -> dict[str, Any]:
        if not values:
            return {
                "n": 0,
                "median": None,
                "minimum": None,
                "maximum": None,
            }
        array = np.asarray(values, dtype=float)
        return {
            "n": int(array.size),
            "median": float(np.median(array)),
            "q1": float(np.quantile(array, 0.25)),
            "q3": float(np.quantile(array, 0.75)),
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
        }

    density_summary = {
        run_id: {
            **descriptive(values),
            **{
                f"fraction_at_or_above_{str(target).replace('.', '_')}": (
                    float(np.mean(np.asarray(values) >= target))
                    if values
                    else None
                )
                for target in PROVISIONAL_DENSITY_TARGETS
            },
        }
        for run_id, values in cohort_density.items()
    }
    assignment_summary = {
        run_id: descriptive(values)
        for run_id, values in cohort_assignment.items()
    }
    checks = {
        "diagnosis_and_outcome_blind": (
            not forbidden and manifest.get("diagnosis_labels_used") is False
        ),
        "builder_source_is_current": (
            manifest.get("builder_source") == file_record(BUILDER)
        ),
        "all_units_technically_pass": bool(unit_results)
        and all(
            result["technical_status"] == "PASS" for result in unit_results
        ),
        "ten_million_seed_reliability_passes_all_units": bool(unit_results)
        and all(
            result["ten_million_independent_seed_pass"]
            for result in unit_results
        ),
        "a_stable_streamline_count_exists_for_all_units": (
            smallest_cohort_stable is not None
        ),
        "density_0_60_recipe_scale_up_target_met": (
            smallest_density_scale_up_count is not None
        ),
        "run_record_evidence_is_exact": (
            set(run_record_evidence) == expected_run_labels
        ),
    }
    status = "PASS" if all(checks.values()) and not failures else "FAIL"
    scale_up_decision_status = (
        "DENSITY_QUALIFIED_UNIFORM_RECIPE_SELECTED"
        if smallest_density_scale_up_count is not None
        else "NO_DENSITY_QUALIFIED_UNIFORM_RECIPE"
    )
    return {
        "schema_version": "1.1.0",
        "record_type": (
            "diagnosis_blind_hcp379_phase_b_recipe_stability_validation"
        ),
        "generated_utc": utc_now(),
        "status": status,
        "scale_up_decision_status": scale_up_decision_status,
        "scale_up_authorized": (
            status == "PASS"
            and smallest_density_scale_up_count is not None
        ),
        "diagnosis_labels_used": False,
        "manifest": file_record(manifest_path),
        "contract": file_record(contract_path),
        "validator_source": file_record(Path(__file__)),
        "hcp379_acceptance": {
            "expected_nodes": EXPECTED_NODES,
            "minimum_connected_nodes": MINIMUM_CONNECTED_NODES,
            "preferred_connected_nodes": PREFERRED_CONNECTED_NODES,
            "minimum_endpoint_assignment_fraction": (
                MINIMUM_ENDPOINT_ASSIGNMENT_FRACTION
            ),
            "preferred_endpoint_assignment_fraction": (
                PREFERRED_ENDPOINT_ASSIGNMENT_FRACTION
            ),
            "minimum_weighted_count_support_fraction": (
                MINIMUM_WEIGHTED_COUNT_SUPPORT_FRACTION
            ),
            "provisional_density_targets": list(
                PROVISIONAL_DENSITY_TARGETS
            ),
            "density_is_subject_inclusion_gate": False,
            "density_0_60_is_recipe_scale_up_gate": True,
            "required_every_canary_primary_and_independent_seed_at_"
            "selected_count_at_or_above_0_60": True,
            "uniform_streamline_count_required_for_all_530": True,
            "per_subject_streamline_count_tuning_allowed": False,
            "failure_action": (
                "STOP_AND_REVISE_UNIFORM_ROUTE_NOT_EXCLUDE_SUBJECTS"
            ),
            "density_0_70_is_descriptive_target": True,
        },
        "stability_thresholds": dict(gate),
        "checks": checks,
        "unit_count": len(unit_results),
        "smallest_cohort_stable_streamline_count": smallest_cohort_stable,
        "cohort_stable_streamline_counts": cohort_stable_counts,
        "smallest_density_qualified_scale_up_streamline_count": (
            smallest_density_scale_up_count
        ),
        "density_qualified_scale_up_streamline_counts": (
            density_scale_up_counts
        ),
        "density_summary": density_summary,
        "assignment_summary": assignment_summary,
        "unit_results": unit_results,
        "failure_count": len(set(failures)),
        "failures": sorted(set(failures)),
    }


def write_report(
    path: Path, report: Mapping[str, Any], *, overwrite: bool
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {path}")
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o444)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    report = evaluate_manifest(args.manifest, args.contract)
    write_report(args.output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": report["status"],
                "smallest_cohort_stable_streamline_count": report[
                    "smallest_cohort_stable_streamline_count"
                ],
                "smallest_density_qualified_scale_up_streamline_count": report[
                    "smallest_density_qualified_scale_up_streamline_count"
                ],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 1 if args.require_pass and report["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
