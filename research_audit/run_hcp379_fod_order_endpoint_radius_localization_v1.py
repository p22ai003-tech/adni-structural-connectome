#!/home/ec2-user/fsl/bin/python
"""Localize the reproducible 29-node lmax4 tractography failure.

This diagnosis-blind diagnostic reuses both completed 3M tractograms and the
unchanged HCP379 atlas.  It reruns endpoint assignment only at 6 and 8 mm,
while replaying the existing 4-mm matrices.  Radii above 4 mm are diagnostic:
the script cannot select a production radius, promote a recipe, or authorize
cohort scale-up.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXP = Path("/home/ec2-user/exp")
HCP = Path("/data/derivatives/hcp379_v2")
BASE_SOURCE = (
    EXP / "scripts/hcp/run_hcp379_endpoint_radius_localization_v1.py"
)
CANARY_SUMMARY = (
    EXP
    / "research_audit/outputs/"
    "hcp379_acquisition_aware_fod_order_tractography_canary_v1/"
    "summary.json"
)
CANARY_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_acquisition_aware_fod_order_tractography_canary_v1/"
    "validation.json"
)
LABELS = Path(
    "/data/derivatives/parc_hcpmmp1/hcpmmp1_subcort_nodes.csv"
)
ATTEMPTS = (
    HCP
    / "acquisition_aware_fod_order_endpoint_radius_localization_v1/"
    "attempts"
)
UNIT = "031_S_4721_I1093826"
SEEDS = ("primary", "independent")
RADII = (4, 6, 8)
STREAMLINES = 3_000_000
EXPECTED_NODE_N = 379


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_SOURCE, "hcp379_fod_order_radius_base")
BASE.STREAMLINES = STREAMLINES
BASE.RADII = RADII
BASE.BALANCED.EXPECTED_BASE_STREAMLINES = STREAMLINES


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def verify_record(record: dict[str, Any]) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    expected = {
        "path": str(path),
        "size_bytes": int(record.get("size_bytes", -1)),
        "sha256": str(record.get("sha256", "")),
    }
    if BASE.BALANCED.file_record(path) != expected:
        raise ValueError(f"artifact record differs: {path}")
    return path


def disconnected(matrix: np.ndarray) -> list[int]:
    return [
        index + 1
        for index, value in enumerate(np.sum(matrix, axis=1))
        if float(value) <= 0.0
    ]


def label_names() -> dict[int, str]:
    with LABELS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {
        int(row["node_id"]): str(row["structure"]) for row in rows
    }
    if set(mapping) != set(range(1, EXPECTED_NODE_N + 1)):
        raise ValueError("HCP379 label dictionary differs")
    return mapping


def bind_inputs() -> tuple[Path, dict[str, dict[str, Any]], list[int]]:
    validation = load_json(CANARY_VALIDATION)
    summary = load_json(CANARY_SUMMARY)
    if (
        validation.get("status") != "PASS"
        or validation.get("checks_passed")
        != validation.get("checks_total")
        or summary.get("status")
        != "FAIL_LOWER_ORDER_TRACTOGRAPHY_TECHNICAL_CANARY"
        or summary.get("production_recipe_selected") is not False
    ):
        raise ValueError("validated fail-closed tractography canary required")
    attempt = Path(str(summary["attempt_root"])).resolve()
    preflight = load_json(attempt / "preflight.json")
    unit_row = next(
        (
            row
            for row in preflight.get("units", [])
            if row.get("unit") == UNIT
        ),
        None,
    )
    if (
        not isinstance(unit_row, dict)
        or unit_row.get("recommended_lmax") != 4
        or unit_row.get("unique_direction_n") != 15
    ):
        raise ValueError("exact lmax4 target binding differs")
    nodes = verify_record(unit_row["hcp379_nodes"])
    names = label_names()
    bound: dict[str, dict[str, Any]] = {}
    missing_sets: list[set[int]] = []
    for seed in SEEDS:
        result_path = attempt / "subjects" / UNIT / seed / "result.json"
        result = load_json(result_path)
        if (
            result.get("status") != "FAIL_TECHNICAL_SEED"
            or result.get("failures")
            != ["connected_nodes_below_minimum"]
            or result.get("actual_streamlines") != STREAMLINES
        ):
            raise ValueError(f"{seed}: exact technical failure differs")
        artifacts = result["artifacts"]
        count = verify_record(artifacts["count_matrix"])
        matrix = BASE.BALANCED.load_count(count)
        missing = set(disconnected(matrix))
        if len(missing) != 29:
            raise ValueError(f"{seed}: disconnected set differs")
        missing_sets.append(missing)
        bound[seed] = {
            "tracks": verify_record(artifacts["tractogram"]),
            "nodes": nodes,
            "baseline_count": count,
            "baseline_assignments": verify_record(
                artifacts["assignments"]
            ),
            "expected_disconnected": {
                node: names[node] for node in sorted(missing)
            },
            "result_path": result_path,
        }
    if missing_sets[0] != missing_sets[1]:
        raise ValueError("seed-specific disconnected sets differ")
    return attempt, bound, sorted(missing_sets[0])


def preflight() -> dict[str, Any]:
    attempt, bound, missing = bind_inputs()
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_fod_order_endpoint_radius_localization_preflight"
        ),
        "generated_utc": utc_now(),
        "status": "READY_ENDPOINT_RADIUS_LOCALIZATION",
        "unit": UNIT,
        "seed_classes": list(SEEDS),
        "baseline_radius_mm": 4,
        "diagnostic_radii_mm": [6, 8],
        "streamlines_per_seed": STREAMLINES,
        "reproducible_disconnected_node_n": len(missing),
        "reproducible_disconnected_nodes": missing,
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used_for_selection": False,
        "new_tractography_generated": False,
        "production_assignment_radius_selected": False,
        "scale_up_authorized": False,
        "source_attempt": str(attempt),
        "inputs": {
            seed: {
                key: BASE.BALANCED.file_record(Path(value))
                for key, value in {
                    "tracks": row["tracks"],
                    "nodes": row["nodes"],
                    "baseline_count": row["baseline_count"],
                    "baseline_assignments": row[
                        "baseline_assignments"
                    ],
                    "result": row["result_path"],
                }.items()
            }
            for seed, row in bound.items()
        },
    }


def execute(workers: int, threads: int) -> dict[str, Any]:
    source_attempt, bound, missing = bind_inputs()
    attempt = ATTEMPTS / (
        f"{stamp()}-fod-order-endpoint-radius-localization"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    BASE.atomic_json(
        attempt / "preflight.json",
        {
            **preflight(),
            "status": "EXECUTION_INPUTS_BOUND",
            "implementation": BASE.BALANCED.file_record(
                Path(__file__)
            ),
        },
    )
    levels: dict[str, dict[str, Any]] = {
        seed: {} for seed in SEEDS
    }
    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for seed, row in bound.items():
            for radius in RADII:
                tasks.append(
                    (
                        seed,
                        radius,
                        pool.submit(
                            BASE.run_radius,
                            UNIT,
                            row,
                            radius,
                            attempt / seed,
                            threads,
                        ),
                    )
                )
        for seed, radius, future in tasks:
            levels[seed][str(radius)] = future.result()
    target_status: dict[str, dict[str, Any]] = {}
    for node in missing:
        by_seed = {
            seed: {
                str(radius): levels[seed][str(radius)][
                    "target_nodes"
                ][str(node)]["connected"]
                for radius in RADII
            }
            for seed in SEEDS
        }
        first_consensus_radius = next(
            (
                radius
                for radius in RADII
                if all(
                    by_seed[seed][str(radius)]
                    for seed in SEEDS
                )
            ),
            None,
        )
        target_status[str(node)] = {
            "name": bound["primary"]["expected_disconnected"][
                node
            ],
            "connected_by_seed_and_radius": by_seed,
            "first_consensus_radius_mm": first_consensus_radius,
        }
    unresolved = [
        int(node)
        for node, row in target_status.items()
        if row["first_consensus_radius_mm"] is None
    ]
    recovered_6 = [
        int(node)
        for node, row in target_status.items()
        if row["first_consensus_radius_mm"] == 6
    ]
    recovered_8 = [
        int(node)
        for node, row in target_status.items()
        if row["first_consensus_radius_mm"] == 8
    ]
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_fod_order_endpoint_radius_localization"
        ),
        "generated_utc": utc_now(),
        "status": "PASS_COMPLETE_ENDPOINT_RADIUS_LOCALIZATION",
        "unit": UNIT,
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "non_overwriting": True,
        "source_files_modified": False,
        "new_tractography_generated": False,
        "production_assignment_radius_selected": False,
        "radii_above_4mm_are_diagnostic_only": True,
        "scale_up_authorized": False,
        "source_attempt": str(source_attempt),
        "levels": levels,
        "targets": target_status,
        "consensus_recovered_at_6mm": recovered_6,
        "consensus_recovered_at_8mm": recovered_8,
        "unresolved_within_8mm": unresolved,
        "decision": (
            "ENDPOINT_RADIUS_CANNOT_RECOVER_ALL_DISCONNECTED_NODES"
            if unresolved
            else "ALL_TARGETS_HAVE_ENDPOINT_SUPPORT_WITHIN_8MM"
        ),
        "next_gate": (
            "parcel-level DWI-mask/FOD-support correction; do not "
            "select a wider production assignment radius"
        ),
        "preflight": BASE.BALANCED.file_record(
            attempt / "preflight.json"
        ),
        "implementation": BASE.BALANCED.file_record(Path(__file__)),
    }
    BASE.atomic_json(attempt / "summary.json", summary)
    return {
        "attempt_root": str(attempt),
        "status": summary["status"],
        "decision": summary["decision"],
        "recovered_at_6mm_n": len(recovered_6),
        "recovered_at_8mm_n": len(recovered_8),
        "unresolved_within_8mm_n": len(unresolved),
        "unresolved_within_8mm": unresolved,
    }


def self_test() -> None:
    matrix = np.zeros((EXPECTED_NODE_N, EXPECTED_NODE_N))
    matrix[0, 1] = matrix[1, 0] = 1
    if disconnected(matrix)[:3] != [3, 4, 5]:
        raise AssertionError("disconnected-node helper differs")
    print("HCP379_FOD_ORDER_ENDPOINT_RADIUS_SELF_TEST_PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads-per-job", type=int, default=2)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not 1 <= args.workers <= 2:
        parser.error("--workers must be in 1..2")
    if not 1 <= args.threads_per_job <= 4:
        parser.error("--threads-per-job must be in 1..4")
    value = (
        execute(args.workers, args.threads_per_job)
        if args.execute
        else preflight()
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
