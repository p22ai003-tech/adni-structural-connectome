#!/usr/bin/env python3
"""Isolate registration effects for the HCP379 036 node-coverage canary.

The selected Recovery3 BBR route and the locked seeded-ANTs rigid candidate
both pass the prospective spatial gates for 036.  This diagnostic holds the
HROI source, both exact 10M tractograms, streamline prefixes, endpoint
assignment radius, and count-matrix construction fixed.  It changes only the
T1-to-b0 transform and reports whether node coverage or density changes.

The diagnostic is diagnosis-blind and non-overwriting.  It does not select a
production route from density, generate tractography, run SIFT2/scalar
matrices, or authorize 530-subject execution.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
DENSITY_SOURCE = (
    EXP
    / "scripts/hcp/"
    "build_hcp379_hroi_balanced_seed_density_canary_v1.py"
)
REGISTRATION_AUDIT = (
    HCP_ROOT
    / "diagnostics/registration_recovery4/"
    "036_S_2380_I1023443/result.json"
)
BASELINE_ATTEMPTS = (
    HCP_ROOT / "phase_b_hroi_balanced_density_canary_v1/attempts"
)
ATTEMPTS = (
    HCP_ROOT / "phase_b_hroi_registration_counterfactual_v1/attempts"
)
UNIT = "036_S_2380_I1023443"
EXPECTED_NODES = 379
EXPECTED_STREAMLINES = 10_000_000


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DENSITY = load_module(DENSITY_SOURCE, "hcp379_registration_cf_density")
ATLAS = DENSITY.ATLAS
BALANCED = DENSITY.BALANCED


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


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


def latest_baseline() -> tuple[Path, dict[str, Any]]:
    for path in sorted(
        BASELINE_ATTEMPTS.glob(f"*-{UNIT}/summary.json"), reverse=True
    ):
        try:
            value = load_json(path)
        except Exception:
            continue
        if (
            value.get("record_type")
            == "diagnosis_blind_hcp379_hroi_balanced_density_canary_summary"
            and value.get("status")
            == "PASS_HROI_BALANCED_DENSITY_CANARY"
            and value.get("unit") == UNIT
            and set(value.get("levels", {}))
            == {"6000000", "10000000", "15000000", "20000000"}
        ):
            return path.resolve(), value
    raise FileNotFoundError("complete 036 HROI density baseline is absent")


def alternate_route() -> dict[str, Any]:
    value = load_json(REGISTRATION_AUDIT)
    affine_record = value.get("artifacts", {}).get("affine", {})
    affine = Path(str(affine_record.get("path", ""))).resolve()
    baseline = ATLAS.BASE.registration_route(UNIT)
    if (
        value.get("record_type")
        != "diagnosis_blind_hcp379_registration_candidate_audit"
        or value.get("status") != "PASS"
        or value.get("diagnosis_labels_used") is not False
        or value.get("unit") != UNIT
        or value.get("candidate", {}).get("transform") != "rigid"
        or BALANCED.file_record(affine) != affine_record
        or baseline.get("route") != "recovery3_fsl_bbr_primary"
        or baseline.get("proxy_status") != "PASS"
    ):
        raise ValueError("036 locked registration counterfactual differs")
    return {
        "unit": UNIT,
        "route": "seeded_ants_rigid_counterfactual_only",
        "proxy_status": "PASS_COUNTERFACTUAL_NOT_SELECTED",
        "transform_format": "ants",
        "transform": affine_record,
        "reason": (
            "bounded same-tractogram counterfactual against the selected "
            "Recovery3 BBR route; not a density-based route selection"
        ),
        "recovery3_bbr": value["recovery3_bbr"],
        "seeded_ants_candidate": {
            "status": value["status"],
            "five_tt_dwi_mask_dice": value["ants_rigid"][
                "five_tt_dwi_mask_dice"
            ],
            "atlas_inside_dwi_mask_fraction": value["ants_rigid"][
                "atlas_inside_dwi_mask_fraction"
            ],
        },
    }


def disconnected_nodes(matrix: np.ndarray) -> list[int]:
    return [
        index + 1
        for index, value in enumerate(np.sum(matrix, axis=1))
        if float(value) <= 0.0
    ]


def bind_inputs() -> dict[str, Any]:
    baseline_path, baseline = latest_baseline()
    pair = DENSITY.validate_phase_pair(UNIT)
    candidate = ATLAS.candidate_record(UNIT)
    route = alternate_route()
    baseline_level = baseline["levels"]["20000000"]
    if (
        not pair.get("ready")
        or set(pair.get("counts", {}).values()) != {EXPECTED_STREAMLINES}
        or int(baseline_level["connected_nodes"]) != 378
        or abs(
            float(baseline_level["edge_density"])
            - 0.4678840166966816
        )
        > 1.0e-15
    ):
        raise ValueError("036 baseline or exact tractogram pair differs")
    baseline_matrix = BALANCED.load_count(
        Path(baseline_level["count_matrix"]["path"])
    )
    if disconnected_nodes(baseline_matrix) != [303]:
        raise ValueError("036 baseline disconnected-node identity differs")
    return {
        "baseline_path": baseline_path,
        "baseline": baseline,
        "pair": pair,
        "candidate": candidate,
        "route": route,
    }


def preflight() -> dict[str, Any]:
    bound = bind_inputs()
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_registration_"
            "counterfactual_preflight"
        ),
        "generated_utc": utc_now(),
        "status": "READY_036_SAME_TRACTOGRAM_REGISTRATION_COUNTERFACTUAL",
        "unit": UNIT,
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "non_overwriting": True,
        "imaging_executed": False,
        "tractography_generated": False,
        "sift2_or_scalar_matrices_generated": False,
        "changed_factor": "T1_to_b0_registration_transform_only",
        "selected_route": bound["baseline"]["hroi_atlas_result"],
        "counterfactual_route": bound["route"],
        "exact_primary_independent_pair_ready": True,
        "baseline_disconnected_nodes": [303],
        "production_route_selected": False,
        "scale_up_authorized": False,
    }


def execute(threads: int) -> dict[str, Any]:
    bound = bind_inputs()
    attempt = (
        ATTEMPTS
        / f"{stamp()}-036-same-tractogram-registration-counterfactual"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    input_record = {
        **preflight(),
        "status": "EXECUTION_INPUTS_BOUND",
        "imaging_executed": True,
        "baseline_summary": BALANCED.file_record(
            bound["baseline_path"]
        ),
        "registration_audit": BALANCED.file_record(REGISTRATION_AUDIT),
        "implementation": BALANCED.file_record(Path(__file__)),
    }
    atomic_json(attempt / "preflight.json", input_record)

    atlas_result = ATLAS.build_unit(
        UNIT,
        candidate=bound["candidate"],
        expected_t1=ATLAS.BASE.BASE.execution_t1_by_unit()[UNIT],
        route=bound["route"],
        attempt=attempt / "atlas_counterfactual",
    )
    if atlas_result.get("status") != "PASS_HROI_CANARY_ATLAS":
        raise RuntimeError(
            "counterfactual HROI atlas failed: "
            f"{atlas_result.get('failure')}"
        )
    nodes = Path(
        atlas_result["artifacts"]["hcp379_nodes_b0_1mm"]["path"]
    ).resolve()
    paths = DENSITY.phase_paths(UNIT)
    assignments: dict[str, np.ndarray] = {}
    seed_rows: dict[str, Any] = {}
    for seed in ("primary", "independent"):
        root = attempt / seed
        root.mkdir()
        count_path = root / "hroi_count_10m.csv"
        assignments_path = root / "hroi_assignments_10m.csv"
        DENSITY.run_connectome(
            tracks=paths[f"{seed}_tracks"],
            nodes=nodes,
            output=count_path,
            assignments=assignments_path,
            log=root / "tck2connectome.log",
            threads=threads,
        )
        observed = BALANCED.load_assignments(assignments_path)
        recorded = BALANCED.load_count(count_path)
        replayed = BALANCED.count_matrix(observed, EXPECTED_STREAMLINES)
        if not np.array_equal(recorded, replayed):
            raise ValueError(f"{seed}: assignment replay differs")
        assignments[seed] = observed
        seed_rows[seed] = {
            "tracks": BALANCED.file_record(paths[f"{seed}_tracks"]),
            "count_matrix": BALANCED.file_record(count_path),
            "assignments": BALANCED.file_record(assignments_path),
            "endpoint_assignment_fraction": (
                BALANCED.assignment_fraction(
                    observed, EXPECTED_STREAMLINES
                )
            ),
            **BALANCED.matrix_qc(recorded),
        }

    levels: dict[str, Any] = {}
    for total, prefix in DENSITY.BALANCED_TOTALS.items():
        primary = BALANCED.count_matrix(assignments["primary"], prefix)
        independent = BALANCED.count_matrix(
            assignments["independent"], prefix
        )
        combined = primary + independent
        path = attempt / "balanced" / f"balanced_{total // 1_000_000}m.csv"
        DENSITY.atomic_matrix(path, combined)
        qc = BALANCED.matrix_qc(combined)
        levels[str(total)] = {
            **qc,
            "disconnected_nodes": disconnected_nodes(combined),
            "primary_endpoint_assignment_fraction": (
                BALANCED.assignment_fraction(
                    assignments["primary"], prefix
                )
            ),
            "independent_endpoint_assignment_fraction": (
                BALANCED.assignment_fraction(
                    assignments["independent"], prefix
                )
            ),
            "support_complementarity": DENSITY.support_complementarity(
                primary, independent
            ),
            "count_matrix": BALANCED.file_record(path),
        }

    baseline = bound["baseline"]["levels"]["20000000"]
    observed = levels["20000000"]
    baseline_density = float(baseline["edge_density"])
    observed_density = float(observed["edge_density"])
    node_change = (
        int(observed["connected_nodes"])
        - int(baseline["connected_nodes"])
    )
    interpretation = (
        "REGISTRATION_CONTRIBUTES_TO_NODE_COVERAGE"
        if node_change > 0
        else "DISCONNECTED_NODE_PERSISTS_ACROSS_PASSING_TRANSFORMS"
    )
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_registration_counterfactual"
        ),
        "generated_utc": utc_now(),
        "status": "PASS_COMPLETE_036_REGISTRATION_COUNTERFACTUAL",
        "unit": UNIT,
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "non_overwriting": True,
        "source_files_modified": False,
        "tractography_reused_not_regenerated": True,
        "changed_factor": "T1_to_b0_registration_transform_only",
        "baseline": {
            "route": "recovery3_fsl_bbr_primary",
            "edge_density_20m": baseline_density,
            "connected_nodes_20m": int(baseline["connected_nodes"]),
            "disconnected_nodes_20m": [303],
            "endpoint_assignment_fraction_20m": (
                float(baseline["primary_endpoint_assignment_fraction"])
                + float(
                    baseline[
                        "independent_endpoint_assignment_fraction"
                    ]
                )
            )
            / 2.0,
            "summary": BALANCED.file_record(bound["baseline_path"]),
        },
        "counterfactual": {
            "route": bound["route"],
            "atlas_result": BALANCED.file_record(
                attempt
                / "atlas_counterfactual/subjects"
                / UNIT
                / "result.json"
            ),
            "levels": levels,
            "seeds": seed_rows,
        },
        "difference": {
            "edge_density_20m": observed_density - baseline_density,
            "connected_nodes_20m": node_change,
            "endpoint_assignment_fraction_20m": (
                (
                    float(
                        observed[
                            "primary_endpoint_assignment_fraction"
                        ]
                    )
                    + float(
                        observed[
                            "independent_endpoint_assignment_fraction"
                        ]
                    )
                )
                / 2.0
                - (
                    float(
                        baseline[
                            "primary_endpoint_assignment_fraction"
                        ]
                    )
                    + float(
                        baseline[
                            "independent_endpoint_assignment_fraction"
                        ]
                    )
                )
                / 2.0
            ),
        },
        "localization": interpretation,
        "production_route_selected": False,
        "scale_up_authorized": False,
        "requires_visual_review_before_any_route_change": True,
        "implementation": BALANCED.file_record(Path(__file__)),
        "preflight": BALANCED.file_record(attempt / "preflight.json"),
    }
    atomic_json(attempt / "summary.json", summary)
    return {
        "attempt_root": str(attempt),
        "status": summary["status"],
        "localization": interpretation,
        "baseline_density_20m": baseline_density,
        "counterfactual_density_20m": observed_density,
        "baseline_connected_nodes_20m": int(
            baseline["connected_nodes"]
        ),
        "counterfactual_connected_nodes_20m": int(
            observed["connected_nodes"]
        ),
        "counterfactual_disconnected_nodes_20m": observed[
            "disconnected_nodes"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be in 1..16")
    result = execute(args.threads) if args.execute else preflight()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
