#!/usr/bin/env python3
"""Apply the predeclared gate to the HCP379 FOD-cutoff screen."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_hroi_fod_cutoff_counterfactual_v1/attempts"
)
PREDECLARATION = (
    EXP
    / "research_audit/outputs/hcp379_fod_cutoff_counterfactual_v1/"
    "predeclaration.json"
)
DEFAULT_VALIDATION = (
    EXP
    / "research_audit/outputs/hcp379_fod_cutoff_counterfactual_v1/"
    "validation.json"
)
DEFAULT_OUTPUT = (
    EXP
    / "research_audit/outputs/hcp379_fod_cutoff_counterfactual_v1/"
    "decision.json"
)
ADVERSE = "007_S_5196_I390043"
CONTROL = "032_S_6804_I1230908"
EXPECTED_UNITS = {ADVERSE, CONTROL}
EXPECTED_SEEDS = {"primary", "independent"}


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


def unit_evidence(
    unit: str, summary: Mapping[str, Any]
) -> dict[str, Any]:
    row = summary["units"][unit]
    baseline = row["baseline_cutoff_0p06"]
    counterfactual = row["counterfactual_cutoff_0p05"]
    baseline_support = baseline["support_complementarity"]
    counterfactual_support = counterfactual[
        "support_complementarity"
    ]
    baseline_seed_edges = {
        seed: int(baseline_support[f"{seed}_supported_edges"])
        for seed in EXPECTED_SEEDS
    }
    counterfactual_seed_edges = {
        seed: int(
            counterfactual_support[f"{seed}_supported_edges"]
        )
        for seed in EXPECTED_SEEDS
    }
    gains = {
        seed: counterfactual_seed_edges[seed]
        - baseline_seed_edges[seed]
        for seed in EXPECTED_SEEDS
    }
    baseline_difference = abs(
        baseline_seed_edges["primary"]
        - baseline_seed_edges["independent"]
    )
    return {
        "unit": unit,
        "role": row["role"],
        "baseline_density_6m": float(baseline["edge_density"]),
        "counterfactual_density_6m": float(
            counterfactual["edge_density"]
        ),
        "density_difference": float(row["difference"]["edge_density"]),
        "baseline_seed_edges": baseline_seed_edges,
        "counterfactual_seed_edges": counterfactual_seed_edges,
        "seed_edge_gains": gains,
        "baseline_seed_support_difference": baseline_difference,
        "baseline_support_jaccard": float(
            baseline_support["support_jaccard"]
        ),
        "counterfactual_support_jaccard": float(
            counterfactual_support["support_jaccard"]
        ),
        "support_jaccard_difference": float(
            row["difference"]["support_jaccard"]
        ),
        "stable_new_edges": int(
            counterfactual["stable_support_change"][
                "stable_new_edges"
            ]
        ),
        "stable_lost_edges": int(
            counterfactual["stable_support_change"][
                "stable_lost_edges"
            ]
        ),
        "counterfactual_connected_nodes": int(
            counterfactual["connected_nodes"]
        ),
        "counterfactual_endpoint_assignment_fraction": float(
            counterfactual["endpoint_assignment_fraction"]
        ),
    }


def evaluate(
    summary_path: Path, validation_path: Path
) -> dict[str, Any]:
    summary = load_json(summary_path)
    validation = load_json(validation_path)
    declaration = load_json(PREDECLARATION)
    if (
        summary.get("record_type")
        != "diagnosis_blind_hcp379_fod_cutoff_counterfactual"
        or summary.get("status")
        != "PASS_COMPLETE_FOD_CUTOFF_COUNTERFACTUAL"
        or set(summary.get("units", {})) != EXPECTED_UNITS
        or validation.get("record_type")
        != "independent_hcp379_fod_cutoff_counterfactual_validation"
        or validation.get("status") != "PASS"
        or Path(str(validation.get("summary", ""))).resolve()
        != summary_path.resolve()
        or declaration.get("status")
        != "PREDECLARED_READY_FOR_BOUNDED_EXECUTION"
        or Path(summary["predeclaration"]["path"]).resolve()
        != PREDECLARATION.resolve()
    ):
        raise ValueError("screen, predeclaration or validation differs")
    criteria = declaration["predeclared_expansion_criteria"]
    evidence = {
        unit: unit_evidence(unit, summary)
        for unit in sorted(EXPECTED_UNITS)
    }
    adverse = evidence[ADVERSE]
    control = evidence[CONTROL]

    def technical_pass(row: Mapping[str, Any]) -> bool:
        return bool(
            row["counterfactual_connected_nodes"]
            == int(criteria["required_connected_nodes_each_unit"])
            and row["counterfactual_endpoint_assignment_fraction"]
            >= float(
                criteria[
                    "minimum_endpoint_assignment_fraction_each_unit"
                ]
            )
            and row["counterfactual_support_jaccard"]
            >= row["baseline_support_jaccard"] - 0.03
        )

    adverse_pass = bool(
        technical_pass(adverse)
        and adverse["density_difference"]
        >= float(criteria["adverse_minimum_absolute_density_gain"])
        and min(adverse["seed_edge_gains"].values())
        > adverse["baseline_seed_support_difference"]
        and adverse["stable_new_edges"] > 0
    )
    control_pass = bool(
        technical_pass(control)
        and control["density_difference"]
        >= -float(criteria["control_maximum_absolute_density_loss"])
        and min(control["seed_edge_gains"].values())
        >= -control["baseline_seed_support_difference"]
    )
    expand = adverse_pass and control_pass
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_fod_cutoff_counterfactual_expansion_gate",
        "status": (
            "EXPAND_FOD_CUTOFF_0P05_SCREEN_TO_REMAINING_FOUR_FAILURES"
            if expand
            else "REJECT_FOD_CUTOFF_0P05_AS_UNIFORM_REMEDY_AT_6M_SCREEN"
        ),
        "summary": str(summary_path.resolve()),
        "independent_validation": str(validation_path.resolve()),
        "predeclaration": str(PREDECLARATION.resolve()),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "predeclared_criteria": criteria,
        "criteria_results": {
            "adverse_case_pass": adverse_pass,
            "passing_control_preserved": control_pass,
        },
        "evidence": evidence,
        "production_recipe_selected": False,
        "scale_up_authorized": False,
        "next_action": (
            "Run the identical paired cutoff-0.05 screen on 009, 036, 126 "
            "and 129. Require the same technical and two-seed support gates "
            "for every case before any 20M/all-nine canary evaluation."
            if expand
            else "Reject cutoff 0.05 as a uniform remedy and return to the "
            "stage-localized candidate audit before any new imaging."
        ),
        "interpretation_boundary": (
            "Passing permits only a bounded four-failure expansion. It does "
            "not establish anatomical truth, select a recipe, relax density "
            "0.60, authorize 530-subject scale-up, or enter inference."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--validation", type=Path, default=DEFAULT_VALIDATION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = (
        args.summary.resolve() if args.summary else latest_summary()
    )
    result = evaluate(summary, args.validation.resolve())
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
