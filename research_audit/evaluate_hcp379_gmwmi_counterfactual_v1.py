#!/usr/bin/env python3
"""Apply a predeclared expansion gate to the HCP379 GMWMI screen.

The screen may advance only when both adverse-case seeds improve beyond the
adverse baseline's seed-to-seed support difference and the passing control
does not lose support beyond its own baseline seed difference. Technical node
coverage and endpoint-assignment gates must also remain satisfied.
"""

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
    "phase_b_hroi_gmwmi_seeding_counterfactual_v1/attempts"
)
DEFAULT_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_gmwmi_seeding_counterfactual_v1/validation.json"
)
DEFAULT_OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_gmwmi_seeding_counterfactual_v1/decision.json"
)
ADVERSE = "007_S_5196_I390043"
CONTROL = "032_S_6804_I1230908"
EXPECTED_UNITS = {ADVERSE, CONTROL}
MINIMUM_ASSIGNMENT = 0.85
EXPECTED_NODES = 379


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
            == "diagnosis_blind_hcp379_gmwmi_seeding_counterfactual"
            and value.get("status")
            == "PASS_COMPLETE_GMWMI_SEEDING_COUNTERFACTUAL"
        ):
            return path.resolve()
    raise FileNotFoundError("complete GMWMI counterfactual summary absent")


def unit_evidence(
    unit: str,
    summary: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    density_summary = load_json(
        Path(
            preflight["inputs"][unit]["density_summary"]["path"]
        ).resolve()
    )
    baseline = density_summary["levels"]["6000000"]
    baseline_support = baseline["support_complementarity"]
    row = summary["units"][unit]
    gmwmi = row["gmwmi_counterfactual"]
    baseline_seed_edges = {
        "primary": int(baseline_support["primary_supported_edges"]),
        "independent": int(
            baseline_support["independent_supported_edges"]
        ),
    }
    gmwmi_seed_edges = {
        seed: int(gmwmi["seeds"][seed]["technical_qc"]["supported_edges"])
        for seed in ("primary", "independent")
    }
    gains = {
        seed: gmwmi_seed_edges[seed] - baseline_seed_edges[seed]
        for seed in ("primary", "independent")
    }
    baseline_variability = abs(
        baseline_seed_edges["primary"]
        - baseline_seed_edges["independent"]
    )
    technical_pass = bool(
        int(gmwmi["connected_nodes"]) == EXPECTED_NODES
        and float(gmwmi["endpoint_assignment_fraction"])
        >= MINIMUM_ASSIGNMENT
    )
    return {
        "unit": unit,
        "baseline_seed_edges": baseline_seed_edges,
        "gmwmi_seed_edges": gmwmi_seed_edges,
        "seed_edge_gains": gains,
        "baseline_seed_support_difference": baseline_variability,
        "dynamic_density_6m": float(
            row["dynamic_seed_baseline"]["edge_density"]
        ),
        "gmwmi_density_6m": float(gmwmi["edge_density"]),
        "density_difference": float(row["difference"]["edge_density"]),
        "gmwmi_connected_nodes": int(gmwmi["connected_nodes"]),
        "gmwmi_endpoint_assignment_fraction": float(
            gmwmi["endpoint_assignment_fraction"]
        ),
        "technical_pass": technical_pass,
    }


def evaluate(
    summary_path: Path, validation_path: Path
) -> dict[str, Any]:
    summary = load_json(summary_path)
    validation = load_json(validation_path)
    if (
        summary.get("record_type")
        != "diagnosis_blind_hcp379_gmwmi_seeding_counterfactual"
        or summary.get("status")
        != "PASS_COMPLETE_GMWMI_SEEDING_COUNTERFACTUAL"
        or set(summary.get("units", {})) != EXPECTED_UNITS
        or validation.get("record_type")
        != "independent_hcp379_gmwmi_seeding_counterfactual_validation"
        or validation.get("status") != "PASS"
        or Path(str(validation.get("summary", ""))).resolve()
        != summary_path.resolve()
    ):
        raise ValueError("counterfactual or independent validation differs")
    preflight = load_json(Path(summary["preflight"]["path"]).resolve())
    evidence = {
        unit: unit_evidence(unit, summary, preflight)
        for unit in sorted(EXPECTED_UNITS)
    }
    adverse = evidence[ADVERSE]
    control = evidence[CONTROL]
    adverse_consistent_gain = bool(
        adverse["technical_pass"]
        and adverse["density_difference"] > 0
        and min(adverse["seed_edge_gains"].values())
        > adverse["baseline_seed_support_difference"]
    )
    control_preserved = bool(
        control["technical_pass"]
        and min(control["seed_edge_gains"].values())
        >= -control["baseline_seed_support_difference"]
    )
    expand = adverse_consistent_gain and control_preserved
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_gmwmi_counterfactual_expansion_gate",
        "status": (
            "EXPAND_BOUNDED_GMWMI_SCREEN_TO_REMAINING_FAILURES"
            if expand
            else "REJECT_GMWMI_AS_UNIFORM_REMEDY_AT_6M_SCREEN"
        ),
        "summary": str(summary_path.resolve()),
        "independent_validation": str(validation_path.resolve()),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "predeclared_criteria": {
            "minimum_endpoint_assignment": MINIMUM_ASSIGNMENT,
            "required_connected_nodes": EXPECTED_NODES,
            "adverse_both_seed_gains_must_exceed_baseline_seed_difference": (
                True
            ),
            "control_each_seed_loss_must_not_exceed_baseline_seed_difference": (
                True
            ),
        },
        "criteria_results": {
            "adverse_consistent_gain": adverse_consistent_gain,
            "passing_control_preserved": control_preserved,
        },
        "evidence": evidence,
        "production_recipe_selected": False,
        "scale_up_authorized": False,
        "next_action": (
            "Run the identical two-seed 3M GMWMI screen on 009, 036, 126 "
            "and 129; require the same gate on every case before any 10M "
            "per-seed all-nine canary expansion."
            if expand
            else "Reject GMWMI as the uniform remedy and predeclare the next "
            "one-factor tractography counterfactual before execution."
        ),
        "interpretation_boundary": (
            "This gate permits or rejects only a bounded technical expansion. "
            "It does not establish anatomical truth, select a production "
            "recipe, change the 0.60 target, or authorize 530-subject scale-up."
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
    summary = args.summary.resolve() if args.summary else latest_summary()
    result = evaluate(summary, args.validation.resolve())
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
