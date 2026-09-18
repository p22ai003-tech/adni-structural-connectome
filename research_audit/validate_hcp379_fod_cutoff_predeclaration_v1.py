#!/usr/bin/env python3
"""Independently validate the bounded HCP379 FOD-cutoff predeclaration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
RUNNER = (
    EXP / "scripts/hcp/run_hcp379_fod_cutoff_counterfactual_v1.py"
)
PREDECLARATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_fod_cutoff_counterfactual_v1/predeclaration.json"
)
GMWMI_DECISION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_gmwmi_seeding_counterfactual_v1/decision.json"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_fod_cutoff_counterfactual_v1/"
    "predeclaration_validation.json"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")
EXPECTED_UNITS = {
    "007_S_5196_I390043",
    "032_S_6804_I1230908",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def record_matches(record: dict[str, Any]) -> bool:
    path = Path(str(record.get("path", "")))
    return (
        path.is_file()
        and not path.is_symlink()
        and int(record.get("size_bytes", -1)) == path.stat().st_size
        and record.get("sha256") == sha256(path)
    )


def main() -> int:
    declaration = load_json(PREDECLARATION)
    decision = load_json(GMWMI_DECISION)
    source = RUNNER.read_text(encoding="utf-8")
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append(
            {"name": name, "pass": bool(passed), "detail": detail}
        )

    check(
        "identity_and_status",
        declaration.get("record_type")
        == "diagnosis_blind_hcp379_fod_cutoff_counterfactual_predeclaration"
        and declaration.get("status")
        == "PREDECLARED_READY_FOR_BOUNDED_EXECUTION",
    )
    check(
        "diagnosis_outcome_and_scaleup_prohibited",
        declaration.get("diagnosis_labels_used") is False
        and declaration.get("outcomes_used") is False
        and declaration.get("imaging_executed") is False
        and declaration.get("production_recipe_selected") is False
        and declaration.get("scale_up_authorized") is False,
    )
    factor = declaration.get("changed_factor_only", {})
    check(
        "one_changed_factor",
        factor
        == {
            "parameter": "normalised_wmfod_termination_cutoff",
            "baseline": 0.06,
            "counterfactual": 0.05,
        },
        factor,
    )
    held = declaration.get("held_fixed", {})
    check(
        "tractography_contract_held_fixed",
        held.get("algorithm") == "iFOD2"
        and held.get("act") is True
        and held.get("backtrack") is True
        and held.get("crop_at_gmwmi") is True
        and held.get("seeding") == "seed_dynamic"
        and held.get("random_seed")
        == "same exact RNG seed as corresponding baseline"
        and held.get("per_seed_streamlines") == 3_000_000
        and held.get("balanced_total_streamlines") == 6_000_000
        and held.get("minimum_length_mm") == 10.0
        and held.get("maximum_length_mm") == 250.0
        and held.get("maximum_angle_degrees") == 45.0
        and held.get("threads_per_job") == 16,
        held,
    )
    check(
        "exact_adverse_control_pair",
        set(declaration.get("units", [])) == EXPECTED_UNITS
        and set(declaration.get("roles", {})) == EXPECTED_UNITS,
    )
    evidence = declaration.get("mechanistic_evidence", {})
    evidence_ok = set(evidence) == EXPECTED_UNITS
    for row in evidence.values():
        total = int(row.get("wm_voxels_n", -1))
        band = int(row.get("peak_0p05_to_0p06_n", -1))
        fraction = float(row.get("peak_0p05_to_0p06_fraction", -1))
        evidence_ok = (
            evidence_ok
            and total > 0
            and 0 <= band <= total
            and abs(fraction - band / total) < 1.0e-15
        )
    evidence_ok = (
        evidence_ok
        and evidence["007_S_5196_I390043"][
            "peak_0p05_to_0p06_fraction"
        ]
        > evidence["032_S_6804_I1230908"][
            "peak_0p05_to_0p06_fraction"
        ]
    )
    check("mechanistic_evidence_arithmetic", evidence_ok, evidence)
    criteria = declaration.get("predeclared_expansion_criteria", {})
    check(
        "stopping_and_expansion_criteria_bound",
        criteria.get("required_connected_nodes_each_unit") == 379
        and criteria.get(
            "minimum_endpoint_assignment_fraction_each_unit"
        )
        == 0.85
        and criteria.get("adverse_minimum_absolute_density_gain") == 0.02
        and criteria.get("control_maximum_absolute_density_loss") == 0.005
        and criteria.get("stable_new_edges_required_for_adverse") is True
        and "only the same cutoff test on the remaining four"
        in declaration.get("decision_boundary", ""),
        criteria,
    )
    check(
        "prior_gmwmi_rejection_bound",
        decision.get("status")
        == "REJECT_GMWMI_AS_UNIFORM_REMEDY_AT_6M_SCREEN"
        and declaration.get("prior_counterfactual_decision", {}).get(
            "path"
        )
        == str(GMWMI_DECISION)
        and record_matches(
            declaration.get("prior_counterfactual_decision", {})
        ),
    )
    check(
        "implementation_record_current",
        declaration.get("implementation", {}).get("path") == str(RUNNER)
        and record_matches(declaration.get("implementation", {})),
    )
    run_seed_source = source[
        source.index("def run_seed(") : source.index(
            "\ndef execute(", source.index("def run_seed(")
        )
    ]
    check(
        "runtime_changes_cutoff_only",
        '"-seed_dynamic"' in run_seed_source
        and '"-seed_gmwmi"' not in run_seed_source
        and "str(COUNTERFACTUAL_CUTOFF)" in run_seed_source
        and "seed[\"source_rng_seed\"]" in run_seed_source
        and '"45.0"' in run_seed_source
        and '"10.0"' in run_seed_source
        and '"250.0"' in run_seed_source,
    )
    attempts = (
        Path("/data/derivatives/hcp379_v2")
        / "phase_b_hroi_fod_cutoff_counterfactual_v1/attempts"
    )
    before = (
        sorted(path.name for path in attempts.iterdir())
        if attempts.is_dir()
        else []
    )
    completed = subprocess.run(
        [str(PYTHON), str(RUNNER)],
        check=True,
        capture_output=True,
        text=True,
    )
    replay = json.loads(completed.stdout)
    after = (
        sorted(path.name for path in attempts.iterdir())
        if attempts.is_dir()
        else []
    )
    check(
        "read_only_preflight_replay",
        replay.get("status") == declaration.get("status")
        and replay.get("changed_factor_only")
        == declaration.get("changed_factor_only")
        and replay.get("mechanistic_evidence")
        == declaration.get("mechanistic_evidence")
        and before == after,
    )
    passed = sum(int(row["pass"]) for row in checks)
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "independent_hcp379_fod_cutoff_predeclaration_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "predeclaration": str(PREDECLARATION),
        "imaging_executed": False,
        "scale_up_authorized": False,
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
