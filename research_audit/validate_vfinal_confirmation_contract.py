#!/usr/bin/env python3
"""Statically validate the prospectively frozen V-final confirmation contract.

This validator performs no image processing, participant analysis, tractography,
matrix generation, or diagnosis-effect estimation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / "outputs" / "vfinal_confirmation_contract_v1"
CONTRACT = RELEASE / "confirmation_contract.json"
NARRATIVE = RELEASE / "confirmation_contract.md"
OUT_JSON = RELEASE / "validation.json"
OUT_MD = RELEASE / "validation.md"

HASHED_SOURCES = {
    "stage_specific_diffusivity_analysis.py": ROOT / "stage_specific_diffusivity_analysis.py",
    "primary_inference_soundness_audit.py": ROOT / "primary_inference_soundness_audit.py",
    "stage_specific_diffusivity_v2_run_manifest.json": ROOT / "outputs" / "stage_specific_diffusivity_v2" / "run_manifest.json",
    "primary_inference_soundness_v2_audit_report.json": ROOT / "outputs" / "primary_inference_soundness_v2" / "audit_report.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate() -> dict[str, object]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    narrative = NARRATIVE.read_text(encoding="utf-8")
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: object) -> None:
        checks.append({"name": name, "pass": bool(condition), "detail": str(detail)})

    systems = contract["primary_system_inventory"]
    endpoints = contract["primary_endpoints"]
    model = contract["primary_support_and_model"]
    success = contract["confirmation_success_rule"]
    source_hashes = contract["frozen_source_hashes"]

    check("contract_frozen_before_results",
          contract["status"] == "FROZEN_BEFORE_CORRECTED_CONNECTOME_RESULTS"
          and contract["execution_authorized"] is False,
          contract["status"])
    check("exact_two_primary_endpoints",
          [row["id"] for row in endpoints] == ["AxD_MCI_minus_CN", "RD_AD_minus_MCI"]
          and all(row["expected_direction"] == "positive" for row in endpoints),
          [row["id"] for row in endpoints])
    check("nonredundant_17_system_inventory",
          len(systems) == 17 and len(set(systems)) == 17
          and "functional::Brainstem" not in systems
          and "functional::Cerebellar" not in systems,
          len(systems))
    check("pair_common_primary_scope",
          model["support_scope"] == "pair-common sites separately for each adjacent-stage contrast",
          model["support_scope"])
    check("small_cluster_inference_frozen",
          model["covariance"].startswith("CR1")
          and model["reference_distribution"] == "two-sided t with df = number of sites - 1"
          and model["primary_resampling"] == "restricted Rademacher wild-cluster bootstrap-t"
          and model["bootstrap_iterations"] == 9999
          and model["bootstrap_seed"] == 20260719,
          {key: model[key] for key in ("bootstrap_iterations", "bootstrap_seed")})
    check("two_endpoint_holm_family",
          model["multiplicity"] == "Holm across the two locked endpoints as one family"
          and model["alpha_two_sided"] == 0.05,
          model["multiplicity"])
    check("success_requires_both_positive_and_adjusted",
          success["both_effects_positive"] is True
          and success["both_restricted_wild_cluster_holm_p_at_most"] == 0.05
          and success["classification_if_all_true"] == "PROCESSING_CONFIRMED_INTERNAL",
          success["classification_if_all_true"])
    check("same_cohort_limitation_explicit",
          "not independent replication" in contract["purpose"].lower()
          and "same source cohort" in contract["claim_policy"]["allowed_after_success"].lower()
          and "not independent or external replication" in narrative.lower(),
          "same-cohort processing confirmation only")
    check("technical_gate_has_no_fallbacks",
          contract["technical_input_gate"]["recipe_must_be_uniform_across_subjects"] is True
          and contract["technical_input_gate"]["no_subject_specific_route_switching"] is True
          and contract["technical_input_gate"]["no_padding_clipping_or_tensor_imputation"] is True
          and contract["cohort"]["outcome_guided_subject_or_site_exclusion_allowed"] is False,
          "uniform route and outcome-blind exclusions")
    check("phase_separation_explicit",
          "cannot launch Phase B" in narrative
          and "Phase B tractography and matrix generation remain under a separate human gate" in narrative,
          "no Phase-B authority")

    observed_hashes = {name: sha256(path) for name, path in HASHED_SOURCES.items()}
    check("frozen_source_hashes_match", observed_hashes == source_hashes,
          {name: observed_hashes[name] == source_hashes.get(name) for name in observed_hashes})

    passed = all(row["pass"] for row in checks)
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pass": passed,
        "checks_passed": sum(bool(row["pass"]) for row in checks),
        "checks_total": len(checks),
        "checks": checks,
        "contract_sha256": sha256(CONTRACT),
        "narrative_sha256": sha256(NARRATIVE),
        "execution_authorized": False,
        "interpretation": "PASS validates a frozen statistical decision contract only; no corrected result has been generated or tested."
    }


def main() -> None:
    report = validate()
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# V-final confirmation-contract validation",
        "",
        f"**Generated:** {report['generated_utc']}",
        f"**Verdict:** {'PASS' if report['pass'] else 'FAIL'}",
        f"**Checks:** {report['checks_passed']}/{report['checks_total']}",
        "",
    ]
    for row in report["checks"]:
        lines.append(f"- [{'x' if row['pass'] else ' '}] **{row['name']}** — {row['detail']}")
    lines.extend(("", report["interpretation"]))
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "checks": f"{report['checks_passed']}/{report['checks_total']}"}))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
