#!/usr/bin/env python3
"""Validate blinded human-QC coverage, second ratings, and reliability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path("/home/ec2-user/exp")
DEFAULT_CONTRACT = (
    ROOT / "research_audit/outputs/thesis_grade_530_release_contract_v1/contract.json"
)
FORBIDDEN_FIELDS = {
    "diagnosis",
    "diagnosis_at_dti",
    "group",
    "research_group",
    "outcome",
    "clinical_outcome",
    "disease_status",
}
VALID_ROLES = {"PRIMARY", "SECONDARY", "ADJUDICATOR"}
VALID_RATINGS = {"PASS", "FAIL"}


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
            if normalized in FORBIDDEN_FIELDS:
                failures.append(child_path)
            failures.extend(forbidden_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(forbidden_paths(child, f"{prefix}[{index}]"))
    return failures


def deterministic_audit_sample(units: list[str], salt: str) -> list[str]:
    count = math.ceil(0.20 * len(units))
    ranked = sorted(
        units,
        key=lambda unit: hashlib.sha256(f"{salt}\0{unit}".encode("utf-8")).hexdigest(),
    )
    return sorted(ranked[:count])


def cohen_kappa(pairs: list[tuple[str, str]]) -> tuple[float | None, float, float, str | None]:
    if not pairs:
        return None, 0.0, 0.0, "no_double_ratings"
    n = len(pairs)
    observed = sum(left == right for left, right in pairs) / n
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(
        (left_counts[label] / n) * (right_counts[label] / n)
        for label in VALID_RATINGS
    )
    denominator = 1.0 - expected
    if denominator <= 0:
        return None, observed, expected, "kappa_not_estimable_single_category"
    value = (observed - expected) / denominator
    return float(value), observed, expected, None


def evaluate_reviews(
    scope_path: Path,
    reviews_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    scope_path = scope_path.expanduser().resolve()
    reviews_path = reviews_path.expanduser().resolve()
    contract_path = contract_path.expanduser().resolve()
    scope = load_json(scope_path)
    contract = load_json(contract_path)
    minimum_kappa = float(contract["human_qc_contract"]["minimum_interrater_kappa"])
    failures: list[str] = []

    forbidden_scope = forbidden_paths(scope)
    failures.extend(f"scope_forbidden_blinding_field:{path}" for path in forbidden_scope)
    if scope.get("record_type") != "diagnosis_blind_human_qc_scope":
        failures.append("unexpected_scope_record_type")
    if scope.get("diagnosis_labels_used") is not False:
        failures.append("diagnosis_labels_used_must_be_false")
    expected_units = scope.get("expected_units")
    warning_units = scope.get("automated_warning_units")
    salt = str(scope.get("audit_salt", "")).strip()
    if not isinstance(expected_units, list):
        expected_units = []
        failures.append("expected_units_must_be_a_list")
    expected_units = [str(unit).strip() for unit in expected_units]
    if not expected_units or not all(expected_units) or len(expected_units) != len(set(expected_units)):
        failures.append("expected_units_must_be_nonempty_nonblank_and_unique")
    if not isinstance(warning_units, list):
        warning_units = []
        failures.append("automated_warning_units_must_be_a_list")
    warning_units = [str(unit).strip() for unit in warning_units]
    if len(warning_units) != len(set(warning_units)) or not set(warning_units).issubset(set(expected_units)):
        failures.append("automated_warning_units_are_not_a_unique_subset")
    if not salt:
        failures.append("audit_salt_is_blank")
    audit_units = deterministic_audit_sample(expected_units, salt) if expected_units and salt else []
    second_required = sorted(set(warning_units) | set(audit_units))

    if not reviews_path.is_file() or reviews_path.is_symlink():
        raise ValueError(f"reviews file missing/nonregular: {reviews_path}")
    with reviews_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        required = {"unit", "role", "status", "reviewer", "reviewed_utc"}
        if not required.issubset(fields):
            failures.append("reviews_csv_lacks_required_fields")
        forbidden_fields = sorted(FORBIDDEN_FIELDS.intersection(field.strip().lower() for field in fields))
        failures.extend(f"reviews_forbidden_blinding_field:{field}" for field in forbidden_fields)
        rows = [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]

    indexed: dict[str, dict[str, list[dict[str, str]]]] = {
        unit: {role: [] for role in VALID_ROLES} for unit in expected_units
    }
    unexpected_units: set[str] = set()
    for row_index, row in enumerate(rows, start=2):
        unit = row.get("unit", "")
        role = row.get("role", "").upper()
        rating = row.get("status", "").upper()
        if unit not in indexed:
            unexpected_units.add(unit)
            continue
        if role not in VALID_ROLES:
            failures.append(f"row_{row_index}:invalid_role")
            continue
        if rating not in VALID_RATINGS:
            failures.append(f"row_{row_index}:invalid_status")
        if not row.get("reviewer") or not row.get("reviewed_utc"):
            failures.append(f"row_{row_index}:reviewer_or_time_blank")
        row["role"] = role
        row["status"] = rating
        indexed[unit][role].append(row)
    if unexpected_units:
        failures.append("unexpected_review_units:" + ",".join(sorted(unexpected_units)))

    pairs: list[tuple[str, str]] = []
    unit_results: list[dict[str, Any]] = []
    final_pass = 0
    disagreement_count = 0
    for unit in expected_units:
        unit_failures: list[str] = []
        roles = indexed[unit]
        primary = roles["PRIMARY"]
        secondary = roles["SECONDARY"]
        adjudicator = roles["ADJUDICATOR"]
        if len(primary) != 1:
            unit_failures.append("requires_exactly_one_primary_rating")
        if unit in second_required and len(secondary) != 1:
            unit_failures.append("required_second_rating_missing_or_duplicated")
        if unit not in second_required and len(secondary) > 1:
            unit_failures.append("multiple_optional_second_ratings")
        if len(adjudicator) > 1:
            unit_failures.append("multiple_adjudications")
        primary_rating = primary[0]["status"] if len(primary) == 1 else None
        secondary_rating = secondary[0]["status"] if len(secondary) == 1 else None
        disagreement = (
            primary_rating is not None
            and secondary_rating is not None
            and primary_rating != secondary_rating
        )
        if len(primary) == 1 and len(secondary) == 1:
            if primary[0]["reviewer"] == secondary[0]["reviewer"]:
                unit_failures.append("primary_and_secondary_reviewer_must_differ")
            pairs.append((primary_rating, secondary_rating))
        if disagreement:
            disagreement_count += 1
            if len(adjudicator) != 1:
                unit_failures.append("disagreement_requires_exactly_one_adjudication")
            elif adjudicator[0]["reviewer"] in {
                primary[0]["reviewer"], secondary[0]["reviewer"]
            }:
                unit_failures.append("adjudicator_must_be_independent")
        elif adjudicator:
            unit_failures.append("adjudication_without_disagreement")
        final_rating = (
            adjudicator[0]["status"]
            if disagreement and len(adjudicator) == 1
            else primary_rating
        )
        if final_rating == "PASS":
            final_pass += 1
        unit_results.append(
            {
                "unit": unit,
                "automated_warning": unit in warning_units,
                "deterministic_audit_sample": unit in audit_units,
                "second_rating_required": unit in second_required,
                "primary_status": primary_rating,
                "secondary_status": secondary_rating,
                "disagreement": disagreement,
                "adjudicated_status": adjudicator[0]["status"] if len(adjudicator) == 1 else None,
                "final_status": final_rating,
                "status": "PASS" if not unit_failures else "FAIL",
                "failures": unit_failures,
            }
        )
        failures.extend(f"{unit}:{failure}" for failure in unit_failures)

    kappa, observed, expected, kappa_failure = cohen_kappa(pairs)
    if kappa_failure:
        failures.append(kappa_failure)
    elif kappa is not None and kappa < minimum_kappa:
        failures.append("cohen_kappa_below_contract")
    checks = {
        "diagnosis_and_outcome_blind": not forbidden_scope and scope.get("diagnosis_labels_used") is False,
        "all_expected_units_have_one_primary_rating": bool(expected_units) and all(len(indexed[u]["PRIMARY"]) == 1 for u in expected_units),
        "all_warnings_and_deterministic_20_percent_audit_have_second_rating": all(len(indexed[u]["SECONDARY"]) == 1 for u in second_required),
        "independent_reviewers": all(
            not indexed[u]["SECONDARY"]
            or indexed[u]["PRIMARY"][0]["reviewer"] != indexed[u]["SECONDARY"][0]["reviewer"]
            for u in expected_units
            if len(indexed[u]["PRIMARY"]) == 1
        ),
        "all_disagreements_adjudicated_with_original_ratings_retained": all(
            not item["disagreement"] or item["adjudicated_status"] in VALID_RATINGS
            for item in unit_results
        ),
        "cohen_kappa_is_estimable_and_meets_contract": kappa is not None and kappa >= minimum_kappa,
        "no_unexpected_units": not unexpected_units,
    }
    status = "PASS" if all(checks.values()) and not failures else "FAIL"
    return {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_human_qc_reliability_validation",
        "generated_utc": utc_now(),
        "status": status,
        "diagnosis_labels_used": False,
        "scope": file_record(scope_path),
        "reviews": file_record(reviews_path),
        "contract": file_record(contract_path),
        "validator_source": file_record(Path(__file__)),
        "expected_unit_count": len(expected_units),
        "primary_review_count": sum(len(indexed[u]["PRIMARY"]) for u in expected_units),
        "double_rated_unit_count": len(pairs),
        "automated_warning_unit_count": len(warning_units),
        "deterministic_audit_unit_count": len(audit_units),
        "second_rating_required_count": len(second_required),
        "disagreement_count": disagreement_count,
        "final_pass_count": final_pass,
        "audit_method": {
            "fraction": 0.20,
            "rounding": "ceil",
            "ranking": "ascending SHA256(audit_salt + NUL + unit)",
            "audit_salt_sha256": hashlib.sha256(salt.encode("utf-8")).hexdigest() if salt else None,
            "selected_units": audit_units,
        },
        "cohen_kappa": {
            "value": kappa,
            "minimum_required": minimum_kappa,
            "observed_agreement": observed,
            "expected_agreement": expected,
            "failure": kappa_failure,
            "categories": sorted(VALID_RATINGS),
        },
        "checks": checks,
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
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    report = evaluate_reviews(args.scope, args.reviews, args.contract)
    write_report(args.output, report, overwrite=args.overwrite)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, sort_keys=True))
    return 1 if args.require_pass and report["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
