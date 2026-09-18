#!/usr/bin/env python3
"""Read-only evaluator for the strict thesis-grade 530-case release contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path("/home/ec2-user/exp")
DEFAULT_CONTRACT = (
    ROOT / "research_audit/outputs/thesis_grade_530_release_contract_v1/contract.json"
)
DEFAULT_MANIFEST = ROOT / "research_audit/outputs/connectome_v2_input_manifest_v2.csv"
STABILITY_VALIDATOR_SOURCE = (
    ROOT / "research_audit/validate_phase_b_recipe_stability.py"
)
HUMAN_QC_VALIDATOR_SOURCE = (
    ROOT / "research_audit/validate_human_qc_reliability.py"
)
sys.path.insert(0, str(ROOT / "scforge"))

from scforge.provenance import (  # noqa: E402
    REQUIRED_OUTCOMES,
    validate_schema,
    validate_terminal_semantics,
)
from scforge.qc import qc_connectome_bundle  # noqa: E402


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


def record_path(record: Mapping[str, Any], *, label: str) -> Path:
    try:
        path = Path(str(record["path"])).expanduser().resolve()
        expected_size = int(record["size_bytes"])
        expected_sha = str(record["sha256"])
    except Exception as exc:
        raise ValueError(f"{label}: malformed file record") from exc
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label}: missing/nonregular artifact: {path}")
    if path.stat().st_size != expected_size:
        raise ValueError(f"{label}: size differs: {path}")
    if len(expected_sha) != 64:
        raise ValueError(f"{label}: SHA256 field malformed")
    return path


def verify_record(record: Mapping[str, Any], *, label: str, full_hash: bool) -> Path:
    path = record_path(record, label=label)
    if full_hash and sha256_file(path) != record["sha256"]:
        raise ValueError(f"{label}: SHA256 differs: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def write_immutable(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def safe_relative(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def deep_matrix_replay(
    record: Mapping[str, Any],
    *,
    matrix_names: list[str],
    thresholds: Mapping[str, Any],
    full_hash: bool,
) -> list[str]:
    failures: list[str] = []
    outputs = record.get("outputs", {})
    intermediate = record.get("intermediate_files", {})
    matrix_paths: dict[str, str] = {}
    for name in matrix_names:
        item = outputs.get(name)
        if not isinstance(item, Mapping):
            failures.append(f"{name}:missing_output_record")
            continue
        try:
            matrix_paths[name] = str(
                verify_record(item, label=f"matrix {name}", full_hash=full_hash)
            )
        except ValueError as exc:
            failures.append(str(exc))
    volume_record = intermediate.get("node_volumes")
    if not isinstance(volume_record, Mapping):
        failures.append("node_volumes:missing_record")
        return failures
    try:
        volume_path = verify_record(
            volume_record, label="node volumes", full_hash=full_hash
        )
        with volume_path.open(newline="", encoding="utf-8") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda row: int(row["node_index"]))
        volumes = np.asarray([float(row["node_volume_mm3"]) for row in rows])
    except Exception as exc:
        failures.append(f"node_volumes:{exc}")
        return failures
    if len(matrix_paths) != len(matrix_names):
        return failures
    replay = qc_connectome_bundle(
        matrix_paths,
        volumes,
        expected_nodes=int(thresholds["matrix_shape"][0]),
        symmetry_tolerance=float(thresholds["matrix_symmetry_absolute_tolerance"]),
        zero_diagonal_tolerance=float(
            thresholds["matrix_zero_diagonal_absolute_tolerance"]
        ),
        nonnegative_tolerance=1.0e-12,
        integer_tolerance=float(thresholds["count_integer_absolute_tolerance"]),
        duplicate_rtol=1.0e-10,
        duplicate_atol=1.0e-12,
        tensor_identity_tolerance=1.0e-8,
        diffusivity_hard_max_mm2_per_s=float(
            thresholds["diffusivity_range_mm2_per_s"][1]
        ),
    ).to_dict()
    if replay.get("status") != "PASS" or replay.get("failures"):
        failures.extend(f"matrix_replay:{item}" for item in replay.get("failures", []))
    return failures


def strict_validation_record(
    path: Path,
    *,
    kind: str,
    contract_path: Path,
    contract: Mapping[str, Any],
    full_hash: bool,
) -> tuple[bool, list[str]]:
    """Reject forged or stale publication-level reliability PASS records."""

    failures: list[str] = []
    try:
        record = load_json(path)
    except Exception as exc:
        return False, [f"parse:{exc}"]
    expected_type = {
        "stability": "diagnosis_blind_phase_b_recipe_stability_validation",
        "human_qc": "diagnosis_blind_human_qc_reliability_validation",
    }[kind]
    expected_source = {
        "stability": STABILITY_VALIDATOR_SOURCE,
        "human_qc": HUMAN_QC_VALIDATOR_SOURCE,
    }[kind]
    if record.get("record_type") != expected_type:
        failures.append("record_type_differs")
    if record.get("status") != "PASS":
        failures.append("status_not_pass")
    if record.get("diagnosis_labels_used") is not False:
        failures.append("diagnosis_blinding_not_explicit")
    if record.get("failure_count") != 0 or record.get("failures") != []:
        failures.append("record_contains_failures")
    checks = record.get("checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        failures.append("checks_are_not_all_true")
    if record.get("contract") != file_record(contract_path):
        failures.append("contract_binding_differs")
    if record.get("validator_source") != file_record(expected_source):
        failures.append("validator_source_binding_differs")

    nested_names = (
        ("manifest", "contract", "validator_source")
        if kind == "stability"
        else ("scope", "reviews", "contract", "validator_source")
    )
    for name in nested_names:
        nested = record.get(name)
        if not isinstance(nested, Mapping):
            failures.append(f"{name}_file_record_missing")
            continue
        try:
            verify_record(
                nested,
                label=f"{kind} {name}",
                full_hash=full_hash,
            )
        except ValueError as exc:
            failures.append(str(exc))

    unit_results = record.get("unit_results")
    if not isinstance(unit_results, list):
        unit_results = []
        failures.append("unit_results_missing")
    units = [str(item.get("unit", "")) for item in unit_results if isinstance(item, Mapping)]
    if len(units) != len(unit_results) or not all(units) or len(units) != len(set(units)):
        failures.append("unit_results_not_unique_and_complete")
    if not all(
        isinstance(item, Mapping)
        and item.get("status") == "PASS"
        and item.get("failures") == []
        for item in unit_results
    ):
        failures.append("one_or_more_unit_results_not_pass")

    if kind == "stability":
        gate = contract["diagnosis_blind_recipe_stability_gate"]
        if record.get("thresholds") != gate:
            failures.append("stability_thresholds_differ_from_contract")
        if (
            not isinstance(record.get("unit_count"), int)
            or record["unit_count"] < 15
            or record.get("passed_unit_count") != record.get("unit_count")
            or len(unit_results) != record.get("unit_count")
        ):
            failures.append("stability_requires_at_least_15_all_pass_units")
        if not all(item.get("comparison_count") == 4 for item in unit_results):
            failures.append("stability_required_comparisons_incomplete")
        aggregate = record.get("aggregate")
        if not isinstance(aggregate, Mapping):
            aggregate = {}
            failures.append("stability_aggregate_missing")
        numeric_gates = (
            (
                "minimum_support_matched_spearman",
                "minimum_support_matched_upper_triangle_spearman",
                "minimum",
            ),
            (
                "minimum_tensor_pearson",
                "minimum_tensor_matrix_upper_triangle_pearson",
                "minimum",
            ),
            (
                "minimum_node_strength_icc_a1",
                "minimum_node_strength_absolute_agreement_icc",
                "minimum",
            ),
            (
                "maximum_global_metric_relative_difference",
                "maximum_global_metric_relative_difference",
                "maximum",
            ),
            (
                "maximum_assignment_fraction_absolute_difference",
                "maximum_assignment_fraction_absolute_difference",
                "maximum",
            ),
        )
        for metric, threshold_name, direction in numeric_gates:
            try:
                value = float(aggregate[metric])
                threshold = float(gate[threshold_name])
            except Exception:
                failures.append(f"{metric}_missing_or_non_numeric")
                continue
            if not np.isfinite(value):
                failures.append(f"{metric}_nonfinite")
            elif direction == "minimum" and value < threshold:
                failures.append(f"{metric}_below_contract")
            elif direction == "maximum" and value > threshold:
                failures.append(f"{metric}_above_contract")
        evidence = record.get("matrix_evidence")
        expected_evidence_count = int(record.get("unit_count", 0)) * 4 * 6
        if not isinstance(evidence, Mapping) or len(evidence) != expected_evidence_count:
            failures.append("matrix_evidence_count_differs")
        elif full_hash:
            for label, item in evidence.items():
                if not isinstance(item, Mapping):
                    failures.append(f"matrix_evidence:{label}:malformed")
                    continue
                try:
                    verify_record(
                        item,
                        label=f"stability matrix evidence {label}",
                        full_hash=True,
                    )
                except ValueError as exc:
                    failures.append(str(exc))
        artifact_evidence = record.get("artifact_evidence")
        expected_artifact_count = int(record.get("unit_count", 0)) * 4 * 4
        if (
            not isinstance(artifact_evidence, Mapping)
            or len(artifact_evidence) != expected_artifact_count
        ):
            failures.append("stability_artifact_evidence_count_differs")
        elif full_hash:
            for label, item in artifact_evidence.items():
                if not isinstance(item, Mapping):
                    failures.append(f"stability_artifact_evidence:{label}:malformed")
                    continue
                try:
                    verify_record(
                        item,
                        label=f"stability artifact evidence {label}",
                        full_hash=True,
                    )
                except ValueError as exc:
                    failures.append(str(exc))
    else:
        minimum_kappa = float(contract["human_qc_contract"]["minimum_interrater_kappa"])
        if (
            record.get("expected_unit_count") != 530
            or record.get("primary_review_count") != 530
            or record.get("final_pass_count") != 530
            or len(unit_results) != 530
        ):
            failures.append("human_qc_requires_530_primary_and_final_pass_reviews")
        if int(record.get("second_rating_required_count", -1)) < math.ceil(0.20 * 530):
            failures.append("human_qc_second_rating_scope_below_20_percent")
        if record.get("double_rated_unit_count") != record.get("second_rating_required_count"):
            failures.append("human_qc_double_rating_count_differs_from_required")
        kappa_record = record.get("cohen_kappa")
        try:
            kappa = float(kappa_record["value"])
        except Exception:
            kappa = float("nan")
        if not np.isfinite(kappa) or kappa < minimum_kappa:
            failures.append("human_qc_kappa_below_or_not_estimable")
        if not all(item.get("final_status") == "PASS" for item in unit_results):
            failures.append("human_qc_final_status_not_pass_for_all_units")
    return not failures, list(dict.fromkeys(failures))


def evaluate_case(
    path: Path,
    *,
    expected_unit: str,
    run_root: Path,
    contract: Mapping[str, Any],
    full_hash: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        record = load_json(path)
    except Exception as exc:
        return {"unit": expected_unit, "green": False, "failures": [f"parse:{exc}"]}
    if record.get("unit") != expected_unit:
        failures.append(f"unit_mismatch:{record.get('unit')}")
    if record.get("status") != "PASS":
        failures.append(f"terminal_status:{record.get('status')}")
    if record.get("terminal_stage") != "complete":
        failures.append(f"terminal_stage:{record.get('terminal_stage')}")
    if record.get("primary_failure_reason") is not None:
        failures.append("primary_failure_reason_present")
    source = record.get("source_identity", {})
    if not isinstance(source, Mapping):
        failures.append("source_identity_missing")
    elif f"{source.get('subject_id')}_I{source.get('dti_image_id')}" != expected_unit:
        failures.append("source_identity_unit_mismatch")

    schema_record = record.get("contract_files", {}).get("provenance_schema")
    if not isinstance(schema_record, Mapping):
        failures.append("provenance_schema_record_missing")
    else:
        try:
            schema_path = verify_record(
                schema_record, label="provenance schema", full_hash=full_hash
            )
            failures.extend(f"schema:{item}" for item in validate_schema(record, schema_path))
        except ValueError as exc:
            failures.append(str(exc))
    failures.extend(
        f"semantic:{item}"
        for item in validate_terminal_semantics(
            record,
            required_outcomes=REQUIRED_OUTCOMES,
            required_matrices=contract["matrix_names"],
        )
    )

    qc = record.get("qc", {})
    if not isinstance(qc, Mapping):
        failures.append("qc_missing")
        qc = {}
    for name in ("gradient", "atlas", "spatial", "preflight", "matrix"):
        value = qc.get(name)
        if not isinstance(value, Mapping) or value.get("status") != "PASS":
            failures.append(f"{name}_qc_not_pass")
    preflight = qc.get("preflight", {})
    if not isinstance(preflight, Mapping) or preflight.get("human_visual_qc") != "PASS":
        failures.append("human_visual_qc_not_pass")
    matrix_qc = qc.get("matrix", {})
    thresholds = contract["fixed_thresholds"]
    if isinstance(matrix_qc, Mapping):
        if float(matrix_qc.get("endpoint_assignment_fraction", -1)) < float(
            thresholds["endpoint_assignment_fraction_minimum"]
        ):
            failures.append("endpoint_assignment_fraction_below_contract")
        if int(matrix_qc.get("unique_assigned_nodes", -1)) < int(
            thresholds["unique_endpoint_assigned_nodes_minimum"]
        ):
            failures.append("unique_endpoint_nodes_below_contract")
        if float(matrix_qc.get("top5_endpoint_fraction", 2)) > float(
            thresholds["top5_endpoint_fraction_maximum"]
        ):
            failures.append("top5_endpoint_concentration_above_contract")
        if int(matrix_qc.get("tractogram_streamlines", -1)) != int(
            thresholds["tractogram_streamlines"]
        ):
            failures.append("tractogram_streamline_count_differs")
        if int(matrix_qc.get("sift2_weight_count", -1)) != int(
            thresholds["tractogram_streamlines"]
        ):
            failures.append("sift2_weight_count_differs")

    outputs = record.get("outputs", {})
    required_output_names = {"tracks", "sift2_weights", "sift2_mu", *contract["matrix_names"]}
    if not isinstance(outputs, Mapping):
        failures.append("outputs_missing")
    else:
        if not required_output_names.issubset(outputs):
            failures.append("required_output_records_missing")
        for name in required_output_names & set(outputs):
            item = outputs[name]
            if not isinstance(item, Mapping):
                failures.append(f"{name}:record_not_mapping")
                continue
            try:
                output_path = verify_record(
                    item, label=f"output {name}", full_hash=full_hash
                )
                if not safe_relative(output_path, run_root):
                    failures.append(f"{name}:outside_run_root")
            except ValueError as exc:
                failures.append(str(exc))
    failures.extend(
        deep_matrix_replay(
            record,
            matrix_names=list(contract["matrix_names"]),
            thresholds=thresholds,
            full_hash=full_hash,
        )
    )
    return {
        "unit": expected_unit,
        "green": not failures,
        "failure_count": len(failures),
        "failures": list(dict.fromkeys(failures)),
        "terminal_record": file_record(path) if path.is_file() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verification-mode", choices=("metadata", "full"), default="metadata")
    parser.add_argument("--require-green", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluator output: {out_dir}")
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    contract = load_json(contract_path)
    if contract.get("record_type") != "thesis_grade_structural_connectome_530_release_contract":
        raise ValueError("unexpected release contract")
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        manifest_rows = list(csv.DictReader(handle))
    expected_units = [
        f"{row['subject_id']}_I{row['dti_image_id']}" for row in manifest_rows
    ]
    terminal_paths = sorted(run_root.glob("subjects/*/08_qc/terminal_record.json"))
    terminal_by_unit = {path.parents[1].name: path for path in terminal_paths}
    duplicates = len(terminal_by_unit) != len(terminal_paths)
    full_hash = args.verification_mode == "full"
    case_results = []
    for unit in expected_units:
        path = terminal_by_unit.get(unit)
        if path is None:
            case_results.append(
                {"unit": unit, "green": False, "failure_count": 1, "failures": ["terminal_record_missing"]}
            )
            continue
        case_results.append(
            evaluate_case(
                path,
                expected_unit=unit,
                run_root=run_root,
                contract=contract,
                full_hash=full_hash,
            )
        )
    unexpected_units = sorted(set(terminal_by_unit) - set(expected_units))
    green_count = sum(bool(item["green"]) for item in case_results)
    failure_reasons = Counter(
        failure
        for item in case_results
        for failure in item.get("failures", [])
    )
    stability_path = run_root / "publication/recipe_stability_validation.json"
    reliability_path = run_root / "publication/human_qc_reliability.json"
    stability_pass, stability_failures = strict_validation_record(
        stability_path,
        kind="stability",
        contract_path=contract_path,
        contract=contract,
        full_hash=full_hash,
    )
    reliability_pass, reliability_failures = strict_validation_record(
        reliability_path,
        kind="human_qc",
        contract_path=contract_path,
        contract=contract,
        full_hash=full_hash,
    )

    checks = {
        "manifest_has_exact_530_unique_units": len(expected_units) == 530
        and len(set(expected_units)) == 530,
        "terminal_record_count_is_530": len(terminal_paths) == 530,
        "no_duplicate_terminal_units": not duplicates,
        "no_unexpected_terminal_units": not unexpected_units,
        "all_530_cases_green": green_count == 530,
        "diagnosis_blind_recipe_stability_pass": stability_pass,
        "human_qc_reliability_pass": reliability_pass,
        "full_sha256_verification_mode": full_hash,
    }
    technical_green = all(checks.values())
    report = {
        "schema_version": "1.0.0",
        "record_type": "thesis_grade_530_release_evaluation",
        "generated_utc": utc_now(),
        "status": "PASS" if technical_green else "NOT_GREEN",
        "technical_green_530": technical_green,
        "run_root": str(run_root),
        "verification_mode": args.verification_mode,
        "expected_cases": 530,
        "terminal_record_count": len(terminal_paths),
        "green_case_count": green_count,
        "non_green_case_count": 530 - green_count,
        "checks": checks,
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "publication_gate_failures": {
            "recipe_stability": stability_failures,
            "human_qc_reliability": reliability_failures,
        },
        "unexpected_units": unexpected_units,
        "contract": file_record(contract_path),
        "manifest": file_record(manifest_path),
        "evaluator_source": file_record(Path(__file__)),
        "recipe_stability_validation": file_record(stability_path) if stability_path.is_file() else None,
        "human_qc_reliability": file_record(reliability_path) if reliability_path.is_file() else None,
        "case_results": case_results,
        "copy_green": False,
        "inference_green": False,
        "thesis_green": False,
        "goal_complete": False,
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    report_path = out_dir / "evaluation.json"
    write_immutable(
        report_path,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    summary = "\n".join(
        [
            "# Thesis-grade 530 release evaluation",
            "",
            f"**Status:** {report['status']}",
            f"**GREEN cases:** {green_count}/530",
            f"**Terminal records:** {len(terminal_paths)}/530",
            f"**Verification mode:** {args.verification_mode}",
            "",
            *[f"- [{'x' if value else ' '}] {name}" for name, value in checks.items()],
            "",
            "This evaluator never upgrades PARTIAL/FAIL records and never treats absence as a zero-valued connectome.",
        ]
    ) + "\n"
    write_immutable(out_dir / "evaluation.md", summary.encode("utf-8"))
    print(
        json.dumps(
            {
                "status": report["status"],
                "green_cases": green_count,
                "terminal_records": len(terminal_paths),
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )
    if args.require_green and not technical_green:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
