#!/usr/bin/env python3
"""Independently validate the selected-HROI all-nine canary runner."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP
    / "scripts/hcp/"
    "build_hcp379_hroi_balanced_all_nine_canary_v1.py"
)
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_hroi_balanced_all_nine_canary_v1/attempts"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_hroi_balanced_all_nine_canary_v1/validation.json"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_module(SOURCE, "hcp379_hroi_all_nine_validation_runner")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def latest_summary() -> Path | None:
    paths = sorted(ATTEMPTS.glob("*/summary.json"))
    return paths[-1].resolve() if paths else None


def validate_runtime(path: Path) -> dict[str, Any]:
    summary = load_json(path)
    failures: list[str] = []
    if (
        summary.get("record_type")
        != "diagnosis_blind_hcp379_hroi_balanced_all_nine_canary_summary"
        or summary.get("diagnosis_labels_used") is not False
        or summary.get("non_overwriting") is not True
        or summary.get("source_files_modified") is not False
        or summary.get("probabilistic_tractography_generated") is not False
        or summary.get("source_10m_tractograms_reused") is not True
        or summary.get("deterministic_prefix_or_concatenation_only")
        is not True
        or summary.get("sift2_fit_separately_on_each_exact_view") is not True
        or summary.get(
            "independent_sift2_matrices_summed_for_balanced_view"
        )
        is not False
        or summary.get("cohort_recipe_selected") is not False
        or summary.get("scale_up_authorized") is not False
        or set(summary.get("matrix_names", [])) != set(RUNNER.MATRIX_NAMES)
        or set(summary.get("views", {})) != set(RUNNER.VIEW_NAMES)
    ):
        failures.append("summary_contract_differs")
    try:
        RUNNER.verify_record(
            summary["implementation"], label="runner implementation"
        )
        if summary["implementation"] != RUNNER.file_record(SOURCE):
            failures.append("runner_source_binding_differs")
        density_path = RUNNER.verify_record(
            summary["density_summary"], label="density summary"
        )
        density = RUNNER.valid_density_summary(density_path)
        selected = int(summary["selected_balanced_total_streamlines"])
        qualified = density[
            "density_qualified_balanced_total_counts"
        ]
        if selected not in qualified:
            failures.append("selected_count_is_not_density_qualified")
        if summary.get("selected_candidate_rank") != (
            qualified.index(selected) + 1
        ):
            failures.append("selected_candidate_rank_differs")
        atlas_result_path = RUNNER.verify_record(
            summary["hroi_atlas_result"], label="HROI atlas result"
        )
        atlas_result = load_json(atlas_result_path)
        node_volumes_path = RUNNER.verify_record(
            atlas_result["artifacts"]["hcp379_node_volumes"],
            label="HROI node volumes",
        )
        node_volumes = RUNNER.TECHNICAL.load_node_volumes(
            node_volumes_path
        )
    except Exception as exc:
        failures.append(f"bound_input_validation:{type(exc).__name__}:{exc}")
        selected = -1
        node_volumes = None

    matrices_by_view: dict[str, dict[str, Any]] = {}
    assignments: dict[str, float] = {}
    recomputed_views: dict[str, Any] = {}
    if node_volumes is not None:
        for view in RUNNER.VIEW_NAMES:
            row = summary.get("views", {}).get(view, {})
            try:
                expected = int(row["streamline_count"])
                track_path = RUNNER.verify_record(
                    row["tractogram"], label=f"{view}/tractogram"
                )
                if RUNNER.tck_count(track_path, expected) != expected:
                    raise ValueError("tractogram count differs")
                RUNNER.verify_record(
                    row["sift2"]["weights"],
                    label=f"{view}/SIFT2 weights",
                )
                if (
                    row["sift2"].get("weight_count") != expected
                    or RUNNER.numeric_count(
                        Path(row["sift2"]["weights"]["path"]), expected
                    )
                    != expected
                ):
                    raise ValueError("SIFT2 weight count differs")
                assignment_path = RUNNER.verify_record(
                    row["assignments"], label=f"{view}/assignments"
                )
                assignment = RUNNER.assignment_summary(
                    assignment_path, expected
                )
                assignments[view] = assignment[
                    "endpoint_assignment_fraction"
                ]
                matrices = {
                    name: RUNNER.matrix(
                        RUNNER.verify_record(
                            row["matrices"][name],
                            label=f"{view}/{name}",
                        )
                    )
                    for name in RUNNER.MATRIX_NAMES
                }
                matrices_by_view[view] = matrices
                technical, technical_failures = (
                    RUNNER.TECHNICAL.matrix_technical_qc(
                        matrices, node_volumes=node_volumes
                    )
                )
                recomputed_views[view] = {
                    "assignment": assignment,
                    "technical_qc": technical,
                    "technical_failures": technical_failures,
                }
                if technical_failures:
                    failures.extend(
                        f"{view}:{failure}"
                        for failure in technical_failures
                    )
                if abs(
                    assignment["endpoint_assignment_fraction"]
                    - float(row["endpoint_assignment_fraction"])
                ) > 1.0e-15:
                    failures.append(f"{view}:assignment_fraction_differs")
                if abs(
                    float(technical["edge_density"])
                    - float(row["technical_qc"]["edge_density"])
                ) > 1.0e-15:
                    failures.append(f"{view}:edge_density_differs")
            except Exception as exc:
                failures.append(
                    f"{view}:runtime_artifact:{type(exc).__name__}:{exc}"
                )

    recomputed_comparison: dict[str, Any] | None = None
    if (
        set(matrices_by_view) == set(RUNNER.VIEW_NAMES)
        and set(assignments) == set(RUNNER.VIEW_NAMES)
    ):
        gate = load_json(RUNNER.CONTRACT)[
            "diagnosis_blind_recipe_stability_gate"
        ]
        recomputed_comparison = RUNNER.TECHNICAL.evaluate_comparison(
            left_id="primary_half",
            right_id="independent_half",
            left=matrices_by_view["primary"],
            right=matrices_by_view["independent"],
            left_assignment=assignments["primary"],
            right_assignment=assignments["independent"],
            gate=gate,
        )
        if recomputed_comparison != summary.get("seed_reliability"):
            failures.append("seed_reliability_replay_differs")
        combined_density = recomputed_views["balanced"]["technical_qc"][
            "edge_density"
        ]
        if abs(
            float(combined_density)
            - float(summary.get("combined_edge_density", -1.0))
        ) > 1.0e-15:
            failures.append("combined_density_replay_differs")
        density_met = combined_density >= RUNNER.DENSITY_TARGET
        if density_met is not summary.get("combined_density_target_met"):
            failures.append("combined_density_gate_differs")

    expected_pass = (
        not failures
        and recomputed_comparison is not None
        and recomputed_comparison["status"] == "PASS"
        and recomputed_views.get("balanced", {})
        .get("technical_qc", {})
        .get("edge_density", -1.0)
        >= RUNNER.DENSITY_TARGET
    )
    claimed_pass = summary.get("status") == (
        "PASS_HROI_BALANCED_ALL_NINE_CANARY"
    )
    if expected_pass != claimed_pass:
        failures.append("terminal_status_differs_from_replay")
    return {
        "summary": RUNNER.file_record(path),
        "unit": summary.get("unit"),
        "selected_balanced_total_streamlines": selected,
        "recomputed_views": recomputed_views,
        "recomputed_seed_reliability": recomputed_comparison,
        "runtime_status": "PASS" if not failures else "FAIL",
        "runtime_failures": sorted(set(failures)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    checks: dict[str, Any] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    compile_probe = subprocess.run(
        [str(PYTHON), "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "source_compiles",
        compile_probe.returncode == 0,
        {
            "returncode": compile_probe.returncode,
            "stderr": compile_probe.stderr[-2000:],
        },
    )
    self_probe = subprocess.run(
        [str(PYTHON), str(SOURCE), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        self_value = json.loads(self_probe.stdout)
    except Exception:
        self_value = {}
    check(
        "self_test_passes",
        self_probe.returncode == 0 and self_value.get("status") == "PASS",
        self_value or {"stderr": self_probe.stderr[-2000:]},
    )
    dry_probe = subprocess.run(
        [str(PYTHON), str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        dry = json.loads(dry_probe.stdout)
    except Exception:
        dry = {}
    check(
        "dry_run_is_fail_closed_and_nonexecuting",
        (
            dry_probe.returncode == 0
            and dry.get("status")
            in {
                "READY_HROI_BALANCED_ALL_NINE_CANARY",
                "WAITING_FOR_COMPLETE_DENSITY_OR_EXACT_PAIR",
            }
            and dry.get("tractography_generated") is False
            and dry.get("all_nine_matrix_generation_started") is False
            and dry.get("cohort_recipe_selected") is False
            and dry.get("scale_up_authorized") is False
            and set(dry.get("matrix_names", [])) == set(RUNNER.MATRIX_NAMES)
        ),
        dry or {"stderr": dry_probe.stderr[-2000:]},
    )
    source_text = SOURCE.read_text(encoding="utf-8")
    required = (
        "tcksift2",
        "tckedit",
        "tck2connectome",
        "tcksample",
        "sift2_fit_separately_on_each_exact_view",
        "independent_sift2_matrices_summed_for_balanced_view",
        "count_invnodevol",
        "requires_complete_15_canary_all_nine_validation",
        "requires_hroi_human_release_gate",
    )
    missing = [fragment for fragment in required if fragment not in source_text]
    check(
        "implementation_contains_exact_all_nine_contract",
        not missing,
        {"missing": missing},
    )
    prohibited = (
        "tckgen",
        "rm -rf",
        "shutil.rmtree",
        ".unlink(",
        "diagnosis.csv",
        "DX_bl",
        '"cohort_recipe_selected": True',
        '"scale_up_authorized": True',
    )
    found = [fragment for fragment in prohibited if fragment in source_text]
    check(
        "implementation_has_no_new_tractography_destructive_or_outcome_shortcut",
        not found,
        {"found": found},
    )

    runtime_path = (
        args.summary.resolve()
        if args.summary is not None
        else latest_summary()
    )
    runtime = validate_runtime(runtime_path) if runtime_path else None
    if runtime is not None:
        check(
            "runtime_summary_replays",
            runtime["runtime_status"] == "PASS",
            runtime,
        )
    passed = sum(row["status"] == "PASS" for row in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_hroi_balanced_all_nine_canary_v1_validation"
        ),
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "runtime_summary_detected": runtime is not None,
        "runtime": runtime,
        "imaging_executed_by_validation": False,
        "source": RUNNER.file_record(SOURCE),
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
