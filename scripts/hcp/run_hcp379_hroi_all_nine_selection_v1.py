#!/usr/bin/env python3
"""Select the smallest jointly qualified balanced HROI construction.

The density cohort supplies an ordered set of totals that reach density 0.60
for all 15 canaries.  This orchestrator tests those totals in ascending order
with the complete final-HROI all-nine cohort runner.  It escalates only after a
complete, independently replayed scientific FAIL at the lower uniform total;
execution errors or incomplete evidence stop rather than trigger escalation.

No probabilistic tractography is generated here because the canary runner
reuses exact Phase-B 10M supersets.  A passing selection remains subject to the
final HROI human gate and does not itself authorize 530-subject execution.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
COHORT_SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_hroi_balanced_all_nine_canary_cohort_v1.py"
)
VALIDATOR_SOURCE = (
    EXP
    / "research_audit/"
    "validate_hcp379_hroi_balanced_all_nine_canary_cohort_v1.py"
)
ROOT = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_hroi_balanced_all_nine_selection_v1"
)
ATTEMPTS = ROOT / "attempts"
SELECTION = ROOT / "selection.json"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COHORT = load_module(COHORT_SOURCE, "hcp379_hroi_all_nine_selector_cohort")
SINGLE = COHORT.SINGLE


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


def density_contract() -> tuple[Path, dict[str, Any]] | None:
    """Return complete density evidence, including a terminal empty set.

    The downstream all-nine builder requires a positive selected count, so its
    helper intentionally rejects a density summary with no qualified totals.
    The selector, however, must distinguish that scientifically complete
    negative result from missing evidence.  Otherwise a watcher can wait
    forever after the calibration has already failed.
    """

    for path in sorted(
        SINGLE.DENSITY_COHORT_ATTEMPTS.glob("*/summary.json"),
        reverse=True,
    ):
        try:
            value = load_json(path)
        except Exception:
            continue
        candidates = value.get(
            "density_qualified_balanced_total_counts", []
        )
        selected = value.get(
            "smallest_density_qualified_balanced_total_count"
        )
        rows = value.get("units", [])
        if (
            value.get("record_type")
            != "diagnosis_blind_hcp379_hroi_balanced_density_canary_cohort_summary"
            or value.get("status")
            != "PASS_COMPLETE_COUNT_ONLY_DENSITY_EVIDENCE"
            or value.get("diagnosis_labels_used") is not False
            or value.get("processed_unit_n") != COHORT.EXPECTED_N
            or value.get("failure_n") != 0
            or value.get("count_only_projection") is not True
            or value.get("requires_joint_all_nine_matrix_stability")
            is not True
            or not isinstance(candidates, list)
            or candidates != sorted(set(candidates))
            or any(
                total not in SINGLE.BALANCED_TOTALS
                for total in candidates
            )
            or len(rows) != COHORT.EXPECTED_N
        ):
            continue
        if candidates:
            if selected != min(candidates) or any(
                not row.get("levels", {}).get(str(total), {}).get(
                    "density_target_0_60_met", False
                )
                for row in rows
                for total in candidates
            ):
                continue
        elif (
            selected is not None
            or any(
                all(
                    row.get("levels", {}).get(str(total), {}).get(
                        "density_target_0_60_met", False
                    )
                    for row in rows
                )
                for total in SINGLE.BALANCED_TOTALS
            )
        ):
            continue
        return path.resolve(), value
    return None


def cohort_summaries() -> dict[int, tuple[Path, dict[str, Any]]]:
    values: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(COHORT.ATTEMPTS.glob("*/summary.json")):
        try:
            value = load_json(path)
            selected = int(value["selected_balanced_total_streamlines"])
        except Exception:
            continue
        if (
            value.get("record_type")
            == "diagnosis_blind_hcp379_hroi_balanced_all_nine_canary_cohort_summary"
            and selected in SINGLE.BALANCED_TOTALS
        ):
            values[selected] = (path.resolve(), value)
    return values


def preflight() -> dict[str, Any]:
    density = density_contract()
    candidates = (
        density[1]["density_qualified_balanced_total_counts"]
        if density is not None
        else []
    )
    summaries = cohort_summaries()
    pair_probe = COHORT.readiness(
        COHORT.units(),
        density[0] if density is not None else None,
        candidates[0] if candidates else None,
    )
    pair_ready_n = sum(row["phase_pair_ready"] for row in pair_probe)
    if density is None:
        status = "WAITING_FOR_COMPLETE_DENSITY_EVIDENCE"
    elif not candidates:
        status = "NO_DENSITY_QUALIFIED_CANDIDATE_UP_TO_20M"
    elif pair_ready_n != COHORT.EXPECTED_N:
        status = "WAITING_FOR_ALL_15_EXACT_PAIRS"
    else:
        status = "READY_TO_TEST_ORDERED_JOINT_CANDIDATES"
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_all_nine_selection_preflight"
        ),
        "generated_utc": utc_now(),
        "status": status,
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "density_summary": (
            str(density[0]) if density is not None else None
        ),
        "ordered_density_qualified_candidates": candidates,
        "density_calibration_terminal": bool(
            density is not None and not candidates
        ),
        "density_calibration_conclusion": (
            "NO_UNIFORM_15_CANARY_CANDIDATE_AT_6M_10M_15M_OR_20M"
            if density is not None and not candidates
            else None
        ),
        "pair_ready_n": pair_ready_n,
        "expected_pair_n": COHORT.EXPECTED_N,
        "existing_candidate_summaries": {
            str(total): {
                "path": str(path),
                "status": value.get("status"),
            }
            for total, (path, value) in summaries.items()
            if total in candidates
        },
        "probabilistic_tractography_generated": False,
        "all_nine_execution_started": False,
        "selected_balanced_total_streamlines": None,
        "selected_count_tractography_authorized": False,
        "final_release_authorized": False,
        "requires_final_hroi_human_review": True,
    }


def run_candidate(
    total: int,
    *,
    workers: int,
    threads: int,
    attempt: Path,
) -> dict[str, Any]:
    candidate_root = attempt / str(total)
    candidate_root.mkdir(parents=True, exist_ok=False)
    cohort_run = subprocess.run(
        [
            str(PYTHON),
            str(COHORT_SOURCE),
            "--balanced-total",
            str(total),
            "--workers",
            str(workers),
            "--threads-per-unit",
            str(threads),
            "--execute",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    (candidate_root / "cohort.stdout.json").write_text(
        cohort_run.stdout, encoding="utf-8"
    )
    (candidate_root / "cohort.stderr.log").write_text(
        cohort_run.stderr, encoding="utf-8"
    )
    summaries = cohort_summaries()
    if total not in summaries:
        raise RuntimeError(
            f"candidate {total}: complete cohort summary is absent"
        )
    summary_path, summary = summaries[total]
    validation_run = subprocess.run(
        [str(PYTHON), str(VALIDATOR_SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    (candidate_root / "validation.stdout.json").write_text(
        validation_run.stdout, encoding="utf-8"
    )
    (candidate_root / "validation.stderr.log").write_text(
        validation_run.stderr, encoding="utf-8"
    )
    try:
        validation = json.loads(validation_run.stdout)
    except Exception as exc:
        raise ValueError("cohort validator stdout is not JSON") from exc
    independently_valid = bool(
        validation_run.returncode == 0
        and validation.get("status") == "PASS"
        and validation.get("runtime_summary_detected") is True
        and validation.get("runtime", {}).get("runtime_status") == "PASS"
        and validation.get("runtime", {}).get(
            "selected_balanced_total_streamlines"
        )
        == total
    )
    complete_pass = bool(
        cohort_run.returncode == 0
        and summary.get("status")
        == "PASS_COMPLETE_HROI_ALL_NINE_CANARY"
        and summary.get("processed_unit_n") == COHORT.EXPECTED_N
        and summary.get("uniform_candidate_recipe_confirmed") is True
        and independently_valid
    )
    complete_scientific_fail = bool(
        cohort_run.returncode == 1
        and summary.get("status")
        == "FAIL_COMPLETE_HROI_ALL_NINE_CANARY"
        and summary.get("processed_unit_n") == COHORT.EXPECTED_N
        and not summary.get("missing_terminal_units")
        and not summary.get("execution_failures")
        and independently_valid
    )
    if not complete_pass and not complete_scientific_fail:
        raise RuntimeError(
            f"candidate {total}: incomplete or non-scientific failure; "
            "escalation prohibited"
        )
    return {
        "balanced_total_streamlines": total,
        "status": (
            "PASS_JOINT_ALL_NINE"
            if complete_pass
            else "FAIL_JOINT_ALL_NINE"
        ),
        "cohort_returncode": cohort_run.returncode,
        "summary": SINGLE.file_record(summary_path),
        # Bind the immutable candidate-local validator stdout.  The validator
        # also updates a convenient global latest-result file, but that path is
        # overwritten when a higher candidate is evaluated and therefore
        # cannot serve as durable evidence for an earlier candidate.
        "validation": SINGLE.file_record(
            candidate_root / "validation.stdout.json"
        ),
    }


def execute(workers: int, threads: int) -> dict[str, Any]:
    ready = preflight()
    if ready["status"] != "READY_TO_TEST_ORDERED_JOINT_CANDIDATES":
        raise RuntimeError(ready["status"])
    density_path, density = density_contract()  # type: ignore[misc]
    candidates = density["density_qualified_balanced_total_counts"]
    attempt = ATTEMPTS / f"{stamp()}-hroi-all-nine-selection"
    attempt.mkdir(parents=True, exist_ok=False)
    bound = {
        **ready,
        "status": "EXECUTION_INPUTS_BOUND",
        "all_nine_execution_started": True,
        "density_summary_record": SINGLE.file_record(density_path),
        "cohort_runner": SINGLE.file_record(COHORT_SOURCE),
        "cohort_validator": SINGLE.file_record(VALIDATOR_SOURCE),
        "implementation": SINGLE.file_record(Path(__file__)),
    }
    atomic_json(attempt / "preflight.json", bound)
    evaluated = []
    selected = None
    for total in candidates:
        result = run_candidate(
            total,
            workers=workers,
            threads=threads,
            attempt=attempt,
        )
        evaluated.append(result)
        if result["status"] == "PASS_JOINT_ALL_NINE":
            selected = total
            break
    status = (
        "PASS_SMALLEST_JOINTLY_QUALIFIED_BALANCED_RECIPE"
        if selected is not None
        else "FAIL_NO_JOINTLY_QUALIFIED_BALANCED_RECIPE"
    )
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_all_nine_recipe_selection"
        ),
        "generated_utc": utc_now(),
        "status": status,
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "ordered_density_qualified_candidates": candidates,
        "evaluated_candidates": evaluated,
        "selected_balanced_total_streamlines": selected,
        "selected_per_seed_streamlines": (
            SINGLE.BALANCED_TOTALS[selected]
            if selected is not None
            else None
        ),
        "selection_is_smallest_joint_pass": selected is not None,
        "per_subject_streamline_tuning_allowed": False,
        "subject_exclusion_for_low_density_allowed": False,
        "selected_count_tractography_authorized": False,
        "final_release_authorized": False,
        "requires_final_hroi_human_review": True,
        "preflight": SINGLE.file_record(attempt / "preflight.json"),
        "density_summary": SINGLE.file_record(density_path),
        "implementation": SINGLE.file_record(Path(__file__)),
    }
    atomic_json(attempt / "selection.json", result)
    atomic_json(SELECTION, result)
    return {
        "attempt_root": str(attempt),
        "status": status,
        "selected_balanced_total_streamlines": selected,
        "evaluated_candidate_n": len(evaluated),
        "selected_count_tractography_authorized": False,
    }


def self_test() -> dict[str, Any]:
    checks = {
        "candidate_order": list(SINGLE.BALANCED_TOTALS)
        == sorted(SINGLE.BALANCED_TOTALS),
        "complete_canary_n": COHORT.EXPECTED_N == 15,
        "density_target": SINGLE.DENSITY_TARGET == 0.60,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-unit", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error("--workers must be in 1..2")
    if not 1 <= args.threads_per_unit <= 16:
        parser.error("--threads-per-unit must be in 1..16")
    if args.workers * args.threads_per_unit > 16:
        parser.error("worker/thread product must not exceed 16")
    if args.self_test:
        value = self_test()
        print(json.dumps(value, sort_keys=True))
        return 0 if value["status"] == "PASS" else 1
    value = (
        execute(args.workers, args.threads_per_unit)
        if args.execute
        else preflight()
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if not args.execute or value["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
