#!/usr/bin/env python3
"""Validate the bounded 135 independent-10M density superset."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXP = Path("/home/ec2-user/exp")
SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_hroi_low_density_135_20m_superset_v1.py"
)
HELPER_SOURCE = (
    EXP
    / "research_audit/"
    "validate_hcp379_hroi_low_density_135_6m_canary_v1.py"
)
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_hroi_low_density_135_20m_superset_v1/attempts"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_hroi_low_density_135_20m_superset_v1/validation.json"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")
EXPECTED_LEVELS = {
    "10000000": 5_000_000,
    "15000000": 7_500_000,
    "20000000": 10_000_000,
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPER = load_module(HELPER_SOURCE, "hcp379_135_superset_validator_base")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def record_matches(record: Any, path: Path) -> bool:
    return isinstance(record, dict) and record == file_record(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
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
        "dry_run_binds_only_remaining_density_levels",
        (
            dry_probe.returncode == 0
            and dry.get("status")
            == "READY_LOW_DENSITY_135_10M_SUPERSET"
            and dry.get("unit") == "135_S_6840_I1263427"
            and dry.get("diagnosis_labels_used") is False
            and dry.get("lower_total_counts_ruled_out")
            == [3_000_000, 5_000_000, 6_000_000]
            and dry.get("balanced_total_counts_to_project")
            == [10_000_000, 15_000_000, 20_000_000]
            and dry.get("independent_streamlines_to_generate")
            == 10_000_000
            and dry.get("imaging_executed") is False
            and dry.get("official_phase_b_modified") is False
            and dry.get("cohort_recipe_selected") is False
            and dry.get("scale_up_authorized") is False
            and dry.get("selected_count_tractography_authorized")
            is False
        ),
        dry or {"stderr": dry_probe.stderr[-2000:]},
    )
    text = SOURCE.read_text(encoding="utf-8")
    required = (
        "INDEPENDENT_STREAMLINES = 10_000_000",
        "10_000_000: 5_000_000",
        "15_000_000: 7_500_000",
        "20_000_000: 10_000_000",
        "tracks.partial.tck",
        "count_hroi.partial.csv",
        "assignments_hroi.partial.csv",
        "official_phase_b_modified",
        "requires_complete_15_canary_family",
        "requires_joint_all_nine_matrix_stability",
    )
    missing = [fragment for fragment in required if fragment not in text]
    check(
        "implementation_contains_exact_superset_contract",
        not missing,
        {"missing": missing},
    )
    prohibited = (
        "rm -rf",
        "shutil.rmtree",
        ".unlink(",
        "diagnosis.csv",
        "DX_bl",
        "cohort_recipe_selected\": True",
        "scale_up_authorized\": True",
        "selected_count_tractography_authorized\": True",
    )
    found = [fragment for fragment in prohibited if fragment in text]
    check(
        "implementation_has_no_destructive_or_outcome_shortcut",
        not found,
        {"found": found},
    )

    summaries = sorted(ATTEMPTS.glob("*/summary.json"))
    projection: dict[str, Any] | None = None
    if summaries:
        summary_path = summaries[-1]
        try:
            summary = load_json(summary_path)
            preflight_path = summary_path.parent / "preflight.json"
            preflight = load_json(preflight_path)
            contract = (
                summary.get("record_type")
                == (
                    "diagnosis_blind_hcp379_hroi_low_density_135_"
                    "20m_superset_summary"
                )
                and summary.get("status")
                == "PASS_LOW_DENSITY_135_20M_SUPERSET_CALIBRATION"
                and summary.get("unit") == "135_S_6840_I1263427"
                and summary.get("diagnosis_labels_used") is False
                and summary.get("non_overwriting") is True
                and summary.get("source_files_modified") is False
                and summary.get("official_phase_b_modified") is False
                and summary.get("primary_tractography_reused") is True
                and summary.get("independent_tractography_generated")
                is True
                and summary.get("generated_independent_streamlines")
                == 10_000_000
                and summary.get("cohort_recipe_selected") is False
                and summary.get("scale_up_authorized") is False
                and summary.get(
                    "selected_count_tractography_authorized"
                )
                is False
                and summary.get("requires_complete_15_canary_family")
                is True
                and summary.get(
                    "requires_joint_all_nine_matrix_stability"
                )
                is True
                and set(summary.get("levels", {}))
                == set(EXPECTED_LEVELS)
            )
            check(
                "runtime_summary_contract",
                contract,
                {
                    "summary": str(summary_path),
                    "status": summary.get("status"),
                },
            )

            primary_path = Path(
                str(
                    preflight.get("inputs", {})
                    .get("primary_hroi_assignments", {})
                    .get("path", "")
                )
            )
            artifacts = summary.get("artifacts", {})
            independent_path = Path(
                str(
                    artifacts.get(
                        "independent_hroi_assignments",
                        {},
                    ).get("path", "")
                )
            )
            independent_count_path = Path(
                str(
                    artifacts.get(
                        "independent_hroi_count",
                        {},
                    ).get("path", "")
                )
            )
            metadata_path = Path(
                str(
                    artifacts.get(
                        "independent_metadata",
                        {},
                    ).get("path", "")
                )
            )
            tracks_path = Path(
                str(
                    artifacts.get(
                        "independent_tracks",
                        {},
                    ).get("path", "")
                )
            )
            paths = (
                primary_path,
                independent_path,
                independent_count_path,
                metadata_path,
                tracks_path,
                preflight_path,
            )
            records_ok = (
                all(
                    path.is_file() and not path.is_symlink()
                    for path in paths
                )
                and record_matches(
                    artifacts["independent_hroi_assignments"],
                    independent_path,
                )
                and record_matches(
                    artifacts["independent_hroi_count"],
                    independent_count_path,
                )
                and record_matches(
                    artifacts["independent_metadata"],
                    metadata_path,
                )
                and record_matches(
                    artifacts["preflight"],
                    preflight_path,
                )
                and record_matches(
                    summary["implementation"],
                    SOURCE,
                )
            )
            check(
                "runtime_small_artifact_hashes_replay",
                records_ok,
                {"hash_records_match": records_ok},
            )

            primary = HELPER.load_assignments(
                primary_path,
                10_000_000,
            )
            independent = HELPER.load_assignments(
                independent_path,
                10_000_000,
            )
            recorded_independent = HELPER.load_count(
                independent_count_path
            )
            replayed_independent = HELPER.count_matrix(
                independent,
                10_000_000,
            )
            independent_exact = np.array_equal(
                recorded_independent,
                replayed_independent,
            )
            check(
                "runtime_independent_assignment_replay",
                independent_exact,
                {
                    "exact": independent_exact,
                    "maximum_absolute_difference": int(
                        np.max(
                            np.abs(
                                recorded_independent
                                - replayed_independent
                            )
                        )
                    ),
                },
            )

            level_evidence: dict[str, Any] = {}
            level_exact = True
            for total, prefix in EXPECTED_LEVELS.items():
                matrix = (
                    HELPER.count_matrix(primary, prefix)
                    + HELPER.count_matrix(independent, prefix)
                )
                recorded_path = Path(
                    str(
                        summary["levels"][total]["count_matrix"][
                            "path"
                        ]
                    )
                )
                recorded = HELPER.load_count(recorded_path)
                qc = HELPER.matrix_qc(matrix)
                matrix_exact = np.array_equal(matrix, recorded)
                record_exact = record_matches(
                    summary["levels"][total]["count_matrix"],
                    recorded_path,
                )
                density_exact = abs(
                    float(summary["levels"][total]["edge_density"])
                    - float(qc["edge_density"])
                ) < 1.0e-15
                target_exact = (
                    summary["levels"][total][
                        "density_target_0_60_met"
                    ]
                    is (float(qc["edge_density"]) >= 0.60)
                )
                this_exact = (
                    matrix_exact
                    and record_exact
                    and density_exact
                    and target_exact
                    and qc["connected_nodes"] == 379
                )
                level_exact = level_exact and this_exact
                level_evidence[total] = {
                    **qc,
                    "recorded_matrix_exact": matrix_exact,
                    "record_hash_exact": record_exact,
                    "recorded_density_exact": density_exact,
                    "recorded_target_exact": target_exact,
                }
            projection = level_evidence
            check(
                "runtime_balanced_10m_15m_20m_projection_replays",
                level_exact,
                {
                    "levels": level_evidence,
                    "cohort_recipe_selected": False,
                },
            )

            metadata = load_json(metadata_path)
            track_count = HELPER.tck_count(tracks_path)
            track_record = artifacts["independent_tracks"]
            track_contract = (
                track_count == 10_000_000
                and track_record.get("path")
                == str(tracks_path.resolve())
                and track_record.get("size_bytes")
                == tracks_path.stat().st_size
                and metadata.get("record_type")
                == "hcp379_stability_tractography_parameters"
                and metadata.get("diagnosis_labels_used") is False
                and metadata.get("unit") == "135_S_6840_I1263427"
                and metadata.get("seed_class") == "independent"
                and metadata.get("actual_streamlines") == 10_000_000
            )
            check(
                "runtime_tractogram_and_metadata_contract",
                track_contract,
                {
                    "track_header_count": track_count,
                    "track_record_size_matches": (
                        track_record.get("size_bytes")
                        == tracks_path.stat().st_size
                    ),
                },
            )
        except Exception as exc:
            check(
                "runtime_execution_replay",
                False,
                {
                    "summary": str(summary_path),
                    "exception": repr(exc),
                },
            )

    passed = sum(row["status"] == "PASS" for row in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_hroi_low_density_135_20m_superset_v1_validation"
        ),
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed_checks": passed,
        "total_checks": len(checks),
        "runtime_summary_detected": bool(summaries),
        "runtime_balanced_density_projection": projection,
        "imaging_executed_by_validation": False,
        "source": file_record(SOURCE),
        "checks": checks,
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
