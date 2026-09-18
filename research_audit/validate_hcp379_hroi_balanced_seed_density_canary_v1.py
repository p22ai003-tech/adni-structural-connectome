#!/usr/bin/env python3
"""Validate the corrected-HROI balanced-density canary implementation."""

from __future__ import annotations

import hashlib
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
    "build_hcp379_hroi_balanced_seed_density_canary_v1.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_hroi_balanced_seed_density_canary_v1/validation.json"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_hroi_balanced_density_canary_v1/attempts"
)
TCK_COUNTER = (
    EXP / "scripts/hcp/tck_header_count_v1.py"
)
EXPECTED_NODES = 379
EXPECTED_STREAMLINES = 10_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    checks: dict[str, Any] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    compiled = subprocess.run(
        [str(PYTHON), "-m", "py_compile", str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "source_compiles",
        compiled.returncode == 0,
        {"returncode": compiled.returncode, "stderr": compiled.stderr[-2000:]},
    )

    self_test = subprocess.run(
        [str(PYTHON), str(SOURCE), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        self_value = json.loads(self_test.stdout)
    except Exception:
        self_value = {}
    check(
        "synthetic_support_complementarity_passes",
        self_test.returncode == 0 and self_value.get("status") == "PASS",
        self_value or {"stderr": self_test.stderr[-2000:]},
    )

    dryrun = subprocess.run(
        [str(PYTHON), str(SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        dry_value = json.loads(dryrun.stdout)
    except Exception:
        dry_value = {}
    check(
        "dry_run_is_fail_closed_and_nonexecuting",
        (
            dryrun.returncode == 0
            and dry_value.get("status")
            in {
                "READY_HROI_BALANCED_DENSITY_CANARY",
                "WAITING_FOR_HROI_OR_EXACT_10M_PAIR",
            }
            and dry_value.get("imaging_executed") is False
            and dry_value.get("tractography_generated") is False
            and dry_value.get("matrix_generation_started") is False
            and dry_value.get("scale_up_authorized") is False
        ),
        dry_value or {"stderr": dryrun.stderr[-2000:]},
    )

    text = SOURCE.read_text(encoding="utf-8")
    required = (
        "PASS_HROI_CANARY_ATLAS",
        "tck2connectome",
        "-assignment_radial_search",
        "hroi_assignments_10m.csv",
        "6_000_000",
        "10_000_000",
        "15_000_000",
        "20_000_000",
        "support_complementarity",
        "assignment_replay_exact",
        "output.stem}.partial{output.suffix",
        "requires_all_nine_matrix_stability",
        "cohort_uniform_recipe_selected",
        "scale_up_authorized",
    )
    missing = [fragment for fragment in required if fragment not in text]
    check(
        "implementation_contains_corrected_balanced_contract",
        not missing,
        {"missing": missing},
    )

    prohibited = (
        "rm -rf",
        "shutil.rmtree",
        ".unlink(",
        "diagnosis.csv",
        "DX_bl",
        "scale_up_authorized\": True",
        "cohort_uniform_recipe_selected\": True",
    )
    found = [fragment for fragment in prohibited if fragment in text]
    check(
        "implementation_has_no_destructive_or_outcome_aware_shortcut",
        not found,
        {"found": found},
    )

    check(
        "dry_run_targets_exact_same_unit_pair",
        (
            dry_value.get("unit") == "032_S_6804_I1230908"
            and dry_value.get("candidate_ready") is True
            and dry_value.get("pair", {})
            .get("paths", {})
            .get("primary_tracks", "")
            .endswith(
                "032_S_6804_I1230908/"
                "09_hcp379_stability/primary_10m/tracks.tck"
            )
            and dry_value.get("pair", {})
            .get("paths", {})
            .get("independent_tracks", "")
            .endswith(
                "032_S_6804_I1230908/"
                "09_hcp379_stability/independent_10m/tracks.tck"
            )
        ),
        {
            "unit": dry_value.get("unit"),
            "candidate_ready": dry_value.get("candidate_ready"),
            "pair": dry_value.get("pair"),
        },
    )

    summaries = sorted(ATTEMPTS.glob("*/summary.json"))
    if summaries:
        summary_path = summaries[-1].resolve()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runtime_failures: dict[str, Any] = {}
        recomputed_levels: dict[str, Any] = {}
        expected_totals = {
            "6000000": 3_000_000,
            "10000000": 5_000_000,
            "15000000": 7_500_000,
            "20000000": 10_000_000,
        }
        for total, prefix in expected_totals.items():
            try:
                level = summary["levels"][total]
                matrix_record = level["count_matrix"]
                matrix_path = Path(str(matrix_record["path"])).resolve()
                if record(matrix_path) != matrix_record:
                    raise ValueError("balanced matrix binding differs")
                matrix = np.loadtxt(matrix_path, delimiter=",")
                if (
                    matrix.shape != (EXPECTED_NODES, EXPECTED_NODES)
                    or not np.isfinite(matrix).all()
                    or np.any(matrix < 0)
                    or not np.allclose(
                        matrix, matrix.T, atol=0.0, rtol=0.0
                    )
                    or not np.allclose(
                        np.diag(matrix), 0.0, atol=0.0, rtol=0.0
                    )
                    or not np.allclose(
                        matrix, np.rint(matrix), atol=0.0, rtol=0.0
                    )
                ):
                    raise ValueError("balanced matrix invariants differ")
                triangle = matrix[np.triu_indices(EXPECTED_NODES, 1)]
                supported = int(np.count_nonzero(triangle > 0))
                possible = EXPECTED_NODES * (EXPECTED_NODES - 1) // 2
                density = supported / possible
                complementarity = level["support_complementarity"]
                if (
                    level.get("total_streamlines") != int(total)
                    or level.get("per_seed_prefix_streamlines") != prefix
                    or level.get("supported_edges") != supported
                    or level.get("possible_edges") != possible
                    or abs(float(level.get("edge_density")) - density)
                    > 1.0e-15
                    or level.get("connected_nodes")
                    != int(np.count_nonzero(np.sum(matrix, axis=1) > 0))
                    or complementarity.get("union_supported_edges")
                    != supported
                    or level.get("density_target_0_60_met")
                    is not (density >= 0.60)
                ):
                    raise ValueError("balanced level replay differs")
                recomputed_levels[total] = {
                    "edge_density": density,
                    "supported_edges": supported,
                    "per_seed_prefix_streamlines": prefix,
                }
            except Exception as exc:
                runtime_failures[f"level_{total}"] = (
                    f"{type(exc).__name__}:{exc}"
                )

        seed_replay: dict[str, Any] = {}
        for seed in ("primary", "independent"):
            try:
                row = summary["seeds"][seed]
                for name in (
                    "tracks",
                    "metadata",
                    "hroi_count_10m",
                    "hroi_assignments_10m",
                ):
                    value = row[name]
                    if record(Path(str(value["path"])).resolve()) != value:
                        raise ValueError(f"{name} binding differs")
                count_probe = subprocess.run(
                    [
                        str(PYTHON),
                        str(TCK_COUNTER),
                        str(row["tracks"]["path"]),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                assignments = np.loadtxt(
                    row["hroi_assignments_10m"]["path"],
                    comments="#",
                    dtype=np.int16,
                )
                if (
                    count_probe.returncode != 0
                    or int(count_probe.stdout.strip())
                    != EXPECTED_STREAMLINES
                    or assignments.shape != (EXPECTED_STREAMLINES, 2)
                    or np.any(assignments < 0)
                    or np.any(assignments > EXPECTED_NODES)
                    or row.get("assignment_replay_exact") is not True
                ):
                    raise ValueError("seed count/assignment contract differs")
                seed_replay[seed] = {
                    "tractogram_streamlines": int(
                        count_probe.stdout.strip()
                    ),
                    "assignment_shape": list(assignments.shape),
                    "endpoint_assignment_fraction": float(
                        np.mean(
                            (assignments[:, 0] > 0)
                            & (assignments[:, 1] > 0)
                        )
                    ),
                }
                if (
                    abs(
                        seed_replay[seed][
                            "endpoint_assignment_fraction"
                        ]
                        - float(row["endpoint_assignment_fraction"])
                    )
                    > 1.0e-15
                ):
                    raise ValueError(
                        "seed endpoint-assignment replay differs"
                    )
            except Exception as exc:
                runtime_failures[f"seed_{seed}"] = (
                    f"{type(exc).__name__}:{exc}"
                )

        atlas_ok = False
        try:
            atlas_record = summary["hroi_atlas_result"]
            atlas_path = Path(str(atlas_record["path"])).resolve()
            atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
            atlas_ok = (
                record(atlas_path) == atlas_record
                and atlas.get("status") == "PASS_HROI_CANARY_ATLAS"
                and atlas.get("unit") == "032_S_6804_I1230908"
                and atlas.get("diagnosis_labels_used") is False
                and atlas.get("tractography_started") is False
                and atlas.get("matrix_generation_started") is False
                and atlas.get("atlas_qc", {}).get("status") == "PASS"
                and atlas.get("atlas_qc", {}).get("labels_found")
                == EXPECTED_NODES
            )
        except Exception as exc:
            runtime_failures["atlas"] = f"{type(exc).__name__}:{exc}"

        qualifying = [
            int(total)
            for total, row in recomputed_levels.items()
            if row["edge_density"] >= 0.60
        ]
        check(
            "latest_executed_canary_replays_exactly",
            (
                summary.get("record_type")
                == (
                    "diagnosis_blind_hcp379_hroi_balanced_"
                    "density_canary_summary"
                )
                and summary.get("status")
                == "PASS_HROI_BALANCED_DENSITY_CANARY"
                and summary.get("diagnosis_labels_used") is False
                and summary.get("non_overwriting") is True
                and summary.get("source_files_modified") is False
                and summary.get("tractography_reused_not_regenerated")
                is True
                and summary.get("sift2_or_scalar_matrices_generated")
                is False
                and summary.get("scale_up_authorized") is False
                and summary.get("cohort_uniform_recipe_selected") is False
                and summary.get("requires_complete_phase_b_canary_family")
                is True
                and summary.get("requires_all_nine_matrix_stability")
                is True
                and summary.get("requires_uniform_530_subject_construction")
                is True
                and set(recomputed_levels) == set(expected_totals)
                and set(seed_replay) == {"primary", "independent"}
                and atlas_ok
                and not runtime_failures
                and summary.get(
                    "density_qualified_balanced_total_counts_for_this_canary"
                )
                == sorted(qualifying)
                and summary.get(
                    "smallest_density_qualified_count_for_this_canary"
                )
                == (min(qualifying) if qualifying else None)
            ),
            {
                "summary": str(summary_path),
                "recomputed_levels": recomputed_levels,
                "seed_replay": seed_replay,
                "atlas_ok": atlas_ok,
                "runtime_failures": runtime_failures,
                "qualifying": sorted(qualifying),
            },
        )

    passed = sum(row["status"] == "PASS" for row in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_hroi_balanced_seed_density_canary_v1_validation"
        ),
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "total_checks": len(checks),
        "passed_checks": passed,
        "imaging_executed_by_validation": False,
        "source": record(SOURCE),
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
