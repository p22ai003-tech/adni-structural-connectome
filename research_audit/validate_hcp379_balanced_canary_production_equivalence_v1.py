#!/home/ec2-user/fsl/bin/python
"""Validate calibration-to-production equivalence for balanced HCP379.

The check is diagnosis blind.  It proves that the 15 calibration canaries and
the 515-subject production runner share the same tractography identity,
parameters, HROI construction semantics, balanced concatenation, SIFT2 fit,
matrix family and QC thresholds.  Missing canary tractograms are reported as
in-progress rather than accepted as evidence; production execution requires
the complete 30/30 primary/independent metadata set.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
PHASE_ROOT = Path(
    "/data/derivatives/scforge_v2/"
    "h04a_r1_recovery_20260719_retry4"
)
RUNNER_SOURCE = (
    EXP / "scripts/hcp/run_hcp379_balanced_production_v1.py"
)
ALL_NINE_SOURCE = (
    EXP
    / "scripts/hcp/"
    "build_hcp379_hroi_balanced_all_nine_canary_v1.py"
)
PHASE_RULE_SOURCE = (
    EXP
    / "scforge/workflow/extensions/"
    "h04a_r1_retry4_phase_b_hcp379/07_hcp379_stability.smk"
)
RECIPE_CONFIG = EXP / "configs/connectome_v2_retry3.yaml"
PRETRACT_MASTER = (
    HCP_ROOT
    / "pretract_promoted_recovery4_v6/"
    "pretract_promoted_recovery4_v6_manifest.json"
)
TOPOLOGY = (
    EXP
    / "research_audit/outputs/"
    "hcp379_balanced_release_topology_v1/topology.json"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_balanced_canary_production_equivalence_v1/validation.json"
)
EXPECTED_CANARIES = 15
EXPECTED_METADATA = EXPECTED_CANARIES * 2
EXPECTED_STREAMLINES = 10_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def verify_record(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if dict(record) != file_record(path):
        raise ValueError(f"{label}: file record differs")
    return path


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_module(
    RUNNER_SOURCE, "hcp379_equivalence_production_runner"
)
ALL_NINE = load_module(
    ALL_NINE_SOURCE, "hcp379_equivalence_all_nine"
)


def expected_seed(
    input_contract: Mapping[str, Any],
    seed_class: str,
) -> tuple[str, int]:
    source = input_contract["source_identity"]
    hashes = input_contract["source_identity_hashes"]
    row = {
        "dti_source_id": source["dti_source_id"],
        "dti_raw_bundle_sha256": hashes["dti_raw_bundle_sha256"],
    }
    return RUNNER.seed_identity(row, seed_class)


def metadata_contract(
    *,
    unit: str,
    seed_class: str,
    input_contract: Mapping[str, Any],
) -> dict[str, Any]:
    path = (
        PHASE_ROOT
        / "subjects"
        / unit
        / "09_hcp379_stability"
        / f"{seed_class}_10m"
        / "tractography_parameters.json"
    )
    tracks = path.with_name("tracks.tck")
    if not path.is_file() or not tracks.is_file():
        return {
            "status": "WAITING",
            "unit": unit,
            "seed_class": seed_class,
            "metadata_path": str(path.resolve()),
            "tracks_path": str(tracks.resolve()),
        }
    metadata = load_json(path)
    seed_id, rng_seed = expected_seed(input_contract, seed_class)
    run_id = f"{seed_class}_10m"
    header_count = ALL_NINE.tck_count(
        tracks, EXPECTED_STREAMLINES
    )
    valid = bool(
        metadata.get("record_type")
        == "hcp379_stability_tractography_parameters"
        and metadata.get("diagnosis_labels_used") is False
        and metadata.get("unit") == unit
        and metadata.get("run_id") == run_id
        and metadata.get("seed_class") == seed_class
        and metadata.get("seed_id") == seed_id
        and metadata.get("rng_seed") == rng_seed
        and metadata.get("algorithm") == "iFOD2"
        and metadata.get("act") is True
        and metadata.get("backtrack") is True
        and metadata.get("crop_at_gmwmi") is True
        and metadata.get("seeding") == "seed_dynamic"
        and metadata.get("requested_streamlines")
        == EXPECTED_STREAMLINES
        and metadata.get("actual_streamlines") == EXPECTED_STREAMLINES
        and isinstance(metadata.get("actual_step_size_mm"), (int, float))
        and float(metadata["actual_step_size_mm"]) > 0
        and header_count == EXPECTED_STREAMLINES
    )
    return {
        "status": "PASS" if valid else "FAIL",
        "unit": unit,
        "seed_class": seed_class,
        "metadata": file_record(path),
        "tracks": {
            "path": str(tracks.resolve()),
            "size_bytes": tracks.stat().st_size,
            "header_streamline_count": header_count,
        },
        "expected_seed_id": seed_id,
        "expected_rng_seed": rng_seed,
    }


def main() -> int:
    checks: dict[str, Any] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    compile_probe = subprocess.run(
        [
            str(RUNNER.ALL_NINE.PYTHON),
            "-m",
            "py_compile",
            str(RUNNER_SOURCE),
            str(ALL_NINE_SOURCE),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "implementations_compile",
        compile_probe.returncode == 0,
        {
            "returncode": compile_probe.returncode,
            "stderr": compile_probe.stderr[-2000:],
        },
    )
    recipe = RUNNER.recipe_contract()
    tract = recipe["tractography"]
    check(
        "locked_tractography_parameters_match_phase_b",
        (
            RUNNER.RECIPE_ID
            == "connectome-v2.1.0-closed-world-canary-candidate"
            and tract["algorithm"] == "iFOD2"
            and tract["act_enabled"] is True
            and tract["backtrack"] is True
            and tract["crop_at_gmwmi"] is True
            and tract["seeding"]["method"] == "seed_dynamic"
            and tract["maximum_seed_attempts"] == 200_000_000
            and float(tract["minimum_length_mm"]) == 10.0
            and float(tract["maximum_length_mm"]) == 250.0
            and float(tract["cutoff"]) == 0.06
            and float(tract["maximum_angle_degrees"]) == 45.0
            and tract["nthreads_per_subject"] == 16
        ),
        {
            "recipe_id": RUNNER.RECIPE_ID,
            "tractography": tract,
        },
    )
    phase_text = PHASE_RULE_SOURCE.read_text(encoding="utf-8")
    phase_fragments = (
        "MRTRIX_RNG_SEED",
        "-algorithm iFOD2",
        "-act",
        "-backtrack",
        "-crop_at_gmwmi",
        "-seed_dynamic",
        "-select {params.requested}",
        "-seeds {params.max_seeds}",
        "-minlength {params.min_length}",
        "-maxlength {params.max_length}",
        "-cutoff {params.cutoff}",
        '-step "$step"',
        "-angle {params.angle}",
        "-nthreads {threads}",
        "primary_10m|independent_10m",
    )
    check(
        "phase_b_uses_the_locked_tckgen_contract",
        all(fragment in phase_text for fragment in phase_fragments),
        {
            "missing": [
                fragment
                for fragment in phase_fragments
                if fragment not in phase_text
            ]
        },
    )
    check(
        "matrix_and_qc_implementation_is_shared",
        (
            RUNNER.ALL_NINE_SOURCE.resolve()
            == ALL_NINE_SOURCE.resolve()
            and RUNNER.ALL_NINE.concatenate is not None
            and RUNNER.ALL_NINE.build_view is not None
            and RUNNER.ALL_NINE.MATRIX_NAMES == ALL_NINE.MATRIX_NAMES
            and RUNNER.ALL_NINE.BALANCED_TOTALS
            == ALL_NINE.BALANCED_TOTALS
            and RUNNER.DENSITY_TARGET == ALL_NINE.DENSITY_TARGET == 0.60
            and RUNNER.EXPECTED_NODES
            == ALL_NINE.EXPECTED_NODES
            == 379
        ),
        {
            "matrix_names": list(RUNNER.ALL_NINE.MATRIX_NAMES),
            "balanced_totals": RUNNER.ALL_NINE.BALANCED_TOTALS,
            "all_nine_source": file_record(ALL_NINE_SOURCE),
        },
    )

    topology = load_json(TOPOLOGY)
    canaries = topology.get("canaries", [])
    canary_units = {
        str(row.get("unit", ""))
        for row in canaries
        if isinstance(row, Mapping)
    }
    check(
        "exact_disjoint_15_plus_515_topology",
        (
            topology.get("total_unit_n") == 530
            and topology.get("canary_reuse_n") == EXPECTED_CANARIES
            and topology.get("production_execution_n") == 515
            and len(canary_units) == EXPECTED_CANARIES
            and len(topology.get("production", [])) == 515
            and not canary_units.intersection(
                str(row.get("unit", ""))
                for row in topology.get("production", [])
            )
        ),
        {
            "canary_n": len(canary_units),
            "production_n": len(topology.get("production", [])),
        },
    )

    master = load_json(PRETRACT_MASTER)
    manifest_by_unit = {
        str(row["unit"]): verify_record(
            row["manifest"], label=f"{row['unit']}/pretract manifest"
        )
        for row in master.get("units", [])
    }
    input_binding_errors: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    for unit in sorted(canary_units):
        try:
            manifest_path = manifest_by_unit[unit]
            manifest = load_json(manifest_path)
            phase = ALL_NINE.phase_inputs(unit)
            tract_inputs = manifest["tractography_inputs"]
            scalar_inputs = manifest["tensor_maps_bounded"]
            for phase_name, manifest_name in (
                ("wmfod", "wmfod_normalised"),
                ("five_tt", "five_tt"),
            ):
                expected_path = Path(
                    str(tract_inputs[manifest_name]["path"])
                ).resolve()
                if phase[phase_name].resolve() != expected_path:
                    raise ValueError(
                        f"{phase_name} path differs from promoted Recovery4"
                    )
            for metric in ("fa", "md", "rd", "ad"):
                expected_path = Path(
                    str(scalar_inputs[metric]["path"])
                ).resolve()
                if phase[metric].resolve() != expected_path:
                    raise ValueError(
                        f"{metric} path differs from promoted Recovery4"
                    )
            input_contract_path = (
                PHASE_ROOT
                / "subjects"
                / unit
                / "00_inputs/input_contract.json"
            )
            input_contract = load_json(input_contract_path)
            if (
                input_contract.get("unit") != unit
                or input_contract.get("diagnosis_used_for_processing_route")
                is not False
                or input_contract.get("pair_gap_used_for_processing_rejection")
                is not False
                or input_contract.get("recipe_id") != RUNNER.RECIPE_ID
            ):
                raise ValueError("diagnosis-blind input contract differs")
            for seed_class in ("primary", "independent"):
                metadata_rows.append(
                    metadata_contract(
                        unit=unit,
                        seed_class=seed_class,
                        input_contract=input_contract,
                    )
                )
        except Exception as exc:
            input_binding_errors.append(
                f"{unit}:{type(exc).__name__}:{exc}"
            )
    check(
        "all_15_canary_inputs_bind_to_promoted_recovery4",
        (
            set(manifest_by_unit) == canary_units
            and not input_binding_errors
        ),
        {
            "bound_unit_n": EXPECTED_CANARIES
            - len(input_binding_errors),
            "errors": input_binding_errors,
        },
    )
    metadata_pass_n = sum(
        row["status"] == "PASS" for row in metadata_rows
    )
    metadata_waiting_n = sum(
        row["status"] == "WAITING" for row in metadata_rows
    )
    metadata_failures = [
        row for row in metadata_rows if row["status"] == "FAIL"
    ]
    complete = bool(
        len(metadata_rows) == EXPECTED_METADATA
        and metadata_pass_n == EXPECTED_METADATA
        and not metadata_failures
    )
    check(
        "available_canary_seed_metadata_matches_production_identity",
        not metadata_failures and metadata_pass_n > 0,
        {
            "pass_n": metadata_pass_n,
            "waiting_n": metadata_waiting_n,
            "failure_n": len(metadata_failures),
        },
    )

    passed = sum(row["status"] == "PASS" for row in checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_balanced_canary_production_equivalence_validation"
        ),
        "generated_utc": utc_now(),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "complete_seed_metadata": complete,
        "seed_metadata_expected_n": EXPECTED_METADATA,
        "seed_metadata_pass_n": metadata_pass_n,
        "seed_metadata_waiting_n": metadata_waiting_n,
        "seed_metadata_failure_n": len(metadata_failures),
        "canary_pair_complete_n": sum(
            all(
                any(
                    row["unit"] == unit
                    and row["seed_class"] == seed
                    and row["status"] == "PASS"
                    for row in metadata_rows
                )
                for seed in ("primary", "independent")
            )
            for unit in canary_units
        ),
        "checks": checks,
        "passed_checks": passed,
        "total_checks": len(checks),
        "metadata": metadata_rows,
        "diagnosis_labels_used": False,
        "imaging_executed": False,
        "production_execution_authorized": complete,
        "records": {
            "production_runner": file_record(RUNNER_SOURCE),
            "all_nine_builder": file_record(ALL_NINE_SOURCE),
            "phase_b_rule": file_record(PHASE_RULE_SOURCE),
            "recipe_config": file_record(RECIPE_CONFIG),
            "promoted_pretract_master": file_record(PRETRACT_MASTER),
            "topology": file_record(TOPOLOGY),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RUNNER.atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
