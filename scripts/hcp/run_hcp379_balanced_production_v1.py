#!/home/ec2-user/fsl/bin/python
"""Run the selected balanced final-HROI recipe on 515 production subjects.

This is the non-overwriting successor to the retired single-seed downstream
runner.  Execution is fail-closed behind four cohort-level gates:

* an independently replayed smallest-joint-pass recipe selection;
* exact 15-canary plus 515-production topology;
* valid compacted Recovery4 pretract inputs for the requested execution set
  (all 515 for the full run, or every member of a bounded pilot); and
* a completed genuine 16-unit final-HROI visual review.

For every production unit, two equal deterministic tractography replicates are
generated with the locked ACT/iFOD2 recipe and concatenated.  SIFT2 is fitted
once on that exact balanced tractogram, and all nine matrices are built from
the final HROI atlas.  Seed reliability is a cohort-level canary gate and is
not re-estimated per production subject.  Generated tractograms are hash
recorded and removed only after matrices, assignments, weights and subject QC
are durable.  No diagnosis or outcome field is read.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
ALL_NINE_SOURCE = (
    EXP
    / "scripts/hcp/"
    "build_hcp379_hroi_balanced_all_nine_canary_v1.py"
)
TOPOLOGY_SOURCE = (
    EXP / "scripts/hcp/build_hcp379_balanced_release_topology_v1.py"
)
TOPOLOGY_VALIDATOR = (
    EXP
    / "research_audit/"
    "validate_hcp379_balanced_release_topology_v1.py"
)
TOPOLOGY_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_balanced_release_topology_v1/validation.json"
)
EQUIVALENCE_VALIDATOR = (
    EXP
    / "research_audit/"
    "validate_hcp379_balanced_canary_production_equivalence_v1.py"
)
EQUIVALENCE_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_balanced_canary_production_equivalence_v1/validation.json"
)
SELECTION = (
    HCP_ROOT
    / "phase_b_hroi_balanced_all_nine_selection_v1/selection.json"
)
SELECTION_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_hroi_all_nine_selection_v1/validation.json"
)
RECIPE_CONFIG = EXP / "configs/connectome_v2_retry3.yaml"
REVIEW_ROOT = (
    HCP_ROOT
    / "source_label_repair_v1/review/hroi_release_v1/attempts/"
    "20260728T083614.440650Z-hroi-release-review"
)
REVIEW_MANIFEST = REVIEW_ROOT / "review_manifest.json"
DEFAULT_HUMAN_HROI_QC = (
    REVIEW_ROOT / "human_visual_qc_hroi_release_completed.csv"
)
ROOT = HCP_ROOT / "balanced_release_v1/production"
EXPECTED_PRODUCTION_N = 515
EXPECTED_REVIEW_N = 16
EXPECTED_NODES = 379
DENSITY_TARGET = 0.60
MINIMUM_START_FREE_BYTES = 800_000_000_000
MINIMUM_FREE_BYTES_PER_WORKER = 80_000_000_000
THREADS_PER_SUBJECT = 16
MAX_PRODUCTION_WORKERS = 12
RECIPE_ID = "connectome-v2.1.0-closed-world-canary-candidate"
REQUIRED_RETAINED = {
    "selected_five_tt": "five_tt",
    "selected_hcp379_nodes": "nodes",
    "selected_hcp379_node_volumes": "node_volumes",
    "selected_hcp379_atlas_qc": "atlas_qc",
    "wmfod_normalised": "wmfod",
    "fa_bounded": "fa",
    "md_bounded": "md",
    "rd_bounded": "rd",
    "ad_bounded": "ad",
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALL_NINE = load_module(ALL_NINE_SOURCE, "hcp379_balanced_production_all_nine")
TOPOLOGY = load_module(TOPOLOGY_SOURCE, "hcp379_balanced_production_topology")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_record(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or record.get("path") != str(path)
        or record.get("size_bytes") != path.stat().st_size
        or record.get("sha256") != sha256(path)
    ):
        raise ValueError(f"{label}: file record differs")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_utc(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return False
    return parsed.tzinfo is not None


def human_review_contract(path: Path) -> dict[str, Any]:
    manifest = load_json(REVIEW_MANIFEST)
    expected_units = {
        str(row.get("unit"))
        for row in manifest.get("units", [])
        if isinstance(row, Mapping)
    }
    if (
        manifest.get("record_type")
        != "diagnosis_blind_hcp379_hroi_release_visual_review"
        or manifest.get("status") != "READY_FOR_HUMAN_VISUAL_QC"
        or manifest.get("diagnosis_labels_used") is not False
        or manifest.get("human_visual_qc_inferred") is not False
        or manifest.get("unit_n") != EXPECTED_REVIEW_N
        or len(expected_units) != EXPECTED_REVIEW_N
    ):
        raise ValueError("final-HROI review manifest differs")
    if not path.is_file() or path.is_symlink():
        return {
            "status": "WAITING_FOR_COMPLETED_HUMAN_HROI_REVIEW",
            "path": str(path.resolve()),
            "expected_unit_n": EXPECTED_REVIEW_N,
            "completed_unit_n": 0,
        }
    rows = read_csv(path)
    required_columns = {
        "unit",
        "source_support_status",
        "dwi_atlas_status",
        "overall_status",
        "reviewer",
        "reviewed_utc",
        "note",
    }
    observed_units = {row.get("unit", "") for row in rows}
    invalid: list[str] = []
    for row in rows:
        unit = row.get("unit", "")
        statuses = (
            row.get("source_support_status", "").strip().upper(),
            row.get("dwi_atlas_status", "").strip().upper(),
            row.get("overall_status", "").strip().upper(),
        )
        if (
            statuses != ("PASS", "PASS", "PASS")
            or not row.get("reviewer", "").strip()
            or not parse_utc(row.get("reviewed_utc", "").strip())
        ):
            invalid.append(unit)
    complete = bool(
        len(rows) == EXPECTED_REVIEW_N
        and observed_units == expected_units
        and set(rows[0]) == required_columns
        and not invalid
    )
    return {
        "status": (
            "PASS_COMPLETED_HUMAN_HROI_REVIEW"
            if complete
            else "FAIL_OR_INCOMPLETE_HUMAN_HROI_REVIEW"
        ),
        "path": str(path.resolve()),
        "expected_unit_n": EXPECTED_REVIEW_N,
        "completed_unit_n": len(rows) - len(invalid),
        "invalid_units": sorted(invalid),
        "record": file_record(path),
        "manifest": file_record(REVIEW_MANIFEST),
    }


def recipe_contract() -> dict[str, Any]:
    value = yaml.safe_load(RECIPE_CONFIG.read_text(encoding="utf-8"))
    tract = value["tractography"]
    contract = value["contract"]
    if (
        contract.get("recipe_id") != RECIPE_ID
        or tract.get("algorithm") != "iFOD2"
        or tract.get("act_enabled") is not True
        or tract.get("backtrack") is not True
        or tract.get("crop_at_gmwmi") is not True
        or tract.get("seeding", {}).get("method") != "seed_dynamic"
        or tract.get("maximum_seed_attempts") != 200_000_000
        or tract.get("minimum_length_mm") != 10
        or tract.get("maximum_length_mm") != 250
        or float(tract.get("cutoff")) != 0.06
        or tract.get("maximum_angle_degrees") != 45
        or tract.get("nthreads_per_subject") != 16
    ):
        raise ValueError("locked tractography recipe differs")
    return value


def selection_contract() -> dict[str, Any]:
    if not SELECTION.is_file() or not SELECTION_VALIDATION.is_file():
        return {
            "status": "WAITING_FOR_RUNTIME_RECIPE_SELECTION",
            "selected_balanced_total_streamlines": None,
            "selected_per_seed_streamlines": None,
        }
    selection = load_json(SELECTION)
    validation = load_json(SELECTION_VALIDATION)
    selected = selection.get("selected_balanced_total_streamlines")
    per_seed = selection.get("selected_per_seed_streamlines")
    runtime = validation.get("runtime", {})
    valid = bool(
        selection.get("record_type")
        == "diagnosis_blind_hcp379_hroi_all_nine_recipe_selection"
        and selection.get("status")
        == "PASS_SMALLEST_JOINTLY_QUALIFIED_BALANCED_RECIPE"
        and selection.get("diagnosis_labels_used") is False
        and selection.get("non_overwriting") is True
        and selected in ALL_NINE.BALANCED_TOTALS
        and per_seed == ALL_NINE.BALANCED_TOTALS.get(selected)
        and selection.get("selection_is_smallest_joint_pass") is True
        and selection.get("per_subject_streamline_tuning_allowed") is False
        and selection.get(
            "subject_exclusion_for_low_density_allowed"
        )
        is False
        and selection.get("selected_count_tractography_authorized")
        is False
        and selection.get("final_release_authorized") is False
        and validation.get("record_type")
        == "hcp379_hroi_all_nine_selection_v1_validation"
        and validation.get("status") == "PASS"
        and validation.get("runtime_selection_detected") is True
        and runtime.get("status") == "PASS"
        and runtime.get("selected_balanced_total_streamlines") == selected
        and runtime.get("selected_per_seed_streamlines") == per_seed
        and runtime.get("selection") == file_record(SELECTION)
    )
    return {
        "status": (
            "PASS_RUNTIME_RECIPE_SELECTION"
            if valid
            else "FAIL_RUNTIME_RECIPE_SELECTION"
        ),
        "selected_balanced_total_streamlines": selected,
        "selected_per_seed_streamlines": per_seed,
        "selection": file_record(SELECTION),
        "validation": file_record(SELECTION_VALIDATION),
    }


def equivalence_contract(
    selection_status: str,
) -> dict[str, Any]:
    if selection_status != "PASS_RUNTIME_RECIPE_SELECTION":
        return {
            "status": "WAITING_FOR_COMPLETE_CANARY_EQUIVALENCE",
            "complete_seed_metadata": False,
            "seed_metadata_pass_n": 0,
            "canary_pair_complete_n": 0,
        }

    def valid(value: Mapping[str, Any]) -> bool:
        records = value.get("records", {})
        return bool(
            value.get("record_type")
            == (
                "hcp379_balanced_canary_production_"
                "equivalence_validation"
            )
            and value.get("status") == "PASS"
            and value.get("complete_seed_metadata") is True
            and value.get("seed_metadata_expected_n") == 30
            and value.get("seed_metadata_pass_n") == 30
            and value.get("seed_metadata_waiting_n") == 0
            and value.get("seed_metadata_failure_n") == 0
            and value.get("canary_pair_complete_n") == 15
            and value.get("production_execution_authorized") is True
            and value.get("diagnosis_labels_used") is False
            and value.get("imaging_executed") is False
            and records.get("production_runner")
            == file_record(Path(__file__))
            and records.get("all_nine_builder")
            == file_record(ALL_NINE_SOURCE)
        )

    value: dict[str, Any] = {}
    if EQUIVALENCE_VALIDATION.is_file():
        try:
            candidate = load_json(EQUIVALENCE_VALIDATION)
            if valid(candidate):
                value = candidate
        except Exception:
            value = {}
    if not value:
        probe = subprocess.run(
            [str(ALL_NINE.PYTHON), str(EQUIVALENCE_VALIDATOR)],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            candidate = json.loads(probe.stdout)
        except Exception as exc:
            raise ValueError(
                "canary-production equivalence stdout is not JSON"
            ) from exc
        if (
            probe.returncode != 0
            or not isinstance(candidate, dict)
            or not valid(candidate)
        ):
            return {
                "status": "FAIL_OR_INCOMPLETE_CANARY_EQUIVALENCE",
                "complete_seed_metadata": candidate.get(
                    "complete_seed_metadata"
                )
                if isinstance(candidate, dict)
                else False,
                "seed_metadata_pass_n": candidate.get(
                    "seed_metadata_pass_n"
                )
                if isinstance(candidate, dict)
                else 0,
                "canary_pair_complete_n": candidate.get(
                    "canary_pair_complete_n"
                )
                if isinstance(candidate, dict)
                else 0,
                "validator_returncode": probe.returncode,
                "validator_stderr": probe.stderr[-2000:],
            }
        value = candidate
    return {
        "status": "PASS_COMPLETE_CANARY_PRODUCTION_EQUIVALENCE",
        "complete_seed_metadata": True,
        "seed_metadata_pass_n": value["seed_metadata_pass_n"],
        "canary_pair_complete_n": value["canary_pair_complete_n"],
        "validation": file_record(EQUIVALENCE_VALIDATION),
    }


def topology_contract() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = TOPOLOGY.build()
    TOPOLOGY.atomic_json(TOPOLOGY.OUTPUT, value)
    validation_probe = subprocess.run(
        [str(ALL_NINE.PYTHON), str(TOPOLOGY_VALIDATOR)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        validation = json.loads(validation_probe.stdout)
    except Exception as exc:
        raise ValueError(
            "topology validator stdout is not JSON"
        ) from exc
    if (
        validation_probe.returncode != 0
        or validation.get("status") != "PASS"
        or validation.get("topology") != file_record(TOPOLOGY.OUTPUT)
    ):
        raise ValueError("independent balanced topology validation differs")
    value["independent_validation"] = file_record(
        TOPOLOGY_VALIDATION
    )
    production = value.get("production", [])
    valid = bool(
        value.get("record_type")
        == "diagnosis_blind_hcp379_balanced_release_topology"
        and value.get("diagnosis_labels_used") is False
        and value.get("non_overwriting") is True
        and value.get("cohort_uniform") is True
        and value.get("total_unit_n") == 530
        and value.get("canary_reuse_n") == 15
        and value.get("production_execution_n")
        == EXPECTED_PRODUCTION_N
        and len(production) == EXPECTED_PRODUCTION_N
        and len({row.get("unit") for row in production})
        == EXPECTED_PRODUCTION_N
        and value.get("selected_count_tractography_authorized") is False
        and value.get("final_release_authorized") is False
    )
    if not valid:
        raise ValueError("balanced 15+515 topology differs")
    return value, production


def preflight(
    human_hroi_qc: Path,
    *,
    planned_workers: int,
    required_units: set[str] | None = None,
) -> dict[str, Any]:
    logical_cpus = os.cpu_count() or 1
    if (
        not 1 <= planned_workers <= MAX_PRODUCTION_WORKERS
        or planned_workers * THREADS_PER_SUBJECT > logical_cpus
    ):
        raise ValueError(
            "planned workers exceed host-bound 16-thread subject capacity"
        )
    selection = selection_contract()
    equivalence = equivalence_contract(selection["status"])
    topology, production = topology_contract()
    human = human_review_contract(human_hroi_qc)
    recipe_contract()
    free_bytes = shutil.disk_usage(HCP_ROOT).free
    minimum_required_free_bytes = max(
        MINIMUM_START_FREE_BYTES,
        planned_workers * MINIMUM_FREE_BYTES_PER_WORKER,
    )
    storage_ready = free_bytes >= minimum_required_free_bytes
    all_pretract_ready = bool(
        topology.get("pretract_ready_n") == EXPECTED_PRODUCTION_N
        and all(row.get("pretract", {}).get("ready") for row in production)
    )
    production_by_unit = {
        str(row["unit"]): row for row in production
    }
    required = (
        set(production_by_unit)
        if required_units is None
        else set(required_units)
    )
    if (
        not required
        or not required.issubset(production_by_unit)
    ):
        raise ValueError("required execution-unit set differs from topology")
    required_ready_units = {
        unit
        for unit in required
        if production_by_unit[unit].get("pretract", {}).get("ready")
    }
    required_pretract_ready = required_ready_units == required
    gates = {
        "runtime_recipe_selection": selection["status"],
        "complete_canary_production_equivalence": equivalence[
            "status"
        ],
        "exact_15_plus_515_topology": "PASS",
        "all_515_pretract_ready": (
            "PASS" if all_pretract_ready else "WAITING"
        ),
        "required_execution_pretract_ready": (
            "PASS" if required_pretract_ready else "WAITING"
        ),
        "completed_human_hroi_review": human["status"],
        "storage_headroom": "PASS" if storage_ready else "WAITING",
    }
    ready = bool(
        selection["status"] == "PASS_RUNTIME_RECIPE_SELECTION"
        and equivalence["status"]
        == "PASS_COMPLETE_CANARY_PRODUCTION_EQUIVALENCE"
        and required_pretract_ready
        and human["status"] == "PASS_COMPLETED_HUMAN_HROI_REVIEW"
        and storage_ready
    )
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_balanced_production_preflight"
        ),
        "generated_utc": utc_now(),
        "status": (
            "READY_FOR_BALANCED_PRODUCTION_EXECUTION"
            if ready
            else "WAITING_FOR_BALANCED_PRODUCTION_GATES"
        ),
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "imaging_executed": False,
        "probabilistic_tractography_generated": False,
        "gates": gates,
        "selected_balanced_total_streamlines": selection.get(
            "selected_balanced_total_streamlines"
        ),
        "selected_per_seed_streamlines": selection.get(
            "selected_per_seed_streamlines"
        ),
        "pretract_ready_n": topology["pretract_ready_n"],
        "pretract_waiting_n": topology["pretract_waiting_n"],
        "required_execution_unit_n": len(required),
        "required_execution_pretract_ready_n": len(
            required_ready_units
        ),
        "required_execution_waiting_units": sorted(
            required - required_ready_units
        ),
        "partial_execution_preflight": (
            len(required) != EXPECTED_PRODUCTION_N
        ),
        "production_n": EXPECTED_PRODUCTION_N,
        "human_hroi_review": human,
        "selection": selection,
        "canary_production_equivalence": equivalence,
        "topology_generated_utc": topology["generated_utc"],
        "topology_validation": topology["independent_validation"],
        "per_subject_streamline_tuning_allowed": False,
        "subject_exclusion_for_low_density_allowed": False,
        "density_target": DENSITY_TARGET,
        "planned_workers": planned_workers,
        "threads_per_subject": THREADS_PER_SUBJECT,
        "logical_cpus": logical_cpus,
        "free_bytes": free_bytes,
        "minimum_start_free_bytes": minimum_required_free_bytes,
        "minimum_free_bytes_per_worker": (
            MINIMUM_FREE_BYTES_PER_WORKER
        ),
        "final_release_authorized": False,
    }


def pretract_inputs(row: Mapping[str, Any]) -> tuple[
    dict[str, Path], dict[str, Any]
]:
    unit = str(row["unit"])
    compaction_path = Path(
        str(row["pretract"]["compaction_path"])
    ).resolve()
    compaction = load_json(compaction_path)
    retained = compaction.get("retained_artifacts", {})
    if (
        compaction.get("record_type")
        != "diagnosis_blind_hcp379_pretract_compaction"
        or compaction.get("status") != "PASS_COMPACTED"
        or compaction.get("diagnosis_labels_used") is not False
        or compaction.get("recovery4_inputs") is not True
        or compaction.get("unit") != unit
        or not isinstance(retained, Mapping)
        or not set(REQUIRED_RETAINED).issubset(retained)
    ):
        raise ValueError(f"{unit}: compacted pretract contract differs")
    paths = {
        output_name: verify_record(
            retained[source_name],
            label=f"{unit}/pretract/{source_name}",
        )
        for source_name, output_name in REQUIRED_RETAINED.items()
    }
    atlas_qc = load_json(paths["atlas_qc"])
    if (
        atlas_qc.get("status") != "PASS"
        or atlas_qc.get("unit") != unit
        or atlas_qc.get("diagnosis_labels_used") is not False
        or atlas_qc.get("labels_found") != EXPECTED_NODES
        or atlas_qc.get("failures")
    ):
        raise ValueError(f"{unit}: final HROI atlas QC differs")
    spacing = compaction.get("dwi_spacing_mm")
    if (
        not isinstance(spacing, list)
        or len(spacing) != 3
        or any(float(value) <= 0 for value in spacing)
    ):
        raise ValueError(f"{unit}: DWI spacing differs")
    return paths, {
        "compaction": file_record(compaction_path),
        "dwi_spacing_mm": [float(value) for value in spacing],
        "registration_route": compaction.get("registration_route"),
    }


def seed_identity(row: Mapping[str, Any], seed_class: str) -> tuple[str, int]:
    token = (
        f"{row['dti_source_id']}|{row['dti_raw_bundle_sha256']}|"
        f"{RECIPE_ID}"
    )
    if seed_class == "independent":
        token += "|independent-replicate-1"
    seed = int.from_bytes(
        hashlib.sha256(token.encode("utf-8")).digest()[:4], "big"
    ) & 0x7FFFFFFF
    if seed_class == "independent":
        seed = seed or 1
    return token, seed


def run_logged(
    command: Sequence[str],
    *,
    log: Path,
    outputs: Iterable[Path],
    rng_seed: int | None = None,
) -> None:
    expected = list(outputs)
    for path in expected:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        raise FileExistsError(log)
    env = ALL_NINE.environment()
    if rng_seed is not None:
        env["MRTRIX_RNG_SEED"] = str(rng_seed)
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as handle:
        handle.write("COMMAND " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            list(command),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
        handle.write(
            f"\nRETURNCODE {completed.returncode}\n"
            f"ELAPSED_SECONDS {time.monotonic() - started:.3f}\n"
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed; see {log}")
    for path in expected:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"expected output is absent: {path}")


def generate_seed_tracks(
    *,
    row: Mapping[str, Any],
    seed_class: str,
    model: Mapping[str, Path],
    count: int,
    step: float,
    root: Path,
    threads: int,
) -> tuple[Path, dict[str, Any]]:
    token, rng_seed = seed_identity(row, seed_class)
    partial = root / ".tracks.partial.tck"
    final = root / "tracks.tck"
    metadata_path = root / "tractography_parameters.json"
    run_logged(
        [
            str(MRTRIX / "tckgen"),
            str(model["wmfod"]),
            str(partial),
            "-algorithm",
            "iFOD2",
            "-act",
            str(model["five_tt"]),
            "-backtrack",
            "-crop_at_gmwmi",
            "-seed_dynamic",
            str(model["wmfod"]),
            "-select",
            str(count),
            "-seeds",
            "200000000",
            "-minlength",
            "10.0",
            "-maxlength",
            "250.0",
            "-cutoff",
            "0.06",
            "-step",
            f"{step:.8f}",
            "-angle",
            "45.0",
            "-nthreads",
            str(threads),
        ],
        log=root / "tckgen.log",
        outputs=(partial,),
        rng_seed=rng_seed,
    )
    actual = ALL_NINE.tck_count(partial, count)
    os.replace(partial, final)
    metadata = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_balanced_production_tractography"
        ),
        "generated_utc": utc_now(),
        "unit": row["unit"],
        "diagnosis_labels_used": False,
        "seed_class": seed_class,
        "seed_id": token,
        "rng_seed": rng_seed,
        "recipe_id": RECIPE_ID,
        "algorithm": "iFOD2",
        "act": True,
        "backtrack": True,
        "crop_at_gmwmi": True,
        "seeding": "seed_dynamic",
        "requested_streamlines": count,
        "actual_streamlines": actual,
        "actual_step_size_mm": step,
        "dti_source_id": row["dti_source_id"],
        "dti_raw_bundle_sha256": row["dti_raw_bundle_sha256"],
        "tractogram": file_record(final),
    }
    atomic_json(metadata_path, metadata)
    return final, metadata


def compact_tracks(
    *,
    unit: str,
    attempt: Path,
    tracks: Mapping[str, Path],
    summary_path: Path,
) -> dict[str, Any]:
    records = {name: file_record(path) for name, path in tracks.items()}
    reclaimed = 0
    for path in tracks.values():
        reclaimed += path.stat().st_size
        os.remove(path)
    if any(path.exists() for path in tracks.values()):
        raise RuntimeError(f"{unit}: generated tract compaction incomplete")
    receipt = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_balanced_production_compaction"
        ),
        "status": "PASS_COMPACTED_GENERATED_TRACKS",
        "completed_utc": utc_now(),
        "unit": unit,
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "source_or_pretract_artifacts_deleted": False,
        "generated_tractograms_deleted": {
            name: {**record, "deleted": True}
            for name, record in records.items()
        },
        "reclaimed_bytes": reclaimed,
        "retained_scientific_summary": file_record(summary_path),
        "attempt_root": str(attempt.resolve()),
    }
    path = attempt / "compaction.json"
    atomic_json(path, receipt)
    return receipt


def existing_pass(
    unit: str,
    *,
    selected_total: int,
    per_seed: int,
) -> dict[str, Any] | None:
    state_path = ROOT / "qc/subjects" / f"{unit}.json"
    if not state_path.is_file() or state_path.is_symlink():
        return None
    try:
        state = load_json(state_path)
        if (
            state.get("status") != "PASS_BALANCED_PRODUCTION"
            or state.get("unit") != unit
            or state.get("diagnosis_labels_used") is not False
            or state.get("selected_balanced_total_streamlines")
            != selected_total
            or state.get("selected_per_seed_streamlines") != per_seed
        ):
            return None
        summary_path = verify_record(
            state["summary"], label=f"{unit}/cached summary"
        )
        compaction_path = verify_record(
            state["compaction"], label=f"{unit}/cached compaction"
        )
        summary = load_json(summary_path)
        compaction = load_json(compaction_path)
        view = summary.get("balanced_view", {})
        matrix_records = view.get("matrices", {})
        if (
            summary.get("status") != "PASS_BALANCED_PRODUCTION"
            or summary.get("selected_balanced_total_streamlines")
            != selected_total
            or summary.get("selected_per_seed_streamlines") != per_seed
            or summary.get("combined_density_target_met") is not True
            or compaction.get("status")
            != "PASS_COMPACTED_GENERATED_TRACKS"
            or set(matrix_records) != set(ALL_NINE.MATRIX_NAMES)
            or view.get("technical_status") != "PASS"
            or compaction.get("retained_scientific_summary")
            != file_record(summary_path)
            or any(
                Path(record["path"]).exists()
                for record in compaction.get(
                    "generated_tractograms_deleted", {}
                ).values()
            )
        ):
            return None
        for name, record in matrix_records.items():
            matrix_path = verify_record(
                record, label=f"{unit}/cached matrix/{name}"
            )
            ALL_NINE.matrix(matrix_path)
        verify_record(
            view["assignments"],
            label=f"{unit}/cached assignments",
        )
        for name in ("weights", "mu", "iterations"):
            verify_record(
                view["sift2"][name],
                label=f"{unit}/cached SIFT2/{name}",
            )
        return state
    except Exception:
        return None


def select_targets(
    production: Sequence[Mapping[str, Any]],
    *,
    limit: int | None,
    stratified_pilot: bool,
) -> list[Mapping[str, Any]]:
    """Choose an execution set without diagnosis or outcome information.

    Full production is always the exact sorted 515-unit topology.  A bounded
    pilot can instead round-robin across source lanes so that the operational
    check is not accidentally confined to the first lexicographic route.
    """
    ordered = sorted(production, key=lambda row: str(row["unit"]))
    if limit is None:
        return ordered
    if not stratified_pilot:
        return ordered[:limit]
    by_lane: dict[str, list[Mapping[str, Any]]] = {}
    for row in ordered:
        by_lane.setdefault(str(row["lane"]), []).append(row)
    selected: list[Mapping[str, Any]] = []
    lane_names = sorted(by_lane)
    index = 0
    while len(selected) < limit:
        added = False
        for lane in lane_names:
            rows = by_lane[lane]
            if index < len(rows):
                selected.append(rows[index])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        index += 1
    if len(selected) != limit:
        raise ValueError(
            f"stratified pilot selected {len(selected)}, expected {limit}"
        )
    return selected


def process_unit(
    row: Mapping[str, Any],
    *,
    selected_total: int,
    per_seed: int,
    gate_records: Mapping[str, Any],
    threads: int,
) -> dict[str, Any]:
    unit = str(row["unit"])
    lock_path = ROOT / "locks" / f"{unit}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        cached = existing_pass(
            unit,
            selected_total=selected_total,
            per_seed=per_seed,
        )
        if cached is not None:
            return cached
        attempt = (
            ROOT
            / "subjects"
            / unit
            / "attempts"
            / f"{stamp()}-balanced-{selected_total}"
        )
        attempt.mkdir(parents=True, exist_ok=False)
        state_path = ROOT / "qc/subjects" / f"{unit}.json"
        started = time.monotonic()
        running = {
            "schema_version": "1.0.0",
            "record_type": (
                "diagnosis_blind_hcp379_balanced_production_subject"
            ),
            "status": "RUNNING_BALANCED_PRODUCTION",
            "started_utc": utc_now(),
            "unit": unit,
            "lane": row["lane"],
            "diagnosis_labels_used": False,
            "non_overwriting": True,
            "attempt_root": str(attempt.resolve()),
            "selected_balanced_total_streamlines": selected_total,
            "selected_per_seed_streamlines": per_seed,
        }
        atomic_json(state_path, running)
        generated_tracks: dict[str, Path] = {}
        try:
            free_bytes = shutil.disk_usage(HCP_ROOT).free
            if free_bytes < MINIMUM_START_FREE_BYTES:
                raise RuntimeError(
                    "insufficient storage headroom before subject start: "
                    f"{free_bytes} < {MINIMUM_START_FREE_BYTES}"
                )
            model, pretract = pretract_inputs(row)
            step = min(pretract["dwi_spacing_mm"]) / 2.0
            bound = {
                **running,
                "status": "EXECUTION_INPUTS_BOUND",
                "step_size_mm": step,
                "free_bytes_at_start": free_bytes,
                "dti_source_id": row["dti_source_id"],
                "dti_raw_bundle_sha256": row[
                    "dti_raw_bundle_sha256"
                ],
                "pretract": pretract,
                "model_inputs": {
                    name: file_record(path)
                    for name, path in model.items()
                },
                "gate_records": dict(gate_records),
                "implementation": file_record(Path(__file__)),
                "all_nine_implementation": file_record(ALL_NINE_SOURCE),
                "recipe_config": file_record(RECIPE_CONFIG),
            }
            atomic_json(attempt / "preflight.json", bound)
            seed_metadata: dict[str, Any] = {}
            for seed_class in ("primary", "independent"):
                tracks, metadata = generate_seed_tracks(
                    row=row,
                    seed_class=seed_class,
                    model=model,
                    count=per_seed,
                    step=step,
                    root=attempt / seed_class,
                    threads=threads,
                )
                generated_tracks[seed_class] = tracks
                seed_metadata[seed_class] = metadata
            if (
                seed_metadata["primary"]["seed_id"]
                == seed_metadata["independent"]["seed_id"]
            ):
                raise ValueError(f"{unit}: replicate seed identities match")
            balanced = ALL_NINE.concatenate(
                generated_tracks["primary"],
                generated_tracks["independent"],
                output=attempt / "balanced/tracks.tck",
                expected=selected_total,
                log=attempt / "balanced/concatenate.log",
                threads=threads,
            )
            generated_tracks["balanced"] = balanced
            _, view = ALL_NINE.build_view(
                name="balanced",
                tracks=balanced,
                expected=selected_total,
                nodes=model["nodes"],
                node_volumes_path=model["node_volumes"],
                model=model,
                root=attempt / "balanced/all_nine",
                threads=threads,
            )
            density = float(view["technical_qc"]["edge_density"])
            technical_pass = view["technical_status"] == "PASS"
            density_pass = density >= DENSITY_TARGET
            status = (
                "PASS_BALANCED_PRODUCTION"
                if technical_pass and density_pass
                else "FAIL_BALANCED_PRODUCTION_SCIENTIFIC_QC"
            )
            summary = {
                "schema_version": "1.0.0",
                "record_type": (
                    "diagnosis_blind_hcp379_balanced_production_summary"
                ),
                "status": status,
                "generated_utc": utc_now(),
                "unit": unit,
                "lane": row["lane"],
                "diagnosis_labels_used": False,
                "non_overwriting": True,
                "source_files_modified": False,
                "selected_balanced_total_streamlines": selected_total,
                "selected_per_seed_streamlines": per_seed,
                "balanced_construction": (
                    f"{per_seed}_primary_plus_{per_seed}_independent"
                ),
                "seed_metadata": seed_metadata,
                "sift2_fit_on_exact_balanced_tractogram": True,
                "independent_sift2_matrices_summed": False,
                "matrix_names": list(ALL_NINE.MATRIX_NAMES),
                "balanced_view": view,
                "combined_edge_density": density,
                "combined_density_target": DENSITY_TARGET,
                "combined_density_target_met": density_pass,
                "technical_qc_pass": technical_pass,
                "per_subject_streamline_tuning_allowed": False,
                "subject_exclusion_for_low_density_allowed": False,
                "preflight": file_record(attempt / "preflight.json"),
                "elapsed_seconds_before_compaction": round(
                    time.monotonic() - started, 3
                ),
            }
            summary_path = attempt / "summary.json"
            atomic_json(summary_path, summary)
            compaction = compact_tracks(
                unit=unit,
                attempt=attempt,
                tracks=generated_tracks,
                summary_path=summary_path,
            )
            final = {
                **running,
                "status": status,
                "completed_utc": utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "combined_edge_density": density,
                "combined_density_target_met": density_pass,
                "summary": file_record(summary_path),
                "compaction": file_record(attempt / "compaction.json"),
                "reclaimed_generated_track_bytes": compaction[
                    "reclaimed_bytes"
                ],
            }
        except Exception as exc:
            final = {
                **running,
                "status": "FAIL_BALANCED_PRODUCTION_EXECUTION",
                "completed_utc": utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error": f"{type(exc).__name__}:{exc}",
            }
        atomic_json(state_path, final)
        return final


def execute(
    *,
    human_hroi_qc: Path,
    workers: int,
    threads: int,
    limit: int | None,
    stratified_pilot: bool,
) -> dict[str, Any]:
    if threads != THREADS_PER_SUBJECT:
        raise ValueError(
            f"exactly {THREADS_PER_SUBJECT} threads per subject required"
        )
    topology, production = topology_contract()
    targets = select_targets(
        production,
        limit=limit,
        stratified_pilot=stratified_pilot,
    )
    target_units = {str(row["unit"]) for row in targets}
    ready = preflight(
        human_hroi_qc,
        planned_workers=workers,
        required_units=target_units,
    )
    if ready["status"] != "READY_FOR_BALANCED_PRODUCTION_EXECUTION":
        raise RuntimeError(ready["status"])
    topology, production = topology_contract()
    selected = int(ready["selected_balanced_total_streamlines"])
    per_seed = int(ready["selected_per_seed_streamlines"])
    targets = select_targets(
        production,
        limit=limit,
        stratified_pilot=stratified_pilot,
    )
    if {str(row["unit"]) for row in targets} != target_units:
        raise ValueError("execution target set changed during gate binding")
    attempt = ROOT / "attempts" / f"{stamp()}-balanced-production"
    attempt.mkdir(parents=True, exist_ok=False)
    bound = {
        **ready,
        "status": "EXECUTION_INPUTS_BOUND",
        "execution_target_n": len(targets),
        "full_production_n": EXPECTED_PRODUCTION_N,
        "partial_execution_only": len(targets) != EXPECTED_PRODUCTION_N,
        "stratified_pilot": stratified_pilot,
        "fail_fast_after_first_nonpass": True,
        "workers": workers,
        "threads_per_worker": threads,
        "topology_snapshot": topology,
        "implementation": file_record(Path(__file__)),
    }
    atomic_json(attempt / "preflight.json", bound)
    gate_records = {
        "cohort_preflight": file_record(attempt / "preflight.json"),
        "recipe_selection": file_record(SELECTION),
        "recipe_selection_validation": file_record(
            SELECTION_VALIDATION
        ),
        "human_hroi_review": file_record(human_hroi_qc),
        "human_hroi_review_manifest": file_record(REVIEW_MANIFEST),
        "topology": file_record(TOPOLOGY.OUTPUT),
        "topology_validation": file_record(TOPOLOGY_VALIDATION),
    }
    results: list[dict[str, Any]] = []
    unstarted = list(targets)
    active: dict[Future[dict[str, Any]], Mapping[str, Any]] = {}
    stop_scheduling = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while unstarted and len(active) < workers:
            row = unstarted.pop(0)
            future = pool.submit(
                process_unit,
                row,
                selected_total=selected,
                per_seed=per_seed,
                gate_records=gate_records,
                threads=threads,
            )
            active[future] = row
        while active:
            completed, _ = wait(
                active, return_when=FIRST_COMPLETED
            )
            for future in completed:
                active.pop(future)
                result = future.result()
                results.append(result)
                if result.get("status") != "PASS_BALANCED_PRODUCTION":
                    stop_scheduling = True
            while (
                not stop_scheduling
                and unstarted
                and len(active) < workers
            ):
                row = unstarted.pop(0)
                future = pool.submit(
                    process_unit,
                    row,
                    selected_total=selected,
                    per_seed=per_seed,
                    gate_records=gate_records,
                    threads=threads,
                )
                active[future] = row
    results.sort(key=lambda row: row["unit"])
    counts: dict[str, int] = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    pass_n = counts.get("PASS_BALANCED_PRODUCTION", 0)
    stopped_early = bool(
        stop_scheduling and len(results) < len(targets)
    )
    status = (
        "PASS_BALANCED_PRODUCTION_COHORT"
        if pass_n == len(targets)
        else (
            "FAIL_FAST_BALANCED_PRODUCTION_COHORT"
            if stopped_early
            else "FAIL_BALANCED_PRODUCTION_COHORT"
        )
    )
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_balanced_production_cohort"
        ),
        "status": status,
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "selected_balanced_total_streamlines": selected,
        "selected_per_seed_streamlines": per_seed,
        "target_n": len(targets),
        "full_production_n": EXPECTED_PRODUCTION_N,
        "processed_n": len(results),
        "unstarted_n": len(targets) - len(results),
        "unstarted_units": sorted(
            str(row["unit"]) for row in unstarted
        ),
        "fail_fast_after_first_nonpass": True,
        "stopped_early": stopped_early,
        "stratified_pilot": stratified_pilot,
        "pass_n": pass_n,
        "status_counts": counts,
        "all_subjects_density_target_met": all(
            row.get("combined_density_target_met") is True
            for row in results
        ),
        "per_subject_streamline_tuning_allowed": False,
        "subject_exclusion_for_low_density_allowed": False,
        "final_release_authorized": False,
        "subjects": results,
        "preflight": file_record(attempt / "preflight.json"),
        "implementation": file_record(Path(__file__)),
    }
    atomic_json(attempt / "summary.json", summary)
    atomic_json(ROOT / "cohort_state.json", summary)
    return {
        "status": status,
        "attempt_root": str(attempt.resolve()),
        "processed_n": len(results),
        "pass_n": pass_n,
        "unstarted_n": len(targets) - len(results),
        "stopped_early": stopped_early,
        "status_counts": counts,
        "final_release_authorized": False,
    }


def self_test() -> dict[str, Any]:
    row = {
        "dti_source_id": "a" * 64,
        "dti_raw_bundle_sha256": "b" * 64,
    }
    primary_token, primary_seed = seed_identity(row, "primary")
    independent_token, independent_seed = seed_identity(
        row, "independent"
    )
    checks = {
        "exact_partition": 15 + EXPECTED_PRODUCTION_N == 530,
        "balanced_candidates_match_canary": ALL_NINE.BALANCED_TOTALS
        == {
            6_000_000: 3_000_000,
            10_000_000: 5_000_000,
            15_000_000: 7_500_000,
            20_000_000: 10_000_000,
        },
        "matrix_family_is_all_nine": len(ALL_NINE.MATRIX_NAMES) == 9,
        "replicate_tokens_and_seeds_differ": (
            primary_token != independent_token
            and primary_seed != independent_seed
        ),
        "fixed_density_target": DENSITY_TARGET == 0.60,
        "fixed_threads_per_subject": THREADS_PER_SUBJECT == 16,
        "resize_ceiling_supports_192_logical_cpus": (
            MAX_PRODUCTION_WORKERS * THREADS_PER_SUBJECT == 192
        ),
        "per_worker_storage_reserve_is_explicit": (
            MINIMUM_FREE_BYTES_PER_WORKER == 80_000_000_000
        ),
        "stratified_pilot_is_lane_diverse": len(
            {
                row["lane"]
                for row in select_targets(
                    [
                        {"unit": "a1", "lane": "a"},
                        {"unit": "a2", "lane": "a"},
                        {"unit": "b1", "lane": "b"},
                        {"unit": "c1", "lane": "c"},
                    ],
                    limit=3,
                    stratified_pilot=True,
                )
            }
        )
        == 3,
        "production_never_authorizes_release": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--human-hroi-qc",
        type=Path,
        default=DEFAULT_HUMAN_HROI_QC,
    )
    default_workers = min(
        MAX_PRODUCTION_WORKERS,
        max(1, (os.cpu_count() or 1) // THREADS_PER_SUBJECT),
    )
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--threads-per-worker", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--stratified-pilot",
        action="store_true",
        help=(
            "With --limit, select subjects round-robin across source lanes "
            "instead of taking only the first lexicographic units."
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_PRODUCTION_WORKERS:
        parser.error(
            f"--workers must be in 1..{MAX_PRODUCTION_WORKERS}"
        )
    if args.threads_per_worker != THREADS_PER_SUBJECT:
        parser.error(
            f"--threads-per-worker must equal {THREADS_PER_SUBJECT}"
        )
    logical_cpus = os.cpu_count() or 1
    if args.workers * args.threads_per_worker > logical_cpus:
        parser.error(
            "worker/thread product exceeds available logical CPU capacity"
        )
    if args.limit is not None and not 1 <= args.limit <= EXPECTED_PRODUCTION_N:
        parser.error("--limit must be in 1..515")
    if args.stratified_pilot and args.limit is None:
        parser.error("--stratified-pilot requires --limit")
    if args.self_test:
        value = self_test()
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0 if value["status"] == "PASS" else 1
    if args.execute:
        value = execute(
            human_hroi_qc=args.human_hroi_qc.resolve(),
            workers=args.workers,
            threads=args.threads_per_worker,
            limit=args.limit,
            stratified_pilot=args.stratified_pilot,
        )
    else:
        required_units = None
        if args.limit is not None:
            _, production = topology_contract()
            required_units = {
                str(row["unit"])
                for row in select_targets(
                    production,
                    limit=args.limit,
                    stratified_pilot=args.stratified_pilot,
                )
            }
        value = preflight(
            args.human_hroi_qc.resolve(),
            planned_workers=args.workers,
            required_units=required_units,
        )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if not args.execute or value["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
