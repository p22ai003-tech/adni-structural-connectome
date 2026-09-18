#!/usr/bin/env python3
"""Run the locked corrected-ACT HCP379 recipe for 216 non-canary subjects.

The script is diagnosis blind and fail closed.  It starts only after:

* all 216 non-canary pre-tractography subjects pass;
* the 15-subject HCP379 Phase-B stability validation passes; and
* Phase B selects a stable, density-qualified streamline count.

For every subject it generates the selected-count iFOD2/ACT tractogram,
track-matched SIFT2 weights, assignments and all nine HCP379 matrices.
Density is a diagnosis-blind whole-cohort technical release gate, not a rule
for dropping individual subjects: an output below 0.60 fails recovery and
must be debugged under a uniformly validated recipe.  The 11 routed canary
subjects are merged from the already validated Phase-B primary run to form
the exact 227-subject corrected-track release.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
ROOT = HCP_ROOT / "corrected_scaleup"
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
AUDIT_CSV = (
    EXP
    / "research_audit/outputs/hcp379_scaleup_input_audit_v2/"
    "scaleup_input_audit.csv"
)
AUDIT_SUMMARY = AUDIT_CSV.with_name("scaleup_input_audit_summary.json")
PRETRACT_SUMMARY = ROOT / "manifests/pretract_summary.json"
PRETRACT_GATE = ROOT / "manifests/pretract_gate.json"
PHASE_MANIFEST = (
    RUN_ROOT / "publication/hcp379_recipe_stability_manifest.json"
)
PHASE_VALIDATION = (
    RUN_ROOT / "publication/hcp379_recipe_stability_validation.json"
)
CONFIG = EXP / "configs/connectome_v2_retry3.yaml"
PRETRACT_SOURCE = EXP / "scripts/hcp/hcp379_scaleup_pretract_v2.py"
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
FSL_PYTHON = Path("/home/ec2-user/fsl/bin/python")
TCK_HEADER_COUNT = EXP / "scripts/hcp/tck_header_count_v1.py"
NUMERIC_VALUE_COUNT = (
    EXP / "scripts/hcp/count_mrtrix_numeric_values_v1.py"
)
EXPECTED_SCALEUP_N = 216
EXPECTED_CORRECTED_N = 227
EXPECTED_CANARY_OVERLAP_N = 11
EXPECTED_NODES = 379
MATRIX_NAMES = (
    "count",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)
HUB_IDS = (361, 365, 366, 370, 374, 375)
MINIMUM_CONNECTED_NODES = 360
MINIMUM_ASSIGNMENT_FRACTION = 0.50
MINIMUM_WEIGHTED_SUPPORT = 0.99
MINIMUM_EDGE_DENSITY = 0.60
ALLOWED_COUNTS = (3_000_000, 5_000_000, 10_000_000)
MINIMUM_START_FREE_BYTES = 700_000_000_000
EMERGENCY_FREE_BYTES = 300_000_000_000
CORRECTED_RELEASE_RECORD_TYPE = (
    "diagnosis_blind_hcp379_corrected_227_release_manifest"
)
CORRECTED_RELEASE_FILENAME = "corrected_227_release_manifest.json"


def load_pretract_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_scaleup_pretract_v2", PRETRACT_SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(PRETRACT_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRETRACT = load_pretract_module()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def partial_path(path: Path) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}.partial{path.suffix}")
    return path.with_name(path.name + ".partial")


def check_free_space() -> int:
    free = shutil.disk_usage(ROOT).free
    if free < EMERGENCY_FREE_BYTES:
        raise RuntimeError(
            f"emergency storage floor crossed: {free} bytes free"
        )
    return free


def run_command(
    command: list[str],
    *,
    log: Any,
    env: Mapping[str, str],
    outputs: tuple[Path, ...] = (),
) -> None:
    for output in outputs:
        if output.is_file() or output.is_symlink():
            output.unlink()
        output.parent.mkdir(parents=True, exist_ok=True)
    log.write(f"\n[{PRETRACT.utc_now()}] COMMAND {json.dumps(command)}\n")
    log.flush()
    result = subprocess.run(
        command,
        check=False,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(env),
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed rc={result.returncode}: {command[0]}"
        )
    missing = [
        str(path)
        for path in outputs
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if missing:
        raise RuntimeError("command outputs absent: " + ",".join(missing))


def tck_count(path: Path, env: Mapping[str, str]) -> int:
    result = subprocess.run(
        [str(FSL_PYTHON), str(TCK_HEADER_COUNT), str(path)],
        check=True,
        capture_output=True,
        text=True,
        env=dict(env),
    )
    return int(result.stdout.strip())


def valid_tck(
    path: Path, expected: int, env: Mapping[str, str]
) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        return tck_count(path, env) == expected
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def nonempty_line_count(path: Path) -> int:
    result = subprocess.run(
        [str(FSL_PYTHON), str(NUMERIC_VALUE_COUNT), str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def assignment_stats(path: Path) -> tuple[int, int]:
    rows = 0
    assigned = 0
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.replace(",", " ").split()
            if len(fields) != 2:
                raise ValueError(
                    "assignment row does not contain exactly two endpoints: "
                    f"line {line_number}"
                )
            left, right = (int(value) for value in fields)
            if not 0 <= left <= 379 or not 0 <= right <= 379:
                raise ValueError(
                    "assignment endpoint outside 0..379: "
                    f"line {line_number}"
                )
            rows += 1
            assigned += int(left > 0 and right > 0)
    return rows, assigned


def node_volumes(path: Path) -> np.ndarray:
    rows = PRETRACT.read_csv(path)
    rows.sort(key=lambda row: int(row["node_id"]))
    if (
        len(rows) != EXPECTED_NODES
        or [int(row["node_id"]) for row in rows]
        != list(range(1, EXPECTED_NODES + 1))
    ):
        raise ValueError("HCP379 node-volume identity differs")
    values = np.asarray(
        [float(row["volume_mm3"]) for row in rows], dtype=float
    )
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("HCP379 node volumes are invalid")
    return values


def matrix_qc(
    matrix_paths: Mapping[str, Path],
    *,
    volumes: np.ndarray,
    assigned_streamlines: int,
) -> tuple[dict[str, Any], list[str]]:
    matrices: dict[str, np.ndarray] = {}
    failures: list[str] = []
    for name in MATRIX_NAMES:
        path = matrix_paths[name]
        try:
            matrix = np.loadtxt(path, delimiter=",")
        except Exception as exc:
            failures.append(f"{name}:read:{type(exc).__name__}")
            continue
        matrices[name] = matrix
        if matrix.shape != (EXPECTED_NODES, EXPECTED_NODES):
            failures.append(f"{name}:shape={matrix.shape}")
            continue
        if not np.isfinite(matrix).all():
            failures.append(f"{name}:nonfinite")
        if not np.allclose(matrix, matrix.T, atol=1.0e-6, rtol=1.0e-6):
            failures.append(f"{name}:asymmetric")
        if not np.allclose(np.diag(matrix), 0.0, atol=1.0e-8):
            failures.append(f"{name}:diagonal")
        if float(np.min(matrix)) < -1.0e-10:
            failures.append(f"{name}:negative")
    if set(matrices) != set(MATRIX_NAMES):
        return {}, sorted(set(failures))
    count = matrices["count"]
    if not np.allclose(count, np.rint(count), atol=1.0e-6, rtol=0.0):
        failures.append("count:not_integer")
    observed_assigned = float(np.sum(count) / 2.0)
    if not math.isclose(
        observed_assigned,
        float(assigned_streamlines),
        abs_tol=0.5,
        rel_tol=0.0,
    ):
        failures.append(
            f"count:assigned={observed_assigned}:"
            f"assignments={assigned_streamlines}"
        )
    count_support = count > 0
    np.fill_diagonal(count_support, False)
    supported_edges = int(np.count_nonzero(np.triu(count_support, 1)))
    possible_edges = EXPECTED_NODES * (EXPECTED_NODES - 1) // 2
    edge_density = supported_edges / possible_edges
    if edge_density < MINIMUM_EDGE_DENSITY:
        failures.append(
            f"edge_density={edge_density:.12f}"
            f"<{MINIMUM_EDGE_DENSITY:.12f}"
        )
    strengths = np.sum(count, axis=1)
    connected = int(np.count_nonzero(strengths > 0))
    if connected < MINIMUM_CONNECTED_NODES:
        failures.append(
            f"connected_nodes={connected}<{MINIMUM_CONNECTED_NODES}"
        )
    hub_strengths = {
        str(node): float(strengths[node - 1]) for node in HUB_IDS
    }
    if any(value <= 0 for value in hub_strengths.values()):
        failures.append("locked_hub_disconnected")
    support_qc: dict[str, Any] = {}
    for name, matrix in matrices.items():
        if name == "count":
            continue
        support = matrix > 0
        np.fill_diagonal(support, False)
        outside = int(
            np.count_nonzero(np.triu(support & ~count_support, 1))
        )
        covered = int(
            np.count_nonzero(np.triu(support & count_support, 1))
        )
        fraction = covered / supported_edges if supported_edges else 0.0
        support_qc[name] = {
            "count_edges_covered": covered,
            "count_supported_edges": supported_edges,
            "count_edge_support_fraction": fraction,
            "weighted_edges_outside_count_support": outside,
        }
        if outside:
            failures.append(f"{name}:outside_count_support={outside}")
        if fraction < MINIMUM_WEIGHTED_SUPPORT:
            failures.append(
                f"{name}:support={fraction:.6f}<"
                f"{MINIMUM_WEIGHTED_SUPPORT:.6f}"
            )
    fa_nonzero = matrices["fa_mean"][matrices["fa_mean"] > 0]
    if fa_nonzero.size and float(np.max(fa_nonzero)) > 1.000001:
        failures.append(f"fa_mean:max={float(np.max(fa_nonzero))}")
    denominator = volumes[:, None] + volumes[None, :]
    expected_inverse = 2.0 * count / denominator
    np.fill_diagonal(expected_inverse, 0.0)
    formula_error = float(
        np.max(np.abs(matrices["count_invnodevol"] - expected_inverse))
    )
    if not np.allclose(
        matrices["count_invnodevol"],
        expected_inverse,
        atol=1.0e-8,
        rtol=1.0e-7,
    ):
        failures.append("count_invnodevol:formula")
    return (
        {
            "supported_edges": supported_edges,
            "possible_edges": possible_edges,
            "edge_density": edge_density,
            "connected_nodes": connected,
            "hub_strengths": hub_strengths,
            "weighted_support": support_qc,
            "count_invnodevol_maximum_formula_error": formula_error,
        },
        sorted(set(failures)),
    )


def tract_paths(root: Path, unit: str) -> dict[str, Any]:
    pretract = PRETRACT.stage_paths(root, unit)
    subject = pretract["subject"]
    matrices = {
        name: subject / "08_connectome" / f"{name}.csv"
        for name in MATRIX_NAMES
    }
    samples = {
        metric: subject / "07_tractography/samples" / f"{metric}.csv"
        for metric in ("fa", "md", "rd", "ad")
    }
    return {
        **pretract,
        "tract_state": root / "qc/tractography" / f"{unit}.json",
        "tract_lock": root / "locks/tractography" / f"{unit}.lock",
        "tract_log": root / "logs/tractography" / f"{unit}.log",
        "tracks": subject / "07_tractography/tracks_selected.tck",
        "tract_metadata": (
            subject / "07_tractography/tractography_parameters.json"
        ),
        "weights": subject / "07_tractography/sift2_weights.txt",
        "sift2_mu": subject / "07_tractography/sift2_mu.txt",
        "sift2_iterations": (
            subject / "07_tractography/sift2_iterations.csv"
        ),
        "assignments": subject / "08_connectome/assignments.csv",
        "matrices": matrices,
        "samples": samples,
    }


def tractogram_compaction_path(root: Path, unit: str) -> Path:
    return root / "qc/tractography_compaction" / f"{unit}.json"


def validate_tractogram_compaction(
    record: Mapping[str, Any],
    *,
    root: Path,
    unit: str,
) -> dict[str, Any]:
    if (
        record.get("record_type")
        != "diagnosis_blind_hcp379_tractogram_compaction"
        or record.get("status") != "PASS_COMPACTED"
        or record.get("diagnosis_labels_used") is not False
        or record.get("unit") != unit
        or Path(str(record.get("root", ""))).resolve()
        != root.resolve()
    ):
        raise ValueError(f"{unit}:tractogram compaction identity differs")
    deleted = record.get("deleted_tractogram")
    retained = record.get("retained_artifacts")
    if (
        not isinstance(deleted, Mapping)
        or deleted.get("deleted") is not True
        or Path(str(deleted.get("path", ""))).exists()
        or not isinstance(retained, Mapping)
    ):
        raise ValueError(f"{unit}:tractogram compaction artifacts differ")
    verified_scaleup_artifacts(retained, unit=unit)
    return dict(record)


def compact_validated_tractogram(
    *,
    root: Path,
    unit: str,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        state.get("record_type")
        != "diagnosis_blind_hcp379_scaleup_tractography_subject"
        or state.get("status") != "PASS_ALL_NINE"
        or state.get("diagnosis_labels_used") is not False
        or state.get("unit") != unit
    ):
        raise ValueError(f"{unit}:tractography state is not compactable")
    result_path = tractogram_compaction_path(root, unit)
    if result_path.is_file():
        return validate_tractogram_compaction(
            PRETRACT.load_json(result_path), root=root, unit=unit
        )
    plan_path = (
        root / "qc/tractography_compaction_plans" / f"{unit}.json"
    )
    if plan_path.is_file():
        plan = PRETRACT.load_json(plan_path)
        if (
            plan.get("record_type")
            != "diagnosis_blind_hcp379_tractogram_compaction_plan"
            or plan.get("status") != "READY"
            or plan.get("diagnosis_labels_used") is not False
            or plan.get("unit") != unit
            or Path(str(plan.get("root", ""))).resolve()
            != root.resolve()
        ):
            raise ValueError(
                f"{unit}:tractogram compaction plan identity differs"
            )
        tractogram = plan.get("tractogram")
        retained = plan.get("retained_artifacts")
        if (
            not isinstance(tractogram, Mapping)
            or not isinstance(retained, Mapping)
        ):
            raise TypeError(
                f"{unit}:tractogram compaction plan artifacts differ"
            )
        verified_scaleup_artifacts(retained, unit=unit)
    else:
        artifacts = state.get("artifacts")
        verified = verified_scaleup_artifacts(artifacts, unit=unit)
        tractogram = verified.get("tractogram")
        if not isinstance(tractogram, Mapping):
            raise ValueError(f"{unit}:tractogram artifact is absent")
        retained = {
            name: record
            for name, record in verified.items()
            if name != "tractogram"
        }
        plan = {
            "schema_version": "1.0.0",
            "record_type": (
                "diagnosis_blind_hcp379_tractogram_compaction_plan"
            ),
            "status": "READY",
            "generated_utc": PRETRACT.utc_now(),
            "diagnosis_labels_used": False,
            "unit": unit,
            "root": str(root.resolve()),
            "source_state": PRETRACT.file_record(
                tract_paths(root, unit)["tract_state"]
            ),
            "tractogram": dict(tractogram),
            "retained_artifacts": retained,
            "reclaimable_bytes": int(tractogram["size_bytes"]),
        }
        PRETRACT.atomic_json(plan_path, plan)
    path = Path(str(tractogram["path"])).resolve()
    expected_track = tract_paths(root, unit)["tracks"].resolve()
    expected_subject = (root / "subjects" / unit).resolve()
    if (
        path != expected_track
        or not path.is_relative_to(expected_subject)
    ):
        raise ValueError(
            f"{unit}:refusing unsafe tractogram compaction target {path}"
        )
    if path.exists():
        verified_artifact_record(
            tractogram, label=f"{unit}/tractogram_before_compaction"
        )
        path.unlink()
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_tractogram_compaction"
        ),
        "status": "PASS_COMPACTED",
        "completed_utc": PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "unit": unit,
        "root": str(root.resolve()),
        "source_state": plan["source_state"],
        "plan": PRETRACT.file_record(plan_path),
        "deleted_tractogram": {**dict(tractogram), "deleted": True},
        "retained_artifacts": {
            name: verified_artifact_record(
                record, label=f"{unit}/retained/{name}"
            )
            for name, record in retained.items()
        },
        "reclaimed_bytes": int(tractogram["size_bytes"]),
        "recoverability": (
            "The deterministic tractography recipe, seed token, validated "
            "weights, assignments, matrices and source derivatives are "
            "retained; the tractogram can be regenerated exactly within the "
            "declared MRtrix environment."
        ),
    }
    PRETRACT.atomic_json(result_path, result)
    return validate_tractogram_compaction(
        result, root=root, unit=unit
    )


def tractography_spacing(
    *,
    root: Path,
    unit: str,
    paths: Mapping[str, Any],
    env: Mapping[str, str],
) -> float:
    dwi = Path(paths["dwi"])
    if dwi.is_file():
        values = subprocess.run(
            [str(MRTRIX / "mrinfo"), str(dwi), "-spacing"],
            check=True,
            capture_output=True,
            text=True,
            env=dict(env),
        ).stdout.split()
        spacing = [float(value) for value in values[:3]]
    else:
        record_path = root / "qc/pretract_compaction" / f"{unit}.json"
        record = PRETRACT.load_json(record_path)
        if (
            record.get("record_type")
            != "diagnosis_blind_hcp379_pretract_compaction"
            or record.get("status") != "PASS_COMPACTED"
            or record.get("diagnosis_labels_used") is not False
            or record.get("unit") != unit
        ):
            raise ValueError(f"{unit}:pretract compaction record differs")
        retained = record.get("retained_artifacts")
        if not isinstance(retained, Mapping):
            raise TypeError(f"{unit}:retained compaction artifacts differ")
        required = {
            "five_tt",
            "wmfod_norm",
            "hcp_nodes",
            "hcp_volumes",
            "fa",
            "md",
            "rd",
            "ad",
            "atlas_qc",
            "automated_qc",
        }
        if not required.issubset(retained):
            raise ValueError(
                f"{unit}:required compacted artifacts are absent"
            )
        for name in required:
            verified_artifact_record(
                retained[name],
                label=f"{unit}/pretract_compaction/{name}",
            )
        deleted = record.get("deleted_artifacts")
        dwi_record = (
            deleted.get("dwi")
            if isinstance(deleted, Mapping)
            else None
        )
        if (
            not isinstance(dwi_record, Mapping)
            or dwi_record.get("deleted") is not True
            or Path(str(dwi_record.get("path", ""))).exists()
        ):
            raise ValueError(f"{unit}:compacted DWI deletion differs")
        raw_spacing = record.get("dwi_spacing_mm")
        if not isinstance(raw_spacing, list) or len(raw_spacing) != 3:
            raise ValueError(f"{unit}:compacted DWI spacing differs")
        spacing = [float(value) for value in raw_spacing]
    if (
        len(spacing) != 3
        or not all(math.isfinite(value) and value > 0 for value in spacing)
    ):
        raise ValueError(f"{unit}:DWI spacing differs: {spacing}")
    step = min(spacing) / 2.0
    if not math.isfinite(step) or step <= 0:
        raise ValueError(f"{unit}:step size differs")
    return step


def load_gate(root: Path) -> dict[str, Any]:
    audit = PRETRACT.load_json(AUDIT_SUMMARY)
    pretract = PRETRACT.load_json(root / PRETRACT_SUMMARY.relative_to(ROOT))
    pretract_gate = PRETRACT.load_json(
        root / PRETRACT_GATE.relative_to(ROOT)
    )
    validation = PRETRACT.load_json(PHASE_VALIDATION)
    manifest = PRETRACT.load_json(PHASE_MANIFEST)
    if (
        audit.get("status") != "PASS"
        or audit.get("audited_n") != EXPECTED_CORRECTED_N
        or pretract.get("status") != "PASS"
        or pretract.get("target_n") != EXPECTED_SCALEUP_N
        or pretract.get("status_counts", {}).get("PASS_PRETRACT")
        != EXPECTED_SCALEUP_N
        or validation.get("status") != "PASS"
        or validation.get("diagnosis_labels_used") is not False
        or manifest.get("record_type")
        != "diagnosis_blind_hcp379_phase_b_recipe_stability_manifest"
        or manifest.get("diagnosis_labels_used") is not False
    ):
        raise ValueError("HCP379 tractography scale-up gate is not PASS")
    if validation.get("manifest") != PRETRACT.file_record(PHASE_MANIFEST):
        raise ValueError("Phase-B validation/manifest binding differs")
    selected = validation.get(
        "smallest_density_qualified_scale_up_streamline_count"
    )
    if selected not in ALLOWED_COUNTS:
        raise ValueError(f"selected streamline count is invalid: {selected}")
    canary_units = {
        str(record.get("unit")) for record in manifest.get("units", [])
    }
    if len(canary_units) != 15:
        raise ValueError("Phase-B manifest does not contain 15 units")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    tractography = config["tractography"]
    recipe_id = str(config["contract"]["recipe_id"])
    required_recipe = {
        "algorithm": "iFOD2",
        "act_enabled": True,
        "backtrack": True,
        "crop_at_gmwmi": True,
    }
    for key, expected in required_recipe.items():
        if tractography.get(key) != expected:
            raise ValueError(f"locked tractography recipe differs: {key}")
    human = pretract_gate.get("human_visual_qc", {})
    human_path = Path(str(human.get("path", "")))
    if human != PRETRACT.file_record(human_path):
        raise ValueError("pretract human-QC binding differs")
    return {
        "selected_count": int(selected),
        "selected_run_id": f"primary_{int(selected / 1_000_000)}m",
        "canary_units": sorted(canary_units),
        "validation": validation,
        "manifest": manifest,
        "tractography": tractography,
        "recipe_id": recipe_id,
        "records": {
            "phase_validation": PRETRACT.file_record(PHASE_VALIDATION),
            "phase_manifest": PRETRACT.file_record(PHASE_MANIFEST),
            "pretract_summary": PRETRACT.file_record(
                root / PRETRACT_SUMMARY.relative_to(ROOT)
            ),
            "pretract_gate": PRETRACT.file_record(
                root / PRETRACT_GATE.relative_to(ROOT)
            ),
            "input_audit": PRETRACT.file_record(AUDIT_CSV),
            "input_audit_summary": PRETRACT.file_record(AUDIT_SUMMARY),
            "config": PRETRACT.file_record(CONFIG),
        },
    }


def input_rows(gate: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = PRETRACT.read_csv(AUDIT_CSV)
    canary = set(str(unit) for unit in gate["canary_units"])
    audit_units = {row["unit"] for row in rows}
    overlap = audit_units & canary
    targets = [row for row in rows if row["unit"] not in canary]
    if (
        len(rows) != EXPECTED_CORRECTED_N
        or len(audit_units) != EXPECTED_CORRECTED_N
        or len(overlap) != EXPECTED_CANARY_OVERLAP_N
        or len(targets) != EXPECTED_SCALEUP_N
    ):
        raise ValueError(
            "corrected-track/canary identity sets differ: "
            f"audit={len(rows)} overlap={len(overlap)} "
            f"targets={len(targets)}"
        )
    for row in targets:
        for field in ("dti_source_id", "dti_raw_bundle_sha256"):
            value = row.get(field, "")
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{row['unit']}:{field} differs")
    return sorted(targets, key=lambda row: row["unit"])


def subject_seed(
    row: Mapping[str, str], recipe_id: str
) -> tuple[str, int]:
    token = (
        f"{row['dti_source_id']}|{row['dti_raw_bundle_sha256']}|"
        f"{recipe_id}"
    )
    seed = int.from_bytes(
        hashlib.sha256(token.encode("utf-8")).digest()[:4], "big"
    ) & 0x7FFFFFFF
    return token, seed


def generate_count_invnodevol(
    count_path: Path, volumes_path: Path, output: Path
) -> None:
    count = np.loadtxt(count_path, delimiter=",")
    volumes = node_volumes(volumes_path)
    if count.shape != (EXPECTED_NODES, EXPECTED_NODES):
        raise ValueError("count matrix shape differs")
    matrix = 2.0 * count / (volumes[:, None] + volumes[None, :])
    np.fill_diagonal(matrix, 0.0)
    temporary = partial_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(temporary, matrix, delimiter=",", fmt="%.12g")
    os.replace(temporary, output)


def process_unit(
    row: Mapping[str, str],
    *,
    root: Path,
    gate: Mapping[str, Any],
    nthreads: int,
) -> dict[str, Any]:
    unit = row["unit"]
    paths = tract_paths(root, unit)
    for name in ("tract_state", "tract_lock", "tract_log"):
        paths[name].parent.mkdir(parents=True, exist_ok=True)
    with paths["tract_lock"].open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        prior = (
            PRETRACT.load_json(paths["tract_state"])
            if paths["tract_state"].is_file()
            else None
        )
        if prior and prior.get("status") == "PASS_ALL_NINE":
            artifacts = prior.get("artifacts", {})
            if artifacts and all(
                Path(str(record["path"])).is_file()
                and Path(str(record["path"])).stat().st_size
                == int(record["size_bytes"])
                for record in artifacts.values()
            ):
                return prior
            compaction_path = tractogram_compaction_path(root, unit)
            if compaction_path.is_file():
                validate_tractogram_compaction(
                    PRETRACT.load_json(compaction_path),
                    root=root,
                    unit=unit,
                )
                return prior
        if not prior or prior.get("status") != "PASS_ALL_NINE":
            # Preserve a complete selected-count tractogram when possible,
            # but never trust partial weights, assignments or matrices from a
            # failed/crashed attempt.
            cleanup = [
                paths["tract_metadata"],
                paths["weights"],
                paths["sift2_mu"],
                paths["sift2_iterations"],
                paths["assignments"],
                *paths["matrices"].values(),
                *paths["samples"].values(),
            ]
            cleanup.extend(
                partial_path(path) for path in tuple(cleanup)
            )
            cleanup.append(partial_path(paths["tracks"]))
            for path in cleanup:
                path.unlink(missing_ok=True)
        started = time.monotonic()
        state: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_type": (
                "diagnosis_blind_hcp379_scaleup_tractography_subject"
            ),
            "status": "RUNNING",
            "unit": unit,
            "diagnosis_labels_used": False,
            "started_utc": PRETRACT.utc_now(),
            "selected_streamline_count": gate["selected_count"],
            "density_is_subject_inclusion_gate": False,
            "density_is_whole_cohort_technical_release_gate": True,
            "minimum_edge_density_inclusive": MINIMUM_EDGE_DENSITY,
        }
        PRETRACT.atomic_json(paths["tract_state"], state)
        try:
            check_free_space()
            pretract_state = PRETRACT.load_json(paths["state"])
            atlas_qc = PRETRACT.load_json(paths["atlas_qc"])
            if (
                pretract_state.get("status") != "PASS_PRETRACT"
                or atlas_qc.get("status") != "PASS"
                or atlas_qc.get("labels_found") != EXPECTED_NODES
                or atlas_qc.get("failures")
            ):
                raise ValueError(f"{unit}:pretract/atlas gate differs")
            selected = int(gate["selected_count"])
            recipe = gate["tractography"]
            seed_token, seed = subject_seed(row, str(gate["recipe_id"]))
            env = PRETRACT.environment(nthreads)
            env["MRTRIX_RNG_SEED"] = str(seed)
            step = tractography_spacing(
                root=root,
                unit=unit,
                paths=paths,
                env=env,
            )

            with paths["tract_log"].open(
                "a", encoding="utf-8"
            ) as log:
                if not valid_tck(paths["tracks"], selected, env):
                    paths["tracks"].unlink(missing_ok=True)
                    track_partial = partial_path(paths["tracks"])
                    run_command(
                        [
                            str(MRTRIX / "tckgen"),
                            str(paths["wmfod_norm"]),
                            str(track_partial),
                            "-algorithm",
                            "iFOD2",
                            "-act",
                            str(paths["five_tt"]),
                            "-backtrack",
                            "-crop_at_gmwmi",
                            "-seed_dynamic",
                            str(paths["wmfod_norm"]),
                            "-select",
                            str(selected),
                            "-seeds",
                            str(recipe["maximum_seed_attempts"]),
                            "-minlength",
                            str(recipe["minimum_length_mm"]),
                            "-maxlength",
                            str(recipe["maximum_length_mm"]),
                            "-cutoff",
                            str(recipe["cutoff"]),
                            "-step",
                            f"{step:.8f}",
                            "-angle",
                            str(recipe["maximum_angle_degrees"]),
                            "-nthreads",
                            str(nthreads),
                        ],
                        log=log,
                        env=env,
                        outputs=(track_partial,),
                    )
                    if tck_count(track_partial, env) != selected:
                        raise ValueError(f"{unit}:tractogram count differs")
                    os.replace(track_partial, paths["tracks"])
                tract_metadata = {
                    "schema_version": "1.0.0",
                    "record_type": (
                        "diagnosis_blind_hcp379_scaleup_tractography_parameters"
                    ),
                    "status": "PASS",
                    "unit": unit,
                    "diagnosis_labels_used": False,
                    "algorithm": "iFOD2",
                    "rng_seed_token": seed_token,
                    "rng_seed": seed,
                    "actual_step_size_mm": step,
                    "requested_streamlines": selected,
                    "actual_streamlines": selected,
                    "maximum_seed_attempts": int(
                        recipe["maximum_seed_attempts"]
                    ),
                    "minimum_length_mm": float(recipe["minimum_length_mm"]),
                    "maximum_length_mm": float(recipe["maximum_length_mm"]),
                    "cutoff": float(recipe["cutoff"]),
                    "maximum_angle_degrees": float(
                        recipe["maximum_angle_degrees"]
                    ),
                    "act": True,
                    "backtrack": True,
                    "crop_at_gmwmi": True,
                    "seeding": "seed_dynamic",
                    "recipe_id": gate["recipe_id"],
                }
                PRETRACT.atomic_json(paths["tract_metadata"], tract_metadata)

                if not (
                    paths["weights"].is_file()
                    and nonempty_line_count(paths["weights"]) == selected
                ):
                    weight_partial = partial_path(paths["weights"])
                    mu_partial = partial_path(paths["sift2_mu"])
                    iteration_partial = partial_path(
                        paths["sift2_iterations"]
                    )
                    run_command(
                        [
                            str(MRTRIX / "tcksift2"),
                            str(paths["tracks"]),
                            str(paths["wmfod_norm"]),
                            str(weight_partial),
                            "-act",
                            str(paths["five_tt"]),
                            "-out_mu",
                            str(mu_partial),
                            "-csv",
                            str(iteration_partial),
                            "-reg_tikhonov",
                            "0.0",
                            "-reg_tv",
                            "0.1",
                            "-min_td_frac",
                            "0.1",
                            "-min_iters",
                            "10",
                            "-max_iters",
                            "1000",
                            "-nthreads",
                            str(nthreads),
                        ],
                        log=log,
                        env=env,
                        outputs=(
                            weight_partial,
                            mu_partial,
                            iteration_partial,
                        ),
                    )
                    if nonempty_line_count(weight_partial) != selected:
                        raise ValueError(f"{unit}:SIFT2 weight count differs")
                    os.replace(weight_partial, paths["weights"])
                    os.replace(mu_partial, paths["sift2_mu"])
                    os.replace(
                        iteration_partial, paths["sift2_iterations"]
                    )

                count_partial = partial_path(paths["matrices"]["count"])
                assignments_partial = partial_path(paths["assignments"])
                if not (
                    paths["matrices"]["count"].is_file()
                    and paths["assignments"].is_file()
                ):
                    run_command(
                        [
                            str(MRTRIX / "tck2connectome"),
                            str(paths["tracks"]),
                            str(paths["hcp_nodes"]),
                            str(count_partial),
                            "-assignment_radial_search",
                            "4",
                            "-symmetric",
                            "-zero_diagonal",
                            "-stat_edge",
                            "sum",
                            "-out_assignments",
                            str(assignments_partial),
                            "-nthreads",
                            str(nthreads),
                        ],
                        log=log,
                        env=env,
                        outputs=(count_partial, assignments_partial),
                    )
                    os.replace(
                        count_partial, paths["matrices"]["count"]
                    )
                    os.replace(assignments_partial, paths["assignments"])

                def connectome(
                    output: Path, extra: list[str]
                ) -> None:
                    if output.is_file() and output.stat().st_size > 0:
                        return
                    temporary = partial_path(output)
                    run_command(
                        [
                            str(MRTRIX / "tck2connectome"),
                            str(paths["tracks"]),
                            str(paths["hcp_nodes"]),
                            str(temporary),
                            "-assignment_radial_search",
                            "4",
                            "-symmetric",
                            "-zero_diagonal",
                            *extra,
                            "-nthreads",
                            str(nthreads),
                        ],
                        log=log,
                        env=env,
                        outputs=(temporary,),
                    )
                    os.replace(temporary, output)

                connectome(
                    paths["matrices"]["fd_sum"],
                    [
                        "-stat_edge",
                        "sum",
                        "-tck_weights_in",
                        str(paths["weights"]),
                    ],
                )
                connectome(
                    paths["matrices"]["len_mean"],
                    ["-scale_length", "-stat_edge", "mean"],
                )
                connectome(
                    paths["matrices"]["invlen_mean"],
                    ["-scale_invlength", "-stat_edge", "mean"],
                )
                for metric in ("fa", "md", "rd", "ad"):
                    sample = paths["samples"][metric]
                    matrix = paths["matrices"][f"{metric}_mean"]
                    if not matrix.is_file():
                        sample_partial = partial_path(sample)
                        run_command(
                            [
                                str(MRTRIX / "tcksample"),
                                str(paths["tracks"]),
                                str(paths[metric]),
                                str(sample_partial),
                                "-stat_tck",
                                "mean",
                                "-nthreads",
                                str(nthreads),
                            ],
                            log=log,
                            env=env,
                            outputs=(sample_partial,),
                        )
                        if nonempty_line_count(sample_partial) != selected:
                            raise ValueError(
                                f"{unit}:{metric} sample count differs"
                            )
                        os.replace(sample_partial, sample)
                        connectome(
                            matrix,
                            [
                                "-scale_file",
                                str(sample),
                                "-stat_edge",
                                "mean",
                            ],
                        )
                    sample.unlink(missing_ok=True)
                if not paths["matrices"]["count_invnodevol"].is_file():
                    generate_count_invnodevol(
                        paths["matrices"]["count"],
                        paths["hcp_volumes"],
                        paths["matrices"]["count_invnodevol"],
                    )

            observed_tracks = tck_count(paths["tracks"], env)
            observed_weights = nonempty_line_count(paths["weights"])
            assignment_rows, assigned = assignment_stats(
                paths["assignments"]
            )
            if not (
                observed_tracks
                == observed_weights
                == assignment_rows
                == selected
            ):
                raise ValueError(
                    f"{unit}:tract/weight/assignment counts differ "
                    f"{observed_tracks}/{observed_weights}/"
                    f"{assignment_rows}/{selected}"
                )
            assignment_fraction = assigned / assignment_rows
            if assignment_fraction < MINIMUM_ASSIGNMENT_FRACTION:
                raise ValueError(
                    f"{unit}:assignment_fraction={assignment_fraction:.6f}"
                )
            qc, failures = matrix_qc(
                paths["matrices"],
                volumes=node_volumes(paths["hcp_volumes"]),
                assigned_streamlines=assigned,
            )
            if failures:
                raise ValueError(";".join(failures))
            artifacts = {
                "tractogram": PRETRACT.file_record(paths["tracks"]),
                "sift2_weights": PRETRACT.file_record(paths["weights"]),
                "assignments": PRETRACT.file_record(paths["assignments"]),
                "tractography_metadata": PRETRACT.file_record(
                    paths["tract_metadata"]
                ),
                "atlas_qc": PRETRACT.file_record(paths["atlas_qc"]),
                **{
                    f"matrix_{name}": PRETRACT.file_record(path)
                    for name, path in paths["matrices"].items()
                },
            }
            state.update(
                {
                    "status": "PASS_ALL_NINE",
                    "completed_utc": PRETRACT.utc_now(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "tractogram_streamline_count": observed_tracks,
                    "sift2_weight_count": observed_weights,
                    "assignment_row_count": assignment_rows,
                    "assigned_streamline_count": assigned,
                    "endpoint_assignment_fraction": assignment_fraction,
                    "edge_density": qc["edge_density"],
                    "density_0_60_technical_release_gate_met": (
                        qc["edge_density"] >= 0.60
                    ),
                    "density_0_70_descriptive_target_met": (
                        qc["edge_density"] >= 0.70
                    ),
                    "matrix_qc": qc,
                    "recipe_id": gate["recipe_id"],
                    "phase_b_validation": gate["records"][
                        "phase_validation"
                    ],
                    "artifacts": artifacts,
                    "error": None,
                }
            )
        except Exception as exc:
            state.update(
                {
                    "status": "FAIL_TRACTOGRAPHY",
                    "completed_utc": PRETRACT.utc_now(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
        PRETRACT.atomic_json(paths["tract_state"], state)
        return state


def write_summary(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    state_root = root / "qc/tractography"
    for path in sorted(state_root.glob("*.json")):
        try:
            state = PRETRACT.load_json(path)
        except Exception:
            continue
        rows.append(
            {
                "unit": state.get("unit"),
                "status": state.get("status"),
                "selected_streamline_count": state.get(
                    "selected_streamline_count"
                ),
                "edge_density": state.get("edge_density"),
                "endpoint_assignment_fraction": state.get(
                    "endpoint_assignment_fraction"
                ),
                "connected_nodes": state.get("matrix_qc", {}).get(
                    "connected_nodes"
                ),
                "elapsed_seconds": state.get("elapsed_seconds"),
                "error": state.get("error"),
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["status"])
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_scaleup_tractography_summary"
        ),
        "status": (
            "PASS"
            if len(rows) == EXPECTED_SCALEUP_N
            and counts.get("PASS_ALL_NINE") == EXPECTED_SCALEUP_N
            else "IN_PROGRESS"
        ),
        "updated_utc": PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "target_n": EXPECTED_SCALEUP_N,
        "states_present_n": len(rows),
        "status_counts": dict(sorted(counts.items())),
    }
    PRETRACT.atomic_csv(
        root / "manifests/tractography_subjects.csv", rows
    )
    PRETRACT.atomic_json(
        root / "manifests/tractography_summary.json", summary
    )
    return summary


def verified_artifact_record(
    record: Any, *, label: str
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label}: artifact record is not a mapping")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    observed = PRETRACT.file_record(path)
    if dict(record) != observed:
        raise ValueError(f"{label}: artifact binding differs")
    return observed


def phase_run_release_artifacts(
    run: Mapping[str, Any], *, unit: str
) -> dict[str, dict[str, Any]]:
    matrices = run.get("matrices")
    artifacts = run.get("artifacts")
    if (
        not isinstance(matrices, Mapping)
        or set(matrices) != set(MATRIX_NAMES)
        or not isinstance(artifacts, Mapping)
    ):
        raise ValueError(f"{unit}: selected Phase-B artifact set differs")
    output = {
        str(name): verified_artifact_record(
            record, label=f"{unit}/artifact/{name}"
        )
        for name, record in artifacts.items()
    }
    for name in MATRIX_NAMES:
        output[f"matrix_{name}"] = verified_artifact_record(
            matrices[name], label=f"{unit}/matrix/{name}"
        )
    return output


def verified_scaleup_artifacts(
    artifacts: Any, *, unit: str
) -> dict[str, dict[str, Any]]:
    if not isinstance(artifacts, Mapping):
        raise TypeError(f"{unit}: scale-up artifacts are not a mapping")
    missing = {
        f"matrix_{name}" for name in MATRIX_NAMES
    } - set(artifacts)
    if missing:
        raise ValueError(
            f"{unit}: scale-up matrix artifacts absent: {sorted(missing)}"
        )
    return {
        str(name): verified_artifact_record(
            record, label=f"{unit}/artifact/{name}"
        )
        for name, record in artifacts.items()
    }


def build_corrected_manifest(
    *,
    root: Path,
    gate: Mapping[str, Any],
    audit_rows: list[dict[str, str]],
) -> dict[str, Any]:
    audit_units = {row["unit"] for row in PRETRACT.read_csv(AUDIT_CSV)}
    selected_run = str(gate["selected_run_id"])
    canary_records: dict[str, dict[str, Any]] = {}
    for unit_record in gate["manifest"]["units"]:
        unit = str(unit_record["unit"])
        if unit not in audit_units:
            continue
        candidates = [
            run
            for run in unit_record["runs"]
            if run["run_id"] == selected_run
        ]
        if len(candidates) != 1:
            raise ValueError(f"{unit}:selected canary run differs")
        canary_records[unit] = candidates[0]
    if len(canary_records) != EXPECTED_CANARY_OVERLAP_N:
        raise ValueError("selected canary overlap differs")
    scaleup_records = {}
    for row in audit_rows:
        unit = row["unit"]
        state = PRETRACT.load_json(
            tract_paths(root, unit)["tract_state"]
        )
        if state.get("status") != "PASS_ALL_NINE":
            raise ValueError(f"{unit}:scale-up state is not PASS")
        scaleup_records[unit] = state
    if len(scaleup_records) != EXPECTED_SCALEUP_N:
        raise ValueError("scale-up PASS record set differs")
    all_units = set(canary_records) | set(scaleup_records)
    if all_units != audit_units or len(all_units) != EXPECTED_CORRECTED_N:
        raise ValueError("corrected release identity differs")
    records = []
    for unit in sorted(all_units):
        if unit in canary_records:
            run = canary_records[unit]
            release_artifacts = phase_run_release_artifacts(
                run, unit=unit
            )
            count_matrix = np.loadtxt(
                Path(str(run["matrices"]["count"]["path"])),
                delimiter=",",
            )
            canary_density = float(
                np.count_nonzero(np.triu(count_matrix > 0, 1))
                / (EXPECTED_NODES * (EXPECTED_NODES - 1) / 2)
            )
            if canary_density < MINIMUM_EDGE_DENSITY:
                raise ValueError(
                    f"{unit}:selected Phase-B density "
                    f"{canary_density:.12f}<{MINIMUM_EDGE_DENSITY:.12f}"
                )
            records.append(
                {
                    "unit": unit,
                    "source_lane": "PHASE_B_PRIMARY_REUSE",
                    "selected_streamline_count": gate["selected_count"],
                    "endpoint_assignment_fraction": run[
                        "endpoint_assignment_fraction"
                    ],
                    "edge_density": canary_density,
                    "matrices": run["matrices"],
                    "artifacts": release_artifacts,
                }
            )
        else:
            state = scaleup_records[unit]
            if float(state.get("edge_density", -1.0)) < MINIMUM_EDGE_DENSITY:
                raise ValueError(
                    f"{unit}:scale-up density "
                    f"{state.get('edge_density')}<{MINIMUM_EDGE_DENSITY}"
                )
            compaction_path = tractogram_compaction_path(root, unit)
            compaction = None
            if compaction_path.is_file():
                compaction = validate_tractogram_compaction(
                    PRETRACT.load_json(compaction_path),
                    root=root,
                    unit=unit,
                )
                release_artifacts = verified_scaleup_artifacts(
                    compaction["retained_artifacts"], unit=unit
                )
            else:
                release_artifacts = verified_scaleup_artifacts(
                    state.get("artifacts"), unit=unit
                )
            records.append(
                {
                    "unit": unit,
                    "source_lane": "NON_CANARY_SCALEUP",
                    "selected_streamline_count": gate["selected_count"],
                    "endpoint_assignment_fraction": state[
                        "endpoint_assignment_fraction"
                    ],
                    "edge_density": state["edge_density"],
                    "artifacts": release_artifacts,
                    "tractogram_compaction": (
                        PRETRACT.file_record(compaction_path)
                        if compaction is not None
                        else None
                    ),
                }
            )
    manifest = {
        "schema_version": "1.0.0",
        "record_type": CORRECTED_RELEASE_RECORD_TYPE,
        "status": "PASS",
        "generated_utc": PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "unit_count": len(records),
        "selected_streamline_count": gate["selected_count"],
        "density_is_subject_inclusion_gate": False,
        "density_is_whole_cohort_technical_release_gate": True,
        "minimum_edge_density_inclusive": MINIMUM_EDGE_DENSITY,
        "subjects_at_or_above_minimum_density": len(records),
        "minimum_observed_edge_density": min(
            float(row["edge_density"]) for row in records
        ),
        "phase_b_validation": gate["records"]["phase_validation"],
        "scaleup_summary": PRETRACT.file_record(
            root / "manifests/tractography_summary.json"
        ),
        "units": records,
    }
    PRETRACT.atomic_json(
        root / "manifests" / CORRECTED_RELEASE_FILENAME, manifest
    )
    return manifest


def self_test() -> None:
    count = np.zeros((EXPECTED_NODES, EXPECTED_NODES), dtype=float)
    target_edges = math.ceil(
        MINIMUM_EDGE_DENSITY
        * EXPECTED_NODES
        * (EXPECTED_NODES - 1)
        / 2
    )
    written = 0
    for left in range(EXPECTED_NODES):
        for right in range(left + 1, EXPECTED_NODES):
            if written >= target_edges:
                break
            count[left, right] = count[right, left] = 1.0
            written += 1
        if written >= target_edges:
            break
    volumes = np.full(EXPECTED_NODES, 1000.0)
    inverse = 2.0 * count / (volumes[:, None] + volumes[None, :])
    with tempfile.TemporaryDirectory(prefix="hcp379-tract-selftest.") as tmp:
        root = Path(tmp)
        paths = {}
        for name in MATRIX_NAMES:
            path = root / f"{name}.csv"
            matrix = (
                inverse
                if name == "count_invnodevol"
                else count
            )
            np.savetxt(path, matrix, delimiter=",")
            paths[name] = path
        qc, failures = matrix_qc(
            paths, volumes=volumes, assigned_streamlines=target_edges
        )
        if (
            failures
            or qc["connected_nodes"] != EXPECTED_NODES
            or qc["edge_density"] < MINIMUM_EDGE_DENSITY
        ):
            raise AssertionError((qc, failures))
        tractogram = root / "tracks.tck"
        weights = root / "weights.txt"
        assignments = root / "assignments.csv"
        metadata = root / "tractography_parameters.json"
        for path in (tractogram, weights, assignments, metadata):
            path.write_text(f"{path.name}\n", encoding="utf-8")
        phase_run = {
            "matrices": {
                name: PRETRACT.file_record(path)
                for name, path in paths.items()
            },
            "artifacts": {
                "tractogram": PRETRACT.file_record(tractogram),
                "sift2_weights": PRETRACT.file_record(weights),
                "assignments": PRETRACT.file_record(assignments),
                "tractography_metadata": PRETRACT.file_record(metadata),
            },
        }
        normalized = phase_run_release_artifacts(
            phase_run, unit="synthetic"
        )
        if set(f"matrix_{name}" for name in MATRIX_NAMES) - set(
            normalized
        ):
            raise AssertionError(
                "Phase-B release artifact normalization failed"
            )
        verified = verified_scaleup_artifacts(
            normalized, unit="synthetic"
        )
        if verified != normalized:
            raise AssertionError("scale-up artifact verification drifted")
        scale_root = root / "corrected_scaleup"
        unit = "synthetic"
        scale_paths = tract_paths(scale_root, unit)
        scale_artifacts = {}
        for name, path in (
            ("tractogram", scale_paths["tracks"]),
            ("sift2_weights", scale_paths["weights"]),
            ("assignments", scale_paths["assignments"]),
            (
                "tractography_metadata",
                scale_paths["tract_metadata"],
            ),
            ("atlas_qc", scale_paths["atlas_qc"]),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{name}\n", encoding="utf-8")
            scale_artifacts[name] = PRETRACT.file_record(path)
        for name, path in scale_paths["matrices"].items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{name}\n", encoding="utf-8")
            scale_artifacts[f"matrix_{name}"] = PRETRACT.file_record(
                path
            )
        scale_state = {
            "schema_version": "1.0.0",
            "record_type": (
                "diagnosis_blind_hcp379_scaleup_tractography_subject"
            ),
            "status": "PASS_ALL_NINE",
            "diagnosis_labels_used": False,
            "unit": unit,
            "artifacts": scale_artifacts,
        }
        PRETRACT.atomic_json(
            scale_paths["tract_state"], scale_state
        )
        compacted = compact_validated_tractogram(
            root=scale_root,
            unit=unit,
            state=scale_state,
        )
        if (
            compacted["status"] != "PASS_COMPACTED"
            or scale_paths["tracks"].exists()
            or not scale_paths["matrices"]["count"].is_file()
        ):
            raise AssertionError(compacted)
        tractogram_compaction_path(scale_root, unit).unlink()
        resumed = compact_validated_tractogram(
            root=scale_root,
            unit=unit,
            state=scale_state,
        )
        if resumed["status"] != "PASS_COMPACTED":
            raise AssertionError("tractogram compaction resume failed")
    token, seed = subject_seed(
        {
            "dti_source_id": "a" * 64,
            "dti_raw_bundle_sha256": "b" * 64,
        },
        "recipe",
    )
    if not token or not 0 <= seed <= 0x7FFFFFFF:
        raise AssertionError("seed self-test failed")
    print(
        json.dumps(
            {
                "status": "PASS",
                "record_type": "hcp379_scaleup_tractography_self_test",
            },
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--nthreads", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--compact-tractograms-after-pass",
        action="store_true",
        help=(
            "After full subject-level QC, retain hashes, weights, "
            "assignments, matrices and provenance but remove only the "
            "reproducible selected-count tractogram."
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if (
        not 1 <= args.workers <= 4
        or not 1 <= args.nthreads <= 16
        or args.workers * args.nthreads > (os.cpu_count() or 1)
    ):
        raise ValueError("worker/thread request exceeds the host contract")
    if shutil.disk_usage(args.root).free < MINIMUM_START_FREE_BYTES:
        raise RuntimeError("storage is below the tractography start floor")
    gate = load_gate(args.root)
    targets = input_rows(gate)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        targets = targets[: args.limit]
    for relative in (
        "qc/tractography",
        "qc/tractography_compaction",
        "qc/tractography_compaction_plans",
        "locks/tractography",
        "logs/tractography",
        "manifests",
    ):
        (args.root / relative).mkdir(parents=True, exist_ok=True)
    PRETRACT.atomic_json(
        args.root / "manifests/tractography_gate.json",
        {
            "schema_version": "1.0.0",
            "record_type": (
                "diagnosis_blind_hcp379_scaleup_tractography_gate"
            ),
            "status": "PASS",
            "diagnosis_labels_used": False,
            "selected_streamline_count": gate["selected_count"],
            "selected_run_id": gate["selected_run_id"],
            "compact_tractograms_after_pass": (
                args.compact_tractograms_after_pass
            ),
            "records": gate["records"],
        },
    )
    print(
        f"[{PRETRACT.utc_now()}] HCP379_TRACTOGRAPHY_SCALEUP_START "
        f"n={len(targets)} workers={args.workers} "
        f"nthreads={args.nthreads} selected={gate['selected_count']}",
        flush=True,
    )
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_unit,
                row,
                root=args.root,
                gate=gate,
                nthreads=args.nthreads,
            ): row["unit"]
            for row in targets
        }
        for index, future in enumerate(as_completed(futures), start=1):
            state = future.result()
            if (
                args.compact_tractograms_after_pass
                and state.get("status") == "PASS_ALL_NINE"
            ):
                compacted = compact_validated_tractogram(
                    root=args.root,
                    unit=str(state["unit"]),
                    state=state,
                )
                state = {
                    **state,
                    "tractogram_compaction_status": compacted["status"],
                    "tractogram_reclaimed_bytes": compacted[
                        "reclaimed_bytes"
                    ],
                }
            results.append(state)
            summary = write_summary(args.root)
            print(
                f"[{PRETRACT.utc_now()}] {index}/{len(targets)} "
                f"{state['unit']} {state['status']} "
                f"density={state.get('edge_density')} "
                f"error={state.get('error')} "
                f"cohort_pass="
                f"{summary['status_counts'].get('PASS_ALL_NINE', 0)}",
                flush=True,
            )
    summary = write_summary(args.root)
    if args.limit is not None:
        return int(any(row["status"] != "PASS_ALL_NINE" for row in results))
    if summary["status"] != "PASS":
        return 1
    manifest = build_corrected_manifest(
        root=args.root, gate=gate, audit_rows=input_rows(gate)
    )
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
