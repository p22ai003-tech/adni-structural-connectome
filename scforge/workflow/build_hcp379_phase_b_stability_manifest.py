#!/usr/bin/env python3
"""Assemble immutable diagnosis-blind HCP379 Phase-B run records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


RUN_IDS = {
    "primary_3m",
    "primary_5m",
    "primary_10m",
    "independent_3m",
    "independent_5m",
    "independent_10m",
}
MATRIX_NAMES = {
    "count",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
}
ARTIFACT_NAMES = {
    "tractogram",
    "sift2_weights",
    "assignments",
    "tractography_metadata",
}
ATLAS_ARTIFACT_NAMES = {
    "source_parcellation",
    "corrected_nodes",
    "node_volumes",
    "source_node_volumes",
    "atlas_qc",
    "visual_overlay",
    "atlas_result",
    "corrected_atlas_gate",
    "selected_t1_to_b0_transform_artifact",
    "corrected_b0_reference",
}
FORBIDDEN_KEYS = {
    "diagnosis",
    "diagnosis_at_dti",
    "group",
    "research_group",
    "outcome",
    "clinical_outcome",
    "disease_status",
}


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
            if normalized in FORBIDDEN_KEYS:
                failures.append(child_path)
            failures.extend(forbidden_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(forbidden_paths(child, f"{prefix}[{index}]"))
    return failures


def build_manifest(run_record_paths: list[Path]) -> dict[str, Any]:
    if not run_record_paths:
        raise ValueError("at least one run record is required")
    units: dict[str, dict[str, dict[str, Any]]] = {}
    record_evidence: dict[str, Any] = {}
    atlas_by_unit: dict[str, dict[str, Any]] = {}
    for source_path in run_record_paths:
        source_path = source_path.expanduser().resolve()
        raw = load_json(source_path)
        forbidden = forbidden_paths(raw)
        if forbidden:
            raise ValueError(f"run record contains forbidden fields: {forbidden}")
        if (
            raw.get("record_type")
            != "diagnosis_blind_hcp379_phase_b_stability_run"
            or raw.get("diagnosis_labels_used") is not False
        ):
            raise ValueError(
                f"run record is not a diagnosis-blind HCP379 stability record: "
                f"{source_path}"
            )
        unit = str(raw.get("unit", "")).strip()
        run_id = str(raw.get("run_id", "")).strip()
        if not unit or run_id not in RUN_IDS:
            raise ValueError(f"invalid unit/run_id in {source_path}")
        if run_id in units.setdefault(unit, {}):
            raise ValueError(f"duplicate run record for {unit}/{run_id}")

        matrices_raw = raw.get("matrices")
        artifacts_raw = raw.get("artifacts")
        atlas_raw = raw.get("atlas")
        if not isinstance(matrices_raw, Mapping) or set(matrices_raw) != MATRIX_NAMES:
            raise ValueError(f"matrix path set differs for {unit}/{run_id}")
        if not isinstance(artifacts_raw, Mapping) or set(artifacts_raw) != ARTIFACT_NAMES:
            raise ValueError(f"artifact path set differs for {unit}/{run_id}")
        if not isinstance(atlas_raw, Mapping) or set(atlas_raw) != ATLAS_ARTIFACT_NAMES:
            raise ValueError(f"atlas path set differs for {unit}/{run_id}")

        expected_count = {
            "primary_3m": 3_000_000,
            "primary_5m": 5_000_000,
            "primary_10m": 10_000_000,
            "independent_3m": 3_000_000,
            "independent_5m": 5_000_000,
            "independent_10m": 10_000_000,
        }[run_id]
        try:
            streamline_count = int(raw["streamline_count"])
            tractogram_count = int(raw["tractogram_streamline_count"])
            weight_count = int(raw["sift2_weight_count"])
            assignment_count = int(raw["assignment_row_count"])
            assigned_count = int(raw["assigned_streamline_count"])
            assignment_fraction = float(raw["endpoint_assignment_fraction"])
        except Exception as exc:
            raise ValueError(f"invalid observed counts for {unit}/{run_id}") from exc
        if not (
            streamline_count
            == tractogram_count
            == weight_count
            == assignment_count
            == expected_count
            and 0 <= assigned_count <= assignment_count
        ):
            raise ValueError(
                f"tractogram/weight/assignment counts differ for {unit}/{run_id}"
            )
        if abs(assignment_fraction - assigned_count / assignment_count) > 1.0e-12:
            raise ValueError(
                f"assignment fraction/counts differ for {unit}/{run_id}"
            )
        seed_id = str(raw.get("seed_id", "")).strip()
        seed_class = str(raw.get("seed_class", "")).strip()
        expected_seed_class = (
            "independent" if run_id.startswith("independent_") else "primary"
        )
        if not seed_id or seed_class != expected_seed_class:
            raise ValueError(f"seed identity is invalid for {unit}/{run_id}")

        atlas_records = {
            name: file_record(Path(str(path)))
            for name, path in sorted(atlas_raw.items())
        }
        prior_atlas = atlas_by_unit.setdefault(unit, atlas_records)
        if prior_atlas != atlas_records:
            raise ValueError(f"atlas evidence differs across runs for {unit}")
        run = {
            "run_id": run_id,
            "streamline_count": streamline_count,
            "tractogram_streamline_count": tractogram_count,
            "sift2_weight_count": weight_count,
            "assignment_row_count": assignment_count,
            "assigned_streamline_count": assigned_count,
            "endpoint_assignment_fraction": assignment_fraction,
            "seed_id": seed_id,
            "seed_class": seed_class,
            "matrices": {
                name: file_record(Path(str(path)))
                for name, path in sorted(matrices_raw.items())
            },
            "artifacts": {
                name: file_record(Path(str(path)))
                for name, path in sorted(artifacts_raw.items())
            },
        }
        units[unit][run_id] = run
        record_evidence[f"{unit}/{run_id}"] = file_record(source_path)

    incomplete = {
        unit: sorted(RUN_IDS - set(runs))
        for unit, runs in units.items()
        if set(runs) != RUN_IDS
    }
    if incomplete:
        raise ValueError(f"incomplete six-run unit sets: {incomplete}")
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_phase_b_recipe_stability_manifest"
        ),
        "diagnosis_labels_used": False,
        "builder_source": file_record(Path(__file__)),
        "run_record_evidence": dict(sorted(record_evidence.items())),
        "units": [
            {
                "unit": unit,
                "atlas": atlas_by_unit[unit],
                "runs": [runs[run_id] for run_id in sorted(RUN_IDS)],
            }
            for unit, runs in sorted(units.items())
        ],
    }


def write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {path}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        os.unlink(temporary)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-record", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.run_record)
    write_immutable_json(args.output, manifest)
    print(args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
