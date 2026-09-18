#!/home/ec2-user/fsl/bin/python
"""Compact only reproducible Recovery4 pretract working images.

The compactor is deliberately conservative:

* a subject must already have an immutable, diagnosis-blind
  ``PASS_PRETRACT_RECOVERY4`` state;
* every retained and deleted file is SHA-256 bound in a write-ahead plan;
* the selected 5TT/GMWMI, HCP379 atlas, WM FOD, raw and bounded tensor maps,
  transforms, overlays and QC are retained;
* only explicitly classified subject-local imaging intermediates are removed;
* unclassified files are retained;
* dry-run is the default and ``--execute`` is required to unlink anything.

The result uses the mature pretract-compaction record type consumed by the
tractography engine, with additional Recovery4 identity and provenance fields.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
SOURCE = EXP / "scripts/hcp/hcp379_scaleup_pretract_recovery4_v3.py"
MRINFO = Path("/home/ec2-user/mrtrix3/bin/mrinfo")
CENTRAL_ROOT = HCP_ROOT / "corrected_scaleup_recovery4"
ARCHIVE_ROOT = HCP_ROOT / "archive_corrected_calibration_recovery4"
ROOT_CONTRACTS = {
    CENTRAL_ROOT.resolve(): 216,
    ARCHIVE_ROOT.resolve(): 6,
}

# These final scientific/QC artifacts must survive compaction.  The raw tensor
# maps are intentionally retained even though tractography samples the bounded
# copies.
RETAIN_STATE_ARTIFACTS = {
    "selected_five_tt",
    "selected_gmwmi",
    "selected_hcp379_nodes",
    "selected_hcp379_node_volumes",
    "selected_hcp379_atlas_qc",
    "selected_hcp379_overlay",
    "selected_t1_in_b0",
    "selected_t1_overlay",
    "selected_five_tt_overlay",
    "selected_transform",
    "spatial_route_qc",
    "wmfod_normalised",
    "scalar_recovery4_qc",
    "wmfod_l0",
    "fa_raw",
    "md_raw",
    "rd_raw",
    "ad_raw",
    "fa_bounded",
    "md_bounded",
    "rd_bounded",
    "ad_bounded",
}

# Only paths produced by spatial_paths() and named here can be deleted.  A
# dynamically selected final artifact is removed from this set by path, not by
# key, so the same policy works for the primary BBR and ANTs failover routes.
PRUNABLE_PATH_KEYS = {
    "dwi",
    "mask",
    "dwi_mask_nifti",
    "eddy_gradfix",
    "b0",
    "b0_mif",
    "b0_series",
    "t1",
    "t1_bias",
    "t1_brain",
    "t1_mask",
    "b0_to_t1_image",
    "five_tt_t1",
    "five_tt",
    "gmwmi",
    "wmseg",
    "fod_dwi",
    "wmfod",
    "wmfod_l0",
    "gm",
    "csf",
    "gm_norm",
    "csf_norm",
    "tensor",
    "hcp_t1_raw",
    "hcp_t1_nodes",
    "hcp_b0_raw",
    "b0_1mm",
    "mask_1mm",
    "atlas_native",
    "ants_t1_mask_sum",
    "ants_t1_mask",
    "ants_t1_brain",
    "ants_warped_t1",
    "ants_inverse_warped_b0",
    "ants_five_tt_t1_nifti",
    "ants_five_tt_raw_nifti",
    "ants_five_tt",
    "ants_gmwmi",
    "ants_five_tt_sum",
    "ants_five_tt_mask",
    "ants_hcp_b0_raw",
    "ants_hcp_nodes",
    "ants_atlas_native",
    "bbr_five_tt_sum",
    "bbr_five_tt_mask",
}

LEGACY_TRACT_ALIASES = {
    "five_tt": "selected_five_tt",
    "wmfod_norm": "wmfod_normalised",
    "hcp_nodes": "selected_hcp379_nodes",
    "hcp_volumes": "selected_hcp379_node_volumes",
    "fa": "fa_bounded",
    "md": "md_bounded",
    "rd": "rd_bounded",
    "ad": "ad_bounded",
    "atlas_qc": "selected_hcp379_atlas_qc",
    "automated_qc": "scalar_recovery4_qc",
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRETRACT = load_module(
    SOURCE, "hcp379_pretract_recovery4_for_compaction"
)


def write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create an immutable JSON record without replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    if path.exists():
        if PRETRACT.load_json(path) != value:
            raise FileExistsError(f"immutable record differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_scoped_root(root: Path) -> tuple[Path, int]:
    resolved = root.expanduser().resolve()
    expected = ROOT_CONTRACTS.get(resolved)
    if expected is None:
        raise ValueError(f"refusing unsafe Recovery4 compaction root: {resolved}")
    return resolved, expected


def verified_record(record: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label}: artifact record is not a mapping")
    path = Path(str(record.get("path", ""))).resolve()
    observed = PRETRACT.file_record(path)
    if observed != dict(record):
        raise ValueError(f"{label}: artifact binding differs")
    return observed


def dwi_spacing(path: Path) -> list[float]:
    result = PRETRACT.subprocess.run(
        [str(MRINFO), str(path), "-spacing"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [float(value) for value in result.stdout.split()[:3]]
    if (
        len(values) != 3
        or not all(math.isfinite(value) and value > 0 for value in values)
    ):
        raise ValueError(f"invalid DWI spacing: {values}")
    return values


def pass_state(root: Path, unit: str) -> tuple[Path, dict[str, Any]]:
    path = PRETRACT.spatial_paths(root, unit)["state"]
    state = PRETRACT.load_json(path)
    if (
        state.get("record_type")
        != "diagnosis_blind_hcp379_scaleup_pretract_recovery4"
        or state.get("status") != "PASS_PRETRACT_RECOVERY4"
        or state.get("diagnosis_labels_used") is not False
        or state.get("unit") != unit
        or state.get("raw_tensor_maps_preserved") is not True
        or state.get("bounded_tensor_maps_separate") is not True
    ):
        raise ValueError(f"{unit}:Recovery4 pretract state is not compactable")
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TypeError(f"{unit}:Recovery4 artifacts are not a mapping")
    missing = RETAIN_STATE_ARTIFACTS - set(artifacts)
    if missing:
        raise ValueError(f"{unit}:required retained artifacts absent: {sorted(missing)}")
    return path, state


def retained_records(
    state: Mapping[str, Any], *, unit: str
) -> dict[str, dict[str, Any]]:
    artifacts = state["artifacts"]
    retained = {
        name: verified_record(
            artifacts[name], label=f"{unit}/retained/{name}"
        )
        for name in sorted(RETAIN_STATE_ARTIFACTS)
    }
    for alias, source in LEGACY_TRACT_ALIASES.items():
        retained[alias] = dict(retained[source])
    return retained


def unique_bytes(records: Mapping[str, Mapping[str, Any]]) -> int:
    observed: dict[str, int] = {}
    for record in records.values():
        path = str(Path(str(record["path"])).resolve())
        size = int(record["size_bytes"])
        if path in observed and observed[path] != size:
            raise ValueError(f"duplicate artifact size differs: {path}")
        observed[path] = size
    return sum(observed.values())


def build_plan(
    *,
    root: Path,
    unit: str,
    spacing: list[float],
) -> dict[str, Any]:
    state_path, state = pass_state(root, unit)
    paths = PRETRACT.spatial_paths(root, unit)
    subject = Path(paths["subject"]).resolve()
    if subject.parent != (root / "subjects").resolve():
        raise ValueError(f"{unit}:subject root differs")

    retained = retained_records(state, unit=unit)
    retained_paths = {
        Path(str(record["path"])).resolve() for record in retained.values()
    }
    prunable: dict[str, dict[str, Any]] = {}
    known_paths: set[Path] = set()
    path_owner: dict[Path, str] = {}
    for name in sorted(paths):
        if name in {
            "subject",
            "state",
            "lock",
            "recovery4_lock",
            "log",
            "ants_log",
            "ants_dir",
            "ants_prefix",
        }:
            continue
        path = Path(paths[name]).resolve()
        known_paths.add(path)
        if (
            name not in PRUNABLE_PATH_KEYS
            or path in retained_paths
            or not path.is_file()
        ):
            continue
        if path.is_symlink() or not path.is_relative_to(subject):
            raise ValueError(f"{unit}:{name}:unsafe candidate {path}")
        owner = path_owner.get(path)
        if owner is not None:
            continue
        path_owner[path] = name
        prunable[name] = PRETRACT.file_record(path)

    if "dwi" not in prunable:
        raise ValueError(f"{unit}:DWI is not available for compaction")
    for name in ("fa_raw", "md_raw", "rd_raw", "ad_raw"):
        if Path(str(retained[name]["path"])).resolve() in {
            Path(str(value["path"])).resolve() for value in prunable.values()
        }:
            raise ValueError(f"{unit}:raw tensor map classified as prunable")

    # Unclassified files are explicitly recorded and retained.  This makes the
    # default safe if a future Recovery4 revision adds a new intermediate.
    unclassified: dict[str, dict[str, Any]] = {}
    for candidate in sorted(subject.rglob("*")):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        resolved = candidate.resolve()
        if resolved in known_paths or resolved in retained_paths:
            continue
        unclassified[str(candidate.relative_to(subject))] = (
            PRETRACT.file_record(candidate)
        )

    source_artifacts = state["artifacts"]
    deleted_state_artifacts = {
        name: dict(record)
        for name, record in source_artifacts.items()
        if isinstance(record, Mapping)
        and Path(str(record.get("path", ""))).resolve()
        in {
            Path(str(value["path"])).resolve()
            for value in prunable.values()
        }
    }
    return {
        "schema_version": "2.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_pretract_compaction_recovery4_plan"
        ),
        "status": "READY",
        "generated_utc": PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "unit": unit,
        "root": str(root),
        "subject_root": str(subject),
        "source_pretract_state": PRETRACT.file_record(state_path),
        "source_pretract_record_type": state["record_type"],
        "source_pretract_status": state["status"],
        "registration_route": state["registration_route"],
        "raw_tensor_maps_preserved": True,
        "bounded_tensor_maps_separate": True,
        "dwi_spacing_mm": spacing,
        "retained_artifacts": retained,
        "prunable_artifacts": prunable,
        "deleted_state_artifacts_if_executed": deleted_state_artifacts,
        "unclassified_retained_artifacts": unclassified,
        "retained_unique_bytes": unique_bytes(retained),
        "reclaimable_bytes": unique_bytes(prunable),
        "unclassified_retained_bytes": unique_bytes(unclassified),
        "execute_requested_by_plan": False,
    }


def validate_completed(
    record: Mapping[str, Any], *, root: Path, unit: str
) -> dict[str, Any]:
    if (
        record.get("record_type")
        != "diagnosis_blind_hcp379_pretract_compaction"
        or record.get("status") != "PASS_COMPACTED"
        or record.get("diagnosis_labels_used") is not False
        or record.get("unit") != unit
        or Path(str(record.get("root", ""))).resolve() != root.resolve()
        or record.get("recovery4_inputs") is not True
        or record.get("source_pretract_record_type")
        != "diagnosis_blind_hcp379_scaleup_pretract_recovery4"
        or record.get("raw_tensor_maps_preserved") is not True
    ):
        raise ValueError(f"{unit}:Recovery4 compaction identity differs")
    verified_record(
        record.get("source_pretract_state"),
        label=f"{unit}/source_pretract_state",
    )
    retained = record.get("retained_artifacts")
    deleted = record.get("deleted_artifacts")
    if not isinstance(retained, Mapping) or not isinstance(deleted, Mapping):
        raise TypeError(f"{unit}:compaction artifacts differ")
    required = RETAIN_STATE_ARTIFACTS | set(LEGACY_TRACT_ALIASES)
    if not required.issubset(retained):
        raise ValueError(f"{unit}:retained compaction contract is incomplete")
    for name in required:
        verified_record(retained[name], label=f"{unit}/retained/{name}")
    dwi = deleted.get("dwi")
    if (
        not isinstance(dwi, Mapping)
        or dwi.get("deleted") is not True
        or Path(str(dwi.get("path", ""))).exists()
    ):
        raise ValueError(f"{unit}:compacted DWI deletion differs")
    for name, raw in deleted.items():
        if (
            not isinstance(raw, Mapping)
            or raw.get("deleted") is not True
            or Path(str(raw.get("path", ""))).exists()
        ):
            raise ValueError(f"{unit}:deleted artifact differs: {name}")
    spacing = record.get("dwi_spacing_mm")
    if (
        not isinstance(spacing, list)
        or len(spacing) != 3
        or not all(
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > 0
            for value in spacing
        )
    ):
        raise ValueError(f"{unit}:stored DWI spacing differs")
    return dict(record)


def compact_unit(
    *,
    root: Path,
    unit: str,
    execute: bool,
    spacing_override: list[float] | None = None,
) -> dict[str, Any]:
    result_path = root / "qc/pretract_compaction" / f"{unit}.json"
    if result_path.is_file():
        return validate_completed(
            PRETRACT.load_json(result_path), root=root, unit=unit
        )

    plan_path = (
        root / "qc/pretract_compaction_plans" / f"{unit}.json"
    )
    if plan_path.is_file():
        plan = PRETRACT.load_json(plan_path)
    else:
        paths = PRETRACT.spatial_paths(root, unit)
        spacing = (
            list(spacing_override)
            if spacing_override is not None
            else dwi_spacing(paths["dwi"])
        )
        plan = build_plan(root=root, unit=unit, spacing=spacing)
        write_once_json(plan_path, plan)
    if (
        plan.get("record_type")
        != "diagnosis_blind_hcp379_pretract_compaction_recovery4_plan"
        or plan.get("status") != "READY"
        or plan.get("diagnosis_labels_used") is not False
        or plan.get("unit") != unit
        or Path(str(plan.get("root", ""))).resolve() != root.resolve()
        or plan.get("raw_tensor_maps_preserved") is not True
    ):
        raise ValueError(f"{unit}:Recovery4 compaction plan differs")
    verified_record(
        plan.get("source_pretract_state"),
        label=f"{unit}/source_pretract_state",
    )
    retained = plan.get("retained_artifacts")
    prunable = plan.get("prunable_artifacts")
    if not isinstance(retained, Mapping) or not isinstance(prunable, Mapping):
        raise TypeError(f"{unit}:compaction plan artifacts differ")
    for name, record in retained.items():
        verified_record(record, label=f"{unit}/retained/{name}")
    if not execute:
        return {
            "unit": unit,
            "status": "DRY_RUN_READY",
            "reclaimable_bytes": int(plan["reclaimable_bytes"]),
            "retained_unique_bytes": int(plan["retained_unique_bytes"]),
            "plan": str(plan_path),
        }

    deleted: dict[str, dict[str, Any]] = {}
    for name, raw in prunable.items():
        if not isinstance(raw, Mapping):
            raise TypeError(f"{unit}:{name}:prunable record differs")
        path = Path(str(raw.get("path", ""))).resolve()
        subject = Path(str(plan["subject_root"])).resolve()
        if (
            path.is_symlink()
            or not path.is_relative_to(subject)
            or subject.parent != (root / "subjects").resolve()
        ):
            raise ValueError(f"{unit}:{name}:unsafe deletion target {path}")
        if path.exists():
            verified_record(raw, label=f"{unit}/prunable/{name}")
            path.unlink()
        deleted[name] = {**dict(raw), "deleted": True}

    result = {
        "schema_version": "2.0.0",
        # Kept for exact compatibility with tractography_spacing().
        "record_type": "diagnosis_blind_hcp379_pretract_compaction",
        "status": "PASS_COMPACTED",
        "completed_utc": PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "unit": unit,
        "root": str(root.resolve()),
        "recovery4_inputs": True,
        "source_pretract_record_type": plan[
            "source_pretract_record_type"
        ],
        "source_pretract_status": plan["source_pretract_status"],
        "source_pretract_state": plan["source_pretract_state"],
        "registration_route": plan["registration_route"],
        "raw_tensor_maps_preserved": True,
        "bounded_tensor_maps_separate": True,
        "plan": PRETRACT.file_record(plan_path),
        "dwi_spacing_mm": plan["dwi_spacing_mm"],
        "retained_artifacts": {
            name: verified_record(
                record, label=f"{unit}/retained/{name}"
            )
            for name, record in retained.items()
        },
        "deleted_artifacts": deleted,
        "deleted_state_artifacts": plan[
            "deleted_state_artifacts_if_executed"
        ],
        "unclassified_retained_artifacts": plan[
            "unclassified_retained_artifacts"
        ],
        "reclaimed_bytes": unique_bytes(prunable),
        "recoverability": (
            "Only subject-local Recovery4 imaging intermediates classified "
            "in the immutable write-ahead plan were deleted. Selected 5TT/"
            "GMWMI, HCP379 nodes and volumes, normalized WM FOD, raw and "
            "bounded tensor maps, transforms, overlays, QC, source identity "
            "and frozen commands remain hash-bound."
        ),
    }
    write_once_json(result_path, result)
    return validate_completed(result, root=root, unit=unit)


def summary_units(root: Path, expected_n: int) -> list[str]:
    summary_path = root / "manifests/pretract_recovery4_summary.json"
    summary = PRETRACT.load_json(summary_path)
    if (
        summary.get("record_type")
        != "diagnosis_blind_hcp379_scaleup_pretract_recovery4_summary"
        or summary.get("status") != "PASS"
        or summary.get("diagnosis_labels_used") is not False
        or summary.get("target_n") != expected_n
        or summary.get("states_present_n") != expected_n
        or summary.get("status_counts", {}).get(
            "PASS_PRETRACT_RECOVERY4"
        )
        != expected_n
    ):
        raise ValueError("Recovery4 pretract summary is not exact PASS")
    units = sorted(
        path.stem for path in (root / "qc/subjects").glob("*.json")
    )
    if len(units) != expected_n or len(set(units)) != expected_n:
        raise ValueError("Recovery4 pretract identity set differs")
    return units


def _synthetic_state(root: Path, unit: str) -> dict[str, Any]:
    paths = PRETRACT.spatial_paths(root, unit)
    artifact_paths = {
        "dwi_bias_corrected": paths["dwi"],
        "mean_b0": paths["b0"],
        "dwi_mask": paths["mask"],
        "t1_n4": paths["t1"],
        "selected_five_tt": paths["five_tt"],
        "selected_gmwmi": paths["gmwmi"],
        "selected_hcp379_nodes": paths["hcp_nodes"],
        "selected_hcp379_node_volumes": paths["hcp_volumes"],
        "selected_hcp379_atlas_qc": paths["atlas_qc"],
        "selected_hcp379_overlay": paths["atlas_overlay"],
        "selected_t1_in_b0": paths["bbr_t1_in_b0"],
        "selected_t1_overlay": paths["bbr_t1_overlay"],
        "selected_five_tt_overlay": paths["bbr_five_tt_overlay"],
        "selected_transform": paths["t1_to_b0"],
        "spatial_route_qc": paths["route_qc"],
        "wmfod_normalised": paths["wmfod_norm"],
        "gm_normalised": paths["gm_norm"],
        "csf_normalised": paths["csf_norm"],
        "scalar_recovery4_qc": paths["automated_qc"],
        "wmfod_l0": paths["wmfod_l0"],
        "fa_raw": paths["fa"],
        "md_raw": paths["md"],
        "rd_raw": paths["rd"],
        "ad_raw": paths["ad"],
    }
    for metric in ("fa", "md", "rd", "ad"):
        artifact_paths[f"{metric}_bounded"] = (
            paths["subject"] / "05_model" / f"{metric}_bounded_recovery4.mif"
        )
    for name, path in paths.items():
        if name in {
            "subject",
            "state",
            "lock",
            "recovery4_lock",
            "log",
            "ants_log",
            "ants_dir",
            "ants_prefix",
        }:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"stage:{name}\n", encoding="utf-8")
    for name, path in artifact_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"artifact:{name}\n", encoding="utf-8")
    artifacts = {
        name: PRETRACT.file_record(path)
        for name, path in artifact_paths.items()
    }
    state = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_scaleup_pretract_recovery4"
        ),
        "status": "PASS_PRETRACT_RECOVERY4",
        "unit": unit,
        "diagnosis_labels_used": False,
        "registration_route": "fsl_bbr_primary",
        "raw_tensor_maps_preserved": True,
        "bounded_tensor_maps_separate": True,
        "artifacts": artifacts,
    }
    PRETRACT.atomic_json(paths["state"], state)
    return state


def self_test() -> None:
    with tempfile.TemporaryDirectory(
        prefix="hcp379-pretract-recovery4-compaction."
    ) as temporary:
        root = Path(temporary) / "corrected_scaleup_recovery4"
        unit = "synthetic"
        state = _synthetic_state(root, unit)
        dry = compact_unit(
            root=root,
            unit=unit,
            execute=False,
            spacing_override=[2.0, 2.0, 2.0],
        )
        if dry["status"] != "DRY_RUN_READY" or dry["reclaimable_bytes"] <= 0:
            raise AssertionError(dry)
        result = compact_unit(
            root=root,
            unit=unit,
            execute=True,
            spacing_override=[2.0, 2.0, 2.0],
        )
        paths = PRETRACT.spatial_paths(root, unit)
        if (
            result["status"] != "PASS_COMPACTED"
            or paths["dwi"].exists()
            or not paths["wmfod_norm"].is_file()
            or not paths["fa"].is_file()
            or PRETRACT.existing_final_pass(
                paths["state"], root=root, unit=unit
            )
            is None
            or result["retained_artifacts"]["fa_raw"]
            != state["artifacts"]["fa_raw"]
        ):
            raise AssertionError(result)
    print(
        json.dumps(
            {
                "record_type": (
                    "hcp379_pretract_compaction_recovery4_self_test"
                ),
                "status": "PASS",
            },
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CENTRAL_ROOT)
    parser.add_argument("--unit")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    root, expected_n = ensure_scoped_root(args.root)
    if args.unit:
        units = [args.unit]
        # Per-subject streaming compaction is allowed only after that subject's
        # own PASS state; it intentionally does not require the cohort summary.
        pass_state(root, args.unit)
    else:
        units = summary_units(root, expected_n)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        units = units[: args.limit]
    results = [
        compact_unit(root=root, unit=unit, execute=args.execute)
        for unit in units
    ]
    status = (
        "PASS_COMPACTED"
        if args.execute
        and all(row.get("status") == "PASS_COMPACTED" for row in results)
        else "DRY_RUN_READY"
    )
    output = {
        "schema_version": "2.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_pretract_compaction_recovery4_summary"
        ),
        "status": status,
        "generated_utc": PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "execute_requested": args.execute,
        "streaming_unit_mode": bool(args.unit),
        "root": str(root),
        "unit_count": len(results),
        "reclaimable_or_reclaimed_bytes": sum(
            int(
                row.get(
                    "reclaimed_bytes",
                    row.get("reclaimable_bytes", 0),
                )
            )
            for row in results
        ),
        "results": results,
    }
    suffix = "result" if args.execute else "dryrun"
    output_path = (
        root
        / "manifests"
        / (
            f"pretract_compaction_recovery4_{args.unit}_{suffix}.json"
            if args.unit
            else f"pretract_compaction_recovery4_{suffix}.json"
        )
    )
    PRETRACT.atomic_json(output_path, output)
    print(
        json.dumps(
            {
                "status": status,
                "unit_count": len(results),
                "bytes": output["reclaimable_or_reclaimed_bytes"],
                "execute_requested": args.execute,
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
