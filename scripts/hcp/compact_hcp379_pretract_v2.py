#!/usr/bin/env python3
"""Compact validated corrected pre-tractography intermediates safely.

Only subject-scoped, reproducible intermediates in the new HCP379-v2
``corrected_scaleup`` root are eligible.  Tractography-critical FOD, ACT,
HCP379 atlas, tensor-scalar and QC artifacts are retained and rehashed.  A
write-ahead plan records every candidate before deletion so an interruption
can be resumed without guessing.  Dry-run is the default; ``--execute`` is
required to remove any file.
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
DEFAULT_ROOT = HCP_ROOT / "corrected_scaleup"
SOURCE = EXP / "scripts/hcp/hcp379_scaleup_pretract_v2.py"
MRINFO = Path("/home/ec2-user/mrtrix3/bin/mrinfo")
EXPECTED_N = 216
RETAIN_KEYS = {
    "five_tt",
    "hcp_nodes",
    "hcp_volumes",
    "atlas_qc",
    "atlas_overlay",
    "wmfod_norm",
    "fa",
    "md",
    "rd",
    "ad",
    "automated_qc",
}
META_KEYS = {"subject", "state", "lock", "log"}


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_scaleup_pretract_v2", SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRETRACT = load_module()


def write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if PRETRACT.load_json(path) != dict(payload):
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
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verified_record(record: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label}: record is not a mapping")
    path = Path(str(record.get("path", ""))).resolve()
    observed = PRETRACT.file_record(path)
    if dict(record) != observed:
        raise ValueError(f"{label}: file binding differs")
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


def ensure_scoped_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    if (
        resolved == HCP_ROOT.resolve()
        or not resolved.is_relative_to(HCP_ROOT.resolve())
        or resolved.name != "corrected_scaleup"
    ):
        raise ValueError(f"refusing unsafe compaction root: {resolved}")
    return resolved


def build_plan(
    *,
    root: Path,
    unit: str,
    spacing: list[float],
) -> dict[str, Any]:
    paths = PRETRACT.stage_paths(root, unit)
    subject = Path(paths["subject"]).resolve()
    if subject.parent != (root / "subjects").resolve():
        raise ValueError(f"{unit}:subject root differs")
    state = PRETRACT.load_json(paths["state"])
    automated = PRETRACT.load_json(paths["automated_qc"])
    if (
        state.get("record_type")
        != "diagnosis_blind_hcp379_scaleup_pretract"
        or state.get("status") != "PASS_PRETRACT"
        or state.get("diagnosis_labels_used") is not False
        or state.get("unit") != unit
        or automated.get("status") != "PASS"
        or automated.get("diagnosis_labels_used") is not False
        or automated.get("unit") != unit
    ):
        raise ValueError(f"{unit}:pretract/QC state is not PASS")
    retained = {}
    for name in sorted(RETAIN_KEYS):
        path = Path(paths[name]).resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{unit}:{name}:{path}")
        retained[name] = PRETRACT.file_record(path)

    prunable = {}
    known_paths = set()
    for name, raw_path in paths.items():
        if name in META_KEYS:
            continue
        path = Path(raw_path).resolve()
        known_paths.add(path)
        if name in RETAIN_KEYS or not path.is_file():
            continue
        if path.is_symlink() or not path.is_relative_to(subject):
            raise ValueError(f"{unit}:{name}:unsafe candidate {path}")
        prunable[name] = PRETRACT.file_record(path)
    if "dwi" not in prunable:
        raise ValueError(f"{unit}:DWI is not available for compaction")

    unclassified = {}
    for path in sorted(subject.rglob("*")):
        resolved = path.resolve()
        if (
            not path.is_file()
            or path.is_symlink()
            or resolved in known_paths
        ):
            continue
        unclassified[str(path.relative_to(subject))] = (
            PRETRACT.file_record(path)
        )
    return {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_pretract_compaction_plan",
        "status": "READY",
        "generated_utc": PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "unit": unit,
        "root": str(root),
        "subject_root": str(subject),
        "source_pretract_state": PRETRACT.file_record(paths["state"]),
        "dwi_spacing_mm": spacing,
        "retained_artifacts": retained,
        "prunable_artifacts": prunable,
        "unclassified_retained_artifacts": unclassified,
        "reclaimable_bytes": sum(
            int(record["size_bytes"]) for record in prunable.values()
        ),
        "unclassified_retained_bytes": sum(
            int(record["size_bytes"])
            for record in unclassified.values()
        ),
        "execute_requested": False,
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
        or Path(str(record.get("root", ""))).resolve() != root
    ):
        raise ValueError(f"{unit}:completed compaction identity differs")
    retained = record.get("retained_artifacts")
    deleted = record.get("deleted_artifacts")
    if not isinstance(retained, Mapping) or not isinstance(deleted, Mapping):
        raise TypeError(f"{unit}:completed compaction artifacts differ")
    for name in RETAIN_KEYS:
        verified_record(retained.get(name), label=f"{unit}/retained/{name}")
    for name, raw in deleted.items():
        if (
            not isinstance(raw, Mapping)
            or raw.get("deleted") is not True
            or Path(str(raw.get("path", ""))).exists()
        ):
            raise ValueError(f"{unit}:deleted artifact differs: {name}")
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
    plan_path = root / "qc/pretract_compaction_plans" / f"{unit}.json"
    if plan_path.is_file():
        plan = PRETRACT.load_json(plan_path)
    else:
        paths = PRETRACT.stage_paths(root, unit)
        spacing = (
            spacing_override
            if spacing_override is not None
            else dwi_spacing(paths["dwi"])
        )
        plan = build_plan(root=root, unit=unit, spacing=spacing)
        write_once_json(plan_path, plan)
    if (
        plan.get("record_type")
        != "diagnosis_blind_hcp379_pretract_compaction_plan"
        or plan.get("status") != "READY"
        or plan.get("diagnosis_labels_used") is not False
        or plan.get("unit") != unit
        or Path(str(plan.get("root", ""))).resolve() != root
    ):
        raise ValueError(f"{unit}:compaction plan identity differs")
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
            "reclaimable_bytes": plan["reclaimable_bytes"],
            "plan": str(plan_path),
        }

    deleted = {}
    for name, raw_record in prunable.items():
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"{unit}:{name}:prunable record differs")
        path = Path(str(raw_record.get("path", ""))).resolve()
        if path.exists():
            verified_record(
                raw_record, label=f"{unit}/prunable/{name}"
            )
            path.unlink()
        deleted[name] = {**dict(raw_record), "deleted": True}
    result = {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_pretract_compaction",
        "status": "PASS_COMPACTED",
        "completed_utc": PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "unit": unit,
        "root": str(root),
        "source_pretract_state": plan["source_pretract_state"],
        "plan": PRETRACT.file_record(plan_path),
        "dwi_spacing_mm": plan["dwi_spacing_mm"],
        "retained_artifacts": {
            name: verified_record(
                record, label=f"{unit}/retained/{name}"
            )
            for name, record in retained.items()
        },
        "deleted_artifacts": deleted,
        "unclassified_retained_artifacts": plan[
            "unclassified_retained_artifacts"
        ],
        "reclaimed_bytes": sum(
            int(record["size_bytes"]) for record in prunable.values()
        ),
        "recoverability": (
            "All deleted files are reproducible from the audited Eddy/T1 "
            "sources and frozen commands; no raw, legacy or matrix artifact "
            "is deleted."
        ),
    }
    write_once_json(result_path, result)
    return validate_completed(result, root=root, unit=unit)


def self_test() -> None:
    with tempfile.TemporaryDirectory(
        prefix="hcp379-pretract-compaction."
    ) as temporary:
        root = Path(temporary) / "corrected_scaleup"
        unit = "synthetic"
        paths = PRETRACT.stage_paths(root, unit)
        for name, path in paths.items():
            if name in META_KEYS:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{name}\n", encoding="utf-8")
        PRETRACT.atomic_json(
            paths["state"],
            {
                "schema_version": "1.0.0",
                "record_type": (
                    "diagnosis_blind_hcp379_scaleup_pretract"
                ),
                "status": "PASS_PRETRACT",
                "diagnosis_labels_used": False,
                "unit": unit,
            },
        )
        PRETRACT.atomic_json(
            paths["automated_qc"],
            {
                "schema_version": "1.0.0",
                "status": "PASS",
                "diagnosis_labels_used": False,
                "unit": unit,
            },
        )
        root = root.resolve()
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
        if (
            result["status"] != "PASS_COMPACTED"
            or Path(paths["dwi"]).exists()
            or not Path(paths["wmfod_norm"]).is_file()
        ):
            raise AssertionError(result)
    print(
        json.dumps(
            {
                "record_type": (
                    "hcp379_pretract_compaction_self_test"
                ),
                "status": "PASS",
            },
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    root = ensure_scoped_root(args.root)
    summary = PRETRACT.load_json(
        root / "manifests/pretract_summary.json"
    )
    if (
        summary.get("status") != "PASS"
        or summary.get("target_n") != EXPECTED_N
        or summary.get("status_counts", {}).get("PASS_PRETRACT")
        != EXPECTED_N
    ):
        raise ValueError("216-subject pretract summary is not PASS")
    units = sorted(
        path.stem for path in (root / "qc/subjects").glob("*.json")
    )
    if len(units) != EXPECTED_N or len(set(units)) != EXPECTED_N:
        raise ValueError("pretract state identity set differs")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        units = units[: args.limit]
    results = [
        compact_unit(root=root, unit=unit, execute=args.execute)
        for unit in units
    ]
    output = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_pretract_compaction_summary"
        ),
        "status": (
            "PASS_COMPACTED"
            if args.execute
            and all(
                result.get("status") == "PASS_COMPACTED"
                for result in results
            )
            else "DRY_RUN_READY"
        ),
        "generated_utc": PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "execute_requested": args.execute,
        "unit_count": len(results),
        "reclaimable_or_reclaimed_bytes": sum(
            int(
                result.get(
                    "reclaimed_bytes",
                    result.get("reclaimable_bytes", 0),
                )
            )
            for result in results
        ),
        "results": results,
    }
    suffix = "result" if args.execute else "dryrun"
    PRETRACT.atomic_json(
        root / f"manifests/pretract_compaction_{suffix}.json",
        output,
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "unit_count": output["unit_count"],
                "bytes": output["reclaimable_or_reclaimed_bytes"],
                "execute_requested": args.execute,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
