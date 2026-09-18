#!/usr/bin/env python3
"""Audit the live HCP379-v2 cohort against the exact 530-subject topology.

This is a read-only audit of imaging and matrix artifacts.  It verifies every
currently usable historical nine-matrix bundle against its recorded hashes,
loads all matrices, checks the 379 x 379 numerical contract and count-matched
edge support, and emits a 530-row integration inventory.  Historical bundles
remain baseline evidence only; this script never promotes them into the final
corrected release and never edits subject outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
ROOT = Path("/data/derivatives/hcp379_v2")
TOPOLOGY = (
    EXP
    / "research_audit/outputs/hcp379_density_execution_topology_v3"
    / "hcp379_density_execution_subjects.csv"
)
FREEZE = ROOT / "manifests/historical_freeze_manifest.csv"
ARCHIVE = ROOT / "manifests/archive_restore_summary.json"
TENSOR = ROOT / "manifests/tensor_repair_summary.json"
CENTRAL_CONNECTOMES = ROOT / "connectomes"
OUTPUT_ROOT = ROOT / "audits/live_cohort_inventory_v1"
OUTPUT_SUMMARY = OUTPUT_ROOT / "hcp379_live_cohort_inventory_summary.json"
OUTPUT_INVENTORY = OUTPUT_ROOT / "hcp379_live_cohort_inventory_530.csv"
OUTPUT_MATRIX_AUDIT = OUTPUT_ROOT / "hcp379_current_matrix_audit.csv"

NODES = 379
POSSIBLE_EDGES = NODES * (NODES - 1) // 2
MINIMUM_DENSITY = 0.60
MATRIX_FAMILIES = (
    "count",
    "fd_sum",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
    "count_invnodevol",
)
MEAN_OR_WEIGHTED = MATRIX_FAMILIES[1:]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(
            dict(value),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        newline="",
        encoding="utf-8",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def expected_bundle_from_freeze(
    row: Mapping[str, str],
) -> tuple[dict[str, Path], dict[str, str]]:
    paths = {
        family: Path(row[f"matrix_{family}_path"])
        for family in MATRIX_FAMILIES
    }
    hashes = {
        family: row[f"matrix_{family}_sha256"]
        for family in MATRIX_FAMILIES
    }
    return paths, hashes


def expected_bundle_from_prefix(
    prefix: Path,
    hashes: Mapping[str, str],
) -> tuple[dict[str, Path], dict[str, str]]:
    return (
        {
            family: Path(f"{prefix}_{family}.csv")
            for family in MATRIX_FAMILIES
        },
        {family: str(hashes[family]) for family in MATRIX_FAMILIES},
    )


def matrix_contract(
    *,
    family: str,
    matrix: np.ndarray,
    count_support: np.ndarray | None,
) -> dict[str, Any]:
    shape_pass = matrix.shape == (NODES, NODES)
    if not shape_pass:
        return {
            "shape_pass": False,
            "finite_pass": False,
            "symmetric_pass": False,
            "diagonal_zero_pass": False,
            "nonnegative_pass": False,
            "semantic_range_pass": False,
            "support_match_pass": False,
            "support_missing_edges": None,
            "support_extra_edges": None,
            "minimum": None,
            "maximum": None,
        }
    finite_pass = bool(np.all(np.isfinite(matrix)))
    symmetric_pass = bool(
        finite_pass
        and np.allclose(matrix, matrix.T, rtol=0, atol=1.0e-8)
    )
    diagonal_zero_pass = bool(
        finite_pass
        and np.allclose(np.diag(matrix), 0, rtol=0, atol=1.0e-8)
    )
    nonnegative_pass = bool(finite_pass and not np.any(matrix < -1.0e-12))
    minimum = float(np.min(matrix)) if finite_pass else None
    maximum = float(np.max(matrix)) if finite_pass else None
    semantic_range_pass = nonnegative_pass
    if family == "fa_mean":
        semantic_range_pass = bool(
            nonnegative_pass and maximum is not None and maximum <= 1.0 + 1e-8
        )

    support_missing_edges: int | None = None
    support_extra_edges: int | None = None
    support_match_pass = family == "count"
    if family != "count" and finite_pass and count_support is not None:
        family_support = np.abs(matrix) > 1.0e-12
        upper = np.triu(np.ones(matrix.shape, dtype=bool), 1)
        support_missing_edges = int(
            np.count_nonzero(upper & count_support & ~family_support)
        )
        support_extra_edges = int(
            np.count_nonzero(upper & ~count_support & family_support)
        )
        support_match_pass = (
            support_missing_edges == 0 and support_extra_edges == 0
        )
    return {
        "shape_pass": shape_pass,
        "finite_pass": finite_pass,
        "symmetric_pass": symmetric_pass,
        "diagonal_zero_pass": diagonal_zero_pass,
        "nonnegative_pass": nonnegative_pass,
        "semantic_range_pass": semantic_range_pass,
        "support_match_pass": support_match_pass,
        "support_missing_edges": support_missing_edges,
        "support_extra_edges": support_extra_edges,
        "minimum": minimum,
        "maximum": maximum,
    }


def audit_bundle(
    unit: str,
    source: str,
    release_status: str,
    paths: Mapping[str, Path],
    expected_hashes: Mapping[str, str] | None,
) -> dict[str, Any]:
    errors: list[str] = []
    family_results: dict[str, dict[str, Any]] = {}
    count_support: np.ndarray | None = None
    density: float | None = None
    connected_nodes: int | None = None
    total_count: float | None = None
    present_n = 0
    hash_pass_n = 0
    order = ("count",) + MEAN_OR_WEIGHTED
    for family in order:
        path = paths[family].resolve()
        result: dict[str, Any] = {
            "path": str(path),
            "present": path.is_file() and not path.is_symlink(),
            "hash_bound": expected_hashes is not None,
            "hash_pass": False,
        }
        if not result["present"]:
            errors.append(f"{family}:missing")
            family_results[family] = result
            continue
        present_n += 1
        actual_hash = sha256_file(path)
        result["sha256"] = actual_hash
        if expected_hashes is None:
            result["hash_pass"] = True
        else:
            result["expected_sha256"] = expected_hashes[family]
            result["hash_pass"] = actual_hash == expected_hashes[family]
        if result["hash_pass"]:
            hash_pass_n += 1
        else:
            errors.append(f"{family}:hash_mismatch")
        try:
            matrix = np.loadtxt(path, delimiter=",")
        except Exception as exc:
            errors.append(f"{family}:unreadable:{type(exc).__name__}")
            result["load_error"] = f"{type(exc).__name__}:{exc}"
            family_results[family] = result
            continue
        contract = matrix_contract(
            family=family,
            matrix=matrix,
            count_support=count_support,
        )
        result.update(contract)
        failed = [
            key
            for key in (
                "shape_pass",
                "finite_pass",
                "symmetric_pass",
                "diagonal_zero_pass",
                "nonnegative_pass",
                "semantic_range_pass",
                "support_match_pass",
            )
            if not result.get(key, False)
        ]
        if failed:
            errors.append(f"{family}:contract:{'|'.join(failed)}")
        if family == "count" and result.get("shape_pass"):
            count_support = matrix > 0
            upper_edges = int(
                np.count_nonzero(np.triu(count_support, 1))
            )
            density = upper_edges / POSSIBLE_EDGES
            connected_nodes = int(
                np.count_nonzero(
                    np.any(count_support, axis=0)
                    | np.any(count_support, axis=1)
                )
            )
            total_count = float(np.triu(matrix, 1).sum())
        family_results[family] = result

    contract_pass = present_n == 9 and hash_pass_n == 9 and not errors
    return {
        "unit": unit,
        "source": source,
        "release_status": release_status,
        "hashes_bound": expected_hashes is not None,
        "present_matrix_n": present_n,
        "hash_pass_n": hash_pass_n,
        "matrix_contract_pass": contract_pass,
        "density": density,
        "density_ge_0_60": (
            density is not None and density >= MINIMUM_DENSITY
        ),
        "connected_nodes": connected_nodes,
        "total_undirected_count": total_count,
        "errors": errors,
        "family_results": family_results,
    }


def density_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray(
        [
            float(row["density"])
            for row in rows
            if row.get("density") is not None
        ],
        dtype=float,
    )
    if not values.size:
        return {"n": 0}
    return {
        "n": int(values.size),
        "minimum": float(values.min()),
        "q1": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q3": float(np.quantile(values, 0.75)),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "n_ge_0_60": int(np.count_nonzero(values >= 0.60)),
    }


def build_sources(
    topology: list[dict[str, str]],
) -> tuple[
    dict[str, tuple[str, str, dict[str, Path], dict[str, str]]],
    list[dict[str, Any]],
]:
    freeze_rows = {row["fsid"]: row for row in read_csv(FREEZE)}
    archive = load_json(ARCHIVE)
    tensor = load_json(TENSOR)
    archive_rows = {
        str(row["fsid"]): row for row in archive["subjects"]
    }
    tensor_rows = {
        str(row["fsid"]): row for row in tensor["subjects"]
    }
    topology_by_unit = {row["unit"]: row for row in topology}
    sources: dict[
        str, tuple[str, str, dict[str, Path], dict[str, str]]
    ] = {}
    for unit, row in topology_by_unit.items():
        lane = row["inventory_lane"]
        if lane == "legacy_keep_214":
            paths, hashes = expected_bundle_from_freeze(freeze_rows[unit])
            sources[unit] = (
                "historical_freeze_legacy_keep",
                "BASELINE_ONLY_CORRECTED_RERUN_PLANNED",
                paths,
                hashes,
            )
        elif lane == "archive_86":
            record = archive_rows[unit]
            prefix = (
                CENTRAL_CONNECTOMES / f"SC_HCPMMP1_{unit}"
            )
            paths, hashes = expected_bundle_from_prefix(
                prefix, record["matrix_sha256"]
            )
            sources[unit] = (
                "archive_track_matched_rebuild",
                "CONDITIONAL_ON_ROUTE_CONCORDANCE"
                if row["density_group"] == "D0_THRESHOLD_QUALIFIED"
                else "BASELINE_ONLY_CORRECTED_RERUN_PLANNED",
                paths,
                hashes,
            )
        elif lane == "tensor_repair_2":
            record = tensor_rows[unit]
            paths, hashes = expected_bundle_from_prefix(
                Path(record["output_prefix"]),
                record["matrix_sha256"],
            )
            sources[unit] = (
                "tensor_weighted_repair",
                "BASELINE_ONLY_CORRECTED_RERUN_PLANNED",
                paths,
                hashes,
            )

    expected_units = {
        row["unit"]
        for row in topology
        if row["inventory_lane"]
        in {"legacy_keep_214", "archive_86", "tensor_repair_2"}
    }
    if (
        len(topology_by_unit) != 530
        or len(sources) != 302
        or set(sources) != expected_units
    ):
        raise ValueError("live historical source partition differs")

    unbound: list[dict[str, Any]] = []
    central_units = {
        path.name.removeprefix("SC_HCPMMP1_").removesuffix("_count.csv")
        for path in CENTRAL_CONNECTOMES.glob(
            "SC_HCPMMP1_*_count.csv"
        )
    }
    for unit in sorted(central_units - set(archive_rows)):
        if unit not in topology_by_unit:
            raise ValueError(f"central matrix unit outside topology: {unit}")
        prefix = CENTRAL_CONNECTOMES / f"SC_HCPMMP1_{unit}"
        unbound.append(
            {
                "unit": unit,
                "source": "unbound_central_candidate",
                "release_status": "NOT_RELEASEABLE_UNBOUND_ROUTE",
                "paths": {
                    family: Path(f"{prefix}_{family}.csv")
                    for family in MATRIX_FAMILIES
                },
            }
        )
    return sources, unbound


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_summary = output_root / OUTPUT_SUMMARY.name
    output_inventory = output_root / OUTPUT_INVENTORY.name
    output_matrix_audit = output_root / OUTPUT_MATRIX_AUDIT.name
    existing = [
        path
        for path in (
            output_summary,
            output_inventory,
            output_matrix_audit,
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "non-overwriting audit outputs already exist: "
            + ", ".join(str(path) for path in existing)
        )
    if args.workers < 1 or args.workers > 8:
        raise ValueError("--workers must be between 1 and 8")

    topology = read_csv(TOPOLOGY)
    if len(topology) != 530 or len({row["unit"] for row in topology}) != 530:
        raise ValueError("execution topology is not exact 530")
    if any(row["diagnosis_or_outcomes_used"] != "False" for row in topology):
        raise ValueError("execution topology is not diagnosis blind")
    if Counter(row["execution_lane"] for row in topology) != {
        "corrected_core_227": 227,
        "corrected_legacy_tensor_216": 216,
        "conditional_archive_D0_56": 56,
        "corrected_archive_low_30": 30,
        "corrected_freesurfer_1": 1,
    }:
        raise ValueError("530 execution-lane partition differs")

    sources, unbound = build_sources(topology)
    jobs = [
        (unit, *source)
        for unit, source in sorted(sources.items())
    ]
    jobs.extend(
        (
            row["unit"],
            row["source"],
            row["release_status"],
            row["paths"],
            None,
        )
        for row in unbound
    )
    matrix_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                audit_bundle,
                unit,
                source,
                release_status,
                paths,
                hashes,
            ): unit
            for unit, source, release_status, paths, hashes in jobs
        }
        completed = 0
        for future in as_completed(futures):
            matrix_rows.append(future.result())
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"AUDIT_PROGRESS {completed}/{len(futures)}",
                    flush=True,
                )
    matrix_rows.sort(key=lambda row: (row["unit"], row["source"]))
    matrix_by_unit = {
        row["unit"]: row
        for row in matrix_rows
        if row["unit"] in sources
    }

    inventory_rows: list[dict[str, Any]] = []
    for row in sorted(topology, key=lambda item: item["unit"]):
        current = matrix_by_unit.get(row["unit"])
        inventory_rows.append(
            {
                **row,
                "current_bundle_source": (
                    current["source"] if current else ""
                ),
                "current_bundle_matrix_n": (
                    current["present_matrix_n"] if current else 0
                ),
                "current_bundle_contract": (
                    "PASS"
                    if current and current["matrix_contract_pass"]
                    else "FAIL"
                    if current
                    else "MISSING"
                ),
                "current_bundle_release_status": (
                    current["release_status"] if current else "NONE"
                ),
                "verified_current_density": (
                    current["density"] if current else ""
                ),
                "verified_density_ge_0_60": (
                    current["density_ge_0_60"] if current else False
                ),
                "connected_nodes": (
                    current["connected_nodes"] if current else ""
                ),
                "final_release_state": "PENDING_CORRECTED_RELEASE",
                "final_matrix_bundle_present": False,
            }
        )

    matrix_fields = [
        "unit",
        "source",
        "release_status",
        "hashes_bound",
        "present_matrix_n",
        "hash_pass_n",
        "matrix_contract_pass",
        "density",
        "density_ge_0_60",
        "connected_nodes",
        "total_undirected_count",
        "error_count",
        "errors",
    ]
    flat_matrix_rows = []
    for row in matrix_rows:
        flat_matrix_rows.append(
            {
                **row,
                "error_count": len(row["errors"]),
                "errors": ";".join(row["errors"]),
            }
        )
    inventory_fields = list(inventory_rows[0])
    atomic_csv(output_inventory, inventory_rows, inventory_fields)
    atomic_csv(output_matrix_audit, flat_matrix_rows, matrix_fields)

    matrix_contract_pass_n = sum(
        bool(row["matrix_contract_pass"])
        for row in matrix_rows
        if row["unit"] in sources
    )
    unbound_rows = [
        row for row in matrix_rows
        if row["source"] == "unbound_central_candidate"
    ]
    summary = {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_live_cohort_inventory_audit",
        "status": (
            "PASS_BASELINE_INVENTORY_FINAL_RELEASE_PENDING"
            if matrix_contract_pass_n == 302
            else "FAIL_CURRENT_MATRIX_CONTRACT"
        ),
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "subject_count": 530,
        "matrix_family_count": 9,
        "expected_final_matrix_files": 4770,
        "final_release_matrix_files_verified": 0,
        "final_release_subjects_verified": 0,
        "non_overwriting": True,
        "historical_current_state": {
            "expected_baseline_bundle_n": 302,
            "audited_baseline_bundle_n": len(sources),
            "baseline_all_nine_contract_pass_n": matrix_contract_pass_n,
            "baseline_density": density_summary(
                row for row in matrix_rows if row["unit"] in sources
            ),
            "baseline_at_or_above_0_60_n": sum(
                bool(row["density_ge_0_60"])
                for row in matrix_rows
                if row["unit"] in sources
            ),
            "baseline_below_0_60_n": sum(
                not bool(row["density_ge_0_60"])
                for row in matrix_rows
                if row["unit"] in sources
            ),
            "no_baseline_bundle_n": 530 - len(sources),
            "baseline_is_final_release": False,
        },
        "unbound_current_candidates": {
            "n": len(unbound_rows),
            "units": [row["unit"] for row in unbound_rows],
            "numerical_contract_pass_n": sum(
                bool(row["matrix_contract_pass"]) for row in unbound_rows
            ),
            "release_eligible_n": 0,
        },
        "execution_lanes": dict(
            sorted(Counter(row["execution_lane"] for row in topology).items())
        ),
        "completion_gap": {
            "corrected_final_subjects_remaining": 530,
            "corrected_final_matrix_files_remaining": 4770,
            "human_visual_gate_pending": True,
            "phase_b_recipe_selection_pending": True,
            "archive_concordance_pending": True,
        },
        "artifacts": {
            "inventory_530": file_record(output_inventory),
            "current_matrix_audit": file_record(output_matrix_audit),
        },
        "sources": {
            "execution_topology": file_record(TOPOLOGY),
            "historical_freeze": file_record(FREEZE),
            "archive_restore_summary": file_record(ARCHIVE),
            "tensor_repair_summary": file_record(TENSOR),
        },
        "interpretation": (
            "A passing baseline contract proves the currently available "
            "matrices are readable and internally matched. It does not "
            "promote them into the corrected 530-subject release."
        ),
    }
    atomic_json(output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if matrix_contract_pass_n == 302 else 2


if __name__ == "__main__":
    raise SystemExit(main())
