#!/home/ec2-user/fsl/bin/python
"""Hash-verify and audit count-matrix density for the 302 extant bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
ROOT = Path("/data/derivatives/hcp379_v2")
ROUTES = EXP / "research_audit/outputs/hcp_rerun_triage_v1"
FREEZE = ROOT / "manifests/historical_freeze_manifest.csv"
ARCHIVE = ROOT / "manifests/archive_restore_summary.json"
TENSOR = ROOT / "manifests/tensor_repair_summary.json"
OUTPUT_ROOT = (
    EXP
    / "research_audit/outputs/hcp379_existing_302_density_v2"
)
OUTPUT_JSON = OUTPUT_ROOT / "existing_302_density_audit.json"
OUTPUT_CSV = OUTPUT_ROOT / "existing_302_density_subjects.csv"
NODES = 379
POSSIBLE_EDGES = NODES * (NODES - 1) // 2


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


def unit_set(name: str, expected: int) -> set[str]:
    path = ROUTES / name
    values = {
        value.strip()
        for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    }
    if len(values) != expected:
        raise ValueError(f"{name}: expected {expected}, found {len(values)}")
    return values


def count_density(
    path: Path, expected_sha256: str
) -> tuple[float, dict[str, Any]]:
    record = file_record(path)
    if record["sha256"] != expected_sha256:
        raise ValueError(f"count matrix hash differs: {path}")
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.shape != (NODES, NODES):
        raise ValueError(f"count matrix shape differs: {path}:{matrix.shape}")
    if (
        not np.all(np.isfinite(matrix))
        or np.any(matrix < 0)
        or not np.allclose(matrix, matrix.T, rtol=0, atol=1.0e-8)
        or not np.allclose(
            np.diag(matrix), 0, rtol=0, atol=1.0e-8
        )
    ):
        raise ValueError(f"count matrix numerical contract differs: {path}")
    edges = int(np.count_nonzero(np.triu(matrix, 1) > 0))
    return edges / POSSIBLE_EDGES, record


def lane_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray(
        [float(row["density"]) for row in rows], dtype=float
    )
    if values.size == 0:
        raise ValueError("empty density lane")
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "n_ge_0_60": int(np.count_nonzero(values >= 0.60)),
        "n_ge_0_70": int(np.count_nonzero(values >= 0.70)),
        "n_lt_0_20": int(np.count_nonzero(values < 0.20)),
    }


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


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "unit",
        "lane",
        "density",
        "nonzero_undirected_edges",
        "possible_undirected_edges",
        "count_matrix_path",
        "count_matrix_size_bytes",
        "count_matrix_sha256",
    ]
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        newline="",
        encoding="utf-8",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def audit() -> dict[str, Any]:
    keep = unit_set("keep_high_confidence_214.txt", 214)
    archive_units = unit_set("archive_recovery_86.txt", 86)
    tensor_units = unit_set("tensor_weighted_repair_2.txt", 2)
    if keep & archive_units or keep & tensor_units or archive_units & tensor_units:
        raise ValueError("302-lane identity sets overlap")

    freeze = {row["fsid"]: row for row in read_csv(FREEZE)}
    archive = load_json(ARCHIVE)
    archive_by_unit = {
        str(row["fsid"]): row for row in archive["subjects"]
    }
    tensor = load_json(TENSOR)
    tensor_by_unit = {
        str(row["fsid"]): row for row in tensor["subjects"]
    }
    if (
        not keep.issubset(freeze)
        or set(archive_by_unit) != archive_units
        or set(tensor_by_unit) != tensor_units
    ):
        raise ValueError("302-lane source identities differ")

    rows: list[dict[str, Any]] = []

    def append(
        unit: str,
        lane: str,
        path: Path,
        expected_sha256: str,
        stored_density: float | None = None,
    ) -> None:
        density, record = count_density(path, expected_sha256)
        if (
            stored_density is not None
            and abs(density - stored_density) > 1.0e-12
        ):
            raise ValueError(f"{unit}:stored archive density differs")
        rows.append(
            {
                "unit": unit,
                "lane": lane,
                "density": density,
                "nonzero_undirected_edges": round(
                    density * POSSIBLE_EDGES
                ),
                "possible_undirected_edges": POSSIBLE_EDGES,
                "count_matrix_path": record["path"],
                "count_matrix_size_bytes": record["size_bytes"],
                "count_matrix_sha256": record["sha256"],
            }
        )

    for unit in sorted(keep):
        source = freeze[unit]
        append(
            unit,
            "legacy_keep_214",
            Path(source["matrix_count_path"]),
            source["matrix_count_sha256"],
        )
    for unit in sorted(archive_units):
        source = archive_by_unit[unit]
        append(
            unit,
            "archive_86",
            ROOT / "connectomes" / f"SC_HCPMMP1_{unit}_count.csv",
            str(source["matrix_sha256"]["count"]),
            float(source["density"]),
        )
    for unit in sorted(tensor_units):
        source = tensor_by_unit[unit]
        append(
            unit,
            "tensor_repairs_2",
            Path(str(source["output_prefix"]) + "_count.csv"),
            str(source["matrix_sha256"]["count"]),
        )

    if len(rows) != 302 or len({row["unit"] for row in rows}) != 302:
        raise ValueError("existing density audit is not exact 302")
    rows.sort(key=lambda row: row["unit"])
    lanes = {
        lane: lane_summary(row for row in rows if row["lane"] == lane)
        for lane in (
            "legacy_keep_214",
            "archive_86",
            "tensor_repairs_2",
        )
    }
    report = {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_existing_302_density_audit",
        "status": "PASS",
        "diagnosis_labels_used": False,
        "unit_count": 302,
        "definition": {
            "nodes": NODES,
            "possible_undirected_off_diagonal_edges": POSSIBLE_EDGES,
            "formula": (
                "nonzero upper-triangle count-matrix edges divided by 71631"
            ),
            "density_is_a_cohort_recipe_qc_measure_not_a_subject_exclusion": (
                True
            ),
        },
        "all_302": lane_summary(rows),
        "lanes": lanes,
        "lowest_10": sorted(
            (
                {
                    "unit": row["unit"],
                    "lane": row["lane"],
                    "density": row["density"],
                }
                for row in rows
            ),
            key=lambda row: row["density"],
        )[:10],
        "highest_10": sorted(
            (
                {
                    "unit": row["unit"],
                    "lane": row["lane"],
                    "density": row["density"],
                }
                for row in rows
            ),
            key=lambda row: row["density"],
            reverse=True,
        )[:10],
        "source_records": {
            "historical_freeze": file_record(FREEZE),
            "archive_summary": file_record(ARCHIVE),
            "tensor_summary": file_record(TENSOR),
            "audit_script": file_record(Path(__file__)),
        },
        "subject_table": str(OUTPUT_CSV.resolve()),
    }
    atomic_csv(OUTPUT_CSV, rows)
    atomic_json(OUTPUT_JSON, report)
    return report


def main() -> int:
    report = audit()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
