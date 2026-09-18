#!/home/ec2-user/fsl/bin/python
"""Build the exact live 530-subject status and density crosswalk.

This audit distinguishes existing AAL3-166 matrices, existing HCP-MMP1-379
matrices, and the not-yet-authorized corrected HCP379 release.  It performs no
imaging, changes no matrix, and uses no diagnosis or outcome label.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
DERIV = Path("/data/derivatives")
TOPOLOGY = (
    EXP
    / "research_audit/outputs/hcp379_balanced_release_topology_v1/"
    "topology.json"
)
LEDGER = (
    EXP
    / "research_audit/outputs/hcp379_pretract_failure_recovery_ledger_v1/"
    "ledger.csv"
)
EXISTING_HCP = (
    EXP
    / "research_audit/outputs/hcp379_existing_302_density_v2/"
    "existing_302_density_subjects.csv"
)
AAL_ROOT = DERIV / "connectomes"
OUTPUT = (
    EXP / "research_audit/outputs/hcp379_live_530_status_density_v1"
)
EXPECTED_CANARY_N = 15
EXPECTED_PRODUCTION_N = 515
EXPECTED_TOTAL_N = 530
EXPECTED_AAL_N = 166
EXPECTED_HCP_N = 379
DENSITY_GATE = 0.60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def matrix_density(path: Path, expected_n: int) -> float:
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.shape != (expected_n, expected_n):
        raise ValueError(f"{path}: shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: non-finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-8):
        raise ValueError(f"{path}: asymmetric matrix")
    possible = expected_n * (expected_n - 1) // 2
    return float(np.count_nonzero(np.triu(matrix, 1)) / possible)


def density_summary(
    units: Iterable[str], values: Mapping[str, float]
) -> dict[str, Any]:
    observed = [values[unit] for unit in units if unit in values]
    return {
        "available_n": len(observed),
        "median": median(observed) if observed else None,
        "pass_0_60_n": sum(value >= DENSITY_GATE for value in observed),
    }


def display_density(value: Any, n: int) -> str:
    return "PENDING" if value is None else f"{float(value):.3f} (n={n})"


def build() -> dict[str, Any]:
    topology_text = TOPOLOGY.read_text(encoding="utf-8")
    ledger_text = LEDGER.read_text(encoding="utf-8-sig")
    topology = json.loads(topology_text)
    if not isinstance(topology, dict):
        raise ValueError(f"JSON object required: {TOPOLOGY}")
    ledger = [
        dict(row) for row in csv.DictReader(io.StringIO(ledger_text))
    ]
    hcp_rows = read_csv(EXISTING_HCP)
    canary_rows = topology.get("canaries", [])
    production_rows = topology.get("production", [])
    if (
        len(canary_rows) != EXPECTED_CANARY_N
        or len(production_rows) != EXPECTED_PRODUCTION_N
        or len(ledger) != EXPECTED_PRODUCTION_N
    ):
        raise ValueError("exact 15 plus 515 source inputs required")
    canaries = [str(row["unit"]) for row in canary_rows]
    production = [str(row["unit"]) for row in production_rows]
    all_units = canaries + production
    if (
        len(set(canaries)) != EXPECTED_CANARY_N
        or len(set(production)) != EXPECTED_PRODUCTION_N
        or set(canaries) & set(production)
        or len(set(all_units)) != EXPECTED_TOTAL_N
    ):
        raise ValueError("exact disjoint 530 identity set required")

    ledger_by_unit = {row["unit"]: row for row in ledger}
    if set(ledger_by_unit) != set(production):
        raise ValueError("ledger and production identities differ")

    aal: dict[str, float] = {}
    aal_records: dict[str, dict[str, Any]] = {}
    for unit in all_units:
        path = AAL_ROOT / f"SC_AAL166_{unit}_count.csv"
        if not path.is_file():
            raise ValueError(f"missing existing AAL3-166 matrix: {unit}")
        aal[unit] = matrix_density(path, EXPECTED_AAL_N)
        aal_records[unit] = file_record(path)

    hcp: dict[str, float] = {}
    hcp_records: dict[str, dict[str, Any]] = {}
    for row in hcp_rows:
        unit = str(row["unit"])
        if unit not in set(all_units):
            continue
        path = Path(row["count_matrix_path"])
        if (
            not path.is_file()
            or sha256_file(path) != row["count_matrix_sha256"]
        ):
            raise ValueError(f"existing HCP379 binding differs: {unit}")
        observed = matrix_density(path, EXPECTED_HCP_N)
        recorded = float(row["density"])
        if abs(observed - recorded) > 1e-12:
            raise ValueError(f"existing HCP379 density differs: {unit}")
        hcp[unit] = observed
        hcp_records[unit] = file_record(path)

    groups: list[tuple[str, list[str]]] = [
        ("Phase-B canaries (reused)", canaries)
    ]
    lane_labels = {
        "archive_calibration": "Archive calibration",
        "corrected_archive_low": "Archive-low recovery",
        "corrected_archive_D0": "Archive-D0 recovery",
        "corrected_core": "Corrected core",
        "corrected_legacy_tensor": "Corrected legacy/tensor",
        "corrected_freesurfer": "Isolated FreeSurfer recovery",
    }
    for lane in (
        "archive_calibration",
        "corrected_archive_low",
        "corrected_archive_D0",
        "corrected_core",
        "corrected_legacy_tensor",
        "corrected_freesurfer",
    ):
        units = [
            str(row["unit"])
            for row in production_rows
            if row["lane"] == lane
        ]
        groups.append((lane_labels[lane], units))

    rows: list[dict[str, Any]] = []
    for label, units in groups:
        if label == "Phase-B canaries (reused)":
            status = Counter({"PASS_COMPACTED": len(units)})
        else:
            status = Counter(ledger_by_unit[unit]["status"] for unit in units)
        aal_stat = density_summary(units, aal)
        hcp_stat = density_summary(units, hcp)
        rows.append(
            {
                "processing_group": label,
                "target_n": len(units),
                "pretract_completed_n": status.get("PASS_COMPACTED", 0),
                "running_n": status.get("RUNNING", 0),
                "corrective_hold_n": status.get("FAILED", 0),
                "waiting_n": status.get("WAITING", 0),
                "completion_fraction": (
                    status.get("PASS_COMPACTED", 0) / len(units)
                ),
                "existing_aal3_166_available_n": aal_stat["available_n"],
                "existing_aal3_166_median_density": aal_stat["median"],
                "existing_aal3_166_pass_0_60_n": aal_stat["pass_0_60_n"],
                "existing_hcp379_available_n": hcp_stat["available_n"],
                "existing_hcp379_median_density": hcp_stat["median"],
                "existing_hcp379_pass_0_60_n": hcp_stat["pass_0_60_n"],
                "corrected_hcp379_available_n": 0,
                "corrected_hcp379_median_density": "",
                "corrected_hcp379_status": "PENDING_RECIPE_AND_PRODUCTION",
            }
        )

    full_aal = density_summary(all_units, aal)
    full_hcp = density_summary(all_units, hcp)
    production_status = Counter(row["status"] for row in ledger)
    total_row = {
        "processing_group": "Full release total",
        "target_n": EXPECTED_TOTAL_N,
        "pretract_completed_n": (
            EXPECTED_CANARY_N + production_status.get("PASS_COMPACTED", 0)
        ),
        "running_n": production_status.get("RUNNING", 0),
        "corrective_hold_n": production_status.get("FAILED", 0),
        "waiting_n": production_status.get("WAITING", 0),
        "completion_fraction": (
            EXPECTED_CANARY_N
            + production_status.get("PASS_COMPACTED", 0)
        )
        / EXPECTED_TOTAL_N,
        "existing_aal3_166_available_n": full_aal["available_n"],
        "existing_aal3_166_median_density": full_aal["median"],
        "existing_aal3_166_pass_0_60_n": full_aal["pass_0_60_n"],
        "existing_hcp379_available_n": full_hcp["available_n"],
        "existing_hcp379_median_density": full_hcp["median"],
        "existing_hcp379_pass_0_60_n": full_hcp["pass_0_60_n"],
        "corrected_hcp379_available_n": 0,
        "corrected_hcp379_median_density": "",
        "corrected_hcp379_status": "PENDING_RECIPE_AND_PRODUCTION",
    }
    rows.append(total_row)

    atomic_text(OUTPUT / "topology_snapshot.json", topology_text)
    atomic_text(OUTPUT / "ledger_snapshot.csv", ledger_text)
    atomic_csv(OUTPUT / "group_status_density.csv", rows)
    subject_rows = [
        {
            "unit": unit,
            "execution_set": (
                "CANARY_REUSE" if unit in set(canaries) else "PRODUCTION"
            ),
            "processing_group": (
                "Phase-B canaries (reused)"
                if unit in set(canaries)
                else lane_labels[
                    next(
                        str(row["lane"])
                        for row in production_rows
                        if row["unit"] == unit
                    )
                ]
            ),
            "pretract_status": (
                "PASS_COMPACTED"
                if unit in set(canaries)
                else ledger_by_unit[unit]["status"]
            ),
            "existing_aal3_166_density": aal[unit],
            "existing_aal3_166_matrix_path": aal_records[unit]["path"],
            "existing_hcp379_density": hcp.get(unit, ""),
            "existing_hcp379_matrix_path": (
                hcp_records[unit]["path"] if unit in hcp_records else ""
            ),
            "corrected_hcp379_density": "",
            "corrected_hcp379_status": "PENDING_RECIPE_AND_PRODUCTION",
        }
        for unit in sorted(all_units)
    ]
    atomic_csv(OUTPUT / "subject_status_density.csv", subject_rows)

    markdown = [
        "# Live HCP379-v2 530-subject status and density",
        "",
        "| Processing group | Target | Pretract complete | Running | Hold | Waiting | Completion | Existing AAL3-166 median | Existing HCP379 median | Corrected HCP379 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {processing_group} | {target_n} | "
            "{pretract_completed_n} | {running_n} | "
            "{corrective_hold_n} | {waiting_n} | "
            "{completion:.1%} | {aal} | {hcp} | PENDING |".format(
                **row,
                completion=float(row["completion_fraction"]),
                aal=display_density(
                    row["existing_aal3_166_median_density"],
                    int(row["existing_aal3_166_available_n"]),
                ),
                hcp=display_density(
                    row["existing_hcp379_median_density"],
                    int(row["existing_hcp379_available_n"]),
                ),
            )
        )
    markdown.extend(
        [
            "",
            "Density is the fraction of nonzero undirected off-diagonal "
            "edges. Existing AAL3-166 and existing HCP379 are retained "
            "inputs/evidence, not relabelled corrected outputs. Corrected "
            "HCP379 stays pending until the uniform recipe and production "
            "release pass.",
            "",
        ]
    )
    atomic_text(OUTPUT / "STATUS.md", "\n".join(markdown))

    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_live_530_status_density",
        "status": "CORRECTED_HCP379_PENDING",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "imaging_executed": False,
        "matrices_modified": False,
        "density_definition": (
            "nonzero undirected off-diagonal edges / n*(n-1)/2"
        ),
        "density_gate": DENSITY_GATE,
        "canary_n": EXPECTED_CANARY_N,
        "production_n": EXPECTED_PRODUCTION_N,
        "total_n": EXPECTED_TOTAL_N,
        "rows": rows,
        "records": {
            "live_topology_at_build": file_record(TOPOLOGY),
            "live_ledger_at_build": file_record(LEDGER),
            "topology_snapshot": file_record(
                OUTPUT / "topology_snapshot.json"
            ),
            "ledger_snapshot": file_record(OUTPUT / "ledger_snapshot.csv"),
            "existing_hcp379_density_audit": file_record(EXISTING_HCP),
            "group_csv": file_record(OUTPUT / "group_status_density.csv"),
            "subject_csv": file_record(OUTPUT / "subject_status_density.csv"),
            "status_markdown": file_record(OUTPUT / "STATUS.md"),
            "builder": file_record(Path(__file__)),
        },
    }
    atomic_json(OUTPUT / "summary.json", payload)
    return payload


def main() -> int:
    payload = build()
    print(
        json.dumps(
            {
                "status": payload["status"],
                "generated_utc": payload["generated_utc"],
                "total": payload["rows"][-1],
                "output": str(OUTPUT),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
