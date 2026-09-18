#!/usr/bin/env python3
"""Audit whether HCP379 edge support is still sampling-limited at 20M.

This diagnosis-blind, non-imaging audit uses the two independent 10M support
sets already generated for each corrected final-HROI canary. It reports the
observed 20M union, marginal gains, and a two-sample Chapman
capture-recapture estimate of the support ceiling. The estimate is diagnostic
only because tractography edges have unequal detection probabilities.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
COHORT_ATTEMPTS = (
    HCP_ROOT / "phase_b_hroi_balanced_density_canary_cohort_v1/attempts"
)
UNIT_ATTEMPTS = (
    HCP_ROOT / "phase_b_hroi_balanced_density_canary_v1/attempts"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/hcp379_support_saturation_audit_v1"
)
EXPECTED_UNITS = 15
EXPECTED_NODES = 379
POSSIBLE_EDGES = EXPECTED_NODES * (EXPECTED_NODES - 1) // 2
DENSITY_TARGET = 0.60
TARGET_EDGE_N = 42_979


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


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


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("empty support-saturation table")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("inconsistent support-saturation columns")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def latest_cohort_summary() -> tuple[Path, dict[str, Any]]:
    for path in sorted(COHORT_ATTEMPTS.glob("*/summary.json"), reverse=True):
        try:
            value = load_json(path)
        except Exception:
            continue
        if (
            value.get("record_type")
            == "diagnosis_blind_hcp379_hroi_balanced_density_canary_cohort_summary"
            and value.get("status")
            == "PASS_COMPLETE_COUNT_ONLY_DENSITY_EVIDENCE"
            and value.get("processed_unit_n") == EXPECTED_UNITS
            and len(value.get("units", [])) == EXPECTED_UNITS
        ):
            return path.resolve(), value
    raise FileNotFoundError("complete density cohort summary absent")


def latest_unit_summary(
    unit: str, fallback: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    for path in sorted(
        UNIT_ATTEMPTS.glob(f"*-{unit}/summary.json"), reverse=True
    ):
        try:
            value = load_json(path)
        except Exception:
            continue
        if (
            value.get("record_type")
            == "diagnosis_blind_hcp379_hroi_balanced_density_canary_summary"
            and value.get("status")
            == "PASS_HROI_BALANCED_DENSITY_CANARY"
            and value.get("unit") == unit
            and set(value.get("levels", {}))
            == {"6000000", "10000000", "15000000", "20000000"}
        ):
            return path.resolve(), value
    path = Path(str(fallback["path"])).resolve()
    return path, load_json(path)


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cohort_path, cohort = latest_cohort_summary()
    rows: list[dict[str, Any]] = []
    unit_records: dict[str, Any] = {}
    for cohort_row in cohort["units"]:
        unit = str(cohort_row["unit"])
        summary_path, summary = latest_unit_summary(
            unit, cohort_row["summary"]
        )
        levels = summary["levels"]
        level20 = levels["20000000"]
        support = level20["support_complementarity"]
        primary = int(support["primary_supported_edges"])
        independent = int(support["independent_supported_edges"])
        overlap = int(support["overlap_supported_edges"])
        union = int(support["union_supported_edges"])
        if primary + independent - overlap != union:
            raise ValueError(f"{unit}: support-set algebra differs")
        chapman = (
            ((primary + 1) * (independent + 1) / (overlap + 1)) - 1
        )
        chapman = min(float(POSSIBLE_EDGES), max(float(union), chapman))
        ceiling_density = chapman / POSSIBLE_EDGES
        observed_density = float(level20["edge_density"])
        if int(level20["supported_edges"]) != union:
            raise ValueError(f"{unit}: 20M union/count support differs")
        if observed_density >= DENSITY_TARGET:
            classification = "OBSERVED_DENSITY_TARGET_PASS"
        elif ceiling_density < DENSITY_TARGET:
            classification = (
                "DIAGNOSTIC_SUPPORT_CEILING_BELOW_DENSITY_TARGET"
            )
        else:
            classification = (
                "DIAGNOSTIC_SUPPORT_CEILING_AT_OR_ABOVE_TARGET"
            )
        rows.append(
            {
                "unit": unit,
                "density_6m": float(levels["6000000"]["edge_density"]),
                "density_10m": float(levels["10000000"]["edge_density"]),
                "density_15m": float(levels["15000000"]["edge_density"]),
                "density_20m": observed_density,
                "edges_20m": union,
                "edges_needed_for_0_60": max(0, TARGET_EDGE_N - union),
                "gain_edges_15m_to_20m": union
                - int(levels["15000000"]["supported_edges"]),
                "primary_10m_edges": primary,
                "independent_10m_edges": independent,
                "overlap_edges": overlap,
                "primary_only_edges": int(
                    support["primary_only_supported_edges"]
                ),
                "independent_only_edges": int(
                    support["independent_only_supported_edges"]
                ),
                "support_jaccard": float(support["support_jaccard"]),
                "chapman_support_estimate": chapman,
                "chapman_ceiling_density": ceiling_density,
                "observed_fraction_of_chapman_support": union / chapman,
                "classification": classification,
            }
        )
        unit_records[unit] = file_record(summary_path)
    rows.sort(key=lambda row: row["unit"])
    if len(rows) != EXPECTED_UNITS or len({row["unit"] for row in rows}) != 15:
        raise ValueError("canary unit set differs")
    summary = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_support_saturation_audit",
        "generated_utc": utc_now(),
        "status": "PASS_DIAGNOSTIC_SUPPORT_SATURATION_AUDIT",
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "imaging_executed": False,
        "matrices_modified": False,
        "density_target": DENSITY_TARGET,
        "target_edge_n": TARGET_EDGE_N,
        "possible_edges": POSSIBLE_EDGES,
        "observed_density_pass_n": sum(
            row["density_20m"] >= DENSITY_TARGET for row in rows
        ),
        "observed_density_fail_n": sum(
            row["density_20m"] < DENSITY_TARGET for row in rows
        ),
        "failed_with_diagnostic_ceiling_below_target_n": sum(
            row["classification"]
            == "DIAGNOSTIC_SUPPORT_CEILING_BELOW_DENSITY_TARGET"
            for row in rows
        ),
        "classification_counts": {
            label: sum(row["classification"] == label for row in rows)
            for label in sorted({row["classification"] for row in rows})
        },
        "method": (
            "Two independent 10M tractography support sets are treated as "
            "two captures. Chapman total=((n1+1)(n2+1)/(m+1))-1."
        ),
        "interpretation_boundary": (
            "This is a sampling-saturation diagnostic, not a production QC "
            "gate or anatomical truth estimate. Unequal edge detectability "
            "violates the simple capture-recapture model, so the ceiling must "
            "not be used to select subjects, tune recipes per subject, or "
            "override density/all-nine/seed-stability requirements."
        ),
        "records": {
            "cohort_summary": file_record(cohort_path),
            "unit_summaries": unit_records,
            "builder": file_record(Path(__file__)),
        },
    }
    return rows, summary


def main() -> int:
    rows, summary = build()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_csv(OUTPUT / "subject_support_saturation.csv", rows)
    summary["records"]["subject_table"] = file_record(
        OUTPUT / "subject_support_saturation.csv"
    )
    atomic_json(OUTPUT / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "density_pass_n": summary["observed_density_pass_n"],
                "density_fail_n": summary["observed_density_fail_n"],
                "failed_with_diagnostic_ceiling_below_target_n": (
                    summary[
                        "failed_with_diagnostic_ceiling_below_target_n"
                    ]
                ),
                "output": str(OUTPUT / "summary.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
