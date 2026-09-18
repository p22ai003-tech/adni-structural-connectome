#!/home/ec2-user/fsl/bin/python
"""Run one lock-safe auxiliary corrected-core Recovery4 worker.

The main worker traverses the exact corrected-core audit in its canonical
order.  This diagnosis-blind auxiliary traverses only the current WAITING
corrected-core identities in reverse order.  It rechecks state immediately
before execution, uses the shared engine's per-subject exclusive lock, applies
unchanged Recovery4 QC, and invokes the audited compactor after PASS.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
ENGINE_SOURCE = (
    EXP / "scripts/hcp/hcp379_scaleup_pretract_recovery4_v3.py"
)
LEDGER = (
    EXP
    / "research_audit/outputs/"
    "hcp379_pretract_failure_recovery_ledger_v1/ledger.csv"
)
TOPOLOGY = (
    EXP
    / "research_audit/outputs/"
    "hcp379_balanced_release_topology_v1/topology.json"
)
HUMAN_QC = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
ROOT = HCP_ROOT / "corrected_scaleup_recovery4"
ATTEMPTS = ROOT / "aux_worker_v1/attempts"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = load_module(ENGINE_SOURCE, "hcp379_core_pretract_aux_engine")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def current_state(unit: str) -> str:
    path = ROOT / "qc/subjects" / f"{unit}.json"
    if not path.is_file():
        return "MISSING"
    return str(load_json(path).get("status", "INVALID"))


def compacted(unit: str) -> bool:
    path = ROOT / "qc/pretract_compaction" / f"{unit}.json"
    return (
        path.is_file()
        and load_json(path).get("status") == "PASS_COMPACTED"
    )


def preflight() -> tuple[
    dict[str, Any],
    list[dict[str, str]],
    dict[str, Any],
]:
    gate = ENGINE.validate_recovery4_gate(
        human_qc=HUMAN_QC,
        execute=True,
    )
    rows, overlap = ENGINE.validate_audit_rows(gate)
    if len(rows) != 216 or len(overlap) != 11:
        raise ValueError("exact corrected-core 227 minus 11 audit required")
    by_unit = {
        str(row["unit"]): row
        for row in rows
    }
    topology = load_json(TOPOLOGY)
    production_core = {
        str(row["unit"])
        for row in topology.get("production", [])
        if row.get("lane") == "corrected_core"
    }
    ledger = read_csv(LEDGER)
    waiting = {
        row["unit"]
        for row in ledger
        if row["lane"] == "corrected_core"
        and row["status"] == "WAITING"
    }
    if (
        len(by_unit) != 216
        or len(production_core) != 216
        or set(by_unit) != production_core
        or not waiting <= production_core
    ):
        raise ValueError("corrected-core topology or waiting set differs")
    selected = [by_unit[unit] for unit in sorted(waiting, reverse=True)]
    return gate, selected, topology


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--nthreads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.nthreads <= 8:
        parser.error("--nthreads must be in 1..8")
    gate, selected, topology = preflight()
    validation = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_core_pretract_aux_worker_preflight",
        "status": "PASS",
        "generated_utc": utc_now(),
        "waiting_n": len(selected),
        "target_units": [str(row["unit"]) for row in selected],
        "order": "reverse_lexical",
        "per_subject_lock": "shared Recovery4 exclusive flock",
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "qc_thresholds_changed": False,
        "imaging_executed": False,
        "production_n": topology.get("production_execution_n"),
    }
    if args.validate_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0

    # The standard Recovery4 CLI installs these guards before processing.
    # This imported-engine worker must do the same.  The guard aggregates
    # finite extrema across all FOD coefficient volumes; it does not relax a
    # scalar, FOD, atlas, or spatial-QC threshold.
    ENGINE.install_provisional_engine_guards()

    attempt = (
        ATTEMPTS
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "-aux-reverse"
        )
    )
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_json(attempt / "preflight.json", validation)
    compactor = ENGINE.load_module(
        "hcp379_core_aux_compactor",
        ENGINE.COMPACTOR_SOURCE,
    )
    results: list[dict[str, Any]] = []
    for row in selected:
        unit = str(row["unit"])
        before = current_state(unit)
        if before in {
            "RUNNING_RECOVERY4",
            "RUNNING",
            "FAIL_PRETRACT_RECOVERY4",
        } or (
            before == "PASS_PRETRACT_RECOVERY4" and compacted(unit)
        ):
            result = {
                "unit": unit,
                "status": "SKIP_CURRENT_STATE",
                "state_before": before,
            }
        else:
            state = ENGINE.process_unit(
                row,
                root=ROOT,
                gate=gate,
                nthreads=args.nthreads,
            )
            compact: dict[str, Any] | None = None
            if state.get("status") == "PASS_PRETRACT_RECOVERY4":
                compact = compactor.compact_unit(
                    root=ROOT.resolve(),
                    unit=unit,
                    execute=True,
                )
            result = {
                "unit": unit,
                "status": (
                    "PASS"
                    if state.get("status") == "PASS_PRETRACT_RECOVERY4"
                    and isinstance(compact, dict)
                    and compact.get("status") == "PASS_COMPACTED"
                    else "FAIL"
                ),
                "state_before": before,
                "state_status": state.get("status"),
                "registration_route": state.get("registration_route"),
                "compaction_status": (
                    compact.get("status")
                    if isinstance(compact, dict)
                    else "NOT_RUN"
                ),
                "error": (
                    state.get("error")
                    or (
                        compact.get("error")
                        if isinstance(compact, dict)
                        else ""
                    )
                ),
            }
        results.append(result)
        atomic_json(attempt / "results" / f"{unit}.json", result)
        print(
            f"[{utc_now()}] core_aux {len(results)}/{len(selected)} "
            f"{unit} {result['status']} "
            f"state_before={result['state_before']}",
            flush=True,
        )
        if result["status"] == "FAIL":
            break
    failed = [row for row in results if row["status"] == "FAIL"]
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_core_pretract_aux_worker_invocation",
        "status": "PASS" if not failed else "FAIL",
        "generated_utc": utc_now(),
        "attempt_root": str(attempt.resolve()),
        "selected_n": len(selected),
        "visited_n": len(results),
        "pass_n": sum(row["status"] == "PASS" for row in results),
        "skip_n": sum(
            row["status"] == "SKIP_CURRENT_STATE" for row in results
        ),
        "fail_n": len(failed),
        "results": results,
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "qc_thresholds_changed": False,
    }
    atomic_json(attempt / "invocation.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
