#!/home/ec2-user/fsl/bin/python
"""Replay exact core failures caused by a missing multi-volume FOD range guard.

The standard Recovery4 CLI installs an aggregate mrstats parser before calling
the provisional image engine.  A separately imported auxiliary worker omitted
that installation, so a complete multi-coefficient WM-FOD was incorrectly
rejected by the legacy single-pair parser.  This recovery is restricted to the
exact ledger class, preserves the failed records, installs the already
validated guard, and replays the unchanged core engine plus Recovery4 QC.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP = Path("/data/derivatives/hcp379_v2")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
SOURCE = EXP / "scripts/hcp/hcp379_scaleup_pretract_recovery4_v3.py"
ROOT = HCP / "corrected_scaleup_recovery4"
LEDGER = (
    EXP
    / "research_audit/outputs/hcp379_pretract_failure_recovery_ledger_v1/"
    "ledger.json"
)
LEDGER_VALIDATION = LEDGER.with_name("validation.json")
ENGINE_VALIDATION = (
    EXP
    / "research_audit/outputs/hcp379_scaleup_pretract_recovery4_v3/"
    "validation.json"
)
DEFAULT_HUMAN_QC = (
    HCP
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
ATTEMPTS = HCP / "pretract_failure_recovery_v1/attempts"
FAILURE_CLASS = "PROVISIONAL_ENGINE_FOD_RANGE_GUARD_MISSING"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = load_module(SOURCE, "hcp379_engine_fod_range_guard_recovery")


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
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
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


def require_validation(path: Path, expected: int) -> dict[str, Any]:
    value = load_json(path)
    if (
        value.get("status") != "PASS"
        or value.get("passed_checks") != expected
        or value.get("total_checks") != expected
    ):
        raise ValueError(f"validation differs: {path}")
    return file_record(path)


def exact_targets(
    human_qc: Path,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    require_validation(LEDGER_VALIDATION, 6)
    require_validation(ENGINE_VALIDATION, 19)
    ledger = load_json(LEDGER)
    targets = [
        row
        for row in ledger.get("failed_units", [])
        if row.get("failure_class") == FAILURE_CLASS
    ]
    if not targets:
        return [], {}, ledger
    if any(row.get("lane") != "corrected_core" for row in targets):
        raise ValueError("range-guard recovery target is outside core lane")
    gate = ENGINE.validate_recovery4_gate(
        human_qc=human_qc,
        execute=True,
    )
    rows, overlap = ENGINE.validate_audit_rows(gate)
    by_unit = {str(row["unit"]): row for row in rows}
    if any(str(row["unit"]) in overlap for row in targets):
        raise ValueError("range-guard target is a promoted canary")
    selected: list[dict[str, str]] = []
    for target in sorted(targets, key=lambda row: str(row["unit"])):
        unit = str(target["unit"])
        if unit not in by_unit:
            raise ValueError(f"range-guard target is outside core audit: {unit}")
        state = load_json(ROOT / "qc/subjects" / f"{unit}.json")
        engine_state = load_json(
            ROOT
            / "subjects"
            / unit
            / "06_preflight/provisional_v2_engine_state.json"
        )
        state_error = str(state.get("error", "")).lower()
        engine_error = str(engine_state.get("error", "")).lower()
        if (
            state.get("status") != "FAIL_PRETRACT_RECOVERY4"
            or engine_state.get("status") != "FAIL_PRETRACT"
            or "invalid mrstats range" not in state_error
            or "wmfod_norm.mif" not in state_error
            or "invalid mrstats range" not in engine_error
            or "wmfod_norm.mif" not in engine_error
        ):
            raise ValueError(f"range-guard failure state differs: {unit}")
        selected.append(by_unit[unit])
    return selected, gate, ledger


def preflight(human_qc: Path) -> dict[str, Any]:
    selected, _, ledger = exact_targets(human_qc)
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_engine_fod_range_guard_recovery_preflight"
        ),
        "status": (
            "READY_FOR_EXACT_ENGINE_GUARD_REPLAY"
            if selected
            else "NO_CURRENT_ENGINE_GUARD_TARGETS"
        ),
        "generated_utc": utc_now(),
        "failure_class": FAILURE_CLASS,
        "target_n": len(selected),
        "target_units": [str(row["unit"]) for row in selected],
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "historical_outputs_modified": False,
        "imaging_executed": False,
        "tractography_generated": False,
        "matrix_generated": False,
        "qc_thresholds_changed": False,
        "guard": {
            "purpose": "aggregate finite extrema across FOD volumes",
            "engine_recipe_changed": False,
            "recovery4_threshold_changed": False,
            "subject_specific_tuning": False,
        },
        "ledger_status_counts": ledger.get("status_counts"),
        "records": {
            "ledger": file_record(LEDGER),
            "ledger_validation": file_record(LEDGER_VALIDATION),
            "engine": file_record(SOURCE),
            "engine_validation": file_record(ENGINE_VALIDATION),
            "human_qc": file_record(human_qc),
            "runner": file_record(Path(__file__)),
        },
    }


def preserve(path: Path, destination: Path) -> dict[str, Any]:
    source = file_record(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = path.read_bytes()
    if destination.exists():
        if destination.read_bytes() != raw:
            raise ValueError(f"preservation collision: {destination}")
    else:
        with destination.open("xb") as handle:
            handle.write(raw)
    archived = file_record(destination)
    if source["sha256"] != archived["sha256"]:
        raise ValueError(f"preserved hash differs: {path}")
    return {"source": source, "archive": archived}


def execute(human_qc: Path, *, nthreads: int) -> dict[str, Any]:
    pre = preflight(human_qc)
    if pre["status"] != "READY_FOR_EXACT_ENGINE_GUARD_REPLAY":
        return pre
    attempt = (
        ATTEMPTS
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "-engine-fod-range-guard-recovery"
        )
    )
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_json(attempt / "preflight.json", pre)
    selected, gate, _ = exact_targets(human_qc)
    ENGINE.install_provisional_engine_guards()
    compactor = ENGINE.load_module(
        "hcp379_engine_guard_recovery_compactor",
        ENGINE.COMPACTOR_SOURCE,
    )
    results: list[dict[str, Any]] = []
    for row in selected:
        unit = str(row["unit"])
        state_path = ROOT / "qc/subjects" / f"{unit}.json"
        engine_path = (
            ROOT
            / "subjects"
            / unit
            / "06_preflight/provisional_v2_engine_state.json"
        )
        prior = {
            "subject_state": preserve(
                state_path,
                attempt
                / "prior_states"
                / f"{unit}-subject-{sha256(state_path)}.json",
            ),
            "engine_state": preserve(
                engine_path,
                attempt
                / "prior_states"
                / f"{unit}-engine-{sha256(engine_path)}.json",
            ),
        }
        state = ENGINE.process_unit(
            row,
            root=ROOT,
            gate=gate,
            nthreads=nthreads,
            reuse_valid_engine=False,
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
            "state": state,
            "compaction": compact,
            "preserved_failed_records": prior,
        }
        results.append(result)
        atomic_json(attempt / "results" / f"{unit}.json", result)
    passed = all(
        row["state"].get("status") == "PASS_PRETRACT_RECOVERY4"
        and isinstance(row["compaction"], Mapping)
        and row["compaction"].get("status") == "PASS_COMPACTED"
        for row in results
    )
    payload = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_engine_fod_range_guard_recovery_attempt"
        ),
        "status": "PASS" if passed else "FAIL",
        "generated_utc": utc_now(),
        "attempt_root": str(attempt.resolve()),
        "target_n": len(results),
        "pass_n": sum(
            row["state"].get("status") == "PASS_PRETRACT_RECOVERY4"
            for row in results
        ),
        "units": [row["unit"] for row in results],
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "historical_outputs_modified": False,
        "qc_thresholds_changed": False,
        "tractography_generated": False,
        "matrix_generated": False,
        "results": results,
        "lane_summary": ENGINE.write_summary(ROOT),
        "preflight": file_record(attempt / "preflight.json"),
    }
    atomic_json(attempt / "attempt.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--human-qc-manifest",
        type=Path,
        default=DEFAULT_HUMAN_QC,
    )
    parser.add_argument("--nthreads", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.nthreads <= 16:
        parser.error("--nthreads must be in 1..16")
    if args.self_test:
        if FAILURE_CLASS != "PROVISIONAL_ENGINE_FOD_RANGE_GUARD_MISSING":
            raise AssertionError(FAILURE_CLASS)
        print("HCP379_ENGINE_FOD_RANGE_GUARD_RECOVERY_SELF_TEST_PASS")
        return 0
    value = (
        execute(args.human_qc_manifest, nthreads=args.nthreads)
        if args.execute
        else preflight(args.human_qc_manifest)
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value.get("status") in {
        "READY_FOR_EXACT_ENGINE_GUARD_REPLAY",
        "NO_CURRENT_ENGINE_GUARD_TARGETS",
        "PASS",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
