#!/home/ec2-user/fsl/bin/python
"""Run diagnosis-blind affine recovery after both rigid spatial routes fail.

Eligible units come only from the exact 515-unit failure ledger.  Each must
have failed BBR and seeded ANTs rigid registration, have a PASS provisional
image engine, and have passing scalar/FOD QC.  The recovery hash-replays that
engine, preserves prior QC, and evaluates a seeded ANTs affine candidate in a
disjoint path namespace under the unchanged Recovery4 tissue and HCP379 atlas
gates.  It does not use diagnosis, outcomes, connectome density, tractography,
or matrices.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
ENGINE_SOURCE = (
    EXP / "scripts/hcp/hcp379_scaleup_pretract_recovery4_v3.py"
)
LEGACY_WRAPPER_SOURCE = (
    EXP / "scripts/hcp/run_hcp379_legacy_density_pretract_recovery4_v3.py"
)
ARCHIVE_D0_WRAPPER_SOURCE = (
    EXP / "scripts/hcp/run_hcp379_archive_D0_pretract_recovery4_v3.py"
)
LEDGER = (
    EXP
    / "research_audit/outputs/"
    "hcp379_pretract_failure_recovery_ledger_v1/ledger.json"
)
LEDGER_CSV = LEDGER.with_name("ledger.csv")
LEDGER_VALIDATION = LEDGER.with_name("validation.json")
ENGINE_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_scaleup_pretract_recovery4_v3/validation.json"
)
ATTEMPTS = HCP_ROOT / "pretract_failure_recovery_v1/attempts"
DEFAULT_HUMAN_QC = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
CORE_ROOT = HCP_ROOT / "corrected_scaleup_recovery4"
LEGACY_ROOT = HCP_ROOT / "corrected_legacy_tensor_recovery4"
ARCHIVE_D0_ROOT = HCP_ROOT / "corrected_archive_D0_recovery4"
FAILURE_CLASS = "SPATIAL_BBR_AND_RIGID_FAIL"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_module(ENGINE_SOURCE, "hcp379_affine_recovery_core")
LEGACY_WRAPPER = load_module(
    LEGACY_WRAPPER_SOURCE, "hcp379_affine_recovery_legacy_wrapper"
)
LEGACY = LEGACY_WRAPPER.PRETRACT
ARCHIVE_D0_WRAPPER = load_module(
    ARCHIVE_D0_WRAPPER_SOURCE,
    "hcp379_affine_recovery_archive_D0_wrapper",
)
ARCHIVE_D0 = ARCHIVE_D0_WRAPPER.PRETRACT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"regular file required: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def require_validation(
    path: Path, *, expected_checks: int
) -> dict[str, Any]:
    value = load_json(path)
    passed = int(value.get("passed_checks", -1))
    total = int(value.get("total_checks", -1))
    if (
        value.get("status") != "PASS"
        or passed != total
        or total < expected_checks
    ):
        raise ValueError(f"validation differs: {path}")
    return file_record(path)


def lane_rows(
    human_qc: Path,
) -> dict[str, tuple[Any, Path, dict[str, Any], dict[str, str]]]:
    core_gate = CORE.validate_recovery4_gate(
        human_qc=human_qc,
        execute=True,
    )
    core_rows, _ = CORE.validate_audit_rows(core_gate)
    LEGACY_WRAPPER.specialize(LEGACY_ROOT)
    legacy_gate = LEGACY.validate_recovery4_gate(
        human_qc=human_qc,
        execute=True,
    )
    legacy_rows, _ = LEGACY.validate_audit_rows(legacy_gate)
    ARCHIVE_D0_WRAPPER.specialize(ARCHIVE_D0_ROOT)
    archive_d0_gate = ARCHIVE_D0.validate_recovery4_gate(
        human_qc=human_qc,
        execute=True,
    )
    archive_d0_rows, _ = ARCHIVE_D0.validate_audit_rows(archive_d0_gate)
    result: dict[
        str, tuple[Any, Path, dict[str, Any], dict[str, str]]
    ] = {}
    for module, root, gate, rows in (
        (CORE, CORE_ROOT, core_gate, core_rows),
        (LEGACY, LEGACY_ROOT, legacy_gate, legacy_rows),
        (
            ARCHIVE_D0,
            ARCHIVE_D0_ROOT,
            archive_d0_gate,
            archive_d0_rows,
        ),
    ):
        for row in rows:
            unit = str(row["unit"])
            if unit in result:
                raise ValueError(f"cross-lane duplicate: {unit}")
            result[unit] = (module, root, gate, row)
    return result


def exact_targets(
    human_qc: Path,
) -> tuple[
    list[tuple[Any, Path, dict[str, Any], dict[str, str]]],
    dict[str, Any],
]:
    require_validation(LEDGER_VALIDATION, expected_checks=6)
    require_validation(ENGINE_VALIDATION, expected_checks=19)
    ledger = load_json(LEDGER)
    failed = ledger.get("failed_units")
    if not isinstance(failed, list):
        raise ValueError("failure ledger differs")
    records = ledger.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("failure-ledger records differ")
    if records.get("ledger_csv") != file_record(LEDGER_CSV):
        raise ValueError("failure-ledger CSV binding differs")
    with LEDGER_CSV.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        ledger_rows = list(csv.DictReader(handle))
    ledger_units = [str(row.get("unit", "")) for row in ledger_rows]
    if (
        len(ledger_rows) != 515
        or len(set(ledger_units)) != 515
        or "" in ledger_units
    ):
        raise ValueError("failure-ledger CSV identity differs")
    failed_target_units = {
        str(row["unit"])
        for row in failed
        if row.get("failure_class") == FAILURE_CLASS
    }
    held_target_units = {
        str(row["unit"])
        for row in ledger_rows
        if (
            row.get("status") == "WAITING"
            and row.get("subject_state_status")
            == "HOLD_MECHANISM_SPECIFIC_RECOVERY_REQUIRED"
            and row.get("error")
            == (
                "Fail-closed mechanism-specific hold: "
                f"{FAILURE_CLASS}"
            )
        )
    }
    target_units = sorted(failed_target_units | held_target_units)
    available = lane_rows(human_qc)
    targets = []
    for unit in target_units:
        if unit not in available:
            raise ValueError(
                "affine target outside core/legacy/archive-D0 lanes: "
                f"{unit}"
            )
        module, root, gate, row = available[unit]
        state_path = root / "qc/subjects" / f"{unit}.json"
        state = load_json(state_path)
        failure_state = state
        if (
            state.get("status")
            == "HOLD_MECHANISM_SPECIFIC_RECOVERY_REQUIRED"
        ):
            if (
                unit not in held_target_units
                or state.get("failure_class") != FAILURE_CLASS
            ):
                raise ValueError(
                    f"affine recovery hold class differs: {unit}"
                )
            archive_record = state.get("prior_state_archive")
            if not isinstance(archive_record, Mapping):
                raise ValueError(
                    f"affine recovery hold lacks prior failure: {unit}"
                )
            archive_path = Path(
                str(archive_record.get("path", ""))
            ).resolve()
            if file_record(archive_path) != {
                "path": str(archive_path),
                "size_bytes": int(
                    archive_record.get("size_bytes", -1)
                ),
                "sha256": str(archive_record.get("sha256", "")),
            }:
                raise ValueError(
                    f"affine prior-failure archive differs: {unit}"
                )
            archive = load_json(archive_path)
            prior = archive.get("prior_state")
            if (
                archive.get("record_type")
                != "hcp379_pretract_recovery4_prior_state_archive"
                or archive.get("unit") != unit
                or not isinstance(prior, Mapping)
            ):
                raise ValueError(
                    f"affine prior-failure archive contract differs: {unit}"
                )
            failure_state = dict(prior)
        scalar_path = (
            root
            / "subjects"
            / unit
            / "06_preflight/scalar_recovery4_qc.json"
        )
        scalar = load_json(scalar_path)
        if (
            failure_state.get("status") != "FAIL_PRETRACT_RECOVERY4"
            or "neither spatial route passed direct gates"
            not in str(failure_state.get("error", ""))
            or scalar.get("status") != "PASS"
        ):
            raise ValueError(f"affine recovery state differs: {unit}")
        module.validate_reusable_pass_engine_state(
            root
            / "subjects"
            / unit
            / "06_preflight/provisional_v2_engine_state.json",
            unit=unit,
        )
        targets.append((module, root, gate, row))
    return targets, ledger


def preserve_prior_qc(
    attempt: Path, root: Path, unit: str
) -> dict[str, Any]:
    archived: dict[str, Any] = {}
    sources = {
        "state": root / "qc/subjects" / f"{unit}.json",
        "scalar_qc": (
            root
            / "subjects"
            / unit
            / "06_preflight/scalar_recovery4_qc.json"
        ),
    }
    for name, source in sources.items():
        record = file_record(source)
        destination = (
            attempt
            / "prior_qc"
            / f"{unit}-{name}-{record['sha256']}.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes()
        if destination.exists():
            if destination.read_bytes() != raw:
                raise ValueError(f"prior QC archive collision: {unit}:{name}")
        else:
            with destination.open("xb") as handle:
                handle.write(raw)
        copy = file_record(destination)
        if copy["sha256"] != record["sha256"]:
            raise ValueError(f"prior QC archive hash differs: {unit}:{name}")
        archived[name] = {"source": record, "archive": copy}
    return archived


def preflight(human_qc: Path) -> dict[str, Any]:
    targets, ledger = exact_targets(human_qc)
    units = [str(row["unit"]) for _, _, _, row in targets]
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_affine_pretract_recovery_preflight",
        "status": (
            "READY_FOR_EXACT_AFFINE_RECOVERY"
            if targets
            else "NO_CURRENT_AFFINE_RECOVERY_TARGETS"
        ),
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "imaging_executed": False,
        "tractography_generated": False,
        "matrix_generated": False,
        "failure_class": FAILURE_CLASS,
        "target_n": len(units),
        "target_units": units,
        "policy": {
            "selection": (
                "only after BBR and seeded rigid direct QC both fail"
            ),
            "transform": "seeded_ants_affine",
            "ants_transform_type": "a",
            "per_subject_tuning_allowed": False,
            "engine_recomputation_allowed": False,
            "tissue_qc_threshold_change_allowed": False,
            "atlas_qc_threshold_change_allowed": False,
            "visual_qc_required": True,
            "mechanism_guard_authorization": (
                "only the exact current ledger target whose failure "
                "class equals SPATIAL_BBR_AND_RIGID_FAIL"
            ),
        },
        "records": {
            "ledger": file_record(LEDGER),
            "ledger_csv": file_record(LEDGER_CSV),
            "ledger_validation": file_record(LEDGER_VALIDATION),
            "engine_validation": file_record(ENGINE_VALIDATION),
            "runner": file_record(Path(__file__)),
            "engine": file_record(ENGINE_SOURCE),
            "legacy_wrapper": file_record(LEGACY_WRAPPER_SOURCE),
            "archive_D0_wrapper": file_record(
                ARCHIVE_D0_WRAPPER_SOURCE
            ),
            "human_qc": file_record(human_qc),
        },
        "ledger_status_counts": ledger.get("status_counts"),
    }


def execute(human_qc: Path, *, nthreads: int) -> dict[str, Any]:
    pre = preflight(human_qc)
    if pre["status"] != "READY_FOR_EXACT_AFFINE_RECOVERY":
        return pre
    attempt = (
        ATTEMPTS
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "-affine-pretract-recovery"
        )
    )
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_json(attempt / "preflight.json", pre)
    targets, _ = exact_targets(human_qc)
    results: list[dict[str, Any]] = []
    touched_modules: dict[Path, Any] = {}
    for module, root, gate, row in targets:
        unit = str(row["unit"])
        prior = preserve_prior_qc(attempt, root, unit)
        holds = gate.get("mechanism_hold_by_unit")
        if not isinstance(holds, Mapping):
            raise TypeError("mechanism hold map differs")
        state_path = root / "qc/subjects" / f"{unit}.json"
        held_state = load_json(state_path)
        prior_archive = held_state.get("prior_state_archive")
        if (
            held_state.get("status")
            != "HOLD_MECHANISM_SPECIFIC_RECOVERY_REQUIRED"
            or held_state.get("failure_class") != FAILURE_CLASS
            or not isinstance(prior_archive, Mapping)
        ):
            raise ValueError(
                f"affine mechanism authorization differs: {unit}"
            )
        guard_hold = holds.get(unit)
        if guard_hold is not None and (
            not isinstance(guard_hold, Mapping)
            or guard_hold.get("failure_class") != FAILURE_CLASS
        ):
            raise ValueError(
                f"affine in-memory mechanism hold differs: {unit}"
            )
        authorized_hold = {
            "authorization_source": (
                "validated_515_row_ledger_plus_hash_bound_held_state"
            ),
            "failure_class": FAILURE_CLASS,
            "subject_state": file_record(state_path),
            "prior_state_archive": dict(prior_archive),
            "ledger_csv": file_record(LEDGER_CSV),
            "in_memory_guard_entry_present": guard_hold is not None,
        }
        recovery_gate = dict(gate)
        recovery_gate["mechanism_hold_by_unit"] = {
            key: value
            for key, value in holds.items()
            if key != unit
        }
        state = module.process_unit(
            row,
            root=root,
            gate=recovery_gate,
            nthreads=nthreads,
            reuse_valid_engine=True,
            allow_affine_failover=True,
        )
        compaction: dict[str, Any] | None = None
        if state.get("status") == "PASS_PRETRACT_RECOVERY4":
            compactor = module.load_module(
                f"hcp379_affine_recovery_compactor_{unit}",
                module.COMPACTOR_SOURCE,
            )
            compaction = compactor.compact_unit(
                root=root.resolve(),
                unit=unit,
                execute=True,
            )
        result = {
            "unit": unit,
            "root": str(root.resolve()),
            "state": state,
            "compaction": compaction,
            "prior_qc": prior,
            "mechanism_guard_authorization": dict(
                authorized_hold
            ),
        }
        results.append(result)
        atomic_json(attempt / "results" / f"{unit}.json", result)
        touched_modules[root] = module
    lane_summaries = {
        str(root): module.write_summary(root)
        for root, module in touched_modules.items()
    }
    passed = all(
        row["state"].get("status") == "PASS_PRETRACT_RECOVERY4"
        and row["state"].get("registration_route")
        == "seeded_ants_affine_recovery"
        and isinstance(row["compaction"], dict)
        and row["compaction"].get("status") == "PASS_COMPACTED"
        for row in results
    )
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_affine_pretract_recovery_attempt",
        "status": "PASS" if passed else "FAIL",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "engine_recomputed": False,
        "target_n": len(targets),
        "pass_n": sum(
            row["state"].get("status") == "PASS_PRETRACT_RECOVERY4"
            and row["state"].get("registration_route")
            == "seeded_ants_affine_recovery"
            for row in results
        ),
        "units": [str(row["unit"]) for _, _, _, row in targets],
        "results": results,
        "lane_summaries": lane_summaries,
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
        if FAILURE_CLASS != "SPATIAL_BBR_AND_RIGID_FAIL":
            raise AssertionError(FAILURE_CLASS)
        print("HCP379_AFFINE_PRETRACT_RECOVERY_SELF_TEST_PASS")
        return 0
    payload = (
        execute(args.human_qc_manifest, nthreads=args.nthreads)
        if args.execute
        else preflight(args.human_qc_manifest)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") in {
        "READY_FOR_EXACT_AFFINE_RECOVERY",
        "NO_CURRENT_AFFINE_RECOVERY_TARGETS",
        "PASS",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
