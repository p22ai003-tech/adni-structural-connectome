#!/home/ec2-user/fsl/bin/python
"""Run deterministic SyN recovery after BBR, rigid and affine all fail.

Targets come only from the exact 515-unit failure ledger and must carry a
failed affine recovery record.  The completed image engine is hash-replayed,
prior state/QC is preserved, and a seeded ANTs SyN candidate is written to a
disjoint namespace.  It must pass the unchanged Recovery4 tissue and HCP379
atlas gates plus a strictly positive Jacobian-determinant gate.  Diagnosis,
outcomes, connectome density, tractography and matrices are not used.
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


CORE = load_module(ENGINE_SOURCE, "hcp379_syn_recovery_core")
LEGACY_WRAPPER = load_module(
    LEGACY_WRAPPER_SOURCE, "hcp379_syn_recovery_legacy_wrapper"
)
LEGACY = LEGACY_WRAPPER.PRETRACT
ARCHIVE_D0_WRAPPER = load_module(
    ARCHIVE_D0_WRAPPER_SOURCE,
    "hcp379_syn_recovery_archive_D0_wrapper",
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


def require_validation(path: Path, expected_checks: int) -> None:
    value = load_json(path)
    if (
        value.get("status") != "PASS"
        or value.get("passed_checks") != expected_checks
        or value.get("total_checks") != expected_checks
    ):
        raise ValueError(f"validation differs: {path}")


def lane_rows(
    human_qc: Path,
) -> dict[str, tuple[Any, Path, dict[str, Any], dict[str, str]]]:
    core_gate = CORE.validate_recovery4_gate(
        human_qc=human_qc, execute=True
    )
    core_rows, _ = CORE.validate_audit_rows(core_gate)
    LEGACY_WRAPPER.specialize(LEGACY_ROOT)
    legacy_gate = LEGACY.validate_recovery4_gate(
        human_qc=human_qc, execute=True
    )
    legacy_rows, _ = LEGACY.validate_audit_rows(legacy_gate)
    ARCHIVE_D0_WRAPPER.specialize(ARCHIVE_D0_ROOT)
    archive_d0_gate = ARCHIVE_D0.validate_recovery4_gate(
        human_qc=human_qc, execute=True
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
) -> list[tuple[Any, Path, dict[str, Any], dict[str, str]]]:
    require_validation(LEDGER_VALIDATION, 6)
    require_validation(ENGINE_VALIDATION, 19)
    ledger = load_json(LEDGER)
    failed = ledger.get("failed_units")
    if not isinstance(failed, list):
        raise ValueError("failure ledger differs")
    units = sorted(
        str(row["unit"])
        for row in failed
        if row.get("failure_class") == FAILURE_CLASS
    )
    available = lane_rows(human_qc)
    targets = []
    for unit in units:
        if unit not in available:
            raise ValueError(
                "SyN target outside core/legacy/archive-D0 lanes: "
                f"{unit}"
            )
        module, root, gate, row = available[unit]
        state = load_json(root / "qc/subjects" / f"{unit}.json")
        scalar = load_json(
            root
            / "subjects"
            / unit
            / "06_preflight/scalar_recovery4_qc.json"
        )
        error = str(state.get("error", ""))
        if (
            state.get("status") != "FAIL_PRETRACT_RECOVERY4"
            or "neither spatial route passed direct gates" not in error
            or "affine=[" not in error
            or scalar.get("status") != "PASS"
        ):
            raise ValueError(f"SyN recovery state differs: {unit}")
        module.validate_reusable_pass_engine_state(
            root
            / "subjects"
            / unit
            / "06_preflight/provisional_v2_engine_state.json",
            unit=unit,
        )
        targets.append((module, root, gate, row))
    return targets


def preserve_prior_qc(
    attempt: Path, root: Path, unit: str
) -> dict[str, Any]:
    preserved: dict[str, Any] = {}
    for name, source in {
        "state": root / "qc/subjects" / f"{unit}.json",
        "scalar_qc": (
            root
            / "subjects"
            / unit
            / "06_preflight/scalar_recovery4_qc.json"
        ),
        "affine_qc": (
            root
            / "subjects"
            / unit
            / "06_preflight/ants_affine_hcp379_atlas_qc_recovery4.json"
        ),
    }.items():
        record = file_record(source)
        destination = (
            attempt
            / "prior_qc"
            / f"{unit}-{name}-{record['sha256']}.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(source.read_bytes())
        copy = file_record(destination)
        if copy["sha256"] != record["sha256"]:
            raise ValueError(f"prior QC archive differs: {unit}:{name}")
        preserved[name] = {"source": record, "archive": copy}
    return preserved


def preflight(human_qc: Path) -> dict[str, Any]:
    targets = exact_targets(human_qc)
    units = [str(row["unit"]) for _, _, _, row in targets]
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_syn_pretract_recovery_preflight",
        "status": (
            "READY_FOR_EXACT_SYN_RECOVERY"
            if targets
            else "NO_CURRENT_SYN_RECOVERY_TARGETS"
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
                "only after BBR, seeded rigid and seeded affine fail"
            ),
            "transform": "seeded_ants_syn",
            "ants_transform_type": "s",
            "random_seed": 1234,
            "all_jacobian_determinants_must_be_positive": True,
            "per_subject_tuning_allowed": False,
            "engine_recomputation_allowed": False,
            "tissue_qc_threshold_change_allowed": False,
            "atlas_qc_threshold_change_allowed": False,
            "visual_qc_required": True,
        },
        "records": {
            "ledger": file_record(LEDGER),
            "ledger_validation": file_record(LEDGER_VALIDATION),
            "engine_validation": file_record(ENGINE_VALIDATION),
            "runner": file_record(Path(__file__)),
            "engine": file_record(ENGINE_SOURCE),
            "legacy_wrapper": file_record(LEGACY_WRAPPER_SOURCE),
            "archive_D0_wrapper": file_record(
                ARCHIVE_D0_WRAPPER_SOURCE
            ),
            "human_qc": file_record(human_qc),
            "jacobian_binary": file_record(
                CORE.ANTS / "CreateJacobianDeterminantImage"
            ),
        },
    }


def execute(human_qc: Path, nthreads: int) -> dict[str, Any]:
    pre = preflight(human_qc)
    if pre["status"] != "READY_FOR_EXACT_SYN_RECOVERY":
        return pre
    attempt = (
        ATTEMPTS
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "-syn-pretract-recovery"
        )
    )
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_json(attempt / "preflight.json", pre)
    targets = exact_targets(human_qc)
    results: list[dict[str, Any]] = []
    touched: dict[Path, Any] = {}
    for module, root, gate, row in targets:
        unit = str(row["unit"])
        prior = preserve_prior_qc(attempt, root, unit)
        state = module.process_unit(
            row,
            root=root,
            gate=gate,
            nthreads=nthreads,
            reuse_valid_engine=True,
            allow_affine_failover=True,
            allow_syn_failover=True,
        )
        compaction: dict[str, Any] | None = None
        if state.get("status") == "PASS_PRETRACT_RECOVERY4":
            compactor = module.load_module(
                f"hcp379_syn_recovery_compactor_{unit}",
                module.COMPACTOR_SOURCE,
            )
            compaction = compactor.compact_unit(
                root=root.resolve(), unit=unit, execute=True
            )
        route_qc_path = (
            root
            / "subjects"
            / unit
            / "06_preflight/spatial_route_recovery4_qc.json"
        )
        route_qc = (
            load_json(route_qc_path) if route_qc_path.is_file() else None
        )
        result = {
            "unit": unit,
            "root": str(root.resolve()),
            "state": state,
            "route_qc": route_qc,
            "compaction": compaction,
            "prior_qc": prior,
        }
        results.append(result)
        atomic_json(attempt / "results" / f"{unit}.json", result)
        touched[root] = module
    lane_summaries = {
        str(root): module.write_summary(root)
        for root, module in touched.items()
    }
    passed = all(
        row["state"].get("status") == "PASS_PRETRACT_RECOVERY4"
        and row["state"].get("registration_route")
        == "seeded_ants_syn_recovery"
        and isinstance(row["compaction"], dict)
        and row["compaction"].get("status") == "PASS_COMPACTED"
        and (
            row["route_qc"]["candidates"]["seeded_ants_syn_recovery"][
                "deformation_qc"
            ]["all_jacobians_positive"]
            is True
        )
        for row in results
    )
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_syn_pretract_recovery_attempt",
        "status": "PASS" if passed else "FAIL",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "engine_recomputed": False,
        "target_n": len(targets),
        "pass_n": sum(
            row["state"].get("registration_route")
            == "seeded_ants_syn_recovery"
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
        print("HCP379_SYN_PRETRACT_RECOVERY_SELF_TEST_PASS")
        return 0
    payload = (
        execute(args.human_qc_manifest, args.nthreads)
        if args.execute
        else preflight(args.human_qc_manifest)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") in {
        "READY_FOR_EXACT_SYN_RECOVERY",
        "NO_CURRENT_SYN_RECOVERY_TARGETS",
        "PASS",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
