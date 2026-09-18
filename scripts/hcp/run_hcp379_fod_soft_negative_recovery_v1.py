#!/home/ec2-user/fsl/bin/python
"""Recover exact Recovery4 FOD soft-negative failures without recomputation.

Only units currently classified as FOD_L0_SOFT_NEGATIVE_FRACTION by the exact
515-unit failure ledger are eligible.  A passing provisional image engine is
hash-replayed, the cohort-uniform FOD numerical policy is rerun, and the
unchanged Recovery4 spatial gates are applied.  Failed states and scalar-QC
records are content-addressed before replacement.  No diagnosis, outcome,
connectome density, tractography, or matrix is read or generated.
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
ARCHIVE_WRAPPER_SOURCE = (
    EXP / "scripts/hcp/run_hcp379_archive_D0_pretract_recovery4_v3.py"
)
LEGACY_WRAPPER_SOURCE = (
    EXP
    / "scripts/hcp/run_hcp379_legacy_density_pretract_recovery4_v3.py"
)
ENGINE_SOURCE = (
    EXP / "scripts/hcp/hcp379_scaleup_pretract_recovery4_v3.py"
)
LEDGER = (
    EXP
    / "research_audit/outputs/"
    "hcp379_pretract_failure_recovery_ledger_v1/ledger.json"
)
LEDGER_VALIDATION = LEDGER.with_name("validation.json")
FOD_POLICY_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_fod_numerical_policy_v2/validation.json"
)
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
FAILURE_CLASS = "FOD_L0_SOFT_NEGATIVE_FRACTION"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARCHIVE_WRAPPER = load_module(
    ARCHIVE_WRAPPER_SOURCE, "hcp379_fod_recovery_archive_wrapper"
)
LEGACY_WRAPPER = load_module(
    LEGACY_WRAPPER_SOURCE, "hcp379_fod_recovery_legacy_wrapper"
)
CORE_ENGINE = load_module(
    ENGINE_SOURCE, "hcp379_fod_recovery_core_engine"
)
LANES = {
    "corrected_core": {
        "engine": CORE_ENGINE,
        "root": HCP_ROOT / "corrected_scaleup_recovery4",
        "wrapper": None,
        "wrapper_source": ENGINE_SOURCE,
    },
    "corrected_legacy_tensor": {
        "engine": LEGACY_WRAPPER.PRETRACT,
        "root": HCP_ROOT / "corrected_legacy_tensor_recovery4",
        "wrapper": LEGACY_WRAPPER,
        "wrapper_source": LEGACY_WRAPPER_SOURCE,
    },
    "corrected_archive_D0": {
        "engine": ARCHIVE_WRAPPER.PRETRACT,
        "root": HCP_ROOT / "corrected_archive_D0_recovery4",
        "wrapper": ARCHIVE_WRAPPER,
        "wrapper_source": ARCHIVE_WRAPPER_SOURCE,
    },
}


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


def exact_targets(
    *,
    human_qc: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require_validation(LEDGER_VALIDATION, expected_checks=6)
    require_validation(FOD_POLICY_VALIDATION, expected_checks=4)
    require_validation(ENGINE_VALIDATION, expected_checks=19)
    ledger = load_json(LEDGER)
    failed = ledger.get("failed_units")
    if not isinstance(failed, list):
        raise ValueError("failure ledger differs")
    target_units = sorted(
        str(row["unit"])
        for row in failed
        if row.get("failure_class") == FAILURE_CLASS
    )
    if not target_units:
        return [], ledger

    failed_by_unit = {
        str(row["unit"]): row
        for row in failed
        if row.get("failure_class") == FAILURE_CLASS
    }
    selected: list[dict[str, Any]] = []
    contexts: dict[str, dict[str, Any]] = {}
    for lane in sorted({str(failed_by_unit[unit]["lane"]) for unit in target_units}):
        if lane not in LANES:
            raise ValueError(f"FOD recovery lane is unsupported: {lane}")
        context = LANES[lane]
        engine = context["engine"]
        wrapper = context["wrapper"]
        root = context["root"]
        if wrapper is not None:
            wrapper.specialize(root)
        gate = engine.validate_recovery4_gate(
            human_qc=human_qc,
            execute=True,
        )
        rows, _ = engine.validate_audit_rows(gate)
        contexts[lane] = {
            **context,
            "gate": gate,
            "by_unit": {str(row["unit"]): row for row in rows},
        }
    for unit in target_units:
        lane = str(failed_by_unit[unit]["lane"])
        context = contexts[lane]
        if unit not in context["by_unit"]:
            raise ValueError(f"FOD recovery target is outside {lane}: {unit}")
        row = context["by_unit"][unit]
        root = context["root"]
        engine = context["engine"]
        state_path = root / "qc/subjects" / f"{unit}.json"
        state = load_json(state_path)
        error = str(state.get("error", "")).lower()
        if (
            state.get("status") != "FAIL_PRETRACT_RECOVERY4"
            or not (
                "wmfod_l0_below_tolerance" in error
                or (
                    "invalid mrstats range" in error
                    and "wmfod_norm.mif" in error
                )
            )
        ):
            raise ValueError(f"current failure state differs: {unit}")
        engine_path = (
            root
            / "subjects"
            / unit
            / "06_preflight/provisional_v2_engine_state.json"
        )
        engine.validate_reusable_pass_engine_state(
            engine_path,
            unit=unit,
        )
        selected.append(
            {
                "unit": unit,
                "lane": lane,
                "row": row,
                "root": root,
                "engine": engine,
                "gate": context["gate"],
                "wrapper_source": context["wrapper_source"],
            }
        )
    return selected, ledger


def preserve_prior_qc(
    attempt: Path, unit: str, *, root: Path
) -> dict[str, Any]:
    source = (
        root
        / "subjects"
        / unit
        / "06_preflight/scalar_recovery4_qc.json"
    )
    record = file_record(source)
    destination = (
        attempt
        / "prior_scalar_qc"
        / f"{unit}-{record['sha256']}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != raw:
            raise ValueError(f"prior QC archive collision: {unit}")
    else:
        with destination.open("xb") as handle:
            handle.write(raw)
    archived = file_record(destination)
    if archived["sha256"] != record["sha256"]:
        raise ValueError(f"prior QC archive hash differs: {unit}")
    return {"source": record, "archive": archived}


def preflight(human_qc: Path) -> dict[str, Any]:
    targets, ledger = exact_targets(human_qc=human_qc)
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_fod_soft_negative_recovery_preflight",
        "status": (
            "READY_FOR_EXACT_FOD_RECOVERY"
            if targets
            else "NO_CURRENT_FOD_RECOVERY_TARGETS"
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
        "target_n": len(targets),
        "target_units": [str(row["unit"]) for row in targets],
        "target_lanes": {
            str(row["unit"]): str(row["lane"]) for row in targets
        },
        "policy": {
            "soft_tolerance": 1.0e-5,
            "hard_negative_tolerance": 1.0e-4,
            "maximum_soft_negative_fraction": 1.0e-3,
            "per_subject_tuning_allowed": False,
            "engine_recomputation_allowed": False,
            "spatial_qc_threshold_change_allowed": False,
        },
        "records": {
            "ledger": file_record(LEDGER),
            "ledger_validation": file_record(LEDGER_VALIDATION),
            "fod_policy_validation": file_record(FOD_POLICY_VALIDATION),
            "engine_validation": file_record(ENGINE_VALIDATION),
            "runner": file_record(Path(__file__)),
            "engine": file_record(ENGINE_SOURCE),
            "lane_wrappers": {
                lane: file_record(Path(str(context["wrapper_source"])))
                for lane, context in LANES.items()
            },
            "human_qc": file_record(human_qc),
        },
        "ledger_status_counts": ledger.get("status_counts"),
    }


def execute(human_qc: Path, *, nthreads: int) -> dict[str, Any]:
    pre = preflight(human_qc)
    if pre["status"] != "READY_FOR_EXACT_FOD_RECOVERY":
        return pre
    attempt = (
        ATTEMPTS
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "-fod-soft-negative-recovery"
        )
    )
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_json(attempt / "preflight.json", pre)
    targets, _ = exact_targets(human_qc=human_qc)
    results: list[dict[str, Any]] = []
    lane_summaries: dict[str, Any] = {}
    for target in targets:
        unit = str(target["unit"])
        lane = str(target["lane"])
        root = target["root"]
        engine = target["engine"]
        prior_qc = preserve_prior_qc(attempt, unit, root=root)
        state = engine.process_unit(
            target["row"],
            root=root,
            gate=target["gate"],
            nthreads=nthreads,
            reuse_valid_engine=True,
        )
        compaction: dict[str, Any] | None = None
        if state.get("status") == "PASS_PRETRACT_RECOVERY4":
            compactor = engine.load_module(
                f"hcp379_fod_recovery_compactor_{lane}",
                engine.COMPACTOR_SOURCE,
            )
            compaction = compactor.compact_unit(
                root=root.resolve(),
                unit=unit,
                execute=True,
            )
        results.append(
            {
                "unit": unit,
                "lane": lane,
                "state": state,
                "compaction": compaction,
                "prior_scalar_qc": prior_qc,
            }
        )
        atomic_json(attempt / "results" / f"{unit}.json", results[-1])
        lane_summaries[lane] = engine.write_summary(root)
    passed = all(
        row["state"].get("status") == "PASS_PRETRACT_RECOVERY4"
        and isinstance(row["compaction"], dict)
        and row["compaction"].get("status") == "PASS_COMPACTED"
        for row in results
    )
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_fod_soft_negative_recovery_attempt",
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
            for row in results
        ),
        "units": [str(row["unit"]) for row in targets],
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
        if FAILURE_CLASS != "FOD_L0_SOFT_NEGATIVE_FRACTION":
            raise AssertionError(FAILURE_CLASS)
        print("HCP379_FOD_SOFT_NEGATIVE_RECOVERY_SELF_TEST_PASS")
        return 0
    payload = (
        execute(args.human_qc_manifest, nthreads=args.nthreads)
        if args.execute
        else preflight(args.human_qc_manifest)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") in {
        "READY_FOR_EXACT_FOD_RECOVERY",
        "NO_CURRENT_FOD_RECOVERY_TARGETS",
        "PASS",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
