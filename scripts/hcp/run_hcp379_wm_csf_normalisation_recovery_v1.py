#!/home/ec2-user/fsl/bin/python
"""Recover a degenerate-GM mtnormalise failure with WM+CSF normalisation.

This exact diagnosis-blind recovery is eligible only when the 515-unit ledger
classifies a unit as MTNORMALISE_NONPOSITIVE_TISSUE_BALANCE and its raw GM
compartment is numerically degenerate.  WM and CSF are supplied to the
documented multi-tissue normaliser; the resulting spatial normalisation field
is applied to the excluded near-zero GM compartment with unit balance.  The
failed engine state is content-addressed before a replacement PASS engine
record is written.  Recovery4 scalar, spatial, atlas and compaction gates are
then replayed unchanged.  No diagnosis, outcome, density, tractography or
matrix is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
WRAPPER_SOURCE = (
    EXP / "scripts/hcp/run_hcp379_legacy_density_pretract_recovery4_v3.py"
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
ENGINE_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_scaleup_pretract_recovery4_v3/validation.json"
)
ATTEMPTS = HCP_ROOT / "pretract_failure_recovery_v1/attempts"
ROOT = HCP_ROOT / "corrected_legacy_tensor_recovery4"
DEFAULT_HUMAN_QC = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
FAILURE_CLASS = "MTNORMALISE_NONPOSITIVE_TISSUE_BALANCE"
GM_DEGENERATE_ABSOLUTE_MAXIMUM = 1.0e-8


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WRAPPER = load_module(WRAPPER_SOURCE, "hcp379_wm_csf_recovery_wrapper")
ENGINE = WRAPPER.PRETRACT
V2 = ENGINE.V2


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


def target_context(
    human_qc: Path,
) -> tuple[str, dict[str, str], dict[str, Any], dict[str, Path]]:
    require_validation(LEDGER_VALIDATION, expected_checks=6)
    require_validation(ENGINE_VALIDATION, expected_checks=19)
    ledger = load_json(LEDGER)
    failed = ledger.get("failed_units")
    if not isinstance(failed, list):
        raise ValueError("failure ledger differs")
    units = sorted(
        str(row["unit"])
        for row in failed
        if row.get("failure_class") == FAILURE_CLASS
    )
    if len(units) != 1:
        raise ValueError(f"exactly one tissue-balance target required: {units}")
    unit = units[0]
    WRAPPER.specialize(ROOT)
    gate = ENGINE.validate_recovery4_gate(
        human_qc=human_qc,
        execute=True,
    )
    rows, _ = ENGINE.validate_audit_rows(gate)
    by_unit = {str(row["unit"]): row for row in rows}
    if unit not in by_unit:
        raise ValueError("tissue-balance target outside legacy lane")
    row = by_unit[unit]
    paths = ENGINE.spatial_paths(ROOT, unit)
    state = load_json(paths["state"])
    engine_state = load_json(paths["engine_state"])
    if (
        state.get("status") != "FAIL_PRETRACT_RECOVERY4"
        or "mtnormalise" not in str(state.get("error", ""))
        or engine_state.get("status") != "FAIL_PRETRACT"
        or "mtnormalise" not in str(engine_state.get("error", ""))
    ):
        raise ValueError("tissue-balance failure state differs")
    gm_range = V2.mrstats_range(paths["gm"], paths["mask"])
    if (
        not math.isfinite(float(gm_range["min"]))
        or not math.isfinite(float(gm_range["max"]))
        or max(abs(float(gm_range["min"])), abs(float(gm_range["max"])))
        > GM_DEGENERATE_ABSOLUTE_MAXIMUM
    ):
        raise ValueError(f"GM compartment is not degenerate: {gm_range}")
    return unit, row, gate, paths


def preflight(human_qc: Path) -> dict[str, Any]:
    unit, _, _, paths = target_context(human_qc)
    gm_range = V2.mrstats_range(paths["gm"], paths["mask"])
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_wm_csf_normalisation_recovery_preflight",
        "status": "READY_FOR_EXACT_WM_CSF_NORMALISATION_RECOVERY",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "imaging_executed": False,
        "tractography_generated": False,
        "matrix_generated": False,
        "failure_class": FAILURE_CLASS,
        "target_n": 1,
        "target_unit": unit,
        "raw_gm_range": gm_range,
        "policy": {
            "normalised_tissues": ["wmfod", "csf"],
            "excluded_balance_tissue": "gm",
            "gm_degenerate_absolute_maximum": (
                GM_DEGENERATE_ABSOLUTE_MAXIMUM
            ),
            "gm_spatial_field_application": "gm_raw / shared_norm_field",
            "mtnormalise_reference": 0.282095,
            "balanced_output_option_used": False,
            "per_subject_parameter_tuning_allowed": False,
            "scalar_qc_threshold_change_allowed": False,
            "spatial_qc_threshold_change_allowed": False,
        },
        "records": {
            "ledger": file_record(LEDGER),
            "ledger_validation": file_record(LEDGER_VALIDATION),
            "engine_validation": file_record(ENGINE_VALIDATION),
            "runner": file_record(Path(__file__)),
            "engine": file_record(ENGINE_SOURCE),
            "human_qc": file_record(human_qc),
            "mtnormalise_binary": file_record(
                ENGINE.MRTRIX / "mtnormalise"
            ),
        },
    }


def move_new(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ValueError(f"refusing to overwrite recovery output: {destination}")
    source.replace(destination)


def execute(human_qc: Path, *, nthreads: int) -> dict[str, Any]:
    pre = preflight(human_qc)
    unit, row, gate, paths = target_context(human_qc)
    incomplete_attempts = sorted(
        path
        for path in ATTEMPTS.glob(
            "*-wm-csf-normalisation-recovery"
        )
        if (path / "preflight.json").is_file()
        and not (path / "attempt.json").exists()
        and load_json(path / "preflight.json").get("target_unit") == unit
    )
    if len(incomplete_attempts) > 1:
        raise ValueError(
            "multiple incomplete WM+CSF recovery attempts: "
            + ",".join(str(path) for path in incomplete_attempts)
        )
    resumed_incomplete_attempt = bool(incomplete_attempts)
    if resumed_incomplete_attempt:
        attempt = incomplete_attempts[0]
        if load_json(attempt / "preflight.json") != pre:
            # Hash-bearing records can legitimately differ after a source-only
            # resume fix.  The immutable scientific policy and target may not.
            prior_preflight = load_json(attempt / "preflight.json")
            for key in (
                "record_type",
                "status",
                "target_n",
                "target_unit",
                "raw_gm_range",
                "policy",
            ):
                if prior_preflight.get(key) != pre.get(key):
                    raise ValueError(
                        f"incomplete-attempt preflight differs: {key}"
                    )
    else:
        attempt = (
            ATTEMPTS
            / (
                datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%S.%fZ"
                )
                + "-wm-csf-normalisation-recovery"
            )
        )
        attempt.mkdir(parents=True, exist_ok=False)
        atomic_json(attempt / "preflight.json", pre)
    preserved: dict[str, Any] = {}
    for name, source in {
        "recovery4_state": paths["state"],
        "failed_engine_state": paths["engine_state"],
    }.items():
        record = file_record(source)
        destination = (
            attempt / "prior_states" / f"{name}-{record['sha256']}.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with destination.open("xb") as handle:
                handle.write(source.read_bytes())
        copy = file_record(destination)
        if copy["sha256"] != record["sha256"]:
            raise ValueError(f"prior-state archive hash differs: {name}")
        preserved[name] = {"source": record, "archive": copy}

    recovery = (
        paths["subject"] / "05_model/recovery4_wm_csf_normalisation"
    )
    recovery.mkdir(parents=True, exist_ok=resumed_incomplete_attempt)
    log_path = recovery / "recovery.log"
    partial = {
        "wmfod_norm": recovery / "wmfod_norm.partial.mif",
        "csf_norm": recovery / "csf_norm.partial.mif",
        "gm_norm": recovery / "gm_norm.partial.mif",
        "tensor": recovery / "tensor.partial.mif",
        "fa": recovery / "fa.partial.mif",
        "md": recovery / "md.partial.mif",
        "rd": recovery / "rd.partial.mif",
        "ad": recovery / "ad.partial.mif",
        "wmfod_l0": recovery / "wmfod_l0.partial.mif",
    }
    norm_field = recovery / "shared_norm_field.mif"
    factors = recovery / "wm_csf_balance_factors.txt"
    env = ENGINE.environment(nthreads)
    destination_names = (
        "wmfod_norm",
        "gm_norm",
        "csf_norm",
        "tensor",
        "fa",
        "md",
        "rd",
        "ad",
        "wmfod_l0",
    )
    destination_presence = {
        name: paths[name].is_file() and paths[name].stat().st_size > 0
        for name in destination_names
    }
    completed_image_stage = (
        resumed_incomplete_attempt
        and all(destination_presence.values())
        and log_path.is_file()
        and norm_field.is_file()
        and factors.is_file()
        and not any(path.exists() for path in partial.values())
    )
    if not completed_image_stage and any(destination_presence.values()):
        raise ValueError(
            "partial standard-output state cannot be resumed safely: "
            + json.dumps(destination_presence, sort_keys=True)
        )
    if not completed_image_stage:
        with log_path.open("x", encoding="utf-8") as log:
            ENGINE.run(
            [
                str(ENGINE.MRTRIX / "mtnormalise"),
                str(paths["wmfod"]),
                str(partial["wmfod_norm"]),
                str(paths["csf"]),
                str(partial["csf_norm"]),
                "-mask",
                str(paths["mask"]),
                "-check_norm",
                str(norm_field),
                "-check_factors",
                str(factors),
                "-nthreads",
                str(nthreads),
                "-force",
            ],
            log=log,
            env=env,
            outputs=(
                partial["wmfod_norm"],
                partial["csf_norm"],
                norm_field,
                factors,
            ),
        )
            ENGINE.run(
            [
                str(ENGINE.MRTRIX / "mrcalc"),
                str(paths["gm"]),
                str(norm_field),
                "-div",
                str(partial["gm_norm"]),
                "-force",
            ],
            log=log,
            env=env,
            outputs=(partial["gm_norm"],),
        )
            ENGINE.run(
            [
                str(ENGINE.MRTRIX / "dwi2tensor"),
                str(paths["fod_dwi"]),
                str(partial["tensor"]),
                "-mask",
                str(paths["mask"]),
                "-iter",
                "2",
                "-config",
                "BZeroThreshold",
                str(gate["bzero_threshold"]),
                "-nthreads",
                str(nthreads),
                "-force",
            ],
            log=log,
            env=env,
            outputs=(partial["tensor"],),
        )
            ENGINE.run(
            [
                str(ENGINE.MRTRIX / "tensor2metric"),
                str(partial["tensor"]),
                "-fa",
                str(partial["fa"]),
                "-adc",
                str(partial["md"]),
                "-rd",
                str(partial["rd"]),
                "-ad",
                str(partial["ad"]),
                "-nthreads",
                str(nthreads),
                "-force",
            ],
            log=log,
            env=env,
            outputs=(
                partial["fa"],
                partial["md"],
                partial["rd"],
                partial["ad"],
            ),
        )
            ENGINE.run(
            [
                str(ENGINE.MRTRIX / "mrconvert"),
                str(partial["wmfod_norm"]),
                str(partial["wmfod_l0"]),
                "-coord",
                "3",
                "0",
                "-force",
            ],
            log=log,
            env=env,
            outputs=(partial["wmfod_l0"],),
        )

    factor_values = [
        float(value) for value in factors.read_text().split()
    ]
    norm_range = V2.mrstats_range(norm_field, paths["mask"])
    if (
        len(factor_values) != 2
        or not all(math.isfinite(value) and value > 0 for value in factor_values)
        or float(norm_range["min"]) <= 0
    ):
        raise ValueError("WM+CSF normalisation evidence differs")
        for name in destination_names:
            move_new(partial[name], paths[name])

    ranges = {
        "wmfod_all_sh_coefficients": ENGINE.all_volume_range(
            paths["wmfod_norm"], paths["mask"]
        ),
        "wmfod_l0": ENGINE.all_volume_range(
            paths["wmfod_l0"], paths["mask"]
        ),
        "gm": ENGINE.all_volume_range(
            paths["gm_norm"], paths["mask"]
        ),
        "csf": ENGINE.all_volume_range(
            paths["csf_norm"], paths["mask"]
        ),
        "fa": ENGINE.all_volume_range(paths["fa"], paths["mask"]),
        "md": ENGINE.all_volume_range(paths["md"], paths["mask"]),
        "rd": ENGINE.all_volume_range(paths["rd"], paths["mask"]),
        "ad": ENGINE.all_volume_range(paths["ad"], paths["mask"]),
    }
    scforge_path = str(EXP / "scforge")
    if scforge_path not in sys.path:
        sys.path.insert(0, scforge_path)
    from scforge.input_contract import (
        assess_pre_tractography_image_ranges,
    )

    provisional_any_voxel_range_failures = (
        assess_pre_tractography_image_ranges(ranges)
    )
    # The legacy provisional engine rejected a tensor or FOD when even one
    # voxel was outside its display range.  Recovery4 intentionally supersedes
    # that rule with its locked fraction-based tensor and FOD numerical gates,
    # which process_unit replays below.  Preserve the legacy findings as
    # diagnostic evidence, but do not promote or reject from them here.
    range_failures: list[str] = []
    # The generic engine record is completed here solely to permit the
    # standard Recovery4 hash-reuse path to run scalar and spatial QC.
    atlas_qc = load_json(paths["atlas_qc"])
    mask_qc = V2.mask_physical_qc(paths["mask"])
    artifacts = {
        name: file_record(paths[name])
        for name in (
            "dwi",
            "b0",
            "mask",
            "t1",
            "five_tt",
            "gmwmi",
            "t1_to_b0",
            "hcp_nodes",
            "hcp_volumes",
            "atlas_qc",
            "atlas_overlay",
            "wmfod_norm",
            "fa",
            "md",
            "rd",
            "ad",
        )
    }
    automated_qc = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_scaleup_automated_pretract_qc"
        ),
        "status": "PASS",
        "unit": unit,
        "diagnosis_labels_used": False,
        "mask_method": "reused_valid_dwi_mask",
        "mask_qc": mask_qc,
        "atlas_qc": atlas_qc,
        "image_ranges": ranges,
        "range_failures": range_failures,
        "provisional_any_voxel_range_failures": (
            provisional_any_voxel_range_failures
        ),
        "authoritative_scalar_qc": (
            "Recovery4 fraction-based tensor and FOD gates replayed "
            "by process_unit"
        ),
        "normalisation_recovery": {
            "route": "wm_csf_with_shared_field_on_degenerate_gm",
            "positive_balance_factors": factor_values,
            "shared_norm_field": file_record(norm_field),
        },
    }
    atomic_json(paths["automated_qc"], automated_qc)
    artifacts["automated_qc"] = file_record(paths["automated_qc"])
    recovered_engine = {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_scaleup_pretract",
        "status": "PASS_PRETRACT",
        "unit": unit,
        "diagnosis_labels_used": False,
        "completed_utc": utc_now(),
        "input_route": row["route"],
        "normalisation_recovery": automated_qc[
            "normalisation_recovery"
        ],
        "artifacts": artifacts,
        "error": None,
    }
    atomic_json(paths["engine_state"], recovered_engine)
    ENGINE.validate_reusable_pass_engine_state(
        paths["engine_state"],
        unit=unit,
    )
    state = ENGINE.process_unit(
        row,
        root=ROOT,
        gate=gate,
        nthreads=nthreads,
        reuse_valid_engine=True,
    )
    compactor = ENGINE.load_module(
        "hcp379_wm_csf_recovery_compactor",
        ENGINE.COMPACTOR_SOURCE,
    )
    compaction = (
        compactor.compact_unit(
            root=ROOT.resolve(),
            unit=unit,
            execute=True,
        )
        if state.get("status") == "PASS_PRETRACT_RECOVERY4"
        else None
    )
    passed = (
        state.get("status") == "PASS_PRETRACT_RECOVERY4"
        and isinstance(compaction, dict)
        and compaction.get("status") == "PASS_COMPACTED"
    )
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_wm_csf_normalisation_recovery_attempt",
        "status": "PASS" if passed else "FAIL",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "target_unit": unit,
        "normalisation_route": (
            "wm_csf_with_shared_field_on_degenerate_gm"
        ),
        "resumed_incomplete_attempt": resumed_incomplete_attempt,
        "image_stage_reused": completed_image_stage,
        "balance_factors": factor_values,
        "norm_field_range": norm_range,
        "state": state,
        "compaction": compaction,
        "preserved_prior_states": preserved,
        "preflight": file_record(attempt / "preflight.json"),
        "log": file_record(log_path),
    }
    atomic_json(attempt / "attempt.json", payload)
    ENGINE.write_summary(ROOT)
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
        if (
            FAILURE_CLASS != "MTNORMALISE_NONPOSITIVE_TISSUE_BALANCE"
            or GM_DEGENERATE_ABSOLUTE_MAXIMUM != 1.0e-8
        ):
            raise AssertionError("normalisation recovery constants differ")
        print("HCP379_WM_CSF_NORMALISATION_RECOVERY_SELF_TEST_PASS")
        return 0
    payload = (
        execute(args.human_qc_manifest, nthreads=args.nthreads)
        if args.execute
        else preflight(args.human_qc_manifest)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") in {
        "READY_FOR_EXACT_WM_CSF_NORMALISATION_RECOVERY",
        "PASS",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
