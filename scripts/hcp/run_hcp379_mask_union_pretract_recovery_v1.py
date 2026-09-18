#!/home/ec2-user/fsl/bin/python
"""Run the fixed, diagnosis-blind mask-union recovery for two spatial failures.

This runner deliberately leaves the shared Recovery4 engines unchanged.  It
replays the independently validated mask policy in a new output namespace:
the default dwi2mask result is unioned with a one-native-voxel dilation of a
second dwi2mask result generated with peninsula cleaning disabled.  All other
Recovery4 processing and locked QC thresholds remain unchanged.

The default action is a non-imaging plan.  ``--execute`` is required to run
the two subjects.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
ENGINE_SOURCE = (
    EXP / "scripts/hcp/hcp379_scaleup_pretract_recovery4_v3.py"
)
LANE_SOURCE = (
    EXP / "scripts/hcp/run_hcp379_legacy_density_pretract_recovery4_v3.py"
)
AUDIT_SUMMARY = (
    HCP_ROOT
    / "pretract_failure_source_audit_v1/"
    "mask_union_recovery_controls_v1/summary.json"
)
INDEPENDENT_VALIDATION = (
    EXP
    / "research_audit/outputs/hcp379_mask_union_recovery_v1/"
    "validation.json"
)
OUTPUT_ROOT = (
    HCP_ROOT / "corrected_legacy_tensor_mask_union_recovery4"
)
DEFAULT_HUMAN_QC = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
TARGETS = ("003_S_0908_I1249292", "003_S_4350_I1252856")
POLICY_NAME = "dwi2mask_uncleaned_dilate1_union_original_v1"
EXPECTED_ENGINE_SHA256 = (
    "dd4ad0ed3b2712fade1ff3ff62ddb219db568206afd6492676049e44966be6c3"
)
EXPECTED_V2_SHA256 = (
    "04e7ef4001b2107a62b65a1e1b3363d10e2e12e883acfba7b16da7b97a8d67d7"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRETRACT = load_module(
    ENGINE_SOURCE, "hcp379_mask_union_recovery4_engine"
)
LANE = load_module(LANE_SOURCE, "hcp379_mask_union_recovery4_lane")


def verify_evidence() -> dict[str, Any]:
    audit = PRETRACT.load_json(AUDIT_SUMMARY)
    validation = PRETRACT.load_json(INDEPENDENT_VALIDATION)
    engine_record = PRETRACT.file_record(ENGINE_SOURCE)
    v2_record = PRETRACT.file_record(PRETRACT.V2_SOURCE)
    audit_targets = {
        str(row.get("unit")) for row in audit.get("targets", [])
    }
    if (
        audit.get("status") != "PASS"
        or audit.get("diagnosis_labels_used") is not False
        or audit.get("outcomes_used") is not False
        or audit.get("connectome_density_used") is not False
        or audit_targets != set(TARGETS)
        or audit.get("policy", {}).get("dwi2mask_clean_scale") != 0
        or audit.get("policy", {}).get(
            "native_voxel_dilation_passes"
        )
        != 1
        or audit.get("policy", {}).get(
            "original_mask_retained_by_union"
        )
        is not True
        or audit.get("policy", {}).get(
            "per_subject_parameter_tuning_allowed"
        )
        is not False
        or validation.get("status") != "PASS"
        or validation.get("passed_checks") != 6
        or validation.get("total_checks") != 6
        or engine_record["sha256"] != EXPECTED_ENGINE_SHA256
        or v2_record["sha256"] != EXPECTED_V2_SHA256
    ):
        raise ValueError("mask-union evidence or shared-engine binding differs")
    return {
        "audit": PRETRACT.file_record(AUDIT_SUMMARY),
        "independent_validation": PRETRACT.file_record(
            INDEPENDENT_VALIDATION
        ),
        "recovery4_engine": engine_record,
        "provisional_v2_engine": v2_record,
    }


def selected_rows(
    human_qc: Path,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    PRETRACT.AUDIT_CSV = LANE.AUDIT_CSV
    PRETRACT.AUDIT_SUMMARY = LANE.AUDIT_SUMMARY
    PRETRACT.OUTPUT_ROOT = OUTPUT_ROOT
    PRETRACT.DEFAULT_HUMAN_QC = DEFAULT_HUMAN_QC
    PRETRACT.EXPECTED_AUDIT_N = LANE.EXPECTED_AUDIT_N
    PRETRACT.EXPECTED_CANARY_OVERLAP_N = LANE.EXPECTED_CANARY_OVERLAP_N
    PRETRACT.EXPECTED_SCALEUP_N = LANE.EXPECTED_EXECUTION_N
    gate = PRETRACT.validate_recovery4_gate(
        human_qc=human_qc, execute=True
    )
    rows, _ = PRETRACT.validate_audit_rows(gate)
    selected = [row for row in rows if row["unit"] in set(TARGETS)]
    if (
        len(selected) != len(TARGETS)
        or tuple(sorted(row["unit"] for row in selected))
        != tuple(sorted(TARGETS))
        or any(
            row["route"] != "READY_FOR_CORRECTED_PRETRACT_FROM_EDDY"
            for row in selected
        )
    ):
        raise ValueError("exact mask-union recovery target set differs")
    return gate, sorted(selected, key=lambda row: row["unit"])


def mask_paths(root: Path, unit: str) -> dict[str, Path]:
    directory = root / "subjects" / unit / "01_dwi"
    return {
        "original": directory / "dwi_brain_mask_original.mif",
        "uncleaned": directory / "dwi_brain_mask_uncleaned.mif",
        "dilated": directory / "dwi_brain_mask_uncleaned_dilate1.mif",
        "union": directory / "dwi_brain_mask.mif",
    }


def install_mask_policy(root: Path) -> None:
    original_run = PRETRACT.V2.run
    original_process = PRETRACT.V2.process_unit

    def fixed_run(
        command: list[str],
        *,
        log: Any,
        env: Mapping[str, str],
        outputs: tuple[Path, ...] = (),
    ) -> None:
        if (
            len(outputs) == 1
            and Path(command[0]).name == "dwi2mask"
            and outputs[0].name == "dwi_brain_mask.mif"
            and root.resolve() in outputs[0].resolve().parents
        ):
            unit = outputs[0].parents[1].name
            if unit not in TARGETS:
                raise ValueError(f"mask policy reached non-target: {unit}")
            paths = mask_paths(root, unit)
            expected = outputs[0].resolve()
            if paths["union"].resolve() != expected:
                raise ValueError("mask-union target path differs")
            original_command = [
                str(paths["original"]) if item == str(outputs[0]) else item
                for item in command
            ]
            original_run(
                original_command,
                log=log,
                env=env,
                outputs=(paths["original"],),
            )
            uncleaned_command = list(command)
            uncleaned_command[
                uncleaned_command.index(str(outputs[0]))
            ] = str(paths["uncleaned"])
            uncleaned_command.extend(("-clean_scale", "0"))
            original_run(
                uncleaned_command,
                log=log,
                env=env,
                outputs=(paths["uncleaned"],),
            )
            original_run(
                [
                    str(PRETRACT.MRTRIX / "maskfilter"),
                    str(paths["uncleaned"]),
                    "dilate",
                    str(paths["dilated"]),
                    "-npass",
                    "1",
                    "-nthreads",
                    command[command.index("-nthreads") + 1],
                ],
                log=log,
                env=env,
                outputs=(paths["dilated"],),
            )
            original_run(
                [
                    str(PRETRACT.MRTRIX / "mrcalc"),
                    str(paths["original"]),
                    str(paths["dilated"]),
                    "-max",
                    str(paths["union"]),
                ],
                log=log,
                env=env,
                outputs=(paths["union"],),
            )
            return
        original_run(command, log=log, env=env, outputs=outputs)

    def fixed_process(
        row: Mapping[str, str],
        *,
        root: Path,
        gate: Mapping[str, Any],
        nthreads: int,
    ) -> dict[str, Any]:
        state = original_process(
            row, root=root, gate=gate, nthreads=nthreads
        )
        unit = str(row["unit"])
        if unit not in TARGETS:
            raise ValueError(f"mask policy processed non-target: {unit}")
        masks = mask_paths(root, unit)
        records = {
            name: PRETRACT.file_record(path)
            for name, path in masks.items()
        }
        original_qc = PRETRACT.V2.mask_physical_qc(masks["original"])
        union_qc = PRETRACT.V2.mask_physical_qc(masks["union"])
        ratio = (
            union_qc["volume_mm3"] / original_qc["volume_mm3"]
            if original_qc["volume_mm3"] > 0
            else float("inf")
        )
        if (
            state.get("status") != "PASS_PRETRACT"
            or not 1.0 <= ratio <= 1.15
        ):
            raise ValueError(
                f"{unit}: mask-union engine or volume gate failed: "
                f"status={state.get('status')} ratio={ratio}"
            )
        qc_path = PRETRACT.V2.stage_paths(root, unit)["automated_qc"]
        automated = PRETRACT.load_json(qc_path)
        automated.update(
            {
                "mask_method": POLICY_NAME,
                "mask_recovery_policy": POLICY_NAME,
                "mask_recovery_artifacts": records,
                "original_mask_qc": original_qc,
                "union_mask_qc": union_qc,
                "mask_volume_ratio": ratio,
            }
        )
        PRETRACT.atomic_json(qc_path, automated)
        state.update(
            {
                "mask_method": POLICY_NAME,
                "mask_recovery_policy": POLICY_NAME,
                "mask_recovery_artifacts": records,
                "original_mask_qc": original_qc,
                "mask_qc": union_qc,
                "mask_volume_ratio": ratio,
            }
        )
        state["artifacts"]["automated_qc"] = PRETRACT.V2.size_record(
            qc_path
        )
        return state

    PRETRACT.V2.run = fixed_run
    PRETRACT.V2.process_unit = fixed_process


def build_plan(
    *,
    evidence: Mapping[str, Any],
    rows: list[dict[str, str]],
    execute: bool,
) -> dict[str, Any]:
    plan = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_mask_union_pretract_recovery_plan"
        ),
        "status": "EXECUTION_GATE_PASS" if execute else "DRY_RUN_PASS",
        "generated_utc": PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "execution_requested": execute,
        "imaging_executed_by_plan": False,
        "output_root": str(OUTPUT_ROOT.resolve()),
        "target_n": len(TARGETS),
        "target_units": list(TARGETS),
        "mask_policy": {
            "name": POLICY_NAME,
            "dwi2mask_clean_scale": 0,
            "native_voxel_dilation_passes": 1,
            "original_mask_retained_by_union": True,
            "maximum_mask_volume_ratio": 1.15,
            "per_subject_parameter_tuning_allowed": False,
        },
        "inputs": [
            {
                "unit": row["unit"],
                "route": row["route"],
                "dti_source_id": row["dti_source_id"],
                "expected_t1_image_id": row["expected_t1_image_id"],
            }
            for row in rows
        ],
        "evidence": dict(evidence),
        "runner": PRETRACT.file_record(Path(__file__)),
    }
    PRETRACT.atomic_json(
        OUTPUT_ROOT / "manifests/mask_union_recovery_plan.json", plan
    )
    return plan


def enrich_final(
    state: dict[str, Any],
    *,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    unit = str(state["unit"])
    masks = mask_paths(OUTPUT_ROOT, unit)
    state.update(
        {
            "mask_recovery_policy": POLICY_NAME,
            "mask_recovery_is_diagnosis_outcome_density_blind": True,
            "mask_recovery_evidence": dict(evidence),
        }
    )
    if state.get("status") == "PASS_PRETRACT_RECOVERY4":
        for name, path in masks.items():
            state["artifacts"][f"mask_recovery_{name}"] = (
                PRETRACT.file_record(path)
            )
    PRETRACT.atomic_json(
        PRETRACT.spatial_paths(OUTPUT_ROOT, unit)["state"], state
    )
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--human-qc-manifest", type=Path, default=DEFAULT_HUMAN_QC
    )
    parser.add_argument("--nthreads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.nthreads <= 8:
        parser.error("--nthreads must be in 1..8")

    evidence = verify_evidence()
    gate, rows = selected_rows(args.human_qc_manifest)
    if args.self_test:
        print("HCP379_MASK_UNION_PRETRACT_RECOVERY_V1_SELF_TEST_PASS")
        return 0
    plan = build_plan(evidence=evidence, rows=rows, execute=args.execute)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    PRETRACT.install_provisional_engine_guards()
    install_mask_policy(OUTPUT_ROOT)
    failures = 0
    for index, row in enumerate(rows, start=1):
        state = PRETRACT.process_unit(
            row,
            root=OUTPUT_ROOT,
            gate=gate,
            nthreads=args.nthreads,
            allow_affine_failover=True,
            allow_syn_failover=True,
        )
        state = enrich_final(state, evidence=evidence)
        if state.get("status") != "PASS_PRETRACT_RECOVERY4":
            failures += 1
        print(
            f"[{PRETRACT.utc_now()}] mask_union_recovery "
            f"{index}/{len(rows)} {row['unit']} "
            f"{state.get('status')} "
            f"route={state.get('registration_route')} "
            f"error={state.get('error')}",
            flush=True,
        )
    summary = PRETRACT.write_summary(OUTPUT_ROOT)
    summary.update(
        {
            "status": (
                "PASS"
                if summary.get("states_present_n") == len(TARGETS)
                and summary.get("status_counts", {}).get(
                    "PASS_PRETRACT_RECOVERY4"
                )
                == len(TARGETS)
                else "IN_PROGRESS"
            ),
            "target_n": len(TARGETS),
            "mask_recovery_policy": POLICY_NAME,
            "expected_target_n": len(TARGETS),
            "evidence": dict(evidence),
        }
    )
    PRETRACT.atomic_json(
        OUTPUT_ROOT / "manifests/pretract_recovery4_summary.json",
        summary,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
