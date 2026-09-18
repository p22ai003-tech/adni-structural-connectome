#!/home/ec2-user/fsl/bin/python
"""Diagnose the isolated Recovery4 raw-tensor failure without promotion.

The experiment is deliberately small and diagnosis blind.  It replays the
frozen WLS+IWLS(2) tensor fit and evaluates four fixed counterfactual fits for
the one current RD-negative-fraction failure and one deterministic,
acquisition-matched Recovery4 control.  Every output is written to a new
attempt directory.  The 1% raw-anomaly gate is unchanged, and no result from
this diagnostic is promoted into the production pretract roots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
TARGET_UNIT = "135_S_6284_I1164439"
CONTROL_UNIT = "009_S_4324_I1186579"
TARGET_ROOT = HCP_ROOT / "corrected_scaleup_recovery4"
CONTROL_RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/"
    "h04a_r1_recovery_20260719_retry4"
)
CONTROL_RECOVERY4_ROOT = HCP_ROOT / "pretract_recovery4"
ATTEMPTS = HCP_ROOT / "tensor_fit_counterfactual_v1/attempts"
INPUT_AUDIT = (
    EXP
    / "research_audit/outputs/hcp379_scaleup_input_audit_v2/"
    "scaleup_input_audit.csv"
)
INPUT_AUDIT_SUMMARY = INPUT_AUDIT.with_name(
    "scaleup_input_audit_summary.json"
)
FAILURE_LEDGER = (
    EXP
    / "research_audit/outputs/"
    "hcp379_pretract_failure_recovery_ledger_v1/ledger.csv"
)
FAILURE_LEDGER_VALIDATION = FAILURE_LEDGER.with_name("validation.json")
RAW_ANOMALY_FRACTION_MAXIMUM = 0.01
BZERO_THRESHOLD = 50.0
DIFFUSIVITY_MAXIMUM = 0.01
ACQUISITION_FIELDS = (
    "dwi_size",
    "dwi_spacing_mm",
    "gradient_rows",
    "b0_rows",
    "diffusion_rows",
    "maximum_b_value",
    "shell_centroids_s_per_mm2",
    "shell_sizes",
)
METHODS = (
    {
        "name": "wls_iwls2_replay",
        "initial_fit": "WLS",
        "iwls_iterations": 2,
        "production_replay": True,
    },
    {
        "name": "wls_initial",
        "initial_fit": "WLS",
        "iwls_iterations": 0,
        "production_replay": False,
    },
    {
        "name": "wls_iwls4",
        "initial_fit": "WLS",
        "iwls_iterations": 4,
        "production_replay": False,
    },
    {
        "name": "ols_initial",
        "initial_fit": "OLS",
        "iwls_iterations": 0,
        "production_replay": False,
    },
    {
        "name": "ols_iwls2",
        "initial_fit": "OLS",
        "iwls_iterations": 2,
        "production_replay": False,
    },
)
REPLAY_LIMITS = {
    "fa": {"mean_abs": 1.0e-6, "max_abs": 1.0e-4},
    "md": {"mean_abs": 1.0e-9, "max_abs": 1.0e-6},
    "rd": {"mean_abs": 1.0e-9, "max_abs": 1.0e-6},
    "ad": {"mean_abs": 1.0e-9, "max_abs": 1.0e-6},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


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


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require_validation(path: Path, minimum_checks: int) -> None:
    value = load_json(path)
    if (
        value.get("status") != "PASS"
        or int(value.get("passed_checks", -1))
        != int(value.get("total_checks", -2))
        or int(value.get("total_checks", -1)) < minimum_checks
    ):
        raise ValueError(f"validation differs: {path}")


def subject_contract(unit: str) -> dict[str, Any]:
    if unit == TARGET_UNIT:
        subject_root = TARGET_ROOT / "subjects" / unit
        scalar_qc = (
            subject_root
            / "06_preflight/scalar_recovery4_qc.json"
        )
        return {
            "unit": unit,
            "role": "isolated_current_raw_tensor_failure",
            "dwi": subject_root / "05_model/dwi_fod_shells.mif",
            "mask": subject_root / "01_dwi/dwi_brain_mask.mif",
            "raw": {
                metric: subject_root / "05_model" / f"{metric}.mif"
                for metric in ("fa", "md", "rd", "ad")
            },
            "scalar_qc": scalar_qc,
            "recovery4_manifest": None,
        }
    if unit == CONTROL_UNIT:
        subject_root = CONTROL_RUN_ROOT / "subjects" / unit
        scalar_qc = (
            HCP_ROOT
            / "pretract_scalar_recovery4/subjects"
            / unit
            / "06_preflight/scalar_recovery4_qc.json"
        )
        return {
            "unit": unit,
            "role": "acquisition_matched_recovery4_pass_control",
            "dwi": subject_root / "05_model/dwi_tensor_shells.mif",
            "mask": subject_root / "01_dwi/dwi_brain_mask.mif",
            "raw": {
                metric: subject_root / "05_model" / f"{metric}.mif"
                for metric in ("fa", "md", "rd", "ad")
            },
            "scalar_qc": scalar_qc,
            "recovery4_manifest": (
                CONTROL_RECOVERY4_ROOT
                / "subjects"
                / unit
                / "pretract_recovery4_manifest.json"
            ),
        }
    raise ValueError(f"unexpected subject: {unit}")


def validated_subjects() -> list[dict[str, Any]]:
    require_validation(FAILURE_LEDGER_VALIDATION, minimum_checks=6)
    input_summary = load_json(INPUT_AUDIT_SUMMARY)
    if (
        input_summary.get("status") != "PASS"
        or input_summary.get("diagnosis_labels_used") is not False
        or int(input_summary.get("audited_n", -1)) != 227
        or int(input_summary.get("target_n", -1)) != 227
    ):
        raise ValueError("input-audit summary differs")
    audit = csv_rows(INPUT_AUDIT)
    audit_by_unit = {row["unit"]: row for row in audit}
    if len(audit_by_unit) != len(audit):
        raise ValueError("input-audit identities differ")
    for unit in (TARGET_UNIT, CONTROL_UNIT):
        if unit not in audit_by_unit:
            raise ValueError(f"subject absent from input audit: {unit}")
        if audit_by_unit[unit].get("diagnosis_labels_used") != "False":
            raise ValueError(f"input audit is not diagnosis blind: {unit}")
    target_acquisition = {
        field: audit_by_unit[TARGET_UNIT][field]
        for field in ACQUISITION_FIELDS
    }
    exact_matches = sorted(
        row["unit"]
        for row in audit
        if row["unit"] != TARGET_UNIT
        and all(
            row.get(field) == target_acquisition[field]
            for field in ACQUISITION_FIELDS
        )
    )
    eligible_preserved_controls: list[str] = []
    for unit in exact_matches:
        manifest_path = (
            CONTROL_RECOVERY4_ROOT
            / "subjects"
            / unit
            / "pretract_recovery4_manifest.json"
        )
        dwi = (
            CONTROL_RUN_ROOT
            / "subjects"
            / unit
            / "05_model/dwi_tensor_shells.mif"
        )
        mask = (
            CONTROL_RUN_ROOT
            / "subjects"
            / unit
            / "01_dwi/dwi_brain_mask.mif"
        )
        if not (manifest_path.is_file() and dwi.is_file() and mask.is_file()):
            continue
        manifest = load_json(manifest_path)
        scalar_record = manifest.get("automated_qc", {}).get("scalar")
        if (
            manifest.get("diagnosis_labels_used") is False
            and manifest.get("status")
            == "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
            and isinstance(scalar_record, Mapping)
        ):
            scalar_path = Path(str(scalar_record.get("path", "")))
            if scalar_path.is_file() and load_json(scalar_path).get(
                "status"
            ) == "PASS":
                eligible_preserved_controls.append(unit)
    if (
        not eligible_preserved_controls
        or eligible_preserved_controls[0] != CONTROL_UNIT
    ):
        raise ValueError(
            "deterministic acquisition-matched control differs: "
            f"{eligible_preserved_controls}"
        )
    ledger = csv_rows(FAILURE_LEDGER)
    ledger_row = next(
        (row for row in ledger if row.get("unit") == TARGET_UNIT),
        None,
    )
    if (
        len(ledger) != 515
        or ledger_row is None
        or ledger_row.get("status") != "FAILED"
        or ledger_row.get("failure_class") != "OTHER_TERMINAL_FAILURE"
        or "rd_below_zero=0.011959860>0.010000000"
        not in ledger_row.get("error", "")
    ):
        raise ValueError("target failure-ledger contract differs")

    subjects: list[dict[str, Any]] = []
    for unit in (TARGET_UNIT, CONTROL_UNIT):
        contract = subject_contract(unit)
        scalar = load_json(contract["scalar_qc"])
        expected_status = "FAIL" if unit == TARGET_UNIT else "PASS"
        if scalar.get("status") != expected_status:
            raise ValueError(f"scalar-QC status differs: {unit}")
        if unit == TARGET_UNIT and scalar.get("failures") != [
            "rd_below_zero=0.011959860>0.010000000"
        ]:
            raise ValueError("target scalar failure differs")
        source_artifacts = scalar.get("source_artifacts")
        if not isinstance(source_artifacts, Mapping):
            raise ValueError(f"scalar source artifacts differ: {unit}")
        if source_artifacts.get("mask") != file_record(contract["mask"]):
            raise ValueError(f"scalar mask binding differs: {unit}")
        for metric, path in contract["raw"].items():
            if source_artifacts.get(f"{metric}_raw") != file_record(path):
                raise ValueError(
                    f"scalar raw-map binding differs: {unit}:{metric}"
                )
        manifest_record = (
            file_record(contract["recovery4_manifest"])
            if contract["recovery4_manifest"] is not None
            else None
        )
        subjects.append(
            {
                **contract,
                "acquisition": {
                    field: audit_by_unit[unit][field]
                    for field in ACQUISITION_FIELDS
                },
                "dwi_record": file_record(contract["dwi"]),
                "mask_record": file_record(contract["mask"]),
                "raw_records": {
                    metric: file_record(path)
                    for metric, path in contract["raw"].items()
                },
                "scalar_qc_record": file_record(contract["scalar_qc"]),
                "recovery4_manifest_record": manifest_record,
            }
        )
    if subjects[0]["acquisition"] != subjects[1]["acquisition"]:
        raise ValueError("target-control acquisition match differs")
    return subjects


def preflight() -> dict[str, Any]:
    subjects = validated_subjects()
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_tensor_fit_counterfactual_preflight",
        "status": "READY_BOUNDED_TENSOR_FIT_DIAGNOSTIC",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "imaging_executed": False,
        "production_outputs_modified": False,
        "raw_anomaly_fraction_gate": RAW_ANOMALY_FRACTION_MAXIMUM,
        "bzero_threshold_s_per_mm2": BZERO_THRESHOLD,
        "subjects": [
            {
                key: value
                for key, value in subject.items()
                if key
                not in {
                    "dwi",
                    "mask",
                    "raw",
                    "scalar_qc",
                    "recovery4_manifest",
                }
            }
            for subject in subjects
        ],
        "methods": [dict(method) for method in METHODS],
        "policy": {
            "target_selection": (
                "exact current isolated RD-negative-fraction failure"
            ),
            "control_selection": (
                "lexicographically first exact acquisition match with "
                "preserved tensor DWI/mask and Recovery4 scalar PASS"
            ),
            "per_subject_method_tuning_allowed": False,
            "qc_threshold_relaxation_allowed": False,
            "production_promotion_allowed": False,
            "candidate_followup_rule": (
                "an alternative must pass the unchanged all-eight raw "
                "tensor anomaly conditions in both target and control; "
                "even then a broader uniform-policy canary is required"
            ),
        },
        "replay_limits": REPLAY_LIMITS,
        "records": {
            "runner": file_record(Path(__file__)),
            "input_audit": file_record(INPUT_AUDIT),
            "input_audit_summary": file_record(INPUT_AUDIT_SUMMARY),
            "failure_ledger": file_record(FAILURE_LEDGER),
            "failure_ledger_validation": file_record(
                FAILURE_LEDGER_VALIDATION
            ),
            "dwi2tensor": file_record(MRTRIX / "dwi2tensor"),
            "tensor2metric": file_record(MRTRIX / "tensor2metric"),
            "mrcalc": file_record(MRTRIX / "mrcalc"),
            "mrstats": file_record(MRTRIX / "mrstats"),
        },
    }


def run_command(
    command: list[str],
    *,
    log: Any,
    outputs: tuple[Path, ...] = (),
    env: Mapping[str, str] | None = None,
) -> None:
    log.write(f"\n[{utc_now()}] COMMAND {json.dumps(command)}\n")
    log.flush()
    completed = subprocess.run(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env=dict(env) if env is not None else None,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {command[0]}"
        )
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError("command outputs missing: " + ",".join(missing))


def mask_voxel_count(mask: Path) -> int:
    completed = subprocess.run(
        [
            str(MRTRIX / "mrstats"),
            str(mask),
            "-output",
            "count",
            "-ignorezero",
            "-quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(float(completed.stdout.strip()))


def condition_count(
    image: Path,
    operator: str,
    threshold: float,
    mask: Path,
) -> int:
    calc = subprocess.Popen(
        [
            str(MRTRIX / "mrcalc"),
            str(image),
            str(threshold),
            operator,
            str(mask),
            "-mult",
            "-",
            "-quiet",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if calc.stdout is None:
        raise RuntimeError("mrcalc pipe unavailable")
    stats = subprocess.run(
        [
            str(MRTRIX / "mrstats"),
            "-",
            "-output",
            "count",
            "-ignorezero",
            "-quiet",
        ],
        stdin=calc.stdout,
        capture_output=True,
        text=True,
        check=False,
    )
    calc.stdout.close()
    _, calc_stderr = calc.communicate()
    if calc.returncode or stats.returncode:
        raise RuntimeError(
            "condition count failed: "
            + calc_stderr.decode("utf-8", errors="replace")
            + stats.stderr
        )
    return int(float(stats.stdout.strip() or "0"))


def image_stats(image: Path, mask: Path) -> dict[str, float]:
    completed = subprocess.run(
        [
            str(MRTRIX / "mrstats"),
            str(image),
            "-mask",
            str(mask),
            "-output",
            "min",
            "-output",
            "max",
            "-output",
            "mean",
            "-quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [float(value) for value in completed.stdout.split()]
    if len(values) != 3:
        raise RuntimeError(f"image stats differ: {image}")
    return {"min": values[0], "max": values[1], "mean": values[2]}


def difference_stats(
    candidate: Path,
    baseline: Path,
    mask: Path,
) -> dict[str, float]:
    calc = subprocess.Popen(
        [
            str(MRTRIX / "mrcalc"),
            str(candidate),
            str(baseline),
            "-subtract",
            "-abs",
            "-",
            "-quiet",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if calc.stdout is None:
        raise RuntimeError("difference mrcalc pipe unavailable")
    stats = subprocess.run(
        [
            str(MRTRIX / "mrstats"),
            "-",
            "-mask",
            str(mask),
            "-output",
            "mean",
            "-output",
            "max",
            "-quiet",
        ],
        stdin=calc.stdout,
        capture_output=True,
        text=True,
        check=False,
    )
    calc.stdout.close()
    _, calc_stderr = calc.communicate()
    if calc.returncode or stats.returncode:
        raise RuntimeError(
            "difference stats failed: "
            + calc_stderr.decode("utf-8", errors="replace")
            + stats.stderr
        )
    values = [float(value) for value in stats.stdout.split()]
    if len(values) != 2:
        raise RuntimeError("difference stats output differs")
    return {"mean_abs": values[0], "max_abs": values[1]}


def scalar_qc(
    maps: Mapping[str, Path],
    mask: Path,
) -> dict[str, Any]:
    mask_n = mask_voxel_count(mask)
    if mask_n <= 0:
        raise ValueError("empty mask")
    conditions = {
        "fa_below_zero": ("fa", "-lt", 0.0),
        "fa_above_one": ("fa", "-gt", 1.0),
        "md_below_zero": ("md", "-lt", 0.0),
        "md_above_0_01": (
            "md",
            "-gt",
            DIFFUSIVITY_MAXIMUM,
        ),
        "rd_below_zero": ("rd", "-lt", 0.0),
        "rd_above_0_01": (
            "rd",
            "-gt",
            DIFFUSIVITY_MAXIMUM,
        ),
        "ad_below_zero": ("ad", "-lt", 0.0),
        "ad_above_0_01": (
            "ad",
            "-gt",
            DIFFUSIVITY_MAXIMUM,
        ),
    }
    counts = {
        name: condition_count(maps[metric], operator, threshold, mask)
        for name, (metric, operator, threshold) in conditions.items()
    }
    fractions = {
        name: count / mask_n for name, count in counts.items()
    }
    failures = [
        (
            f"{name}={fraction:.9f}>"
            f"{RAW_ANOMALY_FRACTION_MAXIMUM:.9f}"
        )
        for name, fraction in fractions.items()
        if fraction > RAW_ANOMALY_FRACTION_MAXIMUM
    ]
    return {
        "status": "PASS" if not failures else "FAIL",
        "mask_voxels": mask_n,
        "ranges": {
            metric: image_stats(path, mask)
            for metric, path in maps.items()
        },
        "anomaly_counts": counts,
        "anomaly_fractions": fractions,
        "failures": failures,
    }


def run_method(
    attempt: Path,
    subject: Mapping[str, Any],
    method: Mapping[str, Any],
    *,
    nthreads: int,
) -> dict[str, Any]:
    unit = str(subject["unit"])
    name = str(method["name"])
    directory = attempt / "subjects" / unit / name
    directory.mkdir(parents=True, exist_ok=False)
    tensor = directory / "tensor.mif"
    maps = {
        metric: directory / f"{metric}.mif"
        for metric in ("fa", "md", "rd", "ad")
    }
    log_path = directory / "fit.log"
    environment = dict(os.environ)
    environment.update(
        {
            "MRTRIX_NTHREADS": str(nthreads),
            "OMP_NUM_THREADS": str(nthreads),
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(nthreads),
        }
    )
    command = [
        str(MRTRIX / "dwi2tensor"),
        str(subject["dwi"]),
        str(tensor),
        "-mask",
        str(subject["mask"]),
        "-iter",
        str(method["iwls_iterations"]),
        "-config",
        "BZeroThreshold",
        str(BZERO_THRESHOLD),
        "-nthreads",
        str(nthreads),
    ]
    if method["initial_fit"] == "OLS":
        command.append("-ols")
    started = time.monotonic()
    with log_path.open("x", encoding="utf-8") as log:
        run_command(
            command,
            log=log,
            outputs=(tensor,),
            env=environment,
        )
        run_command(
            [
                str(MRTRIX / "tensor2metric"),
                str(tensor),
                "-fa",
                str(maps["fa"]),
                "-adc",
                str(maps["md"]),
                "-rd",
                str(maps["rd"]),
                "-ad",
                str(maps["ad"]),
                "-nthreads",
                str(nthreads),
            ],
            log=log,
            outputs=tuple(maps.values()),
            env=environment,
        )
    qc = scalar_qc(maps, Path(str(subject["mask"])))
    differences = {
        metric: difference_stats(
            maps[metric],
            Path(str(subject["raw"][metric])),
            Path(str(subject["mask"])),
        )
        for metric in maps
    }
    replay_within_limits = None
    if method["production_replay"]:
        replay_within_limits = all(
            differences[metric]["mean_abs"]
            <= REPLAY_LIMITS[metric]["mean_abs"]
            and differences[metric]["max_abs"]
            <= REPLAY_LIMITS[metric]["max_abs"]
            for metric in maps
        )
    record = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_tensor_fit_counterfactual_method",
        "status": "PASS_EXECUTED",
        "generated_utc": utc_now(),
        "unit": unit,
        "role": subject["role"],
        "method": dict(method),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "production_outputs_modified": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "qc": qc,
        "difference_from_frozen_production_maps": differences,
        "production_replay_within_limits": replay_within_limits,
        "inputs": {
            "dwi": file_record(Path(str(subject["dwi"]))),
            "mask": file_record(Path(str(subject["mask"]))),
            "frozen_raw_maps": {
                metric: file_record(Path(str(path)))
                for metric, path in subject["raw"].items()
            },
        },
        "artifacts": {
            "tensor": file_record(tensor),
            **{
                metric: file_record(path)
                for metric, path in maps.items()
            },
            "log": file_record(log_path),
        },
    }
    atomic_json(directory / "result.json", record)
    return record


def execute(*, nthreads: int) -> dict[str, Any]:
    pre = preflight()
    attempt = (
        ATTEMPTS
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "-tensor-fit-counterfactual"
        )
    )
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_json(attempt / "preflight.json", pre)
    subjects = validated_subjects()
    results: list[dict[str, Any]] = []
    state = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_tensor_fit_counterfactual_state",
        "status": "RUNNING",
        "generated_utc": utc_now(),
        "attempt": str(attempt),
        "completed_method_n": 0,
        "expected_method_n": len(subjects) * len(METHODS),
    }
    atomic_json(attempt / "state.json", state)
    try:
        for subject in subjects:
            for method in METHODS:
                result = run_method(
                    attempt,
                    subject,
                    method,
                    nthreads=nthreads,
                )
                results.append(result)
                state["completed_method_n"] = len(results)
                state["last_completed"] = {
                    "unit": result["unit"],
                    "method": result["method"]["name"],
                    "qc_status": result["qc"]["status"],
                }
                atomic_json(attempt / "state.json", state)
    except Exception as exc:
        state.update(
            {
                "status": "FAIL",
                "completed_utc": utc_now(),
                "error": f"{type(exc).__name__}:{exc}",
            }
        )
        atomic_json(attempt / "state.json", state)
        raise

    by_unit_method = {
        (row["unit"], row["method"]["name"]): row
        for row in results
    }
    replay_valid = all(
        by_unit_method[(unit, "wls_iwls2_replay")][
            "production_replay_within_limits"
        ]
        is True
        for unit in (TARGET_UNIT, CONTROL_UNIT)
    )
    eligible_methods = [
        str(method["name"])
        for method in METHODS
        if not method["production_replay"]
        and all(
            by_unit_method[(unit, str(method["name"]))]["qc"][
                "status"
            ]
            == "PASS"
            for unit in (TARGET_UNIT, CONTROL_UNIT)
        )
    ]
    if not replay_valid:
        decision = "INCONCLUSIVE_PRODUCTION_REPLAY_MISMATCH"
    elif eligible_methods:
        decision = "UNIFORM_POLICY_CANARY_REQUIRED_BEFORE_PROMOTION"
    else:
        decision = "NO_TENSOR_FIT_VARIANT_RECOVERS_TARGET_AND_CONTROL"
    state.update(
        {
            "status": "PASS",
            "completed_utc": utc_now(),
            "error": None,
            "decision": decision,
            "eligible_followup_methods": eligible_methods,
        }
    )
    atomic_json(attempt / "state.json", state)
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_tensor_fit_counterfactual_attempt",
        "status": "PASS",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "production_outputs_modified": False,
        "raw_anomaly_fraction_gate": RAW_ANOMALY_FRACTION_MAXIMUM,
        "subject_n": len(subjects),
        "method_n_per_subject": len(METHODS),
        "completed_method_n": len(results),
        "production_replay_valid": replay_valid,
        "eligible_followup_methods": eligible_methods,
        "decision": decision,
        "results": results,
        "preflight": file_record(attempt / "preflight.json"),
        "final_state": file_record(attempt / "state.json"),
    }
    atomic_json(attempt / "attempt.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.nthreads <= 8:
        parser.error("--nthreads must be in 1..8")
    if args.self_test:
        if (
            len(METHODS) != 5
            or METHODS[0]["name"] != "wls_iwls2_replay"
            or RAW_ANOMALY_FRACTION_MAXIMUM != 0.01
            or TARGET_UNIT == CONTROL_UNIT
        ):
            raise AssertionError("tensor counterfactual constants differ")
        print("HCP379_TENSOR_FIT_COUNTERFACTUAL_SELF_TEST_PASS")
        return 0
    payload = execute(nthreads=args.nthreads) if args.execute else preflight()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") in {
        "READY_BOUNDED_TENSOR_FIT_DIAGNOSTIC",
        "PASS",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
