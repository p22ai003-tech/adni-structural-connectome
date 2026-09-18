#!/home/ec2-user/fsl/bin/python
"""Run a diagnosis-blind acquisition-aware FOD-order canary.

The frozen production recipe uses an even spherical-harmonic order of six,
which requires 28 unique antipodally canonicalised diffusion directions.
Two already-failed non-canary units provide bounded tests of both lower-order
routes required by the exact-530 acquisition audit:

* 031_S_0618_I1229293: six unique directions, therefore lmax=2.
* 031_S_4721_I1093826: fifteen unique directions, therefore lmax=4.

Only the FOD modelling and normalisation stages are recomputed.  Inputs,
pooled response functions, masks, numerical thresholds, diagnosis blindness,
and all downstream gates remain unchanged.  Large outputs are written to a
new attempt directory; no production or historical file is overwritten.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP = Path("/data/derivatives/hcp379_v2")
INPUT_ROOT = HCP / "corrected_scaleup_recovery4"
OUTPUT_ROOT = HCP / "acquisition_aware_fod_order_canary_v1"
AUDIT_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_acquisition_aware_fod_order_canary_v1"
)
FOD_AUDIT_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_fod_order_compatibility_v1"
)
FOD_AUDIT = FOD_AUDIT_ROOT / "audit.json"
FOD_TABLE = FOD_AUDIT_ROOT / "fod_order_compatibility.csv"
FOD_VALIDATION = FOD_AUDIT_ROOT / "validation.json"
FAILURE_LEDGER = (
    EXP
    / "research_audit/outputs/"
    "hcp379_pretract_failure_recovery_ledger_v1/ledger.json"
)
FAILURE_VALIDATION = FAILURE_LEDGER.with_name("validation.json")
RESPONSE_MANIFEST = (
    Path("/data/derivatives/scforge_v2/")
    / "h04a_r1_recovery_20260719_retry4/"
    "frozen_calibration_retry4_v2/"
    "frozen_response_calibration_manifest.json"
)
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
SS3T_PYTHON = Path("/usr/bin/python3.9")
SS3T = Path("/home/ec2-user/MRtrix3Tissue/bin/ss3t_csd_beta1")
TARGETS = {
    "031_S_0618_I1229293": 2,
    "031_S_4721_I1093826": 4,
}
BZERO_THRESHOLD = 50.0
SOFT_NEGATIVE_TOLERANCE = 1.0e-5
HARD_NEGATIVE_TOLERANCE = 1.0e-4
MAXIMUM_SOFT_NEGATIVE_FRACTION = 1.0e-3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def require_validation(path: Path) -> dict[str, Any]:
    value = load_json(path)
    passed = int(
        value.get("checks_passed", value.get("passed_checks", -1))
    )
    total = int(
        value.get("checks_total", value.get("total_checks", -1))
    )
    if value.get("status") != "PASS" or total <= 0 or passed != total:
        raise ValueError(f"validation differs: {path}")
    return value


def read_fod_rows() -> dict[str, dict[str, str]]:
    with FOD_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 530 or len({row["unit"] for row in rows}) != 530:
        raise ValueError("FOD-order table is not exact unique 530")
    return {row["unit"]: row for row in rows}


def response_paths() -> dict[str, Path]:
    manifest = load_json(RESPONSE_MANIFEST)
    if (
        manifest.get("status") != "PASS"
        or manifest.get("diagnosis_labels_used") is not False
        or manifest.get("dummy_response_files_used") is not False
        or manifest.get("fod_shell_compatibility", {}).get(
            "selected_shells_s_per_mm2"
        )
        != [0.0, 1000.0]
        or float(
            manifest["fod_shell_compatibility"][
                "bzero_threshold_s_per_mm2"
            ]
        )
        != BZERO_THRESHOLD
    ):
        raise ValueError("frozen response manifest differs")
    result: dict[str, Path] = {}
    for tissue in ("wm", "gm", "csf"):
        record = manifest.get("pooled_responses", {}).get(tissue)
        if not isinstance(record, dict):
            raise ValueError(f"missing pooled {tissue} response")
        path = Path(str(record["path"])).resolve()
        if file_record(path) != {
            "path": str(path),
            "size_bytes": int(record["size_bytes"]),
            "sha256": str(record["sha256"]),
        }:
            raise ValueError(f"pooled {tissue} response drift")
        result[tissue] = path
    return result


def preflight() -> dict[str, Any]:
    audit = load_json(FOD_AUDIT)
    require_validation(FOD_VALIDATION)
    require_validation(FAILURE_VALIDATION)
    rows = read_fod_rows()
    responses = response_paths()
    units: list[dict[str, Any]] = []
    for unit, expected_lmax in TARGETS.items():
        row = rows.get(unit)
        current_state_path = (
            INPUT_ROOT / "qc/subjects" / f"{unit}.json"
        )
        current_state = load_json(current_state_path)
        prior_archive_record = current_state.get(
            "prior_state_archive"
        )
        if not isinstance(prior_archive_record, dict):
            raise ValueError(f"missing prior failure archive: {unit}")
        prior_archive_path = Path(
            str(prior_archive_record["path"])
        ).resolve()
        if file_record(prior_archive_path) != {
            "path": str(prior_archive_path),
            "size_bytes": int(prior_archive_record["size_bytes"]),
            "sha256": str(prior_archive_record["sha256"]),
        }:
            raise ValueError(f"prior failure archive drift: {unit}")
        prior_archive = load_json(prior_archive_path)
        prior_failure = prior_archive.get("prior_state")
        if (
            not isinstance(row, dict)
            or row.get("status")
            != "HOLD_ACQUISITION_AWARE_LMAX_REQUIRED"
            or int(row["recommended_lmax"]) != expected_lmax
            or row.get("lane") != "corrected_core"
            or current_state.get("status")
            != "HOLD_FOD_ORDER_RECOVERY_REQUIRED"
            or current_state.get("recommended_lmax") != expected_lmax
            or not isinstance(prior_failure, dict)
            or prior_failure.get("status") != "FAIL_PRETRACT_RECOVERY4"
            or "wmfod_l0_below_hard_tolerance"
            not in str(prior_failure.get("error", ""))
        ):
            raise ValueError(f"canary binding differs: {unit}")
        subject = INPUT_ROOT / "subjects" / unit
        dwi = subject / "05_model/dwi_fod_shells.mif"
        mask = subject / "01_dwi/dwi_brain_mask.mif"
        units.append(
            {
                "unit": unit,
                "recommended_lmax": expected_lmax,
                "unique_direction_n": int(
                    row[
                        "selected_shell_unique_antipodal_direction_n"
                    ]
                ),
                "required_coefficient_n": (
                    (expected_lmax + 1) * (expected_lmax + 2) // 2
                ),
                "source_dwi": file_record(dwi),
                "source_mask": file_record(mask),
                "current_hold_state": file_record(
                    current_state_path
                ),
                "prior_failure_state": file_record(
                    prior_archive_path
                ),
                "prior_failure_class": (
                    "FOD_L0_HARD_NEGATIVE_BURDEN"
                ),
            }
        )
    if (
        audit.get("status")
        != "PASS_AUDIT_COMPLETE_WITH_ACQUISITION_AWARE_HOLDS"
        or audit.get("acquisition_aware_hold_n") != 10
        or audit.get("recommended_lmax_counts")
        != {"2": 2, "4": 8, "6": 520}
    ):
        raise ValueError("FOD-order audit summary differs")
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_acquisition_aware_fod_order_canary_preflight"
        ),
        "status": "READY",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "historical_outputs_modified": False,
        "target_n": 2,
        "target_units": units,
        "policy": {
            "only_changed_parameter": "FOD even spherical-harmonic order",
            "lmax_selection": (
                "largest of 2,4,6 whose even-SH coefficient count does "
                "not exceed the unique antipodal direction count"
            ),
            "pooled_response_calibration_unchanged": True,
            "wm_response_order_policy": (
                "deterministically retain pooled l=0..selected_lmax "
                "coefficients and drop only unsupported higher-order terms"
            ),
            "three_tissue_normalisation_unchanged": True,
            "soft_negative_tolerance": SOFT_NEGATIVE_TOLERANCE,
            "hard_negative_tolerance": HARD_NEGATIVE_TOLERANCE,
            "maximum_soft_negative_fraction": (
                MAXIMUM_SOFT_NEGATIVE_FRACTION
            ),
            "qc_threshold_relaxation_allowed": False,
            "tractography_generated": False,
            "matrices_generated": False,
        },
        "records": {
            "fod_order_audit": file_record(FOD_AUDIT),
            "fod_order_table": file_record(FOD_TABLE),
            "fod_order_validation": file_record(FOD_VALIDATION),
            "failure_ledger": file_record(FAILURE_LEDGER),
            "failure_validation": file_record(FAILURE_VALIDATION),
            "frozen_response_manifest": file_record(RESPONSE_MANIFEST),
            "pooled_responses": {
                tissue: file_record(path)
                for tissue, path in responses.items()
            },
            "ss3t": file_record(SS3T),
            "implementation": file_record(Path(__file__)),
        },
    }


def run_command(
    command: list[str],
    *,
    log: Any,
    outputs: tuple[Path, ...] = (),
) -> None:
    log.write(f"\n[{utc_now()}] COMMAND {json.dumps(command)}\n")
    log.flush()
    completed = subprocess.run(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {command[0]}"
        )
    missing = [
        str(path)
        for path in outputs
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if missing:
        raise RuntimeError("missing command outputs: " + ",".join(missing))


def mrstats_values(
    path: Path,
    mask: Path,
    *outputs: str,
) -> list[float]:
    command = [
        str(MRTRIX / "mrstats"),
        str(path),
        "-mask",
        str(mask),
    ]
    for output in outputs:
        command.extend(["-output", output])
    command.append("-quiet")
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return [float(value) for value in completed.stdout.split()]


def mask_count(mask: Path) -> int:
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
    path: Path,
    operator: str,
    threshold: float,
    mask: Path,
) -> int:
    completed = subprocess.run(
        [
            str(MRTRIX / "mrcalc"),
            str(path),
            str(threshold),
            operator,
            str(mask),
            "-mult",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    counted = subprocess.run(
        [
            str(MRTRIX / "mrstats"),
            "-",
            "-output",
            "count",
            "-ignorezero",
            "-quiet",
        ],
        input=completed.stdout,
        check=True,
        capture_output=True,
    )
    return int(float(counted.stdout.decode().strip() or "0"))


def image_size(path: Path) -> list[int]:
    completed = subprocess.run(
        [str(MRTRIX / "mrinfo"), str(path), "-size"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [int(value) for value in completed.stdout.split()]


def derive_wm_response(
    source: Path,
    destination: Path,
    *,
    lmax: int,
) -> dict[str, Any]:
    retained_coefficients = lmax // 2 + 1
    lines = source.read_text(encoding="utf-8").splitlines()
    output_lines: list[str] = [
        (
            "# derived_from_frozen_pooled_response: "
            f"{source.resolve()}"
        ),
        (
            "# deterministic_order_truncation: "
            f"retain_l0_through_l{lmax}"
        ),
    ]
    numeric_rows: list[list[float]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            output_lines.append(stripped)
            continue
        values = [float(value) for value in stripped.split()]
        if len(values) != 4:
            raise ValueError(
                f"frozen WM response coefficient count differs: {values}"
            )
        numeric_rows.append(values)
        output_lines.append(
            " ".join(
                f"{value:.15f}"
                for value in values[:retained_coefficients]
            )
        )
    if len(numeric_rows) != 2:
        raise ValueError("frozen WM response shell rows differ")
    with destination.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(output_lines) + "\n")
    return {
        "source": file_record(source),
        "derived": file_record(destination),
        "source_coefficient_n": 4,
        "retained_coefficient_n": retained_coefficients,
        "retained_orders": list(range(0, lmax + 1, 2)),
        "dropped_orders": list(range(lmax + 2, 7, 2)),
        "values_replay": [
            {
                "source": row,
                "derived": row[:retained_coefficients],
            }
            for row in numeric_rows
        ],
    }


def process_unit(
    unit_spec: Mapping[str, Any],
    *,
    attempt: Path,
    responses: Mapping[str, Path],
    nthreads: int,
) -> dict[str, Any]:
    unit = str(unit_spec["unit"])
    lmax = int(unit_spec["recommended_lmax"])
    unit_root = attempt / "subjects" / unit
    unit_root.mkdir(parents=True, exist_ok=False)
    dwi = Path(str(unit_spec["source_dwi"]["path"]))
    mask = Path(str(unit_spec["source_mask"]["path"]))
    outputs = {
        "wm_response": (
            unit_root / f"pooled_response_wm_lmax{lmax}.txt"
        ),
        "wmfod": unit_root / "wmfod.mif",
        "gm": unit_root / "gm.mif",
        "csf": unit_root / "csf.mif",
        "wmfod_norm": unit_root / "wmfod_norm.mif",
        "gm_norm": unit_root / "gm_norm.mif",
        "csf_norm": unit_root / "csf_norm.mif",
        "wmfod_l0": unit_root / "wmfod_l0.mif",
        "norm_field": unit_root / "normalisation_field.mif",
        "balance_factors": unit_root / "balance_factors.txt",
        "log": unit_root / "commands.log",
    }
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_acquisition_aware_fod_order_canary_subject"
        ),
        "status": "RUNNING",
        "generated_utc": utc_now(),
        "unit": unit,
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "recommended_lmax": lmax,
        "required_coefficient_n": int(
            unit_spec["required_coefficient_n"]
        ),
        "unique_direction_n": int(unit_spec["unique_direction_n"]),
        "source_dwi": unit_spec["source_dwi"],
        "source_mask": unit_spec["source_mask"],
        "prior_failure_state": unit_spec["prior_failure_state"],
    }
    try:
        response_derivation = derive_wm_response(
            responses["wm"],
            outputs["wm_response"],
            lmax=lmax,
        )
        with outputs["log"].open("x", encoding="utf-8") as log:
            run_command(
                [
                    str(SS3T_PYTHON),
                    str(SS3T),
                    str(dwi),
                    str(outputs["wm_response"]),
                    str(outputs["wmfod"]),
                    str(responses["gm"]),
                    str(outputs["gm"]),
                    str(responses["csf"]),
                    str(outputs["csf"]),
                    "-mask",
                    str(mask),
                    "-lmax",
                    str(lmax),
                    "-config",
                    "BZeroThreshold",
                    str(BZERO_THRESHOLD),
                    "-nthreads",
                    str(nthreads),
                ],
                log=log,
                outputs=(
                    outputs["wmfod"],
                    outputs["gm"],
                    outputs["csf"],
                ),
            )
            run_command(
                [
                    str(MRTRIX / "mtnormalise"),
                    str(outputs["wmfod"]),
                    str(outputs["wmfod_norm"]),
                    str(outputs["gm"]),
                    str(outputs["gm_norm"]),
                    str(outputs["csf"]),
                    str(outputs["csf_norm"]),
                    "-mask",
                    str(mask),
                    "-check_norm",
                    str(outputs["norm_field"]),
                    "-check_factors",
                    str(outputs["balance_factors"]),
                    "-nthreads",
                    str(nthreads),
                ],
                log=log,
                outputs=(
                    outputs["wmfod_norm"],
                    outputs["gm_norm"],
                    outputs["csf_norm"],
                    outputs["norm_field"],
                    outputs["balance_factors"],
                ),
            )
            run_command(
                [
                    str(MRTRIX / "mrconvert"),
                    str(outputs["wmfod_norm"]),
                    str(outputs["wmfod_l0"]),
                    "-coord",
                    "3",
                    "0",
                ],
                log=log,
                outputs=(outputs["wmfod_l0"],),
            )

        voxel_n = mask_count(mask)
        factors = [
            float(value)
            for value in outputs["balance_factors"].read_text().split()
        ]
        tissue_paths = {
            "wmfod_l0": outputs["wmfod_l0"],
            "gm": outputs["gm_norm"],
            "csf": outputs["csf_norm"],
        }
        ranges = {
            tissue: {
                "min": values[0],
                "max": values[1],
            }
            for tissue, path in tissue_paths.items()
            for values in [mrstats_values(path, mask, "min", "max")]
        }
        counts: dict[str, int] = {}
        fractions: dict[str, float] = {}
        failures: list[str] = []
        for tissue, path in tissue_paths.items():
            soft = condition_count(
                path, "-lt", -SOFT_NEGATIVE_TOLERANCE, mask
            )
            hard = condition_count(
                path, "-lt", -HARD_NEGATIVE_TOLERANCE, mask
            )
            counts[f"{tissue}_below_tolerance"] = soft
            counts[f"{tissue}_below_hard_tolerance"] = hard
            fractions[f"{tissue}_below_tolerance_fraction"] = (
                soft / voxel_n
            )
            if hard:
                failures.append(
                    f"{tissue}_below_hard_tolerance={hard}"
                )
            if soft / voxel_n > MAXIMUM_SOFT_NEGATIVE_FRACTION:
                failures.append(
                    f"{tissue}_below_tolerance_fraction="
                    f"{soft / voxel_n:.12g}"
                )
        coefficient_n = image_size(outputs["wmfod_norm"])[3]
        if coefficient_n != int(unit_spec["required_coefficient_n"]):
            failures.append(
                f"wmfod_coefficient_n={coefficient_n}"
            )
        if ranges["wmfod_l0"]["max"] <= SOFT_NEGATIVE_TOLERANCE:
            failures.append("wmfod_l0_all_zero")
        if len(factors) != 3 or not all(
            value > 0 and value == value for value in factors
        ):
            failures.append("nonpositive_or_invalid_balance_factors")
        all_sh_values = mrstats_values(
            outputs["wmfod_norm"], mask, "min", "max"
        )
        if (
            not all_sh_values
            or len(all_sh_values) % 2
            or any(
                value != value or abs(value) == float("inf")
                for value in all_sh_values
            )
        ):
            failures.append("wmfod_nonfinite")
        result.update(
            {
                "status": "PASS" if not failures else "FAIL",
                "completed_utc": utc_now(),
                "elapsed_seconds": round(
                    time.monotonic() - started, 3
                ),
                "mask_voxel_n": voxel_n,
                "wmfod_coefficient_n": coefficient_n,
                "wm_response_derivation": response_derivation,
                "balance_factors": factors,
                "tissue_ranges": ranges,
                "numerical_counts": counts,
                "soft_negative_fractions": fractions,
                "failures": failures,
                "artifacts": {
                    name: file_record(path)
                    for name, path in outputs.items()
                    if name != "log"
                },
                "log": file_record(outputs["log"]),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "ERROR",
                "completed_utc": utc_now(),
                "elapsed_seconds": round(
                    time.monotonic() - started, 3
                ),
                "error": f"{type(exc).__name__}:{exc}",
            }
        )
    atomic_json(unit_root / "result.json", result)
    return result


def execute(*, workers: int, nthreads: int) -> dict[str, Any]:
    if workers not in (1, 2) or not 1 <= nthreads <= 4:
        raise ValueError("workers must be 1..2 and nthreads must be 1..4")
    pre = preflight()
    attempt = (
        OUTPUT_ROOT
        / "attempts"
        / (
            datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S.%fZ"
            )
            + "-fod-order-canary"
        )
    )
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_json(attempt / "preflight.json", pre)
    responses = {
        tissue: Path(str(record["path"]))
        for tissue, record in pre["records"]["pooled_responses"].items()
    }
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_unit,
                unit,
                attempt=attempt,
                responses=responses,
                nthreads=nthreads,
            ): str(unit["unit"])
            for unit in pre["target_units"]
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[{utc_now()}] {result['unit']} {result['status']} "
                f"lmax={result['recommended_lmax']} "
                f"elapsed={result.get('elapsed_seconds')}s",
                flush=True,
            )
    results.sort(key=lambda row: str(row["unit"]))
    passed = sum(row.get("status") == "PASS" for row in results)
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_acquisition_aware_fod_order_canary_summary"
        ),
        "status": (
            "PASS_CANARY_BOTH_ORDERS"
            if passed == len(TARGETS)
            else "FAIL_CANARY"
        ),
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "historical_outputs_modified": False,
        "target_n": len(TARGETS),
        "passed_n": passed,
        "failed_n": len(TARGETS) - passed,
        "tractography_generated": False,
        "matrices_generated": False,
        "attempt_root": str(attempt.resolve()),
        "preflight": file_record(attempt / "preflight.json"),
        "units": [
            {
                "unit": row["unit"],
                "status": row["status"],
                "recommended_lmax": row["recommended_lmax"],
                "unique_direction_n": row["unique_direction_n"],
                "wmfod_coefficient_n": row.get(
                    "wmfod_coefficient_n"
                ),
                "balance_factors": row.get("balance_factors"),
                "tissue_ranges": row.get("tissue_ranges"),
                "numerical_counts": row.get("numerical_counts"),
                "soft_negative_fractions": row.get(
                    "soft_negative_fractions"
                ),
                "failures": row.get("failures"),
                "error": row.get("error"),
                "result": file_record(
                    attempt
                    / "subjects"
                    / str(row["unit"])
                    / "result.json"
                ),
            }
            for row in results
        ],
        "decision": (
            "FOD-only technical canary passed for lmax=2 and lmax=4. "
            "A bounded tractography/connectome canary is still required "
            "before promoting the route to all ten held units."
            if passed == len(TARGETS)
            else "Do not promote acquisition-aware FOD-order recovery."
        ),
        "records": {
            "implementation": file_record(Path(__file__)),
            "fod_order_audit": file_record(FOD_AUDIT),
            "frozen_response_manifest": file_record(RESPONSE_MANIFEST),
        },
    }
    atomic_json(attempt / "summary.json", summary)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(AUDIT_ROOT / "summary.json", summary)
    return summary


def self_test() -> None:
    assert (2 + 1) * (2 + 2) // 2 == 6
    assert (4 + 1) * (4 + 2) // 2 == 15
    assert (6 + 1) * (6 + 2) // 2 == 28
    assert TARGETS == {
        "031_S_0618_I1229293": 2,
        "031_S_4721_I1093826": 4,
    }
    assert SOFT_NEGATIVE_TOLERANCE == 1.0e-5
    assert HARD_NEGATIVE_TOLERANCE == 1.0e-4
    assert MAXIMUM_SOFT_NEGATIVE_FRACTION == 1.0e-3
    print("HCP379_ACQUISITION_AWARE_FOD_ORDER_CANARY_SELF_TEST_PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.execute:
        value = preflight()
    else:
        value = execute(
            workers=args.workers,
            nthreads=args.nthreads,
        )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if str(value["status"]).startswith(("READY", "PASS")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
