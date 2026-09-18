#!/usr/bin/env python3
"""Audit an acquisition-stratified response remedy for 018_S_4868.

The exact acquisition-matched peer is selected from imaging metadata only.
Subject responses are estimated with fixed Dhollander settings, cross-applied,
and averaged.  All reconstructed images live in temporary storage.  The audit
does not modify production subjects or authorize recovery, and never reads
diagnosis, outcome, density, tractography, or matrices.
"""

from __future__ import annotations

import concurrent.futures
import csv
import importlib.util
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
SS3T_PYTHON = Path("/usr/bin/python3.9")
SS3T = Path("/home/ec2-user/MRtrix3Tissue/bin/ss3t_csd_beta1")
TARGET = "018_S_4868_I1026385"
PEER = "018_S_6207_I963728"
TARGET_ROOT = (
    Path("/data/derivatives/hcp379_v2/")
    / "corrected_legacy_tensor_recovery4/subjects"
    / TARGET
)
TARGET_DWI = TARGET_ROOT / "05_model/dwi_fod_shells.mif"
TARGET_MASK = TARGET_ROOT / "01_dwi/dwi_brain_mask.mif"
PEER_EDDY = Path(f"/data/derivatives/eddy/{PEER}_preproc.mif")
SOURCE_METADATA = (
    EXP / "research_audit/outputs/historical_source_preflight_proxy_v1.csv"
)
BASE_AUDIT_SOURCE = (
    EXP / "research_audit/audit_hcp379_mtnormalise_failure_018_v1.py"
)
BASE_AUDIT_JSON = (
    EXP
    / "research_audit/outputs/hcp379_mtnormalise_failure_018_v1/"
    "audit.json"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/hcp379_response_stratum_018_v1/"
    "audit.json"
)
HARD_NEGATIVE_TOLERANCE = 1.0e-4
SOFT_NEGATIVE_TOLERANCE = 1.0e-5
MAXIMUM_SOFT_NEGATIVE_FRACTION = 0.001


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_response_stratum_base_audit", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_AUDIT_SOURCE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run(command: list[str], log: Path, *, timeout: int = 1800) -> int:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env={
            **os.environ,
            "PATH": ":".join(
                (
                    str(MRTRIX),
                    "/home/ec2-user/exp/.envs/ants-2.6.5/bin",
                    os.environ.get("PATH", ""),
                )
            ),
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "8",
        },
    )
    log.write_text(
        json.dumps(command)
        + "\n"
        + completed.stdout
        + completed.stderr
        + f"\nELAPSED_SECONDS={time.monotonic() - started:.6f}\n",
        encoding="utf-8",
    )
    return completed.returncode


def acquisition_rows() -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    with SOURCE_METADATA.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            unit = f"{row['subject_id']}_I{row['dti_image_id']}"
            if unit in {TARGET, PEER}:
                selected[unit] = {
                    key: row[key]
                    for key in (
                        "dti_manufacturer",
                        "dti_protocol_key",
                        "n_bvals",
                        "n_b0",
                        "n_nonzero",
                        "shells_round50",
                        "historical_mif_path",
                    )
                }
    if set(selected) != {TARGET, PEER}:
        raise ValueError("exact target and peer metadata unavailable")
    comparison = (
        "dti_manufacturer",
        "dti_protocol_key",
        "n_bvals",
        "n_b0",
        "n_nonzero",
        "shells_round50",
    )
    if any(
        selected[TARGET][key] != selected[PEER][key]
        for key in comparison
    ):
        raise ValueError("target and peer acquisition metadata differ")
    if (
        selected[TARGET]["n_bvals"],
        selected[TARGET]["n_b0"],
        selected[TARGET]["n_nonzero"],
    ) != ("36", "1", "35"):
        raise ValueError("expected one-b0 plus 35-direction stratum")
    return selected


def build_peer_input(work: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    root = work / "peer_input"
    root.mkdir()
    bias = root / "dwi_biascorr.mif"
    mask = root / "dwi_brain_mask.mif"
    dwi = root / "dwi_fod_shells.mif"
    commands = (
        (
            [
                str(MRTRIX / "dwibiascorrect"),
                "ants",
                str(PEER_EDDY),
                str(bias),
                "-nthreads",
                "8",
            ],
            (bias,),
        ),
        (
            [
                str(MRTRIX / "dwi2mask"),
                str(bias),
                str(mask),
                "-config",
                "BZeroThreshold",
                "50.0",
                "-nthreads",
                "8",
            ],
            (mask,),
        ),
        (
            [
                str(MRTRIX / "dwiextract"),
                str(bias),
                str(dwi),
                "-shells",
                "0,1000",
                "-config",
                "BZeroThreshold",
                "50.0",
                "-nthreads",
                "8",
                "-force",
            ],
            (dwi,),
        ),
    )
    records: list[dict[str, Any]] = []
    for index, (command, outputs) in enumerate(commands):
        log = root / f"{index:02d}.log"
        rc = run(command, log)
        records.append(
            {
                "command": command,
                "returncode": rc,
                "log_tail": log.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[-10:],
            }
        )
        if rc or not all(path.is_file() for path in outputs):
            raise RuntimeError(f"peer input command failed: {command[0]}")
    return dwi, mask, records


def estimate_response(
    unit: str, dwi: Path, mask: Path, work: Path
) -> dict[str, Any]:
    root = work / f"response_{unit}"
    root.mkdir()
    response = {
        tissue: root / f"{tissue}.txt"
        for tissue in ("wm", "gm", "csf")
    }
    command = [
        str(MRTRIX / "dwi2response"),
        "dhollander",
        str(dwi),
        str(response["wm"]),
        str(response["gm"]),
        str(response["csf"]),
        "-mask",
        str(mask),
        "-voxels",
        str(root / "voxels.mif"),
        "-shells",
        "0,1000",
        "-lmax",
        "0,6",
        "-nthreads",
        "8",
        "-force",
    ]
    log = root / "response.log"
    rc = run(command, log)
    if rc or not all(path.is_file() for path in response.values()):
        raise RuntimeError(f"response estimation failed for {unit}")
    return {
        "unit": unit,
        "paths": response,
        "contents": {
            tissue: path.read_text(encoding="utf-8")
            for tissue, path in response.items()
        },
        "log_tail": log.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-15:],
    }


def mean_response(
    target: dict[str, Any],
    peer: dict[str, Any],
    work: Path,
) -> dict[str, Path]:
    root = work / "response_stratum_mean"
    root.mkdir()
    result: dict[str, Path] = {}
    for tissue in ("wm", "gm", "csf"):
        destination = root / f"{tissue}.txt"
        log = root / f"{tissue}.log"
        command = [
            str(MRTRIX / "responsemean"),
            str(target["paths"][tissue]),
            str(peer["paths"][tissue]),
            str(destination),
        ]
        if run(command, log) or not destination.is_file():
            raise RuntimeError(f"response mean failed for {tissue}")
        result[tissue] = destination
    return result


def threshold_count(
    image: Path, mask: Path, threshold: float, work: Path, name: str
) -> int:
    binary = work / f"{name}.mif"
    rc = run(
        [
            str(MRTRIX / "mrcalc"),
            str(image),
            str(-threshold),
            "-lt",
            str(binary),
            "-quiet",
        ],
        work / f"{name}.log",
    )
    if rc:
        raise RuntimeError(f"threshold count failed: {name}")
    values = BASE.mrstats(binary, mask)
    return round(float(values["mean"]) * int(values["count"]))


def model_case(
    *,
    name: str,
    unit: str,
    dwi: Path,
    mask: Path,
    response: dict[str, Path],
    work: Path,
    canary_correlation_range: dict[str, float],
) -> dict[str, Any]:
    root = work / f"model_{name}"
    root.mkdir()
    command = [
        str(SS3T_PYTHON),
        str(SS3T),
        str(dwi),
        str(response["wm"]),
        str(root / "wmfod.mif"),
        str(response["gm"]),
        str(root / "gm.mif"),
        str(response["csf"]),
        str(root / "csf.mif"),
        "-mask",
        str(mask),
        "-lmax",
        "6",
        "-config",
        "BZeroThreshold",
        "50.0",
        "-nthreads",
        "8",
    ]
    ss3t_log = root / "ss3t.log"
    ss3t_rc = run(command, ss3t_log)
    result: dict[str, Any] = {
        "name": name,
        "unit": unit,
        "ss3t_returncode": ss3t_rc,
        "ss3t_log_tail": ss3t_log.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-15:],
    }
    if ss3t_rc:
        result["status"] = "FAIL_SS3T"
        return result

    factors = root / "factors.txt"
    norm = root / "norm.mif"
    norm_command = [
        str(MRTRIX / "mtnormalise"),
        str(root / "wmfod.mif"),
        str(root / "wmfod_norm.mif"),
        str(root / "gm.mif"),
        str(root / "gm_norm.mif"),
        str(root / "csf.mif"),
        str(root / "csf_norm.mif"),
        "-mask",
        str(mask),
        "-check_factors",
        str(factors),
        "-check_norm",
        str(norm),
        "-nthreads",
        "8",
        "-force",
    ]
    norm_log = root / "mtnormalise.log"
    norm_rc = run(norm_command, norm_log)
    factor_values = (
        [float(value) for value in factors.read_text().split()]
        if factors.is_file()
        else []
    )
    result.update(
        {
            "mtnormalise_returncode": norm_rc,
            "balance_factors": factor_values,
            "mtnormalise_log_tail": norm_log.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()[-15:],
        }
    )
    if norm_rc:
        result["status"] = "FAIL_MTNORMALISE"
        return result

    raw_stats = BASE.tissue_stats(root, mask)
    correlations = BASE.pairwise_tissue_correlations(
        root, mask, raw_stats
    )
    wm_l0 = root / "wmfod_l0.mif"
    if run(
        [
            str(MRTRIX / "mrconvert"),
            str(root / "wmfod_norm.mif"),
            str(wm_l0),
            "-coord",
            "3",
            "0",
            "-quiet",
        ],
        root / "wmfod_l0.log",
    ):
        raise RuntimeError("WM-FOD l=0 extraction failed")
    hard_n = threshold_count(
        wm_l0,
        mask,
        HARD_NEGATIVE_TOLERANCE,
        root,
        "below_hard",
    )
    soft_n = threshold_count(
        wm_l0,
        mask,
        SOFT_NEGATIVE_TOLERANCE,
        root,
        "below_soft",
    )
    count = int(BASE.mrstats(wm_l0, mask)["count"])
    soft_fraction = soft_n / count
    correlation = correlations["wmfod_l0__csf"]
    correlation_pass = (
        canary_correlation_range["minimum"]
        <= correlation
        <= canary_correlation_range["maximum"]
    )
    policy_pass = (
        hard_n == 0
        and soft_fraction <= MAXIMUM_SOFT_NEGATIVE_FRACTION
    )
    factors_pass = (
        len(factor_values) == 3
        and all(value > 0 for value in factor_values)
    )
    result.update(
        {
            "raw_tissue_stats": raw_stats,
            "raw_tissue_correlations": correlations,
            "wm_csf_correlation_canary_range": (
                canary_correlation_range
            ),
            "wm_csf_correlation_pass": correlation_pass,
            "wmfod_l0_below_hard_tolerance_n": hard_n,
            "wmfod_l0_below_soft_tolerance_n": soft_n,
            "wmfod_l0_soft_negative_fraction": soft_fraction,
            "fod_numerical_policy_pass": policy_pass,
            "all_three_balance_factors_positive": factors_pass,
            "status": (
                "PASS_TECHNICAL_DIAGNOSTIC"
                if correlation_pass and policy_pass and factors_pass
                else "FAIL_TECHNICAL_DIAGNOSTIC"
            ),
        }
    )
    return result


def main() -> int:
    metadata = acquisition_rows()
    base_audit = BASE.load_json(BASE_AUDIT_JSON)
    if base_audit.get("status") != "PASS_DIAGNOSTIC_COMPLETE":
        raise ValueError("base normalization audit does not pass")
    correlation_range = (
        base_audit["reference_cohort"]["correlation_summary"][
            "wmfod_l0__csf"
        ]
    )
    if not all(path.is_file() for path in (TARGET_DWI, TARGET_MASK, PEER_EDDY)):
        raise FileNotFoundError("response-stratum inputs incomplete")

    with tempfile.TemporaryDirectory(
        prefix="hcp379-response-stratum-018-"
    ) as temporary_name:
        work = Path(temporary_name)
        peer_dwi, peer_mask, peer_input_records = build_peer_input(work)
        target_response = estimate_response(
            TARGET, TARGET_DWI, TARGET_MASK, work
        )
        peer_response = estimate_response(
            PEER, peer_dwi, peer_mask, work
        )
        stratum_response = mean_response(
            target_response, peer_response, work
        )
        cases = (
            {
                "name": "target_self_response",
                "unit": TARGET,
                "dwi": TARGET_DWI,
                "mask": TARGET_MASK,
                "response": target_response["paths"],
            },
            {
                "name": "target_peer_cross_response",
                "unit": TARGET,
                "dwi": TARGET_DWI,
                "mask": TARGET_MASK,
                "response": peer_response["paths"],
            },
            {
                "name": "target_stratum_mean_response",
                "unit": TARGET,
                "dwi": TARGET_DWI,
                "mask": TARGET_MASK,
                "response": stratum_response,
            },
            {
                "name": "peer_stratum_mean_response",
                "unit": PEER,
                "dwi": peer_dwi,
                "mask": peer_mask,
                "response": stratum_response,
            },
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=2
        ) as executor:
            futures = [
                executor.submit(
                    model_case,
                    **case,
                    work=work,
                    canary_correlation_range=correlation_range,
                )
                for case in cases
            ]
            results = [future.result() for future in futures]
        response_contents = {
            "target": target_response["contents"],
            "peer": peer_response["contents"],
            "stratum_mean": {
                tissue: path.read_text(encoding="utf-8")
                for tissue, path in stratum_response.items()
            },
        }

    by_name = {row["name"]: row for row in results}
    stratum_pair_pass = all(
        by_name[name]["status"] == "PASS_TECHNICAL_DIAGNOSTIC"
        for name in (
            "target_stratum_mean_response",
            "peer_stratum_mean_response",
        )
    )
    cross_pass = (
        by_name["target_peer_cross_response"]["status"]
        == "PASS_TECHNICAL_DIAGNOSTIC"
    )
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_response_stratum_018_root_cause_audit",
        "status": "PASS_DIAGNOSTIC_COMPLETE",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "tractography_generated": False,
        "matrix_generated": False,
        "subject_outputs_modified": False,
        "recovery_authorized": False,
        "target_unit": TARGET,
        "acquisition_matched_peer": PEER,
        "acquisition_metadata": metadata,
        "peer_input_reconstruction": peer_input_records,
        "response_functions": response_contents,
        "fixed_model_cases": results,
        "criteria": {
            "positive_three_tissue_balance_factors_required": True,
            "wm_csf_correlation_must_be_within_canary_range": (
                correlation_range
            ),
            "wmfod_l0_hard_negative_n_required": 0,
            "wmfod_l0_maximum_soft_negative_fraction": (
                MAXIMUM_SOFT_NEGATIVE_FRACTION
            ),
        },
        "criteria_results": {
            "peer_response_cross_generalizes_to_target": cross_pass,
            "stratum_mean_passes_target_and_peer": stratum_pair_pass,
        },
        "next_action": (
            "Build and independently validate a non-overwriting, exact "
            "two-subject acquisition-stratum recovery preflight; do not "
            "promote either subject-specific response alone."
            if cross_pass and stratum_pair_pass
            else "Do not use the acquisition-stratum response; continue the "
            "upstream model audit."
        ),
        "interpretation_boundary": (
            "Passing demonstrates a bounded technical remedy for this exact "
            "acquisition stratum. It does not authorize tractography, select "
            "a production recipe, relax QC, or establish inferential "
            "comparability without a separately validated recovery."
        ),
        "records": {
            "base_audit": BASE.file_record(BASE_AUDIT_JSON),
            "base_audit_script": BASE.file_record(BASE_AUDIT_SOURCE),
            "source_metadata": BASE.file_record(SOURCE_METADATA),
            "target_dwi": BASE.file_record(TARGET_DWI),
            "target_mask": BASE.file_record(TARGET_MASK),
            "peer_eddy": BASE.file_record(PEER_EDDY),
            "ss3t": BASE.file_record(SS3T),
            "implementation": BASE.file_record(Path(__file__)),
        },
    }
    BASE.atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
