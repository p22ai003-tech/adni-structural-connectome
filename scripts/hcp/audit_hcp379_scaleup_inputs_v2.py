#!/usr/bin/env python3
"""Audit the 227-subject corrected-HCP379 scale-up queue without diagnosis.

The audit determines whether each subject can start from its existing Eddy
derivative, needs only a gradient-header repair, or requires upstream DWI or
anatomical recovery.  Historical HCP ``nodes_b0`` files are deliberately not
accepted as inputs; every routed subject will receive a newly mapped HCP379
atlas after the canary recipe is locked.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
DERIV = Path("/data/derivatives")
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
TRIAGE = (
    EXP
    / "research_audit/outputs/hcp_rerun_triage_v1/"
    "hcp_rerun_triage_manifest.csv"
)
ACQUISITION = (
    EXP
    / "research_audit/outputs/connectome_v2_input_manifest_v2.csv"
)
OUTPUT_ROOT = (
    EXP
    / "research_audit/outputs/hcp379_scaleup_input_audit_v2"
)
TARGET_ACTIONS = {
    "HCP_REMAP_PROBE_THEN_ESCALATE",
    "CORRECTED_PRETRACT_AND_TRACK_REBUILD",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        newline="",
        encoding="utf-8",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def command_output(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode, result.stdout.strip()


def fastsurfer_t1_path(unit: str) -> Path | None:
    subject = DERIV / "fastsurfer" / unit
    for path in (
        subject / "scripts/deep-seg.log",
        subject / "scripts/exectime.log",
    ):
        if not path.is_file():
            continue
        match = re.search(
            r"--t1\s+(\S+corrected_T1_\S+?\.nii(?:\.gz)?)",
            path.read_text(encoding="utf-8", errors="replace"),
        )
        if match:
            return Path(match.group(1)).expanduser().resolve()
    return None


def image_id(path: Path | None) -> str | None:
    if path is None:
        return None
    match = re.search(r"_I(\d+)\.nii(?:\.gz)?$", path.name)
    return match.group(1) if match else None


def parse_gradients(text: str) -> dict[str, Any]:
    rows: list[list[float]] = []
    for line in text.splitlines():
        values = line.split()
        if len(values) != 4:
            continue
        try:
            row = [float(value) for value in values]
        except ValueError:
            continue
        if all(math.isfinite(value) for value in row):
            rows.append(row)
    b0_count = sum(row[3] <= 50.0 for row in rows)
    diffusion = [row for row in rows if row[3] > 50.0]
    norm_failures = 0
    for row in diffusion:
        norm = math.sqrt(sum(value * value for value in row[:3]))
        norm_failures += int(not 0.80 <= norm <= 1.20)
    return {
        "gradient_rows": len(rows),
        "b0_rows": b0_count,
        "diffusion_rows": len(diffusion),
        "maximum_b_value": max((row[3] for row in rows), default=0.0),
        "diffusion_gradient_norm_failures": norm_failures,
    }


def clustered_bvals(values: list[float], tolerance: float = 100.0) -> tuple[
    list[float], list[int]
]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters or abs(value - sum(clusters[-1]) / len(clusters[-1])) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return (
        [sum(cluster) / len(cluster) for cluster in clusters],
        [len(cluster) for cluster in clusters],
    )


def sidecar_shells(path: Path) -> tuple[list[float], list[int]]:
    try:
        values = [
            float(value)
            for value in path.read_text(encoding="utf-8").split()
        ]
    except (OSError, ValueError):
        return [], []
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        return [], []
    return clustered_bvals(values)


def acquisition_by_unit() -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    rows = read_csv(ACQUISITION)
    if len(rows) != 530:
        raise ValueError(f"acquisition manifest has {len(rows)} rows, not 530")
    for row in rows:
        unit = f"{row['subject_id']}_I{row['dti_image_id']}"
        if unit in output:
            raise ValueError(f"duplicate acquisition identity: {unit}")
        for field in ("dti_source_id", "dti_raw_bundle_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(row[field])):
                raise ValueError(f"{unit}:{field} is not a SHA-256 identity")
        output[unit] = {
            "t1_image_id": str(row["t1_image_id"]),
            "t1_source_id": str(row["t1_source_id"]),
            "dti_source_id": str(row["dti_source_id"]),
            "dti_raw_bundle_sha256": str(row["dti_raw_bundle_sha256"]),
            "dwi_bvec_path": str(row.get("dwi_bvec_path", "")),
            "dwi_bval_path": str(row.get("dwi_bval_path", "")),
        }
    return output


def audit_unit(
    unit: str,
    *,
    recommended_action: str,
    acquisition: Mapping[str, str],
) -> dict[str, Any]:
    eddy = DERIV / "eddy" / f"{unit}_preproc.mif"
    fallback_bvec = DERIV / "mif_dwi" / f"{unit}.bvec"
    fallback_bval = DERIV / "mif_dwi" / f"{unit}.bval"
    response = DERIV / "fod" / unit / "wm_csd.txt"
    hcp = DERIV / "parc_hcpmmp1" / unit / "HCPMMP1+aseg.mgz"
    fastsurfer = DERIV / "fastsurfer" / unit
    observed_t1 = fastsurfer_t1_path(unit)
    expected_t1_id = str(acquisition["t1_image_id"])
    observed_t1_id = image_id(observed_t1)
    size_values: list[int] = []
    spacing_values: list[float] = []
    gradient_qc = {
        "gradient_rows": 0,
        "b0_rows": 0,
        "diffusion_rows": 0,
        "maximum_b_value": 0.0,
        "diffusion_gradient_norm_failures": 0,
    }
    shell_centroids: list[float] = []
    shell_sizes: list[int] = []
    failures: list[str] = []
    repairable: list[str] = []
    if not eddy.is_file() or eddy.stat().st_size <= 0:
        failures.append("eddy_derivative_missing")
    else:
        rc, size_text = command_output(
            [str(MRTRIX / "mrinfo"), str(eddy), "-size"]
        )
        if rc:
            failures.append("eddy_header_unreadable")
        else:
            try:
                size_values = [int(value) for value in size_text.split()]
            except ValueError:
                failures.append("eddy_size_unparseable")
            if len(size_values) != 4 or min(size_values) <= 0:
                failures.append("eddy_size_invalid")
        rc, spacing_text = command_output(
            [str(MRTRIX / "mrinfo"), str(eddy), "-spacing"]
        )
        if not rc:
            try:
                spacing_values = [
                    float(value) for value in spacing_text.split()[:3]
                ]
            except ValueError:
                spacing_values = []
        if (
            len(spacing_values) != 3
            or not all(
                math.isfinite(value) and value > 0
                for value in spacing_values
            )
        ):
            failures.append("eddy_spacing_invalid")
        rc, gradient_text = command_output(
            [str(MRTRIX / "mrinfo"), str(eddy), "-dwgrad"]
        )
        gradient_qc = parse_gradients(gradient_text if not rc else "")
        if (
            len(size_values) == 4
            and gradient_qc["gradient_rows"] != size_values[3]
        ):
            if fallback_bvec.is_file() and fallback_bval.is_file():
                repairable.append("gradient_header_repair_from_locked_sidecars")
                shell_centroids, shell_sizes = sidecar_shells(fallback_bval)
            else:
                failures.append("gradient_rows_do_not_match_dwi_volumes")
        elif (
            gradient_qc["b0_rows"] < 1
            or gradient_qc["diffusion_rows"] < 6
            or gradient_qc["diffusion_gradient_norm_failures"] > 0
        ):
            failures.append("gradient_table_physical_qc_failed")
        else:
            rc_shells, shells_text = command_output(
                [
                    str(MRTRIX / "mrinfo"),
                    str(eddy),
                    "-shell_bvalues",
                    "-config",
                    "BZeroThreshold",
                    "50",
                ]
            )
            rc_sizes, sizes_text = command_output(
                [
                    str(MRTRIX / "mrinfo"),
                    str(eddy),
                    "-shell_sizes",
                    "-config",
                    "BZeroThreshold",
                    "50",
                ]
            )
            if not rc_shells and not rc_sizes:
                try:
                    shell_centroids = [
                        float(value) for value in shells_text.split()
                    ]
                    shell_sizes = [
                        int(value) for value in sizes_text.split()
                    ]
                except ValueError:
                    shell_centroids, shell_sizes = [], []
        if not shell_centroids and fallback_bval.is_file():
            shell_centroids, shell_sizes = sidecar_shells(fallback_bval)

    anatomy_failures = []
    if observed_t1 is None or not observed_t1.is_file():
        anatomy_failures.append("fastsurfer_t1_identity_missing")
    elif observed_t1_id != expected_t1_id:
        anatomy_failures.append(
            f"fastsurfer_t1_id_mismatch:{observed_t1_id}:{expected_t1_id}"
        )
    if not hcp.is_file() or hcp.stat().st_size <= 0:
        anatomy_failures.append("hcp_source_parcellation_missing")
    for relative in (
        "mri/aseg.mgz",
        "surf/lh.sphere.reg",
        "surf/rh.sphere.reg",
    ):
        if not (fastsurfer / relative).is_file():
            anatomy_failures.append(
                "fastsurfer_anatomy_missing:" + relative.replace("/", "_")
            )
    failures.extend(anatomy_failures)

    response_status = "PRESENT" if response.is_file() else "REESTIMATE"
    if failures:
        if any(value.startswith("eddy_") for value in failures):
            route = "DWI_PREPROCESS_REQUIRED"
        elif anatomy_failures:
            route = "ANATOMY_RECOVERY_REQUIRED"
        else:
            route = "TECHNICAL_REVIEW_REQUIRED"
    elif repairable:
        route = "READY_AFTER_GRADIENT_HEADER_REPAIR"
    else:
        route = "READY_FOR_CORRECTED_PRETRACT_FROM_EDDY"
    return {
        "unit": unit,
        "diagnosis_labels_used": False,
        "source_recommended_action": recommended_action,
        "dti_source_id": acquisition["dti_source_id"],
        "dti_raw_bundle_sha256": acquisition["dti_raw_bundle_sha256"],
        "route": route,
        "failures": ";".join(failures),
        "repairable_actions": ";".join(repairable),
        "eddy_path": str(eddy),
        "eddy_size_bytes": eddy.stat().st_size if eddy.is_file() else 0,
        "dwi_size": "x".join(str(value) for value in size_values),
        "dwi_spacing_mm": "x".join(
            f"{value:.8g}" for value in spacing_values
        ),
        **gradient_qc,
        "shell_centroids_s_per_mm2": ";".join(
            f"{value:.8g}" for value in shell_centroids
        ),
        "shell_sizes": ";".join(str(value) for value in shell_sizes),
        "fallback_gradient_sidecars_present": (
            fallback_bvec.is_file() and fallback_bval.is_file()
        ),
        "response_status": response_status,
        "response_path": str(response),
        "expected_t1_image_id": expected_t1_id,
        "fastsurfer_t1_image_id": observed_t1_id or "",
        "fastsurfer_t1_identity_match": observed_t1_id == expected_t1_id,
        "fastsurfer_t1_path": str(observed_t1) if observed_t1 else "",
        "hcp_source_parcellation_path": str(hcp),
        "hcp_source_parcellation_present": hcp.is_file(),
        "historical_nodes_b0_reuse_allowed": False,
        "full_dwi_preprocessing_planned": route == "DWI_PREPROCESS_REQUIRED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("--workers must be in 1..32")
    acquisition = acquisition_by_unit()
    targets = {
        str(row["fsid"]): str(row["recommended_action"])
        for row in read_csv(TRIAGE)
        if str(row["recommended_action"]) in TARGET_ACTIONS
    }
    if len(targets) != 227 or len(set(targets)) != 227:
        raise ValueError(f"corrected-track target set is {len(targets)}, not 227")
    missing_manifest = sorted(set(targets) - set(acquisition))
    if missing_manifest:
        raise ValueError(
            f"acquisition manifest lacks target units: {missing_manifest}"
        )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                audit_unit,
                unit,
                recommended_action=action,
                acquisition=acquisition[unit],
            ): unit
            for unit, action in sorted(targets.items())
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{utc_now()}] {index}/227 {result['unit']} "
                f"{result['route']}",
                flush=True,
            )
    results.sort(key=lambda row: str(row["unit"]))
    route_counts = Counter(str(row["route"]) for row in results)
    summary = {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_scaleup_input_audit",
        "status": "PASS" if len(results) == 227 else "FAIL",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "acquisition_identity_source": str(ACQUISITION.resolve()),
        "target_n": 227,
        "audited_n": len(results),
        "route_counts": dict(sorted(route_counts.items())),
        "historical_nodes_b0_reuse_allowed": False,
        "routing_policy": {
            "eddy_valid": (
                "reuse Eddy, rebuild corrected T1-to-DWI anatomy, HCP379, "
                "FOD/tensors and tractography"
            ),
            "gradient_header_only": (
                "restore gradients from locked bvec/bval, then use the same "
                "corrected-pretract route"
            ),
            "eddy_invalid": "rerun DWI preprocessing before corrected pretract",
            "anatomy_invalid": (
                "repair exact-T1 anatomy/HCP source before corrected pretract"
            ),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output_root / "scaleup_input_audit.csv", results)
    atomic_json(args.output_root / "scaleup_input_audit_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
