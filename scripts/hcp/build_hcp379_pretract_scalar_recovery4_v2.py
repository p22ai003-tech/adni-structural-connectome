#!/usr/bin/env python3
"""Build physically bounded Recovery4 scalar maps for the 15-unit canary.

Raw Recovery3 tensor maps are never changed.  Their non-physical voxel burden
is measured inside the locked DWI mask and must remain at or below the
diagnosis-blind 1% technical gate used by the existing HCP379 tensor-repair
route.  Passing maps are projected to the already locked physical ranges
(FA 0..1; diffusivities 0..0.01 mm2/s) in a separate Recovery4 root.

The script also fixes the Recovery3 WM-FOD l=0 extraction defect by using a
temporary filename that retains the .mif extension and records small numerical
negative values explicitly.  Numerical tissue negatives are guarded by both a
hard magnitude bound and a maximum below-tolerance voxel fraction.  This
avoids rejecting an otherwise valid subject because of one float-scale voxel
while still rejecting spatially extensive or materially negative tissue
estimates.  No diagnosis or outcome field is read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
OUTPUT_ROOT = Path(
    "/data/derivatives/hcp379_v2/pretract_scalar_recovery4"
)
BASE_ATLAS_SCRIPT = (
    EXP / "scripts/hcp/hcp379_corrected_atlas_canary_v2.py"
)
TENSOR_HELPERS_SCRIPT = EXP / "scripts/hcp/hcp_tensor_repair_v2.py"
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
RAW_ANOMALY_FRACTION_MAXIMUM = 0.01
FOD_TISSUE_NUMERICAL_TOLERANCE = 1.0e-5
FOD_TISSUE_HARD_NEGATIVE_TOLERANCE = 1.0e-4
# A soft-negative burden is treated as a numerical condition only when it is
# both spatially rare and bounded in magnitude.  The 0.1% ceiling remains ten
# times stricter than the locked 1% raw-tensor anomaly allowance, while the
# independent 1e-4 hard magnitude gate prevents materially negative tissue
# estimates from being accepted regardless of their spatial extent.
FOD_TISSUE_BELOW_TOLERANCE_FRACTION_MAXIMUM = 1.0e-3
DIFFUSIVITY_MAXIMUM = 0.01


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module("hcp379_corrected_atlas_canary_v2", BASE_ATLAS_SCRIPT)
TENSOR = load_module("hcp_tensor_repair_v2", TENSOR_HELPERS_SCRIPT)


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
        raise ValueError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


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


def run(
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
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError("command outputs missing: " + ",".join(missing))


def mask_voxel_count(mask: Path) -> int:
    result = subprocess.run(
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
    return int(float(result.stdout.strip()))


def image_range(path: Path, mask: Path) -> dict[str, float]:
    return TENSOR.mrstats_range(path, mask)


def all_volume_range(path: Path, mask: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            str(MRTRIX / "mrstats"),
            str(path),
            "-mask",
            str(mask),
            "-output",
            "min",
            "-output",
            "max",
            "-quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [float(value) for value in result.stdout.split()]
    if not values or len(values) % 2:
        raise RuntimeError(f"unexpected multi-volume mrstats output: {path}")
    minima = values[0::2]
    maxima = values[1::2]
    if not all(
        value == value and abs(value) != float("inf")
        for value in values
    ):
        raise RuntimeError(f"non-finite multi-volume range: {path}")
    return {
        "volume_count": len(minima),
        "min": min(minima),
        "max": max(maxima),
        "all_finite": True,
    }


def condition_count(
    path: Path,
    operator: str,
    threshold: float,
    mask: Path,
) -> int:
    return TENSOR.condition_count(path, operator, threshold, mask)


def paths_for(unit: str, output_root: Path) -> dict[str, Any]:
    source = RUN_ROOT / "subjects" / unit
    output = output_root / "subjects" / unit
    raw = {
        metric: source / "05_model" / f"{metric}.mif"
        for metric in ("fa", "md", "rd", "ad")
    }
    bounded = {
        metric: output / "05_model" / f"{metric}_bounded_recovery4.mif"
        for metric in ("fa", "md", "rd", "ad")
    }
    return {
        "source": source,
        "output": output,
        "log": output / "logs/scalar_recovery4.log",
        "result": output / "06_preflight/scalar_recovery4_qc.json",
        "mask": source / "01_dwi/dwi_brain_mask.mif",
        "wmfod": source / "05_model/wmfod_norm.mif",
        "gm": source / "05_model/gm_norm.mif",
        "csf": source / "05_model/csf_norm.mif",
        "wmfod_l0": (
            output / "05_model/wmfod_l0_qc_recovery4.mif"
        ),
        "raw": raw,
        "bounded": bounded,
    }


def existing_pass(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "PASS":
            return None
        for artifact in record.get("artifacts", {}).values():
            candidate = Path(str(artifact["path"]))
            if (
                not candidate.is_file()
                or candidate.is_symlink()
                or candidate.stat().st_size != int(artifact["size_bytes"])
                or sha256_file(candidate) != artifact["sha256"]
            ):
                return None
        return record
    except Exception:
        return None


def process_unit(
    unit: str,
    *,
    output_root: Path,
) -> dict[str, Any]:
    paths = paths_for(unit, output_root)
    cached = existing_pass(paths["result"])
    if cached is not None:
        return cached
    paths["result"].parent.mkdir(parents=True, exist_ok=True)
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    for path in paths["bounded"].values():
        path.parent.mkdir(parents=True, exist_ok=True)
    required = (
        paths["mask"],
        paths["wmfod"],
        paths["gm"],
        paths["csf"],
        *paths["raw"].values(),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing scalar Recovery4 input: " + ",".join(missing)
        )
    started = time.monotonic()
    failures: list[str] = []
    with paths["log"].open("a", encoding="utf-8") as log:
        l0_partial = paths["wmfod_l0"].with_name(
            paths["wmfod_l0"].stem + ".partial.mif"
        )
        l0_partial.unlink(missing_ok=True)
        run(
            [
                str(MRTRIX / "mrconvert"),
                str(paths["wmfod"]),
                str(l0_partial),
                "-coord",
                "3",
                "0",
                "-force",
            ],
            log=log,
            outputs=(l0_partial,),
        )
        os.replace(l0_partial, paths["wmfod_l0"])

        mask_count = mask_voxel_count(paths["mask"])
        if mask_count <= 0:
            raise ValueError("DWI mask is empty")
        raw_ranges = {
            metric: image_range(path, paths["mask"])
            for metric, path in paths["raw"].items()
        }
        anomaly_counts = {
            "fa_below_zero": condition_count(
                paths["raw"]["fa"], "-lt", 0.0, paths["mask"]
            ),
            "fa_above_one": condition_count(
                paths["raw"]["fa"], "-gt", 1.0, paths["mask"]
            ),
            "md_below_zero": condition_count(
                paths["raw"]["md"], "-lt", 0.0, paths["mask"]
            ),
            "md_above_0_01": condition_count(
                paths["raw"]["md"],
                "-gt",
                DIFFUSIVITY_MAXIMUM,
                paths["mask"],
            ),
            "rd_below_zero": condition_count(
                paths["raw"]["rd"], "-lt", 0.0, paths["mask"]
            ),
            "rd_above_0_01": condition_count(
                paths["raw"]["rd"],
                "-gt",
                DIFFUSIVITY_MAXIMUM,
                paths["mask"],
            ),
            "ad_below_zero": condition_count(
                paths["raw"]["ad"], "-lt", 0.0, paths["mask"]
            ),
            "ad_above_0_01": condition_count(
                paths["raw"]["ad"],
                "-gt",
                DIFFUSIVITY_MAXIMUM,
                paths["mask"],
            ),
        }
        anomaly_fractions = {
            name: count / mask_count
            for name, count in anomaly_counts.items()
        }
        for name, fraction in anomaly_fractions.items():
            if fraction > RAW_ANOMALY_FRACTION_MAXIMUM:
                failures.append(
                    f"{name}={fraction:.9f}>"
                    f"{RAW_ANOMALY_FRACTION_MAXIMUM:.9f}"
                )

        run(
            [
                str(MRTRIX / "mrcalc"),
                str(paths["raw"]["fa"]),
                "0",
                "-max",
                "1",
                "-min",
                str(paths["bounded"]["fa"]),
                "-force",
            ],
            log=log,
            outputs=(paths["bounded"]["fa"],),
        )
        for metric in ("md", "rd", "ad"):
            run(
                [
                    str(MRTRIX / "mrcalc"),
                    str(paths["raw"][metric]),
                    "0",
                    "-max",
                    str(DIFFUSIVITY_MAXIMUM),
                    "-min",
                    str(paths["bounded"][metric]),
                    "-force",
                ],
                log=log,
                outputs=(paths["bounded"][metric],),
            )

    bounded_ranges = {
        metric: image_range(path, paths["mask"])
        for metric, path in paths["bounded"].items()
    }
    if not (
        0.0 <= bounded_ranges["fa"]["min"]
        <= bounded_ranges["fa"]["max"]
        <= 1.0
    ):
        failures.append(f"bounded_fa_range={bounded_ranges['fa']}")
    for metric in ("md", "rd", "ad"):
        values = bounded_ranges[metric]
        if not (
            0.0
            <= values["min"]
            <= values["max"]
            <= DIFFUSIVITY_MAXIMUM
        ):
            failures.append(f"bounded_{metric}_range={values}")

    fod_all = all_volume_range(paths["wmfod"], paths["mask"])
    l0_range = image_range(paths["wmfod_l0"], paths["mask"])
    gm_range = image_range(paths["gm"], paths["mask"])
    csf_range = image_range(paths["csf"], paths["mask"])
    numerical_counts = {
        "wmfod_l0_negative": condition_count(
            paths["wmfod_l0"], "-lt", 0.0, paths["mask"]
        ),
        "wmfod_l0_below_tolerance": condition_count(
            paths["wmfod_l0"],
            "-lt",
            -FOD_TISSUE_NUMERICAL_TOLERANCE,
            paths["mask"],
        ),
        "wmfod_l0_below_hard_tolerance": condition_count(
            paths["wmfod_l0"],
            "-lt",
            -FOD_TISSUE_HARD_NEGATIVE_TOLERANCE,
            paths["mask"],
        ),
        "gm_below_tolerance": condition_count(
            paths["gm"],
            "-lt",
            -FOD_TISSUE_NUMERICAL_TOLERANCE,
            paths["mask"],
        ),
        "gm_below_hard_tolerance": condition_count(
            paths["gm"],
            "-lt",
            -FOD_TISSUE_HARD_NEGATIVE_TOLERANCE,
            paths["mask"],
        ),
        "csf_below_tolerance": condition_count(
            paths["csf"],
            "-lt",
            -FOD_TISSUE_NUMERICAL_TOLERANCE,
            paths["mask"],
        ),
        "csf_below_hard_tolerance": condition_count(
            paths["csf"],
            "-lt",
            -FOD_TISSUE_HARD_NEGATIVE_TOLERANCE,
            paths["mask"],
        ),
    }
    below_tolerance_fractions = {
        name: numerical_counts[name] / mask_count
        for name in (
            "wmfod_l0_below_tolerance",
            "gm_below_tolerance",
            "csf_below_tolerance",
        )
    }
    if l0_range["max"] <= FOD_TISSUE_NUMERICAL_TOLERANCE:
        failures.append("wmfod_l0_all_zero")
    for tissue in ("wmfod_l0", "gm", "csf"):
        soft_name = f"{tissue}_below_tolerance"
        hard_name = f"{tissue}_below_hard_tolerance"
        if numerical_counts[hard_name]:
            failures.append(f"{hard_name}={numerical_counts[hard_name]}")
        if (
            below_tolerance_fractions[soft_name]
            > FOD_TISSUE_BELOW_TOLERANCE_FRACTION_MAXIMUM
        ):
            failures.append(
                f"{soft_name}_fraction="
                f"{below_tolerance_fractions[soft_name]:.12g}"
            )

    record = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_pretract_scalar_recovery4_qc"
        ),
        "status": "PASS" if not failures else "FAIL",
        "unit": unit,
        "diagnosis_labels_used": False,
        "completed_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "non_overwriting": True,
        "mask_voxels": mask_count,
        "raw_tensor_policy": {
            "maximum_fraction_per_anomaly_condition": (
                RAW_ANOMALY_FRACTION_MAXIMUM
            ),
            "gate_source": (
                "existing HCP379-v2 tensor repair technical policy"
            ),
            "fa_physical_range": [0.0, 1.0],
            "diffusivity_physical_range_mm2_per_s": [
                0.0,
                DIFFUSIVITY_MAXIMUM,
            ],
            "bounded_maps_are_separate_outputs": True,
            "raw_maps_retained_unchanged": True,
        },
        "raw_tensor_ranges": raw_ranges,
        "raw_anomaly_counts": anomaly_counts,
        "raw_anomaly_fractions": anomaly_fractions,
        "bounded_tensor_ranges": bounded_ranges,
        "fod_tissue_policy": {
            "all_wmfod_sh_coefficients_must_be_finite": True,
            "all_wmfod_sh_coefficients_must_be_nonnegative": False,
            "l0_must_be_nonzero": True,
            "l0_gm_csf_numerical_nonnegativity_tolerance": (
                FOD_TISSUE_NUMERICAL_TOLERANCE
            ),
            "l0_gm_csf_hard_negative_tolerance": (
                FOD_TISSUE_HARD_NEGATIVE_TOLERANCE
            ),
            "maximum_below_tolerance_voxel_fraction": (
                FOD_TISSUE_BELOW_TOLERANCE_FRACTION_MAXIMUM
            ),
            "magnitude_and_fraction_guard": True,
        },
        "fod_tissue_ranges": {
            "wmfod_all_sh_coefficients": fod_all,
            "wmfod_l0": l0_range,
            "gm": gm_range,
            "csf": csf_range,
        },
        "fod_tissue_numerical_counts": numerical_counts,
        "fod_tissue_below_tolerance_fractions": (
            below_tolerance_fractions
        ),
        "failures": sorted(set(failures)),
        "source_artifacts": {
            "mask": file_record(paths["mask"]),
            "wmfod_norm": file_record(paths["wmfod"]),
            "gm_norm": file_record(paths["gm"]),
            "csf_norm": file_record(paths["csf"]),
            **{
                f"{metric}_raw": file_record(path)
                for metric, path in paths["raw"].items()
            },
        },
        "artifacts": {
            "wmfod_l0": file_record(paths["wmfod_l0"]),
            **{
                f"{metric}_bounded": file_record(path)
                for metric, path in paths["bounded"].items()
            },
        },
    }
    atomic_json(paths["result"], record)
    return record


def write_summary(
    output_root: Path,
    results: list[dict[str, Any]],
) -> Path:
    ordered = sorted(results, key=lambda row: str(row["unit"]))
    passed = sum(row.get("status") == "PASS" for row in ordered)
    maximum_anomaly = max(
        (
            max(row.get("raw_anomaly_fractions", {}).values(), default=0.0)
            for row in ordered
        ),
        default=0.0,
    )
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_pretract_scalar_recovery4_summary"
        ),
        "status": "PASS" if passed == 15 else "FAIL",
        "updated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "target_n": 15,
        "passed_n": passed,
        "failed_n": len(ordered) - passed,
        "raw_anomaly_fraction_gate": RAW_ANOMALY_FRACTION_MAXIMUM,
        "maximum_observed_raw_anomaly_fraction": maximum_anomaly,
        "bounded_maps_written": True,
        "units": [
            {
                "unit": row["unit"],
                "status": row.get("status"),
                "maximum_raw_anomaly_fraction": max(
                    row.get("raw_anomaly_fractions", {}).values(),
                    default=None,
                ),
                "bounded_tensor_ranges": row.get(
                    "bounded_tensor_ranges"
                ),
                "wmfod_l0_range": row.get(
                    "fod_tissue_ranges", {}
                ).get("wmfod_l0"),
                "failures": row.get("failures"),
                "result": file_record(
                    output_root
                    / "subjects"
                    / str(row["unit"])
                    / "06_preflight/scalar_recovery4_qc.json"
                ),
            }
            for row in ordered
        ],
    }
    path = output_root / "scalar_recovery4_summary.json"
    atomic_json(path, summary)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be in 1..8")
    units = BASE.load_units()
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[{utc_now()}] SCALAR_RECOVERY4_START "
        f"n={len(units)} workers={args.workers}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_unit,
                unit,
                output_root=args.output_root,
            ): unit
            for unit in units
        }
        for index, future in enumerate(as_completed(futures), start=1):
            unit = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "schema_version": "1.0.0",
                    "record_type": (
                        "diagnosis_blind_hcp379_pretract_"
                        "scalar_recovery4_qc"
                    ),
                    "status": "FAIL",
                    "unit": unit,
                    "diagnosis_labels_used": False,
                    "completed_utc": utc_now(),
                    "failures": [f"{type(exc).__name__}:{exc}"],
                }
                atomic_json(
                    args.output_root
                    / "subjects"
                    / unit
                    / "06_preflight/scalar_recovery4_qc.json",
                    result,
                )
            results.append(result)
            print(
                f"[{utc_now()}] scalar {index}/{len(units)} "
                f"{unit} {result.get('status')} "
                f"max_raw_anomaly={max(result.get('raw_anomaly_fractions', {}).values(), default=None)} "
                f"failures={result.get('failures')}",
                flush=True,
            )
    summary_path = write_summary(args.output_root, results)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
