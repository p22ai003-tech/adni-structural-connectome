#!/usr/bin/env python3
"""Diagnosis-blind root-cause audit of one HCP379 mtnormalise failure.

This script does not alter a subject directory or authorize a recovery.  It
replays four fixed normalisation configurations in temporary storage, compares
the raw tissue-component summaries with the preserved 15-canary reference
models, and writes a hash-bound diagnostic record.  No diagnosis, outcome,
connectome density, tractography, or matrix is read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
UNIT = "018_S_4868_I1026385"
SUBJECT = (
    Path("/data/derivatives/hcp379_v2/")
    / "corrected_legacy_tensor_recovery4/subjects"
    / UNIT
)
MASK = SUBJECT / "01_dwi/dwi_brain_mask.mif"
MODEL = SUBJECT / "05_model"
FAILURE_LOG = (
    Path("/data/derivatives/hcp379_v2/")
    / "corrected_legacy_tensor_recovery4/logs/subjects"
    / f"{UNIT}.log"
)
LEDGER = (
    EXP
    / "research_audit/outputs/hcp379_pretract_failure_recovery_ledger_v1/"
    "ledger.json"
)
LEDGER_VALIDATION = LEDGER.with_name("validation.json")
REFERENCE_ROOT = (
    Path("/data/derivatives/scforge_v2/")
    / "h04a_r1_recovery_20260719_retry4/subjects"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/hcp379_mtnormalise_failure_018_v1/"
    "audit.json"
)
VARIANTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("three_tissue", ("wmfod", "gm", "csf")),
    ("wm_gm", ("wmfod", "gm")),
    ("wm_csf", ("wmfod", "csf")),
    ("wm_only", ("wmfod",)),
)


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


def mrstats(path: Path, mask: Path) -> dict[str, float | int]:
    fields = ("count", "mean", "std", "min", "max", "median")
    command = [str(MRTRIX / "mrstats"), str(path), "-mask", str(mask)]
    for field in fields:
        command.extend(("-output", field))
    command.append("-quiet")
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # For WM FOD the first volume is the l=0/DC coefficient.
    values = completed.stdout.strip().splitlines()[0].split()
    if len(values) != len(fields):
        raise ValueError(f"mrstats output differs for {path}")
    return {
        "count": int(float(values[0])),
        **{
            field: float(value)
            for field, value in zip(fields[1:], values[1:])
        },
    }


def tissue_stats(model: Path, mask: Path) -> dict[str, dict[str, float | int]]:
    return {
        "wmfod_l0": mrstats(model / "wmfod.mif", mask),
        "gm": mrstats(model / "gm.mif", mask),
        "csf": mrstats(model / "csf.mif", mask),
    }


def pairwise_tissue_correlations(
    model: Path,
    mask: Path,
    stats: Mapping[str, Mapping[str, float | int]],
) -> dict[str, float]:
    with tempfile.TemporaryDirectory(
        prefix="hcp379-tissue-correlation-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        wm_l0 = temporary / "wmfod_l0.mif"
        subprocess.run(
            [
                str(MRTRIX / "mrconvert"),
                str(model / "wmfod.mif"),
                str(wm_l0),
                "-coord",
                "3",
                "0",
                "-quiet",
            ],
            check=True,
            timeout=120,
        )
        sources = {
            "wmfod_l0": wm_l0,
            "gm": model / "gm.mif",
            "csf": model / "csf.mif",
        }
        pairs = (
            ("wmfod_l0", "gm"),
            ("wmfod_l0", "csf"),
            ("gm", "csf"),
        )
        result: dict[str, float] = {}
        for left, right in pairs:
            product = temporary / f"{left}__{right}.mif"
            subprocess.run(
                [
                    str(MRTRIX / "mrcalc"),
                    str(sources[left]),
                    str(sources[right]),
                    "-mult",
                    str(product),
                    "-quiet",
                ],
                check=True,
                timeout=120,
            )
            product_mean = float(mrstats(product, mask)["mean"])
            count = int(stats[left]["count"])
            left_std = float(stats[left]["std"])
            right_std = float(stats[right]["std"])
            # mrstats reports the sample SD; convert to the population SD to
            # pair it with E[XY] - E[X]E[Y].
            correction = ((count - 1) / count) ** 0.5
            denominator = left_std * right_std * correction * correction
            correlation = (
                (
                    product_mean
                    - float(stats[left]["mean"])
                    * float(stats[right]["mean"])
                )
                / denominator
                if denominator > 0
                else 0.0
            )
            result[f"{left}__{right}"] = correlation
        return result


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    low = int(location)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (
        location - low
    )


def summarize_reference(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for tissue in ("wmfod_l0", "gm", "csf"):
        result[tissue] = {}
        for field in ("mean", "median", "std", "min", "max"):
            values = [
                float(row["stats"][tissue][field])
                for row in rows
            ]
            result[tissue][field] = {
                "minimum": min(values),
                "p05": quantile(values, 0.05),
                "median": quantile(values, 0.50),
                "p95": quantile(values, 0.95),
                "maximum": max(values),
            }
    return result


def summarize_correlations(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    keys = sorted(rows[0]["correlations"])
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(row["correlations"][key]) for row in rows]
        result[key] = {
            "minimum": min(values),
            "p05": quantile(values, 0.05),
            "median": quantile(values, 0.50),
            "p95": quantile(values, 0.95),
            "maximum": max(values),
        }
    return result


def fixed_variant(
    temporary: Path,
    name: str,
    tissues: tuple[str, ...],
) -> dict[str, Any]:
    root = temporary / name
    root.mkdir()
    command = [str(MRTRIX / "mtnormalise")]
    outputs: dict[str, Path] = {}
    for tissue in tissues:
        source = MODEL / f"{tissue}.mif"
        destination = root / f"{tissue}_norm.mif"
        outputs[tissue] = destination
        command.extend((str(source), str(destination)))
    norm = root / "norm_field.mif"
    factors = root / "factors.txt"
    command.extend(
        (
            "-mask",
            str(MASK),
            "-check_norm",
            str(norm),
            "-check_factors",
            str(factors),
            "-nthreads",
            "4",
            "-force",
        )
    )
    started = time.monotonic()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    elapsed = time.monotonic() - started
    factor_values: list[float] = []
    if factors.is_file():
        factor_values = [
            float(value) for value in factors.read_text().split()
        ]
    output_stats = {
        tissue: mrstats(path, MASK)
        for tissue, path in outputs.items()
        if path.is_file() and path.stat().st_size > 0
    }
    return {
        "name": name,
        "tissues": list(tissues),
        "returncode": completed.returncode,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "elapsed_seconds": elapsed,
        "balance_factors": factor_values,
        "all_balance_factors_positive": bool(factor_values)
        and all(value > 0 for value in factor_values),
        "normalisation_field": (
            mrstats(norm, MASK)
            if norm.is_file() and norm.stat().st_size > 0
            else None
        ),
        "normalised_tissue_stats": output_stats,
        "log_tail": (completed.stdout + completed.stderr).splitlines()[-20:],
    }


def main() -> int:
    validation = load_json(LEDGER_VALIDATION)
    if (
        validation.get("status") != "PASS"
        or validation.get("passed_checks")
        != validation.get("total_checks")
    ):
        raise ValueError("current failure ledger validation does not pass")
    ledger = load_json(LEDGER)
    target_rows = [
        row
        for row in ledger.get("failed_units", [])
        if row.get("unit") == UNIT
    ]
    if (
        len(target_rows) != 1
        or target_rows[0].get("failure_class")
        != "MTNORMALISE_NONPOSITIVE_TISSUE_BALANCE"
    ):
        raise ValueError("exact normalization target differs")
    required = (
        MASK,
        MODEL / "wmfod.mif",
        MODEL / "gm.mif",
        MODEL / "csf.mif",
        FAILURE_LOG,
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("normalization audit inputs incomplete")

    reference_rows: list[dict[str, Any]] = []
    for subject in sorted(REFERENCE_ROOT.iterdir()):
        model = subject / "05_model"
        mask = subject / "01_dwi/dwi_brain_mask.mif"
        expected = (
            model / "wmfod.mif",
            model / "gm.mif",
            model / "csf.mif",
            model / "wmfod_norm.mif",
            model / "gm_norm.mif",
            model / "csf_norm.mif",
            mask,
        )
        if not all(path.is_file() for path in expected):
            continue
        stats = tissue_stats(model, mask)
        reference_rows.append(
            {
                "unit": subject.name,
                "stats": stats,
                "correlations": pairwise_tissue_correlations(
                    model, mask, stats
                ),
            }
        )
    if len(reference_rows) != 15:
        raise ValueError(
            f"exact 15-canary reference required, observed {len(reference_rows)}"
        )

    raw_stats = tissue_stats(MODEL, MASK)
    raw_correlations = pairwise_tissue_correlations(
        MODEL, MASK, raw_stats
    )
    reference_summary = summarize_reference(reference_rows)
    reference_correlation_summary = summarize_correlations(reference_rows)
    within_reference_range = {
        tissue: {
            field: (
                float(reference_summary[tissue][field]["minimum"])
                <= float(raw_stats[tissue][field])
                <= float(reference_summary[tissue][field]["maximum"])
            )
            for field in ("mean", "median", "std", "max")
        }
        for tissue in ("wmfod_l0", "gm", "csf")
    }
    correlations_within_reference_range = {
        key: (
            float(summary["minimum"])
            <= float(raw_correlations[key])
            <= float(summary["maximum"])
        )
        for key, summary in reference_correlation_summary.items()
    }
    failure_text = FAILURE_LOG.read_text(
        encoding="utf-8", errors="replace"
    )
    matches = re.findall(r"Balance factors:\s+([^\n]+)", failure_text)
    original_factors = (
        [float(value) for value in matches[-1].split()]
        if matches
        else []
    )
    with tempfile.TemporaryDirectory(
        prefix="hcp379-mtnormalise-audit-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        variants = [
            fixed_variant(temporary, name, tissues)
            for name, tissues in VARIANTS
        ]

    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_mtnormalise_failure_root_cause_audit",
        "status": "PASS_DIAGNOSTIC_COMPLETE",
        "generated_utc": utc_now(),
        "unit": UNIT,
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "tractography_generated": False,
        "matrix_generated": False,
        "subject_outputs_modified": False,
        "recovery_authorized": False,
        "original_three_tissue_balance_factors": original_factors,
        "raw_tissue_stats": raw_stats,
        "raw_tissue_correlations": raw_correlations,
        "reference_cohort": {
            "definition": (
                "the exact 15 preserved Phase-B canary models with complete "
                "raw and three-tissue-normalized WM, GM, and CSF outputs"
            ),
            "n": len(reference_rows),
            "units": [row["unit"] for row in reference_rows],
            "summary": reference_summary,
            "correlation_summary": reference_correlation_summary,
        },
        "target_within_reference_range": within_reference_range,
        "target_correlations_within_reference_range": (
            correlations_within_reference_range
        ),
        "fixed_temporary_variants": variants,
        "interpretation_boundary": (
            "This diagnostic identifies numerical/model behavior only. A "
            "successful reduced-tissue variant does not establish cross-subject "
            "comparability and does not authorize use in the final cohort."
        ),
        "records": {
            "ledger": file_record(LEDGER),
            "ledger_validation": file_record(LEDGER_VALIDATION),
            "failure_log": file_record(FAILURE_LOG),
            "mask": file_record(MASK),
            "wmfod": file_record(MODEL / "wmfod.mif"),
            "gm": file_record(MODEL / "gm.mif"),
            "csf": file_record(MODEL / "csf.mif"),
            "script": file_record(Path(__file__)),
        },
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
