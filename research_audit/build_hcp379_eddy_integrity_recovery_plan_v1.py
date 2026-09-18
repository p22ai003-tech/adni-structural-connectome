#!/home/ec2-user/fsl/bin/python
"""Build the exact diagnosis-blind recovery plan for corrupt Eddy inputs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


AUDIT_ROOT = Path(
    "/data/derivatives/hcp379_v2/audits/eddy_volume_integrity_v1"
)
ROWS = AUDIT_ROOT / "rows.csv"
SUMMARY = AUDIT_ROOT / "summary.json"
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_eddy_integrity_recovery_plan_v1/plan.json"
)
RECOVERY_ROOT = Path(
    "/data/derivatives/hcp379_v2/eddy_integrity_recovery_v1"
)
EXPECTED_N = 515
MINIMUM_RETAINED_B0 = 3
MINIMUM_RETAINED_DIFFUSION = 30


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {str(k): str(v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


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


def integers(value: str) -> list[int]:
    return [int(item) for item in value.split(";") if item]


def build() -> dict[str, Any]:
    summary = load_json(SUMMARY)
    rows = read_csv(ROWS)
    if (
        summary.get("record_type")
        != "diagnosis_blind_hcp379_eddy_volume_integrity_audit"
        or summary.get("status") not in {"PASS", "RECOVERY_REQUIRED"}
        or summary.get("target_n") != EXPECTED_N
        or summary.get("audited_n") != EXPECTED_N
        or summary.get("diagnosis_labels_used") is not False
        or summary.get("outcomes_used") is not False
        or summary.get("connectome_density_used") is not False
        or len(rows) != EXPECTED_N
        or len({row["unit"] for row in rows}) != EXPECTED_N
    ):
        raise ValueError("complete 515-input Eddy integrity audit required")
    flagged = [
        row for row in rows if row["status"] == "FAIL_ZERO_EDDY_VOLUMES"
    ]
    if len(flagged) != summary.get("flagged_n"):
        raise ValueError("flagged Eddy identity count differs")
    plans: list[dict[str, Any]] = []
    for row in flagged:
        unit = row["unit"]
        zero_eddy = integers(row["eddy_zero_volume_indices"])
        zero_raw = integers(row["raw_zero_volume_indices"])
        bval_path = (
            Path("/data/derivatives/mif_dwi") / f"{unit}.bval"
        )
        bvalues = [float(value) for value in bval_path.read_text().split()]
        if (
            len(bvalues) != int(row["raw_volume_n"])
            or len(zero_raw) != int(row["raw_zero_volume_n"])
            or len(zero_eddy) != int(row["eddy_zero_volume_n"])
        ):
            raise ValueError(f"{unit}: zero-volume binding differs")
        if not zero_raw:
            route = "REGENERATE_EDDY_FROM_INTACT_RAW"
            retained = list(range(len(bvalues)))
            route_ready = True
        else:
            retained = [
                index
                for index in range(len(bvalues))
                if index not in set(zero_raw)
            ]
            retained_b0 = sum(bvalues[index] < 50 for index in retained)
            retained_dwi = sum(
                bvalues[index] >= 50 for index in retained
            )
            route_ready = bool(
                all(bvalues[index] < 50 for index in zero_raw)
                and retained_b0 >= MINIMUM_RETAINED_B0
                and retained_dwi >= MINIMUM_RETAINED_DIFFUSION
            )
            route = (
                "DROP_EXACT_ZERO_B0_THEN_REGENERATE_EDDY"
                if route_ready
                else "UNRESOLVED_SOURCE_ZERO_VOLUMES"
            )
        plans.append(
            {
                "unit": unit,
                "lane": row["lane"],
                "route": route,
                "route_ready": route_ready,
                "historical_eddy": record(Path(row["eddy_path"])),
                "raw_source": record(Path(row["raw_path"])),
                "bval_source": record(bval_path),
                "eddy_zero_volume_indices": zero_eddy,
                "raw_zero_volume_indices": zero_raw,
                "retained_volume_indices": retained,
                "retained_b0_n": sum(
                    bvalues[index] < 50 for index in retained
                ),
                "retained_diffusion_n": sum(
                    bvalues[index] >= 50 for index in retained
                ),
                "output_root": str(
                    (RECOVERY_ROOT / "subjects" / unit).resolve()
                ),
            }
        )
    unresolved = [
        row["unit"] for row in plans if not row["route_ready"]
    ]
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_eddy_integrity_recovery_plan",
        "status": "PASS" if not unresolved else "UNRESOLVED",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "historical_eddy_overwrite_allowed": False,
        "per_subject_parameter_tuning_allowed": False,
        "qc_threshold_relaxation_allowed": False,
        "target_n": len(plans),
        "unresolved_n": len(unresolved),
        "unresolved_units": unresolved,
        "minimum_retained_b0": MINIMUM_RETAINED_B0,
        "minimum_retained_diffusion": MINIMUM_RETAINED_DIFFUSION,
        "execution_backend": (
            "GPU_EDDY_IF_VALIDATED_GPU_HOST_ELSE_EXPLICIT_CPU"
        ),
        "imaging_executed": False,
        "targets": plans,
        "records": {
            "audit_rows": record(ROWS),
            "audit_summary": record(SUMMARY),
            "builder": record(Path(__file__)),
        },
    }
    atomic_json(OUTPUT, payload)
    return payload


def main() -> int:
    value = build()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
