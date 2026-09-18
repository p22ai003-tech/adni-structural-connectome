#!/usr/bin/env python3
"""Freeze the exact Recovery4 pre-tractography inputs for Phase B.

The manifest binds the 15-unit diagnosis-blind canary to the selected
registration route, 5TT/GMWMI, normalised tissue FODs, bounded tensor maps,
HCP379 nodes, node volumes, and all three automated Recovery4 QC records.
It does not infer or record human visual-QC approval and it does not start
tractography.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
ATLAS_ROOT = HCP_ROOT / "corrected_atlas_canary_recovery4"
SCALAR_ROOT = HCP_ROOT / "pretract_scalar_recovery4"
SPATIAL_ROOT = HCP_ROOT / "pretract_spatial_recovery4"
OUTPUT_ROOT = HCP_ROOT / "pretract_recovery4"
ATLAS_SUMMARY = (
    ATLAS_ROOT / "corrected_atlas_canary_recovery4_summary.json"
)
SCALAR_SUMMARY = SCALAR_ROOT / "scalar_recovery4_summary.json"
SPATIAL_SUMMARY = SPATIAL_ROOT / "spatial_recovery4_summary.json"


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def verify_file_record(record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or int(record.get("size_bytes", -1)) != path.stat().st_size
        or str(record.get("sha256", "")) != sha256_file(path)
    ):
        raise ValueError(f"artifact binding differs: {path}")
    return path


def verify_qc(
    path: Path,
    *,
    unit: str,
    record_type: str,
) -> dict[str, Any]:
    record = load_json(path)
    if (
        record.get("record_type") != record_type
        or record.get("status") != "PASS"
        or record.get("diagnosis_labels_used") is not False
        or record.get("unit") != unit
    ):
        raise ValueError(f"QC contract differs: {path}")
    for artifact in record.get("artifacts", {}).values():
        verify_file_record(artifact)
    return record


def immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(
                f"immutable manifest exists with different content: {path}"
            )
        return
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def summary_units(path: Path, count_key: str) -> tuple[dict[str, Any], set[str]]:
    summary = load_json(path)
    rows = summary.get("units")
    if (
        summary.get("status") != "PASS"
        or summary.get("diagnosis_labels_used") is not False
        or not isinstance(rows, list)
        or int(summary.get(count_key, -1)) != 15
    ):
        raise ValueError(f"summary is not an exact automated PASS: {path}")
    units = {str(row.get("unit")) for row in rows}
    if len(units) != 15 or any(
        row.get("status") != "PASS" for row in rows
    ):
        raise ValueError(f"summary unit set differs: {path}")
    return summary, units


def source_file(subject: Path, relative: str) -> dict[str, Any]:
    return file_record(subject / relative)


def build_unit(unit: str) -> dict[str, Any]:
    subject = RUN_ROOT / "subjects" / unit
    atlas_path = ATLAS_ROOT / "subjects" / unit / "result.json"
    scalar_path = (
        SCALAR_ROOT
        / "subjects"
        / unit
        / "06_preflight/scalar_recovery4_qc.json"
    )
    spatial_path = (
        SPATIAL_ROOT
        / "subjects"
        / unit
        / "06_preflight/spatial_recovery4_qc.json"
    )
    atlas = verify_qc(
        atlas_path,
        unit=unit,
        record_type=(
            "diagnosis_blind_hcp379_corrected_atlas_recovery4_build"
        ),
    )
    scalar = verify_qc(
        scalar_path,
        unit=unit,
        record_type=(
            "diagnosis_blind_hcp379_pretract_scalar_recovery4_qc"
        ),
    )
    spatial = verify_qc(
        spatial_path,
        unit=unit,
        record_type=(
            "diagnosis_blind_hcp379_pretract_spatial_recovery4_qc"
        ),
    )
    if (
        atlas.get("atlas_qc", {}).get("status") != "PASS"
        or atlas.get("atlas_qc", {}).get("labels_found") != 379
        or atlas.get("atlas_qc", {}).get("missing_labels") != []
        or spatial.get("registration_route")
        != atlas.get("registration_route", {}).get("route")
    ):
        raise ValueError(f"atlas/spatial contract differs: {unit}")

    atlas_artifacts = atlas["artifacts"]
    scalar_artifacts = scalar["artifacts"]
    spatial_artifacts = spatial["artifacts"]
    record = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_pretract_recovery4_unit_manifest"
        ),
        "status": "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC",
        "unit": unit,
        "diagnosis_labels_used": False,
        "human_visual_qc_inferred": False,
        "registration": {
            "route": spatial["registration_route"],
            "transform": spatial_artifacts["selected_transform"],
            "selected_t1_in_b0": spatial_artifacts["selected_t1_in_b0"],
            "selected_5tt_dwi_mask_dice": spatial[
                "selected_5tt_dwi_mask_dice"
            ],
            "hcp379_atlas_inside_dwi_mask_fraction": spatial[
                "hcp379_atlas_inside_dwi_mask_fraction"
            ],
        },
        "tractography_inputs": {
            "dwi_bias_corrected": source_file(
                subject, "01_dwi/dwi_preproc_biascorr.mif"
            ),
            "dwi_mask": source_file(
                subject, "01_dwi/dwi_brain_mask.mif"
            ),
            "mean_b0": source_file(subject, "03_spatial/mean_b0.nii.gz"),
            "wmfod_normalised": source_file(
                subject, "05_model/wmfod_norm.mif"
            ),
            "gm_normalised": source_file(
                subject, "05_model/gm_norm.mif"
            ),
            "csf_normalised": source_file(
                subject, "05_model/csf_norm.mif"
            ),
            "five_tt": spatial_artifacts["selected_five_tt"],
            "gmwmi": spatial_artifacts["selected_gmwmi"],
        },
        "tensor_maps_bounded": {
            metric: scalar_artifacts[f"{metric}_bounded"]
            for metric in ("fa", "md", "rd", "ad")
        },
        "tensor_maps_raw_preserved": {
            metric: source_file(subject, f"05_model/{metric}.mif")
            for metric in ("fa", "md", "rd", "ad")
        },
        "hcp379": {
            "nodes_b0_1mm": atlas_artifacts[
                "hcp379_nodes_b0_1mm"
            ],
            "nodes_b0_native_qc": atlas_artifacts[
                "hcp379_nodes_b0_native_qc"
            ],
            "node_volumes": atlas_artifacts["hcp379_node_volumes"],
            "source_node_volumes": atlas_artifacts[
                "hcp379_node_volumes_t1_native"
            ],
            "atlas_qc": atlas_artifacts["hcp379_atlas_qc"],
            "visual_overlay": atlas_artifacts[
                "hcp379_visual_overlay"
            ],
            "source_parcellation": atlas_artifacts[
                "hcp_source_parcellation"
            ],
        },
        "provenance": {
            "gradient_contract": source_file(
                subject, "01_dwi/gradient_contract.json"
            ),
            "fod_shell_selection": source_file(
                subject, "05_model/fod_shell_selection.json"
            ),
            "tensor_shell_selection": source_file(
                subject, "05_model/tensor_shell_selection.json"
            ),
            "response_calibration_outcome": source_file(
                subject, "05_model/response_calibration_outcome.json"
            ),
            "wmfod_l0_qc": scalar_artifacts["wmfod_l0"],
        },
        "automated_qc": {
            "atlas": file_record(atlas_path),
            "scalar": file_record(scalar_path),
            "spatial": file_record(spatial_path),
        },
    }
    for category in (
        "tractography_inputs",
        "tensor_maps_bounded",
        "tensor_maps_raw_preserved",
        "hcp379",
        "provenance",
        "automated_qc",
    ):
        for artifact in record[category].values():
            verify_file_record(artifact)
    verify_file_record(record["registration"]["transform"])
    verify_file_record(record["registration"]["selected_t1_in_b0"])
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    _, atlas_units = summary_units(ATLAS_SUMMARY, "passed_unit_count")
    _, scalar_units = summary_units(SCALAR_SUMMARY, "passed_n")
    _, spatial_units = summary_units(SPATIAL_SUMMARY, "passed_n")
    if not atlas_units == scalar_units == spatial_units:
        raise ValueError("Recovery4 automated-QC unit sets differ")

    completed = utc_now()
    unit_refs: list[dict[str, Any]] = []
    routes: dict[str, int] = {}
    for unit in sorted(atlas_units):
        record = build_unit(unit)
        route = str(record["registration"]["route"])
        routes[route] = routes.get(route, 0) + 1
        path = (
            args.output_root
            / "subjects"
            / unit
            / "pretract_recovery4_manifest.json"
        )
        record["completed_utc"] = completed
        immutable_json(path, record)
        unit_refs.append(
            {
                "unit": unit,
                "status": record["status"],
                "registration_route": route,
                "manifest": file_record(path),
            }
        )

    master = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_pretract_recovery4_manifest"
        ),
        "status": "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC",
        "completed_utc": completed,
        "diagnosis_labels_used": False,
        "human_visual_qc_inferred": False,
        "tractography_started": False,
        "matrix_generation_started": False,
        "non_overwriting": True,
        "unit_count": len(unit_refs),
        "registration_route_counts": dict(sorted(routes.items())),
        "automated_qc": {
            "atlas_summary": file_record(ATLAS_SUMMARY),
            "scalar_summary": file_record(SCALAR_SUMMARY),
            "spatial_summary": file_record(SPATIAL_SUMMARY),
        },
        "next_gate": (
            "genuine blinded human review of b0/T1, b0/selected-5TT, "
            "and b0/HCP379 overlays for all 15 units"
        ),
        "units": unit_refs,
        "builder": file_record(Path(__file__)),
    }
    immutable_json(
        args.output_root / "pretract_recovery4_manifest.json",
        master,
    )
    print(json.dumps(master, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
