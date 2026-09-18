#!/usr/bin/env python3
"""Build a corrected categorical-edge Recovery4 visual-review pack.

The Recovery4 v2 pack passed the integer HCP379 node image directly to FSL
``slices``.  Because node IDs 1..180 and 181..360 encode opposing cortical
hemispheres, treating those IDs as scalar overlay intensities can make one
hemisphere appear incompletely rendered.  This builder first converts every
label transition to a binary edge image, giving all 379 parcels identical
display weight.  It does not infer human visual QC.

The prior review pack and all imaging derivatives remain untouched.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
V2_SOURCE = (
    EXP / "scripts/hcp/build_hcp379_visual_review_pack_recovery4_v2.py"
)
OUTPUT_ROOT = HCP_ROOT / "review_recovery4_corrected_overlay"
FSL = Path("/home/ec2-user/fsl")


def load_v2_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "build_hcp379_visual_review_pack_recovery4_v2",
        V2_SOURCE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(V2_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = load_v2_module()


def run_atlas_edges(nodes: Path, output: Path) -> None:
    """Convert a categorical atlas to a uniform binary label-edge image."""
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "FSLDIR": str(FSL),
            "FSLOUTPUTTYPE": "NIFTI_GZ",
            "PATH": f"{FSL / 'bin'}:{env.get('PATH', '')}",
        }
    )
    completed = subprocess.run(
        [
            str(FSL / "bin/fslmaths"),
            str(nodes),
            "-edge",
            "-bin",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if completed.returncode or not output.is_file():
        raise RuntimeError(
            f"fslmaths edge conversion failed rc={completed.returncode}: "
            f"{completed.stdout}\n{completed.stderr}"
        )


def build_atlas_panel(
    *,
    b0: Path,
    nodes: Path,
    atlas_edges: Path,
    atlas_overlay: Path,
) -> tuple[Path, Path]:
    run_atlas_edges(nodes, atlas_edges)
    V2.run_slices(b0, atlas_edges, atlas_overlay)
    return atlas_edges, atlas_overlay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    manifest_path = (
        output_root
        / "hcp379_visual_review_pack_recovery4_corrected_overlay_manifest.json"
    )
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"corrected review pack exists; use --overwrite: {manifest_path}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    unit_dir = output_root / "units"
    overlay_dir = output_root / "source_overlays"
    edge_dir = output_root / "categorical_edges"
    unit_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(parents=True, exist_ok=True)

    bound_units = V2.load_units()
    legacy_manifest = V2.load_json(
        V2.OUTPUT_ROOT
        / "hcp379_visual_review_pack_recovery4_manifest.json"
    )
    legacy_units = {
        str(row["unit"]): row for row in legacy_manifest["units"]
    }
    if set(legacy_units) != {
        unit for unit, _, _ in bound_units
    }:
        raise ValueError("superseded review-pack unit set differs")

    atlas_jobs: dict[str, tuple[Path, Path, Path, Path]] = {}
    for unit, _, record in bound_units:
        b0 = V2.verify_file_record(
            record["tractography_inputs"]["mean_b0"]
        )
        nodes = V2.verify_file_record(
            record["hcp379"]["nodes_b0_native_qc"]
        )
        atlas_edges = edge_dir / f"{unit}_hcp379_label_edges.nii.gz"
        atlas_overlay = (
            overlay_dir / f"{unit}_b0_vs_hcp379_categorical_edges.png"
        )
        atlas_jobs[unit] = (b0, nodes, atlas_edges, atlas_overlay)

    atlas_outputs: dict[str, tuple[Path, Path]] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                build_atlas_panel,
                b0=b0,
                nodes=nodes,
                atlas_edges=atlas_edges,
                atlas_overlay=atlas_overlay,
            ): unit
            for unit, (
                b0,
                nodes,
                atlas_edges,
                atlas_overlay,
            ) in atlas_jobs.items()
        }
        for future in as_completed(futures):
            unit = futures[future]
            atlas_outputs[unit] = future.result()

    pages = []
    unit_records: list[dict[str, Any]] = []
    for index, (unit, unit_manifest_path, record) in enumerate(
        bound_units, start=1
    ):
        dwi = record["tractography_inputs"]
        b0 = V2.verify_file_record(dwi["mean_b0"])
        t1_overlay = V2.verify_file_record(
            legacy_units[unit]["source_panels"]["B0 vs selected T1"]
        )
        five_tt_overlay = V2.verify_file_record(
            legacy_units[unit]["source_panels"][
                "B0 vs selected 5TT support"
            ]
        )
        nodes = V2.verify_file_record(
            record["hcp379"]["nodes_b0_native_qc"]
        )
        legacy_scalar_overlay = V2.verify_file_record(
            record["hcp379"]["visual_overlay"]
        )
        atlas_edges, atlas_overlay = atlas_outputs[unit]

        sources = [
            ("B0 vs selected T1", t1_overlay),
            ("B0 vs selected 5TT support", five_tt_overlay),
            (
                "B0 vs HCP379: uniform parcel edges",
                atlas_overlay,
            ),
        ]
        route = str(record["registration"]["route"])
        dice = float(
            record["registration"]["selected_5tt_dwi_mask_dice"]
        )
        atlas_inside = float(
            record["registration"][
                "hcp379_atlas_inside_dwi_mask_fraction"
            ]
        )
        page = V2.render_unit(
            unit,
            sources,
            route=route,
            dice=dice,
            atlas_inside=atlas_inside,
        )
        page_path = unit_dir / f"{index:02d}_{unit}.png"
        page.save(page_path, format="PNG", optimize=True)
        pages.append(page)
        unit_records.append(
            {
                "unit": unit,
                "diagnosis_labels_used": False,
                "human_visual_qc_inferred": False,
                "registration_route": route,
                "selected_5tt_dwi_mask_dice": dice,
                "hcp379_atlas_inside_dwi_mask_fraction": atlas_inside,
                "unit_pretract_manifest": V2.file_record(
                    unit_manifest_path
                ),
                "hcp379_nodes": V2.file_record(nodes),
                "categorical_label_edges": V2.file_record(atlas_edges),
                "legacy_scalar_overlay_superseded": V2.file_record(
                    legacy_scalar_overlay
                ),
                "source_panels": {
                    label: V2.file_record(path)
                    for label, path in sources
                },
                "review_page": V2.file_record(page_path),
            }
        )

    pdf_path = (
        output_root
        / "hcp379_recovery4_corrected_overlay_blinded_visual_review.pdf"
    )
    pages[0].save(
        pdf_path,
        format="PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=150.0,
    )
    for page in pages:
        page.close()

    template_path = (
        output_root
        / "human_visual_qc_recovery4_corrected_overlay_template.csv"
    )
    V2.write_review_template(
        template_path,
        [unit for unit, _, _ in bound_units],
    )
    manifest = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_visual_review_pack_"
            "recovery4_corrected_overlay"
        ),
        "status": "READY_FOR_HUMAN_REVIEW",
        "generated_utc": V2.utc_now(),
        "diagnosis_labels_used": False,
        "human_visual_qc_inferred": False,
        "unit_count": len(bound_units),
        "pretract_manifest": V2.file_record(V2.PRETRACT_MANIFEST),
        "supersedes_review_pack": V2.file_record(
            V2.OUTPUT_ROOT
            / "hcp379_visual_review_pack_recovery4_manifest.json"
        ),
        "visualization_correction": {
            "problem": (
                "categorical node IDs 1..379 were displayed as scalar "
                "overlay intensities"
            ),
            "correction": (
                "all label transitions are rendered through a binary edge "
                "image with identical display weight"
            ),
            "imaging_derivatives_changed": False,
            "human_hcp379_column_requires_re_review": True,
        },
        "units": unit_records,
        "review_pdf": V2.file_record(pdf_path),
        "blank_human_qc_template": V2.file_record(template_path),
        "instructions": (
            "A human reviewer must inspect every corrected HCP379 panel. "
            "Prior T1 and 5TT decisions remain separate evidence. This pack "
            "does not assert visual PASS."
        ),
        "builder": V2.file_record(Path(__file__)),
    }
    V2.atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "unit_count": len(bound_units),
                "review_pdf": str(pdf_path),
                "template": str(template_path),
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
