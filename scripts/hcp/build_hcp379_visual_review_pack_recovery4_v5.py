#!/usr/bin/env python3
"""Build an exact-input-bound Recovery4 candidate visual-review pack.

Fourteen pages are unchanged from the corrected Recovery4 edge-overlay pack.
The 009 page is regenerated directly from the provisional candidate pretract
manifest: selected T1, finalized five-volume 5TT support, and finalized HCP379
nodes.  The resulting PDF is therefore exactly bound to the route that would
be promoted after genuine human acceptance.  This program performs no
promotion and never infers human QC.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


EXP = Path("/home/ec2-user/exp")
ROOT = Path("/data/derivatives/hcp379_v2")
TARGET = "009_S_4324_I1186579"
ROUTE = "bbr_initialized_syn_conservative"
V2_SOURCE = (
    EXP / "scripts/hcp/build_hcp379_visual_review_pack_recovery4_v2.py"
)
V3_SOURCE = (
    EXP / "scripts/hcp/build_hcp379_visual_review_pack_recovery4_v3.py"
)
V4_ROOT = ROOT / "review_recovery4_candidate_overlay_v4"
V4_MANIFEST = (
    V4_ROOT
    / "hcp379_visual_review_pack_recovery4_candidate_v4_manifest.json"
)
CANDIDATE_MASTER = (
    ROOT
    / "pretract_candidate_recovery4_v4/"
    "pretract_candidate_recovery4_v4_manifest.json"
)
OUTPUT_ROOT = ROOT / "review_recovery4_candidate_overlay_v5"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = load_module(V2_SOURCE, "hcp379_review_v2")
V3 = load_module(V3_SOURCE, "hcp379_review_v3")


def copy_verified_image(
    record: dict[str, Any],
    destination: Path,
) -> None:
    source = V2.verify_file_record(record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    with Image.open(destination) as image:
        image.verify()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    manifest_path = (
        output_root
        / "hcp379_visual_review_pack_recovery4_candidate_v5_manifest.json"
    )
    if output_root.exists():
        raise FileExistsError(
            f"non-overwriting review root exists: {output_root}"
        )

    v4 = V2.load_json(V4_MANIFEST)
    candidate_master = V2.load_json(CANDIDATE_MASTER)
    if (
        v4.get("status") != "READY_FOR_HUMAN_CANDIDATE_REVIEW"
        or v4.get("unit_count") != 15
        or v4.get("diagnosis_labels_used") is not False
        or v4.get("human_visual_qc_inferred") is not False
    ):
        raise ValueError("V4 review pack differs")
    if (
        candidate_master.get("record_type")
        != (
            "diagnosis_blind_hcp379_pretract_candidate_recovery4_v4_"
            "manifest"
        )
        or candidate_master.get("status")
        != "PROVISIONAL_AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or candidate_master.get("unit_count") != 15
        or candidate_master.get("diagnosis_labels_used") is not False
        or candidate_master.get("human_visual_qc_inferred") is not False
        or candidate_master.get("tractography_started") is not False
        or candidate_master.get("matrix_generation_started") is not False
        or candidate_master.get("route_substitution", {}).get(
            "candidate_route"
        )
        != ROUTE
        or candidate_master.get("route_substitution", {}).get(
            "promotion_status"
        )
        != "NOT_PROMOTED"
    ):
        raise ValueError("candidate pretract master differs")

    v4_by_unit = {str(row["unit"]): row for row in v4["units"]}
    candidate_by_unit = {
        str(row["unit"]): row for row in candidate_master["units"]
    }
    if (
        len(v4_by_unit) != 15
        or set(v4_by_unit) != set(candidate_by_unit)
        or TARGET not in candidate_by_unit
    ):
        raise ValueError("candidate review unit set differs")
    target_manifest_path = V2.verify_file_record(
        candidate_by_unit[TARGET]["manifest"]
    )
    target_manifest = V2.load_json(target_manifest_path)
    if (
        target_manifest.get("record_type")
        != (
            "diagnosis_blind_hcp379_pretract_candidate_recovery4_v4_"
            "unit_manifest"
        )
        or target_manifest.get("status")
        != "PROVISIONAL_AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or target_manifest.get("registration", {}).get("route") != ROUTE
    ):
        raise ValueError("009 candidate unit manifest differs")

    page_dir = output_root / "units"
    overlay_dir = output_root / "source_overlays"
    edge_dir = output_root / "categorical_edges"
    page_dir.mkdir(parents=True)
    overlay_dir.mkdir(parents=True)
    edge_dir.mkdir(parents=True)

    b0 = V2.verify_file_record(
        target_manifest["tractography_inputs"]["mean_b0"]
    )
    selected_t1 = V2.verify_file_record(
        target_manifest["registration"]["selected_t1_in_b0"]
    )
    five_tt_mask = V2.verify_file_record(
        target_manifest["automated_qc"]["candidate_pretract"]
    )
    candidate_qc = V2.load_json(five_tt_mask)
    five_tt_mask = V2.verify_file_record(
        candidate_qc["artifacts"]["five_tt_mask"]
    )
    nodes_native = V2.verify_file_record(
        target_manifest["hcp379"]["nodes_b0_native_qc"]
    )
    t1_overlay = overlay_dir / f"{TARGET}_b0_vs_selected_t1.png"
    five_tt_overlay = (
        overlay_dir / f"{TARGET}_b0_vs_finalized_5tt_support.png"
    )
    hcp_edges = edge_dir / f"{TARGET}_hcp379_label_edges.nii.gz"
    hcp_overlay = (
        overlay_dir / f"{TARGET}_b0_vs_finalized_hcp379_edges.png"
    )
    V2.run_slices(b0, selected_t1, t1_overlay)
    V2.run_slices(b0, five_tt_mask, five_tt_overlay)
    V3.run_atlas_edges(nodes_native, hcp_edges)
    V2.run_slices(b0, hcp_edges, hcp_overlay)
    target_sources = [
        ("B0 vs candidate T1", t1_overlay),
        ("B0 vs finalized candidate 5TT support", five_tt_overlay),
        (
            "B0 vs finalized candidate HCP379: uniform parcel edges",
            hcp_overlay,
        ),
    ]
    target_page = V2.render_unit(
        TARGET,
        target_sources,
        route=f"PROVISIONAL {ROUTE}",
        dice=float(
            target_manifest["registration"]["selected_5tt_dwi_mask_dice"]
        ),
        atlas_inside=float(
            target_manifest["registration"][
                "hcp379_atlas_inside_dwi_mask_fraction"
            ]
        ),
    )

    pages: list[Image.Image] = []
    unit_records: list[dict[str, Any]] = []
    for index, unit in enumerate(sorted(v4_by_unit), start=1):
        page_path = page_dir / f"{index:02d}_{unit}.png"
        if unit == TARGET:
            target_page.save(page_path, format="PNG", optimize=True)
            page = target_page.copy()
            source_panels = {
                label: V2.file_record(path)
                for label, path in target_sources
            }
            pretract_manifest = V2.file_record(target_manifest_path)
            route = ROUTE
            exact_candidate_inputs = {
                "selected_t1": target_manifest["registration"][
                    "selected_t1_in_b0"
                ],
                "five_tt": target_manifest["tractography_inputs"][
                    "five_tt"
                ],
                "five_tt_mask": candidate_qc["artifacts"][
                    "five_tt_mask"
                ],
                "hcp379_nodes_native": target_manifest["hcp379"][
                    "nodes_b0_native_qc"
                ],
            }
        else:
            copy_verified_image(
                v4_by_unit[unit]["review_page"], page_path
            )
            page = Image.open(page_path).convert("RGB")
            source_panels = v4_by_unit[unit]["source_panels"]
            pretract_manifest = candidate_by_unit[unit]["manifest"]
            route = candidate_by_unit[unit]["registration_route"]
            exact_candidate_inputs = None
        pages.append(page)
        unit_records.append(
            {
                "unit": unit,
                "diagnosis_labels_used": False,
                "human_visual_qc_inferred": False,
                "registration_route": route,
                "pretract_manifest": pretract_manifest,
                "exact_candidate_inputs": exact_candidate_inputs,
                "source_panels": source_panels,
                "review_page": V2.file_record(page_path),
            }
        )
    target_page.close()

    pdf_path = (
        output_root
        / "hcp379_recovery4_v5_exact_candidate_blinded_review.pdf"
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
        output_root / "human_visual_qc_recovery4_candidate_v5_template.csv"
    )
    with template_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "unit",
                "B0 vsT1",
                "B0 vs 5TT",
                "B0 vs HCP379",
                "reviewer",
                "reviewed_utc",
                "Notes",
            ],
        )
        writer.writeheader()
        for unit in sorted(v4_by_unit):
            target = unit == TARGET
            writer.writerow(
                {
                    "unit": unit,
                    "B0 vsT1": "" if target else "PASS",
                    "B0 vs 5TT": "" if target else "PASS",
                    "B0 vs HCP379": "",
                    "reviewer": "",
                    "reviewed_utc": "",
                    "Notes": (
                        "Review all three exact candidate panels."
                        if target else
                        "T1 and 5TT PASS carried from submitted round 1; "
                        "review corrected HCP379 edge panel."
                    ),
                }
            )

    manifest = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_visual_review_pack_"
            "recovery4_candidate_v5"
        ),
        "status": "READY_FOR_EXACT_INPUT_HUMAN_REVIEW",
        "generated_utc": V2.utc_now(),
        "diagnosis_labels_used": False,
        "human_visual_qc_inferred": False,
        "canonical_human_gate_changed": False,
        "tractography_started": False,
        "matrix_generation_started": False,
        "unit_count": 15,
        "candidate_pretract_master": V2.file_record(CANDIDATE_MASTER),
        "supersedes_review_pack": V2.file_record(V4_MANIFEST),
        "route_substitution": {
            "unit": TARGET,
            "candidate_route": ROUTE,
            "promotion_status": "NOT_PROMOTED",
            "review_panels_bound_to_exact_candidate_inputs": True,
        },
        "carried_human_evidence": {
            "unchanged_units": 14,
            "B0_vs_T1": "PASS",
            "B0_vs_5TT": "PASS",
            "HCP379_status_carried": False,
        },
        "units": unit_records,
        "review_pdf": V2.file_record(pdf_path),
        "human_qc_template": V2.file_record(template_path),
        "instructions": (
            "Review the corrected HCP379 panel for all 15 units. For 009, "
            "review all three panels because they are regenerated from the "
            "exact provisional candidate inputs. No route is promoted."
        ),
        "builder": V2.file_record(Path(__file__)),
    }
    V2.atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "unit_count": 15,
                "review_pdf": str(pdf_path),
                "template": str(template_path),
                "manifest": str(manifest_path),
                "candidate_pretract_master": str(CANDIDATE_MASTER),
                "promotion_status": "NOT_PROMOTED",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
