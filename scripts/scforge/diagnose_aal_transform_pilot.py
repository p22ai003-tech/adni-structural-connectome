#!/usr/bin/env python3
"""Compare AAL-to-B0 transform routes for SC matrix QC repair pilots.

This is read/write only inside the QC folder. It does not alter production
parcellation, track, or connectome outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path("/home/ec2-user/exp")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from connectome_pipeline.connectome_step7 import Step7Config, _base_env, _subject_paths  # noqa: E402
from connectome_pipeline.pipeline_paths import resolve_pipeline_paths  # noqa: E402


def run(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
        log.write(f"[exit {proc.returncode}]\n\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed; see {log_path}: {' '.join(cmd)}")


def read_expected_labels(label_csv: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with label_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                value = int(float(row["atlas_value"]))
            except Exception:
                continue
            labels[value] = row.get("atlas_label") or row.get("node_name") or str(value)
    return labels


def load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.asanyarray(nib.load(str(path)).dataobj)
    return data > 0


def image_metrics(path: Path, expected: dict[int, str], mask: np.ndarray | None) -> dict[str, object]:
    img = nib.load(str(path))
    data = np.rint(np.asanyarray(img.dataobj)).astype(np.int32)
    nonzero = data[data > 0]
    uniq, counts = np.unique(nonzero, return_counts=True) if nonzero.size else (np.array([], dtype=int), np.array([], dtype=int))
    valid = [int(v) for v in uniq if int(v) in expected]
    valid_counts = {int(v): int(c) for v, c in zip(uniq, counts) if int(v) in expected}
    robust = [v for v, c in valid_counts.items() if c >= 50]
    frontal = {k: valid_counts.get(k, 0) for k in (7, 8)}
    overlap_fraction = ""
    if mask is not None and mask.shape == data.shape and nonzero.size:
        overlap_fraction = float(((data > 0) & mask).sum() / max((data > 0).sum(), 1))
    return {
        "image": str(path),
        "shape": "x".join(str(x) for x in data.shape),
        "labeled_voxels": int((data > 0).sum()),
        "unique_nonzero_labels": int(len(uniq)),
        "valid_labels_present": int(len(valid)),
        "valid_labels_ge50vox": int(len(robust)),
        "missing_valid_labels": ";".join(str(v) for v in sorted(set(expected) - set(valid))),
        "aal_007_voxels": frontal[7],
        "aal_008_voxels": frontal[8],
        "mask_overlap_fraction": overlap_fraction,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--qc-root", type=Path, default=resolve_pipeline_paths(create_layout=True).deriv_root / "qc" / "sc_matrix_qc")
    args = parser.parse_args()

    cfg = Step7Config()
    paths = _subject_paths(cfg, args.sid, create_dirs=False)
    label_csv = cfg.aal_mni.parent / "AAL3_labels.csv"
    expected = read_expected_labels(label_csv)
    missing_inputs = [
        name
        for name in ("B0MEAN", "T1_RAS", "T1B", "T12B0")
        if not paths[name].exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(f"missing inputs for {args.sid}: {missing_inputs}")
    if not cfg.aal_mni.exists():
        raise FileNotFoundError(cfg.aal_mni)

    stamp = args.tag or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_root = args.qc_root / "aal_transform_pilot" / args.sid / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "commands.log"
    env = _base_env(cfg)
    flirt = shutil.which("flirt", path=env["PATH"])
    convert_xfm = shutil.which("convert_xfm", path=env["PATH"])
    if not flirt or not convert_xfm:
        raise RuntimeError("FSL flirt/convert_xfm not found")

    fsl_mni = cfg.fsl_dir / "data/standard/MNI152_T1_1mm_brain.nii.gz"
    mni2t1 = out_root / "mni2t1_from_mni_brain.mat"
    mni2b0 = out_root / "mni2b0_composed.mat"
    direct_b0 = out_root / "AAL_b0_direct_composed.nii.gz"
    twostep_t1 = out_root / "AAL_t1_twostep_composed.nii.gz"
    twostep_b0 = out_root / "AAL_b0_twostep_composed.nii.gz"
    bible_t1 = out_root / "AAL_t1_bible_label_est.nii.gz"
    bible_mat = out_root / "aal_mni2t1_bible_label_est.mat"
    bible_b0 = out_root / "AAL_b0_bible_label_est_twostep.nii.gz"

    run([flirt, "-in", str(fsl_mni), "-ref", str(paths["T1_RAS"]), "-omat", str(mni2t1), "-dof", "12"], log_path, env)
    run([convert_xfm, "-omat", str(mni2b0), "-concat", str(paths["T12B0"]), str(mni2t1)], log_path, env)
    run(
        [
            flirt,
            "-in",
            str(cfg.aal_mni),
            "-ref",
            str(paths["B0MEAN"]),
            "-applyxfm",
            "-init",
            str(mni2b0),
            "-interp",
            "nearestneighbour",
            "-datatype",
            "int",
            "-out",
            str(direct_b0),
        ],
        log_path,
        env,
    )
    run(
        [
            flirt,
            "-in",
            str(cfg.aal_mni),
            "-ref",
            str(paths["T1_RAS"]),
            "-applyxfm",
            "-init",
            str(mni2t1),
            "-interp",
            "nearestneighbour",
            "-datatype",
            "int",
            "-out",
            str(twostep_t1),
        ],
        log_path,
        env,
    )
    run(
        [
            flirt,
            "-in",
            str(twostep_t1),
            "-ref",
            str(paths["B0MEAN"]),
            "-applyxfm",
            "-init",
            str(paths["T12B0"]),
            "-interp",
            "nearestneighbour",
            "-datatype",
            "int",
            "-out",
            str(twostep_b0),
        ],
        log_path,
        env,
    )
    run(
        [
            flirt,
            "-in",
            str(cfg.aal_mni),
            "-ref",
            str(paths["T1B"]),
            "-omat",
            str(bible_mat),
            "-interp",
            "nearestneighbour",
            "-datatype",
            "int",
            "-out",
            str(bible_t1),
            "-dof",
            "12",
        ],
        log_path,
        env,
    )
    run(
        [
            flirt,
            "-in",
            str(bible_t1),
            "-ref",
            str(paths["B0MEAN"]),
            "-applyxfm",
            "-init",
            str(paths["T12B0"]),
            "-interp",
            "nearestneighbour",
            "-datatype",
            "int",
            "-out",
            str(bible_b0),
        ],
        log_path,
        env,
    )

    mask = load_mask(paths["MASK_B0"])
    rows: list[dict[str, object]] = []
    candidates = [
        ("production_current", paths["AAL_B0"]),
        ("direct_composed_current_logic", direct_b0),
        ("twostep_composed_nn", twostep_b0),
        ("bible_label_est_twostep", bible_b0),
    ]
    for route, image in candidates:
        if not image.exists():
            continue
        row = {"sid": args.sid, "route": route}
        row.update(image_metrics(image, expected, mask))
        rows.append(row)

    out_csv = out_root / "aal_transform_route_comparison.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_root / "manifest.json").write_text(
        json.dumps(
            {
                "sid": args.sid,
                "output_root": str(out_root),
                "comparison_csv": str(out_csv),
                "log": str(log_path),
                "production_aal_b0": str(paths["AAL_B0"]),
                "aal_mni": str(cfg.aal_mni),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"comparison_csv={out_csv}")
    for row in rows:
        print(
            f"{row['route']}: valid={row['valid_labels_present']} "
            f"ge50={row['valid_labels_ge50vox']} vox={row['labeled_voxels']} "
            f"AAL007={row['aal_007_voxels']} AAL008={row['aal_008_voxels']} "
            f"mask_overlap={row['mask_overlap_fraction']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
