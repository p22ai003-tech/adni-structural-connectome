#!/usr/bin/env python3
"""Route 16: hybrid dense tracks plus zero-ROI targeted AAL3 rescue.

Route15 showed that 12M whole-brain dynamic ACT/SIFT2 tracks improved density
only slightly and left many AAL3 zero rows. This lane keeps the Route15 dense
track cloud, adds a scratch target-seeded tractogram from currently disconnected
AAL3 labels, concatenates both tractograms, recomputes SIFT2, and rebuilds the
fd_sum connectome against the same high-label AAL3-to-B0 parcellation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live import run_aal3_route7_selective_rescue_batch as route7  # noqa: E402
from scripts.scforge.live.run_aal3_route11b_act_sift2_track_coverage_rerun import (  # noqa: E402
    DEFAULT_ROUTE1_ROOT,
    RUN_PARENT,
    collect_route11_parcellations,
    read_csv,
    refresh_outputs,
)
from scripts.scforge.live.run_sc_aal3_source_contract_probe import (  # noqa: E402
    QC_ROOT,
    aal3_valid_labels,
    discover_inputs,
    write_csv,
    write_json,
)


LANE = "R16"
SCHEMA = "route16_zero_roi_hybrid_v1"
SUFFIX = "_route16_zero_roi_hybrid"
PROBE_NAME = "aal3_route16_zero_roi_hybrid_probe"
ROUTE15_PROBE_NAME = "aal3_route15_step7_dense_track_coverage_probe"
TCKEDIT = Path("/home/ec2-user/mrtrix3/bin/tckedit")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_status(root: Path, **updates: Any) -> None:
    status = read_json(root / "status.json")
    status.update(updates)
    status["updated_utc"] = utc()
    write_json(root / "status.json", status)


def zero_row_labels(matrix_path: Path, valid_labels: list[int]) -> list[int]:
    mat = np.loadtxt(matrix_path, delimiter=",")
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"not a square matrix: {matrix_path}")
    valid = set(int(v) for v in valid_labels)
    row_signal = np.abs(mat).sum(axis=0) + np.abs(mat).sum(axis=1)
    return [idx + 1 for idx, value in enumerate(row_signal) if value == 0 and (idx + 1) in valid]


def write_zero_roi_seed_mask(
    *,
    parc_path: Path,
    matrix_path: Path,
    valid_labels: list[int],
    out_path: Path,
    dilation_iters: int,
) -> dict[str, Any]:
    labels = zero_row_labels(matrix_path, valid_labels)
    img = nib.load(str(parc_path))
    data = np.asanyarray(img.dataobj).astype(np.int32)
    seed = np.isin(data, labels)
    original_voxels = int(seed.sum())
    if dilation_iters > 0:
        try:
            from scipy import ndimage as ndi

            seed = ndi.binary_dilation(seed, iterations=dilation_iters)
        except Exception:
            # Keep a non-dilated target mask if scipy is unavailable.
            dilation_iters = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(seed.astype(np.uint8), img.affine, img.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, str(out_path))
    return {
        "seed_mask": str(out_path),
        "zero_label_count": len(labels),
        "zero_labels": ";".join(str(label) for label in labels),
        "zero_roi_voxels": original_voxels,
        "seed_voxels": int(seed.sum()),
        "dilation_iters": dilation_iters,
    }


def run_cmd(cmd: list[str | Path], log_path: Path, env: dict[str, str], timeout_sec: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    str_cmd = [str(item) for item in cmd]
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"[{utc()}] $ {' '.join(str_cmd)}\n")
        handle.write(f"[{utc()}] timeout_sec={timeout_sec}\n")
        handle.flush()
        try:
            proc = subprocess.run(
                str_cmd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            handle.write(f"[{utc()}] timeout after {timeout_sec}s\n\n")
            raise TimeoutError(f"command timed out after {timeout_sec}s; see {log_path}") from exc
        handle.write(f"[{utc()}] rc={proc.returncode}\n\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}; see {log_path}")


def split_streamlines(total: int, shards: int) -> list[int]:
    shards = max(1, int(shards))
    base = total // shards
    rem = total % shards
    return [base + (1 if idx < rem else 0) for idx in range(shards)]


def preserve_interrupted_track(path: Path) -> Path | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = path.with_name(f"{path.stem}.interrupted_{stamp}{path.suffix}")
    path.rename(backup)
    return backup


def route15_dense_track(route1_root: Path, sid: str, dense_streamlines: int) -> Path:
    suffix = f"{dense_streamlines // 1000}k"
    path = (
        QC_ROOT
        / ROUTE15_PROBE_NAME
        / f"{route1_root.name}_route15_step7_dense_track_coverage_{sid}"
        / "tracks"
        / f"{sid}_dynamic_{suffix}.tck"
    )
    if not path.exists() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing Route15 dense track file: {path}")
    return path


def run_subject(
    *,
    sid: str,
    route1_root: Path,
    route2_root: Path,
    run_root: Path,
    target_streamlines: int,
    dense_streamlines: int,
    dilation_iters: int,
    variants: list[str],
    mrtrix_threads: int,
    target_shards: int,
) -> dict[str, Any]:
    probe_root = QC_ROOT / PROBE_NAME / f"{run_root.name}_{sid}"
    probe_root.mkdir(parents=True, exist_ok=True)
    update_status(
        probe_root,
        sid=sid,
        tag=f"{run_root.name}_{sid}",
        phase="starting",
        route16_action="zero_roi_hybrid",
    )

    inputs = discover_inputs(sid)
    missing = [key for key in ("production_fd_sum",) if not inputs.get(key)]
    fpaths = route7.fod_paths(sid)
    missing += [key for key, path in fpaths.items() if key != "deconv_mode" and not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing route16 inputs for {sid}: {missing}")

    env = route7.env_for_mrtrix(mrtrix_threads)
    valid_labels = aal3_valid_labels()
    label_rows, matrix_rows, baseline_fd = route7.production_baseline(inputs, valid_labels)
    parc_candidates = collect_route11_parcellations(sid, route1_root, route2_root, valid_labels)
    parc_candidates = [cand for cand in parc_candidates if int(cand.get("labels") or 0) >= 160]
    if not parc_candidates:
        batch_row = route7.row_from_best(
            sid,
            "ok",
            probe_root,
            {},
            baseline_fd,
            label_rows,
            {"route16_reason": "no high-label parcellation for zero-ROI hybrid rescue"},
        )
        write_csv(probe_root / "label_survival_summary.csv", label_rows)
        write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)
        write_json(probe_root / "decision.json", {"decision": "AAL3_ROUTE16_NO_HIGH_LABEL_PARCELLATION", "best": {}, "batch_row": batch_row})
        update_status(probe_root, phase="complete", completed_utc=utc(), decision="AAL3_ROUTE16_NO_HIGH_LABEL_PARCELLATION")
        return batch_row

    parc_candidate = parc_candidates[0]
    base_stage = str(parc_candidate["candidate"])
    label_rows.append({"stage": base_stage, **parc_candidate["label_stats"]})
    write_csv(probe_root / "label_survival_summary.csv", label_rows)
    write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)

    track_root = probe_root / "tracks"
    logs = probe_root / "logs"
    seeds = probe_root / "seeds"
    track_root.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    seeds.mkdir(exist_ok=True)

    dense_tracks = route15_dense_track(route1_root, sid, dense_streamlines)
    target_suffix = f"{target_streamlines // 1000}k"
    dense_suffix = f"{dense_streamlines // 1000}k"
    seed_mask = seeds / f"{sid}_zero_roi_dil{dilation_iters}_seed.nii.gz"
    seed_info = write_zero_roi_seed_mask(
        parc_path=Path(parc_candidate["path"]),
        matrix_path=Path(inputs["production_fd_sum"]),
        valid_labels=valid_labels,
        out_path=seed_mask,
        dilation_iters=dilation_iters,
    )
    write_csv(probe_root / "zero_roi_seed_summary.csv", [seed_info])

    target_tracks = track_root / f"{sid}_zero_roi_dil{dilation_iters}_{target_suffix}.tck"
    combined_tracks = track_root / f"{sid}_hybrid_dynamic{dense_suffix}_zero_roi{target_suffix}_dil{dilation_iters}.tck"
    weights = track_root / f"{sid}_hybrid_dynamic{dense_suffix}_zero_roi{target_suffix}_dil{dilation_iters}_sift2_weights.txt"
    mu = track_root / f"{sid}_hybrid_dynamic{dense_suffix}_zero_roi{target_suffix}_dil{dilation_iters}_mu.txt"

    if target_shards > 1 and target_tracks.exists() and target_tracks.stat().st_size > 0 and not combined_tracks.exists():
        preserve_interrupted_track(target_tracks)

    if not target_tracks.exists() or target_tracks.stat().st_size <= 0:
        update_status(
            probe_root,
            phase="tckgen_zero_roi",
            zero_label_count=seed_info["zero_label_count"],
            select_streamlines=target_streamlines,
            target_shards=max(1, target_shards),
        )
        cmd: list[str | Path] = [
            route7.TCKGEN,
            fpaths["wmfod"],
            target_tracks,
            "-act",
            fpaths["act"],
            "-backtrack",
            "-cutoff",
            "0.04",
            "-maxlength",
            "250",
            "-minlength",
            "10",
            "-select",
            str(target_streamlines),
            "-seed_image",
            seed_mask,
            "-nthreads",
            str(mrtrix_threads),
            "-force",
        ]
        if target_shards <= 1:
            route7.run_track_cmd(cmd, logs / f"tckgen_zero_roi_dil{dilation_iters}_{target_suffix}.log", env, "tckgen")
        else:
            shard_counts = split_streamlines(target_streamlines, target_shards)
            shard_threads = max(1, mrtrix_threads // len(shard_counts))
            shard_tracks = [
                track_root / f"{sid}_zero_roi_dil{dilation_iters}_{target_suffix}_shard{idx + 1:02d}.tck"
                for idx in range(len(shard_counts))
            ]
            update_status(
                probe_root,
                phase="tckgen_zero_roi_sharded",
                zero_label_count=seed_info["zero_label_count"],
                select_streamlines=target_streamlines,
                target_shards=len(shard_counts),
                shard_threads=shard_threads,
                shard_streamlines=";".join(str(count) for count in shard_counts),
            )

            def run_shard(idx: int, count: int, shard_path: Path) -> None:
                if shard_path.exists() and shard_path.stat().st_size > 0:
                    return
                shard_cmd = [
                    route7.TCKGEN,
                    fpaths["wmfod"],
                    shard_path,
                    "-act",
                    fpaths["act"],
                    "-backtrack",
                    "-cutoff",
                    "0.04",
                    "-maxlength",
                    "250",
                    "-minlength",
                    "10",
                    "-select",
                    str(count),
                    "-seed_image",
                    seed_mask,
                    "-nthreads",
                    str(shard_threads),
                    "-force",
                ]
                route7.run_track_cmd(
                    shard_cmd,
                    logs / f"tckgen_zero_roi_dil{dilation_iters}_{target_suffix}_shard{idx + 1:02d}.log",
                    env,
                    f"tckgen_shard{idx + 1:02d}",
                )

            with ThreadPoolExecutor(max_workers=len(shard_counts)) as pool:
                futures = [
                    pool.submit(run_shard, idx, count, shard_path)
                    for idx, (count, shard_path) in enumerate(zip(shard_counts, shard_tracks))
                ]
                for future in as_completed(futures):
                    future.result()

            update_status(probe_root, phase="tckedit_zero_roi_shards")
            run_cmd(
                [TCKEDIT, *shard_tracks, target_tracks, "-nthreads", str(max(1, mrtrix_threads)), "-force"],
                logs / f"tckedit_zero_roi_dil{dilation_iters}_{target_suffix}_shards.log",
                env,
                timeout_sec=2 * 60 * 60,
            )

    if not combined_tracks.exists() or combined_tracks.stat().st_size <= 0:
        update_status(probe_root, phase="tckedit_hybrid")
        run_cmd(
            [TCKEDIT, dense_tracks, target_tracks, combined_tracks, "-nthreads", str(mrtrix_threads), "-force"],
            logs / f"tckedit_hybrid_dynamic{dense_suffix}_zero_roi{target_suffix}.log",
            env,
            timeout_sec=2 * 60 * 60,
        )

    if not weights.exists() or weights.stat().st_size <= 0:
        update_status(probe_root, phase="tcksift2_hybrid")
        cmd = [
            route7.TCKSIFT2,
            combined_tracks,
            fpaths["wmfod"],
            weights,
            "-act",
            fpaths["act"],
            "-out_mu",
            mu,
            "-nthreads",
            str(max(1, mrtrix_threads)),
        ]
        if route7.deconv_mode(fpaths["deconv_mode"]) == "single_shell":
            cmd.append("-fd_scale_gm")
        route7.run_track_cmd(cmd, logs / f"tcksift2_hybrid_dynamic{dense_suffix}_zero_roi{target_suffix}.log", env, "tcksift2")

    label_stage = f"route16_hybrid_dynamic{dense_suffix}_zero_roi{target_suffix}_dil{dilation_iters}_{base_stage}"
    label_rows.append({"stage": label_stage, **parc_candidate["label_stats"]})
    write_csv(probe_root / "label_survival_summary.csv", label_rows)
    route7.run_connectomes(
        sid=sid,
        probe_root=probe_root,
        label_stage=label_stage,
        parc_path=Path(parc_candidate["path"]),
        tracks=combined_tracks,
        weights=weights,
        variants=variants,
        label_rows=label_rows,
        matrix_rows=matrix_rows,
        valid_labels=valid_labels,
        env=env,
    )

    best = route7.choose_best_matrix(matrix_rows, label_rows, baseline_fd)
    batch_row = route7.row_from_best(
        sid,
        "ok",
        probe_root,
        best,
        baseline_fd,
        label_rows,
        {
            "route16_action": "zero_roi_hybrid",
            "route11_schema": SCHEMA,
            "route11_lane": LANE,
            "target_lane": LANE,
            "dense_streamlines": dense_streamlines,
            "target_streamlines": target_streamlines,
            "target_shards": max(1, target_shards),
            "dilation_iters": seed_info["dilation_iters"],
            "zero_label_count": seed_info["zero_label_count"],
            "seed_voxels": seed_info["seed_voxels"],
            "dense_tracks": str(dense_tracks),
            "target_tracks": str(target_tracks),
            "combined_tracks": str(combined_tracks),
        },
    )
    write_json(probe_root / "decision.json", {"decision": "AAL3_ROUTE16_ZERO_ROI_HYBRID", "best": best, "batch_row": batch_row})
    update_status(probe_root, phase="complete", completed_utc=utc(), decision="AAL3_ROUTE16_ZERO_ROI_HYBRID", latest=batch_row)
    return batch_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, default=DEFAULT_ROUTE1_ROOT)
    parser.add_argument("--route2-root", type=Path, default=None)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--target-streamlines", type=int, default=3_000_000)
    parser.add_argument("--dense-streamlines", type=int, default=12_000_000)
    parser.add_argument("--dilation-iters", type=int, default=2)
    parser.add_argument("--mrtrix-threads", type=int, default=24)
    parser.add_argument("--target-shards", type=int, default=1)
    parser.add_argument("--variants", nargs="*", default=["forward80", "forward120", "radial16", "radial24", "endvox"])
    args = parser.parse_args()

    route1_root = args.route1_root
    route2_root = args.route2_root or route1_root.parent / f"{route1_root.name}_route2"
    run_root = route1_root.parent / f"{route1_root.name}{SUFFIX}"
    run_root.mkdir(parents=True, exist_ok=True)
    results = [row for row in read_csv(run_root / "batch_results.csv") if row.get("sid") not in set(args.subjects)]
    write_json(
        run_root / "run_manifest.json",
        {
            "mode": SCHEMA,
            "route1_root": str(route1_root),
            "route2_root": str(route2_root),
            "root": str(run_root),
            "probe_parent": str(QC_ROOT / PROBE_NAME),
            "subjects": sorted(set(args.subjects)),
            "target_streamlines": args.target_streamlines,
            "dense_streamlines": args.dense_streamlines,
            "dilation_iters": args.dilation_iters,
            "mrtrix_threads": args.mrtrix_threads,
            "target_shards": args.target_shards,
            "variants": args.variants,
            "started_utc": utc(),
        },
    )
    update_status(run_root, route11_schema=SCHEMA, phase="running", total=len(args.subjects), completed=0, failed=0, active=sorted(set(args.subjects)))

    for sid in sorted(set(args.subjects)):
        try:
            row = run_subject(
                sid=sid,
                route1_root=route1_root,
                route2_root=route2_root,
                run_root=run_root,
                target_streamlines=args.target_streamlines,
                dense_streamlines=args.dense_streamlines,
                dilation_iters=args.dilation_iters,
                variants=args.variants,
                mrtrix_threads=max(1, args.mrtrix_threads),
                target_shards=max(1, args.target_shards),
            )
        except Exception as exc:
            row = {
                "sid": sid,
                "status": "failed",
                "route11_schema": SCHEMA,
                "route11_lane": LANE,
                "target_lane": LANE,
                "error": f"{type(exc).__name__}: {exc}",
                "finished_utc": utc(),
            }
        results = [old for old in results if old.get("sid") != sid] + [row]
        write_csv(run_root / "batch_results.csv", results)
        update_status(
            run_root,
            completed=sum(1 for r in results if r.get("status") == "ok"),
            failed=sum(1 for r in results if r.get("status") == "failed"),
            latest=row,
        )
        refresh_outputs(route1_root, run_root)

    update_status(run_root, phase="complete", completed_utc=utc(), active=[])
    refresh_outputs(route1_root, run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
