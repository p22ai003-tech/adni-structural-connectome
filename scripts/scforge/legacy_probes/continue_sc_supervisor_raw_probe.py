#!/usr/bin/env python3
"""Continue a scratch supervisor-style raw replay from 5TT onward.

This is intentionally scratch-only. It reuses the already generated DWI/FOD/T1
intermediate files and writes only inside the selected run directory.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import run_sc_supervisor_raw_end_to_end_probe as probe


def _require(path: Path, label: str) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing required {label}: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        required=True,
        help="Existing supervisor_raw_end_to_end_probe run directory.",
    )
    parser.add_argument("--sid", required=True)
    parser.add_argument("--select-streamlines", type=int, default=1_000_000)
    parser.add_argument("--cutoff", type=float, default=0.06)
    parser.add_argument("--nthreads", type=int, default=8)
    parser.add_argument(
        "--five-tt-mode",
        choices=("b0_resliced_t1", "native_t1_then_b0"),
        default="b0_resliced_t1",
        help="Generate 5TT directly on B0-resliced T1, or in native T1 then transform it to B0.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    work = run_root / "work"
    logs = run_root / "logs"
    env = probe.base_env(args.nthreads)

    try:
        work.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        probe.update_status(
            run_root,
            sid=args.sid,
            phase="continue_8_5tt",
            run_dir=str(run_root),
            select_streamlines=args.select_streamlines,
            five_tt_mode=args.five_tt_mode,
        )

        b0_first = _require(work / "vol0000.nii.gz", "B0")
        corrected_t1 = _require(work / "corrected_T1.nii.gz", "native corrected T1")
        t1_on_b0 = _require(work / "t12b0.nii.gz", "T1-on-B0")
        t12b0 = _require(work / "t12b0.mat", "T1-to-B0 matrix")
        brain = _require(work / "brainmask.nii.gz", "native T1 brain")
        wmfod_norm = _require(work / "wmfod_norm.mif", "normalised WM FOD")

        five_tt = work / "5tt1.mif"
        gmwmi = work / "gmwmSeed.mif"
        if args.five_tt_mode == "b0_resliced_t1":
            # Keep -debug: on this EC2 image, 5ttgen's FSL FIRST wrapper can
            # miss the run_first_all job id without verbose stdout even when
            # FIRST works.
            probe.run(
                [probe.tool("5ttgen", env), "fsl", t1_on_b0, five_tt, "-debug", "-force"],
                logs / "08_5tt.log",
                env,
                cwd=work,
            )
        else:
            # FIRST can fail after downsampling T1 to the B0 grid. This mode
            # tests the anatomically safer route: generate 5TT in native T1
            # space first, then reslice the resulting 5TT image to B0 space
            # using the same FLIRT T1->B0 transform.
            native_5tt = work / "5tt_native_t1.mif"
            mrtrix_xfm = work / "t12b0_mrtrix.txt"
            probe.run(
                [probe.tool("5ttgen", env), "fsl", corrected_t1, native_5tt, "-debug", "-force"],
                logs / "08_5tt_native_to_b0.log",
                env,
                cwd=work,
            )
            probe.run(
                [
                    probe.tool("transformconvert", env),
                    t12b0,
                    corrected_t1,
                    b0_first,
                    "flirt_import",
                    mrtrix_xfm,
                    "-force",
                ],
                logs / "08_5tt_native_to_b0.log",
                env,
                cwd=work,
            )
            probe.run(
                [
                    probe.tool("mrtransform", env),
                    native_5tt,
                    five_tt,
                    "-linear",
                    mrtrix_xfm,
                    "-template",
                    b0_first,
                    "-interp",
                    "linear",
                    "-force",
                ],
                logs / "08_5tt_native_to_b0.log",
                env,
                cwd=work,
            )
        probe.run(
            [probe.tool("5tt2gmwmi", env), five_tt, gmwmi, "-force"],
            logs / "08_5tt.log",
            env,
            cwd=work,
        )

        probe.update_status(run_root, phase="continue_9_tckgen")
        tracks = work / f"tracks_{args.select_streamlines}.tck"
        probe.run(
            [
                probe.tool("tckgen", env),
                "-act",
                five_tt,
                "-backtrack",
                "-seed_gmwmi",
                gmwmi,
                "-maxlength",
                "250",
                "-cutoff",
                str(args.cutoff),
                "-select",
                str(args.select_streamlines),
                "-nthreads",
                str(args.nthreads),
                wmfod_norm,
                tracks,
                "-force",
            ],
            logs / "09_tckgen.log",
            env,
            cwd=work,
        )

        probe.update_status(run_root, phase="continue_10_tcksift2")
        weights = work / f"sift_{args.select_streamlines}.txt"
        probe.run(
            [
                probe.tool("tcksift2", env),
                "-act",
                five_tt,
                "-out_coeffs",
                work / "sift_coeffs.txt",
                "-nthreads",
                str(args.nthreads),
                tracks,
                wmfod_norm,
                weights,
                "-force",
            ],
            logs / "10_tcksift2.log",
            env,
            cwd=work,
        )

        probe.update_status(run_root, phase="continue_11_aal116_to_b0")
        aal2mni_mat = work / "AAL2MNI.mat"
        aal2mni_img = work / "AAL2MNI.nii.gz"
        probe.run(
            [
                str(probe.FSL_BIN / "flirt"),
                "-in",
                probe.AAL116,
                "-ref",
                probe.MNI_BRAIN,
                "-dof",
                "6",
                "-omat",
                aal2mni_mat,
                "-interp",
                "nearestneighbour",
                "-datatype",
                "int",
                "-out",
                aal2mni_img,
            ],
            logs / "11_aal_route.log",
            env,
            cwd=work,
        )
        aalmni2t1 = work / "aalmni2t1.mat"
        aal_t1 = work / "AALmni2t1.nii.gz"
        probe.run(
            [
                str(probe.FSL_BIN / "flirt"),
                "-in",
                aal2mni_img,
                "-ref",
                brain,
                "-omat",
                aalmni2t1,
                "-interp",
                "nearestneighbour",
                "-datatype",
                "int",
                "-out",
                aal_t1,
                "-dof",
                "12",
            ],
            logs / "11_aal_route.log",
            env,
            cwd=work,
        )
        aal_b0 = work / "AAL_sub.nii.gz"
        probe.run(
            [
                str(probe.FSL_BIN / "flirt"),
                "-in",
                aal_t1,
                "-ref",
                b0_first,
                "-applyxfm",
                "-init",
                t12b0,
                "-interp",
                "nearestneighbour",
                "-out",
                aal_b0,
            ],
            logs / "11_aal_route.log",
            env,
            cwd=work,
        )
        aal_b0_contig = work / "AAL_sub_contig116.nii.gz"
        remap_report = probe.aal116_sparse_to_contiguous(aal_b0, aal_b0_contig)
        probe.write_csv(run_root / "aal116_label_remap.csv", [remap_report])

        probe.update_status(run_root, phase="continue_12_connectome")
        sc = run_root / "SC_AAL.csv"
        tl = run_root / "TL_AAL.csv"
        assignments = run_root / "assignmentsaal.csv"
        probe.run(
            [
                probe.tool("tck2connectome", env),
                "-symmetric",
                "-zero_diagonal",
                "-scale_invnodevol",
                "-tck_weights_in",
                weights,
                tracks,
                aal_b0_contig,
                sc,
                "-out_assignment",
                assignments,
                "-force",
            ],
            logs / "12_connectome.log",
            env,
            cwd=work,
        )
        probe.run(
            [
                probe.tool("tck2connectome", env),
                "-symmetric",
                "-zero_diagonal",
                "-tck_weights_in",
                weights,
                tracks,
                aal_b0_contig,
                tl,
                "-scale_length",
                "-stat_edge",
                "mean",
                "-force",
            ],
            logs / "12_connectome.log",
            env,
            cwd=work,
        )

        probe.update_status(run_root, phase="continue_13_matrix_stats")
        stats = [probe.matrix_stats(sc, "probe_SC_AAL"), probe.matrix_stats(tl, "probe_TL_AAL")]
        benchmark = probe.SUPERVISOR_BENCHMARKS.get(probe.subject_id_from_sid(args.sid))
        if benchmark and benchmark.exists():
            stats.append(probe.matrix_stats(benchmark, "supervisor_benchmark_SC_AAL"))
        probe.write_csv(run_root / "matrix_stats.csv", stats)
        probe.write_findings(
            run_root,
            [
                "# Supervisor Raw End-to-End Probe Continuation",
                "",
                f"SID: `{args.sid}`",
                f"Run root: `{run_root}`",
                "Mode: continuation from FOD outputs after patched `5ttgen -debug`.",
                "",
                "## Matrix stats",
                "",
                *(
                    f"- {row['label']}: n={row.get('n')}, density={row.get('density')}, "
                    f"zero_rows={row.get('zero_rows')}, error={row.get('error', '')}"
                    for row in stats
                ),
                "",
                "## Interpretation rule",
                "",
                "If this continuation gives supervisor-like density and few/no whole zero rows, the reproducible route is upstream raw replay plus supervisor-style AAL116 routing.",
                "If it remains sparse, this subject is not rescued by the reference-style downstream route and needs deeper source/provenance or exclusion.",
            ],
        )
        probe.update_status(
            run_root,
            phase="done",
            matrix_stats_csv=str(run_root / "matrix_stats.csv"),
            findings_md=str(run_root / "findings.md"),
        )
        return 0
    except Exception as exc:
        (run_root / "continue_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        probe.update_status(run_root, phase="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
