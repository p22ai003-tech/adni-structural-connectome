#!/usr/bin/env python
"""Imaging pipeline: a folder of T1 and DWI images to connectome matrices.

    python run_imaging.py doctor                       # can this machine run it?
    python run_imaging.py discover --raw-root /data/Images
    python run_imaging.py probe                        # fill the acquisition fields
    python run_imaging.py validate                     # check the input contract
    python run_imaging.py run --dry-run                # plan the Snakemake DAG
    python run_imaging.py run --execution-subset approved.csv --cores 16

The last step needs an approved execution subset. The v2 route will not start
tractography on a cohort nobody has signed off, and says so rather than
defaulting to "everything".

The four steps before ``run`` are cheap and the last one is not: a full cohort
is hours of tractography. They are separate commands so that everything
knowable in advance is known before any of that starts, which is the whole
point of ``doctor`` and ``validate``.

Stages, from scforge/workflow/rules/:

    00_manifest  input contract, checksums
    00_inputs    DICOM -> NIfTI, gradient tables
    01_dwi       denoise, Gibbs, eddy / motion, bias field
    02_anat      T1 processing
    03_spatial   DWI <-> T1 registration contract
    04_atlas     AAL3 into DWI space
    05_5tt_fod   tissue segmentation, response, FOD
    06_preflight the gate before tractography
    07_tracks    tractography, SIFT2, connectome assembly
    08_qc        QC and publication

The retry and recovery rule variants in that directory are not part of this
route: they are one-off extensions from past incidents and the Snakefile does
not include them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sc_config  # noqa: E402

WORKFLOW = PROJECT_ROOT / "scforge" / "workflow"
SNAKEFILE = WORKFLOW / "Snakefile"
DEFAULT_WORKFLOW_CONFIG = PROJECT_ROOT / "configs" / "connectome_v2.yaml"


def _delegate(module: str, argv: list[str]) -> int:
    mod = __import__(module)
    return int(mod.main(argv) or 0)


def find_snakemake() -> str | None:
    """Snakemake lives in its own virtual environment in this project."""
    for cand in (PROJECT_ROOT / ".venv_connectome_workflow" / "bin" / "snakemake",
                 PROJECT_ROOT / ".venv" / "bin" / "snakemake"):
        if cand.exists():
            return str(cand)
    return shutil.which("snakemake")


def apply_study(args) -> "object | None":
    """Load --study, export the paths it implies and make the data readable.

    The study file is the one place a user describes their data. Exporting the
    SC_* variables here means every delegated tool sees the same locations
    without any of them having to know about the file.
    """
    path = getattr(args, "study", None)
    if not path:
        return None
    import sc_study

    try:
        study = sc_study.load_study(path)
    except sc_study.StudyError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
    study.apply_environment()
    if getattr(args, "command", "") in {"discover", "run"} or getattr(args, "stage", False):
        try:
            study.ensure_available()
        except sc_study.StudyError as error:
            print(str(error), file=sys.stderr)
            raise SystemExit(2)
    return study


def cmd_lock_env(args) -> int:
    argv = ["--check"] if args.check else []
    if args.out:
        argv += ["--out", str(args.out)]
    return _delegate("sc_lock_environment", argv)


def cmd_doctor(args) -> int:
    argv = ["--imaging"] if args.imaging_only else []
    study = getattr(args, "_study", None)
    if study is not None:
        argv += ["--layout", study.layout]
    return _delegate("sc_doctor", argv)


def cmd_discover(args) -> int:
    study = getattr(args, "_study", None)
    out = args.out or sc_config.paths().manifest
    raw_root = args.raw_root or (study.staged_root if study else sc_config.paths().raw_images_root)
    layout = args.layout or (study.layout if study else "adni")
    argv = ["--raw-root", str(raw_root), "--layout", layout, "--out", str(out), "--report"]
    if args.unpaired_out:
        argv += ["--unpaired-out", str(args.unpaired_out)]
    gap = args.max_gap_days
    if gap is None and study is not None:
        gap = study.max_gap_days
    if gap is not None:
        argv += ["--max-gap-days", str(gap)]
    if study is not None:
        if study.subjects:
            argv += ["--subjects", *study.subjects]
        if study.participants and study.participants.is_file():
            argv += ["--participants", str(study.participants)]
    return _delegate("sc_discover", argv)


def cmd_probe(args) -> int:
    manifest = args.manifest or sc_config.paths().manifest
    # Written in place: the probe only fills blank acquisition fields.
    return _delegate("sc_probe_acquisition",
                     ["--manifest", str(manifest), "--out", str(args.out or manifest), "--report"])


def cmd_validate(args) -> int:
    manifest = args.manifest or sc_config.paths().manifest
    rc = _delegate("sc_manifest", ["validate", str(manifest)])
    if rc:
        print("\nThe manifest does not satisfy the input contract.")
        print("If phase_encoding_direction / total_readout_time are blank, run:")
        print("    python run_imaging.py probe")
    return rc


def _lifecycle():
    import sc_lifecycle

    return sc_lifecycle


def _fail(error) -> int:
    print(str(error), file=sys.stderr)
    return 1


def cmd_approve(args) -> int:
    """Authorise units for phase A and pre-tractography: the first human gate.

    The decision is drafted by the workflow's launcher -- the same code that
    will validate it before every phase -- and only your name and the time are
    added here. It pins the exact manifest, subset, recipe, workflow source and
    environment, so nothing can change underneath an approved run.
    """
    lc = _lifecycle()
    manifest = Path(args.manifest or sc_config.paths().manifest).resolve()
    if not manifest.is_file():
        return _fail(f"manifest not found: {manifest}")
    run_root = Path(args.run_root).resolve()
    try:
        rows = lc._parent_rows(manifest, run_root)
    except Exception as error:
        return _fail(f"manifest is not a v2 acquisition manifest: {error}")
    units = sorted(r["unit"] for r in rows)
    if args.units:
        chosen = list(args.units)
    elif args.units_file:
        chosen = [l.strip() for l in Path(args.units_file).read_text().splitlines() if l.strip()]
    elif args.all:
        chosen = units
    else:
        chosen = units[: args.first]
    try:
        result = lc.approve(
            manifest=manifest, run_root=run_root, units=chosen, by=args.by, utc=args.utc,
            note=args.note, max_cores=args.max_cores, wall_clock_hours=args.wall_clock_hours,
            storage_gb=args.storage_gb, min_valid_calibration=args.min_valid_calibration,
            min_manufacturers=args.min_manufacturers,
        )
    except lc.LifecycleError as error:
        return _fail(error)
    print(f"approved {len(result['units'])} unit(s) by {args.by}")
    print(f"  scanner families : {', '.join(result['families'])}")
    print(f"  T1 source classes: {', '.join(result['t1_classes'])}")
    print(f"  calibration needs: at least {result['minimum_valid']} valid response(s)")
    print(f"  source files     : {result['source_files']} locked by content hash")
    print(f"\nnext: python run_imaging.py run --run-root {run_root} --cores {args.max_cores}")
    return 0


def cmd_run(args) -> int:
    """Run whichever phase comes next, through the workflow's launcher.

    Each phase ends in an immutable completion record; the next phase will not
    start without it, and the human steps between phases (freeze, review,
    continue) will not start without theirs. Re-running this command after a
    human step picks up at the right phase.
    """
    lc = _lifecycle()
    run_root = Path(args.run_root).resolve()
    mode, what = lc.next_phase(run_root)
    if mode is None:
        print(what)
        return 0 if "complete" in what and "publish" in what else 1
    command = lc.launcher_command(run_root, mode, args.cores)
    print(f"next phase: {what}")
    print("$ " + " ".join(command))
    if args.dry_run:
        return 0
    returncode = subprocess.run(command, cwd=str(PROJECT_ROOT)).returncode
    after, what_next = lc.next_phase(run_root)
    print(f"\n{what_next}")
    return returncode


def cmd_freeze(args) -> int:
    """Pool phase A's valid responses and approve the calibration: a human gate."""
    lc = _lifecycle()
    try:
        result = lc.freeze(run_root=Path(args.run_root), by=args.by, utc=args.utc, note=args.note)
    except lc.LifecycleError as error:
        return _fail(error)
    print(f"calibration frozen: {result['valid']} valid response(s), {result['invalid']} excluded")
    print(f"next: python run_imaging.py run --run-root {Path(args.run_root).resolve()}")
    return 0


def cmd_review(args) -> int:
    """Record your verdict on the pre-tractography QC bundles: a human gate."""
    lc = _lifecycle()
    try:
        path = lc.review(run_root=Path(args.run_root), reviewer=args.reviewer, status=args.status,
                         units=args.units, utc=args.utc, note=args.note)
    except lc.LifecycleError as error:
        return _fail(error)
    print(f"recorded {args.status.upper()} by {args.reviewer} in {path}")
    print(f"next: python run_imaging.py continue --run-root {Path(args.run_root).resolve()} --by \"<name>\"")
    return 0


def cmd_continue(args) -> int:
    """Authorise tractography for the reviewed units: the last human gate."""
    lc = _lifecycle()
    try:
        result = lc.continue_to_tractography(run_root=Path(args.run_root), by=args.by,
                                             utc=args.utc, note=args.note)
    except (lc.LifecycleError, ValueError) as error:
        return _fail(error)
    print(f"tractography authorised for {len(result['units'])} unit(s)")
    print(f"next: python run_imaging.py run --run-root {Path(args.run_root).resolve()}")
    return 0


def cmd_sweep(args) -> int:
    """Reclaim disk from units whose matrices are already published.

    A subject costs about 2 GB through preprocessing and another gigabyte or so
    for its streamlines, so a 500-subject cohort needs a couple of terabytes if
    nothing is ever removed. Most of that is intermediate: the denoised and
    unringed volumes and the .tck are all regenerable, and now that every
    stochastic step is seeded, regenerable to the same answer.

    Nothing is removed for a unit whose matrices have not been published, and
    nothing that cannot be rebuilt is removed at all. Snakemake will simply
    rebuild what it finds missing, so a swept run is still resumable -- it just
    costs the compute again.
    """
    run_root = Path(args.run_root).expanduser().resolve()
    destination = (args.published or sc_config.paths().connectomes_dir).expanduser()
    subjects = run_root / "subjects"
    if not subjects.is_dir():
        print(f"no subjects under {run_root}", file=sys.stderr)
        return 2

    # Two levels, because two situations. `intermediates` keeps the
    # preprocessed DWI, which is what every later stage reads and whose rebuild
    # means eddy again -- the right choice while a cohort is still being worked
    # on. `all-regenerable` keeps only what the analysis and an audit need, for
    # a full-cohort run where the alternative is not finishing: a subject costs
    # about 15 GB complete, so 500 of them is roughly 8 TB.
    sweepable = [
        ("01_dwi/dwi_denoised.mif", "denoised DWI"),
        ("01_dwi/dwi_denoised_degibbs.mif", "unringed DWI"),
        ("00_inputs/dwi_raw.mif", "converted raw DWI"),
    ]
    if not args.keep_tracks:
        sweepable.insert(0, ("07_tractography/tracks_10m.tck", "streamlines"))
    if args.level == "all-regenerable":
        sweepable += [
            ("01_dwi/dwi_preproc.mif", "preprocessed DWI"),
            ("01_dwi/dwi_preproc_biascorr.mif", "bias-corrected DWI"),
            ("05_model/dwi_fod_shells.mif", "FOD shells"),
            ("05_model/dwi_tensor_shells.mif", "tensor shells"),
            ("05_model/5tt_t1_fsl.mif", "five-tissue segmentation"),
            ("03_spatial/mni_to_t1_1Warp.nii.gz", "template warp"),
            ("03_spatial/mni_to_t1_1InverseWarp.nii.gz", "template inverse warp"),
            ("04_atlas/b0_1mm_world_grid.nii.gz", "1 mm reference grid"),
            ("00_inputs/t1_native.nii", "converted T1"),
            ("05_model/wmfod.mif", "white-matter FOD"),
            ("05_model/wmfod_norm.mif", "normalised white-matter FOD"),
            ("05_model/tensor.mif", "diffusion tensor"),
            # eddy's diagnostic volumes. The QC report and its numbers stay;
            # these are the full-size images behind them.
            ("01_dwi/eddy_qc/eddy_residuals.nii.gz", "eddy residual volumes"),
            ("01_dwi/eddy_qc/eddy_outlier_free_data.nii.gz", "eddy outlier-free volumes"),
        ]

    total = 0
    swept_units = skipped = 0
    for unit_dir in sorted(p for p in subjects.iterdir() if p.is_dir()):
        unit = unit_dir.name
        subject, _, image = unit.rpartition("_I")
        # Matrices for this subject may exist from an earlier run; that is no
        # reason to delete this one's inputs. The provenance sidecar names the
        # run that produced them, and only that makes a unit sweepable.
        sidecar = destination / f"SC_AAL_{subject}_I{image}_generation_provenance.json"
        published_here = False
        if sidecar.is_file():
            try:
                record = json.loads(sidecar.read_text())
                published_here = (Path(record.get("run_root", "")) == run_root
                                  and len(record.get("files", [])) >= 9)
            except (ValueError, OSError):
                published_here = False
        if not published_here and not args.force:
            skipped += 1
            continue
        freed = 0
        for relative, _label in sweepable:
            target = unit_dir / relative
            if not target.is_file():
                continue
            freed += target.stat().st_size
            if not args.dry_run:
                target.unlink()
        if freed:
            swept_units += 1
            total += freed
            verb = "would free" if args.dry_run else "freed"
            print(f"  {unit:<28} {verb} {freed / 2**30:.2f} GB")

    verb = "would reclaim" if args.dry_run else "reclaimed"
    print(f"\n{verb} {total / 2**30:.1f} GB from {swept_units} unit(s)")
    if skipped:
        print(f"{skipped} unit(s) left alone: their matrices are not published yet")
    return 0


def cmd_publish(args) -> int:
    argv = ["--run-root", str(args.run_root)]
    if args.out:
        argv += ["--out", str(args.out)]
    if args.units:
        argv += ["--units", *args.units]
    if args.overwrite:
        argv.append("--overwrite")
    if args.dry_run:
        argv.append("--dry-run")
    return _delegate("sc_publish", argv)


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _file_record(path: Path) -> dict:
    return {"path": str(path), "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="run_imaging", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", type=Path, default=os.environ.get("SC_STUDY") or None,
                    help="study configuration describing where the data is and how it is "
                         "laid out (see sc_study.py --init); also read from $SC_STUDY")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check paths, toolchain and packages")
    d.add_argument("--imaging-only", action="store_true")
    d.set_defaults(func=cmd_doctor)

    le = sub.add_parser("lock-env", help="record where this machine's imaging tools are")
    le.add_argument("--check", action="store_true", help="verify the recorded tools, write nothing")
    le.add_argument("--out", type=Path, default=None, help="default: configs/environment.local.yaml")
    le.set_defaults(func=cmd_lock_env)

    s = sub.add_parser("discover", help="raw image folder -> acquisition manifest")
    s.add_argument("--raw-root", type=Path, default=None,
                   help="default: the study's input, else $SC_RAW_IMAGES_ROOT")
    s.add_argument("--layout", choices=("simple", "adni", "bids"), default=None,
                   help="folder arrangement; default: the study's layout, else adni")
    s.add_argument("--out", type=Path, default=None)
    s.add_argument("--unpaired-out", type=Path, default=None)
    s.add_argument("--max-gap-days", type=float, default=None)
    s.set_defaults(func=cmd_discover)

    p = sub.add_parser("probe", help="fill phase encoding and readout time from the DICOMs")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_probe)

    v = sub.add_parser("validate", help="check the manifest against the v2 contract")
    v.add_argument("--manifest", type=Path, default=None)
    v.set_defaults(func=cmd_validate)

    a = sub.add_parser("approve", help="authorise units for processing (the first human gate)")
    a.add_argument("--manifest", type=Path, default=None, help="v2 acquisition manifest")
    a.add_argument("--run-root", type=Path, required=True, help="run directory; its path must contain 'scforge_v2'")
    a.add_argument("--units", nargs="*", default=None, help="unit ids to approve")
    a.add_argument("--units-file", type=Path, default=None, help="file with one unit id per line")
    a.add_argument("--all", action="store_true", help="approve every unit in the manifest")
    a.add_argument("--first", type=int, default=2, help="approve the first N units (default 2)")
    a.add_argument("--by", required=True, help="who is approving this run")
    a.add_argument("--utc", default=None, help="approval timestamp (default: now)")
    a.add_argument("--note", default="", help="why these units")
    a.add_argument("--max-cores", type=int, default=8, help="core ceiling this approval permits")
    a.add_argument("--wall-clock-hours", type=int, default=72, help="wall-clock stop per phase")
    a.add_argument("--storage-gb", type=int, default=150, help="storage stop for the run")
    a.add_argument("--min-valid-calibration", type=int, default=None,
                   help="valid response functions phase A must produce (default: half the batch)")
    a.add_argument("--min-manufacturers", type=int, default=1,
                   help="scanner families the valid pool must span (default 1)")
    a.set_defaults(func=cmd_approve)

    fz = sub.add_parser("freeze", help="pool phase A's responses and approve the calibration (a human gate)")
    fz.add_argument("--run-root", type=Path, required=True)
    fz.add_argument("--by", required=True, help="who is approving the calibration")
    fz.add_argument("--utc", default=None)
    fz.add_argument("--note", default="")
    fz.set_defaults(func=cmd_freeze)

    rv = sub.add_parser("review", help="record your verdict on the QC bundles (a human gate)")
    rv.add_argument("--run-root", type=Path, required=True)
    rv.add_argument("--units", nargs="*", default=None, help="default: every reviewable unit")
    rv.add_argument("--status", default="PASS", choices=("PASS", "FAIL"))
    rv.add_argument("--reviewer", required=True, help="who looked at the QC bundles")
    rv.add_argument("--utc", default=None)
    rv.add_argument("--note", default="")
    rv.set_defaults(func=cmd_review)

    cn = sub.add_parser("continue", help="authorise tractography for the reviewed units (a human gate)")
    cn.add_argument("--run-root", type=Path, required=True)
    cn.add_argument("--by", required=True, help="who is authorising tractography")
    cn.add_argument("--utc", default=None)
    cn.add_argument("--note", default="")
    cn.set_defaults(func=cmd_continue)

    sw = sub.add_parser("sweep", help="reclaim disk from units already published")
    sw.add_argument("--run-root", type=Path, required=True)
    sw.add_argument("--published", type=Path, default=None,
                    help="where matrices were published (default: $SC_CONNECTOMES_DIR)")
    sw.add_argument("--level", choices=("intermediates", "all-regenerable"),
                    default="intermediates",
                    help="intermediates keeps the preprocessed DWI so later stages "
                         "can be rerun without eddy; all-regenerable keeps only what "
                         "the analysis and an audit need, for a full-cohort run")
    sw.add_argument("--keep-tracks", action="store_true",
                    help="keep the .tck files, which are the largest single item")
    sw.add_argument("--force", action="store_true",
                    help="sweep a unit whose matrices are not published")
    sw.add_argument("--dry-run", action="store_true")
    sw.set_defaults(func=cmd_sweep)

    pb = sub.add_parser("publish", help="run matrices -> the analysis connectome directory")
    pb.add_argument("--run-root", type=Path, required=True)
    pb.add_argument("--out", type=Path, default=None)
    pb.add_argument("--units", nargs="*", default=None)
    pb.add_argument("--overwrite", action="store_true")
    pb.add_argument("--dry-run", action="store_true")
    pb.set_defaults(func=cmd_publish)

    r = sub.add_parser("run", help="run the next phase through the workflow's launcher")
    r.add_argument("--run-root", type=Path, required=True)
    r.add_argument("--cores", type=int, default=4, help="must not exceed the approval's core ceiling")
    r.add_argument("--dry-run", action="store_true", help="say which phase is next and how it would run")
    r.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    args.command = args.cmd
    args._study = apply_study(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
