#!/usr/bin/env python
"""Imaging pipeline: a folder of T1 and DWI images to connectome matrices.

    python run_imaging.py doctor                       # can this machine run it?
    python run_imaging.py discover --raw-root /data/Images
    python run_imaging.py probe                        # fill the acquisition fields
    python run_imaging.py validate                     # check the input contract
    python run_imaging.py run --dry-run                # plan the Snakemake DAG
    python run_imaging.py run --cores 16               # execute

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
import os
import shutil
import subprocess
import sys
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


def cmd_doctor(args) -> int:
    return _delegate("sc_doctor", ["--imaging"] if args.imaging_only else [])


def cmd_discover(args) -> int:
    out = args.out or sc_config.paths().manifest
    argv = ["--raw-root", str(args.raw_root), "--out", str(out), "--report"]
    if args.unpaired_out:
        argv += ["--unpaired-out", str(args.unpaired_out)]
    if args.max_gap_days is not None:
        argv += ["--max-gap-days", str(args.max_gap_days)]
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


def cmd_run(args) -> int:
    snakemake = find_snakemake()
    if not snakemake:
        raise SystemExit(
            "snakemake not found. It lives in .venv_connectome_workflow here; "
            "create it with: python3.12 -m venv .venv_connectome_workflow && "
            ".venv_connectome_workflow/bin/pip install -r scforge/workflow/requirements.lock.txt"
        )
    manifest = Path(args.manifest or sc_config.paths().manifest)
    if not manifest.is_file():
        raise SystemExit(f"manifest not found: {manifest}\nRun: python run_imaging.py discover")

    run_root = Path(args.run_root or (sc_config.paths().deriv_root / "scforge_v2_runs" / "run"))
    cmd = [snakemake, "--snakefile", str(SNAKEFILE),
           "--configfile", str(args.workflow_config or DEFAULT_WORKFLOW_CONFIG),
           "--config", f"manifest_path={manifest}", f"run_root={run_root}",
           "--cores", str(args.cores),
           "--printshellcmds", "--rerun-incomplete"]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.until:
        cmd += ["--until", args.until]
    if args.extra:
        cmd += args.extra

    env = dict(os.environ)
    # The workflow shells out to MRtrix, FSL and ANTs. Missing exports here are
    # reported as data failures deep in the run -- "unable to extract WM-FOD l=0
    # coefficient" was exactly that -- so they are checked before launching.
    t = sc_config.tools()
    for name, value in (("MRTRIX_BIN", t.mrtrix_bin), ("FSLDIR", t.fsl_dir),
                        ("ANTSPATH", t.ants_path)):
        if value:
            env[name] = str(value)
    missing = [n for n, v in (("MRTRIX_BIN", t.mrtrix_bin), ("FSLDIR", t.fsl_dir),
                              ("ANTSPATH", t.ants_path)) if not v]
    if missing and not args.dry_run:
        raise SystemExit(
            f"not set: {', '.join(missing)}. Run `python run_imaging.py doctor`, "
            "then `source env.sh`."
        )

    print(f"manifest : {manifest}")
    print(f"run root : {run_root}")
    print(f"snakemake: {snakemake}")
    print(f"$ {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT)).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="run_imaging", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check paths, toolchain and packages")
    d.add_argument("--imaging-only", action="store_true")
    d.set_defaults(func=cmd_doctor)

    s = sub.add_parser("discover", help="raw image folder -> acquisition manifest")
    s.add_argument("--raw-root", type=Path, default=sc_config.paths().raw_images_root)
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

    r = sub.add_parser("run", help="execute the Snakemake workflow")
    r.add_argument("--manifest", type=Path, default=None)
    r.add_argument("--run-root", type=Path, default=None)
    r.add_argument("--workflow-config", type=Path, default=None)
    r.add_argument("--cores", type=int, default=1)
    r.add_argument("--dry-run", action="store_true", help="plan only")
    r.add_argument("--until", default=None, help="stop after this rule")
    r.add_argument("extra", nargs="*", help="further arguments passed to snakemake")
    r.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
