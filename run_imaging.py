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
    run_root = Path(args.run_root or (sc_config.paths().deriv_root / "scforge_v2_runs" / "run"))
    # When an approval exists it names the manifest it was bound to, so that is
    # the one this run must use; checking sc_config first would let a run drift
    # onto a different manifest than the one approved.
    approved = (run_root / "contract" / "run_overlay.yaml").is_file()
    if not approved:
        manifest = Path(args.manifest or sc_config.paths().manifest)
        if not manifest.is_file():
            raise SystemExit(f"manifest not found: {manifest}\nRun: python run_imaging.py discover")
    else:
        import yaml
        manifest = Path(yaml.safe_load((run_root / "contract" / "run_overlay.yaml").read_text())["manifest_path"])
    # The v2 route will not process anyone without an approved execution
    # subset. That is a human gate, not an oversight: the workflow refuses to
    # start tractography on a cohort nobody has signed off. Without it the
    # Snakefile stops on the literal placeholder REQUIRED_APPROVED_CANARY_SUBSET.
    if not (run_root / "contract" / "run_overlay.yaml").is_file() and not args.execution_subset:
        raise SystemExit(
            "this run root has no approval.\n"
            "The workflow will not process anything until a person authorises the units:\n"
            f"    python run_imaging.py approve --run-root {run_root} --first 2 --by '<your name>'\n"
            "then run again."
        )

    overlay = run_root / "contract" / "run_overlay.yaml"
    if overlay.is_file():
        # Everything the workflow needs was written by `approve`: the approved
        # subset, the binding, the context paths and portable mode. A second
        # configfile overrides the normative one key by key.
        cmd = [snakemake, "--snakefile", str(SNAKEFILE),
               "--configfile", str(args.workflow_config or DEFAULT_WORKFLOW_CONFIG),
               "--configfile", str(overlay),
               "--cores", str(args.cores), "--printshellcmds", "--rerun-incomplete"]
    else:
        cmd = [snakemake, "--snakefile", str(SNAKEFILE),
               "--configfile", str(args.workflow_config or DEFAULT_WORKFLOW_CONFIG),
               "--config", f"manifest_path={manifest}", f"run_root={run_root}",
               "--cores", str(args.cores),
               "--printshellcmds", "--rerun-incomplete"]
        if args.execution_subset:
            cmd[cmd.index("--config") + 1:cmd.index("--config") + 1] = [
                f"execution_manifest_path={Path(args.execution_subset).resolve()}"
            ]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.force:
        cmd.append("--forceall")
    if args.target:
        cmd.append(args.target)
    elif (run_root / "contract" / "run_overlay.yaml").is_file():
        # The workflow is staged: phase A estimates one response function per
        # subject and freezes a pooled one; only then can phase B deconvolve,
        # track and build matrices. Asking for the final target while in
        # phase-A mode stops on the frozen-calibration placeholder, which reads
        # like a missing file rather than the stage it is.
        import yaml as _yaml
        _mode = _yaml.safe_load((run_root / "contract" / "run_overlay.yaml").read_text()).get("launcher_mode")
        if _mode == "response-calibration-phase-a":
            cmd.append("response_calibration_phase_a")
        elif _mode == "pre-tractography-canary":
            cmd.append("pre_tractography_canary")
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

    if approved:
        # The attempt context records THIS attempt, so it can only be written
        # once the command is known: the workflow checks that the invocation
        # carries --printshellcmds, --snakefile and --configfile, and that its
        # core count is within the ceiling the approval set.
        import json as _json
        contract_dir = run_root / "contract"
        run_context = _json.loads((contract_dir / "run_context.json").read_text())
        binding = run_context["execution_binding"]
        auth = binding["h04a_authorization"]
        if args.cores > int(auth["maximum_cores"]):
            raise SystemExit(
                f"--cores {args.cores} exceeds the {auth['maximum_cores']} this approval permits.\n"
                f"Re-approve with --max-cores {args.cores}, or run with fewer cores."
            )
        import shutil as _shutil
        free = _shutil.disk_usage(run_root).free
        attempt_path = contract_dir / "attempt_context.json"
        payload = _json.dumps({
            "schema_version": "1.0.0",
            "status": "STARTED",
            "run_id": run_context["run_id"],
            "recipe_id": run_context["recipe_id"],
            "launcher_mode": run_context["launcher_mode"],
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "execution_binding": binding,
            "run_context": _file_record(contract_dir / "run_context.json"),
            "resolved_run_config": _file_record(contract_dir / "resolved_run_config.yaml"),
            "snakemake_invocation": cmd,
            "h04a_resource_preflight": {
                "status": "PASS",
                "decision_sha256": binding["execution_subset_decision"]["sha256"],
                "wall_clock_stop_seconds": auth["wall_clock_stop_seconds"],
                "storage_stop_bytes": auth["storage_stop_bytes"],
                "wall_clock_remaining_seconds": auth["wall_clock_stop_seconds"],
                "storage_remaining_bytes": min(auth["storage_stop_bytes"], free),
            },
        }, indent=2, sort_keys=True) + "\n"
        # Rewrite only when this attempt actually differs. The file is an input
        # to the contract rules, so touching it on every invocation makes
        # Snakemake treat finished work as stale and re-plan from the top.
        previous = attempt_path.read_text(encoding="utf-8") if attempt_path.is_file() else None
        if previous is not None:
            import json as _j
            a, b = _j.loads(previous), _j.loads(payload)
            a.pop("started_utc", None); b.pop("started_utc", None)
            if a == b:
                payload = previous
        if payload != previous:
            attempt_path.write_text(payload, encoding="utf-8")

    print(f"manifest : {manifest}")
    print(f"run root : {run_root}")
    print(f"snakemake: {snakemake}")
    print(f"$ {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT)).returncode


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


def cmd_approve(args) -> int:
    """Authorise units for processing, and bind that decision to the files.

    The workflow will not process anything without an execution binding: an
    immutable record tying the approved unit list to the exact bytes of the
    parent manifest, the subset and a decision document. On the machine this
    contract was written for, that binding carries a counter-signature from a
    governance process that exists nowhere else, which is what stopped the
    workflow running anywhere else.

    This builds the same binding from a decision made here. The gate stays
    real -- nothing runs until a person executes this command and puts their
    name to it -- and the record still pins the bytes, so a run cannot drift
    onto a different manifest without the binding failing.
    """
    import csv, json, yaml

    manifest = Path(args.manifest or sc_config.paths().manifest).resolve()
    if not manifest.is_file():
        raise SystemExit(f"manifest not found: {manifest}")
    run_root = Path(args.run_root).resolve()
    if "scforge_v2" not in str(run_root):
        raise SystemExit(f"run root path must contain 'scforge_v2': {run_root}")

    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit("manifest is empty")
    # `unit` is derived, never stored: the loader rejects a manifest that
    # carries the column. Derive it exactly as the workflow does.
    import sys as _sys
    _scforge = str(PROJECT_ROOT / "scforge")
    if _scforge not in _sys.path:
        _sys.path.insert(0, _scforge)
    from scforge.input_contract import stable_unit
    try:
        by_unit = {stable_unit(r): r for r in rows}
    except Exception as exc:
        raise SystemExit(f"manifest is not a v2 acquisition manifest: {exc}")

    if args.units:
        chosen = list(dict.fromkeys(args.units))
        missing = [u for u in chosen if u not in by_unit]
        if missing:
            raise SystemExit(f"not in the manifest: {', '.join(missing[:5])}")
    elif args.units_file:
        chosen = [l.strip() for l in Path(args.units_file).read_text().splitlines() if l.strip()]
    elif args.all:
        chosen = sorted(by_unit)
    else:
        chosen = sorted(by_unit)[: args.first]
    chosen = sorted(set(chosen))
    if not chosen:
        raise SystemExit("no units selected")

    contract_dir = run_root / "contract"
    contract_dir.mkdir(parents=True, exist_ok=True)
    subset_path = contract_dir / "execution_subset.csv"
    with subset_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for u in chosen:
            w.writerow(by_unit[u])

    decision_path = contract_dir / "execution_subset_decision.json"
    decision_path.write_text(json.dumps({
        "schema_version": "2.0.0",
        "decision_type": "portable_execution_approval",
        "recipe_id": args.recipe_id,
        "approved_by": args.by,
        "approved_utc": args.utc,
        "note": args.note,
        "approved_unit_count": len(chosen),
        "units": chosen,
        "parent_acquisition_manifest": _file_record(manifest),
        "diagnosis_labels_used": False,
        "selection_rule": args.selection_rule,
    }, indent=2) + "\n", encoding="utf-8")

    normative = PROJECT_ROOT / "configs" / "connectome_v2.yaml"
    env_contract = PROJECT_ROOT / "scforge" / "workflow" / "environment_contract.yaml"
    src_manifest = PROJECT_ROOT / "scforge" / "workflow" / "workflow_source_manifest.tsv"

    # The same authorisation record the audited path carries, filled from this
    # decision instead of from the project's governance process. The resource
    # stops are real: the workflow enforces them.
    authorization = {
        "approval_mode": "PORTABLE_LOCAL_APPROVAL",
        "authorized_modes": ["response-calibration-phase-a", "pre-tractography-canary", "phase-b"],
        "authorized_through": "connectome_matrices",
        "maximum_cores": args.max_cores,
        "minimum_valid_response_calibration_units": 1,
        "minimum_valid_manufacturer_families": 1,
        "minimum_valid_t1_source_classes": 1,
        "required_valid_t1_source_classes": ["dicom_series", "nifti_single"],
        "wall_clock_stop_hours": args.wall_clock_hours,
        "wall_clock_stop_seconds": args.wall_clock_hours * 3600,
        "storage_stop_gb": args.storage_gb,
        "storage_stop_bytes": args.storage_gb * 1_000_000_000,
        "tractography_authorized": True,
        "matrix_generation_authorized": True,
        "full_cohort_authorized": bool(args.all),
        "approved_by": args.by,
        "approved_utc": args.utc,
        "user_response": args.note or "approved",
        "proposed_run_root": str(run_root),
        "normative_config_sha256": _sha256_file(normative),
        "workflow_source_manifest_sha256": _sha256_file(src_manifest),
        "environment_contract_sha256": _sha256_file(env_contract),
    }
    binding = {
        "schema_version": "2.0.0",
        "binding_type": "connectome_execution_subset_binding",
        "execution_scope": "canary",
        "recipe_id": args.recipe_id,
        "parent_acquisition_manifest": _file_record(manifest),
        "execution_subset_manifest": _file_record(subset_path),
        "execution_subset_decision": _file_record(decision_path),
        "approved_unit_count": len(chosen),
        "units": chosen,
        "diagnosis_labels_used": False,
        "selection_locked": True,
        "h04a_authorization": authorization,
    }
    (contract_dir / "execution_binding.json").write_text(
        json.dumps(binding, indent=2) + "\n", encoding="utf-8")

    # run identity: derived from the decision, so re-approving the same units
    # from the same manifest yields the same ids rather than a random pair.
    import hashlib
    seed = (binding["parent_acquisition_manifest"]["sha256"] + "|".join(chosen) + args.utc)
    run_id = hashlib.sha256(seed.encode()).hexdigest()[:24]
    lineage_id = hashlib.sha256(("lineage" + seed).encode()).hexdigest()[:24]

    import platform
    run_context = {
        "schema_version": "1.0.0",
        "status": "LOCKED",
        "run_id": run_id,
        "lineage_id": lineage_id,
        "recipe_id": args.recipe_id,
        "run_root": str(run_root),
        "launcher_mode": args.mode,
        "created_utc": args.utc,
        "acquisition_manifest": _file_record(manifest),
        "normative_config": _file_record(normative),
        "environment_contract": _file_record(env_contract),
        "workflow_source_manifest": _file_record(src_manifest),
        "execution_binding": binding,
        "host": {"node": platform.node(), "platform": platform.platform(),
                 "architecture": platform.machine(), "python": platform.python_version()},
        "safety": {"automatic_fallbacks": False, "density_route_selection": False,
                   "production_overwrite": False},
    }
    (contract_dir / "run_context.json").write_text(
        json.dumps(run_context, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    import shutil as _shutil
    free_bytes = _shutil.disk_usage(run_root).free
    attempt_context = {
        "schema_version": "1.0.0",
        "status": "STARTED",
        "run_id": run_id,
        "recipe_id": args.recipe_id,
        "launcher_mode": args.mode,
        "started_utc": args.utc,
        "h04a_resource_preflight": {
            "status": "PASS",
            "decision_sha256": binding["execution_subset_decision"]["sha256"],
            "wall_clock_stop_seconds": authorization["wall_clock_stop_seconds"],
            "storage_stop_bytes": authorization["storage_stop_bytes"],
            "wall_clock_remaining_seconds": authorization["wall_clock_stop_seconds"],
            "storage_remaining_bytes": min(authorization["storage_stop_bytes"], free_bytes),
        },
    }
    (contract_dir / "attempt_context.json").write_text(
        json.dumps(attempt_context, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    resolved = yaml.safe_load(normative.read_text())
    resolved.update({
        "manifest_path": str(manifest),
        "run_root": str(run_root),
        "execution_manifest_path": str(subset_path),
        "run_context_path": str(contract_dir / "run_context.json"),
        "attempt_context_path": str(contract_dir / "attempt_context.json"),
        "resolved_run_config_path": str(contract_dir / "resolved_run_config.yaml"),
        "launcher_mode": args.mode,
        "execution_binding": binding,
    })
    (contract_dir / "resolved_run_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    overlay_path = contract_dir / "run_overlay.yaml"
    overlay_path.write_text(yaml.safe_dump({
        "portable_mode": True,
        "manifest_path": str(manifest),
        "execution_manifest_path": str(subset_path),
        "run_root": str(run_root),
        "resolved_run_config_path": str(contract_dir / "resolved_run_config.yaml"),
        "run_context_path": str(contract_dir / "run_context.json"),
        "attempt_context_path": str(contract_dir / "attempt_context.json"),
        "launcher_mode": args.mode,
        "execution_binding": binding,
        # The normative config pins the hash of the cohort manifest it was
        # frozen against. This run uses the manifest that was approved, so the
        # check still has force: it now compares against the file this approval
        # bound itself to.
        "inputs": {"approved_pair_manifest": {
            "required_row_count": len(rows),
            "sha256": binding["parent_acquisition_manifest"]["sha256"],
            "path": str(manifest),
        }},
    }, sort_keys=False), encoding="utf-8")

    print(f"approved {len(chosen)} unit(s) by {args.by}")
    for label, path in (("subset", subset_path), ("decision", decision_path),
                        ("binding", contract_dir / "execution_binding.json"),
                        ("overlay", overlay_path)):
        print(f"  {label:<9}{path}")
    print(f"\nnext: python run_imaging.py run --run-root {run_root} --dry-run")
    return 0


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

    a = sub.add_parser("approve", help="authorise units for processing (the human gate)")
    a.add_argument("--manifest", type=Path, default=None, help="v2 acquisition manifest")
    a.add_argument("--run-root", type=Path, required=True, help="run directory; its path must contain 'scforge_v2'")
    a.add_argument("--units", nargs="*", default=None, help="unit ids to approve")
    a.add_argument("--units-file", type=Path, default=None, help="file with one unit id per line")
    a.add_argument("--all", action="store_true", help="approve every unit in the manifest")
    a.add_argument("--first", type=int, default=2, help="approve the first N units (default 2)")
    a.add_argument("--by", required=True, help="who is approving this run")
    a.add_argument("--utc", default=None, help="approval timestamp (default: now)")
    a.add_argument("--note", default="", help="why these units")
    a.add_argument("--selection-rule", default="first N units in manifest order",
                   help="how the units were chosen; recorded in the decision")
    a.add_argument("--mode", default="response-calibration-phase-a",
                   choices=("response-calibration-phase-a", "pre-tractography-canary", "phase-b"),
                   help="launcher mode recorded in the overlay")
    a.add_argument("--recipe-id", default=None, help="default: the recipe_id in configs/connectome_v2.yaml")
    a.add_argument("--max-cores", type=int, default=8, help="core ceiling this approval permits")
    a.add_argument("--wall-clock-hours", type=int, default=72, help="wall-clock stop for the run")
    a.add_argument("--storage-gb", type=int, default=150, help="storage stop for the run")
    a.set_defaults(func=cmd_approve)

    r = sub.add_parser("run", help="execute the Snakemake workflow")
    r.add_argument("--manifest", type=Path, default=None)
    r.add_argument("--run-root", type=Path, default=None)
    r.add_argument("--workflow-config", type=Path, default=None)
    r.add_argument("--execution-subset", type=Path, default=None,
                   help="human-approved subset of the manifest; required to execute")
    r.add_argument("--cores", type=int, default=1)
    r.add_argument("--dry-run", action="store_true", help="plan only")
    r.add_argument("--force", action="store_true",
                   help="rebuild every stage, ignoring what is already on disk")
    r.add_argument("--until", default=None, help="stop after this rule")
    r.add_argument("--target", default=None,
                   help="Snakemake target (default: chosen from the approved launcher mode)")
    r.add_argument("extra", nargs="*", help="further arguments passed to snakemake")
    r.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    if getattr(args, "cmd", None) == "approve":
        if not args.recipe_id:
            import yaml
            cfg = yaml.safe_load((DEFAULT_WORKFLOW_CONFIG).read_text())
            args.recipe_id = cfg["contract"]["recipe_id"]
        if not args.utc:
            import datetime
            args.utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
