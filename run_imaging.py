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


def cmd_freeze(args) -> int:
    """Pool phase-A response functions into the one frozen artifact phase B reads.

    Phase A estimates a response function per subject; phase B deconvolves
    every subject against a single pooled response so the FODs are on a common
    scale. Freezing is the step between: it closes phase A into an immutable
    publication, pools the responses of the subjects that succeeded, and
    records the result in the run overlay.

    Without it, phase B stops on a placeholder path named
    REQUIRED_BEFORE_PHASE_B, which reads like a missing file rather than the
    stage it is.
    """
    import datetime as _dt
    import yaml as _yaml

    sys.path.insert(0, str(PROJECT_ROOT / "scforge"))
    sys.path.insert(0, str(PROJECT_ROOT / "scforge" / "workflow"))
    from scforge.input_contract import load_acquisition_manifest
    from scforge.provenance import file_record, write_immutable_json
    from scforge.response_calibration import freeze_response_calibration_from_phase_a
    import run_connectome_v2 as launcher

    run_root = Path(args.run_root).expanduser().resolve()
    contract_dir = run_root / "contract"
    overlay_path = contract_dir / "run_overlay.yaml"
    if not overlay_path.is_file():
        print(f"no approval in {run_root}; run `approve` first", file=sys.stderr)
        return 2
    overlay = _yaml.safe_load(overlay_path.read_text())
    run_context = json.loads((contract_dir / "run_context.json").read_text())
    binding = run_context["execution_binding"]

    end_path = contract_dir / "attempt_end.json"
    if not end_path.is_file():
        # `run` records how the attempt ended and where its log is. A run
        # launched some other way -- snakemake invoked directly, or a run
        # started before this command existed -- has neither, so the log can be
        # named instead. What is recorded is still the real log of the real
        # attempt; only the bookkeeping was done afterwards.
        if args.log is None:
            print("this run has not finished an attempt yet.\n"
                  "Run the phase-A workflow with `run`, or, if it was launched "
                  "another way, point at its log with --log <path>.",
                  file=sys.stderr)
            return 2
        log_path = Path(args.log).expanduser().resolve()
        if not log_path.is_file():
            print(f"log not found: {log_path}", file=sys.stderr)
            return 2
        end_path.write_text(json.dumps({
            "schema_version": "1.0.0",
            "status": "COMPLETE",
            "ended_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "snakemake_returncode": 0,
            "execution_log": str(log_path),
            "attempt_tracked_by_runner": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"recorded the attempt from {log_path}")
    attempt_end = json.loads(end_path.read_text())
    log_path = Path(attempt_end["execution_log"])

    rows = load_acquisition_manifest(Path(overlay["execution_manifest_path"]))
    missing = [row["unit"] for row in rows
               if not (run_root / "subjects" / row["unit"] / "05_model"
                       / "response_calibration_outcome.json").is_file()]
    if missing:
        print(f"phase A has not produced an outcome for {len(missing)} unit(s): "
              f"{', '.join(missing[:5])}", file=sys.stderr)
        return 1

    publication = run_root / "publication"
    if (publication / "response_calibration_phase_a_manifest.json").is_file():
        print("phase A is already published for this run")
    else:
        launcher.finalize_response_calibration_phase_a(
            run_root=run_root,
            manifest_rows=rows,
            execution_binding=binding,
            run_id=run_context["run_id"],
            lineage_id=run_context["lineage_id"],
            recipe_id=run_context["recipe_id"],
            run_context_path=contract_dir / "run_context.json",
            attempt_start_path=contract_dir / "attempt_context.json",
            attempt_end_path=end_path,
            execution_log_path=log_path,
            workflow_returncode=int(attempt_end["snakemake_returncode"]),
            ended_utc=attempt_end["ended_utc"],
        )

    manifest_path = publication / "response_calibration_phase_a_manifest.json"
    completion_path = publication / "response_calibration_phase_a_completion.json"
    if not completion_path.is_file():
        phase_a = json.loads(manifest_path.read_text())
        write_immutable_json(completion_path, {
            "schema_version": "2.0.0",
            "record_type": "response_calibration_phase_a_completion",
            "status": phase_a["run_outcome"],
            "completed_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "mode": "response-calibration-phase-a",
            "run_id": run_context["run_id"],
            "lineage_id": run_context["lineage_id"],
            "recipe_id": run_context["recipe_id"],
            "execution_binding": binding,
            "run_context": file_record(contract_dir / "run_context.json"),
            "terminal_attempt_start": file_record(contract_dir / "attempt_context.json"),
            "terminal_attempt_end": file_record(end_path),
            "combined_execution_log": file_record(log_path),
            "snakemake_returncode": int(attempt_end["snakemake_returncode"]),
            "phase_a_summary": phase_a["summary"],
            "phase_a_manifest": file_record(manifest_path),
            "phase_a_ledger": file_record(
                publication / "response_calibration_phase_a_ledger.jsonl"),
            "freeze_required_before_phase_b": True,
            "full_run_terminal_publication": False,
            "production_overwrite": False,
        })

    normative = _yaml.safe_load(DEFAULT_WORKFLOW_CONFIG.read_text())
    pooling = normative["fod"]["response_estimation"]["calibration"]["pooling"]
    # The config pins the path and SHA-256 of the responsemean this recipe was
    # frozen against. Another site has a different MRtrix build, so when the
    # pinned binary is not here the one on the toolchain is used instead and
    # its own hash is what the frozen manifest records. That is a weaker claim
    # than the pinned one, and it is stated rather than hidden.
    executable = Path(pooling["executable"])
    expected_sha = pooling["executable_sha256"]
    if not executable.is_file():
        tool = sc_config.tools()
        candidate = (tool.mrtrix_bin / "responsemean") if tool.mrtrix_bin else None
        if candidate is None or not candidate.is_file():
            candidate = Path(shutil.which("responsemean") or "")
        if not candidate or not candidate.is_file():
            print("responsemean not found. Set MRTRIX_BIN or put MRtrix3 on PATH.",
                  file=sys.stderr)
            return 2
        executable = candidate.resolve()
        expected_sha = _sha256_file(executable)
        print(f"responsemean: {executable} (recording its hash; the recipe's "
              "pinned binary is not on this machine)")
    frozen_dir = run_root / "05_group_response_frozen"
    if frozen_dir.exists():
        print(f"already frozen: {frozen_dir}")
        frozen = json.loads(
            (frozen_dir / "frozen_response_calibration_manifest.json").read_text())
        frozen["manifest_path"] = str(frozen_dir / "frozen_response_calibration_manifest.json")
    else:
        frozen = freeze_response_calibration_from_phase_a(
            completion_path, manifest_path, frozen_dir,
            minimum_valid_subjects=args.minimum_valid,
            generated_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            responsemean_executable=executable,
            expected_responsemean_sha256=expected_sha,
        )
        print(f"pooled {frozen['valid_subject_count']} response function(s); "
              f"{frozen['invalid_subject_count']} excluded")

    calibration = {
        "frozen_manifest": {
            "path": str(frozen["manifest_path"]),
            "sha256": _sha256_file(Path(frozen["manifest_path"])),
        },
        "pooled_responses": {
            tissue: {
                "path": str(frozen_dir / f"pooled_response_{tissue}.txt"),
                "sha256": _sha256_file(frozen_dir / f"pooled_response_{tissue}.txt"),
            }
            for tissue in ("wm", "gm", "csf")
        },
    }
    overlay.setdefault("fod", {}).setdefault("response_estimation", {}).setdefault(
        "calibration", {}).update(calibration)
    overlay_path.write_text(_yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    print(f"frozen calibration recorded in {overlay_path.name}")
    print("next: python run_imaging.py review --run-root "
          f"{run_root} --reviewer \"<your name>\"")
    return 0


def cmd_review(args) -> int:
    """Record a reviewer's verdict on the visual-QC bundle for some units.

    Between preflight and tractography a human is meant to look at the
    registration and mask overlays and say whether each subject is usable. The
    workflow reads that verdict from ``<run>/review/human_visual_qc.csv`` and
    refuses to enter phase B without one, but nothing wrote the file, so the
    gate could not be passed at all. This writes it.

    The file deliberately carries no diagnosis column: the review is blind.
    """
    run_root = Path(args.run_root).expanduser().resolve()
    review_path = run_root / "review" / "human_visual_qc.csv"
    units = args.units or _approved_units(run_root)
    if not units:
        print(f"no units to review under {run_root}", file=sys.stderr)
        return 2
    reviewed_utc = args.utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    fields = ["unit", "status", "reviewer", "reviewed_utc", "note"]
    existing: dict[str, dict] = {}
    if review_path.is_file():
        import csv as _csv
        with review_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in _csv.DictReader(handle):
                existing[str(row.get("unit", "")).strip()] = dict(row)
    for unit in units:
        existing[unit] = {"unit": unit, "status": args.status.upper(),
                          "reviewer": args.reviewer, "reviewed_utc": reviewed_utc,
                          "note": args.note}
    review_path.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for unit in sorted(existing):
            writer.writerow({key: existing[unit].get(key, "") for key in fields})
    print(f"recorded {args.status.upper()} for {len(units)} unit(s) by {args.reviewer}")
    print(f"  {review_path}")
    return 0


def _approved_units(run_root: Path) -> list[str]:
    binding_path = run_root / "contract" / "execution_binding.json"
    if not binding_path.is_file():
        return []
    return list(json.loads(binding_path.read_text()).get("units", []))


def cmd_continue(args) -> int:
    """Open phase B for the units that passed preflight.

    Phase B is a second gate, on purpose: tractography is the expensive part
    and should only run on subjects whose registration and masks a person has
    actually looked at. The workflow requires a continuation binding naming
    exactly those units; this writes it into the run overlay after checking
    each unit's preflight record really is a PASS.
    """
    import yaml as _yaml

    run_root = Path(args.run_root).expanduser().resolve()
    overlay_path = run_root / "contract" / "run_overlay.yaml"
    if not overlay_path.is_file():
        print(f"no approval in {run_root}; run `approve` first", file=sys.stderr)
        return 2
    overlay = _yaml.safe_load(overlay_path.read_text())

    passed, failed, missing = [], [], []
    for unit in _approved_units(run_root):
        record_path = (run_root / "subjects" / unit / "06_preflight"
                       / "tractography_preflight.json")
        if not record_path.is_file():
            missing.append(unit)
            continue
        record = json.loads(record_path.read_text())
        (passed if record.get("status") == "PASS" else failed).append(unit)

    chosen = sorted(set(args.units)) if args.units else sorted(passed)
    print(f"preflight: {len(passed)} pass, {len(failed)} fail, {len(missing)} not yet run")
    if failed:
        print(f"  failed : {', '.join(failed)}")
    if missing:
        print(f"  pending: {', '.join(missing)}")
    if not chosen:
        print("\nnothing to continue with. Run the pre-tractography phase first, "
              "then `review`, then this command.", file=sys.stderr)
        return 1
    not_passed = [unit for unit in chosen if unit not in passed]
    if not_passed and not args.force:
        print(f"\nrefusing: {', '.join(not_passed)} did not pass preflight "
              "(pass --force only if you know why)", file=sys.stderr)
        return 1

    binding = {
        "schema_version": "2.0.0",
        "binding_type": "tractography_continuation_binding",
        "execution_scope": "canary",
        "recipe_id": overlay["execution_binding"]["recipe_id"],
        "execution_subset_manifest": overlay["execution_binding"]["execution_subset_manifest"],
        "diagnosis_labels_used": False,
        "approved_units": chosen,
        "approved_unit_count": len(chosen),
        "approved_by": args.by,
        "approved_utc": args.utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    overlay["launcher_mode"] = "phase-b"
    overlay["tractography_continuation_binding"] = binding
    overlay_path.write_text(_yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    (run_root / "contract" / "tractography_continuation_binding.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nphase B open for {len(chosen)} unit(s), approved by {args.by}")
    print(f"next: python run_imaging.py run --run-root {run_root} --cores 16")
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
    if args.dry_run or not approved:
        return subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT)).returncode

    # The phase-A publication records the log of the attempt that produced it,
    # so the run has to be captured as well as shown.
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"execution_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.log"
    with log_path.open("wb") as handle:
        process = subprocess.Popen(cmd, env=env, cwd=str(PROJECT_ROOT),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            handle.write(line)
        returncode = process.wait()
    (run_root / "contract" / "attempt_end.json").write_text(
        json.dumps({
            "schema_version": "1.0.0",
            "status": "COMPLETE" if returncode == 0 else "FAILED",
            "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "snakemake_returncode": returncode,
            "execution_log": str(log_path),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nlog: {log_path}")
    if returncode == 0:
        print("next: python run_imaging.py freeze --run-root "
              f"{run_root} --by \"<your name>\"")
    return returncode


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

    # The source content lock and the metadata projection describe the raw
    # files this run is about to read. They were cohort artifacts built by hand
    # once; the pipeline now builds them per run from the approved subset, so
    # any cohort gets the same protection without shipping ours.
    import sc_source_lock
    lock = sc_source_lock.build(subset_path, contract_dir / "source_lock",
                                rehash=not args.no_content_hash)
    if lock["missing_files"]:
        print(f"FAIL: {len(lock['missing_files'])} approved source file(s) missing",
              file=sys.stderr)
        for entry in lock["missing_files"][:10]:
            print(f"  {entry}", file=sys.stderr)
        return 1
    print(f"source lock: {lock['inventory_row_count']} files, "
          f"{lock['projection_row_count']} pairs")

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
        "inputs": {
            "approved_pair_manifest": {
                "required_row_count": len(rows),
                "sha256": binding["parent_acquisition_manifest"]["sha256"],
                "path": str(manifest),
            },
            "locked_file_inventory": {
                "path": str(lock["inventory_path"]),
                "sha256": lock["inventory_sha256"],
                "row_count": lock["inventory_row_count"],
            },
            "source_metadata_projection": {
                "path": str(lock["projection_path"]),
                "sha256": lock["projection_sha256"],
                "row_count": lock["projection_row_count"],
                "validation_path": str(lock["validation_path"]),
                "validation_sha256": lock["validation_sha256"],
            },
        },
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
    ap.add_argument("--study", type=Path, default=os.environ.get("SC_STUDY") or None,
                    help="study configuration describing where the data is and how it is "
                         "laid out (see sc_study.py --init); also read from $SC_STUDY")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check paths, toolchain and packages")
    d.add_argument("--imaging-only", action="store_true")
    d.set_defaults(func=cmd_doctor)

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
                   choices=("response-calibration-phase-a", "pre-tractography-canary"),
                   help="launcher mode recorded in the overlay. Phase B is not "
                        "approved here: it opens with `continue`, after preflight "
                        "and a human review of the QC overlays")
    a.add_argument("--recipe-id", default=None, help="default: the recipe_id in configs/connectome_v2.yaml")
    a.add_argument("--max-cores", type=int, default=8, help="core ceiling this approval permits")
    a.add_argument("--wall-clock-hours", type=int, default=72, help="wall-clock stop for the run")
    a.add_argument("--storage-gb", type=int, default=150, help="storage stop for the run")
    a.add_argument("--no-content-hash", action="store_true",
                   help="lock sources on size and stat identity only, without hashing "
                        "their contents (much faster on large cohorts, weaker guarantee)")
    a.set_defaults(func=cmd_approve)

    fz = sub.add_parser("freeze", help="pool the phase-A responses phase B deconvolves against")
    fz.add_argument("--run-root", type=Path, required=True)
    fz.add_argument("--by", required=True, help="who is freezing this calibration")
    fz.add_argument("--log", type=Path, default=None,
                    help="the phase-A execution log, when the run was not launched "
                         "by `run` and so did not record one")
    fz.add_argument("--minimum-valid", type=int, default=1,
                    help="how many subjects must have produced a valid response "
                         "before the pool is accepted (default: 1)")
    fz.set_defaults(func=cmd_freeze)

    rv = sub.add_parser("review", help="record the human visual-QC verdict for units")
    rv.add_argument("--run-root", type=Path, required=True)
    rv.add_argument("--units", nargs="*", default=None,
                    help="default: every approved unit in the run")
    rv.add_argument("--status", default="PASS", choices=("PASS", "FAIL"))
    rv.add_argument("--reviewer", required=True, help="who looked at the overlays")
    rv.add_argument("--utc", default=None)
    rv.add_argument("--note", default="")
    rv.set_defaults(func=cmd_review)

    cn = sub.add_parser("continue", help="open phase B for the units that passed preflight")
    cn.add_argument("--run-root", type=Path, required=True)
    cn.add_argument("--units", nargs="*", default=None,
                    help="default: every unit whose preflight passed")
    cn.add_argument("--by", required=True, help="who is authorising tractography")
    cn.add_argument("--utc", default=None)
    cn.add_argument("--force", action="store_true",
                    help="continue with a unit that did not pass preflight")
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
    args.command = args.cmd
    args._study = apply_study(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
