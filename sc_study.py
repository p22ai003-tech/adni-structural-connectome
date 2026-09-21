#!/usr/bin/env python3
"""Read one YAML file that says where a study's data is and how it is laid out.

Why this exists
---------------
``sc_config`` resolves every path from environment variables, which is right
for a machine that has been set up once. A study is different: it is one
description that has to travel — where the raw images live, how the folders
are arranged, where outputs should go — and it should be a file the user edits
and keeps, not eleven exports typed into a shell.

The raw images can be somewhere this machine cannot read directly. The config
therefore names a *source*:

``local``
    a directory on this machine, or anything mounted onto it (NFS, SMB, an
    external disk). Nothing is copied.
``s3``
    an ``s3://bucket/prefix`` URI. The listed subjects are staged into
    ``stage_root`` before the run and everything downstream reads the staged
    copy, so the pipeline itself never talks to S3 and never needs credentials
    in a rule.

Anything else a site uses — a URL, an internal object store, a tape robot — is
served by staging it to a directory yourself and pointing ``local`` at it. That
is the whole contract: the pipeline needs a readable directory in one of the
supported layouts.

Usage
-----
    from sc_study import load_study
    study = load_study("configs/study.yaml")
    study.apply_environment()          # sets SC_* for sc_config
    raw_root = study.ensure_available() # stages from S3 if needed
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Study", "load_study", "write_template", "TEMPLATE"]

LAYOUTS = ("simple", "adni", "bids")
SOURCES = ("local", "s3")

TEMPLATE = """\
# Study configuration. Everything the pipeline needs to find your data.
# Copy this file, edit it, and pass it with --study to any of the runners.

study:
  name: my-study
  # How the raw image folders are arranged. See docs/INPUT_LAYOUTS.md.
  #   simple : <raw>/<subject>/{dwi,t1}/...
  #   bids   : <raw>/sub-<id>/[ses-<x>/]{dwi,anat}/...
  #   adni   : <raw>/{dti,mri}/<subject>/<protocol>/<date>/<image-id>/...
  layout: simple

input:
  # Where the raw images are.
  #   local : a directory on this machine, including anything mounted onto it
  #   s3    : an s3:// URI, staged to stage_root before the run
  source: local
  path: /path/to/your/raw
  # uri: s3://your-bucket/your-prefix       # required when source is s3
  # stage_root: /path/to/staging            # required when source is s3
  # subjects: [sub-001, sub-002]            # optional: restrict the study
  # participants: participants.csv          # optional: subject_id,diagnosis,age,sex

output:
  # Everything the pipeline writes lives under here.
  data_root: /path/to/your/work
  # Anything below is derived from data_root unless you set it.
  # deriv_root: /path/to/your/work/derivatives
  # qc_root: /path/to/your/work/derivatives/qc
  # analysis_root: /path/to/your/work/derivatives/qc/analysis_cohort
  # connectomes_dir: /path/to/your/work/derivatives/connectomes

pairing:
  # A DWI and a T1 further apart than this are still processed, but are
  # labelled so an analysis can hold them out.
  max_gap_days: 180
  sensitivity_days: 90
"""


class StudyError(RuntimeError):
    """The study configuration cannot be used as written."""


@dataclass
class Study:
    path: Path
    name: str
    layout: str
    source: str
    raw_path: Path | None
    uri: str | None
    stage_root: Path | None
    subjects: list[str] | None
    participants: Path | None
    data_root: Path
    overrides: dict[str, Path] = field(default_factory=dict)
    max_gap_days: float = 180.0
    sensitivity_days: float = 90.0
    raw: dict[str, Any] = field(default_factory=dict)

    # -- resolution -----------------------------------------------------------

    @property
    def staged_root(self) -> Path:
        """The directory the pipeline will read, once the data is available."""
        if self.source == "local":
            assert self.raw_path is not None
            return self.raw_path
        assert self.stage_root is not None
        return self.stage_root

    def environment(self) -> dict[str, str]:
        """The SC_* variables this study implies, for sc_config."""
        data_root = self.data_root
        env = {
            "SC_DATA_ROOT": str(data_root),
            "SC_RAW_IMAGES_ROOT": str(self.staged_root),
            "SC_DERIV_ROOT": str(self.overrides.get("deriv_root", data_root / "derivatives")),
        }
        deriv = Path(env["SC_DERIV_ROOT"])
        qc = self.overrides.get("qc_root", deriv / "qc")
        env["SC_QC_ROOT"] = str(qc)
        env["SC_ANALYSIS_ROOT"] = str(self.overrides.get("analysis_root", qc / "analysis_cohort"))
        env["SC_CONNECTOMES_DIR"] = str(
            self.overrides.get("connectomes_dir", deriv / "connectomes")
        )
        # A study that names its own participants table owns its cohort
        # directory. Without this the analysis falls back to whatever cohort/
        # happens to sit in the checkout -- which, for anyone who cloned this
        # repository, is a different study's subjects.
        if "cohort_dir" not in self.overrides and self.participants is not None:
            env["SC_COHORT_DIR"] = str(self.participants.parent)
        for key in ("cohort_dir", "atlas_root", "run_state_dir", "manifest"):
            if key in self.overrides:
                env[f"SC_{key.upper()}"] = str(self.overrides[key])
        env.setdefault("SC_STUDY", str(self.path))
        return env

    def apply_environment(self) -> dict[str, str]:
        """Export the study's paths, without overriding an explicit setting.

        A variable already set in the environment wins, so a user can point one
        run somewhere else without editing the file.
        """
        applied = {}
        for key, value in self.environment().items():
            if os.environ.get(key):
                continue
            os.environ[key] = value
            applied[key] = value
        try:  # the resolver caches; a new study must invalidate it
            import sc_config

            sc_config.paths.cache_clear()
        except Exception:
            pass
        return applied

    # -- availability ---------------------------------------------------------

    def ensure_available(self, *, verbose: bool = True) -> Path:
        """Return a readable raw root, staging from S3 first if necessary."""
        if self.source == "local":
            root = self.staged_root
            if not root.is_dir():
                raise StudyError(
                    f"input.path does not exist or is not a directory: {root}\n"
                    "If it is a network or external volume, mount it first; the "
                    "pipeline reads it directly and does not copy it."
                )
            return root
        return self._stage_from_s3(verbose=verbose)

    def _stage_from_s3(self, *, verbose: bool) -> Path:
        assert self.uri and self.stage_root
        destination = self.stage_root
        destination.mkdir(parents=True, exist_ok=True)
        aws = shutil.which("aws")
        if aws is None:
            raise StudyError(
                "input.source is s3 but the AWS CLI is not installed.\n"
                "Install it (pip install awscli, or your platform's package) and "
                "configure credentials, or stage the data yourself and switch to "
                "input.source: local."
            )
        targets = self.subjects or [""]
        for subject in targets:
            source = self.uri.rstrip("/") + (f"/{subject}" if subject else "")
            target = destination / subject if subject else destination
            command = [aws, "s3", "sync", source, str(target)]
            if verbose:
                print(f"staging {source} -> {target}", flush=True)
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise StudyError(
                    f"aws s3 sync failed for {source} (exit {result.returncode}). "
                    "Check the URI and that this machine has read access."
                )
        return destination


def _as_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def load_study(path: str | Path) -> Study:
    """Read and validate a study configuration."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise StudyError(
            f"study configuration not found: {config_path}\n"
            "Create one with: python sc_study.py --init configs/study.yaml"
        )
    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    study_block = document.get("study") or {}
    input_block = document.get("input") or {}
    output_block = document.get("output") or {}
    pairing_block = document.get("pairing") or {}

    problems: list[str] = []
    layout = str(study_block.get("layout", "simple"))
    if layout not in LAYOUTS:
        problems.append(f"study.layout must be one of {', '.join(LAYOUTS)} (got {layout!r})")
    source = str(input_block.get("source", "local"))
    if source not in SOURCES:
        problems.append(f"input.source must be one of {', '.join(SOURCES)} (got {source!r})")

    raw_path = _as_path(input_block.get("path"))
    uri = input_block.get("uri")
    stage_root = _as_path(input_block.get("stage_root"))
    if source == "local" and raw_path is None:
        problems.append("input.path is required when input.source is local")
    if source == "s3":
        if not uri or not str(uri).startswith("s3://"):
            problems.append("input.uri must be an s3:// URI when input.source is s3")
        if stage_root is None:
            problems.append("input.stage_root is required when input.source is s3")

    data_root = _as_path(output_block.get("data_root"))
    if data_root is None:
        problems.append("output.data_root is required")

    if problems:
        raise StudyError(
            f"{config_path} cannot be used:\n  - " + "\n  - ".join(problems)
        )

    overrides = {
        key: _as_path(output_block[key])
        for key in (
            "deriv_root", "qc_root", "analysis_root", "connectomes_dir",
            "cohort_dir", "atlas_root", "run_state_dir", "manifest",
        )
        if output_block.get(key)
    }
    subjects = input_block.get("subjects")
    participants = _as_path(input_block.get("participants"))
    if participants is not None and not participants.is_absolute():
        participants = (config_path.parent / participants).resolve()

    assert data_root is not None
    return Study(
        path=config_path,
        name=str(study_block.get("name", config_path.stem)),
        layout=layout,
        source=source,
        raw_path=raw_path,
        uri=str(uri) if uri else None,
        stage_root=stage_root,
        subjects=[str(s) for s in subjects] if subjects else None,
        participants=participants,
        data_root=data_root,
        overrides=overrides,  # type: ignore[arg-type]
        max_gap_days=float(pairing_block.get("max_gap_days", 180)),
        sensitivity_days=float(pairing_block.get("sensitivity_days", 90)),
        raw=document,
    )


def write_template(path: str | Path, *, force: bool = False) -> Path:
    target = Path(path).expanduser()
    if target.exists() and not force:
        raise StudyError(f"refusing to overwrite {target} (pass --force)")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(TEMPLATE, encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect or create a study configuration")
    parser.add_argument("config", nargs="?", default="configs/study.yaml")
    parser.add_argument("--init", action="store_true", help="write a template and exit")
    parser.add_argument("--force", action="store_true", help="overwrite an existing template")
    parser.add_argument("--stage", action="store_true", help="stage the data now if it is on S3")
    args = parser.parse_args(argv)

    if args.init:
        target = write_template(args.config, force=args.force)
        print(f"wrote {target}")
        print("Edit it, then run: python run_imaging.py doctor --study " + str(target))
        return 0

    try:
        study = load_study(args.config)
    except StudyError as error:
        print(str(error), file=sys.stderr)
        return 2

    print(f"study    : {study.name}")
    print(f"layout   : {study.layout}")
    print(f"source   : {study.source}")
    print(f"raw      : {study.uri or study.raw_path}")
    if study.source == "s3":
        print(f"staged to: {study.stage_root}")
    if study.subjects:
        print(f"subjects : {len(study.subjects)} listed")
    if study.participants:
        mark = "" if study.participants.is_file() else "   <- not found"
        print(f"particip.: {study.participants}{mark}")
    print("environment:")
    for key, value in study.environment().items():
        print(f"  {key:<22} {value}")
    if args.stage:
        root = study.ensure_available()
        print(f"available: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
