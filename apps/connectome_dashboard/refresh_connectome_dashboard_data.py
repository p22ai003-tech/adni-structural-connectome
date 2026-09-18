#!/usr/bin/env python
"""Backwards-compatible entry point for the dashboard refresh.

The analysis pipeline no longer lives here. It is
``connectome_analysis/run_analysis.py``, which executes the stage graph in
``connectome_analysis/stages.py``. Keeping the pipeline inside the dashboard app
meant the analysis could not be run without the dashboard, and that the order of
its stages was implicit in the order of the calls.

This file remains because things depend on it: the refresh loop in
``run_connectome_dashboard_refresh_loop.sh``, a handoff validator in
``research_audit/`` and two deployment runbooks. It accepts the same arguments
it always did and forwards them, so nothing that calls it needs to change.

    --mode quick   the cohort and QC stages
    --mode full    every stage, including the build_*.py layer that previously
                   had to be run by hand and its outputs copied by hand

New work should call the runner directly, which can do things this interface
cannot -- select a single stage, resume, or print the graph:

    python -m connectome_analysis.run_analysis --list
    python -m connectome_analysis.run_analysis --all --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("quick", "full"), default="quick")
    p.add_argument("--snapshot-mode", choices=("provisional", "final"), default="provisional")
    p.add_argument("--deriv-root", type=Path, default=None)
    p.add_argument("--cohort-dti-csv", type=Path, default=None)
    p.add_argument("--cohort-mri-csv", type=Path, default=None)
    p.add_argument("--strict-tracks-count", action="store_true")
    p.add_argument("--edge-perms", type=int, default=None)
    p.add_argument("--brain-age-repeats", type=int, default=None)
    p.add_argument("--lock-timeout-sec", type=int, default=5)
    p.add_argument("--with-literature", action="store_true")
    p.add_argument("--literature-mailto", default=None)
    p.add_argument("--resume", action="store_true",
                   help="skip stages whose outputs are present and current")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from connectome_analysis import run_analysis

    args = parse_args(argv)
    forwarded: list[str] = ["--group", "cohort"] if args.mode == "quick" else ["--all"]
    if args.with_literature:
        forwarded += ["--stage", "literature_retrieval"]
    if args.resume:
        forwarded.append("--resume")
    for flag, value in (("--deriv-root", args.deriv_root),
                        ("--cohort-dti-csv", args.cohort_dti_csv),
                        ("--cohort-mri-csv", args.cohort_mri_csv)):
        if value is not None:
            forwarded += [flag, str(value)]
    forwarded += ["--lock-timeout-sec", str(args.lock_timeout_sec)]

    # Arguments that used to be CLI flags are configuration now. Anything passed
    # explicitly still wins, via a config overlay written for this run only.
    overrides: dict[str, dict[str, object]] = {}
    if args.snapshot_mode:
        overrides["analysis_snapshot"] = {"snapshot_mode": args.snapshot_mode}
    if args.strict_tracks_count:
        overrides["data_completeness"] = {"strict_tracks_count": True}
    if args.edge_perms is not None:
        overrides["edgewise_fd_sum"] = {"edge_perms": args.edge_perms}
    if args.brain_age_repeats is not None:
        overrides["live_brain_age"] = {"brain_age_repeats": args.brain_age_repeats}
    if args.literature_mailto:
        overrides["literature_retrieval"] = {"literature_mailto": args.literature_mailto}

    if overrides:
        import tempfile

        import yaml

        base = run_analysis.load_config(None)
        stages_cfg = dict(base.get("stages") or {})
        for stage, kv in overrides.items():
            stages_cfg[stage] = {**(stages_cfg.get(stage) or {}), **kv}
        base["stages"] = stages_cfg
        tmp = Path(tempfile.mkdtemp(prefix="sc_refresh_")) / "analysis.yaml"
        tmp.write_text(yaml.safe_dump(base, sort_keys=False))
        forwarded += ["--config", str(tmp)]

    print(f"[refresh] delegating to run_analysis: {' '.join(forwarded)}", flush=True)
    return run_analysis.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
