#!/usr/bin/env python
"""Run the analysis + ML pipeline as a dependency-ordered DAG.

    python -m connectome_analysis.run_analysis --list
    python -m connectome_analysis.run_analysis --all
    python -m connectome_analysis.run_analysis --stage build_shap_overall --with-deps
    python -m connectome_analysis.run_analysis --from edr_exceptions --resume

This replaces ``apps/connectome_dashboard/refresh_connectome_dashboard_data.py``,
which was a straight-line script living inside the dashboard app -- so the
analysis could not be run without the dashboard, and the order of its 23 calls
was load-bearing but undeclared. The stage graph now lives in ``stages.py``;
this module only executes it.

Three behaviours are deliberate:

*Inputs are checked before a stage runs.* Several analysis functions guard their
reads with ``if path.exists()`` and degrade silently when an input is missing,
which is how the mediation stage came to run without its brain-age mediator.
Here a missing declared input is an error.

*Ordering comes from the graph, not the file.* ``--stage`` and ``--from`` are
topologically sorted, so a partial run cannot invert a dependency.

*Resume is based on the files on disk*, not on a checkpoint that can disagree
with them.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from connectome_analysis import stages as _stages  # noqa: E402
from connectome_analysis.stages import Stage  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "analysis.yaml"


# ---------------------------------------------------------------- config --
def load_config(path: Path | None) -> dict[str, Any]:
    """Load the parameter surface. Absent file means documented defaults."""
    import yaml

    cfg: dict[str, Any] = {}
    target = path or DEFAULT_CONFIG
    if target.is_file():
        cfg = yaml.safe_load(target.read_text()) or {}
    elif path is not None:
        raise SystemExit(f"config not found: {path}")
    return cfg


def param(cfg: dict[str, Any], stage: str, key: str, default: Any) -> Any:
    """Stage-specific value wins over a global one, which wins over the default."""
    per_stage = (cfg.get("stages") or {}).get(stage) or {}
    if key in per_stage:
        return per_stage[key]
    return (cfg.get("defaults") or {}).get(key, default)


# ------------------------------------------------------------- execution --
class Runner:
    def __init__(self, paths: Any, cfg: dict[str, Any], *, resume: bool,
                 dry_run: bool, python: str) -> None:
        self.paths = paths
        self.cfg = cfg
        self.resume = resume
        self.dry_run = dry_run
        self.python = python
        self.values: dict[str, Any] = {}     # in-memory results between stages
        # Stages earlier in this run. In a dry run nothing is written, so their
        # outputs count as available rather than reporting every later stage
        # as blocked.
        self.planned: set[str] = set()
        self.rows: list[dict[str, Any]] = []
        self.root = Path(paths.output_root)

    # -- resume ------------------------------------------------------------
    def outputs_present(self, stage: Stage) -> bool:
        return bool(stage.produces) and all((self.root / p).is_file() for p in stage.produces)

    def is_stale(self, stage: Stage) -> bool:
        """True when any declared input is newer than the oldest declared output."""
        if not stage.produces:
            return True
        try:
            newest_in = 0.0
            for dep in stage.needs:
                for p in _stages.by_name(dep).produces:
                    f = self.root / p
                    if f.is_file():
                        newest_in = max(newest_in, f.stat().st_mtime)
            oldest_out = min((self.root / p).stat().st_mtime for p in stage.produces)
            return newest_in > oldest_out
        except (OSError, KeyError):
            return True

    # -- input contract ----------------------------------------------------
    def check_inputs(self, stage: Stage) -> list[str]:
        missing = []
        for dep in stage.needs:
            if self.dry_run and dep in self.planned:
                continue
            for p in _stages.by_name(dep).produces:
                if not (self.root / p).is_file():
                    missing.append(f"{p}  (from stage {dep})")
        return missing

    # -- run one -----------------------------------------------------------
    def run_stage(self, stage: Stage) -> str:
        started = time.time()

        if self.resume and self.outputs_present(stage) and not self.is_stale(stage):
            self._record(stage, "skipped", started, reason="outputs present and current")
            print(f"  [skip]  {stage.name}")
            return "skipped"

        missing = self.check_inputs(stage)
        if missing:
            self._record(stage, "blocked", started, missing=len(missing))
            print(f"  [BLOCK] {stage.name}: {len(missing)} declared input(s) missing", file=sys.stderr)
            for m in missing[:4]:
                print(f"            {m}", file=sys.stderr)
            return "blocked"

        if self.dry_run:
            print(f"  [dry]   {stage.name}  -> {len(stage.produces)} output(s)")
            self._record(stage, "dry-run", started)
            self.planned.add(stage.name)
            return "dry-run"

        print(f"  [run]   {stage.name} ...", flush=True)
        try:
            if stage.script:
                self._run_script(stage)
            else:
                result = CALLS[stage.name](self)
                if stage.provides_value:
                    self.values[stage.name] = result
        except Exception as exc:  # a failed stage must not abort the whole run
            self._record(stage, "failed", started, error=f"{type(exc).__name__}: {exc}")
            print(f"  [FAIL]  {stage.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc(limit=3, file=sys.stderr)
            return "failed"

        produced = sum(1 for p in stage.produces if (self.root / p).is_file())
        if stage.produces and produced < len(stage.produces):
            self._record(stage, "incomplete", started, produced=produced,
                         expected=len(stage.produces))
            print(f"  [WARN]  {stage.name}: wrote {produced}/{len(stage.produces)} declared outputs",
                  file=sys.stderr)
            return "incomplete"

        self._record(stage, "ok", started, produced=produced)
        print(f"  [ok]    {stage.name}  ({time.time() - started:.1f}s)")
        return "ok"

    def _run_script(self, stage: Stage) -> None:
        script = PROJECT_ROOT / "hcp_analysis" / stage.script
        if not script.is_file():
            raise FileNotFoundError(f"build script not found: {script}")
        cmd = [self.python, str(script),
               "--analysis-root", str(self.root),
               "--figs", str(self.cfg.get("figs_dir") or (PROJECT_ROOT / "docs" / "figs"))]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(PROJECT_ROOT), str(PROJECT_ROOT / "hcp_analysis"), env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        proc = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT),
                              capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
            raise RuntimeError(f"{stage.script} exited {proc.returncode}\n" + "\n".join(tail))

    def _record(self, stage: Stage, status: str, started: float, **extra: Any) -> None:
        self.rows.append({
            "stage": stage.name, "group": stage.group, "status": status,
            "seconds": round(time.time() - started, 2),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **extra,
        })

    # -- value dependencies ------------------------------------------------
    def value(self, name: str) -> Any:
        """Fetch a prior stage's in-memory result, recomputing it if this run
        did not produce it (which is the normal case for --resume and --from)."""
        if name not in self.values:
            print(f"          (recomputing {name} for its in-memory result)", flush=True)
            self.values[name] = CALLS[name](self)
        return self.values[name]


# ------------------------------------------------- the stage call table --
# Imports stay inside the calls so that --list works without the heavy optional
# ML dependencies installed.
def _c_analysis_snapshot(r: Runner):
    import pandas as pd
    from connectome_analysis.analysis_cohort import load_master_cohort  # noqa: F401
    mode = param(r.cfg, "analysis_snapshot", "snapshot_mode", "provisional")
    snap = {"snapshot_mode": mode, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = r.paths.master_dir / "analysis_snapshot.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([snap]).to_csv(out, index=False)
    return snap


def _c_master_cohort(r: Runner):
    from connectome_analysis.analysis_cohort import load_master_cohort
    return load_master_cohort(r.paths, rebuild=True)


def _c_qc_snapshot(r: Runner):
    from connectome_analysis.analysis_cohort import qc_snapshot
    return qc_snapshot(r.paths, r.value("master_cohort"))


def _c_data_completeness(r: Runner):
    from connectome_analysis.analysis_cohort import run_data_completeness
    strict = param(r.cfg, "data_completeness", "strict_tracks_count", False)
    return run_data_completeness(r.paths, r.value("master_cohort"), strict_tracks_count=strict)


def _c_demographics(r: Runner):
    from connectome_analysis.analysis_cohort import run_demographics_overview
    return run_demographics_overview(r.paths, r.value("master_cohort"))


def _c_connectome_qc(r: Runner):
    from connectome_analysis.analysis_graph import run_connectome_qc
    return run_connectome_qc(r.paths, r.value("master_cohort"))


def _c_functional_placeholder(r: Runner):
    import pandas as pd
    d = r.paths.section_dir("19", "network_analysis") / "functional"
    d.mkdir(parents=True, exist_ok=True)
    note = d / "README_functional.txt"
    note.write_text(
        "Functional-network tables are derived from an approximate AAL3 -> Yeo\n"
        "mapping. This cohort has no fMRI; nothing here is a functional measurement.\n"
    )
    return {"path": str(note)}


def _live(fn_name: str, *value_deps: str) -> Callable[[Runner], Any]:
    def call(r: Runner):
        import connectome_analysis.analysis_live_connectome as m
        fn = getattr(m, fn_name)
        return fn(r.paths, r.value("master_cohort"), *[r.value(d) for d in value_deps])
    return call


def _c_network_analysis(r: Runner):
    from connectome_analysis.analysis_network import run_network_analysis
    return run_network_analysis(r.paths, r.value("master_cohort"))


def _c_findings_catalog(r: Runner):
    from connectome_analysis.analysis_findings import build_findings_catalog
    return build_findings_catalog(r.paths, r.value("master_cohort"))


def _c_mediation(r: Runner):
    from connectome_analysis.analysis_mediation import run_mediation
    return run_mediation(r.paths, r.value("master_cohort"))


def _c_edgewise(r: Runner):
    from connectome_analysis.analysis_edges import run_edgewise_inference
    n = int(param(r.cfg, "edgewise_fd_sum", "edge_perms", 200))
    return run_edgewise_inference(r.paths, r.value("master_cohort"), n_perm=n)


def _c_brain_age(r: Runner):
    from connectome_analysis.analysis_live_connectome import run_live_brain_age
    n = int(param(r.cfg, "live_brain_age", "brain_age_repeats", 3))
    return run_live_brain_age(r.paths, r.value("master_cohort"), n_repeats=n)


def _c_lr_sr(r: Runner):
    from connectome_analysis.analysis_length_delay import run_lr_sr_analysis
    return run_lr_sr_analysis(r.paths, r.value("master_cohort"))


def _c_edr_exceptions(r: Runner):
    from connectome_analysis.analysis_edr_exceptions import run_edr_exception_analysis
    return run_edr_exception_analysis(r.paths, r.value("master_cohort"))


def _c_delay(r: Runner):
    from connectome_analysis.analysis_length_delay import run_delay_analysis
    return run_delay_analysis(r.paths, r.value("master_cohort"))


def _c_clinical(r: Runner):
    from connectome_analysis.analysis_clinical import run_clinical_covariate_review
    return run_clinical_covariate_review(r.paths)


def _c_advanced(r: Runner):
    from connectome_analysis.analysis_advanced import run_structural_innovation_analysis
    return run_structural_innovation_analysis(r.paths, r.value("master_cohort"))


def _c_ml_diagnostics(r: Runner):
    from connectome_analysis.analysis_ml_diagnostics import run_ml_diagnostics
    return run_ml_diagnostics(r.paths, r.value("master_cohort"))


def _c_outcome_search(r: Runner):
    from connectome_analysis.analysis_clinical_outcome_search import run_search
    fast = bool(param(r.cfg, "clinical_outcome_search", "outcome_search_fast", True))
    jobs = int(param(r.cfg, "clinical_outcome_search", "outcome_search_jobs", 4))
    return run_search(str(r.paths.deriv_root), fast=fast, n_jobs=jobs)


def _c_literature(r: Runner):
    from connectome_analysis.literature_openalex import run_literature
    mailto = param(r.cfg, "literature_retrieval", "literature_mailto", "")
    if not mailto:
        raise ValueError("literature_retrieval needs a contact address "
                         "(stages.literature_retrieval.literature_mailto in the config)")
    d = r.paths.section_dir("20", "findings")
    return run_literature(d / "findings_catalog.csv", d, mailto)


CALLS: dict[str, Callable[[Runner], Any]] = {
    "analysis_snapshot": _c_analysis_snapshot,
    "master_cohort": _c_master_cohort,
    "qc_snapshot": _c_qc_snapshot,
    "data_completeness": _c_data_completeness,
    "demographics": _c_demographics,
    "connectome_qc": _c_connectome_qc,
    "functional_placeholder": _c_functional_placeholder,
    "live_global_microstructure": _live("run_live_global_microstructure"),
    "live_node_microstructure": _live("run_live_nodewise_microstructure"),
    "live_multimetric": _live("run_live_multimetric_overview",
                              "live_global_microstructure", "live_node_microstructure"),
    "live_global_graph": _live("run_live_global_graph_metrics"),
    "live_node_graph": _live("run_live_nodewise_graph_metrics"),
    "live_coupling": _live("run_live_structure_microstructure_coupling",
                           "live_node_graph", "live_node_microstructure"),
    "network_analysis": _c_network_analysis,
    "edgewise_fd_sum": _c_edgewise,
    "live_brain_age": _c_brain_age,
    "lr_sr": _c_lr_sr,
    "edr_exceptions": _c_edr_exceptions,
    "delay": _c_delay,
    "clinical": _c_clinical,
    "advanced_structural": _c_advanced,
    "findings_catalog": _c_findings_catalog,
    "mediation": _c_mediation,
    "literature_retrieval": _c_literature,
    "ml_diagnostics": _c_ml_diagnostics,
    "clinical_outcome_search": _c_outcome_search,
}


# --------------------------------------------------------------------- CLI --
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="run_analysis",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sel = ap.add_argument_group("stage selection")
    sel.add_argument("--all", action="store_true", help="run every stage")
    sel.add_argument("--stage", action="append", default=[], metavar="NAME",
                     help="run this stage (repeatable)")
    sel.add_argument("--group", action="append", default=[], metavar="GROUP",
                     choices=list(_stages.GROUPS), help="run every stage in a group")
    sel.add_argument("--from", dest="from_stage", metavar="NAME",
                     help="run this stage and everything downstream of it")
    sel.add_argument("--with-deps", action="store_true",
                     help="also run whatever the selection depends on")
    sel.add_argument("--skip", action="append", default=[], metavar="NAME")

    ap.add_argument("--config", type=Path, default=None,
                    help=f"parameter YAML (default: {DEFAULT_CONFIG})")
    ap.add_argument("--resume", action="store_true",
                    help="skip stages whose declared outputs exist and are current")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--list", action="store_true", help="print the DAG and exit")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the build_*.py subprocess stages")
    loc = ap.add_argument_group(
        "data locations",
        "Each flag sets the matching SC_* environment variable for this run, so the\n"
        "in-process stages and the build_*.py subprocesses resolve the same tree.\n"
        "Setting the variables yourself is equivalent.")
    loc.add_argument("--study", type=Path, default=os.environ.get("SC_STUDY") or None,
                     help="study configuration, the same file the imaging side takes "
                          "(see sc_study.py --init); any flag below overrides it")
    loc.add_argument("--deriv-root", type=Path, default=None,
                     help="derivatives root holding connectomes/ and qc/ (SC_DERIV_ROOT)")
    loc.add_argument("--connectomes-dir", type=Path, default=None,
                     help="connectome matrices (SC_CONNECTOMES_DIR)")
    loc.add_argument("--analysis-root", type=Path, default=None,
                     help="where results are written (SC_ANALYSIS_ROOT)")
    loc.add_argument("--cohort-dir", type=Path, default=None,
                     help="folder with dti.csv, mri.csv, dti_master.csv, mri_master.csv (SC_COHORT_DIR)")
    loc.add_argument("--exclusions", type=Path, default=None,
                     help="subject exclusions YAML (SC_EXCLUSIONS)")
    # Kept for the dashboard shim. They name files, but the build scripts read a
    # folder, so they are converted to --cohort-dir.
    loc.add_argument("--cohort-dti-csv", type=Path, default=None, help=argparse.SUPPRESS)
    loc.add_argument("--cohort-mri-csv", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--lock-timeout-sec", type=int, default=5)
    return ap.parse_args(argv)


def print_dag() -> int:
    order = _stages.topo_order()
    width = max(len(s.name) for s in order)
    group = None
    for i, s in enumerate(order):
        if s.group != group:
            group = s.group
            print(f"\n--- {group} " + "-" * (58 - len(group)))
        flags = []
        if s.script:
            flags.append("script")
        if s.optional:
            flags.append("optional")
        tag = f"  [{','.join(flags)}]" if flags else ""
        needs = ", ".join(s.needs) if s.needs else "-"
        print(f"  {i:>2}. {s.name:<{width}}{tag}")
        print(f"      {s.summary}")
        print(f"      needs: {needs}")
        if s.produces:
            print(f"      makes: {len(s.produces)} file(s), e.g. {s.produces[0]}")
    print(f"\n{len(order)} stages.")
    return 0


def select(args: argparse.Namespace) -> list[str]:
    chosen: set[str] = set()
    if args.all:
        chosen |= {s.name for s in _stages.STAGES if not s.optional}
    for g in args.group:
        chosen |= {s.name for s in _stages.STAGES if s.group == g}
    chosen |= set(args.stage)
    if args.from_stage:
        order = [s.name for s in _stages.topo_order()]
        start = order.index(_stages.by_name(args.from_stage).name)
        chosen |= set(order[start:])
    if not chosen:
        raise SystemExit("nothing selected. Use --all, --group, --stage or --from "
                         "(--list shows the DAG).")
    if args.with_deps:
        chosen = set(_stages.with_dependencies(sorted(chosen)))
    chosen -= set(args.skip)
    return sorted(chosen)


def apply_location_flags(args: argparse.Namespace) -> None:
    """Turn --study and --deriv-root into SC_* variables, then reset sc_config.

    A study file describes where one study's data and outputs live, and the
    imaging side already takes it. Reading the same file here means an analysis
    does not have to be told a second time where the connectomes are. An
    explicit flag still wins over it.
    """
    if getattr(args, "study", None):
        sys.path.insert(0, str(PROJECT_ROOT))
        import sc_study

        try:
            sc_study.load_study(args.study).apply_environment()
        except sc_study.StudyError as error:
            print(str(error), file=sys.stderr)
            raise SystemExit(2)
    mapping = {
        "SC_DERIV_ROOT": args.deriv_root,
        "SC_CONNECTOMES_DIR": args.connectomes_dir,
        "SC_ANALYSIS_ROOT": args.analysis_root,
        "SC_COHORT_DIR": args.cohort_dir,
        "SC_EXCLUSIONS": args.exclusions,
    }
    for legacy in (args.cohort_dti_csv, args.cohort_mri_csv):
        if legacy is not None and mapping["SC_COHORT_DIR"] is None:
            mapping["SC_COHORT_DIR"] = Path(legacy).resolve().parent
    for name, value in mapping.items():
        if value is not None:
            os.environ[name] = str(Path(value).expanduser().resolve())
    try:
        import sc_config
        sc_config.paths.cache_clear()
    except Exception:
        pass
    try:
        import sc_exclusions
        sc_exclusions.load.cache_clear()
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        return print_dag()

    cfg = load_config(args.config)
    selected = select(args)
    plan = _stages.topo_order(selected)

    apply_location_flags(args)

    from connectome_analysis.analysis_config import apply_plot_theme, get_analysis_paths
    apply_plot_theme()
    # No deriv_root is passed: every location comes from sc_config, which the
    # flags above have just set. One mechanism, one tree.
    # A dry run must not create or change anything on disk, so it neither
    # creates folders here nor takes the lock or writes the ledger below.
    paths = get_analysis_paths(notebook_dir=PROJECT_ROOT, ensure=not args.dry_run)

    import sc_config
    p = sc_config.paths()
    print(f"connectomes   : {p.connectomes_dir}")
    print(f"cohort tables : {p.cohort_dir}")
    try:
        import sc_exclusions
        excl = sc_exclusions.load()
        print(f"exclusions    : {len(excl)} subject(s) from {sc_exclusions.path()}"
              if excl else f"exclusions    : NONE ({sc_exclusions.path()} not found)")
        if not excl:
            print("                The published ML results exclude one subject. Without the\n"
                  "                exclusions file, the ML stages will not reproduce them.")
    except Exception:
        pass

    print(f"analysis root : {paths.output_root}")
    print(f"config        : {args.config or DEFAULT_CONFIG}"
          f"{'' if (args.config or DEFAULT_CONFIG).is_file() else '  (absent; using defaults)'}")
    print(f"plan          : {len(plan)} stage(s)"
          f"{' [resume]' if args.resume else ''}{' [dry-run]' if args.dry_run else ''}\n")

    runner = Runner(paths, cfg, resume=args.resume, dry_run=args.dry_run, python=args.python)
    started = time.time()

    if args.dry_run:
        statuses = [runner.run_stage(s) for s in plan]
        print(f"\ndry-run: {len(statuses)} stage(s) planned; nothing was written.")
        return 0

    lock_path = Path(paths.output_root) / "logs" / "run_analysis.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        deadline = time.time() + args.lock_timeout_sec
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() >= deadline:
                    print("another run holds the lock", file=sys.stderr)
                    return 75
                time.sleep(0.25)

        statuses = [runner.run_stage(s) for s in plan]

    import pandas as pd
    ledger = Path(paths.exports_dir) / "stage_status.csv"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runner.rows).to_csv(ledger, index=False)

    counts: dict[str, int] = {}
    for s in statuses:
        counts[s] = counts.get(s, 0) + 1
    print(f"\n{'  '.join(f'{k}={v}' for k, v in sorted(counts.items()))}"
          f"   total {time.time() - started:.1f}s")
    print(f"ledger: {ledger}")

    summary = {
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": len(plan), "counts": counts,
        "analysis_root": str(paths.output_root),
    }
    (Path(paths.master_dir) / "run_analysis_status.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    bad = counts.get("failed", 0) + counts.get("blocked", 0) + counts.get("incomplete", 0)
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
