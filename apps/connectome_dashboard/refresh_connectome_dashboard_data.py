#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent.parent
DEFAULT_DERIV_ROOT = PROJECT_ROOT / "data" / "derivatives"
DEFAULT_COHORT_DTI = PROJECT_ROOT / "cohort" / "dti.csv"
DEFAULT_COHORT_MRI = PROJECT_ROOT / "cohort" / "mri.csv"
CONNECTOME_SUFFIXES = ("ALL", "count", "fd_sum", "fa_mean", "md_mean", "len_mean")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the generated CSV/PNG/markdown source tree used by connectome_app.py."
    )
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--snapshot-mode", choices=("provisional", "final"), default="provisional")
    parser.add_argument("--deriv-root", type=Path, default=DEFAULT_DERIV_ROOT)
    parser.add_argument("--cohort-dti-csv", type=Path, default=DEFAULT_COHORT_DTI)
    parser.add_argument("--cohort-mri-csv", type=Path, default=DEFAULT_COHORT_MRI)
    parser.add_argument("--strict-tracks-count", action="store_true")
    parser.add_argument("--edge-perms", type=int, default=200)
    parser.add_argument("--brain-age-repeats", type=int, default=3)
    parser.add_argument("--lock-timeout-sec", type=int, default=5)
    parser.add_argument("--with-literature", action="store_true",
                        help="also run OpenAlex literature retrieval for the findings catalog (makes network calls)")
    parser.add_argument("--literature-mailto", default="sabeesh90@gmail.com")
    return parser.parse_args()


def status_row(section: str, status: str, started: float, **extra: Any) -> dict[str, Any]:
    row = {
        "section": section,
        "status": status,
        "started_utc": datetime.fromtimestamp(started, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_utc": utc_now(),
        "duration_sec": round(time.time() - started, 3),
    }
    row.update(extra)
    return row


def run_section(name: str, rows: list[dict[str, Any]], fn: Callable[[], Any]) -> Any:
    started = time.time()
    try:
        result = fn()
    except FileNotFoundError as exc:
        rows.append(status_row(name, "skipped_missing_input", started, error=str(exc)))
        return None
    except ValueError as exc:
        rows.append(status_row(name, "skipped_value_error", started, error=str(exc)))
        return None
    except Exception as exc:
        rows.append(
            status_row(
                name,
                "error",
                started,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=8),
            )
        )
        return None
    rows.append(status_row(name, "ok", started))
    return result


def build_snapshot(paths: Any, snapshot_mode: str) -> pd.DataFrame:
    snapshot = {
        "snapshot_utc": utc_now(),
        "mode": snapshot_mode,
        "project_root": str(PROJECT_ROOT),
        "derivatives_root": str(paths.deriv_root),
        "connectomes_dir": str(paths.connectomes_dir),
        "analysis_output_root": str(paths.output_root),
    }
    for suffix in CONNECTOME_SUFFIXES:
        snapshot[f"n_{suffix}"] = len(
            [p for p in paths.connectomes_dir.glob(f"SC_AAL166_*_{suffix}.csv") if p.is_file() and not p.name.startswith("._")]
        )
    out = pd.DataFrame([snapshot])
    out.to_csv(paths.output_root / "analysis_snapshot.csv", index=False)
    return out


def write_functional_placeholder(paths: Any) -> dict[str, Any]:
    out_dir = paths.section_dir("16", "functional_placeholder")
    out_dir.mkdir(parents=True, exist_ok=True)
    requirements = pd.DataFrame(
        [
            {
                "status": "pending",
                "required_input": "subject-aligned FC, EEG, or MEG matrices",
                "reason": "Structural-functional coupling must remain disabled until real functional matrices are supplied.",
            }
        ]
    )
    requirements.to_csv(out_dir / "functional_coupling_requirements.csv", index=False)
    (out_dir / "functional_coupling_inference.md").write_text(
        "- Functional coupling is a placeholder only. No FC/EEG/MEG result is inferred from structural data alone.\n",
        encoding="utf-8",
    )
    return {"out_dir": out_dir}


def write_exports(paths: Any, section_rows: list[dict[str, Any]], tables: dict[str, pd.DataFrame]) -> None:
    from connectome_analysis.analysis_exports import collect_csv_exports, write_excel_bundle

    paths.exports_dir.mkdir(parents=True, exist_ok=True)
    section_status = pd.DataFrame(section_rows)
    section_status.to_csv(paths.exports_dir / "section_status.csv", index=False)
    export_tables = {
        name: df for name, df in tables.items() if isinstance(df, pd.DataFrame) and not df.empty
    }
    export_tables["section_status"] = section_status
    manifest = collect_csv_exports(paths.output_root)
    manifest.to_csv(paths.exports_dir / "export_manifest.csv", index=False)
    if not manifest.empty:
        export_tables["export_manifest"] = manifest
    suffix = "final" if "final" in str((paths.output_root / "analysis_snapshot.csv").read_text(errors="ignore")) else "provisional"
    write_excel_bundle(export_tables, paths.exports_dir / f"cohort_analysis_summary_{suffix}.xlsx")


def refresh(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(PROJECT_ROOT))
    from connectome_analysis.analysis_config import apply_plot_theme, get_analysis_paths
    from connectome_analysis.analysis_cohort import (
        load_master_cohort,
        qc_snapshot,
        run_data_completeness,
        run_demographics_overview,
    )
    from connectome_analysis.analysis_graph import run_connectome_qc

    apply_plot_theme()
    paths = get_analysis_paths(
        notebook_dir=PROJECT_ROOT,
        deriv_root=args.deriv_root,
        cohort_dti_csv=args.cohort_dti_csv,
        cohort_mri_csv=args.cohort_mri_csv,
    )
    paths.ensure()

    lock_path = paths.output_root / "logs" / "dashboard_refresh.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        deadline = time.time() + args.lock_timeout_sec
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() >= deadline:
                    print(f"Another refresh is running; lock timeout after {args.lock_timeout_sec}s", file=sys.stderr)
                    return 75
                time.sleep(0.25)

        rows: list[dict[str, Any]] = []
        tables: dict[str, pd.DataFrame] = {}
        started_all = time.time()

        snapshot = run_section("analysis_snapshot", rows, lambda: build_snapshot(paths, args.snapshot_mode))
        if snapshot is not None:
            tables["analysis_snapshot"] = snapshot

        master = run_section("master_cohort", rows, lambda: load_master_cohort(paths, rebuild=True))
        if master is None:
            pd.DataFrame(rows).to_csv(paths.exports_dir / "section_status.csv", index=False)
            return 1
        tables["master_cohort"] = master

        qc = run_section("qc_snapshot", rows, lambda: qc_snapshot(paths, master))
        if qc is not None:
            tables["qc_snapshot"] = qc

        completeness = run_section(
            "data_completeness",
            rows,
            lambda: run_data_completeness(paths, master, strict_tracks_count=args.strict_tracks_count),
        )
        if completeness:
            tables["qc_completeness"] = completeness.get("completeness", pd.DataFrame())

        demographics = run_section("demographics", rows, lambda: run_demographics_overview(paths, master))
        if demographics:
            tables["demographics"] = demographics.get("counts", pd.DataFrame())

        connectome_qc = run_section("connectome_qc", rows, lambda: run_connectome_qc(paths, master))
        if connectome_qc:
            tables["connectome_qc"] = connectome_qc.get("summary", pd.DataFrame())

        run_section("functional_placeholder", rows, lambda: write_functional_placeholder(paths))

        if args.mode == "full":
            from connectome_analysis.analysis_advanced import run_structural_innovation_analysis
            from connectome_analysis.analysis_clinical import run_clinical_covariate_review
            from connectome_analysis.analysis_edr_exceptions import run_edr_exception_analysis
            from connectome_analysis.analysis_edges import run_edgewise_inference
            from connectome_analysis.analysis_length_delay import run_delay_analysis, run_lr_sr_analysis
            from connectome_analysis.analysis_ml_diagnostics import run_ml_diagnostics
            from connectome_analysis.analysis_network import run_network_analysis
            from connectome_analysis.analysis_findings import build_findings_catalog
            from connectome_analysis.analysis_live_connectome import (
                run_live_brain_age,
                run_live_global_graph_metrics,
                run_live_global_microstructure,
                run_live_multimetric_overview,
                run_live_nodewise_graph_metrics,
                run_live_nodewise_microstructure,
                run_live_structure_microstructure_coupling,
            )

            global_micro = run_section("live_global_microstructure", rows, lambda: run_live_global_microstructure(paths, master))
            node_micro = run_section("live_node_microstructure", rows, lambda: run_live_nodewise_microstructure(paths, master))
            run_section("live_multimetric", rows, lambda: run_live_multimetric_overview(paths, master, global_micro, node_micro))
            run_section("live_global_graph", rows, lambda: run_live_global_graph_metrics(paths, master))
            node_graph = run_section("live_node_graph", rows, lambda: run_live_nodewise_graph_metrics(paths, master))
            run_section("live_coupling", rows, lambda: run_live_structure_microstructure_coupling(paths, master, node_graph, node_micro))
            run_section("network_analysis", rows, lambda: run_network_analysis(paths, master))
            run_section("findings_catalog", rows, lambda: build_findings_catalog(paths, master))
            from connectome_analysis.analysis_mediation import run_mediation
            run_section("mediation", rows, lambda: run_mediation(paths, master))
            if args.with_literature:
                from connectome_analysis.literature_openalex import run_literature
                run_section("literature_retrieval", rows, lambda: run_literature(
                    paths.section_dir("20", "findings") / "findings_catalog.csv",
                    paths.section_dir("20", "findings"), args.literature_mailto))
            run_section("edgewise_fd_sum", rows, lambda: run_edgewise_inference(paths, master, n_perm=args.edge_perms))
            run_section("live_brain_age", rows, lambda: run_live_brain_age(paths, master, n_repeats=args.brain_age_repeats))
            run_section("lr_sr", rows, lambda: run_lr_sr_analysis(paths, master))
            run_section("edr_exceptions", rows, lambda: run_edr_exception_analysis(paths, master))
            run_section("delay", rows, lambda: run_delay_analysis(paths, master))
            run_section("clinical", rows, lambda: run_clinical_covariate_review(paths))
            run_section("advanced_structural", rows, lambda: run_structural_innovation_analysis(paths, master))
            run_section("ml_diagnostics", rows, lambda: run_ml_diagnostics(paths, master))
            # clinical-outcome model search consumes the fresh ML feature matrix -> keep it on the dense cohort
            from connectome_analysis.analysis_clinical_outcome_search import run_search as _run_clinical_outcome_search
            run_section("clinical_outcome_search", rows,
                        lambda: _run_clinical_outcome_search(str(paths.deriv_root), fast=True, n_jobs=4))

        rows.append(
            status_row(
                "refresh_total",
                "ok",
                started_all,
                mode=args.mode,
                snapshot_mode=args.snapshot_mode,
                output_root=str(paths.output_root),
            )
        )
        write_exports(paths, rows, tables)
        refresh_status = {
            "last_refresh_utc": utc_now(),
            "mode": args.mode,
            "snapshot_mode": args.snapshot_mode,
            "output_root": str(paths.output_root),
            "connectomes_dir": str(paths.connectomes_dir),
            "section_status_csv": str(paths.exports_dir / "section_status.csv"),
        }
        status_path = paths.master_dir / "dashboard_refresh_status.json"
        status_path.write_text(json.dumps(refresh_status, indent=2) + "\n", encoding="utf-8")
        pd.DataFrame([refresh_status]).to_csv(paths.master_dir / "dashboard_refresh_status.csv", index=False)
        print(json.dumps(refresh_status, indent=2))
        return 0


def main() -> int:
    args = parse_args()
    return refresh(args)


if __name__ == "__main__":
    raise SystemExit(main())
