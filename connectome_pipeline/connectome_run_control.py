#!/usr/bin/env python3
"""Shared run-control and SC-QC gate helpers for connectome notebooks.

This module is intentionally lightweight: it does not launch pipeline stages and
does not overwrite outputs.  It gives notebooks and command-line wrappers one
place to agree on dry-run/execution flags, include/exclude targeting, backup
policy, and the structural-connectome matrix QC analysis gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from connectome_pipeline.pipeline_paths import resolve_pipeline_paths


DEFAULT_PATHS = resolve_pipeline_paths()
DEFAULT_SC_QC_DIR = DEFAULT_PATHS.deriv_root / "qc" / "sc_matrix_qc"
DEFAULT_ANALYSIS_GATE_CSV = DEFAULT_SC_QC_DIR / "analysis_gate_after_sc_qc.csv"
FALLBACK_QC_DECISIONS_CSV = DEFAULT_SC_QC_DIR / "sc_matrix_qc_decisions.csv"


def normalize_sid_list(items: Iterable[str] | None) -> list[str]:
    """Return unique, non-empty subject IDs preserving input order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items or []:
        sid = str(item).strip()
        if not sid or sid.startswith("#") or sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


@dataclass(frozen=True)
class RunControlConfig:
    """Notebook/CLI execution contract shared across connectome stages."""

    dry_run: bool = True
    execute: bool = False
    force_invalid_only: bool = True
    include: tuple[str, ...] = field(default_factory=tuple)
    exclude: tuple[str, ...] = field(default_factory=tuple)
    backup_before_replace: bool = True
    validation_gate: bool = True
    include_warn: bool = False
    analysis_gate_csv: Path = DEFAULT_ANALYSIS_GATE_CSV

    @classmethod
    def from_values(
        cls,
        *,
        dry_run: bool = True,
        execute: bool = False,
        force_invalid_only: bool = True,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        backup_before_replace: bool = True,
        validation_gate: bool = True,
        include_warn: bool = False,
        analysis_gate_csv: str | Path | None = None,
    ) -> "RunControlConfig":
        return cls(
            dry_run=bool(dry_run),
            execute=bool(execute),
            force_invalid_only=bool(force_invalid_only),
            include=tuple(normalize_sid_list(include)),
            exclude=tuple(normalize_sid_list(exclude)),
            backup_before_replace=bool(backup_before_replace),
            validation_gate=bool(validation_gate),
            include_warn=bool(include_warn),
            analysis_gate_csv=Path(analysis_gate_csv) if analysis_gate_csv else DEFAULT_ANALYSIS_GATE_CSV,
        )

    @property
    def effective_dry_run(self) -> bool:
        return self.dry_run or not self.execute

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["analysis_gate_csv"] = str(self.analysis_gate_csv)
        payload["include"] = list(self.include)
        payload["exclude"] = list(self.exclude)
        payload["effective_dry_run"] = self.effective_dry_run
        return payload


def _analysis_gate_path(path: str | Path | None = None) -> Path | None:
    candidates = [Path(path)] if path else [DEFAULT_ANALYSIS_GATE_CSV, FALLBACK_QC_DECISIONS_CSV]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_analysis_gate(path: str | Path | None = None) -> list[dict[str, str]]:
    """Load the current SC matrix QC analysis gate, if present."""
    gate_path = _analysis_gate_path(path)
    if gate_path is None:
        return []
    with gate_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def analysis_ready_sids(
    path: str | Path | None = None,
    *,
    include_warn: bool = False,
) -> list[str]:
    """Return subject IDs allowed into downstream analysis by the QC gate."""
    rows = load_analysis_gate(path)
    ready: list[str] = []
    for row in rows:
        sid = (row.get("sid") or row.get("subject") or row.get("subject_id") or "").strip()
        if not sid:
            continue
        status = (row.get("whole_matrix_qc_status") or row.get("sc_matrix_qc_status") or row.get("status") or "").strip()
        gate = (row.get("analysis_gate") or row.get("include") or "").strip().lower()
        if status == "PASS" or gate == "include":
            ready.append(sid)
        elif include_warn and status == "WARN":
            ready.append(sid)
    return normalize_sid_list(ready)


def filter_subjects(
    subjects: Iterable[str],
    *,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    gate_path: str | Path | None = None,
    validation_gate: bool = False,
    include_warn: bool = False,
) -> list[str]:
    """Apply include/exclude and, optionally, the SC-QC analysis gate."""
    selected = normalize_sid_list(subjects)
    include_set = set(normalize_sid_list(include))
    exclude_set = set(normalize_sid_list(exclude))
    if include_set:
        selected = [sid for sid in selected if sid in include_set]
    if exclude_set:
        selected = [sid for sid in selected if sid not in exclude_set]
    if validation_gate:
        ready = set(analysis_ready_sids(gate_path, include_warn=include_warn))
        if ready:
            selected = [sid for sid in selected if sid in ready]
    return selected


def invalid_sc_subjects(
    path: str | Path | None = None,
    *,
    include_warn: bool = False,
) -> list[str]:
    """Return subjects currently marked not analysis-ready by the SC matrix QC gate."""
    rows = load_analysis_gate(path)
    invalid: list[str] = []
    for row in rows:
        sid = (row.get("sid") or row.get("subject") or row.get("subject_id") or "").strip()
        if not sid:
            continue
        status = (row.get("whole_matrix_qc_status") or row.get("sc_matrix_qc_status") or row.get("status") or "").strip()
        gate = (row.get("analysis_gate") or row.get("include") or "").strip().lower()
        if gate.startswith("exclude") or status.startswith("FAIL"):
            invalid.append(sid)
        elif status == "WARN" and not include_warn:
            invalid.append(sid)
    return normalize_sid_list(invalid)


def print_run_control(cfg: RunControlConfig) -> None:
    """Print a compact run-control summary in notebooks."""
    print("Run control")
    print("-----------")
    for key, value in cfg.as_dict().items():
        print(f"{key:<22}: {value}")
    gate_path = _analysis_gate_path(cfg.analysis_gate_csv)
    if gate_path:
        rows = load_analysis_gate(gate_path)
        ready = analysis_ready_sids(gate_path, include_warn=cfg.include_warn)
        print(f"analysis_gate_rows   : {len(rows)}")
        print(f"analysis_ready_sids  : {len(ready)}")
        print(f"analysis_gate_source : {gate_path}")
    else:
        print("analysis_gate_source : missing")


def write_run_control_snapshot(
    path: str | Path,
    cfg: RunControlConfig,
    *,
    targets: Sequence[str] | None = None,
    extras: Mapping[str, object] | None = None,
) -> Path:
    """Write a machine-readable run-control snapshot for provenance."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_control": cfg.as_dict(),
        "targets": list(targets or []),
        "extras": dict(extras or {}),
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out)
    return out
