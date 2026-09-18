#!/usr/bin/env python3
"""Build a live, stage-resolved progress snapshot for the corrected 530 pipeline."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
MANIFEST = ROOT / "research_audit/outputs/connectome_v2_input_manifest_v2.csv"
LEGACY_MANIFEST = Path("/data/derivatives/qc/sc_matrix_qc/subject_manifest.csv")
LEGACY_MASTER = Path("/data/derivatives/qc/analysis_cohort/00_master/master_cohort.csv")
DEFAULT_RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260718_retry2"
)
DEFAULT_DECISION = (
    ROOT
    / "research_audit/outputs/h04a_r1_recovery_package_v1/recovery_execution_decision_retry2.json"
)
DEFAULT_OUTPUT = ROOT / "research_audit/outputs/pipeline_530_progress_v1"


CALIBRATION_STAGES = [
    ("T1 normalization", "00_inputs/t1_normalization.json", True),
    ("DWI normalization + recovered metadata", "00_inputs/dwi_normalization.json", True),
    ("Input contract", "00_inputs/input_contract.json", True),
    ("Gradient/shell contract", "01_dwi/gradient_contract.json", True),
    ("Denoise", "01_dwi/dwi_denoised.mif", False),
    ("De-Gibbs", "01_dwi/dwi_denoised_degibbs.mif", False),
    ("CPU Eddy", "01_dwi/dwi_preproc.mif", False),
    ("Bias correction", "01_dwi/dwi_preproc_biascorr.mif", False),
    ("Brain mask", "01_dwi/dwi_brain_mask.mif", False),
    ("FOD shell selection", "05_model/fod_shell_selection.json", True),
    ("Response calibration", "05_model/response_calibration_outcome.json", True),
]


FUTURE_STAGES = [
    ("Anatomy + AAL3 spatial contract", "04_atlas/atlas_contract_qc.json"),
    ("5TT/GMWMI", "05_model/gmwmi_dwi.mif"),
    ("Normalized FOD", "05_model/wmfod_norm.mif"),
    ("Tensor FA/MD/RD/AxD", "05_model/fa.mif"),
    ("Pre-tractography QC", "06_preflight/automated_pre_tractography_qc.json"),
    ("10M iFOD2 tractography", "07_tractography/tracks_10m.tck"),
    ("SIFT2", "07_tractography/sift2_weights.txt"),
    ("Nine AAL3-166 matrices", "07_connectome/matrices/count.csv"),
    ("Matrix QC + provenance", "08_qc/terminal_record.json"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def artifact_passes(path: Path, json_gate: bool) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    if not json_gate:
        return True
    status = str(load_json(path).get("status", "")).upper()
    return status in {"PASS", "COMPLETE", "VALID"}


def bar(done: int, total: int, width: int = 20) -> str:
    fraction = 0.0 if total <= 0 else min(1.0, max(0.0, done / total))
    filled = round(width * fraction)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {fraction:6.1%}"


def service_state(name: str) -> str:
    completed = subprocess.run(
        ["systemctl", "--user", "is-active", name],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


def snakemake_progress(run_root: Path) -> dict[str, int] | None:
    logs = sorted((run_root / "attempts").glob("*.snakemake.log"))
    if not logs:
        return None
    text = logs[-1].read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"(\d+) of (\d+) steps \((\d+)%\) done", text)
    if not matches:
        return None
    done, total, percent = matches[-1]
    return {"done": int(done), "total": int(total), "percent": int(percent)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--service", default="scforge-h04a-r1-phase-a-retry2.service"
    )
    args = parser.parse_args()

    parent_rows = read_csv(MANIFEST)
    parent_units = sorted(
        {
            row.get("unit", "")
            or (
                f"{row.get('subject_id', '').strip()}_I{row.get('dti_image_id', '').strip()}"
                if row.get("subject_id", "").strip()
                and row.get("dti_image_id", "").strip()
                else ""
            )
            for row in parent_rows
        }
        - {""}
    )
    decision = load_json(args.decision)
    calibration_units = [str(unit) for unit in decision.get("units", [])]
    subject_root = args.run_root / "subjects"

    legacy_rows = read_csv(LEGACY_MANIFEST)
    legacy_good = {
        row.get("sid", "")
        for row in legacy_rows
        if row.get("conn_location") == "good_main" and row.get("sid")
    }
    master_rows = read_csv(LEGACY_MASTER)
    old_qc_include = sum(
        str(row.get("sc_matrix_qc_include", "")).strip().lower()
        in {"1", "true", "yes"}
        for row in master_rows
    )

    stage_rows: list[dict[str, Any]] = []
    for stage, relative, json_gate in CALIBRATION_STAGES:
        passed = [
            unit
            for unit in calibration_units
            if artifact_passes(subject_root / unit / relative, json_gate)
        ]
        stage_rows.append(
            {
                "stream": "calibration_subset",
                "stage": stage,
                "artifact": relative,
                "complete": len(passed),
                "denominator": len(calibration_units),
                "units": passed,
            }
        )

    for stage, relative in FUTURE_STAGES:
        present = [
            unit
            for unit in parent_units
            if (subject_root / unit / relative).is_file()
        ]
        stage_rows.append(
            {
                "stream": "corrected_full_cohort",
                "stage": stage,
                "artifact": relative,
                "complete": len(present),
                "denominator": len(parent_units),
                "units": present,
            }
        )

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    workflow_progress = snakemake_progress(args.run_root)
    summary = {
        "schema_version": "1.0.0",
        "generated_utc": now,
        "parent_manifest": str(MANIFEST),
        "parent_denominator": len(parent_units),
        "legacy_good_main_connectomes": len(legacy_good),
        "legacy_old_qc_included": old_qc_include,
        "calibration_decision": str(args.decision),
        "calibration_denominator": len(calibration_units),
        "current_run_root": str(args.run_root),
        "service": args.service,
        "service_state": service_state(args.service),
        "snakemake_progress": workflow_progress,
        "stages": stage_rows,
        "interpretation": {
            "legacy_outputs": "directional existing-data evidence only; not the corrected V-final cohort",
            "calibration_subset": "current bounded response-recovery gate; no tractography or matrices are authorized in this run",
            "full_cohort": "future corrected execution after calibration, pre-tractography QC, canary connectomes, and the human full-run gate",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pipeline_progress.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "pipeline_progress.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stream", "stage", "artifact", "complete", "denominator"],
        )
        writer.writeheader()
        for row in stage_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    lines = [
        "# Corrected 530 pipeline progress",
        "",
        f"Generated: `{now}`",
        "",
        f"- Locked source denominator: **{len(parent_units)}/530**",
        f"- Legacy good-main connectomes present: **{len(legacy_good)}/530**",
        f"- Legacy old-QC include flags: **{old_qc_include}/530** (stale; not the corrected gate)",
        f"- Current service: **{summary['service_state']}**",
    ]
    if workflow_progress:
        lines.append(
            f"- Current bounded DAG: **{workflow_progress['done']}/{workflow_progress['total']} jobs ({workflow_progress['percent']}%)**"
        )
    lines.extend(["", "## Current 15-subject calibration gate", ""])
    for row in stage_rows:
        if row["stream"] == "calibration_subset":
            lines.append(
                f"- {bar(row['complete'], row['denominator'])} — {row['stage']}: "
                f"**{row['complete']}/{row['denominator']}**"
            )
    lines.extend(["", "## Corrected full-cohort continuation", ""])
    for row in stage_rows:
        if row["stream"] == "corrected_full_cohort":
            lines.append(
                f"- {bar(row['complete'], row['denominator'])} — {row['stage']}: "
                f"**{row['complete']}/{row['denominator']}**"
            )
    lines.extend(
        [
            "",
            "The current R1 run deliberately ends after response calibration. The same corrected recipe continues through anatomy/atlas, FOD/tensor, blinded pre-tractography QC, tractography/SIFT2, nine matrices, and publication QC only after its specified gates.",
            "",
        ]
    )
    (args.output_dir / "pipeline_progress.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
