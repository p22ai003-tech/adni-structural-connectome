#!/usr/bin/env python3
"""Scratch-only SC-Forge v2 AAL3 endpoint-catchment canary.

This runner never writes production connectomes. It rebuilds a strict AAL3
contiguous-node/catchment contract under a timestamped QC directory, runs
connectomes from existing tracks, and records candidate/publication gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path("/home/ec2-user/exp")
SCFORGE = ROOT / "scforge"
DERIV = ROOT / "data" / "derivatives"
QC_ROOT = DERIV / "qc" / "sc_matrix_qc"
LATEST_FILE = QC_ROOT / "scforge_v2_latest.txt"

sys.path.insert(0, str(SCFORGE))

from scforge.assignment_diversity_qc import classify_assignment  # noqa: E402
from scforge.atlas_contract_v2 import label_survival_from_image, write_contiguous_from_image  # noqa: E402
from scforge.config import load_config  # noqa: E402
from scforge.connectome import REQUIRED_MATRICES, build_connectome_command_set  # noqa: E402
from scforge.endpoint_catchment import build_endpoint_catchment  # noqa: E402
from scforge.qc import matrix_qc_from_atlas_config  # noqa: E402
from scforge.reprocess_router import route_subject  # noqa: E402


DEFAULT_SUBJECTS = (
    ("003_S_0908_I1249292", "A", "residual_zero_row_recoverable"),
    ("014_S_4401_I1556672", "A", "residual_zero_row_recoverable"),
    ("041_S_5141_I893581", "A", "residual_zero_row_recoverable"),
    ("041_S_4427_I1243839", "A", "residual_zero_row_recoverable"),
    ("021_S_7092_I1597668", "B", "label_contract_failure"),
    ("031_S_4021_I1253150", "B", "label_contract_failure"),
    ("033_S_7114_I11063036", "B", "label_contract_failure"),
    ("003_S_6257_I974346", "B", "label_contract_failure"),
    ("127_S_5028_I401540", "C", "severe_assignment_collapse"),
    ("168_S_6874_I1667523", "C", "severe_assignment_collapse"),
)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fields = list(row)
    if exists:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            fields = next(reader, fields)
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    old_rows: list[dict[str, Any]] = []
    if exists:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            old_rows = list(csv.DictReader(handle))
    old_rows.append(row)
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for old in old_rows:
            writer.writerow({key: old.get(key, "") for key in fields})
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    tmp.replace(path)


def update_status(root: Path, **updates: Any) -> None:
    path = root / "status.json"
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.update(updates)
    data["last_update_utc"] = utc()
    write_json(path, data)


def append_finding(root: Path, message: str) -> None:
    with (root / "findings_log.md").open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc()}] {message.rstrip()}\n\n")


def env_with_tools() -> dict[str, str]:
    env = os.environ.copy()
    prefixes = ["/home/ec2-user/mrtrix3/bin", "/home/ec2-user/fsl/bin", "/home/ec2-user/bin"]
    env["PATH"] = ":".join(prefixes + [env.get("PATH", "")])
    env["PYTHONPATH"] = f"{SCFORGE}:{ROOT}:{env.get('PYTHONPATH', '')}"
    return env


def find_first(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def subject_artifacts(sid: str) -> dict[str, Path | None]:
    tracks_dir = DERIV / "tracks" / sid
    dti_dir = DERIV / "dti" / sid
    return {
        "aal_b0": DERIV / "parc" / sid / "AAL_b0.nii.gz",
        "tracks": find_first(
            [
                tracks_dir / "tracks_final_3000k.tck",
                tracks_dir / "tracks_final.tck",
                tracks_dir / "tracks_10M.tck",
            ]
        ),
        "sift2_weights": find_first([tracks_dir / "sift_weights.txt", tracks_dir / "sift2_weights.txt", tracks_dir / "sift2_weights.csv"]),
        "fa_mif": dti_dir / "fa.mif",
        "md_mif": dti_dir / "md.mif",
        "rd_mif": dti_dir / "rd.mif",
        "ad_mif": dti_dir / "ad.mif",
        "fa_tsf": tracks_dir / "fa_mean.tsf",
        "md_tsf": tracks_dir / "md_mean.tsf",
        "rd_tsf": tracks_dir / "rd_mean.tsf",
        "ad_tsf": tracks_dir / "ad_mean.tsf",
    }


def command_to_str(command: tuple[str, ...]) -> str:
    return " ".join(str(x) for x in command)


def run_command(root: Path, command: tuple[str, ...], *, stage: str, env: dict[str, str]) -> int:
    with (root / "commands.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc()}] {stage}: {command_to_str(command)}\n")
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    with (root / "stdout.log").open("a", encoding="utf-8", errors="replace") as handle:
        if proc.stdout:
            handle.write(f"\n[{utc()}] {stage}\n{proc.stdout}\n")
    with (root / "stderr.log").open("a", encoding="utf-8", errors="replace") as handle:
        if proc.stderr:
            handle.write(f"\n[{utc()}] {stage}\n{proc.stderr}\n")
    if proc.returncode != 0:
        append_finding(root, f"{stage} failed with rc={proc.returncode}: {command_to_str(command)}")
    return int(proc.returncode)


def maybe_reuse_tsf(subject_dir: Path, artifacts: dict[str, Path | None], metric: str) -> bool:
    source = artifacts.get(f"{metric}_tsf")
    target = subject_dir / "connectomes" / f"{metric}_mean.tsf"
    if source and source.exists() and source.stat().st_size > 0:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return True
    return False


def matrix_path(subject_dir: Path, metric: str) -> Path:
    return subject_dir / "connectomes" / f"SC_AAL_{metric}.csv"


def run_subject(root: Path, cfg, sid: str, branch: str, branch_reason: str, *, execute: bool) -> dict[str, Any]:
    subject_dir = root / "subjects" / sid
    atlas_dir = subject_dir / "atlas"
    conn_dir = subject_dir / "connectomes"
    qc_dir = subject_dir / "qc"
    for directory in (atlas_dir, conn_dir, qc_dir):
        directory.mkdir(parents=True, exist_ok=True)

    update_status(root, phase="subject", active_subject=sid, active_stage="locate_artifacts")
    artifacts = subject_artifacts(sid)
    required_inputs = ["aal_b0", "tracks", "fa_mif", "md_mif", "rd_mif", "ad_mif"]
    missing_inputs = [key for key in required_inputs if not artifacts.get(key) or not Path(artifacts[key]).exists()]
    if missing_inputs:
        row = {
            "sid": sid,
            "branch": branch,
            "branch_reason": branch_reason,
            "stage": "locate_artifacts",
            "status": "FAIL_INPUTS",
            "missing_inputs": ";".join(missing_inputs),
        }
        append_csv(root / "subject_stage_log.csv", {**row, "time_utc": utc()})
        append_finding(root, f"{sid}: missing required inputs: {', '.join(missing_inputs)}")
        return row

    aal_orig = atlas_dir / "aal3_orig_b0_1mm.nii.gz"
    aal_nodes = atlas_dir / "aal3_nodes_b0_1mm.nii.gz"
    catchment = atlas_dir / "aal3_endpoint_catchment_b0_1mm.nii.gz"

    shutil.copy2(Path(artifacts["aal_b0"]), aal_orig)
    label_qc = label_survival_from_image(aal_orig, cfg.atlas, cfg.qc)
    append_csv(root / "aal3_label_survival.csv", {"sid": sid, **label_qc.to_dict()})
    write_json(qc_dir / "aal3_label_survival.json", label_qc.to_dict())

    update_status(root, active_stage="build_contiguous_aal3", latest_missing_labels=len(label_qc.missing_valid_labels))
    try:
        write_contiguous_from_image(aal_orig, aal_nodes, cfg.atlas)
    except Exception as exc:
        row = {
            "sid": sid,
            "branch": branch,
            "branch_reason": branch_reason,
            "stage": "build_contiguous_aal3",
            "status": "FAIL_ATLAS_CONTRACT",
            "reason": f"{type(exc).__name__}:{exc}",
        }
        append_csv(root / "subject_stage_log.csv", {**row, "time_utc": utc()})
        append_finding(root, f"{sid}: contiguous AAL3 build failed: {row['reason']}")
        return row

    update_status(root, active_stage="build_endpoint_catchment")
    catchment_result = build_endpoint_catchment(aal_nodes, catchment, radius_mm=3.0)
    write_json(qc_dir / "endpoint_catchment.json", catchment_result.to_dict())

    commands = build_connectome_command_set(
        tracks=Path(artifacts["tracks"]),
        nodes=catchment,
        out_dir=conn_dir,
        assignment_variant="radial2",
        sift2_weights=Path(artifacts["sift2_weights"]) if artifacts.get("sift2_weights") else None,
        fa_mif=Path(artifacts["fa_mif"]),
        md_mif=Path(artifacts["md_mif"]),
        rd_mif=Path(artifacts["rd_mif"]),
        ad_mif=Path(artifacts["ad_mif"]),
    )
    env = env_with_tools()
    command_map = asdict(commands)
    failed_commands: list[str] = []
    for metric in ("fa", "md", "rd", "ad"):
        maybe_reuse_tsf(subject_dir, artifacts, metric)
    for stage, command in command_map.items():
        if stage.endswith("_sample") and (conn_dir / f"{stage.split('_')[0]}_mean.tsf").exists():
            append_csv(root / "subject_stage_log.csv", {"time_utc": utc(), "sid": sid, "stage": stage, "status": "SKIP_REUSE_TSF"})
            continue
        if not execute:
            append_csv(root / "subject_stage_log.csv", {"time_utc": utc(), "sid": sid, "stage": stage, "status": "DRY_RUN", "command": command_to_str(tuple(command))})
            continue
        update_status(root, active_stage=stage)
        rc = run_command(root, tuple(command), stage=f"{sid}:{stage}", env=env)
        append_csv(root / "subject_stage_log.csv", {"time_utc": utc(), "sid": sid, "stage": stage, "status": "OK" if rc == 0 else "FAIL", "returncode": rc})
        if rc != 0:
            failed_commands.append(stage)

    fd_sum_path = matrix_path(subject_dir, "fd_sum")
    matrix_qc = matrix_qc_from_atlas_config(fd_sum_path, cfg.qc, cfg.atlas)
    assignment_qc = classify_assignment(conn_dir / "assignments_fd_sum.txt")
    required_present = {
        metric: int(matrix_path(subject_dir, metric).exists() and matrix_path(subject_dir, metric).stat().st_size > 0)
        for metric in REQUIRED_MATRICES
    }
    missing_required = [metric for metric, present in required_present.items() if not present]

    matrix_n = matrix_qc.shape[0] if matrix_qc.shape else 0
    metrics = {
        "density": matrix_qc.density if matrix_qc.density is not None else 0.0,
        "valid_zero_rows": len(matrix_qc.valid_zero_rows),
        "missing_valid_labels": len(label_qc.missing_valid_labels),
        "unique_assigned_nodes": assignment_qc.unique_assigned_nodes,
        "top5_endpoint_fraction": assignment_qc.top5_endpoint_fraction,
        "matrix_n": matrix_n,
        "collapse_class": assignment_qc.collapse_class,
    }
    decision = route_subject(metrics, candidate_gate=cfg.qc.candidate_pass, publication_gate=cfg.qc.publication_pass)
    if failed_commands or missing_required:
        decision_route = "FAIL_REQUIRED_MATRIX_GENERATION"
        publication_pass = False
        candidate_pass = False
        decision_reasons = ";".join([*failed_commands, *[f"missing_{x}" for x in missing_required]])
    else:
        decision_route = decision.route
        publication_pass = decision.publication_pass
        candidate_pass = decision.candidate_pass
        decision_reasons = ";".join(decision.reasons)

    assignment_row = {"sid": sid, **assignment_qc.to_dict()}
    matrix_row = {
        "sid": sid,
        **matrix_qc.to_dict(),
        "matrix_n": matrix_n,
        "missing_required_matrices": ";".join(missing_required),
        **{f"has_{metric}": value for metric, value in required_present.items()},
    }
    route_row = {
        "sid": sid,
        "planned_branch": branch,
        "branch_reason": branch_reason,
        "candidate_pass": int(candidate_pass),
        "publication_pass": int(publication_pass),
        "route_decision": decision_route,
        "decision_reasons": decision_reasons,
        "density": metrics["density"],
        "valid_zero_rows": metrics["valid_zero_rows"],
        "missing_valid_labels": metrics["missing_valid_labels"],
        "matrix_n": matrix_n,
        "unique_assigned_nodes": assignment_qc.unique_assigned_nodes,
        "top5_endpoint_fraction": assignment_qc.top5_endpoint_fraction,
        "collapse_class": assignment_qc.collapse_class,
        "catchment_mode": catchment_result.mode,
        "catchment_voxels_added": catchment_result.voxels_added,
    }
    append_csv(root / "assignment_diversity_qc.csv", assignment_row)
    append_csv(root / "matrix_publication_qc.csv", matrix_row)
    append_csv(root / "scforge_v2_subject_routes.csv", route_row)
    write_json(qc_dir / "assignment_diversity.json", assignment_qc.to_dict())
    write_json(qc_dir / "matrix_publication_qc.json", matrix_row)
    write_json(qc_dir / "route_decision.json", route_row)
    append_finding(
        root,
        f"{sid}: route={decision_route}; candidate={int(candidate_pass)}; publication={int(publication_pass)}; "
        f"density={metrics['density']:.4f}; zero_rows={metrics['valid_zero_rows']}; "
        f"unique_nodes={assignment_qc.unique_assigned_nodes}; top5={assignment_qc.top5_endpoint_fraction:.3f}",
    )
    update_status(
        root,
        latest_subject=sid,
        latest_route=decision_route,
        latest_density=metrics["density"],
        latest_valid_zero_rows=metrics["valid_zero_rows"],
        latest_unique_assigned_nodes=assignment_qc.unique_assigned_nodes,
        latest_publication_pass=int(publication_pass),
        latest_candidate_pass=int(candidate_pass),
    )
    return route_row


def write_summaries(root: Path, routes: list[dict[str, Any]]) -> None:
    total = len(routes)
    candidate = sum(int(row.get("candidate_pass", 0)) for row in routes)
    publication = sum(int(row.get("publication_pass", 0)) for row in routes)
    by_route: dict[str, int] = {}
    for row in routes:
        by_route[str(row.get("route_decision", "UNKNOWN"))] = by_route.get(str(row.get("route_decision", "UNKNOWN")), 0) + 1
    summary = [
        {
            "total_subjects": total,
            "candidate_pass": candidate,
            "publication_pass": publication,
            "quarantine": total - publication,
            "route_counts": ";".join(f"{k}={v}" for k, v in sorted(by_route.items())),
        }
    ]
    write_csv(root / "scforge_v2_canary_summary.csv", summary)
    write_csv(root / "scforge_v2_publication_manifest.csv", [row for row in routes if int(row.get("publication_pass", 0)) == 1])
    write_csv(root / "scforge_v2_quarantine_manifest.csv", [row for row in routes if int(row.get("publication_pass", 0)) != 1])
    handoff = [
        "# SC-Forge v2 Canary Handoff",
        "",
        f"- UTC: {utc()}",
        f"- Subjects: {total}",
        f"- Candidate-pass: {candidate}",
        f"- Publication-pass: {publication}",
        f"- Scratch root: `{root}`",
        "",
        "## Route Counts",
        "",
    ]
    for route, count in sorted(by_route.items()):
        handoff.append(f"- {route}: {count}")
    handoff.extend(
        [
            "",
            "## Interpretation",
            "",
            "Candidate-pass subjects are not promoted. Publication-pass requires 166 x 166 matrices, no valid zero rows, no missing labels, assignment diversity, and complete metric matrices.",
            "Production connectomes were not overwritten by this run.",
        ]
    )
    (root / "scforge_v2_gpt_handoff.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scratch-only SC-Forge v2 canary")
    parser.add_argument("--config", default=str(SCFORGE / "configs" / "scforge.yaml"))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    tag = args.tag or f"scforge_v2_{stamp()}"
    if not tag.startswith("scforge_v2_"):
        tag = f"scforge_v2_{tag}"
    root = QC_ROOT / tag
    root.mkdir(parents=True, exist_ok=True)
    LATEST_FILE.write_text(str(root) + "\n", encoding="utf-8")
    subjects = list(DEFAULT_SUBJECTS)
    if args.subjects:
        requested = set(args.subjects)
        subjects = [row for row in subjects if row[0] in requested] + [(sid, "custom", "user_requested") for sid in args.subjects if sid not in {row[0] for row in subjects}]

    write_json(
        root / "run_manifest.json",
        {
            "tag": tag,
            "root": str(root),
            "start_utc": utc(),
            "execute": bool(args.execute),
            "subjects": [{"sid": sid, "branch": branch, "reason": reason} for sid, branch, reason in subjects],
            "production_overwrite": False,
            "assignment_variant": "radial2",
        },
    )
    for name in ("commands.log", "stdout.log", "stderr.log", "findings_log.md"):
        (root / name).touch()
    append_finding(root, "Started SC-Forge v2 scratch-only canary. Production outputs are not overwritten.")
    update_status(root, phase="start", total_subjects=len(subjects), completed_subjects=0, failed_subjects=0)

    routes: list[dict[str, Any]] = []
    failed = 0
    for idx, (sid, branch, reason) in enumerate(subjects, start=1):
        update_status(root, phase="subject", active_subject=sid, subject_index=idx, completed_subjects=len(routes), failed_subjects=failed)
        try:
            row = run_subject(root, cfg, sid, branch, reason, execute=bool(args.execute))
            routes.append(row)
            if str(row.get("status", "")) == "FAIL" or str(row.get("route_decision", "")).startswith("FAIL"):
                failed += 1
        except Exception as exc:
            failed += 1
            row = {
                "sid": sid,
                "planned_branch": branch,
                "branch_reason": reason,
                "route_decision": "FAIL_RUNNER_EXCEPTION",
                "decision_reasons": f"{type(exc).__name__}:{exc}",
                "candidate_pass": 0,
                "publication_pass": 0,
            }
            routes.append(row)
            append_csv(root / "scforge_v2_subject_routes.csv", row)
            append_finding(root, f"{sid}: runner exception {type(exc).__name__}: {exc}")
        write_summaries(root, routes)
        update_status(root, completed_subjects=len(routes), failed_subjects=failed)

    write_summaries(root, routes)
    update_status(root, phase="complete", active_subject="", active_stage="", completed_subjects=len(routes), failed_subjects=failed)
    append_finding(root, "SC-Forge v2 canary completed.")
    return 0 if routes else 1


if __name__ == "__main__":
    raise SystemExit(main())
