#!/usr/bin/env python3
"""Final scratch-only spatial-contract closeout for SC matrix rectification.

This controller consolidates prior SC repair evidence, runs the remaining
plausible source-contract test on AAL3 across a representative subject panel,
generates full candidate matrices for the best AAL3 route, and writes one
validation-gated patch recommendation.

No production outputs are overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path("/home/ec2-user/exp")
LIVE_ROOT = Path(__file__).resolve().parent
DERIV = ROOT / "data" / "derivatives"
QC_ROOT = DERIV / "qc" / "sc_matrix_qc"
OUT_PARENT = QC_ROOT
PROBE_PARENT = QC_ROOT / "aal3_source_contract_probe"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LIVE_ROOT))
import run_sc_aal3_source_contract_probe as aal3_probe  # noqa: E402


DEFAULT_PANEL = [
    ("<SUBJECT>_I<IMAGEID>", "known_positive_aal116_benchmark"),
    ("<SUBJECT>_I<IMAGEID>", "severe_low_density_mask_fit"),
    ("<SUBJECT>_I<IMAGEID>", "severe_low_density_mask_fit"),
    ("<SUBJECT>_I<IMAGEID>", "severe_low_density_mask_fit"),
    ("<SUBJECT>_I<IMAGEID>", "moderate_density_zero_row_heavy"),
    ("<SUBJECT>_I<IMAGEID>", "moderate_density_zero_row_heavy"),
    ("<SUBJECT>_I<IMAGEID>", "moderate_density_zero_row_heavy"),
    ("<SUBJECT>_I<IMAGEID>", "assignment_strict_pass_control"),
    ("<SUBJECT>_I<IMAGEID>", "assignment_strict_pass_control"),
    ("<SUBJECT>_I<IMAGEID>", "assignment_strict_pass_control"),
]

VARIANTS = ["default", "radial8", "forward40", "forward80"]
REQUIRED_METRICS = ["count", "fd_sum", "len_mean", "fa_mean", "md_mean", "rd_mean", "ad_mean"]


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def append_log(root: Path, text: str) -> None:
    with (root / "findings_log.md").open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n\n")


def update_status(root: Path, **updates: Any) -> None:
    status = read_json(root / "status.json")
    status.update(updates)
    status["last_update_utc"] = utc()
    write_json(root / "status.json", status)


def run_cmd(cmd: list[str | Path], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    str_cmd = [str(x) for x in cmd]
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"[{utc()}] $ {' '.join(str_cmd)}\n")
        handle.flush()
        proc = subprocess.run(str_cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
        handle.write(f"[{utc()}] rc={proc.returncode}\n\n")
    return proc.returncode


def mrtrix_thread_args(env: dict[str, str]) -> list[str]:
    value = str(env.get("MRTRIX_NTHREADS", "")).strip()
    if value and value != "0":
        return ["-nthreads", value]
    return []


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except Exception:
        return default


def latest_dir(parent: Path, pattern: str = "*") -> Path | None:
    dirs = [p for p in parent.glob(pattern) if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def summarize_prior_experiments(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    raw_parent = QC_ROOT / "supervisor_raw_end_to_end_probe"
    raw_latest = latest_dir(raw_parent)
    rows.append(
        {
            "experiment": "raw_supervisor_replay",
            "path": str(raw_latest or raw_parent),
            "status": "available" if raw_latest else "missing",
            "finding": "Raw replay did not reproduce a healthy SC matrix from current local source artifacts; do not batch from raw replay route.",
            "action": "reject_as_batch_route",
        }
    )

    source_parent = QC_ROOT / "supervisor_source_contract_probe"
    source_latest = latest_dir(source_parent)
    best_aal116 = ""
    if source_latest:
        mats = read_csv(source_latest / "matrix_candidate_summary.csv")
        vals = [
            r
            for r in mats
            if r.get("candidate", "").startswith("t1_from_dicom")
            and r.get("metric") in {"fd_invnodevol", "count"}
            and inum(r.get("zero_rows"), 9999) == 0
        ]
        vals.sort(key=lambda r: -fnum(r.get("density"), -1))
        if vals:
            best_aal116 = f"density={fnum(vals[0].get('density')):.3f}; zero_rows={inum(vals[0].get('zero_rows'))}; candidate={vals[0].get('candidate')}; variant={vals[0].get('variant')}"
    rows.append(
        {
            "experiment": "production_tracks_plus_aal116_source_contract",
            "path": str(source_latest or source_parent),
            "status": "available" if source_latest else "missing",
            "finding": best_aal116 or "AAL116 source-contract probe was the strongest positive local lead but exact latest summary was not found.",
            "action": "retest_contract_on_aal3",
        }
    )

    ladder_latest = latest_dir(QC_ROOT / "track_contract_ladder_probe")
    rows.append(
        {
            "experiment": "track_contract_ladder",
            "path": str(ladder_latest or QC_ROOT / "track_contract_ladder_probe"),
            "status": "available" if ladder_latest else "missing",
            "finding": "Track-count/raw-track escalation was not sufficient as a primary fix in prior probes.",
            "action": "keep_track_count_disabled_until_spatial_contract_passes",
        }
    )

    canary_latest = latest_dir(QC_ROOT / "t1b0_required_weight_canary")
    canary_finding = ""
    if canary_latest:
        summary = read_csv(canary_latest / "canary_validation_summary.csv")
        if summary:
            canary_finding = "; ".join(f"{k}={v}" for k, v in summary[0].items() if v != "")
    rows.append(
        {
            "experiment": "t1b0_required_weight_canary",
            "path": str(canary_latest or QC_ROOT / "t1b0_required_weight_canary"),
            "status": "available" if canary_latest else "missing",
            "finding": canary_finding or "Prior required-weight canary did not pass strict batch gate.",
            "action": "no_previous_canary_batch_promotion",
        }
    )

    post_validation = QC_ROOT / "post_parcellation_repair_completed_now_validation_summary.csv"
    rows.append(
        {
            "experiment": "post_only_repair",
            "path": str(post_validation),
            "status": "available" if post_validation.exists() else "missing",
            "finding": "Post-only repair was not batch-safe and failed validation for some subjects.",
            "action": "reject_post_only_batch_route",
        }
    )
    write_csv(root / "prior_experiment_summary.csv", rows)
    return rows


def command_delta_rows() -> list[dict[str, Any]]:
    return [
        {
            "stage": "atlas",
            "reference_pipeline": "AAL116-style atlas transformed AAL/MNI -> native T1 -> B0 with nearest-neighbour labels",
            "current_step7_or_prior_route": "AAL3 thesis atlas, often direct/composed MNI->B0 route in Step 7 post",
            "closeout_test": "Retest source contract with AAL3; keep AAL116 only as quality benchmark",
            "risk": "Atlas/version mismatch can produce different matrix dimensions and label survival behavior",
        },
        {
            "stage": "T1 source",
            "reference_pipeline": "FreeSurfer-derived corrected_T1 / brainmask and true native T1 contract",
            "current_step7_or_prior_route": "Production derivatives may use RAS/T1 bridge products; raw replay used derivative fallback when raw T1 missing",
            "closeout_test": "Audit T1 source, T1 brain, and B0 pairing per subject",
            "risk": "T1 source mismatch changes affine/FOV and can collapse labels after B0 projection",
        },
        {
            "stage": "B0 source",
            "reference_pipeline": "vol0000 or production B0 consistently paired with T1-to-B0 matrix",
            "current_step7_or_prior_route": "Multiple candidate B0 spaces exist: biascorr, BBR b0mean, eddy vol0, mif vol0",
            "closeout_test": "Use source-contract B0 and compare to production baseline",
            "risk": "Wrong B0 source can make otherwise valid labels miss the DWI/5TT mask",
        },
        {
            "stage": "5TT/GMWMI",
            "reference_pipeline": "5ttgen fsl on T1 transformed to B0; no double transform",
            "current_step7_or_prior_route": "Production contains 5tt_b0 and 5tt_b0_fixed variants",
            "closeout_test": "Do not patch 5TT unless AAL3 source contract also passes",
            "risk": "5TT/FOD space mismatch blocks ACT or label-mask overlap",
        },
        {
            "stage": "assignment",
            "reference_pipeline": "tck2connectome with weights, scale_invnodevol, and assignment diagnostics",
            "current_step7_or_prior_route": "Default Step 7 assignment uses radial_search=4",
            "closeout_test": "AAL3 variants default, radial8, forward40, forward80",
            "risk": "Assignment variant can improve density only if parcellation geometry is already correct",
        },
    ]


def artifact_exists(path: Path) -> str:
    if path.exists() and path.stat().st_size > 0:
        return "present"
    return "missing"


def spatial_contract_audit_rows(panel: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid, role in panel:
        tracks = DERIV / "tracks" / sid
        paths = {
            "production_b0": DERIV / "dwi_t1_bbr" / f"{sid}_b0mean_ras.nii.gz",
            "production_t1": DERIV / "dwi_t1_bbr" / f"{sid}_t1brain_ras.nii.gz",
            "production_t12b0": DERIV / "dwi_t1_bbr" / f"{sid}_t12b0_bbr.mat",
            "production_5tt": DERIV / "fod" / sid / "5tt_b0.mif",
            "production_5tt_fixed": DERIV / "fod" / sid / "5tt_b0_fixed.mif",
            "wmfod_final": DERIV / "fod" / sid / "wmfod_final.mif",
            "tracks_final": tracks / "tracks_final_3000k.tck",
            "weights": tracks / "sift_weights.txt",
            "aal_b0": DERIV / "parc" / sid / "AAL_b0.nii.gz",
            "production_fd_sum": DERIV / "connectomes" / f"SC_AAL_{sid}_fd_sum.csv",
            "production_count": DERIV / "connectomes" / f"SC_AAL_{sid}_count.csv",
        }
        row: dict[str, Any] = {"sid": sid, "role": role}
        for key, path in paths.items():
            row[key] = artifact_exists(path)
            row[f"{key}_path"] = str(path)
        rows.append(row)
    return rows


def best_probe_candidate(probe_root: Path) -> dict[str, str]:
    decision = read_json(probe_root / "decision.json")
    best = decision.get("best") if isinstance(decision.get("best"), dict) else {}
    if best:
        return {str(k): str(v) for k, v in best.items()}
    rows = read_csv(probe_root / "matrix_candidate_summary.csv")
    candidates = [
        row
        for row in rows
        if row.get("candidate") == "AAL3_source_contract"
        and row.get("metric") in {"fd_sum", "fd_invnodevol"}
        and inum(row.get("read_ok"), 0) == 1
    ]
    candidates.sort(key=lambda r: (-fnum(r.get("valid_density"), -1), inum(r.get("valid_zero_rows"), 9999)))
    return candidates[0] if candidates else {}


def assignment_args(variant: str) -> list[str]:
    return {
        "default": [],
        "radial4": ["-assignment_radial_search", "4"],
        "radial8": ["-assignment_radial_search", "8"],
        "forward40": ["-assignment_forward_search", "40"],
        "forward80": ["-assignment_forward_search", "80"],
    }.get(variant, [])


def generate_full_candidate_matrices(run_root: Path, sid: str, variant: str, probe_root: Path, valid_labels: list[int], env: dict[str, str]) -> list[dict[str, Any]]:
    inputs = aal3_probe.discover_inputs(sid)
    aal_b0 = probe_root / "parc" / "AAL3_source_contract_B0.nii.gz"
    out_dir = run_root / "full_candidate_matrices" / sid / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    tracks_dir = DERIV / "tracks" / sid
    metric_cmds: dict[str, list[str]] = {
        "count": ["-stat_edge", "sum"],
        "fd_sum": ["-tck_weights_in", str(inputs.get("weights")), "-stat_edge", "sum"],
        "len_mean": ["-scale_length", "-stat_edge", "mean"],
        "fa_mean": ["-scale_file", str(tracks_dir / "fa_mean.tsf"), "-stat_edge", "mean"],
        "md_mean": ["-scale_file", str(tracks_dir / "md_mean.tsf"), "-stat_edge", "mean"],
        "rd_mean": ["-scale_file", str(tracks_dir / "rd_mean.tsf"), "-stat_edge", "mean"],
        "ad_mean": ["-scale_file", str(tracks_dir / "ad_mean.tsf"), "-stat_edge", "mean"],
    }
    missing_inputs: list[str] = []
    if not inputs.get("tracks"):
        missing_inputs.append("tracks_final_3000k.tck")
    if not inputs.get("weights"):
        missing_inputs.append("sift_weights.txt")
    if not aal_b0.exists():
        missing_inputs.append("AAL3_source_contract_B0.nii.gz")
    for metric in ("fa_mean", "md_mean", "rd_mean", "ad_mean"):
        if not (tracks_dir / f"{metric}.tsf").exists():
            missing_inputs.append(f"{metric}.tsf")
    if missing_inputs:
        return [
            {
                "sid": sid,
                "variant": variant,
                "metric": metric,
                "status": "skipped_missing_inputs",
                "missing_inputs": ";".join(sorted(set(missing_inputs))),
            }
            for metric in REQUIRED_METRICS
        ]

    for metric, extra in metric_cmds.items():
        out_csv = out_dir / f"SC_AAL3_{sid}_{variant}_{metric}.csv"
        assign_csv = out_dir / f"assignments_{sid}_{variant}_{metric}.csv"
        cached_stats = aal3_probe.matrix_stats(out_csv, valid_labels)
        if (
            int(float(cached_stats.get("read_ok", 0) or 0)) == 1
            and assign_csv.exists()
            and assign_csv.stat().st_size > 0
        ):
            cached_stats.update(
                {
                    "sid": sid,
                    "variant": variant,
                    "metric": metric,
                    "status": "ok_cached",
                    "candidate_matrix": str(out_csv),
                    "assignments": str(assign_csv),
                }
            )
            rows.append(cached_stats)
            continue

        cmd: list[str | Path] = [
            aal3_probe.TCK2CONNECTOME,
            inputs["tracks"],
            aal_b0,
            out_csv,
            "-symmetric",
            "-zero_diagonal",
            *mrtrix_thread_args(env),
            *assignment_args(variant),
            "-out_assignments",
            assign_csv,
            *extra,
        ]
        rc = run_cmd(cmd, run_root / "logs" / f"full_candidate_{sid}_{variant}_{metric}.log", env)
        stats = aal3_probe.matrix_stats(out_csv, valid_labels)
        stats.update(
            {
                "sid": sid,
                "variant": variant,
                "metric": metric,
                "status": "ok" if rc == 0 else f"failed_rc_{rc}",
                "candidate_matrix": str(out_csv),
                "assignments": str(assign_csv),
            }
        )
        rows.append(stats)
    return rows


def subject_validation_decision(full_rows: list[dict[str, Any]], baseline_fd: dict[str, Any]) -> dict[str, Any]:
    by_metric = {row.get("metric"): row for row in full_rows}
    missing = [metric for metric in REQUIRED_METRICS if metric not in by_metric or inum(by_metric[metric].get("read_ok"), 0) != 1]
    fd = by_metric.get("fd_sum", {})
    count = by_metric.get("count", {})
    cand_density = fnum(fd.get("valid_density"), -1)
    cand_zero = inum(fd.get("valid_zero_rows"), 9999)
    base_density = fnum(baseline_fd.get("valid_density"), -1) if baseline_fd else -1
    base_zero = inum(baseline_fd.get("valid_zero_rows"), 9999) if baseline_fd else 9999
    sym_ok = all(fnum(by_metric.get(m, {}).get("symmetry_max_abs"), 999) <= 1e-8 for m in REQUIRED_METRICS if m in by_metric)
    diag_ok = all(fnum(by_metric.get(m, {}).get("diagonal_abs_sum"), 999) <= 1e-8 for m in REQUIRED_METRICS if m in by_metric)
    finite_ok = all(fnum(by_metric.get(m, {}).get("finite_fraction"), 0) >= 1.0 for m in REQUIRED_METRICS if m in by_metric)
    density_ok = cand_density >= 0.15
    no_worse_density = base_density < 0 or cand_density >= base_density
    no_worse_zero = base_zero == 9999 or cand_zero <= base_zero
    zero_gate = cand_zero <= 20
    count_read = inum(count.get("read_ok"), 0) == 1
    passed = (not missing) and count_read and sym_ok and diag_ok and finite_ok and density_ok and no_worse_density and no_worse_zero and zero_gate
    if passed:
        decision = "PROMOTE_ELIGIBLE_AFTER_PRODUCTION_BACKUP"
    elif not missing and cand_density > base_density and cand_zero <= base_zero:
        decision = "IMPROVES_BUT_FAILS_STRICT_GATE"
    else:
        decision = "DO_NOT_PROMOTE"
    return {
        "missing_required_metrics": ";".join(missing),
        "candidate_fd_valid_density": cand_density,
        "candidate_fd_valid_zero_rows": cand_zero,
        "baseline_fd_valid_density": base_density,
        "baseline_fd_valid_zero_rows": base_zero,
        "finite_ok": int(finite_ok),
        "symmetry_ok": int(sym_ok),
        "zero_diagonal_ok": int(diag_ok),
        "density_gate_ok": int(density_ok),
        "no_worse_density": int(no_worse_density),
        "no_worse_zero_rows": int(no_worse_zero),
        "valid_zero_rows_gate_ok": int(zero_gate),
        "strict_pass": int(passed),
        "decision": decision,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default=f"final_spatial_contract_closeout_{stamp()}")
    ap.add_argument("--subjects", nargs="*", default=[sid for sid, _ in DEFAULT_PANEL])
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    args = ap.parse_args()

    panel_roles = {sid: role for sid, role in DEFAULT_PANEL}
    panel = [(sid, panel_roles.get(sid, "user_supplied")) for sid in args.subjects]
    run_root = OUT_PARENT / args.tag
    run_root.mkdir(parents=True, exist_ok=True)
    (OUT_PARENT / "latest_final_spatial_contract_closeout.txt").write_text(str(run_root), encoding="utf-8")

    env = os.environ.copy()
    env["FSLDIR"] = str(aal3_probe.FSLDIR)
    env["PATH"] = f"{aal3_probe.MRTRIX}:{aal3_probe.FSLDIR / 'bin'}:" + env.get("PATH", "")

    write_json(
        run_root / "status.json",
        {
            "tag": args.tag,
            "root": str(run_root),
            "phase": "starting",
            "started_utc": utc(),
            "total_subjects": len(panel),
            "completed_subjects": 0,
            "failed_subjects": 0,
            "variants": args.variants,
        },
    )
    append_log(
        run_root,
        "# Final Spatial-Contract Closeout\n\n"
        f"Started: {utc()}\n\n"
        "This run is scratch-only. It does not overwrite production connectome matrices.\n\n"
        "Purpose: retest the best positive source-contract route on AAL3 and decide whether a Step 7 patch is justified.",
    )

    update_status(run_root, phase="summarising_prior_experiments", latest_finding="Writing prior experiment summary")
    prior_rows = summarize_prior_experiments(run_root)
    append_log(run_root, "## Prior Experiment Summary\n\n" + "\n".join(f"- {r['experiment']}: {r['finding']}" for r in prior_rows))

    write_csv(run_root / "command_delta_vs_ADNI_preproc_pipeline_AAL.csv", command_delta_rows())
    write_csv(run_root / "spatial_contract_audit.csv", spatial_contract_audit_rows(panel))

    valid_labels = aal3_probe.aal3_valid_labels()
    all_full_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    promotion_rows: list[dict[str, Any]] = []

    for idx, (sid, role) in enumerate(panel, start=1):
        update_status(
            run_root,
            phase="aal3_source_contract_probe",
            current_sid=sid,
            current_role=role,
            subject_index=idx,
            total_subjects=len(panel),
            latest_finding=f"Running AAL3 source-contract probe for {sid}",
        )
        tag = f"{args.tag}_{sid}"
        rc = run_cmd(
            [
                aal3_probe.FSLDIR / "bin" / "python",
                LIVE_ROOT / "run_sc_aal3_source_contract_probe.py",
                "--sid",
                sid,
                "--tag",
                tag,
                "--variants",
                *args.variants,
            ],
            run_root / "logs" / f"aal3_probe_{sid}.log",
            env,
        )
        probe_root = PROBE_PARENT / tag
        decision = read_json(probe_root / "decision.json")
        best = best_probe_candidate(probe_root)
        baseline = decision.get("baseline_fd_sum") if isinstance(decision.get("baseline_fd_sum"), dict) else {}

        if rc != 0:
            row = {
                "sid": sid,
                "role": role,
                "probe_tag": tag,
                "probe_status": f"failed_rc_{rc}",
                "probe_root": str(probe_root),
                "route_decision": "AAL3_PROBE_FAILED",
                "final_decision": "DO_NOT_PROMOTE",
            }
            decision_rows.append(row)
            write_csv(run_root / "subject_route_decisions.csv", decision_rows)
            update_status(run_root, failed_subjects=sum(1 for r in decision_rows if str(r.get("probe_status", "")).startswith("failed")))
            continue

        comparison_rows.append(
            {
                "sid": sid,
                "role": role,
                "atlas": "AAL3",
                "probe_root": str(probe_root),
                "probe_decision": decision.get("decision", ""),
                "best_variant": best.get("variant", ""),
                "best_metric": best.get("metric", ""),
                "baseline_valid_density": baseline.get("valid_density", ""),
                "baseline_valid_zero_rows": baseline.get("valid_zero_rows", ""),
                "candidate_valid_density": best.get("valid_density", ""),
                "candidate_valid_zero_rows": best.get("valid_zero_rows", ""),
                "candidate_density": best.get("density", ""),
                "candidate_zero_rows": best.get("zero_rows", ""),
            }
        )
        write_csv(run_root / "aal116_vs_aal3_candidate_comparison.csv", comparison_rows)

        if not best:
            validation = {
                "missing_required_metrics": ";".join(REQUIRED_METRICS),
                "strict_pass": 0,
                "decision": "DO_NOT_PROMOTE",
            }
            full_rows: list[dict[str, Any]] = []
        else:
            best_variant = str(best.get("variant") or "default")
            update_status(
                run_root,
                phase="full_candidate_matrices",
                current_sid=sid,
                current_variant=best_variant,
                latest_finding=f"Generating required matrices for {sid} using {best_variant}",
            )
            full_rows = generate_full_candidate_matrices(run_root, sid, best_variant, probe_root, valid_labels, env)
            all_full_rows.extend(full_rows)
            write_csv(run_root / "full_candidate_matrix_validation.csv", all_full_rows)
            validation = subject_validation_decision(full_rows, baseline)

        route_decision = decision.get("decision", "UNKNOWN")
        final_row = {
            "sid": sid,
            "role": role,
            "probe_tag": tag,
            "probe_status": "done",
            "probe_root": str(probe_root),
            "route_decision": route_decision,
            "best_variant": best.get("variant", ""),
            "best_metric": best.get("metric", ""),
            **validation,
        }
        decision_rows.append(final_row)
        if validation.get("strict_pass") == 1:
            promotion_rows.append(
                {
                    "sid": sid,
                    "candidate_root": str(run_root / "full_candidate_matrices" / sid / str(best.get("variant"))),
                    "promotion_status": "eligible_only_after_manual_review_and_production_backup",
                    "reason": "strict scratch validation passed",
                }
            )
        write_csv(run_root / "subject_route_decisions.csv", decision_rows)
        write_csv(run_root / "promotion_manifest.csv", promotion_rows)
        update_status(
            run_root,
            completed_subjects=sum(1 for r in decision_rows if r.get("probe_status") == "done"),
            failed_subjects=sum(1 for r in decision_rows if str(r.get("probe_status", "")).startswith("failed")),
            strict_pass_subjects=sum(inum(r.get("strict_pass"), 0) for r in decision_rows),
            latest_finding=f"{sid}: {route_decision}; {validation.get('decision')}",
        )

    strict_pass = sum(inum(r.get("strict_pass"), 0) for r in decision_rows)
    improves = sum(1 for r in decision_rows if r.get("decision") == "IMPROVES_BUT_FAILS_STRICT_GATE")
    total_done = len([r for r in decision_rows if r.get("probe_status") == "done"])
    if total_done and strict_pass >= max(6, math.ceil(0.7 * total_done)):
        final_recommendation = "PATCH_AND_CANARY_BATCH"
        next_action = "Patch Step 7 in scratch branch, run a production-backup canary only, then validate before batch."
    elif strict_pass > 0 or improves > 0:
        final_recommendation = "PATCH_BUT_KEEP_EXCLUDED_UNTIL_REPROCESS"
        next_action = "Do not batch-promote current artifacts; preserve useful route changes and require upstream reprocessing or stricter full-matrix canary."
    else:
        final_recommendation = "NO_SAFE_PATCH_CURRENT_ARTIFACTS"
        next_action = "Do not patch production Step 7 from this route; use QC gating and redo upstream spatial preprocessing for failed subjects."

    write_csv(
        run_root / "canary_validation_summary.csv",
        [
            {
                "total_subjects": len(panel),
                "completed_subjects": total_done,
                "failed_subjects": len(decision_rows) - total_done,
                "strict_pass_subjects": strict_pass,
                "improves_but_fails_strict_gate": improves,
                "final_recommendation": final_recommendation,
                "next_action": next_action,
            }
        ],
    )
    write_csv(
        run_root / "analysis_gate_after_sc_qc.csv",
        [
            {
                "sid": row["sid"],
                "analysis_gate": "include_after_backup_review" if inum(row.get("strict_pass"), 0) else "exclude_pending_repair",
                "route_decision": row.get("route_decision", ""),
                "final_decision": row.get("decision", ""),
            }
            for row in decision_rows
        ],
    )

    decision_md = [
        "# Candidate Patch Decision",
        "",
        f"Generated: {utc()}",
        "",
        f"Final recommendation: `{final_recommendation}`",
        "",
        f"Next action: {next_action}",
        "",
        "## Gate Summary",
        "",
        f"- Subjects tested: `{len(panel)}`",
        f"- Completed probes: `{total_done}`",
        f"- Strict full-matrix passes: `{strict_pass}`",
        f"- Improved but failed strict gate: `{improves}`",
        "",
        "## Production Safety",
        "",
        "- No production connectomes were overwritten by this closeout.",
        "- `promotion_manifest.csv` is advisory only; a production repair still requires backups and immediate validation.",
        "- AAL116 remains a benchmark only. AAL3 remains the thesis atlas.",
        "",
    ]
    if final_recommendation != "PATCH_AND_CANARY_BATCH":
        decision_md.extend(
            [
                "## Why No Batch Patch Yet",
                "",
                "The AAL3 route did not pass enough strict full-matrix gates to justify batch promotion from current artifacts.",
                "Useful code-level safeguards can still be retained, but failed subjects should remain excluded until upstream source-space reprocessing is corrected.",
                "",
            ]
        )
    (run_root / "candidate_patch_decision.md").write_text("\n".join(decision_md), encoding="utf-8")
    append_log(run_root, "\n".join(decision_md))

    update_status(
        run_root,
        phase="complete",
        completed_utc=utc(),
        final_recommendation=final_recommendation,
        next_allowed_action=next_action,
        latest_finding=f"Closeout complete: {final_recommendation}",
    )
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
