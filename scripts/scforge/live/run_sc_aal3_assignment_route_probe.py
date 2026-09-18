#!/usr/bin/env python3
"""Scratch-only AAL3 assignment-policy probe.

Route 3 keeps the best available AAL3-B0 parcellation for a subject and changes
only the tck2connectome streamline-to-node assignment policy. This targets
subjects where label survival is acceptable but valid zero rows remain high.
No production files are overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from scripts.scforge.live.run_sc_aal3_source_contract_probe import (
    DERIV,
    MRTRIX,
    TCK2CONNECTOME,
    aal3_valid_labels,
    discover_inputs,
    label_stats,
    matrix_stats,
    mrtrix_thread_args,
    run,
    write_csv,
    write_json,
)

OUT_PARENT = DERIV / "qc" / "sc_matrix_qc" / "aal3_route3_assignment_probe"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_status(root: Path, **updates: Any) -> None:
    status = read_json(root / "status.json")
    status.update(updates)
    status["last_update_utc"] = utc()
    write_json(root / "status.json", status)


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def label_count(path: Path, valid_labels: list[int]) -> int:
    return int(fnum(label_stats(path, valid_labels).get("n_labels"), 0))


def route1_probe_root(route1_root: Path, sid: str) -> Path:
    return Path("/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc/aal3_source_contract_probe") / f"{route1_root.name}_{sid}"


def route2_probe_root(route2_root: Path, sid: str) -> Path:
    return Path("/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc/aal3_route2_bbr_probe") / f"{route2_root.name}_{sid}"


def collect_candidate_parcellations(sid: str, route1_root: Path, route2_root: Path | None, valid_labels: list[int]) -> list[dict[str, Any]]:
    inputs = discover_inputs(sid)
    candidates: list[dict[str, Any]] = []
    if inputs.get("production_aal_b0"):
        candidates.append({"candidate": "route3_production_parcellation", "path": inputs["production_aal_b0"], "source": "production_AAL_b0"})

    r1 = route1_probe_root(route1_root, sid) / "parc" / "AAL3_source_contract_B0.nii.gz"
    if r1.exists():
        candidates.append({"candidate": "route3_source_contract_parcellation", "path": r1, "source": "route1_source_contract"})

    if route2_root:
        r2 = route2_probe_root(route2_root, sid)
        for row in read_csv(r2 / "label_survival_summary.csv"):
            stage = row.get("stage", "")
            if not stage.startswith("route2_"):
                continue
            # Common route2 output names are deterministic under parc/<stage>/.
            possible = sorted((r2 / "parc").glob(f"{stage}/*AAL3_b0.nii.gz")) + sorted((r2 / "parc").glob(f"{stage}/*AAL_t1_AAL3_b0.nii.gz"))
            if possible:
                candidates.append({"candidate": f"route3_{stage}_parcellation", "path": possible[-1], "source": stage})

    unique: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        path = Path(cand["path"])
        if not path.exists():
            continue
        stats = label_stats(path, valid_labels)
        cand = dict(cand)
        cand["labels"] = int(fnum(stats.get("n_labels"), 0))
        cand["label_stats"] = stats
        unique[str(path)] = cand
    out = list(unique.values())
    out.sort(key=lambda c: (int(c["labels"]), 1 if c["source"] != "production_AAL_b0" else 0), reverse=True)
    return out


def assignment_args(name: str) -> list[str]:
    table = {
        "allvoxels": ["-assignment_all_voxels"],
        "endvox": ["-assignment_end_voxels"],
        "reverse0": ["-assignment_reverse_search", "0"],
        "reverse5": ["-assignment_reverse_search", "5"],
        "reverse10": ["-assignment_reverse_search", "10"],
        "reverse80": ["-assignment_reverse_search", "80"],
        "radial4": ["-assignment_radial_search", "4"],
        "radial6": ["-assignment_radial_search", "6"],
        "radial8": ["-assignment_radial_search", "8"],
        "radial16": ["-assignment_radial_search", "16"],
        "radial24": ["-assignment_radial_search", "24"],
        "forward120": ["-assignment_forward_search", "120"],
        "forward200": ["-assignment_forward_search", "200"],
    }
    if name not in table:
        raise ValueError(f"Unknown assignment variant: {name}")
    return table[name]


def is_long_forward_variant(name: str) -> bool:
    return name.startswith("forward")


def best_completed_for_candidate(matrix_rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    rows = [
        row for row in matrix_rows
        if row.get("candidate") == candidate
        and row.get("metric") == "fd_sum"
        and int(fnum(row.get("read_ok"), 0)) == 1
    ]
    if not rows:
        return {}
    rows.sort(key=lambda row: (-fnum(row.get("valid_zero_rows"), 9999), fnum(row.get("valid_density"), -1)))
    return rows[-1]


def matrix_pass_like(row: dict[str, Any], baseline_fd: dict[str, Any]) -> bool:
    density = fnum(row.get("valid_density"), 0)
    zero = fnum(row.get("valid_zero_rows"), 9999)
    base_density = fnum(baseline_fd.get("valid_density"), 0)
    base_zero = fnum(baseline_fd.get("valid_zero_rows"), 9999)
    return density >= 0.10 and zero <= 15 and density >= base_density and zero <= base_zero


def best_candidate(matrix_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]], baseline_fd: dict[str, Any]) -> dict[str, Any]:
    labels = {r.get("stage"): fnum(r.get("n_labels"), 0) for r in label_rows}
    base_density = fnum(baseline_fd.get("valid_density"), 0)
    base_zero = fnum(baseline_fd.get("valid_zero_rows"), 9999)
    candidates = [
        row for row in matrix_rows
        if row.get("candidate") != "production_AAL3"
        and row.get("metric") == "fd_sum"
        and int(fnum(row.get("read_ok"), 0)) == 1
    ]

    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        n_labels = labels.get(row.get("label_stage"), 0)
        density = fnum(row.get("valid_density"), -1)
        zero = fnum(row.get("valid_zero_rows"), 9999)
        safe = 1.0 if density >= base_density and zero <= base_zero else 0.0
        return (safe, n_labels, -zero, density)

    candidates.sort(key=key, reverse=True)
    return candidates[0] if candidates else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--route1-root", type=Path, required=True)
    parser.add_argument("--route2-root", type=Path, default=None)
    parser.add_argument("--variants", nargs="*", default=["reverse0", "reverse80", "radial8", "forward120"])
    parser.add_argument("--min-labels-for-connectome", type=int, default=160)
    parser.add_argument("--max-parcellations", type=int, default=2)
    parser.add_argument("--no-early-stop", action="store_true", help="Run every requested assignment variant even when earlier variants are already far from QC.")
    parser.add_argument("--skip-long-forward-zero-gt", type=int, default=50, help="Skip forward-search variants when the best quick variant still has more valid zero rows than this.")
    parser.add_argument("--skip-long-forward-density-lt", type=float, default=0.02, help="Skip forward-search variants when quick variants are both sparse and below this valid density.")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PATH"] = f"{MRTRIX}:" + env.get("PATH", "")

    run_root = OUT_PARENT / args.tag
    run_root.mkdir(parents=True, exist_ok=True)
    (OUT_PARENT / "latest_tag.txt").write_text(args.tag, encoding="utf-8")
    if (run_root / "decision.json").exists():
        print(run_root)
        return 0
    write_json(run_root / "status.json", {"sid": args.sid, "tag": args.tag, "phase": "starting", "started_utc": utc(), "variants": args.variants})

    inputs = discover_inputs(args.sid)
    missing = [key for key in ("tracks", "weights", "production_fd_sum") if not inputs.get(key)]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    valid_labels = aal3_valid_labels()

    label_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    baseline_fd = {"candidate": "production_AAL3", "variant": "production", "metric": "fd_sum", **matrix_stats(inputs["production_fd_sum"], valid_labels)}
    matrix_rows.append(baseline_fd)
    if inputs.get("production_aal_b0"):
        label_rows.append({"stage": "production_AAL_b0", **label_stats(inputs["production_aal_b0"], valid_labels)})
    write_csv(run_root / "matrix_candidate_summary.csv", matrix_rows)

    parc_candidates = collect_candidate_parcellations(args.sid, args.route1_root, args.route2_root, valid_labels)
    parc_candidates = [c for c in parc_candidates if int(c["labels"]) >= args.min_labels_for_connectome][: args.max_parcellations]
    for cand in parc_candidates:
        label_rows.append({"stage": cand["candidate"], **cand["label_stats"]})
    write_csv(run_root / "label_survival_summary.csv", label_rows)
    if not parc_candidates:
        write_json(
            run_root / "decision.json",
            {
                "decision": "AAL3_ROUTE3_ASSIGNMENT_NO_LABEL_CANDIDATE",
                "best": {},
                "baseline_fd_sum": baseline_fd,
                "reason": f"no parcellation had >= {args.min_labels_for_connectome} labels",
            },
        )
        update_status(run_root, phase="complete", completed_utc=utc(), decision="AAL3_ROUTE3_ASSIGNMENT_NO_LABEL_CANDIDATE")
        print(run_root)
        return 0

    total_jobs = len(parc_candidates) * len(args.variants)
    done = 0
    skipped_rows: list[dict[str, Any]] = []
    conn = run_root / "connectomes"
    logs = run_root / "logs"
    conn.mkdir(exist_ok=True)
    for cand in parc_candidates:
        for variant in args.variants:
            update_status(run_root, phase="connectome_candidates", active_parcellation=cand["candidate"], active_variant=variant, connectome_jobs_done=done, connectome_jobs_total=total_jobs)
            out_csv = conn / f"SC_AAL3_{args.sid}_{cand['candidate']}__{variant}__fd_sum.csv"
            assign_csv = conn / f"assignments_{cand['candidate']}__{variant}__fd_sum.csv"
            stats = matrix_stats(out_csv, valid_labels)
            if int(fnum(stats.get("read_ok"), 0)) != 1 or not assign_csv.exists() or assign_csv.stat().st_size <= 0:
                quick_best = best_completed_for_candidate(matrix_rows, cand["candidate"])
                quick_zero = fnum(quick_best.get("valid_zero_rows"), 9999) if quick_best else 9999
                quick_density = fnum(quick_best.get("valid_density"), 0) if quick_best else 0
                if (
                    not args.no_early_stop
                    and is_long_forward_variant(variant)
                    and quick_best
                    and (
                        quick_zero > args.skip_long_forward_zero_gt
                        or (quick_zero > 30 and quick_density < args.skip_long_forward_density_lt)
                    )
                ):
                    skipped_rows.append(
                        {
                            "candidate": cand["candidate"],
                            "variant": variant,
                            "reason": (
                                f"skipped long forward search: best quick zero rows "
                                f"{quick_best.get('valid_zero_rows')} > {args.skip_long_forward_zero_gt}"
                            ),
                            "best_quick_variant": quick_best.get("variant", ""),
                            "best_quick_density": quick_best.get("valid_density", ""),
                            "best_quick_zero_rows": quick_best.get("valid_zero_rows", ""),
                            "skipped_utc": utc(),
                        }
                    )
                    write_csv(run_root / "skipped_assignment_variants.csv", skipped_rows)
                    done += 1
                    continue
                cmd: list[str | Path] = [
                    TCK2CONNECTOME,
                    inputs["tracks"],
                    cand["path"],
                    out_csv,
                    "-symmetric",
                    "-zero_diagonal",
                    *mrtrix_thread_args(env),
                    *assignment_args(variant),
                    "-out_assignments",
                    assign_csv,
                    "-stat_edge",
                    "sum",
                    "-tck_weights_in",
                    str(inputs["weights"]),
                ]
                run(cmd, logs / f"tck2connectome_{cand['candidate']}__{variant}__fd_sum.log", env)
                stats = matrix_stats(out_csv, valid_labels)
            done += 1
            matrix_rows.append({"candidate": cand["candidate"], "label_stage": cand["candidate"], "variant": variant, "metric": "fd_sum", "assignment_file": str(assign_csv), **stats})
            write_csv(run_root / "matrix_candidate_summary.csv", matrix_rows)
            if not args.no_early_stop and matrix_pass_like(matrix_rows[-1], baseline_fd):
                skipped_rows.append(
                    {
                        "candidate": cand["candidate"],
                        "variant": "*remaining*",
                        "reason": "stopped after matrix-level final-QC-like candidate; assignment QC will be checked by scorer",
                        "best_quick_variant": variant,
                        "best_quick_density": stats.get("valid_density", ""),
                        "best_quick_zero_rows": stats.get("valid_zero_rows", ""),
                        "skipped_utc": utc(),
                    }
                )
                write_csv(run_root / "skipped_assignment_variants.csv", skipped_rows)
                break

    best = best_candidate(matrix_rows, label_rows, baseline_fd)
    label_stage = str(best.get("label_stage") or "")
    best_label = next((r for r in label_rows if r.get("stage") == label_stage), {})
    baseline_density = fnum(baseline_fd.get("valid_density"), 0)
    best_density = fnum(best.get("valid_density"), 0)
    baseline_zero = int(fnum(baseline_fd.get("valid_zero_rows"), 9999))
    best_zero = int(fnum(best.get("valid_zero_rows"), 9999))
    decision = {
        "decision": "AAL3_ROUTE3_ASSIGNMENT_CANDIDATE" if best else "AAL3_ROUTE3_ASSIGNMENT_NO_CANDIDATE",
        "best": best,
        "baseline_fd_sum": baseline_fd,
        "source_label_stage": label_stage,
        "source_labels": best_label.get("n_labels", ""),
        "source_label_survival": best_label.get("expected_label_survival_fraction", ""),
        "baseline_density": baseline_density,
        "best_density": best_density,
        "density_delta": best_density - baseline_density,
        "baseline_zero_rows": baseline_zero,
        "best_zero_rows": best_zero,
        "zero_rows_delta": best_zero - baseline_zero,
    }
    write_json(run_root / "decision.json", decision)
    update_status(run_root, phase="complete", completed_utc=utc(), decision=decision["decision"], connectome_jobs_done=total_jobs, connectome_jobs_total=total_jobs)
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
