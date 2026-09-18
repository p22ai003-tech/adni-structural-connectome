#!/usr/bin/env python3
"""Build a diagnosis-blind HCP-MMP1 quality and rerun triage manifest.

This audit does not modify production connectomes. It combines:
  * local 379-node count and weighted-matrix invariants;
  * tractogram provenance and endpoint-assignment support;
  * atlas left/right volume symmetry; and
  * a separately generated count-only screen of distinct S3 tractograms.

The diagnosis column is reported only after classification and is never used
to assign a quality lane.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


N_NODES = 379
HUB_IDS = (365, 374, 361, 370, 366, 375)
MATRIX_SUFFIXES = {
    "count": "count",
    "fd_sum": "fd_sum",
    "len_mean": "len_mean",
    "invlen_mean": "invlen_mean",
    "fa_mean": "fa_mean",
    "md_mean": "md_mean",
    "rd_mean": "rd_mean",
    "ad_mean": "ad_mean",
    "count_invnodevol": "count_invnodevol",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive-screen",
        type=Path,
        required=True,
        help="CSV from the count-only S3 tractogram screen.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/ec2-user/exp/research_audit/outputs/hcp_rerun_triage_v1"
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def tract_header(path: Path) -> str:
    with path.open("rb") as handle:
        raw = handle.read(16384)
    return raw.split(b"END\n", 1)[0].decode("utf-8", "replace")


def track_count(header: str) -> int:
    match = re.search(r"^count:\s*(\d+)", header, re.MULTILINE)
    return int(match.group(1)) if match else 3_000_000


def provenance_class(header: str) -> str:
    if "command_history: variable" in header:
        return "variable"
    if "/Users/sabeesh/SeagateHub/" in header and " -act " in header:
        return "explicit_mac_act"
    if "regen216_0615T045342Z" in header:
        return "regen216_noact"
    if " -act " in header:
        return "other_explicit_act"
    if " -seed_dynamic " in header or "seed_dynamic:" in header:
        return "other_noact_dynamic"
    return "other_or_unknown"


def load_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def audit_weighted_matrices(
    sid: str, root: Path
) -> tuple[bool, list[str], dict[str, float]]:
    arrays: dict[str, np.ndarray] = {}
    reasons: list[str] = []
    support: dict[str, float] = {}

    for metric, suffix in MATRIX_SUFFIXES.items():
        path = root / f"SC_HCPMMP1_{sid}_{suffix}.csv"
        if not path.exists():
            reasons.append(f"{metric}:missing")
            continue
        try:
            matrix = load_matrix(path)
        except Exception:
            reasons.append(f"{metric}:read")
            continue
        arrays[metric] = matrix
        if matrix.shape != (N_NODES, N_NODES):
            reasons.append(f"{metric}:shape")
            continue
        if not np.isfinite(matrix).all():
            reasons.append(f"{metric}:nonfinite")
        if not np.allclose(matrix, matrix.T, atol=1e-6, rtol=1e-6):
            reasons.append(f"{metric}:asymmetric")
        if not np.allclose(np.diag(matrix), 0.0, atol=1e-8):
            reasons.append(f"{metric}:diagonal")
        if metric != "count" and float(np.nanmin(matrix)) < -1e-10:
            reasons.append(f"{metric}:negative")

    fa = arrays.get("fa_mean")
    if fa is not None and fa.shape == (N_NODES, N_NODES):
        nonzero = fa[fa != 0]
        if nonzero.size and (
            float(nonzero.min()) < -1e-6 or float(nonzero.max()) > 1.000001
        ):
            reasons.append("fa_mean:range")

    count = arrays.get("count")
    if count is not None and count.shape == (N_NODES, N_NODES):
        count_support = np.triu(count > 0, 1)
        denominator = max(int(count_support.sum()), 1)
        for metric, matrix in arrays.items():
            if metric == "count" or matrix.shape != (N_NODES, N_NODES):
                continue
            fraction = float(
                (np.triu(matrix != 0, 1) & count_support).sum() / denominator
            )
            support[metric] = fraction
            if fraction < 0.95:
                reasons.append(f"{metric}:support<{fraction:.3f}")

    reasons = sorted(set(reasons))
    return not reasons, reasons, support


def atlas_symmetry(subject_dir: Path) -> tuple[bool, list[str], float]:
    volume_csv = subject_dir / "node_volumes.csv"
    if not volume_csv.exists():
        return False, ["atlas:node_volumes_missing"], float("nan")

    voxels: dict[int, int] = {}
    for row in read_csv(volume_csv):
        voxels[int(row["node_id"])] = int(row["n_voxels"])

    left = sum(voxels.get(node, 0) for node in range(1, 181))
    right = sum(voxels.get(node, 0) for node in range(181, 361))
    cortical_ratio = left / max(right, 1)
    reasons: list[str] = []
    if not 0.6 <= cortical_ratio <= 1.67:
        reasons.append("atlas:cortical_lr")
    for name, left_id, right_id in (
        ("thal", 361, 370),
        ("hipp", 365, 374),
        ("put", 363, 372),
    ):
        ratio = voxels.get(left_id, 0) / max(voxels.get(right_id, 0), 1)
        if not 0.35 <= ratio <= 2.85:
            reasons.append(f"atlas:{name}_lr")
    return not reasons, reasons, cortical_ratio


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort_path = Path("/home/ec2-user/exp/hcp_analysis/cohort_hcp.csv")
    qc_path = Path("/home/ec2-user/exp/hcp_analysis/qc_density.csv")
    track_root = Path("/data/derivatives/tracks")
    matrix_root = Path("/data/derivatives/connectomes_hcp_fs")
    parc_root = Path("/data/derivatives/parc_hcpmmp1")

    cohort = {row["fsid"]: row["group"] for row in read_csv(cohort_path)}
    local_rows = {row["fsid"]: row for row in read_csv(qc_path)}
    archive_rows = {row["sid"]: row for row in read_csv(args.archive_screen)}

    if len(archive_rows) != 86:
        raise RuntimeError(
            f"Expected 86 distinct archive candidates, found {len(archive_rows)}"
        )
    archive_failures = [
        sid
        for sid, row in archive_rows.items()
        if row.get("status") != "PASS_STRICT"
    ]
    if archive_failures:
        raise RuntimeError(f"Archive screen has failures: {archive_failures}")

    manifest: list[dict[str, object]] = []
    for sid, group in cohort.items():
        row: dict[str, object] = {"fsid": sid, "group": group}
        if sid not in local_rows:
            row.update(
                {
                    "lane": "CORRECTED_RERUN",
                    "recommended_action": "FASTSURFER_RETRY_THEN_HCP_BUILD",
                    "reasons": "missing_hcp_connectome",
                    "provenance_class": "missing",
                    "effective_count_density_for_screen": "",
                    "meets_raw_density_0_60": False,
                    "meets_raw_density_0_70": False,
                }
            )
            manifest.append(row)
            continue

        track_path = track_root / sid / "tracks_final_3000k.tck"
        count_path = matrix_root / f"SC_HCPMMP1_{sid}_count.csv"
        header = tract_header(track_path)
        provenance = provenance_class(header)
        n_tracks = track_count(header)
        count = load_matrix(count_path)
        strength = count.sum(axis=1)
        nodes_present = int((strength > 0).sum())
        density = float((count > 0).sum() / (N_NODES * (N_NODES - 1)))
        assigned = float(count.sum() / 2.0)
        assignment_fraction = assigned / max(n_tracks, 1)
        hub_degrees = [float(strength[node - 1] * 2.0) for node in HUB_IDS]
        hub_pass = all(value > 0 for value in hub_degrees)

        atlas_pass, atlas_reasons, cortical_lr = atlas_symmetry(parc_root / sid)
        weighted_pass, weighted_reasons, support = audit_weighted_matrices(
            sid, matrix_root
        )

        count_reasons: list[str] = []
        if count.shape != (N_NODES, N_NODES):
            count_reasons.append("count:shape")
        if not np.isfinite(count).all():
            count_reasons.append("count:nonfinite")
        if not np.allclose(count, count.T, atol=1e-6, rtol=1e-6):
            count_reasons.append("count:asymmetric")
        if not np.allclose(np.diag(count), 0.0, atol=1e-8):
            count_reasons.append("count:diagonal")
        if nodes_present < 360:
            count_reasons.append("nodes_present<360")
        if density < 0.15:
            count_reasons.append("density<0.15")
        if assignment_fraction < 0.50:
            count_reasons.append("assignment_fraction<0.50")
        if not hub_pass:
            count_reasons.append("hub_support")

        local_high_confidence = (
            provenance == "explicit_mac_act"
            and not count_reasons
            and atlas_pass
            and weighted_pass
        )

        if local_high_confidence:
            lane = "KEEP_HIGH_CONFIDENCE"
            recommended_action = "KEEP_NO_RERUN"
            reasons = []
        elif sid in archive_rows:
            lane = "ARCHIVE_RECOVERY"
            recommended_action = "RESTORE_TRACK_REBUILD_MATCHED_WEIGHTS"
            reasons = [
                "distinct_s3_regen216_track_passed_strict_count_screen",
                "rebuild_matching_sift_and_scalar_samples",
                "noact_protocol_requires_sensitivity_gate",
            ]
        else:
            lane = "CORRECTED_RERUN"
            reasons = []
            if provenance != "explicit_mac_act":
                reasons.append(f"provenance:{provenance}")
            reasons.extend(count_reasons)
            reasons.extend(atlas_reasons)
            reasons.extend(weighted_reasons)
            if (
                provenance == "explicit_mac_act"
                and not count_reasons
                and atlas_pass
                and not weighted_pass
            ):
                recommended_action = "TENSOR_WEIGHTED_MATRIX_REPAIR"
            elif provenance == "explicit_mac_act":
                recommended_action = "HCP_REMAP_PROBE_THEN_ESCALATE"
            else:
                recommended_action = "CORRECTED_PRETRACT_AND_TRACK_REBUILD"

        archive = archive_rows.get(sid, {})
        effective_density = (
            float(archive["density"]) if archive else density
        )
        row.update(
            {
                "lane": lane,
                "recommended_action": recommended_action,
                "reasons": ";".join(sorted(set(reasons))),
                "provenance_class": provenance,
                "local_density": density,
                "local_nodes_present": nodes_present,
                "local_assigned_streamlines": assigned,
                "local_assignment_fraction": assignment_fraction,
                "local_min_hub_degree": min(hub_degrees),
                "atlas_cortical_lr_voxel_ratio": cortical_lr,
                "atlas_symmetry_pass": atlas_pass,
                "weighted_matrix_pass": weighted_pass,
                "weighted_matrix_reasons": ";".join(weighted_reasons),
                "min_weighted_edge_support": min(support.values())
                if support
                else float("nan"),
                "archive_density": archive.get("density", ""),
                "archive_nodes_present": archive.get("nodes", ""),
                "archive_assigned_streamlines": archive.get("assigned", ""),
                "archive_min_hub_degree": archive.get("minhub", ""),
                "effective_count_density_for_screen": effective_density,
                "meets_raw_density_0_60": effective_density >= 0.60,
                "meets_raw_density_0_70": effective_density >= 0.70,
            }
        )
        manifest.append(row)

    manifest.sort(key=lambda row: str(row["fsid"]))
    fields = list(manifest[0].keys())
    manifest_path = output_dir / "hcp_rerun_triage_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)

    lane_counts = Counter(str(row["lane"]) for row in manifest)
    group_by_lane = {
        lane: dict(
            Counter(
                str(row["group"]) for row in manifest if str(row["lane"]) == lane
            )
        )
        for lane in sorted(lane_counts)
    }
    summary = {
        "schema_version": "1.0.0",
        "classification_is_diagnosis_blind": True,
        "n_total": len(manifest),
        "lane_counts": dict(lane_counts),
        "recommended_action_counts": dict(
            Counter(str(row["recommended_action"]) for row in manifest)
        ),
        "group_by_lane_post_classification_audit_only": group_by_lane,
        "thresholds": {
            "nodes_present_min": 360,
            "node_coverage_fraction": 360 / 379,
            "density_min": 0.15,
            "assignment_fraction_min": 0.50,
            "all_six_hubs_required": True,
            "atlas_symmetry_required": True,
            "all_weighted_matrix_invariants_required": True,
        },
        "archive_screen": {
            "n": len(archive_rows),
            "pass_strict": len(archive_rows),
            "source": str(args.archive_screen),
            "note": (
                "Count-only evidence. Recompute SIFT2 and scalar samples before "
                "weighted-matrix use."
            ),
        },
        "raw_density_target_scenarios_after_archive_substitution": {
            "at_least_0_60": sum(
                bool(row["meets_raw_density_0_60"]) for row in manifest
            ),
            "below_0_60_or_missing": sum(
                not bool(row["meets_raw_density_0_60"]) for row in manifest
            ),
            "at_least_0_70": sum(
                bool(row["meets_raw_density_0_70"]) for row in manifest
            ),
            "below_0_70_or_missing": sum(
                not bool(row["meets_raw_density_0_70"]) for row in manifest
            ),
        },
        "corrected_pipeline_queue": lane_counts["CORRECTED_RERUN"],
        "uniform_protocol_queue_including_archive": (
            lane_counts["CORRECTED_RERUN"] + lane_counts["ARCHIVE_RECOVERY"]
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    for lane, filename in (
        ("KEEP_HIGH_CONFIDENCE", "keep_high_confidence_214.txt"),
        ("ARCHIVE_RECOVERY", "archive_recovery_86.txt"),
        ("CORRECTED_RERUN", "corrected_rerun_230.txt"),
    ):
        subjects = [
            str(row["fsid"]) for row in manifest if str(row["lane"]) == lane
        ]
        (output_dir / filename).write_text("\n".join(subjects) + "\n")

    for action, filename in (
        ("RESTORE_TRACK_REBUILD_MATCHED_WEIGHTS", "archive_restore_weights_86.txt"),
        ("HCP_REMAP_PROBE_THEN_ESCALATE", "hcp_remap_probe_111.txt"),
        ("CORRECTED_PRETRACT_AND_TRACK_REBUILD", "corrected_track_rebuild_116.txt"),
        ("TENSOR_WEIGHTED_MATRIX_REPAIR", "tensor_weighted_repair_2.txt"),
        ("FASTSURFER_RETRY_THEN_HCP_BUILD", "fastsurfer_retry_1.txt"),
    ):
        subjects = [
            str(row["fsid"])
            for row in manifest
            if str(row["recommended_action"]) == action
        ]
        (output_dir / filename).write_text("\n".join(subjects) + "\n")

    readme = f"""# HCP-MMP1 rerun triage V1

This diagnosis-blind audit replaces the earlier residual count of 132.

**Important:** `KEEP_HIGH_CONFIDENCE` means that the existing output passes the
minimum technical validity gate; it does not mean that it reaches a raw density
target of 0.60 or 0.70. Density is resolution-dependent. AAL166 contains 13,695
possible undirected edges, whereas HCP379 contains 71,631. The legacy AAL median
density of about 0.741 corresponds to about 10,148 nonzero edges; the same
absolute edge count in HCP379 is density about 0.142.

## Decision

- **KEEP_HIGH_CONFIDENCE: {lane_counts['KEEP_HIGH_CONFIDENCE']}**
- **ARCHIVE_RECOVERY: {lane_counts['ARCHIVE_RECOVERY']}**
- **CORRECTED_RERUN: {lane_counts['CORRECTED_RERUN']}**

The local keep gate requires explicit ACT provenance, at least 360/379 connected
nodes, density at least 0.15, at least 50% endpoint assignment, all six locked
hubs, atlas left/right volume symmetry, and valid support/ranges for all nine
matrices. The 0.15 density floor is conservative relative to the diagnosis-blind
5th percentile (~0.16) of explicit-ACT cases with at least 360 connected nodes.

All 86 distinct S3 recovery tracks passed the strict count-only screen with
376-379 connected nodes. They remain a separate lane because their recovered
tracks used no-ACT dynamic seeding and the archived auxiliary SIFT2/TSF files
were not proven to be paired with those tracks. Their weighted matrices must be
recomputed before use.

The corrected-rerun list contains the one subject without an HCP connectome.
It is execution-triaged into 111 cheap HCP remap probes, 116 corrected
pre-tractography/tractography rebuilds, two tensor/weighted-matrix repairs, and
one FastSurfer retry. A remap probe escalates to tractography only if the
diagnosis-blind HCP count gate still fails.

If a single uniform tractography protocol is required for the final inferential
cohort, the archive lane must also be rerun or retained only as sensitivity
evidence; that uniform-protocol queue is
{summary['uniform_protocol_queue_including_archive']} subjects.

As a separate engineering scenario, after substituting the 86 count-validated
archive tracks, only
{summary['raw_density_target_scenarios_after_archive_substitution']['at_least_0_60']}
subjects reach raw HCP density >=0.60, and only
{summary['raw_density_target_scenarios_after_archive_substitution']['at_least_0_70']}
reach >=0.70. Those thresholds should not be adopted as primary QC until a
streamline-count convergence canary shows that corrected ACT reaches them
without excessive false-positive connections.
"""
    (output_dir / "README.md").write_text(readme)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
