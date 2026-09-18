#!/usr/bin/env python3
"""Targeted Step 7 post/connectome recovery for EC2-local derivative gaps.

This runner fills post-stage gaps only for subjects that already have final
tracks and complete DTI scalar maps. It is intended for count mismatch recovery,
not for anatomical matrix QC repair.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path("/home/ec2-user/exp")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from connectome_pipeline.connectome_step7 import Step7Config, run_step7_pipeline  # noqa: E402
from connectome_pipeline.pipeline_paths import resolve_pipeline_paths  # noqa: E402
from connectome_pipeline.pipeline_status import _load_subject_group_map  # noqa: E402

SUBJECT_RE = re.compile(r"^(\d{3}_S_\d{4})")
REQUIRED_POST_SUFFIXES = (
    "ALL",
    "count",
    "fd_sum",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
    "count_invnodevol",
)


def subject_id(sid: str) -> str:
    match = SUBJECT_RE.match(sid)
    return match.group(1) if match else sid.split("_I", 1)[0]


def parse_groups(value: str) -> set[str]:
    groups = {x.strip().upper() for x in value.split(",") if x.strip()}
    valid = {"CN", "MCI", "AD"}
    bad = groups - valid
    if bad:
        raise ValueError(f"Unsupported groups: {', '.join(sorted(bad))}")
    return groups


def directory_sids(root: Path, file_name: str) -> set[str]:
    if not root.exists():
        return set()
    return {
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (path / file_name).exists()
    }


def connectome_sids(conn_dir: Path, suffix: str) -> set[str]:
    if not conn_dir.exists():
        return set()
    prefix = "SC_AAL_"
    postfix = f"_{suffix}.csv"
    return {
        path.name[len(prefix) : -len(postfix)]
        for path in conn_dir.glob(f"{prefix}*{postfix}")
        if path.name.startswith(prefix) and path.name.endswith(postfix)
    }


def post_complete(conn_dir: Path, sid: str) -> bool:
    return all((conn_dir / f"SC_AAL_{sid}_{suffix}.csv").exists() for suffix in REQUIRED_POST_SUFFIXES)


def build_inventory(
    deriv_root: Path,
    group_map: dict[str, str],
    target_groups: set[str],
) -> tuple[list[str], dict[str, set[str]], dict[str, str]]:
    tracks = directory_sids(deriv_root / "tracks", "tracks_final_3000k.tck")
    dti_sets = [
        directory_sids(deriv_root / "dti", f"{metric}.mif")
        for metric in ("fa", "md", "rd", "ad")
    ]
    dti = set.intersection(*dti_sets) if dti_sets else set()
    conn_dir = deriv_root / "connectomes"
    universe = tracks | dti | connectome_sids(conn_dir, "ALL")
    complete_post = {sid for sid in universe if post_complete(conn_dir, sid)}
    sid_groups = {sid: group_map.get(subject_id(sid), "") for sid in universe}
    targets = sorted(
        sid
        for sid in tracks & dti
        if sid_groups.get(sid) in target_groups and sid not in complete_post
    )
    sets = {
        "tracks": tracks,
        "dti": dti,
        "complete_post": complete_post,
        "conn_all": connectome_sids(conn_dir, "ALL"),
        "parc": directory_sids(deriv_root / "parc", "AAL_b0.nii.gz"),
    }
    return targets, sets, sid_groups


def copy_if_exists(src: Path, dst_root: Path, deriv_root: Path) -> None:
    if not src.exists():
        return
    try:
        rel = src.relative_to(deriv_root)
    except ValueError:
        rel = Path(src.name)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def backup_existing_target_outputs(deriv_root: Path, run_root: Path, sids: list[str]) -> None:
    backup_root = run_root / "pre_run_backups"
    for sid in sids:
        copy_if_exists(deriv_root / "parc" / sid / "AAL_b0.nii.gz", backup_root, deriv_root)
        copy_if_exists(deriv_root / "tracks" / sid / "assignments_aal.csv", backup_root, deriv_root)
        for path in (deriv_root / "connectomes").glob(f"SC_AAL_{sid}_*.csv"):
            copy_if_exists(path, backup_root, deriv_root)


def group_counts(sids: set[str] | list[str], sid_groups: dict[str, str]) -> dict[str, int]:
    counts = Counter(sid_groups.get(sid, "UNKNOWN") for sid in sids)
    return {group: counts.get(group, 0) for group in ("CN", "MCI", "AD", "UNKNOWN")}


def write_target_inventory(
    run_root: Path,
    deriv_root: Path,
    targets: list[str],
    sets: dict[str, set[str]],
    sid_groups: dict[str, str],
    target_groups: set[str],
    batch_size: int,
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "targets.txt").write_text("\n".join(targets) + ("\n" if targets else ""), encoding="ascii")
    with (run_root / "target_inventory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sid",
                "group",
                "has_tracks",
                "has_dti",
                "has_parc",
                "has_all",
                "post_complete",
            ],
        )
        writer.writeheader()
        for sid in targets:
            writer.writerow(
                {
                    "sid": sid,
                    "group": sid_groups.get(sid, ""),
                    "has_tracks": int(sid in sets["tracks"]),
                    "has_dti": int(sid in sets["dti"]),
                    "has_parc": int(sid in sets["parc"]),
                    "has_all": int(sid in sets["conn_all"]),
                    "post_complete": int(sid in sets["complete_post"]),
                }
            )
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deriv_root": str(deriv_root),
        "target_groups": sorted(target_groups),
        "target_count": len(targets),
        "targets_by_group": group_counts(targets, sid_groups),
        "baseline_complete_post_by_group": group_counts(sets["complete_post"], sid_groups),
        "tracks_by_group": group_counts(sets["tracks"], sid_groups),
        "dti_by_group": group_counts(sets["dti"], sid_groups),
        "batch_size": batch_size,
        "required_post_suffixes": list(REQUIRED_POST_SUFFIXES),
    }
    (run_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--groups", default="CN,MCI", help="Comma-separated groups to recover: CN,MCI,AD")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    target_groups = parse_groups(args.groups)
    paths = resolve_pipeline_paths(create_layout=True)
    group_map = _load_subject_group_map(paths.cohort_dti_csv)
    targets, sets, sid_groups = build_inventory(paths.deriv_root, group_map, target_groups)
    if args.limit and args.limit > 0:
        targets = targets[: args.limit]
    write_target_inventory(args.run_root, paths.deriv_root, targets, sets, sid_groups, target_groups, args.batch_size)

    print(
        f"Connectome gap recovery | {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        flush=True,
    )
    print(f"derivatives: {paths.deriv_root}", flush=True)
    print(
        f"groups={','.join(sorted(target_groups))} | targets={len(targets)} | "
        f"batch_size={args.batch_size} | execute={args.execute}",
        flush=True,
    )
    print("targets_by_group:", group_counts(targets, sid_groups), flush=True)
    print(f"target list: {args.run_root / 'targets.txt'}", flush=True)
    if targets:
        print("preview:", ", ".join(targets[:10]) + (" ..." if len(targets) > 10 else ""), flush=True)
    if not args.execute:
        print("dry-run only; add --execute to run Step 7 post.", flush=True)
        return
    if not targets:
        print("No targets selected.", flush=True)
        return

    if args.backup:
        backup_existing_target_outputs(paths.deriv_root, args.run_root, targets)

    cfg = Step7Config(
        batch_size=args.batch_size,
        force=False,
        aal_transform_route="current_direct",
        assignment_variant="radial4",
        backup_before_replace=True,
        validation_gate=True,
        qc_log_root=args.run_root / "step7_logs",
        runtime_log=args.run_root / "step7_runtime_notes.txt",
        summary_csv=args.run_root / "step7_summary.csv",
    )
    result = run_step7_pipeline(cfg, stage="post", include=targets)
    (args.run_root / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print("Step 7 post gap recovery finished.", flush=True)


if __name__ == "__main__":
    main()
