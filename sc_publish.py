#!/usr/bin/env python3
"""Publish a workflow run's matrices where the analysis expects to find them.

The imaging workflow writes each subject's matrices inside its run tree:

    <run_root>/subjects/<unit>/07_connectome/matrices/<metric>.csv

The analysis reads a flat directory with one file per subject and metric:

    <connectomes_dir>/SC_AAL166_<subject>_I<image_id>_<metric>.csv

This is the bridge between the two, and it is the only place the two naming
conventions meet. It refuses to publish anything that is not a square matrix
of the expected node count, and it writes a provenance sidecar per subject
recording the run, the recipe and the SHA-256 of every file it published, so a
matrix in the analysis directory can always be traced back to the run that
made it.

    python sc_publish.py --run-root <run_root>
    python sc_publish.py --run-root <run_root> --dry-run
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sc_config  # noqa: E402

METRICS = (
    "count", "fd_sum", "count_invnodevol", "len_mean", "invlen_mean",
    "fa_mean", "md_mean", "rd_mean", "ad_mean",
)
DEFAULT_NODES = 166


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_unit(unit: str) -> tuple[str, str]:
    """``005_S_6393_I1630993`` -> ``('005_S_6393', '1630993')``.

    The unit id is the subject id with the DWI image id appended, which is how
    the workflow keeps two scans of one subject apart.
    """
    subject, _, image = unit.rpartition("_I")
    if not subject or not image:
        raise ValueError(f"unit id does not end in _I<image id>: {unit!r}")
    return subject, image


def check_matrix(path: Path, nodes: int) -> tuple[int, int]:
    """Return the shape, raising if the file is not a square numeric matrix."""
    with path.open(newline="", encoding="utf-8") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "," if sample.count(",") >= sample.count(" ") else " "
        rows = [row for row in csv.reader(handle, delimiter=delimiter) if row]
    if not rows:
        raise ValueError(f"{path.name} is empty")
    widths = {len([cell for cell in row if cell != ""]) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"{path.name} is ragged: row widths {sorted(widths)}")
    width = widths.pop()
    if len(rows) != width:
        raise ValueError(f"{path.name} is {len(rows)}x{width}, not square")
    if nodes and width != nodes:
        raise ValueError(f"{path.name} is {width}x{width}, expected {nodes}x{nodes}")
    for row in rows[:2]:
        for cell in row[:2]:
            float(cell)
    return len(rows), width


def units_in(run_root: Path) -> list[str]:
    subjects = run_root / "subjects"
    if not subjects.is_dir():
        return []
    return sorted(p.name for p in subjects.iterdir() if p.is_dir())


def publish_unit(run_root: Path, unit: str, destination: Path, *,
                 nodes: int, dry_run: bool, overwrite: bool) -> dict:
    subject, image = split_unit(unit)
    source_dir = run_root / "subjects" / unit / "07_connectome" / "matrices"
    record: dict = {"unit": unit, "subject_id": subject, "image_id": image,
                    "published": [], "missing": [], "rejected": []}
    for metric in METRICS:
        source = source_dir / f"{metric}.csv"
        if not source.is_file():
            record["missing"].append(metric)
            continue
        try:
            check_matrix(source, nodes)
        except (ValueError, IndexError) as error:
            record["rejected"].append({"metric": metric, "reason": str(error)})
            continue
        target = destination / f"SC_AAL166_{subject}_I{image}_{metric}.csv"
        if target.exists() and not overwrite:
            record["rejected"].append(
                {"metric": metric, "reason": f"{target.name} already exists (use --overwrite)"}
            )
            continue
        if not dry_run:
            destination.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".csv.partial")
            shutil.copyfile(source, temporary)
            temporary.replace(target)
        record["published"].append(
            {"metric": metric, "file": target.name,
             "sha256": sha256_file(source), "bytes": source.stat().st_size}
        )
    return record


def write_provenance(run_root: Path, destination: Path, record: dict,
                     *, dry_run: bool) -> Path | None:
    if dry_run or not record["published"]:
        return None
    context_path = run_root / "contract" / "run_context.json"
    context = json.loads(context_path.read_text()) if context_path.is_file() else {}
    sidecar = destination / f"SC_AAL_{record['subject_id']}_I{record['image_id']}_generation_provenance.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "published_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "subject_id": record["subject_id"],
                "image_id": record["image_id"],
                "unit": record["unit"],
                "run_root": str(run_root),
                "run_id": context.get("run_id"),
                "recipe_id": context.get("recipe_id"),
                "atlas": "AAL3 166-node",
                "files": record["published"],
                "missing_metrics": record["missing"],
            },
            indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return sidecar


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="destination directory (default: $SC_CONNECTOMES_DIR)")
    parser.add_argument("--units", nargs="*", default=None,
                        help="publish only these units (default: every unit in the run)")
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES,
                        help=f"expected node count (default: {DEFAULT_NODES}; 0 to skip)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace files that are already published")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        print(f"run root not found: {run_root}", file=sys.stderr)
        return 2
    destination = (args.out or sc_config.paths().connectomes_dir).expanduser()
    units = args.units or units_in(run_root)
    if not units:
        print(f"no subject directories under {run_root / 'subjects'}", file=sys.stderr)
        return 2

    print(f"run   : {run_root}")
    print(f"out   : {destination}")
    print(f"units : {len(units)}")
    print()

    complete = incomplete = 0
    for unit in units:
        record = publish_unit(run_root, unit, destination, nodes=args.nodes,
                              dry_run=args.dry_run, overwrite=args.overwrite)
        write_provenance(run_root, destination, record, dry_run=args.dry_run)
        published = len(record["published"])
        if published == len(METRICS):
            complete += 1
            print(f"  {unit:<28} {published}/{len(METRICS)} matrices")
        else:
            incomplete += 1
            print(f"  {unit:<28} {published}/{len(METRICS)} matrices  <- incomplete")
            if record["missing"]:
                print(f"      missing : {', '.join(record['missing'])}")
            for entry in record["rejected"]:
                print(f"      rejected: {entry['metric']}: {entry['reason']}")

    print()
    verb = "would publish" if args.dry_run else "published"
    print(f"{verb}: {complete} complete, {incomplete} incomplete")
    if not args.dry_run and complete:
        print(f"next: python -m connectome_analysis.run_analysis --stage cohort")
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
