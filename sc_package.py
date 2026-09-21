#!/usr/bin/env python
"""Build the distributable pipeline package.

    python sc_package.py                    # build dist/adni-sc-pipeline-<rev>.zip
    python sc_package.py --check            # verify only, write nothing
    python sc_package.py --include-audit    # also include the audit trail

What goes in
------------
Only files tracked by git, taken from ``git archive``. Nothing untracked can be
swept in by accident, which matters here because the data volume, the cohort
tables and the acquisition manifests all live inside the working tree and are
excluded by .gitignore.

Two subtrees are then left out by default:

    research_audit/   the audit trail: decisions, evidence and one-off
                      validators. It is provenance, not pipeline, and it names
                      ADNI participants throughout.
    scripts/hcp/      the separate HCP379 work, on a different cohort.

Neither is needed to run either pipeline. ``--include-audit`` keeps them, for an
internal hand-off rather than a distribution.

The safety gate
---------------
ADNI data are governed by a Data Use Agreement and participant identifiers must
not be redistributed. So this refuses to write a package containing anything
that looks like one, or any of the other patterns below. The check runs on the
assembled contents, not on the source tree, because what ships is what matters.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_EXCLUDE = ("research_audit/", "scripts/hcp/")

# Files inside an excluded subtree that the workflow contract still requires.
# Empty now: the two that used to be rescued -- the matrix data dictionary and
# the contract validator -- moved into the shipped tree, where they belong,
# because both are workflow components rather than audit output. The list stays
# because a future exclusion may again cover a file the workflow needs.
KEEP_DESPITE_EXCLUDE: tuple[str, ...] = ()

# Thesis presentations, reports and manuscripts. They are 72 MB of the tracked
# tree and none of it is pipeline code, so a code package should not carry them.
# Markdown under docs/ is kept: those are runbooks.
DOCUMENT_SUFFIXES = {".pptx", ".docx", ".doc", ".ppt", ".pdf", ".xlsx", ".numbers", ".key"}

# Anything matching these must not leave the machine.
FORBIDDEN = [
    (re.compile(rb"\d{3}_S_\d{4,5}"), "ADNI participant identifier"),
    (re.compile(rb"(?i)aws_access_key_id\s*[=:]"), "AWS access key"),
    (re.compile(rb"(?i)aws_secret_access_key\s*[=:]"), "AWS secret key"),
    (re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"), "private key"),
    (re.compile(rb"(?i)\bpassword\s*[=:]\s*[\"'][^\"'\s]{6,}"), "hard-coded password"),
]

# Binary and data extensions that should never appear in a code package.
BANNED_SUFFIXES = {".nii", ".gz", ".mif", ".tck", ".dcm", ".mgz", ".trk",
                   ".sqlite", ".sqlite3", ".db", ".pem", ".key"}
TEXTLIKE = {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".json", ".cfg", ".toml",
            ".ini", ".smk", ".csv", ".tsv", ".ipynb", ".ts", ".tsx", ".js", ".html", ""}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def export_tracked(dest: Path) -> None:
    """git archive gives exactly the tracked tree, and nothing else."""
    tar = dest / "tracked.tar"
    with tar.open("wb") as fh:
        subprocess.run(["git", "archive", "--format=tar", "HEAD"],
                       cwd=PROJECT_ROOT, stdout=fh, check=True)
    shutil.unpack_archive(str(tar), str(dest / "tree"), format="tar")
    tar.unlink()


def scan(root: Path) -> list[tuple[str, str, str]]:
    """Return (path, what, evidence) for everything that must not ship."""
    problems: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        if path.suffix.lower() in BANNED_SUFFIXES and not rel.endswith(".nii.gz.md"):
            # Atlas volumes are atlas-space templates, not participant data.
            if not rel.startswith("atlas/"):
                problems.append((rel, "data file in a code package", path.suffix))
                continue
        if path.suffix.lower() not in TEXTLIKE:
            continue
        try:
            blob = path.read_bytes()
        except OSError:
            continue
        for pattern, what in FORBIDDEN:
            m = pattern.search(blob)
            if m:
                ev = m.group(0).decode("utf8", "replace")[:40]
                problems.append((rel, what, ev))
                break
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "dist")
    ap.add_argument("--name", default="adni-sc-pipeline")
    ap.add_argument("--check", action="store_true", help="verify only; write nothing")
    ap.add_argument("--include-documents", action="store_true",
                    help="keep the thesis presentations and reports (adds ~72 MB)")
    ap.add_argument("--include-audit", action="store_true",
                    help="keep research_audit/ and scripts/hcp/ (internal hand-off)")
    ap.add_argument("--force", action="store_true",
                    help="write the package even if the safety gate finds something")
    args = ap.parse_args(argv)

    if git("status", "--porcelain"):
        print("working tree is not clean; commit first so the package matches a revision.",
              file=sys.stderr)
        return 2
    rev = git("rev-parse", "--short", "HEAD")

    with tempfile.TemporaryDirectory(prefix="sc_pkg_") as tmp:
        stage = Path(tmp)
        export_tracked(stage)
        tree = stage / "tree"

        removed = 0
        if not args.include_audit:
            keep: dict[str, bytes] = {}
            for rel in KEEP_DESPITE_EXCLUDE:
                src = tree / rel
                if src.is_file():
                    keep[rel] = src.read_bytes()
            for sub in DEFAULT_EXCLUDE:
                target = tree / sub
                if target.exists():
                    removed += sum(1 for _ in target.rglob("*") if _.is_file())
                    shutil.rmtree(target)
            for rel, blob in keep.items():
                dest = tree / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(blob)
                removed -= 1

        if not args.include_documents:
            for doc in list(tree.rglob("*")):
                if doc.is_file() and doc.suffix.lower() in DOCUMENT_SUFFIXES:
                    doc.unlink()
                    removed += 1

        files = [p for p in tree.rglob("*") if p.is_file()]
        size = sum(p.stat().st_size for p in files)
        print(f"revision : {rev}")
        print(f"contents : {len(files)} files, {size / 1e6:.1f} MB"
              + (f"   ({removed} excluded)" if removed else ""))

        print("\nsafety gate")
        print("-" * 68)
        problems = scan(tree)
        if problems:
            by_kind: dict[str, int] = {}
            for _, what, _ev in problems:
                by_kind[what] = by_kind.get(what, 0) + 1
            for what, n in sorted(by_kind.items(), key=lambda kv: -kv[1]):
                print(f"  {n:>4}  {what}")
            print("\n  first 15:")
            for rel, what, ev in problems[:15]:
                print(f"    {rel:<58} {what}: {ev}")
            if not args.force:
                print(f"\n{len(problems)} problem(s). Nothing written.")
                print("Fix them, or re-run with --force if every one is a false positive.")
                return 1
            print("\n--force given; writing anyway.")
        else:
            print("  clean: no participant identifiers, credentials or data files.")

        if args.check:
            print("\n--check: nothing written.")
            return 0

        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / f"{args.name}-{rev}.zip"
        root_name = f"{args.name}-{rev}"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for p in sorted(files):
                z.write(p, Path(root_name) / p.relative_to(tree))
        print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
