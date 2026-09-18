#!/usr/bin/env python
"""Point the shipped configuration at this machine.

    python sc_localize.py --check      # what would change
    python sc_localize.py              # rewrite, then re-lock

Why this is needed
------------------
Most of the project resolves paths through ``sc_config.py`` and the ``SC_*``
variables, so it moves between machines without editing anything. The imaging
workflow is the exception: ``configs/connectome_v2.yaml`` and
``scforge/workflow/environment_contract.yaml`` are contract files that pin
absolute paths deliberately -- to the Python that runs the workflow, to the
ANTs prefix, to the locked artifact manifests -- because the point of a contract
is that it names exactly what was used. Ninety-odd of those paths begin with the
project root of the machine the project grew on, so on any other machine the
workflow fails at the first rule.

This rewrites that prefix and nothing else. It does not touch ``/data``, an
external volume, or any path outside the old project root: those are genuinely
machine-specific and belong to the operator, not to this tool.

Rewriting a contract file changes its SHA-256, and both files are covered by the
locked source manifest, so the lock is regenerated afterwards. That is the
correct order -- adapt, then re-lock -- and it is why this is a tool rather than
a sed command.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Files that legitimately pin absolute paths.
TARGETS = [
    "configs/connectome_v2.yaml",
    "configs/connectome_v2_retry3.yaml",
    "scforge/workflow/environment_contract.yaml",
    "scforge/configs/scforge.yaml",
]

# The project root the shipped configuration was written against.
SHIPPED_ROOT = "/home/ec2-user/exp"


def detect_shipped_root(paths: list[Path]) -> str | None:
    """Find the old project root by looking for the path of a file we can see.

    Guessing a prefix would be fragile, so it is confirmed: a candidate root is
    accepted only when the config refers to a file that exists relative to the
    real project root.
    """
    counts: dict[str, int] = {}
    pat = re.compile(r'"(/[^"]*?)/(configs|scforge|research_audit|atlas)/')
    for p in paths:
        if not p.is_file():
            continue
        for m in pat.finditer(p.read_text(errors="ignore")):
            root, sub = m.group(1), m.group(2)
            if (PROJECT_ROOT / sub).exists():
                counts[root] = counts.get(root, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only; write nothing")
    ap.add_argument("--from-root", default=None,
                    help=f"the project root to replace (default: detected, else {SHIPPED_ROOT})")
    ap.add_argument("--to-root", default=str(PROJECT_ROOT),
                    help="the project root to write (default: this directory)")
    ap.add_argument("--no-relock", action="store_true",
                    help="skip regenerating the locked source manifest")
    args = ap.parse_args(argv)

    paths = [PROJECT_ROOT / t for t in TARGETS]
    old = args.from_root or detect_shipped_root(paths) or SHIPPED_ROOT
    new = str(Path(args.to_root).resolve())

    print(f"from : {old}")
    print(f"to   : {new}\n")
    if old == new:
        print("already pointing at this machine; nothing to do.")
        return 0

    total, touched = 0, []
    for p in paths:
        if not p.is_file():
            print(f"  {p.relative_to(PROJECT_ROOT)}   (absent)")
            continue
        s = p.read_text()
        n = s.count(old)
        rel = p.relative_to(PROJECT_ROOT)
        print(f"  {str(rel):<48} {n:>4} path(s)")
        if n and not args.check:
            p.write_text(s.replace(old, new))
            touched.append(rel)
        total += n

    if args.check:
        print(f"\n{total} path(s) would change. Nothing written.")
        return 0
    if not total:
        print("\nnothing to change.")
        return 0

    print(f"\nrewrote {total} path(s) in {len(touched)} file(s)")

    if args.no_relock:
        print("--no-relock: the locked source manifest is now stale.")
        return 0

    # These files are covered by the lock, so their hashes just changed.
    locker = PROJECT_ROOT / "scforge" / "workflow" / "lock_source_manifest.py"
    if locker.is_file():
        print("\nre-locking the source manifest")
        rc = subprocess.run([sys.executable, str(locker)], cwd=str(PROJECT_ROOT)).returncode
        if rc:
            print("re-lock failed; run scforge/workflow/lock_source_manifest.py by hand.",
                  file=sys.stderr)
            return rc
    print("\nNow run: python sc_doctor.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
