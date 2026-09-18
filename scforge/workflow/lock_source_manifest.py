#!/usr/bin/env python
"""Write the locked source manifest for the v2 workflow.

The manifest pins a SHA-256 for every source file the workflow contract covers,
so that a rule or schema cannot change without the contract check noticing.

It is regenerated rather than hand-edited, and from the same set the validator
checks -- ``validate_environment_contract.required_source_set()`` -- because the
two drifting apart is exactly what went wrong before: seven rule files were
added to ``rules/``, which the validator globs, and never added to the manifest.
The contract then failed although nothing had actually changed.

    python scforge/workflow/lock_source_manifest.py --check   # verify, exit 1 on drift
    python scforge/workflow/lock_source_manifest.py           # rewrite the lock

Use --check in CI. Rewrite only when the change to the covered sources is
intended: regenerating on autopilot defeats the point of a lock file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from validate_environment_contract import required_source_set  # noqa: E402

MANIFEST = HERE / "workflow_source_manifest.tsv"
# The lock has two levels: this manifest pins every source file, and
# environment_contract.yaml pins the manifest itself. Updating one without the
# other just moves the drift, so both are written here.
ENV_CONTRACT = HERE / "environment_contract.yaml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build() -> list[tuple[str, str]]:
    rows = []
    for p in sorted(required_source_set(), key=rel):
        if not p.is_file():
            raise SystemExit(f"required source file is missing: {p}")
        rows.append((rel(p), sha256(p)))
    return rows


def read_existing() -> dict[str, str]:
    if not MANIFEST.is_file():
        return {}
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        return {r["path"]: r["sha256"].strip() for r in csv.DictReader(fh, delimiter="\t")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report differences and exit non-zero; do not write")
    args = ap.parse_args(argv)

    rows = build()
    current = read_existing()
    new = {p: h for p, h in rows}

    added = sorted(set(new) - set(current))
    removed = sorted(set(current) - set(new))
    changed = sorted(p for p in set(new) & set(current) if new[p] != current[p])

    print(f"manifest : {MANIFEST}")
    print(f"required : {len(rows)} files   locked: {len(current)}")
    for label, items in (("not locked", added), ("no longer required", removed),
                         ("CONTENT CHANGED", changed)):
        if items:
            print(f"\n{label} ({len(items)}):")
            for p in items:
                print(f"   {p}")

    # The second level of the lock is checked too: the manifest can be current
    # while environment_contract.yaml still pins an older copy of it.
    env_stale = env_contract_stale(len(rows))

    if not (added or removed or changed):
        if not env_stale:
            print("\nmanifest is current, and the contract pin matches.")
            return 0
        print("\nmanifest is current, but environment_contract.yaml pins an older copy")
        if args.check:
            return 1
        update_env_contract(len(rows))
        return 0
    if args.check:
        print("\nmanifest is out of date (run without --check to rewrite)")
        return 1

    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["path", "sha256"])
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {MANIFEST.name}")
    update_env_contract(len(rows))
    return 0


def env_contract_stale(n_rows: int) -> bool:
    """True when environment_contract.yaml does not pin the manifest as it is now."""
    if not ENV_CONTRACT.is_file() or not MANIFEST.is_file():
        return False
    digest = sha256(MANIFEST)
    inside = False
    for line in ENV_CONTRACT.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("source_manifest:"):
            inside = True
            continue
        if inside:
            st = line.strip()
            if st.startswith("sha256:") and digest not in st:
                return True
            if st.startswith("rows:") and st.split(":", 1)[1].strip() != str(n_rows):
                return True
            if st and not line.startswith((" " * 4, "\t")):
                break
    return False


def update_env_contract(n_rows: int) -> None:
    """Re-pin the manifest's own hash and row count in environment_contract.yaml.

    Edited as text rather than round-tripped through a YAML dump, because that
    file carries comments and an ordering that a dump would discard.
    """
    if not ENV_CONTRACT.is_file():
        print(f"  (no {ENV_CONTRACT.name}; nothing else to re-pin)")
        return
    digest = sha256(MANIFEST)
    text = ENV_CONTRACT.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    inside = False
    hits = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("source_manifest:"):
            inside = True
            continue
        if inside:
            stripped = line.strip()
            if stripped.startswith("sha256:"):
                indent = line[: len(line) - len(line.lstrip())]
                lines[i] = f'{indent}sha256: "{digest}"\n'
                hits += 1
            elif stripped.startswith("rows:"):
                indent = line[: len(line) - len(line.lstrip())]
                lines[i] = f"{indent}rows: {n_rows}\n"
                hits += 1
            elif stripped and not line.startswith((" " * 4, "\t")):
                break
    if hits:
        ENV_CONTRACT.write_text("".join(lines), encoding="utf-8")
        print(f"  re-pinned {ENV_CONTRACT.name}: sha256 {digest[:12]}..., rows {n_rows}")
    else:
        print(f"  WARNING: no source_manifest sha256/rows found in {ENV_CONTRACT.name}")


if __name__ == "__main__":
    raise SystemExit(main())
