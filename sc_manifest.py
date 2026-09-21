"""Validate an acquisition manifest against the project's input contract.

The manifest is the pipeline's only input contract: one row per DWI/T1 pair,
carrying source paths, checksums, acquisition parameters and the analysis role.
Discovery produces it; every later stage consumes it. Nothing downstream should
glob the raw tree.

The schema is not defined here. It already exists and is authoritative:

    scforge/workflow/schemas/acquisition_manifest_v2.schema.json   (47 fields)

This module makes that schema usable: it validates a CSV against it, and it
checks the two acquisition fields that are known to block the v2 route, because
a manifest can satisfy the schema and still be unprocessable.

Usage
-----
    python sc_manifest.py validate path/to/manifest.csv
    python sc_manifest.py fields            # print the contract
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import sc_config
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sc_config

SCHEMA_REL = Path("scforge/workflow/schemas/acquisition_manifest_v2.schema.json")

# Fields that must be non-empty for the v2 route to run at all. Every one of the
# 530 rows in the shipped manifest is blank for these two, which is why all 530
# units fail at the eddy stage: dwifslpreproc needs the phase-encoding direction
# and the total readout time, and the rule hard-fails rather than guessing.
EDDY_REQUIRED = ("phase_encoding_direction", "total_readout_time")


def schema_path() -> Path:
    return sc_config.paths().project_root / SCHEMA_REL


def load_schema() -> dict:
    p = schema_path()
    if not p.is_file():
        raise SystemExit(f"schema not found: {p}")
    return json.loads(p.read_text())


def _required(schema: dict) -> list[str]:
    if "required" in schema:
        return list(schema["required"])
    return list((schema.get("items") or {}).get("required") or [])


def validate(csv_path: Path) -> int:
    schema = load_schema()
    required = _required(schema)
    if not csv_path.is_file():
        print(f"manifest not found: {csv_path}", file=sys.stderr)
        return 2

    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("manifest is empty", file=sys.stderr)
        return 2

    present = set(rows[0].keys())
    missing_cols = [c for c in required if c not in present]
    extra_cols = sorted(present - set(required))

    print(f"manifest : {csv_path}")
    print(f"rows     : {len(rows)}")
    print(f"columns  : {len(present)} present, {len(required)} required by the contract")

    problems = 0
    if missing_cols:
        problems += 1
        print(f"\nMISSING REQUIRED COLUMNS ({len(missing_cols)}):")
        for c in missing_cols:
            print(f"  - {c}")
    if extra_cols:
        print(f"\nextra columns not in the contract ({len(extra_cols)}): {', '.join(extra_cols[:8])}"
              + (" ..." if len(extra_cols) > 8 else ""))

    # Per-row emptiness, counted only for fields the schema says cannot be
    # null. A DICOM series legitimately leaves the NIfTI-bundle members empty,
    # and an unlabelled cohort leaves the diagnosis empty; reporting those as
    # problems tells a user their correct manifest is broken.
    props = schema.get("properties") or (schema.get("items") or {}).get("properties") or {}

    def _nullable(name: str) -> bool:
        """True when the schema permits null, including via anyOf."""
        spec = props.get(name) or {}
        declared = spec.get("type")
        if isinstance(declared, list) and "null" in declared:
            return True
        for branch in spec.get("anyOf", []) or spec.get("oneOf", []):
            if isinstance(branch, dict) and branch.get("type") == "null":
                return True
        if declared is None and "enum" in spec:
            return None in spec["enum"]
        return False

    # Dates and free-text context are cohort metadata: the workflow never reads
    # them, so an empty one is worth noting, not failing.
    INFORMATIONAL = {"t1_study_date", "dti_study_date", "phase", "site", "protocol",
                     "scanner_model", "field_strength_t", "manufacturer"}
    blank_counts: dict[str, int] = {}
    nullable_blanks: dict[str, int] = {}
    for row in rows:
        for c in required:
            if c in row and (row[c] is None or str(row[c]).strip() == ""):
                if _nullable(c) or c in INFORMATIONAL:
                    nullable_blanks[c] = nullable_blanks.get(c, 0) + 1
                else:
                    blank_counts[c] = blank_counts.get(c, 0) + 1
    if nullable_blanks:
        print(f"\nempty but allowed to be ({len(nullable_blanks)} nullable field(s)): "
              + ", ".join(sorted(nullable_blanks)[:6])
              + (" ..." if len(nullable_blanks) > 6 else ""))
    if blank_counts:
        print(f"\nREQUIRED FIELDS BLANK IN SOME ROWS ({len(blank_counts)} fields):")
        for c, n in sorted(blank_counts.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {n:>5}/{len(rows)} rows blank   {c}")
        problems += 1

    # the specific blocker
    print("\nacquisition readiness (needed by dwifslpreproc):")
    blocked = 0
    for c in EDDY_REQUIRED:
        if c not in present:
            print(f"  {c}: COLUMN ABSENT")
            blocked = len(rows)
            continue
        n_blank = sum(1 for r in rows if str(r.get(c, "")).strip() == "")
        blocked = max(blocked, n_blank)
        state = "OK" if n_blank == 0 else f"{n_blank}/{len(rows)} blank"
        print(f"  {c}: {state}")
    if blocked:
        problems += 1
        print(f"\n  => {blocked} row(s) cannot reach tractography until these are supplied.")
        print("     They are recoverable from the DICOM headers of the source series")
        print("     at discovery time; the pipeline must not guess them.")

    print(f"\nverdict: {'FAIL' if problems else 'PASS'} ({problems} problem class(es))")
    return 1 if problems else 0


def fields() -> int:
    schema = load_schema()
    req = _required(schema)
    props = schema.get("properties") or (schema.get("items") or {}).get("properties") or {}
    print(f"{schema.get('title', 'acquisition manifest')}  —  {len(req)} required fields")
    print(f"source: {schema_path()}\n")
    for name in req:
        spec = props.get(name, {})
        typ = spec.get("type", "?")
        typ = "/".join(typ) if isinstance(typ, list) else typ
        desc = (spec.get("description") or "").strip().replace("\n", " ")
        print(f"  {name:<34} {typ:<16} {desc[:70]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="validate a manifest CSV against the contract")
    v.add_argument("csv", nargs="?", type=Path, default=None,
                   help="manifest CSV (default: $SC_MANIFEST)")
    sub.add_parser("fields", help="print the contract's required fields")
    args = ap.parse_args(argv)

    if args.cmd == "fields":
        return fields()
    target = args.csv or sc_config.paths().manifest
    return validate(Path(target))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
