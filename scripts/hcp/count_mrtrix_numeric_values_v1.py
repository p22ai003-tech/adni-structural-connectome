#!/usr/bin/env python3
"""Count numeric values in an MRtrix text output without assuming line layout."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def count_values(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for token in stripped.replace(",", " ").split():
                try:
                    value = float(token)
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: non-numeric token {token!r}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"{path}:{line_number}: non-finite value {token!r}"
                    )
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--expect", type=int)
    args = parser.parse_args()
    if not args.path.is_file() or args.path.is_symlink():
        parser.error(f"not a regular file: {args.path}")
    if args.expect is not None and args.expect < 0:
        parser.error("--expect must be non-negative")
    observed = count_values(args.path)
    if args.expect is not None and observed != args.expect:
        parser.error(
            f"numeric value count {observed} differs from expected {args.expect}"
        )
    print(observed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
