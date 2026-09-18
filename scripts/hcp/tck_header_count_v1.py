#!/usr/bin/env python3
"""Read and validate the streamline count stored in an MRtrix TCK header."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


MAX_HEADER_BYTES = 1024 * 1024
COUNT_PATTERN = re.compile(r"^count:\s*([0-9]+)\s*$", re.MULTILINE)


def read_tck_header(path: Path) -> str:
    with path.open("rb") as handle:
        payload = handle.read(MAX_HEADER_BYTES)
    end = payload.find(b"END\n")
    if end < 0:
        raise ValueError(
            f"TCK header terminator not found in first {MAX_HEADER_BYTES} bytes"
        )
    return payload[: end + len(b"END\n")].decode("ascii")


def read_count(path: Path) -> int:
    header = read_tck_header(path)
    matches = COUNT_PATTERN.findall(header)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one TCK count field, found {len(matches)}")
    return int(matches[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read an MRtrix .tck count from its bounded ASCII header."
    )
    parser.add_argument("tck", type=Path)
    parser.add_argument("--expect", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        count = read_count(args.tck)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.expect is not None and count != args.expect:
        print(
            f"ERROR: header count {count} does not match expected {args.expect}",
            file=sys.stderr,
        )
        return 3
    print(count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
