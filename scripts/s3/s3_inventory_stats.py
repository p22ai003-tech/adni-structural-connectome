#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


SCAN_KEY_RE = re.compile(r"^(\d{3}_S_\d{4}_I\d+)")


def summarize_bundle_counts(bundle_counts: Counter[str]) -> tuple[str, str]:
    if not bundle_counts:
        return "--", "--"

    histogram = Counter(bundle_counts.values())
    mode_count = max(histogram.items(), key=lambda item: (item[1], item[0]))[0]
    summary = " + ".join(
        f"{files_per_subject}x{subject_count}"
        for files_per_subject, subject_count in sorted(histogram.items(), reverse=True)
    )
    return str(mode_count), summary


def bundle_key_from_name(name: str) -> str | None:
    match = SCAN_KEY_RE.match(name)
    return match.group(1) if match else None


def collect_local(path_str: str) -> tuple[int, int, int, str, str]:
    path = Path(path_str)
    if not path.exists():
        return 0, 0, 0, "--", "--"

    file_count = 0
    total_bytes = 0
    bundle_counts: Counter[str] = Counter()

    for root, _, files in os.walk(path):
        for filename in files:
            full_path = Path(root) / filename
            if full_path.is_symlink():
                continue
            try:
                stat = full_path.stat()
            except FileNotFoundError:
                continue
            file_count += 1
            total_bytes += stat.st_size
            bundle_key = bundle_key_from_name(filename)
            if bundle_key:
                bundle_counts[bundle_key] += 1

    files_per_subject, distribution = summarize_bundle_counts(bundle_counts)
    return file_count, total_bytes, len(bundle_counts), files_per_subject, distribution


def collect_s3(prefix: str) -> tuple[int, int, int, str, str]:
    result = subprocess.run(
        ["aws", "s3", "ls", prefix, "--recursive"],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return 0, 0, 0, "--", "--"

    file_count = 0
    total_bytes = 0
    bundle_counts: Counter[str] = Counter()

    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) != 4:
            continue
        _, _, size_str, key = parts
        try:
            size = int(size_str)
        except ValueError:
            continue
        file_count += 1
        total_bytes += size
        bundle_key = bundle_key_from_name(Path(key).name)
        if bundle_key:
            bundle_counts[bundle_key] += 1

    files_per_subject, distribution = summarize_bundle_counts(bundle_counts)
    return file_count, total_bytes, len(bundle_counts), files_per_subject, distribution


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in {"local", "s3"}:
        print(
            "Usage: s3_inventory_stats.py [local|s3] <path-or-prefix>",
            file=sys.stderr,
        )
        return 1

    mode = sys.argv[1]
    target = sys.argv[2]

    if mode == "local":
        file_count, total_bytes, subjects, files_per_subject, distribution = collect_local(target)
    else:
        file_count, total_bytes, subjects, files_per_subject, distribution = collect_s3(target)

    print(
        f"{file_count}|{total_bytes}|{subjects}|{files_per_subject}|{distribution}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
