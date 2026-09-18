#!/usr/bin/env python3
"""Release retry4 binding V7 with an explicit parent/worker seed policy."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
AUDIT = ROOT / "research_audit"
SOURCE = AUDIT / "outputs/h04a_r1_recovery_extension_binding_v6.json"
DESTINATION = AUDIT / "outputs/h04a_r1_recovery_extension_binding_v7.json"
POLICY = AUDIT / "decisions/sl_h04a_r1_retry4_worker_seed_validation_20260719.md"
RUN_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular evidence file: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    binding = json.loads(SOURCE.read_text(encoding="utf-8"))
    binding["source_files"] = {
        label: file_record(Path(str(record["path"])))
        for label, record in binding["source_files"].items()
    }
    binding["supersedes_binding"] = file_record(SOURCE)
    binding["package_builder"] = file_record(Path(__file__))
    binding["worker_seed_validation_policy"] = (
        "full_content_rehash_in_launcher_and_parent_dag;"
        "signed_manifest_path_and_size_validation_in_snakemake_workers"
    )
    binding["worker_seed_validation_decision"] = file_record(POLICY)
    write_immutable_json(DESTINATION, binding)

    sys.path.insert(0, str(ROOT / "scforge"))
    from scforge.h04a_r1_retry4 import validate_recovery_extension_binding

    validate_recovery_extension_binding(binding, expected_run_root=RUN_ROOT)
    print(json.dumps({"status": "PASS", "binding": file_record(DESTINATION)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
