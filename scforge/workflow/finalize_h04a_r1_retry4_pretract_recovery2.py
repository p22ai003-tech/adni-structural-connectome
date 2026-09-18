"""Versioned terminal publication adapter for retry4 pretract Recovery2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import finalize_h04a_r1_retry4_pretract_recovery1 as recovery1


LEDGER_NAME = "pre_tractography_canary_recovery2_ledger.jsonl"
MANIFEST_NAME = "pre_tractography_canary_recovery2_manifest.json"


class _Recovery2PublicationProxy:
    """Reuse the audited outcome logic while versioning immutable records."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def write_immutable_jsonl(self, path: Path, entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            entry["entry_type"] = "pre_tractography_canary_recovery2_outcome"
        self._delegate.write_immutable_jsonl(path, entries)

    def write_immutable_json(self, path: Path, payload: dict[str, Any]) -> None:
        if payload.get("record_type") == "pre_tractography_canary_recovery_manifest":
            payload["record_type"] = "pre_tractography_canary_recovery2_manifest"
            payload["mode"] = "pre-tractography-canary-recovery2"
        self._delegate.write_immutable_json(path, payload)


def finalize_recovery2(**kwargs: Any) -> dict[str, Any]:
    """Publish Recovery2 without changing or replacing either earlier release."""

    recovery1.LEDGER_NAME = LEDGER_NAME
    recovery1.MANIFEST_NAME = MANIFEST_NAME
    kwargs["base"] = _Recovery2PublicationProxy(kwargs["base"])
    return recovery1.finalize_recovery1(**kwargs)
