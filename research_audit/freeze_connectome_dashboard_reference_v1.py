#!/home/ec2-user/exp/.venv_connectome_app/bin/python
"""Freeze deterministic structural snapshots of the Streamlit reference app.

This does not copy raw scientific tables.  It records page structure, controls,
table schemas/content hashes and normalized chart-specification hashes for each
analysis section.  The resulting manifest is the migration oracle for the
FastAPI/React shadow application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from streamlit.testing.v1 import AppTest


EXP = Path("/home/ec2-user/exp")
APP = EXP / "apps/connectome_dashboard/connectome_app.py"
ANALYSIS_ROOT = EXP / "data/derivatives/qc/analysis_cohort"
OUTPUT_ROOT = (
    EXP
    / "research_audit/outputs/connectome_dashboard_reference_v1"
)
SECTIONS = (
    "Demographics",
    "Novel Findings",
    "Global DTI",
    "Local / ROI DTI",
    "Global Graph",
    "SC Matrix Viewer",
    "Node Metrics",
    "Coupling",
    "Coupling-AAL",
    "Brain Age",
    "LR / SR",
    "LR-SR Analysis",
    "EDR Exceptions",
    "ML Diagnostics",
    "Delay",
    "Advanced",
    "Network Analysis",
    "Functional Pending",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def normalized_spec_hash(raw: str) -> tuple[str, int]:
    parsed = json.loads(raw)

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): normalize(item)
                for key, item in sorted(value.items())
                if key not in {"uid"}
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value

    encoded = json.dumps(
        normalize(parsed),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded), len(encoded)


def dataframe_record(frame: pd.DataFrame, index: int) -> dict[str, Any]:
    normalized = frame.copy()
    normalized.columns = [str(column) for column in normalized.columns]
    payload = normalized.to_json(
        orient="split",
        date_format="iso",
        date_unit="ns",
        double_precision=15,
        default_handler=str,
    ).encode("utf-8")
    return {
        "index": index,
        "rows": int(len(normalized)),
        "columns": list(normalized.columns),
        "dtypes": {
            str(column): str(dtype)
            for column, dtype in normalized.dtypes.items()
        },
        "content_sha256": sha256_bytes(payload),
        "canonical_json_bytes": len(payload),
        "head": json_safe(
            normalized.head(2).to_dict(orient="records")
        ),
        "tail": json_safe(
            normalized.tail(2).to_dict(orient="records")
        ),
    }


def text_values(elements: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for element in elements:
        value = getattr(element, "value", None)
        if value is not None:
            result.append(str(value))
    return result


def control_records(at: AppTest) -> dict[str, list[dict[str, Any]]]:
    controls: dict[str, list[dict[str, Any]]] = {}
    for kind in (
        "radio",
        "selectbox",
        "multiselect",
        "checkbox",
        "toggle",
        "slider",
        "text_input",
        "button",
    ):
        records: list[dict[str, Any]] = []
        for element in at.get(kind):
            record: dict[str, Any] = {
                "label": str(getattr(element, "label", "")),
            }
            for attribute in (
                "value",
                "options",
                "min",
                "max",
                "step",
                "disabled",
            ):
                try:
                    value = getattr(element, attribute)
                except Exception:
                    continue
                record[attribute] = json_safe(value)
            records.append(record)
        if records:
            controls[kind] = records
    return controls


def chart_records(at: AppTest, kind: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, element in enumerate(at.get(kind)):
        raw = str(element.proto.spec)
        digest, byte_n = normalized_spec_hash(raw)
        records.append(
            {
                "index": index,
                "normalized_spec_sha256": digest,
                "normalized_spec_bytes": byte_n,
            }
        )
    return records


def page_record(at: AppTest, label: str, index: int) -> dict[str, Any]:
    type_counts: dict[str, int] = {}
    for element in at:
        type_counts[element.type] = type_counts.get(element.type, 0) + 1
    dataframes = [
        dataframe_record(element.value, frame_index)
        for frame_index, element in enumerate(at.dataframe)
    ]
    exceptions = [
        str(getattr(element, "value", ""))
        for element in at.exception
    ]
    warnings = text_values(at.warning)
    return {
        "label": label,
        "section_index": index,
        "element_type_counts": dict(sorted(type_counts.items())),
        "headers": text_values(at.header),
        "subheaders": text_values(at.subheader),
        "captions": text_values(at.caption),
        "markdown_sha256": sha256_bytes(
            json.dumps(
                text_values(at.markdown),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ),
        "controls": control_records(at),
        "dataframes": dataframes,
        "plotly_charts": chart_records(at, "plotly_chart"),
        "vega_lite_charts": chart_records(at, "vega_lite_chart"),
        "exceptions": exceptions,
        "warnings": warnings,
        "status": "PASS" if not exceptions else "FAIL_REFERENCE_PAGE",
    }


def analysis_artifact_manifest() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(
        item
        for item in ANALYSIS_ROOT.rglob("*")
        if item.is_file()
        and item.suffix.lower()
        in {".csv", ".json", ".md", ".png", ".svg", ".pdf"}
    ):
        stat = path.stat()
        records.append(
            {
                "relative_path": str(path.relative_to(ANALYSIS_ROOT)),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    encoded = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "root": str(ANALYSIS_ROOT),
        "file_n": len(records),
        "inventory_sha256": sha256_bytes(encoded),
        "files": records,
    }


def run_page(index: int, timeout: int) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.session_state["selected_section_idx"] = index
    at.run(timeout=timeout)
    return at


def run_monitor(timeout: int) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run(timeout=timeout)
    view = next(
        radio for radio in at.radio if radio.label == "View"
    )
    view.set_value("🔴 Live Pipeline Monitor")
    at.run(timeout=timeout)
    return at


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--stop-index",
        type=int,
        default=len(SECTIONS),
    )
    parser.add_argument(
        "--skip-monitor",
        action="store_true",
    )
    args = parser.parse_args()
    if not 0 <= args.start_index <= args.stop_index <= len(SECTIONS):
        raise ValueError("invalid section range")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []
    for index in range(args.start_index, args.stop_index):
        label = SECTIONS[index]
        print(f"[reference] {index:02d} {label}", flush=True)
        record = page_record(
            run_page(index, args.timeout),
            label,
            index,
        )
        pages.append(record)
        atomic_json(
            OUTPUT_ROOT / f"section_{index:02d}.json",
            record,
        )

    monitor: dict[str, Any] | None = None
    if not args.skip_monitor:
        print("[reference] live pipeline monitor", flush=True)
        monitor = page_record(
            run_monitor(args.timeout),
            "Live Pipeline Monitor",
            len(SECTIONS),
        )
        atomic_json(OUTPUT_ROOT / "pipeline_monitor.json", monitor)

    existing_pages: list[dict[str, Any]] = []
    for index in range(len(SECTIONS)):
        path = OUTPUT_ROOT / f"section_{index:02d}.json"
        if path.exists():
            existing_pages.append(
                json.loads(path.read_text(encoding="utf-8"))
            )
    monitor_path = OUTPUT_ROOT / "pipeline_monitor.json"
    if monitor is None and monitor_path.exists():
        monitor = json.loads(
            monitor_path.read_text(encoding="utf-8")
        )

    all_records = [
        *existing_pages,
        *([monitor] if monitor is not None else []),
    ]
    failures = [
        record["label"]
        for record in all_records
        if record["status"] != "PASS"
    ]
    summary = {
        "schema_version": "1.0.0",
        "record_type": "connectome_dashboard_streamlit_reference",
        "status": (
            "PASS"
            if len(existing_pages) == len(SECTIONS)
            and monitor is not None
            and not failures
            else "INCOMPLETE_OR_FAILED"
        ),
        "generated_utc": utc_now(),
        "implementation": {
            "path": str(APP),
            "size_bytes": APP.stat().st_size,
            "sha256": sha256_file(APP),
        },
        "analysis_artifacts": analysis_artifact_manifest(),
        "expected_section_n": len(SECTIONS),
        "captured_section_n": len(existing_pages),
        "monitor_captured": monitor is not None,
        "failed_pages": failures,
        "sections": existing_pages,
        "pipeline_monitor": monitor,
    }
    atomic_json(OUTPUT_ROOT / "reference.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "captured_section_n": len(existing_pages),
                "monitor_captured": monitor is not None,
                "failed_pages": failures,
            },
            indent=2,
        )
    )
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
