#!/usr/bin/env python3
"""Audit AD production tractogram headers against endpoint-space failure classes.

This script is diagnostic-only. It reads existing production ``tracks_final_3000k.tck``
headers, joins them to the endpoint-space action report, and writes a CSV that
shows whether endpoint-collapse cases share a tractogram-generation signature.
It does not read streamline payloads and does not write production outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
REPORT_DIR = ROOT / "reports" / "aal3_track_header_audit"
ENDPOINT_ACTION_CSV = (
    ROOT
    / "reports"
    / "aal3_endpoint_space_contract_ad_production_5k"
    / "aal3_ad_endpoint_space_action_latest.csv"
)
DERIV_TRACKS = Path("/data/derivatives/tracks")
LOCAL_DERIV_TRACKS = ROOT / "data" / "derivatives" / "tracks"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    tmp.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def fnum(value: Any, default: float | None = None) -> float | None:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def parse_tck_header_multimap(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    header: dict[str, str] = {}
    multi: dict[str, list[str]] = defaultdict(list)
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii", errors="replace").strip()
        if first != "mrtrix tracks":
            raise ValueError(f"{path} is not an MRtrix .tck file")
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"{path} header ended before END")
            line = raw.decode("ascii", errors="replace").strip()
            if line == "END":
                break
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            header[key] = value
            multi[key].append(value)
    return header, multi


def extract_arg_values(command: str, flag: str) -> list[str]:
    tokens = command.split()
    values: list[str] = []
    for idx, token in enumerate(tokens[:-1]):
        if token == flag:
            values.append(tokens[idx + 1])
    return values


def first_tckgen_source(command: str) -> str:
    tokens = command.split()
    if not tokens:
        return ""
    try:
        idx = next(i for i, token in enumerate(tokens) if token.endswith("tckgen") or token == "tckgen")
    except StopIteration:
        idx = 0
    positional: list[str] = []
    skip_next = False
    flags_with_values = {
        "-act",
        "-seed_gmwmi",
        "-seed_dynamic",
        "-select",
        "-seeds",
        "-cutoff",
        "-maxlength",
        "-minlength",
        "-nthreads",
        "-algorithm",
        "-include",
        "-exclude",
        "-mask",
        "-stop",
        "-step",
        "-angle",
    }
    for token in tokens[idx + 1 :]:
        if skip_next:
            skip_next = False
            continue
        if token in flags_with_values:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        positional.append(token)
    return positional[0] if positional else ""


def audit_row(endpoint_row: dict[str, str]) -> dict[str, Any]:
    sid = endpoint_row.get("sid", "")
    raw_tracks = str(endpoint_row.get("tracks_path") or "").strip()
    candidates = []
    if raw_tracks:
        candidates.append(Path(raw_tracks))
    candidates.extend(
        [
            LOCAL_DERIV_TRACKS / sid / "tracks_final_3000k.tck",
            DERIV_TRACKS / sid / "tracks_final_3000k.tck",
        ]
    )
    tracks_path = next((path for path in candidates if path.exists() and path.is_file()), candidates[-1])
    out: dict[str, Any] = {
        "sid": sid,
        "recommended_action": endpoint_row.get("recommended_action", ""),
        "endpoint_failure_class": endpoint_row.get("endpoint_failure_class", ""),
        "best_route": endpoint_row.get("best_route", ""),
        "best_density": endpoint_row.get("best_density", ""),
        "best_zero_rows": endpoint_row.get("best_zero_rows", ""),
        "parc_label_count": endpoint_row.get("parc_label_count", ""),
        "unique_endpoint_labels": endpoint_row.get("unique_endpoint_labels", ""),
        "top3_endpoint_label_fraction": endpoint_row.get("top3_endpoint_label_fraction", ""),
        "endpoint_on_label_fraction": endpoint_row.get("endpoint_on_label_fraction", ""),
        "tracks_path": str(tracks_path),
        "tracks_exists": tracks_path.exists(),
        "header_status": "missing_tracks" if not tracks_path.exists() else "pending",
    }
    if not tracks_path.exists():
        return out
    try:
        header, multi = parse_tck_header_multimap(tracks_path)
    except Exception as exc:
        out.update({"header_status": "error", "header_error": f"{type(exc).__name__}: {exc}"})
        return out

    history = multi.get("command_history", [])
    history_text = "\n".join(history)
    tckgen_history = [line for line in history if re.search(r"(^|[/\s])tckgen(\s|$)", line)]
    last_tckgen = tckgen_history[-1] if tckgen_history else ""
    source = first_tckgen_source(last_tckgen) or header.get("source", "")
    chunk_manifest = tracks_path.parent / "_chunks" / "chunk_manifest.txt"
    manifest_text = chunk_manifest.read_text(encoding="ascii", errors="ignore") if chunk_manifest.exists() else ""
    manifest: dict[str, str] = {}
    for line in manifest_text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            manifest[key.strip()] = value.strip()
    select_values = []
    seeds_values = []
    cutoffs = []
    for cmd in tckgen_history:
        select_values.extend(extract_arg_values(cmd, "-select"))
        seeds_values.extend(extract_arg_values(cmd, "-seeds"))
        cutoffs.extend(extract_arg_values(cmd, "-cutoff"))

    out.update(
        {
            "header_status": "ok",
            "datatype": header.get("datatype", ""),
            "count": header.get("count", ""),
            "total_count": header.get("total_count", ""),
            "file_entry": header.get("file", ""),
            "source": header.get("source", ""),
            "source_basename": Path(header.get("source", "")).name if header.get("source") else "",
            "act": header.get("act", ""),
            "act_basename": Path(header.get("act", "")).name if header.get("act") else "",
            "prior_roi": header.get("prior_roi", ""),
            "seed_dynamic_header": header.get("seed_dynamic", ""),
            "max_num_tracks": header.get("max_num_tracks", ""),
            "max_num_seeds": header.get("max_num_seeds", ""),
            "command_history_count": len(history),
            "tckgen_command_count": len(tckgen_history),
            "command_history_has_chunks": int("/_chunks/" in history_text or "_chunks/chunk_" in history_text),
            "command_history_has_chunk_outputs": int(bool(re.search(r"/_chunks/chunk_[^/\s]+\.tck", history_text))),
            "command_history_has_wmfod_final": int("wmfod_final.mif" in history_text),
            "command_history_has_plain_wmfod": int(bool(re.search(r"(^|[/\s])wmfod\.mif(\s|$)", history_text))),
            "chunk_manifest_exists": int(chunk_manifest.exists()),
            "chunk_manifest_cutoff": manifest.get("cutoff", ""),
            "chunk_manifest_seed_mode": manifest.get("seed_mode", ""),
            "chunk_manifest_select_streamlines": manifest.get("select_streamlines", ""),
            "chunk_manifest_chunk_streamlines": manifest.get("chunk_streamlines", ""),
            "last_tckgen_source": source,
            "last_tckgen_source_basename": Path(source).name if source else "",
            "tckgen_select_values": ";".join(select_values),
            "tckgen_seed_values": ";".join(seeds_values),
            "tckgen_cutoffs": ";".join(cutoffs),
            "has_seed_dynamic": int("-seed_dynamic" in history_text or bool(header.get("seed_dynamic"))),
            "has_seed_gmwmi": int("-seed_gmwmi" in history_text),
            "has_crop_at_gmwmi": int("-crop_at_gmwmi" in history_text or manifest.get("seed_mode") == "gmwmi"),
            "has_backtrack": int("-backtrack" in history_text or bool(header.get("backtrack"))),
            "has_act": int("-act" in history_text or bool(header.get("act"))),
            "last_tckgen_command": last_tckgen[:1000],
        }
    )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_action: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        action = str(row.get("recommended_action") or "")
        signature = (
            f"chunks={row.get('command_history_has_chunks', '')};"
            f"src={row.get('last_tckgen_source_basename', '')};"
            f"maxtracks={row.get('max_num_tracks', '')};"
            f"seed_dynamic={row.get('has_seed_dynamic', '')};"
            f"seed_gmwmi={row.get('has_seed_gmwmi', '')};"
            f"crop={row.get('has_crop_at_gmwmi', '')}"
        )
        by_action[action][signature] += 1
    collapse_rows = [r for r in rows if r.get("recommended_action") == "upstream_track_fod_act_repair"]
    plausible_rows = [r for r in rows if r.get("recommended_action") == "continue_parcellation_assignment_routes"]
    return {
        "generated_utc": utc(),
        "rows": len(rows),
        "actions": dict(Counter(str(r.get("recommended_action") or "") for r in rows)),
        "signature_by_action": {
            action: dict(counter.most_common(10)) for action, counter in sorted(by_action.items())
        },
        "collapse_chunked_n": sum(int(r.get("command_history_has_chunks") or 0) for r in collapse_rows),
        "collapse_n": len(collapse_rows),
        "plausible_chunked_n": sum(int(r.get("command_history_has_chunks") or 0) for r in plausible_rows),
        "plausible_n": len(plausible_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-action-csv", type=Path, default=ENDPOINT_ACTION_CSV)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    endpoint_rows = read_csv(args.endpoint_action_csv)
    rows = [audit_row(row) for row in endpoint_rows]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    now = stamp()
    out_csv = args.out_dir / f"aal3_ad_track_header_audit_{now}.csv"
    out_json = args.out_dir / f"aal3_ad_track_header_audit_{now}.json"
    write_csv(out_csv, rows)
    payload = summarize(rows)
    payload["csv"] = str(out_csv)
    write_json(out_json, payload)
    write_csv(args.out_dir / "aal3_ad_track_header_audit_latest.csv", rows)
    write_json(args.out_dir / "aal3_ad_track_header_audit_latest.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
