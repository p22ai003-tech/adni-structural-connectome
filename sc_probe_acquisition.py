#!/usr/bin/env python
"""Recover phase-encoding direction and total readout time from the source DICOMs.

Why this is needed
------------------
Every one of the 530 rows in the v2 acquisition manifest is blank for
``phase_encoding_direction`` and ``total_readout_time``. Those two fields are
what ``dwifslpreproc`` needs, and the v2 route hard-fails without them rather
than guessing -- correctly, because guessing is what the older pipeline did:
``scripts/preprocessing/ADNI_preproc_pipeline_AAL.sh`` passes a fixed
``-pe_dir j-`` to every subject in a cohort spanning several vendors, scanner
models and ADNI phases.

The values are recoverable. The raw DICOM tree is intact, and the readout maths
is vendor-specific, so this does not reimplement it: it runs dcm2niix in
sidecar-only mode (``-b o``, no image written, about half a second per series)
and reads the BIDS fields dcm2niix computes.

The manifest schema already carries ``phase_encoding_source`` and
``total_readout_time_source``. Those are filled in too, so a reader can always
tell a measured value from an assumed one. Nothing here ever writes an assumed
value: a series whose header does not carry the field is reported as missing.

    python sc_probe_acquisition.py --manifest <csv> --limit 24 --report
    python sc_probe_acquisition.py --manifest <csv> --out <csv>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sc_config  # noqa: E402

FIELDS = ("phase_encoding_direction", "phase_encoding_source",
          "total_readout_time", "total_readout_time_source")


def find_dcm2niix() -> str:
    """Prefer the version pinned in the repo over whatever is on PATH.

    The readout calculation is vendor-specific and has changed between
    dcm2niix releases, so which binary produced a value is part of its
    provenance, not an implementation detail.
    """
    for cand in sorted((PROJECT_ROOT / "tools" / "dcm2niix").glob("v*/dcm2niix"), reverse=True):
        if os.access(cand, os.X_OK):
            return str(cand)
    for cand in sorted((PROJECT_ROOT / ".envs").glob("dcm2niix-*/bin/dcm2niix"), reverse=True):
        if os.access(cand, os.X_OK):
            return str(cand)
    # `pip install dcm2niix` puts the binary beside the interpreter, which is
    # not on PATH unless the venv is activated.
    beside = Path(sys.executable).parent / "dcm2niix"
    if beside.exists() and os.access(beside, os.X_OK):
        return str(beside)
    found = shutil.which("dcm2niix")
    if found:
        return found
    raise SystemExit("dcm2niix not found: pip install -r requirements.txt, "
                     "or put the release binary under tools/dcm2niix/vX/")


def dcm2niix_version(binary: str) -> str:
    try:
        out = subprocess.run([binary, "-v"], capture_output=True, text=True, timeout=30)
        for line in (out.stdout + out.stderr).splitlines():
            if "version" in line.lower():
                return line.strip()
    except Exception:
        pass
    return "unknown"


def probe_series(binary: str, series_dir: Path, timeout: int = 180) -> dict:
    """Read one series' BIDS sidecar without writing an image."""
    if not series_dir.is_dir():
        return {"error": "series directory not found"}
    with tempfile.TemporaryDirectory(prefix="sc_probe_") as tmp:
        proc = subprocess.run(
            [binary, "-b", "o", "-f", "probe", "-o", tmp, str(series_dir)],
            capture_output=True, text=True, timeout=timeout,
        )
        sidecars = sorted(Path(tmp).glob("*.json"))
        if not sidecars:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-2:]
            return {"error": "no sidecar produced: " + " | ".join(tail)}
        # A series can split (multiple echoes / series numbers). Take the one
        # with the most volumes, which is the DWI series proper.
        best, best_n = None, -1
        for s in sidecars:
            try:
                d = json.loads(s.read_text())
            except json.JSONDecodeError:
                continue
            n = d.get("AcquisitionMatrixPE") or 0
            if n > best_n:
                best, best_n = d, n
        if best is None:
            return {"error": "sidecar unreadable"}
        return {"sidecar": best, "n_sidecars": len(sidecars)}


def row_values(sidecar: dict) -> dict:
    """Map the BIDS sidecar onto the manifest's four acquisition fields.

    dcm2niix distinguishes a value it read from one it estimated: when the
    header lacks what the exact calculation needs it emits Estimated* instead.
    That distinction is preserved rather than flattened, because an estimated
    readout time and a measured one are not the same evidence.
    """
    out = dict.fromkeys(FIELDS, "")
    ped = sidecar.get("PhaseEncodingDirection")
    if ped:
        out["phase_encoding_direction"] = ped
        out["phase_encoding_source"] = "dicom_header"
    elif sidecar.get("PhaseEncodingAxis"):
        # Axis without polarity: usable for the axis, but the sign is unknown
        # and must not be invented.
        out["phase_encoding_direction"] = sidecar["PhaseEncodingAxis"]
        out["phase_encoding_source"] = "dicom_header_axis_only_no_polarity"
    if sidecar.get("TotalReadoutTime") is not None:
        out["total_readout_time"] = sidecar["TotalReadoutTime"]
        out["total_readout_time_source"] = "dcm2niix_computed"
    elif sidecar.get("EstimatedTotalReadoutTime") is not None:
        out["total_readout_time"] = sidecar["EstimatedTotalReadoutTime"]
        out["total_readout_time_source"] = "dcm2niix_estimated"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="acquisition manifest CSV (default: $SC_MANIFEST)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write a manifest with the four fields filled in")
    ap.add_argument("--limit", type=int, default=None, help="probe only the first N rows")
    ap.add_argument("--stratify", action="store_true",
                    help="with --limit, spread the sample across manufacturer and phase")
    ap.add_argument("--report", action="store_true", help="print a distribution summary")
    ap.add_argument("--path-column", default="dti_source_path")
    args = ap.parse_args(argv)

    manifest = args.manifest or sc_config.paths().manifest
    if not Path(manifest).is_file():
        raise SystemExit(f"manifest not found: {manifest}")
    rows = list(csv.DictReader(Path(manifest).open(newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit("manifest is empty")

    binary = find_dcm2niix()
    print(f"manifest : {manifest}  ({len(rows)} rows)")
    print(f"dcm2niix : {binary}")
    print(f"           {dcm2niix_version(binary)}\n")

    todo = rows
    if args.limit:
        if args.stratify:
            buckets: dict[tuple, list] = {}
            for r in rows:
                buckets.setdefault((r.get("manufacturer", ""), r.get("phase", "")), []).append(r)
            todo, i = [], 0
            while len(todo) < args.limit and any(buckets.values()):
                for key in list(buckets):
                    if buckets[key] and len(todo) < args.limit:
                        todo.append(buckets[key].pop(0))
                i += 1
                if i > len(rows):
                    break
        else:
            todo = rows[: args.limit]

    results: dict[str, dict] = {}
    errors = 0
    for n, r in enumerate(todo, 1):
        sid = r.get("subject_id", "?")
        src = Path(str(r.get(args.path_column, "")).strip())
        res = probe_series(binary, src)
        if "error" in res:
            errors += 1
            results[id(r)] = {"error": res["error"]}
            print(f"  [{n}/{len(todo)}] {sid:<14} ERROR  {res['error'][:70]}")
            continue
        vals = row_values(res["sidecar"])
        results[id(r)] = vals
        print(f"  [{n}/{len(todo)}] {sid:<14} pe={vals['phase_encoding_direction'] or '-':<4} "
              f"trt={vals['total_readout_time'] or '-':<12} "
              f"{res['sidecar'].get('Manufacturer','?')}")

    if args.report:
        print("\n" + "=" * 62)
        pe = Counter(v.get("phase_encoding_direction", "ERROR") or "BLANK"
                     for v in results.values())
        src = Counter(v.get("phase_encoding_source", "ERROR") or "BLANK"
                      for v in results.values())
        print("phase-encoding direction:")
        for k, c in pe.most_common():
            print(f"   {k:<36} {c:>4}")
        print("source:")
        for k, c in src.most_common():
            print(f"   {k:<36} {c:>4}")
        trt = [float(v["total_readout_time"]) for v in results.values()
               if v.get("total_readout_time") not in (None, "")]
        if trt:
            print(f"total readout time: n={len(trt)} "
                  f"min={min(trt):.5f} median={sorted(trt)[len(trt)//2]:.5f} max={max(trt):.5f}")
        print(f"errors: {errors}")
        # the assumption the old pipeline made
        assumed = "j-"
        agree = sum(1 for v in results.values() if v.get("phase_encoding_direction") == assumed)
        print(f"\nrows whose header matches the pipeline's hard-coded -pe_dir {assumed}: "
              f"{agree}/{len(results)}")

    if args.out:
        for r in todo:
            for k, v in results.get(id(r), {}).items():
                if k in FIELDS:
                    r[k] = v
        with Path(args.out).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
