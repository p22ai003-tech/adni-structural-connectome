#!/usr/bin/env python3
"""
Guarded MRtrix denoise/Gibbs helpers for structural connectome Step 2/3.

The module keeps notebook and command-line behavior aligned:
- input:  derivatives/mif_dwi/<series>.mif
- output: derivatives/mif_denoised/<series>_den.mif
- output: derivatives/mif_unringed/<series>_den_unr.mif
- QC:     derivatives/qc/denoise_gibbs_summary.csv

It is intentionally conservative. Existing valid outputs are skipped unless
``force`` is set, and dry-run/status-only modes never overwrite files.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - notebook convenience fallback
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


DEFAULT_DERIV_ROOT = Path.home() / "exp" / "data" / "derivatives"
SUMMARY_HEADER = [
    "series",
    "input_mif",
    "denoised_mif",
    "unringed_mif",
    "denoise_status",
    "gibbs_status",
    "geometry_status",
    "gradient_status",
    "action",
    "duration_s",
    "notes",
]


def tool_path(cmd: str) -> str | None:
    found = shutil.which(cmd)
    if found:
        return found
    local = Path.home() / "mrtrix3" / "bin" / cmd
    if local.exists():
        return str(local)
    return None


def tool_cmd(cmd: str) -> str:
    return tool_path(cmd) or cmd


def _visible_glob(folder: Path, pattern: str) -> Iterable[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.glob(pattern)
        if path.exists() and not path.name.startswith(".") and not path.name.startswith("._")
    )


def _series_from_raw(path: Path) -> str:
    return path.name[:-4] if path.name.endswith(".mif") else path.stem


def denoised_path(deriv_root: Path, series: str) -> Path:
    return deriv_root / "mif_denoised" / f"{series}_den.mif"


def noise_path(deriv_root: Path, series: str) -> Path:
    return deriv_root / "qc" / "denoise_noise" / f"{series}_noise.mif"


def unringed_path(deriv_root: Path, series: str) -> Path:
    return deriv_root / "mif_unringed" / f"{series}_den_unr.mif"


def _run(cmd: Sequence[str], *, dry_run: bool) -> tuple[int, str, str]:
    if dry_run:
        return 0, "", ""
    proc = subprocess.run(list(cmd), text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def _mrinfo(path: Path, option: str) -> str:
    try:
        return subprocess.check_output(
            [tool_cmd("mrinfo"), str(path), option, "-quiet"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _normalize_transform(text: str) -> str:
    return " ".join(text.split())


def _normalize_dwgrad(text: str) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            rows.append(tuple(round(float(value), 6) for value in parts[:4]))  # type: ignore[arg-type]
        except ValueError:
            continue
    return rows


def _readable_mif(path: Path) -> bool:
    return path.exists() and bool(_mrinfo(path, "-size"))


def geometry_problem(reference: Path, candidate: Path) -> str:
    if not _readable_mif(reference):
        return f"reference not readable: {reference}"
    if not _readable_mif(candidate):
        return f"candidate not readable: {candidate}"
    for option, label in [("-size", "size"), ("-spacing", "spacing")]:
        ref = _mrinfo(reference, option)
        got = _mrinfo(candidate, option)
        if not ref or not got:
            return f"missing {label} header"
        if ref.split() != got.split():
            return f"{label} changed ({ref} -> {got})"
    ref_xfm = _normalize_transform(_mrinfo(reference, "-transform"))
    got_xfm = _normalize_transform(_mrinfo(candidate, "-transform"))
    if ref_xfm and got_xfm and ref_xfm != got_xfm:
        return "transform changed"
    return ""


def gradient_problem(reference: Path, candidate: Path) -> str:
    ref = _normalize_dwgrad(_mrinfo(reference, "-dwgrad"))
    got = _normalize_dwgrad(_mrinfo(candidate, "-dwgrad"))
    if not ref:
        return "reference gradients unavailable"
    if not got:
        return "candidate gradients unavailable"
    if len(ref) != len(got):
        return f"gradient row count changed ({len(ref)} -> {len(got)})"
    if ref != got:
        return "gradient table changed"
    return ""


def _validate_stage(input_mif: Path, output_mif: Path) -> tuple[str, str, str]:
    geom = geometry_problem(input_mif, output_mif)
    grad = gradient_problem(input_mif, output_mif)
    status = "ok" if not geom and not grad else "invalid"
    return status, geom, grad


def _row(
    *,
    series: str,
    input_mif: Path,
    denoised_mif: Path,
    unringed_mif: Path,
    denoise_status: str,
    gibbs_status: str,
    geometry_status: str,
    gradient_status: str,
    action: str,
    duration_s: float,
    notes: str,
) -> dict[str, object]:
    return {
        "series": series,
        "input_mif": str(input_mif),
        "denoised_mif": str(denoised_mif),
        "unringed_mif": str(unringed_mif),
        "denoise_status": denoise_status,
        "gibbs_status": gibbs_status,
        "geometry_status": geometry_status,
        "gradient_status": gradient_status,
        "action": action,
        "duration_s": f"{duration_s:.2f}",
        "notes": notes,
    }


def _process_one(
    raw_mif: Path,
    *,
    deriv_root: Path,
    dry_run: bool,
    force: bool,
    force_invalid_only: bool,
    run_denoise: bool,
    run_gibbs: bool,
) -> dict[str, object]:
    start = time.time()
    series = _series_from_raw(raw_mif)
    den = denoised_path(deriv_root, series)
    noise = noise_path(deriv_root, series)
    unr = unringed_path(deriv_root, series)
    den.parent.mkdir(parents=True, exist_ok=True)
    unr.parent.mkdir(parents=True, exist_ok=True)
    noise.parent.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []
    actions: list[str] = []

    den_status, den_geom, den_grad = _validate_stage(raw_mif, den) if den.exists() else ("missing", "missing", "missing")
    if run_denoise:
        should_run = force or den_status != "ok"
        if force_invalid_only:
            should_run = den_status != "ok"
        if should_run:
            actions.append("denoise_dry_run" if dry_run else "denoise")
            cmd = [
                tool_cmd("dwidenoise"),
                str(raw_mif),
                str(den),
                "-noise",
                str(noise),
            ]
            if force or den.exists() or not dry_run:
                cmd.append("-force")
            code, _, stderr = _run(cmd, dry_run=dry_run)
            if code != 0:
                notes.append(f"dwidenoise failed: {stderr.strip()[:500]}")
                den_status = "error"
            elif dry_run:
                den_status = "dry_run_pending"
            else:
                den_status, den_geom, den_grad = _validate_stage(raw_mif, den)
        else:
            actions.append("skip_denoise")
    else:
        actions.append("denoise_disabled")

    if den.exists() and not dry_run:
        den_status, den_geom, den_grad = _validate_stage(raw_mif, den)

    gibbs_status = "missing"
    gibbs_geom = "missing"
    gibbs_grad = "missing"
    if unr.exists():
        gibbs_status, gibbs_geom, gibbs_grad = _validate_stage(den if den.exists() else raw_mif, unr)

    if run_gibbs:
        den_available = den.exists() and den_status == "ok"
        if dry_run and den_status in {"ok", "dry_run_pending"}:
            den_available = den.exists()
        should_run = force or gibbs_status != "ok"
        if force_invalid_only:
            should_run = gibbs_status != "ok"
        if not den_available and not (dry_run and den_status == "dry_run_pending"):
            actions.append("wait_denoise")
            gibbs_status = "blocked"
            notes.append("Gibbs blocked until denoised input is valid")
        elif should_run:
            actions.append("gibbs_dry_run" if dry_run else "gibbs")
            cmd = [tool_cmd("mrdegibbs"), str(den), str(unr)]
            if force or unr.exists() or not dry_run:
                cmd.append("-force")
            code, _, stderr = _run(cmd, dry_run=dry_run)
            if code != 0:
                notes.append(f"mrdegibbs failed: {stderr.strip()[:500]}")
                gibbs_status = "error"
            elif dry_run:
                gibbs_status = "dry_run_pending"
            else:
                gibbs_status, gibbs_geom, gibbs_grad = _validate_stage(den, unr)
        else:
            actions.append("skip_gibbs")
    else:
        actions.append("gibbs_disabled")

    if unr.exists() and not dry_run and den.exists():
        gibbs_status, gibbs_geom, gibbs_grad = _validate_stage(den, unr)

    geometry_status = "; ".join(
        part
        for part in [
            f"denoise:{den_geom}" if den_geom else "denoise:ok",
            f"gibbs:{gibbs_geom}" if gibbs_geom else "gibbs:ok",
        ]
    )
    gradient_status = "; ".join(
        part
        for part in [
            f"denoise:{den_grad}" if den_grad else "denoise:ok",
            f"gibbs:{gibbs_grad}" if gibbs_grad else "gibbs:ok",
        ]
    )

    return _row(
        series=series,
        input_mif=raw_mif,
        denoised_mif=den,
        unringed_mif=unr,
        denoise_status=den_status,
        gibbs_status=gibbs_status,
        geometry_status=geometry_status,
        gradient_status=gradient_status,
        action=",".join(actions),
        duration_s=time.time() - start,
        notes=" | ".join(notes),
    )


def _write_summary(rows: Sequence[dict[str, object]], summary_csv: Path) -> Path:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_HEADER})
    return summary_csv


def run_denoise_gibbs_batch(
    *,
    deriv_root: Path = DEFAULT_DERIV_ROOT,
    jobs: int = 1,
    force: bool = False,
    dry_run: bool = True,
    force_invalid_only: bool = True,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    run_denoise: bool = True,
    run_gibbs: bool = True,
    summary_csv: Path | None = None,
) -> dict[str, object]:
    deriv_root = Path(deriv_root)
    summary_csv = summary_csv or deriv_root / "qc" / "denoise_gibbs_summary.csv"
    include_set = {str(item).strip() for item in include or [] if str(item).strip()}
    exclude_set = {str(item).strip() for item in exclude or [] if str(item).strip()}
    raw_inputs = list(_visible_glob(deriv_root / "mif_dwi", "*.mif"))
    if include_set:
        raw_inputs = [path for path in raw_inputs if _series_from_raw(path) in include_set]
    if exclude_set:
        raw_inputs = [path for path in raw_inputs if _series_from_raw(path) not in exclude_set]

    rows: list[dict[str, object]] = []
    jobs = max(1, int(jobs))
    kwargs = {
        "deriv_root": deriv_root,
        "dry_run": bool(dry_run),
        "force": bool(force),
        "force_invalid_only": bool(force_invalid_only),
        "run_denoise": bool(run_denoise),
        "run_gibbs": bool(run_gibbs),
    }

    if jobs == 1 or len(raw_inputs) <= 1:
        for raw_mif in tqdm(raw_inputs, desc="denoise/gibbs", unit="series"):
            rows.append(_process_one(raw_mif, **kwargs))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_process_one, raw_mif, **kwargs): raw_mif for raw_mif in raw_inputs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="denoise/gibbs", unit="series"):
                rows.append(future.result())
        rows.sort(key=lambda row: str(row.get("series", "")))

    _write_summary(rows, summary_csv)
    denoise_counts = Counter(str(row.get("denoise_status", "")) for row in rows)
    gibbs_counts = Counter(str(row.get("gibbs_status", "")) for row in rows)
    action_counts = Counter(
        action
        for row in rows
        for action in str(row.get("action", "")).split(",")
        if action
    )
    return {
        "summary_csv": str(summary_csv),
        "n_inputs": len(raw_inputs),
        "dry_run": bool(dry_run),
        "force": bool(force),
        "force_invalid_only": bool(force_invalid_only),
        "denoise_counts": dict(denoise_counts),
        "gibbs_counts": dict(gibbs_counts),
        "action_counts": dict(action_counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run guarded MRtrix dwidenoise/mrdegibbs stages.")
    parser.add_argument("--deriv-root", type=Path, default=DEFAULT_DERIV_ROOT)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default is dry-run/status scan.")
    parser.add_argument("--force", action="store_true", help="Force replacement of existing outputs.")
    parser.add_argument("--no-force-invalid-only", action="store_true", help="Do not restrict reruns to missing/invalid outputs.")
    parser.add_argument("--include", nargs="*", default=None)
    parser.add_argument("--exclude", nargs="*", default=None)
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--no-gibbs", action="store_true")
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_denoise_gibbs_batch(
        deriv_root=args.deriv_root,
        jobs=args.jobs,
        force=args.force,
        dry_run=not args.execute,
        force_invalid_only=not args.no_force_invalid_only,
        include=args.include,
        exclude=args.exclude,
        run_denoise=not args.no_denoise,
        run_gibbs=not args.no_gibbs,
        summary_csv=args.summary_csv,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
