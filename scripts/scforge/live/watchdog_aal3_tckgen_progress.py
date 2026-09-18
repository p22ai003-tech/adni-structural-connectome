#!/usr/bin/env python3
"""Watch AAL3 rescue tckgen jobs and stop runs that cannot finish in time.

This watchdog is intentionally narrow. It only inspects MRtrix ``tckgen``
children writing into the AAL3 Route 7+ scratch probe
folders. If a job's selected-streamline progress predicts it cannot finish
within the configured timeout, the watchdog sends SIGTERM. The parent route
runner then records the subject as failed and moves on to the next queued
subject.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import time
from pathlib import Path


ROUTE_FRAGMENTS = (
    "/aal3_route7_selective_rescue_probe/",
    "/aal3_route11b_act_sift2_track_coverage_rerun_probe/",
    "/aal3_route11c_zero_row_coverage_rescue_probe/",
    "/aal3_route12a_tckgen_fallback_rescue_probe/",
    "/aal3_route12b_zero_row_targeted_rerun_probe/",
    "/aal3_route14_step7_rebuilt_track_coverage_probe/",
    "/aal3_route14b_assignment_retry_probe/",
    "/aal3_route15_",
    "/aal3_route16_",
    "/aal3_route17_",
    "/aal3_route18_",
    "/aal3_route19_",
)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def elapsed_sec(pid: int) -> int:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace").split()
        start_ticks = int(stat[21])
        hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        uptime = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
        return max(0, int(uptime - (start_ticks / hz)))
    except Exception:
        return 0


def arg_after(cmd: list[str], name: str, default: str = "") -> str:
    try:
        idx = cmd.index(name)
    except ValueError:
        return default
    if idx + 1 >= len(cmd):
        return default
    return cmd[idx + 1]


def parse_selected_from_log(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    for line in reversed(lines[-200:]):
        match = re.search(r"(\d+)\s+selected\b", line)
        if match:
            return int(match.group(1))
    return None


def infer_log_path(track_path: Path, requested: int, mode: str) -> Path:
    suffix = f"{requested // 1000}k"
    logs_dir = track_path.parent.parent / "logs"
    preferred = logs_dir / f"tckgen_{mode}_{suffix}.log"
    if preferred.exists():
        return preferred
    candidates = sorted(logs_dir.glob("tckgen_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else preferred


def iter_live_tckgen() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmd = read_cmdline(pid)
        if not cmd:
            continue
        exe = Path(cmd[0]).name
        if exe != "tckgen":
            continue
        if len(cmd) < 3:
            continue
        track_path = Path(cmd[2])
        track_text = str(track_path)
        if not any(fragment in track_text for fragment in ROUTE_FRAGMENTS):
            continue
        requested = int(float(arg_after(cmd, "-select", "0") or 0))
        mode = "gmwmi" if "-seed_gmwmi" in cmd else "dynamic" if "-seed_dynamic" in cmd else "unknown"
        log_path = infer_log_path(track_path, requested, mode)
        selected = parse_selected_from_log(log_path)
        out.append(
            {
                "pid": pid,
                "elapsed": elapsed_sec(pid),
                "requested": requested,
                "selected": selected,
                "mode": mode,
                "track_path": track_path,
                "log_path": log_path,
            }
        )
    return out


def should_stop(job: dict[str, object], args: argparse.Namespace) -> tuple[bool, str]:
    elapsed = int(job["elapsed"])
    requested = int(job["requested"])
    selected = job["selected"]
    if requested <= 0:
        return False, "no requested streamline count"
    if elapsed < args.min_elapsed_sec:
        return False, f"elapsed {elapsed}s < {args.min_elapsed_sec}s"
    if selected is None:
        if elapsed >= args.no_progress_sec:
            return True, f"no progress log after {elapsed}s"
        return False, "no progress log yet"
    selected = int(selected)
    if selected <= 0:
        if elapsed >= args.no_progress_sec:
            return True, f"0 selected after {elapsed}s"
        return False, "0 selected but still inside no-progress grace"
    fraction = selected / requested
    estimated_total = elapsed / fraction
    cutoff = args.timeout_sec * args.estimate_timeout_factor
    if estimated_total > cutoff:
        pct = 100.0 * fraction
        return True, f"{pct:.1f}% after {elapsed}s predicts {estimated_total:.0f}s > {cutoff:.0f}s"
    return False, f"estimated {estimated_total:.0f}s <= {cutoff:.0f}s"


def check_once(args: argparse.Namespace) -> int:
    stopped = 0
    jobs = iter_live_tckgen()
    for job in jobs:
        stop, reason = should_stop(job, args)
        prefix = (
            f"[{utc()}] pid={job['pid']} mode={job['mode']} elapsed={job['elapsed']}s "
            f"selected={job['selected']}/{job['requested']} track={job['track_path']}"
        )
        if stop:
            print(f"{prefix} STOP {reason}", flush=True)
            stopped += 1
            if args.execute:
                os.kill(int(job["pid"]), signal.SIGTERM)
        elif args.verbose:
            print(f"{prefix} keep {reason}", flush=True)
    if not jobs:
        print(f"[{utc()}] no live AAL3 route7/11/12 tckgen jobs", flush=True)
    elif stopped == 0 and not args.verbose:
        print(f"[{utc()}] checked {len(jobs)} tckgen jobs; no stops", flush=True)
    return stopped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--min-elapsed-sec", type=int, default=600)
    parser.add_argument("--no-progress-sec", type=int, default=1200)
    parser.add_argument("--estimate-timeout-factor", type=float, default=1.10)
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    while True:
        check_once(args)
        if not args.loop:
            return 0
        time.sleep(max(30, args.interval_sec))


if __name__ == "__main__":
    raise SystemExit(main())
