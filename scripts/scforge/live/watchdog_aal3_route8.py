#!/usr/bin/env python3
"""Watch the live AAL3 Route8 run for stale connectome workers.

The active Route8 process was originally started before the built-in
tck2connectome timeout was added. This watchdog is intentionally narrow: it
only sends SIGINT to Route8 tck2connectome children that have exceeded the
cutoff and have not produced an output matrix. It then refreshes scoring,
promotion, and the global route ledger.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path


EXP = Path("/home/ec2-user/exp")
PYTHON = EXP / ".venv_connectome_app" / "bin" / "python"
LIVE = EXP / "scripts" / "scforge" / "live"
SCORER = LIVE / "score_aal3_force_connectome_qc.py"
PROMOTER = LIVE / "promote_aal3_qc_passed.py"
LEDGER = LIVE / "summarize_aal3_route_ledger.py"
DEFAULT_RUN_ROOT = Path(
    "/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch/"
    "ad_existing_tracks_20260529T055752Z_route8_full_t1_fnirt"
)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def read_ppid(pid: int) -> int:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace").split()
        return int(fields[3])
    except Exception:
        return -1


def elapsed_sec(pid: int) -> int:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace").split()
        start_ticks = int(stat[21])
        hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        uptime = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
        return max(0, int(uptime - (start_ticks / hz)))
    except Exception:
        return 0


def find_route8_parent(run_root: Path) -> int | None:
    route1_root = Path(str(run_root).removesuffix("_route8_full_t1_fnirt"))
    route1_needle = str(route1_root)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmd = read_cmdline(pid)
        if not cmd:
            continue
        joined = " ".join(cmd)
        if "run_aal3_route8_full_t1_fnirt_batch.py" in joined and route1_needle in joined:
            return pid
    return None


def iter_route8_tck2connectome(parent_pid: int, run_root: Path):
    probe_root_name = f"{run_root.name}_"
    route8_output_fragment = "/aal3_route8_full_t1_fnirt_probe/"
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmd = read_cmdline(pid)
        if not cmd:
            continue
        if "tck2connectome" not in Path(cmd[0]).name:
            continue
        if len(cmd) < 4:
            continue
        output = Path(cmd[3])
        output_text = str(output)
        # Route 9 also consumes Route 8 parcellations as input. Restrict matching
        # to the Route 8 output matrix path so assignment-retry jobs are untouched.
        if route8_output_fragment not in output_text or probe_root_name not in output_text:
            continue
        yield {
            "pid": pid,
            "ppid": read_ppid(pid),
            "elapsed_sec": elapsed_sec(pid),
            "output": output,
            "cmd": cmd,
        }


def refresh_outputs(run_root: Path, execute_promotion: bool) -> None:
    subprocess.run([str(PYTHON), str(SCORER), "--run-root", str(run_root)], cwd=EXP, check=False)
    promote_cmd = [
        str(PYTHON),
        str(PROMOTER),
        "--run-root",
        str(run_root),
        "--allow-interim-without-visual-pass",
    ]
    if execute_promotion:
        promote_cmd.append("--execute")
    subprocess.run(promote_cmd, cwd=EXP, check=False)
    ledger_root = Path(str(run_root).removesuffix("_route8_full_t1_fnirt"))
    subprocess.run([str(PYTHON), str(LEDGER), "--run-root", str(ledger_root)], cwd=EXP, check=False)


def check_once(run_root: Path, cutoff_sec: int, execute_promotion: bool, refresh: bool) -> int:
    parent = find_route8_parent(run_root)
    if parent is None:
        print(f"[{utc()}] route8 parent not found; refreshing ledger only")
        refresh_outputs(run_root, execute_promotion=execute_promotion)
        return 1

    killed: list[int] = []
    for proc in iter_route8_tck2connectome(parent, run_root):
        output = proc["output"]
        has_output = output.exists() and output.stat().st_size > 0
        if proc["elapsed_sec"] >= cutoff_sec and not has_output:
            print(
                f"[{utc()}] stale tck2connectome pid={proc['pid']} "
                f"ppid={proc['ppid']} elapsed={proc['elapsed_sec']}s output_missing_or_empty={output}"
            )
            os.kill(proc["pid"], signal.SIGINT)
            killed.append(proc["pid"])

    if killed:
        time.sleep(10)
        refresh_outputs(run_root, execute_promotion=execute_promotion)
    elif refresh:
        print(f"[{utc()}] no stale tck2connectome workers; refreshing score/ledger")
        refresh_outputs(run_root, execute_promotion=execute_promotion)
    else:
        print(f"[{utc()}] no stale tck2connectome workers")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--cutoff-sec", type=int, default=6 * 60 * 60)
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--refresh-every-sec", type=int, default=900)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--execute-promotion", action="store_true")
    args = parser.parse_args()

    last_refresh = 0.0
    while True:
        now = time.time()
        should_refresh = args.refresh or (now - last_refresh >= args.refresh_every_sec)
        rc = check_once(args.run_root, args.cutoff_sec, args.execute_promotion, should_refresh)
        if should_refresh:
            last_refresh = now
        if not args.loop:
            return rc
        time.sleep(max(30, args.interval_sec))


if __name__ == "__main__":
    raise SystemExit(main())
