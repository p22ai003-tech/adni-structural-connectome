from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class StageResult:
    subject: str
    stage: str
    status: str
    command: tuple[str, ...]
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    started_utc: str = ""
    finished_utc: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["command"] = " ".join(self.command)
        return data


def ensure_dirs(paths: Iterable[str | Path], *, execute: bool = False) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        directory = Path(path)
        if execute:
            directory.mkdir(parents=True, exist_ok=True)
        records.append({"path": str(directory), "exists": directory.exists(), "execute": bool(execute)})
    return records


def run_command(
    command: list[str] | tuple[str, ...],
    *,
    subject: str,
    stage: str,
    execute: bool = False,
    cwd: str | Path | None = None,
    timeout: int | None = None,
) -> StageResult:
    command = tuple(str(part) for part in command)
    started = datetime.now(timezone.utc).isoformat()
    if not execute:
        finished = datetime.now(timezone.utc).isoformat()
        return StageResult(
            subject=subject,
            stage=stage,
            status="DRY_RUN",
            command=command,
            started_utc=started,
            finished_utc=finished,
        )
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    finished = datetime.now(timezone.utc).isoformat()
    return StageResult(
        subject=subject,
        stage=stage,
        status="PASS" if proc.returncode == 0 else "FAIL",
        command=command,
        returncode=int(proc.returncode),
        stdout=proc.stdout,
        stderr=proc.stderr,
        started_utc=started,
        finished_utc=finished,
    )
