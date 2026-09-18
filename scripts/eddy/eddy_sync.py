from pathlib import Path
import shutil
import subprocess

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


EDDY_SYNC_PATTERN = "*_preproc.mif"


def fmt_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB", "PB", "EB"]:
        if value < 1024 or unit == "EB":
            return f"{value:,.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def describe_eddy_outputs(path, pattern: str = EDDY_SYNC_PATTERN, label: str = "Eddy outputs") -> dict:
    path = Path(path)
    files = sorted(p for p in path.glob(pattern) if p.is_file()) if path.exists() else []
    total_bytes = sum(p.stat().st_size for p in files)
    summary = {
        "label": label,
        "path": path,
        "exists": path.exists(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sample_names": [p.name for p in files[:10]],
    }
    print(label)
    print("-" * len(label))
    print(f"path       : {path}")
    print(f"exists     : {path.exists()}")
    print(f"file count : {len(files)}")
    print(f"total size : {fmt_bytes(total_bytes)}")
    if files:
        print("sample files:")
        for name in summary["sample_names"]:
            print(f"  - {name}")
    return summary


def _print_sync_preview(names, total_bytes=None, source_label=None, dest_dir=None, limit: int = 20) -> None:
    print("Eddy sync preview")
    print("-----------------")
    if source_label is not None:
        print(f"source      : {source_label}")
    if dest_dir is not None:
        print(f"destination : {dest_dir}")
    print(f"missing files: {len(names)}")
    if total_bytes is not None:
        print(f"total bytes : {fmt_bytes(total_bytes)}")
    if not names:
        print("Nothing to copy. Destination already has all matching Eddy outputs.")
        return
    print("\nFiles to add")
    for name in names[:limit]:
        print(f"  + {name}")
    if len(names) > limit:
        print(f"  ... and {len(names) - limit} more")


def _stream_subprocess(cmd) -> int:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="")
    finally:
        if process.stdout is not None:
            process.stdout.close()
    return process.wait()


def _rsync_cmd(
    remote_source: str,
    dest_dir,
    pattern: str = EDDY_SYNC_PATTERN,
    dry_run: bool = True,
    rsync_rsh: str = None,
    extra_rsync_args=None,
):
    cmd = [
        "rsync",
        "-avh",
        "--ignore-existing",
        "--prune-empty-dirs",
        "--include",
        pattern,
        "--exclude",
        "*",
    ]
    if dry_run:
        cmd.append("--dry-run")
    else:
        cmd.append("--info=progress2,stats1")
    if rsync_rsh:
        cmd.extend(["-e", rsync_rsh])
    if extra_rsync_args:
        cmd.extend(extra_rsync_args)
    cmd.extend([remote_source.rstrip("/") + "/", str(Path(dest_dir)) + "/"])
    return cmd


def preview_missing_eddy_outputs(
    source_dir,
    dest_dir,
    pattern: str = EDDY_SYNC_PATTERN,
    remote_source: str = None,
    rsync_rsh: str = None,
    extra_rsync_args=None,
) -> dict:
    dest_dir = ensure_dir(dest_dir)
    source_dir = Path(source_dir)

    if source_dir.exists():
        source_files = sorted(p for p in source_dir.glob(pattern) if p.is_file())
        existing = {p.name for p in dest_dir.glob(pattern) if p.is_file()}
        missing_files = [p for p in source_files if p.name not in existing]
        missing_names = [p.name for p in missing_files]
        total_bytes = sum(p.stat().st_size for p in missing_files)
        _print_sync_preview(
            missing_names,
            total_bytes=total_bytes,
            source_label=source_dir,
            dest_dir=dest_dir,
        )
        return {
            "mode": "local",
            "source_dir": source_dir,
            "dest_dir": dest_dir,
            "missing_files": missing_files,
            "missing_names": missing_names,
            "total_bytes": total_bytes,
        }

    if not remote_source:
        raise FileNotFoundError(
            f"Source directory {source_dir} is not visible. "
            "Pass remote_source='ec2-user@<host>:~/exp/data/derivatives/eddy' to use rsync over SSH."
        )

    cmd = _rsync_cmd(
        remote_source=remote_source,
        dest_dir=dest_dir,
        pattern=pattern,
        dry_run=True,
        rsync_rsh=rsync_rsh,
        extra_rsync_args=extra_rsync_args,
    )
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    output_text = "\n".join(part for part in [result.stdout, result.stderr] if part)
    if result.returncode != 0:
        raise RuntimeError(f"rsync dry run failed with code {result.returncode}\n{output_text}")

    missing_names = []
    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(
            (
                "sending incremental file list",
                "receiving incremental file list",
                "sent ",
                "total size is ",
                "created directory ",
            )
        ):
            continue
        if line.endswith(".mif"):
            missing_names.append(Path(line).name)

    _print_sync_preview(missing_names, source_label=remote_source, dest_dir=dest_dir)
    return {
        "mode": "rsync",
        "remote_source": remote_source,
        "dest_dir": dest_dir,
        "missing_files": missing_names,
        "missing_names": missing_names,
        "total_bytes": None,
    }


def sync_missing_eddy_outputs(
    source_dir,
    dest_dir,
    pattern: str = EDDY_SYNC_PATTERN,
    remote_source: str = None,
    dry_run: bool = True,
    rsync_rsh: str = None,
    extra_rsync_args=None,
    chunk_size: int = 32 * 1024 * 1024,
):
    plan = preview_missing_eddy_outputs(
        source_dir=source_dir,
        dest_dir=dest_dir,
        pattern=pattern,
        remote_source=remote_source,
        rsync_rsh=rsync_rsh,
        extra_rsync_args=extra_rsync_args,
    )
    if dry_run:
        print("\nDry run only. No files were copied.")
        return plan

    if not plan["missing_names"]:
        print("\nDestination is already up to date.")
        return plan

    if plan["mode"] == "rsync":
        cmd = _rsync_cmd(
            remote_source=plan["remote_source"],
            dest_dir=plan["dest_dir"],
            pattern=pattern,
            dry_run=False,
            rsync_rsh=rsync_rsh,
            extra_rsync_args=extra_rsync_args,
        )
        print("\nRunning rsync sync")
        print("------------------")
        print(" ".join(cmd))
        return_code = _stream_subprocess(cmd)
        if return_code != 0:
            raise RuntimeError(f"rsync sync failed with code {return_code}")
        print("\nRsync sync complete.")
        return plan

    copied_paths = []
    total_bytes = plan["total_bytes"] or 0
    missing_files = plan["missing_files"]
    if tqdm is None:
        print("tqdm is not installed; falling back to file-by-file progress messages.")
        for index, src in enumerate(missing_files, start=1):
            dest = plan["dest_dir"] / src.name
            print(f"[{index}/{len(missing_files)}] Copying {src.name}")
            with src.open("rb") as fsrc, dest.open("wb") as fdst:
                while True:
                    chunk = fsrc.read(chunk_size)
                    if not chunk:
                        break
                    fdst.write(chunk)
            shutil.copystat(src, dest)
            copied_paths.append(dest)
    else:
        progress = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Copying Eddy outputs")
        try:
            for src in missing_files:
                dest = plan["dest_dir"] / src.name
                progress.set_postfix(file=src.name)
                with src.open("rb") as fsrc, dest.open("wb") as fdst:
                    while True:
                        chunk = fsrc.read(chunk_size)
                        if not chunk:
                            break
                        fdst.write(chunk)
                        progress.update(len(chunk))
                shutil.copystat(src, dest)
                copied_paths.append(dest)
        finally:
            progress.close()

    print(f"\nCopied {len(copied_paths)} Eddy outputs into {plan['dest_dir']}")
    return {
        **plan,
        "copied_paths": copied_paths,
    }
