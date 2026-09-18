#!/usr/bin/env python3
"""
ADNI DWI bias-field correction + first b0 (ANTs-preferred, mask-aware),
plus an optional b0 backfill mode for existing *_unbiased.mif files.

This consolidates:
- Notebook Cell 1  → main biascorr_1 + b0 export
- Notebook Cell 2  → simple b0 extractor (superseded)
- Notebook Cell 3  → robust b0 backfill for missing _b0.nii.gz

Usage (examples)
----------------
# Run full bias correction + b0 export on all series, skipping existing outputs
python bias_correction.py

# Same, but overwrite existing outputs
python bias_correction.py --force

# Only patch missing _b0.nii.gz from existing *_unbiased.mif
python bias_correction.py --b0-backfill-only

# Restrict to specific series IDs
python bias_correction.py --series <SUBJECT>_I<IMAGEID> <SUBJECT>_I<IMAGEID>

# Debug mode: run only failing series with verbose MRtrix/FSL output
python bias_correction.py --debug --series <SUBJECT>_I<IMAGEID> <SUBJECT>_I<IMAGEID>

You can also tune parallelism:
python bias_correction.py --processes 4 --threads-per-job 2 --run-mode thread
"""

import os
import csv
import shutil
import tempfile
import subprocess
import sys
import time
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    class _TqdmFallback:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
            self.total = total

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n=1):
            return None

        def write(self, msg):
            print(msg)

    def tqdm(iterable=None, *args, **kwargs):
        return _TqdmFallback(iterable=iterable, total=kwargs.get("total"))

try:
    import SimpleITK as sitk
except ModuleNotFoundError:
    sitk = None

# ---------- DEFAULT CONFIG (can be overridden via CLI) ----------
def _default_deriv() -> Path:
    """Derivatives root, from the canonical resolver (SC_DERIV_ROOT overrides)."""
    try:
        import sc_config  # noqa: PLC0415
    except ModuleNotFoundError:  # pragma: no cover
        import sys

        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import sc_config  # noqa: PLC0415
    return sc_config.paths().deriv_root


DEFAULT_DERIV           = _default_deriv()
DEFAULT_PROCESSES       = min(2, max(1, (os.cpu_count() or 4) // 3))  # parallel series
DEFAULT_THREADS_PER_JOB = 1                                           # OpenMP threads per job
DEFAULT_FORCE           = False                               # overwrite existing outputs if True
DEFAULT_STAGE_ROOT      = Path.home() / "staging" / "biascorr_stage"
DEFAULT_RUN_MODE        = "thread"                            # "thread" or "process"
DEFAULT_DEBUG           = False                               # quiet by default
DEFAULT_ALLOW_FSL_FALLBACK = False
# ---------------------------------------------------------------

# These globals will be updated from CLI args in main()
DERIV: Path           = DEFAULT_DERIV
PROCESSES: int        = DEFAULT_PROCESSES
THREADS_PER_JOB: int  = DEFAULT_THREADS_PER_JOB
FORCE: bool           = DEFAULT_FORCE
STAGE_ROOT: Path      = DEFAULT_STAGE_ROOT
RUN_MODE: str         = DEFAULT_RUN_MODE
DEBUG: bool           = DEFAULT_DEBUG
ALLOW_FSL_FALLBACK: bool = DEFAULT_ALLOW_FSL_FALLBACK
SELECT_SERIES: List[str] = []

IN_DIR: Path          = DERIV / "eddy"
OUT_BC: Path          = DERIV / "biascorr_1"
QC_DIR: Path          = DERIV / "qc"
LOG_DIR: Path         = QC_DIR / "biascorr_logs"
QC_CSV: Path          = QC_DIR / "biascorr_summary.csv"

# ANTs + MRtrix / FSL tools --------------------------------------------------


def need(tool: str, required: bool = True) -> bool:
    """Check if a CLI tool is available in PATH."""
    from shutil import which
    ok = which(tool) is not None
    if required and not ok:
        raise RuntimeError(f"Required tool '{tool}' not found in PATH")
    return ok


# MRtrix & friends (fail fast if missing)
for t in (
    "mrconvert",
    "mrmath",
    "mrstats",
    "mrinfo",
    "mrcalc",
    "dwibiascorrect",
    "dwiextract",
    "dwi2mask",
):
    need(t)

# Optional FSL BET (for mask fallback)
HAS_BET: bool = need("bet", required=False)

# Resolve ANTs install (we rely on PATH / ANTSPATH)
ANTS_BIN: Optional[Path] = None
ANTS_LIB: Optional[str] = None
N4_CMD: Optional[str] = None

_ants_candidates = [
    os.environ.get("ANTSPATH", ""),
    shutil.which("N4BiasFieldCorrection") or "",
    str(Path.home() / "bin" / "N4BiasFieldCorrection"),
    str(Path.home() / ".local" / "bin" / "N4BiasFieldCorrection"),
    str(Path.home() / "miniconda3" / "bin" / "N4BiasFieldCorrection"),
    str(Path.home() / "miniconda3" / "envs" / "antsfix" / "bin" / "N4BiasFieldCorrection"),
    str(Path(os.environ["ANTSPATH"]) / "N4BiasFieldCorrection") if os.environ.get("ANTSPATH") else "",
    shutil.which("N4BiasFieldCorrection") or "",
]
_ants_hint = next((p for p in _ants_candidates if p), "")
if _ants_hint:
    ANTS_BIN = Path(_ants_hint)
    if ANTS_BIN.is_file():  # if the env points to the executable, not the dir
        N4_CMD = str(ANTS_BIN)
        ANTS_BIN = ANTS_BIN.parent
    if ANTS_BIN.exists():
        n4_candidate = ANTS_BIN / "N4BiasFieldCorrection"
        if n4_candidate.exists():
            N4_CMD = str(n4_candidate)
        env_root = ANTS_BIN.parent
        lib_dir = env_root / "lib"
        if lib_dir.exists():
            ANTS_LIB = str(lib_dir)


# ------------------------ Helper functions -----------------------------------


def _run(cmd, env=None, log_file: Optional[Path] = None):
    """
    Run a command, optionally logging stdout/stderr to a file.

    Behaviour:
    - Normal mode: quiet (stdout/stderr → logfile or /dev/null).
    - DEBUG mode: print commands, stream output to terminal, and still
      save to logfile if provided.
    """
    cmd_str = " ".join(map(str, cmd))

    if DEBUG:
        print(f"[CMD] {cmd_str}")

    # With a logfile
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        if DEBUG:
            # Capture output once, write to log, and echo to stdout
            proc = subprocess.run(
                cmd,
                check=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            with open(log_file, "a") as lf:
                lf.write("\n> " + cmd_str + "\n")
                lf.write(proc.stdout)
            print(proc.stdout, end="")
            return proc
        else:
            with open(log_file, "a") as lf:
                lf.write("\n> " + cmd_str + "\n")
                return subprocess.run(
                    cmd, check=True, stdout=lf, stderr=lf, env=env
                )

    # No logfile
    if DEBUG:
        # Stream output to terminal
        return subprocess.run(cmd, check=True, env=env)
    else:
        # Quiet
        return subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )


def _env_for_job(tmpdir: Path):
    """Build environment for MRtrix/ANTs job with controlled threading."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(max(1, int(THREADS_PER_JOB)))
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(
        max(1, int(THREADS_PER_JOB))
    )
    env["TMPDIR"] = str(tmpdir)

    # Make sure ANTs is visible to each worker
    if ANTS_BIN and str(ANTS_BIN):
        env["PATH"] = f"{str(ANTS_BIN)}:{env.get('PATH', '')}"
        env["ANTSPATH"] = str(ANTS_BIN)

    # Help the dynamic loader find ANTs libs (macOS/Linux)
    if ANTS_LIB:
        if sys.platform == "darwin":
            env["DYLD_FALLBACK_LIBRARY_PATH"] = (
                f"{ANTS_LIB}:{env.get('DYLD_FALLBACK_LIBRARY_PATH', '')}"
            )
        else:
            env["LD_LIBRARY_PATH"] = f"{ANTS_LIB}:{env.get('LD_LIBRARY_PATH', '')}"

    return env


def _std_of_mif(path: Path) -> float:
    out = subprocess.check_output(
        ["mrstats", "-output", "std", str(path)], text=True
    ).split()[0]
    return float(out)


def _stat_of_mif(path: Path, stat: str) -> float:
    out = subprocess.check_output(
        ["mrstats", "-output", stat, str(path)], text=True
    ).split()[0]
    return float(out)


def _has_valid_gradients(path: Path) -> bool:
    proc = subprocess.run(
        ["mrinfo", "-shell_bvalues", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _extract_b0_reference(
    clean: Path,
    out_b0: Path,
    *,
    env: dict,
    log_file: Path,
) -> None:
    b0s = out_b0.with_name(out_b0.stem + "_all.mif")
    try:
        _run(
            ["dwiextract", "-bzero", str(clean), str(b0s), "-force"],
            env=env,
            log_file=log_file,
        )
        _run(
            ["mrmath", str(b0s), "mean", "-axis", "3", str(out_b0), "-force"],
            env=env,
            log_file=log_file,
        )
    except subprocess.CalledProcessError:
        with open(log_file, "a") as lf:
            lf.write("!! dwiextract -bzero failed; using volume 0 as pseudo-b0 reference\n")
        _run(
            ["mrconvert", str(clean), str(out_b0), "-coord", "3", "0", "-quiet", "-force"],
            env=env,
            log_file=log_file,
        )
    finally:
        try:
            if b0s.exists():
                b0s.unlink()
        except Exception:
            pass


def _run_b0_n4_fallback(
    *,
    clean: Path,
    mask_mif: Path,
    ub: Path,
    bias: Path,
    td: Path,
    env: dict,
    log_file: Path,
) -> str:
    """Recover bias correction from a 3D mean-b0 reference and apply it to all volumes."""
    b0_ref = td / "b0_ref.mif"
    b0_for_n4 = td / "b0_for_n4.mif"
    b0_nii = td / "b0_for_n4.nii.gz"
    mask_nii = td / "mask.nii.gz"
    bias_nii = td / "bias.nii.gz"
    bias_safe = td / "bias_safe.mif"

    _extract_b0_reference(clean, b0_ref, env=env, log_file=log_file)

    min_b0 = _stat_of_mif(b0_ref, "min")
    if min_b0 < 1.0:
        offset = 1.0 - min_b0 + 10.0
        with open(log_file, "a") as lf:
            lf.write(f"!! b0 reference min={min_b0:.4f}; shifting by +{offset:.4f} before N4\n")
        _run(
            ["mrcalc", str(b0_ref), str(offset), "-add", str(b0_for_n4), "-force"],
            env=env,
            log_file=log_file,
        )
    else:
        _run(
            ["mrconvert", str(b0_ref), str(b0_for_n4), "-quiet", "-force"],
            env=env,
            log_file=log_file,
        )

    _run(
        ["mrconvert", str(b0_for_n4), str(b0_nii), "-quiet", "-force"],
        env=env,
        log_file=log_file,
    )
    _run(
        ["mrconvert", str(mask_mif), str(mask_nii), "-quiet", "-force"],
        env=env,
        log_file=log_file,
    )
    if N4_CMD:
        _run(
            [
                N4_CMD,
                "-d",
                "3",
                "-i",
                str(b0_nii),
                "-w",
                str(mask_nii),
                "-s",
                "2",
                "-b",
                "[200]",
                "-c",
                "[200x200x100x50,1e-6]",
                "-o",
                f"[{td / 'b0_corrected.nii.gz'},{bias_nii}]",
                "-v",
            ],
            env=env,
            log_file=log_file,
        )
        method = "n4_b0_ants"
    else:
        if sitk is None:
            raise RuntimeError(
                "N4BiasFieldCorrection is unavailable and SimpleITK is not installed"
            )
        with open(log_file, "a") as lf:
            lf.write("!! N4BiasFieldCorrection not found; using SimpleITK N4 fallback on mean b0 (shrink=2)\n")
        image = sitk.Cast(sitk.ReadImage(str(b0_nii)), sitk.sitkFloat32)
        mask = sitk.Cast(sitk.ReadImage(str(mask_nii)) > 0, sitk.sitkUInt8)
        shrink = [2] * image.GetDimension()
        image_small = sitk.Shrink(image, shrink)
        mask_small = sitk.Cast(sitk.Shrink(mask, shrink) > 0, sitk.sitkUInt8)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([200, 200, 100, 50])
        corrector.SetConvergenceThreshold(1e-6)
        corrector.Execute(image_small, mask_small)
        log_bias = corrector.GetLogBiasFieldAsImage(image)
        bias_img = sitk.Exp(log_bias)
        corrected_img = sitk.Divide(image, bias_img)
        sitk.WriteImage(corrected_img, str(td / "b0_corrected.nii.gz"))
        sitk.WriteImage(bias_img, str(bias_nii))
        method = "n4_b0_sitk"
    _run(
        ["mrconvert", str(bias_nii), str(bias), "-quiet", "-force"],
        env=env,
        log_file=log_file,
    )
    _run(
        ["mrcalc", str(bias), "1e-6", "-max", str(bias_safe), "-force"],
        env=env,
        log_file=log_file,
    )
    _run(
        ["mrcalc", str(clean), str(bias_safe), "-div", str(ub), "-force"],
        env=env,
        log_file=log_file,
    )
    return method


def _pick_eddy_input(series: str) -> Path:
    """Pick between <series>_preproc.mif and <series>_eddy.mif."""
    p1 = IN_DIR / f"{series}_preproc.mif"
    p2 = IN_DIR / f"{series}_eddy.mif"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(
        f"No eddy output found for {series} (looked for {p1.name} and {p2.name})"
    )


def _series_list() -> List[str]:
    """Enumerate series IDs from eddy/."""
    seen = set()
    for p in sorted(IN_DIR.glob("*_preproc.mif")):
        if p.name.startswith("._"):
            continue
        seen.add(p.name.replace("_preproc.mif", ""))
    for p in sorted(IN_DIR.glob("*_eddy.mif")):
        if p.name.startswith("._"):
            continue
        s = p.name.replace("_eddy.mif", "")
        if s not in seen:
            seen.add(s)
    out = sorted(seen)
    if SELECT_SERIES:
        sel = set(SELECT_SERIES)
        out = [s for s in out if s in sel]
    return out


def _has_nontrivial_size(path: Path, *, min_bytes: int) -> bool:
    try:
        return path.exists() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def _outputs_look_complete(series: str) -> bool:
    ub_mif = OUT_BC / f"{series}_unbiased.mif"
    ub_nii = OUT_BC / f"{series}_unbiased.nii.gz"
    bf_mif = OUT_BC / f"{series}_biasfield.mif"
    b0_nii = OUT_BC / f"{series}_b0.nii.gz"
    return (
        _has_nontrivial_size(ub_mif, min_bytes=1_000_000)
        and _has_nontrivial_size(ub_nii, min_bytes=1_000_000)
        and _has_nontrivial_size(bf_mif, min_bytes=100_000)
        and _has_nontrivial_size(b0_nii, min_bytes=100_000)
    )


def _already_done(series: str) -> bool:
    """Check if all expected outputs already exist and look sane."""
    return _outputs_look_complete(series)


def _append_rows(rows: List[Dict[str, str]]):
    header = [
        "series",
        "status",
        "used",
        "input",
        "unbiased_mif",
        "unbiased_nii",
        "biasfield_mif",
        "b0_nii",
        "elapsed_sec",
        "log",
    ]
    write_header = not QC_CSV.exists()
    with QC_CSV.open("a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=header)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


# ------------------------- Main worker (biascorr) ----------------------------


def worker(series: str) -> dict:
    t0 = time.time()
    log = LOG_DIR / f"{series}.log"
    row = {
        "series": series,
        "status": "",
        "used": "",
        "input": "",
        "unbiased_mif": "",
        "unbiased_nii": "",
        "biasfield_mif": "",
        "b0_nii": "",
        "elapsed_sec": "",
        "log": str(log),
    }

    try:
        # --- resolve inputs/outputs ---
        inp = _pick_eddy_input(series)
        row["input"] = inp.name
        out_ub_mif = OUT_BC / f"{series}_unbiased.mif"
        out_ub_nii = OUT_BC / f"{series}_unbiased.nii.gz"
        out_bias = OUT_BC / f"{series}_biasfield.mif"
        out_b0 = OUT_BC / f"{series}_b0.nii.gz"

        # quick skip if everything's already there and looks sane
        if (not FORCE) and _outputs_look_complete(series):
            row.update(
                {
                    "status": "skip_exists",
                    "used": "existing",
                    "unbiased_mif": out_ub_mif.name,
                    "unbiased_nii": out_ub_nii.name,
                    "biasfield_mif": out_bias.name,
                    "b0_nii": out_b0.name,
                }
            )
            return row

        if FORCE:
            for p in (out_ub_mif, out_ub_nii, out_bias, out_b0):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

        with tempfile.TemporaryDirectory(dir=STAGE_ROOT) as td_:
            td = Path(td_)
            env = _env_for_job(td)

            staged = td / "in.mif"
            _run(
                ["mrconvert", str(inp), str(staged), "-quiet", "-force"],
                env=env,
                log_file=log,
            )

            # 0) scrub NaNs (N4 can fail on NaNs)
            clean = td / "in_clean.mif"
            _run(
                ["mrcalc", str(staged), "-isnan", "0", str(staged), "-if", str(clean), "-force"],
                env=env,
                log_file=log,
            )

            # 0b) brain mask (dwi2mask preferred; BET fallback on b0 if needed)
            mask_mif = td / "mask.mif"
            try:
                _run(
                    ["dwi2mask", str(clean), str(mask_mif), "-force"],
                    env=env,
                    log_file=log,
                )
            except subprocess.CalledProcessError:
                with open(log, "a") as lf:
                    lf.write(
                        "!! dwi2mask failed; falling back to BET on first volume\n"
                    )
                if not HAS_BET:
                    raise RuntimeError(
                        "dwi2mask failed and FSL BET not available for fallback."
                    )

                b0_1 = td / "b0_1.nii.gz"

                # Try proper b=0 extraction first
                try:
                    b0s = td / "b0s.mif"
                    _run(
                        ["dwiextract", "-bzero", str(clean), str(b0s), "-force"],
                        env=env,
                        log_file=log,
                    )
                    _run(
                        [
                            "mrconvert",
                            str(b0s),
                            str(b0_1),
                            "-coord",
                            "3",
                            "0",
                            "-quiet",
                            "-force",
                        ],
                        env=env,
                        log_file=log,
                    )
                except subprocess.CalledProcessError as e2:
                    # This is typically the “no valid diffusion gradient table” case
                    with open(log, "a") as lf:
                        lf.write(
                            "!! dwiextract -bzero failed in mask fallback "
                            "(no valid grad table?); using volume 0 as pseudo-b0 "
                            f"(rc={e2.returncode})\n"
                        )
                    _run(
                        [
                            "mrconvert",
                            str(clean),
                            str(b0_1),
                            "-coord",
                            "3",
                            "0",
                            "-quiet",
                            "-force",
                        ],
                        env=env,
                        log_file=log,
                    )

                # Now run BET on that volume and make a mask
                _run(
                    ["bet", str(b0_1), str(td / "b0_brain.nii.gz"), "-m", "-f", "0.2"],
                    env=env,
                    log_file=log,
                )
                _run(
                    [
                        "mrconvert",
                        str(td / "b0_brain_mask.nii.gz"),
                        str(mask_mif),
                        "-quiet",
                        "-force",
                    ],
                    env=env,
                    log_file=log,
                )

            ub = td / "unbiased.mif"
            bias = td / "biasfield.mif"
            ub_nii = td / "unbiased.nii.gz"
            b0_out = td / "b0.nii.gz"
            used = None

            # 1) Preferred: MRtrix wrapper with ANTs (mask-aware)
            if N4_CMD:
                try:
                    _run(
                        [
                            "dwibiascorrect",
                            "ants",
                            str(clean),
                            str(ub),
                            "-bias",
                            str(bias),
                            "-mask",
                            str(mask_mif),
                            "-force",
                        ],
                        env=env,
                        log_file=log,
                    )
                    used = "ants"
                except (subprocess.CalledProcessError, FileNotFoundError) as e1:
                    with open(log, "a") as lf:
                        lf.write(
                            f"!! dwibiascorrect ants failed ({type(e1).__name__}: {e1}); trying mean-b0 N4 fallback\n"
                        )
            else:
                with open(log, "a") as lf:
                    lf.write(
                        "!! ANTs CLI not found; skipping dwibiascorrect ants and using mean-b0 N4 fallback\n"
                    )

            if used is None:
                try:
                    used = _run_b0_n4_fallback(
                        clean=clean,
                        mask_mif=mask_mif,
                        ub=ub,
                        bias=bias,
                        td=td,
                        env=env,
                        log_file=log,
                    )
                except Exception as e2:
                    with open(log, "a") as lf:
                        lf.write(
                            f"!! mean-b0 N4 fallback failed ({type(e2).__name__}: {e2}); considering FSL fallback\n"
                        )
                    if not ALLOW_FSL_FALLBACK:
                        raise RuntimeError(
                            "mean-b0 N4 fallback failed and FSL fallback is disabled"
                        ) from e2
                    if not _has_valid_gradients(clean):
                        raise RuntimeError(
                            "mean-b0 N4 fallback failed and no valid DW gradient scheme is present for FSL fallback"
                        ) from e2
                    _run(
                        [
                            "dwibiascorrect",
                            "fsl",
                            str(clean),
                            str(ub),
                            "-bias",
                            str(bias),
                            "-mask",
                            str(mask_mif),
                            "-force",
                        ],
                        env=env,
                        log_file=log,
                    )
                    used = "fsl"

            # Sanity: unbiased not all zeros
            if _std_of_mif(ub) == 0.0:
                row["status"] = f"fail:unbiased_zero_after_{used or 'unknown'}"
                row["used"] = used or ""
                return row

            # Save unbiased as NIfTI
            _run(
                ["mrconvert", str(ub), str(ub_nii), "-quiet", "-force"],
                env=env,
                log_file=log,
            )

            # First b0 export
            b0_4d = td / "b0s.mif"
            b0_1 = td / "b0.mif"
            try:
                _run(
                    ["dwiextract", "-bzero", str(ub), str(b0_4d), "-force"],
                    env=env,
                    log_file=log,
                )
                _run(
                    [
                        "mrconvert",
                        str(b0_4d),
                        str(b0_1),
                        "-coord",
                        "3",
                        "0",
                        "-quiet",
                        "-force",
                    ],
                    env=env,
                    log_file=log,
                )
            except subprocess.CalledProcessError:
                with open(log, "a") as lf:
                    lf.write(
                        "!! dwiextract -bzero failed; taking volume 0 as b0\n"
                    )
                _run(
                    ["mrconvert", str(ub), str(b0_1), "-coord", "3", "0", "-quiet", "-force"],
                    env=env,
                    log_file=log,
                )

            _run(
                ["mrconvert", str(b0_1), str(b0_out), "-quiet", "-force"],
                env=env,
                log_file=log,
            )

            # Move staged outputs back only after the full job succeeds.
            shutil.move(str(ub), str(out_ub_mif))
            shutil.move(str(bias), str(out_bias))
            shutil.move(str(ub_nii), str(out_ub_nii))
            shutil.move(str(b0_out), str(out_b0))

        row.update(
            {
                "status": "ok",
                "used": used or "ants",
                "unbiased_mif": out_ub_mif.name,
                "unbiased_nii": out_ub_nii.name,
                "biasfield_mif": out_bias.name,
                "b0_nii": out_b0.name,
            }
        )
        return row

    except Exception as e:
        row["status"] = f"fail:{type(e).__name__}:{e}"
        return row
    finally:
        try:
            row["elapsed_sec"] = f"{time.time() - t0:.1f}"
        except Exception:
            pass


# ---------------------- b0 backfill (from Cell 3) ---------------------------


def backfill_b0() -> Dict[str, int]:
    """
    Fix / backfill *_b0.nii.gz for biascorr_1.

    For every <series>_unbiased.mif in OUT_BC:

    - If <series>_b0.nii.gz already exists → skip
    - Else:
        1) Try dwiextract -bzero unbiased.mif → b0s.mif
        2) If that fails (no grad table), fall back to volume 0:
               mrconvert unbiased.mif -coord 3 0 → b0.mif
        3) Save final b0 as NIfTI: <series>_b0.nii.gz
    """
    bc_dir = OUT_BC
    stage_root = STAGE_ROOT / "b0_fix"
    stage_root.mkdir(parents=True, exist_ok=True)

    if not bc_dir.is_dir():
        raise SystemExit(f"biascorr_1 folder not found: {bc_dir}")

    unbiased_files = sorted(bc_dir.glob("*_unbiased.mif"))
    unbiased_files = [p for p in unbiased_files if not p.name.startswith("._")]
    print(f"Found {len(unbiased_files)} *_unbiased.mif in {bc_dir}")

    if not unbiased_files:
        return {"total_unbiased": 0, "created": 0, "skipped": 0, "failed": 0}

    created = 0
    skipped = 0
    failed = 0

    for ub in tqdm(unbiased_files, desc="Backfilling b0", unit="series"):
        series = ub.name.replace("_unbiased.mif", "")
        out_b0 = bc_dir / f"{series}_b0.nii.gz"

        if out_b0.exists():
            skipped += 1
            continue

        # Use a private temp dir for this series
        with tempfile.TemporaryDirectory(dir=stage_root) as td_str:
            td = Path(td_str)
            tmp_b0s = td / "b0s.mif"
            tmp_b0 = td / "b0.mif"

            try:
                # --- attempt #1: dwiextract -bzero ---
                rc = subprocess.run(
                    ["dwiextract", "-bzero", str(ub), str(tmp_b0s), "-force"],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if rc.returncode == 0:
                    # take first b0 volume
                    subprocess.run(
                        [
                            "mrconvert",
                            str(tmp_b0s),
                            str(tmp_b0),
                            "-coord",
                            "3",
                            "0",
                            "-quiet",
                            "-force",
                        ],
                        check=True,
                    )
                else:
                    # dwiextract failed → fall back to volume 0 of unbiased
                    msg = rc.stderr.strip().splitlines()[0] if rc.stderr else ""
                    print(
                        f"[{series}] dwiextract -bzero failed (rc={rc.returncode}): {msg}\n"
                        f"  → falling back to first volume as b0"
                    )
                    subprocess.run(
                        [
                            "mrconvert",
                            str(ub),
                            str(tmp_b0),
                            "-coord",
                            "3",
                            "0",
                            "-quiet",
                            "-force",
                        ],
                        check=True,
                    )

            except subprocess.CalledProcessError as e:
                # Very defensive: if something still blows up, try volume 0 fallback
                print(
                    f"[{series}] ERROR running dwiextract/mrconvert: {e}\n"
                    f"  → attempting direct volume-0 fallback"
                )
                try:
                    subprocess.run(
                        [
                            "mrconvert",
                            str(ub),
                            str(tmp_b0),
                            "-coord",
                            "3",
                            "0",
                            "-quiet",
                            "-force",
                        ],
                        check=True,
                    )
                except subprocess.CalledProcessError as e2:
                    print(f"[{series}] FINAL FAIL: {e2}")
                    failed += 1
                    continue

            # --- save final b0 as NIfTI ---
            subprocess.run(
                ["mrconvert", str(tmp_b0), str(out_b0), "-quiet", "-force"],
                check=True,
            )
            created += 1
            print(f"[{series}] ✓ wrote {out_b0.name}")

    print("\n=== b0 backfill summary ===")
    print(f"Created new _b0.nii.gz files: {created}")
    print(f"Skipped (already had b0):    {skipped}")
    print(f"Hard failures (even vol0):   {failed}")
    return {
        "total_unbiased": len(unbiased_files),
        "created": created,
        "skipped": skipped,
        "failed": failed,
    }


# --------------------------- CLI / main() ------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="ADNI DWI bias-field correction + first b0 (ANTs-preferred, mask-aware)."
    )
    parser.add_argument(
        "--deriv",
        type=Path,
        default=DEFAULT_DERIV,
        help=f"Derivatives root (default: {DEFAULT_DERIV})",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=DEFAULT_PROCESSES,
        help="Number of parallel workers (series-level).",
    )
    parser.add_argument(
        "--threads-per-job",
        type=int,
        default=DEFAULT_THREADS_PER_JOB,
        help="OpenMP threads per MRtrix/ANTs job.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs for each series.",
    )
    parser.add_argument(
        "--run-mode",
        choices=("thread", "process"),
        default=DEFAULT_RUN_MODE,
        help="Parallel mode for series: 'thread' (default) or 'process'.",
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=DEFAULT_STAGE_ROOT,
        help=f"Disk-backed staging directory (default: {DEFAULT_STAGE_ROOT}).",
    )
    parser.add_argument(
        "--allow-fsl-fallback",
        action="store_true",
        help="Allow dwibiascorrect fsl only if ANTs/SimpleITK N4 rescue fails.",
    )
    parser.add_argument(
        "--series",
        nargs="*",
        default=None,
        help=(
            "Optional list of series IDs to process "
            "(e.g. <SUBJECT>_I<IMAGEID>). Default: all series in eddy/."
        ),
    )
    parser.add_argument(
        "--b0-backfill-only",
        action="store_true",
        help=(
            "Skip bias-field correction and only patch missing *_b0.nii.gz "
            "from existing *_unbiased.mif in biascorr_1/."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Debug mode: stream all MRtrix/FSL/ANTs commands to the terminal, "
            "and force single-worker execution for easier reading."
        ),
    )
    return parser.parse_args()


def configure_from_args(args):
    """Update globals from CLI args and ensure directories exist."""
    global DERIV, PROCESSES, THREADS_PER_JOB, FORCE, STAGE_ROOT, RUN_MODE, DEBUG
    global ALLOW_FSL_FALLBACK, SELECT_SERIES
    global IN_DIR, OUT_BC, QC_DIR, LOG_DIR, QC_CSV

    DERIV = args.deriv
    PROCESSES = max(1, int(args.processes))
    THREADS_PER_JOB = max(1, int(args.threads_per_job))
    FORCE = bool(args.force)
    STAGE_ROOT = args.stage_root
    RUN_MODE = args.run_mode
    DEBUG = bool(args.debug)
    ALLOW_FSL_FALLBACK = bool(getattr(args, "allow_fsl_fallback", DEFAULT_ALLOW_FSL_FALLBACK))
    SELECT_SERIES = args.series or []

    # In debug mode, force thread-mode + 1 worker (simpler to read output)
    if DEBUG:
        RUN_MODE = "thread"
        PROCESSES = 1
        print("⚙️  DEBUG mode enabled: forcing RUN_MODE=thread, PROCESSES=1")

    IN_DIR = DERIV / "eddy"
    OUT_BC = DERIV / "biascorr_1"
    QC_DIR = DERIV / "qc"
    LOG_DIR = QC_DIR / "biascorr_logs"
    QC_CSV = QC_DIR / "biascorr_summary.csv"

    for p in (OUT_BC, LOG_DIR, STAGE_ROOT):
        p.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)


def run_biascorr():
    series = _series_list()
    print("DEBUG: series count =", len(series), "example:", series[:5])

    if not series:
        print(f"No eddy outputs found in {IN_DIR}")
        return {
            "rows": [],
            "status_counts": Counter(),
            "total_series": 0,
            "scheduled_now": 0,
            "already_done": 0,
        }

    pending = [s for s in series if FORCE or not _already_done(s)]
    done_already = len(series) - len(pending)

    print(
        f"Found {len(series)} series | already_done={done_already} | "
        f"scheduled_now={len(pending)} | PARALLEL={PROCESSES} ({RUN_MODE}) | "
        f"OMP={THREADS_PER_JOB} | FORCE={FORCE} | STAGE_ROOT={STAGE_ROOT} | "
        f"ALLOW_FSL_FALLBACK={ALLOW_FSL_FALLBACK}"
    )

    if not pending:
        print("Nothing to do.")
        return {
            "rows": [],
            "status_counts": Counter({"skip_exists": len(series)}),
            "total_series": len(series),
            "scheduled_now": 0,
            "already_done": len(series),
        }

    if ANTS_BIN and ANTS_BIN.exists():
        print(f"Using ANTs from: {ANTS_BIN}")
    else:
        print("⚠️  ANTs not found on PATH/ANTSPATH; 'dwibiascorrect ants' may fail.")

    rows: List[Dict[str, str]] = []
    desc = "Bias (ANTs) + first b0"

    # Compute actual workers to use (DEBUG may have overridden PROCESSES)
    workers = max(1, PROCESSES)

    if RUN_MODE == "process":
        Executor = ProcessPoolExecutor
    else:
        Executor = ThreadPoolExecutor

    with Executor(max_workers=workers) as ex, tqdm(
        total=len(pending), unit="series", desc=desc, dynamic_ncols=True
    ) as pbar:
        futs = {ex.submit(worker, s): s for s in pending}
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            pbar.update(1)
            pbar.write(
                f"→ {r.get('series','<?>')}: {r.get('status','')} ({r.get('used','')})"
            )

    _append_rows(rows)
    status_counts = Counter(r.get("status", "") for r in rows)

    # Also dump a simple list of failed series, for convenience
    failed = [r["series"] for r in rows if r.get("status", "").startswith("fail")]
    failed_list_path = None
    if failed:
        failed_list_path = QC_DIR / "biascorr_failed_series.txt"
        with failed_list_path.open("w") as fp:
            fp.write("\n".join(failed) + "\n")
        print(f"⚠️  {len(failed)} series failed; list written to {failed_list_path}")

    print(f"✅ Bias-corrected DWIs → {OUT_BC}")
    print(f"🧾 Summary CSV        → {QC_CSV}")
    return {
        "rows": rows,
        "status_counts": status_counts,
        "total_series": len(series),
        "scheduled_now": len(pending),
        "already_done": done_already,
        "failed_list_path": str(failed_list_path) if failed_list_path else "",
        "qc_csv": str(QC_CSV),
        "out_dir": str(OUT_BC),
    }


def run_bias_correction(
    *,
    deriv: Path = DEFAULT_DERIV,
    processes: int = DEFAULT_PROCESSES,
    threads_per_job: int = DEFAULT_THREADS_PER_JOB,
    force: bool = DEFAULT_FORCE,
    stage_root: Path = DEFAULT_STAGE_ROOT,
    run_mode: str = DEFAULT_RUN_MODE,
    select_series: Optional[List[str]] = None,
    debug: bool = DEFAULT_DEBUG,
    allow_fsl_fallback: bool = DEFAULT_ALLOW_FSL_FALLBACK,
) -> Dict[str, object]:
    args = SimpleNamespace(
        deriv=Path(deriv),
        processes=processes,
        threads_per_job=threads_per_job,
        force=force,
        run_mode=run_mode,
        stage_root=Path(stage_root),
        series=list(select_series or []),
        debug=debug,
        allow_fsl_fallback=allow_fsl_fallback,
        b0_backfill_only=False,
    )
    configure_from_args(args)
    return run_biascorr()


def run_b0_backfill(
    *,
    deriv: Path = DEFAULT_DERIV,
    stage_root: Path = DEFAULT_STAGE_ROOT,
) -> Dict[str, int]:
    args = SimpleNamespace(
        deriv=Path(deriv),
        processes=1,
        threads_per_job=1,
        force=False,
        run_mode="thread",
        stage_root=Path(stage_root),
        series=[],
        debug=False,
        allow_fsl_fallback=False,
        b0_backfill_only=True,
    )
    configure_from_args(args)
    return backfill_b0()


def main():
    args = parse_args()
    configure_from_args(args)

    if args.b0_backfill_only:
        print(f"Running b0 backfill only in {OUT_BC}")
        backfill_b0()
        return

    run_biascorr()


if __name__ == "__main__":
    main()
