"""Independent Recovery2 tests for FSL-to-ITK affine conversion."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml


ROOT = Path("/home/ec2-user/exp")
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
CONTRACT = ROOT / "scforge/workflow/environment_contract_recovery2_convert3d.yaml"
LOCK = ROOT / "scforge/workflow/locks/convert3d-1.3.0-recovery2-linux-64.explicit.txt"
C3D = ROOT / ".envs/convert3d-1.3.0-recovery2/bin/c3d_affine_tool"
CONVERT_XFM = Path("/home/ec2-user/fsl/bin/convert_xfm")
FLIRT = Path("/home/ec2-user/fsl/bin/flirt")
TRANSFORMCONVERT = Path("/home/ec2-user/mrtrix3/bin/transformconvert")
ANTS_APPLY = ROOT / ".envs/ants-2.6.5/bin/antsApplyTransforms"
ANTS_INFO = ROOT / ".envs/ants-2.6.5/bin/antsTransformInfo"
EXPECTED_C3D_SHA256 = "ea5a0bdd79ea419ff37feccb202218cdc7c14c1f8adcdf099ce70dd273c937d6"
def _diverse_units(count: int = 4) -> tuple[str, ...]:
    """Units to exercise the converter on.

    The original selection was four units chosen for vendor and T1-format
    diversity (GE and Siemens, single-NIfTI and DICOM-series T1). Those were
    named by ADNI participant identifier, which cannot live in a tracked file,
    so the list is taken from the gitignored local file when it is there and
    otherwise discovered from the run root. The test needs units that exist and
    carry the three inputs, not particular ones.
    """
    try:
        import sc_exclusions
        named = sc_exclusions.subject_list("recovery2_diverse_units")
    except Exception:
        named = ()
    if named:
        return tuple(named)
    base = RUN_ROOT / "subjects"
    if not base.is_dir():
        return ()
    found = [
        d.name for d in sorted(base.iterdir())
        if (d / "02_anat/t1_n4.nii.gz").is_file()
        and (d / "03_spatial/b0_to_t1_bbr.mat").is_file()
        and (d / "03_spatial/mean_b0.nii.gz").is_file()
    ]
    return tuple(found[:count])


DIVERSE_UNITS = _diverse_units()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_recovery2_converter_contract_is_exact() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "TESTED_CANDIDATE"
    assert Path(contract["binary"]["path"]).resolve() == C3D.resolve()
    assert contract["binary"]["sha256"] == EXPECTED_C3D_SHA256
    assert _sha256(C3D) == EXPECTED_C3D_SHA256
    assert Path(contract["explicit_conda_lock"]["path"]).resolve() == LOCK.resolve()
    assert C3D.is_file() and os.access(C3D, os.X_OK)
    assert "convert3d-1.3.0-h4bd325d_0" in LOCK.read_text(encoding="utf-8")
    linked = subprocess.run(
        ["ldd", str(C3D)], check=True, capture_output=True, text=True
    ).stdout
    assert "not found" not in linked


def test_recovery2_converter_roundtrip_and_cross_backend_resampling() -> None:
    environment = dict(os.environ)
    environment.update(
        {
            "FSLDIR": "/home/ec2-user/fsl",
            "FSLOUTPUTTYPE": "NIFTI_GZ",
            "PATH": ":".join(
                (
                    "/home/ec2-user/fsl/bin",
                    "/home/ec2-user/mrtrix3/bin",
                    "/home/ec2-user/exp/.envs/ants-2.6.5/bin",
                    "/usr/bin",
                )
            ),
        }
    )
    if not DIVERSE_UNITS:
        import pytest
        pytest.skip("no processed units available under the recovery run root")
    metrics: list[dict[str, float | str]] = []
    for unit in DIVERSE_UNITS:
        subject = RUN_ROOT / "subjects" / unit
        b0_to_t1 = subject / "03_spatial/b0_to_t1_bbr.mat"
        t1 = subject / "02_anat/t1_n4.nii.gz"
        b0 = subject / "03_spatial/mean_b0.nii.gz"
        assert all(path.is_file() for path in (b0_to_t1, t1, b0))
        with tempfile.TemporaryDirectory(prefix=f"recovery2_{unit}_") as temporary:
            temporary = Path(temporary)
            fsl = temporary / "t1_to_b0.mat"
            mrtrix = temporary / "t1_to_b0_mrtrix.txt"
            itk = temporary / "t1_to_b0_itk.txt"
            roundtrip = temporary / "t1_to_b0_roundtrip.mat"
            fsl_image = temporary / "t1_in_b0_fsl.nii.gz"
            ants_image = temporary / "t1_in_b0_ants.nii.gz"
            _run(
                [str(CONVERT_XFM), "-omat", str(fsl), "-inverse", str(b0_to_t1)],
                environment=environment,
            )
            _run(
                [
                    str(TRANSFORMCONVERT),
                    str(fsl),
                    str(t1),
                    str(b0),
                    "flirt_import",
                    str(mrtrix),
                ],
                environment=environment,
            )
            _run(
                [
                    str(C3D),
                    "-ref",
                    str(b0),
                    "-src",
                    str(t1),
                    str(fsl),
                    "-fsl2ras",
                    "-oitk",
                    str(itk),
                ],
                environment=environment,
            )
            info = _run([str(ANTS_INFO), str(itk)], environment=environment).stdout
            assert "Number of transforms = 1" in info
            assert "Singular: 0" in info
            _run(
                [
                    str(C3D),
                    "-ref",
                    str(b0),
                    "-src",
                    str(t1),
                    "-itk",
                    str(itk),
                    "-ras2fsl",
                    "-o",
                    str(roundtrip),
                ],
                environment=environment,
            )
            original_matrix = np.loadtxt(fsl)
            roundtrip_matrix = np.loadtxt(roundtrip)
            assert np.max(np.abs(original_matrix - roundtrip_matrix)) < 1e-3
            assert abs(
                np.linalg.det(original_matrix[:3, :3])
                - np.linalg.det(roundtrip_matrix[:3, :3])
            ) < 1e-5
            _run(
                [
                    str(FLIRT),
                    "-in",
                    str(t1),
                    "-ref",
                    str(b0),
                    "-applyxfm",
                    "-init",
                    str(fsl),
                    "-interp",
                    "trilinear",
                    "-out",
                    str(fsl_image),
                ],
                environment=environment,
            )
            _run(
                [
                    str(ANTS_APPLY),
                    "-d",
                    "3",
                    "-i",
                    str(t1),
                    "-r",
                    str(b0),
                    "-o",
                    str(ants_image),
                    "-n",
                    "Linear",
                    "-t",
                    str(itk),
                ],
                environment=environment,
            )
            fsl_result = nib.load(fsl_image)
            ants_result = nib.load(ants_image)
            x = fsl_result.get_fdata(dtype=np.float32)
            y = ants_result.get_fdata(dtype=np.float32)
            assert x.shape == y.shape
            assert np.max(np.abs(fsl_result.affine - ants_result.affine)) == 0
            finite = np.isfinite(x) & np.isfinite(y)
            xv = x[finite]
            yv = y[finite]
            scale = max(float(np.percentile(np.abs(xv), 99.5)), 1e-8)
            correlation = float(np.corrcoef(xv, yv)[0, 1])
            normalized_mae = float(np.mean(np.abs(xv - yv))) / scale
            support_x = xv > 0
            support_y = yv > 0
            support_dice = float(
                2
                * np.count_nonzero(support_x & support_y)
                / (np.count_nonzero(support_x) + np.count_nonzero(support_y))
            )
            metrics.append(
                {
                    "unit": unit,
                    "correlation": correlation,
                    "normalized_mae": normalized_mae,
                    "support_dice": support_dice,
                }
            )
    assert all(float(item["correlation"]) > 0.99 for item in metrics), metrics
    assert all(float(item["normalized_mae"]) < 0.02 for item in metrics), metrics
    assert all(float(item["support_dice"]) > 0.93 for item in metrics), metrics
