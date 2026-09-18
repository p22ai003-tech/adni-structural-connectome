from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .config import AtlasConfig, QCConfig


@dataclass(frozen=True)
class AAL3LabelTable:
    labels: pd.DataFrame
    original_to_node: dict[int, int]
    node_to_original: dict[int, int]
    label_lookup: dict[int, str]
    expected_gaps: tuple[int, ...]
    required_labels: tuple[int, ...]

    @property
    def n_valid_labels(self) -> int:
        return len(self.original_to_node)

    @property
    def valid_original_labels(self) -> tuple[int, ...]:
        return tuple(sorted(self.original_to_node))


@dataclass(frozen=True)
class MatrixQCResult:
    path: str
    readable: bool
    shape: tuple[int, int] | None
    label_space: str
    expected_nodes: int
    square: bool
    finite: bool
    symmetric: bool
    zero_diagonal: bool
    density: float | None
    nonzero_upper_edges: int | None
    valid_zero_rows: tuple[int, ...]
    pass_strict: bool
    status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["shape"] = "" if self.shape is None else f"{self.shape[0]}x{self.shape[1]}"
        data["valid_zero_rows"] = ";".join(str(x) for x in self.valid_zero_rows)
        data["reasons"] = ";".join(self.reasons)
        return data


@dataclass(frozen=True)
class ConnectomeBundleQCResult:
    """Fail-closed QC result for the complete frozen nine-matrix contract."""

    status: str
    pass_strict: bool
    expected_nodes: int
    matrix_names: tuple[str, ...]
    density: float | None
    zero_count_nodes: tuple[int, ...]
    failures: tuple[str, ...]
    matrix_summaries: dict[str, dict[str, object]]

    def to_dict(self) -> dict:
        return asdict(self)


def load_aal3_label_table(atlas: AtlasConfig) -> AAL3LabelTable:
    labels = pd.read_csv(atlas.labels_csv)
    required = {"node", "node_name", "atlas_label", "atlas_value"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"AAL3 label table missing columns {sorted(missing)}: {atlas.labels_csv}")
    labels = labels.copy()
    labels["node"] = pd.to_numeric(labels["node"], errors="coerce").astype("Int64")
    labels["atlas_value"] = pd.to_numeric(labels["atlas_value"], errors="coerce").astype("Int64")
    labels = labels.dropna(subset=["node", "atlas_value"]).copy()
    labels["node"] = labels["node"].astype(int)
    labels["atlas_value"] = labels["atlas_value"].astype(int)
    expected_gaps = tuple(sorted(int(x) for x in atlas.expected_gaps))
    observed = set(labels["atlas_value"])
    gap_observed = sorted(set(expected_gaps).intersection(observed))
    if gap_observed:
        raise ValueError(f"AAL3 expected gap labels are present in label table: {gap_observed}")
    original_to_node = {int(v): idx + 1 for idx, v in enumerate(sorted(observed))}
    node_to_original = {node: original for original, node in original_to_node.items()}
    label_lookup = dict(zip(labels["atlas_value"].astype(int), labels["atlas_label"].astype(str)))
    return AAL3LabelTable(
        labels=labels.sort_values("atlas_value").reset_index(drop=True),
        original_to_node=original_to_node,
        node_to_original=node_to_original,
        label_lookup=label_lookup,
        expected_gaps=expected_gaps,
        required_labels=tuple(sorted(int(x) for x in atlas.required_labels)),
    )


def aal3_label_summary(atlas: AtlasConfig) -> dict:
    table = load_aal3_label_table(atlas)
    observed = set(table.original_to_node)
    required_present = {label: label in observed for label in table.required_labels}
    all_original = set(range(1, atlas.max_original_label + 1))
    missing_original = sorted(all_original.difference(observed).difference(table.expected_gaps))
    return {
        "atlas": atlas.name,
        "labels_csv": str(atlas.labels_csv),
        "mni_image": str(atlas.mni_image),
        "n_valid_labels": table.n_valid_labels,
        "max_original_label": atlas.max_original_label,
        "expected_gaps": ";".join(str(x) for x in table.expected_gaps),
        "missing_unexpected_labels": ";".join(str(x) for x in missing_original),
        "required_labels_present": all(required_present.values()),
        "required_label_status": ";".join(f"{label}:{int(present)}" for label, present in required_present.items()),
        "contiguous_node_count": len(table.node_to_original),
    }


def read_numeric_matrix(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        matrix = pd.read_csv(path, header=None).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    except Exception:
        matrix = np.genfromtxt(path, delimiter=",", dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Matrix is not 2D: {path}")
    return matrix


def qc_connectome_bundle(
    matrices: Mapping[str, np.ndarray | str | Path],
    node_volumes_mm3: np.ndarray,
    *,
    expected_nodes: int = 166,
    symmetry_tolerance: float = 1e-8,
    zero_diagonal_tolerance: float = 1e-8,
    nonnegative_tolerance: float = 1e-12,
    integer_tolerance: float = 1e-6,
    duplicate_rtol: float = 1e-10,
    duplicate_atol: float = 1e-12,
    tensor_identity_tolerance: float = 1e-8,
    diffusivity_hard_max_mm2_per_s: float = 0.01,
) -> ConnectomeBundleQCResult:
    """Validate a complete connectome-v2 matrix bundle without sanitising it.

    Density is measured for reporting only.  It is deliberately absent from
    the pass/fail criteria, so this function cannot select processing routes or
    subjects on the basis of connectome density.
    """

    required = (
        "count",
        "fd_sum",
        "count_invnodevol",
        "len_mean",
        "invlen_mean",
        "fa_mean",
        "md_mean",
        "rd_mean",
        "ad_mean",
    )
    mean_names = ("len_mean", "invlen_mean", "fa_mean", "md_mean", "rd_mean", "ad_mean")
    failures: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, object]] = {}

    supplied = tuple(sorted(matrices))
    missing = sorted(set(required).difference(supplied))
    unexpected = sorted(set(supplied).difference(required))
    if missing:
        failures.append("missing_matrices:" + ",".join(missing))
    if unexpected:
        failures.append("unexpected_matrices:" + ",".join(unexpected))

    for name in required:
        if name not in matrices:
            continue
        try:
            source = matrices[name]
            array = read_numeric_matrix(source) if isinstance(source, (str, Path)) else np.asarray(source)
            if not np.issubdtype(array.dtype, np.number):
                raise TypeError("matrix is not numeric")
            array = array.astype(np.float64, copy=False)
        except Exception as exc:
            failures.append(f"{name}:read_error:{type(exc).__name__}:{exc}")
            continue

        shape = tuple(int(value) for value in array.shape)
        exact_shape = array.ndim == 2 and shape == (expected_nodes, expected_nodes)
        finite = bool(np.isfinite(array).all())
        symmetric = bool(exact_shape and finite and np.allclose(
            array, array.T, rtol=0.0, atol=symmetry_tolerance, equal_nan=False
        ))
        zero_diagonal = bool(
            exact_shape
            and finite
            and np.all(np.abs(np.diag(array)) <= zero_diagonal_tolerance)
        )
        nonnegative = bool(finite and np.all(array >= -nonnegative_tolerance))
        summaries[name] = {
            "shape": shape,
            "finite": finite,
            "symmetric": symmetric,
            "zero_diagonal": zero_diagonal,
            "nonnegative": nonnegative,
        }
        if not exact_shape:
            failures.append(f"{name}:shape_{shape}_expected_{expected_nodes}x{expected_nodes}")
        if not finite:
            failures.append(f"{name}:nonfinite")
        if not symmetric:
            failures.append(f"{name}:asymmetric")
        if not zero_diagonal:
            failures.append(f"{name}:nonzero_diagonal")
        if not nonnegative:
            failures.append(f"{name}:negative")
        if exact_shape:
            arrays[name] = array

    count = arrays.get("count")
    density: float | None = None
    zero_count_nodes: tuple[int, ...] = ()
    if count is not None and np.isfinite(count).all():
        if not np.allclose(count, np.rint(count), rtol=0.0, atol=integer_tolerance):
            failures.append("count:fractional")
        upper = np.triu(np.ones(count.shape, dtype=bool), 1)
        density = float(np.count_nonzero(count[upper] > integer_tolerance) / np.count_nonzero(upper))
        zero_count_nodes = tuple(
            int(index + 1)
            for index, value in enumerate(np.sum(np.abs(count), axis=1))
            if value <= integer_tolerance
        )
        # Node support is quantitative QC, not a source-compatible hard
        # validity gate.  Small canonical parcels can legitimately receive no
        # assigned inter-node streamline at a fixed tractogram size.  Preserve
        # the exact node list for blinded threshold/SAP decisions downstream.

    fd_sum = arrays.get("fd_sum")
    if count is not None and fd_sum is not None and np.isfinite(count).all() and np.isfinite(fd_sum).all():
        if np.allclose(count, fd_sum, rtol=duplicate_rtol, atol=duplicate_atol):
            failures.append("count_and_fd_sum:allclose_duplicate")

    volumes = np.asarray(node_volumes_mm3, dtype=np.float64)
    volumes_valid = (
        volumes.ndim == 1
        and volumes.shape == (expected_nodes,)
        and np.isfinite(volumes).all()
        and np.all(volumes > 0.0)
    )
    if not volumes_valid:
        failures.append("node_volumes_mm3:must_be_exactly_one_finite_positive_value_per_node")

    count_invnodevol = arrays.get("count_invnodevol")
    if count is not None and count_invnodevol is not None and volumes_valid and np.isfinite(count).all():
        expected = 2.0 * count / (volumes[:, None] + volumes[None, :])
        if not np.allclose(count_invnodevol, expected, rtol=1e-10, atol=1e-12):
            failures.append("count_invnodevol:formula_mismatch")

    if count is not None and np.isfinite(count).all():
        connected = count > integer_tolerance
        absent = ~connected
        for name in mean_names:
            matrix = arrays.get(name)
            if matrix is None or not np.isfinite(matrix).all():
                continue
            if np.any(np.abs(matrix[absent]) > zero_diagonal_tolerance):
                failures.append(f"{name}:nonzero_outside_count_support")
            if np.any(np.abs(matrix[connected]) <= nonnegative_tolerance):
                failures.append(f"{name}:zero_on_connected_edge")

        fa = arrays.get("fa_mean")
        if fa is not None and np.isfinite(fa).all():
            if np.any(fa[connected] < -nonnegative_tolerance) or np.any(fa[connected] > 1.0 + nonnegative_tolerance):
                failures.append("fa_mean:outside_0_to_1")

        tensor = {name: arrays.get(name) for name in ("md_mean", "rd_mean", "ad_mean")}
        for name, matrix in tensor.items():
            if matrix is not None and np.isfinite(matrix).all():
                values = matrix[connected]
                if np.any(values < -nonnegative_tolerance) or np.any(
                    values > diffusivity_hard_max_mm2_per_s + nonnegative_tolerance
                ):
                    failures.append(f"{name}:outside_diffusivity_range")
        if all(matrix is not None and np.isfinite(matrix).all() for matrix in tensor.values()):
            md = tensor["md_mean"]
            rd = tensor["rd_mean"]
            ad = tensor["ad_mean"]
            assert md is not None and rd is not None and ad is not None
            if np.any(ad[connected] + tensor_identity_tolerance < md[connected]):
                failures.append("tensor_order:ad_below_md")
            if np.any(md[connected] + tensor_identity_tolerance < rd[connected]):
                failures.append("tensor_order:md_below_rd")
            identity = (ad[connected] + 2.0 * rd[connected]) / 3.0
            if not np.allclose(md[connected], identity, rtol=0.0, atol=tensor_identity_tolerance):
                failures.append("tensor_identity:md_ne_ad_plus_2rd_over_3")

    unique_failures = tuple(dict.fromkeys(failures))
    passed = not unique_failures
    return ConnectomeBundleQCResult(
        status="PASS" if passed else "FAIL",
        pass_strict=passed,
        expected_nodes=expected_nodes,
        matrix_names=tuple(name for name in required if name in matrices),
        density=density,
        zero_count_nodes=zero_count_nodes,
        failures=unique_failures,
        matrix_summaries=summaries,
    )


def qc_matrix(
    path: str | Path,
    *,
    expected_nodes: int = 166,
    density_min: float = 0.15,
    strict_zero_valid_rows: int = 0,
    expected_gap_labels: tuple[int, ...] = (),
    max_original_label: int | None = None,
    allow_original_label_space: bool = True,
) -> MatrixQCResult:
    reasons: list[str] = []
    path = Path(path)
    try:
        matrix = read_numeric_matrix(path)
    except Exception as exc:
        return MatrixQCResult(
            path=str(path),
            readable=False,
            shape=None,
            label_space="unknown",
            expected_nodes=expected_nodes,
            square=False,
            finite=False,
            symmetric=False,
            zero_diagonal=False,
            density=None,
            nonzero_upper_edges=None,
            valid_zero_rows=(),
            pass_strict=False,
            status="FAIL",
            reasons=(f"read_error:{type(exc).__name__}:{exc}",),
        )

    shape = tuple(int(x) for x in matrix.shape)
    square = shape[0] == shape[1]
    if not square:
        reasons.append("nonsquare")

    label_space = "contiguous_nodes"
    row_labels = list(range(1, shape[0] + 1))
    col_labels = list(range(1, shape[1] + 1))
    max_original_label = int(max_original_label or expected_nodes)
    legacy_original = (
        allow_original_label_space
        and square
        and shape == (max_original_label, max_original_label)
        and max_original_label != expected_nodes
    )
    if legacy_original:
        label_space = "original_aal3_with_gaps"
        valid_original = [label for label in range(1, max_original_label + 1) if label not in set(expected_gap_labels)]
        row_labels = valid_original
        col_labels = valid_original
        if len(valid_original) != expected_nodes:
            reasons.append(f"valid_original_label_count_{len(valid_original)}_expected_{expected_nodes}")
    elif shape != (expected_nodes, expected_nodes):
        reasons.append(f"shape_{shape[0]}x{shape[1]}_expected_{expected_nodes}x{expected_nodes}")

    finite = bool(np.isfinite(matrix).all())
    if not finite:
        reasons.append("nonfinite")
    clean = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if legacy_original:
        row_idx = [label - 1 for label in row_labels]
        col_idx = [label - 1 for label in col_labels]
        qc_clean = clean[np.ix_(row_idx, col_idx)]
    else:
        qc_clean = clean
    n = min(qc_clean.shape)
    if n > 0:
        diagonal_abs = float(np.abs(np.diag(qc_clean[:n, :n])).sum())
        zero_diagonal = diagonal_abs <= 1e-8
        symmetric = bool(np.allclose(qc_clean[:n, :n], qc_clean[:n, :n].T, atol=1e-8))
    else:
        zero_diagonal = False
        symmetric = False
    if not zero_diagonal:
        reasons.append("nonzero_diagonal")
    if not symmetric:
        reasons.append("asymmetric")

    if n > 1:
        upper = np.triu(np.ones((n, n), dtype=bool), 1)
        nonzero_upper = int((np.abs(qc_clean[:n, :n]) > 0)[upper].sum())
        possible_upper = n * (n - 1) / 2
        density = float(nonzero_upper / possible_upper)
    else:
        nonzero_upper = 0
        density = 0.0
    if density < density_min:
        reasons.append(f"density_below_{density_min:g}")

    row_sums = np.abs(qc_clean).sum(axis=1) if qc_clean.size else np.array([])
    valid_zero_rows = tuple(int(row_labels[idx]) for idx, value in enumerate(row_sums[:expected_nodes]) if value == 0)
    if len(valid_zero_rows) > strict_zero_valid_rows:
        reasons.append(f"valid_zero_rows_{len(valid_zero_rows)}_gt_{strict_zero_valid_rows}")

    pass_strict = not reasons
    status = "PASS" if pass_strict else "FAIL"
    return MatrixQCResult(
        path=str(path),
        readable=True,
        shape=shape,
        label_space=label_space,
        expected_nodes=expected_nodes,
        square=square,
        finite=finite,
        symmetric=symmetric,
        zero_diagonal=zero_diagonal,
        density=density,
        nonzero_upper_edges=nonzero_upper,
        valid_zero_rows=valid_zero_rows,
        pass_strict=pass_strict,
        status=status,
        reasons=tuple(reasons),
    )


def matrix_qc_from_config(path: str | Path, qc: QCConfig) -> MatrixQCResult:
    return qc_matrix(
        path,
        expected_nodes=qc.required_valid_nodes,
        density_min=qc.density_fd_sum_min,
        strict_zero_valid_rows=qc.strict_zero_valid_rows,
    )


def matrix_qc_from_atlas_config(path: str | Path, qc: QCConfig, atlas: AtlasConfig) -> MatrixQCResult:
    return qc_matrix(
        path,
        expected_nodes=qc.required_valid_nodes,
        density_min=qc.density_fd_sum_min,
        strict_zero_valid_rows=qc.strict_zero_valid_rows,
        expected_gap_labels=tuple(atlas.expected_gaps),
        max_original_label=atlas.max_original_label,
        allow_original_label_space=True,
    )
