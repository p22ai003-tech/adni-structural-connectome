from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AtlasConfig, QCConfig
from .qc import load_aal3_label_table


EXPECTED_GAPS = (35, 36, 81, 82)
VALID_ORIGINAL = tuple(x for x in range(1, 171) if x not in EXPECTED_GAPS)


@dataclass(frozen=True)
class AAL3ContractQC:
    image: str
    readable: bool
    n_valid_expected: int
    n_valid_present: int
    missing_valid_labels: tuple[int, ...]
    unexpected_gap_labels_present: tuple[int, ...]
    required_label_status: str
    aal_007_present: bool
    aal_008_present: bool
    min_present_voxels: int
    status: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["missing_valid_labels"] = ";".join(str(x) for x in self.missing_valid_labels)
        data["unexpected_gap_labels_present"] = ";".join(str(x) for x in self.unexpected_gap_labels_present)
        return data


def valid_original_labels(atlas: AtlasConfig | None = None) -> tuple[int, ...]:
    if atlas is None:
        return VALID_ORIGINAL
    gaps = set(int(x) for x in atlas.expected_gaps)
    return tuple(x for x in range(1, int(atlas.max_original_label) + 1) if x not in gaps)


def contiguous_mapping(atlas: AtlasConfig) -> dict[int, int]:
    return {original: idx for idx, original in enumerate(valid_original_labels(atlas), start=1)}


def node_table_dataframe(atlas: AtlasConfig) -> pd.DataFrame:
    table = load_aal3_label_table(atlas)
    rows: list[dict[str, Any]] = []
    for original, node in contiguous_mapping(atlas).items():
        rows.append(
            {
                "node_index": node,
                "original_aal_label": original,
                "atlas_label": table.label_lookup.get(original, f"AAL_{original:03d}"),
            }
        )
    return pd.DataFrame(rows)


def write_node_table(atlas: AtlasConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    node_table_dataframe(atlas).to_csv(path, sep="\t", index=False)
    return path


def build_contiguous_aal3_from_array(data: np.ndarray, atlas: AtlasConfig) -> tuple[np.ndarray, pd.DataFrame]:
    rounded = np.rint(data).astype(np.int32)
    out = np.zeros_like(rounded, dtype=np.int16)
    for original, node in contiguous_mapping(atlas).items():
        out[rounded == original] = node
    return out, node_table_dataframe(atlas)


def label_survival_from_array(data: np.ndarray, atlas: AtlasConfig, *, min_voxels: int = 1) -> AAL3ContractQC:
    rounded = np.rint(data).astype(np.int32)
    counts = {int(label): int((rounded == int(label)).sum()) for label in valid_original_labels(atlas)}
    missing = tuple(label for label, count in counts.items() if count < int(min_voxels))
    gap_present = tuple(label for label in atlas.expected_gaps if int((rounded == int(label)).sum()) > 0)
    required = {int(label): int(counts.get(int(label), 0)) >= int(min_voxels) for label in atlas.required_labels}
    present_counts = [count for count in counts.values() if count >= int(min_voxels)]
    status = "PASS" if not missing and not gap_present and all(required.values()) else "FAIL"
    return AAL3ContractQC(
        image="array",
        readable=True,
        n_valid_expected=len(counts),
        n_valid_present=len(present_counts),
        missing_valid_labels=missing,
        unexpected_gap_labels_present=gap_present,
        required_label_status=";".join(f"{label}:{int(ok)}" for label, ok in required.items()),
        aal_007_present=bool(required.get(7, False)),
        aal_008_present=bool(required.get(8, False)),
        min_present_voxels=int(min(present_counts) if present_counts else 0),
        status=status,
    )


def _load_image(path: str | Path):
    try:
        import nibabel as nib
    except Exception as exc:  # pragma: no cover - dependency check
        raise RuntimeError(f"nibabel is required for AAL3 image operations: {exc}") from exc
    return nib.load(str(path))


def label_survival_from_image(path: str | Path, atlas: AtlasConfig, qc: QCConfig | None = None) -> AAL3ContractQC:
    path = Path(path)
    try:
        image = _load_image(path)
        min_voxels = 1 if qc is None else int(qc.min_label_voxels_1mm)
        result = label_survival_from_array(np.asarray(image.get_fdata()), atlas, min_voxels=min_voxels)
        return AAL3ContractQC(**{**asdict(result), "image": str(path)})
    except Exception as exc:
        return AAL3ContractQC(
            image=str(path),
            readable=False,
            n_valid_expected=len(valid_original_labels(atlas)),
            n_valid_present=0,
            missing_valid_labels=valid_original_labels(atlas),
            unexpected_gap_labels_present=(),
            required_label_status=";".join(f"{label}:0" for label in atlas.required_labels),
            aal_007_present=False,
            aal_008_present=False,
            min_present_voxels=0,
            status="FAIL",
            reason=f"read_error:{type(exc).__name__}:{exc}",
        )


def write_contiguous_from_image(orig_path: str | Path, out_path: str | Path, atlas: AtlasConfig) -> tuple[Path, Path]:
    import nibabel as nib

    orig_path = Path(orig_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.load(str(orig_path))
    out, table = build_contiguous_aal3_from_array(np.asarray(image.get_fdata()), atlas)
    header = image.header.copy()
    header.set_data_dtype(np.int16)
    nib.save(nib.Nifti1Image(out, image.affine, header), str(out_path))
    table_path = out_path.with_name("aal3_node_table.tsv")
    table.to_csv(table_path, sep="\t", index=False)
    return out_path, table_path
