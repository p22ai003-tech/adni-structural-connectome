from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import AtlasConfig
from .qc import load_aal3_label_table


def build_aal3_node_table(atlas: AtlasConfig) -> pd.DataFrame:
    table = load_aal3_label_table(atlas)
    rows = []
    for original_label, node_index in sorted(table.original_to_node.items(), key=lambda item: item[1]):
        label_row = table.labels.loc[table.labels["atlas_value"] == original_label].iloc[0]
        rows.append(
            {
                "node_index": node_index,
                "original_aal_label": original_label,
                "node_name": label_row["node_name"],
                "atlas_label": label_row["atlas_label"],
            }
        )
    return pd.DataFrame(rows)


def write_aal3_node_table(atlas: AtlasConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_aal3_node_table(atlas).to_csv(path, sep="\t", index=False)
    return path


@dataclass(frozen=True)
class AAL3AtlasProducts:
    aal3_orig_b0_1mm: Path
    aal3_nodes_b0_1mm: Path
    aal3_node_table: Path
    aal3_b0_dwi_grid_qc: Path

    def to_dict(self) -> dict:
        return {
            "aal3_orig_b0_1mm": str(self.aal3_orig_b0_1mm),
            "aal3_nodes_b0_1mm": str(self.aal3_nodes_b0_1mm),
            "aal3_node_table": str(self.aal3_node_table),
            "aal3_b0_dwi_grid_qc": str(self.aal3_b0_dwi_grid_qc),
        }


def aal3_product_paths(out_dir: Path) -> AAL3AtlasProducts:
    return AAL3AtlasProducts(
        aal3_orig_b0_1mm=out_dir / "aal3_orig_b0_1mm.nii.gz",
        aal3_nodes_b0_1mm=out_dir / "aal3_nodes_b0_1mm.nii.gz",
        aal3_node_table=out_dir / "aal3_node_table.tsv",
        aal3_b0_dwi_grid_qc=out_dir / "aal3_b0_dwi_grid_qc.nii.gz",
    )


def contiguous_mapping_records(atlas: AtlasConfig) -> list[dict]:
    table = load_aal3_label_table(atlas)
    records: list[dict] = []
    for original, node in sorted(table.original_to_node.items(), key=lambda item: item[1]):
        records.append({"original_aal_label": original, "node_index": node, "atlas_label": table.label_lookup.get(original, "")})
    return records


def build_aal_to_t1_to_b0_command(
    ants_apply: str,
    aal_mni: Path,
    out_image: Path,
    reference_b0: Path,
    transforms: list[Path],
) -> tuple[str, ...]:
    command = [
        ants_apply,
        "-d",
        "3",
        "-i",
        str(aal_mni),
        "-r",
        str(reference_b0),
        "-o",
        str(out_image),
        "-n",
        "NearestNeighbor",
    ]
    for transform in transforms:
        command.extend(["-t", str(transform)])
    return tuple(command)


def build_contiguous_reindex_command(orig_label_image: Path, node_table_tsv: Path, out_nodes_image: Path) -> tuple[str, ...]:
    return (
        "python",
        "-m",
        "scforge.cli",
        "reindex-aal3-labels",
        str(orig_label_image),
        str(node_table_tsv),
        str(out_nodes_image),
    )


def build_qc_grid_copy_command(label_image: Path, dwi_reference: Path, out_image: Path) -> tuple[str, ...]:
    return (
        "mrtransform",
        str(label_image),
        str(out_image),
        "-template",
        str(dwi_reference),
        "-interp",
        "nearest",
    )
