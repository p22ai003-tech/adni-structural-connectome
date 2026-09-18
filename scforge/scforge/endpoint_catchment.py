from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EndpointCatchmentResult:
    input_nodes: str
    output: str
    mode: str
    radius_mm: float
    iterations: int
    labels_expanded: int
    voxels_added: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_nodes": self.input_nodes,
            "output": self.output,
            "mode": self.mode,
            "radius_mm": self.radius_mm,
            "iterations": self.iterations,
            "labels_expanded": self.labels_expanded,
            "voxels_added": self.voxels_added,
        }


def build_endpoint_catchment(
    nodes_image: str | Path,
    output: str | Path,
    *,
    radius_mm: float = 3.0,
    constraint_mask: str | Path | None = None,
) -> EndpointCatchmentResult:
    """Create a local endpoint catchment image from a contiguous AAL3 node image.

    The first implementation is deliberately local and conservative: it expands
    labels by a small voxel radius, optionally constrained by a mask, and never
    overwrites existing labeled voxels. This replaces blind long forward-search
    assignment with an explicit scratch artifact that can be QC'ed.
    """

    import nibabel as nib

    nodes_image = Path(nodes_image)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = nib.load(str(nodes_image))
    data = np.rint(np.asarray(image.get_fdata())).astype(np.int16)

    if constraint_mask:
        mask_img = nib.load(str(constraint_mask))
        allowed = np.asarray(mask_img.get_fdata()) > 0
        mode = "local_dilation_constrained"
    else:
        allowed = np.ones(data.shape, dtype=bool)
        mode = "local_dilation_unconstrained"

    zooms = tuple(float(x) for x in image.header.get_zooms()[:3])
    min_zoom = min(x for x in zooms if x > 0) if zooms else 1.0
    iterations = max(1, int(round(float(radius_mm) / min_zoom)))

    try:
        from scipy import ndimage as ndi

        structure = ndi.generate_binary_structure(3, 1)
        out = data.copy()
        occupied = out > 0
        labels = [int(x) for x in np.unique(data) if int(x) > 0]
        voxels_added = 0
        expanded = 0
        for label in labels:
            core = data == label
            grown = ndi.binary_dilation(core, structure=structure, iterations=iterations)
            add = grown & allowed & ~occupied
            if add.any():
                out[add] = label
                occupied[add] = True
                voxels_added += int(add.sum())
                expanded += 1
    except Exception:
        out = data.copy()
        voxels_added = 0
        expanded = 0
        iterations = 0
        mode = "copy_no_scipy"

    header = image.header.copy()
    header.set_data_dtype(np.int16)
    nib.save(nib.Nifti1Image(out.astype(np.int16), image.affine, header), str(output))
    return EndpointCatchmentResult(
        input_nodes=str(nodes_image),
        output=str(output),
        mode=mode,
        radius_mm=float(radius_mm),
        iterations=int(iterations),
        labels_expanded=int(expanded),
        voxels_added=int(voxels_added),
    )
