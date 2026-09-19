# atlas/: the brain-region maps

## What is this folder?

A **brain atlas** has two parts:

1. a 3-D brain image in which every **voxel** (a 1 mm cube of brain) holds a
   whole number saying which region it belongs to, and
2. a table that gives each number a name, such as `Hippocampus_L` (the left
   hippocampus).

A way of dividing the brain into regions is called a **parcellation**, and an
atlas is one parcellation. This project divides the brain into **166 regions**
using the **AAL3** atlas (Automated Anatomical Labelling, version 3). A
**connectome** is a table of how strongly each pair of brain regions is
connected. Each connectome matrix in this project has one row and one column per
region. This folder tells you
which brain region each matrix row stands for, and which larger brain network
each region belongs to.

## Do I need it?

| You want to... | Do you need this folder? |
|---|---|
| run the analysis | Yes, but you don't have to do anything. The files are already in the repository, and the analysis reads them from `atlas/AAL/` on its own. **Do not move or rename this folder**, because several analysis modules expect it in exactly this place. |
| read matrices or results yourself | Yes, whenever you turn a row number into a region name. Read [the key idea](#the-key-idea-row-numbers-are-not-atlas-numbers) first. |
| run the imaging half | The imaging half uses the AAL3 images here too. How it uses them is covered in the Imaging section of the top-level [README.md](../README.md). |

The analysis does not use `AAL116_spm12/`, `Schaefer2018/` or `nilearn/`. You
can ignore them.

## The key idea: row numbers are not atlas numbers

AAL3 numbers its regions from 1 to 170, but four of those numbers
(**35, 36, 81 and 82**) have no region. AAL3 dropped the old whole anterior
cingulate (35, 36) and whole thalamus (81, 82) and replaced them with smaller
parts that have higher numbers. That leaves 166 regions.

The matrices close these gaps and number the regions 1 to 166 with nothing
skipped. So from row 35 onward, a region's **matrix row number** and its
**atlas number** are different:

| matrix row | atlas number | region |
|---|---|---|
| 1 to 34 | 1 to 34 | the same number in both |
| 35 | 37 | `Cingulate_Mid_L` |
| 37 | 39 | `Cingulate_Post_L` |
| 39 | 41 | `Hippocampus_L` |
| 166 | 170 | `Raphe_M` |

That gives three rules:

- **Name matrix rows with `AAL/aal3_node_map_166.csv`.** Its `new_id` column is
  the matrix row number.
- **Never name matrix rows with `AAL/AAL3_labels.csv`.** Its `node` column is the
  *atlas* number, not the row number. If you look up row numbers in it, 132 of
  the 166 rows go wrong: rows 35, 36, 81 and 82 find no entry at all, and the
  other 128 rows from 37 onward get the wrong name. For example, it calls row 39
  `Cingulate_Post_L`, but row 39 is `Hippocampus_L`. (Its lines are in matrix
  order, so reading it line by line, where the k-th line after the header is
  row k, does give the right names. The mistake is looking rows up by the
  `node` column.)
- **Never match tables on `node_name`** (`AAL_037` and so on). In some tables the
  number in that name is the atlas number, and in others it is the row number.
  Always match on the row number (`new_id` in the node map, `matrix_idx` in the
  network table).

Rows are counted from 1: the first line of a matrix file is row 1
(`Precentral_L`). Python counts from 0, so row 1 is `M[0]` and row 39 is `M[38]`.

## What's inside

| Path | What it is | Used by |
|---|---|---|
| `AAL/aal3_node_map_166.csv` | **The table for naming matrix rows.** One line per region, with columns `new_id` (matrix row 1 to 166), `orig_value` (atlas number), `name` (region name) and `node_name` (`AAL_` plus the atlas number). | analysis, imaging |
| `AAL/AAL3_network_mapping.csv` | Places each region in a larger group. Columns: `matrix_idx` (matrix row), `atlas_value`, `atlas_label`, `functional_network` (10 groups, listed [below](#network-groups)), `anatomical_system` (9 groups) and `rationale` (one line on why). | analysis (network stages), dashboard |
| `AAL/AAL3_network_mapping_notes.md` | Explains the network table: how it is keyed, the borderline choices, and which regions are in each group. | people |
| `AAL/AAL3v1_1mm.nii.gz` | The original AAL3 (version 1) atlas image in MNI space (MNI = the standard template brain), 1 mm voxels, 181 × 217 × 181. Voxel values run from 1 to 170 and skip 35, 36, 81 and 82. | imaging |
| `AAL/AAL3v1_1mm_166.nii.gz` | The same image with the regions renumbered 1 to 166 and no gaps. A voxel's value here is its region's matrix row number. | imaging |
| `AAL/AAL3_labels.csv` | An older table keyed on the **atlas** number (columns `node`, `node_name`, `atlas_label`, `atlas_value`, `atlas_version`, where `node` is the atlas number). It is kept because code that needs the original numbers still reads it. **Do not use it to name matrix rows.** | imaging checks, older QC (quality-control) code, dashboard |
| `AAL/AAL3v1_1mm.nii.txt` | The label list that comes with AAL3v1. It has 170 lines of `number name number`, and lines 35, 36, 81 and 82 have no number at the end. | reference only |
| `AAL/ROI_MNI_V7_1mm_vol.txt` | The AAL3 size table. For each of the 166 regions it gives the short name, the full name, the atlas number, the voxel count and the volume in mm³. | reference only |
| `AAL/aal3_in_orig.txt`, `AAL/aal3_out_166.txt` | A pair of look-up tables in the format of `labelconvert`, the renumbering tool of MRtrix3 (the diffusion-MRI software the imaging half uses). One lists the original numbers and the other lists the new numbers 1 to 166, and they are matched by region name. Together they describe how `AAL3v1_1mm_166.nii.gz` relates to `AAL3v1_1mm.nii.gz`. | reference only |
| `AAL116_spm12/` | The older original AAL atlas (116 regions on a 2 mm grid) with its tools for SPM12, a brain-imaging toolbox that runs inside MATLAB (a commercial programming environment). It is kept for reference, and the pipeline does not use it. The image files (`aal/ROI_MNI_V4.nii`, `aal/atlas/AAL.nii` and `aal_for_SPM12.tar.gz`) are **not** in git. | nothing |
| `Schaefer2018/` | The Schaefer 2018 parcellation of the cortex (the brain's folded outer layer of grey matter): 200 regions in 17 networks, MNI space, 1 mm (a slightly different grid of 182 × 218 × 182). Only the imaging side uses it, as a second parcellation. The analysis does not read it. In the label file, each line's number is **one more** than the voxel value: the line `2  17Networks_LH_VisCent_ExStr_1` is voxel value 1, and line 1 is the background. | imaging only |
| `nilearn/` | An empty download folder made by the `nilearn` Python library. Only its `README.md` is in git. Nothing uses it, and it is safe to delete. | nothing |

In the "Used by" column, *analysis* is the analysis runner, *imaging* is the
half of the project that turns scans into matrices, and *dashboard* means the
result viewers in `apps/` (see [apps/README.md](../apps/README.md)).

### Network groups

`functional_network` in `AAL3_network_mapping.csv` (number of regions in
brackets): Subcortical (38), Cerebellar (26), DMN (20), Limbic (18), Visual (14),
Salience_VAN (14), Somatomotor (12), Brainstem (12), Frontoparietal (8),
DorsalAttention (4). DMN is the default mode network, and Salience_VAN is the
salience / ventral attention network. Seven of these groups are the seven
networks of the published Yeo 7-network map. The other three (Subcortical,
Cerebellar, Brainstem) are anatomical groups. Each region is assigned to exactly
one group, and `AAL3_network_mapping_notes.md` lists the borderline choices.

`anatomical_system`: Frontal (32), Thalamus (30), Cerebellum (26), Limbic (18),
Temporal (14), Parietal (14), Occipital (12), Brainstem (12), BasalGanglia (8).

## Before you start

1. Install the project as the top-level [README.md](../README.md) describes. It
   creates a **venv**, which is a private Python installation just for this
   project.
2. Open a terminal in the repository root (the folder that holds `sc_config.py`)
   and switch the venv on:

   ```bash
   source .venv/bin/activate
   ```

   Every command below is run from the repository root. On this page,
   `<repo>` stands for that folder, the one you cloned.
3. There is nothing to download. The AAL3 files the pipeline needs are already
   in the repository.
4. Steps 1 to 3 and 5 below need no ADNI data. Step 4 reads a connectome
   matrix. Real matrices are ADNI-derived data, and step 4 shows how to make a
   made-up one if you have none. ADNI data come only from
   [adni.loni.usc.edu](https://adni.loni.usc.edu/), under your own Data Use
   Agreement, and must never be committed to git.

## How to use it

### Step 1: check that the atlas files are the right ones

A **checksum** is a long fingerprint of a file. If even one byte differs, the
fingerprint is completely different.

```bash
sha256sum atlas/AAL/AAL3v1_1mm.nii.gz atlas/AAL/AAL3v1_1mm_166.nii.gz atlas/AAL/AAL3_labels.csv atlas/AAL/aal3_node_map_166.csv
```

On a Mac, use `shasum -a 256` in place of `sha256sum`.

What you should see is exactly this:

```
a3721422dffd9ca1345b9c33b95431d719a29d86ef0d5ef0c9806bdf31e9b700  atlas/AAL/AAL3v1_1mm.nii.gz
99185db14c0ed741c13a23306373cd4aa2478ff072ab4cbf05bd143f0511f415  atlas/AAL/AAL3v1_1mm_166.nii.gz
d10b81494243eb0c639c143978177da5439b29c522e0292847e6752795886f4a  atlas/AAL/AAL3_labels.csv
321391d605a4dc5b6742c156b3efdf815c37b124d677e6a058405d3b21d68939  atlas/AAL/aal3_node_map_166.csv
```

The same four values are recorded in `configs/connectome_v2.yaml`, under
`atlas:`.

### Step 2: check that the four AAL3 files agree with each other

Paste the whole block, from `python` down to the final `EOF`, into the terminal:

```bash
python - <<'EOF'
import nibabel as nib, numpy as np, pandas as pd

orig = np.asarray(nib.load("atlas/AAL/AAL3v1_1mm.nii.gz").dataobj).astype(int)
new = np.asarray(nib.load("atlas/AAL/AAL3v1_1mm_166.nii.gz").dataobj).astype(int)
node_map = pd.read_csv("atlas/AAL/aal3_node_map_166.csv")
networks = pd.read_csv("atlas/AAL/AAL3_network_mapping.csv")

print("atlas values with no voxels:", sorted(set(range(1, 171)) - set(np.unique(orig).tolist())))
print("regions in the 166 image:  ", len(np.unique(new)) - 1)

lut = np.zeros(orig.max() + 1, dtype=int)
lut[node_map["orig_value"]] = node_map["new_id"]
print("166 image = original, renumbered by the node map:", bool(np.array_equal(lut[orig], new)))

same = (networks["matrix_idx"].tolist() == node_map["new_id"].tolist()
        and networks["atlas_label"].tolist() == node_map["name"].tolist())
print("network table in the same order as the node map: ", same)
EOF
```

What you should see:

```
atlas values with no voxels: [35, 36, 81, 82]
regions in the 166 image:   166
166 image = original, renumbered by the node map: True
network table in the same order as the node map:  True
```

### Step 3: look up the name of a matrix row

```bash
python -c "
import pandas as pd
right = pd.read_csv('atlas/AAL/aal3_node_map_166.csv').set_index('new_id')['name']
wrong = pd.read_csv('atlas/AAL/AAL3_labels.csv').set_index('node')['atlas_label']
print('matrix row 39 -> right:', right[39], '| wrong:', wrong[39])
print('rows AAL3_labels.csv gets wrong (no entry or wrong name):', sum(wrong.get(i) != n for i, n in right.items()))
"
```

What you should see:

```
matrix row 39 -> right: Hippocampus_L | wrong: Cingulate_Post_L
rows AAL3_labels.csv gets wrong (no entry or wrong name): 132
```

To find which network a row belongs to, look it up by `matrix_idx`:

```bash
python -c "import pandas as pd; net = pd.read_csv('atlas/AAL/AAL3_network_mapping.csv'); print(net.loc[net.matrix_idx.isin([35, 37, 39]), ['matrix_idx','atlas_value','atlas_label','functional_network','anatomical_system']].to_string(index=False))"
```

```
 matrix_idx  atlas_value      atlas_label functional_network anatomical_system
         35           37  Cingulate_Mid_L       Salience_VAN            Limbic
         37           39 Cingulate_Post_L                DMN            Limbic
         39           41    Hippocampus_L                DMN            Limbic
```

### Step 4: put region names on a connectome matrix

A connectome matrix is a file named
`SC_AAL166_<SUBJECT>_I<IMAGEID>_<type>.csv` in `$SC_CONNECTOMES_DIR` (by
default `data/derivatives/connectomes/`). It has no header and holds 166 × 166
numbers separated by commas. `<type>` says what the numbers are. This step uses
`fd_sum`, the summed fibre density between two regions, which is the main
connection weight. Fibre density is an estimate of how much nerve fibre (the
brain's wiring) runs between two regions, so a bigger `fd_sum` means a stronger
connection.

**No ADNI data yet?** First make a made-up matrix in a temporary folder and point
the step at it. Paste all of these lines:

```bash
DEMO=$(mktemp -d)
python -c "
import numpy as np
M = np.zeros((166, 166))
M[0, 1] = M[1, 0] = 5.0      # row 1 Precentral_L  <-> row 2 Precentral_R
M[36, 38] = M[38, 36] = 2.5  # row 37 Cingulate_Post_L <-> row 39 Hippocampus_L
np.savetxt('$DEMO/SC_AAL166_XXX_S_0001_I000001_fd_sum.csv', M, delimiter=',')
"
export SC_CONNECTOMES_DIR="$DEMO"
```

`export` sets an **environment variable** (a named setting that programs
started from this terminal can read) until you close the terminal. When you are
done with the demo, run `unset SC_CONNECTOMES_DIR` to go back to the default
folder.

The block below takes the first `fd_sum` matrix it finds and labels its rows and
columns:

```bash
python - <<'EOF'
import numpy as np
import pandas as pd
import sc_config

# 1. The 166 region names, in matrix order (row 1 first).
names = pd.read_csv("atlas/AAL/aal3_node_map_166.csv").sort_values("new_id")["name"].tolist()

# 2. One connectome matrix. This takes the first fd_sum matrix it finds. To
#    choose a particular one, put its file name in place of the pattern
#    "SC_AAL166_*_fd_sum.csv" (the * matches any text).
folder = sc_config.paths().connectomes_dir
path = sorted(folder.glob("SC_AAL166_*_fd_sum.csv"))[0]
M = np.loadtxt(path, delimiter=",")

# 3. Put the names on the rows and the columns.
matrix = pd.DataFrame(M, index=names, columns=names)
print(matrix.shape)
print(matrix.loc[["Precentral_L", "Cingulate_Post_L"], ["Precentral_R", "Hippocampus_L"]])
EOF
```

What you should see is `(166, 166)` and then a small labelled table. With the
made-up matrix above, it is exactly this. With real data, the numbers depend on
your data:

```
(166, 166)
                  Precentral_R  Hippocampus_L
Precentral_L               5.0            0.0
Cingulate_Post_L           0.0            2.5
```

### Step 5 (optional): run the atlas tests

```bash
python -m pytest -q scforge/tests/test_atlas_contract_v2.py
```

What you should see is `2 passed`. These tests check that the renumbering from
1–170 to 1–166 skips exactly 35, 36, 81 and 82, and that atlas number 170 ends
up as region 166.

## Inputs and outputs

- **Inputs:** none. The pipeline does not produce anything in this folder.
- **Outputs:** none. Nothing writes into this folder.
- **Who reads what:**

| File | Read by |
|---|---|
| `AAL/aal3_node_map_166.csv` | the analysis stages that write region names (`edr_exceptions`, `build_consensus_core`, `build_exception_specificity`), the repair tool `connectome_analysis/relabel_edr_rois.py` and the imaging workflow |
| `AAL/AAL3_network_mapping.csv` | the `network_analysis` stage and the dashboard |
| `AAL/AAL3v1_1mm.nii.gz`, `AAL/AAL3v1_1mm_166.nii.gz` | the imaging workflow (see the Imaging section of the top-level README.md) |
| `AAL/AAL3_labels.csv` | the imaging workflow's atlas checks, older QC code (`connectome_analysis/analysis_sc_matrix_qc.py`) and the dashboards. The web dashboard's back end (`connectome_dashboard_core/connectomes.py`) takes the names by line order (line k = matrix row k), which gives the right names. The Streamlit dashboard (`apps/connectome_dashboard/connectome_app.py`) matches on the `node` column in its region views, which is the lookup [the key idea](#the-key-idea-row-numbers-are-not-atlas-numbers) warns against. |
| `Schaefer2018/*` | the imaging side only |

## Settings you can change

| Setting | Where | What it does | Default |
|---|---|---|---|
| `SC_ATLAS_ROOT` | environment variable, resolved in `sc_config.py` | where `sc_config` looks for `AAL/` | `<repo>/atlas` |

Leave `SC_ATLAS_ROOT` unset. Some analysis modules read `atlas/AAL/` inside the
repository directly, whatever the variable says, so pointing it somewhere else
would split the pipeline between two copies.

The tables are data, not settings. If you edit `AAL3_network_mapping.csv`, every
network-level result changes. If you edit `aal3_node_map_166.csv`, the region
names in the results change.

## Licences and where the files come from

The MIT licence in [`LICENSE`](../LICENSE) covers the project's **code**. For
this folder, `LICENSE` says:

- **MIT (this project's own work):** `aal3_node_map_166.csv`,
  `AAL3_network_mapping.csv` and `AAL3_network_mapping_notes.md`.
- **Their own licences (third-party):** the AAL3 atlas and the Schaefer 2018
  parcellation, which are redistributed here under their authors' terms.

| Files | Source | Licence | Cite |
|---|---|---|---|
| `AAL/AAL3v1_1mm.nii.gz`, `AAL/AAL3v1_1mm.nii.txt`, `AAL/ROI_MNI_V7_1mm_vol.txt` | AAL3 from the Groupe d'Imagerie Neurofonctionnelle (GIN), <https://www.gin.cnrs.fr/en/tools/aal/>. These files were first released as AAL3v1 on 10 June 2020. They are byte-for-byte the same (same SHA-256 checksum) as the files with the same names in GIN's current download, `AAL3v2_for_SPM12.tar.gz`, which keeps the version 1 file names. | The GIN page says AAL3v2 (5 April 2024) added the "GNU General Public Licence" to its readme as the operating licence. Follow those terms when you share these files. | Rolls ET, Huang CC, Lin CP, Feng J, Joliot M. Automated anatomical labelling atlas 3. *NeuroImage* 2020; 206: 116189 |
| `AAL/AAL3v1_1mm_166.nii.gz`, `AAL/AAL3_labels.csv`, `AAL/aal3_in_orig.txt`, `AAL/aal3_out_166.txt` | made by this project from the AAL3 files above (renumbered image and look-up tables that carry AAL3's region names) | `LICENSE` does not name these files. Since they are built from AAL3, follow the AAL3 terms when you share them. | cite AAL3 as above |
| `AAL116_spm12/` | AAL for SPM12, from the same GIN page | GNU General Public Licence, as stated in `AAL116_spm12/aal/readme_aal_for_SPM12.txt` | Tzourio-Mazoyer N et al. *NeuroImage* 2002; 15: 273–289 |
| `Schaefer2018/` | CBIG repository, <https://github.com/ThomasYeoLab/CBIG> (`stable_projects/brain_parcellation/Schaefer2018_LocalGlobal`) | that repository is MIT-licensed (Copyright (c) 2016 Computational Brain Imaging Group) | Schaefer A et al. Local-Global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. *Cerebral Cortex* 2018 |
| `nilearn/README.md` | written automatically by the nilearn library | nilearn's licence | — |

The functional network groups follow Yeo BTT et al., *Journal of
Neurophysiology* 2011; 106: 1125–1165.

None of these atlas files contain participant data. They are all standard-brain
templates.

## If something goes wrong

| What you see | Why | Fix |
|---|---|---|
| Step 1: `sha256sum: atlas/AAL/AAL3v1_1mm.nii.gz: No such file or directory`<br>Step 2: `FileNotFoundError: No such file or no access: 'atlas/AAL/AAL3v1_1mm.nii.gz'`<br>Step 3: `FileNotFoundError: [Errno 2] No such file or directory: 'atlas/AAL/aal3_node_map_166.csv'` | You are not in the repository root. | `cd` to the folder that holds `sc_config.py`, then run the command again. |
| `ModuleNotFoundError: No module named 'sc_config'` (step 4) | Same cause: Python can only find `sc_config.py` from the repository root. | Same fix. |
| `ModuleNotFoundError: No module named 'nibabel'` (or `pandas`, `numpy`) | The venv is not switched on. | `source .venv/bin/activate`. If the venv does not exist yet, follow the top-level README.md. |
| `IndexError: list index out of range` in step 4 | No `SC_AAL166_*_fd_sum.csv` files were found in the connectomes folder. | Print the folder being searched with `python -c "import sc_config; print(sc_config.paths().connectomes_dir)"`. Put your matrices there, or point the variable at the folder that holds them with `export SC_CONNECTOMES_DIR=/path/to/folder` and run step 4 again. With no data at all, use the made-up matrix in step 4. |
| A checksum in step 1 differs | The file changed, or it was not fully downloaded. | Restore the file from git with `git checkout -- atlas/AAL/<file>`, where `<file>` is the file's name, for example `AAL3_labels.csv`. |
| Step 2 prints `False`, or a list other than `[35, 36, 81, 82]`, or a number other than `166` | One of the atlas files was changed. | Run step 1 and `git status atlas/`. Restore every file whose checksum differs, or that `git status` lists as modified, with `git checkout -- atlas/AAL/<file>`. Then run step 2 again. |
| Only the `AAL3_labels.csv` checksum in step 1 differs, and `git status` shows no change | Your git changed the line endings when it wrote the file (the setting `core.autocrlf`, which is common on Windows). The other files in step 1 are not affected. The analysis reads the file the same way either way. | To get a matching checksum, run `git config core.autocrlf false`, delete the file, then run `git checkout -- atlas/AAL/AAL3_labels.csv`. |
| Region names in older result files look shifted (for example `Cingulate_Post_L` where you expected `Hippocampus_L`) | Those files (the `17_edr_exceptions` results; EDR, the exponential distance rule, says that longer connections are weaker) were written by looking rows up in `AAL3_labels.csv`. | Only the name columns are wrong, and the numbers are fine. `python -m connectome_analysis.relabel_edr_rois` reports what would change, and `--apply` rewrites the names. See [Repairing region names](../connectome_analysis/README.md#repairing-region-names-relabel_edr_roispy) in `connectome_analysis/README.md`. |
| You need `AAL116_spm12/aal/ROI_MNI_V4.nii` and it is missing | Atlas image files for this old atlas are not in git. | Download "AAL for SPM12" from the GIN AAL page. The pipeline does not need it. |
