from __future__ import annotations

import csv
import datetime as dt
import html
import json
import math
import statistics
import zipfile
from pathlib import Path


ROOT = Path("/home/ec2-user/exp")
OUT_DOCX = ROOT / "structural_connectome_context.docx"
OUT_MD = ROOT / "structural_connectome_context.md"

QC_ROOT = ROOT / "data/derivatives/qc/sc_matrix_qc"
LATEST_QC = QC_ROOT / "completed_connectomes_after_gap_20260524T085201Z"
FINAL_CLOSEOUT = QC_ROOT / "final_spatial_contract_closeout_20260521T114956Z"
SCFORGE_LATEST = QC_ROOT / "scforge_v1_density_latest.txt"


def read_csv_dict(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_text(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def esc(text: object) -> str:
    return html.escape("" if text is None else str(text), quote=False)


def slug_text(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def latest_density_group_summary(run_root: Path) -> list[dict[str, str]]:
    if not run_root.exists():
        return []
    baseline_path = run_root / "baseline_density.csv"
    comparison_path = run_root / "candidate_density_comparison.csv"
    if baseline_path.exists() and comparison_path.exists():
        baseline_rows = read_csv_dict(baseline_path)
        comparison_rows = read_csv_dict(comparison_path)
        completed_sids = {row.get("sid") or row.get("sample_id") for row in comparison_rows}
        remaining_rows = [
            row for row in baseline_rows if (row.get("sample_id") or row.get("sid")) not in completed_sids
        ]

        def fnum(value: object) -> float:
            try:
                if value in ("", None):
                    return math.nan
                val = float(value)
                return val if math.isfinite(val) else math.nan
            except Exception:
                return math.nan

        def mean(values: list[float]) -> str:
            values = [value for value in values if math.isfinite(value)]
            return str(statistics.mean(values)) if values else ""

        def median(values: list[float]) -> str:
            values = [value for value in values if math.isfinite(value)]
            return str(statistics.median(values)) if values else ""

        rows: list[dict[str, str]] = []
        for label, source_rows in [
            ("completed_baseline", comparison_rows),
            ("remaining_baseline", remaining_rows),
            ("completed_candidate_decisions", comparison_rows),
        ]:
            for group in ["ALL", "CN", "MCI", "AD"]:
                group_rows = source_rows if group == "ALL" else [r for r in source_rows if r.get("group") == group]
                baseline_density = [
                    fnum(r.get("baseline_valid_density") or r.get("present_density")) for r in group_rows
                ]
                candidate_density = [fnum(r.get("candidate_valid_density")) for r in group_rows]
                density_delta = [fnum(r.get("density_delta")) for r in group_rows]
                rows.append(
                    {
                        "set": label,
                        "group": group,
                        "n": str(len(group_rows)),
                        "baseline_density_mean": mean(baseline_density),
                        "baseline_density_median": median(baseline_density),
                        "candidate_density_mean": mean(candidate_density) if label == "completed_candidate_decisions" else "",
                        "candidate_density_median": median(candidate_density) if label == "completed_candidate_decisions" else "",
                        "density_delta_median": median(density_delta) if label == "completed_candidate_decisions" else "",
                    }
                )
        return rows
    files = sorted(run_root.glob("completed_vs_remaining_group_density_*.csv"))
    return read_csv_dict(files[-1]) if files else []


def latest_scforge_run() -> tuple[dict[str, object], list[dict[str, str]], Path | None]:
    if SCFORGE_LATEST.exists():
        run_root = Path(SCFORGE_LATEST.read_text(errors="replace").strip())
    else:
        runs = sorted(QC_ROOT.glob("scforge_v1_density_batch_*"))
        run_root = runs[-1] if runs else None
    if not run_root or not run_root.exists():
        return {}, [], None
    status_path = run_root / "status.json"
    status: dict[str, object] = {}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(errors="replace"))
        except Exception:
            status = {}
    return status, latest_density_group_summary(run_root), run_root


class SimpleDocx:
    def __init__(self) -> None:
        self.body: list[str] = []

    def paragraph(self, text: str = "", style: str | None = None) -> None:
        ppr = ""
        if style:
            ppr = f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>"
        runs = []
        for idx, part in enumerate(str(text).split("\n")):
            if idx:
                runs.append("<w:r><w:br/></w:r>")
            runs.append(f"<w:r><w:t xml:space=\"preserve\">{esc(part)}</w:t></w:r>")
        self.body.append(f"<w:p>{ppr}{''.join(runs)}</w:p>")

    def heading(self, text: str, level: int = 1) -> None:
        self.paragraph(text, f"Heading{max(1, min(level, 3))}")

    def bullet(self, text: str) -> None:
        self.body.append(
            "<w:p><w:pPr><w:pStyle w:val=\"ListParagraph\"/>"
            "<w:numPr><w:ilvl w:val=\"0\"/><w:numId w:val=\"1\"/></w:numPr></w:pPr>"
            f"<w:r><w:t xml:space=\"preserve\">{esc(text)}</w:t></w:r></w:p>"
        )

    def table(self, rows: list[list[object]]) -> None:
        if not rows:
            return
        tbl = [
            "<w:tbl><w:tblPr><w:tblStyle w:val=\"TableGrid\"/>"
            "<w:tblW w:w=\"0\" w:type=\"auto\"/>"
            "<w:tblBorders>"
            "<w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"A6A6A6\"/>"
            "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"A6A6A6\"/>"
            "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"A6A6A6\"/>"
            "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"A6A6A6\"/>"
            "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"A6A6A6\"/>"
            "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"A6A6A6\"/>"
            "</w:tblBorders></w:tblPr><w:tblGrid>"
        ]
        for _ in rows[0]:
            tbl.append("<w:gridCol w:w=\"2400\"/>")
        tbl.append("</w:tblGrid>")
        for r_idx, row in enumerate(rows):
            tbl.append("<w:tr>")
            for cell in row:
                shade = "<w:shd w:fill=\"D9E2F3\"/>" if r_idx == 0 else ""
                tbl.append(
                    f"<w:tc><w:tcPr>{shade}<w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
                    f"<w:p><w:r><w:t xml:space=\"preserve\">{esc(cell)}</w:t></w:r></w:p></w:tc>"
                )
            tbl.append("</w:tr>")
        tbl.append("</w:tbl>")
        self.body.append("".join(tbl))

    def page_break(self) -> None:
        self.body.append("<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>")

    def save(self, path: Path) -> None:
        document = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
            "<w:body>"
            + "".join(self.body)
            + "<w:sectPr><w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
            "<w:pgMar w:top=\"720\" w:right=\"720\" w:bottom=\"720\" w:left=\"720\" "
            "w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/></w:sectPr></w:body></w:document>"
        )
        styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:sz w:val="22"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:sz w:val="23"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="720"/></w:pPr></w:style>
  <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/><w:left w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/><w:right w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/></w:tblBorders></w:tblPr></w:style>
</w:styles>"""
        numbering = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="hybridMultilevel"/><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>"""
        content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>"""
        rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
        doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", rels)
            zf.writestr("word/document.xml", document)
            zf.writestr("word/styles.xml", styles)
            zf.writestr("word/numbering.xml", numbering)
            zf.writestr("word/_rels/document.xml.rels", doc_rels)


def make_markdown(
    stage_rows: list[dict[str, str]],
    qc_rows: list[dict[str, str]],
    canary_rows: list[dict[str, str]],
    experiment_rows: list[dict[str, str]],
    route_rows: list[dict[str, str]],
    scforge_status: dict[str, object],
    scforge_density_rows: list[dict[str, str]],
    scforge_run_root: Path | None,
) -> str:
    lines: list[str] = []
    add = lines.append
    today = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    add("# Structural Connectome Project Context for External Review")
    add("")
    add(f"Generated: {today}")
    add("")
    add("## Executive Summary")
    add("")
    add("This document summarizes the structural connectome pipeline under `/home/ec2-user/exp` for external review. The immediate problem is not missing files alone; it is anatomical validity of the completed structural connectivity matrices. Supervisor feedback identified whole zero rows/columns in valid AAL regions, and local QC confirms this is a broad whole-matrix issue.")
    add("")
    add("Current working conclusion: the current AAL3 connectome artifacts are not safe for blind downstream inference without strict QC gating. Some code-level improvements and positive routes were found, but no current AAL3 batch repair has passed enough strict canary validation to justify promoting failed subjects into analysis. Existing production outputs should be preserved; failed subjects should remain excluded until upstream spatial-contract reprocessing is corrected.")
    add("")
    add("Current implementation direction: SC-Forge is the replacement approach for connectome generation and recovery. It treats DWI source selection, anatomical registration, AAL3 label survival, 5TT/GMWMI validity, tractography, streamline assignment, matrix generation, and publication as explicit contracts with QC gates rather than a simple file-production chain.")
    add("")
    add("## Working Files and Roles")
    add("")
    for item in [
        "`notebooks/structural_connectome_A.ipynb`: early DWI pipeline, raw DICOM/DWI to MIF, denoise, Gibbs, Eddy/B0 provenance.",
        "`notebooks/structural_connectome_B.ipynb`: T1/BBR, 5TT/GMWMI, FOD, tractography, parcellation, DTI, connectome generation.",
        "`notebooks/structural_connectome_C.ipynb`: PhD analysis notebook and dashboard export source.",
        "`notebooks/structural_connectome_QC.ipynb`: SC matrix QC and rectification command center.",
        "`connectome_pipeline/connectome_step7.py`: Step 7 implementation used by notebooks and command-line/tmux wrappers.",
        "`scripts/preprocessing/ADNI_preproc_pipeline_AAL.sh`: supervisor/reference shell pipeline used as the spatial-contract benchmark.",
        "`apps/connectome_dashboard/connectome_app.py`: hosted Streamlit dashboard.",
        "`scforge/`: contract-first implementation package with configs, QC helpers, route selection, workflow rules, and tests.",
        "`run_scforge_v1_density_batch.py`: active scratch-only density-improvement batch runner.",
        "`run_sc_aal3_source_contract_probe.py`: active subject-level AAL3 source-contract probe used by SC-Forge v1.",
        "`run_sc_final_spatial_contract_closeout.py`: imported validation logic from the May 21 final spatial-contract closeout.",
        "`scripts/scforge/run_scforge_v2_canary.py`: retained SC-Forge v2 canary entrypoint for future validation work.",
        "`watch_scforge_v1_density_monitor.sh`: only active SC monitor kept at top level.",
    ]:
        add(f"- {item}")
    add("")
    add("## Atlas and Matrix Definition")
    add("")
    add("- Primary thesis atlas: AAL3, located at `/home/ec2-user/exp/atlas/AAL/AAL3v1_1mm.nii.gz`.")
    add("- Label table: `/home/ec2-user/exp/atlas/AAL/AAL3_labels.csv`.")
    add("- AAL3 valid labels: 166 valid nodes; maximum atlas label value is 170.")
    add("- Expected AAL3 label gaps: 35, 36, 81, 82. These are excluded from zero-row failure counts.")
    add("- Valid frontal opercular labels include AAL_007 = `Frontal_Inf_Oper_L` and AAL_008 = `Frontal_Inf_Oper_R`; these must not be treated as atlas gaps.")
    add("- Supervisor benchmark matrices are 116 x 116, consistent with an AAL116/AAL90-style SPM atlas route. They are used as quality benchmarks only, not as the primary thesis atlas.")
    add("")
    add("## Pipeline Methodology")
    add("")
    add("### Steps 1-4: DWI Preprocessing")
    add("")
    add("- Source selection is meant to use original/axial DTI series only, with bval/bvec/json validation, shell count, volume count, and gradient sanity checks.")
    add("- Denoise uses MRtrix `dwidenoise`; Gibbs correction uses `mrdegibbs`; geometry and gradients are checked after each step.")
    add("- Eddy is supervisor-compatible by default: `dwifslpreproc -rpe_none -pe_dir j- -eddy_options \"--slm=linear --data_is_shelled\"`.")
    add("- Eddy/B0 provenance is recorded so B0 source ambiguity can be audited later.")
    add("")
    add("### T1, BBR, 5TT, FOD, and Tracks")
    add("")
    add("- Reference T1 source from the supervisor shell is FreeSurfer-derived corrected T1/brainmask, for example `nu.mgz` and `brainmask.mgz` converted to NIfTI.")
    add("- Supervisor T1-to-B0 bridge uses FSL FLIRT with 6 DOF: `flirt -in corrected_T1.mgz -ref vol0000.nii.gz -omat t12b0.mat -out t12b0.nii.gz -dof 6`.")
    add("- 5TT/GMWMI are generated in the B0-aligned T1 space in the reference shell: `5ttgen fsl t12b0.nii.gz 5tt1.mif`, then `5tt2gmwmi`.")
    add("- FOD generation in the reference shell uses Dhollander response and SS3T CSD with the B0-space mask.")
    add("- Tractography reference shell uses ACT/GMWMI and 10M streamlines: `tckgen -act 5tt1.mif -seed_gmwmi gmwmSeed.mif -select 10000000`.")
    add("- Our production Step 7 default is 3M selected streamlines, chunked/resume-safe CPU parallelism. Track-count escalation alone was tested and did not solve the matrix zero-row problem when spatial alignment was bad.")
    add("")
    add("### Parcellation and Connectomes")
    add("")
    add("- Reference AAL route: AAL/MNI -> native T1 -> B0, using nearest-neighbour interpolation for labels.")
    add("- Step 7 now exposes explicit route options: `t1_b0_route`, `five_tt_route`, `aal_transform_route`, and `assignment_variant`.")
    add("- Implemented AAL transform route options include current direct/composed MNI->B0 and supervisor-style two-step native-T1 route.")
    add("- Implemented assignment variants include radial4, radial8, forward40, and forward80.")
    add("- Required connectome outputs are `ALL`, `count`, `fd_sum`, `len_mean`, `invlen_mean`, `fa_mean`, `md_mean`, `rd_mean`, `ad_mean`, and `count_invnodevol`.")
    add("- `fd_sum` is the primary structural weight for whole-matrix QC; it is generated using SIFT2 weights and `tck2connectome`.")
    add("")
    add("## SC-Forge Implementation")
    add("")
    add("SC-Forge is implemented under `/home/ec2-user/exp/scforge` with a package/module layout, configs, workflow rules, and tests. The durable design is contract-first and scratch-first:")
    add("")
    add("Implementation package map:")
    add("")
    for item in [
        "`scforge/configs/scforge.yaml`: project defaults, paths, QC thresholds, and command configuration.",
        "`scforge/configs/canary_subjects.yaml`: smoke/canary subject panels used before broad execution.",
        "`scforge/configs/aal3_valid_labels.yaml`: AAL3 valid-label and expected-gap contract.",
        "`scforge/scforge/manifest.py`: subject inventory and source-contract manifest logic.",
        "`scforge/scforge/provenance.py`: sidecar/provenance helpers for commands, inputs, outputs, and QC decisions.",
        "`scforge/scforge/dwi.py`, `anat.py`, `registration.py`, `atlas.py`, `five_tt.py`, `fod.py`, `tractography.py`, `connectome.py`: staged implementation modules.",
        "`scforge/scforge/qc.py`, `assignment_diversity_qc.py`, `spatial_contract.py`, `route_selector.py`, `reprocess_router.py`: QC gates, route selection, and reprocessing decisions.",
        "`scforge/workflow/Snakefile` and `workflow/rules/*.smk`: planned reproducible workflow entrypoints.",
        "`scforge/tests/*.py`: unit tests for AAL3 labels, affine sanity, matrix QC, assignment diversity, atlas contract v2, route selection, and reprocess routing.",
    ]:
        add(f"- {item}")
    add("")
    for item in [
        "Source contract: reject derived or ambiguous DWI, gradient mismatches, missing B0, missing bval/bvec/json, and orientation/affine problems before preprocessing.",
        "DWI preprocessing contract: preserve gradient provenance through conversion, denoise, Gibbs correction, Eddy, B0 extraction, and bias correction.",
        "Anatomical contract: prefer the supervisor-style FreeSurfer corrected T1/brainmask route when available and keep T1-to-B0 registration explicit.",
        "AAL3 spatial contract: transform labels with nearest-neighbour interpolation, track label survival, exclude expected AAL3 gaps 35/36/81/82, and never count AAL_007/AAL_008 as gaps.",
        "Tractography and assignment contract: generate ACT/GMWMI tracks, run `tck2connectome`, test assignment variants such as radial8 and forward40, and require assignment diversity checks.",
        "Matrix publication contract: write candidates in QC/scratch space first; only promote after backup, manifest, density/zero-row checks, required-matrix checks, and explicit review.",
    ]:
        add(f"- {item}")
    add("")
    add("The current SC-Forge v1 density lane reuses existing completed production tracks and replays the best AAL3 source-contract route in scratch space. It produces candidate matrices and decision manifests under QC; the runner itself does not overwrite production connectomes. Promotion is a separate audited step through `scripts/scforge/promote_scforge_v1_candidates.py`.")
    add("")
    if scforge_run_root:
        add("### Active SC-Forge v1 Density Batch")
        add("")
        add(f"- Run root: `{scforge_run_root}`")
        add(f"- Status updated: `{scforge_status.get('last_update_utc', '')}`")
        add(f"- Progress: `{scforge_status.get('completed_subjects', '')}/{scforge_status.get('total_subjects', '')}`")
        add(f"- Workers: `{scforge_status.get('workers', '')}` subject workers x `{scforge_status.get('mrtrix_threads', '')}` MRtrix threads per subject.")
        add(f"- Decisions so far: retained candidate `{scforge_status.get('retained_candidate_count', '')}`, retained current `{scforge_status.get('retained_current_count', '')}`, quarantine `{scforge_status.get('quarantine_count', '')}`, failed `{scforge_status.get('failed_subjects', '')}`.")
        add("- Attach monitor: `tmux attach -t scforge_v1_density_monitor`.")
        add("- Non-attached monitor: `cd /home/ec2-user/exp && ./watch_scforge_v1_density_monitor.sh`.")
        add("")
    if scforge_density_rows:
        add("### Completed Versus Remaining Density Snapshot")
        add("")
        add("This table separates subjects already processed by SC-Forge from those not processed yet. Remaining subjects only have baseline/current-production density because candidate matrices do not exist until SC-Forge processes them.")
        add("")
        add("| Set | Group | n | Baseline mean | Baseline median | Candidate mean | Candidate median | Median delta |")
        add("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in scforge_density_rows:
            add("| " + " | ".join(slug_text(row.get(c, "")) for c in [
                "set",
                "group",
                "n",
                "baseline_density_mean",
                "baseline_density_median",
                "candidate_density_mean",
                "candidate_density_median",
                "density_delta_median",
            ]) + " |")
        add("")
    add("### Current Top-Level Script Policy")
    add("")
    add("After the May 25 cleanup, only active SC-Forge density entrypoints and importable library modules should remain at `/home/ec2-user/exp` top level. Runnable helpers now live under `/home/ec2-user/exp/scripts`, app code under `/home/ec2-user/exp/apps`, and old evidence remains recoverable under `/home/ec2-user/exp/archive`.")
    add("")
    for item in [
        "Keep top-level while the density run is active: `run_scforge_v1_density_batch.py`, `run_sc_aal3_source_contract_probe.py`, `run_sc_final_spatial_contract_closeout.py`, `launch_scforge_v1_density_batch.sh`, and `watch_scforge_v1_density_monitor.sh`.",
        "Current non-live SC-Forge helpers live under `/home/ec2-user/exp/scripts/scforge`, including `run_scforge_v2_canary.py`, `promote_scforge_v1_candidates.py`, and the context generator.",
        "Do not move active runner/probe files while `scforge_v1_density_batch` is running because restarts and child probes call these paths directly.",
        "Use `/home/ec2-user/exp/scforge` for reusable implementation, configs, workflow rules, and tests.",
        "Use `/home/ec2-user/exp/archive` only for historical debugging scripts, old monitors, duplicate context files, sample matrices, and earlier canary launchers.",
    ]:
        add(f"- {item}")
    add("")
    add("## Current Output Counts")
    add("")
    add("Latest completed-connectome inventory from the EC2 derivatives:")
    add("")
    add("| Group | FOD | Tracks | Parc | DTI | Connectomes |")
    add("|---|---:|---:|---:|---:|---:|")
    for row in stage_rows:
        add("| " + " | ".join(slug_text(row.get(c, "")) for c in ["group", "FOD", "Tracks", "Parc", "DTI", "Connectomes"]) + " |")
    add("")
    add("## Strict Whole-Matrix QC Criteria")
    add("")
    for item in [
        "Completed connectome requires all required matrix weights to exist.",
        "Matrices must be readable, square, finite, symmetric, and zero diagonal.",
        "Primary whole-matrix QC uses `fd_sum`.",
        "Valid AAL3 zero rows are counted after excluding expected AAL3 gaps.",
        "AAL_b0 label coverage, label survival, AAL_007/AAL_008 survival, 5TT/B0 mask overlap, assignment availability, density, and unexpected zero rows are evaluated.",
        "Configured QC constants include MIN_LABEL_VOXELS=50, MIN_MASK_OVERLAP_FRAC=0.01, and WARN_VALID_ZERO_ROWS=25.",
        "A density gate of 0.15 has been used for strict repair/canary validation.",
    ]:
        add(f"- {item}")
    add("")
    add("### Current QC Breakdown")
    add("")
    add("| Group | Completed | PASS | WARN | FAIL_REGISTRATION | FAIL_LABEL_COVERAGE | FAIL_STREAMLINE_ASSIGNMENT | Analysis-ready | Median density | Median valid zero rows |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in qc_rows:
        add("| " + " | ".join(slug_text(row.get(c, "")) for c in [
            "group",
            "completed_connectomes",
            "PASS",
            "WARN_SPARSE_NODE",
            "FAIL_REGISTRATION",
            "FAIL_LABEL_COVERAGE",
            "FAIL_STREAMLINE_ASSIGNMENT",
            "analysis_ready_PASS_or_WARN",
            "median_density",
            "median_unexpected_zero_rows",
        ]) + " |")
    add("")
    add("Interpretation: the number of file-complete connectomes is much higher than the number of strict analysis-ready matrices. The current strict QC marks nearly all completed AAL3 matrices as needing registration, label-coverage, or assignment repair. This is intentionally conservative because whole zero rows in valid AAL3 regions can invalidate downstream graph measures.")
    add("")
    add("## Debugging and Repair Experiments Already Run")
    add("")
    add("| Experiment | Finding | Action |")
    add("|---|---|---|")
    for row in experiment_rows:
        add("| " + " | ".join(slug_text(row.get(c, "")) for c in ["experiment", "finding", "action"]) + " |")
    add("")
    add("### Final AAL3 Spatial-Contract Closeout")
    add("")
    if canary_rows:
        row = canary_rows[0]
        add("| Subjects tested | Completed | Strict passes | Improved but failed strict gate | Final recommendation |")
        add("|---:|---:|---:|---:|---|")
        add("| " + " | ".join(slug_text(row.get(c, "")) for c in [
            "total_subjects",
            "completed_subjects",
            "strict_pass_subjects",
            "improves_but_fails_strict_gate",
            "final_recommendation",
        ]) + " |")
    add("")
    add("The final recommendation from the closeout was `PATCH_BUT_KEEP_EXCLUDED_UNTIL_REPROCESS`. This means useful safeguards and route options should be retained, but current failed AAL3 artifacts should not be batch-promoted into analysis.")
    add("")
    add("### Representative AAL3 Route Results")
    add("")
    add("| Subject | Role | Route decision | Best variant | Baseline density | Candidate density | Baseline zero rows | Candidate zero rows | Final decision |")
    add("|---|---|---|---|---:|---:|---:|---:|---|")
    for row in route_rows[:10]:
        add("| " + " | ".join(slug_text(row.get(c, "")) for c in [
            "sid",
            "role",
            "route_decision",
            "best_variant",
            "baseline_fd_valid_density",
            "candidate_fd_valid_density",
            "baseline_fd_valid_zero_rows",
            "candidate_fd_valid_zero_rows",
            "decision",
        ]) + " |")
    add("")
    add("## What Appears to be Wrong")
    add("")
    add("The evidence points to a mixed spatial-contract problem rather than one isolated tckgen/track-count problem:")
    add("")
    for item in [
        "Severe subjects often have labels that fail before or during B0 projection, consistent with native T1/MNI label survival or T1-to-B0/mask-space failure.",
        "Some subjects have acceptable label coverage but still sparse rows, suggesting assignment-radius or streamline-assignment limitations after anatomy is valid.",
        "AAL116 with production tracks reached high density and zero rows in a positive benchmark, proving that healthy matrices are possible when the source contract is consistent.",
        "AAL3 source-contract canaries improved several subjects but did not pass enough strict full-matrix gates to justify batch repair.",
        "Increasing streamline count alone did not resolve failures when parcellation or mask geometry was wrong.",
    ]:
        add(f"- {item}")
    add("")
    add("## Recommended Conservative Action Plan")
    add("")
    add("1. Do not use all file-complete connectomes for final inference without QC gating.")
    add("2. Preserve all existing production outputs and keep failed repair outputs separate for evidence.")
    add("3. Retain code-level safeguards: explicit route config, nearest-neighbour label transforms, required-matrix generation including RD/AD, assignment variants, backup-before-replace, validation gate, and analysis gate.")
    add("4. For immediate analysis, use only PASS plus explicitly reviewed WARN; if this is too small, analysis must be labelled exploratory/provisional.")
    add("5. For thesis-grade final matrices, rerun upstream preprocessing for affected subjects using a fully validated source contract, then regenerate AAL3 parcellation and connectomes.")
    add("6. If external review suggests AAL3 is the wrong atlas route for these artifacts, consider atlas-route redesign, but do not silently replace AAL3 with AAL116 without documenting the thesis implication.")
    add("")
    add("## Questions for External Review")
    add("")
    for item in [
        "Is the AAL3 MNI -> native T1 -> B0 route with FLIRT/nearest-neighbour sufficient, or should nonlinear MNI-to-T1 registration be used for atlas labels?",
        "Should 5TT be generated after T1-to-B0 alignment as in the supervisor shell, or generated in native T1 then transformed to B0? What is the safest MRtrix/FSL sequence?",
        "What quantitative atlas-DWI alignment QC thresholds are defensible without GUI inspection: label survival, mask overlap, Dice/Jaccard, endpoint assignment fraction, density, valid zero rows?",
        "What matrix density and zero-row thresholds are acceptable for a 166-node AAL3 structural connectome in ADNI-style DWI?",
        "Are assignment variants such as radial8 or forward80 defensible for AAL3 if anatomy is valid, or do they risk anatomically implausible assignments?",
        "Given the AAL116 benchmark is healthy while AAL3 often fails, is this an atlas-version/label-density issue, a transform issue, or a combination?",
        "For the current completed outputs, should analysis proceed on a strict PASS/WARN subset, a manually reviewed subset, or wait for upstream reprocessing?",
    ]:
        add(f"- {item}")
    add("")
    add("## Key Evidence Paths")
    add("")
    for item in [
        "`/home/ec2-user/exp/structural_connectome_context.md`",
        "`/home/ec2-user/exp/structural_connectome_context.docx`",
        "`/home/ec2-user/exp/scforge/README.md`",
        "`/home/ec2-user/exp/scripts/preprocessing/ADNI_preproc_pipeline_AAL.sh`",
        "`/home/ec2-user/exp/connectome_pipeline/connectome_step7.py`",
        "`/home/ec2-user/exp/connectome_analysis/analysis_sc_matrix_qc.py`",
        "`/home/ec2-user/exp/run_scforge_v1_density_batch.py`",
        "`/home/ec2-user/exp/run_sc_aal3_source_contract_probe.py`",
        "`/home/ec2-user/exp/run_sc_final_spatial_contract_closeout.py`",
        "`/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc/completed_connectomes_after_gap_20260524T085201Z/completed_connectome_qc_report.md`",
        "`/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc/final_spatial_contract_closeout_20260521T114956Z/candidate_patch_decision.md`",
        "`/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc/final_spatial_contract_closeout_20260521T114956Z/aal116_vs_aal3_candidate_comparison.csv`",
        "`/home/ec2-user/exp/archive/sample_matrices/SC_033_S_0908_AAL.csv` and `/home/ec2-user/exp/archive/sample_matrices/SC_033_S_2374_AAL.csv` as supervisor benchmark matrices.",
    ]:
        add(f"- {item}")
    add("")
    return "\n".join(lines) + "\n"


def md_table_to_rows(section: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        if set(line.replace("|", "").strip()) <= {"-", ":"}:
            continue
        rows.append([cell.strip().replace("\\|", "|") for cell in line.strip("|").split("|")])
    return rows


def build_docx(markdown: str, out: Path) -> None:
    doc = SimpleDocx()
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            doc.heading(line[2:].strip(), 1)
        elif line.startswith("## "):
            doc.heading(line[3:].strip(), 2)
        elif line.startswith("### "):
            doc.heading(line[4:].strip(), 3)
        elif line.startswith("- "):
            doc.bullet(line[2:].strip())
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            rows = md_table_to_rows("\n".join(block))
            doc.table(rows)
            continue
        elif line.strip():
            doc.paragraph(line.strip())
        else:
            doc.paragraph("")
        i += 1
    doc.save(out)


def main() -> None:
    stage_rows = read_csv_dict(LATEST_QC / "connectome_stage_counts.csv")
    qc_rows = read_csv_dict(LATEST_QC / "connectome_qc_group_summary.csv")
    canary_rows = read_csv_dict(FINAL_CLOSEOUT / "canary_validation_summary.csv")
    experiment_rows = read_csv_dict(FINAL_CLOSEOUT / "prior_experiment_summary.csv")
    route_rows = read_csv_dict(FINAL_CLOSEOUT / "subject_route_decisions.csv")
    scforge_status, scforge_density_rows, scforge_run_root = latest_scforge_run()
    markdown = make_markdown(
        stage_rows,
        qc_rows,
        canary_rows,
        experiment_rows,
        route_rows,
        scforge_status,
        scforge_density_rows,
        scforge_run_root,
    )
    OUT_MD.write_text(markdown)
    build_docx(markdown, OUT_DOCX)
    print(f"Wrote {OUT_DOCX}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
