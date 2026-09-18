#!/usr/bin/env python3
"""Build an image-led PowerPoint atlas from the live connectome dashboard.

The main deck follows the dashboard's Novel Findings narrative but does not use
statistical significance as an image-inclusion filter.  Every stored dashboard
PNG is included in a sectioned appendix.  Tabs that generate figures only at
runtime are captured through Streamlit's application-test runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "apps" / "connectome_dashboard" / "connectome_app.py"
STATIC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "qc" / "analysis_cohort"
OUTPUT_ROOT = PROJECT_ROOT / "research_audit" / "outputs" / "dashboard_visual_atlas_v1"
DYNAMIC_ROOT = OUTPUT_ROOT / "live_chart_exports"
PPTX_PATH = OUTPUT_ROOT / "structural_connectome_dashboard_visual_atlas_v1.pptx"
INVENTORY_CSV = OUTPUT_ROOT / "dashboard_visual_inventory.csv"
VALIDATION_JSON = OUTPUT_ROOT / "validation.json"
README_PATH = OUTPUT_ROOT / "README.md"

SECTION_NAMES = [
    "Demographics",
    "Novel Findings",
    "Global DTI",
    "Local / ROI DTI",
    "Global Graph",
    "SC Matrix Viewer",
    "Node Metrics",
    "Coupling",
    "Coupling-AAL",
    "Brain Age",
    "LR / SR",
    "EDR Exceptions",
    "ML Diagnostics",
    "Delay",
    "Advanced",
    "Network Analysis",
    "Functional Pending",
]

# These sections contain figures that are not represented by stored PNGs, or
# provide the narrative spine requested for the review version.
CAPTURE_SECTION_INDICES = (0, 1, 5, 8, 11, 12, 15)

STATIC_SECTION_MAP = {
    "02_demographics": "Demographics",
    "03_global_microstructure_live": "Global DTI",
    "04_node_microstructure_live": "Local / ROI DTI",
    "05_multimetric_live": "Multimetric Synthesis",
    "06_global_graph_live": "Global Graph",
    "07_node_graph_live": "Node Metrics",
    "09_coupling_live": "Coupling",
    "10_edgewise_fd_sum": "Edgewise Connectivity",
    "11_brain_age_live": "Brain Age",
    "12_length_delay": "LR / SR",
    "13_delay": "Delay",
    "15_advanced_structural": "Advanced / EDR",
}

STATIC_SECTION_ORDER = tuple(STATIC_SECTION_MAP.values())

SECTION_NOTES = {
    "Demographics": (
        "Cohort composition, available-connectome counts, age, and sex distributions provide the context for every downstream comparison. "
        "Imbalance and demographic structure should remain visible when interpreting later panels."
    ),
    "Novel Findings": (
        "The dashboard synthesis links AAL3 localization, global disconnection, distance-aware organization, and clinical prediction. "
        "This deck presents that synthesis as a working narrative rather than a certified novelty claim."
    ),
    "Global DTI": (
        "FA, MD, AxD, and RD describe complementary microstructural summaries. Their directions should be read jointly; no single scalar is treated as a cellular mechanism."
    ),
    "Local / ROI DTI": (
        "Node-level maps and distributions localize where group differences appear. Multiplicity, site/acquisition sensitivity, and spatial QC still determine inferential weight."
    ),
    "Multimetric Synthesis": (
        "The multimetric view is useful for pattern comparison across correlated measurements, not for counting each metric as an independent discovery."
    ),
    "Global Graph": (
        "Density, efficiency, path length, and strength summarize related aspects of whole-connectome integration and segregation."
    ),
    "SC Matrix Viewer": (
        "Single-subject matrices are descriptive QC views. They show support, weight structure, and density but do not provide individual diagnostic inference."
    ),
    "Node Metrics": (
        "Degree, strength, and nodal efficiency provide complementary views of regional topology; high-ranking nodes should be interpreted alongside microstructure and edge support."
    ),
    "Coupling": (
        "Topology-microstructure coupling links structural graph properties with diffusion measures across nodes. It is structural coupling—not functional connectivity."
    ),
    "Coupling-AAL": (
        "Regional coupling rankings identify where topology and diffusion vary together. They are an integration layer over the underlying node measurements."
    ),
    "Brain Age": (
        "Predicted age, raw brain-age gap, and age-corrected gap should be compared side by side because age correction can materially change the interpretation."
    ),
    "LR / SR": (
        "Short- and long-range panels ask whether the structural phenotype depends on physical pathway range rather than only global magnitude."
    ),
    "EDR Exceptions": (
        "Distance-decay exceptions highlight edges that depart from the cohort distance-weight relationship. They are exploratory spatial signatures."
    ),
    "Edgewise Connectivity": (
        "Edgewise maps show distributed pair-level effects. Direction, support, multiplicity, and reproducibility must be checked in the accompanying source tables."
    ),
    "ML Diagnostics": (
        "Prediction panels report cross-validated performance and error structure. Predictive separation does not by itself establish causality, mechanism, or clinical validity."
    ),
    "Delay": (
        "Delay metrics are length-derived proxies, not measured conduction delays. Their value is in testing the geometry of surviving structural routes."
    ),
    "Advanced / EDR": (
        "Residual, compensation, vulnerability, and disease-gradient constructs are hypothesis-generating dimensions and should not inflate the primary claim."
    ),
    "Network Analysis": (
        "System-level aggregation connects regional and edge results into a broader anatomical story, while retaining the distinction between description and inference."
    ),
}

FALLBACK_TITLES = {
    "Demographics": {
        "vega": [
            "Total subjects by diagnostic group",
            "Available final connectomes by diagnostic group",
            "AAL3 rescue subjects by diagnostic group",
            "Sex distribution — total cohort",
            "Sex distribution — available connectomes",
        ],
        "plotly": ["Age by diagnostic group", "Age by sex"],
    },
    "SC Matrix Viewer": {
        "plotly": [
            "Connectome density by group",
            "Connectome density versus age",
            "Selected subject structural-connectome matrix",
            "Cohort tract-count density distribution",
            "Selected subject tract-count edge distribution",
            "Selected subject connectome graph",
            "Selected subject adjacency heatmap",
        ]
    },
    "Coupling-AAL": {
        "plotly": [
            "Regional topology–microstructure coupling ranking",
            "Regional coupling group comparison",
            "Coupling signals aggregated by network",
        ]
    },
    "EDR Exceptions": {
        "plotly": [
            "EDR exception-rate distribution",
            "EDR lambda distribution",
            "Distance-decay fit",
            "Short-range exception profile",
            "Long-range exception profile",
            "EDR exception edge localization",
        ]
    },
    "ML Diagnostics": {
        "plotly": [
            "Global CDR class distribution",
            "Global CDR classifier performance",
            "Global CDR one-vs-rest ROC",
            "Per-class sensitivity and specificity",
        ]
    },
    "Network Analysis": {
        "plotly": [
            "Network affectedness — CN versus AD effect-size heatmap",
            "Per-network distributions by group",
            "Within/between-network connectivity — CN",
            "Within/between-network connectivity — MCI",
            "Within/between-network connectivity — AD",
            "Within/between-network connectivity — CN minus AD",
        ]
    },
}

COLORS = {
    "navy": "11233F",
    "blue": "245EA8",
    "cyan": "15A4C8",
    "teal": "128C8C",
    "gold": "F2B84B",
    "red": "CE5A67",
    "ink": "17243A",
    "muted": "5F6F85",
    "line": "D8E0EA",
    "paper": "F6F8FC",
    "white": "FFFFFF",
    "pale": "EAF2FA",
}


def rgb(hex_value: str) -> RGBColor:
    return RGBColor.from_string(hex_value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def slugify(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value[:90] or "chart"


def clean_chart_title(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("text", "")
    value = html.unescape(str(value or ""))
    value = re.sub(r"<br\s*/?>", " — ", value, flags=re.I)
    value = re.sub(r"</?sup>", "", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\s+", " ", value).strip(" —")
    return value


def filename_title(path: Path) -> str:
    stem = path.stem
    replacements = {
        "fa_mean": "FA",
        "md_mean": "MD",
        "ad_mean": "AxD",
        "rd_mean": "RD",
        "nodal_eff": "Nodal efficiency",
        "global_eff": "Global efficiency",
        "charpath_len": "Characteristic path length",
        "sr_lr": "SR–LR",
        "lr_sr": "LR/SR",
        "edr": "EDR",
        "bag": "BAG",
        "cnvsmci": "CN vs MCI",
        "cnvsad": "CN vs AD",
        "mcivsad": "MCI vs AD",
        "boxplot": "distribution",
        "manhattan": "AAL3 node map",
        "pvalue": "p-value",
    }
    text = stem.lower()
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1].upper() + text[1:]


def fallback_title(section: str, chart_type: str, index: int) -> str:
    choices = FALLBACK_TITLES.get(section, {}).get(chart_type, [])
    if index < len(choices):
        return choices[index]
    label = "Plot" if chart_type == "plotly" else "Chart"
    return f"{section} — {label} {index + 1}"


def render_plotly(spec: dict[str, Any], destination: Path, section: str) -> None:
    import plotly.graph_objects as go

    figure = go.Figure(spec)
    figure.update_xaxes(automargin=True)
    figure.update_yaxes(automargin=True)
    original_margin = spec.get("layout", {}).get("margin", {}) or {}
    heatmap_present = any(str(trace.get("type", "")).lower() == "heatmap" for trace in spec.get("data", []))
    left_floor = 220 if section == "Network Analysis" and heatmap_present else 90
    bottom_floor = 110 if heatmap_present else 70
    figure.update_layout(
        margin={
            "l": max(left_floor, int(original_margin.get("l", 0) or 0)),
            "r": max(55, int(original_margin.get("r", 0) or 0)),
            "t": max(88, int(original_margin.get("t", 0) or 0)),
            "b": max(bottom_floor, int(original_margin.get("b", 0) or 0)),
        }
    )
    layout_height = spec.get("layout", {}).get("height", 700)
    height = max(700, min(1000, int(layout_height or 700)))
    figure.write_image(str(destination), width=1500, height=height, scale=1.15)


def render_vega(proto: Any, spec: dict[str, Any], destination: Path) -> None:
    import pyarrow as pa
    import vl_convert as vlc

    datasets: dict[str, list[dict[str, Any]]] = {}
    for dataset in proto.datasets:
        table = pa.ipc.open_stream(pa.BufferReader(bytes(dataset.data.data))).read_all().to_pandas()
        table = table.where(table.notna(), None)
        datasets[dataset.name] = table.to_dict(orient="records")
    if datasets:
        spec["datasets"] = datasets
    spec["width"] = 900
    if not spec.get("height"):
        spec["height"] = 420
    spec["autosize"] = {"type": "fit", "contains": "padding"}
    destination.write_bytes(vlc.vegalite_to_png(spec, scale=2))


def capture_section(section_index: int) -> int:
    from streamlit.testing.v1 import AppTest

    section = SECTION_NAMES[section_index]
    destination = DYNAMIC_ROOT / slugify(section)
    destination.mkdir(parents=True, exist_ok=True)
    app = AppTest.from_file(str(APP_PATH), default_timeout=300)
    app.session_state["selected_section_idx"] = section_index
    app.run()

    records: list[dict[str, Any]] = []
    counters = defaultdict(int)
    render_errors: list[str] = []
    for element in app:
        if type(element).__name__ != "UnknownElement":
            continue
        element_type = getattr(element, "type", "")
        if element_type not in {"plotly_chart", "vega_lite_chart"}:
            continue
        chart_type = "plotly" if element_type == "plotly_chart" else "vega"
        chart_index = counters[chart_type]
        counters[chart_type] += 1

        # The first three Vega charts are the same cohort-summary header on
        # every page. Capture them only once under Demographics.
        if section_index != 0 and chart_type == "vega" and chart_index < 3:
            continue

        spec = json.loads(element.proto.spec)
        if chart_type == "plotly":
            raw_title = spec.get("layout", {}).get("title", {}).get("text", "")
        else:
            raw_title = spec.get("title", "")
        title = clean_chart_title(raw_title) or fallback_title(section, chart_type, chart_index)
        file_name = f"{chart_type}_{chart_index + 1:02d}_{slugify(title)}.png"
        image_path = destination / file_name
        try:
            if chart_type == "plotly":
                render_plotly(spec, image_path, section)
            else:
                render_vega(element.proto, spec, image_path)
            records.append(
                {
                    "section": section,
                    "chart_type": chart_type,
                    "chart_index": chart_index + 1,
                    "title": title,
                    "path": str(image_path),
                    "sha256": sha256(image_path),
                }
            )
        except Exception as exc:  # preserve partial capture and report exactly
            render_errors.append(f"{chart_type} {chart_index + 1}: {type(exc).__name__}: {exc}")

    metadata = {
        "section_index": section_index,
        "section": section,
        "captured": len(records),
        "app_exceptions": [str(item) for item in app.exception],
        "render_errors": render_errors,
        "charts": records,
    }
    (destination / "capture_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0 if records or section == "Functional Pending" else 2


def add_text(
    slide: Any,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    size: float = 18,
    color: str = COLORS["ink"],
    bold: bool = False,
    font: str = "Aptos",
    align: PP_ALIGN = PP_ALIGN.LEFT,
    valign: MSO_ANCHOR = MSO_ANCHOR.TOP,
) -> Any:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.margin_left = Inches(0.02)
    frame.margin_right = Inches(0.02)
    frame.margin_top = Inches(0.02)
    frame.margin_bottom = Inches(0.02)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_after = Pt(0)
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return box


def add_rect(slide: Any, x: float, y: float, w: float, h: float, fill: str, line: str | None = None, radius: bool = False) -> Any:
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(line or fill)
    if radius:
        shape.adjustments[0] = 0.08
    return shape


def slide_base(prs: Presentation, title: str, section: str, slide_number: int) -> Any:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background.fill
    background.solid()
    background.fore_color.rgb = rgb(COLORS["paper"])
    add_rect(slide, 0, 0, 13.333, 0.12, COLORS["cyan"])
    add_text(slide, section.upper(), 0.55, 0.26, 3.2, 0.24, 9, COLORS["cyan"], True)
    add_text(slide, title, 0.55, 0.53, 12.1, 0.58, 25, COLORS["navy"], True)
    add_rect(slide, 0.55, 7.15, 12.2, 0.01, COLORS["line"])
    add_text(slide, "Structural Connectome Dashboard · Visual Atlas v1", 0.55, 7.21, 6.8, 0.18, 8, COLORS["muted"])
    add_text(slide, str(slide_number), 12.2, 7.19, 0.55, 0.2, 8, COLORS["muted"], False, align=PP_ALIGN.RIGHT)
    return slide


def add_image_contain(slide: Any, path: Path, x: float, y: float, w: float, h: float) -> Any:
    with Image.open(path) as image:
        image_w, image_h = image.size
    ratio = min(w / image_w, h / image_h)
    draw_w = image_w * ratio
    draw_h = image_h * ratio
    draw_x = x + (w - draw_w) / 2
    draw_y = y + (h - draw_h) / 2
    return slide.shapes.add_picture(str(path), Inches(draw_x), Inches(draw_y), Inches(draw_w), Inches(draw_h))


def add_note_band(slide: Any, note: str, y: float = 6.48, h: float = 0.48) -> None:
    add_rect(slide, 0.55, y, 12.2, h, COLORS["pale"], COLORS["line"], True)
    add_text(slide, note, 0.78, y + 0.11, 11.75, h - 0.14, 10.5, COLORS["muted"])


def add_title_slide(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = rgb(COLORS["navy"])
    add_rect(slide, 0, 0, 13.333, 0.15, COLORS["cyan"])
    add_text(slide, "STRUCTURAL CONNECTOME ANALYSIS", 0.75, 0.8, 7.5, 0.3, 11, COLORS["cyan"], True)
    add_text(slide, "Dashboard Visual Atlas", 0.75, 1.38, 11.7, 0.85, 36, COLORS["white"], True)
    add_text(slide, "Version 1 · CN, MCI and Alzheimer disease", 0.77, 2.34, 10.5, 0.45, 19, "C5D6EB")
    add_rect(slide, 0.75, 3.15, 2.0, 0.08, COLORS["gold"])
    add_text(
        slide,
        "Image-led review deck built from the hosted dashboard.\nAll stored figures are retained irrespective of p-value; interpretation notes are intentionally editable.",
        0.75,
        3.55,
        10.8,
        1.2,
        18,
        COLORS["white"],
    )
    add_text(slide, "Sabeesh Ethiraj · Dipanjan Roy", 0.75, 6.1, 8.0, 0.35, 14, "D7E2F0", True)
    add_text(slide, "School of Artificial Intelligence and Data Science · IIT Jodhpur", 0.75, 6.5, 10.4, 0.3, 11, "AFC4DC")
    add_text(slide, datetime.now(timezone.utc).strftime("Generated %d %B %Y"), 10.45, 6.9, 2.1, 0.25, 9, "AFC4DC", align=PP_ALIGN.RIGHT)


def add_reading_guide(prs: Presentation) -> None:
    slide = slide_base(prs, "How to read this first version", "Orientation", len(prs.slides) + 1)
    items = [
        ("Main deck", "A visual narrative anchored to the dashboard's Novel Findings page."),
        ("Section highlights", "Representative panels from every analytical family, including null and exploratory views."),
        ("Complete appendix", "Every one of the 245 stored dashboard PNGs, grouped by analysis section."),
        ("Interpretation boundary", "Images were not selected by p-value. Notes describe what each family asks; they do not certify novelty or clinical validity."),
    ]
    for idx, (head, body) in enumerate(items):
        row, col = divmod(idx, 2)
        x = 0.7 + col * 6.25
        y = 1.45 + row * 2.25
        add_rect(slide, x, y, 5.8, 1.75, COLORS["white"], COLORS["line"], True)
        add_rect(slide, x, y, 0.12, 1.75, COLORS["cyan"])
        add_text(slide, head, x + 0.32, y + 0.24, 5.1, 0.34, 17, COLORS["navy"], True)
        add_text(slide, body, x + 0.32, y + 0.7, 5.05, 0.78, 12, COLORS["muted"])
    add_note_band(slide, "Use the main deck to edit the story. Use the appendix as the image bank and provenance index.", 6.45, 0.48)


def add_dashboard_map(prs: Presentation) -> None:
    slide = slide_base(prs, "Dashboard sections represented in the atlas", "Orientation", len(prs.slides) + 1)
    groups = [
        ("Cohort", "Demographics · QC · availability"),
        ("Microstructure", "Global DTI · local/ROI DTI · multimetric"),
        ("Topology", "Global graph · node metrics · matrix viewer"),
        ("Integration", "Coupling · Coupling-AAL · network analysis"),
        ("Geometry", "LR/SR · delay · EDR exceptions · advanced"),
        ("Prediction", "Brain age · ML diagnostics"),
    ]
    for idx, (head, body) in enumerate(groups):
        row, col = divmod(idx, 3)
        x = 0.65 + col * 4.2
        y = 1.38 + row * 2.18
        color = [COLORS["blue"], COLORS["cyan"], COLORS["teal"], COLORS["gold"], COLORS["red"], COLORS["blue"]][idx]
        add_rect(slide, x, y, 3.8, 1.7, COLORS["white"], COLORS["line"], True)
        add_rect(slide, x, y, 3.8, 0.11, color)
        add_text(slide, head, x + 0.23, y + 0.27, 3.3, 0.34, 17, COLORS["navy"], True)
        add_text(slide, body, x + 0.23, y + 0.78, 3.3, 0.62, 11, COLORS["muted"])
    add_note_band(slide, "Functional Pending contains no scientific image and is therefore documented rather than illustrated.", 6.33, 0.48)


def add_narrative_spine(prs: Presentation) -> None:
    slide = slide_base(prs, "Working narrative from the Novel Findings page", "Narrative spine", len(prs.slides) + 1)
    stages = [
        ("1", "AAL3 localization", "Regional microstructure and node topology"),
        ("2", "Global disconnection", "DTI burden, density, efficiency and strength"),
        ("3", "Distance-aware organization", "LR/SR, delay proxies, EDR and gradient edges"),
        ("4", "Clinical prediction", "Brain age and cross-validated severity models"),
    ]
    for idx, (num, head, body) in enumerate(stages):
        x = 0.55 + idx * 3.12
        add_rect(slide, x, 2.05, 2.72, 2.35, COLORS["white"], COLORS["line"], True)
        add_rect(slide, x + 0.18, 2.28, 0.56, 0.56, COLORS["cyan"], COLORS["cyan"], True)
        add_text(slide, num, x + 0.18, 2.37, 0.56, 0.3, 16, COLORS["white"], True, align=PP_ALIGN.CENTER)
        add_text(slide, head, x + 0.18, 3.02, 2.35, 0.48, 16, COLORS["navy"], True)
        add_text(slide, body, x + 0.18, 3.55, 2.34, 0.62, 11, COLORS["muted"])
        if idx < 3:
            add_text(slide, "→", x + 2.76, 2.94, 0.34, 0.42, 25, COLORS["gold"], True, align=PP_ALIGN.CENTER)
    add_note_band(slide, "This is a presentation scaffold—not a claim that every displayed panel is positive, independent, novel, or mechanistic.", 5.55, 0.55)


def add_section_divider(prs: Presentation, section: str, count_text: str, note: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = rgb(COLORS["navy"])
    add_rect(slide, 0, 0, 0.18, 7.5, COLORS["cyan"])
    add_text(slide, "SECTION", 0.85, 1.08, 2.0, 0.28, 11, COLORS["cyan"], True)
    add_text(slide, section, 0.85, 1.58, 11.5, 0.85, 34, COLORS["white"], True)
    add_text(slide, count_text, 0.88, 2.62, 8.0, 0.38, 15, "BCD0E5")
    add_rect(slide, 0.87, 3.35, 1.8, 0.08, COLORS["gold"])
    add_text(slide, note, 0.87, 3.78, 10.8, 1.45, 18, COLORS["white"])
    add_text(slide, str(len(prs.slides)), 12.15, 7.05, 0.55, 0.2, 8, "AFC4DC", align=PP_ALIGN.RIGHT)


def is_complex_image(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in ("manhattan", "heatmap", "brain", "gradient", "matrix", "roc", "performance", "scatter", "multimetric"))


def add_visual_slide(
    prs: Presentation,
    section: str,
    items: list[dict[str, Any]],
    role: str,
    inventory_rows: list[dict[str, Any]],
    note: str | None = None,
) -> int:
    if not items:
        return len(prs.slides)
    title = items[0]["title"] if len(items) == 1 else f"{section} — visual panel"
    if len(title) > 96:
        title = title.split(" — ", 1)[0]
    if len(title) > 96:
        title = title[:93].rstrip() + "..."
    slide = slide_base(prs, title, section, len(prs.slides) + 1)
    if len(items) == 1:
        item = items[0]
        add_rect(slide, 0.58, 1.23, 12.15, 5.08, COLORS["white"], COLORS["line"], True)
        add_image_contain(slide, Path(item["path"]), 0.75, 1.38, 11.8, 4.72)
    elif len(items) == 2:
        for idx, item in enumerate(items):
            x = 0.58 + idx * 6.12
            add_rect(slide, x, 1.24, 5.92, 4.92, COLORS["white"], COLORS["line"], True)
            add_image_contain(slide, Path(item["path"]), x + 0.12, 1.36, 5.68, 4.32)
            add_text(slide, item["title"], x + 0.15, 5.72, 5.62, 0.3, 10, COLORS["muted"], True, align=PP_ALIGN.CENTER)
    else:
        positions = [(0.58, 1.26), (6.72, 1.26), (0.58, 3.78), (6.72, 3.78)]
        for idx, item in enumerate(items[:4]):
            x, y = positions[idx]
            add_rect(slide, x, y, 6.0, 2.31, COLORS["white"], COLORS["line"], True)
            add_image_contain(slide, Path(item["path"]), x + 0.10, y + 0.08, 5.8, 1.86)
            add_text(slide, item["title"], x + 0.14, y + 1.98, 5.72, 0.22, 8.5, COLORS["muted"], True, align=PP_ALIGN.CENTER)
    add_note_band(slide, note or SECTION_NOTES.get(section, "Displayed for visual review; consult the source tables for exact statistical status."), 6.47, 0.48)
    slide_number = len(prs.slides)
    for item in items:
        inventory_rows.append(
            {
                "source_type": item["source_type"],
                "section": section,
                "title": item["title"],
                "source_path": item["path"],
                "sha256": item["sha256"],
                "deck_role": role,
                "slide_number": slide_number,
            }
        )
    return slide_number


def chunked(items: list[Any], n: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), n):
        yield items[index : index + n]


def load_dynamic_items() -> dict[str, list[dict[str, Any]]]:
    sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for metadata_path in sorted(DYNAMIC_ROOT.glob("*/capture_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for chart in metadata.get("charts", []):
            image_path = Path(chart["path"])
            if not image_path.exists():
                continue
            digest = chart.get("sha256") or sha256(image_path)
            if digest in seen:
                continue
            seen.add(digest)
            sections[chart["section"]].append(
                {
                    "source_type": "dashboard_live_export",
                    "section": chart["section"],
                    "title": chart["title"],
                    "path": str(image_path),
                    "sha256": digest,
                    "chart_type": chart["chart_type"],
                }
            )
    return sections


def load_static_items() -> dict[str, list[dict[str, Any]]]:
    sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image_path in sorted(STATIC_ROOT.glob("*/*.png")):
        folder = image_path.parent.name
        section = STATIC_SECTION_MAP.get(folder)
        if section is None:
            continue
        sections[section].append(
            {
                "source_type": "stored_dashboard_png",
                "section": section,
                "title": filename_title(image_path),
                "path": str(image_path),
                "sha256": sha256(image_path),
            }
        )
    return sections


def representative_items(items: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    if len(items) <= limit:
        return items
    priority_tokens = ("manhattan", "heatmap", "gradient", "multimetric", "pvalue", "bag", "efficiency", "density")
    ranked = sorted(
        items,
        key=lambda item: (
            0 if any(token in Path(item["path"]).name.lower() for token in priority_tokens) else 1,
            Path(item["path"]).name,
        ),
    )
    selected = ranked[:limit]
    return sorted(selected, key=lambda item: Path(item["path"]).name)


def build_presentation() -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    dynamic = load_dynamic_items()
    static = load_static_items()
    inventory_rows: list[dict[str, Any]] = []
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    add_title_slide(prs)
    add_reading_guide(prs)
    add_dashboard_map(prs)
    add_narrative_spine(prs)

    # Main narrative: all live charts from Novel Findings, followed by the
    # live-only views that are not covered by the stored PNG archive.
    main_dynamic_order = (
        "Demographics",
        "Novel Findings",
        "SC Matrix Viewer",
        "Coupling-AAL",
        "EDR Exceptions",
        "ML Diagnostics",
        "Network Analysis",
    )
    for section in main_dynamic_order:
        items = dynamic.get(section, [])
        if not items:
            continue
        add_section_divider(prs, section, f"{len(items)} live dashboard visual(s)", SECTION_NOTES.get(section, ""))
        for item_group in chunked(items, 1 if section == "Novel Findings" else 2):
            add_visual_slide(prs, section, item_group, "main_live", inventory_rows)

    # Main-deck highlights from every stored-output family.
    add_section_divider(
        prs,
        "Section highlights",
        "Representative stored figures from every analytical family",
        "These overview panels make the deck editable and navigable. The complete, unfiltered image bank follows in the appendix.",
    )
    for section in STATIC_SECTION_ORDER:
        items = representative_items(static.get(section, []), limit=6)
        if not items:
            continue
        for item_group in chunked(items, 4):
            add_visual_slide(prs, section, item_group, "main_highlight", inventory_rows)

    # Complete appendix: every stored PNG exactly once, grouped by section.
    add_section_divider(
        prs,
        "Complete image appendix",
        "All 245 stored dashboard PNGs · no p-value inclusion filter",
        "The appendix is the full visual archive. File-derived titles and source paths are retained in the accompanying CSV inventory.",
    )
    appendix_counts: dict[str, int] = {}
    for section in STATIC_SECTION_ORDER:
        items = static.get(section, [])
        if not items:
            continue
        appendix_counts[section] = len(items)
        add_section_divider(prs, section, f"Appendix · {len(items)} stored image(s)", SECTION_NOTES.get(section, ""))
        cursor = 0
        while cursor < len(items):
            remaining = items[cursor:]
            if is_complex_image(Path(remaining[0]["path"])):
                group = remaining[:1]
            elif len(remaining) >= 2 and is_complex_image(Path(remaining[1]["path"])):
                group = remaining[:1]
            elif all("_node_" in Path(item["path"]).name.lower() and "manhattan" not in Path(item["path"]).name.lower() for item in remaining[:4]):
                group = remaining[:4]
            else:
                group = remaining[:2]
            add_visual_slide(prs, section, group, "appendix_complete", inventory_rows)
            cursor += len(group)

    prs.save(PPTX_PATH)
    summary = {
        "static_count": sum(len(items) for items in static.values()),
        "dynamic_count": sum(len(items) for items in dynamic.values()),
        "appendix_counts": appendix_counts,
        "slides": len(prs.slides),
    }
    return len(prs.slides), inventory_rows, summary


def write_inventory(rows: list[dict[str, Any]]) -> None:
    fields = ["source_type", "section", "title", "source_path", "sha256", "deck_role", "slide_number"]
    with INVENTORY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_package(summary: dict[str, Any], inventory_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from pptx import Presentation as ReadPresentation

    reopened = ReadPresentation(PPTX_PATH)
    slide_w, slide_h = reopened.slide_width, reopened.slide_height
    out_of_bounds: list[dict[str, Any]] = []
    picture_count = 0
    for slide_number, slide in enumerate(reopened.slides, start=1):
        for shape in slide.shapes:
            if int(shape.shape_type) == 13:  # MSO_SHAPE_TYPE.PICTURE
                picture_count += 1
            if (
                shape.left < 0
                or shape.top < 0
                or shape.left + shape.width > slide_w + 100
                or shape.top + shape.height > slide_h + 100
            ):
                out_of_bounds.append({"slide": slide_number, "shape": shape.name})
    stored_paths = {str(path) for path in STATIC_ROOT.glob("*/*.png") if path.parent.name in STATIC_SECTION_MAP}
    appendix_rows = [row for row in inventory_rows if row["deck_role"] == "appendix_complete"]
    appendix_paths = {row["source_path"] for row in appendix_rows}
    all_images_load = True
    corrupt_images: list[str] = []
    for source_path in sorted(stored_paths):
        try:
            with Image.open(source_path) as image:
                image.verify()
        except Exception as exc:
            all_images_load = False
            corrupt_images.append(f"{source_path}: {exc}")
    checks = {
        "pptx_exists": PPTX_PATH.exists(),
        "pptx_reopens": len(reopened.slides) == summary["slides"],
        "all_245_stored_pngs_discovered": len(stored_paths) == 245,
        "all_stored_pngs_in_appendix": appendix_paths == stored_paths,
        "no_duplicate_static_appendix_rows": len(appendix_rows) == len(appendix_paths),
        "all_stored_images_decode": all_images_load,
        "dynamic_live_exports_present": summary["dynamic_count"] > 0,
        "all_dynamic_exports_placed_in_main_deck": len([row for row in inventory_rows if row["deck_role"] == "main_live"]) == summary["dynamic_count"],
        "all_shapes_within_slide_bounds": not out_of_bounds,
        "presentation_contains_images": picture_count > 0,
        "inventory_exists": INVENTORY_CSV.exists(),
    }
    validation = {
        "record_type": "dashboard_visual_atlas_v1_validation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "summary": summary,
        "pptx": str(PPTX_PATH),
        "pptx_sha256": sha256(PPTX_PATH),
        "inventory": str(INVENTORY_CSV),
        "inventory_sha256": sha256(INVENTORY_CSV),
        "corrupt_images": corrupt_images,
        "out_of_bounds_shapes": out_of_bounds,
        "picture_count": picture_count,
    }
    VALIDATION_JSON.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return validation


def write_readme(validation: dict[str, Any]) -> None:
    summary = validation["summary"]
    README_PATH.write_text(
        "\n".join(
            [
                "# Structural Connectome Dashboard Visual Atlas — V1",
                "",
                "Image-led review deck built from the hosted Streamlit dashboard and its generated PNG archive.",
                "",
                f"- PowerPoint: `{PPTX_PATH.name}`",
                f"- Slides: {summary['slides']}",
                f"- Stored dashboard PNGs in complete appendix: {summary['static_count']}",
                f"- Unique live-only/dashboard narrative exports: {summary['dynamic_count']}",
                f"- Validation: **{validation['status']}**",
                "- Image inclusion did not use p-value or significance status.",
                "- Interpretive notes are working prompts, not certified novelty or clinical claims.",
                "",
                "The CSV inventory maps every displayed source image to its slide number and SHA-256 digest.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_all() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if DYNAMIC_ROOT.exists():
        shutil.rmtree(DYNAMIC_ROOT)
    DYNAMIC_ROOT.mkdir(parents=True, exist_ok=True)

    capture_results: list[dict[str, Any]] = []
    for section_index in CAPTURE_SECTION_INDICES:
        command = [sys.executable, str(Path(__file__).resolve()), "--capture-section", str(section_index)]
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=420)
        metadata_path = DYNAMIC_ROOT / slugify(SECTION_NAMES[section_index]) / "capture_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        capture_results.append(
            {
                "section": SECTION_NAMES[section_index],
                "returncode": completed.returncode,
                "captured": metadata.get("captured", 0),
                "render_errors": metadata.get("render_errors", []),
                "app_exceptions": metadata.get("app_exceptions", []),
                "stderr_tail": completed.stderr[-2000:],
            }
        )

    slides, rows, summary = build_presentation()
    summary["capture_results"] = capture_results
    write_inventory(rows)
    validation = validate_package(summary, rows)
    write_readme(validation)
    print(json.dumps({"pptx": str(PPTX_PATH), "slides": slides, "status": validation["status"], "summary": summary}, indent=2))
    return 0 if validation["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-section", type=int)
    args = parser.parse_args()
    if args.capture_section is not None:
        if args.capture_section < 0 or args.capture_section >= len(SECTION_NAMES):
            raise SystemExit("invalid section index")
        return capture_section(args.capture_section)
    return build_all()


if __name__ == "__main__":
    raise SystemExit(main())
