#!/usr/bin/env node
/*
 * Independent Office-compatible rebuild of the structural-connectome visual
 * atlas.  This intentionally does not reuse the python-pptx package that made
 * the first file.  It produces one full deck plus smaller core/appendix
 * volumes so PowerPoint never has to repair a large monolithic package.
 */

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const PptxGenJS = require("/tmp/pptxgenjs-runtime/node_modules/pptxgenjs");

const ROOT = "/home/ec2-user/exp";
const STATIC_ROOT = path.join(ROOT, "data/derivatives/qc/analysis_cohort");
const SOURCE_ROOT = path.join(ROOT, "research_audit/outputs/dashboard_visual_atlas_v1");
const LIVE_ROOT = path.join(SOURCE_ROOT, "live_chart_exports");
const OUT = path.join(ROOT, "research_audit/outputs/dashboard_visual_atlas_v1_compatible");

const STATIC_SECTION_MAP = {
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
};

const STATIC_SECTION_ORDER = [
  "Demographics", "Global DTI", "Local / ROI DTI", "Multimetric Synthesis",
  "Global Graph", "Node Metrics", "Coupling", "Edgewise Connectivity",
  "Brain Age", "LR / SR", "Delay", "Advanced / EDR",
];

const DYNAMIC_SECTION_ORDER = [
  "Demographics", "Novel Findings", "SC Matrix Viewer", "Coupling-AAL",
  "EDR Exceptions", "ML Diagnostics", "Network Analysis",
];

const APPENDIX_A_SECTIONS = [
  "Demographics", "Global DTI", "Local / ROI DTI", "Multimetric Synthesis", "Global Graph",
];
const APPENDIX_B_SECTIONS = [
  "Node Metrics", "Coupling", "Edgewise Connectivity", "Brain Age", "LR / SR", "Delay", "Advanced / EDR",
];

const NOTES = {
  "Demographics": "Cohort composition, available-connectome counts, age and sex distributions provide the context for every downstream comparison.",
  "Novel Findings": "Working synthesis: AAL3 localization → global disconnection → distance-aware organization → clinical prediction. This is a narrative scaffold, not a certified novelty claim.",
  "Global DTI": "FA, MD, AxD and RD are complementary microstructural summaries. Read their directions jointly; no scalar is treated as a cellular mechanism.",
  "Local / ROI DTI": "Node-level panels localize group differences. Multiplicity, acquisition/site sensitivity and spatial QC determine inferential weight.",
  "Multimetric Synthesis": "Use this panel to compare correlated measurements, not to count each metric as an independent discovery.",
  "Global Graph": "Density, efficiency, path length and strength summarize related aspects of whole-connectome integration and segregation.",
  "SC Matrix Viewer": "Single-subject matrices are descriptive QC views and do not provide individual diagnostic inference.",
  "Node Metrics": "Degree, strength and nodal efficiency provide complementary views of regional topology and should be interpreted with edge support.",
  "Coupling": "Topology–microstructure coupling links structural graph properties with diffusion measures. It is structural coupling, not functional connectivity.",
  "Coupling-AAL": "Regional coupling rankings are an integration layer over the underlying node measurements.",
  "Brain Age": "Compare predicted age, raw BAG and age-corrected BAG side by side because age correction changes interpretation.",
  "LR / SR": "Short- and long-range panels test whether the structural phenotype depends on physical pathway range.",
  "EDR Exceptions": "Distance-decay exceptions are exploratory edges that depart from the cohort distance–weight relationship.",
  "Edgewise Connectivity": "Edge maps are included for completeness. Direction, support, multiplicity and reproducibility remain in the source tables.",
  "ML Diagnostics": "Cross-validated prediction does not by itself establish causality, mechanism or clinical validity.",
  "Delay": "Delay measures are length-derived proxies, not measured conduction delays.",
  "Advanced / EDR": "Residual, compensation, vulnerability and gradient constructs are hypothesis-generating dimensions.",
  "Network Analysis": "System aggregation connects regional and edge results while retaining the distinction between description and inference.",
};

const C = {
  navy: "11233F", blue: "245EA8", cyan: "15A4C8", teal: "128C8C",
  gold: "F2B84B", red: "CE5A67", ink: "17243A", muted: "5F6F85",
  line: "D8E0EA", paper: "F6F8FC", white: "FFFFFF", pale: "EAF2FA",
};

function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); }
function sha256(p) { return crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex"); }

function cleanTitle(text) {
  return String(text || "")
    .replace(/<br\s*\/?>/gi, " — ")
    .replace(/<\/?sup>/gi, "")
    .replace(/<[^>]+>/g, "")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .trim();
}

function fileTitle(filePath) {
  let text = path.basename(filePath, path.extname(filePath)).toLowerCase();
  const replacements = [
    ["fa_mean", "FA"], ["md_mean", "MD"], ["ad_mean", "AxD"], ["rd_mean", "RD"],
    ["nodal_eff", "Nodal efficiency"], ["global_eff", "Global efficiency"],
    ["charpath_len", "Characteristic path length"], ["sr_lr", "SR–LR"], ["lr_sr", "LR/SR"],
    ["edr", "EDR"], ["bag", "BAG"], ["cnvsmci", "CN vs MCI"],
    ["cnvsad", "CN vs AD"], ["mcivsad", "MCI vs AD"],
    ["boxplot", "distribution"], ["manhattan", "AAL3 node map"], ["pvalue", "p-value"],
  ];
  for (const [a, b] of replacements) text = text.split(a).join(b);
  text = text.replace(/_/g, " ").replace(/\s+/g, " ").trim();
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function imageDimensions(p) {
  const b = fs.readFileSync(p);
  if (b.length < 24 || b.toString("ascii", 1, 4) !== "PNG") throw new Error(`Not a PNG: ${p}`);
  return { width: b.readUInt32BE(16), height: b.readUInt32BE(20) };
}

function imageContain(slide, imagePath, x, y, w, h) {
  const dim = imageDimensions(imagePath);
  const scale = Math.min(w / dim.width, h / dim.height);
  const dw = dim.width * scale;
  const dh = dim.height * scale;
  slide.addImage({ path: imagePath, x: x + (w - dw) / 2, y: y + (h - dh) / 2, w: dw, h: dh });
}

function addText(slide, text, x, y, w, h, size = 18, color = C.ink, bold = false, align = "left", valign = "top") {
  slide.addText(String(text), {
    x, y, w, h, fontFace: "Aptos", fontSize: size, color, bold, align, valign,
    margin: 0, breakLine: false, fit: "shrink", paraSpaceAfterPt: 0,
  });
}

function addRect(slide, x, y, w, h, fill, line = fill, radius = false) {
  slide.addShape(radius ? "roundRect" : "rect", {
    x, y, w, h, fill: { color: fill }, line: { color: line, width: line === fill ? 0.25 : 0.8 }, radius: 0.08,
  });
}

function newPresentation() {
  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "Sabeesh Ethiraj and Dipanjan Roy";
  pptx.company = "Indian Institute of Technology Jodhpur";
  pptx.subject = "Structural connectome dashboard visual atlas";
  pptx.title = "Structural Connectome Dashboard Visual Atlas V1";
  pptx.lang = "en-US";
  pptx.theme = {
    headFontFace: "Aptos Display", bodyFontFace: "Aptos", lang: "en-US",
  };
  pptx.defineSlideMaster({
    title: "BLANK_SAFE",
    background: { color: C.paper },
    objects: [],
    slideNumber: { x: 12.2, y: 7.18, w: 0.55, h: 0.2, color: C.muted, fontSize: 8, align: "right" },
  });
  return pptx;
}

function baseSlide(pptx, title, section) {
  const slide = pptx.addSlide("BLANK_SAFE");
  slide.background = { color: C.paper };
  addRect(slide, 0, 0, 13.333, 0.12, C.cyan);
  addText(slide, section.toUpperCase(), 0.55, 0.26, 3.6, 0.24, 9, C.cyan, true);
  let display = cleanTitle(title);
  if (display.length > 96) display = display.split(" — ")[0];
  if (display.length > 96) display = display.slice(0, 93).trim() + "...";
  addText(slide, display, 0.55, 0.53, 12.1, 0.58, 25, C.navy, true);
  addRect(slide, 0.55, 7.15, 12.2, 0.01, C.line);
  addText(slide, "Structural Connectome Dashboard · Visual Atlas v1", 0.55, 7.21, 6.8, 0.18, 8, C.muted);
  return slide;
}

function noteBand(slide, note) {
  addRect(slide, 0.55, 6.47, 12.2, 0.48, C.pale, C.line, true);
  addText(slide, note, 0.78, 6.58, 11.75, 0.24, 10.5, C.muted);
}

function renderTitle(pptx, descriptor) {
  const slide = pptx.addSlide("BLANK_SAFE");
  slide.background = { color: C.navy };
  addRect(slide, 0, 0, 13.333, 0.15, C.cyan);
  addText(slide, "STRUCTURAL CONNECTOME ANALYSIS", 0.75, 0.8, 7.5, 0.3, 11, C.cyan, true);
  addText(slide, descriptor.title || "Dashboard Visual Atlas", 0.75, 1.38, 11.7, 0.85, 36, C.white, true);
  addText(slide, descriptor.subtitle || "Version 1 · CN, MCI and Alzheimer disease", 0.77, 2.34, 10.5, 0.45, 19, "C5D6EB");
  addRect(slide, 0.75, 3.15, 2.0, 0.08, C.gold);
  addText(slide, descriptor.body || "Image-led review deck built from the hosted dashboard.\nAll figures are retained irrespective of p-value; interpretation notes are editable.", 0.75, 3.55, 10.8, 1.2, 18, C.white);
  addText(slide, "Sabeesh Ethiraj · Dipanjan Roy", 0.75, 6.1, 8.0, 0.35, 14, "D7E2F0", true);
  addText(slide, "School of Artificial Intelligence and Data Science · IIT Jodhpur", 0.75, 6.5, 10.4, 0.3, 11, "AFC4DC");
}

function renderGuide(pptx) {
  const slide = baseSlide(pptx, "How to read this first version", "Orientation");
  const items = [
    ["Main deck", "Visual narrative anchored to the dashboard's Novel Findings page."],
    ["Section highlights", "Representative panels from every analytical family, including null and exploratory views."],
    ["Complete appendix", "Every stored dashboard PNG, grouped by analysis section."],
    ["Interpretation boundary", "No p-value inclusion filter. Notes describe the analytical question, not certified novelty or clinical validity."],
  ];
  items.forEach(([head, body], idx) => {
    const row = Math.floor(idx / 2), col = idx % 2;
    const x = 0.7 + col * 6.25, y = 1.45 + row * 2.25;
    addRect(slide, x, y, 5.8, 1.75, C.white, C.line, true);
    addRect(slide, x, y, 0.12, 1.75, C.cyan);
    addText(slide, head, x + 0.32, y + 0.24, 5.1, 0.34, 17, C.navy, true);
    addText(slide, body, x + 0.32, y + 0.7, 5.05, 0.78, 12, C.muted);
  });
  noteBand(slide, "Use the main deck to edit the story. Use the appendix as the complete image bank.");
}

function renderMap(pptx) {
  const slide = baseSlide(pptx, "Dashboard sections represented in the atlas", "Orientation");
  const groups = [
    ["Cohort", "Demographics · QC · availability"], ["Microstructure", "Global DTI · local/ROI DTI · multimetric"],
    ["Topology", "Global graph · node metrics · matrix viewer"], ["Integration", "Coupling · Coupling-AAL · network analysis"],
    ["Geometry", "LR/SR · delay · EDR exceptions · advanced"], ["Prediction", "Brain age · ML diagnostics"],
  ];
  groups.forEach(([head, body], idx) => {
    const row = Math.floor(idx / 3), col = idx % 3;
    const x = 0.65 + col * 4.2, y = 1.38 + row * 2.18;
    addRect(slide, x, y, 3.8, 1.7, C.white, C.line, true);
    addRect(slide, x, y, 3.8, 0.11, [C.blue, C.cyan, C.teal, C.gold, C.red, C.blue][idx]);
    addText(slide, head, x + 0.23, y + 0.27, 3.3, 0.34, 17, C.navy, true);
    addText(slide, body, x + 0.23, y + 0.78, 3.3, 0.62, 11, C.muted);
  });
  noteBand(slide, "Functional Pending contains no scientific image and is documented rather than illustrated.");
}

function renderNarrative(pptx) {
  const slide = baseSlide(pptx, "Working narrative from the Novel Findings page", "Narrative spine");
  const stages = [
    ["1", "AAL3 localization", "Regional microstructure and node topology"],
    ["2", "Global disconnection", "DTI burden, density, efficiency and strength"],
    ["3", "Distance-aware organization", "LR/SR, delay proxies, EDR and gradient edges"],
    ["4", "Clinical prediction", "Brain age and cross-validated severity models"],
  ];
  stages.forEach(([num, head, body], idx) => {
    const x = 0.55 + idx * 3.12;
    addRect(slide, x, 2.05, 2.72, 2.35, C.white, C.line, true);
    addRect(slide, x + 0.18, 2.28, 0.56, 0.56, C.cyan, C.cyan, true);
    addText(slide, num, x + 0.18, 2.38, 0.56, 0.3, 16, C.white, true, "center");
    addText(slide, head, x + 0.18, 3.02, 2.35, 0.48, 16, C.navy, true);
    addText(slide, body, x + 0.18, 3.55, 2.34, 0.62, 11, C.muted);
    if (idx < 3) addText(slide, "→", x + 2.76, 2.94, 0.34, 0.42, 25, C.gold, true, "center");
  });
  noteBand(slide, "Presentation scaffold only: displayed panels are not automatically positive, independent, novel or mechanistic.");
}

function renderDivider(pptx, d) {
  const slide = pptx.addSlide("BLANK_SAFE");
  slide.background = { color: C.navy };
  addRect(slide, 0, 0, 0.18, 7.5, C.cyan);
  addText(slide, "SECTION", 0.85, 1.08, 2.0, 0.28, 11, C.cyan, true);
  addText(slide, d.section, 0.85, 1.58, 11.5, 0.85, 34, C.white, true);
  addText(slide, d.countText || "", 0.88, 2.62, 8.0, 0.38, 15, "BCD0E5");
  addRect(slide, 0.87, 3.35, 1.8, 0.08, C.gold);
  addText(slide, d.note || NOTES[d.section] || "", 0.87, 3.78, 10.8, 1.45, 18, C.white);
}

function renderVisual(pptx, d, deckName, inventory) {
  const items = d.items;
  const title = items.length === 1 ? items[0].title : `${d.section} — visual panel`;
  const slide = baseSlide(pptx, title, d.section);
  if (items.length === 1) {
    addRect(slide, 0.58, 1.23, 12.15, 5.08, C.white, C.line, true);
    imageContain(slide, items[0].path, 0.75, 1.38, 11.8, 4.72);
  } else if (items.length === 2) {
    items.forEach((item, idx) => {
      const x = 0.58 + idx * 6.12;
      addRect(slide, x, 1.24, 5.92, 4.92, C.white, C.line, true);
      imageContain(slide, item.path, x + 0.12, 1.36, 5.68, 4.32);
      addText(slide, item.title, x + 0.15, 5.72, 5.62, 0.3, 10, C.muted, true, "center");
    });
  } else {
    const pos = [[0.58, 1.26], [6.72, 1.26], [0.58, 3.78], [6.72, 3.78]];
    items.slice(0, 4).forEach((item, idx) => {
      const [x, y] = pos[idx];
      addRect(slide, x, y, 6.0, 2.31, C.white, C.line, true);
      imageContain(slide, item.path, x + 0.10, y + 0.08, 5.8, 1.86);
      addText(slide, item.title, x + 0.14, y + 1.98, 5.72, 0.22, 8.5, C.muted, true, "center");
    });
  }
  noteBand(slide, d.note || NOTES[d.section] || "Displayed for visual review; consult source tables for exact statistical status.");
  const slideNumber = pptx._slides.length;
  items.forEach(item => inventory.push({
    deck: deckName, slide_number: slideNumber, role: d.role, source_type: item.sourceType,
    section: d.section, title: item.title, source_path: item.path, sha256: item.sha256,
  }));
}

function loadDynamic() {
  const out = {};
  if (!fs.existsSync(LIVE_ROOT)) return out;
  for (const dir of fs.readdirSync(LIVE_ROOT).sort()) {
    const metaPath = path.join(LIVE_ROOT, dir, "capture_metadata.json");
    if (!fs.existsSync(metaPath)) continue;
    const meta = JSON.parse(fs.readFileSync(metaPath, "utf8"));
    out[meta.section] = (meta.charts || []).filter(c => fs.existsSync(c.path)).map(c => ({
      sourceType: "dashboard_live_export", section: meta.section, title: cleanTitle(c.title),
      path: c.path, sha256: c.sha256 || sha256(c.path),
    }));
  }
  return out;
}

function loadStatic() {
  const out = {};
  for (const [folder, section] of Object.entries(STATIC_SECTION_MAP)) {
    const dir = path.join(STATIC_ROOT, folder);
    if (!fs.existsSync(dir)) continue;
    out[section] = fs.readdirSync(dir).filter(n => n.toLowerCase().endsWith(".png")).sort().map(n => {
      const p = path.join(dir, n);
      return { sourceType: "stored_dashboard_png", section, title: fileTitle(p), path: p, sha256: sha256(p) };
    });
  }
  return out;
}

function complex(item) {
  const n = path.basename(item.path).toLowerCase();
  return ["manhattan", "heatmap", "brain", "gradient", "matrix", "roc", "performance", "scatter", "multimetric"].some(t => n.includes(t));
}

function chunks(items, n) {
  const out = [];
  for (let i = 0; i < items.length; i += n) out.push(items.slice(i, i + n));
  return out;
}

function representative(items, limit = 6) {
  if (items.length <= limit) return items;
  const tokens = ["manhattan", "heatmap", "gradient", "multimetric", "pvalue", "bag", "efficiency", "density"];
  return [...items].sort((a, b) => {
    const ap = tokens.some(t => path.basename(a.path).toLowerCase().includes(t)) ? 0 : 1;
    const bp = tokens.some(t => path.basename(b.path).toLowerCase().includes(t)) ? 0 : 1;
    return ap - bp || a.path.localeCompare(b.path);
  }).slice(0, limit).sort((a, b) => a.path.localeCompare(b.path));
}

function appendixVisualDescriptors(section, items, role = "appendix_complete") {
  const out = [];
  for (let i = 0; i < items.length;) {
    const remaining = items.slice(i);
    let group;
    if (complex(remaining[0]) || (remaining[1] && complex(remaining[1]))) group = remaining.slice(0, 1);
    else if (remaining.slice(0, 4).every(it => path.basename(it.path).toLowerCase().includes("_node_") && !path.basename(it.path).toLowerCase().includes("manhattan"))) group = remaining.slice(0, 4);
    else group = remaining.slice(0, 2);
    out.push({ type: "visual", section, items: group, role });
    i += group.length;
  }
  return out;
}

function coreDescriptors(dynamic, statics) {
  const out = [
    { type: "title", title: "Dashboard Visual Atlas", subtitle: "Office-compatible Version 1 · CN, MCI and Alzheimer disease" },
    { type: "guide" }, { type: "map" }, { type: "narrative" },
  ];
  for (const section of DYNAMIC_SECTION_ORDER) {
    const items = dynamic[section] || [];
    if (!items.length) continue;
    out.push({ type: "divider", section, countText: `${items.length} live dashboard visual(s)`, note: NOTES[section] });
    const n = section === "Novel Findings" ? 1 : 2;
    chunks(items, n).forEach(group => out.push({ type: "visual", section, items: group, role: "main_live" }));
  }
  out.push({ type: "divider", section: "Section highlights", countText: "Representative stored figures from every analytical family", note: "Editable overview panels; the complete unfiltered image bank is in the appendix volumes." });
  for (const section of STATIC_SECTION_ORDER) {
    const items = representative(statics[section] || [], 6);
    chunks(items, 4).forEach(group => out.push({ type: "visual", section, items: group, role: "main_highlight" }));
  }
  return out;
}

function appendixDescriptors(statics, sections, label) {
  const count = sections.reduce((n, s) => n + (statics[s] || []).length, 0);
  const out = [
    { type: "title", title: `Dashboard Image Appendix ${label}`, subtitle: `${count} stored images · no p-value inclusion filter`, body: "Complete section-wise visual archive. Use with the Core Review Deck; exact source paths are listed in the inventory." },
    { type: "guide" },
  ];
  for (const section of sections) {
    const items = statics[section] || [];
    if (!items.length) continue;
    out.push({ type: "divider", section, countText: `Appendix · ${items.length} stored image(s)`, note: NOTES[section] });
    out.push(...appendixVisualDescriptors(section, items));
  }
  return out;
}

function fullDescriptors(core, statics) {
  const out = [...core];
  out.push({ type: "divider", section: "Complete image appendix", countText: "All 245 stored dashboard PNGs · no p-value inclusion filter", note: "Full visual archive; file-derived titles and source paths are retained in the inventory." });
  for (const section of STATIC_SECTION_ORDER) {
    const items = statics[section] || [];
    if (!items.length) continue;
    out.push({ type: "divider", section, countText: `Appendix · ${items.length} stored image(s)`, note: NOTES[section] });
    out.push(...appendixVisualDescriptors(section, items));
  }
  return out;
}

async function writeDeck(fileName, descriptors, inventory) {
  const pptx = newPresentation();
  for (const d of descriptors) {
    if (d.type === "title") renderTitle(pptx, d);
    else if (d.type === "guide") renderGuide(pptx);
    else if (d.type === "map") renderMap(pptx);
    else if (d.type === "narrative") renderNarrative(pptx);
    else if (d.type === "divider") renderDivider(pptx, d);
    else if (d.type === "visual") renderVisual(pptx, d, fileName, inventory);
  }
  const destination = path.join(OUT, fileName);
  await pptx.writeFile({ fileName: destination, compression: true });
  return { file: destination, slides: descriptors.length, bytes: fs.statSync(destination).size, sha256: sha256(destination) };
}

function csvCell(v) { const s = String(v ?? ""); return `"${s.replace(/"/g, '""')}"`; }

async function main() {
  ensureDir(OUT);
  const dynamic = loadDynamic();
  const statics = loadStatic();
  const staticCount = Object.values(statics).reduce((n, items) => n + items.length, 0);
  const dynamicCount = Object.values(dynamic).reduce((n, items) => n + items.length, 0);
  if (staticCount !== 245) throw new Error(`Expected 245 stored PNGs, found ${staticCount}`);
  if (dynamicCount !== 54) throw new Error(`Expected 54 live exports, found ${dynamicCount}`);

  const core = coreDescriptors(dynamic, statics);
  const appA = appendixDescriptors(statics, APPENDIX_A_SECTIONS, "A");
  const appB = appendixDescriptors(statics, APPENDIX_B_SECTIONS, "B");
  const full = fullDescriptors(core, statics);
  const inventory = [];
  const files = [];
  files.push(await writeDeck("01_structural_connectome_core_review_deck_v1.pptx", core, inventory));
  files.push(await writeDeck("02_structural_connectome_image_appendix_A_v1.pptx", appA, inventory));
  files.push(await writeDeck("03_structural_connectome_image_appendix_B_v1.pptx", appB, inventory));
  files.push(await writeDeck("structural_connectome_dashboard_visual_atlas_v1_FULL.pptx", full, inventory));

  const fields = ["deck", "slide_number", "role", "source_type", "section", "title", "source_path", "sha256"];
  const csv = [fields.map(csvCell).join(","), ...inventory.map(r => fields.map(f => csvCell(r[f])).join(","))].join("\n") + "\n";
  fs.writeFileSync(path.join(OUT, "dashboard_visual_inventory_compatible.csv"), csv);
  const build = {
    generated_utc: new Date().toISOString(), generator: "PptxGenJS 4.0.1", static_count: staticCount,
    dynamic_count: dynamicCount, files,
  };
  fs.writeFileSync(path.join(OUT, "build_manifest.json"), JSON.stringify(build, null, 2));
  fs.writeFileSync(path.join(OUT, "README.md"), [
    "# Structural Connectome Dashboard Visual Atlas V1 — Office-compatible rebuild", "",
    "Open `01_structural_connectome_core_review_deck_v1.pptx` first.", "",
    "The two appendix volumes contain all 245 stored dashboard images. The FULL file is supplied for convenience, but the smaller volumes are the safest files for desktop PowerPoint.", "",
    "This package was generated with a second, independent presentation engine and is validated separately with the Microsoft Open XML SDK.", "",
  ].join("\n"));
  console.log(JSON.stringify(build, null, 2));
}

main().catch(err => { console.error(err.stack || err); process.exit(1); });
