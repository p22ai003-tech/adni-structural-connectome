#!/usr/bin/env python3
"""Generate a current AAL3 route postmortem from live route ledgers."""

from __future__ import annotations

import csv
import json
import math
import statistics
import xml.sax.saxutils as sx
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


RUN_ROOT = Path(
    "/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch/"
    "ad_existing_tracks_20260529T055752Z"
)
REPORT_DIR = Path("/home/ec2-user/exp/reports")
NODE_QC = REPORT_DIR / "aal3_node_qc" / "aal3_node_qc_latest.csv"
ENDPOINT_ACTION_CSV = (
    REPORT_DIR
    / "aal3_endpoint_space_contract_ad_production_5k"
    / "aal3_ad_endpoint_space_action_latest.csv"
)
ENDPOINT_ACTION_JSON = (
    REPORT_DIR
    / "aal3_endpoint_space_contract_ad_production_5k"
    / "aal3_ad_endpoint_space_action_latest.json"
)


ROUTE_METHODS = {
    "R1": "Existing 3M tracks plus source-contract AAL3 forced connectome.",
    "R2": "Second-pass existing-track rescue after R1 failures.",
    "R3": "Assignment-focused retry with expanded endpoint assignment variants.",
    "R4": "Label-rescue route testing parcellation/label survival variants.",
    "R5": "Deep assignment route with deeper endpoint searches.",
    "R6": "All-voxels diagnostic route; not production-safe.",
    "R7": "Selective track rescue with alternate track subsets/settings.",
    "R8": "Full native T1/FNIRT AAL3-to-B0 registration route.",
    "R9": "Route8 assignment retry with alternate endpoint settings.",
    "R10": "Endpoint overlay triage diagnostic.",
    "R11A": "Space/affine registration repair.",
    "R11B": "ACT/SIFT2 track-coverage rerun.",
    "R11C": "Zero-row coverage rescue.",
    "R12A": "Fallback tckgen rescue.",
    "R12B": "Zero-row targeted rerun.",
    "R12C": "Registration-contract repair.",
    "R13": "Step7 preflight rebuild of native T1/ACT prerequisites.",
    "R14": "Step7 rebuilt track coverage.",
    "R14B": "Step7 assignment retry.",
    "R15": "Step7 dense track coverage.",
    "R16": "Zero-ROI hybrid route.",
    "R17": "Step7 dense route for unpushed unresolved subjects.",
    "R18": "GMWMI dense track coverage, 6M ACT/GMWMI seeded tracks.",
    "R19": "Staged GMWMI 3M retry.",
    "R20": "Zero-row endpoint canary; targeted per-ROI GMWMI/FOD endpoint coverage.",
    "R21": (
        "Endpoint-shell assignment-map canary; label-shell/dilation rescue using "
        "existing 3M tracks without production overwrite."
    ),
}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def fmt(value: object, digits: int = 4) -> str:
    val = to_float(value)
    if val is None:
        return ""
    return f"{val:.{digits}f}"


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.fmean(vals) if vals else None


def median(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.median(vals) if vals else None


def route_root(route: str, routes: dict) -> Path | None:
    info = routes.get(route, {})
    root = info.get("root")
    return Path(root) if root else None


def summarize_routes(summary: dict, ledger_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    routes = summary.get("routes", {})
    full_by_route = Counter(
        r.get("promoted_route")
        for r in ledger_rows
        if r.get("state") == "solved_full_qc_production" and r.get("promoted_route")
    )
    review_by_route = Counter(
        r.get("promoted_route")
        for r in ledger_rows
        if r.get("state") != "solved_full_qc_production" and r.get("promoted_route")
    )
    rows: list[dict[str, object]] = []
    for route in sorted(routes, key=route_sort_key):
        info = routes[route]
        root = route_root(route, routes)
        decisions = read_csv(root / "route_qc_decisions.csv") if root else []
        density_deltas = [to_float(r.get("density_delta")) for r in decisions]
        density_deltas = [v for v in density_deltas if v is not None]
        zero_deltas = [to_float(r.get("zero_rows_delta")) for r in decisions]
        zero_deltas = [v for v in zero_deltas if v is not None and abs(v) < 9000]
        both_assigned = [to_float(r.get("both_assigned_fraction")) for r in decisions]
        both_assigned = [v for v in both_assigned if v is not None]
        labels = [
            to_float(r.get("source_labels") or r.get("production_labels"))
            for r in decisions
        ]
        labels = [v for v in labels if v is not None and v > 0]
        reason_counter: Counter[str] = Counter()
        for r in decisions:
            reasons = r.get("qc_reasons") or r.get("final_qc_reasons") or r.get("review_qc_reasons") or ""
            for part in reasons.split(";"):
                clean = part.strip()
                if clean:
                    reason_counter[clean] += 1
        top_reasons = " | ".join(f"{k} ({v})" for k, v in reason_counter.most_common(3))
        decision_counts = info.get("decisions", {})
        rows.append(
            {
                "route": route,
                "method": ROUTE_METHODS.get(route, ""),
                "phase": info.get("phase", ""),
                "completed": info.get("completed", 0),
                "total": info.get("total", 0),
                "decision_rows": info.get("decision_rows", 0),
                "promote_pending_visual": decision_counts.get("PROMOTE_CANDIDATE_PENDING_VISUAL", 0),
                "interim_review": decision_counts.get("INTERIM_REVIEW_PROMOTION_PENDING_VISUAL", 0),
                "keep_production_numeric_ok": decision_counts.get("KEEP_PRODUCTION_NUMERIC_OK", 0),
                "try_another_route": decision_counts.get("TRY_ANOTHER_ROUTE", 0),
                "promoted_rows": info.get("promoted_rows", 0),
                "full_qc_solved_by_route": full_by_route.get(route, 0),
                "review_push_by_route": review_by_route.get(route, 0),
                "density_improved_n": sum(1 for v in density_deltas if v > 0),
                "density_worse_n": sum(1 for v in density_deltas if v < 0),
                "mean_density_delta": mean(density_deltas),
                "median_density_delta": median(density_deltas),
                "zero_rows_improved_n": sum(1 for v in zero_deltas if v < 0),
                "zero_rows_worse_n": sum(1 for v in zero_deltas if v > 0),
                "mean_zero_rows_delta": mean(zero_deltas),
                "mean_source_labels": mean(labels),
                "mean_both_assignment": mean(both_assigned),
                "top_failure_reasons": top_reasons,
                "root": str(root) if root else "",
            }
        )
    return rows


def route_sort_key(route: str) -> tuple[int, str]:
    if not route.startswith("R"):
        return (9999, route)
    rest = route[1:]
    number = ""
    suffix = ""
    for ch in rest:
        if ch.isdigit() and not suffix:
            number += ch
        else:
            suffix += ch
    return (int(number) if number else 9999, suffix)


def node_qc_summary() -> tuple[Counter[str], list[dict[str, str]]]:
    rows = read_csv(NODE_QC)
    actions = Counter(r.get("aal3cc_action_recommendation", "") for r in rows)
    top = sorted(rows, key=lambda r: to_float(r.get("zero_row_frequency")) or 0, reverse=True)[:15]
    return actions, top


def endpoint_action_summary() -> tuple[dict, Counter[str], list[dict[str, str]], list[dict[str, str]]]:
    payload = read_json(ENDPOINT_ACTION_JSON)
    rows = read_csv(ENDPOINT_ACTION_CSV)
    actions = Counter(r.get("recommended_action", "") for r in rows)
    collapsed = [r for r in rows if r.get("recommended_action") == "upstream_track_fod_act_repair"]
    collapsed = sorted(
        collapsed,
        key=lambda r: to_float(r.get("top3_endpoint_label_fraction")) or 0,
        reverse=True,
    )
    plausible_open = [
        r
        for r in rows
        if r.get("recommended_action") == "continue_parcellation_assignment_routes"
    ]
    plausible_open = sorted(
        plausible_open,
        key=lambda r: to_float(r.get("best_zero_rows")) or 9999,
        reverse=True,
    )
    return payload, actions, collapsed[:15], plausible_open[:15]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def make_markdown(
    stamp: str,
    summary: dict,
    route_rows: list[dict[str, object]],
    r20_rows: list[dict[str, str]],
    node_actions: Counter[str],
    top_nodes: list[dict[str, str]],
    endpoint_payload: dict,
    endpoint_actions: Counter[str],
    collapsed_endpoint_rows: list[dict[str, str]],
    plausible_open_rows: list[dict[str, str]],
) -> str:
    current = [
        ["AD subjects in scope", summary.get("total_subjects", "")],
        ["Production-visible pushed", summary.get("production_unique_subjects", "")],
        ["Full QC solved", summary.get("production_full_qc_solved", "")],
        ["Review/interim pushed but not solved", summary.get("production_interim_review_not_solved", "")],
        ["Numeric pending visual, still open", summary.get("production_numeric_pending_visual_not_solved", "")],
        ["Need next route / unresolved", summary.get("state_counts", {}).get("needs_next_route", "")],
    ]
    route_table_rows = []
    for r in route_rows:
        route_table_rows.append(
            [
                r["route"],
                r["phase"],
                f"{r['completed']}/{r['total']}",
                r["promote_pending_visual"],
                r["interim_review"],
                r["try_another_route"],
                r["promoted_rows"],
                r["full_qc_solved_by_route"],
                r["review_push_by_route"],
                r["density_improved_n"],
                r["density_worse_n"],
                fmt(r["mean_density_delta"], 4),
                r["zero_rows_improved_n"],
                r["zero_rows_worse_n"],
                fmt(r["mean_both_assignment"], 3),
            ]
        )
    r20_table = []
    for r in r20_rows:
        r20_table.append(
            [
                r.get("sid", ""),
                r.get("qc_decision", ""),
                r.get("best_variant", ""),
                fmt(r.get("baseline_density"), 4),
                fmt(r.get("best_density"), 4),
                fmt(r.get("density_delta"), 4),
                r.get("baseline_zero_rows", ""),
                r.get("best_zero_rows", ""),
                r.get("zero_rows_delta", ""),
                fmt(r.get("both_assigned_fraction"), 3),
                r.get("final_qc_status", ""),
            ]
        )
    node_table = [
        [
            r.get("aal3_label", ""),
            r.get("aal3_name", ""),
            r.get("subjects_zero_row", ""),
            r.get("subjects_with_label", ""),
            fmt(r.get("label_survival_frequency"), 2),
            fmt(r.get("zero_row_frequency"), 2),
            r.get("median_label_voxels", ""),
            r.get("aal3cc_action_recommendation", ""),
        ]
        for r in top_nodes
    ]
    action_text = ", ".join(f"{k}: {v}" for k, v in sorted(node_actions.items()))
    endpoint_action_text = ", ".join(
        f"{k}: {v}" for k, v in sorted(endpoint_actions.items())
    )
    collapsed_table = [
        [
            r.get("sid", ""),
            r.get("best_route", ""),
            fmt(r.get("best_density"), 4),
            r.get("best_zero_rows", ""),
            r.get("parc_label_count", ""),
            r.get("unique_endpoint_labels", ""),
            fmt(r.get("top3_endpoint_label_fraction"), 3),
            fmt(r.get("endpoint_on_label_fraction"), 3),
            fmt(r.get("bbox_center_distance_mm"), 1),
            r.get("recommended_action", ""),
        ]
        for r in collapsed_endpoint_rows
    ]
    plausible_table = [
        [
            r.get("sid", ""),
            r.get("best_route", ""),
            fmt(r.get("best_density"), 4),
            r.get("best_zero_rows", ""),
            r.get("parc_label_count", ""),
            r.get("unique_endpoint_labels", ""),
            fmt(r.get("top3_endpoint_label_fraction"), 3),
            fmt(r.get("endpoint_on_label_fraction"), 3),
            r.get("recommended_action", ""),
        ]
        for r in plausible_open_rows
    ]
    return "\n\n".join(
        [
            "# AAL3 AD Sparse Connectome Current Postmortem",
            f"Generated UTC: {stamp}",
            "## Executive Conclusion",
            (
                "No new route should be described as 'the solution' until it clears "
                "the numeric and visual QC gates on more than a canary. R18/R20/R21 "
                "are hypotheses, not proven fixes. The newest endpoint-space audit "
                "shows two different unresolved failure modes, so one more atlas or "
                "assignment variant cannot solve the full cohort."
            ),
            "## Current AD State",
            md_table(["Metric", "Value"], current),
            (
                "Important: pushed is not solved. Pushed review matrices are allowed "
                "as temporary production refreshes, but they remain open until numeric "
                "QC and visual overlay QC both pass."
            ),
            "## Route Summary",
            md_table(
                [
                    "Route",
                    "Phase",
                    "Done",
                    "Full-candidate",
                    "Review",
                    "Fail/next",
                    "Route-pushed",
                    "Full QC",
                    "Review push",
                    "Density +",
                    "Density -",
                    "Mean dDensity",
                    "Zero improved",
                    "Zero worsened",
                    "Mean both assign",
                ],
                route_table_rows,
            ),
            "## R20 Canary Evidence",
            md_table(
                [
                    "Subject",
                    "Decision",
                    "Variant",
                    "Base dens",
                    "Cand dens",
                    "Delta dens",
                    "Base zero",
                    "Cand zero",
                    "Delta zero",
                    "Both assign",
                    "Final status",
                ],
                r20_table,
            ),
            (
                "R20 interpretation: targeted per-ROI tracks can be generated for some "
                "zero-row labels, but the merged candidate did not improve the matrix. "
                "For <SUBJECT>_I<IMAGEID> it reduced density from 0.5035 to 0.2420 and "
                "increased zero rows from 15 to 40. This rejects the current R20 design "
                "as a scalable production route."
            ),
            "## Endpoint-Space Triage",
            (
                f"Endpoint audit generated UTC: {endpoint_payload.get('generated_utc', '')}. "
                f"Action split: {endpoint_action_text}."
            ),
            (
                "This is the important correction to the route narrative. The 10 "
                "endpoint-collapse subjects have endpoints inside the image but "
                "collapsed into only a few AAL3 labels, so parcellation-only rescue "
                "cannot create a scientifically valid dense matrix. The 80 plausible "
                "open subjects still have a broad enough endpoint cloud; those should "
                "continue through parcellation/assignment/label-contract repair rather "
                "than upstream tractography first."
            ),
            "### Endpoint-Collapse Subjects",
            md_table(
                [
                    "Subject",
                    "Best route",
                    "Best density",
                    "Best zero",
                    "Labels",
                    "Unique endpoint labels",
                    "Top3 frac",
                    "On-label frac",
                    "BBox dist mm",
                    "Action",
                ],
                collapsed_table,
            ),
            "### Highest-Zero Open Plausible Subjects",
            md_table(
                [
                    "Subject",
                    "Best route",
                    "Best density",
                    "Best zero",
                    "Labels",
                    "Unique endpoint labels",
                    "Top3 frac",
                    "On-label frac",
                    "Action",
                ],
                plausible_table,
            ),
            "## AAL3 Node-Level Pattern",
            f"AAL3-CC screening actions: {action_text}",
            md_table(
                [
                    "Label",
                    "ROI",
                    "Zero-row subjects",
                    "Subjects with label",
                    "Label survival freq",
                    "Zero-row freq",
                    "Median voxels",
                    "Recommendation",
                ],
                node_table,
            ),
            "## Root-Cause Interpretation",
            (
                "The dominant failure is not simply global tract count. Existing-track "
                "routes can increase density and still fail because zero-row AAL3 ROIs "
                "remain. Dense reruns can also fail when the endpoint cloud is collapsed "
                "or when small/deep AAL3 labels do not intersect a valid endpoint/GM-WM "
                "interface. For the endpoint-collapse subset, the likely root is upstream "
                "track/FOD/ACT generation or a bad chunked tractogram contract. For the "
                "endpoint-plausible subset, the likely root remains atlas/assignment/"
                "label-contract mismatch."
            ),
            "## Non-Circular Next Action",
            (
                "Stop presenting any route as a guaranteed fix. Split the queue: "
                "1) endpoint-collapse subjects get upstream tract/FOD/ACT repair and "
                "must pass an endpoint-space preflight before SIFT2/connectome work; "
                "2) endpoint-plausible subjects get high-resolution T1-native nonlinear "
                "AAL3 contract and assignment repair; 3) full AAL3 remains forensic/"
                "sensitivity until a robust AAL3-compatible node map is defined. "
                "Promotion still requires numeric QC plus visual overlay, not density "
                "improvement alone."
            ),
            "## Source Artifacts",
            f"- Ledger: {RUN_ROOT / 'route_subject_ledger.csv'}",
            f"- Ledger summary: {RUN_ROOT / 'route_subject_ledger_summary.json'}",
            f"- R20 decisions: {RUN_ROOT}_route20_zero_roi_gmwmi_endpoint/route_qc_decisions.csv",
            f"- Node QC: {NODE_QC}",
            f"- Endpoint-space action CSV: {ENDPOINT_ACTION_CSV}",
        ]
    )


def docx_paragraph(text: str, style: str | None = None) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    runs = []
    for line in text.split("\n"):
        runs.append(f"<w:r><w:t>{sx.escape(line)}</w:t></w:r>")
        runs.append("<w:r><w:br/></w:r>")
    if runs:
        runs.pop()
    return f"<w:p>{ppr}{''.join(runs)}</w:p>"


def markdown_to_docx(md: str, out_path: Path) -> None:
    body_parts = []
    for block in md.split("\n\n"):
        if block.startswith("# "):
            body_parts.append(docx_paragraph(block[2:], "Heading1"))
        elif block.startswith("## "):
            body_parts.append(docx_paragraph(block[3:], "Heading2"))
        else:
            body_parts.append(docx_paragraph(block))
    document = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(body_parts)
        + '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>'
        "</w:sectPr></w:body></w:document>"
    )
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = read_json(RUN_ROOT / "route_subject_ledger_summary.json")
    ledger_rows = read_csv(RUN_ROOT / "route_subject_ledger.csv")
    route_rows = summarize_routes(summary, ledger_rows)
    r20_rows = read_csv(Path(str(RUN_ROOT) + "_route20_zero_roi_gmwmi_endpoint") / "route_qc_decisions.csv")
    node_actions, top_nodes = node_qc_summary()
    (
        endpoint_payload,
        endpoint_actions,
        collapsed_endpoint_rows,
        plausible_open_rows,
    ) = endpoint_action_summary()

    csv_path = REPORT_DIR / f"aal3_route_consolidated_summary_{stamp}.csv"
    md_path = REPORT_DIR / f"aal3_route_consolidated_postmortem_{stamp}.md"
    docx_path = REPORT_DIR / f"aal3_route_consolidated_postmortem_{stamp}.docx"
    write_csv(csv_path, route_rows)
    md = make_markdown(
        stamp,
        summary,
        route_rows,
        r20_rows,
        node_actions,
        top_nodes,
        endpoint_payload,
        endpoint_actions,
        collapsed_endpoint_rows,
        plausible_open_rows,
    )
    md_path.write_text(md)
    markdown_to_docx(md, docx_path)
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "docx": str(docx_path)}, indent=2))


if __name__ == "__main__":
    main()
