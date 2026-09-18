"""Phase K: assemble a paper-grade novelty report (findings -> evidence -> novelty + numbered references)
from novelty_cards.json + findings_catalog.csv. Writes 20_findings/novelty_report.md."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FINDINGS_DIR = Path("/data/derivatives/qc/analysis_cohort/20_findings")


def _ref_line(n: int, c: dict) -> str:
    bits = [c.get("authors", "").strip(" ;"), f"({c.get('year')})", c.get("title", ""), f"*{c.get('venue','')}*"]
    line = ". ".join(b for b in bits if b)
    url = c.get("doi") or c.get("url") or ""
    return f"{n}. {line}. {url}".strip()


def build_report(out_dir: Path = FINDINGS_DIR) -> Path:
    cards = json.loads((out_dir / "novelty_cards.json").read_text()) if (out_dir / "novelty_cards.json").exists() else []
    cat = pd.read_csv(out_dir / "findings_catalog.csv") if (out_dir / "findings_catalog.csv").exists() else pd.DataFrame()
    lit = pd.read_csv(out_dir / "literature_records.csv") if (out_dir / "literature_records.csv").exists() else pd.DataFrame()

    # global numbered reference list (dedup by url/title)
    refs, key2num = [], {}
    for c in cards:
        for cit in c.get("citations", []):
            k = (cit.get("doi") or cit.get("url") or cit.get("title"))
            if k not in key2num:
                key2num[k] = len(refs) + 1
                refs.append(cit)

    L = []
    L.append("# Network-resolved structural-connectome findings in AD: literature-grounded novelty\n")
    L.append("Cohort: 530 dense AAL3 (166-node) structural connectomes (AD 78 / MCI 201 / CN 251); diffusivity "
             "metrics on n=529. Networks: functional (Yeo-7 + subcortical/cerebellar/brainstem) and anatomical "
             "systems. Group statistics: Kruskal–Wallis + permutation-ANCOVA, Cliff's δ for CN-vs-AD, BH-FDR.\n")
    L.append("## How to read this\n")
    L.append("Each theme separates **(a) our finding** (this cohort's data), **(b) what the literature says** "
             "(real retrieved 2024-26 records), and **(c) the novelty claim** (a literature-validated hypothesis, "
             "not a proven discovery). Two citation tiers: *landmark* (≥2024, established) and *frontier* (2025-26, "
             "venue-ranked). Preprints are flagged and are not peer-reviewed.\n")
    L.append("### Honest limitations\n")
    L.append("- Structural (DWI tractography) connectomes — not functional connectivity.\n"
             "- Functional-network labels are an approximation of Yeo networks on an anatomical atlas (see "
             "`AAL3_network_mapping_notes.md`).\n"
             "- Automated keyword retrieval (OpenAlex) surfaces high-citation but sometimes off-topic papers; the "
             "citations below were web-verified for on-topic relevance, but literature coverage is not exhaustive.\n"
             "- Age associations are cross-sectional; age and diagnosis are partially confounded.\n")

    for i, c in enumerate(cards, 1):
        L.append(f"## {i}. {c['theme']}")
        L.append(f"*Novelty type:* {c.get('novelty_type','')} · *confidence:* {c.get('confidence','')}\n")
        L.append(f"**Our finding.** {c.get('our_finding','')}\n")
        L.append(f"**What's known (landmark).** {c.get('prior_literature','')}\n")
        if c.get("frontier_context"):
            L.append(f"**Frontier (2025-26).** {c['frontier_context']}\n")
        L.append(f"**Novelty.** {c.get('novelty_claim','')}\n")
        if c.get("caveat"):
            L.append(f"**Caveat.** {c['caveat']}\n")
        nums = [str(key2num[(cit.get('doi') or cit.get('url') or cit.get('title'))]) for cit in c.get("citations", [])]
        if nums:
            L.append(f"*References:* [{'], ['.join(nums)}]\n")

    if not cat.empty:
        L.append("## Appendix — significant-findings catalog (top 30 by q)")
        top = cat.sort_values("q").head(30)
        L.append("| family | scheme | network | metric | direction | δ/ρ | q |")
        L.append("|---|---|---|---|---|---|---|")
        for r in top.itertuples():
            L.append(f"| {r.feature_family} | {r.scheme} | {r.network} | {r.metric} | {r.direction} | "
                     f"{r.effect_cliffs_delta:+.2f} | {r.q:.2g} |")
        L.append(f"\n*Total significant findings: {len(cat)}.*\n")

    L.append("## References")
    for cit in refs:
        L.append(_ref_line(key2num[(cit.get('doi') or cit.get('url') or cit.get('title'))], cit))

    out = out_dir / "novelty_report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build_report()
    print(f"wrote {p} ({p.stat().st_size} bytes)")
