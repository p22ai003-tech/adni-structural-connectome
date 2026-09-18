"""Literature retrieval for the findings catalog via the OpenAlex works API (free, no key).

Two-tier retrieval per finding (deduped by search-term signature):
  * landmark  — publication_year >= 2024, ranked by cited_by_count (validated/established base)
  * frontier  — publication_year >= 2025, ranked by venue prestige then recency (citations haven't accrued)

Only REAL retrieved records are stored (title/authors/year/venue/cited_by_count/doi/url/abstract/tier).
Egress-guarded: if api.openalex.org is unreachable, callers should fall back to agent WebSearch retrieval
writing the same record schema.

Run:  python -m connectome_analysis.literature_openalex --mailto you@example.com
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

OPENALEX = "https://api.openalex.org/works"
# high-impact venues used to rank the frontier (2025-26) tier where citations are ~0
PRESTIGE = {
    "nature", "nature medicine", "nature neuroscience", "nature communications", "nature aging",
    "nature human behaviour", "science", "science translational medicine", "neuron", "cell",
    "brain", "the lancet neurology", "lancet neurology", "jama neurology", "alzheimer s & dementia",
    "alzheimers & dementia", "molecular psychiatry", "biological psychiatry", "annals of neurology",
    "neurology", "pnas", "proceedings of the national academy of sciences", "elife", "neuroimage",
    "national science review", "the lancet", "jama",
}


def _get_json(params: dict, timeout: int = 25) -> dict:
    url = OPENALEX + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "connectome-dashboard/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def egress_ok(mailto: str) -> bool:
    try:
        _get_json({"filter": "from_publication_date:2024-01-01", "per-page": 1, "mailto": mailto}, timeout=12)
        return True
    except Exception:
        return False


def _abstract(work: dict) -> str:
    inv = work.get("abstract_inverted_index")
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))[:1500]


def _venue(work: dict) -> str:
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return src.get("display_name") or ""


def _record(work: dict, tier: str, finding_id: str) -> dict:
    auth = [a.get("author", {}).get("display_name", "") for a in (work.get("authorships") or [])[:5]]
    doi = work.get("doi") or ""
    return {
        "finding_id": finding_id,
        "tier": tier,
        "title": work.get("display_name") or "",
        "authors": "; ".join(a for a in auth if a),
        "year": work.get("publication_year"),
        "venue": _venue(work),
        "cited_by_count": work.get("cited_by_count", 0),
        "doi": doi,
        "url": doi if doi else (work.get("id") or ""),
        "abstract": _abstract(work),
    }


def _query(search: str, from_date: str, sort: str, mailto: str, per_page: int = 8) -> list[dict]:
    params = {
        "filter": f"from_publication_date:{from_date},language:en,type:article",
        "search": search,
        "sort": sort,
        "per-page": per_page,
        "mailto": mailto,
    }
    try:
        return (_get_json(params).get("results") or [])
    except Exception:
        return []


def retrieve_for_query(search: str, finding_id: str, mailto: str,
                       n_landmark: int = 5, n_frontier: int = 5) -> list[dict]:
    # landmark: >=2024 by citations
    landmark = _query(search, "2024-01-01", "cited_by_count:desc", mailto, per_page=n_landmark + 3)
    # frontier: >=2025 ranked by prestige then recency
    frontier = _query(search, "2025-01-01", "publication_date:desc", mailto, per_page=20)
    frontier.sort(key=lambda w: (_venue(w).lower() in PRESTIGE, w.get("cited_by_count", 0),
                                 w.get("publication_year", 0)), reverse=True)
    recs, seen = [], set()
    for w in landmark[:n_landmark]:
        k = w.get("id")
        if k and k not in seen:
            seen.add(k); recs.append(_record(w, "landmark", finding_id))
    for w in frontier[:n_frontier]:
        k = w.get("id")
        if k and k not in seen:
            seen.add(k); recs.append(_record(w, "frontier", finding_id))
    return recs


def run_literature(findings_csv: Path, out_dir: Path, mailto: str, max_queries: int = 200,
                   pause: float = 0.15) -> dict:
    cat = pd.read_csv(findings_csv)
    if cat.empty:
        return {"status": "no findings"}
    out_dir.mkdir(parents=True, exist_ok=True)
    lit_dir = out_dir / "literature"
    lit_dir.mkdir(exist_ok=True)
    if not egress_ok(mailto):
        return {"status": "no_egress",
                "note": "OpenAlex unreachable — use agent WebSearch fallback writing literature_records.csv."}
    # dedupe queries by search_terms (one query backs all findings sharing it); prioritise by significance
    cat = cat.sort_values("q")
    seen_terms: dict[str, str] = {}
    all_records: list[dict] = []
    n_q = 0
    for r in cat.itertuples():
        terms = str(getattr(r, "search_terms", ""))
        fid = str(r.finding_id)
        if terms in seen_terms:
            # reuse: copy that query's records under this finding_id
            base = seen_terms[terms]
            for rec in [x for x in all_records if x["finding_id"] == base]:
                all_records.append({**rec, "finding_id": fid})
            continue
        if n_q >= max_queries:
            continue
        search = terms.replace(";", " ")
        recs = retrieve_for_query(search, fid, mailto)
        all_records.extend(recs)
        seen_terms[terms] = fid
        json.dump(recs, open(lit_dir / f"{fid}.json", "w"), indent=2)
        n_q += 1
        time.sleep(pause)
    df = pd.DataFrame(all_records)
    df.to_csv(out_dir / "literature_records.csv", index=False)
    tiers = df["tier"].value_counts().to_dict() if not df.empty else {}
    return {"status": "ok", "queries": n_q, "records": int(len(df)), "tiers": tiers,
            "unique_findings_backed": int(df["finding_id"].nunique()) if not df.empty else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--findings-csv", type=Path,
                    default=Path("/data/derivatives/qc/analysis_cohort/20_findings/findings_catalog.csv"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/data/derivatives/qc/analysis_cohort/20_findings"))
    ap.add_argument("--mailto", default="sabeesh90@gmail.com")
    ap.add_argument("--max-queries", type=int, default=200)
    args = ap.parse_args()
    print(run_literature(args.findings_csv, args.out_dir, args.mailto, args.max_queries))


if __name__ == "__main__":
    main()
