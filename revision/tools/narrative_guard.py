"""Narrative guard: a document builder stops when computed numbers contradict a fixed sentence.

The builders print sentences such as "there was no crossover" or "13 of 14 kept their sign" whatever the data say. Every
document now passes through check_document() before it can be saved (iit_chain.save_new, iit_chain.publish). It reads the
document's live text (struck runs are ignored), finds each sentence that makes one of the claims below, and tests that
claim against this chain's own phase tables.

A claim stops the build in three cases: the numbers contradict it, the table behind it does not exist, or the sentence
uses the phrase in a way no test here covers. A new sentence with one of these phrases therefore needs a test.

The thresholds were written down before any rehearsal or recipe-v4 number was seen; they are not to be tuned to a result:
  "agreed"              median Pearson r of whole-brain values >= 0.80 for every measure (the cut p5_act.py calls "closely")
  "stable"              median split-half Dice of the core >= 0.70, whole sample and controls alone
  "near-perfect"        the same Dice >= 0.90
  "no crossover"        no sign reversal between CN-MCI and a later contrast with either interval excluding zero, in the
                        length slopes or in the short-minus-long difference of the primary analysis
  "at every stage"      as a statement of length dependence: the slope has the stated sign for MD and RD in all three
                        contrasts; as "similar degree" or "about equal": every short-minus-long interval includes zero
  "robust to"           the named effect is significant (q < .05) with one sign in every planned condition it lists

Usage (read only): narrative_guard.py FILE.docx ...    prints what the guard finds in existing documents.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iit_chain as chain  # noqa: E402
import sc_cohort_settings as S  # noqa: E402

AGREE_R = 0.80
DICE_STABLE = 0.70
DICE_NEAR_PERFECT = 0.90
STAGES = ("CN-MCI", "CN-AD", "MCI-AD")
PLANNED = ("C1", "C2", "C3", "C4", "C5", "C8")
TAG = "uniform_v3"


class Evidence:
    """This chain's tables, read once; None when a table does not exist."""

    def __init__(self):
        self._cache = {}

    def _csv(self, path):
        import pandas as pd

        path = Path(path)
        if path not in self._cache:
            self._cache[path] = pd.read_csv(path) if path.is_file() else None
        return self._cache[path]

    def slopes(self, dkt=False):
        if dkt:
            return self._csv(chain.P(5) / f"dkt/phase2/results/interaction_{TAG}_dkt_P2.csv")
        d = self._csv(chain.P(2) / f"tables/interaction_{TAG}.csv")
        return None if d is None else d[d.condition == "P2"]

    def short_long(self):
        d = self._csv(chain.P(1) / "sr_lr_summary.csv")
        if d is None:
            return None
        return d[(d.condition.str.split().str[0] == "C2") & (d.kind == "TR")]

    def effects(self, dkt=False):
        d = self._csv((chain.P(5) / "dkt/phase1" if dkt else chain.P(1)) / "effects_long.csv")
        if d is None:
            return None
        return d.assign(c=d.condition.str.split().str[0])

    def split_half(self, dkt=False):
        base = chain.P(5) / "dkt/phase3/results" if dkt else chain.P(3) / "results"
        d = self._csv(base / f"C_split_half_{TAG}{'_dkt' if dkt else ''}.csv")
        return None if d is None else d[d.condition == "P2"]

    def controls_only(self):
        d = self._csv(chain.P(3) / f"results/B_controls_only_split_{TAG}.csv")
        return None if d is None else d[d.condition == "P2"]

    def curves(self):
        d = self._csv(chain.P(2) / f"tables/curves_{TAG}.csv")
        return None if d is None else d[(d.condition == "P2") & (d.region == "Whole brain")]

    def act_agreement(self):
        return self._csv(chain.P(5) / "tables/act_agreement.csv")

    def act_effects(self):
        return self._csv(chain.P(5) / "tables/act_key_effects.csv")

    def nulls_variants(self):
        return self._csv(chain.P(5) / "tables/nulls_variants.csv")

    def n_sessions(self):
        d = self._csv(chain.P(1) / "covariates.csv")
        return None if d is None else len(d)

    def n_before_holdout(self):
        """Passing QC, plus those held out for no other reason than the analysis's own hold-out (SMC)."""
        d = self._csv(chain.P(1) / "qc.csv")
        if d is None:
            return None
        return int((~d.qc_fail.astype(bool)).sum()) + int((d.qc_reason == "R0 excluded before the analysis").sum())

    def recipe(self):
        conn = Path(os.environ.get("IIT_CONN_RAW") or os.environ.get("IIT_CONN") or "/data/derivatives/connectomes_uniform_v3/matrices")
        for f in sorted(conn.glob("SC_*_qc.json")):
            return json.loads(f.read_text())
        return None


def _excludes_zero(lo, hi):
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)


NO_EVIDENCE = "the table behind this sentence does not exist"


# ---------------------------------------------------------------------------- tests: each returns None or why it fails
def t_no_crossover(sentence, ev):
    sl, sr = ev.slopes(), ev.short_long()
    if sl is None or sr is None:
        return NO_EVIDENCE + " (Phase 2 interaction table, Phase 1 short/long summary)"
    why = []
    for m in ("FA", "MD", "RD", "AxD"):
        d = sl[(sl.region == "Whole brain") & (sl.measure == m)].set_index("contrast")
        if "CN-MCI" not in d.index:
            continue
        a = d.loc["CN-MCI"]
        for later in ("CN-AD", "MCI-AD"):
            if later not in d.index:
                continue
            b = d.loc[later]
            if a.I_delta * b.I_delta < 0 and (_excludes_zero(a.I_delta_lo, a.I_delta_hi) or _excludes_zero(b.I_delta_lo, b.I_delta_hi)):
                why.append(f"{m} slope per doubling reverses: CN-MCI {a.I_delta:+.3f} [{a.I_delta_lo:+.3f}, {a.I_delta_hi:+.3f}], "
                           f"{later} {b.I_delta:+.3f} [{b.I_delta_lo:+.3f}, {b.I_delta_hi:+.3f}]")
    d = sr.set_index("contrast")
    if "CN-MCI" in d.index:
        a = d.loc["CN-MCI"]
        for later in ("CN-AD", "MCI-AD"):
            if later in d.index:
                b = d.loc[later]
                if a["diff"] * b["diff"] < 0 and (_excludes_zero(a.ci_lo, a.ci_hi) or _excludes_zero(b.ci_lo, b.ci_hi)):
                    why.append(f"short minus long reverses: CN-MCI {a['diff']:+.3f} [{a.ci_lo:+.3f}, {a.ci_hi:+.3f}], "
                               f"{later} {b['diff']:+.3f} [{b.ci_lo:+.3f}, {b.ci_hi:+.3f}]")
    return "; ".join(why) or None


def t_every_stage(sentence, ev):
    why, tested = [], False
    if re.search(r"similar degree|about equal|\bsimilar\b", sentence):
        tested = True
        sr = ev.short_long()
        if sr is None:
            return NO_EVIDENCE + " (Phase 1 short/long summary)"
        for r in sr.itertuples():
            if _excludes_zero(r.ci_lo, r.ci_hi):
                why.append(f"short minus long {r.contrast} {r.diff:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] excludes zero")
    if re.search(r"limbic exception effects", sentence):
        tested = True
        e = ev.effects()
        if e is None:
            return NO_EVIDENCE + " (Phase 1 effects)"
        x = e[(e.c == "C2") & (e.kind == "EX") & (e.network == "Limbic")]
        for ct in STAGES:
            if not (x[x.contrast == ct].q < 0.05).any():
                why.append(f"no limbic exception effect at {ct} has q < .05")
    if re.search(r"mean and radial|short connections|short-range weighting|same way|per doubling|with (edge )?length", sentence):
        tested = True
        sl = ev.slopes(dkt="DKT" in sentence)
        if sl is None:
            return NO_EVIDENCE + " (length slopes)"
        for m in ("MD", "RD"):
            for ct in STAGES:
                x = sl[(sl.region == "Whole brain") & (sl.measure == m) & (sl.contrast == ct)]
                if x.empty:
                    why.append(f"{m} {ct}: no slope")
                elif not x.I_delta.iloc[0] < 0:
                    why.append(f"{m} {ct} slope {x.I_delta.iloc[0]:+.3f} is not negative (short connections not more affected)")
    if not tested:
        return "no test covers 'every stage' as used in this sentence"
    return "; ".join(why) or None


def t_n_of_m(sentence, ev):
    ke = ev.act_effects()
    if ke is None:
        return NO_EVIDENCE + " (Phase 5 ACT effects)"
    import numpy as np

    num = lambda x: float(str(x).replace("−", "-"))
    same = sum(np.sign(num(r[f"unconstrained: {c}"])) == np.sign(num(r[f"ACT: {c}"])) for _, r in ke.iterrows() for c in ("CN–MCI", "CN–AD"))
    total = 2 * len(ke)
    m = re.search(r"(\d+) of (\d+) leading group effects|(13) of (14)", sentence)
    said = [int(x) for x in m.groups() if x]
    if said != [same, total]:
        return f"the sentence says {said[0]} of {said[1]}; this chain's ACT table gives {same} of {total}"
    return None


def t_agreed(sentence, ev):
    ag = ev.act_agreement()
    if ag is None:
        return NO_EVIDENCE + " (Phase 5 ACT agreement)"
    wb = ag[ag.Edges == "whole brain"]
    if re.search(r"\b(?:not|never|disagreed?|poorly)\b", sentence):
        return None                                  # says the two do not agree: the claim this test exists to stop is not made
    rr = {str(m): float(str(v).split(" ")[0]) for m, v in zip(wb["Measure"], wb["Pearson r, median [range]"])}
    low = {k: v for k, v in rr.items() if v < AGREE_R}
    if low:
        return f"median Pearson r below {AGREE_R} for " + ", ".join(f"{k} {v:.2f}" for k, v in low.items())
    return None


def _dice(ev, dkt):
    d = ev.split_half(dkt=dkt)
    if d is None or d.empty:
        return None
    return {r.sample: float(r.dice_median) for r in d.itertuples()}


def t_stable(sentence, ev):
    if re.search(r"\bcore\b|anatomical feature|exception", sentence, re.I):
        dice = _dice(ev, dkt="DKT" in sentence)
        if dice is None:
            return NO_EVIDENCE + " (Phase 3 split-half)"
        low = {k: v for k, v in dice.items() if v < DICE_STABLE}
        return f"split-half Dice below {DICE_STABLE}: " + ", ".join(f"{k} {v:.2f}" for k, v in low.items()) if low else None
    if re.search(r"\beffects?\b", sentence):
        return _same_sign_significant(ev, ("WB|Limbic|AxD",), ("CN-MCI",), PLANNED)
    return "no test covers 'stable' as used in this sentence"


def t_near_perfect(sentence, ev):
    dice = _dice(ev, dkt=False)
    if dice is None:
        return NO_EVIDENCE + " (Phase 3 split-half)"
    low = {k: v for k, v in dice.items() if v < DICE_NEAR_PERFECT}
    return f"split-half Dice below {DICE_NEAR_PERFECT}: " + ", ".join(f"{k} {v:.2f}" for k, v in low.items()) if low else None


def _same_sign_significant(ev, features, contrasts, conditions, dkt_too=False):
    e = ev.effects()
    if e is None:
        return NO_EVIDENCE + " (Phase 1 effects)"
    why = []
    for f in features:
        for ct in contrasts:
            signs = []
            for c in conditions:
                x = e[(e.c == c) & (e.feature == f) & (e.contrast == ct)]
                if x.empty or not float(x.q.iloc[0]) < 0.05:
                    why.append(f"{f} {ct} {c}: " + ("no estimate" if x.empty else f"δ {x.delta.iloc[0]:+.3f}, q {x.q.iloc[0]:.3f}"))
                elif len(x):
                    signs.append(x.delta.iloc[0] > 0)
            if len(set(signs)) > 1:
                why.append(f"{f} {ct}: the sign changes between conditions")
    if dkt_too:
        d = ev.effects(dkt=True)
        if d is None:
            why.append(NO_EVIDENCE + " (Phase 5 DKT effects)")
        else:
            for f in features:
                for ct in contrasts:
                    x = d[(d.c == "C2") & (d.feature == f) & (d.contrast == ct)]
                    if x.empty or not float(x.q.iloc[0]) < 0.05:
                        why.append(f"{f} {ct} DKT: " + ("no estimate" if x.empty else f"δ {x.delta.iloc[0]:+.3f}, q {x.q.iloc[0]:.3f}"))
    return "; ".join(why) or None


def t_robust(sentence, ev):
    limbic = ("WB|Limbic|AxD", "WB|Limbic|MD")
    if re.search(r"non-Gaussian|rank-based", sentence, re.I):
        return None                                  # a property of the statistics, not a claim about a finding
    if re.search(r"streamline counts", sentence):
        n = ev.nulls_variants()
        if n is None:
            return NO_EVIDENCE + " (Phase 5 variants)"
        x = n[n.Variant == "Streamline-count weights"]
        if x.empty or float(x["Core edges (pooled)"].iloc[0]) <= 0 or float(x["Split-half Dice (P2)"].iloc[0]) < DICE_STABLE:
            return "the core does not hold with streamline-count weights"
        return None
    if re.search(r"null models", sentence):
        b = ev.controls_only()
        if b is None:
            return NO_EVIDENCE + " (Phase 3 controls-only)"
        weak = b[b.contrast.isin(["CN-AD", "MCI-AD"]) & (b.share_p_below_05 < 0.5)]
        return "controls-only coverage loss is significant in under half of splits for " + ", ".join(weak.contrast) if len(weak) else None
    if re.search(r"AD-stage", sentence):
        return _same_sign_significant(ev, limbic, ("CN-AD",), PLANNED)
    if re.search(r"parcellation", sentence):
        return _same_sign_significant(ev, limbic, ("CN-MCI", "CN-AD"), PLANNED, dkt_too=True)
    if re.search(r"early|widespread|most robust", sentence, re.I):
        return _same_sign_significant(ev, limbic, ("CN-MCI",), PLANNED)
    return "no test covers 'robust' as used in this sentence"


def t_every_network(sentence, ev):
    e = ev.effects()
    if e is None:
        return NO_EVIDENCE + " (Phase 1 effects)"
    x = e[(e.c == "C2") & (e.kind == "WB") & e.measure.isin(["MD", "RD", "AxD"]) & e.contrast.isin(["CN-MCI", "CN-AD"])]
    if x.empty:
        return "no whole-brain diffusivity effects in the primary analysis"
    bad = x[~(x.delta < 0)]
    return f"{len(bad)} of {len(x)} whole-brain diffusivity effects are not raised in the later group" if len(bad) else None


def t_every_length(sentence, ev):
    c = ev.curves()
    if c is None:
        return NO_EVIDENCE + " (Phase 2 curves)"
    x = c[c.measure.isin(["MD", "RD", "AxD"]) & c.contrast.isin(["CN-MCI", "CN-AD"])]
    bad = x[~(x.delta < 0)]
    return f"{len(bad)} of {len(x)} length bins show no rise in diffusivity" if len(bad) else None


def t_sample(sentence, ev):
    if re.search(r"\bsensitivity\b", sentence, re.I):
        return None                                  # named as a sensitivity analysis, not as the sample
    if chain.sample() != "adni34":
        return f"the sentence names ADNI-3/4 as the sample; this chain's primary sample is {chain.sample_label()}"
    return None


def t_floor(sentence, ev):
    if S.density_floor() != S.DEFAULT_FLOOR:
        return f"the sentence states a density floor of 0.6; this chain's floor is {S.density_floor():g}"
    return None


def t_sessions(sentence, ev):
    n = ev.n_sessions()
    if n is None:
        return NO_EVIDENCE + " (covariates)"
    if n != 530:
        return f"the sentence states the 530-session cohort (251/201/78); this chain has {n} sessions"
    return None


def t_recipe(sentence, ev):
    rec = ev.recipe()
    if rec is None:
        return None if chain.root() is None else "this chain's connectomes record no recipe"
    why = []
    select = rec.get("select")
    if re.search(r"Three million", sentence) and select != 3000000:
        why.append(f"the sentence says three million streamlines; the recipe recorded {select}")
    name = str(rec.get("recipe", "")).lower()
    is_act = bool(rec.get("act")) or ("act" in name and "noact" not in name and "no_act" not in name)
    if is_act and not re.search(r"anatomically constrained", sentence):
        why.append(f"the sentence describes tracking without anatomical constraints; the connectomes come from recipe {rec.get('recipe')}")
    return "; ".join(why) or None


def t_nodes(sentence, ev):
    k = S.n_nodes()
    if k != S.DEFAULTS["n_nodes"]:
        return f"the sentence names the 166 regions or ten networks; this chain analyses {k} nodes"
    return None


def t_nan(sentence, ev):
    return "a number that could not be estimated is printed as nan"


def t_v3_counts(sentence, ev):
    """Counts typed into the sentences for the 25 Oct rebuild; a chain of its own has its own."""
    if chain.root() is None:
        return None
    if re.search(r"\b519 participants\b", sentence) and re.search(r"\bbefore\b[^.]{0,60}\b(?:hold|held|SMC)", sentence):
        n = ev.n_before_holdout()
        if n is None:
            return NO_EVIDENCE + " (Phase 1 QC)"
        return None if n == 519 else f"the sentence states 519 before the hold-out; this chain's QC gives {n}"
    return "a count typed for the 25 Oct rebuild; this chain must compute its own"


CLAIMS = [
    ("no crossover", re.compile(r"\bno crossover\b|\bnot support (?:a|the)\b[^.]{0,60}crossover|crossover[^.]{0,40}\b(?:not supported|not seen|not found|not observed|absent)\b|\b(?:no evidence|no sign|no indication|absence|lack) (?:of|for) (?:a |any )?crossover\b|\bwithout (?:a |any )?crossover\b|\bNo analysis showed larger early effects\b", re.I), t_no_crossover),
    ("at every stage", re.compile(r"\bevery stage\b|\b(?:at|across) (?:all|each|every)(?: three)? (?:disease )?stages?\b", re.I), t_every_stage),
    ("13 of 14", re.compile(r"\b\d+ of \d+ leading group effects\b|\b13 of 14\b"), t_n_of_m),
    ("agreed", re.compile(r"\bagree(?:d|s)?\b|\b(?:in|good|close|strong|high) agreement\b", re.I), t_agreed),
    ("stable", re.compile(r"\bstable\b"), t_stable),
    ("near-perfect", re.compile(r"\bnear-perfect\b"), t_near_perfect),
    ("robust to", re.compile(r"\brobust to\b|\bsimilarly robust\b|\bmost robust\b"), t_robust),
    ("every network", re.compile(r"\bin every network\b"), t_every_network),
    ("every connection length", re.compile(r"\bat every (?:connection )?length\b"), t_every_length),
    ("primary sample", re.compile(r"ADNI-3 and ADNI-4|ADNI-3/4"), t_sample),
    ("density floor", re.compile(r"density (?:was )?below 0\.6"), t_floor),
    ("530 sessions", re.compile(r"\b(?:530 (?:participants|sessions|subjects|ADNI)|251 CN|78 with AD)\b"), t_sessions),
    ("recipe", re.compile(r"Three million probabilistic streamlines|probabilistic streamlines were generated"), t_recipe),
    ("nodes", re.compile(r"\b166 regions\b|\bten (?:functional )?networks\b|\b10 networks\b"), t_nodes),
    ("not estimable", re.compile(r"(?<![A-Za-z])nan(?![A-Za-z])"), t_nan),
    ("typed counts", re.compile(r"\b(?:eight|two) participants\b|\b55 participants\b|\bnone occurred\b|\b519 participants\b|\b10 failed reconstructions\b"), t_v3_counts),
]


def live_text(doc) -> list[str]:
    """Paragraph texts of a python-docx document, tables included, without struck runs."""
    out = []

    def add(p):
        t = "".join(r.text for r in p.runs if not r.font.strike)
        if t.strip():
            out.append(t)

    for p in doc.paragraphs:
        add(p)
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    add(p)
    return out


def sentences(paragraphs) -> list[str]:
    seen, out = set(), []
    for para in paragraphs:
        for s in re.split(r"(?<=[.!?])\s+", para):
            s = re.sub(r"\s+", " ", s).strip()
            if s and s not in seen:
                seen.add(s); out.append(s)
    return out


def findings_for(paragraphs) -> list[dict]:
    ev = Evidence(); out = []
    for s in sentences(paragraphs):
        for label, pat, test in CLAIMS:
            if pat.search(s):
                why = test(s, ev)
                if why:
                    out.append(dict(claim=label, sentence=s, why=why))
    return out


def report_path(name: str) -> Path:
    return chain.docs_dir() / ".guard" / f"{name}.json"


def check_document(doc, name: str) -> list[dict]:
    """Stop with a clear error when the document says what the numbers contradict.

    Only a rehearsal may ask for IIT_NARRATIVE_GUARD=record, which lists the contradictions in docs/.guard/ and carries on, so
    that the rest of the builder can be exercised on data whose numbers are meant to be meaningless.
    """
    found = findings_for(live_text(doc))
    if chain.root() is not None:
        file = report_path(name); file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(dict(document=name, contradictions=found), indent=1, ensure_ascii=False) + "\n")
    if not found:
        print(f"narrative guard: {name}: no fixed sentence contradicted")
        return found
    lines = [f"NARRATIVE GUARD: {name}: {len(found)} fixed sentence(s) contradicted by the computed numbers. Nothing was saved."]
    for f in found[:40]:
        lines.append(f"  [{f['claim']}] \"{f['sentence'][:160]}\"\n      {f['why']}")
    if os.environ.get("IIT_NARRATIVE_GUARD") == "record":
        if not chain.rehearsal():
            raise SystemExit("IIT_NARRATIVE_GUARD=record is allowed only in a rehearsal.")
        print(f"narrative guard: {name}: {len(found)} contradiction(s) recorded (rehearsal), see {report_path(name)}")
        return found
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(chain.EXIT_REFUSED)


def main(files) -> int:
    from docx import Document

    chain.boot()
    for f in files:
        found = findings_for(live_text(Document(f)))
        print(f"== {Path(f).name}: {len(found)} finding(s)")
        for x in found:
            print(f"  [{x['claim']}] {x['sentence'][:170]}\n      {x['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
