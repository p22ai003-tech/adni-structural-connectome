"""Render a .docx to text preserving tracked changes, comments, tables, headings.
   [+text+] = insertion   [-text-] = deletion   {{C#n: comment}} anchors   tables as | rows |"""
import sys, zipfile, re
from lxml import etree
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
def q(t): return "{%s}%s" % (W, t)

def load(z, name):
    try: return etree.fromstring(z.read(name))
    except KeyError: return None

def run_text(r, mode):
    out = []
    for ch in r:
        tag = etree.QName(ch).localname
        if tag in ("t", "delText"): out.append(ch.text or "")
        elif tag == "tab": out.append("\t")
        elif tag in ("br", "cr"): out.append("\n")
        elif tag == "noBreakHyphen": out.append("-")
    s = "".join(out)
    if not s: return ""
    return {"ins": "[+%s+]", "del": "[-%s-]"}.get(mode, "%s") % s

def para_text(p, comments_seen):
    parts = []
    def walk(node, mode=None):
        for ch in node:
            tag = etree.QName(ch).localname
            if tag == "r": parts.append(run_text(ch, mode))
            elif tag == "ins": walk(ch, "ins")
            elif tag == "del": walk(ch, "del")
            elif tag == "commentRangeStart":
                cid = ch.get(q("id")); parts.append("⟦C%s:" % cid)
            elif tag == "commentRangeEnd":
                cid = ch.get(q("id")); parts.append("⟧")
            elif tag in ("hyperlink", "smartTag", "sdt", "sdtContent", "fldSimple", "customXml"):
                walk(ch, mode)
            elif tag == "oMath" or tag == "oMathPara":
                parts.append("".join(ch.itertext()))
    walk(p)
    s = "".join(parts)
    s = re.sub(r"\[\+\+\]|\[--\]", "", s)
    s = re.sub(r"\+\]\[\+", "", s); s = re.sub(r"-\]\[-", "", s)   # merge adjacent marks
    return s

def style_of(p, styles):
    ps = p.find("w:pPr/w:pStyle", NS)
    sid = ps.get(q("val")) if ps is not None else None
    return styles.get(sid, sid or "")

def main(path):
    z = zipfile.ZipFile(path)
    doc = load(z, "word/document.xml")
    sty = load(z, "word/styles.xml"); styles = {}
    if sty is not None:
        for s in sty.findall("w:style", NS):
            n = s.find("w:name", NS)
            styles[s.get(q("styleId"))] = n.get(q("val")) if n is not None else s.get(q("styleId"))
    cm = load(z, "word/comments.xml"); comments = {}
    if cm is not None:
        for c in cm.findall("w:comment", NS):
            txt = "\n".join(para_text(p, None) for p in c.findall(".//w:p", NS)).strip()
            comments[c.get(q("id"))] = (c.get(q("author")), c.get(q("date")), txt)
    body = doc.find("w:body", NS)
    lines = []; tcount = 0
    for el in body:
        tag = etree.QName(el).localname
        if tag == "p":
            st = style_of(el, styles); t = para_text(el, None).rstrip()
            if not t.strip():
                if el.findall(".//w:drawing", NS) or el.findall(".//w:pict", NS): lines.append("[FIGURE/IMAGE]")
                continue
            if st.lower().startswith("heading") or st.lower() == "title":
                lines.append("\n" + "#" * (int(re.sub(r"\D", "", st) or 1)) + " " + t)
            else:
                lines.append(t)
        elif tag == "tbl":
            tcount += 1; lines.append("\n[TABLE %d]" % tcount)
            for tr in el.findall("w:tr", NS):
                cells = [" ".join(para_text(p, None).strip() for p in tc.findall(".//w:p", NS)).strip()
                         for tc in tr.findall("w:tc", NS)]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("[/TABLE %d]\n" % tcount)
    print("\n".join(lines))
    if comments:
        print("\n\n==================== COMMENTS (%d) ====================" % len(comments))
        for cid, (a, d, t) in comments.items():
            print("C%s [%s, %s]: %s" % (cid, a, (d or "")[:10], t))
    # tracked-change stats
    ins = len(doc.findall(".//w:ins", NS)); dele = len(doc.findall(".//w:del", NS))
    authors = sorted(set(e.get(q("author")) for e in doc.iter() if e.get(q("author"))))
    print("\n[STATS] insertions=%d deletions=%d comments=%d tables=%d change-authors=%s" % (ins, dele, len(comments), tcount, authors))

if __name__ == "__main__":
    main(sys.argv[1])
