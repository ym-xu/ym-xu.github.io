#!/usr/bin/env python3
"""Fetch recent document-understanding papers from arXiv and write data/survey.json.

Runs in the Pages workflow before `hugo` (daily on a schedule), and can be run
by hand:  python3 scripts/fetch_survey.py

Only the standard library is used. Papers are put into subfields by the keyword
rules in SUBFIELDS below — transparent and easy to tune; a paper can sit in
several subfields, and its first match (in list order) is its primary one.
If arXiv cannot be reached, the existing data file is left untouched, so the
site still builds with the last good snapshot.

Thank you to arXiv for use of its open access interoperability.
"""
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "survey.json")
WINDOW_DAYS = 90          # how far back the survey looks
LIST_DAYS = 90            # how many days of papers the page lists (the page shows 30 at a time)
PAGE = 400                # results per API call (arXiv asks for modest pages)
MAX_RESULTS = 1600        # hard cap per run
UA = "living-survey-fetch/1.0 (static site build; contact via repository)"

CATS = ["cs.CV", "cs.CL", "cs.IR", "cs.AI", "cs.LG", "cs.DL", "cs.MM"]
TERMS = [
    'ti:document', 'ti:documents', 'abs:"document understanding"', 'abs:"document image"',
    'abs:"document analysis"', 'abs:"visually rich"', 'abs:"visually-rich"', 'abs:DocVQA',
    'abs:"document question answering"', 'ti:OCR', 'abs:"optical character recognition"',
    'abs:"layout analysis"', 'abs:"table recognition"', 'abs:"table structure"',
    'abs:"chart understanding"', 'abs:"chart question answering"', 'ti:chart', 'ti:charts',
    'ti:infographic', 'ti:handwritten', 'abs:"handwritten text"', 'abs:"key information extraction"',
    'abs:"form understanding"', 'abs:"document parsing"', 'abs:"PDF parsing"', 'ti:PDF',
    'abs:"multi-page"', 'abs:"scene text"', 'abs:"text recognition"', 'abs:"historical documents"',
]

# (id, name, regex). Order = priority for the primary subfield.
SUBFIELDS = [
    ("rag",       "Retrieval & RAG",                r"\bRAG\b|retrieval[- ]augmented|document (retrieval|search)|page retrieval|visual(ly)? (rich )?document (retrieval|search)|colpali|multimodal retriev|retriev\w* (over|from|across) (long |multiple )?(documents|pages|pdfs)"),
    ("agents",    "Agents & Reasoning",             r"(document|page|pdf|reading)[- ](centric |understanding |qa )?agents?\b|agents? (for|over|on|that)\b.{0,60}\b(documents?|pages|pdfs?)\b|agentic (document|rag|retrieval|search|parsing|reading|framework for document)|\bdocuments?\b.{0,80}\bagent(s|ic)?\b|deep research|navigat\w+ (the |a |long )?(document|page)|reasoning (over|across) (long |multiple )?documents"),
    ("longdoc",   "Long & Multi-page Documents",    r"long[- ]document|multi[- ]page|multipage|long[- ]context document|book[- ]length|slide decks?|\bslides\b|lengthy documents|document collections?"),
    ("docvqa",    "Document QA & VLMs",             r"document (visual )?question answering|docvqa|visually[- ]rich|document (vlm|mllm|lvlm|understanding model)|document understanding|text[- ]rich image|text[- ]centric|document intelligence|document ai\b"),
    ("layout",    "Layout & Reading Order",         r"layout (analysis|detection|understanding|parsing)|page (object )?segmentation|reading order|document structure|logical structure|region detection|document layout"),
    ("parsing",   "Parsing & Conversion",           r"document parsing|pdf (parsing|extraction|to)|document conversion|to markdown|\bmarkdown\b|latex (generation|conversion)|formula recognition|math(ematical)? expression recognition|end-to-end (document )?pars"),
    ("ocr",       "OCR & Text Recognition",         r"\bocr\b|optical character|text recognition|text spotting|scene text|text detection|character recognition"),
    ("tables",    "Tables",                         r"table (recognition|structure|detection|extraction|understanding|question|parsing|reasoning)|tabular image|table images?"),
    ("charts",    "Charts, Figures & Infographics", r"chartqa|\bcharts? (understanding|question|reasoning|qa|comprehension|summari\w*|generation|parsing|derendering|caption\w*|images?|data|benchmark|to[- ]code|editing)|(understand|reason|answer)\w* (over |about |on )?(scientific )?charts|infographic|plot (understanding|question)|figure (understanding|question|captioning)|scientific figures?|diagram (understanding|question)"),
    ("kie",       "Information Extraction & Forms", r"key information extraction|information extraction from (visually|documents|scanned)|form (understanding|parsing)|\binvoices?\b|\breceipts?\b|document (information |entity )?extraction|named entit\w+ .*document"),
    ("historical","Handwriting & Historical",       r"handwrit|historical (documents?|manuscripts?|newspapers?)|manuscripts?|archival|\bhtr\b|palaeograph|paleograph"),
    ("imaging",   "Imaging & Generation",           r"dewarp|binari[sz]ation|document (image )?(enhancement|restoration|rectification|deblur)|layout generation|document generation|text rendering|visual text generation|forgery|tamper"),
]
# A paper must also look *visual* — documents as images/pages, not just text —
# so text-only RAG or "documentation" papers that happen to match stay out.
VISUAL = re.compile(
    r"document images?|visually[- ]rich|visual(ly)? (rich )?documents?|multimodal documents?|\bocr\b|optical character|scann\w+"
    r"|\blayouts?\b|page images?|document pages?|\bpdfs?\b|\btables?\b|\bcharts?\b|infographic|handwrit|manuscript|\bforms?\b"
    r"|invoice|receipt|scene text|text (recognition|spotting|detection)|\bslides?\b|text[- ]rich|docvqa|screenshots?|scientific figures?"
    r"|document (vlm|mllm|lvlm|understanding|parsing|analysis|intelligence)|historical (documents?|newspapers?)", re.I)
NOT_FIELD = re.compile(r"channel charting|flow ?chart(ing)? of (the )?(study|patients)", re.I)
BENCH_TITLE = re.compile(r"\b(benchmark|dataset|corpus)s?\b|\bbench\b", re.I)
BENCH_ABS = re.compile(r"\b(introduce|present|release|construct|propose|curate|build|collect)\w*\s+(a|an|the)?\s*((new|novel|large[- ]scale|first|comprehensive|challenging|multilingual|synthetic)\s+)*([\w-]+\s+){0,3}(benchmark|dataset|corpus)\b", re.I)
COMPILED = [(i, n, re.compile(p, re.I)) for i, n, p in SUBFIELDS]
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def query():
    cats = " OR ".join(f"cat:{c}" for c in CATS)
    terms = " OR ".join(TERMS)
    return f"({cats}) AND ({terms})"


def fetch(start):
    params = urllib.parse.urlencode({
        "search_query": query(), "start": start, "max_results": PAGE,
        "sortBy": "submittedDate", "sortOrder": "descending",
    })
    req = urllib.request.Request("https://export.arxiv.org/api/query?" + params, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return ET.fromstring(r.read())
        except Exception as e:  # network blip or 5xx: back off and retry
            print(f"  arXiv request failed ({e}); retry {attempt + 1}/3", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("arXiv unreachable")


def clean(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def classify(text):
    return [i for i, _, rx in COMPILED if rx.search(text)]


def main():
    today = dt.datetime.now(dt.timezone.utc).date()
    since = today - dt.timedelta(days=WINDOW_DAYS - 1)
    papers, seen, start = [], set(), 0
    while start < MAX_RESULTS:
        root = fetch(start)
        entries = root.findall("a:entry", NS)
        if not entries:
            break
        oldest = None
        for e in entries:
            aid = clean(e.findtext("a:id", "", NS)).rsplit("/", 1)[-1]
            base = re.sub(r"v\d+$", "", aid)
            if not base or base in seen:
                continue
            seen.add(base)
            pub = dt.date.fromisoformat(clean(e.findtext("a:published", "", NS))[:10])
            oldest = pub if oldest is None or pub < oldest else oldest
            if pub < since:
                continue
            title = clean(e.findtext("a:title", "", NS))
            abstract = clean(e.findtext("a:summary", "", NS))
            text = f"{title}. {abstract}"
            tags = classify(text)
            if not tags or not VISUAL.search(text) or NOT_FIELD.search(text):
                continue  # matched the search but isn't about (visual) documents
            prim = e.find("arxiv:primary_category", NS)
            authors = [clean(a.findtext("a:name", "", NS)) for a in e.findall("a:author", NS)]
            papers.append({
                "id": base,
                "title": title,
                "authors": authors[:4] + (["et al."] if len(authors) > 4 else []),
                "date": pub.isoformat(),
                "cat": prim.get("term") if prim is not None else "",
                "tags": tags,
                "bench": bool(BENCH_TITLE.search(title) or BENCH_ABS.search(abstract)),
                "abs": abstract[:420] + ("…" if len(abstract) > 420 else ""),
            })
        print(f"  fetched {start}–{start + len(entries)}; oldest {oldest}", file=sys.stderr)
        if oldest and oldest < since:
            break
        start += PAGE
        time.sleep(3.5)  # arXiv asks for ≥3 s between calls

    papers.sort(key=lambda p: (p["date"], p["id"]), reverse=True)
    days = [(since + dt.timedelta(days=k)).isoformat() for k in range(WINDOW_DAYS)]
    per_day = {d: 0 for d in days}
    for p in papers:
        per_day[p["date"]] = per_day.get(p["date"], 0) + 1
    def weekly(counts):
        """Sum a per-day list into 7-day buckets ending today (oldest first)."""
        out, n = [], len(counts)
        for end in range(n, 0, -7):
            out.append(sum(counts[max(0, end - 7):end]))
        return out[::-1]

    def trend(now, before):
        return None if before < 3 else round(100 * (now - before) / before)

    # growth: the last 4 weeks against the 4 weeks before (less noisy than 2 weeks)
    cut14 = (today - dt.timedelta(days=13)).isoformat()
    cut28 = (today - dt.timedelta(days=27)).isoformat()
    cut56 = (today - dt.timedelta(days=55)).isoformat()
    total = len(papers) or 1
    subfields = []
    for sid, name, _ in COMPILED:
        mine = [p for p in papers if sid in p["tags"]]
        daily = {d: 0 for d in days}
        for p in mine:
            daily[p["date"]] += 1
        recent = sum(1 for p in mine if p["date"] >= cut28)
        prior = sum(1 for p in mine if cut56 <= p["date"] < cut28)
        subfields.append({
            "id": sid, "name": name, "count": len(mine),
            "share": round(100 * len(mine) / total, 1),
            "primary": sum(1 for p in mine if p["tags"][0] == sid),
            "last28": recent, "prev28": prior, "trend": trend(recent, prior),
            "weekly": weekly([daily[d] for d in days]),
            "latest": ({"id": mine[0]["id"], "title": mine[0]["title"]} if mine else None),
        })
    subfields.sort(key=lambda s: -s["count"])
    names = {sid: name for sid, name, _ in COMPILED}
    for p in papers:
        p["tagnames"] = [names[t] for t in p["tags"]]
    list_from = (today - dt.timedelta(days=LIST_DAYS - 1)).isoformat()
    out = {
        "updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "window_days": WINDOW_DAYS,
        "list_days": LIST_DAYS,
        "since": since.isoformat(),
        "total": len(papers),
        "benchmarks": sum(1 for p in papers if p["bench"]),
        "days": days,
        "daily": [per_day[d] for d in days],
        "max_daily": max([per_day[d] for d in days] or [0]),
        "max_count": max([s["count"] for s in subfields] or [0]),
        "max_weekly": max([max(s["weekly"] or [0]) for s in subfields] or [0]),
        "last14": sum(1 for p in papers if p["date"] >= cut14),
        "subfields": subfields,
        "papers": [p for p in papers if p["date"] >= list_from],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {len(out['papers'])} listed / {len(papers)} total papers to {os.path.relpath(OUT)}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never break the site build: keep the previous snapshot.
        print(f"survey fetch failed, keeping existing data: {exc}", file=sys.stderr)
        sys.exit(0)
