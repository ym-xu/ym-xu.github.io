#!/usr/bin/env python3
"""Fetch recent document-understanding papers from arXiv and write data/survey.json.

Runs in the Pages workflow before `hugo` (daily on a schedule), and can be run
by hand:  python3 scripts/fetch_survey.py

Standard library only. How a paper is filed:
  * every subfield rule is counted in the title (x3) and the abstract (x1);
  * a subfield is tagged when its score >= 2 — a title hit, or at least two
    mentions in the abstract — so a single passing "we use OCR" doesn't count;
  * the highest-scoring subfield is the paper's *primary* one;
  * the paper must also look visual (pages, scans, layout, tables, charts …),
    which keeps out text-only "document" NLP.
If arXiv can't be reached, the existing data file is left untouched, so the
site still builds with the last good snapshot.

Thank you to arXiv for use of its open access interoperability.
"""
import collections
import datetime as dt
import json
import os
import re
import sys
import time
import tomllib
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "survey.json")
CONFIG = os.path.join(HERE, "..", "config", "_default", "config.toml")
WEEKS = 26                # history shown in the trend charts
LIST_DAYS = WEEKS * 7     # papers listed on the page (shown 30 at a time) — same window as the charts, so counts match
PAGE = 400                # results per API call
MAX_RESULTS = 3200        # hard cap per run
UA = "living-survey-fetch/2.0 (static site build)"

CATS = ["cs.CV", "cs.CL", "cs.IR", "cs.AI", "cs.LG", "cs.DL", "cs.MM"]
TERMS = [
    'ti:document', 'ti:documents', 'abs:"document understanding"', 'abs:"document image"',
    'abs:"document analysis"', 'abs:"visually rich"', 'abs:"visually-rich"', 'abs:"visual document"',
    'abs:"visual documents"', 'abs:"multimodal document"', 'abs:"text-rich"', 'abs:DocVQA',
    'abs:"document question answering"', 'abs:"document VQA"', 'abs:"OCR-free"', 'ti:OCR',
    'abs:"optical character recognition"', 'abs:"layout analysis"', 'abs:"document layout"',
    'abs:"reading order"', 'abs:"table recognition"', 'abs:"table structure"', 'abs:"chart understanding"',
    'abs:"chart question answering"', 'abs:ChartQA', 'ti:chart', 'ti:charts', 'ti:infographic',
    'ti:handwritten', 'abs:"handwritten text"', 'abs:"key information extraction"', 'abs:"form understanding"',
    'abs:"document parsing"', 'abs:"PDF parsing"', 'ti:PDF', 'ti:PDFs', 'abs:"multi-page"', 'abs:"scanned"',
    'abs:"scene text"', 'abs:"text recognition"', 'abs:"historical documents"', 'abs:"math formula recognition"',
]

# (id, name, one-line scope, regex). Order breaks ties for the primary subfield.
SUBFIELDS = [
    ("ocr", "OCR & Text Recognition", "Reading the characters: printed, scene and multilingual text recognition and spotting.",
     r"\bocr\b|optical character|text recognition|text spotting|scene text|character recognition|text line recognition"),
    ("docvqa", "Document QA & VLMs", "Vision-language models and question answering over document images.",
     r"document (visual )?question answering|docvqa|document vqa|visually[- ]rich|document (vlm|mllm|lvlm|understanding model)s?|document understanding|text[- ]rich (image|document)s?|text[- ]centric|document intelligence|document ai\b|ocr[- ]free"),
    ("rag", "Retrieval & RAG", "Finding evidence in documents: visual retrieval, multimodal RAG, page search.",
     r"\bRAG\b|retrieval[- ]augmented|document (retrieval|search)|page retrieval|visual(ly)? (rich )?document (retrieval|search)|colpali|multimodal retriev\w*|retriev\w* (over|from|across) (long |multiple )?(documents|pages|pdfs)"),
    ("agents", "Agents & Reasoning", "Agents that read, navigate and reason across documents.",
     r"(document|page|pdf|reading)[- ](centric |understanding |qa )?agents?\b|agents? (for|over|on|that)\b.{0,60}\b(documents?|pages|pdfs?)\b|agentic (document|rag|retrieval|search|parsing|reading)|deep research|navigat\w+ (the |a |long )?(document|page)s?|reasoning (over|across) (long |multiple )?documents"),
    ("longdoc", "Long & Multi-page Documents", "Documents beyond a single page: long PDFs, slide decks, collections.",
     r"long[- ]documents?|multi[- ]page|multipage|long[- ]context document|book[- ]length|slide decks?|\bslides\b|lengthy documents|document collections?"),
    ("layout", "Layout & Reading Order", "Page structure: layout detection, reading order, logical structure.",
     r"layout (analysis|detection|understanding|parsing|model)|page (object )?segmentation|reading order|document structure|logical structure|document layout|heading hierarch\w+"),
    ("parsing", "Parsing & Conversion", "Turning pages into structured text: PDF-to-Markdown, formulas, end-to-end parsers.",
     r"document parsing|pdf (parsing|extraction|to)|document conversion|to markdown|\bmarkdown\b|latex (generation|conversion)|formula recognition|math(ematical)? expression recognition|end-to-end (document )?pars\w+|page[- ]to[- ](token|text|markdown)"),
    ("tables", "Tables", "Table detection, structure recognition and table question answering.",
     r"table (recognition|structure|detection|extraction|understanding|question|parsing|reasoning)|tabular images?|table images?|image tables?"),
    ("charts", "Charts, Figures & Infographics", "Understanding and generating charts, plots, scientific figures and infographics.",
     r"chartqa|\bcharts? (understanding|question|reasoning|qa|comprehension|summari\w*|generation|parsing|derendering|caption\w*|images?|data|benchmark|to[- ]code|editing)|(understand|reason|answer)\w* (over |about |on )?(scientific )?charts|infographics?|plot (understanding|question)|figure (understanding|question|captioning)|scientific figures?|diagram (understanding|question)"),
    ("kie", "Information Extraction & Forms", "Key information extraction from forms, invoices, receipts and records.",
     r"key information extraction|information extraction from (visually|documents|scanned)|form (understanding|parsing)|\binvoices?\b|\breceipts?\b|document (information |entity )?extraction"),
    ("historical", "Handwriting & Historical", "Handwritten text, manuscripts, archives and historical newspapers.",
     r"handwrit\w+|historical (documents?|manuscripts?|newspapers?)|manuscripts?|archival|\bhtr\b|palaeograph\w*|paleograph\w*|palimpsest"),
    ("imaging", "Imaging & Generation", "Document image processing and synthesis: dewarping, binarisation, layout and text generation.",
     r"dewarp\w*|binari[sz]ation|document (image )?(enhancement|restoration|rectification|deblur\w*)|layout generation|document generation|text rendering|visual text generation|forgery|tamper\w*"),
]
VISUAL = re.compile(
    r"document images?|visually[- ]rich|visual(ly)? (rich )?documents?|multimodal documents?|\bocr\b|optical character|scann\w+"
    r"|\blayouts?\b|page images?|document pages?|\bpdfs?\b|\btables?\b|\bcharts?\b|infographic|handwrit|manuscript|\bforms?\b"
    r"|invoice|receipt|scene text|text (recognition|spotting|detection)|\bslides?\b|text[- ]rich|docvqa|screenshots?|scientific figures?"
    r"|document (vlm|mllm|lvlm|understanding|parsing|analysis|intelligence)|historical (documents?|newspapers?)", re.I)
# Subfields where text-only work slips in (RAG over text "documents", research
# agents, long text): their papers need a *strong* visual cue as well.
NEEDS_STRONG = {"rag", "agents", "longdoc", "kie"}
STRONG_VISUAL = re.compile(
    r"document images?|visually[- ]rich|visual(ly)? (rich )?documents?|multimodal|page images?|screenshots?|\bocr\b|scann\w+"
    r"|\blayouts?\b|\bcharts?\b|infographic|vision[- ]language|\bvlms?\b|\bmllms?\b|\blvlms?\b|\bimages?\b|\bvisual\b|figures?|\bslides?\b|\bpixels?\b", re.I)

# Where a paper was accepted, from arXiv's author comment / journal ref.
VENUES = ["CVPR", "ICCV", "ECCV", "NeurIPS", "ICLR", "ICML", "ACL", "EMNLP", "NAACL", "COLING", "EACL", "AAAI", "IJCAI",
          "ICDAR", "WACV", "BMVC", "ACM MM", "KDD", "SIGIR", "WWW", "CIKM", "WSDM", "ICASSP", "ICPR", "DAS", "LREC",
          "TPAMI", "IJCV", "TIP", "TACL", "Pattern Recognition", "ICME", "ACCV", "MICCAI", "JCDL", "TPDL"]
VENUE_RX = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(v).replace(r"\ ", r"\s*") for v in VENUES) + r"|NIPS|ACMMM)(?![A-Za-z])"
                      r"(?:\s*(?:'|’)?\s*(20\d\d|\d\d))?", re.I)
ACCEPT_RX = re.compile(r"accept|to appear|published|camera[- ]ready|proceedings|main conference|findings|oral|spotlight|poster|workshop|\bat\b|\bin\b", re.I)
UNDER_REVIEW = re.compile(r"submitted|under review|in submission|under submission|preprint|extended version of", re.I)
CODE_RX = re.compile(r"https?://(?:www\.)?(?:github\.com|gitlab\.com|huggingface\.co)/[\w.\-/]+[\w/]", re.I)

NOT_FIELD = re.compile(r"channel charting|flow ?chart(ing)? of (the )?(study|patients)", re.I)
BENCH_TITLE = re.compile(r"\b(benchmark|dataset|corpus)s?\b|\bbench\b", re.I)
BENCH_ABS = re.compile(r"\b(introduce|present|release|construct|propose|curate|build|collect)\w*\s+(a|an|the)?\s*((new|novel|large[- ]scale|first|comprehensive|challenging|multilingual|synthetic)\s+)*([\w-]+\s+){0,3}(benchmark|dataset|corpus)\b", re.I)
COMPILED = [(i, n, desc, re.compile(p, re.I)) for i, n, desc, p in SUBFIELDS]
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# words that say nothing about a trend (for "rising terms")
STOP = set("""a an and are as at be by can for from has have how in into is it its of on or our over than that the their this
through to toward towards under using via vs we what when where which while with without beyond across based new novel
paper approach method methods framework model models large language vision visual multimodal multi modal document documents
understanding learning efficient towards study analysis data task tasks benchmark benchmarking dataset system systems toward
llm llms mllm mllms vlm vlms lvlm lvlms evaluating evaluation improving improved enhancing robust unified simple
not no one two more all any only just do does you your why visually rich text image images aware driven guided""".split())


def config_watchlist():
    try:
        with open(CONFIG, "rb") as f:
            cfg = tomllib.load(f)
        return [a for a in cfg.get("params", {}).get("survey", {}).get("watch", []) if a]
    except Exception:
        return []


def query():
    return "({}) AND ({})".format(" OR ".join(f"cat:{c}" for c in CATS), " OR ".join(TERMS))


def fetch(start):
    params = urllib.parse.urlencode({"search_query": query(), "start": start, "max_results": PAGE,
                                     "sortBy": "submittedDate", "sortOrder": "descending"})
    req = urllib.request.Request("https://export.arxiv.org/api/query?" + params, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return ET.fromstring(r.read())
        except Exception as e:
            print(f"  arXiv request failed ({e}); retry {attempt + 1}/3", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("arXiv unreachable")


def clean(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def score(title, abstract):
    """{subfield id: score}; title hits weigh 3x."""
    out = {}
    for sid, _, _, rx in COMPILED:
        s = 3 * sum(1 for _ in rx.finditer(title)) + sum(1 for _ in rx.finditer(abstract))
        if s >= 2:
            out[sid] = s
    return out


def venue_of(comment, jref):
    """'CVPR 2026' / 'EMNLP 2025 Findings' / 'ICDAR 2026 Workshop', or ''."""
    for text in (jref, comment):
        if not text:
            continue
        m = VENUE_RX.search(text)
        if not m or (text is comment and (not ACCEPT_RX.search(text)
                                           or (UNDER_REVIEW.search(text) and not re.search(r"accept", text, re.I)))):
            continue
        name = m.group(1)
        canon = next((v for v in VENUES if re.fullmatch(re.escape(v).replace(r"\ ", r"\s*"), name, re.I)), name)
        canon = {"NIPS": "NeurIPS", "ACMMM": "ACM MM"}.get(canon.upper(), canon)
        year = m.group(2) or ""
        if len(year) == 2:
            year = "20" + year
        extra = " Findings" if re.search(r"findings", text, re.I) else (" Workshop" if re.search(r"workshop", text, re.I) else "")
        return f"{canon} {year}".strip() + extra
    return ""


def terms_of(title):
    words = [w for w in re.findall(r"[a-z][a-z0-9-]+", title.lower()) if len(w) > 2]
    uni = {w for w in words if w not in STOP}
    bi = {f"{a} {b}" for a, b in zip(words, words[1:]) if a not in STOP and b not in STOP}
    return uni | bi


def main():
    today = dt.datetime.now(dt.timezone.utc).date()
    since = today - dt.timedelta(days=WEEKS * 7 - 1)
    watch = config_watchlist()
    watch_rx = re.compile("|".join(re.escape(a) for a in watch), re.I) if watch else None

    papers, seen, start = [], set(), 0
    while start < MAX_RESULTS:
        root = fetch(start)
        entries = root.findall("a:entry", NS)
        if not entries:
            break
        oldest = None
        for e in entries:
            base = re.sub(r"v\d+$", "", clean(e.findtext("a:id", "", NS)).rsplit("/", 1)[-1])
            if not base or base in seen:
                continue
            seen.add(base)
            pub = dt.date.fromisoformat(clean(e.findtext("a:published", "", NS))[:10])
            oldest = pub if oldest is None or pub < oldest else oldest
            if pub < since:
                continue
            title = clean(e.findtext("a:title", "", NS))
            abstract = clean(e.findtext("a:summary", "", NS))
            sc = score(title, abstract)
            if not sc or not VISUAL.search(f"{title} {abstract}") or NOT_FIELD.search(f"{title} {abstract}"):
                continue
            order = [sid for sid, *_ in COMPILED]
            tags = sorted(sc, key=lambda k: (-sc[k], order.index(k)))
            if tags[0] in NEEDS_STRONG and not STRONG_VISUAL.search(f"{title} {abstract}"):
                continue  # text-only RAG / agents / long text — not visual document understanding
            prim = e.find("arxiv:primary_category", NS)
            authors = [clean(a.findtext("a:name", "", NS)) for a in e.findall("a:author", NS)]
            comment = clean(e.findtext("arxiv:comment", "", NS))
            jref = clean(e.findtext("arxiv:journal_ref", "", NS))
            code = CODE_RX.search(f"{abstract} {comment}")
            papers.append({
                "id": base, "title": title,
                "authors": authors[:4] + (["et al."] if len(authors) > 4 else []),
                "_all_authors": authors,           # used for counts, not written out
                "date": pub.isoformat(),
                "cat": prim.get("term") if prim is not None else "",
                "tags": tags,                      # tags[0] is the primary subfield
                "bench": bool(BENCH_TITLE.search(title) or BENCH_ABS.search(abstract)),
                "watch": bool(watch_rx and any(watch_rx.fullmatch(a) for a in authors)),
                "venue": venue_of(comment, jref),
                "code": code.group(0).rstrip(".,);") if code else "",
                "abs": abstract[:320] + ("…" if len(abstract) > 320 else ""),
            })
        print(f"  fetched {start}–{start + len(entries)}; oldest {oldest}", file=sys.stderr)
        if oldest and oldest < since:
            break
        start += PAGE
        time.sleep(3.5)  # arXiv asks for ≥3 s between calls

    papers.sort(key=lambda p: (p["date"], p["id"]), reverse=True)

    # ---- weekly buckets, oldest first; the last bucket ends today ----
    def week_of(date_iso):
        return (today - dt.date.fromisoformat(date_iso)).days // 7   # 0 = this week
    def weekly(ps):
        w = [0] * WEEKS
        for p in ps:
            k = week_of(p["date"])
            if 0 <= k < WEEKS:
                w[WEEKS - 1 - k] += 1
        return w
    week_starts = [(today - dt.timedelta(days=7 * (WEEKS - 1 - i) + 6)).isoformat() for i in range(WEEKS)]

    def growth(ps):
        now = sum(1 for p in ps if week_of(p["date"]) < 4)
        before = sum(1 for p in ps if 4 <= week_of(p["date"]) < 8)
        return now, before, (None if min(now, before) < 4 else round(100 * (now - before) / before))

    total = len(papers) or 1
    subfields = []
    for sid, name, desc, _ in COMPILED:
        mine = [p for p in papers if sid in p["tags"]]
        primary = [p for p in mine if p["tags"][0] == sid]
        now, before, g = growth(mine)
        subfields.append({
            "id": sid, "name": name, "desc": desc,
            "count": len(mine), "primary": len(primary),
            "share": round(100 * len(primary) / total, 1),     # primary shares sum to 100%
            "last4w": now, "prev4w": before, "trend": g,
            "weekly": weekly(mine),
            "latest": [{"id": p["id"], "title": p["title"]} for p in mine[:2]],
        })
    subfields.sort(key=lambda s: (-s["primary"], -s["count"]))
    names = {sid: name for sid, name, *_ in COMPILED}
    for p in papers:
        p["tagnames"] = [names[t] for t in p["tags"]]

    # ---- who and where: most active authors, accepted venues ----
    au = collections.Counter(a for p in papers for a in p["_all_authors"])
    active = [{"name": a, "n": n} for a, n in au.most_common(40) if n >= 3][:12]
    ven = collections.Counter(re.sub(r"\s+(20\d\d)", "", p["venue"]).replace(" Findings", "").replace(" Workshop", "")
                              for p in papers if p["venue"])
    venues = [{"name": v, "n": n} for v, n in ven.most_common(10)]
    for p in papers:
        del p["_all_authors"]

    # ---- where subfields meet: most common pairs ----
    pairs = collections.Counter()
    for p in papers:
        t = sorted(p["tags"])
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                pairs[(t[i], t[j])] += 1
    top_pairs = [{"a": names[a], "b": names[b], "ida": a, "idb": b, "n": n} for (a, b), n in pairs.most_common(8)]

    # ---- rising terms: title words/bigrams, last 4 weeks vs the 22 before ----
    recent = [p for p in papers if week_of(p["date"]) < 4]
    base = [p for p in papers if week_of(p["date"]) >= 4]
    rc, bc = collections.Counter(), collections.Counter()
    for p in recent:
        rc.update(terms_of(p["title"]))
    for p in base:
        bc.update(terms_of(p["title"]))
    nr, nb = max(len(recent), 1), max(len(base), 1)
    rising = []
    for term, c in rc.items():
        if c < 3:
            continue
        lift = (c / nr) / ((bc[term] + 1) / nb)
        rising.append((lift, c, term))
    rising.sort(reverse=True)
    picked, words_used = [], set()
    for lift, c, term in rising:
        parts = set(term.split())
        if parts & words_used and " " not in term:   # skip a word already shown inside a bigram
            continue
        picked.append({"term": term, "n": c, "lift": round(lift, 1)})
        words_used |= parts
        if len(picked) == 10:
            break

    benches = [{"id": p["id"], "title": p["title"], "date": p["date"]} for p in papers if p["bench"]][:6]
    list_from = (today - dt.timedelta(days=LIST_DAYS - 1)).isoformat()
    out = {
        "updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "weeks": WEEKS, "since": since.isoformat(), "today": today.isoformat(),
        "total": len(papers),
        "last4w": sum(1 for p in papers if week_of(p["date"]) < 4),
        "this_week": sum(1 for p in papers if week_of(p["date"]) == 0),
        "benchmarks": sum(1 for p in papers if p["bench"]),
        "watch": watch, "watched": sum(1 for p in papers if p["watch"]),
        "week_starts": week_starts,
        "weekly": weekly(papers),
        "max_weekly_total": max(weekly(papers) or [0]),
        "max_weekly": max([max(s["weekly"]) for s in subfields] or [0]),
        "max_count": max([s["count"] for s in subfields] or [0]),
        "subfields": subfields,
        "pairs": top_pairs,
        "active_authors": active,
        "venues": venues,
        "accepted": sum(1 for p in papers if p["venue"]),
        "with_code": sum(1 for p in papers if p["code"]),
        "rising": picked,
        "new_benchmarks": benches,
        "list_days": LIST_DAYS,
        "listed_total": sum(1 for p in papers if p["date"] >= list_from),
        "papers": [p for p in papers if p["date"] >= list_from],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {len(out['papers'])} listed / {len(papers)} total papers ({WEEKS} weeks) to {os.path.relpath(OUT)}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"survey fetch failed, keeping existing data: {exc}", file=sys.stderr)
        sys.exit(0)
