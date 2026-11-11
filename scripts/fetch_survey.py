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
CURATED = os.path.join(HERE, "..", "data", "survey_curated.toml")   # your hand edits: exclude / featured / picks
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=venue,publicationVenue,year,citationCount,authors"
MIN_FOR_TREND = 10        # show 4-week change only when both windows have at least this many papers
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
     # papers are already document-gated, so plain "agent(ic)" is safe here
     r"\bagent(s|ic)?\b|multi[- ]agent|tool[- ](use|using|calling|augmented)|deep research|self[- ]evolving|\bharness(es)?\b|navigat\w+|\bplanner\b"
     r"|evidence (seeking|search|acquisition|discovery|routing|threading|selection|navigation)|multi[- ]hop|working memory|memory[- ]guided|iterative (retrieval|reasoning)"),
    ("longdoc", "Long & Multi-page Documents", "Documents beyond a single page: long PDFs, slide decks, collections.",
     r"long[- ]documents?|multi[- ]page|multipage|long[- ]context document|book[- ]length|slide decks?|presentation slides|lecture slides|\bpptx?\b|lengthy documents|document collections?|extra[- ]long"),
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
    r"|invoice|receipt|scene text|text (recognition|spotting|detection)|slide decks?|presentation slides|lecture slides|text[- ]rich|docvqa|screenshots?|scientific figures?"
    r"|document (vlm|mllm|lvlm|understanding|parsing|analysis|intelligence)|historical (documents?|newspapers?)", re.I)
# Subfields where text-only work slips in (RAG over text "documents", research
# agents, long text): their papers need a *strong* visual cue as well.
NEEDS_STRONG = {"rag", "agents", "longdoc", "kie"}
# Every paper must be *about* documents: scored like a subfield (title x3,
# abstract x1, capped) and must reach 2 to enter; the broad subfields above
# additionally need a strong visual cue, where general agent / RAG work otherwise leaks in.
DOC_CORE = re.compile(
    r"\bdocuments?\b|\bpdfs?\b|\bocr\b|optical character|document images?|text recognition|scene text|text spotting"
    r"|handwrit\w*|manuscripts?|\bcharts?\b|infographics?|chartqa|docvqa|\binvoices?\b|\breceipts?\b|slide decks?"
    r"|presentation slides|text[- ]rich|visually[- ]rich|historical (newspapers?|records|archives?)|table (recognition|structure|images?)"
    r"|forms? understanding|page[- ]level|multi[- ]page|scanned (documents?|pages?|forms?|records)"
    r"|text[- ]lines?|glyphs?|characters? recognition|logograph\w*|reading order|layout analysis|document layout"
    r"|digiti[sz]\w* (of )?(books?|documents?|archives?|dictionar\w+|records|newspapers?)|visual text (rendering|generation)|text rendering", re.I)
STRONG_VISUAL = re.compile(
    r"document images?|visually[- ]rich|visual(ly)? (rich )?documents?|multimodal|page images?|screenshots?|\bocr\b|scann\w+"
    r"|\blayouts?\b|\bcharts?\b|infographic|handwrit\w*|manuscripts?|\bpdfs?\b|vision[- ]language|\bvlms?\b|\bmllms?\b|\blvlms?\b|\bimages?\b|\bvisual\b|figures?|slide decks?|presentation slides|\bpixels?\b", re.I)

# Where a paper was accepted, from arXiv's author comment / journal ref.
VENUES = ["CVPR", "ICCV", "ECCV", "NeurIPS", "ICLR", "ICML", "ACL", "EMNLP", "NAACL", "COLING", "EACL", "AAAI", "IJCAI",
          "ICDAR", "WACV", "BMVC", "ACM MM", "KDD", "SIGIR", "WWW", "CIKM", "WSDM", "ICASSP", "ICPR", "DAS", "LREC",
          "TPAMI", "IJCV", "TIP", "TACL", "Pattern Recognition", "ICME", "ACCV", "MICCAI", "JCDL", "TPDL"]
VENUE_RX = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(v).replace(r"\ ", r"\s*") for v in VENUES) + r"|NIPS|ACMMM)(?![A-Za-z])"
                      r"(?:\s*(?:'|’)?\s*(20\d\d|\d\d))?", re.I)
ACCEPT_RX = re.compile(r"accept|to appear|published|camera[- ]ready|proceedings|main conference|findings|oral|spotlight|poster|workshop|\bat\b|\bin\b", re.I)
UNDER_REVIEW = re.compile(r"submitted|under review|in submission|under submission|preprint|extended version of", re.I)
CODE_RX = re.compile(r"https?://(?:www\.)?(?:github\.com|gitlab\.com|huggingface\.co)/[\w.\-/]+[\w/]", re.I)

NOT_FIELD = re.compile(
    r"channel charting|flow ?chart(ing)? of (the )?(study|patients)"
    r"|whole[- ]slide|histopatholog\w*|patholog\w+ (slides?|images?)|gigapixel|\bWSIs?\b|cross[- ]hospital"   # pathology "slides"
    r"|footwear|outsole|shoe ?prints?|motion sensors?|accelerometer|gyroscope|side[- ]channel|keystroke"     # forensics / sensing
    r"|robot(ic)? (learning|grasp\w*|manipulation)|everyday objects|robot learning"                         # robotics
    r"|video (text |subtitle |caption )?overlay|overlay removal|subtitle removal"                            # video editing
    r"|circuit (analysis|diagrams?|simulation)|causal topology"
    r"|navigational charts?|nautical charts?|maritime", re.I)
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


def curated():
    try:
        with open(CURATED, "rb") as f:
            c = tomllib.load(f)
    except FileNotFoundError:
        return {"exclude": set(), "include": set(), "featured": {}, "picks": {}}
    return {
        "exclude": {re.sub(r"v\d+$", "", e["id"] if isinstance(e, dict) else e) for e in c.get("exclude", [])},
        "include": {re.sub(r"v\d+$", "", e["id"] if isinstance(e, dict) else e) for e in c.get("include", [])},
        "featured": {k: list(v) for k, v in c.get("featured", {}).items()},
        "picks": {p["id"]: p.get("note", "") for p in c.get("picks", [])},
    }


def arxiv_titles(ids):
    """{id: (title, date)} for a few explicit ids (featured papers outside the window)."""
    if not ids:
        return {}
    params = urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
    req = urllib.request.Request("https://export.arxiv.org/api/query?" + params, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            root = ET.fromstring(r.read())
    except Exception as e:
        print(f"  featured lookup failed ({e})", file=sys.stderr)
        return {}
    out = {}
    for e in root.findall("a:entry", NS):
        i = re.sub(r"v\d+$", "", clean(e.findtext("a:id", "", NS)).rsplit("/", 1)[-1])
        out[i] = (clean(e.findtext("a:title", "", NS)), clean(e.findtext("a:published", "", NS))[:10])
    return out


def s2_enrich(papers):
    """Semantic Scholar, in batches: author IDs (to tell people with the same name
    apart), citation counts and published venue. Optional — on any failure the
    survey just goes without; set S2_API_KEY for a higher rate limit."""
    out = {}
    headers = {"User-Agent": UA, "Content-Type": "application/json"}
    if os.environ.get("S2_API_KEY"):
        headers["x-api-key"] = os.environ["S2_API_KEY"]
    ids = [p["id"] for p in papers]
    for k in range(0, len(ids), 400):
        chunk = ids[k:k + 400]
        body = json.dumps({"ids": [f"ARXIV:{i}" for i in chunk]}).encode()
        for attempt in range(4):
            try:
                req = urllib.request.Request(S2_URL, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=90) as r:
                    rows = json.loads(r.read())
                break
            except Exception as e:
                print(f"  Semantic Scholar failed ({e}); retry {attempt + 1}/4", file=sys.stderr)
                time.sleep(10 * (attempt + 1))
        else:
            return None
        for i, row in zip(chunk, rows):
            if row:
                out[i] = row
        time.sleep(1.5)
    return out


def s2_venue(row):
    """A short venue like 'ACL 2026' from S2's publicationVenue names, or ''."""
    pv = row.get("publicationVenue") or {}
    names = [pv.get("name") or "", row.get("venue") or ""] + list(pv.get("alternate_names") or [])
    for n in names:
        for v in VENUES:
            if re.fullmatch(re.escape(v).replace(r"\ ", r"\s*"), n.strip(), re.I):
                return f"{v} {row.get('year') or ''}".strip()
    return ""


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
        s = 3 * sum(1 for _ in rx.finditer(title)) + min(3, sum(1 for _ in rx.finditer(abstract)))
        if s >= 2:
            out[sid] = s
    return out


def doc_strength(title, abstract):
    return 3 * sum(1 for _ in DOC_CORE.finditer(title)) + min(4, sum(1 for _ in DOC_CORE.finditer(abstract)))


# Primary subfield = strongest match. "Document QA & VLMs" is the umbrella every
# paper partly matches ("document understanding"…), so it counts 0.6x for the
# primary choice; ties go to the more specific subfield in this order.
PRIMARY_WEIGHT = {"docvqa": 0.6}
TIE_ORDER = ["tables", "charts", "kie", "layout", "parsing", "imaging", "historical", "ocr", "agents", "rag", "longdoc", "docvqa"]


def rank(sc):
    return sorted(sc, key=lambda k: (-sc[k] * PRIMARY_WEIGHT.get(k, 1.0), TIE_ORDER.index(k)))


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
    cur = curated()
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
            if base in cur["exclude"]:
                continue
            pub = dt.date.fromisoformat(clean(e.findtext("a:published", "", NS))[:10])
            oldest = pub if oldest is None or pub < oldest else oldest
            if pub < since:
                continue
            m_id = re.match(r"(\d{2})(\d{2})\.", base)
            if m_id and (2000 + int(m_id.group(1)), int(m_id.group(2))) < (since.year, since.month):
                continue  # an old paper resurfacing (new version / cross-list): count first releases only
            title = clean(e.findtext("a:title", "", NS))
            abstract = clean(e.findtext("a:summary", "", NS))
            sc = score(title, abstract)
            forced = base in cur["include"]       # hand-confirmed: skip the relevance gates
            if not sc and forced:
                sc = {"docvqa": 2}
            if not forced and (not sc or not VISUAL.search(f"{title} {abstract}") or NOT_FIELD.search(f"{title} {abstract}")):
                continue
            tags = rank(sc)
            ds = doc_strength(title, abstract)
            if not forced and (ds < 2 or (tags[0] == "agents" and ds < 3)
                               or (tags[0] in NEEDS_STRONG and not STRONG_VISUAL.search(f"{title} {abstract}"))):
                continue  # not about documents, or text-only RAG / agents / long text
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
                "pick": ((cur["picks"][base] or True) if base in cur["picks"] else None),   # note text, or true
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
        return now, before, (None if min(now, before) < MIN_FOR_TREND else round(100 * (now - before) / before))

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
            "featured": [],
        })
    subfields.sort(key=lambda s: (-s["primary"], -s["count"]))
    # hand-picked representative papers (data/survey_curated.toml) replace "latest"
    by_id = {p["id"]: p for p in papers}
    want = sorted({i for ids in cur["featured"].values() for i in ids if i not in by_id})
    extra = arxiv_titles(want)
    for sf in subfields:
        for i in cur["featured"].get(sf["id"], []):
            if i in by_id:
                sf["featured"].append({"id": i, "title": by_id[i]["title"]})
            elif i in extra:
                sf["featured"].append({"id": i, "title": extra[i][0]})
    names = {sid: name for sid, name, *_ in COMPILED}
    for p in papers:
        p["tagnames"] = [names[t] for t in p["tags"]]

    # ---- Semantic Scholar: author IDs, citations, venues it knows about ----
    s2 = s2_enrich(papers) if papers else None
    active, cited = [], []
    if s2:
        au, au_name = collections.Counter(), {}
        for p in papers:
            row = s2.get(p["id"])
            if not row:
                continue
            p["cites"] = row.get("citationCount") or 0
            if not p["venue"]:
                p["venue"] = s2_venue(row)
            for a in row.get("authors") or []:
                if a.get("authorId"):
                    au[a["authorId"]] += 1
                    au_name[a["authorId"]] = a.get("name") or ""
        active = [{"name": au_name[i], "id": i, "n": n} for i, n in au.most_common(12) if n >= 3]
        cited = sorted((p for p in papers if p.get("cites")), key=lambda p: -p["cites"])[:6]
        cited = [{"id": p["id"], "title": p["title"], "cites": p["cites"]} for p in cited]
        print(f"  Semantic Scholar: {len(s2)}/{len(papers)} papers matched", file=sys.stderr)
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
        "authors_disambiguated": bool(s2),
        "most_cited": cited,
        "picks": [{"id": p["id"], "title": p["title"], "note": p["pick"] if isinstance(p["pick"], str) else "", "date": p["date"]}
                  for p in papers if p.get("pick") is not None],
        "min_for_trend": MIN_FOR_TREND,
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
