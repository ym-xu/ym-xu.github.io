"""Structured paper cards: schema, prompt, checks and DeepSeek extraction.

Shared by scripts/survey_cards.py (the site build) and the private prototype tools.
A card holds only what an abstract supports, and every field carries an exact quote
from the abstract, checked after extraction (verify()).
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from deepseek import chat_json

HERE = os.path.dirname(os.path.abspath(__file__))
ABSTRACTS = os.path.join(HERE, ".cache", "abstracts.json")
SCHEMA_VERSION = "card-v3"   # v2: pilot vocabulary + LaTeX cleaning; v3: mechanism roles, relations, problem, limitations
UA = "living-survey-fetch/2.0 (static site build)"
NS = {"a": "http://www.w3.org/2005/Atom"}

# ---- controlled vocabularies: what makes a method x task matrix possible ----
TASKS = [
    "document QA (single page)", "multi-page / long-document QA", "document retrieval", "retrieval-augmented generation",
    "layout analysis", "reading order / logical structure", "printed OCR", "post-OCR correction / alignment", "scene text recognition / spotting",
    "handwritten text recognition", "handwriting analysis (signatures, writers)", "table recognition", "table QA / reasoning", "chart / figure understanding",
    "chart-to-data / chart-to-code", "infographic understanding", "key information extraction", "document parsing / conversion",
    "document classification", "document image restoration", "text-rich image generation / editing", "document agents / navigation",
    "answer grounding / attribution", "evaluation methodology", "other",
]
MECHANISMS = [
    "vision-language model (general)", "OCR-free encoder-decoder", "layout-aware text encoder", "OCR pipeline + text model",
    "dense single-vector retrieval", "late interaction / multi-vector retrieval", "reranking", "hierarchical / tree document representation",
    "graph representation", "agent with tool use", "multi-agent system", "planning / navigation policy", "reinforcement learning",
    "chain-of-thought / test-time reasoning", "token pruning / visual compression", "long-context modelling", "memory module",
    "region-level grounding / evidence selection", "synthetic data generation", "distillation", "diffusion model",
    "mixture-of-experts", "self-supervised pretraining", "contrastive learning", "prompting / in-context learning",
    "fine-tuning (full or LoRA)", "efficient decoding / inference", "query expansion / rewriting", "output verification / correction gate",
    "new metric / evaluation protocol", "human annotation study", "other",
]
INPUTS = ["page images", "OCR text", "born-digital PDF structure", "layout boxes", "multi-page documents",
          "document collections", "scene images", "video frames", "handwriting scans", "charts / plots", "tables", "slides",
          "editable office documents (Word, HTML)", "maps / remote sensing"]
CONTRIBUTIONS = ["method", "benchmark / dataset", "analysis / evaluation", "system / tool", "model release / technical report", "survey",
                 "application study"]   # applying OCR / an LLM to a domain task (grading, clinical, legal…)


def quoted(desc):
    return {"type": "object", "additionalProperties": False, "required": ["value", "evidence"],
            "properties": {"value": desc, "evidence": {"type": "string", "description": "Exact quote from the abstract supporting this value."}}}


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["one_line", "problem", "contribution", "tasks", "inputs", "mechanisms", "benchmarks_used", "new_resource",
                 "relations", "claims", "limitations", "compute"],
    "properties": {
        "one_line": {"type": "string", "description": "What the paper does, in at most 25 words, plain language."},
        "problem": quoted({"type": "string", "description": "The gap or problem the paper addresses, paraphrased in under 25 words."}),
        "contribution": quoted({"type": "string", "enum": CONTRIBUTIONS}),
        "tasks": {"type": "array", "items": quoted({"type": "string", "enum": TASKS})},
        "inputs": {"type": "array", "items": {"type": "string", "enum": INPUTS}},
        "mechanisms": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["value", "role", "evidence"],
            "properties": {"value": {"type": "string", "enum": MECHANISMS},
                           "role": {"type": "string", "enum": ["proposed", "used"],
                                    "description": "proposed = part of this paper's contribution; used = an existing backbone, tool or baseline it relies on."},
                           "evidence": {"type": "string", "description": "Exact quote from the abstract supporting this value."}}}},
        "benchmarks_used": {"type": "array", "items": quoted({"type": "string", "description": "Benchmark or dataset name as written, e.g. DocVQA, MMLongBench-Doc."})},
        "new_resource": {"type": "object", "additionalProperties": False, "required": ["kind", "name", "evidence"],
                         "properties": {"kind": {"type": "string", "enum": ["none", "benchmark", "dataset", "model", "code / tool"]},
                                        "name": {"type": "string", "description": "Its name, or empty string."},
                                        "evidence": {"type": "string", "description": "Exact quote, or empty string when kind is none."}}},
        "relations": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "relation", "evidence"],
            "properties": {"name": {"type": "string", "description": "A named prior method, system, model or baseline, as written."},
                           "relation": {"type": "string", "enum": ["builds_on", "compared_against", "evaluates"],
                                        "description": "builds_on = extends or is built from it; compared_against = a baseline it is measured against; evaluates = a system it studies."},
                           "evidence": {"type": "string", "description": "Exact quote from the abstract naming it."}}}},
        "claims": {"type": "array", "items": quoted({"type": "string", "description": "A key result claim, paraphrased in under 20 words."})},
        "limitations": {"type": "array", "items": quoted({"type": "string", "description": "A limitation or open problem the authors themselves state, paraphrased in under 20 words."})},
        "compute": {"type": "object", "additionalProperties": False, "required": ["stated", "evidence"],
                    "properties": {"stated": {"type": "string", "description": "Model size / GPUs / cost if the abstract states it, else empty string."},
                                   "evidence": {"type": "string", "description": "Exact quote, or empty string."}}},
    },
}

SYSTEM = """You fill in a structured card for a research paper in document understanding, using only its title and abstract.

Rules:
- Use only what the abstract states. If something is not stated, leave the list empty or the string empty — never guess.
- Every "evidence" field is an exact, contiguous quote copied from the abstract (or title): same words, same order, 5–30 words. Do not paraphrase evidence.
- tasks / mechanisms: choose from the given enums; use "other" only when nothing fits. A paper that applies OCR or an LLM to a domain task (grading, clinical, legal…) has contribution "application study". List at most 3 tasks and 5 mechanisms, most central first.
- benchmarks_used: only named benchmarks or datasets the paper evaluates on or builds on, not ones it merely mentions as background. A benchmark the paper itself introduces goes in new_resource, not here.
- claims: at most 3, the main results, each paraphrased briefly; keep numbers exactly as written.
- compute: only if the abstract states model size, hardware or cost.
- problem: the gap the paper sets out to close, as the authors frame it.
- mechanisms: mark each as "proposed" (part of this paper's contribution) or "used" (an existing backbone, tool or baseline it relies on). A VLM that merely powers an agent is "used".
- relations: only works the abstract names explicitly (methods, systems, models, baselines). Generic phrases like "strong baselines" or "existing methods" are not relations.
- limitations: only limitations or open problems the authors themselves state. Never add your own critique; leave empty when none is stated."""


# ---------------------------------------------------------------- abstracts ----
def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def clean_tex(s):
    """Strip LaTeX markup some abstracts carry, so quotes can match plain text."""
    s = re.sub(r"\\(?:emph|textbf|textit|texttt|mathrm|mathbf|text)\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\%", "%").replace("\\&", "&").replace("``", '"').replace("''", '"').replace("--", "–")
    s = re.sub(r"\\name\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def fetch_abstracts(ids, path=ABSTRACTS):
    """Full abstracts from arXiv (survey.json keeps a short excerpt only), cached in `path`."""
    cache = {k: {"title": clean_tex(v["title"]), "abstract": clean_tex(v["abstract"])} for k, v in load_json(path, {}).items()}
    todo = [i for i in ids if i not in cache]
    for k in range(0, len(todo), 100):
        chunk = todo[k:k + 100]
        q = urllib.parse.urlencode({"id_list": ",".join(chunk), "max_results": len(chunk)})
        req = urllib.request.Request("https://export.arxiv.org/api/query?" + q, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=90) as r:
            root = ET.fromstring(r.read())
        for e in root.findall("a:entry", NS):
            i = re.sub(r"v\d+$", "", e.findtext("a:id", "", NS).rsplit("/", 1)[-1])
            cache[i] = {"title": clean_tex(e.findtext("a:title", "", NS)),
                        "abstract": clean_tex(e.findtext("a:summary", "", NS))}
        print(f"  abstracts: {min(k + 100, len(todo))}/{len(todo)} fetched", file=sys.stderr)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        time.sleep(3.5)
    return cache


# ------------------------------------------------------------------- checks ----
def norm(s):
    """Lower-case words; dashes and hyphens of every kind count as word breaks, so
    'OCR-free', 'OCR free' and 'questions — spanning' / 'questions—spanning' all match."""
    s = s.lower().replace("’", "'")
    s = re.sub(r"[‐-―\-]", " ", s)
    return re.sub(r"[^a-z0-9%.+' ]+", " ", s).split()


def found(quote, source):
    """Is the quote really in the source text? Word-level, punctuation-insensitive."""
    q, src = " ".join(norm(quote)), " ".join(norm(source))
    return bool(q) and q in src


def verify(card, title, abstract):
    """Mark every evidence quote as found / not found in title + abstract."""
    src = f"{title}. {abstract}"
    checked = total = 0
    def chk(item):
        nonlocal checked, total
        ev = item.get("evidence", "")
        if not ev:
            item["verified"] = None
            return
        total += 1
        item["verified"] = found(ev, src)
        checked += item["verified"]
    chk(card["contribution"])
    chk(card["problem"])
    for key in ("tasks", "mechanisms", "benchmarks_used", "relations", "claims", "limitations"):
        for item in card[key]:
            chk(item)
    chk(card["new_resource"])
    chk(card["compute"])
    card["_quotes"] = {"verified": checked, "total": total}
    return card


# ------------------------------------------------------------- DeepSeek path ----
DS_PROMPT = "ds-2"   # bump when DS_SYSTEM changes, so cached DeepSeek cards are redone
DS_SYSTEM = SYSTEM + """

Be sparing — a short accurate card beats a full one:
- mechanisms: only the techniques the abstract presents as central, usually 1–3. Never pad
  the list and never repeat a value. "proposed" is only for what this paper itself introduces;
  the backbone VLM/LLM, the OCR engine, the detector or the base model it fine-tunes is "used",
  and a standard step such as fine-tuning is "used" unless the training recipe is the contribution.
- tasks: the task(s) the paper is about, usually 1–2; "evaluation methodology" only when a
  new metric or protocol is the point of the paper.
- relations: one entry per named work; split "GPT-5.4, Gemini 3.5 Flash and Qwen3.6" into three.
  A model the paper builds its system from is builds_on; models it benchmarks are evaluates or
  compared_against.

Output format: a single JSON object (json) that validates against this JSON Schema. Use the
enum strings exactly as written. Every key in "required" must be present; use [] or "" when
the abstract says nothing.

""" + json.dumps(SCHEMA, ensure_ascii=False)

ENUMS = {"tasks": TASKS, "mechanisms": MECHANISMS}
LIMITS = {"tasks": 3, "mechanisms": 5, "claims": 3}


def conform(card):
    """Bring a JSON-mode card into the schema. Returns (card, fixes, fatal):
    fixes are small repairs that were applied, fatal are problems worth a retry."""
    fixes, fatal = [], []
    if not isinstance(card, dict):
        return None, fixes, ["not an object"]

    def obj(key, fields):
        v = card.get(key)
        if not isinstance(v, dict):
            fatal.append(f"{key} missing")
            v = {}
        card[key] = {f: str(v.get(f) or "") for f in fields}

    if not isinstance(card.get("one_line"), str):
        fatal.append("one_line missing")
        card["one_line"] = ""
    obj("problem", ("value", "evidence"))
    obj("contribution", ("value", "evidence"))
    alias = {"evaluation methodology": "analysis / evaluation", "analysis": "analysis / evaluation",
             "evaluation": "analysis / evaluation", "benchmark": "benchmark / dataset", "dataset": "benchmark / dataset",
             "system": "system / tool", "tool": "system / tool", "technical report": "model release / technical report"}
    cv = card["contribution"]["value"]
    if cv not in CONTRIBUTIONS and cv.lower() in alias:
        fixes.append(f"contribution {cv!r} -> {alias[cv.lower()]!r}")
        card["contribution"]["value"] = alias[cv.lower()]
    if card["contribution"]["value"] not in CONTRIBUTIONS:
        fatal.append(f"contribution {card['contribution']['value']!r} not in enum")
    obj("new_resource", ("kind", "name", "evidence"))
    if card["new_resource"]["kind"] not in ("none", "benchmark", "dataset", "model", "code / tool"):
        fixes.append(f"new_resource.kind {card['new_resource']['kind']!r} -> none")
        card["new_resource"]["kind"] = "none"
    obj("compute", ("stated", "evidence"))

    for key in ("tasks", "mechanisms", "benchmarks_used", "relations", "claims", "limitations", "inputs"):
        v = card.get(key)
        if not isinstance(v, list):
            if v is not None:
                fixes.append(f"{key} not a list")
            else:
                fatal.append(f"{key} missing")
            v = []
        card[key] = v
    card["inputs"] = [x for x in card["inputs"] if x in INPUTS] or []
    for key in ("tasks", "mechanisms", "benchmarks_used", "claims", "limitations", "relations"):
        items = []
        for it in card[key]:
            if not isinstance(it, dict):
                fixes.append(f"{key}: dropped a non-object item")
                continue
            it = {k: str(x) for k, x in it.items()}
            if key in ENUMS and it.get("value") not in ENUMS[key]:
                fixes.append(f"{key}: {it.get('value')!r} -> other")
                it["value"] = "other"
            if key == "mechanisms" and it.get("role") not in ("proposed", "used"):
                fixes.append(f"mechanism role {it.get('role')!r} -> proposed")
                it["role"] = "proposed"
            if key == "relations":
                if it.get("relation") not in ("builds_on", "compared_against", "evaluates") or not it.get("name"):
                    fixes.append(f"relations: dropped {it.get('name')!r}/{it.get('relation')!r}")
                    continue
                items.append({"name": it["name"], "relation": it["relation"], "evidence": it.get("evidence", "")})
                continue
            keep = ("value", "role", "evidence") if key == "mechanisms" else ("value", "evidence")
            items.append({k: it.get(k, "") for k in keep})
        if key in ENUMS:
            seen, uniq = set(), []
            for it in items:
                if it["value"] not in seen:
                    seen.add(it["value"])
                    uniq.append(it)
            if len(uniq) < len(items):
                fixes.append(f"{key}: removed {len(items) - len(uniq)} duplicate(s)")
            items = uniq
        if key in LIMITS and len(items) > LIMITS[key]:
            fixes.append(f"{key}: trimmed {len(items)} -> {LIMITS[key]}")
            items = items[:LIMITS[key]]
        card[key] = items
    for k in [k for k in card if k not in SCHEMA["properties"]]:
        fixes.append(f"dropped extra key {k}")
        card.pop(k)
    return card, fixes, fatal


def extract_one_deepseek(key, pid, title, abstract):
    user = f"Title: {title}\n\nAbstract: {abstract}"
    usage = {}
    for attempt in range(2):
        raw, meta = chat_json(key, DS_SYSTEM, user, max_tokens=3000)
        for k, v in meta["usage"].items():
            if isinstance(v, int):
                usage[k] = usage.get(k, 0) + v
        card, fixes, fatal = conform(raw)
        if not fatal:
            return card, {"model": meta["model"], "prompt": DS_PROMPT, "fixes": fixes, "attempts": attempt + 1}, usage
        user = (f"Title: {title}\n\nAbstract: {abstract}\n\nYour previous card had these problems: "
                + "; ".join(fatal) + ". Return the full corrected JSON object.")
    return None, {"error": "schema", "problems": fatal}, usage
