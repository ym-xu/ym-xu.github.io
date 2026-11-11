#!/usr/bin/env python3
"""Paper cards for the survey page: data/survey_cards.json.

For every paper in data/survey.json, a compact card (what it proposes, for which task)
extracted from the abstract by DeepSeek. Only tags whose evidence quote is found
verbatim in the abstract are kept. From the cards it builds the methods x tasks
matrix shown on /survey/, including "quiet" combinations: pairs that are common on
their own but rarely meet.

    python3 scripts/survey_cards.py                 # add cards for new papers, rebuild
    python3 scripts/survey_cards.py --no-extract    # rebuild the matrix only
    python3 scripts/survey_cards.py --import prototype/cards.jsonl   # seed from full cards

Runs in the daily build after fetch_survey.py. Without a DeepSeek key (repository
secret DEEPSEEK_API_KEY) it only rebuilds from the cards it already has, so new papers
just appear without tags until a key is available. Only public arXiv abstracts are sent.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import sys

import cards_core as core
import deepseek

HERE = os.path.dirname(os.path.abspath(__file__))
SURVEY = os.path.join(HERE, "..", "data", "survey.json")
OUT = os.path.join(HERE, "..", "data", "survey_cards.json")
LIMIT = 120          # at most this many new cards per run (a backfill can be run locally)

# Short labels for the matrix; ids are stable slugs used in data attributes and URLs.
SHORT = {
    "vision-language model (general)": "VLM", "OCR-free encoder-decoder": "OCR-free model",
    "layout-aware text encoder": "Layout-aware encoder", "OCR pipeline + text model": "OCR + text model",
    "dense single-vector retrieval": "Dense retrieval", "late interaction / multi-vector retrieval": "Multi-vector retrieval",
    "reranking": "Reranking", "hierarchical / tree document representation": "Tree / hierarchy",
    "graph representation": "Graphs", "agent with tool use": "Tool-using agent", "multi-agent system": "Multi-agent",
    "planning / navigation policy": "Planning / navigation", "reinforcement learning": "RL",
    "chain-of-thought / test-time reasoning": "Test-time reasoning", "token pruning / visual compression": "Token compression",
    "long-context modelling": "Long context", "memory module": "Memory", "region-level grounding / evidence selection": "Grounding / evidence",
    "synthetic data generation": "Synthetic data", "distillation": "Distillation", "diffusion model": "Diffusion",
    "mixture-of-experts": "MoE", "self-supervised pretraining": "Self-supervised", "contrastive learning": "Contrastive",
    "prompting / in-context learning": "Prompting", "fine-tuning (full or LoRA)": "Fine-tuning",
    "efficient decoding / inference": "Efficient inference", "query expansion / rewriting": "Query rewriting",
    "output verification / correction gate": "Verification", "new metric / evaluation protocol": "New metric",
    "human annotation study": "Human study",
    "document QA (single page)": "Page QA", "multi-page / long-document QA": "Long-doc QA", "document retrieval": "Retrieval",
    "retrieval-augmented generation": "RAG", "layout analysis": "Layout", "reading order / logical structure": "Reading order",
    "printed OCR": "Printed OCR", "post-OCR correction / alignment": "Post-OCR", "scene text recognition / spotting": "Scene text",
    "handwritten text recognition": "HTR", "handwriting analysis (signatures, writers)": "Writers / signatures",
    "table recognition": "Table recognition", "table QA / reasoning": "Table QA", "chart / figure understanding": "Charts",
    "chart-to-data / chart-to-code": "Chart-to-code", "infographic understanding": "Infographics",
    "key information extraction": "Extraction", "document parsing / conversion": "Parsing", "document classification": "Classification",
    "document image restoration": "Restoration", "text-rich image generation / editing": "Text-image generation",
    "document agents / navigation": "Doc agents", "answer grounding / attribution": "Attribution",
    "evaluation methodology": "Evaluation",
}
ROWS, COLS = 14, 10  # matrix size: most-proposed mechanisms x most-studied tasks
# Evaluation work is kept out of the matrix: a benchmark paper proposes a metric, not a
# method, so "RL x evaluation" would always look like a gap. It is counted elsewhere.
META = {"new-metric-evaluation-protocol", "human-annotation-study", "evaluation-methodology"}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def compact(card, model):
    """Public card: verified tags only, no quotes (they stay in the full card)."""
    ok = lambda it: it.get("verified") is True and it.get("value") not in (None, "", "other")
    q = card.get("_quotes", {})
    return {
        "c": card["contribution"]["value"],
        "o": card.get("one_line", "")[:220],
        "t": [slug(t["value"]) for t in card["tasks"] if ok(t)],
        "mp": [slug(m["value"]) for m in card["mechanisms"] if ok(m) and m.get("role") == "proposed"],
        "mu": [slug(m["value"]) for m in card["mechanisms"] if ok(m) and m.get("role") == "used"],
        "r": card["new_resource"]["kind"] if card["new_resource"].get("verified") else "none",
        "rn": card["new_resource"].get("name", "")[:60] if card["new_resource"].get("verified") else "",
        "q": [q.get("verified", 0), q.get("total", 0)],
        "m": model,
        "s": core.SCHEMA_VERSION + "/" + core.DS_PROMPT,
    }


def import_full(path):
    """Seed from a full cards.jsonl (the latest current-prompt DeepSeek row per paper)."""
    rows = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if (r.get("provider") == "deepseek" and r.get("schema") == core.SCHEMA_VERSION
                and r.get("meta", {}).get("prompt") == core.DS_PROMPT):
            rows[r["id"]] = compact(r["card"], r["meta"].get("model", ""))
    return rows


def extract(papers, key, workers=6):
    abstracts = core.fetch_abstracts([p["id"] for p in papers])
    out, errors = {}, 0

    def work(p):
        a = abstracts.get(p["id"]) or {"title": p["title"], "abstract": p.get("abs", "")}
        card, meta, _ = core.extract_one_deepseek(key, p["id"], a["title"], a["abstract"])
        if card is None:
            raise RuntimeError(meta)
        return p["id"], compact(core.verify(card, a["title"], a["abstract"]), meta["model"])

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in cf.as_completed([pool.submit(work, p) for p in papers]):
            try:
                pid, c = fut.result()
                out[pid] = c
            except Exception as e:                 # retried on the next run
                errors += 1
                print(f"  ! card failed: {type(e).__name__}: {e}", file=sys.stderr)
    return out, errors


def matrix(papers, cards):
    """Proposed mechanism x task counts over the counted (non-application) papers."""
    pool = [(p, cards[p["id"]]) for p in papers if p["id"] in cards and not p.get("app")]
    n = len(pool)
    mech_n, task_n = {}, {}
    for _, c in pool:
        for m in set(c["mp"]):
            mech_n[m] = mech_n.get(m, 0) + 1
        for t in set(c["t"]):
            task_n[t] = task_n.get(t, 0) + 1
    rows = sorted((k for k in mech_n if k not in META), key=lambda k: -mech_n[k])[:ROWS]
    cols = sorted((k for k in task_n if k not in META), key=lambda k: -task_n[k])[:COLS]
    names = {slug(k): k for k in core.MECHANISMS + core.TASKS}
    cells, best = [], 1
    quiet = []
    for m in rows:
        line = []
        for t in cols:
            k = sum(1 for _, c in pool if m in c["mp"] and t in c["t"])
            e = mech_n[m] * task_n[t] / max(1, n)          # expected if they were independent
            cell = {"n": k, "e": round(e, 1)}
            if e >= 3 and k <= e / 4:
                cell["quiet"] = True
                quiet.append({"m": m, "t": t, "n": k, "e": round(e, 1)})
            elif k >= 5 and k >= 2 * e:
                cell["hot"] = True
            best = max(best, k)
            line.append(cell)
        cells.append(line)
    lab = lambda k, tot: {"id": k, "name": SHORT.get(names.get(k, k), names.get(k, k)), "full": names.get(k, k), "n": tot[k]}
    quiet.sort(key=lambda q: q["n"] - q["e"])
    for q in quiet:
        q["mname"] = SHORT.get(names[q["m"]], names[q["m"]])
        q["tname"] = SHORT.get(names[q["t"]], names[q["t"]])
    return {"rows": [lab(m, mech_n) for m in rows], "cols": [lab(t, task_n) for t in cols],
            "cells": cells, "max": best, "n": n, "quiet": quiet[:8]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--import", dest="imp", help="seed from a full cards.jsonl")
    ap.add_argument("--limit", type=int, default=LIMIT)
    args = ap.parse_args()

    papers = core.load_json(SURVEY, {}).get("papers", [])
    store = core.load_json(OUT, {}).get("papers", {})
    if args.imp:
        store.update(import_full(args.imp))
    current = {p["id"] for p in papers}
    store = {k: v for k, v in store.items() if k in current}      # keep only the survey window
    missing = [p for p in papers if p["id"] not in store]
    errors = 0
    if missing and not args.no_extract:
        key = deepseek.api_key(required=False)
        if not key:
            print(f"cards: {len(missing)} papers without a card, no DeepSeek key — skipped", file=sys.stderr)
        else:
            batch = missing[:args.limit]
            print(f"cards: extracting {len(batch)} of {len(missing)} missing", file=sys.stderr)
            new, errors = extract(batch, key)
            store.update(new)

    mx = matrix(papers, store)
    ver = sum(c["q"][0] for c in store.values())
    tot = sum(c["q"][1] for c in store.values()) or 1
    models = sorted({c["m"] for c in store.values() if c.get("m")})
    out = {"updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "models": models, "n": len(store), "of": len(papers), "quotes": round(100 * ver / tot, 1),
           "matrix": mx, "papers": store,
           "names": {slug(k): SHORT.get(k, k) for k in core.MECHANISMS + core.TASKS if k != "other"}}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"cards: {len(store)}/{len(papers)} papers · matrix {len(mx['rows'])}x{len(mx['cols'])} over {mx['n']} · "
          f"{len(mx['quiet'])} quiet pairs · quotes {out['quotes']}% · errors {errors}", file=sys.stderr)


if __name__ == "__main__":
    main()
