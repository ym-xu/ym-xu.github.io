---
title: "Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections"
date: 2026-06-01
lastmod: 2026-06-01
authors: "Łukasz Borchmann, Jordy Van Landeghem, Michał Turski, Shreyansh Padarha, Ryan Othniel Kearns, Adam Mahdi, Niels Rogge, Clémentine Fourrier, Siwei Han, Huaxiu Yao, Artemis Llabrés, Yiming Xu, Dimosthenis Karatzas, Hao Zhang, Anupam Datta"
venue: "ICML"
year: 2026
badge: "Oral"
summary: "MADQA — a multimodal agentic document-QA benchmark that scores full search trajectories, not just answers, and shows agents match human accuracy only by working ~5× harder."
links:
  pdf: "https://arxiv.org/abs/2603.12180"
  code: "https://github.com/OxRML/MADQA"
  leaderboard: "https://huggingface.co/spaces/Snowflake/MADQA-Leaderboard"
  data: "https://huggingface.co/datasets/OxRML/MADQA"
draft: false
hideMeta: false
showCover: false
---

## Motivation

Enterprise document intelligence increasingly relies on agents that iteratively search, gather, and reason over visual and textual evidence spread across large document collections. Final-answer accuracy alone is a deceptive way to judge them: it tells you whether a system solved a task, but not whether it did so in one well-aimed query or by flailing through dozens of expensive steps until it got lucky.

## The benchmark

MADQA (Multimodal Agentic Document Question Answering) was built with academic and industrial partners including the University of Oxford, UNC-Chapel Hill, CVC, and Hugging Face. Unlike benchmarks that check only the final output, MADQA logs the whole search trajectory — every query, page view, and timestamp — enabling the first direct comparison of how agents navigate documents versus how humans do. Questions are human-authored, strictly grounded in a closed-world document set, and most require genuine visual understanding (correlating table rows, reading a checkbox or a stamped date) that pure OCR cannot solve. We curated the set with Classical Test Theory so it discriminates real capability rather than luck.

## Architectural standoff

We compared three paradigms. Static RAG is fast but brittle, suffering "last-mile" failures that find the right document yet miss the right page — Gemini 3 Pro File Search reached 78.6%. Constrained agents won: by decomposing problems and reformulating queries, the Gemini 3 Pro Agent reached state-of-the-art 82.2%, beating RAG by 4.6 points. Unconstrained recursive language models were a cautionary tale — one processed over 270 million tokens at roughly $850 yet still failed to match the much cheaper agent on the same base model.

## Strategic navigation vs. stochastic search

The headline gap is efficiency, not accuracy. Humans solved about 50% of questions on their first query; the best agent started near 12% and needed up to 9 rounds of search to reach the same ~82% accuracy — roughly 5× the effort. To quantify this we adapted the Kuiper statistic into an accuracy-to-effort score that penalizes "flailing" on lost causes: humans scored ~14.6 (well calibrated, spending effort only where it pays off) while the best agents scored ~22.9 (persisting in stochastic search loops). Agents brute-force their way to answers; humans navigate strategically.

## Takeaway

A bigger context window is a bigger flashlight, not a map. Enterprise AI should be judged not only on whether it reaches the right answer but on how efficiently and strategically it gets there — recognizing when it is lost and halting the moment the answer is found. MADQA is released to push the field from brute-force retrieval toward calibrated, cost-aware intelligence.
