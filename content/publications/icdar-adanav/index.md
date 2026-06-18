---
title: "AdaNav: Query-Adaptive Multi-Granularity Navigation for Long Document Understanding"
date: 2026-04-01
lastmod: 2026-04-01
authors: "Yiming Xu, Eric López, Artemis Llabrés, Maximiliano Hormazábal, Ernest Valveny, Dimosthenis Karatzas"
venue: "ICDAR"
year: 2026
badge: "Accepted"
summary: "AdaNav builds a multimodal document tree and navigates it at query-adaptive granularity — no embedding retriever — beating open-source VLM agent systems by over 5% on MMLongBench-Doc while reading fewer pages."
links:
  pdf: ""
draft: false
hideMeta: false
showCover: false
---

## Problem

Long document understanding is hard for Vision-Language Models: relevant evidence is scattered across dozens of pages of text, tables, and figures. Existing pipelines fall into two extremes. Embedding-based retrieval is scalable but misses pages when the query–evidence relation is compositional (multi-hop, cross-page, or cross-modal). Embedding-free approaches feed every page to a VLM, avoiding retrieval bias but incurring large token budgets and high inference cost. Neither exploits the document's inherent hierarchical structure for efficient navigation.

## Approach

AdaNav constructs a multimodal document tree that hierarchically organizes sections, pages, and visual elements, enabling structured navigation without an embedding retriever. The navigator adapts retrieval granularity to the question, moving coarse-to-fine from sections to pages to elements and filtering irrelevant branches early. A multi-agent system pairs a question router with specialized navigators whose differing action spaces let them traverse the tree at the appropriate granularity.

## Results

On MMLongBench-Doc, AdaNav outperforms existing open-source VLM-based agent systems by over 5% accuracy without using any embedding model. Compared with embedding-free approaches it retrieves far fewer pages through tree-structured navigation. The results show that heterogeneous tree representations with query-adaptive navigation are an effective alternative to both retrieval paradigms for multimodal long-document understanding.
