---
title: "Retrieval-based Question Answering with Passage Expansion Using a Knowledge Graph"
date: 2024-05-01
lastmod: 2024-05-01
authors: "Benno Kruit, Yiming Xu, Jan-Christoph Kalo"
venue: "LREC-COLING"
year: 2024
summary: "A retrieval QA method that expands passages with KG context to improve recall and answer quality."
links:
  pdf: "https://aclanthology.org/2024.lrec-main.1225.pdf"
  project: "https://aclanthology.org/2024.lrec-main.1225/"
draft: false
case: true
layout: "case"
hideMeta: true
showCover: false
---

<section class="wide">
  <h2>Overview</h2>
  <p>
    We propose a retrieval-based QA method that <strong>expands candidate passages</strong> with structured context from a knowledge graph (KG). By injecting entities and relations that connect the question and passages, we improve recall on entity-centric queries while preserving precision.
  </p>
</section>

<figure class="wide">
  <img src="./cover.jpg" alt="Teaser: KG expansion improves retrieval and QA" />
  <figcaption>KG-driven passage expansion exposes relevant entities and relations for better retrieval and answer grounding.</figcaption>
</figure>

<!-- <div class="cols-1">
  <div class="col">

    <h3>Method</h3>
    <p>
      Given a query and a set of retrieved passages, we match entities in a KG and expand each passage with a small set of <em>salient</em> neighbors (entities/relations) subject to a budget. The expanded text is fed to a reader (or LLM) for answer extraction or generation.
    </p>
    <ul>
      <li>Entity linking → KG lookup → expansion candidate scoring</li>
      <li>Budgeted selection to avoid noise and verbosity</li>
      <li>Plug-and-play with standard RAG pipelines</li>
    </ul>
  </div>
  <div class="col">
    <figure>
      <img src="./diagram.svg" alt="System diagram of KG expansion within a RAG pipeline" />
      <figcaption>System diagram</figcaption>
    </figure>
  </div>
</div> -->

<!-- <div class="callout">
  <strong>Key takeaway.</strong> KG expansion improves recall for entity-centric questions and yields higher answer quality with modest added cost.
</div> -->

<!-- <h3>Results</h3>
<div class="grid-3">
  <figure>
    <img src="./ex1.svg" alt="Example 1" />
    <figcaption>Qualitative example 1</figcaption>
  </figure>
  <figure>
    <img src="./ex2.svg" alt="Example 2" />
    <figcaption>Qualitative example 2</figcaption>
  </figure>
  <figure>
    <img src="./ex3.svg" alt="Example 3" />
    <figcaption>Qualitative example 3</figcaption>
  </figure>
</div> -->

<!-- <h3>Resources</h3>
<ul>
  <li><a href="https://aclanthology.org/2024.lrec-main.1225.pdf" target="_blank" rel="noopener">Paper (PDF)</a></li>
  <li><a href="https://aclanthology.org/2024.lrec-main.1225/" target="_blank" rel="noopener">Anthology page</a></li>
</ul> -->
