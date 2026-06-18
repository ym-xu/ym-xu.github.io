---
title: "Retrieval-based Question Answering with Passage Expansion Using a Knowledge Graph"
date: 2024-05-01
lastmod: 2024-05-01
authors: "Benno Kruit, Yiming Xu, Jan-Christoph Kalo"
venue: "LREC-COLING"
year: 2024
badge: "Oral"
summary: "A multimodal retriever that combines knowledge-graph entity features with dense text retrieval, improving open-domain QA precision on rare, entity-centric questions where dense retrievers fall short."
links:
  pdf: "https://aclanthology.org/2024.lrec-main.1225.pdf"
  page: "https://aclanthology.org/2024.lrec-main.1225/"
draft: false
case: true
layout: "case"
hideMeta: true
showCover: false
---

<section class="wide">
  <h2>Overview</h2>
  <p>
    We explore whether a multimodal passage retriever can bolster open-domain QA by combining <strong>knowledge-graph entity features</strong> with dense text retrieval, compensating for dense neural retrievers' weakness on rare entities and facts.
  </p>
</section>

<figure class="wide">
  <img src="./cover.jpg" alt="Teaser: KG expansion improves retrieval and QA" />
  <figcaption>KG-driven passage expansion exposes relevant entities and relations for better retrieval and answer grounding.</figcaption>
</figure>

<section class="wide">
  <h3>Approach</h3>
  <p>
    We devise a multimodal retriever that combines entity features from a knowledge graph with textual data. A distantly supervised question–relation extraction model enhances retrieval through the KG, injecting structured context that connects questions to passages reachable via entities and relations rather than surface similarity alone.
  </p>
  <h3>Results</h3>
  <p>
    Across several datasets, the approach improves retrieval precision in some settings and confirms enhanced performance on entity-centric questions, indicating that knowledge-graph context complements dense retrieval for rare entities. Challenges remain for complex, generalized questions.
  </p>
</section>
