---
title: "Fine-grained label learning via siamese network for cross-modal information retrieval"
date: 2019-06-08
lastmod: 2019-06-08
authors: "Yiming Xu, Jing Yu, Jingjing Guo, Yue Hu, Jianlong Tan"
venue: "ICCS"
year: 2019
summary: "Fine-grained labels capture the \"hardness\" of text–image pairs; a siamese network and a weighted pairwise loss exploit them to improve cross-modal retrieval on three benchmarks."
links:
  pdf: "https://www.iccs-meeting.org/archive/iccs2019/papers/115370293.pdf"
  doi: "10.1007/978-3-030-22741-8_22"
  page: "https://scholar.google.com/citations?view_op=view_citation&hl=en&user=n1skXDsAAAAJ&citation_for_view=n1skXDsAAAAJ:u5HHmVD_uO8C"
draft: false
showCover: false
---

## Problem

Cross-modal information retrieval searches for semantically relevant data in one modality given a query in another. For text–image retrieval, the common solution maps both into a shared semantic space and measures similarity directly, training on positive and negative pairs. Existing work treats all positive/negative pairs as equally positive/negative — yet many positives resemble negatives to some degree, and vice versa. These "hard examples" are exactly what existing models handle poorly.

## Approach

We assign fine-grained labels that capture each example's degree of hardness. A siamese network operates on both positive and negative examples to obtain their semantic similarities: for each pair we use the image's text description to compute its similarity with the text in the example, and from these similarities derive the fine-grained labels. The labels feed a pairwise similarity loss that increases the influence of hard examples while maximizing similarity for relevant text–image pairs and minimizing it for irrelevant ones.

## Results

Across the English Wikipedia, Chinese Wikipedia, and TVGraz datasets, incorporating fine-grained labels yields significant improvement in retrieval performance over state-of-the-art models, confirming the value of difficulty-aware supervision for cross-modal alignment.
