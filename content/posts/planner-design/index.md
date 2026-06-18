---
title: "Planner design"
date: 2025-10-07
summary: "Hierarchical + leaf embeddings for navigation, not naive top-k retrieval."
tags: ["planner", "rag"]
---

我们会给"层次节点 + 叶子节点"都建 embedding（分层、分模态），不是只嵌入中间节点，也不是只嵌入叶子。

为什么不直接像 RAG 一样"扔进向量库 → 取前 k 段就回答"？因为我们的任务大量涉及版面结构、跨页聚合、图表计数/读数、多模态对齐与可解释路径。

## Embedding 覆盖策略

L1/L2/L3 section：对每个章节节点生成 summary（80–150 token），再对 summary 做 embedding。用途：Planner 的第一跳定向。

```python
def expand(passage, kg):
    neighbors = salient_neighbors(kg, budget=8)
    return passage + " ".join(neighbors)
```
