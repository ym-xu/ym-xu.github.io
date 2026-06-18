---
title: "Context engineering"
date: 2025-09-21
summary: "Notes on shaping the context window as a first-class design surface."
tags: ["context", "agents"]
---

Context engineering treats the prompt window as a budget to be allocated, not a bucket to be filled. Every token competes; the planner decides what earns a slot.

## Navigation over retrieval

Rather than dumping the top-k passages, we expose a navigable structure — sections summarised, leaves embedded — and let the agent walk it. Embeddings act as navigation hints; the real extraction happens at the observer.

```python
budget = Budget(tokens=8_000)
for node in planner.walk(doc):
    if budget.fits(node):
        ctx.add(node)
```
