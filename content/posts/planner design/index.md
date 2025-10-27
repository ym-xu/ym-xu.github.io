---
title: planner work process
date: 2025-10-07
lastmod: 2025-10-07T10:06:17+02:00
slug:
summary: ""
tags:
  - ""
series: ""
draft: false
toc: true
comments: true
key: value
---
planner

**索引产物**
- dense_coarse.jsonl / coarse.faiss：section/image/table 的语义摘要（MMR）
- sparse_coarse.jsonl / bm25_coarse/：多域（title/caption/table_schema/labels/aliases/body）
- dense_leaf.jsonl（仅 chunk） / leaf.faiss / bm25_leaf/：叶子级检索（Observer 用）
- graph_edges.jsonl：child / parent / sibling / same_page / ref / has_col
- id_maps.json：label2id、figure、table
**角色**：section、image（figure）、table
**Planner 输出**：**计划**，不是执行结果；执行交给 **Observer**，判定与补证由 **Reasoner** 完成。
**LLM-Only**：无 LLM 直接报错，不走规则回退。

让agents自己决定使用哪种检索方法

1 特定任务 医疗场景

2 小模型碰瓷大模型

3 可解释性，怎么证明你的可解释性比CoT优越呢

4 长文档理解，但是做攻击或隐私保护


| Type         | Samples | Correct | Accuracy |
|--------------|--------:|--------:|---------:|
| Single-page  |     370 |      89 |   24.05% |
| Cross-page   |     465 |      32 |    6.88% |
| Unknown      |     219 |     146 |   66.67% |
| **Overall**  |   1,054 |     267 |   25.33% |


| Evidence Source  | Samples | Correct | Accuracy |
|------------------|--------:|--------:|---------:|
| Chart            |     174 |      36 |   20.69% |
| Table            |     210 |      30 |   14.29% |
| Pure-text        |     300 |      51 |   17.00% |
| Generalized-text |     115 |      11 |    9.57% |
| Figure           |     296 |      26 |    8.78% |
| **Overall**      |   1,054 |     267 |   25.33% |