---
title: "Lesson 7 · 火山图与热图 Volcano & Heatmap"
summary: "把差异表达结果可视化:火山图展示 DEGs,热图展示样本聚类。"
weight: 70
toc: true
---

[Lesson 6](../06-deseq2/) 跑出了差异表达结果,但一张几万行的表没法直接看。这一课把它变成论文里最常见的两张图:**火山图**(一眼看出哪些基因显著上/下调)和**热图**(看 DEG 在各样本里的表达模式与聚类)。

## 这一课你会做出什么

- 一张标注了关键基因的火山图
- 一张带分组注释、行聚类的表达热图

## 0. 准备:加载包 + 快速重建 L6 的结果

```r
library(DESeq2)
library(airway)
library(EnhancedVolcano)
library(pheatmap)

# —— 快速重建 Lesson 6 的 dds 和 res(自包含,可直接跑)——
data(airway)
airway$dex <- relevel(airway$dex, ref = "untrt")
dds <- DESeqDataSet(airway, design = ~ cell + dex)
dds <- dds[rowSums(counts(dds)) >= 10, ]
dds <- DESeq(dds)
res <- results(dds)
```

## 1. 火山图的原理

火山图把每个基因画成一个点:

- **x 轴 = log2FoldChange**:越往右上调越强,越往左下调越强
- **y 轴 = −log10(padj)**:越往上越显著

于是图的**左上角 = 显著下调**、**右上角 = 显著上调**,中间底部 = 没差异。形状像火山喷发,故名。

> 🔬 审稿人扫一眼火山图就能判断"差异基因多不多、对不对称"。

## 2. 画火山图

`EnhancedVolcano` 一个函数搞定,还能自动标注、画阈值线:

```r
EnhancedVolcano(res,
  lab    = rownames(res),       # 点的标签(airway 里是 Ensembl ID)
  x      = 'log2FoldChange',
  y      = 'padj',              # 用校正后的 p 值
  pCutoff  = 0.05,              # 显著性阈值(横虚线)
  FCcutoff = 1,                 # fold-change 阈值(竖虚线)
  title  = 'Tumor vs Normal',
  pointSize = 1.5, labSize = 3)
```

> 标签是 Ensembl ID 不好读时,可以先把 ID 映射成基因名(symbol)再传给 `lab=`,映射方法见 [Lesson 8](../08-enrichment/) 的 `bitr()`。

## 3. 为什么热图不能直接用原始 counts

原始 counts 量级差异巨大(有的基因几十、有的上万),且方差随均值变化。直接画热图会被高表达基因主导。标准做法是先做**方差稳定变换 vst**:

```r
vsd <- vst(dds, blind = FALSE)   # 方差稳定变换,得到适合可视化/聚类的值
mat <- assay(vsd)                # 取出变换后的矩阵
```

## 4. 取 top 差异基因

热图通常只画最显著的若干基因(全画看不清):

```r
top_genes <- head(order(res$padj), 30)   # padj 最小的 30 个
mat_top <- mat[top_genes, ]
```

## 5. 画带注释的热图

`scale = "row"` 对每个基因按行做 z-score 标准化,这样看的是"相对高低"而非绝对值;`annotation_col` 给样本加上分组注释条:

```r
anno <- as.data.frame(colData(dds)[, c("dex", "cell")])  # 样本注释:处理/细胞系

pheatmap(mat_top,
  scale = "row",              # ⚠️ 按行 z-score,热图的关键
  annotation_col = anno,      # 顶部加分组注释条
  show_colnames = FALSE,
  main = "Top 30 DEGs")
```

**怎么读**:颜色 = 相对表达(红高蓝低,默认);顶部聚类树把表达模式相似的样本聚在一起——如果同一处理组的样本自然聚到一块,说明分组的差异是真实的。

> 想要更精细的注释和配色,可换 `ComplexHeatmap::Heatmap()`,功能更强但语法稍复杂。

## 小结

| 你想做 | 用什么 |
|--------|--------|
| 火山图 | `EnhancedVolcano(res, x='log2FoldChange', y='padj', ...)` |
| 适合画图的表达值 | `vst(dds)` → `assay()` |
| 取 top 基因 | `head(order(res$padj), N)` |
| 带注释热图 | `pheatmap(mat, scale="row", annotation_col=)` |

下一课 [Lesson 8 · 功能富集](../08-enrichment/),解释这些差异基因背后是哪些通路。

## 延伸 Further reading

- EnhancedVolcano:<https://bioconductor.org/packages/EnhancedVolcano/>
- ComplexHeatmap 完全手册:<https://jokergoo.github.io/ComplexHeatmap-reference/book/>

## 常见报错 Troubleshooting

- 热图一片单色 / 看不出差异 → 忘了 `scale = "row"`,被绝对量级主导。
- `EnhancedVolcano` 标签太多挤成一团 → 调小 `labSize`,或只标注感兴趣的基因(用 `selectLab=`)。
- `vst()` 很慢或报错 → 样本很少时可改用 `rlog(dds)`;确认 `dds` 已经过 `DESeq()`。
- 注释条没出现 → `annotation_col` 的行名必须和矩阵列名(样本名)一致。
