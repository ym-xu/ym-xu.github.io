---
title: "Lesson 6 · 差异表达分析 DESeq2"
summary: "RNA-seq 找肿瘤 vs 正常的差异基因:从 counts 到差异表达结果表。"
weight: 60
toc: true
---

**差异表达分析(Differential Expression)** 回答一个核心问题:**哪些基因在两组之间表达量不同?**(比如肿瘤 vs 正常)。`DESeq2` 是 RNA-seq 计数数据做这件事的金标准。这一课用 [L5](../05-bioconductor/) 的 `airway` 数据走一遍完整流程。

> airway 的分组是"地塞米松处理 vs 未处理"。在你自己的数据里,把这个分组换成 **肿瘤 vs 正常** 即可,流程一字不改。

## 这一课你会做出什么

- 从 counts 矩阵跑出一张差异表达结果表(log2FC + 校正 p 值)
- 按阈值筛出显著差异基因(DEGs),导出成表

## 0. 准备:加载包

```r
library(DESeq2)
library(airway)
data(airway)
```

## 1. 差异表达在问什么

每个基因在两组里都有一组表达值。差异表达就是逐个基因检验"两组均值是否不同",并给两个关键数:

- **log2FoldChange(log2FC)**:两组表达量之比取以 2 为底的对数。+1 = 处理组高一倍,−1 = 低一半。
- **padj**:多重检验校正后的 p 值(见第 5 步)。

> 🔬 这是几乎所有"找肿瘤标志基因"分析的第一步。

## 2. 设定分组的参照水平

⚠️ 一定要先告诉 DESeq2 **谁是对照组**,否则 log2FC 的正负方向可能反掉:

```r
airway$dex <- relevel(airway$dex, ref = "untrt")  # 以 "未处理" 为参照
```

> 在你的数据里就是 `relevel(condition, ref = "Normal")`,让"正常"当参照,这样正的 log2FC = 肿瘤里上调。

## 3. 构建 DESeq2 数据对象

`airway` 本身就是个 `SummarizedExperiment`([L5](../05-bioconductor/)),直接喂进去,并用 `design` 写明实验设计:

```r
dds <- DESeqDataSet(airway, design = ~ cell + dex)
```

> `design = ~ cell + dex` 表示:在校正了细胞系 `cell` 的影响后,看处理 `dex` 的效应。`~` 右边最后一项通常是你关心的分组。你自己的数据若只有分组,写 `~ condition` 即可。

## 4. 预过滤低表达基因

几乎不表达的基因是噪音,先去掉,能加速并提高检验效力:

```r
keep <- rowSums(counts(dds)) >= 10   # 总 counts 太低的基因丢掉
dds <- dds[keep, ]
```

## 5. 一键运行

```r
dds <- DESeq(dds)     # 核心:归一化 + 估计离散度 + 负二项检验,一步到位
res <- results(dds)   # 取出结果
summary(res)          # 概览:多少上调、下调、被过滤
```

`results()` 默认给 design 里最后一个变量的最末水平 vs 参照。要明确指定可以用:

```r
res <- results(dds, name = "dex_trt_vs_untrt")  # 处理 vs 未处理
resultsNames(dds)                               # 看有哪些可选的比较名
```

> **padj 为什么重要**:几万个基因同时检验,光看原始 p<0.05 会有大量假阳性。`padj`(BH 法校正)控制的是错误发现率,**筛 DEG 一定用 padj 而不是 p**(回顾 [Lesson 4](../04-stats-concepts/))。

## 6. 筛出差异基因 DEGs 并导出

```r
res <- res[order(res$padj), ]                    # 按显著性排序
deg <- subset(res, padj < 0.05 & abs(log2FoldChange) > 1)  # 常用阈值
nrow(deg)                                        # 有多少个 DEG
head(as.data.frame(deg))

write.csv(as.data.frame(deg), "DEGs.csv")        # 导出
```

> 阈值不是铁律:`padj < 0.05`、`|log2FC| > 1`(即变化 2 倍)是常用起点,可按数据和领域惯例调整,但**要在文中写清楚你用了什么阈值**。

## 小结

| 你想做 | 用什么 |
|--------|--------|
| 设参照组 | `relevel(分组, ref = "Normal")` |
| 建对象 | `DESeqDataSet(se, design = ~ condition)` |
| 预过滤 | `rowSums(counts(dds)) >= 10` |
| 跑分析 | `DESeq(dds)` |
| 取结果 | `results(dds)` / `summary(res)` |
| 筛 DEG | `subset(res, padj < 0.05 & abs(log2FoldChange) > 1)` |

下一课 [Lesson 7 · 火山图与热图](../07-volcano-heatmap/),把这张 DEG 结果可视化;[Lesson 8](../08-enrichment/) 再做功能富集。

## 延伸 Further reading

- DESeq2 官方 vignette(最权威):<https://bioconductor.org/packages/DESeq2/>
- 配套教程 RNA-seq workflow:<https://bioconductor.org/packages/rnaseqGene/>

## 常见报错 Troubleshooting

- `some values in assay are not integers` → DESeq2 要原始整数 counts,不能用 TPM/FPKM/已归一化的值。
- log2FC 方向反了 → 没设 `relevel()` 参照组,见第 2 步。
- `every gene contains at least one zero` → 预过滤太松或数据问题,先按第 4 步过滤。
- 结果里很多 `NA` 的 padj → 是 DESeq2 的独立过滤和离群点处理所致,属正常。
