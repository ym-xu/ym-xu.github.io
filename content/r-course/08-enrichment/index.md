---
title: "Lesson 8 · 功能富集 Enrichment"
summary: "用 clusterProfiler 做 GO / KEGG / GSEA,解释差异基因背后的通路。"
weight: 80
toc: true
---

[Lesson 6](../06-deseq2/) 找出几百个差异基因,但"一长串基因名"不是结论。**功能富集**回答更有意义的问题:**这些基因集中在哪些生物学通路上?**(比如细胞周期、免疫应答)。`clusterProfiler` 是做这件事最常用的包。

## 这一课你会做出什么

- 把基因列表富集到 GO / KEGG 通路
- 区分并各做一次 ORA 和 GSEA
- 画出富集结果的点图、条形图、GSEA 曲线

## 0. 准备:加载包

```r
library(clusterProfiler)
library(org.Hs.eg.db)   # 人类基因注释数据库
library(enrichplot)     # 富集结果可视化
```

## 1. ORA vs GSEA:两种思路

- **过表征分析 ORA(Over-Representation Analysis)**:你先卡阈值挑出一批显著基因(如 [L6](../06-deseq2/) 的 DEGs),问"这批基因里某通路是不是出现得比随机更多"。
- **GSEA(Gene Set Enrichment Analysis)**:不卡阈值,把**所有**基因按 log2FC 排序,问"某通路的基因是不是整体集中在排序的一端"。

> 一句话直觉:ORA 用"显著基因名单",GSEA 用"全基因的排名"。两者互补,论文里常一起报。

## 2. 准备基因数据

为了能直接跑,用 `clusterProfiler`/`DOSE` 自带的示例 `geneList`——一个**按 log2FC 从高到低排好序、名字是 Entrez ID 的数值向量**。真实场景里它就来自 [L6](../06-deseq2/) 的结果。

```r
data(geneList, package = "DOSE")
head(geneList)                       # 名字=ENTREZID,值=log2FC,已降序
gene <- names(geneList)[abs(geneList) > 2]  # 取 |log2FC|>2 的显著基因,供 ORA 用
length(gene)
```

## 3. 基因 ID 转换 bitr

各数据库认的 ID 不同:GO/KEGG 富集常要 **Entrez ID**,而你手里可能是基因名(SYMBOL)。用 `bitr()` 转换:

```r
ids <- bitr(c("TP53", "BRCA1", "EGFR"),
            fromType = "SYMBOL", toType = "ENTREZID",
            OrgDb = org.Hs.eg.db)
ids
```

> ⚠️ 转换一定会**丢掉一部分**匹配不上的基因(过时/别名/非编码),`bitr` 会提示丢了百分之几,属正常。

## 4. ORA:GO 与 KEGG 富集

```r
# GO 富集(BP = 生物学过程;还可选 MF 分子功能、CC 细胞组分)
ego <- enrichGO(gene = gene, OrgDb = org.Hs.eg.db,
                keyType = "ENTREZID", ont = "BP",
                pAdjustMethod = "BH", qvalueCutoff = 0.05,
                readable = TRUE)            # readable=TRUE 把 ID 转回基因名
head(ego)

# KEGG 通路富集(hsa = 人)
ekegg <- enrichKEGG(gene = gene, organism = "hsa")
head(ekegg)
```

## 5. GSEA:用全基因排名

```r
gse <- gseGO(geneList = geneList, OrgDb = org.Hs.eg.db,
             ont = "BP", pvalueCutoff = 0.05)
head(gse)

gkegg <- gseKEGG(geneList = geneList, organism = "hsa")
```

> `geneList` 必须是**已降序排好、带 Entrez 名字**的数值向量,这是 GSEA 最容易出错的地方。

## 6. 可视化

```r
dotplot(ego, showCategory = 15)              # 点图:最常用,点大小=基因数,颜色=显著性
barplot(ego, showCategory = 15)              # 条形图
gseaplot2(gse, geneSetID = 1, title = gse$Description[1])  # 经典 GSEA 富集曲线
```

## 7. 写进论文

富集结果表(`as.data.frame(ego)`)里,通常报告:通路名(Description)、富集到的基因数(Count)、校正 p 值(p.adjust)、以及该通路里命中的基因(geneID)。结果部分一般这样写:"差异基因显著富集于 XX、YY 通路(p.adjust < 0.05),提示……"。

```r
write.csv(as.data.frame(ego), "GO_enrichment.csv")
```

## 小结

| 你想做 | 用什么 |
|--------|--------|
| ID 转换 | `bitr(..., fromType, toType, OrgDb)` |
| GO 富集(ORA) | `enrichGO(gene, ont="BP", ...)` |
| KEGG 富集(ORA) | `enrichKEGG(gene, organism="hsa")` |
| GSEA | `gseGO()` / `gseKEGG()`(输入排好序的 geneList) |
| 可视化 | `dotplot()` / `barplot()` / `gseaplot2()` |

下一课 [Lesson 9 · 生存分析](../09-survival/),换一个维度:这些基因和病人**预后**有没有关系。

## 延伸 Further reading

- clusterProfiler 官方书(强烈推荐):<https://yulab-smu.top/biomedical-knowledge-mining-book/>

## 常见报错 Troubleshooting

- `--> No gene can be mapped` → 基因 ID 类型不对,多半要先 `bitr()` 转成 ENTREZID。
- `enrichKEGG` 联网失败 → KEGG 富集需要联网取通路库;网络不稳时多试或换时段。
- GSEA 结果为空 → `geneList` 没排序、或名字不是 Entrez ID;检查 `head(geneList)`。
- `org.Hs.eg.db` 找不到 → 没装/没加载这个注释包(`BiocManager::install("org.Hs.eg.db")`)。
