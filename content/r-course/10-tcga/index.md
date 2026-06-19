---
title: "Lesson 10 · 公共数据 TCGA"
summary: "用 TCGAbiolinks 直接下载 TCGA 数据,复现或对照自己的结果。"
weight: 100
toc: true
---

**TCGA(The Cancer Genome Atlas)** 是最大的公共肿瘤组学数据库,涵盖三十多种癌症的表达、突变、临床随访数据。`TCGAbiolinks` 让你在 R 里直接查询、下载、整理这些数据——用来**验证自己的发现**或**在没有自己样本时做分析**都极有价值。

> ⚠️ **先读这条**:这一课会**真正联网下载数据**,一个完整癌种的表达数据可达数 GB、耗时很久。学习时**务必只下几个样本**(见第 3 步的 `barcode=`),别一上来就拉整个项目。

## 这一课你会做出什么

- 从 TCGA 查询并下载一小部分表达数据
- 整理成一个 `SummarizedExperiment`(呼应 [L5](../05-bioconductor/)),接入前面的分析流程

## 0. 准备:加载包

```r
library(TCGAbiolinks)
library(SummarizedExperiment)
```

## 1. TCGA / GDC 数据结构

TCGA 数据托管在 **GDC(Genomic Data Commons)**。查询时主要指定三件事:

- **project**:哪个癌种,如 `"TCGA-BRCA"`(乳腺癌)、`"TCGA-LUAD"`(肺腺癌)
- **data.category / data.type**:要哪类数据,如表达定量
- **workflow.type**:用哪条处理流程的结果,如 `"STAR - Counts"`

## 2. 构建查询

```r
query <- GDCquery(
  project       = "TCGA-BRCA",
  data.category = "Transcriptome Profiling",
  data.type     = "Gene Expression Quantification",
  workflow.type = "STAR - Counts"
)
```

这一步只是"问 GDC 有哪些文件",还没下载。`getResults(query)` 可以看查到的文件清单。

## 3. ⚠️ 只取几个样本再下载

直接 `GDCdownload(query)` 会下载整个 BRCA 项目(上千样本)。学习阶段先用 `barcode=` 限定几个样本:

```r
# 先拿到前几个样本的 barcode
all_barcodes <- getResults(query, cols = "cases")
few <- all_barcodes[1:6]                 # 只取 6 个样本

query_small <- GDCquery(
  project       = "TCGA-BRCA",
  data.category = "Transcriptome Profiling",
  data.type     = "Gene Expression Quantification",
  workflow.type = "STAR - Counts",
  barcode       = few                    # 关键:限定样本
)

GDCdownload(query_small)                  # 现在只下这 6 个,很快
data <- GDCprepare(query_small)           # 整理成 SummarizedExperiment
data
```

## 4. 取出表达矩阵与临床信息

`GDCprepare()` 返回的就是 [L5](../05-bioconductor/) 学过的 `SummarizedExperiment`,用法完全一样:

```r
assay(data)[1:5, 1:3]   # 表达矩阵(counts)
colData(data)           # 样本信息,含肿瘤/正常、分期、随访等
```

样本的肿瘤/正常状态可以从 barcode 或 `colData` 的样本类型字段判断(TCGA 里 `01` 开头多为肿瘤、`11` 为正常组织)。

## 5. 单独取临床随访表

要做生存分析,用专门的函数取临床表(含生存时间/结局):

```r
clin <- GDCquery_clinic(project = "TCGA-BRCA", type = "clinical")
colnames(clin)
# 关键列:vital_status(存活/死亡)、days_to_death、days_to_last_follow_up
```

## 6. 🔬 接入前面的流程

拿到 TCGA 数据后,就能复用前面所有技能:

- 把 counts 喂给 [L6 DESeq2](../06-deseq2/) 做 **肿瘤 vs 正常** 差异表达 → [L7](../07-volcano-heatmap/) 火山图/热图 → [L8](../08-enrichment/) 富集
- 用 `clin` 的 `vital_status` + `days_to_*` 构造 `status`/`time`,做 [L9 生存分析](../09-survival/),验证"自己数据里发现的基因在 TCGA 大队列里是否也和预后相关"

> 这正是很多论文的标准套路:**自有数据发现 → TCGA 验证**。

## 7. 数据使用规范

TCGA 是开放数据,但发表时需**按要求引用 TCGA / GDC**,受控数据(如部分原始测序)需另行申请。用前查一下对应项目的使用条款。

## 小结

| 你想做 | 用什么 |
|--------|--------|
| 构建查询 | `GDCquery(project, data.category, data.type, workflow.type)` |
| 只取部分样本 | 在 `GDCquery` 里加 `barcode = ...` |
| 下载 | `GDCdownload(query)` |
| 整理成对象 | `GDCprepare(query)` → SummarizedExperiment |
| 取临床随访 | `GDCquery_clinic(project, type="clinical")` |

下一课 [Lesson 11 · 用自己的数据发论文](../11-capstone/),把全流程串起来落到一篇可投稿的分析上。

## 延伸 Further reading

- TCGAbiolinks 官方教程:<https://bioconductor.org/packages/TCGAbiolinks/>
- GDC 数据门户(可在网页先浏览):<https://portal.gdc.cancer.gov/>

## 常见报错 Troubleshooting

- 下载极慢 / 中断 → 几乎都是没限定 `barcode=`,数据太大;先按第 3 步只取几个样本。
- `GDCquery` 查不到结果 → `workflow.type`/`data.type` 拼写或取值过时,去 GDC 门户对照当前可用值。
- `GDCprepare` 报错 → 确认 `GDCdownload` 已成功、文件完整;重跑下载。
- 版本差异 → TCGAbiolinks 的参数随 GDC 更新偶有变化,以官方 vignette 为准。
