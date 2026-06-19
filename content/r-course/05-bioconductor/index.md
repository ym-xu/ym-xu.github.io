---
title: "Lesson 5 · 进入 Bioconductor"
summary: "认识组学数据的标准容器 SummarizedExperiment,装好分析全家桶。"
weight: 50
toc: true
---

从这一课起进入**生物信息学主线**。生信的包大多不在普通的 CRAN 上,而在 **Bioconductor**——一个专为基因组数据维护的包仓库。这一课装好后面要用的全套工具,并认识组学数据的标准"容器" `SummarizedExperiment`。

> 前面 [L1–L4](../01-setup/) 是通用基础;从这里到 [L10](../10-tcga/) 是 RNA-seq / 表达分析的主线。

## 这一课你会做出什么

- 装好课程后续要用的 Bioconductor 包
- 看懂并能探索一个 `SummarizedExperiment` 对象(表达矩阵 + 样本信息 + 基因信息)

## 1. Bioconductor 和 CRAN 有什么区别

- **CRAN**:R 的通用包仓库(tidyverse、ggplot2 都在这,用 `install.packages()` 装)。
- **Bioconductor**:专门的生信包仓库(DESeq2、clusterProfiler、TCGAbiolinks 都在这),用专门的 `BiocManager` 来装,**版本统一发布**,保证一堆生信包之间互相兼容。

## 0. 准备:装好 BiocManager 和全套包

先装"包管理器"(只需一次):

```r
# install.packages("BiocManager")
```

再用它安装本课程后续要用的包(第一次会比较慢):

```r
# BiocManager::install(c(
#   "airway",          # 示例 RNA-seq 数据
#   "DESeq2",          # 差异表达(L6)
#   "EnhancedVolcano", # 火山图(L7)
#   "ComplexHeatmap",  # 热图(L7)
#   "clusterProfiler", # 功能富集(L8)
#   "org.Hs.eg.db",    # 人类基因注释(L8)
#   "TCGAbiolinks"     # TCGA 数据(L10)
# ))
```

> ⚠️ 第一次安装可能要十几分钟甚至更久,必须联网;某些系统可能要先装编译工具。耐心等,中途让你 `Update all/some/none? [a/s/n]` 时一般输入 `a` 回车即可。

加载并确认版本:

```r
library(BiocManager)
BiocManager::version()   # 看你的 Bioconductor 版本
```

## 2. 为什么需要 SummarizedExperiment

做表达分析时你手里通常有三样东西:

1. **表达矩阵**:行是基因、列是样本的一张大表(counts)
2. **样本信息**:每个样本是肿瘤还是正常、分期、性别……(临床表)
3. **基因信息**:每行对应哪个基因、在染色体哪个位置

如果分开存成三个对象,一旦你筛掉几个样本或基因,三者就**对不齐**了,极易出错。`SummarizedExperiment` 把这三样**打包在一个对象里**,你筛样本/基因时三者同步变化,行列永远对齐。

> 🔬 这是 Bioconductor 里表达数据的通用"集装箱",[L6 的 DESeq2](../06-deseq2/) 和 [L10 的 TCGA 数据](../10-tcga/) 都用它。

## 3. 上手探索:airway 数据

`airway` 是一份标准的 RNA-seq 示例数据(气道平滑肌细胞,4 个个体 ± 地塞米松处理,共 8 个样本)。它本身不是肿瘤数据,但**分析流程和肿瘤 vs 正常完全一样**——学会它,换成你的数据照搬即可。

```r
library(airway)
data(airway)
airway              # 打印对象概况:多少基因 × 多少样本
dim(airway)         # 维度:行(基因)× 列(样本)
```

## 4. 取出三个部分

```r
assay(airway)[1:5, 1:3]   # 表达矩阵:前 5 个基因、前 3 个样本的 counts
assayNames(airway)        # 这个对象里有哪些矩阵(这里是 "counts")
colData(airway)           # 样本信息:每个样本的处理(dex)、细胞系(cell)等
rowRanges(airway)         # 基因信息:每个基因的 ID 和染色体位置
```

重点看 `colData(airway)`:其中 `dex` 列就是分组(`untrt` 未处理 / `trt` 处理),`cell` 是细胞系。**在你自己的数据里,这一列就是"肿瘤 / 正常"。**

## 5. 像数据框一样筛选

`SummarizedExperiment` 支持 `[行, 列]` 取子集,且三部分同步:

```r
airway[, airway$dex == "trt"]   # 只保留处理组样本,基因信息/表达矩阵自动跟着筛
```

> 这种"筛一处、处处同步"正是用它而不是三个散表的核心好处。

## 6. 你自己的数据怎么进来

如果你有一个 counts 矩阵和一张样本表,可以手动组装:

```r
# se <- SummarizedExperiment(
#   assays  = list(counts = my_counts_matrix),  # 基因 × 样本
#   colData = my_sample_table                   # 行 = 样本,要和矩阵列一一对应
# )
```

不过实际做差异表达时,[L6 的 DESeq2](../06-deseq2/) 会用 `DESeqDataSetFromMatrix()` 帮你直接从矩阵 + 样本表建好,通常不用手动调 `SummarizedExperiment()`。

## 小结

| 你想做 | 用什么 |
|--------|--------|
| 装生信包 | `BiocManager::install("包名")` |
| 看 Bioconductor 版本 | `BiocManager::version()` |
| 表达矩阵 | `assay(se)` |
| 样本信息(分组在这) | `colData(se)` |
| 基因信息 | `rowRanges(se)` / `rowData(se)` |
| 取子集 | `se[基因, 样本]` |

下一课 [Lesson 6 · DESeq2](../06-deseq2/),用 airway 跑出差异表达基因。

## 延伸 Further reading

- Bioconductor 官网:<https://bioconductor.org/>
- SummarizedExperiment 教程:<https://bioconductor.org/packages/SummarizedExperiment/>

## 常见报错 Troubleshooting

- `there is no package called 'BiocManager'` → 先 `install.packages("BiocManager")`。
- `installation path not writeable` → 没有写权限,按提示创建个人库,或以管理员身份运行。
- 安装巨慢 / 卡住 → 换 Bioconductor 镜像;耐心等首次编译;确认网络通畅。
- `could not find function "assay"` → 没 `library(airway)` 或 `library(SummarizedExperiment)`。
