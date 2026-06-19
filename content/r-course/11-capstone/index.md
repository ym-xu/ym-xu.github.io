---
title: "Lesson 11 · 用自己的数据发论文 Capstone"
summary: "把前面所有技能用到你自己的真实数据上,走完整流程并产出可投稿的图表与可复现脚本。"
weight: 110
toc: true
---

前面十课每课练一个技能,都用的是干净的示例数据。这一课的目标完全不同:**把你自己的真实数据,从一张 Excel 走到能投稿的结果**。真正的难点不在 R 语法,而在这一课讲的判断和流程。

## 这一课你会做出什么

- 一份把全流程串起来的、可一键重跑的 Quarto 分析文档
- 一组可直接放进论文的图与表

## 1. 先想清楚:研究问题与队列

动手前先写下三句话:

- **研究问题**:比如"基因 X 高表达是否与肺腺癌患者更差的预后相关?"
- **纳入/排除标准**:哪些病人算数,哪些剔除(缺关键随访、非目标亚型……)
- **样本量**:有多少例、事件(死亡/复发)有多少个。⚠️ 生存分析里**事件数**比总人数更决定统计效力,几十个事件起步才比较稳。

> 没有清晰的问题,再花哨的图也没用。这一步决定后面所有分析。

## 2. ⚠️🔬 整理你自己的数据(最难的一步)

示例数据是干净的,你的数据不是。这一步最花时间,也最容易出错:

- **从日期算生存时间**:`time` 很少现成,通常 = 末次随访/死亡日期 − 诊断/入组日期(用 [L2](../02-data-wrangling/) 的整理 + `lubridate`,见 [L9 第 9 节](../09-survival/))
- **定义 event**:死亡/复发记 `1`,随访结束仍存活/失访记 `0`(回顾 [L4](../04-stats-concepts/) 删失)
- **编码分组与协变量**:基因高低、分期、年龄、性别,整理成规范的列
- **处理缺失**:统计缺多少、缺在哪,再决定怎么处理(别盲删,[L2](../02-data-wrangling/))

```r
library(tidyverse); library(lubridate)
clinical <- read.csv("my_clinical.csv") %>%
  mutate(
    time   = as.numeric(as_date(last_followup) - as_date(dx_date)),  # 随访天数
    status = if_else(vital_status == "Dead", 1, 0)                   # 事件指示
  ) %>%
  drop_na(time, status)
```

## 3. 分析流水线:把各课串起来

按研究问题选用前面的模块,典型一条龙:

| 步骤 | 用哪一课 | 产出 |
|------|---------|------|
| 差异表达(肿瘤 vs 正常) | [L6 DESeq2](../06-deseq2/) | DEG 表 |
| 可视化 DEG | [L7 火山图/热图](../07-volcano-heatmap/) | 火山图、热图 |
| 通路解释 | [L8 富集](../08-enrichment/) | GO/KEGG/GSEA |
| 预后关联 | [L9 生存分析](../09-survival/) | KM 曲线、Cox HR 表 |
| (可选)外部验证 | [L10 TCGA](../10-tcga/) | 大队列复现 |

> 不是每篇论文都要全跑。围绕你第 1 步的问题,挑必要的模块即可。

## 4. 统计把关:别让分析"翻车"

出图容易,出**对**的结论难。投稿前自查:

- Cox 模型查了**比例风险假设**吗?(`cox.zph()`,[L9](../09-survival/) / [L4](../04-stats-concepts/))
- 单因素显著的变量,放进**多因素**校正后还成立吗?
- 基因表达是不是被你按中位数粗暴二分了?能否当连续变量?([L4 第 7 节](../04-stats-concepts/))
- p 值是不是和**效应大小 + 置信区间**一起报告的?

## 5. 发表级报告

- **Table 1(患者特征表)** 和 **回归结果表**:用 `gtsummary`(`tbl_summary()` / `tbl_regression()`),一行生成、直接复制进论文
- **KM 曲线**:用 `ggsurvfit` 出带**风险表(risk table)**的图([L9](../09-survival/))
- **方法部分怎么写**:写清楚软件与版本(`sessionInfo()`)、统计方法、阈值、检验、校正方式——审稿人会看可复现性

```r
library(gtsummary)
clinical %>%
  select(age, sex, stage, expr_group) %>%
  tbl_summary(by = expr_group) %>%       # 按高/低表达分组的 Table 1
  add_p()                                # 加组间比较 p 值
```

## 6. 可复现:用 Quarto 封装

把"数据 → 分析 → 图表"写进一个 **Quarto 文档(`.qmd`)**,而不是一堆零散脚本。好处:一键重跑、图表自动更新、审稿人/合作者要数据时直接给整份文档。

`.qmd` 文件的骨架长这样:

````markdown
---
title: "Gene X and prognosis in LUAD"
author: "Your Name"
format: html
---

## Setup
```{r}
library(tidyverse); library(DESeq2); library(survival); library(ggsurvfit); library(gtsummary)
```

## 1. Data preparation
```{r}
clinical <- read.csv("my_clinical.csv") %>% mutate(...)   # 见第 2 节
```

## 2. Differential expression
```{r}
# DESeq2 流程(L6)
```

## 3. Survival analysis
```{r}
# KM + Cox(L9)
```

## 4. Session info
```{r}
sessionInfo()    # 记录所有包版本,可复现的关键
```
````

渲染成报告:

```r
# 在 RStudio 点 "Render" 按钮,或命令行:
# quarto render analysis.qmd
```

> Quarto 是 R Markdown 的现代继任者,RStudio 自带支持。把分析写成 `.qmd`,你交出去的就不只是"几张图",而是"可被重跑、可被检验的完整证据链"。

## 交付物 Deliverable

- 一份可复现的 `analysis.qmd` + 渲染出的 HTML/PDF 报告
- 一组可投稿的图(火山图/热图/KM 曲线)和表(Table 1 / HR 表)
- 方法部分草稿 + `sessionInfo()` 记录的版本信息

## 小结

| 阶段 | 关键 |
|------|------|
| 设计 | 研究问题、纳排标准、事件数 |
| 整理 | 算 time/status、编码、处理缺失 |
| 分析 | 串起 L6→L7→L8→L9(按需) |
| 把关 | 假设检验、多因素、效应+CI |
| 报告 | gtsummary 表、ggsurvfit 图、方法写清 |
| 复现 | Quarto `.qmd` + sessionInfo |

## 延伸 Further reading

- Quarto 官网:<https://quarto.org/>
- gtsummary:<https://www.danieldsjoberg.com/gtsummary/>
- 回到起点,把这套流程套到你自己的第一个真实问题上——这才是这门课真正的"毕业作业"。

## 常见报错 Troubleshooting

- Quarto 渲染失败 → 先确认每个代码块能单独跑通;看报错定位到具体哪块。
- 报告里图缺失 → 确认产生图的代码块没设 `eval: false`,且图被打印出来。
- 数据路径在别人电脑上失效 → 用相对路径、把数据和 `.qmd` 放一起,别用绝对路径。
- 结果无法复现 → 一定保留 `sessionInfo()`;随机过程加 `set.seed()`。
