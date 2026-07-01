---
title: "Lesson 13 · 用自己的数据发论文 Capstone"
summary: "把前面所有技能用到你自己的回顾性临床队列上,走完整流程并产出可投稿的图表与可复现脚本。"
weight: 110
toc: true
---

前面每课练一个技能,用的都是干净的示例数据。这一课目标不同:**把你自己的回顾性临床队列,从一张 Excel 走到能投稿的结果**。真正的难点不在 R 语法,而在这一课讲的判断和流程。

## 这一课你会做出什么

- 一份把全流程串起来的、可一键重跑的 Quarto 分析文档
- 一组可直接放进论文的表(Table 1 / 回归表)和图(KM / 累积发生率曲线)

## 1. 先想清楚:研究问题与队列

动手前先写下三句话:

- **研究问题**:比如"在术后子宫内膜癌患者中,膀胱 D2cc 是否预测晚期毒性?"或"两种近距离放疗方案的 cause-specific survival 是否不同?"
- **纳入/排除标准**:哪些病人算数,哪些剔除(缺关键随访、非目标分期……)
- **样本量与事件数**:有多少例、结局事件(毒性 / 死亡 / 复发)有多少个。⚠️ 回顾性研究里**事件数**决定能放几个自变量(逻辑/Cox 的 EPV 规则,见 [Lesson 7](../07-logistic-regression/))。如果像多数病例系列那样只有十来例,先读 [L12 极小样本的诚实分析](../12-small-samples/):它会告诉你这一篇大概率应停在"描述 + KM"。

## 2. ⚠️🔬 整理你自己的数据(最难的一步)

示例数据是干净的,你的病历导出表不是。这步最花时间,也最容易出错:

- **从日期算随访/生存时间**:`time` 很少现成,通常 = 末次随访/事件日期 − 治疗开始日期(用 [Lesson 2](../02-data-wrangling/) 的整理 + `lubridate`,见 [Lesson 9 第 9 节](../09-survival/))
- **定义结局**:毒性是/否(0/1)、死亡/删失、若区分死因则整理成竞争风险因子(肿瘤死亡/他因死亡/存活,见 [Lesson 10](../10-competing-risks/))
- **编码变量**:治疗方案、FIGO 分期、剂量学指标(EBRT、近距离剂量、D90、D2cc)、年龄,整理成规范的列。⚠️ 剂量先用 [L11 EQD2/BED](../11-radiobiology-eqd2/) 换成 EQD2 再当协变量——不同分次方案的病人才可比
- **处理缺失**:统计缺多少、缺在哪,再决定怎么处理(别盲删,[Lesson 2](../02-data-wrangling/))

```r
#| norun
library(tidyverse); library(lubridate)
cohort <- read.csv("my_cohort.csv") %>%
  mutate(
    time   = as.numeric(as_date(last_followup) - as_date(rt_start)) / 30.44,  # 月
    status = if_else(vital_status == "Dead", 1, 0),
    late_tox = factor(late_tox, levels = c(0,1), labels = c("No","Yes"))
  ) %>%
  drop_na(time, status)
```

> 上面这块读的是**你自己电脑上的文件**,所以网页里没有 Run 按钮。为了能在浏览器里把整条流水线跑通,下面用一份**合成队列**代替——真实分析时把它换成上面的 `read.csv` 即可。

```r
library(tidyverse)
set.seed(7)
n <- 120
cohort <- tibble(
  arm    = factor(sample(c("EBRT+BT", "BT only"), n, replace = TRUE)),
  age    = round(rnorm(n, 66, 9)),
  figo   = factor(sample(c("I","II","III"), n, replace = TRUE), levels = c("I","II","III")),
  d2cc   = round(rnorm(n, 80, 8), 1),                 # 膀胱 D2cc(Gy)
  time   = round(rexp(n, 1/40)) + 1,                  # 随访月数
  status = rbinom(n, 1, 0.4)                          # 1=事件,0=删失
)
cohort$late_tox <- factor(rbinom(n, 1, plogis(-9 + 0.10 * cohort$d2cc)),
                          levels = c(0, 1), labels = c("No", "Yes"))
head(cohort)
```

> ⚠️ 这份合成数据里,只有 `late_tox` 和 `d2cc` 之间有意造了关联;生存(`time`/`status`)和分组是**独立随机**生成的。所以下面 KM 曲线基本重叠、组间 p 值不显著都属正常——这里的目的是**跑通整条流程**,不是看出真实差异。换成你自己的数据,才会看到真实效应。

## 3. 分析流水线:把各课串起来

按研究问题选用前面的模块,临床论文典型一条龙:

| 步骤 | 用哪一课 | 产出 |
|------|---------|------|
| 剂量换算(分次→EQD2) | [L11 EQD2/BED](../11-radiobiology-eqd2/) | 可比的 EQD2 列 |
| 患者基线特征 | [L5 Table 1](../05-table-one/) | Table 1 |
| 组间比较(方案/剂量) | [L6 组间比较](../06-group-comparisons/) | 检验 p 值 |
| 毒性预测因子(是/否) | [L7 逻辑回归](../07-logistic-regression/) | OR 表、ROC |
| 剂量-反应/连续结局 | [L8 线性回归](../08-linear-regression/) | 回归表 |
| 生存/预后 | [L9 生存分析](../09-survival/) | KM 曲线、Cox HR 表 |
| 死因竞争(cause-specific) | [L10 竞争风险](../10-competing-risks/) | 累积发生率曲线、Fine-Gray |

> 不是每篇都要全跑。围绕你第 1 步的问题,挑必要的模块即可。

## 4. 统计把关:别让分析"翻车"

出图容易,出**对**的结论难。投稿前自查:

- Cox 模型查了**比例风险假设**吗?(`cox.zph()`,[L9](../09-survival/) / [L4](../04-stats-concepts/))
- 单因素显著的变量,放进**多因素**校正后还成立吗?事件数够放这么多变量吗(EPV,[L7](../07-logistic-regression/))?
- 连续变量(如剂量)是不是被粗暴二分了?能否当连续变量?([L4 第 7 节](../04-stats-concepts/))
- 有竞争事件(他因死亡)时,用了竞争风险方法、还是误用了普通 KM?([L10](../10-competing-risks/))
- p 值是不是和**效应大小 + 置信区间**一起报告的?
- 样本极小(十来例)时,是否克制住了多因素回归、比例用了**精确 CI**、分类比较用了 **Fisher**、中位随访用了 **reverse KM**、结论限定为描述性?([L12](../12-small-samples/))

## 5. 发表级报告

- **Table 1** 与**回归表**:用 `gtsummary`(`tbl_summary` / `tbl_regression`),一行生成、直接复制进论文
- **KM 曲线 / 累积发生率曲线**:用 `ggsurvfit` 出带**风险表(risk table)**的图([L9](../09-survival/) / [L10](../10-competing-risks/))
- **方法部分怎么写**:写清软件与版本(`sessionInfo()`)、统计方法、检验、阈值、校正方式——审稿人会看可复现性

在前面那份合成队列上,这条流水线的三个关键产出都能直接 **Run**:

```r
library(gtsummary)
# ① 基线表 Table 1(L5)
cohort %>%
  select(age, figo, d2cc, late_tox, arm) %>%
  tbl_summary(by = arm) %>%      # 按治疗组的 Table 1
  add_p()                        # 组间比较 p 值
```

```r
library(survival); library(ggsurvfit)
# ② 生存曲线 KM(L9)
survfit2(Surv(time, status) ~ arm, data = cohort) %>%
  ggsurvfit() +
  add_confidence_interval() +
  add_risktable() +
  labs(x = "Months", y = "Overall survival")
```

```r
library(gtsummary)
# ③ 毒性预测的逻辑回归 OR 表(L7)
glm(late_tox ~ d2cc + age, data = cohort, family = binomial) %>%
  tbl_regression(exponentiate = TRUE)
```

## 6. 可复现:用 Quarto 封装

把"数据 → 分析 → 图表"写进一个 **Quarto 文档(`.qmd`)**,而不是一堆零散脚本。好处:一键重跑、图表自动更新、审稿人/合作者要数据时直接给整份文档。

`.qmd` 文件的骨架长这样:

````markdown
---
title: "Bladder D2cc and late toxicity in endometrial cancer"
author: "Yao Qiang"
format: html
---

## Setup
```{r}
library(tidyverse); library(gtsummary); library(survival); library(ggsurvfit); library(tidycmprsk)
```

## 1. Data preparation
```{r}
cohort <- read.csv("my_cohort.csv") %>% mutate(...)   # 见第 2 节
```

## 2. Patient characteristics (Table 1)
```{r}
cohort %>% tbl_summary(by = arm) %>% add_p()
```

## 3. Predictors of late toxicity
```{r}
glm(late_tox ~ d2cc + age, data = cohort, family = binomial) %>%
  tbl_regression(exponentiate = TRUE)
```

## 4. Survival
```{r}
survfit2(Surv(time, status) ~ arm, data = cohort) %>% ggsurvfit() + add_risktable()
```

## 5. Session info
```{r}
sessionInfo()    # 记录所有包版本,可复现的关键
```
````

渲染成报告:

```r
#| norun
# 在 RStudio 点 "Render" 按钮,或命令行:
# quarto render analysis.qmd
```

> Quarto 是 R Markdown 的现代继任者,RStudio 自带支持。把分析写成 `.qmd`,你交出去的就不只是"几张图",而是"可被重跑、可被检验的完整证据链"。

## 交付物 Deliverable

- 一份可复现的 `analysis.qmd` + 渲染出的 HTML/PDF 报告
- 一组可投稿的表(Table 1 / OR / HR 表)和图(KM / 累积发生率曲线)
- 方法部分草稿 + `sessionInfo()` 记录的版本信息

## 小结

| 阶段 | 关键 |
|------|------|
| 设计 | 研究问题、纳排标准、事件数 |
| 整理 | 算 time/status、定义结局、编码、处理缺失 |
| 分析 | 串起 L5→L6→L7/L8→L9/L10(按需) |
| 把关 | 假设检验、多因素、EPV、效应+CI |
| 报告 | gtsummary 表、ggsurvfit 图、方法写清 |
| 复现 | Quarto `.qmd` + sessionInfo |

## 延伸 Further reading

- Quarto 官网:<https://quarto.org/>
- gtsummary:<https://www.danieldsjoberg.com/gtsummary/>
- 把这套流程套到你自己的第一个真实问题上——这才是这门课真正的"毕业作业"。

## 常见报错 Troubleshooting

- Quarto 渲染失败 → 先确认每个代码块能单独跑通;看报错定位到具体哪块。
- 报告里图缺失 → 确认产生图的代码块没设 `eval: false`,且图被打印出来。
- 数据路径在别人电脑上失效 → 用相对路径、把数据和 `.qmd` 放一起,别用绝对路径。
- 结果无法复现 → 一定保留 `sessionInfo()`;随机过程加 `set.seed()`。
