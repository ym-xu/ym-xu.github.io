---
title: "Lesson 5 · 描述性统计与基线表 Table 1"
summary: "临床论文的第一张表:用 gtsummary 一行生成患者基线特征表(Table 1)。"
weight: 50
toc: true
---

几乎每篇临床论文的第一张表都是 **Table 1 —— 患者基线特征表**:多少人、年龄中位数、分期分布、各组是否可比。这一课用 `gtsummary` 把它一行生成,产出的就是能直接放进论文的表。

> 接着 [Lesson 2](../02-data-wrangling/) 的数据整理来。从这一课起进入临床统计主线。

## 这一课你会做出什么

- 用 base R 快速看懂一份队列的描述性统计
- 用 `gtsummary` 生成分组的 Table 1,带组间比较 p 值
- 把表导出成可放进论文的格式

## 0. 准备:安装并加载包

```r
# install.packages(c("gtsummary", "dplyr"))
library(gtsummary)
library(dplyr)
```

> 本课程的临床统计包都在 CRAN 上,用 `install.packages()` 一句就能装,不需要任何特殊仓库。

## 1. 造一份练习队列

为了能直接跑,用 `tribble()` 手敲一个小型**子宫内膜癌近距离放疗队列**(真实里你 `read.csv()` 自己的数据):

```r
cohort <- tribble(
  ~id,  ~age, ~figo,  ~ebrt, ~brachy, ~d2cc, ~late_tox, ~arm,
  "P01", 64,  "II",   45,    28,      78,            1,         "EBRT+BT",
  "P02", 71,  "III",  45,    21,      82,            1,         "EBRT+BT",
  "P03", 58,  "I",     0,    28,      71,            0,         "BT only",
  "P04", 67,  "II",   45,    28,      85,            1,         "EBRT+BT",
  "P05", 75,  "III",  45,    21,      90,            1,         "EBRT+BT",
  "P06", 60,  "I",     0,    28,      68,            0,         "BT only",
  "P07", 69,  "II",   45,    28,      80,            0,         "EBRT+BT",
  "P08", 55,  "I",     0,    28,      66,            0,         "BT only",
  "P09", 73,  "III",  45,    21,      88,            1,         "EBRT+BT",
  "P10", 62,  "II",    0,    28,      74,            0,         "BT only"
) %>%
  mutate(
    figo = factor(figo, levels = c("I","II","III")),
    late_tox = factor(late_tox, levels = c(0,1), labels = c("No","Yes")),
    arm = factor(arm)
  )
```

> `d2cc` = 膀胱受到的 2cc 剂量(剂量学常用指标);`late_tox` = 晚期毒性是否发生。

## 2. base R 的描述性统计

先用最基础的方式认识数据:

```r
summary(cohort$age)                 # 连续变量:最小/四分位/中位/最大
sd(cohort$age)                      # 标准差
median(cohort$d2cc); IQR(cohort$d2cc)  # 中位数 + 四分位距
table(cohort$figo)                  # 分类变量计数
prop.table(table(cohort$late_tox))  # 比例
```

> ⚠️ 连续变量先看分布再决定报"均值±标准差"还是"中位数(IQR)"——偏态数据用后者更稳(为什么,见 [Lesson 4](../04-stats-concepts/) 的思路,以及 [Lesson 6](../06-group-comparisons/) 的正态性检查)。

## 3. 一行生成 Table 1

`tbl_summary()` 自动判断变量类型、给出合适的统计量:

```r
cohort %>%
  select(age, figo, ebrt, d2cc, late_tox) %>%
  tbl_summary()
```

连续变量默认报"中位数(IQR)",分类变量报"n(%)"。

## 4. 按组比较的 Table 1(最常用)

临床论文里 Table 1 通常**按治疗组分列**,并加组间比较 p 值:

```r
cohort %>%
  select(age, figo, d2cc, late_tox, arm) %>%
  tbl_summary(
    by = arm,                                   # 按治疗组分列
    statistic = list(all_continuous() ~ "{mean} ({sd})")  # 连续变量改成 均值(标准差)
  ) %>%
  add_p() %>%        # 自动选检验:连续→默认 Wilcoxon(可改),分类→卡方/Fisher(详见 L6)
  add_overall() %>%  # 再加一列总体
  bold_labels()
```

> `add_p()` 会自动挑检验方法,但**你要懂它在做什么**——这正是 [Lesson 6](../06-group-comparisons/) 的内容。别把 p 值当黑箱。

## 5. 导出

```r
#| norun
tbl <- cohort %>% select(age, figo, d2cc, late_tox) %>% tbl_summary()

# 存成 Word(临床论文最常用)
tbl %>% as_gt() %>% gt::gtsave("table1.docx")
# 或转成数据框再写 CSV
tbl %>% as_tibble() %>% write.csv("table1.csv", row.names = FALSE)
```

> ⚠️ 导出 **Word(.docx)** 需要本地的 R/RStudio(底层用到 pandoc),在网页的 **Run** 里跑不了——这一步请在自己电脑上做。在浏览器里想看表格本身,直接运行上一节的 `tbl_summary()` 即可。

## 小结

| 你想做 | 用什么 |
|--------|--------|
| 连续变量概况 | `summary()` / `mean()`+`sd()` / `median()`+`IQR()` |
| 分类变量计数 | `table()` / `prop.table()` |
| 基线表 Table 1 | `tbl_summary()` |
| 按组分列 + p 值 | `tbl_summary(by=) %>% add_p()` |
| 导出 Word/CSV | `as_gt() %>% gt::gtsave()` / `as_tibble()` |

下一课 [Lesson 6 · 组间比较](../06-group-comparisons/),把 `add_p()` 背后的检验一个个讲清楚。

## 延伸 Further reading

- gtsummary 官方教程:<https://www.danieldsjoberg.com/gtsummary/>

## 常见报错 Troubleshooting

- `could not find function "tbl_summary"` → 没 `library(gtsummary)`。
- 连续变量被当成分类(出一堆 n%) → 该列其实是字符/因子,用 `as.numeric()` 转成数值。
- `add_p()` 报 group too small → 某些格子样本太少,Fisher 精确检验更合适(gtsummary 通常会自动切换)。
- 导出 Word 报错 → 需要 `gt` 包:`install.packages("gt")`。
