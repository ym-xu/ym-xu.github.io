---
title: "Lesson 2 · 数据整理 Data Wrangling"
summary: "用 tidyverse 清洗临床表与疗效表:筛选、排序、分组、合并。"
weight: 20
toc: true
---

真实分析里,**80% 的时间花在整理数据上**,而不是跑模型。这一课用 `tidyverse` 把一张乱糟糟的临床表整理成能直接分析的样子:筛选、排序、分组统计、和另一张表合并。

> 接着 [Lesson 1](../01-setup/) 来。这一课的产出(干净的、带 `time`/`status` 的表)正是 [Lesson 9 生存分析](../09-survival/) 的输入。

## 这一课你会做出什么

- 把临床表筛选、排序、加新列、分组统计
- 正确处理缺失值 `NA`
- 把"临床表"和"疗效表"按病人 ID 合并成一张分析用表

## 0. 准备:安装并加载包

第一次用时安装一次(去掉 `#` 运行):

```r
# install.packages("tidyverse")
```

每次分析开头加载:

```r
library(tidyverse)   # 一次性加载 dplyr、tidyr、ggplot2 等一整套
```

## 1. 先造一张练习表

为了让你能直接跑(不依赖外部文件),我们用 `tribble()` 手敲一张小临床表。`tribble()` 是"按行"写的 tibble,一眼能看出表长什么样:

```r
clinical <- tribble(
  ~patient_id, ~age, ~sex,     ~stage, ~gene_expr, ~time, ~status,
  "P01",        58,  "Male",   "III",   12.4,       455,   1,
  "P02",        63,  "Female", "II",     8.1,       1010,  0,
  "P03",        47,  "Female", "I",      5.3,       NA,    0,
  "P04",        71,  "Male",   "IV",    15.9,       210,   1,
  "P05",        55,  "Female", "II",     7.7,       883,   0,
  "P06",        68,  "Male",   "III",   13.2,       310,   1,
  "P07",        49,  "Female", "I",      4.8,       1220,  0,
  "P08",        60,  "Male",   "IV",    16.5,       145,   1
)
clinical
```

> **tibble** 是 tidyverse 版的 data.frame,打印更友好、行为更可控。`status`:1=事件(死亡/复发),0=删失([Lesson 4](../04-stats-concepts/) 会细讲)。

## 2. 管道符 `%>%`:把"然后"连起来

管道符 `%>%`(或 R 自带的 `|>`)把左边的结果"喂"给右边的函数,读作"然后"。下面两种写法等价:

```r
summary(clinical$age)            # 传统写法,从里往外读
clinical$age %>% summary()       # 管道写法,从左往右读:age,然后求概况
```

管道的好处是能把多步操作串成一条自然的句子,后面你会一直用到。

## 3. dplyr 五个核心动词

整理数据 90% 靠这五个动词:

```r
# filter:按条件筛行
clinical %>% filter(stage == "IV")              # 只要 IV 期

# select:挑列
clinical %>% select(patient_id, age, gene_expr) # 只看这几列

# mutate:加新列 / 改列
clinical %>% mutate(expr_high = gene_expr > 10) # 加一列"是否高表达"

# arrange:排序
clinical %>% arrange(desc(gene_expr))           # 按表达量从高到低

# group_by + summarise:分组统计
clinical %>%
  group_by(stage) %>%
  summarise(n = n(), mean_age = mean(age))      # 每个分期的人数与平均年龄
```

> 🔬 "按基因表达高/低分组、再比较两组" 是肿瘤研究的常见操作。但 ⚠️ 别随手按中位数二分——代价见 [Lesson 4](../04-stats-concepts/)。

## 4. 处理缺失值 NA

`P03` 的 `time` 是 `NA`(缺失)。缺失值不处理会让很多计算直接返回 `NA`:

```r
mean(clinical$time)                 # 结果是 NA!
mean(clinical$time, na.rm = TRUE)   # 加 na.rm=TRUE 才会忽略缺失
clinical %>% filter(is.na(time))    # 找出哪些行缺失
clinical %>% drop_na(time)          # 丢掉 time 缺失的行
```

> ⚠️ 别盲目删缺失值——先搞清楚为什么缺。删之前最好统计一下缺多少、缺在哪。

## 5. 合并两张表 `left_join`

真实场景:临床信息在一张表,疗效/分子数据在另一张表,靠**病人 ID** 对应。先造第二张表:

```r
response <- tribble(
  ~patient_id, ~response,
  "P01",       "PR",
  "P02",       "CR",
  "P04",       "PD",
  "P05",       "SD"
)

merged <- clinical %>% left_join(response, by = "patient_id")
merged
```

`left_join` 以左表(clinical)为准,把 response 按 `patient_id` 贴上来;左表里没有匹配的病人,response 列填 `NA`。

> 🔬 把"临床表 + 疗效/剂量学数据"合并成一张分析表,是动手分析前的标准准备步骤。

## 6. 串成一条流水线

把上面几步用管道连起来,就是一句完整的"数据整理":

```r
analysis_tbl <- clinical %>%
  filter(!is.na(time)) %>%                       # 去掉生存时间缺失的
  mutate(expr_group = if_else(gene_expr > median(gene_expr),
                              "High", "Low")) %>% # 加分组列
  left_join(response, by = "patient_id") %>%      # 合并疗效
  arrange(stage)                                  # 按分期排序
analysis_tbl
```

## 小结

| 你想做 | 用什么 |
|--------|--------|
| 按条件筛行 | `filter()` |
| 挑/去列 | `select()` |
| 加列/改列 | `mutate()` |
| 排序 | `arrange()` / `arrange(desc())` |
| 分组统计 | `group_by() %>% summarise()` |
| 处理缺失 | `is.na()` / `na.rm = TRUE` / `drop_na()` |
| 合并两张表 | `left_join(by = "id")` |

下一课 [Lesson 3 · ggplot2](../03-ggplot/),把整理好的数据画成图。

## 延伸 Further reading

- R for Data Science(免费在线书):<https://r4ds.hadley.nz/>,tidyverse 作者写的权威入门。

## 常见报错 Troubleshooting

- `could not find function "%>%"` 或 `filter()` → 没 `library(tidyverse)`。
- `Error: object 'gene_expr' not found` → 在 `filter()`/`mutate()` 里要直接写列名(不加引号、不加 `clinical$`)。
- 结果全是 `NA` → 多半是没加 `na.rm = TRUE`,见第 4 步。
- `filter()` 没筛出东西 → 判断相等要用 `==`(两个等号),写成 `=` 会报错。
