---
title: "Lesson 9 · 生存分析 Survival Analysis"
summary: "肿瘤研究最高频的图:Kaplan-Meier 曲线与 Cox 回归,用现代 ggsurvfit + gtsummary 出投稿级图表。"
weight: 90
toc: true
---

本课对标 [Emily Zabor《Survival Analysis in R》](https://www.emilyzabor.com/survival-analysis-in-r.html) 的 **Part 1**,用的是和它一致的现代包栈:`survival` + `ggsurvfit` + `gtsummary`。学完这一课,你应该能自己打开那份教程顺着读下去。

> 开始前最好已经过完 **Lesson 4**(删失、风险比 HR、p 值、比例风险假设这些概念),否则你跑得出结果但读不懂。

## 这一课你会做出什么

- 一条带置信区间和**风险表(risk table)**的 Kaplan-Meier 生存曲线
- 男女两组的生存曲线对比 + log-rank 检验的 p 值
- 1 年生存率、中位生存期的报告表
- 一张 Cox 回归的风险比(HR)表 —— 投稿能直接用

---

## 0. 准备:安装并加载包

只需在**第一次**用时安装一次(去掉每行开头的 `#` 运行):

```r
# install.packages(c("survival", "ggsurvfit", "gtsummary", "dplyr"))
```

每次开始分析时加载:

```r
library(survival)    # 生存分析的核心:Surv()、survfit()、coxph()
library(ggsurvfit)   # 画现代、好看的生存曲线
library(gtsummary)   # 一行生成投稿级表格
library(dplyr)       # 数据整理(Lesson 2 学过)
```

---

## 1. 认识数据:`lung`

`survival` 包自带一份**真实的肺癌数据** `lung`(NCCTG 晚期肺癌患者),正好适合练手。直接就能用:

```r
head(lung)
```

我们这节课只用到这几列:

| 列名 | 含义 |
|------|------|
| `time` | 随访时间(**天**) |
| `status` | 结局:`1` = 删失(随访结束仍存活),`2` = 死亡 |
| `sex` | 性别:`1` = 男,`2` = 女 |
| `age` | 年龄 |
| `ph.ecog` | ECOG 体力评分(0 最好,越大越差) |

⚠️ **最容易踩的坑就是 `status` 的编码**。`lung` 里用的是 `1/2`,但 R 里习惯用 `0 = 删失 / 1 = 事件`。我们先把它和 `sex` 一起整理好:

```r
lung2 <- lung %>%
  mutate(
    status = status - 1,   # 1→0(删失)、2→1(死亡=事件)
    sex = factor(sex, levels = c(1, 2), labels = c("Male", "Female"))
  )
```

> 把 `sex` 变成带标签的 factor,后面表格和图例里就会直接显示 "Male/Female" 而不是 1/2。

---

## 2. 第一步:生存对象 `Surv()`

生存分析的所有函数都不直接吃 `time` 和 `status`,而是吃一个把两者打包好的**生存对象**:

```r
Surv(lung2$time, lung2$status)[1:10]
```

输出里带 `+` 号的(如 `306  455  1010+`)就是**删失**的观测——这个病人到 1010 天时还活着,我们只知道"活过了 1010 天"。这正是 Lesson 4 讲的删失。

---

## 3. Kaplan-Meier 曲线(整体)

KM 曲线估计的是"活过某个时间点的概率 S(t)"。`~ 1` 表示不分组、看全体:

```r
survfit2(Surv(time, status) ~ 1, data = lung2) %>%
  ggsurvfit() +
  labs(x = "天数 Days", y = "总生存概率 Overall survival") +
  add_confidence_interval() +   # 加上 95% 置信区间的阴影带
  add_risktable()               # 在图下方加风险表(各时点还在随访的人数)
```

**怎么读**:曲线每往下一个台阶代表发生了一次死亡;曲线越平、越靠上,生存越好。风险表(risk table)是论文里 KM 图的标配,审稿人会看。

---

## 4. 按组比较:男 vs 女

把 `~ 1` 换成 `~ sex` 就分组了:

```r
survfit2(Surv(time, status) ~ sex, data = lung2) %>%
  ggsurvfit() +
  labs(x = "天数 Days", y = "总生存概率 Overall survival") +
  add_confidence_interval() +
  add_risktable()
```

你会看到女性(Female)的曲线在男性上方——女性生存更好。但**光看图不够,得做统计检验**(下一步)。

> 🔬 在你自己的数据里,这里的分组常常是治疗方案、FIGO 分期、或某个剂量学指标的高/低。怎么分组、为什么别随便按中位数二分,回顾 [Lesson 4](../04-stats-concepts/)。

---

## 5. x 年生存率 与 中位生存期

论文里常报"1 年生存率"和"中位生存期"。注意 `time` 是天,1 年 = `365.25`:

```r
# 1 年生存率
survfit(Surv(time, status) ~ 1, data = lung2) %>%
  tbl_survfit(times = 365.25, label_header = "**1 年生存率 (95% CI)**")

# 中位生存期(生存概率降到 0.5 的时间)
survfit(Surv(time, status) ~ 1, data = lung2) %>%
  tbl_survfit(probs = 0.5, label_header = "**中位生存期 (95% CI)**")
```

`tbl_survfit()` 直接给你带 95% 置信区间的、能复制进论文的表。

---

## 6. log-rank 检验:两组差异显著吗?

log-rank 检验回答"两条生存曲线的差异是不是统计上显著":

```r
survdiff(Surv(time, status) ~ sex, data = lung2)
```

看输出最后的 `p` 值(`lung` 里约 `p = 0.001`)。p < 0.05 通常认为差异显著——但记住 Lesson 4 说的:**显著 ≠ 重要**。

---

## 7. Cox 回归:估计风险比 HR

KM + log-rank 只能比较分组、且一次只看一个因素。**Cox 比例风险模型**能同时校正多个变量,并给出**风险比(HR)**。

单因素(只看性别):

```r
coxph(Surv(time, status) ~ sex, data = lung2) %>%
  tbl_regression(exp = TRUE)   # exp=TRUE 把系数转成 HR
```

多因素(同时校正年龄、体力评分):

```r
coxph(Surv(time, status) ~ sex + age + ph.ecog, data = lung2) %>%
  tbl_regression(exp = TRUE)
```

**怎么读 HR**(Lesson 4 讲过):

- HR < 1 → 该因素**降低**死亡风险(如 Female 的 HR ≈ 0.6,即女性死亡风险约为男性的 60%)
- HR = 1 → 没影响
- HR > 1 → **增加**风险
- 看 95% CI 是否跨过 1:跨过 1 通常就不显著

`tbl_regression()` 出的就是论文里那张标准的 HR 表(变量、HR、95% CI、p 值)。

---

## 8. 检验比例风险假设

Cox 模型有个**前提**:各组的风险比不随时间变化(比例风险)。这个前提不查就用,结论可能是错的:

```r
fit <- coxph(Surv(time, status) ~ sex + age + ph.ecog, data = lung2)
cox.zph(fit)
```

每个变量和整体(GLOBAL)都给一个 p 值:**p > 0.05 表示没有违反假设**(好事)。若某个变量 p 很小,说明它违反了比例风险,需要进阶处理(见 Zabor 教程 Part 2/4)。

---

## 9. 把它用到你自己的数据

`lung` 是干净的内置数据;你自己的数据通常是一张 Excel,真正的工作量在**把它整理成 `time` + `status`**:

1. **`status`(事件)**:死亡/复发记 `1`,随访结束仍存活/失访记 `0`。
2. **`time`(时间)**:很少现成,通常要从两个日期算。例如随访时长 =(末次随访或死亡日期)−(诊断/入组日期)。用 `lubridate`:

   ```r
   library(lubridate)
   mydata <- mydata %>%
     mutate(time = as.numeric(as_date(last_date) - as_date(dx_date)))  # 天数
   ```
3. **分组/协变量**:治疗方案、FIGO 分期、年龄、剂量等,整理成列(回顾 [Lesson 2](../02-data-wrangling/))。

整理好后,把上面所有代码里的 `lung2` 换成你的数据、列名换成你的列名,就能跑通整套流程。

---

## 小结

| 你想要 | 用什么 |
|--------|--------|
| 生存曲线 | `survfit2() %>% ggsurvfit()` |
| 比较两组 | `survdiff()`(log-rank) |
| x 年生存率 / 中位生存期 | `tbl_survfit()` |
| 多因素 + 风险比 HR | `coxph() %>% tbl_regression(exp = TRUE)` |
| 检验模型前提 | `cox.zph()` |

## 延伸 Further reading

- 本课对标 **Zabor 教程 Part 1**,强烈建议照着原文再走一遍。
- **竞争风险**(患者可能死于非肿瘤原因、cause-specific survival)是放疗/高龄人群的常见问题,单独放在 [Lesson 10](../10-competing-risks/) 讲(对应 Zabor Part 3)。
- Zabor **Part 2** 时间依赖协变量、**Part 4** 假设诊断——需要时再学。

## 常见报错 Troubleshooting

- `could not find function "survfit2"` → 没加载 `library(ggsurvfit)`。
- 曲线方向反了 / 事件数对不上 → 八成是 `status` 编码没整理(回到第 1 步)。
- HR 看起来是反的 → 确认 factor 的参照组(`levels` 第一个是参照),或 `exp = TRUE` 忘了写。
