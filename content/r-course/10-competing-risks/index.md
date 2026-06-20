---
title: "Lesson 10 · 竞争风险 Competing Risks"
summary: "当患者可能死于其他原因:cause-specific survival、累积发生率与 Fine-Gray 回归。"
weight: 100
toc: true
---

肿瘤患者(尤其高龄、放疗人群)可能**死于非肿瘤原因**——这构成一个**竞争事件**。这时普通 Kaplan-Meier 会**高估**你关心的事件(如肿瘤特异性死亡)。这一课讲竞争风险分析,对标 [Emily Zabor 教程](https://www.emilyzabor.com/survival-analysis-in-r.html) 的 **Part 3**,衔接 [Lesson 9](../09-survival/)。

> 🔬 当你的论文报告 **cause-specific survival**(肿瘤特异性生存)而不是总生存时,背后就是这套方法。

## 这一课你会做出什么

- 用累积发生率(cumulative incidence)正确估计竞争事件下的发生概率
- 画分组的累积发生率曲线 + Gray 检验
- 用 Fine-Gray 回归估计竞争风险下的效应

## 0. 准备:加载包 + 数据

```r
# install.packages(c("tidycmprsk", "ggsurvfit", "gtsummary"))
library(tidycmprsk); library(ggsurvfit); library(gtsummary)

trial <- tidycmprsk::trial   # 用 tidycmprsk 自带的示例临床试验数据
levels(trial$death_cr)        # 结局因子:"censor" / "death from cancer" / "death other causes"
```

> ⚠️ 这里特意写成 `tidycmprsk::trial` 而不是 `data(trial)`:`gtsummary` 也有一个同名数据集 `trial`(但没有 `death_cr` 列),后加载会把它**屏蔽**掉,导致后面报 `object 'death_cr' not found`。加上 `包名::` 就能精确指定,避免这类命名冲突——这是 R 里常见的坑。

> `death_cr` 是一个**带竞争事件的因子**:第一层必须是删失("censor"),其余是各类事件。你自己的数据要先整理成这种因子(肿瘤死亡 / 他因死亡 / 存活)。

## 1. 为什么不能直接用 KM

普通 KM 估"肿瘤死亡"时,会把"他因死亡"当成删失——等于假设这些人将来还可能发生肿瘤死亡,于是**高估**肿瘤死亡概率。竞争风险分析改用**累积发生率函数(CIF)**,正确处理"被对手事件抢先"的情况。

- **Cause-specific hazard**:在仍存活的人中,某类死亡的瞬时风险(对应 cause-specific Cox)
- **Subdistribution hazard(Fine-Gray)**:直接对累积发生率建模(对应 `crr`)

## 2. 累积发生率(整体)

```r
cuminc(Surv(ttdeath, death_cr) ~ 1, data = trial)
```

输出给出各时点"肿瘤死亡"和"他因死亡"的累积发生率(带 95%CI)。

## 3. 分组曲线 + Gray 检验

```r
cuminc(Surv(ttdeath, death_cr) ~ trt, data = trial) %>%
  ggcuminc(outcome = "death from cancer") +     # 画"肿瘤死亡"这一事件
  add_confidence_interval() +
  add_risktable() +
  labs(x = "Months", y = "Cumulative incidence of cancer death")
```

组间差异检验(Gray's test,竞争风险版的 log-rank):

```r
cuminc(Surv(ttdeath, death_cr) ~ trt, data = trial) %>%
  tbl_cuminc(times = c(12, 24), outcome = "death from cancer") %>%
  add_p()        # Gray 检验的 p 值
```

## 4. Fine-Gray 回归(subdistribution hazard)

校正多个因素、估计对累积发生率的效应:

```r
crr(Surv(ttdeath, death_cr) ~ age + trt, data = trial) %>%
  tbl_regression(exponentiate = TRUE)   # 给出 subdistribution HR + 95%CI
```

## 5. Cause-specific Cox(另一种思路)

如果你关心的是"病因机制",而非绝对发生概率,可用 cause-specific Cox:把他因死亡当作删失,套用 [Lesson 9](../09-survival/) 的 `coxph`:

```r
library(survival)
trial$cancer_death <- as.numeric(trial$death_cr == "death from cancer")
coxph(Surv(ttdeath, cancer_death) ~ age + trt, data = trial) %>%
  tbl_regression(exponentiate = TRUE)
```

> **怎么选**:想报告"实际发生概率/做预测" → Fine-Gray;想讲"病因学/机制" → cause-specific Cox。论文里两者常都报,并说明用了哪种。

## 小结

| 你想做 | 用什么 |
|--------|--------|
| 累积发生率 | `cuminc(Surv(time, status_factor) ~ 1)` |
| 分组曲线 | `cuminc(~ group) %>% ggcuminc(outcome=)` |
| 组间检验(Gray) | `tbl_cuminc() %>% add_p()` |
| Fine-Gray 回归 | `crr() %>% tbl_regression(exponentiate = TRUE)` |
| Cause-specific Cox | 把他因事件当删失,用 `coxph()` |

下一课 [Lesson 11 · 用自己的数据发论文](../11-capstone/),把全流程串到你自己的回顾性队列上。

## 延伸 Further reading

- Zabor 教程 **Part 3 · Competing Risks**:<https://www.emilyzabor.com/survival-analysis-in-r.html>
- tidycmprsk:<https://mskcc-epi-bio.github.io/tidycmprsk/>

## 常见报错 Troubleshooting

- `death_cr` 不是因子 / 第一层不是删失 → 整理成因子,删失放第一层:`factor(x, levels = c("censor", ...))`。
- `ggcuminc` 画出多条线分不清 → 用 `outcome =` 指定只画某一类事件。
- `crr` 报错 → 确认结局是带竞争事件的因子;协变量无缺失。
- 找不到 `cuminc`/`crr` → 没 `library(tidycmprsk)`。
