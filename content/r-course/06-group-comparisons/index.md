---
title: "Lesson 6 · 组间比较 Group Comparisons"
summary: "比较两组或多组:t 检验、Wilcoxon、卡方/Fisher、ANOVA,以及什么时候用哪个。"
weight: 60
toc: true
---

临床研究里最常问的一类问题:**这个指标在两组之间有没有差别?**(比如两种放疗方案的膀胱 D2cc 是否不同、两组的毒性发生率是否不同)。这一课讲清楚常用的组间比较检验,以及**什么情况用哪个**——也就是 [Lesson 5](../05-table-one/) 里 `add_p()` 背后到底在做什么。

## 这一课你会做出什么

- 判断一个连续变量是否近似正态,从而选对检验
- 比较两组(t 检验 / Wilcoxon)、多组(ANOVA / Kruskal)、分类变量(卡方 / Fisher)
- 知道每种检验的适用前提

## 0. 准备:加载包 + 造数据

```r
library(dplyr)
set.seed(1)
dat <- tibble(
  arm = rep(c("EBRT+BT", "BT only"), each = 25),
  d2cc = c(rnorm(25, 82, 6), rnorm(25, 72, 6)),         # 膀胱 D2cc(连续)
  figo = sample(c("I","II","III"), 50, replace = TRUE), # 分期
  late_tox = rbinom(50, 1, rep(c(.4, .15), each = 25))  # 晚期毒性 0/1
)
```

## 1. 先看分布:选检验的前提

连续变量该用 t 检验(要求近似正态)还是 Wilcoxon(不要求),先看分布:

```r
hist(dat$d2cc)                       # 直方图看形状
qqnorm(dat$d2cc); qqline(dat$d2cc)   # Q-Q 图,点贴直线≈正态
shapiro.test(dat$d2cc)               # 正态性检验:p>0.05 ≈ 不违反正态
```

> ⚠️ 小样本时正态性检验不敏感,**图比检验更可靠**。拿不准时优先选不要求正态的非参数检验更稳。

## 2. 两组连续变量:t 检验 vs Wilcoxon

```r
# 近似正态 → 两样本 t 检验(默认 Welch,不假设方差相等,更稳)
t.test(d2cc ~ arm, data = dat)

# 偏态 / 小样本 / 有序 → Mann-Whitney(Wilcoxon 秩和)
wilcox.test(d2cc ~ arm, data = dat)
```

看输出的 `p-value` 和**置信区间**(t 检验给均值差的 CI)。报告时别只报 p,把效应大小一起报(回顾 [Lesson 4](../04-stats-concepts/))。

> 配对数据(同一批病人前后、左右配对)用 `t.test(..., paired = TRUE)` 或 `wilcox.test(..., paired = TRUE)`。

## 3. 分类变量:卡方 vs Fisher

比较两组的毒性发生率(都是分类):

```r
tab <- table(dat$arm, dat$late_tox)   # 2×2 列联表
tab
chisq.test(tab)                       # 卡方检验
fisher.test(tab)                      # 期望计数 <5 的格子多时,用 Fisher 精确检验
```

> ⚠️ 规则:任何格子的**期望计数 < 5** 时,卡方不可靠,改用 `fisher.test()`。卡方若想看校正,`chisq.test(tab, correct = TRUE)`。

## 4. 多于两组:ANOVA / Kruskal

比较三个分期组的 D2cc:

```r
# 近似正态 → 单因素方差分析 ANOVA
fit <- aov(d2cc ~ figo, data = dat)
summary(fit)
TukeyHSD(fit)          # 显著后,两两比较看是哪几组不同

# 偏态 → Kruskal-Wallis(非参数版 ANOVA)
kruskal.test(d2cc ~ figo, data = dat)
```

> ⚠️ **别用多次 t 检验代替 ANOVA**:做的比较越多,假阳性越容易堆积。先用 ANOVA 看整体,再用 TukeyHSD 做校正后的两两比较。

## 5. 速查:什么时候用哪个

| 数据 | 两组 | 多组(>2) |
|------|------|-----------|
| 连续 · 近似正态 | t 检验 `t.test` | ANOVA `aov` + `TukeyHSD` |
| 连续 · 偏态/小样本/有序 | Wilcoxon `wilcox.test` | Kruskal `kruskal.test` |
| 分类 | 卡方 `chisq.test`(或 Fisher) | 卡方 `chisq.test` |
| 配对 | 加 `paired = TRUE` | 重复测量(进阶) |

## 小结

| 你想比较 | 用什么 |
|--------|--------|
| 看是否正态 | `hist()` / `qqnorm()` / `shapiro.test()` |
| 两组连续 | `t.test()` / `wilcox.test()` |
| 分类(率) | `chisq.test()` / `fisher.test()` |
| 多组连续 | `aov()`+`TukeyHSD()` / `kruskal.test()` |

下一课 [Lesson 7 · 逻辑回归](../07-logistic-regression/):当你想同时校正多个因素来预测一个"是/否"结局(如毒性)时,组间比较就不够了,需要回归。

## 延伸 Further reading

- 这些都是 base R `stats` 自带的函数,`?t.test` 等可直接看帮助。

## 常见报错 Troubleshooting

- `grouping factor must have exactly 2 levels`(t.test) → 分组列多于两组,改用 ANOVA,或先筛成两组。
- `Chi-squared approximation may be incorrect` → 期望计数太小,改用 `fisher.test()`。
- 公式 `y ~ group` 报错 → 确认 `group` 是因子或字符,且 `data =` 指对了数据框。
