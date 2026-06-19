---
title: "Lesson 8 · 相关与线性回归 Correlation & Linear Regression"
summary: "连续结局的建模:相关系数、散点+拟合线、线性回归与剂量-反应关系。"
weight: 80
toc: true
---

[Lesson 7](../07-logistic-regression/) 处理"是/否"结局;这一课处理**连续结局**——比如某个剂量学指标和另一个连续量之间的关系,或**剂量-反应**趋势。核心工具是相关分析和线性回归。

## 这一课你会做出什么

- 算两个连续变量的相关系数并检验
- 画散点图 + 拟合线
- 用线性回归量化关系、并校正其他变量

## 0. 准备:加载包 + 造数据

```r
library(ggplot2); library(broom); library(gtsummary)

set.seed(1)
n <- 80
dat <- data.frame(
  brachy_dose = rnorm(n, 25, 4),                 # 近距离放疗剂量(Gy)
  age = rnorm(n, 66, 8)
)
dat$d90 <- 1.4 * dat$brachy_dose + rnorm(n, 0, 3)  # D90 随处方剂量上升(造关系)
```

## 1. 相关:Pearson vs Spearman

```r
cor(dat$brachy_dose, dat$d90)                              # Pearson(线性、要求近似正态)
cor(dat$brachy_dose, dat$d90, method = "spearman")         # Spearman(秩相关,抗偏态/非线性)
cor.test(dat$brachy_dose, dat$d90)                         # 带 p 值和置信区间
```

相关系数 r 在 −1~1 之间:接近 ±1 强相关,接近 0 无线性关系。

> ⚠️ **相关 ≠ 因果**,而且相关只衡量**线性**关系——明显弯曲的关系 Pearson 会低估,先画散点图看形状。

## 2. 散点图 + 拟合线

```r
ggplot(dat, aes(x = brachy_dose, y = d90)) +
  geom_point(alpha = .6) +
  geom_smooth(method = "lm") +          # 线性拟合 + 置信带
  labs(x = "Brachy dose (Gy)", y = "D90 (Gy)") +
  theme_classic()
```

> 用了 [Lesson 3](../03-ggplot/) 的 ggplot;`method = "lm"` 画的就是下面线性回归的那条线。

## 3. 线性回归

```r
fit <- lm(d90 ~ brachy_dose, data = dat)
summary(fit)
```

输出里重点看:

- **斜率(brachy_dose 的 Estimate)**:处方剂量每升高 1 Gy,D90 平均升高多少 Gy
- **p 值**:斜率是否显著不为 0
- **R²(Multiple R-squared)**:模型解释了 y 多少比例的变异(0~1)

```r
tidy(fit, conf.int = TRUE)   # broom:整洁的系数表(含 95%CI)
```

## 4. 多元线性回归:校正其他变量

```r
fit2 <- lm(d90 ~ brachy_dose + age, data = dat)
tbl_regression(fit2)         # 论文级回归表(连续结局不用 exponentiate)
```

> 🔬 这就是剂量学论文里"在校正年龄后,处方剂量仍与 D90 显著相关"那类结论的来源。

## 5. 检查模型假设

线性回归有前提(残差正态、等方差、线性)。一行画四张诊断图:

```r
par(mfrow = c(2, 2)); plot(fit2); par(mfrow = c(1, 1))
```

- 左上 *Residuals vs Fitted*:点应随机散布,无明显弯曲(线性 & 等方差)
- 右上 *Q-Q*:点贴直线(残差正态)
- ⚠️ 若残差呈漏斗形或弯曲,说明关系非线性或方差不齐,结论要打折扣

## 小结

| 你想做 | 用什么 |
|--------|--------|
| 相关系数 + 检验 | `cor()` / `cor.test()`(Pearson/Spearman) |
| 散点 + 拟合线 | `geom_point()` + `geom_smooth(method="lm")` |
| 线性回归 | `lm(y ~ x)` + `summary()` |
| 整洁系数表 | `broom::tidy()` / `tbl_regression()` |
| 检查假设 | `plot(fit)` |

下一课 [Lesson 9 · 生存分析](../09-survival/):当结局是"发生事件的时间"(生存、复发)时,需要专门的生存模型。

## 延伸 Further reading

- `?lm`、`?cor.test` 自带帮助;broom 让模型输出可直接进表格/绘图。

## 常见报错 Troubleshooting

- 散点拟合线没出现 → `geom_smooth(method = "lm")` 拼写;数据点太少也可能不画置信带。
- R² 很低但 p 显著 → 关系真实但很弱,样本大也能显著;别只看 p,看效应大小。
- 残差图明显弯曲 → 关系非线性,考虑对变量取对数或加二次项。
- `cor()` 返回 `NA` → 有缺失值,加 `use = "complete.obs"`。
