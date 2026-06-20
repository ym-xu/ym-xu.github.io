---
title: "Lesson 7 · 逻辑回归 Logistic Regression"
summary: "预测一个是/否结局(如晚期毒性):优势比 OR、单/多因素、ROC 曲线。"
weight: 70
toc: true
---

当结局是**二分类**(发生 / 不发生),比如"是否出现晚期阴道毒性",而你想知道哪些因素(剂量、年龄、分期)能预测它、并同时**校正**多个因素时,就要用**逻辑回归 logistic regression**。它给出的是**优势比 OR**。

> 🔬 这正是放疗剂量学研究里最常见的分析之一——"哪些剂量-体积参数预测毒性"。

## 这一课你会做出什么

- 用逻辑回归估计某因素对"是/否"结局的优势比 OR
- 做单因素 → 多因素分析,用 `gtsummary` 出 OR 表
- 画 ROC 曲线、算 AUC,评估模型区分能力

## 0. 准备:加载包 + 造数据

```r
# install.packages(c("gtsummary", "pROC", "broom"))
library(gtsummary); library(pROC); library(broom)

set.seed(1)
n <- 150
dat <- data.frame(
  d2cc = rnorm(n, 80, 8),                       # 膀胱 D2cc(Gy)
  age  = rnorm(n, 66, 8),
  ebrt = rbinom(n, 1, 0.6)                       # 是否加外照射 0/1
)
# 毒性概率随 D2cc 升高而升高(造一个有信号的结局)
lp <- -10 + 0.11 * dat$d2cc + 0.5 * dat$ebrt
dat$late_tox <- rbinom(n, 1, plogis(lp))         # 0/1 结局
```

## 1. 为什么不能用线性回归

结局只有 0/1,线性回归会预测出 <0 或 >1 的"概率",没意义。逻辑回归把概率经 logit 变换后再建线性模型,保证预测落在 0–1 之间,并天然输出**优势比**。

## 2. 单因素逻辑回归

```r
fit1 <- glm(late_tox ~ d2cc, data = dat, family = binomial)  # family=binomial 是关键
summary(fit1)
exp(cbind(OR = coef(fit1), confint(fit1)))                    # 系数取指数 → OR + 95%CI
```

**怎么读 OR**(回顾 [Lesson 4](../04-stats-concepts/)):`d2cc` 的 OR ≈ 1.12 表示 **D2cc 每升高 1 Gy,毒性的优势增加约 12%**。OR>1 风险升高,<1 降低,95% CI 跨过 1 即不显著。

## 3. 多因素:同时校正

单因素显著的因素,放进多因素模型看是否依然成立(排除混杂):

```r
fit2 <- glm(late_tox ~ d2cc + age + ebrt, data = dat, family = binomial)

tbl_regression(fit2, exponentiate = TRUE) %>%  # exponentiate=TRUE 直接出 OR 表
  bold_p()
```

`tbl_regression(exponentiate = TRUE)` 出的就是论文里那张标准的"多因素逻辑回归 OR 表"(变量、OR、95%CI、p)。

## 4. 模型能区分吗:ROC 与 AUC

OR 显著不代表模型预测得准。用 ROC 曲线 / AUC 评估区分能力:

```r
dat$prob <- predict(fit2, type = "response")   # 每个病人的预测概率
roc_obj <- roc(dat$late_tox, dat$prob)
auc(roc_obj)                                    # AUC:0.5=瞎猜,1=完美;0.7–0.8 还行
plot(roc_obj, print.auc = TRUE)
```

> 🔬 剂量学预测毒性的论文里,常用 ROC 找一个**剂量阈值**(如 D2cc 的最佳切点)。可用 `coords(roc_obj, "best")` 取最佳阈值——但 ⚠️ 数据驱动的"最佳切点"容易过拟合,需在独立数据验证(同 [Lesson 4](../04-stats-concepts/) 对"最优切点"的提醒)。

## 5. 一条经验法则:别塞太多变量

逻辑回归有个 **EPV(events per variable)** 经验:**每个自变量至少要 ~10 个事件**。如果只有 20 个毒性事件,模型最多放 2 个自变量,否则结果不稳、OR 乱跳。样本小就别贪多。

## 小结

| 你想做 | 用什么 |
|--------|--------|
| 二分类结局建模 | `glm(y ~ x, family = binomial)` |
| 系数转 OR | `exp(coef())` / `tbl_regression(exponentiate = TRUE)` |
| 多因素校正 | `glm(y ~ x1 + x2 + ...)` |
| 区分能力 | `pROC::roc()` + `auc()` |
| 找阈值 | `coords(roc, "best")`(慎用) |

下一课 [Lesson 8 · 相关与线性回归](../08-linear-regression/):当结局是**连续**的(而非是/否)时该怎么建模。

## 延伸 Further reading

- gtsummary 回归表教程:<https://www.danieldsjoberg.com/gtsummary/articles/tbl_regression.html>

## 常见报错 Troubleshooting

- 忘了 `family = binomial` → 默认成了线性回归(高斯),OR 就错了。
- `glm.fit: fitted probabilities numerically 0 or 1 occurred` → 完全分离(某变量完美预测结局),多见于小样本;减少变量或检查数据。
- `confint()` 很慢/报错 → 样本小时正常;可用 `confint.default()` 取近似 CI。
- OR 大得离谱(几百万) → 多半是完全分离或样本太小,见 EPV 规则。
