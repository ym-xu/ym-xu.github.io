---
title: "Lesson 11 · 校准 Calibration"
summary: "模型说 30%,是不是真有三成: 校准图、O/E、校准截距和斜率、Brier 分数;加权和过拟合怎样把概率弄坏,怎样修回来。"
weight: 110
toc: true
duration: "约 80 分钟,可以分两次学: 第 0–3 节,第 4–7 节"
prereq: "本课程[第 10 课](../10-discrimination/)(区分度)和[第 6 课](../06-regularization/)(正则化)"
questions:
  - "模型说 30%,这样的病人里真有三成出事吗?"
  - "概率不准有哪几种样子,各用哪个数来量?"
  - '为了"处理类别不平衡"给事件加权,概率会怎样?'
  - "概率不准了,怎么修?"
objectives:
  - "做出校准表和校准图"
  - "算 O/E、校准截距和校准斜率,说出三个数各量什么"
  - "说明类别加权和过拟合各把概率往哪个方向弄坏"
  - "用收缩和重新校准把概率修回来"
---

第 10 课结尾做过一个试验: 把所有概率减半,AUC 一点没变。AUC 只管排序。

可是临床上用模型,用的是概率的**数值**。"这位病人换方案后有 70% 的可能降到 68 Gy 以下",医生和病人是按 70% 这个数来商量的。如果模型说 70% 的病人里其实只有 40% 降了,排序再好,这个数也在误导人。

**校准**(calibration)问的就是这件事: 模型说出来的概率,能不能照字面相信。

**练习怎么做**: 讲解用**体积优化**后是否降到 68 Gy 以下(`event`),练习换成**图形优化**(`event_gra`)。

## 0. 准备

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from cohort import make_cohort
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

def prepare(d):
    d = d.dropna()
    hi = d[d["d2cc_5mm"] >= 68].copy()
    hi["excess"] = hi["d2cc_5mm"] - 68
    hi["event"] = (hi["d2cc_vol"] < 68).astype(int)
    hi["event_gra"] = (hi["d2cc_gra"] < 68).astype(int)
    return hi

hi, new_hi = prepare(make_cohort()), prepare(make_cohort(n=5000, seed=99))
cols = ["excess", "wall_mean"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

def out_of_fold(model, data, columns, outcome):
    """每位病人被留出的那一折里,模型给她的概率。"""
    proba = cross_val_predict(model, data[columns], data[outcome], cv=cv, method="predict_proba")
    return pd.Series(proba[:, 1], index=data.index)

y = hi["event"]
p = out_of_fold(LogisticRegression(C=1e6), hi, cols, "event")
print(f"{len(hi)} 位病人,{y.sum()} 个事件;AUC {roc_auc_score(y, p):.2f}")
```

和第 10 课是同一个模型、同一批留出时的概率。区分度已经知道了(AUC 0.93)。这一课检查这些概率的数值。

## 1. 分组对一对: 校准表和校准图

一位病人要么降了,要么没降,没法说给她的"30%"对不对。能核对的是**一群**病人: 把模型给的概率差不多的病人放在一起,看这一群里实际降了几成。

```python
quarter = pd.qcut(p, 4, labels=["最低", "较低", "较高", "最高"])        # 按模型给的概率分成四组
table = pd.DataFrame({"prob": p, "event": y}).groupby(quarter, observed=True).agg(
    人数=("event", "size"), 模型说的=("prob", "mean"), 实际的=("event", "mean"), 事件数=("event", "sum"))
print(table.round(3))
```

四组各 36 位。模型说平均 0.2% 的一组,实际 0 位降了;说 1.7% 的一组,1 位(2.8%);说 19.6% 的一组,7 位(19.4%);说 73.4% 的一组,26 位(72.2%)。每一组,模型说的和实际的都对得上。

**打个比方**: 天气预报说"明天降水概率 30%"。明天要么下雨要么不下,单看这一天没法说预报对不对。但可以把一年里所有预报"30%"的日子找出来,数一数是不是有三成下了雨。校准查的就是这个。

画成图,就是**校准图**(calibration plot): 横轴是模型说的概率,纵轴是实际的比例,点落在对角线上最好。

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="Perfect calibration")
ax.plot(table["模型说的"], table["实际的"], "o-", label="Quarters of patients (36 each)")
ax.set_xlabel("Predicted probability (group mean)")
ax.set_ylabel("Observed proportion")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.show()
```

144 位病人只够分四组,而且每个点自己也不准: 36 位里 7 位,换一批病人可能是 4 位或 10 位。小样本的校准图只能看个大概,不要去读每个点的细节。

<div class="demo">

**设定**: 1500 位模拟病人。按模型说的概率从低到高分成 8 组,每组一个点: 横轴是这一组模型说的平均概率,纵轴是这一组实际出事的比例。四个按钮是四种"模型"。

**你做什么**: 依次点四个按钮,每次看点落在斜线的哪一边,再读下面的两个数。

**你会看到**: "准的": 点都在斜线上,O/E = 1.03,斜率 0.99。"整体偏高": 点全在斜线下面,O/E = 0.68,斜率还是 0.99。"说得太极端": O/E 还是 1.03,但左边的点在线上面、右边的在线下面,斜率 0.52。"说得太保守": 正好反过来,斜率 1.81。

**这说明**: 概率不准主要有两种样子。一种是整体偏了(偏高或偏低),用 O/E 量;一种是幅度不对(太极端或太保守),用校准斜率量。下面讲这两个数怎么算。

<div class="widget" data-widget="calib">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

## 2. 总体偏不偏: O/E

<div class="frame">

**背景**: 校准图要用眼睛看。先问最粗的一个问题: 模型说的事件总数,和实际的总数对得上吗?

**目标**: 一个数,说明概率整体偏高还是偏低。

**做法**: 把所有病人的概率加起来,就是模型"预期"的事件数 E(expected)。实际的事件数是 O(observed)。看 O ÷ E。

</div>

**它在算什么**

1. 每位病人有一个概率。10 位各说 30% 的病人,模型预期其中 3 位出事。所以把概率加起来,就是预期的事件数。
2. 数实际出事的人数。
3. O/E = 1: 总数对得上。小于 1: 模型说多了(概率整体偏高)。大于 1: 说少了。

```python
expected = p.sum()
print(f"模型预期 {expected:.1f} 位降下来,实际 {y.sum()} 位: O/E = {y.sum() / expected:.2f}")
```

预期 34.2 位,实际 34 位,O/E = 1.00。总体上不偏。

<div class="exercise">

**练习 1 · 轮到你: 图形优化的 O/E**

对图形优化的结局,算模型预期的事件数(`expected_g`)和 O/E(`oe_g`)。

```python
yg = hi["event_gra"]
pg = out_of_fold(LogisticRegression(C=1e6), hi, cols, "event_gra")
expected_g = pg.___()
oe_g = yg.___() / expected_g
print(f"预期 {expected_g:.1f} 位,实际 {yg.sum()} 位: O/E = {oe_g:.2f}")
```

```pycheck
need("expected_g", "oe_g")
if close(expected_g, 34.17, 0.05):
    fail('这是体积优化的结果。结局要用 "event_gra"(起始代码里已经是),检查有没有改动。')
if close(expected_g, 0.355, 0.005):
    fail("这是平均概率。预期的事件数是把概率加起来: sum。")
if close(expected_g, 51.18, 0.05) and close(oe_g, 0.9965, 0.004):
    ok("对了: 预期 51.2 位,实际 51 位,O/E = 1.00。图形优化的模型总体上也不偏。")
fail(f"现在是预期 {expected_g:.2f}、O/E {oe_g:.3f},应该是 51.18 和 0.997。两个空都是 sum。")
```

<details><summary>提示</summary>

预期的事件数是概率的总和,实际的事件数是结局(0 和 1)的总和: 两个空都是 `sum`。

</details>

<details><summary>参考答案</summary>

```python
yg = hi["event_gra"]
pg = out_of_fold(LogisticRegression(C=1e6), hi, cols, "event_gra")
expected_g = pg.sum()
oe_g = yg.sum() / expected_g
print(f"预期 {expected_g:.1f} 位,实际 {yg.sum()} 位: O/E = {oe_g:.2f}")
```

</details>

</div>

## 3. 幅度对不对: 校准斜率和校准截距

O/E 等于 1,概率也可能不准: 该说 10% 的说成 1%,该说 60% 的说成 90%,两头抵消,总数照样对得上。这是演示里"说得太极端"的那种样子。

<div class="frame">

**背景**: 需要量出概率的"幅度": 模型把病人之间的差别说得太大、太小,还是正好。

**目标**: 一个数,等于 1 时正好。

**做法**: 和第 9 课连续结局的校准斜率是同一个想法: 用模型的预测去预测实际的结局,看斜率。结局是是/否,所以用逻辑回归;预测值要先从概率换成 log-odds(第 2 课)。

</div>

**它在算什么**

1. 把每位病人的概率换成 log-odds: ln(p ÷ (1 − p))。
2. **校准斜率**: 用这个 log-odds 当唯一的变量,对实际的结局做一次逻辑回归,看它的系数。
   - 等于 1: 幅度正好。
   - 小于 1: 说得太极端。模型说的高低差别,实际只兑现了一部分。过拟合的模型是这样。
   - 大于 1: 说得太保守。
3. **校准截距**: 把斜率固定在 1,只估计一个截距。等于 0 是总体不偏;负的是概率整体偏高;正的是偏低。它和 O/E 回答同一个问题,只是在 log-odds 的尺度上。

```python
def logit(prob):
    prob = np.clip(prob, 1e-6, 1 - 1e-6)                 # 概率正好是 0 或 1 时,log 算不了
    return np.log(prob / (1 - prob))

lp = logit(p.to_numpy())                                                        # 第 1 步
slope_fit = sm.Logit(y.to_numpy(), sm.add_constant(lp)).fit(disp=0)             # 第 2 步
offset_fit = sm.GLM(y.to_numpy(), np.ones(len(y)), family=sm.families.Binomial(), offset=lp).fit()   # 第 3 步
print(f"校准斜率 {slope_fit.params[1]:.2f},校准截距 {offset_fit.params[0]:.2f}")
```

斜率 0.93,截距 −0.01: 幅度略微偏大一点,总体不偏。对一个只有两个变量的模型,这是很好的校准。

第 3 步里的 `offset=lp` 是 statsmodels 的写法: 把 `lp` 直接加进公式,系数固定为 1,不去估计它。

后面要反复用,把三个数包成一个函数:

```python
def calibration(actual, prob):
    """O/E、校准截距、校准斜率。"""
    actual, prob = np.asarray(actual), np.asarray(prob)
    lp = logit(prob)
    slope = sm.Logit(actual, sm.add_constant(lp)).fit(disp=0).params[1]
    intercept = sm.GLM(actual, np.ones(len(actual)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return actual.sum() / prob.sum(), intercept, slope

print("O/E %.2f,截距 %.2f,斜率 %.2f" % calibration(y, p))
```

**再加一个综合的数: Brier 分数**。把每位病人的概率和她的结局(0 或 1)相减、平方、取平均:

```python
brier = ((p - y) ** 2).mean()
brier_base = ((y.mean() - y) ** 2).mean()                # 基线: 对每个人都说总体的比例
print(f"Brier 分数 {brier:.3f},基线 {brier_base:.3f},比基线好了 {1 - brier / brier_base:.0%}")
```

Brier 分数 0.088,越小越好。它同时受区分度和校准的影响: 排序差,或者数值不准,都会让它变大。和第 9 课的 R² 一样,它要和基线比才有意义: 对每个人都说 23.6%,Brier 是 0.180;模型比它好了 51%。

<div class="exercise">

**练习 2 · 轮到你: 图形优化的校准斜率和 Brier 分数**

对图形优化的结局,用 `calibration` 取出校准斜率(`slope_g`),再算 Brier 分数(`brier_g`)。

```python
oe, intercept, slope_g = calibration(___, pg)
brier_g = ((pg - yg) ** ___).mean()
print(f"斜率 {slope_g:.2f},截距 {intercept:.2f},Brier {brier_g:.3f}")
```

```pycheck
need("slope_g", "brier_g")
if close(slope_g, 0.929, 0.004):
    fail("这是体积优化的斜率。第一个空是图形优化的结局 yg。")
if close(slope_g, 0.960, 0.004) and close(brier_g, 0.1331, 0.002):
    ok("对了: 斜率 0.96,Brier 0.133。校准和体积优化一样好;Brier 比体积优化的 0.088 大,是因为区分度差一些(AUC 0.89 对 0.93),而且图形优化的事件更接近一半,本来就更难说准。")
fail(f"现在是斜率 {slope_g:.3f}、Brier {brier_g:.4f},应该是 0.960 和 0.1331。第一个空是 yg,第二个空是 2。")
```

<details><summary>提示</summary>

`calibration(实际的结局, 概率)`: 第一个空是 `yg`。Brier 是差的平方的平均: 第二个空是 `2`。

</details>

<details><summary>参考答案</summary>

```python
oe, intercept, slope_g = calibration(yg, pg)
brier_g = ((pg - yg) ** 2).mean()
print(f"斜率 {slope_g:.2f},截距 {intercept:.2f},Brier {brier_g:.3f}")
```

</details>

</div>

## 4. 第一种毛病: 加权把概率抬高

144 位病人里只有 34 个事件。机器学习的教程常常建议: 类别不平衡,要给少的那一类加权(或者过采样、SMOTE)。scikit-learn 里是 `class_weight="balanced"`。试一下:

```python
weighted = LogisticRegression(C=1e6, class_weight="balanced")
pw = out_of_fold(weighted, hi, cols, "event")
print(f"AUC: 不加权 {roc_auc_score(y, p):.3f},加权 {roc_auc_score(y, pw):.3f}")
print(f"平均概率: 不加权 {p.mean():.2f},加权 {pw.mean():.2f};实际降了的比例 {y.mean():.2f}")
print("加权: O/E %.2f,截距 %.2f,斜率 %.2f" % calibration(y, pw))
print(f"加权: Brier {((pw - y) ** 2).mean():.3f}")
```

- AUC 没有变(0.930 和 0.929): 加权没有让模型分得更开。
- 概率整体被抬高了: 平均概率从 0.24 变成 0.35,而实际只有 0.24 的病人降了。模型预期 50 位,实际 34 位,O/E = 0.68。
- 斜率还是 0.99,截距是 −1.21: 幅度没问题,只是整体偏高。
- Brier 从 0.088 变差到 0.117。

**为什么**: 加权让 34 个事件和 110 个非事件在拟合时"一样重",等于告诉模型"降和不降各占一半"。模型照此把截距调高了。偏了多少可以直接算出来: ln(110 ÷ 34) = 1.17,和量到的截距 −1.21 差不多。过采样、欠采样、SMOTE 做的是同一件事,后果相同。

所以: **要用概率的模型,不要为了"类别不平衡"去加权或重采样**。事件少的真正问题是信息少(第 13 课),加权并没有增加信息。读论文时看到 SMOTE 加上"预测概率""风险分层",要先怀疑它的概率。

<div class="exercise">

**练习 3 · 轮到你: 图形优化,加权以后**

对图形优化的结局拟合加权的模型,算平均概率(`mean_w_g`)和 O/E(`oe_w_g`)。

```python
pw_g = out_of_fold(LogisticRegression(C=1e6, class_weight=___), hi, cols, "event_gra")
mean_w_g = pw_g.mean()
oe_w_g, intercept_w_g, slope_w_g = calibration(yg, ___)
print(f"平均概率 {mean_w_g:.2f}(实际比例 {yg.mean():.2f});O/E {oe_w_g:.2f},截距 {intercept_w_g:.2f},斜率 {slope_w_g:.2f}")
```

```pycheck
need("mean_w_g", "oe_w_g")
if close(mean_w_g, 0.346, 0.004):
    fail('这是体积优化的结果。结局要用 "event_gra"(起始代码里已经是),检查有没有改动。')
if close(oe_w_g, 0.9965, 0.004):
    fail("O/E 是 1.00,说明算的是不加权的概率。第二个空是加权模型的概率 pw_g。")
if close(mean_w_g, 0.430, 0.004) and close(oe_w_g, 0.823, 0.004):
    ok("对了: 平均概率 0.43,实际 0.35;O/E 0.82,截距 −0.63。图形优化的事件本来就比较多(51/144),加权抬高得少一些: ln(93 ÷ 51) = 0.60,正好和截距对上。")
fail(f"现在是平均概率 {mean_w_g:.3f}、O/E {oe_w_g:.3f},应该是 0.430 和 0.823。第一个空是 'balanced',第二个空是 pw_g。")
```

<details><summary>提示</summary>

第一个空是 `"balanced"`(带引号);第二个空是加权模型的概率 `pw_g`。

</details>

<details><summary>参考答案</summary>

```python
pw_g = out_of_fold(LogisticRegression(C=1e6, class_weight="balanced"), hi, cols, "event_gra")
mean_w_g = pw_g.mean()
oe_w_g, intercept_w_g, slope_w_g = calibration(yg, pw_g)
print(f"平均概率 {mean_w_g:.2f}(实际比例 {yg.mean():.2f});O/E {oe_w_g:.2f},截距 {intercept_w_g:.2f},斜率 {slope_w_g:.2f}")
```

</details>

</div>

## 5. 第二种毛病: 过拟合让概率太极端

第 6 课讲过: 变量多、病人少,模型会把偶然当规律。对给概率的模型,这个毛病有一个专门的样子: 概率太极端。给两个真变量再加 8 个随机数,34 个事件配 10 个变量:

```python
rng = np.random.default_rng(11)
noise = [f"z{i}" for i in range(8)]
H = pd.concat([hi, pd.DataFrame(rng.normal(size=(len(hi), 8)), columns=noise, index=hi.index)], axis=1)
many = cols + noise

pz = out_of_fold(LogisticRegression(C=1e6, max_iter=5000), H, many, "event")
print(f"10 个变量,不加约束: AUC {roc_auc_score(y, pz):.2f},Brier {((pz - y) ** 2).mean():.3f}")
print("  O/E %.2f,截距 %.2f,斜率 %.2f" % calibration(y, pz))
print(f"  概率低于 1% 或高于 99% 的病人占 {((pz < 0.01) | (pz > 0.99)).mean():.0%}")
```

AUC 只从 0.93 掉到 0.91,看区分度几乎没事。校准斜率却只有 0.53: 模型说的高低差别,实际只兑现了一半。它对 47% 的病人说"不到 1%"或"超过 99%",把话说得太满。

**怎么修: 收缩**。第 6 课的约束正是干这个的: 把系数往 0 压,概率就往中间收。约束的强度在交叉验证里选,选的标准用 log loss(第 2 课: 概率说得越离谱,罚得越重):

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

search = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression()),
                      {"logisticregression__C": [0.03, 0.1, 0.3, 1, 3, 10]}, cv=cv, scoring="neg_log_loss")
ps = out_of_fold(search, H, many, "event")               # 外面再套一层: 嵌套交叉验证(第 5、6 课)
print(f"10 个变量,约束由交叉验证选: AUC {roc_auc_score(y, ps):.2f},Brier {((ps - y) ** 2).mean():.3f}")
print("  O/E %.2f,截距 %.2f,斜率 %.2f" % calibration(y, ps))
```

斜率从 0.53 回到 1.06,Brier 从 0.114 降到 0.094。变量还是那 10 个,只是系数被压小了。

<div class="quiz">
<p><b>想一想</b>: 一个模型的 AUC 是 0.85,校准斜率 0.60。它说某位病人的风险是 90%。这个数该怎么读?</p>
<button class="quiz-opt" data-why="AUC 高只说明排序好。斜率 0.60 说明概率的数值太极端,90% 不能照字面读。">AUC 0.85 不错,所以 90% 可信</button>
<button class="quiz-opt" data-correct data-why="斜率小于 1,说明模型说得太极端: 说很高的,实际没那么高。这位病人在模型眼里确实排在高风险一头(排序可信),但真实的风险多半明显低于 90%。按 90% 去跟病人谈,是在夸大。">她确实属于风险高的一头,但真实的风险多半低于 90%</button>
<button class="quiz-opt" data-why="斜率小于 1 是说得太极端,不是太保守。太保守时斜率大于 1。">斜率小于 1,说明模型保守,真实风险比 90% 还高</button>
</div>

## 6. 不重新建模的修法: 重新校准

有时候模型的排序是好的,只是数值偏了。第 4 节的加权模型就是: AUC 没问题,概率整体偏高。还有一种常见的情形: 把别的医院发表的模型拿到自己的病人身上,事件的比例不一样,概率整体偏了。

这时不必重新建模,只在 log-odds 上做一次修正,叫**重新校准**(recalibration):

新的 log-odds = a + b × 原来的 log-odds

a 和 b 就是第 3 节的校准截距和校准斜率。最简单的一种只改截距(b 保持 1)。加权模型偏了多少是知道的,ln(110 ÷ 34),直接减掉:

```python
shift = np.log((y == 0).sum() / (y == 1).sum())                  # ln(110 / 34): 加权把事件放大的倍数
p_fixed = 1 / (1 + np.exp(-(logit(pw.to_numpy()) - shift)))      # log-odds 减去它,再变回概率
print(f"修正以前: 平均概率 {pw.mean():.2f};修正以后: {p_fixed.mean():.2f};实际 {y.mean():.2f}")
print("修正以后: O/E %.2f,截距 %.2f,斜率 %.2f" % calibration(y, p_fixed))
print(f"AUC {roc_auc_score(y, p_fixed):.3f},Brier {((p_fixed - y) ** 2).mean():.3f}")
```

O/E 回到 0.99,Brier 回到 0.088,和不加权的模型一样;AUC 还是 0.929。重新校准只是把所有病人的 log-odds 一起平移或拉伸,**不改变排序**,所以 AUC 不变。

平常 a 和 b 不是算出来的,而是从数据里估计的。估计它们用的病人,不能再拿来评价修正后的模型: 否则就是用同一批病人既出题又判卷(第 5 课)。在交叉验证里做,或者用单独的一批病人。

## 7. 在新病人身上对答案

```python
final = LogisticRegression(C=1e6).fit(hi[cols], y)
p_new = final.predict_proba(new_hi[cols])[:, 1]
y_new = new_hi["event"]
print(f"{len(new_hi)} 位新病人: 模型预期 {p_new.sum():.0f} 位,实际 {y_new.sum()} 位")
print("O/E %.2f,截距 %.2f,斜率 %.2f" % calibration(y_new, p_new))

tenth = pd.qcut(p_new, 10, labels=False)
check = pd.DataFrame({"prob": p_new, "event": y_new.to_numpy()}).groupby(tenth).mean()
fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
ax.plot(check["prob"], check["event"], "o-")
ax.set_xlabel("Predicted probability (group mean)")
ax.set_ylabel("Observed proportion")
ax.set_title("3457 new patients, tenths", fontsize=10)
plt.tight_layout()
plt.show()
```

模型预期 779 位,实际 848 位: O/E = 1.09,斜率 0.89。交叉验证说的是 1.00 和 0.93。方向一致(幅度略大),但新病人身上显出一点交叉验证没看出来的低估: 144 位病人、34 个事件,校准的估计本身就不精确。新病人够多,可以分十组,点基本贴着对角线。

<div class="exercise">

**练习 4 · 轮到你: 图形优化,新病人身上的校准**

用全部 144 位病人拟合图形优化的模型,算它在新病人身上的 O/E(`oe_new_g`)和校准斜率(`slope_new_g`)。

```python
final_g = LogisticRegression(C=1e6).fit(hi[cols], hi["event_gra"])
p_new_g = final_g.predict_proba(new_hi[cols])[:, 1]
oe_new_g, intercept_new_g, slope_new_g = calibration(new_hi[___], ___)
print(f"O/E {oe_new_g:.2f},截距 {intercept_new_g:.2f},斜率 {slope_new_g:.2f}")
```

```pycheck
need("oe_new_g", "slope_new_g")
if close(oe_new_g, 1.088, 0.004):
    fail('这是体积优化的结果。第一个空填 "event_gra"。')
if close(oe_new_g, 1.0685, 0.004) and close(slope_new_g, 0.966, 0.004):
    ok("对了: O/E 1.07,斜率 0.97。和体积优化一样,新病人身上略有低估,幅度基本正好。")
fail(f'现在是 O/E {oe_new_g:.3f}、斜率 {slope_new_g:.3f},应该是 1.069 和 0.966。第一个空是 "event_gra",第二个空是 p_new_g。')
```

<details><summary>提示</summary>

`calibration(实际的结局, 概率)`: 第一个空是新病人的结局 `"event_gra"`,第二个空是给新病人的概率 `p_new_g`。

</details>

<details><summary>参考答案</summary>

```python
final_g = LogisticRegression(C=1e6).fit(hi[cols], hi["event_gra"])
p_new_g = final_g.predict_proba(new_hi[cols])[:, 1]
oe_new_g, intercept_new_g, slope_new_g = calibration(new_hi["event_gra"], p_new_g)
print(f"O/E {oe_new_g:.2f},截距 {intercept_new_g:.2f},斜率 {slope_new_g:.2f}")
```

</details>

</div>

## 小结

- **校准**: 模型说的概率能不能照字面相信。只能在一群病人身上核对: 按概率分组,比较模型说的和实际的(校准表、校准图)。
- **O/E** 和**校准截距**量总体偏不偏;**校准斜率**量幅度对不对(小于 1 是说得太极端)。**Brier 分数**是综合的一个数,要和基线比。
- **加权、过采样**不提高 AUC,却把概率整体抬高。要用概率的模型不要做。
- **过拟合**让概率太极端(斜率小于 1)。用**收缩**(第 6 课的约束)修。
- 排序好而数值偏的模型,可以**重新校准**: 在 log-odds 上改截距(和斜率),AUC 不变。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 校准 | calibration | 预测的概率和实际发生的比例一致 |
| 校准图 | calibration plot (reliability diagram) | 按概率分组,模型说的对实际的 |
| 实际与预期之比 | O/E (observed/expected ratio) | 实际事件数 ÷ 概率的总和 |
| 校准截距 | calibration intercept (calibration-in-the-large) | 斜率固定为 1 时的截距;0 是总体不偏 |
| 校准斜率 | calibration slope | 结局对预测的 log-odds 回归的系数;1 是幅度正好 |
| Brier 分数 | Brier score | (概率 − 结局)² 的平均 |
| 类别加权 | class weighting | 拟合时让少的那一类更重 |
| 重新校准 | recalibration | 不重新建模,只在 log-odds 上修正截距和斜率 |

## 这一课没讲什么

- **光滑的校准曲线**: 不分组,用 loess 或样条画一条连续的线(论文里常见)。病人少时它同样不稳。
- **树模型的校准**: 随机森林的概率偏向中间,梯度提升的可能太极端(第 7 课提过)。检查和修正的办法和这一课相同;scikit-learn 的 `CalibratedClassifierCV` 把重新校准包进了交叉验证。
- **Hosmer–Lemeshow 检验**: 一个老的校准检验。它依赖怎么分组,样本小时查不出问题、样本大时什么都显著,现在一般不推荐当主要证据。
- **校准的置信区间**: O/E、截距、斜率都可以像第 10 课那样用 bootstrap 给区间。

## 写进论文 Writing it up

<div class="paper">

方法和结果可以这样写(数字换成你自己的):

> Calibration was assessed on out-of-fold predictions with the observed-to-expected ratio (O/E), the calibration intercept and slope (logistic regression of the outcome on the logit of the predicted probability), and a calibration plot by quarters of predicted probability. No class weighting or resampling was used. The O/E ratio was 1.00, the calibration intercept −0.01 and the calibration slope 0.93; the Brier score was 0.088 (0.180 for a model predicting the overall proportion for every patient).

**审稿人可能问**: 类别不平衡,为什么不用 SMOTE 或类别加权?

**怎么回答**: 这些做法不改善区分度,却使预测概率系统性偏高;本研究的目的是给出可用的概率。可以把加权后的 O/E 放在补充材料里作为说明。

**审稿人可能问**: 只报告了 AUC,校准呢?

**怎么回答**: 补上校准图、O/E、截距和斜率。只有区分度的模型不能用于按风险做决定。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| `divide by zero encountered in log` | 有的概率正好是 0 或 1。换算 log-odds 之前先 `np.clip`,像上面的 `logit` 那样。 |
| `PerfectSeparationWarning` 或系数大得离谱 | 变量把事件和非事件完全分开了,多见于变量多、事件少。减少变量,或者加约束。 |
| `ConvergenceWarning: lbfgs failed to converge` | 变量没有标准化、约束又很弱时容易出现。加大 `max_iter`,或者先标准化。 |
| 校准图上的点忽高忽低 | 每组病人太少。少分几组,并且不要去读每个点的细节。 |
| 结果和网页上写的不一样 | 检查 `random_state=0` 和 `default_rng(11)`;代码块要从上往下依次运行。 |

## 延伸阅读 Further reading

- Van Calster B, McLernon DJ, van Smeden M, 等. Calibration: the Achilles heel of predictive analytics. *BMC Medicine* 2019;17:230。为什么校准比区分度更容易被忽略、后果更大。
- van den Goorbergh R, van Smeden M, Timmerman D, Van Calster B. The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression. *J Am Med Inform Assoc* 2022;29(9):1525–1534。第 4 节的内容,用模拟和真实数据做了一遍。
- scikit-learn 用户指南: [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)。

**下一课**: 区分度和校准都好,模型就有用吗? 还差一步: 按模型的概率做决定,比"所有人都换方案"或"都不换"更好吗? 第 12 课讲决策曲线。
