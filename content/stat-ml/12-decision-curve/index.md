---
title: "Lesson 12 · 决策曲线 Decision Curves"
summary: "按模型做决定,比对所有人都做、或者都不做更好吗: 阈值概率、净获益和决策曲线。"
weight: 120
toc: true
duration: "约 70 分钟,可以分两次学: 第 0–3 节,第 4–6 节"
prereq: "本课程[第 10 课](../10-discrimination/)(阈值、灵敏度和特异度)和[第 11 课](../11-calibration/)(校准)"
questions:
  - "AUC 和校准都不错,模型就值得用吗?"
  - "概率到多少才值得动手?这个数从哪里来?"
  - "净获益是什么?怎么和'全都做''都不做'比?"
  - "没有信息的模型、概率不准的模型,在决策曲线上是什么样子?"
objectives:
  - "把一个临床上的取舍换算成阈值概率"
  - "自己算出按模型、全都做、都不做三种做法的净获益"
  - "画出并读懂决策曲线"
  - "说明为什么阈值的范围要在看结果之前定好"
---

第 10 课量了区分度(AUC 0.93),第 11 课量了校准(O/E 1.00,斜率 0.93)。两样都好。可是这两个数都没有回答临床上最后要问的那一句: **用这个模型来做决定,患者(和科室)比不用它更好吗?**

这一课的决定是: 对 5 mm 计划达到 68 Gy 的患者,要不要再做一个体积优化的计划? 做了而且降到 68 Gy 以下,这次计划有用;做了没降下来,白做。不做的话,本来能降下来的患者就错过了。

有两种不用模型的做法: **全都做**,和**都不做**。模型要有用,得比这两种都好。

**练习怎么做**: 讲解用**体积优化**(`event`),练习换成**图形优化**(`event_gra`)。

## 0. 准备

```python
import numpy as np
import pandas as pd
from cohort import make_cohort
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

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

def out_of_fold(model, columns, outcome):
    proba = cross_val_predict(model, hi[columns], hi[outcome], cv=cv, method="predict_proba")
    return pd.Series(proba[:, 1], index=hi.index)

y = hi["event"]
p = out_of_fold(LogisticRegression(C=1e6), cols, "event")
print(f"{len(hi)} 位患者,其中 {y.sum()} 位做体积优化能降到 68 Gy 以下")
```

## 1. 三种做法,先数一数

假定概率达到 0.2 就做(这个 0.2 从哪来,下一节讲):

```python
do = p >= 0.2                                             # 按模型: 概率达到 0.2 的患者做体积优化
useful = (do & (y == 1)).sum()                            # 做了,降下来了
wasted = (do & (y == 0)).sum()                            # 做了,没降下来
missed = (~do & (y == 1)).sum()                           # 没做,其实能降下来
print(f"按模型: 做 {do.sum()} 次计划,{useful} 次有用,{wasted} 次白做;漏掉 {missed} 位")
print(f"全都做: 做 {len(y)} 次计划,{y.sum()} 次有用,{(y == 0).sum()} 次白做;漏掉 0 位")
print(f"都不做: 做 0 次计划;漏掉 {y.sum()} 位")
```

- 全都做: 144 次计划,34 次有用,110 次白做。
- 按模型: 53 次计划,28 次有用,25 次白做;漏掉 6 位。
- 都不做: 一次不做,34 位全部错过。

哪一种最好? 按模型比全都做少白做了 85 次,代价是漏掉 6 位。这笔账划不划算,取决于**白做一次有多糟,漏掉一位有多糟**。没有这个判断,三种做法没法比。

## 2. 阈值概率: 把取舍变成一个数

<div class="frame">

**背景**: 要比较三种做法,先得说清楚: 一次有用的计划值多少,一次白做的计划亏多少。

**目标**: 把这个取舍变成一个数。

**做法**: 问自己一个问题: **一位患者降下来的概率至少要有多大,才值得给她做这个计划?** 这个概率叫**阈值概率**(threshold probability)。

</div>

**它在算什么**

1. 假如你回答"至少 20%"。意思是: 5 位这样的患者都做,1 次有用、4 次白做,你觉得刚好值。
2. 换句话说: 在你看来,1 次有用的计划顶得上 4 次白做。白做 1 次,抵掉 1/4 次有用的。
3. 一般地,阈值概率是 pt 时,白做 1 次抵掉 pt ÷ (1 − pt) 次有用的。10% 是 1/9,20% 是 1/4,50% 是 1。

**打个比方**: 泌尿科医生问: 前列腺癌的可能性到多少,才值得做穿刺活检? 回答"10%"的医生,愿意做 9 次阴性的活检来找出 1 个癌;回答"30%"的医生更在意不必要的活检。阈值概率不是模型算出来的,是临床判断: 它反映的是漏掉和白做的代价之比。

对这一课的决定: 多做一个体积优化的计划,花的是物理师一两个小时;有用的话,患者的阴道剂量降到了限值以下。白做的代价小、有用的好处大,所以阈值概率应该比较低,比如 10% 到 30%。不同的科室可以有不同的回答,所以后面看的是一段**范围**,不是一个点。

## 3. 净获益

<div class="frame">

**背景**: 有了阈值概率,有用和白做就能换算了。

**目标**: 给每种做法一个分数: 扣掉白做的代价以后,净得多少次有用的计划。

**做法**: **净获益**(net benefit) = 有用的比例 − 白做的比例 × pt ÷ (1 − pt)。

</div>

**它在算什么**

1. 数这种做法下有用的次数和白做的次数。
2. 都除以患者总数,变成"每位患者"。
3. 白做的那一项乘以 pt ÷ (1 − pt): 按上一节的换算,折成有用的次数。
4. 相减。

```python
pt = 0.2
price = pt / (1 - pt)                                     # 白做 1 次抵掉多少次有用的: 0.25
n = len(y)
nb_model = useful / n - wasted / n * price
nb_all = y.sum() / n - (y == 0).sum() / n * price
print(f"按模型: {useful}/{n} − {wasted}/{n} × {price} = {nb_model:.3f}")
print(f"全都做: {y.sum()}/{n} − {(y == 0).sum()}/{n} × {price} = {nb_all:.3f}")
print("都不做: 0")
```

按模型 0.151,全都做 0.045,都不做 0。

**这个数怎么读**: 0.151 的意思是,每 100 位患者里,按模型做相当于**净得 15.1 次有用的计划,一次也不白做**。全都做相当于净得 4.5 次: 它有用的次数更多(34 对 28),但 110 次白做折算下来抵掉了大部分。两者相差 0.106: 每 100 位患者,用模型多净得 10.6 次有用的计划。

"都不做"的净获益永远是 0: 没有好处,也没有白做。净获益是负的做法,比什么都不做还糟。

包成函数:

```python
def net_benefit(actual, prob, pt):
    """概率达到 pt 的患者就做: 这种做法的净获益。"""
    actual, prob = np.asarray(actual), np.asarray(prob)
    do = prob >= pt
    useful, wasted = (do & (actual == 1)).sum(), (do & (actual == 0)).sum()
    return useful / len(actual) - wasted / len(actual) * pt / (1 - pt)

def net_benefit_all(actual, pt):
    """对所有人都做的净获益。"""
    rate = np.mean(actual)
    return rate - (1 - rate) * pt / (1 - pt)

print(f"pt = 0.2: 按模型 {net_benefit(y, p, 0.2):.3f},全都做 {net_benefit_all(y, 0.2):.3f}")
```

<div class="exercise">

**练习 1 · 轮到你: 图形优化,阈值概率 0.2**

对图形优化的结局,算按模型做的净获益(`nb_g`)和全都做的净获益(`nb_all_g`),阈值概率 0.2。

```python
yg = hi["event_gra"]
pg = out_of_fold(LogisticRegression(C=1e6), cols, "event_gra")
nb_g = net_benefit(yg, ___, 0.2)
nb_all_g = net_benefit_all(___, 0.2)
print(f"按模型 {nb_g:.3f},全都做 {nb_all_g:.3f},相差 {nb_g - nb_all_g:.3f}")
```

```pycheck
need("nb_g", "nb_all_g")
if close(nb_g, 0.151, 0.002):
    fail('这是体积优化的结果。结局要用 "event_gra"(起始代码里已经是),检查有没有改动。')
if close(nb_g, 0.2882, 0.002) and close(nb_all_g, 0.1927, 0.002):
    ok("对了: 按模型 0.288,全都做 0.193,相差 0.095。图形优化能降下来的患者更多(51 位),所以全都做也有不小的净获益;模型仍然更好: 每 100 位患者多净得 9.5 次有用的计划。")
fail(f"现在是 {nb_g:.3f} 和 {nb_all_g:.3f},应该是 0.288 和 0.193。第一个空是 pg,第二个空是 yg。")
```

<details><summary>提示</summary>

`net_benefit(实际的结局, 概率, 阈值概率)`: 第一个空是概率 `pg`。`net_benefit_all(实际的结局, 阈值概率)`: 第二个空是 `yg`。

</details>

<details><summary>参考答案</summary>

```python
yg = hi["event_gra"]
pg = out_of_fold(LogisticRegression(C=1e6), cols, "event_gra")
nb_g = net_benefit(yg, pg, 0.2)
nb_all_g = net_benefit_all(yg, 0.2)
print(f"按模型 {nb_g:.3f},全都做 {nb_all_g:.3f},相差 {nb_g - nb_all_g:.3f}")
```

</details>

</div>

## 4. 所有的阈值概率: 决策曲线

不同的医生、不同的科室,阈值概率不一样。所以不只算一个点,而是把一段范围里的阈值概率都算一遍,画成线。这就是**决策曲线**(decision curve)。

<div class="demo">

**设定**: 第 10 课演示里的那 40 位模拟患者: 12 位做新计划有用,28 位没用。粗线是按模型做的净获益,细线是全都做,点线是都不做(0)。滑块是阈值概率。

**你做什么**: 把阈值概率从 0.20 往左拖到 0.05,再往右一路拖到 0.60。看三个条谁最长。

**你会看到**: 阈值概率 0.20: 按模型 0.200,全都做 0.125,都不做 0;按模型做 28 次计划,12 次有用、16 次白做。拖到 0.05: 模型让 40 位全都做了,和"全都做"完全一样(0.263)。往右拖: 全都做的条越来越短,到 0.30 正好是 0,再往右是负的(0.40 时 −0.167);按模型一直是正的(0.40 时 0.150,0.60 时 0.075)。

**这说明**: 阈值概率很低(白做几乎没有代价)时,全都做就够了,模型帮不上忙。阈值概率升高,全都做的净获益直线下降,到阈值概率等于"有用的患者所占的比例"(这里是 0.30)时降到 0,再往右是负的。模型有用的地方在中间这一段。

<div class="widget" data-widget="netbenefit">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

在课程的队列上画:

```python
import matplotlib.pyplot as plt

grid = np.arange(0.02, 0.61, 0.01)
fig, ax = plt.subplots(figsize=(4.4, 3.6))
ax.plot(grid, [net_benefit(y, p, t) for t in grid], label="Model")
ax.plot(grid, [net_benefit_all(y, t) for t in grid], label="Replan all", ls="--")
ax.axhline(0, color="grey", lw=1, label="Replan none")
ax.set_ylim(-0.05, 0.26)
ax.set_xlabel("Threshold probability")
ax.set_ylabel("Net benefit")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

for t in [0.05, 0.1, 0.2, 0.3]:
    print(f"pt = {t}: 按模型 {net_benefit(y, p, t):.3f},全都做 {net_benefit_all(y, t):.3f}")
```

读这张图:

- **看哪条线在最上面**。在事先定好的范围(这里是 0.1 到 0.3)里,模型的线一直在最上面: 0.1 时 0.207 对 0.151,0.3 时 0.137 对 −0.091。在这个范围里,按模型做比两种不用模型的做法都好。
- **"全都做"的线**从左上往右下走,在阈值概率等于事件比例(34/144 = 0.236)的地方穿过 0。阈值概率比这更高的人,宁可都不做,也不会全都做。
- **最左边**,模型和全都做几乎重合(0.05 时 0.216 对 0.196): 白做没什么代价时,不需要模型。
- 线是一级一级的,因为只有 144 位患者: 阈值每跨过一位患者的概率,计数就变一次。

<div class="exercise">

**练习 2 · 轮到你: 图形优化,阈值概率低一些**

如果科室认为白做一次计划几乎没有代价,阈值概率定在 0.1。对图形优化,算按模型和全都做的净获益(`nb10_g`、`nb10_all_g`)。

```python
nb10_g = net_benefit(yg, pg, ___)
nb10_all_g = net_benefit_all(yg, ___)
print(f"pt = 0.1: 按模型 {nb10_g:.3f},全都做 {nb10_all_g:.3f},相差 {nb10_g - nb10_all_g:.3f}")
```

```pycheck
need("nb10_g", "nb10_all_g")
if close(nb10_g, 0.2882, 0.002):
    fail("这是阈值概率 0.2 的结果。两个空都填 0.1。")
if close(nb10_g, 0.3156, 0.002) and close(nb10_all_g, 0.2824, 0.002):
    ok("对了: 按模型 0.316,全都做 0.282,只差 0.033。阈值概率 0.2 时差 0.095。白做的代价看得越轻,模型的用处越小: 这时候干脆全都做也差不了多少。")
fail(f"现在是 {nb10_g:.3f} 和 {nb10_all_g:.3f},应该是 0.316 和 0.282。两个空都是 0.1。")
```

<details><summary>提示</summary>

两个空都是阈值概率 `0.1`。

</details>

<details><summary>参考答案</summary>

```python
nb10_g = net_benefit(yg, pg, 0.1)
nb10_all_g = net_benefit_all(yg, 0.1)
print(f"pt = 0.1: 按模型 {nb10_g:.3f},全都做 {nb10_all_g:.3f},相差 {nb10_g - nb10_all_g:.3f}")
```

</details>

</div>

## 5. 没用的模型和概率不准的模型

**没有信息的模型会帮倒忙**。第 10 课里只用年龄和分级的模型,AUC 是 0.42:

```python
p_noise = out_of_fold(LogisticRegression(C=1e6), ["age", "grade"], "event")
for t in [0.2, 0.3]:
    print(f"pt = {t}: 年龄和分级 {net_benefit(y, p_noise, t):.3f},全都做 {net_benefit_all(y, t):.3f},都不做 0")
```

阈值概率 0.2 时,它的净获益(0.021)不如全都做(0.045);0.3 时是负的(−0.027),不如都不做。用一个没有信息的模型做决定,比不用模型更糟: 它随机地漏掉一些患者,又随机地让另一些白做。

**概率不准的模型**。第 11 课里加权的模型,AUC 和不加权的一样,但概率整体偏高:

```python
p_weighted = out_of_fold(LogisticRegression(C=1e6, class_weight="balanced"), cols, "event")
for t in [0.2, 0.3, 0.4, 0.5]:
    do_w = (p_weighted >= t).sum()
    print(f"pt = {t}: 不加权 {net_benefit(y, p, t):.3f}(做 {(p >= t).sum()} 次),加权 {net_benefit(y, p_weighted, t):.3f}(做 {do_w} 次)")
```

概率被抬高以后,同一个阈值下有更多的患者"过线",做的计划多了。阈值概率低的时候(0.2、0.3)差别不大,加权的甚至略高一点(患者少,几位之差);到 0.4 和 0.5,加权的模型多做了十七到十九次计划,其中十五六次是白做的,净获益掉了一大半(0.065 对 0.120,0.035 对 0.125)。

两个模型的 AUC 一样,净获益却可以差这么多。净获益把区分度和校准都算了进去: 阈值是按概率的数值划的,数值不准,线就划错了地方。

<div class="quiz">
<p><b>想一想</b>: 一篇论文画了决策曲线,结论是"模型在阈值概率 0.62 到 0.71 之间有净获益"。这个决定是"要不要多做一个优化计划"。这个结论有多大用处?</p>
<button class="quiz-opt" data-why="有净获益的范围确实存在。但要看这个范围在临床上有没有人会用。">有用: 模型在一段范围内有净获益</button>
<button class="quiz-opt" data-correct data-why="阈值概率 0.62 的意思是: 白做 1 次计划抵得上 1.6 次有用的。对'多做一个计划'这种代价很小的事,没有人会定这么高的阈值。在临床上合理的范围(比如 0.1 到 0.3)里模型有没有净获益,才是要回答的问题。而且这个范围像是看了曲线以后挑出来的。">用处很小: 没有人会给一件代价很小的事定这么高的阈值</button>
<button class="quiz-opt" data-why="决策曲线可以用在任何'做或不做'的决定上,包括这一个。问题在阈值的范围。">没用: 决策曲线不适合这种决定</button>
</div>

## 6. 在新患者身上对答案

```python
final = LogisticRegression(C=1e6).fit(hi[cols], y)
p_new = final.predict_proba(new_hi[cols])[:, 1]
y_new = new_hi["event"]
for t in [0.1, 0.2, 0.3]:
    print(f"pt = {t}: {len(new_hi)} 位新患者,按模型 {net_benefit(y_new, p_new, t):.3f},全都做 {net_benefit_all(y_new, t):.3f}")
```

新患者身上,阈值概率 0.2 时按模型 0.174,全都做 0.057。交叉验证给的是 0.151 和 0.045。结论一样: 在 0.1 到 0.3 的范围里,模型都在上面。

<div class="exercise">

**练习 3 · 轮到你: 图形优化,在新患者身上对答案**

用全部 144 位患者拟合图形优化的模型,在新患者身上算阈值概率 0.2 时按模型的净获益(`nb_new_g`)和全都做的净获益(`nb_new_all_g`)。

```python
final_g = LogisticRegression(C=1e6).fit(hi[cols], hi["event_gra"])
p_new_g = final_g.predict_proba(new_hi[cols])[:, 1]
nb_new_g = net_benefit(new_hi[___], p_new_g, 0.2)
nb_new_all_g = net_benefit_all(new_hi[___], 0.2)
print(f"新患者: 按模型 {nb_new_g:.3f},全都做 {nb_new_all_g:.3f}")
```

```pycheck
need("nb_new_g", "nb_new_all_g")
if close(nb_new_g, 0.1744, 0.002):
    fail('这是体积优化的结果。两个空都填 "event_gra"。')
if close(nb_new_g, 0.2970, 0.002) and close(nb_new_all_g, 0.2298, 0.002):
    ok("对了: 新患者身上按模型 0.297,全都做 0.230。交叉验证给的是 0.288 和 0.193: 模型比全都做好,这个结论在新患者身上也成立。")
fail(f"现在是 {nb_new_g:.3f} 和 {nb_new_all_g:.3f},应该是 0.297 和 0.230。两个空都是 'event_gra'。")
```

<details><summary>提示</summary>

两个空都是新患者的实际结局 `"event_gra"`。

</details>

<details><summary>参考答案</summary>

```python
final_g = LogisticRegression(C=1e6).fit(hi[cols], hi["event_gra"])
p_new_g = final_g.predict_proba(new_hi[cols])[:, 1]
nb_new_g = net_benefit(new_hi["event_gra"], p_new_g, 0.2)
nb_new_all_g = net_benefit_all(new_hi["event_gra"], 0.2)
print(f"新患者: 按模型 {nb_new_g:.3f},全都做 {nb_new_all_g:.3f}")
```

</details>

</div>

## 小结

- 模型有没有用,要和两种不用模型的做法比: **全都做**和**都不做**。
- **阈值概率**是临床判断: 概率至少多大才值得做。它定下了白做一次抵掉多少次有用的: pt ÷ (1 − pt)。
- **净获益** = 有用的比例 − 白做的比例 × pt ÷ (1 − pt)。可以读成"每 100 位患者净得多少次有用的处理"。
- **决策曲线**是一段阈值概率范围里的净获益。范围要按临床理由**事先定好**;看模型的线在这个范围里是不是最高。
- 没有信息的模型净获益可以是负的;概率不准的模型,AUC 不变,净获益会掉。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 阈值概率 | threshold probability | 概率至少多大才值得做;反映白做和漏掉的代价之比 |
| 净获益 | net benefit | 有用的比例,减去按阈值折算的白做的比例 |
| 决策曲线 | decision curve | 各个阈值概率下的净获益 |
| 全都做 / 都不做 | treat all / treat none | 两种不用模型的做法,模型要比它们都好 |
| 临床效用 | clinical utility | 用模型做决定带来的实际好处 |

## 这一课没讲什么

- **净获益的不确定性**: 和 AUC 一样可以用 bootstrap 给区间。患者少时,两条线之间的差别可能只是运气。
- **减少了多少不必要的处理**(net reduction in interventions): 净获益的另一种说法,读成"每 100 位患者少白做几次"。
- **不止两种选择**: 这一课的决定是"做或不做"。在几种方案里挑一种(比如 5 mm、体积优化、图形优化),要比较每种挑法的结果,原理相同,算法更复杂。
- **阈值概率不是模型的切点**: 第 10 课说不要挑"最佳阈值"。决策曲线的做法是反过来的: 阈值由临床定,再看模型在那里有没有用。
- **代价不只是白做**: 真实的决定里还有费用、等待时间、患者的意愿。阈值概率把这些都压进了一个数,所以要给范围。

## 写进论文 Writing it up

<div class="paper">

方法和结果可以这样写(数字换成你自己的):

> Clinical utility was assessed by decision curve analysis on out-of-fold predictions. Because an additional optimised plan costs little relative to the benefit of meeting the dose constraint, threshold probabilities between 0.10 and 0.30 were prespecified as clinically reasonable. Across this range the net benefit of replanning according to the model exceeded that of replanning all patients and of replanning none (at a threshold of 0.20: 0.151 vs 0.045 and 0).

**审稿人可能问**: 阈值概率的范围是怎么定的?

**怎么回答**: 在看结果之前按临床理由定的,并写出理由(白做和漏掉各意味着什么)。不要写"模型在某某范围内有净获益"而不说明这个范围有没有人会用。

**审稿人可能问**: AUC 已经 0.93 了,为什么还要决策曲线?

**怎么回答**: AUC 不涉及决定的代价。全都做在低阈值下可能和模型一样好;只有净获益能回答"用它做决定值不值"。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| 净获益是负数 | 不是错。这种做法在这个阈值概率下比都不做还糟。 |
| "全都做"的线一路掉到很低 | 正常: 阈值概率越高,白做的代价越重。画图时用 `set_ylim` 把纵轴的下端截在略低于 0 的地方。 |
| 决策曲线是锯齿状的 | 患者少。可以把阈值的格子放粗一些;不要去解释每一个小台阶。 |
| `ZeroDivisionError` 或净获益极大 | 阈值概率取到了 1。范围取到 0.9 以内。 |
| 结果和网页上写的不一样 | 检查 `random_state=0`;代码块要从上往下依次运行。 |

## 延伸阅读 Further reading

- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making* 2006;26(6):565–574。提出决策曲线的论文。
- Vickers AJ, van Calster B, Steyerberg EW. A simple, step-by-step guide to interpreting decision curve analysis. *Diagn Progn Res* 2019;3:18。怎么读、怎么定阈值范围,开放获取。

**下一课**: 这几课的结论都带着同一句话: "144 位患者、34 个事件,估计不精确。"到底多少患者才够建一个预测模型? 第 13 课讲样本量。

{{< practice >}}
