---
title: "Lesson 1 · 线性回归: 模型在算什么 Linear Regression"
summary: "用一条直线(或一个平面)做预测: 系数怎么找出来、怎么读,怎么用它预测新病人。"
weight: 10
toc: true
duration: "约 80 分钟,可以分两次学: 第 0–4 节,第 5–7 节"
prereq: "[Python 课](/python/)第 1–4 课(会用 pandas 选列、画散点图)"
questions:
  - '相关系数说壁厚和剂量变化"有关系"。来了一位新病人,变化会是多少?'
  - '"最好的那条直线"是怎么找出来的?'
  - "回归表里的系数 1.57,到底是什么意思?"
objectives:
  - "说出一条直线好不好是用什么量的,并且自己动手调一条线"
  - "用几行 numpy 算出最小二乘的斜率和截距,和 scikit-learn 的结果对上"
  - "把一位新病人的数代进公式得到预测值,用残差说明预测差了多少"
  - '读懂多个变量时的系数: "其他变量不变时"'
  - "把是/否的变量写成 0 和 1 放进模型"
---

Python 课第 6 课算过相关系数: 平均壁厚和 Δ(换成体积优化以后阴道 D2cc 的变化)的 Pearson 相关系数是 0.62。壁越厚,D2cc 降得越少。相关系数回答的是"有没有关系,关系有多紧"。

在门诊,问题是另一个样子: 眼前这位病人平均壁厚 3.0 mm,如果把 5 mm 计划换成体积优化,D2cc 大概会变多少 Gy?0.62 给不出这个数。要回答它,需要一条能把"壁厚"换算成"Δ"的公式。这样的公式就是一个**模型**(model),最简单的一种是**线性回归**(linear regression)。

线性回归的输出你在统计软件里见过: 系数、标准误、P 值。这一课先不看 P 值,只把这条线本身拆开: 它是怎么找出来的,系数怎么读,怎么拿它做预测。P 值和区间留到第 3 课。

> 🔬 这门课后面的每一种模型(逻辑回归、LASSO、树),都是在这一课的三件事上做改动: 公式长什么样,用什么尺子量"好",怎么找到最好的那一组数。这一课把这三件事看清楚,后面就只是在换零件。

**练习怎么做**: 讲解里预测的是**体积优化**计划的 Δ。练习都换成**图形优化**计划: 每讲完一步,你就对图形优化把这一步再做一遍。5 道练习做完,图形优化的模型就是你自己建的。

## 0. 准备: 模拟队列和要预测的量

这门课用和 Python 课相同的模拟队列(数字都是编的,[源码](/python/cohort.py))。要预测的量叫 `delta`: 换成体积优化计划后,阴道 D2cc 比 5 mm 计划变了多少 Gy,负数表示降低。

```python
import numpy as np
import pandas as pd
from cohort import make_cohort

df = make_cohort().dropna()     # 先去掉缺壁厚测量的几位病人,更好的做法在第 5 课
df["delta"] = df["d2cc_vol"] - df["d2cc_5mm"]
print(len(df), "位病人")
print(df["delta"].describe().round(2))
df[["wall_mean", "diameter", "n_fx", "arm", "d2cc_5mm", "d2cc_vol", "delta"]].head()
```

第一次运行要下载 numpy 和 pandas,会多等一会儿。196 位病人,Δ 平均 −2.09 Gy,最多降了 8.8 Gy,也有人升了 4.9 Gy。表里是前 5 位病人: 平均壁厚(mm)、施源器直径(mm)、近距离次数、治疗方式、两套计划的 D2cc 和它们的差。

## 1. 先不看壁厚: 给所有人猜同一个数

预测,说到底是猜: 给每位病人猜一个 Δ,猜得离实际越近越好。先从最省事的猜法开始: 什么信息都不看,给 196 位病人猜同一个数。

每位病人猜错了多少,叫**误差**: 实际值减去猜的值。把每个人的误差平方,再取平均,得到一个数,代表这种猜法总体上错得多厉害。为什么要平方,第 3 节再说;先用它当尺子。

<div class="quiz">
<p><b>先猜一猜</b>: 给所有人猜同一个数,猜哪个数,误差平方的平均最小?</p>
<button class="quiz-opt" data-why="猜 0 等于说'换方案没有影响'。可是大多数病人是降低的,这样猜,对多数人都偏高。运行下面的代码,看它和别的猜法差多少。">0 Gy: 假设换方案没有变化</button>
<button class="quiz-opt" data-correct data-why="对所有人只猜一个数时,平均值让误差平方的平均最小。运行下面的代码,看它比别的数小多少。">−2.09 Gy: 这些病人的平均值</button>
<button class="quiz-opt" data-why="降得最多的病人只是个别情况,拿它去猜所有人,对几乎每个人都偏低。运行下面的代码看看。">−8.8 Gy: 降得最多的那位病人的值</button>
</div>

```python
y = df["delta"]
for guess in [0, -8.8, y.mean()]:
    error = y - guess                      # 每位病人的误差: 实际值 − 猜的值
    print(f"都猜 {guess:6.2f} Gy: 误差平方的平均 = {(error ** 2).mean():.2f}")
```

都猜 0,误差平方的平均是 12.31;都猜 −8.8,是 53.03;都猜平均值 −2.09,是 7.96,三种里最小。可以证明,任何别的数都比平均值差。"给所有人猜平均值"是最简单的预测模型,叫**基线模型**(baseline model)。后面每一个模型都要先回答一个问题: 比 7.96 好多少?

误差平方的平均有个英文名,以后会常见: **均方误差**(mean squared error, MSE)。

## 2. 让猜的数跟着壁厚走: 自己调一条线

同一个数对谁都不合适。画出来看:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(df["wall_mean"], df["delta"], s=12, alpha=0.6)
ax.axhline(y.mean(), color="grey", linestyle="--", linewidth=1)
ax.set_xlabel("Mean wall thickness (mm)")
ax.set_ylabel("Change in D2cc (Gy)")
plt.show()
```

虚线是平均值。壁薄的病人(左边)大多在虚线下面,壁厚的(右边)大多在上面: 对左边的人猜高了,对右边的人猜低了。更好的猜法是让猜的数跟着壁厚走: 壁薄就猜得低一些,壁厚就猜得高一些。最简单的"跟着走",是一条直线。

直线有无数条,哪一条最好?先自己动手试。

<div class="demo">

**设定**: 30 位模拟病人(蓝点),横轴是平均壁厚,纵轴是 Δ。黑线是用来预测的直线: 一开始是平的,对谁都猜 −2.5 Gy(这 30 位的平均值)。每位病人到黑线的那段竖线,是这位病人的误差。下面的条是误差平方的平均。

**你做什么**: 先拖"斜率",让黑线顺着蓝点的走向斜过来;再拖"高度"微调。目标是让下面的数尽量小。调不动了,点"交给最小二乘"。

**你会看到**: 平线时是 6.60。斜率调到 1.4 上下、高度调到 −2.1 上下时,降到 3.90 左右。点"交给最小二乘",得到 Δ = −7.73 + 1.41 × 壁厚,也是 3.90: 你怎么调,都低不过它。

**这说明**: "最好的直线"不是看着顺眼的那一条。有一把尺子(误差平方的平均),让尺子读数最小的直线只有一条,而且可以直接算出来。

<div class="widget" data-widget="fitline">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

图里的竖线是竖直的,不垂直于黑线。这不是画错了: 误差量的是"壁厚相同时,实际的 Δ 和预测的 Δ 差多少",所以沿着纵轴量,单位是 Gy。到直线的垂直距离是另一种方法用的,它把壁厚也当成有误差的量;那样量,距离里混着 mm 和 Gy,把横轴的单位从 mm 换成 cm,最好的直线就变了。

图下面那行字把黑线写成了公式: 预测 Δ = −7.73 + 1.41 × 壁厚。两个数各有名字:

- **斜率**(slope) 1.41: 壁厚每厚 1 mm,预测的 Δ 高 1.41 Gy,也就是少降 1.41 Gy。
- **截距**(intercept) −7.73: 壁厚是 0 时的预测值。没有哪位病人的壁厚是 0,所以它没有临床意义,只是把直线放到合适高度的那个数。图里的滑块用"壁厚 4 mm 处的预测值"来表示高度,就是因为它比截距好懂。

## 3. 它在算什么: 最小二乘

<div class="frame">

**背景**: 给每位病人猜同一个数,误差平方的平均是 7.96。Δ 明显跟着壁厚变,这个信息还没用上。

**目标**: 找一条公式,把壁厚换算成预测的 Δ,让预测值离实际值尽量近。

**做法**: 公式定成一条直线,预测 Δ = b₀ + b₁ × 壁厚;在所有直线里,取误差平方和最小的那一条。这个办法叫**最小二乘**(least squares)。

</div>

**它在算什么**

1. 先定公式的样子: 预测 Δ = b₀ + b₁ × 壁厚。b₀ 是截距,b₁ 是斜率,现在都还不知道。
2. 任给一组 b₀、b₁,就能对每位病人算出预测值,再算误差: 实际值 − 预测值。
3. 把每个误差平方,全部加起来,得到**误差平方和**。(除以人数就是上面的均方误差;人数是固定的,让哪一个最小都是同一条线。)
4. 找出让误差平方和最小的那一组 b₀、b₁。不用一条一条地试,有现成的公式,第 4 节用代码写出来。

下面这张图在第 2 节那 30 位病人身上把这四步走一遍,最后你会看到第 4 步为什么可以"不用试"。点"下一步",图往前走一步;点 ↻ 把这一步重放一遍。

<div class="widget stage" data-widget="leastsq">

<div class="stage-step" data-label="平线">
<p class="stage-h">先随便取一条线: 平的</p>

黑线对谁都猜平均值 −2.51 Gy。每位病人到黑线的竖线是他的误差。这是第 2 步: 给一条线,算出每个人的误差。

</div>

<div class="stage-step" data-label="平方">
<p class="stage-h">第 3 步: 每个误差画成一个方块</p>

方块的边长是误差,面积就是误差的平方。误差大一倍,方块大四倍: 差得远的病人占的面积特别大。

30 个方块的面积取平均,是 6.60。要找的,就是让方块的总面积最小的那条线。

</div>

<div class="stage-step" data-label="平均点">
<p class="stage-h">先定高度: 线要穿过平均点</p>

红圈是平均点: 平均壁厚 3.70 mm,平均 Δ −2.51 Gy。把整条线抬高 1.5 Gy,方块的平均面积从 6.60 变成 8.85;压低 1.5 Gy,也是 8.85。不管斜率是多少,最好的高度都是让线穿过平均点的那一个。

所以高度不用找了,剩下的只有一件事: 斜率。

</div>

<div class="stage-step" data-label="转一圈">
<p class="stage-h">每个斜率试一遍</p>

让线绕着平均点转,斜率从 −1 到 4。方块跟着变大变小;图下面记下每个斜率对应的误差。

记下来的点连成一条光滑的碗形曲线: 斜率 −1 时 11.79,斜率 1 时 4.12,斜率 4 时 13.02。

</div>

<div class="stage-step" data-label="最低点">
<p class="stage-h">第 4 步: 碗底可以直接算</p>

碗只有一个底: 斜率 1.41,误差 3.90。这条曲线是抛物线,抛物线的最低点有公式,不用一个一个试: 斜率 = 壁厚和 Δ 一起偏离平均值的程度 ÷ 壁厚自己偏离平均值的程度 = 57.53 ÷ 40.81 = 1.41。

放完以后可以拖滑块,让线离开碗底,看方块怎么变大。

</div>

<div class="stage-step" data-label="新病人">
<p class="stage-h">写成公式,给新病人用</p>

截距让线穿过平均点: −2.51 − 1.41 × 3.70 = −7.73。于是预测 Δ = −7.73 + 1.41 × 壁厚,和你在上一节调出来、再"交给最小二乘"得到的是同一条线。

来一位壁厚 3.0 mm 的新病人,顺着线读: −3.50 Gy。

</div>

</div>

拿前三位病人,把第 2、3 步手算一遍。先随手取一条线: 预测 Δ = −8 + 1.5 × 壁厚。

| 病人 | 壁厚 (mm) | 实际 Δ | 预测 Δ = −8 + 1.5 × 壁厚 | 误差 | 误差的平方 |
|---|---|---|---|---|---|
| P001 | 3.0 | −4.0 | −3.5 | −0.5 | 0.25 |
| P002 | 2.6 | −2.6 | −4.1 | +1.5 | 2.25 |
| P003 | 3.8 | −5.6 | −2.3 | −3.3 | 10.89 |

三个人的误差平方和是 0.25 + 2.25 + 10.89 = 13.39。第三位病人差得最远,一个人就占了 10.89。对全部 196 位病人这样算,再换一条线重算,比较谁的和更小: 最小二乘做的就是这件事,只是它用公式一步到位。

**为什么要平方**

- 误差有正有负。直接相加,猜高的和猜低的会互相抵消,一条很差的线也能得到接近 0 的和。
- 平方让大的误差更显眼: 一位病人差 4 Gy(平方是 16),比两位病人各差 2 Gy(4 + 4 = 8)算得更糟。临床上也是这样: 偶尔错得离谱,比经常小错更要紧。
- 平方以后,最小值可以用公式直接算出来,而且只有一个答案。第 1 节看到的"都猜平均值最好",就是它最简单的情形。

不平方、取绝对值也可以,那是另一种回归,对个别离得很远的病人不那么敏感。这门课评价模型时常用绝对值(第 9 课),因为它的单位就是 Gy,好读;找直线时用平方。

**打个比方**: 卡铂按 Calvert 公式给药: 剂量(mg) = 目标 AUC × (GFR + 25)。肾功能每高一点,剂量按固定的比例加上去;GFR 是 0 时剂量也不是 0,因为括号里还有那个 25。有斜率,有截距,这就是一条直线,当年是从一批病人的数据里总结出来的。线性回归做的是同一件事: 从过去的病人里,把斜率和截距找出来,给下一位病人用。

<div class="quiz">
<p><b>想一想</b>: 一条线对 196 位病人的误差(不平方)加起来正好是 0。能说明这条线预测得好吗?</p>
<button class="quiz-opt" data-why="误差的和是 0,只说明猜高的和猜低的总量相等。第 1 节'都猜平均值'的误差之和也是 0,而它是最粗的猜法。">能: 和是 0,说明没有误差</button>
<button class="quiz-opt" data-correct data-why="猜高的和猜低的抵消了,和是 0 并不表示每个人都猜得准。'都猜平均值'的误差之和就是 0。所以要先平方(或取绝对值),再相加。">不能: 正的和负的互相抵消了</button>
<button class="quiz-opt" data-why="和是 0 与人数无关。问题在于正负抵消,不在于人数。">要看病人够不够多</button>
</div>

## 4. 从零算一遍,再用 scikit-learn

第 4 步的公式只有两行。斜率是"壁厚和 Δ 一起偏离各自平均值的程度",除以"壁厚自己偏离平均值的程度";截距让直线穿过"平均壁厚、平均 Δ"这个点。

```python
x = df["wall_mean"]
y = df["delta"]

b1 = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()    # 斜率
b0 = y.mean() - b1 * x.mean()                                                 # 截距
print(f"预测 Δ = {b0:.2f} + {b1:.2f} × 壁厚")

pred = b0 + b1 * x                                                            # 每位病人的预测值
print(f"误差平方的平均: {((y - pred) ** 2).mean():.2f}")
print(f"随手取的那条线: {((y - (-8 + 1.5 * x)) ** 2).mean():.2f}")
```

196 位病人的最小二乘直线是: 预测 Δ = −8.08 + 1.62 × 壁厚。误差平方的平均是 4.87,比基线的 7.96 小了不少;上面随手取的那条线是 5.02,已经很接近,但还是差一点。

实际工作里不用自己写公式,用 scikit-learn 的 `LinearRegression`。它做的是同一件事:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(df[["wall_mean"]], y)       # fit: 找出让误差平方和最小的截距和斜率
print(f"截距 {model.intercept_:.2f},斜率 {model.coef_[0]:.2f}")
```

也是 −8.08 和 1.62。两点说明:

- `fit(X, y)` 的第一个参数是"用来预测的变量",要给一张**表**,所以写两层方括号 `df[["wall_mean"]]`(Python 课第 2 课讲过一层和两层的区别)。第二个参数是要预测的那一列。
- 截距在 `model.intercept_` 里,斜率在 `model.coef_` 里。`coef_` 是一个数组,每个变量一个数;现在只有一个变量,所以取 `[0]`。

<div class="exercise">

**练习 1 · 轮到你: 图形优化的直线**

先算出图形优化的 Δ(`delta_gra`),再用上面的公式算它的斜率和截距,存进 `b1_gra` 和 `b0_gra`。

```python
df["delta_gra"] = df["d2cc_gra"] - df["d2cc_5mm"]
yg = df["delta_gra"]

b1_gra = ((x - x.mean()) * (yg - ___)).sum() / ((x - x.mean()) ** 2).sum()
b0_gra = yg.mean() - ___ * x.mean()
print(f"预测 Δ = {b0_gra:.2f} + {b1_gra:.2f} × 壁厚")
```

```pycheck
need("b1_gra", "b0_gra")
if close(b1_gra, 1.622, 0.005):
    fail("这是体积优化的斜率 1.62。第一个空要用图形优化的平均值: yg.mean()。")
if close(b1_gra, 1.809, 0.005) and close(b0_gra, 3.18, 0.02):
    fail("斜率对了。截距的公式是 yg.mean() − b1_gra × x.mean(): 检查第二个空填的是不是 b1_gra。")
if close(b1_gra, 1.809, 0.005) and close(b0_gra, -10.18, 0.02):
    ok("对了: 预测 Δ = −10.18 + 1.81 × 壁厚。图形优化的斜率比体积优化的(1.62)大一点: 壁厚每厚 1 mm,少降 1.81 Gy。")
fail(f"现在斜率是 {b1_gra:.2f}、截距是 {b0_gra:.2f},应该是 1.81 和 −10.18。第一个空是 yg.mean(),第二个空是 b1_gra。")
```

<details><summary>提示</summary>

和上面体积优化的两行一样,只是把 `y` 换成 `yg`: 第一个空是 `yg.mean()`,第二个空是刚算出的斜率 `b1_gra`。

</details>

<details><summary>参考答案</summary>

```python
df["delta_gra"] = df["d2cc_gra"] - df["d2cc_5mm"]
yg = df["delta_gra"]

b1_gra = ((x - x.mean()) * (yg - yg.mean())).sum() / ((x - x.mean()) ** 2).sum()
b0_gra = yg.mean() - b1_gra * x.mean()
print(f"预测 Δ = {b0_gra:.2f} + {b1_gra:.2f} × 壁厚")
```

</details>

</div>

## 5. 用这条线: 预测一位新病人,看差了多少

回到开头的那位病人: 平均壁厚 3.0 mm。把 3.0 代进公式:

```python
print(f"手算: {b0 + b1 * 3.0:.2f} Gy")

new_patient = pd.DataFrame({"wall_mean": [3.0]})          # 新病人也要放进一张表,列名和建模时一样
print(f"model.predict: {model.predict(new_patient)[0]:.2f} Gy")
```

两种算法都是 −3.21 Gy: 预测换成体积优化后,这位病人的 D2cc 降 3.2 Gy 左右。`predict` 做的就是"代进公式"这一步。

预测值不是实际值。队列里第一位病人(P001)的壁厚正好也是 3.0 mm,实际降了 4.0 Gy。实际值减预测值,−4.0 − (−3.21) = −0.79 Gy,叫这位病人的**残差**(residual)。它就是前面说的误差,只是专指"用拟合好的模型算出来的误差"。

一位病人说明不了什么,看全部 196 位:

```python
residual = y - model.predict(df[["wall_mean"]])
print(f"平均差多少(不管正负): {residual.abs().mean():.2f} Gy")
print(f"差在 2 Gy 以内: {(residual.abs() <= 2).sum()} 位;差在 4 Gy 以内: {(residual.abs() <= 4).sum()} 位")
print(f"差得最多的一位: {residual.abs().max():.2f} Gy")

fig, ax = plt.subplots()
ax.hist(residual, bins=20)
ax.set_xlabel("Residual: actual - predicted (Gy)")
ax.set_ylabel("Number of patients")
plt.show()
```

预测值和实际值平均差 1.78 Gy。196 位里 126 位(64%)差在 2 Gy 以内,184 位(94%)差在 4 Gy 以内,差得最多的一位差了 6.72 Gy。"平均差多少"就是第 4 课要用的**平均绝对误差**(mean absolute error, MAE);基线模型(都猜平均值)的 MAE 是 2.24 Gy,这条线把它降到了 1.78 Gy。

所以,对那位新病人更完整的说法是: 预计降 3.2 Gy 左右,多数情况下和实际相差不到 2 Gy,个别病人会差出 4 Gy 以上。报告一个预测值而不说它有多准,是只说了一半。

这里的 1.78 Gy 是在**建模用的病人**身上算的。模型见过这些病人,对他们的成绩会偏好一些;对没见过的新病人准不准,是第 4 课的主题。

<div class="quiz">
<p><b>想一想</b>: 一位病人的平均壁厚是 12 mm。代进公式,预测 Δ = −8.08 + 1.62 × 12 = +11.4 Gy。这个预测能用吗?</p>
<button class="quiz-opt" data-why="公式确实能算出一个数,但建模的 196 位病人,壁厚在 1.6 到 7.5 mm 之间。12 mm 处直线是不是还成立,数据里没有任何依据。">能: 公式对任何壁厚都能算</button>
<button class="quiz-opt" data-correct data-why="建模的病人壁厚在 1.6 到 7.5 mm 之间,直线只在这个范围里被数据检验过。伸到范围外面去用,叫外推(extrapolation);模型不会报错,但结果没有依据。先用 describe() 看看每个变量的范围,是好习惯。">不能: 队列里没有这么厚的病人,直线在那里成不成立不知道</button>
<button class="quiz-opt" data-why="壁厚的系数是正的,壁越厚预测越高,+11.4 在算术上没有错。问题不在正负,而在 12 mm 超出了建模病人的范围。">不能: Δ 应该是负数,正数说明算错了</button>
</div>

<div class="exercise">

**练习 2 · 轮到你: 图形优化的预测和残差**

用练习 1 的 `b0_gra`、`b1_gra`: 壁厚 3.0 mm 的病人,图形优化的预测 Δ 是多少?存进 `pred_gra`。队列里第一位病人(P001,壁厚 3.0 mm)的残差是多少?存进 `resid_first`。

```python
pred_gra = b0_gra + b1_gra * ___
resid_first = yg.iloc[0] - ___
print(f"预测 {pred_gra:.2f} Gy;P001 实际 {yg.iloc[0]:.1f} Gy,残差 {resid_first:.2f} Gy")
```

```pycheck
need("pred_gra", "resid_first")
if close(pred_gra, -3.21, 0.02):
    fail("这是体积优化的预测值。要用图形优化的 b0_gra 和 b1_gra。")
if not close(pred_gra, -4.76, 0.02):
    fail(f"pred_gra 现在是 {pred_gra:.2f},应该是 −4.76: b0_gra + b1_gra × 3.0。")
if close(resid_first, -6.86, 0.02):
    fail("符号反了: 残差是实际值减预测值,yg.iloc[0] − pred_gra。")
if close(resid_first, 6.86, 0.02):
    ok("对了: 预测降 4.76 Gy,这位病人实际却升了 2.1 Gy,残差 +6.86 Gy。同样是壁厚 3.0 mm,体积优化只差 0.79 Gy。一条线说的是'这样的病人平均会怎样',落到某一位病人身上,可以差得很远。")
fail(f"resid_first 现在是 {resid_first:.2f},应该是 6.86: 实际值 yg.iloc[0] 减去预测值 pred_gra。")
```

<details><summary>提示</summary>

第一个空是壁厚 `3.0`;第二个空是刚算出的预测值 `pred_gra`(P001 的壁厚正好是 3.0 mm)。

</details>

<details><summary>参考答案</summary>

```python
pred_gra = b0_gra + b1_gra * 3.0
resid_first = yg.iloc[0] - pred_gra
print(f"预测 {pred_gra:.2f} Gy;P001 实际 {yg.iloc[0]:.1f} Gy,残差 {resid_first:.2f} Gy")
```

</details>

</div>

## 6. 不止一个变量

Δ 不只跟壁厚有关。施源器的直径、近距离做几次,在治疗前也都知道,都可以放进公式:

> 预测 Δ = b₀ + b₁ × 平均壁厚 + b₂ × 施源器直径 + b₃ × 近距离次数

**它在算什么**: 和一个变量时完全一样的四步。公式多了几项,任给一组 b 仍然能算出每位病人的预测值和误差;最小二乘找的,还是让误差平方和最小的那一组 b。一个变量时是一条直线,两个变量时是一个平面,再多就画不出来了,算术没有变。

```python
X = df[["wall_mean", "diameter", "n_fx"]]
model3 = LinearRegression().fit(X, y)
b0_3 = model3.intercept_
b_wall, b_diam, b_fx = model3.coef_
print(f"预测 Δ = {b0_3:.2f} {b_wall:+.2f} × 壁厚 {b_diam:+.2f} × 直径 {b_fx:+.2f} × 次数")

pred3 = model3.predict(X)
print(f"误差平方的平均: {((y - pred3) ** 2).mean():.2f};平均差 {(y - pred3).abs().mean():.2f} Gy")
```

公式是: 预测 Δ = −9.94 + 1.57 × 壁厚 + 0.17 × 直径 − 1.21 × 次数。误差平方的平均从 4.87 降到 4.21,平均差从 1.78 Gy 降到 1.67 Gy: 多用了两个变量,预测准了一点。

**系数怎么读**: 每个系数说的是,**其他变量不变时**,这个变量增加 1 个单位,预测值变多少。拿两位只差一样的病人验证一下:

```python
two = pd.DataFrame({"wall_mean": [3.0, 3.0], "diameter": [30, 30], "n_fx": [2, 3]})
p = model3.predict(two)
print(f"做 2 次: {p[0]:.2f} Gy;做 3 次: {p[1]:.2f} Gy;相差 {p[1] - p[0]:.2f} Gy")
```

壁厚都是 3.0 mm、直径都是 30 mm,一位做 2 次、一位做 3 次: 预测值是 −2.64 Gy 和 −3.84 Gy,相差 −1.21 Gy,正是次数的系数。多做一次近距离,累计的 D2cc 多降 1.21 Gy。

壁厚的系数从 1.62 变成了 1.57。模型里有谁,会影响每个系数: 1.62 是"壁厚每厚 1 mm,Δ 平均差多少",1.57 是"直径和次数都相同的病人里,壁厚每厚 1 mm,Δ 平均差多少"。这里两个数很接近,因为在这个队列里,壁厚和另外两个变量几乎不相关;如果壁厚的病人恰好也多做一次近距离,两个数就会差得多。这件事第 3 课专门讲。

<div class="quiz">
<p><b>想一想</b>: 直径的系数是 0.17,次数的系数是 −1.21。能不能说"近距离次数对 Δ 的影响是直径的 7 倍"?</p>
<button class="quiz-opt" data-why="系数的大小取决于变量的单位。直径的 0.17 是'每 1 mm',次数的 −1.21 是'每 1 次',单位不同的两个数不能直接比。">能: 系数的绝对值越大,影响越大</button>
<button class="quiz-opt" data-correct data-why="0.17 是直径每大 1 mm 的变化,−1.21 是次数每多 1 次的变化。队列里直径从 25 到 35 mm,差 10 mm 就是 1.7 Gy;次数只有 2 和 3,差 1 次是 1.2 Gy。按各自实际的变化范围算,两者差不多。要比较,先想清楚'各变化多少'。">不能: 两个系数的单位不同,要看各自实际能变化多少</button>
<button class="quiz-opt" data-why="正负只表示方向: 直径越大 Δ 越高,次数越多 Δ 越低。方向不同不妨碍比较大小,妨碍比较的是单位。">不能: 一正一负,没法比</button>
</div>

<div class="exercise">

**练习 3 · 轮到你: 图形优化的三变量模型**

用同样的三个变量预测图形优化的 Δ(`yg`),模型存进 `model3_gra`。然后取出近距离次数的系数,存进 `coef_fx`。

```python
model3_gra = LinearRegression().fit(X, ___)
coef_fx = model3_gra.coef_[___]
print(f"截距 {model3_gra.intercept_:.2f},系数 {model3_gra.coef_.round(2)},次数的系数 {coef_fx:.2f}")
```

```pycheck
need("model3_gra", "coef_fx")
if close(coef_fx, -1.206, 0.005):
    fail("这是体积优化的模型。fit 的第二个参数要换成图形优化的 yg。")
if close(coef_fx, 1.761, 0.005):
    fail("这是壁厚的系数(第 0 个)。X 的三列依次是壁厚、直径、次数,次数是第 2 个。")
if close(coef_fx, -0.069, 0.005):
    fail("这是直径的系数(第 1 个)。次数是第 2 个。")
if close(coef_fx, -1.81, 0.005):
    ok("对了: 预测 Δ = −3.50 + 1.76 × 壁厚 − 0.07 × 直径 − 1.81 × 次数。壁厚和直径相同的病人,多做一次近距离,图形优化的 D2cc 多降 1.81 Gy。")
fail(f"coef_fx 现在是 {coef_fx:.2f},应该是 −1.81。")
```

<details><summary>提示</summary>

第一个空是 `yg`。`coef_` 里系数的顺序和 `X` 的列一样,从 0 开始数: 壁厚 0,直径 1,次数 2。

</details>

<details><summary>参考答案</summary>

```python
model3_gra = LinearRegression().fit(X, yg)
coef_fx = model3_gra.coef_[2]
print(f"截距 {model3_gra.intercept_:.2f},系数 {model3_gra.coef_.round(2)},次数的系数 {coef_fx:.2f}")
```

</details>

</div>

## 7. 是/否的变量: 写成 0 和 1

公式里只能放数。治疗方式(`arm`)是文字: "VCB"(单纯近距离)或 "EBRT+VCB"(外照射加近距离)。放进模型之前,先把它写成 0 和 1: 做了外照射记 1,没做记 0。

```python
df["ebrt"] = (df["arm"] == "EBRT+VCB").astype(int)        # 是 → 1,否 → 0
print(pd.crosstab(df["arm"], df["ebrt"]))

X_e = df[["wall_mean", "diameter", "ebrt"]]
model_e = LinearRegression().fit(X_e, y)
print(f"预测 Δ = {model_e.intercept_:.2f} {model_e.coef_[0]:+.2f} × 壁厚 {model_e.coef_[1]:+.2f} × 直径 {model_e.coef_[2]:+.2f} × 外照射")
```

公式是: 预测 Δ = −13.56 + 1.57 × 壁厚 + 0.17 × 直径 + 1.21 × 外照射。`ebrt` 只取 0 和 1,所以它的系数读起来最简单: **壁厚和直径相同时,做了外照射的病人比没做的,Δ 平均高 1.21 Gy**(少降 1.21 Gy)。0/1 变量的系数,就是两组之间的差。

1.21 这个数上一节见过。在这个队列里,做外照射的病人近距离都做 2 次,不做的都做 3 次(上面的交叉表可以看到: 107 位和 89 位)。"做没做外照射"和"做几次"说的是同一件事,所以两个模型的预测值完全一样,只是系数换了写法: 次数每多 1 次是 −1.21,外照射从无到有(次数从 3 变 2)是 +1.21。

<div class="quiz">
<p><b>想一想</b>: 既然这样,把 <code>n_fx</code> 和 <code>ebrt</code> 两个变量都放进模型,会怎样?</p>
<button class="quiz-opt" data-why="不会更准。第二个变量没有带来任何新信息: 知道做没做外照射,就知道了做几次。">预测更准: 变量多了,信息也多了</button>
<button class="quiz-opt" data-correct data-why="两个变量带的是同一份信息,1.21 这个差可以任意分给两个系数(各分一半,或者全给其中一个),预测值都一样,所以系数没有唯一的答案。scikit-learn 不会报错,会给出其中一种分法,但这样的系数不能解读。两个变量说的是同一件事时,只放一个。">预测值不变,但两个系数不再有确定的含义</button>
<button class="quiz-opt" data-why="scikit-learn 不会报错: 它会给出一组系数,预测值也照常能算。麻烦在于这组系数只是无数种分法里的一种,不能解读。">程序报错,不能运行</button>
</div>

<div class="exercise">

**练习 4 · 轮到你: 图形优化里外照射的系数**

用壁厚、直径和 `ebrt` 预测图形优化的 Δ,取出 `ebrt` 的系数,存进 `coef_ebrt`。

```python
model_e_gra = LinearRegression().fit(df[["wall_mean", "diameter", ___]], yg)
coef_ebrt = model_e_gra.coef_[___]
print(f"外照射的系数: {coef_ebrt:.2f}")
```

```pycheck
need("coef_ebrt")
if close(coef_ebrt, 1.206, 0.005):
    fail("这是体积优化的系数。fit 的第二个参数应该是 yg(起始代码里已经是),检查有没有改动。")
if close(coef_ebrt, -1.81, 0.005):
    fail('符号是负的: 放进去的是 n_fx(次数),不是 ebrt。第一个空填 "ebrt"。')
if close(coef_ebrt, 1.761, 0.005) or close(coef_ebrt, -0.069, 0.005):
    fail("取错了位置。三列依次是壁厚、直径、ebrt,ebrt 是第 2 个(从 0 开始数)。")
if close(coef_ebrt, 1.81, 0.005):
    ok("对了: +1.81。壁厚和直径相同时,做了外照射的病人,图形优化的 Δ 平均高 1.81 Gy。它和练习 3 里次数的系数 −1.81 是同一件事的两种写法。")
fail(f"coef_ebrt 现在是 {coef_ebrt:.2f},应该是 1.81。")
```

<details><summary>提示</summary>

第一个空是列名 `"ebrt"`(带引号),第二个空是它在三列里的位置 `2`。

</details>

<details><summary>参考答案</summary>

```python
model_e_gra = LinearRegression().fit(df[["wall_mean", "diameter", "ebrt"]], yg)
coef_ebrt = model_e_gra.coef_[2]
print(f"外照射的系数: {coef_ebrt:.2f}")
```

</details>

</div>

最后给图形优化的模型打个分,和基线比一比。

<div class="exercise">

**练习 5 · 轮到你: 图形优化的模型准了多少**

算练习 3 的三变量模型(`model3_gra`)在这 196 位病人身上的平均绝对误差,存进 `mae_gra`。再算基线模型(都猜平均值)的,存进 `mae_base`。

```python
pred_g = model3_gra.predict(___)
mae_gra = (yg - pred_g).abs().___()
mae_base = (yg - yg.mean()).abs().mean()
print(f"基线 {mae_base:.2f} Gy,三变量模型 {mae_gra:.2f} Gy")
```

```pycheck
need("mae_gra", "mae_base")
if close(mae_gra, 9.31, 0.02):
    fail("9.31 是误差平方的平均。题目要的是平均绝对误差: 先 abs() 再 mean()(起始代码里已经有 abs(),第二个空填 mean)。")
if close(mae_gra, 474.3, 1):
    fail("这是绝对误差的总和。要平均值: 第二个空填 mean。")
if close(mae_gra, 2.51, 0.006):
    fail("这是只用壁厚的那条线的成绩。predict 里要给三列的表 X。")
if close(mae_gra, 2.42, 0.006) and close(mae_base, 3.0, 0.01):
    ok("对了: 基线 3.00 Gy,三变量模型 2.42 Gy。图形优化比体积优化难预测: 同样的三个变量,体积优化的平均误差是 1.67 Gy。记住这两个数都是在建模病人身上算的,第 4 课会重新算一遍诚实的版本。")
fail(f"mae_gra 现在是 {mae_gra:.2f},应该是 2.42: predict 里是 X,最后取 mean()。")
```

<details><summary>提示</summary>

`predict` 要的是建模时用的那张三列的表 `X`;绝对值之后取平均,用 `mean`。

</details>

<details><summary>参考答案</summary>

```python
pred_g = model3_gra.predict(X)
mae_gra = (yg - pred_g).abs().mean()
mae_base = (yg - yg.mean()).abs().mean()
print(f"基线 {mae_base:.2f} Gy,三变量模型 {mae_gra:.2f} Gy")
```

</details>

</div>

## 小结

- 一个预测模型由三样东西组成: **公式的样子**(这里是直线)、**量好坏的尺子**(误差平方和)、**找最好那组数的办法**(最小二乘)。
- 给所有人猜平均值是**基线模型**;任何模型都要和它比。
- **斜率**: 变量每增加 1 个单位,预测值变多少。**截距**: 所有变量都是 0 时的预测值,常常没有临床意义。
- 多个变量时,每个系数的意思是"**其他变量不变时**";模型里换了变量,系数会跟着变。
- 0/1 变量的系数是两组之间的差。两个变量说的是同一件事时,只放一个。
- 预测值要和它的误差一起报告;只在建模病人的范围以内用,不外推。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 模型 | model | 把已知的信息换算成预测值的一条公式 |
| 线性回归 | linear regression | 公式是"截距 + 系数 × 变量"的模型 |
| 基线模型 | baseline model | 对所有人猜同一个数(平均值) |
| 截距 | intercept | 所有变量为 0 时的预测值 |
| 斜率、系数 | slope, coefficient | 变量增加 1 个单位,预测值变多少 |
| 最小二乘 | (ordinary) least squares, OLS | 取误差平方和最小的那组系数 |
| 均方误差 | mean squared error (MSE) | 误差平方的平均 |
| 残差 | residual | 实际值减去模型的预测值 |
| 预测变量 | predictor | 用来预测的变量,也叫特征(feature) |
| 结局 | outcome | 要预测的量 |
| 外推 | extrapolation | 在建模数据的范围以外使用模型 |
| 哑变量 | dummy variable | 把是/否写成 1 和 0 的变量 |

## 这一课没讲什么

- **系数的置信区间和 P 值**(回归表里的另外几列): 第 3 课。
- **对新病人准不准**: 这一课的误差都是在建模病人身上算的,偏乐观。第 4 课。
- **不是直线的关系**: 如果残差随着壁厚有规律地偏向一边,说明直线不够用,可以加平方项或者换模型。第 7 课的树模型不需要事先假定直线。
- **交互作用**(interaction): 这个模拟队列里,壁厚的斜率在两种治疗方式里其实不一样(分开拟合,一组是 1.02,一组是 2.42)。让斜率随另一个变量变化,要在公式里加交互项;这门课不展开。
- **三个以上类别的变量**(比如分期 IA、IB、II): 要拆成几个 0/1 变量,用 `pd.get_dummies`。第 5 课会用到。

## 写进论文 Writing it up

<div class="paper">

方法部分可以这样写(变量换成你实际用的):

> The change in vaginal D2cc after re-optimisation (Δ, Gy) was modelled by ordinary least-squares linear regression with mean vaginal wall thickness, applicator diameter and number of brachytherapy fractions as predictors. Coefficients are reported as the change in Δ per unit increase of each predictor with the other predictors held constant.

**审稿人可能问**: 为什么假定是直线关系?

**怎么回答**: 画出残差对每个预测变量的散点图,说明没有看到有规律的弯曲;或者报告加入平方项以后模型没有改善。如果看到了弯曲,就如实改用更灵活的模型。

**审稿人可能问**: 预测误差是在哪些病人身上算的?

**怎么回答**: 如果是在建模用的病人身上算的,要明说这是表观误差(apparent performance),并补上交叉验证的结果(第 4 课)。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| `ValueError: Expected 2D array, got 1D array instead` | 给 `fit` 或 `predict` 的是一列(一层方括号)。预测变量要给一张表: `df[["wall_mean"]]`。 |
| `ValueError: Input X contains NaN` | 数据里有缺失值,线性回归不接受。这一课先用 `dropna()` 去掉;第 5 课讲怎样补。 |
| `UserWarning: X does not have valid feature names` | `predict` 时给的是没有列名的数。把新病人放进一张列名和建模时相同的表。 |
| `ValueError: The feature names should match those that were passed during fit` | 新病人的表,列名或列的顺序和建模时不一样。 |
| 结果和网页上写的不一样 | 本页的代码块要从上往下依次运行;练习的变量(如 `yg`、`b1_gra`)要先通过前面的练习。 |

## 延伸阅读 Further reading

- scikit-learn 用户指南: [Linear Models, Ordinary Least Squares](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)。`LinearRegression` 的说明。
- Schneider A, Hommel G, Blettner M. [Linear regression analysis: part 14 of a series on evaluation of scientific publications](https://pmc.ncbi.nlm.nih.gov/articles/PMC2992018/). *Dtsch Arztebl Int* 2010;107(44):776–782。写给临床医生的线性回归综述,可以引用。
- Calvert AH, Newell DR, Gumbrell LA, 等. Carboplatin dosage: prospective evaluation of a simple formula based on renal function. *J Clin Oncol* 1989;7(11):1748–1756。第 3 节打比方用的那条给药公式的出处。

**下一课**: 这一课预测的是一个连续的量(变了多少 Gy)。临床上更常问的是"会不会": 换方案以后,D2cc 会不会降到 68 Gy 以下?结局只有是和否,直线就不合适了。第 2 课讲逻辑回归: 同样的三件事,换了公式和尺子。
