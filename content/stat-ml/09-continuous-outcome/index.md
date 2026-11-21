---
title: "Lesson 9 · 连续结局的评价 Evaluating a Continuous Outcome"
summary: "MAE、RMSE 和 R² 各量的是什么;R² 为什么会是负的;预测值为什么比实际值保守;几个相关的结局怎么报告。"
weight: 90
toc: true
duration: "约 70 分钟,可以分两次学: 第 0–2 节,第 3–5 节"
prereq: "本课程[第 4 课](../04-cross-validation/)(交叉验证和 out-of-fold 预测)"
questions:
  - "平均差 1.7 Gy,算好还是算差?"
  - "R² 是什么?它为什么会是负的?"
  - "模型对最极端的患者总是说得不够极端,是模型的毛病吗?"
  - "一个队列有好几个结局,成绩怎么比、怎么报告?"
objectives:
  - "自己算出 MAE、RMSE 和 R²,说出三者各回答什么问题"
  - "解释留出患者身上的 R² 为什么可以是负的,并分清 R² 和相关系数的平方"
  - "按预测值分组检查模型,算出校准斜率"
  - "用 R² 比较不同的结局,用 MAE 向临床报告"
---

前面八课评价模型,用的一直是同一把尺子: 预测值和实际值平均差多少 Gy(MAE)。它好懂,但回答不了一个问题: **1.7 Gy 算好还是算差?**

体积优化的模型平均差 1.70 Gy,图形优化的平均差 2.46 Gy。能说前一个模型更好吗? 不一定: 图形优化的 Δ 本身就变化得更大,更难猜。评价一个预测连续结局的模型,要同时回答三个问题:

1. 平均差多少(带单位,临床能读)?
2. 和"不用模型"相比,好了多少(不带单位,能跨结局比较)?
3. 预测值能不能照字面相信(模型说 −4 Gy 的患者,实际平均是不是 −4 Gy)?

这一课一个一个讲。

**练习怎么做**: 讲解用**体积优化**计划的 Δ,练习换成**图形优化**。

## 0. 准备

```python
import numpy as np
import pandas as pd
from cohort import make_cohort
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score

def prepare(d):
    d = d.dropna().copy()
    d["delta"] = d["d2cc_vol"] - d["d2cc_5mm"]
    d["delta_gra"] = d["d2cc_gra"] - d["d2cc_5mm"]
    d["delta_d90"] = d["d90_vol"] - d["d90_5mm"]             # 靶区 D90 的变化,第 4 节用
    d["delta_d90_gra"] = d["d90_gra"] - d["d90_5mm"]
    return d

df, new = prepare(make_cohort()), prepare(make_cohort(n=5000, seed=99))
cols = ["wall_mean", "diameter", "n_fx"]
cv = KFold(n_splits=5, shuffle=True, random_state=0)

def out_of_fold(model, columns, outcome):
    """每位患者被留出的那一折里得到的预测值(第 4 课)。"""
    return pd.Series(cross_val_predict(model, df[columns], df[outcome], cv=cv), index=df.index)

y = df["delta"]
oof = out_of_fold(LinearRegression(), cols, "delta")
base = out_of_fold(DummyRegressor(), cols, "delta")                      # 基线: 都猜建模患者的平均值
print(f"{len(y)} 位患者,每人一个留出时的预测值。")
print(f"模型的 MAE {(y - oof).abs().mean():.2f} Gy,基线的 MAE {(y - base).abs().mean():.2f} Gy")
```

`oof` 里是 196 个预测值,每一个都是这位患者**没有参与建模**时得到的。这一课所有的成绩都用它算: 评价模型,要在它没见过的患者身上评价。模型平均差 1.70 Gy,都猜平均值是 2.25 Gy。

## 1. 三把尺子: MAE、RMSE、R²

```python
error = y - oof
mae = error.abs().mean()                       # 误差取绝对值,平均
rmse = np.sqrt((error ** 2).mean())            # 误差先平方,平均,再开方
print(f"MAE {mae:.2f} Gy,RMSE {rmse:.2f} Gy")

worst = error.abs().sort_values(ascending=False).head(10)
print(f"误差最大的 10 位患者: 占绝对误差总和的 {worst.sum() / error.abs().sum():.0%},占误差平方总和的 {(worst ** 2).sum() / (error ** 2).sum():.0%}")
```

- **MAE**(mean absolute error): 平均差 1.70 Gy。每位患者的误差一视同仁。
- **RMSE**(root mean squared error): 2.09 Gy。单位也是 Gy,但它先把误差平方,大的误差就占了更大的分量: 误差最大的 10 位患者(196 位里的 5%)只占绝对误差的 14%,却占了误差平方的 26%。

RMSE 总是不小于 MAE。两者差得越多,说明越有少数患者错得离谱。临床上如果"偶尔错很多"比"经常错一点"更要紧,看 RMSE;否则 MAE 更直观。两个都可以报告。

这两个数都带单位,都回答"差多少"。它们回答不了"算好还是算差",因为没有参照。

<div class="frame">

**背景**: 误差 1.70 Gy 的意义取决于 Δ 本身变化有多大。如果所有患者的 Δ 都在 −2 Gy 上下 0.5 Gy 以内,1.70 Gy 是很糟的成绩;如果 Δ 从 −10 到 +5 Gy 都有,1.70 Gy 就不错。

**目标**: 一个不带单位的数,说明模型比"不用模型"好了多少。

**做法**: 拿"都猜平均值"当参照。模型的误差平方和,比参照的误差平方和少了几成,就是 **R²**。

</div>

**它在算什么**

1. 参照: 所有患者都猜平均值。把每位患者的误差平方,加起来。
2. 模型: 每位患者用模型的预测值。同样把误差平方,加起来。
3. R² = 1 − 模型的误差平方和 ÷ 参照的误差平方和。

模型和参照一样差,R² = 0;模型完全猜中,R² = 1。

**打个比方**: 报告一种药的效果,可以说"死亡率从 10% 降到 6%,少了 4 个百分点"(绝对的差,像 MAE),也可以说"死亡少了四成"(和对照比的相对值,像 R²)。两种说法各有用处: 前一种告诉患者实际差多少,后一种可以在不同的研究之间比较。

```python
sse_model = (error ** 2).sum()
sse_base = ((y - y.mean()) ** 2).sum()
r2 = 1 - sse_model / sse_base
print(f"模型的误差平方和 {sse_model:.0f},都猜平均值的 {sse_base:.0f},R² = {r2:.2f}")
print(f"scikit-learn: R² = {r2_score(y, oof):.2f}")
```

R² = 0.45: 和都猜平均值相比,模型把误差平方和减少了 45%。还剩 55% 是这三个变量解释不了的。

<div class="demo">

**设定**: 第 1 课的那 30 位模拟患者。虚线是都猜平均值。黑线是"模型",它经过平均点,斜率由你定。两个条是两种做法的误差平方的平均,下面是由它们算出的 R²。

**你做什么**: 把斜率从 0 慢慢拖到 4,看 R² 怎么变。

**你会看到**: 斜率是 0 时,黑线和虚线重合,R² = 0。斜率 1.4 时误差最小(3.90 比 6.60),R² = 0.41。再往上拖,R² 反而下降;到 2.8,又回到了 0.01;拖到 4,误差是 13.02,R² = −0.97。

**这说明**: R² 是在和"都猜平均值"比。方向对了但说得太过头的模型(这里是斜率大了一倍以上),会比不用模型还差,这时候 R² 是负的。

<div class="widget" data-widget="rsquared">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

<div class="exercise">

**练习 1 · 轮到你: 图形优化的 RMSE 和 R²**

对图形优化的 Δ,用留出时的预测值算 RMSE(`rmse_g`)和 R²(`r2_g`)。R² 要自己按定义算,不用 `r2_score`。

```python
yg = df["delta_gra"]
oof_g = out_of_fold(LinearRegression(), cols, "delta_gra")
error_g = yg - oof_g
rmse_g = np.sqrt((error_g ** 2).___())
r2_g = 1 - (error_g ** 2).sum() / ((yg - yg.___()) ** 2).sum()
print(f"RMSE {rmse_g:.2f} Gy,R² {r2_g:.2f}")
```

```pycheck
need("rmse_g", "r2_g")
if close(r2_g, 0.453, 0.004):
    fail("这是体积优化的结果。结局要用 yg(起始代码里已经是),检查有没有改动。")
if close(rmse_g, 3.112, 0.004) and close(r2_g, 0.311, 0.004):
    ok("对了: RMSE 3.11 Gy,R² 0.31。图形优化的 MAE(2.46 Gy)比体积优化(1.70 Gy)大了四成多;R² 是 0.31 对 0.45。两把尺子都说图形优化更难预测,但 R² 已经扣掉了'图形优化的 Δ 本来就变化更大'这一层。")
fail(f"现在是 RMSE {rmse_g:.3f}、R² {r2_g:.3f},应该是 3.112 和 0.311。两个空都是 mean。")
```

<details><summary>提示</summary>

RMSE 是误差平方的平均再开方: 第一个空是 `mean`。参照是都猜平均值: 第二个空也是 `mean`。

</details>

<details><summary>参考答案</summary>

```python
yg = df["delta_gra"]
oof_g = out_of_fold(LinearRegression(), cols, "delta_gra")
error_g = yg - oof_g
rmse_g = np.sqrt((error_g ** 2).mean())
r2_g = 1 - (error_g ** 2).sum() / ((yg - yg.mean()) ** 2).sum()
print(f"RMSE {rmse_g:.2f} Gy,R² {r2_g:.2f}")
```

</details>

</div>

## 2. R² 会是负的

很多人学过的说法是"R² 在 0 和 1 之间,是模型解释的变异的比例"。这句话只在一种情况下成立: 在**建模的患者**身上算最小二乘模型的 R²。那时模型至少能学会"都猜平均值",R² 不会小于 0;而且变量越多,它只升不降。

在留出的患者身上就不一样了:

```python
rng = np.random.default_rng(9)
Z = pd.DataFrame(rng.normal(size=(len(df), 40)), index=df.index)        # 40 个随机数,和 Δ 毫无关系
fit_z = LinearRegression().fit(Z, y)
oof_z = cross_val_predict(LinearRegression(), Z, y, cv=cv)
print(f"40 个随机数: 建模患者身上 R² {r2_score(y, fit_z.predict(Z)):.2f},留出时 R² {r2_score(y, oof_z):.2f}")

tree = DecisionTreeRegressor(random_state=0)                             # 不限深度的树(第 7 课)
oof_t = out_of_fold(tree, cols, "delta")
print(f"不限深度的树: 建模患者身上 R² {r2_score(y, tree.fit(df[cols], y).predict(df[cols])):.2f},留出时 R² {r2_score(y, oof_t):.2f}")
```

- 40 个随机数: 在建模患者身上 R² 是 0.22,好像"解释了 22% 的变异";留出时是 −0.27,比都猜平均值还差 27%。
- 不限深度的树: 0.80 掉到 0.14。

负的 R² 不是算错了。它如实地说: 这个模型在新患者身上,不如不用。所以看到论文里的 R²,先问一句: **是在谁身上算的?** 在建模患者身上算的 R²(有时写作 apparent R²)几乎不说明问题。

**R² 不是相关系数的平方**。另一个常见的做法,是算预测值和实际值的相关系数 r,把 r² 当作 R² 报告:

```python
r = np.corrcoef(y, oof_t)[0, 1]
print(f"树: 相关系数的平方 {r ** 2:.2f},R² {r2_score(y, oof_t):.2f}")
r = np.corrcoef(y, oof)[0, 1]
print(f"线性回归: 相关系数的平方 {r ** 2:.2f},R² {r2_score(y, oof):.2f}")
```

线性回归的两个数一样(0.45)。树的不一样: r² 是 0.30,R² 只有 0.14。相关系数只管预测值和实际值"是不是一起高、一起低",不管数值对不对: 把所有预测值都乘以 2,相关系数不变,误差却大得多。树的预测值方向大体是对的,但说得太极端(下一节会量出来),所以 r² 比 R² 好看。评价预测,用 R²。

<div class="quiz">
<p><b>想一想</b>: 一篇论文写: "模型纳入 25 个变量,R² = 0.62(n = 80)。"没有提到交叉验证或验证集。这个 0.62 该怎么读?</p>
<button class="quiz-opt" data-why="0.62 确实不低。可是不知道它是在谁身上算的,就不知道它说明了什么。">模型解释了 62% 的变异,效果不错</button>
<button class="quiz-opt" data-correct data-why="80 位患者、25 个变量,在建模患者身上算的 R² 会很高,哪怕变量都是随机数(上面 196 位患者、40 个随机数就有 0.22)。没有留出患者身上的成绩,这个数不能当作预测能力的证据。">多半是在建模患者身上算的;80 位患者配 25 个变量,它会高得没有意义</button>
<button class="quiz-opt" data-why="R² 本身没有问题,它在留出的患者身上是一把好尺子。问题是这里不知道它是在谁身上算的。">R² 不适合评价预测模型</button>
</div>

<div class="exercise">

**练习 2 · 轮到你: 图形优化,不限深度的树**

对图形优化的 Δ,算不限深度的树在建模患者身上的 R²(`r2_train_g`)和留出时的 R²(`r2_oof_g`)。

```python
tree_g = DecisionTreeRegressor(random_state=0)
oof_tg = out_of_fold(tree_g, cols, ___)
r2_oof_g = r2_score(yg, oof_tg)
r2_train_g = r2_score(yg, tree_g.fit(df[cols], yg).___(df[cols]))
print(f"建模患者身上 {r2_train_g:.2f},留出时 {r2_oof_g:.2f}")
```

```pycheck
need("r2_train_g", "r2_oof_g")
if close(r2_oof_g, 0.135, 0.004):
    fail('这是体积优化的结果。第一个空填 "delta_gra"。')
if close(r2_train_g, 0.778, 0.004) and close(r2_oof_g, -0.270, 0.004):
    ok("对了: 建模患者身上 0.78,留出时 −0.27。同一棵树,一个数说它解释了 78% 的变异,另一个数说它比都猜平均值还差 27%。后一个才是它对新患者的本事。")
fail(f"现在是 {r2_train_g:.3f} 和 {r2_oof_g:.3f},应该是 0.778 和 −0.270。第二个空是 predict。")
```

<details><summary>提示</summary>

第一个空是结局 `"delta_gra"`;第二个空是让拟合好的树给出预测值: `predict`。

</details>

<details><summary>参考答案</summary>

```python
tree_g = DecisionTreeRegressor(random_state=0)
oof_tg = out_of_fold(tree_g, cols, "delta_gra")
r2_oof_g = r2_score(yg, oof_tg)
r2_train_g = r2_score(yg, tree_g.fit(df[cols], yg).predict(df[cols]))
print(f"建模患者身上 {r2_train_g:.2f},留出时 {r2_oof_g:.2f}")
```

</details>

</div>

## 3. 预测值比实际值"保守"

把患者按**实际的** Δ 从低到高分成五组,比较每组的实际平均值和预测平均值:

```python
labels = ["最低", "较低", "中间", "较高", "最高"]
print(f"实际值的标准差 {y.std():.2f} Gy,预测值的标准差 {oof.std():.2f} Gy")
by_actual = pd.qcut(y, 5, labels=labels)
print(pd.DataFrame({"实际": y, "预测": oof}).groupby(by_actual, observed=True).mean().round(2).T)
```

实际降得最多的一组平均 −6.11 Gy,模型只说 −3.90 Gy;实际升得最多的一组平均 +1.88 Gy,模型说 −0.15 Gy。预测值的波动(标准差 1.96 Gy)也比实际值(2.83 Gy)小。模型好像两头都不敢说。

这不是毛病,是任何预测都有的性质。一位患者的 Δ 由两部分组成: 模型能预测的部分(由壁厚、直径、次数决定),和模型预测不了的部分(没有量到的因素、偶然)。实际值最极端的患者,往往是两部分恰好偏向同一边的人。模型只知道第一部分,所以给不出那么极端的数。

**打个比方**: 门诊第一次量血压特别高的人,复查时多半没那么高。不是降压药起效了(也许还没吃),而是第一次那个特别高的读数里,有一部分是当天的偶然。这叫**回归均值**(regression to the mean)。

所以,"实际值很高的患者被低估了"不能当作模型失败的证据: 把残差对实际值画图,永远会看到一条斜线。该检查的是反过来的方向: 按**预测值**分组。

```python
by_predicted = pd.qcut(oof, 5, labels=labels)
print(pd.DataFrame({"预测": oof, "实际": y}).groupby(by_predicted, observed=True).mean().round(2).T)
```

模型说平均 −4.64 Gy 的那一组,实际平均 −4.67 Gy;说 +0.84 Gy 的那一组,实际 +0.64 Gy。五组都对得上。这才是使用模型时要紧的事: **模型说出一个数,能不能照字面相信它**。这个性质叫**校准**(calibration)。

<div class="frame">

**背景**: 五组的表要用眼睛比。想要一个数,说明预测值是说得太极端、太保守,还是正好。

**目标**: 量出预测值的"幅度"对不对。

**做法**: 用留出时的预测值当 x、实际值当 y,拟合一条直线。它的斜率叫**校准斜率**(calibration slope)。

</div>

**它在算什么**

1. 每位患者有一个留出时的预测值,和一个实际值。
2. 用预测值预测实际值,做一次最简单的线性回归(第 1 课)。
3. 看斜率:
   - 等于 1: 预测值高 1 Gy,实际平均也高 1 Gy。可以照字面读。
   - 小于 1: 预测值说得太极端。模型说差 4 Gy,实际只差 2 Gy 多。过拟合的模型都是这样。
   - 大于 1: 预测值说得太保守。

```python
def calibration_slope(actual, predicted):
    return LinearRegression().fit(np.asarray(predicted).reshape(-1, 1), actual).coef_[0]

print(f"线性回归: {calibration_slope(y, oof):.2f}")
print(f"不限深度的树: {calibration_slope(y, oof_t):.2f}")
```

线性回归是 0.97,几乎正好。不限深度的树是 0.57: 它记住了建模患者的偶然,把偶然也当成规律说了出来,预测值的差别有将近一半是虚的。上一节里树的 r²(0.30)比 R²(0.14)好看,差的就是这一块。

画出来:

```python
import matplotlib.pyplot as plt

lims = [y.min() - 1, y.max() + 1]
fig, axes = plt.subplots(2, 1, figsize=(4.2, 7), sharex=True, sharey=True)
for ax, pred, title in [(axes[0], oof, "Linear regression"), (axes[1], oof_t, "Unpruned tree")]:
    ax.scatter(pred, y, s=12, alpha=0.6)
    ax.plot(lims, lims, color="grey", lw=1)                               # 预测 = 实际
    slope = calibration_slope(y, pred)
    xs = np.array([pred.min(), pred.max()])
    ax.plot(xs, y.mean() + slope * (xs - pred.mean()), color="C1", lw=1.5)   # 实际值对预测值的回归线
    ax.set_title(f"{title}: slope {slope:.2f}, R² {r2_score(y, pred):.2f}", fontsize=10)
    ax.set_ylabel("Observed Δ (Gy)")
axes[1].set_xlabel("Out-of-fold predicted Δ (Gy)")
plt.tight_layout()
plt.show()
```

灰线是"预测 = 实际",橙线是实际值对预测值的回归线。上面一张两条线几乎重合;下面一张橙线明显更平: 预测值铺得很开,实际值没有跟上。

第 11 课讲是/否结局的校准,用的是同一个想法。

<div class="exercise">

**练习 3 · 轮到你: 图形优化的校准斜率**

对图形优化的 Δ,算线性回归的校准斜率(`slope_g`)和不限深度的树的校准斜率(`slope_tree_g`)。`oof_g` 和 `oof_tg` 是前两个练习里算好的。

```python
slope_g = calibration_slope(yg, ___)
slope_tree_g = calibration_slope(___, oof_tg)
print(f"线性回归 {slope_g:.2f},不限深度的树 {slope_tree_g:.2f}")
```

```pycheck
need("slope_g", "slope_tree_g")
if close(slope_g, 0.970, 0.004):
    fail("这是体积优化的结果。第一个空是图形优化的留出预测 oof_g。")
if close(slope_g, 0.945, 0.004) and close(slope_tree_g, 0.364, 0.004):
    ok("对了: 线性回归 0.95,树 0.36。树说两位患者相差 10 Gy 的时候,实际平均只差 3.6 Gy。它在图形优化上的 R² 是负的(练习 2),原因就在这里: 说得太极端,错得比都猜平均值还多。")
fail(f"现在是 {slope_g:.3f} 和 {slope_tree_g:.3f},应该是 0.945 和 0.364。第一个空是 oof_g,第二个空是 yg。")
```

<details><summary>提示</summary>

`calibration_slope(实际值, 预测值)`: 第一个空是线性回归的留出预测 `oof_g`,第二个空是实际值 `yg`。

</details>

<details><summary>参考答案</summary>

```python
slope_g = calibration_slope(yg, oof_g)
slope_tree_g = calibration_slope(yg, oof_tg)
print(f"线性回归 {slope_g:.2f},不限深度的树 {slope_tree_g:.2f}")
```

</details>

</div>

## 4. 一个队列,好几个结局

换一种计划方案,变的不只是阴道的 D2cc,靶区的 D90 也在变;而且有两种优化方案。这个队列一共有四个可以预测的 Δ。同样三个变量,各建一个模型:

```python
outcomes = {"阴道 D2cc,体积优化": "delta", "阴道 D2cc,图形优化": "delta_gra",
            "靶区 D90,体积优化": "delta_d90", "靶区 D90,图形优化": "delta_d90_gra"}
rows = []
for name, column in outcomes.items():
    pred = out_of_fold(LinearRegression(), cols, column)
    rows.append({"结局": name, "标准差": df[column].std(), "MAE": mean_absolute_error(df[column], pred), "R2": r2_score(df[column], pred)})
print(pd.DataFrame(rows).round(2).to_string(index=False))
```

只看 MAE,靶区 D90(体积优化)的模型是 1.06 Gy,比阴道 D2cc(体积优化)的 1.70 Gy 好得多。但 D90 的 Δ 本来就变化得小(标准差 1.82 Gy 对 2.83 Gy),猜平均值也错不了多少。看 R²: 0.52 对 0.45,只好一点。

两把尺子各有各的用处:

- **跨结局比较**(哪个结局更好预测),用 R²: 它已经扣掉了各个结局自己的波动。
- **对临床说话**(预测值能差多少),用 MAE: 它带单位。"平均差 1.7 Gy"医生能判断够不够用;"R² 0.45"不能。

**每个结局各建一个模型**。四个结局要不要放进一个模型里一起预测? 对线性回归,一起拟合和分开拟合是同一回事:

```python
together = LinearRegression().fit(df[cols], df[["delta", "delta_gra"]])     # 一次给两个结局
alone = LinearRegression().fit(df[cols], df["delta_gra"])
print("一起拟合,图形优化的系数:", together.coef_[1].round(2))
print("单独拟合,图形优化的系数:", alone.coef_.round(2))
```

系数一模一样。所以做法很简单: 每个结局一个模型,各有各的基线,各报各的成绩。

**但四个成绩不是四份独立的证据**。看四个模型的残差(实际值减留出时的预测值)之间的相关:

```python
residuals = pd.DataFrame({column: df[column] - out_of_fold(LinearRegression(), cols, column) for column in outcomes.values()})
print(residuals.corr().round(2))
```

残差两两之间的相关系数在 0.5 上下。一位患者的 D2cc 被低估,她的 D90 多半也被低估: 四个结局背后有共同的、模型没有看到的东西(真实数据里可能是解剖结构或施源器的位置)。所以"四个结局都预测得不错"不等于验证了四次;它们来自同一批患者、同一次计划,好运气和坏运气是一起来的。

最后一件事。既然四个结局相关,能不能用一个去预测另一个?

```python
with_vol = out_of_fold(LinearRegression(), cols + ["delta"], "delta_gra")
print(f"把体积优化的 Δ 也当作变量: 图形优化的 R² {r2_score(df['delta_gra'], with_vol):.2f}")
```

<div class="quiz">
<p><b>想一想</b>: 加上体积优化的 Δ 以后,图形优化的 R² 从 0.31 升到了 0.48。这个模型能用吗?</p>
<button class="quiz-opt" data-why="成绩确实是在留出的患者身上算的,没有泄漏患者。问题在另一个地方: 用模型的那一刻,这个变量手里有没有。">能: 成绩是交叉验证算的,没有问题</button>
<button class="quiz-opt" data-correct data-why="第 5 课的规矩: 模型只能用预测那一刻已经知道的东西。如果体积优化的计划总是先做,它的 Δ 到时候是现成的,可以用;如果模型的目的正是在做任何优化之前就判断值不值得做,到时候还没有这个数,用了它就是泄漏。写论文时要写明预测是在哪一刻做的。">要看用模型的那一刻,体积优化的计划做出来了没有</button>
<button class="quiz-opt" data-why="相关的变量可以放进同一个模型(第 6、8 课)。这里的问题不是相关,而是预测那一刻这个变量有没有。">不能: 两个结局相关,不能互相预测</button>
</div>

<div class="exercise">

**练习 4 · 轮到你: 图形优化,在新患者身上对答案**

用全部 196 位患者拟合图形优化的线性回归,在 4900 位新患者身上算 R²(`r2_new_g`)和校准斜率(`slope_new_g`)。

```python
final_g = LinearRegression().fit(df[cols], df["delta_gra"])
pred_new = final_g.predict(new[___])
r2_new_g = r2_score(new["delta_gra"], pred_new)
slope_new_g = calibration_slope(new[___], pred_new)
print(f"新患者: R² {r2_new_g:.2f},校准斜率 {slope_new_g:.2f}")
```

```pycheck
need("r2_new_g", "slope_new_g")
if close(r2_new_g, 0.425, 0.004):
    fail('这是体积优化的结果。模型和结局都要用 "delta_gra"。')
if close(r2_new_g, 0.328, 0.004) and close(slope_new_g, 1.023, 0.004):
    ok("对了: 新患者身上 R² 0.33,校准斜率 1.02。交叉验证给的是 0.31 和 0.95: 都和新患者身上的数差不多,这正是交叉验证的用处。")
fail(f'现在是 R² {r2_new_g:.3f}、斜率 {slope_new_g:.3f},应该是 0.328 和 1.023。第一个空是 cols,第二个空是 "delta_gra"。')
```

<details><summary>提示</summary>

第一个空是三个变量的名单 `cols`;第二个空是新患者的实际结局 `"delta_gra"`。

</details>

<details><summary>参考答案</summary>

```python
final_g = LinearRegression().fit(df[cols], df["delta_gra"])
pred_new = final_g.predict(new[cols])
r2_new_g = r2_score(new["delta_gra"], pred_new)
slope_new_g = calibration_slope(new["delta_gra"], pred_new)
print(f"新患者: R² {r2_new_g:.2f},校准斜率 {slope_new_g:.2f}")
```

</details>

</div>

## 5. 报告一个连续结局的模型

把这一课合起来,一个预测连续结局的模型,至少报告这几样,而且都在留出的患者身上算:

| 报告什么 | 回答的问题 | 这一课的例子(体积优化) |
|---|---|---|
| MAE(可加 RMSE) | 预测值平均差多少 | 1.70 Gy(RMSE 2.09 Gy) |
| 基线的 MAE | 不用模型差多少 | 2.25 Gy |
| R² | 比不用模型好了几成 | 0.45 |
| 校准斜率 | 预测值能不能照字面读 | 0.97 |
| 预测值对实际值的散点图 | 有没有哪一段特别不准 | 第 3 节的图 |

## 小结

- **MAE** 和 **RMSE** 带单位,回答"差多少";RMSE 更看重大的误差。
- **R²** 是和"都猜平均值"比: 误差平方和少了几成。它不带单位,可以跨结局比较。
- 在留出的患者身上,**R² 可以是负的**: 模型不如不用。建模患者身上的 R² 不说明预测能力。相关系数的平方不是 R²。
- 实际值极端的患者总是被"低估": 这是**回归均值**,不是模型的毛病。检查模型要按**预测值**分组,或者看**校准斜率**。
- 几个结局各建一个模型、各报各的成绩;跨结局用 R² 比,对临床用 MAE 说。相关的结局不是独立的证据。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 平均绝对误差 | mean absolute error (MAE) | 误差取绝对值的平均 |
| 均方根误差 | root mean squared error (RMSE) | 误差平方的平均再开方;看重大的误差 |
| 决定系数 | R² (coefficient of determination) | 1 − 模型的误差平方和 ÷ 都猜平均值的误差平方和 |
| 留出时的预测 | out-of-fold prediction | 这位患者没参与建模时得到的预测值 |
| 回归均值 | regression to the mean | 极端的值里有偶然的成分,再看一次往往没那么极端 |
| 校准 | calibration | 预测值能不能照字面相信 |
| 校准斜率 | calibration slope | 实际值对预测值回归的斜率;1 最好,小于 1 是说得太极端 |

## 这一课没讲什么

- **预测区间**: 模型只给了一个数。"这位患者的 Δ 有 90% 的可能在 −5 到 +1 Gy 之间"这样的区间,需要另外的方法(分位数回归、conformal prediction),这门课不讲。
- **调整后的 R²**(adjusted R²): 在建模患者身上按变量个数打折。它是对"变量越多 R² 越高"的修补,不能代替留出患者身上的成绩。
- **每一折各算一个 R² 再平均**: `cross_val_score(..., scoring="r2")` 是这样做的,和这一课"把所有留出预测放在一起算"略有出入。患者少时,每一折里的 R² 很不稳定,放在一起算更稳。
- **Bland–Altman 图**: 比较两种测量方法的一致性时常用,和这一课的散点图回答的问题相近。

## 写进论文 Writing it up

<div class="paper">

方法和结果可以这样写(数字换成你自己的):

> Performance was assessed on out-of-fold predictions from 5-fold cross-validation. We report the mean absolute error (MAE) with that of a mean-only baseline, R² relative to that baseline, and the calibration slope (the slope of observed on predicted values). The model predicted the change in vaginal D2cc with an MAE of 1.70 Gy (baseline 2.25 Gy), an R² of 0.45 and a calibration slope of 0.97.

**审稿人可能问**: R² 是怎么算的? 是相关系数的平方吗?

**怎么回答**: 写出定义: 1 − 模型的误差平方和 ÷ 只用平均值的误差平方和,在留出的预测上算。不是相关系数的平方。

**审稿人可能问**: 残差图显示实际值高的患者被系统地低估了。

**怎么回答**: 残差对**实际值**作图一定有这种趋势(回归均值)。补一张残差对**预测值**的图,并报告校准斜率。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| R² 是负数 | 不是错。模型在这些患者身上不如都猜平均值。检查是不是过拟合,或者变量本来就没有预测能力。 |
| `r2_score(预测值, 实际值)` 的结果很奇怪 | 参数的顺序是 `r2_score(实际值, 预测值)`。写反了,数会变。 |
| `Expected 2D array, got 1D array` | 只有一个变量时,要写成一列: `.reshape(-1, 1)` 或 `df[["x"]]`。 |
| `pd.qcut` 报 `Bin edges must be unique` | 很多患者的值相同,分不成五等份。加 `duplicates="drop"`,或少分几组。 |
| 结果和网页上写的不一样 | 检查 `random_state=0` 和 `default_rng(9)`;代码块要从上往下依次运行。 |

## 延伸阅读 Further reading

- scikit-learn 用户指南: [Metrics and scoring: regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)。`r2_score`、`mean_absolute_error` 等的定义。
- Bland JM, Altman DG. Regression towards the mean. *BMJ* 1994;308(6942):1499。一页纸讲清楚回归均值。

**下一课**: 结局换成是/否(会不会出现晚期毒性、会不会降到 68 Gy 以下),模型给的是概率,尺子也要换。第 10 课讲第一把: 模型能不能把会出事的患者排在前面,也就是区分度、ROC 曲线和 AUC。
