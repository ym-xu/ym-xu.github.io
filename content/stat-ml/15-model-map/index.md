---
title: "Lesson 15 · 模型全景: 六种模型怎么算 A Map of the Models"
summary: "同一位患者交给六种模型: 各自怎么算、各自假定什么、先试哪一个;还有这门课没讲的模型。"
weight: 150
# 用户 2026-11-21: 这一课先不用(要的是算法本身的动态讲解)。draft 的页面不会被构建,源文件留着。
draft: true
toc: true
duration: "约 40 分钟"
prereq: "本课程[第 1 课](../01-linear-regression/)、[第 2 课](../02-logistic-regression/)、[第 6 课](../06-regularization/)、[第 7 课](../07-trees/)"
questions:
  - "前面学过的六种模型,算预测的办法有什么不同?"
  - "同一位患者,不同的模型为什么给出不同的数?"
  - "面对一份新数据,先试哪个模型?"
  - "这门课没讲的模型(神经网络、支持向量机……)是怎么回事?"
objectives:
  - "用一句话说出每种模型怎样从患者的数算出预测值"
  - "给同一位患者算出各个模型的预测,并说出差别从哪里来"
  - "按结局的类型、患者的数量和模型的用途,选一个先试的模型"
  - "读到没学过的模型时,知道该问哪几个问题"
---

前面的课里,模型是一个一个出场的: 第 1 课线性回归,第 2 课逻辑回归,第 6 课岭回归和 LASSO,第 7 课树、随机森林和梯度提升。每一个都单独讲过它在算什么。

这一课不讲新方法,只做一件事: 把它们并排放在一起。同一位患者,依次交给每一种模型,看它们各自怎么得出一个数,这些数为什么不一样。

**这一课的图和别的课不同**: 它是一张分步的图。点"下一步",图往前走一步,图上方的三个格子跟着填数;点 ↻ 把这一步重放一遍。

## 0. 准备

```python
import numpy as np
import pandas as pd
from cohort import make_cohort
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def prepare(d):
    d = d.dropna().copy()
    d["delta"] = d["d2cc_vol"] - d["d2cc_5mm"]
    d["delta_gra"] = d["d2cc_gra"] - d["d2cc_5mm"]
    return d

df, new = prepare(make_cohort()), prepare(make_cohort(n=5000, seed=99))
cols = ["wall_mean", "diameter", "n_fx"]
one = pd.DataFrame({"wall_mean": [3.0], "diameter": [30], "n_fx": [3]})     # 图里的那位患者

def six_models():
    """六种模型,设置和第 6、7 课相同。每次调用都交回一套新的、还没拟合的模型。"""
    return {
        "都猜平均值": DummyRegressor(),
        "线性回归": LinearRegression(),
        "LASSO": make_pipeline(StandardScaler(), Lasso(alpha=0.2)),
        "一棵树": DecisionTreeRegressor(max_depth=2, random_state=0),
        "随机森林": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0),
        "梯度提升": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=0),
    }

print(len(df), "位建模患者,", len(new), "位新患者")
```

和第 7 课一样: 196 位建模患者,三个变量(平均壁厚、施源器直径、近距离次数),要预测的是换成 Volume 计划以后阴道 D2cc 的变化 Δ。另外 4900 位新患者只用来检验。

## 1. 同一位患者,六种模型

<div class="widget stage" data-widget="models">

<div class="stage-step" data-label="一位患者">
<p class="stage-h">一位新患者,交给六种模型</p>

平均壁厚 3.0 mm,施源器直径 30 mm,近距离做 3 次。换成 Volume 的计划以后,她的阴道 D2cc 会变多少?

模型是一条规则: 把这三个数变成一个预测值。下面每一步换一种规则,患者不变。图上方的三格依次记下: 用的是哪种模型,它给这位患者的预测,它在 4900 位新患者身上平均差多少。

</div>

<div class="stage-step" data-label="基线">
<p class="stage-h">基线: 谁来都猜平均值</p>

最省事的规则不看患者: 196 位建模患者的 Δ 平均是 −2.09 Gy,对谁都猜这个数。

它在新患者身上平均差 2.28 Gy。后面每个模型都要先赢过它,否则那三个变量就白用了。

</div>

<div class="stage-step" data-label="线性回归">
<p class="stage-h">线性回归: 每个变量各加减一笔</p>

从平均的患者出发。她的壁比平均薄 0.69 mm,每毫米值 1.57 Gy,减 1.09;直径和平均几乎一样;近距离次数比平均多 0.55 次,再减 0.66。加起来是 −3.84 Gy。

这就是第 1 课的公式 −9.94 + 1.57 × 壁厚 + 0.17 × 直径 − 1.21 × 次数,换成"和平均的患者比"来写,结果相同。

**它假定**: 每个变量的作用处处相同,几笔之间互不影响。

</div>

<div class="stage-step" data-label="正则化">
<p class="stage-h">正则化: 同一条公式,系数往 0 收</p>

LASSO 和岭回归用的还是那条公式,只是找系数的时候多一条约束: 系数越大,罚得越多(第 6 课)。三个系数都小了一点,预测从 −3.84 退到 −3.50,往平均值靠。约束再强,作用弱的变量先归零。

**什么时候用**: 候选变量多、患者少的时候。这里只有三个真变量,它没帮上忙: 1.74 对 1.73 Gy。

</div>

<div class="stage-step" data-label="一棵树">
<p class="stage-h">一棵树: 回答两个问题</p>

树不做加法。它问: 壁厚不超过 3.35 mm 吗?是。近距离做 2 次吗?不是,做 3 次。她落进的那片叶子里有 41 位建模患者,他们的平均值 −5.13 Gy 就是她的预测。

**它假定**: 落进同一片叶子的患者都一样。壁厚 2.0 mm 和 3.3 mm,只要都做 3 次,得到同一个数。好处是它自己发现了"做 3 次的患者规律不同",不用你事先写进公式。

</div>

<div class="stage-step" data-label="随机森林">
<p class="stage-h">随机森林: 300 棵树,取平均</p>

每棵树用重新抽的一批患者长成,所以各不相同: 对这位患者,300 棵树的答案从 −7.23 到 −2.75 都有。一棵一棵平均下去,数字很快稳住,最后是 −4.86 Gy。

**为什么要平均**: 单独一棵树跟着它那批患者的偶然走。很多棵树的偶然方向不同,平均以后互相抵消。

</div>

<div class="stage-step" data-label="梯度提升">
<p class="stage-h">梯度提升: 100 棵小树接力</p>

从平均值 −2.09 出发。第 1 棵小树去猜她"还差多少",猜的是 −3.04;只把其中 5% 加进去,预测变成 −2.24。第 2 棵再去猜剩下的差距。开头步子大,越往后越小,100 棵之后停在 −4.98 Gy。

**和森林的区别**: 森林的树互不相干,最后取平均。这里的树一棵接一棵,每一棵都在补前面的不足。

</div>

<div class="stage-step" data-label="是或否">
<p class="stage-h">结局是"是/否": 加起来,再过一条 S 形曲线</p>

换一个问题: 她的 5 mm 计划超出 68 Gy 2.0 Gy,换计划以后能不能降到 68 Gy 以下?逻辑回归照样乘系数、加起来: 7.24 − 0.83 × 2.0 − 1.59 × 3.0 = 0.81。这个数可大可小,S 形曲线把它压到 0 和 1 之间,得到概率 0.69(第 2 课)。

树、森林、梯度提升也都有预测是/否的版本: 叶子里放的不是平均值,而是"这片叶子里有几成发生了"。

</div>

<div class="stage-step" data-label="六个答案">
<p class="stage-h">同一位患者,六个答案</p>

六个预测从 −5.13 到 −2.09,最多差 3 Gy。

患者是模拟的,所以这里有正确答案: 按生成数据的公式,这样的患者平均是 −3.84 Gy(红色虚线)。直线几乎正中。三种树模型都低了 1 Gy 多,有两个原因: 叶子把她和壁更薄的患者算成一组;建模的 196 位里,这一组的 41 位又恰好比公式多降了 0.8 Gy,树跟着他们走了。

一位患者说明不了哪个模型好。看下一步。

</div>

<div class="stage-step" data-label="新患者">
<p class="stage-h">在 4900 位新患者身上</p>

六个模型都拿去预测 4900 位没见过的患者。误差最小的是线性回归,1.73 Gy;LASSO 几乎一样;梯度提升 1.78,随机森林 1.83,一棵树 1.92。都比猜平均值的 2.28 强。

更灵活的模型没有赢: 这个队列只有 196 位患者,关系又接近直线。换一份数据,名次可能不同。**所以每次都要比,而且先和简单的模型比**。

</div>

</div>

图里的每个数都来自下一节的代码,你可以自己跑出来。

## 2. 自己算一遍

`compare` 把六种模型各拟合一次,交回两列: 给这位患者的预测,在新患者身上的平均绝对误差。

```python
def compare(outcome="delta", patient=one):
    """每种模型: 给这位患者的预测,和它在 4900 位新患者身上的平均绝对误差(Gy)。"""
    rows = {}
    for name, model in six_models().items():
        model.fit(df[cols], df[outcome])
        error = (new[outcome] - model.predict(new[cols])).abs().mean()
        rows[name] = {"这位患者": model.predict(patient)[0], "新患者身上的误差": error}
    return pd.DataFrame(rows).T

table = compare()
print(table.to_string(float_format="%.2f"))
```

六行和图里的六步一一对应。下面把图里几步的细节也算出来。

**线性回归的三笔**(第 3 步):

```python
line = LinearRegression().fit(df[cols], df["delta"])
diff = one.iloc[0] - df[cols].mean()                  # 这位患者比平均的患者多多少
terms = line.coef_ * diff                             # 每个变量加减的那一笔
print(pd.DataFrame({"和平均差": diff, "系数": line.coef_, "加减 (Gy)": terms}).round(2))
print(f"平均值 {df['delta'].mean():.2f},加上三笔 {terms.sum():.2f},得 {df['delta'].mean() + terms.sum():.2f};predict 给的是 {line.predict(one)[0]:.2f}")
```

三笔各自四舍五入到两位是 −1.09、−0.01 和 −0.66,所以你手算加起来会差 0.01。

**森林里的每一棵树**(第 6 步)。`forest.estimators_` 里放着 300 棵树,可以一棵一棵问:

```python
forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0).fit(df[cols], df["delta"])
each = np.array([one_tree.predict(one.to_numpy())[0] for one_tree in forest.estimators_])
print(f"300 棵树的答案: 最低 {each.min():.2f},最高 {each.max():.2f},平均 {each.mean():.2f} Gy")
print("前 1、5、20、100、300 棵的平均:", [round(float(each[:k].mean()), 2) for k in (1, 5, 20, 100, 300)])
```

第 1 棵树说 −5.89,前 5 棵平均 −4.78,到 100 棵时已经是 −4.84,和最后的 −4.86 只差 0.02。

**梯度提升的每一步**(第 7 步)。`staged_predict` 交回加到第 1 棵、第 2 棵……第 100 棵时的预测:

```python
boost = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=0).fit(df[cols], df["delta"])
path = [float(p[0]) for p in boost.staged_predict(one)]
print("第 1、2、10、50、100 棵之后:", [round(path[k - 1], 2) for k in (1, 2, 10, 50, 100)])
print(f"第 1 棵小树猜的差距: {(path[0] - df['delta'].mean()) / 0.05:.2f} Gy,只加进去 5%")
```

**逻辑回归**(第 8 步)。和第 2 课一样,只看 5 mm 计划达到 68 Gy 的 144 位患者:

```python
hi = df[df["d2cc_5mm"] >= 68].copy()
hi["excess"] = hi["d2cc_5mm"] - 68
hi["event"] = (hi["d2cc_vol"] < 68).astype(int)
logit = LogisticRegression(C=1e6).fit(hi[["excess", "wall_mean"]], hi["event"])
b0, (b1, b2) = logit.intercept_[0], logit.coef_[0]
total = b0 + b1 * 2.0 + b2 * 3.0
print(f"{b0:.2f} {b1:+.2f} × 2.0 {b2:+.2f} × 3.0 = {total:.2f};概率 {1 / (1 + np.exp(-total)):.2f}")
```

**正确答案**(第 9 步)。队列是用 `cohort.py` 里的一条公式加上随机数生成的。把随机的部分去掉,就是"这样的患者平均会怎样":

```python
def by_formula(d):
    """生成 Δ 的公式,去掉随机的部分(见 cohort.py)。"""
    return d["n_fx"] / 2.5 * (1.6 * (d["wall_mean"] - 5) + 0.15 * (d["diameter"] - 30))

leaf = df[(df["wall_mean"] <= 3.35) & (df["n_fx"] == 3)]        # 她在那棵树里落进的叶子
print(f"这位患者按公式: {by_formula(one)[0]:.2f} Gy")
print(f"叶子里 {len(leaf)} 位患者: 实际平均 {leaf['delta'].mean():.2f},按公式平均 {by_formula(leaf).mean():.2f} Gy")
```

叶子里的 41 位,按公式平均应该是 −4.33 Gy,实际是 −5.13: 这一批患者恰好多降了 0.8 Gy。树和森林没有办法知道这是偶然。真实的数据没有公式可查,所以只能靠新患者(或交叉验证)来判断谁更准。

<div class="exercise">

**练习 1 · 轮到你: Graphical**

把结局换成 Graphical 的 Δ(`delta_gra`),患者还是同一位。算出六种模型的预测,再把六个预测里最高和最低的差存进 `gap_g`。

```python
table_g = compare(___)
print(table_g.to_string(float_format="%.2f"))
gap_g = table_g[___].max() - table_g[___].min()
print(f"六个预测最多相差 {gap_g:.2f} Gy")
```

```pycheck
need("table_g", "gap_g")
if close(table_g.loc["线性回归", "这位患者"], -3.84, 0.01):
    fail('这是 Volume 的结果。第一个空填 "delta_gra"。')
if close(gap_g, 0.54, 0.01):
    fail('你算的是"新患者身上的误差"那一列的差。两个空都填 "这位患者"。')
if close(gap_g, 4.19, 0.01):
    ok("对了: 六个预测从 −7.69(随机森林)到 −3.50(都猜平均值),相差 4.19 Gy。线性回归给 −5.71,三种树模型都在 −7.3 到 −7.7 之间。按生成数据的公式,她平均是 −6.00 Gy: 这一次直线高了,树低了,谁也没有正中。")
fail(f"gap_g 现在是 {gap_g:.2f},应该是 4.19。")
```

<details><summary>提示</summary>

第一个空是 `"delta_gra"`。后两个空是同一个列名: `"这位患者"`。

</details>

<details><summary>参考答案</summary>

```python
table_g = compare("delta_gra")
print(table_g.to_string(float_format="%.2f"))
gap_g = table_g["这位患者"].max() - table_g["这位患者"].min()
print(f"六个预测最多相差 {gap_g:.2f} Gy")
```

</details>

</div>

<div class="exercise">

**练习 2 · 换一位患者**

另一位患者: 平均壁厚 5.5 mm,直径 30 mm,近距离做 2 次。结局还是 Volume 的 Δ。把六种模型给她的预测存进 `pred_other`(`compare` 交回的表里"这位患者"那一列)。

```python
other = pd.DataFrame({"wall_mean": [___], "diameter": [30], "n_fx": [___]})
pred_other = compare("delta", patient=___)["这位患者"]
print(pred_other.round(2))
```

```pycheck
need("other", "pred_other")
if close(pred_other["线性回归"], -3.84, 0.01):
    fail("这还是原来那位患者。最后一个空填 other。")
if not close(other["wall_mean"].iloc[0], 5.5, 0.001) or not close(other["n_fx"].iloc[0], 2, 0.001):
    fail("壁厚填 5.5,次数填 2。")
if close(pred_other["线性回归"], 1.29, 0.01) and close(pred_other["一棵树"], 1.40, 0.01):
    ok("对了。五个模型给她的预测在 0.81 到 1.40 Gy 之间,只差 0.6 Gy;上一位患者差了 1.6 Gy。模型之间分歧有多大,因患者而异: 上一位患者壁薄又做 3 次,正好落在直线和树看法不同的角落里。")
fail("检查三个空: 5.5、2 和 other。")
```

<details><summary>提示</summary>

`other` 里壁厚填 `5.5`,次数填 `2`;`compare` 的第二个参数填 `other`。

</details>

<details><summary>参考答案</summary>

```python
other = pd.DataFrame({"wall_mean": [5.5], "diameter": [30], "n_fx": [2]})
pred_other = compare("delta", patient=other)["这位患者"]
print(pred_other.round(2))
```

</details>

</div>

## 3. 一张表: 各自假定什么

| 模型 | 怎么算出预测 | 假定什么,要你定什么 |
|---|---|---|
| **线性回归**<br>第 1 课 | 每个变量乘一个系数,加起来 | 每个变量的作用处处相同,互不影响。要定: 放哪些变量 |
| **逻辑回归**<br>第 2 课 | 同上,再过 S 形曲线,得到概率 | 同上,只是加的是 log-odds。要定: 放哪些变量 |
| **岭回归、LASSO**<br>第 6 课 | 同一条公式,系数往 0 收 | 同上。要定: 约束的强度 |
| **决策树**<br>第 7 课 | 一串是/否的问题,走到叶子 | 同一片叶子里的患者都一样。要定: 分几层,叶子最少几人 |
| **随机森林**<br>第 7 课 | 很多棵树各自预测,取平均 | 几乎不假定形状。要定: 多少棵,叶子最少几人 |
| **梯度提升**<br>第 7 课 | 很多棵小树接力,每棵补一点 | 几乎不假定形状。要定: 多少棵,每步加多少,每棵几层 |

从上往下,模型假定得越来越少。假定少听起来是好事,代价有三个:

- **要学的东西多,需要的患者也多**。线性回归只要估计 4 个数(三个系数和一个截距)。一棵两层的树要定 3 个分叉点和 4 个叶子的值;300 棵没有层数限制的树,要定的东西成千上万。患者不够时,多出来的灵活都拿去记偶然了(第 4 课、第 13 课)。
- **"要你定的"越多,越容易泄漏**。这些设置如果看着成绩来回调,成绩就偏乐观。要在交叉验证里面选(第 5、6 课)。
- **越难交给别人用**。回归模型是一条公式,几个数就能写进论文(第 14 课);森林只能把整个模型文件交出去。

线性回归的假定也可以自己放宽: 加一个交互项,它就能表示"做 3 次的患者斜率不同"(第 7 课第 6 节)。前提是你事先想到了。

## 4. 先试哪一个

面对一份新数据,按这三个问题来。

**结局是什么?** 一个连续的量(剂量变了多少),用线性回归那一族。有没有发生(出没出现副作用),用逻辑回归那一族。随访时间不等、有人失访,是 Cox 回归那一族,这门课没有讲。

**数据撑得起多少?** 先按第 13 课算一算,现有的患者和事件数撑得起几个参数。只撑得起两三个参数的数据,用不了几百棵树。

**模型给谁用?** 要写进论文、让别的医院照着算,公式最方便。只在自己科室的系统里运行,哪种都行。

多数临床数据(几百位患者,十几个候选变量)走下来是同一条路:

1. 先拟合一个事先定好变量的回归模型,作为基准。
2. 再试一个灵活的模型,通常是随机森林或梯度提升。
3. 两者用同一种交叉验证比(第 4、5 课),比区分度,也比校准(第 10、11 课)。
4. 灵活的模型没有明显更好,就留下简单的。

<div class="quiz">
<p><b>想一想</b>: 一篇论文有 150 位患者、30 个事件、12 个候选变量。作者用了梯度提升,报告 AUC 0.95。你最先想问的是什么?</p>
<button class="quiz-opt" data-why="神经网络更灵活,需要的数据更多。30 个事件时它不会更可靠。">为什么不用神经网络</button>
<button class="quiz-opt" data-correct data-why="30 个事件撑不起 12 个变量的灵活模型(第 13 课)。AUC 0.95 很可能是在建模患者身上算的,或者设置是看着成绩调出来的。先问: 这个数是怎么验证的?同样的验证下,一个两三个变量的逻辑回归是多少?">这个 AUC 是怎么验证的,和逻辑回归比过吗</button>
<button class="quiz-opt" data-why="学习率是该报告的设置之一,但它回答不了这个 AUC 可不可信。先问验证的办法,和有没有同简单的模型比。">学习率设的是多少</button>
</div>

## 5. 这门课没讲的模型

读论文会遇到别的名字。它们多数可以归到上面那张表的某一行旁边。

- **样条(spline)和广义可加模型(generalized additive model, GAM)**: 还是回归,还是"每个变量各加一笔",但允许这一笔随变量的取值沿一条弯的曲线变化。灵活的程度在直线和树之间。
- **XGBoost、LightGBM、CatBoost**: 都是梯度提升,做法和第 7 课相同,算得更快,带有更多防止过拟合的设置。论文里看到它们,按梯度提升来读。
- **k 近邻(k-nearest neighbours)**: 在建模患者里找和这位患者最像的 k 位,取他们结局的平均。没有公式;"像不像"取决于各个变量用什么单位,所以要先标准化。
- **支持向量机(support vector machine, SVM)**: 在两类患者之间找一条分界线(可以是弯的),让它离两边最近的患者尽量远。它直接给的是"在分界线的哪一边",不是概率;要概率得另外校准(第 11 课)。
- **神经网络(neural network)**: 把"乘系数、加起来、过一个弯曲的函数"这件事叠很多层。没有中间层的神经网络就是逻辑回归。层数多了能学很复杂的形状,在影像和文字上很强;只有几百位患者、十几个表格变量时,很少比回归和树模型好。
- **生存模型**: 随访时间不等、有删失的结局,用 Cox 回归。随机森林和梯度提升也有对应的版本(如 random survival forest)。评价的思路和第 10–12 课相同,算法不同。

遇到任何没学过的模型,问这四个问题就够了:

1. 它怎样从患者的数算出预测?一句话说不清的,先存疑。
2. 它有哪些要定的设置?是事先定的,还是在交叉验证里选的?
3. 它和一个简单的回归模型比过吗?用的是同一种验证吗?
4. 它给的概率校准过吗?

<div class="quiz">
<p><b>想一想</b>: 同一位患者,线性回归预测 −3.84 Gy,随机森林预测 −4.86 Gy。能不能说"随机森林考虑了更复杂的关系,所以它的数更可信"?</p>
<button class="quiz-opt" data-why="更灵活只说明它能表示更复杂的关系,不说明它在这份数据上学对了。这位患者按生成数据的公式是 −3.84 Gy,森林偏了 1 Gy。">能。森林不假定直线,更接近真实</button>
<button class="quiz-opt" data-correct data-why="在这个队列里,森林对她的预测偏了 1 Gy: 它跟着 41 位恰好降得多的患者走了。在 4900 位新患者身上,森林平均差 1.83 Gy,线性回归 1.73 Gy。哪个模型更可信,要看它在没见过的患者身上的成绩,不看它有多复杂。">不能。可不可信要看在新患者身上的成绩,不看模型有多复杂</button>
<button class="quiz-opt" data-why="两个数差 1 Gy,不能都对。只有一位患者时确实分不出谁对,所以要看很多位新患者身上的成绩。">都可信。两个模型只是角度不同</button>
</div>

## 小结

- 六种模型是两族。**公式一族**(线性回归、逻辑回归、岭回归、LASSO): 每个变量乘系数,加起来。**树一族**(决策树、随机森林、梯度提升): 问一串是/否的问题,落进叶子;森林把很多棵树平均,梯度提升让很多棵小树接力。
- 同一位患者,不同的模型可以差几个 Gy。模型越灵活,越跟着建模患者里的偶然走。
- 假定得少,要学的就多,需要的患者就多,也更难交给别人用。
- 先拟合一个简单的回归模型当基准。灵活的模型要在同一种验证下赢过它,才值得用。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 基线模型 | baseline model | 不看患者、只猜平均值(或总体比例)的模型 |
| 参数模型 | parametric model | 形状事先定好、只要估计几个数的模型,如线性回归 |
| 灵活的模型 | flexible model | 不事先规定形状、由数据决定的模型,如随机森林 |
| 广义可加模型 | generalized additive model (GAM) | 每个变量各加一笔,这一笔可以是一条弯的曲线 |
| k 近邻 | k-nearest neighbours | 取最像的 k 位建模患者的平均 |
| 支持向量机 | support vector machine (SVM) | 在两类之间找一条离两边都尽量远的分界线 |
| 神经网络 | neural network | 很多层"加权求和再过弯曲的函数"叠起来 |

## 这一课没讲什么

- **第 5 节里的模型怎么用**: 这里只有一句话的说明,为的是读论文时知道它们在哪个位置。
- **每种模型的设置怎么调**: 在交叉验证里选,成绩用嵌套交叉验证报告(第 6 课)。
- **把几种模型合起来**(stacking): 让另一个模型去学怎样组合它们的预测。患者少的时候很少值得。

## 写进论文 Writing it up

<div class="paper">

方法部分交代选了哪个模型、和谁比过、为什么留下它:

> A linear regression model with three prespecified predictors was the reference model. For comparison we fitted a LASSO model, a single regression tree, a random forest and gradient-boosted trees to the same predictors, and evaluated all models with the same cross-validation splits. As none of the alternatives reduced the cross-validated error relative to the reference model, the linear model was retained.

**审稿人可能问**: 为什么不用更先进的机器学习模型?

**怎么回答**: 给出同一种验证下各个模型的成绩。说明样本量撑得起几个参数(第 13 课),以及更灵活的模型没有带来更好的预测。

**审稿人可能问**: 你们试了多少种模型?最后报告的是最好的那个吗?

**怎么回答**: 把试过的模型全部列出来,连同各自的成绩。试了很多种、只报告最好的一种,成绩偏乐观;要么事先定下主要的模型,要么用嵌套交叉验证(第 6 课)。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| `compare()` 要等一会儿 | 里面有一个 300 棵树的随机森林。浏览器里要几秒。 |
| `X does not have valid feature names` 的警告 | 模型用带列名的表拟合,预测时给的却是没有列名的数组(或反过来)。预测时也用带同样列名的 DataFrame。 |
| `KeyError: '这位患者'` | 列名要和 `compare` 交回的表完全一样,包括中文。 |
| 结果和网页上写的不一样 | 检查 `random_state=0`;代码块要从上往下依次运行。 |

## 延伸阅读 Further reading

- scikit-learn 用户指南: [Supervised learning](https://scikit-learn.org/stable/supervised_learning.html)。这一课提到的每种模型都有一节,包括没讲的那几种。
- Christodoulou E, Ma J, Collins GS, 等. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. *J Clin Epidemiol* 2019;110:12–22。
