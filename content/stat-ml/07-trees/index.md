---
title: "Lesson 7 · 树、随机森林和梯度提升 Trees and Ensembles"
summary: "一棵树怎么分叉;很多棵树怎么投票、怎么接力;和简单的基线模型比。"
weight: 70
toc: true
duration: "约 80 分钟,可以分两次学: 第 0–3 节,第 4–6 节"
prereq: "本课程[第 4 课](../04-cross-validation/)(交叉验证);[第 1 课](../01-linear-regression/)(线性回归)"
questions:
  - "不用公式,模型还能长什么样?"
  - "一棵树是怎么长出来的?长得太大会怎样?"
  - "很多棵树合起来,为什么比一棵准?"
  - "复杂的模型一定比线性回归好吗?"
objectives:
  - "自己找出一棵树的第一个分叉点,和 scikit-learn 的结果对上"
  - "说明树的深度怎样影响训练误差和新病人身上的误差"
  - "说出随机森林和梯度提升各自是怎样把很多棵树合起来的"
  - "把树模型和线性回归、基线放在一起比,再决定用哪个"
---

前六课的模型都是一条公式: 系数乘以变量,加起来。公式的好处是简单,坏处是它只会画直线: 壁厚每厚 1 mm,预测值变多少,对谁都一样。

可是第 1 课末尾提过,在这个队列里并不一样。单纯近距离的病人做 3 次,壁厚的斜率是 2.42;加外照射的病人做 2 次,斜率只有 1.02。直线模型看不出这一点,除非你事先想到,并且把它写进公式。

这一课讲另一类模型。它们不用公式,而是问一串是/否的问题: "壁厚超过 3.35 mm 吗?""做了 3 次吗?"顺着答案走下去,走到头就是预测值。这就是**决策树**(decision tree)。一棵树不太准;把几百棵合起来,就是临床预测模型的论文里最常见的两种"机器学习模型": **随机森林**(random forest)和**梯度提升**(gradient boosting)。

> 🔬 这一课的结尾可能出乎意料: 在这个队列上,最准的不是它们。先和简单的模型比,是这一课最要紧的一句话。

**练习怎么做**: 讲解用**体积优化**计划的 Δ,练习换成**图形优化**。

## 0. 准备

```python
import numpy as np
import pandas as pd
from cohort import make_cohort
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import KFold, cross_val_score

def prepare(d):
    d = d.dropna().copy()
    d["delta"] = d["d2cc_vol"] - d["d2cc_5mm"]
    d["delta_gra"] = d["d2cc_gra"] - d["d2cc_5mm"]
    return d

df, new = prepare(make_cohort()), prepare(make_cohort(n=5000, seed=99))
cols = ["wall_mean", "diameter", "n_fx"]
cv = KFold(n_splits=5, shuffle=True, random_state=0)

def three_errors(model, outcome="delta"):
    """建模病人身上的、交叉验证的、新病人身上的平均绝对误差(Gy)。"""
    cv_error = -cross_val_score(model, df[cols], df[outcome], cv=cv, scoring="neg_mean_absolute_error").mean()
    model.fit(df[cols], df[outcome])
    train_error = (df[outcome] - model.predict(df[cols])).abs().mean()
    new_error = (new[outcome] - model.predict(new[cols])).abs().mean()
    return train_error, cv_error, new_error

print("基线: 训练 %.2f,交叉验证 %.2f,新病人 %.2f Gy" % three_errors(DummyRegressor()))
print("线性回归: 训练 %.2f,交叉验证 %.2f,新病人 %.2f Gy" % three_errors(LinearRegression()))
```

`three_errors` 给每个模型算三个数: 在建模病人身上的误差,交叉验证的误差,在 4900 位新病人身上的误差。两个老朋友先垫底: 都猜平均值,新病人身上 2.28 Gy;三个变量的线性回归,1.73 Gy。后面每个模型都和这两个数比。

## 1. 一棵树的第一步: 分成两组

最简单的树只问一个问题,把病人分成两组,每组都猜自己这组的平均值。

<div class="demo">

**设定**: 第 1 课的那 30 位模拟病人。竖的虚线是分叉点: 壁厚不超过它的归左边,超过的归右边。两段黑线是两组各自的平均值,也就是这棵"树"的预测值。下面的条是误差平方的平均,和第 1 课调直线时是同一把尺子。

**你做什么**: 拖动分叉点,找误差最小的位置。然后点"找最好的分叉点"。

**你会看到**: 不分组时是 6.60。分叉点在 2.5 mm 时是 5.77;挪到 4.3 mm,降到 3.42: 壁厚不超过 4.3 mm 的 20 位都猜 −3.77 Gy,超过的 10 位都猜 0.01 Gy。

**这说明**: 分在哪里,由误差决定。树没有系数;它要"学"的,就是在哪个变量的哪个位置下刀。

<div class="widget" data-widget="split">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

顺便说一句: 第 1 课的最小二乘直线在这 30 位病人身上是 3.90,一次分叉是 3.42,比直线还低。别急着下结论,这是在建模的病人身上算的。

<div class="frame">

**背景**: 直线假定变量的作用处处相同。实际的关系可能有台阶,可能在不同的病人里不一样。

**目标**: 一个不假定直线的模型,让数据自己说在哪里拐弯。

**做法**: 反复把病人一分为二。每次都挑能让误差平方和降得最多的那一刀。

</div>

**它在算什么**

1. 对每一个变量、每一个可能的分叉点,试着把病人分成两组;每组用自己的平均值当预测值,算误差平方和。
2. 取误差平方和最小的那一个"变量 + 分叉点",下第一刀。
3. 对分出来的每一组,重复第 1、2 步。
4. 什么时候停: 分到规定的层数(深度),或者一组里的人太少。最后的每一组叫一片**叶子**(leaf),落进这片叶子的病人,预测值是叶子里建模病人的平均值。

第 1、2 步自己写一遍,只看壁厚:

```python
x, y = df["wall_mean"], df["delta"]
best = None
for cut in np.sort(x.unique())[:-1]:                     # 每一个可能的分叉点
    left, right = y[x <= cut], y[x > cut]
    sse = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
    if best is None or sse < best[0]:
        best = (sse, cut, left.mean(), right.mean(), len(left), len(right))
print(f"最好的一刀: 壁厚 ≤ {best[1]} mm。左边 {best[4]} 位,猜 {best[2]:.2f} Gy;右边 {best[5]} 位,猜 {best[3]:.2f} Gy")
print(f"误差平方的平均: 不分 {((y - y.mean()) ** 2).mean():.2f},分一刀 {best[0] / len(y):.2f}")

stump = DecisionTreeRegressor(max_depth=1, random_state=0).fit(df[cols], y)
print("scikit-learn:", cols[stump.tree_.feature[0]], "≤", round(stump.tree_.threshold[0], 2))
```

最好的一刀是壁厚 ≤ 3.3 mm: 左边 81 位,猜 −3.93 Gy;右边 115 位,猜 −0.79 Gy。误差平方的平均从 7.96 降到 5.57(第 1 课的直线是 4.87)。scikit-learn 的树(`max_depth=1`,只分一层)选的是同一刀;它报的分叉点是 3.35,取在 3.3 和下一个值 3.4 的正中间。三个变量里它也选了壁厚: 没有哪个变量的哪一刀比这更好。

## 2. 接着分: 树自己找到了"次数不同,规律不同"

让树分两层,打印出来看:

```python
tree2 = DecisionTreeRegressor(max_depth=2, random_state=0).fit(df[cols], y)
print(export_text(tree2, feature_names=cols, decimals=2))
print("训练 %.2f,交叉验证 %.2f,新病人 %.2f Gy" % three_errors(DecisionTreeRegressor(max_depth=2, random_state=0)))
```

这棵树读起来像一张流程图:

- 壁厚 ≤ 3.35 mm 的病人,再看近距离做了几次: 2 次的,预测 −2.69 Gy;3 次的,预测 −5.13 Gy。
- 壁厚 > 3.35 mm 的病人,再按壁厚分一次: 不超过 5.10 mm 的 −1.34 Gy,超过的 +1.40 Gy。

没有人告诉它"次数会影响壁厚的作用",它自己在薄壁的病人里按次数分了一刀: 壁薄、又做 3 次的病人,降得最多。这种"一个变量的作用取决于另一个变量",叫**交互作用**(interaction)。线性回归要你事先写进公式;树在分叉的过程中自然会碰上。

两层的树有 4 片叶子,也就是只会说 4 个不同的预测值。新病人身上的误差是 1.92 Gy: 比基线(2.28)好,比线性回归(1.73)差。

<div class="exercise">

**练习 1 · 轮到你: 图形优化的第一刀**

对图形优化的 Δ,找只看壁厚时最好的一刀: 分叉点存进 `cut_g`,左右两组的预测值存进 `left_g`、`right_g`。

```python
yg = df["delta_gra"]
best_g = None
for cut in np.sort(x.unique())[:-1]:
    left, right = yg[x <= cut], yg[x > ___]
    sse = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).___()
    if best_g is None or sse < best_g[0]:
        best_g = (sse, cut, left.mean(), right.mean())
cut_g, left_g, right_g = best_g[1], best_g[2], best_g[3]
print(f"壁厚 ≤ {cut_g} mm: 左边猜 {left_g:.2f} Gy,右边猜 {right_g:.2f} Gy")
```

```pycheck
need("cut_g", "left_g", "right_g")
if close(left_g, -3.93, 0.02):
    fail("这是体积优化的结果。循环里要用 yg(起始代码里已经是),检查有没有改动。")
if close(cut_g, 3.3, 0.001) and close(left_g, -5.64, 0.02) and close(right_g, -2.00, 0.02):
    ok("对了: 也是壁厚 ≤ 3.3 mm。左边猜 −5.64 Gy,右边猜 −2.00 Gy。图形优化降得更多,两组的差距也更大(3.6 Gy;体积优化是 3.1 Gy)。")
fail(f"现在是 ≤ {cut_g} mm,左 {left_g:.2f}、右 {right_g:.2f},应该是 3.3、−5.64、−2.00。第一个空是 cut,第二个空是 sum。")
```

<details><summary>提示</summary>

右边一组是壁厚超过这个分叉点的病人: `x > cut`。两组的误差平方各自加起来: `sum`。

</details>

<details><summary>参考答案</summary>

```python
yg = df["delta_gra"]
best_g = None
for cut in np.sort(x.unique())[:-1]:
    left, right = yg[x <= cut], yg[x > cut]
    sse = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
    if best_g is None or sse < best_g[0]:
        best_g = (sse, cut, left.mean(), right.mean())
cut_g, left_g, right_g = best_g[1], best_g[2], best_g[3]
print(f"壁厚 ≤ {cut_g} mm: 左边猜 {left_g:.2f} Gy,右边猜 {right_g:.2f} Gy")
```

</details>

</div>

## 3. 分得越细越好吗

树可以一直分下去,直到每片叶子里只剩一位病人。

<div class="quiz">
<p><b>先猜一猜</b>: 树越分越深,新病人身上的误差会怎样?</p>
<button class="quiz-opt" data-why="训练误差确实一直降。新病人身上的误差是另一回事: 运行下面的代码,看它在哪一层开始回升。">一直降: 分得越细,预测越精确</button>
<button class="quiz-opt" data-correct data-why="这是第 4 课的老故事: 树越深,越有本事把建模病人的偶然也记下来。运行下面的代码,看具体在哪一层回升。">先降后升: 分到一定程度就开始记住偶然</button>
<button class="quiz-opt" data-why="深度不同,叶子的数目和每片叶子里的人数都不同,预测值会跟着变。运行下面的代码看看。">不变: 深度只影响训练误差</button>
</div>

```python
for depth in [1, 2, 3, 4, 6, 10]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=0)
    errors = three_errors(model)
    print(f"深度 {depth:2d}: {model.get_n_leaves():3d} 片叶子,训练 {errors[0]:.2f},交叉验证 {errors[1]:.2f},新病人 {errors[2]:.2f} Gy")
```

训练误差一路下降: 深度 10 时有 128 片叶子,196 位病人差不多一人一片,训练误差只有 0.81 Gy。新病人身上的误差在深度 3 时最低(1.89 Gy),然后回升,到深度 10 是 2.19 Gy,和都猜平均值(2.28)差不多了。交叉验证的误差跟着新病人的走,不跟着训练误差走: 这就是第 4 课要它的原因。

单独一棵树有两个毛病。太浅,只会说几个数;太深,记住了偶然。它还很不稳定: 换一批病人,第一刀可能就下在别处,整棵树都跟着变。

## 4. 随机森林: 很多棵树,取平均

<div class="frame">

**背景**: 一棵深的树把建模病人的偶然也记了进去。换一批病人,记住的偶然就不同。

**目标**: 留下各棵树都认同的规律,让各自的偶然互相抵消。

**做法**: 造很多批"差不多的病人",每批种一棵树,预测时取所有树的平均。这叫**随机森林**。

</div>

**它在算什么**

1. 从建模病人里有放回地抽一批同样多的病人(Python 课第 5 课的 bootstrap)。
2. 用这一批种一棵树,让它长得比较深。
3. 重复几百次,得到几百棵各不相同的树。
4. 预测一位新病人: 每棵树各给一个数,取平均。

每棵树记住的偶然来自它那一批病人,各不相同,一平均就抵消了大半;而"壁薄的降得多"这样的规律,每一批病人里都有,平均以后留了下来。

**打个比方**: 一位医生的判断会受最近几个病例的影响。多学科会诊时,几位医生各有各的偏差,合起来的意见往往比其中任何一位都稳。

先自己做一遍: 抽 300 批,种 300 棵树,取平均。

```python
rng = np.random.default_rng(0)
X_np, y_np, X_new = df[cols].to_numpy(), df["delta"].to_numpy(), new[cols].to_numpy()
total = np.zeros(len(new))
for _ in range(300):
    idx = rng.integers(0, len(df), len(df))                          # 第 1 步: 有放回地抽一批
    one_tree = DecisionTreeRegressor(min_samples_leaf=5, random_state=0).fit(X_np[idx], y_np[idx])    # 第 2 步
    total += one_tree.predict(X_new)
forest_by_hand = total / 300                                         # 第 4 步: 取平均

single = DecisionTreeRegressor(min_samples_leaf=5, random_state=0).fit(X_np, y_np)
print(f"一棵树:         新病人 {np.abs(new['delta'] - single.predict(X_new)).mean():.2f} Gy")
print(f"300 棵树取平均: 新病人 {np.abs(new['delta'] - forest_by_hand).mean():.2f} Gy")
```

`min_samples_leaf=5` 让每片叶子至少有 5 位病人。一棵这样的树,新病人身上 1.92 Gy;300 棵取平均,1.84 Gy。树还是那种树,只是多了,误差就降了。

scikit-learn 的 `RandomForestRegressor` 做的是同一件事,另外还可以让每次分叉只看一部分变量,使各棵树更不一样:

```python
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0)
print("随机森林: 训练 %.2f,交叉验证 %.2f,新病人 %.2f Gy" % three_errors(forest))
```

新病人身上 1.83 Gy,和自己做的 1.84 Gy 是一个水平(抽到的病人不同,所以不会一模一样)。这段要种 1800 棵树,等几秒。

<div class="exercise">

**练习 2 · 轮到你: 图形优化的随机森林**

对图形优化的 Δ,算同一个随机森林的三个误差,把新病人身上的那个存进 `forest_new_g`。

```python
forest_g = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0)
errors_g = three_errors(forest_g, ___)
forest_new_g = errors_g[___]
print("训练 %.2f,交叉验证 %.2f,新病人 %.2f Gy" % errors_g)
```

```pycheck
need("forest_new_g")
if close(forest_new_g, 1.834, 0.004):
    fail('这是体积优化的结果。第一个空填 "delta_gra"。')
if close(forest_new_g, 2.083, 0.004):
    fail("这是训练误差(第 0 个)。three_errors 交回的三个数依次是训练、交叉验证、新病人,新病人是第 2 个。")
if close(forest_new_g, 2.466, 0.004):
    fail("这是交叉验证的误差(第 1 个)。新病人是第 2 个。")
if close(forest_new_g, 2.654, 0.004):
    ok("对了: 训练 2.08,交叉验证 2.47,新病人 2.65 Gy。线性回归在新病人身上是 2.55 Gy: 图形优化上,随机森林也没有赢过它。")
fail(f"forest_new_g 现在是 {forest_new_g:.3f},应该是 2.654。")
```

<details><summary>提示</summary>

第一个空是 `"delta_gra"`;三个数从 0 开始数,新病人是第 `2` 个。

</details>

<details><summary>参考答案</summary>

```python
forest_g = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0)
errors_g = three_errors(forest_g, "delta_gra")
forest_new_g = errors_g[2]
print("训练 %.2f,交叉验证 %.2f,新病人 %.2f Gy" % errors_g)
```

</details>

</div>

## 5. 梯度提升: 很多棵小树,一棵接一棵

随机森林的树各种各的,最后取平均。梯度提升的树是接力的: 每一棵都去补前面那些树还没做对的部分。

<div class="frame">

**背景**: 一棵浅的树只会说几个数,太粗。

**目标**: 用很多棵浅树,一点一点把预测修准。

**做法**: 先给所有人猜平均值。然后反复做: 看每位病人现在还差多少(残差),种一棵小树去预测这些残差,把小树的预测乘以一个小的比例加到现有的预测上。

</div>

**它在算什么**

1. 一开始,每位病人的预测值都是平均值。
2. 算残差: 实际值减去现在的预测值。
3. 种一棵很浅的树(这里是 2 层),让它去预测**残差**。
4. 把这棵树的预测乘以**学习率**(learning rate,这里是 0.05),加到现在的预测上。只加一小步,免得被某一棵树带偏。
5. 回到第 2 步。重复 100 次。

**打个比方**: 调剂量。先按常规给一个起始剂量,复查,看离目标还差多少,调一小步;再复查,再调。每次只调一点,多调几次。

自己写一遍:

```python
pred = np.full(len(df), df["delta"].mean())              # 第 1 步
pred_new = np.full(len(new), df["delta"].mean())
for step in range(1, 101):
    residual = df["delta"] - pred                        # 第 2 步
    small = DecisionTreeRegressor(max_depth=2, random_state=0).fit(df[cols], residual)      # 第 3 步
    pred += 0.05 * small.predict(df[cols])               # 第 4 步
    pred_new += 0.05 * small.predict(new[cols])
    if step in (1, 10, 50, 100):
        print(f"第 {step:3d} 棵以后: 训练 {(df['delta'] - pred).abs().mean():.2f},新病人 {(new['delta'] - pred_new).abs().mean():.2f} Gy")
```

第 1 棵以后,新病人身上 2.24 Gy,只比都猜平均值好一点点。10 棵以后 2.01,50 棵以后 1.80,100 棵以后 1.78 Gy。

scikit-learn 的 `GradientBoostingRegressor`,设成同样的 100 棵、学习率 0.05、每棵 2 层:

```python
from sklearn.ensemble import GradientBoostingRegressor

boost = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=0)
print("梯度提升: 训练 %.2f,交叉验证 %.2f,新病人 %.2f Gy" % three_errors(boost))
print(f"和自己写的最多相差 {np.abs(boost.predict(new[cols]) - pred_new).max():.1e} Gy")
```

新病人身上 1.78 Gy,和自己写的每一个预测值都对得上(相差不到一亿分之一 Gy)。XGBoost、LightGBM 这些名字,做的是同一件事,加了许多让它更快、更不容易过拟合的细节。

梯度提升有三个要定的设置: 多少棵树,学习率多大,每棵多深。它们互相牵制(学习率小,就要更多棵),定得不好很容易过拟合。把上面的 `learning_rate` 改成 0.5 再运行一次,看训练误差和新病人的误差怎样分开。实际使用时,这三个数要像第 6 课那样在交叉验证里选。

<div class="exercise">

**练习 3 · 轮到你: 图形优化的梯度提升**

对图形优化的 Δ,用同样设置的梯度提升,把交叉验证的误差存进 `boost_cv_g`,新病人身上的存进 `boost_new_g`。

```python
boost_g = GradientBoostingRegressor(n_estimators=___, learning_rate=0.05, max_depth=2, random_state=0)
errors_b = three_errors(boost_g, "delta_gra")
boost_cv_g, boost_new_g = errors_b[1], errors_b[___]
print("训练 %.2f,交叉验证 %.2f,新病人 %.2f Gy" % errors_b)
```

```pycheck
need("boost_cv_g", "boost_new_g")
if close(boost_new_g, 1.784, 0.004):
    fail('这是体积优化的结果。three_errors 的第二个参数应该是 "delta_gra"(起始代码里已经是),检查有没有改动。')
if close(boost_cv_g, 2.514, 0.004) and close(boost_new_g, 2.617, 0.004):
    ok("对了: 交叉验证 2.51,新病人 2.62 Gy。三个模型在图形优化上排下来: 线性回归 2.55,梯度提升 2.62,随机森林 2.65 Gy。")
fail(f"现在是交叉验证 {boost_cv_g:.3f}、新病人 {boost_new_g:.3f},应该是 2.514 和 2.617。第一个空是 100,第二个空是 2。")
```

<details><summary>提示</summary>

第一个空是树的棵数 `100`;第二个空是新病人误差在三个数里的位置 `2`。

</details>

<details><summary>参考答案</summary>

```python
boost_g = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=0)
errors_b = three_errors(boost_g, "delta_gra")
boost_cv_g, boost_new_g = errors_b[1], errors_b[2]
print("训练 %.2f,交叉验证 %.2f,新病人 %.2f Gy" % errors_b)
```

</details>

</div>

## 6. 和简单的模型比一比

把这一课的模型排在一起(体积优化,4900 位新病人身上的平均误差):

| 模型 | 新病人身上的误差 |
|---|---|
| 都猜平均值 | 2.28 Gy |
| 一棵树,2 层 | 1.92 Gy |
| 随机森林,300 棵 | 1.83 Gy |
| 梯度提升,100 棵小树 | 1.78 Gy |
| 线性回归,3 个变量 | 1.73 Gy |

最简单的线性回归最准。几百棵树没有赢过一条三个系数的公式。

这在临床数据上很常见,原因有三个:

- **病人少**。196 位病人,撑不起很灵活的模型。树模型要靠数据自己发现"在哪里拐弯",每发现一处都要花掉一些病人;公式把"是直线"这个假定白送给了模型,不花病人。
- **规律本来就接近直线**。壁厚对 Δ 的作用大体是直线,树只能用一级一级的台阶去凑。
- **变量少**。树模型擅长在很多变量里找出复杂的搭配。只有三个变量时,没多少可找的。

树模型确实发现了线性回归没有的东西: 次数不同,壁厚的规律不同。那把这个交互作用写进线性回归(加一列"壁厚 × 次数"),会不会更准?

```python
d2 = df.assign(wall_x_fx=df["wall_mean"] * df["n_fx"])
n2 = new.assign(wall_x_fx=new["wall_mean"] * new["n_fx"])
with_x = cols + ["wall_x_fx"]
cv_x = -cross_val_score(LinearRegression(), d2[with_x], d2["delta"], cv=cv, scoring="neg_mean_absolute_error").mean()
inter = LinearRegression().fit(d2[with_x], d2["delta"])
print(f"加上交互项: 交叉验证 {cv_x:.2f},新病人 {(n2['delta'] - inter.predict(n2[with_x])).abs().mean():.2f} Gy")
for k in (2, 3):
    part = df[df["n_fx"] == k]
    slope = LinearRegression().fit(part[["wall_mean"]], part["delta"]).coef_[0]
    print(f"做 {k} 次的 {len(part)} 位: 壁厚的斜率 {slope:.2f}")
```

交叉验证的误差从 1.70 降到 1.61 Gy,看上去有帮助。新病人身上却是 1.73 Gy,和不加一样。

这是模拟队列,可以翻开底牌。造数据时,两组的真实斜率是 1.28 和 1.92: 规律是真的。这 196 位病人估出来的是 1.02 和 2.42: 方向对,差距被夸大了一倍多。按夸大的差距去预测新病人,得到的和赔掉的正好相抵。交叉验证没有看出来,是因为它只能在这 196 位病人内部检验,每一折都带着同一个偶然(第 4 课第 4 节: 它消除不了"碰巧是这批病人"的运气)。

所以,论文里写"我们用了随机森林(或 XGBoost)"本身不说明什么。要问的是: **和一个简单的、事先定好的回归模型相比,它在新病人身上好了多少?** 好得不多,就用简单的: 好解释,好核对,换一家医院也更经得住。

<div class="quiz">
<p><b>想一想</b>: 一篇论文报告: 随机森林的训练误差 1.41 Gy,线性回归的训练误差 1.67 Gy,结论是随机森林更好。这个比较有什么问题?</p>
<button class="quiz-opt" data-why="两个数的单位相同,可以比。问题在于这两个数都是在建模病人身上算的。">两种模型的误差不能放在一起比</button>
<button class="quiz-opt" data-correct data-why="灵活的模型在建模病人身上总是更好看: 它能记住更多偶然。这一课里随机森林的训练误差正是 1.41 Gy,线性回归 1.67 Gy;到新病人身上,是 1.83 对 1.73,排名反过来了。比较模型,只能用交叉验证或新病人身上的成绩。">比的是训练误差。越灵活的模型训练误差越小,到新病人身上排名可能反过来</button>
<button class="quiz-opt" data-why="300 棵树只是让森林更稳定。问题不在树的数目,而在比的是训练误差。">随机森林的树太少</button>
</div>

## 小结

- **决策树**: 反复把病人一分为二,每次挑让误差平方和降得最多的那一刀。叶子里的平均值就是预测值。它不假定直线,能自己发现交互作用。
- 树越深,训练误差越小,新病人身上的误差先降后升。单独一棵树不稳定。
- **随机森林**: 很多批重抽的病人,各种一棵树,取平均。偶然互相抵消。
- **梯度提升**: 很多棵小树接力,每一棵去补前面还差的部分,每次只加一小步。
- **先和简单的模型比**。病人不多、变量不多、关系接近直线时,线性回归常常不输,而且更好解释。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 决策树 | decision tree | 一串是/否的问题,走到头是预测值 |
| 分叉 | split | 按一个变量的一个值把病人分成两组 |
| 叶子 | leaf | 不再分的一组病人 |
| 深度 | depth | 从头到叶子最多问几个问题 |
| 交互作用 | interaction | 一个变量的作用取决于另一个变量 |
| 随机森林 | random forest | 很多棵树各自预测,取平均 |
| 装袋 | bagging (bootstrap aggregating) | 重抽很多批病人各建一个模型,再平均 |
| 梯度提升 | gradient boosting | 很多棵小树接力,每棵去预测前面的残差 |
| 学习率 | learning rate | 每棵小树的预测只加进去多大比例 |
| 集成模型 | ensemble | 把很多个模型合起来用 |

## 这一课没讲什么

- **分类的树**: 结局是是/否时,叶子里给的是比例,分叉的标准也换了(基尼系数或 log loss),做法相同。`RandomForestClassifier`、`GradientBoostingClassifier` 的用法和这一课一样;第 10–12 课评价是/否结局的模型时会用到。
- **树模型给出的概率常常不准**: 随机森林的概率偏向中间,梯度提升的可能过于极端。第 11 课讲校准。
- **调参数**: 深度、棵数、学习率、每片叶子最少几位病人,都应该在交叉验证里选(第 6 课的 `GridSearchCV`),成绩用嵌套交叉验证报告。
- **模型看重哪些变量**: 几百棵树没法像公式那样读系数。第 8 课讲怎么问一个黑箱"你看重什么"。

## 写进论文 Writing it up

<div class="paper">

方法部分要写明模型的设置,并且交代和谁比过:

> We compared a prespecified linear regression model with a random forest (300 trees, minimum leaf size 5) and gradient-boosted trees (100 trees of depth 2, learning rate 0.05). All models used the same three predictors and were evaluated with the same cross-validation splits. The linear model was retained because the tree-based models did not reduce the cross-validated error.

**审稿人可能问**: 为什么不用机器学习模型?(或者反过来: 为什么用了?)

**怎么回答**: 给出同一批病人、同一种验证下各个模型的成绩。如果灵活的模型没有明显更好,保留简单的模型是合理的选择,也符合报告指南"先和简单模型比较"的要求。

**审稿人可能问**: 随机森林的参数是怎么定的?

**怎么回答**: 要么事先固定并写明,要么在交叉验证里选并用嵌套交叉验证报告成绩。不能看着最终成绩来回调。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| 每次运行,树或森林的结果略有不同 | 没有固定 `random_state`。 |
| 训练误差接近 0,交叉验证的误差很大 | 树太深,过拟合了。限制 `max_depth`,或加大 `min_samples_leaf`。 |
| 随机森林那一段很慢 | 300 棵树乘以 6 次拟合,浏览器里要几秒到十几秒。 |
| `export_text` 打印出来的分叉点带 .5 或 .35 | 树把分叉点取在相邻两个取值的正中间。 |
| 结果和网页上写的不一样 | 检查 `random_state` 和 `default_rng(0)`;代码块要从上往下依次运行。 |

## 延伸阅读 Further reading

- scikit-learn 用户指南: [Decision Trees](https://scikit-learn.org/stable/modules/tree.html) 和 [Ensembles: Gradient boosting, random forests, bagging, voting, stacking](https://scikit-learn.org/stable/modules/ensemble.html)。
- Christodoulou E, Ma J, Collins GS, 等. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. *J Clin Epidemiol* 2019;110:12–22。汇总了很多项临床预测研究的比较,结论和这一课第 6 节一致。

**下一课**: 线性回归的系数可以读。300 棵树的森林没有系数,它看重哪个变量、怎么用的,从外面看不出来。第 8 课讲三种问法: 去掉一组变量看成绩掉多少,把一个变量打乱看成绩掉多少,以及画出预测值怎样随一个变量变化。
