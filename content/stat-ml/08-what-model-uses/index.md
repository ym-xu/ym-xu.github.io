---
title: "Lesson 8 · 模型看重什么 What the Model Relies On"
summary: "去掉一组变量、打乱一个变量、画出预测怎样随变量变化;三种问法,都不是因果。"
weight: 80
toc: true
duration: "约 75 分钟,可以分两次学: 第 0–3 节,第 4–6 节"
prereq: "本课程[第 7 课](../07-trees/)(随机森林)和[第 4 课](../04-cross-validation/)(交叉验证)"
questions:
  - "随机森林没有系数,怎么知道它靠哪些变量在预测?"
  - '"去掉一个变量"和"打乱一个变量",问的是同一件事吗?'
  - "模型看重一个变量,能说明这个变量影响结局吗?"
objectives:
  - "用成组的消融比较几组变量各自贡献了多少"
  - "自己算一个变量的置换重要性,并用 permutation_importance 核对"
  - "算出偏依赖曲线,看预测值怎样随一个变量变化"
  - "说明这三种结果为什么都不能读成因果"
---

线性回归可以读: 壁厚的系数是 1.6,就是"壁厚每厚 1 mm,预测的 Δ 高 1.6 Gy"。第 7 课的随机森林没有系数。它确实在用壁厚,可是怎么用的、用了多少,从外面看不出来。

审稿人和临床同事都会问: 这个模型靠的是什么? 这一课讲三种问法:

1. **去掉一组变量**,重新建模,看成绩掉多少。
2. **打乱一个变量**,模型不动,看成绩掉多少。
3. **把一个变量从小到大拨一遍**,看预测值怎么跟着变。

三种问法的答案都是关于**模型**的,不是关于病人的。最后一节讲为什么不能把它们读成"这个因素影响结局"。

> 🔬 第 7 课的结论是: 在这个队列上线性回归就够了。这一课故意用随机森林,不是因为它更好,而是因为它没有系数可读,正好用来练习怎么问一个"黑箱"。

**练习怎么做**: 讲解用**体积优化**计划的 Δ,练习换成**图形优化**。

## 0. 准备: 7 个变量,分成三组

```python
import numpy as np
import pandas as pd
from cohort import make_cohort
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.inspection import permutation_importance

def prepare(d):
    d = d.dropna().copy()
    d["delta"] = d["d2cc_vol"] - d["d2cc_5mm"]
    d["delta_gra"] = d["d2cc_gra"] - d["d2cc_5mm"]
    return d

df, new = prepare(make_cohort()), prepare(make_cohort(n=5000, seed=99))
groups = {
    "壁厚": ["wall_min", "wall_mean", "wall_max"],        # 同一位病人壁厚的三个测量值
    "治疗": ["diameter", "n_fx"],                         # 施源器直径、近距离次数
    "病人": ["age", "grade"],                             # 年龄、分级
}
cols = [c for members in groups.values() for c in members]            # 一共 7 个变量
cv = KFold(n_splits=5, shuffle=True, random_state=0)

def forest():
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=0)

def cv_mae(model, columns, outcome="delta"):
    return -cross_val_score(model, df[columns], df[outcome], cv=cv, scoring="neg_mean_absolute_error").mean()

def new_mae(model, columns, outcome="delta"):
    model.fit(df[columns], df[outcome])
    return (new[outcome] - model.predict(new[columns])).abs().mean()

print(f"随机森林: 交叉验证 {cv_mae(forest(), cols):.2f} Gy,新病人 {new_mae(forest(), cols):.2f} Gy")
print(f"线性回归: 交叉验证 {cv_mae(LinearRegression(), cols):.2f} Gy,新病人 {new_mae(LinearRegression(), cols):.2f} Gy")

model = forest().fit(df[cols], df["delta"])
print("这片森林一共有", sum(tree.tree_.node_count - tree.get_n_leaves() for tree in model.estimators_), "个分叉")
```

这一课的森林是 100 棵树(比第 7 课少,为了运行得快一些),用 7 个变量。壁厚有三个测量值: 最小、平均、最大,它们互相相关(平均和最小的相关系数是 0.82,和最大的是 0.62)。这是临床数据的常态: 同一件事量了好几遍。

森林在新病人身上是 1.84 Gy,线性回归是 1.73 Gy,和第 7 课一样,线性回归不输。线性回归要解释,读 7 个系数就行;森林有 1889 个分叉,没法读。

## 1. 第一种问法: 去掉一组变量

<div class="frame">

**背景**: 想知道"壁厚的信息对预测有多大用"。壁厚有三个变量,它们说的是同一件事。

**目标**: 量出每一**类**信息的贡献。

**做法**: 把一组变量整个拿掉,用剩下的变量重新建模、重新交叉验证,看误差涨了多少。这叫**消融**(ablation)。

</div>

**它在算什么**

1. 用全部变量做交叉验证,记下误差。
2. 拿掉一组变量。用剩下的变量**重新拟合**模型,再做一遍交叉验证。
3. 两次误差的差,就是这一组变量的贡献。
4. 每一组各做一次。

**打个比方**: 多学科会诊时,请影像科的医生先离开,其余的人照常讨论。少了他,结论差了多少,就是影像这一类信息的分量。

```python
full = cv_mae(forest(), cols)
print(f"全部 7 个变量: {full:.2f} Gy")
for name, members in groups.items():
    rest = [c for c in cols if c not in members]
    without = cv_mae(forest(), rest)
    print(f"去掉{name}组({len(members)} 个变量): {without:.2f} Gy,多了 {without - full:.2f}")
```

去掉壁厚组,误差从 1.67 涨到 2.16 Gy(多了 0.49);去掉治疗组,涨到 1.89 Gy(多了 0.22);去掉年龄和分级,还是 1.67 Gy。壁厚的信息最要紧,治疗的信息其次,年龄和分级对预测没有帮助。

**为什么按组去掉,而不是一个一个去掉?** 试一下只去掉平均壁厚:

```python
rest = [c for c in cols if c != "wall_mean"]
print(f"只去掉 wall_mean: {cv_mae(forest(), rest):.2f} Gy,多了 {cv_mae(forest(), rest) - full:.2f}")
```

只多了 0.14 Gy,而去掉整个壁厚组多了 0.49 Gy。平均壁厚不在了,最小壁厚和最大壁厚还在,它们顶上了大部分。单独去掉一个变量,量到的是"它比别的变量多知道多少",不是"它代表的那类信息有多重要"。相关的变量要成组地去掉;怎么分组,按临床意义事先定好。

<div class="exercise">

**练习 1 · 轮到你: 图形优化,去掉壁厚组**

对图形优化的 Δ,算全部 7 个变量时的交叉验证误差(`full_g`)和去掉壁厚组以后的(`no_wall_g`)。

```python
full_g = cv_mae(forest(), cols, "delta_gra")
rest = [c for c in cols if c not in groups[___]]
no_wall_g = cv_mae(forest(), ___, "delta_gra")
print(f"全部 {full_g:.2f} Gy,去掉壁厚组 {no_wall_g:.2f} Gy,多了 {no_wall_g - full_g:.2f}")
```

```pycheck
need("full_g", "no_wall_g")
if close(no_wall_g, 2.160, 0.004):
    fail('这是体积优化的结果。cv_mae 的第三个参数要是 "delta_gra"。')
if close(no_wall_g, 2.499, 0.004):
    fail("两个数一样: 第二个空要给去掉壁厚以后剩下的变量 rest,不是 cols。")
if close(no_wall_g, 2.633, 0.004):
    fail('这是去掉治疗组的结果。第一个空填 "壁厚"。')
if close(full_g, 2.499, 0.004) and close(no_wall_g, 3.002, 0.004):
    ok("对了: 2.50 Gy 涨到 3.00 Gy,多了 0.50 Gy。3.00 Gy 就是都猜平均值的水平(第 1 课的练习里,基线也是 3.00 Gy): 没有壁厚的信息,剩下的 4 个变量在图形优化上几乎什么也预测不了。")
fail(f"现在是 {full_g:.3f} 和 {no_wall_g:.3f},应该是 2.499 和 3.002。")
```

<details><summary>提示</summary>

第一个空是组的名字 `"壁厚"`;第二个空是剩下的变量 `rest`。

</details>

<details><summary>参考答案</summary>

```python
full_g = cv_mae(forest(), cols, "delta_gra")
rest = [c for c in cols if c not in groups["壁厚"]]
no_wall_g = cv_mae(forest(), rest, "delta_gra")
print(f"全部 {full_g:.2f} Gy,去掉壁厚组 {no_wall_g:.2f} Gy,多了 {no_wall_g - full_g:.2f}")
```

</details>

</div>

## 2. 第二种问法: 打乱一个变量

<div class="frame">

**背景**: 消融每次都要重新建模,回答的是"没有这类信息,能建出多好的模型"。有时想问的是另一件事: 眼前这个已经建好的模型,靠的是哪个变量?

**目标**: 不重新建模,量出这个模型对每个变量的依赖。

**做法**: 在没参与建模的病人里,把一个变量的值随机互换,让模型再预测一次,看误差涨了多少。这叫**置换重要性**(permutation importance)。

</div>

**它在算什么**

1. 拿一批没参与建模的病人,算模型的误差。
2. 把其中一个变量的那一列打乱: 每位病人拿到的是另一位病人的值。别的列不动。
3. 用同一个模型(不重新拟合)再预测,再算误差。
4. 误差涨了多少,就是这个变量的置换重要性。打乱有运气的成分,所以重复多次,取平均。

**为什么这样能量出依赖**: 打乱以后,这一列的数值还是那些数值,但它和这位病人、和结局的关系断了。模型如果靠这个变量,拿到错的值,预测就跟着错;模型如果不看它,预测照旧。

**打个比方**: 一叠病历里,把每个人的某一项化验单互相装错。医生的判断如果因此大乱,说明他很看重这项化验;如果照旧,说明他本来就没怎么看。

<div class="demo">

**设定**: 一个用壁厚、次数、年龄三个变量建好的模型,和 40 位没参与建模的模拟病人。横轴是模型的预测值,纵轴是实际的 Δ: 点越靠近斜线,预测越准。

**你做什么**: 点"打乱壁厚": 这 40 位病人的壁厚被随机互换,别的不动,模型重新预测。多点几次。再试"打乱次数"和"打乱年龄"。

**你会看到**: 原来的误差是 1.98 Gy。第 1 次打乱壁厚,点被横向拉开,误差多了 0.73 Gy;再点两次,是 0.99 和 0.71 Gy。打乱次数,只多 0.11、0.02、0.10 Gy。打乱年龄,点几乎不动,误差的变化不超过 0.01 Gy。

**这说明**: 模型越依赖一个变量,把它打乱,预测坏得越厉害。同一个变量每次打乱的结果不一样,所以要多打乱几次取平均。

<div class="widget" data-widget="permute">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

在课程的队列上自己做一遍。先用交叉验证的第 1 折: 156 位病人建模,40 位留出。

```python
train_idx, test_idx = next(cv.split(df))
train, test = df.iloc[train_idx], df.iloc[test_idx]
model = forest().fit(train[cols], train["delta"])
base = (test["delta"] - model.predict(test[cols])).abs().mean()            # 第 1 步

rng = np.random.default_rng(0)
shuffled = test[cols].copy()
shuffled["wall_mean"] = rng.permutation(shuffled["wall_mean"].to_numpy())   # 第 2 步: 只打乱这一列
after = (test["delta"] - model.predict(shuffled)).abs().mean()             # 第 3 步: 模型没有重新拟合
print(f"留出的 {len(test)} 位病人: 原来 {base:.2f} Gy,打乱平均壁厚以后 {after:.2f} Gy,多了 {after - base:.2f}")
```

打乱平均壁厚,误差从 1.74 涨到 2.48 Gy,多了 0.74 Gy。

scikit-learn 的 `permutation_importance` 做的是同一件事,一次算完所有变量,每个变量打乱很多次。下面把它放进交叉验证: 每一折都在留出的病人身上算,5 折取平均。

```python
def importance_cv(columns, outcome="delta"):
    """每一折: 用建模的病人拟合,在留出的病人身上把每个变量打乱 10 次;5 折取平均。"""
    total = np.zeros(len(columns))
    for train_idx, test_idx in cv.split(df):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        fitted = forest().fit(train[columns], train[outcome])
        result = permutation_importance(fitted, test[columns], test[outcome],
                                        scoring="neg_mean_absolute_error", n_repeats=10, random_state=0)
        total += result.importances_mean / 5
    return pd.Series(total, index=columns)

imp = importance_cv(cols)
print(imp.round(2))
```

平均壁厚 0.70 Gy,次数 0.19 Gy,直径 0.05 Gy;最小壁厚、最大壁厚、年龄、分级都不超过 0.02 Gy。

**两种问法的答案不一样**。只去掉平均壁厚,误差多了 0.14 Gy;打乱平均壁厚,多了 0.70 Gy。因为问的不是同一件事:

- 消融问的是: 没有这个变量,**重新建**的模型差多少? 别的变量可以顶上。
- 置换问的是: **这个已经建好的**模型有多依赖它? 模型来不及找替补。

**相关的变量会分走重要性**。把最小壁厚和最大壁厚从模型里拿掉,平均壁厚成了唯一的壁厚变量,再算一遍:

```python
five = ["wall_mean", "diameter", "n_fx", "age", "grade"]
print(importance_cv(five).round(2))
```

平均壁厚的重要性从 0.70 涨到 0.91 Gy。变量本身没变,只是同一个模型里和它相关的变量少了。所以一个变量的重要性不是它固有的性质: 它取决于模型里还有谁。

打乱还有一个副作用: 只打乱平均壁厚,最小和最大壁厚不动,会造出"平均壁厚 2 mm、最小壁厚 4 mm"这种不存在的病人。模型在这些病人身上的表现没有现实意义。相关的变量多时,成组的消融更可靠。

森林还自带一个 `feature_importances_`,不用留出病人、不用打乱,拟合完就有:

```python
all_fit = forest().fit(df[cols], df["delta"])
print(pd.Series(all_fit.feature_importances_, index=cols).round(2))
```

<div class="quiz">
<p><b>想一想</b>: 把这 7 个分数和上面的置换重要性比一比,哪里不对劲?</p>
<button class="quiz-opt" data-why="平均壁厚排第一,两种算法是一致的。不对劲的在后面几位。">平均壁厚的分数太高</button>
<button class="quiz-opt" data-correct data-why="年龄对 Δ 没有任何作用(消融和置换都说是 0),这里却得了 0.06,排在直径(0.05)前面。这个分数是在建模的病人身上数出来的: 每次分叉让误差降了多少。取值多的连续变量(年龄有几十个不同的值)有更多机会碰巧凑出一个'有用'的分叉。论文里的变量重要性图常常画的就是它;要用留出病人身上的置换重要性。">没用的年龄排在了有用的直径前面</button>
<button class="quiz-opt" data-why="分数加起来等于 1 是它的定义,不是问题。">7 个分数加起来正好是 1</button>
</div>

<div class="exercise">

**练习 2 · 轮到你: 图形优化,打乱 20 次**

对图形优化的 Δ,在第 1 折留出的 40 位病人身上把平均壁厚打乱 20 次,每次记下误差多了多少。平均值存进 `mean_gain_g`。

```python
model_g = forest().fit(train[cols], train["delta_gra"])
base_g = (test["delta_gra"] - model_g.predict(test[cols])).abs().mean()
rng = np.random.default_rng(0)
gains_g = []
for _ in range(20):
    shuffled = test[cols].copy()
    shuffled["wall_mean"] = rng.permutation(shuffled[___].to_numpy())
    after = (test["delta_gra"] - model_g.predict(shuffled)).abs().mean()
    gains_g.append(after - ___)
mean_gain_g = np.mean(gains_g)
print(f"第 1 次打乱: 多了 {gains_g[0]:.2f} Gy;20 次平均: 多了 {mean_gain_g:.2f} Gy")
```

```pycheck
need("gains_g", "mean_gain_g")
if len(gains_g) != 20:
    fail(f"gains_g 里应该有 20 个数,现在是 {len(gains_g)} 个。")
if close(mean_gain_g, 0.735, 0.004):
    fail('这是体积优化的结果。模型和误差都要用 "delta_gra"(起始代码里已经是),检查有没有改动。')
if close(gains_g[0], 0.895, 0.004) and close(mean_gain_g, 0.478, 0.004):
    ok("对了: 第 1 次打乱多了 0.90 Gy,20 次平均只有 0.48 Gy。40 位病人打乱一次,结果可以差出将近一倍: 只打乱一次就报告,是在报告运气。")
fail(f'现在第 1 次是 {gains_g[0]:.3f}、平均是 {mean_gain_g:.3f},应该是 0.895 和 0.478。第一个空是 "wall_mean",第二个空是 base_g。')
```

<details><summary>提示</summary>

第一个空是被打乱的那一列 `"wall_mean"`;第二个空是打乱之前的误差 `base_g`。

</details>

<details><summary>参考答案</summary>

```python
model_g = forest().fit(train[cols], train["delta_gra"])
base_g = (test["delta_gra"] - model_g.predict(test[cols])).abs().mean()
rng = np.random.default_rng(0)
gains_g = []
for _ in range(20):
    shuffled = test[cols].copy()
    shuffled["wall_mean"] = rng.permutation(shuffled["wall_mean"].to_numpy())
    after = (test["delta_gra"] - model_g.predict(shuffled)).abs().mean()
    gains_g.append(after - base_g)
mean_gain_g = np.mean(gains_g)
print(f"第 1 次打乱: 多了 {gains_g[0]:.2f} Gy;20 次平均: 多了 {mean_gain_g:.2f} Gy")
```

</details>

</div>

<div class="exercise">

**练习 3 · 轮到你: 图形优化,7 个变量的置换重要性**

用 `importance_cv` 算图形优化的 Δ 的置换重要性(`imp_g`),并把最重要的两个变量的名字存进 `top_two`。

```python
imp_g = importance_cv(cols, ___)
top_two = imp_g.sort_values(ascending=___).index[:2].tolist()
print(imp_g.round(2))
print("最重要的两个:", top_two)
```

```pycheck
need("imp_g", "top_two")
if close(imp_g["wall_mean"], 0.700, 0.004):
    fail('这是体积优化的结果。第一个空填 "delta_gra"。')
if list(top_two) == ["wall_mean", "n_fx"] and close(imp_g["wall_mean"], 0.845, 0.004) and close(imp_g["n_fx"], 0.312, 0.004):
    ok("对了: 平均壁厚 0.85 Gy,次数 0.31 Gy,其余都在 0 附近(负数是打乱的运气,读成 0)。直径在体积优化里还有一点分量,在图形优化里没有: 和第 3 课读系数时看到的一致。")
if len(top_two) == 2 and top_two[0] != "wall_mean":
    fail("排在最前面的应该是 wall_mean: 从大到小排,第二个空是 False。")
fail(f"wall_mean 现在是 {imp_g['wall_mean']:.3f},应该是 0.845;最重要的两个应该是 wall_mean 和 n_fx。")
```

<details><summary>提示</summary>

第一个空是结局 `"delta_gra"`;从大到小排,`ascending` 是 `False`。

</details>

<details><summary>参考答案</summary>

```python
imp_g = importance_cv(cols, "delta_gra")
top_two = imp_g.sort_values(ascending=False).index[:2].tolist()
print(imp_g.round(2))
print("最重要的两个:", top_two)
```

</details>

</div>

## 3. 第三种问法: 预测值怎样随一个变量变化

前两种问法回答"多重要"。还有一个问题没回答: 模型是**怎么用**壁厚的? 壁越厚,预测值是高还是低? 是一条直线,还是有台阶?

<div class="frame">

**背景**: 线性回归里,一个系数就说完了变量的用法。森林没有系数。

**目标**: 画出模型眼里"壁厚 → 预测值"的形状。

**做法**: 把每位病人的壁厚都改成同一个值,其他变量各用各的,让模型预测,取平均。换一个值再做一遍。把这些平均值连起来,就是**偏依赖曲线**(partial dependence)。

</div>

**它在算什么**

1. 选一个值,比如 3 mm。
2. 把所有病人的平均壁厚都改成 3 mm。年龄、次数、直径等等不动。
3. 让模型给每位病人预测,取平均。
4. 换成 4 mm、5 mm……重复。每个值得到一个平均预测值。

这是在问模型一个"假如": 假如这些病人别的都不变,只是壁厚都是 3 mm,你平均会给多少? 注意,问的是模型,不是病人。

```python
model = forest().fit(df[cols], df["delta"])
line = LinearRegression().fit(df[cols], df["delta"])

def dependence(fitted, data, values, column="wall_mean"):
    out = []
    for v in values:
        changed = data[cols].copy()
        changed[column] = float(v)                       # 第 2 步: 所有人的这个变量都改成 v
        out.append(fitted.predict(changed).mean())       # 第 3 步
    return np.round(out, 2)

grid = [2, 3, 4, 5, 6]
print("平均壁厚 (mm):", grid)
print("随机森林:     ", dependence(model, df, grid))
print("线性回归:     ", dependence(line, df, grid))
```

线性回归的一行,每格大约相差 1.67 Gy: 就是壁厚的系数,一条直线,不用算也知道。森林的一行不是等距的: 从 2 mm 到 3 mm 几乎不动(−3.87 到 −3.56),从 3 mm 到 4 mm 跳了 2.1 Gy。它是用台阶在凑一条斜线(第 7 课)。

画出来更清楚。细一些的格子,再把病人按次数分开各画一条:

```python
import matplotlib.pyplot as plt

fine = np.arange(2, 6.01, 0.1)
fig, axes = plt.subplots(2, 1, figsize=(4.4, 6.4), sharex=True, sharey=True)
axes[0].plot(fine, dependence(model, df, fine), label="Random forest")
axes[0].plot(fine, dependence(line, df, fine), label="Linear regression", ls="--")
axes[0].set_title("All patients", fontsize=10)
for k in (2, 3):
    axes[1].plot(fine, dependence(model, df[df["n_fx"] == k], fine), label=f"{k} fractions")
axes[1].set_title("Random forest, by number of fractions", fontsize=10)
axes[1].set_xlabel("Mean wall thickness (mm)")
for ax in axes:
    ax.set_ylabel("Mean predicted Δ (Gy)")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

for k in (2, 3):
    print(f"做 {k} 次的病人:", dependence(model, df[df["n_fx"] == k], grid))
```

下面一张图是第 7 课那个交互作用的样子。壁薄的时候(2 mm),做 3 次的病人预测值是 −5.47 Gy,做 2 次的是 −2.54 Gy;壁厚到 4 mm 以上,两条线合在了一起。森林自己学到了"壁越薄,次数的差别越大"。

**曲线两头不可信**。平均壁厚不小于 5.5 mm 的病人只有 13 位。6 mm 处的那个数,是模型在很少的数据上外推出来的;而且"所有人的平均壁厚都是 6 mm,最小和最大壁厚还是原来的",这样的病人大多并不存在。读偏依赖曲线,只读数据密集的那一段。

<div class="exercise">

**练习 4 · 轮到你: 图形优化,壁厚从 2 mm 到 6 mm**

对图形优化的 Δ 拟合一片森林,算平均壁厚为 2 mm 和 6 mm 时的平均预测值(`pd_g`),以及两者相差多少(`rise_g`)。

```python
model_pg = forest().fit(df[cols], df[___])
pd_g = dependence(model_pg, df, [2, 6])
rise_g = pd_g[___] - pd_g[0]
print(f"2 mm: {pd_g[0]} Gy;6 mm: {pd_g[1]} Gy;相差 {rise_g:.2f} Gy")
```

```pycheck
need("pd_g", "rise_g")
if close(rise_g, 5.52, 0.02):
    fail('这是体积优化的结果。第一个空填 "delta_gra"。')
if close(pd_g[0], -5.69, 0.02) and close(rise_g, 6.50, 0.02):
    ok("对了: 2 mm 时平均预测 −5.69 Gy,6 mm 时 0.81 Gy,相差 6.50 Gy。体积优化是 5.52 Gy: 图形优化的 Δ 随壁厚变得更快。")
fail(f"现在 2 mm 是 {pd_g[0]}、相差 {rise_g:.2f},应该是 −5.69 和 6.50。第二个空是 1(6 mm 是第二个值)。")
```

<details><summary>提示</summary>

第一个空是结局 `"delta_gra"`;`pd_g` 里第一个数(位置 0)是 2 mm,第二个数(位置 `1`)是 6 mm。

</details>

<details><summary>参考答案</summary>

```python
model_pg = forest().fit(df[cols], df["delta_gra"])
pd_g = dependence(model_pg, df, [2, 6])
rise_g = pd_g[1] - pd_g[0]
print(f"2 mm: {pd_g[0]} Gy;6 mm: {pd_g[1]} Gy;相差 {rise_g:.2f} Gy")
```

</details>

</div>

## 4. 这些都不是因果

三种问法的答案,说的都是**模型怎样用这些变量来预测**。它们不说明: 如果改变这个变量,病人的结局会跟着变。这一点最容易被读过头。

**例一: 次数,还是外照射?** 置换重要性里,近距离次数排第二。能不能说"次数影响 Δ"? 看一眼这个队列:

```python
print(pd.crosstab(df["arm"], df["n_fx"]))

df_b = df.assign(ebrt=(df["arm"] == "EBRT+VCB").astype(int))
swapped = [c if c != "n_fx" else "ebrt" for c in cols]                   # 把"次数"换成"是否做了外照射"
cv_b = -cross_val_score(forest(), df_b[swapped], df_b["delta"], cv=cv, scoring="neg_mean_absolute_error").mean()
print(f"用次数: {cv_mae(forest(), cols):.2f} Gy;换成是否外照射: {cv_b:.2f} Gy")
```

加外照射的 107 位全都做 2 次,单纯近距离的 89 位全都做 3 次。"次数"和"治疗方式"在这批病人里是同一列数。把次数换成"是否做了外照射",模型的成绩一样。模型说"这一列有用";至于有用的是次数、是外照射,还是两组病人别的差别,它分不出来,数据里也没有能分出来的信息。

**例二: 没用的变量也能领到功劳**。最小壁厚和最大壁厚在造数据时根本没有参与 Δ 的计算(这是模拟队列,可以翻底牌)。可是去掉平均壁厚以后,模型靠它们补回了大半(第 1 节: 误差只多 0.14 Gy,不是 0.49 Gy)。它们和平均壁厚相关,所以能预测。能预测,不等于有作用。

**例三: 一个有名的真事**。上世纪九十年代,一个预测肺炎病人死亡风险的模型学到了一条规律: 有哮喘病史的病人,死亡风险**更低**。数据没有错: 有哮喘的肺炎病人通常直接收进 ICU,治疗更积极,所以结局确实更好。如果按模型的说法把他们当作低风险、安排回家,后果正好相反。模型如实地描述了数据,数据里装着医生的处理方式。

所以,变量重要性的用处是**检查模型**: 它靠的东西合不合临床常识? 有没有靠着不该靠的东西(比如医院编号、检查日期、病历号)? 至于"这个因素是不是原因",要靠研究设计和临床知识来回答,第 3 课说的"解释"的分析也只是其中一步。

<div class="quiz">
<p><b>想一想</b>: 假设有一篇论文用随机森林预测晚期毒性,置换重要性排第一的是"是否做过阴道扩张器指导"。作者写: "扩张器指导是毒性最重要的危险因素。"这句话的问题在哪里?</p>
<button class="quiz-opt" data-why="置换重要性是在留出的病人身上算的,正是为了避免过拟合造成的假象。问题不在这里。">随机森林过拟合了,重要性不可信</button>
<button class="quiz-opt" data-correct data-why="重要性只说明模型靠这个变量来预测。很可能是医生看到病人已经有粘连或狭窄的苗头,才安排了指导: 变量是结局的信号,不是原因。按这句话去减少指导,就像把哮喘病人送回家。">重要性说的是模型靠什么预测,不是什么导致了毒性;这个变量可能是医生对早期症状的反应</button>
<button class="quiz-opt" data-why="排第一就是数值最大。问题不在排序,而在把'对预测重要'读成了'危险因素'。">排第一不代表数值大</button>
</div>

## 5. 三种问法放在一起

| 问法 | 问的是什么 | 要重新拟合吗 | 要留心 |
|---|---|---|---|
| 成组的消融 | 没有这类信息,能建出多好的模型 | 要 | 组要按临床意义事先分好 |
| 置换重要性 | 这个建好的模型有多依赖这个变量 | 不要 | 相关的变量互相分走重要性;在留出的病人身上算 |
| 偏依赖曲线 | 模型的预测值怎样随这个变量变 | 不要 | 数据稀少的两头不可信 |

三种都在回答"模型靠什么",都不回答"什么导致了结局"。

## 小结

- **消融**: 去掉一组变量,重新建模,看误差涨多少。相关的变量要成组地去掉。
- **置换重要性**: 在留出的病人身上打乱一个变量,模型不动,看误差涨多少。多打乱几次取平均。
- 两者回答的问题不同: 平均壁厚的消融是 0.14 Gy,置换是 0.70 Gy。
- **偏依赖曲线**: 把一个变量拨到各个值,看平均预测值。能看出直线、台阶和交互作用。
- 三种结果都是关于模型的,**不是因果**。它们的用处是检查模型靠的东西合不合常识。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 消融 | ablation | 去掉一部分变量重新建模,看成绩掉多少 |
| 置换重要性 | permutation importance | 打乱一个变量,看建好的模型成绩掉多少 |
| 偏依赖 | partial dependence | 把一个变量固定在某个值时,模型的平均预测 |
| 不纯度重要性 | impurity-based importance | 森林自带的 `feature_importances_`;在建模病人身上算,偏向取值多的变量 |
| 黑箱 | black box | 内部规则没法直接读的模型 |
| 混杂 | confounding | 两个变量一起变,分不出是谁的作用 |

## 这一课没讲什么

- **SHAP**: 把每一位病人的预测值拆成各个变量的贡献,论文里很常见。它回答的是"这位病人的预测为什么是这个数",和这一课的三种问法一样,解释的是模型,不是因果;相关的变量同样会互相分走贡献。
- **个体条件期望曲线**(ICE): 偏依赖曲线是所有病人的平均;不取平均,每位病人画一条,就是 ICE。上面按次数分开画的两条线是它的简化版。
- **重要性的不确定性**: 置换重要性也是从这批病人估出来的,换一批会变。可以对病人做 bootstrap 给它一个区间。
- **因果推断**: 用观察性数据回答"改变这个因素会怎样",需要另一套方法和假定,这门课不讲。

## 写进论文 Writing it up

<div class="paper">

方法部分写明用的是哪一种、在哪批病人身上算的:

> The contribution of each predictor group (wall thickness, treatment, patient characteristics) was assessed by refitting the model without that group and comparing the cross-validated error. Permutation importance was computed on the held-out fold of each cross-validation split (10 permutations per predictor) and averaged over folds. These analyses describe how the model uses the predictors and do not support causal interpretation.

**审稿人可能问**: 变量重要性是在训练集上算的吗?

**怎么回答**: 置换重要性在每一折留出的病人身上算;没有使用模型自带的不纯度重要性。

**审稿人可能问**: 几个壁厚指标高度相关,单个变量的重要性怎么解读?

**怎么回答**: 相关的变量会互相分走重要性,所以正文以成组的消融为主,单个变量的置换重要性放在补充材料,并注明这一点。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| 置换重要性出现负数 | 打乱以后误差碰巧小了一点。说明这个变量的重要性在 0 附近,读成 0。 |
| 每次运行,重要性都不一样 | `permutation_importance` 和 `default_rng` 没有固定随机种子。 |
| `SettingWithCopyWarning` | 改一列之前先 `.copy()`,像上面的 `test[cols].copy()`。 |
| 重要性那一段很慢 | 每一折要打乱 7 个变量各 10 次;等十几秒。 |
| 结果和网页上写的不一样 | 检查 `random_state` 和 `default_rng(0)`;代码块要从上往下依次运行。 |

## 延伸阅读 Further reading

- scikit-learn 用户指南: [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html) 和 [Partial Dependence and Individual Conditional Expectation plots](https://scikit-learn.org/stable/modules/partial_dependence.html)。
- Molnar C. [*Interpretable Machine Learning*](https://christophm.github.io/interpretable-ml-book/)。一本免费的在线书,这一课的三种方法和 SHAP 都有专门的章节。
- Caruana R, Lou Y, Gehrke J, 等. Intelligible models for healthcare: predicting pneumonia risk and hospital 30-day readmission. *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2015: 1721–1730。第 4 节里哮喘和肺炎的例子出自这里。

**下一课**: 前面八课一直用"平均差多少 Gy"(MAE)来评价模型。连续结局还有别的尺子: R² 是什么,它为什么会是负的;预测值为什么总比实际值"保守"。第 9 课专门讲连续结局的评价。

{{< practice >}}
