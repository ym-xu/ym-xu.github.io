---
title: "练习 2 · 这个成绩可信吗"
summary: "训练和交叉验证;缺的肿瘤体积放进 Pipeline 里补;先挑变量再验证的虚高;事件少时的分层。"
weight: 20
after: 5
toc: false
duration: "约 35 分钟"
prereq: "[统计机器学习课第 4–5 课](/stat-ml/);[练习 1](../01-first-model/)"
questions:
  - "练习 1 的 3.99 Gy,到新病人身上还剩多少?"
  - "缺肿瘤体积的 5 位病人怎么办?"
  - "随机数也能'预测'靶区剂量吗?"
objectives:
  - "交叉验证和重复交叉验证"
  - "用 Pipeline 把补缺失值、挑变量和建模绑在一起"
  - "事件少时的分层 k 折"
---

练习 1 的模型在建模病人身上平均差 3.99 Gy。这个数是模型给自己打的分。这一部分给它一个诚实的分数。

先运行这一块。和练习 1 一样的三张表;另外准备了 5 折的分法 `cv`、一个算交叉验证平均绝对误差的函数 `cv_mae`,和 30 列随机数 `noise`(第 4 步用)。

```python
import numpy as np
import pandas as pd
from cervix import load_cervix, make_cervix_followup
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

full = load_cervix()                                            # 200 位病人,5 位缺肿瘤体积
full["needles"] = (full["technique"] == "IC/IS").astype(int)
df = full.dropna().copy()                                       # 肿瘤体积齐全的 195 位
tox = full.merge(make_cervix_followup(), on="id")               # 有随访的 185 位

cols = ["hrctv_vol", "needles"]
y = df["d90_total"]
cv = KFold(n_splits=5, shuffle=True, random_state=0)

def cv_mae(model, X, y):
    """5 折交叉验证的平均绝对误差(Gy)。"""
    return -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error").mean()

rng = np.random.default_rng(5)
noise = pd.DataFrame(rng.normal(size=(len(df), 30)), columns=[f"z{i}" for i in range(30)], index=df.index)

print(len(full), len(df), len(tox), noise.shape)
```

<div class="steps" data-steps>

<div class="step exercise">

**第 1 步 · 训练误差和交叉验证误差**

模型还是练习 1 的那个: 用 `cols` 里的两个变量预测 `y`,数据是 `df`。算它在建模病人身上的平均绝对误差(`mae_train`)和 5 折交叉验证的平均绝对误差(`mae_cv`)。

```python
# 在这里写你的代码。结果存进 mae_train 和 mae_cv。
```

```pycheck
need("mae_train", "mae_cv")
if close(mae_cv, -4.16, 0.01):
    fail("符号反了: scikit-learn 的 neg_mean_absolute_error 是负的误差。用准备好的 cv_mae,它已经把负号去掉了。")
if close(mae_train, 3.99, 0.01) and close(mae_cv, 4.16, 0.01):
    ok("对了: 训练 3.99 Gy,交叉验证 4.16 Gy。差得不多,因为模型只有两个变量,没有多少地方可以去凑建模病人。对谁都猜平均值,交叉验证是 4.48 Gy: 模型只好了 0.3 Gy。")
fail(f"现在是 {mae_train:.2f} 和 {mae_cv:.2f},应该是 3.99 和 4.16。")
```

<details><summary>提示 1: 思路</summary>

训练误差: 用全部 195 位拟合,再预测这 195 位。交叉验证误差: 把一个还没拟合的模型交给 `cv_mae`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`model = LinearRegression().fit(df[cols], y)`;`(y - model.predict(df[cols])).abs().mean()`;`cv_mae(LinearRegression(), df[cols], y)`。

</details>

<details><summary>参考答案</summary>

```python
model = LinearRegression().fit(df[cols], y)
mae_train = (y - model.predict(df[cols])).abs().mean()
mae_cv = cv_mae(LinearRegression(), df[cols], y)
print(round(mae_train, 2), round(mae_cv, 2))
```

</details>

</div>

<div class="step exercise">

**第 2 步 · 换一种分法,结果会变吗**

4.16 Gy 来自一种分法。用 `RepeatedKFold` 把 5 折交叉验证重复 10 次(`random_state=1`),得到 10 次重复各自的平均误差。其中最小的存进 `rep_low`,最大的存进 `rep_high`。

```python
# 在这里写你的代码。结果存进 rep_low 和 rep_high。
```

```pycheck
need("rep_low", "rep_high")
if close(rep_low, 3.18, 0.02) or close(rep_high, 5.06, 0.02):
    fail("这是 50 个折各自的分数里最小和最大的。题目要的是 10 次重复各自的平均: 把 50 个分数排成 10 行 5 列(reshape(10, 5)),每行取平均,再找最小和最大。")
if close(rep_low, -4.14, 0.01) or close(rep_high, -4.03, 0.01):
    fail("符号反了: neg_mean_absolute_error 是负的,前面加一个负号。")
if close(rep_low, 4.03, 0.01) and close(rep_high, 4.14, 0.01):
    ok("对了: 10 次重复在 4.03 到 4.14 Gy 之间。一种分法给出的 4.16 落在边上;报告时写重复交叉验证的平均(4.08 Gy)和这个范围,比报一个数更诚实。")
fail(f"现在是 {rep_low:.2f} 和 {rep_high:.2f},应该是 4.03 和 4.14。")
```

<details><summary>提示 1: 思路</summary>

`RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)` 当作 `cv` 交给 `cross_val_score`,得到 50 个分数,顺序是第 1 次重复的 5 折、第 2 次重复的 5 折……

</details>

<details><summary>提示 2: 用到的写法</summary>

`scores = -cross_val_score(LinearRegression(), df[cols], y, cv=rkf, scoring="neg_mean_absolute_error")`;`scores.reshape(10, 5).mean(axis=1)` 是 10 次重复各自的平均。

</details>

<details><summary>参考答案</summary>

```python
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
scores = -cross_val_score(LinearRegression(), df[cols], y, cv=rkf, scoring="neg_mean_absolute_error")
each = scores.reshape(10, 5).mean(axis=1)
rep_low, rep_high = each.min(), each.max()
print(each.round(2), round(each.mean(), 2))
```

</details>

</div>

<div class="step exercise">

**第 3 步 · 缺肿瘤体积的 5 位病人**

到现在都只用了 195 位。把全部 200 位(`full`)用上: 做一个 Pipeline,先用中位数补上缺的肿瘤体积,再做线性回归;把它的交叉验证误差存进 `mae_all`。

```python
# 在这里写你的代码。结果存进 mae_all。
```

```pycheck
need("mae_all")
if close(mae_all, 4.16, 0.005):
    fail("这还是 195 位病人的结果。X 用 full[cols],y 用 full['d90_total']。")
if close(mae_all, 4.05, 0.01):
    ok("对了: 200 位病人,4.05 Gy。补缺失值这一步在 Pipeline 里,每一轮只用留下来建模的那 4 份算中位数,被拿出来的病人没有参与。这里只缺 5 个值,放在外面补结果也差不多;病人少、缺得多的时候就不一样了。")
fail(f"现在是 {mae_all:.2f},应该是 4.05。")
```

<details><summary>提示 1: 思路</summary>

`make_pipeline(第一步, 第二步)` 把两步绑成一个模型,像普通模型一样交给 `cv_mae`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`pipe = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())`;`cv_mae(pipe, full[cols], full["d90_total"])`。

</details>

<details><summary>参考答案</summary>

```python
pipe = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
mae_all = cv_mae(pipe, full[cols], full["d90_total"])
print(round(mae_all, 2))
```

</details>

</div>

<div class="step exercise">

**第 4 步 · 随机数也能"预测"靶区剂量吗**

`noise` 里是 30 列随机数,和靶区剂量毫无关系。做两件事,都只用这 30 列,数据是 195 位病人(`y`):

- 泄漏的做法: 先用**全部**病人挑出和 `y` 关系最强的 5 列(`SelectKBest(f_regression, k=5)`),再对这 5 列做交叉验证,误差存进 `leaky`。
- 诚实的做法: 把"挑 5 列"和线性回归绑成一个 Pipeline,整个交给交叉验证,误差存进 `honest`。

```python
# 在这里写你的代码。结果存进 leaky 和 honest。
```

```pycheck
need("leaky", "honest")
if close(leaky, 4.62, 0.01) and close(honest, 4.41, 0.01):
    fail("两个数放反了: 先在全部病人里挑、再交叉验证的那个是 leaky。")
if close(leaky, honest, 0.001):
    fail("两个数一样,说明两次用的是同一种做法。泄漏的做法里,挑变量在交叉验证之前、用全部病人做;诚实的做法里,挑变量在 Pipeline 里面。")
if close(leaky, 4.41, 0.01) and close(honest, 4.62, 0.01):
    ok("对了: 泄漏的做法 4.41 Gy,比'都猜平均值'的 4.48 还好,好像随机数能预测靶区剂量。诚实的做法 4.62 Gy,比猜平均值还差: 这才是随机数的真实水平。差别只在挑变量这一步有没有偷看被拿出来的病人。")
fail(f"现在是 {leaky:.2f} 和 {honest:.2f},应该是 4.41 和 4.62。")
```

<details><summary>提示 1: 思路</summary>

泄漏的做法分两步: `SelectKBest(...).fit(noise, y)`,用 `get_support()` 拿到被挑中的 5 列,再把这 5 列交给 `cv_mae`。诚实的做法只有一步: Pipeline。

</details>

<details><summary>提示 2: 用到的写法</summary>

`picked = noise.columns[SelectKBest(f_regression, k=5).fit(noise, y).get_support()]`;`cv_mae(LinearRegression(), noise[picked], y)`;`cv_mae(make_pipeline(SelectKBest(f_regression, k=5), LinearRegression()), noise, y)`。

</details>

<details><summary>参考答案</summary>

```python
picked = noise.columns[SelectKBest(f_regression, k=5).fit(noise, y).get_support()]
leaky = cv_mae(LinearRegression(), noise[picked], y)
honest = cv_mae(make_pipeline(SelectKBest(f_regression, k=5), LinearRegression()), noise, y)
print(list(picked), round(leaky, 2), round(honest, 2))
```

</details>

</div>

<div class="step">

<div class="quiz">
<p><b>第 5 步 · 想一想</b>: 同事的做法是: 先在全部 200 位病人里做单因素分析,把 P 值小于 0.05 的变量挑出来,再用这些变量做 5 折交叉验证,报告交叉验证的成绩。这个成绩可信吗?</p>
<button class="quiz-opt" data-why="交叉验证本身没有做错,但被拿出来的病人已经参与了挑变量。他们的结局影响了哪些变量被选中,等于提前看过考卷。">可信: 成绩是交叉验证的,不是训练的</button>
<button class="quiz-opt" data-correct data-why="这和第 4 步是同一件事。单因素筛选用了全部病人的结局,被拿出来的那一份也在里面。成绩会偏乐观,候选变量越多、病人越少,偏得越厉害。筛选要放进 Pipeline,在每一轮里只用建模的那几份来做。">不可信: 挑变量用了全部病人,这一步要放进交叉验证里</button>
<button class="quiz-opt" data-why="P 值的界值不是问题所在。换成 0.01 或 0.10,只要筛选用了全部病人,泄漏就在。">不可信: 界值应该更严格,用 0.01</button>
</div>

</div>

<div class="step exercise">

**第 6 步 · 事件少的时候: 分层**

换到副作用的模型。185 位病人里只有 15 位出现了晚期直肠副作用(`tox["rect_tox"]`)。数一数每一份里有几个事件: 用普通的 5 折(准备好的 `cv`)数一次,存进 `counts_plain`;再用分层的 5 折(`StratifiedKFold`,`shuffle=True, random_state=0`)数一次,存进 `counts_strat`。两个都是 5 个数的列表。

```python
# 在这里写你的代码。结果存进 counts_plain 和 counts_strat。
```

```pycheck
need("counts_plain", "counts_strat")
try:
    plain, strat = [int(v) for v in counts_plain], [int(v) for v in counts_strat]
except Exception:
    fail("两个结果都应该是 5 个数的列表,每个数是那一份里的事件数。")
if plain == [3, 3, 3, 3, 3] and strat == [1, 3, 2, 3, 6]:
    fail("两个列表放反了。")
if strat == [1, 3, 2, 3, 6]:
    fail("counts_strat 用的还是普通的分法。分层要用 StratifiedKFold,split 的第二个参数给结局那一列。")
if plain == [1, 3, 2, 3, 6] and strat == [3, 3, 3, 3, 3]:
    ok("对了: 普通分法,五份里的事件数是 1、3、2、3、6: 有一份只有 1 个事件,在这一份上几乎没法评价模型。分层以后每份都是 3 个。事件越少,越要分层。")
fail(f"现在是 {plain} 和 {strat},应该是 [1, 3, 2, 3, 6] 和 [3, 3, 3, 3, 3]。")
```

<details><summary>提示 1: 思路</summary>

第 4 课第 6 节的写法: `分法.split(X, 结局)` 每一轮交回建模和留出的行号,数留出的那些行里结局是 1 的有几个。

</details>

<details><summary>提示 2: 用到的写法</summary>

`event = tox["rect_tox"]`;`[int(event.iloc[test].sum()) for _, test in cv.split(tox, event)]`;分层时把 `cv` 换成 `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)`。

</details>

<details><summary>参考答案</summary>

```python
event = tox["rect_tox"]
counts_plain = [int(event.iloc[test].sum()) for _, test in cv.split(tox, event)]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
counts_strat = [int(event.iloc[test].sum()) for _, test in skf.split(tox, event)]
print(counts_plain, counts_strat)
```

</details>

</div>

<div class="step exercise">

**第 7 步 · 交叉验证之后: 给一位缺数据的新病人预测**

交叉验证只用来打分。真正拿去用的,是在全部 200 位病人上重新拟合的那一个。把第 3 步的 Pipeline 在全部病人上拟合,然后给一位新病人预测: 打算加插植针,肿瘤体积还没量(`np.nan`)。预测值存进 `pred_unknown`。

```python
# 在这里写你的代码。结果存进 pred_unknown。
```

```pycheck
need("pred_unknown")
if close(pred_unknown, 92.22, 0.02):
    fail("这是不加针的预测。这位病人打算加插植针: needles 是 1。")
if close(pred_unknown, 94.20, 0.02):
    ok("对了: 94.2 Gy。体积没量,Pipeline 自动用建模病人的中位数(32.6 cm³)补上,再代进公式。补缺失值的规则也是模型的一部分: 新病人来了,用的是同一条规则。")
fail(f"现在是 {float(pred_unknown):.2f},应该是 94.20。")
```

<details><summary>提示 1: 思路</summary>

Pipeline 和普通模型一样有 `fit` 和 `predict`。新病人放进一张表,列名和 `cols` 一样,缺的那格写 `np.nan`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`final = make_pipeline(SimpleImputer(strategy="median"), LinearRegression()).fit(full[cols], full["d90_total"])`;`final.predict(pd.DataFrame({"hrctv_vol": [np.nan], "needles": [1]}))[0]`。

</details>

<details><summary>参考答案</summary>

```python
final = make_pipeline(SimpleImputer(strategy="median"), LinearRegression()).fit(full[cols], full["d90_total"])
pred_unknown = final.predict(pd.DataFrame({"hrctv_vol": [np.nan], "needles": [1]}))[0]
print(round(pred_unknown, 2))
```

</details>

</div>

</div>

**这一部分做完了**: 靶区剂量的模型,诚实的成绩是交叉验证的 4.1 Gy 左右(10 次重复在 4.03 到 4.14 之间),只比猜平均值的 4.48 Gy 好一点。挑变量和补缺失值都要放进 Pipeline。下一部分学完[第 8 课](/stat-ml/08-what-model-uses/)以后做: 换几种模型试试,看能不能预测得更好。
