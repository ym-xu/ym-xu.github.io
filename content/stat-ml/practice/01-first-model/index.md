---
title: "练习 1 · 第一个预测模型"
summary: "用肿瘤体积和施源器预测靶区剂量,给新患者预测;直肠剂量和晚期副作用的 OR。"
weight: 10
after: 3
toc: false
duration: "约 35 分钟"
prereq: "[统计机器学习课第 1–3 课](/stat-ml/)"
questions:
  - "做计划之前,能从肿瘤体积和施源器猜到靶区剂量吗?"
  - "直肠剂量每高 5 Gy,出现晚期副作用的 odds 变成几倍?"
objectives:
  - "线性回归: 拟合、读系数、给新患者预测"
  - "逻辑回归: 系数换成 OR,算一位患者的概率"
  - "分清'系数是多少'和'预测得准不准'"
---

审计已经做完了。科里接着问: 靶区剂量不够的那些患者,能不能在做计划之前就看出来?

先运行这一块。`full` 是全部 200 位患者;`df` 是肿瘤体积齐全的 195 位,这一部分的靶区剂量模型都用它(缺体积的 5 位留到练习 2);`tox` 是有随访的 185 位,多一列 `rect_tox`。

```python
import numpy as np
import pandas as pd
from cervix import load_cervix, make_cervix_followup
from sklearn.linear_model import LinearRegression, LogisticRegression

full = load_cervix()                                            # 200 位患者
full["needles"] = (full["technique"] == "IC/IS").astype(int)    # 加了插植针记 1,单纯腔内记 0
df = full.dropna().copy()                                       # 肿瘤体积齐全的 195 位
tox = full.merge(make_cervix_followup(), on="id")               # 有随访的 185 位,多一列 rect_tox

print(len(full), len(df), len(tox))
```

<div class="steps" data-steps>

<div class="step exercise">

**第 1 步 · 要预测的量长什么样**

在 `df` 里,靶区总剂量 `d90_total` 的平均值是多少?有几位患者低于 85 Gy?分别存进 `mean_d90` 和 `n_low`。

```python
# 在这里写你的代码。结果存进 mean_d90 和 n_low。
```

```pycheck
need("mean_d90", "n_low")
if close(mean_d90, 92.49, 0.01) or n_low == 19:
    fail("这是全部 200 位患者(full)的结果。这一部分用 df: 肿瘤体积齐全的 195 位。")
if close(mean_d90, 92.28, 0.01):
    fail("这是中位数。题目要平均值: mean()。")
if close(mean_d90, 92.54, 0.01) and n_low == 18:
    ok("对了: 平均 92.5 Gy,195 位里有 18 位低于 85 Gy。如果对谁都猜 92.5 Gy,就永远猜不出这 18 位。")
fail(f"现在是 {mean_d90} 和 {n_low},应该是 92.54 和 18。")
```

<details><summary>提示 1: 思路</summary>

一列的平均值;一个条件成立的行数,是那一列 True/False 加起来。

</details>

<details><summary>提示 2: 用到的写法</summary>

`df["d90_total"].mean()`;`(df["d90_total"] < 85).sum()`。

</details>

<details><summary>参考答案</summary>

```python
mean_d90 = df["d90_total"].mean()
n_low = (df["d90_total"] < 85).sum()
print(round(mean_d90, 2), n_low)
```

</details>

</div>

<div class="step exercise">

**第 2 步 · 一个变量: 肿瘤体积**

用肿瘤体积 `hrctv_vol` 一个变量预测 `d90_total`,拟合一条最小二乘直线。斜率存进 `slope_vol`,截距存进 `icpt_vol`。

```python
# 在这里写你的代码。结果存进 slope_vol 和 icpt_vol。
```

```pycheck
need("slope_vol", "icpt_vol")
if close(slope_vol, 96.49, 0.01) and close(icpt_vol, -0.111, 0.002):
    fail("两个数放反了: 斜率在 coef_[0] 里,截距在 intercept_ 里。")
if close(slope_vol, -0.111, 0.002) and close(icpt_vol, 96.49, 0.02):
    ok("对了: 预测 D90 = 96.49 − 0.111 × 体积。肿瘤每大 10 cm³,预测的靶区剂量低 1.1 Gy。")
fail(f"现在斜率是 {slope_vol:.3f},截距是 {icpt_vol:.2f};应该是 −0.111 和 96.49。检查 X 是不是 df[['hrctv_vol']](两层方括号),y 是不是 df['d90_total']。")
```

<details><summary>提示 1: 思路</summary>

第 1 课的三件事: 建一个 `LinearRegression()`,用 `fit(X, y)` 拟合,再从模型里取出斜率和截距。`X` 要是一张表。

</details>

<details><summary>提示 2: 用到的写法</summary>

`model = LinearRegression().fit(df[["hrctv_vol"]], df["d90_total"])`;斜率是 `model.coef_[0]`,截距是 `model.intercept_`。

</details>

<details><summary>参考答案</summary>

```python
model1 = LinearRegression().fit(df[["hrctv_vol"]], df["d90_total"])
slope_vol = model1.coef_[0]
icpt_vol = model1.intercept_
print(f"预测 D90 = {icpt_vol:.2f} {slope_vol:+.3f} × 体积")
```

</details>

</div>

<div class="step exercise">

**第 3 步 · 加上施源器**

再加一个变量: `needles`(加了插植针是 1,没加是 0)。用 `hrctv_vol` 和 `needles` 两个变量拟合,模型存进 `model2`;两个系数分别存进 `b_vol` 和 `b_needles`。

```python
# 在这里写你的代码。模型存进 model2,系数存进 b_vol 和 b_needles。
```

```pycheck
need("model2", "b_vol", "b_needles")
if close(b_vol, 1.925, 0.01) and close(b_needles, -0.138, 0.002):
    fail("两个系数放反了。coef_ 里的顺序和 fit 时列的顺序一样: 先体积,后施源器。")
if close(b_vol, -0.111, 0.002):
    fail("这还是只有体积一个变量的模型。X 里要有两列: df[['hrctv_vol', 'needles']]。")
if close(b_vol, -0.138, 0.002) and close(b_needles, 1.925, 0.01):
    ok("对了: 预测 D90 = 96.79 − 0.138 × 体积 + 1.92 × 加针。体积相同时,加了针的患者靶区剂量高 1.9 Gy。注意体积的斜率从 −0.111 变成了 −0.138。")
fail(f"现在是 {b_vol:.3f} 和 {b_needles:.3f},应该是 −0.138 和 1.925。")
```

<details><summary>提示 1: 思路</summary>

和上一步一样,只是 `X` 里放两列。系数按列的顺序排在 `coef_` 里。

</details>

<details><summary>提示 2: 用到的写法</summary>

`model2 = LinearRegression().fit(df[["hrctv_vol", "needles"]], df["d90_total"])`;`model2.coef_[0]` 和 `model2.coef_[1]`。

</details>

<details><summary>参考答案</summary>

```python
model2 = LinearRegression().fit(df[["hrctv_vol", "needles"]], df["d90_total"])
b_vol, b_needles = model2.coef_
print(f"预测 D90 = {model2.intercept_:.2f} {b_vol:+.3f} × 体积 {b_needles:+.2f} × 加针")
```

</details>

</div>

<div class="step">

<div class="quiz">
<p><b>第 4 步 · 想一想</b>: 只看体积时,斜率是 −0.111;把施源器也放进模型以后,体积的斜率变成 −0.138,更陡了。为什么?</p>
<button class="quiz-opt" data-why="系数的大小和变量的个数没有这样的关系。加一个和结局、和体积都无关的变量,体积的斜率几乎不变。">变量越多,每个系数都会变大</button>
<button class="quiz-opt" data-correct data-why="大肿瘤更常加针(加针的患者体积中位数 44.8 cm³,没加的 28.2 cm³),而针把剂量补回来一些。只看体积时,'肿瘤大'这件事里混着'更可能加了针',体积的害处被针的好处抵掉一部分,斜率就显得平。把施源器放进模型,问的是'施源器相同时,体积每大 1 cm³ 降多少',答案更陡。">大肿瘤更常加针,针补回了一部分剂量;不看施源器时,体积的斜率被冲淡了</button>
<button class="quiz-opt" data-why="两次都是最小二乘,都没有算错。同一个变量,模型里别的变量不同,系数回答的问题就不同(第 1 课的'其他不变时')。">其中一个模型算错了</button>
</div>

</div>

<div class="step exercise">

**第 5 步 · 给两位新患者预测**

两位新患者,肿瘤体积都是 60 cm³: 一位打算单纯腔内,一位打算加插植针。用 `model2` 预测他们的靶区剂量,分别存进 `pred_ic` 和 `pred_is`。

```python
# 在这里写你的代码。结果存进 pred_ic 和 pred_is。
```

```pycheck
need("pred_ic", "pred_is")
if close(pred_ic, 90.43, 0.02) and close(pred_is, 88.50, 0.02):
    fail("两位患者放反了: needles 是 0 的那位是单纯腔内。")
if close(pred_ic, 89.85, 0.02):
    fail("这是只用体积的那个模型的预测。题目要用 model2: 新患者的表里要有 hrctv_vol 和 needles 两列。")
if close(pred_ic, 88.50, 0.02) and close(pred_is, 90.43, 0.02):
    ok("对了: 单纯腔内 88.5 Gy,加针 90.4 Gy。两个数都在 85 Gy 以上,但不加针的那位只高出 3.5 Gy。这个预测有多可靠,下一步来量。")
fail(f"现在是 {float(pred_ic):.2f} 和 {float(pred_is):.2f},应该是 88.50 和 90.43。")
```

<details><summary>提示 1: 思路</summary>

新患者也要放进一张表,列名和拟合时一样;`predict` 交回一个数组,每位患者一个数。

</details>

<details><summary>提示 2: 用到的写法</summary>

`new = pd.DataFrame({"hrctv_vol": [60, 60], "needles": [0, 1]})`;`model2.predict(new)`,再用 `[0]` 和 `[1]` 取出两个数。

</details>

<details><summary>参考答案</summary>

```python
new = pd.DataFrame({"hrctv_vol": [60, 60], "needles": [0, 1]})
pred_ic, pred_is = model2.predict(new)
print(round(pred_ic, 2), round(pred_is, 2))
```

</details>

</div>

<div class="step exercise">

**第 6 步 · 预测得有多准**

系数说的是"有没有关系",预测要问的是"差多少"。在这 195 位患者身上算两个平均绝对误差: 对谁都猜平均值时差多少(`mae_base`),用 `model2` 预测时差多少(`mae_model`)。

```python
# 在这里写你的代码。结果存进 mae_base 和 mae_model。
```

```pycheck
need("mae_base", "mae_model")
if close(mae_base, 5.45, 0.02) or close(mae_model, 5.04, 0.02):
    fail("这是先平方、取平均、再开方的结果。题目要平均绝对误差: 误差取绝对值 abs(),再取平均。")
if close(mae_model, 4.07, 0.01):
    fail("这是只用体积的那个模型。题目要 model2。")
if close(mae_base, 4.38, 0.01) and close(mae_model, 3.99, 0.01):
    ok("对了: 都猜平均值,平均差 4.38 Gy;用模型,平均差 3.99 Gy。两个系数都'有关系',预测却只好了 0.4 Gy。而且这是在建模的患者身上算的,对新患者只会更差: 这是练习 2 的题目。")
fail(f"现在是 {mae_base:.2f} 和 {mae_model:.2f},应该是 4.38 和 3.99。")
```

<details><summary>提示 1: 思路</summary>

误差是实际值减预测值。基线的预测值对每个人都是平均值;模型的预测值来自 `model2.predict`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`(y - y.mean()).abs().mean()`;`(y - model2.predict(df[["hrctv_vol", "needles"]])).abs().mean()`,其中 `y = df["d90_total"]`。

</details>

<details><summary>参考答案</summary>

```python
y = df["d90_total"]
mae_base = (y - y.mean()).abs().mean()
mae_model = (y - model2.predict(df[["hrctv_vol", "needles"]])).abs().mean()
print(round(mae_base, 2), round(mae_model, 2))
```

</details>

</div>

<div class="step exercise">

**第 7 步 · 换成是/否的结局: 直肠剂量和晚期副作用**

在有随访的 185 位患者(`tox`)里,用直肠总剂量 `rectum_total` 预测有没有出现晚期直肠副作用 `rect_tox`。用课上的写法 `LogisticRegression(C=1e6)`。算两个数: 直肠总剂量每高 **5 Gy**,odds 变成几倍(`or_5gy`);直肠总剂量 70 Gy 的患者,出现副作用的概率(`p70`)。

骨架已经给出,把三处 `___` 填上。

```python
logit = LogisticRegression(C=1e6).fit(tox[["rectum_total"]], tox["rect_tox"])
# 系数: 直肠总剂量每高 1 Gy,log-odds 增加多少
b = logit.coef_[0][0]
# 每高 5 Gy,odds 变成几倍
or_5gy = ___
# 一位直肠总剂量 70 Gy 的新患者,放进一张表
new = pd.DataFrame({"rectum_total": [___]})
# [0, 1]: 第一位患者、结局为 1(出现副作用)的概率
p70 = logit.___(new)[0, 1]
print(round(or_5gy, 2), round(p70, 3))
```

```pycheck
need("or_5gy", "p70")
if close(or_5gy, 1.147, 0.005):
    fail("这是每高 1 Gy 的 OR。每高 5 Gy: 系数先乘以 5,再取 exp。")
if close(or_5gy, 5.73, 0.03):
    fail("每高 5 Gy 的 OR 不是 1 Gy 的 OR 乘以 5,而是它的 5 次方: np.exp(5 * 系数)。")
if close(p70, 0.872, 0.005):
    fail("这是'不出现副作用'的概率。predict_proba 交回两列,第 1 列([:, 1])才是结局为 1 的概率。")
if close(or_5gy, 1.98, 0.02) and close(p70, 0.128, 0.003):
    ok("对了: 直肠总剂量每高 5 Gy,odds 变成 1.98 倍,差不多翻一倍;70 Gy 的患者,概率是 13%。185 位里只有 15 位出现了副作用: 这么少的事件能撑起多大的模型,是最后一部分的题目。")
fail(f"现在是 {or_5gy:.2f} 和 {p70:.3f},应该是 1.98 和 0.128。")
```

<details><summary>提示 1: 思路</summary>

第 2 课: 系数是 log-odds 的变化,取 exp 才是 OR;变量变 5 个单位,先把系数乘以 5。概率用 `predict_proba`,新患者放进一张表。

</details>

<details><summary>提示 2: 用到的写法</summary>

`np.exp(5 * b)`;`{"rectum_total": [70]}`;概率是 `logit.predict_proba(new)`。

</details>

<details><summary>参考答案</summary>

```python
logit = LogisticRegression(C=1e6).fit(tox[["rectum_total"]], tox["rect_tox"])
# 系数: 直肠总剂量每高 1 Gy,log-odds 增加多少
b = logit.coef_[0][0]
# 每高 5 Gy,odds 变成几倍
or_5gy = np.exp(5 * b)
# 一位直肠总剂量 70 Gy 的新患者,放进一张表
new = pd.DataFrame({"rectum_total": [70]})
# [0, 1]: 第一位患者、结局为 1(出现副作用)的概率
p70 = logit.predict_proba(new)[0, 1]
print(round(or_5gy, 2), round(p70, 3))
```

</details>

</div>

</div>

**这一部分做完了**: 你有了两个模型。靶区剂量: 预测 D90 = 96.79 − 0.138 × 体积 + 1.92 × 加针,在建模患者身上平均差 3.99 Gy。晚期直肠副作用: 直肠总剂量每高 5 Gy,odds 变成 1.98 倍。下一部分学完[第 5 课](/stat-ml/05-data-leakage/)以后做: 这些成绩,到新患者身上还剩多少。
