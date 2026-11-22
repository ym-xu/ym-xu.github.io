---
title: "练习 4 · 模型够不够好"
summary: "靶区剂量模型的误差、R² 和校准斜率;副作用模型的 AUC 和区间、校准、净获益。"
weight: 40
after: 12
toc: false
duration: "约 40 分钟"
prereq: "[统计机器学习课第 9–12 课](/stat-ml/);[练习 3](../03-more-models/)"
questions:
  - "靶区剂量的模型能不能把剂量不够的患者挑出来?"
  - "副作用的模型分得开患者吗?它说的概率准吗?"
  - "按模型做决定,比对所有人都处理更好吗?"
objectives:
  - "out-of-fold 预测、MAE、R² 和校准斜率"
  - "AUC 和 bootstrap 区间、O/E 和校准斜率"
  - "阈值概率和净获益"
---

两个模型都有了: 靶区剂量用带交互项的直线(练习 3),晚期直肠副作用用直肠总剂量的逻辑回归(练习 1)。这一部分给它们各开一张成绩单。

先运行这一块。准备好的有: 两张表,两种分法(`cv` 给连续的结局,分层的 `skf` 给是/否的结局),和第 9、11、12 课里的几个函数。

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from cervix import load_cervix, make_cervix_followup
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, r2_score

full = load_cervix()
full["needles"] = (full["technique"] == "IC/IS").astype(int)
df = full.dropna().copy()                                       # 195 位,靶区剂量的模型
df["vol_x_needles"] = df["hrctv_vol"] * df["needles"]
xcols = ["hrctv_vol", "needles", "age", "vol_x_needles"]
y = df["d90_total"]
tox = full.merge(make_cervix_followup(), on="id")               # 185 位,副作用的模型
event = tox["rect_tox"]

cv = KFold(n_splits=5, shuffle=True, random_state=0)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

def calibration_slope(actual, predicted):
    """连续结局的校准斜率: 实际值对预测值的直线的斜率(第 9 课)。"""
    return LinearRegression().fit(np.asarray(predicted).reshape(-1, 1), actual).coef_[0]

def logit(prob):
    return np.log(prob / (1 - prob))

def calibration(actual, prob):
    """是/否结局: O/E、校准截距、校准斜率(第 11 课)。"""
    actual, prob = np.asarray(actual), np.asarray(prob)
    lp = logit(prob)
    slope = sm.Logit(actual, sm.add_constant(lp)).fit(disp=0).params[1]
    intercept = sm.GLM(actual, np.ones(len(actual)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return actual.sum() / prob.sum(), intercept, slope

def net_benefit(actual, prob, pt):
    """概率达到 pt 的患者就处理: 这种做法的净获益(第 12 课)。"""
    actual, prob = np.asarray(actual), np.asarray(prob)
    do = prob >= pt
    useful, wasted = (do & (actual == 1)).sum(), (do & (actual == 0)).sum()
    return useful / len(actual) - wasted / len(actual) * pt / (1 - pt)

def net_benefit_all(actual, pt):
    """对所有人都处理的净获益。"""
    rate = np.mean(actual)
    return rate - (1 - rate) * pt / (1 - pt)

print(len(df), "位患者,", int((y < 85).sum()), "位靶区剂量低于 85 Gy;", len(tox), "位有随访,", int(event.sum()), "位出现副作用")
```

<div class="steps" data-steps>

<div class="step exercise">

**第 1 步 · 靶区剂量: 每位患者一个诚实的预测值**

用 `cross_val_predict` 得到带交互项的直线(`xcols`)给每位患者的 out-of-fold 预测值,存进 `pred`(一个和 `y` 一样长的 Series)。再算平均绝对误差(`mae_oof`)和 R²(`r2_oof`)。

骨架已经给出,把三处 `___` 填上。

```python
# 每位患者由没见过他的那一折的模型来预测
oof = cross_val_predict(LinearRegression(), ___, y, cv=cv)
# 变成和 y 对得上的 Series
pred = pd.Series(oof, index=df.index)
# 平均绝对误差和 R²
mae_oof = ___
r2_oof = ___(y, pred)
print(round(mae_oof, 2), round(r2_oof, 2))
```

```pycheck
need("pred", "mae_oof", "r2_oof")
if close(r2_oof, 0.295, 0.01):
    fail("这是在建模患者身上的 R²: 预测值来自用全部患者拟合的模型。pred 要用 cross_val_predict,每位患者的预测值来自没见过她的那一轮。")
if close(mae_oof, 3.71, 0.01) and close(r2_oof, 0.24, 0.01):
    ok("对了: 平均差 3.71 Gy,R² 0.24。模型解释了靶区剂量差异的四分之一: 比没有强,但四分之三的差异它说不出来。")
fail(f"现在是 {mae_oof:.2f} 和 {r2_oof:.2f},应该是 3.71 和 0.24。")
```

<details><summary>提示 1: 思路</summary>

`cross_val_predict(模型, X, y, cv=cv)` 交回每位患者被留出时的预测值。R² 可以用 `r2_score(实际, 预测)`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`cross_val_predict(LinearRegression(), df[xcols], y, cv=cv)`;`(y - pred).abs().mean()`;`r2_score(y, pred)`。

</details>

<details><summary>参考答案</summary>

```python
# 每位患者由没见过他的那一折的模型来预测
oof = cross_val_predict(LinearRegression(), df[xcols], y, cv=cv)
# 变成和 y 对得上的 Series
pred = pd.Series(oof, index=df.index)
# 平均绝对误差和 R²
mae_oof = (y - pred).abs().mean()
r2_oof = r2_score(y, pred)
print(round(mae_oof, 2), round(r2_oof, 2))
```

</details>

</div>

<div class="step exercise">

**第 2 步 · 靶区剂量: 校准斜率**

用准备好的 `calibration_slope` 算实际值对 out-of-fold 预测值的斜率,存进 `slope_d90`。

```python
# 在这里写你的代码。结果存进 slope_d90。
```

```pycheck
need("slope_d90")
if close(slope_d90, 0.27, 0.02):
    fail("两个参数放反了: 第一个是实际值 y,第二个是预测值 pred。")
if close(slope_d90, 0.89, 0.01):
    ok("对了: 0.89,比 1 小一点。预测值每高 1 Gy,实际只高 0.89 Gy: 模型说得稍微过头了,预测得特别高和特别低的患者,实际没有那么极端。195 位患者、5 个系数,这个程度的过头是常见的。")
fail(f"现在是 {slope_d90:.2f},应该是 0.89。")
```

<details><summary>提示 1: 思路</summary>

`calibration_slope(实际值, 预测值)`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`calibration_slope(y, pred)`。

</details>

<details><summary>参考答案</summary>

```python
slope_d90 = calibration_slope(y, pred)
print(round(slope_d90, 2))
```

</details>

</div>

<div class="step exercise">

**第 3 步 · 用预测值挑患者**

模型的用处是提前找出剂量可能不够的患者。定一条规则: out-of-fold 预测值低于 87 Gy 的患者,提前安排加针。按这条规则会挑出几位患者(`flagged`)?实际低于 85 Gy 的 18 位里,挑中了几位(`found`)?

```python
# 在这里写你的代码。结果存进 flagged 和 found。
```

```pycheck
need("flagged", "found")
if int(flagged) == 4 and int(found) == 2:
    fail("这是按 85 Gy 挑的。规则是预测值低于 87 Gy。")
if int(flagged) == 10 and int(found) == 5:
    ok("对了: 挑出 10 位,其中 5 位确实低于 85 Gy,另外 5 位是虚惊;18 位剂量不够的患者里只找到 5 位,漏了 13 位。把界线放宽到 90 Gy,能找到 9 位,代价是 21 位虚惊。平均误差 3.7 Gy 的模型,做不了这件事的全部。")
fail(f"现在是 {flagged} 和 {found},应该是 10 和 5。")
```

<details><summary>提示 1: 思路</summary>

两个条件: 预测值低于 87;实际值低于 85。挑出的人数是第一个条件成立的行数;挑中的是两个条件都成立的行数(`&`)。

</details>

<details><summary>提示 2: 用到的写法</summary>

`(pred < 87).sum()`;`((pred < 87) & (y < 85)).sum()`。

</details>

<details><summary>参考答案</summary>

```python
flagged = (pred < 87).sum()
found = ((pred < 87) & (y < 85)).sum()
print(flagged, found)
```

</details>

</div>

<div class="step exercise">

**第 4 步 · 副作用: 分得开吗**

换到副作用的模型: 用直肠总剂量 `rectum_total` 预测 `event`,`LogisticRegression(C=1e6)`,数据是 `tox`。用分层的 `skf` 得到每位患者的 out-of-fold 概率,存进 `p`(一个 Series);再算 AUC,存进 `auc`。

骨架已经给出,把三处 `___` 填上。

```python
model = LogisticRegression(C=1e6)
# 用分层的 skf;要的是概率,不是 0 和 1
proba = cross_val_predict(model, tox[["rectum_total"]], event, cv=___, method=___)
# [:, 1]: 每位患者结局为 1(出现副作用)的那一列
p = pd.Series(proba[:, 1], index=tox.index)
auc = ___
print(round(auc, 3))
```

```pycheck
need("p", "auc")
if close(auc, 0.714, 0.004):
    fail("这是在建模患者身上的 AUC。p 要用 cross_val_predict(..., cv=skf, method='predict_proba'),取第 1 列。")
if close(auc, 0.319, 0.01):
    fail("取成了'不出现副作用'的概率。predict_proba 的第 1 列([:, 1])才是结局为 1 的概率。")
if close(auc, 0.681, 0.004):
    ok("对了: AUC 0.68(在建模患者身上是 0.71)。随便挑一位出现副作用的和一位没出现的,模型把前者排在前面的机会是 68%。比抛硬币强,离'分得开'还远。")
fail(f"现在是 {auc:.3f},应该是 0.681。")
```

<details><summary>提示 1: 思路</summary>

第 10 课的写法: `cross_val_predict` 加上 `method="predict_proba"`,交回两列概率,要第 1 列。

</details>

<details><summary>提示 2: 用到的写法</summary>

`cv=skf, method="predict_proba"`;`roc_auc_score(event, p)`。

</details>

<details><summary>参考答案</summary>

```python
model = LogisticRegression(C=1e6)
# 用分层的 skf;要的是概率,不是 0 和 1
proba = cross_val_predict(model, tox[["rectum_total"]], event, cv=skf, method="predict_proba")
# [:, 1]: 每位患者结局为 1(出现副作用)的那一列
p = pd.Series(proba[:, 1], index=tox.index)
auc = roc_auc_score(event, p)
print(round(auc, 3))
```

</details>

</div>

<div class="step exercise">

**第 5 步 · 副作用: AUC 有多不确定**

0.68 来自 15 个事件。用 bootstrap 给它一个 95% 区间: `rng = np.random.default_rng(0)`,重抽 1000 次,每次有放回地抽 185 位(连同各自的概率和结局),算一次 AUC;取中间的 95%,存进 `low` 和 `high`。

骨架已经给出,把四处 `___` 填上。

```python
rng = np.random.default_rng(0)
y_np, p_np = event.to_numpy(), p.to_numpy()
aucs = []
# 重抽 1000 次
for _ in range(___):
    # 有放回地抽 185 个行号
    idx = rng.integers(0, len(y_np), len(y_np))
    # 抽到的这些患者的结局和概率: 用同一组行号
    aucs.append(roc_auc_score(___, ___))
# 中间的 95%
low, high = np.percentile(aucs, ___)
print(round(low, 3), round(high, 3))
```

```pycheck
need("low", "high")
if close(low, 0.523, 0.01) and close(high, 0.821, 0.01):
    ok("对了: 0.52 到 0.82。区间的下端几乎碰到 0.5: 这批数据不能排除'模型和抛硬币差不多',也不能排除'模型相当不错'。15 个事件就只能说到这个程度。")
fail(f"现在是 {low:.3f} 到 {high:.3f},应该是 0.523 到 0.821。检查: 随机数是不是 np.random.default_rng(0),是不是重抽 1000 次、每次抽 185 位(rng.integers(0, 185, 185)),百分位是不是 2.5 和 97.5。")
```

<details><summary>提示 1: 思路</summary>

第 10 课第 4 节的三步: 抽一批行号,在这一批里算 AUC,重复 1000 次,取 2.5% 和 97.5% 的位置。先把 `event` 和 `p` 变成 numpy 数组,用行号取起来方便。

</details>

<details><summary>提示 2: 用到的写法</summary>

`range(1000)`;`roc_auc_score(y_np[idx], p_np[idx])`;`np.percentile(aucs, [2.5, 97.5])`。

</details>

<details><summary>参考答案</summary>

```python
rng = np.random.default_rng(0)
y_np, p_np = event.to_numpy(), p.to_numpy()
aucs = []
# 重抽 1000 次
for _ in range(1000):
    # 有放回地抽 185 个行号
    idx = rng.integers(0, len(y_np), len(y_np))
    # 抽到的这些患者的结局和概率: 用同一组行号
    aucs.append(roc_auc_score(y_np[idx], p_np[idx]))
# 中间的 95%
low, high = np.percentile(aucs, [2.5, 97.5])
print(round(low, 3), round(high, 3))
```

</details>

</div>

<div class="step exercise">

**第 6 步 · 副作用: 概率准不准**

用准备好的 `calibration(event, p)` 算 O/E 和校准斜率,存进 `oe` 和 `cal_slope`(中间那个校准截距这里不用)。

```python
# 在这里写你的代码。结果存进 oe 和 cal_slope。
```

```pycheck
need("oe", "cal_slope")
if close(oe, 1.00, 0.01) and close(cal_slope, -0.003, 0.01):
    fail("cal_slope 取成了校准截距。calibration 交回三个数: O/E、截距、斜率,斜率是第三个。")
if close(oe, 1.00, 0.01) and close(cal_slope, 0.74, 0.01):
    ok("对了: O/E 1.00,总数上不偏: 模型预计的事件总数和实际的 15 个对得上。校准斜率 0.74,明显小于 1: 概率说得太散了,说 30% 的患者实际没那么高。事件少的模型常常这样,第 11 课讲过可以收缩。")
fail(f"现在是 {oe:.2f} 和 {cal_slope:.2f},应该是 1.00 和 0.74。")
```

<details><summary>提示 1: 思路</summary>

`calibration` 交回三个数,可以一次接住: `a, b, c = calibration(...)`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`oe, intercept, cal_slope = calibration(event, p)`。

</details>

<details><summary>参考答案</summary>

```python
oe, intercept, cal_slope = calibration(event, p)
print(round(oe, 2), round(intercept, 2), round(cal_slope, 2))
```

</details>

</div>

<div class="step exercise">

**第 7 步 · 副作用: 按模型做决定,值不值**

设想一种处理: 给风险高的患者安排更密的随访和预防性用药。科里认为风险到 10% 就值得这样做(阈值概率 0.10)。算两种做法的净获益: 按模型,概率达到 10% 的才处理(`nb_model`);对所有人都处理(`nb_all`)。

```python
# 在这里写你的代码。结果存进 nb_model 和 nb_all。
```

```pycheck
need("nb_model", "nb_all")
if close(nb_model, 0.034, 0.002):
    fail("这是阈值概率 0.05 的结果。题目是 0.10。")
if close(nb_model, 0.016, 0.002) and close(nb_all, -0.021, 0.002):
    ok("对了: 按模型 0.016,全都处理 −0.021,都不处理是 0。按模型处理 43 位,其中 7 位确实会出现副作用。净获益是正的,但很小: 每 100 位患者里,相当于多找对了 1.6 位而没有多余的处理。阈值换成 5%,按模型和全都处理一样(0.034 和 0.033);换成 20%,按模型成了负的。")
fail(f"现在是 {nb_model:.3f} 和 {nb_all:.3f},应该是 0.016 和 −0.021。")
```

<details><summary>提示 1: 思路</summary>

两个准备好的函数: `net_benefit(实际, 概率, 阈值概率)` 和 `net_benefit_all(实际, 阈值概率)`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`net_benefit(event, p, 0.10)`;`net_benefit_all(event, 0.10)`。

</details>

<details><summary>参考答案</summary>

```python
nb_model = net_benefit(event, p, 0.10)
nb_all = net_benefit_all(event, 0.10)
print(round(nb_model, 3), round(nb_all, 3))
```

</details>

</div>

<div class="step">

<div class="quiz">
<p><b>第 8 步 · 想一想</b>: 副作用模型的成绩单: AUC 0.68(95% 区间 0.52 到 0.82),O/E 1.00,校准斜率 0.74,阈值 10% 时净获益 0.016。写进论文的结论,哪一句站得住?</p>
<button class="quiz-opt" data-why="区间的下端是 0.52,几乎就是抛硬币。'能可靠地识别'说过头了;净获益也只在 10% 附近是正的。">模型能可靠地识别会出现直肠副作用的患者,可以用于临床决策</button>
<button class="quiz-opt" data-correct data-why="这句话把看到的和没把握的都说了: 有区分的迹象(AUC 0.68,置换检验也支持不是碰巧),但区间很宽;总数不偏,概率偏散;净获益在一个窄的阈值范围里略大于 0。15 个事件支持不了更强的说法,需要更多患者和外部验证。下一部分算一算,15 个事件到底撑得起多大的模型。">直肠总剂量对副作用有一定的区分能力,但估计很不精确;模型还不能用于决策,需要更大的队列</button>
<button class="quiz-opt" data-why="AUC 0.68、置换检验 P 约 0.015,说明剂量和副作用之间有关系。'没有预测价值'也说过头了: 这批数据同样排除不了一个相当不错的模型。">直肠剂量对副作用没有预测价值</button>
</div>

</div>

</div>

**这一部分做完了**: 两张成绩单。靶区剂量: 平均差 3.7 Gy,R² 0.24,校准斜率 0.89;用它挑患者,18 位里只找到 5 位。副作用: AUC 0.68(0.52 到 0.82),O/E 1.00,校准斜率 0.74,阈值 10% 时净获益 0.016。两个模型都"有点用",都还不够。下一部分学完[第 14 课](/stat-ml/14-writing-up/)以后做: 样本量,和怎样把这些如实写出来。
