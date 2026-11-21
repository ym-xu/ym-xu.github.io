---
title: "练习 5 · 样本量和报告"
summary: "15 个事件撑得起几个参数;最终模型写全;让结果段落从代码里来;该写哪些局限。"
weight: 50
after: 14
toc: false
duration: "约 30 分钟"
prereq: "[统计机器学习课第 13–14 课](/stat-ml/);[练习 4](../04-good-enough/)"
questions:
  - "185 位患者、15 个事件,撑得起多大的模型?"
  - "最终的模型怎么写,别人才能照着用?"
  - "这份小研究的结论和局限,怎样如实写出来?"
objectives:
  - "Cox–Snell R² 和 Riley 的三条准则"
  - "用 statsmodels 写出最终模型的系数、OR 和区间"
  - "让结果段落里的数字从代码里来"
---

练习 4 的成绩单上,副作用模型的 AUC 区间从 0.52 到 0.82。这一部分先回答为什么这么宽,再把整件事如实写出来。

先运行这一块。它把练习 4 里副作用模型的成绩重新算了一遍,放在字典 `r` 里;`riley_binary` 是第 13 课的函数。

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from cervix import load_cervix, make_cervix_followup
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

full = load_cervix()                                            # 200 位患者
tox = full.merge(make_cervix_followup(), on="id")               # 有随访的 185 位
event = tox["rect_tox"]

def riley_binary(parameters, prevalence, r2_cs, shrinkage=0.9):
    """Riley 等的三条准则各要求多少位患者(是/否的结局)。"""
    max_r2 = 1 - np.exp(2 * (prevalence * np.log(prevalence) + (1 - prevalence) * np.log(1 - prevalence)))
    n1 = parameters / ((shrinkage - 1) * np.log(1 - r2_cs / shrinkage))          # 准则 1: 过拟合要小
    s2 = r2_cs / (r2_cs + 0.05 * max_r2)
    n2 = parameters / ((s2 - 1) * np.log(1 - r2_cs / s2))                        # 准则 2: 表面成绩不要太乐观
    n3 = (1.96 / 0.05) ** 2 * prevalence * (1 - prevalence)                      # 准则 3: 总体风险估得准
    return [int(np.ceil(v)) for v in (n1, n2, n3)]

# 练习 4 的成绩: out-of-fold 概率、AUC 和 bootstrap 区间、O/E、校准斜率
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
p = cross_val_predict(LogisticRegression(C=1e6), tox[["rectum_total"]], event, cv=skf, method="predict_proba")[:, 1]
rng = np.random.default_rng(0)
y_np = event.to_numpy()
aucs = []
for _ in range(1000):
    idx = rng.integers(0, len(y_np), len(y_np))
    aucs.append(roc_auc_score(y_np[idx], p[idx]))
lp = np.log(p / (1 - p))
r = {"AUC": roc_auc_score(event, p), "low": np.percentile(aucs, 2.5), "high": np.percentile(aucs, 97.5),
     "O/E": event.sum() / p.sum(), "slope": sm.Logit(y_np, sm.add_constant(lp)).fit(disp=0).params[1]}

print(len(full), "位患者,", len(tox), "位有随访,", int(event.sum()), "位出现副作用")
print({k: round(float(v), 2) for k, v in r.items()})
```

<div class="steps" data-steps>

<div class="step exercise">

**第 1 步 · 这个模型有多强: R²cs**

Riley 的准则要一个"预期的 R²cs"。手头没有以前的研究,先用自己这批患者算一个: 用 statsmodels 拟合副作用对直肠总剂量的逻辑回归(存进 `fit`),再按第 13 课的式子算 Cox–Snell R²,存进 `r2_cs`。

```python
# 在这里写你的代码。模型存进 fit,结果存进 r2_cs。
```

```pycheck
need("fit", "r2_cs")
if close(r2_cs, 0.0723, 0.003):
    fail("这是 McFadden 的 pseudo R²(prsquared),不是 Cox–Snell R²。按第 13 课: 1 − exp(−llr / 患者数)。")
if close(r2_cs, 0.040, 0.002):
    ok("对了: R²cs 是 0.040。事件比例 8% 时它的上限是 0.43,这个模型只到上限的一成: 直肠剂量和副作用有关系,但关系很弱。而且 0.040 是在建模患者身上算的,偏乐观;真实的只会更小。")
fail(f"现在是 {r2_cs:.4f},应该是 0.040。式子是 1 - np.exp(-fit.llr / len(tox))。")
```

<details><summary>提示 1: 思路</summary>

`sm.Logit(结局, sm.add_constant(变量的表)).fit(disp=0)`。拟合结果的 `llr` 是模型比"只有截距"好了多少;R²cs = 1 − exp(−llr ÷ 患者数)。

</details>

<details><summary>提示 2: 用到的写法</summary>

`fit = sm.Logit(event, sm.add_constant(tox[["rectum_total"]])).fit(disp=0)`;`1 - np.exp(-fit.llr / len(tox))`。

</details>

<details><summary>参考答案</summary>

```python
fit = sm.Logit(event, sm.add_constant(tox[["rectum_total"]])).fit(disp=0)
r2_cs = 1 - np.exp(-fit.llr / len(tox))
print(round(r2_cs, 3))
```

</details>

</div>

<div class="step exercise">

**第 2 步 · 一个参数,需要多少患者**

模型只有 1 个参数(直肠总剂量)。用 `riley_binary` 算三条准则各要多少位患者: 参数个数 1,事件比例用 `event.mean()`,R²cs 用上一步的 `r2_cs`。三个数存进 `sizes`,其中最大的存进 `n_needed`。

```python
# 在这里写你的代码。结果存进 sizes 和 n_needed。
```

```pycheck
need("sizes", "n_needed")
if int(n_needed) == 443:
    fail("这是 2 个参数的结果。这个模型只有 1 个参数。")
if list(sizes) == [222, 46, 115] and int(n_needed) == 222:
    ok("对了: 三条准则各要 222、46、115 位,取最大的: 222 位。现在有 185 位。也就是说,连只有一个变量的模型,这批患者都还差一点;每多一个参数,要再加 221 位。练习 4 里 AUC 的区间那么宽、校准斜率只有 0.74,原因就在这里。")
fail(f"现在是 {list(sizes)} 和 {n_needed},应该是 [222, 46, 115] 和 222。")
```

<details><summary>提示 1: 思路</summary>

`riley_binary(参数个数, 事件的比例, R²cs)` 交回三个数的列表;最大的那个用 `max`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`sizes = riley_binary(1, event.mean(), r2_cs)`;`n_needed = max(sizes)`。

</details>

<details><summary>参考答案</summary>

```python
sizes = riley_binary(1, event.mean(), r2_cs)
n_needed = max(sizes)
print(sizes, n_needed)
```

</details>

</div>

<div class="step exercise">

**第 3 步 · 反过来: 185 位患者撑得起几个参数**

把准则 1 反过来用(第 13 课): 最多的参数个数 = 患者数 × (0.9 − 1) × ln(1 − R²cs ÷ 0.9)。对这 185 位患者算出来,存进 `p_max`。

```python
# 在这里写你的代码。结果存进 p_max。
```

```pycheck
need("p_max")
if close(p_max, -0.836, 0.01):
    fail("符号反了: (0.9 − 1) 是负的,ln(...) 也是负的,乘起来是正的。检查括号。")
if close(p_max, 0.836, 0.01):
    ok("对了: 0.84 个参数,不到 1 个。在这批患者上,往副作用的模型里再加年龄、肿瘤体积、膀胱剂量,多半只会让它在新患者身上更差(第 13 课): 多出来的系数都是拿 15 个事件硬估的。")
fail(f"现在是 {p_max:.3f},应该是 0.836。")
```

<details><summary>提示 1: 思路</summary>

照式子写: 患者数是 `len(tox)`,`ln` 是 `np.log`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`len(tox) * (0.9 - 1) * np.log(1 - r2_cs / 0.9)`。

</details>

<details><summary>参考答案</summary>

```python
p_max = len(tox) * (0.9 - 1) * np.log(1 - r2_cs / 0.9)
print(round(p_max, 2))
```

</details>

</div>

<div class="step">

<div class="quiz">
<p><b>第 4 步 · 想一想</b>: 患者就这么多,短期内不会再有。下面哪种做法是对的?</p>
<button class="quiz-opt" data-why="换一个更灵活的模型,要学的东西更多,需要的患者也更多。样本不够时,它只会更糟。">换成随机森林或梯度提升,它们能从少量数据里学到更多</button>
<button class="quiz-opt" data-correct data-why="这是第 13 课的结论: 样本不够不等于不能做,而是要少做、如实写。事先只定一个最有依据的变量;报告区间、校准斜率和三条准则算出来的数;把研究称为探索性的;说明模型在用于决策之前需要更大的队列或外部验证。">只保留事先定好的一个变量,把区间、校准和样本量的计算如实报告,并说明这是探索性的</button>
<button class="quiz-opt" data-why="多试几个变量、挑成绩最好的,正是练习 2 里的泄漏: 挑到的是运气最好的那一个。样本越小,运气的成分越大。">把候选变量都试一遍,报告 AUC 最高的那个模型</button>
</div>

</div>

<div class="step exercise">

**第 5 步 · 最终的模型,写全**

论文里要给出完整的模型。第 1 步的 `fit` 里有截距和系数,还有系数的 95% 区间(`fit.conf_int()`)。算出直肠总剂量每高 5 Gy 的 OR(`or5`)和它的 95% 区间(`or5_low`、`or5_high`)。

```python
# 在这里写你的代码。结果存进 or5、or5_low 和 or5_high。
```

```pycheck
need("or5", "or5_low", "or5_high")
if close(or5, 1.147, 0.005):
    fail("这是每高 1 Gy 的 OR。每高 5 Gy: 系数(和它的区间两端)先乘以 5,再取 exp。")
if close(or5, 1.98, 0.02) and close(or5_low, 1.21, 0.02) and close(or5_high, 3.26, 0.03):
    ok("对了: 每高 5 Gy,OR 1.98(95% 区间 1.21 到 3.26)。完整的模型是 log-odds = −11.51 + 0.137 × 直肠总剂量(Gy)。有了截距和系数,别的中心才能拿自己的患者来验证它。")
fail(f"现在是 {or5:.2f}({or5_low:.2f} 到 {or5_high:.2f}),应该是 1.98(1.21 到 3.26)。")
```

<details><summary>提示 1: 思路</summary>

系数在 `fit.params["rectum_total"]`;区间在 `fit.conf_int()` 里,是一张表,行是变量名,两列是下限和上限。乘以 5 再取 `np.exp`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`np.exp(5 * fit.params["rectum_total"])`;`ci = fit.conf_int()`,`np.exp(5 * ci.loc["rectum_total", 0])` 和 `np.exp(5 * ci.loc["rectum_total", 1])`。

</details>

<details><summary>参考答案</summary>

```python
ci = fit.conf_int()
or5 = np.exp(5 * fit.params["rectum_total"])
or5_low = np.exp(5 * ci.loc["rectum_total", 0])
or5_high = np.exp(5 * ci.loc["rectum_total", 1])
print(f"log-odds = {fit.params['const']:.2f} + {fit.params['rectum_total']:.3f} × 直肠总剂量")
print(f"每高 5 Gy: OR {or5:.2f} (95% CI {or5_low:.2f} to {or5_high:.2f})")
```

</details>

</div>

<div class="step exercise">

**第 6 步 · 让结果段落从代码里来**

论文里的数字不要手抄。下面给了结果段落的第一句;接着写第二句,存进 `text`: 报告交叉验证的 AUC 和它的 95% 区间、O/E、校准斜率,数字都从字典 `r` 里取,保留两位小数。

```python
text = (
    f"Of {len(full)} patients, {len(tox)} had follow-up and {event.sum()} ({event.mean():.0%}) developed late rectal toxicity. "
    # 在这里接着写第二句(用 f"..."),数字从 r 里取。
)
print(text)
```

```pycheck
need("text")
t = str(text)
if "{" in t:
    fail("text 里还有没算出来的花括号: 字符串前面要有 f,写成 f\"...\"。")
missing = [name for name, s in [("AUC 0.68", "0.68"), ("区间的下限 0.52", "0.52"), ("区间的上限 0.82", "0.82"), ("O/E 1.00", "1.00"), ("校准斜率 0.74", "0.74")] if s not in t]
if missing:
    fail("第二句里还缺: " + "、".join(missing) + "。从 r 里取,例如 {r['AUC']:.2f}。")
if "185" in t and "15 (8%)" in t:
    ok("对了。这一段里每个数都是代码算出来的: 数据或分析改了,重跑一遍,段落跟着改,不会出现正文和表格对不上的情况。")
fail("第一句被改动了: 保留给出的那一行,在它后面接着写。")
```

<details><summary>提示 1: 思路</summary>

圆括号里可以连着写几个 `f"..."`,Python 会把它们接成一个字符串。花括号里放要填的数,冒号后面是格式: `{r['AUC']:.2f}`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`f"... a cross-validated AUC of {r['AUC']:.2f} (95% CI {r['low']:.2f} to {r['high']:.2f}), ..."`;O/E 是 `r['O/E']`,校准斜率是 `r['slope']`。

</details>

<details><summary>参考答案</summary>

```python
text = (
    f"Of {len(full)} patients, {len(tox)} had follow-up and {event.sum()} ({event.mean():.0%}) developed late rectal toxicity. "
    f"A logistic model with total rectal D2cc as the only prespecified predictor had a cross-validated AUC of {r['AUC']:.2f} "
    f"(95% CI {r['low']:.2f} to {r['high']:.2f}), an observed-to-expected ratio of {r['O/E']:.2f} "
    f"and a calibration slope of {r['slope']:.2f}."
)
print(text)
```

</details>

</div>

<div class="step">

<div class="quiz">
<p><b>第 7 步 · 想一想</b>: 对照 TRIPOD+AI 写"局限"这一段。这份小研究里,哪一条是读者最需要知道、而前面还没有交代的?</p>
<button class="quiz-opt" data-why="样本量小已经在第 2、3 步算过,结果段落里的宽区间也体现了。它要写,但不是'还没有交代的'。">事件只有 15 个,估计不精确</button>
<button class="quiz-opt" data-correct data-why="185 位患者的随访从 9 个月到 90 个月不等,47 位不到 2 年。把结局当成'有没有发生',等于默认每位患者都被观察了同样长的时间: 随访短的患者,副作用可能只是还没出现。这会低估风险,也会让模型把'随访长'和'风险高'混在一起。这门课没有讲处理它的方法(生存分析),所以更要写明。">随访时间长短不一,却把结局当成了"有没有发生": 随访短的患者可能只是还没出现副作用</button>
<button class="quiz-opt" data-why="只用一个变量是事先的决定,而且是样本量允许的上限,不算局限;要写的是为什么这样定。">模型里只有一个变量</button>
</div>

</div>

</div>

**五部分都做完了。** 回头看,你对同一份审计数据做了一次完整的预测模型研究:

- **靶区剂量**: 带交互项的直线(体积、施源器、年龄、体积 × 施源器),交叉验证平均差 3.7 Gy,R² 0.24。它说明了一件临床上有用的事: 肿瘤大的时候加针才要紧;但它还不足以把剂量不够的患者一个个挑出来。
- **晚期直肠副作用**: 直肠总剂量每高 5 Gy,OR 1.98(1.21 到 3.26);AUC 0.68(0.52 到 0.82),校准斜率 0.74。185 位患者、15 个事件,连一个参数都还差一点。
- **怎么写**: 事先定变量,所有步骤放进交叉验证,报告区分度、校准和净获益,算样本量,把局限写明,数字从代码里来。

换成你自己的数据,要做的事是同一串。
