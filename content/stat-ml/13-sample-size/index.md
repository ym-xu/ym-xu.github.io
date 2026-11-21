---
title: "Lesson 13 · 样本量 Sample Size"
summary: "多少患者才够建一个预测模型: 用模拟看患者少会怎样,Riley 的三条准则,以及样本不够时怎么如实写。"
weight: 130
toc: true
duration: "约 70 分钟,可以分两次学: 第 0–2 节,第 3–5 节"
prereq: "本课程[第 11 课](../11-calibration/)(校准斜率)和[第 6 课](../06-regularization/)(变量多了会怎样)"
questions:
  - "患者太少,模型会出什么毛病?"
  - '"每个变量 10 个事件"够不够?'
  - "建一个预测模型至少要多少患者,怎么算?"
  - "患者就是不够,论文该怎么写?"
objectives:
  - "用模拟说明样本量怎样影响校准斜率和 AUC"
  - "按 Riley 等的三条准则算出需要的患者数"
  - "反过来算手里的患者撑得起几个参数"
  - "在样本不足时如实地限定模型的用途和结论"
---

前面几课的结论后面总跟着一句: "144 位患者、34 个事件,估计不精确。"这一课正面回答: **建一个预测模型,到底要多少患者?**

这一课换一个结局,也是这门课最终要预测的那一类: **晚期阴道毒性**(Python 课里的随访表)。假定你手里有一个 200 位患者的队列,想用 8 个候选变量建一个预测晚期毒性的模型。够吗?

临床研究里算样本量,通常是为了一个检验: 两组的差别有多大,要多少患者才有 80% 的把握检出。预测模型的样本量问的是另一件事: **要多少患者,建出来的模型在新患者身上才不走样**。

## 0. 准备

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from cohort import make_cohort, make_followup
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def load(n=200, seed=22):
    """队列和随访表合在一起: 有随访、有壁厚测量的患者,一人一行。"""
    return make_cohort(n=n, seed=seed).merge(make_followup(n=n, seed=seed), on="id").dropna()

data, new = load(), load(5000, 99)
candidates = ["d2cc_5mm", "d90_5mm", "age", "grade", "diameter", "wall_mean", "height_cm", "weight_kg"]
print(f"研究队列: {len(data)} 位患者,{data['tox'].sum()} 位出现晚期毒性({data['tox'].mean():.0%})")
print(f"新患者: {len(new)} 位,{new['tox'].mean():.0%} 出现晚期毒性")
print(f"{len(candidates)} 个候选变量,每个变量 {data['tox'].sum() / len(candidates):.1f} 个事件")
```

入组 200 位,有随访、有壁厚测量的是 185 位,其中 41 位出现晚期毒性。8 个候选变量: 阴道 D2cc、靶区 D90、年龄、分级、施源器直径、壁厚、身高、体重。平均每个变量 5.1 个事件。

这是模拟队列,可以翻底牌: 造数据时,毒性的概率只取决于 5 mm 计划的阴道 D2cc,其余 7 个变量都没有作用。真实研究里你不知道这一点,所以才把它们都列为候选。

## 1. 患者少,会发生什么

不用公式,先做试验。模拟数据的好处是可以要多少患者有多少: 入组 100、200、400、800、1600 位,每种大小各造 20 个队列,每个队列建一个 8 变量的模型,拿到同一批新患者身上去量。

```python
def logit(prob):
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return np.log(prob / (1 - prob))

def calibration_slope(actual, prob):                      # 第 11 课: 1 是正好,小于 1 是说得太极端
    return sm.Logit(np.asarray(actual), sm.add_constant(logit(np.asarray(prob)))).fit(disp=0).params[1]

def try_size(n, columns, repeats=20):
    """造 repeats 个入组 n 位的队列,各建一个模型,在新患者身上量校准斜率和 AUC。"""
    slopes, aucs, events = [], [], []
    for k in range(repeats):
        d = load(n, 1000 + k)
        model = LogisticRegression(C=1e6, max_iter=5000).fit(d[columns], d["tox"])
        p_new = model.predict_proba(new[columns])[:, 1]
        slopes.append(calibration_slope(new["tox"], p_new))
        aucs.append(roc_auc_score(new["tox"], p_new))
        events.append(d["tox"].sum())
    return np.mean(events), np.median(slopes), np.min(slopes), np.max(slopes), np.mean(aucs)

for n in [100, 200, 400, 800, 1600]:
    events, slope, low, high, auc = try_size(n, candidates)
    print(f"入组 {n:4d} 位(约 {events:3.0f} 个事件): 校准斜率中位数 {slope:.2f}(最低 {low:.2f},最高 {high:.2f}),AUC 平均 {auc:.3f}")
```

这一段要建 100 个模型,等十几秒。结果有三层意思:

- **患者越少,概率越极端**。入组 200 位时,校准斜率的中位数只有 0.59: 模型说的高低差别,到新患者身上只兑现了六成。入组 1600 位(约 288 个事件),中位数才到 0.90。
- **患者越少,排序也越差**。AUC 从 0.614 升到 0.664。同样 8 个变量、同一种模型,区别只在建模的患者多少。
- **同样大小的队列,运气差别很大**。入组 200 位的 20 个队列里,校准斜率从 0.24 到 1.01 都有。你手里只有其中一个,而且不知道是哪一个。

原因在第 6 课讲过: 每个系数都要从数据里估计,患者少,估出来的系数里偶然的成分就多。8 个系数,就有 8 份偶然加在一起。对是/否的结局,提供信息的是**事件**: 200 位患者里只有 37 个事件,它们要撑起 8 个系数。

如果事先就知道只用阴道 D2cc 一个变量呢?

```python
for n in [100, 200, 400]:
    events, slope, low, high, auc = try_size(n, ["d2cc_5mm"])
    print(f"入组 {n:4d} 位,1 个变量: 校准斜率中位数 {slope:.2f}(最低 {low:.2f},最高 {high:.2f}),AUC 平均 {auc:.3f}")
```

一个变量时,校准斜率的中位数在 1 上下,不再系统性地偏低;AUC 是 0.670,比 8 个变量、入组 1600 位的模型还高。只是单个队列仍然晃得厉害(入组 100 位时从 0.61 到 2.30): 患者少,少估几个系数能去掉系统性的偏差,去不掉运气。

## 2. "每个变量 10 个事件"够不够

很多人学过一条经验法则: 每个候选变量至少要 10 个事件(events per variable,EPV ≥ 10)。按它算,8 个变量要 80 个事件;毒性的比例是 18%,大约要 440 位患者。

上面的试验里,入组 400 位(约 71 个事件)时校准斜率的中位数是 0.70。差不多每个变量 9 个事件,模型说的差别还是只兑现了七成。

这条法则的毛病是只看了一个数。需要多少患者,至少还取决于:

- **模型有多强**。信号越弱(变量和结局的关系越淡),同样多的偶然就占得越重,需要的患者越多。
- **事件的比例**。
- **你要保证什么**。是概率不走样,还是总体风险估得准,还是成绩的区间够窄? 目的不同,数也不同。

提出下面三条准则的作者们的说法是: 这类经验法则过于简化,每个参数需要多少事件要看具体情形。

## 3. Riley 的三条准则

<div class="frame">

**背景**: 需要一个按目的来算的办法: 先说清楚要保证模型的哪些性质,再算出各需要多少患者。

**目标**: 建出来的模型 (1) 不太过拟合,(2) 在建模患者身上的成绩和真实成绩差得不多,(3) 总体风险估得准。

**做法**: Riley 等人为三个目标各给了一个公式,需要的患者数取三个里最大的。要事先给出三样东西: 候选**参数**的个数、事件的比例、预期的模型强度(R²)。

</div>

**它在算什么**

先说三样输入:

- **参数的个数**: 不是变量的个数。一个连续变量是 1 个参数;分三类的变量是 2 个;加一个非线性项或交互项再加 1 个。而且要数**候选**的: 筛掉的变量也算,因为筛的过程用了数据。
- **事件的比例**: 从以前的研究或本单位的数据估计。
- **预期的 R²**(Cox–Snell R²,记作 R²cs): 是/否结局的一种 R²,量的是模型比"对谁都说总体比例"好了多少。它的上限不是 1,而由事件比例决定。它从以前类似模型的报告里来;拿不准时往小里取(模型越弱,要的患者越多,这样算出来偏保守)。

三条准则:

1. **过拟合要小**: 让预期的收缩系数不低于 0.9,也就是校准斜率大约在 0.9 以上。需要的患者数 = 参数个数 ÷ ((0.9 − 1) × ln(1 − R²cs ÷ 0.9))。参数个数加倍,患者数加倍;R²cs 越小,患者数越多。
2. **成绩不虚高**: 建模患者身上的 R² 和校正以后的 R² 相差不超过 0.05。公式和第 1 条同形,只是收缩系数换了一个算法。
3. **总体风险估得准**: 事件比例的误差在 ±0.05 以内。需要的患者数 = (1.96 ÷ 0.05)² × 比例 × (1 − 比例)。这一条和参数个数无关。

先得到 R²cs。在这个模拟的世界里,可以用那 4624 位新患者当作"以前的一项大研究":

```python
big = sm.Logit(new["tox"], sm.add_constant(new[["d2cc_5mm"]])).fit(disp=0)
r2_cs = 1 - np.exp(-big.llr / len(new))                   # llr: 模型比"只有截距"好了多少(似然比)
phi = 0.18
max_r2 = 1 - np.exp(2 * (phi * np.log(phi) + (1 - phi) * np.log(1 - phi)))
print(f"R²cs = {r2_cs:.3f};事件比例 {phi} 时,R²cs 的上限是 {max_r2:.2f}")
```

R²cs 是 0.054,只有上限(0.61)的一成不到: 晚期毒性的模型通常就是这么弱(AUC 0.67)。往小里取,用 0.05。

```python
def riley_binary(parameters, prevalence, r2_cs, shrinkage=0.9):
    """Riley 等的三条准则各要求多少位患者(是/否的结局)。"""
    max_r2 = 1 - np.exp(2 * (prevalence * np.log(prevalence) + (1 - prevalence) * np.log(1 - prevalence)))
    n1 = parameters / ((shrinkage - 1) * np.log(1 - r2_cs / shrinkage))          # 准则 1
    s2 = r2_cs / (r2_cs + 0.05 * max_r2)
    n2 = parameters / ((s2 - 1) * np.log(1 - r2_cs / s2))                        # 准则 2
    n3 = (1.96 / 0.05) ** 2 * prevalence * (1 - prevalence)                      # 准则 3
    return [int(np.ceil(v)) for v in (n1, n2, n3)]

for parameters in [8, 3, 1]:
    need = riley_binary(parameters, 0.18, 0.05)
    print(f"{parameters} 个参数: 三条准则各要 {need} 位,取最大的 {max(need)} 位(约 {max(need) * 0.18:.0f} 个事件)")
```

- 8 个参数: 1400 位患者,约 252 个事件,每个参数 31.5 个事件。第 1 节的试验里,入组 1600 位(有随访的约 1470 位)时校准斜率的中位数是 0.90,和"不低于 0.9"的目标对得上。
- 3 个参数: 525 位。
- 1 个参数: 227 位。这时决定数目的是准则 3: 哪怕只有一个变量,要把总体风险估到 ±5 个百分点以内,也得有两百多位患者。

你手里有 185 位。

<div class="demo">

**设定**: 一个计算器,算的就是上面的三条准则。三个滑块: 候选参数的个数、事件的比例、预期的 R²cs。三个条是三条准则各要求的患者数,深色的那条是要求最高的。

**你做什么**: 起始是 8 个参数、18%、0.05。先把参数个数从 8 拖到 16,再拖到 1。然后回到 8,把 R²cs 从 0.05 拖到 0.15。

**你会看到**: 起始需要 1400 位。参数个数拖到 16,变成 2800 位: 正好加倍。拖到 1,是 227 位,要求最高的换成了准则 3。回到 8 个参数,R²cs 拖到 0.15,只要 439 位。

**这说明**: 需要的患者数和候选参数的个数成正比;模型越弱(R²cs 越小),需要的越多。弱信号、多变量的研究最费患者,而预测毒性的研究多半正是这一种。

<div class="widget" data-widget="samplesize">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

<div class="exercise">

**练习 1 · 轮到你: 另一项研究要多少患者**

一项研究打算用 5 个候选参数预测一个比例为 30% 的结局,以前的研究提示 R²cs 大约 0.10。按三条准则各算出需要的患者数(`need_ex`),并取最大的(`n_ex`)。

```python
need_ex = riley_binary(___, ___, 0.10)
n_ex = max(need_ex)
print(f"三条准则各要 {need_ex} 位,至少 {n_ex} 位(约 {n_ex * 0.3:.0f} 个事件)")
```

```pycheck
need("need_ex", "n_ex")
if list(need_ex) == [1400, 252, 227]:
    fail("这是课文里的例子。这道题是 5 个参数、比例 0.3、R²cs 0.10。")
if list(need_ex) == [425, 132, 323] and n_ex == 425:
    ok("对了: 三条准则各要 425、132、323 位,至少 425 位(约 128 个事件,每个参数 25.5 个事件)。按'每个变量 10 个事件'只要 50 个事件、167 位患者: 少了六成。")
fail(f"现在是 {list(need_ex)},应该是 [425, 132, 323]。第一个空是参数个数 5,第二个空是事件的比例 0.3。")
```

<details><summary>提示</summary>

`riley_binary(参数个数, 事件的比例, R²cs)`: 第一个空是 `5`,第二个空是 `0.3`。

</details>

<details><summary>参考答案</summary>

```python
need_ex = riley_binary(5, 0.3, 0.10)
n_ex = max(need_ex)
print(f"三条准则各要 {need_ex} 位,至少 {n_ex} 位(约 {n_ex * 0.3:.0f} 个事件)")
```

</details>

</div>

## 4. 反过来算: 手里的患者撑得起几个参数

回顾性研究里,患者数往往是定死的。这时把准则 1 倒过来用: 给定患者数,最多能有几个候选参数?

```python
n_have = len(data)
p_max = n_have * (0.9 - 1) * np.log(1 - 0.05 / 0.9)       # 准则 1 的公式,解出参数个数
print(f"{n_have} 位患者,按准则 1 最多 {p_max:.1f} 个参数")
```

185 位患者,最多 1.1 个参数: 一个。8 个候选变量的计划,从一开始就超出了这批患者能支撑的范围。

这个算法最有用的时候是在**看数据之前**: 先算出能有几个参数,再按临床知识把候选变量减到这个数以内。看了数据再减(比如留下 P 值小的),省下的参数是假的: 筛选本身用掉了信息(第 5 课)。

<div class="exercise">

**练习 2 · 轮到你: 300 位患者撑得起几个参数**

另一个中心有 300 位患者,预期的 R²cs 是 0.10。按准则 1,最多可以有几个候选参数? 算出来存进 `p_max_ex`,向下取整的结果存进 `p_use`。

```python
p_max_ex = ___ * (0.9 - 1) * np.log(1 - ___ / 0.9)
p_use = int(np.floor(p_max_ex))
print(f"最多 {p_max_ex:.2f} 个参数: 用 {p_use} 个")
```

```pycheck
need("p_max_ex", "p_use")
if close(p_max_ex, 1.057, 0.01) or close(p_max_ex, 1.145, 0.01):
    fail("这是课文里 185 位患者的结果。这道题是 300 位、R²cs 0.10。")
if close(p_max_ex, 3.533, 0.01) and p_use == 3:
    ok("对了: 3.53,向下取整是 3 个参数。向下取,不向上取: 多出来的半个参数,代价是过拟合超出了事先说好的限度。")
fail(f"现在是 {p_max_ex:.3f},应该是 3.533。第一个空是患者数 300,第二个空是 R²cs 0.10。")
```

<details><summary>提示</summary>

第一个空是患者数 `300`;第二个空是预期的 R²cs `0.10`。

</details>

<details><summary>参考答案</summary>

```python
p_max_ex = 300 * (0.9 - 1) * np.log(1 - 0.10 / 0.9)
p_use = int(np.floor(p_max_ex))
print(f"最多 {p_max_ex:.2f} 个参数: 用 {p_use} 个")
```

</details>

</div>

## 5. 患者不够,怎么办,怎么写

先看手里这一个队列建出来的模型,在新患者身上是什么样:

```python
for name, columns in [("8 个候选变量", candidates), ("事先定的 1 个变量", ["d2cc_5mm"])]:
    model = LogisticRegression(C=1e6, max_iter=5000).fit(data[columns], data["tox"])
    p_new = model.predict_proba(new[columns])[:, 1]
    print(f"{name}: AUC {roc_auc_score(new['tox'], p_new):.3f},校准斜率 {calibration_slope(new['tox'], p_new):.2f},O/E {new['tox'].sum() / p_new.sum():.2f}")
```

两件事值得看。

第一,8 个变量的模型,校准斜率是 0.98: 这一个队列的运气不算差。可是第 1 节里同样大小的 20 个队列,斜率从 0.24 到 1.01 都有。真实研究里没有 4624 位新患者给你对答案,你不知道自己抽到的是哪一个。样本量不够,说的就是这个: 不是模型一定坏,而是**你没有办法知道它坏不坏**。

第二,两个模型的 O/E 都是 0.84: 概率整体偏高。这批患者里 22% 出现了毒性,总体是 18%。这正是准则 3 管的事: 185 位患者估出来的总体风险,偏差几个百分点是常事。

**能做的**

- **减少候选参数**: 按临床知识事先定下少数几个变量,数目不超过第 4 节算出来的。这是最有效的一条。上面事先定 1 个变量的模型,AUC 比 8 个变量的高。
- **收缩**(第 6、11 课): 不得不多放变量时,用带约束的模型。它减轻过拟合,但患者太少时连约束的强度也选不稳,不能指望它补上缺的患者。
- **不要做的**: 不要按单因素 P 值筛变量,不要把连续变量切成两段,不要为"类别不平衡"加权(第 11 课)。这些都在浪费或者歪曲本来就不多的信息。

**如实地写**

- 写出按准则**需要**多少患者、**实际**多少、差在哪一条。差在准则 1,就说明概率可能过于极端;差在准则 3,就说明总体风险可能有偏。
- 所有成绩带区间(第 10 课);区间很宽,就照实说很宽。
- 把研究定位成**模型开发和内部验证**,结论写成"需要在更大的、独立的队列中验证之后才能用于临床",而不是"该模型可用于指导治疗"。
- 写出模型的全部系数,让别人能在自己的数据上验证。

<div class="quiz">
<p><b>想一想</b>: 一篇论文: 96 位患者、19 个事件,从 22 个候选变量里用 LASSO 选出 6 个,交叉验证的 AUC 0.84。作者写"样本量较小是本研究的局限"。这句话够不够?</p>
<button class="quiz-opt" data-why="承认局限是对的,但这句话没有告诉读者局限有多大、影响的是哪些结论。">够了: 已经承认了局限</button>
<button class="quiz-opt" data-correct data-why="19 个事件对 22 个候选参数,每个参数不到 1 个事件。按准则算,需要的患者数是现有的很多倍。如实的写法是: 给出需要的患者数和实际的差距,说明选出的 6 个变量和 0.84 都很可能换一批患者就变,把结论限定为'提出假说',而不只是一句套话。">不够: 应该写出差多少、哪些结论因此不可靠,并相应地收窄结论</button>
<button class="quiz-opt" data-why="交叉验证防的是'在建模患者身上评成绩'。它防不了'患者太少、选出的变量和成绩都不稳定'。">够了: 用了交叉验证,样本量小的影响已经消除</button>
</div>

<div class="exercise">

**练习 3 · 轮到你: 3 个变量,入组 400 位**

折中一下: 事先定 3 个变量(阴道 D2cc、年龄、壁厚),入组 400 位。用 `try_size` 看 20 个队列的校准斜率中位数(`slope3`)和最低值(`low3`)。

```python
three = ["d2cc_5mm", "age", "wall_mean"]
events3, slope3, low3, high3, auc3 = try_size(___, ___)
print(f"约 {events3:.0f} 个事件: 校准斜率中位数 {slope3:.2f}(最低 {low3:.2f},最高 {high3:.2f}),AUC 平均 {auc3:.3f}")
```

```pycheck
need("slope3", "low3")
if close(slope3, 0.698, 0.004):
    fail("这是 8 个候选变量的结果。第二个空是三个变量的名单 three。")
if close(slope3, 0.833, 0.004) and close(low3, 0.639, 0.004):
    ok("对了: 中位数 0.83,最低 0.64。同样入组 400 位,8 个变量时中位数是 0.70。按准则,3 个参数要 525 位: 入组 400 位(有随访的约 370 位)还差一些,所以中位数还没到 0.9,最差的队列只有 0.64。")
fail(f"现在中位数是 {slope3:.3f}、最低是 {low3:.3f},应该是 0.833 和 0.639。第一个空是 400,第二个空是 three。")
```

<details><summary>提示</summary>

`try_size(入组人数, 变量名单)`: 第一个空是 `400`,第二个空是 `three`。

</details>

<details><summary>参考答案</summary>

```python
three = ["d2cc_5mm", "age", "wall_mean"]
events3, slope3, low3, high3, auc3 = try_size(400, three)
print(f"约 {events3:.0f} 个事件: 校准斜率中位数 {slope3:.2f}(最低 {low3:.2f},最高 {high3:.2f}),AUC 平均 {auc3:.3f}")
```

</details>

</div>

## 小结

- 患者(准确地说是**事件**)太少,模型的概率过于极端、排序变差,而且同样大小的研究结果相差很大。
- "每个变量 10 个事件"只看了一个数,常常不够。
- **Riley 的三条准则**: 过拟合要小、成绩不虚高、总体风险估得准。需要的患者数和**候选参数的个数**成正比,模型越弱需要越多。
- 患者数定死时,倒过来算**撑得起几个参数**,在看数据之前把候选变量减到这个数以内。
- 样本不够时: 少而事先定好的变量、收缩、带区间的成绩、收窄的结论。写出需要多少、实际多少。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 候选参数 | candidate predictor parameters | 建模时考虑过的所有系数,筛掉的也算 |
| 每个参数的事件数 | events per parameter (EPP, EPV) | 事件数 ÷ 候选参数个数 |
| 收缩系数 | shrinkage factor | 预期的校准斜率;0.9 表示过拟合约 10% |
| Cox–Snell R² | Cox–Snell R² | 是/否结局的一种 R²,上限由事件比例决定 |
| 内部验证 | internal validation | 用建模的同一批患者(交叉验证、bootstrap)估计成绩 |
| 外部验证 | external validation | 在另一批独立的患者身上检验模型 |

## 这一课没讲什么

- **连续结局的样本量**: 同一批作者有对应的准则(收缩系数、R² 的差、残差标准差和截距的精度),R 包 `pmsampsize` 都能算。
- **随访时间不等、有删失的结局**: 也有对应的公式,输入换成事件率和随访时间。
- **验证研究的样本量**: 检验一个已有的模型,需要的患者数按"成绩的区间够窄"来算,是另一套公式。
- **树模型等更灵活的模型**: 这些准则是为回归模型推的。更灵活的模型通常需要更多患者,没有同样简单的公式,可以用第 1 节那样的模拟来估计。
- **从 C 统计量换算 R²cs**: 以前的研究只报告了 AUC 时,`pmsampsize` 提供了换算的办法。

## 写进论文 Writing it up

<div class="paper">

方法部分可以这样写(数字换成你自己的):

> The minimum sample size was calculated with the criteria of Riley et al. for a binary outcome. With 8 candidate predictor parameters, an anticipated outcome proportion of 0.18 and an anticipated Cox–Snell R² of 0.05, at least 1400 patients (252 events) are required to target a shrinkage factor of 0.9. The available cohort comprised 185 patients (41 events), which supports about one predictor parameter. We therefore prespecified a single predictor on clinical grounds, and regard the study as model development with internal validation only.

**审稿人可能问**: 样本量是怎么确定的?

**怎么回答**: 回顾性研究的患者数是现成的,但仍然要算: 按准则需要多少,现有多少,因此候选参数限制在几个。没有这一段,审稿人会默认作者没有考虑过。

**审稿人可能问**: 每个变量有 10 个事件,为什么还说不够?

**怎么回答**: 给出按三条准则算的结果,并说明所需的事件数取决于模型强度和事件比例。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| `riley_binary` 算出负数或 `nan` | R²cs 大于收缩系数 0.9 或大于它的上限。检查 R²cs: 是/否结局的 R²cs 通常远小于 0.5。 |
| `PerfectSeparationWarning` | 小队列里某个变量把事件和非事件完全分开了。这本身就是患者太少的信号。 |
| 第 1 节那一段很慢 | 要造 100 个队列、建 100 个模型。等十几秒。 |
| 算出来的患者数比想的多得多 | 多半是对的。先检查参数个数有没有把分类变量的每一类、每个交互项都数进去。 |
| 结果和网页上写的不一样 | `load(n, 1000 + k)` 里的种子要保持原样;代码块要从上往下依次运行。 |

## 延伸阅读 Further reading

- Riley RD, Ensor J, Snell KIE, 等. Calculating the sample size required for developing a clinical prediction model. *BMJ* 2020;368:m441。面向临床研究者的讲解,有逐步的算例。
- Riley RD, Snell KIE, Ensor J, 等. Minimum sample size for developing a multivariable prediction model: PART II - binary and time-to-event outcomes. *Stat Med* 2019;38(7):1276–1296。三条准则的出处。这一课的函数照 R 包 `pmsampsize` 的算法写。

**下一课**: 第 14 课把前面十三课的内容串起来: 一篇预测模型的论文该报告什么(TRIPOD+AI),审稿人常问什么,并用模拟队列从头到尾走一遍。
