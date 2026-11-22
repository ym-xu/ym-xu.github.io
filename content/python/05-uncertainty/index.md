---
title: "Lesson 5 · 描述和不确定性 Description and Uncertainty"
summary: "均值还是中位数;换一批患者结果会变;95% 区间的意思;bootstrap 怎么算。"
weight: 50
toc: true
duration: "约 80 分钟,可以分两次学: 第 1–3 节,第 4–6 节"
prereq: "[第 2 课](../02-pandas/)和[第 4 课](../04-plots/)"
questions:
  - "一组数据,该报均值还是中位数?"
  - "从 200 位患者算出来的结果,换一批患者会差多少?"
  - "论文里的 95% CI 到底是什么意思,怎么算?"
objectives:
  - "根据数据的形状,选择报均值(标准差)还是中位数(四分位间距)"
  - "说明为什么同一个研究重做一次,结果会不一样"
  - "正确解释 95% 置信区间,指出常见的误读"
  - "自己写几行代码,用 bootstrap 算中位数和比例的 95% 区间"
  - "说明样本量怎样影响区间的宽度"
---

前几课算出了不少数: 换成 Volume 后,阴道 D2cc 变化的中位数是 −2.0 Gy;148 位患者里有 36 位降到了 68 Gy 及以下。每个数都是从**这 200 位**患者算出来的。读者关心的却不是这 200 位,而是以后的患者: 换一家医院、换一批患者,这个 −2.0 会变成多少?

这一课回答两个问题。第一,用哪个数来概括一组数据: 均值还是中位数。第二,怎样说清楚一个结果有多不确定: 论文里每个结果后面跟着的"95% CI"是什么,怎么算出来。

> 🔬 这一课的 bootstrap,就是我们自己的研究里给每个中位数配 95% 区间时用的办法。

**练习怎么做**: 讲解里算的是 **Volume** 的 Δ,练习换成 **Graphical** 的 Δ。

## 0. 准备

```python
import numpy as np
import pandas as pd
from cohort import make_cohort, make_followup

df = make_cohort()
df["delta_vol"] = df["d2cc_vol"] - df["d2cc_5mm"]
df["delta_gra"] = df["d2cc_gra"] - df["d2cc_5mm"]
fu = make_followup()
print(df.shape, fu.shape)
```

## 1. 均值还是中位数

概括一组数的"中间",有两个常用的量:

- **均值**(mean): 全部加起来,除以个数。
- **中位数**(median): 从小到大排好,正中间的那一个。一半患者比它小,一半比它大。

数据左右对称时,两者差不多。数据有长尾时,它们会分开。看随访时间:

```python
months = fu["fu_months"]
print(f"均值 {months.mean():.1f} 个月,中位数 {months.median():.0f} 个月")
print(f"最短 {months.min()} 个月,最长 {months.max()} 个月")
```

均值 41.2 个月,中位数 37 个月。少数随访了七八年的患者把均值往上拉,中位数不受影响。所以论文里写的几乎总是"中位随访时间"(median follow-up)。住院天数、化验值、费用也是这样: 大多数人挤在一头,少数人拖出一条长尾。

均值还有一个弱点: 怕极端值。假设录入时把体重最大的那位患者的 145.2 kg 敲成了 1452:

```python
w = df["weight_kg"]
print(f"原来:   均值 {w.mean():.1f} kg,中位数 {w.median():.1f} kg")

typo = w.copy()
typo[w.idxmax()] = 1452.0                      # 一个录入错误
print(f"敲错后: 均值 {typo.mean():.1f} kg,中位数 {typo.median():.1f} kg")
```

200 位患者里只错了一位,均值就从 72.8 变成 79.3 kg;中位数纹丝不动。`w.idxmax()` 给出最大值所在的行号,`w.copy()` 先复制一份,免得改动原来的数据。

**离散程度**也有对应的两种说法:

- **标准差**(standard deviation, SD): 每位患者离均值大约差多少。和均值配着用。
- **四分位间距**(interquartile range, IQR): 下四分位数到上四分位数,中间一半患者所在的范围。和中位数配着用(第 3 课的 `median_iqr`、第 4 课箱线图的箱子)。

**怎么选**: 数据大致对称、没有极端值,写"均值(标准差)";有长尾或者有离得很远的患者,写"中位数(四分位间距)"。拿不准就先画直方图(第 4 课)。同一张 Table 1 里可以两种都有,表头写清楚就行。

<div class="quiz">
<p><b>想一想</b>: 一个病房 10 位患者的住院天数是 3、3、4、4、4、5、5、6、7、59(最后一位有严重的并发症)。要告诉别人"这个病房的患者一般住几天",用哪个数?</p>
<button class="quiz-opt" data-why="均值是 10 天,可是 10 位患者里有 9 位都没有住到 10 天。一位住了 59 天的患者把均值拉高了一倍多。">均值: 10 天</button>
<button class="quiz-opt" data-correct data-why="排好序,中间两个数是 4 和 5,中位数是 4.5 天: 一半患者比它短,一半比它长。那位住了 59 天的患者需要单独说明,但不该让它代表一般的患者。">中位数: 4.5 天</button>
<button class="quiz-opt" data-why="最大值只描述了最特殊的那一位患者。">最大值: 59 天</button>
</div>

<div class="exercise">

**练习 1 · 轮到你: 描述 Graphical 的 Δ**

算出 Graphical 的 Δ(`delta_gra`)的均值、中位数,以及下四分位数和上四分位数。

```python
x_gra = df["delta_gra"]
mean_gra = x_gra.___()
median_gra = x_gra.___()
q1_gra, q3_gra = x_gra.quantile([___, ___])
print(f"均值 {mean_gra:.2f},中位数 {median_gra:.1f} ({q1_gra:.1f} 到 {q3_gra:.1f}) Gy")
```

```pycheck
need("mean_gra", "median_gra", "q1_gra", "q3_gra")
if close(mean_gra, -3.4, 0.01) and close(median_gra, -3.55, 0.01):
    fail("均值和中位数填反了: 第一处是 mean,第二处是 median。")
if not close(mean_gra, -3.55, 0.01):
    fail(f"mean_gra 现在是 {mean_gra:.2f},应该是 −3.55。")
if not close(median_gra, -3.4, 0.01):
    fail(f"median_gra 现在是 {median_gra:.2f},应该是 −3.4。")
if not (close(q1_gra, -6.2, 0.01) and close(q3_gra, -0.975, 0.01)):
    fail(f"四分位数现在是 {q1_gra:.2f} 和 {q3_gra:.2f},应该是 −6.2 和 −0.98: quantile([0.25, 0.75])。")
ok("对了: 均值 −3.55,中位数 −3.4(四分位间距 −6.2 到 −1.0)Gy。均值比中位数更负一点: 第 4 课练习 1 的直方图上,左边那条降了十几 Gy 的长尾把均值拉了过去。")
```

<details><summary>提示</summary>

`.mean()`、`.median()`;四分位数是 `quantile([0.25, 0.75])`。

</details>

<details><summary>参考答案</summary>

```python
x_gra = df["delta_gra"]
mean_gra = x_gra.mean()
median_gra = x_gra.median()
q1_gra, q3_gra = x_gra.quantile([0.25, 0.75])
print(f"均值 {mean_gra:.2f},中位数 {median_gra:.1f} ({q1_gra:.1f} 到 {q3_gra:.1f}) Gy")
```

</details>

</div>

## 2. 换一批患者,结果会变

Volume 的 Δ,在我们的 200 位患者里中位数是 −2.0 Gy。如果另一家医院也收 200 位同类的患者,做同样的研究,还会正好得到 −2.0 吗?

真实的研究里没法试。模拟的数据可以: 先造一个很大的**总体**(population),假装它是"所有这类患者";每个研究是从里面随机抽到的 200 位,叫一个**样本**(sample)。

```python
pop = make_cohort(n=20000, seed=7)                          # 总体: 2 万位患者
pop_delta = (pop["d2cc_vol"] - pop["d2cc_5mm"]).to_numpy()
print("总体的中位数:", np.median(pop_delta), "Gy")

rng = np.random.default_rng(0)
for i in range(5):
    study = rng.choice(pop_delta, size=200, replace=False)  # 一个研究: 随机抽到的 200 位
    print(f"研究 {i + 1}: 中位数 {np.median(study):.2f} Gy")
```

总体的中位数是 −2.0 Gy,这是"真实值"。五个研究得到的却是 −1.95、−1.60、−2.10、−2.25、−1.85: 没有哪个研究做错了,只是抽到的患者不同。`np.random.default_rng(0)` 是一个固定了种子的随机数发生器,所以每次运行抽到的都是同样的患者;`rng.choice(..., size=200, replace=False)` 从总体里不重复地抽 200 个。

做 1000 个这样的研究:

```python
medians = np.array([np.median(rng.choice(pop_delta, size=200, replace=False)) for _ in range(1000)])
print(f"1000 个研究的中位数: 最小 {medians.min():.2f},最大 {medians.max():.2f} Gy")
print("中间 95% 的研究落在:", np.percentile(medians, [2.5, 97.5]).round(2), "Gy")
```

同一个研究重复 1000 次,中位数从 −2.70 到 −1.10 Gy 都出现过,中间 95% 的研究落在 −2.4 到 −1.5 Gy 之间。这种"因为抽到的患者不同而产生的波动",叫**抽样误差**(sampling error)。它和测量不准、录入错误无关,只要是从一部分患者去推断全体患者,就一定有。

所以只报一个 −2.0 是不够的。还要告诉读者: 这个数大概能晃动多少。

## 3. 95% 区间是什么意思

<div class="frame">

**背景**: 研究只做了一次,得到一个结果。上一节看到,换一批患者它会变。

**目标**: 给结果配上一个范围,告诉读者真实值大概在哪里。

**做法**: **95% 置信区间**(95% confidence interval, 95% CI)。先看最经典的一种: 均值的 95% 区间。

</div>

**它在算什么**(均值的 95% 区间)

1. 算出样本的均值。
2. 算出**标准误**(standard error, SE) = 标准差 ÷ √人数。它估计的是: 换一批患者,均值大约会晃动多少。人越多,标准误越小。
3. 区间 = 均值 ± 约 2 个标准误(人数很多时是 1.96 个;30 位患者时是 2.05 个)。

**为什么是"约 2 个"**: 上一节那 1000 个研究的结果,堆起来是中间高、两头低的钟形,其中大约 95% 落在离中心 2 个标准误以内。所以从你的结果往两边各走 2 个标准误,大多数时候能把真实值盖进去。

"大多数时候"是多少?下面这张图让你自己数。

<div class="demo">

**设定**: 假设所有这类患者的 Δ 的真实平均值是 −2.0 Gy(竖线);真实的研究里这个数是不知道的。每个"研究"随机抽 30 位患者,算出平均值(圆点)和它的 95% 区间(横线)。

**你做什么**: 点几次"做 1 个研究",再点"做 20 个",直到做满 100 个。

**你会看到**: 每个研究的圆点和区间都不一样。前 9 个研究的区间都盖住了竖线,第 10 个没有盖住(红色粗线)。做满 100 个: 95 个盖住了,5 个没有。

**这说明**: "95%"说的是这个方法的成功率: 用这种办法算出来的区间,100 次里大约有 95 次包含真实值。

<div class="widget" data-widget="ci">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

在真实的研究里,你手里只有这 100 条横线中的一条,而且看不到竖线。你不知道自己的这一条是不是那 5 条红线之一。能说的是: 它是用一个 95% 的时候都能包含真实值的办法算出来的。

读一个 95% 区间,看两样东西:

- **它有多宽**: 窄,说明估计得准;宽,说明患者太少或者数据太散,结论要留余地。
- **它盖住了哪些值**: 区间里的每一个值,都是和数据相容的"真实值"的候选。如果 Δ 的区间是 −2.4 到 −1.5 Gy,那么"真实的下降是 1.5 Gy"和"是 2.4 Gy"都说得通,"没有下降(0)"则说不通。

<div class="quiz">
<p><b>想一想</b>: 一篇论文写: "换成 Volume 后,阴道 D2cc 变化的中位数是 −2.0 Gy(95% CI −2.4 到 −1.5)"。下面哪种理解是对的?</p>
<button class="quiz-opt" data-why="这是把置信区间当成了患者数据的范围。95% 的患者落在哪里,要看数据本身的分布(这批数据里是 −7.6 到 3.1 Gy 左右),那个范围宽得多。置信区间说的是中位数这个数估计得准不准。">95% 的患者的 Δ 在 −2.4 到 −1.5 Gy 之间</button>
<button class="quiz-opt" data-correct data-why="置信区间描述的是对真实中位数的估计有多准。产生这个区间的办法,在反复做研究时有 95% 的次数能包含真实值。">真实的中位数很可能在 −2.4 到 −1.5 Gy 之间;这个范围是用一个 95% 的时候都能包含真实值的办法算出来的</button>
<button class="quiz-opt" data-why="P 值和置信区间有关系,但 95% 不是这个意思,而且这句话把两个概念混在了一起。P 值第 6 课讲。">结果有 95% 的可能是真的,5% 的可能是假的</button>
</div>

## 4. bootstrap: 中位数的 95% 区间

<div class="frame">

**背景**: 均值的区间有现成的公式(均值 ± 2 个标准误)。可是我们报的是中位数,还有比例、两个中位数的差……这些量大多没有简单的公式。

**目标**: 一个对什么量都能用的办法,只用手里这 200 位患者,估计"换一批患者,结果会晃动多少"。

**做法**: **bootstrap**(自助法): 从这 200 位患者里**有放回地**重新抽 200 位,当作"另一批患者",这样重复几千次。

</div>

**它在算什么**

1. 把 200 位患者的 Δ 写在 200 张卡片上,放进一个盒子。
2. 抽一张,记下来,**放回去**;再抽一张……一共抽 200 次。这 200 个数是一个"重抽样本": 有的患者被抽到两三次,有的一次也没被抽到。
3. 算这个重抽样本的中位数。
4. 把第 2–3 步重复 2000 次,得到 2000 个中位数。
5. 把它们从小到大排好,取第 2.5% 和第 97.5% 的位置上的两个数,就是 95% 区间。

**为什么用它**: 第 2 节的办法(从总体里一遍遍地抽)才是正宗的,可是真实研究里没有总体,只有手里的 200 位。bootstrap 的想法是: 这 200 位患者是我们对总体最好的了解,那就把他们当成总体的缩影,从里面去抽。它不需要公式,也不要求数据是什么形状,所以对中位数、比例、差值都能用。

**打个比方**: 想知道一锅汤咸不咸,舀一勺尝一尝;想知道"这一勺能不能代表整锅",就在这一勺里反复小口地尝,看每一口差多少。

先做第 1–2 步,看看一次重抽是什么样子:

```python
rng = np.random.default_rng(1)
picked = pd.Series(rng.choice(df["id"], size=200, replace=True))    # 有放回地抽 200 位患者
print("被抽到次数最多的 3 位:")
print(picked.value_counts().head(3))
print("一次也没被抽到的患者:", 200 - picked.nunique(), "位")
```

`replace=True` 就是"放回去"。有的患者被抽到了好几次,大约三分之一的患者一次也没被抽到: 这个重抽样本和原来的 200 位不一样,就像换了一批患者。

再把 5 步连起来做一遍:

```python
x = df["delta_vol"].to_numpy()
rng = np.random.default_rng(1)
boot = np.array([np.median(rng.choice(x, size=len(x), replace=True)) for _ in range(2000)])   # 第 2–4 步
low, high = np.percentile(boot, [2.5, 97.5])                                                  # 第 5 步
print(f"中位数 {np.median(x):.1f} Gy,95% 区间 {low:.1f} 到 {high:.1f} Gy")
```

中位数 −2.0 Gy,95% 区间 −2.4 到 −1.5 Gy。回头看第 2 节: 从总体里真的抽 1000 个研究,中间 95% 落在 −2.4 到 −1.5 Gy。只用手里的 200 位患者,bootstrap 给出了几乎一样的范围。

把 2000 个重抽的中位数画出来:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.hist(boot, bins=20)
for v in (low, high):
    ax.axvline(v, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Median change in vaginal D2cc in a resample (Gy)")
ax.set_ylabel("Number of resamples")
plt.show()
```

两条虚线之间是中间的 95%。

**不用自己写循环**。SciPy 里有现成的函数,结果一样:

```python
from scipy import stats

res = stats.bootstrap((x,), np.median, n_resamples=2000, confidence_level=0.95,
                      method="percentile", rng=np.random.default_rng(1))
print(round(res.confidence_interval.low, 1), "到", round(res.confidence_interval.high, 1), "Gy")
```

`(x,)` 是"只有一组数据"的写法(括号里的逗号不能省);`np.median` 是要算的量;`method="percentile"` 就是上面第 5 步的取法。

<div class="quiz">
<p><b>想一想</b>: bootstrap 抽卡片时为什么要"放回去"?如果不放回,抽 200 次会怎样?</p>
<button class="quiz-opt" data-correct data-why="不放回地从 200 张里抽 200 张,每次抽到的都是原来那 200 位患者,只是顺序不同,中位数永远是同一个数,也就看不出任何波动。放回去,每次重抽的组成才会不一样。">不放回,每次抽到的都是原来那 200 位,2000 个中位数全都一样</button>
<button class="quiz-opt" data-why="放不放回都抽得满 200 次。区别在于抽到的是谁。">不放回,抽不满 200 次</button>
<button class="quiz-opt" data-why="放回只是为了让每次重抽的组成不同,和计算快慢没有关系。">放回去算得更快</button>
</div>

<div class="exercise">

**练习 2 · 轮到你: Graphical 的 95% 区间**

对 Graphical 的 Δ 做 bootstrap(2000 次),算出中位数的 95% 区间,存进 `low_gra` 和 `high_gra`。

```python
xg = df["delta_gra"].to_numpy()
rng = np.random.default_rng(1)
boot_gra = np.array([np.median(rng.choice(xg, size=len(xg), replace=___)) for _ in range(2000)])
low_gra, high_gra = np.percentile(boot_gra, [___, ___])
print(f"中位数 {np.median(xg):.1f} Gy,95% 区间 {low_gra:.1f} 到 {high_gra:.1f} Gy")
```

```pycheck
need("low_gra", "high_gra")
if close(low_gra, high_gra, 1e-9):
    fail("2000 个中位数全都一样,区间缩成了一个点: 要有放回地抽,replace=True。")
if close(low_gra, -4.1, 0.01) and close(high_gra, -2.9, 0.01):
    fail("这是中间的 90%(第 5% 到第 95%)。95% 区间取第 2.5% 和第 97.5%。")
if close(low_gra, -4.3, 0.01) and close(high_gra, -2.8, 0.01):
    ok("对了: Graphical 的 Δ 中位数 −3.4 Gy(95% CI −4.3 到 −2.8)。区间比 Volume 的宽(1.5 Gy 对 0.9 Gy): Graphical 的 Δ 在患者之间差别更大,中位数也就估计得没那么准。")
fail(f"现在是 {low_gra:.2f} 到 {high_gra:.2f},应该是 −4.3 到 −2.8。检查 replace=True 和 [2.5, 97.5]。")
```

<details><summary>提示</summary>

有放回: `replace=True`;95% 区间的两头是第 2.5 和第 97.5 百分位数。

</details>

<details><summary>参考答案</summary>

```python
xg = df["delta_gra"].to_numpy()
rng = np.random.default_rng(1)
boot_gra = np.array([np.median(rng.choice(xg, size=len(xg), replace=True)) for _ in range(2000)])
low_gra, high_gra = np.percentile(boot_gra, [2.5, 97.5])
print(f"中位数 {np.median(xg):.1f} Gy,95% 区间 {low_gra:.1f} 到 {high_gra:.1f} Gy")
```

</details>

</div>

## 5. 什么量都能用: 比例的区间

把上面的循环包成一个函数,要算的量作为参数传进去:

```python
def boot_ci(values, stat=np.median, n_boot=2000, seed=1):
    """bootstrap 的 95% 区间(百分位数法)。stat 是要算的量,默认是中位数。"""
    rng = np.random.default_rng(seed)
    boot = np.array([stat(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)])
    return np.percentile(boot, [2.5, 97.5])

print("中位数的区间:", boot_ci(x))
print("均值的区间:  ", boot_ci(x, stat=np.mean).round(2))
```

比例也一样。5 mm 计划超过 68 Gy 的 148 位患者里,换成 Volume 后降到 68 Gy 及以下的占多少?把每位患者记成 1(降到了)或 0(没有),比例就是这串 0 和 1 的均值:

```python
high = df["d2cc_5mm"] > 68
crossed_vol = (df.loc[high, "d2cc_vol"] <= 68).to_numpy().astype(int)     # 148 个 0 或 1
print(f"{crossed_vol.sum()} / {len(crossed_vol)} = {crossed_vol.mean():.1%}")
print("95% 区间:", boot_ci(crossed_vol, stat=np.mean).round(3))
```

36 / 148 = 24.3%,95% 区间 17.6% 到 31.1%。单说"四分之一的患者能降到 68 Gy 及以下"听起来很确定;加上区间,读者才知道真实的比例可能低到六分之一,也可能高到三分之一。

单个比例的区间也有现成的公式,论文里常用 Wilson 法或精确法(Clopper–Pearson): 这里分别是 18.1% 到 31.8% 和 17.7% 到 32.1%,和 bootstrap 相差在 1 个百分点以内。bootstrap 的好处是同一个办法什么量都能用。

<div class="exercise">

**练习 3 · 轮到你: Graphical 的比例**

同样的 148 位患者,换成 **Graphical** 后降到 68 Gy 及以下的比例是多少?用 `boot_ci` 算它的 95% 区间,存进 `ci_gra`。

```python
crossed_gra = (df.loc[high, ___] <= 68).to_numpy().astype(int)
p_gra = crossed_gra.mean()
ci_gra = boot_ci(crossed_gra, stat=___)
print(f"{crossed_gra.sum()} / {len(crossed_gra)} = {p_gra:.1%},95% 区间 {ci_gra.round(3)}")
```

```pycheck
need("p_gra", "ci_gra")
if close(p_gra, 0.2432, 0.001):
    fail('这是 Volume 的比例: 第一处填 "d2cc_gra"。')
if close(p_gra, 0.3649, 0.001):
    fail("这是用 < 68 算的: 有两位换成 Graphical 后正好是 68.0 Gy,它们也算降到了 68 Gy 及以下。用 <= 68。")
if not close(p_gra, 0.3784, 0.001):
    fail(f"p_gra 现在是 {p_gra:.3f},应该是 0.378(56 / 148)。")
if close(ci_gra[0], ci_gra[1], 1e-9):
    fail("区间缩成了一个点: 0 和 1 的中位数不是 0 就是 1。比例是均值,stat 填 np.mean。")
if close(ci_gra[0], 0.2973, 0.002) and close(ci_gra[1], 0.4527, 0.002):
    ok("对了: 56 / 148 = 37.8%(95% CI 29.7% 到 45.3%)。Volume 是 24.3%(17.6% 到 31.1%)。两个区间有一小段重叠;两种计划的差别是不是碰巧,要用第 6 课的办法来回答。")
fail(f"区间现在是 {ci_gra},应该是 0.297 到 0.453。stat 填 np.mean。")
```

<details><summary>提示</summary>

列名是 `"d2cc_gra"`;比例是 0 和 1 的均值,所以 `stat=np.mean`。

</details>

<details><summary>参考答案</summary>

```python
crossed_gra = (df.loc[high, "d2cc_gra"] <= 68).to_numpy().astype(int)
p_gra = crossed_gra.mean()
ci_gra = boot_ci(crossed_gra, stat=np.mean)
print(f"{crossed_gra.sum()} / {len(crossed_gra)} = {p_gra:.1%},95% 区间 {ci_gra.round(3)}")
```

</details>

</div>

## 6. 患者越多,区间越窄

如果研究只收了 50 位患者,区间会变成什么样?

```python
print("200 位患者:", boot_ci(x))
print("只有前 50 位:", boot_ci(x[:50]))
```

200 位患者的区间是 −2.4 到 −1.5 Gy,宽 0.9 Gy;只有 50 位时是 −3.35 到 −1.40 Gy,宽约 2 Gy。**患者数变成 4 倍,区间大约窄一半**: 区间的宽度和 √人数 成反比(第 3 节的标准误 = 标准差 ÷ √人数)。

这条规律有两个用处。设计研究时: 想把区间缩窄一半,需要 4 倍的患者,而不是 2 倍。读论文时: 小样本的研究,区间一定宽;一个只有 30 位患者的研究说"没有发现差别",很可能只是区间太宽,什么都盖得住。

<div class="exercise">

**练习 4 · 轮到你: 只有 50 位患者**

只用前 50 位患者,Graphical 的 Δ 中位数的 95% 区间是多少?存进 `ci50_gra`,再算它的宽度是 200 位患者时的多少倍。

```python
ci50_gra = boot_ci(xg[:___])
ci200_gra = boot_ci(xg)
ratio = (ci50_gra[1] - ci50_gra[0]) / (ci200_gra[1] - ci200_gra[0])
print(f"50 位: {ci50_gra},200 位: {ci200_gra},宽度是 {ratio:.1f} 倍")
```

```pycheck
need("ci50_gra", "ratio")
if close(ci50_gra[0], -4.3, 0.01) and close(ci50_gra[1], -2.8, 0.01):
    fail("ci50_gra 现在用的还是全部 200 位患者: 方括号里填 50,取前 50 位。")
if close(ci50_gra[0], -6.0, 0.01) and close(ci50_gra[1], -3.4, 0.01):
    ok(f"对了: 50 位患者时区间是 −6.0 到 −3.4 Gy,宽度是 200 位患者时的 {ratio:.1f} 倍,接近理论上的 2 倍(√4)。")
fail(f"现在是 {ci50_gra},应该是 −6.0 到 −3.4。xg[:50] 取前 50 位。")
```

<details><summary>提示</summary>

`xg[:50]` 取前 50 个。

</details>

<details><summary>参考答案</summary>

```python
ci50_gra = boot_ci(xg[:50])
ci200_gra = boot_ci(xg)
ratio = (ci50_gra[1] - ci50_gra[0]) / (ci200_gra[1] - ci200_gra[0])
print(f"50 位: {ci50_gra},200 位: {ci200_gra},宽度是 {ratio:.1f} 倍")
```

</details>

</div>

## 小结

- 对称的数据报均值(标准差);有长尾或极端值的数据报中位数(四分位间距)。均值怕极端值,中位数不怕。
- 任何从样本算出来的数,换一批患者都会变,这是抽样误差。
- 95% 置信区间给结果配上一个范围。"95%"是方法的成功率: 这样算出的区间,100 次里约有 95 次包含真实值。它不是 95% 的患者所在的范围。
- bootstrap: 从手里的患者里有放回地重抽,重复几千次,取中间的 95%。对中位数、比例、均值都能用。
- 区间的宽度和 √人数 成反比: 4 倍的患者,区间窄一半。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 均值、中位数 | mean、median | 概括"中间"的两种办法 |
| 标准差 | standard deviation (SD) | 每位患者离均值大约差多少 |
| 四分位间距 | interquartile range (IQR) | 中间一半患者所在的范围 |
| 总体、样本 | population、sample | 所有这类患者;研究里实际收到的患者 |
| 抽样误差 | sampling error | 因为抽到的患者不同,结果产生的波动 |
| 标准误 | standard error (SE) | 换一批患者,估计值大约晃动多少 |
| 置信区间 | confidence interval (CI) | 和数据相容的真实值的范围 |
| 自助法 | bootstrap | 有放回地重抽患者,估计结果的波动 |
| 重抽样本 | resample | 一次有放回的重抽得到的那批患者 |

## 这一课没讲什么

- **bootstrap 的其他取法**(BCa 等): 比百分位数法更讲究,数据里重复的值很多时不一定算得出来。这门课只用百分位数法。
- **比例的公式区间**(Wilson、Clopper–Pearson): 患者少、比例接近 0 或 1 时比 bootstrap 更稳,论文里也常见。
- **两组之间差别的区间**和 **P 值**: 第 6 课。
- **贝叶斯的可信区间**(credible interval): 另一套思路,这门课不讲。

## 写进论文 Writing it up

<div class="paper">

方法里写清楚区间是怎么算的:

> Changes are summarised as medians with 95% confidence intervals obtained by bootstrap resampling of patients (2000 resamples, percentile method).

结果里,估计值和区间写在一起:

> Compared with the 5 mm plan, vaginal D2cc with the Volume plan changed by a median of −2.0 Gy (95% CI −2.4 to −1.5).

**审稿人可能问**: 为什么用 bootstrap,而不是常规的公式?

**怎么回答**: 我们报告的是中位数,数据有长尾;中位数的区间没有简单的公式,bootstrap 不需要对数据的分布做假设。重抽的是患者(不是单次测量),重抽 2000 次,随机种子固定,结果可以重复。

**审稿人可能问**: 只给了 P 值,效应有多大、估计得有多准?

**怎么回答**: 每个主要结果都给出估计值和 95% 区间。区间比单独一个 P 值告诉读者的更多。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| bootstrap 的区间上下限是同一个数 | 忘了 `replace=True`,或者对 0/1 的数据算了中位数(应该算均值)。 |
| 每次运行,区间都差一点 | 没有固定随机种子。用 `np.random.default_rng(1)` 这样带种子的写法。 |
| 区间里出现 NaN | 数据里有缺失值。先用 `dropna()` 去掉,并记下用了多少位患者。 |
| `TypeError: bootstrap() got an unexpected keyword argument 'rng'` | 自己电脑上的 SciPy 版本较旧(早于 1.15),把 `rng=` 换成 `random_state=`。 |

## 延伸阅读 Further reading

- SciPy 文档: [scipy.stats.bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)。
- Altman DG, Bland JM. [Standard deviations and standard errors](https://www.bmj.com/content/331/7521/903). *BMJ* 2005;331:903。一页纸讲清标准差和标准误的区别。

{{< practice >}}

**下一课**: 有了区间,下一个问题是: 两组之间的差别、两个量之间的关系,会不会只是碰巧?第 6 课讲 P 值、常用的几种检验和相关系数。
