---
title: "Lesson 2 · 逻辑回归: 概率和 OR Logistic Regression"
summary: "预测是/否的结局: 从直线到 S 形曲线;概率、odds 和 OR。"
weight: 20
toc: true
duration: "约 80 分钟,可以分两次学: 第 0–3 节,第 4–7 节"
prereq: "本课程[第 1 课](../01-linear-regression/)(线性回归)"
questions:
  - "结局只有是和否,为什么不能再用直线?"
  - "论文里的 OR 是怎么算出来的?它和概率是什么关系?"
  - "来了一位新患者,模型给出的到底是什么?"
objectives:
  - "把概率换成 odds 和 log-odds,再换回来"
  - "说出逻辑回归用什么尺子量好坏,并且自己动手调一条 S 形曲线"
  - "用 scikit-learn 拟合逻辑回归,把系数换成 OR"
  - "给一位新患者算出概率,而不只是是或否"
  - "知道 scikit-learn 默认给系数加了约束,以及怎么关掉"
---

第 1 课预测的是"变多少 Gy"。临床上更常问的是"会不会": 这位患者 5 mm 计划的阴道 D2cc 是 71 Gy,超过了 68 Gy;换成 Volume,能不能降到 68 Gy 以下?

为什么是 68 Gy: 已发表的研究(Qiang 等,*Cancers* 2026;18:2350)里,阴道 D2cc 超过 68 Gy 的 70 位患者中 25 位出现了晚期阴道副作用,不超过的 50 位中只有 9 位。能不能降到这条线以下,是换方案值不值得的一个判断依据。

结局只有"是"和"否"时,标准的模型是**逻辑回归**(logistic regression)。你在论文的多因素分析表里见过它的输出: 那一列 OR 就是它给的。这一课把它拆开,还是第 1 课的三件事: 公式长什么样,用什么尺子量好坏,怎么找到最好的那一组数。

> 🔬 逻辑回归和线性回归只差两处: 公式外面套了一层,把直线弯成 S 形,让结果落在 0 和 1 之间;尺子从"误差平方"换成了"对实际结果给了多大的概率"。其余的,包括系数怎么读、怎么加变量,都和第 1 课一样。

**练习怎么做**: 讲解里预测的是 **Volume** 能不能降到 68 Gy 以下。练习都换成 **Graphical**: 每讲完一步,你就对 Graphical 把这一步再做一遍。

## 0. 准备: 哪些患者,什么结局

只看 5 mm 计划达到或超过 68 Gy 的患者: 对他们来说,"能不能降到 68 Gy 以下"才是个问题。

```python
import numpy as np
import pandas as pd
from cohort import make_cohort

df = make_cohort().dropna()
hi = df[df["d2cc_5mm"] >= 68].copy()               # 5 mm 计划达到或超过 68 Gy 的患者
hi["excess"] = hi["d2cc_5mm"] - 68                 # 超出 68 Gy 多少
hi["event"] = (hi["d2cc_vol"] < 68).astype(int)    # 换成 Volume 后降到 68 Gy 以下: 是 1,否 0
print(len(hi), "位患者,其中", hi["event"].sum(), "位降到了 68 Gy 以下")
hi[["id", "d2cc_5mm", "excess", "d2cc_vol", "event"]].head()
```

144 位患者,34 位(23.6%)降到了 68 Gy 以下。`event` 是这一课要预测的结局: 1 表示降下来了,0 表示没有。`excess` 是 5 mm 计划超出 68 Gy 多少,这一课先用它来预测: 超得越多,越难降下来。按超出的多少分组看一眼:

```python
hi["group"] = pd.cut(hi["excess"], [0, 2, 4, 6, 8, 20], right=False)
hi.groupby("group", observed=True)["event"].agg(["sum", "count", "mean"]).round(2)
```

超出不到 2 Gy 的 27 位里,18 位降下来了(67%);超出 2–4 Gy 的是 36%;超出 8 Gy 以上的 32 位,一位也没有。比例随着超出的剂量一路下降,从大半到零。

## 1. 直线为什么不行

第 1 课的办法是拟合一条直线。结局是 0 和 1,也是数,直线照样能拟合。

<div class="quiz">
<p><b>先猜一猜</b>: 把 0 和 1 当成普通的数,用第 1 课的直线去拟合"降下来的可能性"。最先出问题的是什么?</p>
<button class="quiz-opt" data-why="scikit-learn 不管这些数代表什么,照样会算出一条直线。问题出在这条直线给出的数上。运行下面的代码看看。">程序报错: 结局不是连续的数</button>
<button class="quiz-opt" data-correct data-why="直线没有上下限。超出的剂量大到一定程度,直线就穿到 0 以下去了,而可能性不能是负数。运行下面的代码,看有多少位患者的预测值小于 0。">对有些患者,直线给出的"可能性"小于 0</button>
<button class="quiz-opt" data-why="会有一条下降的直线,方向是对的。问题不在方向,而在它给出的数会跑到 0 和 1 的外面。运行下面的代码看看。">直线是平的,看不出关系</button>
</div>

```python
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

line = LinearRegression().fit(hi[["excess"]], hi["event"])
pred_line = line.predict(hi[["excess"]])
print(f"直线: 可能性 = {line.intercept_:.2f} {line.coef_[0]:+.3f} × 超出的剂量")
print(f"超出 10 Gy 时: {line.intercept_ + line.coef_[0] * 10:.2f}")
print("预测值小于 0 的患者:", (pred_line < 0).sum(), "位")

fig, ax = plt.subplots()
ax.scatter(hi["excess"], hi["event"], s=12, alpha=0.4)
xs = np.linspace(0, 15, 50)
ax.plot(xs, line.intercept_ + line.coef_[0] * xs, color="black")
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_xlabel("5 mm plan D2cc above 68 Gy (Gy)")
ax.set_ylabel("Below 68 Gy with the Volume plan (1 = yes)")
plt.show()
```

直线是: 可能性 = 0.56 − 0.061 × 超出的剂量。超出 10 Gy 时,它给出 −0.05;144 位患者里有 20 位的预测值是负数。"降下来的可能性是 −5%"没有意义。

毛病出在形状上: 可能性在两头应该变平,靠近 1 就不再升,靠近 0 就不再降,中间才变化得快,是一条 S 形的曲线。直线做不到。办法不是放弃直线,而是先把"可能性"换一种说法,让它可以取任意大小的数,再对它用直线。

## 2. 概率、odds、log-odds

同一个可能性,有三种说法。

- **概率**(probability): 100 位这样的患者里有几位发生。在 0 和 1 之间。
- **odds**(比值,有的书译作"优势"): 发生的概率除以不发生的概率。概率 0.8 时,odds = 0.8 ÷ 0.2 = 4,也就是"4 比 1"。在 0 和无穷大之间。
- **log-odds**: 对 odds 取自然对数。odds 是 1(五五开)时,log-odds 是 0;odds 小于 1 时是负数。可以是任意的数,没有上下限。

```python
for p in [0.1, 0.5, 0.8, 0.9]:
    odds = p / (1 - p)
    print(f"概率 {p:.1f} → odds {odds:.2f} → log-odds {np.log(odds):+.2f}")
```

概率 0.1、0.5、0.8、0.9,对应的 odds 是 0.11、1、4、9,log-odds 是 −2.20、0、+1.39、+2.20。三种说法是一回事,可以互相换算;log-odds 没有上下限,正好可以放心地交给一条直线。

**OR**(odds ratio,比值比)是两个 odds 相除。拿开头那张已发表的表手算一遍:

```python
odds_high = 25 / 45      # D2cc 超过 68 Gy: 25 位有晚期阴道副作用,45 位没有
odds_low = 9 / 41        # 不超过 68 Gy: 9 位有,41 位没有
print(f"odds: {odds_high:.2f} 和 {odds_low:.2f};OR = {odds_high / odds_low:.2f}")
print(f"概率: {25 / 70:.2f} 和 {9 / 50:.2f};概率之比 = {(25 / 70) / (9 / 50):.2f}")
```

超过 68 Gy 的患者,出现副作用的 odds 是 25 ÷ 45 = 0.56;不超过的是 9 ÷ 41 = 0.22。OR = 0.56 ÷ 0.22 = 2.53。

<div class="quiz">
<p><b>想一想</b>: OR = 2.53。能不能说"D2cc 超过 68 Gy 的患者,出现副作用的概率是不超过的 2.53 倍"?</p>
<button class="quiz-opt" data-why="OR 是两个 odds 的比,不是两个概率的比。上面的代码同时算了概率: 0.36 和 0.18,相差 1.98 倍,不是 2.53 倍。">能: OR 就是概率的倍数</button>
<button class="quiz-opt" data-correct data-why="两组的概率是 0.36 和 0.18,概率之比是 1.98;OR 是 odds 之比,2.53。事件越常见,两者差得越多;事件很少见(比如 1% 对 2%)时才接近。把 OR 读成'概率是几倍',会把效应说大。">不能: OR 是 odds 的倍数。这里概率是 1.98 倍</button>
<button class="quiz-opt" data-why="OR 大于 1 说明超过 68 Gy 的一组 odds 更高,方向没有错。问题在于它是 odds 的倍数,不是概率的倍数。">不能: OR 大于 1 说明副作用更少</button>
</div>

## 3. 它在算什么: 一条 S 形曲线和一把新尺子

<div class="frame">

**背景**: 结局是 0 和 1。直线给出的"可能性"会跑到 0 以下。

**目标**: 一条公式,把"超出的剂量"换算成"降到 68 Gy 以下的概率",结果永远在 0 和 1 之间。

**做法**: 让直线去预测 log-odds,再把 log-odds 换回概率。这就是逻辑回归。系数用**最大似然**(maximum likelihood)来找。

</div>

**它在算什么**

1. 公式: log-odds = b₀ + b₁ × 超出的剂量。还是第 1 课的直线,只是直线预测的不是结局本身,而是 log-odds。
2. 换回概率: 概率 = 1 ÷ (1 + e<sup>−log-odds</sup>)。log-odds 很大时概率接近 1,很小时接近 0,中间是 S 形。
3. 换一把尺子。对每位患者,看模型给"**实际发生的那个结果**"的概率: 降下来了的患者,看模型说的"降下来的概率";没降下来的,看"降不下来的概率"。这个概率越低,模型越该罚: 罚分 = −ln(这个概率)。
4. 找出让所有患者的平均罚分最小的 b₀、b₁。这等于让"这批患者实际的结局"在模型眼里尽量可能,所以叫最大似然。这一步没有一步到位的公式,计算机是一点一点逼近的。

第 3 步的罚分,代几个数看看: 模型给实际结果的概率是 0.9,罚 0.11;0.5(五五开),罚 0.69;0.1,罚 2.30;0.01,罚 4.61。说对了几乎不罚;说错了而且说得很肯定,罚得很重。

**为什么用这把尺子**

- 结局只有 0 和 1,模型给的是概率,"差了几 Gy"这种误差不存在。能比的只有: 你给实际发生的事多大的概率。
- 它鼓励模型说实话: 没有把握时说 50%,只罚 0.69;硬说 99% 而说错了,罚 4.61。
- 每位患者的概率相乘,就是"这批患者的结局全部如此"的概率;取对数以后乘法变成加法,所以罚分用 ln。

**打个比方**: 术前谈话。告诉患者"并发症的可能不到 1%",结果发生了;和告诉患者"大约有三成可能",结果发生了。两次都"说错了",但前一次错得重得多。这把尺子量的就是这个轻重。

<div class="demo">

**设定**: 40 位模拟患者,5 mm 计划都超过 68 Gy。横轴是超出多少;实心点在上面一排,是降到了 68 Gy 以下的患者,空心点在下面一排,是没降下来的。黑色曲线是模型说的"降下来的概率"。每位患者到曲线的竖线越长,模型给实际结果的概率越低。

**你做什么**: 拖"中点"(概率正好 50% 的位置),把曲线挪到实心点和空心点交界的地方;再拖"陡度"。让下面的平均罚分尽量小,然后点"交给最大似然"。

**你会看到**: 一开始是 0.57。中点调到 2.5 Gy 上下、陡度调到 0.7 上下时,降到 0.35。点"交给最大似然",得到 log-odds = 1.72 − 0.68 × 超出的剂量,也是 0.35。

**这说明**: 和第 1 课一样,最好的曲线由一把尺子决定,只有一条。尺子换成了平均罚分。

<div class="widget" data-widget="sigmoid">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

把陡度拖到最大再看: 曲线变成一级台阶,对交界处附近的患者说得非常肯定;只要有一位说错,平均罚分就上去了。曲线太平也不好: 对谁都说得含糊。最大似然找的是两者之间的那一条。

上面第 4 步说,计算机是"一点一点逼近"的。下面这张图把这个过程放慢给你看,还是那 40 位患者。图分两半: 上面是概率,下面是同一个模型画成 log-odds 的样子。点"下一步",图往前走一步;点 ↻ 把这一步重放一遍。

<div class="widget stage" data-widget="mle">

<div class="stage-step" data-label="40 位患者">
<p class="stage-h">40 位患者,10 位降下来了</p>

横轴是 5 mm 计划超出 68 Gy 多少。实心点在上面一排,是降到 68 Gy 以下的 10 位;空心点在下面一排,是没降下来的 30 位。

要找的是一条曲线: 超出多少,降下来的概率是多少。

</div>

<div class="stage-step" data-label="平线">
<p class="stage-h">起点: 对谁都说 25%</p>

40 位里 10 位降下来了,所以先对每位患者都说 25%: 一条平线。下面的小图是同一个模型在 log-odds 上的样子,也是平的: ln(0.25 ÷ 0.75) = −1.10。

每位患者到曲线的竖线,是模型说的概率离实际结果有多远。这时的平均罚分是 0.56。

</div>

<div class="stage-step" data-label="罚分">
<p class="stage-h">每位患者罚多少</p>

红圈里的患者降下来了,而模型只给了 25% 的概率,罚 −ln(0.25) = 1.39。下面一排的患者没降下来,模型给"没降下来"的概率是 75%,只罚 −ln(0.75) = 0.29。

40 位的罚分平均起来,就是 0.56。接下来要挪动曲线,让这个数变小。

</div>

<div class="stage-step" data-label="挪一步">
<p class="stage-h">挪一步: 左边抬高,右边压低</p>

降下来的患者大多在左边(超出得少),没降下来的大多在右边。左边的实心点把曲线往上拉,右边的空心点把它往下拉,于是下面的直线向右下倾斜,上面的曲线弯了过来。

算法每一步做的是: 看每位患者"实际结果 − 模型说的概率"有多大、朝哪边,算出截距和斜率各该改多少。这一步以后,log-odds = 1.19 − 0.39 × 超出,平均罚分从 0.56 降到 0.39。

</div>

<div class="stage-step" data-label="再挪几步">
<p class="stage-h">再挪几步,直到挪不动</p>

同样的事再做一遍。第 2 步以后平均罚分是 0.354,第 3 步 0.348,第 4、5 步几乎不再变。到它不再下降,算法就停: 这里一共 5 步。放完以后可以拖滑块,回看每一步之后的曲线。

线性回归有公式,一步到位(第 1 课的碗底)。这里没有这样的公式,只能一步一步走,不过通常几步就到。

</div>

<div class="stage-step" data-label="停下来">
<p class="stage-h">停下来的地方: 最大似然的答案</p>

log-odds = 1.72 − 0.68 × 超出的剂量,平均罚分 0.35。这和你手动调完、再点"交给最大似然"得到的是同一条曲线。

竖的虚线是曲线的中点,超出 2.54 Gy: 概率正好 50%,在下面的小图里就是直线穿过 0 的地方。斜率 −0.68 的意思是每多超出 1 Gy,odds 乘以 0.51(第 5 节讲怎么换算)。

</div>

<div class="stage-step" data-label="新患者">
<p class="stage-h">给一位新患者算概率</p>

一位超出 2.0 Gy 的新患者: 先在下面的直线上读出 log-odds(1.72 − 0.68 × 2.0,约 0.4),再换成概率: 59%。

两张图是同一个模型的两种画法: 直线好算,曲线好读。

</div>

</div>

## 4. 从零算一遍,再用 scikit-learn

最大似然没有现成的公式。最笨的办法是把 b₀、b₁ 的各种组合都试一遍,留下平均罚分最小的那一组:

```python
x = hi["excess"].to_numpy()
y = hi["event"].to_numpy()

def mean_penalty(b0, b1):
    p = 1 / (1 + np.exp(-(b0 + b1 * x)))             # 每位患者"降下来"的概率
    p_actual = np.where(y == 1, p, 1 - p)            # 模型给"实际发生的结果"的概率
    return -np.log(p_actual).mean()                  # 平均罚分

best = min((mean_penalty(b0, b1), b0, b1)
           for b0 in np.arange(0, 3.01, 0.02) for b1 in np.arange(-1.5, 0.01, 0.02))
print(f"平均罚分最小是 {best[0]:.3f},在 b0 = {best[1]:.2f}、b1 = {best[2]:.2f}")
print(f"对谁都说 50% 时(b0 = 0, b1 = 0): {mean_penalty(0, 0):.3f}")
```

试了一万多组,平均罚分最小是 0.370,在 b₀ = 1.24、b₁ = −0.64。对谁都说 50%,是 0.693。

实际工作里用 scikit-learn 的 `LogisticRegression`。它不是一组一组地试,而是从一组系数出发,每一步都往罚分更小的方向挪一点:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1e6).fit(hi[["excess"]], hi["event"])
b0, b1 = model.intercept_[0], model.coef_[0][0]
print(f"log-odds = {b0:.2f} {b1:+.2f} × 超出的剂量;平均罚分 {mean_penalty(b0, b1):.3f}")

rate = y.mean()
print(f"基线(对谁都说 {rate:.1%}): 平均罚分 {mean_penalty(np.log(rate / (1 - rate)), 0):.3f}")
```

scikit-learn 给出 log-odds = 1.25 − 0.65 × 超出的剂量,和上面试出来的 1.24、−0.64 对得上(我们的格子只有 0.02 那么细)。平均罚分 0.370。基线模型(对每位患者都说 23.6%,也就是总体的比例)是 0.547: 用上"超出的剂量"以后,罚分少了三分之一。

`C=1e6` 是什么: scikit-learn 的逻辑回归**默认给系数加了一道约束**,让系数往 0 靠一些(第 6 课讲它的用处)。`C` 越大,约束越松;`C=1e6` 相当于不加约束,得到的才是统计软件里那种逻辑回归。不写 `C`,这里会得到 1.22 和 −0.64,和上面略有出入;变量一多,出入会更大。要报告 OR,就要写上 `C=1e6`。

平均罚分的英文名是 **log loss**,也叫交叉熵(cross-entropy)。

<div class="exercise">

**练习 1 · 轮到你: 手算 Graphical 的 OR**

换成 Graphical: 超出不到 2 Gy 的 27 位患者里,19 位降到了 68 Gy 以下;超出 2 Gy 及以上的 117 位里,32 位降下来了。算两组的 odds(存进 `odds_near`、`odds_far`)和 OR(`or_hand`,近的一组比远的一组)。

```python
odds_near = 19 / ___          # 超出不到 2 Gy: 19 位降下来了,几位没有?
odds_far = 32 / ___           # 超出 2 Gy 及以上
or_hand = odds_near / odds_far
print(f"odds: {odds_near:.2f} 和 {odds_far:.2f};OR = {or_hand:.2f}")
```

```pycheck
need("odds_near", "odds_far", "or_hand")
if close(odds_near, 19 / 27, 0.005):
    fail("19 ÷ 27 是概率(0.70)。odds 是发生的除以没发生的: 27 位里 19 位降下来了,没降下来的是 8 位。")
if close(or_hand, 2.573, 0.01):
    fail("2.57 是两个概率的比(0.70 ÷ 0.27)。odds 的分母是没发生的人数: 8 和 85。")
if close(odds_near, 2.375, 0.005) and close(odds_far, 0.3765, 0.005) and close(or_hand, 6.31, 0.02):
    ok("对了: odds 是 2.38 和 0.38,OR = 6.31。两组的概率是 70% 和 27%,只差 2.6 倍,OR 却是 6.3: 事件常见的时候,OR 比概率的倍数大得多。")
fail(f"现在 odds 是 {odds_near:.2f} 和 {odds_far:.2f},应该是 2.38(19 ÷ 8)和 0.38(32 ÷ 85)。")
```

<details><summary>提示</summary>

odds = 发生的人数 ÷ 没发生的人数。27 位里 19 位发生,没发生的是 27 − 19 = 8 位;117 位里 32 位发生,没发生的是 85 位。

</details>

<details><summary>参考答案</summary>

```python
odds_near = 19 / 8
odds_far = 32 / 85
or_hand = odds_near / odds_far
print(f"odds: {odds_near:.2f} 和 {odds_far:.2f};OR = {or_hand:.2f}")
```

</details>

</div>

## 5. 读系数: 系数怎么变成 OR,概率怎么算

系数 b₁ = −0.65 的单位是 log-odds: 5 mm 计划每多超出 1 Gy,降下来的 log-odds 减少 0.65。log-odds 不好懂,换成 odds: 减去 0.65 个 log-odds,等于 odds 乘以 e<sup>−0.65</sup>。

```python
print(f"OR = exp({b1:.3f}) = {np.exp(b1):.2f}")

for excess in [0, 1]:
    log_odds = b0 + b1 * excess
    p = 1 / (1 + np.exp(-log_odds))
    print(f"超出 {excess} Gy: log-odds {log_odds:+.2f},概率 {p:.3f},odds {p / (1 - p):.2f}")
```

OR = 0.52: **每多超出 1 Gy,降到 68 Gy 以下的 odds 乘以 0.52**,差不多减半。下面两行验证了这一点: 超出 0 Gy 时 odds 是 3.49,超出 1 Gy 时是 1.83,1.83 ÷ 3.49 = 0.52。论文里的 OR,就是系数取 e 的指数。

注意概率不是减半: 从 0.777 到 0.646。OR 说的是 odds 的倍数,第 2 节的提醒在这里一样适用。

给新患者算概率,用 `predict_proba`:

```python
new = pd.DataFrame({"excess": [0.0, 2.0, 5.0]})
print("概率:", model.predict_proba(new)[:, 1].round(2))      # 每行两个数: 不发生的概率,发生的概率;取后一个
print("predict:", model.predict(new))
```

超出 0、2、5 Gy 的三位患者,降到 68 Gy 以下的概率是 0.78、0.49、0.12。`predict` 给的是 1、0、0: 它只是把概率在 0.5 处切了一刀。0.49 和 0.12 都成了"0",可是对患者来说,"一半的机会"和"八分之一的机会"是两回事。**逻辑回归真正的输出是概率**,要用 `predict_proba`。

<div class="quiz">
<p><b>想一想</b>: 一位患者超出 3 Gy,另一位超出 5 Gy。模型说后者降到 68 Gy 以下的 odds 是前者的多少?</p>
<button class="quiz-opt" data-why="0.52 是每多 1 Gy 的倍数。差 2 Gy,要乘两次。">0.52 倍</button>
<button class="quiz-opt" data-correct data-why="每多 1 Gy 乘以 0.52,多 2 Gy 就乘两次: 0.52 × 0.52 = 0.27。OR 是连乘的,不是相加的;在 log-odds 上才是相加(−0.65 × 2)。">0.27 倍(0.52 × 0.52)</button>
<button class="quiz-opt" data-why="倍数不能相加。每多 1 Gy 是'乘以 0.52',多 2 Gy 是乘两次。">1.04 倍(0.52 + 0.52)</button>
</div>

<div class="exercise">

**练习 2 · 轮到你: Graphical 的逻辑回归**

先定义 Graphical 的结局 `event_gra`,再拟合逻辑回归(记得 `C=1e6`),把每多超出 1 Gy 的 OR 存进 `or_gra`。

```python
hi["event_gra"] = (hi["d2cc_gra"] < 68).astype(int)
model_gra = LogisticRegression(C=___).fit(hi[["excess"]], hi[___])
or_gra = np.exp(model_gra.coef_[0][0])
print(f"log-odds = {model_gra.intercept_[0]:.2f} {model_gra.coef_[0][0]:+.2f} × 超出的剂量;OR = {or_gra:.2f}")
```

```pycheck
need("model_gra", "or_gra")
if close(or_gra, 0.5238, 0.002):
    fail('这是 Volume 的 OR。第二个空要填 Graphical 的结局: "event_gra"。')
if close(or_gra, 0.6106, 0.0008):
    fail("很接近,但这是带着默认约束(C=1)的结果。第一个空填 1e6。")
if close(or_gra, 0.6083, 0.0012):
    ok("对了: log-odds = 1.62 − 0.50 × 超出的剂量,OR = 0.61。换成 Graphical 时,每多超出 1 Gy,降到 68 Gy 以下的 odds 乘以 0.61;Volume 是 0.52。Graphical 降得更多,超得多一些也还有机会。")
fail(f"or_gra 现在是 {or_gra:.3f},应该是 0.608。")
```

<details><summary>提示</summary>

第一个空是 `1e6`,第二个空是列名 `"event_gra"`(带引号)。

</details>

<details><summary>参考答案</summary>

```python
hi["event_gra"] = (hi["d2cc_gra"] < 68).astype(int)
model_gra = LogisticRegression(C=1e6).fit(hi[["excess"]], hi["event_gra"])
or_gra = np.exp(model_gra.coef_[0][0])
print(f"log-odds = {model_gra.intercept_[0]:.2f} {model_gra.coef_[0][0]:+.2f} × 超出的剂量;OR = {or_gra:.2f}")
```

</details>

</div>

<div class="exercise">

**练习 3 · 轮到你: 一位新患者的概率**

一位患者 5 mm 计划的 D2cc 是 70 Gy(超出 2 Gy)。用 `model_gra` 算 Graphical 后降到 68 Gy 以下的概率,存进 `p_gra`。

```python
one = pd.DataFrame({"excess": [___]})
p_gra = model_gra.___(one)[0, 1]
print(f"概率 {p_gra:.2f}")
```

```pycheck
need("p_gra")
if close(p_gra, 0.489, 0.004):
    fail("这是 Volume 的概率。要用 model_gra。")
if close(p_gra, 0.349, 0.004):
    fail("这是'降不下来'的概率: 取的是第 0 个数。发生的概率是 [0, 1]。")
if close(p_gra, 0.6506, 0.004):
    ok("对了: 0.65。同样超出 2 Gy,Volume 降下来的概率是 0.49,Graphical 是 0.65。")
fail(f"p_gra 现在是 {p_gra},应该是 0.65。第一个空是 2.0;第二个空是 predict_proba(predict 只给 0 或 1)。")
```

<details><summary>提示</summary>

第一个空是超出的剂量 `2.0`;第二个空是给出概率的那个方法 `predict_proba`。

</details>

<details><summary>参考答案</summary>

```python
one = pd.DataFrame({"excess": [2.0]})
p_gra = model_gra.predict_proba(one)[0, 1]
print(f"概率 {p_gra:.2f}")
```

</details>

</div>

## 6. 不止一个变量

和第 1 课一样,公式里可以再加变量。第 1 课看到,阴道壁越薄,换成 Volume 后 D2cc 降得越多。把平均壁厚加进来:

```python
X2 = hi[["excess", "wall_mean"]]
model2 = LogisticRegression(C=1e6).fit(X2, hi["event"])
print(f"log-odds = {model2.intercept_[0]:.2f} {model2.coef_[0][0]:+.2f} × 超出的剂量 {model2.coef_[0][1]:+.2f} × 壁厚")
print("OR:", np.exp(model2.coef_[0]).round(2))

two = pd.DataFrame({"excess": [2.0, 2.0], "wall_mean": [3.0, 5.0]})
print("同样超出 2 Gy,壁厚 3 mm 和 5 mm 的概率:", model2.predict_proba(two)[:, 1].round(2))
```

公式是 log-odds = 7.24 − 0.83 × 超出的剂量 − 1.59 × 壁厚。两个 OR 是 0.44 和 0.20,读法和第 1 课一样,要带上"其他变量不变时":

- 壁厚相同时,5 mm 计划每多超出 1 Gy,降下来的 odds 乘以 0.44。
- 超出的剂量相同时,壁厚每厚 1 mm,降下来的 odds 乘以 0.20。

同样超出 2 Gy,壁厚 3 mm 的患者降下来的概率是 0.69,壁厚 5 mm 的只有 0.08。只用"超出的剂量"时,这两位患者得到的是同一个数(0.49)。多一个有用的变量,模型就能把他们分开。

<div class="exercise">

**练习 4 · 轮到你: Graphical 的两变量模型**

用超出的剂量和平均壁厚预测 `event_gra`,把壁厚的 OR 存进 `or_wall`。

```python
model2_gra = LogisticRegression(C=1e6).fit(X2, hi[___])
or_wall = np.exp(model2_gra.coef_[0][___])
print("OR:", np.exp(model2_gra.coef_[0]).round(2), " 壁厚的 OR:", round(or_wall, 2))
```

```pycheck
need("model2_gra", "or_wall")
if close(or_wall, 0.203, 0.004):
    fail('这是 Volume 的结果。第一个空填 "event_gra"。')
if close(or_wall, 0.5626, 0.004):
    fail("这是超出的剂量的 OR(第 0 个)。X2 的两列依次是超出的剂量、壁厚,壁厚是第 1 个。")
if close(or_wall, -1.0986, 0.01):
    fail("这是系数(log-odds),还没有换成 OR: 要取 np.exp。起始代码里已经有 np.exp,检查有没有删掉。")
if close(or_wall, 0.3333, 0.004):
    ok("对了: 0.33。超出的剂量相同时,壁厚每厚 1 mm,Graphical 后降到 68 Gy 以下的 odds 乘以 0.33,只剩三分之一。")
fail(f"or_wall 现在是 {or_wall:.3f},应该是 0.33。")
```

<details><summary>提示</summary>

第一个空是 `"event_gra"`;第二个空是壁厚在 `X2` 里的位置,从 0 开始数是 `1`。

</details>

<details><summary>参考答案</summary>

```python
model2_gra = LogisticRegression(C=1e6).fit(X2, hi["event_gra"])
or_wall = np.exp(model2_gra.coef_[0][1])
print("OR:", np.exp(model2_gra.coef_[0]).round(2), " 壁厚的 OR:", round(or_wall, 2))
```

</details>

</div>

## 7. 模型给的是概率,不是结论

逻辑回归到概率为止。"概率 0.49,换不换方案?"不是模型能回答的: 它取决于换方案的代价和没降下来的后果。把概率在某处切一刀变成"是/否",是另外一个决定,0.5 也不是天然的界线。

这带出后面三个问题,各占一课:

- 模型给高概率的患者,真的更常降下来吗?能不能把会降下来的和不会的分开: **区分度**,第 10 课。
- 模型说 0.49 的那些患者里,真的有一半左右降下来了吗: **校准**,第 11 课。
- 按模型的概率做决定,比对所有人都换方案、或者都不换,更好吗: **决策曲线**,第 12 课。

这一课先用平均罚分给模型打一个总分。scikit-learn 里它叫 `log_loss`:

<div class="exercise">

**练习 5 · 轮到你: Graphical 的模型比基线好多少**

算 `model_gra`(只用超出的剂量)在这 144 位患者身上的平均罚分,存进 `loss_gra`;再算基线(对每位患者都说总体的比例)的,存进 `loss_base`。

```python
from sklearn.metrics import log_loss

p_model = model_gra.predict_proba(hi[["excess"]])[:, 1]
loss_gra = log_loss(hi["event_gra"], ___)
p_base = np.full(len(hi), hi["event_gra"].___())        # 每个人都是同一个数: 总体的比例
loss_base = log_loss(hi["event_gra"], p_base)
print(f"基线 {loss_base:.3f},模型 {loss_gra:.3f}")
```

```pycheck
need("loss_gra", "loss_base")
if close(loss_gra, 0.3704, 0.003):
    fail("0.370 是 Volume 的模型。p_model 要用 model_gra 来算(起始代码里已经是),检查有没有改动。")
if close(loss_base, 0.693, 0.003):
    fail('0.693 是对每个人都说 50% 的罚分。基线用的是总体的比例: hi["event_gra"].mean()。')
if close(loss_gra, 0.4709, 0.003) and close(loss_base, 0.6500, 0.003):
    ok("对了: 基线 0.650,模型 0.471。Graphical 的事件更多(35%),基线本身的罚分就高;用上超出的剂量,降了四分之一多。和第 1 课一样,这是在建模患者身上的成绩,第 4 课换成诚实的版本。")
fail(f"现在是基线 {loss_base:.3f}、模型 {loss_gra:.3f},应该是 0.650 和 0.471。第一个空是 p_model,第二个空是 mean。")
```

<details><summary>提示</summary>

`log_loss(实际的结局, 模型给的概率)`: 第一个空是 `p_model`。总体的比例就是这一列 0 和 1 的平均值: `mean`。

</details>

<details><summary>参考答案</summary>

```python
from sklearn.metrics import log_loss

p_model = model_gra.predict_proba(hi[["excess"]])[:, 1]
loss_gra = log_loss(hi["event_gra"], p_model)
p_base = np.full(len(hi), hi["event_gra"].mean())
loss_base = log_loss(hi["event_gra"], p_base)
print(f"基线 {loss_base:.3f},模型 {loss_gra:.3f}")
```

</details>

</div>

## 小结

- 结局是是/否时,直线会给出 0 以下、1 以上的"概率"。逻辑回归让直线预测 **log-odds**,再换回概率,得到一条 S 形曲线。
- **odds** = 发生的概率 ÷ 不发生的概率。**OR** 是两个 odds 的比,不是两个概率的比;事件越常见,两者差得越多。
- 尺子是**平均罚分**(log loss): 模型给实际结果的概率越低,罚得越重。让它最小,就是**最大似然**。
- 系数取 e 的指数就是 OR: 其他变量不变时,这个变量每增加 1 个单位,odds 乘以多少。
- 模型的输出是**概率**(`predict_proba`)。`predict` 只是在 0.5 处切了一刀。
- scikit-learn 默认给系数加约束;要得到统计软件里的逻辑回归,写 `C=1e6`。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 逻辑回归 | logistic regression | 用直线预测 log-odds,再换回概率的模型 |
| 概率 | probability | 这样的患者里有多大比例发生 |
| 比值 | odds | 发生的概率除以不发生的概率 |
| 对数比值 | log-odds (logit) | odds 的自然对数,没有上下限 |
| 比值比 | odds ratio (OR) | 两个 odds 相除 |
| 最大似然 | maximum likelihood | 取让实际结局在模型眼里最可能的那组系数 |
| 对数损失 | log loss, cross-entropy | 平均罚分: −ln(给实际结果的概率) 的平均 |
| 预测概率 | predicted probability | 模型对一位患者给出的概率 |
| 二分类结局 | binary outcome | 只有是和否两种取值的结局 |

## 这一课没讲什么

- **OR 的置信区间和 P 值**: 第 3 课。
- **事件太少**: 144 位患者里 34 个事件,能撑得起几个变量?第 13 课。
- **完全分离**: 某个变量把是和否分得一干二净时,最大似然的系数会冲向无穷大,程序会给出很大的数或警告。约束(第 6 课)能解决。
- **多于两类的结局**(比如副作用分 0、1、2、3 级): 要用有序或多分类的逻辑回归,这门课不讲。
- **随访时间不等的结局**: 有的患者随访 1 年,有的 5 年,"有没有发生"就不公平了。那是 Kaplan–Meier 和 Cox 回归的范围,这门课不讲。

## 写进论文 Writing it up

<div class="paper">

方法部分可以这样写(变量换成你实际用的):

> The probability that the Volume plan brought the vaginal D2cc below 68 Gy was modelled by logistic regression, fitted by maximum likelihood without penalisation, with the excess of the 5 mm plan D2cc over 68 Gy and mean vaginal wall thickness as predictors. Results are reported as odds ratios per unit increase of each predictor.

**审稿人可能问**: OR = 0.44 是不是说概率降低了 56%?

**怎么回答**: 不是。OR 是 odds 的倍数。可以补上几位有代表性的患者的预测概率(比如第 6 节的 0.69 和 0.08),读者更容易理解效应的大小。

**审稿人可能问**: 软件用的是哪种拟合?有没有加正则化?

**怎么回答**: 用 scikit-learn 时要明说。默认设置带 L2 约束,得到的系数比最大似然的小;报告 OR 时用的是不加约束的拟合(`C=1e6`)。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| 系数和统计软件(R、SPSS)算的不一样 | 多半是没写 `C=1e6`,带着默认的约束。 |
| `ConvergenceWarning: lbfgs failed to converge` | 逼近的步数不够。加上 `max_iter=1000`;变量的大小相差悬殊时先标准化(第 5 课)。 |
| `ValueError: This solver needs samples of at least 2 classes` | 结局那一列全是 0 或者全是 1。检查结局是怎么定义的。 |
| `predict_proba` 的结果有两列 | 第 0 列是"不发生"的概率,第 1 列是"发生"的概率。取 `[:, 1]`。 |
| 结果和网页上写的不一样 | 本页的代码块要从上往下依次运行;练习 3–5 要先通过练习 2。 |

## 延伸阅读 Further reading

- scikit-learn 用户指南: [Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)。`LogisticRegression` 的说明,包括 `C` 的含义。
- Sperandei S. [Understanding logistic regression analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC3936971/). *Biochem Med (Zagreb)* 2014;24(1):12–18。写给临床研究者的逻辑回归入门,可以引用。
- Qiang Y, 等. D90 and D2cc dose in vaginal CTV as dosimetric predictors of late vaginal side effects in postoperative vaginal modern brachytherapy after external beam irradiation in endometrial cancer. *Cancers* 2026;18(14):2350。第 2 节那张 68 Gy 的表的出处。

**下一课**: 两课下来,同一个模型已经有了两种读法: 系数和 OR 说的是"这个变量和结局有什么关系",预测值和概率说的是"这位患者会怎样"。第 3 课把这两种用法摆在一起: 回归表里的区间和 P 值回答的是哪一个问题,它们为什么不能代替"预测得准不准"。
