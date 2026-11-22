---
title: "Lesson 6 · 检验和相关 Tests and Correlation"
summary: "P 值是什么意思;两组比较、配对比较、四格表;相关系数;检验做多了要校正。"
weight: 60
toc: true
duration: "约 90 分钟,可以分三次学: 第 1–2 节,第 3–4 节,第 5–7 节"
prereq: "[第 5 课](../05-uncertainty/)"
questions:
  - "两组之间的差别,会不会只是碰巧?P 值到底在说什么?"
  - "什么时候用哪一种检验?"
  - "两个量一起变,怎么量?检验做了很多个,又该怎么办?"
objectives:
  - "用'打乱分组'解释 P 值,并自己写代码算一个"
  - "根据数据的类型和设计,选择 Mann–Whitney、Wilcoxon 符号秩、卡方或 Fisher 检验"
  - "说明配对的数据为什么要用配对的检验"
  - "计算并解释 Pearson 和 Spearman 相关系数"
  - "对多个检验的 P 值做 Holm 校正"
  - "指出 P 值不能说明什么"
---

第 5 课给每个结果配上了区间。这一课问另一个问题: 两组之间看到的差别,两个量之间看到的关系,**会不会只是碰巧**?

回答这个问题的工具叫假设检验,它的产物就是论文里到处都是的 **P 值**。P 值大概是医学文献里被误读得最多的一个数。这一课不从公式讲起,而是让你先亲手"打乱"一次分组,看看碰巧能造出多大的差别;然后再讲临床研究里最常用的几种检验,各自在什么时候用。

> 🔬 第 4 节的例题是一张来自已发表论文的四格表: 我们用 Python 算出的 P 值,和论文里印的一模一样。三套计划两两比较再做校正(第 6 节),是我们自己的研究里的固定做法。

**练习怎么做**: 讲解里比的是 **Volume** 和阴道 D2cc,练习换成 **Graphical** 和靶区 D90。

## 0. 准备

```python
import numpy as np
import pandas as pd
from scipy import stats
from cohort import make_cohort, make_followup

df = make_cohort()
df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
df["delta_vol"] = df["d2cc_vol"] - df["d2cc_5mm"]
df["delta_gra"] = df["d2cc_gra"] - df["d2cc_5mm"]
merged = df.merge(make_followup(), on="id", how="left")
merged["high_5mm"] = merged["d2cc_5mm"] > 68
print(df.shape, merged.shape)
```

第一次运行要下载 SciPy(科学计算的包,各种检验都在它的 `stats` 里),会多等一会儿。

## 1. P 值是什么意思: 把分组打乱

<div class="frame">

**背景**: 两组患者,一个指标的平均值差了一截。第 5 课说过,只要是抽到的一批患者,任何数都会晃动。

**目标**: 判断这一截差别是真的有,还是晃动就能晃出来。

**做法**: **打乱分组**(置换检验, permutation test): 假装分组和结果毫无关系,看看在这种情况下,碰巧能出现多大的差别。

</div>

**它在算什么**

1. 算出实际的两组平均值之差。
2. 假设"分组和结果没有关系"(这叫无效假设, null hypothesis)。如果真是这样,组别只是贴在患者身上的标签,换一种贴法也无所谓。于是把标签**随机打乱**重新贴一遍。
3. 用打乱后的分组,再算一次两组之差。
4. 把打乱和再算这两步重复几千次,得到"纯粹碰巧时"差别的分布。
5. 数一数: 有多少次打乱出来的差别(不管正负),不小于实际的差别。这个比例就是 **P 值**。

所以 P 值回答的问题是: **如果分组和结果其实没有关系,像现在这么大(或者更大)的差别,有多容易碰上?** P 值小,说明"只是碰巧"这个解释很勉强。

<div class="demo">

**设定**: 40 位模拟患者,单纯近距离和外照射加近距离各 20 位。上面一行实心点是单纯近距离组,下面一行空心点是外照射加近距离组,横轴是换方案后的 Δ,蓝色短线是每组的平均值。实际的两组平均值相差 1.79 Gy。

**你做什么**: 点几次"打乱 1 次",看患者怎样被重新分到两行,两组的差变成多少;再点"打乱 200 次"。

**你会看到**: 打乱以后,两组的差大多只有零点几 Gy(前 5 次是 0.18、0.65、0.78、0.06、0.97)。打乱 205 次,只有 6 次不小于 1.79 Gy;一直打乱到 2000 次,是 65 次,占 0.033。

**这说明**: 如果分组和 Δ 没有关系,1.79 Gy 这么大的差别,100 次里大约只会碰上 3 次。这个 0.03 就是 P 值。

<div class="widget" data-widget="shuffle">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

**约定**: P 小于 0.05,习惯上说差别"有统计学意义"(statistically significant)。0.05 只是一条约定俗成的线,第 4 节和第 7 节会看到,不能把它当成"真"和"假"的分界。

<div class="quiz">
<p><b>想一想</b>: 上面的图得到 P ≈ 0.03。下面哪句话说对了?</p>
<button class="quiz-opt" data-why="P 值不是'没有差别'这件事的概率。它是在假设没有差别的前提下算出来的,回答不了这个前提本身有多大可能成立。这是对 P 值最常见的误读。">两组其实没有差别的可能性是 3%</button>
<button class="quiz-opt" data-correct data-why="这正是打乱分组算出来的东西: 先假设没有关系,再看这么大的差别有多罕见。">如果分组和 Δ 没有关系,这么大或更大的差别,碰巧出现的机会大约是 3%</button>
<button class="quiz-opt" data-why="P 值不是结论正确的概率,也不说明差别有多大、临床上重不重要。">这个结论有 97% 的把握是对的</button>
</div>

把这 5 步写成代码,用在课程的队列上。问题: 两种治疗方式的患者,BMI 有没有差别?

```python
def perm_test(values, in_group, n=2000, seed=1):
    """打乱分组的检验。values: 每位患者的数;in_group: 每位患者是否在第一组(True 或 False)。"""
    values = np.asarray(values, dtype=float)
    in_group = np.asarray(in_group)
    observed = values[in_group].mean() - values[~in_group].mean()     # 第 1 步: 实际的差
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n):                                                  # 第 4 步: 重复 n 次
        shuffled = rng.permutation(in_group)                            # 第 2 步: 把组别打乱
        diff = values[shuffled].mean() - values[~shuffled].mean()       # 第 3 步: 打乱后的差
        if abs(diff) >= abs(observed):
            count += 1
    return observed, count / n                                          # 第 5 步: 这个比例就是 P

is_ebrt = df["arm"] == "EBRT+VCB"
diff, p = perm_test(df["bmi"], is_ebrt)
print(f"BMI: 外照射加近距离组比单纯近距离组 {diff:+.2f},P = {p:.3f}")
```

`~in_group` 里的 `~` 是"取反": True 变 False,False 变 True,所以 `values[~in_group]` 是不在第一组的患者。`rng.permutation(...)` 把一串东西随机打乱。

常用的软件(比如 SciPy 的 `stats.permutation_test`)算 P 时,分子和分母各加 1: (次数 + 1) ÷ (打乱次数 + 1)。实际的分组本身也是一种分法,算进去以后,P 就不会是 0。打乱 2000 次时,两种算法只差万分之几。

外照射加近距离组的 BMI 平均低 0.81,P = 0.31: 把组别打乱 2000 次,有三成的时候都能碰上这么大的差别。数据并不支持"两组的 BMI 不同"。在这个模拟里我们知道真相: 两组的 BMI 是按同一个规律产生的,0.81 的差别纯属碰巧。

但是要小心这句话的另一面: **P = 0.31 并不证明两组相同**。它只是说,这批数据不足以看出差别。第 7 节再谈。

<div class="exercise">

**练习 1 · 轮到你: 两组的壁厚**

两种治疗方式的患者,平均壁厚(`wall_mean`)有没有差别?先去掉没有壁厚测量的患者,再用 `perm_test`,把差值和 P 值存进 `diff_wall` 和 `p_wall`。

```python
d = df.dropna(subset=[___])
diff_wall, p_wall = perm_test(d["wall_mean"], d["arm"] == ___)
print(f"壁厚: 相差 {diff_wall:+.2f} mm,P = {p_wall:.3f}")
```

```pycheck
need("diff_wall", "p_wall")
if diff_wall != diff_wall:
    fail('结果是 NaN: 数据里还有缺失值。dropna(subset=["wall_mean"]) 先去掉没有测量的患者。')
if close(diff_wall, -0.142, 0.002):
    fail('差值的符号反了: 第二处填 "EBRT+VCB",让外照射加近距离组做第一组。')
if close(diff_wall, 0.142, 0.002) and close(p_wall, 0.36, 0.004):
    ok("对了: 外照射加近距离组的平均壁厚多 0.14 mm,P = 0.36。打乱分组,三成多的时候都能碰上这么大的差别,所以谈不上两组不同。")
fail(f"现在 diff_wall = {diff_wall:.3f},p_wall = {p_wall:.3f},应该是 0.142 和 0.36。")
```

<details><summary>提示</summary>

`subset=["wall_mean"]`;第一组是 `"EBRT+VCB"`。

</details>

<details><summary>参考答案</summary>

```python
d = df.dropna(subset=["wall_mean"])
diff_wall, p_wall = perm_test(d["wall_mean"], d["arm"] == "EBRT+VCB")
print(f"壁厚: 相差 {diff_wall:+.2f} mm,P = {p_wall:.3f}")
```

</details>

</div>

## 2. 两组不相干的患者: Mann–Whitney 检验

打乱分组什么情况都能用,只是要自己写循环。常见的情况都有现成的检验,原理相同: 先假设"没有差别",再算数据在这个假设下有多罕见。区别只在于用公式代替了几千次打乱。

两组**不同的**患者、比一个连续的指标,最常用的两个:

- **Mann–Whitney U 检验**(也叫 Wilcoxon 秩和检验): 把两组患者放在一起排名次,比较两组名次的高低。只用名次,所以不怕长尾和极端值。
- **t 检验**: 比较两组的均值。数据大致对称、没有极端值时用。

```python
a = df.loc[is_ebrt, "bmi"]
b = df.loc[~is_ebrt, "bmi"]
print(f"BMI   Mann–Whitney: P = {stats.mannwhitneyu(a, b).pvalue:.3f}")
print(f"BMI   t 检验:        P = {stats.ttest_ind(a, b, equal_var=False).pvalue:.3f}")

a = df.loc[is_ebrt, "d2cc_5mm"]
b = df.loc[~is_ebrt, "d2cc_5mm"]
print(f"D2cc  Mann–Whitney: P = {stats.mannwhitneyu(a, b).pvalue:.2g}")
```

BMI 的两个 P 值是 0.25 和 0.31,和打乱分组得到的 0.31 是一回事。5 mm 计划的阴道 D2cc 就不一样了: 外照射加近距离组的中位数是 73.0 Gy,单纯近距离组是 69.4 Gy(第 3 课),P = 1.6e-07,也就是 0.00000016。这种写法叫科学计数法,`e-07` 表示小数点往左移 7 位。论文里写成 *P* < 0.001 就可以了。

`equal_var=False` 让 t 检验不假设两组的离散程度相同(叫 Welch t 检验),平时都用这个写法。

**用哪一个**: 数据有长尾、有极端值,或者患者不多、看不清分布的形状,用 Mann–Whitney;它和"报中位数"是一套。数据对称时两者结论一样。拿不准就用 Mann–Whitney。

## 3. 同一批患者比两次: 配对的检验

<div class="frame">

**背景**: 这门课的主线不是两组患者,而是**同一位患者**的两套计划。第 3 课和第 4 课都说过: 成对的数据,要把"对"用上。

**目标**: 判断同一批患者身上,换计划前后的差别是不是碰巧。

**做法**: **Wilcoxon 符号秩检验**(Wilcoxon signed-rank test): 对每位患者自己的差值做检验。

</div>

**它在算什么**

1. 每位患者算一个差值: Volume 减 5 mm。
2. 不管正负,把差值的大小排名次: 差得最小的是第 1 名。
3. 把正的差值的名次加起来,负的差值的名次也加起来。
4. 如果换计划其实没有影响,差值应该有正有负、大小相当,两个名次和差不多。两个和差得越远,P 值越小。

**为什么不能用上一节的检验**: 患者和患者之间本来就差得很远,有人的 D2cc 是 60 Gy,有人是 80 Gy。把两套计划当成两组不相干的患者来比,计划之间 2 Gy 的差别,会被患者之间十几 Gy 的差别淹没。让每位患者自己和自己比,患者之间的差别就抵消了。

**打个比方**: 评价一种降压药,给同一批人量服药前后的血压,看每个人降了多少。不会有人把"服药前"和"服药后"当成两批不同的人去比。

只用前 30 位患者,两种做法的差距看得最清楚:

```python
first30 = df.head(30)
paired = stats.wilcoxon(first30["d2cc_vol"], first30["d2cc_5mm"])
unpaired = stats.mannwhitneyu(first30["d2cc_vol"], first30["d2cc_5mm"])
print(f"配对的检验(Wilcoxon 符号秩):        P = {paired.pvalue:.4f}")
print(f"当成两组不相干的患者(Mann–Whitney): P = {unpaired.pvalue:.2f}")
```

同样的 30 位患者、同样的数据: 配对的检验 P = 0.0003,当成两组来比 P = 0.14。前一个结论是"Volume 确实降低了 D2cc",后一个是"没看出差别"。数据没有变,变的只是有没有用上"这两个数来自同一位患者"。

全部 200 位患者:

```python
res = stats.wilcoxon(df["d2cc_vol"], df["d2cc_5mm"])
print(f"Volume − 5 mm: 中位数 {df['delta_vol'].median():.1f} Gy,P = {res.pvalue:.1g}")
```

<div class="quiz">
<p><b>想一想</b>: 一项研究给 40 位贫血的患者补铁,比较治疗前和治疗 3 个月后的血红蛋白。应该用哪种检验?</p>
<button class="quiz-opt" data-why="Mann–Whitney 用于两组不同的患者。这里是同一批患者量了两次,用它会丢掉配对的信息,本来能看出来的差别可能看不出来。">Mann–Whitney 检验: 治疗前一组,治疗后一组</button>
<button class="quiz-opt" data-correct data-why="同一位患者治疗前后各一个数,是配对的数据。对每位患者自己的变化做检验: Wilcoxon 符号秩检验,数据对称时也可以用配对 t 检验。">Wilcoxon 符号秩检验: 每位患者自己的前后变化</button>
<button class="quiz-opt" data-why="卡方检验用于分类变量(比如有没有副作用)的人数。血红蛋白是连续的数。">卡方检验</button>
</div>

<div class="exercise">

**练习 2 · 轮到你: Graphical 的配对检验**

用 Wilcoxon 符号秩检验比较 Graphical 和 5 mm 计划的阴道 D2cc(全部 200 位患者),把 P 值存进 `p_gra`。

```python
res_gra = stats.___(df["d2cc_gra"], df[___])
p_gra = res_gra.pvalue
print(f"Graphical − 5 mm: 中位数 {df['delta_gra'].median():.1f} Gy,P = {p_gra:.1g}")
```

```pycheck
need("p_gra")
if 5e-9 < p_gra < 2e-8:
    fail("这是 Mann–Whitney 的结果,把两套计划当成了两组不相干的患者。配对的检验是 stats.wilcoxon。")
if 3e-18 < p_gra < 1e-17:
    fail('这是 Volume 的结果。第一个参数已经是 d2cc_gra,第二处填 "d2cc_5mm"。')
if 5e-24 < p_gra < 1.5e-23:
    ok("对了: Graphical − 5 mm 的中位数是 −3.4 Gy,P = 9e-24,写进论文是 P < 0.001。200 位患者里这么一致的下降,不可能是碰巧。")
fail(f"现在 p_gra = {p_gra:.2g},应该在 9e-24 左右: stats.wilcoxon(df[\"d2cc_gra\"], df[\"d2cc_5mm\"])。")
```

<details><summary>提示</summary>

函数是 `stats.wilcoxon`,两个参数是同一批患者的两列。

</details>

<details><summary>参考答案</summary>

```python
res_gra = stats.wilcoxon(df["d2cc_gra"], df["d2cc_5mm"])
p_gra = res_gra.pvalue
print(f"Graphical − 5 mm: 中位数 {df['delta_gra'].median():.1f} Gy,P = {p_gra:.1g}")
```

</details>

</div>

## 4. 分类变量: 四格表和卡方检验

结局是"有/没有"时,数据是一张人数的表。先用一张真实的: 第 3 课见过的那篇论文(Qiang 等,*Cancers* 2026;18:2350)Table 1 里的四个数。外照射加近距离的 120 位患者,按阴道 D2cc 的总 EQD2 是否超过 68 Gy、有没有出现晚期阴道副作用,分成四格:

```python
paper = pd.DataFrame({"no_toxicity": [41, 45], "toxicity": [9, 25]},
                     index=["D2cc <= 68 Gy", "D2cc > 68 Gy"])
paper["toxicity_rate"] = paper["toxicity"] / (paper["no_toxicity"] + paper["toxicity"])
paper.round(3)
```

D2cc 不超过 68 Gy 的 50 位里,9 位出现副作用(18%);超过 68 Gy 的 70 位里,25 位出现(35.7%)。差了将近一倍。会不会是碰巧?

<div class="frame">

**背景**: 两组患者,各有一部分人出现了某个结局,比例不一样。

**目标**: 判断两个比例的差别是不是碰巧。

**做法**: **卡方检验**(chi-square test): 比较每一格实际的人数,和"两者没有关系"时期望的人数。

</div>

**它在算什么**

1. 假设副作用和 D2cc 没有关系,那么两组出现副作用的比例应该都等于总体的比例: 34 / 120 = 28.3%。
2. 按这个比例,算出每一格**期望**有多少人: 比如 D2cc ≤ 68 Gy 的 50 位里,期望有 50 × 28.3% = 14.2 位出现副作用,实际只有 9 位。
3. 每一格算"实际和期望差多少",四格合起来得到一个数(卡方值)。差得越多,这个数越大,P 值越小。

```python
table = paper[["no_toxicity", "toxicity"]]
res = stats.chi2_contingency(table)
print(f"卡方检验: P = {res.pvalue:.3f}")
print("两者没有关系时,每一格期望的人数:")
print(res.expected_freq.round(1))
```

P = 0.055。论文里这一行印的也是 0.055: 用 Python 和论文作者用的软件,从同样的四个数算出了同样的结果。

按 0.05 这条线,0.055 "没有统计学意义"。可是再算两种同样常用的做法:

```python
print(f"卡方检验,不做连续性校正: P = {stats.chi2_contingency(table, correction=False).pvalue:.3f}")
print(f"Fisher 精确检验:          P = {stats.fisher_exact(table).pvalue:.3f}")
```

0.034 和 0.041。**同一张表,三种都说得过去的算法,P 值落在了 0.05 的两边**。

- `chi2_contingency` 默认对四格表做"连续性校正"(Yates 校正),结果偏保守;`correction=False` 是不校正的卡方检验。
- **Fisher 精确检验**(Fisher's exact test)不靠近似,直接算。只要有一格的期望人数小于 5,卡方的近似就不可靠,这时用 Fisher。这张表期望人数最小的一格是 14.2,两种都可以。

这个例子的教训不是"该选哪一个",而是: **P 值在 0.05 附近时,不要让这条线替你下结论**。18% 对 35.7% 这个差别本身,比 P 值是 0.055 还是 0.041 重要;120 位患者、34 个事件,这个差别估计得还不够准。应该报告的是两个比例、它们的差和这个差的区间,再给出 P 值,让读者自己判断。

<div class="exercise">

**练习 3 · 轮到你: 模拟队列的四格表**

对课程的模拟队列做同样的事: 用 `merged`(第 0 节合并好的表),做"5 mm 计划的 D2cc 是否超过 68 Gy"(`high_5mm`)和"有没有晚期副作用"(`tox`)的四格表 `table_sim`,再做卡方检验,P 值存进 `p_sim`。

```python
table_sim = pd.crosstab(merged[___], merged["tox"])
p_sim = stats.___(table_sim).pvalue
print(table_sim)
print(f"P = {p_sim:.3f}")
```

```pycheck
need("table_sim", "p_sim")
if table_sim.shape != (2, 2):
    fail(f"table_sim 应该是 2 行 2 列,现在是 {table_sim.shape}。")
if int(table_sim.values.sum()) != 188:
    fail(f"表里应该是有随访的 188 位患者,现在是 {int(table_sim.values.sum())} 位。")
if close(p_sim, 0.0712, 0.001):
    fail("这是 Fisher 精确检验的结果。题目要的是卡方检验: stats.chi2_contingency。")
if close(p_sim, 0.0761, 0.001):
    ok("对了: 超过 68 Gy 的 139 位里 36 位有副作用(25.9%),68 Gy 及以下的 49 位里 6 位(12.2%),P = 0.076。和论文里那张表很像: 比例差了一倍,P 值却在 0.05 附近。不做连续性校正是 0.048。结论不该取决于选了哪一种算法。")
fail(f"现在 p_sim = {p_sim:.4f},应该是 0.076。crosstab 的第一个变量是 \"high_5mm\"。")
```

<details><summary>提示</summary>

`pd.crosstab(merged["high_5mm"], merged["tox"])`;卡方检验是 `stats.chi2_contingency`。没有随访的 12 位患者,`crosstab` 会自动跳过。

</details>

<details><summary>参考答案</summary>

```python
table_sim = pd.crosstab(merged["high_5mm"], merged["tox"])
p_sim = stats.chi2_contingency(table_sim).pvalue
print(table_sim)
print(f"P = {p_sim:.3f}")
```

</details>

</div>

## 5. 两个量一起变: 相关系数

第 4 课的散点图上,壁越厚,Δ 越靠上。**相关系数**(correlation coefficient)用一个 −1 到 1 之间的数来量这种关系: 1 是完全同升同降,−1 是完全此升彼降,0 是看不出关系。常用的有两个:

- **Pearson 相关系数** *r*: 量的是"点有多贴近一条直线"。
- **Spearman 相关系数** *ρ*: 先把两个量各自换成名次,再算名次之间的 Pearson 相关。量的是"一个升高时,另一个是不是也跟着升高",不要求是直线,也不怕极端值。

```python
d = df.dropna(subset=["wall_mean"])                      # 相关系数的函数不接受缺失值
r = stats.pearsonr(d["wall_mean"], d["delta_vol"])
rho = stats.spearmanr(d["wall_mean"], d["delta_vol"])
print(f"Pearson r = {r.statistic:.2f},Spearman ρ = {rho.statistic:.2f}(P = {rho.pvalue:.1g})")
```

两个都在 0.6 左右: 壁越厚,Volume 后 D2cc 降得越少。这里的 P 值检验的是"相关系数其实是 0"。

两者的区别在有极端值时才显出来。假设队列里多了一位很特殊的患者: 壁厚 15 mm,Δ 却是 −9 Gy。

```python
x_out = pd.concat([d["wall_mean"], pd.Series([15.0])])
y_out = pd.concat([d["delta_vol"], pd.Series([-9.0])])
print(f"加上这一位: Pearson r = {stats.pearsonr(x_out, y_out).statistic:.2f},"
      f"Spearman ρ = {stats.spearmanr(x_out, y_out).statistic:.2f}")
```

197 位患者里只多了 1 位,Pearson 从 0.62 掉到 0.39;Spearman 只从 0.59 变到 0.57。Pearson 用的是数值本身,一个离得很远的点有很大的"杠杆";Spearman 只看名次,15 mm 和 7.5 mm 都只是"最厚的那一位"。

真实的数据里就可能有这样的患者。这是我们自己的研究用 Spearman 的原因,也是第 4 课说"先画散点图"的原因: 相关系数是一个数,藏得住一位特殊的患者,图藏不住。

**相关不等于因果**。壁厚和 Δ 相关,可以解释成"壁薄的患者,靶区离施源器近,调整的余地大";但相关系数本身不能证明这个解释。两个量都受第三个量影响时,它们也会相关。

<div class="exercise">

**练习 4 · 轮到你: 壁厚和 Graphical 的 Δ**

算平均壁厚和 Graphical 的 Δ(`delta_gra`)的 Spearman 相关系数,把系数存进 `rho_gra`,P 值存进 `p_rho`。用上面去掉了缺失值的 `d`。

```python
res_rho = stats.___(d["wall_mean"], d[___])
rho_gra, p_rho = res_rho.statistic, res_rho.pvalue
print(f"Spearman ρ = {rho_gra:.2f},P = {p_rho:.1g}")
```

```pycheck
need("rho_gra", "p_rho")
if rho_gra != rho_gra:
    fail("结果是 NaN: 用去掉了缺失值的 d,不要用 df。")
if close(rho_gra, 0.5230, 0.0005):
    fail("这是 Pearson 相关系数。题目要的是 Spearman: stats.spearmanr。")
if close(rho_gra, 0.5900, 0.0005):
    fail('这是 Volume 的结果: 第二处填 "delta_gra"。')
if close(rho_gra, 0.5253, 0.0005):
    ok("对了: ρ = 0.53,P = 3e-15。Graphical 也一样: 壁越厚,D2cc 降得越少。相关比 Volume 的 0.59 弱一点,因为 Graphical 的 Δ 本身散得更开。")
fail(f"现在 rho_gra = {rho_gra:.3f},应该是 0.525。")
```

<details><summary>提示</summary>

`stats.spearmanr(d["wall_mean"], d["delta_gra"])`。

</details>

<details><summary>参考答案</summary>

```python
res_rho = stats.spearmanr(d["wall_mean"], d["delta_gra"])
rho_gra, p_rho = res_rho.statistic, res_rho.pvalue
print(f"Spearman ρ = {rho_gra:.2f},P = {p_rho:.1g}")
```

</details>

</div>

## 6. 检验做多了: 多重比较

P < 0.05 的意思是: 没有关系时,碰巧出现的机会不到二十分之一。那么做 20 个检验呢?造 20 个和患者毫无关系的随机变量,逐个和 Δ 算相关:

```python
rng = np.random.default_rng(1)
p_values = np.array([stats.spearmanr(rng.normal(size=len(df)), df["delta_vol"]).pvalue for _ in range(20)])
print("20 个随机变量里,P < 0.05 的有", (p_values < 0.05).sum(), "个;最小的 P =", p_values.min().round(3))
```

20 个纯粹的随机数里,有 1 个"和 Δ 显著相关",P = 0.027。这不是运气不好,而是意料之中: 每个检验有 5% 的机会碰巧过线,做 20 个,平均就有 1 个过线。如果一项研究测了 20 个指标,只报告那个 P = 0.027 的,读者就被骗了。这个问题叫**多重比较**(multiple comparisons)。

<div class="frame">

**背景**: 一项分析里做了不止一个检验。检验越多,至少有一个碰巧过线的机会越大。

**目标**: 把"整组检验里至少有一个碰巧过线"的机会控制在 5% 以内。

**做法**: **Holm 校正**: 把 P 值按从小到大的顺序放大,最小的放大得最多。

</div>

**它在算什么**(m 个检验)

1. 把 m 个 P 值从小到大排好。
2. 最小的乘以 m,第二小的乘以 m − 1,第三小的乘以 m − 2……最大的乘以 1。
3. 保证顺序不乱: 如果某个校正后的值比它前面的还小,就取前面那个值。超过 1 的记成 1。
4. 校正后的 P 值再和 0.05 比。

最简单的校正(Bonferroni)是把每个 P 值都乘以 m。Holm 只对最小的那个这么严,后面的逐步放松,所以过线的结果不会比 Bonferroni 少,而且同样控制住了错误。

```python
def holm(p_values):
    """Holm 校正: 返回和输入顺序相同的校正后 P 值。"""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    largest = 0.0
    for rank, i in enumerate(np.argsort(p)):          # 从最小的 P 开始
        largest = max(largest, (m - rank) * p[i])     # 乘以 m、m − 1、…,并保证不比前一个小
        adjusted[i] = min(1.0, largest)
    return adjusted

print(holm([0.012, 0.030, 0.045]))
```

三个 P 值本来都小于 0.05。校正以后: 0.012 × 3 = 0.036,还过线;0.030 × 2 = 0.06 和 0.045 × 1(不能比前一个小,所以也是 0.06)都不过线了。

什么时候要校正: 同一个问题用了好几个检验来回答时。这门课的主线就是一个例子: 三套计划两两比较,是三个检验(Volume − 5 mm、Graphical − 5 mm、Graphical − Volume)。我们自己的研究里,这三个比较总是一起报告,P 值做 Holm 校正。

另一种常用的校正叫 **Benjamini–Hochberg**(BH),控制的是"过线的结果里有多少是假的"(错误发现率, false discovery rate),比 Holm 宽松,适合一次筛很多个指标的探索性分析。SciPy 里有现成的: `stats.false_discovery_control(p_values)`。

<div class="quiz">
<p><b>想一想</b>: 一篇论文比较了两组患者的 20 项化验指标,其中 1 项 P = 0.04,其余都大于 0.05。作者的结论是"该指标在两组间有显著差异"。怎么看?</p>
<button class="quiz-opt" data-why="单看这一项是过线了,但它是 20 个检验里挑出来的。没有任何真实差别时,20 个检验里平均也会有 1 个过线。">P &lt; 0.05,结论成立</button>
<button class="quiz-opt" data-correct data-why="20 个检验里有 1 个 P = 0.04,正是纯粹碰巧时预期的样子。校正以后(0.04 × 20)远远过不了线。除非这一项是事先指定的主要指标,否则只能当作有待验证的线索。">很可能是碰巧: 做了 20 个检验,应该做多重比较的校正</button>
<button class="quiz-opt" data-why="其余 19 项 P > 0.05 只说明没有检出差别,不能证明两组在这些指标上相同。">其余 19 项证明了两组在这些指标上没有差别</button>
</div>

<div class="exercise">

**练习 5 · 轮到你: 三套计划两两比较**

对靶区 D90 做三个配对比较(全部 200 位患者),得到三个 P 值,再做 Holm 校正。

```python
pairs = [("d90_vol", "d90_5mm"), ("d90_gra", "d90_5mm"), ("d90_gra", "d90_vol")]
p_raw = [stats.wilcoxon(df[a], df[___]).pvalue for a, b in pairs]
p_holm = holm(___)
for (a, b), p0, p1 in zip(pairs, p_raw, p_holm):
    print(f"{a} − {b}: 中位数 {(df[a] - df[b]).median():+.2f} Gy,P = {p0:.2g},Holm 校正后 {p1:.2g}")
```

```pycheck
need("p_raw", "p_holm")
if len(p_raw) != 3 or len(p_holm) != 3:
    fail("p_raw 和 p_holm 都应该有 3 个数。")
want_raw = [8.980834780082407e-21, 2.4564943610257053e-26, 8.191661197388221e-14]
if not all(abs(a / b - 1) < 1e-4 for a, b in zip(p_raw, want_raw)):
    fail("p_raw 不对: 每一对是 df[a] 和 df[b],空里填 b。")
want_holm = [2 * want_raw[0], 3 * want_raw[1], want_raw[2]]
if not all(abs(a / b - 1) < 1e-4 for a, b in zip(p_holm, want_holm)):
    fail("p_holm 不对: 把三个原始的 P 值交给 holm,也就是 holm(p_raw)。")
ok("对了。三个比较校正后仍然远小于 0.05: 最小的 P 乘了 3,第二小的乘了 2,最大的没有变。三套计划的 D90 两两都不同: Volume 比 5 mm 低 1.45 Gy,Graphical 比 5 mm 低 2.45 Gy,Graphical 比 Volume 又低 1.1 Gy(都是中位数)。")
```

<details><summary>提示</summary>

第一处填 `b`(每一对的第二列);第二处把 `p_raw` 交给 `holm`。

</details>

<details><summary>参考答案</summary>

```python
pairs = [("d90_vol", "d90_5mm"), ("d90_gra", "d90_5mm"), ("d90_gra", "d90_vol")]
p_raw = [stats.wilcoxon(df[a], df[b]).pvalue for a, b in pairs]
p_holm = holm(p_raw)
for (a, b), p0, p1 in zip(pairs, p_raw, p_holm):
    print(f"{a} − {b}: 中位数 {(df[a] - df[b]).median():+.2f} Gy,P = {p0:.2g},Holm 校正后 {p1:.2g}")
```

</details>

</div>

## 7. P 值说明不了什么

- **P 值小,不等于差别大**。患者足够多时,临床上毫无意义的微小差别也能得到很小的 P 值。差别有多大,看估计值和它的 95% 区间(第 5 课)。
- **P 值大,不等于没有差别**。"没有检出差别"可能只是患者太少、区间太宽。说两组"没有差别",需要一个很窄的、贴着 0 的区间,而不是一个大的 P 值。
- **0.05 不是真假的分界**。P = 0.049 和 P = 0.051 的证据几乎一样强。第 4 节那张表的三个 P 值是 0.055、0.034 和 0.041,数据是同一份。
- **P 值不是"结论正确的概率"**,也不是"没有差别的概率"(第 1 节的想一想)。
- **统计上显著,不等于临床上重要**。D2cc 降低 0.3 Gy 就算 P < 0.001,也改变不了任何治疗决定。多大的差别才有意义,是临床问题,要在看数据之前想好。

所以报告一个比较时,三样东西写在一起: **估计值、95% 区间、P 值**,比如"差值的中位数 −2.0 Gy(95% CI −2.4 到 −1.5;*P* < 0.001)"。

<div class="quiz">
<p><b>想一想</b>: 一项只有 24 位患者的研究比较两种计划方式的副作用,P = 0.40。作者写道: "两种方式的副作用没有差别。"这句话有什么问题?</p>
<button class="quiz-opt" data-why="P = 0.40 确实不小于 0.05,但'没有检出差别'和'没有差别'是两回事。">没有问题: P > 0.05 就是没有差别</button>
<button class="quiz-opt" data-correct data-why="24 位患者的区间会非常宽,很可能既盖得住'没有差别',也盖得住'差一倍'。应该写'没有检出差别',并给出差值的 95% 区间,让读者看到这个研究能排除什么、不能排除什么。">没有检出差别,不等于没有差别: 患者太少,差值的区间一定很宽</button>
<button class="quiz-opt" data-why="不能为了让 P 值变小而换检验方法。问题不在选了哪种检验,而在样本量不足以下结论。">应该换一种检验方法,直到 P &lt; 0.05</button>
</div>

## 用哪一种检验

| 要回答的问题 | 数据 | 常用的检验 | SciPy 的函数 |
|---|---|---|---|
| 两组不同的患者,一个连续的指标 | 有长尾或患者少 | Mann–Whitney U | `stats.mannwhitneyu(a, b)` |
| | 大致对称 | t 检验(Welch) | `stats.ttest_ind(a, b, equal_var=False)` |
| 同一批患者比两次(配对) | 有长尾或患者少 | Wilcoxon 符号秩 | `stats.wilcoxon(x, y)` |
| | 差值大致对称 | 配对 t 检验 | `stats.ttest_rel(x, y)` |
| 两个分类变量有没有关系 | 每格期望人数 ≥ 5 | 卡方 | `stats.chi2_contingency(table)` |
| | 有的格期望人数 < 5 | Fisher 精确 | `stats.fisher_exact(table)` |
| 两个连续的量是否一起变 | 有极端值,或关系不是直线 | Spearman | `stats.spearmanr(x, y)` |
| | 大致是直线,没有极端值 | Pearson | `stats.pearsonr(x, y)` |
| 做了好几个检验 | | Holm 校正(探索性的筛选用 BH) | 本课的 `holm`;`stats.false_discovery_control` |

## 小结

- P 值: 如果其实没有关系,像现在这么大(或者更大)的差别有多容易碰上。把分组打乱几千次,数一数就知道。
- 两组不同的患者用 Mann–Whitney(或 t 检验);同一批患者比两次用 Wilcoxon 符号秩(或配对 t 检验)。配对的数据用了不配对的检验,可能看不出本来存在的差别。
- 四格表用卡方检验,有的格期望人数小于 5 时用 Fisher。
- Spearman 相关只看名次,不怕极端值;相关不等于因果。
- 同一个问题做了几个检验,就要校正(Holm)。
- 估计值、95% 区间、P 值写在一起;P 值大不等于没有差别,P 值小不等于差别重要。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 假设检验 | hypothesis test | 判断看到的差别会不会只是碰巧 |
| 无效假设 | null hypothesis | "其实没有差别、没有关系"这个假设 |
| P 值 | P value | 无效假设成立时,出现这么大或更大差别的机会 |
| 有统计学意义 | statistically significant | P 值小于事先约定的线(通常 0.05) |
| 置换检验 | permutation test | 打乱分组来算 P 值 |
| 配对 | paired | 两个数来自同一位患者 |
| 四格表 | 2 × 2 table | 两个二分类变量交叉的人数表 |
| 相关系数 | correlation coefficient | −1 到 1,量两个量一起变的程度 |
| 多重比较 | multiple comparisons | 做了很多检验,碰巧过线的机会变大 |
| 错误发现率 | false discovery rate (FDR) | 过线的结果里,假的占多大比例 |

## 这一课没讲什么

- **三组以上的整体比较**(方差分析、Kruskal–Wallis、Friedman 检验): 论文里常见,思路和这一课一样。
- **回归**: 同时考虑几个因素,而不是一次只比两组。统计机器学习课的第 1–3 课。
- **生存数据的检验**(log-rank): 属于生存分析,这门课不讲。
- **检验效能和样本量**: 研究开始之前要算的东西。预测模型的样本量在统计机器学习课的第 13 课。

## 写进论文 Writing it up

<div class="paper">

方法里写清楚每种比较用了什么检验、是不是双侧(差别不管正负都算,这一课用的都是双侧)、有没有校正:

> Paired comparisons between plans used the Wilcoxon signed-rank test, with Holm correction for the three pairwise comparisons. Independent groups were compared with the Mann–Whitney U test and categorical variables with the chi-square or Fisher's exact test. Associations were assessed with Spearman's rank correlation. All tests were two-sided; *P* < 0.05 was considered statistically significant.

结果里写估计值、区间和确切的 P 值:

> Vaginal D2cc was lower with the Volume plan than with the 5 mm prescription (median difference −2.0 Gy, 95% CI −2.4 to −1.5; *P* < 0.001).

**审稿人可能问**: 做了多少个检验?有没有做多重比较的校正?

**怎么回答**: 列出事先计划的比较(这里是三个两两比较),说明用了 Holm 校正;探索性的分析单独标明。

**审稿人可能问**: 同一位患者的数据,为什么用了两组独立样本的检验?

**怎么回答**: 这是配对的数据,应该用配对的检验。如果确实用错了,重新分析。

**审稿人可能问**: P = 0.06,结论里却写"有降低的趋势"?

**怎么回答**: 不用"趋势"来包装没过线的结果。如实报告估计值、区间和 P 值,结论写"没有检出差别",并讨论样本量的限制。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| 相关系数或 P 值是 NaN | 数据里有缺失值。先 `dropna(subset=[...])`,并记下用了多少位患者。 |
| `ValueError: Array shapes are incompatible for broadcasting.` | 配对的检验要求两列一样长、一一对应。检查两列是不是来自同一张表的同一批患者。 |
| P 值显示成 `1.6e-07` | 科学计数法: 小数点往左移 7 位。论文里写 *P* < 0.001。 |
| 卡方检验的 P 值和别人的软件不一样 | 检查有没有做连续性校正(`correction=True` 是默认)。 |

## 延伸阅读 Further reading

- SciPy 文档: [Statistical functions (scipy.stats)](https://docs.scipy.org/doc/scipy/reference/stats.html)。本课用到的检验都在这一页。
- Wasserstein RL, Lazar NA. [The ASA statement on p-values: context, process, and purpose](https://doi.org/10.1080/00031305.2016.1154108). *The American Statistician* 2016;70:129–133。美国统计学会关于 P 值的六条声明,第 7 节的内容大多出自这里。
- Phipson B, Smyth GK. [Permutation P-values should never be zero](https://arxiv.org/abs/1603.05766). *Stat Appl Genet Mol Biol* 2010;9:39。打乱分组的检验里,P 为什么要算上实际的那一种分法。
- Altman DG, Bland JM. [Absence of evidence is not evidence of absence](https://www.bmj.com/content/311/7003/485). *BMJ* 1995;311:485。一页的短文,讲的就是"没有检出差别"不等于"没有差别"。

{{< practice >}}

**下一课**: 读表、整理、Table 1、画图、区间、检验,都学过了,但都是在网页里、用准备好的数据。第 7 课在你自己的电脑上装好 Python,把一份 Excel 从头做到尾。
