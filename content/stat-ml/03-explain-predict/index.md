---
title: "Lesson 3 · 解释和预测 Explaining and Predicting"
summary: '同一个模型的两种用法: 回答"有没有关系"和"准不准",要看的东西不一样。'
weight: 30
toc: true
duration: "约 70 分钟,可以分两次学: 第 0–3 节,第 4–7 节"
prereq: "本课程[第 1 课](../01-linear-regression/);[Python 课](/python/)第 5 课(bootstrap 和 95% 区间)"
questions:
  - "回归表里的区间和 P 值,回答的是什么问题?"
  - '一个变量"显著",是不是说明它对预测有用?'
  - "同一个模型,用来解释和用来预测,要检查的东西有什么不同?"
objectives:
  - "用 statsmodels 得到回归表,读懂系数、95% 区间和 P 值"
  - "用 bootstrap 自己算一遍系数的区间,和回归表对上"
  - '用一批新病人检验预测误差,看出"显著"和"对预测有用"是两回事'
  - "举例说明: 模型里加一个变量,另一个变量的系数会变"
  - "说出解释和预测各自要报告什么"
---

前两课只看了模型的公式和预测值,故意没看你最熟悉的那张表: 系数、95% 区间、P 值。这一课把它请回来,然后问一个问题: 这张表回答的是什么?

它回答的是**解释**的问题: 壁厚和 Δ 有没有关系,关系有多大,这个数有多确定。多因素分析、"独立危险因素",都是在回答这一类问题。

这门课要做的是**预测**: 对一位还没治疗的病人,模型给的数和实际会差多少。模型是同一个,问题不同,要看的东西也就不同。回归表再漂亮,也没有回答第二个问题。把两者分清楚,是读懂和写好预测模型论文的第一步。

> 🔬 预测模型的论文里最常见的毛病,是拿解释的证据(P 值、OR)去证明预测的结论("这个模型能用于个体化决策")。这一课之后,你应该能一眼看出来。

**练习怎么做**: 讲解用**体积优化**计划的 Δ,练习换成**图形优化**。

## 0. 准备: 建模的病人,和一批新病人

```python
import numpy as np
import pandas as pd
from cohort import make_cohort

def prepare(d):
    d = d.dropna().copy()
    d["delta"] = d["d2cc_vol"] - d["d2cc_5mm"]
    d["delta_gra"] = d["d2cc_gra"] - d["d2cc_5mm"]
    d["ebrt"] = (d["arm"] == "EBRT+VCB").astype(int)
    d["stage_ia"] = (d["stage"] == "IA").astype(int)
    return d

df = prepare(make_cohort())                    # 建模用的 196 位病人
new = prepare(make_cohort(n=5000, seed=99))    # 另一批新病人: 产生方式相同,模型没见过
cols = ["wall_mean", "diameter", "n_fx"]
print(len(df), "位建模病人;", len(new), "位新病人")
```

`df` 是前两课一直用的 196 位病人。`new` 是另外造的 4900 位: 产生方式和 `df` 完全相同,但建模时不用他们。真实研究里没有这样一批现成的新病人;模拟数据的方便之处,就是可以造出来,直接看模型对新病人准不准。(真实研究里怎么办,是第 4 课。)

## 1. 回归表

scikit-learn 只给系数,不给区间和 P 值。要那张熟悉的表,用 statsmodels:

```python
import statsmodels.api as sm

X = sm.add_constant(df[cols])                  # statsmodels 要自己加一列常数,代表截距
fit = sm.OLS(df["delta"], X).fit()             # OLS: 最小二乘,和第 1 课是同一个模型
table = pd.DataFrame({"coef": fit.params, "lower": fit.conf_int()[0], "upper": fit.conf_int()[1], "P": fit.pvalues})
table.round(3)
```

第一次运行要下载 statsmodels,会多等一会儿。系数和第 1 课的一样: 壁厚 1.57,直径 0.17,次数 −1.21。多出来三列:

- **lower、upper**: 系数的 95% 置信区间。壁厚是 1.30 到 1.84,直径是 0.08 到 0.25,次数是 −1.80 到 −0.62。
- **P**: 三个变量都小于 0.001(四舍五入以后,表里显示成 0.0)。

`print(fit.summary())` 会打印出统计软件那种完整的表,内容相同。

## 2. 区间和 P 值是怎么来的

<div class="frame">

**背景**: 系数是从这 196 位病人算出来的。换一批病人,系数会不一样。

**目标**: 说清楚系数有多确定: 换一批病人,它会在多大范围里变。

**做法**: 回归表用公式直接算出区间。Python 课第 5 课学过另一个办法,不用公式: **bootstrap**。这里用它自己算一遍,看两者是不是一回事。

</div>

**它在算什么**

1. 从 196 位病人里有放回地抽 196 位,当作"另一批病人"。
2. 用这一批重新拟合,记下壁厚的系数。
3. 重复 2000 次,得到 2000 个系数。
4. 取中间 95% 的范围(第 2.5 和第 97.5 百分位数),就是 95% 区间。

```python
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(1)
X_np, y_np = df[cols].to_numpy(), df["delta"].to_numpy()
n = len(df)
boot = []
for _ in range(2000):
    idx = rng.integers(0, n, n)                                    # 第 1 步: 有放回地抽 196 位
    boot.append(LinearRegression().fit(X_np[idx], y_np[idx]).coef_[0])   # 第 2 步: 重新拟合,记下壁厚的系数
low, high = np.percentile(boot, [2.5, 97.5])                       # 第 4 步
print(f"bootstrap: {low:.2f} 到 {high:.2f};回归表: {fit.conf_int().loc['wall_mean', 0]:.2f} 到 {fit.conf_int().loc['wall_mean', 1]:.2f}")
```

bootstrap 给出 1.32 到 1.83,回归表是 1.30 到 1.84。两种算法,几乎同一个答案: 区间说的是"换一批病人,这个系数大概落在哪里"。

**P 值**是同一件事的另一种问法(Python 课第 6 课): 如果壁厚和 Δ 其实没有关系,真实的系数是 0,碰巧得到 1.57 这么大的系数有多容易?区间离 0 很远,所以 P 很小。

到这里为止,都是在回答一个问题: **这个系数有多确定**。

<div class="exercise">

**练习 1 · 轮到你: 图形优化的回归表**

对图形优化的 Δ(`delta_gra`)拟合同样的三变量模型,取出壁厚系数 95% 区间的下限和上限,存进 `lo_gra`、`hi_gra`。

```python
fit_gra = sm.OLS(df[___], X).fit()
lo_gra, hi_gra = fit_gra.conf_int().loc[___]
print(fit_gra.params.round(2))
print(f"壁厚的系数: 95% 区间 {lo_gra:.2f} 到 {hi_gra:.2f}")
```

```pycheck
need("fit_gra", "lo_gra", "hi_gra")
if close(lo_gra, 1.3004, 0.005):
    fail('这是体积优化的区间。第一个空填 "delta_gra"。')
if close(lo_gra, -0.1919, 0.005) or close(lo_gra, -2.6871, 0.005):
    fail('取的是另一个变量的那一行。第二个空填 "wall_mean"。')
if close(lo_gra, 1.3593, 0.005) and close(hi_gra, 2.1631, 0.005):
    ok("对了: 壁厚的系数是 1.76,95% 区间 1.36 到 2.16,离 0 很远。再看直径那一行: 系数 −0.07,区间跨过了 0(P = 0.27)。体积优化里直径是显著的,图形优化里不是。")
fail(f"现在是 {lo_gra:.2f} 到 {hi_gra:.2f},应该是 1.36 到 2.16。")
```

<details><summary>提示</summary>

第一个空是列名 `"delta_gra"`;`conf_int()` 是一张表,每个变量一行,第二个空是壁厚那一行的名字 `"wall_mean"`。

</details>

<details><summary>参考答案</summary>

```python
fit_gra = sm.OLS(df["delta_gra"], X).fit()
lo_gra, hi_gra = fit_gra.conf_int().loc["wall_mean"]
print(fit_gra.params.round(2))
print(f"壁厚的系数: 95% 区间 {lo_gra:.2f} 到 {hi_gra:.2f}")
```

</details>

</div>

<div class="exercise">

**练习 2 · 轮到你: 用 bootstrap 再算一遍**

对图形优化,用 bootstrap 算壁厚系数的 95% 区间,存进 `low_g`、`high_g`。

```python
rng = np.random.default_rng(1)
yg_np = df["delta_gra"].to_numpy()
boot_g = []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    boot_g.append(LinearRegression().fit(X_np[___], yg_np[___]).coef_[0])
low_g, high_g = np.percentile(boot_g, [___, 97.5])
print(f"bootstrap: {low_g:.2f} 到 {high_g:.2f}")
```

```pycheck
need("low_g", "high_g")
if close(low_g, 1.316, 0.01) and close(high_g, 1.828, 0.01):
    fail("这是体积优化的区间。拟合时第二个参数要用 yg_np[idx]。")
if close(low_g, 1.326, 0.02) and close(high_g, 2.19, 0.02):
    ok("对了: 1.33 到 2.19,回归表给的是 1.36 到 2.16。两种算法又对上了。")
if high_g - low_g < 0.2:
    fail("区间太窄了: 每次拟合用的是不是同一批病人?前两个空都填 idx,让 X 和 y 取同一批抽出来的病人。")
fail(f"现在是 {low_g:.2f} 到 {high_g:.2f},应该是 1.33 到 2.19。前两个空是 idx,第三个空是 2.5。")
```

<details><summary>提示</summary>

`X_np[idx]` 和 `yg_np[idx]` 要用同一个 `idx`: 抽到哪位病人,就同时取这位病人的变量和结局。95% 区间的下端是第 2.5 百分位数。

</details>

<details><summary>参考答案</summary>

```python
rng = np.random.default_rng(1)
yg_np = df["delta_gra"].to_numpy()
boot_g = []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    boot_g.append(LinearRegression().fit(X_np[idx], yg_np[idx]).coef_[0])
low_g, high_g = np.percentile(boot_g, [2.5, 97.5])
print(f"bootstrap: {low_g:.2f} 到 {high_g:.2f}")
```

</details>

</div>

## 3. "显著"不等于"预测得准"

三个变量的 P 值都小于 0.001。这是不是说,用这三个变量就能把 Δ 预测准?先看一个更极端的例子。

<div class="demo">

**设定**: 另一份模拟数据。年龄对 Δ 有一点真实的影响: 每大 1 岁,Δ 高 0.03 Gy。病人之间本来就相差 2 Gy 左右,年龄的影响比它小得多。滑块是建模用了多少位病人。

**你做什么**: 把滑块从 30 位拖到 3000 位。先看 P 值,再看下面两根条。

**你会看到**: P 值从 0.433(30 位)降到 0.028(300 位),再到小于 0.001(1000 位以上)。下面两根条几乎不动: 都猜平均值,在新病人身上平均差 1.78 Gy;用上年龄,是 1.76 到 1.77 Gy。

**这说明**: 病人越多,P 值越小,因为越有把握说"年龄的影响不是 0"。可是知道了年龄,预测只准了 0.02 Gy。P 值量的是"影响存在"的把握,不是"预测准了多少"。

<div class="widget" data-widget="signif">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

回到我们的三个变量。它们各自让预测准了多少,回归表里没有,要到新病人身上去量。模型用 196 位病人拟合,误差在 4900 位新病人身上算:

<div class="quiz">
<p><b>先猜一猜</b>: 壁厚、直径、次数的 P 值都小于 0.001。哪一个让预测准得最多?</p>
<button class="quiz-opt" data-why="三个 P 值都很小,只说明三个系数都'不是 0'。各自让预测准了多少,P 值看不出来。运行下面的代码。">三个差不多: P 值都一样小</button>
<button class="quiz-opt" data-correct data-why="运行下面的代码看具体的数: 壁厚把平均误差降了将近 0.5 Gy,另外两个变量加起来不到 0.1 Gy。">壁厚,比另外两个多得多</button>
<button class="quiz-opt" data-why="次数的系数绝对值最大(−1.21),但队列里次数只有 2 和 3 两种,能解释的差别有限(第 1 课的想一想)。运行下面的代码看看。">次数: 它的系数最大</button>
</div>

```python
def mae_new(columns, outcome="delta"):
    model = LinearRegression().fit(df[columns], df[outcome])           # 用 196 位病人拟合
    return (new[outcome] - model.predict(new[columns])).abs().mean()   # 在新病人身上算平均误差

print(f"都猜平均值:           {(new['delta'] - df['delta'].mean()).abs().mean():.2f} Gy")
print(f"壁厚:                 {mae_new(['wall_mean']):.2f} Gy")
print(f"壁厚 + 次数:          {mae_new(['wall_mean', 'n_fx']):.2f} Gy")
print(f"壁厚 + 次数 + 直径:   {mae_new(['wall_mean', 'n_fx', 'diameter']):.2f} Gy")
```

都猜平均值,平均差 2.28 Gy。加上壁厚,降到 1.82 Gy。再加次数,1.79 Gy;再加直径,1.73 Gy。三个变量在回归表里一样"显著",对预测的贡献却差了一个数量级: 壁厚贡献了 0.46 Gy,次数和直径一共 0.09 Gy。

这不是说次数和直径没有用,而是说: 一个变量值不值得放进预测模型,要看它让新病人的误差降了多少,不看它的 P 值。按 P 值筛变量("单因素分析 P < 0.1 的进入多因素分析")是解释的习惯,用来建预测模型并不合适: 第 5、6 课会讲更好的办法。

<div class="exercise">

**练习 3 · 轮到你: 图形优化,直径帮上忙了吗**

练习 1 里看到,图形优化时直径不显著(P = 0.27)。用 `mae_new` 算两个模型在新病人身上的平均误差: 壁厚加次数(`mae_two`),再加上直径(`mae_three`)。结局是 `"delta_gra"`。

```python
mae_two = mae_new(["wall_mean", "n_fx"], ___)
mae_three = mae_new([___, ___, ___], "delta_gra")
print(f"壁厚 + 次数: {mae_two:.3f} Gy;再加直径: {mae_three:.3f} Gy")
```

```pycheck
need("mae_two", "mae_three")
if close(mae_two, 1.787, 0.004):
    fail('这是体积优化的结果。第一个空填 "delta_gra"。')
if close(mae_two, 2.541, 0.004) and close(mae_three, 2.548, 0.004):
    ok("对了: 2.541 Gy 和 2.548 Gy。加上直径,新病人身上的误差没有变小,还大了一点点。直径在图形优化里没有真实的作用,模型却给了它一个系数(−0.07): 这个系数学到的只是这 196 位病人的偶然,到新病人身上就成了噪声。")
fail(f"现在是 {mae_two:.3f} 和 {mae_three:.3f},应该是 2.541 和 2.548。三个变量是 wall_mean、n_fx、diameter。")
```

<details><summary>提示</summary>

第一个空是 `"delta_gra"`;后三个空是三个列名(带引号): `"wall_mean"`、`"n_fx"`、`"diameter"`。

</details>

<details><summary>参考答案</summary>

```python
mae_two = mae_new(["wall_mean", "n_fx"], "delta_gra")
mae_three = mae_new(["wall_mean", "n_fx", "diameter"], "delta_gra")
print(f"壁厚 + 次数: {mae_two:.3f} Gy;再加直径: {mae_three:.3f} Gy")
```

</details>

</div>

## 4. 系数的含义,取决于模型里还有谁

解释有它自己的陷阱,预测不太在乎。看分期: IA 期的病人和其他病人相比,Δ 有没有差别?

```python
def coef_table(columns, outcome="delta"):
    f = sm.OLS(df[outcome], sm.add_constant(df[columns])).fit()
    return pd.DataFrame({"coef": f.params, "lower": f.conf_int()[0], "upper": f.conf_int()[1], "P": f.pvalues}).round(2).drop("const")

print(coef_table(["stage_ia"]))
print(coef_table(["stage_ia", "ebrt"]))
```

只放分期: IA 期的系数是 −0.67(95% 区间 −1.46 到 0.13,P = 0.10)。IA 期的病人,D2cc 看起来多降了 0.67 Gy。

再放进"做没做外照射": IA 期的系数变成 −0.06(P = 0.89),外照射的系数是 +1.31(P < 0.01)。差别原来是治疗方式带来的: IA 期的病人大多不做外照射,近距离做 3 次,累计降得多;分期本身和 Δ 没有关系。

```python
print(pd.crosstab(df["stage_ia"], df["ebrt"]))
```

98 位 IA 期的病人里,67 位没做外照射;其他分期的 98 位里只有 22 位。分期和治疗方式纠缠在一起,只放分期时,它替治疗方式"背了账"。这就是**混杂**(confounding): 你在多因素分析里调整的,正是这个。

<div class="quiz">
<p><b>想一想</b>: 只看预测,用"壁厚 + 分期"和用"壁厚 + 外照射",在新病人身上的平均误差是 1.80 Gy 和 1.79 Gy,几乎一样。这说明什么?</p>
<button class="quiz-opt" data-why="预测差不多,不等于两个变量的含义一样。分期能预测,是因为它带着治疗方式的信息;改变一位病人的分期并不会改变 Δ。">分期和外照射对 Δ 的作用一样大</button>
<button class="quiz-opt" data-correct data-why="预测只要信息,不问信息从哪个变量来: 分期带着一部分治疗方式的信息,所以也能帮上忙。解释要问'是谁的作用',这时混杂就要紧了。所以预测模型的系数不能直接当作因果来读;反过来,解释里很重要的调整,对预测不一定有帮助。">预测只需要信息,不在乎是哪个变量带来的;解释才在乎</button>
<button class="quiz-opt" data-why="模型没有错。两个模型都是合格的预测模型,只是其中的系数不能拿来回答'分期有没有作用'。">其中一个模型一定是错的</button>
</div>

<div class="exercise">

**练习 4 · 轮到你: 图形优化里分期的系数**

对图形优化的 Δ 做同样的两步,取出 IA 期的系数: 只放分期时(`ia_alone`),和加上外照射以后(`ia_adjusted`)。

```python
ia_alone = coef_table(["stage_ia"], "delta_gra").loc["stage_ia", ___]
ia_adjusted = coef_table([___, ___], "delta_gra").loc["stage_ia", "coef"]
print(f"只放分期: {ia_alone:.2f};加上外照射: {ia_adjusted:.2f}")
```

```pycheck
need("ia_alone", "ia_adjusted")
if close(ia_alone, -0.67, 0.011):
    fail('这是体积优化的结果。coef_table 的第二个参数要是 "delta_gra"(起始代码里已经是),检查有没有改动。')
if close(ia_alone, 0.06, 0.011):
    fail('取到的是 P 值那一列。第一个空填 "coef"。')
if close(ia_alone, -1.0, 0.011) and close(ia_adjusted, -0.05, 0.011):
    ok("对了: −1.00 变成 −0.05。只放分期时,IA 期的病人图形优化多降 1 Gy(P = 0.06,差一点就'显著'了);调整治疗方式以后,什么也没剩下。")
fail(f'现在是 {ia_alone:.2f} 和 {ia_adjusted:.2f},应该是 −1.00 和 −0.05。第一个空是 "coef",后两个空是 "stage_ia" 和 "ebrt"。')
```

<details><summary>提示</summary>

`coef_table` 给的表有 coef、lower、upper、P 四列,第一个空取 `"coef"`。后两个空是两个列名: `"stage_ia"`、`"ebrt"`。

</details>

<details><summary>参考答案</summary>

```python
ia_alone = coef_table(["stage_ia"], "delta_gra").loc["stage_ia", "coef"]
ia_adjusted = coef_table(["stage_ia", "ebrt"], "delta_gra").loc["stage_ia", "coef"]
print(f"只放分期: {ia_alone:.2f};加上外照射: {ia_adjusted:.2f}")
```

</details>

</div>

## 5. 预测要到新病人身上量

第 1 课算过三变量模型的平均误差: 1.67 Gy。那是在建模用的 196 位病人身上算的。上面在新病人身上算,是 1.73 Gy。

```python
model3 = LinearRegression().fit(df[cols], df["delta"])
train_error = (df["delta"] - model3.predict(df[cols])).abs().mean()
new_error = (new["delta"] - model3.predict(new[cols])).abs().mean()
print(f"建模的病人: {train_error:.2f} Gy;新病人: {new_error:.2f} Gy")
```

模型对自己见过的病人,总是比对新病人准一些: 它拟合的时候,连这 196 位病人的偶然也一起凑上了。这里只差 0.06 Gy,因为模型很简单(3 个变量,196 位病人);变量一多,差距会大得多。这个差距叫**乐观**(optimism),它有多大、没有新病人的时候怎么估计,是第 4 课的主题。

<div class="exercise">

**练习 5 · 轮到你: 图形优化的模型乐观了多少**

对图形优化的三变量模型,算建模病人身上的平均误差(`train_g`)、新病人身上的(`new_g`),以及两者之差(`gap`)。

```python
model3_g = LinearRegression().fit(df[cols], df["delta_gra"])
train_g = (df["delta_gra"] - model3_g.predict(df[cols])).abs().mean()
new_g = (new[___] - model3_g.predict(___)).abs().mean()
gap = new_g - train_g
print(f"建模的病人 {train_g:.2f} Gy,新病人 {new_g:.2f} Gy,相差 {gap:.2f} Gy")
```

```pycheck
need("train_g", "new_g", "gap")
if close(new_g, 2.42, 0.004):
    fail("new_g 和 train_g 一样: 第二个空给的还是建模病人的表。要预测的是新病人: new[cols]。")
if close(new_g, 1.73, 0.004):
    fail('这是体积优化的新病人误差。第一个空填 "delta_gra"。')
if close(new_g, 2.548, 0.004) and close(gap, 0.128, 0.006):
    ok("对了: 2.42 Gy 和 2.55 Gy,相差 0.13 Gy。图形优化的模型比体积优化的(差 0.06 Gy)更乐观: 它里面有一个没有真实作用的变量(直径),结局本身的波动也更大。")
fail(f'现在新病人的误差是 {new_g:.3f},应该是 2.548。第一个空是 "delta_gra",第二个空是 new[cols]。')
```

<details><summary>提示</summary>

和上面体积优化的两行一样,只是换成新病人的表: 结局是 `new["delta_gra"]`,变量是 `new[cols]`。

</details>

<details><summary>参考答案</summary>

```python
model3_g = LinearRegression().fit(df[cols], df["delta_gra"])
train_g = (df["delta_gra"] - model3_g.predict(df[cols])).abs().mean()
new_g = (new["delta_gra"] - model3_g.predict(new[cols])).abs().mean()
gap = new_g - train_g
print(f"建模的病人 {train_g:.2f} Gy,新病人 {new_g:.2f} Gy,相差 {gap:.2f} Gy")
```

</details>

</div>

## 6. 两种用法,对照着看

| | 解释 | 预测 |
|---|---|---|
| 问的是 | 这个因素和结局有没有关系,关系多大 | 对一位新病人,模型给的数和实际差多少 |
| 看什么 | 系数(或 OR)、95% 区间、P 值 | 新病人身上的误差;后面还有区分度、校准、决策曲线 |
| 在谁身上算 | 建模的全部病人 | 模型没见过的病人 |
| 变量怎么选 | 按临床上的因果关系: 该调整的混杂都要放 | 按能不能让新病人的误差变小 |
| 最怕什么 | 混杂: 系数替别的变量背了账 | 过拟合: 把建模病人的偶然当成了规律 |
| 结论的样子 | "壁厚每厚 1 mm,Δ 高 1.57 Gy(95% CI 1.30–1.84)" | "对新病人,预测值和实际值平均相差 1.73 Gy" |

同一篇论文可以两样都做,但要各用各的证据。常见的三种张冠李戴:

- 用 P 值证明模型"能预测"。P 值小,只说明关系存在(第 3 节)。
- 用预测模型的系数讲因果。预测只管信息,不管信息是谁带来的(第 4 节)。
- 在建模的病人身上报告预测的成绩。那是乐观的版本(第 5 节)。

## 小结

- 回归表(系数、区间、P 值)回答的是**解释**: 关系在不在,有多大,有多确定。区间可以用 bootstrap 自己算出来,和公式的结果一致。
- **"显著"不等于"对预测有用"**: 病人越多 P 值越小,预测的误差却不会因此变小。一个变量对预测的贡献,要到新病人身上量。
- 系数的含义取决于模型里还有谁。解释怕**混杂**;预测只要信息,不问来源。
- 预测的成绩要在模型**没见过的病人**身上算。在建模病人身上算的偏乐观。
- 从下一课起,这门课只问预测的问题。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 解释 | explanation, inference | 回答因素和结局有没有关系、关系多大 |
| 预测 | prediction | 回答对新病人,模型给的数准不准 |
| 回归表 | regression table | 系数、标准误、区间和 P 值的那张表 |
| 置信区间 | confidence interval (CI) | 换一批病人,系数大概落在的范围 |
| 混杂 | confounding | 一个变量的系数里混进了另一个变量的作用 |
| 调整 | adjustment | 把混杂的变量放进模型 |
| 新病人 | new (unseen) patients | 模型拟合时没有用到的病人 |
| 表观成绩 | apparent performance | 在建模病人身上算的成绩 |
| 乐观 | optimism | 表观成绩比新病人身上的成绩好出来的那一截 |

## 这一课没讲什么

- **没有新病人怎么办**: 真实研究里造不出 4900 位新病人。第 4 课的交叉验证,是用手里这一批病人模拟"新病人"。
- **逻辑回归的回归表**: 用 `sm.Logit` 可以得到 OR 的区间和 P 值,读法和这一课相同。
- **因果推断**: 要回答"改变这个因素,结局会不会变",光调整混杂还不够,需要专门的设计和方法。这门课不讲。
- **标准误那一列**: 它是区间的原料(区间大约是系数 ± 2 个标准误),这一课没有单独讲。

## 写进论文 Writing it up

<div class="paper">

一篇预测模型的论文,方法部分要让读者分得清哪些分析是解释,哪些是预测:

> Associations between predictors and Δ are reported as regression coefficients with 95% confidence intervals. These describe associations in the development cohort and were not used to select predictors. Predictive performance was assessed separately, on patients not used for model fitting (see Validation).

**审稿人可能问**: 所有预测变量都高度显著,为什么模型的预测误差还这么大?

**怎么回答**: 显著性说明关系存在,不说明关系能解释多少个体差异。报告预测误差(或 R²、AUC)本身,并说明还有多少个体差异模型解释不了。

**审稿人可能问**: 模型里的系数能不能解读为这些因素对结局的作用?

**怎么回答**: 不能直接这样读。这是预测模型,变量是按预测的需要选的,没有按因果关系调整混杂;系数只能描述为"在这个模型里的权重"。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| statsmodels 的系数里没有截距 | 忘了 `sm.add_constant`。statsmodels 不会自动加截距。 |
| `MissingDataError: exog contains inf or nans` | 数据里有缺失值。这一课的 `prepare` 已经用 `dropna()` 去掉了。 |
| bootstrap 的区间和网页上的略有不同 | 检查 `np.random.default_rng(1)` 的种子,以及循环里抽样和拟合的顺序有没有改。 |
| `KeyError: 'wall_mean'`(在 `conf_int().loc[...]`) | 行名写错了,或者拟合时用的表里没有这个变量。 |
| 第一次运行很慢 | statsmodels 比较大,第一次要下载;2000 次 bootstrap 在浏览器里也要几秒。 |

## 延伸阅读 Further reading

- Shmueli G. [To explain or to predict?](https://doi.org/10.1214/10-STS330) *Statistical Science* 2010;25(3):289–310。专门讲解释和预测为什么是两件事,常被引用。
- statsmodels 文档: [Linear Regression](https://www.statsmodels.org/stable/regression.html)。`OLS`、`conf_int`、`summary` 的说明。

**下一课**: 这一课靠 4900 位造出来的新病人,才看到模型对新病人准不准。真实研究里只有手里这一批。第 4 课讲怎样只用这一批病人,诚实地估计模型对新病人的误差: 交叉验证。

{{< practice >}}
