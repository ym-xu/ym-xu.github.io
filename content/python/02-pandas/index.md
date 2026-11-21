---
title: "Lesson 2 · 表格数据 pandas Tabular Data"
summary: "读入一张表,选列、筛行、算新的一列;空着的格子不等于 0。"
weight: 20
toc: true
duration: "约 70 分钟,可以分两次学: 第 1–3 节,第 4–7 节"
prereq: "[第 1 课](../01-first-steps/)"
questions:
  - "一张 Excel 或 CSV 表,怎么读进 Python?"
  - "怎么只看需要的几列、需要的那些患者?"
  - "表里空着的格子怎么办?"
objectives:
  - "读入一张表,看它有多大、有哪些列、每列是什么类型"
  - "选出几列,按条件筛出一部分患者"
  - "用已有的列算出新的一列: BMI、两套计划的剂量差"
  - "找出缺失值,说明为什么不能把它当成 0"
  - "把整理好的表存成文件"
---

第 1 课里,一位患者的记录是一个字典。真实的研究是一张表: 一行一位患者,一列一个变量,几百行,就像一本病例登记表。

这样的表在 Excel 里也能处理: 拖公式、筛选、复制粘贴。问题是做过的每一步都没有留下记录,数据一更新就要全部重做,拖错一行也很难发现。用代码,每一步都写了下来: 数据更新了,重新运行一遍就行;别人想知道你是怎么算的,看代码就行。

Python 里处理表格的包叫 **pandas**。这一课用它读入课程的模拟队列(200 位患者),做临床研究里最常做的四件事: 看一眼、挑出一部分、算新的一列、处理空格子。

> 🔬 这一课的第 6 节,把一篇已发表论文里的一张小表敲进来算一算。除此以外,表里的患者都是模拟的。

**练习怎么做**: 讲解里算的是**体积优化**计划(`vol`)和阴道 D2cc。练习换成**图形优化**计划(`gra`): 每讲完一步,你对图形优化再做一遍。

## 1. 把表读进来

浏览器里的 Python 读不到你电脑上的文件,所以这门课先读网页自带的一个文件 `cohort.csv`。**CSV**(comma-separated values)是最简单的表格文件: 每行一位患者,列和列之间用逗号隔开,Excel 可以直接打开,也可以把表另存成这种格式。

```python
import pandas as pd

df = pd.read_csv("cohort.csv")
df.head()
```

第一次运行要下载 pandas,会多等一会儿。`import pandas as pd` 是大家通用的写法;`pd.read_csv(...)` 把文件读成一张表,存进 `df`;`df.head()` 显示前 5 行(表比较宽,可以左右滑动)。

pandas 管这样一张表叫**数据框**(DataFrame),`df` 是它惯用的名字。最左边没有列名的 0、1、2……是行号,叫**索引**(index),和列表一样从 0 开始数。

读 Excel 的写法几乎一样,第 7 课在自己电脑上会用到:

```python
#| norun
df = pd.read_excel("我的数据.xlsx", sheet_name="patients")
```

表里的列:

| 列 | 含义 |
|---|---|
| `id` | 患者编号(研究里自己编的号,不是病历号) |
| `age`、`height_cm`、`weight_kg` | 年龄(岁)、身高(cm)、体重(kg) |
| `arm` | 治疗方式: `VCB` 单纯近距离,`EBRT+VCB` 外照射加近距离 |
| `stage`、`grade` | 分期、分级 |
| `n_fx`、`diameter` | 近距离治疗的次数、施源器直径(mm) |
| `wall_min`、`wall_mean`、`wall_max`、`wall_range` | 阴道壁厚度(mm): 最小、平均、最大、最大减最小 |
| `d2cc_5mm`、`d2cc_vol`、`d2cc_gra` | 三套计划的阴道 D2cc(Gy) |
| `d90_5mm`、`d90_vol`、`d90_gra` | 三套计划的靶区 D90(Gy) |

D2cc、D90 和三套计划是什么,见[课程首页的说明](../#例子的临床背景)。

## 2. 先看一眼

拿到一张表,先看它有多大、每一列是什么类型:

```python
print(df.shape)       # (行数, 列数)
df.dtypes
```

200 行、19 列。`dtypes` 列出每一列的**数据类型**(dtype): `int64` 是整数,`float64` 是小数,`str` 是文字。这一步值得每次都做: 如果一列本该是数字,却显示成文字,多半是表里混进了"无""未测"或者带单位的格子,后面就没法计算。

数字列的概况,用 `describe()`:

```python
df[["age", "height_cm", "weight_kg"]].describe().round(1)
```

每一列给出个数(count)、均值(mean)、标准差(std)、最小值、三个四分位数和最大值。这也是查错的好办法: 年龄 40 到 87 岁,合理;如果身高最小值是 1.6,就说明有人按米录入了。

## 3. 选列和筛行

**选一列**,用方括号加列名,得到的是一列数,pandas 叫它 **Series**:

```python
ages = df["age"]
print(ages.head(3))
print(ages.mean(), ages.min(), ages.max())
```

**选几列**,方括号里放一个列名的列表,所以有两层方括号。得到的还是一张表:

```python
small = df[["id", "age", "arm", "d2cc_5mm"]]
small.head(3)
```

**筛行**: 只要 75 岁及以上的患者。

```python
old = df[df["age"] >= 75]
print(len(old), "位患者 75 岁及以上")
old[["id", "age", "arm"]].head()
```

这一句分两步看。里面的 `df["age"] >= 75` 对每位患者做一次比较,得到一串 True 和 False(和第 1 课的 `doses >= 68` 一样);外面的 `df[...]` 只留下 True 的那些行。最左边的 4、7、11……还是原来表里的行号。

两个条件一起用时,**每个条件加括号**,"并且"写 `&`,"或者"写 `|`:

```python
mask = (df["arm"] == "VCB") & (df["age"] >= 75)     # 单纯近距离,并且 75 岁及以上
print(mask.sum(), "位")
df["arm"].value_counts()
```

`mask.sum()` 数出有几个 True(第 1 课: True 算 1)。`value_counts()` 数每个取值各有多少行: 外照射加近距离 109 位,单纯近距离 91 位。分类变量拿到手,先这样数一遍,同一类写成了两种样子(比如 `VCB` 和 `vcb`)一眼就能发现。

<div class="quiz">
<p><b>想一想</b>: <code>df["age"]</code> 和 <code>df[["age"]]</code> 有什么不同?</p>
<button class="quiz-opt" data-why="不一样: 方括号的层数决定了得到的是一列数,还是一张表。">没有不同,多一层方括号只是写法啰嗦</button>
<button class="quiz-opt" data-correct data-why="一层方括号得到一列数(Series);两层方括号里面是一个列表,得到的是只有一列的表(DataFrame)。大多数时候两者都能用,但有些函数只收表,报错时可以先想到这一点。">前一个是一列数(Series),后一个是只有一列的表(DataFrame)</button>
<button class="quiz-opt" data-why="两种写法都只是取出数据,不会改动原来的表。">后一个会把 age 这一列从表里拿走</button>
</div>

<div class="exercise">

**练习 1 · 轮到你: 筛出一组患者**

外照射加近距离(`arm` 是 `"EBRT+VCB"`),并且 5 mm 计划的阴道 D2cc 达到 68 Gy 的患者有几位?把人数存进 `n_ebrt_high`。

```python
mask = (df["arm"] == ___) & (df["d2cc_5mm"] >= ___)
n_ebrt_high = mask.sum()
print(n_ebrt_high, "位")
```

```pycheck
need("n_ebrt_high")
if n_ebrt_high == 90:
    ok("对了: 109 位外照射加近距离的患者里,有 90 位的 5 mm 计划达到 68 Gy。")
if n_ebrt_high == 58:
    fail("这是单纯近距离的人数: 第一处要填 \"EBRT+VCB\"。")
if n_ebrt_high == 167:
    fail('这是"两个条件至少满足一个"的人数。"并且"要用 &。')
fail(f"现在是 {n_ebrt_high} 位,应该是 90 位。第一处填 \"EBRT+VCB\"(文字要加引号),第二处填 68。")
```

<details><summary>提示</summary>

`"EBRT+VCB"` 是文字,要加引号,大小写和加号都要和表里的一样;阈值是 68。

</details>

<details><summary>参考答案</summary>

```python
mask = (df["arm"] == "EBRT+VCB") & (df["d2cc_5mm"] >= 68)
n_ebrt_high = mask.sum()
print(n_ebrt_high, "位")
```

</details>

</div>

## 4. 算新的一列

对一整列做运算,不用写循环: 列和列之间直接加减乘除,pandas 会一行一行对齐着算。给 `df["新列名"]` 赋值,就在表里加了一列。

```python
df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
df["delta_vol"] = df["d2cc_vol"] - df["d2cc_5mm"]
df[["id", "bmi", "d2cc_5mm", "d2cc_vol", "delta_vol"]].round(1).head()
```

`bmi` 就是第 1 课练习 1 的公式,这次一下算了 200 位患者。`delta_vol` 是换成体积优化计划后,阴道 D2cc 比 5 mm 计划变了多少: 负数表示降低。第一位患者从 76.9 降到 72.9 Gy,Δ = −4.0 Gy。这一列后面每一课都会用到。

比较的结果也可以存成一列:

```python
df["high_5mm"] = df["d2cc_5mm"] >= 68
print(df["high_5mm"].sum(), "位患者的 5 mm 计划达到 68 Gy")
```

<div class="quiz">
<p><b>先猜一猜</b>: <code>df["high_5mm"]</code> 是一列 True 和 False,其中 148 个是 True。<code>df["high_5mm"].mean()</code> 会是多少?</p>
<button class="quiz-opt" data-why="True 和 False 可以求平均: True 算 1,False 算 0。">报错: True 和 False 不能求平均</button>
<button class="quiz-opt" data-correct data-why="True 算 1,False 算 0,所以平均值就是 True 所占的比例: 148 / 200 = 0.74。求&quot;占百分之几&quot;,最常用的就是这个写法。运行下面的代码验证。">0.74</button>
<button class="quiz-opt" data-why="148 是 sum() 的结果。mean() 还要再除以总人数 200。">148</button>
</div>

```python
print(f"占 {df['high_5mm'].mean():.1%}")          # :.1% 把 0.74 显示成 74.0%
df.sort_values("delta_vol").head(3)[["id", "wall_mean", "d2cc_5mm", "d2cc_vol", "delta_vol"]]
```

`sort_values("delta_vol")` 按这一列从小到大排序,前 3 行就是降得最多的 3 位患者(最多的降了 8.8 Gy)。要从大到小排,加上 `ascending=False`。

<div class="exercise">

**练习 2 · 轮到你: 图形优化的剂量差**

加一列 `delta_gra`: 图形优化计划的阴道 D2cc 减 5 mm 计划的。再算出这一列的中位数,存进 `median_gra`(中位数的函数是 `median`)。

```python
df["delta_gra"] = df[___] - df[___]
median_gra = df["delta_gra"].___()
print(f"图形优化: Δ 的中位数 {median_gra:.1f} Gy")
```

```pycheck
need("median_gra")
if "delta_gra" not in df.columns:
    fail("表里还没有 delta_gra 这一列。")
if close(median_gra, -3.4, 0.01):
    ok("对了: 换成图形优化,阴道 D2cc 的变化中位数是 −3.4 Gy,比体积优化的 −2.0 Gy 降得多。")
if close(median_gra, 3.4, 0.01):
    fail("大小对了,符号反了: 要用图形优化减 5 mm,降低才是负数。")
if close(median_gra, -2.0, 0.01):
    fail("这是体积优化的结果: 第一处要填 \"d2cc_gra\"。")
if close(median_gra, -3.55, 0.01):
    fail("这是平均数。题目要的是中位数: median()。")
fail(f"现在 median_gra = {median_gra:.2f},应该是 −3.4。")
```

<details><summary>提示</summary>

列名是 `"d2cc_gra"` 和 `"d2cc_5mm"`,新计划在前;中位数是 `.median()`。

</details>

<details><summary>参考答案</summary>

```python
df["delta_gra"] = df["d2cc_gra"] - df["d2cc_5mm"]
median_gra = df["delta_gra"].median()
print(f"图形优化: Δ 的中位数 {median_gra:.1f} Gy")
```

</details>

</div>

<div class="exercise">

**练习 3 · 轮到你: 多少位患者降到了 68 Gy 以下**

5 mm 计划达到 68 Gy 的 148 位患者里,换成图形优化后降到 68 Gy 以下的有几位?把人数存进 `n_crossed`。

```python
crossed = df["high_5mm"] & (df["d2cc_gra"] ___ 68)
n_crossed = crossed.sum()
print(f"{n_crossed} 位,占这 148 位的 {n_crossed / df['high_5mm'].sum():.1%}")
```

```pycheck
need("n_crossed")
if n_crossed == 54:
    ok("对了: 148 位里有 54 位(36.5%)换成图形优化后降到了 68 Gy 以下。体积优化是 36 位。")
if n_crossed == 94:
    fail("这是换了以后仍然达到 68 Gy 的人数: 要找的是降到 68 Gy 以下的,用 <。")
if n_crossed == 104:
    fail("这是全部 200 位里图形优化计划低于 68 Gy 的人数。别去掉前面的 df[\"high_5mm\"] &。")
fail(f"现在是 {n_crossed} 位,应该是 54 位。空里填一个比较符号。")
```

<details><summary>提示</summary>

"降到 68 Gy 以下"是 `< 68`。

</details>

<details><summary>参考答案</summary>

```python
crossed = df["high_5mm"] & (df["d2cc_gra"] < 68)
n_crossed = crossed.sum()
print(f"{n_crossed} 位,占这 148 位的 {n_crossed / df['high_5mm'].sum():.1%}")
```

</details>

</div>

## 5. 空着的格子: 缺失值

真实的表总有空格子: 没做的检查、没量的指标、没回来随访的患者。pandas 把空格子读成 **NaN**(not a number),叫**缺失值**(missing value)。先数一数每一列有几个:

```python
df.isna().sum()
```

`isna()` 把每个格子变成 True(空)或 False(不空),`sum()` 再按列数出 True 的个数。三列壁厚各有 4 个空: 有 4 位患者没有壁厚测量。把这 4 位患者找出来看看:

```python
df[df["wall_mean"].isna()][["id", "wall_min", "wall_mean", "wall_max", "wall_range"]]
```

三列壁厚是 NaN,这没问题。可是 `wall_range`(最大减最小)却是 **0.0**。这一列在 Excel 里是用公式算的,公式把空格子当成了 0,算出来就是 0 − 0 = 0。于是这 4 位没有测量的患者,在表里成了"阴道壁厚薄完全均匀"的患者。

这不是编出来的情节。我们整理真实研究的数据时就遇到过: 原始表里"最大壁厚减最小壁厚"一列是公式,没有测量的患者显示成 0。不改的话,这几个假的 0 会混进后面所有和壁厚有关的分析。

<div class="quiz">
<p><b>想一想</b>: 一位患者没有做壁厚测量。录入数据时,这一格应该怎么填?</p>
<button class="quiz-opt" data-why="0 会被当成真的测量值: 壁厚 0 mm。算平均数时它会把结果拉低,而且以后没有人能分清哪些 0 是真的。">填 0</button>
<button class="quiz-opt" data-correct data-why="空着,就是如实地说&quot;不知道&quot;。软件能认出空格子,算平均数时自动跳过,也能数出缺了几个。">空着不填</button>
<button class="quiz-opt" data-why="这样数据看起来完整了,其实多出一个编出来的值,数据的离散程度也被人为缩小。确实有用别的值来补缺失的方法,但那是分析时的一步,要在论文里说明;录入时应该空着。">填全体患者的平均值</button>
</div>

把这 4 个假的 0 改回缺失。`df.loc[行的条件, 列名] = 值` 的意思是: 把满足条件的那些行的这一列,改成这个值。

```python
print("改之前:", df["wall_range"].mean().round(2), "mm,最小", df["wall_range"].min())
df.loc[df["wall_min"].isna(), "wall_range"] = float("nan")
print("改之后:", df["wall_range"].mean().round(2), "mm,最小", df["wall_range"].min())
print("还剩", df["wall_range"].count(), "个有效值")
```

平均值从 2.89 变成 2.94 mm,最小值从 0 变成 0.8 mm。4 个假的 0 就把平均值拉低了 0.05 mm;如果缺的不是 4 位而是 40 位,结果就面目全非了。

pandas 算 `mean()`、`median()`、`sum()` 这些量时会**自动跳过缺失值**,`count()` 数的是有效值的个数。所以报告结果时要同时说清楚用了多少位患者:

```python
print("平均壁厚:", df["wall_mean"].mean().round(2), "mm,来自", df["wall_mean"].count(), "位患者")
print("如果把缺失当成 0:", df["wall_mean"].fillna(0).mean().round(2), "mm")
measured = df.dropna(subset=["wall_mean"])       # 只保留有壁厚测量的患者
print(len(measured), "位患者有壁厚测量")
```

`fillna(0)` 把缺失填成 0,这里只是为了演示它会把结果带偏,平时不要这样做。`dropna(subset=[...])` 去掉指定的列里有缺失的行;不写 `subset` 的话,任何一列有缺失的患者都会被去掉,列多的时候会丢掉很多人,所以总是写上 `subset`。

<div class="exercise">

**练习 4 · 轮到你: 有测量的患者**

有壁厚测量的患者有几位?存进 `n_measured`(`notna()` 和 `isna()` 正好相反)。再算外照射加近距离这一组的平均壁厚(`wall_mean`)的均值,存进 `mean_ebrt`。

```python
n_measured = df["wall_mean"].___().sum()
mean_ebrt = df.loc[df["arm"] == "EBRT+VCB", ___].mean()
print(f"{n_measured} 位有测量;外照射加近距离组的平均壁厚 {mean_ebrt:.2f} mm")
```

```pycheck
need("n_measured", "mean_ebrt")
if n_measured == 4:
    fail("4 是没有测量的人数(isna)。要数有测量的,用 notna()。")
if n_measured != 196:
    fail(f"n_measured 现在是 {n_measured},应该是 196。")
if close(mean_ebrt, 3.76, 0.006):
    ok("对了: 196 位患者有壁厚测量;外照射加近距离组的平均壁厚是 3.76 mm(单纯近距离组 3.62 mm)。pandas 算均值时自动跳过了缺失的患者。")
if close(mean_ebrt, 3.69, 0.006):
    fail("这是全体患者的均值。检查 .loc 里的行条件。")
fail(f"mean_ebrt 现在是 {mean_ebrt:.2f},应该是 3.76: 第二处填列名 \"wall_mean\"。")
```

<details><summary>提示</summary>

`df["wall_mean"].notna()`;`.loc[行的条件, "wall_mean"]`,列名要加引号。

</details>

<details><summary>参考答案</summary>

```python
n_measured = df["wall_mean"].notna().sum()
mean_ebrt = df.loc[df["arm"] == "EBRT+VCB", "wall_mean"].mean()
print(f"{n_measured} 位有测量;外照射加近距离组的平均壁厚 {mean_ebrt:.2f} mm")
```

</details>

</div>

<div class="exercise">

**练习 5 · 改错**

这一句想数出单纯近距离、并且 70 岁及以上的患者,但运行会报错。读报错的最后一行,把它改对,人数存在 `n_vcb_old` 里。

```python
n_vcb_old = len(df[df["arm"] == "VCB" and df["age"] >= 70])
print(n_vcb_old, "位")
```

```pycheck
need("n_vcb_old")
if n_vcb_old == 30:
    ok('对了: 30 位。对一整列做"并且",要用 &,而且两个条件各加括号。and 只能用在单个的 True 或 False 上。')
fail(f"现在是 {n_vcb_old} 位,应该是 30 位。两个条件各加括号,中间用 &。")
```

<details><summary>提示</summary>

报错说 "The truth value of a Series is ambiguous": `and` 不知道怎么处理一整列的 True 和 False。换成 `&`,并且给两个条件各加上括号。

</details>

<details><summary>参考答案</summary>

```python
n_vcb_old = len(df[(df["arm"] == "VCB") & (df["age"] >= 70)])
print(n_vcb_old, "位")
```

</details>

</div>

## 6. 例题: 把论文里的一张表敲进来

表不一定要从文件里读,数据少的时候可以直接敲进来: 一个字典,键是列名,值是这一列的数。

下面的数字来自一篇已发表的论文(Qiang 等,*Cancers* 2026;18:2350)的 Table 5: 外照射加近距离、出现了晚期阴道副作用的 34 位患者,同一位患者的三种计划方式下,阴道 D2cc 的总 EQD2 的平均值和标准差(Gy)。

```python
paper = pd.DataFrame({
    "plan": ["5 mm", "Volume", "Graphical"],
    "mean": [70.8, 69.3, 68.1],
    "sd": [3.34, 4.06, 5.50],
})
paper["diff_vs_5mm"] = (paper["mean"] - paper.loc[0, "mean"]).round(1)
paper
```

`paper.loc[0, "mean"]` 取出第 0 行(5 mm)的平均值 70.8;整列减去它,就是每种方式比 5 mm 低多少: 体积优化低 1.5 Gy,图形优化低 2.7 Gy。

论文里印的差值是 −1.56 和 −2.69,和这里的 −1.5、−2.7 差一点。这不是谁算错了: 论文是用没有四舍五入的平均值相减的,我们手里只有保留了一位小数的平均值。**先算、最后再四舍五入**,是处理数据的一条通用规矩,所以中间结果尽量保留原样,只在显示的时候用 `round()`。

注意表里的平均值分别是 70.8、69.3 和 68.1 Gy,正好在 68 Gy 这条线附近: 同一位患者,换一种计划方式,就可能从线的一边跨到另一边。练习 3 数的就是这件事。

## 7. 存成文件

整理好的表可以存起来,下次直接读:

```python
df.to_csv("cohort_clean.csv", index=False)       # index=False: 不把行号存进去
print(pd.read_csv("cohort_clean.csv").shape)
```

在浏览器里,文件存在一块临时空间里,刷新页面就没了。在自己电脑上,它会出现在代码所在的文件夹里。存成 Excel 是 `df.to_excel("cohort_clean.xlsx", index=False)`。

**原始数据不要改**。读进来、在代码里整理、另存一个新文件;原始的那份表保持收到时的样子。这样任何时候都能从头重做一遍,也能回答"这个数是怎么来的"。

## 录入数据的几条规矩

如果表是你自己(或者你带的学生)在 Excel 里录的,下面几条能省掉后面大部分的清理工作:

- 一行一位患者,一列一个变量,第一行是列名。列名用英文字母、数字和下划线,不留空格。
- 一格只放一个值,不带单位(单位写在列名里,比如 `weight_kg`),不写"约""大于"。
- 没有的数据空着,不填 0,不填"无""未测"。
- 分类变量用统一的写法: 全表都写 `VCB`,不要有的写 `vcb`,有的后面多一个空格。
- 不合并单元格,不用颜色、加粗来表示信息(软件读不到颜色)。
- 姓名、病历号这些能认出患者的信息,不放进分析用的表。用研究里自己编的编号;编号和病历号的对照表单独、安全地保存。

## 小结

- `pd.read_csv` 把文件读成一张表(DataFrame);先看 `shape`、`dtypes`、`describe()`。
- `df["列"]` 取一列,`df[["列1", "列2"]]` 取几列,`df[条件]` 筛行;多个条件各加括号,用 `&` 和 `|`。
- 给 `df["新列"]` 赋值就加了一列;对 True/False 的一列,`sum()` 是人数,`mean()` 是比例。
- 空格子是缺失值(NaN),不是 0;`isna().sum()` 数缺了几个,计算时自动跳过,报告时说清用了多少位患者。
- 原始数据不改,整理的每一步都写在代码里。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 数据框 | DataFrame | pandas 里的一张表 |
| 一列 | Series、column | 表里的一个变量 |
| 索引 | index | 最左边的行号 |
| 数据类型 | data type (dtype) | 一列是整数、小数还是文字 |
| 筛选 | filter | 按条件留下一部分行 |
| 缺失值 | missing value (NaN) | 空着的格子 |
| 完整病例分析 | complete-case analysis | 只用没有缺失的患者来分析 |
| 逗号分隔文件 | CSV | 最简单的表格文件,列之间用逗号隔开 |

## 这一课没讲什么

- **分组汇总**(每一组各算一个中位数)和**合并两张表**: 第 3 课。
- **日期和时间**、**清理文字**(去空格、统一大小写): 用到的时候查 pandas 的文档,思路和这一课一样。
- **用别的值补缺失**(插补, imputation): 属于统计方法,统计机器学习课的第 5 课讲。

## 写进论文 Writing it up

<div class="paper">

缺失数据要在方法里交代怎么处理,在结果或表注里交代缺了多少:

> Vaginal wall thickness was not available in 4 of 200 patients (2%); analyses involving wall thickness were restricted to patients with available measurements. No missing values were imputed.

**审稿人可能问**: 缺失的数据是怎么处理的?会不会影响结论?

**怎么回答**: 说明每个变量缺了多少、为什么缺(没有做这项测量)、分析时怎么办(只用有测量的患者,没有用 0 或平均值去填)。缺得很少(这里 2%)时,对结论的影响通常很小;缺得多时,要比较有测量和没有测量的患者是否相似,并考虑插补。

**审稿人可能问**: 数据是怎么整理的,能重复吗?

**怎么回答**: 原始数据没有手工改动,所有整理步骤都在分析代码里,代码可以提供。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| `KeyError: 'Age'` | 表里没有这个列名。列名区分大小写,用 `df.columns` 看一下确切的写法。 |
| `ValueError: The truth value of a Series is ambiguous` | 对一整列用了 `and` 或 `or`。换成 `&` 或 `\|`,每个条件加括号。 |
| `TypeError: Cannot perform 'rand_' with a dtyped ... array and scalar of type [bool]` | 两个条件没有各自加括号,`&` 先和旁边的数字结合了。 |
| 筛出来的行数是 0 | 文字的写法和表里不一样(大小写、空格、加号)。用 `value_counts()` 看表里到底是怎么写的。 |
| 均值算出来是 NaN | 这一列全是缺失值,或者列的类型是文字。用 `dtypes` 和 `isna().sum()` 检查。 |

## 延伸阅读 Further reading

- pandas 官方的入门教程: [Getting started tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/)。前 6 节和这一课的内容对应,例子换成了另一套数据。

{{< practice >}}

**下一课**: 现在能对整张表算一个数了。临床论文的 Table 1 需要的是"每一组各算一遍",而且一位患者有三套计划。第 3 课讲分组汇总、宽表和长表,最后做出一张 Table 1。
