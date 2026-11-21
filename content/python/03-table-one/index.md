---
title: "Lesson 3 · 整理汇总和 Table 1 Summaries and Table 1"
summary: "每一组各算一遍;人数和百分比;宽表和长表;合并两张表;做出基线表 Table 1。"
weight: 30
toc: true
duration: "约 80 分钟,可以分两次学: 第 1–4 节,第 5–7 节"
prereq: "[第 2 课](../02-pandas/)"
questions:
  - "两组患者各自的中位数、各类的人数和百分比,怎么一次算出来?"
  - "一位患者有三套计划,数据该排成什么样?随访数据在另一张表里,怎么合进来?"
  - "论文的 Table 1 是怎么做出来的?"
objectives:
  - "按组算中位数、人数等汇总量"
  - "做人数和百分比的交叉表,分清行百分比和列百分比"
  - "把连续的数分成几段,比如 BMI 分级"
  - "在宽表和长表之间转换"
  - "按患者编号合并两张表,并检查合并的结果"
  - "用代码做出一张基线特征表 Table 1"
---

临床论文的第一张表几乎都长一个样: **Table 1**,各组患者的基线特征(baseline characteristics)。年龄的中位数是多少,各分期占百分之几,两组像不像。读论文时先看它,写论文时先做它。

做这张表要用到这一课的全部内容: 把患者分组、每组各算一遍、把人数变成百分比。另外还有两件整理数据的事: 一位患者有三套计划,数据有时要排成"一人一行",有时要排成"一套计划一行";随访的结果记在另一张表里,要按患者编号合进来。

> 🔬 第 7 节会对照一篇已发表论文的 Table 1(节选),看它每一格是怎么来的。

**练习怎么做**: 讲解里算的是阴道 **D2cc**,练习换成靶区 **D90**,同样的步骤你再做一遍。

## 0. 准备

和第 2 课是同一批模拟患者。这一次直接用课程提供的函数拿到整理好的表(没有那一列带假 0 的 `wall_range`),再把上一课算过的两列加上:

```python
import pandas as pd
from cohort import make_cohort, make_followup

df = make_cohort()
df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
df["delta_vol"] = df["d2cc_vol"] - df["d2cc_5mm"]
print(df.shape)
```

## 1. 每一组各算一遍: groupby

单纯近距离和外照射加近距离两组患者,年龄的中位数各是多少?上一课的办法是先筛出一组、算一次,再筛出另一组、再算一次。组多了就很啰嗦。pandas 用 **groupby**(分组)一句话做完:

```python
df.groupby("arm")["age"].median()
```

**它在算什么**

1. 按 `arm` 这一列的取值,把患者分成几堆: 这里是两堆。
2. 在每一堆里,取出 `age` 这一列,各算一个中位数。
3. 把每一堆的结果排成一张小表,最左边是组名。

所以这一句从左往右读就是: 按什么分组(`groupby("arm")`),看哪一列(`["age"]`),算什么(`.median()`)。三处都可以换:

```python
df.groupby("arm")[["age", "bmi", "wall_mean"]].median().round(1)
```

```python
df.groupby("arm")["d2cc_5mm"].agg(["count", "median", "min", "max"])
```

第一句一次看三列;第二句用 `agg`(aggregate,汇总)对同一列一次算几个量。外照射加近距离组 5 mm 计划的阴道 D2cc 中位数是 73.0 Gy,单纯近距离组 69.4 Gy: 前一组的总剂量里包含了外照射的部分,所以更高。

`count` 数的是**有效值**的个数,遇到缺失值会少:

```python
df.groupby("arm")["wall_mean"].agg(["size", "count"])
```

`size` 是每组的总人数(109 和 91),`count` 是有壁厚测量的人数(107 和 89): 两组各有 2 位患者没有测量。写 Table 1 时,有缺失的变量要用 `count` 的人数做分母,并在表注里说明。

<div class="exercise">

**练习 1 · 轮到你: 两组的 D90**

5 mm 计划的靶区 D90(`d90_5mm`),两种治疗方式各自的中位数是多少?结果存进 `d90_by_arm`。

```python
d90_by_arm = df.groupby(___)[___].median()
print(d90_by_arm)
```

```pycheck
need("d90_by_arm")
if "IA" in getattr(d90_by_arm, "index", []):
    fail('这是按分期分的组。按治疗方式分组: groupby("arm")。')
if not all(k in getattr(d90_by_arm, "index", []) for k in ["VCB", "EBRT+VCB"]):
    fail('结果里应该有 VCB 和 EBRT+VCB 两行: 第一处填 "arm"。')
a, b = d90_by_arm["EBRT+VCB"], d90_by_arm["VCB"]
if close(a, 61.4, 0.01) and close(b, 33.7, 0.01):
    ok("对了: 外照射加近距离组 61.4 Gy,单纯近距离组 33.7 Gy。两组差这么多,是因为前一组的总剂量里包含了外照射的部分。所以这门课里,凡是总剂量的比较,都要两组分开看。")
if close(a, 73.0, 0.01):
    fail('这是阴道 D2cc。题目要的是靶区 D90: 第二处填 "d90_5mm"。')
fail(f"现在是 {a:.2f} 和 {b:.2f},应该是 61.4 和 33.7。检查列名,并且用的是 median()。")
```

<details><summary>提示</summary>

按什么分组: `"arm"`;看哪一列: `"d90_5mm"`。两个都是文字,要加引号。

</details>

<details><summary>参考答案</summary>

```python
d90_by_arm = df.groupby("arm")["d90_5mm"].median()
print(d90_by_arm)
```

</details>

</div>

## 2. 分类变量: 人数和百分比

分期、分级这样的分类变量,汇总的办法是数人数、算百分比。一个变量用上一课的 `value_counts()`;两个变量交叉,用 `pd.crosstab`(交叉表):

```python
print(df["stage"].value_counts())
pd.crosstab(df["stage"], df["arm"])
```

交叉表的行是分期,列是治疗方式,每一格是人数: 单纯近距离的 91 位里,68 位是 IA 期;II 期的 29 位里,27 位做了外照射加近距离。这符合临床上的做法: 分期越晚,越倾向于加外照射。

人数要变成百分比,问题是**除以谁**。加上 `normalize`:

```python
(pd.crosstab(df["stage"], df["arm"], normalize="columns") * 100).round(1)
```

`normalize="columns"` 让每一**列**加起来是 100%: 单纯近距离的患者里 74.7% 是 IA 期,外照射加近距离的患者里只有 29.4%。换成 `normalize="index"`,就是每一**行**加起来是 100%。

<div class="quiz">
<p><b>想一想</b>: Table 1 按治疗方式分成两列,要写"外照射加近距离组里,II 期占百分之几"。用上面这张交叉表,应该怎么算?</p>
<button class="quiz-opt" data-correct data-why="分母是这一组的总人数: 27 / 109 = 24.8%。Table 1 按组分列时,百分比都是按列算的,读别人的 Table 1 也先确认这一点。">27 除以这一列的合计 109(列百分比)</button>
<button class="quiz-opt" data-why="27 / 29 = 93.1% 回答的是另一个问题: II 期的患者里,有多少做了外照射加近距离。两个数都对,问的事情不一样。">27 除以这一行的合计 29(行百分比)</button>
<button class="quiz-opt" data-why="27 / 200 = 13.5% 是 II 期并且做了外照射加近距离的患者占全部患者的比例,不是组内的构成。">27 除以总人数 200</button>
</div>

<div class="exercise">

**练习 2 · 轮到你: 分级的构成**

外照射加近距离组里,2 级(`grade` 是 2)的患者占百分之几?先做一张按列算百分比的交叉表 `pct`,再从里面取出这个数,存进 `pct_g2_ebrt`。

```python
pct = pd.crosstab(df["grade"], df["arm"], normalize=___) * 100
pct_g2_ebrt = pct.loc[2, "EBRT+VCB"]
print(pct.round(1))
```

```pycheck
need("pct_g2_ebrt")
if close(pct_g2_ebrt, 45.0, 0.06):
    ok("对了: 外照射加近距离组的 109 位里,49 位是 2 级,占 45.0%。pct.loc[2, \"EBRT+VCB\"] 按行名和列名取出一格。")
if close(pct_g2_ebrt, 57.6, 0.06):
    fail('这是行百分比: 2 级的患者里有 57.6% 做了外照射加近距离。题目问的是这一组里 2 级占多少,要按列算: normalize="columns"。')
if close(pct_g2_ebrt, 24.5, 0.06):
    fail('这是占全部 200 位患者的百分比。要按列算: normalize="columns"。')
fail(f'现在是 {pct_g2_ebrt:.1f},应该是 45.0。normalize 填 "columns"。')
```

<details><summary>提示</summary>

每一列加起来是 100%: `normalize="columns"`。

</details>

<details><summary>参考答案</summary>

```python
pct = pd.crosstab(df["grade"], df["arm"], normalize="columns") * 100
pct_g2_ebrt = pct.loc[2, "EBRT+VCB"]
print(pct.round(1))
```

</details>

</div>

## 3. 把数字分成几段

临床上常把连续的数分成几段来说: BMI 分成偏瘦、正常、超重、肥胖,年龄分成几个年龄组。`pd.cut` 做这件事: 给出分界点和每一段的名字。

```python
df["bmi_cat"] = pd.cut(df["bmi"], bins=[0, 18.5, 25, 30, 100], right=False,
                       labels=["<18.5", "18.5-24.9", "25-29.9", ">=30"])
df["bmi_cat"].value_counts(sort=False)
```

`bins` 是分界点: 0 到 18.5、18.5 到 25、25 到 30、30 到 100,一共四段。`right=False` 表示每一段包含左边的分界点、不包含右边的,所以 BMI 正好是 25 的患者算在"25–29.9"里,和 WHO 的定义一致。分界点上的患者归哪一段,每次都要想清楚,第 1 课的"达到 68 Gy 算不算"是同一个问题。

这批模拟患者里,超重和肥胖的占了大多数(子宫内膜癌的患者确实常常如此)。分段让结果好读,代价是丢掉了信息: 30.1 和 45 都叫"肥胖"。做统计分析时,尽量用原来的连续值;分段主要用在表格和描述里。

## 4. 论文里的写法: 中位数(四分位间距)

Table 1 里的连续变量,常写成"中位数(下四分位数–上四分位数)",比如 `64 (58–71)`。为什么常用中位数而不是均值,第 5 课再讲。这里先写一个函数,把一列数变成这样的一段文字:

```python
def median_iqr(x, digits=1):
    """一列数 -> '中位数 (Q1–Q3)' 这样的文字。"""
    q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
    return f"{med:.{digits}f} ({q1:.{digits}f}–{q3:.{digits}f})"

print(median_iqr(df["age"], 0))
df.groupby("arm")["age"].apply(median_iqr, digits=0)
```

`x.quantile([0.25, 0.5, 0.75])` 一次算出三个分位数;`{med:.{digits}f}` 按指定的小数位数显示。`groupby(...).apply(函数)` 对每一组调用一次这个函数: 第 1 节的 `median()` 是 pandas 自带的,`apply` 让你用自己写的。

## 5. 宽表和长表

一位患者有三套计划。现在的表是**宽表**(wide): 一位患者一行,三套计划的 D2cc 并排成三列。要回答"三套计划各自的中位数是多少",得一列一列地算。换一种排法会方便得多: **长表**(long),一位患者的一套计划占一行。

```python
long = df.melt(id_vars=["id", "arm"], value_vars=["d2cc_5mm", "d2cc_vol", "d2cc_gra"],
               var_name="plan", value_name="d2cc")
long["plan"] = long["plan"].str.replace("d2cc_", "")
print(long.shape)
long.sort_values(["id", "plan"]).head(6)
```

`melt`(熔化)把宽表变长: `id_vars` 是每一行都要带着的列(患者编号和治疗方式),`value_vars` 是要叠起来的三列;原来的列名放进新的一列 `plan`,原来的值放进 `d2cc`。200 位患者、三套计划,所以长表有 600 行。第二句把 `d2cc_5mm` 这样的名字去掉前缀,只留 `5mm`。

长表配上 groupby,按计划、按治疗方式和计划汇总都只要一句:

```python
print(long.groupby("plan")["d2cc"].median())
long.groupby(["arm", "plan"])["d2cc"].median().unstack()
```

按两个变量分组时,`groupby` 里放一个列表;`unstack()` 把结果排成"行是治疗方式、列是计划"的小表。两组里都是 5 mm 计划的 D2cc 最高,图形优化最低。

需要的时候,`pivot` 把长表变回宽表:

```python
long.pivot(index="id", columns="plan", values="d2cc").head(3)
```

**什么时候用哪一种**: 算同一位患者两套计划的差(比如 `delta_vol`),用宽表,两列相减就行;按计划分组汇总、画图(第 4 课),用长表。两种排法装的是同样的数据,随时可以互相转换。

<div class="quiz">
<p><b>想一想</b>: 要比较体积优化和 5 mm 两套计划,下面哪种做法保留了"这两个数来自同一位患者"这个信息?</p>
<button class="quiz-opt" data-correct data-why="每位患者自己和自己比,患者之间的差别(有人整体偏高,有人整体偏低)在相减时抵消了。这叫配对(paired)的比较,第 6 课会用到。">在宽表里两列相减,得到每位患者自己的差</button>
<button class="quiz-opt" data-why="这样比的是两堆数,丢掉了谁和谁是同一位患者。患者之间本来的差别会混进来,把计划之间的差别盖住。">在长表里把 5mm 的 200 个数和 vol 的 200 个数当成两组患者来比</button>
<button class="quiz-opt" data-why="两个中位数相减是可以算,但它不等于每位患者的差的中位数,也没有用上配对的信息。">分别算两套计划的中位数,再相减</button>
</div>

<div class="exercise">

**练习 3 · 轮到你: D90 的长表**

把三套计划的 D90 变成长表 `d90_long`(值的那一列叫 `d90`),再按治疗方式和计划算中位数,排成小表 `d90_med`。

```python
d90_long = df.melt(id_vars=["id", "arm"], value_vars=["d90_5mm", "d90_vol", "d90_gra"],
                   var_name="plan", value_name=___)
d90_med = d90_long.groupby(["arm", ___])["d90"].median().unstack()
print(d90_med)
```

```pycheck
need("d90_long", "d90_med")
if d90_long.shape != (600, 4):
    fail(f"d90_long 应该有 600 行、4 列,现在是 {d90_long.shape}。")
if "d90" not in d90_long.columns:
    fail('值的那一列要叫 d90: value_name="d90"。')
try:
    a, b = d90_med.loc["VCB", "d90_gra"], d90_med.loc["EBRT+VCB", "d90_5mm"]
except KeyError:
    fail('d90_med 的行应该是治疗方式,列应该是三套计划: groupby(["arm", "plan"])。')
if close(a, 30.3, 0.01) and close(b, 61.4, 0.01):
    ok("对了。两组里都是 5 mm 计划的 D90 最高,图形优化最低: 阴道的剂量降下来,靶区的剂量也跟着降了一些。这是换计划的代价,后面的课会反复看到。")
fail("数值不对。检查 groupby 里是不是 arm 和 plan,算的是不是中位数。")
```

<details><summary>提示</summary>

`value_name="d90"`;第二处是计划那一列的名字 `"plan"`。

</details>

<details><summary>参考答案</summary>

```python
d90_long = df.melt(id_vars=["id", "arm"], value_vars=["d90_5mm", "d90_vol", "d90_gra"],
                   var_name="plan", value_name="d90")
d90_med = d90_long.groupby(["arm", "plan"])["d90"].median().unstack()
print(d90_med)
```

</details>

</div>

## 6. 合并两张表

研究的数据很少只有一张表: 基线和剂量在一张表里,随访的结果在另一张表里,就像化验结果和病历首页不在同一页上。课程的随访表这样拿到:

```python
fu = make_followup()
print(fu.shape)
fu.head(3)
```

每位有随访的患者一行: 随访了几个月(`fu_months`),有没有出现晚期阴道副作用(`tox`,1 是有,0 是没有)。注意两件事: 只有 188 行,有 12 位患者没有随访记录;行的顺序和 `df` 也不一样。所以不能把两张表并排一贴了事,要**按患者编号对上**。这就是 `merge`(合并):

```python
merged = df.merge(fu, on="id", how="left")
print(merged.shape)
print(merged["tox"].isna().sum(), "位患者没有随访记录")
```

**它在算什么**

1. 对左边的表(`df`)的每一行,拿着它的 `id`,到右边的表(`fu`)里去找 `id` 相同的那一行。
2. 找到了,就把右边那一行的列接在后面。
3. 没找到,`how="left"` 的意思是左边的行照样保留,接上去的列空着(NaN)。

**打个比方**: 查房前把化验单按床号夹进病历。每本病历都留着,化验单没回来的,那一页就空着。

合并是最容易悄悄出错的一步,做完一定要检查:

- **行数**: `how="left"` 合并后,行数应该和左边的表一样(这里还是 200)。如果变多了,说明右边的表里同一个 `id` 出现了不止一次,左边的行被复制了。
- **对上了多少**: 数一数新接上来的列有几个空的(这里 12 个),和你预期的是否一致。如果大部分都是空的,多半是两张表里编号的写法不一样(比如 `P001` 和 `p001`,或者一边是数字、一边是文字)。

合并以后,就能把剂量和随访结果放在一起看了:

```python
merged["high_5mm"] = merged["d2cc_5mm"] >= 68
merged.groupby("high_5mm")["tox"].agg(["count", "sum", "mean"]).round(3)
```

`tox` 是 0 和 1,所以 `sum` 是出现副作用的人数,`mean` 是比例,`count` 是有随访的人数。在这批模拟患者里,D2cc 达到 68 Gy 的 139 位里有 36 位出现了晚期副作用(25.9%),没达到的 49 位里有 6 位(12.2%)。这个差别算不算"真的",是第 6 课的问题。

<div class="exercise">

**练习 4 · 轮到你: 两种治疗方式的副作用比例**

用上面合并好的 `merged`,算两种治疗方式各自出现晚期副作用的比例,存进 `rate_by_arm`;再数出外照射加近距离组里有随访记录的人数,存进 `n_fu_ebrt`。

```python
rate_by_arm = merged.groupby(___)["tox"].mean()
n_fu_ebrt = merged.loc[merged["arm"] == "EBRT+VCB", "tox"].___()
print(rate_by_arm.round(3))
print(n_fu_ebrt, "位有随访")
```

```pycheck
need("rate_by_arm", "n_fu_ebrt")
if not all(k in getattr(rate_by_arm, "index", []) for k in ["VCB", "EBRT+VCB"]):
    fail('rate_by_arm 应该按治疗方式分组: groupby("arm")。')
if n_fu_ebrt == 109:
    fail("109 是这一组的总人数。有随访记录的人数要数有效值: count()。")
if n_fu_ebrt == 26:
    fail("26 是出现副作用的人数(sum)。有随访记录的人数是 count()。")
if n_fu_ebrt != 102:
    fail(f"n_fu_ebrt 现在是 {n_fu_ebrt},应该是 102。")
if close(rate_by_arm["EBRT+VCB"], 0.2549, 0.001) and close(rate_by_arm["VCB"], 0.1860, 0.001):
    ok("对了: 外照射加近距离组 25.5%(102 位有随访的患者里 26 位),单纯近距离组 18.6%(86 位里 16 位)。算比例时,没有随访的患者被自动跳过了: 分母是 102 和 86,不是 109 和 91。论文里要把分母写清楚。")
fail("比例不对: 对 tox 这一列用 mean()。")
```

<details><summary>提示</summary>

分组用 `"arm"`;有效值的个数是 `count()`。

</details>

<details><summary>参考答案</summary>

```python
rate_by_arm = merged.groupby("arm")["tox"].mean()
n_fu_ebrt = merged.loc[merged["arm"] == "EBRT+VCB", "tox"].count()
print(rate_by_arm.round(3))
print(n_fu_ebrt, "位有随访")
```

</details>

</div>

## 7. 做一张 Table 1

先看一张真实的。下面是一篇已发表论文(Qiang 等,*Cancers* 2026;18:2350)Table 1 的节选: 外照射加近距离的 120 位患者,按有没有出现晚期阴道副作用分成两列。

| | 全部(n = 120) | 没有副作用(n = 86) | 有副作用(n = 34) | *p* |
|---|---|---|---|---|
| 年龄(岁),均值(标准差) | 64.9 (10.6) | 65.6 (10.2) | 63.0 (11.3) | 0.242 |
| 平均壁厚(mm),均值(标准差) | 3.09 (0.48) | 3.10 (0.50) | 3.05 (0.42) | 0.576 |
| 阴道 D2cc 的总 EQD2 ≤ 68 Gy,人数(%) | 50 (41.7) | 41 (47.7) | 9 (26.5) | 0.055 |
| 阴道 D2cc 的总 EQD2 > 68 Gy,人数(%) | 70 (58.3) | 45 (52.3) | 25 (73.5) | |

读这张表,用的就是这一课的内容:

- **列是分组**(这里按结局分),每一格是这一组的一个汇总量: 第 1 节的 groupby。
- 连续变量写"均值(标准差)",分类变量写"人数(百分比)",表头里写明了是哪一种。
- **百分比是按列算的**(第 2 节): 26.5% 是 9 / 34,不是 9 / 50。
- 最后一列是比较两组的 *p* 值,怎么算的、怎么读,第 6 课讲。

现在用模拟队列做一张自己的,按治疗方式分列。先写两个小函数,再写一个函数算出"一列":

```python
def n_pct(flag):
    """一列 True/False -> '人数 (百分比)'。"""
    return f"{flag.sum()} ({flag.mean():.1%})"

def table1_column(d):
    return pd.Series({
        "Patients, n": len(d),
        "Age, years": median_iqr(d["age"], 0),
        "BMI, kg/m2": median_iqr(d["bmi"]),
        "Stage IA, n (%)": n_pct(d["stage"] == "IA"),
        "Stage IB, n (%)": n_pct(d["stage"] == "IB"),
        "Stage II, n (%)": n_pct(d["stage"] == "II"),
        "Grade 3, n (%)": n_pct(d["grade"] == 3),
        "Mean wall thickness, mm": median_iqr(d["wall_mean"]),
        "Vaginal D2cc, 5 mm plan, Gy": median_iqr(d["d2cc_5mm"]),
    })

table1 = pd.DataFrame({
    "All": table1_column(df),
    "VCB alone": table1_column(df[df["arm"] == "VCB"]),
    "EBRT+VCB": table1_column(df[df["arm"] == "EBRT+VCB"]),
})
table1
```

`table1_column(d)` 收进一组患者,交出这一组在 Table 1 里的那一列: 左边是每一行的名字,右边是算好的文字。然后对全部患者、单纯近距离组、外照射加近距离组各调用一次,拼成三列。

这样做的好处是: 数据更新了,重新运行一遍,整张表就更新了,不会有哪一格忘了改。要加一个变量,只要在函数里加一行。表做好以后存成文件,再贴进 Word 调整格式:

```python
table1.to_csv("table1.csv")
print(pd.read_csv("table1.csv", index_col=0).shape)
```

<div class="exercise">

**练习 5 · 轮到你: 给 Table 1 加一行**

给 Table 1 加一行"5 mm 计划的阴道 D2cc 达到 68 Gy 的人数(百分比)"。用 `n_pct` 分别算出三列里的这一格,存进 `row`。

```python
groups = {"All": df, "VCB alone": df[df["arm"] == "VCB"], "EBRT+VCB": df[df["arm"] == "EBRT+VCB"]}
row = {name: n_pct(d["d2cc_5mm"] ___ 68) for name, d in groups.items()}
print(row)
```

```pycheck
need("row")
want = {"All": "148 (74.0%)", "VCB alone": "58 (63.7%)", "EBRT+VCB": "90 (82.6%)"}
if row == want:
    ok("对了: 全部患者 148 (74.0%),单纯近距离 58 (63.7%),外照射加近距离 90 (82.6%)。把它加进 table1_column 里,就是 Table 1 的一行。花括号里的那一句,是对三组各算一次的简写。")
if row.get("All") == "52 (26.0%)":
    fail("这是没有达到 68 Gy 的人数。达到 68 Gy 是 >= 68。")
fail(f"现在是 {row}。空里填比较符号 >=。")
```

<details><summary>提示</summary>

"达到 68 Gy"是 `>= 68`。

</details>

<details><summary>参考答案</summary>

```python
groups = {"All": df, "VCB alone": df[df["arm"] == "VCB"], "EBRT+VCB": df[df["arm"] == "EBRT+VCB"]}
row = {name: n_pct(d["d2cc_5mm"] >= 68) for name, d in groups.items()}
print(row)
```

</details>

</div>

## 小结

- `df.groupby("组")["列"].median()`: 按什么分组、看哪一列、算什么。`agg` 一次算几个量,`apply` 用自己写的函数。
- 分类变量用 `value_counts` 和 `pd.crosstab`;百分比先想清楚除以谁: Table 1 按组分列时用列百分比。
- `pd.cut` 把连续的数分段;分界点归哪一段要想清楚。
- 宽表(一人一行)方便算同一位患者的差;长表(一套计划一行)方便分组汇总和画图;`melt` 和 `pivot` 互相转换。
- `merge` 按编号对上两张表;合并以后检查行数和空了多少。
- Table 1 用函数生成,数据更新后重新运行即可。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 基线特征 | baseline characteristics | 治疗开始前患者的情况,Table 1 的内容 |
| 分组汇总 | group by、aggregate | 每一组各算一个汇总量 |
| 交叉表 | cross-tabulation、contingency table | 两个分类变量交叉的人数表 |
| 行百分比、列百分比 | row percentage、column percentage | 除以行的合计,还是除以列的合计 |
| 四分位间距 | interquartile range (IQR) | 下四分位数到上四分位数 |
| 宽表、长表 | wide format、long format | 一人一行,还是一次测量一行 |
| 合并 | merge、join | 按共同的编号把两张表对上 |
| 失访 | lost to follow-up | 没有随访记录的患者 |

## 这一课没讲什么

- **Table 1 里的 *p* 值**: 怎么算、要不要放,第 6 课讲。
- **均值还是中位数**、标准差还是四分位间距: 第 5 课。
- **一对多的合并**(一位患者有多次随访记录): 思路相同,合并前先想好一位患者该占几行。

## 写进论文 Writing it up

<div class="paper">

方法里用一句话说明汇总的方式:

> Continuous variables are summarised as median (interquartile range) and categorical variables as number (percentage). Percentages were calculated among patients with available data.

**审稿人可能问**: Table 1 里的百分比,分母是什么?各行的人数为什么加起来不等于总人数?

**怎么回答**: 在表注里写明百分比按列计算,有缺失的变量写出实际用的人数(比如 "wall thickness available in 196 patients")。

**审稿人可能问**: 有多少患者失访?失访的患者和其他患者有没有差别?

**怎么回答**: 给出每组有随访的人数(练习 4 的 102 和 86),比例的分母用有随访的人数,并在结果里交代失访的人数。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| `KeyError: 'arm'` | 分组或取列用的名字表里没有。`melt` 以后只剩下 `id_vars` 里列出的那几列,要用的列记得带上。 |
| 合并以后行数变多了 | 右边的表里同一个编号出现了多次。先用 `fu["id"].value_counts().head()` 检查,或者在 `merge` 里加上 `validate="one_to_one"`,有重复时它会直接报错。 |
| 合并以后新的列全是 NaN | 两张表的编号对不上: 大小写、空格,或者一边是数字、一边是文字。分别用 `dtypes` 和 `head()` 看一下。 |
| 百分比加起来不是 100% | `normalize` 的方向反了,或者有缺失的患者没有算进去。 |

## 延伸阅读 Further reading

- pandas 用户指南: [Group by: split-apply-combine](https://pandas.pydata.org/docs/user_guide/groupby.html) 和 [Reshaping and pivot tables](https://pandas.pydata.org/docs/user_guide/reshaping.html)。
- von Elm E, et al. [The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement](https://www.bmj.com/content/335/7624/806). *BMJ* 2007;335:806。观察性研究的报告规范,其中有 Table 1 该写什么。

{{< practice >}}

**下一课**: 一张表只能给出几个数。数据的样子(有没有长尾、有没有离得很远的患者、两个量是不是一起变)要画出来才看得见。第 4 课讲画图。
