---
title: "练习 2 · 读入和清理审计数据"
summary: "读入 200 位患者的表,找出重复、写法不统一和丢了小数点的地方,算出每位患者的总剂量。"
weight: 20
toc: false
duration: "约 40 分钟"
prereq: "[Python 课第 2 课](/python/02-pandas/);[练习 1](../01-one-patient/)"
questions:
  - "科里导出的这张表,能直接用吗?有哪些问题?"
  - "200 位患者里,达到三条审计标准的各有多少?"
objectives:
  - "读表、看表、筛行、算新的一列"
  - "找出并改正数据里的错误"
  - "缺失值的处理"
---

物理师从计划系统和登记表里导出了一份 Excel,另存成了 `cervix.csv`。拿到手的第一件事不是算结果,而是**查这张表**: 导出的数据几乎总有问题。

审计标准(背景见[练习目录](../)): HR-CTV D90 ≥ 85 Gy,直肠 D2cc < 65 Gy,膀胱 D2cc < 80 Gy,都是外照射加 4 次近距离的总 EQD2。

先运行这一块。`eqd2` 就是你在练习 1 写的那个函数,这里直接拿来用;它对一整列也能算。

```python
import pandas as pd
from cervix import eqd2            # eqd2(d, n, ab),和练习 1 里你写的一样

raw = pd.read_csv("cervix.csv")
print("准备好了:", raw.shape)
raw.head(3)
```

<div class="steps" data-steps>

<div class="step exercise">

**第 1 步 · 有没有人被录了两次**

表里应该是 200 位患者。数一数有几行是多录的(患者编号 `id` 在前面已经出现过),存进 `n_dup`;然后去掉重复的行,把干净的表存进 `df`。后面几步都在 `df` 上做。

```python
# 在这里写你的代码。结果存进 n_dup 和 df。
```

```pycheck
need("n_dup", "df")
if n_dup != 1:
    fail(f"n_dup 现在是 {n_dup},应该是 1: 有一位患者多录了一行。df[列].duplicated() 标出重复出现的值。")
if len(df) == 201:
    fail("df 还是 201 行: 重复的那一行还没有去掉。")
if len(df) != 200:
    fail(f"df 应该有 200 行,现在是 {len(df)} 行。")
if not df["id"].is_unique:
    fail("df 里的患者编号还有重复。")
ok("对了: 表有 201 行,编号 C057 的患者被录了两次;去掉以后是 200 位。重复录入会让这位患者在所有统计里算两遍,而且不查就永远发现不了。")
```

<details><summary>提示 1: 思路</summary>

pandas 有两个现成的方法: 一个标出"前面已经出现过"的行,一个把重复的行去掉。标出来的是一列 True 和 False,数 True 的个数用 `sum()`。

</details>

<details><summary>提示 2: 用到的函数</summary>

`raw["id"].duplicated().sum()`;`raw.drop_duplicates()`。

</details>

<details><summary>参考答案</summary>

```python
n_dup = raw["id"].duplicated().sum()
print("重复的行:", n_dup)
print(raw[raw["id"].duplicated(keep=False)][["id", "age", "stage", "technique"]])

df = raw.drop_duplicates()
print(len(df), "位患者")
```

</details>

</div>

<div class="step exercise">

**第 2 步 · 施源器到底有几种**

`technique` 这一列应该只有 `IC` 和 `IC/IS` 两种。先看看表里实际有哪些写法,再把它们统一成这两种(改 `df["technique"]`)。

```python
# 在这里写你的代码。把 df["technique"] 统一成 "IC" 和 "IC/IS" 两种。
```

```pycheck
need("df")
counts = df["technique"].value_counts().to_dict()
if set(counts) != {"IC", "IC/IS"}:
    fail(f"现在 technique 里有这些写法: {counts}。应该只剩 IC 和 IC/IS。")
if counts != {"IC": 130, "IC/IS": 70}:
    fail(f"两种的人数应该是 130 和 70,现在是 {counts}。检查有没有把哪一种改错了。")
ok('对了: 原来有四种写法(IC、ic、IC/IS、IC+IS),统一以后是单纯腔内 130 位,腔内加插植 70 位。不统一的话,按施源器分组时 "ic" 的 6 位患者会自成一组。分类变量拿到手,先 value_counts() 看一遍。')
```

<details><summary>提示 1: 思路</summary>

第 2 课说过: 分类变量先用 `value_counts()` 看有哪些取值。看到不该有的写法以后,把它们换成正确的。

</details>

<details><summary>提示 2: 用到的函数</summary>

`df["technique"].replace({"旧的写法": "新的写法", ...})` 交回替换以后的一列,把它存回 `df["technique"]`。

</details>

<details><summary>参考答案</summary>

```python
print(df["technique"].value_counts())
df["technique"] = df["technique"].replace({"ic": "IC", "IC+IS": "IC/IS"})
print(df["technique"].value_counts())
```

</details>

</div>

<div class="step exercise">

**第 3 步 · 一个不可能的剂量**

近距离每次的 HR-CTV D90 一般在 5 到 10 Gy 之间。检查 `d90_1` 到 `d90_4` 四列,找出不可能的值。假设你已经核对过原始计划,确认是录入时丢了小数点(比如 7.6 录成了 76)。把它改正。

```python
# 在这里写你的代码。找出并改正 df 里不可能的 D90。
```

```pycheck
need("df")
cols = ["d90_1", "d90_2", "d90_3", "d90_4"]
if df[cols].max().max() > 10:
    fail(f"四列里最大的值还是 {df[cols].max().max()}: 那个不可能的剂量还在。用 describe() 或 max() 找到它在哪一列。")
if len(df) != 200:
    fail(f"df 应该还是 200 位患者,现在是 {len(df)} 位。有问题的值是改正,不是把患者删掉。")
v = float(df.loc[df["id"] == "C037", "d90_3"].iloc[0])
if not close(v, 7.6, 0.001):
    fail(f"C037 的 d90_3 应该改成 7.6,现在是 {v}。")
ok("对了: C037 第 3 次的 D90 录成了 76 Gy,改回 7.6 Gy。如果不改,这位患者的总 EQD2 会算成 623 Gy,把均值和图都带偏。数值的范围合不合理,是每份数据都要查的一项。")
```

<details><summary>提示 1: 思路</summary>

先看四列各自的最大值,找到是哪一列;再把那一列里大于 10 的那一行找出来;最后用第 2 课的 `df.loc[行的条件, 列名] = 值` 改掉它。

</details>

<details><summary>提示 2: 用到的写法</summary>

`df[["d90_1", "d90_2", "d90_3", "d90_4"]].max()`;`df[df["d90_3"] > 10]`;`df.loc[df["d90_3"] > 10, "d90_3"] = 7.6`。

</details>

<details><summary>参考答案</summary>

```python
cols = ["d90_1", "d90_2", "d90_3", "d90_4"]
print(df[cols].max())
print(df[df["d90_3"] > 10][["id"] + cols])

df.loc[df["d90_3"] > 10, "d90_3"] = 7.6
print(df[cols].max())
```

</details>

</div>

<div class="step">

<div class="quiz">
<p><b>第 4 步 · 想一想</b>: 用 <code>df.isna().sum()</code> 会看到,有 5 位患者的 HR-CTV 体积(<code>hrctv_vol</code>)是空的: 他们近距离时没有做 MRI。这 5 个空格子怎么办?</p>
<button class="quiz-opt" data-why="填 0 会把 5 位患者变成'肿瘤体积为 0'。体积的中位数会被拉低,体积和剂量的关系也会被带偏。">填 0</button>
<button class="quiz-opt" data-why="体积是缺失的,但这 5 位患者的剂量、施源器、随访都在。删掉他们,达标比例这些不需要体积的结果就白白少了 5 位患者。">把这 5 位患者整行删掉</button>
<button class="quiz-opt" data-correct data-why="空着就是如实地说不知道。算达标比例时 200 位都能用;分析体积时只用有体积的 195 位,并在报告里写明这个人数。">保持缺失;用到体积的分析只用有体积的 195 位,并写明人数</button>
</div>

</div>

<div class="step exercise">

**第 5 步 · 每位患者的总剂量**

给 `df` 加三列: `d90_total`(HR-CTV D90,α/β = 10)、`rectum_total` 和 `bladder_total`(直肠和膀胱的 D2cc,α/β = 3)。每一列都是外照射的 EQD2 加上 4 次近距离各自的 EQD2。注意外照射的剂量和次数不是每位患者都一样(看 `ebrt_gy` 和 `ebrt_fx` 两列)。

```python
# 在这里写你的代码。给 df 加上 d90_total、rectum_total、bladder_total 三列。
```

```pycheck
need("df")
from cervix import load_cervix
want = load_cervix().set_index("id")
for col in ["d90_total", "rectum_total", "bladder_total"]:
    if col not in df.columns:
        fail(f"df 里还没有 {col} 这一列。")
got = df.set_index("id")
gap = (got["d90_total"] - want["d90_total"]).abs()
if gap.max() > 0.01:
    if close(got["d90_total"].median(), want["d90_total"].median() - 44.25, 0.6):
        fail("d90_total 少了外照射的部分: 外照射的 EQD2 也要加进去。")
    if (gap > 0.01).sum() == 22:
        fail("有 22 位患者的 d90_total 不对: 他们的外照射是 50.4 Gy 分 28 次。外照射的部分要用每位患者自己的 ebrt_gy 和 ebrt_fx 来算,不能写死成 45 和 25。")
    fail(f"d90_total 不对(中位数应该是 92.3 Gy,现在是 {got['d90_total'].median():.1f})。每一次近距离用 eqd2(那一列, 1, 10),再加上外照射的 eqd2(ebrt_gy / ebrt_fx, ebrt_fx, 10)。")
for col, name in [("rectum_total", "直肠"), ("bladder_total", "膀胱")]:
    if (got[col] - want[col]).abs().max() > 0.01:
        fail(f"{col} 不对。{name}用 α/β = 3: 外照射和 4 次近距离都按 3 来算。")
ok("对了: 三列总剂量的中位数是 HR-CTV D90 92.3 Gy、直肠 64.7 Gy、膀胱 73.8 Gy。eqd2 对一整列也能算,所以 200 位患者不用写循环;有 22 位患者的外照射是 50.4 Gy 分 28 次,也都按自己的方案算对了。")
```

<details><summary>提示 1: 思路</summary>

和练习 1 的 `total_eqd2` 是同一件事,只是现在每个量是一列。先算外照射的 EQD2(每次剂量是 `ebrt_gy / ebrt_fx`),再把四次近距离的 EQD2 一列一列加上去。三个总剂量做法相同,只是列名和 α/β 不同。

</details>

<details><summary>提示 2: 用到的写法</summary>

`ebrt_10 = eqd2(df["ebrt_gy"] / df["ebrt_fx"], df["ebrt_fx"], 10)`;`df["d90_total"] = ebrt_10 + eqd2(df["d90_1"], 1, 10) + eqd2(df["d90_2"], 1, 10) + …`。也可以用循环: 列名用 `f"d90_{k}"` 拼出来,`k` 从 1 到 4。

</details>

<details><summary>参考答案</summary>

```python
per_fraction = df["ebrt_gy"] / df["ebrt_fx"]
for name, ab in [("d90", 10), ("rectum", 3), ("bladder", 3)]:
    total = eqd2(per_fraction, df["ebrt_fx"], ab)            # 外照射的部分
    for k in range(1, 5):
        total = total + eqd2(df[f"{name}_{k}"], 1, ab)       # 加上第 k 次近距离
    df[f"{name}_total"] = total

print(df[["d90_total", "rectum_total", "bladder_total"]].median().round(1))
```

</details>

</div>

<div class="step exercise">

**第 6 步 · 多少人达标**

审计的第一个问题。三条标准各有多少位患者达到?三条都达到的有几位?把三条都达到的人数存进 `n_all_ok`,把靶区达标(D90 ≥ 85 Gy)的比例存进 `share_d90`。

```python
# 在这里写你的代码。结果存进 n_all_ok 和 share_d90。
```

```pycheck
need("n_all_ok", "share_d90")
if close(share_d90, 181, 0.5):
    fail("share_d90 要的是比例(0 到 1 之间),181 是人数。对一列 True/False 用 mean()。")
if close(share_d90, 0.645, 0.002):
    fail("这是 D90 超过 90 Gy 的比例。审计标准是不低于 85 Gy。")
if not close(share_d90, 0.905, 0.002):
    fail(f"share_d90 现在是 {share_d90:.3f},应该是 0.905(181 / 200)。")
if n_all_ok == 181 or n_all_ok == 108 or n_all_ok == 158:
    fail("这是只满足其中一条标准的人数。三条都满足,要把三个条件用 & 连起来。")
if n_all_ok != 80:
    fail(f"n_all_ok 现在是 {n_all_ok},应该是 80。三个条件: d90_total >= 85,rectum_total < 65,bladder_total < 80。")
ok("对了: 靶区达标 181 位(90.5%),直肠 108 位(54%),膀胱 158 位(79%);三条都达到的只有 80 位(40%)。靶区的硬性要求九成患者做到了,直肠的计划目标只有一半多做到。这是审计第一个问题的答案;它有多准、和什么有关,后面几部分接着看。")
```

<details><summary>提示 1: 思路</summary>

每条标准是对一列的一次比较,得到一列 True 和 False: 人数是 `sum()`,比例是 `mean()`。三条都满足,用 `&` 把三个条件连起来,每个条件加括号。

</details>

<details><summary>提示 2: 用到的写法</summary>

`ok_d90 = df["d90_total"] >= 85`,另外两条类似;`(ok_d90 & ok_rectum & ok_bladder).sum()`。

</details>

<details><summary>参考答案</summary>

```python
ok_d90 = df["d90_total"] >= 85
ok_rectum = df["rectum_total"] < 65
ok_bladder = df["bladder_total"] < 80
for name, flag in [("HR-CTV D90 >= 85 Gy", ok_d90), ("直肠 D2cc < 65 Gy", ok_rectum), ("膀胱 D2cc < 80 Gy", ok_bladder)]:
    print(f"{name}: {flag.sum()} 位({flag.mean():.1%})")

share_d90 = ok_d90.mean()
n_all_ok = (ok_d90 & ok_rectum & ok_bladder).sum()
print(f"三条都达到: {n_all_ok} 位({n_all_ok / len(df):.1%})")
```

</details>

</div>

</div>

**这一部分做完了**: 一张有重复、有错写、有错值的表,变成了 200 位患者、带三列总剂量的干净的表。每一处改动都是一行代码,谁都能看到你改了什么。下一部分: 学完 [Python 课第 3 课](/python/03-table-one/)以后做。
