---
title: "练习 3 · 两种施源器的患者像不像"
summary: "分组汇总、分期的构成、长表、合并随访,做出一张按施源器分列的基线表。"
weight: 30
toc: false
duration: "约 40 分钟"
prereq: "[Python 课第 3 课](/python/03-table-one/);[练习 2](../02-clean/)"
questions:
  - "单纯腔内和腔内加插植的患者,年龄、分期、肿瘤体积一样吗?"
  - "四次近距离的靶区剂量有没有哪一次偏低?有多少患者有随访?"
objectives:
  - "分组汇总和交叉表"
  - "宽表变长表、合并两张表"
  - "用函数做基线表"
---

审计的第二个问题要比较两种施源器。比较之前先要知道: 用单纯腔内(IC)的患者和加了插植针(IC/IS)的患者,本来是不是同一类患者?这就是基线表要回答的事。

先运行这一块。`load_cervix()` 给出的,就是你在练习 2 清理好、算好总剂量的那张表。

```python
import pandas as pd
from cervix import load_cervix, make_cervix_followup

df = load_cervix()                 # 200 位患者,带 d90_total、rectum_total、bladder_total
fu = make_cervix_followup()        # 随访表: id、fu_months、rect_tox

def median_iqr(x, digits=1):
    """一列数 -> '中位数 (Q1–Q3)'。"""
    q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
    return f"{med:.{digits}f} ({q1:.{digits}f}–{q3:.{digits}f})"

def n_pct(flag):
    """一列 True/False -> '人数 (百分比)'。"""
    return f"{flag.sum()} ({flag.mean():.1%})"

print("准备好了:", df.shape, fu.shape)
```

<div class="steps" data-steps>

<div class="step exercise">

**第 1 步 · 两组的年龄和肿瘤体积**

按施源器(`technique`)分组,算年龄(`age`)和 HR-CTV 体积(`hrctv_vol`)的中位数,结果(一张两行两列的小表)存进 `by_tech`。

```python
# 在这里写你的代码。结果存进 by_tech。
```

```pycheck
need("by_tech")
try:
    v_ic, v_is = float(by_tech.loc["IC", "hrctv_vol"]), float(by_tech.loc["IC/IS", "hrctv_vol"])
    a_ic, a_is = float(by_tech.loc["IC", "age"]), float(by_tech.loc["IC/IS", "age"])
except Exception:
    fail('by_tech 应该是一张小表: 行是 IC 和 IC/IS,列是 age 和 hrctv_vol。groupby("technique") 之后取这两列,再 median()。')
if close(v_ic, 29.2, 0.3) or close(v_is, 47.7, 0.3):
    fail("这是均值。题目要的是中位数: median()。")
if close(v_ic, 28.2, 0.01) and close(v_is, 44.8, 0.01) and close(a_ic, 50, 0.01) and close(a_is, 47, 0.01):
    ok("对了: 加插植针的患者,HR-CTV 体积的中位数是 44.8 cm³,单纯腔内的只有 28.2 cm³;年龄差不多(47 和 50 岁)。两组患者的肿瘤大小差得很远: 这不奇怪,肿瘤大才需要加针。")
fail(f"现在体积是 {v_ic} 和 {v_is},应该是 28.2 和 44.8;年龄应该是 50 和 47。")
```

<details><summary>提示 1: 思路</summary>

第 3 课的句式: 按什么分组、看哪几列、算什么。看两列时,方括号里放一个列名的列表。

</details>

<details><summary>提示 2: 用到的写法</summary>

`df.groupby("technique")[["age", "hrctv_vol"]].median()`。

</details>

<details><summary>参考答案</summary>

```python
by_tech = df.groupby("technique")[["age", "hrctv_vol"]].median()
print(by_tech)
print(df.groupby("technique")["hrctv_vol"].agg(["size", "count"]))    # 每组的人数,和有体积记录的人数
```

</details>

</div>

<div class="step exercise">

**第 2 步 · 分期的构成**

做分期(`stage`)和施源器的交叉表,算成百分比,让每种施源器的各个分期加起来是 100%,存进 `pct`。再取出"加插植针的患者里 IIIB 期占百分之几",存进 `pct_iiib_is`。

```python
# 在这里写你的代码。结果存进 pct 和 pct_iiib_is。
```

```pycheck
need("pct", "pct_iiib_is")
if close(pct_iiib_is, 54.5, 0.1):
    fail('这是行百分比: IIIB 期的患者里有 54.5% 加了针。题目问的是加针的患者里 IIIB 期占多少,要按列算: normalize="columns"。')
if close(pct_iiib_is, 6.0, 0.1):
    fail('这是占全部 200 位患者的百分比。要按列算: normalize="columns"。')
if close(pct_iiib_is, 0.171, 0.002):
    fail("这是比例 0.171。题目要百分比: 乘以 100。")
if close(pct_iiib_is, 17.1, 0.1):
    ok("对了: 加插植针的 70 位里 IIIB 期占 17.1%,单纯腔内的 130 位里只占 7.7%;IB3 期正好反过来(5.7% 对 14.6%)。分期晚的患者更多地用了插植针。")
fail(f"现在是 {pct_iiib_is},应该是 17.1。")
```

<details><summary>提示 1: 思路</summary>

`pd.crosstab` 加上 `normalize`: 要每一列(每种施源器)加起来是 100%。取出一格用 `.loc[行名, 列名]`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`pd.crosstab(df["stage"], df["technique"], normalize="columns") * 100`;`pct.loc["IIIB", "IC/IS"]`。

</details>

<details><summary>参考答案</summary>

```python
pct = pd.crosstab(df["stage"], df["technique"], normalize="columns") * 100
print(pct.round(1))
pct_iiib_is = pct.loc["IIIB", "IC/IS"]
```

</details>

</div>

<div class="step">

<div class="quiz">
<p><b>第 3 步 · 想一想</b>: 前两步看到,加插植针的患者肿瘤更大、分期更晚。接下来要比较两种施源器的靶区剂量,这件事意味着什么?</p>
<button class="quiz-opt" data-why="两组患者不一样,正是要小心的地方。肿瘤越大,靶区剂量越难给够;加针的患者本来就是难给够的那一批。">没有影响: 直接比较两组的 D90 就行</button>
<button class="quiz-opt" data-correct data-why="肿瘤大小既影响用不用针,又影响剂量给不给得够。直接比两组,比的是'小肿瘤不加针'和'大肿瘤加针',分不清差别来自针还是来自肿瘤大小。这叫混杂(confounding);因为是按病情选的治疗,也叫适应证混杂。办法是在肿瘤大小相近的患者里比,练习 5 和练习 6 会这样做。">两组的肿瘤大小不同,直接比较会把'施源器的作用'和'肿瘤大小的作用'混在一起</button>
<button class="quiz-opt" data-why="患者不是随机分到两组的,但不等于不能比。可以在肿瘤大小相近的患者里比较,只是结论要比随机试验谨慎。">两组完全不能比较,这个问题没法回答</button>
</div>

</div>

<div class="step exercise">

**第 4 步 · 四次近距离,有没有哪一次偏低**

把四次的 HR-CTV D90(`d90_1` 到 `d90_4`)变成长表 `long`: 每位患者的每一次占一行,带着 `id` 和 `technique`,记第几次的那一列叫 `fraction`,剂量那一列叫 `d90`。再算每一次的中位数,存进 `d90_by_fx`。

```python
# 在这里写你的代码。结果存进 long 和 d90_by_fx。
```

```pycheck
need("long", "d90_by_fx")
if len(long) != 800:
    fail(f"long 应该有 800 行(200 位患者 × 4 次),现在是 {len(long)} 行。")
for col in ["id", "technique", "fraction", "d90"]:
    if col not in long.columns:
        fail(f'long 里应该有 {col} 这一列: melt 的 id_vars 是 ["id", "technique"],var_name="fraction",value_name="d90"。')
try:
    vals = [float(v) for v in d90_by_fx.sort_index()]
except Exception:
    fail('d90_by_fx 应该是每一次的中位数: long.groupby("fraction")["d90"].median()。')
if len(vals) == 4 and close(vals[0], 7.9, 0.01) and close(vals[1], 8.0, 0.01) and close(vals[2], 7.9, 0.01) and close(vals[3], 7.95, 0.01):
    ok("对了: 四次的中位数是 7.9、8.0、7.9、7.95 Gy,几乎一样,没有哪一次系统性地偏低。长表有 800 行;按第几次分组只要一句,这就是把宽表变长的好处。")
fail(f"四次的中位数应该是 7.9、8.0、7.9、7.95,现在是 {vals}。")
```

<details><summary>提示 1: 思路</summary>

第 3 课的 `melt`: 哪几列每一行都要带着(`id_vars`),哪几列要叠起来(`value_vars`),叠出来的两列叫什么名字。然后按 `fraction` 分组算中位数。

</details>

<details><summary>提示 2: 用到的写法</summary>

`df.melt(id_vars=["id", "technique"], value_vars=["d90_1", "d90_2", "d90_3", "d90_4"], var_name="fraction", value_name="d90")`。

</details>

<details><summary>参考答案</summary>

```python
long = df.melt(id_vars=["id", "technique"], value_vars=["d90_1", "d90_2", "d90_3", "d90_4"],
               var_name="fraction", value_name="d90")
print(long.shape)
d90_by_fx = long.groupby("fraction")["d90"].median()
print(d90_by_fx)
```

</details>

</div>

<div class="step exercise">

**第 5 步 · 把随访合进来**

随访表 `fu` 里是有随访的患者: 随访了几个月(`fu_months`),有没有出现 2 级及以上的晚期直肠副作用(`rect_tox`,1 是有)。把它按患者编号合进 `df`,合并后的表存进 `m`(200 位患者都要留着)。再算: 没有随访记录的有几位(`n_lost`),有随访的患者里出现副作用的比例(`tox_rate`)。

```python
# 在这里写你的代码。结果存进 m、n_lost 和 tox_rate。
```

```pycheck
need("m", "n_lost", "tox_rate")
if len(m) == 185:
    fail('m 只有 185 行: 没有随访的 15 位患者被丢掉了。要留住全部 200 位,用 how="left"。')
if len(m) != 200:
    fail(f"m 应该有 200 行,现在是 {len(m)} 行。")
if "rect_tox" not in m.columns:
    fail("m 里没有 rect_tox 这一列: 随访表还没有合进来。")
if n_lost != 15:
    fail(f"n_lost 现在是 {n_lost},应该是 15: 合并以后 rect_tox 是空的患者数。")
if close(tox_rate, 0.075, 0.0005):
    fail("这是 15 / 200。没有随访的患者不知道有没有副作用,不能算进分母: 用 m[\"rect_tox\"].mean(),它会自动跳过缺失。")
if close(tox_rate, 15, 0.1):
    fail("15 是出现副作用的人数。比例用 mean()。")
if close(tox_rate, 0.0811, 0.0005):
    ok("对了: 15 位患者没有随访;有随访的 185 位里,15 位出现了 2 级及以上的晚期直肠副作用,占 8.1%。分母是 185,不是 200。")
fail(f"tox_rate 现在是 {tox_rate},应该是 0.081(15 / 185)。")
```

<details><summary>提示 1: 思路</summary>

`merge`: 左边是 `df`,右边是 `fu`,按 `id` 对上,左边的行全部保留。没有随访的患者,合并后 `rect_tox` 是空的,数空格子用 `isna().sum()`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`df.merge(fu, on="id", how="left")`;`m["rect_tox"].isna().sum()`;`m["rect_tox"].mean()`。

</details>

<details><summary>参考答案</summary>

```python
m = df.merge(fu, on="id", how="left")
print(m.shape)
n_lost = m["rect_tox"].isna().sum()
tox_rate = m["rect_tox"].mean()
print(f"{n_lost} 位没有随访;有随访的 {m['rect_tox'].count()} 位里,{tox_rate:.1%} 有晚期直肠副作用")
```

</details>

</div>

<div class="step exercise">

**第 6 步 · 一张基线表**

把前面看到的写成一张按施源器分列的基线表。行名已经给出,把四处 `___` 换成算这一格的代码(用准备代码里的 `median_iqr` 和 `n_pct`;年龄不留小数)。

```python
def column(d):
    return pd.Series({
        "Patients, n": len(d),
        "Age, years": ___,
        "FIGO stage IIIB or IVA, n (%)": ___,
        "HR-CTV volume, cm3": ___,
        "HR-CTV D90, Gy (EQD2)": ___,
    })

table1 = pd.DataFrame({
    "IC": column(df[df["technique"] == "IC"]),
    "IC/IS": column(df[df["technique"] == "IC/IS"]),
})
table1
```

```pycheck
need("table1")
def want(d):
    return {
        "Patients, n": len(d),
        "Age, years": median_iqr(d["age"], 0),
        "FIGO stage IIIB or IVA, n (%)": n_pct(d["stage"].isin(["IIIB", "IVA"])),
        "HR-CTV volume, cm3": median_iqr(d["hrctv_vol"]),
        "HR-CTV D90, Gy (EQD2)": median_iqr(d["d90_total"]),
    }
for tech in ["IC", "IC/IS"]:
    w = want(df[df["technique"] == tech])
    for row, value in w.items():
        try:
            got = table1.loc[row, tech]
        except Exception:
            fail(f"table1 里找不到 {row} 这一行或 {tech} 这一列。")
        if str(got) != str(value):
            fail(f'"{row}" 这一行,{tech} 那一格应该是 {value},现在是 {got}。')
ok("对了。这张表把两组的差别摆在了一起: 加针的患者分期更晚(IIIB 或 IVA 期占 21.4%,另一组 9.2%)、肿瘤更大(44.8 对 28.2 cm³),而靶区剂量的中位数几乎一样(92.2 对 92.4 Gy)。'剂量一样'是不是说明加针没有用?练习 5 和练习 6 回答。")
```

<details><summary>提示 1: 思路</summary>

连续的量用 `median_iqr(一列, 小数位数)`;"是不是 IIIB 或 IVA 期"是一列 True/False,交给 `n_pct`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`median_iqr(d["age"], 0)`;`n_pct(d["stage"].isin(["IIIB", "IVA"]))`: `isin` 判断每一行的值在不在给出的列表里。

</details>

<details><summary>参考答案</summary>

```python
def column(d):
    return pd.Series({
        "Patients, n": len(d),
        "Age, years": median_iqr(d["age"], 0),
        "FIGO stage IIIB or IVA, n (%)": n_pct(d["stage"].isin(["IIIB", "IVA"])),
        "HR-CTV volume, cm3": median_iqr(d["hrctv_vol"]),
        "HR-CTV D90, Gy (EQD2)": median_iqr(d["d90_total"]),
    })

table1 = pd.DataFrame({
    "IC": column(df[df["technique"] == "IC"]),
    "IC/IS": column(df[df["technique"] == "IC/IS"]),
})
table1
```

</details>

</div>

</div>

**这一部分做完了**: 两种施源器的患者不是同一类患者。记住这张基线表,后面比较两组时用得上。下一部分: 学完 [Python 课第 4 课](/python/04-plots/)以后做。
