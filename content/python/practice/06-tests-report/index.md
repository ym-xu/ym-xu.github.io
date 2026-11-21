---
title: "练习 6 · 是不是碰巧,和一份审计报告"
summary: "检验、混杂、直肠剂量和晚期副作用,最后写出审计的结论。"
weight: 60
toc: false
duration: "约 45 分钟"
prereq: "[Python 课第 6 课](/python/06-tests/);[练习 5](../05-intervals/)"
questions:
  - "两种施源器的差别是不是碰巧?为什么混在一起比看不出来?"
  - "直肠剂量高的患者,晚期直肠副作用是不是更多?"
  - "这次审计的结论是什么?"
objectives:
  - "Mann–Whitney 检验、Spearman 相关、Fisher 精确检验"
  - "识别混杂"
  - "把分析写成结论"
---

最后一部分回答审计剩下的两个问题,然后把六部分的结果写成结论。

先运行这一块:

```python
import numpy as np
import pandas as pd
from scipy import stats
from cervix import load_cervix, make_cervix_followup

df = load_cervix()                                              # 200 位患者,带三列总剂量
m = df.merge(make_cervix_followup(), on="id", how="left")       # 合并了随访(练习 3 第 5 步)
print("准备好了:", df.shape, m.shape)
```

<div class="steps" data-steps>

<div class="step exercise">

**第 1 步 · 两种施源器: 混在一起比,和只在大肿瘤里比**

两种施源器是两组不同的患者。用 Mann–Whitney 检验比较它们的 `d90_total`,做两次: 全部患者(P 值存进 `p_all`),和只看 HR-CTV 体积 40 cm³ 以上的患者(存进 `p_big`)。

```python
# 在这里写你的代码。结果存进 p_all 和 p_big。
```

```pycheck
need("p_all", "p_big")
if not close(p_all, 0.5705, 0.002):
    fail(f"p_all 现在是 {p_all:.4f},应该是 0.570: stats.mannwhitneyu(IC 的 d90_total, IC/IS 的 d90_total).pvalue,用全部患者。")
if close(p_big, 0.438, 0.003):
    fail("这是体积小于 40 cm³ 的患者的结果。要的是 40 cm³ 以上的: hrctv_vol >= 40。")
if not (5e-5 < p_big < 1e-4):
    fail(f"p_big 现在是 {p_big:.2g},应该在 7e-05 左右。先筛出 hrctv_vol >= 40 的患者,再在里面比较两种施源器。")
ok("对了: 全部患者混在一起,P = 0.57,看不出差别;只看大肿瘤,P = 0.00007,差别非常清楚。同一份数据、同一种检验,问的患者不同,答案完全不同。")
```

<details><summary>提示 1: 思路</summary>

`stats.mannwhitneyu(第一组的数, 第二组的数).pvalue`。先对全部患者做;再筛出大肿瘤的患者,存成一张新表,在这张表上重做一遍。

</details>

<details><summary>提示 2: 用到的写法</summary>

`ic = df.loc[df["technique"] == "IC", "d90_total"]`;`big = df[df["hrctv_vol"] >= 40]`,然后在 `big` 上同样筛两组。

</details>

<details><summary>参考答案</summary>

```python
def compare(d):
    ic = d.loc[d["technique"] == "IC", "d90_total"]
    needles = d.loc[d["technique"] == "IC/IS", "d90_total"]
    return stats.mannwhitneyu(ic, needles).pvalue

p_all = compare(df)
p_big = compare(df[df["hrctv_vol"] >= 40])
print(f"全部患者: P = {p_all:.2f};体积 40 cm3 以上: P = {p_big:.1g}")
```

</details>

</div>

<div class="step">

<div class="quiz">
<p><b>第 2 步 · 想一想</b>: 混在一起比 P = 0.57,只看大肿瘤 P = 0.00007。如果只做了第一个检验,报告里写"两种施源器的靶区剂量没有差别(P = 0.57)",错在哪里?</p>
<button class="quiz-opt" data-why="换一种检验解决不了这个问题。错不在检验方法,而在比的是两批本来就不一样的患者。">应该用 t 检验,而不是 Mann–Whitney</button>
<button class="quiz-opt" data-correct data-why="肿瘤大小同时影响'用不用针'和'剂量给不给得够',这就是混杂。因为治疗是按病情选的,这种情况也叫适应证混杂(confounding by indication),回顾性研究里比较两种治疗时最常见。P = 0.57 只说明没有检出差别,而且这个比较本身就不公平。">加针的患者肿瘤更大,本来更难给够剂量。混在一起比,针的好处被肿瘤大小的坏处抵消了</button>
<button class="quiz-opt" data-why="样本量是够的: 同样这 200 位患者,分层以后差别非常清楚。问题在于怎么比,不在患者多少。">200 位患者太少,所以没检出来</button>
</div>

</div>

<div class="step exercise">

**第 3 步 · 肿瘤越大,剂量越低吗**

只看单纯腔内(IC)的患者: HR-CTV 体积和 `d90_total` 的 Spearman 相关系数是多少?存进 `rho_ic`。记得体积有缺失值。

```python
# 在这里写你的代码。结果存进 rho_ic。
```

```pycheck
need("rho_ic")
if rho_ic != rho_ic:
    fail('结果是 NaN: 体积有 5 个缺失值。先 dropna(subset=["hrctv_vol"])。')
if close(rho_ic, -0.404, 0.003):
    fail("这是全部患者的相关系数。题目要的是只看单纯腔内的患者: 先筛出 technique 是 IC 的。")
if close(rho_ic, -0.134, 0.003):
    fail("这是加了插植针的患者的结果。题目要的是单纯腔内: technique 是 IC。")
if close(rho_ic, -0.623, 0.003):
    fail("这是 Pearson 相关系数。题目要的是 Spearman: stats.spearmanr。")
if close(rho_ic, -0.630, 0.003):
    ok("对了: ρ = −0.63(126 位有体积记录的患者)。单纯腔内时,肿瘤越大,靶区剂量越低,关系很强。加了插植针的患者里,这个相关只有 −0.13(P = 0.27): 有了针,肿瘤大也不再意味着剂量低。这和散点图上看到的一样。")
fail(f"rho_ic 现在是 {rho_ic:.3f},应该是 −0.63。")
```

<details><summary>提示 1: 思路</summary>

两步: 筛出单纯腔内的患者,并去掉体积缺失的;然后 `stats.spearmanr(体积, 剂量)`,取它的 `.statistic`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`d = df[df["technique"] == "IC"].dropna(subset=["hrctv_vol"])`;`stats.spearmanr(d["hrctv_vol"], d["d90_total"]).statistic`。

</details>

<details><summary>参考答案</summary>

```python
for tech in ["IC", "IC/IS"]:
    d = df[df["technique"] == tech].dropna(subset=["hrctv_vol"])
    res = stats.spearmanr(d["hrctv_vol"], d["d90_total"])
    print(f"{tech}: ρ = {res.statistic:.2f},P = {res.pvalue:.2g},n = {len(d)}")
    if tech == "IC":
        rho_ic = res.statistic
```

</details>

</div>

<div class="step exercise">

**第 4 步 · 直肠剂量和晚期直肠副作用**

审计的第三个问题。在合并了随访的表 `m` 里,把患者按直肠 D2cc 是否达到 65 Gy(`rectum_total >= 65`)分成两组,和有没有晚期直肠副作用(`rect_tox`)做四格表,存进 `table_tox`;用 Fisher 精确检验,P 值存进 `p_tox`。

```python
# 在这里写你的代码。结果存进 table_tox 和 p_tox。
```

```pycheck
need("table_tox", "p_tox")
if getattr(table_tox, "shape", None) != (2, 2):
    fail("table_tox 应该是 2 行 2 列的四格表: pd.crosstab(直肠是否达到 65 Gy, m[\"rect_tox\"])。")
if int(table_tox.values.sum()) != 185:
    fail(f"四格表里应该是有随访的 185 位患者,现在是 {int(table_tox.values.sum())} 位。")
if sorted(table_tox.values.ravel().tolist()) != [3, 12, 72, 98]:
    fail(f"四格表的四个数应该是 98、3、72、12,现在是 {table_tox.values.tolist()}。分组的条件是 rectum_total >= 65。")
if close(p_tox, 0.0112, 0.0005):
    fail("这是卡方检验的 P 值。题目要的是 Fisher 精确检验: stats.fisher_exact(table_tox).pvalue。")
if close(p_tox, 0.0062, 0.0003):
    ok("对了: 直肠 D2cc 低于 65 Gy 的 101 位里,3 位出现晚期直肠副作用(3.0%);达到 65 Gy 的 84 位里有 12 位(14.3%)。Fisher 精确检验 P = 0.006。事件一共只有 15 个,所以用 Fisher;卡方检验是 0.011,结论相同。")
fail(f"p_tox 现在是 {p_tox:.4f},应该是 0.006。")
```

<details><summary>提示 1: 思路</summary>

先在 `m` 里加一列 True/False 表示直肠剂量是否达到 65 Gy,再和 `rect_tox` 做 `pd.crosstab`(没有随访的患者会自动跳过),最后交给 `stats.fisher_exact`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`m["rect_high"] = m["rectum_total"] >= 65`;`pd.crosstab(m["rect_high"], m["rect_tox"])`;`stats.fisher_exact(table_tox).pvalue`。

</details>

<details><summary>参考答案</summary>

```python
m["rect_high"] = m["rectum_total"] >= 65
table_tox = pd.crosstab(m["rect_high"], m["rect_tox"])
print(table_tox)
print(m.groupby("rect_high")["rect_tox"].mean().round(3))
p_tox = stats.fisher_exact(table_tox).pvalue
print(f"Fisher 精确检验: P = {p_tox:.3f}")
```

</details>

</div>

<div class="step">

<div class="quiz">
<p><b>第 5 步 · 想一想</b>: 直肠 D2cc 达到 65 Gy 的患者,14.3% 出现晚期直肠副作用;低于 65 Gy 的是 3.0%;P = 0.006。审计报告里哪种写法最恰当?</p>
<button class="quiz-opt" data-why="这是回顾性的数据,能说明两者有关联,不能单凭它证明因果;'必须'这样的硬性结论,也超出了 15 个事件能支持的范围。">证明了直肠剂量超过 65 Gy 会导致副作用,今后所有患者都必须低于 65 Gy</button>
<button class="quiz-opt" data-correct data-why="差别不像是碰巧,方向也和已有的研究一致(EMBRACE 的数据同样显示直肠 D2cc 越高,出血和直肠炎越多),所以支持把 65 Gy 作为计划目标。同时要写明局限: 只有 15 个事件,两个比例都估计得很粗;剂量高的患者还可能在别的方面不同。">直肠剂量高的患者副作用更多,差别不像是碰巧,和已有的研究一致,支持把 65 Gy 作为计划目标;但事件只有 15 个,比例估计得还很粗</button>
<button class="quiz-opt" data-why="P 值不是出现副作用的概率。它说的是: 如果直肠剂量和副作用没有关系,这么大的差别有多容易碰上。剂量高的患者里出现副作用的是 14.3%。">P = 0.006 说明直肠剂量高的患者有 99.4% 的可能出现副作用</button>
</div>

</div>

<div class="step exercise">

**第 6 步 · 审计报告的最后一个数**

报告里要点出最需要改进的地方: 肿瘤大、却没有加针的患者。下面的代码会打印出整份审计摘要,只差一个数: 体积 40 cm³ 以上、只用了腔内施源器的患者里,靶区没有达到 85 Gy 的有几位。把 `___` 换成算这个数的代码。

```python
big_ic = df[(df["technique"] == "IC") & (df["hrctv_vol"] >= 40)]
n_big_ic_fail = ___

ok_d90, ok_rectum, ok_bladder = df["d90_total"] >= 85, df["rectum_total"] < 65, df["bladder_total"] < 80
tox = m.dropna(subset=["rect_tox"])
high = tox["rectum_total"] >= 65
print("宫颈癌近距离放疗剂量审计(模拟数据)")
print(f"1. 达标: 靶区 {ok_d90.mean():.1%},直肠 {ok_rectum.mean():.1%},膀胱 {ok_bladder.mean():.1%};三条都达到 {(ok_d90 & ok_rectum & ok_bladder).mean():.1%}。")
print(f"2. 大肿瘤(40 cm3 以上)只用腔内施源器的 {len(big_ic)} 位里,{n_big_ic_fail} 位靶区没有达到 85 Gy。")
print(f"3. 晚期直肠副作用: 直肠 D2cc 达到 65 Gy 的 {tox.loc[high, 'rect_tox'].mean():.1%},低于 65 Gy 的 {tox.loc[~high, 'rect_tox'].mean():.1%}。")
```

```pycheck
need("n_big_ic_fail")
if n_big_ic_fail == 10:
    fail("10 是达到了 85 Gy 的人数。要的是没有达到的: d90_total < 85。")
if n_big_ic_fail == 18:
    fail("18 是这些患者的总数。要的是其中 d90_total < 85 的人数。")
if n_big_ic_fail != 8:
    fail(f"现在是 {n_big_ic_fail},应该是 8: (big_ic[\"d90_total\"] < 85).sum()。")
ok("对了: 18 位里有 8 位没有达到 85 Gy。审计摘要完成了: 从一位患者的 EQD2,到 200 位患者的清理、基线表、图、区间和检验,最后落到一条可以执行的建议上。整套主题练习到这里做完了。")
```

<details><summary>提示 1: 思路</summary>

`big_ic` 已经筛好了。在它里面,`d90_total` 小于 85 的是一列 True 和 False,数 True 的个数。

</details>

<details><summary>参考答案</summary>

```python
big_ic = df[(df["technique"] == "IC") & (df["hrctv_vol"] >= 40)]
n_big_ic_fail = (big_ic["d90_total"] < 85).sum()

ok_d90, ok_rectum, ok_bladder = df["d90_total"] >= 85, df["rectum_total"] < 65, df["bladder_total"] < 80
tox = m.dropna(subset=["rect_tox"])
high = tox["rectum_total"] >= 65
print("宫颈癌近距离放疗剂量审计(模拟数据)")
print(f"1. 达标: 靶区 {ok_d90.mean():.1%},直肠 {ok_rectum.mean():.1%},膀胱 {ok_bladder.mean():.1%};三条都达到 {(ok_d90 & ok_rectum & ok_bladder).mean():.1%}。")
print(f"2. 大肿瘤(40 cm3 以上)只用腔内施源器的 {len(big_ic)} 位里,{n_big_ic_fail} 位靶区没有达到 85 Gy。")
print(f"3. 晚期直肠副作用: 直肠 D2cc 达到 65 Gy 的 {tox.loc[high, 'rect_tox'].mean():.1%},低于 65 Gy 的 {tox.loc[~high, 'rect_tox'].mean():.1%}。")
```

</details>

</div>

<div class="step exercise homework">

**第 7 步 · 写出结论和建议**

你是这次审计的负责人。用自己的话写出三条结论和一条建议(中文或英文都可以),要有数字,也要有局限。先写,再点开参考写法对照。这一步不自动检查。

<textarea class="hw-write" data-save rows="8" aria-label="你写的结论和建议" placeholder="在这里写。内容只保存在你自己的浏览器里。"></textarea>

<details><summary>参考写法和自查</summary>

**结论**

1. 200 位患者里,HR-CTV D90 达到 85 Gy 的占 90.5%(95% CI 86.0% 到 94.5%);直肠 D2cc 低于 65 Gy 的占 54%,膀胱 D2cc 低于 80 Gy 的占 79%;三条都达到的占 40%。
2. 两种施源器的靶区剂量粗看没有差别(中位数 92.4 对 92.2 Gy,*P* = 0.57),但加针的患者肿瘤更大。在 HR-CTV 体积 40 cm³ 以上的患者里,加针的靶区剂量中位数高 5.3 Gy(95% CI 2.5 到 8.1;*P* < 0.001);这些患者里只用腔内施源器的 18 位,有 8 位没有达到 85 Gy。
3. 直肠 D2cc 达到 65 Gy 的患者,晚期直肠副作用更多(14.3% 对 3.0%,*P* = 0.006)。

**建议**: HR-CTV 体积 40 cm³ 以上的患者,计划时优先考虑腔内加插植针;同时把直肠 D2cc 低于 65 Gy 作为每个计划都要核对的目标。

**局限**: 回顾性的比较,施源器不是随机分配的;15 位患者没有随访;副作用只有 15 例,比例估计得很粗;40 cm³ 这个分界是看了数据以后选的,需要在以后的患者里验证。

对照你写的:

- 每条结论都有数字,比例带分母或区间。
- 第 2 条说清了"粗看没有差别"和"分层以后有差别",没有只报其中一个。
- 建议是能执行的,而且来自上面的结论。
- 写了局限。最容易漏的是最后一条: 分界点是事后选的。

</details>

</div>

</div>

**整套主题练习做完了**。回头看这六部分: 你做的事情,和 Python 课里对子宫内膜癌数据做的是同一套: 查数据、清理、描述、画图、给出区间、做检验、写结论。换了一个病种、一份新的数据,这套办法照样用得上。下一份数据,可以是你自己的。
