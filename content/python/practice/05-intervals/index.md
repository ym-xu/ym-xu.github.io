---
title: "练习 5 · 估计得有多准"
summary: "达标比例和中位数的 95% 区间;在大肿瘤里,两种施源器的靶区剂量差多少。"
weight: 50
toc: false
duration: "约 35 分钟"
prereq: "[Python 课第 5 课](/python/05-uncertainty/);[练习 4](../04-figures/)"
questions:
  - "'九成病人靶区达标'这个数有多准?"
  - "在大肿瘤里,加不加插植针,靶区剂量差多少?这个差值有多准?"
objectives:
  - "bootstrap 的 95% 区间: 比例、中位数"
  - "两组之差的区间(新的一步,给了骨架)"
  - "把结果写成一句话"
---

练习 2 算出"90.5% 的病人靶区达标",练习 4 的散点图显示"大肿瘤里,加针的病人剂量更高"。这两个结论都来自这 200 位病人;报告给科里之前,要说清楚它们有多准。

先运行这一块:

```python
import numpy as np
import pandas as pd
from cervix import load_cervix

df = load_cervix()                 # 200 位病人,带三列总剂量

def boot_ci(values, stat=np.median, n_boot=2000, seed=1):
    """bootstrap 的 95% 区间(百分位数法)。"""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boot = np.array([stat(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)])
    return np.percentile(boot, [2.5, 97.5])

print("准备好了:", df.shape)
```

<div class="steps" data-steps>

<div class="step exercise">

**第 1 步 · 达标比例的区间**

靶区达标(`d90_total` ≥ 85 Gy)的比例是多少?它的 95% 区间呢?比例存进 `p_d90`,区间存进 `ci_p`。

```python
# 在这里写你的代码。结果存进 p_d90 和 ci_p。
```

```pycheck
need("p_d90", "ci_p")
try:
    lo, hi = float(ci_p[0]), float(ci_p[1])
except Exception:
    fail("ci_p 应该是两个数: boot_ci 交回的区间。")
if not close(p_d90, 0.905, 0.001):
    fail(f"p_d90 现在是 {p_d90},应该是 0.905(181 / 200)。")
if close(lo, hi, 1e-9):
    fail("区间缩成了一个点: 对 0 和 1 算中位数,不是 0 就是 1。比例是均值,给 boot_ci 加上 stat=np.mean。")
if close(lo, 0.86, 0.002) and close(hi, 0.945, 0.002):
    ok("对了: 90.5% 的病人靶区达标(95% CI 86.0% 到 94.5%)。EMBRACE II 要求硬性的剂量下限在八九成的病人身上做到;按这个区间,科里是做到了的,即使取区间的下限也有 86%。")
fail(f"区间现在是 {lo:.3f} 到 {hi:.3f},应该是 0.860 到 0.945。把'达标与否'变成一串 0 和 1,交给 boot_ci,stat=np.mean。")
```

<details><summary>提示 1: 思路</summary>

第 5 课第 5 节的做法: 把每位病人记成 1(达标)或 0(没达标),比例就是这串数的均值;区间用 `boot_ci`,要算的量是均值。

</details>

<details><summary>提示 2: 用到的写法</summary>

`met = (df["d90_total"] >= 85).to_numpy().astype(int)`;`met.mean()`;`boot_ci(met, stat=np.mean)`。

</details>

<details><summary>参考答案</summary>

```python
met = (df["d90_total"] >= 85).to_numpy().astype(int)
p_d90 = met.mean()
ci_p = boot_ci(met, stat=np.mean)
print(f"{met.sum()} / {len(met)} = {p_d90:.1%},95% 区间 {ci_p.round(3)}")
```

</details>

</div>

<div class="step exercise">

**第 2 步 · 两种施源器各自的中位数**

两种施源器的病人,`d90_total` 的中位数各自的 95% 区间是多少?存进字典 `ci_by_tech`: 键是 `"IC"` 和 `"IC/IS"`,值是区间。

```python
# 在这里写你的代码。结果存进字典 ci_by_tech。
```

```pycheck
need("ci_by_tech")
try:
    a = [float(v) for v in ci_by_tech["IC"]]
    b = [float(v) for v in ci_by_tech["IC/IS"]]
except Exception:
    fail('ci_by_tech 应该是一个字典,有 "IC" 和 "IC/IS" 两个键,每个键下面是区间的两个数。')
if close(a[0], b[0], 0.001) and close(a[1], b[1], 0.001):
    fail("两组的区间一模一样: 每一组要先筛出这一组的病人,再交给 boot_ci。")
if close(a[0], 90.44, 0.02) and close(b[0], 91.83, 0.02):
    fail("两组的结果放反了。")
if close(a[0], 91.83, 0.02) and close(a[1], 93.79, 0.02) and close(b[0], 90.44, 0.02) and close(b[1], 93.04, 0.02):
    ok("对了: 单纯腔内 92.4 Gy(95% CI 91.8 到 93.8),腔内加插植 92.2 Gy(90.4 到 93.0)。两个区间几乎完全重叠。")
fail(f"现在是 IC {a},IC/IS {b},应该是 [91.83, 93.79] 和 [90.44, 93.04]。")
```

<details><summary>提示 1: 思路</summary>

对每一组: 筛出这一组的病人,取出 `d90_total`,交给 `boot_ci`(默认算中位数)。可以写两遍,也可以用 `groupby` 写成循环。

</details>

<details><summary>提示 2: 用到的写法</summary>

`ci_by_tech = {}`;`for tech, d in df.groupby("technique"):`,循环里 `ci_by_tech[tech] = boot_ci(d["d90_total"].to_numpy())`。

</details>

<details><summary>参考答案</summary>

```python
ci_by_tech = {}
for tech, d in df.groupby("technique"):
    ci_by_tech[tech] = boot_ci(d["d90_total"].to_numpy())
    print(f"{tech}: {len(d)} 位,中位数 {d['d90_total'].median():.1f} Gy,95% 区间 {ci_by_tech[tech].round(1)}")
```

</details>

</div>

<div class="step">

<div class="quiz">
<p><b>第 3 步 · 想一想</b>: 两种施源器的中位数是 92.4 和 92.2 Gy,区间几乎完全重叠。能不能在审计报告里写"加不加插植针,靶区剂量没有差别"?</p>
<button class="quiz-opt" data-why="区间重叠只说明,把所有病人混在一起时看不出差别。练习 3 的基线表和练习 4 的散点图已经显示,两组病人的肿瘤大小差得很远。">能: 两个区间重叠,就是没有差别</button>
<button class="quiz-opt" data-correct data-why="这是把'小肿瘤不加针'和'大肿瘤加针'放在一起比。要回答针有没有用,得在肿瘤大小相近的病人里比。下一步就这样做。">不能: 两组的肿瘤大小不同,要在肿瘤大小相近的病人里比</button>
<button class="quiz-opt" data-why="病人再多,混在一起比的问题也还在: 增加样本量缩窄的是区间,纠正不了两组病人本来就不一样。">不能: 病人太少,多收一些就能看出差别</button>
</div>

</div>

<div class="step exercise">

**第 4 步 · 大肿瘤里,两种施源器差多少**

只看 HR-CTV 体积在 40 cm³ 以上的病人。加针的(`a`)和不加针的(`b`)的 `d90_total` 中位数相差多少?这个差值的 95% 区间呢?

两组之差的 bootstrap 是新的一步,所以这一题给了骨架: 每一次重抽,**两组各自**有放回地重抽,再算两个中位数的差。把三处 `___` 填上。

```python
big = df[df["hrctv_vol"] >= 40]
a = big.loc[big["technique"] == "IC/IS", "d90_total"].to_numpy()     # 加了插植针
b = big.loc[big["technique"] == "IC", "d90_total"].to_numpy()        # 单纯腔内

rng = np.random.default_rng(1)
diffs = np.array([np.median(rng.choice(a, size=len(a), replace=True)) - ___ for _ in range(2000)])
diff = np.median(a) - np.median(b)
low, high = np.percentile(diffs, [___, ___])
print(f"{len(a)} 位加针,{len(b)} 位不加针;中位数相差 {diff:.1f} Gy,95% 区间 {low:.1f} 到 {high:.1f} Gy")
```

```pycheck
need("diff", "low", "high")
if not close(diff, 5.25, 0.01):
    fail(f"diff 现在是 {diff:.2f},应该是 5.25。")
if close(low, 3.75, 0.05) and close(high, 6.75, 0.05):
    fail("区间偏窄: 第一处要对 b 也做有放回的重抽,np.median(rng.choice(b, size=len(b), replace=True)),不能直接用 np.median(b)。两组都有抽样的波动。")
if close(low, 2.48, 0.02) and close(high, 8.05, 0.02):
    ok("对了: 体积 40 cm³ 以上的病人里,加针的 42 位比不加针的 18 位,靶区剂量的中位数高 5.3 Gy(95% CI 2.5 到 8.1)。区间离 0 很远。在这些病人里,加针的有 92.9% 达到 85 Gy,不加针的只有 55.6%(18 位里 10 位)。混在一起看不出来的差别,在肿瘤大小相近的病人里一目了然。")
fail(f"区间现在是 {low:.2f} 到 {high:.2f},应该是 2.48 到 8.05。第一处是 b 的重抽样本的中位数;后两处是 2.5 和 97.5。")
```

<details><summary>提示 1: 思路</summary>

`a` 的那一半已经写好了: `np.median(rng.choice(a, size=len(a), replace=True))`。`b` 的写法一模一样,把 `a` 换成 `b`。95% 区间取第 2.5 和第 97.5 百分位数。

</details>

<details><summary>参考答案</summary>

```python
big = df[df["hrctv_vol"] >= 40]
a = big.loc[big["technique"] == "IC/IS", "d90_total"].to_numpy()     # 加了插植针
b = big.loc[big["technique"] == "IC", "d90_total"].to_numpy()        # 单纯腔内

rng = np.random.default_rng(1)
diffs = np.array([np.median(rng.choice(a, size=len(a), replace=True)) - np.median(rng.choice(b, size=len(b), replace=True)) for _ in range(2000)])
diff = np.median(a) - np.median(b)
low, high = np.percentile(diffs, [2.5, 97.5])
print(f"{len(a)} 位加针,{len(b)} 位不加针;中位数相差 {diff:.1f} Gy,95% 区间 {low:.1f} 到 {high:.1f} Gy")
```

</details>

</div>

<div class="step exercise homework">

**第 5 步 · 写一句结果**

把第 4 步的结果写成审计报告(或论文结果部分)里的**一句英文**。先自己写,再点开参考写法对照。这一步不自动检查。

<textarea class="hw-write" data-save rows="5" aria-label="你写的句子" placeholder="在这里写。内容只保存在你自己的浏览器里。"></textarea>

<details><summary>参考写法和自查</summary>

> Among patients with an HR-CTV volume of 40 cm³ or more, the median HR-CTV D90 was 5.3 Gy higher with combined intracavitary/interstitial applicators (n = 42) than with intracavitary applicators alone (n = 18) (95% CI 2.5 to 8.1 Gy).

对照你写的,看这几样有没有:

- 说清了是哪些病人(体积 40 cm³ 以上),每组多少位。
- 差值带单位,后面跟着 95% 区间。
- 方向清楚: 谁比谁高。
- 没有写 "caused" 或 "improved": 这是回顾性的比较,只能说"更高",不能说"加针使剂量提高了多少"。

</details>

</div>

</div>

**这一部分做完了**: 达标比例有了区间;大肿瘤里两种施源器的差别也量出来了。还剩一个问题: 这些差别会不会是碰巧?下一部分: 学完 [Python 课第 6 课](/python/06-tests/)以后做。
