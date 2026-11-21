---
title: "练习 1 · 一位患者的总剂量"
summary: "写 EQD2 函数,把外照射和 4 次近距离加起来,判断一位患者达不达标。"
weight: 10
toc: false
duration: "约 30 分钟"
prereq: "[Python 课第 1 课](/python/01-first-steps/)"
questions:
  - "一位患者做了外照射和 4 次近距离,靶区和直肠的总剂量各是多少?"
  - "这位患者达到审计标准了吗?"
objectives:
  - "变量、列表、循环和条件"
  - "自己写函数"
---

审计从一位患者开始。这位患者做了盆腔外照射 45 Gy(分 25 次),然后做了 4 次近距离放疗。每次近距离的计划都不一样,所以 HR-CTV D90 和直肠 D2cc 每次都不同。

审计标准: HR-CTV D90 的总 EQD2 不低于 85 Gy,直肠 D2cc 的总 EQD2 低于 65 Gy(背景见[练习目录](../))。

先运行这一块,把这位患者的数据准备好:

```python
ebrt_gy, ebrt_fx = 45, 25            # 外照射: 总剂量 45 Gy,分 25 次
d90 = [7.6, 8.1, 7.4, 7.9]           # 4 次近距离的 HR-CTV D90(Gy)
rectum = [3.9, 4.4, 3.6, 4.1]        # 4 次近距离的直肠 D2cc(Gy)

patients = [                         # 第 5 步要用的五位患者
    {"id": "A", "d90": [7.6, 8.1, 7.4, 7.9], "rectum": [3.9, 4.4, 3.6, 4.1]},
    {"id": "B", "d90": [8.2, 8.0, 8.4, 8.1], "rectum": [3.2, 3.5, 3.0, 3.4]},
    {"id": "C", "d90": [6.4, 6.8, 6.5, 6.9], "rectum": [3.6, 3.3, 3.8, 3.5]},
    {"id": "D", "d90": [7.9, 8.3, 8.0, 7.7], "rectum": [4.6, 4.9, 4.4, 4.7]},
    {"id": "E", "d90": [7.5, 7.8, 7.6, 8.0], "rectum": [3.4, 3.1, 3.6, 3.3]},
]
print("准备好了")
```

<div class="steps" data-steps>

<div class="step exercise">

**第 1 步 · 写 EQD2 函数,算外照射的部分**

写一个函数 `eqd2(d, n, ab)`: 每次剂量 `d`、照射 `n` 次、α/β 是 `ab`。公式: EQD2 = n × d × (d + α/β) ÷ (2 + α/β)。

然后用它算外照射部分的 EQD2: 靶区(α/β = 10)存进 `ebrt_10`,直肠(α/β = 3)存进 `ebrt_3`。

```python
# 在这里写你的代码: 先定义 eqd2,再算 ebrt_10 和 ebrt_3。
```

```pycheck
need("eqd2", "ebrt_10", "ebrt_3")
if not callable(eqd2):
    fail("eqd2 应该是一个函数: def eqd2(d, n, ab): ...")
try:
    probe = eqd2(2, 25, 3)
except TypeError:
    fail("eqd2 要有三个参数,顺序是 d、n、ab。")
if probe is None:
    fail("函数没有交回结果: 别忘了 return。")
if not close(probe, 50, 0.01):
    fail(f"eqd2(2, 25, 3) 应该是 50(每次 2 Gy 时,EQD2 就等于总剂量),现在是 {probe}。检查公式: n * d * (d + ab) / (2 + ab)。")
if close(ebrt_10, 206.25, 0.1):
    fail("每次剂量不是 45 Gy: 45 Gy 分 25 次,每次是 45 / 25 = 1.8 Gy。")
if close(ebrt_10, 43.2, 0.01) and close(ebrt_3, 44.25, 0.01):
    fail("两个数放反了: 靶区用 α/β = 10,直肠用 α/β = 3。")
if close(ebrt_10, 44.25, 0.01) and close(ebrt_3, 43.2, 0.01):
    ok("对了: 外照射 45 Gy 分 25 次,对靶区相当于 44.25 Gy,对直肠相当于 43.2 Gy。每次只有 1.8 Gy,比 2 Gy 小,所以 EQD2 比 45 Gy 略低;α/β 越小,低得越多。")
fail(f"现在 ebrt_10 = {ebrt_10},ebrt_3 = {ebrt_3},应该是 44.25 和 43.2。每次剂量是 ebrt_gy / ebrt_fx。")
```

<details><summary>提示 1: 思路</summary>

函数的写法和第 1 课第 7 节一样,只是 α/β 没有默认值。外照射每次的剂量要自己算: 总剂量除以次数。

</details>

<details><summary>提示 2: 用到的写法</summary>

`def eqd2(d, n, ab):` 下面一行 `return n * d * (d + ab) / (2 + ab)`。然后 `ebrt_10 = eqd2(ebrt_gy / ebrt_fx, ebrt_fx, 10)`。

</details>

<details><summary>参考答案</summary>

```python
def eqd2(d, n, ab):
    """每次剂量 d(Gy)、照射 n 次、α/β = ab 时的 EQD2(Gy)。"""
    return n * d * (d + ab) / (2 + ab)

ebrt_10 = eqd2(ebrt_gy / ebrt_fx, ebrt_fx, 10)
ebrt_3 = eqd2(ebrt_gy / ebrt_fx, ebrt_fx, 3)
print(ebrt_10, ebrt_3)
```

</details>

</div>

<div class="step">

<div class="quiz">
<p><b>第 2 步 · 先想清楚</b>: 4 次近距离的 D90 是 7.6、8.1、7.4、7.9 Gy,加起来 31 Gy。算近距离部分的 EQD2 时,能不能把 31 Gy 当成"一次照了 31 Gy"代进公式?</p>
<button class="quiz-opt" data-why="EQD2 的公式里,每次剂量 d 出现了两次(d 和 d + α/β),所以结果取决于每次照多少。把 31 Gy 当成一次,算出来是 106 Gy;分成 4 次各自算再相加,只有 46 Gy。一次照 31 Gy 的损伤,远大于分 4 次照。">可以: 总剂量一样,结果就一样</button>
<button class="quiz-opt" data-correct data-why="生物效应取决于每次照多少。每一次近距离要按它自己的剂量单独换算成 EQD2,然后把 4 个 EQD2 相加。">不可以: 每一次按自己的剂量单独换算,再把 4 个 EQD2 加起来</button>
<button class="quiz-opt" data-why="物理剂量直接相加(45 + 31 = 76 Gy)没有考虑每次剂量的大小。近距离每次七八 Gy,效应比同样总量的常规分割大得多,所以才要换算成 EQD2。">不用换算: 把外照射的 45 Gy 和近距离的 31 Gy 直接相加</button>
</div>

</div>

<div class="step exercise">

**第 3 步 · 把 4 次近距离加上去**

用循环,把每一次近距离的 EQD2 加到外照射的部分上: 靶区的总剂量存进 `d90_total`,直肠的存进 `rectum_total`。

```python
# 在这里写你的代码。要用到第 1 步的 eqd2、ebrt_10、ebrt_3,和上面的 d90、rectum。
```

```pycheck
need("d90_total", "rectum_total")
if close(d90_total, 150.17, 0.05):
    fail("这是把 31 Gy 当成一次照射算出来的。每一次要单独换算: 循环里用 eqd2(d, 1, 10)。")
if close(d90_total, 76.0, 0.05):
    fail("这是物理剂量直接相加。每一次都要先换算成 EQD2。")
if close(d90_total, 45.88, 0.05):
    fail("近距离的部分算对了,但没有加上外照射的 ebrt_10。")
if not close(d90_total, 90.13, 0.02):
    fail(f"d90_total 现在是 {d90_total:.2f},应该是 90.13。从 ebrt_10 开始,每一次加上 eqd2(d, 1, 10)。")
if close(rectum_total, 62.0, 0.2) or close(rectum_total, 66.72, 0.05):
    fail("直肠要用 α/β = 3: 从 ebrt_3 开始,每一次加上 eqd2(d, 1, 3)。")
if not close(rectum_total, 65.67, 0.02):
    fail(f"rectum_total 现在是 {rectum_total:.2f},应该是 65.67。从 ebrt_3 开始,每一次加上 eqd2(d, 1, 3)。")
ok("对了: 靶区 D90 的总 EQD2 是 90.1 Gy,直肠 D2cc 是 65.7 Gy。直肠每次只受到 4 Gy 左右,4 次的 EQD2 加起来却有 22 Gy: α/β 小的组织,对每次剂量的大小更敏感。")
```

<details><summary>提示 1: 思路</summary>

先让总剂量等于外照射的部分,然后对列表里的每一次剂量,换算成 EQD2 再加上去。"一次"就是 `n = 1`。靶区和直肠各做一遍。

</details>

<details><summary>提示 2: 用到的写法</summary>

`d90_total = ebrt_10`,然后 `for d in d90:`,循环里 `d90_total = d90_total + eqd2(d, 1, 10)`。

</details>

<details><summary>参考答案</summary>

```python
d90_total = ebrt_10
for d in d90:
    d90_total = d90_total + eqd2(d, 1, 10)

rectum_total = ebrt_3
for d in rectum:
    rectum_total = rectum_total + eqd2(d, 1, 3)

print(f"HR-CTV D90: {d90_total:.1f} Gy,直肠 D2cc: {rectum_total:.1f} Gy")
```

</details>

</div>

<div class="step exercise">

**第 4 步 · 写成一个函数,判断达不达标**

审计有 200 位患者,不能每位都这样写一遍。把上面的计算写成函数 `total_eqd2(ebrt_gy, ebrt_fx, bt_doses, ab)`: 给出外照射的总剂量和次数、近距离各次的剂量(一个列表)和 α/β,交回总 EQD2。

再用它判断这位患者是否达标(靶区 ≥ 85 Gy,并且直肠 < 65 Gy),结果(True 或 False)存进 `meets`。

```python
# 在这里写你的代码: 定义 total_eqd2,再算 meets。
```

```pycheck
need("total_eqd2", "meets")
if not callable(total_eqd2):
    fail("total_eqd2 应该是一个函数。")
try:
    a = total_eqd2(45, 25, [7, 7, 7, 7], 10)
    b = total_eqd2(50.4, 28, [7, 7, 7, 7], 3)
except TypeError:
    fail("total_eqd2 要有四个参数,顺序是 ebrt_gy、ebrt_fx、bt_doses、ab。")
if a is None:
    fail("函数没有交回结果: 别忘了 return。")
if not close(a, 83.92, 0.02):
    fail(f"用 45 Gy / 25 次加 7 Gy × 4 次、α/β = 10 来试,应该得到 83.9 Gy,你的函数给出 {a:.2f}。")
if not close(b, 104.38, 0.02):
    fail(f"换成外照射 50.4 Gy / 28 次、α/β = 3,应该得到 104.4 Gy,你的函数给出 {b:.2f}。函数里要用参数,不要直接写 45、25 或上面的变量名。")
if meets is True or meets == True:
    fail("这位患者靶区是 90.1 Gy(达标),直肠是 65.7 Gy: 没有低于 65 Gy。两条都满足才算达标,用 and。")
ok("对了: 这位患者靶区达标(90.1 Gy),直肠超了一点(65.7 Gy),所以 meets 是 False。顺便看到两个熟悉的数: 45 Gy 外照射加 7 Gy × 4 次,靶区是 83.9 Gy;函数换一组数照样能算。")
```

<details><summary>提示 1: 思路</summary>

函数里做的就是第 1 步和第 3 步的事: 先算外照射的 EQD2,再用循环把每一次近距离加上去,最后 `return`。判断用比较和 `and`。

</details>

<details><summary>提示 2: 用到的写法</summary>

函数里: `total = eqd2(ebrt_gy / ebrt_fx, ebrt_fx, ab)`,`for d in bt_doses:`,`total = total + eqd2(d, 1, ab)`,`return total`。然后 `meets = total_eqd2(45, 25, d90, 10) >= 85 and total_eqd2(45, 25, rectum, 3) < 65`。

</details>

<details><summary>参考答案</summary>

```python
def total_eqd2(ebrt_gy, ebrt_fx, bt_doses, ab):
    """外照射加上各次近距离的总 EQD2(Gy)。"""
    total = eqd2(ebrt_gy / ebrt_fx, ebrt_fx, ab)
    for d in bt_doses:
        total = total + eqd2(d, 1, ab)
    return total

meets = total_eqd2(ebrt_gy, ebrt_fx, d90, 10) >= 85 and total_eqd2(ebrt_gy, ebrt_fx, rectum, 3) < 65
print(meets)
```

</details>

</div>

<div class="step exercise">

**第 5 步 · 五位患者里几位达标**

页面开头的准备代码里有一个列表 `patients`: 五位患者(都是外照射 45 Gy 分 25 次),每位是一个字典,有 `"id"`、`"d90"`、`"rectum"` 三项。数出靶区和直肠**都达标**的有几位,存进 `n_ok`;把他们的编号放进列表 `ok_ids`。

```python
# 在这里写你的代码。结果存进 n_ok 和 ok_ids。
```

```pycheck
need("n_ok", "ok_ids")
if n_ok == 4:
    fail("4 位是靶区达标的人数。还要直肠也低于 65 Gy: 两个条件用 and。")
if n_ok == 3:
    fail("3 位是直肠达标的人数。还要靶区不低于 85 Gy: 两个条件用 and。")
if n_ok != 2:
    fail(f"n_ok 现在是 {n_ok},应该是 2。")
if sorted(ok_ids) != ["B", "E"]:
    fail(f"达标的应该是 B 和 E,现在 ok_ids 是 {ok_ids}。")
ok("对了: 只有 B 和 E 两位都达标。A 和 D 靶区够了,直肠超了;C 直肠没问题,靶区只有 81.2 Gy。靶区和直肠常常此消彼长,这正是做计划时要权衡的地方。")
```

<details><summary>提示 1: 思路</summary>

先让 `n_ok = 0`、`ok_ids = []`。对每位患者: 用第 4 步的函数算靶区和直肠的总剂量,两条都满足,就把 `n_ok` 加 1,并把编号加进列表。

</details>

<details><summary>提示 2: 用到的写法</summary>

`for p in patients:`,循环里用 `p["d90"]`、`p["rectum"]`、`p["id"]` 取出这位患者的数据;往列表里加一项是 `ok_ids.append(p["id"])`。

</details>

<details><summary>参考答案</summary>

```python
n_ok = 0
ok_ids = []
for p in patients:
    target = total_eqd2(45, 25, p["d90"], 10)
    rect = total_eqd2(45, 25, p["rectum"], 3)
    if target >= 85 and rect < 65:
        n_ok = n_ok + 1
        ok_ids.append(p["id"])
    print(p["id"], round(target, 1), round(rect, 1))
print(n_ok, ok_ids)
```

</details>

</div>

<div class="step">

<div class="quiz">
<p><b>第 6 步 · 想一想</b>: 假设某一次近距离,靶区和直肠的某一小块都受到了 7 Gy。换算成 EQD2,靶区(α/β = 10)是 9.9 Gy,直肠(α/β = 3)是多少?这说明什么?</p>
<button class="quiz-opt" data-why="物理剂量相同,EQD2 并不相同: 公式里的 α/β 不一样。直肠的 EQD2 是 7 × (7 + 3) ÷ (2 + 3) = 14 Gy。">也是 9.9 Gy: 物理剂量相同,效应就相同</button>
<button class="quiz-opt" data-correct data-why="7 × (7 + 3) ÷ (2 + 3) = 14 Gy。同样的一次大剂量,对 α/β 小的晚反应组织(直肠、膀胱)伤害更大。所以近距离放疗里,危及器官的剂量要盯得很紧: 物理剂量看着不高,换算以后并不低。">14 Gy: 同样的一次大剂量,对直肠这样的晚反应组织伤害更大</button>
<button class="quiz-opt" data-why="算反了。α/β 越小,单次大剂量的 EQD2 越大: 7 × (7 + 3) ÷ (2 + 3) = 14 Gy,比靶区的 9.9 Gy 大。">比 9.9 Gy 小: 直肠的 α/β 小,所以 EQD2 也小</button>
</div>

</div>

</div>

**这一部分做完了**: 你有了两个函数 `eqd2` 和 `total_eqd2`,能算任何一位患者的总剂量。下一部分把它用到 200 位患者的表上: 学完 [Python 课第 2 课](/python/02-pandas/)以后做。
