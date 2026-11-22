---
title: "练习 4 · 把审计结果画出来"
summary: "靶区剂量的分布、两种施源器的比较、剂量和肿瘤体积的关系,再导出一张图。"
weight: 40
toc: false
duration: "约 35 分钟"
prereq: "[Python 课第 4 课](/python/04-plots/);[练习 3](../03-groups/)"
questions:
  - "200 位患者的靶区剂量是怎么分布的,有多少落在 85 Gy 以下?"
  - "靶区剂量和肿瘤体积是什么关系?两种施源器一样吗?"
objectives:
  - "直方图、箱线图、散点图"
  - "参考线、图例、导出图片"
---

上一部分的基线表说两种施源器的靶区剂量"中位数差不多"。画出来看,故事要复杂得多。图里的文字都用英文。

先运行这一块:

```python
import matplotlib.pyplot as plt
from cervix import load_cervix

df = load_cervix()                 # 200 位患者,带三列总剂量
print("准备好了:", df.shape)
```

<div class="steps" data-steps>

<div class="step exercise">

**第 1 步 · 靶区剂量的分布**

画 HR-CTV D90 总剂量(`d90_total`)的直方图,分 20 段;在审计标准 85 Gy 的位置画一条竖线;两个轴都加标签,剂量带单位。

骨架已经给出,把四处 `___` 填上。

```python
fig, ax = plt.subplots()
# 直方图,分 20 段
ax.___(df["d90_total"], bins=20)
# 85 Gy 处的竖线
ax.___(85, color="black", linestyle="--", linewidth=1)
# 两个轴的标签,剂量带单位
ax.set_xlabel(___)
ax.set_ylabel(___)
plt.show()
```

```pycheck
need("ax")
bars = list(ax.patches)
if not bars:
    fail("图里还没有柱子: 用 ax.hist(...) 画直方图。")
if abs(sum(p.get_height() for p in bars) - 200) > 0.5:
    fail("柱子加起来应该是 200 位患者。画的是 df[\"d90_total\"] 吗?")
if abs(min(p.get_x() for p in bars) - df["d90_total"].min()) > 0.01:
    fail('画的不是 d90_total 这一列。')
if not any(len(l.get_xdata()) == 2 and l.get_xdata()[0] == 85 and l.get_xdata()[1] == 85 for l in ax.lines):
    fail("还缺 85 Gy 那条竖线: ax.axvline(85, ...)。")
if not ax.get_xlabel() or not ax.get_ylabel():
    fail("两个轴都要有标签: ax.set_xlabel(...) 和 ax.set_ylabel(...)。")
ok("对了。大多数患者在 88 到 98 Gy 之间;竖线左边有 19 位患者没有达到 85 Gy。右边还有一条尾巴,少数患者超过了 100 Gy。")
```

<details><summary>提示 1: 思路</summary>

三步: 准备坐标系,画直方图,加竖线和轴标签。竖线用 `ax.axvline`。

</details>

<details><summary>提示 2: 用到的写法</summary>

直方图是 `ax.hist`,竖线是 `ax.axvline`;标签是一段文字,比如 `"HR-CTV D90, total EQD2 (Gy)"` 和 `"Number of patients"`。

</details>

<details><summary>参考答案</summary>

```python
fig, ax = plt.subplots()
# 直方图,分 20 段
ax.hist(df["d90_total"], bins=20)
# 85 Gy 处的竖线
ax.axvline(85, color="black", linestyle="--", linewidth=1)
# 两个轴的标签,剂量带单位
ax.set_xlabel("HR-CTV D90, total EQD2 (Gy)")
ax.set_ylabel("Number of patients")
plt.show()
```

</details>

</div>

<div class="step exercise">

**第 2 步 · 两种施源器放在一起比**

画箱线图: 左边是单纯腔内(IC)的 `d90_total`,右边是腔内加插植(IC/IS)的;加上 85 Gy 的横线。`ax.boxplot(...)` 交回的结果存进 `bp`。

骨架已经给出,把三处 `___` 填上。

```python
# 单纯腔内的患者,和加了插植针的(IC/IS)
ic = df.loc[df["technique"] == "IC", "d90_total"]
needles = df.loc[___, "d90_total"]

fig, ax = plt.subplots()
# 两组放进一个列表,IC 在前
bp = ax.boxplot(___, tick_labels=["IC", "IC/IS"])
# 85 Gy 处的横线
ax.___(85, color="grey", linestyle="--", linewidth=1)
ax.set_ylabel("HR-CTV D90, total EQD2 (Gy)")
plt.show()
```

```pycheck
need("bp")
med = [float(l.get_ydata()[0]) for l in bp["medians"]]
if len(med) != 2:
    fail(f"应该有两个箱子(IC 和 IC/IS),现在是 {len(med)} 个。")
if close(med[0], 92.15, 0.01) and close(med[1], 92.39, 0.01):
    fail("两个箱子的顺序反了: 第一个是 IC,第二个是 IC/IS。")
if close(med[0], 92.39, 0.01) and close(med[1], 92.15, 0.01):
    ok("对了: 两个箱子的中位数是 92.4 和 92.2 Gy,几乎一样高,箱子也大致重叠。只看这张图,会得出'加不加针,靶区剂量没有差别'的结论。先别急着下这个结论,看下一步。")
fail("箱子里的数据不对: 第一个箱子是 technique 为 IC 的患者的 d90_total,第二个是 IC/IS 的。")
```

<details><summary>提示 1: 思路</summary>

先筛出两组各自的 `d90_total`,放进一个列表交给 `ax.boxplot`;标签用 `tick_labels`。横线是 `ax.axhline`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`needles = df.loc[df["technique"] == "IC/IS", "d90_total"]`;`ax.boxplot([ic, needles], ...)`;横线是 `ax.axhline`。

</details>

<details><summary>参考答案</summary>

```python
# 单纯腔内的患者,和加了插植针的(IC/IS)
ic = df.loc[df["technique"] == "IC", "d90_total"]
needles = df.loc[df["technique"] == "IC/IS", "d90_total"]

fig, ax = plt.subplots()
# 两组放进一个列表,IC 在前
bp = ax.boxplot([ic, needles], tick_labels=["IC", "IC/IS"])
# 85 Gy 处的横线
ax.axhline(85, color="grey", linestyle="--", linewidth=1)
ax.set_ylabel("HR-CTV D90, total EQD2 (Gy)")
plt.show()
```

</details>

</div>

<div class="step exercise">

**第 3 步 · 剂量和肿瘤体积**

画散点图: 横轴是 HR-CTV 体积(`hrctv_vol`),纵轴是 `d90_total`;两种施源器用不同的颜色,带图例;加上 85 Gy 的横线。

骨架已经给出,把三处 `___` 填上。

```python
fig, ax = plt.subplots()
# 每种施源器的患者(d)各画一次,颜色自动不同
for tech, d in df.groupby("technique"):
    # 横轴体积,纵轴剂量
    ax.scatter(___, ___, s=14, alpha=0.6, label=tech)
ax.axhline(85, color="grey", linestyle="--", linewidth=1)
ax.set_xlabel("HR-CTV volume (cm3)")
ax.set_ylabel("HR-CTV D90, total EQD2 (Gy)")
# 图例
ax.___(title="Applicator")
plt.show()
```

```pycheck
import numpy as np
need("ax")
groups = list(ax.collections)
if len(groups) != 2:
    fail(f"应该有两组点(两种施源器各画一次),现在是 {len(groups)} 组。用 for tech, d in df.groupby(\"technique\"): 每组 scatter 一次。")
xy = np.vstack([np.asarray(g.get_offsets()) for g in groups])
if close(np.nansum(xy[:, 0]), df["d90_total"].sum(), 1):
    fail("横轴和纵轴放反了: 横轴是体积 hrctv_vol,纵轴是 d90_total。")
if not close(np.nansum(xy[:, 0]), df["hrctv_vol"].sum(), 1):
    fail('横轴应该是 HR-CTV 体积: "hrctv_vol"。')
if not close(np.nansum(xy[:, 1]), df["d90_total"].sum(), 1):
    fail('纵轴应该是 "d90_total"。')
if not any(len(l.get_ydata()) == 2 and l.get_ydata()[0] == 85 and l.get_ydata()[1] == 85 for l in ax.lines):
    fail("还缺 85 Gy 那条横线: ax.axhline(85, ...)。")
if ax.get_legend() is None:
    fail("还缺图例: 每组 scatter 时加 label=tech,最后 ax.legend()。")
ok("对了。这张图讲出了箱线图讲不出的事: 单纯腔内的点随着体积增大一路往下走,到 40 cm³ 以上有不少掉到了 85 Gy 线的下面;加了插植针的点几乎是平的,大肿瘤也大多在线的上面。")
```

<details><summary>提示 1: 思路</summary>

第 4 课第 4 节的写法: 对每一组各画一次散点,颜色会自动不同。没有体积记录的 5 位患者不会被画出来。

</details>

<details><summary>提示 2: 用到的写法</summary>

循环里画的是这一组 `d` 的点: `ax.scatter(d["hrctv_vol"], d["d90_total"], ...)`;图例是 `ax.legend`。

</details>

<details><summary>参考答案</summary>

```python
fig, ax = plt.subplots()
# 每种施源器的患者(d)各画一次,颜色自动不同
for tech, d in df.groupby("technique"):
    # 横轴体积,纵轴剂量
    ax.scatter(d["hrctv_vol"], d["d90_total"], s=14, alpha=0.6, label=tech)
ax.axhline(85, color="grey", linestyle="--", linewidth=1)
ax.set_xlabel("HR-CTV volume (cm3)")
ax.set_ylabel("HR-CTV D90, total EQD2 (Gy)")
# 图例
ax.legend(title="Applicator")
plt.show()
```

</details>

</div>

<div class="step">

<div class="quiz">
<p><b>第 4 步 · 读图</b>: 箱线图上两种施源器的靶区剂量几乎一样,散点图上却很不一样。哪一种解释对?</p>
<button class="quiz-opt" data-why="两张图用的是同一份数据,都没有画错。区别在于箱线图把所有患者混在一起,散点图把肿瘤体积也摆了出来。">两张图有一张画错了</button>
<button class="quiz-opt" data-correct data-why="单纯腔内的患者多是小肿瘤,剂量容易给够;加针的患者多是大肿瘤,靠着针才把剂量拉到了同样的水平。混在一起比,两个效应正好抵消。按肿瘤大小分开看,针的作用就显出来了。">加针的患者肿瘤更大。小肿瘤不加针,和大肿瘤加针,得到了差不多的剂量;混在一起比,针的作用被肿瘤大小的差别盖住了</button>
<button class="quiz-opt" data-why="散点图上,大肿瘤里加针的患者明显高于不加针的。说'没有作用'和图不符。">插植针对靶区剂量没有作用</button>
</div>

</div>

<div class="step exercise">

**第 5 步 · 导出审计报告用的图**

把第 3 步的散点图按 3.5 × 2.8 英寸重画,存成 300 dpi 的 `audit_figure.png`。点和图例的字要小一些,才放得下。

骨架已经给出: 和第 3 步的图一样,点和图例的字已经调小。把三处 `___` 填上。

```python
# 图的尺寸: 宽 3.5 英寸、高 2.8 英寸
fig, ax = plt.subplots(figsize=___)
for tech, d in df.groupby("technique"):
    ax.scatter(d["hrctv_vol"], d["d90_total"], s=8, alpha=0.6, label=tech)
ax.axhline(85, color="grey", linestyle="--", linewidth=1)
ax.set_xlabel("HR-CTV volume (cm3)")
ax.set_ylabel("HR-CTV D90 (Gy)")
ax.legend(title="Applicator", fontsize=7, title_fontsize=7)
# 存成文件: 文件名,分辨率
fig.savefig(___, dpi=___, bbox_inches="tight")
plt.show()
```

```pycheck
import os
import matplotlib.pyplot as plt
if not os.path.exists("audit_figure.png"):
    fail('还没有存出 audit_figure.png: fig.savefig("audit_figure.png", dpi=300, bbox_inches="tight")。')
h, w = plt.imread("audit_figure.png").shape[:2]
if w < 900:
    fail(f"图只有 {w} 像素宽。图的尺寸是 figsize=(3.5, 2.8),分辨率 dpi=300。")
if w > 1300:
    fail(f"图有 {w} 像素宽,比 3.5 英寸 × 300 dpi 大得多。在 plt.subplots 里写 figsize=(3.5, 2.8)。")
ok(f"对了: {w} × {h} 像素,3.5 英寸宽、300 dpi。这张图可以直接放进审计报告。")
```

<details><summary>提示 1: 思路</summary>

和第 3 步的代码一样,只是准备坐标系时写明尺寸,点和字调小,最后多一句存文件。

</details>

<details><summary>提示 2: 用到的写法</summary>

`figsize=(3.5, 2.8)`;`fig.savefig("audit_figure.png", dpi=300, bbox_inches="tight")`。

</details>

<details><summary>参考答案</summary>

```python
# 图的尺寸: 宽 3.5 英寸、高 2.8 英寸
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for tech, d in df.groupby("technique"):
    ax.scatter(d["hrctv_vol"], d["d90_total"], s=8, alpha=0.6, label=tech)
ax.axhline(85, color="grey", linestyle="--", linewidth=1)
ax.set_xlabel("HR-CTV volume (cm3)")
ax.set_ylabel("HR-CTV D90 (Gy)")
ax.legend(title="Applicator", fontsize=7, title_fontsize=7)
# 存成文件: 文件名,分辨率
fig.savefig("audit_figure.png", dpi=300, bbox_inches="tight")
plt.show()
```

</details>

</div>

</div>

**这一部分做完了**: 同一份数据,箱线图说"没差别",散点图说"大肿瘤里差别很大"。先画图再下结论,而且要画对图。下一部分: 学完 [Python 课第 5 课](/python/05-uncertainty/)以后做。
