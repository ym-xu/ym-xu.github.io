---
title: "Lesson 4 · 画图 Plotting"
summary: "直方图、箱线图、散点图;把同一位患者连起来;导出论文用的图。"
weight: 40
toc: true
duration: "约 70 分钟,可以分两次学: 第 1–3 节,第 4–7 节"
prereq: "[第 3 课](../03-table-one/)"
questions:
  - "一个指标在患者中间是怎么分布的,怎么画出来?"
  - "几组、几套计划之间怎么比?两个量的关系怎么看?"
  - "什么样的图能放进论文,怎么导出?"
objectives:
  - "用直方图看一个指标的分布"
  - "用箱线图比较几组数据,并说出箱线图每一部分的含义"
  - "用散点图看两个量的关系,按组上色"
  - "给图加上带单位的轴标签、参考线和图例"
  - "把图按期刊的要求导出成文件"
---

第 3 课的 Table 1 用几个数概括了一批患者。可是数会藏起很多东西: 中位数都是 −2 Gy 的两批数据,可以一批挤在一起,另一批拖着一条长尾巴;相关系数相同的两张散点图,可以长得完全不一样。看片子不能只读报告,看数据也不能只读汇总表: **先画出来看一眼**。

这一课用 Python 最常用的画图包 **matplotlib**,画临床论文里最常见的三种图: 直方图、箱线图、散点图,最后把图导出成期刊要的文件。

> 🔬 图里的文字都用英文: 论文的图本来就要英文,而且浏览器里的 Python 没有中文字体。

**练习怎么做**: 讲解里画的是**体积优化**计划和阴道 D2cc;练习换成**图形优化**计划和靶区 D90。每道练习都会检查你画出来的图。

## 0. 准备

```python
import matplotlib.pyplot as plt
import pandas as pd
from cohort import make_cohort

df = make_cohort()
df["delta_vol"] = df["d2cc_vol"] - df["d2cc_5mm"]
df["delta_gra"] = df["d2cc_gra"] - df["d2cc_5mm"]
print(df.shape)
```

第一次运行要下载 matplotlib,会多等一会儿。`import matplotlib.pyplot as plt` 是通用的写法。

## 1. 第一张图: 直方图

平均壁厚在这 200 位患者里是怎么分布的?

```python
fig, ax = plt.subplots()
ax.hist(df["wall_mean"], bins=20)
ax.set_xlabel("Mean vaginal wall thickness (mm)")
ax.set_ylabel("Number of patients")
plt.show()
```

**直方图**(histogram)把横轴切成一样宽的小段(这里 20 段,`bins=20`),数每一段里有几位患者,画成柱子。从图上能读到表里读不到的东西: 大多数患者在 2.5 到 4.5 mm 之间,右边拖着一条尾巴,有几位患者超过 6 mm。这种"一边有长尾"的形状在临床数据里很常见(住院天数、化验值、费用),第 5 课会讲它对"报均值还是中位数"的影响。

每张图都是这三步:

1. `fig, ax = plt.subplots()`: 准备一张空白的图。`fig` 是整张图,`ax` 是里面的坐标系(axes),画什么都通过 `ax`。
2. `ax.hist(...)`: 往坐标系里画东西。
3. `ax.set_xlabel(...)`、`ax.set_ylabel(...)`: 写上轴标签。**轴标签要带单位**。没有标签和单位的图,就像没写单位的化验单。

`plt.show()` 把图显示出来。4 位没有壁厚测量的患者(缺失值)不会被画进去。

<div class="exercise">

**练习 1 · 轮到你: 图形优化的 Δ**

画出图形优化计划的 Δ(`delta_gra`)的直方图,20 段,并在 0 的位置加一条竖的虚线,标出"没有变化"。`ax.axvline(x)` 在横轴的 x 处画一条竖线。

```python
fig, ax = plt.subplots()
ax.hist(df[___], bins=20)
ax.axvline(___, color="grey", linestyle="--")
ax.set_xlabel("Change in vaginal D2cc, Graphical − 5 mm (Gy)")
ax.set_ylabel("Number of patients")
plt.show()
```

```pycheck
need("ax")
bars = list(ax.patches)
if not bars:
    fail("图里还没有柱子: 用 ax.hist(...) 画直方图。")
total = sum(p.get_height() for p in bars)
if abs(total - 200) > 0.5:
    fail(f"柱子加起来应该是 200 位患者,现在是 {total:.0f}。")
if abs(min(p.get_x() for p in bars) - df["delta_gra"].min()) > 0.01:
    fail('画的不是图形优化的 Δ: 第一处填 "delta_gra"。')
if not any(len(l.get_xdata()) == 2 and l.get_xdata()[0] == 0 and l.get_xdata()[1] == 0 for l in ax.lines):
    fail('还缺 0 那里的竖线: ax.axvline(0, ...)。')
ok("对了。大多数患者在 0 的左边(D2cc 降低了),左边拖着一条长尾,有 10 位患者降了 10 Gy 以上;也有 31 位患者在 0 的右边: 换了计划,D2cc 反而升高了。只看中位数 −3.4 Gy,这两件事都看不到。")
```

<details><summary>提示</summary>

列名 `"delta_gra"` 要加引号;竖线画在 0。

</details>

<details><summary>参考答案</summary>

```python
fig, ax = plt.subplots()
ax.hist(df["delta_gra"], bins=20)
ax.axvline(0, color="grey", linestyle="--")
ax.set_xlabel("Change in vaginal D2cc, Graphical − 5 mm (Gy)")
ax.set_ylabel("Number of patients")
plt.show()
```

</details>

</div>

## 2. 图上该有什么: 参考线和图例

两套计划的 D2cc 画在同一张图里,再把 68 Gy 这条线标出来:

```python
bins = range(52, 92, 2)                    # 两组用同样的分段: 52, 54, …, 90
fig, ax = plt.subplots()
ax.hist(df["d2cc_5mm"], bins=bins, alpha=0.6, label="5 mm")
ax.hist(df["d2cc_vol"], bins=bins, alpha=0.6, label="Volume")
ax.axvline(68, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Vaginal D2cc, total EQD2 (Gy)")
ax.set_ylabel("Number of patients")
ax.legend()
plt.show()
```

新出现的几样:

- `alpha=0.6` 让柱子半透明,两组重叠的地方都看得见。
- `label="..."` 给每一组起名字,`ax.legend()` 把它们列成**图例**(legend)。
- `ax.axvline(68, ...)` 画一条**参考线**,就像化验单上的参考范围。读图的人一眼能看到: 换成体积优化后,整个分布往左移了一点,68 Gy 左边的人多了。
- 两组用**同样的分段**(`bins=bins`)。如果让两组各自分段,柱子的宽度和位置不一样,就没法比。

论文里的图一般不在图上写标题,标题和说明写在图下面的图注(figure legend)里。

## 3. 箱线图: 比较几组

三套计划的阴道 D2cc 放在一起比:

```python
fig, ax = plt.subplots()
ax.boxplot([df["d2cc_5mm"], df["d2cc_vol"], df["d2cc_gra"]],
           tick_labels=["5 mm", "Volume", "Graphical"])
ax.axhline(68, color="grey", linestyle="--", linewidth=1)
ax.set_ylabel("Vaginal D2cc, total EQD2 (Gy)")
plt.show()
```

**箱线图**(box plot)用五个数概括一组数据,读法是:

- 箱子中间的线是**中位数**;
- 箱子的下沿和上沿是下四分位数和上四分位数,所以箱子装着中间的 50% 患者,箱子的高度就是第 3 课的四分位间距;
- 上下两根"须"伸到离箱沿不超过 1.5 倍箱高的最远的患者;
- 再远的患者单独画成圆点,常被叫作离群值(outlier)。它们是提醒你去看一眼的患者,不是可以随手删掉的患者。

图上能看到: 从 5 mm 到体积优化再到图形优化,中位数一路下降(71.0、69.3、67.4 Gy),图形优化的中位数已经落在 68 Gy 线的下面;同时箱子越拉越长,也就是患者之间的差别越来越大。

<div class="quiz">
<p><b>想一想</b>: 箱线图的箱子中间那条线是什么?箱子里装着多少患者?</p>
<button class="quiz-opt" data-why="箱线图里画的是中位数,不是均值。数据有长尾时两者会差不少。箱子装的也不是全部患者。">均值;全部患者</button>
<button class="quiz-opt" data-correct data-why="线是中位数;箱子从下四分位数到上四分位数,装着中间的一半患者,另外一半在箱子外面(须和圆点)。">中位数;中间的 50%</button>
<button class="quiz-opt" data-why="线确实是中位数,但箱子不是 95%: 箱子的上下沿是两个四分位数,只装着中间的一半患者。">中位数;中间的 95%</button>
</div>

<div class="exercise">

**练习 2 · 轮到你: 三套计划的 D90**

只看外照射加近距离的患者(已经筛好,放在 `ebrt` 里),画出三套计划的靶区 D90 的箱线图,顺序是 5 mm、体积优化、图形优化。

```python
ebrt = df[df["arm"] == "EBRT+VCB"]
fig, ax = plt.subplots()
bp = ax.boxplot([ebrt["d90_5mm"], ebrt[___], ebrt[___]],
                tick_labels=["5 mm", "Volume", "Graphical"])
ax.set_ylabel("CTV D90, total EQD2 (Gy)")
plt.show()
```

```pycheck
need("bp")
med = [float(m.get_ydata()[0]) for m in bp["medians"]]
if len(med) != 3:
    fail(f"应该有三个箱子,现在是 {len(med)} 个。")
if close(med[0], 61.4, 0.01) and close(med[1], 60.4, 0.01) and close(med[2], 59.5, 0.01):
    ok("对了: 中位数是 61.4、60.4、59.5 Gy。阴道 D2cc 降下来的同时,靶区 D90 也降了一点。两组患者的总剂量差得很远(第 3 课练习 1),所以这里只画了一组,没有把两组混在一起。")
if close(med[1], 59.5, 0.01) and close(med[2], 60.4, 0.01):
    fail('后两个箱子的顺序反了: 标签写的是 Volume 在前,数据也要 "d90_vol" 在前。')
if close(med[0], 61.4, 0.01):
    fail('第一个箱子对了。后两个依次填 "d90_vol" 和 "d90_gra"。')
fail("箱子里的数据不对: 三个箱子依次是 d90_5mm、d90_vol、d90_gra,都取自 ebrt。")
```

<details><summary>提示</summary>

两处依次填 `"d90_vol"` 和 `"d90_gra"`。数据的顺序要和 `tick_labels` 里标签的顺序一致,否则图上的名字就张冠李戴了。

</details>

<details><summary>参考答案</summary>

```python
ebrt = df[df["arm"] == "EBRT+VCB"]
fig, ax = plt.subplots()
bp = ax.boxplot([ebrt["d90_5mm"], ebrt["d90_vol"], ebrt["d90_gra"]],
                tick_labels=["5 mm", "Volume", "Graphical"])
ax.set_ylabel("CTV D90, total EQD2 (Gy)")
plt.show()
```

</details>

</div>

## 4. 散点图: 两个量的关系

阴道壁的厚度,和换成体积优化后 D2cc 的变化有什么关系?两个连续的量之间的关系,用**散点图**(scatter plot): 一位患者一个点,横轴是一个量,纵轴是另一个量。

```python
fig, ax = plt.subplots()
ax.scatter(df["wall_mean"], df["delta_vol"], s=14, alpha=0.6)
ax.axhline(0, color="grey", linewidth=1)
ax.set_xlabel("Mean vaginal wall thickness (mm)")
ax.set_ylabel("Change in vaginal D2cc, Volume − 5 mm (Gy)")
plt.show()
```

`s=14` 是点的大小,`alpha=0.6` 让重叠的点看得出来。点从左下往右上走: 壁越薄,D2cc 降得越多;壁厚到 5 mm 以上,不少患者反而在 0 线的上面。这个关系有多强、是不是碰巧,第 6 课用相关系数来量。

想让两种治疗方式用不同的颜色,只要对每一组各画一次:

```python
fig, ax = plt.subplots()
for arm, d in df.groupby("arm"):
    ax.scatter(d["wall_mean"], d["delta_vol"], s=14, alpha=0.6, label=arm)
ax.axhline(0, color="grey", linewidth=1)
ax.set_xlabel("Mean vaginal wall thickness (mm)")
ax.set_ylabel("Change in vaginal D2cc, Volume − 5 mm (Gy)")
ax.legend(title="Treatment")
plt.show()
```

`for arm, d in df.groupby("arm")` 是第 3 课的分组,这次不汇总,而是把每一组(`d`)拿出来画一次。matplotlib 自动给每一组换一种颜色。

<div class="quiz">
<p><b>想一想</b>: 想知道"三套计划里,哪一套的靶区 D90 更高",该画哪种图?</p>
<button class="quiz-opt" data-why="直方图适合看一个指标自己的分布。三组叠在一起会很难看清。">直方图</button>
<button class="quiz-opt" data-correct data-why="比较几组数据的高低和离散程度,用箱线图(或者把每位患者的点画出来)。">箱线图</button>
<button class="quiz-opt" data-why="散点图看的是两个连续的量之间的关系,比如壁厚和 Δ。计划是分类,不是连续的量。">散点图</button>
</div>

<div class="exercise">

**练习 3 · 轮到你: 壁厚和图形优化的 Δ**

画散点图: 横轴是平均壁厚(`wall_mean`),纵轴是图形优化的 Δ(`delta_gra`),两种治疗方式用不同的颜色,带图例。

```python
fig, ax = plt.subplots()
for arm, d in df.groupby("arm"):
    ax.scatter(d[___], d[___], s=14, alpha=0.6, label=arm)
ax.axhline(0, color="grey", linewidth=1)
ax.set_xlabel("Mean vaginal wall thickness (mm)")
ax.set_ylabel("Change in vaginal D2cc, Graphical − 5 mm (Gy)")
ax.legend(title="Treatment")
plt.show()
```

```pycheck
import numpy as np
need("ax")
groups = list(ax.collections)
if len(groups) != 2:
    fail(f"应该有两组点(两种治疗方式),现在是 {len(groups)} 组。")
xy = np.vstack([np.asarray(g.get_offsets()) for g in groups])
if len(xy) != 200:
    fail(f"一共应该画 200 位患者,现在是 {len(xy)} 位。")
if close(np.nansum(xy[:, 0]), df["delta_gra"].sum(), 0.5) and close(np.nansum(xy[:, 1]), df["wall_mean"].sum(), 0.5):
    fail("横轴和纵轴放反了: 第一处是横轴 wall_mean,第二处是纵轴 delta_gra。")
if not close(np.nansum(xy[:, 0]), df["wall_mean"].sum(), 0.5):
    fail('横轴应该是平均壁厚: 第一处填 "wall_mean"。')
if not close(np.nansum(xy[:, 1]), df["delta_gra"].sum(), 0.5):
    fail('纵轴应该是图形优化的 Δ: 第二处填 "delta_gra"。')
if ax.get_legend() is None:
    fail("还缺图例: ax.legend()。")
ok("对了。和体积优化一样,壁越薄降得越多;单纯近距离的患者(近距离做了 3 次)散得更开,降得也更多。")
```

<details><summary>提示</summary>

`ax.scatter(横轴, 纵轴, ...)`: 先 `d["wall_mean"]`,后 `d["delta_gra"]`。

</details>

<details><summary>参考答案</summary>

```python
fig, ax = plt.subplots()
for arm, d in df.groupby("arm"):
    ax.scatter(d["wall_mean"], d["delta_gra"], s=14, alpha=0.6, label=arm)
ax.axhline(0, color="grey", linewidth=1)
ax.set_xlabel("Mean vaginal wall thickness (mm)")
ax.set_ylabel("Change in vaginal D2cc, Graphical − 5 mm (Gy)")
ax.legend(title="Treatment")
plt.show()
```

</details>

</div>

## 5. 同一位患者的三套计划: 把点连起来

第 3 节的箱线图把三套计划画成了三组互不相干的数。可是它们来自**同一批患者**: 每位患者在三个箱子里各有一个数。把同一位患者的三个数用线连起来,能看到箱线图看不到的东西。画前 30 位患者:

```python
plans = ["5 mm", "Volume", "Graphical"]
fig, ax = plt.subplots()
for _, row in df.head(30).iterrows():
    ax.plot(plans, [row["d2cc_5mm"], row["d2cc_vol"], row["d2cc_gra"]],
            color="grey", alpha=0.6, linewidth=0.8, marker="o", markersize=3)
ax.axhline(68, color="black", linestyle="--", linewidth=1)
ax.set_ylabel("Vaginal D2cc, total EQD2 (Gy)")
plt.show()
```

`df.head(30).iterrows()` 一行一行地取出前 30 位患者(`row` 就是第 1 课的"一位患者一个字典");`ax.plot(横轴的值, 纵轴的值)` 把这位患者的三个点连成线。

每条线是一位患者。大多数线往右下走,但不是全部: 有的患者换成图形优化后反而升高;穿过 68 Gy 虚线的那些线,就是换了计划以后从虚线上面到了虚线下面(或者反过来)的患者。**成对的数据,画图时把"对"画出来**: 这和第 3 课说的"在宽表里两列相减"是同一个道理,第 6 课的配对检验用的也是它。

## 6. 一张图里放几个小图

把体积优化和图形优化的 Δ 上下排在一起,共用横轴,方便比较:

```python
fig, axes = plt.subplots(2, 1, figsize=(4.5, 5.5), sharex=True)
for ax, col, name in [(axes[0], "delta_vol", "Volume − 5 mm"), (axes[1], "delta_gra", "Graphical − 5 mm")]:
    ax.hist(df[col], bins=range(-18, 9, 1))
    ax.axvline(0, color="grey", linestyle="--")
    ax.set_ylabel("Number of patients")
    ax.set_title(name, fontsize=10)
axes[1].set_xlabel("Change in vaginal D2cc (Gy)")
plt.tight_layout()
plt.show()
```

`plt.subplots(2, 1)` 给出 2 行 1 列的小图,`axes[0]` 是上面那个,`axes[1]` 是下面那个;`figsize=(宽, 高)` 的单位是英寸;`sharex=True` 让两个小图用同一个横轴;`plt.tight_layout()` 自动调整间距,免得文字叠在一起。

共用横轴以后一眼能看出: 图形优化的分布整体更靠左,也更宽。**要比较的图,用同样的坐标范围**;各画各的范围,读图的人会被柱子的样子骗到。

## 7. 导出论文用的图

```python
fig, ax = plt.subplots(figsize=(3.5, 2.8))
ax.scatter(df["wall_mean"], df["delta_vol"], s=8, alpha=0.6)
ax.axhline(0, color="grey", linewidth=1)
ax.set_xlabel("Mean vaginal wall thickness (mm)")
ax.set_ylabel("Change in D2cc (Gy)")
fig.savefig("figure1.png", dpi=300, bbox_inches="tight")
print(plt.imread("figure1.png").shape[:2], "像素(高, 宽)")
plt.show()
```

`fig.savefig(文件名, ...)` 把图存成文件,格式由文件名的后缀决定。投稿时要留意这几处:

- `dpi=300`: 每英寸 300 个像素。期刊对位图(PNG、TIFF)一般要求 300 dpi 以上。屏幕上看着清楚的截图通常不到 100 dpi,印出来是糊的,所以**不要用截图投稿**。
- `figsize=(3.5, 2.8)`: 图的实际尺寸(英寸)。很多期刊的单栏宽度在 3.5 英寸(约 9 cm)上下。按最终印出来的大小来画,字号才合适;画得很大再缩小,字就小得看不清。
- `bbox_inches="tight"`: 裁掉四周多余的白边。
- 把后缀换成 `.pdf` 或 `.svg`,存出来的是矢量图,放多大都清楚,期刊也常常接受。

具体要什么格式、多大尺寸,以投稿期刊的作者须知为准。在浏览器里,文件存在临时空间里;第 7 课在自己电脑上运行时,它会出现在你的文件夹里。

<div class="exercise">

**练习 4 · 轮到你: 导出一张图**

把练习 3 的散点图按 3.5 × 2.8 英寸重新画一遍,并存成 300 dpi 的 `figure_gra.png`。

```python
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for arm, d in df.groupby("arm"):
    ax.scatter(d["wall_mean"], d["delta_gra"], s=8, alpha=0.6, label=arm)
ax.axhline(0, color="grey", linewidth=1)
ax.set_xlabel("Mean vaginal wall thickness (mm)")
ax.set_ylabel("Change in D2cc (Gy)")
ax.legend(title="Treatment", fontsize=7, title_fontsize=7)
fig.savefig(___, dpi=___, bbox_inches="tight")
plt.show()
```

```pycheck
import os
if not os.path.exists("figure_gra.png"):
    fail('还没有存出 figure_gra.png: 第一处填文件名 "figure_gra.png"(带引号)。')
import matplotlib.pyplot as plt
h, w = plt.imread("figure_gra.png").shape[:2]
if w < 900:
    fail(f"图只有 {w} 像素宽,分辨率不够。dpi 填 300。")
if w > 1300:
    fail(f"图有 {w} 像素宽,比 3.5 英寸 × 300 dpi 大得多。dpi 填 300。")
ok(f"对了: 存出来的图是 {w} × {h} 像素,也就是 3.5 英寸宽、300 dpi 左右。这样的文件可以直接交给期刊。")
```

<details><summary>提示</summary>

文件名是文字,要加引号;`dpi=300`。

</details>

<details><summary>参考答案</summary>

```python
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for arm, d in df.groupby("arm"):
    ax.scatter(d["wall_mean"], d["delta_gra"], s=8, alpha=0.6, label=arm)
ax.axhline(0, color="grey", linewidth=1)
ax.set_xlabel("Mean vaginal wall thickness (mm)")
ax.set_ylabel("Change in D2cc (Gy)")
ax.legend(title="Treatment", fontsize=7, title_fontsize=7)
fig.savefig("figure_gra.png", dpi=300, bbox_inches="tight")
plt.show()
```

</details>

</div>

## 好图的几条规矩

- **把数据画出来**。患者不多时,画每位患者的点(散点、连线),不要只画"柱子加误差线": 柱子只告诉读者一个均值,把分布、离群的患者、成对的关系都藏了起来。
- **轴标签带单位**;要比较的图用同样的坐标范围。
- **柱状图的纵轴从 0 开始**;把纵轴截断会夸大差别。散点图和箱线图不需要从 0 开始。
- **不用装饰**: 三维效果、阴影、饼图,都让数据更难读。
- **颜色**: 不要只靠红和绿来区分两组(红绿色盲的读者分不出来);组不多时可以同时换点的形状,比如 `marker="s"`。
- **一张图只讲一件事**。想讲两件事,就画两张。

<div class="quiz">
<p><b>想一想</b>: 两组各有 12 位患者,要比较一个连续的指标。下面哪种图最好?</p>
<button class="quiz-opt" data-why="柱子加误差线只给出均值和一个离散程度,12 位患者各自的值、有没有离得很远的患者,都看不到。两组完全不同的数据,可以画出一模一样的柱子。">两根柱子(均值)加误差线</button>
<button class="quiz-opt" data-correct data-why="患者这么少,每一位都画得下。读者能直接看到分布、重叠了多少、有没有特殊的患者;再加一条短线标出中位数就够了。">把每位患者画成一个点,每组再标出中位数</button>
<button class="quiz-opt" data-why="饼图只能表示构成(各部分占整体的百分之几),不能比较两组连续指标的高低。">两个饼图</button>
</div>

## 小结

- 每张图三步: `fig, ax = plt.subplots()`,用 `ax` 画,写上带单位的轴标签。
- 直方图看一个指标的分布;箱线图比较几组;散点图看两个量的关系。
- 箱线图: 中间的线是中位数,箱子装着中间一半的患者。
- 成对的数据(同一位患者的几套计划),把同一位患者连起来。
- `fig.savefig("名字.png", dpi=300, bbox_inches="tight")` 导出;按最终印出来的尺寸画图。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 直方图 | histogram | 把数值分段,数每一段有几位患者 |
| 箱线图 | box plot | 用中位数、四分位数和须概括一组数据 |
| 散点图 | scatter plot | 一位患者一个点,看两个量的关系 |
| 离群值 | outlier | 离其他患者很远的值 |
| 坐标轴、轴标签 | axis、axis label | 横轴和纵轴,以及旁边说明含义和单位的文字 |
| 图例 | legend | 说明每种颜色或形状代表哪一组 |
| 图注 | figure legend、caption | 论文里图下面的标题和说明 |
| 分辨率 | resolution (dpi) | 每英寸有多少个像素 |
| 矢量图 | vector graphics | 放大不会变糊的图,如 PDF、SVG |

## 这一课没讲什么

- **生存曲线**(Kaplan–Meier)、**森林图**、**ROC 曲线**: 都是专门的图。ROC 曲线在统计机器学习课的第 10 课。
- **seaborn** 等更高层的画图包: 一句代码就能画出更复杂的统计图,底层仍然是 matplotlib,这一课学的都用得上。
- **配色和排版的细节**: 以期刊的要求为准。

## 写进论文 Writing it up

<div class="paper">

图注要让读者不看正文也能读懂这张图: 画的是什么、每个符号代表什么、有多少患者。

> **Figure 1.** Change in vaginal D2cc (total EQD2) after volume-based optimisation compared with the 5 mm prescription, plotted against mean vaginal wall thickness. Each point is one patient (n = 196 with wall measurements); the horizontal line marks no change.

箱线图的图注要说明箱子和须的含义,因为不同的软件画法不一样:

> Boxes show the median and interquartile range; whiskers extend to the most extreme values within 1.5 times the interquartile range; points beyond are shown individually.

**审稿人可能问**: 为什么不画每位患者的数据?误差线是标准差、标准误还是置信区间?

**怎么回答**: 能画出每位患者就画出来;用了误差线,就在图注里写明它是什么。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| 图是空白的,或者没有显示 | 画同一张图的几句代码要在同一个代码块里;先 `plt.subplots()`,再用得到的 `ax` 画。 |
| `ValueError: x and y must be the same size` | 散点图的横轴和纵轴数据的个数不同。检查两列是不是来自同一张表。 |
| 图上的汉字变成方框 | 浏览器里的 Python 没有中文字体。图里的文字用英文。 |
| 箱线图的标签和数据对不上 | 数据的顺序和 `tick_labels` 的顺序不一致。 |
| `TypeError: ... got an unexpected keyword argument 'tick_labels'` | 自己电脑上的 matplotlib 版本比较旧(早于 3.9),把 `tick_labels` 换成 `labels`。 |

## 延伸阅读 Further reading

- matplotlib 官方教程: [Quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html)。
- Weissgerber TL, et al. [Beyond bar and line graphs: time for a new data presentation paradigm](https://doi.org/10.1371/journal.pbio.1002128). *PLoS Biology* 2015;13:e1002128。为什么小样本不该只画柱子,写论文时可以引用。

{{< practice >}}

**下一课**: 直方图上的那条长尾,决定了该报均值还是中位数。而任何一个从 200 位患者算出来的数,换一批患者都会变。第 5 课讲怎么描述数据,怎么说清楚一个结果有多不确定。
