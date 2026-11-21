---
title: "Lesson 7 · 从 Excel 到 Table 1 和图 From Spreadsheet to Results"
summary: "在自己电脑上装好 Python;病人数据的规矩;把一份 Excel 从读入做到 Table 1、主要结果和图。"
weight: 70
toc: true
duration: "约 2 小时,可以分三次学: 第 1–3 节(安装和规矩),第 4 节(走一遍),第 5 节(你自己的数据)"
prereq: "[第 1–6 课](../)"
questions:
  - "怎么在自己的电脑上运行 Python?"
  - "病人的数据,哪些事不能做?"
  - "拿到一份 Excel,从头到尾该怎么做,才能随时重做一遍?"
objectives:
  - "在自己的电脑上安装 Python,用 JupyterLab 运行代码"
  - "按固定的文件夹结构组织一个分析项目"
  - "说出处理病人数据的几条规矩,并照着做"
  - "把一份表从读入、检查、清理,做到 Table 1、主要结果、图和结果里的句子"
  - "用自己的数据重复这套流程"
---

前六课都在网页里,用的是准备好的数据。真正的研究要用你自己的数据,而病人的数据不能传到任何网页上。所以最后一课做三件事: 在你自己的电脑上装好 Python;说清楚病人数据的规矩;然后把一份 Excel 从头到尾做一遍,做出 Table 1、主要结果、一张图,甚至结果部分的句子。

这一课的分析没有新的方法,用的全是前六课学过的东西。新的是把它们**连起来**,以及一条贯穿始终的原则: 从原始数据到论文里的每一个数,中间的每一步都写在代码里,随时可以重新运行一遍。

> 🔬 第 4 节走的流程(三套计划两两比较、bootstrap 区间、配对检验加校正、和壁厚的相关),就是我们自己那项研究的分析流程的缩小版,只是换成了模拟病人。

**练习怎么做**: 讲解里分析的是阴道 **D2cc**。练习让你对靶区 **D90** 把同样的分析做一遍。网页里的代码照样可以运行;装好 Python 以后,建议在自己的电脑上再做一遍。

## 1. 在自己的电脑上装好 Python

需要三样东西: Python 本身、这门课用过的几个包、一个写代码和运行代码的地方(JupyterLab)。

**第一步: 安装 Python**。到 Python 的官方网站 [python.org](https://www.python.org/downloads/) 下载安装程序,双击安装。Windows 上安装时,如果看到"Add python.exe to PATH"这个选项,把它勾上。

**第二步: 打开终端**(terminal)。终端是一个用打字来下命令的窗口。

- Windows: 在开始菜单里搜索 "PowerShell",打开它。
- macOS: 打开"终端"(在"应用程序 → 实用工具"里)。

**第三步: 安装这门课用过的包**。在终端里输入下面这一行,按回车,等它下载完:

```text
Windows:   py -m pip install jupyterlab pandas matplotlib scipy openpyxl
macOS:     python3 -m pip install jupyterlab pandas matplotlib scipy openpyxl
```

`pip` 是 Python 自带的装包工具。`jupyterlab` 是写代码的地方,`openpyxl` 让 pandas 能读写 Excel 文件,其余几个前面的课都用过。下载很慢的话(在国内常见),在这一行的最后加上 `-i https://pypi.tuna.tsinghua.edu.cn/simple`,换成国内的镜像。

**第四步: 启动 JupyterLab**。在终端里输入:

```text
Windows:   py -m jupyterlab
macOS:     python3 -m jupyterlab
```

也可以直接输入 `jupyter lab`,效果一样。浏览器会自动打开 JupyterLab。它看起来像网页,其实完全运行在你自己的电脑上,不联网也能用,数据不会离开这台电脑。用完以后,关掉浏览器的页面,再回到终端按 Ctrl + C 结束它。

**第五步: 新建一个笔记本**(notebook)。在 JupyterLab 左边的文件列表里找到你的项目文件夹(下一节),点 "Python 3 (ipykernel)" 新建一个笔记本。笔记本由一个个格子(cell)组成,和这门课网页上的代码块一样: 在格子里写代码,按 Shift + 回车运行,结果显示在下面。前面的格子里定义的名字,后面的格子可以接着用。笔记本存成 `.ipynb` 文件。

试一下: 在第一个格子里输入 `import pandas as pd` 和 `print(pd.__version__)`,运行。能显示出版本号,就装好了。

如果你的单位已经装了 Anaconda,不用再装一遍: 它自带 Python、这些包和 JupyterLab,从开始菜单里打开 JupyterLab 就行。

## 2. 一个项目的文件夹

每个研究一个文件夹,里面固定分成三块:

```text
vcb-practice/
  data/        原始数据。放进来以后不再手工修改
  outputs/     代码生成的表和图。随时可以删掉重新生成
  analysis.ipynb   分析的代码(笔记本)
```

规矩只有一条: **箭头是单向的**。代码读 `data/` 里的原始数据,把结果写进 `outputs/`;原始数据永远不被改动,`outputs/` 里的东西永远不手工修改。这样任何一个数出了问题,都能顺着代码查回去;数据更新了,重新运行一遍,所有的表和图就都更新了。

练习用的文件在这里: [cohort.xlsx](/python/cohort.xlsx)(模拟病人,可以放心下载)。下载以后放进 `data/` 文件夹。它有两张工作表: `patients`(200 位病人的基线和三套计划的剂量)和 `followup`(随访)。

## 3. 病人数据的规矩

模拟的数据随便用。真实病人的数据,下面这些是底线。具体的规定以你所在的医院和伦理委员会为准。

- **分析用的表里没有能认出病人的信息**: 不放姓名、病历号、身份证号、电话、住址;出生日期换成年龄。每位病人用研究里自己编的编号。编号和病历号的对照表单独保存、加密,不和分析用的数据放在一起。这一步叫**去标识**(de-identification)。
- **数据不离开允许的设备**: 不用微信、邮件发送,不传到网盘,不传给任何在线工具(在线格式转换、在线统计网站、在线的代码运行环境)。
- **不把病人数据贴给 AI**。让 AI 帮忙写代码或查错时,只描述表的样子(有哪些列、是什么类型),或者编几行假数据。病历号和逐例的数值都不要贴。
- **代码可以共享,数据不可以**。把代码发给合作者、放进代码仓库都没问题,前提是里面没有夹带数据: 不把数据文件和代码放在一起上传,也不把笔记本里显示出来的病人数据一起发出去。
- **备份**放在同样受保护的地方。

这也是我们自己做研究时的规矩: 原始数据只留在医院允许的电脑和服务器上,网页和论文里出现的只有汇总的结果。

<div class="quiz">
<p><b>想一想</b>: 代码报了一个看不懂的错,你想让 AI 帮忙看看。下面哪种做法可以?</p>
<button class="quiz-opt" data-why="报错信息和代码可以贴,但数据文件里是真实病人的数据。即使去掉了姓名,逐例的数据加上年龄、日期等信息,仍然可能认出病人,而且数据一旦上传就收不回来了。">把代码、报错信息和数据文件一起发过去,方便它重现问题</button>
<button class="quiz-opt" data-correct data-why="AI 需要的是代码、报错信息和表的结构。告诉它有哪些列、各是什么类型,必要时自己编两三行假数据,就足够定位大多数问题。">贴上代码和报错信息,再说明表里有哪些列、各是什么类型</button>
<button class="quiz-opt" data-why="截图里同样是真实病人的逐例数据,和上传文件没有区别。">不发文件,只把前 20 行数据截图发过去</button>
</div>

## 4. 从头到尾走一遍

下面的每一步,都可以在网页里运行,也可以抄进你自己电脑上的笔记本里运行。

### 4.1 读入和检查

在自己的电脑上,读的是 `data/` 里的 Excel:

```python
#| norun
import pandas as pd

df = pd.read_excel("data/cohort.xlsx", sheet_name="patients")
fu = pd.read_excel("data/cohort.xlsx", sheet_name="followup")
```

网页里读不到你电脑上的文件,读的是内容相同的两个 CSV:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("cohort.csv")
fu = pd.read_csv("followup.csv")
print(df.shape, fu.shape)
```

数据到手,**先检查,再分析**。每一份新数据都问这几个问题:

```python
print("病人编号有没有重复:", not df["id"].is_unique, "| 随访表:", not fu["id"].is_unique)
print("每一列缺了几个:")
print(df.isna().sum()[lambda s: s > 0])
print(df["arm"].value_counts().to_dict(), df["stage"].value_counts().to_dict())
df[["age", "height_cm", "weight_kg", "diameter", "n_fx"]].describe().loc[["min", "max"]]
```

- **编号有没有重复**: 一位病人被录了两次,是最常见的错误之一。
- **缺了多少**: 三列壁厚各缺 4 个。`[lambda s: s > 0]` 只留下缺失数大于 0 的列。
- **分类变量有哪些取值**: 有没有同一类写成两种样子的。
- **数值的范围合不合理**: 年龄 40 到 87 岁,身高 143 到 175 cm,施源器直径 25 到 35 mm,近距离 2 或 3 次,都在合理的范围里。

### 4.2 清理和合并

第 2 课已经发现了这份表的问题: `wall_range` 那一列是 Excel 公式算的,没有壁厚测量的 4 位病人被算成了 0。改回缺失,加上 BMI,再把随访表合进来:

```python
df.loc[df["wall_min"].isna(), "wall_range"] = np.nan          # Excel 公式留下的假 0
df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2

data = df.merge(fu, on="id", how="left", validate="one_to_one")
print(data.shape, "|", data["tox"].isna().sum(), "位病人没有随访记录")
```

`validate="one_to_one"` 是给合并上的保险: 如果哪一张表里同一个编号出现了两次,它会直接报错,而不是悄悄把行复制一份(第 3 课)。合并后还是 200 行,12 位病人没有随访记录。

清理的每一步都是一行代码,旁边写着为什么。半年以后审稿人问"那 4 位病人是怎么处理的",答案就在这里。

<div class="exercise">

**练习 1 · 轮到你: 数一数能用的病人**

后面要分析壁厚和副作用的关系,只能用**既有壁厚测量、又有随访记录**的病人。有几位?存进 `n_both`。

```python
has_wall = data["wall_mean"].notna()
has_fu = data["tox"].___()
n_both = (has_wall ___ has_fu).sum()
print(f"有壁厚测量 {has_wall.sum()} 位,有随访 {has_fu.sum()} 位,两样都有 {n_both} 位")
```

```pycheck
need("n_both")
want = int((data["wall_mean"].notna() & data["tox"].notna()).sum())
either = int((data["wall_mean"].notna() | data["tox"].notna()).sum())
if n_both == either and either != want:
    fail('这是"至少有一样"的人数。两样都有,用 &。')
if n_both == want:
    ok(f"对了: {want} 位。196 位有壁厚测量,188 位有随访,两样都有的是 {want} 位。每做一个分析,都要知道它用了多少位病人,并写进论文。")
fail(f"现在是 {n_both},应该是 {want}。第一处填 notna,第二处填 &。")
```

<details><summary>提示</summary>

"不是缺失"是 `notna()`;"两个条件都成立"是 `&`。

</details>

<details><summary>参考答案</summary>

```python
has_wall = data["wall_mean"].notna()
has_fu = data["tox"].notna()
n_both = (has_wall & has_fu).sum()
print(f"有壁厚测量 {has_wall.sum()} 位,有随访 {has_fu.sum()} 位,两样都有 {n_both} 位")
```

</details>

</div>

### 4.3 把前几课写过的函数拿过来

前几课写过四个小函数。在网页里,每一课的代码是分开的,所以这里再写一遍;在自己的电脑上,可以把它们存进一个文件 `my_functions.py`,以后每个项目里写一句 `from my_functions import median_iqr, boot_ci` 就能用,和导入别人的包是一回事。

```python
def median_iqr(x, digits=1):
    """一列数 -> '中位数 (Q1–Q3)'。第 3 课。"""
    q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
    return f"{med:.{digits}f} ({q1:.{digits}f}–{q3:.{digits}f})"

def n_pct(flag):
    """一列 True/False -> '人数 (百分比)'。第 3 课。"""
    return f"{flag.sum()} ({flag.mean():.1%})"

def boot_ci(values, stat=np.median, n_boot=2000, seed=1):
    """bootstrap 的 95% 区间。第 5 课。"""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boot = np.array([stat(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)])
    return np.percentile(boot, [2.5, 97.5])

def holm(p_values):
    """Holm 校正。第 6 课。"""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    largest = 0.0
    for rank, i in enumerate(np.argsort(p)):
        largest = max(largest, (m - rank) * p[i])
        adjusted[i] = min(1.0, largest)
    return adjusted

print(median_iqr(data["age"], 0), "|", n_pct(data["arm"] == "VCB"))
```

### 4.4 Table 1

```python
def table1_column(d):
    followed = d["tox"].dropna()                       # 只有有随访的病人才能算副作用的比例
    return pd.Series({
        "Patients, n": len(d),
        "Age, years": median_iqr(d["age"], 0),
        "BMI, kg/m2": median_iqr(d["bmi"]),
        "Stage IA, n (%)": n_pct(d["stage"] == "IA"),
        "Stage IB, n (%)": n_pct(d["stage"] == "IB"),
        "Stage II, n (%)": n_pct(d["stage"] == "II"),
        "Mean wall thickness, mm": median_iqr(d["wall_mean"]),
        "Vaginal D2cc, 5 mm plan, Gy": median_iqr(d["d2cc_5mm"]),
        "Follow-up available, n (%)": n_pct(d["tox"].notna()),
        "Late vaginal toxicity, n (%)": n_pct(followed == 1),
    })

table1 = pd.DataFrame({
    "All": table1_column(data),
    "VCB alone": table1_column(data[data["arm"] == "VCB"]),
    "EBRT+VCB": table1_column(data[data["arm"] == "EBRT+VCB"]),
})
table1
```

和第 3 课相比多了两行随访的内容,注意**分母**: 副作用的比例是在有随访的病人里算的(`followed`),所以全部病人那一列是 42 / 188 = 22.3%。如果直接写 `n_pct(d["tox"] == 1)`,分母会变成 200,得到 21.0%: 12 位没有随访的病人被悄悄当成了"没有副作用"。

### 4.5 主要结果: 三套计划两两比较

研究的主要问题: 同一位病人,三套计划的阴道 D2cc 差多少?三个两两比较,每个都报告: 差值的中位数、bootstrap 的 95% 区间(第 5 课)、Wilcoxon 符号秩检验的 P 值(第 6 课),三个 P 值再做 Holm 校正。

把它写成一个函数,要比较的指标作为参数:

```python
def compare_plans(d, metric):
    """三套计划两两比较。metric 是指标的名字,比如 'd2cc'。"""
    pairs = [("vol", "5mm", "Volume − 5 mm"), ("gra", "5mm", "Graphical − 5 mm"), ("gra", "vol", "Graphical − Volume")]
    rows = []
    for a, b, label in pairs:
        x, y = d[f"{metric}_{a}"], d[f"{metric}_{b}"]          # 比如 d["d2cc_vol"] 和 d["d2cc_5mm"]
        low, high = boot_ci(x - y)
        rows.append({"comparison": label, "median": (x - y).median(), "low": low, "high": high,
                     "p": stats.wilcoxon(x, y).pvalue})
    out = pd.DataFrame(rows).set_index("comparison")
    out["p_holm"] = holm(out["p"])
    return out

results = compare_plans(data, "d2cc")
results
```

`f"{metric}_{a}"` 用 f 字符串拼出列名: `metric` 是 `"d2cc"`、`a` 是 `"vol"` 时,拼出来就是 `"d2cc_vol"`。每一对算好以后是一个字典,攒成一张表;`set_index("comparison")` 把比较的名字当成行名。

读这张表: 体积优化比 5 mm 低 2.0 Gy(95% CI −2.4 到 −1.5),图形优化比 5 mm 低 3.4 Gy(−4.3 到 −2.8),图形优化比体积优化又低 1.45 Gy(−2.1 到 −0.9)。三个区间都不包含 0,校正后的 P 值都远小于 0.001。

<div class="exercise">

**练习 2 · 轮到你: D90 的三个比较**

对靶区 D90 做同样的三个比较,结果存进 `results_d90`。

```python
results_d90 = compare_plans(data, ___)
results_d90
```

```pycheck
need("results_d90")
try:
    med = [float(results_d90.loc[k, "median"]) for k in ["Volume − 5 mm", "Graphical − 5 mm", "Graphical − Volume"]]
except Exception:
    fail('results_d90 应该是 compare_plans 交回的那张表: compare_plans(data, "d90")。')
if close(med[0], -2.0, 0.01):
    fail('这是 D2cc 的结果。指标的名字填 "d90"。')
if close(med[0], -1.45, 0.01) and close(med[1], -2.45, 0.01) and close(med[2], -1.10, 0.01):
    ok("对了: 靶区 D90 也跟着降了,体积优化比 5 mm 低 1.45 Gy(−1.8 到 −1.15),图形优化低 2.45 Gy(−2.7 到 −1.85)。阴道的剂量降下来是好处,靶区的剂量降下来是代价: 这就是这类研究要权衡的两头。换一个指标只改了一个词,这是把分析写成函数的好处。")
fail(f'现在三个中位数是 {med},应该是 −1.45、−2.45、−1.10。指标的名字填 "d90"(带引号)。')
```

<details><summary>提示</summary>

指标的名字是文字: `"d90"`。

</details>

<details><summary>参考答案</summary>

```python
results_d90 = compare_plans(data, "d90")
results_d90
```

</details>

</div>

### 4.6 和壁厚的关系

第二个问题: 阴道壁越厚,换计划带来的变化是不是越小?用 Spearman 相关(第 6 课),只用有壁厚测量的病人:

```python
d = data.dropna(subset=["wall_mean"])
for plan, name in [("vol", "Volume"), ("gra", "Graphical")]:
    res = stats.spearmanr(d["wall_mean"], d[f"d2cc_{plan}"] - d["d2cc_5mm"])
    print(f"{name} − 5 mm 和平均壁厚: Spearman ρ = {res.statistic:.2f},P = {res.pvalue:.1g},n = {len(d)}")
```

两个相关系数是 0.59 和 0.53: 壁越厚,D2cc 降得越少。

<div class="exercise">

**练习 3 · 轮到你: D90 的变化和壁厚**

图形优化计划的 D90 的变化(`d90_gra` 减 `d90_5mm`)和平均壁厚的 Spearman 相关系数是多少?存进 `rho_d90`。

```python
res_d90 = stats.spearmanr(d["wall_mean"], d[___] - d[___])
rho_d90 = res_d90.statistic
print(f"Spearman ρ = {rho_d90:.2f},n = {len(d)}")
```

```pycheck
need("rho_d90")
if rho_d90 != rho_d90:
    fail("结果是 NaN: 用去掉了缺失值的 d。")
if close(rho_d90, -0.6373, 0.001):
    fail("大小对了,符号反了: 变化是新计划减 5 mm,d90_gra 在前。")
if close(rho_d90, 0.6995, 0.001):
    fail('这是体积优化的结果: 第一处填 "d90_gra"。')
if close(rho_d90, 0.6373, 0.001):
    ok("对了: ρ = 0.64(n = 196)。壁越厚,靶区 D90 降得越少,比 D2cc 的相关(0.53)还强一些。")
fail(f"现在 rho_d90 = {rho_d90:.3f},应该是 0.637。")
```

<details><summary>提示</summary>

两处依次填 `"d90_gra"` 和 `"d90_5mm"`。

</details>

<details><summary>参考答案</summary>

```python
res_d90 = stats.spearmanr(d["wall_mean"], d["d90_gra"] - d["d90_5mm"])
rho_d90 = res_d90.statistic
print(f"Spearman ρ = {rho_d90:.2f},n = {len(d)}")
```

</details>

</div>

### 4.7 一张图,存成文件

```python
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for arm, g in d.groupby("arm"):
    ax.scatter(g["wall_mean"], g["d2cc_vol"] - g["d2cc_5mm"], s=8, alpha=0.6, label=arm)
ax.axhline(0, color="grey", linewidth=1)
ax.set_xlabel("Mean vaginal wall thickness (mm)")
ax.set_ylabel("Change in D2cc, Volume − 5 mm (Gy)")
ax.legend(title="Treatment", fontsize=7, title_fontsize=7)
fig.savefig("figure1.png", dpi=300, bbox_inches="tight")
plt.show()
```

表也存下来。在自己的电脑上,文件名前面加上 `outputs/`,它们就都进了输出的文件夹:

```python
table1.to_csv("table1.csv")
results.to_csv("results_d2cc.csv")
print(pd.read_csv("results_d2cc.csv", index_col=0)[["median", "low", "high"]])
```

存成 Excel 是 `table1.to_excel("outputs/table1.xlsx")`,从那里贴进 Word 再调格式。

<div class="exercise">

**练习 4 · 轮到你: D90 的图**

画图形优化的 D90 变化和平均壁厚的散点图(两种治疗方式不同颜色),存成 300 dpi 的 `figure_d90.png`。

```python
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for arm, g in d.groupby("arm"):
    ax.scatter(g["wall_mean"], g[___] - g["d90_5mm"], s=8, alpha=0.6, label=arm)
ax.axhline(0, color="grey", linewidth=1)
ax.set_xlabel("Mean vaginal wall thickness (mm)")
ax.set_ylabel("Change in D90, Graphical − 5 mm (Gy)")
ax.legend(title="Treatment", fontsize=7, title_fontsize=7)
fig.savefig(___, dpi=300, bbox_inches="tight")
plt.show()
```

```pycheck
import os
need("ax")
if not os.path.exists("figure_d90.png"):
    fail('还没有存出 figure_d90.png: 文件名填 "figure_d90.png"。')
xy = np.vstack([np.asarray(c.get_offsets()) for c in ax.collections])
want = (d["d90_gra"] - d["d90_5mm"]).sum()
if close(np.nansum(xy[:, 1]), (d["d90_vol"] - d["d90_5mm"]).sum(), 0.5):
    fail('纵轴画的是体积优化: 第一处填 "d90_gra"。')
if not close(np.nansum(xy[:, 1]), want, 0.5):
    fail('纵轴应该是 d90_gra 减 d90_5mm: 第一处填 "d90_gra"。')
ok("对了。图和文件都有了: 196 位病人,壁越厚,D90 降得越少。")
```

<details><summary>提示</summary>

第一处是列名 `"d90_gra"`,第二处是文件名 `"figure_d90.png"`。

</details>

<details><summary>参考答案</summary>

```python
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for arm, g in d.groupby("arm"):
    ax.scatter(g["wall_mean"], g["d90_gra"] - g["d90_5mm"], s=8, alpha=0.6, label=arm)
ax.axhline(0, color="grey", linewidth=1)
ax.set_xlabel("Mean vaginal wall thickness (mm)")
ax.set_ylabel("Change in D90, Graphical − 5 mm (Gy)")
ax.legend(title="Treatment", fontsize=7, title_fontsize=7)
fig.savefig("figure_d90.png", dpi=300, bbox_inches="tight")
plt.show()
```

</details>

</div>

### 4.8 让代码来写结果里的句子

论文里出错最多的地方,是从表格往正文里**手抄数字**: 数据更新了,表改了,正文里的数忘了改。办法是让代码把数填进句子:

```python
def p_text(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"

r = results.loc["Volume − 5 mm"]
sentence = (f"Vaginal D2cc was lower with volume-based optimisation than with the 5 mm prescription "
            f"(median difference {r['median']:.2f} Gy, 95% CI {r['low']:.2f} to {r['high']:.2f}; P {p_text(r['p_holm'])}).")
print(sentence)
```

`results.loc["Volume − 5 mm"]` 取出结果表的一行,`r['median']` 是这一行里的中位数。句子里的每个数都直接来自结果表,不可能和表对不上。

<div class="exercise">

**练习 5 · 轮到你: D90 的句子**

用练习 2 的 `results_d90`,写出图形优化和 5 mm 比较的那一句,存进 `sentence_d90`。

```python
r = results_d90.loc[___]
sentence_d90 = (f"CTV D90 was lower with graphical optimisation than with the 5 mm prescription "
                f"(median difference {r['median']:.2f} Gy, 95% CI {r[___]:.2f} to {r['high']:.2f}; P {p_text(r['p_holm'])}).")
print(sentence_d90)
```

```pycheck
need("sentence_d90")
s = str(sentence_d90)
if "-1.45" in s:
    fail('这是体积优化那一行。第一处填 "Graphical − 5 mm"。')
if "-2.45" in s and "-2.70" in s and "-1.85" in s and "< 0.001" in s:
    ok("对了: median difference −2.45 Gy, 95% CI −2.70 to −1.85; P < 0.001。从原始的表,到 Table 1、结果、图、正文里的句子,全部由代码生成。数据如果更新,重新运行一遍,这些就都更新了。Python 课到这里结束。")
fail('句子里应该有 -2.45、-2.70、-1.85 和 < 0.001。第一处填行名 "Graphical − 5 mm",第二处填 "low"。')
```

<details><summary>提示</summary>

行名要和结果表里的一模一样: `"Graphical − 5 mm"`(中间是一个长的减号,可以从上面的代码里复制);区间的下限那一列叫 `"low"`。

</details>

<details><summary>参考答案</summary>

```python
r = results_d90.loc["Graphical − 5 mm"]
sentence_d90 = (f"CTV D90 was lower with graphical optimisation than with the 5 mm prescription "
                f"(median difference {r['median']:.2f} Gy, 95% CI {r['low']:.2f} to {r['high']:.2f}; P {p_text(r['p_holm'])}).")
print(sentence_d90)
```

</details>

</div>

## 5. 用你自己的数据

如果[主题练习](/python/practice/)还没有做完,先把它做完: 那是一份陌生的数据(宫颈癌近距离放疗的剂量审计),没有填空,六部分正好对应前六课。

然后是真正的毕业任务: 把上面的流程用在一份你自己的数据上。照这张清单做:

1. **建文件夹**: `data/`、`outputs/` 和一个笔记本。原始的表放进 `data/`,先确认里面没有姓名和病历号。
2. **读入和检查**: 行数和列数对不对;每列的类型;编号有没有重复;每列缺了多少;分类变量有哪些取值;数值的范围合不合理。
3. **清理**: 每一处改动写一行代码,旁边写上原因。不在 Excel 里手工改。
4. **Table 1**: 先想好按什么分组、列哪些变量,连续变量用中位数(四分位间距)还是均值(标准差)。
5. **主要结果**: 一个事先想好的问题,对应一个估计值、一个 95% 区间、一个 P 值。比较了几次,就校正几次。
6. **图**: 一张图讲一件事,能画出每位病人就画出来,轴标签带单位。
7. **存结果**: 表存成文件,图存成 300 dpi 的文件,正文里的数由代码填。
8. **从头运行一遍**: 在 JupyterLab 的菜单里选 "Kernel → Restart Kernel and Run All Cells"。全部重新运行一遍,结果不变,才算做完。

第 8 步最容易被省掉,却最重要。笔记本里的格子可以不按顺序运行,改来改去以后,屏幕上的结果可能已经不是现在这份代码算出来的了。从头运行一遍,才能保证别人(包括半年后的你)拿到同样的数据和代码,能得到同样的结果。这叫**可重复**(reproducible)。

## 小结

- 在自己的电脑上: 安装 Python,用 `pip` 装包,用 JupyterLab 写和运行代码。数据不离开这台电脑。
- 一个项目三块: 原始数据(不改)、代码、输出(随时可以重新生成)。
- 病人数据: 去标识;不上传、不外发、不贴给 AI;代码可以共享,数据不可以。
- 流程: 读入 → 检查 → 清理 → Table 1 → 主要结果(估计值、区间、P 值)→ 图 → 存文件 → 从头再运行一遍。
- 把重复的分析写成函数,换一个指标只改一个词;正文里的数让代码来填。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 终端 | terminal | 用打字来下命令的窗口 |
| 笔记本 | notebook | 由格子组成的文件,代码和结果在一起 |
| 工作表 | sheet | Excel 文件里的一张表 |
| 去标识 | de-identification | 去掉能认出病人的信息 |
| 原始数据 | raw data | 收到时的那份数据,不再改动 |
| 可重复 | reproducible | 同样的数据和代码,得到同样的结果 |
| 分析流程 | analysis pipeline | 从原始数据到结果的全部步骤 |

## 这一课没讲什么

- **版本管理**(git): 记录代码每一次改动的工具,多人合作和长期项目很有用。
- **环境管理**: 把每个项目用的包和版本固定下来,换一台电脑也能得到同样的结果。至少把 `pip freeze` 的输出存一份。
- **预测模型**: 用病人治疗前的信息,预测换计划以后会怎样。这是下一门课的内容。

## 写进论文 Writing it up

<div class="paper">

方法的最后一段,交代软件和可重复性(版本号换成你自己的,用 `pd.__version__` 这样的写法查):

> All analyses were performed in Python 3.13 with pandas 2.3, SciPy 1.16 and Matplotlib 3.10. Tables, figures and the numbers quoted in the text were generated by scripts from the de-identified dataset; the analysis code is available from the corresponding author on request.

**审稿人可能问**: 能不能提供数据和代码?

**怎么回答**: 代码可以提供。病人级的数据能不能共享,取决于伦理批准和知情同意的范围;不能共享时如实说明原因和申请的途径。

**审稿人可能问**: 正文、表格和图里的数字对不上。

**怎么回答**: 这正是让代码生成表、图和句子要避免的问题。如果发生了,从头运行一遍,把所有输出一起更新。

</div>

## 常见问题 Troubleshooting

| 现象 | 原因和办法 |
|---|---|
| 终端里说找不到 `py` 或 `python3` | Python 没有装好。Windows 上重新运行安装程序;macOS 上确认装的是 python.org 的安装包。 |
| `pip` 下载很慢或中断 | 换镜像: 命令最后加上 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。 |
| `ModuleNotFoundError: No module named 'pandas'` | 这个包还没有装,或者装到了另一个 Python 里。用第 1 节第三步的那一行再装一次。 |
| `FileNotFoundError: data/cohort.xlsx` | 笔记本和 `data/` 不在同一个文件夹里,或者文件名不对。JupyterLab 左边的文件列表里应该同时看得到笔记本和 `data/`。 |
| `ImportError: Missing optional dependency 'openpyxl'` | 读写 Excel 需要 `openpyxl`,用 `pip` 装上。 |
| 重新打开笔记本,变量都没有了 | 笔记本只保存代码和显示过的结果,不保存变量。从头运行一遍。 |

## 延伸阅读 Further reading

- [JupyterLab 的官方文档](https://jupyterlab.readthedocs.io/en/stable/): 界面和快捷键的说明。
- Wilson G, et al. [Good enough practices in scientific computing](https://doi.org/10.1371/journal.pcbi.1005510). *PLoS Computational Biology* 2017;13:e1005510。给不是程序员的研究者的一份实用清单: 数据怎么放、代码怎么写、项目怎么组织。

**接下来**: Python 课到这里结束。你已经能把一份数据从 Excel 做到 Table 1、主要结果和图。这些都是在**描述**已经发生的事。下一步是**预测**: 一位新病人,还没有做优化的计划,能不能先知道换了计划会怎样?这是下一门课[放疗副作用的预测模型](/stat-ml/)(统计机器学习课)要讲的。
