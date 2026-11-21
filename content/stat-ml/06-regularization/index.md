---
title: "Lesson 6 · 正则化 Regularization"
summary: "岭回归、LASSO、弹性网: 给系数加上约束,变量多、患者少时预测更稳。"
weight: 60
toc: true
duration: "约 70 分钟,可以分两次学: 第 0–3 节,第 4–6 节"
prereq: "本课程[第 5 课](../05-data-leakage/)(Pipeline 和嵌套交叉验证)"
questions:
  - "候选变量很多、患者不多时,最小二乘错在哪里?"
  - '给系数加"约束"是什么意思?岭回归和 LASSO 有什么不同?'
  - "约束的强度怎么定?"
objectives:
  - "说明变量多时最小二乘为什么对新患者不准"
  - "用 Ridge 和 Lasso 拟合带约束的模型,看懂系数是怎样被收缩的"
  - "用 GridSearchCV 在交叉验证里选约束的强度,并用嵌套交叉验证报告成绩"
  - "知道 LASSO 挑出来的变量不稳定,不能当作结论"
---

第 5 课里,多放进模型的变量(随机数、分期)都让它在新患者身上变差了一点。原因在最小二乘本身: 你给它多少个变量,它就给每个变量配一个系数,让建模患者的误差尽量小,不管这个变量有没有用。变量一多,凑出来的"规律"里偶然的成分就多。

临床研究常常是这种处境: 两百位患者,几十个候选变量(剂量学参数、解剖测量、病理、合并症)。按 P 值筛变量是一种办法,第 5 课说了它的毛病。这一课讲另一种: 变量都留着,但给系数的大小加一个**约束**,不让它们随便长。这叫**正则化**(regularization)。

> 🔬 逻辑回归的 `C=1e6`(第 2 课)关掉的就是这个约束。这一课讲它为什么默认是开着的。

**练习怎么做**: 讲解用 **Volume** 计划的 Δ,练习换成 **Graphical**。

## 0. 准备: 3 个有用的变量,加 40 个没用的

```python
import numpy as np
import pandas as pd
from cohort import make_cohort
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def prepare(d):
    d = d.dropna().copy()
    d["delta"] = d["d2cc_vol"] - d["d2cc_5mm"]
    d["delta_gra"] = d["d2cc_gra"] - d["d2cc_5mm"]
    return d

df, new = prepare(make_cohort()), prepare(make_cohort(n=5000, seed=99))
real = ["wall_mean", "diameter", "n_fx"]                 # 真的有用的 3 个变量
noise = [f"z{i}" for i in range(40)]                     # 40 个随机数,冒充"别的测量值"
rng = np.random.default_rng(6)
D = pd.concat([df, pd.DataFrame(rng.normal(size=(len(df), 40)), columns=noise, index=df.index)], axis=1)
N = pd.concat([new, pd.DataFrame(rng.normal(size=(len(new), 40)), columns=noise, index=new.index)], axis=1)
many = real + noise                                      # 43 个候选变量
cv = KFold(n_splits=5, shuffle=True, random_state=0)

def cv_mae(model, columns, outcome="delta"):
    return -cross_val_score(model, D[columns], D[outcome], cv=cv, scoring="neg_mean_absolute_error").mean()

def new_mae(model, columns, outcome="delta"):
    model.fit(D[columns], D[outcome])
    return (N[outcome] - model.predict(N[columns])).abs().mean()

print(len(D), "位患者,", len(many), "个候选变量")
```

`D` 是建模的 196 位患者,`N` 是 4900 位新患者(只用来对答案)。候选变量有 43 个: 壁厚、直径、次数,加上 40 个随机数。真实研究里你不知道哪些变量没用;这里知道,是为了看清各种办法做对了多少。

## 1. 变量一多,最小二乘怎么了

```python
ols = make_pipeline(StandardScaler(), LinearRegression())
print(f"3 个变量:  交叉验证 {cv_mae(ols, real):.2f} Gy,新患者 {new_mae(ols, real):.2f} Gy")
print(f"43 个变量: 交叉验证 {cv_mae(ols, many):.2f} Gy,新患者 {new_mae(ols, many):.2f} Gy")

ols.fit(D[many], D["delta"])
coef = pd.Series(ols[-1].coef_, index=many)
print("三个真变量的系数:", coef[real].round(2).tolist())
print(f"40 个随机数的系数,绝对值平均 {coef[noise].abs().mean():.2f}")
```

只用 3 个真变量,新患者身上平均差 1.73 Gy。43 个变量一起上,是 1.97 Gy: 多了 40 个没用的变量,预测反而差了 0.24 Gy。

看系数就明白了。变量都标准化过(每个变量减均值、除以标准差),所以系数可以直接比大小: 三个真变量是 1.62、0.61、−0.68;40 个随机数本该是 0,最小二乘却给了它们平均 0.15 的系数。每一个都不大,40 个加在一起,就在每位新患者的预测值里掺进了不小的噪声。

## 2. 它在算什么: 给系数加约束

<div class="frame">

**背景**: 最小二乘只管让建模患者的误差平方和最小,系数多大都行。没用的变量也拿到了系数。

**目标**: 让模型"没有把握就少说": 一个变量,除非它确实能让误差明显变小,否则系数应该接近 0。

**做法**: 在误差平方和后面加一项**罚分**,系数越大罚得越多。拟合时两项一起最小化。

</div>

**它在算什么**

1. 先把每个变量标准化。不然,罚分会跟着单位变: 用 mm 记的壁厚和用 cm 记的壁厚,系数差 10 倍,受的罚也不一样。
2. 最小二乘要最小化的是: 误差平方和。
3. 现在改成最小化: 误差平方和 + λ × (系数大小的总和)。λ(lambda)是约束的强度: λ = 0 就是最小二乘;λ 越大,系数越要往 0 靠。
4. "系数大小的总和"有两种算法,得到两种模型:
   - **岭回归**(ridge): 把每个系数平方,再相加。
   - **LASSO**: 把每个系数取绝对值,再相加。

模型现在面对一笔交易: 给某个变量一个系数,误差平方和能降一点,但罚分要涨一点。有用的变量降得多,值得;没用的变量降得少,不值得,系数就被压下去。

**两种罚法的区别**: 平方的罚法(岭回归)对小系数很宽容,0.1 的平方只有 0.01,所以系数会变小,但不会变成 0。绝对值的罚法(LASSO)对小系数也照价收费,不值得的系数会被直接压到 0: 这个变量就从模型里消失了。

**打个比方**: 查房时汇报病情,每多说一项检查结果,就多占一点时间。岭回归像是要求"每一项都说得简短些";LASSO 像是要求"不重要的干脆不提"。

<div class="demo">

**设定**: 40 位模拟患者,12 个标准化过的变量: 壁厚和次数真的有用,另外 10 个是随机数。每根横条是一个变量的系数。两个按钮选约束的种类,滑块是约束的强度。

**你做什么**: 先选 LASSO,把强度从 0 慢慢拖到最大,看随机数的横条和下面的误差。再点"岭回归",重新拖一遍。

**你会看到**: 不加约束时,10 个随机数都有系数,新患者身上的误差是 2.04 Gy。LASSO: 强度 4 时只剩 5 个系数不是 0,误差降到 1.80 Gy;拖到最强只剩 1 个,误差回到 1.95 Gy。岭回归: 所有横条一起缩短,但没有一根变成 0,误差最低是 1.89 Gy(强度 3)。

**这说明**: 约束把没用的系数压了下去,预测更准。约束太强,有用的系数也被压住,预测又变差。强度需要选。

<div class="widget" data-widget="shrink">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

为什么平方的罚法压不到 0,绝对值的罚法能?把问题缩小到两个系数,就能画出来。下面这张图用 40 位模拟患者和两个标准化过的变量: 壁厚(真的有用)和一个随机数(没用)。点"下一步",图往前走一步;点 ↻ 把这一步重放一遍。

<div class="widget stage" data-widget="budget">

<div class="stage-step" data-label="两个系数">
<p class="stage-h">一张地图: 每个点是一个模型</p>

横轴是壁厚的系数,纵轴是随机数的系数。平面上每一个点都是一个模型。

原点是两个系数都为 0 的模型: 对谁都猜平均值,在 2000 位新患者身上平均差 1.96 Gy。

</div>

<div class="stage-step" data-label="碗底">
<p class="stage-h">最小二乘: 走到碗底</p>

把建模患者身上的误差想成地形。最小二乘走到最低的地方(红圈): 壁厚的系数 1.76,随机数的系数 0.53。一圈一圈的线是等高线: 同一圈上的模型,在建模患者身上的误差相同,越往外越大。

随机数本来没用,却拿到了 0.53: 这是 40 位患者里的偶然。新患者身上平均差 1.62 Gy。

</div>

<div class="stage-step" data-label="岭回归">
<p class="stage-h">岭回归: 罚分是一个圆</p>

岭回归的罚分是两个系数的平方和: 离原点越远,罚得越多,罚分相同的模型排成一个圆。现在要同时顾两头: 误差想留在碗底,罚分想回到原点。约束越强,答案离碗底越远;它总落在误差的一圈和罚分的圆刚好相切的地方。

看下面两根条: 两个系数一起变短,但随机数的系数始终不是 0。alpha = 14.3 时新患者身上最好,1.58 Gy;到 400,两个系数只剩 0.16 和 0.06,误差回到 1.88 Gy。放完以后可以自己拖。

</div>

<div class="stage-step" data-label="LASSO">
<p class="stage-h">LASSO: 罚分是一个菱形</p>

LASSO 的罚分是两个系数的绝对值之和,罚分相同的模型排成一个菱形,角在坐标轴上。约束加强时,答案在 alpha = 0.56 时落到菱形的角上: 随机数的系数正好是 0,只剩壁厚的 1.23。之后它贴着横轴往原点走。

虚线是刚才岭回归走过的路: 它弯向原点,但不碰坐标轴。

</div>

<div class="stage-step" data-label="角">
<p class="stage-h">为什么能正好是 0: 因为有角</p>

这是随机数的系数刚变成 0 的那一刻。误差的等高线碰到菱形的地方是一个角,而角在轴上。

圆没有角。等高线和圆相切的地方几乎不会正好在轴上,所以岭回归的系数只会变小,不会变成 0。可以拖滑块前后看: 到了角上以后,答案就贴着轴走。

</div>

<div class="stage-step" data-label="多强">
<p class="stage-h">约束多强合适</p>

图下面是新患者身上的误差随约束强度的变化: 实线是 LASSO,虚线是岭回归。不加约束 1.62 Gy;LASSO 在 alpha = 0.42 时最低,1.56 Gy;再强,有用的壁厚也被压住,误差升回 1.96 Gy。

两个变量时好处不大。变量一多(上面那张图里的 12 个,正文里的 43 个),差别才明显。真实数据里没有"新患者"可查,强度要用交叉验证来选(第 4 节)。

</div>

</div>

## 3. 岭回归和 LASSO

在课程的队列上试。scikit-learn 里约束的强度叫 `alpha`:

```python
ridge = make_pipeline(StandardScaler(), Ridge(alpha=30))
print(f"岭回归(alpha=30): 交叉验证 {cv_mae(ridge, many):.2f} Gy,新患者 {new_mae(ridge, many):.2f} Gy")
c = pd.Series(ridge[-1].coef_, index=many)
print("  真变量的系数:", c[real].round(2).tolist(), f";随机数的系数,绝对值平均 {c[noise].abs().mean():.2f};不是 0 的有 {(c != 0).sum()} 个")

lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.2))
print(f"LASSO(alpha=0.2): 交叉验证 {cv_mae(lasso, many):.2f} Gy,新患者 {new_mae(lasso, many):.2f} Gy")
c = pd.Series(lasso[-1].coef_, index=many)
print("  真变量的系数:", c[real].round(2).tolist(), f";不是 0 的有 {(c != 0).sum()} 个")
```

- **岭回归**: 新患者身上 1.91 Gy,比最小二乘的 1.97 Gy 好一点。43 个系数都还在,只是都缩小了: 真变量从 1.62、0.61、−0.68 缩到 1.40、0.51、−0.57,随机数的平均从 0.15 缩到 0.12。
- **LASSO**: 新患者身上 1.78 Gy,接近只用 3 个真变量的 1.73 Gy。43 个系数里只剩 11 个不是 0: 3 个真变量都留下了,40 个随机数被压掉了 32 个。

留意两件事。第一,真变量的系数也被缩小了(壁厚从 1.62 到 1.50): 约束不分好坏,一律收费。带约束的系数是有偏的,不能再读成"壁厚每厚 1 个标准差,Δ 变 1.50 Gy"。第 3 课的话在这里更要紧: 这是预测模型,系数不是用来解释的。第二,LASSO 留下的 11 个里还有 8 个随机数: 它压掉了大部分,不是全部。

<div class="quiz">
<p><b>想一想</b>: LASSO 留下了 11 个变量。能不能在论文里写"LASSO 筛选出 11 个与 Δ 相关的因素"?</p>
<button class="quiz-opt" data-why="这 11 个里有 8 个是随机数,和 Δ 毫无关系。LASSO 留下一个变量,只说明在这批患者、这个强度下,给它一个系数划得来。">能: LASSO 留下的就是有关的</button>
<button class="quiz-opt" data-correct data-why="留下的 11 个里 8 个是随机数。换一批患者,或者把强度调一点,留下来的名单就会变(把 alpha 改成 0.5 再运行,只剩 4 个)。LASSO 是为了预测得准,名单只是副产品,不稳定,不能当作'发现了哪些因素'的证据。">不能: 留下谁取决于这批患者和约束的强度,名单不稳定</button>
<button class="quiz-opt" data-why="压掉变量正是 LASSO 的设计,不是出错。问题在于留下的名单不能读成'有关的因素'。">不能: 模型不应该丢掉变量</button>
</div>

<div class="exercise">

**练习 1 · 轮到你: Graphical,变量多了以后**

对 Graphical 的 Δ,算最小二乘用 3 个变量和 43 个变量时的交叉验证误差,存进 `cv_real_g`、`cv_many_g`。

```python
cv_real_g = cv_mae(ols, real, "delta_gra")
cv_many_g = cv_mae(ols, ___, ___)
print(f"3 个变量 {cv_real_g:.2f} Gy,43 个变量 {cv_many_g:.2f} Gy")
```

```pycheck
need("cv_real_g", "cv_many_g")
if close(cv_many_g, 1.827, 0.004):
    fail('这是 Volume 的结果。第二个空填 "delta_gra"。')
if close(cv_many_g, 2.454, 0.004):
    fail("两个数一样: 第一个空要给 43 个变量的名单 many。")
if close(cv_real_g, 2.454, 0.004) and close(cv_many_g, 2.687, 0.004):
    ok("对了: 2.45 Gy 和 2.69 Gy。多放 40 个没用的变量,交叉验证的误差大了 0.23 Gy。")
fail(f"现在是 {cv_real_g:.3f} 和 {cv_many_g:.3f},应该是 2.454 和 2.687。")
```

<details><summary>提示</summary>

第一个空是 43 个变量的名单 `many`,第二个空是结局 `"delta_gra"`。

</details>

<details><summary>参考答案</summary>

```python
cv_real_g = cv_mae(ols, real, "delta_gra")
cv_many_g = cv_mae(ols, many, "delta_gra")
print(f"3 个变量 {cv_real_g:.2f} Gy,43 个变量 {cv_many_g:.2f} Gy")
```

</details>

</div>

<div class="exercise">

**练习 2 · 轮到你: Graphical 的 LASSO 留下了谁**

对 Graphical 的 Δ,拟合 `alpha=0.5` 的 LASSO(43 个变量),数一数有几个系数不是 0(`n_kept`),并把它们的名字存进 `kept`。

```python
lasso_g = make_pipeline(StandardScaler(), Lasso(alpha=___))
lasso_g.fit(D[many], D["delta_gra"])
c_g = pd.Series(lasso_g[-1].coef_, index=many)
kept = c_g[c_g != ___].index.tolist()
n_kept = len(kept)
print(n_kept, kept)
```

```pycheck
need("n_kept", "kept")
if n_kept == 4:
    fail('4 个是 Volume 的结果。fit 的结局应该是 D["delta_gra"](起始代码里已经是),检查有没有改动。')
if n_kept == 18:
    fail("18 个是 alpha=0.2 的结果。题目要的是 alpha=0.5。")
if n_kept == 3 and set(kept) == {"wall_mean", "n_fx", "z22"}:
    ok("对了: 只剩 3 个: 壁厚、次数,还有一个随机数 z22。直径被压掉了: 第 3 课看到过,Graphical 里直径本来就不起作用。LASSO 做对了两件事,做错了一件(z22)。")
fail(f"现在留下了 {n_kept} 个,应该是 3 个。第一个空是 0.5,第二个空是 0。")
```

<details><summary>提示</summary>

第一个空是约束的强度 `0.5`;第二个空是 `0`: 留下系数不等于 0 的那些。

</details>

<details><summary>参考答案</summary>

```python
lasso_g = make_pipeline(StandardScaler(), Lasso(alpha=0.5))
lasso_g.fit(D[many], D["delta_gra"])
c_g = pd.Series(lasso_g[-1].coef_, index=many)
kept = c_g[c_g != 0].index.tolist()
n_kept = len(kept)
print(n_kept, kept)
```

</details>

</div>

## 4. 约束的强度怎么选

上面的 `alpha=30` 和 `alpha=0.2` 不是凭空来的。强度不同,结果差很多:

```python
for alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
    m = make_pipeline(StandardScaler(), Lasso(alpha=alpha))
    kept_n = (m.fit(D[many], D["delta"])[-1].coef_ != 0).sum()
    print(f"alpha = {alpha:<4}: 留下 {kept_n:2d} 个变量,交叉验证 {cv_mae(m, many):.2f} Gy")
```

alpha 从 0.05 到 1.0,留下的变量从 32 个减到 1 个。交叉验证的误差先降后升: 0.2 时最低(1.70 Gy),到 1.0 时连直径和次数也被压掉了,只剩壁厚(1.94 Gy)。

选强度的办法,就是上面这张表: 每个候选的强度做一次交叉验证,取误差最小的。`GridSearchCV` 把这件事自动化了:

```python
grid = {"lasso__alpha": [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]}        # Pipeline 里 Lasso 这一步的 alpha
search = GridSearchCV(make_pipeline(StandardScaler(), Lasso()), grid, cv=cv, scoring="neg_mean_absolute_error")
search.fit(D[many], D["delta"])
print("选中的 alpha:", search.best_params_["lasso__alpha"], f";它的交叉验证误差 {-search.best_score_:.2f} Gy")
print(f"新患者: {(N['delta'] - search.predict(N[many])).abs().mean():.2f} Gy")
```

选中的是 0.2,新患者身上 1.78 Gy。`search` 拟合完以后本身就是一个模型: 它用选中的强度在全部建模患者上重新拟合过,可以直接 `predict`。

**这里有第 5 课说过的那个问题**: 1.70 Gy 是"从 6 个候选里挑出来的最好成绩",偏乐观(新患者身上是 1.78 Gy)。要报告成绩,把整个 `search` 当作一个模型,再套一层交叉验证,也就是嵌套交叉验证:

```python
outer = KFold(n_splits=5, shuffle=True, random_state=1)
nested = -cross_val_score(search, D[many], D["delta"], cv=outer, scoring="neg_mean_absolute_error").mean()
print(f"嵌套交叉验证: {nested:.2f} Gy")
```

1.76 Gy,比 1.70 更接近新患者身上的 1.78。外层每一轮里,`search` 都只用留下来的患者重新选一次强度;被拿出来的患者既没参与拟合,也没参与选强度。

<div class="exercise">

**练习 3 · 轮到你: Graphical,选强度并对答案**

对 Graphical 的 Δ,用同样的 `search` 选 LASSO 的强度。把选中的强度存进 `alpha_g`,新患者身上的误差存进 `new_g`。

```python
search_g = GridSearchCV(make_pipeline(StandardScaler(), Lasso()), grid, cv=cv, scoring="neg_mean_absolute_error")
search_g.fit(D[many], D[___])
alpha_g = search_g.best_params_[___]
new_g = (N["delta_gra"] - search_g.predict(N[many])).abs().mean()
print(f"选中 alpha = {alpha_g},新患者 {new_g:.2f} Gy")
```

```pycheck
need("alpha_g", "new_g")
if close(new_g, 1.778, 0.004):
    fail('这是 Volume 的结果。第一个空填 "delta_gra"。')
if close(alpha_g, 0.2, 1e-9) and close(new_g, 2.615, 0.004):
    ok("对了: 选中 0.2,新患者身上 2.61 Gy。最小二乘用 43 个变量是 2.87 Gy,只用 3 个真变量是 2.55 Gy: LASSO 把多放 40 个随机数的损失挽回了大半,但没有全部挽回。")
fail(f'现在是 alpha = {alpha_g}、新患者 {new_g:.3f},应该是 0.2 和 2.615。第二个空是 "lasso__alpha"。')
```

<details><summary>提示</summary>

第一个空是 `"delta_gra"`;第二个空是 `grid` 里用的那个名字 `"lasso__alpha"`(中间是两条下划线)。

</details>

<details><summary>参考答案</summary>

```python
search_g = GridSearchCV(make_pipeline(StandardScaler(), Lasso()), grid, cv=cv, scoring="neg_mean_absolute_error")
search_g.fit(D[many], D["delta_gra"])
alpha_g = search_g.best_params_["lasso__alpha"]
new_g = (N["delta_gra"] - search_g.predict(N[many])).abs().mean()
print(f"选中 alpha = {alpha_g},新患者 {new_g:.2f} Gy")
```

</details>

</div>

## 5. 弹性网,和逻辑回归里的约束

**弹性网**(elastic net)把两种罚法各用一部分: 既压掉一些变量,又让留下的系数缩小。几个变量高度相关时(比如最小、平均、最大壁厚),LASSO 往往随便留一个;弹性网倾向于把它们一起留下、各给一点。scikit-learn 里是 `ElasticNet(alpha=..., l1_ratio=0.5)`,`l1_ratio` 是 LASSO 那部分占的比例。用法和上面完全一样。

**逻辑回归**同样可以加约束,而且 scikit-learn 默认就加了(岭回归那种)。强度用 `C` 表示,方向相反: `C` 越小,约束越强。第 2 课写 `C=1e6`,是把约束松到可以忽略,为的是得到能读成 OR 的系数。做预测、变量又多的时候,约束是有好处的: 把 `C` 当作要选的参数,用 `GridSearchCV` 选。

<div class="quiz">
<p><b>想一想</b>: 论文想同时回答两个问题: "哪些因素和结局有关(报告 OR)"和"能不能预测"。该用带约束的模型,还是不带的?</p>
<button class="quiz-opt" data-why="带约束的系数被有意压小了,由它算出的 OR 偏向 1,而且没有通常的置信区间。解释的问题不适合用它。">都用带约束的: 预测更准</button>
<button class="quiz-opt" data-correct data-why="两个问题用两个分析(第 3 课): 解释,用不带约束的回归,事先按临床知识定变量,报告 OR 和区间;预测,用带约束的模型,强度在交叉验证里选,报告新患者身上的成绩。论文里分开写,不要拿预测模型的系数去回答第一个问题。">分开做: 解释用不带约束的,预测用带约束的</button>
<button class="quiz-opt" data-why="不带约束的模型在变量多时对新患者不准(第 1 节)。预测的问题用它会吃亏。">都用不带约束的: 结果好解读</button>
</div>

## 6. 什么时候值得用

回头看这一课的几个数(Volume,新患者身上的平均误差):

| 做法 | 误差 |
|---|---|
| 最小二乘,3 个真变量(事先就知道谁有用) | 1.73 Gy |
| 最小二乘,43 个变量 | 1.97 Gy |
| 岭回归,43 个变量 | 1.91 Gy |
| LASSO,43 个变量,强度由交叉验证选 | 1.78 Gy |

正则化没有创造信息。最好的结果仍然是"事先就知道该用哪 3 个变量": 临床知识比任何算法都管用。正则化的用处在于: 当你不确定哪些变量有用、又不想按 P 值去筛的时候,它让模型自己把没把握的系数压下去,损失比什么都不做小得多。

变量不多、患者够多的时候(比如 3 个变量、200 位患者),加不加约束几乎没有差别。候选变量和患者数相比不算少的时候,它才显出用处: 多少算"不算少",第 13 课讲样本量时会有具体的算法。

## 小结

- 变量多的时候,最小二乘给没用的变量也配了系数,对新患者的预测变差。
- **正则化**: 在误差平方和后面加一项罚分,系数越大罚得越多。**岭回归**把系数一起缩小;**LASSO** 把不值得的系数直接压成 0。
- 用之前要**标准化**,而且标准化要放在 Pipeline 里。
- 约束的**强度**用交叉验证选(`GridSearchCV`);报告成绩时再套一层(嵌套交叉验证)。
- 带约束的系数是有偏的,LASSO 留下的变量名单也不稳定: 都不能当作解释的证据。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 正则化 | regularization, penalization | 给系数的大小加约束 |
| 罚分项 | penalty | 加在误差平方和后面、随系数变大的那一项 |
| 岭回归 | ridge regression (L2) | 罚系数的平方和;系数缩小但不为 0 |
| LASSO | lasso (L1) | 罚系数的绝对值之和;一部分系数变成 0 |
| 弹性网 | elastic net | 两种罚法各用一部分 |
| 收缩 | shrinkage | 系数被压向 0 |
| 超参数 | hyperparameter | 拟合之前要定下来的设置,比如约束的强度 |
| 网格搜索 | grid search | 把候选的设置逐个试一遍,用交叉验证挑 |

## 这一课没讲什么

- **约束的强度为什么叫 alpha、λ、C**: 只是不同软件的叫法。R 的 glmnet 叫 lambda,scikit-learn 的线性模型叫 alpha,逻辑回归用 C(方向相反)。
- **一个标准误规则**(1-SE rule): 有人不取误差最小的强度,而取"误差在最小值的一个标准误以内、约束最强"的那个,模型更简单。
- **带约束模型的置信区间**: 没有简单的算法,通常不报告。
- **分组的变量**(比如一个多类别变量拆成的几个 0/1 变量)应该一起留或一起去: 要用 group lasso,这门课不讲。

## 写进论文 Writing it up

<div class="paper">

方法部分可以这样写(数字换成你实际用的):

> Because the number of candidate predictors was large relative to the sample size, models were fitted with lasso penalisation after standardising all predictors. The penalty strength was chosen by 5-fold cross-validation from a prespecified grid, and performance was estimated by nested cross-validation, with standardisation and tuning repeated within each outer training fold. Coefficients of the penalised model are reported for transparency and should not be interpreted as effect estimates.

**审稿人可能问**: 为什么不用逐步回归或者单因素筛选来选变量?

**怎么回答**: 这两种办法的选择过程不稳定,而且如果不放进交叉验证,会让成绩偏乐观(第 5 课)。正则化把"选变量"和"拟合"合成一步,可以整个放进交叉验证。

**审稿人可能问**: LASSO 选出的变量有临床意义吗?

**怎么回答**: 不把选出的名单当作发现。可以补一个稳定性分析(bootstrap 重复多次,报告每个变量被选中的比例),并说明模型的目的是预测。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| `ConvergenceWarning: Objective did not converge` | LASSO 逼近的步数不够,多见于 alpha 很小的时候。加 `max_iter=10000`,或者去掉太小的 alpha。 |
| `ValueError: Invalid parameter 'alpha' for estimator Pipeline` | 在 Pipeline 里,参数名要带上步骤的名字: `lasso__alpha`(两条下划线)。 |
| LASSO 把所有系数都压成了 0 | alpha 太大。把候选的强度往小里放。 |
| 没有标准化,结果很奇怪 | 约束对单位敏感。把 `StandardScaler` 放在模型前面。 |
| 结果和网页上写的不一样 | 检查 `default_rng(6)` 和两处 `random_state`;代码块要从上往下依次运行。 |

## 延伸阅读 Further reading

- scikit-learn 用户指南: [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)。Ridge、Lasso、Elastic-Net 各有一节。
- Tibshirani R. Regression shrinkage and selection via the lasso. *J R Stat Soc Series B* 1996;58(1):267–288。提出 LASSO 的论文。

**下一课**: 到现在为止,模型的公式都是"系数 × 变量,加起来"。如果壁厚的作用在两种治疗方式里不一样(第 1 课末尾提过,确实不一样),直线模型看不出来,除非你事先想到并写进公式。第 7 课讲树模型: 它不用公式,而是问一串是/否的问题,这类规律它自己能找到。
