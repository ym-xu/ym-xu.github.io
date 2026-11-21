---
title: "Lesson 5 · 数据泄漏 Data Leakage"
summary: "Pipeline;同一病人的记录放在同一折;挑模型和报成绩用同一份交叉验证会偏乐观。"
weight: 50
toc: true
duration: "约 80 分钟,可以分两次学: 第 0–3 节,第 4–6 节"
prereq: "本课程[第 4 课](../04-cross-validation/)(交叉验证)"
questions:
  - "交叉验证已经做了,成绩为什么还可能偏乐观?"
  - "补缺失值、标准化、挑变量,哪些步骤要放进交叉验证?怎么放?"
  - "一位病人有好几条记录时,折该怎么分?"
  - "试了很多模型,挑了最好的一个,报告出来的成绩还算数吗?"
objectives:
  - "说出什么是数据泄漏,判断一个步骤会不会造成泄漏"
  - "用 Pipeline 把补缺失值、标准化和建模绑在一起做交叉验证"
  - "同一位病人有多条记录时,用 GroupKFold"
  - '说明"从很多模型里挑最好的"以后成绩为什么偏乐观,嵌套交叉验证做的是什么'
---

第 4 课的交叉验证有一个承诺: 每一轮被拿出来的那份病人,模型事先一点也没有见过。成绩可信,全靠这一条。

这个承诺很容易在不知不觉中被打破。第 4 课自己就走了两处捷径: 先用 `dropna()` 处理了缺失值,变量也是事先定好的。真实研究里,补缺失值、标准化、挑变量、挑模型,这些步骤只要有一步用到了被拿出来的病人,模型就等于偷看了考卷。这叫**数据泄漏**(data leakage)。泄漏之后,交叉验证照样能跑,照样给出一个数,只是这个数偏好,而且没有任何报错提醒你。

> 🔬 泄漏是预测模型论文里最常见的硬伤。成绩好得不合常理时,先怀疑泄漏,再相信模型。

**练习怎么做**: 讲解用**体积优化**计划的 Δ,练习换成**图形优化**。

## 0. 准备

```python
import numpy as np
import pandas as pd
from cohort import make_cohort
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, cross_val_score

def prepare(d, drop=True):
    d = d.dropna().copy() if drop else d.copy()
    d["delta"] = d["d2cc_vol"] - d["d2cc_5mm"]
    d["delta_gra"] = d["d2cc_gra"] - d["d2cc_5mm"]
    return d

df = prepare(make_cohort())                    # 去掉缺壁厚的 4 位,196 位病人
full = prepare(make_cohort(), drop=False)      # 全部 200 位,壁厚有缺失
new = prepare(make_cohort(n=5000, seed=99))    # 新病人,只用来对答案
cv = KFold(n_splits=5, shuffle=True, random_state=0)

def cv_mae(model, X, y, cv=cv, groups=None):
    """交叉验证的平均绝对误差(Gy)。"""
    scores = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="neg_mean_absolute_error")
    return -scores.mean()

print(len(df), len(full), len(new))
```

`cv_mae` 把第 4 课的做法包成了一个函数: 给它模型、变量和结局,它做 5 折交叉验证,交回平均绝对误差。`DummyRegressor` 是基线模型: 对谁都猜建模病人的平均值。

## 1. 最典型的泄漏: 先挑变量,再交叉验证

先看一个极端的例子,它把泄漏的后果放大到一眼能看出来。

<div class="demo">

**设定**: 60 位模拟病人。结局是随机数,候选的变量也全是随机数,和结局毫无关系: 这里没有任何东西可以预测。从候选变量里挑出和结局相关最强的 5 个建模,用 5 折交叉验证打分。三根条: 都猜平均值;先用全部病人挑好变量,再做交叉验证;把"挑变量"也放进交叉验证的每一轮里做。

**你做什么**: 把滑块从 10 个候选变量拖到 1000 个,看中间那根条。

**你会看到**: 都猜平均值是 1.77 Gy。先挑变量再验证: 10 个候选时是 1.82 Gy,1000 个候选时降到 1.40 Gy,好像模型很有用。挑变量也放进验证的那一根,始终不比 1.77 Gy 小。

**这说明**: 候选越多,越容易碰巧挑到和结局"相关"的随机数。用全部病人来挑,被拿出来的病人也参与了挑选,交叉验证就查不出这是碰巧。

<div class="widget" data-widget="leak">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

在课程的队列上重做一遍。给 196 位病人各造 300 个随机数当候选变量,目标是预测 Δ:

```python
rng = np.random.default_rng(0)
noise = pd.DataFrame(rng.normal(size=(len(df), 300)), columns=[f"z{i}" for i in range(300)], index=df.index)
y = df["delta"]

corr = noise.apply(lambda col: abs(np.corrcoef(col, y)[0, 1]))     # 每个候选变量和 Δ 的相关系数(用了全部病人)
top10 = corr.sort_values(ascending=False).index[:10]              # 挑相关最强的 10 个
print(f"都猜平均值:             {cv_mae(DummyRegressor(), noise, y):.2f} Gy")
print(f"先挑 10 个变量,再交叉验证: {cv_mae(LinearRegression(), noise[top10], y):.2f} Gy")
```

都猜平均值,2.25 Gy。用挑出来的 10 个随机数建模,交叉验证是 2.08 Gy,比基线"准"了 0.17 Gy。随机数不可能预测 Δ,这 0.17 Gy 完全是泄漏造出来的。

<div class="frame">

**背景**: 挑变量时算相关系数,用的是全部 196 位病人,其中包括每一轮被拿出来的那 39 位。

**目标**: 让被拿出来的病人在整个建模过程中都没被用过,包括挑变量这一步。

**做法**: 把"挑变量"和"拟合模型"绑成一个整体,叫 **Pipeline**(流水线),整个交给交叉验证。每一轮里,挑变量只用留下来建模的那 4 份病人。

</div>

**它在算什么**(每一轮)

1. 拿出 1 份病人,放在一边。
2. 只用剩下的 4 份,算每个候选变量和 Δ 的相关,挑出 10 个。
3. 还是只用这 4 份,拟合模型。
4. 用这个模型预测被拿出来的那 1 份,算误差。

和上面的区别只在第 2 步: 挑选不再偷看被拿出来的病人。

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression

honest = make_pipeline(SelectKBest(f_regression, k=10), LinearRegression())    # 挑 10 个 → 线性回归,绑在一起
print(f"挑变量放进交叉验证: {cv_mae(honest, noise, y):.2f} Gy")
```

2.49 Gy,比都猜平均值(2.25 Gy)还差。这才是实情: 10 个随机数帮不上忙,它们的系数只会添乱。

**打个比方**: 出一套模拟题,先看了全班同学的答卷,把大家普遍做对的题挑出来组成考卷,再拿这张考卷去考其中几位同学。他们的分数当然高,但不能说明他们学得好。

<div class="exercise">

**练习 1 · 轮到你: 图形优化,同样的陷阱**

换成图形优化的 Δ。先算泄漏的版本(`leaky_g`): 用全部病人挑 10 个随机变量,再交叉验证。再算诚实的版本(`honest_g`): 把挑变量放进 Pipeline。

```python
yg = df["delta_gra"]
corr_g = noise.apply(lambda col: abs(np.corrcoef(col, yg)[0, 1]))
top10_g = corr_g.sort_values(ascending=False).index[:10]
leaky_g = cv_mae(LinearRegression(), noise[___], yg)
honest_g = cv_mae(make_pipeline(SelectKBest(f_regression, k=10), ___), noise, yg)
print(f"基线 {cv_mae(DummyRegressor(), noise, yg):.2f},泄漏 {leaky_g:.2f},诚实 {honest_g:.2f} Gy")
```

```pycheck
need("leaky_g", "honest_g")
if close(leaky_g, 2.083, 0.006):
    fail("这是体积优化的结果。挑变量和交叉验证都要用 yg(起始代码里已经是),检查第一个空是不是 top10_g。")
if close(leaky_g, 2.770, 0.006) and close(honest_g, 3.331, 0.006):
    ok("对了: 基线 3.01 Gy,泄漏的版本 2.77 Gy,诚实的版本 3.33 Gy。同一批随机数,泄漏让它们看起来比基线好 0.24 Gy;诚实地验证,它们比基线差。")
fail(f"现在是 {leaky_g:.2f} 和 {honest_g:.2f},应该是 2.77 和 3.33。第一个空是 top10_g,第二个空是 LinearRegression()。")
```

<details><summary>提示</summary>

第一个空是刚挑出来的 10 个列名 `top10_g`;第二个空是流水线的最后一步,模型 `LinearRegression()`。

</details>

<details><summary>参考答案</summary>

```python
yg = df["delta_gra"]
corr_g = noise.apply(lambda col: abs(np.corrcoef(col, yg)[0, 1]))
top10_g = corr_g.sort_values(ascending=False).index[:10]
leaky_g = cv_mae(LinearRegression(), noise[top10_g], yg)
honest_g = cv_mae(make_pipeline(SelectKBest(f_regression, k=10), LinearRegression()), noise, yg)
print(f"基线 {cv_mae(DummyRegressor(), noise, yg):.2f},泄漏 {leaky_g:.2f},诚实 {honest_g:.2f} Gy")
```

</details>

</div>

## 2. 哪些步骤要放进交叉验证

规则只有一条: **凡是要从别的病人那里"学"出一个数的步骤,都要放进交叉验证**。

- 要放进去的: 用中位数或均值补缺失值(中位数是从病人里算的);标准化(均值和标准差是从病人里算的);按相关系数或 P 值挑变量;挑模型、调参数。
- 不用放的: 只用这位病人自己的数就能做的换算。两套计划相减得到 Δ,把"EBRT+VCB"写成 1,把 Gy 换成 cGy,都属于这一类。

<div class="quiz">
<p><b>想一想</b>: 做交叉验证之前,先对全部病人做了单因素分析,只把 P &lt; 0.1 的变量留下来建模。这样有没有泄漏?</p>
<button class="quiz-opt" data-why="单因素分析的 P 值是用全部病人算的,其中包括每一轮被拿出来的病人。这和第 1 节'先挑变量'是同一回事,只是把相关系数换成了 P 值。">没有: 单因素分析和建模是两个步骤</button>
<button class="quiz-opt" data-correct data-why="按 P 值筛变量,用到了全部病人的结局,包括后面被拿出来的那些。这是论文里最常见的泄漏。筛变量要放进交叉验证的每一轮,或者干脆不按 P 值筛(第 6 课)。">有: 筛变量用到了被拿出来的病人的结局</button>
<button class="quiz-opt" data-why="变量少一些并不能抵消泄漏。问题不在留下几个,而在'留下谁'这个决定用到了全部病人。">没有: 筛过以后变量更少,模型更稳</button>
</div>

## 3. 用 Pipeline 补缺失值和标准化

队列有 200 位病人,其中 4 位没有壁厚测量。前几课直接把他们去掉了。更好的做法是把缺的值补上: 最简单的补法是用其他病人的中位数。"其他病人的中位数"是从数据里学出来的数,按上面的规则,要放进交叉验证。

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

cols = ["wall_mean", "diameter", "n_fx"]
pipe = make_pipeline(SimpleImputer(strategy="median"),     # 第 1 步: 用中位数补缺失值
                     StandardScaler(),                     # 第 2 步: 标准化(减均值,除以标准差)
                     LinearRegression())                   # 第 3 步: 线性回归
print(f"200 位病人,Pipeline: {cv_mae(pipe, full[cols], full['delta']):.2f} Gy")

train_idx, test_idx = next(iter(cv.split(full)))
print(f"壁厚的中位数: 全部病人 {full['wall_mean'].median():.2f} mm,第一轮留下来建模的病人 {full['wall_mean'].iloc[train_idx].median():.2f} mm")
```

交叉验证的平均误差是 1.74 Gy,200 位病人都用上了。最后一行让你看到"放进交叉验证"具体是什么意思: 全部病人的壁厚中位数是 3.50 mm,第一轮留下来建模的 160 位是 3.60 mm。Pipeline 在这一轮里用 3.60 去补,包括补被拿出来的那 40 位里缺的值;它不知道、也不该知道 3.50。

**它在算什么**(每一轮): 从留下的 4 份里算中位数,补上缺失值;从这 4 份里算均值和标准差,做标准化;拟合模型。然后对被拿出来的 1 份,用**同样的**中位数、均值和标准差处理,再预测。

**标准化**(standardization)对线性回归的预测没有影响,这里先放上,是因为下一课的模型需要它: 加了约束的模型,变量的单位不同会受到不同的约束。

老实说,这个例子里泄漏的后果小到看不出来: 先用全部病人的中位数补好,再交叉验证,结果也是 1.74 Gy,因为只缺 4 个值。养成用 Pipeline 的习惯,是为了后果大的那些情况(第 1 节),也是为了审稿人问起来时不用解释。

多类别的变量,比如分期(IA、IB、II),要拆成几个 0/1 变量才能放进公式。这一步只看病人自己的值,不算泄漏,可以先做:

```python
X_stage = pd.get_dummies(full[cols + ["stage"]], columns=["stage"], drop_first=True, dtype=int)
print(X_stage.columns.tolist())
print(f"加上分期: {cv_mae(pipe, X_stage, full['delta']):.2f} Gy")
```

`get_dummies` 把分期拆成 `stage_IB` 和 `stage_II` 两列(IA 是对照,所以少一列)。加上分期,误差是 1.77 Gy,比不加(1.74 Gy)还大了一点: 分期和 Δ 没有关系(第 3 课),多两个系数只是多了噪声。

<div class="exercise">

**练习 2 · 轮到你: 图形优化的 Pipeline**

用同样的三步 Pipeline,在全部 200 位病人上预测图形优化的 Δ,把交叉验证的平均误差存进 `mae_pipe_g`。

```python
pipe_g = make_pipeline(SimpleImputer(strategy=___), StandardScaler(), LinearRegression())
mae_pipe_g = cv_mae(pipe_g, full[cols], full[___])
print(f"{mae_pipe_g:.2f} Gy")
```

```pycheck
need("mae_pipe_g")
if close(mae_pipe_g, 1.738, 0.004):
    fail('这是体积优化的结果。第二个空填 "delta_gra"。')
if close(mae_pipe_g, 2.454, 0.004):
    fail("这是去掉缺失病人以后(196 位)的结果。题目要用全部 200 位: full[cols]。")
if close(mae_pipe_g, 2.483, 0.004):
    ok("对了: 2.48 Gy,200 位病人都用上了。第 3 课在新病人身上量到的是 2.55 Gy,交叉验证给出的数和它很接近。")
fail(f'现在是 {mae_pipe_g:.3f},应该是 2.483。第一个空是 "median",第二个空是 "delta_gra"。')
```

<details><summary>提示</summary>

第一个空是补法的名字 `"median"`,第二个空是结局的列名 `"delta_gra"`。都带引号。

</details>

<details><summary>参考答案</summary>

```python
pipe_g = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
mae_pipe_g = cv_mae(pipe_g, full[cols], full["delta_gra"])
print(f"{mae_pipe_g:.2f} Gy")
```

</details>

</div>

## 4. 一位病人好几条记录

另一种泄漏和步骤无关,出在"怎么分折"上。

每位病人有两套优化计划: 体积优化和图形优化。如果把它们当成两条记录,一条预测体积优化的 Δ,一条预测图形优化的 Δ,数据就从 196 行变成了 392 行,看起来样本翻了一倍。

```python
long = pd.concat([df.assign(plan="vol", y=df["delta"]),
                  df.assign(plan="gra", y=df["delta_gra"])], ignore_index=True)
print(len(long), "行,", long["id"].nunique(), "位病人")
long[["id", "plan", "wall_mean", "y"]].sort_values("id").head(4)
```

同一位病人的两行,壁厚、直径、年龄完全相同,两个 Δ 也相关(同一个人的解剖决定了两套计划都好降或都难降)。随机分折时,一位病人的两行很可能一行在建模的那边,一行在被拿出来的那边。模型预测被拿出来的那一行时,其实见过这位病人。

用一个"会记人"的模型来看后果。**决策树**(decision tree,第 7 课专门讲)不拟合公式: 它把各项数值相近的病人归成一小组一小组,预测一位病人时,看这位病人落在哪一组,用同组病人结局的平均值。下面让每一组最少可以只有 2 行。

<div class="quiz">
<p><b>先猜一猜</b>: 随机分折时,被拿出来的某一行,最可能和谁落在同一个小组里?</p>
<button class="quiz-opt" data-why="壁厚相近的别的病人确实可能同组,但有一行一定同组: 各项数值一模一样的那一行。运行下面的代码看后果。">壁厚最接近的另一位病人</button>
<button class="quiz-opt" data-correct data-why="同一位病人的另一行,各项数值完全相同,一定落在同一组。模型给出的预测里,就有这位病人另一套计划的 Δ,而两套计划的 Δ 是相关的。运行下面的代码,看这让成绩好了多少。">同一位病人的另一行(另一套计划)</button>
<button class="quiz-opt" data-why="分组看的是数值像不像,不是随机的。同一位病人的另一行,数值一模一样。">随机的一位</button>
</div>

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GroupKFold

feat = ["wall_mean", "wall_min", "wall_max", "diameter", "n_fx", "age"]
tree = DecisionTreeRegressor(min_samples_leaf=2, random_state=0)
by_patient = GroupKFold(n_splits=5, shuffle=True, random_state=0)       # 同一位病人的行永远在同一份里
print(f"都猜平均值:   {cv_mae(DummyRegressor(), long[feat], long['y']):.2f} Gy")
print(f"随机分折:     {cv_mae(tree, long[feat], long['y']):.2f} Gy")
print(f"按病人分折:   {cv_mae(tree, long[feat], long['y'], cv=by_patient, groups=long['id']):.2f} Gy")
```

都猜平均值是 2.70 Gy。随机分折,这棵树是 2.54 Gy,好像比基线强。**按病人分折**(同一位病人的两行永远在同一份里),是 2.99 Gy,比基线差。后一个数才是它对新病人的水平: 新病人来的时候,建模的数据里没有"这位病人的另一行"可以参考。

`GroupKFold` 做的就是按病人分折: `groups` 告诉它每一行属于哪位病人,它保证同一位病人的所有行在同一份里。

<div class="quiz">
<p><b>想一想</b>: 下面哪种数据要按病人分折?</p>
<button class="quiz-opt" data-why="每位病人只有一行时,普通的 KFold 就够了。">每位病人一行: 基线特征和一个结局</button>
<button class="quiz-opt" data-correct data-why="只要同一位病人贡献了不止一行,就要按病人分: 多次近距离治疗各算一行、双侧器官各算一行、同一病人的多个病灶、多次随访,都是这样。分的单位是'病人',不是'行'。">每位病人的 3 次近距离治疗各算一行</button>
<button class="quiz-opt" data-why="两家医院的病人互不相同,每位病人仍然只有一行。(如果想知道模型换一家医院还准不准,可以按医院分折,那是另一个问题。)">来自两家医院的病人,每人一行</button>
</div>

<div class="exercise">

**练习 3 · 轮到你: 换成 D90**

对靶区的 D90 做同样的事: 每位病人两行,结局是两套优化计划各自的 D90 变化。算同一棵树在随机分折(`tree_random`)和按病人分折(`tree_group`)下的平均误差。

```python
long90 = pd.concat([df.assign(y=df["d90_vol"] - df["d90_5mm"]),
                    df.assign(y=df["d90_gra"] - df["d90_5mm"])], ignore_index=True)
tree_random = cv_mae(tree, long90[feat], long90["y"])
tree_group = cv_mae(tree, long90[feat], long90["y"], cv=by_patient, groups=long90[___])
print(f"随机分折 {tree_random:.2f} Gy,按病人分折 {tree_group:.2f} Gy")
```

```pycheck
need("tree_random", "tree_group")
if close(tree_group, 1.4816, 0.004):
    fail('两个数一样: 按病人分折没有生效。groups 要给每一行所属的病人: long90["id"]。')
if close(tree_random, 1.4816, 0.004) and close(tree_group, 1.5758, 0.004):
    ok("对了: 随机分折 1.48 Gy,按病人分折 1.58 Gy。同一个模型、同一份数据,只因为分折的办法不同,成绩差了 0.09 Gy。诚实的数是 1.58。")
fail(f'现在是 {tree_random:.3f} 和 {tree_group:.3f},应该是 1.482 和 1.576。空里填 "id"。')
```

<details><summary>提示</summary>

`groups` 要的是每一行属于哪位病人,也就是病人编号那一列: `"id"`。

</details>

<details><summary>参考答案</summary>

```python
long90 = pd.concat([df.assign(y=df["d90_vol"] - df["d90_5mm"]),
                    df.assign(y=df["d90_gra"] - df["d90_5mm"])], ignore_index=True)
tree_random = cv_mae(tree, long90[feat], long90["y"])
tree_group = cv_mae(tree, long90[feat], long90["y"], cv=by_patient, groups=long90["id"])
print(f"随机分折 {tree_random:.2f} Gy,按病人分折 {tree_group:.2f} Gy")
```

</details>

</div>

## 5. 挑了最好的模型以后

最后一种泄漏最隐蔽: 每个模型的交叉验证都做得规规矩矩,泄漏出在"试了很多个,报告最好的那一个"。

造 40 个候选模型: 每个都是"壁厚 + 一个随机数"。随机数没有用,所以 40 个模型的真实水平都和"只用壁厚"一样。

```python
rng = np.random.default_rng(3)
names = [f"z{i}" for i in range(40)]
D = pd.concat([df, pd.DataFrame(rng.normal(size=(len(df), 40)), columns=names, index=df.index)], axis=1)
N = pd.concat([new, pd.DataFrame(rng.normal(size=(len(new), 40)), columns=names, index=new.index)], axis=1)

def new_mae(columns, outcome="delta"):
    m = LinearRegression().fit(D[columns], D[outcome])
    return (N[outcome] - m.predict(N[columns])).abs().mean()

scores = pd.Series({z: cv_mae(LinearRegression(), D[["wall_mean", z]], D["delta"]) for z in names})
best = scores.idxmin()
print(f"只用壁厚: 交叉验证 {cv_mae(LinearRegression(), D[['wall_mean']], D['delta']):.2f} Gy,新病人 {new_mae(['wall_mean']):.2f} Gy")
print(f"40 个候选里最好的({best}): 交叉验证 {scores.min():.2f} Gy,新病人 {new_mae(['wall_mean', best]):.2f} Gy")
```

只用壁厚,交叉验证是 1.79 Gy,新病人身上是 1.82 Gy。40 个候选里交叉验证成绩最好的那个是 1.75 Gy,看起来比只用壁厚强;到新病人身上是 1.83 Gy,并不比只用壁厚好。

40 个模型的交叉验证成绩在 1.75 到 1.82 之间波动,全是运气。挑最好的,挑到的就是运气最好的那一个;把它的 1.75 报出来,等于把运气算成了本事。试的模型越多,这个便宜占得越大。挑模型、调参数,用的都是交叉验证的成绩,所以"挑"这一步也是从数据里学了东西,同样要算进去。

<div class="frame">

**背景**: 用同一份交叉验证,既挑了模型,又报了成绩。成绩偏乐观。

**目标**: 一个不偏的成绩,同时仍然允许挑模型。

**做法**: **嵌套交叉验证**(nested cross-validation)。外面一层只管打分,里面一层只管挑。

</div>

**它在算什么**

1. 外层: 拿出 1 份病人,放在一边。
2. 内层: 在剩下的 4 份里,**再做一次**完整的交叉验证,给 40 个候选打分,挑出最好的。
3. 用挑出来的那个,在这 4 份上拟合,预测外层拿出来的那 1 份。
4. 外层换一份,重复。每一轮挑出来的可以不是同一个模型。

打分的是"挑模型加拟合"这一整套流程,被拿出来的病人既没参与拟合,也没参与挑选。

```python
outer = KFold(n_splits=5, shuffle=True, random_state=1)
errors = []
for train_idx, test_idx in outer.split(D):
    train, test = D.iloc[train_idx], D.iloc[test_idx]
    inner = {z: cv_mae(LinearRegression(), train[["wall_mean", z]], train["delta"]) for z in names}   # 第 2 步
    pick = min(inner, key=inner.get)
    m = LinearRegression().fit(train[["wall_mean", pick]], train["delta"])                            # 第 3 步
    errors.append((test["delta"] - m.predict(test[["wall_mean", pick]])).abs().mean())
print(f"嵌套交叉验证: {np.mean(errors):.2f} Gy")
```

这一段要拟合一千多次模型,等几秒。嵌套交叉验证给出 1.84 Gy,和新病人身上的 1.83 Gy 对上了;直接报最好的那个 1.75 Gy 则偏了 0.08 Gy。

更省事的办法是不挑: 事先按临床知识把模型和变量定下来,写进方案,交叉验证只做一次。这门课后面的课大多这样做。

<div class="exercise">

**练习 4 · 轮到你: 图形优化,挑出来的那个有多乐观**

对图形优化的 Δ,给 40 个候选打分,挑出交叉验证成绩最好的(`best_g`),记下它的交叉验证成绩(`cv_best_g`)和新病人身上的成绩(`new_best_g`)。

```python
scores_g = pd.Series({z: cv_mae(LinearRegression(), D[["wall_mean", z]], D[___]) for z in names})
best_g = scores_g.___()
cv_best_g = scores_g.min()
new_best_g = new_mae(["wall_mean", best_g], "delta_gra")
print(f"{best_g}: 交叉验证 {cv_best_g:.2f} Gy,新病人 {new_best_g:.2f} Gy;只用壁厚在新病人身上 {new_mae(['wall_mean'], 'delta_gra'):.2f} Gy")
```

```pycheck
need("best_g", "cv_best_g", "new_best_g")
if close(cv_best_g, 1.755, 0.004):
    fail('这是体积优化的结果。第一个空填 "delta_gra"。')
if best_g != "z21" and close(cv_best_g, 2.470, 0.004):
    fail("成绩对了,但挑出来的不是成绩最好的那一个。误差越小越好: 第二个空是 idxmin。")
if best_g == "z21" and close(cv_best_g, 2.470, 0.004) and close(new_best_g, 2.627, 0.004):
    ok("对了: 挑出来的是 z21,交叉验证 2.47 Gy,新病人身上 2.63 Gy;只用壁厚在新病人身上是 2.60 Gy。交叉验证说它最好,到新病人身上它比什么都不加还差一点。")
fail(f'现在挑的是 {best_g},交叉验证 {cv_best_g:.3f},应该是 z21 和 2.470。第一个空是 "delta_gra",第二个空是 idxmin。')
```

<details><summary>提示</summary>

第一个空是 `"delta_gra"`。`scores_g` 里是每个候选的误差,最好的是误差最小的: 取最小值所在的名字用 `idxmin`。

</details>

<details><summary>参考答案</summary>

```python
scores_g = pd.Series({z: cv_mae(LinearRegression(), D[["wall_mean", z]], D["delta_gra"]) for z in names})
best_g = scores_g.idxmin()
cv_best_g = scores_g.min()
new_best_g = new_mae(["wall_mean", best_g], "delta_gra")
print(f"{best_g}: 交叉验证 {cv_best_g:.2f} Gy,新病人 {new_best_g:.2f} Gy;只用壁厚在新病人身上 {new_mae(['wall_mean'], 'delta_gra'):.2f} Gy")
```

</details>

</div>

## 6. 一张自查清单

写论文之前,对着交叉验证的每一轮问三个问题:

1. **被拿出来的病人,有没有在任何一步里被用过?** 补缺失值、标准化、挑变量,是不是都在 Pipeline 里。
2. **被拿出来的病人,有没有"亲属"留在建模的那边?** 同一位病人的其他记录,要在同一份里。
3. **报告的这个成绩,是不是从很多次尝试里挑出来的?** 是的话,用嵌套交叉验证,或者如实写明试过多少种。

## 小结

- **数据泄漏**: 被拿出来的病人以任何方式参与了建模。交叉验证不会报错,只会给出偏好的成绩。
- 凡是从别的病人那里"学"出数来的步骤(补缺失值、标准化、挑变量、挑模型),都要放进交叉验证。**Pipeline** 把这些步骤和模型绑在一起,自动做到这一点。
- 同一位病人有多条记录时,用 **GroupKFold** 按病人分折。
- 从很多模型里挑出最好的,它的交叉验证成绩偏乐观。用**嵌套交叉验证**,或者事先把模型定下来。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 数据泄漏 | data leakage | 被拿出来的病人以某种方式参与了建模 |
| 流水线 | pipeline | 把预处理和模型绑成一个整体,一起交叉验证 |
| 补缺失值 | imputation | 用一个估计的值填上缺失的值 |
| 标准化 | standardization | 减去均值、除以标准差 |
| 特征选择 | feature selection | 从候选变量里挑出一部分 |
| 决策树 | decision tree | 把数值相近的病人归成小组,用同组的平均结局来预测(第 7 课) |
| 分组交叉验证 | grouped cross-validation | 同一位病人的记录永远在同一份里 |
| 嵌套交叉验证 | nested cross-validation | 外层打分,内层挑模型 |
| 独热编码 | one-hot (dummy) encoding | 把多类别的变量拆成几个 0/1 变量 |

## 这一课没讲什么

- **更好的补缺失值的办法**: 多重插补(multiple imputation)比用中位数补更合理,尤其缺得多的时候。原则不变: 插补模型也要放进交叉验证。
- **时间上的泄漏**: 用治疗以后才知道的信息去预测治疗前要做的决定(比如用计划完成后的剂量预测要不要做这个计划)。这不是交叉验证能发现的,要靠临床判断: 每个预测变量,在需要做决定的那一刻是不是已经知道了?
- **调参数的工具**: `GridSearchCV` 把"试很多组参数、挑最好的"自动化了,它本身可以放进 `cross_val_score`,那就是嵌套交叉验证。第 6 课会用到。

## 写进论文 Writing it up

<div class="paper">

方法部分要让审稿人看得出没有泄漏:

> Missing wall-thickness values were imputed with the median, and predictors were standardised; both steps were estimated within each training fold and applied to the held-out fold. Predictors were specified in advance from clinical knowledge and no data-driven variable selection was performed. Records from the same patient were always assigned to the same fold.

**审稿人可能问**: 变量是怎么选出来的?是不是先做了单因素分析?

**怎么回答**: 如实写。如果变量是事先定的,写明依据;如果按数据筛过,筛选必须在交叉验证的每一轮里重做,并报告这样得到的成绩。

**审稿人可能问**: 比较了几种模型?报告的是哪一个的成绩?

**怎么回答**: 写明试过的所有模型和它们的成绩,不只报最好的;如果报告的是挑出来的那一个,成绩用嵌套交叉验证估计。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| `ValueError: Input X contains NaN` | 模型前面没有补缺失值的步骤。把 `SimpleImputer` 放进 Pipeline。 |
| `ValueError: The 'groups' parameter should not be None` | 用了 `GroupKFold` 但没给 `groups`。把病人编号那一列传给 `groups`。 |
| 交叉验证的成绩好得出奇 | 先查泄漏: 有没有在交叉验证之前用全部数据做过挑变量、标准化或补缺失值;同一位病人是不是出现在了不同的份里。 |
| 最后一段代码很慢 | 嵌套交叉验证要拟合一千多次模型,浏览器里要等几秒到十几秒。 |
| 结果和网页上写的不一样 | 随机数的种子(`default_rng(0)`、`default_rng(3)`)和 `random_state` 要和课文一致;代码块要从上往下依次运行。 |

## 延伸阅读 Further reading

- scikit-learn 用户指南: [Common pitfalls and recommended practices, Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)。官方文档里专门讲泄漏和 Pipeline 的一节。
- Kapoor S, Narayanan A. [Leakage and the reproducibility crisis in machine-learning-based science](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/). *Patterns (N Y)* 2023;4(9):100804。整理了各个学科里因为泄漏而站不住的机器学习论文,可以引用。

**下一课**: 这一课里,多放进去的变量(随机数、分期)都让模型在新病人身上变差了一点。变量多、病人少的时候,最小二乘会给每个变量都配一个系数,不管它有没有用。第 6 课讲正则化: 给系数加上约束,让没用的变量自己安静下来。

{{< practice >}}
