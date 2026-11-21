---
title: "练习 3 · 换模型,问它看重什么"
summary: "直线、树、森林比一比;发现大肿瘤里施源器才要紧,把它写成交互项;LASSO 和置换重要性。"
weight: 30
after: 8
toc: false
duration: "约 40 分钟"
prereq: "[统计机器学习课第 6–8 课](/stat-ml/);[练习 2](../02-honest-score/)"
questions:
  - "换成树和随机森林,靶区剂量能预测得更准吗?"
  - "森林比直线好在哪里?能不能把它看到的东西写回公式?"
  - "模型看重哪个变量?这能说明什么,不能说明什么?"
objectives:
  - "用同一种交叉验证比较几个模型"
  - "读一棵树,发现交互作用,写成交互项"
  - "LASSO 和置换重要性"
---

练习 2 的结论有点扫兴: 两个变量的直线,只比猜平均值好 0.3 Gy。这一部分换几种模型试试。

先运行这一块。靶区剂量的模型现在有三个变量(多了年龄)。准备好的还有: 交叉验证的分法和 `cv_mae`,随机森林的写法 `forest()`,第 8 课的 `importance_cv`,以及一张多了 10 列随机数的表 `D`。

```python
import numpy as np
import pandas as pd
from cervix import load_cervix
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

full = load_cervix()
full["needles"] = (full["technique"] == "IC/IS").astype(int)
df = full.dropna().copy()                                       # 肿瘤体积齐全的 195 位
y = df["d90_total"]
cols = ["hrctv_vol", "needles", "age"]
cv = KFold(n_splits=5, shuffle=True, random_state=0)

def cv_mae(model, X, y):
    """5 折交叉验证的平均绝对误差(Gy)。"""
    return -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error").mean()

def forest():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0)

def importance_cv(columns):
    """每一折: 用建模的病人拟合森林,在留出的病人身上把每个变量打乱 10 次;5 折取平均。"""
    total = np.zeros(len(columns))
    for train_idx, test_idx in cv.split(df):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        fitted = forest().fit(train[columns], train["d90_total"])
        result = permutation_importance(fitted, test[columns], test["d90_total"],
                                        scoring="neg_mean_absolute_error", n_repeats=10, random_state=0)
        total += result.importances_mean / 5
    return pd.Series(total, index=columns)

rng = np.random.default_rng(5)
names = [f"z{i}" for i in range(10)]                             # 10 列随机数,冒充"别的测量值"
D = pd.concat([df, pd.DataFrame(rng.normal(size=(len(df), 10)), columns=names, index=df.index)], axis=1)
many = cols + names                                              # 13 个候选变量

print(len(df), "位病人;猜平均值的交叉验证误差是 4.48 Gy")
```

<div class="steps" data-steps>

<div class="step exercise">

**第 1 步 · 三个模型,同一种考法**

用 `cols` 里的三个变量,算三个模型的交叉验证误差: 线性回归(`mae_lin`),一棵两层的树(`DecisionTreeRegressor(max_depth=2, random_state=0)`,存进 `mae_tree`),随机森林(`forest()`,存进 `mae_rf`)。森林要算几秒钟。

```python
# 在这里写你的代码。结果存进 mae_lin、mae_tree 和 mae_rf。
```

```pycheck
need("mae_lin", "mae_tree", "mae_rf")
if close(mae_lin, 4.16, 0.004):
    fail("这是两个变量的直线。这一部分的 cols 有三个变量,多了年龄。")
if close(mae_lin, 4.17, 0.01) and close(mae_tree, 4.07, 0.01) and close(mae_rf, 3.98, 0.01):
    ok("对了: 直线 4.17 Gy,一棵两层的树 4.07 Gy,随机森林 3.98 Gy。和课文里相反,这一次灵活的模型赢了: 森林比直线好 0.2 Gy。说明这份数据里有直线表示不了的东西。")
fail(f"现在是 {mae_lin:.2f}、{mae_tree:.2f}、{mae_rf:.2f},应该是 4.17、4.07、3.98。")
```

<details><summary>提示 1: 思路</summary>

三次都是 `cv_mae(模型, df[cols], y)`,只换模型。

</details>

<details><summary>提示 2: 用到的写法</summary>

`cv_mae(LinearRegression(), df[cols], y)`;`cv_mae(DecisionTreeRegressor(max_depth=2, random_state=0), df[cols], y)`;`cv_mae(forest(), df[cols], y)`。

</details>

<details><summary>参考答案</summary>

```python
mae_lin = cv_mae(LinearRegression(), df[cols], y)
mae_tree = cv_mae(DecisionTreeRegressor(max_depth=2, random_state=0), df[cols], y)
mae_rf = cv_mae(forest(), df[cols], y)
print(round(mae_lin, 2), round(mae_tree, 2), round(mae_rf, 2))
```

</details>

</div>

<div class="step exercise">

**第 2 步 · 读一棵树**

用全部 195 位病人种一棵两层的树(`max_depth=2, random_state=0`),用 `export_text` 打印出来。读出两件事: 第一刀切在肿瘤体积多少(`first_cut`,一个数);肿瘤大的那一边,接着是按哪个变量分的(`right_var`,写列名)。

```python
# 在这里写你的代码。读出来的两样存进 first_cut 和 right_var。
```

```pycheck
need("first_cut", "right_var")
if close(first_cut, 16.7, 0.2):
    fail("16.7 是小肿瘤那一边的第二刀。第一刀在最外层,没有缩进的那一行。")
if str(right_var) == "hrctv_vol":
    fail("按体积再分一次的是小肿瘤那一边(体积 ≤ 26.7)。看体积 > 26.7 下面缩进的那两行。")
if close(first_cut, 26.7, 0.15) and str(right_var) == "needles":
    ok("对了: 第一刀是体积 26.7 cm³。小肿瘤那边再按体积分(100.2 和 94.3 Gy);大肿瘤那边按加不加针分: 不加 89.9 Gy,加了 92.3 Gy。树自己看出来了: 肿瘤小,加不加针无所谓;肿瘤大,加针才要紧。")
fail(f"现在是 {first_cut} 和 {right_var};应该是 26.7 和 needles。")
```

<details><summary>提示 1: 思路</summary>

`export_text(树, feature_names=cols)` 打印出来的每一行是一个问题,缩进一层是下一个问题,`value` 是叶子里的平均值。

</details>

<details><summary>提示 2: 用到的写法</summary>

`tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(df[cols], y)`;`print(export_text(tree, feature_names=cols, decimals=1))`。然后照着打印出来的树,把数和列名写进两个变量。

</details>

<details><summary>参考答案</summary>

```python
tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(df[cols], y)
print(export_text(tree, feature_names=cols, decimals=1))
first_cut = 26.7
right_var = "needles"
```

</details>

</div>

<div class="step exercise">

**第 3 步 · 两组病人,两个斜率**

树说加针的和不加针的病人规律不同。直接验证: 把病人按 `needles` 分成两组,每组单独做"靶区剂量对肿瘤体积"的直线,斜率分别存进 `slope_ic`(没加针)和 `slope_is`(加了针)。

```python
# 在这里写你的代码。结果存进 slope_ic 和 slope_is。
```

```pycheck
need("slope_ic", "slope_is")
if close(slope_ic, -0.044, 0.004) and close(slope_is, -0.349, 0.004):
    fail("两组放反了: needles 是 0 的是没加针的(IC)。")
if close(slope_ic, -0.349, 0.004) and close(slope_is, -0.044, 0.004):
    ok("对了: 没加针的病人,肿瘤每大 10 cm³,靶区剂量低 3.5 Gy;加了针的只低 0.4 Gy。针把肿瘤体积的害处几乎抹平了。一条直线只能给体积一个斜率(−0.138),它是这两个数的折中,对哪一组都不对。")
fail(f"现在是 {slope_ic:.3f} 和 {slope_is:.3f},应该是 −0.349 和 −0.044。")
```

<details><summary>提示 1: 思路</summary>

先用条件选出一组病人,再对这一组做一个变量的线性回归,取 `coef_[0]`。做两次。

</details>

<details><summary>提示 2: 用到的写法</summary>

`part = df[df["needles"] == 0]`;`LinearRegression().fit(part[["hrctv_vol"]], part["d90_total"]).coef_[0]`。

</details>

<details><summary>参考答案</summary>

```python
slopes = []
for v in [0, 1]:
    part = df[df["needles"] == v]
    slopes.append(LinearRegression().fit(part[["hrctv_vol"]], part["d90_total"]).coef_[0])
slope_ic, slope_is = slopes
print(round(slope_ic, 3), round(slope_is, 3))
```

</details>

</div>

<div class="step exercise">

**第 4 步 · 把它写回公式: 交互项**

"体积的作用取决于加没加针",写进直线模型就是一个交互项: 体积乘以 `needles`。在 `df` 里加一列 `vol_x_needles`,再用 `cols` 加上这一列做线性回归,交叉验证误差存进 `mae_inter`。

```python
# 在这里写你的代码。结果存进 mae_inter。
```

```pycheck
need("mae_inter")
if "vol_x_needles" not in df.columns:
    fail("df 里还没有 vol_x_needles 这一列: df['vol_x_needles'] = df['hrctv_vol'] * df['needles']。")
if close(mae_inter, 4.17, 0.01):
    fail("这是没有交互项的模型。X 里要有四列: cols 加上 vol_x_needles。")
if close(mae_inter, 3.71, 0.01):
    ok("对了: 3.71 Gy。加一个交互项,直线模型从 4.17 降到 3.71,比随机森林的 3.98 还好。按这个模型,体积 60 cm³ 的病人不加针预测 82 Gy,加针 92 Gy: 差 9 Gy,而且不加针的那位低于 85 Gy 的标准。练习 1 的模型说只差 1.9 Gy。")
fail(f"现在是 {mae_inter:.2f},应该是 3.71。")
```

<details><summary>提示 1: 思路</summary>

新的一列是两列相乘。列的列表可以相加: `cols + ["vol_x_needles"]`。

</details>

<details><summary>提示 2: 用到的写法</summary>

`df["vol_x_needles"] = df["hrctv_vol"] * df["needles"]`;`cv_mae(LinearRegression(), df[cols + ["vol_x_needles"]], y)`。

</details>

<details><summary>参考答案</summary>

```python
df["vol_x_needles"] = df["hrctv_vol"] * df["needles"]
mae_inter = cv_mae(LinearRegression(), df[cols + ["vol_x_needles"]], y)
print(round(mae_inter, 2))
```

</details>

</div>

<div class="step">

<div class="quiz">
<p><b>第 5 步 · 想一想</b>: 随机森林不用人告诉它,自己就发现了"大肿瘤里加针才要紧"。可是把这一条写成交互项以后,直线模型(3.71 Gy)反而比森林(3.98 Gy)准。为什么?</p>
<button class="quiz-opt" data-why="森林的设置和课文相同,没有调坏。换别的设置,它也很难在 195 位病人上赢过形式正确的四个系数。">森林的参数没调好</button>
<button class="quiz-opt" data-correct data-why="森林不知道关系是什么样子,要从 195 位病人里把形状一点一点学出来,学到的台阶里混着偶然。带交互项的直线只要估计 5 个数,其余都由公式的形式定好了。形式对的时候,少学的那部分就是省下来的误差。所以先用灵活的模型去发现,再把发现写回简单的模型,常常是最好的做法。">森林要从数据里学出整个形状;公式的形式对了,只要估计几个数</button>
<button class="quiz-opt" data-why="交叉验证对两个模型用的是同一种分法,比较是公平的。">交叉验证对直线模型更有利</button>
</div>

</div>

<div class="step exercise">

**第 6 步 · LASSO: 13 个候选变量**

换一个问题: 候选变量很多的时候怎么办。表 `D` 里除了三个真变量,还有 10 列随机数,一共 13 个候选(`many`)。拟合一个标准化以后的 LASSO(`alpha=0.3`),数一数有几个系数不是 0(`n_kept`);再算它的交叉验证误差(`mae_lasso`)。

```python
# 在这里写你的代码。结果存进 n_kept 和 mae_lasso。
```

```pycheck
need("n_kept", "mae_lasso")
if int(n_kept) == 13:
    fail("13 个都留下了,说明没有约束或没有标准化。用 make_pipeline(StandardScaler(), Lasso(alpha=0.3))。")
if int(n_kept) == 6 and close(mae_lasso, 4.30, 0.01):
    ok("对了: 13 个候选里留下 6 个,三个真变量都在,10 列随机数去掉了 7 列。交叉验证 4.30 Gy,比什么都不挑的最小二乘(4.32)好一点,但不如三个真变量的直线(4.17),更不如带交互项的(3.71)。LASSO 会删变量,不会替你发现交互作用。")
fail(f"现在是 {n_kept} 和 {mae_lasso:.2f},应该是 6 和 4.30。")
```

<details><summary>提示 1: 思路</summary>

第 6 课的写法: 标准化和 LASSO 绑成一个 Pipeline。拟合以后,LASSO 在 Pipeline 的最后一步(`[-1]`),它的 `coef_` 里不等于 0 的个数就是留下的变量数。

</details>

<details><summary>提示 2: 用到的写法</summary>

`lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.3)).fit(D[many], y)`;`(lasso[-1].coef_ != 0).sum()`;`cv_mae(make_pipeline(StandardScaler(), Lasso(alpha=0.3)), D[many], y)`。

</details>

<details><summary>参考答案</summary>

```python
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.3)).fit(D[many], y)
coef = pd.Series(lasso[-1].coef_, index=many)
print(coef[coef != 0].round(2))
n_kept = (coef != 0).sum()
mae_lasso = cv_mae(make_pipeline(StandardScaler(), Lasso(alpha=0.3)), D[many], y)
print(n_kept, round(mae_lasso, 2))
```

</details>

</div>

<div class="step exercise">

**第 7 步 · 森林看重什么**

用准备好的 `importance_cv(cols)` 算随机森林里三个变量的置换重要性(要算十几秒)。把肿瘤体积的重要性存进 `imp_vol`,施源器的存进 `imp_needles`。

```python
# 在这里写你的代码。结果存进 imp_vol 和 imp_needles。
```

```pycheck
need("imp_vol", "imp_needles")
if close(imp_vol, 0.29, 0.02) and close(imp_needles, 1.03, 0.03):
    fail("两个数放反了。")
if close(imp_vol, 1.03, 0.03) and close(imp_needles, 0.29, 0.02):
    ok("对了: 把肿瘤体积打乱,误差平均多 1.03 Gy;把施源器打乱,多 0.29 Gy;年龄几乎是 0。森林主要靠体积。")
fail(f"现在是 {imp_vol:.2f} 和 {imp_needles:.2f},应该是 1.03 和 0.29。")
```

<details><summary>提示 1: 思路</summary>

`importance_cv(cols)` 交回一个 Series,每个变量一个数,用列名取。

</details>

<details><summary>提示 2: 用到的写法</summary>

`imp = importance_cv(cols)`;`imp["hrctv_vol"]`、`imp["needles"]`。

</details>

<details><summary>参考答案</summary>

```python
imp = importance_cv(cols)
print(imp.round(2))
imp_vol, imp_needles = imp["hrctv_vol"], imp["needles"]
```

</details>

</div>

<div class="step">

<div class="quiz">
<p><b>第 8 步 · 想一想</b>: 施源器的置换重要性只有 0.29 Gy,不到肿瘤体积的三分之一。能不能说"加不加针对靶区剂量影响不大"?</p>
<button class="quiz-opt" data-why="0.29 是在全部病人身上平均出来的。肿瘤小的病人加不加针几乎一样,把他们的施源器打乱,误差不变;肿瘤大的病人差 9 Gy。平均以后数字不大,不等于对谁都不重要。">能: 重要性低,就是影响小</button>
<button class="quiz-opt" data-correct data-why="两个理由。第一,重要性是对全部病人的平均: 小肿瘤的病人加不加针几乎一样,把这个平均拉低了;对大肿瘤的病人,第 4 步算出来差 9 Gy。第二,重要性说的是'模型的预测有多依赖这个变量',不是'改变这个变量,病人的剂量会变多少'。加针的病人本来就是另一批病人(肿瘤更大、分期更晚),这份数据回答不了因果的问题。">不能: 它是全部病人的平均,而且说的是模型依赖什么,不是因果</button>
<button class="quiz-opt" data-why="重要性的大小和变量的取值个数没有这样的关系。问题在于它是平均值,而且不是因果。">不能: 施源器只有两个取值,所以重要性天然偏低</button>
</div>

</div>

</div>

**这一部分做完了**: 靶区剂量最好的模型是带交互项的直线: 体积、施源器、年龄,加上"体积 × 施源器",交叉验证 3.71 Gy。是随机森林和一棵树让你看到了这个交互作用;写回公式以后,简单的模型更准,也能写进论文。下一部分学完[第 12 课](/stat-ml/12-decision-curve/)以后做: 这两个模型够不够好。
