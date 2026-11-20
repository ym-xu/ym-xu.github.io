---
title: "Lesson 8 · 在新病人身上表现如何 Cross-Validation"
summary: "训练集上的成绩为什么不可信;k 折和重复交叉验证;out-of-fold 预测。"
weight: 80
toc: true
---

第 7 课的回归用来**解释**: 壁厚每增加 1 mm,Δ 平均变多少。从这一课起,同一个模型换一种用法,用来**预测**: 来了一位新病人,只知道壁厚、施源器直径和治疗方式,能不能事先算出换成体积优化后,阴道 D2cc 会降多少?

用来预测,第一个问题就是: **它对新病人准不准**。这一课要说明,最顺手的那个答案(看模型在建模病人身上的误差)是错的,以及论文里通用的办法: 交叉验证。

> 🔬 预测模型论文的结果部分,几乎每个数字(误差、AUC、校准、决策曲线)都来自交叉验证。审稿人第一件要确认的事,也是"这些数是不是在模型没见过的病人身上算的"。

## 这一课你会做出什么

- 亲眼看到训练集上的成绩为什么偏乐观,变量越多越乐观
- 手写一遍 5 折交叉验证,再用 scikit-learn 的一行代码得到同样的结果
- 用重复交叉验证,看结果随随机分组变化多少
- 给每位病人算一个 out-of-fold 预测值,后面的校准和决策曲线都要用它
- 结局是"是/否"时用分层的交叉验证

## 0. 准备: 模拟队列和要预测的量

和前几课一样,用课程的模拟队列(数字都是编的,[源码](/python-ml/cohort.py))。要预测的量叫 `delta`: 换成体积优化计划后,阴道 D2cc 比 5 mm 计划变了多少 Gy,负数表示降低。

```python
import numpy as np
import pandas as pd
from cohort import make_cohort

df = make_cohort().dropna()     # 先去掉缺壁厚测量的几位病人,更好的做法在第 9 课
df["delta"] = df["d2cc_vol"] - df["d2cc_5mm"]
print(len(df), "位病人")
df[["wall_mean", "diameter", "n_fx", "d2cc_5mm", "d2cc_vol", "delta"]].head()
```

第一次运行要下载 numpy 和 pandas,会多等一会儿。表里是前 5 位病人: 平均壁厚(mm)、施源器直径(mm)、近距离次数、两套计划的 D2cc 和它们的差。

## 1. 问题: 训练集上的成绩偏乐观

先建两个线性回归模型(第 7 课),都用来预测 `delta`:

- **模型 A**: 用 3 个确实和 Δ 有关的变量: 平均壁厚、施源器直径、近距离次数。
- **模型 B**: 在 A 的基础上,再加 50 列**随机数**。这 50 列是我们用随机数造的,和 Δ 毫无关系,模型 B 理应不会更好。

衡量误差用 **MAE**(mean absolute error,平均绝对误差): 每位病人的预测值和实际值差几 Gy,不管正负,取平均。MAE = 1.5 Gy 就是"平均差 1.5 Gy"。第 13 课再细讲这类指标。

<div class="demo">

**设定**: 两个模型用同样的 196 位病人建模;另外再造 4900 位**新病人**,产生方式和这 196 人完全一样。模拟数据的好处就在这里: 真的能拿到一批新病人。

**你做什么**: 依次运行下面两个代码块,先看训练集上的误差,再看新病人身上的误差。

**你会看到**: 在建模的 196 人身上,模型 B 的误差更小(1.39 Gy,A 是 1.67 Gy);换到新病人身上,B 反而更大(1.95 Gy,A 是 1.73 Gy)。

**这说明**: 模型在建模病人身上的成绩不能代表它在新病人身上的成绩,而且变量越多,两者差得越远。

</div>

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

y = df["delta"]
X_a = df[["wall_mean", "diameter", "n_fx"]]                 # 模型 A 的 3 个变量
noise = pd.DataFrame(np.random.default_rng(0).normal(size=(len(df), 50)),
                     index=df.index, columns=[f"z{i}" for i in range(1, 51)])
X_b = pd.concat([X_a, noise], axis=1)                       # 模型 B: 再加 50 列随机数

model_a = LinearRegression().fit(X_a, y)
model_b = LinearRegression().fit(X_b, y)
print("训练集上的平均误差(MAE):")
print(f"  模型 A  {mean_absolute_error(y, model_a.predict(X_a)):.2f} Gy")
print(f"  模型 B  {mean_absolute_error(y, model_b.predict(X_b)):.2f} Gy")
```

```python
new = make_cohort(n=5000, seed=99).dropna()                 # 另一批病人,产生方式完全相同
new["delta"] = new["d2cc_vol"] - new["d2cc_5mm"]
noise_new = pd.DataFrame(np.random.default_rng(1).normal(size=(len(new), 50)),
                         index=new.index, columns=noise.columns)
Xn_a = new[["wall_mean", "diameter", "n_fx"]]
Xn_b = pd.concat([Xn_a, noise_new], axis=1)
print(f"新病人({len(new)} 位)身上的平均误差(MAE):")
print(f"  模型 A  {mean_absolute_error(new['delta'], model_a.predict(Xn_a)):.2f} Gy")
print(f"  模型 B  {mean_absolute_error(new['delta'], model_b.predict(Xn_b)):.2f} Gy")
```

为什么会这样?线性回归会尽量让每位建模病人的误差变小。50 列随机数里,总有几列在这 196 人身上**碰巧**和 Δ 有一点关系,模型就把这点巧合也用上了,于是在这 196 人身上误差变小。换一批病人,这些巧合不复存在,用上它们反而添乱。这叫**过拟合**(overfitting): 模型记住了这批病人的偶然,而不是学到了规律。

可是真实研究里没有第二批病人。下一节的办法,是从手里这批病人里"造"出新病人来。

## 2. k 折交叉验证

<div class="frame">

**背景**: 训练集上的误差偏乐观,而手里只有这 196 位病人,没有第二批。

**目标**: 只用这 196 人,估计模型用在新病人身上时的误差。

**做法**: **5 折交叉验证**(5-fold cross-validation): 轮流把五分之一的病人藏起来,当作新病人来考模型。

</div>

**它在算什么**

1. 把 196 位病人随机分成 5 份,每份约 39 人。
2. 拿出第 1 份放在一边,用其余 4 份(约 157 人)建模。
3. 用这个模型预测第 1 份的病人,算 MAE。这个模型没见过这些病人,所以对它来说,他们就是新病人。
4. 换成拿出第 2 份、第 3 份……重复第 2–3 步,一共建 5 次模型。
5. 每位病人都恰好被预测了一次,而且预测他的模型都没见过他。5 个 MAE 的平均,就是交叉验证的 MAE。

<div class="folds" role="img" aria-label="5 折交叉验证: 每一轮拿出一份病人来预测,用另外四份建模">
<div class="fold-row"><span class="fold-label">第 1 轮</span><span class="cell test">预测</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell">建模</span></div>
<div class="fold-row"><span class="fold-label">第 2 轮</span><span class="cell">建模</span><span class="cell test">预测</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell">建模</span></div>
<div class="fold-row"><span class="fold-label">第 3 轮</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell test">预测</span><span class="cell">建模</span><span class="cell">建模</span></div>
<div class="fold-row"><span class="fold-label">第 4 轮</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell test">预测</span><span class="cell">建模</span></div>
<div class="fold-row"><span class="fold-label">第 5 轮</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell test">预测</span></div>
<p class="folds-note">每一列是一份病人(约 39 人)。每一轮用 4 份建模,预测剩下的 1 份;5 轮下来,每份都被预测了一次。</p>
</div>

**为什么用它**: 常见的另一种做法,是一开始就留出一部分病人当测试集,比如 150 人建模、46 人测试,只考一次。病人多时这样可以;只有两百人时有两个问题: 只考 46 人,成绩好坏很大程度上取决于碰巧留出了哪 46 人;而且建模白白少用了四分之一的病人。交叉验证让每位病人都当一次"新病人",也都参与了 4 次建模。

**打个比方**: 带规培医生时,考试不能用讲过的病例出题,背下来的病例考得再好,也说明不了会不会看新病人。交叉验证相当于把病例分成 5 组,每次讲 4 组、用没讲过的 1 组来考,轮 5 次,取平均分。

<div class="demo">

**设定**: 还是模型 A 和模型 B,还是这 196 位病人,这次不用新病人。

**你做什么**: 运行下面的代码。函数 `cv_mae` 就是上面的 5 步,注释标出了每一步。

**你会看到**: 模型 A 的交叉验证 MAE 是 1.73 Gy,模型 B 是 2.04 Gy。

**这说明**: 只用手里的病人,交叉验证得到的成绩就和新病人身上的成绩(A 1.73,B 1.95)很接近;训练集认为 B 更好,交叉验证和新病人都认为 A 更好。

</div>

```python
from sklearn.model_selection import KFold

def cv_mae(X, y, seed=1):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)   # 第 1 步: 随机分成 5 份
    errors = []
    for train_idx, test_idx in kf.split(X):                    # 第 2–4 步: 每轮拿出 1 份
        model = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        errors.append(mean_absolute_error(y.iloc[test_idx], pred))
    return np.mean(errors), errors                             # 第 5 步: 取平均

for name, X in [("模型 A", X_a), ("模型 B", X_b)]:
    mean, errors = cv_mae(X, y)
    print(f"{name}  每一份: {np.round(errors, 2)}  平均 {mean:.2f} Gy")
```

`random_state=seed` 固定了随机分组的方式,所以每次运行结果都一样,别人也能重复。`train_idx` 和 `test_idx` 是每一轮建模用的、预测用的病人的位置,`X.iloc[...]` 按位置取出这些病人。

三个成绩放在一起(MAE,Gy):

| | 训练集 | 5 折交叉验证 | 4900 位新病人 |
|---|---|---|---|
| 模型 A(3 个有关的变量) | 1.67 | 1.73 | 1.73 |
| 模型 B(再加 50 列随机数) | 1.39 | 2.04 | 1.95 |

模型 B 的交叉验证结果比新病人身上还差一点(2.04 比 1.95)。这是因为交叉验证每一轮只用 4/5 的病人建模,模型比用全部病人建的稍差;变量越多,这一点越明显。所以交叉验证的估计**略偏保守**,论文里一般不需要为此修正。

实际写代码时不用自己写循环。scikit-learn 的 `cross_val_score` 一行就做完上面的 5 步:

```python
from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(LinearRegression(), X_a, y, cv=kf, scoring="neg_mean_absolute_error")
print(scores.round(2))
print(f"平均 {-scores.mean():.2f} Gy")
```

和手写的结果一模一样,只是带了负号。scikit-learn 规定"分数越大越好",误差是越小越好,所以它把误差取了负数(`neg_` 就是 negative),用的时候再取一次负号。

**回到目标**: 不需要第二批病人,交叉验证就给出了接近新病人的成绩,也挑对了模型。**接下来**: 每份是随机分的,换一种分法,结果会不会变?

## 3. 重复交叉验证

<div class="frame">

**背景**: 5 折是随机分的。换一个随机种子,5 份里的病人就不同,交叉验证的结果也会跟着变。

**目标**: 让报告的成绩不取决于某一次碰巧的分法。

**做法**: **重复交叉验证**(repeated cross-validation): 用 10 种不同的随机分法各做一遍 5 折,取平均。

</div>

**它在算什么**

1. 用种子 0 把病人分成 5 份,做一遍 5 折,得到一个 MAE。
2. 换种子 1、2……9,各重新分一次、做一遍,一共得到 10 个 MAE。
3. 10 个 MAE 取平均作为成绩,再看它们的范围或标准差,知道结果随分法变化多大。

**为什么用它**: 只做一次 5 折,结果里混着"这一次怎么分"的运气。做 10 次取平均,运气大体抵消,代价只是多建一些模型(这里是 10 × 5 = 50 个)。病人少、结局事件少时,分法的影响更大,论文里常见重复 10 到 20 次。

**打个比方**: 一套考卷可能碰上偏题,考 10 套卷子取平均,更能反映真实水平。

<div class="demo">

**设定**: 同样的两个模型,种子从 0 到 9,各做一遍 5 折交叉验证。

**你做什么**: 运行下面的代码。

**你会看到**: 模型 A 的 10 个结果在 1.70 到 1.73 Gy 之间,模型 B 在 1.92 到 2.08 Gy 之间。

**这说明**: 换一种分法,结果确实会变,变量多的模型变得更多。只报一次 5 折,等于在碰运气。

</div>

```python
for name, X in [("模型 A", X_a), ("模型 B", X_b)]:
    results = [cv_mae(X, y, seed=s)[0] for s in range(10)]
    print(f"{name}  10 次: {min(results):.2f} 到 {max(results):.2f} Gy,平均 {np.mean(results):.2f} Gy")
```

scikit-learn 里对应的是 `RepeatedKFold`。它一次给出 10 × 5 = 50 个分数,按每次重复整理成 10 行 5 列,每行取平均,就是 10 次重复各自的成绩:

```python
from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
for name, X in [("模型 A", X_a), ("模型 B", X_b)]:
    scores = -cross_val_score(LinearRegression(), X, y, cv=rkf, scoring="neg_mean_absolute_error")
    per_repeat = scores.reshape(10, 5).mean(axis=1)
    print(f"{name}  平均 {per_repeat.mean():.2f} Gy,标准差 {per_repeat.std(ddof=1):.2f}")
```

模型 A 平均 1.71 Gy(标准差 0.01),模型 B 平均 2.03 Gy(标准差 0.05)。随机分组和上面手写的不同,所以数字略有差别,结论一样。

<details>
<summary>展开: 重复能消除哪一种运气?</summary>

重复交叉验证只能消除"这 196 人怎么分成 5 份"的运气,消除不了"研究里碰巧是这 196 人"的运气。换一家医院、换一批病人,成绩还会变,而且通常变得更多。前一种不确定性靠重复来减小;后一种要用置信区间来表达(第 14 课讲 bootstrap 区间),最终要靠外部验证: 在另一批病人身上检验。

</details>

**回到目标**: 重复 10 次取平均,成绩不再取决于某一次分法,还能顺便报告它随分法变化有多大。**接下来**: 后面的课不只需要一个平均误差,还需要每位病人一个预测值。

## 4. out-of-fold 预测

<div class="frame">

**背景**: 判断模型够不够好,不只看平均误差。第 15 课的校准图、第 16 课的决策曲线,都要对每位病人比较"预测的"和"实际的"。

**目标**: 给每位病人一个预测值,而且做这个预测的模型没见过这位病人。

**做法**: **out-of-fold 预测**: 交叉验证里,每位病人在被拿出来的那一轮得到的预测值;重复 10 次,就取 10 个预测值的平均。

</div>

**它在算什么**

1. 给每位病人准备一个空格。
2. 做一遍 5 折。每一轮建好模型后,只预测被拿出来的那一份病人,把预测值填进他们的格子。
3. 5 轮做完,每个格子都填上了,每个值都来自没见过这位病人的模型。
4. 用 10 种分法各做一遍,每位病人就有 10 个预测值,取平均。

**为什么用它**: 如果拿训练集上的预测值去画校准图、算决策曲线,就会和第 1 节一样偏乐观: 模型对见过的病人总是"预测得"特别准。out-of-fold 预测和交叉验证的成绩是同一回事,只是没有先取平均,而是把每位病人的预测都留了下来。

<div class="demo">

**设定**: 模型 B(带 50 列随机数),对同样的 196 位病人,分别用训练集上的预测(上图)和 out-of-fold 预测(下图),画"预测 Δ"对"实际 Δ"的散点图。灰线是"预测 = 实际",点离灰线越远,预测差得越多。

**你做什么**: 运行下面两个代码块。第二个画图,第一次要下载 matplotlib。

**你会看到**: 上图的点离灰线更近(MAE 1.39 Gy),下图的点散得更开(MAE 1.94 Gy)。

**这说明**: 训练集上的预测看起来比实际准。后面凡是要用到每位病人预测值的地方,都用下图这种。

</div>

```python
def oof_predict(X, y, n_repeats=10):
    pred = np.zeros((n_repeats, len(y)))              # 第 1 步: 每次重复、每位病人一个空格
    for r in range(n_repeats):                        # 第 4 步: 10 种分法
        kf = KFold(n_splits=5, shuffle=True, random_state=r)
        for train_idx, test_idx in kf.split(X):       # 第 2–3 步: 5 折
            model = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
            pred[r, test_idx] = model.predict(X.iloc[test_idx])
    return pred.mean(axis=0)                          # 10 个预测值取平均

oof_b = oof_predict(X_b, y)
train_b = model_b.predict(X_b)
print(f"训练集上的预测  MAE {mean_absolute_error(y, train_b):.2f} Gy")
print(f"out-of-fold 预测  MAE {mean_absolute_error(y, oof_b):.2f} Gy")
```

```python
import matplotlib.pyplot as plt

lims = [y.min() - 1, y.max() + 1]
fig, axes = plt.subplots(2, 1, figsize=(4.2, 7), sharex=True, sharey=True)
for ax, pred, title in [(axes[0], train_b, "Training predictions"),
                        (axes[1], oof_b, "Out-of-fold predictions")]:
    ax.scatter(pred, y, s=12, alpha=0.6)
    ax.plot(lims, lims, color="grey", lw=1)           # 预测 = 实际
    ax.set_title(f"{title}, MAE {mean_absolute_error(y, pred):.2f} Gy", fontsize=10)
    ax.set_ylabel("Observed Δ (Gy)")
axes[1].set_xlabel("Predicted Δ (Gy)")
plt.tight_layout()
plt.show()
```

图里的文字用英文: 论文的图本来就要英文,而且浏览器里的 Python 没有中文字体。

只做一次 5 折时,scikit-learn 的 `cross_val_predict` 一行就能得到 out-of-fold 预测,例如 `cross_val_predict(LinearRegression(), X_a, y, cv=KFold(n_splits=5, shuffle=True, random_state=0))`。要重复多次再取平均,就像上面这样自己写循环。

**回到目标**: 每位病人都有了一个诚实的预测值。第 15、16 课的校准和决策曲线就从这里开始。

## 5. 结局是"是/否"时: 分层

<div class="frame">

**背景**: 预测模型常常预测一个是/否的结局,比如"换成体积优化后,阴道 D2cc 从 68 Gy 以上降到 68 Gy 以下"。这样的事件往往不多,随机分成 5 份时,有的份可能事件很多,有的很少。

**目标**: 让每一份里的事件数差不多,每一轮考试的难度相近。

**做法**: **分层**的 5 折交叉验证(stratified k-fold): 有事件和没事件的病人各自分成 5 份,再一一合起来。

</div>

**它在算什么**

1. 把病人按结局分成两堆: 发生了事件的,没发生的。
2. 两堆各自随机分成 5 份。
3. 第 1 份由两堆各自的第 1 份合起来,第 2 份同理……这样每份的事件数几乎一样。

<div class="demo">

**设定**: 196 位病人里有 34 位从 68 Gy 以上降到了 68 Gy 以下。分别用普通的和分层的 5 折来分,数每一份里有几个事件。

**你做什么**: 运行下面的代码,再把两处 `random_state=3` 换成别的数试试。

**你会看到**: 普通分法这一次每份有 4 到 9 个事件;分层后每份 6 或 7 个。

**这说明**: 普通的随机分法有时很不均匀,分层就不会。

</div>

```python
from sklearn.model_selection import StratifiedKFold

event = ((df["d2cc_5mm"] >= 68) & (df["d2cc_vol"] < 68)).astype(int)
print("降到 68 Gy 以下的病人:", event.sum(), "/", len(event))
for name, cv in [("普通 5 折", KFold(n_splits=5, shuffle=True, random_state=3)),
                 ("分层 5 折", StratifiedKFold(n_splits=5, shuffle=True, random_state=3))]:
    counts = [int(event.iloc[test_idx].sum()) for _, test_idx in cv.split(X_a, event)]
    print(name, "每一份的事件数:", counts)
```

给 `cross_val_score` 传 `cv=5` 这样的整数、模型又是分类模型时,它默认就用分层(但不打乱顺序);自己指定分法时要记得用 `StratifiedKFold`,重复时用 `RepeatedStratifiedKFold`。事件越少,分层越重要。

**回到目标**: 分层以后,每一轮考的都是事件数相近的一份病人。

## 6. 交叉验证之后: 最终用哪个模型?

上面一共建了几十个模型,但它们**只用来打分**,用完就丢掉。论文里报告的系数、以后给新病人用的模型,是用**全部病人**重新拟合的那一个:

```python
final = LinearRegression().fit(X_a, y)
new_patient = pd.DataFrame({"wall_mean": [3.0], "diameter": [30], "n_fx": [3]})
print(f"这位病人预测 Δ = {final.predict(new_patient)[0]:.1f} Gy")
```

交叉验证的成绩,估计的是"用这种做法、在这么多病人上建出来的模型"用于新病人时的表现。最终模型用了全部病人,通常比交叉验证里的每个模型都稍好一点,所以用交叉验证的成绩来描述它,是保守的。

## 小结

- 模型在建模病人身上的误差偏乐观,变量越多越乐观,不能当作成绩报告。
- **k 折交叉验证**: 轮流把一份病人藏起来当新病人,每位病人恰好被预测一次,预测他的模型没见过他。
- **重复**交叉验证消除"怎么分"的运气,报告平均值和变化范围。
- **out-of-fold 预测**给每位病人一个诚实的预测值,校准和决策曲线都用它。
- 结局是是/否、事件少时用**分层**。
- 交叉验证只用来打分;报告的模型用全部病人重新拟合。

## 写进论文 Writing it up

<div class="paper">

方法部分可以这样写(数字换成你实际用的):

> Predictive performance was assessed by 5-fold cross-validation repeated 10 times with different random splits; in each split, the model was fitted on four folds and evaluated on the held-out fold. For binary outcomes, folds were stratified by outcome. We report the mean performance across repeats. Out-of-fold predictions, averaged across repeats, were used for calibration and decision-curve analyses. The final model was refitted on all patients.

**审稿人可能问**: 为什么没有独立的测试集,或者外部验证?

**怎么回答**: 两百例左右的队列,留出一部分做测试集,测试集太小,估计不稳,建模也少用了病人;重复交叉验证让每位病人都参与建模和评估,是小样本内部验证的常用做法,报告指南(TRIPOD)也推荐用交叉验证或 bootstrap 做内部验证,而不是一次性地拆分数据。同时在局限性里写明: 还没有外部验证,需要在其他中心的病人身上检验。

**审稿人可能问**: 交叉验证的结果还会不会偏乐观?

**怎么回答**: 如果挑模型、挑变量和报告成绩用的是同一份交叉验证,会偏乐观一些。要么事先固定模型和变量,要么用嵌套交叉验证(第 9 课),并如实写明用了哪一种。

**审稿人可能问**: 结果随随机分组变化多大?

**怎么回答**: 报告各次重复之间的标准差或范围(第 3 节),并说明随机种子是固定的。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| `ValueError: Input X contains NaN` | 数据里有缺失值,线性回归不接受。这一课先用 `dropna()` 去掉;第 9 课讲怎样在交叉验证里正确地补缺失值。 |
| `ValueError: n_splits=5 cannot be greater than the number of members in each class` | 用分层交叉验证时,某一类(通常是事件)的病人少于 5 个。减少折数,或者先想想这么少的事件够不够建模(第 17 课)。 |
| 交叉验证的分数是负数 | `neg_` 开头的评分把误差取了负数,再取一次负号就是 MAE。 |
| 结果和网页上写的不一样 | 检查 `random_state` 有没有改动;本页的代码块要从上往下依次运行。 |
| 运行很慢 | 建模次数 = 折数 × 重复次数,模型复杂时会慢。浏览器里比自己电脑上慢不少。 |

## 延伸阅读 Further reading

- scikit-learn 用户指南: [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)。本课用到的 `KFold`、`StratifiedKFold`、`RepeatedKFold`、`cross_val_score`、`cross_val_predict` 都在这一页。
- Collins GS, et al. Evaluation of clinical prediction models (part 1): from development to external validation. *BMJ* 2024. 讲内部验证和外部验证的区别,写论文时可以引用。

**下一课**: 这一课有两处走了捷径: 先用 `dropna()` 去掉了缺失值,模型 A 的变量也是事先就知道的。真实研究里,补缺失值、标准化、挑变量、挑模型,只要有一步用到了被拿出来的那份病人,交叉验证就会偏乐观。第 9 课讲这些"数据泄漏"和怎样避免。
