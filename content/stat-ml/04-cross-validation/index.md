---
title: "Lesson 4 · 在新患者身上表现如何 Cross-Validation"
summary: "训练集上的成绩为什么不可信;k 折和重复交叉验证;out-of-fold 预测。"
weight: 40
aliases: ["/python-ml/08-cross-validation/"]
toc: true
duration: "约 80 分钟,可以分两次学: 第 0–3 节,第 4–7 节"
prereq: "[Python 课](/python/)第 1–4 课;本课程第 1 课(线性回归)"
questions:
  - "模型在建模用的患者身上很准,对新患者也一样准吗?"
  - "手里只有一批患者,怎么估计模型对新患者的误差?"
  - "论文里报告的成绩该怎么算?最后用的是哪个模型?"
objectives:
  - "说明训练集上的成绩为什么偏乐观,什么是过拟合"
  - "手写一遍 5 折交叉验证,再用 scikit-learn 一行得到同样的结果"
  - "用重复交叉验证报告成绩和它随分组的波动"
  - "给每位患者算一个 out-of-fold 预测值"
  - '结局是"是/否"时,用分层的交叉验证'
---

前三课把线性回归当作一个模型讲清楚了,也讲了它的两种用法。用来**解释**时,问的是: 壁厚每增加 1 mm,Δ 平均变多少?回归表里的系数、置信区间和 P 值,回答的是这个问题。

这一课换一个问题,回归表回答不了: 来了一位新患者,只知道壁厚、施源器直径和治疗方式,模型事先算出的 Δ,和这位患者实际的 Δ 会差多少?这是**预测**的问题。最顺手的回答,是看模型在建模患者身上差多少。这一课先让你亲眼看到这个办法为什么不行,再讲论文里通用的办法: 交叉验证。

> 🔬 预测模型论文的结果部分,几乎每个数字(误差、AUC、校准、决策曲线)都来自交叉验证。审稿人第一件要确认的事,也是"这些数是不是在模型没见过的患者身上算的"。

**练习怎么做**: 讲解里预测的是 **Volume** 计划的 Δ。练习都换成 **Graphical** 计划: 每讲完一步,你就对 Graphical 把这一步再做一遍。5 道练习做完,Graphical 的全套结果就是你自己算出来的。

## 0. 准备: 模拟队列和要预测的量

和前几课一样,用课程的模拟队列(数字都是编的,[源码](/python/cohort.py))。要预测的量叫 `delta`: 换成 Volume 计划后,阴道 D2cc 比 5 mm 计划变了多少 Gy,负数表示降低。

```python
import numpy as np
import pandas as pd
from cohort import make_cohort

df = make_cohort().dropna()     # 先去掉缺壁厚测量的几位患者,更好的做法在第 5 课
df["delta"] = df["d2cc_vol"] - df["d2cc_5mm"]
print(len(df), "位患者")
df[["wall_mean", "diameter", "n_fx", "d2cc_5mm", "d2cc_vol", "delta"]].head()
```

第一次运行要下载 numpy 和 pandas,会多等一会儿。表里是前 5 位患者: 平均壁厚(mm)、施源器直径(mm)、近距离次数、两套计划的 D2cc 和它们的差。

## 1. 回顾: 用线性回归做预测

用**线性回归**(linear regression)做预测,就是用一条公式:

> 预测的 Δ = b₀ + b₁ × 平均壁厚 + b₂ × 施源器直径 + b₃ × 近距离次数

**它在算什么**

1. 给 b₀、b₁、b₂、b₃ 随便一组值,对每位建模患者算出预测的 Δ,和实际的 Δ 相减,得到每个人的误差。
2. 把每个误差平方,再全部加起来。
3. 找出让这个"误差平方和"最小的那一组 b,这就是**最小二乘**(least squares)。不用一组一组地试,有公式可以直接算出来。
4. 来了一位新患者,把这位患者的三个数代进公式,就得到预测值。

第 1 课专门讲这几步,包括为什么要平方。下面的代码做第 1–3 步,然后把第一位患者代进公式,手算一遍,和 scikit-learn 算的比一比:

```python
from sklearn.linear_model import LinearRegression

y = df["delta"]
X_a = df[["wall_mean", "diameter", "n_fx"]]
model_a = LinearRegression().fit(X_a, y)          # 第 1–3 步: 找出让误差平方和最小的系数
b0 = model_a.intercept_
b1, b2, b3 = model_a.coef_
print(f"预测 Δ = {b0:.2f} {b1:+.2f} × 壁厚 {b2:+.2f} × 直径 {b3:+.2f} × 次数")

first = X_a.iloc[0]                                # 第 4 步: 把第一位患者代进公式
by_hand = b0 + b1 * first["wall_mean"] + b2 * first["diameter"] + b3 * first["n_fx"]
print(f"第一位患者: 手算 {by_hand:.2f} Gy,model_a.predict 算出 {model_a.predict(X_a.iloc[[0]])[0]:.2f} Gy,实际 {y.iloc[0]:.1f} Gy")
```

公式是: 预测 Δ = −9.94 + 1.57 × 壁厚 + 0.17 × 直径 − 1.21 × 次数。壁厚的系数 1.57 的意思是: 另外两个变量不变时,壁厚每增加 1 mm,预测的 Δ 增加 1.57 Gy,也就是降得少 1.57 Gy。第一位患者(壁厚 3.0 mm、直径 30 mm、近距离 2 次)手算是 −2.64 Gy,和 `model_a.predict` 一样;实际是 −4.0 Gy,差了 1.4 Gy 左右。`predict` 做的就是第 4 步。

一个预测模型就是这样一条"公式",只是公式的样子可以不同: 后面的课会讲到树模型,它的"公式"是一串"是/否"的问题。不管什么样的模型,这一课讲的检验方法都一样用。

## 2. 模型在建模患者身上的成绩,可信吗?

模型有了,建模用的患者也在手里。看模型准不准,最顺手的办法是直接比一比它对这些患者的预测值和实际值。先用一张图看看这个办法会出什么问题。

<div class="demo">

**设定**: 30 位模拟患者(蓝点),横轴是平均壁厚,纵轴是换方案后的 Δ。黑线是用这 30 位患者建的模型。空心点是另外的新患者: 产生方式和这 30 位完全一样,但模型没见过他们。灵活度是 1 时,模型是一条直线;数字越大,曲线越可以弯向附近的患者。

**你做什么**: 把滑块从 1 慢慢拖到 10,看曲线怎么变,再看下面两根条。

**你会看到**: 曲线越来越贴近蓝点,最后几乎穿过每一个蓝点: 在建模的 30 位患者身上,平均误差从 1.53 Gy 一路降到 0.24 Gy。新患者身上的误差却从 1.66 Gy 升到了 2.22 Gy。

**这说明**: 模型在建模患者身上的误差,想要多小就可以有多小;它对新患者准不准,是另一回事。

<div class="widget" data-widget="overfit">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

曲线为什么会这样?这 30 位患者的 Δ,一部分来自壁厚决定的规律(在这个模拟里,真实的规律就是一条直线),另一部分只是每位患者各自的偶然。模型越灵活,越有本事把偶然也一起"学"进去: 曲线为了凑近某一位患者而拐的弯,换一批患者就对不上了。

模型把建模患者的偶然当成规律学了进去,这叫**过拟合**(overfitting)。它的后果是: 模型在建模患者身上的误差(叫**训练误差**, training error)比它在新患者身上的误差小;模型越灵活,两者差得越多。所以训练误差不能当作模型的成绩。

<details>
<summary>展开: 图里的模型是怎么变灵活的?</summary>

先拟合一条直线,算出每位患者离直线差多少;再让曲线在每个位置朝附近患者的方向弯过去,弯多少由附近几位患者的平均差距决定。灵活度越高,"附近"的范围越小(从 1.5 mm 缩到 0.02 mm);到 10 时,曲线在每个位置几乎只听最近一位患者的。

</details>

真实研究里,灵活度更常见的来源不是曲线,而是**变量太多**。换成课程的队列和线性回归,再看一遍。建两个模型,都用来预测 `delta`:

- **模型 A**: 就是上一节的模型,用 3 个确实和 Δ 有关的变量: 平均壁厚、施源器直径、近距离次数。
- **模型 B**: 在 A 的基础上,再加 50 列**随机数**。这 50 列是用随机数造的,和 Δ 毫无关系,模型 B 理应不会更好。

衡量误差用 **MAE**(mean absolute error,平均绝对误差): 每位患者的预测值和实际值差几 Gy,不管正负,取平均。图里的"平均误差"就是它。第 9 课再细讲这类指标。

<div class="quiz">
<p><b>先猜一猜</b>: 在建模用的 196 位患者身上,哪个模型的平均误差更小?</p>
<button class="quiz-opt" data-why="直觉上是这样: 随机数里没有信息。可是多出来的 50 个系数,让模型多了 50 处可以去凑这 196 位患者的地方,和上面曲线多拐的弯是一回事。运行下面的代码看看。">模型 A: 随机数帮不上忙,只会添乱</button>
<button class="quiz-opt" data-correct data-why="多出来的 50 个系数,让模型多了 50 处可以去凑这 196 位患者的地方,和上面曲线多拐的弯是一回事。运行下面的代码,看差多少。">模型 B: 变量多,更能贴近这些患者</button>
<button class="quiz-opt" data-why="不会一样。只要能让这 196 人的误差平方和再小一点,最小二乘就会用上新加的变量。运行下面的代码看看。">一样: 随机数的系数会是 0</button>
</div>

先看训练误差:

```python
from sklearn.metrics import mean_absolute_error

noise = pd.DataFrame(np.random.default_rng(0).normal(size=(len(df), 50)),
                     index=df.index, columns=[f"z{i}" for i in range(1, 51)])
X_b = pd.concat([X_a, noise], axis=1)                       # 模型 B: 再加 50 列随机数
model_b = LinearRegression().fit(X_b, y)

print("训练集上的平均误差(MAE):")
print(f"  模型 A  {mean_absolute_error(y, model_a.predict(X_a)):.2f} Gy")
print(f"  模型 B  {mean_absolute_error(y, model_b.predict(X_b)):.2f} Gy")
```

模型 B 是 1.39 Gy,比模型 A 的 1.67 Gy 小。再造 4900 位新患者(模拟数据的好处就在这里: 真的能拿到一批新患者),看两个模型在他们身上的误差:

```python
new = make_cohort(n=5000, seed=99).dropna()                 # 另一批患者,产生方式完全相同
new["delta"] = new["d2cc_vol"] - new["d2cc_5mm"]
noise_new = pd.DataFrame(np.random.default_rng(1).normal(size=(len(new), 50)),
                         index=new.index, columns=noise.columns)
Xn_a = new[["wall_mean", "diameter", "n_fx"]]
Xn_b = pd.concat([Xn_a, noise_new], axis=1)
print(f"新患者({len(new)} 位)身上的平均误差(MAE):")
print(f"  模型 A  {mean_absolute_error(new['delta'], model_a.predict(Xn_a)):.2f} Gy")
print(f"  模型 B  {mean_absolute_error(new['delta'], model_b.predict(Xn_b)):.2f} Gy")
```

和图里一样: 在新患者身上,模型 B 是 1.95 Gy,反而比模型 A 的 1.73 Gy 差。看看模型 B 的公式,就知道它把什么学了进去:

```python
coef_b = pd.Series(model_b.coef_, index=X_b.columns)
z = coef_b.drop(["wall_mean", "diameter", "n_fx"])         # 50 列随机数的系数
print(f"50 列随机数里,系数绝对值超过 0.1 的有 {(z.abs() > 0.1).sum()} 列,最大的是 {z.abs().max():.2f}")
```

这 50 列和 Δ 毫无关系,它们的系数本该是 0。可是最小二乘只管让**这 196 人**的误差平方和最小: 50 列随机数里,总有一些在这 196 人身上**碰巧**和 Δ 有一点关系,模型就给它们配上系数,把这点巧合也用上了(这里有 29 列的系数超过 0.1,最大的 0.35)。换一批患者,这些巧合不复存在,这些系数反而添乱。

<div class="quiz">
<p><b>想一想</b>: 一位同事说: "我的模型在 200 位患者身上平均只差 0.5 Gy,所以很准。"你首先要问什么?</p>
<button class="quiz-opt" data-correct data-why="这是第一件要弄清的事。如果 0.5 Gy 是在建模用的那 200 人身上算的,它就像模型 B 的 1.39 Gy 一样偏乐观,说明不了对新患者准不准。">这 0.5 Gy 是在建模用的那 200 人身上算的吗?</button>
<button class="quiz-opt" data-why="模型种类不是关键: 任何模型在建模用的患者身上都会显得更准,越灵活的模型越明显。">用的是线性回归,还是机器学习模型?</button>
<button class="quiz-opt" data-why="样本量很重要(第 13 课),但先要弄清 0.5 Gy 是在哪些患者身上算的。如果是建模用的患者,患者再多,这个数也偏乐观。">200 位患者够不够?</button>
</div>

<div class="exercise">

**练习 1 · 轮到你: Graphical 的训练误差**

把结局换成 Graphical 计划的 Δ(`d2cc_gra` 减 `d2cc_5mm`),存进 `y_gra`。用同样的两张变量表各建一个模型,算它们在训练集上的 MAE,存进 `train_a` 和 `train_b`。

```python
y_gra = df[___] - df["d2cc_5mm"]             # Graphical 计划的 Δ
fit_a = LinearRegression().fit(X_a, y_gra)
fit_b = LinearRegression().fit(X_b, y_gra)
train_a = mean_absolute_error(y_gra, fit_a.predict(X_a))
train_b = mean_absolute_error(y_gra, fit_b.predict(___))
print(f"训练集上的 MAE: 模型 A {train_a:.2f} Gy,模型 B {train_b:.2f} Gy")
```

```pycheck
need("y_gra", "train_a", "train_b")
expected = df["d2cc_gra"] - df["d2cc_5mm"]
if not hasattr(y_gra, "__len__") or len(y_gra) != len(expected):
    fail("y_gra 应该是每位患者一个数,一共 196 个。")
if not close(abs(y_gra - expected).max(), 0, 1e-9):
    if close(abs(y_gra - y).max(), 0, 1e-9):
        fail("y_gra 现在还是 Volume 的 Δ: 第一处要填 Graphical 那一列的列名 \"d2cc_gra\"。")
    fail("y_gra 不对: 它应该是 Graphical 计划的 D2cc 减 5 mm 计划的 D2cc。")
if not close(train_a, 2.42, 0.006):
    fail(f"train_a 现在是 {train_a:.2f},应该是 2.42。")
if not close(train_b, 2.13, 0.006):
    fail(f"train_b 现在是 {train_b:.2f},应该是 2.13: 模型 B 预测时要用它自己的变量表 X_b。")
ok("对了: 在建模患者身上,模型 A 2.42 Gy,模型 B 2.13 Gy,B 又显得更好。Graphical 的 Δ 本身变化更大,所以两个误差都比 Volume 的(1.67 和 1.39 Gy)大。")
```

<details><summary>提示</summary>

列名是文字,要加引号: `df["d2cc_gra"]`。模型 B 是用 `X_b` 建的,预测时也要给它 `X_b`。

</details>

<details><summary>参考答案</summary>

```python
y_gra = df["d2cc_gra"] - df["d2cc_5mm"]      # Graphical 计划的 Δ
fit_a = LinearRegression().fit(X_a, y_gra)
fit_b = LinearRegression().fit(X_b, y_gra)
train_a = mean_absolute_error(y_gra, fit_a.predict(X_a))
train_b = mean_absolute_error(y_gra, fit_b.predict(X_b))
print(f"训练集上的 MAE: 模型 A {train_a:.2f} Gy,模型 B {train_b:.2f} Gy")
```

</details>

</div>

真实研究里没有第二批患者可以用来检验。下一节的办法,是从手里这批患者里"造"出新患者来。

## 3. 没有新患者怎么办: 5 折交叉验证

<div class="frame">

**背景**: 训练误差偏乐观,而手里只有这一批患者,没有第二批。

**目标**: 只用手里的患者,估计模型用在新患者身上时的误差。

**做法**: **5 折交叉验证**(5-fold cross-validation): 轮流把五分之一的患者藏起来,当作新患者来考模型。

</div>

**它在算什么**

1. 把患者随机分成 5 份(196 位患者,每份约 39 人)。
2. 拿出第 1 份放在一边,用其余 4 份建模。
3. 用这个模型预测第 1 份的患者,算 MAE。这个模型没见过这些患者,所以对它来说,他们就是新患者。
4. 换成拿出第 2 份、第 3 份……重复第 2–3 步,一共建 5 次模型。
5. 每位患者都恰好被预测了一次,而且预测这位患者的模型没见过这位患者。5 个 MAE 的平均,就是交叉验证的 MAE。

<div class="folds" role="img" aria-label="5 折交叉验证: 每一轮拿出一份患者来预测,用另外四份建模">
<div class="fold-row"><span class="fold-label">第 1 轮</span><span class="cell test">预测</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell">建模</span></div>
<div class="fold-row"><span class="fold-label">第 2 轮</span><span class="cell">建模</span><span class="cell test">预测</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell">建模</span></div>
<div class="fold-row"><span class="fold-label">第 3 轮</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell test">预测</span><span class="cell">建模</span><span class="cell">建模</span></div>
<div class="fold-row"><span class="fold-label">第 4 轮</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell test">预测</span><span class="cell">建模</span></div>
<div class="fold-row"><span class="fold-label">第 5 轮</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell">建模</span><span class="cell test">预测</span></div>
<p class="folds-note">每一列是一份患者。每一轮用 4 份建模,预测剩下的 1 份;5 轮下来,每份都被预测了一次。</p>
</div>

<div class="quiz">
<p><b>想一想</b>: 5 折交叉验证做完,每位患者被预测了几次?参与建模几次?</p>
<button class="quiz-opt" data-why="每一轮只预测被拿出来的那一份。5 轮里,每位患者只在其中一轮被拿出来。">预测 5 次,建模 5 次</button>
<button class="quiz-opt" data-why="被拿出来的那一轮,这位患者不参与建模,否则就不算新患者了。">预测 1 次,建模 5 次</button>
<button class="quiz-opt" data-correct data-why="被拿出来的那一轮被预测 1 次,其余 4 轮都在建模的那 4 份里。">预测 1 次,建模 4 次</button>
</div>

**为什么用它**: 常见的另一种做法,是一开始就留出一部分患者当测试集,比如 150 人建模、46 人测试,只考一次。患者多时这样可以;只有两百人时有两个问题: 只考 46 人,成绩好坏很大程度上取决于碰巧留出了哪 46 人;而且建模白白少用了四分之一的患者。交叉验证让每位患者都当一次"新患者",也都参与了 4 次建模。

**打个比方**: 带规培医生时,考试不能用讲过的病例出题: 背下来的病例考得再好,也说明不了会不会看新患者。交叉验证相当于把病例分成 5 组,每次讲 4 组、用没讲过的 1 组来考,轮 5 次,取平均分。

<div class="demo">

**设定**: 还是图里那 30 位模拟患者,模型是一条直线。他们已经被随机分成 5 份,每份 6 人。

**你做什么**: 点"开始第 1 轮",然后一轮一轮点下去,5 轮之后看合计。

**你会看到**: 每一轮,蓝点是拿出来的 6 位,直线只用灰点拟合,所以每一轮的直线都略有不同。5 份的误差有大有小(0.82 到 2.33 Gy),平均 1.65 Gy;用全部 30 位患者拟合的直线,在 2000 位新患者身上是 1.66 Gy。

**这说明**: 交叉验证里的每一次预测,都是模型在预测它没见过的患者,所以平均下来接近新患者身上的误差。

<div class="widget" data-widget="kfold">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

下面对课程的队列做同样的事,先自己写一遍。函数 `cv_mae` 就是上面的 5 步,注释标出了每一步:

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

`random_state=seed` 固定了随机分组的方式,所以每次运行结果都一样,别人也能重复。`train_idx` 和 `test_idx` 是每一轮建模用的、预测用的患者的位置,`X.iloc[...]` 按位置取出这些患者。

三个成绩放在一起(MAE,Gy):

| | 训练集 | 5 折交叉验证 | 4900 位新患者 |
|---|---|---|---|
| 模型 A(3 个有关的变量) | 1.67 | 1.73 | 1.73 |
| 模型 B(再加 50 列随机数) | 1.39 | 2.04 | 1.95 |

训练集认为 B 更好;交叉验证和新患者都认为 A 更好。模型 B 的交叉验证结果比新患者身上还差一点(2.04 比 1.95): 交叉验证每一轮只用 4/5 的患者建模,模型比用全部患者建的稍差,变量越多,这一点越明显。所以交叉验证的估计**略偏保守**,论文里一般不需要为此修正。

**不用自己写循环**。scikit-learn 的 `cross_val_score` 一行就做完上面的 5 步:

```python
from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(LinearRegression(), X_a, y, cv=kf, scoring="neg_mean_absolute_error")
print(scores.round(2))
print(f"平均 {-scores.mean():.2f} Gy")
```

和手写的结果一模一样,只是带了负号。scikit-learn 规定"分数越大越好",误差是越小越好,所以它把误差取了负数(`neg_` 就是 negative),用的时候再取一次负号。最常用的是 5 折和 10 折: 折数越多,每一轮建模用的患者越多,要建的模型也越多。

现在回到开头那张图,给它加上第三根条: 对每一档灵活度,都只用那 30 位患者做一遍 5 折交叉验证。

<div class="demo">

**设定**: 和第 2 节是同一张图,多了"5 折交叉验证"这根条。算它的时候,没有用到任何一位新患者。

**你做什么**: 再把滑块从 1 拖到 10,这次盯着中间这根条。

**你会看到**: 交叉验证的条从 1.65 Gy 升到 2.16 Gy,大体跟着新患者的那根走(1.66 到 2.22 Gy),不跟着训练误差走(1.53 降到 0.24 Gy)。

**这说明**: 只用手里的患者,交叉验证就能看出: 更灵活的模型其实更差。

<div class="widget" data-widget="overfit" data-cv>这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

仔细看会发现,灵活度是 6 的时候,交叉验证的条比新患者的短了一截(1.62 对 1.74 Gy)。这是因为图里只用了一种分法,有运气的成分。下一节讲怎么办。

<div class="exercise">

**练习 2 · 轮到你: Graphical 的 5 折交叉验证**

用上面写好的 `cv_mae`,对 Graphical 的 Δ(`y_gra`)算模型 A 和模型 B 的交叉验证 MAE,存进 `cv_a` 和 `cv_b`。

```python
cv_a, errors_a = cv_mae(X_a, ___)
cv_b, errors_b = cv_mae(___, y_gra)
print(f"5 折交叉验证的 MAE: 模型 A {cv_a:.2f} Gy,模型 B {cv_b:.2f} Gy")
```

```pycheck
need("cv_a", "cv_b")
if close(cv_a, 2.46, 0.006) and close(cv_b, 2.97, 0.006):
    ok("对了: 交叉验证下,模型 A 2.46 Gy,模型 B 2.97 Gy。训练集上 B 显得更好(2.13 比 2.42),交叉验证把顺序反了过来。在 4900 位新患者身上,它们是 2.55 和 2.86 Gy: 确实是 A 更好。")
if close(cv_a, 1.73, 0.006):
    fail("cv_a 现在是 Volume 的结果(1.73): 第二个参数要传 y_gra。")
if close(cv_b, 2.46, 0.006):
    fail("cv_b 现在和模型 A 的一样: 模型 B 的变量表是 X_b。")
fail(f"现在 cv_a = {cv_a:.2f},cv_b = {cv_b:.2f},应该是 2.46 和 2.97。cv_mae 的两个参数依次是变量表和结局。")
```

<details><summary>提示</summary>

`cv_mae(变量表, 结局)`: 模型 A 是 `cv_mae(X_a, y_gra)`。函数交回两样东西(平均值和每一份的误差),所以等号左边写了两个名字。

</details>

<details><summary>参考答案</summary>

```python
cv_a, errors_a = cv_mae(X_a, y_gra)
cv_b, errors_b = cv_mae(X_b, y_gra)
print(f"5 折交叉验证的 MAE: 模型 A {cv_a:.2f} Gy,模型 B {cv_b:.2f} Gy")
```

</details>

</div>

**回到目标**: 不需要第二批患者,交叉验证就给出了接近新患者的成绩,也挑对了模型。**接下来**: 5 份是随机分的,换一种分法,结果会不会变?

## 4. 换一种分法,结果会变吗: 重复交叉验证

<div class="frame">

**背景**: 5 折是随机分的。换一个随机种子,5 份里的患者就不同,交叉验证的结果也会跟着变。

**目标**: 让报告的成绩不取决于某一次碰巧的分法。

**做法**: **重复交叉验证**(repeated cross-validation): 用 10 种不同的随机分法各做一遍 5 折,取平均。

</div>

**它在算什么**

1. 用一种随机分法把患者分成 5 份,做一遍 5 折,得到一个 MAE。
2. 换一种分法重新分、重新做,一共做 10 遍,得到 10 个 MAE。
3. 10 个 MAE 取平均作为成绩,再看它们的范围或标准差,知道结果随分法变化多大。

**为什么用它**: 只做一次 5 折,结果里混着"这一次怎么分"的运气。做 10 次取平均,运气大体抵消,代价只是多建一些模型(10 × 5 = 50 个)。患者少、结局事件少时,分法的影响更大,论文里常见重复 10 到 20 次。

**打个比方**: 一套考卷可能碰上偏题,考 10 套卷子取平均,更能反映真实水平。

<div class="demo">

**设定**: 还是那 30 位模拟患者和直线模型。每个点是一次完整的 5 折交叉验证的平均误差;第一个点就是上一节那种分法的 1.65 Gy。

**你做什么**: 点几次"再分 1 次",再点"再分 20 次"。

**你会看到**: 每换一种分法,结果都不一样: 前 5 次是 1.65、1.63、1.63、1.57、1.61 Gy;做到 25 次,最小 1.56,最大 1.76。它们的平均(竖线)却很快稳定在 1.64 Gy 左右。

**这说明**: 只做一次 5 折,报告的数字带着运气;换不同的分法多做几次、取平均,数字就稳定了。

<div class="widget" data-widget="repeatcv">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

对课程的队列,用种子 0 到 9 各做一遍 5 折:

```python
for name, X in [("模型 A", X_a), ("模型 B", X_b)]:
    results = [cv_mae(X, y, seed=s)[0] for s in range(10)]
    print(f"{name}  10 次: {min(results):.2f} 到 {max(results):.2f} Gy,平均 {np.mean(results):.2f} Gy")
```

模型 A 的 10 个结果在 1.70 到 1.73 Gy 之间,模型 B 在 1.92 到 2.08 Gy 之间: 变量多的模型,随分法变得更多。

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

重复交叉验证只能消除"这批患者怎么分成 5 份"的运气,消除不了"研究里碰巧是这批患者"的运气。换一家医院、换一批患者,成绩还会变,而且通常变得更多。前一种不确定性靠重复来减小;后一种要用置信区间来表达(第 10 课讲 bootstrap 区间),最终要靠外部验证: 在另一批患者身上检验。

</details>

<div class="exercise">

**练习 3 · 轮到你: Graphical,重复 10 次**

用上面定义好的 `rkf`(5 折重复 10 次),对 Graphical 的 Δ 算模型 A 的成绩: 10 次重复的平均存进 `rep_mean`,10 次之间的标准差存进 `rep_sd`。

```python
scores = -cross_val_score(LinearRegression(), X_a, y_gra, cv=___, scoring="neg_mean_absolute_error")
per_repeat = scores.reshape(10, 5).mean(axis=1)     # 10 次重复各自的成绩
rep_mean = per_repeat.mean()
rep_sd = ___
print(f"Graphical,模型 A: 平均 {rep_mean:.2f} Gy,标准差 {rep_sd:.2f}")
```

```pycheck
need("rep_mean", "rep_sd")
if not close(rep_mean, 2.46, 0.006):
    fail(f"rep_mean 现在是 {rep_mean:.2f},应该是 2.46。cv 要填上面定义好的 rkf。")
if close(rep_sd, 0.28, 0.02):
    fail("这是 50 个分数的标准差。题目要的是 10 次重复各自的成绩(per_repeat)之间的标准差。")
if close(rep_sd, 0.02, 0.006):
    ok("对了: 重复 10 次,平均 2.46 Gy,各次之间的标准差只有 0.02 Gy。论文里可以写成 2.46 Gy (SD 0.02)。")
fail(f"rep_sd 现在是 {rep_sd:.2f},应该是 0.02: 对 per_repeat 求标准差。")
```

<details><summary>提示</summary>

`cv=rkf`。标准差是 `per_repeat.std(ddof=1)`;`ddof=1` 表示除以 n − 1,是统计里通常报告的那一种。

</details>

<details><summary>参考答案</summary>

```python
scores = -cross_val_score(LinearRegression(), X_a, y_gra, cv=rkf, scoring="neg_mean_absolute_error")
per_repeat = scores.reshape(10, 5).mean(axis=1)     # 10 次重复各自的成绩
rep_mean = per_repeat.mean()
rep_sd = per_repeat.std(ddof=1)
print(f"Graphical,模型 A: 平均 {rep_mean:.2f} Gy,标准差 {rep_sd:.2f}")
```

</details>

</div>

**回到目标**: 重复 10 次取平均,成绩不再取决于某一次分法,还能顺便报告它随分法变化有多大。**接下来**: 后面的课不只需要一个平均误差,还需要每位患者一个预测值。

## 5. 每位患者一个诚实的预测值: out-of-fold 预测

<div class="frame">

**背景**: 判断模型够不够好,不只看平均误差。第 11 课的校准图、第 12 课的决策曲线,都要对每位患者比较"预测的"和"实际的"。

**目标**: 给每位患者一个预测值,而且做这个预测的模型没见过这位患者。

**做法**: **out-of-fold 预测**: 交叉验证里,每位患者在被拿出来的那一轮得到的预测值;重复 10 次,就取 10 个预测值的平均。

</div>

**它在算什么**

1. 给每位患者准备一个空格。
2. 做一遍 5 折。每一轮建好模型后,只预测被拿出来的那一份患者,把预测值填进他们的格子。
3. 5 轮做完,每个格子都填上了,每个值都来自没见过这位患者的模型。第 3 节那张图最后的"合计",画的就是这些预测值。
4. 用 10 种分法各做一遍,每位患者就有 10 个预测值,取平均。

**为什么用它**: 如果拿训练集上的预测值去画校准图、算决策曲线,就会和第 2 节一样偏乐观: 模型对见过的患者总是"预测得"特别准。out-of-fold 预测和交叉验证的成绩是同一回事,只是没有先取平均,而是把每位患者的预测都留了下来。

<div class="demo">

**设定**: 模型 B(带 50 列随机数),对同样的 196 位患者,分别用训练集上的预测(上图)和 out-of-fold 预测(下图),画"预测 Δ"对"实际 Δ"的散点图。灰线是"预测 = 实际",点离灰线越远,预测差得越多。

**你做什么**: 运行下面两个代码块。第二个画图,第一次要下载 matplotlib。

**你会看到**: 上图的点离灰线更近(MAE 1.39 Gy),下图的点散得更开(MAE 1.94 Gy)。

**这说明**: 训练集上的预测看起来比实际准。后面凡是要用到每位患者预测值的地方,都用下图这种。

</div>

```python
def oof_predict(X, y, n_repeats=10):
    pred = np.zeros((n_repeats, len(y)))              # 第 1 步: 每次重复、每位患者一个空格
    for r in range(n_repeats):                        # 第 4 步: 10 种分法
        kf = KFold(n_splits=5, shuffle=True, random_state=r)
        for train_idx, test_idx in kf.split(X):       # 第 2–3 步: 5 折
            model = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
            pred[r, test_idx] = model.predict(X.iloc[test_idx])
    return pred.mean(axis=0)                          # 10 个预测值取平均

oof_b = oof_predict(X_b, y)
train_b_pred = model_b.predict(X_b)
print(f"训练集上的预测  MAE {mean_absolute_error(y, train_b_pred):.2f} Gy")
print(f"out-of-fold 预测  MAE {mean_absolute_error(y, oof_b):.2f} Gy")
```

```python
import matplotlib.pyplot as plt

lims = [y.min() - 1, y.max() + 1]
fig, axes = plt.subplots(2, 1, figsize=(4.2, 7), sharex=True, sharey=True)
for ax, pred, title in [(axes[0], train_b_pred, "Training predictions"),
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

<div class="exercise">

**练习 4 · 轮到你: Graphical 的 out-of-fold 预测**

用 `oof_predict` 给每位患者算一个 Graphical Δ 的 out-of-fold 预测值(模型 A),存进 `oof_gra`;再算它的 MAE 比练习 1 的训练误差 `train_a` 多多少,存进 `gap_gra`。

```python
oof_gra = oof_predict(X_a, ___)
gap_gra = mean_absolute_error(y_gra, oof_gra) - train_a
print(f"out-of-fold MAE {mean_absolute_error(y_gra, oof_gra):.2f} Gy,比训练集多差 {gap_gra:.2f} Gy")
```

```pycheck
need("oof_gra", "gap_gra")
if len(oof_gra) != len(y_gra):
    fail("oof_gra 应该是每位患者一个预测值,一共 196 个。")
if close(gap_gra, 0.05, 0.006):
    ok("对了: Graphical 的 out-of-fold MAE 是 2.47 Gy,只比训练误差多 0.05 Gy(模型 B 在 Volume 上多了 0.5 Gy 还多)。模型 A 只有 3 个变量,能记住的偶然很少。这 196 个预测值,第 11、12 课的校准和决策曲线会用到。")
if close(mean_absolute_error(y_gra, oof_gra), 2.78, 0.01):
    fail("oof_predict 的第二个参数填成了 Volume 的 y: 要预测的是 y_gra。")
fail(f"现在 gap_gra = {gap_gra:.2f},应该是 0.05。检查 oof_predict 的第二个参数。")
```

<details><summary>提示</summary>

`oof_predict(变量表, 结局)`,结局是 `y_gra`。

</details>

<details><summary>参考答案</summary>

```python
oof_gra = oof_predict(X_a, y_gra)
gap_gra = mean_absolute_error(y_gra, oof_gra) - train_a
print(f"out-of-fold MAE {mean_absolute_error(y_gra, oof_gra):.2f} Gy,比训练集多差 {gap_gra:.2f} Gy")
```

</details>

</div>

**回到目标**: 每位患者都有了一个诚实的预测值。第 11、12 课的校准和决策曲线就从这里开始。

## 6. 结局是"是/否"时: 分层

<div class="frame">

**背景**: 预测模型常常预测一个是/否的结局,比如"换成 Volume 后,阴道 D2cc 从 68 Gy 以上降到 68 Gy 以下"。这样的事件往往不多,随机分成 5 份时,有的份可能事件很多,有的很少。

**目标**: 让每一份里的事件数差不多,每一轮考试的难度相近。

**做法**: **分层**的 5 折交叉验证(stratified k-fold): 有事件和没事件的患者各自分成 5 份,再一一合起来。

</div>

**它在算什么**

1. 把患者按结局分成两堆: 发生了事件的,没发生的。
2. 两堆各自随机分成 5 份。
3. 第 1 份由两堆各自的第 1 份合起来,第 2 份同理……这样每份的事件数几乎一样。

<div class="demo">

**设定**: 196 位患者里有 34 位从 68 Gy 以上降到了 68 Gy 以下。分别用普通的和分层的 5 折来分,数每一份里有几个事件。

**你做什么**: 运行下面的代码,再把两处 `random_state=3` 换成别的数试试。

**你会看到**: 普通分法这一次每份有 4 到 9 个事件;分层后每份 6 或 7 个。

**这说明**: 普通的随机分法有时很不均匀,分层就不会。

</div>

```python
from sklearn.model_selection import StratifiedKFold

event = ((df["d2cc_5mm"] >= 68) & (df["d2cc_vol"] < 68)).astype(int)
print("降到 68 Gy 以下的患者:", event.sum(), "/", len(event))
for name, cv in [("普通 5 折", KFold(n_splits=5, shuffle=True, random_state=3)),
                 ("分层 5 折", StratifiedKFold(n_splits=5, shuffle=True, random_state=3))]:
    counts = [int(event.iloc[test_idx].sum()) for _, test_idx in cv.split(X_a, event)]
    print(name, "每一份的事件数:", counts)
```

给 `cross_val_score` 传 `cv=5` 这样的整数、模型又是分类模型时,它默认就用分层(但不打乱顺序);自己指定分法时要记得用 `StratifiedKFold`,重复时用 `RepeatedStratifiedKFold`。事件越少,分层越重要。

<div class="exercise">

**练习 5 · 轮到你: Graphical 的分层**

换成 Graphical 计划: 5 mm 计划 ≥ 68 Gy、Graphical 计划 < 68 Gy 的算一个事件。补全代码,用分层 5 折(`random_state=0`)数出每一份的事件数,存进 `counts_gra`。

```python
event_gra = ((df["d2cc_5mm"] >= 68) & (df["d2cc_gra"] < ___)).astype(int)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
counts_gra = [int(event_gra.iloc[test_idx].sum()) for _, test_idx in skf.split(X_a, ___)]
print("事件数:", event_gra.sum(), " 每一份:", counts_gra)
```

```pycheck
need("event_gra", "counts_gra")
if int(event_gra.sum()) != 51:
    fail(f"现在有 {int(event_gra.sum())} 个事件,应该是 51 个。检查定义: 5 mm 计划 ≥ 68 Gy,Graphical 计划 < 68 Gy。")
if sum(counts_gra) != 51:
    fail("每一份的事件数加起来应该等于事件总数 51。")
if max(counts_gra) - min(counts_gra) > 1:
    fail(f"每一份是 {counts_gra},差得有点多: split 的第二个参数要传 event_gra,分层才会按它来分。")
ok(f"对了: 51 位患者换成 Graphical 后降到 68 Gy 以下(Volume 是 34 位),分层后每一份是 {counts_gra}。到这里,Graphical 的这一套你都亲手算过了。")
```

<details><summary>提示</summary>

阈值还是 68。`skf.split(X, 结局)` 的第二个参数告诉它按什么分层,这里是 `event_gra`。

</details>

<details><summary>参考答案</summary>

```python
event_gra = ((df["d2cc_5mm"] >= 68) & (df["d2cc_gra"] < 68)).astype(int)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
counts_gra = [int(event_gra.iloc[test_idx].sum()) for _, test_idx in skf.split(X_a, event_gra)]
print("事件数:", event_gra.sum(), " 每一份:", counts_gra)
```

</details>

</div>

**回到目标**: 分层以后,每一轮考的都是事件数相近的一份患者。

## 7. 交叉验证之后: 最后用哪个模型?

上面一共建了几十个模型,但它们**只用来打分**,用完就丢掉。论文里报告的系数、以后给新患者用的模型,是用**全部患者**重新拟合的那一个:

```python
final = LinearRegression().fit(X_a, y)
new_patient = pd.DataFrame({"wall_mean": [3.0], "diameter": [30], "n_fx": [3]})
print(f"这位患者预测 Δ = {final.predict(new_patient)[0]:.1f} Gy")
```

交叉验证的成绩,估计的是"用这种做法、在这么多患者上建出来的模型"用于新患者时的表现。最终模型用了全部患者,通常比交叉验证里的每个模型都稍好一点,所以用交叉验证的成绩来描述它,是保守的。

<div class="quiz">
<p><b>想一想</b>: 论文的表格里要报告模型的系数。这些系数应该来自哪个模型?</p>
<button class="quiz-opt" data-why="挑成绩最好的一折,等于挑了一次运气;而且那个模型只用了 4/5 的患者。">5 折里成绩最好的那一个</button>
<button class="quiz-opt" data-why="通常不这样做。交叉验证里的模型只用来打分,用完就丢掉。">5 个模型的系数取平均</button>
<button class="quiz-opt" data-correct data-why="交叉验证只用来估计成绩;报告和以后使用的,是用全部患者重新拟合的那一个。">用全部患者重新拟合的模型</button>
</div>

## 小结

- 训练误差偏乐观,模型越灵活(变量越多)越乐观,不能当作成绩报告。这叫过拟合。
- **k 折交叉验证**: 轮流把一份患者藏起来当新患者。每位患者恰好被预测一次,预测用的模型没见过这位患者。
- **重复**交叉验证消除"怎么分"的运气,报告平均值和变化范围。
- **out-of-fold 预测**给每位患者一个诚实的预测值,校准和决策曲线都用它。
- 结局是是/否、事件少时用**分层**。
- 交叉验证只用来打分;报告的模型用全部患者重新拟合。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 训练误差 | training error | 模型在建模用的患者身上的误差,偏乐观 |
| 过拟合 | overfitting | 模型把建模患者的偶然当成规律学了进去 |
| 平均绝对误差 | mean absolute error (MAE) | 预测值和实际值平均差多少 |
| 交叉验证 | cross-validation (CV) | 轮流拿出一部分患者当新患者,检验模型 |
| 折 | fold | 交叉验证里分出来的一份患者 |
| 重复交叉验证 | repeated cross-validation | 换不同的随机分法做多遍,取平均 |
| out-of-fold 预测 | out-of-fold prediction | 患者被拿出来的那一轮得到的预测值 |
| 分层 | stratification | 分组时让每一份的事件数差不多 |
| 内部验证 | internal validation | 用建模的这批患者检验模型,如交叉验证 |
| 外部验证 | external validation | 在另一批患者(别的医院、别的时期)身上检验 |

## 这一课没讲什么

- **留一法**(每次只拿出 1 位患者)和 **bootstrap 验证**: 目的和交叉验证相同的另外两种做法,读论文时会遇到。
- **挑模型、调参数之后**,成绩会偏乐观,要用嵌套交叉验证: 第 5 课。
- **补缺失值、标准化**这些步骤怎样放进交叉验证: 第 5 课。
- **成绩的置信区间**: 第 10 课。
- **外部验证**需要另一批患者。这门课只讲到: 还没有做外部验证时,论文里怎样如实写。

## 写进论文 Writing it up

<div class="paper">

方法部分可以这样写(数字换成你实际用的):

> Predictive performance was assessed by 5-fold cross-validation repeated 10 times with different random splits; in each split, the model was fitted on four folds and evaluated on the held-out fold. For binary outcomes, folds were stratified by outcome. We report the mean performance across repeats. Out-of-fold predictions, averaged across repeats, were used for calibration and decision-curve analyses. The final model was refitted on all patients.

**审稿人可能问**: 为什么没有独立的测试集,或者外部验证?

**怎么回答**: 两百例左右的队列,留出一部分做测试集,测试集太小,估计不稳,建模也少用了患者;重复交叉验证让每位患者都参与建模和评估,是小样本内部验证的常用做法,报告指南(TRIPOD)也推荐用交叉验证或 bootstrap 做内部验证,而不是一次性地拆分数据,可以引用 Collins 等 2024 年在 BMJ 上的综述(见延伸阅读)。同时在局限性里写明: 还没有外部验证,需要在其他中心的患者身上检验。

**审稿人可能问**: 交叉验证的结果还会不会偏乐观?

**怎么回答**: 如果挑模型、挑变量和报告成绩用的是同一份交叉验证,会偏乐观一些。要么事先固定模型和变量,要么用嵌套交叉验证(第 5 课),并如实写明用了哪一种。

**审稿人可能问**: 结果随随机分组变化多大?

**怎么回答**: 报告各次重复之间的标准差或范围(第 4 节),并说明随机种子是固定的。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| `ValueError: Input X contains NaN` | 数据里有缺失值,线性回归不接受。这一课先用 `dropna()` 去掉;第 5 课讲怎样在交叉验证里正确地补缺失值。 |
| `ValueError: n_splits=5 cannot be greater than the number of members in each class` | 用分层交叉验证时,某一类(通常是事件)的患者少于 5 个。减少折数,或者先想想这么少的事件够不够建模(第 13 课)。 |
| 交叉验证的分数是负数 | `neg_` 开头的评分把误差取了负数,再取一次负号就是 MAE。 |
| 结果和网页上写的不一样 | 检查 `random_state` 有没有改动;本页的代码块要从上往下依次运行。 |
| 运行很慢 | 建模次数 = 折数 × 重复次数,模型复杂时会慢。浏览器里比自己电脑上慢不少。 |

## 延伸阅读 Further reading

- scikit-learn 用户指南: [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)。本课用到的 `KFold`、`StratifiedKFold`、`RepeatedKFold`、`cross_val_score`、`cross_val_predict` 都在这一页。
- Collins GS, et al. [Evaluation of clinical prediction models (part 1): from development to external validation](https://www.bmj.com/content/384/bmj-2023-074819). *BMJ* 2024;384:e074819。讲内部验证和外部验证的区别,写论文时可以引用。

**下一课**: 这一课有两处走了捷径: 先用 `dropna()` 去掉了缺失值,模型 A 的变量也是事先就知道的。真实研究里,补缺失值、标准化、挑变量、挑模型,只要有一步用到了被拿出来的那份患者,交叉验证就会偏乐观。第 5 课讲这些"数据泄漏"和怎样避免。
