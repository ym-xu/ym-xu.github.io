---
title: "Lesson 10 · 区分度 Discrimination"
summary: "灵敏度和特异度、ROC 曲线和 AUC;AUC 的 bootstrap 区间和置换检验;AUC 说明不了什么。"
weight: 100
toc: true
duration: "约 75 分钟,可以分两次学: 第 0–3 节,第 4–6 节"
prereq: "本课程[第 2 课](../02-logistic-regression/)(逻辑回归给出概率)和[第 4 课](../04-cross-validation/)(交叉验证)"
questions:
  - "模型给的是概率,怎么判断它分得开会出事和不会出事的患者?"
  - "ROC 曲线上的每个点是什么?AUC 到底是什么的面积?"
  - "34 个事件算出来的 AUC,有多不确定?"
  - "AUC 高,模型就好用吗?"
objectives:
  - "在一个阈值下数出 2×2 表,算灵敏度和特异度"
  - '用"一对一对地比"的办法自己算出 AUC,和 roc_auc_score 对上'
  - "用 bootstrap 给 AUC 一个 95% 区间,用置换检验判断它是不是碰巧"
  - "说明 AUC 不管概率准不准,而且随患者的范围变化"
---

第 2 课的逻辑回归给每位患者一个概率: 换成 Volume 以后,她的阴道 D2cc 降到 68 Gy 以下的概率是多少。第 9 课的尺子(MAE、R²)量的是"差多少 Gy",用不到概率上。

评价一个给概率的模型,要问三件事,各占一课:

- **区分度**(discrimination): 会降的患者,是不是比不会降的患者得到了更高的概率? 这一课。
- **校准**(calibration): 模型说 30%,这样的患者里是不是真有三成降了? 第 11 课。
- **有没有用**: 按模型的概率做决定,比不用模型更好吗? 第 12 课。

**练习怎么做**: 讲解用 **Volume** 后是否降到 68 Gy 以下(`event`),练习换成 **Graphical**(`event_gra`)。

## 0. 准备

```python
import numpy as np
import pandas as pd
from cohort import make_cohort
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve

def prepare(d):
    d = d.dropna()
    hi = d[d["d2cc_5mm"] >= 68].copy()                    # 5 mm 计划达到或超过 68 Gy 的患者(第 2 课)
    hi["excess"] = hi["d2cc_5mm"] - 68                    # 超出 68 Gy 多少
    hi["event"] = (hi["d2cc_vol"] < 68).astype(int)       # 换成 Volume 后降到 68 Gy 以下: 是 1,否 0
    hi["event_gra"] = (hi["d2cc_gra"] < 68).astype(int)
    return hi

hi, new_hi = prepare(make_cohort()), prepare(make_cohort(n=5000, seed=99))
cols = ["excess", "wall_mean"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

def out_of_fold(columns, outcome):
    """每位患者被留出的那一折里,模型给她的概率。"""
    proba = cross_val_predict(LogisticRegression(C=1e6), hi[columns], hi[outcome], cv=cv, method="predict_proba")
    return pd.Series(proba[:, 1], index=hi.index)

y = hi["event"]
p = out_of_fold(cols, "event")
print(f"{len(hi)} 位患者,{y.sum()} 位降到了 68 Gy 以下;新患者 {len(new_hi)} 位")
print("前 5 位患者的概率:", p.head().round(2).tolist(), ";实际:", y.head().tolist())
```

模型还是第 2 课的那个: 用"超出 68 Gy 多少"和平均壁厚,预测会不会降到 68 Gy 以下。`p` 是 144 位患者各自被留出时得到的概率(第 4 课)。

有一处新东西: `StratifiedKFold`。是/否的结局用**分层**的 5 折: 分的时候让每一折里事件的比例都和总体差不多(这里是 34/144)。事件少的时候,随便分可能让某一折只有两三个事件,成绩就很不稳。

## 1. 划一条线: 灵敏度和特异度

概率要变成"是"或"否",得定一个**阈值**(threshold): 概率达到它,就判为"会降"。然后和实际的结局对一对,数出四种患者:

```python
def two_by_two(actual, prob, threshold):
    called = prob >= threshold                       # 模型判为"会降"
    tp = (called & (actual == 1)).sum()              # 判为会降,真的降了
    fn = (~called & (actual == 1)).sum()             # 判为不会,其实降了: 漏掉
    fp = (called & (actual == 0)).sum()              # 判为会降,其实没降: 误判
    tn = (~called & (actual == 0)).sum()             # 判为不会,真的没降
    return tp, fn, fp, tn

for threshold in [0.5, 0.2, 0.1]:
    tp, fn, fp, tn = two_by_two(y, p, threshold)
    print(f"阈值 {threshold}: 降了的 {tp + fn} 位里判对 {tp} 位,灵敏度 {tp / (tp + fn):.0%};"
          f"没降的 {fp + tn} 位里判对 {tn} 位,特异度 {tn / (fp + tn):.0%}")
```

这和诊断试验是同一回事。模型给的概率,就像一项连续的化验结果(比如 D-二聚体);阈值就是化验单上的界值。

- **灵敏度**(sensitivity): 真的降了的患者里,模型认出了几成。阈值 0.5 时是 25/34 = 74%。
- **特异度**(specificity): 没降的患者里,模型判对了几成。阈值 0.5 时是 103/110 = 94%。

阈值从 0.5 降到 0.1,灵敏度从 74% 升到 97%(只漏掉 1 位),特异度从 94% 降到 74%(误判从 7 位增加到 29 位)。界值定得低,漏诊少、误诊多;定得高则相反。同一个模型,没有哪个阈值是"对的": 要看漏掉和误判各有什么代价(第 12 课)。

<div class="demo">

**设定**: 40 位模拟患者,每人有一个模型给的概率。上面一排是实际降下来的 12 位,下面一排是没降的 28 位。竖的虚线是阈值: 线右边的患者被判为"会降"。下面是 ROC 曲线,大圆点是现在这个阈值。

**你做什么**: 把阈值从 0.5 往左拖到 0.2,再往右拖到 0.7。看两排点各有多少落在线的右边,和下面的圆点怎么走。

**你会看到**: 阈值 0.5: 降了的 12 位判对 7 位(灵敏度 58%),没降的 28 位判对 26 位(特异度 93%)。拖到 0.2: 12 位全判对了(100%),但没降的只判对 12 位(43%),圆点走到了右上方。拖到 0.7: 灵敏度只剩 33%,圆点退到了左下方。

**这说明**: 每个阈值是 ROC 曲线上的一个点。挪动阈值,只是沿着曲线走: 灵敏度和特异度一个升、一个降。曲线本身的形状才是模型的本事。

<div class="widget" data-widget="threshold">这里是一张可以操作的图,需要浏览器打开 JavaScript 才能显示。</div>

</div>

**准确率为什么不好用**。还有一个更简单的数: 判对的患者占几成。

```python
tp, fn, fp, tn = two_by_two(y, p, 0.5)
print(f"模型的准确率 {(tp + tn) / len(y):.0%};对谁都说'不会降',准确率 {(y == 0).mean():.0%}")
```

模型的准确率是 89%,听起来不错。可是什么都不做、对每位患者都说"不会降",也有 76%: 因为 144 位里本来就有 110 位没降。事件越少,"全猜没有"的准确率越高。准确率还依赖阈值。所以预测模型的论文一般不拿它当主要指标。

<div class="exercise">

**练习 1 · 轮到你: Graphical,阈值 0.5**

对 Graphical 的结局,算阈值 0.5 时的灵敏度(`sens_g`)和特异度(`spec_g`)。

```python
yg = hi["event_gra"]
pg = out_of_fold(cols, "event_gra")
tp, fn, fp, tn = two_by_two(yg, pg, 0.5)
sens_g = tp / (tp + ___)
spec_g = ___ / (tn + fp)
print(f"降了的 {tp + fn} 位里判对 {tp} 位,灵敏度 {sens_g:.0%};没降的 {tn + fp} 位里判对 {tn} 位,特异度 {spec_g:.0%}")
```

```pycheck
need("sens_g", "spec_g")
if close(sens_g, 25 / 34, 0.004):
    fail('这是 Volume 的结果。结局要用 "event_gra"(起始代码里已经是),检查有没有改动。')
if close(sens_g, 34 / 51, 0.004) and close(spec_g, 79 / 93, 0.004):
    ok("对了: 51 位降了的患者里判对 34 位,灵敏度 67%;93 位没降的里判对 79 位,特异度 85%。两个数都比 Volume 低(74% 和 94%): Graphical 的结局更难分。")
fail(f"现在是灵敏度 {sens_g:.3f}、特异度 {spec_g:.3f},应该是 0.667 和 0.849。灵敏度的分母是真的降了的患者(tp + fn),特异度的分子是 tn。")
```

<details><summary>提示</summary>

灵敏度 = 判对的降了的患者 ÷ 所有降了的患者: 第一个空是 `fn`。特异度 = 判对的没降的患者 ÷ 所有没降的患者: 第二个空是 `tn`。

</details>

<details><summary>参考答案</summary>

```python
yg = hi["event_gra"]
pg = out_of_fold(cols, "event_gra")
tp, fn, fp, tn = two_by_two(yg, pg, 0.5)
sens_g = tp / (tp + fn)
spec_g = tn / (tn + fp)
print(f"降了的 {tp + fn} 位里判对 {tp} 位,灵敏度 {sens_g:.0%};没降的 {tn + fp} 位里判对 {tn} 位,特异度 {spec_g:.0%}")
```

</details>

</div>

## 2. 所有的阈值: ROC 曲线

把阈值从 1 一路降到 0,每个阈值算一次灵敏度和特异度,画成一条线: 横轴是 1 − 特异度(没降的患者里被误判的比例),纵轴是灵敏度。这就是 **ROC 曲线**(receiver operating characteristic curve)。

```python
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y, p)               # fpr = 1 − 特异度,tpr = 灵敏度
fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.plot(fpr, tpr, label="Model")
ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="No information")
for threshold in [0.5, 0.2, 0.1]:
    tp, fn, fp, tn = two_by_two(y, p, threshold)
    ax.plot(fp / (fp + tn), tp / (tp + fn), "o", color="C1")
    ax.annotate(f"threshold {threshold}", (fp / (fp + tn), tp / (tp + fn)), textcoords="offset points", xytext=(8, -4), fontsize=8)
ax.set_xlabel("1 − specificity")
ax.set_ylabel("Sensitivity")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.show()
print(f"曲线上一共 {len(fpr)} 个点")
```

三个橙色的点是上一节的三个阈值。读这张图:

- 曲线越靠近**左上角**越好: 灵敏度高,同时误判少。
- **对角线**是完全没有信息的模型: 不管阈值定在哪,认出的降了的患者和误判的没降的患者一样多,和抛硬币没有区别。
- 曲线是一级一级的台阶,不是光滑的: 144 位患者,阈值每跨过一位患者的概率,曲线就往上或往右走一步。

## 3. 一个数: AUC

<div class="frame">

**背景**: ROC 曲线是一条线。比较两个模型,或者写进摘要,需要一个数。

**目标**: 一个不依赖阈值的数,说明模型把"会降的"和"不会降的"分得多开。

**做法**: 量 ROC 曲线下面的面积,叫 **AUC**(area under the curve)。它有一个不用画图的算法: 把降了的患者和没降的患者**一对一对地比**。

</div>

**它在算什么**

1. 从降了的 34 位里取一位,从没降的 110 位里取一位,配成一对。一共 34 × 110 = 3740 对。
2. 每一对里,看模型给谁的概率更高。给降了的那位更高,这一对就"排对了"。
3. AUC = 排对的对数 ÷ 总对数(并列的算半对)。

所以 AUC 的意思很具体: **随便挑一位降了的患者和一位没降的患者,模型给前者更高概率的机会**。0.5 是抛硬币,1 是每一对都排对。

**打个比方**: 做过 Cox 回归的话,见过 C-index(一致性指数): 随便挑两位患者,模型预测风险更高的那位,是不是真的先出事。AUC 就是是/否结局的 C-index,所以论文里也叫它 C 统计量(C statistic)。

```python
p_event = p[y == 1].to_numpy()                       # 降了的 34 位各自的概率
p_none = p[y == 0].to_numpy()                        # 没降的 110 位
pairs = len(p_event) * len(p_none)
right = (p_event[:, None] > p_none[None, :]).sum()   # 每一对: 降了的那位概率更高吗
ties = (p_event[:, None] == p_none[None, :]).sum()
print(f"{pairs} 对里,{right} 对排对了,{ties} 对并列: AUC = {(right + 0.5 * ties) / pairs:.3f}")
print(f"roc_auc_score: {roc_auc_score(y, p):.3f}")
```

3740 对里排对了 3478 对,AUC = 0.930。和 scikit-learn 算曲线下面积得到的数一样。

**AUC 只看排序,不看概率的数值**:

```python
print(f"把所有概率减半: AUC {roc_auc_score(y, p / 2):.3f}")
```

概率全部减半,每一对里谁高谁低没有变,AUC 还是 0.930。可是减半以后的"概率"显然是错的: 原来说 60% 的患者,现在说 30%。AUC 看不出这种错。概率的数值对不对,是第 11 课的事。

<div class="exercise">

**练习 2 · 轮到你: Graphical 的 AUC,一对一对地比**

对 Graphical 的结局,数出一共多少对(`pairs_g`)、排对了多少对(`right_g`),算出 AUC(`auc_g`)。`yg` 和 `pg` 是练习 1 里的。

```python
pg_event = pg[yg == ___].to_numpy()
pg_none = pg[yg == 0].to_numpy()
pairs_g = len(pg_event) * len(pg_none)
right_g = (pg_event[:, None] ___ pg_none[None, :]).sum()
auc_g = right_g / pairs_g
print(f"{pairs_g} 对里排对了 {right_g} 对,AUC = {auc_g:.3f};roc_auc_score: {roc_auc_score(yg, pg):.3f}")
```

```pycheck
need("pairs_g", "right_g", "auc_g")
if pairs_g == 3740:
    fail("3740 对是 Volume 的。用 yg 和 pg(Graphical)。")
if pairs_g == 4743 and right_g == 519:
    fail("排对的意思是降了的那位概率更高: 第二个空是 >,不是 <。")
if pairs_g == 4743 and right_g == 4224 and close(auc_g, 0.8906, 0.002):
    ok("对了: 51 × 93 = 4743 对,排对了 4224 对,AUC = 0.891。Volume 是 0.930。")
fail(f"现在是 {pairs_g} 对、排对 {right_g} 对,应该是 4743 对和 4224 对。第一个空是 1,第二个空是 >。")
```

<details><summary>提示</summary>

降了的患者是 `yg == 1`;"排对了"是降了的那位概率更高: `>`。

</details>

<details><summary>参考答案</summary>

```python
pg_event = pg[yg == 1].to_numpy()
pg_none = pg[yg == 0].to_numpy()
pairs_g = len(pg_event) * len(pg_none)
right_g = (pg_event[:, None] > pg_none[None, :]).sum()
auc_g = right_g / pairs_g
print(f"{pairs_g} 对里排对了 {right_g} 对,AUC = {auc_g:.3f};roc_auc_score: {roc_auc_score(yg, pg):.3f}")
```

</details>

</div>

## 4. AUC 有多不确定: bootstrap 区间

0.93 是从 144 位患者、34 个事件算出来的。换一批患者,它会变。给它一个区间,用 Python 课第 5 课和本课程第 3 课用过的 **bootstrap**:

**它在算什么**

1. 从 144 位患者里有放回地抽 144 位(同一位可以抽到几次),连同各自的概率和结局。
2. 在这一批里算一次 AUC。
3. 重复 1000 次,取这 1000 个 AUC 中间的 95%。

```python
rng = np.random.default_rng(0)
y_np, p_np = y.to_numpy(), p.to_numpy()
aucs = []
for _ in range(1000):
    idx = rng.integers(0, len(y_np), len(y_np))          # 第 1 步
    aucs.append(roc_auc_score(y_np[idx], p_np[idx]))     # 第 2 步
low, high = np.percentile(aucs, [2.5, 97.5])             # 第 3 步
print(f"AUC {roc_auc_score(y, p):.2f},95% 区间 {low:.2f} 到 {high:.2f}")
```

AUC 0.93,95% 区间 0.88 到 0.97。区间的宽度主要由**事件数**决定,不是患者数: 34 个事件,区间就有这么宽。只报告一个 0.93 而不给区间,读者不知道它可能是 0.88。

<div class="exercise">

**练习 3 · 轮到你: Graphical,AUC 的 95% 区间**

对 Graphical 的结局,用同样的 bootstrap 算 AUC 的 95% 区间(`low_g`、`high_g`)。

```python
rng = np.random.default_rng(0)
yg_np, pg_np = yg.to_numpy(), pg.to_numpy()
aucs_g = []
for _ in range(1000):
    idx = rng.integers(0, len(yg_np), len(yg_np))
    aucs_g.append(roc_auc_score(yg_np[___], pg_np[___]))
low_g, high_g = np.percentile(aucs_g, [___, 97.5])
print(f"AUC {roc_auc_score(yg, pg):.2f},95% 区间 {low_g:.2f} 到 {high_g:.2f}")
```

```pycheck
need("low_g", "high_g")
if close(low_g, 0.8786, 0.004):
    fail("这是 Volume 的区间。用 yg_np 和 pg_np(起始代码里已经是),检查有没有改动。")
if close(low_g, 0.8382, 0.004) and close(high_g, 0.9384, 0.004):
    ok("对了: AUC 0.89,95% 区间 0.84 到 0.94。Volume 是 0.93(0.88 到 0.97): 两个区间有一大段重叠,只凭这批患者,还不能说模型在 Volume 上区分得更好。")
fail(f"现在是 {low_g:.3f} 到 {high_g:.3f},应该是 0.838 到 0.938。前两个空都是 idx,第三个空是 2.5。")
```

<details><summary>提示</summary>

抽到的那一批患者是 `idx`: 结局和概率都按它取。95% 区间的下端是第 `2.5` 百分位。

</details>

<details><summary>参考答案</summary>

```python
rng = np.random.default_rng(0)
yg_np, pg_np = yg.to_numpy(), pg.to_numpy()
aucs_g = []
for _ in range(1000):
    idx = rng.integers(0, len(yg_np), len(yg_np))
    aucs_g.append(roc_auc_score(yg_np[idx], pg_np[idx]))
low_g, high_g = np.percentile(aucs_g, [2.5, 97.5])
print(f"AUC {roc_auc_score(yg, pg):.2f},95% 区间 {low_g:.2f} 到 {high_g:.2f}")
```

</details>

</div>

## 5. 是不是碰巧: 置换检验

另一个问题: 如果这些变量和结局**其实毫无关系**,靠运气能得到多高的 AUC? 第 4 课用置换检验回答过同类的问题,这里是同一个办法:

**它在算什么**

1. 把 144 位患者的结局随机打乱。变量不动。现在变量和结局肯定没有关系了。
2. 在打乱的数据上把**整个流程**重做一遍: 5 折交叉验证,拟合,算 AUC。
3. 重复 200 次,得到"没有关系时"AUC 的样子。
4. 看实际的 AUC 在不在这一堆里。P 值 = 打乱后不低于实际值的次数所占的比例(分子分母各加 1)。

```python
rng = np.random.default_rng(1)
null = []
for _ in range(200):
    shuffled = rng.permutation(y_np)                                     # 第 1 步
    proba = cross_val_predict(LogisticRegression(C=1e6), hi[cols], shuffled, cv=cv, method="predict_proba")
    null.append(roc_auc_score(shuffled, proba[:, 1]))                    # 第 2 步
null = np.array(null)
observed = roc_auc_score(y, p)
print(f"打乱 200 次: AUC 平均 {null.mean():.2f},最大 {null.max():.2f};实际的 AUC {observed:.2f}")
print(f"P = {((null >= observed).sum() + 1) / (len(null) + 1):.3f}")
```

打乱以后,AUC 在 0.49 上下,200 次里最高的一次是 0.67,没有一次够到 0.93。P = 1/201 = 0.005: 这个 AUC 不是碰巧。这一段要拟合 1000 次模型,等十几秒。

留意打乱后最高到过 0.67。患者少的时候,毫无关系的变量靠运气也能得到 0.6 以上的 AUC。看到小样本研究里 0.65 的 AUC,先想到这一点。

反过来,没有信息的变量是什么样子:

```python
p_noise = out_of_fold(["age", "grade"], "event")
print(f"只用年龄和分级: AUC {roc_auc_score(y, p_noise):.2f}")
```

0.42,比 0.5 还低。不是模型"反着准": 没有信息的模型在留出的患者身上,就是在 0.5 上下晃,这一次晃到了下面。

## 6. AUC 说明不了什么

**它取决于患者的范围**。同一个模型、同一批预测,只看其中一部分患者:

```python
narrow = hi["excess"] < 4
print(f"全部 {len(hi)} 位: AUC {roc_auc_score(y, p):.2f}")
print(f"只看超出不到 4 Gy 的 {narrow.sum()} 位({y[narrow].sum()} 个事件): AUC {roc_auc_score(y[narrow], p[narrow]):.2f}")
```

从 0.93 掉到 0.89,模型一点没变。原因是: 超出 8 Gy 以上的患者几乎不可能降到 68 Gy 以下,谁都能把他们排在后面。这些"容易"的患者在全体里把 AUC 抬高了;只留下难分的患者,AUC 就低。

所以两篇论文的 AUC 不能直接比,除非患者的范围差不多。纳入的患者越杂(轻的重的都有),AUC 越高。

**它不管概率准不准**(第 3 节),也**不管模型有没有用**。AUC 0.93 的模型,如果所有患者的最佳处理本来就一样,对决定没有帮助;AUC 0.70 的模型,如果正好在难以决定的患者身上提供了信息,可能很有用。这是第 12 课的内容。

最后在新患者身上对一下答案:

```python
final = LogisticRegression(C=1e6).fit(hi[cols], y)
print(f"建模患者身上 {roc_auc_score(y, final.predict_proba(hi[cols])[:, 1]):.2f},交叉验证 {roc_auc_score(y, p):.2f},"
      f"{len(new_hi)} 位新患者 {roc_auc_score(new_hi['event'], final.predict_proba(new_hi[cols])[:, 1]):.2f}")
```

建模患者身上 0.94,交叉验证 0.93,新患者 0.91。新患者身上的数落在 bootstrap 区间(0.88 到 0.97)里面。

<div class="quiz">
<p><b>想一想</b>: 你的模型 AUC 是 0.78(区间 0.66 到 0.89)。一篇已发表的论文报告了一个预测同一种结局的模型,AUC 0.85。能不能说对方的模型更好?</p>
<button class="quiz-opt" data-why="0.85 确实比 0.78 大。但这两个数是在不同的患者身上算的,而且你的区间把 0.85 包在里面。">能: 0.85 大于 0.78</button>
<button class="quiz-opt" data-correct data-why="三个理由: 你的区间(0.66 到 0.89)包含 0.85,差别可能只是运气;两个研究的患者范围不同,AUC 不可比;对方的 0.85 如果是在建模患者身上算的,还要打折。要比较两个模型,得在同一批患者身上算。">不能: 区间包含 0.85,两批患者的范围也可能不同</button>
<button class="quiz-opt" data-why="AUC 可以比较,前提是在同一批患者身上算。这里不满足这个前提。">不能: AUC 不能用来比较模型</button>
</div>

<div class="exercise">

**练习 4 · 轮到你: Graphical,在新患者身上对答案**

用全部 144 位患者拟合 Graphical 的模型,算它在新患者身上的 AUC(`auc_new_g`)。

```python
final_g = LogisticRegression(C=1e6).fit(hi[cols], hi["event_gra"])
p_new_g = final_g.predict_proba(new_hi[cols])[:, ___]
auc_new_g = roc_auc_score(new_hi[___], p_new_g)
print(f"新患者身上的 AUC: {auc_new_g:.2f}")
```

```pycheck
need("auc_new_g")
if close(auc_new_g, 0.9096, 0.003):
    fail('这是 Volume 的结果。模型和结局都要用 "event_gra"。')
if close(auc_new_g, 1 - 0.8731, 0.003):
    fail("AUC 小于 0.5,是取了'不发生'的概率。predict_proba 的两列依次是不发生、发生: 第一个空是 1。")
if close(auc_new_g, 0.8731, 0.003):
    ok("对了: 新患者身上 0.87。交叉验证给的是 0.89(区间 0.84 到 0.94): 新患者身上的数在区间里面。")
fail(f'auc_new_g 现在是 {auc_new_g:.3f},应该是 0.873。第一个空是 1,第二个空是 "event_gra"。')
```

<details><summary>提示</summary>

`predict_proba` 的第 `1` 列是发生的概率;新患者的实际结局是 `"event_gra"`。

</details>

<details><summary>参考答案</summary>

```python
final_g = LogisticRegression(C=1e6).fit(hi[cols], hi["event_gra"])
p_new_g = final_g.predict_proba(new_hi[cols])[:, 1]
auc_new_g = roc_auc_score(new_hi["event_gra"], p_new_g)
print(f"新患者身上的 AUC: {auc_new_g:.2f}")
```

</details>

</div>

## 小结

- 定一个**阈值**,概率就变成是/否;和实际结局对照,得到**灵敏度**和**特异度**。阈值往下调,灵敏度升、特异度降。
- **ROC 曲线**是所有阈值的灵敏度和 1 − 特异度;**AUC** 是它下面的面积,等于"随便挑一位出事的和一位没出事的,模型把前者排在前面的机会"。
- AUC 要带**区间**(bootstrap);区间的宽度主要看事件数。要知道是不是碰巧,用**置换检验**。
- AUC 只看排序: 不管概率的数值准不准,不管模型有没有用;患者的范围越宽,AUC 越高。
- **准确率**在事件少的时候会骗人。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 区分度 | discrimination | 模型能不能给出事的患者更高的概率 |
| 阈值 | threshold, cut-off | 概率达到多少就判为"是" |
| 灵敏度 | sensitivity (recall, true positive rate) | 实际出事的患者里,判对的比例 |
| 特异度 | specificity | 实际没出事的患者里,判对的比例 |
| ROC 曲线 | receiver operating characteristic curve | 所有阈值下的灵敏度对 1 − 特异度 |
| 曲线下面积 | AUC (AUROC, C statistic) | 一对患者里,模型把出事的那位排在前面的机会 |
| 分层交叉验证 | stratified cross-validation | 每一折里事件的比例和总体相同 |
| 准确率 | accuracy | 判对的患者占全部的比例 |

## 这一课没讲什么

- **阳性预测值和阴性预测值**(PPV、NPV): 判为"会降"的患者里真的降了几成。它们随事件的比例变化,和灵敏度、特异度是 2×2 表的另一种读法。
- **精确率–召回率曲线**(precision–recall curve): 事件很少(比如不到 5%)时比 ROC 更有信息。
- **比较两个模型的 AUC**: 在同一批患者身上,用配对的 bootstrap(每次重抽以后两个模型各算一个 AUC,看差的区间);有名的 DeLong 检验是公式解法。
- **这里的 bootstrap 只重抽了患者**,没有每次重新拟合模型。更完整的做法是把整个建模流程放进 bootstrap(optimism correction),读论文时会遇到。
- **选一个"最佳阈值"**(比如 Youden 指数): 那是在假定漏掉和误判一样糟。第 12 课讲怎么按代价选阈值。

## 写进论文 Writing it up

<div class="paper">

方法和结果可以这样写(数字换成你自己的):

> Discrimination was assessed with the area under the ROC curve (AUC), computed from out-of-fold predictions of stratified 5-fold cross-validation. The 95% confidence interval was obtained by bootstrap resampling of patients (1000 resamples), and a permutation test (200 permutations of the outcome, repeating the full cross-validation each time) assessed whether the AUC exceeded chance. The AUC was 0.93 (95% CI 0.88–0.97; permutation P = 0.005).

**审稿人可能问**: 灵敏度和特异度用的阈值是怎么定的?

**怎么回答**: 阈值要在看结果之前按临床理由定好(或者报告一系列阈值下的结果);不能在同一批患者里挑让成绩最好看的那个。

**审稿人可能问**: AUC 0.93 是不是太高了?

**怎么回答**: 说明患者的范围(这里包括了超出很多、几乎不可能降下来的患者),并报告在更窄、更难分的患者里的 AUC。

</div>

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| `Only one class present in y_true. ROC AUC score is not defined` | 这一批患者里全是事件或全不是事件。bootstrap 时偶尔会抽到这样的一批(事件很少时);跳过那一次。 |
| AUC 小于 0.5 很多,比如 0.1 | 多半是取错了列: `predict_proba(...)[:, 1]` 才是发生的概率。 |
| `roc_auc_score` 用了 `predict` 的结果 | `predict` 只给 0 和 1,等于只用了一个阈值。要传概率。 |
| 置换检验那一段很慢 | 200 次打乱各做一遍 5 折,共拟合 1000 次;等十几秒。 |
| 结果和网页上写的不一样 | 检查 `random_state=0`、`default_rng(0)` 和 `default_rng(1)`;代码块要从上往下依次运行。 |

## 延伸阅读 Further reading

- scikit-learn: [`roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) 的说明。
- Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology* 1982;143(1):29–36。AUC 等于"一对患者排对的机会"出自这里。

**下一课**: AUC 只看排序。把所有概率减半,它一点不变,可是这些概率已经不能照字面相信了。第 11 课讲怎么检查概率的数值: 模型说 30% 的患者里,是不是真有三成出了事。
