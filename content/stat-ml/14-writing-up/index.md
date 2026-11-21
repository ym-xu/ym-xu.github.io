---
title: "Lesson 14 · 写成论文 Writing It Up"
summary: "把十三课串起来: 用模拟队列从头到尾走一遍,每一步对应论文里的一句话和 TRIPOD+AI 的一条;审稿人常问的问题。"
weight: 140
toc: true
duration: "约 70 分钟,可以分两次学: 第 0–4 节,第 5–8 节"
prereq: "本课程第 1–13 课"
questions:
  - "一篇预测模型的论文,必须报告哪些东西?"
  - "前面十三课的内容,各落在论文的哪一段?"
  - "审稿人会问什么?"
  - "怎样保证论文里的数字和分析对得上?"
objectives:
  - "说出 TRIPOD+AI 是什么、管什么"
  - "把一个完整的分析流程写成一页可以重跑的代码"
  - "让论文里的每个数字直接从代码里来"
  - "对照清单检查自己的稿子,预先回答审稿人常问的问题"
---

前面十三课,每一课末尾都有一个"写进论文"的小框。这一课把它们合成一篇: 用课程的模拟队列,从"入组多少病人"一直做到"结果那一段怎么写",中间每一步都指出它对应论文的哪一部分、报告规范的哪一条。

这一课没有新的方法。它的用处是让你在动手做自己的研究之前,看一遍完整的流程长什么样。

**练习怎么做**: 讲解用**体积优化**(`event`),练习换成**图形优化**(`event_gra`)。

## 0. TRIPOD+AI 是什么

预测模型的论文有专门的报告规范,叫 **TRIPOD**(Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis)。2024 年的版本叫 **TRIPOD+AI**: 不管模型是回归还是机器学习,都按同一张清单报告;它取代了 2015 年的旧版。

它是一张 27 条的清单(另有一张 13 条的摘要清单),按论文的结构排列: 标题、摘要、引言、方法、开放科学、患者和公众参与、结果、讨论。投稿时很多期刊要求附上填好的清单,注明每一条写在第几页。

两点要分清:

- TRIPOD+AI 规定的是**报告什么**,不是**怎么做**。它不要求你用哪种模型,只要求你把做了什么说清楚,让读者能判断、让别人能重复。
- 满足清单不等于研究做得好。一篇把过拟合的模型如实报告出来的论文是合规的;但如实报告以后,毛病就藏不住了。这正是它的目的。

下面每一步后面的括号,是清单上对应的条目号。

## 1. 第一步: 问题、病人和结局

动手之前要能用一句话说清: **对哪些病人,在哪个时刻,用哪些信息,预测什么,给谁用**(条目 3–6、8、9)。

这门课的例子: 对术后做阴道近距离放疗、5 mm 计划的阴道 D2cc 达到 68 Gy 的病人,在 5 mm 计划做完的时候,用计划的剂量和 CT 上量的壁厚,预测换成体积优化后能不能降到 68 Gy 以下;给物理师和医生用,决定要不要多做一个优化计划。

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from cohort import make_cohort
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

def select(cohort):
    """分析对象: 有壁厚测量、5 mm 计划达到 68 Gy 的病人。"""
    d = cohort.dropna()
    d = d[d["d2cc_5mm"] >= 68].copy()
    d["excess"] = d["d2cc_5mm"] - 68
    d["event"] = (d["d2cc_vol"] < 68).astype(int)
    d["event_gra"] = (d["d2cc_gra"] < 68).astype(int)
    return d

raw = make_cohort()
hi, new_hi = select(raw), select(make_cohort(n=5000, seed=99))
cols = ["excess", "wall_mean"]

print(f"入组 {len(raw)} 位 → 没有壁厚测量,排除 {len(raw) - len(raw.dropna())} 位 → "
      f"5 mm 计划不到 68 Gy,不在研究范围 {len(raw.dropna()) - len(hi)} 位 → 进入分析 {len(hi)} 位")
print(f"结局: {hi['event'].sum()} 位({hi['event'].mean():.1%})换成体积优化后降到 68 Gy 以下")
print(hi[cols].describe().loc[["mean", "std", "min", "max"]].round(1))
```

**写进论文**: 这几个数就是病人流程图(条目 20a): 入组 200 位,4 位因为没有壁厚测量被排除,52 位不在研究范围,144 位进入分析,34 个事件(23.6%)。再加一张基线表(Table 1,Python 课第 3 课;条目 20b)。

这一步最容易出的毛病不在统计上,而在定义上:

- **预测的时刻**(第 5、9 课): 模型里的每个变量,在要做决定的那一刻是不是已经有了? 这里 5 mm 计划的剂量和壁厚都是现成的。如果放进了"体积优化后的 D90",那是泄漏。
- **结局的定义**(条目 8): 怎么量的,什么时候量的,量的人知不知道预测值。毒性这类结局还要写随访多久。
- **排除了谁**(条目 11): 4 位没有壁厚测量的病人被去掉了。要写出人数和原因;如果缺失很多,直接去掉会让剩下的病人不再有代表性。

## 2. 第二步: 样本量

在看结果之前算(条目 10;第 13 课)。两个候选参数,事件比例约 0.24。预期的 R²cs 来自"以前的研究",这里用那批新病人代替:

```python
def riley_binary(parameters, prevalence, r2_cs, shrinkage=0.9):                  # 第 13 课
    max_r2 = 1 - np.exp(2 * (prevalence * np.log(prevalence) + (1 - prevalence) * np.log(1 - prevalence)))
    n1 = parameters / ((shrinkage - 1) * np.log(1 - r2_cs / shrinkage))
    s2 = r2_cs / (r2_cs + 0.05 * max_r2)
    n2 = parameters / ((s2 - 1) * np.log(1 - r2_cs / s2))
    n3 = (1.96 / 0.05) ** 2 * prevalence * (1 - prevalence)
    return [int(np.ceil(v)) for v in (n1, n2, n3)]

earlier = sm.Logit(new_hi["event"], sm.add_constant(new_hi[cols])).fit(disp=0)
r2_cs = 1 - np.exp(-earlier.llr / len(new_hi))
need = riley_binary(2, 0.24, 0.38)
print(f"以前的研究: R²cs = {r2_cs:.2f}")
print(f"2 个参数、事件比例 0.24、R²cs 0.38: 三条准则各要 {need} 位;现有 {len(hi)} 位")
```

这个模型很强(R²cs 0.38),所以防过拟合只要 37 位,144 位绰绰有余。差的是第 3 条: 把总体风险估到 ±0.05 以内要 281 位。

**写进论文**: "样本量满足控制过拟合的两条准则(分别需要 37 位和 47 位),但不足以精确估计总体风险(需要 281 位,现有 144 位);因此模型的校准需要在更大的队列中确认。"差在哪里就写哪里,比一句"样本量较小"有用得多。

## 3. 第三步: 模型和验证的方案

也是在看结果之前定(条目 12、13、15、16):

- **模型**: 逻辑回归,两个事先定好的变量,不做筛选(第 2、3 课)。不加类别权重(第 11 课;条目 13)。
- **对照**: 一个更灵活的模型,看看多出来的灵活有没有用(第 7 课)。这里用随机森林。
- **验证**: 分层的 5 折交叉验证;所有成绩都用留出时的概率算(第 4、10 课)。
- **成绩**: 区分度(AUC 和 bootstrap 区间)、校准(O/E、校准斜率)、Brier 分数、净获益(第 10–12 课)。阈值概率事先定为 0.1 到 0.3,报告 0.2。

把四种成绩包成一个函数,后面两个模型共用:

```python
def logit(prob):
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return np.log(prob / (1 - prob))

def performance(actual, prob, pt=0.2):
    actual, prob = np.asarray(actual), np.asarray(prob)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(1000):                                                 # AUC 的 bootstrap 区间(第 10 课)
        idx = rng.integers(0, len(actual), len(actual))
        boot.append(roc_auc_score(actual[idx], prob[idx]))
    do = prob >= pt
    return {
        "AUC": roc_auc_score(actual, prob),
        "AUC 下限": np.percentile(boot, 2.5),
        "AUC 上限": np.percentile(boot, 97.5),
        "O/E": actual.sum() / prob.sum(),                                 # 第 11 课
        "校准斜率": sm.Logit(actual, sm.add_constant(logit(prob))).fit(disp=0).params[1],
        "Brier": ((prob - actual) ** 2).mean(),
        "净获益": (do & (actual == 1)).mean() - (do & (actual == 0)).mean() * pt / (1 - pt),   # 第 12 课
        "全都做": actual.mean() - (1 - actual.mean()) * pt / (1 - pt),
    }

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
models = {
    "逻辑回归": LogisticRegression(C=1e6),
    "随机森林": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0),
}
print("方案已定:", list(models), ";5 折分层交叉验证;阈值概率 0.2")
```

**写进论文**: 方法部分照这四条写,每条一两句。要点是让读者看得出这些是**事先定的**。最好的办法是在分析之前写一份方案并注册或存档(条目 18c、18d): 哪些变量、什么模型、怎么验证、阈值范围是多少。事后再说"这些是预先设定的",没有凭据。

## 4. 第四步: 跑一遍,得到成绩

```python
oof = {name: cross_val_predict(model, hi[cols], hi["event"], cv=cv, method="predict_proba")[:, 1]
       for name, model in models.items()}
results = pd.DataFrame({name: performance(hi["event"], prob) for name, prob in oof.items()})
print(results.round(3))
```

逻辑回归: AUC 0.930(0.879 到 0.968),O/E 0.995,校准斜率 0.929,Brier 0.088,净获益 0.151(全都做 0.045)。

随机森林: AUC 0.911(0.860 到 0.956),O/E 1.004,校准斜率 1.172,Brier 0.096,净获益 0.161。两个模型的 AUC 区间大半重叠;森林没有显出优势,校准斜率大于 1(概率偏保守,第 7 课提过)。按第 7 课的规矩: 灵活的模型没有明显更好,就保留简单的。

**写进论文**: 这张表就是结果部分的主表(条目 23a)。每个成绩带区间(这里只给 AUC 做了,其余的同样可以 bootstrap);配一张校准图和一张决策曲线(第 11、12 课)。两个模型都报告,不只报告好看的那个。

<div class="exercise">

**练习 1 · 轮到你: 图形优化的成绩表**

对图形优化的结局,算逻辑回归留出时的概率(`oof_g`)和成绩(`perf_g`)。

```python
oof_g = cross_val_predict(LogisticRegression(C=1e6), hi[cols], hi[___], cv=cv, method="predict_proba")[:, 1]
perf_g = performance(hi["event_gra"], ___)
print(pd.Series(perf_g).round(3))
```

```pycheck
need("oof_g", "perf_g")
if close(perf_g["AUC"], 0.9299, 0.002):
    fail('这是体积优化的结果。第一个空填 "event_gra"。')
if close(perf_g["AUC"], 0.8906, 0.002) and close(perf_g["校准斜率"], 0.960, 0.004) and close(perf_g["净获益"], 0.2882, 0.002):
    ok("对了: AUC 0.891(0.838 到 0.938),O/E 0.997,校准斜率 0.960,Brier 0.133,净获益 0.288(全都做 0.193)。第 10、11、12 课分三次算出来的数,现在一个函数一次给全。")
fail(f"现在 AUC 是 {perf_g['AUC']:.3f}、校准斜率 {perf_g['校准斜率']:.3f},应该是 0.891 和 0.960。第二个空是 oof_g。")
```

<details><summary>提示</summary>

第一个空是结局 `"event_gra"`;第二个空是刚算出来的概率 `oof_g`。

</details>

<details><summary>参考答案</summary>

```python
oof_g = cross_val_predict(LogisticRegression(C=1e6), hi[cols], hi["event_gra"], cv=cv, method="predict_proba")[:, 1]
perf_g = performance(hi["event_gra"], oof_g)
print(pd.Series(perf_g).round(3))
```

</details>

</div>

## 5. 第五步: 最终的模型,写全

成绩是用交叉验证估的;给别人用的模型,是用**全部**病人拟合的那一个(第 4 课)。论文里要把它完整地写出来(条目 22): 截距和每个系数。只给 OR 不给截距,别人算不出概率,也就没法在自己的病人身上验证。

```python
final = sm.Logit(hi["event"], sm.add_constant(hi[cols])).fit(disp=0)
b, ci = final.params, final.conf_int()
table = pd.DataFrame({"系数": b, "OR": np.exp(b), "OR 下限": np.exp(ci[0]), "OR 上限": np.exp(ci[1])})
print(table.loc[cols].round(2))
print(f"log-odds = {b['const']:.2f} + ({b['excess']:.2f}) × 超出 68 Gy 的剂量 + ({b['wall_mean']:.2f}) × 平均壁厚")

one = pd.DataFrame({"const": [1.0], "excess": [2.0], "wall_mean": [3.0]})
print(f"例: 超出 2 Gy、壁厚 3 mm 的病人,概率 {final.predict(one)[0]:.2f}")
```

**写进论文**:

> log-odds = 7.24 − 0.83 × (D2cc with the 5 mm plan − 68 Gy) − 1.59 × mean wall thickness (mm); probability = 1 / (1 + exp(−log-odds)).

再配一个算例(超出 2 Gy、壁厚 3 mm,概率 0.69),读者可以用它核对自己有没有算对。OR 可以放在表里(超出的剂量每多 1 Gy,OR 0.44,95% 区间 0.31 到 0.62;壁厚每厚 1 mm,OR 0.20,0.10 到 0.43),但要记得第 3 课: 这是预测模型,系数不当作因果效应来解读。

如果模型不是一条公式(比如随机森林),就要提供代码或者模型文件(条目 18f、22)。

<div class="exercise">

**练习 2 · 轮到你: 图形优化的最终模型**

用全部 144 位病人拟合图形优化的模型,取出截距(`intercept_g`)和壁厚的 OR(`or_wall_g`)。

```python
final_g = sm.Logit(hi["event_gra"], sm.add_constant(hi[cols])).fit(disp=0)
intercept_g = final_g.params[___]
or_wall_g = np.exp(final_g.params["wall_mean"])
print(f"log-odds = {intercept_g:.2f} + ({final_g.params['excess']:.2f}) × 超出的剂量 + ({final_g.params['wall_mean']:.2f}) × 平均壁厚;壁厚的 OR {or_wall_g:.2f}")
```

```pycheck
need("intercept_g", "or_wall_g")
if close(intercept_g, 7.24, 0.02):
    fail('这是体积优化的模型。结局要用 "event_gra"(起始代码里已经是),检查有没有改动。')
if close(intercept_g, 5.88, 0.02) and close(or_wall_g, 0.333, 0.004):
    ok("对了: log-odds = 5.88 − 0.58 × 超出的剂量 − 1.10 × 平均壁厚;壁厚的 OR 0.33,和第 2 课练习里用 scikit-learn 得到的一样。三个数都写进论文,别人才能用。")
fail(f"现在截距是 {intercept_g:.3f}、OR 是 {or_wall_g:.3f},应该是 5.880 和 0.333。空里填 'const'。")
```

<details><summary>提示</summary>

statsmodels 把截距叫 `"const"`(`sm.add_constant` 加的那一列)。

</details>

<details><summary>参考答案</summary>

```python
final_g = sm.Logit(hi["event_gra"], sm.add_constant(hi[cols])).fit(disp=0)
intercept_g = final_g.params["const"]
or_wall_g = np.exp(final_g.params["wall_mean"])
print(f"log-odds = {intercept_g:.2f} + ({final_g.params['excess']:.2f}) × 超出的剂量 + ({final_g.params['wall_mean']:.2f}) × 平均壁厚;壁厚的 OR {or_wall_g:.2f}")
```

</details>

</div>

## 6. 第六步: 让数字从代码里来

论文里最常见的一类错,不是方法错,而是**抄错**: 表里是 0.93,正文是 0.92;分析重跑过,摘要里还是旧数。办法是不手抄: 让代码把句子写出来。

```python
r = results["逻辑回归"]
paragraph = (
    f"Of {len(raw)} patients, {len(hi)} had a wall thickness measurement and a vaginal D2cc of at least 68 Gy with the 5 mm plan; "
    f"{hi['event'].sum()} ({hi['event'].mean():.0%}) fell below 68 Gy with the volume-optimised plan. "
    f"The prespecified logistic model had a cross-validated AUC of {r['AUC']:.2f} "
    f"(95% CI {r['AUC 下限']:.2f} to {r['AUC 上限']:.2f}), an observed-to-expected ratio of {r['O/E']:.2f} "
    f"and a calibration slope of {r['校准斜率']:.2f}. At a threshold probability of 0.20 its net benefit was "
    f"{r['净获益']:.3f}, compared with {r['全都做']:.3f} for replanning all patients."
)
print(paragraph)
```

数据改了、病人多了、分析重跑了,这段话跟着变,不会和表对不上。这也是这门课从 Python 课就开始强调的: 整个分析写成可以从头重跑的代码,交给合作者或审稿人时,连同代码一起(条目 18e、18f)。

<div class="exercise">

**练习 3 · 轮到你: 图形优化的结果句**

用练习 1 的 `perf_g` 写出图形优化的一句结果(`sentence_g`): AUC 和它的区间。

```python
sentence_g = (
    f"For the graphically optimised plan the AUC was {perf_g[___]:.2f} "
    f"(95% CI {perf_g['AUC 下限']:.2f} to {perf_g[___]:.2f})."
)
print(sentence_g)
```

```pycheck
need("sentence_g")
if "0.93" in sentence_g and "0.89" not in sentence_g:
    fail("句子里是体积优化的数。用 perf_g(图形优化)。")
if "0.89" in sentence_g and "0.84 to 0.94" in sentence_g:
    ok("对了: 'the AUC was 0.89 (95% CI 0.84 to 0.94)'。没有一个数是手敲的。")
fail("句子里应该有 0.89 和 '0.84 to 0.94'。第一个空是 'AUC',第二个空是 'AUC 上限'。")
```

<details><summary>提示</summary>

`perf_g` 是一个字典: 第一个空是 `"AUC"`,第二个空是 `"AUC 上限"`。

</details>

<details><summary>参考答案</summary>

```python
sentence_g = (
    f"For the graphically optimised plan the AUC was {perf_g['AUC']:.2f} "
    f"(95% CI {perf_g['AUC 下限']:.2f} to {perf_g['AUC 上限']:.2f})."
)
print(sentence_g)
```

</details>

</div>

## 7. 审稿人常问的问题

这些问题在前面各课的"写进论文"里都出现过。投稿之前,自己先答一遍:

| 审稿人问 | 要能回答 | 哪一课 |
|---|---|---|
| 成绩是在哪些病人身上算的? | 留出的病人(交叉验证或独立的验证集),不是建模的病人 | 第 4 课 |
| 变量筛选、标准化、调参,在交叉验证里面吗? | 都在每一折的建模病人里重做 | 第 5 课 |
| 预测的那一刻,这些变量都有了吗? | 逐个核对过,没有用到之后才知道的信息 | 第 5、9 课 |
| 多少个候选变量,多少个事件? | 给出数目和按准则算的样本量 | 第 13 课 |
| 和简单的模型比过吗? | 比过,同一批病人、同一种验证 | 第 7 课 |
| 只有 AUC,校准呢? | 校准图、O/E、校准斜率 | 第 11 课 |
| 成绩的区间呢? | bootstrap 的 95% 区间 | 第 10 课 |
| 用了 SMOTE 或类别加权吗? | 没有;或者用了,并且重新校准过 | 第 11 课 |
| 变量重要性说明这些因素影响结局吗? | 不说明;它只描述模型靠什么预测 | 第 8 课 |
| 模型能用来做决定吗? 阈值谁定的? | 决策曲线;阈值范围按临床理由事先定 | 第 12 课 |
| 别人能用、能验证你的模型吗? | 完整的公式或代码 | 本课 |

<div class="quiz">
<p><b>想一想</b>: 一篇论文的摘要写: "我们开发的机器学习模型能准确预测晚期毒性(AUC 0.91),可用于指导个体化治疗。"正文里: 88 位病人,AUC 是在全部病人上拟合后算的,没有校准,没有决策曲线。下面哪一句评价最准确?</p>
<button class="quiz-opt" data-why="AUC 0.91 是在建模病人身上算的(第 4 课),不代表新病人身上的成绩。">AUC 很高,结论基本成立,补一个外部验证就好</button>
<button class="quiz-opt" data-correct data-why="三处都超出了证据: '准确预测'用的是建模病人身上的 AUC;'个体化'需要概率可信,而没有校准;'指导治疗'需要净获益,而没有决策曲线。88 位病人也撑不起多少参数。如实的结论是'提出了一个待验证的模型'。">结论超出了证据: 成绩不是留出的,概率没有校准,也没有证明用它做决定有好处</button>
<button class="quiz-opt" data-why="问题不在用了哪一类模型,而在怎么评价、结论说到哪一步。">问题在于用了机器学习而不是逻辑回归</button>
</div>

## 8. TRIPOD+AI 和这门课

清单的条目和课程的对应。左边的条目号可以直接用来填投稿时的清单。

| 条目 | 要报告的 | 这门课 |
|---|---|---|
| 1–2 标题和摘要 | 标题里说明是开发还是评价预测模型、对象和结局;摘要另有清单 | 本课 |
| 3–4 背景和目的 | 临床背景,模型用在诊疗的哪一步、给谁用 | 第 3 课 |
| 5–6 数据和病人 | 数据来源、时间范围、纳入标准、接受的治疗 | 本课第 1 节 |
| 7 数据准备 | 预处理和质量检查 | 第 5 课 |
| 8 结局 | 定义、评估的时间和方法 | 第 2 课,本课 |
| 9 变量 | 候选变量怎么选的、什么时候测的 | 第 5、13 课 |
| 10 样本量 | 怎么确定的,够不够 | 第 13 课 |
| 11 缺失数据 | 怎么处理,排除了谁、为什么 | 第 5 课 |
| 12 分析方法 | 数据怎么用于开发和评价、变量怎么进模型、模型和调参、内部验证、评价指标 | 第 4–7、9–12 课 |
| 13 类别不平衡 | 用没用加权或重采样;用了怎么重新校准 | 第 11 课 |
| 14 公平性 | 模型在不同人群里是否同样可靠 | 这门课没讲 |
| 15 模型输出 | 输出的是概率还是分类;阈值怎么定的 | 第 10、12 课 |
| 16 开发和评价数据的差别 | 两批数据在场景、标准、变量上的不同 | 第 4 课 |
| 17–19 伦理、开放科学、患者参与 | 伦理批准;经费、利益冲突、方案、注册、数据和代码;患者和公众的参与 | 本课第 3、6 节 |
| 20–21 病人和开发 | 病人流程、基线特征;每一步分析的病人数和事件数 | 本课第 1 节 |
| 22 模型的完整写法 | 公式或代码,让别人能预测、能验证 | 本课第 5 节 |
| 23 成绩 | 带区间的成绩,配图 | 第 9–12 课 |
| 24 模型更新 | 重新校准等更新的结果 | 第 11 课 |
| 25–27 讨论 | 解释、局限、能不能用于现在的诊疗、下一步 | 第 8、13 课 |

两条这门课没有展开的: **公平性**(条目 14: 模型在不同年龄、不同人群里是不是同样准)和**患者与公众的参与**(条目 19)。清单要求你至少写明做了没有。

## 你自己的第一篇

方法到这里讲完了。动手做自己的预测模型之前,按顺序做这几件事,都在看数据之前:

1. **写一句话**: 对哪些病人,在哪个时刻,用哪些信息,预测什么,给谁用(第 1 节)。
2. **数事件**,按第 13 课算出撑得起几个参数,据此按临床知识定下候选变量。
3. **定方案**: 模型、对照、验证方式、评价指标、阈值概率的范围。写下来,存档或注册。
4. **写成一页可以重跑的代码**,像这一课这样;论文里的数字从代码里来。
5. **对照清单**检查稿子,把第 7 节的问题自己先答一遍。

这门课没讲、而你多半会用到的: **随访时间不等、有删失的结局**(晚期毒性正是这一类;用 Cox 模型做预测,评价的想法相同,尺子换成随时间变化的版本),**缺失值的填补**,以及**外部验证**(在另一家医院的病人身上检验)。前两样要另外学;第三样要靠合作。

## 小结

- **TRIPOD+AI**: 预测模型论文的报告清单,27 条,回归和机器学习通用。它管报告什么,不管怎么做。
- 先定问题和方案,再看数据: 病人、时刻、变量、结局、样本量、模型、验证、阈值。
- 报告**全部**: 病人流程、所有模型的成绩(带区间)、校准、净获益、完整的模型。
- 论文里的数字**从代码里来**,不手抄。
- 结论不超出证据: 留出的成绩、校准过的概率、证明过的净获益,各支撑一层说法。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 报告规范 | reporting guideline | 规定一类研究的论文必须交代哪些内容 |
| TRIPOD+AI | TRIPOD+AI | 预测模型研究的报告规范(2024),回归和机器学习通用 |
| 病人流程图 | participant flow diagram | 从入组到进入分析,每一步多少人、为什么排除 |
| 模型开发 | model development | 建立模型的研究 |
| 模型评价(验证) | model evaluation (validation) | 检验模型在一批病人身上的成绩 |
| 可重复的分析 | reproducible analysis | 从原始数据到论文里的数字,能用代码从头重跑 |
| 研究方案 | protocol | 分析之前写下的计划 |

## 这一课没讲什么

- **PROBAST**: 评价预测模型研究偏倚风险的工具,读别人的论文、做系统综述时用。它和 TRIPOD 互补: 一个管报告,一个管质量。
- **摘要的清单**: TRIPOD+AI for Abstracts,13 条。
- **模型上线以后的事**: 监测模型的成绩是不是随时间变差,什么时候需要更新。

## 常见报错 Troubleshooting

| 报错或现象 | 原因和办法 |
|---|---|
| 成绩表那一段要等一会儿 | 随机森林 300 棵树、5 折,再加 2000 次 bootstrap。等十秒左右。 |
| `KeyError: 'const'` | 拟合时没有 `sm.add_constant`,模型里就没有截距这一项。 |
| 重跑以后数字和论文里不一样 | 检查随机种子有没有固定;检查数据文件是不是换过。这正是让数字从代码里来的理由。 |
| 结果和网页上写的不一样 | 检查 `random_state=0` 和 `default_rng(0)`;代码块要从上往下依次运行。 |

## 延伸阅读 Further reading

- Collins GS, Moons KGM, Dhiman P, 等. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 2024;385:e078378。清单和逐条的说明,开放获取;清单的可填写版本在 [tripod-statement.org](https://www.tripod-statement.org/)。
- Qiang Y, 等. D90 and D2cc dose in vaginal CTV as dosimetric predictors of late vaginal side effects in postoperative vaginal modern brachytherapy after external beam irradiation in endometrial cancer. *Cancers* 2026;18(14):2350。这门课的临床背景;第 2 课里 68 Gy 这个界值的出处,开放获取。

**下一课**: 第 15 课不讲新方法。它把学过的六种模型并排放在一张分步的图里: 同一位病人,看每种模型各自怎么算,为什么给出不同的数。
