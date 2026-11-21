"""本课程的模拟队列。

所有数字都是编的: 只是让数据的样子像一个子宫内膜癌术后阴道近距离放疗(VCB)的队列,
不是任何真实研究的结果。每位患者在同一个 CT 上有三套计划:

    5mm  在施源器表面下 5 mm 处给处方(传统做法)
    vol  Volume
    gra  Graphical

用法:

    from cohort import make_cohort, make_followup
    df = make_cohort()                   # 200 位患者, 一人一行
    new = make_cohort(n=5000, seed=99)   # 另一批"新患者", 产生方式完全相同
    fu = make_followup()                 # 随访表: 有随访的患者一人一行

列(一人一行):

    id          患者编号
    age         年龄(岁)
    height_cm   身高(cm)
    weight_kg   体重(kg)
    arm         治疗方式: "VCB"(单纯近距离)或 "EBRT+VCB"(外照射加近距离)
    stage       分期: "IA" "IB" "II"
    grade       分级: 1 2 3
    n_fx        近距离治疗的次数
    diameter    施源器直径(mm)
    wall_min    阴道壁厚度的最小值(mm),少数患者缺失
    wall_mean   阴道壁厚度的平均值(mm),少数患者缺失
    wall_max    阴道壁厚度的最大值(mm),少数患者缺失
    d2cc_5mm    阴道 D2cc, 外照射加近距离的总 EQD2(Gy, α/β = 3), 5 mm 计划
    d2cc_vol    同上, Volume 计划
    d2cc_gra    同上, Graphical 计划
    d90_5mm     靶区 CTV 的 D90, 总 EQD2(Gy, α/β = 10), 5 mm 计划
    d90_vol     同上, Volume 计划
    d90_gra     同上, Graphical 计划

随访表(make_followup)的列:

    id          患者编号
    fu_months   随访时间(月)
    tox         晚期阴道毒性: 1 = 出现过(G1 及以上), 0 = 没有
"""

import numpy as np
import pandas as pd


def make_cohort(n=200, seed=22):
    """造一个模拟队列: n 位患者, 一人一行。seed 相同, 得到的数据就完全相同。"""
    rng = np.random.default_rng(seed)

    # 患者和治疗
    age = np.clip(rng.normal(64, 9, n), 38, 88).round()
    ebrt = rng.random(n) < 0.55                                   # 约一半多加外照射
    arm = np.where(ebrt, "EBRT+VCB", "VCB")
    stage = np.where(
        ebrt,
        rng.choice(["IA", "IB", "II"], n, p=[0.30, 0.45, 0.25]),
        rng.choice(["IA", "IB", "II"], n, p=[0.75, 0.22, 0.03]),
    )
    grade = rng.choice([1, 2, 3], n, p=[0.35, 0.40, 0.25])
    n_fx = np.where(ebrt, 2, 3)                                   # 近距离次数
    diameter = rng.choice([25, 30, 35], n, p=[0.25, 0.50, 0.25])

    # 阴道壁厚度(mm)
    wall_mean = np.exp(rng.normal(np.log(3.6), 0.30, n))
    wall_max = wall_mean + rng.gamma(2.0, 0.8, n)
    wall_min = wall_mean * rng.uniform(0.45, 0.85, n)

    # 5 mm 计划的剂量: 施源器越细, 表面剂量越高
    d2cc_5mm = np.where(ebrt, 72.0, 69.0) + 0.5 * (30 - diameter) + rng.normal(0, 4.5, n)
    d90_5mm = np.where(ebrt, 60.0, 31.0) + 0.6 * n_fx * (5 - wall_mean) + rng.normal(0, 1.5, n)

    # 换成 Volume 或 Graphical 计划后的变化: 壁越薄, 靶区越浅, 剂量降得越多。
    # 近距离次数越多, 累计的变化越大。两个剂量指标的随机部分互相相关。
    shared = rng.normal(0, 1, n)
    k = n_fx / 2.5
    d_d2cc_vol = k * (1.6 * (wall_mean - 5) + 0.15 * (diameter - 30)) + 1.6 * shared + rng.normal(0, 1.4, n)
    d_d2cc_gra = k * (2.0 * (wall_mean - 5) - 1.0) + 2.2 * shared + rng.normal(0, 2.2, n)
    d_d90_vol = k * 1.2 * (wall_mean - 5) + 0.9 * shared + rng.normal(0, 0.8, n)
    d_d90_gra = k * (1.5 * (wall_mean - 5) - 0.6) + 1.2 * shared + rng.normal(0, 1.3, n)

    # 身高和体重: 另用一串随机数, 不影响上面各列
    rng_b = np.random.default_rng([seed, 1])
    height = np.clip(rng_b.normal(159, 6, n), 142, 178).round()
    bmi = np.exp(rng_b.normal(np.log(28.5), 0.19, n))
    weight = (bmi * (height / 100) ** 2).round(1)

    df = pd.DataFrame({
        "id": [f"P{i:03d}" for i in range(1, n + 1)],
        "age": age.astype(int),
        "height_cm": height.astype(int),
        "weight_kg": weight,
        "arm": arm,
        "stage": stage,
        "grade": grade,
        "n_fx": n_fx,
        "diameter": diameter,
        "wall_min": wall_min.round(1),
        "wall_mean": wall_mean.round(1),
        "wall_max": wall_max.round(1),
        "d2cc_5mm": d2cc_5mm.round(1),
        "d2cc_vol": (d2cc_5mm + d_d2cc_vol).round(1),
        "d2cc_gra": (d2cc_5mm + d_d2cc_gra).round(1),
        "d90_5mm": d90_5mm.round(1),
        "d90_vol": (d90_5mm + d_d90_vol).round(1),
        "d90_gra": (d90_5mm + d_d90_gra).round(1),
    })

    # 少数患者没有壁厚测量
    missing = rng.choice(n, size=max(1, n // 50), replace=False)
    df.loc[missing, ["wall_min", "wall_mean", "wall_max"]] = np.nan
    return df


def make_followup(n=200, seed=22):
    """随访表: 有随访的患者一人一行, 少数患者失访, 不在表里; 行的顺序是打乱的。

    晚期阴道毒性的概率随 5 mm 计划的阴道 D2cc 升高而升高(同样是编的)。
    """
    df = make_cohort(n, seed)
    rng = np.random.default_rng([seed, 2])
    p = 1 / (1 + np.exp(-(-1.9 + 0.13 * (df["d2cc_5mm"] - 68))))
    tox = (rng.random(n) < p).astype(int)
    months = np.clip(rng.gamma(4.0, 9.0, n) + 6, 6, 96).round().astype(int)
    out = pd.DataFrame({"id": df["id"], "fu_months": months, "tox": tox})
    out = out[rng.random(n) > 0.06]                      # 约 6% 的患者失访
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)
