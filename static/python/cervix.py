"""主题练习的模拟数据: 宫颈癌图像引导近距离放疗的剂量审计。

所有数字都是编的: 只是让数据的样子像一个科室的审计数据, 不是任何真实病人或研究的结果。
每位病人做盆腔外照射, 再做 4 次高剂量率近距离放疗; 每次近距离都记录了三个剂量。

用法:

    from cervix import make_cervix, make_cervix_followup, load_cervix
    df = load_cervix()             # 整理好的表, 带三列总剂量(第 3 部分起用它)
    fu = make_cervix_followup()    # 随访表

列(一人一行):

    id            病人编号
    age           年龄(岁)
    stage         FIGO 分期: "IB3" "IIB" "IIIB" "IIIC1" "IVA"
    histology     病理: "SCC" 鳞癌, "AC" 腺癌
    hrctv_vol     近距离时高危靶区(HR-CTV)的体积(cm3), 少数病人缺失
    technique     施源器: "IC" 单纯腔内, "IC/IS" 腔内加插植针
    ebrt_gy       外照射总剂量(Gy)
    ebrt_fx       外照射次数
    d90_1..4      第 1-4 次近距离的 HR-CTV D90(Gy, 物理剂量)
    rectum_1..4   第 1-4 次近距离的直肠 D2cc(Gy)
    bladder_1..4  第 1-4 次近距离的膀胱 D2cc(Gy)

load_cervix() 另外加上三列总剂量(外照射加 4 次近距离的 EQD2, 单位 Gy):

    d90_total      HR-CTV D90, alpha/beta = 10
    rectum_total   直肠 D2cc, alpha/beta = 3
    bladder_total  膀胱 D2cc, alpha/beta = 3

随访表(make_cervix_followup)的列:

    id          病人编号
    fu_months   随访时间(月)
    rect_tox    晚期直肠副作用: 1 = 出现过 2 级及以上, 0 = 没有
"""

import numpy as np
import pandas as pd


def eqd2(d, n, ab):
    """每次剂量 d(Gy)、照射 n 次、alpha/beta = ab 时的 EQD2(Gy)。d 可以是一个数, 也可以是一列。"""
    return n * d * (d + ab) / (2 + ab)


def make_cervix(n=200, seed=21):
    """造一份模拟的审计数据: n 位病人, 一人一行。seed 相同, 得到的数据就完全相同。"""
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(50, 12, n), 26, 82).round().astype(int)
    stage = rng.choice(["IB3", "IIB", "IIIB", "IIIC1", "IVA"], n, p=[0.12, 0.40, 0.12, 0.31, 0.05])
    histology = rng.choice(["SCC", "AC"], n, p=[0.82, 0.18])

    # 近距离时的 HR-CTV 体积: 分期越晚, 残留的肿瘤越大
    shift = pd.Series(stage).map({"IB3": -0.25, "IIB": 0.0, "IIIB": 0.30, "IIIC1": 0.05, "IVA": 0.45}).to_numpy()
    vol = np.clip(np.exp(rng.normal(np.log(30) + shift, 0.40, n)), 8, 110)

    # 肿瘤越大, 越可能加插植针
    needles = rng.random(n) < 1 / (1 + np.exp(-(vol - 40) / 10))
    technique = np.where(needles, "IC/IS", "IC")

    long_course = rng.random(n) < 0.10                 # 少数病人外照射 50.4 Gy / 28 次
    ebrt_gy = np.where(long_course, 50.4, 45.0)
    ebrt_fx = np.where(long_course, 28, 25)

    # 每次近距离的剂量: 每位病人有自己的水平, 每一次再有一点波动。
    # 单纯腔内时, 肿瘤越大靶区剂量越难给够; 加了插植针, 大肿瘤也能给够, 直肠也能躲开一些。
    def fractions(mean, sd_patient, sd_fraction, low, high):
        patient = rng.normal(0, sd_patient, n)
        return [np.clip(mean + patient + rng.normal(0, sd_fraction, n), low, high).round(1) for _ in range(4)]

    d90 = fractions(np.where(needles, 8.1 - 0.008 * (vol - 30), 7.9 - 0.040 * (vol - 30)), 0.45, 0.35, 4.5, 10.0)
    rectum = fractions(np.where(needles, 3.5, 3.9) + 0.004 * (vol - 30), 0.55, 0.45, 1.5, 6.5)
    bladder = fractions(np.where(needles, 4.8, 5.0), 0.70, 0.50, 2.5, 8.0)

    df = pd.DataFrame({
        "id": [f"C{i:03d}" for i in range(1, n + 1)],
        "age": age,
        "stage": stage,
        "histology": histology,
        "hrctv_vol": vol.round(1),
        "technique": technique,
        "ebrt_gy": ebrt_gy,
        "ebrt_fx": ebrt_fx,
    })
    for name, doses in [("d90", d90), ("rectum", rectum), ("bladder", bladder)]:
        for k in range(4):
            df[f"{name}_{k + 1}"] = doses[k]

    # 少数病人近距离时没有做 MRI, 体积没有记录
    missing = rng.choice(n, size=max(1, n // 40), replace=False)
    df.loc[missing, "hrctv_vol"] = np.nan
    return df


def add_totals(df):
    """加上三列总剂量: 外照射的 EQD2 加上 4 次近距离各自的 EQD2。"""
    out = df.copy()
    per_fraction = out["ebrt_gy"] / out["ebrt_fx"]
    ebrt = {10: eqd2(per_fraction, out["ebrt_fx"], 10), 3: eqd2(per_fraction, out["ebrt_fx"], 3)}
    for name, ab in [("d90", 10), ("rectum", 3), ("bladder", 3)]:
        out[f"{name}_total"] = ebrt[ab] + sum(eqd2(out[f"{name}_{k}"], 1, ab) for k in range(1, 5))
    return out


def load_cervix(n=200, seed=21):
    """整理好的审计数据, 带三列总剂量。和第 2 部分清理、计算以后得到的表相同。"""
    return add_totals(make_cervix(n, seed))


def make_cervix_followup(n=200, seed=21):
    """随访表: 有随访的病人一人一行, 少数病人失访, 不在表里; 行的顺序是打乱的。

    晚期直肠副作用的概率随直肠 D2cc 的总 EQD2 升高而升高(同样是编的)。
    """
    df = load_cervix(n, seed)
    rng = np.random.default_rng([seed, 2])
    p = 1 / (1 + np.exp(-(-2.6 + 0.11 * (df["rectum_total"] - 65))))
    tox = (rng.random(n) < p).astype(int)
    months = np.clip(rng.gamma(3.5, 9.0, n) + 6, 6, 90).round().astype(int)
    out = pd.DataFrame({"id": df["id"], "fu_months": months, "rect_tox": tox})
    out = out[rng.random(n) > 0.07]                      # 约 7% 的病人失访
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)
