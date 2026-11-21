---
title: "学习与研究"
description: "目标: 用数据和 AI,为每一位患者制定更合适的个性化治疗方案。"
noindex: true
# 不进全站列表、RSS 和 sitemap,只能从链接进入。Projects 里的项目页直接链接三门课,不链接这一页(用户 2026-11-21 定)
outputs: ["HTML"]
build:
  list: never
  render: always
sitemap:
  disable: true
# 卡片:有 url 的可以点进去;没有 url 的显示为"准备中"。section 用来自动数课时:
# 课程的 _index.md 里有大纲(parts)时显示"已上线 N / M 课",否则显示"N 节课"。
groups:
  - title: "学习"
    items:
      - title: "R for Cancer Research"
        url: "/r-course/"
        section: "/r-course"
        kicker: "R"
        role: "一门为肿瘤研究者准备的 R 入门课"
        text: "从安装、数据整理、作图到统计检验、回归、生存分析和 EQD2,代码在浏览器里直接运行。"
        foot: ["中文", "零基础起步"]
      - title: "使用 Python 进行放疗临床数据分析"
        url: "/python/"
        section: "/python"
        kicker: "Python"
        role: "把已经会做的分析,换成代码来做"
        text: "读表、清理、Table 1、论文用的图和常用的检验。例子是近距离放疗的剂量和副作用;用过 R 或 SPSS 的话,有一份写法对照。"
        foot: ["中文", "不需要编程基础", "配一套主题练习"]
      - title: "使用机器学习方法进行放疗临床预测"
        url: "/stat-ml/"
        section: "/stat-ml"
        kicker: "统计机器学习"
        role: "从\"哪些因素有关\"到\"对这位患者准不准\""
        text: "回归模型的另一种用法: 预测。怎样在没见过的患者身上检验模型,怎样看校准和决策曲线,多少患者才够,怎样按 TRIPOD+AI 写成论文。"
        foot: ["中文", "先学 Python 课", "算法配分步动态图", "配一套主题练习"]
  - title: "研究"
    items:
      - title: "D90、D2cc 与晚期阴道毒性"
        url: "/vcb-reproduction/"
        kicker: "VCB 研究 · 论文复现"
        status: "需要密码"
        role: "从原始数据重算这篇论文"
        text: "左边是已发表论文(Cancers 2026)的摘要,右边逐页讲每张表用的方法,以及和论文逐个比对的结果。"
        foot: ["R 复现", "逐格核对", "可交互"]
      - title: "阴道壁厚度与剂量变化"
        url: "/vcb-thickness/"
        kicker: "VCB 研究 · 统计讲解"
        status: "需要密码"
        role: "读懂这份摘要"
        text: "左边是摘要,右边逐页讲解: 先用模拟病人弄懂方法,再看本研究的完整结果和它们的意义。"
        foot: ["逐页讲解", "可交互"]
      - title: "哪些病人值得换一种计划"
        url: "/vcb-ml/"
        kicker: "VCB 研究 · 预测模型讲解"
        status: "需要密码"
        role: "读懂这项预测研究"
        text: "从 5 mm 改用 Volume 或 Graphical: 有没有差别,能不能预测,有什么用,靠不靠得住。先用模拟病人弄懂方法,再看本研究的结果。"
        foot: ["逐页讲解", "可交互"]
---

**起点**: 临床研究的经验已经有了: 回顾性队列的数据收集,DVH 参数的提取,晚期副作用的分级评估,用 R 和 SPSS 做的描述统计和生存分析。

**终点**: 自己建立、验证并报告一个副作用的预测模型,用它给每位患者估计风险,帮助选择近距离放疗的计划。

**路线**:

1. [使用 Python 进行放疗临床数据分析](/python/)(Python 课): 把已经会做的分析(Table 1、画图、常用的检验)换成代码来做,配一套宫颈癌近距离放疗的主题练习。
2. [使用机器学习方法进行放疗临床预测](/stat-ml/)(统计机器学习课): 从回归模型讲到预测模型的检验、校准、决策曲线、样本量和论文写作。
3. **研究**: 同样的方法用在 VCB 的数据上,回答一个具体的问题: 每位患者用哪一种近距离放疗计划更合适。

R 课是另一门独立的入门课,不在这条路线上;用到 R 的时候可以查。
