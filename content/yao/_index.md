---
title: "学习与研究"
description: "目标: 用数据和 AI,为每一位病人制定更合适的个性化治疗方案。"
noindex: true
# 不进全站列表、RSS 和 sitemap,只能从链接进入(和 /r-course/ 一样)
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
      - title: "面向医学生的 Python"
        url: "/python/"
        section: "/python"
        kicker: "Python"
        role: "从零开始,用代码处理临床数据"
        text: "读表、整理、Table 1、画论文用的图和基础统计,代码在浏览器里直接运行。"
        foot: ["中文", "零基础起步", "每课有练习"]
      - title: "面向医学生的统计机器学习"
        url: "/stat-ml/"
        section: "/stat-ml"
        kicker: "统计机器学习"
        role: "从回归到预测模型"
        text: "线性回归、逻辑回归、交叉验证、树模型、校准和决策曲线,每个模型先讲它在算什么再用。"
        foot: ["中文", "先学 Python 课", "每课有练习"]
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
      - title: "预测模型(ML 系列)"
        kicker: "VCB 研究 · 统计讲解"
        status: "准备中"
        text: "ML 系列的方法和结果,讲解形式和壁厚统计相同。"
---

**学习**: R、Python、统计机器学习三门课,从写第一行代码,到能读懂、能做出预测模型。

**研究**: VCB 研究的论文复现和统计讲解,把这个目标落到一个具体问题上: 每位病人用哪一种近距离放疗计划更合适。
