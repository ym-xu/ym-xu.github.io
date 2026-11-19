---
title: "学习与研究"
description: "学习: R 和 Python 两门课。研究: VCB 研究的统计讲解。"
noindex: true
# 不进全站列表、RSS 和 sitemap,只能从链接进入(和 /r-course/ 一样)
outputs: ["HTML"]
build:
  list: never
  render: always
sitemap:
  disable: true
# 卡片:有 url 的可以点进去;没有 url 的显示为"准备中"。section 用来自动数课时。
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
      - title: "Python 课程"
        kicker: "Python"
        status: "准备中"
        text: "正在准备,完成后从这里进入。"
  - title: "研究"
    items:
      - title: "阴道壁厚度与剂量变化"
        url: "/vcb-thickness/"
        kicker: "VCB 研究 · 统计讲解"
        status: "需要密码"
        role: "读懂这份摘要"
        text: "左边是摘要,右边逐页讲解: 先用模拟病人弄懂方法,再看本研究的完整结果和它们的意义。"
        foot: ["14 页", "可交互"]
      - title: "预测模型(ML 系列)"
        kicker: "VCB 研究 · 统计讲解"
        status: "准备中"
        text: "ML 系列的方法和结果,讲解形式和壁厚统计相同。"
---
