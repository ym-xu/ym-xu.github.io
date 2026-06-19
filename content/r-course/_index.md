---
title: "R for Cancer Research"
subtitle: "一门为肿瘤研究者准备的 R 入门课 · From zero to your first survival curve"
noindex: true
# 不输出本 section 的 RSS,避免 /r-course/index.xml 公开列出课时
outputs: ["HTML"]
# section 首页本身:照常渲染,但不进任何全站列表 / sitemap
build:
  list: never
  render: always
sitemap:
  disable: true
# 下发给所有课时的默认设置:课程内部可列出(local),但不进全站 RSS/sitemap/首页
cascade:
  build:
    list: local
    render: always
  sitemap:
    disable: true
  noindex: true
---

写给正在做肿瘤研究、但还没怎么写过代码的朋友。这门课的目标不是把你变成程序员,而是让你能**独立把自己的数据从 Excel 读进 R,做出投稿能用的差异表达图和生存曲线**。

每节课都围绕一个肿瘤研究里真实会遇到的问题展开,代码可以直接抄着改。建议按顺序学:**L1–3** 打编程基础,**L4** 建立统计直觉,**L5–10** 进入 Bioconductor 主线(差异表达、富集、生存分析、TCGA),**L11** 把全流程用到你自己的数据上、产出可投稿的结果。

生存分析部分(L9)对标并衔接 [Emily Zabor《Survival Analysis in R》](https://www.emilyzabor.com/survival-analysis-in-r.html) 这份权威教程——前面的课就是为了让零基础的人也能顺利读懂它。

> 🔒 这个页面没有出现在网站导航里,只有拿到链接的人能访问。
