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

写给正在做临床肿瘤研究(放疗 / 妇科肿瘤 / 临床结局)、但还没怎么写过代码的朋友。这门课的目标不是把你变成程序员,而是让你能**独立把自己的回顾性队列从 Excel 读进 R,做出投稿能用的基线表、回归表和生存曲线**。

每节课都围绕一个临床研究里真实会遇到的问题展开,代码可以直接抄着改。建议按顺序学:**L1–3** 打编程基础(R、数据整理、画图),**L4** 建立统计直觉,**L5–8** 临床统计主线(基线表 Table 1、组间比较、逻辑回归、线性回归),**L9–10** 生存与竞争风险分析,**L11** 把全流程用到你自己的数据上、产出可投稿的结果。

生存分析部分(L9)对标并衔接 [Emily Zabor《Survival Analysis in R》](https://www.emilyzabor.com/survival-analysis-in-r.html) 这份权威教程——前面的课就是为了让零基础的人也能顺利读懂它。

> ▶️ **可以在浏览器里直接运行代码**:每个代码块右上角有 **Run** 按钮,点一下就地运行、显示结果和图,代码也能改了再跑——不用先在电脑上装 R。
> - 首次点 Run 会下载 R 运行时和所需的包,可能要等 **10–60 秒**(之后会快);
> - 少数操作浏览器里跑不了:读取你自己电脑上的文件(如 `read.csv("你的数据.csv")`)、导出 Word 文档——这些需要在本地装好 R/RStudio 再做。课程末尾(L11)会讲本地安装。

> 🔒 这个页面没有出现在网站导航里,只有拿到链接的人能访问。
