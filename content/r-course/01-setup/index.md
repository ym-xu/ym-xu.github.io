---
title: "Lesson 1 · 环境搭建 Setup"
summary: "安装 R 与 RStudio,认识工作界面,把第一个 CSV 读进来。"
weight: 10
toc: true
---

这一课不讲分析,只做一件事:**把 R 装好,把你自己的一份数据读进去**。这是后面所有课的起点。完全没写过代码也没关系,跟着抄就行。

## 这一课你会做出什么

- 一个装好 R + RStudio、能运行代码的环境
- 看懂 RStudio 的四块面板,知道代码该敲在哪
- 把一个 CSV 文件读进 R,并用三行命令看清它长什么样

## 1. 为什么肿瘤研究常用 R

- **免费、开源**,不像 SPSS/GraphPad 要付费授权
- **临床统计齐全**:基线表 Table 1、组间比较、逻辑回归、Cox 生存分析、竞争风险……临床研究要用的方法都有成熟的包
- **出图质量高**,图可以做到投稿级别,而且分析过程可记录、可复现

> 🔬 你在论文里看到的 Kaplan-Meier 生存曲线、森林图、患者基线表,绝大多数都是 R 做的。

## 2. 安装 R 和 RStudio

要装**两个**东西,顺序不能反:

1. 先装 **R**(真正干活的"引擎"):打开 <https://cran.r-project.org/>,选你的系统(Windows / macOS)下载安装。
2. 再装 **RStudio**(好用的"驾驶舱"界面):打开 <https://posit.co/download/rstudio-desktop/>,下载免费版 RStudio Desktop。

> ⚠️ R 和 RStudio 是两个软件。RStudio 只是给 R 套了个好用的外壳,**必须先装 R**,否则 RStudio 打开会找不到引擎。

装完后,日常**只打开 RStudio** 就行。

## 3. 认识 RStudio 的四块面板

打开 RStudio,你会看到四个区域(第一次可能只有三个,新建一个脚本就齐了):

| 面板 | 作用 |
|------|------|
| **Source(左上)** | 写脚本的地方,代码存成 `.R` 文件,可反复运行 |
| **Console(左下)** | R 的命令行,代码在这里真正执行、显示结果 |
| **Environment(右上)** | 显示当前内存里有哪些变量、数据 |
| **Plots / Files(右下)** | 显示画出来的图、浏览文件 |

新手建议:**代码写在 Source 面板**(菜单 File → New File → R Script),把光标放在某一行按 `Ctrl/Cmd + Enter` 就能把这行送到 Console 运行。

## 4. 第一行代码

在 Console 里(或 Source 里运行)试试:

```r
1 + 1
print("hello, R")
```

第一行会返回 `[1] 2`,第二行会打印那串字。能看到结果,说明环境正常。

## 5. 变量与向量

R 用 `<-` 把右边的值"存进"左边的名字(也可以用 `=`,但社区习惯 `<-`):

```r
age <- 58           # 存一个数
age                 # 直接打名字就能看到它

ages <- c(58, 63, 47, 71)   # c() 把多个值拼成一个"向量"
mean(ages)                  # 求平均
```

> 向量(vector)= 一串同类型的值。R 里几乎所有东西都是从向量开始的。

## 6. data.frame:R 里的"表格"

真实数据是带行列的表。R 里管它叫 **data.frame**,和 Excel 的一张表一样:行是样本,列是变量。先看一个内置例子:

```r
head(mtcars)        # 看前 6 行
str(mtcars)         # 看每列的类型和概况
```

## 7. 读入你自己的 CSV

把你的数据从 Excel 另存为 `.csv`,然后:

```r
clinical <- read.csv("clinical.csv")   # 读进来,存成 clinical
head(clinical)      # 看前几行
str(clinical)       # 每列是什么类型(数字?文字?)
summary(clinical)   # 每列的统计概况
```

> ⚠️ **新手最常卡的点:工作目录**。`read.csv("clinical.csv")` 只会在"当前工作目录"里找这个文件。用 `getwd()` 看当前目录,用 `setwd("你的路径")` 切换;或者在 RStudio 里 Session → Set Working Directory → Choose Directory 点一下。找不到文件报 `cannot open file` 基本都是这个原因。

## 小结

| 你想做 | 用什么 |
|--------|--------|
| 存一个值 / 一串值 | `x <- 5` / `c(1, 2, 3)` |
| 读 CSV | `read.csv("file.csv")` |
| 看数据长什么样 | `head()` / `str()` / `summary()` |
| 查/改工作目录 | `getwd()` / `setwd()` |

下一课 [Lesson 2 · 数据整理](../02-data-wrangling/),我们用 tidyverse 把这张表筛选、整理、合并成能分析的样子。

## 常见报错 Troubleshooting

- `Error: object 'clinical' not found` → 这个变量还没创建,或拼错了名字(R 区分大小写)。
- `cannot open file 'clinical.csv': No such file or directory` → 工作目录不对,见第 7 步的 ⚠️。
- 中文乱码 → 读取时指定编码:`read.csv("clinical.csv", fileEncoding = "UTF-8")`。
- `could not find function "xxx"` → 该函数所在的包还没加载(`library()`),base R 的函数不会报这个。
