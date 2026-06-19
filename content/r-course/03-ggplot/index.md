---
title: "Lesson 3 · 可视化基础 ggplot2"
summary: "用图层语法画箱线图、散点图、柱状图,导出投稿质量的图片。"
weight: 30
toc: true
---

`ggplot2` 是 R 画图的事实标准,论文里的图大多用它做。它的核心是**图层语法**:一张图 = 数据 + 映射(aes)+ 几何对象(geom),像搭积木一样一层层叠。学会这套心智模型,什么图都能画。

> 接着 [Lesson 2](../02-data-wrangling/)。后面的生存曲线([Lesson 9](../09-survival/))、回归诊断图([Lesson 8](../08-linear-regression/))也都是 ggplot 风格。

## 这一课你会做出什么

- 一张比较两组(如两种放疗方案)某指标的箱线图
- 加上分组配色、分面、标题
- 导出一张 300 dpi、投稿能用的图片

## 0. 准备:加载包

`ggplot2` 已包含在 [Lesson 2](../02-data-wrangling/) 装的 `tidyverse` 里,直接加载:

```r
library(ggplot2)   # 或 library(tidyverse),一起把 dplyr 也加载了
```

## 1. 先造一张练习数据

模拟一份"两种放疗方案"的膀胱 D2cc 剂量数据,让你能直接跑:

```r
set.seed(1)
dat <- data.frame(
  arm   = rep(c("EBRT+BT", "BT only"), each = 30),
  d2cc  = c(rnorm(30, mean = 82, sd = 6),    # 联合组剂量偏高
            rnorm(30, mean = 72, sd = 6)),   # 单纯近距离组偏低
  figo  = sample(c("I/II", "III"), 60, replace = TRUE)
)
head(dat)
```

> `set.seed(1)` 固定随机数,保证你跑出来的图和这里一样。`d2cc` = 膀胱 2cc 剂量。

## 2. 图层语法:三件套

每张 ggplot 都从这个模板起步——**数据 → 映射 `aes()` → 几何对象 `geom_*()`**:

```r
ggplot(dat, aes(x = arm, y = d2cc)) +   # 数据 + 把 arm/d2cc 映射到 x/y
  geom_point()                          # 用点把它画出来
```

> ⚠️ ggplot 用 `+` 连接图层(不是管道 `%>%`)。这是新手最常混的一点。

## 3. 箱线图:比较两组

比较分组最常用箱线图,再叠一层抖动点看清每个病人:

```r
ggplot(dat, aes(x = arm, y = d2cc)) +
  geom_boxplot(outlier.shape = NA) +     # 箱线图(先不画离群点,避免和下面的点重复)
  geom_jitter(width = 0.15, alpha = 0.5) # 抖动散点,alpha 设透明度
```

> 🔬 这就是论文里"两种方案的膀胱受量是否不同"那类图的雏形(检验见 [Lesson 6](../06-group-comparisons/))。

## 4. 用颜色区分分组

把分组映射到 `fill`(填充色),图例会自动生成:

```r
ggplot(dat, aes(x = arm, y = d2cc, fill = arm)) +
  geom_boxplot()
```

> 小贴士:`color` 控制描边颜色,`fill` 控制填充颜色——箱子、柱子用 `fill`,点、线用 `color`。

## 5. 分面:一张拆成多张

按某个变量把图拆成并排的小图,用 `facet_wrap()`:

```r
ggplot(dat, aes(x = arm, y = d2cc, fill = arm)) +
  geom_boxplot() +
  facet_wrap(~ figo)        # 按 FIGO 分期拆成 I/II 与 III 两块
```

## 6. 主题与标注:让图能见人

```r
p <- ggplot(dat, aes(x = arm, y = d2cc, fill = arm)) +
  geom_boxplot() +
  labs(title = "Bladder D2cc by treatment",
       x = NULL, y = "Bladder D2cc (Gy)") +   # 标题与轴标签
  theme_classic(base_size = 14) +             # 干净的经典主题,字号 14
  theme(legend.position = "none")             # 分组已在 x 轴,隐藏图例
p
```

## 7. 导出投稿级图片

把图对象 `p` 存成文件。`ggsave` 按文件后缀自动选格式:

```r
ggsave("d2cc_boxplot.pdf", plot = p, width = 5, height = 4)            # 矢量,首选
ggsave("d2cc_boxplot.png", plot = p, width = 5, height = 4, dpi = 300) # 位图,投稿要 300 dpi
```

> ⚠️ **投稿尽量用 PDF 等矢量格式**:放大不糊,期刊也偏好。要 PNG 时务必 `dpi = 300`,默认 72 会被审稿人嫌糊。

## 小结

| 你想做 | 用什么 |
|--------|--------|
| 起一张图 | `ggplot(data, aes(x, y))` |
| 箱线图 / 散点 / 柱状 | `geom_boxplot()` / `geom_point()` / `geom_col()` |
| 分组上色 | `aes(fill = 组)` / `aes(color = 组)` |
| 拆成多张小图 | `facet_wrap(~ 变量)` |
| 标题/轴标签 | `labs()` |
| 换主题/字号 | `theme_classic(base_size = 14)` |
| 导出 | `ggsave("f.pdf", width=, height=, dpi=300)` |

下一课 [Lesson 4 · 统计概念入门](../04-stats-concepts/),在动手做统计分析前补上必须的统计直觉。

## 延伸 Further reading

- ggplot2 官方速查表(Cheatsheet):<https://rstudio.github.io/cheatsheets/data-visualization.pdf>

## 常见报错 Troubleshooting

- 图层之间用了 `%>%` 而不是 `+` → ggplot 必须用 `+` 连接。
- 颜色没变 / 没出图例 → 颜色要写在 `aes()` 里(`aes(fill = arm)`)才会按数据上色;写在 `aes()` 外是固定颜色。
- 图不显示 → 在脚本里要把图对象打印出来(单独写一行 `p`),或确认 RStudio 右下 Plots 面板。
- 中文标题变方框 → 需要设置支持中文的字体(如 `theme(text = element_text(family = "PingFang SC"))`)。
