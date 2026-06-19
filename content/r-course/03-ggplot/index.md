---
title: "Lesson 3 · 可视化基础 ggplot2"
summary: "用图层语法画箱线图、散点图、柱状图,导出投稿质量的图片。"
weight: 30
toc: true
---

`ggplot2` 是 R 画图的事实标准,论文里的图大多用它做。它的核心是**图层语法**:一张图 = 数据 + 映射(aes)+ 几何对象(geom),像搭积木一样一层层叠。学会这套心智模型,什么图都能画。

> 接着 [Lesson 2](../02-data-wrangling/)。后面的火山图([Lesson 7](../07-volcano-heatmap/))、生存曲线([Lesson 9](../09-survival/))也都是 ggplot 风格。

## 这一课你会做出什么

- 一张比较两组基因表达的箱线图
- 加上分组配色、分面、标题
- 导出一张 300 dpi、投稿能用的图片

## 0. 准备:加载包

`ggplot2` 已包含在 [Lesson 2](../02-data-wrangling/) 装的 `tidyverse` 里,直接加载:

```r
library(ggplot2)   # 或 library(tidyverse),一起把 dplyr 也加载了
```

## 1. 先造一张练习数据

模拟一份"肿瘤 vs 正常"的基因表达数据,让你能直接跑:

```r
set.seed(1)
expr <- data.frame(
  group    = rep(c("Tumor", "Normal"), each = 30),
  gene_expr = c(rnorm(30, mean = 12, sd = 2),   # 肿瘤组表达偏高
                rnorm(30, mean = 8,  sd = 2)),  # 正常组偏低
  stage    = sample(c("Early", "Late"), 60, replace = TRUE)
)
head(expr)
```

> `set.seed(1)` 固定随机数,保证你跑出来的图和这里一样。

## 2. 图层语法:三件套

每张 ggplot 都从这个模板起步——**数据 → 映射 `aes()` → 几何对象 `geom_*()`**:

```r
ggplot(expr, aes(x = group, y = gene_expr)) +   # 数据 + 把 group/expr 映射到 x/y
  geom_point()                                  # 用点把它画出来
```

> ⚠️ ggplot 用 `+` 连接图层(不是管道 `%>%`)。这是新手最常混的一点。

## 3. 箱线图:比较两组表达

比较分组最常用箱线图,再叠一层抖动点看清每个样本:

```r
ggplot(expr, aes(x = group, y = gene_expr)) +
  geom_boxplot(outlier.shape = NA) +     # 箱线图(先不画离群点,避免和下面的点重复)
  geom_jitter(width = 0.15, alpha = 0.5) # 抖动散点,alpha 设透明度
```

> 🔬 这就是论文里"基因 X 在肿瘤组显著高表达"那类图的雏形。

## 4. 用颜色区分分组

把分组映射到 `fill`(填充色),图例会自动生成:

```r
ggplot(expr, aes(x = group, y = gene_expr, fill = group)) +
  geom_boxplot()
```

> 小贴士:`color` 控制描边颜色,`fill` 控制填充颜色——箱子、柱子用 `fill`,点、线用 `color`。

## 5. 分面:一张拆成多张

按某个变量把图拆成并排的小图,用 `facet_wrap()`:

```r
ggplot(expr, aes(x = group, y = gene_expr, fill = group)) +
  geom_boxplot() +
  facet_wrap(~ stage)        # 按 stage 拆成 Early / Late 两块
```

## 6. 主题与标注:让图能见人

```r
p <- ggplot(expr, aes(x = group, y = gene_expr, fill = group)) +
  geom_boxplot() +
  labs(title = "Gene X expression",
       x = NULL, y = "Expression (log2)") +   # 标题与轴标签
  theme_classic(base_size = 14) +             # 干净的经典主题,字号 14
  theme(legend.position = "none")             # 分组已在 x 轴,隐藏图例
p
```

## 7. 导出投稿级图片

把图对象 `p` 存成文件。`ggsave` 按文件后缀自动选格式:

```r
ggsave("gene_x_boxplot.pdf", plot = p, width = 5, height = 4)            # 矢量,首选
ggsave("gene_x_boxplot.png", plot = p, width = 5, height = 4, dpi = 300) # 位图,投稿要 300 dpi
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

下一课 [Lesson 4 · 统计概念入门](../04-stats-concepts/),在动手做生信分析前补上必须的统计直觉。

## 延伸 Further reading

- ggplot2 官方速查表(Cheatsheet):<https://rstudio.github.io/cheatsheets/data-visualization.pdf>

## 常见报错 Troubleshooting

- 图层之间用了 `%>%` 而不是 `+` → ggplot 必须用 `+` 连接。
- 颜色没变 / 没出图例 → 颜色要写在 `aes()` 里(`aes(fill = group)`)才会按数据上色;写在 `aes()` 外是固定颜色。
- 图不显示 → 在脚本里要把图对象打印出来(单独写一行 `p`),或确认 RStudio 右下 Plots 面板。
- 中文标题变方框 → 需要设置支持中文的字体(如 `theme(text = element_text(family = "PingFang SC"))`)。
