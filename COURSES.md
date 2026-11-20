# 两门课的写作规范

给继续写这两门课的人(和 Claude 会话)看。课程和目录页 `/yao/` 都**只给拿到链接的人看**: 不在站点导航里,带 noindex,
站内任何公开页面都不链接它们(用户 2026-11-21 定)。

Projects 里有一页讲这套教程(`content/projects/clinical-data-tutorials.md`,英文,Data Science Tutorials for Clinicians):
只讲怎么设计、怎么教,**不放任何通向课程或 `/yao/` 的链接**。改课程的设计或教法时,看一下这一页的说法还对不对。

| 课程 | 地址 | 课文 | 状态 |
|---|---|---|---|
| 面向医学生的 Python(7 课) | `/python/` | `content/python/` | 7 课全部上线(2026-11-20) |
| 面向医学生的统计机器学习(14 课) | `/stat-ml/` | `content/stat-ml/` | 第 4 课已上线 |
| 主题练习: 宫颈癌近距离放疗的剂量审计(6 部分) | `/python/practice/` | `content/python/practice/` | 6 部分全部上线(2026-11-21) |

大纲在各自的 `_index.md` 的 `parts` 里;没写的课在目录里显示"准备中"。第 n 课的 `weight` 是 n × 10。
两门课共用模板 `layouts/course/`(`type: course`)。R 课(`/r-course/`)是另一套,不要改它的写法。

## 这两门课是写给谁的

用户的要求(2026-11-21): 课程的描述按读者的背景来写。读者的起点和目标:

- 做临床研究的医生、博士生,方向是妇科肿瘤的近距离放疗: 做过回顾性队列的数据收集、DVH 参数的提取、晚期副作用的分级评估。
- 用过 R 和 SPSS;做过描述统计、单因素和多因素分析、Kaplan–Meier、log-rank 检验、Cox 回归。没有假设会 Python 或机器学习。
- 目标: 自己建立、验证并报告副作用的预测模型,用于估计每位病人的风险和选择近距离放疗的计划。
- 论文用英文写: 术语给英文,"写进论文"一栏给英文句式。

写描述和课文时照这个来:

- 写"这门课从什么起点出发"和"不需要什么",不写"读者不会什么";不写任何个人信息(姓名、单位、经历)。
- 从读者已经会的东西讲起: Python 课说"把已经会做的分析换成代码"(首页有 R 和 Python 的写法对照,只列课里真的讲了的函数);
  统计机器学习课从"哪些因素有关"(HR、OR、P 值)讲到"对这位病人准不准"(区分度、校准、决策曲线)。
- 例子的临床背景不用向读者解释(阴道近距离放疗是读者自己的领域),给外行的解释折叠起来。
- 统计机器学习课的 14 课没有随访时间和删失的结局(Kaplan–Meier、Cox 那一类),首页写明了;大纲要不要加这一课由用户定。

## 设计是从哪里学来的

用户的要求(2026-11-20): 参考知名的教学博客和课程来**设计**课程,不是在课里放它们的链接;每个回归模型、预测模型都要有讲解。
读过的: MLU-Explain、Google Machine Learning Crash Course、Software Carpentry、Kaggle Learn、《动手学深度学习》、
Teaching Tech Together。学来的做法,每课都要做到:

1. **课前概览**(Carpentry、MLCC): 这一课回答的问题(2–3 个,问句)、学完能做什么(动词开头)、用时、先会什么。
   写在 front matter 的 `questions`、`objectives`、`duration`、`prereq` 里,模板画成页首的框。
2. **从一个上一课回答不了的问题开始**(MLCC),一个场景贯穿全课(MLU)。全课程用同一个模拟队列。
3. **先看见,再起名字**(MLU 的"fail first, then name it"): 先让读者在图里或代码里看到现象,再给术语。
   术语第一次出现时加粗,中文后面括号里给英文。
4. **能动手的图**(MLU、MLCC): 讲模型或方法的地方配交互图,一张图一个控件、一个读数,只说明一件事;
   图前面一句祈使句告诉读者做什么。大约每 300–500 字有一次互动(图、Run、想一想、练习)。
5. **每个模型先讲它在算什么**(StatQuest、MLCC): 几步具体操作,公式对应到读者已经认识的东西,公式后面紧跟一个
   代入真实数字的手算例子;然后**先自己用几行代码写一遍,再用库函数**,两者结果对得上(d2l 的"从零实现、简洁实现")。
6. **先猜再验证**(MLCC 的课前题): 在演示或代码之前放一道"先猜一猜",反馈里不直接给答案,而是让读者去运行。
7. **想一想**(Carpentry、Teaching Tech Together): 选择题的每个错误选项对应一种具体的误解,点开后解释这种误解;
   半对的选项用"……,但是……"。大约每 10–15 分钟一道。
8. **练习: 讲解用一套数据,练习换一套**(Kaggle Learn): 统计机器学习课里,讲解用体积优化计划(`vol`),
   练习用图形优化计划(`gra`),每讲完一步就让读者对图形优化再做一遍;一题只有一两个空,当场检查,
   答错时说出"现在是多少、应该是多少、可能错在哪"。提示一句话,参考答案折叠。
9. **课末**: 小结(几条结论)、本课术语(中英对照,为写英文论文准备)、这一课没讲什么(MLU 会明说跳过了什么)、
   写进论文(英文 Methods 的写法、审稿人可能问什么)、常见报错、延伸阅读(只放官方文档和可引用的论文)、下一课。

10. **贴近医学生**(用户的要求): 例子从人人熟悉的量开始(BMI、住院天数、随访时间、血压),主线的临床背景在课程首页用一段话讲清
    (`content/python/_index.md` 的"例子的临床背景"),正文里第一次用到 EQD2、D2cc 时再解释一句;打比方用临床上的事(查房前把化验单夹进病历、降压药前后量血压)。
11. **必要的术语给英文**(用户的要求): 第一次出现时写成"包(package)""置信区间(confidence interval, CI)",标题里也可以这样写;课末的术语表再汇总一次。
12. **和我们自己的研究挂钩,但只用已发表的数字**(用户: 可以放一点点之前研究的内容当例题或练习): 已发表的论文(Qiang 等,*Cancers* 2026;18:2350)
    里的数字可以用,要写明出处(第 2 课 Table 5 的三个均值,第 3 课 Table 1 的节选,第 6 课 68 Gy 的四格表,卡方检验算出的 P = 0.055 和论文印的一致);
    未发表的结果(壁厚统计、ML 系列)一个数字也不放,只借用分析的做法,在模拟队列上重做(三套计划两两比较加 Holm、bootstrap 区间、Spearman)。
13. **做对了有反馈**(用户的要求): 练习通过、想一想选对,放一小段烟花(`assets/js/fireworks.js`);一课的练习全部通过时放大的。

14. **课后练习独立成一套"主题练习"**(用户 2026-11-21: 课程和课后练习拆开;练习要贴近医生的专业,不必和她的研究相同;做成一个完整的主题):
    `/python/practice/`(`content/python/practice/`),主题是**宫颈癌图像引导近距离放疗的剂量审计**(用户选的),一份模拟的科室数据,
    六个部分对应 Python 课第 1–6 课,从算一位病人的 EQD2 做到写出审计结论。课文里只留跟在讲解后面的小练习,每课末尾有一个框链接到对应的部分。
    - 形式(用户问过要不要像 DataCamp;结论: 课文保持长文讲解,练习用一步一屏): 每个部分一页,开头一个准备数据的代码块,然后
      `<div class="steps" data-steps>` 里放若干 `<div class="step exercise">`(代码题)和 `<div class="step">`(里面一个想一想)。
      `assets/js/course-steps.js` 一次只显示一步,顶上是"第 k / N 步"和每一步的圆点,做对了"下一步"变成实心;可以"全部展开";
      宽屏上任务在左、代码在右。位置记在浏览器里。
    - 题目: 没有讲解;代码题只给一行注释的空代码框(新的技巧才给带 `___` 的骨架),提示分两级(思路、用到的写法),参考答案折叠;
      检查要对常见的错法给出针对性的话;每部分有一两道解读题;第 5、6 部分各有一道写作题(`<div class="step exercise homework">` 加
      `<textarea class="hw-write" data-save>`,不自动检查,给参考写法和自查清单)。
    - 数据: `static/python/cervix.py`(`make_cervix`、`load_cervix`、`make_cervix_followup`,种子 21),`cervix.csv` 是故意弄乱的导出文件
      (一位病人录了两次、施源器有四种写法、一个剂量丢了小数点),第 2 部分清理它;第 3 部分起用 `load_cervix()`。数字都是编的。
      主线: 混在一起比,两种施源器的靶区剂量没有差别;按肿瘤体积分层以后差别很大(适应证混杂)。审计标准引自 EMBRACE II
      (Pötter 等,*Clin Transl Radiat Oncol* 2018;9:48–60,Table 4,原文核对过): D90 > 85 Gy 是硬性下限,直肠 < 65 Gy、膀胱 < 80 Gy 是计划目标。
    - 给统计机器学习课做课后练习时,沿用同一份审计数据往下做(比如预测哪些病人达不到 85 Gy)。

讲统计方法的小节仍然用壁厚讲解网页那套写法(用户定的): 框里写背景、目标、做法;做法之后紧接着讲
"它在算什么"(几步)、"为什么用它"、"打个比方";演示框四行(设定、你做什么、你会看到、这说明);最后"回到目标"和"接下来"。
不讲方法的课(比如 Python 第 1 课)按内容直接写。

## 课文里的几种块

可以运行的代码写成普通的 `python` 代码块;同一页的代码块共用变量,要按顺序运行。第一行写 `#| norun` 的代码块只显示、不运行。
其余的块直接写 HTML(块的开头和结尾各占一行,中间的 Markdown 前后留空行):

```html
<div class="frame"> **背景** … **目标** … **做法** … </div>

<div class="demo"> **设定** … **你做什么** … **你会看到** … **这说明** …
<div class="widget" data-widget="overfit">没有 JavaScript 时显示的话</div>
</div>

<div class="quiz">
<p><b>想一想</b>: 问题</p>
<button class="quiz-opt" data-why="为什么不对">选项</button>
<button class="quiz-opt" data-correct data-why="为什么对">选项</button>
</div>

<div class="exercise">
**练习 1 · 轮到你: 标题** 和题目
(一个 python 代码块: 起始代码,空用三个下划线 ___;还有空没填时不运行)
(一个 pycheck 代码块: 不显示,读者的代码运行之后再运行它)
<details><summary>提示</summary> 一句话 </details>
<details><summary>参考答案</summary> 一个完整的 python 代码块 </details>
</div>
```

`pycheck` 代码块的例子:

```python
need("x")                                   # x 还没定义就提示
if close(x, 2.46, 0.006):
    ok("对了: ……")
fail(f"现在是 {x:.2f},应该是 2.46。……")
```

- 检查代码里可以用读者定义的所有名字,加上 `ok(msg)`、`fail(msg)`(调用后检查就结束)、`need(名字…)`、`close(a, b, tol)`。
- 通过的练习会打 ✓,进度记在读者浏览器的 localStorage 里(折叠的参考答案通过不算)。
- 交互图在 `assets/js/course-widgets.js`: `overfit`(加 `data-cv` 多一根交叉验证的条)、`kfold`、`repeatcv`(统计机器学习第 4 课),
  `ci`(100 个研究的 95% 区间,Python 第 5 课)、`shuffle`(打乱分组算 P 值,Python 第 6 课)。
  只用滑块和按钮,不做拖动(手机上拖动容易和滚动打架);颜色用站点的 CSS 变量,深色模式自动跟随;
  数据是固定种子的模拟病人,所有读者看到的数字相同。加新图时在这个文件里加一个函数,登记到 `FIGURES`。
- Python 运行器在 `assets/js/pyodide-run.js`(Pyodide,第一次点 Run 才下载);报错只显示读者自己的那几行,下面加一句中文提示。
- 模拟队列在 `static/python/cohort.py`: `make_cohort(n=200, seed=22)` 和随访表 `make_followup()`,两门课共用。数字是编的,不是任何研究的结果。
  给队列加新的列时另用一串随机数(`np.random.default_rng([seed, k])`),并且不要带缺失值,否则已上线的课里的数字会全部改变。
  `cohort.csv`、`followup.csv`、`cohort.xlsx` 由 `tools/make_course_files.py` 生成(`cohort.csv` 里故意留了一列被 Excel 公式算成 0 的 `wall_range`,第 2 课用);
  运行器启动时把两个 CSV 放进浏览器里的文件系统,所以课里可以 `pd.read_csv("cohort.csv")`。
- 引号: `pycheck` 里的提示如果有双引号,Python 的字符串就用单引号;想一想的 `data-why="..."` 里的双引号要写成 `&quot;`,按钮文字里的 `<` 写成 `&lt;`
  (`tools/lint_lesson.py` 会检查,并自动改 `data-why`)。

## 写作规则

- 课程不加密,所以**不放任何未发表的研究数字**,不放病人数据;例子全部用模拟队列。
- 正文里的每个数字都要来自实际运行: 先跑,再写"你会看到"。改了代码或换了 Pyodide 版本,要重新核对。
- 图里的文字用英文(论文的图要英文,浏览器里的 Python 也没有中文字体);图要窄(不超过约 4.5 英寸),多张图上下排。
- 标点: 半角的 `, ; : ( ) ?`,逗号和分号后面接中文时不空格,冒号后空一格;句号和顿号用全角。
- 指病人用"这位病人""这一例",不用"他/她"。不用一个术语去解释另一个术语。
- 用户多半在手机上看: 页面和交互都要在 390 px 宽度、触摸操作下检查。

## 检查和发布

```bash
python tools/lint_lesson.py content/python/*/index.md content/stat-ml/*/index.md    # 引号、标点、代码能不能编译
python tools/run_lesson.py content/python/05-uncertainty/index.md                   # 用本机的 Python 快速跑一遍(需要 pandas、scipy、matplotlib)
hugo --minify -d /tmp/public
python tools/check_course.py /tmp/public python/01-first-steps --expect-errors 6     # 第 1 课有 6 段故意出错的代码
python tools/check_course.py /tmp/public python/05-uncertainty stat-ml/04-cross-validation --expect-errors 0
```

写一课的顺序: 先用一个草稿脚本把代码和练习的答案跑出来,再写课文(数字照着输出写),然后 `lint_lesson`、`run_lesson`,最后 `check_course`。
`run_lesson.py` 用的是本机的包,版本和浏览器里的 Pyodide 可能不同,所以它不能代替 `check_course.py`。

脚本在手机宽度的无头浏览器里点每一道"想一想"、运行每个代码块和每个参考答案、操作每张交互图,
把读者会看到的东西打印出来,用来和正文里的数字逐个核对(需要 playwright 和 Chromium;页面通过请求拦截提供,不开 HTTP 端口)。
提交并推送到 `main` 后,GitHub Actions 约半分钟发布;发布后用 `python tools/check_course.py LIVE …` 再跑一遍。
一课上线后不用改 `/yao/`: 卡片上的"已上线 N / M 课"是自动数的。
