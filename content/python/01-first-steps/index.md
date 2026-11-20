---
title: "Lesson 1 · 第一次运行 Python First Steps"
summary: "在浏览器里运行第一行代码;变量、列表、字典、函数和包;看懂报错,读懂别人或 AI 写的代码。"
weight: 10
aliases: ["/python-ml/01-first-steps/"]
toc: true
duration: "约 90 分钟,可以分三次学: 第 1–3 节,第 4–7 节,第 8–10 节"
questions:
  - "不装任何软件,怎么运行 Python?"
  - "一位病人的几个数怎么存?一个公式怎么写成可以反复用的函数?"
  - "代码报错了怎么办?别人或 AI 写的代码怎么读?"
objectives:
  - "在浏览器里运行 Python,改了代码再运行"
  - "用变量、列表和字典存一位病人的三套计划"
  - "用循环和条件找出达到 68 Gy 的计划"
  - "把 EQD2 公式写成一个可以反复用的函数"
  - "读懂 6 种最常见的报错,并把它们改好"
  - "一行一行读懂别人或 AI 写的代码"
---

整理数据、做基线表、画图、做统计检验,这门课都用 Python 来做。第一步是能读、能改、能写几行 Python。

这一课不讲统计,只讲 Python 本身: 怎么给一个数起名字,怎么把一位病人的几个数放在一起,怎么把一个公式写成函数,出错了怎么看。例子都是放疗里熟悉的计算: EQD2、68 Gy 阈值、一位病人的三套计划。

> 🔬 不用先装任何东西。每个代码块右上角都有 **▶ Run**,点一下就在浏览器里运行。

## 1. 运行第一行代码

点下面代码块右上角的 **▶ Run**。第一次要下载 Python,大约十几秒,之后就快了。

```python
print("你好, Python")
1 + 1
```

下面会出现两行: `你好, Python` 和 `2`。`print(...)` 把括号里的东西显示出来;代码块**最后一行**如果是一个值,也会自动显示,就像计算器。(这一点只在网页和 Jupyter 里成立。以后在自己电脑上把代码存成 `.py` 文件运行时,要显示什么都得用 `print`。)

代码可以直接改。把 `1 + 1` 改成 `7 * 3`,再点一次 Run 试试。

Python 可以当计算器用: `+ - * /` 是加减乘除,`**` 是乘方,括号和数学里一样。单纯近距离 7 Gy × 3 次,阴道黏膜按 α/β = 3 算的 EQD2 是:

```python
3 * 7 * (7 + 3) / (2 + 3)
```

结果是 `42.0`。用 `/` 做除法,结果总是带小数点。

**同一页上的代码块共用同一个 Python**: 前面代码块里定义的名字,后面的代码块可以接着用。所以请**按顺序运行**。刷新页面会全部清空,从头再来。

## 2. 变量: 给一个数起名字

上面那行算式过一个月再看,就不知道 7 和 3 各是什么了。给它们起名字:

```python
d = 7        # 每次剂量(Gy)
n = 3        # 次数
ab = 3       # α/β,阴道黏膜按 3
eqd2 = n * d * (d + ab) / (2 + ab)
eqd2
```

几条规矩:

- `=` 的意思是"把右边算出来的结果存进左边这个名字",不是数学里的"等于"。所以 `n = n + 1` 是对的: 把 n 加 1 再存回 n。
- `#` 后面是**注释**,Python 不管它,是写给人看的。好的注释写"这个数是什么",比如单位。
- 名字区分大小写,`D` 和 `d` 是两个名字。名字用英文字母、数字和下划线,不能以数字开头: `d2cc_5mm` 可以,`5mm_d2cc` 不行。

还有一条和 Excel 不一样,最容易出错。先猜一猜:

<div class="quiz">
<p><b>想一想</b>: 上面算出 <code>eqd2</code> 是 42.0。现在运行 <code>d = 6</code>,再看 <code>eqd2</code>,它是多少?</p>
<button class="quiz-opt" data-why="Excel 的单元格公式会自动重算,Python 不会。运行下面的代码就能看到。">按 6 Gy 重新算出的 32.4</button>
<button class="quiz-opt" data-correct data-why="eqd2 里存的是当时算出来的数 42.0,不是公式。d 后来变了,它不会跟着变。">还是 42.0</button>
<button class="quiz-opt" data-why="不会报错: d 和 eqd2 都已经定义过了,只是 eqd2 不会自动重算。">报错</button>
</div>

运行下面的代码验证:

```python
d = 6
eqd2
```

`eqd2` 还是 `42.0`,并没有因为 `d` 变成 6 而跟着变。Excel 的单元格公式会自动重算,Python 不会: `eqd2` 里存的是**当时算出来的数**,不是公式。要得到新结果,得把算 `eqd2` 的那一行再运行一次。第 7 节的"函数"就是为了解决这个问题。

**打个比方**: 变量像贴在值上的标签。`d = 7` 是把写着 `d` 的标签贴到 7 上;再写 `d = 6`,标签挪到了 6 上。`eqd2` 这张标签贴着的还是当初算出来的 42.0,没有人去动它。

## 3. 数字、文字和真假

Python 里的值有不同的类型。最常用的四种:

```python
arm = "EBRT+VCB"      # 文字(字符串),要加引号
age = 63              # 整数
d2cc = 71.4           # 小数
high = d2cc >= 68     # 比较的结果: True 或 False
print(type(arm), type(age), type(d2cc), type(high))
print(high)
```

`str` 是文字,`int` 是整数,`float` 是小数,`bool` 是真假。比较用 `>`、`>=`、`<`、`<=`,**"等于"要写两个等号 `==`**(一个等号是存值),"不等于"写 `!=`。几个条件可以用 `and`、`or`、`not` 连起来。

<div class="quiz">
<p><b>想一想</b>: <code>arm</code> 里存着治疗方式。要判断它是不是 <code>"VCB"</code>,应该写哪一个?</p>
<button class="quiz-opt" data-why="一个等号是存值: 这一句会把 arm 改成 &quot;VCB&quot;,而不是去比较。">arm = "VCB"</button>
<button class="quiz-opt" data-correct data-why="两个等号才是比较,结果是 True 或 False。文字要加引号。">arm == "VCB"</button>
<button class="quiz-opt" data-why="没有引号,Python 会把 VCB 当成一个变量的名字去找,找不到就报 NameError。文字要加引号。">arm == VCB</button>
</div>

把数字放进一句话里显示,用 **f 字符串**: 引号前加 `f`,花括号里放名字:

```python
print(f"这位病人 {age} 岁,D2cc = {d2cc} Gy,超过 68 Gy: {high}")
total = 43.2 + 18.7
print(total)
print(f"总 EQD2 = {total:.1f} Gy")    # :.1f 表示保留 1 位小数
```

`43.2 + 18.7` 显示成 `61.900000000000006`,不是算错了: 电脑用二进制存小数,大多数小数存不精确,会有极小的误差。显示结果时用 `:.1f` 这样的格式保留几位小数就好。

## 4. 列表: 一串数

一位病人三套计划的阴道 D2cc,可以放在一个**列表**里,用方括号,逗号隔开:

```python
plans = ["5mm", "vol", "gra"]
d2cc = [71.4, 67.9, 66.2]
print(d2cc[0])                  # 第一个
print(d2cc[-1])                 # 最后一个
print(d2cc[1] - d2cc[0])        # 体积优化减 5 mm
print(len(d2cc), max(d2cc), min(d2cc), sum(d2cc) / len(d2cc))
```

- **位置从 0 开始数**: `d2cc[0]` 是第一个,`d2cc[2]` 是第三个。`-1` 表示倒数第一个。这是新手最常踩的坑。
- `len` 是个数,`max`、`min`、`sum` 是最大、最小、总和。
- `d2cc` 这个名字上一节存的是 `71.4`,这里又存了一个列表。同一个名字再赋值,旧的值就被换掉了。

<div class="quiz">
<p><b>想一想</b>: <code>plans = ["5mm", "vol", "gra"]</code>,<code>plans[1]</code> 是哪一个?</p>
<button class="quiz-opt" data-why="位置从 0 开始数: plans[0] 才是 5mm。">"5mm"</button>
<button class="quiz-opt" data-correct data-why="位置从 0 开始数: [0] 是 5mm,[1] 是 vol,[2] 是 gra。">"vol"</button>
<button class="quiz-opt" data-why="[2] 才是 gra。plans[1] 是第二个。">"gra"</button>
</div>

<div class="exercise">

**练习 1 · 图形优化比 5 mm 降了多少**

还是这位病人,`d2cc` 里的三个数依次是 5 mm、体积优化、图形优化计划的阴道 D2cc。算出**图形优化减 5 mm** 的差,存进 `delta_gra`。把 `___` 换成数字,再点"运行并检查"。

```python
d2cc = [71.4, 67.9, 66.2]
delta_gra = d2cc[___] - d2cc[___]
print(f"图形优化比 5 mm 变了 {delta_gra:.1f} Gy")
```

```pycheck
need("delta_gra")
if close(delta_gra, -5.2):
    ok("对了: 66.2 − 71.4 = −5.2 Gy,负数表示降低。")
elif close(delta_gra, 5.2):
    fail("差的大小对了,符号反了: 要用图形优化减 5 mm。")
elif close(delta_gra, -1.7):
    fail("这是图形优化减体积优化。5 mm 在第几个位置?")
elif close(delta_gra, -3.5):
    fail("这是体积优化减 5 mm。图形优化在第几个位置?")
else:
    fail(f"现在 delta_gra = {delta_gra:.1f}。位置从 0 开始数: 5 mm 是 [0]。")
```

<details><summary>提示</summary>

位置从 0 开始数: 5 mm 是 `d2cc[0]`,图形优化是第三个。

</details>

<details><summary>参考答案</summary>

```python
d2cc = [71.4, 67.9, 66.2]
delta_gra = d2cc[2] - d2cc[0]
print(f"图形优化比 5 mm 变了 {delta_gra:.1f} Gy")
```

也可以写 `d2cc[-1] - d2cc[0]`,`-1` 是最后一个。

</details>

</div>

## 5. 字典: 有名字的一组数

列表只按位置找,`d2cc[1]` 是哪套计划得自己记住。**字典**给每个值配一个名字(叫"键"),用花括号:

```python
patient = {"id": "P001", "age": 63, "arm": "VCB", "wall_mean": 3.2}
patient["d2cc_5mm"] = 71.4       # 加一项
print(patient["age"])
print(patient)
```

这就是表格里的**一行**: 列名对应值。第 2 课的 pandas 表格,可以想成很多个这样的字典排在一起,每位病人一行。

<div class="quiz">
<p><b>想一想</b>: 上面的 <code>patient</code> 里有一项 <code>"age": 63</code>。运行 <code>patient["Age"]</code> 会怎样?</p>
<button class="quiz-opt" data-why="键区分大小写: &quot;Age&quot; 和 &quot;age&quot; 是两个不同的键,字典里没有 &quot;Age&quot;。">得到 63</button>
<button class="quiz-opt" data-correct data-why="键区分大小写,字典里只有 &quot;age&quot;。找不到键时,Python 报 KeyError。以后在表格里把列名的大小写写错,报的也是这个错。">报错: KeyError</button>
<button class="quiz-opt" data-why="找不到键时,Python 不会悄悄给一个空值,而是直接报错(KeyError),这样拼写错误才不会被漏掉。">得到一个空值</button>
</div>

## 6. 循环和条件: 一个一个地检查

想知道三套计划里哪几套超过 68 Gy:

```python
for plan, dose in zip(plans, d2cc):
    if dose >= 68:
        print(f"{plan}: {dose} Gy,超过 68 Gy")
    else:
        print(f"{plan}: {dose} Gy")
```

- `for ... in ...:` 对后面的每一项都做一遍下面的事。`zip(plans, d2cc)` 把两个列表一对一对地配起来: ("5mm", 71.4)、("vol", 67.9)、("gra", 66.2)。
- `if 条件:` 条件成立才做下面的事,否则做 `else:` 下面的事。
- **缩进就是结构**。`for`、`if`、`else` 那一行以冒号结尾,下面缩进 4 个空格的几行属于它。别的语言用括号,Python 用缩进,所以缩进不能随便加减。

<div class="exercise">

**练习 2 · 数一数有几位病人达到 68 Gy**

下面是 8 位病人 5 mm 计划的阴道 D2cc。补上 `if` 后面的条件,数出**达到 68 Gy**(大于或等于 68)的有几位,结果在 `n_high` 里。

```python
doses = [71.4, 66.0, 69.0, 73.2, 64.8, 68.0, 70.5, 67.9]
n_high = 0
for dose in doses:
    if ___:
        n_high = n_high + 1
print(n_high, "位病人达到 68 Gy")
```

```pycheck
need("n_high")
if n_high == 5:
    ok("对了: 71.4、69.0、73.2、68.0、70.5,一共 5 位。")
elif n_high == 4:
    fail("差一位: 正好 68.0 Gy 的那位算不算达到 68 Gy?用 >= 而不是 >。")
elif n_high == 3:
    fail("你数的是低于 68 Gy 的病人。")
elif n_high == 8:
    fail("每一位都数进去了: if 后面的条件是不是总是成立?")
else:
    fail(f"现在 n_high = {n_high}。条件应该比较 dose 和 68。")
```

<details><summary>提示</summary>

条件写成 `dose >= 68`。"达到"包括正好等于 68。

</details>

<details><summary>参考答案</summary>

```python
doses = [71.4, 66.0, 69.0, 73.2, 64.8, 68.0, 70.5, 67.9]
n_high = 0
for dose in doses:
    if dose >= 68:
        n_high = n_high + 1
print(n_high, "位病人达到 68 Gy")
```

学了第 8 节的 numpy 以后,也可以一行写完: `(np.array(doses) >= 68).sum()`。

</details>

</div>

## 7. 函数: 把公式存起来反复用

第 2 节说过,改了 `d` 以后 `eqd2` 不会自己重算。把公式写成**函数**,每次用不同的数去调用它:

```python
def eqd2(d, n, ab=3):
    """每次剂量 d(Gy)、照射 n 次、α/β = ab 时的 EQD2(Gy)。"""
    return n * d * (d + ab) / (2 + ab)

print(eqd2(7, 3))                       # 单纯近距离 7 Gy × 3 次,阴道黏膜
print(eqd2(7, 3, ab=10))                # 同样的照射,靶区按 α/β = 10
print(eqd2(1.8, 25) + eqd2(5.5, 2))     # 外照射 1.8 Gy × 25 次,加近距离 5.5 Gy × 2 次
```

- `def 名字(参数):` 开始定义函数,下面缩进的几行是函数的内容,`return` 后面是它交回的结果。
- `ab=3` 是**默认值**: 调用时不写 `ab`,就按 3 算;要按 10 算,写 `ab=10`。
- 三引号里的一句话是函数的说明,写给以后的自己和合作者看。

函数就像一张印好的处方模板: 格式和算法固定,每次只填剂量和次数。论文里的每一步分析,最后都会写成这样的函数,同一个算法对每位病人、每套计划都一样地用。

<div class="exercise">

**练习 3 · 用函数算总 EQD2**

一位病人先做外照射 2 Gy × 25 次,再做近距离 6 Gy × 2 次。用上面的 `eqd2` 函数,算阴道黏膜(α/β = 3)的总 EQD2,存进 `total`。

```python
total = eqd2(___, ___) + eqd2(___, ___)
print(f"总 EQD2 = {total:.1f} Gy")
```

```pycheck
need("total")
if close(total, 71.6):
    ok("对了: 外照射 50.0 Gy,加近距离 21.6 Gy,总 EQD2 = 71.6 Gy。")
elif close(total, 292.0):
    fail("每次剂量和次数填反了: eqd2(每次剂量, 次数),每次剂量在前。")
elif close(total, 66.0):
    fail("阴道黏膜按 α/β = 3 算: 不写 ab 就是 3,不要写 ab=10。")
else:
    fail(f"现在 total = {total:.1f} Gy。对照题目检查四个数: 外照射每次 2 Gy、25 次,近距离每次 6 Gy、2 次。")
```

<details><summary>提示</summary>

`eqd2(每次剂量, 次数)`: 外照射是 `eqd2(2, 25)`。

</details>

<details><summary>参考答案</summary>

```python
total = eqd2(2, 25) + eqd2(6, 2)
print(f"总 EQD2 = {total:.1f} Gy")
```

</details>

</div>

<div class="exercise">

**练习 4 · 自己写一个函数: BED**

生物等效剂量 BED = 次数 × 每次剂量 × (1 + 每次剂量 / (α/β))。仿照 `eqd2`,把 `bed` 函数写完整。

```python
def bed(d, n, ab=3):
    """每次剂量 d(Gy)、照射 n 次、α/β = ab 时的 BED(Gy)。"""
    return ___

print(bed(7, 3), bed(7, 3, ab=10))
```

```pycheck
need("bed")
if not callable(bed):
    fail("bed 应该是一个函数,用 def bed(d, n, ab=3): 来定义。")
try:
    a, b, c = bed(7, 3), bed(7, 3, ab=10), bed(6, 2)
except TypeError:
    fail("调用 bed(7, 3, ab=10) 出错了: 参数要写成 d, n, ab=3 这三个。")
if a is None:
    fail("函数没有交回结果: 别忘了 return。")
elif close(a, 70) and close(b, 35.7) and close(c, 36):
    ok("对了: bed(7, 3) = 70.0 Gy,bed(7, 3, ab=10) = 35.7 Gy。顺便验证一下: EQD2 = BED / (1 + 2/(α/β)),70.0 / (1 + 2/3) = 42.0,和 eqd2(7, 3) 一样。")
elif close(a, 70):
    fail("α/β = 3 时对了,换成 ab=10 就不对: 公式里要用参数 ab,不要直接写 3。")
else:
    fail(f"bed(7, 3) 现在是 {a},应该是 70.0。公式: n × d × (1 + d / ab)。")
```

<details><summary>提示</summary>

乘法写 `*`,除法写 `/`,注意括号: `n * d * (1 + d / ab)`。

</details>

<details><summary>参考答案</summary>

```python
def bed(d, n, ab=3):
    """每次剂量 d(Gy)、照射 n 次、α/β = ab 时的 BED(Gy)。"""
    return n * d * (1 + d / ab)

print(bed(7, 3), bed(7, 3, ab=10))
```

</details>

</div>

## 8. 包: 别人写好的函数

Python 本身只带基本功能,更多的功能装在**包**里,用 `import` 拿进来:

```python
import numpy as np

doses = np.array([71.4, 67.9, 66.2, 73.0, 69.5])   # 5 位病人的 D2cc
print(np.mean(doses), np.median(doses))
print(doses - 68)       # 一次算出每位病人比 68 Gy 高多少
print(doses >= 68)
```

`import numpy as np` 拿进来 numpy(数值计算的包),并约定用简称 `np`,这是大家通用的写法。numpy 的数组可以整列一起算: `doses - 68` 对每个数都减 68,不用写循环。普通列表不行,第 9 节会看到。

这门课后面会用到 pandas(表格)和 matplotlib(画图),统计机器学习课还会用到 scikit-learn。浏览器里第一次导入一个大包要先下载,会多等一会儿。

<div class="quiz">
<p><b>想一想</b>: <code>doses &gt;= 68</code> 得到一串 True 和 False。那么 <code>(doses &gt;= 68).sum()</code> 算出来的是什么?</p>
<button class="quiz-opt" data-why="求和的对象是那一串 True 和 False,不是剂量本身。">达到 68 Gy 的那几个剂量加起来</button>
<button class="quiz-opt" data-correct data-why="求和时 True 算 1,False 算 0,所以得到的是 True 的个数,这里是 3。数人数最常用的就是这个写法。">达到 68 Gy 的病人有几位</button>
<button class="quiz-opt" data-why="可以相加: Python 把 True 当 1,把 False 当 0。">报错: True 和 False 不能相加</button>
</div>

## 9. 看懂报错

代码出错时,Python 会显示一段**报错**(traceback)。读法:

1. **从最后一行读起**。最后一行是 `错误类型: 说明`,比如 `NameError: name 'Dose' is not defined`。
2. 往上看 `line 2` 这样的字样,它指出出错的是第几行。
3. 这个网页会在报错下面加一行中文**提示**,猜可能的原因。它只是猜测,而且在自己电脑上运行时没有,所以要学会自己读最后一行。

下面每个代码块都有一处错误。先点 Run 看报错,读最后一行,再改好它、重新运行。

**① 名字写错**

```python
dose = 7
print(Dose)
```

`NameError`: 这个名字没有定义。大小写不同就是不同的名字。如果名字是在前面的代码块里定义的,也可能是前面的代码块还没运行。

**② 少了冒号**

```python
for dose in [71.4, 67.9, 66.2]
    print(dose)
```

`SyntaxError: expected ':'`: 写法不合语法。`for`、`if`、`def` 那一行要以冒号结尾。

**③ 缩进不对**

```python
for dose in [71.4, 67.9, 66.2]:
print(dose)
```

`IndentationError`: 冒号下面那一行要缩进。

**④ 文字和数字相加**

```python
age = "63"        # 从表格读进来时,数字有时会变成文字
age + 1
```

`TypeError`: 类型不对。`"63"` 带引号,是文字。改成 `int(age) + 1`,先把文字变成整数。

**⑤ 位置超出范围**

```python
d2cc = [71.4, 67.9, 66.2]
d2cc[3]
```

`IndexError: list index out of range`: 三个元素的位置是 0、1、2,没有 3。

**⑥ 列表不能整列减**

```python
[71.4, 67.9, 66.2] - 68
```

`TypeError: unsupported operand type(s) for -: 'list' and 'int'`: 普通列表不支持减一个数。用第 8 节的 `np.array(...)`。

这六种报错,占了初学时遇到的大部分。剩下的,把最后一行原样复制去搜索,或者问 AI(见下一节的注意事项),一般都能找到原因。

<div class="exercise">

**练习 5 · 改错**

这段代码想找出 D2cc 超过 68 Gy 的病人编号,但有两处错误。先运行,读报错,改好一处再运行,直到通过检查。`patients.items()` 每次同时取出一个键(病人编号)和它的值(D2cc);`high.append(pid)` 把一个编号加到列表 `high` 的末尾。

```python
patients = {"P001": 71.4, "P002": 66.0, "P003": 69.5}
high = []
for pid, dose in patients.items()
    if dose > 68:
    high.append(pid)
print(high)
```

```pycheck
need("high")
if high == ["P001", "P003"]:
    ok("对了: for 那一行补上冒号,high.append(pid) 缩进到 if 下面。")
else:
    fail(f"现在 high = {high},应该是 ['P001', 'P003']。")
```

<details><summary>提示</summary>

报错一次只指出一处。第一处: `for` 那一行少了什么?改好再运行,看第二处报的是什么。

</details>

<details><summary>参考答案</summary>

```python
patients = {"P001": 71.4, "P002": 66.0, "P003": 69.5}
high = []
for pid, dose in patients.items():
    if dose > 68:
        high.append(pid)
print(high)
```

</details>

</div>

## 10. 读懂别人或 AI 写的代码

以后你会经常读到合作者或 AI 写的代码。不必一眼看懂,按这四步读:

1. **先看进出**: 它要什么数据进来,最后交出什么结果。
2. **一行一行读**,每一行用自己的话说一遍。不认识的函数,查一下或者问 AI"这一行在做什么"。
3. **把中间结果 print 出来**。拿不准某个名字里存的是什么,就打印它,或者用 `type()` 看类型。
4. **用一个能心算的小例子试**。三位病人,结果你自己能算出来,再看代码算得对不对。

试试读下面这段。它判断换方案后阴道 D2cc 有没有从 68 Gy 以上降到 68 Gy 以下:

```python
def crossing(d2cc_5mm, d2cc_new, threshold=68):
    """5 mm 计划达到阈值,换方案后降到阈值以下,返回 True。"""
    return d2cc_5mm >= threshold and d2cc_new < threshold

pairs = [(71.4, 67.9), (69.0, 68.5), (66.0, 64.2)]
flags = [crossing(a, b) for a, b in pairs]
print(flags)
print(sum(flags), "位病人降到 68 Gy 以下")
```

逐行看:

- `def crossing(...)`: 一个函数,进来两个剂量和一个阈值(默认 68),交出 `True` 或 `False`。
- `pairs`: 一个列表,每一项是一对数,用圆括号括起来(叫元组),分别是 5 mm 计划和新计划的 D2cc。
- `[crossing(a, b) for a, b in pairs]`: 第 6 节那种循环的简写,意思是"对 pairs 里的每一对 (a, b),算一次 crossing,把结果排成一个新列表"。
- `sum(flags)`: 对真假求和时 `True` 算 1,`False` 算 0,所以得到的是 `True` 的个数。

心算核对: 第一位从 71.4 降到 67.9,算;第二位 68.5 还在 68 以上,不算;第三位本来就低于 68,不算。结果应该是 `[True, False, False]` 和 1 位,和代码一致。

> ⚠️ 让 AI 帮忙写代码或解释报错时,只描述数据的样子: 有哪些列、是什么类型,最多编几行假数据。**不要把病人数据贴进对话**,病历号和逐例数值都不要。

## 小结

- 代码块最后一行的值会自动显示;要显示别的,用 `print`。同一页的代码块按顺序运行。
- `=` 是存值;存进去的是算好的数,不会像 Excel 那样自动重算。
- 文字、整数、小数、真假是四种基本类型;"等于"写 `==`。
- 列表按位置找,**从 0 开始**;字典按名字找,一个字典就像表格的一行。
- `for` 和 `if` 那一行以冒号结尾,下面的内容靠缩进。
- 函数把一个算法存起来,对每位病人都一样地用。
- 读报错从最后一行读起;读别人的代码时打印中间结果,用小例子核对。

## 本课术语 Key terms

| 中文 | English | 一句话 |
|---|---|---|
| 变量 | variable | 给一个值起的名字 |
| 字符串 | string (`str`) | 文字,要加引号 |
| 整数、小数 | integer (`int`)、float | 两种数字 |
| 布尔值 | boolean (`bool`) | True 或 False |
| 列表 | list | 一串值,按位置取,从 0 开始数 |
| 字典 | dictionary (`dict`) | 一组"键: 值",按键取 |
| 循环 | loop (`for`) | 对每一项做同样的事 |
| 条件 | condition (`if`) | 条件成立才做 |
| 函数 | function | 存起来可以反复用的一段算法 |
| 参数 | argument | 调用函数时填进去的值 |
| 包 | package、library | 别人写好的一批函数 |
| 报错 | error、traceback | 出错时 Python 给出的说明 |
| 注释 | comment | `#` 后面写给人看的话 |

## 这一课没讲什么

- **读文件和表格**: 第 2 课。
- **`while` 循环、自己定义"类"(class)**: 整理临床数据很少用到,遇到时再学也来得及。
- **在自己电脑上安装 Python、把代码存成文件**: 第 7 课。

## 写进论文 Writing it up

<div class="paper">

方法部分最后要写清楚用了什么软件、什么版本,别人才能重复你的分析。一般这样写(版本号换成你实际用的):

> Analyses were performed in Python 3.12 using NumPy 2.4, pandas 3.0 and scikit-learn 1.8.

查版本:

```python
import sys
import numpy
print(sys.version)
print(numpy.__version__)
```

浏览器里显示的是这个网页用的版本。论文里要写的是你在自己电脑上做分析时的版本(第 7 课讲安装)。

**审稿人可能问**: 分析能重复吗?代码能公开吗?

**怎么回答**: 写明软件版本;用到随机数的地方固定随机种子(统计机器学习课里会经常用到);把分析代码整理好,放进代码仓库或附在补充材料里,病人数据不放。论文里可以写 "The analysis code is available from the corresponding author on request",或者直接给出仓库地址。

</div>

## 常见问题 Troubleshooting

| 现象 | 原因和办法 |
|---|---|
| 点了 Run 一直显示"正在启动" | 第一次要下载 Python,网速慢时要等久一些。超过一分钟,检查网络后刷新页面再试。 |
| 后面的代码块报 `NameError` | 它用到的名字在前面的代码块里定义,而前面的还没运行。从上往下依次运行。 |
| 把代码改乱了,想恢复原样 | 刷新页面。改动不会保存。 |
| 复制到自己电脑上运行,没有显示结果 | 在 `.py` 文件里,最后一行的值不会自动显示,要用 `print(...)`。 |

## 延伸阅读 Further reading

- [Python 官方教程(中文)](https://docs.python.org/zh-cn/3/tutorial/): 第 3–5 章和这一课的内容对应,想多练可以看。

**下一课**: 一位病人的记录是一个字典,一个队列就是一张表。第 2 课用 pandas 读入和整理整张表格。
