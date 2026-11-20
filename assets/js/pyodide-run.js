// In-browser Python for the "Python 与机器学习" course, powered by Pyodide
// (CPython compiled to WebAssembly). Mirrors webr-run.js:
// - lazy: Pyodide is only downloaded and booted on the first "Run" click;
// - all blocks on a page share one namespace, like cells of one notebook,
//   so a block can use variables from the blocks above it;
// - blocks are editable before running.
// Packages (numpy, pandas, scikit-learn, matplotlib ...) are fetched from the
// imports of the code being run, the first time they are needed.
// Exercises: a code block inside <div class="exercise"> that also holds a hidden
// "pycheck" block (rendered as pre.py-check) is checked after it runs. The check
// code reads the learner's variables and calls ok(...) or fail(...); need("x")
// fails with a standard message when x is not defined yet.
const PYODIDE_VERSION = '314.0.7';
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const COURSE_FILES = ['cohort.py'];            // served at /python/ (static/python/), used by both courses
const LOCAL_DEPS = { cohort: ['numpy', 'pandas'] };
const HEAVY = new Set(['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'statsmodels']);

let pyPromise = null;
const seenImports = new Set();

// Runs once, after boot. Everything the runner needs lives in the _runner module,
// so the learner's namespace stays clean.
const SETUP = String.raw`
import os, sys, types
os.environ["MPLBACKEND"] = "AGG"
# Figures are read on phones: a default size that keeps 10 pt text legible at ~350 px wide.
with open("/home/pyodide/matplotlibrc", "w") as f:
    f.write("figure.figsize: 4.8, 3.6\n")
os.environ["MATPLOTLIBRC"] = "/home/pyodide/matplotlibrc"
if "/home/pyodide" not in sys.path:
    sys.path.insert(0, "/home/pyodide")

_runner = types.ModuleType("_runner")
exec('''
import io, sys, base64, traceback, warnings, contextlib
from pyodide.code import eval_code_async

warnings.filterwarnings("ignore", message=".*non-interactive.*")
FILE = "<代码块>"
ns = {"__name__": "__main__"}

HINTS = {
    "IndentationError": "缩进不对。同一层的代码要对齐;以冒号结尾的那一行(if、for、def)下面要缩进 4 个空格。",
    "SyntaxError": "写法不合 Python 的语法。看 ^ 指的位置和它前面一行: 常见原因是少了冒号、括号或引号没有配对。",
    "NameError": "这个名字还没有定义。检查拼写和大小写;如果它是在本页前面的代码块里定义的,先按顺序运行前面的代码块。",
    "TypeError": "类型不对,比如把文字和数字相加,或者给函数传错了参数。最后一行会说是哪两种类型。",
    "IndexError": "位置超出了范围。Python 从 0 开始数: 有 3 个元素的列表,最后一个是 [2]。",
    "KeyError": "找不到这个键(字典的键,或者表格的列名)。检查拼写和大小写。",
    "ModuleNotFoundError": "找不到这个包。检查包名;浏览器里只有常用的科学计算包,其他的要在自己电脑上安装(Python 课第 7 课)。",
    "AttributeError": "这个对象没有这个属性或方法。检查拼写,或者用 type() 看看它到底是什么。",
    "ValueError": "值不合要求,看最后一行的说明。建模时最常见的原因是数据里有缺失值(NaN)。",
    "ZeroDivisionError": "除以了 0。",
}

def _hint(e):
    for cls in type(e).__mro__:
        if cls.__name__ in HINTS:
            return HINTS[cls.__name__]
    return None

def _format_error(e):
    tb = e.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename != FILE:
        tb = tb.tb_next
    return "".join(traceback.format_exception(type(e), e, tb, chain=False)).rstrip()

def _is_mpl(v):
    if isinstance(v, (list, tuple)) and v:
        return all(_is_mpl(x) for x in v)
    return type(v).__module__.startswith("matplotlib")

def _figures():
    if "matplotlib.pyplot" not in sys.modules:
        return []
    import matplotlib.pyplot as plt
    out = []
    for num in plt.get_fignums():
        buf = io.BytesIO()
        plt.figure(num).savefig(buf, format="png", dpi=110, bbox_inches="tight")
        out.append(base64.b64encode(buf.getvalue()).decode())
    plt.close("all")
    return out

def _pandas_options():
    if "pandas" in sys.modules:
        import pandas as pd
        pd.set_option("display.max_rows", 20)
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 100)

async def run(code):
    buf = io.StringIO()
    res = {"text": "", "html": None, "images": [], "error": None, "hint": None}
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            value = await eval_code_async(code, globals=ns, filename=FILE)
            _pandas_options()
            if value is not None and not _is_mpl(value):
                if type(value).__module__.startswith("pandas") and hasattr(value, "_repr_html_"):
                    res["html"] = value._repr_html_()
                else:
                    print(repr(value))
    except BaseException as e:
        res["error"] = _format_error(e)
        res["hint"] = _hint(e)
    res["text"] = buf.getvalue()
    try:
        res["images"] = _figures()
    except Exception as e:
        res["text"] += "\\n(图没能画出来: " + repr(e) + ")"
    return res

class _Stop(Exception):
    pass

async def check(code):
    found = []
    def ok(msg="对了。"):
        found.append(["ok", msg]); raise _Stop
    def fail(msg):
        found.append(["fail", msg]); raise _Stop
    def need(*names):
        for name in names:
            if name not in ns:
                fail(f"还没有定义 {name}。按题目把结果存进 {name},再运行。")
    def close(a, b, tol=0.05):
        try:
            return abs(float(a) - float(b)) <= tol
        except (TypeError, ValueError):
            return False
    g = dict(ns)
    g.update(ok=ok, fail=fail, need=need, close=close)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            await eval_code_async(code, globals=g, filename="<检查>")
    except _Stop:
        pass
    except Exception as e:
        found.append(["fail", f"检查没能完成({type(e).__name__}: {e})。对照题目看看结果的类型对不对。"])
    return found[0] if found else ["fail", "检查没有得到结果。"]
''', _runner.__dict__)
sys.modules["_runner"] = _runner
`;

function boot(setStatus) {
  if (!pyPromise) {
    setStatus('正在启动 Python(第一次要下载,约十几秒)…');
    pyPromise = (async () => {
      const { loadPyodide } = await import(INDEX_URL + 'pyodide.mjs');
      const py = await loadPyodide({ indexURL: INDEX_URL });
      for (const name of COURSE_FILES) {
        try {
          const r = await fetch(new URL('/python/' + name, location.href));
          if (r.ok) py.FS.writeFile('/home/pyodide/' + name, await r.text());
        } catch (_) { /* the lesson will report the missing module */ }
      }
      await py.runPythonAsync(SETUP);
      return py;
    })().catch((err) => {
      pyPromise = null; // let the next click retry (e.g. after a network blip)
      throw err;
    });
  }
  return pyPromise;
}

function importsOf(py, code) {
  try {
    const found = py.pyodide_py.code.find_imports(code);
    const list = found.toJs();
    found.destroy();
    return list.map((n) => n.split('.')[0]);
  } catch (_) {
    return []; // a syntax error: let the run itself report it
  }
}

async function loadPackages(py, code, setStatus) {
  const imports = importsOf(py, code).filter((n) => !seenImports.has(n));
  if (!imports.length) return;
  const extra = imports.flatMap((n) => LOCAL_DEPS[n] || []);
  const shown = [...new Set([...imports, ...extra].filter((n) => HEAVY.has(n)))];
  if (shown.length) setStatus('正在下载 ' + shown.join('、') + '(第一次需要较长时间,之后会快)…');
  const quiet = { messageCallback: () => {}, errorCallback: () => {} };
  await py.loadPackagesFromImports(code, quiet);
  if (extra.length) await py.loadPackage(extra, quiet);
  imports.forEach((n) => seenImports.add(n));
}

// The code of a block as plain text. innerText would double every line break
// (Chroma wraps each line in a flex span that already ends with "\n"), which
// would also double the line numbers in tracebacks. Edits made in the browser
// may add <br> or <div>, and &nbsp; for spaces, which would break indentation.
function codeText(pre) {
  let s = '';
  const walk = (node) => {
    for (const ch of node.childNodes) {
      if (ch.nodeType === 3) { s += ch.nodeValue; continue; }
      if (ch.nodeType !== 1) continue;
      if (ch.nodeName === 'BR') { s += '\n'; continue; }
      const block = ch.nodeName === 'DIV' || ch.nodeName === 'P';
      if (block && s && !s.endsWith('\n')) s += '\n';
      walk(ch);
      if (block && !s.endsWith('\n')) s += '\n';
    }
  };
  walk(pre);
  return s.replace(/\u00a0/g, ' ');
}

function render(out, res) {
  out.innerHTML = '';
  const add = (tag, cls, text) => {
    const el = document.createElement(tag);
    el.className = cls;
    el.textContent = text;
    out.appendChild(el);
    return el;
  };
  if (res.text && res.text.trim()) add('pre', 'code-output-text', res.text.replace(/\s+$/, ''));
  if (res.html) {
    const div = document.createElement('div');
    div.className = 'code-output-html';
    div.innerHTML = res.html; // a pandas table produced by the learner's own code in this browser
    out.appendChild(div);
  }
  (res.images || []).forEach((b64) => {
    const img = document.createElement('img');
    img.className = 'code-output-plot';
    img.alt = '代码画出的图';
    img.src = 'data:image/png;base64,' + b64;
    out.appendChild(img);
  });
  if (res.error) {
    add('pre', 'code-output-err', res.error);
    if (res.hint) add('p', 'code-output-hint', '提示: ' + res.hint);
  }
  if (res.check) add('p', 'code-output-check ' + res.check[0], (res.check[0] === 'ok' ? '✓ ' : '✗ ') + res.check[1]);
  if (!out.childNodes.length) add('pre', 'code-output-text code-output-status', '(运行完毕,没有输出)');
}

// The hidden answer check of the exercise a code card belongs to, if any.
function checkCodeFor(card) {
  const ex = card.closest('.exercise');
  const el = ex && ex.querySelector('pre.py-check');
  return el ? el.textContent : null;
}

async function runCard(card) {
  const codeEl = card.querySelector('pre');
  const out = card.querySelector('.code-output');
  const btn = card.querySelector('.py-run');
  if (!codeEl || !out || !btn) return;

  const code = codeText(codeEl);
  const checkCode = checkCodeFor(card);
  out.hidden = false;
  if (checkCode && code.includes('___')) {
    out.innerHTML = '';
    const p = document.createElement('p');
    p.className = 'code-output-check fail';
    p.textContent = '先把代码里的 ___ 换成你的答案,再点运行。';
    out.appendChild(p);
    return;
  }
  const setStatus = (msg) => { out.innerHTML = '<pre class="code-output-text code-output-status"></pre>'; out.firstChild.textContent = msg; };
  const paint = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = '运行中…';

  try {
    const py = await boot(setStatus);
    await loadPackages(py, code, setStatus);
    setStatus('运行中…');
    await paint(); // let the status show before the (blocking) run
    const runner = py.pyimport('_runner');
    const proxy = await runner.run(code);
    const res = proxy.toJs({ dict_converter: Object.fromEntries });
    proxy.destroy();
    if (checkCode && !res.error) {
      const c = await runner.check(checkCode);
      res.check = c.toJs();
      c.destroy();
    }
    runner.destroy();
    render(out, res);
  } catch (err) {
    out.innerHTML = '';
    const pre = document.createElement('pre');
    pre.className = 'code-output-err';
    pre.textContent = '没能运行: ' + String((err && err.message) || err) + '\n(检查网络后再点一次 Run;还不行就刷新页面。)';
    out.appendChild(pre);
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

function init() {
  document.querySelectorAll('.py-runnable pre').forEach((pre) => {
    pre.setAttribute('contenteditable', 'plaintext-only');
    if (pre.contentEditable !== 'plaintext-only') pre.setAttribute('contenteditable', 'true');
    pre.setAttribute('spellcheck', 'false');
    pre.setAttribute('autocapitalize', 'off');
    pre.setAttribute('autocorrect', 'off');
    pre.setAttribute('title', '可以直接在这里改代码,再点 Run');
  });
  document.querySelectorAll('.exercise .py-run').forEach((btn) => {
    if (checkCodeFor(btn.closest('.code-card'))) btn.textContent = '▶ 运行并检查';
  });
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.py-run');
    if (btn) runCard(btn.closest('.code-card'));
  });
}

if (document.readyState !== 'loading') init();
else document.addEventListener('DOMContentLoaded', init);
