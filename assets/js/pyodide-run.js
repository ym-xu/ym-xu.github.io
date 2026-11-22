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
// A course is read over several sittings, and each time the page is opened Python starts empty:
// - what the learner typed into an exercise is kept in this browser and put back (with a 重置 button);
// - when a block fails with a NameError and blocks above it have not been run since the page was
//   opened, those are run first (silently, in order) and the block is run again.
// The code boxes are editors (assets/js/vendor/py-editor.js, CodeMirror: indentation, colours while
// typing, Shift-Enter runs); when that file did not load, they stay plain editable <pre> blocks.
const PYODIDE_VERSION = '314.0.7';
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
// served at /python/ (static/python/): the simulated cohort of both courses, and the audit data of the themed practice
const COURSE_FILES = ['cohort.py', 'cohort.csv', 'followup.csv', 'cervix.py', 'cervix.csv', 'cervix_followup.csv'];
const LOCAL_DEPS = { cohort: ['numpy', 'pandas'], cervix: ['numpy', 'pandas'] };
const HEAVY = new Set(['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'statsmodels']);

let pyPromise = null;
const seenImports = new Set();

// Completion and call hints in the code boxes are being tried out: on with ?try=complete in the address
// (kept in this browser, for every lesson), off again with ?try=off.
const TRY_KEY = 'course-try:complete';
const ASSIST = (() => {
  try {
    const q = new URLSearchParams(location.search).get('try');
    if (q === 'complete') localStorage.setItem(TRY_KEY, '1');
    if (q === 'off') localStorage.removeItem(TRY_KEY);
    return localStorage.getItem(TRY_KEY) === '1';
  } catch (_) {
    return false;
  }
})();
let pyRunner = null;                              // the _runner module once Python is up (for completion)
let busy = false;                                 // a block is running: completion does not call Python then
let nsNames = [];                                 // [name, type] of the learner's variables, after each run

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
    "ValueError": "值不合要求,看最后一行的说明。",
    "ZeroDivisionError": "除以了 0。",
}

def _hint(e):
    msg = str(e)
    if isinstance(e, SyntaxError) and msg.startswith("invalid character"):
        ch = msg.split("'")[1] if msg.count("'") >= 2 else ""
        return f"代码里有一个中文标点 {ch}。Python 只认英文标点(引号、括号、逗号、冒号): 删掉它,切换到英文输入法再输一次。"
    if isinstance(e, ValueError):
        if "NaN" in msg:
            return "数据里有缺失值(NaN),这个模型不接受。先处理缺失值,比如用 dropna() 去掉。"
        if "feature names" in msg or "features, but" in msg:
            return "给模型的变量表,和建模时用的不一样。用哪张表建的模型,预测时就要给它同样的那几列。"
        if "reshape" in msg:
            return "形状对不上: 元素的个数和要整理成的行数、列数不符。检查上一步得到了几个数。"
    for cls in type(e).__mro__:
        if cls.__name__ in HINTS:
            return HINTS[cls.__name__]
    return None

def _format_error(e):
    # only the learner's own lines: frames inside the runner or inside libraries are left out
    te = traceback.TracebackException.from_exception(e)
    te.stack = traceback.StackSummary.from_list([f for f in te.stack if f.filename == FILE])
    return "".join(te.format(chain=False)).rstrip()

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

# Completion and call hints for the code boxes (py-editor.js, opts.assist). Only plain expressions are
# looked at (names, attributes, subscripts with a constant): nothing is called, nothing changes.
def _expr(expr):
    import ast
    node = ast.parse(expr, mode="eval").body
    def plain(n):
        if isinstance(n, ast.Name):
            return True
        if isinstance(n, ast.Attribute):
            return plain(n.value)
        if isinstance(n, ast.Subscript):
            return plain(n.value) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, (str, int))
        return False
    if not plain(node):
        raise ValueError("not a plain expression")
    return eval(compile(ast.Expression(node), "<completion>", "eval"), ns)

def names():
    return [[k, type(v).__name__] for k, v in list(ns.items()) if not k.startswith("_")]

def attrs(expr):
    try:
        obj = _expr(expr)
    except Exception:
        return None
    return [n for n in dir(obj) if not n.startswith("_")][:500]

def keys(expr):
    try:
        obj = _expr(expr)
    except Exception:
        return None
    if isinstance(obj, dict):
        return [k for k in obj if isinstance(k, str)][:300]
    cols = getattr(obj, "columns", None)
    if cols is not None:
        return [str(c) for c in cols][:300]
    return None

def signature(expr):
    import inspect
    try:
        obj = _expr(expr)
    except Exception:
        return None
    if not callable(obj):
        return None
    name = getattr(obj, "__name__", expr.split(".")[-1])
    lines = [l.strip() for l in (inspect.getdoc(obj) or "").splitlines() if l.strip()]
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        if lines and lines[0].startswith(name + "("):
            return {"name": "", "parts": [[lines[0], -1]], "params": [], "doc": lines[1][:160] if len(lines) > 1 else ""}
        return None
    plist = list(sig.parameters.values())
    has_star = any(p.kind == p.VAR_POSITIONAL for p in plist)
    parts, params, starred = [], [], False
    for i, p in enumerate(plist):
        if p.kind == p.KEYWORD_ONLY and not has_star and not starred:
            parts.append(["*", -1])
            starred = True
        if p.kind == p.VAR_POSITIONAL:
            text = "*" + p.name
        elif p.kind == p.VAR_KEYWORD:
            text = "**" + p.name
        elif p.default is p.empty:
            text = p.name
        else:
            d = repr(p.default)
            text = p.name + "=" + (d if len(d) <= 24 else d[:21] + "...")
        parts.append([text, i])
        params.append([p.name, p.kind.name])
        if p.kind == p.POSITIONAL_ONLY and (i + 1 == len(plist) or plist[i + 1].kind != p.POSITIONAL_ONLY):
            parts.append(["/", -1])
    doc = lines[0] if lines and not lines[0].startswith(name + "(") else (lines[1] if len(lines) > 1 else "")
    return {"name": name, "parts": parts, "params": params, "doc": doc[:160]}
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
      pyRunner = py.pyimport('_runner');
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
  if (res.replayed) add('p', 'code-output-note', `已先运行本页前面的 ${res.replayed} 个代码块(每次打开页面,Python 都从空白开始)。`);
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

// ---- exercise progress, kept in this browser only (localStorage) ----
// A lesson's exercises are the .exercise boxes that hold a hidden check. Passing one (with the
// learner's own block, not the folded reference answer) marks it, and the overview box at the
// top of the lesson and the course syllabus show how many have passed.
const PROGRESS_KEY = 'course-progress:v1:' + location.pathname.replace(/index\.html$/, '');
function checkedExercises() {
  return [...document.querySelectorAll('.exercise')].filter((ex) => ex.querySelector('pre.py-check'));
}
function readProgress() {
  try { return JSON.parse(localStorage.getItem(PROGRESS_KEY)) || {}; } catch (_) { return {}; }
}
// In-lesson exercises and homework (.exercise.homework, at the end of a lesson) are counted apart,
// so a lesson reads as complete without its optional homework.
function tally(all, done) {
  const t = { core: 0, coreDone: 0, hw: 0, hwDone: 0 };
  all.forEach((ex, i) => {
    const kind = ex.classList.contains('homework') ? 'hw' : 'core';
    t[kind] += 1;
    if (done[i]) t[kind + 'Done'] += 1;
  });
  return t;
}
function showProgress() {
  const all = checkedExercises();
  if (!all.length) return;
  const done = readProgress().done || [];
  all.forEach((ex, i) => ex.classList.toggle('passed', !!done[i]));
  const t = tally(all, done);
  const chip = document.querySelector('[data-lesson-progress]');
  if (chip) {
    chip.hidden = false;
    chip.textContent = (t.coreDone === t.core ? `练习 ${t.core} / ${t.core} 全部通过 ✓` : `练习 已通过 ${t.coreDone} / ${t.core}`)
      + (t.hw ? ` · 课后 ${t.hwDone} / ${t.hw}` : '');
  }
}
function markPassed(card) {
  if (card.closest('details')) return;          // the reference answer does not count
  const all = checkedExercises(), i = all.indexOf(card.closest('.exercise'));
  if (i < 0) return;
  const p = readProgress();
  p.done = p.done || [];
  const first = !p.done[i];
  p.done[i] = true;
  const t = tally(all, p.done);
  p.total = all.length;
  p.core = t.core;                                // the syllabus shows progress on the in-lesson exercises
  p.coreDone = t.coreDone;
  p.hw = t.hw;
  p.hwDone = t.hwDone;
  try { localStorage.setItem(PROGRESS_KEY, JSON.stringify(p)); } catch (_) { /* private mode: no progress kept */ }
  showProgress();
  // fireworks (assets/js/fireworks.js) from the Run button; a bigger show when this was the last one
  // left of its kind (in-lesson exercises, or homework)
  const isHw = all[i].classList.contains('homework');
  const allDone = first && (isHw ? t.hwDone === t.hw : t.coreDone === t.core);
  const r = card.querySelector('.py-run').getBoundingClientRect();
  if (window.courseFireworks) window.courseFireworks(r.left + r.width / 2, r.top + r.height / 2, allDone);
}

// ---- the learner's code in the exercises, kept in this browser only (localStorage) ----
// Keyed by the position of the exercise on the page, with a hash of the starter code: if a lesson
// is edited later and the starter changed, the old text is not put into the wrong exercise.
const CODE_KEY = 'course-code:v1:' + location.pathname.replace(/index\.html$/, '');
const originals = new Map();                      // card -> { text, html, hash } of the starter code
const editors = new Map();                        // card -> editor (py-editor.js), when it loaded
function codeOf(card) {
  const ed = editors.get(card);
  if (ed) return ed.get();
  const pre = card.querySelector('pre');
  return pre ? codeText(pre) : null;
}
window.pyCodeOf = codeOf;                         // the Copy button (site.js) and the check scripts read code here
function hashOf(s) {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return String(h);
}
function exerciseCards() {
  return [...document.querySelectorAll('.exercise .py-runnable')].filter((c) => !c.closest('details') && checkCodeFor(c));
}
function codeBox(card) {
  const pre = card.querySelector('pre');
  return pre.querySelector('code') || pre;
}
function readCode() {
  try { return JSON.parse(localStorage.getItem(CODE_KEY)) || {}; } catch (_) { return {}; }
}
function saveCode(card) {
  const i = exerciseCards().indexOf(card), orig = originals.get(card);
  if (i < 0 || !orig) return;
  const all = readCode(), now = codeOf(card);
  if (now === orig.text) delete all[i];
  else all[i] = { c: now, o: orig.hash };
  try { localStorage.setItem(CODE_KEY, JSON.stringify(all)); } catch (_) { /* private mode: nothing kept */ }
  card.classList.toggle('edited', i in all);
}
function restoreCode() {
  const saved = readCode();
  exerciseCards().forEach((card, i) => {
    const pre = card.querySelector('pre'), box = codeBox(card);
    const text = codeText(pre);
    originals.set(card, { text, html: box.innerHTML, hash: hashOf(text) });
    addReset(card);
    const s = saved[i];
    if (s && typeof s.c === 'string' && s.o === hashOf(text) && s.c !== text) {
      box.textContent = s.c;
      card.classList.add('edited');
    }
    let timer = null;
    pre.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => saveCode(card), 500); });
  });
}

function addReset(card) {
  const reset = document.createElement('button');
  reset.type = 'button';
  reset.className = 'code-reset';
  reset.textContent = '重置';
  reset.title = '回到题目给出的起始代码';
  reset.addEventListener('click', () => {
    const ed = editors.get(card), orig = originals.get(card);
    if (ed) ed.set(orig.text);
    else codeBox(card).innerHTML = orig.html;
    saveCode(card);
  });
  const actions = card.querySelector('.code-actions');
  if (actions) actions.insertBefore(reset, actions.firstChild);
}

// ---- completion and call hints (py-editor.js, opts.assist), while ?try=complete is on ----
const KEYWORDS = ('False None True and as assert async await break class continue def del elif else except finally for from '
  + 'global if import in is lambda nonlocal not or pass raise return try while with yield').split(' ');
const BUILTIN_NAMES = ('abs all any bool callable chr dict dir divmod enumerate filter float format getattr hasattr help id '
  + 'input int isinstance issubclass iter len list map max min next object open ord pow print range repr reversed round '
  + 'set setattr slice sorted str sum super tuple type vars zip').split(' ');
let pageNameList = null;
function pageNames() {                            // names written in this page's code boxes (not the folded answers)
  if (!pageNameList) {
    const seen = new Set();
    document.querySelectorAll('.py-runnable').forEach((card) => {
      if (card.closest('details')) return;
      for (const m of (codeOf(card) || '').matchAll(/[A-Za-z_][A-Za-z0-9_]+/g)) seen.add(m[0]);
    });
    pageNameList = [...seen];
  }
  return pageNameList;
}
function fromPython(fn, ...args) {
  if (!pyRunner || busy) return null;
  try {
    const r = pyRunner[fn](...args);
    if (r == null) return null;
    const js = typeof r.toJs === 'function' ? r.toJs({ dict_converter: Object.fromEntries }) : r;
    if (typeof r.destroy === 'function') r.destroy();
    return js;
  } catch (_) {
    return null;
  }
}
function refreshNames() {
  const r = fromPython('names');
  if (r) nsNames = r;
}
const assist = {
  names() {
    const out = new Map();
    for (const [n, t] of nsNames) out.set(n, { label: n, detail: t, boost: 2 });
    for (const n of pageNames()) if (!out.has(n) && !KEYWORDS.includes(n)) out.set(n, { label: n, boost: 1 });
    for (const n of BUILTIN_NAMES) if (!out.has(n)) out.set(n, { label: n, detail: 'builtin' });
    for (const n of KEYWORDS) if (!out.has(n)) out.set(n, { label: n, detail: 'keyword', boost: -1 });
    return [...out.values()];
  },
  attrs: (expr) => fromPython('attrs', expr),
  keys: (expr) => fromPython('keys', expr),
  signature: (expr) => fromPython('signature', expr),
};

// The code boxes become editors. What was typed into an exercise comes back as with the plain boxes
// (same record, same hash of the starter code); an exercise box ends with an empty line, so a tap
// below the starter comment lands where the code goes.
function setupEditors() {
  if (ASSIST) pageNames();                        // the names in the code as the page comes, before anything is typed
  const saved = readCode();
  const exercises = exerciseCards();
  document.querySelectorAll('.py-runnable').forEach((card) => {
    const pre = card.querySelector('pre');
    if (!pre) return;
    const raw = codeText(pre);
    const i = exercises.indexOf(card);
    let text = raw.replace(/\n+$/, '');
    if (i >= 0) {
      const start = text + '\n';
      originals.set(card, { text: start, html: codeBox(card).innerHTML, hash: hashOf(raw) });
      addReset(card);
      const s = saved[i];
      if (s && typeof s.c === 'string' && s.o === hashOf(raw) && s.c !== raw && s.c !== start) {
        text = s.c;
        card.classList.add('edited');
      } else text = start;
    }
    let timer = null;
    try {
      editors.set(card, window.PyEditor.create(pre, text, {
        exercise: i >= 0,
        assist: ASSIST ? assist : null,
        onRun: () => runCard(card),
        onChange: i >= 0 ? () => { clearTimeout(timer); timer = setTimeout(() => saveCode(card), 500); } : null,
      }));
    } catch (_) {
      plainBox(pre);                              // this one stays a plain editable block
      if (i >= 0 && text !== raw) codeBox(card).textContent = text;
      if (i >= 0) pre.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => saveCode(card), 500); });
    }
  });
}

// ---- blocks run since the page was opened ----
const ran = new WeakSet();
function cardsBefore(card) {
  const all = [...document.querySelectorAll('.py-runnable')].filter((c) => !c.closest('details'));
  return all.slice(0, Math.max(0, all.indexOf(card)));
}

// The hidden answer check of the exercise a code card belongs to, if any.
function checkCodeFor(card) {
  const ex = card.closest('.exercise');
  const el = ex && ex.querySelector('pre.py-check');
  return el ? el.textContent : null;
}

async function runCard(card) {
  const out = card.querySelector('.code-output');
  const btn = card.querySelector('.py-run');
  const code = codeOf(card);
  if (code == null || !out || !btn || btn.disabled) return;

  const checkCode = checkCodeFor(card);
  saveCode(card);                                 // an exercise: keep what was typed, also when it fails
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
  busy = true;
  btn.textContent = '运行中…';

  try {
    const py = await boot(setStatus);
    await loadPackages(py, code, setStatus);
    setStatus('运行中…');
    await paint(); // let the status show before the (blocking) run
    const runner = py.pyimport('_runner');
    const runOnce = async () => {
      const proxy = await runner.run(code);
      const r = proxy.toJs({ dict_converter: Object.fromEntries });
      proxy.destroy();
      return r;
    };
    let res = await runOnce();
    // A name is missing and blocks above this one have not been run since the page was opened
    // (the usual case when a lesson is continued on another day): run those first, then this one again.
    const earlier = res.error && /(^|\n)NameError:/.test(res.error) ? cardsBefore(card).filter((c) => !ran.has(c)) : [];
    if (earlier.length) {
      let k = 0;
      for (const c of earlier) {
        k += 1;
        ran.add(c);
        setStatus(`这一块用到了前面定义的东西,先运行本页前面的代码块(${k} / ${earlier.length})…`);
        await paint();
        const src = codeOf(c);
        if (src == null) continue;
        if (src.includes('___')) continue;       // an exercise not filled in yet
        try {
          await loadPackages(py, src, () => {});
          (await runner.run(src)).destroy();      // output and errors of earlier blocks are not shown again
        } catch (_) { /* go on with the next block */ }
      }
      setStatus('运行中…');
      await paint();
      res = await runOnce();
      res.replayed = earlier.length;
    }
    ran.add(card);
    if (checkCode && !res.error) {
      const c = await runner.check(checkCode);
      res.check = c.toJs();
      c.destroy();
      if (res.check[0] === 'ok') markPassed(card);
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
    busy = false;
    if (ASSIST) refreshNames();
    btn.disabled = false;
    btn.textContent = label;
  }
}

function plainBox(pre) {
  pre.setAttribute('contenteditable', 'plaintext-only');
  if (pre.contentEditable !== 'plaintext-only') pre.setAttribute('contenteditable', 'true');
  pre.setAttribute('spellcheck', 'false');
  pre.setAttribute('autocapitalize', 'off');
  pre.setAttribute('autocorrect', 'off');
  pre.setAttribute('title', '可以直接在这里改代码,再点 Run');
}

function init() {
  document.querySelectorAll('.exercise .py-run').forEach((btn) => {
    if (checkCodeFor(btn.closest('.code-card'))) btn.textContent = '▶ 运行并检查';
  });
  if (window.PyEditor) setupEditors();
  else {
    document.querySelectorAll('.py-runnable pre').forEach(plainBox);
    restoreCode();
  }
  showProgress();
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.py-run');
    if (btn) runCard(btn.closest('.code-card'));
  });
}

if (document.readyState !== 'loading') init();
else document.addEventListener('DOMContentLoaded', init);
