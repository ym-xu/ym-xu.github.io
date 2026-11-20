"""Run the code of a lesson with the local Python, as a quick check before the browser check.

    python tools/run_lesson.py content/python/02-pandas/index.md [--full]

It executes the lesson's python blocks in page order in one namespace (like the page does), prints
what each block prints (and the value of its last expression), and for every exercise runs the
folded reference answer followed by the hidden check. Starter blocks of exercises are skipped.
The working directory is static/python, so `from cohort import ...` and the course data files work.
This does not replace tools/check_course.py: package versions in the browser (Pyodide) may differ.
"""
import ast
import contextlib
import io
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def blocks(text):
    """Yield (kind, code, in_exercise, in_answer) for python and pycheck fences, in order."""
    in_ex = in_details = False
    fence = None
    buf = []
    for line in text.split('\n'):
        if fence is None:
            if re.match(r'<div class="[^"]*\bexercise\b', line):
                in_ex = True
            elif line.startswith('<details'):
                in_details = True
            elif line.startswith('</details>'):
                in_details = False
            elif line.startswith('</div>') and in_ex and not in_details:
                in_ex = False
            m = re.match(r'^```(python|pycheck)\s*$', line)
            if m:
                fence, buf = m.group(1), []
            elif line.startswith('```'):
                fence, buf = 'other', []
        elif line.startswith('```'):
            if fence != 'other':
                yield fence, '\n'.join(buf), in_ex, in_details
            fence = None
        else:
            buf.append(line)


def run(code, ns):
    """Execute like the page: print stdout, then the value of a final expression."""
    tree = ast.parse(code)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(tree, '<block>', 'exec'), ns)
        if last is not None:
            value = eval(compile(ast.Expression(last.value), '<block>', 'eval'), ns)
            if value is not None and not type(value).__module__.startswith('matplotlib') and not (
                    isinstance(value, (list, tuple)) and value and all(type(v).__module__.startswith('matplotlib') for v in value)):
                print(value.to_string() if hasattr(value, 'to_string') else repr(value))
    return out.getvalue().rstrip()


class Stop(Exception):
    pass


def check(code, ns):
    found = []

    def ok(msg='对了。'):
        found.append('✓ ' + msg); raise Stop

    def fail(msg):
        found.append('✗ ' + msg); raise Stop

    def need(*names):
        for name in names:
            if name not in ns:
                fail(f'还没有定义 {name}。')

    def close(a, b, tol=0.05):
        try:
            return abs(float(a) - float(b)) <= tol
        except (TypeError, ValueError):
            return False

    g = dict(ns); g.update(ok=ok, fail=fail, need=need, close=close)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, g)
    except Stop:
        pass
    except Exception as e:      # noqa: BLE001
        found.append(f'✗ 检查没能完成({type(e).__name__}: {e})')
    return found[0] if found else '✗ 检查没有得到结果'


def main():
    path = pathlib.Path(sys.argv[1]).resolve()
    full = '--full' in sys.argv
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.chdir(ROOT / 'static' / 'python')
    sys.path.insert(0, str(ROOT / 'static' / 'python'))
    ns = {'__name__': '__main__'}
    failed = False
    pending_check = None
    for i, (kind, code, in_ex, in_answer) in enumerate(blocks(path.read_text())):
        head = code.strip().split('\n')[0][:60]
        if kind == 'pycheck':
            pending_check = code
            continue
        if code.lstrip().startswith('#| norun'):
            print(f'[{i}] norun: {head}')
            continue
        if in_ex and not in_answer:
            print(f'[{i}] starter skipped: {head}')
            continue
        try:
            out = run(code, ns)
        except Exception as e:      # noqa: BLE001
            print(f'[{i}] RAISED {type(e).__name__}: {e}   | {head}')
            failed = failed or in_answer
            continue
        if in_answer:
            verdict = check(pending_check, ns) if pending_check else '(no check)'
            failed = failed or not verdict.startswith('✓')
            print(f'[{i}] answer -> {verdict}')
            pending_check = None
        else:
            lines = out.split('\n')
            shown = out if full or len(lines) <= 14 else '\n'.join(lines[:14]) + f'\n… ({len(lines)} lines)'
            print(f'[{i}] {head}\n{shown}' if out else f'[{i}] {head}  (no output)')
        if 'matplotlib.pyplot' in sys.modules:
            import matplotlib.pyplot as plt
            if plt.get_fignums():
                print(f'    ({len(plt.get_fignums())} figure)')
                plt.close('all')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
