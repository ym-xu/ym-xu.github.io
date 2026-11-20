"""Lint the Markdown of course lessons; fix what can be fixed safely.

    python tools/lint_lesson.py content/python/*/index.md content/stat-ml/*/index.md

- quiz options: quotes inside data-why="..." are rewritten as &quot; (they would end the attribute);
- pycheck and python blocks must be valid Python (starter blocks with ___ blanks and blocks that are
  meant to fail are reported, not counted as errors);
- no full-width , ; : ( ) ? ! (the site's Chinese text uses the half-width ones).
"""
import ast
import pathlib
import re
import sys

OPT = re.compile(r'^(<button class="quiz-opt"(?: data-correct)? data-why=")(.*?)(">[^<]*</button>)$')
bad = 0
for name in sys.argv[1:]:
    path = pathlib.Path(name)
    lines = path.read_text().split('\n')
    changed = False
    fence, start, buf = None, 0, []
    for i, line in enumerate(lines):
        m = OPT.match(line)
        if m and '"' in m.group(2):
            lines[i] = m.group(1) + m.group(2).replace('"', '&quot;') + m.group(3)
            changed = True
            print(f'{name}:{i + 1}: quotes in data-why rewritten as &quot;')
        elif line.startswith('<button class="quiz-opt"') and not m:
            print(f'{name}:{i + 1}: quiz option not in the expected form'); bad += 1
        if fence is None:
            fm = re.match(r'^```(\w+)?', line)
            if fm:
                fence, start, buf = fm.group(1) or 'text', i + 1, []
        elif line.startswith('```'):
            if fence in ('python', 'pycheck'):
                code = '\n'.join(buf)
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    if fence == 'pycheck':
                        print(f'{name}:{start + (e.lineno or 1)}: pycheck does not compile: {e.msg}'); bad += 1
                    elif '___' not in code:
                        print(f'{name}:{start + (e.lineno or 1)}: python block does not compile (meant to fail?): {e.msg}')
            fence = None
        else:
            buf.append(line)
        if fence is None or fence not in ('python', 'pycheck'):
            for ch in re.findall(r'[，；：（）？！]', line):
                print(f'{name}:{i + 1}: full-width {ch}'); bad += 1
    if changed:
        path.write_text('\n'.join(lines))
print(f'{bad} problem(s)')
sys.exit(1 if bad else 0)
