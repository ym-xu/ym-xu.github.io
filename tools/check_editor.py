"""The code boxes of the two courses as editors (assets/js/vendor/py-editor.js, CodeMirror), at phone width with touch.

    hugo --minify -d /tmp/public
    python tools/check_editor.py /tmp/public

Taps the empty part of an exercise box and types; checks indentation (Enter after a colon, Backspace,
Tab, Shift+Tab), straight quotes from curly ones, colours while typing, keeping and resetting the typed
code, Shift+Enter, the check of an exercise, Copy, steps shown later, folded answers, dark mode, the
height of the exercise boxes and long lines. Then blocks the editor file: the plain editable boxes of
before must still work, and code typed in them must come back in the editor.
Needs playwright with Chromium; Pyodide is fetched from the CDN, so it needs network access.
The site is served to the browser by request interception: no HTTP port is opened.
"""
import sys, json, pathlib, mimetypes
from playwright.sync_api import sync_playwright

SITE = 'https://ym-xu.github.io/'
root = pathlib.Path(sys.argv[1])
engine = 'chromium'
ok_all = True


def check(name, cond, detail=''):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(('  ok   ' if cond else '  FAIL ') + name + (f'  | {detail}' if detail != '' else ''))


def serve(route):
    path = route.request.url.split('#')[0].split('?')[0].replace(SITE, '')
    f = root / path
    if f.is_dir() or path.endswith('/') or path == '':
        f = f / 'index.html'
    if not f.exists():
        return route.fulfill(status=404, body='nf')
    ctype = 'text/plain; charset=utf-8' if f.suffix in ('.py', '.csv') else (mimetypes.guess_type(str(f))[0] or 'application/octet-stream')
    route.fulfill(status=200, body=f.read_bytes(), headers={'content-type': ctype})


CODE = 'c => window.pyCodeOf(c)'


def classes_of_text(card, text):
    """the class of the innermost span holding `text` in the editor"""
    return card.evaluate("""(c, t) => { for (const s of c.querySelectorAll('.cm-line span')) if (s.textContent === t) return s.className; return null; }""", text)


with sync_playwright() as p:
    browser = getattr(p, engine).launch()
    phone = dict(viewport={'width': 390, 'height': 844}, has_touch=True, device_scale_factor=3)
    if engine == 'chromium':
        phone['is_mobile'] = True
    ctx = browser.new_context(**phone)
    ctx.route(SITE + '**', serve)
    errors = []

    # ---- 1. the practice step: tap below the starter comment and type ----
    print(f'== {engine}: themed practice, part 5')
    pg = ctx.new_page(); pg.on('pageerror', lambda e: errors.append(str(e)))
    pg.goto(SITE + 'python/practice/05-intervals/', wait_until='networkidle')
    check('every Python box became an editor', pg.evaluate("[...document.querySelectorAll('.py-runnable')].every(c => c.querySelector('.cm-editor') && !c.querySelector('pre'))"))
    check('page does not scroll sideways', pg.evaluate('document.documentElement.scrollWidth') <= 390, pg.evaluate('document.documentElement.scrollWidth'))
    card = pg.locator('.step.exercise .py-runnable').first
    ed = card.locator('.cm-content'); ed.scroll_into_view_if_needed(); box = ed.bounding_box()
    check('the exercise box is tall enough for six lines', box['height'] >= 140, round(box['height']))
    check('the starter ends with an empty line', card.evaluate(CODE).endswith('\n'), repr(card.evaluate(CODE)))
    pg.touchscreen.tap(box['x'] + 200, box['y'] + box['height'] - 15)     # the empty lower part
    pg.wait_for_timeout(100)
    check('a tap in the empty part puts the caret in the box', pg.evaluate("document.activeElement.classList.contains('cm-content')"))
    pg.keyboard.type('p_d90 = 0.905')
    code = card.evaluate(CODE)
    check('what is typed lands on its own line, below the comment', code.split('\n')[1] == 'p_d90 = 0.905', repr(code))
    check('the number is coloured as a number while typing', classes_of_text(card, '0.905') == 'mi', classes_of_text(card, '0.905'))
    check('the comment keeps its colour, the new line is not grey', classes_of_text(card, 'p_d90') in (None, '', 'ͼ'), classes_of_text(card, 'p_d90'))

    # ---- 2. indentation ----
    pg.keyboard.press('Enter'); pg.keyboard.type('for x in [1, 2]:'); pg.keyboard.press('Enter'); pg.keyboard.type('print(x)')
    code = card.evaluate(CODE)
    check('Enter after a colon indents by four spaces', '\n    print(x)' in code, repr(code[-40:]))
    check('the closing bracket was added, not doubled', code.count('print(x)') == 1 and 'print(x))' not in code, repr(code[-20:]))
    pg.keyboard.press('Enter')
    code = card.evaluate(CODE)
    check('the next line keeps the indentation', code.endswith('\n    '), repr(code[-12:]))
    pg.keyboard.press('Backspace')
    check('one Backspace removes the whole indentation step', card.evaluate(CODE).endswith('print(x)\n'), repr(card.evaluate(CODE)[-12:]))
    pg.keyboard.press('Tab')
    check('Tab inserts four spaces and stays in the box', card.evaluate(CODE).endswith('\n    ') and pg.evaluate("document.activeElement.classList.contains('cm-content')"))
    pg.keyboard.press('Shift+Tab')
    check('Shift+Tab takes them away', card.evaluate(CODE).endswith('print(x)\n'), repr(card.evaluate(CODE)[-12:]))

    # ---- 3. quotes from a phone keyboard or a Chinese input method ----
    pg.keyboard.type('s = len(“abc”)')
    code = card.evaluate(CODE)
    check('curly quotes typed in code become straight ones', code.endswith('s = len("abc")'), repr(code[-20:]))
    pg.keyboard.press('Enter'); pg.keyboard.type('# 他说“好”')
    check('curly quotes inside a comment are left alone', card.evaluate(CODE).endswith('# 他说“好”'), repr(card.evaluate(CODE)[-8:]))
    check('len is coloured as a built-in', classes_of_text(card, 'len') == 'nb', classes_of_text(card, 'len'))

    # ---- 4. what was typed is kept, 重置 brings the starter back ----
    pg.wait_for_timeout(700)
    saved = pg.evaluate("JSON.parse(localStorage.getItem('course-code:v1:/python/practice/05-intervals/') || '{}')")
    check('the typed code is kept in this browser', '0' in saved and 'p_d90 = 0.905' in saved['0']['c'])
    check('the 重置 button is shown', card.locator('.code-reset').is_visible())
    typed = card.evaluate(CODE)
    pg.reload(wait_until='networkidle')
    card = pg.locator('.step.exercise .py-runnable').first
    check('after reopening the page the code is back', card.evaluate(CODE) == typed)
    card.locator('.code-reset').click(); pg.wait_for_timeout(700)
    check('重置 puts the starter back and forgets the typed code', card.evaluate(CODE).startswith('#') and 'p_d90' not in card.evaluate(CODE).split('\n', 1)[1] and not card.locator('.code-reset').is_visible())

    # ---- 5. run: Shift+Enter, the check, the copy button ----
    ctx.grant_permissions(['clipboard-read', 'clipboard-write'], origin=SITE.rstrip('/'))
    first = pg.locator('.py-runnable').first                         # the setup block of the page
    first.locator('.cm-content').click(); pg.keyboard.press('Shift+Enter')
    pg.wait_for_function('(el) => !el.querySelector(".py-run").disabled && el.querySelector(".code-output") && !el.querySelector(".code-output").hidden', arg=first.element_handle(), timeout=300000)
    out = first.locator('.code-output').inner_text()
    check('Shift+Enter runs the block', '准备好了' in out, out[:60])
    check('Shift+Enter did not add a line', first.evaluate(CODE).count('\n') == first.evaluate(CODE).rstrip('\n').count('\n') or not first.evaluate(CODE).endswith('\n\n'))
    answer = card.evaluate("c => window.pyCodeOf(c.closest('.exercise').querySelector('details .py-runnable'))")
    card.locator('.cm-content').click(); pg.keyboard.press('Control+A'); pg.keyboard.insert_text(answer)
    card.locator('.py-run').click()
    pg.wait_for_function('(el) => !el.querySelector(".py-run").disabled', arg=card.element_handle(), timeout=300000)
    check('the reference answer typed into the box passes the check', '✓' in card.locator('.code-output').inner_text(), card.locator('.code-output').inner_text()[:50])
    card.locator('.code-copy').click(); pg.wait_for_timeout(300)
    if engine == 'chromium':
        check('Copy copies what is in the box now', pg.evaluate('navigator.clipboard.readText()') == card.evaluate(CODE))

    # ---- 6. the next steps (hidden at first) and the folded answers ----
    pg.locator('.steps-next').first.click(); pg.wait_for_timeout(300)
    card2 = pg.locator('.step.exercise:not([hidden]) .py-runnable').first
    lh = card2.locator('.cm-line').first.bounding_box()
    check('a step shown later draws its editor normally', lh and lh['height'] > 15, lh)
    det = pg.locator('.step.exercise:not([hidden]) details').last
    det.locator('summary').click(); pg.wait_for_timeout(200)
    a = det.locator('.cm-line').first.bounding_box()
    check('the folded answer opens as a normal editor', a and a['height'] > 15)

    # ---- 7. dark mode ----
    pg.evaluate("document.documentElement.setAttribute('data-theme', 'dark')")
    col = first.evaluate("c => { const s = c.querySelector('.cm-content .k, .cm-content .kn'); return s ? getComputedStyle(s).color : null; }")
    caret = card.evaluate("c => getComputedStyle(c.querySelector('.cm-content')).caretColor")
    check('dark mode: keywords take the dark colour', col in ('rgb(255, 123, 114)',), col)
    check('dark mode: the caret is light, not black', caret not in ('rgb(0, 0, 0)',), caret)

    # ---- 8. a lesson: boxes, long lines, page width ----
    print(f'== {engine}: lesson 3 of the Python course')
    pg2 = ctx.new_page(); pg2.on('pageerror', lambda e: errors.append(str(e)))
    pg2.goto(SITE + 'python/03-table-one/', wait_until='networkidle')
    check('page does not scroll sideways', pg2.evaluate('document.documentElement.scrollWidth') <= 390, pg2.evaluate('document.documentElement.scrollWidth'))
    hs = pg2.evaluate("[...document.querySelectorAll('.exercise .py-runnable')].filter(c => !c.closest('details')).map(c => Math.round(c.querySelector('.cm-content').getBoundingClientRect().height))")
    check('every exercise box in the lesson is at least six lines tall', min(hs) >= 140, hs)
    ex = pg2.evaluate("[...document.querySelectorAll('.py-runnable')].filter(c => !c.closest('.exercise')).map(c => Math.round(c.querySelector('.cm-content').getBoundingClientRect().height))")
    check('example boxes keep their own height (no empty line at the end)', min(ex) < 60, ex[:6])
    wide = pg2.evaluate("[...document.querySelectorAll('.cm-scroller')].some(s => s.scrollWidth > s.clientWidth + 2)")
    check('long lines scroll inside the box', wide)
    print('  JavaScript errors:', errors or 'none')
    check('no JavaScript errors', not errors)
    browser.close()

# ---- the editor file does not load ----
block = {'on': True}
def maybe_block(route):
    if block['on'] and '/js/vendor/py-editor' in route.request.url: return route.abort()
    return serve(route)
with sync_playwright() as p:
    b = p.chromium.launch(); ctx = b.new_context(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True, device_scale_factor=3)
    ctx.route(SITE + '**', maybe_block); errors = []
    pg = ctx.new_page(); pg.on('pageerror', lambda e: errors.append(str(e)))
    url = SITE + 'python/01-first-steps/'
    pg.goto(url, wait_until='networkidle')
    print('== editor file blocked: the plain boxes of before')
    check('no editor, the boxes are editable <pre>', pg.evaluate("document.querySelectorAll('.cm-editor').length === 0 && [...document.querySelectorAll('.py-runnable pre')].every(p => p.isContentEditable)"))
    ex = pg.locator('.exercise').first; card = ex.locator('.py-runnable').first
    answer = ex.locator('details .py-runnable').last.evaluate('c => window.pyCodeOf(c)')
    pre = card.locator('pre'); pre.scroll_into_view_if_needed(); pre.click(); pg.keyboard.press('Control+A'); pg.keyboard.insert_text(answer)
    card.locator('.py-run').click(); pg.wait_for_function('(el) => !el.querySelector(".py-run").disabled', arg=card.element_handle(), timeout=300000)
    check('an exercise still runs and passes with the plain box', '✓' in card.locator('.code-output').inner_text(), card.locator('.code-output').inner_text()[:40])
    ctx.grant_permissions(['clipboard-read', 'clipboard-write'], origin=SITE.rstrip('/'))
    card.locator('.code-copy').click(); pg.wait_for_timeout(300)
    check('Copy works with the plain box', pg.evaluate('navigator.clipboard.readText()').strip() == answer.strip())
    pg.wait_for_timeout(700)
    print('== the editor loads again: what was typed with the plain box comes back')
    block['on'] = False
    pg.reload(wait_until='networkidle')
    card = pg.locator('.exercise').first.locator('.py-runnable').first
    check('the box is an editor now', card.locator('.cm-editor').count() == 1)
    check('the code typed in the plain box is in the editor', card.evaluate('c => window.pyCodeOf(c)').strip() == answer.strip(), repr(card.evaluate('c => window.pyCodeOf(c)')[:40]))
    check('the exercise still counts as passed', 'passed' in (pg.locator('.exercise').first.get_attribute('class') or ''))
    print('  JavaScript errors:', errors or 'none'); check('no JavaScript errors', not errors)
    b.close()
print('ALL OK' if ok_all else 'SOME CHECKS FAILED')
sys.exit(0 if ok_all else 1)
