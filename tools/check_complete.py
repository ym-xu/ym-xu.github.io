"""Completion and call hints in the code boxes (py-editor.js with opts.assist), on a computer and on a phone.

    hugo --minify -d /tmp/public
    python tools/check_complete.py /tmp/public

While the feature is on trial it is switched on with ?try=complete (kept in the browser) and off with ?try=off.
Checks: Tab completion of names, attributes, column names; the parameter hint and its bold parameter;
on a phone the list opens by itself, a tap takes a choice and Enter stays a new line; ?try=off turns it off.
Needs playwright with Chromium; Pyodide comes from the CDN. The site is served by request interception.
"""
import sys, pathlib, mimetypes
from playwright.sync_api import sync_playwright

SITE = 'https://ym-xu.github.io/'
root = pathlib.Path(sys.argv[1])
SHOTS = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path('/tmp')     # where the pictures go
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
POPUP = '.cm-tooltip-autocomplete.py-complete'


def options(pg):
    return pg.evaluate(f"[...document.querySelectorAll('{POPUP} li .cm-completionLabel')].map(e => e.textContent)")


def run_first_block(pg):
    first = pg.locator('.py-runnable').first
    first.locator('.py-run').scroll_into_view_if_needed(); first.locator('.py-run').click()
    pg.wait_for_function('(el) => !el.querySelector(".py-run").disabled && !el.querySelector(".code-output").hidden', arg=first.element_handle(), timeout=300000)
    return first.locator('.code-output').inner_text()


def fresh(pg, card, text=''):
    """empty the box and type text"""
    ed = card.locator('.cm-content'); ed.scroll_into_view_if_needed(); ed.click()
    pg.keyboard.press('Control+A'); pg.keyboard.press('Backspace')
    if text:
        pg.keyboard.type(text)


with sync_playwright() as p:
    browser = p.chromium.launch()
    errors = []

    # ---- a computer: Tab opens the list, Enter takes the choice ----
    print('== computer (no touch)')
    ctx = browser.new_context(viewport={'width': 1280, 'height': 900})
    ctx.route(SITE + '**', serve)
    pg = ctx.new_page(); pg.on('pageerror', lambda e: errors.append(str(e)))
    pg.goto(SITE + 'python/02-pandas/?try=complete', wait_until='networkidle')
    check('the trial switch is kept in this browser', pg.evaluate("localStorage.getItem('course-try:complete')") == '1')
    card = pg.locator('.exercise .py-runnable').first
    fresh(pg, card, 'pri'); pg.keyboard.press('Tab'); pg.wait_for_timeout(150)
    check('before Python runs: Tab after "pri" lists print', 'print' in options(pg), options(pg)[:5])
    pg.keyboard.press('Tab'); pg.wait_for_timeout(100)
    check('Tab again takes it', card.evaluate(CODE).strip() == 'print', repr(card.evaluate(CODE)))
    fresh(pg, card); pg.keyboard.press('Tab')
    check('Tab at the start of a line still indents', card.evaluate(CODE) == '    ', repr(card.evaluate(CODE)))
    out = run_first_block(pg)
    check('the first block runs (Python is up)', 'Traceback' not in out, out[:60])
    fresh(pg, card, 'df.grou'); pg.keyboard.press('Tab'); pg.wait_for_timeout(150)
    check('df.grou + Tab lists groupby', 'groupby' in options(pg), options(pg)[:5])
    pg.keyboard.press('Enter'); pg.wait_for_timeout(100)
    check('Enter takes the choice', card.evaluate(CODE).strip() == 'df.groupby', repr(card.evaluate(CODE)))
    fresh(pg, card, 'df["wall_me'); pg.keyboard.press('Tab'); pg.wait_for_timeout(150)
    check('df["wall_me + Tab lists the column wall_mean', options(pg) == ['wall_mean'], options(pg))
    pg.keyboard.press('Enter'); pg.wait_for_timeout(100)
    check('the column name goes in between the quotes', card.evaluate(CODE).strip() == 'df["wall_mean"]', repr(card.evaluate(CODE)))
    fresh(pg, card, 'pd.read_c'); pg.keyboard.press('Tab'); pg.wait_for_timeout(150)
    check('a module: pd.read_c lists read_csv', 'read_csv' in options(pg), options(pg)[:5])
    pg.keyboard.press('Escape')
    fresh(pg, card, 'df.groupby('); pg.wait_for_timeout(200)
    sig = pg.locator('.py-sig')
    check('typing "(" shows the parameters of groupby', sig.count() == 1 and 'groupby(' in sig.inner_text(), sig.inner_text() if sig.count() else '')
    first_bold = pg.locator('.py-sig b').inner_text() if pg.locator('.py-sig b').count() else ''
    pg.keyboard.type('"arm", '); pg.wait_for_timeout(150)
    second_bold = pg.locator('.py-sig b').inner_text() if pg.locator('.py-sig b').count() else ''
    check('the parameter being typed is in bold, and moves on after a comma', first_bold.startswith('by') and second_bold and second_bold != first_bold, (first_bold, second_bold))
    pg.keyboard.type('dropna='); pg.wait_for_timeout(150)
    check('a keyword argument puts that parameter in bold', pg.locator('.py-sig b').inner_text().startswith('dropna') if pg.locator('.py-sig b').count() else False)
    pg.keyboard.press('Escape'); pg.wait_for_timeout(100)
    check('Escape closes the hint', pg.locator('.py-sig').count() == 0)
    fresh(pg, card, 'print(len('); pg.wait_for_timeout(200)
    check('the hint follows the inner call', 'len(' in (pg.locator('.py-sig').inner_text() if pg.locator('.py-sig').count() else ''))
    pg.keyboard.type('df)'); pg.wait_for_timeout(150)
    check('closing the inner call closes its hint', pg.locator('.py-sig').count() == 0)
    print('  how the hint looks:', pg.evaluate("1"), end='')
    fresh(pg, card, 'df.groupby('); pg.wait_for_timeout(200); print(' |', pg.locator('.py-sig').inner_text().replace('\n', ' / ')[:150])
    pg.locator('.py-sig').screenshot(path=str(SHOTS / 'sig-computer.png'))
    fresh(pg, card, 'df.dr'); pg.keyboard.press('Tab'); pg.wait_for_timeout(150)
    pg.locator(POPUP).screenshot(path=str(SHOTS / 'list-computer.png')); pg.keyboard.press('Escape')
    ctx.close()

    # ---- a phone: the list opens by itself, a tap takes it, Enter stays a new line ----
    print('== phone (touch)')
    ctx = browser.new_context(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True, device_scale_factor=2)
    ctx.route(SITE + '**', serve)
    pg = ctx.new_page(); pg.on('pageerror', lambda e: errors.append(str(e)))
    pg.goto(SITE + 'python/02-pandas/?try=complete', wait_until='networkidle')
    out = run_first_block(pg)
    card = pg.locator('.exercise .py-runnable').first
    fresh(pg, card, 'df.'); pg.wait_for_timeout(250)
    check('after "df." the list of DataFrame methods opens by itself', len(options(pg)) > 10 and 'abs' in options(pg), options(pg)[:4])
    pg.keyboard.press('Enter'); pg.wait_for_timeout(100)
    check('Enter makes a new line instead of taking a choice', card.evaluate(CODE).startswith('df.\n'), repr(card.evaluate(CODE)))
    fresh(pg, card, 'df.grou'); pg.wait_for_timeout(250)
    opt = pg.locator(f'{POPUP} li').filter(has_text='groupby').first
    box = opt.bounding_box(); pg.touchscreen.tap(box['x'] + 20, box['y'] + box['height'] / 2); pg.wait_for_timeout(150)
    check('a tap on a choice takes it', card.evaluate(CODE).strip() == 'df.groupby', repr(card.evaluate(CODE)))
    fresh(pg, card, 'x = 1'); pg.wait_for_timeout(250)
    check('one letter does not open the list', len(options(pg)) == 0, options(pg)[:3])
    fresh(pg, card, 'me'); pg.wait_for_timeout(250)
    check('two letters do: names from this page (merged ...)', any(o.startswith('me') for o in options(pg)), options(pg)[:4])
    pg.screenshot(path=str(SHOTS / 'list-phone.png'))
    fresh(pg, card, 'df.groupby('); pg.wait_for_timeout(250)
    sig = pg.locator('.py-sig')
    if sig.count():
        b = sig.bounding_box()
        check('the hint fits the phone screen', b['x'] >= 0 and b['x'] + b['width'] <= 390, b)
        pg.screenshot(path=str(SHOTS / 'sig-phone.png'))
    else:
        check('the hint shows on the phone', False)
    check('page does not scroll sideways', pg.evaluate('document.documentElement.scrollWidth') <= 390, pg.evaluate('document.documentElement.scrollWidth'))
    ctx.close()

    # ---- switched off again ----
    print('== switched off with ?try=off')
    ctx = browser.new_context(viewport={'width': 1280, 'height': 900})
    ctx.route(SITE + '**', serve)
    pg = ctx.new_page(); pg.on('pageerror', lambda e: errors.append(str(e)))
    pg.goto(SITE + 'python/02-pandas/?try=complete', wait_until='networkidle')
    pg.goto(SITE + 'python/02-pandas/?try=off', wait_until='networkidle')
    card = pg.locator('.exercise .py-runnable').first
    fresh(pg, card, 'pri'); pg.keyboard.press('Tab'); pg.wait_for_timeout(150)
    check('no list, Tab inserts spaces as before', len(options(pg)) == 0 and card.evaluate(CODE) == 'pri ', repr(card.evaluate(CODE)))
    check('the switch is gone from this browser', pg.evaluate("localStorage.getItem('course-try:complete')") is None)
    ctx.close()
    print('  JavaScript errors:', errors or 'none')
    check('no JavaScript errors', not errors)
    browser.close()
print('ALL OK' if ok_all else 'SOME CHECKS FAILED')
sys.exit(0 if ok_all else 1)
