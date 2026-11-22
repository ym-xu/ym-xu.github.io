"""Headless check of the fill-in exercises (starter code with ___ blanks) at phone width, with touch.

For every exercise whose starter has ___ blanks, on the given pages:
- each blank is shaded in the code box;
- running the starter as it is gives the note "先把代码里的 ___ 换成你的答案" and no verdict (only the
  statements before the first blank run);
- what goes into each blank is read off the folded reference answer; the reader's way is then
  followed: tap a blank (the whole blank must be selected), type what goes there, the next blank,
  ..., run; the box must then hold the reference answer and the check must say ✓.
Before an exercise, the page's own blocks and the reference answers of the exercises above it are
run, as a reader who did the earlier steps would have.

    hugo --minify -d /tmp/public
    python tools/check_blanks.py /tmp/public python/practice/02-clean stat-ml/04-cross-validation
    python tools/check_blanks.py LIVE python/practice/02-clean                  # the published site

Exit status 1 if any of this fails or the page logs a JavaScript error. Same serving as check_course.py
(request interception, no HTTP port).
"""
import mimetypes
import pathlib
import re
import sys
import time

from playwright.sync_api import sync_playwright

SITE = 'https://ym-xu.github.io/'
NOTE = '先把代码里的 ___ 换成你的答案,再点运行。'
root = None if sys.argv[1] == 'LIVE' else pathlib.Path(sys.argv[1])
pages = sys.argv[2:]
bad = []


def serve(route):
    path = route.request.url.split('#')[0].split('?')[0].replace(SITE, '')
    f = root / path
    if f.is_dir() or path.endswith('/') or path == '':
        f = f / 'index.html'
    if not f.exists():
        return route.fulfill(status=404, body='not found')
    ctype = 'text/plain; charset=utf-8' if f.suffix in ('.py', '.csv') else (mimetypes.guess_type(str(f))[0] or 'application/octet-stream')
    route.fulfill(status=200, body=f.read_bytes(), headers={'content-type': ctype})


def fills_for(starter, answer):
    """what goes into each blank, read off the reference answer; None if the two do not line up"""
    drop = lambda s: re.sub(r'\s+#.*$', '', s).rstrip()
    ans = [drop(a) for a in answer.split('\n')]
    out, j = [], 0
    for line in starter.split('\n'):
        core = drop(line)
        if not core.strip() or core.strip().startswith('#'):
            continue
        parts = core.split('___')
        pat = re.compile('^' + '(.+?)'.join(re.escape(p) for p in parts) + '$')
        while j < len(ans) and not pat.match(ans[j]):
            j += 1
        if j == len(ans):
            return None
        out += list(pat.match(ans[j]).groups())
        j += 1
    return out


def run(page, card):
    card.locator('.py-run').scroll_into_view_if_needed()
    card.locator('.py-run').tap()
    page.wait_for_function('(el) => !el.querySelector(".py-run").disabled && !el.querySelector(".code-output").hidden',
                           arg=card.element_handle(), timeout=600000)
    return card.locator('.code-output').inner_text()


with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
    if root is not None:
        ctx.route(SITE + '**', serve)
    for name in pages:
        page = ctx.new_page()
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(f'{SITE}{name.strip("/")}/?t={time.time()}', wait_until='networkidle')
        page.evaluate("document.documentElement.classList.add('steps-all')")     # themed practice: show every step
        page.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
        cards = page.locator('.py-runnable')
        kinds = page.evaluate("""[...document.querySelectorAll('.py-runnable')].map(el =>
            el.closest('details') ? 'answer' : (el.closest('.exercise') ? 'starter' : 'block'))""")
        print(f'== {name}: {kinds.count("starter")} exercises')
        t0, n_blank = time.time(), 0
        for i, kind in enumerate(kinds):
            card = cards.nth(i)
            if kind != 'starter':
                run(page, card)                     # the page's own blocks and the answers of the exercises done so far
                continue
            starter = card.evaluate('c => window.pyCodeOf(c)')
            if '___' not in starter:
                continue
            n_blank += 1
            ex = card.locator('xpath=ancestor::div[contains(@class, "exercise")][1]')
            title = ex.locator('strong').first.inner_text() if ex.locator('strong').count() else f'block {i}'
            answer = ex.locator('details .py-runnable').last.evaluate('c => window.pyCodeOf(c)')
            problems = []
            n = starter.count('___')
            if card.locator('.cm-blank').count() != n:
                problems.append(f'{card.locator(".cm-blank").count()} shaded marks for {n} blanks')
            out = run(page, card)
            if not out.strip().endswith(NOTE) or '✓' in out or '✗' in out:
                problems.append('running it unfilled: ' + out[-120:].replace('\n', ' / '))
            fills = fills_for(starter, answer)
            if fills is None or len(fills) != n:
                problems.append('the starter and the reference answer do not line up')
            else:
                for k, fill in enumerate(fills):
                    mark = card.locator('.cm-blank').first
                    mark.scroll_into_view_if_needed()
                    b = mark.bounding_box()
                    page.touchscreen.tap(b['x'] + b['width'] / 2, b['y'] + b['height'] / 2)
                    page.wait_for_timeout(80)
                    sel = page.evaluate('document.getSelection().toString()')
                    if sel != '___':
                        problems.append(f'a tap on blank {k + 1} selected {sel!r}')
                        break
                    page.keyboard.type(fill)
                    page.keyboard.press('Escape')
                else:
                    typed = card.evaluate('c => window.pyCodeOf(c)')
                    want = starter
                    for f in fills:
                        want = want.replace('___', f, 1)
                    if typed.strip() != want.strip():
                        problems.append('after typing, the box holds ' + repr(typed[:120]))
                    out = run(page, card)
                    if '✓' not in out:
                        problems.append('filled in, the check says: ' + out[-140:].replace('\n', ' / '))
            print(f'  {"ok  " if not problems else "FAIL"} {title[:40]}  ({n} blank{"s" if n > 1 else ""})' + ('' if not problems else '  | ' + ' ; '.join(problems)))
            if problems:
                bad.append(f'{name}: {title}')
        print(f'  {n_blank} fill-in exercises in {time.time() - t0:.0f} s; JavaScript errors: {errors}')
        if errors:
            bad.append(f'{name}: JavaScript errors')
        page.close()
    browser.close()
print('ALL OK' if not bad else 'FAILED: ' + ' | '.join(bad))
sys.exit(1 if bad else 0)
